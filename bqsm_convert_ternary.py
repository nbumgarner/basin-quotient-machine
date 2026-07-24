#!/usr/bin/env python3
"""bqsm_convert_ternary.py — Proper TWN ternarization from dequantized floats.
Dequantizes Q4_K/Q6_K from GGUF using correct ggml formula:
  Q4_K: y = d * sc * (nibble - m) - dmin
  Q6_K: y = d * (lo + hi*16 - 32)
Then TWN ternarizes: t = 0.7 * mean(|w|), q = sign(w) if |w| > t else 0.
Packs as 2-bit {0:-1, 1:0, 2:+1} in BQSM v3 format.
Stores per-tensor alpha in the header (replaces ffn_dim slot, repurposed)."""
import struct, numpy as np, os
from gguf import GGUFReader

INP = "/home/compunerd/Downloads/gemma-2-2b-it-abliterated-Q4_K_M.gguf"
OUT = "/home/compunerd/Downloads/gemma-2b-ternary.bqsm"

reader = GGUFReader(INP)
d_model=2304; ffn_dim=9216; n_layers=26; n_heads=8; n_kv_heads=4; vocab=256000
print(f"TWN ternarization: {len(reader.tensors)} tensors")

def dequant_q4(raw, nelems):
    """Proper ggml Q4_K: y = d * sc * (nibble - m) - dmin"""
    nb = nelems // 256
    blocks = raw[:nb*144].reshape(nb, 144)
    
    d   = blocks[:, 0:2].view(np.float16).flatten().astype(np.float32)
    dmin = blocks[:, 2:4].view(np.float16).flatten().astype(np.float32)
    
    qs = blocks[:, 4:132]
    lo = (qs & 0x0F).astype(np.float32)
    hi = (qs >> 4).astype(np.float32)
    nibbles = np.empty((nb, 256), dtype=np.float32)
    nibbles[:, 0::2] = lo; nibbles[:, 1::2] = hi
    
    # ggml Q4_K scale unpacking
    sc_raw = blocks[:, 132:144].astype(np.int32)
    sc = np.empty((nb, 8), dtype=np.float32)
    m  = np.empty((nb, 8), dtype=np.float32)
    sc[:,0]=(sc_raw[:,0]&0x3F); sc[:,1]=(sc_raw[:,1]&0x3F)
    sc[:,2]=(sc_raw[:,2]&0x3F); sc[:,3]=(sc_raw[:,3]&0x3F)
    sc[:,4]=(sc_raw[:,4]&0x3F); sc[:,5]=(sc_raw[:,5]&0x3F)
    sc[:,6]=((sc_raw[:,6]&0x3F)|((sc_raw[:,9]&0x30)<<2))
    sc[:,7]=((sc_raw[:,7]&0x3F)|((sc_raw[:,9]&0xC0)))
    m[:,0]=(sc_raw[:,8]&0x3F);  m[:,1]=(sc_raw[:,9]&0x3F)
    m[:,2]=(sc_raw[:,10]&0x3F); m[:,3]=(sc_raw[:,11]&0x3F)
    m[:,4]=((sc_raw[:,11]>>6)|((sc_raw[:,10]&0x30)))
    m[:,5]=((sc_raw[:,10]>>6)|((sc_raw[:,9]&0x0C)<<2))
    m[:,6]=(sc_raw[:,9]>>6);     m[:,7]=(sc_raw[:,8]>>6)
    sc = sc.astype(np.float32); m = m.astype(np.float32)
    
    sc_br = np.empty((nb, 256), dtype=np.float32)
    m_br  = np.empty((nb, 256), dtype=np.float32)
    for s in range(8):
        sc_br[:, s*32:(s+1)*32] = sc[:, s:s+1]
        m_br[:,  s*32:(s+1)*32] = m[:,  s:s+1]
    
    d_br = d.reshape(nb, 1); dm_br = dmin.reshape(nb, 1)
    return (d_br * sc_br * (nibbles - m_br) - dm_br).ravel()[:nelems]

def dequant_q6(raw, nelems, chunk_blks=32768):
    """Proper ggml Q6_K: y = d * sc[s] * (qval - 32), qval = lo + hi*16"""
    nb = nelems // 256
    result = np.empty(nelems, dtype=np.float32)
    
    for start in range(0, nb, chunk_blks):
        end = min(start + chunk_blks, nb)
        nbc = end - start
        bstart = start * 210; bend = end * 210
        blocks = raw[bstart:bend].reshape(nbc, 210)
        
        d = blocks[:, 0:2].view(np.float16).flatten().astype(np.float32)
        # NaN safety
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        ql = blocks[:, 2:130]; qh = blocks[:, 130:194]
        scales = blocks[:, 194:210].astype(np.float32)  # (nbc, 16) int8 scales
        
        lo = np.empty((nbc, 256), dtype=np.float32)
        lo[:, 0::2] = (ql & 0x0F).astype(np.float32)
        lo[:, 1::2] = (ql >> 4).astype(np.float32)
        hi = np.empty((nbc, 256), dtype=np.float32)
        for s in range(4):
            hi[:, s::4] = ((qh >> (2*s)) & 0x03).astype(np.float32)
        
        qval = lo + hi * 16.0  # 0-63
        
        # Broadcast 16 scales to 256 elements (each scale covers 16 elements)
        sc_br = np.empty((nbc, 256), dtype=np.float32)
        for s in range(16):
            sc_br[:, s*16:(s+1)*16] = scales[:, s:s+1]
        
        d_br = d.reshape(nbc, 1)
        result[start*256:end*256] = (d_br * sc_br * (qval - 32.0)).ravel()
    
    return result

def ternarize_twn(w):
    """Ternary Weight Networks: threshold at 0.7 * mean(|w|)"""
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    t = 0.7 * np.mean(np.abs(w))
    if t <= 0 or np.isnan(t):
        # All weights near zero — no structure to ternarize
        return np.zeros(len(w), dtype=np.int8), 0.0
    q = np.where(w > t, 1, np.where(w < -t, -1, 0)).astype(np.int8)
    nz = max(np.count_nonzero(q), 1)
    alpha = float(np.sum(w.astype(np.float64) * q) / nz)
    return q, alpha

def pack_ternary(q_ternary):
    """Pack ternary weights {-1,0,1} into 2-bit {0:-1, 1:0, 2:+1, 3:unused}"""
    bqsm = np.where(q_ternary == -1, 0, np.where(q_ternary == 1, 2, 1)).astype(np.uint8)
    n = len(bqsm)
    pad = (4 - (n % 4)) % 4
    if pad: bqsm = np.append(bqsm, np.ones(pad, np.uint8))  # pad with 1 (0-weight)
    d = bqsm.reshape(-1, 4)
    packed = d[:,0]|(d[:,1]<<2)|(d[:,2]<<4)|(d[:,3]<<6)
    return packed, n

# Write file
f = open(OUT, 'wb')
f.write(b'BQSM')
f.write(struct.pack('<I', 3))
# Repurpose ffn_dim slot for alpha_count placeholder; actual alphas stored per-tensor
f.write(struct.pack('<6I', d_model, ffn_dim, n_layers, n_heads, n_kv_heads, vocab))
f.write(struct.pack('<I', len(reader.tensors)))

tp = 0
alphas = []  # store for diagnostics

for ti, t in enumerate(reader.tensors):
    name = t.name
    shape = tuple(int(d) for d in t.shape)
    dtype = t.tensor_type.name
    ne = int(np.prod(shape))
    raw = np.array(t.data.flat, dtype=np.uint8)
    
    if dtype == 'F32':
        w = np.array(t.data.flat, dtype=np.float32)[:ne]
    elif dtype == 'Q4_K':
        w = dequant_q4(raw, ne)
    elif dtype == 'Q6_K':
        w = dequant_q6(raw, ne)
    else:
        continue
    
    q, alpha = ternarize_twn(w)
    alphas.append((name, alpha))
    
    # Store norm tensors as raw float32 (needed for signed RMS norm with Gemma's 1+w formula)
    is_norm = ('norm' in name) and (dtype == 'F32')
    
    if is_norm:
        pkb = w.astype(np.float32).tobytes()
        on = ne
        tensor_type = 1  # F32
    else:
        pkb, on = pack_ternary(q)
        tensor_type = 0  # packed ternary
    
    nb = name.encode()
    f.write(struct.pack('<H', len(nb))); f.write(nb)
    f.write(struct.pack('<B', len(shape)))
    for d in shape: f.write(struct.pack('<Q', d))
    f.write(struct.pack('<I', on))
    f.write(struct.pack('<B', tensor_type))  # 0=packed ternary, 1=raw float32
    f.write(struct.pack('<I', len(pkb)))
    f.write(pkb)
    tp += len(pkb)
    
    if ti % 25 == 0:
        nz = (q != 0).sum()
        print(f"  [{ti:3d}/{len(reader.tensors)}] {name:45s} {ne:>12,}e → {on:>10,}B  "
              f"alpha={alpha:.4f}  nz={nz/ne*100:.0f}%  {tp/2**20:.0f}MB")

f.close()
print(f"\nDone: {OUT} = {os.path.getsize(OUT)/2**20:.1f} MB")

# Alpha stats
anonzero = [a for _,a in alphas if abs(a) > 0.0001]
print(f"Alphas: {len(anonzero)}/{len(alphas)} non-zero, range [{min(anonzero):.4f}, {max(anonzero):.4f}], mean={np.mean(anonzero):.4f}")
