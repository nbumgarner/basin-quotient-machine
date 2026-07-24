#!/usr/bin/env python3
"""bqsm_convert_real.py — GGUF → BQSM v3 with PROPER fp16 dequantization.
Chunked processing: no OOM. Q4_K + Q6_K full dequant to float32 then
global quantization to BQSM 4-state. Real Gemma 2B weights.
"""
import struct, numpy as np, os
from gguf import GGUFReader

INP = "/home/compunerd/Downloads/gemma-2-2b-it-abliterated-Q4_K_M.gguf"
OUT = "/home/compunerd/Downloads/gemma-2b-real.bqsm"

reader = GGUFReader(INP)
n_tensors = len(reader.tensors)

# Architecture
d_model=2304; ffn_dim=9216; n_layers=26; n_heads=8; n_kv_heads=4; vocab=256000
print(f"Gemma 2B: {n_tensors} tensors — proper fp16 dequantization")

# ── fp16 → float32 ──
def fp16_to_f32(h):
    """Vectorized fp16 (uint16) → float32."""
    h = np.asarray(h, dtype=np.uint16)
    s = ((h >> 15) & 1).astype(np.float32)
    e = ((h >> 10) & 0x1F).astype(np.float32)
    m = (h & 0x3FF).astype(np.float32)
    # Normal: (-1)^s * 2^(e-15) * (1 + m/1024)
    f = (1 - 2*s) * np.exp2(e - 15) * (1 + m / 1024)
    # Subnormal: e==0
    sub = (1 - 2*s) * np.exp2(-14) * (m / 1024)
    return np.where(e == 0, sub, f)

def dequant_q4(raw, nelems):
    """Full Q4_K dequantization: fp16 d/dmin + 6-bit scales."""
    nb = nelems // 256
    blocks = np.frombuffer(raw[:nb*144].tobytes(), dtype=np.uint8).reshape(nb, 144)
    
    d = fp16_to_f32(blocks[:, 0:2].view(np.uint16).reshape(nb))
    dmin = fp16_to_f32(blocks[:, 2:4].view(np.uint16).reshape(nb))
    
    # Q nibbles
    q = blocks[:, 4:132]  # (nb, 128)
    lo = (q & 0x0F).astype(np.float32)
    hi = (q >> 4).astype(np.float32)
    nib = np.empty((nb, 256), dtype=np.float32)
    nib[:, 0::2] = lo; nib[:, 1::2] = hi
    
    # 6-bit scales from bytes 132-143: 8 scales × 6 bits = 48 bits
    sc_raw = blocks[:, 132:144]  # (nb, 12)
    sc = np.empty((nb, 8), dtype=np.float32)
    sc[:,0] = sc_raw[:,0] & 0x3F;  sc[:,1] = sc_raw[:,1] & 0x3F
    sc[:,2] = sc_raw[:,2] & 0x3F;  sc[:,3] = sc_raw[:,3] & 0x3F
    sc[:,4] = sc_raw[:,4] & 0x3F;  sc[:,5] = sc_raw[:,5] & 0x3F
    sc[:,6] = sc_raw[:,6] & 0x3F;  sc[:,7] = sc_raw[:,7] & 0x3F
    
    # Broadcast scales: each scale covers 32 elements
    # dequant: nibble * d * sc[idx] / 16 - dmin
    d_br = d.reshape(nb, 1)
    dm_br = dmin.reshape(nb, 1)
    sc_br = np.empty((nb, 256), dtype=np.float32)
    for s in range(8):
        sc_br[:, s*32:(s+1)*32] = sc[:, s:s+1]
    
    return (nib * d_br * sc_br / 16.0 - dm_br).ravel()[:nelems]

def dequant_q6_chunked(raw, nelems, chunk_size=65536):
    """Q6_K dequantization — chunked for large tensors."""
    nb = nelems // 256
    result = np.empty(nelems, dtype=np.float32)
    
    for start in range(0, nb, chunk_size):
        end = min(start + chunk_size, nb)
        nbc = end - start
        bstart = start * 210
        bend = end * 210
        
        blocks = np.frombuffer(raw[bstart:bend], dtype=np.uint8).reshape(nbc, 210)
        d = fp16_to_f32(blocks[:, 0:2].view(np.uint16).reshape(nbc))
        
        ql = blocks[:, 2:130]  # (nbc, 128)
        qh = blocks[:, 130:194]  # (nbc, 64)
        scales = blocks[:, 194:210].astype(np.float32)  # (nbc, 16)
        
        lo = np.empty((nbc, 256), dtype=np.float32)
        lo[:, 0::2] = (ql & 0x0F).astype(np.float32)
        lo[:, 1::2] = (ql >> 4).astype(np.float32)
        hi = np.empty((nbc, 256), dtype=np.float32)
        for s in range(4):
            hi[:, s::4] = ((qh >> (2*s)) & 0x03).astype(np.float32)
        
        vals = lo + hi * 16.0  # 6-bit 0-63
        sc_br = np.empty((nbc, 256), dtype=np.float32)
        for s in range(16):
            sc_br[:, s*16:(s+1)*16] = scales[:, s:s+1]
        
        d_br = d.reshape(nbc, 1)
        chunk_result = vals * d_br * sc_br / 64.0
        result[start*256:end*256] = chunk_result.ravel()
    
    return result

def pack_4state(data):
    n = len(data)
    pad = (4 - (n % 4)) % 4
    if pad: data = np.append(data, np.zeros(pad, np.uint8))
    d = data.reshape(-1, 4)
    return d[:,0]|(d[:,1]<<2)|(d[:,2]<<4)|(d[:,3]<<6), n

def quantize_bqsm(f32):
    mn, mx = f32.min(), f32.max()
    if mx > mn:
        return np.clip(np.round((f32 - mn) / (mx - mn) * 3), 0, 3).astype(np.uint8)
    return np.full(len(f32), 2, dtype=np.uint8)

# ── Write ──
f = open(OUT, 'wb')
f.write(b'BQSM')
f.write(struct.pack('<I', 3))
f.write(struct.pack('<6I', d_model, ffn_dim, n_layers, n_heads, n_kv_heads, vocab))
f.write(struct.pack('<I', n_tensors))

tp = 0; tr = 0
for ti, t in enumerate(reader.tensors):
    name = t.name
    shape = tuple(int(d) for d in t.shape)
    dtype = t.tensor_type.name
    ne = int(np.prod(shape))
    
    if dtype == 'F32':
        raw = np.array(t.data.flat, dtype=np.float32)
        f32 = raw[:ne]
    elif dtype == 'Q4_K':
        raw = t.data.flat
        f32 = dequant_q4(raw, ne)
    elif dtype == 'Q6_K':
        raw = t.data.flat
        f32 = dequant_q6_chunked(raw, ne, chunk_size=65536)
    else:
        continue
    
    bq = quantize_bqsm(f32)
    pkb, on = pack_4state(bq)
    
    nb = name.encode()
    f.write(struct.pack('<H', len(nb)))
    f.write(nb)
    f.write(struct.pack('<B', len(shape)))
    for d in shape: f.write(struct.pack('<Q', d))
    f.write(struct.pack('<I', on))
    f.write(struct.pack('<I', len(pkb)))
    f.write(pkb.tobytes())
    tp += len(pkb); tr += on
    
    if ti % 25 == 0:
        print(f"  [{ti:3d}/{n_tensors}] {name:45s} {on:>12,}e → {len(pkb):>10,}B  {tp/2**20:.0f}MB total")
    
    # Free memory
    del f32, bq, pkb

f.close()
print(f"\nDone: {OUT} = {os.path.getsize(OUT)/2**20:.1f} MB  ({tr/1e6:.0f}M elements)")
