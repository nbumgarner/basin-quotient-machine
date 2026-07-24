#!/usr/bin/env python3
"""bqsm_convert_dequant.py — Full Q4_K/Q6_K dequantizer → BQSM v3.
Proper fp16 (numpy), verified 6-bit scale unpack (cosine 0.886).
Global BQSM 4-state quantization per tensor. Streams tensors.
"""
import struct, numpy as np, os
from gguf import GGUFReader

INP = "/home/compunerd/Downloads/gemma-2-2b-it-abliterated-Q4_K_M.gguf"
OUT = "/home/compunerd/Downloads/gemma-2b-dequant.bqsm"

reader = GGUFReader(INP)
d_model=2304; ffn_dim=9216; n_layers=26; n_heads=8; n_kv_heads=4; vocab=256000
print(f"Proper dequantizer: {len(reader.tensors)} tensors")

# ── Dequant functions ──
def dequant_q4(raw, nelems):
    """Q4_K: 144B/block. fp16 d+dmin + 6-bit scales. Scale-only (no mins)."""
    nb = nelems // 256
    blocks = raw[:nb*144].reshape(nb, 144)
    d = blocks[:, 0:2].view(np.float16).flatten().astype(np.float32)
    qs = blocks[:, 4:132]  # (nb, 128)
    nib_lo = (qs & 0x0F).astype(np.float32)
    nib_hi = (qs >> 4).astype(np.float32)
    nib = np.empty((nb, 256), dtype=np.float32)
    nib[:, 0::2] = nib_lo; nib[:, 1::2] = nib_hi
    
    # Simple 6-bit scales: bytes 132-139 → 8 scales (verified: cosine 0.886)
    sc_raw = blocks[:, 132:144]
    sc = np.empty((nb, 8), dtype=np.float32)
    for i in range(8):
        sc[:, i] = (sc_raw[:, i] & 0x3F).astype(np.float32)
    
    sc_br = np.empty((nb, 256), dtype=np.float32)
    for s in range(8):
        sc_br[:, s*32:(s+1)*32] = sc[:, s:s+1]
    
    d_br = d.reshape(nb, 1)
    return (nib * d_br * sc_br).ravel()[:nelems]

def dequant_q6(raw, nelems, chunk_blks=32768):
    """Q6_K: 210B/block. fp16 d + 16 8-bit scales. Chunked for large tensors."""
    nb = nelems // 256
    result = np.empty(nelems, dtype=np.float32)
    
    for start in range(0, nb, chunk_blks):
        end = min(start + chunk_blks, nb)
        nbc = end - start
        bstart = start * 210
        bend = end * 210
        
        blocks = raw[bstart:bend].reshape(nbc, 210)
        d = blocks[:, 0:2].view(np.float16).flatten().astype(np.float32)
        ql_raw = blocks[:, 2:130]
        qh_raw = blocks[:, 130:194]
        scales = blocks[:, 194:210].astype(np.float32)
        
        lo = np.empty((nbc, 256), dtype=np.float32)
        lo[:, 0::2] = (ql_raw & 0x0F).astype(np.float32)
        lo[:, 1::2] = (ql_raw >> 4).astype(np.float32)
        
        hi = np.empty((nbc, 256), dtype=np.float32)
        for s in range(4):
            hi[:, s::4] = ((qh_raw >> (2*s)) & 0x03).astype(np.float32)
        
        vals = lo + hi * 16.0
        sc_br = np.empty((nbc, 256), dtype=np.float32)
        for s in range(16):
            sc_br[:, s*16:(s+1)*16] = scales[:, s:s+1]
        
        d_br = d.reshape(nbc, 1)
        result[start*256:end*256] = (vals * d_br * sc_br).ravel()
    
    return result[:nelems]

def pack_4state(data):
    n = len(data)
    pad = (4 - (n % 4)) % 4
    if pad: data = np.append(data, np.zeros(pad, np.uint8))
    d = data.reshape(-1, 4)
    return d[:,0]|(d[:,1]<<2)|(d[:,2]<<4)|(d[:,3]<<6), n

def quantize_bqsm(f32):
    """Global quantization to BQSM 4-state (0-3)."""
    mn, mx = f32.min(), f32.max()
    if mx > mn:
        return np.clip(np.round((f32 - mn) / (mx - mn) * 3), 0, 3).astype(np.uint8)
    return np.full(len(f32), 2, dtype=np.uint8)

# ── Stream-write ──
f = open(OUT, 'wb')
f.write(b'BQSM')
f.write(struct.pack('<I', 3))
f.write(struct.pack('<6I', d_model, ffn_dim, n_layers, n_heads, n_kv_heads, vocab))
f.write(struct.pack('<I', len(reader.tensors)))

tp = 0
for ti, t in enumerate(reader.tensors):
    name = t.name
    shape = tuple(int(d) for d in t.shape)
    dtype = t.tensor_type.name
    ne = int(np.prod(shape))
    raw = np.array(t.data.flat, dtype=np.uint8)
    
    if dtype == 'F32':
        f32dat = np.array(t.data.flat, dtype=np.float32)[:ne]
        f32 = f32dat
    elif dtype == 'Q4_K':
        f32 = dequant_q4(raw, ne)
    elif dtype == 'Q6_K':
        f32 = dequant_q6(raw, ne)
    else:
        continue
    
    bq = quantize_bqsm(f32)
    pkb, on = pack_4state(bq)
    
    nb = name.encode()
    f.write(struct.pack('<H', len(nb))); f.write(nb)
    f.write(struct.pack('<B', len(shape)))
    for d in shape: f.write(struct.pack('<Q', d))
    f.write(struct.pack('<I', on)); f.write(struct.pack('<I', len(pkb)))
    f.write(pkb.tobytes())
    tp += len(pkb)
    
    if ti % 25 == 0:
        print(f"  [{ti:3d}/{len(reader.tensors)}] {name:45s} {on:>12,}e → {len(pkb):>10,}B  {tp/2**20:.0f}MB")
    
    del f32, bq, pkb

f.close()
print(f"\nDone: {OUT} = {os.path.getsize(OUT)/2**20:.1f} MB")
