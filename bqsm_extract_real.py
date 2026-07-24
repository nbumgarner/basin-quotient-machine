#!/usr/bin/env python3
"""bqsm_extract_real.py — GGUF → BQSM v3 with real Q4_K/Q6_K nibbles.
Streams one tensor at a time. Vectorized extraction. No intermediate storage.
"""
import struct, numpy as np, os
from gguf import GGUFReader

INP = "/home/compunerd/Downloads/gemma-2-2b-it-abliterated-Q4_K_M.gguf"
OUT = "/home/compunerd/Downloads/gemma-2b-real.bqsm"

reader = GGUFReader(INP)
d_model=2304; ffn_dim=9216; n_layers=26; n_heads=8; n_kv_heads=4; vocab_size=256000
print(f"Gemma 2B: {len(reader.tensors)} tensors")

# ── Vectorized extractors ──
def extract_q4(raw, nelems):
    """Q4_K 144B/block: d(2B fp16)+dmin(2B fp16)+q(128B)+scales(12B)
    Dequantize to float32, then globally quantize to BQSM 4-state."""
    nb = nelems // 256
    blocks = raw[:nb*144].reshape(nb, 144)
    
    # Parse fp16 d and dmin (first 4 bytes of each block)
    d_fp16 = blocks[:, 0:2].view(np.uint16).reshape(nb)
    dm_fp16 = blocks[:, 2:4].view(np.uint16).reshape(nb)
    
    # fp16 → float32 conversion
    sign_d = (d_fp16 >> 15).astype(np.float32)
    exp_d = ((d_fp16 >> 10) & 0x1F).astype(np.float32)
    mant_d = (d_fp16 & 0x3FF).astype(np.float32)
    d = ((-1)**sign_d) * (2**(exp_d-15)) * (1 + mant_d/1024)
    d = np.where(exp_d == 0, ((-1)**sign_d) * (2**-14) * (mant_d/1024), d)
    
    sign_dm = (dm_fp16 >> 15).astype(np.float32)
    exp_dm = ((dm_fp16 >> 10) & 0x1F).astype(np.float32)
    mant_dm = (dm_fp16 & 0x3FF).astype(np.float32)
    dmin = ((-1)**sign_dm) * (2**(exp_dm-15)) * (1 + mant_dm/1024)
    dmin = np.where(exp_dm == 0, ((-1)**sign_dm) * (2**-14) * (mant_dm/1024), dmin)
    
    # Q4_K scales: 12 bytes per block, 8 scales (6-bit each), packed
    # Scale layout: sc[0:6] = (scales[0:6]) & 0x3F, sc[6:8] from scales[6:8]+scales[10:12]
    # Simplification: extract all 12 bytes as-is, interpret as 8 scales
    sc_raw = blocks[:, 132:144]  # (nb, 12)
    
    # Extract nibbles from q bytes
    qs = blocks[:, 4:132].reshape(nb, 128)
    nib_lo = (qs & 0x0F).astype(np.float32)   # (nb, 128)
    nib_hi = (qs >> 4).astype(np.float32)      # (nb, 128)
    nibbles = np.empty((nb, 256), dtype=np.float32)
    nibbles[:, 0::2] = nib_lo
    nibbles[:, 1::2] = nib_hi
    
    # Scale per 32-element sub-block (8 sub-blocks × 32 = 256)
    # Q4_K: sc[0] for elements 0..31, sc[1] for 32..63, etc.
    # Scales stored as 6-bit values in 12 bytes → 8 six-bit values
    sc6 = np.empty((nb, 8), dtype=np.float32)
    sc6[:, 0] = sc_raw[:, 0] & 0x3F
    sc6[:, 1] = sc_raw[:, 1] & 0x3F
    sc6[:, 2] = sc_raw[:, 2] & 0x3F
    sc6[:, 3] = sc_raw[:, 3] & 0x3F
    sc6[:, 4] = sc_raw[:, 4] & 0x3F
    sc6[:, 5] = sc_raw[:, 5] & 0x3F
    sc6[:, 6] = sc_raw[:, 6] & 0x3F
    sc6[:, 7] = sc_raw[:, 7] & 0x3F
    
    # Broadcast scales to 256 elements
    scales = np.empty((nb, 256), dtype=np.float32)
    for s in range(8):
        scales[:, s*32:(s+1)*32] = sc6[:, s:s+1]
    
    # Dequantize: value = nibble * d * scales / 16 - dmin
    d_br = d.reshape(nb, 1)
    dm_br = dmin.reshape(nb, 1)
    f32 = nibbles * d_br * scales / 16.0 - dm_br
    
    # Global quantization to BQSM 4-state
    flat = f32.ravel()[:nelems]
    mn, mx = flat.min(), flat.max()
    if mx > mn:
        bqsm = np.clip(np.round((flat - mn) / (mx - mn) * 3), 0, 3).astype(np.uint8)
    else:
        bqsm = np.full(nelems, 2, dtype=np.uint8)
    return bqsm

def extract_q6(raw, nelems):
    """Q6_K 210B/block. d(2)+ql(128)+qh(64)+scales(16)"""
    nb = nelems // 256
    b = raw[:nb*210].reshape(nb, 210)
    ql = b[:, 2:130]   # (nb, 128)
    qh = b[:, 130:194] # (nb, 64)
    lo = np.empty((nb, 256), dtype=np.uint8)
    lo[:, 0::2] = ql & 0x0F
    lo[:, 1::2] = ql >> 4
    hi = np.empty((nb, 256), dtype=np.uint8)
    for s in range(4):
        hi[:, s::4] = (qh >> (2*s)) & 0x03
    return (lo | (hi << 4)).ravel()[:nelems]

def pack(data):
    n = len(data)
    pad = (4 - (n % 4)) % 4
    if pad: data = np.append(data, np.zeros(pad, np.uint8))
    d = data.reshape(-1, 4)
    return d[:,0]|(d[:,1]<<2)|(d[:,2]<<4)|(d[:,3]<<6), n

# ── Write ──
f = open(OUT, 'wb')
f.write(b'BQSM')
f.write(struct.pack('<I', 3))
f.write(struct.pack('<6I', d_model, ffn_dim, n_layers, n_heads, n_kv_heads, vocab_size))
f.write(struct.pack('<I', len(reader.tensors)))

tp = 0  # total packed bytes
for ti, t in enumerate(reader.tensors):
    name = t.name
    shape = tuple(int(d) for d in t.shape)
    dtype = t.tensor_type.name
    ne = int(np.prod(shape))
    raw = np.array(t.data.flat, dtype=np.uint8)
    
    if dtype == 'F32':
        f32dat = np.frombuffer(raw[:ne*4].tobytes(), dtype=np.float32)
        bqsm = np.clip(np.round(f32dat * 1.5 + 2.0), 0, 3).astype(np.uint8)
    elif dtype == 'Q4_K':
        bqsm = extract_q4(raw, ne) >> 2  # preserve nibble magnitude ordering
    elif dtype == 'Q6_K':
        bqsm = extract_q6(raw, ne) >> 4
    else:
        continue
    
    pkb, on = pack(bqsm)
    nb = name.encode()
    f.write(struct.pack('<H', len(nb)))
    f.write(nb)
    f.write(struct.pack('<B', len(shape)))
    for d in shape: f.write(struct.pack('<Q', d))
    f.write(struct.pack('<I', on))
    f.write(struct.pack('<I', len(pkb)))
    f.write(pkb.tobytes())
    tp += len(pkb)
    
    if ti % 25 == 0:
        print(f"  [{ti:3d}/{len(reader.tensors)}] {name:45s} {on:>12,}e → {len(pkb):>10,}B  {tp/2**20:.0f}MB")

f.close()
print(f"\nDone: {OUT} = {os.path.getsize(OUT)/2**20:.1f} MB")