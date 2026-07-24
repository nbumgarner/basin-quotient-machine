#!/usr/bin/env python3
"""bqsm_convert_nibble.py — Q4_K/Q6_K nibbles → BQSM v3 directly.
No dequantization. No global min/max. Just >>2 (Q4_K) or >>4 (Q6_K).
Preserves the GGUF quantization structure block-by-block."""
import struct, numpy as np, os
from gguf import GGUFReader

INP = "/home/compunerd/Downloads/gemma-2-2b-it-abliterated-Q4_K_M.gguf"
OUT = "/home/compunerd/Downloads/gemma-2b-nibble.bqsm"

reader = GGUFReader(INP)
d_model=2304; ffn_dim=9216; n_layers=26; n_heads=8; n_kv_heads=4; vocab=256000
print(f"Q4_K/Q6_K nibble extractor: {len(reader.tensors)} tensors")

def extract_q4(raw, nelems):
    """Q4_K: 144B/block. Extract 4-bit nibbles from qs[4:132] (128 bytes).
    Interleave lo/hi nibbles. Map 0-15 → 0-3 via >>2."""
    nb = nelems // 256
    blocks = raw[:nb*144].reshape(nb, 144)
    qs = blocks[:, 4:132]  # (nb, 128)
    lo = qs & 0x0F
    hi = qs >> 4
    nibbles = np.empty((nb, 256), dtype=np.uint8)
    nibbles[:, 0::2] = lo
    nibbles[:, 1::2] = hi
    return (nibbles.ravel()[:nelems] >> 2).astype(np.uint8)  # 0-15 → 0-3

def extract_q6(raw, nelems):
    """Q6_K: 210B/block. ql[2:130] (128B lo nibbles) + qh[130:194] (64B hi bits).
    Full 6-bit: lo | (hi << 4). Map 0-63 → 0-3 via >>4."""
    nb = nelems // 256
    b = raw[:nb*210].reshape(nb, 210)
    ql = b[:, 2:130]
    qh = b[:, 130:194]
    lo = np.empty((nb, 256), dtype=np.uint8)
    lo[:, 0::2] = ql & 0x0F
    lo[:, 1::2] = ql >> 4
    hi = np.empty((nb, 256), dtype=np.uint8)
    for s in range(4):
        hi[:, s::4] = (qh >> (2*s)) & 0x03
    full = lo | (hi << 4)  # 6-bit: 0-63
    return (full.ravel()[:nelems] >> 4).astype(np.uint8)  # 0-63 → 0-3

def pack_4state(data):
    n = len(data)
    pad = (4 - (n % 4)) % 4
    if pad: data = np.append(data, np.zeros(pad, np.uint8))
    d = data.reshape(-1, 4)
    return d[:,0]|(d[:,1]<<2)|(d[:,2]<<4)|(d[:,3]<<6), n

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
        mn, mx = f32dat.min(), f32dat.max()
        if mx > mn:
            bqsm = np.clip(np.round((f32dat - mn) / (mx - mn) * 3), 0, 3).astype(np.uint8)
        else:
            bqsm = np.full(ne, 2, dtype=np.uint8)
    elif dtype == 'Q4_K':
        bqsm = extract_q4(raw, ne)
    elif dtype == 'Q6_K':
        bqsm = extract_q6(raw, ne)
    else:
        continue
    
    pkb, on = pack_4state(bqsm)
    nb = name.encode()
    f.write(struct.pack('<H', len(nb))); f.write(nb)
    f.write(struct.pack('<B', len(shape)))
    for d in shape: f.write(struct.pack('<Q', d))
    f.write(struct.pack('<I', on)); f.write(struct.pack('<I', len(pkb)))
    f.write(pkb.tobytes())
    tp += len(pkb)
    
    if ti % 25 == 0:
        print(f"  [{ti:3d}/{len(reader.tensors)}] {name:45s} {on:>12,}e → {len(pkb):>10,}B  {tp/2**20:.0f}MB")

f.close()
print(f"\nDone: {OUT} = {os.path.getsize(OUT)/2**20:.1f} MB")
