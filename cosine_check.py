#!/usr/bin/env python3
"""cosine_check.py — Proper Q4_K dequant + BQSM nibble-mapping cosine gate.
Dequantizes one ffn_up tensor from GGUF using correct ggml formula:
  y = d * sc * (nibble - m) - dmin
Compares float reference matmul against BQSM nibble-mapped matmul.
Reports cosine similarity — the actual number distinguishing correct from noise."""
import numpy as np, struct
from gguf import GGUFReader

GGUF = '/home/compunerd/Downloads/gemma-2-2b-it-abliterated-Q4_K_M.gguf'
SLICE = 1024  # square slice for matmul comparison

reader = GGUFReader(GGUF)

# Find a Q4_K tensor of interest
tnames = ['blk.0.ffn_up.weight', 'blk.5.ffn_up.weight', 'blk.20.ffn_up.weight']
results = {}

for tname in tnames:
    for t in reader.tensors:
        if t.name == tname:
            raw = np.array(t.data.flat, dtype=np.uint8)
            shape = tuple(int(d) for d in t.shape)
            nelems = int(np.prod(shape))
            M, N = shape
            break
    else:
        print(f"  {tname}: not found")
        continue

    print(f"\n{'='*60}")
    print(f"  {tname}: {M}x{N} = {nelems:,} elements")
    
    nb = nelems // 256
    blocks = raw[:nb*144].reshape(nb, 144)
    
    # --- Proper ggml Q4_K dequant ---
    # d, dmin at bytes 0-1, 2-3
    d   = blocks[:, 0:2].view(np.float16).flatten().astype(np.float32)
    dmin = blocks[:, 2:4].view(np.float16).flatten().astype(np.float32)
    
    # Q nibbles at bytes 4-131 (128 bytes)
    qs = blocks[:, 4:132]
    lo = (qs & 0x0F).astype(np.float32)
    hi = (qs >> 4).astype(np.float32)
    nibbles = np.empty((nb, 256), dtype=np.float32)
    nibbles[:, 0::2] = lo
    nibbles[:, 1::2] = hi
    
    # Scales + mins at bytes 132-143 (12 bytes, 6-bit each)
    sc_raw = blocks[:, 132:144]
    # ggml Q4_K scale unpacking (all operations on uint8 before float cast):
    sc_raw_i = sc_raw.astype(np.int32)
    sc = np.empty((nb, 8), dtype=np.float32)
    m  = np.empty((nb, 8), dtype=np.float32)
    
    sc[:, 0] = (sc_raw_i[:, 0] & 0x3F).astype(np.float32)
    sc[:, 1] = (sc_raw_i[:, 1] & 0x3F).astype(np.float32)
    sc[:, 2] = (sc_raw_i[:, 2] & 0x3F).astype(np.float32)
    sc[:, 3] = (sc_raw_i[:, 3] & 0x3F).astype(np.float32)
    sc[:, 4] = (sc_raw_i[:, 4] & 0x3F).astype(np.float32)
    sc[:, 5] = (sc_raw_i[:, 5] & 0x3F).astype(np.float32)
    sc[:, 6] = ((sc_raw_i[:, 6] & 0x3F) | ((sc_raw_i[:, 9] & 0x30) << 2)).astype(np.float32)
    sc[:, 7] = ((sc_raw_i[:, 7] & 0x3F) | ((sc_raw_i[:, 9] & 0xC0) << 0)).astype(np.float32)
    
    m[:, 0] = (sc_raw_i[:, 8] & 0x3F).astype(np.float32)
    m[:, 1] = (sc_raw_i[:, 9] & 0x3F).astype(np.float32)
    m[:, 2] = (sc_raw_i[:, 10] & 0x3F).astype(np.float32)
    m[:, 3] = (sc_raw_i[:, 11] & 0x3F).astype(np.float32)
    m[:, 4] = ((sc_raw_i[:, 11] >> 6) | ((sc_raw_i[:, 10] & 0x30) << 0)).astype(np.float32)
    m[:, 5] = ((sc_raw_i[:, 10] >> 6) | ((sc_raw_i[:, 9]  & 0x0C) << 2)).astype(np.float32)
    m[:, 6] = (sc_raw_i[:, 9] >> 6).astype(np.float32)
    m[:, 7] = (sc_raw_i[:, 8] >> 6).astype(np.float32)
    
    # Broadcast to 256 elements
    sc_br = np.empty((nb, 256), dtype=np.float32)
    m_br  = np.empty((nb, 256), dtype=np.float32)
    for s in range(8):
        sc_br[:, s*32:(s+1)*32] = sc[:, s:s+1]
        m_br[:,  s*32:(s+1)*32] = m[:,  s:s+1]
    
    d_br   = d.reshape(nb, 1)
    dm_br  = dmin.reshape(nb, 1)
    
    # CORRECT ggml dequant: y = d * sc * (nibble - m) - dmin
    W_float = (d_br * sc_br * (nibbles - m_br) - dm_br).ravel()[:nelems]
    W = W_float.reshape(M, N).astype(np.float32)
    
    print(f"  Float weights: range [{W.min():.4f}, {W.max():.4f}]  mean={W.mean():.6f}  std={W.std():.6f}")
    
    # --- BQSM nibble mapping: nibble >> 2 ---
    nib_raw = np.empty((nb, 256), dtype=np.uint8)
    nib_raw[:, 0::2] = qs & 0x0F
    nib_raw[:, 1::2] = qs >> 4
    bqsm_states = (nib_raw.ravel()[:nelems] >> 2).astype(np.int32)  # 0-15 → 0-3
    W_bqsm = bqsm_states.reshape(M, N)
    
    b0,b1,b2,b3 = [(W_bqsm==i).sum()/nelems*100 for i in range(4)]
    print(f"  BQSM states: {b0:.0f}/{b1:.0f}/{b2:.0f}/{b3:.0f} pct")
    
    # --- Cosine gate ---
    np.random.seed(42)
    x = np.random.randn(M).astype(np.float32) * 0.5
    
    # Float reference: x @ W_slice (using first SLICE cols)
    W_slice = W[:, :SLICE]
    ref = x @ W_slice
    
    # BQSM matmul: use ternary LUT {-1,0,1,0}
    LUT = np.array([-1, 0, 1, 0], dtype=np.int32)
    W_bqsm_slice = W_bqsm[:, :SLICE]
    x_int = np.clip(np.round(x * 2 + 2), 0, 3).astype(np.int32)
    
    bqsm_out = np.zeros(SLICE, dtype=np.int32)
    for j in range(SLICE):
        bqsm_out[j] = np.sum(LUT[W_bqsm_slice[:, j]] * x_int)
    
    # Cosine
    ref_n = ref / (np.linalg.norm(ref) + 1e-8)
    bqsm_n = bqsm_out.astype(np.float32) / (np.linalg.norm(bqsm_out.astype(np.float32)) + 1e-8)
    cos_sim = float(np.dot(ref_n, bqsm_n))
    
    # Also: cosine between float ref and random (baseline)
    rand_out = np.random.randn(SLICE).astype(np.float32)
    rand_n = rand_out / (np.linalg.norm(rand_out) + 1e-8)
    cos_rand = abs(float(np.dot(ref_n, rand_n)))
    
    print(f"\n  Cosine (float ref vs BQSM nibble): {cos_sim:+.4f}")
    print(f"  Cosine (float ref vs random baseline): {cos_rand:.4f}")
    
    if cos_sim > 0.5:
        print(f"  ✓ Above noise floor — nibble mapping preserves direction")
    elif cos_sim > cos_rand * 2:
        print(f"  ~ Marginally above random")
    else:
        print(f"  ✗ At or below random — nibble mapping loses weight structure")
    
    results[tname] = cos_sim

print(f"\n{'='*60}")
print("Summary:")
for name, c in results.items():
    print(f"  {name}: cosine = {c:+.4f}")
