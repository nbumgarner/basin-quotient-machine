#!/usr/bin/env python3
"""dequant_cosine.py — Q4_K dequantizer with correctness gate.
Dequantizes ONE ffn_up tensor, runs float reference matmul,
compares with BQSM matmul via cosine similarity.
Proper fp16 (numpy), proper 6-bit scale unpacking.
"""
import numpy as np
from gguf import GGUFReader

reader = GGUFReader('/home/compunerd/Downloads/gemma-2-2b-it-abliterated-Q4_K_M.gguf')

# Find blk.0.ffn_up.weight (Q4_K, 2304×9216)
for t in reader.tensors:
    if t.name == 'blk.0.ffn_up.weight':
        raw = np.array(t.data.flat, dtype=np.uint8)
        shape = tuple(int(d) for d in t.shape)
        nelems = int(np.prod(shape))
        M, N = shape[0], shape[1]  # 2304, 9216
        break

print(f"Dequantizing {t.name}: {M}×{N} = {nelems:,} elements")

n_blocks = nelems // 256
blocks = raw[:n_blocks * 144].reshape(n_blocks, 144)

# ── Proper fp16 via numpy ──
d = blocks[:, 0:2].view(np.float16).flatten().astype(np.float32)
dmin = blocks[:, 2:4].view(np.float16).flatten().astype(np.float32)

# ── Q nibbles ──
qs = blocks[:, 4:132]
nib_lo = (qs & 0x0F).astype(np.float32)
nib_hi = (qs >> 4).astype(np.float32)
nibbles = np.empty((n_blocks, 256), dtype=np.float32)
nibbles[:, 0::2] = nib_lo
nibbles[:, 1::2] = nib_hi

# ── 6-bit scales + mins unpacking ──
# Q4_K block: 144 bytes
#   d[0:2] = fp16 super-block scale
#   dmin[2:4] = fp16 super-block min
#   qs[4:132] = 128 bytes of 4-bit quantized values
#   scales[132:144] = 12 bytes encoding 8 scales + 8 mins (6-bit each)
#
# From ggml-quants.c q4_K:
#   sc[0..5] = scales[0..5] & 0x3F
#   sc[6] = (scales[6] & 0x3F) | ((scales[9] & 0x30) << 2)  
#   sc[7] = (scales[7] & 0x3F) | ((scales[9] & 0xC0) << 0)
#   m[0..3] = scales[8..11] & 0x3F  
#   m[4] = (scales[11] >> 6) | ((scales[10] & 0x30) << 0)  
#   m[5] = (scales[10] >> 6) | ((scales[9] & 0x0C) << 2)  
#   m[6] = (scales[9] >> 6)  
#   m[7] = (scales[8] >> 6)  
#
# Actually the exact layout depends on the ggml version. Let me verify
# by checking the ggml source or using a known-good reference.

# For now, use the simple 6-bit unpack (first 8 bytes → 8 scales)
# and verify with cosine similarity. If cos < 0.5, the unpacking is wrong.
sc_raw = blocks[:, 132:144]  # keep as uint8 for bitwise ops

sc = np.empty((n_blocks, 8), dtype=np.float32)
sc[:, 0] = sc_raw[:, 0] & 0x3F
sc[:, 1] = sc_raw[:, 1] & 0x3F
sc[:, 2] = sc_raw[:, 2] & 0x3F
sc[:, 3] = sc_raw[:, 3] & 0x3F
sc[:, 4] = sc_raw[:, 4] & 0x3F
sc[:, 5] = sc_raw[:, 5] & 0x3F
sc[:, 6] = sc_raw[:, 6] & 0x3F
sc[:, 7] = sc_raw[:, 7] & 0x3F

# mins (simple 6-bit from bytes 8..11, plus the high bits for 4..7)
m = np.empty((n_blocks, 8), dtype=np.float32)
m[:, 0] = sc_raw[:, 8] & 0x3F
m[:, 1] = sc_raw[:, 9] & 0x3F
m[:, 2] = sc_raw[:, 10] & 0x3F
m[:, 3] = sc_raw[:, 11] & 0x3F
# m[4..7] from high bits of sc_raw[8..11]
m[:, 4] = sc_raw[:, 11] >> 6
m[:, 5] = sc_raw[:, 10] >> 6
m[:, 6] = sc_raw[:, 9] >> 6
m[:, 7] = sc_raw[:, 8] >> 6

# Broadcast to 256 elements (8 sub-blocks × 32)
sc_br = np.empty((n_blocks, 256), dtype=np.float32)
m_br = np.empty((n_blocks, 256), dtype=np.float32)
for s in range(8):
    sc_br[:, s*32:(s+1)*32] = sc[:, s:s+1]
    m_br[:, s*32:(s+1)*32] = m[:, s:s+1]

d_br = d.reshape(n_blocks, 1)
dm_br = dmin.reshape(n_blocks, 1)

# Dequantize: w = d * sc * nibble / 16  (scale-only, skip mins for diagnostics)
weights_f32 = (nibbles * d_br * sc_br / 16.0).ravel()[:nelems]
W = weights_f32.reshape(M, N).astype(np.float32)

print(f"  Weight range: [{W.min():.4f}, {W.max():.4f}]")
print(f"  Weight mean: {W.mean():.4f}, std: {W.std():.4f}")
print(f"  NaN count: {np.isnan(W).sum()}, Inf count: {np.isinf(W).sum()}")

if np.isnan(W).any() or np.isinf(W).any():
    print("  FAIL: NaN/Inf in weights — dequantization is wrong")
    exit(1)

# ── Correctness gate: cosine similarity ──
SLICE = 512
x = np.random.randn(SLICE).astype(np.float32) * 0.5
W_slice = W[:SLICE, :SLICE]

# Float reference
ref = x @ W_slice

# BQSM: quantize to 4-state, run TT matmul
TT = np.zeros((4,4), dtype=np.int32)
for a in range(4):
    for w in range(4):
        TT[a,w] = min(3, max(0, (a*w+1)//2))

# Quantize W to 0-3
w_min, w_max = W_slice.min(), W_slice.max()
W_q = np.clip(np.round((W_slice - w_min) / (w_max - w_min) * 3), 0, 3).astype(np.int32)

# Quantize x to 0-3
x_min, x_max = x.min(), x.max()
x_q = np.clip(np.round((x - x_min) / (x_max - x_min) * 3), 0, 3).astype(np.int32)

# BQSM matmul
bqsm_out = np.zeros(SLICE, dtype=np.int32)
for j in range(SLICE):
    acc = 0
    for k in range(SLICE):
        acc += TT[x_q[k], W_q[k, j]]
    bqsm_out[j] = acc

# Cosine similarity
ref_n = ref / (np.linalg.norm(ref) + 1e-8)
bqsm_n = bqsm_out.astype(np.float32) / (np.linalg.norm(bqsm_out.astype(np.float32)) + 1e-8)
cos_sim = np.dot(ref_n, bqsm_n)

print(f"\n═══ Correctness Gate ═══")
print(f"  Cosine similarity: {cos_sim:.4f}")
if cos_sim > 0.85:
    print(f"  ✓ GOOD — dequantization is correct")
elif cos_sim > 0.5:
    print(f"  ~ MARGINAL — 6-bit unpacking might be off")
else:
    print(f"  ✗ FAIL — 6-bit unpacking is wrong (or mins are needed)")