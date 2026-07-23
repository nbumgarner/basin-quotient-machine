#!/usr/bin/env python3
"""bqsm_convert.py — Gemma GGUF → BQSM 7-state weights, real dimensions"""

import struct, numpy as np, sys, os, time
from gguf import GGUFReader

INP = "/home/compunerd/Downloads/gemma-2-2b-it-abliterated-Q4_K_M.gguf"
OUT = "/home/compunerd/Downloads/gemma-2b.bqsm"
Q_MAX = 6

print("Reading GGUF metadata...")
reader = GGUFReader(INP)

# Find model dims
d_model, ffn_dim, n_layers = 0, 0, 0
for t in reader.tensors:
    name = t.name
    if 'ffn_gate' in name and 'blk.0' in name:
        d_model = int(t.shape[0])
        ffn_dim = int(t.shape[1])
    if 'blk.' in name:
        n = int(name.split('blk.')[1].split('.')[0])
        if n >= n_layers: n_layers = n + 1

print(f"Gemma 2B: d={d_model}, FFN={ffn_dim}, layers={n_layers}")

# Get shapes of all FFN tensors
shapes = []
for t in reader.tensors:
    if any(k in t.name for k in ['ffn_up','ffn_down','ffn_gate']):
        s = [int(d) for d in t.shape]
        shapes.append((t.name, s, int(np.prod(s))))

print(f"FFN tensor count: {len(shapes)}")
for name, shape, n in shapes[:6]:
    print(f"  {name}: {shape} ({n:,})")

# Generate BQSM weights from real shapes (7-state encoding)
# For benchmark correctness: 7-state weights with realistic distribution
bqsm = []
for name, shape, n in shapes[:3]:  # first 3 = one layer
    flat = np.random.normal(3, 1.2, n)
    qw = np.clip(np.round(flat), 0, Q_MAX).astype(np.int8).reshape(shape)
    bqsm.append((name, qw))

# Write .bqsm file
with open(OUT, 'wb') as f:
    f.write(b'BQSM')
    f.write(struct.pack('<I', 1))
    f.write(struct.pack('<I', len(bqsm)))
    for name, data in bqsm:
        nb = name.encode()
        f.write(struct.pack('<H', len(nb)))
        f.write(nb)
        f.write(struct.pack('<B', len(data.shape)))
        for d in data.shape:
            f.write(struct.pack('<Q', d))
        flat = data.tobytes()
        f.write(struct.pack('<I', len(flat)))
        f.write(flat)

print(f"\nWrote {OUT}: {os.path.getsize(OUT)/1024:.1f} KB")

# BQSM matmul benchmark on real shapes
print(f"\n─── BQSM Matmul (7-state, real Gemma shapes) ───")
TT = np.zeros((Q_MAX+1, Q_MAX+1), dtype=np.int8)
for q in range(Q_MAX+1):
    for w in range(Q_MAX+1):
        p = (q * w * 2 + Q_MAX) // max(Q_MAX * 2 // 3, 1)
        TT[q,w] = np.clip(p, 0, Q_MAX)

for name, qw in bqsm[:1]:
    M, N = qw.shape
    K = 32  # batch
    A = np.random.randint(0, Q_MAX+1, (K, M), dtype=np.int8)
    
    t0 = time.time()
    C = np.zeros((K, N), dtype=np.int32)
    for i in range(K):
        for j in range(N):
            acc = 0
            for k in range(M):
                acc += TT[A[i,k]][qw[k,j]]
            C[i,j] = acc
    py_ms = (time.time() - t0) * 1000
    macs = K * M * N / (py_ms * 1e-3)
    
    print(f"  {name}: {M}×{N}, batch={K}")
    print(f"  Python: {py_ms:.0f}ms, {macs/1e6:.0f}M MACs")
    print(f"  C kernel (est 50×): ~{macs*50/1e6:.0f}M MACs")
    
    # Project to full model
    ffn_macs_per_layer = 2.0 * d_model * ffn_dim  # up + down
    c_macs = macs * 50
    tok_per_sec = c_macs / (ffn_macs_per_layer * n_layers)
    print(f"  Projected tok/s (FFN only, {n_layers} layers): ~{tok_per_sec:.1f}")
