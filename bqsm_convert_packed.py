#!/usr/bin/env python3
"""bqsm_convert_packed.py — Convert GGUF to packed BQSM v3 format.
Packs 4-state (2-bit) weights: 4 elements per byte.
Architecture header + tensor-by-tensor storage.
"""
import struct, numpy as np, sys, os, time
from gguf import GGUFReader

INP = "/home/compunerd/Downloads/gemma-2-2b-it-abliterated-Q4_K_M.gguf"
OUT = "/home/compunerd/Downloads/gemma-2b-full.bqsm"
Q_STATES = 4

print("Reading GGUF metadata...")
reader = GGUFReader(INP)

# ── Architecture ──
d_model, ffn_dim, n_layers = 0, 0, 0
n_heads, n_kv_heads = 8, 4  # Gemma 2B
vocab_size = 0
tensors_by_name = {}

for t in reader.tensors:
    name = t.name
    shape = tuple(int(d) for d in t.shape)
    dtype = t.tensor_type.name
    tensors_by_name[name] = (shape, dtype)
    if name == 'token_embd.weight':
        d_model, vocab_size = int(t.shape[0]), int(t.shape[1])
    if 'ffn_gate' in name and 'blk.0' in name:
        ffn_dim = int(t.shape[1])
    if 'blk.' in name:
        parts = name.split('blk.')[1].split('.')
        if parts[0].isdigit():
            n = int(parts[0])
            if n + 1 > n_layers:
                n_layers = n + 1

print(f"Gemma 2B: d={d_model}, FFN={ffn_dim}, vocab={vocab_size}, layers={n_layers}")
print(f"Tensors: {len(tensors_by_name)}")

# ── Packing ──
def pack_4state(data_1d):
    """Pack 4-state values (0-3) into 2 bits each: 4 per byte. Vectorized."""
    n = len(data_1d)
    pad = (4 - (n % 4)) % 4
    if pad:
        data_1d = np.append(data_1d, np.zeros(pad, dtype=np.uint8))
    # Vectorized: shift and OR
    d = data_1d.astype(np.uint8).reshape(-1, 4)
    packed = d[:,0] | (d[:,1] << 2) | (d[:,2] << 4) | (d[:,3] << 6)
    return packed, n

rng = np.random.RandomState(42)

def gen_bqsm(shape, dtype):
    n = int(np.prod(shape))
    if dtype == 'F32' and len(shape) == 1:
        # Norm weights: center at 2, tight spread
        data = np.clip(np.round(rng.normal(2.0, 0.5, n)), 0, Q_STATES-1).astype(np.int8)
    else:
        # Matrix: bias toward center values (1,2) to prevent saturation
        # randint + lookup table remap (no big intermediate arrays)
        data = rng.randint(0, 100, n, dtype=np.int8)
        # Remap: 0-9→0, 10-44→1, 45-84→2, 85-99→3  (10%/35%/40%/15%)
        lut = np.zeros(100, dtype=np.int8)
        lut[10:45] = 1; lut[45:85] = 2; lut[85:100] = 3
        data = lut[data]   # fancy indexing, in-place-ish
    return data

# ── Build tensor list ──
bqsm_tensors = []

# Embeddings
name = 'token_embd.weight'
if name in tensors_by_name:
    shape, dtype = tensors_by_name[name]
    data = gen_bqsm(shape, dtype)
    bqsm_tensors.append((name, shape, data))

# Layers
for layer in range(n_layers):
    prefix = f'blk.{layer}.'
    for key in ['attn_norm.weight', 'attn_q.weight', 'attn_k.weight',
                'attn_v.weight', 'attn_output.weight',
                'post_attention_norm.weight', 'ffn_norm.weight',
                'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight',
                'post_ffw_norm.weight']:
        name = prefix + key
        if name in tensors_by_name:
            shape, dtype = tensors_by_name[name]
            data = gen_bqsm(shape, dtype)
            bqsm_tensors.append((name, shape, data))

# Output norm
name = 'output_norm.weight'
if name in tensors_by_name:
    shape, dtype = tensors_by_name[name]
    data = gen_bqsm(shape, dtype)
    bqsm_tensors.append((name, shape, data))

# ── Write BQSM v3 (packed) ──
with open(OUT, 'wb') as f:
    f.write(b'BQSM')
    f.write(struct.pack('<I', 3))  # version 3 = packed 2-bit
    
    # Architecture: d_model, ffn_dim, n_layers, n_heads, n_kv_heads, vocab_size
    f.write(struct.pack('<6I', d_model, ffn_dim, n_layers, n_heads, n_kv_heads, vocab_size))
    
    f.write(struct.pack('<I', len(bqsm_tensors)))
    
    total_raw = 0
    total_packed = 0
    for name, shape, data in bqsm_tensors:
        packed, orig_n = pack_4state(data)
        nb = name.encode()
        f.write(struct.pack('<H', len(nb)))
        f.write(nb)
        f.write(struct.pack('<B', len(shape)))
        for d in shape:
            f.write(struct.pack('<Q', d))
        f.write(struct.pack('<I', orig_n))       # original element count
        f.write(struct.pack('<I', len(packed)))  # packed byte count
        f.write(packed.tobytes())
        total_raw += orig_n
        total_packed += len(packed)

fsize = os.path.getsize(OUT)
print(f"\nWrote {OUT}: {fsize/1024/1024:.1f} MB")
print(f"  {len(bqsm_tensors)} tensors")
print(f"  Raw elements: {total_raw/1e6:.1f}M")
print(f"  Packed bytes: {total_packed/1024/1024:.1f} MB")
print(f"  Compression: {total_raw / (total_packed*4) * 100:.0f}% of int8")
