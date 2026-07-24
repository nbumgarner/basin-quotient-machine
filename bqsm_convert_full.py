#!/usr/bin/env python3
"""bqsm_convert_full.py — Extract ALL tensor shapes from GGUF, write full BQSM model.
Generates 4-state BQSM weights. Correct shapes, complete architecture.
For real weights: use C-based Q4_K dequantizer (future work).
"""
import struct, numpy as np, sys, os, time
from collections import defaultdict
from gguf import GGUFReader

INP = "/home/compunerd/Downloads/gemma-2-2b-it-abliterated-Q4_K_M.gguf"
OUT = "/home/compunerd/Downloads/gemma-2b-full.bqsm"
Q_STATES = 4  # 4-state (2-bit) BQSM encoding

print("Reading GGUF metadata and tensor shapes...")
reader = GGUFReader(INP)

# ── Extract architecture ──
d_model, ffn_dim, n_layers, n_heads, n_kv_heads = 0, 0, 0, 0, 0
vocab_size = 0

for t in reader.tensors:
    name = t.name
    if name == 'token_embd.weight':
        d_model = int(t.shape[0])
        vocab_size = int(t.shape[1])
    if 'ffn_gate' in name and 'blk.0' in name:
        ffn_dim = int(t.shape[1])
    if 'blk.' in name:
        parts = name.split('blk.')[1].split('.')
        if parts[0].isdigit():
            n = int(parts[0])
            if n >= n_layers:
                n_layers = n + 1
    if 'attn_q' in name and 'blk.0' in name:
        n_heads = 8  # Gemma 2B
        n_kv_heads = 4

print(f"Gemma 2B Architecture:")
print(f"  d_model={d_model}, ffn_dim={ffn_dim}, vocab={vocab_size}")
print(f"  n_layers={n_layers}, n_heads={n_heads}, n_kv_heads={n_kv_heads}")
print(f"  head_dim={d_model // n_heads}, kv_head_dim={d_model // n_heads}")

# ── Collect all tensors by logical name ──
# Group by logical tensor (strip any q-weight suffixes if present)
tensors_by_name = {}
for t in reader.tensors:
    name = t.name
    shape = tuple(int(d) for d in t.shape)
    dtype = t.tensor_type.name
    tensors_by_name[name] = (shape, dtype)

print(f"\nTotal logical tensors: {len(tensors_by_name)}")

# ── Generate BQSM weights ──
# Map each tensor to 4-state BQSM: values in [0, Q_STATES-1]
# For norms (1D, F32): convert to 4-state by bucketing
# For weight matrices: generate random 4-state (placeholder)
# For embeddings: 4-state (large, use random)

def quantize_to_bqsm(data_1d, n_states=Q_STATES):
    """Quantize a float32 array to BQSM n-state encoding."""
    if len(data_1d) == 0:
        return np.array([], dtype=np.int8)
    # Normalize to [0, n_states-1]
    mn, mx = data_1d.min(), data_1d.max()
    if mx - mn < 1e-8:
        return np.zeros(len(data_1d), dtype=np.int8)
    scaled = (data_1d - mn) / (mx - mn) * (n_states - 1)
    return np.clip(np.round(scaled), 0, n_states - 1).astype(np.int8)

def generate_bqsm_weights(shape, dtype, rng):
    """Generate BQSM weights with correct shape."""
    n = int(np.prod(shape))
    
    if dtype == 'F32' and len(shape) == 1:
        # Norm weights: generate plausible float range, then quantize
        data = rng.normal(1.0, 0.2, n).astype(np.float32)
        return quantize_to_bqsm(data)
    elif dtype == 'F32':
        data = rng.normal(0, 0.5, n).astype(np.float32)
        return quantize_to_bqsm(data)
    else:
        # Quantized tensors: generate random 4-state weights
        # For real model: these would come from Q4_K dequantization
        return rng.randint(0, Q_STATES, n, dtype=np.int8)

rng = np.random.RandomState(42)

# Write order: embeddings, then layers 0..n_layers-1, then output
bqsm_tensors = []

# 1. Token embeddings
name = 'token_embd.weight'
if name in tensors_by_name:
    shape, dtype = tensors_by_name[name]
    data = generate_bqsm_weights(shape, dtype, rng)
    bqsm_tensors.append((name, shape, data))
    print(f"  {name}: {shape} -> {data.nbytes/1024:.1f} KB BQSM")

# 2. Each layer
for layer in range(n_layers):
    prefix = f'blk.{layer}.'
    layer_tensors = []
    # Order matters for loading: norms, attention, FFN
    for key in ['attn_norm.weight', 'attn_q.weight', 'attn_k.weight', 
                'attn_v.weight', 'attn_output.weight',
                'post_attention_norm.weight', 'ffn_norm.weight',
                'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight',
                'post_ffw_norm.weight']:
        name = prefix + key
        if name in tensors_by_name:
            shape, dtype = tensors_by_name[name]
            data = generate_bqsm_weights(shape, dtype, rng)
            bqsm_tensors.append((name, shape, data))
            print(f"  {name}: {shape} -> {data.nbytes/1024:.1f} KB BQSM")

# 3. Output norm + lm_head (if exists)
for name in ['output_norm.weight']:
    if name in tensors_by_name:
        shape, dtype = tensors_by_name[name]
        data = generate_bqsm_weights(shape, dtype, rng)
        bqsm_tensors.append((name, shape, data))
        print(f"  {name}: {shape} -> {data.nbytes/1024:.1f} KB BQSM")

# ── Write BQSM file ──
# New format v2: includes architecture metadata
with open(OUT, 'wb') as f:
    # Magic + version
    f.write(b'BQSM')
    f.write(struct.pack('<I', 2))  # version 2
    
    # Architecture header
    arch_bytes = struct.pack('<6I', d_model, ffn_dim, n_layers, n_heads, n_kv_heads, vocab_size)
    f.write(struct.pack('<I', len(arch_bytes)))
    f.write(arch_bytes)
    
    # Tensor count
    f.write(struct.pack('<I', len(bqsm_tensors)))
    
    # Each tensor: name, ndim, shape[], data_len, data[]
    total_bytes = 0
    for name, shape, data in bqsm_tensors:
        nb = name.encode()
        f.write(struct.pack('<H', len(nb)))
        f.write(nb)
        f.write(struct.pack('<B', len(shape)))
        for d in shape:
            f.write(struct.pack('<Q', d))
        flat = data.tobytes()
        f.write(struct.pack('<I', len(flat)))
        f.write(flat)
        total_bytes += len(flat)

fsize = os.path.getsize(OUT)
print(f"\nWrote {OUT}: {fsize/1024:.1f} KB ({fsize/1024/1024:.1f} MB)")
print(f"  {len(bqsm_tensors)} tensors, {total_bytes/1024/1024:.1f} MB weight data")
print(f"  Architecture: Gemma 2B, {n_layers} layers, d={d_model}, FFN={ffn_dim}")
