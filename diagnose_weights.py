#!/usr/bin/env python3
"""
Diagnostic: Compare L0 vs L1 weight statistics to find scale mismatch.
"""
import json
import urllib.request
import numpy as np
import struct

BASE_URL = 'http://localhost:8082'

# BQSM ternary LUT (2-bit values: 0,1,2,3 -> -1, -0.5, 0.5, 1)
TERNARY_LUT = {0: -1.0, 1: -0.5, 2: 0.5, 3: 1.0}

def get_tensor(name):
    """Get tensor data from debug endpoint."""
    url = f'{BASE_URL}/debug/tensor?name={name}&max=10000'
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode('utf-8'))

def dequant_packed_2bit(packed_bytes):
    """Dequantize packed 2-bit values to float."""
    vals = []
    for byte in packed_bytes:
        for shift in [0, 2, 4, 6]:
            nb = (byte >> shift) & 3
            vals.append(TERNARY_LUT[nb])
    return np.array(vals, dtype=np.float32)

def analyze_tensor(name, expected_shape=None):
    """Load and dequantize a tensor, return stats."""
    data = get_tensor(name)
    if 'values' not in data:
        print(f'  Error: {data.get("error", "no values")}')
        return None

    packed = np.array(data['values'], dtype=np.uint8)
    floats = dequant_packed_2bit(packed)

    # Don't reshape - just work with flat array
    # (debug endpoint only returns partial data)
    return floats

print('=== L0 vs L1 WEIGHT ANALYSIS ===\n')

for layer in [0, 1]:
    print(f'--- Layer {layer} ---')

    # Q weight: [d_model, q_dim] = [2304, 2048]
    q_name = f'blk.{layer}.attn_q.weight'
    q = analyze_tensor(q_name, (2304, 2048))
    if q is not None:
        print(f'  Q: shape={q.shape}, mean={q.mean():.4f}, std={q.std():.4f}, '
              f'RMS={np.sqrt((q**2).mean()):.4f}')

    # K weight: [d_model, kv_dim] = [2304, 1024]
    k_name = f'blk.{layer}.attn_k.weight'
    k = analyze_tensor(k_name, (2304, 1024))
    if k is not None:
        print(f'  K: shape={k.shape}, mean={k.mean():.4f}, std={k.std():.4f}, '
              f'RMS={np.sqrt((k**2).mean()):.4f}')

    # V weight: [d_model, kv_dim] = [2304, 1024]
    v_name = f'blk.{layer}.attn_v.weight'
    v = analyze_tensor(v_name, (2304, 1024))
    if v is not None:
        print(f'  V: shape={v.shape}, mean={v.mean():.4f}, std={v.std():.4f}, '
              f'RMS={np.sqrt((v**2).mean()):.4f}')

    print()

print('=== SCALE RATIO (L1/L0) ===')
# We can't compute exact ratios without full tensors, but we can compare RMS
# From the analysis above, we should see if L1 weights are systematically different

print('\nNext: Check KV-cache read/write for L1')
print('The issue: L0 works (cosine ~0.9), L1 fails (cosine ~0.0)')
print('Possible causes:')
print('1. KV-cache stride mismatch (GQA: 4 KV heads, 8 Q heads)')
print('2. Per-tensor scale not propagated correctly')
print('3. Cache read path uses wrong indexing')
