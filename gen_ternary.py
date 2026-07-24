#!/usr/bin/env python3
"""gen_ternary.py — Generate placeholder ternary BQSM model (packed 2-bit, -1/0/1).
Same architecture as Gemma 2B. For signed-path kernel benchmarking.
"""
import struct, numpy as np, os

OUT = "/home/compunerd/Downloads/gemma-ternary.bqsm"
d_model=2304; ffn_dim=9216; n_layers=26; n_heads=8; n_kv_heads=4; vocab=256000

# Ternary mapping: 0→-1, 1→0, 2→1  (or 0→0, 1→1, 2→-1 for signed)
# Packed: 2 bits per value, 4 per byte. Encoding: 00=-1, 01=0, 10=1, 11=unused
# For _mm_sign_epi8: we store raw int8 values -1/0/1
# But for packed storage: nibble encoding above

def gen_weights(shape, dtype_name):
    n = int(np.prod(shape))
    if dtype_name == 'F32' and len(shape) == 1:
        return np.full(n, 1, dtype=np.int8)
    else:
        # randint -1..1 is much faster than choice()
        rng = np.random.RandomState(abs(hash(str(shape))) % 2**31)
        data = rng.randint(0, 3, n, dtype=np.int8) - 1  # 0,1,2 → -1,0,1
        return data

def pack_2bit(data):
    """Pack int8 values (-1,0,1) into 2 bits each: 00=-1, 01=0, 10=1"""
    # Map to packed encoding
    # -1 → 0b00, 0 → 0b01, 1 → 0b10
    n = len(data)
    pad = (4 - (n % 4)) % 4
    if pad: data = np.append(data, np.zeros(pad, dtype=np.int8))
    d = data.reshape(-1, 4)
    packed = np.zeros(len(d), dtype=np.uint8)
    for i in range(4):
        # Map: -1→0, 0→1, 1→2
        vals = d[:, i].copy()
        vals[vals == -1] = 0
        vals[vals == 0] = 1
        vals[vals == 1] = 2
        packed |= (vals.astype(np.uint8) & 3) << (i * 2)
    return packed, n

# Build tensor list same as extract_real
tensor_names = []
# Embeddings
tensor_names.append(('token_embd.weight', (d_model, 256), 'Q6_K'))

for L in range(n_layers):
    prefix = f'blk.{L}.'
    for key in ['attn_norm.weight', 'attn_q.weight', 'attn_k.weight',
                'attn_v.weight', 'attn_output.weight',
                'post_attention_norm.weight', 'ffn_norm.weight',
                'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight',
                'post_ffw_norm.weight']:
        name = prefix + key
        if 'norm' in key:
            shape = (d_model,)
            dtype = 'F32'
        elif 'ffn_gate' in key or 'ffn_up' in key:
            shape = (d_model, ffn_dim)
            dtype = 'Q4_K'
        elif 'ffn_down' in key:
            shape = (ffn_dim, d_model)
            dtype = 'Q4_K'
        elif 'attn_q' in key or 'attn_output' in key:
            shape = (d_model, d_model)
            dtype = 'Q4_K'
        elif 'attn_k' in key or 'attn_v' in key:
            shape = (d_model, d_model // 2)
            dtype = 'Q4_K'
        else:
            continue
        tensor_names.append((name, shape, dtype))

tensor_names.append(('output_norm.weight', (d_model,), 'F32'))

# Write
f = open(OUT, 'wb')
f.write(b'BQSM')
f.write(struct.pack('<I', 3))
f.write(struct.pack('<6I', d_model, ffn_dim, n_layers, n_heads, n_kv_heads, 256))
f.write(struct.pack('<I', len(tensor_names)))

tp = 0
for name, shape, dtype in tensor_names:
    data = gen_weights(shape, dtype)
    pkb, on = pack_2bit(data)
    nb = name.encode()
    f.write(struct.pack('<H', len(nb)))
    f.write(nb)
    f.write(struct.pack('<B', len(shape)))
    for d in shape: f.write(struct.pack('<Q', d))
    f.write(struct.pack('<I', on))
    f.write(struct.pack('<I', len(pkb)))
    f.write(pkb.tobytes())
    tp += len(pkb)

print(f"Wrote {OUT}: {os.path.getsize(OUT)/2**20:.1f} MB, {len(tensor_names)} tensors, {tp/2**20:.1f} MB packed")
print("Ternary weights: -1/0/1, 2 bits each — ready for signed-path kernel")
