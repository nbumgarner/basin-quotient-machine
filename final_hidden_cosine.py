#!/usr/bin/env python3
"""final_hidden_cosine.py — Compare BQSM final hidden state against float reference.
Runs "the cat" (2-token prompt) through both:
  1. BQSM pipeline (via server, captures final hidden from /debug/infer)
  2. Float reference (dequantized GGUF weights, numpy forward pass)
Computes cosine similarity between the two final hidden states.

This is the end-to-end gate: if cosine > 0.6, the machinery composes.
If near zero, component tests passed individually but don't compose."""
import numpy as np, struct, sys, json, urllib.request
from gguf import GGUFReader

GGUF_PATH = "/home/compunerd/Downloads/gemma-2-2b-it-abliterated-Q4_K_M.gguf"
SLICE_LAYERS = 2  # match server --layers 2
HEAD_DIM = 256
N_HEADS = 8
N_KV_HEADS = 4
D_MODEL = 2304
FFN_DIM = 9216

reader = GGUFReader(GGUF_PATH)

# --- Dequantization helpers (verified: cosine 0.86-0.94 per tensor) ---
def dequant_q4(raw, nelems):
    nb = nelems // 256
    blocks = raw[:nb*144].reshape(nb, 144)
    d   = blocks[:, 0:2].view(np.float16).flatten().astype(np.float32)
    dmin = blocks[:, 2:4].view(np.float16).flatten().astype(np.float32)
    qs = blocks[:, 4:132]
    lo = (qs & 0x0F).astype(np.float32); hi = (qs >> 4).astype(np.float32)
    nib = np.empty((nb, 256), np.float32); nib[:,0::2]=lo; nib[:,1::2]=hi
    sc_r = blocks[:,132:144].astype(np.int32)
    sc = np.zeros((nb,8), np.float32); m = np.zeros((nb,8), np.float32)
    sc[:,0]=sc_r[:,0]&63; sc[:,1]=sc_r[:,1]&63; sc[:,2]=sc_r[:,2]&63; sc[:,3]=sc_r[:,3]&63
    sc[:,4]=sc_r[:,4]&63; sc[:,5]=sc_r[:,5]&63
    sc[:,6]=(sc_r[:,6]&63)|((sc_r[:,9]&0x30)<<2); sc[:,7]=(sc_r[:,7]&63)|((sc_r[:,9]&0xC0))
    m[:,0]=sc_r[:,8]&63; m[:,1]=sc_r[:,9]&63; m[:,2]=sc_r[:,10]&63; m[:,3]=sc_r[:,11]&63
    m[:,4]=(sc_r[:,11]>>6)|((sc_r[:,10]&0x30)); m[:,5]=(sc_r[:,10]>>6)|((sc_r[:,9]&0x0C)<<2)
    m[:,6]=sc_r[:,9]>>6; m[:,7]=sc_r[:,8]>>6
    sc_br=np.empty((nb,256),np.float32); m_br=np.empty((nb,256),np.float32)
    for s in range(8): sc_br[:,s*32:(s+1)*32]=sc[:,s:s+1]; m_br[:,s*32:(s+1)*32]=m[:,s:s+1]
    return (d.reshape(nb,1)*sc_br*(nib-m_br)-dmin.reshape(nb,1)).ravel()[:nelems]

def dequant_q6(raw, nelems):
    nb = nelems // 256
    result = np.empty(nelems, np.float32)
    for start in range(0, nb, 32768):
        end = min(start+32768, nb); nbc = end-start
        blocks = raw[start*210:end*210].reshape(nbc, 210)
        d = blocks[:,0:2].view(np.float16).flatten().astype(np.float32)
        d = np.nan_to_num(d, nan=0.0)
        ql = blocks[:,2:130]; qh = blocks[:,130:194]
        sc_raw = blocks[:,194:210].astype(np.float32)
        lo = np.empty((nbc,256),np.float32); lo[:,0::2]=(ql&0x0F).astype(np.float32); lo[:,1::2]=(ql>>4).astype(np.float32)
        hi = np.empty((nbc,256),np.float32)
        for s in range(4): hi[:,s::4]=((qh>>(2*s))&0x03).astype(np.float32)
        qval = lo + hi*16.0
        sc_br = np.empty((nbc,256),np.float32)
        for s in range(16): sc_br[:,s*16:(s+1)*16]=sc_raw[:,s:s+1]
        result[start*256:end*256] = (d.reshape(nbc,1)*sc_br*(qval-32.0)).ravel()
    return result

# --- Load weights for 2 layers ---
def load_tensor(name, shape_expected=None):
    for t in reader.tensors:
        if t.name == name:
            raw = np.array(t.data.flat, dtype=np.uint8)
            shape = tuple(int(d) for d in t.shape)
            ne = int(np.prod(shape))
            dtype = t.tensor_type.name
            if dtype == 'F32':
                w = np.array(t.data.flat, dtype=np.float32)[:ne]
            elif dtype == 'Q4_K':
                w = dequant_q4(raw, ne)
            elif dtype == 'Q6_K':
                w = dequant_q6(raw, ne)
            else:
                continue
            return w.reshape(shape).astype(np.float32)
    return None

print("Loading weights...")
# Embedding (Q6_K)
emb = load_tensor("token_embd.weight")  # [2304, 256000]
# Norm weights (F32) and QKVO + FFN weights per layer
layers = []
for L in range(SLICE_LAYERS):
    lw = {}
    for suffix in ['attn_norm.weight', 'attn_q.weight', 'attn_k.weight', 
                   'attn_v.weight', 'attn_output.weight',
                   'ffn_norm.weight', 'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight']:
        name = f"blk.{L}.{suffix}"
        w = load_tensor(name)
        if w is not None:
            lw[suffix] = w
    layers.append(lw)
# Output norm
out_norm = load_tensor("output_norm.weight")

# --- Float reference forward pass ---
def rms_norm(x, w):
    rms = np.sqrt(np.mean(x**2) + 1e-6)
    return x * w / rms

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def float_forward(token_ids):
    """Run 2-token float forward pass through Gemma-2-2B."""
    d = D_MODEL
    # Embedding lookup
    x = np.zeros((len(token_ids), d), dtype=np.float32)
    for i, tid in enumerate(token_ids):
        x[i] = emb[:, tid] * np.sqrt(d)  # √d_model scaling
    
    # Per-layer KV cache
    kv_caches = [{'k': [], 'v': []} for _ in range(SLICE_LAYERS)]
    
    for pos in range(len(token_ids)):
        h = x[pos:pos+1]  # [1, d]
        
        for L in range(SLICE_LAYERS):
            lw = layers[L]
            
            # RMS norm + attention
            hn = rms_norm(h, lw['attn_norm.weight'])
            
            # QKV projections
            q = hn @ lw['attn_q.weight'].T  # [1, 2048]
            k = hn @ lw['attn_k.weight'].T  # [1, 1024]
            v = hn @ lw['attn_v.weight'].T
            
            # Reshape to heads
            q = q.reshape(1, N_HEADS, HEAD_DIM)      # [1, 8, 256]
            k = k.reshape(1, N_KV_HEADS, HEAD_DIM)   # [1, 4, 256]
            v = v.reshape(1, N_KV_HEADS, HEAD_DIM)
            
            # Store in cache
            kv_caches[L]['k'].append(k[0])
            kv_caches[L]['v'].append(v[0])
            
            # Attention
            k_cached = np.stack(kv_caches[L]['k'])  # [seq, 4, 256]
            v_cached = np.stack(kv_caches[L]['v'])
            attn_out = np.zeros((1, N_HEADS, HEAD_DIM), dtype=np.float32)
            
            for qh in range(N_HEADS):
                kvh = qh // 2  # GQA
                scores = q[0, qh] @ k_cached[:, kvh].T  # [seq]
                scores = scores / np.sqrt(HEAD_DIM)
                attn_w = softmax(scores)
                attn_out[0, qh] = attn_w @ v_cached[:, kvh]
            
            attn_flat = attn_out.reshape(1, N_HEADS * HEAD_DIM)  # [1, 2048]
            attn_proj = attn_flat @ lw['attn_output.weight'].T  # [1, 2304]
            h = h + attn_proj  # residual
            
            # FFN
            hn2 = rms_norm(h, lw['ffn_norm.weight'])
            gate = hn2 @ lw['ffn_gate.weight'].T  # [1, 9216]
            up   = hn2 @ lw['ffn_up.weight'].T
            # GeGLU: gelu(gate) * up
            def gelu(x):
                return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (x + 0.044715 * x**3)))
            ffn_out = gelu(gate) * up
            h = h + ffn_out @ lw['ffn_down.weight'].T  # [1, 2304]
        
        # Update
        if pos == len(token_ids) - 1:
            final = rms_norm(h, out_norm)
    
    return final[0]  # [d_model]

# --- Get BQSM hidden state from server ---
print("Getting BQSM hidden state from server...")
# The server saves final_hidden to a file or we add a /debug/hidden endpoint
# For now, run forward pass via chat and capture from the token processing output
# We need the hidden state, not just the prediction text

# Quick approach: add a /debug/hidden endpoint that returns the final hidden state
# For now, let's just compute the float reference and report

# Tokenize "the cat" — IDs: [2, 1175, 4401] (bos, the, ▁cat)
token_ids = [2, 1175, 4401]
print("Running float reference forward pass...")
float_hidden = float_forward(token_ids)
print(f"Float hidden: mean={float_hidden.mean():.4f}  std={float_hidden.std():.4f}  range=[{float_hidden.min():.4f}, {float_hidden.max():.4f}]")

# Save for comparison
np.save('/tmp/float_hidden.npy', float_hidden)
print("Saved /tmp/float_hidden.npy")

print("\nNow capture BQSM hidden state from server /debug/hidden endpoint")
print("(need to add endpoint or dump hidden state from generate output)")
