#!/usr/bin/env python3
"""Pure decode speed of the engine, the counterpart of llama-bench's tg test: per token =
model step (append) + bf16 readout + argmax, nothing else (no memory bias, no sampling
extras, no server). Prompt processing is excluded, as in llama-bench -p 0.
usage: pure_decode.py QB [n_tokens] [runs]     (QB 0 = exact float path, 16 = live int8 kernel)"""
import os, sys, time
# Run from a bqsm-neuron checkout with q8_hook.patch applied and libint8q8.so built there
# (see README.md). BQSM_MODEL / BQSM_BLOB select the model exactly as for the engine.
sys.path.insert(0, os.getcwd())
import numpy as np, bqsm_int8 as E, bqsm_full_settle as FS
from bqsm_llama import Safetensors, BASE
from bqsm_tokenizer import Tokenizer
qb = int(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 48; R = int(sys.argv[3]) if len(sys.argv) > 3 else 3
st = Safetensors(BASE); tok = Tokenizer(BASE); wnorm = st.get("model.norm.weight")
eraw, esh = E.bf16_view(st, "model.embed_tokens.weight")
hraw, hsh = E.bf16_view(st, "model.embed_tokens.weight" if FS.CFG.get("tie_word_embeddings") else "lm_head.weight")
emb = lambda t: E.bf16_row(eraw, esh, t).reshape(1, -1)
eng = E.Engine(FS.make_invf()); eng.q8 = qb
ids = [128000] + tok.encode("Explain, step by step, how a mechanical clock keeps time, from the spring to the hands.")
for r in range(R + 1):                                   # run 0 = warm-up, not reported
    eng.kv = [None] * E.NL; eng.ids = []
    z = eng.settle_batch(np.concatenate([emb(t) for t in ids]), 0, wnorm); eng.ids = list(ids)
    out = []; t0 = time.perf_counter()
    for _ in range(N):
        nx = int(np.argmax(E.bf16_logits(hraw, hsh, z[-1]))); out.append(nx)
        z = eng.append(nx, emb, wnorm)
    dt = (time.perf_counter() - t0) / N
    if r:
        print(f"QB={qb} run {r}: {1000 * dt:.1f} ms/token = {1 / dt:.2f} tok/s  ({N} tokens, ids hash {hash(tuple(out)) & 0xffffff:06x})", flush=True)
