#!/usr/bin/env python3
"""Quality, not agreement: next-token negative log-likelihood (perplexity) of real
held-out text under the exact engine vs the SDOT engine, same passages; plus
prompt-processing (batched prefill) time."""
import os, sys, time
# Run from a bqsm-neuron checkout with q8_hook.patch applied and libint8q8.so built there
# (see README.md). BQSM_MODEL / BQSM_BLOB select the model exactly as for the engine.
sys.path.insert(0, os.getcwd())
CORPUS = os.environ.get("BQSM_CORPUS", "wikitext103.txt:gutenberg.txt").split(":")
import numpy as np, bqsm_int8 as E, bqsm_full_settle as FS
from bqsm_llama import Safetensors, BASE
from bqsm_tokenizer import Tokenizer
st = Safetensors(BASE); tok = Tokenizer(BASE); wnorm = st.get("model.norm.weight")
eraw, esh = E.bf16_view(st, "model.embed_tokens.weight")
hraw, hsh = E.bf16_view(st, "model.embed_tokens.weight" if FS.CFG.get("tie_word_embeddings") else "lm_head.weight")
emb = lambda t: E.bf16_row(eraw, esh, t).reshape(1, -1)
ref = E.Engine(FS.make_invf()); ref.q8 = 0
alt = E.Engine(FS.make_invf(), blob=ref.blob)
P = []
for path in CORPUS:
    with open(path, errors="ignore") as f:
        f.seek(0, 2); size = f.tell()
        for k in range(8):
            f.seek(int(size * (0.05 + 0.115 * k))); f.readline()
            P.append([128000] + tok.encode(f.read(5000))[:128])
def run(eng, ids):
    eng.kv = [None] * E.NL
    t = time.perf_counter(); z = eng.settle_batch(np.concatenate([emb(x) for x in ids]), 0, wnorm); tp = time.perf_counter() - t
    nll = 0.0
    for i in range(len(ids) - 1):
        lg = np.asarray(E.bf16_logits(hraw, hsh, z[i]), np.float64); m = lg.max()
        nll += (m + np.log(np.exp(lg - m).sum())) - lg[ids[i + 1]]
    return nll, len(ids) - 1, tp
base = [run(ref, p) for p in P]
N = sum(b[1] for b in base); nr = sum(b[0] for b in base)
print(f"exact: {N} tokens, mean NLL {nr / N:.4f} nats, perplexity {np.exp(nr / N):.3f}, prefill {np.mean([b[2] for b in base]):.2f} s per 129-token passage", flush=True)
for qb in [int(a) for a in sys.argv[1:]]:
    alt.q8 = qb
    r = [run(alt, p) for p in P]; na = sum(x[0] for x in r)
    print(f"SDOT QB={qb}: mean NLL {na / N:.4f} nats, perplexity {np.exp(na / N):.3f} (delta {100 * (np.exp(na / N) / np.exp(nr / N) - 1):+.2f}%), "
          f"prefill {np.mean([x[2] for x in r]):.2f} s", flush=True)
