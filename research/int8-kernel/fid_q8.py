#!/usr/bin/env python3
"""Fidelity + speed of the SDOT int8-activation kernel against the exact engine.

Two Engine objects share one weight blob: ref (q8 = 0, the live numerics) and q8
(activation block size QB). Measured, per QB:
  * teacher-forced on corpus passages: per position, top-1 agreement, mean
    KL(p_ref || p_q8), mean logit cosine, ref top-1 inside q8 top-5;
  * greedy generations from 6 prompts: tokens identical before first divergence;
  * decode step time (append) for both engines.
Pure model: no memory bias / sampling extras (those sit outside the matrix kernels).
usage: fid_q8.py QB [QB ...]      (QB = 16, 32, 64, ... or -1 for one scale per vector)"""
import os, sys, time
# Run from a bqsm-neuron checkout with q8_hook.patch applied and libint8q8.so built there
# (see README.md). BQSM_MODEL / BQSM_BLOB select the model exactly as for the engine.
sys.path.insert(0, os.getcwd())
CORPUS = os.environ.get("BQSM_CORPUS", "wikitext103.txt:gutenberg.txt").split(":")
import numpy as np
import bqsm_int8 as E
import bqsm_full_settle as FS
from bqsm_llama import Safetensors, BASE
from bqsm_tokenizer import Tokenizer

assert E._lq8 is not None, "libint8q8.so not loaded"
st = Safetensors(BASE); tok = Tokenizer(BASE)
wnorm = st.get("model.norm.weight")
eraw, esh = E.bf16_view(st, "model.embed_tokens.weight")
ename = "model.embed_tokens.weight" if FS.CFG.get("tie_word_embeddings") else "lm_head.weight"
hraw, hsh = E.bf16_view(st, ename)
def emb(t):
    r = E.bf16_row(eraw, esh, t); return r.reshape(1, -1)
def logits(z): return np.asarray(E.bf16_logits(hraw, hsh, z), np.float64)
ref = E.Engine(FS.make_invf()); ref.q8 = 0
q8e = E.Engine(FS.make_invf(), blob=ref.blob)

passages = []
for path in CORPUS:
    with open(path, errors="ignore") as f:
        f.seek(0, 2); size = f.tell()
        for k in range(4):
            f.seek(int(size * (0.13 + 0.21 * k))); f.readline()
            ids = tok.encode(f.read(4000))[:96]
            passages.append([128000] + ids)
prompts = ["Explain how a mechanical clock keeps time.", "Write a short note to a friend about learning to cook.",
           "What causes the seasons on Earth?", "List three uses for a magnet and explain one.",
           "Describe a quiet morning in a small town.", "How does a bicycle stay upright while moving?"]

def teacher(eng, ids):
    eng.kv = [None] * E.NL
    X = np.concatenate([emb(t) for t in ids]); z = eng.settle_batch(X, 0, wnorm)
    return [logits(z[i]) for i in range(len(ids) - 1)]

def greedy(eng, ids, n=48):
    eng.kv = [None] * E.NL; eng.ids = []
    z = eng.settle_batch(np.concatenate([emb(t) for t in ids]), 0, wnorm); eng.ids = list(ids)
    out, ts = [], []
    for _ in range(n):
        nx = int(np.argmax(logits(z[-1]))); out.append(nx)
        t = time.perf_counter(); z = eng.append(nx, emb, wnorm); ts.append(time.perf_counter() - t)
    return out, float(np.median(ts))

ref_tf = [teacher(ref, p) for p in passages]
ref_gen = [greedy(ref, [128000] + tok.encode(p)) for p in prompts]
print(f"exact engine: decode step median {1000 * np.median([g[1] for g in ref_gen]):.1f} ms", flush=True)
for qb in [int(a) for a in sys.argv[1:]]:
    q8e.q8 = qb
    agree = kl = cos = top5 = n = 0
    for p, lr in zip(passages, ref_tf):
        lq = teacher(q8e, p)
        for a, b in zip(lr, lq):
            pa = np.exp(a - a.max()); pa /= pa.sum(); pb = np.exp(b - b.max()); pb /= pb.sum()
            agree += int(np.argmax(a) == np.argmax(b)); top5 += int(np.argmax(a) in np.argpartition(b, -5)[-5:])
            kl += float(np.sum(pa * (np.log(pa + 1e-30) - np.log(pb + 1e-30))))
            cos += float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))); n += 1
    same, tot, tq = [], 0, []
    for p, (go, _) in zip(prompts, ref_gen):
        gq, t = greedy(q8e, [128000] + tok.encode(p)); tq.append(t)
        k = next((i for i, (u, v) in enumerate(zip(go, gq)) if u != v), len(go)); same.append(k)
        tot += sum(u == v for u, v in zip(go, gq))
    print(f"QB={qb:>5}: teacher-forced {n} positions: top-1 agree {100 * agree / n:5.1f}%  ref-top1-in-q8-top5 {100 * top5 / n:5.1f}%  "
          f"KL {kl / n:.4f} nats  logit cos {cos / n:.5f} | greedy: identical prefix {same} of 48 | "
          f"decode step median {1000 * np.median(tq):.1f} ms", flush=True)
