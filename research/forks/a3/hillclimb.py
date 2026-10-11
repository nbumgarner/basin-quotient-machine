#!/usr/bin/env python3
"""A3 stage 3: hill-climb a lens configuration on the MUX robustness level.

Objective (lexicographic, all measured by fork_lens on the furnace):
  1. mux_k  — largest k such that the late-selector MUX is computable using only
              table entries that passed >= k of R noisy trials (R = 32, sigma 0.15)
  2. any    — robust (k = R) any-readout coverage, as a tie-break
Neighbourhood: one lens delta +/- HC_STEP (env, default 0.05), or one lens site +/- 1 (site 0 fixed
as the rotation anchor), or one delta removed/added at a free site.
Every evaluation is a fresh `fork_lens run` (deterministic per config: the
robust-filter noise seed is derived from the omega profile).

usage: hillclimb.py "<spec>" [max_steps] [workers] [extra env as K=V ...]
"""
import os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "..", "fork_lens")
LINE = re.compile(r"alive=(\d+) robust=(\d+)/(\d+) maps=(\d+) any=(\d+) nz=(\d+) pos=(\d+)\s+"
                  r"MUX8: any=(\d+) nz=(\d+) pos=(\d+)\s+mux_k=(-?\d+) dec10=(-?\d+)")
CACHE = {}

STEP = float(os.environ.get("HC_STEP", "0.05"))   # delta nudge; lens grid resolution

def parse(spec):
    return tuple(sorted((int(s), round(float(d), 3)) for s, d in (t.split(":") for t in spec.split(",") if t)))

def fmt(cfg):
    return ",".join(f"{s}:{d:.3f}" for s, d in cfg)

def canon(cfg):
    best = None
    for r in range(16):
        t = tuple(sorted(((s - r) % 16, d) for s, d in cfg))
        if best is None or t < best: best = t
    return best

def evaluate(cfg, env_extra):
    key = canon(cfg)
    if key in CACHE: return CACHE[key]
    env = dict(os.environ, ROBUST="32", **env_extra)
    out = subprocess.run([BIN, "run", fmt(key)], env=env, capture_output=True, text=True).stdout
    m = None
    for ln in out.splitlines():
        m = LINE.search(ln) or m
    if not m: raise RuntimeError(out)
    v = dict(zip("alive kept total maps any nz pos mux_any mux_nz mux_pos mux_k dec10".split(), map(int, m.groups())))
    CACHE[key] = v
    return v

def score(v):
    return (v["mux_k"], v["any"])

def neighbours(cfg):
    sites = {s for s, _ in cfg}
    out = set()
    for i, (s, d) in enumerate(cfg):
        for dd in (-STEP, STEP):                          # nudge a delta
            nd = round(d + dd, 3)
            if abs(nd) <= 0.8 + 1e-9:
                c = list(cfg); c[i] = (s, nd)
                c = [x for x in c if x[1] != 0.0]
                if c: out.add(canon(c))
        if s != 0:                                        # slide a lens
            for ds in (-1, 1):
                ns = (s + ds) % 16
                if ns not in sites:
                    c = list(cfg); c[i] = (ns, d); out.add(canon(c))
    return [c for c in out if canon(c) != canon(cfg)]

def main():
    start = canon(parse(sys.argv[1]))
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    env_extra = dict(kv.split("=", 1) for kv in sys.argv[4:])
    cur, cv = start, evaluate(start, env_extra)
    print(f"start {fmt(cur)} mux_k={cv['mux_k']} any={cv['any']} alive={cv['alive']} dec10={cv['dec10']}", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        for step in range(steps):
            nb = neighbours(cur)
            vals = list(ex.map(lambda c: evaluate(c, env_extra), nb))
            best = max(zip(nb, vals), key=lambda t: score(t[1]))
            if score(best[1]) <= score(cv):
                print(f"step {step}: local optimum ({len(nb)} neighbours, best {score(best[1])})", flush=True)
                break
            cur, cv = best
            print(f"step {step}: {fmt(cur)} mux_k={cv['mux_k']} any={cv['any']} alive={cv['alive']} dec10={cv['dec10']}", flush=True)
            if cv["mux_k"] >= 32:
                print("reached mux_k = 32 (robust MUX)", flush=True); break
    print(f"final {fmt(cur)} {cv}")

if __name__ == "__main__":
    main()
