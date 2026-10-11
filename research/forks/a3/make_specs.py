#!/usr/bin/env python3
"""Regenerate the lens-configuration lists the A3 searches ran over.

Each list is split into shards (specs_<k>.txt) so the run_*.sh scripts can
evaluate them in parallel. A lens spec is "site:delta,site:delta,...".

  grid2  two lenses, 0.1 grid, lens 1 at site 0, lens 2 at every offset 1..15  (2,940)
  fine   two lenses, 0.05 grid in [-0.5, 0.5], offsets 4..8                    (2,200)
  rand3  three lenses, seed 20261010: 12,000 uniform + 12,000 near zero-sum,
         deduplicated up to rotation of the ring                              (24,000)

usage: make_specs.py grid2|fine|rand3 [outdir]
"""
import os, random, sys


def grid2():
    D = [round(x * 0.1, 1) for x in range(-7, 8) if x != 0]
    return [f"0:{a:.1f},{d}:{b:.1f}" for d in range(1, 16) for a in D for b in D], 6


def fine():
    D = [round(x * 0.05, 2) for x in range(-10, 11)]
    specs = []
    for d in range(4, 9):
        for a in D:
            for b in D:
                if a == 0 and b == 0:
                    continue
                parts = []
                if a:
                    parts.append(f"0:{a:.2f}")
                if b:
                    parts.append(f"{d}:{b:.2f}")
                specs.append(",".join(parts))
    return specs, 6


def rand3():
    random.seed(20261010)
    D = [round(x * 0.05, 2) for x in range(-14, 15) if x != 0]
    seen, specs = set(), []

    def canon(cfg):                       # rotation-canonical form of a lens set
        best = None
        for r in range(16):
            t = tuple(sorted(((s - r) % 16, d) for s, d in cfg))
            if best is None or t < best:
                best = t
        return best

    def add(cfg):
        c = canon(cfg)
        if c in seen:
            return
        seen.add(c)
        specs.append(",".join(f"{s}:{d:.2f}" for s, d in c))

    while len(specs) < 12000:             # uniform three-lens
        d1, d2 = sorted(random.sample(range(1, 16), 2))
        add([(0, random.choice(D)), (d1, random.choice(D)), (d2, random.choice(D))])
    while len(specs) < 24000:             # near zero-sum: every state can stay alive
        d1, d2 = sorted(random.sample(range(1, 16), 2))
        a, b = random.choice(D), random.choice(D)
        c = round(-(a + b) + random.choice([-0.1, -0.05, 0.05, 0.1, 0.0]), 2)
        if c == 0 or abs(c) > 0.7:
            continue
        add([(0, a), (d1, b), (d2, c)])
    return specs, 7


if __name__ == "__main__":
    which = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), which)
    specs, K = {"grid2": grid2, "fine": fine, "rand3": rand3}[which]()
    os.makedirs(out, exist_ok=True)
    for k in range(K):
        with open(os.path.join(out, f"specs_{k}.txt"), "w") as f:
            f.write("\n".join(specs[k::K]) + "\n")
    print(f"{which}: {len(specs)} specs in {K} shards -> {out}")
