#!/usr/bin/env python3
"""Abstract check: which 3-input functions can ANY alphabet of 'physical-class'
maps compute in the run-02 program model (start state, per-bit instruction
pairs, any-readout)?  Map classes on states {-Q..Q}:
  NOP        identity
  WRITE+     q -> q+1 (q = +Q -> dead)
  WRITE-     q -> q-1 (q = -Q -> dead)          [only if --neg]
  ERASE      every map f with sign(f(q)) in {sign(q), 0}, |f(q)| <= |q|  (a
             contraction toward 0 — the furnace's measured ERASE class), split
             into monotone (f non-decreasing) and non-monotone members.
Everything is taken at once (the most generous alphabet of that class), so a
missing function here is missing for EVERY lens that keeps maps in the class.
usage: mux_obstruction.py Q [--neg] [--nonmono]
"""
import itertools, sys
Q = int(sys.argv[1]); NEG = "--neg" in sys.argv; NONMONO = "--nonmono" in sys.argv
S = list(range(-Q, Q + 1)); idx = {q: i for i, q in enumerate(S)}
maps = {tuple(S)}                                        # NOP
maps.add(tuple(q + 1 if q < Q else None for q in S))   # WRITE+
if NEG: maps.add(tuple(q - 1 if q > -Q else None for q in S))
choices = [[v for v in S if (v == 0 or (v > 0) == (q > 0)) and abs(v) <= abs(q)] for q in S]
ne = 0
for f in itertools.product(*choices):
    mono = all(f[i] <= f[i + 1] for i in range(len(S) - 1))
    if mono or NONMONO: maps.add(f); ne += 1
maps = [tuple(None if v is None else idx[v] for v in m) for m in maps]
level = {(i,) for i in range(len(S))}                    # constant maps over 1 input word
for m in range(3):
    nxt = set()
    for code in level:
        for a in maps:
            if any(a[s] is None for s in code): continue
            for b in maps:
                if any(b[s] is None for s in code): continue
                nxt.add(tuple(a[s] for s in code) + tuple(b[s] for s in code))
    level = nxt
full = set()
for code in level:
    used = sorted(set(code))
    for r in range(1 << len(used)):
        R = {used[k] for k in range(len(used)) if r >> k & 1}
        full.add(sum(1 << x for x in range(8) if code[x] in R))
MUX8 = [0x35,0x3a,0x53,0x5c,0xa3,0xac,0xc5,0xca]
print(f"Q={Q} states={len(S)} neg={NEG} nonmono={NONMONO} erase_maps={ne} alphabet={len(maps)} "
      f"final_maps={len(level)} coverage={len(full)}/256 MUX8={sum(f in full for f in MUX8)}/8")
miss = [f for f in range(256) if f not in full]
if miss: print("  missing:", " ".join(f"{f:02x}" for f in miss))
