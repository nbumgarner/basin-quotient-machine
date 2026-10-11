#!/usr/bin/env python3
"""Which ERASE-class maps (monotone sign-preserving contractions) does the
late-selector MUX need, given NOP + WRITE+ on states {-Q..Q}?  Tries every
single map, then every pair, and reports the ones that give MUX8 = 8/8.
usage: mux_minimal.py Q [alive list e.g. -2,-1,0,1,2]"""
import itertools, sys
Q = int(sys.argv[1])
S = list(range(-Q, Q + 1)) if len(sys.argv) < 3 else [int(t) for t in sys.argv[2].split(",")]
idx = {q: i for i, q in enumerate(S)}
NOP = tuple(range(len(S)))
WP = tuple(idx[q + 1] if q + 1 in idx else None for q in S)
choices = [[v for v in S if (v == 0 or (v > 0) == (q > 0)) and abs(v) <= abs(q)] for q in S]
ER = [f for f in itertools.product(*choices) if all(f[i] <= f[i+1] for i in range(len(S)-1))]
ER = [tuple(idx[v] for v in f) for f in ER if tuple(idx[v] for v in f) != NOP]
MUX8 = {0x35,0x3a,0x53,0x5c,0xa3,0xac,0xc5,0xca}
def mux_count(maps):
    level = {(i,) for i in range(len(S))}
    for m in range(3):
        nxt = set()
        for code in level:
            ok = [a for a in maps if all(a[s] is not None for s in code)]
            for a in ok:
                for b in ok: nxt.add(tuple(a[s] for s in code) + tuple(b[s] for s in code))
        level = nxt
    full = set()
    for code in level:
        used = sorted(set(code))
        for r in range(1 << len(used)):
            R = {used[k] for k in range(len(used)) if r >> k & 1}
            full.add(sum(1 << x for x in range(8) if code[x] in R))
    return len(MUX8 & full), len(full)
show = lambda f: "{" + ", ".join(f"{q:+d}->{S[f[i]]:+d}" for i, q in enumerate(S) if S[f[i]] != q) + "}"
print(f"states {S}: {len(ER)} non-identity monotone contractions")
print("NOP+WRITE+ alone:", mux_count([NOP, WP]))
singles = [f for f in ER if mux_count([NOP, WP, f])[0] == 8]
print(f"single ERASE maps that give MUX8 (with NOP, WRITE+): {len(singles)}")
for f in singles: print("  ", show(f))
if not singles:
    pairs = [(f, g) for f, g in itertools.combinations(ER, 2) if mux_count([NOP, WP, f, g])[0] == 8]
    print(f"pairs that give MUX8: {len(pairs)}")
    for f, g in pairs[:40]: print("  ", show(f), " + ", show(g))
