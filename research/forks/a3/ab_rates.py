#!/usr/bin/env python3
"""Best per-instruction noisy pass rates (R=32, sigma 0.15) for the two ERASE
behaviours the 5-state late-selector MUX needs (see mux_minimal.py):
  A: +1 -> 0 AND -1 -> -1 in the same instruction (sign-selective decrement)
  B: -2 -> 0 AND -1 -> 0 in the same instruction (double collapse)
Score of an instruction = min of its two entry pass counts; report the max."""
import os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); BIN = os.path.join(HERE, "..", "fork_lens")
pat = re.compile(r"q=([+-]\d) (\S+)\s+-> ([+-]\d)\s+pass=(\d+)/(\d+)")
for spec in sys.argv[1:]:
    out = subprocess.run([BIN, "table", spec], env=dict(os.environ, ROBUST="32"), capture_output=True, text=True).stdout
    t = {}
    for q, ins, to, p, r in pat.findall(out): t[(int(q), ins)] = (int(to), int(p))
    ins = {i for _, i in t}
    def sc(i, a, b):
        x, y = t.get((a[0], i)), t.get((b[0], i))
        if not x or not y or x[0] != a[1] or y[0] != b[1]: return -1
        return min(x[1], y[1])
    A = max(((sc(i, (1, 0), (-1, -1)), i) for i in ins), default=(-1, ""))
    B = max(((sc(i, (-2, 0), (-1, 0)), i) for i in ins), default=(-1, ""))
    print(f"{spec or 'pristine':34s} A={A[0]:>3}/32 {A[1]:24s} B={B[0]:>3}/32 {B[1]}")
