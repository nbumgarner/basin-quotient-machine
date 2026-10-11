# BQSM fork research — handoff to a code session

## Setup in the repo

Copy these files into `basin-quotient-machine/research/forks/`:

- `fork_logic.c` — Fork A on the real fabric. It does `#define main rf_main` and then `#include "ring_furnace.c"`, so it compiles from a directory where `ring_furnace.c` is reachable. Either place it next to `ring_furnace.c` or change the include path to `../../ring_furnace.c`.
- `kring.c` — the independent reference simulator (random coupling fabric plus the WS integrability test). It is not BQSM; keep it only as a counterfactual and baseline.
- `RESULTS.md` (run 02), `../bqsm-fork-run-01/RESULTS.md` (run 01), and the CSVs.

## Reproduce

```bash
gcc -O3 -std=c11 ring_furnace.c -o ring_furnace -lm && ./ring_furnace selftest  # must be ALL PASS
gcc -O3 -std=c11 fork_logic.c -o fork_logic -lm
./fork_logic                                  # coverage per lens delta -> fork_logic.csv
ROBUST=32 RSIG=0.15 ./fork_logic              # robust-only coverage -> fork_logic_robust.csv
./fork_logic spot 0.50 400                    # Markov check: physical 3-step runs vs table composition
NOISE=0.15 ROBUST=32 RSIG=0.15 ./fork_logic spot 0.50 400   # noisy check on robust entries
```

## Definitions used

- **Register:** the winding number q. 7 states pristine; 5 under a lens with Δ ≥ 0.20, where ±3 die.
- **Instructions (65):**
  - NOP.
  - ERASE: the furnace's 3-site kick, at 16 sites × amplitudes {1.0, 2.2}.
  - WRITE: a localized 2π linear ramp over `RAMP_W = 8` sites starting at site s, at 16 sites × sign ±. WRITE transitions are *measured* by settling on the furnace. The ramp shape is Claude's choice; swap in Nick's own WRITE if it differs.
- **Program:** a lens (site 0, Δ), a start state q0, and for each input bit k a pair of instructions (one for bit = 0, one for bit = 1).
- **Output:** a predicate on the final q. Reported three ways: any subset of states, q≠0, and q>0.
- **Robust entry:** a (q, instruction) transition that is correct in 32/32 trials under N(0, 0.15) phase noise on all 16 sites.

## Findings so far

1. The shipped kick alphabet is ERASE-only, which makes the ring a one-way funnel (118/256 functions with any readout). WRITE is required.
2. With ERASE + WRITE, coverage is 256/256 with any readout. 800/800 physical 3-step runs match table composition.
3. Using only robust entries: 248/256 functions, 399/400 noisy runs correct, and READ(q≠0) coverage of 78 → 186 as Δ goes 0 → 0.65. The lens is the programming knob.
4. The 8 functions a single ring cannot compute robustly are the 2:1 MUX with the selector arriving last (0x35 0x3a 0x53 0x5c 0xa3 0xac 0xc5 0xca). That is a memory-width limit: one ring is one register.
5. Fork B (reference sim only): WS cross-ratios are conserved to 1e-12 under global coupling and break at any ε > 0, but twisted-state multistability appears only at ε ≈ 0.85–0.9. "Multistability starts where integrability ends" is not supported. Next step is an analytic linear-stability calculation for ε*, not more simulation.

## Next tasks, in priority order

1. **REF_BUILD re-run:** repeat findings 2–4 with `-DREF_BUILD` (DT=0.02, libm sin) before anything is published.
2. **A3, multi-lens programs:** extend `build_table` usage to lenses at 2–3 sites (the furnace currently supports a single lens). Question: can lenses alone close the late-selector MUX gap on one ring?
3. **A2, two-ring register file:** two N=16 rings with a weak inter-ring coupling or a conditional-kick rule. Target: the late-selector MUX robustly.
4. **C, noise curves:** per-3-step-word accuracy vs σ ∈ {0.05 … 0.6} for robust-only programs, pristine vs lensed. This is the head-to-head number against nonlinear spectral computing (arXiv 2610.09081), which is stateless.
5. **Fleet:** shard the Δ grid (0.00–0.70, step 0.01) across the T7400 cores and the Oracle ARM nodes (shard ids 0–15 / 100+). Diff x86_64 vs aarch64 tables (FORK[5]).
6. **Fork B, analytic:** linear stability of twisted state q under K = (1−ε)·global + ε·ring; predict ε* and compare with 0.85–0.9.

## Context links

- Nonlinear Spectral Computing, Falsi/Coppini/Conti, arXiv 2610.09081 (6 Oct 2026). All 256 3-input functions as stateless soliton-spectrum logic.
- Zhang & Strogatz, Basins with Tentacles, PRL 127, 194101 (2021). Already in the octopus battery.
