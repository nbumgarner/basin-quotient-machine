# BQSM fork run 03: REF re-run, Nick's WRITE, A3 multi-lens (2026-10-10)

**Machine.** Laptop, Intel Core 3 100U (8 threads), gcc 15.3.0, `-O3 -std=c11`.
**Engine.** `../../ring_furnace.c`, the working tree on branch `bqsm-quantum` @ 5286cbe. Its dynamics match 60890cd (origin/main, the build run 02 measured). The diff since then adds only the vQPU functions and removes `static` from five helpers. `selftest: ALL PASS` passes on this machine for **both** files in **both** builds (fast 0.03 s, REF 3.3 s).

## 0. Setup and fast-build reproduction

- `fork_logic.c` is the handoff file, verbatim except for its include path (`../../ring_furnace.c`).
- With the fast build on this machine, `fork_logic.csv` and `fork_logic_robust.csv` are **byte-identical** to the handoff CSVs (`handoff_csv/`).
- Spot checks:
  - Noiseless 3-step words: 399/400 at Δ=0 and 400/400 at Δ=0.50 (run 02 reported 800/800).
  - Noisy, raw table: 82.0% at Δ=0 and 93.25% at Δ=0.50.
  - Noisy, robust-only table: 399/400 at both Δ.

## 1. Task 1: the REF_BUILD re-run (DT=0.02, libm sin, no early exit)

The code is the same `fork_logic.c` built with `-DREF_BUILD` (`run_ref.sh`, outputs in `out_ref/`).

| Δ | fast: any / q≠0 / q>0 (maps) | REF: any / q≠0 / q>0 (maps) |
|---|---|---|
| 0.00 | 256 / 86 / **118** (3093) | 256 / 86 / **104** (1857) |
| 0.05–0.65 | identical | identical |

| Δ | fast robust: any / q≠0 / q>0 | REF robust: any / q≠0 / q>0 |
|---|---|---|
| 0.00 | 240 / 78 / 96 | 240 / 78 / 96 |
| 0.15–0.65 | 248 / 86–186 / 104–112 | **identical at every Δ** |

| 400 random 3-step words | fast | REF |
|---|---|---|
| noiseless, Δ=0 / 0.50 | 399 / 400 | 400 / 400 |
| σ=0.15, raw table, Δ=0 / 0.50 | 82.0% / 93.25% | 86.5% / 93.25% |
| σ=0.15, robust-only, Δ=0 / 0.50 | 399 / 399 | 398 / 399 |

**Verdict: findings 2–4 of run 02 hold under REF.**
- The robust coverage numbers are identical in both builds.
- The same 8 late-selector MUX functions are missing at every Δ ≥ 0.15.

The only table difference is 8 of 455 entries at Δ=0: **WRITE overflow from |q|=3**.
- The fast build folds these back to q=±1 at sites 0, 11, 13 and 14.
- REF says they do not land on a valid state, which matches BQSM_ARCHITECTURE ("WRITE from |q|=3 → DEAD").
- This is a DT=0.5 artifact at the stability edge. None of these entries was robust.

**Seed caveat on the robust numbers (new).** The 32/32 filter is a stochastic test, so entries with a true pass rate near 97% pass or fail by chance. I re-drew the noise 7 times (`SEED=1..7`).
- Robust any-readout coverage is stable.
- Robust READ(q≠0) at Δ=0.50 is **bimodal**: 182 in 5 of 7 draws and 94 in 2 of 7.
- Robust READ(q>0) at Δ=0 is 96 in 6 of 7 draws and 88 in 1.
- One marginal entry decides each case, so quote READ(q≠0) robust coverage with that spread.

## 2. WRITE: Nick's definition replaces the handoff ramp

Every one of Nick's instruments defines WRITE as **+1 only**:
- `bqsm_vm.c` `apply_ramp` (WRITE_INC: width N by default; RAMP_AT: any site and width): centered, clamped to ±π.
- `bqsm_asm.py`: width 6.
- `bqsm_energetics.py`: widths 4/6/8 measured.
- `basin_visual.py` `ramp_v`: the same family, start-anchored.

The handoff ramp was an 8-site start-anchored ramp **with both signs**. `fork_lens.c` copies `apply_ramp` verbatim. `WRITE_NEG=1` adds a mirrored −1 ramp, which is **not** in Nick's ISA and exists only to separate the effect of shape from the effect of sign.

| alphabet (raw, fast) | Δ=0 any | Δ=0.50 any / q≠0 / q>0 | MUX8 |
|---|---|---|---|
| handoff ramp, ± (run 02) | 256 | 256 / 234 / 112 | 8 |
| Nick's `apply_ramp` w=8, ± | 256 | 256 / 234 / 112 (identical maps) | 8 |
| Nick's `apply_ramp` w=16, ± | 256 | 256 / 226 / 112 | 8 |
| handoff ramp, + only | 224 | 232 / 142 / 104 | **0** |
| **Nick's WRITE_INC (w=16, +1)** | **184** | **232 / 142 / 104** | **0** |
| Nick's RAMP_AT w∈{16,4,6,8}, +1 | 232 | 232 / 162 / 126 | **0** |

**Ramp shape does not matter; the sign does.** Run 02's 256/256 and its raw late-selector MUX both depended on a **WRITE−1** that Nick's ISA does not have. With Nick's WRITE:
- Robust any-readout coverage peaks at 224 (Δ=0.50).
- 24 functions are missing **even before the robustness filter**: the 8 late-selector MUXes plus 16 others of the form "b2 selects between a held bit and b0⊕b1".

Single-lens table: `a3/single/`.

## 3. A3: multi-lens programs on one ring (Nick's WRITE)

**Instrument.** `fork_lens.c` generalizes the lens to an ω profile with any number of sites.
- Every reachable state map keeps a back-pointer, so each covered function prints its program.
- `verify` runs the program kick by kick on the furnace under noise.
- `xcheck` rebuilds run 02's tables through the new builder. The result is **0/455 entries different at Δ = 0, 0.20, 0.35, 0.50 and 0.65**, and identical non-robust coverage.

**A methodology bug found on the way.** The fast build's early-exit lock (gaps inside (−π+0.1, π−0.1) on two checks) is "rigorous for pristine" per the furnace comment. It is **not** rigorous under a multi-lens profile.
- Example: under `0:−0.40,1:0.55,5:0.60`, q=−2 "locks", and then slips to −1 on the next settle.
- That config showed a fake robust MUX: robustness level 32/32 in the table, but only 62.5% correct in physical runs even without noise.
- `fork_lens` now defaults to **strict settle**. A lane counts as locked only if the furnace's own velocity criterion (spread < LOCK_TOL) is met within MAXCHUNK more chunks and its winding never changes.
- Strict settle changes **0 entries** on every single-lens table and on the dipole below. It exposed all of the high-scoring 3-lens hits as drifting-state artifacts.
- A fixed 4-chunk confirmation was wrong too: it rejected the dipole's true but slowly converging q=+3.
- **Any future multi-lens work in the fast build must use strict settle or REF.**

**Search** (fast build, strict settle, R=32, σ=0.15):

| stage | configs | raw late-selector MUX | best MUX robustness level (k/32) |
|---|---|---|---|
| 1 lens, Δ ∈ ±[0, 1.2] | 13 | 0 | — |
| 2 lenses, 0.1 grid, all offsets | 2,940 | 1 (dipole 0:−0.2, 7:+0.2, 2 rotations) | 12 |
| 2 lenses, 0.05 grid, offsets 4–8 | 2,200 | 1 (the same dipole) | 12 |
| 3 lenses, random (24k, seed 20261010) | 24,000 | 45 non-strict → **18 strict** | 21 |
| hill-climb at 0.05 and 0.01 steps from the top 5 | ~150 | — | **21** (plateau) |

"MUX robustness level" is the largest k such that all 8 late-selector MUX functions are computable using only entries that passed ≥ k of 32 noisy trials. k = 32 is run 02's robust criterion.

- Top configs: `0:−0.20,6:0.15,11:0.05` (21), `0:−0.20,7:0.18` (21), `0:−0.45,6:0.15,15:0.30` (20), `0:−0.55,10:−0.10,15:0.66` (20).
- **MUX-capable lens settings are isolated islands.** `0:−0.20,6:0.15,11:0.05` has no MUX at **any** ±0.01 neighbour.
- Physical runs of the dipole's raw MUX programs (8 inputs × 256 trials each):
  - `0:−0.2,7:0.2`: 100% clean, **81–84% at σ=0.15**. REF gives the same (83.0%, 81.0%), and REF keeps the raw MUX with robustness level 12/32.
  - The hill-climbed `0:−0.2,7:0.18`: 100% clean, **83–85% at σ=0.15**.
  - Run 02's robust-only words reach 99.75%.

**Why lenses alone stall: an exact diagnosis.** `a3/mux_obstruction.py` and `a3/mux_minimal.py` work on abstract maps.
- Monotone, sign-preserving contractions plus WRITE+1 *can* express all 256 functions. There is no class-level obstruction.
- With 5 states {−2..2}, NOP and WRITE+1, **no single ERASE map suffices**. The late MUX needs two behaviours, each from one instruction:
  - **(A)** +1→0 while −1 stays at −1 (a sign-selective decrement);
  - **(B)** −2→0 and −1→0 together (a two-quantum collapse).
- With 7 states, (A) alone suffices, but the programs then pass through q=±3, which is marginal at N=16.

On the furnace (`a3/ab_rates.py`, best instruction, min of its two pass counts):

| config | A (k/32) | B (k/32) |
|---|---|---|
| pristine, 0:0.35 | none | none |
| 0:0.50 / 0:0.65 | 18 / 26 | none |
| dipole 0:−0.2,7:0.2 / 0:−0.2,7:0.18 | 20 / 23 | none |
| best 3-lens configs | 21–22 | none |
| + KICK_AT 2.6 (bqsm_vm.c allows any amplitude), 0:0.50 | **32** | none |

**(B) never appears, raw or robust, in any configuration measured, including with kicks up to 3.4.** On 5 states, lenses can supply (A) but never (B). On 7 states, the q=±3 entries cap the programs: the dipole's best is 21/32 and its noisy physical accuracy is 81–84%.

**A3 answer: no.** On one N=16 ring with Nick's ISA (ERASE {1.0, 2.2} + WRITE+1), lenses at 1–3 sites with |Δ| ≤ 0.7 do not close the late-selector MUX robustly. The ceiling is MUX robustness level 21/32, at isolated lens settings.

## 4. Corrections to run 02's narrative

- Run 02's "256/256" and "800/800" needed WRITE−1. Under Nick's +1-only WRITE, the sequential machine reaches 232/256 raw and 224/256 robust.
- Finding 4 called the late-MUX gap "a memory-width limit, since one ring is one register". That is not quite right:
  - With ± writes, 5 states hold the MUX raw. The gap there is the noise-robustness of specific transitions.
  - With +1 writes, the gap is the missing (B) collapse, or a robust q=±3. It is still not register width as such.

## 5. Next levers (ranked)

1. **N=32 ring.** Q_MAX becomes 7, so q=3 is well inside the stable band and the (A)-only route opens. KICK_AT 2.6 already makes (A) robust. This needs `fork_lens` to encode 4 bits per state (15 states) and a furnace built at N=32.
2. **A2, a two-ring register file.** Hold b0 and b1 in different rings so neither (B) nor q=±3 is needed.
3. **Add WRITE−1 to the ISA** (mirrored `apply_ramp`). The problem then reverts to run 02's robustness gap.

## Reproduce

```bash
cd research/forks
gcc -O3 -std=c11 ../../ring_furnace.c -o rf -lm && ./rf selftest             # ALL PASS (also with -DREF_BUILD)
gcc -O3 -std=c11 fork_logic.c -o fork_logic -lm                                # run 02, fast
gcc -O3 -std=c11 -DREF_BUILD fork_logic.c -o fork_logic_ref -lm && ./run_ref.sh  # task 1 (~13 min)
gcc -O3 -std=c11 fork_lens.c -o fork_lens -lm
./fork_lens xcheck 0.0 0.2 0.35 0.5 0.65                                        # trust anchor: IDENTICAL
ROBUST=32 ./fork_lens run "0:-0.2,7:0.18" 256                                    # best MUX programs + physical check
TRACE=1 STRICT=0 ROBUST=32 ./fork_lens run "0:-0.40,1:0.55,5:0.60"               # the early-exit artifact, traced
python3 a3/ab_rates.py "0:0.50" "0:-0.2,7:0.2"                                   # (A)/(B) rates
python3 a3/mux_minimal.py 2                                                      # the (A)+(B) requirement
```

Environment variables for `fork_lens`:
- `WRITE=vm|handoff`, `WRITE_NEG=0|1`, `RAMP_WIDTHS=16[,4,6,8]`, `EXTRA_AMPS=2.6,...`
- `ROBUST=32`, `RSIG=0.15`, `SEED=n`
- `STRICT=1`, `CONFIRM=12`
