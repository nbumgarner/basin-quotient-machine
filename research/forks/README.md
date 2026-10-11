# Ring-logic forks

Experiments on what coupled phase-oscillator rings can compute, built on
[`ring_furnace.c`](../../ring_furnace.c). The published write-ups are in the
[research library](https://emerging.systems/research/index.html) under the tag `ring-logic`:

| note | instrument | detailed results |
|---|---|---|
| [One-ring winding logic … needed an instruction we do not have](https://emerging.systems/research/2026-10-10-ring-logic-write-sign.html) | `fork_logic.c`, `fork_lens.c` | `RESULTS_run03.md` §1–2 |
| [Lenses alone do not make a one-ring multiplexer survive noise](https://emerging.systems/research/2026-10-10-ring-logic-lenses.html) | `fork_lens.c`, `a3/` | `RESULTS_run03.md` §3 |
| [Once rings are coupled, the winding number is no longer the state](https://emerging.systems/research/2026-10-10-ring-logic-coupled-windings.html) | `fork_macro.c` | `RESULTS_run04_A2.md` §1–2 |
| [Three coupled rings compute a multiplexer in one settle …](https://emerging.systems/research/2026-10-11-ring-logic-continuous-settle.html) | `fork_macro.c` | `RESULTS_run04_A2.md` §3–4 |
| [Two days of ring-logic experiments, including everything that failed](https://emerging.systems/research/2026-10-11-ring-logic-log.html) | — | all of the above |

Every number in the notes comes from a file in this directory.

## Build

Each instrument `#include`s `../../ring_furnace.c`, so build from here:

```bash
gcc -O3 -std=c11 ../../ring_furnace.c -o ring_furnace -lm && ./ring_furnace selftest   # must be ALL PASS
gcc -O3 -std=c11 fork_logic.c -o fork_logic -lm
gcc -O3 -std=c11 fork_lens.c  -o fork_lens  -lm
gcc -O3 -std=c11 fork_macro.c -o fork_macro -lm
# reference builds (step 0.02, libm sin, no early-exit lock): add -DREF_BUILD
```

Trust anchors, which must print IDENTICAL:

```bash
./fork_lens xcheck 0.0 0.2 0.35 0.5 0.65   # generalised table builder vs build_table()
./fork_macro xcheck                         # coupled integrator at coupling 0 vs build_table()
```

## What is here

| path | what |
|---|---|
| `HANDOFF.md`, `RESULTS_run02.md`, `bqsm-fork-run-01/` | the earlier runs these experiments start from, including `kring.c` (an independent reference simulator) |
| `fork_logic.c` | one ring, register = winding number; which of the 256 three-input functions are reachable |
| `run_ref.sh`, `out_fast/`, `out_ref/`, `handoff_csv/` | the fast-build reproduction and the reference-build re-run |
| `fork_lens.c` | the lens generalised to any number of sites; the instruction set's own write (`apply_ramp` from `bqsm_vm.c`); witness programs and kick-by-kick verification; strict settle |
| `a3/` | the lens searches: `make_specs.py` regenerates every configuration list; `run_*.sh` run them; `hillclimb.py`; `mux_obstruction.py` and `mux_minimal.py` (abstract map enumeration); `ab_rates.py` (the two required erase behaviours, measured) |
| `fork_macro.c` | two register rings and an output ring coupled site to site: gates, attractor census, phasor readout, frequency-drive programs, Langevin noise |
| `a2/`, `a2c/` | raw outputs of the coupled-ring and drive-program runs |

## Notes on the data

- `fork_lens.c` settles strictly by default: a lane is locked only once its velocity spread is below the furnace's tolerance and its winding has not changed. The fast build's early-exit test is exact only for an undetuned ring.
  - The raw lens searches (`a3/grid2`, `a3/fine`, `a3/rand3/raw_vm16_all.csv`) ran before strict settle existed, so `run_*.sh` pin `STRICT=0` to reproduce them.
  - The 45 raw multiplexer hits were then re-run strict (`a3/rand3/strict_robust_hits.csv`).
- CSVs written before the robustness-level columns existed have 12 columns; later ones have 14 (`mux_k`, `dec10`).
- `fork_macro.c` uses the velocity lock only, everywhere. Its phasor readout is taken against the array's collective phase.
- Machine: Intel Core 3 100U laptop CPU (8 threads, no GPU), gcc 15.3, `-O3`.
- Verified against the `ring_furnace.c` in this repository's `main`:
  - selftest ALL PASS;
  - `fork_logic` reproduces `handoff_csv/fork_logic.csv` byte for byte;
  - both cross-checks IDENTICAL;
  - re-run lens-grid rows match `a3/grid2/raw_vm16_all.csv`.
