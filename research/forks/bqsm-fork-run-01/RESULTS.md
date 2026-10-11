# BQSM fork run 01 — reference results (2026-10-10)

**Scope.** Independent Kuramoto reference simulator (`kring.c`), NOT ring_furnace. N=16, identical-form sinusoidal coupling, RK4, double precision. No lenses, no gauge-corrected orbit metric. Results are a baseline to replicate on the T7400/Oracle shards.

## Fork A — Boolean programmability (3 inputs, 1 output, 256 possible truth tables)

Setup: random "programs" = per-edge coupling signs (±1) + small frequency biases. Inputs on oscillators 0/5/10, output = sign of time-averaged cos(θ13 − θ15). A function counts as **robust** only if it reproduces under 3 noisy re-runs (IC noise σ = 0.15 rad).

| Encoding | Topology | Programs | Distinct tables | Robust distinct |
|---|---|---|---|---|
| Input as initial-phase kick (π) | ring | 3000 | 233 | 3 (constants + 1) |
| Input as initial-phase kick (π) | ring + 4 chords | 3000 | 230 | 2 (constants only) |
| Input as initial-phase kick, σ=0.05 | ring | 1200 | 169 | 2 (constants only) |
| **Input as persistent frequency bias (±0.3)** | ring | 1200 | 162 | **7** |

Robust non-constant functions found with frequency encoding (input index = b0 + 2·b1 + 4·b2):
- `10000000` → NOR3 (universal gate)
- `00001000` → b0 ∧ b1 ∧ ¬b2
- `00000101` → b0 ∧ b2
- `11111011` → ¬(b0 ∧ ¬b1 ∧ b2)
- `10101101` → non-trivial 5-minterm function

**Findings.**
1. Raw coverage (233/256) is an illusion: almost all of it comes from sensitivity at basin boundaries, and it dies under small noise.
2. Inputs written into the *initial state* get erased by dissipation — only constants survive.
3. Inputs written into the *parameters* (persistent drive that reshapes the landscape) produce real, noise-robust logic, including NOR3. This is the same move nonlinear spectral computing makes (input lives in the potential).
4. Each robust non-constant function was found by only ~1 program in 1200: random search is the bottleneck → FORK[1] (a program-design method) is confirmed as the critical path.

## Fork B — integrability boundary

Identical oscillators, K = (1−ε)·global + ε·ring. WS cross-ratio drift measured over t∈[0,1]; attractors counted from 150 random ICs by winding number.

| ε | WS drift | Attractors (by |q|) |
|---|---|---|
| 0.00 | 7.1e-13 | sync only |
| 0.02 | 9.2e-02 | sync only |
| 0.10 | 3.8e-01 | sync only |
| 0.50 | 3.4e+00 | sync only |
| 0.80 | 1.0e+01 | sync only |
| 0.90 | 6.4e+00 | q=0,±1,±2 (50 runs slow to converge) |
| 1.00 | 5.6e+00 | q=0,±1,±2 |

**Findings.**
1. Simulator validated: global coupling conserves WS cross-ratios to ~1e-12.
2. Integrability breaks immediately (any ε>0), but multistability only appears at ε ≈ 0.85–0.9.
3. Hypothesis "multistability begins where integrability ends" is **not supported** in this model; loss of integrability is necessary but far from sufficient. The onset is set by twisted-state stability against the global mean-field pull — an analytic linear-stability question, cheaper than simulation.

## Recommended next runs
- **A-next:** targeted search (hill-climb on coupling signs/weights + frequency biases) for each of the 256 functions under the robustness constraint, frequency-encoded inputs. Output: robust-coverage number. Then replicate on ring_furnace with lenses as the program knobs.
- **B-next:** replace simulation with linear stability of twisted states under mixed coupling; predict ε* and check against 0.85–0.9. If it matches, Fork B becomes a short note, not a program.
- **C (unlocked by A):** robustness is where dissipative rings beat integrable computing — measure noise tolerance curves for the robust functions.
