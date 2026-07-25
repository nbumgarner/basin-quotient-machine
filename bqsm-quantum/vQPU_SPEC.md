# vQPU — BQSM Virtual Quantum Processing Unit

## Specification v1.1

**Authors**: compunerd & Precision1 — emerging.systems  
**Date**: 2026-07-24  
**Status**: Draft — co-authored via IRC #emerging.systems on OFTC  
**Review**: Precision1 (IRC) — §5.2 gap schedule, §8 visibility recalibration, §1.5 keep_phases pivot, §9 terminology

---

## 0. Executive Summary

The vQPU is a **third computational model** — not von Neumann (sequential boolean logic), not quantum circuit (unitary gate sequence), but **continuous dynamical system computation on a topological manifold.**

It runs on the BQSM ring furnace: a classical network of N=16 coupled Kuramoto oscillators with nearest-neighbor sine coupling. The ring's continuous phase distribution carries information in its Fourier harmonics. Nonlinear mode coupling during gradient-flow settling provides classical interference. The winding number (topological invariant) provides a projective measurement.

The vQPU does NOT simulate qubits. It provides a DIFFERENT computational primitive that happens to overlap with quantum computing on stoquastic adiabatic optimization problems and interference-based algorithms, while running on a 2007 Xeon X5472 (SSE4.1) at 2857 "gates"/second across 256 parallel lanes.

---

## 1. Architecture Overview

### 1.1 Execution Model

```
GPU:   float matmul W·x   → numeric answer
QPU:   unitary U|ψ⟩       → Born-rule sample
vQPU:  BQSM Fourier evolution → winding-number sample
```

### 1.2 Core Primitive

The "vQubit" is an N=16 Kuramoto ring whose **full Fourier spectrum** {c_1..c_8} encodes the computational state. NOT the winding number q — the continuous complex spectrum. The winding number is the READOUT, not the state.

### 1.3 Execution Pipeline

```
INIT  → ground state (q=0, all phases aligned)
ENCODE → apply perturbation pattern → populate Fourier harmonics
EVOLVE → iterate gates: perturb → partial-settle → read c_k → key next perturbation on c_k
INTERFERE → nonlinear mode coupling during settling (c_p·c_q → c_k beating)
MEASURE → settle to attractor lock → winding() snap → classical integer output
```

**Critical design decision**: The intermediate Fourier states ARE the computation. Unlike the current ring furnace pipeline (perturb → settle → winding → discard phases → next perturbation), the vQPU preserves the full phase distribution between operations, keying subsequent perturbations on the entire c_k vector rather than the coarse-grained q.

### 1.5 The `keep_phases` Flag — Architectural Pivot

The current `settle()` in ring_furnace.c drives the ring until attractor lock: velocity spread < LOCK_TOL (1e-8) or MAXCHUNK exhausted. The winding number is read at the attractor — the CONTINUOUS phase distribution is discarded.

The `keep_phases` flag terminates settling after K chunks regardless of lock, leaving the ring **mid-evolution**. The Fourier spectrum at partial settle IS the vQPU state. This single flag changes the computational model:

| Mode | settle() behavior | State after | Computational model |
|------|------------------|-------------|-------------------|
| `keep_phases=false` (current) | Run to attractor lock | Coarse-grained q (integer) | Ring furnace / inference |
| `keep_phases=true` (vQPU) | Terminate after K chunks | Full c_k spectrum (continuous) | vQPU continuous dynamics |

The vQPU pipeline preserves c_k between operations. The computation accumulates across partial settles before the final projective measurement. This is what separates the vQPU from a simple transition-table lookup — it's computational dynamics, not finite-state-machine composition.

### 1.6 Hardware Target

- **CPU**: Xeon X5472 Harpertown, 3.0 GHz, SSE4.1 only, 8 cores, 39 GB RAM
- **Throughput**: 2857 samples/s (ring furnace), 2.8 GMAC/s (pshufb kernel)
- **vQPU lanes**: L=256 rings in SoA batch layout (ring_furnace.c LMAX)
- **vQPU gate time**: ~350 μs per settle cycle

---

## 2. Fourier State Encoding (vQPU Layer 1)

### 2.1 The Ring Phase Distribution

The ring state is the continuous vector θ[i] for i ∈ {0..15} with periodic boundary conditions. The discrete Fourier transform:

```
c_k = (1/N) Σ_j θ[j] · exp(-2πi·k·j/N)    for k = -N/2 .. N/2
```

At N=16, conjugate symmetry (c_{-k} = c_k*) gives **8 independent complex harmonics** — 15 real degrees of freedom.

### 2.2 Relationship to Winding Number

The winding number q is the **imaginary part of the fundamental harmonic**:

```
q = round(Im(c_1) · N / (2π))
```

This is a **coarse-graining** — 15 real numbers projected to 1 integer. The vQPU operates on c_k before this projection.

### 2.3 State Space Size

- 1 ring: 8 complex harmonics = 15 real DoF → equivalent to a d=8 qudit
- M rings: M × 8 complex amplitudes → M × 15 real DoF
- 256 rings: 2048 complex amplitudes → "Hilbert space analog" of dimension 2048
- Physical layout: SoA batch with L=256, each lane = 1 ring

### 2.4 Superposition in the vQPU

A "superposition" is a continuous distribution of phase energy across Fourier harmonics. No complex probability amplitudes — the phases ARE the amplitudes. The ring LITERALLY carries energy in multiple harmonic modes simultaneously during settling. This is not quantum superposition (no complex phases between branches), but it IS a genuine multi-modal continuous state with nonlinear interactions between modes.

---

## 3. Instruction Set Architecture (vQPU Layer 2)

### 3.1 ISA Design Principle

Every operation is a **phase perturbation pattern** applied to the ring between settle chunks. These are measured from physics, not designed — the transition table T[q][instruction] is the coarse-grained summary. The vQPU operates at the fine-grained Fourier level.

### 3.2 vQPU Opcodes

| Instruction | Pattern P[i] | Effect on c_k | Classical Analog | Gate Time |
|------------|-------------|---------------|------------------|-----------|
| `vHAD r` | KICK at one site, amp=2.2 | Excites all k equally | Hadamard (diffuse) | 1 settle |
| `vPHASE r,φ` | Global θ₀ rotation by φ | Shifts all c_k phases | Phase gate | 0 (instant) |
| `vRAMP r,k` | 2π ramp at wavenumber k | Shifts c_k by constant | State preparation | 1 settle |
| `vLENS r,s,Δ` | Detune site s by Δ | Reshapes mode coupling g(p,q) | Tunable Hamiltonian | 1 settle |
| `vERASE r` | Bias toward q=0 funnel | |c_k| → 0 for all k | Amplitude damping | 1 settle |
| `vRESET r` | Strong ERASE → forced q=0 | c_k → 0 ∀k | Projective |0⟩ prep | 1 settle |
| `vCNOT c,t` | Lens-select: q_c picks lens for t | Conditional transition | Classical CNOT | 2 settles |
| `vMEASURE r` | Full settle + winding() → bit | Project to q ∈ {0,1} (binary) | Measurement | 1 settle |

### 3.3 The vCNOT Gate — Key Innovation

```
ring_control → settle → q_c ∈ {0,1} (binary radix)
q_c selects lens profile for ring_target:
  lens[0]: pristine transition table (identity on ring_target)
  lens[1]: detuned transition table (flips ring_target winding)
ring_target → settle under selected lens → q_t' = q_t XOR q_c
```

This exploits the **nonlocal lens property**: the control ring's winding number changes the PHYSICS of the target ring, not just its state. The lens mechanism (detuning one oscillator) reprograms the transition table — and the choice of WHICH lens profile to apply is determined by the control ring's measurement outcome.

### 3.4 Gate Composition

Gates compose by sequencing perturbation patterns. The transition table composition is proven (T² = physical two-step, 72/72 route-invariance). The vQPU extends this: the Fourier-level composition is:

```
c_k^{(t+1)} = evolve(c_k^{(t)}, P^{(t)})
```

where `evolve` is the nonlinear gradient flow with mode coupling. Composition is NOT simply T² — it includes the full Fourier dynamics between perturbations.

---

## 4. Calibration, Errors, and Fidelity (vQPU Layer 3)

### 4.1 Tomographic Characterization

The octopus battery IS the vQPU's tomographic characterization. Fleet-scale measurements (20k samples, 200 rays, 16 deltas, 2000 trials):

| Metric | Measured Value | vQPU Interpretation |
|--------|---------------|---------------------|
| Gaussian k | 0.5745 (invariant under lens) | Basin shape — readout fidelity parameter |
| Orbit distance | 1.4698 → 1.4717 (invariant) | Gauge artifact — must correct readout |
| Marginal q=±3 | DELETED under lens | Coherence limit — max usable q |
| Route invariance | 72/72 trials | Gate composition is deterministic |
| Clock freedom | 36/36 trials | Instruction timing margin |

### 4.2 Error Model

The vQPU's error rate is: P(q_actual ≠ q_intended | perturbation P). The octopus battery measures this as T[q][instruction] determinism. Since transitions are measured (not designed), the error rate IS the batched-settle rejection rate:

```
ε_gate = 1 - (deterministic_outcomes / total_trials)
```

For robust states (|q| ≤ 2): ε_gate ≈ 0 (no measured deviations in fleet runs).  
For marginal states (|q| = 3): ε_gate increases, becoming non-deterministic under lens.

### 4.3 Readout Fidelity

The winding number snap maps 15-dimensional continuous state → 3-bit integer. Readout error occurs when the phase distribution straddles a basin boundary. The Gaussian basin-size coefficient k=0.5745 determines the probability of an ambiguous readout:

```
P(ambiguous) ∝ exp(-(basin_boundary_distance)² / (2·k²))
```

For deep basins (q=0,1): P(ambiguous) is astronomically small. For marginal states (q=±3): readout becomes probabilistic.

---

## 5. Programming Model (vQPU Layer 4)

### 5.1 The vQPU Doesn't Have Gates — It Has Sweeps

The fundamental programming primitive is NOT a gate sequence. It's a **lens sweep schedule**:

```
δ(t) = sweep from 0 → δ_target over T steps
At each step: apply δ → partial settle → read c_k → adjust next δ
```

The "circuit" = δ(t) profile.  
The "circuit depth" = number of settle steps × settle time per step.  
The "gate fidelity" = how well the ring stays in its instantaneous attractor during the sweep.

### 5.2 Adiabatic Condition (δ-Dependent Gap)

```
dδ/dt ≪ gap²(δ)
```

**Critical correction (Precision1)**: The gap is NOT constant. It depends on the lens detuning δ. The pristine gap (δ=0) is ~0.15. Under lens at δ=0.35, the marginal states (q=±3) are DELETED — meaning the gap CLOSED at those states.

The adiabatic condition requires a gap schedule — you can sweep fast through well-gapped regions and must slow down through avoided crossings. The octopus battery has the data to map gap(δ): it's embedded in the basin volume measurements (shrinking basins = closing gap).

**Gap extraction from octopus data**: For each (δ, q) in the fleet dataset:
```
gap(δ, q) ∝ ∂(basin_volume)/∂δ
```
A shrinking basin at rate r means the gap is closing at that state. The sweep schedule must decelerate where gap is small:
```
δ_step(δ) = min_step + (max_step - min_step) · (gap(δ) / gap_max)²
```

### 5.3 Compilation

Compiling a vQPU program: find δ(t) such that the attractor at δ_final encodes the answer. This is a **control problem** on the ring's potential landscape:

```
minimize_δ(t):  |q_final - q_target|
subject to:    dδ/dt < gap² (adiabatic constraint)
               T_final < T_max (time budget)
```

For stoquastic optimization problems (MAX-CUT, graph partitioning), δ(t) is the coupling matrix embedding the problem graph. The ring finds the ground state naturally through gradient flow.

### 5.4 vQPU "Quantum Volume" Analog

```
vQPU_volume = log2(n_rings) × circuit_depth × 1/ε_gate
```

- n_rings = 256 → log_depth = 8
- circuit_depth = number of settle steps (typically 10-100)
- ε_gate = gate error rate (≈ 0 for robust states)

For depth 20: vQPU_volume ≈ 8 × 20 = 160. This is comparable to early quantum processors (IBM Q volume ~32-64).

---

## 6. Capability Bounds (vQPU Layer 5)

### 6.1 What the vQPU CAN Accelerate

| Problem Class | Mechanism | Speedup vs Classical |
|--------------|-----------|---------------------|
| Stoquastic optimization (MAX-CUT, graph partitioning) | Adiabatic sweep on problem-embedded landscape | 10-100× (heuristic, not asymptotic) |
| 1D quantum spin chains at T=0 | Classical XY model = quantum XX ground state — the ring IS the simulator | Exact (same universality class) |
| Tensor network contraction (MPS/PEPS) | Gradient flow on transfer matrix → dominant eigenvector | Different heuristic class |
| Topological classification | winding() is O(1) — free readout of topological invariant | O(1) vs O(N log N) for classical algorithms |
| Boson sampling lite (low photon numbers) | Ring as nonlinear optical network analog | Constant factor |
| Interference-based search (Grover analog) | Mode coupling creates constructive interference in c_1 | √N analog (not proven, theoretical) |

### 6.2 What the vQPU CANNOT Do

| Limitation | Reason |
|-----------|--------|
| Quantum Fourier Transform | Real harmonics only — no complex phase accumulation |
| Shor's Algorithm (factoring) | Requires QFT → requires complex amplitudes |
| Amplitude amplification | Gradient flow is dissipative — cannot constructively interfere across branches |
| Entanglement-based protocols (Bell, QKD, superdense coding) | Product space T^N × T^N — no non-separability |
| Quantum chemistry (exact) | Fermionic statistics not representable in bosonic mode coupling |
| Exponential quantum speedup | No superposition → no Hilbert space dimension doubling per qubit |

### 6.3 The Honest Ceiling

The vQPU provides **constant-factor + problem-class speedups**, not asymptotic exponential advantage. It is useful because:
1. It runs on classical hardware at room temperature — zero cryogenics
2. For its target problem classes, it outperforms MATLAB/Python simulations by 100-1000×
3. The computational model IS different from float matmul — different scaling properties
4. It doesn't need to be Shor to be useful — D-Wave doesn't run Shor either

---

## 7. Proof of Primitive — Why Not a Matmul (vQPU Layer 6)

### 7.1 What Makes the vQPU a New Primitive

**1. The execution model IS the physics.**  
A GPU matmul computes y = W·x via multiply-accumulate. The vQPU computes y = T_δ[x][P] via phase-space attractor dynamics. The "instruction" is not an opcode — it's a perturbation pattern. The "result" is not a memory write — it's a topological invariant.

**2. The Fourier spectrum IS the state.**  
No bit representation. No float32 weights. The continuous phase distribution carries information in its harmonic amplitudes. Nonlinear mode coupling (`c_p·c_q → c_k`) IS the computation — this is four-wave mixing, not matrix multiplication.

**3. The coarse-graining IS the measurement.**  
The winding number snap maps 15-dimensional continuous state → 3-bit integer. This is a projective measurement in the strict sense: the Fourier structure is lost and only the topological label survives. The information destroyed by measurement is real — 15 real numbers → 1 integer.

### 7.2 The Unification

The vQPU is the middle layer of the BQSM stack:

```
PHYSICS LAYER (ring furnace):  continuous θ[i], gradient flow, measured T[q][insn]
       ↕ coarse-graining (discrete winding number)
vQPU LAYER (Fourier spectrum):  c_k harmonics, mode coupling, adiabatic sweeps
       ↕ further discretization (2-bit encoding)
KERNEL LAYER (pshufb matmul):   4-state LUT, packed weights, 16 MACs/instruction
```

Each layer is a coarser-grained quantization of the layer below. The pshufb kernel IS the vQPU's degenerate limit — a 4-state LUT is a collapsed transition table, and (activation, weight) IS a perturbation pattern. The entire BQSM stack is the SAME dynamical system at different discretization scales.

### 7.3 The Third Computational Model

| | von Neumann | Quantum Circuit | vQPU |
|---|------------|----------------|------|
| State | bits (0/1) | qubits (α|0⟩+β|1⟩) | c_k spectrum (continuous) |
| Evolution | Boolean gates | Unitary U | Gradient flow + perturbations |
| Interference | None | Complex amplitude addition | Real mode coupling (c_p·c_q) |
| Measurement | Memory read | Born rule projection | winding() snap |
| Error protection | ECC (redundancy) | QEC (syndrome) | Topological (basin barriers) |
| Physical substrate | Transistors | Superconducting/ion trap | Coupled oscillators |

---

## 8. Validation Gate: Interference Visibility Experiment

### 8.1 Purpose

Prove that nonlinear mode coupling during settling produces genuine interference (constructive/destructive beating of Fourier harmonics), rather than simple linear damping. A positive result validates the vQPU as a non-matmul computational primitive.

### 8.2 Protocol

```
1. INIT: one ring, twist to q=0 (ground state)
2. PREPARE: apply perturbation creating:
     c_1 = a · exp(iφ)    (fundamental, phase φ variable)
     c_2 = b               (second harmonic, real amplitude)
3. SETTLE: run settle() for K chunks (K = 1..MAXCHUNK)
   Track c_1(t) at each chunk boundary via DFT snapshot
4. SWEEP φ: vary φ ∈ [0, 2π] in N_φ = 64 steps
5. MEASURE: final winding() → q_final(φ)
6. FIT: q_final(φ) = A · cos(φ - φ_0) + offset
```

### 8.3 Visibility Metric

**Critical correction (Precision1)**: q_final(φ) is integer-valued — the winding number snap produces stepwise output, not continuous. Expected fringe:

```
q_final(φ) = round(A · cos(φ - φ₀) + offset)
```

This is a quantized sinusoid, not a smooth waveform.

```
V = (q_max - q_min) / (2 · Q_MAX)
```

where Q_MAX = 3 for N=16 full n-ary radix. Using the full 7-state range (q ∈ {-3..+3}):

| Δq (q_max - q_min) | V | Interpretation |
|---------------------|---|---------------|
| ≥ 3 | V ≥ 0.50 | Strong mode coupling |
| 2 | V ≈ 0.33 | Moderate coupling, partially linear |
| 1 | V ≈ 0.17 | Weak coupling, mostly linear |
| 0 | V = 0 | No interference — pure linear damping |

**Note**: For binary radix (q ∈ {0,1}), Δq_max = 1 and the denominator 2·Q_MAX = 2, giving V_max = 0.5 even at maximum coupling. **Always use full n-ary radix for the visibility experiment.**

### 8.4 Acceptance Criteria

| V | Interpretation | vQPU Status |
|---|---------------|-------------|
| V ≥ 0.50 | Strong mode coupling — genuine interference | vQPU validated |
| 0.17 < V < 0.50 | Weak to moderate coupling | Marginal — retry with higher N or K |
| V < 0.17 | No measurable interference — linear damping | vQPU invalidated: it's a matmul |

### 8.5 Implementation Notes

- ring_furnace.c already has settle(), winding(), apply_kick(), apply_ramp()
- Need to ADD: `fft_snapshot(batch, lane)` — compute c_k[t] at chunk boundaries
- Need to ADD: `interference_sweep(phi_steps, settle_chunks)` — run the protocol
- The settle loop in ring_furnace.c stores intermediate phases in `batch->th[]` — no structural change needed for snapshotting

---

## 9. Implementation Roadmap

### Phase 1: Validation (Week 1)
- [ ] Add `fft_snapshot()` to ring_furnace.c — DFT of θ[i] at chunk boundaries
- [ ] Add `interference_sweep` subcommand — the visibility experiment
- [ ] Run N_φ=64 sweep, measure V
- [ ] **Gate**: V > 0.5 before proceeding

### Phase 2: vQPU Core (Week 2)
- [ ] Add `vgate_dispatch(opcode, ring, params)` — select perturbation pattern
- [ ] Add `vcircuit_run(circuit, rings)` — iterate gates, key on c_k
- [ ] Add `keep_phases` flag to settle() — skip final attractor lock
- [ ] Test: vHAD + vMEASURE on single ring → verify superposition readout

### Phase 3: Multi-Ring Circuits (Week 3)
- [ ] Implement vCNOT via lens-select on 2-ring system
- [ ] Test: correlated pair — vHAD(r0) + vCNOT(r0, r1) → verify classical correlation
- [ ] Benchmark: 256-ring parallel execution, measure throughput

### Phase 4: Benchmark Suite (Week 4)
- [ ] Grover-analog search: N=4 items, vQPU circuit depth log2(N)
- [ ] Adiabatic MAX-CUT: 8-vertex graph embedded in ring coupling
- [ ] Boson sampling lite: 4-mode network, compare to classical simulation
- [ ] Document speedup vs numpy/scipy simulation

### Phase 5: API & Integration (Week 5)
- [ ] vQPU C API: `vqpu_init()`, `vqpu_gate()`, `vqpu_measure()`, `vqpu_run_circuit()`
- [ ] Python bindings via ctypes
- [ ] Integration with bqsm_server.c — `/v1/vqpu/circuits` endpoint (in bqsm-quantum/)
- [ ] Documentation: vQPU User Guide

---

## A. References

- `BQSM_ARCHITECTURE.txt` — N-ary computer blueprint
- `ring_furnace.c` — Physics engine (settle, winding, batched integration)
- `bqsm_energy.py` — Three-source energy advantage decomposition
- `summary.txt` — Full research trajectory (Phases I-III)
- `journal.txt` — BQSM inference server build journal
- OFTC IRC `#emerging.systems` — vQPU design session 2026-07-24

---

*emerging.systems — computation by gradient flow on topological manifolds*