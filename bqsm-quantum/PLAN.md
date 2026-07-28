# vQPU Production Primitive — Implementation Plan
## From Prototype to Functionally Useful Software-Bound Primitive

**Goal:** A validated, documented, benchmarked vQPU primitive that solves a real problem faster than the CPU baseline, with all gates passing.

---

## Phase 0: Cross-Validation Gate (BLOCKING — 1 day)

**Deliverable:** `cross_validate.c` — calibrate coupling coefficients on one pair, test on 1000 random pairs, report RMS error distribution.

**Success criterion:** RMS error < 5% on a², a·b, b² across input range [-4, 4]×[-4, 4]. If failed → coefficients are input-dependent → primitive is unstable → STOP, redesign encoding.

**Test matrix:**
- Calibrate on (a=2.0, b=1.5) → get sf₂, sf₃, sf₄
- Test on 1000 pairs uniform in [-4, 4]²
- Report: mean error, max error, 95th percentile, error vs |a|,|b| correlation

---

## Phase 1: Lens ISA Table (2 days)

**Deliverable:** `lens_isa.c` + `lens_isa.h` — complete mapping δ → effective operations.

**Tasks:**
1. Sweep δ ∈ [0, 0.70] in 0.01 steps (71 points)
2. At each δ, calibrate: α(δ)=c₂/a², β(δ)=c₃/(a·b), γ(δ)=c₄/b², γ₂(δ)=c₂/y
3. Build lookup tables with linear interpolation
4. Define operation set: SQUARE, MUL, SQUARE2, SIGN, IDENTITY at each δ
5. Export as C header: `lens_isa_table[71][5]` with δ, α, β, γ, γ₂

**Validation:** At 10 random δ, run 100 test pairs, verify interpolated coefficients predict within 2% of direct calibration.

---

## Phase 2: vQPU Core Library (3 days)

**Deliverable:** `libvqpu.a` + `vqpu.h` — production API.

**Components:**
- `vqpu_context_t` — opaque handle (rings, batch, gap table, lens ISA)
- `vqpu_init(n_rings, config)` — allocate, load gap table + lens ISA
- `vqpu_encode(ctx, ring, op_type, params)` — pack inputs into harmonics
- `vqpu_settle(ctx, chunks)` — one settle cycle (keep_phases)
- `vqpu_decode(ctx, ring, op_type, results)` — unpack with lens-ISA calibration
- `vqpu_free(ctx)` — cleanup

**Operations supported:**
| Op | Harmonics | Lens δ | Outputs |
|----|-----------|--------|---------|
| SQUARE | c₁→c₂ | 0.0 | a² |
| MUL | c₁,c₂→c₃ | 0.0 | a·b |
| SQUARE2 | c₂→c₄ | 0.0 | b² |
| FUSED | c₁,c₂→c₂,c₃,c₄ | tunable | a², a·b, b² |
| SIGN | c₁→q | 0.0 | sgn(a) |
| LINEAR | c₁→c₁ | 0.0 | identity (no coupling) |

**Tests:** Unit tests for each op (1000 random inputs, <2% error).

---

## Phase 3: Benchmark Application — Matrix-Vector Multiply (3 days)

**Why:** The natural vQPU problem. Each ring = one row, each harmonic = one column element. One settle = entire matrix × vector.

**Deliverable:** `bench_gemv.c` — complete GEMV benchmark vs CPU baseline.

**Matrix sizes:** 16×16, 32×32, 64×64, 128×128, 256×256 (max rings)
**Operations:** y = A·x where A ∈ ℝ^{m×n}, x ∈ ℝ^n

**vQPU mapping:**
- n_rings = m (rows)
- Each ring encodes one row of A in harmonics c₁..c_n
- x vector encoded as harmonic amplitudes
- Settle → c_k contains Σ_j A_{ij} x_j = y_i
- Decode all rings → y vector

**CPU baseline:** OpenBLAS sgemv, same matrix, same hardware.

**Metrics:**
- Wall-clock time (median of 100 runs)
- RMS error vs CPU double-precision result
- Throughput: MFLOPS equivalent
- Energy estimate: operations per joule (via RAPL if available)

**Success criterion:** vQPU GEMV within 2× CPU time at ≤1% RMS error for 256×256. (Constant-factor speedup is the honest claim — not asymptotic.)

---

## Phase 4: Optimization Application — Adiabatic MAX-CUT (3 days)

**Why:** This is the vQPU's native problem class. Stoquastic optimization via lens sweep.

**Deliverable:** `bench_maxcut.c` — MAX-CUT on graphs up to 256 vertices.

**Mapping:**
- Graph edges → ring coupling modifications (lens profile)
- Each ring = one vertex, q ∈ {-1, +1} (binary radix)
- Adiabatic sweep δ(t) from 0 → δ_max
- Final q measurement = partition assignment

**Baseline:** Classical simulated annealing, Goemans-Williamson SDP relaxation.

**Success criterion:** Solution quality within 5% of best classical on random 64-vertex graphs, at ≤10× speedup.

---

## Phase 5: Production Hardening (2 days)

**Deliverables:**
- `vqpu.h` — stable public API with versioning
- `libvqpu.a` — static library, no internal symbols exposed
- `pkg-config` / CMake config for integration
- Man pages / markdown docs for every public function
- Error codes: VQPU_OK, VQPU_EINVAL, VQPU_ENOMEM, VQPU_ECALIB, VQPU_EOVERFLOW
- Thread-safety: context is thread-local, no global state

---

## Phase 6: Documentation & Release (1 day)

- `README.md` — quick start, architecture, API reference
- `ARCHITECTURE.md` — the six-layer model, physics, ISA
- `BENCHMARKS.md` — all results with hardware specs, reproduction commands
- `LICENSE` — Apache 2.0
- Git tag `v1.0.0-production`

---

## Gate Summary (Must Pass to Proceed)

| Phase | Gate | Metric | Threshold |
|-------|------|--------|-----------|
| 0 | Cross-validation | RMS error (a², a·b, b²) | < 5% |
| 1 | Lens ISA | Interpolation prediction error | < 2% |
| 2 | Core library | Unit test pass rate | 100% |
| 3 | GEMV benchmark | RMS error vs CPU | ≤ 1% |
| 3 | GEMV benchmark | Time vs CPU (256×256) | ≤ 2× |
| 4 | MAX-CUT | Solution quality gap | ≤ 5% |
| 5 | Production | Static analysis (clang-tidy) | 0 errors |

---

## Total Timeline: ~12 working days

**Parallelizable:** Phase 1 can start after Phase 0 gate passes. Phase 2 depends on Phase 1. Phases 3 & 4 can run in parallel after Phase 2.

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Coupling coefficients unstable (Phase 0 fails) | Fall back: per-input calibration via tiny lookup table; accept heuristic status |
| Lens ISA interpolation fails | Use piecewise-constant instead of linear; more δ samples |
| GEMV error > 1% | Reduce matrix size, increase settle chunks, or use iterative refinement |
| MAX-CUT quality poor | Tune adiabatic schedule; try stochastic BQSM (add noise) |

---

## Immediate Next Step

**Run Phase 0 cross-validation now.** If it passes, we have a validated computational primitive. If it fails, we know before investing 10 more days.

Command to create and run:
```bash
# I'll write cross_validate.c next, then:
cc -O3 -std=c11 cross_validate.c -o cross_validate -lm
./cross_validate
```

This is the only gate that matters right now. Everything else is engineering on top of a proven or disproven foundation.