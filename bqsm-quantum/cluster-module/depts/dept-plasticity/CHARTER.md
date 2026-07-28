# Department Charter: dept-plasticity

## Mission
Design, validate, and optimize **lens plasticity** — Hebbian adaptation of per-ring lens profiles via local correlation between incoming pulses and filter output.

## Scope
- Pre-synaptic trace: pulse arrival per oscillator site
- Post-synaptic trace: harmonic filter output magnitudes
- Weight update: Oja's rule + competitive normalization in harmonic basis
- Homeostasis: energy normalization, weight decay
- Multi-timescale: fast (per-chunk) + slow (structural)

## Deliverables
| Design | Description | Validation |
|--------|-------------|------------|
| VQPU-PLST-BASE-v1 | From PLASTICITY_RULE_DESIGN.md | Convergence < 1000 chunks |
| VQPU-PLST-FAST-vN | Per-chunk updates | Stability > 95% |
| VQPU-PLST-MULTI-vN | Multi-harmonic competitive learning | Selectivity > 10x |
| VQPU-PLST-META-vN | Neuromodulated learning (attention-gated) | Task-dependent switching |

## Interface Contract
```c
// Per micro ring (added to micro_ring_t)
typedef struct {
    double lens_coeff[8];       // Harmonic basis coefficients
    double pre_trace[16];       // Pulse arrival traces per site
    double post_trace[8];       // Output harmonic traces
    int best_harmonic;          // Dominant harmonic (competitive)
    double lens_energy;         // For normalization
} plasticity_state_t;

// Update called after EACH settle chunk
void plasticity_update(micro_ring_t *ring) {
    // 1. Project pre_trace to harmonic basis
    // 2. Δcoeff[h] = η * pre_proj[h] * post_trace[h] - λ * coeff[h]
    // 3. Reconstruct lens[16] from coeff[8]
    // 4. Normalize energy
    // 5. Decay traces
}
```

## Dependencies
- **Uses**: dept-micro (ring state), dept-pulse-fabric (pre_trace source), dept-attention-routing (neuromodulation)
- **Used by**: dept-micro (lens profiles), dept-macro (feature learning), dept-cluster (task adaptation)

## Validation Suite
- `test_plasticity_converge.c` — learning curve, stability
- `test_plasticity_discrim.c` — harmonic separation
- `test_plasticity_forget.c` — catastrophic forgetting resistance
- `test_plasticity_energy.c` — energy boundedness

## Metrics Dashboard
| Metric | Target | Current (BASE-v1) |
|--------|--------|-------------------|
| Convergence (chunks) | < 1000 | N/A (not implemented) |
| Filter selectivity | > 10x | N/A |
| Forgetting resistance | > 80% retention | N/A |
| Energy stability | max(E)/E_max < 1.0 | N/A |
| Multi-task switching | < 100 chunks | N/A |

## Starter Design: VQPU-PLST-BASE-v1
From `PLASTICITY_RULE_DESIGN.md` — Oja + competition in harmonic basis

## Known Issues (BASE-v1)
1. **Not implemented** — only math design exists
2. **Timescale unknown** — per-chunk vs per-10-chunks vs adaptive
3. **Neuromodulation interface** — how attention gates plasticity
4. **Structural plasticity** — adding/removing lens sites
5. **Cross-ring coordination** — macro-level credit assignment

## Research Priorities
1. **Implement + validate** — single-ring learning curves
2. **Timescale sweep** — find optimal update frequency
3. **Competition dynamics** — winner-take-all vs softmax
4. **Neuromodulation** — attention weight as dopamine signal
5. **Hardware-friendly** — fixed-point, local-only ops

---

*Charter ratified: 2025-07-25 | Department lead: Precision1*