# Department Charter: dept-super

## Mission
Design, validate, and optimize the **super ring** — 16 macro rings (4096 oscillators) with global attention routing and cross-vQPU coordination.

## Scope
- Macro-to-super aggregation (global context)
- Attention mechanism: weight computation over macro rings
- Inter-vQPU routing decisions (super→super attention)
- Top-down feedback (super→macro attention weights)
- Global state coherence and synchronization

## Deliverables
| Design | Description | Validation |
|--------|-------------|------------|
| VQPU-SUP-BASE-v1 | 16× macro with global attention | super_ring.c unit test |
| VQPU-SUP-ROUT-vN | Cross-vQPU routing policies | Routing accuracy > 85% |
| VQPU-SUP-FB-vN | Feedback attention weights | Modulation depth > 5x |

## Interface Contract
```c
// From dept-macro (×16)
macro_output_t macro_outputs[16];

// To dept-macro (feedback)
double attention_weights[16];  // Per-macro gain modulation

// To dept-pulse-fabric (super-level pulses)
typedef struct {
    int dst_vqpu;
    int dst_macro;
    double phase;          // Global phase state
    double weight;         // Attention weight
    int harmonic;          // Target harmonic
} super_pulse_t;
```

## Dependencies
- **Uses**: dept-macro (16×), dept-attention-routing (weight computation), dept-pulse-fabric (super pulses)
- **Used by**: dept-cluster (vQPU-to-vQPU routing), dept-macro (feedback)

## Validation Suite
- `test_super_attention.c` — attention weight correctness
- `test_super_routing.c` — cross-vQPU routing accuracy
- `test_super_feedback.c` — top-down modulation effect
- `test_super_pulse.c` — super pulse emission/reception

## Metrics Dashboard
| Metric | Target | Current (BASE-v1) |
|--------|--------|-------------------|
| Routing accuracy | > 85% | 50% (chance) |
| Attention entropy | Low (focused) | High (uniform) |
| Feedback modulation | > 5x | ~1.5x |
| Super pulse bandwidth | > 1 Kbps | ~0.02 Mbps |

## Starter Design: VQPU-SUP-BASE-v1
From `superior_model.c` attention scaffolding

## Known Issues (BASE-v1)
1. Attention weights = uniform (no learning)
2. Routing = random (no policy)
3. Feedback to macro = simple gain (no feature selectivity)
4. No super-super cross-vQPU protocol defined

## Research Priorities
1. **Attention learning** — Hebbian on macro harmonic correlations
2. **Routing policies** — task-dependent (filter vs memory vs attention)
3. **Feedback selectivity** — harmonic-specific top-down gain
4. **Cross-vQPU protocol** — super pulse handshake/ack

---

*Charter ratified: 2025-07-25 | Department lead: inference_lead*