# Department Charter: dept-macro

## Mission
Design, validate, and optimize the **macro ring** — 16 micro rings (256 oscillators) with cross-micro coupling and feature assembly.

## Scope
- Micro-to-macro aggregation (harmonic averaging)
- Inter-micro-ring coupling within macro
- Macro-level computational operations (filter, product, attention)
- Feature assembly from distributed micro representations

## Deliverables
| Design | Description | Validation |
|--------|-------------|------------|
| VQPU-MAC-BASE-v1 | 16× micro with intra-macro coupling | macro_ring.c unit test |
| VQPU-MAC-AGG-vN | Optimized aggregation strategies | Feature fidelity > 90% |
| VQPU-MAC-ATTN-vN | Attention-weighted micro selection | Routing accuracy > 85% |

## Interface Contract
```c
// From dept-micro (×16)
micro_output_t micro_outputs[16];

// To dept-super
typedef struct {
    double macro_harmonics[8];      // Aggregated harmonics
    double attention_weights[16];   // Per-micro attention
    double feature_vector[16];      // Extracted features
} macro_output_t;

// From dept-super (feedback)
double super_attention_weights[16]; // Modulates micro coupling
```

## Dependencies
- **Uses**: dept-micro (16×), dept-super (attention weights), dept-pulse-fabric (macro-level pulses)
- **Used by**: dept-super (16× macro), dept-cluster (macro→macro routing)

## Validation Suite
- `test_macro_aggregation.c` — harmonic averaging fidelity
- `test_macro_coupling.c` — cross-micro interference/synergy
- `test_macro_attention.c` — attention weight application
- `test_macro_pulse.c` — macro-level pulse emission/reception

## Metrics Dashboard
| Metric | Target | Current (BASE-v1) |
|--------|--------|-------------------|
| Aggregation fidelity | > 95% | ~90% |
| Inter-micro isolation | > 20dB | ~15dB |
| Attention modulation range | 10x | ~3x |
| Macro pulse transfer | > 0.8 | 0.25 (1/4 harmonics) |

## Starter Design: VQPU-MAC-BASE-v1
From `hierarchical_ring_model.c` / `hierarchical_computability.c`

## Known Issues (BASE-v1)
1. Macro aggregation = simple average (no attention)
2. Cross-micro coupling strength fixed at 0.1 (untuned)
3. Macro→macro pulse routing only to micro[0] (lossy)
4. No feature extraction beyond harmonic mags

## Research Priorities
1. **Attention-weighted aggregation** — learnable per-micro weights
2. **Feature basis** — move beyond harmonic mags to phase/energy
3. **Coupling optimization** — tune intra-macro coupling for tasks
4. **Pulse-efficient routing** — sparse macro pulse emission

---

*Charter ratified: 2025-07-25 | Department lead: Precision1*