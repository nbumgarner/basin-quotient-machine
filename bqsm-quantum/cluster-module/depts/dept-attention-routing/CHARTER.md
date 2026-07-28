# Department Charter: dept-attention-routing

## Mission
Design, validate, and optimize the **super ring attention mechanism** — learned routing weights that gate cross-vQPU macro feature flow based on harmonic similarity.

## Scope
- Attention weight computation: similarity between macro harmonic vectors
- Weight update: Hebbian on macro harmonic correlations
- Feedback modulation: super→macro attention weights as gain control
- Task-dependent routing: filter→filter, memory→memory, attention→attention

## Deliverables
| Design | Description | Validation |
|--------|-------------|------------|
| VQPU-ATTN-BASE-v1 | Uniform weights (current) | Chance routing (50%) |
| VQPU-ATTN-HEBB-vN | Hebbian learning on macro correlations | Routing accuracy > 80% |
| VQPU-ATTN-META-vN | Neuromodulated by task context | Task-switching < 100 chunks |
| VQPU-ATTN-XVQPU-vN | Cross-vQPU super↔super protocol | 3+ vQPUs coordinate |

## Interface Contract
```c
// From dept-super (×16 macro harmonic vectors)
double macro_harmonics[16][8];  // [macro_idx][harmonic]

// Attention weights (to dept-macro)
double attention_weights[16];   // Per-macro gain modulation

// Cross-vQPU super pulses (to dept-pulse-fabric)
typedef struct {
    int dst_vqpu;
    int dst_macro;
    double weight;      // Attention strength
    int harmonic;       // Target harmonic
} super_pulse_t;
```

## Dependencies
- **Uses**: dept-super (macro harmonics), dept-macro (feature vectors), dept-pulse-fabric (super pulses)
- **Used by**: dept-macro (feedback gain), dept-cluster (routing decisions), dept-plasticity (neuromodulation)

## Validation Suite
- `test_attn_weight_learning.c` — Hebbian convergence, stability
- `test_attn_routing.c` — Task-dependent routing accuracy
- `test_attn_feedback.c` — Top-down modulation depth
- `test_attn_xvqpu.c` — Cross-vQPU coordination

## Metrics Dashboard
| Metric | Target | BASE-v1 |
|--------|--------|---------|
| Routing accuracy | > 85% | 50% (chance) |
| Attention entropy | Low (focused) | High (uniform) |
| Feedback modulation | > 5x gain | ~1.5x |
| Cross-vQPU sync | < 10 chunks | N/A |

## Starter Design: VQPU-ATTN-BASE-v1
Current `super_ring_t.routing_weights[MAX_VQPUS]` — uniform initialization

## Known Issues (BASE-v1)
1. **No learning** — weights static uniform
2. **No task context** — same routing for all tasks
3. **No cross-vQPU protocol** — super↔super undefined
4. **Feedback = simple gain** — no feature selectivity

## Research Priorities
1. **Hebbian weight update** — ΔW = η * Σ_h (macro_src[h] * macro_dst[h])
2. **Task modulation** — attention weights gated by super ring "task vector"
3. **Cross-vQPU handshake** — super pulse ACK/NACK for reliable routing
4. **Neuromodulation interface** — dopamine-like signal from dept-cluster for meta-learning

---

*Charter ratified: 2025-07-25 | Department lead: inference_lead*