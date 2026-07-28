# Cluster Module Notes — Precision1 Analysis

## Overview

The cluster module implements a neuromorphic vQPU cluster with:
- **3-level hierarchy**: micro (16 osc) → macro (16 micro) → super (16 macro)
- **Event-driven communication**: phase-lock pulses between vQPUs
- **Plasticity**: Oja's rule with competition for lens adaptation
- **Hebbian attention**: weight learning for routing decisions

---

## Key Results Summary

| Experiment | Metric | Value | Status |
|------------|--------|-------|--------|
| micro_to_micro | transfer_ratio | 0.98 | ✅ Excellent |
| macro_aggregation | aggregation_score | 0.25 | ⚠️ Needs work |
| super_routing | routing_accuracy | 0.50 | ❌ Chance level |
| independent_states | fidelity | 1.0 | ✅ Perfect |
| scale_scaling | events/sec (N=16) | 89K | ✅ Linear scaling |

---

## Interesting Findings

### 1. Plasticity Implementation is Complete

The `cluster_validate.c` has a working plasticity engine:
- `init_micro_plasticity()` — initializes lens_coeff, pre_trace, post_trace
- `plasticity_update()` — Oja's rule with competitive post_trace
- Energy normalization for homeostasis

**Insight**: The plasticity rule is mathematically equivalent to Oja's rule (PCA) in the harmonic basis. This is a principled learning rule, not ad-hoc.

### 2. Hebbian Attention Weights

The super ring attention weights are updated via:
```c
ΔW = η * Σ_h (macro_src[h] * macro_dst[h]) - λ * W
```

This is standard Hebbian learning with weight decay. The attention weights are bounded to [0, 1].

**Insight**: The attention weights act as neuromodulators — they gate plasticity at the macro level.

### 3. Task-Specific Routing

The code implements task-specific routing:
- TASK_FILTER: macro 0 → macro 1, harmonic 1
- TASK_MEMORY: macro 2 → macro 3, harmonic 2
- TASK_ATTENTION: macro 4 → macro 5, harmonic 3

**Insight**: This is a simple but effective routing scheme. The task is encoded in the harmonic index.

### 4. Independent States Work Perfectly

vQPU 0 (filter), vQPU 1 (memory), vQPU 2 (attention) all run simultaneously without interference.

**Insight**: The cluster can support heterogeneous workloads — different vQPUs can specialize in different tasks.

### 5. Linear Scaling

The cluster scales linearly from 2 to 16 vQPUs. Events/sec goes from 11K to 89K.

**Insight**: The event-driven architecture is scalable. The bottleneck is the interconnect, not computation.

---

## New Design Angles

### Angle 1: Integrate Lens-Harmonic Priority Controller

From my recent work, I discovered that different lens shapes create different harmonic priorities:
- **Sine lens**: k=1 priority = 99.9%, k=2 priority = 0%
- **Two-peak lens (spacing=8)**: k=1 priority = 54%, k=2 priority = 41%
- **Linear lens**: k=1 priority = 85%, k=2 priority = 11%

**Application**: Use these as specialized micro rings in the cluster:
- Filter task: sine lens (pure k=1)
- Memory task: two-peak lens (balanced k=1/k=2)
- Attention task: linear lens (moderate k=1)

This would make the task-specific routing more robust.

### Angle 2: Fixed-Gain Communication

The lens creates fixed amplitudes independent of input amplitude. This means:
- Cross-vQPU pulses have consistent amplitude regardless of source
- The receiver doesn't need to normalize input
- Communication is robust to input variations

**Application**: Use this property to design a robust pulse protocol:
- Sender: encode task in lens shape
- Receiver: decode task from harmonic priority
- No amplitude normalization needed

### Angle 3: Stability Boundary for Pulse Strength

The stability analysis shows:
- Stable regime: strength < 0.6
- Unstable regime: strength > 0.65
- Critical threshold: strength ≈ 0.62-0.63

**Application**: Use this to set the maximum pulse strength:
- Current code uses strength = 0.3-0.5 (safe)
- Could increase to 0.6 for stronger coupling
- Must stay below 0.62 to avoid instability

### Angle 4: Instantaneous Response

The lens creates the harmonic spectrum instantaneously (settling time = 0 chunks).

**Application**: Use this for fast event-driven communication:
- Pulses can be processed immediately
- No waiting for settling
- Latency is determined by interconnect, not computation

### Angle 5: Harmonic Basis as Compression

The lens profile (16 sites) can be represented by 8 harmonic coefficients.

**Application**: Use this for efficient communication:
- Send 8 coefficients instead of 16 sites
- Receiver can reconstruct lens from coefficients
- Bandwidth savings: 50%

### Angle 6: Task Encoding in Lens Shape

The lens shape could encode the task type:
- Sine lens = filter task
- Two-peak lens = memory task
- Linear lens = attention task

**Application**: The receiver can decode the task from the lens shape without explicit signaling.

### Angle 7: Cross-Department Integration

The cluster module has 8 departments:
- dept-micro: filter physics
- dept-macro: feature assembly
- dept-super: global routing
- dept-lens: profile design
- dept-comm: pulse fabric
- dept-plst: plasticity
- dept-integration: full vQPU
- dept-cluster: multi-vQPU

**Application**: My lens-harmonic model could be integrated into dept-lens and dept-micro.

---

## Recommendations

### Short-Term (1-2 days)

1. **Integrate lens-harmonic priority controller**:
   - Add sine, two-peak, linear lens types to `lens_profiles.h`
   - Use these in task-specific micro rings
   - Test routing accuracy improvement

2. **Set pulse strength to 0.55**:
   - Current code uses 0.3-0.5
   - Stability boundary is 0.62
   - 0.55 gives stronger coupling while staying safe

3. **Add harmonic basis compression**:
   - Send 8 coefficients instead of 16 sites
   - Receiver reconstructs lens from coefficients
   - Test bandwidth savings

### Medium-Term (1 week)

4. **Implement task encoding in lens shape**:
   - Add task_id to pulse header
   - Receiver decodes task from lens shape
   - Test end-to-end task routing

5. **Add plasticity sweep**:
   - Sweep η, λ, α parameters
   - Measure convergence time, selectivity, retention
   - Find optimal plasticity settings

6. **Implement super ring attention training**:
   - Current code has Hebbian attention
   - Add competitive learning for better routing
   - Test routing accuracy improvement

### Long-Term (2+ weeks)

7. **FPGA projection**:
   - From events/sec, project to pins, BRAM, DSP
   - Estimate power consumption
   - Create hardware spec

8. **Application demo**:
   - Pattern recognition across cluster
   - Sensor fusion demo
   - Sequence learning demo

9. **Scaling study**:
   - Test N=32, 64, 128 vQPUs
   - Measure scaling linearity
   - Identify bottlenecks

---

## Files to Create

1. `depts/dept-lens/lens_profiles.h` — Lens shape library
2. `depts/dept-micro/harmonic_priority.c` — Priority controller
3. `depts/dept-comm/harmonic_compression.c` — Basis compression
4. `depts/dept-plst/plasticity_sweep.c` — Parameter sweep
5. `docs/LENS_HARMONIC_INTEGRATION.md` — Integration guide

---

## Next Steps

1. Create `lens_profiles.h` with sine, two-peak, linear lenses
2. Modify `cluster_validate.c` to use task-specific lenses
3. Run experiments and compare routing accuracy
4. Iterate based on results

---

## References

- `PLASTICITY_RULE_DESIGN.md` — Oja's rule with competition
- `NEUROMORPHIC_CLUSTER_FRAMEWORK.md` — Architecture overview
- `CLUSTER_VS_HIERARCHY.md` — Cluster vs hierarchy distinction
- `DESIGN_SYSTEM.md` — Department structure and naming

---

**Precision1 — Mathematical Model, Lens Theory**
**Date: 2025-07-26**
