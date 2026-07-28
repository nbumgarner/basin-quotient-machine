# Priority Research Trajectories

## Executive Summary
Based on validation data from `cluster_validate.c` and the department architecture, here are the **5 highest-impact research trajectories** ranked by: (1) decision-unblocking value, (2) compounding leverage, (3) feasibility.

---

## Trajectory 1: Plasticity Engine Implementation
**Department**: dept-plasticity  
**Unblocks**: All adaptive behavior, lens optimization, task learning  
**Leverage**: Enables automatic lens design → removes manual tuning bottleneck

### Current State
- Math design complete: `docs/PLASTICITY_RULE_DESIGN.md` (Oja + competition in harmonic basis)
- Not implemented in simulation
- No learning curves, convergence data, or forgetting measurements

### Milestones
| Week | Milestone | Success Criteria |
|------|-----------|------------------|
| 1 | Implement `plasticity_update()` in `cluster_validate.c` | Compiles, runs without crash |
| 2 | Single-ring learning: harmonic 1 → 2 switch | Selectivity > 5x in < 1000 chunks |
| 3 | Catastrophic forgetting test: learn A, then B, retain A | > 80% retention |
| 4 | Energy stability: 10K chunks, max(E)/E_max < 1.0 | Bounded weights |
| 5 | Multi-ring: macro ring learns feature assembly | Aggregation fidelity > 90% |

### Key Experiments to Add
```c
// In cluster_validate.c
void run_plasticity_learning_curve() {
    // Sweep: η ∈ [0.001, 0.01], λ ∈ [1e-5, 1e-3], α ∈ [0.9, 0.99]
    // Measure: convergence_chunks, final_selectivity, stability
}

void run_plasticity_forgetting() {
    // Task A: learn harmonic 1 filter
    // Task B: learn harmonic 2 filter  
    // Measure: Task A retention after B training
}

void run_plasticity_meta() {
    // Attention weight from dept-super gates η (neuromodulation)
    // Measure: task-switching speed
}
```

### Decision Unblocked
- "What lens profile for this task?" → **Learned automatically**
- "What quantization level works?" → **Plasticity adapts to quantization noise**
- "How to share features across vQPUs?" → **Plasticity aligns lens profiles**

---

## Trajectory 2: Macro Aggregation Fix
**Department**: dept-macro + dept-pulse-fabric  
**Unblocks**: Cross-vQPU feature routing, hierarchical processing  
**Leverage**: Current 25% → target 90%+ harmonic detection

### Current Problem
- vQPU 0 micro rings emit pulses → vQPU 1 macro ring 0 receives
- Only 1/4 harmonics detected (0.25 aggregation score)
- Root cause: **Destination micro rings have wrong lens for pulse reception**

### Root Cause Analysis
```c
// Current: dest micro rings use site lens (for input encoding)
// But pulses arrive as phase perturbations, not harmonic inputs
// Need: lens profile that CONVERTS pulse phase → harmonic magnitude
```

### Milestones
| Week | Milestone | Success Criteria |
|------|-----------|------------------|
| 1 | Characterize pulse→harmonic transfer function | Map: pulse phase/strength → output harmonic mags |
| 2 | Design "pulse-receptive" lens profiles | > 80% harmonic detection |
| 3 | Test macro aggregation with optimized lenses | Aggregation score > 0.9 |
| 4 | Cross-vQPU feature routing demo | vQPU 0 feature → vQPU 1 macro assembly |

### Key Experiments
```c
// Pulse reception characterization
void run_pulse_reception_sweep() {
    for each lens_type in {site, harmonic, gaussian, custom}:
        for pulse_strength in [0.1, 0.3, 0.5, 0.8]:
            for pulse_phase in [0, π/4, π/2, ...]:
                measure output harmonic mags[1..8]
    // Build lookup: lens_type × pulse_params → harmonic_response
}

// Optimized lens design
void run_macro_aggregation_optimized() {
    // Use pulse-receptive lenses on vQPU 1 macro 0
    // Target: all 4 harmonics detected > 0.1 magnitude
}
```

### Decision Unblocked
- "Can macro rings share features across vQPUs?" → **Yes, with right lenses**
- "What's the macro→macro pulse protocol?" → **Defined by reception lens**

---

## Trajectory 3: Super Ring Attention Training
**Department**: dept-super + dept-attention-routing  
**Unblocks**: Intelligent cross-vQPU routing, global coherence  
**Leverage**: Current 50% (chance) → target 85%+ routing accuracy

### Current Problem
- Super ring attention weights = uniform (no learning)
- Routing = random, not task-dependent
- No Hebbian mechanism on macro harmonic correlations

### Milestones
| Week | Milestone | Success Criteria |
|------|-----------|------------------|
| 1 | Implement attention weight update (Hebbian) | Weights correlate with macro harmonic similarity |
| 2 | Task-dependent routing: filter→filter, memory→memory | Routing accuracy > 80% |
| 3 | Feedback modulation: super→macro attention weights | Macro feature selection gain > 5x |
| 4 | Cross-vQPU super↔super protocol | 3 vQPUs coordinate on task |

### Key Experiments
```c
// Hebbian attention learning
void run_super_attention_hebbian() {
    // ΔW[dst] = η * Σ_h (macro_src[h] * macro_dst[h]) - λ * W[dst]
    // Measure: weight alignment with task similarity
}

// Task-routing benchmark
void run_super_routing_tasks() {
    // Task 1: vQPU 0 filter → vQPU 1 filter (harmonic 1)
    // Task 2: vQPU 0 memory → vQPU 2 memory (harmonic 2)
    // Task 3: vQPU 0 attention → vQPU 3 attention
    // Measure: correct routing %
}
```

### Decision Unblocked
- "How does the cluster decide where to send data?" → **Learned attention**
- "Can super rings coordinate without central controller?" → **Yes, via Hebbian**

---

## Trajectory 4: Application Demonstration
**Department**: dept-cluster  
**Unblocks**: Funding, hardware commitment, external collaboration  
**Leverage**: Proves the architecture does something useful

### Target Applications (Ranked by Fit)
| Application | Why It Fits | vQPU Allocation | Success Metric |
|-------------|-------------|-----------------|----------------|
| **Sensor Fusion** (2-modal) | Natural micro↔macro hierarchy | 2 vQPUs: vision + audio | Classification > 90% |
| **Temporal Sequence Learning** | Plasticity + super feedback | 4 vQPUs: encode, predict, attend, decode | Prediction accuracy > 85% |
| **Anomaly Detection** | Pulse sparsity = event detection | 2 vQPUs: normal model + monitor | F1 > 0.9 |
| **MNIST-like Pattern Rec** | Hierarchical features map to digits | 8 vQPUs: edge→shape→digit | Accuracy > 95% |

### Milestones
| Week | Milestone | Success Criteria |
|------|-----------|------------------|
| 1-2 | Sensor fusion: 2-modality classification on synthetic data | > 90% accuracy |
| 3-4 | Sequence learning: predict next pattern from history | > 85% accuracy |
| 5-6 | Anomaly detection: learn normal, flag deviation | F1 > 0.9 |
| 7-8 | MNIST-like: hierarchical feature extraction | > 95% accuracy |

### Key Experiments
```c
// In cluster_validate.c
void run_sensor_fusion_demo() {
    // vQPU 0: modality A (e.g., spatial pattern)
    // vQPU 1: modality B (e.g., temporal pattern)
    // Super ring fuses → classification
    // Sweep: SNR, correlation, latency
}

void run_sequence_learning_demo() {
    // vQPU 0: encode sequence
    // vQPU 1: predict next (plasticity on macro)
    // vQPU 2: attention gate (super)
    // vQPU 3: decode prediction
}
```

### Decision Unblocked
- "Does this actually work on real tasks?" → **Yes, here's the demo**
- "What cluster size for X application?" → **Empirical scaling laws**

---

## Trajectory 5: FPGA Hardware Projection
**Department**: dept-cluster + dept-pulse-fabric  
**Unblocks**: Physical realization, power/area budget, pinout  
**Leverage**: Turns simulation into buildable spec

### Current Data (from validation)
| Parameter | Value | Source |
|-----------|-------|--------|
| Events/sec/link | 5,556 | micro→micro |
| Bits/event | ~2.3 (harmonic) | pulse encoding |
| Links/vQPU | 16 (micro) + 16 (macro) + 1 (super) | topology |
| Precision | 2.32 bits (thresh=0.2) | analog_precision_bits |

### Milestones
| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1 | Detailed event rate model (all levels) | Events/sec per link type |
| 2 | Pin count estimation (LVDS vs GPIO vs analog) | FPGA pinout spreadsheet |
| 3 | BRAM/DSP/power estimation | Resource budget table |
| 4 | 4-vQPU FPGA prototype spec | Complete build plan |

### Key Calculations
```python
# Projected for 16-vQPU cluster:
micro_events = 5_556 * 16 * 16   # 1.4M events/sec
macro_events = 500 * 16 * 16     # 128K events/sec (est)
super_events = 50 * 16           # 800 events/sec (est)

# Bandwidth:
total_bps = (micro_events * 32 + macro_events * 64 + super_events * 128)  # bits
# ≈ 50 Mbps total — trivial for single FPGA transceiver

# Pin count (LVDS pairs):
tx_rx_pairs = 16_vqpu * (16_micro + 16_macro + 1_super) / 8  # muxed
# ≈ 66 LVDS pairs — fits in mid-range FPGA
```

### Decision Unblocked
- "Can we build this?" → **Yes, here's the FPGA spec**
- "What's the power/area?" → **Quantified budget**

---

## Dependency Graph

```
TRAJECTORY 1 (Plasticity)
    │
    ├── enables → TRAJECTORY 2 (Macro lens optimization via plasticity)
    ├── enables → TRAJECTORY 3 (Attention weight learning)
    └── enables → TRAJECTORY 4 (Task adaptation in demos)

TRAJECTORY 2 (Macro Aggregation)
    │
    ├── enables → TRAJECTORY 3 (Better macro features for attention)
    └── enables → TRAJECTORY 4 (Cross-vQPU feature routing in demos)

TRAJECTORY 3 (Super Attention)
    │
    └── enables → TRAJECTORY 4 (Intelligent routing in demos)

TRAJECTORY 4 (Demos)
    │
    └── validates → TRAJECTORY 5 (Real workload for FPGA sizing)

TRAJECTORY 5 (FPGA)
    │
    └── standalone: can start in parallel with Trajectory 1
```

---

## Recommended Execution Order

| Phase | Weeks | Trajectories | Parallel? |
|-------|-------|--------------|-----------|
| **Foundation** | 1-4 | 1 (Plasticity) + 5 (FPGA projection) | ✅ Yes |
| **Integration** | 5-8 | 2 (Macro) + 3 (Super Attention) | ✅ Yes |
| **Validation** | 9-12 | 4 (Demos) | Depends on 1-3 |

### Parallelization Strategy
- **Plasticity** (dept-plasticity) and **FPGA projection** (dept-cluster) are independent → run simultaneously
- **Macro fix** (dept-macro) and **Super attention** (dept-super) share pulse fabric → coordinate weekly
- **Demos** require all three → gate on integration phase completion

---

## Resource Requirements

| Trajectory | Lead Dept | Support Depts | Compute | Human |
|------------|-----------|---------------|---------|-------|
| 1. Plasticity | dept-plasticity | dept-micro, dept-lens | Low (sim) | 1-2 |
| 2. Macro Fix | dept-macro | dept-pulse-fabric, dept-micro | Low | 1 |
| 3. Super Attention | dept-super | dept-attention-routing, dept-macro | Low | 1 |
| 4. Demos | dept-cluster | ALL | Medium | 2-3 |
| 5. FPGA | dept-cluster | dept-pulse-fabric | Low | 1 |

---

## Success Metrics (End of 12 Weeks)

| Metric | Target | Current |
|--------|--------|---------|
| Plasticity convergence | < 1000 chunks | N/A |
| Macro aggregation score | > 0.9 | 0.25 |
| Super routing accuracy | > 85% | 50% |
| Sensor fusion accuracy | > 90% | N/A |
| FPGA pin count (16 vQPU) | < 200 | N/A |
| Cluster GOPS/W | > 500 | 678 (single) |

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Plasticity doesn't converge | Medium | High | Start with simpler Oja (no competition), add competition incrementally |
| Macro lens optimization fails | Low | Medium | Characterize pulse reception first — data-driven design |
| Super attention learns wrong routing | Medium | Medium | Constrain with task labels initially (semi-supervised) |
| Demos too slow on sim | Low | Low | Use scaled-down (4 vQPU) for rapid iteration |
| FPGA projection unrealistic | Low | High | Cross-check with inference_lead's BQSM FPGA work |

---

*Priority list ratified: 2025-07-25 | Review weekly at IRC sync | Update DESIGN_REGISTRY.csv as designs complete*