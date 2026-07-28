# Research Trajectories Implementation Plan

## Overview
Execute the 5 priority research trajectories for the neuromorphic vQPU cluster, each decomposed into specific, reviewable tasks with 2-stage review (spec compliance → code quality).

---

## Trajectory 1: Plasticity Engine (HIGH — highest leverage)

### Task 1.1: Implement Plasticity State in micro_ring_t
**File**: `src/cluster_validate.c`
**Spec**: Add `plasticity_state_t` struct to `micro_ring_t` with:
- `double lens_coeff[8]` — harmonic basis coefficients
- `double pre_trace[16]` — pulse arrival traces per site  
- `double post_trace[8]` — output harmonic traces
- `int best_harmonic` — competitive winner
- `double lens_energy` — for normalization

### Task 1.2: Implement plasticity_update() Function
**File**: `src/cluster_validate.c`
**Spec**: Function called after each settle chunk:
1. Project pre_trace → harmonic basis: `pre_proj[h] = (2/16) * Σ_i pre_trace[i] * cos(2πhi/16)`
2. Update coefficients: `Δcoeff[h] = η * pre_proj[h] * post_trace[h] - λ * coeff[h]`
3. Reconstruct lens: `lens[i] = Σ_h coeff[h] * cos(2πhi/16)`
4. Normalize energy: if Σ lens[i]² > E_max, scale all
5. Decay traces: `pre_trace *= α`, `post_trace *= α`

### Task 1.3: Integrate Plasticity into Settlement Loop
**File**: `src/cluster_validate.c`
**Spec**: Call `plasticity_update()` after each settle chunk in all experiments. Initialize plasticity state in `init_vqpu()`.

### Task 1.4: Add Learning Curve Validation Experiment
**File**: `src/cluster_validate.c`
**Spec**: New experiment `run_plasticity_learning_curve()`:
- Sweep: η ∈ [0.001, 0.01], λ ∈ [1e-5, 1e-3], α ∈ [0.9, 0.99]
- Measure: convergence_chunks, final_selectivity, stability_score
- Output CSV row per parameter set

### Task 1.5: Add Catastrophic Forgetting Test
**File**: `src/cluster_validate.c`
**Spec**: New experiment `run_plasticity_forgetting()`:
- Task A: learn harmonic 1 filter (1000 chunks)
- Task B: learn harmonic 2 filter (1000 chunks)
- Measure: Task A retention (harmonic 1 selectivity) after B training
- Target: > 80% retention

### Task 1.6: Add Multi-Ring Plasticity (Macro Level)
**File**: `src/cluster_validate.c`
**Spec**: Extend plasticity to macro ring level — aggregate micro plasticities for feature assembly.

---

## Trajectory 2: Macro Aggregation Fix (HIGH — unblocks cross-vQPU features)

### Task 2.1: Characterize Pulse→Harmonic Transfer Function
**File**: `src/cluster_validate.c`
**Spec**: New experiment `run_pulse_reception_sweep()`:
- For each lens_type ∈ {site, harmonic, gaussian, custom}:
- For pulse_strength ∈ [0.1, 0.3, 0.5, 0.8]:
- For pulse_phase ∈ [0, π/4, π/2, 3π/4, π]:
- Measure output harmonic mags[1..8]
- Build lookup table: lens_type × pulse_params → harmonic_response

### Task 2.2: Design Pulse-Receptive Lens Profiles
**File**: `src/cluster_validate.c` + `depts/dept-lens/designs/`
**Spec**: Create 4 new lens profiles in `make_pulse_receptive_lens()`:
- `PULSE_RECEPTIVE_K1`: maximizes harmonic 1 from phase perturbation
- `PULSE_RECEPTIVE_K2`: maximizes harmonic 2
- `PULSE_RECEPTIVE_K3`: maximizes harmonic 3
- `PULSE_RECEPTIVE_K4`: maximizes harmonic 4

### Task 2.3: Fix Macro Aggregation Cross-vQPU
**File**: `src/cluster_validate.c`
**Spec**: Modify `run_macro_aggregation_cross()`:
- Use pulse-receptive lenses on destination macro rings (vQPU 1 macro 0)
- Verify all 4 harmonics detected > 0.1 magnitude
- Target aggregation_score > 0.9 (currently 0.25)

### Task 2.4: Implement Weighted Feature Assembly
**File**: `src/cluster_validate.c`
**Spec**: Macro ring aggregation = weighted average by pulse strength, not simple mean. Add `feature_weight` to macro_ring_t.

---

## Trajectory 3: Super Attention Routing (HIGH — enables intelligent routing)

### Task 3.1: Implement Hebbian Attention Weight Learning
**File**: `src/cluster_validate.c`
**Spec**: Update `super_ring_t.routing_weights[MAX_VQPUS]` after each chunk:
```
ΔW[dst] = η * Σ_h (macro_src[h] * macro_dst[h]) - λ * W[dst]
```
- η = 0.01, λ = 0.001
- Only update when macro_src and macro_dst both active

### Task 3.2: Task-Dependent Routing Benchmark
**File**: `src/cluster_validate.c`
**Spec**: New experiment `run_super_routing_tasks()`:
- Task 1: vQPU 0 filter (harmonic 1) → vQPU 1 filter
- Task 2: vQPU 0 memory (harmonic 2) → vQPU 2 memory  
- Task 3: vQPU 0 attention → vQPU 3 attention
- Measure: correct routing % per task
- Target: > 85% overall

### Task 3.3: Cross-vQPU Super↔Super Protocol
**File**: `src/cluster_validate.c`
**Spec**: Implement `super_pulse_t` with ACK/NACK:
- Fields: src_vqpu, dst_vqpu, weight, harmonic, seq_num, ack_requested
- Sender: emit with ack_requested=1
- Receiver: respond with ACK pulse (same seq_num)
- Sender: resend if no ACK within 2 chunks
- Timeout: drop after 3 retries

### Task 3.4: Super→Macro Feedback Modulation
**File**: `src/cluster_validate.c`
**Spec**: Super ring computes attention weights → modulates macro ring coupling:
```c
mr->inter_coupling = base_coupling * (1.0 + attention_weight[m])
```
Test: attention gain > 5x for attended macro, < 0.5x for unattended.

---

## Trajectory 4: Application Demos (MEDIUM — grant evidence)

### Task 4.1: Sensor Fusion Demo (2 Modality)
**File**: `src/cluster_validate.c` + new demo file
**Spec**: 
- vQPU 0: "vision" modality (spatial pattern on harmonics 1-4)
- vQPU 1: "audio" modality (temporal pattern on harmonics 1-4)  
- Super ring: fuses via attention → classification (harmonic 5 = class)
- Target: > 90% accuracy on synthetic correlated patterns

### Task 4.2: Temporal Sequence Learning
**File**: `src/cluster_validate.c`
**Spec**:
- vQPU 0: encode sequence (harmonic sequence = time steps)
- vQPU 1: predict next (plasticity on macro level)
- vQPU 2: attention gate (super ring)
- vQPU 3: decode prediction
- Target: > 85% next-step prediction accuracy

### Task 4.3: Anomaly Detection
**File**: `src/cluster_validate.c`
**Spec**:
- vQPU 0: learn normal pulse patterns (plasticity converges)
- vQPU 1: monitor incoming pulses → flag deviations
- Metric: F1 > 0.9 on injected anomalies

---

## Trajectory 5: FPGA Hardware Projection (MEDIUM — buildable spec)

### Task 5.1: Detailed Event Rate Model
**File**: `depts/dept-cluster/designs/VQPU-CLUST-FPGA-v1/`
**Spec**: From validation data, compute per-link event rates:
- micro→micro: 5,556 events/sec/link
- macro→macro: estimate from pulse counts
- super→super: estimate from attention pulses
- Total for 16 vQPUs: events/sec, bandwidth, peak

### Task 5.2: FPGA Resource Budget
**File**: `depts/dept-cluster/designs/VQPU-CLUST-FPGA-v1/fpga_budget.md`
**Spec**: Project for 16-vQPU cluster:
- LVDS pairs: 66 (16 vQPU × 33 links / 8 muxed)
- BRAM: 2 MB (event buffers: 1.4M events × 16 bytes)
- DSP slices: 0 (no DSP needed, pure logic)
- Power: < 2W (static 136mW + event-driven ~200mW)
- Logic cells: ~50K for pulse fabric + routing

### Task 5.3: Pinout & Interface Spec
**File**: `depts/dept-cluster/designs/VQPU-CLUST-FPGA-v1/pinout.csv`
**Spec**: Complete pin assignment for target FPGA (e.g., Artix-7 200T):
- Clock: 100 MHz differential
- Event fabric: 66 LVDS TX/RX pairs
- Debug: 8 GPIO, 1 UART
- Power: 1.0V, 1.8V, 3.3V rails

### Task 5.4: 4-vQPU Prototype Spec
**File**: `depts/dept-cluster/designs/VQPU-CLUST-FPGA-v1/prototype_4vqpu.md`
**Spec**: Minimal viable prototype:
- 4 vQPUs = 16,384 oscillators
- 2 sensor modalities
- Single FPGA (Artix-7 100T)
- Target: demonstrate sensor fusion > 90%

---

## Execution Order & Dependencies

```
Phase 1 (Foundation):     T1.1 → T1.2 → T1.3 → T1.4 → T1.5
                          T2.1 → T2.2
Phase 2 (Integration):    T1.6 ← T2.3 ← T2.4
                          T3.1 → T3.2 → T3.3 → T3.4
Phase 3 (Validation):     T4.1, T4.2, T4.3 (demos)
                          T5.1 → T5.2 → T5.3 → T5.4 (FPGA)
```

**Parallelizable**: T1.x, T2.x can run concurrently (different files)
**Sequential**: T3 depends on T1.3 (plasticity) + T2.3 (macro features)

---

## Review Gates

Each task passes through:
1. **Spec Compliance Review** — matches task spec exactly
2. **Code Quality Review** — style, tests, edge cases, security
3. Only when both PASS → mark complete, proceed to next task

Final integration review after all tasks complete.