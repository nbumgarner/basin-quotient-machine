# Neuromorphic vQPU Cluster — Concept Framework

## Core Premise
Each vQPU (4096 oscillators, 3-level hierarchy) is a **programmable harmonic filter bank with nonlinear thresholding**. Connect N vQPUs via **event-driven phase-lock pulses** to form a neuromorphic cluster where:
- **Lens profiles = synaptic weights** (adapt via local correlation)
- **Hierarchical attention = routing** (micro→macro→super feedback)
- **Sparse events = communication** (not continuous analog coupling)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NEUROMORPHIC vQPU CLUSTER (N vQPUs)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   vQPU 0                    vQPU 1                    vQPU N-1              │
│  ┌─────────┐              ┌─────────┐              ┌─────────┐             │
│  │ SUPER   │◄────────────►│ SUPER   │◄────────────►│ SUPER   │  ← Global   │
│  │ (16 MR) │  Attention   │ (16 MR) │  Attention   │ (16 MR) │  routing    │
│  └────┬────┘              └────┬────┘              └────┬────┘             │
│       │                        │                        │                  │
│       ▼                        ▼                        ▼                  │
│  ┌─────────┐              ┌─────────┐              ┌─────────┐             │
│  │ MACRO   │◄────────────►│ MACRO   │◄────────────►│ MACRO   │  ← Pattern  │
│  │ (16 µR) │  Feedback    │ (16 µR) │  Feedback    │ (16 µR) │  assembly   │
│  └────┬────┘              └────┬────┘              └────┬────┘             │
│       │                        │                        │                  │
│       ▼                        ▼                        ▼                  │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    EVENT-DRIVEN INTERCONNECT FABRIC                 │  │
│  │  • Phase-lock pulses (threshold-triggered)                          │  │
│  │  • Sparse: only active rings transmit                               │  │
│  │  • Targeted: pulse routed to specific ring in target vQPU           │  │
│  │  • Plastic: receiver lens adapts via local STDP-like rule           │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Abstractions

| Level | Oscillators | Function | Communicates |
|-------|-------------|----------|--------------|
| **Micro ring** | 16 | Harmonic filter (lens-programmable) | Pulses when |q| crosses threshold |
| **Macro ring** | 256 (16 µR) | Feature assembly, cross-µR coupling | Attention weights to other vQPU macro |
| **Super ring** | 4096 (16 MR) | Global context, routing decisions | Attention weights to other vQPU super |

---

## Event-Driven Communication Protocol

### Pulse Generation (Sender)
```c
// In each micro ring, after each settle chunk:
double q = measure_winding(ring);  // winding number ∈ [-3, 3]
if (fabs(q - ring->last_q) > PULSE_THRESHOLD) {
    emit_pulse(target_vqpu_id, target_ring_id, q, ring->id);
    ring->last_q = q;
}
```

### Pulse Reception (Receiver)
```c
// Perturbs receiver ring phases (like a lens kick)
void receive_pulse(ring_t *r, double sender_phase, double strength) {
    for (int i = 0; i < OSC_PER_RING; i++) {
        r->th[i] += strength * sin(sender_phase - r->th[i]);
    }
    // This shifts the ring's attractor basin → changes filter output
}
```

### Why This Works
| Property | Benefit |
|----------|---------|
| **Threshold-triggered** | Only rings with meaningful state changes communicate |
| **Phase-encoded** | Single float (winding number) carries the signal |
| **Lens-mediated effect** | Receiver's lens profile determines how pulse affects computation |
| **Sparse** | ~1-5% of rings pulse per settle cycle (empirical) |

---

## Plasticity Rule: Lens Adaptation via Local Correlation

### Mathematical Formulation

**State variables per micro ring:**
- `lens[i]` — lens profile (16 values, one per oscillator site)
- `best_harmonic` — harmonic this ring currently selects best (1-8)
- `pre_trace[t]` — exponentially decaying trace of incoming pulses
- `post_trace[t]` — exponentially decaying trace of own filter output

**Plasticity update (per settle chunk):**
```
Δlens[i] = η * (pre_trace[i] * post_trace[best_harmonic] - λ * lens[i])

where:
  pre_trace[i]  = α * pre_trace[i] + (1-α) * pulse_received_at_site_i
  post_trace[h] = α * post_trace[h] + (1-α) * |c_h|  (filter output magnitude)
  η = learning rate (0.001 - 0.01)
  λ = weight decay for homeostasis
  α = trace decay (0.9 - 0.99)
```

### Interpretation
- **Hebbian**: "Fire together, wire together" — if a pulse from site `i` correlates with strong output at harmonic `h`, strengthen `lens[i]` for that harmonic
- **Competitive**: Other harmonics get slight depression (`-0.1` factor)
- **Homeostatic**: Weight decay prevents runaway lens growth

### Site-Harmonic Mapping
Each lens site `i` contributes to multiple harmonics via Fourier basis:
```
lens_profile[i] = Σ_h lens_coeff[h] * cos(2π*h*i/N + φ[h])
```
Plasticity operates on `lens_coeff[h]` — the harmonic coefficients of the lens.

---

## Hierarchical Attention Routing

### Macro-Level Feedback (vQPU_A ← vQPU_B)
```
vQPU_B macro rings compute feature vectors (16 harmonics × 16 rings = 256 values)
vQPU_A super ring computes attention weights over vQPU_B macro rings:
    weight[m] = softmax( similarity(vQPU_A_super_query, vQPU_B_macro_keys[m]) )
Top-k macro rings from vQPU_B → routed to vQPU_A micro rings for detailed processing
```

### Micro-Level Detail (within vQPU)
```
vQPU_A micro rings receive routed macro features
→ Apply lens to extract specific harmonics (filter operation)
→ Results feed back to vQPU_A macro for pattern completion
```

---

## Energy/Scale Projections

| Cluster Size | vQPUs | Total Oscillators | Static Power | Event Comms | Total Power | GOPS | GOPS/W |
|--------------|-------|-------------------|--------------|-------------|-------------|------|--------|
| **Single** | 1 | 4,096 | 8 mW | 0 | **8 mW** | 5.5 | 678 |
| **Small** | 4 | 16,384 | 32 mW | 2 mW | **34 mW** | 22 | 647 |
| **Medium** | 16 | 65,536 | 128 mW | 8 mW | **136 mW** | 88 | 647 |
| **Large** | 64 | 262,144 | 512 mW | 32 mW | **544 mW** | 352 | 647 |
| **Full** | 256 | 1,048,576 | 2 W | 128 mW | **2.1 W** | 1,408 | 647 |

**Assumptions:**
- Event comms: 5% rings active × 1 pulse/settle × 4 bytes/pulse × 5.5 MHz settle rate
- Efficiency stays ~650 GOPS/W because communication is sparse and analog

---

## Research Questions to Validate

1. **Event-driven coupling works**: 2-vQPU simulation shows information transfer via pulses
2. **Plasticity converges**: Lens profiles self-organize to match input statistics
3. **Hierarchical attention routes correctly**: Super ring selects relevant macro features
4. **Cluster scales**: N-vQPU performance ∝ N (linear scaling)
5. **Applications emerge**: Pattern recognition, sequence learning, sensor fusion

---

## File Structure (Proposed)

```
bqsm-quantum/
├── neuromorphic_cluster/
│   ├── CLUSTER_FRAMEWORK.md          # This file
│   ├── cluster_sim.c                 # N-vQPU event-driven simulator
│   ├── plasticity.c                  # Lens adaptation rules
│   ├── attention_routing.c           # Hierarchical attention
│   ├── two_vqpu_validate.c           # 2-vQPU coupling test
│   ├── cluster_bench.c               # Benchmarks: pattern rec, seq learning
│   ├── hardware_mapping.c            # I/O, power, area for N-vQPU chip
│   └── docs/
│       ├── PLASTICITY_MATH.md        # Formal plasticity derivation
│       ├── ATTENTION_ROUTING.md      # Attention mechanism spec
│       └── EVENT_PROTOCOL.md         # Pulse protocol spec
```

---

## Next Steps (Implementation Order)

1. **Two-vQPU validation** — Minimal simulator proving pulse-based info transfer
2. **Plasticity math** — Formal derivation + convergence analysis
3. **Cluster simulator** — N-vQPU with event fabric, plasticity, attention
4. **Benchmarks** — MNIST-like pattern rec, temporal sequence learning
5. **Hardware spec** — I/O, power, area for 16-vQPU cluster chip

---

**Status**: Concept framework complete. Ready for validation experiments.