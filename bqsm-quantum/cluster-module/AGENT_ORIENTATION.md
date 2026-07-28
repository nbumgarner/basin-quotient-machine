# Neuromorphic vQPU Cluster — Agent Orientation

## Purpose
This folder (`cluster-module/`) is a **self-contained research branch** for exploring neuromorphic clusters of hierarchical vQPUs. It contains everything needed to run experiments, analyze results, and make architectural decisions.

**Critical distinction**: This is NOT a 4th ring level. Each vQPU is a COMPLETE 16³=4096 oscillator hierarchical machine (micro→macro→super). The cluster connects N of these via sparse, event-driven phase-lock pulses — a different substrate entirely.

---

## Quick Start

```bash
cd /home/compunerd/Downloads/basin-quotient-machine/bqsm-quantum/cluster-module/src

# Compile the cluster simulator
cc -O3 -std=c11 cluster_validate.c -o cluster_validate -lm

# Run all experiments, output CSV
./cluster_validate 2>/dev/null > results.csv

# View results
head -20 results.csv
```

---

## File Map

```
cluster-module/
├── README.md                           # This file (human readable)
├── AGENT_ORIENTATION.md                # This file (agent readable)
├── docs/
│   ├── NEUROMORPHIC_CLUSTER_FRAMEWORK.md   # Full architecture concept
│   ├── PLASTICITY_RULE_DESIGN.md           # Lens adaptation math (Hebbian + homeostasis)
│   └── CLUSTER_VS_HIERARCHY.md             # Why cluster ≠ 4th ring level
└── src/
    ├── cluster_validate.c           # MAIN: N-vQPU simulator (5 experiments)
    ├── cluster_validate             # Compiled binary
    ├── results.csv                  # Latest experimental data
    ├── two_vqpu_validate.c          # Legacy: 2-ring validation
    └── two_vqpu_validate            # Legacy binary
```

---

## Simulation: `cluster_validate.c`

### What It Simulates
- **N complete hierarchical vQPUs** (each 4096 oscillators = 16³)
- **3 levels per vQPU**: micro (16 osc) → macro (16 micro) → super (16 macro)
- **Event-driven pulses** between vQPUs at micro/macro/super levels
- **5 experiments** sweeping key design parameters

### Data Structures
```c
vqpu_t          // One complete 16³ machine
  macro_ring_t macro_rings[16]   // 16 macro rings
    micro_ring_t micro[16]       // 16 micro rings each (16 osc)
  super_ring_t super_ring        // 1 super ring (attention routing)

pulse_t         // Inter-vQPU communication
  src_vqpu, src_level, src_idx   // level: 0=micro, 1=macro, 2=super
  dst_vqpu, dst_level, dst_idx
  phase, strength, harmonic
  timestep
```

### Experiments (in order)

| Experiment | Function | Levels Tested | Parameters Swept |
|------------|----------|---------------|------------------|
| **1. micro→micro** | `run_micro_to_micro_transfer()` | micro→micro | threshold (0.2,0.5,0.7), strength (0.2,0.5,0.8) |
| **2. macro_agg** | `run_macro_aggregation_cross()` | micro→macro | fixed (pulse on Δwinding>0.5) |
| **3. super_routing** | `run_super_routing()` | macro→macro (attention-gated) | fixed |
| **4. independent** | `run_independent_states()` | NONE (isolation test) | fixed |
| **5. scale** | `run_scale_scaling()` | macro→macro (ring topology) | N=2,4,8,16 vQPUs |

### Output CSV Schema (14 columns)
```
experiment,num_vqpus,src_level,dst_level,pulse_threshold,pulse_strength,
transfer_ratio,routing_accuracy,macro_aggregation_score,super_routing_accuracy,
independent_state_fidelity,events_per_sec,bandwidth_mbps,analog_precision_bits
```

---

## Key Results (from latest run)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **micro→micro transfer** | 0.98 (98%) | Pulse coupling VIABLE |
| **events/sec/link** | 5,556 | Low but functional bandwidth |
| **analog precision** | 2.32→0.52 bits | Threshold trades precision for sparsity |
| **macro aggregation** | 25% (1/4 harmonics) | Needs better destination lens |
| **super routing** | 50% (chance) | Attention weights need plasticity |
| **independent states** | 1.0 (perfect) | N vQPUs run different modes simultaneously |
| **scale scaling** | Linear 11K→89K events/s | Cluster scales well |

---

## Decision Matrix (What the Data Tells Us)

| Design Decision | Evidence | Recommendation |
|-----------------|----------|----------------|
| Pulse trigger | Input-driven works; Δwinding fails (fixed point) | **Emit on input change** |
| Plasticity timescale | Independent states work per-chunk (~60μs) | **Per-chunk updates** |
| Routing granularity | Micro→micro 98%, macro→macro 25% | **Micro primary, macro for attention** |
| Scale target | Linear to 16 vQPUs (65K osc) | **Start 4-8, scale to 16** |
| Hardware target | 5.5K events/s/link, 2.3 bits precision | **FPGA feasible** (sparse, low pin count) |

---

## How to Extend

### Add a New Experiment
1. Add function `static void run_my_experiment(void)` in `cluster_validate.c`
2. Use `init_result()`, `print_result()` for CSV output
3. Call from `main()` after existing experiments
4. Recompile: `cc -O3 -std=c11 cluster_validate.c -o cluster_validate -lm`

### Add Plasticity (per `PLASTICITY_RULE_DESIGN.md`)
```c
// In micro_ring_t, add:
double lens_coeff[8];       // Harmonic basis coefficients
double pre_trace[16];       // Input pulse traces
double post_trace[8];       // Output harmonic traces

// Call after each settle chunk:
plasticity_update(micro_ring_t *r) {
    // Project pre_trace to harmonic basis
    // Δlens_coeff[h] = η * pre_proj[h] * post_trace[h] - λ * lens_coeff[h]
    // Reconstruct lens from coefficients
    // Normalize energy
    // Decay traces
}
```

### Add Super Ring Attention Training
```c
// In super_ring_t, routing_weights[MAX_VQPUS] are the attention weights
// Update via Hebbian on macro harmonic correlation:
routing_weights[dst] += η * (macro_harmonic(src_macro, h) * macro_harmonic(dst_macro, h))
```

### Scale to Hardware Projection
```c
// From results.csv: events_per_sec × bytes_per_event × N_vqpus
// 5,556 events/s × 4 bytes × 16 vQPUs = 355 KB/s
// FPGA: 1-2 transceivers, minimal BRAM, no DSPs needed
```

---

## Common Commands

```bash
# Clean rebuild
rm cluster_validate && cc -O3 -std=c11 cluster_validate.c -o cluster_validate -lm

# Run single experiment (edit main() to call only one)
./cluster_validate 2>/dev/null

# Save with timestamp
./cluster_validate 2>/dev/null > ../results/cluster_$(date +%Y%m%d_%H%M%S).csv

# Compare runs
diff results_20240115.csv results_20240116.csv

# Python analysis (create analyze.py)
python3 -c "
import pandas as pd
df = pd.read_csv('results.csv')
print(df[df.experiment=='micro_to_micro'].groupby('pulse_threshold')['transfer_ratio'].mean())
"
```

---

## Collaboration Context

**IRC**: #emerging.systems on 132.145.153.167:6667 (key: bqsmlab)
- **Precision2** (compunerd) — architecture, experiments, this module
- **Precision1** — mathematical model, lens theory
- **inference_lead** — BQSM inference, transformer mapping
- **Fold6** — channel participant

**Related repos**:
- `~/Downloads/basin-quotient-machine/bqsm-quantum/` — parent (hierarchical models)
- `~/phoenix-os/` — OS/inference stack (separate)

---

## Pitfalls to Avoid

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Δwinding pulse trigger | 0 pulses (fixed point reached) | Use input-driven emission |
| Dest ring has same lens as src | 98% transfer but fake (both converge) | Dest lens = 0 or different |
| Fidelity > 1.0 | Unnormalized metric | Divide by (signal + noise) |
| Macro aggregation low | 25% harmonics detected | Improve destination lens design |
| Super routing at chance | 50% accuracy | Train attention weights via plasticity |

---

## Next Research Priorities

1. **Fix macro aggregation** — destination micro rings need lens profiles that respond to incoming pulses
2. **Implement plasticity** — `PLASTICITY_RULE_DESIGN.md` → working code
3. **Train super attention** — Hebbian on macro harmonic correlations
4. **FPGA pin estimation** — from events/sec, project to real hardware
5. **Application demo** — pattern recognition or sensor fusion across cluster

---

## Version History

| Date | Change |
|------|--------|
| 2025-07-25 | Created cluster-module branch, cluster_validate.c with 5 experiments |
| 2025-07-25 | Fixed pulse emission (input-driven), macro/super routing |
| 2025-07-25 | Added CLUSTER_VS_HIERARCHY.md, PLASTICITY_RULE_DESIGN.md |

---

**This file is the entry point for any agent/human working in this branch. Update it when architecture changes.**