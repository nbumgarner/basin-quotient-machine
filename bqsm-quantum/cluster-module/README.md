# Neuromorphic vQPU Cluster Module

## Quick Start
```bash
cd src
cc -O3 -std=c11 cluster_validate.c -o cluster_validate -lm
./cluster_validate 2>/dev/null > results.csv
```

## What This Is
A **neuromorphic cluster** of complete hierarchical vQPUs (each 16³ = 4096 oscillators with 3-level micro/macro/super hierarchy), connected via **event-driven phase-lock pulses** — NOT a 4th ring level.

See `docs/CLUSTER_VS_HIERARCHY.md` for the critical distinction.

## Current Experiments (in `cluster_validate.c`)

| Experiment | What It Tests | Key Metric |
|------------|---------------|------------|
| `micro_to_micro` | Single ring → single ring across vQPUs | `transfer_ratio` (0.98 achieved) |
| `macro_aggregation_cross` | 16 micro → macro (intra) → macro (inter) | `macro_aggregation_score` (0.25) |
| `super_routing` | Super ring attention routes macro→macro | `super_routing_accuracy` (0.50) |
| `independent_states` | 3 vQPUs: filter, memory, attention simultaneously | `independent_state_fidelity` (1.0) |
| `scale_scaling` | 2→4→8→16 vQPUs ring topology | `routing_accuracy` (speedup) |

## Output CSV Schema
```
experiment,num_vqpus,src_level,dst_level,pulse_threshold,pulse_strength,
transfer_ratio,routing_accuracy,macro_aggregation_score,super_routing_accuracy,
independent_state_fidelity,events_per_sec,bandwidth_mbps,analog_precision_bits
```

## Key Files
| File | Purpose |
|------|---------|
| `src/cluster_validate.c` | Main simulation (5 experiments, CSV output) |
| `src/two_vqpu_validate.c` | Legacy single-ring validation |
| `docs/CLUSTER_VS_HIERARCHY.md` | Architecture clarification |
| `docs/PLASTICITY_RULE_DESIGN.md` | Lens adaptation math (Hebbian + homeostasis) |
| `docs/NEUROMORPHIC_CLUSTER_FRAMEWORK.md` | Full concept framework |
| `AGENT_ORIENTATION.md` | **Read this first** — entry point for agents/humans |

## Design Decisions Pending Data
| Decision | Experiment Needed |
|----------|-------------------|
| Pulse trigger: threshold vs input-driven | `micro_to_micro` sweep |
| Plasticity timescale | Implement `PLASTICITY_RULE_DESIGN.md` then sweep |
| Routing granularity: micro/macro/super | Compare modes in `super_routing` |
| Hardware target: FPGA pins, BRAM | `scale_scaling` + `hw_projection` |

## Collaboration
IRC: #emerging.systems on 132.145.153.167:6667 (key: bqsmlab)
- Precision2 (compunerd) — architecture, experiments
- Precision1 — mathematical model
- inference_lead — BQSM inference, transformer mapping
- Fold6 — channel participant

## Next Steps
1. Implement plasticity rule from `PLASTICITY_RULE_DESIGN.md`
2. Add parameter sweeps to all experiments
3. Build analysis script: `python analyze.py --csv results/*.csv --plot`
4. Project to FPGA: events/sec → pins, BRAM, DSP slices