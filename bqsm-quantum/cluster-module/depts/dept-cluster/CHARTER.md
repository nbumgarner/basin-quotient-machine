# Department Charter: dept-cluster

## Mission
Design, scale, and validate **neuromorphic clusters of complete hierarchical vQPUs** — the system-level integration layer.

## Scope
- Cluster topologies (mesh, tree, small-world, hierarchical)
- Cross-vQPU routing (micro↔micro, macro↔macro, super↔super)
- Scaling laws (perf vs N, energy vs N, comms vs N)
- Application mapping (sensor fusion, pattern rec, sequence learning)
- Hardware projection (FPGA pins, BRAM, power, area)

## Current State (from cluster_validate.c)
| Experiment | Result | Implication |
|------------|--------|-------------|
| micro→micro transfer | 98% fidelity | Pulse coupling VIABLE |
| macro aggregation | 25% (1/4 harmonics) | Needs better dest lens |
| super routing | 50% (chance) | Attention needs training |
| independent states | 100% fidelity | N vQPUs run different modes |
| scale scaling | Linear to 16 | Cluster scales well |

## Deliverables
| Design | Description | Target |
|--------|-------------|--------|
| VQPU-CLUST-EXP-v1 | 5-experiment suite (current) | ✅ Done |
| VQPU-CLUST-MESH-vN | Mesh topology + adaptive routing | Q3 2025 |
| VQPU-CLUST-APP-vN | Sensor fusion / pattern rec demo | Q4 2025 |
| VQPU-CLUST-FPGA-vN | FPGA projection + pinout | Q1 2026 |

## Interface Contract
```c
// Cluster config
typedef struct {
    int num_vqpus;              // 2, 4, 8, 16
    topology_t topology;        // MESH, TREE, RING, SMALL_WORLD
    routing_policy_t policy;    // SHORTEST_PATH, ATTENTION_GATED, FLOOD
    int max_hops;               // TTL for pulses
} cluster_config_t;

// Cluster produces:
double cluster_throughput;      // events/sec
double cluster_energy_per_op;   // pJ
double app_accuracy;            // task-specific
```

## Dependencies
- **Consumes from ALL**: dept-micro, dept-macro, dept-super, dept-lens, dept-pulse-fabric, dept-plasticity, dept-attention-routing
- **Provides to**: Application layer (sensor fusion, inference)

## Validation Suite (from cluster_validate.c)
- `test_cluster_topology.c` — MESH/TREE/RING scaling
- `test_cluster_routing.c` — policy comparison
- `test_cluster_app_sensor_fusion.c` — 2-modality classification
- `test_cluster_app_sequence.c` — temporal pattern learning
- `test_cluster_fpga_projection.c` — pins, BRAM, power

## Metrics Dashboard
| Metric | Target | BASE-v1 |
|--------|--------|---------|
| Scaling efficiency (N=16) | > 0.9 | 1.0 (linear) |
| Cross-vQPU routing accuracy | > 90% | 50% (super) / 25% (macro) |
| Energy/op (cluster) | < 100pJ | ~1nJ (est) |
| FPGA pin count (16 vQPU) | < 200 | ~50 (sparse) |
| Application accuracy | > 90% | N/A |

## Starter Design: VQPU-CLUST-EXP-v1
Current `cluster_validate.c` with 5 experiments

## Known Issues (BASE-v1)
1. **Macro aggregation weak** — 25% harmonic detection
2. **Super routing untrained** — 50% = chance
3. **No adaptive topology** — static ring only
4. **No application demo** — pure infrastructure
5. **Hardware projection rough** — event rate estimates only

## Research Priorities
1. **Fix macro aggregation** — optimize destination lens for pulse reception
2. **Train super attention** — Hebbian on macro harmonic correlations
3. **Topology exploration** — small-world for diameter reduction
4. **Application demo** — MNIST-like or sensor fusion
5. **FPGA projection** — detailed resource estimation

---

*Charter ratified: 2025-07-25 | Department lead: Precision2 (compunerd)*