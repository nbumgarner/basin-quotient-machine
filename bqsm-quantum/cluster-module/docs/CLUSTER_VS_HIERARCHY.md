# Neuromorphic vQPU Cluster — Architecture Clarification

## The Distinction

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HIERARCHICAL vQPU (Single)                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  SUPER RING (16 macro rings = 4096 osc)                             │   │
│  │    ┌─────────────────────────────────────────────────────────┐      │   │
│  │    │  MACRO RING 0 (16 micro rings = 256 osc)                │      │   │
│  │    │    ┌─────────────────────────────────────────────────┐  │      │   │
│  │    │    │  MICRO RING 0 (16 oscillators)                  │  │      │   │
│  │    │    │  MICRO RING 1 (16 oscillators)                  │  │      │   │
│  │    │    │  ...                                            │  │      │   │
│  │    │    │  MICRO RING 15 (16 oscillators)                 │  │      │   │
│  │    │    └─────────────────────────────────────────────────┘  │      │   │
│  │    │  MACRO RING 1 ... MACRO RING 15                       │      │   │
│  │    └─────────────────────────────────────────────────────────┘      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  Internal coupling: Continuous Kuramoto (sin phase diffs)                  │
│  State space: (SU(4) × S¹) per ring → astronomical total                   │
└─────────────────────────────────────────────────────────────────────────────┘

        ▲                    ▲                    ▲                    ▲
        │  NEUROMORPHIC      │  NEUROMORPHIC      │  NEUROMORPHIC      │
        │  EVENT FABRIC      │  EVENT FABRIC      │  EVENT FABRIC      │
        │  (sparse pulses)   │  (sparse pulses)   │  (sparse pulses)   │
        ▼                    ▼                    ▼                    ▼

┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│    vQPU 0        │  │    vQPU 1        │  │    vQPU 2        │  │    vQPU N-1      │
│  (4096 osc)      │  │  (4096 osc)      │  │  (4096 osc)      │  │  (4096 osc)      │
│  3-level hier.   │  │  3-level hier.   │  │  3-level hier.   │  │  3-level hier.   │
│  Independent     │  │  Independent     │  │  Independent     │  │  Independent     │
│  state space     │  │  state space     │  │  state space     │  │  state space     │
└──────────────────┘  └──────────────────┘  └──────────────────┘  └──────────────────┘
```

**Key**: Each vQPU is a COMPLETE hierarchical machine. The cluster connects them via a **different substrate** — sparse, event-driven, neuromorphic pulses — not by extending the ring hierarchy.

---

## Why This Matters

| Aspect | Hierarchical (Internal) | Neuromorphic Cluster (External) |
|--------|------------------------|---------------------------------|
| **Coupling** | Continuous, all-to-all within ring | Sparse, threshold-triggered pulses |
| **Topology** | Fixed 16×16×16 grid | Arbitrary graph (mesh, tree, random) |
| **State** | Single unified attractor landscape | N independent attractor landscapes |
| **Communication** | Phase diffusion (analog) | Phase-lock events (digital-like) |
| **Plasticity** | Lens profiles (slow, local) | Lens + routing weights (Hebbian + attention) |
| **Scaling** | Limited by I/O (12K lines) | Limited by event bandwidth (much higher) |

---

## Cluster Capabilities the Hierarchy Alone Cannot Do

1. **Independent simultaneous states**: vQPU 0 in filter mode, vQPU 1 in attention mode, vQPU 2 in memory mode — all at once
2. **Dynamic reconfiguration**: Rewire cluster topology without changing internal vQPU structure
3. **Fault isolation**: One vQPU fails → others continue
4. **Heterogeneous specialization**: Different vQPUs with different quantization/lens configs
5. **Temporal binding**: Phase-lock pulses synchronize specific harmonics across vQPUs
6. **Attention routing**: Super rings gate which vQPU's macro outputs get processed where

---

## Simulation Must Reflect This

The current `two_vqpu_validate.c` simulates 2 vQPUs with only **1 ring each** (micro ring only). That's wrong.

**Correct simulation structure per vQPU:**
```c
typedef struct {
    micro_ring_t  micro_rings[16];     // 16 × 16 osc
    macro_ring_t  macro_rings[16];     // 16 macro rings
    super_ring_t  super_rings[1];      // 1 super ring
} vqpu_t;
```

**Inter-vQPU connections:**
- Micro→Micro: Direct harmonic filter output pulses
- Macro→Macro: Feature assembly pulses (attention-gated)
- Super→Super: Global context/routing pulses

---

## Updated Experiment Matrix

| Experiment | What It Tests | Cluster-Specific |
|------------|---------------|------------------|
| **Pulse transfer** | Micro ring → micro ring across vQPUs | ✓ |
| **Macro aggregation** | 16 micro → 1 macro within vQPU, then macro→macro across vQPUs | ✓ |
| **Super routing** | Super ring attention weights gate cross-vQPU macro traffic | ✓ |
| **Independent states** | vQPU 0 filter, vQPU 1 memory, vQPU 2 attention simultaneously | ✓ |
| **Plasticity** | Lens adapts within vQPU; routing weights adapt across vQPUs | ✓ |
| **Scale** | N vQPUs (each 4096 osc), not N rings | ✓ |

---

## Next Implementation

1. **Fix `two_vqpu_validate.c`** to simulate full hierarchical vQPUs
2. **Add inter-vQPU pulse routing** at micro/macro/super levels
3. **Measure**: Can vQPU 0's macro features route to vQPU 1's micro filters via super attention?

This is the actual cluster research tool — not a 4th ring level.