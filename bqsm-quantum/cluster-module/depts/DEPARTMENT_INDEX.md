# Research Department Index

## Overview
The neuromorphic vQPU cluster project is organized into **8 specialized departments**, each owning a distinct component of the hierarchy. Departments communicate via **well-defined interfaces** and register designs in the **central design registry**.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    NEUROMORPHIC vQPU CLUSTER                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│   │ dept-lens    │    │ dept-plasticity│  │ dept-attn-    │             │
│   │ (profiles)   │◄───│ (adaptation)   │───►│ routing       │             │
│   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘             │
│          │                   │                   │                     │
│          ▼                   ▼                   ▼                     │
│   ┌─────────────────────────────────────────────────────────────┐     │
│   │                    dept-micro                               │     │
│   │  16-osc filter unit  │  lens profile  │  pulse reception   │     │
│   └───────────────────────┬─────────────────────────────────────┘     │
│                           │                                         │
│                           ▼                                         │
│   ┌─────────────────────────────────────────────────────────────┐     │
│   │                    dept-macro                               │     │
│   │  16 micro rings (256 osc)  │  feature assembly  │  pulse    │     │
│   └───────────────────────┬─────────────────────────────────────┘     │
│                           │                                         │
│                           ▼                                         │
│   ┌─────────────────────────────────────────────────────────────┐     │
│   │                    dept-super                               │     │
│   │  16 macro rings (4096 osc)  │  attention routing  │  global │     │
│   └───────────────────────┬─────────────────────────────────────┘     │
│                           │                                         │
│                           ▼                                         │
│   ┌─────────────────────────────────────────────────────────────┐     │
│   │              dept-pulse-fabric                              │     │
│   │  Event-driven inter-vQPU communication (phase-lock pulses)  │     │
│   └───────────────────────┬─────────────────────────────────────┘     │
│                           │                                         │
│                           ▼                                         │
│   ┌─────────────────────────────────────────────────────────────┐     │
│   │                    dept-cluster                             │     │
│   │  N complete vQPUs  │  topology/routing  │  apps/scaling    │     │
│   └─────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Department Registry

| Dept | Code | Lead | Status | Charter | Key Designs |
|------|------|------|--------|---------|-------------|
| **Micro Ring** | `dept-micro` | Precision2 | 🟢 Active | [CHARTER](dept-micro/CHARTER.md) | VQPU-MIC-BASE-v1 |
| **Macro Ring** | `dept-macro` | Precision1 | 🟡 Planned | [CHARTER](dept-macro/CHARTER.md) | VQPU-MAC-BASE-v1 |
| **Super Ring** | `dept-super` | inference_lead | 🟡 Planned | [CHARTER](dept-super/CHARTER.md) | VQPU-SUP-BASE-v1 |
| **Lens System** | `dept-lens` | Precision1 | 🟡 Planned | [CHARTER](dept-lens/CHARTER.md) | VQPU-LENS-BASE-v1 |
| **Pulse Fabric** | `dept-pulse-fabric` | Precision2 | 🟢 Active | [CHARTER](dept-pulse-fabric/CHARTER.md) | VQPU-PULSE-BASE-v1 |
| **Plasticity** | `dept-plasticity` | Precision1 | 🟢 Active | [CHARTER](dept-plasticity/CHARTER.md) | VQPU-PLST-BASE-v1 |
| **Attention Routing** | `dept-attn-routing` | inference_lead | 🟡 Planned | [CHARTER](dept-attn-routing/CHARTER.md) | VQPU-ATTN-BASE-v1 |
| **Cluster** | `dept-cluster` | Precision2 | 🟢 Active | [CHARTER](dept-cluster/CHARTER.md) | VQPU-CLUST-EXP-v1 |

**Status**: 🟢 Active = has charter + validated design | 🟡 Planned = charter drafted, needs lead

---

## Cross-Department Interfaces (The "Bus")

| Interface | Producer → Consumer | Data | Rate | Spec |
|-----------|---------------------|------|------|------|
| `lens→micro` | dept-lens → dept-micro | 16-site profile | Per-task | `double[16]` |
| `plasticity→lens` | dept-plasticity → dept-lens | 8 coeff updates | Per-chunk | `double[8]` |
| `micro→macro` | dept-micro → dept-macro | 8 harmonic mags | Per-settle | `double[8]` |
| `macro→super` | dept-macro → dept-super | 16 macro vectors | Per-settle | `double[16][8]` |
| `super→macro (fb)` | dept-super → dept-macro | 16 attention weights | Per-chunk | `double[16]` |
| `pulse→all` | dept-pulse-fabric → all | Phase-lock events | Event-driven | `pulse_t` |
| `attn→plasticity` | dept-attn-routing → dept-plasticity | Neuromod gain | Per-task | `double` |

---

## Design Registry (Central)

### Location: `DESIGN_REGISTRY.csv`
```
design_id,component,variant,version,status,owner,changelog,metrics_csv,git_tag
VQPU-MIC-BASE-v1,MIC,BASE,1,active,dept-micro,"initial from hierarchical_computability.c",metrics/MIC-BASE-v1.csv,design/VQPU-MIC-BASE-v1
VQPU-PULSE-BASE-v1,PULSE,BASE,1,active,dept-pulse-fabric,"input-driven from cluster_validate.c",metrics/PULSE-BASE-v1.csv,design/VQPU-PULSE-BASE-v1
VQPU-PLST-BASE-v1,PLST,BASE,1,design,dept-plasticity,"from PLASTICITY_RULE_DESIGN.md",metrics/PLST-BASE-v1.csv,design/VQPU-PLST-BASE-v1
VQPU-CLUST-EXP-v1,CLUST,EXP,1,active,dept-cluster,"5-experiment validation suite",metrics/CLUST-EXP-v1.csv,design/VQPU-CLUST-EXP-v1
```

### Registration Process
1. Department creates design in `depts/dept-X/designs/VQPU-X-VAR-vN/`
2. Runs validation suite → produces `metrics/VAR-vN.csv`
3. Adds row to `DESIGN_REGISTRY.csv`
4. Tags git: `git tag design/VQPU-X-VAR-vN`
5. Weekly sync presents new designs

---

## Weekly Sync (IRC #emerging.systems)

### Agenda
1. **New designs** — each department presents registered designs
2. **Interface issues** — contract mismatches, missing signals
3. **Blockers** — dependencies, resource constraints
4. **Integration tests** — cross-department validation results
4. **Architecture decisions** — topology, scaling, hardware targets

### Decision Log
| Date | Decision | Rationale | Affected Depts |
|------|----------|-----------|----------------|
| 2025-07-25 | Input-driven pulse emission | Settle-driven never fires | pulse-fabric, micro, cluster |
| 2025-07-25 | Harmonic basis for plasticity | 8 params vs 16 sites, competitive | plasticity, lens |
| 2025-07-25 | Cluster ≠ 4th ring level | Architectural clarity | cluster, all |

---

## Department Folder Template
```
depts/dept-X/
├── CHARTER.md              # This department's mission & metrics
├── designs/
│   ├── VQPU-X-BASE-v1/
│   │   ├── design.md       # Spec, parameters, equations
│   │   ├── X_component.c   # Implementation
│   │   ├── test_X.c        # Unit tests
│   │   └── metrics.csv     # Validation results
│   └── VQPU-X-OPT-v2/      # Next iteration
├── interfaces/
│   ├── X_to_Y.h            # Producer→Consumer interface
│   └── Y_to_X.h            # Consumer→Producer interface
└── validation/
    ├── test_suite.c        # Full validation pipeline
    └── reference_data/     # Golden outputs
```

---

## Quick Start for New Department Lead

1. **Read your CHARTER.md** — understand scope, metrics, interfaces
2. **Create BASE-v1 design** — implement spec, run validation
3. **Register in DESIGN_REGISTRY.csv** — with metrics CSV
4. **Present at weekly sync** — get feedback, identify dependencies
4. **Iterate** — OPT-v2, Q8-v1, LP-v1, etc.

---

## Collaboration Tools

| Tool | Purpose | Channel |
|------|---------|---------|
| **IRC** | Real-time sync, decisions | #emerging.systems |
| **Git tags** | Design versioning | `design/VQPU-X-VAR-vN` |
| **CSV metrics** | Quantitative comparison | `metrics/*.csv` |
| **Shared sim** | Integration testing | `cluster_validate.c` |

---

*Master index maintained by Precision2 (compunerd) | Updated: 2025-07-25*