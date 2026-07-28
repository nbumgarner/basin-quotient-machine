# Design Naming & Research Department System

## Design Naming Convention

### Format: `VQPU-{COMPONENT}-{VARIANT}-v{VERSION}`

| Component | Code | Description |
|-----------|------|-------------|
| **Micro Ring** | `MIC` | 16-oscillator filter unit |
| **Macro Ring** | `MAC` | 16 micro rings (256 osc) |
| **Super Ring** | `SUP` | 16 macro rings (4096 osc) |
| **Lens System** | `LENS` | Lens profile generator/optimizer |
| **Pulse Fabric** | `PULSE` | Event-driven inter-vQPU communication |
| **Plasticity** | `PLST` | Lens adaptation / Hebbian learning |
| **Attention Router** | `ATTN` | Super ring routing weights |
| **Full vQPU** | `VQPU` | Complete 16³ hierarchical machine |
| **Cluster** | `CLUST` | N-vQPU neuromorphic cluster |

### Variant Codes
| Variant | Code | Meaning |
|---------|------|---------|
| Baseline | `BASE` | Reference implementation |
| Optimized | `OPT` | Performance tuned |
| Low-power | `LP` | Minimal energy |
| High-precision | `HP` | Maximum accuracy |
| Sparse | `SPR` | Event-sparse operation |
| Quantized | `Q{level}` | Q4, Q8, Q16, etc. |
| Experimental | `EXP` | Research prototype |

### Examples
```
VQPU-MIC-BASE-v1       # Baseline micro ring
VQPU-MIC-OPT-v3        # 3rd optimization of micro ring
VQPU-LENS-SPR-v2       # Sparse lens system
VQPU-PULSE-Q8-v1       # 8-bit quantized pulse fabric
VQPU-VQPU-BASE-v1      # Complete baseline vQPU
VQPU-CLUST-EXP-v4      # 4th experimental cluster design
```

---

## Design Registry (Living Document)

### Current Registered Designs

| Design ID | Component | Variant | Version | Status | Owner | Notes |
|-----------|-----------|---------|---------|--------|-------|-------|
| VQPU-MIC-BASE-v1 | Micro | BASE | 1 | Active | Dept-Micro | From hierarchical_computability.c |
| VQPU-MAC-BASE-v1 | Macro | BASE | 1 | Active | Dept-Macro | Aggregation logic |
| VQPU-SUP-BASE-v1 | Super | BASE | 1 | Active | Dept-Super | Attention routing |
| VQPU-LENS-BASE-v1 | Lens | BASE | 1 | Active | Dept-Lens | 5 types: uniform, site, harmonic, gaussian, alt |
| VQPU-PULSE-BASE-v1 | Pulse | BASE | 1 | Active | Dept-Comm | Input-driven emission |
| VQPU-VQPU-BASE-v1 | Full | BASE | 1 | Active | Dept-Integration | cluster_validate.c |
| VQPU-CLUST-EXP-v1 | Cluster | EXP | 1 | Active | Dept-Cluster | 5-experiment suite |

### Registration Process
1. Create design in appropriate department folder
2. Run validation suite (CSV output)
3. Register in `DESIGN_REGISTRY.csv` with metrics
4. Tag git commit: `design/VQPU-MIC-OPT-v2`

---

## Research Departments

### Department Structure
```
cluster-module/
├── depts/
│   ├── dept-micro/          # Micro ring: filter physics, lens response
│   ├── dept-macro/          # Macro ring: aggregation, feature assembly
│   ├── dept-super/          # Super ring: attention, global routing
│   ├── dept-lens/           # Lens system: profile design, optimization
│   ├── dept-comm/           # Pulse fabric: encoding, routing, bandwidth
│   ├── dept-plst/           # Plasticity: Hebbian, homeostasis, convergence
│   ├── dept-integration/    # Full vQPU assembly, cross-level coupling
│   └── dept-cluster/        # Multi-vQPU: topology, scaling, apps
```

### Department Charters

#### Dept-Micro: "The Filter Foundry"
- **Scope**: Single 16-oscillator ring physics
- **Deliverables**:
  - Lens profile → harmonic response maps
  - Quantization impact curves (Q4→float)
  - Amplitude threshold characterization (A≈3.0)
  - Settle time vs accuracy tradeoffs
- **Key Metrics**: Filter selectivity (|c_target|/|c_other|), latency, energy/op
- **Owns**: `VQPU-MIC-*-v*` designs

#### Dept-Macro: "The Feature Assembler"
- **Scope**: 16 micro rings → 256 oscillator macro ring
- **Deliverables**:
  - Cross-micro coupling schemes
  - Macro harmonic aggregation functions
  - Feature binding across micro rings
  - Intra-vQPU pulse routing (micro→macro)
- **Key Metrics**: Aggregation fidelity, feature separation, pulse efficiency
- **Owns**: `VQPU-MAC-*-v*` designs

#### Dept-Super: "The Global Router"
- **Scope**: 16 macro rings → 4096 oscillator super ring
- **Deliverables**:
  - Attention weight computation
  - Inter-vQPU routing decisions
  - Global state coherence
  - Super→macro feedback (top-down attention)
- **Key Metrics**: Routing accuracy, attention entropy, feedback stability
- **Owns**: `VQPU-SUP-*-v*` designs

#### Dept-Lens: "The Profile Architect"
- **Scope**: Lens profile generation, optimization, quantization
- **Deliverables**:
  - Profile library: site, harmonic, gaussian, custom
  - Quantization-aware lens design (Q4→float)
  - Lens profile → filter response predictor
  - Automated lens synthesis from spec
- **Key Metrics**: Profile diversity, quantization robustness, synthesis speed
- **Owns**: `VQPU-LENS-*-v*` designs

#### Dept-Comm: "The Pulse Fabric"
- **Scope**: Event-driven inter-vQPU communication
- **Deliverables**:
  - Pulse encoding schemes (phase, harmonic, amplitude)
  - Routing protocols (unicast, multicast, broadcast)
  - Bandwidth/latency characterization
  - Error correction for analog pulses
- **Key Metrics**: Bits/pulse, events/sec, error rate, pin count
- **Owns**: `VQPU-PULSE-*-v*` designs

#### Dept-Plst: "The Plasticity Engine"
- **Scope**: Lens adaptation, Hebbian learning, homeostasis
- **Deliverables**:
  - Plasticity rule implementations (Oja + competition)
  - Convergence proofs / empirical bounds
  - Catastrophic forgetting mitigation
  - Multi-timescale adaptation (fast/slow)
- **Key Metrics**: Convergence epochs, final selectivity, retention, energy stability
- **Owns**: `VQPU-PLST-*-v*` designs

#### Dept-Integration: "The vQPU Assembler"
- **Scope**: Complete 16³ hierarchical vQPU assembly
- **Deliverables**:
  - Cross-level coupling (micro↔macro↔super)
  - Full validation suite (cluster_validate.c)
  - Quantization configuration (7 levels)
  - Power/area/I/O budget
- **Key Metrics**: End-to-end accuracy, GOPS/W, I/O lines, config space
- **Owns**: `VQPU-VQPU-*-v*` designs

#### Dept-Cluster: "The Neuromorphic Cluster"
- **Scope**: N-vQPU clusters, topology, applications
- **Deliverables**:
  - Cluster topologies (mesh, tree, small-world)
  - Scaling laws (perf vs N)
  - Application demos (sensor fusion, pattern rec)
  - FPGA/ASIC projection
- **Key Metrics**: Cluster efficiency (GOPS/W), scaling linearity, app accuracy
- **Owns**: `VQPU-CLUST-*-v*` designs

---

## Cross-Department Interfaces

| Interface | Producer | Consumer | Contract |
|-----------|----------|----------|----------|
| Micro→Macro | Dept-Micro | Dept-Macro | 16 harmonic magnitudes per micro ring |
| Macro→Super | Dept-Macro | Dept-Super | 16 macro harmonic vectors |
| Super→Macro (feedback) | Dept-Super | Dept-Macro | 16 attention weights |
| Lens→Micro | Dept-Lens | Dept-Micro | 16-site lens profile |
| Pulse→All | Dept-Comm | All | Phase-lock pulse event |
| Plasticity→Lens | Dept-Plst | Dept-Lens | Updated lens coefficients |

---

## Design Review Process

### Weekly Sync (IRC #emerging.systems)
- Each department presents: new designs, metrics, blockers
- Cross-department dependencies resolved
- Design registry updated

### Design Gate Criteria (to promote BASE→OPT)
| Gate | Requirement |
|------|-------------|
| **Functional** | Passes unit tests + integration test |
| **Measured** | CSV output from validation suite |
| **Documented** | Design doc in dept folder |
| **Reviewed** | 1+ other department sign-off |
| **Registered** | Entry in DESIGN_REGISTRY.csv |

---

## File Structure per Department

```
depts/dept-micro/
├── CHARTER.md              # This department's scope
├── designs/
│   ├── VQPU-MIC-BASE-v1/
│   │   ├── design.md       # Design specification
│   │   ├── micro_ring.c    # Implementation
│   │   ├── test_mic  │   ├── test_mic.c    # Unit tests
│   │   └── metrics.csv     # Validation results
│   └── VQPU-MIC-OPT-v2/
│       └── ...
├── shared/
│   ├── lens_profiles.h     # Standard lens definitions
│   └── physics_constants.h # DT, SETTLE_STEPS, etc.
└── REGISTRY.md             # Local design log
```

---

## Getting Started

### For New Department Member
```bash
cd cluster-module/depts/dept-micro
cat CHARTER.md
cat designs/VQPU-MIC-BASE-v1/design.md
# Run baseline
cc -O3 designs/VQPU-MIC-BASE-v1/micro_ring.c -o test -lm
./test
```

### To Propose New Design
1. Fork from BASE in your department
2. Implement + test
3. Document in `design.md`
4. Run cross-department integration test
5. Submit PR with registry entry

---

## Version Control Strategy

```
git tag design/VQPU-MIC-OPT-v2
git tag design/VQPU-CLUST-EXP-v3
```

- Each design version = git tag
- BASE designs on `main` branch
- EXP/OPT designs on feature branches
- Integration tests run on merge to `main`

---

## Metrics Dashboard (Auto-generated)

Run `./generate_dashboard.py` → `dashboard.html` with:
- Per-design metric trends
- Cross-department dependency graph
- Scaling projections
- Hardware cost estimates

---

*This system enables parallel, specialized research while maintaining integration coherence. Each department owns its component end-to-end; cross-cutting concerns (plasticity, pulse fabric) have dedicated departments.*