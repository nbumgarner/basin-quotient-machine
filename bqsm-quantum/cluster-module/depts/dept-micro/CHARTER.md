# Department Charter: dept-micro

## Mission
Characterize, optimize, and deliver the **16-oscillator micro ring** — the fundamental filter unit of the vQPU hierarchy.

## Scope
- Harmonic filter physics (Kuramoto + lens)
- Lens profile → frequency response mapping
- Quantization impact (binary → float64)
- Amplitude nonlinearity (threshold A≈3.0)
- Settle time vs accuracy tradeoffs

## Key Findings (from hierarchical_computability.c, filter_characterization.c)
| Lens Type | Filter Character | Best Harmonic |
|-----------|------------------|---------------|
| site (site0_±0.5) | Band-pass | k=1 |
| site (site4_+0.5) | Through-pass | all |
| harmonic (cos_k1) | Resonant | k=2 (30x) |
| gaussian (center) | Low-pass | k=1 |
| uniform | Block | none |

## Deliverables
| Design | Description | Status |
|--------|-------------|--------|
| VQPU-MIC-BASE-v1 | From hierarchical_computability.c | ✅ Validated |
| VQPU-MIC-OPT-vN | Optimized settle (1 chunk = 60μs) | 🔄 |
| VQPU-MIC-Q4-vN | Quantized lens/state (int4) | 🔄 |
| VQPU-MIC-LP-vN | Low-power (int8, reduced steps) | 🔄 |

## Interface Contract
```c
// Micro ring produces:
double harmonic_magnitudes[8];  // |c_1| ... |c_8|

// Consumes:
double lens_profile[16];        // Per-site omega offset
double input_phases[16];        // Initial condition encoding
```

## Dependencies
- **Provides to**: dept-macro (harmonic mags), dept-super (via macro)
- **Consumes from**: dept-lens (lens profiles), dept-plasticity (adapted lenses), dept-pulse-fabric (incoming pulses)

## Validation Suite (from filter_characterization.c, filter_selectivity.c)
- `test_micro_filter_sweep.c` — lens type × harmonic × amplitude
- `test_micro_quantization.c` — Q4..float64 accuracy
- `test_micro_threshold.c` — amplitude nonlinearity scan
- `test_micro_settle.c` — chunks vs accuracy

## Metrics Dashboard
| Metric | Target | BASE-v1 |
|--------|--------|---------|
| Filter selectivity (k1/k2) | > 10x | 3.8x (site lens) |
| Settle time | 1 chunk (60μs) | 3 chunks |
| Quantization (int8) loss | < 5% | ~15% |
| Threshold sharpness | > 5x jump | ~20x (A=3.0) |
| Energy/op | < 1nJ | 0.36μJ (sim) |

## Starter Design: VQPU-MIC-BASE-v1
From `hierarchical_computability.c` — 16 osc, RK4 settle, 5 lens types

## Known Issues (BASE-v1)
1. **3 chunks default** — 1 chunk sufficient (per proactive_experiment.c)
2. **Lens resolution** — 0.01 steps, could be adaptive
3. **No pulse interface** — add pulse reception handler
3. **Quantization noise** — int4 unusable, int8 marginal

## Research Priorities
1. **1-chunk settle validation** — confirm accuracy at 60μs
2. **Lens library expansion** — learned profiles from dept-plasticity
3. **Pulse response characterization** — phase perturbation transfer function
4. **Quantization-aware lens design** — optimize for Q8/Q4
5. **Energy measurement** — correlate sim steps to physical energy

---

*Charter ratified: 2025-07-25 | Department lead: Precision2*