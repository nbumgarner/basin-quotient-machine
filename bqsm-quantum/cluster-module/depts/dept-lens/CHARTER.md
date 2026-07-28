# Department Charter: dept-lens

## Mission
Design, catalog, and optimize **lens profiles** — the per-oscillator omega offsets that program micro ring filter behavior. The lens IS the weight matrix of the vQPU.

## Scope
- Lens profile library: site, harmonic, gaussian, custom learned
- Lens → harmonic response mapping (characterization)
- Quantization-aware lens design (Q4 → float64)
- Automated lens synthesis from filter spec
- Lens profile quantization and compression

## Key Findings (from filter_characterization.c, filter_selectivity.c, superior_model.c)
| Lens Type | Profile Resolution | Filter Character | Best Harmonic | Enhancement |
|-----------|-------------------|------------------|---------------|-------------|
| uniform | 0.0 | Block | none | 0x |
| site (site0_±0.5) | 0.01 | Band-pass | k=1 | 10-24x |
| site (site4_+0.5) | 0.01 | Through-pass | all | 1x |
| harmonic (cos_k1) | 0.01 | Resonant | k=2 | 30x |
| gaussian (center) | 0.05 | Low-pass | k=1 | ~1x |
| alternating | 0.01 | Block | none | 0x |

## Deliverables
| Design | Description | Validation |
|--------|-------------|------------|
| VQPU-LENS-BASE-v1 | 5 canonical profiles | Characterized |
| VQPU-LENS-LIB-vN | 64+ profiles (hybrid, learned) | Cataloged |
| VQPU-LENS-Q4-vN | Quantized for int4 | < 10% loss |
| VQPU-LENS-SYNTH-vN | Spec → profile optimizer | Automated |
| VQPU-LENS-PLST-vN | Plasticity-compatible (basis coeffs) | Converges |

## Interface Contract
```c
// Lens profile (16 sites per micro ring)
double lens_profile[16];  // omega offsets per oscillator

// Harmonic basis representation (8 coeffs for plasticity)
double lens_coeff[8];     // lens[i] = Σ_h coeff[h] * cos(2πhi/16)

// Characterization output
typedef struct {
    double harmonic_gain[8];    // |c_h| for unit input at each h
    double selectivity[8][8];   // gain[h_in][h_out]
    double threshold_amp;       // Nonlinearity onset
} lens_characterization_t;
```

## Dependencies
- **Uses**: dept-micro (characterization), dept-plasticity (basis coeffs), dept-macro (macro lens)
- **Used by**: dept-micro (filter programming), dept-pulse-fabric (pulse-receptive lenses), dept-plasticity (learning target)

## Validation Suite
- `test_lens_characterize.c` — Full sweep: type × harmonic × amplitude
- `test_lens_quantize.c` — Q4, Q8, Q16, float32, float64 comparison
- `test_lens_synthesize.c` — Spec → profile optimizer
- `test_lens_pulse_receptive.c` — NEW: pulse phase → harmonic response

## Metrics Dashboard
| Metric | Target | BASE-v1 |
|--------|--------|---------|
| Filter selectivity (k1/k2) | > 10x | 3.8x (site) |
| Profile library size | > 64 | 5 |
| Quantization loss (int8) | < 5% | ~15% |
| Synthesis time | < 1ms | N/A |
| Pulse-receptive gain | > 0.8 | N/A |

## Starter Design: VQPU-LENS-BASE-v1
From `superior_model.c` — 5 types, `make_lens()` function

## Known Issues (BASE-v1)
1. **Only 5 profiles** — need 64+ for plasticity
2. **No quantization-aware design** — Q4 unusable, Q8 marginal
3. **No pulse-receptive profiles** — macro aggregation broken
4. **No synthesis tool** — manual profile design only
5. **Plasticity uses site basis** — should use harmonic basis (8 coeffs vs 16 sites)

## Research Priorities
1. **Pulse-receptive lens design** — fixes Trajectory 2 (macro aggregation)
2. **Quantization-aware lens optimization** — enable Q4/Q8 operation
3. **Harmonic basis representation** — 8 coeffs for plasticity (vs 16 sites)
4. **Automated synthesis** — filter spec → lens profile
5. **Profile interpolation** — continuous lens space for fine control

---

*Charter ratified: 2025-07-25 | Department lead: Precision2 (compunerd)*