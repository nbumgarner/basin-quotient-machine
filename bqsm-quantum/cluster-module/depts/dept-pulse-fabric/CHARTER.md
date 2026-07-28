# Department Charter: dept-pulse-fabric

## Mission
Design, validate, and optimize the **neuromorphic event fabric** — sparse, threshold-triggered phase-lock pulses between vQPUs at micro/macro/super levels.

## Scope
- Pulse emission policies (input-driven vs state-driven)
- Pulse encoding (phase, harmonic, amplitude → bits)
- Routing fabric (mesh, tree, small-world topologies)
- Latency/bandwidth/energy characterization
- Pulse reception and phase perturbation dynamics

## Deliverables
| Design | Description | Validation |
|--------|-------------|------------|
| VQPU-PULSE-BASE-v1 | Input-driven emission from cluster_validate.c | Transfer ratio 0.98 |
| VQPU-PULSE-ENC-vN | Multi-bit encoding (harmonic + phase) | > 3 bits/pulse |
| VQPU-PULSE-ROUT-vN | Multi-hop routing with TTL | < 5μs hop latency |
| VQPU-PULSE-HW-vN | FPGA/ASIC interface spec | Pin count, BRAM, power |

## Interface Contract
```c
// Emission (from any level)
typedef struct {
    int src_vqpu, src_level, src_idx;
    double phase;           // Winding number q ∈ [-3,3]
    double strength;        // 0.1-1.0
    int harmonic;           // 1-8 (which harmonic this represents)
    int ttl;                // Max hops
    uint64_t timestamp;     // For ordering
} pulse_t;

// Reception (by target level)
void receive_pulse(micro_ring_t *ring, pulse_t *pulse) {
    // Phase perturbation: θ[i] += strength * sin(phase - θ[i])
}
```

## Dependencies
- **Uses**: dept-micro/macro/super (pulse sources/destinations)
- **Used by**: ALL departments (inter-vQPU communication)

## Validation Suite
- `test_pulse_emission.c` — trigger policies, rate control
- `test_pulse_encoding.c` — bits/pulse, error rate
- `test_pulse_routing.c` — topology, latency, delivery rate
- `test_pulse_reception.c` — phase perturbation fidelity

## Metrics Dashboard
| Metric | Target | Current (BASE-v1) |
|--------|--------|-------------------|
| Transfer ratio (micro→micro) | > 0.9 | 0.98 |
| Bits per pulse | > 3 | ~2.3 (harmonic only) |
| Events/sec/link | > 10K | 5.5K |
| Latency (src→dst) | < 10μs | ~5μs (1 chunk) |
| Energy per pulse | < 1pJ | ~0.1pJ (est) |
| Analog precision (bits) | > 4 | 2.3→0.5 (thresh dep) |

## Starter Design: VQPU-PULSE-BASE-v1
From `cluster_validate.c` — input-driven emission, phase-encoded

## Known Issues (BASE-v1)
1. **No phase noise model** — perfect sin perturbation unrealistic
2. **Single harmonic encoding** — wasting phase channel
3. **Fixed topology** — ring only, no adaptive routing
4. **No congestion control** — pulse queue can overflow
5. **No pulse acknowledgment** — unreliable delivery

## Research Priorities
1. **Phase+amplitude encoding** — 2-3 bits/pulse
2. **Stochastic pulse emission** — rate-coded for analog values
3. **Adaptive routing** — topology discovery + shortest path
4. **Hardware mapping** — LVDS/SerDes vs GPIO vs analog lines
5. **Pulse compression** — batch multiple rings into single event

---

*Charter ratified: 2025-07-25 | Department lead: Precision2 (compunerd)*