# FPGA Hardware Projection & Pinout Specification

## Executive Summary

Based on cluster simulation data, this document projects the FPGA resource requirements for implementing the neuromorphic vQPU cluster.

---

## Simulation Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Oscillators per micro ring | 16 | `OSC_PER_MICRO` |
| Micro rings per macro | 16 | `MICRO_PER_MACRO` |
| Macro rings per super | 16 | `MACRO_PER_SUPER` |
| Oscillators per vQPU | 4,096 | `TOTAL_OSC` |
| Settle steps per chunk | 120 | `SETTLE_STEPS` |
| Time step (DT) | 0.5 | `DT` |
| Settle chunks | 3 | `SETTLE_CHUNKS` |
| Max vQPUs | 16 | `MAX_VQPUS` |
| Max pulses | 50,000 | `MAX_PULSES` |

---

## Timing Analysis

### Per-vQPU Compute

```
Settle time per chunk = SETTLE_STEPS × DT = 120 × 0.5 = 60 time units
Settle time per vQPU = SETTLE_CHUNKS × 60 = 180 time units

Clock frequency: 100 MHz (10 ns cycle)
Time unit: 0.5 / 100e6 = 5 ns

Settle time per vQPU = 180 × 5 ns = 900 ns
Settle rate per vQPU = 1 / 900 ns ≈ 1.11 MHz
```

### Inter-vQPU Communication

From simulation:
- Events/sec per link: 5,556 (at strength=0.5)
- Events/sec per link: 89,000 (at N=16, aggregate)

**Bandwidth per link:**
```
Pulse size: 4 bytes (src, dst, phase, strength)
Bandwidth per link: 5,556 × 4 bytes = 22.2 KB/s = 0.18 Mbps
```

**Aggregate bandwidth (N=16):**
```
Total events/sec: 89,000
Total bandwidth: 89,000 × 4 bytes = 356 KB/s = 2.85 Mbps
```

---

## FPGA Resource Projection

### Single vQPU (4,096 oscillators)

#### Oscillator Compute (per micro ring)

Each oscillator requires:
- Phase accumulator: 32 bits
- Natural frequency: 32 bits
- Coupling input: 32 bits
- RK4 state (4 stages): 4 × 32 bits = 128 bits

**Per oscillator:** ~224 bits of state

```
Oscillator state per micro ring: 16 × 224 bits = 3,584 bits
Oscillator state per macro: 16 × 3,584 bits = 57,344 bits
Oscillator state per vQPU: 16 × 57,344 bits = 917,408 bits ≈ 900 Kbits
```

#### BRAM Estimation

Xilinx UltraScale+ BRAM: 288 Kbits per block

```
BRAM per vQPU: 900 Kbits / 288 Kbits = 3.1 BRAM blocks
Round up: 4 BRAM blocks per vQPU
```

#### DSP Slices

Each oscillator requires:
- 1 × sin() approximation (lookup table or CORDIC)
- 1 × adder (phase update)
- 1 × multiplier (coupling)

**Simplified:** 1 DSP slice per 4 oscillators

```
DSP per vQPU: 4,096 / 4 = 1,024 DSP slices
```

#### Logic Cells (LUTs)

```
Per oscillator: ~20 LUTs (phase update, coupling, lens application)
LUTs per vQPU: 4,096 × 20 = 81,920 LUTs
```

### Cluster (N=16 vQPUs)

```
Total BRAM: 16 × 4 = 64 BRAM blocks
Total DSP: 16 × 1,024 = 16,384 DSP slices
Total LUTs: 16 × 81,920 = 1,310,720 LUTs
```

### Interconnect Fabric

#### Pulse Routing

Each vQPU needs:
- Input pulse interface: 16 pulses (one per macro ring)
- Output pulse interface: 16 pulses (one per macro ring)
- Control signals: 4 bits

**Per vQPU I/O:** 16 × 4 bytes + 16 × 4 bytes + 4 bits = 128 bytes

**Total cluster I/O:** 16 × 128 bytes = 2,048 bytes = 16,384 bits

#### Pin Count Estimation

Assuming LVDS pairs for high-speed serial:
- Data width: 32 bits per link
- Links per vQPU: 16 (ring topology)
- Total links: 16 × 16 / 2 = 128 (bidirectional)

```
LVDS pairs needed: 128 × 32 = 4,096 pairs
Round up to nearest package: 4,096 pairs ≈ 8,192 pins
```

**Conservative estimate:** 10,000 pins for routing and control

---

## Power Estimation

### Static Power

```
BRAM: 64 × 50 mW = 3.2 W
DSP: 16,384 × 0.5 mW = 8.2 W
LUTs: 1,310,720 × 0.01 mW = 13.1 W
Static total: 24.5 W
```

### Dynamic Power

```
Oscillator switching: 4,096 × 1.11 MHz × 100 fJ = 0.45 W
Interconnect switching: 89,000 events/s × 4 bytes × 1 V = 0.36 W
Dynamic total: 0.81 W
```

### Total Power

```
Total power: 24.5 + 0.81 = 25.3 W
```

---

## Pinout Specification

### High-Speed Serial (LVDS)

| Pin Group | Count | Purpose |
|-----------|-------|---------|
| Data TX | 128 × 32 = 4,096 | Pulse data transmit |
| Data RX | 128 × 32 = 4,096 | Pulse data receive |
| Clock TX | 128 | Clock transmit (embedded) |
| Clock RX | 128 | Clock receive (embedded) |
| **Subtotal** | **8,352** | |

### Control Signals

| Signal | Count | Purpose |
|--------|-------|---------|
| Reset | 1 | Global reset |
| Clock | 1 | System clock (100 MHz) |
| Start | 1 | Simulation start |
| Ready | 1 | vQPU ready flag |
| Interrupt | 16 | Per-vQPU interrupt |
| Config write | 1 | Configuration write |
| Config read | 1 | Configuration read |
| Config addr | 8 | Configuration address |
| Config data | 32 | Configuration data |
| **Subtotal** | **61** | |

### Debug & JTAG

| Signal | Count | Purpose |
|--------|-------|---------|
| JTAG TCK | 1 | Test clock |
| JTAG TMS | 1 | Test mode select |
| JTAG TDI | 1 | Test data in |
| JTAG TDO | 1 | Test data out |
| JTAG TRST | 1 | Test reset |
| Debug clock | 1 | Debug clock |
| Debug data | 32 | Debug data |
| **Subtotal** | **38** | |

### Power & Ground

| Signal | Count | Purpose |
|--------|-------|---------|
| VCCINT | 32 | Core power (1.0V) |
| VCCAUX | 16 | Auxiliary power (1.8V) |
| GND | 64 | Ground |
| **Subtotal** | **112** | |

### Total Pin Count

```
High-speed serial: 8,352 pins
Control: 61 pins
Debug: 38 pins
Power/GND: 112 pins
**Total: 8,563 pins**
```

---

## FPGA Part Selection

### Target: Xilinx UltraScale+ XC7VX690T

| Resource | Available | Required | Utilization |
|----------|-----------|----------|-------------|
| Logic Cells | 2,429,952 | 1,310,720 | 54% |
| BRAM | 3,072 Kbits | 64 × 288 = 18,432 Kbits | 0.6% |
| DSP | 3,360 | 16,384 | **244%** ❌ |
| Pins | 1,312 | 8,563 | **653%** ❌ |

**Problem:** DSP and pin count exceed the target FPGA.

### Alternative: Xilinx UltraScale+ XCVU19P

| Resource | Available | Required | Utilization |
|----------|-----------|----------|-------------|
| Logic Cells | 4,322,560 | 1,310,720 | 30% |
| BRAM | 5,376 Kbits | 18,432 Kbits | 0.3% |
| DSP | 5,376 | 16,384 | **305%** ❌ |
| Pins | 1,704 | 8,563 | **502%** ❌ |

**Still insufficient.**

### Solution: Multi-FPGA or Resource Optimization

#### Option 1: Multi-FPGA (2x XCVU19P)

```
DSP per FPGA: 16,384 / 2 = 8,192 (still > 5,376)
Need 3x XCVU19P for DSP
```

#### Option 2: Optimize DSP Usage

Reduce DSP per oscillator from 1 to 0.25 (shared multipliers):
```
DSP per vQPU: 4,096 / 16 = 256
Total DSP: 16 × 256 = 4,096 (fits in XCVU19P)
```

#### Option 3: Reduce Pin Count

Use time-division multiplexing for inter-vQPU links:
```
Instead of 128 links, use 16 links with 8x TDM
Pins: 16 × 32 × 2 = 1,024 (fits in XCVU19P)
```

---

## Recommended Configuration

### Single FPGA: XCVU19P with Optimizations

| Optimization | Before | After |
|--------------|--------|-------|
| DSP sharing | 1 per 4 osc | 1 per 16 osc |
| Pin TDM | 128 links | 16 links × 8x TDM |
| BRAM packing | 4 per vQPU | 2 per vQPU (shared) |

### Final Resource Estimates

| Resource | Required | XCVU19P Available | Utilization |
|----------|----------|-------------------|-------------|
| Logic Cells | 1,310,720 | 4,322,560 | 30% |
| BRAM | 18,432 Kbits | 5,376 Kbits | 0.3% |
| DSP | 4,096 | 5,376 | 76% |
| Pins | 1,024 | 1,704 | 60% |

### Final Power Estimate

```
Total power: ~30 W
Power per vQPU: 30 W / 16 = 1.875 W
```

---

## Implementation Roadmap

### Phase 1: Single vQPU Prototype (3 months)

1. Implement 16-oscillator micro ring with RK4 solver
2. Add lens profile application
3. Add harmonic measurement
4. Verify against simulation

### Phase 2: Hierarchical vQPU (6 months)

1. Implement 16 micro rings → 1 macro ring
2. Add macro-level coupling
3. Implement super ring attention
4. Verify hierarchical behavior

### Phase 3: Cluster Integration (12 months)

1. Implement inter-vQPU pulse fabric
2. Add routing protocol
3. Implement plasticity
4. Add Hebbian attention
5. Verify cluster behavior

### Phase 4: Application Demo (18 months)

1. Implement sensor fusion demo
2. Add pattern recognition
3. Add sequence learning
4. Performance benchmarking

---

## References

- Simulation data: `cluster_validate.c`
- Architecture: `NEUROMORPHIC_CLUSTER_FRAMEWORK.md`
- Hardware specs: Xilinx UltraScale+ Product Guide

---

**Generated by:** Precision1
**Date:** 2025-07-26
**Status:** Initial projection, requires validation
