# BQSM fork run 04: A2 macro-ring slice, winding vs phasor readout (2026-10-10)

**System.** Two register rings A and B plus an output ring O, each N=16. They are coupled site-to-site in a chain A — O — B, the slice of the 16-ring macro ring around one ring. Code: `fork_macro.c`.

**Dynamics.** Ring r, site i: the furnace equations plus one term, ε·sin(θˢᵢ − θʳᵢ).

**Geometry.**
- `GEOM=half`: O couples to A on sites 0–7 and to B on sites 8–15.
- `GEOM=full`: all 16 sites couple.

**Instructions.** NOP; ERASE (the furnace's kick, {1.0, 2.2}); WRITE = `apply_ramp` from `bqsm_vm.c`, +1 only, widths 2–8 and 16. Each instruction targets one ring.

**Lock criterion.** Velocity lock only, the REF semantics. Patience scales as 12/ε chunks, because weak coupling relaxes on a ~1/ε timescale. At ε=0.05, alive joint states are 5, 59 and 107 of 125 at patience 12, 48 and 120.

**Trust anchor.** At ε=0, ring A's ERASE rows are identical to the furnace's `build_table`: 160/160 in both the fast build and REF.

## 1. Winding readout

**Gates.** With half geometry, ε=0.3, and O starting at q=−1, ERASE(a=2.2) on O acts as a site-addressed copy, measured from reference attractors under σ=0.15:

| Kick site on O | Arc | Gate | Pass |
|---|---|---|---|
| 2, 3 | A arc | COPY-A | 32/32 |
| 10, 11 | B arc | COPY-B | 32/32 and 31/32 |

A width-2 WRITE on O from q=0 acts as NOR(A, B). Full geometry gives no copy gates.

**The physical program fails.** Program: load b0→A and b1→B with WRITE, then b2 picks the kick site. It is only **75% correct, even with no noise.** COPY-A fails after A is loaded physically.

**Cause: a winding triple is not a complete state when rings are coupled.** From 256 random relative-phase starts at ε=0.3 with half geometry:

| Winding triple (qA,qB,qO) | Distinct attractors |
|---|---|
| (1, 0, −1) | 2 |
| (1, 1, −1) | 4 |
| (0, 0, −1) | 3 |
| (0, 0, 0) | 1 |

The relative phases between rings are real state that windings discard. The table-level enumeration (256/256, all 8 MUXes) rests on that incomplete label, so it does **not** count.

## 2. Phasor readout (O's mode-0 phasor, one complex number, one half-plane decoder)

- Contexts after the loads are physical states, not reference attractors.
- The decoder is fitted max-margin on clean points only.
- With 8 points in 2-D, only a few dozen labellings are half-plane separable, so a hit means the rings placed the points.

| Readout | Late-selector MUX functions (clean, physical) |
|---|---|
| Winding of O | 4 of 8 (0x3a 0xc5 0x5c 0xa3; 24 programs each) |
| **Phasor of O** | **8 of 8** (margins 0.19–0.26; \|Z\| ≤ 1) |

**Under noise, on the same runs** (8 inputs × 256 trials each, noise before every instruction):

| σ | 0x3a, phasor-best program: phasor / winding | 0x3a, winding-best program: phasor / winding | 0xca (phasor only) |
|---|---|---|---|
| 0.05 | 94.6 / 94.0 | 92.7 / 92.6 | 88.3 |
| 0.15 | 79.1 / 79.2 | 83.8 / 82.3 | 71.3 |
| 0.30 | 69.6 / 69.5 | 73.9 / 70.1 | 62.0 |

**Diagnosis.** Per-run phasor dumps show attractor hopping, not diffusion.
- Within one attractor the phasor is tight: 231 of 256 runs fall in a single 11° cluster with |Z| = 0.977.
- Errors are jumps into another basin. Either the slip-inducing operation fails, or the run lands in a second attractor with the same winding triple (15 of 256 for input x=1, about 60° away).

## What this establishes

1. **The phasor carries what the winding throws away.** Same physical runs: all 8 late-selector MUXes vs 4.
2. **The readout is not what limits noise.** Phasor and winding fail together because they share one failure mode, basin hopping. The phasor is marginally better as σ grows.
3. **The bottleneck is the quantized operations.** WRITE and ERASE are phase-slip kicks into a landscape with several attractors per winding. Those discrete jumps are where noise acts.

## Next

Make the inputs continuous as well: persistent ω (lens) drives on A and B, in a regime with one equilibrium and no slips. The settle is then the linearized solve L(θ)·θ = ω (the engine's paradigm). O's phasor becomes a smooth function of the inputs, and noise moves it continuously instead of hopping it.

Open question: whether the sin nonlinearity's second-order terms supply the product a selector needs (b2·b1).

## Reproduce

```bash
cd research/forks
gcc -O3 -std=c11 fork_macro.c -o fork_macro -lm
./fork_macro xcheck                                                    # IDENTICAL at eps=0
PATIENCE=40 GEOM=half EPS=0.3 ROBUST=32 ./fork_macro gates -1           # addressed COPY gates
PATIENCE=40 GEOM=half EPS=0.3 ./fork_macro multi 256 0.05 1,0,-1 1,1,-1 # hidden attractors
PATIENCE=40 GEOM=half EPS=0.3 ./fork_macro prog 0,0,-1 NOP "A.WRITE+(s=0,w=16)" NOP "B.WRITE+(s=0,w=16)" "O.ERASE(s=2,a=2.2)" "O.ERASE(s=10,a=2.2)" 0 ca 256
PATIENCE=40 GEOM=half EPS=0.3 ./fork_macro phasor -1 0                  # phasor vs winding, MUX8
PATIENCE=40 GEOM=half EPS=0.3 ./fork_macro pverify -1 0 "O.WRITE+(s=0,w=2)" "O.ERASE(s=9,a=2.2)" 3a 256
```

Raw outputs are in `a2/`.

## 3. Continuous inputs: ω drives on one equilibrium, O's phasor as readout

**Setup.**
- Each input bit switches on a **zero-sum dipole lens**: +U at one site and −U at another, either on the same ring or across rings. Σω = 0, so nothing co-rotates.
- Every input pattern must settle **locked at windings (0,0,0)**. No slips are allowed; the operating point stays in the q = 0 basin.
- **Readout:** O's mode-0 phasor relative to the array's collective phase Θ = (1/48)Σθ. Because dΣθ/dt = Σω exactly, Θ is the co-rotating frame.
  - A lab-frame readout collapsed under frequency disorder (58% at s = 0.01): the whole array drifts at Σω/48.
  - The Θ frame fixes that. Clean points are unchanged, because Θ ≡ 0 there.
- **Decoder:** one half-plane, fitted max-margin on the 8 clean points only.
- **Winding readout:** constant here by construction (every pattern sits at (0,0,0)), so it computes nothing.
- **Combinational, not sequential.** With one equilibrium the final state cannot depend on the order the inputs arrived, so "selector arrives last" is no longer a memory question. The question becomes whether ONE settle computes the MUX, which is not linearly separable.

**Search.** Half geometry, ε = 0.3, patience 40.
- 28,672 random drive programs (7 seeds, U ≤ 1.0 / 1.5 / 2.0).
- Between 23% and 90% of programs keep all 8 patterns in the basin, depending on the U cap.
- About a third of the in-basin programs compute a late-selector MUX with positive margin; the best random margin was 0.097.
- Coordinate hill-climbing (site ±1, U ±0.05, ring swap) raised the best to **0.179** (|Z| ≤ 1).

**Best program:** `A11-O14:1.450;B2-O1:0.850;O12-A5:0.900`. Input b0 is a dipole from A site 11 to O site 14, and so on. It computes 0xca (b2 ? b1 : b0) with margin 0.179. With other half-plane directions the same physics yields 50 of the 256 functions (margin > 0.02).

Noise, decoder fixed from the clean points (fast build n = 512; REF n = 128):

| noise | fast | REF |
|---|---|---|
| phase kicks σ = 0.15 / 0.30 / 0.60 | **100 / 100 / 100%** (displacement 0.0000) | 100 / 100 / 100% |
| phase kicks σ = 1.0 | 98.8% | 99.2% |
| ω disorder s = 0.005 / 0.01 / 0.02 / 0.05 | 100 / 99.8 / 99.4 / 98.6% | 100 / 100 / 99.2 / 96.1% |
| drive jitter 5% / 10% / 20% | 99.2 / 97.9 / 91.0% | 99.2 / 99.2 / 93.8% |

- REF clean phasor points are identical to the fast build to the printed precision, with the same margin of 0.1794.
- Failures are true slips: 2–33 of 512 lanes, mostly under drive jitter, because U = 1.45 sits near the slip edge.
- Lanes that are still converging (spread > 1e-8 at the patience limit) are counted separately and mostly read correctly.

**Compared with everything earlier, on the same σ = 0.15 phase-kick noise:**

| machine | late-selector MUX | accuracy at σ = 0.15 |
|---|---|---|
| 1 ring, winding, Nick's ISA (run 03) | not robust (ceiling 21/32) | 83–85% (raw programs) |
| 3 coupled rings, kick ops, winding readout | 4/8 | ~80% |
| 3 coupled rings, kick ops, phasor readout | 8/8 | 71–84% |
| **3 coupled rings, ω drives, phasor readout** | **8/8** | **100%** (also at σ = 0.6) |

**What this establishes.** Once both the inputs and the readout are continuous, the coupled rings compute the late-selector MUX in a single settle, without winding numbers.
- The settle forgets state noise completely: displacement exactly 0 at σ ≤ 0.6.
- It tolerates frequency disorder up to s ≈ 0.02 and drive jitter up to about 10%.
- The remaining errors are slips caused by strong drives near the slip edge.

**Caveats.**
1. One half-plane on one complex number still picks *which* function is read out; 50 of 256 are available from this drive program.
2. No memory: the inputs persist as drives, so a stateful machine needs its memory elsewhere.
3. The readout frame Θ is the array's own collective phase. A physical device needs a matching reference.
4. Langevin (continuous-time) noise during the settle has not been measured yet.

**Reproduce:**

```bash
PATIENCE=40 GEOM=half EPS=0.3 ./fork_macro dsearch 4096 1.5 14
PATIENCE=40 GEOM=half EPS=0.3 ./fork_macro dclimb "A11-O14:1.400;B1-O3:0.850;O13-A10:0.750" 40
PATIENCE=40 GEOM=half EPS=0.3 ./fork_macro dverify "A11-O14:1.450;B2-O1:0.850;O12-A5:0.900" 64
gcc -O3 -std=c11 -DREF_BUILD fork_macro.c -o fork_macro_ref -lm && PATIENCE=40 GEOM=half EPS=0.3 ./fork_macro_ref dverify "A11-O14:1.450;B2-O1:0.850;O12-A5:0.900" 16
```

Raw outputs are in `a2c/`.

## 4. Langevin noise during the settle (2026-10-11)

**Model.** dθ = F(θ)dt + √(2D)·dW on all 48 oscillators.
- Integrator: the RK4 step plus the additive increment √(2D·dt)·N(0,1). REF re-runs at dt = 0.02 as the check.
- With noise the state never velocity-locks, so the clock is explicit. T_SET is the noiseless lock time of this program, measured at 15 chunks = 900 time units. The phasor is then averaged over a read window.
- The decoder is the clean half-plane, never refitted.

**Program:** `A11-O14:1.450;B2-O1:0.850;O12-A5:0.900` (0xca, margin 0.179). Fast build, n = 512.

| D | phasor scatter (instantaneous) | slipped | accuracy, read window 0.5 / 60 / 600 time units |
|---|---|---|---|
| 0.0001 | 0.006 | 0 | 100 / 100 / 100% |
| 0.0003 | 0.009 | 0 | 100 / 100 / 100% |
| 0.001 | 0.071 | 1 | 99.6 / 99.8 / 100% |
| 0.003 | 0.405 | 2 | 95.3 / 95.5 / 97.9% |
| 0.01 | 0.565 | 13 | 94.5 / 96.1 / 97.9% |
| 0.03 | 0.572 | 41 | 93.6 / 96.3 / 98.4% |
| 0.1 | 0.771 | 106 | 76.8 / 81.8 / 80.3% |

REF (dt = 0.02, n = 128): D = 0.001 → 99.2% (all windows); D = 0.01 → 93.0 / 94.5 / 96.9%. The two builds agree.

**Reading it.** D = 0.01 is about a σ = 0.14 phase kick every time unit, sustained through a 900-unit settle.
- At that level the program still reads 94.5% from a single sample and 97.9% when averaged.
- Averaging helps, but the scatter jump between D = 0.001 and D = 0.003 (0.07 → 0.41) is large while accuracy only drops to about 95%. That pattern points to intermittent large excursions, not a widening Gaussian.
- The floor is set by slips: 13 of 512 at D = 0.01, 106 at D = 0.1. The strong drives (U up to 1.45) keep the operating point near the slip edge.
- Reproduce: `PATIENCE=40 GEOM=half EPS=0.3 ./fork_macro langevin "<program>" 64 0.0001,0.0003,0.001,0.003,0.01,0.03,0.1 0.5,6,60,240,600`
