# BQSM fork run 02 — Fork A on the real fabric (2026-10-10)

**Engine.** `ring_furnace.c` from nbumgarner/basin-quotient-machine @ 60890cd, fast build (DT=0.50, fast_sin). `selftest: ALL PASS` in this container before any measurement. `fork_logic.c` `#include`s the furnace unchanged and uses its own `build_table`, `settle`, `winding`, `apply_kick`, `settle_attractor`.

**Question.** Which of the 256 three-input Boolean functions can one N=16 ring compute when:
- the register is the winding number q,
- the input bits arrive as a sequence of instructions (bit k selects one of two instructions for step k),
- the program is the lens (site 0, Δ), the start state, and the instruction pairs,
- the output is READ (winding).

**Instruction set.**
- **ERASE:** the furnace's 3-site kicks, at 16 sites × amplitudes {1.0, 2.2}.
- **WRITE:** added in this run. It is a localized 2π phase ramp over 8 sites, at 16 sites × 2 signs. Its transitions are measured on the furnace, not assumed.
- **NOP.**

## Results

### 1. ERASE-only alphabet (the furnace as shipped)

Every kick lowers |q| or leaves it unchanged. That makes the machine a one-way funnel: at most 3 distinct final states, and 118/256 functions even with an arbitrary readout. With the single physical readout READ(q≠0), it reaches 2/256 pristine and 64/256 under the lens.

### 2. ERASE + WRITE (measured)

WRITE moves q to q±1 exactly as the summary describes. Under the lens, the dead ±3 states block writes past ±2. Table composition was checked against physical kick-by-kick runs: **800/800 3-step words match** (Markov property holds with WRITE).

| Δ | alive | coverage, any readout | READ(q≠0) | READ(q>0) |
|---|---|---|---|---|
| 0.00 | 7 | **256/256** | 86 | 118 |
| 0.20 | 5 | 248 | 86 | 104 |
| 0.35 | 5 | 248 | 186 | 112 |
| 0.50 | 5 | **256/256** | 234 | 112 |
| 0.65 | 5 | **256/256** | 234 | 126 |

### 3. Noise robustness — the honest numbers

Adding σ = 0.15 rad of Gaussian phase noise to every oscillator before each instruction drops 3-step accuracy to 82% (pristine) and 93% (Δ=0.50). Some table entries sit on basin boundaries.

**Fix:** re-run every (q, instruction) entry 32 times under that noise, and keep only entries that are correct 32/32. That keeps about 75–85% of entries. With only robust entries, **399/400** noisy 3-step words are correct at both Δ=0 and Δ=0.50.

| Δ | robust entries kept | coverage, any readout | READ(q≠0) | READ(q>0) |
|---|---|---|---|---|
| 0.00 | 323/424 | 240/256 | 78 | 96 |
| 0.15 | 335/448 | 248/256 | 86 | 104 |
| 0.35 | 255/320 | 248/256 | 86 | 104 |
| 0.50 | 247/320 | 248/256 | 182 | 104 |
| 0.65 | 254/320 | 248/256 | **186** | 112 |

**The 8 functions a single ring cannot compute robustly** are exactly the 2:1 multiplexer whose *selector arrives last* (b2 ? x : y, all polarity variants: 0x35 0x3a 0x53 0x5c 0xa3 0xac 0xc5 0xca). Multiplexers whose selector arrives first or second are all covered. These 8 appear only when marginal (boundary) table entries are allowed — the same robust-versus-marginal trade-off the lens-sweep results already established.

## What this establishes

1. **The ring is a programmable, noise-tolerant 3-input logic machine.** With measured ERASE + WRITE, 248/256 functions are computable using only noise-robust transitions, and the 8 missing ones have a precise structural cause.
2. **The lens is the programming knob, quantified.** With one physical readout (q≠0), robust coverage goes 78 → 186 as Δ goes 0 → 0.65. This turns FORK[1] from "landscape shaping" into a number to push up.
3. **The remaining gap is memory width, not expressiveness.** A late-selector multiplexer has to hold two independent data bits until the selector arrives, and one ring is one register. That points to two rings (a register file) or a compiler that orders inputs selector-first.
4. **Contrast with nonlinear spectral computing (arXiv 2610.09081).** That system programs all 256 functions as stateless one-shot logic. BQSM does it as a *sequential* machine with noise tolerance from basins, at 99.75% per 3-step word under σ=0.15 when restricted to robust instructions.

## Caveats

- WRITE ramp shape (8-site linear 2π ramp) is my choice. Your WRITE may differ, so re-measure with your ramp.
- Fast build only (DT=0.50). Repeat under REF_BUILD before publishing.
- Readout predicates other than q≠0 and q>0 assume you can READ the full winding number, which your summary says is free.
- Single lens at site 0. Multi-lens programs are untested and could close the late-MUX gap without a second ring.

## Next runs (all fleet-shardable)

- **A2:** two coupled rings as a 2-register machine → test the late-selector MUX.
- **A3:** multi-lens programs on one ring → can lenses alone close the gap?
- **C:** accuracy vs σ curves for robust-only programs; that is the noise-tolerance number to set against spectral computing.
- **Replicate on the T7400 and the Oracle ARM shards** (FORK[5] cross-arch).
