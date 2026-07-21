# basin-quotient machine

**A small computer discovered inside a ring of coupled oscillators — states
are basins of attraction, instructions are perturbations, and the transition
table is measured, not designed.**

```
   q=-3  → ↘ ↙ ← ↑ ↗ ↘ ↙ ← ↖ ↗ ↘ ↓ ↙ ↖ ↗   V =  -6.1229
   q=-2  → ↘ ↓ ↙ ← ↖ ↑ ↗ → ↘ ↓ ↙ ← ↖ ↑ ↗   V = -11.3137
   q=-1  → ↘ ↘ ↘ ↓ ↙ ↙ ↙ ← ← ↖ ↖ ↑ ↗ ↗ ↗   V = -14.7821
   q=+0  → → → → → → → → → → → → → → → →   V = -16.0000
   q=+1  → ↗ ↗ ↑ ↑ ↖ ↖ ← ← ↙ ↙ ↙ ↓ ↘ ↘ ↘   V = -14.7821
   q=+2  → ↗ ↑ ↖ ← ↙ ↓ ↘ → ↗ ↑ ↖ ← ↙ ↓ ↘   V = -11.3137
   q=+3  → ↑ ↖ ↙ ↓ ↘ ↗ ↖ ← ↙ ↘ ↗ ↑ ↖ ↙ ↘   V =  -6.1229
```

Each row is one stable attractor of a 16-site Kuramoto ring; the arrows are
the oscillator phases as you walk around the ring. The needle rotates `q`
full turns per row — that integer, the **winding number**, is a topological
invariant, which makes it a natural discrete state label living inside a
continuous dynamical system. (The `±3` rows look ragged only because eight
arrow glyphs alias three turns over sixteen sites; the underlying phases are
a perfectly uniform gradient.)

## Quick start

```
g++ -O2 -std=c++17 basin_machine.cpp -o basin_machine
./basin_machine
```

One file, standard library only, x86_64 and aarch64, deterministic output,
runs in a few seconds. The program *performs* the findings below rather than
asserting them.

## What the measurements found

Treating attractors as states and perturbations as instructions, the ring's
emergent machine turns out to have structure nobody put in:

**It is a genuine state machine.** Prepared via four different histories,
every state responds identically to every probe (72/72 trials), and the
physical two-instruction path agrees with composing the measured table
twice (36/36). The quotient satisfies the Markov property.

**It has a clock.** Probe the fabric before it finishes relaxing and the
table lies; probe after, and it is exact. Measured minimum reliable
inter-instruction wait: between 5 and 10 time units — matching the ring's
slowest relaxation mode, 1/(K·(2π/N)²) ≈ 6.7. Behavior and spectrum agree.

**The instruction set is typed, and the write is quantized.** This is the
part the demo shows live:

```
3. THE FUNNEL — scalar bumps from q=0, amplitudes 1.0..3.0:
   amp 1.0 -> q=+0 ... amp 3.0 -> q=+0
   No scalar bump adds winding: the instruction set only runs downhill.

4. QUANTIZED WRITE — gradient ramps from q=0 (width 6):
   ramp 1.00π -> q=+0  (sub-quantum: rejected)
   ramp 2.00π -> q=+1  (one quantum: written)
   two ramps chained: q=+2  (an increment instruction)
```

*Read* is free (a topological invariant). *Erase* is any localized scalar
bump — and it is one-way: across the tested alphabet, no bump ever
increases |q|; the reachability graph is a funnel with q=0 absorbing.
*Write* requires input shaped like winding itself — a phase gradient — and
is quantized: exactly +1 per full 2π of ramp, nothing for less.

**Edits to the fabric are local — when states are robust.** Detuning a
single oscillator (a "lens") changes transition-table entries only for
instructions applied at or beside the defect (zero changes at ring distance
≥ 2 across a wide lens-strength band). But locality has a failure mode with
a mechanism: while a *marginal* state family (|q|=3, near its stability
boundary) is alive-but-dying, its globally soft basin boundaries carry the
local edit to arbitrarily distant table entries. Robust states edit
locally; marginal states are nonlocality channels. Keep the working
alphabet away from stability margins and the machine is composable.

**A negative result that matters:** a random multistable fabric (a damped
tanh network with 130+ attractors) does *not* form a usable machine at any
tested drive level — its edges go from frozen to scattered with no mobility
window between. Multistability is cheap; *organization* of the attractor
set is the scarce resource. The ring computes because symmetry organizes
its landscape.

## Ancestry, and what is actually claimed

The concept family is well documented and this project stands on it:
attractor networks and energy landscapes (Hopfield), reservoir computing,
twisted-state stability on Kuramoto rings (Wiley–Strogatz–Girvan),
basin-hopping on energy surfaces, and coupled-oscillator / bifurcation
machines in hardware. Winding quantization and phase slips are classical
physics.

What is claimed here is narrower and, we believe, new in its framing: the
**basin quotient read as a programmable machine**, with a *typed, measured
instruction set*, an experimentally established *clock period*, *locality
rules for structural edits* including the marginal-state mechanism of their
violation, and exact *energetic accounting* of instructions against
Newton-verified saddle barriers. Every claim above was produced by an
instrument with its command line and caveats logged; several ship with
falsifiable predictions (clock period scaling as N²/K; lens effects
superposing below an interaction radius; the next locality breakdown being
mediated by the |q|=2 family).

Simulating the fabric on a CPU does not, of course, escape any bottleneck
of conventional hardware — the interest is in what the mathematics
volunteers, and in what a physical substrate that *natively* relaxes could
do with it. That question is open, and this repository is the measured
foundation for asking it carefully.

## Files

- `basin_machine.cpp` — the live demonstration (this repo's whole build).
- The full instrument suite (locality sweeps, Markov/clock batteries,
  saddle-point energetics, and a batched C engine that reproduces the
  Python instruments' published numbers exactly) is being prepared for
  release alongside.

---
*emerging.systems — measured claims only.*
