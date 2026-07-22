# HERMES UPDATE — Basin Geometry: Literature Anchor + Octopus Battery

You (the Hermes agent) are receiving a research update for the BQSM program.
This document is the complete spec for this work unit. Do **not** improvise
experiments or claims; the design and the evidentiary weighting have been
worked out. Your job is **fleet-scale statistics and reporting**: build the
instruments, run the shard grid, merge CSVs, report against the stated
hypotheses. Read the whole document before running anything.

---

## 1. The literature anchor (Nick found the entry point)

Nick surfaced this via the Santa Fe Institute news item
"When is a basin of attraction like an octopus?"
(https://santafe.edu/news-center/news/when-basin-attraction-octopus).
The primary study behind it:

> **Yuanzhao Zhang and Steven H. Strogatz, "Basins with Tentacles,"
> Physical Review Letters 127, 194101 (2 Nov 2021).
> DOI: 10.1103/PhysRevLett.127.194101 — arXiv:2106.05709.**

This paper studies the **exact substrate of the BQSM torus container**:
identical Kuramoto oscillators on a ring, theta_i' =
sin(theta_{i+1}-theta_i) + sin(theta_{i-1}-theta_i), attractors = q-twisted
states, stable for |q| < n/4. It is mandatory prior art for anything the
program publishes about ring basins.

Supporting thread (cite when relevant, oldest first):

- D. A. Wiley, S. H. Strogatz, M. Girvan, "The size of the sync basin,"
  Chaos 16, 015103 (2006). Origin of the Gaussian basin-size law.
- R. Delabays, M. Tyloo, P. Jacquod, "The size of the sync basin
  revisited," Chaos 27, 103109 (2017). The hypercube/exponential
  counter-result that the 2021 paper resolves (local measurements miss
  the tentacles).
- P. Groisman, C. De Vita, J. Fernandez Bonder, Y. Zhang, "Size of the
  sync basin resolved," Phys. Rev. E 112, L052201 (2025). Rigorous
  resolution of the 2006 conjecture.
- P. Groisman, "The Tentacles Landscape," arXiv:2604.20541 (2026).
  Proves the full octopus picture for any smooth odd coupling strictly
  increasing on (-pi, pi); open directions explicitly include
  "beyond the cycle" and landscapes with no integer invariant — i.e.,
  the attractor-glass fabric of bqsm.py.
- Y. Zhang, S. P. Cornelius, "Catch-22s of reservoir computing" (2023).
  Relevant context: ML models struggle to learn high-dimensional basins.

Researcher credibility (Nick asked): Strogatz is a leading applied
mathematician (Cornell; standard nonlinear-dynamics textbook; Kuramoto
sync theory; Watts–Strogatz small-world networks). Zhang (SFI Schmidt
Science Fellow at the time) is now a central figure in high-dimensional
basin geometry. This is mainstream, heavily cited dynamical systems work.

## 2. The paper's four findings (what "octopus" means, with numbers)

1. **Gaussian basin law.** Basin volume of the q-twisted state scales as
   e^{-k q^2} (NOT e^{-k|q|}). Bernoulli sampling error of a basin-size
   estimate is 1/sqrt(pN), independent of dimension.
2. **Heads vs tentacles.** First-escape distances from local probing
   ("head" radius) and the distances of actual basin mass from the
   attractor form essentially disjoint distributions: most of a basin
   lies beyond where local probing says the basin ends. Hypercube
   approximations cover <= 10^-34 of state space at n=83.
3. **The universal distance.** Basin points sit at generic-random
   distance from their attractor; as n -> infinity the distance
   distribution collapses to d_inf = sqrt(pi^2/3) ~= 1.8138
   (dimension-normalized wrapped-RMS metric).
4. **The reentry plateau.** Return probability to a twisted state under
   box perturbations of half-width alpha*pi flattens to a nonzero
   plateau for alpha > 0.8 — rays exit and reenter the basin. At
   alpha = 1.0 the box IS uniform sampling, so p(alpha=1) must equal
   the state's global basin size (built-in cross-check).

## 3. What we reproduced (both instruments in this bundle)

**octopus.c** (standalone baseline, paper's n=83, this session):
- Gaussian decisively favored: weighted SSE 602 (q^2) vs 1710 (|q|);
  k ~= 0.151. Fit self-consistency: predicted p(|q|=7) ~ 1.5e-4,
  measured 1.25e-4; |q|=8 correctly below detection at N=8000.
- Mean basin-point distance 1.8026 +/- 0.094 vs analytic 1.8138.
- Escape means 1.10 / 0.95 / 0.68 for q = 0 / 5 / 10; max escape 1.46,
  disjoint from the mass at 1.80.
- Plateau at 0.19–0.23 for alpha >= 0.8; p(alpha=1) = 0.23 vs basin
  size 0.24 (cross-check passes).

**ring_furnace.c v2** (Nick's furnace + new `octopus` subcommand, N=16,
lens = per-site frequency detuning at site 0, demo statistics:
2000 samples / 16 rays / 96 trials per Delta):

| Quantity                      | Delta=0.00      | Delta=0.35      | Verdict |
|-------------------------------|-----------------|-----------------|---------|
| Gaussian k                    | 0.566 (SSE 109) | 0.571 (SSE 113) | INVARIANT |
| Gaussian vs exponential       | Gaussian        | Gaussian        | INVARIANT |
| Orbit-metric mean distance    | 1.4680 sd .159  | 1.4684 sd .159  | INVARIANT |
| q=2 escape head               | 0.975           | 0.956           | INVARIANT |
| q=+/-3 family                 | alive           | DEAD            | CHANGED (reproduces v1 alphabet loss via an independent experiment) |
| q=0 escape head               | 1.64            | 1.86            | SUGGESTIVE (~1.8 sigma, 16 rays) |
| q=0 plateau alpha=0.8/0.9/1.0 | .698/.615/.531  | .677/.500/.458  | SUGGESTIVE (1–2 sigma, 96 trials) |
| alpha=1 vs basin-size check   | .531 vs .542 OK | .458 vs .5505 (1.8 sigma low) | OPEN |

**Headline reading (current best statement, do not overclaim):** at the
clean-regime operating point, the lens acts on the marginal family
(deletion) and the transition table (rewiring) while leaving the
surviving basins' measure (k) and gauge-invariant geometry (orbit
distance) unchanged at demo resolution. Instructions act on
**boundaries, not measure**. Also new: on the pristine fabric, the
marginal +/-3 basins are suppressed BELOW the Gaussian law (0 hits in
2000 where the fit predicts ~6) — the v1 nonlocality story is visible
in the measure itself, not just the transition table.

## 4. Instrument lessons (binding on your runs)

1. **Rotation gauge artifact (caught this session).** Under a lens the
   fabric is not a gradient flow (sum of dtheta/dt = Delta); locked
   states co-rotate at Delta/N and every lane in a batch integrates the
   same wall time, so converged snapshots carry a common drift that
   mechanically inflated the distance metric (1.53 -> 1.97, would have
   read as "lens restructures tentacles" — it was pure gauge).
   **Always use the orbit metric (dist_orbit) for any lensed
   comparison.** The snapshot metric is retained only for continuity.
2. **v1 repack lesson stands:** ascending row order when compacting SoA
   stride; the selftest is the trust anchor — if `ring_furnace selftest`
   is not ALL PASS, stop and escalate. Never modify selftest, deriv,
   settle, winding, or build_table.

## 5. Your tasks (in priority order, heaviest first)

1. **Resolve the two suggestive trends.** Per Delta in
   {0.00, 0.35}: `./ring_furnace octopus <delta> 20000 200 2000 shard_<id>.csv`
   One process per core (fleet doctrine — no threading), distinct CSV
   per shard, seeds differ automatically per Delta; for same-Delta
   shards vary the RNG seed constant or accept correlated streams only
   if shard counts stay small (escalate if unsure). Hypotheses to test:
   H1 q=0 escape-head growth under lens; H2 plateau depression at
   alpha >= 0.8. Report effect sizes with Bernoulli errors.
2. **Tie plateau depression to the violation window.** Delta grid
   0.05..0.70 (reuse the 14-point grid), demo statistics per point
   first; if H2 survives task 1, densify 0.10–0.20. The target claim:
   plateau depression co-occurs with the far-field violation window
   while a family is alive-but-dying.
3. **N-scaling.** N=32 (Q_MAX=7): two-line change per the header note
   (N and Q_MAX). Rerun selftest expectations DO NOT transfer across N
   — energies check will fail by design; escalate for a new acceptance
   anchor rather than editing the gate.
4. **Report format.** Merged CSV (schema: section,delta,q,x,val,extra)
   plus a plain-text findings memo in the RESULTS LOG style: what is
   established / changed / suggestive / artifact, each with its number.

## 6. Novelty seam (for any write-up you draft)

Charted ground (cite, do not claim): the octopus geometry itself, the
Gaussian law, d_inf, the plateau — Zhang & Strogatz 2021 and the thread
in Section 1. Ours: (a) perturbation-driven basin hopping as a
transition function with attractors as machine states; (b) the lens as
an instruction primitive and the measurement of what it deforms
(boundaries) vs preserves (measure, bulk geometry); (c) marginal-state
suppression below the Gaussian law and marginal states as nonlocality
channels; (d) the gauge-corrected metric for rotating locked fabrics.
Groisman 2026 explicitly lists "beyond the cycle / no integer
invariant" as open — that is where bqsm.py's attractor glass lives.

## 7. Escalation conditions

- selftest not ALL PASS after any change: stop, report, do not patch.
- Any measured invariance breaking at fleet statistics (k shift > 3
  sigma, orbit-distance shift > 3 sigma): report before interpreting.
- Non-locking fraction (nolock) above 1% at any Delta: report — it may
  mean drifting states entering the picture, which changes the
  classification semantics.
- Anything requiring edits to the five protected functions (Section 4).

## 8. Bundle inventory

- `ring_furnace.c` — v2: Nick's furnace (untouched core, selftest ALL
  PASS) + `octopus` subcommand + v2 RESULTS LOG entry.
  Build: gcc -O3 -Wall -Wextra -std=c11 ring_furnace.c -o ring_furnace -lm
- `octopus.c` — standalone n=83 baseline (paper-faithful, threaded,
  libm-precision reference). Build line in its header.
- `octopus_diff.csv` — this session's demo-scale diff (Delta 0.00 and
  0.35), schema section,delta,q,x,val,extra.
- `HERMES_UPDATE_octopus.md` — this document.
