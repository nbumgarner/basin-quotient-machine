#!/usr/bin/env python3
# ===========================================================================
# bqsm_batteries.py — this session's instruments, ported onto the real fabric
# ===========================================================================
#   emerging.systems — runs AGAINST bqsm.py (imported, never modified)
#
# WHAT PORTS, WHAT DOESN'T, AND WHY THAT'S STATED UP FRONT
#   The ring's winding-based results (typed instructions, quantized write)
#   rested on a topological invariant this fabric doesn't have — symbols
#   here are random drive directions in R^n. So we port the PROTOCOLS, not
#   the conclusions, and measure what THIS fabric's math actually does:
#
#   BATTERY G — graph anatomy: extract the automaton via bqsm.census/probe,
#     then compute reachability, absorbing states, and funnel structure.
#     The ring funneled totally (erase-only). Does a random reciprocal
#     fabric share that arrow, and does flow correlate with basin volume
#     (the only energy-proxy available without an exact potential)?
#
#   BATTERY K — clock pressure (Battery B, re-hosted): apply symbol y1,
#     free-run only w ticks (instead of settling), apply y2, settle fully.
#     Compare against the table composed twice. Fidelity vs w gives the
#     fabric's minimum reliable inter-symbol wait — its clock period, in
#     ticks, measured behaviorally. Cross-check: census already records
#     mean settle time per state; the two numbers should be same-scale.
#
# Determinism: every battery seeds its own Generator; bqsm.py's own seeds
# govern container/census/probe. Results print + JSON-export.
# ===========================================================================

import json                                                                    # Results artifact for the kit.
import numpy as np                                                             # Shared numerical spine.
import bqsm                                                                    # THE fabric under test — imported, surgically untouched.

# --- Test-bench parameters --------------------------------------------------
CENSUS_SAMPLES = 600                                                           # Denser survey than selftest for stabler basin volumes.
TOP_M          = 10                                                            # Machine states admitted to the batteries: top-M by basin volume.
N_SYMBOLS      = 4                                                             # Input alphabet size (bqsm.probe default, kept for comparability).
WAITS          = (0, 5, 10, 20, 40, 80, 160)                                   # Clock sweep, in fabric ticks.
PAIR_SAMPLES   = 12                                                            # (state, y1, y2) triples sampled for Battery K.
LAND_TOL       = 1e-3                                                          # Landing classification tolerance (matches bqsm.probe's).

def classify(x, attractors):
    """Nearest-known-attractor label, or -1 — bqsm.probe's rule, shared."""
    for i, a in enumerate(attractors):                                         # Linear scan is fine at TOP_M states.
        if np.max(np.abs(a - x)) < LAND_TOL:                                   # Sup-norm, same tolerance as probe().
            return i                                                           # Landed in known basin i.
    return -1                                                                  # Off the map.

def free_run(c, x, ticks):
    """Advance the fabric `ticks` steps with no drive — the wait interval."""
    for _ in range(ticks):                                                     # Exactly `ticks` clock edges.
        x = c.step(x)                                                          # Undriven fabric evolution.
    return x                                                                   # Possibly mid-transient state.

# ---------------------------------------------------------------------------
# BATTERY G — automaton anatomy: funnels, sinks, and the volume arrow.
# ---------------------------------------------------------------------------
def battery_G(c, cen, prb):
    attractors = cen["attractors"][:TOP_M]                                     # The admitted state set.
    vols = cen["basin_fraction"][:TOP_M]                                       # Their basin volumes (energy proxy).
    table = [row[:] for row in prb["table"][:TOP_M]]                           # Transition rows for admitted states.
    for row in table:                                                          # Landings outside TOP_M are off-map
        for y in range(N_SYMBOLS):                                             # for this analysis: mark them LOST
            if row[y] >= TOP_M: row[y] = -1                                    # rather than silently keeping them.

    # Reachability closure over resolved edges.
    reach = {s: {s} for s in range(TOP_M)}                                     # Reflexive start.
    for _ in range(TOP_M):                                                     # Relax to fixpoint (diameter bound).
        for s in range(TOP_M):                                                 # Each state inherits its targets'
            for y in range(N_SYMBOLS):                                         # closures across every symbol.
                t = table[s][y]                                                # Edge target.
                if t >= 0: reach[s] |= reach[t]                                # Merge closures.
    sinks = [s for s in range(TOP_M)                                           # Absorbing = every resolved edge
             if all(t in (s, -1) for t in table[s])]                           # self-loops (or is off-map).

    # The volume arrow: over resolved non-self edges, how often does the
    # machine move to a LARGER basin? The ring's funnel was 100% one-way;
    # this is that measurement generalized to volume as the energy proxy.
    up = down = flat = 0                                                       # Direction tallies.
    for s in range(TOP_M):                                                     # Every resolved hop...
        for y in range(N_SYMBOLS):
            t = table[s][y]                                                    # ...to a known state...
            if t < 0 or t == s: continue                                       # (skip self-loops / off-map).
            if   vols[t] > vols[s] * 1.05: up += 1                             # Toward bigger basin (5% guard band
            elif vols[t] < vols[s] * 0.95: down += 1                           # against Monte-Carlo volume noise).
            else: flat += 1                                                    # Within noise: no direction call.
    print("BATTERY G — automaton anatomy")
    print(f"  admitted states     : {TOP_M} of {cen['n_attractors']} "
          f"(volumes {', '.join(f'{v:.3f}' for v in vols)})")
    print(f"  flaky edges (probe) : {prb['flaky_edges']}")
    print(f"  absorbing states    : {sinks if sinks else 'none'}")
    print(f"  reach(0)            : {sorted(reach[0])}")
    lost = sum(row.count(-1) for row in table)                                 # Off-map edge count.
    print(f"  off-map edges       : {lost}/{TOP_M*N_SYMBOLS}")
    tot = up + down + flat                                                     # Directional sample size.
    if tot:                                                                    # Print the arrow if measurable.
        print(f"  volume arrow        : toward-larger {up}/{tot}, "
              f"toward-smaller {down}/{tot}, flat {flat}/{tot}")
    return {"sinks": sinks, "up": up, "down": down, "flat": flat,
            "lost": lost, "reach0": sorted(reach[0]), "table": table}

# ---------------------------------------------------------------------------
# BATTERY K — clock pressure: two-symbol words under truncated settling.
# ---------------------------------------------------------------------------
def battery_K(c, cen, prb, G):
    attractors = cen["attractors"][:TOP_M]                                     # Admitted states.
    table = G["table"]                                                         # Off-map-cleaned table.
    rng = np.random.default_rng(101)                                           # Battery's own seed.
    symbols = np.random.default_rng(23).normal(0.0, 1.0, (N_SYMBOLS, c.n))     # REBUILD probe's symbol set:
    symbols *= prb["kick"] / np.linalg.norm(symbols, axis=1, keepdims=True)    # same seed(23), same normalization,
                                                                               # so batteries drive the SAME inputs
                                                                               # probe() measured the table with.
    # Sample (s, y1, y2) triples whose two-step table walk is fully resolved.
    triples = []                                                               # The test word list.
    guard = 0                                                                  # Rejection-sampling guard.
    while len(triples) < PAIR_SAMPLES and guard < 500:                         # Bounded search.
        guard += 1                                                             # Count the attempt.
        s  = int(rng.integers(0, TOP_M))                                       # Random start state.
        y1 = int(rng.integers(0, N_SYMBOLS))                                   # First symbol.
        y2 = int(rng.integers(0, N_SYMBOLS))                                   # Second symbol.
        t1 = table[s][y1]                                                      # Table step one.
        if t1 < 0: continue                                                    # Unresolved: reject.
        t2 = table[t1][y2]                                                     # Table step two.
        if t2 < 0: continue                                                    # Unresolved: reject.
        triples.append((s, y1, y2, t2))                                        # Admit with its reference.

    print("\nBATTERY K — clock pressure (two-symbol words, truncated wait)")
    print(f"  {'wait':>5}  fidelity")
    results = {}                                                               # wait -> fidelity.
    for w in WAITS:                                                            # The clock sweep.
        ok = 0                                                                 # Match tally at this wait.
        for (s, y1, y2, ref) in triples:                                       # Every sampled word.
            x = attractors[s] + rng.normal(0, 1e-6, c.n)                       # Prep jitter, as probe() does.
            x, st1, _ = bqsm.relax(c, x, u=symbols[y1],                        # Symbol 1: kick + relax, but we
                                   kick_steps=prb["kick_steps"],               # DISCARD the settled guarantee by
                                   max_iters=prb["kick_steps"] + w)            # capping iters at kick+wait ticks —
                                                                               # 'undecided' here IS the point:
                                                                               # the fabric is mid-transient.
            xf, st2, _ = bqsm.relax(c, x, u=symbols[y2],                       # Symbol 2: kick, then settle FULLY.
                                    kick_steps=prb["kick_steps"])              # Full relaxation for the readout.
            land = classify(xf, attractors) if st2 == "fixed" else -1          # Final label (or off-map).
            ok += (land == ref)                                                # Compare to the composed table.
        fid = ok / len(triples)                                                # Fidelity at this wait.
        results[w] = fid                                                       # Record.
        print(f"  {w:>5}  {ok}/{len(triples)} = {fid:.3f}")
    # Cross-check scale: census mean settle times are the same clock in the
    # same units — print them side by side so agreement (or not) is visible.
    ms = cen["mean_settle"][:TOP_M]                                            # Per-state settle ticks.
    print(f"  census mean settle  : min {min(ms):.0f}, "
          f"median {sorted(ms)[len(ms)//2]:.0f}, max {max(ms):.0f} ticks")
    return results

if __name__ == "__main__":
    c = bqsm.Container()                                                       # THE default fabric (n=64, seed=7).
    cen = bqsm.census(c, samples=CENSUS_SAMPLES)                               # Alphabet survey.
    print(f"census: {cen['n_attractors']} attractors, "
          f"{cen['cycles']} cycles, {cen['undecided']} undecided\n")
    prb = bqsm.probe(c, cen["attractors"])                                     # The automaton, probe's way.
    G = battery_G(c, cen, prb)                                                 # Anatomy.
    K = battery_K(c, cen, prb, G)                                              # Clock.
    out = {"census": {k: v for k, v in cen.items() if k != "attractors"},      # JSON artifact (vectors omitted:
           "batteryG": {k: v for k, v in G.items() if k != "table"},           # they're reproducible from seeds).
           "batteryK": K}
    with open("batteries_result.json", "w") as f:                              # Kit-consumable record.
        json.dump(out, f, indent=2)
    print("\nwritten: batteries_result.json")

# [Block rationale] Symbols are REBUILT from probe's seed rather than passed
# out of probe() because probe doesn't return them — and modifying bqsm.py
# to return them would violate the surgical-edit contract for a convenience.
# The rebuild is exact (same generator, same seed, same normalization); if
# bqsm.py's probe seed ever changes, this file's rebuild breaks loudly via
# nonsense fidelities, not silently. Battery K caps relax() iterations to
# implement truncated waiting instead of adding a wait parameter to bqsm.py —
# same contract, same reasoning.
# ===========================================================================

# ===========================================================================
# RESULTS LOG — first execution, 2026-07-20 (chat container; bqsm.py untouched)
# ===========================================================================
# FABRIC: default Container (n=64, k=8, gain=1.6, symmetry=1.0, seed=7).
#   bqsm.py selftest: 4/4 PASS on this host before any battery ran.
# CENSUS: 134 attractors in 600 samples, 0 cycles, 0 undecided — survey NOT
#   saturated (top-10 states hold ~33% of volume). The fabric is a glass.
# BATTERY G: automaton at probe defaults (kick 1.2) is FROZEN — 9/10
#   admitted states absorbing, one resolved non-self edge, reach(0)={0}.
# BATTERY K: fidelity 1.000 at all waits — DEGENERATE, measured nothing:
#   with a near-identity table the sampled words were self-loop walks.
#   Clock measurement requires a mobile machine first. (The census settle
#   times, 132–713 ticks, are the fabric's only clock datum so far.)
# MOBILITY CURVE (kick sweep at gain 1.6, top-8, edges of 32):
#   kick 1.2: frozen 30, mobile 1, lost 1     kick 6.0: frozen 19, mobile 3, lost 10
#   kick 9.0: frozen 8, mobile 4, lost 20     — edges convert frozen -> LOST
#   (scattered into the glass's long tail); mobile never exceeds 4/32.
#   NO computing phase between ice and plasma at this operating point.
# GAIN CONTROL (same protocol): gain 1.1 -> 33 attractors, best window
#   mobile 5/32 with lost 12/32 at kick 4.0; gain 1.3 similar but thinner.
#   Lower gain widens the window marginally; nowhere does it open cleanly.
# FINDING (the honest cross-dataset statement, with the ring as contrast):
#   Multistability is cheap; ORGANIZATION is the scarce resource. The ring
#   (few, symmetry-protected attractors) yielded a total-funnel machine with
#   typed instructions and a measurable clock; the random reciprocal fabric
#   yields a glass whose inferred barrier spectrum spans the whole kick
#   range — frozen and lost coexist at every amplitude, so no single drive
#   level is simultaneously above many barriers and below scatter. Random W
#   does not spontaneously organize into a machine on the sampled
#   (gain, kick) plane. IMPLICATION: FORK[1] (landscape shaping) is the
#   critical path, and structured coupling topologies (ring/lattice blocks
#   inside Container) are the cheap intermediate to test before optimizing W.
# NOT YET ESTABLISHED: full (gain, kick, symmetry) phase diagram; whether
#   symmetry<1 or larger k changes the picture; window behavior with the
#   full 134-state alphabet admitted (top-8 truncation inflates 'lost').
# ===========================================================================
