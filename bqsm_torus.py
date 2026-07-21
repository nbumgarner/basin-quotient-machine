#!/usr/bin/env python3
# ===========================================================================
# bqsm_torus.py — BQSM rung-two experiment: single-lens bifurcation locality
# ===========================================================================
#   emerging.systems — Basin-Quotient State Machine research fork
#
# QUESTION UNDER TEST
#   If the fabric is edited at exactly one site (a "lens" defect), do the
#   machine's transitions change only near the edit, or does the change
#   cascade through the whole transition table? Locality = the fabric is
#   composable/programmable. Cascade = rung two fails.
#
# THE CONTAINER
#   N identical phase oscillators on a ring (1-torus), nearest-neighbor
#   Kuramoto coupling:  dθ_i/dt = ω_i + K·[sin(θ_{i+1}−θ_i) + sin(θ_{i−1}−θ_i)]
#   Its stable attractors are the classic "twisted states": θ_i = 2πqi/N,
#   one per integer winding number q with |q|/N < 1/4. Winding number is a
#   topological invariant — a perfect discrete, enumerable machine-state
#   label that survives small parameter changes. This is why the ring is the
#   right first container: the attractor alphabet is known in closed form,
#   so basin-hopping is measurable without any attractor-discovery machinery.
#
# MACHINE DEFINITION
#   States   : winding numbers q ∈ {−Q..+Q}
#   Inputs   : localized kicks — add phase offset A to a 3-site window at
#              site s. Alphabet = every site × two amplitudes.
#   T[q][p]  : settle into attractor q, apply kick p, integrate, read the
#              winding number of the attractor reached.
#
# THE LENS
#   Detune one oscillator: ω_L += Δ. One scalar at one site — the smallest
#   possible structural edit, matching the "single-lens" framing.
#
# METRIC (as specified in the research plan)
#   Hamming distance between pre-lens and post-lens transition tables,
#   bucketed by ring distance between the kick site and the lens site.
#   Locality prediction: changed entries concentrate at small distance.
#   Cascade prediction: changes are flat across distance.
#
# Runtime: seconds. stdlib + numpy only. Deterministic (no RNG in the
# dynamics at all; integration is a fixed RK4 grid).
# ===========================================================================

import numpy as np                      # vector math for the oscillator field
import sys                              # exit codes / argv

# --- Experiment parameters (small enough to enumerate, big enough to mean it)
N       = 16                            # oscillators on the ring
K       = 1.0                           # coupling strength (sets time scale)
Q_MAX   = 3                             # stable twisted states: |q| ≤ N/4 → 3
DT      = 0.02                          # RK4 step (K·DT ≪ 1 for accuracy)
T_SETTLE= 60.0                          # settle time after a kick (many 1/K)
KICK_W  = 3                             # kick window width in sites
AMPS    = (1.0, 2.2)                    # kick amplitudes: sub- and super-critical
LENS_SITE = 0                           # where the structural edit lives
LENS_DELTA= 0.35                        # detuning magnitude (moderate defect)

# ---------------------------------------------------------------------------
# Dynamics: nearest-neighbor Kuramoto on a ring with per-site frequencies.
# ---------------------------------------------------------------------------
def deriv(theta, omega):
    """dθ/dt for the ring. np.roll gives periodic (torus) neighbors free."""
    left  = np.roll(theta,  1)          # θ_{i-1} (ring wrap via roll)
    right = np.roll(theta, -1)          # θ_{i+1}
    return omega + K * (np.sin(right - theta) + np.sin(left - theta))

def integrate(theta, omega, t_total):
    """Classic RK4 over a fixed grid — deterministic, no adaptivity to vary."""
    steps = int(t_total / DT)           # fixed step count
    for _ in range(steps):              # march the field forward
        k1 = deriv(theta,            omega)     # slope at start
        k2 = deriv(theta + 0.5*DT*k1, omega)    # slope at midpoint (k1)
        k3 = deriv(theta + 0.5*DT*k2, omega)    # slope at midpoint (k2)
        k4 = deriv(theta +     DT*k3, omega)    # slope at end
        theta = theta + (DT/6.0)*(k1 + 2*k2 + 2*k3 + k4)  # RK4 update
    return np.mod(theta, 2*np.pi)       # keep phases wrapped for readout

# ---------------------------------------------------------------------------
# State readout: winding number = (1/2π) Σ wrapped phase differences.
# Topological, integer-valued, robust to the small phase distortions a lens
# defect introduces — exactly why it is the state label and not raw phases.
# ---------------------------------------------------------------------------
def winding(theta):
    d = np.diff(np.concatenate([theta, theta[:1]]))   # successive diffs, ring-closed
    d = np.mod(d + np.pi, 2*np.pi) - np.pi            # wrap each diff to (−π, π]
    q = np.sum(d) / (2*np.pi)                         # total winding
    return int(np.rint(q))                            # snap to the integer invariant

def twisted(q):
    """Exact twisted-state initial condition for winding number q."""
    return np.mod(2*np.pi*q*np.arange(N)/N, 2*np.pi)  # θ_i = 2πqi/N

def settle_check(theta, omega):
    """Attractor sanity: after settling, velocities must be uniform (locked).
    max−min (not distance from zero) so co-rotating locked states — which any
    net detuning forces, since the locked field drifts at ω̄ = ΣΔ/N — pass."""
    v = deriv(theta, omega)                            # instantaneous velocities
    return float(np.max(v) - np.min(v))                # 0 ⇒ phase-locked

def settle(theta, omega, tol=1e-8, max_chunks=12):
    """Integrate in T_SETTLE chunks until phase-locked to `tol`, or give up.
    Convergence-based rather than fixed-clock: the ring's slowest mode decays
    at ~K·(2π/N)² ≈ 0.15, so a fixed 60-unit clock leaves ~1e-4 residual and
    a naive tight threshold misclassifies real attractors as unsettled —
    which is exactly the bug the first run of this experiment exposed."""
    for _ in range(max_chunks):                        # bounded patience
        theta = integrate(theta, omega, T_SETTLE)      # one settling chunk
        if settle_check(theta, omega) < tol:           # locked to tolerance?
            return theta, True                         # converged attractor
    return theta, False                                # never locked (drift/chaos)

# ---------------------------------------------------------------------------
# The machine: build the full transition table for a given ω profile.
# ---------------------------------------------------------------------------
def kick(theta, site, amp):
    """Localized input symbol: add `amp` to a KICK_W window centered at site."""
    out = theta.copy()                                 # never mutate caller state
    for k in range(-(KICK_W//2), KICK_W//2 + 1):       # window around the site
        out[(site + k) % N] += amp                     # ring-wrapped index
    return np.mod(out, 2*np.pi)                        # keep wrapped

def transition_table(omega):
    """T[(q, site, amp)] = q'. Also returns the settled attractor per q so a
    lensed system's (slightly distorted) attractors are used as the true
    start states — starting from the *ideal* twist under a defect would
    conflate settling transients with input response."""
    states = {}                                        # q -> settled attractor
    for q in range(-Q_MAX, Q_MAX + 1):                 # enumerate the alphabet
        th, ok = settle(twisted(q), omega)             # relax to lock under THIS ω
        if not ok or winding(th) != q:                 # unlocked, or fell to other q
            states[q] = None                           # state died under the lens
        else:
            states[q] = th                             # its true fixed point
    table = {}                                         # the machine itself
    for q, th0 in states.items():                      # every start state
        if th0 is None: continue                       # skip dead states
        for site in range(N):                          # every kick location
            for amp in AMPS:                           # every kick strength
                th, ok = settle(kick(th0, site, amp), omega)   # relax the kick
                table[(q, site, amp)] = winding(th) if ok else None  # None = no lock
    return table, states

# ---------------------------------------------------------------------------
# The experiment: pre-lens table, post-lens table, Hamming-by-distance.
# ---------------------------------------------------------------------------
def ring_dist(a, b):
    """Shortest ring distance between two sites (the locality coordinate)."""
    d = abs(a - b) % N
    return min(d, N - d)

def main():
    omega0 = np.zeros(N)                               # pristine fabric: identical ω
    omegaL = omega0.copy()                             # lensed fabric...
    omegaL[LENS_SITE] += LENS_DELTA                    # ...one scalar changed

    print(f"BQSM torus locality experiment  N={N} K={K} lens: ω[{LENS_SITE}]+={LENS_DELTA}")
    T0, S0 = transition_table(omega0)                  # machine before the edit
    T1, S1 = transition_table(omegaL)                  # machine after the edit

    alive0 = [q for q, s in S0.items() if s is not None]   # surviving states, pre
    alive1 = [q for q, s in S1.items() if s is not None]   # surviving states, post
    print(f"states pre-lens : {alive0}")
    print(f"states post-lens: {alive1}")
    if alive0 != alive1:                               # state-set change is itself
        print("NOTE: lens altered the state alphabet itself — report reflects "
              "the intersection.")                     # a (reported) finding

    # Hamming distance bucketed by kick-site distance from the lens.
    common = sorted(set(T0) & set(T1))                 # comparable entries only
    buckets = {}                                       # dist -> [changed, total]
    for key in common:                                 # every comparable entry
        q, site, amp = key                             # unpack the input symbol
        d = ring_dist(site, LENS_SITE)                 # locality coordinate
        c, t = buckets.get(d, (0, 0))                  # current bucket tallies
        buckets[d] = (c + (T0[key] != T1[key]), t + 1) # count change + total

    if not common:                                     # nothing comparable
        print("\nRESULT: no comparable entries — the lens destroyed the state "
              "alphabet outright. That is a CASCADE-class outcome at the "
              "alphabet level; reduce LENS_DELTA to probe the graded regime.")
        return 1
    print("\ndist_from_lens  changed/total   hamming")
    total_c = total_t = 0                              # global tallies
    near_c = near_t = far_c = far_t = 0                # near/far split (d≤2 vs d≥5)
    for d in sorted(buckets):                          # ascending distance
        c, t = buckets[d]                              # bucket tallies
        print(f"      {d:2d}          {c:3d}/{t:<3d}      {c/t:.3f}")
        total_c += c; total_t += t                     # accumulate global
        if d <= 2: near_c += c; near_t += t            # lens-adjacent bucket
        if d >= 5: far_c  += c; far_t  += t            # lens-distant bucket
    print(f"\noverall hamming        : {total_c}/{total_t} = {total_c/total_t:.3f}")
    near = near_c / near_t if near_t else 0.0          # adjacent change rate
    far  = far_c  / far_t  if far_t  else 0.0          # distant change rate
    print(f"lens-adjacent (d<=2)   : {near:.3f}")
    print(f"lens-distant  (d>=5)   : {far:.3f}")

    # Verdict criterion, stated before anyone sees the data it judges:
    # LOCAL if adjacent change rate is at least 3× the distant rate AND the
    # distant rate is under 10%. CASCADE if distant ≥ 25% or near ≈ far.
    if far < 0.10 and (far == 0 or near / max(far, 1e-9) >= 3.0):
        print("\nVERDICT: LOCAL — bifurcation effects concentrate at the lens.")
    elif far >= 0.25 or (near > 0 and abs(near - far) / max(near, far) < 0.3):
        print("\nVERDICT: CASCADE — the edit propagates through the table.")
    else:
        print("\nVERDICT: MIXED — partial locality; see distance profile.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

# [Block rationale] Why this container and not the bqsm.py fabric directly:
# the ring's attractors are known in closed form (twisted states, labeled by
# a topological invariant), so every observed transition is unambiguous —
# no attractor-clustering heuristics whose own errors could masquerade as
# cascade. The cost is generality: this tests the locality QUESTION in the
# cleanest fabric that has it, not bqsm.py's specific equations. If LOCAL
# here, the next step is porting the same protocol onto bqsm.py's fabric;
# if CASCADE even here, in the friendliest possible container, that is
# strong evidence against rung two everywhere. Alternatives considered:
# 2-D torus grid (richer, but attractor census is no longer closed-form)
# and coupled logistic maps (bqsm-adjacent, but chaotic attractors need
# the clustering machinery this design deliberately avoids).

# ===========================================================================
# RESULTS LOG — first execution, 2026-07-20 (Claude chat container, numpy 2.4.4)
# ===========================================================================
# Config: N=16, K=1.0, lens ω[0]+=0.35, kicks: 16 sites × amps (1.0, 2.2)
#
# FINDING 1 — LOCALITY HOLDS at this operating point:
#   dist 0: 0.200 | dist 1: 0.200 | dist ≥2: 0.000 (exactly zero, 140 entries)
#   overall Hamming 6/160 = 0.037; adjacent (d≤2) 0.120 vs distant (d≥5) 0.000
#   Changed transitions confined to kicks landing on or beside the lens.
#
# FINDING 2 — the lens edits the ALPHABET, not just the table:
#   pre-lens states {−3..+3}; post-lens {−2..+2}. The |q|=3 twisted states
#   (marginal at N=16, stability boundary |q|<N/4=4) were destabilized by a
#   single-site detuning. A lens can delete states — rung-two editing has
#   both a transition-rewiring channel and a state-deletion channel.
#
# BUG FOUND AND FIXED BY THE FIRST RUN:
#   Fixed-clock settling (60 units) + 1e-6 lock threshold misclassified every
#   post-lens attractor as dead: the slowest ring mode decays at ~K(2π/N)²
#   ≈ 0.15, leaving ~1e-4 residual at t=60. Replaced with convergence-based
#   chunked settling. Lesson recorded: settle criteria must be derived from
#   the fabric's spectral gap, not chosen as round numbers.
#
# NOT YET ESTABLISHED (do not overclaim):
#   - One lens strength only. Δ-sweep (0.1 → 0.7) timed out in the chat
#     container as marginal states multiply settle cost; run on the fleet.
#   - One container (Kuramoto ring), one size (N=16), kick alphabet only.
#   - Locality here ≠ locality in bqsm.py's fabric; this is the friendliest
#     container. Port the identical protocol there next.
# ===========================================================================
