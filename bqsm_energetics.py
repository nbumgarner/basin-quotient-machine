#!/usr/bin/env python3
# ===========================================================================
# bqsm_energetics.py — the machine's thermodynamic anatomy
# ===========================================================================
#   emerging.systems — Basin-Quotient State Machine research fork
#
# WHAT THIS MEASURES, AND WHY IT CAN BE EXACT
#   The symmetric ring is a GRADIENT FLOW:  dθ/dt = −∇V  with
#       V(θ) = −K Σ_i cos(θ_{i+1} − θ_i)
#   so the machine has a true energy landscape. Attractor energies are
#   analytic:  V(q) = −K·N·cos(2πq/N). Saddles between adjacent basins are
#   index-1 equilibria findable by Newton's method to machine precision —
#   no sampling, no estimation. From these we extract:
#
#   1  BARRIER MAP      ΔV(q → q±1) = V(saddle) − V(q). Prediction to test:
#                       barriers shrink with |q|, which would explain the
#                       |q|=3 deaths in bqsm_torus.py QUANTITATIVELY.
#   2  INSTRUCTION COST a*(q, amp-direction): the critical kick amplitude
#                       at the transition flip, found by bisection, and the
#                       potential energy the kick actually injects there.
#                       efficiency = barrier / injected — how well a
#                       localized kick couples to the saddle direction.
#                       This is the machine's cost-per-transition figure.
#   3  INSTRUCTION ALGEBRA  Ring symmetry predicts outcomes depend only on
#                       (q, amp), not site. Verify, then read the collapsed
#                       table as a function on winding numbers: fixed
#                       points, absorbing states, reversibility.
#
# Everything is deterministic; Newton residuals and Hessian indices are
# printed so every claimed saddle carries its own proof of saddlehood.
# ===========================================================================

import numpy as np                      # field math + linear algebra
import sys                              # exit code

# --- Fabric parameters (identical across the instrument suite) -------------
N, K     = 16, 1.0                      # ring size, coupling
Q_MAX    = 3                            # include marginal states: they're
                                        # the test of prediction 1
DT       = 0.02                         # RK4 step
T_CHUNK  = 60.0                         # settling chunk
LOCK_TOL = 1e-8                         # lock criterion
KICK_W   = 3                            # kick window width (as before)

# ---------------------------------------------------------------------------
# Dynamics + potential (the gradient pair — deriv IS −∇V, stated explicitly
# so the energy bookkeeping below is grounded in the same code that flows).
# ---------------------------------------------------------------------------
def deriv(theta):
    """dθ/dt = −∇V: nearest-neighbor sine coupling on the ring."""
    return K * (np.sin(np.roll(theta, -1) - theta) +
                np.sin(np.roll(theta,  1) - theta))

def V(theta):
    """The exact potential. Every energy number in this file is this sum."""
    return float(-K * np.sum(np.cos(np.roll(theta, -1) - theta)))

def integrate(theta, t_total):
    """Fixed-grid RK4 (deterministic)."""
    for _ in range(int(t_total / DT)):
        k1 = deriv(theta); k2 = deriv(theta + 0.5*DT*k1)
        k3 = deriv(theta + 0.5*DT*k2); k4 = deriv(theta + DT*k3)
        theta = theta + (DT/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return np.mod(theta, 2*np.pi)

def settle(theta, max_chunks=12):
    """Chunked integrate-until-locked; (state, ok)."""
    for _ in range(max_chunks):
        theta = integrate(theta, T_CHUNK)
        v = deriv(theta)
        if float(np.max(v) - np.min(v)) < LOCK_TOL:
            return theta, True
    return theta, False

def winding(theta):
    """Topological state label."""
    d = np.diff(np.concatenate([theta, theta[:1]]))
    d = np.mod(d + np.pi, 2*np.pi) - np.pi
    return int(np.rint(np.sum(d) / (2*np.pi)))

def twisted(q):
    """Analytic attractor for winding q."""
    return np.mod(2*np.pi*q*np.arange(N)/N, 2*np.pi)

def kick(theta, site, amp):
    """The instruction: amp added to a KICK_W window at `site`."""
    out = theta.copy()
    for k in range(-(KICK_W//2), KICK_W//2 + 1):
        out[(site + k) % N] += amp
    return np.mod(out, 2*np.pi)

# ---------------------------------------------------------------------------
# PART 1 — Saddles by Newton, barriers by subtraction.
# ---------------------------------------------------------------------------
# Equilibria solve F(θ)=deriv(θ)=0. Global rotation gives F a permanent zero
# mode, so we pin θ_0 = 0 and Newton-solve the remaining N−1 components.
# Saddle seeds: the classic single-phase-slip picture — between q and q+1
# the transition state concentrates one extra 2π of winding as a localized
# slip. Seed = attractor q with a smooth localized bump of height π inserted,
# which sits near the ridge; Newton does the rest. Saddlehood is then PROVEN
# per-solution by the Hessian index (exactly one negative eigenvalue).
# ---------------------------------------------------------------------------
def jac(theta):
    """Analytic Jacobian of deriv (tridiagonal + corner terms on the ring)."""
    J = np.zeros((N, N))                                # dense is fine at N=16
    for i in range(N):                                  # each oscillator row
        r, l = (i + 1) % N, (i - 1) % N                 # ring neighbors
        cr = K * np.cos(theta[r] - theta[i])            # right-link stiffness
        cl = K * np.cos(theta[l] - theta[i])            # left-link stiffness
        J[i, i] = -(cr + cl)                            # self term
        J[i, r] += cr                                   # right coupling
        J[i, l] += cl                                   # left coupling
    return J

def newton_equilibrium(seed, iters=200):
    """Newton with θ_0 pinned; returns (θ, residual) or (None, inf)."""
    th = seed.copy(); th -= th[0]                       # gauge-fix start
    for _ in range(iters):                              # damped Newton loop
        F = deriv(th)                                   # residual field
        if float(np.max(np.abs(F))) < 1e-12:            # converged hard
            return np.mod(th, 2*np.pi), float(np.max(np.abs(F)))
        J = jac(th)[1:, 1:]                             # pinned subsystem
        try:
            step = np.linalg.solve(J, -F[1:])           # Newton step
        except np.linalg.LinAlgError:
            return None, np.inf                         # singular: give up
        s = 1.0                                         # damping factor
        while s > 1e-4:                                 # backtracking line search
            trial = th.copy(); trial[1:] += s * step    # damped update
            if np.max(np.abs(deriv(trial))) < np.max(np.abs(F)):
                th = trial; break                       # residual decreased
            s *= 0.5                                    # damp harder
        else:
            return None, np.inf                         # line search failed
    return None, np.inf                                 # never converged

def hessian_index(theta):
    """Number of unstable directions (Hessian of V = −Jacobian of flow),
    ignoring the global-rotation zero mode. 0 = minimum, 1 = saddle."""
    w = np.linalg.eigvalsh(-jac(theta))                 # V's Hessian spectrum
    w = w[np.abs(w) > 1e-9]                             # drop the zero mode
    return int(np.sum(w < 0)), w                        # index + spectrum

def slip_seed(q, direction):
    """Seed near the q → q+direction transition state: attractor q plus a
    smooth localized bump carrying ±π of extra phase at one site."""
    th = twisted(q).copy()                              # start on the attractor
    bump = direction * np.pi * np.exp(                  # gaussian bump, width~1.2
        -0.5 * ((np.arange(N) - N//2) / 1.2) ** 2)      # centered mid-ring
    return th + bump                                    # ridge-adjacent guess

def part1_barriers():
    print("PART 1 — exact landscape")
    print(f"{'q':>3} {'V(q) numeric':>14} {'V(q) analytic':>14}")
    for q in range(-Q_MAX, Q_MAX + 1):                  # attractor energies
        va = -K * N * np.cos(2*np.pi*q/N)               # closed form
        vn = V(twisted(q))                              # from the code's V
        print(f"{q:>3} {vn:>14.6f} {va:>14.6f}")        # must agree exactly
    print(f"\n{'transition':>12} {'V(saddle)':>12} {'barrier ΔV':>12} "
          f"{'index':>6}  residual")
    barriers = {}                                       # (q,dir) -> ΔV
    for q in range(0, Q_MAX):                           # q → q+1 (symmetry
        for direction in (+1,):                         # gives the negatives)
            th_s, res = newton_equilibrium(slip_seed(q, direction))
            if th_s is None:                            # Newton failed: report
                print(f"{q:>4}→{q+direction:<5} Newton failed"); continue
            idx, _ = hessian_index(th_s)                # prove saddlehood
            dV = V(th_s) - V(twisted(q))                # barrier height
            barriers[(q, direction)] = dV               # record
            print(f"{q:>4}→{q+direction:<7} {V(th_s):>12.6f} {dV:>12.6f} "
                  f"{idx:>6}  {res:.1e}")
    return barriers

# ---------------------------------------------------------------------------
# PART 2 — Instruction cost: critical amplitude by bisection + injected
# energy vs barrier at the flip point.
# ---------------------------------------------------------------------------
def outcome(q, site, amp):
    """Asymptotic result of instruction (site, amp) from state q."""
    th, ok = settle(kick(twisted(q), site, amp))        # kick then relax
    return winding(th) if ok else None                  # label or no-lock

def critical_amp(q, site, lo=0.0, hi=3.5, steps=16):
    """Smallest amplitude at `site` that changes the state label, by
    bisection on the first flip; returns (a*, q_target) or (None, None)."""
    if outcome(q, site, hi) == q: return None, None     # no flip in range
    for _ in range(steps):                              # bisection loop
        mid = 0.5 * (lo + hi)                           # midpoint amplitude
        if outcome(q, site, mid) == q: lo = mid         # still below ridge
        else: hi = mid                                  # at/above ridge
    return hi, outcome(q, site, hi)                     # a* and destination

def part2_costs(barriers):
    print("\nPART 2 — instruction energetics (site 0; symmetry makes site "
          "choice immaterial — verified in Part 3)")
    print(f"{'q':>3} {'a*':>8} {'target':>7} {'E_injected':>11} "
          f"{'barrier':>9} {'efficiency':>11}")
    for q in range(0, Q_MAX):                           # states with a known
        a, tgt = critical_amp(q, 0)                     # +1 barrier from P1
        if a is None:                                   # no transition found
            print(f"{q:>3}   none in range"); continue
        Ein = V(kick(twisted(q), 0, a)) - V(twisted(q)) # energy the kick adds
        # Efficiency uses the barrier toward the OBSERVED target when we
        # have it; a kick can also overshoot to a non-adjacent state, in
        # which case the adjacent barrier is still the honest denominator
        # floor and we say so in the caption.
        bar = barriers.get((q, +1))                     # adjacent barrier
        eff = (bar / Ein) if (bar is not None and Ein > 0) else float("nan")
        print(f"{q:>3} {a:>8.4f} {tgt:>7} {Ein:>11.4f} "
              f"{bar if bar is not None else float('nan'):>9.4f} {eff:>11.3f}")
    print("efficiency = barrier/E_injected: 1.0 would mean the kick climbs "
          "the saddle direction and nothing else; the shortfall measures how "
          "poorly a 3-site window aligns with the transition mode.")

# ---------------------------------------------------------------------------
# PART 3 — Instruction algebra: symmetry collapse + graph structure.
# ---------------------------------------------------------------------------
def part3_algebra():
    print("\nPART 3 — instruction-set algebra")
    amps = (1.0, 1.6, 2.2, 3.0)                         # widened alphabet
    sites = (0, 5, 11)                                  # symmetry witnesses
    collapsed = {}                                      # (q, amp) -> outcome
    site_dep = 0                                        # symmetry violations
    for q in range(-Q_MAX, Q_MAX + 1):                  # full alphabet
        for amp in amps:                                # each amplitude
            outs = {outcome(q, s, amp) for s in sites}  # across sites
            if len(outs) > 1: site_dep += 1             # symmetry broken?
            collapsed[(q, amp)] = sorted(outs)[0] if len(outs)==1 else tuple(sorted(outs, key=str))
    print(f"site-dependence violations: {site_dep} "
          f"(0 ⇒ table collapses to (q, amp) as symmetry predicts)")
    print(f"\n{'amp':>5} | " + " ".join(f"{q:>4}" for q in range(-Q_MAX, Q_MAX+1)))
    for amp in amps:                                    # collapsed table rows
        row = " ".join(f"{str(collapsed[(q, amp)]):>4}"
                       for q in range(-Q_MAX, Q_MAX+1))
        print(f"{amp:>5} | {row}")
    # Graph reading: which states can reach which under the full alphabet?
    reach = {q: {q} for q in range(-Q_MAX, Q_MAX+1)}    # reflexive closure
    for _ in range(2*Q_MAX + 1):                        # relax to fixpoint
        for q in range(-Q_MAX, Q_MAX+1):                # each node
            for amp in amps:                            # each instruction
                t = collapsed[(q, amp)]                 # its target
                if isinstance(t, int) and abs(t) <= Q_MAX:
                    reach[q] |= reach[t]                # inherit closure
    print("\nreachability under the alphabet:")
    for q in range(-Q_MAX, Q_MAX+1):                    # print closures
        print(f"  {q:>3} → {sorted(reach[q])}")
    sinks = [q for q in reach if reach[q] == {q}]       # absorbing states
    print(f"absorbing states: {sinks if sinks else 'none'}")
    return collapsed

if __name__ == "__main__":
    barriers = part1_barriers()                         # exact landscape
    part2_costs(barriers)                               # instruction cost
    part3_algebra()                                     # instruction algebra
    sys.exit(0)

# [Block rationale] Three instruments share one file because they share one
# object (the potential) and one honesty mechanism: Part 1's Newton residual
# + Hessian index prove each saddle; Part 2's bisection is a measurement
# whose denominator comes from Part 1's proof; Part 3's symmetry check
# validates the site-collapse both other parts quietly assume. Alternatives
# considered: string/NEB methods for saddles (needed for rough landscapes;
# overkill when Newton converges), and Kramers-rate noise probes (the
# independent cross-check — deliberately left for the fleet, where the
# long noisy integrations belong).
# ===========================================================================

# ===========================================================================
# RESULTS LOG — first execution, 2026-07-20 (chat container, numpy 2.4.4)
# ===========================================================================
# PART 1 — landscape (with post-run corrections; the diagnostics worked):
#   V(q) numeric = analytic to 1e-6 for all seven states. Saddle hunt:
#   - "0→1" row: Newton fell back to the q=0 MINIMUM (index 0) — seed
#     failure, not a saddle. Disregard that row's ΔV.
#   - Proven saddle (index 1, residual 1e-14) at V=−13.6490. Energy ordering
#     (above V(0)=−16 and V(1)=−14.78, BELOW V(2)=−11.31) identifies it as
#     the 0↔1 transition state: barrier 1→0 = 1.133, barrier 0→1 = 2.351.
#   - "2→3" row: index-2 equilibrium — real, but not a transition state.
#     True 1↔2 and 2↔3 saddles remain unfound (better seeds: fleet work).
# PART 2 — instruction cost, cross-checking Part 1 independently:
#   q=1 critical bump a*=2.439 (target 0): injects 3.258 across a 1.133
#   barrier → coupling efficiency 0.348. q=0: NO bump ≤3.5 escapes —
#   consistent with barrier 2.351 at ~0.35 efficiency (≈6.7 needed).
#   q=2: a*=1.640 → q=1, injects 1.512 (1↔2 barrier not yet proven).
# PART 3 — instruction algebra (the headline):
#   Site-independence: 0 violations in 84 checks — table provably collapses
#   to (q, amp). Collapsed table is a pure CONTRACTION: no scalar bump at
#   any tested amplitude ever increases |q|; reachability is a funnel with
#   q=0 the unique absorbing state. Localized scalar input can only ERASE.
# WRITE TEST (follow-up falsification run, same session):
#   Localized phase-RAMP inputs: 2π ramp writes 0→1 at widths 4/6/8 and
#   1→2 from q=1 (an increment instruction, +1 per quantum); π ramp writes
#   nothing (sub-quantum injection relaxes away). The instruction set is
#   TYPED and the write is QUANTIZED:
#     read  = winding (topological, free)
#     erase = local scalar bump (funnel toward 0, one-way)
#     write = shaped gradient input, exactly one quantum per 2π, else nil
# NOT YET ESTABLISHED: whether erase-only holds for ALL scalar inputs
#   (tested alphabet only, amps ≤3.5, width 3); true 1↔2 / 2↔3 barriers;
#   whether the trichotomy survives on a lensed fabric or in bqsm.py's
#   equations. The underlying physics (winding quantization, phase slips)
#   is classical; the claimed contribution is the machine-theoretic typing
#   and its energetic accounting — frame it that way, nothing grander.
# ===========================================================================
