#!/usr/bin/env python3
# ===========================================================================
# bqsm_probe.py — BQSM measurement harness, instrument #1: the Markov battery
# ===========================================================================
#   emerging.systems — Basin-Quotient State Machine research fork
#
# QUESTION UNDER TEST
#   Is the basin quotient actually a state machine? A machine's next state
#   must depend only on (current label, input) — not on how the label was
#   reached (route-invariance), not on when it is probed (clock-freedom),
#   and its table must compose (T applied twice = two-step walk).
#
# WHY THE NAIVE TEST IS VACUOUS, AND WHAT WE MEASURE INSTEAD
#   The pristine ring's attractors are exact fixed points of a gradient
#   flow: every preparation route, settled to tolerance, lands on the SAME
#   microstate, so full-settle route-invariance is guaranteed by geometry.
#   Battery A therefore doubles as an instrument check (and logs microstate
#   deviation to prove the routes genuinely differed before settling).
#   The real physics lives in Battery B: probe after a FINITE wait. The
#   fabric then still carries sub-symbolic memory (the decaying transient),
#   and the measurable is the minimum wait after which behavior matches the
#   asymptotic table — the machine's minimum reliable CLOCK PERIOD. That
#   number is a hidden constant of the system, set by the spectral gap.
#
# BATTERIES
#   A  route-invariance : prepare q via distinct routes, full settle, same
#                         probe → outcomes must match; log route microstate
#                         spread so a vacuous pass is visible as vacuous.
#   B  clock-pressure   : arrive at q via a real transition, wait T_wait,
#                         probe; sweep T_wait. Fidelity(T_wait) → clock spec.
#   C  composition      : settle-free two-kick path vs two table lookups.
#
# HARNESS
#   Every trial is one CSV row keyed by a deterministic trial_id; on start
#   the harness reads the CSV and skips completed ids — kill it anywhere,
#   rerun the same command, it resumes (WAL doctrine, applied to science).
#   Batteries are runnable independently:  bqsm_probe.py A|B|C [csv_path]
#   No RNG anywhere in dynamics; the one stochastic prep uses a per-trial
#   seeded Generator, so any row is exactly reproducible from its id.
# ===========================================================================

import numpy as np                      # oscillator field math
import csv, os, sys                     # results log + resume + CLI

# --- Fabric parameters: identical to bqsm_torus.py so results cross-compare.
N        = 16                           # oscillators on the ring
K        = 1.0                          # coupling strength
Q_MAX    = 2                            # use robust states only; |q|=3 is
                                        # marginal (bqsm_torus finding 2) and
                                        # would tangle marginality with memory
DT       = 0.02                         # RK4 step
T_CHUNK  = 60.0                         # settling chunk length
LOCK_TOL = 1e-8                         # phase-lock criterion (spread of dθ/dt)
PROBE_SITES = (0, 5, 11)                # spread around the ring (no lens here,
                                        # so site choice is symmetry-arbitrary)
PROBE_AMPS  = (1.0, 2.2)                # sub- and super-critical, as before
WAITS = (2.0, 5.0, 10.0, 20.0, 40.0)   # clock-pressure sweep (Battery B)

# ---------------------------------------------------------------------------
# Dynamics core (self-contained duplicate of bqsm_torus.py's, so this file
# runs standalone inside the kit; divergence risk accepted and noted).
# ---------------------------------------------------------------------------
def deriv(theta):
    """dθ/dt on the pristine ring (uniform ω = 0; lensed probes are fleet work)."""
    return K * (np.sin(np.roll(theta, -1) - theta) +   # right-neighbor pull
                np.sin(np.roll(theta,  1) - theta))    # left-neighbor pull

def integrate(theta, t_total):
    """Fixed-grid RK4 — deterministic by construction, no adaptivity to vary."""
    for _ in range(int(t_total / DT)):                 # fixed step count
        k1 = deriv(theta)                              # slope at start
        k2 = deriv(theta + 0.5*DT*k1)                  # midpoint slope (k1)
        k3 = deriv(theta + 0.5*DT*k2)                  # midpoint slope (k2)
        k4 = deriv(theta +     DT*k3)                  # endpoint slope
        theta = theta + (DT/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return np.mod(theta, 2*np.pi)                      # wrapped phases

def lock_spread(theta):
    """max−min of velocities; 0 ⇒ phase-locked (co-rotation-safe criterion)."""
    v = deriv(theta)                                   # instantaneous field
    return float(np.max(v) - np.min(v))                # lock residual

def settle(theta, max_chunks=12):
    """Chunked integration until locked; returns (state, ok, chunks_used).
    Convergence-based, per the settle-tolerance lesson logged in bqsm_torus."""
    for c in range(1, max_chunks + 1):                 # bounded patience
        theta = integrate(theta, T_CHUNK)              # one chunk forward
        if lock_spread(theta) < LOCK_TOL:              # locked to tolerance?
            return theta, True, c                      # converged attractor
    return theta, False, max_chunks                    # never locked

def winding(theta):
    """Topological state label: total wrapped phase winding, snapped to int."""
    d = np.diff(np.concatenate([theta, theta[:1]]))    # ring-closed diffs
    d = np.mod(d + np.pi, 2*np.pi) - np.pi             # wrap to (−π, π]
    return int(np.rint(np.sum(d) / (2*np.pi)))         # the invariant

def twisted(q):
    """Ideal twisted state for winding q (the analytic attractor location)."""
    return np.mod(2*np.pi*q*np.arange(N)/N, 2*np.pi)   # θ_i = 2πqi/N

def kick(theta, site, amp, width=3):
    """Input symbol: add `amp` to a `width`-site window centered at `site`."""
    out = theta.copy()                                 # never mutate input
    for k in range(-(width//2), width//2 + 1):         # window sweep
        out[(site + k) % N] += amp                     # ring-wrapped write
    return np.mod(out, 2*np.pi)                        # wrapped result
# [Block rationale] Duplicated ~50 lines instead of importing bqsm_torus:
# the kit's files must each run standalone on any fleet node with only
# numpy present. The parameters header is the single point of truth to keep
# in sync; a shared bqsm_core.py is the right refactor once the instrument
# count exceeds three.

# ---------------------------------------------------------------------------
# Asymptotic transition table, built lazily and cached per process — the
# reference behavior that Batteries B and C measure deviations FROM.
# ---------------------------------------------------------------------------
_TT = {}                                               # (q,site,amp) -> q' or None
_ATT = {}                                              # q -> settled attractor

def attractor(q):
    """Settled attractor for label q (cached)."""
    if q not in _ATT:                                  # first request
        th, ok, _ = settle(twisted(q))                 # relax the ideal twist
        _ATT[q] = th if (ok and winding(th) == q) else None
    return _ATT[q]

def table(q, site, amp):
    """Asymptotic T[q][site,amp] (cached); None if start dead or no lock."""
    key = (q, site, amp)                               # table coordinate
    if key not in _TT:                                 # first request
        th0 = attractor(q)                             # true start microstate
        if th0 is None: _TT[key] = None                # dead start state
        else:
            th, ok, _ = settle(kick(th0, site, amp))   # kick then relax fully
            _TT[key] = winding(th) if ok else None     # asymptotic outcome
    return _TT[key]

# ---------------------------------------------------------------------------
# Harness: CSV rows + checkpoint-resume keyed on deterministic trial ids.
# ---------------------------------------------------------------------------
FIELDS = ["trial_id", "battery", "q", "route", "wait", "site", "amp",
          "outcome", "reference", "match", "micro_dev", "chunks"]

def load_done(path):
    """Set of trial_ids already recorded — the resume point."""
    if not os.path.exists(path): return set()          # fresh run
    with open(path) as f:                              # scan existing log
        return {row["trial_id"] for row in csv.DictReader(f)}

def open_log(path):
    """Append-mode CSV writer; writes the header only on file creation."""
    new = not os.path.exists(path)                     # header needed?
    f = open(path, "a", newline="")                    # append forever
    w = csv.DictWriter(f, fieldnames=FIELDS)           # fixed schema
    if new: w.writeheader()                            # once, at birth
    return f, w

def emit(f, w, **row):
    """Write one trial row and flush — a kill can lose at most zero rows."""
    w.writerow(row); f.flush()                         # durability per trial

# ---------------------------------------------------------------------------
# Preparation routes for Battery A/B: distinct histories ending at label q.
# ---------------------------------------------------------------------------
def find_entry_kick(q_from, q_to):
    """Search the probe alphabet for a kick that maps q_from -> q_to; the
    cache makes repeated searches free. Returns (site, amp) or None."""
    for site in PROBE_SITES:                           # small honest search
        for amp in PROBE_AMPS:                         # over the alphabet
            if table(q_from, site, amp) == q_to:       # transition found?
                return site, amp                       # use it as the route
    return None                                        # no route in alphabet

def prepare(q, route, trial_seed):
    """Return an UNSETTLED microstate whose asymptotic label is q, reached
    via the named route — the histories the Markov property must erase.
      direct : the ideal analytic twist (zero transient)
      noisy  : ideal twist + seeded phase noise (σ=0.3, well inside basin)
      via_up : arrive by real transition q+1 -> q (carries a true transient)
      via_dn : arrive by real transition q-1 -> q
    Returns (microstate, ok)."""
    if route == "direct":                              # zero-history baseline
        return twisted(q), True
    if route == "noisy":                               # seeded, reproducible
        rng = np.random.default_rng(trial_seed)        # per-trial generator
        return np.mod(twisted(q) + rng.normal(0, 0.3, N), 2*np.pi), True
    src = q + 1 if route == "via_up" else q - 1        # neighbor start label
    if abs(src) > Q_MAX: return None, False            # off the alphabet
    ek = find_entry_kick(src, q)                       # a real q-changing kick
    if ek is None: return None, False                  # no such route exists
    th0 = attractor(src)                               # settled source state
    if th0 is None: return None, False                 # dead source
    return kick(th0, *ek), True                        # mid-transition state!
# [Block rationale] via_up/via_dn return the state IMMEDIATELY after the
# entry kick, before any settling — maximal legitimate transient. Battery A
# settles it fully (history should be erased); Battery B settles it only
# for T_wait (history is still decaying) — same preparation, two questions.

# ---------------------------------------------------------------------------
# Battery A — route-invariance at full settle (+ vacuity instrumentation).
# ---------------------------------------------------------------------------
def battery_A(csv_path):
    done = load_done(csv_path); f, w = open_log(csv_path)
    routes = ("direct", "noisy", "via_up", "via_dn")   # the history alphabet
    for q in range(-Q_MAX, Q_MAX + 1):                 # every robust state
        settled = {}                                   # route -> settled state
        for route in routes:                           # prepare each history
            tid_base = f"A|{q}|{route}"                # id prefix for skips
            th, ok = prepare(q, route, abs(hash(tid_base)) % 2**32)
            if not ok: continue                        # route unavailable
            th_s, lock, ch = settle(th)                # erase the history
            if not lock or winding(th_s) != q: continue# prep failed: skip q/route
            settled[route] = (th_s, ch)                # keep for probing
        if "direct" not in settled: continue           # need the baseline
        ref_state = settled["direct"][0]               # microstate reference
        for route, (th_s, ch) in settled.items():      # each surviving route
            # Vacuity meter: if routes converge to identical microstates,
            # a pass proves geometry, not physics — record the deviation.
            dev = float(np.max(np.abs(np.angle(np.exp(1j*(th_s - ref_state))))))
            for site in PROBE_SITES:                   # every probe symbol
                for amp in PROBE_AMPS:
                    tid = f"A|{q}|{route}|{site}|{amp}"# deterministic id
                    if tid in done: continue           # resume: skip done
                    th, ok, ch2 = settle(kick(th_s, site, amp))  # probe
                    out = winding(th) if ok else None  # observed outcome
                    ref = table(q, site, amp)          # asymptotic truth
                    emit(f, w, trial_id=tid, battery="A", q=q, route=route,
                         wait="", site=site, amp=amp, outcome=out,
                         reference=ref, match=int(out == ref),
                         micro_dev=f"{dev:.2e}", chunks=ch2)
    f.close()

# ---------------------------------------------------------------------------
# Battery B — clock pressure: probe mid-transient, sweep the wait.
# ---------------------------------------------------------------------------
def battery_B(csv_path):
    done = load_done(csv_path); f, w = open_log(csv_path)
    for q in (-1, 0, 1):                               # central, robust states
        for route in ("via_up", "via_dn"):             # real-transition arrival
            th_raw, ok = prepare(q, route, 0)          # mid-transition state
            if not ok: continue                        # route unavailable
            for wait in WAITS:                         # the clock sweep
                th_w = integrate(th_raw.copy(), wait)  # PARTIAL settle only
                lbl = winding(th_w)                    # label at probe time
                for site in PROBE_SITES:               # every probe symbol
                    for amp in PROBE_AMPS:
                        tid = f"B|{q}|{route}|{wait}|{site}|{amp}"
                        if tid in done: continue       # resume: skip done
                        th, okp, ch = settle(kick(th_w, site, amp))
                        out = winding(th) if okp else None
                        # Reference = what the table predicts from the label
                        # the machine WOULD read at probe time. If the label
                        # itself hasn't arrived at q yet, that is also a
                        # clock violation — captured because reference uses
                        # the intended q, not the transient label.
                        ref = table(q, site, amp)      # intended-state truth
                        emit(f, w, trial_id=tid, battery="B", q=q, route=route,
                             wait=wait, site=site, amp=amp, outcome=out,
                             reference=ref, match=int(out == ref),
                             micro_dev=f"label@probe={lbl}", chunks=ch)
    f.close()

# ---------------------------------------------------------------------------
# Battery C — composition: physical two-kick path vs two table lookups.
# ---------------------------------------------------------------------------
def battery_C(csv_path):
    done = load_done(csv_path); f, w = open_log(csv_path)
    pairs = [((s1, a1), (s2, a2)) for s1 in PROBE_SITES for a1 in PROBE_AMPS
             for s2 in PROBE_SITES for a2 in PROBE_AMPS][::3]  # every 3rd: 12 pairs
    for q in (-1, 0, 1):                               # central states
        for (s1, a1), (s2, a2) in pairs:               # sampled kick pairs
            tid = f"C|{q}|{s1}|{a1}|{s2}|{a2}"         # deterministic id
            if tid in done: continue                   # resume: skip done
            q1 = table(q, s1, a1)                      # table step one
            ref = table(q1, s2, a2) if q1 is not None and abs(q1) <= Q_MAX else None
            th0 = attractor(q)                         # physical path start
            th1, ok1, _ = settle(kick(th0, s1, a1))    # step one, settled
            out = None                                 # pessimistic default
            if ok1:                                    # step one locked
                th2, ok2, ch = settle(kick(th1, s2, a2))  # step two, settled
                out = winding(th2) if ok2 else None    # physical outcome
            emit(f, w, trial_id=tid, battery="C", q=q, route="compose",
                 wait="", site=f"{s1}>{s2}", amp=f"{a1}>{a2}",
                 outcome=out, reference=ref, match=int(out == ref),
                 micro_dev="", chunks="")
    f.close()

# ---------------------------------------------------------------------------
# Report: fold the CSV into the verdict table.
# ---------------------------------------------------------------------------
def report(csv_path):
    if not os.path.exists(csv_path): print("no data"); return
    rows = list(csv.DictReader(open(csv_path)))        # full log in memory
    for bat in ("A", "B", "C"):                        # per battery
        sel = [r for r in rows if r["battery"] == bat] # its rows
        if not sel: continue                           # not run yet
        if bat == "B":                                 # clock sweep: by wait
            print("Battery B — table fidelity vs wait (clock pressure):")
            for wait in sorted({r["wait"] for r in sel}, key=float):
                ws = [r for r in sel if r["wait"] == wait]
                m = sum(int(r["match"]) for r in ws)   # matches at this wait
                print(f"  wait {float(wait):5.1f}: {m}/{len(ws)} = {m/len(ws):.3f}")
        else:                                          # A and C: single rate
            m = sum(int(r["match"]) for r in sel)      # total matches
            print(f"Battery {bat}: {m}/{len(sel)} = {m/len(sel):.3f} match")
            if bat == "A":                             # vacuity disclosure
                devs = [r["micro_dev"] for r in sel if r["route"] != "direct"]
                if devs: print(f"  microstate dev (non-direct routes): "
                               f"max {max(devs)}  — near-zero ⇒ pass is "
                               f"geometric, as predicted for fixed points")

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ABC"        # battery select
    path  = sys.argv[2] if len(sys.argv) > 2 else "markov.csv" # results log
    if "A" in which: battery_A(path)                   # route-invariance
    if "B" in which: battery_B(path)                   # clock pressure
    if "C" in which: battery_C(path)                   # composition
    report(path)                                       # always summarize
