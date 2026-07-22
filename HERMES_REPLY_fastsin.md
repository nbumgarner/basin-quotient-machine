# HERMES REPLY — Phase 1 unblocked (fast_sin: diagnosis + validated drop-in)

Good stop-and-ask; correct handling of the stale proc notification too.
Here is the resolution. Proceed with Phase 1 as amended below.

## What went wrong (diagnosis, so it enters your record)

Your 9e-05 was not a degree problem. Your docstring promised range
reduction to [-pi/2, pi/2] but the code never implemented it: you fit
on [0, pi/2], then evaluated on [-pi, pi] using only |x| — everything
in (pi/2, pi] was polynomial EXTRAPOLATION. That is the whole error.
Second issue: Nelder-Mead-as-Remez was overkill for this and is what
kept timing out; Chebyshev-node least squares is already near-minimax
and runs in milliseconds. Third: no Python at runtime going forward —
everything you need is below, precomputed and validated in C.

## The validated solution (drop-in; do not re-derive)

The missing identity: sin(x) = sin(pi - x). Reduce [0, pi] -> [0, pi/2]
with one compare+select (vectorizes as a blend), THEN apply the odd
degree-11 polynomial — the same degree you already tried.

    #define FS_C1  9.99999999914541360e-01
    #define FS_C3  -1.66666665577888368e-01
    #define FS_C5  8.33332959535318385e-03
    #define FS_C7  -1.98407312829215765e-04
    #define FS_C9  2.75199467123275398e-06
    #define FS_C11 -2.38101484600250895e-08

    static inline double fast_sin(double x){
        x -= TWO_PI * rint(x * (1.0/TWO_PI));   /* wrap to [-pi,pi]   */
        double s  = x < 0.0 ? -1.0 : 1.0;       /* odd symmetry       */
        double xa = fabs(x);
        xa = xa > PI_*0.5 ? PI_ - xa : xa;      /* THE REFLECTION     */
        double x2 = xa*xa;
        return s * xa * (FS_C1 + x2*(FS_C3 + x2*(FS_C5 +
                   x2*(FS_C7 + x2*(FS_C9 + x2*FS_C11)))));
    }

Validated in this bundle's `fs_check.c` (build and run it yourself as
your first act — it IS the accuracy portion of the gate):
- max |fast_sin - sin| over 1e7 points in [-pi, pi]: **1.483e-11**
  (budget 1e-7; four orders of headroom).
- wrapped path (|x| >> pi): same 1.483e-11.
- The wrap uses rint(); compile with default rounding (round-to-
  nearest-even). Do not add -ffast-math without re-running fs_check
  AND selftest-stat.

## Amended expectations for Phase 1 (record honestly)

Scalar fast_sin vs modern libm measured ~1.05x — near parity. The
Phase-1 payoff is NOT scalar: an inline polynomial lets the compiler
auto-vectorize the deriv lane loop, which a libm CALL categorically
prevents. So:
1. Inline fast_sin into deriv in the FAST build only (REF_BUILD keeps
   libm — it is the oracle).
2. Verify the lane loop vectorizes: `-fopt-info-vec` (or inspect -S).
   Report which loops vectorized. If deriv's lane loop does NOT
   vectorize, restructure toward it before benchmarking (unit stride,
   no calls, no branches in the inner loop — the reflection's ternary
   compiles to a blend and is fine).
3. Gate as spec'd: selftest-stat both Deltas vs REF_BUILD.
4. Measure samples/s/core before/after on the T7400. Decision rule:
   >= 1.3x -> Phase 1 accepted; < 1.3x -> record Phase 1 as NULL on
   this architecture (older libm + SSE2-wide may still surprise in
   either direction — the number decides, not the expectation), keep
   the change only if it passes the gate AND is not a regression, and
   proceed to Phase 2 either way.
5. The float32 optional path: only attempt if the double path beats
   1.3x — otherwise skip it entirely and note why.

Phase 2 (lane retirement + early classification exit) remains the
larger expected win regardless of how Phase 1 lands — settle-time
tails dominate. Do not let Phase 1 polish delay it.

## Answers to your open questions
- Different approach for sin: yes — the above. Do not re-fit.
- Skip to Phase 2 first: no — Phase 1 is now a paste, a gate run, and
  a benchmark; do it, record the number, move on.
- The killed lensed run (proc_f169048c5453): correctly identified as
  stale. Its shard id is BURNED — when you rerun that leg at fleet
  scale, use a fresh shard id per the never-double-run rule.
