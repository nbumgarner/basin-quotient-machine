# HERMES WORK ORDER — Octopus Engine, Phases 0–2 (T7400 local)

You are Hermes (DeepSeek-v4-pro) running locally on the Dell T7400.
This work order accompanies `hermes-octopus-update.tar.gz` (the research
brief, `ring_furnace.c` v2, `octopus.c`, `octopus_diff.csv`). Read
`HERMES_UPDATE_octopus.md` first; its Sections 4 (instrument lessons),
6 (novelty seam), and 7 (escalation conditions) remain binding. This
document adds an engineering mission on top: turn the furnace into a
high-throughput engine, then spend the throughput on the two open
hypotheses. Do **not** improvise beyond the specs below; where a spec
says "measure and report," measure and report — do not guess.

## 0. Your hardware, stated so you optimize for what exists

T7400: 2x Xeon Harpertown = 8 cores / 8 threads, **SSE4.1 only** — no
AVX, no FMA, 16 XMM registers, 128-bit vectors (2 doubles or 4 floats
wide). 24 MB L2 total. Memory bandwidth ~6.5 GB/s but IRRELEVANT here:
one batch's working set (~230 KB at N=16, L=256) lives in L2. This
workload is compute-bound on sin(). Consequences: `-march=native` gives
SSE4.1 codegen; vectorization win is 2x (double) / 4x (float), not the
8x an AVX machine would see; there is no FMA so polynomial evaluation
uses mul+add — still a large win over libm calls. Do not use intrinsics:
dual-arch cleanliness is a standing contract; achieve vectorization via
plain C the compiler can auto-vectorize, verified by inspection of `-S`
output or `-fopt-info-vec` (report which loops vectorized).

## 1. The two-tier acceptance gate (build this FIRST — everything else
##    passes through it)

Two builds from one source, selected by a compile flag:

- **REFERENCE build** (`-DREF_BUILD`): libm sin, DT=0.02, current
  settle semantics. `./ring_furnace selftest` must remain bit-anchored
  ALL PASS (energies to 1e-5; Delta=0.35 table exactly 6/160; alphabet
  {-2..+2}). This build never changes behavior. It is the oracle.
- **FAST build** (default): everything you optimize. It is gated by a
  NEW subcommand you implement: `selftest-stat`, which runs the octopus
  battery at fixed parameters (seed fixed, samples=4000, rays=32,
  trials=400, Delta in {0.00, 0.35}) on BOTH builds and compares:
  * k(gauss) within 2 sigma (combine fit standard errors; if you do not
    propagate fit errors, use bootstrap over 8 resamples).
  * orbit-metric mean distance within 2 * SEM (SEM = sd/sqrt(samples)).
  * every plateau point p(alpha) within 2 * sqrt(p(1-p)/trials).
  * alphabet identical at both Deltas (+/-3 dead at 0.35, +/-2 alive).
  * Gaussian-vs-exponential verdict identical.
  PASS = all criteria at both Deltas. Print each comparison with its
  tolerance. A FAIL anywhere = the optimization is rejected, not the
  tolerance loosened.

Rule restated from the brief, amended for this mission: the five
protected functions (selftest, deriv, settle, winding, build_table) MAY
now be modified — but only through this gate, one change at a time,
with the reference build untouched as oracle. Never silently.

## 2. Phase 0 — freebies (do these in order, gate after each)

0a. **Shard-seed fix** (flagged in the brief): add an optional shard-id
    argument to `octopus` and fold it into the RNG seed
    (e.g., seed ^= 0x9E3779B97F4A7C15 * (shard_id+1)). Same-Delta
    shards must have independent streams. No gate needed (RNG-only),
    but record it.
0b. **Edge-walk deriv**: the current deriv computes 2 sins per site
    (2N per lane per stage). Replace with the edge form: for each edge
    (i, i+1), s = sin(theta_{i+1} - theta_i); out[i] += s;
    out[i+1] -= s; plus out[i] += om[i] initialization. Exactly N sins,
    identical mathematics (antisymmetry of sin), sum(out) = sum(om)
    exactly. Keep the lane loop innermost and unit-stride. Gate: the
    REFERENCE build with edge-walk deriv must still pass bit-anchored
    selftest (it should, to the ULP; if it does not, report the diff —
    do not force it).
0c. **DT sweep**: candidate DT in {0.05, 0.10, 0.20, 0.35, 0.50}.
    For each: does bit-anchored selftest still reproduce 6/160 and the
    alphabet? Take the LARGEST DT that does, then confirm with
    selftest-stat. Report the full sweep table (DT, selftest verdict,
    battery time). Expected: large win; RK4 on this gradient flow is
    stable to DT ~ 0.69.
0d. **Flags**: `-O3 -march=native -funroll-loops`; report samples/s/core
    before and after Phase 0 (use `octopus 0.0 2000 0 0` timing; add a
    `--bench` note in the results log rather than new machinery).

## 3. Phase 1 — sin replacement (one change, heavily gated)

Implement `fast_sin(x)` for x already wrapped to [-pi, pi]:
- Method: odd minimax/Chebyshev polynomial (degree 7–9 in x, i.e.
  terms x, x^3, ..., x^9). Taylor coefficients are NOT acceptable
  (error blows up at range edges); fit minimax coefficients (Remez, or
  least-squares on Chebyshev nodes over [0, pi] — your choice, but
  MEASURE the result).
- Accuracy check (new, part of the gate): max |fast_sin - sin| over
  10^7 uniform points in [-pi, pi], printed by selftest-stat. Budget:
  < 1e-7 for the double path. Report the measured value.
- Range handling: deriv's arguments are phase differences that can
  exceed pi transiently for kicked/probed states — wrap with
  `remainder(x, 2pi)` equivalent BEFORE the polynomial, or a cheaper
  branchless wrap; correctness first, then speed.
- Gate: selftest-stat vs the reference build, both Deltas.
- OPTIONAL (only if the above passes cleanly): a float32 lane path
  behind a second flag, gated identically. If any criterion fails,
  abandon float32 without negotiation — report and move on.

## 4. Phase 2 — batch mechanics (two changes, gate after each)

2a. **Lane retirement**: inside settle, after each chunk, compact
    still-unlocked lanes to the front (ASCENDING row order — the v1
    corruption lesson is in the results log) and shrink the working
    lane count; retired lanes' final states are copied out at
    retirement. Semantics change: lanes no longer share total
    integration time. This is fine for classification but interacts
    with the co-rotation gauge note — the orbit metric already handles
    per-lane drift, which is why the metric fix preceded this phase.
    Gate: selftest-stat both Deltas (the bit-anchored selftest also
    still applies on the reference build if retirement is compiled
    there — simpler: keep retirement OUT of the reference build).
2b. **Early classification exit**: retire a lane when all N wrapped
    gaps lie strictly inside (-pi + m, pi - m) with margin m = 0.1 for
    2 consecutive checks (flow-invariance argument, Groisman
    arXiv:2604.20541 — rigorous for the pristine sine ring; treated as
    heuristic under lens, which is exactly what the Delta=0.35 leg of
    selftest-stat validates). Velocity-spread lock remains as the
    fallback criterion. Gate: selftest-stat both Deltas.

## 5. Fleet extension — Oracle ARM node (dual-arch gate + 4 more cores)

You have passwordless SSH (Tailscale) to the Oracle box:
`ssh ubuntu@100.75.234.123`. It is aarch64 (4x Neoverse N1, NEON
128-bit = same 2-wide-double vector width as your SSE4.1, 24 GB RAM).
Two roles, in this order:

5a. **Cross-arch acceptance (do this at the END of each phase, before
    the science payload).** Copy the current source over and build:
    ```
    scp ring_furnace.c ubuntu@100.75.234.123:~/octopus/
    ssh ubuntu@100.75.234.123 'cd ~/octopus && \
      gcc -O3 -Wall -Wextra -std=c11 -march=native -DREF_BUILD \
          ring_furnace.c -o ring_furnace_ref -lm && \
      gcc -O3 -Wall -Wextra -std=c11 -march=native \
          ring_furnace.c -o ring_furnace -lm && \
      ./ring_furnace_ref selftest && ./ring_furnace selftest-stat'
    ```
    Expected: reference selftest ALL PASS on ARM. If the bit-anchored
    table differs across architectures (x86_64 libm vs ARM libm can
    differ by ULPs, and basin boundaries are the sensitive object),
    that is NOT a failure to hide — it is data. Report exactly which
    table entries flipped; this feeds the deferred quantization study
    (FORK[5]). selftest-stat must pass on ARM against the ARM
    reference build. Report which loops auto-vectorized on aarch64
    (they should — NEON, no intrinsics needed).
5b. **Shard runner (science payload only, after all gates green on
    BOTH architectures).** CAUTION: this box is the always-on
    conductor anchor. Before launching, check for a live daemon
    (`systemctl is-active` the conductor service / `pgrep -f conductor`).
    If active, use 3 cores, not 4, and `nice -n 10` every shard. Run
    shards detached so your session can drop:
    ```
    ssh ubuntu@100.75.234.123 'cd ~/octopus && for i in 0 1 2; do \
      nohup nice -n 10 ./ring_furnace octopus 0.35 2500 25 250 \
        shard_arm_035_$i.csv $((100+i)) > shard_arm_035_$i.log 2>&1 & \
    done'
    ```
    Shard-id offsets: T7400 uses ids 0–15, Oracle uses 100+ — id space
    must not collide or the seed fix from 0a is defeated. Retrieve with
    scp when logs show completion; merge into the fleet CSV. Do not
    store or transcribe any credentials anywhere; the SSH trust is
    Tailscale's job, not yours.

## 6. The science payload (run only after all gates are green on both
##    architectures)

Across both nodes, one process per core (no threading — fleet
doctrine): T7400 contributes 8 slots, Oracle 3–4 (per 5b's contention
check). Per Delta in {0.00, 0.35}, split the brief's Task 1 totals
(20000 samples, 200 rays, 2000 trials) evenly across the available
slots — e.g., 12 slots -> `octopus <delta> 1667 17 167 shard_... <id>`
(round up; exact totals per shard do not matter, only that the merged
counts are reported with the merged denominators).
- Merge CSVs; evaluate H1 (q=0 escape-head growth under lens) and H2
  (plateau depression at alpha >= 0.8) with Bernoulli error bars.
  State each verdict as ESTABLISHED (>3 sigma), SUGGESTIVE (1–3), or
  NULL (<1). Check the alpha=1 cross-check on both Deltas.
- If wall time permits after H1/H2: the brief's Task 2 Delta-grid at
  demo statistics.

## 7. Deliverables

1. `ring_furnace.c` v3: all accepted changes, REF_BUILD flag,
   selftest-stat subcommand, shard-id seeding, and a v3 RESULTS LOG
   entry in the established style (what changed, each gate's numbers,
   throughput table: samples/s/core at v2 baseline, post-Phase-0,
   post-Phase-1, post-Phase-2 on this machine).
2. The DT sweep table and the fast_sin measured max error.
3. Merged `octopus_fleet.csv` + a findings memo for H1/H2 with the
   established/suggestive/null verdicts and numbers.
4. A list of anything rejected by a gate, with the failing criterion.
5. Cross-arch report: ARM selftest/selftest-stat verdicts, any table
   entries that differ x86_64 vs aarch64 (verbatim list — FORK[5]
   input), vectorization report per arch, and samples/s/core on the
   Oracle box alongside the T7400 numbers.

## 8. Escalation (stop and report, do not work around)

- Bit-anchored selftest fails on the REFERENCE build at any point.
- Any selftest-stat criterion fails for a change you believe is
  mathematically exact (e.g., edge-walk deriv) — that signals a bug,
  not a tolerance problem.
- nolock fraction > 1% at any Delta after early-exit is enabled.
- fast_sin measured error > 1e-7 with your best polynomial.
- Any need to alter tolerances, the gate procedure, or this spec.
- Cross-arch: bit-anchored selftest DIVERGES on ARM (report the exact
  entries; do not "fix" it — boundary ULP sensitivity is a finding).
- Oracle node: conductor daemon health degrades during shard runs
  (kill your shards first, then report), or SSH connectivity is lost
  mid-run (shards are nohup-detached and resume-safe by design; verify
  completion via logs before re-launching anything — never double-run
  a shard id).
