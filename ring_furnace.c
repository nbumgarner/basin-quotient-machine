/* ===========================================================================
 * ring_furnace.c — batched ring-fabric integrator for fleet-scale sweeps
 * ===========================================================================
 *
 *   emerging.systems — BQSM instrument suite, native engine
 *
 * WHAT THIS IS
 *   The C descendant of bqsm_torus.py / bqsm_probe.py's dynamics core,
 *   built for the workload the fleet actually has: thousands of INDEPENDENT
 *   rings (parameter grids, trial batches) rather than one ring fast.
 *
 * ARCHITECTURE (the two ideas from the design discussion, implemented)
 *   RING BUFFER  Neighbor access is `(i±1) & (N-1)` — the ring-buffer mask
 *                trick. N must be a power of two; enforced at startup.
 *   SoA LANES    State is theta[site*L + lane]: for a fixed site, all L
 *                lanes are contiguous, so the inner lane loop is a straight
 *                unit-stride array op the compiler auto-vectorizes on both
 *                x86_64 and aarch64. No intrinsics anywhere — dual-arch
 *                cleanliness by construction, per the code contract.
 *   BATCH SETTLE Every lane integrates in lockstep; converged lanes keep
 *                integrating harmlessly (a fixed point is idempotent under
 *                the flow) until the whole batch locks or patience runs out.
 *                Uniform control flow is what keeps the SIMD lanes full.
 *
 * SUBCOMMANDS
 *   ring_furnace selftest
 *       Cross-validation against the Python instruments (the acceptance
 *       test): analytic twisted-state energies, and a full reproduction of
 *       the bqsm_torus.py Δ=0.35 locality experiment (published numbers:
 *       overall Hamming 6/160, zero changes at distance ≥ 2, post-lens
 *       alphabet {−2..+2}).
 *   ring_furnace lens-sweep <d_lo> <d_hi> <steps> <out.csv>
 *       The Δ-sweep that exceeded the Python session budget: per Δ, build
 *       pre/post tables (each table = ONE batched integration of all 224
 *       trials), emit per-distance Hamming rows + alphabet survival.
 *
 * BUILD
 *   gcc -O3 -Wall -Wextra -std=c11 ring_furnace.c -o ring_furnace -lm
 *   (Optionally -ffast-math for vectorized libm sin via libmvec; selftest
 *    MUST still pass afterward — it is the guard for that flag.)
 * ======================================================================= */

#define _GNU_SOURCE                     /* expose M_PI under -std=c11        */
#include <stdio.h>                      /* printf, fopen                     */
#include <stdlib.h>                     /* malloc, atof, exit                */
#include <string.h>                     /* memcpy, strcmp                    */
#include <math.h>                       /* sin, cos, fabs, M_PI              */
#include <stdint.h>                     /* fixed-width ints                  */
#include <time.h>                       /* wall-clock for throughput report  */

/* --- Fabric constants: identical to the Python instruments so every number
 * is directly comparable. N runtime-fixed here (16) but the code paths use
 * the mask throughout, so widening N for clock-scaling work is a two-line
 * change (N, and the Q_MAX stability bound floor(N/4)−1).                   */
#define N        16                     /* sites per ring (power of two)     */
#define NMASK    (N - 1)                /* the ring-buffer index mask        */
#define KCPL     1.0                    /* coupling strength K               */
#define Q_MAX    3                      /* twisted states |q| ≤ 3 at N=16    */
#define DT       0.02                   /* RK4 step (matches Python exactly) */
#define T_CHUNK  60.0                   /* settling chunk length             */
#define LOCK_TOL 1e-8                   /* velocity-spread lock criterion    */
#define MAXCHUNK 12                     /* settling patience, in chunks      */
#define KICK_W   3                      /* kick window width                 */
#define LMAX     256                    /* max lanes per batch (128 KB state)*/

/* Batch state, SoA: value of site i in lane l lives at [i*L + l].           */
typedef struct {
    int     L;                          /* live lanes in this batch          */
    double *th;                         /* phases        [N*L]               */
    double *om;                         /* per-lane ω    [N*L] (lens differs
                                           per experiment, so ω is per-lane
                                           even when identical — uniformity
                                           beats a special case)             */
    double *k1, *k2, *k3, *k4, *tmp;    /* RK4 workspaces [N*L] each         */
} batch_t;

static batch_t *batch_new(int L){
    batch_t *b = malloc(sizeof *b);                     /* the batch object  */
    b->L = L;                                           /* lane count        */
    size_t sz = (size_t)N * L * sizeof(double);         /* one field's bytes */
    b->th = malloc(sz); b->om  = calloc(N * L, sizeof(double));
    b->k1 = malloc(sz); b->k2  = malloc(sz);            /* RK4 slopes        */
    b->k3 = malloc(sz); b->k4  = malloc(sz);
    b->tmp = malloc(sz);                                /* staged state      */
    if (!b->th || !b->om || !b->k1 || !b->k2 || !b->k3 || !b->k4 || !b->tmp)
        { fprintf(stderr, "alloc failed\n"); exit(1); } /* fail loud         */
    return b;
}
static void batch_free(batch_t *b){
    free(b->th); free(b->om); free(b->k1); free(b->k2); /* every field       */
    free(b->k3); free(b->k4); free(b->tmp); free(b);    /* then the shell    */
}

/* dθ/dt for every lane at once. Outer loop = sites (the ring); inner loop =
 * lanes (unit stride — this is the loop the compiler turns into SIMD).      */
static void deriv(const double *th, const double *om, double *out, int L){
    for (int i = 0; i < N; i++) {                       /* each ring site    */
        int ir = ((i + 1) & NMASK) * L;                 /* right neighbor row*/
        int il = ((i - 1) & NMASK) * L;                 /* left neighbor row */
        int ii = i * L;                                 /* this site's row   */
        for (int l = 0; l < L; l++)                     /* vectorized lanes  */
            out[ii + l] = om[ii + l]
                        + KCPL * (sin(th[ir + l] - th[ii + l])
                                +  sin(th[il + l] - th[ii + l]));
    }
}

/* One RK4 step for the whole batch (classic tableau, batched arithmetic).   */
static void rk4_step(batch_t *b){
    int M = N * b->L;                                   /* total scalars     */
    deriv(b->th, b->om, b->k1, b->L);                   /* k1 at start       */
    for (int j = 0; j < M; j++) b->tmp[j] = b->th[j] + 0.5*DT*b->k1[j];
    deriv(b->tmp, b->om, b->k2, b->L);                  /* k2 at midpoint    */
    for (int j = 0; j < M; j++) b->tmp[j] = b->th[j] + 0.5*DT*b->k2[j];
    deriv(b->tmp, b->om, b->k3, b->L);                  /* k3 at midpoint    */
    for (int j = 0; j < M; j++) b->tmp[j] = b->th[j] +     DT*b->k3[j];
    deriv(b->tmp, b->om, b->k4, b->L);                  /* k4 at end         */
    for (int j = 0; j < M; j++)                         /* combine + advance */
        b->th[j] += (DT/6.0)*(b->k1[j] + 2*b->k2[j] + 2*b->k3[j] + b->k4[j]);
}

/* Per-lane lock residual: max−min of dθ/dt down the lane's column — the
 * co-rotation-safe criterion carried over from the Python instruments.      */
static void lock_spread(batch_t *b, double *spread){
    deriv(b->th, b->om, b->k1, b->L);                   /* velocities in k1  */
    for (int l = 0; l < b->L; l++) {                    /* each lane         */
        double lo = b->k1[l], hi = b->k1[l];            /* site 0 seeds      */
        for (int i = 1; i < N; i++) {                   /* remaining sites   */
            double v = b->k1[i * b->L + l];             /* velocity (i, l)   */
            if (v < lo) lo = v;                         /* track min         */
            if (v > hi) hi = v;                         /* track max         */
        }
        spread[l] = hi - lo;                            /* lane residual     */
    }
}

/* Settle the batch: chunked integration until EVERY lane locks (or patience
 * ends). Converged lanes ride along at their fixed points — idempotent, so
 * correctness is untouched and control flow stays uniform for SIMD. Returns
 * per-lane lock flags. Total steps wasted on riders is bounded by the
 * slowest lane, which is exactly the lane we must wait for anyway.          */
static void settle(batch_t *b, int *locked){
    double spread[LMAX];                                /* per-lane residuals*/
    int steps = (int)(T_CHUNK / DT);                    /* steps per chunk   */
    for (int l = 0; l < b->L; l++) locked[l] = 0;       /* none locked yet   */
    for (int c = 0; c < MAXCHUNK; c++) {                /* bounded patience  */
        for (int s = 0; s < steps; s++) rk4_step(b);    /* one chunk forward */
        lock_spread(b, spread);                         /* audit every lane  */
        int all = 1;                                    /* optimism          */
        for (int l = 0; l < b->L; l++) {                /* per-lane verdicts */
            locked[l] = spread[l] < LOCK_TOL;           /* locked now?       */
            all &= locked[l];                           /* batch-wide AND    */
        }
        if (all) return;                                /* everyone locked   */
    }                                                   /* patience expired: */
}                                                       /* flags say who won */

/* Winding number of lane l: wrapped phase differences summed around the
 * ring, snapped to the integer topological invariant.                       */
static int winding(const batch_t *b, int l){
    double s = 0.0;                                     /* winding accumulator*/
    for (int i = 0; i < N; i++) {                       /* around the ring   */
        double d = b->th[((i + 1) & NMASK) * b->L + l]  /* neighbor phase    */
                 - b->th[i * b->L + l];                 /* minus this phase  */
        d = remainder(d, 2.0 * M_PI);                   /* wrap to (−π, π]   */
        s += d;                                         /* accumulate        */
    }
    return (int)lround(s / (2.0 * M_PI));               /* snap to integer   */
}

/* Write the ideal twisted state q into lane l.                              */
static void set_twisted(batch_t *b, int l, int q){
    for (int i = 0; i < N; i++)                         /* θ_i = 2πqi/N      */
        b->th[i * b->L + l] = fmod(2.0*M_PI*q*i/N, 2.0*M_PI);
}

/* Apply the standard 3-site kick to lane l at `site` with amplitude `amp`. */
static void apply_kick(batch_t *b, int l, int site, double amp){
    for (int k = -(KICK_W/2); k <= KICK_W/2; k++)       /* window sweep      */
        b->th[((site + k) & NMASK) * b->L + l] += amp;  /* masked ring index */
}
/* [Block rationale] Everything above is a line-for-line semantic port of
 * the Python dynamics (same DT, same RK4 tableau, same lock criterion) —
 * deliberate, because selftest's job is to prove THIS code and the Python
 * instruments are the same instrument. remainder() replaces the mod-wrap
 * dance because C guarantees its (−π, π] behavior in one call. sin() from
 * libm rather than a polynomial: basin boundaries are the measured object,
 * so we do not trade ULPs for speed until a quantization study (FORK[5])
 * says how many ULPs the boundaries tolerate.                               */

/* ---------------------------------------------------------------------------
 * TRANSITION TABLE — one batched integration per table.
 * ---------------------------------------------------------------------------
 * Trial layout: for each live start state q, N sites × 2 amps = 32 trials;
 * 7 states → up to 224 lanes. Phase A settles the 7 bare attractors (one
 * tiny batch); Phase B loads every kicked copy into one big batch and
 * settles them all simultaneously — the whole table in one settle() call.
 * ------------------------------------------------------------------------- */
static const double AMPS[2] = {1.0, 2.2};               /* the kick alphabet */
#define TBL_IDX(q, s, a) ((((q) + Q_MAX) * N + (s)) * 2 + (a))
#define TBL_SZ ((2*Q_MAX + 1) * N * 2)                  /* 224 entries       */
#define DEAD  (-99)                                     /* start state dead  */
#define NOLCK (-98)                                     /* trial never locked*/

/* Build the table for lens configuration (site, delta); delta 0 = pristine.
 * Fills table[TBL_SZ]; alive[q+Q_MAX] flags which start states survived.    */
static void build_table(int lens_site, double lens_delta,
                        int *table, int *alive){
    /* Phase A — settle the 7 candidate attractors under this ω profile.     */
    batch_t *ba = batch_new(2*Q_MAX + 1);               /* one lane per q    */
    for (int q = -Q_MAX; q <= Q_MAX; q++) {             /* load ideal twists */
        set_twisted(ba, q + Q_MAX, q);                  /* lane = q + Q_MAX  */
        ba->om[lens_site * ba->L + (q + Q_MAX)] = lens_delta; /* the lens    */
    }
    int lockA[LMAX]; settle(ba, lockA);                 /* relax them all    */
    double att[2*Q_MAX + 1][N];                         /* surviving states  */
    for (int q = -Q_MAX; q <= Q_MAX; q++) {             /* verdict per q     */
        int l = q + Q_MAX;                              /* its lane          */
        alive[l] = lockA[l] && (winding(ba, l) == q);   /* locked AND label  */
        if (alive[l])                                   /* keep its phases   */
            for (int i = 0; i < N; i++) att[l][i] = ba->th[i * ba->L + l];
    }
    batch_free(ba);                                     /* phase A done      */

    /* Phase B — every (q, site, amp) trial as one lane of one batch.        */
    int trials = 0;                                     /* live lane count   */
    int key_of[LMAX];                                   /* lane -> table idx */
    batch_t *bb = batch_new(TBL_SZ);                    /* worst-case width  */
    for (int q = -Q_MAX; q <= Q_MAX; q++) {             /* each start state  */
        if (!alive[q + Q_MAX]) continue;                /* dead: no trials   */
        for (int s = 0; s < N; s++)                     /* each kick site    */
            for (int a = 0; a < 2; a++) {               /* each amplitude    */
                int l = trials++;                       /* claim a lane      */
                for (int i = 0; i < N; i++) {           /* load attractor    */
                    bb->th[i * bb->L + l] = att[q + Q_MAX][i];
                    bb->om[i * bb->L + l] = 0.0;        /* pristine row      */
                }
                bb->om[lens_site * bb->L + l] = lens_delta; /* same lens     */
                apply_kick(bb, l, s, AMPS[a]);          /* the input symbol  */
                key_of[l] = TBL_IDX(q, s, a);           /* remember address  */
            }
    }
    bb->L = trials;                                     /* shrink to live set:
                                                           SoA stride is L, so
                                                           this must happen
                                                           BEFORE any step — 
                                                           loads above used
                                                           full-width stride,
                                                           so re-pack:       */
    /* Re-pack from allocation stride TBL_SZ to live stride `trials`.
     * ASCENDING row order is the safe direction when compacting to a
     * smaller stride: dst of row i ends at (i+1)*trials, and every
     * not-yet-copied src row k>i starts at k*TBL_SZ ≥ (i+1)*TBL_SZ ≥
     * (i+1)*trials — dst never overruns pending src. (The first build of
     * this file copied descending and corrupted every shrunk table; the
     * selftest caught it, which is what the selftest is for.)              */
    if (trials != TBL_SZ) {                             /* only if shrunk    */
        for (int i = 0; i < N; i++)                     /* rows front-to-back*/
            for (int l = 0; l < trials; l++) {          /* is safe: dst row  */
                bb->th[i * trials + l] = bb->th[i * TBL_SZ + l]; /* offset ≤ */
                bb->om[i * trials + l] = bb->om[i * TBL_SZ + l]; /* src row  */
            }                                           /* offset for i≥0    */
    }
    for (int j = 0; j < TBL_SZ; j++) table[j] = DEAD;   /* default: dead     */
    int lockB[LMAX]; settle(bb, lockB);                 /* THE big settle    */
    for (int l = 0; l < trials; l++)                    /* read every lane   */
        table[key_of[l]] = lockB[l] ? winding(bb, l) : NOLCK;
    batch_free(bb);                                     /* table complete    */
}
/* [Block rationale] The re-pack after lane-count shrink exists because SoA
 * stride is baked into every index; loading at full width then re-packing
 * keeps the load loop simple and pays one O(N·L) copy.
 * Alternative was counting live trials first — two passes over the same
 * logic; the re-pack is less code and self-contained.                       */

/* ---------------------------------------------------------------------------
 * EXPERIMENTS
 * ------------------------------------------------------------------------- */
static int ring_dist(int a, int b2){
    int d = abs(a - b2) & NMASK;                        /* forward distance  */
    return d < N - d ? d : N - d;                       /* shortest way round*/
}

/* Locality comparison of two tables; prints per-distance rows, returns
 * overall changed count via out-params for the caller's verdicting.         */
static void hamming_report(const int *t0, const int *t1, int lens_site,
                           FILE *csv, double delta,
                           int *o_changed, int *o_total){
    int chg[N/2 + 1] = {0}, tot[N/2 + 1] = {0};         /* per-distance bins */
    for (int q = -Q_MAX; q <= Q_MAX; q++)               /* every table cell  */
        for (int s = 0; s < N; s++)
            for (int a = 0; a < 2; a++) {
                int j = TBL_IDX(q, s, a);               /* cell address      */
                if (t0[j] <= DEAD+1 || t1[j] <= DEAD+1) continue; /* dead in
                                                           either table: not
                                                           comparable        */
                int d = ring_dist(s, lens_site);        /* locality coord    */
                tot[d]++;                               /* comparable cell   */
                chg[d] += (t0[j] != t1[j]);             /* changed?          */
            }
    int C = 0, T = 0;                                   /* global tallies    */
    for (int d = 0; d <= N/2; d++) {                    /* emit distance rows*/
        if (!tot[d]) continue;                          /* empty bin         */
        C += chg[d]; T += tot[d];                       /* accumulate        */
        if (csv) fprintf(csv, "%.4f,%d,%d,%d\n", delta, d, chg[d], tot[d]);
        else     printf("  dist %2d: %3d/%-3d  %.3f\n",
                        d, chg[d], tot[d], (double)chg[d]/tot[d]);
    }
    *o_changed = C; *o_total = T;                       /* hand back totals  */
}

/* selftest — the acceptance gate against the Python instruments.            */
static int cmd_selftest(void){
    int fails = 0;                                      /* strike counter    */
    /* 1. Twisted-state energies vs the analytic form (and the Python log). */
    printf("energies:\n");
    static const double VREF[4] = {-16.0, -14.782073, -11.313708, -6.122935};
    for (int q = 0; q <= Q_MAX; q++) {                  /* symmetric in ±q   */
        double v = 0.0;                                 /* potential sum     */
        for (int i = 0; i < N; i++)                     /* −K Σ cos(Δθ)      */
            v -= KCPL * cos(2.0*M_PI*q*((i+1) - i)/N);  /* uniform gradient  */
        printf("  V(%d) = %10.6f  (ref %10.6f)\n", q, v, VREF[q]);
        fails += fabs(v - VREF[q]) > 1e-5;              /* must match        */
    }
    /* 2. Full Δ=0.35 locality reproduction (bqsm_torus.py published run).  */
    int t0[TBL_SZ], t1[TBL_SZ], a0[7], a1[7];           /* tables + survival */
    clock_t w0 = clock();                               /* time the furnace  */
    build_table(0, 0.00, t0, a0);                       /* pristine machine  */
    build_table(0, 0.35, t1, a1);                       /* lensed machine    */
    double secs = (double)(clock() - w0) / CLOCKS_PER_SEC;
    printf("pre-lens alive : ");                        /* alphabet, pre     */
    for (int q = -Q_MAX; q <= Q_MAX; q++) if (a0[q+Q_MAX]) printf("%d ", q);
    printf("\npost-lens alive: ");                      /* alphabet, post    */
    for (int q = -Q_MAX; q <= Q_MAX; q++) if (a1[q+Q_MAX]) printf("%d ", q);
    int alive_ok = !a1[0] && !a1[6] && a1[1] && a1[5];  /* ±3 die, ±2 live   */
    printf("  (%s)\n", alive_ok ? "matches Python" : "MISMATCH");
    fails += !alive_ok;                                 /* alphabet gate     */
    int C, T;                                           /* locality tallies  */
    hamming_report(t0, t1, 0, NULL, 0.35, &C, &T);      /* print the profile */
    printf("overall: %d/%d = %.3f   (Python published: 6/160 = 0.037)\n",
           C, T, (double)C/T);
    fails += !(C == 6 && T == 160);                     /* exact reproduction*/
    printf("two full tables in %.2f s (Python session: ~minutes)\n", secs);
    printf(fails ? "selftest: FAIL (%d)\n" : "selftest: ALL PASS\n", fails);
    return fails ? 1 : 0;                               /* exit code = truth */
}

/* lens-sweep — the experiment the Python session could not afford.          */
static int cmd_lens_sweep(double lo, double hi, int steps, const char *path){
    FILE *csv = fopen(path, "w");                       /* fresh results file*/
    if (!csv) { perror(path); return 1; }               /* must be writable  */
    fprintf(csv, "delta,dist,changed,total\n");         /* schema header     */
    int t0[TBL_SZ], a0[7];                              /* pristine reference*/
    build_table(0, 0.0, t0, a0);                        /* built once        */
    printf("%8s %10s %8s %8s  alive\n", "delta", "hamming", "near", "far");
    for (int k = 0; k < steps; k++) {                   /* the Δ grid        */
        double delta = lo + (hi - lo) * k / (steps - 1);/* grid point        */
        int t1[TBL_SZ], a1[7];                          /* lensed machine    */
        build_table(0, delta, t1, a1);                  /* one batched build */
        int C, T; hamming_report(t0, t1, 0, csv, delta, &C, &T);
        int nearC=0, nearT=0, farC=0, farT=0;           /* near/far split    */
        for (int q = -Q_MAX; q <= Q_MAX; q++)           /* re-walk for split */
            for (int s = 0; s < N; s++)
                for (int a = 0; a < 2; a++) {
                    int j = TBL_IDX(q, s, a);           /* cell address      */
                    if (t0[j] <= DEAD+1 || t1[j] <= DEAD+1) continue;
                    int d = ring_dist(s, 0);            /* distance to lens  */
                    if (d <= 2) { nearT++; nearC += t0[j] != t1[j]; }
                    if (d >= 5) { farT++;  farC  += t0[j] != t1[j]; }
                }
        int nalive = 0;                                 /* survivor count    */
        for (int i = 0; i < 7; i++) nalive += a1[i];    /* tally             */
        printf("%8.3f %6d/%-3d %5.3f    %5.3f    %d/7\n", delta, C, T,
               nearT ? (double)nearC/nearT : 0.0,
               farT  ? (double)farC/farT  : 0.0, nalive);
    }
    fclose(csv);                                        /* results durable   */
    printf("written: %s\n", path);                      /* hand-off line     */
    return 0;
}

int main(int argc, char **argv){
    if ((N & NMASK) != 0 || N < 8)                      /* mask trick guard  */
        { fprintf(stderr, "N must be a power of two ≥ 8\n"); return 2; }
    if (argc >= 2 && !strcmp(argv[1], "selftest"))      /* acceptance gate   */
        return cmd_selftest();
    if (argc == 6 && !strcmp(argv[1], "lens-sweep"))    /* the Δ furnace     */
        return cmd_lens_sweep(atof(argv[2]), atof(argv[3]),
                              atoi(argv[4]), argv[5]);
    fprintf(stderr, "usage: ring_furnace selftest\n"
                    "       ring_furnace lens-sweep <d_lo> <d_hi> <steps> <out.csv>\n");
    return 2;                                           /* bad invocation    */
}
/* [Block rationale] Two subcommands only, on purpose: selftest is the
 * trust anchor (this engine and the Python instruments are provably the
 * same instrument or the build fails), lens-sweep is the first paying
 * customer. Clock-scaling across N and the mobility batteries port onto
 * this same batch core next — they are new experiment functions, not new
 * infrastructure. Threading is deliberately absent: one process per grid
 * shard (GNU parallel / one shard per fleet core) beats in-process threads
 * for embarrassingly parallel science — simpler, checkpointable, and the
 * shards merge as CSVs, per the fleet doctrine.                             */

/* ===========================================================================
 * RESULTS LOG — first execution, 2026-07-20 (chat container, gcc -O3, x86_64)
 * ===========================================================================
 * ACCEPTANCE: selftest ALL PASS — twisted-state energies match analytic to
 *   1e-5, and the Δ=0.35 locality experiment reproduces bqsm_torus.py's
 *   published run EXACTLY (6/160 overall, 0.200/0.200 at dist 0/1, zero at
 *   dist ≥2, post-lens alphabet {−2..+2}). Two full 224-trial tables in
 *   3.1 s single-core; the Python session needed minutes per pair. The C
 *   engine and the Python instruments are the same instrument.
 * BUG CAUGHT BY THE SELFTEST (kept here as the cautionary record): the
 *   first build re-packed shrunk batches in DESCENDING row order, which
 *   overruns not-yet-copied source rows when compacting to a smaller
 *   stride; every lensed table was corrupted (144/160 spurious changes).
 *   Ascending order is the safe direction — proof in the code comment.
 * LENS SWEEP (Δ = 0.05 → 0.70, 14 points, ~40 s total):
 *   Δ ≤ 0.10          : near-silent (0–6 changed entries, near-field only).
 *   Δ = 0.15          : FAR-FIELD VIOLATION (far 0.041) while all 7 states
 *                       alive. Debug build attribution: 100% of far changes
 *                       are in the marginal |q|=3 rows, weak amplitude only,
 *                       all of the form "erase ±3→±2 no longer fires" —
 *                       i.e., the marginal states' basin boundaries moved
 *                       GLOBALLY under the strictly local edit.
 *   Δ = 0.20 – 0.65   : |q|=3 dead; CLEAN local regime — far exactly 0.000
 *                       across 10 grid points, near-field rising
 *                       monotonically 0.04 → 0.32 with lens strength.
 *   Δ = 0.70          : far-field reappears (0.029) — next breakdown
 *                       approaching (presumably ±2 going marginal).
 * FINDING: locality is a property of ROBUST states. Marginal states are
 *   the fabric's nonlocality channels: their globally soft basin
 *   boundaries let a local edit reach arbitrarily distant table entries,
 *   and the violation window opens exactly while a state family is
 *   alive-but-dying. Composability design rule for BQSM: operate well
 *   inside stability margins; the earlier torus finding (lenses can
 *   delete states) and this one are two faces of the same margin.
 * NOT YET ESTABLISHED: per-Δ resolution inside 0.10–0.20 (the violation
 *   window's exact edges); whether the Δ=0.70 far-field is ±2-mediated
 *   (same debug method applies); N-scaling of all of the above. All are
 *   grid shards for the fleet: one process per core, CSVs merge.
 * ======================================================================= */
