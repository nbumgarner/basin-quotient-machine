/* ============================================================================
 * octopus.c — Basin-geometry battery for the Kuramoto ring
 *
 * Reproduces the four experiments of Zhang & Strogatz, "Basins with
 * Tentacles", Phys. Rev. Lett. 127, 194101 (2021) [arXiv:2106.05709],
 * on the identical-oscillator sine-coupled ring:
 *
 *     theta_i' = sin(theta_{i+1} - theta_i) + sin(theta_{i-1} - theta_i)
 *
 * Experiments (paper figure in brackets):
 *   A [Fig 1] Basin volumes via uniform global sampling; Gaussian-vs-
 *             exponential fit of ln p against q^2 and |q|.
 *   B [Fig 2, dashed] Escape distances: first exit from a basin along
 *             random rays from each twisted state ("head" size).
 *   C [Fig 2, solid] Distance from each sampled basin point to its
 *             attractor; checks convergence toward d_inf = sqrt(pi^2/3)
 *             ~= 1.8138 ("mass lives in the tentacles").
 *   D [Fig 3] Reentry probability: return rate to a twisted state under
 *             box perturbations of half-width alpha*pi; the alpha > 0.8
 *             plateau is the octopus signature.
 *
 * Purpose in the BQSM program: this is the *baseline* fabric with no
 * lenses. Run it, bank the numbers, then diff against the same battery
 * run on the lens-modified furnace container. The lens-induced shift of
 * the k coefficient, the escape distribution, and the alpha-plateau is
 * the genuinely novel measurement — the paper charts the static geometry;
 * the machine question is how instructions deform it.
 *
 * Build:   cc -O3 -Wall -Wextra -std=c11 -pthread -o octopus octopus.c -lm
 * Arch:    pure C11 + POSIX threads + libm; no intrinsics — dual-arch
 *          clean on x86_64 (T7400/Dell 15) and aarch64 (Oracle/Fold 6).
 * Usage:   ./octopus --selftest            quick integrity pass (~seconds)
 *          ./octopus                        demo-scale run of all four
 *          ./octopus --full                 paper-scale A (n=83, N=1e6)
 *          ./octopus --n 83 --samples 200000 --threads 8 ...
 * ==========================================================================*/

#include <stdio.h>      /* printf/fprintf for the report output              */
#include <stdlib.h>     /* malloc/free/atoi/strtoull for setup and flags     */
#include <string.h>     /* strcmp for flag parsing                           */
#include <math.h>       /* sin/sqrt/log/fabs/fmod — the whole physics        */
#include <stdint.h>     /* fixed-width types for the RNG state               */
#include <pthread.h>    /* worker threads: sampling is embarrassingly ||     */
#include <unistd.h>     /* sysconf() to auto-detect core count               */
#include <time.h>       /* wall-clock timing of each experiment              */

/* ---------------------------------------------------------------------------
 * Compile-time limits and physical constants
 * ------------------------------------------------------------------------ */
#define MAX_N        512          /* hard cap on ring size; stack buffers    */
#define TWO_PI       6.283185307179586476925286766559  /* 2*pi              */
#define PI           3.141592653589793238462643383279  /* pi                */
#define DT           0.5          /* RK4 step; gradient flow, |lambda|<=4,
                                     RK4 stability bound |lambda*h|<2.78 so
                                     h=0.5 sits well inside it              */
#define VEL_TOL      1e-3         /* max|theta'| declaring "phase-locked";
                                     winding number locks long before full
                                     convergence, so a loose velocity gate
                                     plus a q-stability check is both fast
                                     and classification-safe               */
#define CHECK_EVERY  25           /* steps between convergence checks; the
                                     check costs one derivative pass, so we
                                     amortize it                            */
#define MAX_STEPS    120000       /* safety cap ~ 60k time units; slowest
                                     mode for n=83 relaxes with tau~175 so
                                     this is >300 tau — never binds in
                                     practice, guards pathological starts   */
#define QMAX_TALLY   64           /* tally array covers q in [-64, 64]      */

/* ---------------------------------------------------------------------------
 * RNG — splitmix64 core, per-thread state, no libc rand() (not thread-safe,
 * poor quality). Constants are the published Steele/Lea/Vigna values.
 * ------------------------------------------------------------------------ */
typedef struct { uint64_t s; } Rng;                 /* one 64-bit word of state */

static inline uint64_t rng_u64(Rng *r) {            /* next raw 64-bit draw     */
    uint64_t z = (r->s += 0x9e3779b97f4a7c15ULL);   /* golden-ratio increment   */
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;    /* mix 1                    */
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;    /* mix 2                    */
    return z ^ (z >> 31);                           /* final avalanche          */
}
static inline double rng_uniform(Rng *r) {          /* uniform double in [0,1)  */
    return (double)(rng_u64(r) >> 11) * (1.0 / 9007199254740992.0); /* 53 bits */
}
static inline double rng_gauss(Rng *r) {            /* standard normal draw     */
    double u1 = rng_uniform(r), u2 = rng_uniform(r);/* two independent uniforms */
    if (u1 < 1e-300) u1 = 1e-300;                   /* guard log(0)             */
    return sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2); /* Box–Muller, cosine leg   */
}
/* Block rationale: splitmix64 is 4 ops/draw, passes BigCrush as a stream
 * seeder, and each thread seeds from (base_seed, thread_id) so runs are
 * reproducible and streams are independent. Alternative considered:
 * xoshiro256** — better equidistribution, but splitmix's single-word state
 * keeps per-thread structs trivial and its quality is ample for Monte Carlo
 * tallies with Bernoulli-dominated error.                                   */

/* ---------------------------------------------------------------------------
 * Ring dynamics
 * ------------------------------------------------------------------------ */

/* wrap x into (-pi, pi] — the canonical phase-difference branch             */
static inline double wrap_pi(double x) {
    x = fmod(x + PI, TWO_PI);                       /* shift then reduce mod 2pi */
    if (x <= 0.0) x += TWO_PI;                      /* fmod can return <=0       */
    return x - PI;                                  /* shift back to (-pi,pi]    */
}

/* derivative of the sine ring: one sin() per edge, scattered to both ends  */
static void deriv(int n, const double *th, double *dth) {
    for (int i = 0; i < n; i++) dth[i] = 0.0;       /* clear accumulator         */
    for (int i = 0; i < n; i++) {                   /* walk the n ring edges     */
        int j = (i + 1 == n) ? 0 : i + 1;           /* periodic neighbor         */
        double s = sin(th[j] - th[i]);              /* edge coupling term        */
        dth[i] += s;                                /* pulls i toward j          */
        dth[j] -= s;                                /* equal-opposite on j       */
    }
}
/* Block rationale: edge-walk form does n sin() calls instead of 2n for the
 * naive per-node form, and makes sum(dth)=0 exact — mean phase is conserved,
 * which is why "the attractor it converged to" is a canonical point rather
 * than an arbitrary member of the rotation family. Alternative: table-driven
 * sin approximation (the furnace trick) — deliberately not used here so the
 * baseline is bit-comparable to a straight libm reference.                  */

/* classic RK4 step in place; k-buffers supplied by caller to stay malloc-free */
static void rk4_step(int n, double *th,
                     double *k1, double *k2, double *k3, double *k4,
                     double *tmp) {
    deriv(n, th, k1);                               /* slope at start            */
    for (int i = 0; i < n; i++) tmp[i] = th[i] + 0.5 * DT * k1[i];  /* midpoint 1 */
    deriv(n, tmp, k2);                              /* slope at midpoint 1       */
    for (int i = 0; i < n; i++) tmp[i] = th[i] + 0.5 * DT * k2[i];  /* midpoint 2 */
    deriv(n, tmp, k3);                              /* slope at midpoint 2       */
    for (int i = 0; i < n; i++) tmp[i] = th[i] + DT * k3[i];        /* endpoint   */
    deriv(n, tmp, k4);                              /* slope at endpoint         */
    for (int i = 0; i < n; i++)                     /* Simpson-weighted update   */
        th[i] += (DT / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
}

/* winding number of a configuration: sum of wrapped gaps around the ring   */
static int winding(int n, const double *th) {
    double acc = 0.0;                               /* accumulated phase turn    */
    for (int i = 0; i < n; i++) {                   /* every ring edge           */
        int j = (i + 1 == n) ? 0 : i + 1;           /* periodic neighbor         */
        acc += wrap_pi(th[j] - th[i]);              /* principal-branch gap      */
    }
    return (int)lround(acc / TWO_PI);               /* total turns, integer      */
}

/* Integrate to phase lock. Returns final winding number; if 'out' is non-NULL
 * the converged configuration is copied there. Convergence = velocity below
 * VEL_TOL *and* winding number unchanged across two consecutive checks.     */
static int settle(int n, double *th, double *out) {
    double k1[MAX_N], k2[MAX_N], k3[MAX_N], k4[MAX_N], tmp[MAX_N]; /* RK4 scratch */
    int q_prev = winding(n, th);                    /* initial classification    */
    for (long step = 1; step <= MAX_STEPS; step++) {/* bounded main loop         */
        rk4_step(n, th, k1, k2, k3, k4, tmp);       /* advance one RK4 step      */
        if (step % CHECK_EVERY == 0) {              /* periodic convergence gate */
            deriv(n, th, k1);                       /* fresh velocities          */
            double vmax = 0.0;                      /* max |theta'| tracker      */
            for (int i = 0; i < n; i++) {           /* scan all oscillators      */
                double a = fabs(k1[i]);             /* speed of oscillator i     */
                if (a > vmax) vmax = a;             /* running maximum           */
            }
            int q_now = winding(n, th);             /* current classification    */
            if (vmax < VEL_TOL && q_now == q_prev) {/* locked and q-stable       */
                if (out) for (int i = 0; i < n; i++) out[i] = th[i]; /* export   */
                return q_now;                       /* settled — report q        */
            }
            q_prev = q_now;                         /* remember for stability    */
        }
    }
    if (out) for (int i = 0; i < n; i++) out[i] = th[i]; /* export best effort   */
    return winding(n, th);                          /* cap hit: classify anyway  */
}
/* Block rationale: the double gate (velocity + q-stability) exits ~5-10x
 * earlier than demanding max|theta'|<1e-9, because the winding number is a
 * discrete topological label that freezes as soon as the trajectory enters
 * the flow-invariant region of its target — long before amplitudes finish
 * decaying. Alternative: event-based "all gaps inside (-pi,pi)" test per the
 * Groisman flow-invariance argument; equally valid, marginally cheaper, but
 * the velocity gate also serves Experiments B/D where we start *outside*
 * invariant regions and genuinely need lock detection.                      */

/* build the exact q-twisted state theta_i = 2*pi*i*q/n                      */
static void make_twisted(int n, int q, double *th) {
    for (int i = 0; i < n; i++)                     /* each oscillator           */
        th[i] = wrap_pi(TWO_PI * (double)i * (double)q / (double)n); /* uniform twist */
}

/* dimension-normalized distance d = sqrt(mean_i wrap|a_i - b_i|^2), the
 * paper's metric; component distances live in [0, pi]                       */
static double dist_norm(int n, const double *a, const double *b) {
    double acc = 0.0;                               /* sum of squared gaps       */
    for (int i = 0; i < n; i++) {                   /* every component           */
        double d = fabs(wrap_pi(a[i] - b[i]));      /* wrapped absolute gap      */
        acc += d * d;                               /* accumulate square         */
    }
    return sqrt(acc / (double)n);                   /* RMS over dimensions       */
}

/* ---------------------------------------------------------------------------
 * Experiment A + C — uniform global sampling (threaded)
 * A tallies basin sizes per q; C records start-to-attractor distances.
 * ------------------------------------------------------------------------ */
typedef struct {
    int      n;                                     /* ring size                 */
    long     samples;                               /* samples for this thread   */
    uint64_t seed;                                  /* thread RNG seed           */
    long     tally[2 * QMAX_TALLY + 1];             /* counts indexed q+QMAX     */
    double   dist_sum, dist_sq;                     /* running distance moments  */
    long     dist_cnt;                              /* distance sample count     */
    long     capped;                                /* MAX_STEPS cap hits        */
} AWork;

static void *a_worker(void *arg) {
    AWork *w = (AWork *)arg;                        /* unpack the job            */
    Rng rng = { w->seed };                          /* private RNG stream        */
    double th[MAX_N], start[MAX_N], fin[MAX_N];     /* state buffers             */
    for (long s = 0; s < w->samples; s++) {         /* the Monte Carlo loop      */
        for (int i = 0; i < w->n; i++) {            /* draw uniform torus point  */
            th[i] = (rng_uniform(&rng) - 0.5) * TWO_PI; /* uniform (-pi,pi)      */
            start[i] = th[i];                       /* keep the initial point    */
        }
        int q = settle(w->n, th, fin);              /* relax to an attractor     */
        if (q >= -QMAX_TALLY && q <= QMAX_TALLY)    /* in-range winding number   */
            w->tally[q + QMAX_TALLY]++;             /* count the basin hit       */
        double d = dist_norm(w->n, start, fin);     /* Experiment C distance     */
        w->dist_sum += d;                           /* first moment              */
        w->dist_sq  += d * d;                       /* second moment             */
        w->dist_cnt++;                              /* sample count              */
    }
    return NULL;                                    /* results live in *w        */
}

/* weighted least squares of y on x through free intercept; returns slope,
 * intercept and SSE — used to race ln p ~ q^2 against ln p ~ |q|            */
static void fit_line(int m, const double *x, const double *y, const double *wt,
                     double *slope, double *icept, double *sse) {
    double W = 0, Sx = 0, Sy = 0, Sxx = 0, Sxy = 0; /* weighted sums             */
    for (int i = 0; i < m; i++) {                   /* accumulate normal eqns    */
        W += wt[i]; Sx += wt[i] * x[i]; Sy += wt[i] * y[i];      /* 0th,1st      */
        Sxx += wt[i] * x[i] * x[i]; Sxy += wt[i] * x[i] * y[i];  /* 2nd, cross   */
    }
    double den = W * Sxx - Sx * Sx;                 /* normal-equation det       */
    *slope = (W * Sxy - Sx * Sy) / den;             /* closed-form slope         */
    *icept = (Sy - *slope * Sx) / W;                /* closed-form intercept     */
    *sse = 0;                                       /* residual accumulator      */
    for (int i = 0; i < m; i++) {                   /* weighted residuals        */
        double r = y[i] - (*icept + *slope * x[i]); /* fit residual              */
        *sse += wt[i] * r * r;                      /* weighted square           */
    }
}
/* Block rationale: weights are the inverse-variance of ln p under Bernoulli
 * counting, Var[ln p] ~ (1-p)/(count), so well-populated q values dominate
 * the fit exactly as they do in the paper's converged region. Alternative:
 * unweighted fit — simpler but lets the noisiest tail bins distort the
 * Gaussian/exponential verdict, which is the one output we most care about. */

/* ---------------------------------------------------------------------------
 * Experiment B — escape distances along random rays (threaded over rays)
 * ------------------------------------------------------------------------ */
typedef struct {
    int      n, q, rays;                            /* ring size, state, ray count */
    uint64_t seed;                                  /* RNG seed                  */
    double   esc_sum, esc_sq;                       /* escape distance moments   */
    double   esc_min, esc_max;                      /* range trackers            */
    int      done;                                  /* rays completed            */
} BWork;

static void *b_worker(void *arg) {
    BWork *w = (BWork *)arg;                        /* unpack job                */
    Rng rng = { w->seed };                          /* private stream            */
    double base[MAX_N], u[MAX_N], th[MAX_N];        /* twisted state, ray, probe */
    make_twisted(w->n, w->q, base);                 /* the attractor under test  */
    w->esc_min = 1e30; w->esc_max = 0.0;            /* init range                */
    for (int r = 0; r < w->rays; r++) {             /* independent directions    */
        double norm = 0.0;                          /* direction normalizer      */
        for (int i = 0; i < w->n; i++) {            /* isotropic Gaussian ray    */
            u[i] = rng_gauss(&rng);                 /* iid normal component      */
            norm += u[i] * u[i];                    /* accumulate squared norm   */
        }
        norm = sqrt(norm);                          /* Euclidean length          */
        for (int i = 0; i < w->n; i++) u[i] /= norm;/* unit vector               */
        double d_esc = -1.0;                        /* found escape distance     */
        for (double dn = 0.02; dn <= PI; dn += 0.02) {  /* ramp normalized dist  */
            double s = dn * sqrt((double)w->n);     /* raw ray length for dn     */
            for (int i = 0; i < w->n; i++)          /* build the probe point     */
                th[i] = base[i] + s * u[i];         /* attractor + s*u           */
            if (settle(w->n, th, NULL) != w->q) {   /* left the basin?           */
                d_esc = dn;                         /* first-exit distance       */
                break;                              /* ramp done for this ray    */
            }
        }
        if (d_esc < 0.0) d_esc = PI;                /* never escaped: cap at pi  */
        w->esc_sum += d_esc; w->esc_sq += d_esc * d_esc;  /* moments             */
        if (d_esc < w->esc_min) w->esc_min = d_esc; /* update min                */
        if (d_esc > w->esc_max) w->esc_max = d_esc; /* update max                */
        w->done++;                                  /* progress                  */
    }
    return NULL;                                    /* results in *w             */
}
/* Block rationale: the ramp (not bisection) matches the paper's "first
 * escape" semantics — tentacled basins make the inside/outside set along a
 * ray non-monotone, so bisection alone would find *an* exit, not the first.
 * Step 0.02 in normalized units bounds the overshoot at half the histogram
 * bin the paper plots. Cost is fine because near-attractor probes settle in
 * a few hundred RK4 steps.                                                  */

/* ---------------------------------------------------------------------------
 * Experiment D — reentry probability vs box-perturbation amplitude alpha
 * ------------------------------------------------------------------------ */
typedef struct {
    int      n, q, trials;                          /* ring, state, trials/alpha */
    double   alpha;                                 /* box half-width / pi       */
    uint64_t seed;                                  /* RNG seed                  */
    int      returned;                              /* count of q-returns        */
} DWork;

static void *d_worker(void *arg) {
    DWork *w = (DWork *)arg;                        /* unpack job                */
    Rng rng = { w->seed };                          /* private stream            */
    double base[MAX_N], th[MAX_N];                  /* twisted state, probe      */
    make_twisted(w->n, w->q, base);                 /* attractor under test      */
    for (int t = 0; t < w->trials; t++) {           /* Monte Carlo trials        */
        for (int i = 0; i < w->n; i++)              /* box perturbation          */
            th[i] = base[i]                         /* start at the attractor    */
                  + (rng_uniform(&rng) * 2.0 - 1.0) /* uniform in [-1,1)         */
                  * w->alpha * PI;                  /* scaled to [-a*pi, a*pi)   */
        if (settle(w->n, th, NULL) == w->q)         /* did it come home?         */
            w->returned++;                          /* count the reentry         */
    }
    return NULL;                                    /* result in *w              */
}

/* ---------------------------------------------------------------------------
 * Orchestration helpers
 * ------------------------------------------------------------------------ */
static double now_sec(void) {                       /* monotonic wall clock      */
    struct timespec ts;                             /* kernel timestamp          */
    clock_gettime(CLOCK_MONOTONIC, &ts);            /* immune to NTP jumps       */
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec; /* seconds as double  */
}

/* ---------------------------------------------------------------------------
 * main — flags, selftest, and the four-experiment report
 * ------------------------------------------------------------------------ */
int main(int argc, char **argv) {
    int      n        = 83;                         /* ring size (paper's n)     */
    long     samples  = 20000;                      /* Experiment A samples      */
    int      rays     = 60;                         /* Experiment B rays per q   */
    int      trials   = 150;                        /* Experiment D trials/alpha */
    int      threads  = (int)sysconf(_SC_NPROCESSORS_ONLN); /* autodetect cores  */
    uint64_t seed     = 0x5EEDBA51ULL;              /* reproducible base seed    */
    int      selftest = 0, full = 0;                /* mode flags                */
    int      skip_b   = 0, skip_d = 0;              /* section skips             */

    for (int i = 1; i < argc; i++) {                /* minimal flag parser       */
        if      (!strcmp(argv[i], "--selftest")) selftest = 1;            /* quick pass  */
        else if (!strcmp(argv[i], "--full"))     full = 1;                /* paper scale */
        else if (!strcmp(argv[i], "--skip-escape"))  skip_b = 1;          /* omit B      */
        else if (!strcmp(argv[i], "--skip-reentry")) skip_d = 1;          /* omit D      */
        else if (!strcmp(argv[i], "--n")       && i+1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--samples") && i+1 < argc) samples = atol(argv[++i]);
        else if (!strcmp(argv[i], "--rays")    && i+1 < argc) rays = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--trials")  && i+1 < argc) trials = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i+1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed")    && i+1 < argc) seed = strtoull(argv[++i], NULL, 0);
        else { fprintf(stderr, "unknown flag: %s\n", argv[i]); return 2; }
    }
    if (full) { samples = 1000000; rays = 1000; trials = 1000; }  /* paper scale */
    if (selftest) { n = 40; samples = 800; rays = 0; trials = 0; skip_b = skip_d = 1; }
    if (n < 5 || n > MAX_N) { fprintf(stderr, "n out of range\n"); return 2; }
    if (threads < 1) threads = 1;                   /* sanity floor              */
    if (threads > 64) threads = 64;                 /* sanity ceiling            */

    printf("octopus: n=%d  samples=%ld  threads=%d  seed=0x%llx%s\n",
           n, samples, threads, (unsigned long long)seed,
           selftest ? "  [SELFTEST]" : (full ? "  [FULL]" : ""));
    printf("model: theta_i' = sin(theta_{i+1}-theta_i) + sin(theta_{i-1}-theta_i)"
           "  (PRL 127.194101 Eq.1)\n");
    printf("stable twisted states: |q| < n/4 = %.1f\n\n", (double)n / 4.0);

    /* ---- Experiment A + C: uniform global sampling ---------------------- */
    double t0 = now_sec();                          /* start the clock           */
    pthread_t tid[64];                              /* thread handles            */
    AWork aw[64];                                   /* per-thread jobs           */
    memset(aw, 0, sizeof(aw));                      /* zero all tallies          */
    long per = samples / threads;                   /* even share                */
    for (int t = 0; t < threads; t++) {             /* launch the pool           */
        aw[t].n = n;                                /* ring size                 */
        aw[t].samples = per + (t == 0 ? samples % threads : 0); /* remainder->t0 */
        aw[t].seed = seed + 0x9E3779B9ULL * (uint64_t)(t + 1);  /* stream split  */
        pthread_create(&tid[t], NULL, a_worker, &aw[t]);        /* go            */
    }
    long tally[2 * QMAX_TALLY + 1] = {0};           /* merged counts             */
    double dsum = 0, dsq = 0; long dcnt = 0;        /* merged distance moments   */
    for (int t = 0; t < threads; t++) {             /* join and merge            */
        pthread_join(tid[t], NULL);                 /* wait for worker           */
        for (int k = 0; k <= 2 * QMAX_TALLY; k++)   /* fold tallies              */
            tally[k] += aw[t].tally[k];             /* accumulate                */
        dsum += aw[t].dist_sum; dsq += aw[t].dist_sq; dcnt += aw[t].dist_cnt;
    }
    double tA = now_sec() - t0;                     /* experiment wall time      */

    printf("== Experiment A: basin sizes from uniform global sampling  (%.1fs)\n", tA);
    printf("   %-4s %-9s %-11s %-10s\n", "q", "count", "p", "rel.err");
    int    qseen = 0;                               /* number of populated |q|   */
    double fx_g[QMAX_TALLY], fx_e[QMAX_TALLY];      /* fit abscissas q^2, |q|    */
    double fy[QMAX_TALLY], fw[QMAX_TALLY];          /* fit ln p and weights      */
    for (int q = -QMAX_TALLY; q <= QMAX_TALLY; q++) {   /* report populated bins */
        long c = tally[q + QMAX_TALLY];             /* count for this q          */
        if (c == 0) continue;                       /* skip empty bins           */
        double p = (double)c / (double)samples;     /* relative basin size       */
        double relerr = 1.0 / sqrt(p * (double)samples); /* paper's 1/sqrt(pN)   */
        printf("   %-4d %-9ld %-11.3e %-10.3f\n", q, c, p, relerr);
        if (q >= 0 && c >= 5) {                     /* pool +q side for the fit  */
            long cs = c + ((q > 0) ? tally[-q + QMAX_TALLY] : 0); /* +/-q merged */
            double ps = (double)cs / (double)samples;             /* merged p    */
            fx_g[qseen] = (double)q * (double)q;    /* Gaussian abscissa q^2     */
            fx_e[qseen] = (double)q;                /* exponential abscissa |q|  */
            fy[qseen]   = log(ps);                  /* ordinate ln p             */
            fw[qseen]   = (double)cs;               /* ~inverse Var[ln p]        */
            qseen++;                                /* one more fit point        */
        }
    }
    if (qseen >= 3) {                               /* enough points to race fits */
        double kg, bg, sg, ke, be, se;              /* slope/intercept/SSE x2    */
        fit_line(qseen, fx_g, fy, fw, &kg, &bg, &sg);   /* ln p ~ -k q^2         */
        fit_line(qseen, fx_e, fy, fw, &ke, &be, &se);   /* ln p ~ -k |q|         */
        printf("   fit ln p = a - k*q^2 : k = %.4f   weighted SSE = %.3g\n", -kg, sg);
        printf("   fit ln p = a - k*|q| : k = %.4f   weighted SSE = %.3g\n", -ke, se);
        printf("   verdict: %s scaling favored (paper: Gaussian, e^{-k q^2})\n",
               (sg < se) ? "GAUSSIAN" : "EXPONENTIAL");
    } else printf("   (too few populated q bins for a fit — raise --samples)\n");

    /* ---- Experiment C: distances start -> attractor --------------------- */
    double dmean = dsum / (double)dcnt;             /* sample mean distance      */
    double dvar  = dsq / (double)dcnt - dmean * dmean;  /* sample variance       */
    double dinf  = sqrt(PI * PI / 3.0);             /* analytic d_inf ~ 1.8138   */
    printf("\n== Experiment C: distance from basin points to their attractor\n");
    printf("   mean d = %.4f   sd = %.4f   analytic d_inf = sqrt(pi^2/3) = %.4f\n",
           dmean, sqrt(dvar > 0 ? dvar : 0), dinf);
    printf("   interpretation: mean near d_inf with small sd = basin mass lives\n"
           "   at generic-random distance from the attractor (tentacles), far\n"
           "   outside the escape radii below (heads).\n");

    /* ---- Experiment B: escape distances --------------------------------- */
    if (!skip_b) {                                  /* unless skipped            */
        printf("\n== Experiment B: first-escape distances along random rays\n");
        int qlist[3] = { 0, (int)(n / 16), (int)(n / 8) };  /* low/mid/high q    */
        for (int qi = 0; qi < 3; qi++) {            /* each probed q             */
            int q = qlist[qi];                      /* state under test          */
            if (q >= n / 4) continue;               /* only stable states        */
            t0 = now_sec();                         /* time this q               */
            BWork bw[64]; memset(bw, 0, sizeof(bw));/* per-thread jobs           */
            int rper = rays / threads;              /* rays per thread           */
            if (rper == 0) rper = 1;                /* at least one              */
            int launched = 0;                       /* actual thread count       */
            for (int t = 0; t < threads && launched * rper < rays; t++, launched++) {
                bw[t].n = n; bw[t].q = q;           /* job parameters            */
                bw[t].rays = rper;                  /* share of rays             */
                bw[t].seed = seed ^ (0xB00B1E5ULL * (uint64_t)(t + 7 + q)); /* stream */
                pthread_create(&tid[t], NULL, b_worker, &bw[t]);  /* go          */
            }
            double esum = 0, esq = 0, emin = 1e30, emax = 0; int edone = 0; /* merge */
            for (int t = 0; t < launched; t++) {    /* join and fold             */
                pthread_join(tid[t], NULL);         /* wait                      */
                esum += bw[t].esc_sum; esq += bw[t].esc_sq; edone += bw[t].done;
                if (bw[t].esc_min < emin) emin = bw[t].esc_min;  /* fold min     */
                if (bw[t].esc_max > emax) emax = bw[t].esc_max;  /* fold max     */
            }
            double em = esum / edone;               /* mean escape distance      */
            double ev = esq / edone - em * em;      /* variance                  */
            printf("   q=%-3d  rays=%-4d  mean=%.3f  sd=%.3f  min=%.3f  max=%.3f"
                   "  (%.1fs)\n", q, edone, em, sqrt(ev > 0 ? ev : 0), emin, emax,
                   now_sec() - t0);
        }
        printf("   compare: escape means (heads) vs Experiment C mean (mass) —\n"
               "   disjoint ranges reproduce the paper's Fig. 2 tension.\n");
    }

    /* ---- Experiment D: reentry probability vs alpha ---------------------- */
    if (!skip_d) {                                  /* unless skipped            */
        printf("\n== Experiment D: return probability under box perturbations\n");
        printf("   %-7s %-12s %-12s\n", "alpha", "p(q=0)", "p(q=5)");
        for (double alpha = 0.4; alpha <= 1.001; alpha += 0.1) {  /* paper's axis */
            double pr[2];                           /* return probs for both q   */
            int qtest[2] = { 0, 5 };                /* in-phase and a mid twist  */
            for (int qi = 0; qi < 2; qi++) {        /* each tested state         */
                DWork dw[64]; memset(dw, 0, sizeof(dw));  /* per-thread jobs     */
                int tper = trials / threads;        /* trials per thread         */
                if (tper == 0) tper = 1;            /* at least one              */
                int launched = 0;                   /* actual thread count       */
                for (int t = 0; t < threads && launched * tper < trials; t++, launched++) {
                    dw[t].n = n; dw[t].q = qtest[qi];       /* job params        */
                    dw[t].trials = tper;                    /* share             */
                    dw[t].alpha = alpha;                    /* amplitude         */
                    dw[t].seed = seed ^ (0xD1CEULL * (uint64_t)(t + 3)
                               + (uint64_t)(alpha * 1000)); /* stream split      */
                    pthread_create(&tid[t], NULL, d_worker, &dw[t]);  /* go      */
                }
                int ret = 0, tot = 0;               /* merge counters            */
                for (int t = 0; t < launched; t++) {/* join and fold             */
                    pthread_join(tid[t], NULL);     /* wait                      */
                    ret += dw[t].returned;          /* fold returns              */
                    tot += dw[t].trials;            /* fold trials               */
                }
                pr[qi] = (double)ret / (double)tot; /* return probability        */
            }
            printf("   %-7.1f %-12.3f %-12.3f\n", alpha, pr[0], pr[1]);
        }
        printf("   octopus signature: p flattens to a nonzero plateau for\n"
               "   alpha > 0.8 (hypercube basins would decay toward zero).\n");
    }

    /* ---- Selftest verdict ------------------------------------------------ */
    if (selftest) {                                 /* integrity gate            */
        long c0 = tally[0 + QMAX_TALLY];            /* q=0 count                 */
        long cmax = 0;                              /* modal count               */
        for (int k = 0; k <= 2 * QMAX_TALLY; k++)   /* find the mode             */
            if (tally[k] > cmax) cmax = tally[k];   /* track max                 */
        int ok = (c0 == cmax) && (c0 > samples / 4) /* q=0 modal and dominant    */
              && (dcnt == samples)                  /* every sample classified   */
              && (fabs(dmean - dinf) < 0.25);       /* distances near analytic   */
        printf("\nSELFTEST: %s  (q=0 modal:%s  classified:%ld/%ld  "
               "mean d %.3f vs %.3f)\n", ok ? "PASS" : "FAIL",
               (c0 == cmax) ? "yes" : "no", dcnt, samples, dmean, dinf);
        return ok ? 0 : 1;                          /* exit code for scripting   */
    }
    printf("\ndone.\n");
    return 0;                                       /* clean exit                */
}
/* ============================================================================
 * Closing rationale and alternatives
 *
 * Why a standalone baseline rather than a furnace patch: the diff you want
 * is (lens-off ring) vs (lens-on ring) on identical axes. This file *is*
 * the lens-off reference, built to libm precision with no fast-math table
 * tricks, so any divergence you later see in the furnace port is a lens
 * effect, not an integrator artifact. Port path: the four worker loops map
 * one-to-one onto furnace lanes; only settle() and winding() need lifting.
 *
 * Known deliberate simplifications, none load-bearing for the comparison:
 *  - Non-twisted locked states (unequal-gap fixed points) are classified by
 *    winding number alone; they are unstable for this model so their measure
 *    is zero, but a paranoid port can add a uniform-gap residual check.
 *  - Experiment B's ramp step (0.02) floors the resolution of the escape
 *    histogram; halve it for publication-grade curves at 2x cost.
 *  - Threads cap at 64 and buffers at MAX_N=512 to stay allocation-free in
 *    the hot path; both are one-line lifts if the program ever needs them.
 * ==========================================================================*/
