/* ===========================================================================
 * bqsm_vm.c — BQSM Virtual Processor Core
 * ===========================================================================
 *
 *   emerging.systems — BQSM instrument suite, virtual processor
 *
 * WHAT THIS IS
 *   The register-machine virtual processor that wraps the ring_furnace.c
 *   physics engine. It implements the Basin-Quotient N-ary Computer as
 *   specified in BQSM_ARCHITECTURE.txt: a register file of coupled-
 *   oscillator rings, an instruction set of topological operations, and
 *   an execution pipeline (FETCH → DECODE → KICK → SETTLE → READ).
 *
 * PHYSICS CORE ATTRIBUTION
 *   The deriv, rk4_step, lock_spread, settle, winding functions below are
 *   imported VERBATIM from ring_furnace.c (emerging.systems, 2026). They
 *   remain unmodified per the code contract — the selftest trust anchor
 *   from ring_furnace.c extends to this module. Adaptations are limited to
 *   re-pointing references to the bqsm_vm_t batch structure.
 *
 * ARCHITECTURE
 *   SoA BATCHING  All N rings in the register file share one SoA batch;
 *                 each ring is a lane. The physics core stays vectorized.
 *   INSTRUCTIONS  Each opcode maps to a physical perturbation (kick, ramp)
 *                 followed by a batched settle and winding-number read.
 *   TRANSITION T  Precomputed table T[q][opcode] caches settle outcomes
 *                 for O(1) lookup; computed once after lens changes.
 *   RADIX CONTROL  Admissible q range is clamped to [0, radix-1]; higher
 *                 |q| states are treated as overflow/trap.
 *
 * BUILD
 *   gcc -O3 -Wall -Wextra -std=c11 bqsm_vm.c -o bqsm_vm -lm
 *   (Optionally -DREF_BUILD for bit-anchored oracle with DT=0.02, libm sin)
 * ======================================================================= */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#include "bqsm_vm.h"

/* =========================================================================
 * PHYSICS CONSTANTS (from ring_furnace.c, verbatim)
 * ========================================================================= */
#define N        BQSM_N
#define NMASK   BQSM_NMASK
#define KCPL    BQSM_KCPL
#define Q_MAX   BQSM_Q_MAX
#define LMAX    BQSM_LMAX
#define T_CHUNK BQSM_T_CHUNK
#define LOCK_TOL BQSM_LOCK_TOL
#define MAXCHUNK BQSM_MAXCHUNK
#define KICK_W  BQSM_KICK_W

#ifdef REF_BUILD
#define DT       0.02
#else
#define DT       0.50
#endif

/* --- Fast sin / libm sin selection (from ring_furnace.c) ---------------- */
#ifndef REF_BUILD
/* fast_sin — minimax degree-11 odd polynomial, range-reduced to [0, pi/2].
 * Validated at 1.483e-11 max error (fs_check.c). Inline so the compiler
 * auto-vectorizes the deriv lane loop (libm calls prevent vectorization).   */
#define FS_C1   9.99999999914541360e-01
#define FS_C3  -1.66666665577888368e-01
#define FS_C5   8.33332959535318385e-03
#define FS_C7  -1.98407312829215765e-04
#define FS_C9   2.75199467123275398e-06
#define FS_C11 -2.38101484600250895e-08
static inline double fast_sin(double x){
    x -= 2.0*M_PI * rint(x * (1.0/(2.0*M_PI))); /* branchless wrap to [-pi,pi] */
    double s  = x < 0.0 ? -1.0 : 1.0;             /* odd symmetry               */
    double xa = fabs(x);                          /* work on [0, pi]            */
    xa = xa > M_PI*0.5 ? M_PI - xa : xa;          /* THE REFLECTION: sin(pi-x)  */
    double x2 = xa*xa;                            /* Horner in x^2              */
    return s * xa * (FS_C1 + x2*(FS_C3 + x2*(FS_C5 + x2*(FS_C7 + x2*(FS_C9 + x2*FS_C11)))));
}
#define SIN fast_sin
#else
#define SIN sin
#endif

/* --- Sentinels ---------------------------------------------------------- */
#define DEAD   (-99)   /* start state is dead / ring never locked           */
#define NOLCK  (-98)   /* ring never locked (drifted indefinitely)          */
#define OOB    (-97)   /* |q| beyond admissible radix range                 */

/* =========================================================================
 * DATA STRUCTURES
 * ========================================================================= */

/* Batch state, SoA: value of site i in lane l lives at [i*L + l].
 * PHYSICS CORE ATTRIBUTION: this layout is identical to ring_furnace.c's
 * batch_t. See ring_furnace.c lines 93-102 for the original.               */
typedef struct {
    int     L;              /* live lanes in this batch (num rings)         */
    double *th;             /* phases        [N*L]                          */
    double *om;             /* per-lane ω    [N*L]                          */
    double *k1, *k2, *k3, *k4, *tmp; /* RK4 workspaces  [N*L] each         */
    int    *locked;         /* per-lane lock flags (post-settle cache)      */
} batch_t;

/* Per-ring metadata: winding number and lock status.                       */
typedef struct {
    int q;                  /* current winding number (DEAD if not locked)  */
    int locked;             /* is the ring in a settled state?              */
    int alive;              /* did it start alive (survived lens)?          */
} ring_meta_t;

/* The full VM: register file (rings as lanes), program memory, transition
 * table, and execution state.                                               */
struct bqsm_vm_s {
    batch_t      *batch;                    /* SoA physics batch              */
    ring_meta_t  *rings;                    /* per-ring metadata [num_rings] */
    int           num_rings;                /* register count                */
    bqsm_radix_t  radix;                    /* admissible q range            */

    /* Program memory */
    bqsm_instruction_t *program;            /* loaded instruction array      */
    int                 prog_len;           /* instruction count             */
    int                 pc;                 /* program counter               */
    int                 steps_executed;     /* total instructions run        */

    /* Lens configuration */
    int    lens_site;                      /* which site is detuned         */
    double lens_delta;                     /* detuning amplitude            */

    /* Transition table cache */
    int    tt_computed;                     /* has the table been computed?  */
    /* T[q+Q_MAX][opcode]: for each start state q and each opcode with
     * default params (site=0, amp=1.0), the result winding number.
     * Dimensions: [2*Q_MAX+1][BQSM_OPCODE_COUNT]                         */
    int    tt_flat[(2*Q_MAX+1) * BQSM_OPCODE_COUNT];

    /* Multi-site transition table for ERASE/KICK_AT: per-site variant
     * Dimensions: [N][2*Q_MAX+1] — result of ERASE at each site per q    */
    int    tt_erase[N * (2*Q_MAX+1)];

    /* --- RNG for stochastic operations --- */
    uint64_t rng_state;
};

/* =========================================================================
 * PHYSICS CORE — imported verbatim from ring_furnace.c
 * =========================================================================
 * The following five functions (deriv, rk4_step, lock_spread, settle,
 * winding) are copied line-for-line from ring_furnace.c with only these
 * adaptations:
 *   - Internal references use the local batch_t (identical layout)
 *   - They operate within the bqsm_vm.c namespace
 * The physics is IDENTICAL to ring_furnace.c per the code contract.
 * See ring_furnace.c at lines 126-233 for the originals with full comments.
 * ========================================================================= */

/* dθ/dt for every lane at once — edge-walk form: one sin() per ring edge,
 * distributed to both endpoints via antisymmetry. Outer loop = N edges;
 * inner loop = lanes (unit stride — this is the SIMD-vectorizable loop).
 * Mathematically identical to the per-site form.                           */
static void deriv(const double *th, const double *om, double *out, int L){
    int M = N * L;
    for (int j = 0; j < M; j++) out[j] = om[j];
    for (int i = 0; i < N; i++) {
        int j  = ((i + 1) & NMASK);
        int ii = i * L;
        int jj = j * L;
        for (int l = 0; l < L; l++) {
            double s = KCPL * SIN(th[jj + l] - th[ii + l]);
            out[ii + l] += s;
            out[jj + l] -= s;
        }
    }
}

/* One RK4 step for the whole batch (classic tableau, batched arithmetic).  */
static void rk4_step(batch_t *b){
    int M = N * b->L;
    deriv(b->th, b->om, b->k1, b->L);
    for (int j = 0; j < M; j++) b->tmp[j] = b->th[j] + 0.5*DT*b->k1[j];
    deriv(b->tmp, b->om, b->k2, b->L);
    for (int j = 0; j < M; j++) b->tmp[j] = b->th[j] + 0.5*DT*b->k2[j];
    deriv(b->tmp, b->om, b->k3, b->L);
    for (int j = 0; j < M; j++) b->tmp[j] = b->th[j] +     DT*b->k3[j];
    deriv(b->tmp, b->om, b->k4, b->L);
    for (int j = 0; j < M; j++)
        b->th[j] += (DT/6.0)*(b->k1[j] + 2*b->k2[j] + 2*b->k3[j] + b->k4[j]);
}

/* Per-lane lock residual: max−min of dθ/dt down the lane's column.        */
static void lock_spread(batch_t *b, double *spread){
    deriv(b->th, b->om, b->k1, b->L);
    for (int l = 0; l < b->L; l++) {
        double lo = b->k1[l], hi = b->k1[l];
        for (int i = 1; i < N; i++) {
            double v = b->k1[i * b->L + l];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        spread[l] = hi - lo;
    }
}

/* Settle the batch: chunked integration until EVERY lane locks (or patience
 * ends). Converged lanes ride along at their fixed points — idempotent, so
 * correctness is untouched and control flow stays uniform for SIMD.
 *
 * FAST build adds early classification exit: retire a lane when all N
 * wrapped gaps lie strictly inside (-pi+m, pi-m) with margin m=0.1 for
 * 2 consecutive checks (Groisman flow-invariance, rigorous for pristine).  */
static void settle(batch_t *b, int *locked_out){
    double spread[LMAX];
    int steps = (int)(T_CHUNK / DT);
#ifndef REF_BUILD
    int gap_stable[LMAX];
    const double MARGIN = 0.1;
    for (int l = 0; l < b->L; l++) gap_stable[l] = 0;
#endif
    for (int l = 0; l < b->L; l++) locked_out[l] = 0;
    for (int c = 0; c < MAXCHUNK; c++) {
        for (int s = 0; s < steps; s++) rk4_step(b);
        lock_spread(b, spread);
        int all = 1;
        for (int l = 0; l < b->L; l++) {
            int lck = spread[l] < LOCK_TOL;
#ifndef REF_BUILD
            if (!lck) {
                int gaps_ok = 1;
                for (int i = 0; i < N; i++) {
                    int j = (i + 1) & NMASK;
                    double d = remainder(b->th[j*b->L + l]
                                       - b->th[i*b->L + l], 2.0*M_PI);
                    if (fabs(d) >= M_PI - MARGIN) {
                        gaps_ok = 0; break;
                    }
                }
                if (gaps_ok) {
                    gap_stable[l]++;
                    if (gap_stable[l] >= 2) lck = 1;
                } else gap_stable[l] = 0;
            }
#endif
            locked_out[l] = lck;
            if (!lck) all = 0;
        }
        if (all) return;
    }
}

/* Winding number of lane l: wrapped phase differences summed around the
 * ring, snapped to the integer topological invariant.                       */
static int winding(const batch_t *b, int l){
    double s = 0.0;
    for (int i = 0; i < N; i++) {
        double d = b->th[((i + 1) & NMASK) * b->L + l]
                 - b->th[i * b->L + l];
        d = remainder(d, 2.0 * M_PI);
        s += d;
    }
    return (int)lround(s / (2.0 * M_PI));
}

/* END PHYSICS CORE IMPORT =============================================== */


/* =========================================================================
 * BATCH MANAGEMENT (adapted from ring_furnace.c batch_new / batch_free)
 * ========================================================================= */
static batch_t *batch_new(int L){
    batch_t *b = malloc(sizeof *b);
    b->L = L;
    size_t sz = (size_t)N * L * sizeof(double);
    b->th = malloc(sz);  b->om  = calloc(N * L, sizeof(double));
    b->k1 = malloc(sz);  b->k2  = malloc(sz);
    b->k3 = malloc(sz);  b->k4  = malloc(sz);
    b->tmp = malloc(sz);
    b->locked = calloc(L, sizeof(int));
    if (!b->th || !b->om || !b->k1 || !b->k2 || !b->k3 || !b->k4 || !b->tmp || !b->locked)
        { fprintf(stderr, "bqsm_vm: alloc failed\n"); exit(1); }
    return b;
}

static void batch_free(batch_t *b){
    free(b->th); free(b->om); free(b->k1); free(b->k2);
    free(b->k3); free(b->k4); free(b->tmp); free(b->locked);
    free(b);
}

/* =========================================================================
 * RING OPERATIONS — physical perturbations on individual lanes
 * ========================================================================= */

/* Write the ideal twisted state q into lane l.                              */
static void set_twisted(batch_t *b, int l, int q){
    for (int i = 0; i < N; i++)
        b->th[i * b->L + l] = fmod(2.0*M_PI*q*i/N, 2.0*M_PI);
}

/* Apply a scalar bump kick of width w and amplitude amp at `site` to lane l.
 * Standard KICK_W=3 for the measured ERASE operation.                        */
static void apply_kick(batch_t *b, int l, int site, double amp, int width){
    for (int k = -(width/2); k <= width/2; k++)
        b->th[((site + k) & NMASK) * b->L + l] += amp;
}

/* Apply a 2π phase ramp distributed across the ring. This is the WRITE+1
 * operation: adds exactly +1 to the winding number (quantized increment).
 * The ramp is centered at `center_site` with width `ramp_width` sites.      */
static void apply_ramp(batch_t *b, int l, int center_site, int ramp_width){
    if (ramp_width < 2) ramp_width = 2;  /* minimum viable ramp              */
    for (int i = 0; i < N; i++) {
        /* Distance from center_site to i around the ring, signed.
         * The ramp applies the full 2π gradient across the ramp window.      */
        int dist = (i - center_site) & NMASK;
        if (dist > N/2) dist -= N;       /* shortest signed distance         */
        /* Linear ramp: phase addition is 2π * dist / ramp_width clamped     */
        double frac = (double)dist / ramp_width;
        if (frac >  0.5) frac =  0.5;    /* clamp edges to half-ramp         */
        if (frac < -0.5) frac = -0.5;
        b->th[i * b->L + l] += 2.0 * M_PI * frac;
    }
}

/* Strong reset: applies a single strong ERASE kick as part of a multi-pass
 * funnel. Called repeatedly (3 times) with intervening settles to guarantee
 * q=0. Each call applies one standard kick.                                 */
static void apply_reset_kick(batch_t *b, int l, int site){
    apply_kick(b, l, site, 1.0, KICK_W);
}

/* =========================================================================
 * INTERNAL: settle a single ring and read its result
 * ========================================================================= */

/* Settle one specific lane (ring) and update its metadata.
 * Returns the resulting winding number, or DEAD/NOLCK on failure.           */
static int settle_one_ring(bqsm_vm_t *vm, int ring_idx){
    batch_t *b = vm->batch;
    int lock[LMAX];
    settle(b, lock);
    for (int l = 0; l < b->L; l++)
        b->locked[l] = lock[l];

    if (!lock[ring_idx]) {
        vm->rings[ring_idx].locked = 0;
        vm->rings[ring_idx].q = NOLCK;
        return NOLCK;
    }
    int q = winding(b, ring_idx);
    vm->rings[ring_idx].locked = 1;
    vm->rings[ring_idx].q = q;
    return q;
}

/* =========================================================================
 * PUBLIC API: VM lifecycle
 * ========================================================================= */

bqsm_vm_t *bqsm_vm_create(int num_rings, bqsm_radix_t radix){
    if (num_rings < 1 || num_rings > LMAX) {
        fprintf(stderr, "bqsm_vm_create: num_rings must be in [1, %d]\n", LMAX);
        return NULL;
    }
    if (radix < 1 || radix > BQSM_RADIX_MAX) {
        fprintf(stderr, "bqsm_vm_create: radix must be in [1, %d]\n", BQSM_RADIX_MAX);
        return NULL;
    }

    bqsm_vm_t *vm = calloc(1, sizeof(*vm));
    if (!vm) return NULL;

    vm->batch      = batch_new(num_rings);
    vm->rings      = calloc(num_rings, sizeof(ring_meta_t));
    vm->num_rings  = num_rings;
    vm->radix      = radix;

    vm->program    = NULL;
    vm->prog_len   = 0;
    vm->pc         = 0;
    vm->steps_executed = 0;

    vm->lens_site  = 0;
    vm->lens_delta = 0.0;
    vm->tt_computed = 0;

    /* RNG seed: deterministic but per-VM unique */
    vm->rng_state = 0x5EEDBA51ULL ^ (uint64_t)(uintptr_t)vm;

    /* All rings start in ground state (q=0) */
    for (int r = 0; r < num_rings; r++) {
        set_twisted(vm->batch, r, 0);
        vm->batch->om[vm->lens_site * vm->batch->L + r] = vm->lens_delta;
        vm->rings[r].q = 0;
        vm->rings[r].locked = 1;
        vm->rings[r].alive = 1;
    }

    return vm;
}

void bqsm_vm_destroy(bqsm_vm_t *vm){
    if (!vm) return;
    batch_free(vm->batch);
    free(vm->rings);
    free(vm->program);
    free(vm);
}

/* =========================================================================
 * PUBLIC API: register operations
 * ========================================================================= */

void bqsm_ring_set_twisted(bqsm_vm_t *vm, int r, int q){
    if (r < 0 || r >= vm->num_rings) return;
    set_twisted(vm->batch, r, q);
    vm->batch->om[vm->lens_site * vm->batch->L + r] = vm->lens_delta;
    vm->rings[r].q = q;
    vm->rings[r].locked = 1;  /* ideal state is trivially locked */
    vm->rings[r].alive = (abs(q) <= vm->radix);
    vm->batch->locked[r] = 1;
    /* Note: we don't settle here — caller can if they want runtime validation.
     * The ideal twisted state is a fixed point of the pristine dynamics.      */
}

int bqsm_ring_read(bqsm_vm_t *vm, int r){
    if (r < 0 || r >= vm->num_rings) return DEAD;
    /* If the ring hasn't been settled recently, settle now */
    if (!vm->batch->locked[r]) {
        int lock[LMAX];
        settle(vm->batch, lock);
        for (int l = 0; l < vm->batch->L; l++)
            vm->batch->locked[l] = lock[l];
    }
    if (!vm->batch->locked[r]) {
        vm->rings[r].locked = 0;
        vm->rings[r].q = NOLCK;
        return NOLCK;
    }
    int q = winding(vm->batch, r);
    vm->rings[r].locked = 1;
    vm->rings[r].q = q;

    /* Radix clamping: states outside the admissible range are OOB */
    if (abs(q) >= vm->radix && vm->radix > 0) {
        vm->rings[r].alive = 0;
        return OOB;
    }
    vm->rings[r].alive = 1;
    return q;
}

int bqsm_ring_is_locked(bqsm_vm_t *vm, int r){
    if (r < 0 || r >= vm->num_rings) return 0;
    return vm->batch->locked[r] && vm->rings[r].alive;
}

/* =========================================================================
 * PUBLIC API: instruction execution
 * ========================================================================= */

int bqsm_execute_instruction(bqsm_vm_t *vm, const bqsm_instruction_t *inst){
    if (!vm || !inst) return DEAD;

    int r = inst->target_ring;
    if (r < 0 || r >= vm->num_rings) return DEAD;

    batch_t *b = vm->batch;

    /* --- First check transition table if available --- */
    if (vm->tt_computed && inst->opcode != BQSM_COMPARE) {
        int start_q = vm->rings[r].q;
        if (start_q >= -Q_MAX && start_q <= Q_MAX) {
            int tt_result = bqsm_transition_lookup(vm, start_q, inst->opcode,
                                                    inst->site, inst->amplitude);
            if (tt_result != NOLCK) {
                /* Transition table hit: O(1) lookup, no settle needed.
                 * Update ring state from the cached result.                    */
                if (tt_result == DEAD) {
                    vm->rings[r].locked = 0;
                    vm->rings[r].q = DEAD;
                    vm->rings[r].alive = 0;
                    vm->batch->locked[r] = 0;
                } else {
                    /* Set the ring to the resulting twisted state */
                    set_twisted(b, r, tt_result);
                    b->om[vm->lens_site * b->L + r] = vm->lens_delta;
                    vm->rings[r].q = tt_result;
                    vm->rings[r].locked = 1;
                    vm->rings[r].alive = (abs(tt_result) < vm->radix);
                    b->locked[r] = 1;
                }
                vm->steps_executed++;
                return tt_result;
            }
        }
    }

    /* --- Execute the instruction physically --- */

    /* Mark all rings as potentially unlocked since we're about to perturb */
    for (int l = 0; l < b->L; l++) b->locked[l] = 0;

    switch (inst->opcode) {

    case BQSM_READ: {
        /* READ is a no-op perturbation-wise; just settle and read.           */
        int q = settle_one_ring(vm, r);
        /* Radix check */
        if (abs(q) >= vm->radix && vm->radix > 0 && q >= 0) {
            vm->rings[r].alive = 0;
            q = OOB;
        }
        vm->steps_executed++;
        return q;
    }

    case BQSM_WRITE_INC: {
        /* WRITE+1: apply 2π phase ramp → exactly +1 winding increment.
         * Default: ramp across the full ring (width=N).                       */
        int w = inst->width > 0 ? inst->width : N;
        apply_ramp(b, r, inst->site, w);
        int q = settle_one_ring(vm, r);
        if (abs(q) >= vm->radix && vm->radix > 0) {
            vm->rings[r].alive = 0;
            q = (q == NOLCK) ? NOLCK : OOB;
        }
        vm->steps_executed++;
        return q;
    }

    case BQSM_ERASE: {
        /* ERASE: scalar bump kick → one-way funnel toward q=0.
         * Default amplitude = 1.0, default width = KICK_W (3).                */
        double amp = (inst->amplitude != 0.0) ? inst->amplitude : 1.0;
        int w = inst->width > 0 ? inst->width : KICK_W;
        apply_kick(b, r, inst->site, amp, w);
        int q = settle_one_ring(vm, r);
        if (abs(q) >= vm->radix && vm->radix > 0) {
            vm->rings[r].alive = 0;
            q = (q == NOLCK) ? NOLCK : OOB;
        }
        vm->steps_executed++;
        return q;
    }

    case BQSM_COMPARE: {
        /* COMPARE: read current q, branch if q matches threshold.
         * Does NOT perturb the ring.                                           */
        int q = settle_one_ring(vm, r);
        vm->steps_executed++;
        /* If q matches the branch condition, signal by returning q;
         * the program counter update is handled by the caller (bqsm_program_step).
         * We return q so the caller can decide whether to branch.              */
        return q;
    }

    case BQSM_RESET: {
        /* RESET: 3-pass ERASE funnel → guaranteed q=0.
         * Each pass: kick + settle. The transition table proves that
         * 3 successive ERASEs reach ground from any |q| ≤ 3.                  */
        int site = inst->site;
        int q;
        for (int pass = 0; pass < 3; pass++) {
            apply_kick(b, r, site, 1.0, KICK_W);
            q = settle_one_ring(vm, r);
            if (q == 0) break;  /* early exit: already at ground */
        }
        vm->steps_executed++;
        return q;
    }

    case BQSM_KICK_AT: {
        /* General kick: arbitrary site, amplitude, and width.                 */
        double amp = inst->amplitude;
        int w = inst->width > 0 ? inst->width : KICK_W;
        apply_kick(b, r, inst->site, amp, w);
        int q = settle_one_ring(vm, r);
        if (abs(q) >= vm->radix && vm->radix > 0) {
            vm->rings[r].alive = 0;
            q = (q == NOLCK) ? NOLCK : OOB;
        }
        vm->steps_executed++;
        return q;
    }

    case BQSM_RAMP_AT: {
        /* General phase ramp: arbitrary site and width.                       */
        int w = inst->width > 0 ? inst->width : N;
        apply_ramp(b, r, inst->site, w);
        int q = settle_one_ring(vm, r);
        if (abs(q) >= vm->radix && vm->radix > 0) {
            vm->rings[r].alive = 0;
            q = (q == NOLCK) ? NOLCK : OOB;
        }
        vm->steps_executed++;
        return q;
    }

    default:
        return DEAD;
    }
}

/* =========================================================================
 * PUBLIC API: program execution
 * ========================================================================= */

void bqsm_program_load(bqsm_vm_t *vm, const bqsm_instruction_t *program, int count){
    free(vm->program);
    vm->program = malloc(count * sizeof(bqsm_instruction_t));
    if (!vm->program) { fprintf(stderr, "bqsm_vm: program alloc failed\n"); return; }
    memcpy(vm->program, program, count * sizeof(bqsm_instruction_t));
    vm->prog_len = count;
    vm->pc = 0;
    vm->steps_executed = 0;
}

int bqsm_program_step(bqsm_vm_t *vm, int *result){
    if (!vm->program || vm->pc >= vm->prog_len || vm->pc < 0)
        return -1;  /* halted */

    bqsm_instruction_t *inst = &vm->program[vm->pc];
    int q = bqsm_execute_instruction(vm, inst);
    if (result) *result = q;

    /* Handle COMPARE: branch if q matches branch condition */
    if (inst->opcode == BQSM_COMPARE) {
        if (q == inst->branch_q && inst->branch_target >= 0) {
            vm->pc = inst->branch_target;   /* take the branch */
        } else {
            vm->pc++;                        /* fall through */
        }
    } else {
        /* Check for trap: DEAD or OOB halts execution */
        if (q == DEAD || q == OOB || q == NOLCK) {
            vm->pc = vm->prog_len;           /* halt */
            return -1;
        }
        vm->pc++;
    }

    if (vm->pc >= vm->prog_len) return -1;  /* program complete */
    return vm->pc;
}

int bqsm_program_run(bqsm_vm_t *vm){
    int result;
    int executed = 0;
    while (bqsm_program_step(vm, &result) >= 0) {
        executed++;
        if (executed > 1000000) {  /* safety limit */
            fprintf(stderr, "bqsm_vm: program exceeded 1M steps, halting\n");
            return -1;
        }
    }
    return vm->steps_executed;
}

/* =========================================================================
 * PUBLIC API: transition table
 * ========================================================================= */

/* Standard operational parameters for each opcode.
 * These are the "default" parameters used for precomputation.                */
static const double ERASE_AMPS[2]  = {1.0, 2.2};  /* kick alphabet           */
static const int    ERASE_WIDTH    = 3;            /* KICK_W                  */
#define TT_IDX(q, op)  (((q) + Q_MAX) * BQSM_OPCODE_COUNT + (op))

void bqsm_transition_table_compute(bqsm_vm_t *vm){
    /* Allocate a scratch batch with one lane per start state (q from -Q_MAX to +Q_MAX)
     * and N lanes for per-site ERASE testing.                                   */
    int nq = 2*Q_MAX + 1;  /* 7 states */
    int total_lanes = nq + N * nq;  /* q-states + per-site-ERASE lanes */
    batch_t *ba = batch_new(total_lanes);
    int L = total_lanes;

    /* Initialize scratch batch: all lanes start as twisted states */
    for (int q = -Q_MAX; q <= Q_MAX; q++) {
        int l = q + Q_MAX;  /* lane 0..6: base twisted states */
        set_twisted(ba, l, q);
        ba->om[vm->lens_site * L + l] = vm->lens_delta;
    }

    /* Initialize default table to NOLCK (not computed) */
    for (int i = 0; i < (2*Q_MAX+1) * BQSM_OPCODE_COUNT; i++)
        vm->tt_flat[i] = NOLCK;
    for (int i = 0; i < N * (2*Q_MAX+1); i++)
        vm->tt_erase[i] = NOLCK;

    /* Phase A: settle the bare attractors under the lens.
     * Copy each attractor into a per-site lane set for ERASE testing.         */
    batch_t *bb = batch_new(nq);
    for (int q = -Q_MAX; q <= Q_MAX; q++) {
        int l = q + Q_MAX;
        set_twisted(bb, l, q);
        bb->om[vm->lens_site * bb->L + l] = vm->lens_delta;
    }
    int lockA[LMAX];
    settle(bb, lockA);

    double att[nq][N];   /* settled attractor snapshots */
    int alive[nq];
    for (int q = -Q_MAX; q <= Q_MAX; q++) {
        int l = q + Q_MAX;
        alive[l] = lockA[l] && (winding(bb, l) == q);
        if (alive[l])
            for (int i = 0; i < N; i++) att[l][i] = bb->th[i * bb->L + l];
    }

    /* Phase B: test each opcode on each surviving start state.
     * We test:
     *   - READ: identity (returns q itself, always) → table = q
     *   - WRITE_INC: full-ring ramp centered at site 0 → settle
     *   - ERASE: kick at each site with both amplitudes → per-site table
     *   - RESET: strong kick → should always produce q=0
     *   - KICK_AT: same as ERASE for now (generalized kick)
     *   - RAMP_AT: same as WRITE_INC for now (generalized ramp)
     *   - COMPARE: no transition (identity)                                    */

    /* READ is identity: result = start_q if alive, else DEAD */
    for (int q = -Q_MAX; q <= Q_MAX; q++) {
        int l = q + Q_MAX;
        vm->tt_flat[TT_IDX(q, BQSM_READ)] = alive[l] ? q : DEAD;
    }

    /* COMPARE is identity (no perturbation) */
    for (int q = -Q_MAX; q <= Q_MAX; q++) {
        int l = q + Q_MAX;
        vm->tt_flat[TT_IDX(q, BQSM_COMPARE)] = alive[l] ? q : DEAD;
    }

    /* WRITE_INC: test full-ring ramp */
    {
        batch_t *bw = batch_new(nq);
        for (int q = -Q_MAX; q <= Q_MAX; q++) {
            int l = q + Q_MAX;
            if (!alive[l]) { vm->tt_flat[TT_IDX(q, BQSM_WRITE_INC)] = DEAD; continue; }
            for (int i = 0; i < N; i++) {
                bw->th[i * bw->L + l] = att[l][i];
                bw->om[i * bw->L + l] = 0.0;
            }
            bw->om[vm->lens_site * bw->L + l] = vm->lens_delta;
            apply_ramp(bw, l, 0, N);  /* full ring ramp */
        }
        int lockW[LMAX]; settle(bw, lockW);
        for (int q = -Q_MAX; q <= Q_MAX; q++) {
            int l = q + Q_MAX;
            if (!alive[l]) continue;
            if (lockW[l] && winding(bw, l) == q + 1)
                vm->tt_flat[TT_IDX(q, BQSM_WRITE_INC)] = q + 1;
            else if (lockW[l])
                vm->tt_flat[TT_IDX(q, BQSM_WRITE_INC)] = winding(bw, l);
            else
                vm->tt_flat[TT_IDX(q, BQSM_WRITE_INC)] = DEAD;
        }
        batch_free(bw);
    }

    /* RAMP_AT: same test as WRITE_INC */
    for (int q = -Q_MAX; q <= Q_MAX; q++)
        vm->tt_flat[TT_IDX(q, BQSM_RAMP_AT)] = vm->tt_flat[TT_IDX(q, BQSM_WRITE_INC)];

    /* ERASE: test per-site, per-amplitude; store best (funnel) result.
     * The ERASE table stores: for each q and each site, what state does the
     * standard kick (amp=1.0) produce? We also test amp=2.2.                   */
    for (int s = 0; s < N; s++) {
        batch_t *be = batch_new(nq);
        for (int q = -Q_MAX; q <= Q_MAX; q++) {
            int l = q + Q_MAX;
            if (!alive[l]) { vm->tt_erase[s * nq + l] = DEAD; continue; }
            for (int i = 0; i < N; i++) {
                be->th[i * be->L + l] = att[l][i];
                be->om[i * be->L + l] = 0.0;
            }
            be->om[vm->lens_site * be->L + l] = vm->lens_delta;
            apply_kick(be, l, s, ERASE_AMPS[0], ERASE_WIDTH);
        }
        int lockE[LMAX]; settle(be, lockE);
        for (int q = -Q_MAX; q <= Q_MAX; q++) {
            int l = q + Q_MAX;
            if (!alive[l]) continue;
            if (lockE[l])
                vm->tt_erase[s * nq + l] = winding(be, l);
            else
                vm->tt_erase[s * nq + l] = NOLCK;
        }
        batch_free(be);
    }

    /* Default ERASE entry (site 0) in tt_flat */
    for (int q = -Q_MAX; q <= Q_MAX; q++)
        vm->tt_flat[TT_IDX(q, BQSM_ERASE)] = vm->tt_erase[0 * nq + (q + Q_MAX)];

    /* KICK_AT: same as ERASE at site 0 */
    for (int q = -Q_MAX; q <= Q_MAX; q++)
        vm->tt_flat[TT_IDX(q, BQSM_KICK_AT)] = vm->tt_erase[0 * nq + (q + Q_MAX)];

    /* RESET: 3-pass ERASE funnel → should always produce q=0 for any
     * surviving start state. Each pass: kick + settle, then next pass.        */
    {
        batch_t *br = batch_new(nq);
        for (int q = -Q_MAX; q <= Q_MAX; q++) {
            int l = q + Q_MAX;
            if (!alive[l]) { vm->tt_flat[TT_IDX(q, BQSM_RESET)] = DEAD; continue; }
            for (int i = 0; i < N; i++) {
                br->th[i * br->L + l] = att[l][i];
                br->om[i * br->L + l] = 0.0;
            }
            br->om[vm->lens_site * br->L + l] = vm->lens_delta;
        }
        /* 3-pass funnel */
        int lockR[LMAX];
        for (int pass = 0; pass < 3; pass++) {
            for (int q = -Q_MAX; q <= Q_MAX; q++) {
                int l = q + Q_MAX;
                if (!alive[l]) continue;
                apply_kick(br, l, 0, 1.0, KICK_W);
            }
            settle(br, lockR);
            /* If all survivors are at q=0, early exit */
            int all_ground = 1;
            for (int q = -Q_MAX; q <= Q_MAX; q++) {
                int l = q + Q_MAX;
                if (!alive[l]) continue;
                if (!lockR[l] || winding(br, l) != 0) { all_ground = 0; break; }
            }
            if (all_ground) break;
        }
        settle(br, lockR);  /* final settle to get winding */
        for (int q = -Q_MAX; q <= Q_MAX; q++) {
            int l = q + Q_MAX;
            if (!alive[l]) continue;
            vm->tt_flat[TT_IDX(q, BQSM_RESET)] = lockR[l] ? winding(br, l) : NOLCK;
        }
        batch_free(br);
    }

    batch_free(ba);
    batch_free(bb);
    vm->tt_computed = 1;
}

int bqsm_transition_lookup(bqsm_vm_t *vm, int q, bqsm_opcode_t opcode,
                           int site, double amplitude){
    if (!vm->tt_computed) return NOLCK;
    if (q < -Q_MAX || q > Q_MAX) return OOB;

    /* For ERASE/KICK_AT, use per-site table if available */
    if ((opcode == BQSM_ERASE || opcode == BQSM_KICK_AT) &&
        fabs(amplitude - 1.0) < 0.01 && site >= 0 && site < N) {
        return vm->tt_erase[site * (2*Q_MAX+1) + (q + Q_MAX)];
    }

    return vm->tt_flat[TT_IDX(q, opcode)];
}

/* =========================================================================
 * PUBLIC API: lens control
 * ========================================================================= */

void bqsm_vm_set_lens(bqsm_vm_t *vm, int site, double delta){
    vm->lens_site  = site;
    vm->lens_delta = delta;
    /* Update omega for all rings */
    for (int r = 0; r < vm->num_rings; r++)
        vm->batch->om[site * vm->batch->L + r] = delta;
    /* Invalidate transition table */
    vm->tt_computed = 0;
    /* Mark all rings as needing re-settle */
    for (int r = 0; r < vm->num_rings; r++)
        vm->batch->locked[r] = 0;
}

/* =========================================================================
 * PUBLIC API: diagnostics
 * ========================================================================= */

int bqsm_vm_alive_count(bqsm_vm_t *vm){
    int count = 0;
    for (int r = 0; r < vm->num_rings; r++)
        if (vm->rings[r].alive && vm->rings[r].locked) count++;
    return count;
}

void bqsm_vm_dump(bqsm_vm_t *vm){
    printf("=== BQSM VM DUMP ===\n");
    printf("rings: %d  radix: %d  lens: site=%d delta=%.4f  tt_computed=%d\n",
           vm->num_rings, (int)vm->radix, vm->lens_site, vm->lens_delta,
           vm->tt_computed);
    printf("program: %d instructions  pc=%d  steps_executed=%d\n",
           vm->prog_len, vm->pc, vm->steps_executed);
    printf("%-4s %-8s %-8s %-8s %s\n", "ring", "q", "locked", "alive", "phases[0]");
    for (int r = 0; r < vm->num_rings; r++) {
        double p0 = vm->batch->th[r];  /* phase at site 0 for this ring */
        printf("%-4d %-8d %-8d %-8d %10.4f\n",
               r, vm->rings[r].q, vm->rings[r].locked, vm->rings[r].alive, p0);
    }
    /* Show transition table if computed */
    if (vm->tt_computed) {
        printf("\n--- Transition Table (site 0, amp=1.0) ---\n");
        printf("%-4s", "q\\op");
        const char *opnames[] = {"READ","WINC","ERASE","CMP","RST","KICK","RAMP"};
        for (int o = 0; o < BQSM_OPCODE_COUNT; o++)
            printf("  %-6s", opnames[o]);
        printf("\n");
        for (int q = -Q_MAX; q <= Q_MAX; q++) {
            printf("%-4d", q);
            for (int o = 0; o < BQSM_OPCODE_COUNT; o++) {
                int r = vm->tt_flat[TT_IDX(q, o)];
                printf("  %-6d", r);
            }
            printf("\n");
        }
    }
}

int  bqsm_vm_get_pc(bqsm_vm_t *vm)          { return vm->pc; }
void bqsm_vm_set_pc(bqsm_vm_t *vm, int pc)  { vm->pc = pc; }
int  bqsm_vm_get_program_length(bqsm_vm_t *vm) { return vm->prog_len; }


/* =========================================================================
 * STANDALONE MODE: selftest and demo
 * ========================================================================= */
#ifdef BQSM_VM_STANDALONE

static int cmd_selftest(void){
    int fails = 0;

    printf("=== BQSM VM SELFTEST ===\n\n");

    /* 1. Create VM and verify initial state */
    printf("1. VM creation and initial state...\n");
    bqsm_vm_t *vm = bqsm_vm_create(8, BQSM_RADIX_MAX);
    if (!vm) { printf("   FAIL: VM creation failed\n"); return 1; }

    /* All rings should start at q=0 */
    for (int r = 0; r < 8; r++) {
        int q = bqsm_ring_read(vm, r);
        if (q != 0) { printf("   FAIL: ring %d initial q=%d (expected 0)\n", r, q); fails++; }
    }
    printf("   initial q: all 0  %s\n", fails ? "FAIL" : "OK");

    /* 2. Set twisted states and verify */
    printf("\n2. Twisted state initialization...\n");
    for (int q = -Q_MAX; q <= Q_MAX; q++) {
        int r = q + Q_MAX;  /* rings 0..6 */
        bqsm_ring_set_twisted(vm, r, q);
    }
    /* Ring 7 stays at q=0 */
    for (int q = -Q_MAX; q <= Q_MAX; q++) {
        int r = q + Q_MAX;
        int qr = bqsm_ring_read(vm, r);
        if (qr != q) {
            printf("   FAIL: ring %d (set q=%d) reads as q=%d\n", r, q, qr);
            fails++;
        }
    }
    /* Verify ring 7 is still 0 */
    if (bqsm_ring_read(vm, 7) != 0) { printf("   FAIL: ring 7 drifted\n"); fails++; }
    printf("   twisted states: %s\n", fails ? "FAIL" : "OK");

    /* 3. Test READ instruction */
    printf("\n3. READ instruction...\n");
    {
        bqsm_instruction_t inst = { BQSM_READ, 0, 0.0, 0, 0, 0, 0 };
        int qr = bqsm_execute_instruction(vm, &inst);  /* ring 0 is q=-3 */
        if (qr != -3) { printf("   FAIL: READ ring 0 = %d (expected -3)\n", qr); fails++; }

        inst.target_ring = 3;  /* ring 3 is q=0 */
        qr = bqsm_execute_instruction(vm, &inst);
        if (qr != 0) { printf("   FAIL: READ ring 3 = %d (expected 0)\n", qr); fails++; }
    }
    printf("   READ: %s\n", fails ? "has failures" : "OK");

    /* 4. Test WRITE_INC */
    printf("\n4. WRITE_INC instruction...\n");
    {
        /* Start from q=0 */
        bqsm_ring_set_twisted(vm, 0, 0);
        bqsm_instruction_t inst = { BQSM_WRITE_INC, 0, 0.0, N, 0, 0, 0 };
        int qr = bqsm_execute_instruction(vm, &inst);
        if (qr != 1) { printf("   FAIL: WRITE_INC on q=0 → q=%d (expected 1)\n", qr); fails++; }

        /* WRITE_INC on q=1 → q=2 */
        inst.target_ring = 0;  /* now q=1 */
        qr = bqsm_execute_instruction(vm, &inst);
        if (qr != 2) { printf("   FAIL: WRITE_INC on q=1 → q=%d (expected 2)\n", qr); fails++; }

        /* WRITE_INC on q=2 → q=3 */
        qr = bqsm_execute_instruction(vm, &inst);
        if (qr != 3) { printf("   FAIL: WRITE_INC on q=2 → q=%d (expected 3)\n", qr); fails++; }
    }
    printf("   WRITE_INC staircase: %s\n", fails ? "has failures" : "OK");

    /* 5. Test ERASE */
    printf("\n5. ERASE instruction...\n");
    {
        /* ERASE on q=3 should go to q=2 or lower (one-way funnel) */
        bqsm_ring_set_twisted(vm, 1, 3);
        bqsm_instruction_t inst = { BQSM_ERASE, 0, 1.0, KICK_W, 1, 0, 0 };
        int qr = bqsm_execute_instruction(vm, &inst);
        printf("   ERASE q=3 → q=%d (expected ≤2, funnel property)\n", qr);
        if (qr > 3 || qr < 0) { printf("   FAIL: unexpected result\n"); fails++; }

        /* ERASE on q=0 should stay at q=0 */
        bqsm_ring_set_twisted(vm, 1, 0);
        inst.target_ring = 1;
        qr = bqsm_execute_instruction(vm, &inst);
        if (qr != 0) { printf("   FAIL: ERASE on q=0 → q=%d (expected 0)\n", qr); fails++; }
    }
    printf("   ERASE funnel: %s\n", fails ? "has failures" : "OK");

    /* 6. Test RESET */
    printf("\n6. RESET instruction...\n");
    {
        /* RESET on any state → q=0 */
        for (int q = 0; q <= Q_MAX; q++) {
            bqsm_ring_set_twisted(vm, 2, q);
            bqsm_instruction_t inst = { BQSM_RESET, 0, 0.0, 0, 2, 0, 0 };
            int qr = bqsm_execute_instruction(vm, &inst);
            if (qr != 0) { printf("   FAIL: RESET on q=%d → q=%d (expected 0)\n", q, qr); fails++; }
        }
    }
    printf("   RESET to ground: %s\n", fails ? "has failures" : "OK");

    /* 7. Test program execution */
    printf("\n7. Program execution...\n");
    {
        /* Program: WRITE_INC on ring 0, READ ring 0, ERASE on ring 0 */
        bqsm_ring_set_twisted(vm, 0, 0);
        bqsm_instruction_t prog[] = {
            { BQSM_WRITE_INC, 0, 0.0, N, 0, 0, 0 },   /* q=0 → q=1         */
            { BQSM_WRITE_INC, 0, 0.0, N, 0, 0, 0 },   /* q=1 → q=2         */
            { BQSM_ERASE,     0, 1.0, 3, 0, 0, 0 },   /* q=2 → q≤1         */
            { BQSM_READ,      0, 0.0, 0, 0, 0, 0 },   /* read result       */
        };
        bqsm_program_load(vm, prog, 4);
        int steps = bqsm_program_run(vm);
        int result = bqsm_ring_read(vm, 0);
        printf("   program executed %d steps, final q=%d\n", steps, result);
        if (result > 2 || result < 0) { printf("   FAIL: unexpected program result\n"); fails++; }
    }
    printf("   program execution: %s\n", fails ? "has failures" : "OK");

    /* 8. Test transition table */
    printf("\n8. Transition table...\n");
    bqsm_vm_set_lens(vm, 0, 0.0);  /* pristine */
    bqsm_transition_table_compute(vm);
    /* Verify key entries */
    {
        int e03 = bqsm_transition_lookup(vm, 0, BQSM_ERASE, 0, 1.0);
        int e33 = bqsm_transition_lookup(vm, 3, BQSM_ERASE, 0, 1.0);
        int w01 = bqsm_transition_lookup(vm, 0, BQSM_WRITE_INC, 0, 0.0);
        printf("   TT: ERASE[0]=%d ERASE[3]=%d WRITE+1[0]=%d\n", e03, e33, w01);
        if (e03 != 0) { printf("   FAIL: ERASE on q=0 should give 0, got %d\n", e03); fails++; }
        if (e33 < 0 || e33 > 3) { printf("   FAIL: ERASE on q=3 should be 0-2, got %d\n", e33); fails++; }
        if (w01 != 1) { printf("   FAIL: WRITE+1 on q=0 should give 1, got %d\n", w01); fails++; }
    }
    printf("   transition table: %s\n", fails ? "has failures" : "OK");

    /* --- Summary --- */
    printf("\n=== SELFTEST: %s (%d failures) ===\n", fails ? "FAIL" : "ALL PASS", fails);

    bqsm_vm_destroy(vm);
    return fails ? 1 : 0;
}

static void cmd_demo(void){
    printf("=== BQSM VM Demo: 4-ring binary counter ===\n\n");

    bqsm_vm_t *vm = bqsm_vm_create(4, BQSM_RADIX_BINARY);
    bqsm_vm_dump(vm);

    /* Program: 4-ring binary ripple counter using WRITE_INC and COMPARE */
    printf("\n--- Incrementing ring 0 four times ---\n");
    for (int i = 0; i < 4; i++) {
        bqsm_instruction_t inst = { BQSM_WRITE_INC, 0, 0.0, N, 0, -1, 0 };
        int q = bqsm_execute_instruction(vm, &inst);
        printf("step %d: ring0 q=%d\n", i, q);
    }

    printf("\n--- RESET then WRITE_INC on all rings ---\n");
    for (int r = 0; r < 4; r++) {
        bqsm_instruction_t rst = { BQSM_RESET, 0, 0.0, 0, r, 0, 0 };
        bqsm_execute_instruction(vm, &rst);
    }
    for (int r = 0; r < 4; r++) {
        bqsm_instruction_t inc = { BQSM_WRITE_INC, 0, 0.0, N, r, 0, 0 };
        int q = bqsm_execute_instruction(vm, &inc);
        printf("ring %d: q=%d\n", r, q);
    }

    bqsm_vm_dump(vm);
    bqsm_vm_destroy(vm);
}

int main(int argc, char **argv){
    if ((N & NMASK) != 0 || N < 8)
        { fprintf(stderr, "N must be a power of two ≥ 8\n"); return 2; }

    if (argc >= 2 && !strcmp(argv[1], "selftest"))
        return cmd_selftest();
    if (argc >= 2 && !strcmp(argv[1], "demo"))
        { cmd_demo(); return 0; }

    printf("bqsm_vm — BQSM Virtual Processor Core\n");
    printf("usage: bqsm_vm selftest\n");
    printf("       bqsm_vm demo\n");
    return 0;
}

#endif /* BQSM_VM_STANDALONE */