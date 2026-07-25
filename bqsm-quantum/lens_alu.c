/* ===========================================================================
 * lens_alu.c — Programmable Lens-ALU on the BQSM ring furnace (v3)
 *
 *   Demonstrates: one ring, one settle = multiple fused math operations.
 *   The lens profile (detuning vector) changes WHICH operations the physics
 *   computes and with what coupling strength.
 *
 *   Calibration: per-lens direct scale factors measured from physics.
 *   sf₂ = a²_true / c₂_measured  →  a² = c₂ · sf₂
 *   sf₃ = (a·b)_true / c₃_measured  →  a·b = c₃ · sf₃  
 *   sf₄ = b²_true / c₄_measured  →  b² = c₄ · sf₄
 *
 *   These scale factors ARE the lens — different ω vectors change the
 *   coupling, producing different c_k for the same inputs.
 *
 * BUILD: cc -O3 -std=c11 lens_alu.c ../ring_furnace.c -o lens_alu -lm -Wl,--allow-multiple-definition
 * ======================================================================== */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    int L;
    double *th, *om, *k1, *k2, *k3, *k4, *tmp;
    int *locked;
} batch_t;

extern batch_t *batch_new(int L);
extern void     batch_free(batch_t *b);
extern void     settle_partial(batch_t *b, int max_chunks);
extern void     fft_snapshot(const batch_t *b, int l, double *c_real, double *c_imag);
extern int      winding(const batch_t *b, int l);

#define N           16
#define N_HALF      (N/2 + 1)
#define MAX_LENSES  8
#define CHUNKS      2

static uint64_t rng_s = 0xCAFEBABE5EEDBA51ULL;
static inline double rng_uniform(void) {
    uint64_t z = (rng_s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return (double)((z ^ (z >> 31)) >> 11) * (1.0/9007199254740992.0);
}

/* ─── Lens profile ──────────────────────────────────────────────────────── */
typedef struct {
    const char *name;
    double omega[N];
    
    /* Coupling scale factors (calibrated once per lens at reference point) */
    double sf_a2;    /* a² = |c₂| · sf_a2 */
    double sf_ab;    /* a·b = |c₃| · sf_ab */
    double sf_b2;    /* b² = |c₄| · sf_b2 */
    
    /* Active operations (which harmonics carry usable signal) */
    int has_a2, has_ab, has_b2;
} lens_profile_t;

static lens_profile_t lenses[MAX_LENSES];
static int n_lenses = 0;

/* ══════════════════════════════════════════════════════════════════════════ */
static void define_lens(const char *name, const double *omega) {
    lens_profile_t *l = &lenses[n_lenses++];
    l->name = name;
    memcpy(l->omega, omega, N * sizeof(double));
    l->sf_a2 = l->sf_ab = l->sf_b2 = 0;
    l->has_a2 = l->has_ab = l->has_b2 = 0;
}

/* ══════════════════════════════════════════════════════════════════════════
 * CORE: cos-based encoding → apply lens → settle → read spectrum
 * ════════════════════════════════════════════════════════════════════════ */
static void lens_forward(double a, double b, lens_profile_t *lens,
                         double c_real[N_HALF], double c_imag[N_HALF]) {
    batch_t *r = batch_new(1);
    
    /* Cos-based encoding: matches standalone mult_op_demo.c exactly.
     * θ[j] = 2a·cos(2π·1·j/N) + 2b·cos(2π·2·j/N)
     * Pre-settle DFT: |c₁| = a, |c₂| = b */
    for (int j = 0; j < N; j++) {
        double ph = 2.0*a*cos(2.0*M_PI*1.0*j/N)
                  + 2.0*b*cos(2.0*M_PI*2.0*j/N);
        r->th[j] = ph;  /* L=1 batch */
    }
    
    /* Apply lens detuning */
    for (int i = 0; i < N; i++)
        r->om[i] = lens->omega[i];
    
    settle_partial(r, CHUNKS);
    fft_snapshot(r, 0, c_real, c_imag);
    batch_free(r);
}

/* ══════════════════════════════════════════════════════════════════════════
 * CALIBRATION: measure scale factors at reference point (a=2, b=1.5)
 * ════════════════════════════════════════════════════════════════════════ */
static void calibrate_lens(lens_profile_t *lens) {
    double a = 2.0, b = 1.5;   /* calibration reference point */
    double cr[N_HALF], ci[N_HALF];
    lens_forward(a, b, lens, cr, ci);
    
    double c1_amp = sqrt(cr[1]*cr[1] + ci[1]*ci[1]);
    double c2_amp = sqrt(cr[2]*cr[2] + ci[2]*ci[2]);
    double c3_amp = sqrt(cr[3]*cr[3] + ci[3]*ci[3]);
    double c4_amp = sqrt(cr[4]*cr[4] + ci[4]*ci[4]);
    
    /* a² from c₂ — always active (c₁·c₁ → c₂ is always above threshold) */
    lens->sf_a2 = (c2_amp > 0.001) ? (a*a) / c2_amp : 0;
    lens->has_a2 = (c2_amp > 0.001);
    
    /* a·b from c₃ — active when c₃ measurable */
    lens->sf_ab = (c3_amp > 0.001) ? (a*b) / c3_amp : 0;
    lens->has_ab = (c3_amp > 0.001);
    
    /* b² from c₄ — threshold-gated (c₂·c₂ → c₄ needs higher amplitude) */
    lens->sf_b2 = (c4_amp > 0.001) ? (b*b) / c4_amp : 0;
    lens->has_b2 = (c4_amp > 0.001);
    
    printf("  %-16s |c₁|=%.4f |c₂|=%.4f |c₃|=%.4f |c₄|=%.4f  sf₂=%.4f sf₃=%.4f sf₄=%.4f",
           lens->name, c1_amp, c2_amp, c3_amp, c4_amp,
           lens->sf_a2, lens->sf_ab, lens->sf_b2);
    if (!lens->has_b2) printf("  [b² below threshold]");
    printf("\n");
}

/* ══════════════════════════════════════════════════════════════════════════
 * DECODE: apply calibrated scale factors
 * ════════════════════════════════════════════════════════════════════════ */
static void decode_results(lens_profile_t *lens,
                           const double c_real[N_HALF], const double c_imag[N_HALF],
                           double *a2_d, double *ab_d, double *b2_d) {
    double c2 = sqrt(c_real[2]*c_real[2] + c_imag[2]*c_imag[2]);
    double c3 = sqrt(c_real[3]*c_real[3] + c_imag[3]*c_imag[3]);
    double c4 = sqrt(c_real[4]*c_real[4] + c_imag[4]*c_imag[4]);
    
    *a2_d = c2 * lens->sf_a2;
    *ab_d = c3 * lens->sf_ab;
    *b2_d = c4 * lens->sf_b2;
}

/* ══════════════════════════════════════════════════════════════════════════
 * DEMO: one lens, one (a,b) pair
 * ════════════════════════════════════════════════════════════════════════ */
static void demo_lens(lens_profile_t *lens, double a, double b) {
    double cr[N_HALF], ci[N_HALF];
    lens_forward(a, b, lens, cr, ci);
    double a2_d, ab_d, b2_d;
    decode_results(lens, cr, ci, &a2_d, &ab_d, &b2_d);
    
    double e_a2 = fabs(a2_d - a*a)/fmax(fabs(a*a),0.01)*100;
    double e_ab = fabs(ab_d - a*b)/fmax(fabs(a*b),0.01)*100;
    double e_b2 = fabs(b2_d - b*b)/fmax(fabs(b*b),0.01)*100;
    
    printf("  %-16s a²=%.4f (%.1f%%) a·b=%.4f (%.1f%%) b²=%.4f (%.1f%%)",
           lens->name, a2_d, e_a2, ab_d, e_ab, b2_d, e_b2);
    if (!lens->has_b2) printf("  [no b²]");
    printf("\n");
}

/* ══════════════════════════════════════════════════════════════════════════
 * PROGRAMMABLE ALU: same inputs, all lenses — which ops?
 * ════════════════════════════════════════════════════════════════════════ */
static void demo_programmable(double a, double b) {
    printf("\n");
    printf("═════════════════════════════════════════════════════════════\n");
    printf("═══ PROGRAMMABLE LENS-ALU: a=%.1f b=%.1f, %d lenses ═══\n", 
           a, b, n_lenses);
    printf("  Ground truth: a²=%.2f  a·b=%.2f  b²=%.2f\n", a*a, a*b, b*b);
    printf("═════════════════════════════════════════════════════════════\n\n");
    
    printf("%-18s %10s %10s %10s %8s %8s %8s  %s\n",
           "LENS", "|c₂|", "|c₃|", "|c₄|", 
           "a²", "a·b", "b²", "err(a·b)");
    printf("%-18s %10s %10s %10s %8s %8s %8s  %s\n",
           "────", "────", "────", "────",
           "──", "───", "──", "────────");
    
    for (int i = 0; i < n_lenses; i++) {
        lens_profile_t *l = &lenses[i];
        double cr[N_HALF], ci[N_HALF];
        lens_forward(a, b, l, cr, ci);
        
        double c2 = sqrt(cr[2]*cr[2] + ci[2]*ci[2]);
        double c3 = sqrt(cr[3]*cr[3] + ci[3]*ci[3]);
        double c4 = sqrt(cr[4]*cr[4] + ci[4]*ci[4]);
        
        double a2_d = c2 * l->sf_a2;
        double ab_d = c3 * l->sf_ab;
        double b2_d = c4 * l->sf_b2;
        
        double e_ab = fabs(ab_d - a*b)/fmax(fabs(a*b),0.01)*100;
        
        printf("%-18s %10.4f %10.4f %10.4f %8.2f %8.2f %8.2f",
               l->name, c2, c3, c4, a2_d, ab_d, b2_d);
        if (e_ab <= 10.0)
            printf("  ✓ a·b=%.2f (%.1f%%)", ab_d, e_ab);
        else
            printf("  ✗ a·b=%.2f (%.1f%%)", ab_d, e_ab);
        printf("\n");
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * LENS SWEEP: show how lens gain steers coupling
 * ════════════════════════════════════════════════════════════════════════ */
static void demo_lens_sweep(double a, double b) {
    printf("\n\n─── LENS SWEEP: detuning ω[0] vs coupling strength ───\n");
    printf("  Fixed inputs a=%.1f b=%.1f, sweep ω[0] from 0 to 0.5\n", a, b);
    printf("  %6s %10s %10s %11s %10s %10s\n",
           "ω[0]", "|c₁|", "|c₂|(a²)", "sf₂=a²/c₂", "|c₃|(a·b)", "|c₄|(b²)");
    
    for (double w = 0.0; w <= 0.51; w += 0.05) {
        double om[N] = {0};
        om[0] = w;
        lens_profile_t tmp = {.name = "sweep"};
        memcpy(tmp.omega, om, N*sizeof(double));
        
        double cr[N_HALF], ci[N_HALF];
        lens_forward(a, b, &tmp, cr, ci);
        
        double c1 = sqrt(cr[1]*cr[1]+ci[1]*ci[1]);
        double c2 = sqrt(cr[2]*cr[2]+ci[2]*ci[2]);
        double c3 = sqrt(cr[3]*cr[3]+ci[3]*ci[3]);
        double c4 = sqrt(cr[4]*cr[4]+ci[4]*ci[4]);
        
        double sf2 = (c2 > 0.001) ? a*a/c2 : 0;
        
        printf("  %6.2f %10.4f %10.4f %11.4f %10.4f %10.4f\n",
               w, c1, c2, sf2, c3, c4);
    }
    
    printf("\n  Key: as ω[0] increases, sf₂=a²/c₂ changes — the lens STEERS\n");
    printf("        how much of the a² signal survives in c₂.\n");
    printf("        c₃ (a·b coupling) also shifts with lens gain.\n");
}

/* ══════════════════════════════════════════════════════════════════════════
 * ACCURACY SWEEP: test scale factor validity across input range
 * ════════════════════════════════════════════════════════════════════════ */
static void sweep_lens(lens_profile_t *lens, int n_trials) {
    int n_ok_ab = 0;
    double sum_e_ab = 0, max_e_ab = 0;
    
    for (int t = 0; t < n_trials; t++) {
        double a = 1.0 + rng_uniform() * 3.0;  /* [1, 4] */
        double b = 1.0 + rng_uniform() * 3.0;
        
        double cr[N_HALF], ci[N_HALF];
        lens_forward(a, b, lens, cr, ci);
        double a2_d, ab_d, b2_d;
        decode_results(lens, cr, ci, &a2_d, &ab_d, &b2_d);
        
        double e_ab = fabs(ab_d - a*b)/fmax(fabs(a*b),0.01);
        sum_e_ab += e_ab; 
        if (e_ab > max_e_ab) max_e_ab = e_ab;
        if (e_ab < 0.10) n_ok_ab++;  /* < 10% error */
    }
    
    printf("  %-16s a·b: RMS=%.1f%% max=%.1f%% passed(<10%%)=%d/%d\n",
           lens->name, sum_e_ab/n_trials*100, max_e_ab*100, n_ok_ab, n_trials);
}

/* ══════════════════════════════════════════════════════════════════════════
 * TIMING: benchmark throughput
 * ════════════════════════════════════════════════════════════════════════ */
static void demo_timing(void) {
    printf("\n─── Throughput (single ring, IDENTITY lens) ───\n");
    struct timespec t0, t1;
    const int N_BENCH = 2000;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < N_BENCH; i++) {
        double cr[N_HALF], ci[N_HALF];
        lens_forward(2.0, 1.5, &lenses[0], cr, ci);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt_us = (t1.tv_sec-t0.tv_sec)*1e6 + (t1.tv_nsec-t0.tv_nsec)/1e3;
    printf("  %d forward passes in %.0f µs (%.1f µs/pass)\n", 
           N_BENCH, dt_us, dt_us/N_BENCH);
    printf("  3 fused ops/pass → %.1f M fused-op/s single-core\n",
           N_BENCH * 3.0 / dt_us);
    
    /* Batch: 256 rings in parallel */
    printf("\n─── Batch throughput (256 rings, IDENTITY lens) ───\n");
    printf("  (Conceptual — batch_new(256) + single settle)\n");
    printf("  At 2857 samples/s (ring_furnace benchmark):\n");
    printf("  256 × 3 fused ops × 2857 Hz = %.1f M fused-op/s\n",
           256 * 3 * 2857.0 / 1e6);
}

/* ══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    int quick = (argc > 1 && !strcmp(argv[1], "quick"));
    
    printf("═══ BQSM Lens-ALU v3: Programmable Multi-Op Co-Processor ═══\n");
    printf("   N=16 ring furnace, cos encoding, %d settle chunks\n", CHUNKS);
    printf("   Lens = detuning vector ω[0..15] — selects coupling profile\n\n");
    
    /* ── Define 7 lenses ── */
    const double om_id[N]     = {0};
    const double om_s0[N]     = {0.3, 0};
    const double om_s0_2[N]   = {0.1, 0};
    const double om_anti[N]   = {0.3, 0,0,0,0,0,0,0, -0.3, 0,0,0,0,0,0,0};
    const double om_pair[N]   = {0.25, 0.25, 0};
    const double om_ramp[N]   = {0.00,0.03,0.06,0.09,0.12,0.15,0.18,0.21,
                                  0.24,0.27,0.30,0.33,0.36,0.39,0.42,0.45};
    const double om_quad[N]   = {0.2, 0,0,0, 0.2, 0,0,0, -0.2, 0,0,0, -0.2, 0,0,0};
    
    define_lens("IDENTITY",     om_id);
    define_lens("SITE0_0.1",    om_s0_2);
    define_lens("SITE0_0.3",    om_s0);
    define_lens("ANTIPODAL",    om_anti);
    define_lens("QUAD_PAIR",    om_pair);
    define_lens("LINEAR_RAMP",  om_ramp);
    define_lens("QUAD",         om_quad);
    
    /* ── Phase 1: Calibrate all lenses ── */
    printf("─── Calibration (ref: a=2.0, b=1.5) ───\n");
    for (int i = 0; i < n_lenses; i++)
        calibrate_lens(&lenses[i]);
    
    /* ── Phase 2: Verify calibration (same reference point → should be 0% err) ── */
    printf("\n─── Self-consistency check (should be ≈0%% error) ───\n");
    for (int i = 0; i < n_lenses; i++)
        demo_lens(&lenses[i], 2.0, 1.5);
    
    /* ── Phase 3: Programmable ALU — same inputs, different lenses ── */
    demo_programmable(2.0, 1.5);
    demo_programmable(3.0, 2.0);
    
    /* ── Phase 4: Lens sweep ── */
    demo_lens_sweep(2.0, 1.5);
    
    if (quick) return 0;
    
    /* ── Phase 5: Cross-validation with different inputs ── */
    printf("\n\n─── Cross-validation: same scale factors, new inputs ───\n");
    printf("  (calibrated at a=2.0,b=1.5, tested elsewhere)\n\n");
    double test_pairs[][2] = {{1.5, 1.0}, {2.5, 2.0}, {3.5, 1.5}, {3.0, 3.0}};
    for (int tp = 0; tp < 4; tp++) {
        double a = test_pairs[tp][0], b = test_pairs[tp][1];
        printf("  a=%.1f b=%.1f (a²=%.2f a·b=%.2f b²=%.2f):\n", 
               a, b, a*a, a*b, b*b);
        for (int i = 0; i < n_lenses; i++)
            demo_lens(&lenses[i], a, b);
        printf("\n");
    }
    
    /* ── Phase 6: Accuracy sweep ── */
    printf("\n─── Random sweep: 200 pairs per lens ───\n");
    for (int i = 0; i < n_lenses; i++)
        sweep_lens(&lenses[i], 200);
    
    /* ── Phase 7: Timing ── */
    demo_timing();
    
    printf("\n═══ Lens-ALU: DONE ═══\n");
    printf("  Key result: the lens IS the instruction. Different ω vectors\n");
    printf("  produce different coupling profiles. Calibrate once — then\n");
    printf("  select lens → encode inputs → settle → decode fused operations.\n");
    printf("  One ring, one settle = M fused math ops. Physics = ALU.\n");
    
    return 0;
}