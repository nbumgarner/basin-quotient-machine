/* ===========================================================================
 * cross_validate.c — Phase 0 Gate: Cross-Validation of Coupling Coefficients
 *
 *   Calibrate sf2, sf3, sf4 on ONE pair (2.0, 1.5).
 *   Test on 1000 random pairs in [-4,4]x[-4,4].
 *   Report RMS error distribution.
 *
 *   SUCCESS: RMS < 5% on all three ops → coefficients are stable physics constants.
 *   FAILURE: RMS > 5% → coefficients input-dependent → primitive unstable.
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

#define N       16
#define NMASK   (N-1)
#define NH      8
#define DT      0.50
#define T_CHUNK 60.0

/* ─── RNG ─────────────────────────────────────────────────────────────────── */
static uint64_t rng_s = 0x5EEDBA51ULL;
static inline double rng_uniform(void) {
    uint64_t z = (rng_s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return (double)((z ^ (z >> 31)) >> 11) * (1.0/9007199254740992.0);
}

/* ─── Mini-furnace (single ring) ──────────────────────────────────────────── */
typedef struct { double th[N], om[N], k1[N], k2[N], k3[N], k4[N], tmp[N]; } ring_t;

static void deriv(ring_t *r) {
    for (int i=0;i<N;i++) {
        int ip=(i+1)&NMASK, im=(i-1)&NMASK;
        r->tmp[i] = r->om[i] + sin(r->th[ip]-r->th[i]) + sin(r->th[im]-r->th[i]);
    }
}
static void rk4_step(ring_t *r) {
    deriv(r); memcpy(r->k1, r->tmp, N*sizeof(double));
    for (int i=0;i<N;i++) r->tmp[i]=r->th[i]+0.5*DT*r->k1[i]; deriv(r); memcpy(r->k2, r->tmp, N*sizeof(double));
    for (int i=0;i<N;i++) r->tmp[i]=r->th[i]+0.5*DT*r->k2[i]; deriv(r); memcpy(r->k3, r->tmp, N*sizeof(double));
    for (int i=0;i<N;i++) r->tmp[i]=r->th[i]+DT*r->k3[i]; deriv(r); memcpy(r->k4, r->tmp, N*sizeof(double));
    for (int i=0;i<N;i++) r->th[i]+=(DT/6.0)*(r->k1[i]+2.0*r->k2[i]+2.0*r->k3[i]+r->k4[i]);
}
static void settle(ring_t *r, int chunks) {
    int steps = (int)(T_CHUNK/DT);
    for (int c=0;c<chunks;c++) for (int s=0;s<steps;s++) rk4_step(r);
}

/* ─── Fourier analysis ────────────────────────────────────────────────────── */
static double ck_mag(ring_t *r, int k) {
    double re=0, im=0;
    for (int j=0;j<N;j++) {
        double a = -2.0*M_PI*k*j/N;
        re += r->th[j]*cos(a);
        im += r->th[j]*sin(a);
    }
    return sqrt(re*re + im*im) / N;
}

/* ─── Encode two values into c1 and c2 ────────────────────────────────────── */
static void encode(ring_t *r, double a, double b) {
    memset(r->th, 0, N*sizeof(double));
    memset(r->om, 0, N*sizeof(double));
    for (int j=0;j<N;j++) r->th[j] += 2.0*a*cos(2.0*M_PI*1.0*j/N);  /* c1 = a */
    for (int j=0;j<N;j++) r->th[j] += 2.0*b*cos(2.0*M_PI*2.0*j/N);  /* c2 = b */
}

/* ─── Run one multi-op: returns (c2, c3, c4) magnitudes ───────────────────── */
static void run_multiop(double a, double b, double *c2, double *c3, double *c4) {
    ring_t r;
    encode(&r, a, b);
    settle(&r, 2);
    *c2 = ck_mag(&r, 2);
    *c3 = ck_mag(&r, 3);
    *c4 = ck_mag(&r, 4);
}

/* ─── Calibrate on one pair ───────────────────────────────────────────────── */
static void calibrate(double a, double b, double *sf2, double *sf3, double *sf4) {
    double c2, c3, c4;
    run_multiop(a, b, &c2, &c3, &c4);
    *sf2 = (c2 > 1e-12) ? (a*a / c2) : 0;
    *sf3 = (c3 > 1e-12) ? (a*b / c3) : 0;
    *sf4 = (c4 > 1e-12) ? (b*b / c4) : 0;
}

/* ─── Test with given coefficients ────────────────────────────────────────── */
static void test_pair(double a, double b, double sf2, double sf3, double sf4,
                      double *err2, double *err3, double *err4) {
    double c2, c3, c4;
    run_multiop(a, b, &c2, &c3, &c4);

    double pred2 = c2 * sf2;
    double pred3 = c3 * sf3;
    double pred4 = c4 * sf4;

    double true2 = a*a;
    double true3 = a*b;
    double true4 = b*b;

    *err2 = (fabs(true2) > 1e-6) ? fabs(pred2 - true2) / fabs(true2) : fabs(pred2 - true2);
    *err3 = (fabs(true3) > 1e-6) ? fabs(pred3 - true3) / fabs(true3) : fabs(pred3 - true3);
    *err4 = (fabs(true4) > 1e-6) ? fabs(pred4 - true4) / fabs(true4) : fabs(pred4 - true4);
}

/* ─── Main ────────────────────────────────────────────────────────────────── */
int main() {
    printf("=== Phase 0: Cross-Validation Gate ===\n\n");

    /* Step 1: Calibrate on reference pair */
    const double a_cal = 2.0, b_cal = 1.5;
    double sf2, sf3, sf4;
    calibrate(a_cal, b_cal, &sf2, &sf3, &sf4);

    printf("Calibration pair: a=%.2f, b=%.2f\n", a_cal, b_cal);
    printf("  sf2 (a^2->c2) = %.6f\n", sf2);
    printf("  sf3 (a*b->c3) = %.6f\n", sf3);
    printf("  sf4 (b^2->c4) = %.6f\n\n", sf4);

    /* Verify calibration pair gives zero error (tautology check) */
    {
        double e2, e3, e4;
        test_pair(a_cal, b_cal, sf2, sf3, sf4, &e2, &e3, &e4);
        printf("Calibration pair error (should be ~0): e2=%.2e e3=%.2e e4=%.2e\n\n", e2, e3, e4);
    }

    /* Step 2: Test on 1000 random pairs */
    const int N_TEST = 1000;
    double sum_e2 = 0, sum_e3 = 0, sum_e4 = 0;
    double sumsq_e2 = 0, sumsq_e3 = 0, sumsq_e4 = 0;
    double max_e2 = 0, max_e3 = 0, max_e4 = 0;
    int count_e2 = 0, count_e3 = 0, count_e4 = 0;
    int large_e2 = 0, large_e3 = 0, large_e4 = 0;

    printf("Testing %d random pairs in [-4, 4] x [-4, 4]...\n\n", N_TEST);

    for (int i = 0; i < N_TEST; i++) {
        double a = (rng_uniform() - 0.5) * 8.0;
        double b = (rng_uniform() - 0.5) * 8.0;

        double e2, e3, e4;
        test_pair(a, b, sf2, sf3, sf4, &e2, &e3, &e4);

        if (fabs(a*a) > 1e-6) { sum_e2 += e2; sumsq_e2 += e2*e2; count_e2++; if (e2 > max_e2) max_e2 = e2; if (e2 > 0.5) large_e2++; }
        if (fabs(a*b) > 1e-6) { sum_e3 += e3; sumsq_e3 += e3*e3; count_e3++; if (e3 > max_e3) max_e3 = e3; if (e3 > 0.5) large_e3++; }
        if (fabs(b*b) > 1e-6) { sum_e4 += e4; sumsq_e4 += e4*e4; count_e4++; if (e4 > max_e4) max_e4 = e4; if (e4 > 0.5) large_e4++; }
    }

    /* Step 3: Report results */
    double rms_e2 = count_e2 > 0 ? sqrt(sumsq_e2 / count_e2) : 0;
    double rms_e3 = count_e3 > 0 ? sqrt(sumsq_e3 / count_e3) : 0;
    double rms_e4 = count_e4 > 0 ? sqrt(sumsq_e4 / count_e4) : 0;
    double mean_e2 = count_e2 > 0 ? sum_e2 / count_e2 : 0;
    double mean_e3 = count_e3 > 0 ? sum_e3 / count_e3 : 0;
    double mean_e4 = count_e4 > 0 ? sum_e4 / count_e4 : 0;

    printf("=== RESULTS ===\n\n");
    printf("%-12s %8s %8s %8s %8s %8s %8s\n", "op", "count", "mean%", "RMS%", "max%", ">50%", "PASS?");
    printf("%-12s %8d %8.2f %8.2f %8.2f %8d %8s\n", "a^2->c2", count_e2, mean_e2*100, rms_e2*100, max_e2*100, large_e2, rms_e2 < 0.05 ? "YES" : "NO");
    printf("%-12s %8d %8.2f %8.2f %8.2f %8d %8s\n", "a*b->c3", count_e3, mean_e3*100, rms_e3*100, max_e3*100, large_e3, rms_e3 < 0.05 ? "YES" : "NO");
    printf("%-12s %8d %8.2f %8.2f %8.2f %8d %8s\n", "b^2->c4", count_e4, mean_e4*100, rms_e4*100, max_e4*100, large_e4, rms_e4 < 0.05 ? "YES" : "NO");

    printf("\n");
    int all_pass = (rms_e2 < 0.05) && (rms_e3 < 0.05) && (rms_e4 < 0.05);
    printf("=== GATE: %s ===\n", all_pass ? "PASS - Coefficients are stable physics constants" : "FAIL - Coefficients are input-dependent");

    if (!all_pass) {
        printf("\nDiagnostics:\n");
        printf("  RMS errors > 5%% means the coupling coefficients vary with input amplitude.\n");
        printf("  This makes the primitive non-deterministic without per-input calibration.\n");
        printf("  Options: (a) per-input calibration table, (b) different encoding, (c) accept heuristic mode.\n");
    }

    /* Step 4: Error vs amplitude correlation */
    printf("\n=== Error vs |a|,|b| correlation ===\n");
    printf("Binning by max(|a|,|b|):\n");

    double bin_sum[4][3] = {{0}};
    double bin_sumsq[4][3] = {{0}};
    int bin_cnt[4][3] = {{0}};
    for (int i = 0; i < N_TEST; i++) {
        double a = (rng_uniform() - 0.5) * 8.0;
        double b = (rng_uniform() - 0.5) * 8.0;
        double m = fmax(fabs(a), fabs(b));
        int bin = (m < 1.0) ? 0 : (m < 2.0) ? 1 : (m < 3.0) ? 2 : 3;

        double e2, e3, e4;
        test_pair(a, b, sf2, sf3, sf4, &e2, &e3, &e4);

        if (fabs(a*a) > 1e-6) { bin_sum[bin][0] += e2; bin_sumsq[bin][0] += e2*e2; bin_cnt[bin][0]++; }
        if (fabs(a*b) > 1e-6) { bin_sum[bin][1] += e3; bin_sumsq[bin][1] += e3*e3; bin_cnt[bin][1]++; }
        if (fabs(b*b) > 1e-6) { bin_sum[bin][2] += e4; bin_sumsq[bin][2] += e4*e4; bin_cnt[bin][2]++; }
    }

    printf("  Range     a^2_RMS%%  a*b_RMS%%  b^2_RMS%%  (n)\n");
    const char *labels[4] = {"[0,1)", "[1,2)", "[2,3)", "[3,4]"};
    for (int b = 0; b < 4; b++) {
        double rms2 = bin_cnt[b][0] > 0 ? sqrt(bin_sumsq[b][0] / bin_cnt[b][0]) : 0;
        double rms3 = bin_cnt[b][1] > 0 ? sqrt(bin_sumsq[b][1] / bin_cnt[b][1]) : 0;
        double rms4 = bin_cnt[b][2] > 0 ? sqrt(bin_sumsq[b][2] / bin_cnt[b][2]) : 0;
        printf("  %s  %8.1f  %8.1f  %8.1f  (n=%d/%d/%d)\n",
               labels[b], rms2*100, rms3*100, rms4*100,
               bin_cnt[b][0], bin_cnt[b][1], bin_cnt[b][2]);
    }

    return all_pass ? 0 : 1;
}