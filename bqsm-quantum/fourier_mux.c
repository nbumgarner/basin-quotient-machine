/* ===========================================================================
 * fourier_mux.c — vQPU Fourier Multiplexing Demo
 *
 *   Packs 8 independent scalar values into the Fourier harmonics of one
 *   BQSM ring, settles once (physics computes all harmonic interactions
 *   simultaneously), then unpacks 8 separate results.
 *
 *   This demonstrates "multiple ops packed into one operation, decomposed
 *   into separate results" — the ring's nonlinear mode coupling IS the
 *   fused computation.
 *
 * BUILD: cc -O3 -std=c11 fourier_mux.c -o fourier_mux -lm
 * ======================================================================== */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define N  16
#define NMASK (N-1)
#define NH  8   /* N/2 independent harmonics */
#define DT  0.50
#define T_CHUNK 60.0

typedef struct { double th[N], om[N], k1[N], k2[N], k3[N], k4[N], tmp[N]; } ring_t;

/* ─── Mini-furnace (single ring) ───────────────────────────────────────── */
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

/* ─── Fourier analysis ─────────────────────────────────────────────────── */
static void fft_analyze(ring_t *r, double *mag, double *phase) {
    for (int k=1;k<=NH;k++) {
        double re=0, im=0;
        for (int j=0;j<N;j++) {
            double a = -2.0*M_PI*k*j/N;
            re += r->th[j]*cos(a);
            im += r->th[j]*sin(a);
        }
        re/=N; im/=N;
        mag[k-1] = sqrt(re*re+im*im);
        phase[k-1] = atan2(im,re);
    }
}

/* ─── Inverse DFT: synthesize phases from harmonic amplitudes ──────────── */
static void fft_synthesize(ring_t *r, const double *c_mag, const double *c_phase) {
    memset(r->th, 0, N*sizeof(double));
    memset(r->om, 0, N*sizeof(double));
    for (int k=1;k<=NH;k++) {
        double cr = c_mag[k-1]*cos(c_phase[k-1]);
        double ci = c_mag[k-1]*sin(c_phase[k-1]);
        for (int j=0;j<N;j++) {
            double angle = 2.0*M_PI*k*j/N;
            r->th[j] += 2.0*(cr*cos(angle) - ci*sin(angle));
        }
    }
}

/* ─── Main: Pack → Settle → Unpack ─────────────────────────────────────── */
int main() {
    printf("═══ vQPU Fourier Multiplexing Demo ═══\n");
    printf("  Pack 8 values into harmonics → settle → unpack 8 results\n\n");

    /* ── PACK: encode 8 independent "input" values as harmonic magnitudes ── */
    double inputs[NH] = {0.5, 1.0, 0.3, 0.8, 0.2, 0.6, 0.9, 0.1};
    double phases[NH] = {0, M_PI/4, 0, M_PI/3, 0, 0, M_PI/2, 0};

    printf("─── PACKED INPUTS (8 harmonics) ───\n");
    printf("  k   |c_k|    phase    meaning\n");
    printf("  ───────────────────────────────\n");
    for (int k=0;k<NH;k++)
        printf("  %d   %.3f   %+.3f   input_%d\n", k+1, inputs[k], phases[k], k);

    ring_t r;
    fft_synthesize(&r, inputs, phases);

    /* Verify: read back the packed values */
    double mag_before[NH], phase_before[NH];
    fft_analyze(&r, mag_before, phase_before);
    printf("\n  Verify (pre-settle): ");
    for (int k=0;k<NH;k++) printf("%.2f ", mag_before[k]);
    printf("\n");

    /* ── PROCESS: settle → physics couples all harmonics simultaneously ── */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    settle(&r, 2);  /* 2 chunks = 240 RK4 steps */
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt_us = (t1.tv_sec-t0.tv_sec)*1e6 + (t1.tv_nsec-t0.tv_nsec)/1e3;

    /* ── UNPACK: read 8 separate results from Fourier spectrum ── */
    double mag_after[NH], phase_after[NH];
    fft_analyze(&r, mag_after, phase_after);

    printf("\n─── UNPACKED RESULTS ───\n");
    printf("  k   |c_k|_out  change   meaning\n");
    printf("  ────────────────────────────────\n");
    for (int k=0;k<NH;k++) {
        double delta = mag_after[k] - mag_before[k];
        printf("  %d   %.4f    %+.4f   result_%d\n", k+1, mag_after[k], delta, k);
    }

    /* ── Throughput ── */
    double ops_per_sec = (double)NH / (dt_us * 1e-6);
    printf("\n─── THROUGHPUT ───\n");
    printf("  8 harmonic results in %.1f µs (1 settle, 2 chunks)\n", dt_us);
    printf("  Effective: %.0f results/second (single ring)\n", ops_per_sec);
    printf("  At 256 rings: %.0f results/second\n", ops_per_sec * 256.0);

    /* ── Key insight ── */
    printf("\n─── ARCHITECTURAL INSIGHT ───\n");
    printf("  Each harmonic k carries one independent computation.\n");
    printf("  The ring's nonlinear coupling g(p,q) computes all\n");
    printf("  mode interactions in PARALLEL during one settle cycle.\n");
    printf("  Pack N/2 values → settle once → unpack N/2 results.\n");
    printf("  This is NOT SIMD — the coupling IS the fused operation.\n");

    return 0;
}