/* ===========================================================================
 * fft_snapshot.h — Fourier Spectrum Tracking for vQPU
 *
 *   Computes the discrete Fourier transform of ring phase distributions
 *   during settling. This is the vQPU's "state readout" — the complex
 *   harmonic amplitudes {c_k} that carry the computational state between
 *   gate operations.
 *
 *   Part of bqsm-quantum/ — vQPU Phase 1 implementation.
 *   Uses ring_furnace.c's batch_t struct (SoA layout, N oscillators).
 *
 * BUILD (standalone test):
 *   cc -O3 -std=c11 -DTEST_FFT_SNAPSHOT fft_snapshot.c -o fft_snapshot -lm
 *
 * INTEGRATION:
 *   #include "fft_snapshot.h" in ring_furnace.c or vqpu.c
 *   Call fft_snapshot(b, lane, c_real, c_imag) inside the settle loop
 *   at each chunk boundary.
 * ======================================================================== */

#ifndef FFT_SNAPSHOT_H
#define FFT_SNAPSHOT_H

#include <stdint.h>
#include <math.h>
#include <complex.h>  /* C11 complex numbers — standard, no deps */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ─── Ring constants (match ring_furnace.c) ─────────────────────────────── */
#define FFT_N      16              /* oscillators per ring */
#define FFT_N_HALF 8               /* N/2 independent complex harmonics */

/* ─── batch_t forward declaration (match ring_furnace.c SoA layout) ─────── */
typedef struct {
    int     L;                     /* number of lanes */
    double *th;                    /* phases: [N * L] SoA, th[i*L + lane] */
    double *om;                    /* natural frequencies */
    double *k1, *k2, *k3, *k4;    /* RK4 scratch */
    double *tmp;                   /* temporary workspace */
    int    *locked;                /* per-lane lock flags */
} batch_t;

/* ─── DFT of one lane's phase distribution ─────────────────────────────── */

/* Compute the N-point DFT of lane l's phases.
 * c_real[k] = Re(c_k), c_imag[k] = Im(c_k) for k = 0..N-1.
 * c_0 is the DC component (mean phase, usually irrelevant).
 * c_k and c_{N-k} are complex conjugates for real-valued θ.
 * Normalization: c_k = (1/N) Σ θ[j] · exp(-2πi·k·j/N)
 *
 * The winding number q ∝ Im(c_1) · N / (2π).
 */
static inline void fft_snapshot(const batch_t *b, int lane,
                                double *c_real, double *c_imag) {
    int L = b->L;
    for (int k = 0; k < FFT_N; k++) {
        double re = 0.0, im = 0.0;
        for (int j = 0; j < FFT_N; j++) {
            double theta_j = b->th[j * L + lane];
            double angle   = -2.0 * M_PI * (double)(k * j) / (double)FFT_N;
            re += theta_j * cos(angle);
            im += theta_j * sin(angle);
        }
        c_real[k] = re / (double)FFT_N;
        c_imag[k] = im / (double)FFT_N;
    }
}

/* ─── Convenience: independent harmonics (k=1..N/2-1) ──────────────────── */

/* Extract the N_HALF independent complex harmonics (k=1..N_HALF).
 * k=0 (DC) is skipped — it's gauge (global phase rotation).
 * k=N_HALF (Nyquist) is real-only (conjugate symmetry forces Im=0).
 * Returns the total power (sum |c_k|²) as a diagnostic.
 */
static inline double fft_snapshot_harmonics(const batch_t *b, int lane,
                                            double complex *harmonics) {
    double c_real[FFT_N], c_imag[FFT_N];
    fft_snapshot(b, lane, c_real, c_imag);

    double power = 0.0;
    for (int k = 1; k <= FFT_N_HALF; k++) {
        double ck_re = c_real[k];
        double ck_im = (k == FFT_N_HALF) ? 0.0 : c_imag[k]; /* Nyquist: Im=0 */
        harmonics[k - 1] = ck_re + ck_im * I;  /* C11 complex */
        power += ck_re * ck_re + ck_im * ck_im;
    }
    return power;
}

/* ─── Winding number from FFT ──────────────────────────────────────────── */

/* The winding number q is the imaginary part of the fundamental harmonic,
 * scaled to wrap [0, 2π) per oscillator. This should match winding() in
 * ring_furnace.c when the ring is settled.
 *
 *   q = round(Im(c_1) · N / (2π))
 *
 * For diagnostic: compare fft_winding(b, lane) to winding(b, lane).
 * They should agree within ±1 at full settle, diverge during mid-settle.
 */
static inline int fft_winding(const batch_t *b, int lane) {
    double c_real[FFT_N], c_imag[FFT_N];
    fft_snapshot(b, lane, c_real, c_imag);
    double q_continuous = c_imag[1] * (double)FFT_N / (2.0 * M_PI);
    return (int)lround(q_continuous);
}

/* ─── Interference visibility computation ──────────────────────────────── */

/* Track c_1(t) through a settle cycle and compute the mode-coupling
 * visibility. Called at each chunk boundary inside the settle loop.
 *
 * Parameters:
 *   b         — batch (one lane used)
 *   lane      — which lane to track
 *   c1_history — array of length max_chunks, filled with c_1 at each chunk
 *   chunk     — current chunk index (0-based)
 *
 * Returns: complex c_1 at this chunk (for the caller to key perturbations on).
 */
static inline double complex fft_track_c1(const batch_t *b, int lane,
                                          double complex *c1_history,
                                          int chunk) {
    double c_real[FFT_N], c_imag[FFT_N];
    fft_snapshot(b, lane, c_real, c_imag);

    double complex c1 = c_real[1] + c_imag[1] * I;
    if (c1_history) c1_history[chunk] = c1;
    return c1;
}

/* ─── Interference visibility metric ───────────────────────────────────── */

/* Given c_1 values at each chunk boundary across a full settle cycle,
 * compute the visibility V = Δq / (2·Q_MAX) where Δq is the spread of
 * winding number across the settle trajectory.
 *
 * For the interference visibility experiment:
 *   - Sweep φ and record c1_history for each φ
 *   - For each φ, compute q_final = fft_winding at last chunk
 *   - V = (max(q_final) - min(q_final)) / (2*Q_MAX) across φ
 */
static inline double fft_visibility(const int *q_final, int n_phi, int q_max) {
    int q_min_val = q_final[0], q_max_val = q_final[0];
    for (int i = 1; i < n_phi; i++) {
        if (q_final[i] < q_min_val) q_min_val = q_final[i];
        if (q_final[i] > q_max_val) q_max_val = q_final[i];
    }
    return (double)(q_max_val - q_min_val) / (2.0 * (double)q_max);
}

/* ─── Diagnostic: print Fourier spectrum ───────────────────────────────── */

#include <stdio.h>
static inline void fft_print_spectrum(const batch_t *b, int lane) {
    double c_real[FFT_N], c_imag[FFT_N];
    fft_snapshot(b, lane, c_real, c_imag);

    printf("─── Fourier Spectrum (lane %d) ───\n", lane);
    printf("  k    Re(c_k)     Im(c_k)     |c_k|       phase\n");
    printf("  ─────────────────────────────────────────────\n");
    for (int k = 1; k <= FFT_N_HALF; k++) {
        double ck_re = c_real[k];
        double ck_im = (k == FFT_N_HALF) ? 0.0 : c_imag[k];
        double mag   = sqrt(ck_re * ck_re + ck_im * ck_im);
        double phase = atan2(ck_im, ck_re);
        printf("  %d   %+10.6f  %+10.6f  %10.6f  %+8.4f\n",
               k, ck_re, ck_im, mag, phase);
    }
    printf("  q (FFT) = %d\n", fft_winding(b, lane));
}

#endif /* FFT_SNAPSHOT_H */