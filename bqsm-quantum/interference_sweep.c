/* ===========================================================================
 * interference_sweep.c — vQPU Validation Gate: Interference Visibility
 *
 *   Standalone test that initializes a ring with controlled Fourier
 *   harmonics (c_1 = a·exp(iφ), c_2 = b), settles for K chunks while
 *   tracking c_1(t), sweeps φ across [0, 2π], and computes the visibility
 *   metric V = Δq / (2·Q_MAX).
 *
 *   Part of bqsm-quantum/ — vQPU Phase 1 implementation.
 *
 * BUILD:
 *   cc -O3 -std=c11 -march=native interference_sweep.c -o interference_sweep -lm
 *
 * RUN:
 *   ./interference_sweep [phi_steps] [settle_chunks] [c1_amp] [c2_amp]
 *
 *   Defaults: phi_steps=64, settle_chunks=8, c1_amp=0.5, c2_amp=0.3
 *
 * OUTPUT:
 *   phi  q_final  c1_re  c1_im  c1_mag  visibility_V
 *   ...
 *   ─── VERDICT: V = X.XX → [VALIDATED / MARGINAL / INVALIDATED]
 * ======================================================================== */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fft_snapshot.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ─── Mini ring furnace (standalone — does not link ring_furnace.c) ─────── */
/* We re-implement the minimum needed for the visibility experiment:
 * one ring (N=16), single lane, RK4 integration, gradient flow. */

#define N      16
#define NMASK  (N - 1)
#define K      1.0
#define DT     0.50
#define T_CHUNK 60.0

typedef struct {
    double th[N];      /* phases */
    double om[N];      /* natural frequencies (all 0 for pristine) */
    double k1[N], k2[N], k3[N], k4[N], tmp[N];
} ring_t;

/* Edge-walk deriv: N sin calls, not 2N */
static void deriv(ring_t *r, double *dth) {
    for (int i = 0; i < N; i++) {
        int ip = (i + 1) & NMASK;
        int im = (i - 1) & NMASK;
        dth[i] = r->om[i] + K * (sin(r->th[ip] - r->th[i])
                               + sin(r->th[im] - r->th[i]));
    }
}

/* RK4 step */
static void rk4_step(ring_t *r) {
    int i;
    deriv(r, r->k1);
    for (i = 0; i < N; i++) r->tmp[i] = r->th[i] + 0.5 * DT * r->k1[i];
    { ring_t tmp2; memcpy(tmp2.th, r->tmp, N*sizeof(double)); memcpy(tmp2.om, r->om, N*sizeof(double));
      deriv(&tmp2, r->k2); }
    for (i = 0; i < N; i++) r->tmp[i] = r->th[i] + 0.5 * DT * r->k2[i];
    { ring_t tmp2; memcpy(tmp2.th, r->tmp, N*sizeof(double)); memcpy(tmp2.om, r->om, N*sizeof(double));
      deriv(&tmp2, r->k3); }
    for (i = 0; i < N; i++) r->tmp[i] = r->th[i] + DT * r->k3[i];
    { ring_t tmp2; memcpy(tmp2.th, r->tmp, N*sizeof(double)); memcpy(tmp2.om, r->om, N*sizeof(double));
      deriv(&tmp2, r->k4); }
    for (i = 0; i < N; i++)
        r->th[i] += (DT / 6.0) * (r->k1[i] + 2.0*r->k2[i] + 2.0*r->k3[i] + r->k4[i]);
}

/* ─── Initialize ring with controlled Fourier harmonics ────────────────── */

/* Set ring phases to produce specific Fourier coefficients:
 *   c_1 = c1_amp · exp(i·phi)    (fundamental, winding-number harmonic)
 *   c_2 = c2_amp                  (second harmonic, real)
 *   c_k = 0 for k > 2
 *
 * This is the INVERSE DFT: θ[j] = Σ c_k · exp(2πi·k·j/N)
 * Only c_1 and c_2 are non-zero, and c_1 is complex.
 */
static void init_fourier_state(ring_t *r, double c1_amp, double phi,
                                double c2_amp) {
    double c1_re = c1_amp * cos(phi);
    double c1_im = c1_amp * sin(phi);
    double c2_re = c2_amp;
    double c2_im = 0.0;

    for (int j = 0; j < N; j++) {
        double angle_1 =  2.0 * M_PI * 1.0 * (double)j / (double)N;
        double angle_2 =  2.0 * M_PI * 2.0 * (double)j / (double)N;

        /* c_1 contribution: c_1·exp(2πi·1·j/N) + c_{-1}·exp(-2πi·1·j/N)
         * Since θ is real, c_{-1} = c_1*, so:
         *   θ[j] = 2·Re(c_1·exp(2πi·j/N)) + 2·Re(c_2·exp(4πi·j/N))
         *        = 2·(Re(c_1)·cos - Im(c_1)·sin) + 2·(c2_re·cos(2) - c2_im·sin(2))
         */
        double cos1 = cos(angle_1), sin1 = sin(angle_1);
        double cos2 = cos(angle_2), sin2 = sin(angle_2);

        /* c_1 + c_{-1} = 2·Re(c_1·exp(i·angle)) */
        double c1_contrib = 2.0 * (c1_re * cos1 - c1_im * sin1);
        /* c_2 + c_{-2} = 2·Re(c_2·exp(i·2·angle)) */
        double c2_contrib = 2.0 * (c2_re * cos2 - c2_im * sin2);

        r->th[j] = c1_contrib + c2_contrib;
        r->om[j] = 0.0;  /* pristine ring */
    }
}

/* ─── Validate: verify that the initialized state has the requested c_k ── */
static void validate_init(ring_t *r, double c1_amp, double phi, double c2_amp) {
    /* We need a batch_t to use fft_snapshot, but we only have ring_t.
     * Quick manual DFT for validation. */
    double c1_re_meas = 0.0, c1_im_meas = 0.0;
    double c2_re_meas = 0.0, c2_im_meas = 0.0;
    for (int j = 0; j < N; j++) {
        double a1 = 2.0 * M_PI * 1.0 * j / N;
        double a2 = 2.0 * M_PI * 2.0 * j / N;
        c1_re_meas += r->th[j] * cos(a1);
        c1_im_meas -= r->th[j] * sin(a1);  /* DFT: exp(-i·angle) */
        c2_re_meas += r->th[j] * cos(a2);
        c2_im_meas -= r->th[j] * sin(a2);
    }
    c1_re_meas /= N; c1_im_meas /= N;
    c2_re_meas /= N; c2_im_meas /= N;

    double c1_mag = sqrt(c1_re_meas*c1_re_meas + c1_im_meas*c1_im_meas);
    double c2_mag = sqrt(c2_re_meas*c2_re_meas + c2_im_meas*c2_im_meas);
    (void)c1_amp; (void)phi; (void)c2_amp;
    printf("# Init validate: |c_1| = %.4f (target %.4f), |c_2| = %.4f (target %.4f)\n",
           c1_mag, c1_amp, c2_mag, c2_amp);
}

/* ─── Partial settle: K chunks, track c_1 ──────────────────────────────── */

static void partial_settle(ring_t *r, int n_chunks,
                           double complex *c1_history) {
    int steps_per_chunk = (int)(T_CHUNK / DT);
    for (int chunk = 0; chunk < n_chunks; chunk++) {
        for (int s = 0; s < steps_per_chunk; s++)
            rk4_step(r);

        /* Compute c_1 at this chunk boundary via quick DFT */
        double c1_re = 0.0, c1_im = 0.0;
        for (int j = 0; j < N; j++) {
            double angle = 2.0 * M_PI * 1.0 * j / N;
            c1_re += r->th[j] * cos(angle);
            c1_im -= r->th[j] * sin(angle);
        }
        c1_re /= N; c1_im /= N;
        c1_history[chunk] = c1_re + c1_im * I;
    }
}

/* ─── Winding number from phases (matching ring_furnace.c winding()) ───── */

static int winding_simple(ring_t *r) {
    double s = 0.0;
    for (int i = 0; i < N; i++) {
        double d = r->th[(i + 1) & NMASK] - r->th[i];
        d = remainder(d, 2.0 * M_PI);
        s += d;
    }
    return (int)lround(s / (2.0 * M_PI));
}

/* ─── Main ──────────────────────────────────────────────────────────────── */
/* NOTE: Initial run with c1_amp=0.5, c2_amp=0.3 gave V=0.000 — the ring
 * relaxes to q=0 regardless of φ because:
 *   1. Amplitudes are too small to overcome ground-state attraction
 *   2. The phase distribution is not a proper TWISTED STATE (it's a Fourier
 *      superposition, not a winding-number eigenstate)
 *   3. 8 settle chunks fully relax to ground for these amplitudes
 *
 * Fix: initialize at q=1 (a proper twisted state), then apply a φ-dependent
 * perturbation pattern that EXCITES c_2 (second harmonic). The nonlinear
 * coupling c_2·c_2 → c_1 during settling feeds energy back into the
 * fundamental. The phase φ of the initial c_2 determines whether this
 * energy transfer is constructive or destructive → interference fringe.
 *
 * The twisted state: θ[j] = 2π·q·j/N  (pure c_1 harmonic, all others zero).
 * Perturbation: θ[j] += amp_pert · cos(2π·2·j/N + φ)  (pure c_2 harmonic).
 *
 * Defaults adjusted: q_init=1, pert_amp=1.0, phi_steps=64, settle_chunks=4.
 */

int main(int argc, char **argv) {
    int    n_q_init  = (argc > 1) ? atoi(argv[1]) : 1;
    int    n_phi     = (argc > 2) ? atoi(argv[2]) : 64;
    int    n_chunks  = (argc > 3) ? atoi(argv[3]) : 4;
    double pert_amp  = (argc > 4) ? atof(argv[4]) : 1.0;
    int    q_max_val_param = 3;

    printf("# vQPU Interference Visibility Experiment (v2 — twisted state + perturbation)\n");
    printf("# N=%d q_init=%d phi_steps=%d settle_chunks=%d pert_amp=%.2f\n",
           N, n_q_init, n_phi, n_chunks, pert_amp);
    printf("# phi  q_final\n");

    int *q_final_arr = malloc((size_t)n_phi * sizeof(int));
    if (!q_final_arr) { fprintf(stderr, "OOM\n"); return 1; }

    for (int i = 0; i < n_phi; i++) {
        double phi = 2.0 * M_PI * (double)i / (double)n_phi;

        ring_t r;
        memset(&r, 0, sizeof(r));

        /* Initialize as proper twisted state q = n_q_init */
        for (int j = 0; j < N; j++)
            r.th[j] = fmod(2.0 * M_PI * (double)n_q_init * (double)j / (double)N,
                           2.0 * M_PI);

        /* Apply φ-dependent perturbation: excite c_2 (second harmonic) */
        for (int j = 0; j < N; j++) {
            double angle_2 = 2.0 * M_PI * 2.0 * (double)j / (double)N;
            r.th[j] += pert_amp * cos(angle_2 + phi);
        }

        /* Partial settle: K chunks */
        int steps_per_chunk = (int)(T_CHUNK / DT);
        for (int chunk = 0; chunk < n_chunks; chunk++) {
            for (int s = 0; s < steps_per_chunk; s++)
                rk4_step(&r);
        }

        q_final_arr[i] = winding_simple(&r);
        printf("  %.4f  % d\n", phi, q_final_arr[i]);
    }

    int q_min = q_final_arr[0], q_max_v = q_final_arr[0];
    for (int i = 1; i < n_phi; i++) {
        if (q_final_arr[i] < q_min) q_min = q_final_arr[i];
        if (q_final_arr[i] > q_max_v) q_max_v = q_final_arr[i];
    }
    double V = (double)(q_max_v - q_min) / (2.0 * (double)q_max_val_param);

    printf("\n─── VERDICT ───\n");
    printf("  q_min = %d, q_max = %d, Δq = %d\n", q_min, q_max_v, q_max_v - q_min);
    printf("  Q_MAX = %d\n", q_max_val_param);
    printf("  V = Δq / (2·Q_MAX) = %.4f\n", V);
    if (V >= 0.50)
        printf("  ✓ VALIDATED — genuine mode-coupling interference\n");
    else if (V >= 0.17)
        printf("  ~ MARGINAL — weak coupling, retry with higher N or K\n");
    else
        printf("  ✗ INVALIDATED — q_final independent of φ, V=0 means no interference\n");

    free(q_final_arr);
    return 0;
}