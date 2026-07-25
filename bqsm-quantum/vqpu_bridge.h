/* ===========================================================================
 * vqpu_bridge.h — bridge between vQPU and ring_furnace.c
 *
 *   Declares the batch_t struct and functions exported from ring_furnace.c
 *   for vQPU use. ring_furnace.c is the single source of truth — this header
 *   mirrors its public API only.
 *
 *   Part of bqsm-quantum/.
 * ======================================================================== */

#ifndef VQPU_BRIDGE_H
#define VQPU_BRIDGE_H

#include <stddef.h>   /* size_t */
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Batch struct (mirrors ring_furnace.c typedef) ─────────────────────── */
#ifndef BATCH_T_DEFINED
#define BATCH_T_DEFINED
typedef struct {
    int     L;              /* live lanes in this batch                      */
    double *th;             /* phases        [N*L] SoA layout                */
    double *om;             /* per-lane ω    [N*L] (lens per lane)           */
    double *k1, *k2;        /* RK4 work arrays [N*L] (declared here so      */
    double *k3, *k4, *tmp;  /*   vqpu.c can access th[] and om[] directly)   */
    int    *locked;         /* per-lane lock flags (used by TEST_VQPU)       */
} batch_t;
#endif

/* ─── Lifecycle ──────────────────────────────────────────────────────────── */
batch_t *batch_new(int L);
void     batch_free(batch_t *b);

/* ─── Settling ────────────────────────────────────────────────────────────── */
/* Partial settle: run K chunks, no lock check. Ring stays mid-evolution.
 * The Fourier spectrum at partial settle IS the vQPU state. */
void settle_partial(batch_t *b, int max_chunks);

/* ─── Fourier spectrum ─────────────────────────────────────────────────────  */
/* DFT of lane l's phases. c_real[k] + i·c_imag[k] for k=0..N/2.            */
void fft_snapshot(const batch_t *b, int l, double *c_real, double *c_imag);

/* Apply pure Fourier-mode perturbation: P[j] = amp·sin(2π·k·j/N + phase).  */
void apply_fourier_kick(batch_t *b, int l, int k_mode, double amp, double phase);

/* ─── State manipulation ──────────────────────────────────────────────────── */
/* Winding number: topological invariant of lane l.                          */
int  winding(const batch_t *b, int l);

/* Write the ideal twisted state q into lane l: θ[j] = 2πqj/N.              */
void set_twisted(batch_t *b, int l, int q);

/* Standard 3-site kick at position `site` with amplitude `amp`.            */
void apply_kick(batch_t *b, int l, int site, double amp);

#ifdef __cplusplus
}
#endif

#endif /* VQPU_BRIDGE_H */