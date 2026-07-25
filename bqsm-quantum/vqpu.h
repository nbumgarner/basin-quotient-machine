/* ===========================================================================
 * vqpu.h — vQPU Core Architecture
 *
 *   Virtual Quantum Processing Unit built on BQSM ring furnace physics.
 *   Implements the six-layer vQPU architecture:
 *     1. Fourier state encoding (fft_snapshot from ring_furnace.c)
 *     2. Gate ISA (perturbation patterns on Fourier harmonics)
 *     3. Calibration (wiring to octopus battery gap table)
 *     4. Adiabatic programming (sweep schedules)
 *     5. Capability bounds (what CAN / CANNOT accelerate)
 *     6. Validation (interference visibility gate)
 *
 *   Part of bqsm-quantum/ — Phase 2 implementation.
 *   Requires ring_furnace.c for batch_t, settle_partial(), fft_snapshot(),
 *   apply_fourier_kick(), winding(), set_twisted(), apply_kick().
 *
 * BUILD:
 *   cc -O3 -std=c11 -march=native -c vqpu.c -o vqpu.o
 *   cc -O3 -std=c11 -march=native ring_furnace.c vqpu_test.c -o vqpu_test -lm
 *
 * INTEGRATION:
 *   #include "vqpu.h"
 *   vqpu_circuit_run(&circuit);
 * ======================================================================== */

#ifndef VQPU_H
#define VQPU_H

#include <stdint.h>
#include <complex.h>
#include <math.h>
#include <string.h>    /* memcpy for ring save/restore */
#include <stdio.h>     /* debug printing */

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Constants (match ring_furnace.c) ──────────────────────────────────── */
#define VQPU_N         16       /* oscillators per ring */
#define VQPU_Q_MAX     3        /* max stable |q| at N=16 */
#define VQPU_N_HALF    8        /* N/2 for DFT */
#define VQPU_K          1.0     /* coupling strength */
#define VQPU_DT         0.50    /* FAST integration step */
#define VQPU_T_CHUNK    60.0    /* simulation time per settle chunk */
#define VQPU_LMAX       256     /* max lanes per batch */
#define VQPU_DEFAULT_CHUNKS  4  /* default partial settle chunks */

/* ─── Opcodes ────────────────────────────────────────────────────────────── */
typedef enum {
    VOP_NOP     = 0,    /* no-op (identity gate)                              */
    VOP_INIT    = 1,    /* set_twisted to |q⟩ — bypasses gradient flow       */
    VOP_HAD     = 2,    /* k=2 Fourier kick for superposition from |q⟩>0     */
    VOP_PHASE   = 3,    /* global phase rotation (instant, no settle)         */
    VOP_RAMP    = 4,    /* 2π ramp at wavenumber k, excites c_k               */
    VOP_LENS    = 5,    /* detune one site, reprogram transition table        */
    VOP_ERASE   = 6,    /* bias toward q=0 funnel                             */
    VOP_RESET   = 7,    /* strong erase → forced q=0                          */
    VOP_MEASURE = 8,    /* full settle + winding() snap                       */
    VOP_CNOT    = 9,    /* conditional NOT: q_c selects lens for target       */
    VOP_SWEEP   = 10,   /* adiabatic lens δ(t) sweep                          */
    VOP_FOURIER = 11,   /* apply Fourier kick at specific (k, amp, phase)     */
    VOP_COUNT          /* number of opcodes                                  */
} vopcode_t;

/* ─── Gate parameters ────────────────────────────────────────────────────── */
typedef struct {
    int    site;           /* perturbation site or lens site                  */
    int    k_mode;         /* Fourier wavenumber (1..N/2)                     */
    double amp;            /* perturbation amplitude                          */
    double phase;          /* phase offset (radians)                          */
    double delta;          /* lens detuning strength                          */
    int    control_ring;   /* control ring index (vCNOT)                      */
    int    target_ring;    /* target ring index                               */
    int    settle_chunks;  /* chunks for partial settle (0 = default)         */
    /* Sweep parameters */
    int    sweep_steps;    /* number of δ steps in adiabatic sweep            */
    double sweep_delta_start;
    double sweep_delta_end;
} vparam_t;

/* ─── Gate structure ─────────────────────────────────────────────────────── */
typedef struct {
    vopcode_t op;
    int       target;      /* ring index (or -1 for all)                      */
    vparam_t  params;
} vgate_t;

/* ─── Ring state ─────────────────────────────────────────────────────────── */
typedef struct {
    double th[VQPU_N];              /* phases (the actual physical state)      */
    double om[VQPU_N];              /* natural frequencies (lens = detune)     */
    double c_real[VQPU_N];          /* Fourier coefficients (cached after FFT) */
    double c_imag[VQPU_N];
    double c_power;                 /* total harmonic power Σ|c_k|²           */
    int    q;                       /* cached winding number (-1 = unstale)   */
    int    measured;                /* flag: has been measured since last gate */
    double visibility;              /* interference visibility (after sweep)   */
} vring_t;

/* ─── Circuit ────────────────────────────────────────────────────────────── */
#define VQPU_MAX_RINGS  256
#define VQPU_MAX_GATES  1024

typedef struct {
    int       n_rings;
    vring_t  *rings;               /* allocated ring array                     */
    int       n_gates;
    vgate_t  *gates;               /* allocated gate array                     */
    int      *measures;            /* measurement results (one per ring)       */
    double    visibility;          /* circuit-level visibility (for sweeps)    */
    int       step_count;          /* total settle chunks executed             */

    /* Diagnostics */
    double    total_sim_time;      /* total simulation time units              */
    double    total_wall_ms;       /* estimated wall-clock time                */

    /* Gap table for adiabatic scheduling (populated from gap_extract.py)     */
    double    gap_table[16][VQPU_Q_MAX+1];  /* gap(delta_index, q)              */
    int       gap_table_loaded;    /* flag: gap table is valid                 */
} vcircuit_t;

/* ─── Gap table (from gap_extract.py, measured) ─────────────────────────── */
static const double VQPU_GAP_0       = 0.15224;    /* pristine gap N=16 K=1.0 */
static const double VQPU_DELTA_CRIT[4] = {          /* δ_crit for each q       */
    2.0,      /* q=0  — centuries (never closes in measured range)           */
    1.0,      /* q=±1 — survives all measured δ                             */
    0.55,     /* q=±2 — estimated closure                                   */
    0.175     /* q=±3 — measured closure at δ=0.175                         */
};
static const double VQPU_MARGIN      = 0.10;       /* 10% of gap² for safety  */

#include "vqpu_bridge.h"

/* ─── External functions (from ring_furnace.c) ──────────────────────────── */
/* Declared in vqpu_bridge.h — no extern needed here. */

/* ─── vQPU API ───────────────────────────────────────────────────────────── */

/* Circuit lifecycle */
void        vqpu_circuit_init(vcircuit_t *c, int n_rings);
void        vqpu_circuit_free(vcircuit_t *c);
void        vqpu_circuit_reset(vcircuit_t *c);
void        vqpu_circuit_add_gate(vcircuit_t *c, vopcode_t op, int target, vparam_t params);

/* Gate execution */
int         vqpu_gate_dispatch(vcircuit_t *c, vgate_t *gate);
int         vqpu_circuit_run(vcircuit_t *c);

/* Fourier state management */
void        vqpu_fourier_snapshot(vcircuit_t *c, int ring_idx);
void        vqpu_apply_fourier_kick(vcircuit_t *c, int ring_idx, int k_mode, double amp, double phase);

/* Adiabatic scheduling */
void        vqpu_load_gap_table(vcircuit_t *c);
double      vqpu_gap(double delta, int q);
double      vqpu_adiabatic_step(double delta, double gap_min);
int         vqpu_adiabatic_sweep(vcircuit_t *c, int ring_idx, double d_start, double d_end, int max_steps);

/* Validation */
double      vqpu_interference_visibility(vcircuit_t *c, int ring_idx, int q_init, double pert_amp, int phi_steps, int chunks);

/* Diagnostics */
void        vqpu_print_spectrum(vcircuit_t *c, int ring_idx);
void        vqpu_print_circuit_stats(vcircuit_t *c);
void        vqpu_print_measures(vcircuit_t *c);

/* ─── Inline helpers ────────────────────────────────────────────────────── */

/* 2π modulus with C standard remainder() */
static inline double vqpu_mod2pi(double x) {
    return remainder(x, 2.0 * M_PI);
}

/* Initialize a single ring to ground state |0⟩ (all phases aligned) */
static inline void vqpu_ring_init(vring_t *r) {
    for (int i = 0; i < VQPU_N; i++) {
        r->th[i] = 0.0;
        r->om[i] = 0.0;
    }
    memset(r->c_real, 0, sizeof(r->c_real));
    memset(r->c_imag, 0, sizeof(r->c_imag));
    r->c_power  = 0.0;
    r->q        = 0;         /* q=0 at ground */
    r->measured = 0;
    r->visibility = 0.0;
}

/* Set ring to twisted state |q⟩: θ[j] = 2πqj/N */
static inline void vqpu_ring_set_twisted(vring_t *r, int q) {
    for (int i = 0; i < VQPU_N; i++)
        r->th[i] = fmod(2.0 * M_PI * q * i / VQPU_N, 2.0 * M_PI);
    for (int i = 0; i < VQPU_N; i++) r->om[i] = 0.0;
    r->q = q;
    r->measured = 0;
}

/* Copy ring state (save/restore for reversible computation) */
static inline void vqpu_ring_copy(vring_t *dst, const vring_t *src) {
    memcpy(dst->th, src->th, VQPU_N * sizeof(double));
    memcpy(dst->om, src->om, VQPU_N * sizeof(double));
    memcpy(dst->c_real, src->c_real, VQPU_N * sizeof(double));
    memcpy(dst->c_imag, src->c_imag, VQPU_N * sizeof(double));
    dst->c_power  = src->c_power;
    dst->q        = src->q;
    dst->measured = src->measured;
    dst->visibility = src->visibility;
}

/* Debug: print ring state summary */
static inline void vqpu_ring_debug(const vring_t *r, int idx) {
    printf("[ring %2d] q=%2d  power=%.4f  phase_range=[%.2f, %.2f]  lens=%s\n",
           idx, r->q, r->c_power,
           r->th[0], r->th[VQPU_N-1],
           (r->om[0] != 0.0) ? "YES" : "no");
}

#ifdef __cplusplus
}
#endif

#endif /* VQPU_H */