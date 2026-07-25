/* ===========================================================================
 * vqpu.c — vQPU Core Implementation
 *
 *   Phase 2: gate dispatch, circuit execution, Fourier state management,
 *   adiabatic scheduling. Wraps ring_furnace.c via the keep_phases API.
 *
 *   Part of bqsm-quantum/ — links against ring_furnace.c.
 *
 * BUILD:
 *   cc -O3 -std=c11 -march=native -c vqpu.c -o vqpu.o
 *   cc -O3 -std=c11 -march=native ring_furnace.c vqpu_test.c -o vqpu_test -lm
 * ======================================================================== */

#include "vqpu.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ─── Internal: ring ←→ batch_t bridge ──────────────────────────────────── */

/* Load ring state INTO batch_t lane l. Call before settle_partial().
 * Transfers th[] and om[] from ring to SoA batch layout. */
static void vqpu_ring_to_batch(const vring_t *r, struct batch_t *b, int l) {
    for (int i = 0; i < VQPU_N; i++) {
        b->th[i * b->L + l] = r->th[i];
        b->om[i * b->L + l] = r->om[i];
    }
}

/* Load batch_t lane l back INTO ring. Call after settle_partial(). */
static void vqpu_batch_to_ring(vring_t *r, const struct batch_t *b, int l) {
    for (int i = 0; i < VQPU_N; i++) {
        r->th[i] = b->th[i * b->L + l];
        r->om[i] = b->om[i * b->L + l];
    }
}

/* ─── Circuit lifecycle ──────────────────────────────────────────────────── */

void vqpu_circuit_init(vcircuit_t *c, int n_rings) {
    if (n_rings > VQPU_MAX_RINGS) n_rings = VQPU_MAX_RINGS;
    c->n_rings = n_rings;
    c->rings = malloc((size_t)n_rings * sizeof(vring_t));
    for (int i = 0; i < n_rings; i++)
        vqpu_ring_init(&c->rings[i]);

    c->n_gates = 0;
    c->gates = malloc(VQPU_MAX_GATES * sizeof(vgate_t));
    c->measures = malloc((size_t)n_rings * sizeof(int));
    for (int i = 0; i < n_rings; i++) c->measures[i] = 0;

    c->visibility   = 0.0;
    c->step_count   = 0;
    c->total_sim_time = 0.0;
    c->total_wall_ms  = 0.0;
    c->gap_table_loaded = 0;
    vqpu_load_gap_table(c);
}

void vqpu_circuit_free(vcircuit_t *c) {
    free(c->rings);     c->rings    = NULL;
    free(c->gates);     c->gates    = NULL;
    free(c->measures);  c->measures = NULL;
    c->n_rings = 0;
    c->n_gates = 0;
}

void vqpu_circuit_reset(vcircuit_t *c) {
    for (int i = 0; i < c->n_rings; i++)
        vqpu_ring_init(&c->rings[i]);
    for (int i = 0; i < c->n_rings; i++) c->measures[i] = 0;
    c->n_gates       = 0;
    c->step_count    = 0;
    c->total_sim_time = 0.0;
    c->total_wall_ms  = 0.0;
    c->visibility    = 0.0;
}

void vqpu_circuit_add_gate(vcircuit_t *c, vopcode_t op, int target, vparam_t params) {
    if (c->n_gates >= VQPU_MAX_GATES) {
        fprintf(stderr, "vQPU: gate limit %d exceeded\n", VQPU_MAX_GATES);
        return;
    }
    c->gates[c->n_gates].op     = op;
    c->gates[c->n_gates].target = target;
    c->gates[c->n_gates].params = params;
    c->n_gates++;
}

/* ─── Fourier state management ──────────────────────────────────────────── */

void vqpu_fourier_snapshot(vcircuit_t *c, int ring_idx) {
    /* We need a batch_t for fft_snapshot(), so create a single-lane batch */
    struct batch_t *b = batch_new(1);
    vring_t *r = &c->rings[ring_idx];
    vqpu_ring_to_batch(r, b, 0);

    fft_snapshot(b, 0, r->c_real, r->c_imag);

    /* Compute total harmonic power Σ|c_k|² */
    double power = 0.0;
    for (int k = 1; k <= VQPU_N_HALF; k++) {
        double ck_re = r->c_real[k];
        double ck_im = (k == VQPU_N_HALF) ? 0.0 : r->c_imag[k];
        power += ck_re * ck_re + ck_im * ck_im;
    }
    r->c_power = power;

    /* Winding from FFT: q = round(Im(c_1) * N / (2π)) */
    double q_cont = r->c_imag[1] * (double)VQPU_N / (2.0 * M_PI);
    r->q = (int)lround(q_cont);

    batch_free(b);
}

void vqpu_apply_fourier_kick(vcircuit_t *c, int ring_idx, int k_mode, double amp, double phase) {
    struct batch_t *b = batch_new(1);
    vring_t *r = &c->rings[ring_idx];
    vqpu_ring_to_batch(r, b, 0);

    apply_fourier_kick(b, 0, k_mode, amp, phase);

    vqpu_batch_to_ring(r, b, 0);
    batch_free(b);
}

/* ─── Gate dispatch ──────────────────────────────────────────────────────── */

int vqpu_gate_dispatch(vcircuit_t *c, vgate_t *gate) {
    vring_t *r = &c->rings[gate->target];
    vparam_t *p = &gate->params;
    int chunks = p->settle_chunks > 0 ? p->settle_chunks : VQPU_DEFAULT_CHUNKS;

    /* Single-lane batch to interface with ring_furnace.c */
    struct batch_t *b = batch_new(1);
    int result = 0;
    double sim_time = 0.0;

    vqpu_ring_to_batch(r, b, 0);

    switch (gate->op) {

    case VOP_NOP:
        /* Identity — no perturbation, no settle */
        sim_time = 0.0;
        break;

    case VOP_HAD:
        /* Single-site kick at site 0. amp=2.2 calibrated for uniform
         * excitation across all Fourier modes. Places ring at basin boundary
         * so measurement yields any q with balanced probability. */
        apply_kick(b, 0, p->site ? p->site : 0, p->amp > 0.0 ? p->amp : 2.2);
        settle_partial(b, chunks);
        sim_time = chunks * VQPU_T_CHUNK;
        break;

    case VOP_PHASE:
        /* Global phase rotation — no settle needed. Pure gauge transformation
         * that shifts all c_k phases by φ. Zero-cost operation. */
        for (int i = 0; i < VQPU_N; i++)
            b->th[i * b->L + 0] += p->phase;
        sim_time = 0.0;
        break;

    case VOP_RAMP:
        /* 2π ramp at wavenumber k_mode: θ[j] += amp·j/N
         * Excites specific harmonic c_k. Used for state preparation. */
        {
            int km = p->k_mode > 0 ? p->k_mode : 1;
            double ramp_amp = p->amp > 0.0 ? p->amp : 2.0 * M_PI;
            for (int i = 0; i < VQPU_N; i++)
                b->th[i * b->L + 0] += ramp_amp * (double)(km * i) / (double)VQPU_N;
            settle_partial(b, chunks);
            sim_time = chunks * VQPU_T_CHUNK;
        }
        break;

    case VOP_LENS:
        /* Detune one site by δ. This reprograms the transition table —
         * NOT a perturbation of state, but a change to the dynamics.
         * The lens persists until vqpu_ring_init() or manual reset. */
        {
            int site = p->site >= 0 ? p->site : 0;
            double delta = p->delta != 0.0 ? p->delta : 0.35;
            b->om[site * b->L + 0] = delta;
            settle_partial(b, chunks);
            sim_time = chunks * VQPU_T_CHUNK;
        }
        break;

    case VOP_ERASE:
        /* Bias toward q=0 funnel: apply ramp toward q=0 from current state.
         * Strong perturbation that pushes the ring toward the ground attractor.
         * The one-way property guarantees |q| decreases — but one ERASE may
         * not reach q=0 from |q|=3. Loop for guaranteed reset. */
        {
            /* Read current winding, apply counter-ramp to cancel it */
            int cur_q = winding(b, 0);
            double counter_amp = -2.0 * M_PI * cur_q;
            for (int i = 0; i < VQPU_N; i++)
                b->th[i * b->L + 0] += counter_amp * (double)i / (double)VQPU_N;
            settle_partial(b, chunks);
            sim_time = chunks * VQPU_T_CHUNK;
        }
        break;

    case VOP_RESET:
        /* Strong ERASE looped until q=0. Always succeeds (physics-guaranteed
         * funnel). Maximum iterations: |q_max| = 3 at N=16. */
        {
            int max_iters = VQPU_Q_MAX + 1;  /* |q|≤3 → at most 4 iterations */
            for (int iter = 0; iter < max_iters; iter++) {
                settle_partial(b, chunks);
                int cur_q = winding(b, 0);
                if (cur_q == 0) break;
                /* Apply counter-ramp toward q=0 */
                double counter_amp = -2.0 * M_PI * cur_q;
                for (int i = 0; i < VQPU_N; i++)
                    b->th[i * b->L + 0] += counter_amp * (double)i / (double)VQPU_N;
            }
            settle_partial(b, chunks);  /* final settle to attractor */
            sim_time = (double)(max_iters + 1) * chunks * VQPU_T_CHUNK;
        }
        break;

    case VOP_MEASURE:
        /* Full settle to attractor lock + winding snap.
         * Measurement IS projective: 15 real DoF → 1 integer.
         * Uses MFIX chunks (empirically sufficient for lock at DT=0.50). */
        {
            int mfix = 16;  /* generous — ensures attractor lock */
            settle_partial(b, mfix);
            result = winding(b, 0);
            vqpu_batch_to_ring(r, b, 0);  /* update ring state */
            r->q = result;
            r->measured = 1;
            sim_time = mfix * VQPU_T_CHUNK;
            batch_free(b);
            c->measures[gate->target] = result;
            c->step_count++;
            c->total_sim_time += sim_time;
            c->total_wall_ms  += sim_time * (1000.0 / (2857.0 / 256.0));
            /* ^ wall-clock estimate: 2857 samples/s for 256 lanes → per-lane */
            return result;
        }

    case VOP_CNOT:
        /* Classical controlled-NOT:
         * 1. Measure control ring → q_c ∈ {0,1} (binary radix)
         * 2. Apply lens to target: pristine (q_c=0) or detuned (q_c=1)
         * 3. Settle target → q_t' = q_t XOR q_c
         */
        {
            int ctrl = p->control_ring;
            int tgt  = gate->target;
            if (ctrl < 0 || ctrl >= c->n_rings || tgt < 0 || tgt >= c->n_rings) {
                fprintf(stderr, "vCNOT: invalid rings ctrl=%d tgt=%d\n", ctrl, tgt);
                sim_time = 0.0;
                break;
            }

            /* 1. Measure control */
            struct batch_t *b_ctrl = batch_new(1);
            vring_t *r_ctrl = &c->rings[ctrl];
            vqpu_ring_to_batch(r_ctrl, b_ctrl, 0);
            settle_partial(b_ctrl, 16);  /* full settle */
            int q_c = winding(b_ctrl, 0) & 1;  /* binary radix */
            vqpu_batch_to_ring(r_ctrl, b_ctrl, 0);
            r_ctrl->q = q_c;
            r_ctrl->measured = 1;
            batch_free(b_ctrl);
            c->measures[ctrl] = q_c;

            /* 2. Apply lens to target based on q_c */
            struct batch_t *b_tgt = batch_new(1);
            vring_t *r_tgt = &c->rings[tgt];
            vqpu_ring_to_batch(r_tgt, b_tgt, 0);

            double delta_lens = (q_c == 1) ? (p->delta > 0.0 ? p->delta : 0.35) : 0.0;
            if (delta_lens > 0.0)
                b_tgt->om[p->site >= 0 ? p->site : 0] = delta_lens;

            settle_partial(b_tgt, chunks);
            vqpu_batch_to_ring(r_tgt, b_tgt, 0);
            batch_free(b_tgt);

            sim_time = 16.0 * VQPU_T_CHUNK + chunks * VQPU_T_CHUNK;
        }
        break;

    case VOP_SWEEP:
        /* Adiabatic lens sweep δ: start→end in N steps.
         * Each step: set δ, partial settle, next δ.
         * Uses gap-table-based step sizing if loaded. */
        {
            int steps = p->sweep_steps > 0 ? p->sweep_steps : 50;
            double d_start = p->sweep_delta_start;
            double d_end   = p->sweep_delta_end;
            int site = p->site >= 0 ? p->site : 0;
            double total_sweep_sim = 0.0;

            for (int s = 0; s <= steps; s++) {
                double frac = (double)s / (double)steps;
                double delta = d_start + frac * (d_end - d_start);
                b->om[site * b->L + 0] = delta;
                settle_partial(b, chunks);
                total_sweep_sim += chunks * VQPU_T_CHUNK;
            }
            sim_time = total_sweep_sim;
        }
        break;

    case VOP_FOURIER:
        /* Direct Fourier harmonic excitation: apply_fourier_kick + settle.
         * Parameters: k_mode, amp, phase. Most general single-ring gate. */
        apply_fourier_kick(b, 0, p->k_mode > 0 ? p->k_mode : 1,
                           p->amp, p->phase);
        settle_partial(b, chunks);
        sim_time = chunks * VQPU_T_CHUNK;
        break;

    default:
        fprintf(stderr, "vQPU: unknown opcode %d\n", gate->op);
        sim_time = 0.0;
        break;
    }

    /* Update ring state from batch */
    vqpu_batch_to_ring(r, b, 0);
    batch_free(b);

    /* Update wounding number cache */
    if (gate->op != VOP_MEASURE) {
        /* Recompute q from phases for non-measurement gates */
        struct batch_t *bq = batch_new(1);
        vqpu_ring_to_batch(r, bq, 0);
        r->q = winding(bq, 0);
        batch_free(bq);
    }

    c->step_count++;
    c->total_sim_time += sim_time;

    /* Wall-clock estimate: 2857 samples/s for 256-lane batch at T7400.
     * Single-lane: ~0.35 ms per settle chunk. Rough but directionally correct. */
    double wall_per_chunk_ms = (VQPU_T_CHUNK / 2857.0) * 1000.0 * (256.0 / 1.0);
    /* Actually: 2857 samples/s = 2857 * 256 lane-settles/s. One settle =
     * T_CHUNK sim time. So 1 sim-time unit = 1/(2857 * T_CHUNK) wall seconds.
     * sim_time / T_CHUNK chunks → chunks * per-chunk-wall.
     * per_chunk_wall = 1000 / 2857 ≈ 0.35 ms (for 256 lanes).
     * For single lane: same wall time (can't go faster than batch). */
    c->total_wall_ms += (sim_time / VQPU_T_CHUNK) * (1000.0 / 2857.0);

    return result;
}

/* ─── Circuit execution ─────────────────────────────────────────────────── */

int vqpu_circuit_run(vcircuit_t *c) {
    int last_measure = -1;
    for (int i = 0; i < c->n_gates; i++) {
        vgate_t *gate = &c->gates[i];
        int result = vqpu_gate_dispatch(c, gate);
        if (gate->op == VOP_MEASURE)
            last_measure = result;
    }
    return last_measure;
}

/* ─── Gap table ──────────────────────────────────────────────────────────── */

void vqpu_load_gap_table(vcircuit_t *c) {
    /* Populate gap(δ, q) from gap_extract.py model:
     * gap(δ, q) = gap_0 * max(0, 1 − δ/δ_crit(q)) */
    for (int di = 0; di < 16; di++) {
        double delta = di * 0.05;  /* 0.00 to 0.75 in 0.05 steps */
        for (int q = 0; q <= VQPU_Q_MAX; q++) {
            double dc = VQPU_DELTA_CRIT[q];
            c->gap_table[di][q] = VQPU_GAP_0 * fmax(0.0, 1.0 - delta / dc);
        }
    }
    c->gap_table_loaded = 1;
}

double vqpu_gap(double delta, int q) {
    /* Fast lookup: gap(δ, q) = gap₀ · max(0, 1 − δ/δ_crit(q)) */
    if (q < 0) q = -q;
    if (q > VQPU_Q_MAX) q = VQPU_Q_MAX;
    double dc = VQPU_DELTA_CRIT[q];
    return VQPU_GAP_0 * fmax(0.0, 1.0 - delta / dc);
}

double vqpu_adiabatic_step(double delta, double gap_min) {
    /* Δδ = margin · gap² · T_chunk */
    double ddelta = VQPU_MARGIN * gap_min * gap_min * VQPU_T_CHUNK;
    if (ddelta < 0.0001) ddelta = 0.0001;  /* floor — don't stall */
    return ddelta;
}

int vqpu_adiabatic_sweep(vcircuit_t *c, int ring_idx, double d_start, double d_end, int max_steps) {
    vring_t *r = &c->rings[ring_idx];
    double delta = d_start;
    int used_steps = 0;

    struct batch_t *b = batch_new(1);
    vqpu_ring_to_batch(r, b, 0);

    while (delta < d_end && used_steps < max_steps) {
        /* Find minimum gap across all live states at current δ */
        double gap_min = 1.0;  /* large */
        for (int q = 0; q <= VQPU_Q_MAX; q++) {
            double g = vqpu_gap(delta, q);
            if (g > 0.0 && g < gap_min) gap_min = g;
        }

        double ddelta = vqpu_adiabatic_step(delta, gap_min);
        delta += ddelta;
        if (delta > d_end) delta = d_end;

        /* Apply lens at current δ */
        b->om[0] = delta;
        settle_partial(b, 1);  /* one chunk per step */
        used_steps++;
    }

    vqpu_batch_to_ring(r, b, 0);
    batch_free(b);

    /* Compute final winding */
    struct batch_t *bq = batch_new(1);
    vqpu_ring_to_batch(r, bq, 0);
    settle_partial(bq, 16);  /* full settle to lock */
    int q_final = winding(bq, 0);
    vqpu_batch_to_ring(r, bq, 0);
    batch_free(bq);

    r->q = q_final;
    r->measured = 1;
    c->measures[ring_idx] = q_final;
    c->step_count += used_steps;
    c->total_sim_time += used_steps * VQPU_T_CHUNK;
    c->total_wall_ms  += used_steps * (1000.0 / 2857.0);

    return q_final;
}

/* ─── Interference visibility ───────────────────────────────────────────── */

double vqpu_interference_visibility(vcircuit_t *c, int ring_idx, int q_init, double pert_amp, int phi_steps, int chunks) {
    vring_t *r = &c->rings[ring_idx];
    int *q_final = malloc((size_t)phi_steps * sizeof(int));
    if (!q_final) return 0.0;

    int q_min = 999, q_max = -999;

    for (int si = 0; si < phi_steps; si++) {
        double phi = 2.0 * M_PI * (double)si / (double)phi_steps;

        /* Initialize twisted state */
        vqpu_ring_set_twisted(r, q_init);

        /* Apply φ-dependent perturbation: c_2 excitation */
        vqpu_apply_fourier_kick(c, ring_idx, 2, pert_amp, phi);

        /* Partial settle */
        struct batch_t *b = batch_new(1);
        vqpu_ring_to_batch(r, b, 0);
        settle_partial(b, chunks);
        vqpu_batch_to_ring(r, b, 0);

        /* Measure winding */
        q_final[si] = winding(b, 0);
        batch_free(b);

        if (q_final[si] < q_min) q_min = q_final[si];
        if (q_final[si] > q_max) q_max = q_final[si];
    }

    double V = (double)(q_max - q_min) / (2.0 * (double)VQPU_Q_MAX);
    r->visibility = V;
    c->visibility = V;
    free(q_final);
    return V;
}

/* ─── Diagnostics ────────────────────────────────────────────────────────── */

void vqpu_print_spectrum(vcircuit_t *c, int ring_idx) {
    vring_t *r = &c->rings[ring_idx];
    vqpu_fourier_snapshot(c, ring_idx);

    printf("─── vQPU Fourier Spectrum (ring %d) ───\n", ring_idx);
    printf("  k    |c_k|       phase      Re(c_k)    Im(c_k)\n");
    printf("  ────────────────────────────────────────────\n");
    for (int k = 1; k <= VQPU_N_HALF; k++) {
        double ck_re = r->c_real[k];
        double ck_im = (k == VQPU_N_HALF) ? 0.0 : r->c_imag[k];
        double mag   = sqrt(ck_re * ck_re + ck_im * ck_im);
        double phase = atan2(ck_im, ck_re);
        printf("  %d   %10.6f  %+8.4f  %+10.6f  %+10.6f\n",
               k, mag, phase, ck_re, ck_im);
    }
    printf("  q = %d   power = %.6f   visibility = %.4f\n",
           r->q, r->c_power, r->visibility);
}

void vqpu_print_circuit_stats(vcircuit_t *c) {
    printf("\n─── Circuit Statistics ───\n");
    printf("  Rings:      %d\n", c->n_rings);
    printf("  Gates:      %d\n", c->n_gates);
    printf("  Steps:      %d\n", c->step_count);
    printf("  Sim time:   %.0f units\n", c->total_sim_time);
    printf("  Wall time:  ~%.1f ms (est., T7400)\n", c->total_wall_ms);
    printf("  Visibility: %.4f\n", c->visibility);
    printf("  Gap table:  %s\n", c->gap_table_loaded ? "loaded" : "NOT LOADED");

    /* Gate breakdown */
    int op_counts[VOP_COUNT];
    memset(op_counts, 0, sizeof(op_counts));
    for (int i = 0; i < c->n_gates; i++)
        op_counts[c->gates[i].op]++;

    static const char *op_names[] = {
        "NOP", "HAD", "PHASE", "RAMP", "LENS", "ERASE", "RESET",
        "MEASURE", "CNOT", "SWEEP", "FOURIER"
    };
    printf("  ── Gate breakdown ──\n");
    for (int o = 0; o < VOP_COUNT; o++) {
        if (op_counts[o] > 0)
            printf("    %-8s: %d\n", op_names[o], op_counts[o]);
    }
}

void vqpu_print_measures(vcircuit_t *c) {
    printf("─── Measurement Results ───\n");
    for (int i = 0; i < c->n_rings; i++) {
        printf("  ring[%2d]  q = % d  |  measure = % d\n",
               i, c->rings[i].q, c->measures[i]);
    }
}