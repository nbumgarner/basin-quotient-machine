/* ===========================================================================
 * two_vqpu_validate.c — Two vQPU Event-Driven Coupling Validation
 *
 * Parameter-sweep
 * Sweeps key design parameters, outputs CSV for decision analysis.
 *
 * Compile: cc -O3 -std=c11 -lm two_vqpu_validate.c -o two_vqpu_validate
 * Run: ./two_vqpu_validate > results.csv
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

/* =========== CONFIGURATION =========== */
#define OSC_PER_RING  16
#define MICRO_RINGS   16
#define MACRO_RINGS   16
#define TOTAL_RINGS   (MICRO_RINGS * MACRO_RINGS)  /* 256 micro rings per vQPU */

#define DT              0.5
#define SETTLE_STEPS    120
#define SETTLE_CHUNKS   3

#define MAX_PULSES      10000
#define NUM_TRIALS      50

/* =========== SWEEP PARAMETERS =========== */
/* Pulse threshold sweep */
double pulse_thresholds[] = {0.1, 0.2, 0.3, 0.5, 0.7, 1.0};
int num_thresholds = 6;

/* Pulse strength sweep */
double pulse_strengths[] = {0.1, 0.3, 0.5, 0.7};
int num_strengths = 4;

/* Plasticity timescale sweep (chunks per update) */
int plasticity_timescales[] = {1, 3, 10, 30, 100};
int num_timescales = 5;

/* Routing modes */
typedef enum { ROUTE_MICRO=0, ROUTE_MACRO=1, ROUTE_SUPER=2, ROUTE_HYBRID=3 } route_mode_t;
route_mode_t route_modes[] = {ROUTE_MICRO, ROUTE_MACRO, ROUTE_SUPER, ROUTE_HYBRID};
int num_route_modes = 4;

/* Scale sweep */
int scale_vqpus[] = {2, 4, 8, 16};
int num_scales = 4;

/* =========== DATA STRUCTURES =========== */
typedef struct {
    double th[OSC_PER_RING];
    double om[OSC_PER_RING];
    double v[OSC_PER_RING];
    double last_q;
    int id;
    int vqpu_id;
} ring_t;

typedef struct {
    ring_t rings[TOTAL_RINGS];
    double lens_profiles[TOTAL_RINGS][OSC_PER_RING];
} vqpu_t;

typedef struct {
    int src_vqpu, src_ring;
    int dst_vqpu, dst_ring;
    double phase;
    double strength;
    int timestep;
} pulse_t;

typedef struct {
    pulse_t pulses[MAX_PULSES];
    int count;
} pulse_queue_t;

typedef struct {
    /* Experiment metadata */
    char experiment[64];
    /* Sweep parameters */
    double pulse_threshold;
    double pulse_strength;
    int plasticity_timescale;
    int route_mode;
    int num_vqpus;
    /* Measured metrics */
    double transfer_ratio;
    double pulse_rate;
    double latency_us;
    double bandwidth_mbps;
    double info_capacity;
    int harmonics_detected;
    double routing_accuracy;
    double convergence_epochs;
    double final_accuracy;
    double stability_score;
    double events_per_sec;
    double pulses_per_event;
    double analog_precision_bits;
} result_t;

/* =========== PHYSICS =========== */
static void deriv_ring(ring_t *r) {
    for (int i = 0; i < OSC_PER_RING; i++) {
        int ip = (i + 1) % OSC_PER_RING;
        int im = (i - 1 + OSC_PER_RING) % OSC_PER_RING;
        r->v[i] = r->om[i] + sin(r->th[ip] - r->th[i]) + sin(r->th[im] - r->th[i]);
    }
}

static void settle_ring(ring_t *r, int chunks) {
    for (int c = 0; c < chunks; c++) {
        for (int s = 0; s < SETTLE_STEPS; s++) {
            deriv_ring(r);
            for (int i = 0; i < OSC_PER_RING; i++) {
                r->th[i] += DT * r->v[i];
            }
        }
    }
}

static double measure_winding(ring_t *r) {
    double sum = 0;
    for (int i = 0; i < OSC_PER_RING; i++) {
        int ip = (i + 1) % OSC_PER_RING;
        double dth = r->th[ip] - r->th[i];
        while (dth > M_PI) dth -= 2 * M_PI;
        while (dth < -M_PI) dth += 2 * M_PI;
        sum += dth;
    }
    return sum / (2 * M_PI);  /* q ∈ [-3, 3] for N=16 */
}

static double harmonic_mag(ring_t *r, int k) {
    double re = 0, im = 0;
    for (int j = 0; j < OSC_PER_RING; j++) {
        double a = -2.0 * M_PI * k * j / OSC_PER_RING;
        re += r->th[j] * cos(a);
        im += r->th[j] * sin(a);
    }
    return sqrt(re*re + im*im) / OSC_PER_RING;
}

/* =========== LENS PROFILES =========== */
static void make_site_lens(double *lens, int site, double amp) {
    memset(lens, 0, OSC_PER_RING * sizeof(double));
    lens[site % OSC_PER_RING] = amp;
}

static void make_harmonic_lens(double *lens, int harmonic, double amp) {
    for (int i = 0; i < OSC_PER_RING; i++) {
        lens[i] = amp * cos(2.0 * M_PI * harmonic * i / OSC_PER_RING);
    }
}

static void make_gaussian_lens(double *lens, int center, double width, double amp) {
    for (int i = 0; i < OSC_PER_RING; i++) {
        double d = fmin(fabs(i - center), OSC_PER_RING - fabs(i - center));
        lens[i] = amp * exp(-(d*d) / (2 * width * width));
    }
}

static void apply_lens(ring_t *r, const double *lens) {
    memcpy(r->om, lens, OSC_PER_RING * sizeof(double));
}

/* =========== PULSE QUEUE =========== */
static void pulse_queue_init(pulse_queue_t *q) { q->count = 0; }

static void pulse_queue_push(pulse_queue_t *q, int src_vqpu, int src_ring,
                              int dst_vqpu, int dst_ring, double phase, double strength,
                              int timestep) {
    if (q->count >= MAX_PULSES) return;
    pulse_t *p = &q->pulses[q->count++];
    p->src_vqpu = src_vqpu; p->src_ring = src_ring;
    p->dst_vqpu = dst_vqpu; p->dst_ring = dst_ring;
    p->phase = phase; p->strength = strength; p->timestep = timestep;
}

static void pulse_queue_deliver(pulse_queue_t *q, vqpu_t *vqpus, int num_vqpus, int current_chunk) {
    for (int i = 0; i < q->count; i++) {
        pulse_t *p = &q->pulses[i];
        if (p->timestep != current_chunk) continue;
        if (p->dst_vqpu < 0 || p->dst_vqpu >= num_vqpus) continue;
        if (p->dst_ring < 0 || p->dst_ring >= TOTAL_RINGS) continue;
        ring_t *dst = &vqpus[p->dst_vqpu].rings[p->dst_ring];
        for (int j = 0; j < OSC_PER_RING; j++) {
            dst->th[j] += p->strength * sin(p->phase - dst->th[j]);
        }
    }
    int remaining = 0;
    for (int i = 0; i < q->count; i++) {
        if (q->pulses[i].timestep > current_chunk) {
            q->pulses[remaining++] = q->pulses[i];
        }
    }
    q->count = remaining;
}

/* =========== INITIALIZATION =========== */
static void init_vqpu(vqpu_t *vqpu, int vqpu_id) {
    for (int r = 0; r < TOTAL_RINGS; r++) {
        ring_t *ring = &vqpu->rings[r];
        ring->id = r; ring->vqpu_id = vqpu_id; ring->last_q = 0;
        memset(ring->th, 0, sizeof(ring->th));
        memset(ring->om, 0, sizeof(ring->om));
        memset(ring->v, 0, sizeof(ring->v));
        make_site_lens(vqpu->lens_profiles[r], 0, 0.5);
    }
}

static void encode_input(ring_t *r, int harmonic, double amplitude) {
    memset(r->th, 0, sizeof(r->th));
    for (int j = 0; j < OSC_PER_RING; j++) {
        r->th[j] = amplitude * cos(2.0 * M_PI * harmonic * j / OSC_PER_RING);
    }
}

/* =========== CSV OUTPUT =========== */
static void print_csv_header() {
    printf("experiment,pulse_threshold,pulse_strength,plasticity_timescale,route_mode,num_vqpus,"
           "transfer_ratio,pulse_rate,latency_us,bandwidth_mbps,info_capacity,harmonics_detected,"
           "routing_accuracy,convergence_epochs,final_accuracy,stability_score,"
           "events_per_sec,pulses_per_event,analog_precision_bits\n");
}

static void print_result(result_t *r) {
    printf("%s,%.3f,%.3f,%d,%d,%d,%.6f,%.3f,%.3f,%.3f,%.6f,%d,%.6f,%.3f,%.6f,%.6f,%.3f,%.3f,%.3f\n",
           r->experiment, r->pulse_threshold, r->pulse_strength, r->plasticity_timescale,
           r->route_mode, r->num_vqpus,
           r->transfer_ratio, r->pulse_rate, r->latency_us, r->bandwidth_mbps,
           r->info_capacity, r->harmonics_detected,
           r->routing_accuracy, r->convergence_epochs, r->final_accuracy, r->stability_score,
           r->events_per_sec, r->pulses_per_event, r->analog_precision_bits);
}

static void init_result(result_t *r, const char *exp_name) {
    memset(r, 0, sizeof(result_t));
    strncpy(r->experiment, exp_name, 63);
}

/* =========== EXPERIMENT 1: Pulse Trigger Sweep =========== */
void run_experiment_pulse_sweep() {
    vqpu_t vqpu0, vqpu1;
    pulse_queue_t pq;
    result_t res;

    for (int ti = 0; ti < num_thresholds; ti++) {
        for (int si = 0; si < num_strengths; si++) {
            double thresh = pulse_thresholds[ti];
            double strength = pulse_strengths[si];

            init_vqpu(&vqpu0, 0);
            init_vqpu(&vqpu1, 1);
            pulse_queue_init(&pq);

            ring_t *src = &vqpu0.rings[0];
            ring_t *dst = &vqpu1.rings[0];
            encode_input(src, 1, 2.0);
            apply_lens(src, vqpu0.lens_profiles[0]);
            apply_lens(dst, vqpu1.lens_profiles[0]);
            src->last_q = 0;

            int pulse_count = 0;
            int first_pulse_chunk = -1;

            for (int chunk = 0; chunk < SETTLE_CHUNKS; chunk++) {
                settle_ring(src, 1);
                double q = measure_winding(src);
                if (fabs(q - src->last_q) > thresh) {
                    pulse_queue_push(&pq, 0, 0, 1, 0, q, strength, chunk);
                    pulse_count++;
                    if (first_pulse_chunk == -1) first_pulse_chunk = chunk;
                }
                src->last_q = q;
                pulse_queue_deliver(&pq, (vqpu_t[]){vqpu0, vqpu1}, 2, chunk);
                settle_ring(dst, 1);
            }

            double src_k1 = harmonic_mag(src, 1);
            double dst_k1 = harmonic_mag(dst, 1);
            double transfer = dst_k1 / (src_k1 + 1e-10);

            double settle_time_us = SETTLE_CHUNKS * SETTLE_STEPS * DT * 1e-6;
            double pulse_rate = pulse_count / settle_time_us * 1e6;
            double latency = first_pulse_chunk * SETTLE_STEPS * DT * 1e-6;
            double bw = pulse_rate * log2(16) / 1e6;  /* Mbps */

            init_result(&res, "pulse_sweep");
            res.pulse_threshold = thresh;
            res.pulse_strength = strength;
            res.transfer_ratio = transfer;
            res.pulse_rate = pulse_rate;
            res.latency_us = latency;
            res.bandwidth_mbps = bw;
            res.events_per_sec = pulse_rate;
            res.pulses_per_event = (pulse_count > 0) ? 1.0 : 0.0;
            res.analog_precision_bits = log2(1.0 / (thresh * 0.1));  /* heuristic */
            print_result(&res);
        }
    }
}

/* =========== EXPERIMENT 2: Information Capacity Sweep =========== */
void run_experiment_info_capacity() {
    vqpu_t vqpu0, vqpu1;
    pulse_queue_t pq;
    result_t res;

    for (int ti = 0; ti < num_thresholds; ti++) {
        for (int si = 0; si < num_strengths; si++) {
            double thresh = pulse_thresholds[ti];
            double strength = pulse_strengths[si];

            double mi_sum[5] = {0};
            int valid[5] = {0};

            for (int trial = 0; trial < NUM_TRIALS; trial++) {
                double noise = 0.1 * ((double)rand() / RAND_MAX - 0.5);

                for (int p = 1; p <= 4; p++) {
                    init_vqpu(&vqpu0, 0);
                    init_vqpu(&vqpu1, 1);
                    pulse_queue_init(&pq);

                    ring_t *src = &vqpu0.rings[0];
                    ring_t *dst = &vqpu1.rings[0];
                    encode_input(src, p, 2.0 + noise);
                    apply_lens(src, vqpu0.lens_profiles[0]);
                    apply_lens(dst, vqpu1.lens_profiles[0]);
                    src->last_q = 0;

                    for (int chunk = 0; chunk < SETTLE_CHUNKS; chunk++) {
                        settle_ring(src, 1);
                        double q = measure_winding(src);
                        if (fabs(q - src->last_q) > thresh) {
                            pulse_queue_push(&pq, 0, 0, 1, 0, q, strength, chunk);
                        }
                        src->last_q = q;
                        pulse_queue_deliver(&pq, (vqpu_t[]){vqpu0, vqpu1}, 2, chunk);
                        settle_ring(dst, 1);
                    }

                    double out_mag = harmonic_mag(dst, p);
                    if (out_mag > 0.01) {
                        mi_sum[p] += out_mag;
                        valid[p]++;
                    }
                }
            }

            double total_mi = 0;
            int total_valid = 0;
            for (int p = 1; p <= 4; p++) {
                if (valid[p] > 0) {
                    total_mi += mi_sum[p] / valid[p];
                    total_valid++;
                }
            }
            double avg_mi = total_valid > 0 ? total_mi / total_valid : 0;

            init_result(&res, "info_capacity");
            res.pulse_threshold = thresh;
            res.pulse_strength = strength;
            res.info_capacity = avg_mi;
            res.harmonics_detected = total_valid;
            print_result(&res);
        }
    }
}

/* =========== EXPERIMENT 3: Plasticity Timescale =========== */
/* Simplified: measure how fast a lens adapts to match input statistics */
void run_experiment_plasticity() {
    result_t res;
    vqpu_t vqpu;
    ring_t *ring = &vqpu.rings[0];

    for (int ts_idx = 0; ts_idx < num_timescales; ts_idx++) {
        int timescale = plasticity_timescales[ts_idx];

        init_vqpu(&vqpu, 0);
        make_site_lens(vqpu.lens_profiles[0], 0, 0.5);

        /* Target: lens should adapt to favor harmonic 2 */
        double lens_coeff[8] = {0};  /* harmonic coefficients */
        double pre_trace[OSC_PER_RING] = {0};
        double post_trace[8] = {0};
        double alpha = 0.9;
        double eta = 0.01;
        double lambda = 0.001;

        int converged_epoch = -1;
        double prev_lens_norm = 0;

        for (int epoch = 0; epoch < 500; epoch++) {
            /* Present input: harmonic 2 with noise */
            encode_input(ring, 2, 2.0 + 0.1 * ((double)rand()/RAND_MAX - 0.5));
            apply_lens(ring, vqpu.lens_profiles[0]);
            settle_ring(ring, SETTLE_CHUNKS);

            /* Measure output */
            double out_mags[5];
            for (int k = 1; k <= 4; k++) out_mags[k] = harmonic_mag(ring, k);

            /* Simulate pulse received at site 0 (simplified) */
            int pulse_site = 0;
            pre_trace[pulse_site] = alpha * pre_trace[pulse_site] + (1-alpha) * 1.0;
            for (int k = 1; k <= 4; k++) {
                post_trace[k] = alpha * post_trace[k] + (1-alpha) * out_mags[k];
            }

            /* Plasticity update every `timescale` epochs */
            if (epoch % timescale == 0 && epoch > 0) {
                for (int i = 0; i < OSC_PER_RING; i++) {
                    double best_h = 0, best_val = 0;
                    for (int k = 1; k <= 4; k++) {
                        double corr = pre_trace[i] * post_trace[k];
                        if (corr > best_val) { best_val = corr; best_h = k; }
                    }
                    /* Update harmonic coefficient */
                    lens_coeff[(int)best_h] += eta * (best_val - lambda * lens_coeff[(int)best_h]);
                }
                /* Reconstruct lens from coefficients */
                for (int i = 0; i < OSC_PER_RING; i++) {
                    double val = 0;
                    for (int k = 1; k <= 4; k++) {
                        val += lens_coeff[k] * cos(2*M_PI*k*i/OSC_PER_RING);
                    }
                    vqpu.lens_profiles[0][i] = val;
                }
            }

            /* Check convergence: lens norm stabilizes */
            double lens_norm = 0;
            for (int i = 0; i < OSC_PER_RING; i++) lens_norm += vqpu.lens_profiles[0][i]*vqpu.lens_profiles[0][i];
            lens_norm = sqrt(lens_norm);

            if (epoch > 10 && fabs(lens_norm - prev_lens_norm) < 0.001 && converged_epoch == -1) {
                converged_epoch = epoch;
            }
            prev_lens_norm = lens_norm;
        }

        /* Final test: how well does lens select harmonic 2? */
        encode_input(ring, 2, 2.0);
        apply_lens(ring, vqpu.lens_profiles[0]);
        settle_ring(ring, SETTLE_CHUNKS);
        double sel_h2 = harmonic_mag(ring, 2);
        double sel_h1 = harmonic_mag(ring, 1);
        double selectivity = sel_h2 / (sel_h1 + sel_h2 + 1e-10);

        init_result(&res, "plasticity_timescale");
        res.plasticity_timescale = timescale;
        res.convergence_epochs = converged_epoch > 0 ? converged_epoch : 500;
        res.final_accuracy = selectivity;
        res.stability_score = (converged_epoch > 0 && converged_epoch < 400) ? 1.0 : 0.5;
        print_result(&res);
    }
}

/* =========== EXPERIMENT 4: Routing Mode Comparison =========== */
void run_experiment_routing() {
    result_t res;
    vqpu_t vqpus[4];
    pulse_queue_t pq;

    for (int rm = 0; rm < num_route_modes; rm++) {
        route_mode_t mode = route_modes[rm];
        int num_vqpu = 4;

        for (int i = 0; i < num_vqpu; i++) init_vqpu(&vqpus[i], i);
        pulse_queue_init(&pq);

        /* vQPU0: source pattern (harmonics 1,2 in rings 0,1) */
        encode_input(&vqpus[0].rings[0], 1, 2.0);
        encode_input(&vqpus[0].rings[1], 2, 2.0);
        apply_lens(&vqpus[0].rings[0], vqpus[0].lens_profiles[0]);
        apply_lens(&vqpus[0].rings[1], vqpus[0].lens_profiles[1]);
        vqpus[0].rings[0].last_q = 0;
        vqpus[0].rings[1].last_q = 0;

        /* vQPU1-3: targets with different lens configs */
        for (int v = 1; v < num_vqpu; v++) {
            for (int r = 0; r < 4; r++) {
                apply_lens(&vqpus[v].rings[r], vqpus[v].lens_profiles[r]);
                vqpus[v].rings[r].last_q = 0;
            }
        }

        int correct_routes = 0;
        int total_routes = 0;
        int pulse_count = 0;

        for (int chunk = 0; chunk < SETTLE_CHUNKS; chunk++) {
            for (int v = 0; v < num_vqpu; v++) {
                for (int r = 0; r < 4; r++) {
                    settle_ring(&vqpus[v].rings[r], 1);
                }
            }

            /* Emit pulses based on routing mode */
            for (int src_r = 0; src_r < 2; src_r++) {
                double q = measure_winding(&vqpus[0].rings[src_r]);
                if (fabs(q - vqpus[0].rings[src_r].last_q) > 0.5) {
                    int target_harmonic = (src_r == 0) ? 1 : 2;

                    for (int dst_v = 1; dst_v < num_vqpu; dst_v++) {
                        int dst_ring = -1;

                        switch (mode) {
                            case ROUTE_MICRO:
                                dst_ring = target_harmonic - 1;  /* micro ring per harmonic */
                                break;
                            case ROUTE_MACRO:
                                dst_ring = 4 + (target_harmonic - 1);  /* macro ring offset */
                                break;
                            case ROUTE_SUPER:
                                dst_ring = 8;  /* super ring aggregator */
                                break;
                            case ROUTE_HYBRID:
                                dst_ring = (chunk == 0) ? (target_harmonic - 1) : (4 + target_harmonic - 1);
                                break;
                        }

                        if (dst_ring >= 0 && dst_ring < TOTAL_RINGS) {
                            pulse_queue_push(&pq, 0, src_r, dst_v, dst_ring, q, 0.3, chunk);
                            pulse_count++;
                            total_routes++;
                            /* Check if routed to correct harmonic-preferring ring */
                            double dst_k1 = harmonic_mag(&vqpus[dst_v].rings[dst_ring], target_harmonic);
                            double dst_other = 0;
                            for (int k = 1; k <= 4; k++) if (k != target_harmonic)
                                dst_other += harmonic_mag(&vqpus[dst_v].rings[dst_ring], k);
                            if (dst_k1 > dst_other) correct_routes++;
                        }
                    }
                }
                vqpus[0].rings[src_r].last_q = q;
            }

            pulse_queue_deliver(&pq, vqpus, num_vqpu, chunk);
        }

        double accuracy = total_routes > 0 ? (double)correct_routes / total_routes : 0;

        init_result(&res, "routing_mode");
        res.route_mode = rm;
        res.num_vqpus = num_vqpu;
        res.routing_accuracy = accuracy;
        res.pulse_rate = pulse_count / (SETTLE_CHUNKS * SETTLE_STEPS * DT * 1e-6);
        print_result(&res);
    }
}

/* =========== EXPERIMENT 5: Scale Scaling =========== */
void run_experiment_scale() {
    result_t res;

    for (int s = 0; s < num_scales; s++) {
        int N = scale_vqpus[s];
        vqpu_t *vqpus = calloc(N, sizeof(vqpu_t));
        pulse_queue_t pq;
        pulse_queue_init(&pq);

        for (int i = 0; i < N; i++) init_vqpu(&vqpus[i], i);

        /* Distributed task: each vQPU encodes a piece, all must combine */
        int pulses = 0;
        int delivered = 0;

        for (int chunk = 0; chunk < SETTLE_CHUNKS; chunk++) {
            for (int v = 0; v < N; v++) {
                for (int r = 0; r < 4; r++) {
                    settle_ring(&vqpus[v].rings[r], 1);
                }
            }

            /* Ring 0 of each vQPU sends to ring 0 of next vQPU (ring) */
            for (int v = 0; v < N; v++) {
                double q = measure_winding(&vqpus[v].rings[0]);
                if (fabs(q - vqpus[v].rings[0].last_q) > 0.5) {
                    int dst_v = (v + 1) % N;
                    pulse_queue_push(&pq, v, 0, dst_v, 0, q, 0.3, chunk);
                    pulses++;
                }
                vqpus[v].rings[0].last_q = q;
            }

            pulse_queue_deliver(&pq, vqpus, N, chunk);
        }

        /* Measure: how many vQPUs received the signal? */
        int active = 0;
        for (int v = 0; v < N; v++) {
            if (harmonic_mag(&vqpus[v].rings[0], 1) > 0.05) active++;
        }

        double speedup = (double)active / (N > 1 ? N-1 : 1);  /* ideal = N */
        double comms_overhead = pulses * 32.0 / 1e6;  /* bytes per pulse */

        init_result(&res, "scale_scaling");
        res.num_vqpus = N;
        res.routing_accuracy = (double)active / N;
        res.convergence_epochs = speedup;
        res.events_per_sec = pulses / (SETTLE_CHUNKS * SETTLE_STEPS * DT * 1e-6);
        print_result(&res);

        free(vqpus);
    }
}

/* =========== EXPERIMENT 6: Hardware Projection =========== */
void run_experiment_hw_projection() {
    result_t res;

    /* From simulation: measure event rates at different thresholds */
    for (int ti = 0; ti < num_thresholds; ti++) {
        double thresh = pulse_thresholds[ti];
        vqpu_t vqpu0, vqpu1;
        init_vqpu(&vqpu0, 0); init_vqpu(&vqpu1, 1);
        pulse_queue_t pq; pulse_queue_init(&pq);

        ring_t *src = &vqpu0.rings[0];
        ring_t *dst = &vqpu1.rings[0];
        encode_input(src, 1, 2.0);
        apply_lens(src, vqpu0.lens_profiles[0]);
        apply_lens(dst, vqpu1.lens_profiles[0]);
        src->last_q = 0;

        int pulses = 0;
        for (int chunk = 0; chunk < SETTLE_CHUNKS; chunk++) {
            settle_ring(src, 1);
            double q = measure_winding(src);
            if (fabs(q - src->last_q) > thresh) {
                pulse_queue_push(&pq, 0, 0, 1, 0, q, 0.3, chunk);
                pulses++;
            }
            src->last_q = q;
            pulse_queue_deliver(&pq, (vqpu_t[]){vqpu0, vqpu1}, 2, chunk);
            settle_ring(dst, 1);
        }

        double settle_s = SETTLE_CHUNKS * SETTLE_STEPS * DT * 1e-6;
        double event_rate = pulses / settle_s;
        double pulse_bits = log2(16);  /* 16 oscillator phases */
        double analog_snr = 1.0 / thresh;  /* heuristic */
        double precision_bits = log2(analog_snr);

        /* Project to 16-vQPU cluster */
        double cluster_events = event_rate * 16 * 256;  /* 256 rings per vQPU active */
        int fpga_pins = (int)ceil(cluster_events * 4 / 1e6);  /* 4 bytes/event, 1M events/s per pin */
        int bram_kb = (int)(cluster_events * 8 / 1024);  /* 8 bytes/event buffer */

        init_result(&res, "hw_projection");
        res.pulse_threshold = thresh;
        res.events_per_sec = cluster_events;
        res.pulses_per_event = 1.0;
        res.analog_precision_bits = precision_bits;
        res.bandwidth_mbps = cluster_events * pulse_bits / 1e6;
        res.latency_us = settle_s * 1e6 / 16;  /* placeholder: 16 vQPUs */
        print_result(&res);
    }
}

/* =========== MAIN =========== */
int main() {
    print_csv_header();

    srand(42);

    fprintf(stderr, "Running Experiment 1: Pulse Trigger Sweep...\n");
    run_experiment_pulse_sweep();

    fprintf(stderr, "Running Experiment 2: Information Capacity...\n");
    run_experiment_info_capacity();

    fprintf(stderr, "Running Experiment 3: Plasticity Timescale...\n");
    run_experiment_plasticity();

    fprintf(stderr, "Running Experiment 4: Routing Mode Comparison...\n");
    run_experiment_routing();

    fprintf(stderr, "Running Experiment 5: Scale Scaling...\n");
    run_experiment_scale();

    fprintf(stderr, "Running Experiment 6: Hardware Projection...\n");
    run_experiment_hw_projection();

    fprintf(stderr, "Done.\n");
    return 0;
}