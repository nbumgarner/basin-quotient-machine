/* Quick test: macro aggregation + feature routing with pulse triggering */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define OSC_PER_MICRO   16
#define MICRO_PER_MACRO 16
#define MACRO_PER_SUPER 16
#define DT              0.5
#define SETTLE_STEPS    120
#define SETTLE_CHUNKS   5
#define MAX_VQPUS       16
#define MAX_PULSES      50000

#define PULSE_STRENGTH  0.5
#define TRACE_DECAY     0.95

typedef struct {
    double th[OSC_PER_MICRO];
    double om[OSC_PER_MICRO];
    double v[OSC_PER_MICRO];
    double lens[OSC_PER_MICRO];
    double last_q;
    int id;
    int macro_id;
    double lens_coeff[8];
    double pre_trace[OSC_PER_MICRO];
    double post_trace[8];
    int best_harmonic;
    double lens_energy;
    double feature_weight[MICRO_PER_MACRO];
    double feature_weight_decay;
} micro_ring_t;

typedef struct {
    micro_ring_t micro[MICRO_PER_MACRO];
    double inter_coupling;
    double attention_weights[MICRO_PER_MACRO];
    int id;
    double super_feedback_weights[MACRO_PER_SUPER];
    double super_feedback_phase[MACRO_PER_SUPER];
    int super_feedback_harmonic[MACRO_PER_SUPER];
    double feature_weight[MICRO_PER_MACRO];
    double feature_weight_decay;
} macro_ring_t;

typedef struct {
    macro_ring_t macro[MACRO_PER_SUPER];
    double global_coupling;
    double routing_weights[MAX_VQPUS];
    int id;
} vqpu_t;

typedef struct {
    int src_vqpu, src_level, src_idx;
    int dst_vqpu, dst_level, dst_idx;
    double phase;
    double strength;
    int harmonic;
    int timestep;
} pulse_t;

typedef struct {
    pulse_t pulses[MAX_PULSES];
    int count;
} pulse_queue_t;

static void init_micro(vqpu_t *vqpu, int id) {
    vqpu->id = id;
    vqpu->global_coupling = 0.1;
    for (int i = 0; i < MAX_VQPUS; i++) {
        vqpu->routing_weights[i] = 0.1;
    }
    for (int m = 0; m < MACRO_PER_SUPER; m++) {
        vqpu->macro[m].id = m;
        vqpu->macro[m].inter_coupling = 0.05;
        for (int u = 0; u < MICRO_PER_MACRO; u++) {
            vqpu->macro[m].micro[u].id = m * MICRO_PER_MACRO + u;
            vqpu->macro[m].micro[u].macro_id = m;
            vqpu->macro[m].micro[u].feature_weight_decay = 0.9;
            for (int h = 0; h < 8; h++) {
                vqpu->macro[m].micro[u].lens_coeff[h] = 0.0;
                vqpu->macro[m].micro[u].post_trace[h] = 0.0;
            }
            for (int i = 0; i < OSC_PER_MICRO; i++) {
                vqpu->macro[m].micro[u].pre_trace[i] = 0.0;
                vqpu->macro[m].micro[u].lens[i] = 0.0;
                vqpu->macro[m].micro[u].th[i] = 0.0;
                vqpu->macro[m].micro[u].om[i] = 0.0;
                vqpu->macro[m].micro[u].v[i] = 0.0;
            }
            vqpu->macro[m].micro[u].best_harmonic = 1;
            vqpu->macro[m].micro[u].lens_energy = 0.0;
        }
        for (int u = 0; u < MICRO_PER_MACRO; u++) {
            vqpu->macro[m].feature_weight[u] = 0.0;
            vqpu->macro[m].feature_weight_decay = 0.9;
            vqpu->macro[m].attention_weights[u] = 0.05;
        }
    }
}

static void deriv_micro(micro_ring_t *r) {
    for (int i = 0; i < OSC_PER_MICRO; i++) {
        int ip = (i + 1) % OSC_PER_MICRO;
        int im = (i - 1 + OSC_PER_MICRO) % OSC_PER_MICRO;
        r->v[i] = r->om[i] + sin(r->th[ip] - r->th[i]) + sin(r->th[im] - r->th[i]);
    }
}

static void settle_micro(micro_ring_t *r, int chunks) {
    for (int c = 0; c < chunks; c++) {
        for (int s = 0; s < SETTLE_STEPS; s++) {
            deriv_micro(r);
            for (int i = 0; i < OSC_PER_MICRO; i++) {
                r->th[i] += DT * r->v[i];
            }
        }
    }
}

static double measure_winding(micro_ring_t *r) {
    double sum = 0;
    for (int i = 0; i < OSC_PER_MICRO; i++) {
        int ip = (i + 1) % OSC_PER_MICRO;
        double dth = r->th[ip] - r->th[i];
        while (dth > M_PI) dth -= 2 * M_PI;
        while (dth < -M_PI) dth += 2 * M_PI;
        sum += dth;
    }
    return sum / (2 * M_PI);
}

static double harmonic_mag(micro_ring_t *r, int k) {
    double re = 0, im = 0;
    for (int j = 0; j < OSC_PER_MICRO; j++) {
        double a = -2.0 * M_PI * k * j / OSC_PER_MICRO;
        re += r->th[j] * cos(a);
        im += r->th[j] * sin(a);
    }
    return sqrt(re*re + im*im) / OSC_PER_MICRO;
}

static double macro_harmonic(macro_ring_t *mr, int k) {
    double weighted_sum = 0.0;
    double total_weight = 0.0;
    for (int m = 0; m < MICRO_PER_MACRO; m++) {
        double weight = mr->feature_weight[m];
        if (weight < 1e-10) weight = 1.0;
        double h = harmonic_mag(&mr->micro[m], k);
        weighted_sum += weight * h;
        total_weight += weight;
    }
    if (total_weight < 1e-10) return 0.0;
    return weighted_sum / total_weight;
}

static void make_harmonic_lens(double *lens, int harmonic, double amp) {
    memset(lens, 0, OSC_PER_MICRO * sizeof(double));
    double coeff = amp;
    for (int i = 0; i < OSC_PER_MICRO; i++) {
        lens[i] = coeff * cos(2.0 * M_PI * harmonic * i / OSC_PER_MICRO);
    }
}

static void make_pulse_receptive_lens(double *lens, int profile_k, double amp) {
    memset(lens, 0, OSC_PER_MICRO * sizeof(double));
    
    if (profile_k == 1) {
        double coeff[8] = {1.00, -0.15, 0.05, -0.02, 0.01, -0.01, 0.0, 0.0};
        for (int i = 0; i < OSC_PER_MICRO; i++) {
            for (int h = 0; h < 8; h++) {
                lens[i] += coeff[h] * cos(2.0 * M_PI * (h+1) * i / OSC_PER_MICRO);
            }
        }
    }
    else if (profile_k == 2) {
        double coeff[8] = {-0.20, 1.00, -0.10, 0.15, -0.05, 0.02, -0.01, 0.0};
        for (int i = 0; i < OSC_PER_MICRO; i++) {
            for (int h = 0; h < 8; h++) {
                lens[i] += coeff[h] * cos(2.0 * M_PI * (h+1) * i / OSC_PER_MICRO);
            }
        }
    }
    else if (profile_k == 3) {
        double coeff[8] = {-0.10, -0.15, 1.00, -0.10, 0.0, 0.15, -0.05, 0.02};
        for (int i = 0; i < OSC_PER_MICRO; i++) {
            for (int h = 0; h < 8; h++) {
                lens[i] += coeff[h] * cos(2.0 * M_PI * (h+1) * i / OSC_PER_MICRO);
            }
        }
    }
    else if (profile_k == 4) {
        double coeff[8] = {-0.05, -0.10, -0.15, 1.00, -0.10, 0.0, 0.10, 0.05};
        for (int i = 0; i < OSC_PER_MICRO; i++) {
            for (int h = 0; h < 8; h++) {
                lens[i] += coeff[h] * cos(2.0 * M_PI * (h+1) * i / OSC_PER_MICRO);
            }
        }
    }
    
    double max_val = 0.0;
    for (int i = 0; i < OSC_PER_MICRO; i++) {
        double abs_val = fabs(lens[i]);
        if (abs_val > max_val) max_val = abs_val;
    }
    if (max_val > 0.0) {
        double scale = amp / max_val;
        for (int i = 0; i < OSC_PER_MICRO; i++) {
            lens[i] *= scale;
        }
    }
}

static void apply_lens_micro(micro_ring_t *r) {
    memcpy(r->om, r->lens, OSC_PER_MICRO * sizeof(double));
}

static void encode_input(micro_ring_t *r, int harmonic, double amp) {
    memset(r->th, 0, sizeof(r->th));
    for (int j = 0; j < OSC_PER_MICRO; j++) {
        r->th[j] = amp * cos(2.0 * M_PI * harmonic * j / OSC_PER_MICRO);
    }
}

static void pulse_queue_init(pulse_queue_t *q) { q->count = 0; }

static void pulse_queue_push(pulse_queue_t *q, int src_vqpu, int src_level, int src_idx,
                              int dst_vqpu, int dst_level, int dst_idx,
                              double phase, double strength, int harmonic, int timestep) {
    if (q->count >= MAX_PULSES) return;
    pulse_t *p = &q->pulses[q->count++];
    p->src_vqpu = src_vqpu; p->src_level = src_level; p->src_idx = src_idx;
    p->dst_vqpu = dst_vqpu; p->dst_level = dst_level; p->dst_idx = dst_idx;
    p->phase = phase; p->strength = strength; p->harmonic = harmonic; p->timestep = timestep;
}

static void pulse_queue_deliver(pulse_queue_t *q, vqpu_t *vqpus, int num_vqpus, int current_chunk) {
    for (int i = 0; i < q->count; i++) {
        pulse_t *p = &q->pulses[i];
        if (p->timestep != current_chunk) continue;
        if (p->dst_vqpu < 0 || p->dst_vqpu >= num_vqpus) continue;

        vqpu_t *dst_vqpu = &vqpus[p->dst_vqpu];
        micro_ring_t *dst = NULL;
        int macro_id = -1;
        int micro_id = -1;

        if (p->dst_level == 0) {
            if (p->dst_idx < 0 || p->dst_idx >= MACRO_PER_SUPER * MICRO_PER_MACRO) continue;
            macro_id = p->dst_idx / MICRO_PER_MACRO;
            micro_id = p->dst_idx % MICRO_PER_MACRO;
            if (macro_id >= MACRO_PER_SUPER) continue;
            dst = &dst_vqpu->macro[macro_id].micro[micro_id];
        } else if (p->dst_level == 1) {
            if (p->dst_idx < 0 || p->dst_idx >= MACRO_PER_SUPER) continue;
            macro_id = p->dst_idx;
            micro_id = 0;
            dst = &dst_vqpu->macro[macro_id].micro[0];
        }

        if (dst && macro_id >= 0 && micro_id >= 0) {
            for (int j = 0; j < OSC_PER_MICRO; j++) {
                dst->th[j] += p->strength * sin(p->phase - dst->th[j]);
            }
            int pulse_site = (int)(p->phase * OSC_PER_MICRO / (2 * M_PI)) % OSC_PER_MICRO;
            if (pulse_site < 0) pulse_site += OSC_PER_MICRO;
            dst->pre_trace[pulse_site] = TRACE_DECAY * dst->pre_trace[pulse_site] + (1.0 - TRACE_DECAY) * 1.0;
            
            macro_ring_t *mr = &dst_vqpu->macro[macro_id];
            mr->feature_weight[micro_id] = mr->feature_weight_decay * mr->feature_weight[micro_id] + (1.0 - mr->feature_weight_decay) * p->strength;
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

static void test_macro_aggregation() {
    printf("\n=== MACRO AGGREGATION CROSS-vQPU ===\n");
    
    vqpu_t vqpu[2];
    init_micro(&vqpu[0], 0);
    init_micro(&vqpu[1], 1);
    pulse_queue_t pq; pulse_queue_init(&pq);

    /* vQPU 0: source - 16 micro rings encoding harmonics 1-4 */
    for (int u = 0; u < MICRO_PER_MACRO; u++) {
        micro_ring_t *mic = &vqpu[0].macro[0].micro[u];
        int harm = (u % 4) + 1;
        encode_input(mic, harm, 2.0);
        make_harmonic_lens(mic->lens, harm, 0.5);
        apply_lens_micro(mic);
        mic->last_q = measure_winding(mic);
    }

    /* vQPU 1: destination - pulse-receptive lenses, RANDOM initial state */
    for (int u = 0; u < MICRO_PER_MACRO; u++) {
        micro_ring_t *mic = &vqpu[1].macro[0].micro[u];
        int harm = (u % 4) + 1;
        make_pulse_receptive_lens(mic->lens, harm, 0.5);
        apply_lens_micro(mic);
        /* Random initial phase */
        for (int j = 0; j < OSC_PER_MICRO; j++) {
            mic->th[j] = ((double)rand() / RAND_MAX - 0.5) * 2.0;
        }
        mic->last_q = measure_winding(mic);
    }

    int total_pulses = 0;

    for (int chunk = 0; chunk < SETTLE_CHUNKS; chunk++) {
        for (int u = 0; u < MICRO_PER_MACRO; u++) {
            settle_micro(&vqpu[0].macro[0].micro[u], 1);
            settle_micro(&vqpu[1].macro[0].micro[u], 1);
        }

        for (int u = 0; u < MICRO_PER_MACRO; u++) {
            double q = measure_winding(&vqpu[0].macro[0].micro[u]);
            if (fabs(q - vqpu[0].macro[0].micro[u].last_q) > 0.5) {
                pulse_queue_push(&pq, 0, 0, u,  1, 0, u,  q, 0.3, (u%4)+1, chunk);
                total_pulses++;
            }
            vqpu[0].macro[0].micro[u].last_q = q;
        }

        pulse_queue_deliver(&pq, vqpu, 2, chunk);
    }

    double agg_score = 0;
    printf("Harmonic detection in vQPU 1 macro 0:\n");
    for (int k = 1; k <= 4; k++) {
        double h = macro_harmonic(&vqpu[1].macro[0], k);
        int detected = h > 0.05 ? 1 : 0;
        printf("  h%d = %.6f %s\n", k, h, detected ? "DETECTED" : "missed");
        if (detected) agg_score++;
    }
    agg_score /= 4.0;
    
    printf("macro_aggregation_score = %.3f (%d/4 harmonics)\n", agg_score, (int)(agg_score * 4));
    printf("total_pulses = %d\n", total_pulses);
}

static void test_feature_routing() {
    printf("\n=== MACRO FEATURE ROUTING CROSS-vQPU ===\n");
    
    vqpu_t vqpu[2];
    init_micro(&vqpu[0], 0);
    init_micro(&vqpu[1], 1);
    pulse_queue_t pq; pulse_queue_init(&pq);

    /* vQPU 0 (source): 16 micro rings, 4 per harmonic */
    for (int u = 0; u < MICRO_PER_MACRO; u++) {
        micro_ring_t *mic = &vqpu[0].macro[0].micro[u];
        int harm = (u / 4) + 1;
        encode_input(mic, harm, 2.0);
        make_harmonic_lens(mic->lens, harm, 0.5);
        apply_lens_micro(mic);
        mic->last_q = measure_winding(mic);
    }

    /* vQPU 1 (destination): pulse-receptive lenses, RANDOM initial state */
    for (int u = 0; u < MICRO_PER_MACRO; u++) {
        micro_ring_t *mic = &vqpu[1].macro[0].micro[u];
        int harm = (u / 4) + 1;
        make_pulse_receptive_lens(mic->lens, harm, 0.5);
        apply_lens_micro(mic);
        for (int j = 0; j < OSC_PER_MICRO; j++) {
            mic->th[j] = ((double)rand() / RAND_MAX - 0.5) * 2.0;
        }
        mic->last_q = measure_winding(mic);
    }

    int total_pulses = 0;
    double weighted_feature_scores[5] = {0};

    for (int chunk = 0; chunk < SETTLE_CHUNKS; chunk++) {
        for (int u = 0; u < MICRO_PER_MACRO; u++) {
            settle_micro(&vqpu[0].macro[0].micro[u], 1);
            settle_micro(&vqpu[1].macro[0].micro[u], 1);
        }

        for (int u = 0; u < MICRO_PER_MACRO; u++) {
            double q = measure_winding(&vqpu[0].macro[0].micro[u]);
            if (fabs(q - vqpu[0].macro[0].micro[u].last_q) > 0.5) {
                int target_micro = u;
                int harm = (u / 4) + 1;
                pulse_queue_push(&pq, 0, 0, u,  1, 0, target_micro,  q, 0.4 * (1.0 + harm * 0.1), harm, chunk);
                total_pulses++;
                weighted_feature_scores[harm] += 0.4 * (1.0 + harm * 0.1);
            }
            vqpu[0].macro[0].micro[u].last_q = q;
        }

        pulse_queue_deliver(&pq, vqpu, 2, chunk);
    }

    double weighted_agg_score = 0.0;
    int detected_harmonics = 0;
    
    printf("Feature routing results:\n");
    for (int k = 1; k <= 4; k++) {
        double h = macro_harmonic(&vqpu[1].macro[0], k);
        double expected_weight = weighted_feature_scores[k];
        int detected = h > 0.03 ? 1 : 0;
        printf("  h%d: macro_harm=%.6f, feature_weight=%.3f %s\n", 
               k, h, expected_weight, detected ? "DETECTED" : "missed");
        if (detected) {
            detected_harmonics++;
            weighted_agg_score += h;
        }
    }
    
    weighted_agg_score = detected_harmonics > 0 ? weighted_agg_score / detected_harmonics : 0.0;

    printf("routing_accuracy = %d/4 = %.3f\n", detected_harmonics, (double)detected_harmonics / 4.0);
    printf("weighted_aggregation_score = %.3f\n", weighted_agg_score);
    printf("total_pulses = %d\n", total_pulses);
}

int main() {
    srand(42);
    test_macro_aggregation();
    test_feature_routing();
    return 0;
}
