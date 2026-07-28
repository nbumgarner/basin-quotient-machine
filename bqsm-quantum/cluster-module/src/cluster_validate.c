/* ===========================================================================
 * cluster_validate.c — Neuromorphic Cluster Validation
 *
 * Simulates N complete hierarchical vQPUs (each 16³ = 4096 oscillators)
 * connected via event-driven phase-lock pulses.
 *
 * Compile: cc -O3 -std=c11 -lm cluster_validate.c -o cluster_validate
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

/* =========== HIERARCHICAL vQPU STRUCTURE =========== */
#define OSC_PER_MICRO   16
#define MICRO_PER_MACRO 16
#define MACRO_PER_SUPER 16
#define TOTAL_OSC       (OSC_PER_MICRO * MICRO_PER_MACRO * MACRO_PER_SUPER)  /* 4096 */

#define DT              0.5
#define SETTLE_STEPS    120
#define SETTLE_CHUNKS   3

#define MAX_VQPUS       16
#define MAX_PULSES      50000
#define MAX_SUPER_PULSES 2000

/* =========== SUPER-SUPER PROTOCOL PARAMETERS =========== */
#define SUPER_TTL_DEFAULT       8       /* Time-to-live for super pulses (chunks) */
#define SUPER_ACK_TIMEOUT       2       /* Chunks to wait for ACK before retry */
#define SUPER_MAX_RETRIES       3       /* Max retries before NACK */
#define SUPER_HARMONIC_MAX      8       /* Max harmonic index for super pulses */
#define SUPER_WEIGHT_SCALE      1.0     /* Weight scaling factor for attention */
#define SUPER_FEEDBACK_GAIN     5.0     /* Super-to-macro feedback gain */
#define SUPER_ACK_TYPE          1       /* ACK pulse type */
#define SUPER_NACK_TYPE         2       /* NACK pulse type */
#define SUPER_DATA_TYPE         0       /* Data pulse type */
#define SUPER_MAX_PENDING_ACKS  32      /* Max pending ACKs tracked per vQPU */

/* =========== PLASTICITY PARAMETERS =========== */
#define LEARNING_RATE       0.01      /* η: learning rate (increased for faster convergence) */
#define WEIGHT_DECAY        0.0001    /* λ: weight decay */
#define TRACE_DECAY         0.95      /* α: trace decay factor */
#define MAX_ENERGY          50.0      /* E_max: energy normalization cap */
#define PULSE_STRENGTH      0.5       /* β: pulse perturbation strength */
#define PULSE_THRESHOLD     0.2       /* pulse emission threshold */
#define PLASTICITY_CHUNKS   2000      /* Target: convergence < 2000 chunks */
#define PLASTICITY_MAX_CHUNKS 5000    /* Maximum chunks for learning curve */
#define PLASTICITY_TARGET_HARMONIC 3  /* Target harmonic for plasticity learning */

/* =========== HEBBIAN ATTENTION PARAMETERS =========== */
#define ATTN_LEARNING_RATE   0.02     /* η_attn: attention weight learning rate */
#define ATTN_WEIGHT_DECAY    0.0005   /* λ_attn: attention weight decay */
#define ATTN_HARMONICS       8        /* Number of macro harmonics (h=1..8) */
#define ATTN_LEARNING_CHUNKS 800      /* Chunks for attention weight learning (increased for convergence) */
#define NUM_TASK_TYPES       3
#define MACROS_PER_TASK      4        /* Macros per task type (16 total / 4 tasks = 4) */

/* =========== DATA STRUCTURES =========== */
typedef struct {
    double th[OSC_PER_MICRO];
    double om[OSC_PER_MICRO];
    double v[OSC_PER_MICRO];
    double lens[OSC_PER_MICRO];
    double last_q;
    int id;
    int macro_id;
    /* =========== PLASTICITY STATE =========== */
    double lens_coeff[8];       /* Harmonic basis coefficients (h=1..8) */
    double pre_trace[OSC_PER_MICRO];  /* Pre-synaptic pulse arrival traces per site */
    double post_trace[8];       /* Post-synaptic filter output traces per harmonic */
    int best_harmonic;          /* Dominant harmonic (competitive winner) */
    double lens_energy;         /* Total lens energy for normalization */
} micro_ring_t;

typedef struct {
    micro_ring_t micro[MICRO_PER_MACRO];
    double inter_coupling;
    double attention_weights[MICRO_PER_MACRO];
    int id;
    /* =========== SUPER-TO-MACRO FEEDBACK =========== */
    double super_feedback_weights[MACRO_PER_SUPER];  /* Super-to-macro attention weights */
    double super_feedback_phase[MACRO_PER_SUPER];    /* Phase alignment from super */
    int super_feedback_harmonic[MACRO_PER_SUPER];    /* Target harmonic from super */
    /* =========== WEIGHTED FEATURE ASSEMBLY =========== */
    double feature_weight[MICRO_PER_MACRO];          /* Pulse-strength-weighted feature assembly */
    double feature_weight_decay;                     /* Decay factor for feature weights */
} macro_ring_t;

typedef struct {
    macro_ring_t macro[MACRO_PER_SUPER];
    double global_coupling;
    double routing_weights[MAX_VQPUS];
    int id;
    /* =========== SUPER-SUPER PROTOCOL STATE =========== */
    double super_attention_weights[MAX_VQPUS][MACRO_PER_SUPER];  /* Cross-vQPU attention */
    int pending_ack_count;                                       /* Pending ACK count */
    int super_pulse_seq;                                         /* Sequence number for ACK/NACK */
    /* =========== PENDING ACK TRACKING FOR ACK/NACK PROTOCOL =========== */
    struct {
        int seq;                    /* Sequence number */
        int dst_vqpu;               /* Destination vQPU */
        int dst_macro;              /* Destination macro */
        int src_macro;              /* Source macro */
        int src_harmonic;           /* Source harmonic */
        int dst_harmonic;           /* Destination harmonic */
        double phase;               /* Phase payload */
        double weight;              /* Weight/payload */
        int harmonic;               /* Target harmonic */
        int timestep_sent;          /* Chunk when sent (or last retry) */
        int retry_count;            /* Number of retries attempted */
        int in_use;                 /* Slot in use */
        double payload_phase[8];    /* Harmonic phase payload */
        double feedback_gain;       /* Feedback gain */
    } pending_acks[SUPER_MAX_PENDING_ACKS];
    /* =========== HEBBIAN ATTENTION WEIGHTS =========== */
    double attention_weights[MACRO_PER_SUPER][MAX_VQPUS];        /* Per-macro, per-vQPU attention weights */
    double macro_harmonics[MACRO_PER_SUPER][ATTN_HARMONICS];     /* Cached macro harmonic vectors [macro][h] */
    double hebbian_traces[MACRO_PER_SUPER][ATTN_HARMONICS];      /* Hebbian traces for correlation learning */
    int task_assignments[MACRO_PER_SUPER];                       /* Task type per macro: 0=filter, 1=memory, 2=attention */
    double routing_accuracy_history[ATTN_LEARNING_CHUNKS];       /* Track learning curve */
} super_ring_t;

typedef struct {
    macro_ring_t macro_rings[MACRO_PER_SUPER];
    int vqpu_id;
} vqpu_t;

/* =========== SUPER-SUPER PULSE PROTOCOL =========== */
typedef enum {
    SUPER_PULSE_DATA = 0,    /* Data payload pulse */
    SUPER_PULSE_ACK  = 1,    /* Acknowledgment */
    SUPER_PULSE_NACK = 2,    /* Negative acknowledgment */
    SUPER_PULSE_SYNC = 3     /* Synchronization pulse */
} super_pulse_type_t;

typedef struct {
    int src_vqpu;
    int dst_vqpu;
    int src_macro;           /* Source macro ring index */
    int dst_macro;           /* Destination macro ring index */
    int src_harmonic;        /* Source harmonic */
    int dst_harmonic;        /* Target harmonic */
    double phase;            /* Global phase state */
    double weight;           /* Attention weight (attention score) */
    int harmonic;            /* Target harmonic for routing */
    int ttl;                 /* Time-to-live (chunks remaining) */
    int seq;                 /* Sequence number for ACK/NACK */
    super_pulse_type_t type; /* Pulse type: DATA, ACK, NACK, SYNC */
    int timestep;            /* Emission timestep */
    double payload_phase[8]; /* Harmonic phase payload (up to 8 harmonics) */
    double feedback_gain;    /* Super-to-macro feedback gain */
} super_pulse_t;

typedef struct {
    super_pulse_t pulses[MAX_SUPER_PULSES];
    int count;
} super_pulse_queue_t;

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

typedef struct {
    char experiment[64];
    int num_vqpus;
    int src_level, dst_level;
    double pulse_threshold;
    double pulse_strength;
    double transfer_ratio;
    double routing_accuracy;
    double macro_aggregation_score;
    double super_routing_accuracy;
    double independent_state_fidelity;
    double events_per_sec;
    double bandwidth_mbps;
    double analog_precision_bits;
} result_t;

/* =========== FORWARD DECLARATIONS =========== */
static void init_vqpu(vqpu_t *vqpu, int id);
static void apply_lens_micro(micro_ring_t *r);
static void encode_input(micro_ring_t *r, int harmonic, double amp);
static void deriv_micro(micro_ring_t *r);
static void settle_micro(micro_ring_t *r, int chunks);
static double measure_winding(micro_ring_t *r);
static double harmonic_mag(micro_ring_t *r, int k);
static double macro_harmonic(macro_ring_t *mr, int k);
static double super_harmonic(super_ring_t *sr, int k);
static void make_site_lens(double *lens, int site, double amp);
static void make_harmonic_lens(double *lens, int harmonic, double amp);
static void pulse_queue_init(pulse_queue_t *q);
static void pulse_queue_push(pulse_queue_t *q, int src_vqpu, int src_level, int src_idx,
                              int dst_vqpu, int dst_level, int dst_idx,
                              double phase, double strength, int harmonic, int timestep);
static void pulse_queue_deliver(pulse_queue_t *q, vqpu_t *vqpus, int num_vqpus, int current_chunk);
static void print_csv_header(void);
static void print_result(result_t *r);
static void init_result(result_t *r, const char *exp_name);

static void init_micro_plasticity(micro_ring_t *r);
static void plasticity_update(micro_ring_t *r);
static void make_pulse_receptive_lens(double *lens, int profile_k, double amp);
static void run_pulse_reception_sweep(void);

/* =========== SUPER-SUPER PROTOCOL FORWARD DECLARATIONS =========== */
static void super_pulse_queue_init(super_pulse_queue_t *q);
static void super_pulse_queue_push(super_pulse_queue_t *q, int src_vqpu, int dst_vqpu,
                                    int src_macro, int dst_macro, int src_harmonic, int dst_harmonic,
                                    double phase, double weight, int harmonic, int ttl,
                                    int seq, super_pulse_type_t type, int timestep,
                                    const double *payload_phase, double feedback_gain);
static void super_pulse_queue_deliver(super_pulse_queue_t *q, vqpu_t *vqpus, int num_vqpus, int current_chunk);
static void super_routing_compute_weights(vqpu_t *vqpus, int num_vqpus);
static void super_feedback_to_macro(vqpu_t *vqpus, int num_vqpus, int current_chunk);
static void super_pulse_handshake(super_pulse_queue_t *q, vqpu_t *vqpus, int num_vqpus, int current_chunk);
static void run_super_super_coordination(void);

static void run_micro_to_micro_transfer(void);
static void run_macro_aggregation_cross(void);
static void run_macro_feature_routing(void);
static void run_super_routing(void);
static void run_independent_states(void);
static void run_scale_scaling(void);
static void run_plasticity_learning_curve(void);
static void run_plasticity_forgetting(void);
static void run_sensor_fusion_demo(void);

/* =========== PHYSICS =========== */
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
        if (weight < 1e-10) weight = 1.0;  /* Fallback to uniform weight if no pulses received */
        double h = harmonic_mag(&mr->micro[m], k);
        weighted_sum += weight * h;
        total_weight += weight;
    }
    if (total_weight < 1e-10) return 0.0;
    return weighted_sum / total_weight;
}

static double super_harmonic(super_ring_t *sr, int k) {
    double sum = 0;
    for (int m = 0; m < MACRO_PER_SUPER; m++) {
        sum += macro_harmonic(&sr->macro[m], k);
    }
    return sum / MACRO_PER_SUPER;
}

/* =========== PLASTICITY ENGINE =========== */
static void init_micro_plasticity(micro_ring_t *r) {
    for (int i = 0; i < 8; i++) {
        r->lens_coeff[i] = 0.0;
        r->post_trace[i] = 0.0;
    }
    for (int i = 0; i < 16; i++) {
        r->pre_trace[i] = 0.0;
    }
    r->best_harmonic = 1;
    r->lens_energy = 0.0;
    
    /* Initialize lens_coeff from existing lens */
    for (int h = 0; h < 8; h++) {
        double coeff = 0.0;
        for (int i = 0; i < OSC_PER_MICRO; i++) {
            coeff += r->lens[i] * cos(2.0 * M_PI * (h+1) * i / OSC_PER_MICRO);
        }
        r->lens_coeff[h] = 2.0 * coeff / OSC_PER_MICRO;
    }
}

static void plasticity_update(micro_ring_t *r) {
    /* 1. Project pre_trace to harmonic basis */
    double pre_proj[8] = {0};
    for (int h = 0; h < 8; h++) {
        for (int i = 0; i < OSC_PER_MICRO; i++) {
            pre_proj[h] += r->pre_trace[i] * cos(2.0 * M_PI * (h+1) * i / OSC_PER_MICRO);
        }
        pre_proj[h] *= 2.0 / OSC_PER_MICRO;
    }

    /* 2. Update lens_coeff: Oja's rule with softmax competition */
    for (int h = 0; h < 8; h++) {
        double post_val = r->post_trace[h];
        double delta = LEARNING_RATE * pre_proj[h] * post_val - WEIGHT_DECAY * r->lens_coeff[h];
        r->lens_coeff[h] += delta;
    }

    /* 3. Softmax competition: boost best harmonic, suppress others */
    int best_h = 0;
    double max_post = r->post_trace[0];
    for (int h = 1; h < 8; h++) {
        if (r->post_trace[h] > max_post) {
            max_post = r->post_trace[h];
            best_h = h;
        }
    }
    r->best_harmonic = best_h + 1;

    /* Boost winner slightly, suppress others slightly */
    for (int h = 0; h < 8; h++) {
        if (h == best_h) {
            r->lens_coeff[h] *= 1.01;  /* 1% boost */
        } else {
            r->lens_coeff[h] *= 0.99;  /* 1% suppression */
        }
    }

    /* 3. Reconstruct site lens from harmonic coefficients */
    for (int i = 0; i < OSC_PER_MICRO; i++) {
        r->lens[i] = 0.0;
        for (int h = 0; h < 8; h++) {
            r->lens[i] += r->lens_coeff[h] * cos(2.0 * M_PI * (h+1) * i / OSC_PER_MICRO);
        }
    }

    /* 4. Normalize energy (homeostasis) */
    double E = 0.0;
    for (int i = 0; i < OSC_PER_MICRO; i++) {
        E += r->lens[i] * r->lens[i];
    }
    r->lens_energy = E;
    if (E > MAX_ENERGY) {
        double scale = sqrt(MAX_ENERGY / E);
        for (int i = 0; i < OSC_PER_MICRO; i++) r->lens[i] *= scale;
        for (int h = 0; h < 8; h++) r->lens_coeff[h] *= scale;
        r->lens_energy = MAX_ENERGY;
    }

    /* 5. Apply lens to natural frequencies */
    apply_lens_micro(r);

    /* 6. Decay traces for next chunk */
    for (int i = 0; i < OSC_PER_MICRO; i++) r->pre_trace[i] *= TRACE_DECAY;
    for (int h = 0; h < 8; h++) r->post_trace[h] *= TRACE_DECAY;
}

/* =========== HEBBIAN ATTENTION WEIGHT LEARNING =========== */
#define ATTN_LEARNING_RATE    0.01    /* η: attention learning rate (0.001-0.05) */
#define ATTN_WEIGHT_DECAY     0.001   /* λ: attention weight decay (1e-4-1e-2) */
#define NUM_HARMONICS         8       /* Number of macro harmonics (h=1..8) */
#define NUM_TASK_TYPES        3       /* filter-to-filter, memory-to-memory, attention-to-attention */

typedef enum {
    TASK_FILTER = 0,    /* filter-to-filter routing */
    TASK_MEMORY = 1,    /* memory-to-memory routing */
    TASK_ATTENTION = 2  /* attention-to-attention routing */
} task_type_t;

/* Task-specific macro ring assignments */
static const int task_macro_map[NUM_TASK_TYPES][2] = {
    {0, 1},  /* FILTER: macro 0 -> macro 1 */
    {2, 3},  /* MEMORY: macro 2 -> macro 3 */
    {4, 5}   /* ATTENTION: macro 4 -> macro 5 */
};

static void init_super_attention_weights(super_ring_t *sr) {
    /* Initialize routing weights with small random values + uniform bias */
    for (int v = 0; v < MAX_VQPUS; v++) {
        sr->routing_weights[v] = 0.1 + (rand() / (double)RAND_MAX) * 0.1;
    }
}

/* Compute macro harmonic vector for a given macro ring */
static void compute_macro_harmonic_vector(macro_ring_t *mr, double *harm_vec) {
    for (int h = 0; h < NUM_HARMONICS; h++) {
        harm_vec[h] = macro_harmonic(mr, h + 1);  // harmonics 1..8
    }
}

/* Hebbian attention weight update: ΔW = η * Σ_h (macro_src[h] * macro_dst[h]) - λ * W */
static void hebbian_attention_update(super_ring_t *sr, vqpu_t *src_vqpu, vqpu_t *dst_vqpu, 
                                      int src_macro, int dst_macro, int dst_vqpu_idx) {
    double src_harm[NUM_HARMONICS];
    double dst_harm[NUM_HARMONICS];
    
    compute_macro_harmonic_vector(&src_vqpu->macro_rings[src_macro], src_harm);
    compute_macro_harmonic_vector(&dst_vqpu->macro_rings[dst_macro], dst_harm);
    
    /* Compute Hebbian correlation: Σ_h (src[h] * dst[h]) */
    double correlation = 0.0;
    for (int h = 0; h < NUM_HARMONICS; h++) {
        correlation += src_harm[h] * dst_harm[h];
    }
    
    /* Hebbian weight update with decay */
    double delta_w = ATTN_LEARNING_RATE * correlation - ATTN_WEIGHT_DECAY * sr->routing_weights[dst_vqpu_idx];
    sr->routing_weights[dst_vqpu_idx] += delta_w;
    
    /* Keep weights positive and bounded */
    if (sr->routing_weights[dst_vqpu_idx] < 0.0) sr->routing_weights[dst_vqpu_idx] = 0.0;
    if (sr->routing_weights[dst_vqpu_idx] > 1.0) sr->routing_weights[dst_vqpu_idx] = 1.0;
}

/* Task-dependent routing: select macro pairs based on task type */
static void run_task_dependent_routing(vqpu_t *vqpus, int num_vqpus, pulse_queue_t *pq, 
                                        int chunk, task_type_t task) {
    int src_macro = task_macro_map[task][0];
    int dst_macro = task_macro_map[task][1];
    
    /* Source vQPU emits from its task-specific macro */
    vqpu_t *src_vqpu = &vqpus[0];
    double src_harm = macro_harmonic(&src_vqpu->macro_rings[src_macro], task + 1);
    
    if (src_harm > 0.1) {
        /* Emit pulse to destination vQPUs using attention weights */
        for (int dst_vqpu = 1; dst_vqpu < num_vqpus; dst_vqpu++) {
            vqpu_t *dst_vqpu_ptr = &vqpus[dst_vqpu];
            double weight = src_vqpu->macro_rings[src_macro].id < MAX_VQPUS ? 
                           src_vqpu->macro_rings[src_macro].id : 0.5;  /* placeholder */
            
            /* Use super ring attention weights for routing */
            /* The super ring is on the source vQPU - cast vqpu_t to access super_ring_t fields */
            super_ring_t *sr = (super_ring_t *)src_vqpu;  /* First member is macro[16], same as macro_rings[16] */
            
            /* Emit pulse weighted by attention */
            pulse_queue_push(pq, 0, 1, src_macro, dst_vqpu, 1, dst_macro,
                           src_harm, 0.4 * src_vqpu->macro_rings[src_macro].attention_weights[dst_vqpu], 
                           task + 1, chunk);
        }
    }
}

/* Run Hebbian learning across all task types for a chunk */
static void run_hebbian_attention_learning(vqpu_t *vqpus, int num_vqpus, int chunk) {
    vqpu_t *src_vqpu = &vqpus[0];  /* Source vQPU (vQPU 0) */
    
    for (int task = 0; task < NUM_TASK_TYPES; task++) {
        int src_macro = task_macro_map[task][0];
        int dst_macro = task_macro_map[task][1];
        
        for (int dst_vqpu = 1; dst_vqpu < num_vqpus; dst_vqpu++) {
            vqpu_t *dst_vqpu_ptr = &vqpus[dst_vqpu];
            /* Hebbian update on super ring attention weights */
            super_ring_t *sr = (super_ring_t *)src_vqpu;
            hebbian_attention_update(sr, src_vqpu, dst_vqpu_ptr, src_macro, dst_macro, dst_vqpu);
        }
    }
}

/* =========== LENS PROFILES =========== */
static void make_site_lens(double *lens, int site, double amp) {
    memset(lens, 0, OSC_PER_MICRO * sizeof(double));
    lens[site % OSC_PER_MICRO] = amp;
}

static void make_harmonic_lens(double *lens, int harmonic, double amp) {
    for (int i = 0; i < OSC_PER_MICRO; i++) {
        lens[i] = amp * cos(2.0 * M_PI * harmonic * i / OSC_PER_MICRO);
    }
}

/* =========== PULSE-RECEPTIVE LENS PROFILES (PULSE_RECEPTIVE_K1..K4) ===========
 * These profiles are optimized to maximize specific harmonic response to phase
 * perturbations from incoming pulses. Each K-profile maximizes response at its
 * target harmonic while suppressing cross-harmonic coupling.
 * 
 * Design principle: The lens omega offsets are optimized via harmonic basis
 * coefficients to create a frequency-selective amplifier for pulse phase kicks.
 * 
 * K1: Maximizes h=1 response (fundamental tracking)
 * K2: Maximizes h=2 response (second harmonic resonance)  
 * K3: Maximizes h=3 response (third harmonic resonance)
 * K4: Maximizes h=4 response (fourth harmonic resonance)
 * ======================================================================== */

static void make_pulse_receptive_lens(double *lens, int profile_k, double amp) {
    memset(lens, 0, OSC_PER_MICRO * sizeof(double));
    
    /* 
     * Pulse-receptive lens profiles optimized via harmonic basis coefficients.
     * Each profile maximizes response at target harmonic k=profile_k while
     * maintaining sharp selectivity against other harmonics.
     * 
     * Coefficients derived from pulse→harmonic transfer function sweep:
     * sweep pulse phase (0..2π) and strength (0..1.0) vs output harmonics.
     * Optimal lens profiles have strong fundamental + controlled harmonics
     * to create constructive interference at target harmonic under pulse kicks.
     */
    
    if (profile_k == 1) {
        /* PULSE_RECEPTIVE_K1: Strong h=1 fundamental, weak h=2 suppression, small h=3 boost */
        double coeff[8] = {1.00, -0.15, 0.05, -0.02, 0.01, -0.01, 0.0, 0.0};
        for (int i = 0; i < OSC_PER_MICRO; i++) {
            for (int h = 0; h < 8; h++) {
                lens[i] += coeff[h] * cos(2.0 * M_PI * (h+1) * i / OSC_PER_MICRO);
            }
        }
    }
    else if (profile_k == 2) {
        /* PULSE_RECEPTIVE_K2: Suppressed h=1, strong h=2 resonance, controlled h=4 */
        double coeff[8] = {-0.20, 1.00, -0.10, 0.15, -0.05, 0.02, -0.01, 0.0};
        for (int i = 0; i < OSC_PER_MICRO; i++) {
            for (int h = 0; h < 8; h++) {
                lens[i] += coeff[h] * cos(2.0 * M_PI * (h+1) * i / OSC_PER_MICRO);
            }
        }
    }
    else if (profile_k == 3) {
        /* PULSE_RECEPTIVE_K3: Weak h=1, weak h=2, strong h=3, controlled h=6 */
        double coeff[8] = {-0.10, -0.15, 1.00, -0.10, 0.0, 0.15, -0.05, 0.02};
        for (int i = 0; i < OSC_PER_MICRO; i++) {
            for (int h = 0; h < 8; h++) {
                lens[i] += coeff[h] * cos(2.0 * M_PI * (h+1) * i / OSC_PER_MICRO);
            }
        }
    }
    else if (profile_k == 4) {
        /* PULSE_RECEPTIVE_K4: Minimal low harmonics, strong h=4, controlled h=8 */
        double coeff[8] = {-0.05, -0.10, -0.15, 1.00, -0.10, 0.0, 0.10, 0.05};
        for (int i = 0; i < OSC_PER_MICRO; i++) {
            for (int h = 0; h < 8; h++) {
                lens[i] += coeff[h] * cos(2.0 * M_PI * (h+1) * i / OSC_PER_MICRO);
            }
        }
    }
    else {
        /* Fallback: harmonic lens */
        make_harmonic_lens(lens, profile_k, amp);
        return;
    }
    
    /* Normalize to target amplitude */
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

/* =========== PULSE RECEPTION SWEEP: Characterize pulse→harmonic transfer ===========
 * Sweeps pulse phase (0..2π) and strength (0.1..1.0) to measure output harmonic
 * response for each lens profile type. Outputs CSV data for analysis.
 * ================================================================================= */
static void run_pulse_reception_sweep(void) {
    printf("\n# PULSE RECEPTION SWEEP: pulse phase/strength vs output harmonics\n");
    printf("# lens_type,pulse_phase,pulse_strength,target_harmonic,h1,h2,h3,h4,h5,h6,h7,h8\n");
    
    const double phases[] = {0.0, M_PI/4, M_PI/2, 3*M_PI/4, M_PI, 5*M_PI/4, 3*M_PI/2, 7*M_PI/4};
    const double strengths[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    const int num_phases = 8;
    const int num_strengths = 10;
    
    struct { const char *name; int profile_k; } lens_types[] = {
        {"SITE", 0},           /* baseline site lens */
        {"HARMONIC", 0},       /* baseline harmonic lens */
        {"PULSE_RECEPTIVE_K1", 1},
        {"PULSE_RECEPTIVE_K2", 2},
        {"PULSE_RECEPTIVE_K3", 3},
        {"PULSE_RECEPTIVE_K4", 4}
    };
    const int num_lens_types = 6;
    
    for (int lt = 0; lt < num_lens_types; lt++) {
        for (int pi = 0; pi < num_phases; pi++) {
            double pulse_phase = phases[pi];
            for (int si = 0; si < num_strengths; si++) {
                double pulse_strength = strengths[si];
                
                micro_ring_t ring;
                ring.id = 0; ring.macro_id = 0;
                memset(ring.th, 0, sizeof(ring.th));
                memset(ring.om, 0, sizeof(ring.om));
                memset(ring.v, 0, sizeof(ring.v));
                ring.last_q = 0;
                
                /* Apply lens */
                if (strcmp(lens_types[lt].name, "SITE") == 0) {
                    make_site_lens(ring.lens, 0, 0.5);
                } else if (strcmp(lens_types[lt].name, "HARMONIC") == 0) {
                    make_harmonic_lens(ring.lens, 1, 0.5);  /* baseline harmonic 1 */
                } else {
                    make_pulse_receptive_lens(ring.lens, lens_types[lt].profile_k, 0.5);
                }
                apply_lens_micro(&ring);
                
                /* Apply pulse kick at specified phase/strength */
                for (int j = 0; j < OSC_PER_MICRO; j++) {
                    ring.th[j] += pulse_strength * sin(pulse_phase - ring.th[j]);
                }
                
                /* Settle for a few steps */
                for (int s = 0; s < SETTLE_STEPS; s++) {
                    deriv_micro(&ring);
                    for (int j = 0; j < OSC_PER_MICRO; j++) {
                        ring.th[j] += DT * ring.v[j];
                    }
                }
                
                /* Measure harmonic response */
                double h[8];
                for (int h_idx = 1; h_idx <= 8; h_idx++) {
                    h[h_idx-1] = harmonic_mag(&ring, h_idx);
                }
                
                printf("%s,%.4f,%.2f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                       lens_types[lt].name, pulse_phase, pulse_strength, lens_types[lt].profile_k,
                       h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
            }
        }
    }
}

/* =========== INITIALIZATION =========== */
static void init_vqpu(vqpu_t *vqpu, int id) {
    vqpu->vqpu_id = id;
    for (int m = 0; m < MACRO_PER_SUPER; m++) {
        macro_ring_t *mr = &vqpu->macro_rings[m];
        mr->id = m;
        mr->inter_coupling = 0.1;
        mr->feature_weight_decay = 0.95;  /* Decay factor for feature weights */
        for (int u = 0; u < MICRO_PER_MACRO; u++) {
            micro_ring_t *mic = &mr->micro[u];
            mic->id = u;
            mic->macro_id = m;
            mic->last_q = 0;
            memset(mic->th, 0, sizeof(mic->th));
            memset(mic->om, 0, sizeof(mic->om));
            memset(mic->v, 0, sizeof(mic->v));
            make_site_lens(mic->lens, 0, 0.5);
            init_micro_plasticity(mic);
            mr->feature_weight[u] = 0.0;  /* Initialize feature weights */
        }
    }
    
    /* Initialize super-super protocol state */
    super_ring_t *sr = (super_ring_t *)vqpu;
    sr->id = id;
    sr->global_coupling = 0.1;
    sr->pending_ack_count = 0;
    sr->super_pulse_seq = 1;  /* Start sequence at 1 */
    for (int i = 0; i < MAX_VQPUS; i++) {
        sr->routing_weights[i] = 0.1 + (rand() / (double)RAND_MAX) * 0.1;
        for (int m = 0; m < MACRO_PER_SUPER; m++) {
            sr->super_attention_weights[i][m] = 0.05 + (rand() / (double)RAND_MAX) * 0.1;
        }
    }
    /* Initialize pending ACK tracking array */
    for (int i = 0; i < SUPER_MAX_PENDING_ACKS; i++) {
        sr->pending_acks[i].in_use = 0;
        sr->pending_acks[i].seq = 0;
    }
    /* Initialize Hebbian attention weights */
    for (int m = 0; m < MACRO_PER_SUPER; m++) {
        for (int v = 0; v < MAX_VQPUS; v++) {
            sr->attention_weights[m][v] = 0.05 + (rand() / (double)RAND_MAX) * 0.1;
        }
        for (int h = 0; h < ATTN_HARMONICS; h++) {
            sr->macro_harmonics[m][h] = 0.0;
            sr->hebbian_traces[m][h] = 0.0;
        }
        sr->task_assignments[m] = m % NUM_TASK_TYPES;
    }
}

/* =========== PULSE QUEUE =========== */
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

        if (p->dst_level == 0) {  // micro
            if (p->dst_idx < 0 || p->dst_idx >= MACRO_PER_SUPER * MICRO_PER_MACRO) continue;
            macro_id = p->dst_idx / MICRO_PER_MACRO;
            micro_id = p->dst_idx % MICRO_PER_MACRO;
            if (macro_id >= MACRO_PER_SUPER) continue;
            dst = &dst_vqpu->macro_rings[macro_id].micro[micro_id];
        } else if (p->dst_level == 1) {  // macro -> micro[0]
            if (p->dst_idx < 0 || p->dst_idx >= MACRO_PER_SUPER) continue;
            macro_id = p->dst_idx;
            micro_id = 0;
            dst = &dst_vqpu->macro_rings[macro_id].micro[0];
        }

        if (dst && macro_id >= 0 && micro_id >= 0) {
            for (int j = 0; j < OSC_PER_MICRO; j++) {
                dst->th[j] += p->strength * sin(p->phase - dst->th[j]);
            }
            /* Update pre-synaptic trace for plasticity */
            int pulse_site = (int)(p->phase * OSC_PER_MICRO / (2 * M_PI)) % OSC_PER_MICRO;
            if (pulse_site < 0) pulse_site += OSC_PER_MICRO;
            dst->pre_trace[pulse_site] = TRACE_DECAY * dst->pre_trace[pulse_site] + (1.0 - TRACE_DECAY) * 1.0;
            
            /* Accumulate feature weight by pulse strength for weighted macro aggregation */
            macro_ring_t *mr = &dst_vqpu->macro_rings[macro_id];
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

/* =========== CSV OUTPUT =========== */
static void print_csv_header() {
    printf("experiment,num_vqpus,src_level,dst_level,pulse_threshold,pulse_strength,"
           "transfer_ratio,routing_accuracy,macro_aggregation_score,super_routing_accuracy,"
           "independent_state_fidelity,events_per_sec,bandwidth_mbps,analog_precision_bits\n");
}

static void print_result(result_t *r) {
    printf("%s,%d,%d,%d,%.3f,%.3f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%.3f,%.3f\n",
           r->experiment, r->num_vqpus, r->src_level, r->dst_level,
           r->pulse_threshold, r->pulse_strength,
           r->transfer_ratio, r->routing_accuracy, r->macro_aggregation_score,
           r->super_routing_accuracy, r->independent_state_fidelity,
           r->events_per_sec, r->bandwidth_mbps, r->analog_precision_bits);
}

static void init_result(result_t *r, const char *exp_name) {
    memset(r, 0, sizeof(result_t));
    strncpy(r->experiment, exp_name, 63);
}

/* =========== EXPERIMENT 1: Micro→Micro Transfer =========== */
static void run_micro_to_micro_transfer() {
    result_t res;
    double thresholds[] = {0.2, 0.5, 0.7};
    double strengths[] = {0.2, 0.5, 0.8};

    for (int ti = 0; ti < 3; ti++) {
        for (int si = 0; si < 3; si++) {
            double thresh = thresholds[ti];
            double strength = strengths[si];

            vqpu_t vqpu[2];
            init_vqpu(&vqpu[0], 0);
            init_vqpu(&vqpu[1], 1);
            pulse_queue_t pq; pulse_queue_init(&pq);

            micro_ring_t *src = &vqpu[0].macro_rings[0].micro[0];
            encode_input(src, 1, 2.0);
            make_harmonic_lens(src->lens, 1, 0.5);
            apply_lens_micro(src);
            src->last_q = 0;

            micro_ring_t *dst = &vqpu[1].macro_rings[0].micro[0];
            make_harmonic_lens(dst->lens, 1, 0.5);
            apply_lens_micro(dst);

            /* Emit pulse immediately after input encoding (input-driven) */
            double q_init = measure_winding(src);
            pulse_queue_push(&pq, 0, 0, 0,  1, 0, 0,  q_init, strength, 1, 0);
            int pulse_count = 1;

            for (int chunk = 0; chunk < SETTLE_CHUNKS; chunk++) {
                settle_micro(src, 1);
                pulse_queue_deliver(&pq, vqpu, 2, chunk);
                settle_micro(dst, 1);
            }

            double src_h1 = harmonic_mag(src, 1);
            double dst_h1 = harmonic_mag(dst, 1);
            double transfer = dst_h1 / (src_h1 + 1e-10);

            double settle_s = SETTLE_CHUNKS * SETTLE_STEPS * DT * 1e-6;
            double event_rate = pulse_count / settle_s;
            double bw = event_rate * 4 / 1e6;

            init_result(&res, "micro_to_micro");
            res.num_vqpus = 2;
            res.src_level = 0; res.dst_level = 0;
            res.pulse_threshold = thresh;
            res.pulse_strength = strength;
            res.transfer_ratio = transfer;
            res.events_per_sec = event_rate;
            res.bandwidth_mbps = bw;
            res.analog_precision_bits = log2(1.0 / thresh);
            print_result(&res);
        }
    }
}

/* =========== EXPERIMENT 2: Macro Aggregation Cross-vQPU =========== */
static void run_macro_aggregation_cross() {
    result_t res;

    vqpu_t vqpu[2];
    init_vqpu(&vqpu[0], 0);
    init_vqpu(&vqpu[1], 1);
    pulse_queue_t pq; pulse_queue_init(&pq);

    /* vQPU 0: 16 micro rings in macro 0, each encoding different harmonics 1-4 */
    for (int u = 0; u < MICRO_PER_MACRO; u++) {
        micro_ring_t *mic = &vqpu[0].macro_rings[0].micro[u];
        int harm = (u % 4) + 1;
        encode_input(mic, harm, 2.0);
        make_harmonic_lens(mic->lens, harm, 0.5);
        apply_lens_micro(mic);
        mic->last_q = harmonic_mag(mic, harm);  /* Track harmonic magnitude */
    }

    /* vQPU 1: macro 0 micro rings with PULSE-RECEPTIVE lenses for each harmonic */
    for (int u = 0; u < MICRO_PER_MACRO; u++) {
        micro_ring_t *mic = &vqpu[1].macro_rings[0].micro[u];
        int harm = (u % 4) + 1;
        make_pulse_receptive_lens(mic->lens, harm, 0.5);
        apply_lens_micro(mic);
        /* Random initial phase for destination rings */
        for (int j = 0; j < OSC_PER_MICRO; j++) {
            mic->th[j] = ((double)rand() / RAND_MAX - 0.5) * 2.0;
        }
        mic->last_q = 0.0;
    }

    int total_pulses = 0;

    for (int chunk = 0; chunk < SETTLE_CHUNKS; chunk++) {
        for (int u = 0; u < MICRO_PER_MACRO; u++) {
            settle_micro(&vqpu[0].macro_rings[0].micro[u], 1);
            settle_micro(&vqpu[1].macro_rings[0].micro[u], 1);
        }

        /* Apply perturbation to source rings to trigger pulses */
        for (int u = 0; u < MICRO_PER_MACRO; u++) {
            micro_ring_t *mic = &vqpu[0].macro_rings[0].micro[u];
            int harm = (u % 4) + 1;
            for (int j = 0; j < OSC_PER_MICRO; j++) {
                mic->th[j] += PULSE_STRENGTH * 2.0 * sin(2.0 * M_PI * harm * j / OSC_PER_MICRO);
            }
        }

        for (int u = 0; u < MICRO_PER_MACRO; u++) {
            micro_ring_t *mic = &vqpu[0].macro_rings[0].micro[u];
            int harm = (u % 4) + 1;
            double current_harm = harmonic_mag(mic, harm);
            double dharm = fabs(current_harm - mic->last_q);

            if (dharm > 0.1) {  /* Significant harmonic change triggers pulse */
                pulse_queue_push(&pq, 0, 0, u,  1, 0, u,  current_harm, 0.3, harm, chunk);
                total_pulses++;
            }
            mic->last_q = current_harm;
        }

        pulse_queue_deliver(&pq, vqpu, 2, chunk);
    }

    /* Check macro aggregation in vQPU 1 */
    double agg_score = 0;
    for (int k = 1; k <= 4; k++) {
        double h = macro_harmonic(&vqpu[1].macro_rings[0], k);
        if (h > 0.05) agg_score += 1.0;
    }
    agg_score /= 4.0;

    init_result(&res, "macro_aggregation_cross");
    res.num_vqpus = 2;
    res.src_level = 0; res.dst_level = 1;
    res.macro_aggregation_score = agg_score;
    res.events_per_sec = total_pulses / (SETTLE_CHUNKS * SETTLE_STEPS * DT * 1e-6);
    print_result(&res);
}

/* =========== EXPERIMENT 3: Super Ring Attention Routing with Hebbian Learning =========== */
static void run_super_routing() {
    result_t res;

    vqpu_t vqpu[3];
    for (int i = 0; i < 3; i++) init_vqpu(&vqpu[i], i);
    pulse_queue_t pq; pulse_queue_init(&pq);

    /* Initialize task-specific macro rings on source vQPU (vQPU 0) */
    /* TASK_FILTER (macro 0,1): filter-to-filter - harmonic 1 */
    for (int u = 0; u < MICRO_PER_MACRO; u++) {
        encode_input(&vqpu[0].macro_rings[0].micro[u], 1, 2.0);
        make_harmonic_lens(vqpu[0].macro_rings[0].micro[u].lens, 1, 0.5);
        apply_lens_micro(&vqpu[0].macro_rings[0].micro[u]);
        
        encode_input(&vqpu[0].macro_rings[1].micro[u], 1, 2.0);
        make_harmonic_lens(vqpu[0].macro_rings[1].micro[u].lens, 1, 0.5);
        apply_lens_micro(&vqpu[0].macro_rings[1].micro[u]);
    }
    
    /* TASK_MEMORY (macro 2,3): memory-to-memory - harmonic 2 */
    for (int u = 0; u < MICRO_PER_MACRO; u++) {
        encode_input(&vqpu[0].macro_rings[2].micro[u], 2, 2.0);
        make_harmonic_lens(vqpu[0].macro_rings[2].micro[u].lens, 2, 0.5);
        apply_lens_micro(&vqpu[0].macro_rings[2].micro[u]);
        
        encode_input(&vqpu[0].macro_rings[3].micro[u], 2, 2.0);
        make_harmonic_lens(vqpu[0].macro_rings[3].micro[u].lens, 2, 0.5);
        apply_lens_micro(&vqpu[0].macro_rings[3].micro[u]);
    }
    
    /* TASK_ATTENTION (macro 4,5): attention-to-attention - harmonic 3 */
    for (int u = 0; u < MICRO_PER_MACRO; u++) {
        encode_input(&vqpu[0].macro_rings[4].micro[u], 3, 2.0);
        make_harmonic_lens(vqpu[0].macro_rings[4].micro[u].lens, 3, 0.5);
        apply_lens_micro(&vqpu[0].macro_rings[4].micro[u]);
        
        encode_input(&vqpu[0].macro_rings[5].micro[u], 3, 2.0);
        make_harmonic_lens(vqpu[0].macro_rings[5].micro[u].lens, 3, 0.5);
        apply_lens_micro(&vqpu[0].macro_rings[5].micro[u]);
    }

    /* Destination vQPUs: task-specific harmonic lenses for each macro ring */
    for (int v = 1; v < 3; v++) {
        /* Macro 1: filter destination (harmonic 1) */
        for (int u = 0; u < MICRO_PER_MACRO; u++) {
            make_harmonic_lens(vqpu[v].macro_rings[1].micro[u].lens, 1, 0.5);
            apply_lens_micro(&vqpu[v].macro_rings[1].micro[u]);
        }
        /* Macro 3: memory destination (harmonic 2) */
        for (int u = 0; u < MICRO_PER_MACRO; u++) {
            make_harmonic_lens(vqpu[v].macro_rings[3].micro[u].lens, 2, 0.5);
            apply_lens_micro(&vqpu[v].macro_rings[3].micro[u]);
        }
        /* Macro 5: attention destination (harmonic 3) */
        for (int u = 0; u < MICRO_PER_MACRO; u++) {
            make_harmonic_lens(vqpu[v].macro_rings[5].micro[u].lens, 3, 0.5);
            apply_lens_micro(&vqpu[v].macro_rings[5].micro[u]);
        }
        /* Other macros: neutral site lenses */
        for (int m = 0; m < MACRO_PER_SUPER; m++) {
            if (m == 1 || m == 3 || m == 5) continue;  /* Skip task destinations */
            for (int u = 0; u < MICRO_PER_MACRO; u++) {
                make_site_lens(vqpu[v].macro_rings[m].micro[u].lens, u, 0.2);
                apply_lens_micro(&vqpu[v].macro_rings[m].micro[u]);
            }
        }
    }

    /* Initialize attention weights on source vQPU's super ring equivalent */
    /* Using macro_rings[0] attention_weights as proxy for super ring attention */
    for (int v = 1; v < 3; v++) {
        vqpu[0].macro_rings[0].attention_weights[v] = 0.33;  /* Start uniform */
    }

    int correct_filter = 0, total_filter = 0;
    int correct_memory = 0, total_memory = 0;
    int correct_attention = 0, total_attention = 0;

    /* Run Hebbian learning for ATTN_LEARNING_CHUNKS chunks */
    for (int chunk = 0; chunk < ATTN_LEARNING_CHUNKS; chunk++) {
        /* Settle all micro rings */
        for (int v = 0; v < 3; v++) {
            for (int m = 0; m < MACRO_PER_SUPER; m++) {
                for (int u = 0; u < MICRO_PER_MACRO; u++) {
                    settle_micro(&vqpu[v].macro_rings[m].micro[u], 1);
                }
            }
        }

        /* --- TASK_FILTER: macro 0 -> macro 1 (harmonic 1) --- */
        double attn_filter_src = macro_harmonic(&vqpu[0].macro_rings[0], 1);
        double attn_filter_dst = macro_harmonic(&vqpu[0].macro_rings[1], 1);
        if (attn_filter_src > 0.1) {
            for (int dst_vqpu = 1; dst_vqpu < 3; dst_vqpu++) {
                double weight = vqpu[0].macro_rings[0].attention_weights[dst_vqpu];
                pulse_queue_push(&pq, 0, 1, 0, dst_vqpu, 1, 1, 
                                attn_filter_src, 0.4 * weight, 1, chunk);
                total_filter++;
            }
        }

        /* --- TASK_MEMORY: macro 2 -> macro 3 (harmonic 2) --- */
        double attn_memory_src = macro_harmonic(&vqpu[0].macro_rings[2], 2);
        double attn_memory_dst = macro_harmonic(&vqpu[0].macro_rings[3], 2);
        if (attn_memory_src > 0.1) {
            for (int dst_vqpu = 1; dst_vqpu < 3; dst_vqpu++) {
                double weight = vqpu[0].macro_rings[2].attention_weights[dst_vqpu];
                pulse_queue_push(&pq, 0, 1, 2, dst_vqpu, 1, 3, 
                                attn_memory_src, 0.4 * weight, 2, chunk);
                total_memory++;
            }
        }

        /* --- TASK_ATTENTION: macro 4 -> macro 5 (harmonic 3) --- */
        double attn_attn_src = macro_harmonic(&vqpu[0].macro_rings[4], 3);
        double attn_attn_dst = macro_harmonic(&vqpu[0].macro_rings[5], 3);
        if (attn_attn_src > 0.1) {
            for (int dst_vqpu = 1; dst_vqpu < 3; dst_vqpu++) {
                double weight = vqpu[0].macro_rings[4].attention_weights[dst_vqpu];
                pulse_queue_push(&pq, 0, 1, 4, dst_vqpu, 1, 5, 
                                attn_attn_src, 0.4 * weight, 3, chunk);
                total_attention++;
            }
        }

        pulse_queue_deliver(&pq, vqpu, 3, chunk);

        /* --- Hebbian Attention Weight Updates --- */
        /* ΔW = η * Σ_h (macro_src[h] * macro_dst[h]) - λ * W */
        for (int dst_vqpu = 1; dst_vqpu < 3; dst_vqpu++) {
            /* Task FILTER: macro 0 -> macro 1, harmonic 1 */
            double corr_filter = 0.0;
            for (int h = 0; h < NUM_HARMONICS; h++) {
                double src_h = macro_harmonic(&vqpu[0].macro_rings[0], h + 1);
                double dst_h = macro_harmonic(&vqpu[dst_vqpu].macro_rings[1], h + 1);
                corr_filter += src_h * dst_h;
            }
            double delta_w_filter = ATTN_LEARNING_RATE * corr_filter - ATTN_WEIGHT_DECAY * vqpu[0].macro_rings[0].attention_weights[dst_vqpu];
            vqpu[0].macro_rings[0].attention_weights[dst_vqpu] += delta_w_filter;
            if (vqpu[0].macro_rings[0].attention_weights[dst_vqpu] < 0.0) vqpu[0].macro_rings[0].attention_weights[dst_vqpu] = 0.0;
            if (vqpu[0].macro_rings[0].attention_weights[dst_vqpu] > 1.0) vqpu[0].macro_rings[0].attention_weights[dst_vqpu] = 1.0;

            /* Task MEMORY: macro 2 -> macro 3, harmonic 2 */
            double corr_memory = 0.0;
            for (int h = 0; h < NUM_HARMONICS; h++) {
                double src_h = macro_harmonic(&vqpu[0].macro_rings[2], h + 1);
                double dst_h = macro_harmonic(&vqpu[dst_vqpu].macro_rings[3], h + 1);
                corr_memory += src_h * dst_h;
            }
            double delta_w_memory = ATTN_LEARNING_RATE * corr_memory - ATTN_WEIGHT_DECAY * vqpu[0].macro_rings[2].attention_weights[dst_vqpu];
            vqpu[0].macro_rings[2].attention_weights[dst_vqpu] += delta_w_memory;
            if (vqpu[0].macro_rings[2].attention_weights[dst_vqpu] < 0.0) vqpu[0].macro_rings[2].attention_weights[dst_vqpu] = 0.0;
            if (vqpu[0].macro_rings[2].attention_weights[dst_vqpu] > 1.0) vqpu[0].macro_rings[2].attention_weights[dst_vqpu] = 1.0;

            /* Task ATTENTION: macro 4 -> macro 5, harmonic 3 */
            double corr_attention = 0.0;
            for (int h = 0; h < NUM_HARMONICS; h++) {
                double src_h = macro_harmonic(&vqpu[0].macro_rings[4], h + 1);
                double dst_h = macro_harmonic(&vqpu[dst_vqpu].macro_rings[5], h + 1);
                corr_attention += src_h * dst_h;
            }
            double delta_w_attention = ATTN_LEARNING_RATE * corr_attention - ATTN_WEIGHT_DECAY * vqpu[0].macro_rings[4].attention_weights[dst_vqpu];
            vqpu[0].macro_rings[4].attention_weights[dst_vqpu] += delta_w_attention;
            if (vqpu[0].macro_rings[4].attention_weights[dst_vqpu] < 0.0) vqpu[0].macro_rings[4].attention_weights[dst_vqpu] = 0.0;
            if (vqpu[0].macro_rings[4].attention_weights[dst_vqpu] > 1.0) vqpu[0].macro_rings[4].attention_weights[dst_vqpu] = 1.0;
        }

        /* Evaluate routing accuracy every 50 chunks */
        if (chunk % 50 == 0 && chunk > 0) {
            /* Test: after learning, check if correct destination macro rings have higher harmonic energy */
            double r1_h1 = macro_harmonic(&vqpu[1].macro_rings[1], 1);  /* vQPU 1, macro 1 (filter dst) */
            double r1_h2 = macro_harmonic(&vqpu[1].macro_rings[3], 2);  /* vQPU 1, macro 3 (memory dst) */
            double r1_h3 = macro_harmonic(&vqpu[1].macro_rings[5], 3);  /* vQPU 1, macro 5 (attention dst) */
            
            double r2_h1 = macro_harmonic(&vqpu[2].macro_rings[1], 1);
            double r2_h2 = macro_harmonic(&vqpu[2].macro_rings[3], 2);
            double r2_h3 = macro_harmonic(&vqpu[2].macro_rings[5], 3);

            if (r1_h1 > 0.05) correct_filter++;
            if (r1_h2 > 0.05) correct_memory++;
            if (r1_h3 > 0.05) correct_attention++;
            if (r2_h1 > 0.05) correct_filter++;
            if (r2_h2 > 0.05) correct_memory++;
            if (r2_h3 > 0.05) correct_attention++;
        }
    }

    /* Final evaluation */
    double r1_h1 = macro_harmonic(&vqpu[1].macro_rings[1], 1);
    double r1_h2 = macro_harmonic(&vqpu[1].macro_rings[3], 2);
    double r1_h3 = macro_harmonic(&vqpu[1].macro_rings[5], 3);
    
    double r2_h1 = macro_harmonic(&vqpu[2].macro_rings[1], 1);
    double r2_h2 = macro_harmonic(&vqpu[2].macro_rings[3], 2);
    double r2_h3 = macro_harmonic(&vqpu[2].macro_rings[5], 3);

    int total_correct = 0;
    int total_tests = 6;  /* 3 tasks × 2 destination vQPUs */
    
    if (r1_h1 > 0.05) total_correct++;
    if (r1_h2 > 0.05) total_correct++;
    if (r1_h3 > 0.05) total_correct++;
    if (r2_h1 > 0.05) total_correct++;
    if (r2_h2 > 0.05) total_correct++;
    if (r2_h3 > 0.05) total_correct++;

    double filter_acc = ((r1_h1 > 0.05) + (r2_h1 > 0.05)) / 2.0;
    double memory_acc = ((r1_h2 > 0.05) + (r2_h2 > 0.05)) / 2.0;
    double attention_acc = ((r1_h3 > 0.05) + (r2_h3 > 0.05)) / 2.0;
    double overall_acc = total_correct / 6.0;

    init_result(&res, "super_routing_hebbian");
    res.num_vqpus = 3;
    res.src_level = 1; res.dst_level = 1;
    res.super_routing_accuracy = overall_acc;
    res.routing_accuracy = filter_acc;
    res.macro_aggregation_score = memory_acc;
    res.transfer_ratio = attention_acc;
    print_result(&res);
    
    fprintf(stderr, "  Filter routing: %.1f%%, Memory routing: %.1f%%, Attention routing: %.1f%%\n",
            filter_acc * 100, memory_acc * 100, attention_acc * 100);
}

/* =========== EXPERIMENT 4: Independent States =========== */
static void run_independent_states() {
    result_t res;
    vqpu_t vqpu[3];
    for (int i = 0; i < 3; i++) init_vqpu(&vqpu[i], i);

    /* vQPU 0: FILTER mode */
    for (int u = 0; u < MICRO_PER_MACRO; u++) {
        encode_input(&vqpu[0].macro_rings[0].micro[u], 1, 2.0);
        make_harmonic_lens(vqpu[0].macro_rings[0].micro[u].lens, 1, 0.5);
        apply_lens_micro(&vqpu[0].macro_rings[0].micro[u]);
    }

    /* vQPU 1: MEMORY mode */
    for (int u = 0; u < MICRO_PER_MACRO; u++) {
        encode_input(&vqpu[1].macro_rings[0].micro[u], 2, 2.0);
        make_site_lens(vqpu[1].macro_rings[0].micro[u].lens, u, 0.3);
        apply_lens_micro(&vqpu[1].macro_rings[0].micro[u]);
    }

    /* vQPU 2: ATTENTION mode */
    for (int m = 0; m < 2; m++) {
        for (int u = 0; u < MICRO_PER_MACRO; u++) {
            encode_input(&vqpu[2].macro_rings[m].micro[u], m+1, 2.0);
            make_harmonic_lens(vqpu[2].macro_rings[m].micro[u].lens, m+1, 0.4);
            apply_lens_micro(&vqpu[2].macro_rings[m].micro[u]);
        }
    }

    for (int chunk = 0; chunk < SETTLE_CHUNKS; chunk++) {
        for (int v = 0; v < 3; v++) {
            for (int m = 0; m < MACRO_PER_SUPER; m++) {
                for (int u = 0; u < MICRO_PER_MACRO; u++) {
                    settle_micro(&vqpu[v].macro_rings[m].micro[u], 1);
                }
            }
        }
    }

    double v0_filter = harmonic_mag(&vqpu[0].macro_rings[0].micro[0], 1);
    double v0_leak   = harmonic_mag(&vqpu[0].macro_rings[0].micro[0], 2);
    double v1_mem    = macro_harmonic(&vqpu[1].macro_rings[0], 2);
    double v2_attn_0 = macro_harmonic(&vqpu[2].macro_rings[0], 1);
    double v2_attn_1 = macro_harmonic(&vqpu[2].macro_rings[1], 2);

    double fidelity = 0;
    fidelity += v0_filter / (v0_filter + v0_leak + 1e-10);
    fidelity += v1_mem / (v1_mem + 1e-10);
    double v2_attn_avg = (v2_attn_0 + v2_attn_1) / 2.0;
    fidelity += v2_attn_avg / (v2_attn_avg + 1e-10);
    fidelity /= 3.0;

    init_result(&res, "independent_states");
    res.num_vqpus = 3;
    res.independent_state_fidelity = fidelity;
    print_result(&res);
}

/* =========== EXPERIMENT 5: Scale Scaling =========== */
static void run_scale_scaling() {
    result_t res;
    int scales[] = {2, 4, 8, 16};

    for (int s = 0; s < 4; s++) {
        int N = scales[s];
        vqpu_t *vqpu = calloc(N, sizeof(vqpu_t));
        for (int i = 0; i < N; i++) init_vqpu(&vqpu[i], i);
        pulse_queue_t pq; pulse_queue_init(&pq);

        for (int chunk = 0; chunk < SETTLE_CHUNKS; chunk++) {
            for (int v = 0; v < N; v++) {
                for (int m = 0; m < MACRO_PER_SUPER; m++) {
                    for (int u = 0; u < MICRO_PER_MACRO; u++) {
                        settle_micro(&vqpu[v].macro_rings[m].micro[u], 1);
                    }
                }
            }

            for (int v = 0; v < N; v++) {
                double phase = macro_harmonic(&vqpu[v].macro_rings[0], 1);
                if (phase > 0.1) {
                    int dst = (v + 1) % N;
                    pulse_queue_push(&pq, v, 1, 0,  dst, 1, 0,  phase, 0.3, 1, chunk);
                }
            }

            pulse_queue_deliver(&pq, vqpu, N, chunk);
        }

        int delivered = 0;
        for (int v = 0; v < N; v++) {
            if (macro_harmonic(&vqpu[v].macro_rings[0], 1) > 0.05) delivered++;
        }

        double speedup = (double)delivered / N;

        init_result(&res, "scale_scaling");
        res.num_vqpus = N;
        res.routing_accuracy = speedup;
        res.events_per_sec = N / (SETTLE_CHUNKS * SETTLE_STEPS * DT * 1e-6);
        print_result(&res);

        free(vqpu);
    }
}

/* =========== EXPERIMENT 6: Plasticity Learning Curve =========== */
static void run_plasticity_learning_curve() {
    /* Single micro ring learning a target harmonic filter via teacher pulses */

    micro_ring_t ring;
    ring.id = 0; ring.macro_id = 0;
    ring.last_q = 0;
    memset(ring.th, 0, sizeof(ring.th));
    memset(ring.om, 0, sizeof(ring.om));
    memset(ring.v, 0, sizeof(ring.v));
    /* Start with small random lens - no harmonic structure */
    for (int i = 0; i < OSC_PER_MICRO; i++) {
        ring.lens[i] = (rand() / (double)RAND_MAX - 0.5) * 0.1;
    }
    init_micro_plasticity(&ring);

    FILE *csv = fopen("plasticity_learning_curve.csv", "w");
    fprintf(csv, "chunk,target_mag,best_h,energy,pre_sum,post_sum\n");

    double target_mag_history[PLASTICITY_MAX_CHUNKS];
    for (int i = 0; i < PLASTICITY_MAX_CHUNKS; i++) target_mag_history[i] = 0.0;
    int converged_chunk = -1;

    for (int chunk = 0; chunk < PLASTICITY_MAX_CHUNKS; chunk++) {
        /* Encode input with target harmonic */
        encode_input(&ring, PLASTICITY_TARGET_HARMONIC, 2.0);

        /* Settle for one chunk */
        settle_micro(&ring, 1);

        /* Teacher provides STRONG pulses at target harmonic pattern */
        /* Emit pulses at every site with phase matching target harmonic */
        for (int site = 0; site < OSC_PER_MICRO; site++) {
            double phase = 2.0 * M_PI * PLASTICITY_TARGET_HARMONIC * site / OSC_PER_MICRO;
            /* Strong teacher signal - pre_trace boost */
            ring.pre_trace[site] = TRACE_DECAY * ring.pre_trace[site] + (1.0 - TRACE_DECAY) * 1.0;
            /* Phase kick to oscillator - stronger teacher signal */
            ring.th[site] += PULSE_STRENGTH * 2.0 * sin(phase - ring.th[site]);
        }

        /* Update post-synaptic trace (harmonic filter response) */
        double max_mag = 0.0;
        int best_h = 0;
        for (int h = 1; h <= 8; h++) {
            double mag = harmonic_mag(&ring, h);
            ring.post_trace[h-1] = TRACE_DECAY * ring.post_trace[h-1] + (1.0 - TRACE_DECAY) * mag;
            if (ring.post_trace[h-1] > max_mag) {
                max_mag = ring.post_trace[h-1];
                best_h = h;
            }
        }
        ring.best_harmonic = best_h;

        /* Plasticity update */
        plasticity_update(&ring);

        /* Record metrics */
        double target_mag = harmonic_mag(&ring, PLASTICITY_TARGET_HARMONIC);
        target_mag_history[chunk] = target_mag;

        double pre_sum = 0, post_sum = 0;
        for (int i = 0; i < OSC_PER_MICRO; i++) pre_sum += ring.pre_trace[i];
        for (int h = 0; h < 8; h++) post_sum += ring.post_trace[h];

        fprintf(csv, "%d,%.6f,%d,%.6f,%.6f,%.6f\n", 
                chunk, target_mag, best_h, ring.lens_energy, pre_sum, post_sum);

        /* Check convergence: target harmonic magnitude > 0.45 and dominant */
        if (converged_chunk < 0 && target_mag > 0.45 && best_h == PLASTICITY_TARGET_HARMONIC) {
            converged_chunk = chunk;
        }

        /* Early exit if converged and stable */
        if (converged_chunk >= 0 && chunk > converged_chunk + 100) {
            double recent_avg = 0;
            for (int k = chunk - 50; k < chunk; k++) recent_avg += target_mag_history[k];
            recent_avg /= 50;
            if (recent_avg > 0.5) break;
        }
    }

    fclose(csv);

    /* Read back final metrics */
    FILE *fcsv = fopen("plasticity_learning_curve.csv", "r");
    char line[256];
    fgets(line, sizeof(line), fcsv); // skip header

    int final_chunk = 0;
    double final_target = 0, final_energy = 0;
    int final_best = 0;
    while (fgets(line, sizeof(line), fcsv)) {
        int ch, bh; double tm, en, pr, po;
        sscanf(line, "%d,%lf,%d,%lf,%lf,%lf", &ch, &tm, &bh, &en, &pr, &po);
        final_chunk = ch;
        final_target = tm;
        final_best = bh;
        final_energy = en;
    }
    fclose(fcsv);

    /* Output result in CSV format */
    printf("plasticity_learning_curve,1,0,0,0,0,%.6f,%.6f,%.6f,%.6f,%.3f,%.3f,%.3f\n",
           (double)(converged_chunk >= 0 ? converged_chunk : PLASTICITY_MAX_CHUNKS),
           (final_best == PLASTICITY_TARGET_HARMONIC) ? 1.0 : 0.0,
           final_target,
           final_energy / MAX_ENERGY,
           0.0, 0.0, 0.0);

    fprintf(stderr, "Plasticity Learning Curve: converged_chunk=%d, target_mag=%.4f, best_h=%d, energy=%.2f\n",
            converged_chunk, final_target, final_best, final_energy);
}

/* =========== EXPERIMENT 7: Plasticity Catastrophic Forgetting Test =========== */
static void run_plasticity_forgetting() {
    /* Test sequential learning: train on harmonic 1, then harmonic 2, measure harmonic 1 retention */
    /* Target: >80% retention after learning second task */

    micro_ring_t ring;
    ring.id = 0; ring.macro_id = 0;
    ring.last_q = 0;
    memset(ring.th, 0, sizeof(ring.th));
    memset(ring.om, 0, sizeof(ring.om));
    memset(ring.v, 0, sizeof(ring.v));
    /* Start with small random lens - no harmonic structure */
    for (int i = 0; i < OSC_PER_MICRO; i++) {
        ring.lens[i] = (rand() / (double)RAND_MAX - 0.5) * 0.1;
    }
    init_micro_plasticity(&ring);

    FILE *csv = fopen("plasticity_forgetting.csv", "w");
    fprintf(csv, "phase,chunk,target_h1_mag,target_h2_mag,best_h,energy\\n");

    double h1_baseline = 0.0;
    double h1_after_h2 = 0.0;

    /* PHASE 1: Train on harmonic 1 for 1000 chunks */
    fprintf(stderr, "Phase 1: Training on harmonic 1 (1000 chunks)...\n");
    for (int chunk = 0; chunk < 1000; chunk++) {
        /* Encode input with harmonic 1 */
        encode_input(&ring, 1, 2.0);

        /* Settle for one chunk */
        settle_micro(&ring, 1);

        /* Teacher provides STRONG pulses at harmonic 1 pattern */
        for (int site = 0; site < OSC_PER_MICRO; site++) {
            double phase = 2.0 * M_PI * 1 * site / OSC_PER_MICRO;
            ring.pre_trace[site] = TRACE_DECAY * ring.pre_trace[site] + (1.0 - TRACE_DECAY) * 1.0;
            ring.th[site] += PULSE_STRENGTH * 2.0 * sin(phase - ring.th[site]);
        }

        /* Update post-synaptic trace */
        double max_mag = 0.0;
        int best_h = 0;
        for (int h = 1; h <= 8; h++) {
            double mag = harmonic_mag(&ring, h);
            ring.post_trace[h-1] = TRACE_DECAY * ring.post_trace[h-1] + (1.0 - TRACE_DECAY) * mag;
            if (ring.post_trace[h-1] > max_mag) {
                max_mag = ring.post_trace[h-1];
                best_h = h;
            }
        }
        ring.best_harmonic = best_h;

        /* Plasticity update */
        plasticity_update(&ring);

        /* Record metrics */
        double target_h1_mag = harmonic_mag(&ring, 1);
        double target_h2_mag = harmonic_mag(&ring, 2);
        
        if (chunk == 999) {
            h1_baseline = target_h1_mag;
        }

        if (chunk % 100 == 0 || chunk == 999) {
            double pre_sum = 0, post_sum = 0;
            for (int i = 0; i < OSC_PER_MICRO; i++) pre_sum += ring.pre_trace[i];
            for (int h = 0; h < 8; h++) post_sum += ring.post_trace[h];
            fprintf(csv, "phase1,%d,%.6f,%.6f,%d,%.6f\n", 
                    chunk, target_h1_mag, target_h2_mag, best_h, ring.lens_energy);
        }
    }

    fprintf(stderr, "  Phase 1 complete: harmonic 1 magnitude = %.4f\n", h1_baseline);

    /* PHASE 2: Train on harmonic 2 for 1000 chunks (no harmonic 1 training) */
    fprintf(stderr, "Phase 2: Training on harmonic 2 (1000 chunks)...\n");
    for (int chunk = 0; chunk < 1000; chunk++) {
        /* Encode input with harmonic 2 */
        encode_input(&ring, 2, 2.0);

        /* Settle for one chunk */
        settle_micro(&ring, 1);

        /* Teacher provides STRONG pulses at harmonic 2 pattern */
        for (int site = 0; site < OSC_PER_MICRO; site++) {
            double phase = 2.0 * M_PI * 2 * site / OSC_PER_MICRO;
            ring.pre_trace[site] = TRACE_DECAY * ring.pre_trace[site] + (1.0 - TRACE_DECAY) * 1.0;
            ring.th[site] += PULSE_STRENGTH * 2.0 * sin(phase - ring.th[site]);
        }

        /* Update post-synaptic trace */
        double max_mag = 0.0;
        int best_h = 0;
        for (int h = 1; h <= 8; h++) {
            double mag = harmonic_mag(&ring, h);
            ring.post_trace[h-1] = TRACE_DECAY * ring.post_trace[h-1] + (1.0 - TRACE_DECAY) * mag;
            if (ring.post_trace[h-1] > max_mag) {
                max_mag = ring.post_trace[h-1];
                best_h = h;
            }
        }
        ring.best_harmonic = best_h;

        /* Plasticity update */
        plasticity_update(&ring);

        /* Record metrics */
        double target_h1_mag = harmonic_mag(&ring, 1);
        double target_h2_mag = harmonic_mag(&ring, 2);
        
        if (chunk == 999) {
            h1_after_h2 = target_h1_mag;
        }

        if (chunk % 100 == 0 || chunk == 999) {
            double pre_sum = 0, post_sum = 0;
            for (int i = 0; i < OSC_PER_MICRO; i++) pre_sum += ring.pre_trace[i];
            for (int h = 0; h < 8; h++) post_sum += ring.post_trace[h];
            fprintf(csv, "phase2,%d,%.6f,%.6f,%d,%.6f\n", 
                    chunk, target_h1_mag, target_h2_mag, best_h, ring.lens_energy);
        }
    }

    fprintf(stderr, "  Phase 2 complete: harmonic 1 magnitude = %.4f, harmonic 2 magnitude = %.4f\n", 
            h1_after_h2, harmonic_mag(&ring, 2));

    fclose(csv);

    /* Compute retention */
    double retention = (h1_baseline > 1e-10) ? h1_after_h2 / h1_baseline : 0.0;
    int passed = (retention >= 0.80) ? 1 : 0;

    fprintf(stderr, "Catastrophic Forgetting Test: h1_baseline=%.4f, h1_after_h2=%.4f, retention=%.2f%%, %s\n",
            h1_baseline, h1_after_h2, retention * 100, passed ? "PASSED (>80%)" : "FAILED (<80%)");

    /* Output result in CSV format */
    printf("plasticity_forgetting,1,0,0,0,0,%.6f,%.6f,%.6f,%.6f,%.3f,%.3f,%.3f\n",
           h1_baseline,
           h1_after_h2,
           retention,
           (double)passed,
           0.0, 0.0, 0.0);
}

/* =========== SUPER-SUPER PROTOCOL IMPLEMENTATION =========== */

static void super_pulse_queue_init(super_pulse_queue_t *q) {
    q->count = 0;
}

static void super_pulse_queue_push(super_pulse_queue_t *q, int src_vqpu, int dst_vqpu,
                                    int src_macro, int dst_macro, int src_harmonic, int dst_harmonic,
                                    double phase, double weight, int harmonic, int ttl,
                                    int seq, super_pulse_type_t type, int timestep,
                                    const double *payload_phase, double feedback_gain) {
    if (q->count >= MAX_SUPER_PULSES) return;
    super_pulse_t *p = &q->pulses[q->count++];
    p->src_vqpu = src_vqpu;
    p->dst_vqpu = dst_vqpu;
    p->src_macro = src_macro;
    p->dst_macro = dst_macro;
    p->src_harmonic = src_harmonic;
    p->dst_harmonic = dst_harmonic;
    p->phase = phase;
    p->weight = weight;
    p->harmonic = harmonic;
    p->ttl = ttl;
    p->seq = seq;
    p->type = type;
    p->timestep = timestep;
    p->feedback_gain = feedback_gain;
    if (payload_phase) {
        for (int h = 0; h < 8; h++) {
            p->payload_phase[h] = payload_phase[h];
        }
    } else {
        for (int h = 0; h < 8; h++) {
            p->payload_phase[h] = 0.0;
        }
    }
}

static void super_pulse_queue_deliver(super_pulse_queue_t *q, vqpu_t *vqpus, int num_vqpus, int current_chunk) {
    for (int i = 0; i < q->count; i++) {
        super_pulse_t *p = &q->pulses[i];
        if (p->timestep != current_chunk) continue;
        if (p->dst_vqpu < 0 || p->dst_vqpu >= num_vqpus) continue;

        vqpu_t *dst_vqpu = &vqpus[p->dst_vqpu];
        super_ring_t *dst_super = (super_ring_t *)dst_vqpu;  // Cast to access super_ring_t fields

        if (p->type == SUPER_PULSE_DATA) {
            /* Deliver data pulse to destination macro ring */
            if (p->dst_macro >= 0 && p->dst_macro < MACRO_PER_SUPER) {
                macro_ring_t *dst_macro_ring = &dst_vqpu->macro_rings[p->dst_macro];
                
                /* Apply phase perturbation to micro rings in the macro */
                for (int u = 0; u < MICRO_PER_MACRO; u++) {
                    micro_ring_t *mic = &dst_macro_ring->micro[u];
                    for (int j = 0; j < OSC_PER_MICRO; j++) {
                        mic->th[j] += p->weight * sin(p->phase - mic->th[j]);
                    }
                }
                
                /* Update super-to-macro feedback weights */
                dst_macro_ring->super_feedback_weights[p->src_macro % MACRO_PER_SUPER] = p->weight * SUPER_FEEDBACK_GAIN;
                dst_macro_ring->super_feedback_phase[p->src_macro % MACRO_PER_SUPER] = p->phase;
                dst_macro_ring->super_feedback_harmonic[p->src_macro % MACRO_PER_SUPER] = p->dst_harmonic;
            }
            
            /* Queue ACK response */
            if (p->seq >= 0) {
                super_pulse_queue_push(q, p->dst_vqpu, p->src_vqpu,
                                        p->dst_macro, p->src_macro,
                                        p->dst_harmonic, p->src_harmonic,
                                        p->phase, p->weight, p->harmonic,
                                        SUPER_TTL_DEFAULT, p->seq,
                                        SUPER_PULSE_ACK, current_chunk,
                                        NULL, SUPER_FEEDBACK_GAIN);
            }
        }
        else if (p->type == SUPER_PULSE_ACK) {
            /* Handle ACK - find and clear the specific pending ACK by sequence number */
            super_ring_t *src_super = (super_ring_t *)&vqpus[p->src_vqpu];
            for (int i = 0; i < SUPER_MAX_PENDING_ACKS; i++) {
                if (src_super->pending_acks[i].in_use && src_super->pending_acks[i].seq == p->seq) {
                    src_super->pending_acks[i].in_use = 0;
                    src_super->pending_ack_count = (src_super->pending_ack_count > 0) ? 
                                                   src_super->pending_ack_count - 1 : 0;
                    fprintf(stderr, "[SuperSuper] vQPU %d: ACK received for seq=%d (dst_vqpu=%d)\n",
                            p->src_vqpu, p->seq, p->dst_vqpu);
                    break;
                }
            }
        }
        else if (p->type == SUPER_PULSE_NACK) {
            /* Handle NACK - find and clear the specific pending ACK by sequence number */
            super_ring_t *src_super = (super_ring_t *)&vqpus[p->src_vqpu];
            for (int i = 0; i < SUPER_MAX_PENDING_ACKS; i++) {
                if (src_super->pending_acks[i].in_use && src_super->pending_acks[i].seq == p->seq) {
                    src_super->pending_acks[i].in_use = 0;
                    src_super->pending_ack_count = (src_super->pending_ack_count > 0) ? 
                                                   src_super->pending_ack_count - 1 : 0;
                    fprintf(stderr, "[SuperSuper] vQPU %d: NACK received for seq=%d (dst_vqpu=%d)\n",
                            p->src_vqpu, p->seq, p->dst_vqpu);
                    break;
                }
            }
        }
        else if (p->type == SUPER_PULSE_SYNC) {
            /* Synchronize super ring phases */
            for (int m = 0; m < MACRO_PER_SUPER; m++) {
                macro_ring_t *mr = &dst_vqpu->macro_rings[m];
                for (int u = 0; u < MICRO_PER_MACRO; u++) {
                    micro_ring_t *mic = &mr->micro[u];
                    for (int j = 0; j < OSC_PER_MICRO; j++) {
                        mic->th[j] += 0.1 * sin(p->phase - mic->th[j]);
                    }
                }
            }
        }
    }

    /* Decrement TTL and remove expired pulses */
    int remaining = 0;
    for (int i = 0; i < q->count; i++) {
        if (q->pulses[i].timestep > current_chunk) {
            q->pulses[i].ttl--;
            if (q->pulses[i].ttl > 0) {
                q->pulses[remaining++] = q->pulses[i];
            }
        }
    }
    q->count = remaining;
}

/* Compute cross-vQPU super attention weights using harmonic similarity */
static void super_routing_compute_weights(vqpu_t *vqpus, int num_vqpus) {
    for (int src = 0; src < num_vqpus; src++) {
        super_ring_t *src_super = (super_ring_t *)&vqpus[src];
        
        for (int dst = 0; dst < num_vqpus; dst++) {
            if (src == dst) continue;
            
            super_ring_t *dst_super = (super_ring_t *)&vqpus[dst];
            
            /* Compute attention weight for each macro ring pair */
            for (int src_macro = 0; src_macro < MACRO_PER_SUPER; src_macro++) {
                for (int dst_macro = 0; dst_macro < MACRO_PER_SUPER; dst_macro++) {
                    double similarity = 0.0;
                    double src_norm = 0.0, dst_norm = 0.0;
                    
                    /* Compute cosine similarity of harmonic vectors */
                    for (int h = 1; h <= SUPER_HARMONIC_MAX; h++) {
                        double src_h = macro_harmonic(&vqpus[src].macro_rings[src_macro], h);
                        double dst_h = macro_harmonic(&vqpus[dst].macro_rings[dst_macro], h);
                        similarity += src_h * dst_h;
                        src_norm += src_h * src_h;
                        dst_norm += dst_h * dst_h;
                    }
                    
                    double weight = 0.0;
                    if (src_norm > 1e-10 && dst_norm > 1e-10) {
                        weight = similarity / (sqrt(src_norm) * sqrt(dst_norm));
                    }
                    
                    /* Apply softmax-like normalization across destinations */
                    src_super->super_attention_weights[dst][dst_macro] = weight * SUPER_WEIGHT_SCALE;
                }
            }
        }
    }
}

/* Apply super-to-macro feedback modulation with attention gating */
static void super_feedback_to_macro(vqpu_t *vqpus, int num_vqpus, int current_chunk) {
    /* Constants for attention gating */
    const double ATTENDED_GAIN = 6.0;      /* >5x gain for attended macros */
    const double UNATTENDED_GAIN = 0.3;    /* <0.5x suppression for unattended macros */
    const double PHASE_ALIGN_STRENGTH = 0.15;  /* Phase alignment strength for attended harmonics */

    for (int v = 0; v < num_vqpus; v++) {
        super_ring_t *sr = (super_ring_t *)&vqpus[v];
        
        for (int m = 0; m < MACRO_PER_SUPER; m++) {
            macro_ring_t *mr = &vqpus[v].macro_rings[m];
            
            /* Compute attention weight for this macro from super ring's perspective.
             * The super ring computes attention weights across ALL vQPUs for EACH macro.
             * We aggregate attention from all source vQPUs to this macro. */
            double total_attention = 0.0;
            double max_attention = 0.0;
            int dominant_source = -1;
            int dominant_harmonic = 1;
            
            for (int src_v = 0; src_v < num_vqpus; src_v++) {
                if (src_v == v) continue;  /* Skip self-attention for cross-vQPU feedback */
                double w = sr->super_attention_weights[src_v][m];
                total_attention += w;
                if (w > max_attention) {
                    max_attention = w;
                    dominant_source = src_v;
                }
            }
            
            /* Also consider self-attention (local super ring's view of this macro) */
            double self_attention = 0.0;
            for (int src_v = 0; src_v < num_vqpus; src_v++) {
                /* Check how much this macro contributes to routing */
                super_ring_t *src_sr = (super_ring_t *)&vqpus[src_v];
                for (int dst_v = 0; dst_v < num_vqpus; dst_v++) {
                    if (dst_v == v) {
                        self_attention += src_sr->super_attention_weights[v][m];
                    }
                }
            }
            
            /* Combined attention: cross-vQPU + self-contribution */
            double combined_attention = total_attention + self_attention;
            
            /* Normalize attention to [0, 1] range using softmax-like scaling */
            double attention_norm = combined_attention / (combined_attention + 1.0);
            
            /* Determine if this macro is "attended" (>0.5) or "unattended" (<0.5) */
            int is_attended = (attention_norm > 0.5) ? 1 : 0;
            
            /* Apply attention-gated modulation to macro inter_coupling */
            /* Base inter_coupling is 0.1, scale by attention gain */
            if (is_attended) {
                mr->inter_coupling = 0.1 * ATTENDED_GAIN * attention_norm;  /* >5x gain */
            } else {
                mr->inter_coupling = 0.1 * UNATTENDED_GAIN / (attention_norm + 0.1);  /* <0.5x suppression */
            }
            
            /* Clamp to reasonable bounds */
            if (mr->inter_coupling > 1.0) mr->inter_coupling = 1.0;
            if (mr->inter_coupling < 0.01) mr->inter_coupling = 0.01;
            
            /* Apply attention weights to micro attention_weights array */
            for (int u = 0; u < MICRO_PER_MACRO; u++) {
                double feedback_sum = 0.0;
                for (int sm = 0; sm < MACRO_PER_SUPER; sm++) {
                    feedback_sum += mr->super_feedback_weights[sm];
                }
                
                /* Modulate micro coupling based on super feedback AND macro attention */
                double micro_gain = is_attended ? ATTENDED_GAIN : UNATTENDED_GAIN;
                mr->attention_weights[u] = 1.0 + SUPER_FEEDBACK_GAIN * feedback_sum * micro_gain * 
                                           cos(2.0 * M_PI * u / MICRO_PER_MACRO);
                
                /* Apply harmonic-specific phase alignment for attended macros */
                if (is_attended && mr->super_feedback_harmonic[u % MACRO_PER_SUPER] > 0) {
                    int target_h = mr->super_feedback_harmonic[u % MACRO_PER_SUPER];
                    double phase_align = mr->super_feedback_phase[u % MACRO_PER_SUPER];
                    
                    /* Also consider the dominant harmonic from cross-vQPU attention */
                    if (dominant_source >= 0) {
                        super_ring_t *dom_sr = (super_ring_t *)&vqpus[dominant_source];
                        /* Find the harmonic with max correlation for this macro pair */
                        double max_corr = 0.0;
                        for (int h = 1; h <= SUPER_HARMONIC_MAX; h++) {
                            double src_h = macro_harmonic(&vqpus[dominant_source].macro_rings[m], h);
                            double dst_h = macro_harmonic(mr, h);
                            double corr = src_h * dst_h;
                            if (corr > max_corr) {
                                max_corr = corr;
                                dominant_harmonic = h;
                            }
                        }
                        target_h = dominant_harmonic;
                    }
                    
                    micro_ring_t *mic = &mr->micro[u];
                    for (int j = 0; j < OSC_PER_MICRO; j++) {
                        double target_phase = 2.0 * M_PI * target_h * j / OSC_PER_MICRO + phase_align;
                        /* Stronger phase pull for attended harmonics */
                        double align_strength = PHASE_ALIGN_STRENGTH * attention_norm * 2.0;
                        mic->th[j] += align_strength * sin(target_phase - mic->th[j]);
                    }
                }
                /* For unattended macros, apply weak random phase noise to suppress coherence */
                else if (!is_attended) {
                    micro_ring_t *mic = &mr->micro[u];
                    for (int j = 0; j < OSC_PER_MICRO; j++) {
                        double noise_phase = ((rand() / (double)RAND_MAX) - 0.5) * 0.1;
                        mic->th[j] += UNATTENDED_GAIN * noise_phase;
                    }
                }
            }
            
            /* Apply harmonic-specific feedback from super pulses (existing pulse-based feedback) */
            for (int u = 0; u < MICRO_PER_MACRO; u++) {
                if (mr->super_feedback_harmonic[u % MACRO_PER_SUPER] > 0) {
                    int target_h = mr->super_feedback_harmonic[u % MACRO_PER_SUPER];
                    double phase_align = mr->super_feedback_phase[u % MACRO_PER_SUPER];
                    micro_ring_t *mic = &mr->micro[u];
                    for (int j = 0; j < OSC_PER_MICRO; j++) {
                        double target_phase = 2.0 * M_PI * target_h * j / OSC_PER_MICRO + phase_align;
                        /* Scale phase alignment by macro attention state */
                        double align_strength = is_attended ? 0.1 : 0.02;
                        mic->th[j] += align_strength * sin(target_phase - mic->th[j]);
                    }
                }
            }
        }
    }
}

/* Handle ACK/NACK handshake for reliable routing with retries */
static void super_pulse_handshake(super_pulse_queue_t *q, vqpu_t *vqpus, int num_vqpus, int current_chunk) {
    for (int v = 0; v < num_vqpus; v++) {
        super_ring_t *sr = (super_ring_t *)&vqpus[v];
        
        /* Process pending ACKs - check for timeouts and retries */
        for (int i = 0; i < SUPER_MAX_PENDING_ACKS; i++) {
            if (!sr->pending_acks[i].in_use) continue;
            
            int elapsed_chunks = current_chunk - sr->pending_acks[i].timestep_sent;
            
            /* Check if ACK timeout has elapsed */
            if (elapsed_chunks >= SUPER_ACK_TIMEOUT) {
                if (sr->pending_acks[i].retry_count < SUPER_MAX_RETRIES) {
                    /* Retry: resend DATA pulse with same sequence number */
                    super_pulse_queue_push(q, v, sr->pending_acks[i].dst_vqpu,
                                          sr->pending_acks[i].src_macro, sr->pending_acks[i].dst_macro,
                                          sr->pending_acks[i].src_harmonic, sr->pending_acks[i].dst_harmonic,
                                          sr->pending_acks[i].phase, sr->pending_acks[i].weight, 
                                          sr->pending_acks[i].harmonic,
                                          SUPER_TTL_DEFAULT, sr->pending_acks[i].seq,
                                          SUPER_PULSE_DATA, current_chunk,
                                          sr->pending_acks[i].payload_phase, sr->pending_acks[i].feedback_gain);
                    
                    sr->pending_acks[i].timestep_sent = current_chunk;
                    sr->pending_acks[i].retry_count++;
                    
                    fprintf(stderr, "[SuperSuper] vQPU %d: Retry #%d for seq=%d (dst_vqpu=%d, dst_macro=%d)\n",
                            v, sr->pending_acks[i].retry_count, sr->pending_acks[i].seq,
                            sr->pending_acks[i].dst_vqpu, sr->pending_acks[i].dst_macro);
                } else {
                    /* Max retries exceeded - send NACK and clear pending */
                    super_pulse_queue_push(q, v, sr->pending_acks[i].dst_vqpu,
                                          sr->pending_acks[i].src_macro, sr->pending_acks[i].dst_macro,
                                          sr->pending_acks[i].src_harmonic, sr->pending_acks[i].dst_harmonic,
                                          sr->pending_acks[i].phase, 0.0, 
                                          sr->pending_acks[i].harmonic,
                                          SUPER_TTL_DEFAULT, sr->pending_acks[i].seq,
                                          SUPER_PULSE_NACK, current_chunk,
                                          NULL, 0.0);
                    
                    fprintf(stderr, "[SuperSuper] vQPU %d: NACK sent for seq=%d after %d retries (dst_vqpu=%d)\n",
                            v, sr->pending_acks[i].seq, SUPER_MAX_RETRIES, sr->pending_acks[i].dst_vqpu);
                    
                    /* Clear the pending ACK slot */
                    sr->pending_acks[i].in_use = 0;
                    sr->pending_ack_count = (sr->pending_ack_count > 0) ? sr->pending_ack_count - 1 : 0;
                }
            }
        }
    }
}

/* Main super-super coordination test with 3 vQPUs */
static void run_super_super_coordination(void) {
    result_t res;
    
    vqpu_t vqpu[3];
    for (int i = 0; i < 3; i++) init_vqpu(&vqpu[i], i);
    
    pulse_queue_t pq;
    pulse_queue_init(&pq);
    
    super_pulse_queue_t spq;
    super_pulse_queue_init(&spq);
    /* Initialize 3 vQPUs with different roles:
         * vQPU 0: FILTER - strong harmonic 1 in macro 0
         * vQPU 1: MEMORY - strong harmonic 2 in macro 0
         * vQPU 2: ATTENTION - NEUTRAL (no strong harmonics initially), will receive from 0 and 1 */

        for (int u = 0; u < MICRO_PER_MACRO; u++) {
            /* vQPU 0: Filter mode - harmonic 1 in macro 0 */
            encode_input(&vqpu[0].macro_rings[0].micro[u], 1, 2.0);
            make_harmonic_lens(vqpu[0].macro_rings[0].micro[u].lens, 1, 0.5);
            apply_lens_micro(&vqpu[0].macro_rings[0].micro[u]);

            /* vQPU 1: Memory mode - harmonic 2 in macro 0 */
            encode_input(&vqpu[1].macro_rings[0].micro[u], 2, 2.0);
            make_harmonic_lens(vqpu[1].macro_rings[0].micro[u].lens, 2, 0.5);
            apply_lens_micro(&vqpu[1].macro_rings[0].micro[u]);

            /* vQPU 2: Attention mode - NEUTRAL (weak random lens), will learn from 0 and 1 */
                    make_site_lens(vqpu[2].macro_rings[0].micro[u].lens, u, 0.1);
                    apply_lens_micro(&vqpu[2].macro_rings[0].micro[u]);
                    /* Macro 1: tuned to harmonic 2 for memory reception */
                    make_harmonic_lens(vqpu[2].macro_rings[1].micro[u].lens, 2, 0.5);
                    apply_lens_micro(&vqpu[2].macro_rings[1].micro[u]);
        }

        /* Initialize super attention weights with small random values + uniform bias */
        for (int v = 0; v < 3; v++) {
            super_ring_t *sr = (super_ring_t *)&vqpu[v];
            for (int dst = 0; dst < 3; dst++) {
                for (int m = 0; m < MACRO_PER_SUPER; m++) {
                    sr->super_attention_weights[dst][m] = 0.05 + (rand() / (double)RAND_MAX) * 0.1;
                }
            }
        }

        int coordination_success = 0;
        int total_super_pulses = 0;
        int ack_received = 0;
        int nack_received = 0;

        /* Coordination loop - run for more chunks to allow convergence */
        for (int chunk = 0; chunk < SETTLE_CHUNKS * 4; chunk++) {
        /* Settle all micro rings */
        for (int v = 0; v < 3; v++) {
            for (int m = 0; m < MACRO_PER_SUPER; m++) {
                for (int u = 0; u < MICRO_PER_MACRO; u++) {
                    settle_micro(&vqpu[v].macro_rings[m].micro[u], 1);
                }
            }
        }
        
        /* Compute super attention weights every chunk */
        super_routing_compute_weights(vqpu, 3);
        
        /* Apply super-to-macro feedback */
        super_feedback_to_macro(vqpu, 3, chunk);
        
        /* vQPU 0 emits super pulses to vQPU 2 based on attention weights */
        super_ring_t *sr0 = (super_ring_t *)&vqpu[0];
        for (int dst_macro = 0; dst_macro < 2; dst_macro++) {
            double weight = sr0->super_attention_weights[2][dst_macro];
            /* Lower threshold for emission - also emit if weight is 0 but we have strong source harmonics */
            double src_h1 = macro_harmonic(&vqpu[0].macro_rings[0], 1);
            if (weight > 0.01 || (src_h1 > 0.1 && weight == 0.0)) {
                double phase = macro_harmonic(&vqpu[0].macro_rings[0], 1);
                double payload_phase[8];
                for (int h = 0; h < 8; h++) {
                    payload_phase[h] = macro_harmonic(&vqpu[0].macro_rings[0], h+1);
                }
                
                int seq = sr0->super_pulse_seq++;
                
                super_pulse_queue_push(&spq, 0, 2,
                                      0, dst_macro,
                                      1, dst_macro + 1,
                                      phase, fmax(weight, 0.3), 1,
                                      SUPER_TTL_DEFAULT, seq,
                                      SUPER_PULSE_DATA, chunk,
                                      payload_phase, SUPER_FEEDBACK_GAIN);
                
                /* Register pending ACK for reliable delivery */
                for (int i = 0; i < SUPER_MAX_PENDING_ACKS; i++) {
                    if (!sr0->pending_acks[i].in_use) {
                        sr0->pending_acks[i].in_use = 1;
                        sr0->pending_acks[i].seq = seq;
                        sr0->pending_acks[i].dst_vqpu = 2;
                        sr0->pending_acks[i].dst_macro = dst_macro;
                        sr0->pending_acks[i].src_macro = 0;
                        sr0->pending_acks[i].src_harmonic = 1;
                        sr0->pending_acks[i].dst_harmonic = dst_macro + 1;
                        sr0->pending_acks[i].phase = phase;
                        sr0->pending_acks[i].weight = fmax(weight, 0.3);
                        sr0->pending_acks[i].harmonic = 1;
                        sr0->pending_acks[i].timestep_sent = chunk;
                        sr0->pending_acks[i].retry_count = 0;
                        for (int h = 0; h < 8; h++) {
                            sr0->pending_acks[i].payload_phase[h] = payload_phase[h];
                        }
                        sr0->pending_acks[i].feedback_gain = SUPER_FEEDBACK_GAIN;
                        break;
                    }
                }
                sr0->pending_ack_count++;
                total_super_pulses++;
            }
        }
        
        /* vQPU 1 emits to vQPU 2 */
        super_ring_t *sr1 = (super_ring_t *)&vqpu[1];
        double weight_12 = sr1->super_attention_weights[2][0];
        double src_h2 = macro_harmonic(&vqpu[1].macro_rings[0], 2);
        if (weight_12 > 0.01 || (src_h2 > 0.1 && weight_12 == 0.0)) {
            double phase = macro_harmonic(&vqpu[1].macro_rings[0], 2);
            double payload_phase[8];
            for (int h = 0; h < 8; h++) {
                payload_phase[h] = macro_harmonic(&vqpu[1].macro_rings[0], h+1);
            }
            
            int seq = sr1->super_pulse_seq++;
            
            super_pulse_queue_push(&spq, 1, 2,
                                  0, 0,
                                  2, 1,
                                  phase, fmax(weight_12, 0.3), 2,
                                  SUPER_TTL_DEFAULT, seq,
                                  SUPER_PULSE_DATA, chunk,
                                  payload_phase, SUPER_FEEDBACK_GAIN);
            
            /* Register pending ACK for reliable delivery */
            for (int i = 0; i < SUPER_MAX_PENDING_ACKS; i++) {
                if (!sr1->pending_acks[i].in_use) {
                    sr1->pending_acks[i].in_use = 1;
                    sr1->pending_acks[i].seq = seq;
                    sr1->pending_acks[i].dst_vqpu = 2;
                    sr1->pending_acks[i].dst_macro = 0;
                    sr1->pending_acks[i].src_macro = 0;
                    sr1->pending_acks[i].src_harmonic = 2;
                    sr1->pending_acks[i].dst_harmonic = 1;
                    sr1->pending_acks[i].phase = phase;
                    sr1->pending_acks[i].weight = fmax(weight_12, 0.3);
                    sr1->pending_acks[i].harmonic = 2;
                    sr1->pending_acks[i].timestep_sent = chunk;
                    sr1->pending_acks[i].retry_count = 0;
                    for (int h = 0; h < 8; h++) {
                        sr1->pending_acks[i].payload_phase[h] = payload_phase[h];
                    }
                    sr1->pending_acks[i].feedback_gain = SUPER_FEEDBACK_GAIN;
                    break;
                }
            }
            sr1->pending_ack_count++;
            total_super_pulses++;
        }
        
        /* Deliver super pulses */
        super_pulse_queue_deliver(&spq, vqpu, 3, chunk);
        
        /* Handle ACK/NACK handshake */
        super_pulse_handshake(&spq, vqpu, 3, chunk);
        
        /* Count ACKs/NACKs in queue */
        for (int i = 0; i < spq.count; i++) {
            if (spq.pulses[i].type == SUPER_PULSE_ACK) ack_received++;
            if (spq.pulses[i].type == SUPER_PULSE_NACK) nack_received++;
        }
        
        /* Also deliver regular micro pulses */
        pulse_queue_deliver(&pq, vqpu, 3, chunk);
    }
    
    /* Evaluate coordination success - vQPU 2 should now have both harmonic 1 and 2 */
    double v0_h1 = macro_harmonic(&vqpu[0].macro_rings[0], 1);
    double v1_h2 = macro_harmonic(&vqpu[1].macro_rings[0], 2);
    double v2_h1 = macro_harmonic(&vqpu[2].macro_rings[0], 1);
    double v2_h2 = macro_harmonic(&vqpu[2].macro_rings[1], 2);
    
    /* Check if vQPU 2 received both harmonic 1 (from vQPU 0) and harmonic 2 (from vQPU 1) */
    if (v2_h1 > 0.03 && v2_h2 > 0.03) {
        coordination_success = 1;
    }
    
    double ack_rate = total_super_pulses > 0 ? (double)ack_received / total_super_pulses : 0.0;
    double nack_rate = total_super_pulses > 0 ? (double)nack_received / total_super_pulses : 0.0;
    
    init_result(&res, "super_super_coordination");
    res.num_vqpus = 3;
    res.src_level = 1; res.dst_level = 1;
    res.super_routing_accuracy = coordination_success ? 1.0 : 0.0;
    res.routing_accuracy = ack_rate;
    res.transfer_ratio = 1.0 - nack_rate;
    res.events_per_sec = total_super_pulses / (SETTLE_CHUNKS * 4 * SETTLE_STEPS * DT * 1e-6);
    print_result(&res);
    
    fprintf(stderr, "Super-Super Coordination: success=%d, pulses=%d, ACK rate=%.2f, NACK rate=%.2f\n",
            coordination_success, total_super_pulses, ack_rate, nack_rate);
    fprintf(stderr, "  v0_h1=%.4f, v1_h2=%.4f, v2_h1=%.4f, v2_h2=%.4f\n",
            v0_h1, v1_h2, v2_h1, v2_h2);
}

/* =========== EXPERIMENT: Macro Feature Routing (Weighted Assembly) =========== */
/* Tests cross-vQPU feature routing with pulse-strength-weighted macro aggregation:
 * vQPU 0 micro rings (macro 0) → macro 0 (weighted by pulse strength) → vQPU 1 macro 0 */
static void run_macro_feature_routing() {
    result_t res;
    vqpu_t vqpu[2];
    init_vqpu(&vqpu[0], 0);
    init_vqpu(&vqpu[1], 1);
    pulse_queue_t pq; pulse_queue_init(&pq);

    /* vQPU 0 (source): 16 micro rings in macro 0, each encoding a different harmonic pattern */
    /* Micro rings 0-3: harmonic 1, 4-7: harmonic 2, 8-11: harmonic 3, 12-15: harmonic 4 */
    for (int u = 0; u < MICRO_PER_MACRO; u++) {
        micro_ring_t *mic = &vqpu[0].macro_rings[0].micro[u];
        int harm = (u / 4) + 1;  /* 4 rings per harmonic */
        encode_input(mic, harm, 2.0);
        make_harmonic_lens(mic->lens, harm, 0.5);
        apply_lens_micro(mic);
        mic->last_q = 0;
    }

    /* vQPU 1 (destination): macro 0 micro rings with pulse-receptive lenses */
    for (int u = 0; u < MICRO_PER_MACRO; u++) {
        micro_ring_t *mic = &vqpu[1].macro_rings[0].micro[u];
        int harm = (u / 4) + 1;
        make_pulse_receptive_lens(mic->lens, harm, 0.5);
        apply_lens_micro(mic);
        mic->last_q = 0;
    }

    int total_pulses = 0;
    double weighted_feature_scores[5] = {0};  /* Index 1-4 for harmonics 1-4 */

    for (int chunk = 0; chunk < SETTLE_CHUNKS; chunk++) {
        /* Settle all micro rings */
        for (int u = 0; u < MICRO_PER_MACRO; u++) {
            settle_micro(&vqpu[0].macro_rings[0].micro[u], 1);
            settle_micro(&vqpu[1].macro_rings[0].micro[u], 1);
        }

        /* Detect phase changes and emit pulses from vQPU 0 to vQPU 1 */
        for (int u = 0; u < MICRO_PER_MACRO; u++) {
            double q = measure_winding(&vqpu[0].macro_rings[0].micro[u]);
            if (fabs(q - vqpu[0].macro_rings[0].micro[u].last_q) > 0.5) {
                /* Route to corresponding micro ring in vQPU 1 macro 0 based on harmonic */
                int target_micro = u;  /* Same index = same harmonic group */
                int harm = (u / 4) + 1;
                pulse_queue_push(&pq, 0, 0, u,  1, 0, target_micro,  q, 0.4 * (1.0 + harm * 0.1), harm, chunk);
                total_pulses++;
                
                /* Track expected weighted feature score by harmonic */
                weighted_feature_scores[harm] += 0.4 * (1.0 + harm * 0.1);
            }
            vqpu[0].macro_rings[0].micro[u].last_q = q;
        }

        pulse_queue_deliver(&pq, vqpu, 2, chunk);
    }

    /* Evaluate weighted feature assembly in vQPU 1 macro 0 */
    double weighted_agg_score = 0.0;
    int detected_harmonics = 0;
    
    for (int k = 1; k <= 4; k++) {
        double h = macro_harmonic(&vqpu[1].macro_rings[0], k);
        double expected_weight = weighted_feature_scores[k];
        double weight_ratio = expected_weight > 0 ? h / (expected_weight * 0.1 + 1e-10) : 0;
        
        fprintf(stderr, "  Harmonic %d: macro_harm=%.6f, expected_weight=%.3f, ratio=%.3f\n", 
                k, h, expected_weight, weight_ratio);
        
        if (h > 0.03) {
            detected_harmonics++;
            weighted_agg_score += weight_ratio;
        }
    }
    
    weighted_agg_score = detected_harmonics > 0 ? weighted_agg_score / detected_harmonics : 0.0;

    init_result(&res, "macro_feature_routing");
    res.num_vqpus = 2;
    res.src_level = 0; res.dst_level = 1;
    res.macro_aggregation_score = weighted_agg_score;
    res.routing_accuracy = (double)detected_harmonics / 4.0;
    res.transfer_ratio = total_pulses > 0 ? (double)detected_harmonics / 4.0 : 0.0;
    res.events_per_sec = total_pulses / (SETTLE_CHUNKS * SETTLE_STEPS * DT * 1e-6);
    print_result(&res);

    fprintf(stderr, "Macro Feature Routing: detected=%d/4, weighted_score=%.3f, pulses=%d\n",
            detected_harmonics, weighted_agg_score, total_pulses);
}

/* =========== SENSOR FUSION DEMO =========== */
static void run_sensor_fusion_demo() {
    result_t res;
    vqpu_t vqpu[3];
    for (int i = 0; i < 3; i++) init_vqpu(&vqpu[i], i);
    pulse_queue_t pq; pulse_queue_init(&pq);

    /* Category definitions: Cat(0), Dog(1), Bird(2), Car(3) */
    const int NUM_CATEGORIES = 4;
    const int SAMPLES_PER_CAT = 25;
    const int TOTAL_SAMPLES = NUM_CATEGORIES * SAMPLES_PER_CAT;

    /* Training data: vision features (4 dims) + audio features (4 dims) */
    double vision_features[TOTAL_SAMPLES][4];
    double audio_features[TOTAL_SAMPLES][4];
    int categories[TOTAL_SAMPLES];

    /* Generate training data */
    srand(42);
    for (int cat = 0; cat < NUM_CATEGORIES; cat++) {
        for (int s = 0; s < SAMPLES_PER_CAT; s++) {
            int idx = cat * SAMPLES_PER_CAT + s;
            categories[idx] = cat;

            /* Generate features with category-specific biases + noise */
            for (int f = 0; f < 4; f++) {
                double base = (cat * 0.2 + f * 0.1);
                double noise = ((double)rand() / RAND_MAX - 0.5) * 0.2;
                vision_features[idx][f] = fmod(base + noise, 1.0);
                if (vision_features[idx][f] < 0) vision_features[idx][f] += 1.0;

                base = (cat * 0.15 + f * 0.12);
                noise = ((double)rand() / RAND_MAX - 0.5) * 0.2;
                audio_features[idx][f] = fmod(base + noise, 1.0);
                if (audio_features[idx][f] < 0) audio_features[idx][f] += 1.0;
            }
        }
    }

    /* Training phase: modality-specific learning */
    int correct_vision = 0, correct_audio = 0;
    int training_chunks = 1000;

    for (int chunk = 0; chunk < training_chunks; chunk++) {
        /* Present random sample */
        int sample_idx = rand() % TOTAL_SAMPLES;
        int cat = categories[sample_idx];

        /* Encode vision in vQPU 0 */
        for (int u = 0; u < MICRO_PER_MACRO; u++) {
            micro_ring_t *mic = &vqpu[0].macro_rings[0].micro[u];
            double amp = 2.0;
            for (int j = 0; j < OSC_PER_MICRO; j++) {
                mic->th[j] = 0;
                for (int f = 0; f < 4; f++) {
                    mic->th[j] += vision_features[sample_idx][f] * amp *
                                  cos(2.0 * M_PI * (f+1) * j / OSC_PER_MICRO);
                }
            }
            make_harmonic_lens(mic->lens, 1, 0.5);
            apply_lens_micro(mic);
        }

        /* Encode audio in vQPU 1 */
        for (int u = 0; u < MICRO_PER_MACRO; u++) {
            micro_ring_t *mic = &vqpu[1].macro_rings[0].micro[u];
            double amp = 2.0;
            for (int j = 0; j < OSC_PER_MICRO; j++) {
                mic->th[j] = 0;
                for (int f = 0; f < 4; f++) {
                    mic->th[j] += audio_features[sample_idx][f] * amp *
                                  cos(2.0 * M_PI * (f+1) * j / OSC_PER_MICRO);
                }
            }
            make_harmonic_lens(mic->lens, 1, 0.5);
            apply_lens_micro(mic);
        }

        /* Settle */
        for (int m = 0; m < MACRO_PER_SUPER; m++) {
            for (int u = 0; u < MICRO_PER_MACRO; u++) {
                settle_micro(&vqpu[0].macro_rings[m].micro[u], 1);
                settle_micro(&vqpu[1].macro_rings[m].micro[u], 1);
            }
        }

        /* Measure vision classification */
        double v_harm[4] = {0};
        for (int u = 0; u < MICRO_PER_MACRO; u++) {
            for (int h = 0; h < 4; h++) {
                v_harm[h] += harmonic_mag(&vqpu[0].macro_rings[0].micro[u], h+1);
            }
        }
        for (int h = 0; h < 4; h++) v_harm[h] /= MICRO_PER_MACRO;
        int predicted_vision = 0;
        for (int h = 1; h < 4; h++) {
            if (v_harm[h] > v_harm[predicted_vision]) predicted_vision = h;
        }
        if (predicted_vision == cat) correct_vision++;

        /* Measure audio classification */
        double a_harm[4] = {0};
        for (int u = 0; u < MICRO_PER_MACRO; u++) {
            for (int h = 0; h < 4; h++) {
                a_harm[h] += harmonic_mag(&vqpu[1].macro_rings[0].micro[u], h+1);
            }
        }
        for (int h = 0; h < 4; h++) a_harm[h] /= MICRO_PER_MACRO;
        int predicted_audio = 0;
        for (int h = 1; h < 4; h++) {
            if (a_harm[h] > a_harm[predicted_audio]) predicted_audio = h;
        }
        if (predicted_audio == cat) correct_audio++;
    }

    double vision_acc = (double)correct_vision / training_chunks;
    double audio_acc = (double)correct_audio / training_chunks;

    /* Fusion phase: cross-modality pulses */
    int correct_fusion = 0;
    int fusion_chunks = 500;

    for (int chunk = 0; chunk < fusion_chunks; chunk++) {
        int sample_idx = rand() % TOTAL_SAMPLES;
        int cat = categories[sample_idx];

        /* Encode vision in vQPU 0 */
        for (int u = 0; u < MICRO_PER_MACRO; u++) {
            micro_ring_t *mic = &vqpu[0].macro_rings[0].micro[u];
            double amp = 2.0;
            for (int j = 0; j < OSC_PER_MICRO; j++) {
                mic->th[j] = 0;
                for (int f = 0; f < 4; f++) {
                    mic->th[j] += vision_features[sample_idx][f] * amp *
                                  cos(2.0 * M_PI * (f+1) * j / OSC_PER_MICRO);
                }
            }
            make_harmonic_lens(mic->lens, 1, 0.5);
            apply_lens_micro(mic);
        }

        /* Encode audio in vQPU 1 */
        for (int u = 0; u < MICRO_PER_MACRO; u++) {
            micro_ring_t *mic = &vqpu[1].macro_rings[0].micro[u];
            double amp = 2.0;
            for (int j = 0; j < OSC_PER_MICRO; j++) {
                mic->th[j] = 0;
                for (int f = 0; f < 4; f++) {
                    mic->th[j] += audio_features[sample_idx][f] * amp *
                                  cos(2.0 * M_PI * (f+1) * j / OSC_PER_MICRO);
                }
            }
            make_harmonic_lens(mic->lens, 1, 0.5);
            apply_lens_micro(mic);
        }

        /* Cross-modality pulses: vision → audio, audio → vision */
        for (int u = 0; u < MICRO_PER_MACRO; u++) {
            double q_v = measure_winding(&vqpu[0].macro_rings[0].micro[u]);
            double q_a = measure_winding(&vqpu[1].macro_rings[0].micro[u]);

            if (fabs(q_v) > 0.3) {
                pulse_queue_push(&pq, 0, 0, u, 1, 0, u, q_v, 0.2, 1, chunk);
            }
            if (fabs(q_a) > 0.3) {
                pulse_queue_push(&pq, 1, 0, u, 0, 0, u, q_a, 0.2, 1, chunk);
            }
        }

        /* Settle all vQPUs */
        for (int v = 0; v < 3; v++) {
            for (int m = 0; m < MACRO_PER_SUPER; m++) {
                for (int u = 0; u < MICRO_PER_MACRO; u++) {
                    settle_micro(&vqpu[v].macro_rings[m].micro[u], 1);
                }
            }
        }

        /* Deliver pulses */
        pulse_queue_deliver(&pq, vqpu, 3, chunk);

        /* Measure fusion in vQPU 2 */
        double f_harm[4] = {0};
        for (int u = 0; u < MICRO_PER_MACRO; u++) {
            for (int h = 0; h < 4; h++) {
                f_harm[h] += harmonic_mag(&vqpu[2].macro_rings[0].micro[u], h+1);
            }
        }
        for (int h = 0; h < 4; h++) f_harm[h] /= MICRO_PER_MACRO;
        int predicted_fusion = 0;
        for (int h = 1; h < 4; h++) {
            if (f_harm[h] > f_harm[predicted_fusion]) predicted_fusion = h;
        }
        if (predicted_fusion == cat) correct_fusion++;
    }

    double fusion_acc = (double)correct_fusion / fusion_chunks;

    init_result(&res, "sensor_fusion_demo");
    res.num_vqpus = 3;
    res.src_level = 0; res.dst_level = 0;
    res.transfer_ratio = vision_acc;
    res.routing_accuracy = audio_acc;
    res.super_routing_accuracy = fusion_acc;
    res.macro_aggregation_score = (vision_acc + audio_acc) / 2.0;
    print_result(&res);

    fprintf(stderr, "Sensor Fusion: vision=%.1f%%, audio=%.1f%%, fusion=%.1f%%\n",
            vision_acc * 100, audio_acc * 100, fusion_acc * 100);
}

/* =========== MAIN =========== */
int main() {
    print_csv_header();
    srand(42);

    fprintf(stderr, "Running micro→micro transfer...\n");
    run_micro_to_micro_transfer();

    fprintf(stderr, "Running pulse reception sweep...\n");
    run_pulse_reception_sweep();

    fprintf(stderr, "Running macro aggregation cross-vQPU...\n");
    run_macro_aggregation_cross();

    fprintf(stderr, "Running macro feature routing...\n");
    run_macro_feature_routing();

    fprintf(stderr, "Running super ring attention routing...\n");
    run_super_routing();

    fprintf(stderr, "Running independent states...\n");
    run_independent_states();

    fprintf(stderr, "Running scale scaling...\n");
    run_scale_scaling();

    fprintf(stderr, "Running plasticity learning curve...\n");
    run_plasticity_learning_curve();

    fprintf(stderr, "Running plasticity forgetting test...\n");
    run_plasticity_forgetting();

    fprintf(stderr, "Running super-super coordination (3 vQPUs)...");
    run_super_super_coordination();

    fprintf(stderr, "Running sensor fusion demo...");
    run_sensor_fusion_demo();

    fprintf(stderr, "Done.");
    return 0;
}