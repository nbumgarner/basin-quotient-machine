#ifndef ATTN_XVQPU_H
#define ATTN_XVQPU_H

#include <stddef.h>

/* Hierarchical vQPU structure constants (from cluster_validate.c) */
#define OSC_PER_MICRO   16
#define MICRO_PER_MACRO 16
#define MACRO_PER_SUPER 16
#define TOTAL_OSC       (OSC_PER_MICRO * MICRO_PER_MACRO * MACRO_PER_SUPER)  /* 4096 */

#define MAX_VQPUS       16
#define SUPER_MAX_PENDING_ACKS  32

/* Super-Super Protocol Constants */
#define SUPER_MAX_RETRIES       3
#define SUPER_ACK_TIMEOUT       2
#define SUPER_PULSE_TTL         8
#define SUPER_HARMONIC_MAX      8
#define SUPER_WEIGHT_SCALE      1.0
#define SUPER_FEEDBACK_GAIN     5.0

/* Pulse Types */
typedef enum {
    SUPER_PULSE_DATA = 0,
    SUPER_PULSE_ACK  = 1,
    SUPER_PULSE_NACK = 2,
    SUPER_PULSE_SYNC = 3
} super_pulse_type_t;

/* Super Pulse Structure */
typedef struct {
    int src_vqpu;
    int dst_vqpu;
    int src_macro;
    int dst_macro;
    int src_harmonic;
    int dst_harmonic;
    double phase;
    double weight;
    int harmonic;
    int ttl;
    int seq;
    super_pulse_type_t type;
    int timestep;
    double payload_phase[8];
    double feedback_gain;
} super_pulse_t;

/* Pending ACK Tracking */
typedef struct {
    int seq;
    int dst_vqpu;
    int dst_macro;
    int src_macro;
    int src_harmonic;
    int dst_harmonic;
    double phase;
    double weight;
    int harmonic;
    int timestep_sent;
    int retry_count;
    int in_use;
    double payload_phase[8];
    double feedback_gain;
} pending_ack_t;

/* Super Pulse Queue */
#define MAX_SUPER_PULSES 2000

typedef struct {
    super_pulse_t pulses[MAX_SUPER_PULSES];
    int count;
} super_pulse_queue_t;

/* Forward declarations for vQPU structures */
typedef struct micro_ring_t micro_ring_t;
typedef struct macro_ring_t macro_ring_t;
typedef struct super_ring_t super_ring_t;
typedef struct vqpu_t vqpu_t;

struct micro_ring_t {
    double th[OSC_PER_MICRO];
    double om[OSC_PER_MICRO];
    double v[OSC_PER_MICRO];
    double lens[OSC_PER_MICRO];
    double last_q;
    int id;
    int macro_id;
    /* Plasticity state */
    double lens_coeff[8];
    double pre_trace[OSC_PER_MICRO];
    double post_trace[8];
    int best_harmonic;
    double lens_energy;
};

struct macro_ring_t {
    micro_ring_t micro[MICRO_PER_MACRO];
    double inter_coupling;
    double attention_weights[MICRO_PER_MACRO];
    int id;
    /* Super-to-macro feedback */
    double super_feedback_weights[MACRO_PER_SUPER];
    double super_feedback_phase[MACRO_PER_SUPER];
    int super_feedback_harmonic[MACRO_PER_SUPER];
    /* Weighted feature assembly */
    double feature_weight[MICRO_PER_MACRO];
    double feature_weight_decay;
};

struct super_ring_t {
    macro_ring_t macro[MACRO_PER_SUPER];
    double global_coupling;
    double routing_weights[MAX_VQPUS];
    int id;
    /* Super-super protocol state */
    double super_attention_weights[MAX_VQPUS][MACRO_PER_SUPER];
    int pending_ack_count;
    int super_pulse_seq;
    pending_ack_t pending_acks[SUPER_MAX_PENDING_ACKS];
    /* Hebbian attention weights */
    double attention_weights[MACRO_PER_SUPER][MAX_VQPUS];
    double macro_harmonics[MACRO_PER_SUPER][8];
    double hebbian_traces[MACRO_PER_SUPER][8];
    int task_assignments[MACRO_PER_SUPER];
    double routing_accuracy_history[800];
};

struct vqpu_t {
    macro_ring_t macro_rings[MACRO_PER_SUPER];
    int vqpu_id;
};

/* Function Declarations */

/* Initialize super pulse queue */
void super_pulse_queue_init(super_pulse_queue_t *q);

/* Push super pulse to queue */
void super_pulse_queue_push(super_pulse_queue_t *q, int src_vqpu, int dst_vqpu,
                             int src_macro, int dst_macro, int src_harmonic, int dst_harmonic,
                             double phase, double weight, int harmonic, int ttl,
                             int seq, super_pulse_type_t type, int timestep,
                             const double *payload_phase, double feedback_gain);

/* Deliver super pulses and handle ACK/NACK */
void super_pulse_queue_deliver(super_pulse_queue_t *q, vqpu_t *vqpus, int num_vqpus, int current_chunk);

/* ACK/NACK handshake with retries */
void super_pulse_handshake(super_pulse_queue_t *q, vqpu_t *vqpus, int num_vqpus, int current_chunk);

/* Compute cross-vQPU routing weights using cosine similarity of macro harmonic vectors */
void super_routing_compute_weights(vqpu_t *vqpus, int num_vqpus);

/* Apply super-to-macro feedback modulation with attention gating */
void super_feedback_to_macro(vqpu_t *vqpus, int num_vqpus, int current_chunk);

/* Compute macro harmonic vector for a given macro ring */
void compute_macro_harmonic_vector(macro_ring_t *mr, double *harm_vec);

/* Measure macro harmonic magnitude */
double macro_harmonic(macro_ring_t *mr, int k);

/* Run 3 vQPU coordination test */
int run_three_vqpu_coordination_test(void);

#endif /* ATTN_XVQPU_H */
