# Design: VQPU-ATTN-XVQPU-v1
## Cross-vQPU Super-Super Attention Routing Protocol with ACK/NACK

### Mission
Implement reliable cross-vQPU super-ring coordination using harmonic similarity-based attention weights with ACK/NACK handshake and retry protocol.

### Constants
```c
#define SUPER_MAX_RETRIES       3       /* Max retries before NACK */
#define SUPER_ACK_TIMEOUT       2       /* Chunks to wait for ACK before retry */
#define SUPER_PULSE_TTL         8       /* Time-to-live for super pulses (chunks) */
#define SUPER_MAX_PENDING_ACKS  32      /* Max pending ACKs tracked per vQPU */
#define SUPER_HARMONIC_MAX      8       /* Max harmonic index for super pulses */
#define SUPER_WEIGHT_SCALE      1.0     /* Attention weight scaling factor */
#define SUPER_FEEDBACK_GAIN     5.0     /* Super-to-macro feedback gain */
```

### Pulse Types
```c
typedef enum {
    SUPER_PULSE_DATA = 0,   /* Data payload pulse */
    SUPER_PULSE_ACK  = 1,   /* Acknowledgment */
    SUPER_PULSE_NACK = 2,   /* Negative acknowledgment */
    SUPER_PULSE_SYNC = 3    /* Synchronization pulse */
} super_pulse_type_t;
```

### Data Structures

#### Super Pulse
```c
typedef struct {
    int src_vqpu;              /* Source vQPU ID */
    int dst_vqpu;              /* Destination vQPU ID */
    int src_macro;             /* Source macro ring index */
    int dst_macro;             /* Destination macro ring index */
    int src_harmonic;          /* Source harmonic */
    int dst_harmonic;          /* Target harmonic */
    double phase;              /* Global phase state */
    double weight;             /* Attention weight (routing strength) */
    int harmonic;              /* Target harmonic for routing */
    int ttl;                   /* Time-to-live (chunks remaining) */
    int seq;                   /* Sequence number for ACK/NACK */
    super_pulse_type_t type;   /* Pulse type: DATA, ACK, NACK, SYNC */
    int timestep;              /* Emission timestep (chunk) */
    double payload_phase[8];   /* Harmonic phase payload (up to 8 harmonics) */
    double feedback_gain;      /* Super-to-macro feedback gain */
} super_pulse_t;
```

#### Pending ACK Tracking
```c
typedef struct {
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
} pending_ack_t;
```

### Functions

#### super_pulse_handshake()
Handles ACK/NACK protocol with sequence numbers, timeout-based retries, and max retry limit.

**Signature:**
```c
void super_pulse_handshake(super_pulse_queue_t *q, vqpu_t *vqpus, int num_vqpus, int current_chunk);
```

**Behavior:**
1. Iterate through all vQPUs' pending ACK queues
2. For each pending ACK:
   - Calculate elapsed chunks: `current_chunk - timestep_sent`
   - If `elapsed >= SUPER_ACK_TIMEOUT`:
     - If `retry_count < SUPER_MAX_RETRIES`:
       - Resend DATA pulse with same sequence number
       - Increment `retry_count`
       - Update `timestep_sent = current_chunk`
     - Else (max retries exceeded):
       - Send NACK pulse with same sequence number
       - Clear pending ACK slot
       - Log NACK event

#### super_routing_compute_weights()
Computes cross-vQPU attention weights using cosine similarity of macro harmonic vectors.

**Signature:**
```c
void super_routing_compute_weights(vqpu_t *vqpus, int num_vqpus);
```

**Algorithm:**
For each source vQPU `src` and destination vQPU `dst` (src != dst):
  For each source macro `sm` and destination macro `dm`:
    1. Extract harmonic vectors `src_harm[h]` and `dst_harm[h]` for h=1..8
    2. Compute cosine similarity:
       ```
       similarity = Sum(src_harm[h] * dst_harm[h]) / (||src_harm|| * ||dst_harm||)
       ```
    3. Store weight: `super_attention_weights[dst][dm] = similarity * SUPER_WEIGHT_SCALE`
    4. Apply softmax normalization across destinations for each source macro

#### super_pulse_queue_deliver()
Delivers super pulses and handles ACK/NACK generation.

**Behavior on receiving DATA pulse:**
1. Apply phase perturbation to destination macro's micro rings
2. Generate and queue ACK pulse with matching sequence number
3. Update destination's macro harmonic state

**Behavior on receiving ACK pulse:**
1. Find matching pending ACK by sequence number in source vQPU
2. Clear the pending ACK slot
3. Decrement pending ACK count

**Behavior on receiving NACK pulse:**
1. Find matching pending ACK by sequence number in source vQPU
2. Clear the pending ACK slot
3. Log NACK reception

### Test: 3 vQPU Coordination
**Scenario:**
- vQPU 0: FILTER mode - strong harmonic 1 in macro 0
- vQPU 1: MEMORY mode - strong harmonic 2 in macro 0
- vQPU 2: ATTENTION mode - neutral, receives from both

**Validation:**
1. vQPU 0 emits super pulses to vQPU 2 (macro 0 -> macro 0, harmonic 1 -> harmonic 1)
2. vQPU 1 emits super pulses to vQPU 2 (macro 0 -> macro 1, harmonic 2 -> harmonic 2)
3. ACK/NACK handshake ensures reliable delivery
4. vQPU 2 should develop both harmonic 1 (macro 0) and harmonic 2 (macro 1)
5. Coordination success: both harmonics > 0.03 threshold

### Metrics
| Metric | Target |
|--------|--------|
| ACK rate | > 95% |
| NACK rate | < 5% |
| Coordination success | 100% (3 vQPUs) |
| Routing accuracy | > 85% |
| Sync time | < 10 chunks |

### Dependencies
- dept-super: macro_harmonics[16][8], super_ring_t structure
- dept-pulse-fabric: super_pulse_t, super_pulse_queue_t
- dept-macro: macro_ring_t, macro_harmonic()

### Files
- `attn_xvqpu.h` - Header with declarations
- `attn_xvqpu.c` - Implementation
- `test_attn_xvqpu.c` - Validation test for 3 vQPU coordination
- `Makefile` - Build configuration
