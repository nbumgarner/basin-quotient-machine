# Temporal Sequence Learning Demo

## Overview

This document describes a temporal sequence learning demonstration using the neuromorphic vQPU cluster. The demo shows how the cluster can learn and reproduce temporal sequences through plasticity and Hebbian learning.

---

## Learning Objective

**Target:** Learn a 5-step temporal sequence and reproduce it with >90% accuracy

**Sequence:** A → B → C → D → E → (repeat)

**Mechanism:** 
- Each step encoded as a harmonic pattern
- Cross-vQPU pulses encode temporal transitions
- Hebbian plasticity learns the transition weights

---

## Architecture

### vQPU Roles

| vQPU | Role | Harmonic Encoding |
|------|------|-------------------|
| vQPU 0 | Sequence memory | Encodes all 5 steps |
| vQPU 1 | Transition learner | Learns A→B, B→C, C→D, D→E, E→A |
| vQPU 2 | Pattern recognizer | Recognizes current step |

### Harmonic Encoding

Each step in the sequence is encoded as a unique harmonic pattern:

```
Step A: harmonic 1 (fundamental)
Step B: harmonic 2
Step C: harmonic 3
Step D: harmonic 4
Step E: harmonic 5
```

### Transition Encoding

Transitions are encoded as pulse events:
- When vQPU 0 encodes step A, it emits a pulse to vQPU 1
- vQPU 1 learns to associate pulse from A with next step B
- This creates a chain: A → pulse → B → pulse → C → ...

---

## Implementation

### Step 1: Sequence Generation

```c
/* Sequence definition */
#define SEQ_LENGTH 5
int sequence[SEQ_LENGTH] = {1, 2, 3, 4, 5};  // Harmonics 1-5
int current_step = 0;

/* Generate sequence step */
void generate_sequence_step(vqpu_t *vqpu, int step_idx) {
    int harmonic = sequence[step_idx % SEQ_LENGTH];
    
    /* Encode harmonic in macro ring 0 */
    for (int u = 0; u < MICRO_PER_MACRO; u++) {
        micro_ring_t *mic = &vqpu->macro_rings[0].micro[u];
        encode_input(mic, harmonic, 2.0);
        make_harmonic_lens(mic->lens, harmonic, 0.5);
        apply_lens_micro(mic);
    }
}
```

### Step 2: Transition Learning

```c
/* Hebbian transition learning */
void learn_transition(vqpu_t *src, vqpu_t *dst, int src_harmonic, int dst_harmonic) {
    /* Compute correlation between source and destination harmonics */
    double correlation = 0.0;
    for (int h = 0; h < ATTN_HARMONICS; h++) {
        double src_h = macro_harmonic(&src->macro_rings[0], h + 1);
        double dst_h = macro_harmonic(&dst->macro_rings[0], h + 1);
        correlation += src_h * dst_h;
    }
    
    /* Update transition weight */
    double delta_w = ATTN_LEARNING_RATE * correlation 
                   - ATTN_WEIGHT_DECAY * transition_weight[src_harmonic][dst_harmonic];
    transition_weight[src_harmonic][dst_harmonic] += delta_w;
    
    /* Clamp weights */
    if (transition_weight[src_harmonic][dst_harmonic] < 0.0) 
        transition_weight[src_harmonic][dst_harmonic] = 0.0;
    if (transition_weight[src_harmonic][dst_harmonic] > 1.0) 
        transition_weight[src_harmonic][dst_harmonic] = 1.0;
}
```

### Step 3: Sequence Reproduction

```c
/* Reproduce learned sequence */
int reproduce_sequence(vqpu_t *vqpu) {
    int predicted_step = 0;
    
    /* Get current harmonic from pattern recognizer */
    int current_harmonic = detect_harmonic(vqpu);
    
    /* Find next step with highest transition weight */
    double max_weight = 0.0;
    for (int next = 0; next < SEQ_LENGTH; next++) {
        double weight = transition_weight[current_harmonic][sequence[next]];
        if (weight > max_weight) {
            max_weight = weight;
            predicted_step = next;
        }
    }
    
    return sequence[predicted_step];
}
```

---

## Training Protocol

### Phase 1: Exposure (1000 chunks)

1. Present sequence A→B→C→D→E repeatedly
2. Each step lasts 200 chunks
3. Pulses encode transitions between steps
4. Hebbian learning updates transition weights

### Phase 2: Testing (500 chunks)

1. Present first step (A) only
2. Let system reproduce sequence autonomously
3. Measure accuracy of each transition

### Phase 3: Variations (500 chunks)

1. Test with partial sequences (A→C→E)
2. Test with noise (random perturbations)
3. Test with speed variations

---

## Metrics

### Primary Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Sequence accuracy | >90% | Correct steps / Total steps |
| Transition accuracy | >85% | Correct transitions / Total transitions |
| Learning speed | <2000 chunks | Chunks to >80% accuracy |
| Recall speed | <100 chunks/step | Chunks per reproduced step |

### Secondary Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Energy per step | <1 mJ | Energy / Step |
| Plasticity stability | No divergence | Max weight / Min weight |
| Catastrophic forgetting | >80% retention | Accuracy after new learning |

---

## Expected Results

### Learning Curve

```
Chunks | Accuracy
-------|----------
0      | 20% (random)
500    | 60%
1000   | 80%
1500   | 90%
2000   | 95%
```

### Transition Weights (After Training)

```
        A      B      C      D      E
A    0.00   0.85   0.10   0.02   0.01
B    0.05   0.00   0.82   0.08   0.02
C    0.02   0.08   0.00   0.80   0.05
D    0.03   0.05   0.07   0.00   0.82
E    0.90   0.03   0.02   0.02   0.01
```

**Note:** E→A has high weight (0.90) due to sequence closure.

---

## Code Structure

```c
/* temporal_sequence_learning.c */

/* Sequence definition */
#define SEQ_LENGTH 5
int sequence[SEQ_LENGTH] = {1, 2, 3, 4, 5};

/* Transition weight matrix */
double transition_weight[9][9];  /* Harmonics 1-5 + padding */

/* Training state */
int current_step = 0;
int training_chunks = 0;
double learning_curve[2000];

/* Functions */
void init_sequence_learning(void);
void train_sequence(int chunks);
double test_sequence(void);
void print_learning_curve(void);
```

---

## Integration with Cluster

### vQPU 0: Sequence Memory

- Stores the full sequence in harmonic patterns
- Each macro ring encodes one step
- Emits pulses to trigger next step

### vQPU 1: Transition Learner

- Learns A→B, B→C, C→D, D→E, E→A
- Updates Hebbian weights on each transition
- Stores transition matrix

### vQPU 2: Pattern Recognizer

- Detects current step from harmonic input
- Provides feedback to vQPU 0
- Measures recognition accuracy

---

## Validation

### Test 1: Full Sequence Reproduction

```
Input: Step A
Expected: A → B → C → D → E → A → ...
Accuracy: >90%
```

### Test 2: Partial Sequence

```
Input: Step C
Expected: C → D → E → A → B → C → ...
Accuracy: >85%
```

### Test 3: Noise Robustness

```
Input: Step A + random perturbation
Expected: Correct sequence after 1-2 steps
Accuracy: >80%
```

### Test 4: Speed Variation

```
Input: Fast sequence (50 chunks/step)
Expected: Correct reproduction
Accuracy: >75%
```

---

## Extensions

### Extension 1: Multiple Sequences

- Learn sequence 1: A→B→C→D→E
- Learn sequence 2: A→C→E→B→D
- Switch between sequences based on context

### Extension 2: Hierarchical Sequences

- Learn meta-sequence: (A→B) → (C→D) → E
- Recognize sub-sequences as units

### Extension 3: Temporal Binding

- Bind sequence to sensory input
- Learn: visual pattern → auditory pattern → motor pattern

---

## References

- Simulation: `cluster_validate.c`
- Plasticity: `PLASTICITY_RULE_DESIGN.md`
- Attention: Hebbian weight learning

---

**Generated by:** Precision1
**Date:** 2025-07-26
**Status:** Demo specification, ready for implementation
