# Sensor Fusion Demo (2-Modality Classification)

## Overview

This document describes a sensor fusion demonstration using the neuromorphic vQPU cluster. The demo shows how two vQPUs can process different sensory modalities (vision + audio) and fuse their outputs for classification.

---

## Learning Objective

**Target:** Classify 4 categories with >90% accuracy using fused vision + audio input

**Categories:**
1. Cat meow (visual: cat, audio: meow)
2. Dog bark (visual: dog, audio: bark)
3. Bird chirp (visual: bird, audio: chirp)
4. Car horn (visual: car, audio: horn)

**Mechanism:**
- vQPU 0: Vision processor (encodes visual patterns)
- vQPU 1: Audio processor (encodes audio patterns)
- vQPU 2: Fusion processor (combines vision + audio)
- Super ring attention routes relevant features

---

## Architecture

### vQPU Roles

| vQPU | Role | Input | Output |
|------|------|-------|--------|
| vQPU 0 | Vision | Visual features | Visual harmonic pattern |
| vQPU 1 | Audio | Audio features | Audio harmonic pattern |
| vQPU 2 | Fusion | Vision + audio pulses | Classification |

### Input Encoding

#### Vision Features (vQPU 0)

```c
/* Visual feature encoding */
typedef struct {
    double edge_density;      /* 0-1: edge density in image */
    double color_hue;         /* 0-2π: dominant hue */
    double motion_speed;      /* 0-1: motion speed */
    double texture_complexity; /* 0-1: texture complexity */
} vision_features_t;

/* Encode vision features as harmonic pattern */
void encode_vision(vision_features_t *features, micro_ring_t *ring) {
    /* Map 4 features to 4 harmonics */
    double amp1 = features->edge_density * 2.0;
    double amp2 = features->color_hue / (2.0 * M_PI) * 2.0;
    double amp3 = features->motion_speed * 2.0;
    double amp4 = features->texture_complexity * 2.0;
    
    /* Encode as superposition of harmonics */
    for (int j = 0; j < OSC_PER_MICRO; j++) {
        ring->th[j] = amp1 * cos(2.0 * M_PI * 1 * j / OSC_PER_MICRO)
                    + amp2 * cos(2.0 * M_PI * 2 * j / OSC_PER_MICRO)
                    + amp3 * cos(2.0 * M_PI * 3 * j / OSC_PER_MICRO)
                    + amp4 * cos(2.0 * M_PI * 4 * j / OSC_PER_MICRO);
    }
}
```

#### Audio Features (vQPU 1)

```c
/* Audio feature encoding */
typedef struct {
    double fundamental_freq;  /* 0-1: normalized fundamental frequency */
    double spectral_centroid; /* 0-1: spectral centroid */
    double rms_energy;        /* 0-1: root mean square energy */
    double zero_crossing_rate; /* 0-1: zero crossing rate */
} audio_features_t;

/* Encode audio features as harmonic pattern */
void encode_audio(audio_features_t *features, micro_ring_t *ring) {
    /* Map 4 features to 4 harmonics */
    double amp1 = features->fundamental_freq * 2.0;
    double amp2 = features->spectral_centroid * 2.0;
    double amp3 = features->rms_energy * 2.0;
    double amp4 = features->zero_crossing_rate * 2.0;
    
    /* Encode as superposition of harmonics */
    for (int j = 0; j < OSC_PER_MICRO; j++) {
        ring->th[j] = amp1 * cos(2.0 * M_PI * 1 * j / OSC_PER_MICRO)
                    + amp2 * cos(2.0 * M_PI * 2 * j / OSC_PER_MICRO)
                    + amp3 * cos(2.0 * M_PI * 3 * j / OSC_PER_MICRO)
                    + amp4 * cos(2.0 * M_PI * 4 * j / OSC_PER_MICRO);
    }
}
```

### Training Data

| Category | Edge Density | Hue | Motion | Texture | Freq | Spec Cent | RMS | ZCR |
|----------|--------------|-----|--------|---------|------|-----------|-----|-----|
| Cat | 0.3 | 0.8 | 0.2 | 0.4 | 0.6 | 0.5 | 0.3 | 0.2 |
| Dog | 0.5 | 0.6 | 0.7 | 0.6 | 0.3 | 0.4 | 0.7 | 0.4 |
| Bird | 0.2 | 0.9 | 0.8 | 0.3 | 0.9 | 0.7 | 0.2 | 0.6 |
| Car | 0.7 | 0.1 | 0.9 | 0.8 | 0.4 | 0.6 | 0.9 | 0.8 |

---

## Fusion Mechanism

### Cross-vQPU Pulse Protocol

```c
/* Vision → Audio pulse (synchronization) */
void vision_to_audio_pulse(vqpu_t *vision_vqpu, vqpu_t *audio_vqpu) {
    /* Get dominant harmonic from vision */
    int dominant_harmonic = detect_dominant_harmonic(vision_vqpu);
    
    /* Emit pulse to audio vQPU */
    pulse_queue_push(&pq, 
                     0, 1, 0,           /* src: vQPU 0, macro 0 */
                     1, 1, 0,           /* dst: vQPU 1, macro 0 */
                     dominant_harmonic, /* phase = harmonic */
                     0.3,               /* strength */
                     dominant_harmonic, /* harmonic */
                     current_chunk);
}

/* Audio → Vision pulse (synchronization) */
void audio_to_vision_pulse(vqpu_t *audio_vqpu, vqpu_t *vision_vqpu) {
    /* Get dominant harmonic from audio */
    int dominant_harmonic = detect_dominant_harmonic(audio_vqpu);
    
    /* Emit pulse to vision vQPU */
    pulse_queue_push(&pq,
                     1, 1, 0,           /* src: vQPU 1, macro 0 */
                     0, 1, 0,           /* dst: vQPU 0, macro 0 */
                     dominant_harmonic, /* phase = harmonic */
                     0.3,               /* strength */
                     dominant_harmonic, /* harmonic */
                     current_chunk);
}
```

### Fusion in vQPU 2

```c
/* Fusion processor receives pulses from vision and audio */
void process_fusion(vqpu_t *fusion_vqpu, pulse_queue_t *pq, int chunk) {
    /* Deliver pulses to fusion vQPU */
    pulse_queue_deliver(pq, vqpus, 3, chunk);
    
    /* Measure harmonic response in fusion vQPU */
    double h1 = macro_harmonic(&fusion_vqpu->macro_rings[0], 1);
    double h2 = macro_harmonic(&fusion_vqpu->macro_rings[0], 2);
    double h3 = macro_harmonic(&fusion_vqpu->macro_rings[0], 3);
    double h4 = macro_harmonic(&fusion_vqpu->macro_rings[0], 4);
    
    /* Classify based on dominant harmonic */
    int classification = classify_fusion(h1, h2, h3, h4);
    
    return classification;
}

/* Classification logic */
int classify_fusion(double h1, double h2, double h3, double h4) {
    /* Find dominant harmonic */
    double max_h = h1;
    int class_idx = 0;
    
    if (h2 > max_h) { max_h = h2; class_idx = 1; }
    if (h3 > max_h) { max_h = h3; class_idx = 2; }
    if (h4 > max_h) { max_h = h4; class_idx = 3; }
    
    return class_idx;
}
```

---

## Training Protocol

### Phase 1: Modality-Specific Learning (2000 chunks)

1. Present vision-only samples to vQPU 0
2. Present audio-only samples to vQPU 1
3. Learn modality-specific harmonic patterns
4. Measure classification accuracy per modality

### Phase 2: Cross-Modality Synchronization (2000 chunks)

1. Present paired vision + audio samples
2. Emit synchronization pulses between vQPUs
3. Learn cross-modality associations
4. Measure fusion accuracy

### Phase 3: Fusion Classification (1000 chunks)

1. Present fused samples to vQPU 2
2. Learn fusion classification weights
3. Measure final classification accuracy
4. Test generalization to unseen samples

---

## Metrics

### Primary Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Vision accuracy | >85% | Correct / Total vision samples |
| Audio accuracy | >85% | Correct / Total audio samples |
| Fusion accuracy | >90% | Correct / Total fused samples |
| Cross-modality correlation | >0.7 | Pearson correlation of harmonics |

### Secondary Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Synchronization latency | <100 chunks | Chunks between pulse and response |
| Fusion energy | <1 mJ | Energy per classification |
| Plasticity stability | No divergence | Max weight / Min weight |

---

## Expected Results

### Learning Curves

```
Chunks | Vision | Audio | Fusion
-------|--------|-------|--------
0      | 25%    | 25%   | 25%
500    | 60%    | 60%   | 50%
1000   | 75%    | 75%   | 70%
1500   | 85%    | 85%   | 85%
2000   | 90%    | 90%   | 90%
2500   | 90%    | 90%   | 92%
3000   | 90%    | 90%   | 93%
```

### Confusion Matrix (Final)

```
          Predicted
          Cat  Dog  Bird  Car
Actual Cat  92   4    2     2
       Dog   3   91    4     2
       Bird  1   3    94    2
       Car   2   4    2    92
```

---

## Code Structure

```c
/* sensor_fusion_demo.c */

/* Feature structures */
typedef struct { double edge_density; double color_hue; ... } vision_features_t;
typedef struct { double fundamental_freq; double spectral_centroid; ... } audio_features_t;

/* Training data */
typedef struct {
    vision_features_t vision;
    audio_features_t audio;
    int category;  /* 0=Cat, 1=Dog, 2=Bird, 3=Car */
} training_sample_t;

training_sample_t training_data[100];  /* 25 samples per category */

/* Functions */
void init_sensor_fusion(void);
void train_modality_specific(int chunks);
void train_cross_modality(int chunks);
void train_fusion(int chunks);
double test_fusion_accuracy(void);
void print_confusion_matrix(void);
```

---

## Validation

### Test 1: Vision-Only Classification

```
Input: Vision features for "Cat"
Expected: Correct classification
Accuracy: >85%
```

### Test 2: Audio-Only Classification

```
Input: Audio features for "Dog"
Expected: Correct classification
Accuracy: >85%
```

### Test 3: Fused Classification

```
Input: Vision + Audio for "Bird"
Expected: Correct classification
Accuracy: >90%
```

### Test 4: Generalization

```
Input: Unseen samples (not in training set)
Expected: >80% accuracy
```

### Test 5: Noise Robustness

```
Input: Fused samples + 10% Gaussian noise
Expected: >85% accuracy
```

---

## Extensions

### Extension 1: Multi-Modality (3+ modalities)

- Add thermal vision
- Add Lidar data
- Fuse 3+ modalities

### Extension 2: Temporal Fusion

- Process sequences of frames
- Learn temporal patterns within modality
- Fuse temporal patterns across modalities

### Extension 3: Active Fusion

- vQPU 2 can request specific features from vQPU 0/1
- Learn which features are most discriminative
- Optimize fusion based on task

---

## References

- Simulation: `cluster_validate.c`
- Plasticity: `PLASTICITY_RULE_DESIGN.md`
- Attention: Hebbian weight learning

---

**Generated by:** Precision1
**Date:** 2025-07-26
**Status:** Demo specification, ready for implementation
