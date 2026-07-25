/* ===========================================================================
 * bqsm_kernel_4bit.h — BQSM Matmul Kernels v7 (4-bit quantization)
 *
 *   4-bit signed weights: 16 levels (-7.5 to 7.5)
 *   Packed: 2 values per byte (4 bits each)
 *   Support both 2-bit (ternary) and 4-bit modes
 * ======================================================================== */

#ifndef BQSM_KERNEL_4BIT_H
#define BQSM_KERNEL_4BIT_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#ifdef __SSSE3__
#include <tmmintrin.h>
#include <smmintrin.h>
#endif

#define BQSM_Q_2BIT 3  /* 2-bit: 0,1,2,3 -> -1,0,1,0 */
#define BQSM_Q_4BIT 15 /* 4-bit: 0..15 -> -7.5..7.5 */

/* ─── 2-bit Ternary LUT ─────────────────────────────────────────────────── */
static const int8_t bqsm_ternary_lut[16] __attribute__((aligned(16))) = {
    -1, 0, 1, 0,  -1, 0, 1, 0,  -1, 0, 1, 0,  -1, 0, 1, 0
};

/* ─── 4-bit Signed LUT ─────────────────────────────────────────────────── */
/* 0=-7.5, 1=-6.5, ..., 15=7.5 */
static const float bqsm_4bit_lut[16] __attribute__((aligned(16))) = {
    -7.5f, -6.5f, -5.5f, -4.5f, -3.5f, -2.5f, -1.5f, -0.5f,
     0.5f,  1.5f,  2.5f,  3.5f,  4.5f,  5.5f,  6.5f,  7.5f
};

/* ─── Scalar fallback: 4-bit matmul ─────────────────────────────────────── */
static inline void bqsm_matmul_vec_4bit_scalar(const int8_t *x, const uint8_t *W_packed,
                                                int M, int N, float *C) {
    memset(C, 0, N * sizeof(float));
    for (int i = 0; i < M; i++) {
        float a = (float)x[i];
        for (int j = 0; j < N; j++) {
            int bi = i * (N/2) + j/2;
            int shift = (j % 2) * 4;
            int nb = (W_packed[bi] >> shift) & 0x0F;
            float w = bqsm_4bit_lut[nb];
            C[j] += a * w;
        }
    }
}

/* ─── Scalar fallback: 2-bit matmul ─────────────────────────────────────── */
static inline void bqsm_matmul_vec_2bit_scalar(const int8_t *x, const uint8_t *W_packed,
                                                int M, int N, float *C) {
    memset(C, 0, N * sizeof(float));
    for (int i = 0; i < M; i++) {
        float a = (float)x[i];
        for (int j = 0; j < N; j++) {
            int bi = i * (N/4) + j/4;
            int shift = (j % 4) * 2;
            int nb = (W_packed[bi] >> shift) & 3;
            float w = (float)bqsm_ternary_lut[nb];
            C[j] += a * w;
        }
    }
}

/* ─── Main matmul function (auto-detect bit width) ─────────────────────── */
static inline void bqsm_matmul_vec(const int8_t *x, const uint8_t *W_packed,
                                    int M, int N, void *C_out, int bits=4) {
    if (bits == 4) {
        float *C = (float*)C_out;
        bqsm_matmul_vec_4bit_scalar(x, W_packed, M, N, C);
    } else {
        float *C = (float*)C_out;
        bqsm_matmul_vec_2bit_scalar(x, W_packed, M, N, C);
    }
}

#endif /* BQSM_KERNEL_4BIT_H */
