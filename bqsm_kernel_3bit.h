/* ===========================================================================
 * bqsm_kernel_3bit.h — BQSM Matmul Kernels v8 (3-bit quantization)
 *
 *   3-bit signed weights: 8 levels (-3.5 to 3.5)
 *   Packed: 2 values per byte (4 bits each, uses only 3 bits)
 *   LUT: 0=-3.5, 1=-2.5, 2=-1.5, 3=-0.5, 4=0.5, 5=1.5, 6=2.5, 7=3.5
 * ======================================================================== */

#ifndef BQSM_KERNEL_3BIT_H
#define BQSM_KERNEL_3BIT_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#ifdef __SSSE3__
#include <tmmintrin.h>
#include <smmintrin.h>
#endif

#define BQSM_Q_3BIT 7  /* 3-bit: 0..7 -> -3.5..3.5 */

/* ─── 3-bit Signed LUT ─────────────────────────────────────────────────── */
/* 0=-3.5, 1=-2.5, 2=-1.5, 3=-0.5, 4=0.5, 5=1.5, 6=2.5, 7=3.5 */
static const float bqsm_3bit_lut[8] __attribute__((aligned(16))) = {
    -3.5f, -2.5f, -1.5f, -0.5f,
     0.5f,  1.5f,  2.5f,  3.5f
};

/* ─── Scalar fallback: 3-bit matmul ─────────────────────────────────────── */
static inline void bqsm_matmul_vec_3bit_scalar(const int8_t *x, const uint8_t *W_packed,
                                                int M, int N, float *C) {
    memset(C, 0, N * sizeof(float));
    for (int i = 0; i < M; i++) {
        float a = (float)x[i];
        for (int j = 0; j < N; j++) {
            int bi = i * (N/2) + j/2;
            int shift = (j % 2) * 4;
            int nb = (W_packed[bi] >> shift) & 0x0F;
            /* Mask to 3 bits (0-7) */
            nb = nb & 7;
            float w = bqsm_3bit_lut[nb];
            C[j] += a * w;
        }
    }
}

/* ─── SSE4.1 optimized: 3-bit matmul (16 values per iteration) ─────────── */
#ifdef __SSSE3__
static inline void bqsm_matmul_vec_3bit_sse(const int8_t *x, const uint8_t *W_packed,
                                             int M, int N, float *C) {
    /* Simple scalar fallback for now - SSE optimization later */
    bqsm_matmul_vec_3bit_scalar(x, W_packed, M, N, C);
}
#endif

/* ─── Main matmul function ─────────────────────────────────────────────── */
static inline void bqsm_matmul_vec_3bit(const int8_t *x, const uint8_t *W_packed,
                                         int M, int N, float *C) {
#ifdef __SSSE3__
    bqsm_matmul_vec_3bit_sse(x, W_packed, M, N, C);
#else
    bqsm_matmul_vec_3bit_scalar(x, W_packed, M, N, C);
#endif
}

#endif /* BQSM_KERNEL_3BIT_H */
