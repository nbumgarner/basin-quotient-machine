/* ===========================================================================
 * bqsm_kernel.h — BQSM Matmul Kernels v5 (packed, phase-serial, low reg pressure)
 *
 *   Packed 2-bit weights, in-register unpack per nibble phase.
 *   One phase at a time (4 acc regs reused), no scatter stores.
 *   4 passes over k: each pass reads packed weights (same total traffic
 *   as one unpacked pass since packed is 4× smaller).
 * ======================================================================== */

#ifndef BQSM_KERNEL_H
#define BQSM_KERNEL_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#ifdef __SSSE3__
#include <tmmintrin.h>
#endif

#define BQSM_Q 3

static int8_t bqsm_tt[16] __attribute__((aligned(16)));

static inline void bqsm_tt_init(void) {
    for (int a = 0; a <= BQSM_Q; a++)
        for (int b = 0; b <= BQSM_Q; b++) {
            int p = (a * b + 1) / 2;
            bqsm_tt[(a << 2) | b] = (int8_t)(p < 0 ? 0 : (p > BQSM_Q ? BQSM_Q : p));
        }
}

static inline void bqsm_matmul_vec_scalar(const int8_t *x, const int8_t *W,
                                          int M, int N, int32_t *C) {
    memset(C, 0, N * sizeof(int32_t));
    for (int k = 0; k < M; k++) {
        int a = (int)x[k] << 2;
        const int8_t *Wrow = W + k * N;
        for (int j = 0; j < N; j++)
            C[j] += bqsm_tt[a | (int)Wrow[j]];
    }
}

#ifdef __SSSE3__

/* Process 64 columns (one 16-byte packed block) for all k.
 * One nibble phase per call (phase = 0..3).
 * Uses 4 XMM accumulator registers — no spilling.
 * Output: C[j0 + phase + 4*i] for i=0..15 (stride-4, stored once at end). */
static inline void bqsm_sse_block(const int8_t *x, const uint8_t *W_packed,
                                  int M, int N, int j0, int phase,
                                  int32_t *C) {
    __m128i tt   = _mm_load_si128((__m128i*)bqsm_tt);
    __m128i zero = _mm_setzero_si128();
    __m128i m03  = _mm_set1_epi8(0x03);
    int shift    = phase * 2;

    __m128i acc0 = zero, acc1 = zero, acc2 = zero, acc3 = zero;

    for (int k = 0; k < M; k++) {
        __m128i a4 = _mm_set1_epi8((char)((int)x[k] << 2));
        __m128i p  = _mm_loadu_si128((__m128i*)&W_packed[k * (N/4) + j0/4]);
        __m128i w  = _mm_and_si128(_mm_srli_epi32(p, shift), m03);
        __m128i pr = _mm_shuffle_epi8(tt, _mm_or_si128(a4, w));

        __m128i lo16 = _mm_unpacklo_epi8(pr, zero);
        __m128i hi16 = _mm_unpackhi_epi8(pr, zero);

        acc0 = _mm_add_epi32(acc0, _mm_unpacklo_epi16(lo16, zero));
        acc1 = _mm_add_epi32(acc1, _mm_unpackhi_epi16(lo16, zero));
        acc2 = _mm_add_epi32(acc2, _mm_unpacklo_epi16(hi16, zero));
        acc3 = _mm_add_epi32(acc3, _mm_unpackhi_epi16(hi16, zero));
    }

    /* Store to stride-4 positions: C[j0 + phase + 0], C[j0 + phase + 4], ... */
    int32_t *out = C + j0 + phase;
    int32_t tmp[16] __attribute__((aligned(16)));
    _mm_store_si128((__m128i*)&tmp[0],  acc0);
    _mm_store_si128((__m128i*)&tmp[4],  acc1);
    _mm_store_si128((__m128i*)&tmp[8],  acc2);
    _mm_store_si128((__m128i*)&tmp[12], acc3);
    for (int i = 0; i < 16; i++)
        out[i * 4] += tmp[i];  /* stride-4 write, once per block */
}

static inline void bqsm_matmul_vec_sse(const int8_t *x, const uint8_t *W_packed,
                                       int M, int N, int32_t *C) {
    memset(C, 0, N * sizeof(int32_t));

    for (int j0 = 0; j0 < N; j0 += 64) {
        int np = (j0 + 64 <= N) ? 64 : N - j0;
        if (np == 64) {
            for (int phase = 0; phase < 4; phase++)
                bqsm_sse_block(x, W_packed, M, N, j0, phase, C);
        } else {
            for (int k = 0; k < M; k++) {
                int a = (int)x[k] << 2;
                for (int j = j0; j < N; j++) {
                    int bi = k * (N/4) + j/4;
                    int w = (W_packed[bi] >> (2 * (j % 4))) & 3;
                    C[j] += bqsm_tt[a | w];
                }
            }
        }
    }
}
#endif

static inline void bqsm_matmul_vec(const int8_t *x, const int8_t *W,
                                   int M, int N, int32_t *C) {
#ifdef __SSSE3__
    bqsm_matmul_vec_sse(x, (const uint8_t*)W, M, N, C);
#else
    bqsm_matmul_vec_scalar(x, W, M, N, C);
#endif
}

#endif /* BQSM_KERNEL_H */