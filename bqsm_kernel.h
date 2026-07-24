/* ===========================================================================
 * bqsm_kernel.h — BQSM Matmul Kernels (scalar + pshufb SIMD)
 * ===========================================================================
 *   Single header. Auto-detects SSSE3 at compile time.
 *   Provides:
 *     bqsm_matmul_vec()  — vector × matrix → output vector
 *     bqsm_matmul_batch() — batch × matrix → batch output
 *
 *   All operations are integer table-lookup: C[j] = sum_k TT[A[k], W[k,j]]
 *   4-state (2-bit) encoding. TT is 16-entry int8 table.
 */

#ifndef BQSM_KERNEL_H
#define BQSM_KERNEL_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#ifdef __SSSE3__
#include <tmmintrin.h>
#endif

/* ─── Transition Table ──────────────────────────────────────────────────── */
#define BQSM_Q 3
static int8_t bqsm_tt[16] __attribute__((aligned(16)));

static inline void bqsm_tt_init(void) {
    for (int a = 0; a <= BQSM_Q; a++)
        for (int b = 0; b <= BQSM_Q; b++) {
            int p = (a * b + 1) / 2;
            bqsm_tt[(a << 2) | b] = (int8_t)(p < 0 ? 0 : (p > BQSM_Q ? BQSM_Q : p));
        }
}

/* ─── Scalar Matmul ─────────────────────────────────────────────────────── */
/* C[j] = sum_k TT[x[k], W[k*N+j]]   x: M, W: M×N, C: N */
static inline void bqsm_matmul_vec_scalar(const int8_t *x, const int8_t *W,
                                          int M, int N, int32_t *C) {
    for (int j = 0; j < N; j++) {
        int32_t acc = 0;
        for (int k = 0; k < M; k++)
            acc += bqsm_tt[((int)x[k] << 2) | (int)W[k * N + j]];
        C[j] = acc;
    }
}

/* ─── SIMD Matmul (SSSE3: pshufb) ──────────────────────────────────────── */
#ifdef __SSSE3__
static inline void bqsm_matmul_vec_sse(const int8_t *x, const int8_t *W,
                                       int M, int N, int32_t *C) {
    /* Load TT into XMM register once */
    __m128i tt = _mm_load_si128((__m128i*)bqsm_tt);
    __m128i zero = _mm_setzero_si128();

    int j = 0;
    /* Process 16 output columns at a time */
    for (; j + 15 < N; j += 16) {
        __m128i acc0 = zero, acc1 = zero, acc2 = zero, acc3 = zero;

        for (int k = 0; k < M; k++) {
            int a = (int)x[k];
            /* Broadcast activation into all 16 lanes, shifted to TT index */
            __m128i a4 = _mm_set1_epi8((char)(a << 2));

            /* Load 16 weight values, mask to 4-state */
            __m128i w = _mm_loadu_si128((__m128i*)&W[k * N + j]);
            w = _mm_and_si128(w, _mm_set1_epi8(3));

            /* pshufb: TT[activation | weight] → 16 results */
            __m128i idx = _mm_or_si128(a4, w);
            __m128i prod = _mm_shuffle_epi8(tt, idx);

            /* Zero-extend int8 → int32 (values are 0..3, non-negative) */
            __m128i p0 = _mm_unpacklo_epi16(_mm_unpacklo_epi8(prod, zero), zero);
            __m128i p1 = _mm_unpackhi_epi16(_mm_unpacklo_epi8(prod, zero), zero);
            __m128i p2 = _mm_unpacklo_epi16(_mm_unpackhi_epi8(prod, zero), zero);
            __m128i p3 = _mm_unpackhi_epi16(_mm_unpackhi_epi8(prod, zero), zero);

            acc0 = _mm_add_epi32(acc0, p0);
            acc1 = _mm_add_epi32(acc1, p1);
            acc2 = _mm_add_epi32(acc2, p2);
            acc3 = _mm_add_epi32(acc3, p3);
        }

        _mm_storeu_si128((__m128i*)&C[j + 0],  acc0);
        _mm_storeu_si128((__m128i*)&C[j + 4],  acc1);
        _mm_storeu_si128((__m128i*)&C[j + 8],  acc2);
        _mm_storeu_si128((__m128i*)&C[j + 12], acc3);
    }

    /* Scalar tail */
    for (; j < N; j++) {
        int32_t acc = 0;
        for (int k = 0; k < M; k++)
            acc += bqsm_tt[((int)x[k] << 2) | (int)W[k * N + j]];
        C[j] = acc;
    }
}
#endif

/* ─── Dispatch ──────────────────────────────────────────────────────────── */
static inline void bqsm_matmul_vec(const int8_t *x, const int8_t *W,
                                   int M, int N, int32_t *C) {
#ifdef __SSSE3__
    bqsm_matmul_vec_sse(x, W, M, N, C);
#else
    bqsm_matmul_vec_scalar(x, W, M, N, C);
#endif
}

#endif /* BQSM_KERNEL_H */