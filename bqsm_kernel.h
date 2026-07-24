/* ===========================================================================
 * bqsm_kernel.h — BQSM Matmul Kernels v3
 *
 *   k-outer nest, N-blocked accumulators (L1D-resident), contiguous W stream.
 *   Panel-major weight repack forthcoming.
 *   BQSM_PANEL = 2048 cols → int32 accs = 8KB → L1D resident.
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
#define BQSM_PANEL 2048  /* cols per panel: 2048 int32 = 8KB → L1D */

static int8_t bqsm_tt[16] __attribute__((aligned(16)));

static inline void bqsm_tt_init(void) {
    for (int a = 0; a <= BQSM_Q; a++)
        for (int b = 0; b <= BQSM_Q; b++) {
            int p = (a * b + 1) / 2;
            bqsm_tt[(a << 2) | b] = (int8_t)(p < 0 ? 0 : (p > BQSM_Q ? BQSM_Q : p));
        }
}

/* ─── Scalar fallback (k-outer, N-blocked) ─────────────────────────────── */
static inline void bqsm_matmul_vec_scalar(const int8_t *x, const int8_t *W,
                                          int M, int N, int32_t *C) {
    memset(C, 0, N * sizeof(int32_t));
    for (int j0 = 0; j0 < N; j0 += BQSM_PANEL) {
        int np = (j0 + BQSM_PANEL <= N) ? BQSM_PANEL : N - j0;
        /* Accumulators live in L1 for this panel traversal */
        for (int k = 0; k < M; k++) {
            int a = (int)x[k] << 2;
            const int8_t *Wrow = W + k * N + j0;
            for (int j = 0; j < np; j++)
                C[j0 + j] += bqsm_tt[a | (int)Wrow[j]];
        }
    }
}

/* ─── SIMD (SSSE3: pshufb, N-blocked accumulators) ─────────────────────── */
#ifdef __SSSE3__

static inline void bqsm_matmul_vec_sse(const int8_t *x, const int8_t *W,
                                       int M, int N, int32_t *C) {
    __m128i tt    = _mm_load_si128((__m128i*)bqsm_tt);
    __m128i zero  = _mm_setzero_si128();
    __m128i mask3 = _mm_set1_epi8(3);

    memset(C, 0, N * sizeof(int32_t));

    /* N-blocking: accumulators stay in L1D */
    for (int j0 = 0; j0 < N; j0 += BQSM_PANEL) {
        int np = (j0 + BQSM_PANEL <= N) ? BQSM_PANEL : N - j0;

        /* Walk all k for this panel — accumulators are hot in L1 */
        for (int k = 0; k < M; k++) {
            __m128i a4 = _mm_set1_epi8((char)((int)x[k] << 2));
            const int8_t *Wrow = W + k * N + j0;

            int j = 0;
            for (; j + 15 < np; j += 16) {
                __m128i w    = _mm_loadu_si128((__m128i*)&Wrow[j]);
                w             = _mm_and_si128(w, mask3);
                __m128i prod = _mm_shuffle_epi8(tt, _mm_or_si128(a4, w));

                /* Zero-extend int8 → int32 */
                __m128i p0 = _mm_unpacklo_epi16(_mm_unpacklo_epi8(prod, zero), zero);
                __m128i p1 = _mm_unpackhi_epi16(_mm_unpacklo_epi8(prod, zero), zero);
                __m128i p2 = _mm_unpacklo_epi16(_mm_unpackhi_epi8(prod, zero), zero);
                __m128i p3 = _mm_unpackhi_epi16(_mm_unpackhi_epi8(prod, zero), zero);

                __m128i *cp = (__m128i*)&C[j0 + j];
                cp[0] = _mm_add_epi32(cp[0], p0);
                cp[1] = _mm_add_epi32(cp[1], p1);
                cp[2] = _mm_add_epi32(cp[2], p2);
                cp[3] = _mm_add_epi32(cp[3], p3);
            }
            for (; j < np; j++)
                C[j0 + j] += bqsm_tt[((int)x[k] << 2) | (int)Wrow[j]];
        }
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