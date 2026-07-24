/* ===========================================================================
 * bqsm_kernel.h — BQSM Matmul Kernels (scalar + pshufb SIMD, cache-optimized)
 * ===========================================================================
 *   v2: Swapped loop nest — k outer, j inner. W streamed contiguously.
 *   Accumulators held across row: fits L1 for N≤2048, panels for N>2048.
 *
 *   C[j] = sum_k TT[x[k], W[k*N+j]]   for M input dims, N output dims
 */

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

/* ─── Scalar fallback ───────────────────────────────────────────────────── */
static inline void bqsm_matmul_vec_scalar(const int8_t *x, const int8_t *W,
                                          int M, int N, int32_t *C) {
    /* k-outer for contiguous W access */
    memset(C, 0, N * sizeof(int32_t));
    for (int k = 0; k < M; k++) {
        int a = (int)x[k] << 2;
        const int8_t *Wrow = W + k * N;
        for (int j = 0; j < N; j++)
            C[j] += bqsm_tt[a | (int)Wrow[j]];
    }
}

/* ─── SIMD (SSSE3: pshufb, k-outer, contiguous W stream) ──────────────── */
#ifdef __SSSE3__

/* Panel size: keep accumulators in L1D (32KB).
 * int32 accs: PANEL ≤ 8192 to fit 32KB.
 * Use PANEL=2048 for safety with other L1 uses. */
#define BQSM_PANEL 2048

/* Process a panel of N columns at a time, streaming W contiguously.
 * Called with pre-zeroed accumulator panel C[0..np-1]. */
static inline void bqsm_sse_panel(const int8_t *x, const int8_t *W,
                                  int M, int N, int np, int32_t *C) {
    __m128i tt   = _mm_load_si128((__m128i*)bqsm_tt);
    __m128i zero = _mm_setzero_si128();
    __m128i mask3 = _mm_set1_epi8(3);

    /* k-outer: stream W rows contiguously */
    for (int k = 0; k < M; k++) {
        int a = (int)x[k];
        __m128i a4 = _mm_set1_epi8((char)(a << 2));
        const int8_t *Wrow = W + k * N;

        int j = 0;
        for (; j + 15 < np; j += 16) {
            /* Load 16 contiguous weight bytes — hardware prefetcher engages */
            __m128i w = _mm_loadu_si128((__m128i*)&Wrow[j]);
            w = _mm_and_si128(w, mask3);

            /* pshufb: TT[activation | weight] → 16 results */
            __m128i prod = _mm_shuffle_epi8(tt, _mm_or_si128(a4, w));

            /* Zero-extend int8 → int32, accumulate */
            __m128i p0 = _mm_unpacklo_epi16(_mm_unpacklo_epi8(prod, zero), zero);
            __m128i p1 = _mm_unpackhi_epi16(_mm_unpacklo_epi8(prod, zero), zero);
            __m128i p2 = _mm_unpacklo_epi16(_mm_unpackhi_epi8(prod, zero), zero);
            __m128i p3 = _mm_unpackhi_epi16(_mm_unpackhi_epi8(prod, zero), zero);

            __m128i *cp = (__m128i*)&C[j];
            cp[0] = _mm_add_epi32(cp[0], p0);
            cp[1] = _mm_add_epi32(cp[1], p1);
            cp[2] = _mm_add_epi32(cp[2], p2);
            cp[3] = _mm_add_epi32(cp[3], p3);
        }

        /* Scalar tail */
        for (; j < np; j++)
            C[j] += bqsm_tt[a | (int)Wrow[j]];
    }
}

/* Full vector matmul: split into panels if N > BQSM_PANEL */
static inline void bqsm_matmul_vec_sse(const int8_t *x, const int8_t *W,
                                       int M, int N, int32_t *C) {
    memset(C, 0, N * sizeof(int32_t));
    for (int j0 = 0; j0 < N; j0 += BQSM_PANEL) {
        int np = (j0 + BQSM_PANEL <= N) ? BQSM_PANEL : N - j0;
        bqsm_sse_panel(x, W + j0, M, N, np, C + j0);
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