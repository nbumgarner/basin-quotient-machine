/* ===========================================================================
 * bqsm_kernel.h — BQSM Matmul Kernels v6 (signed path, cvtepi8, phase-serial)
 *
 *   Ternary weights (-1/0/1) packed 2-bit. Signed path:
 *   pshufb LUT → sign_epi8 → cvtepi8_epi16 → accumulate in int16.
 *   5 ops per 16 MACs (vs 11 for unsigned TT path).
 *   int16 accumulators: max 9216 → no overflow. Widen to int32 at output.
 * ======================================================================== */

#ifndef BQSM_KERNEL_H
#define BQSM_KERNEL_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#ifdef __SSSE3__
#include <tmmintrin.h>
#include <smmintrin.h>  /* SSE4.1 for _mm_cvtepi8_epi16 */
#endif

#define BQSM_Q 3

/* ─── TT table (unsigned path, kept for backward compat) ───────────────── */
static int8_t bqsm_tt[16] __attribute__((aligned(16)));
static inline void bqsm_tt_init(void) {
    for (int a = 0; a <= BQSM_Q; a++)
        for (int b = 0; b <= BQSM_Q; b++) {
            int p = (a * b + 1) / 2;
            bqsm_tt[(a << 2) | b] = (int8_t)(p < 0 ? 0 : (p > BQSM_Q ? BQSM_Q : p));
        }
}

/* ─── Ternary LUT: nibble (0,1,2) → int8 (-1,0,1) ─────────────────────── */
static const int8_t bqsm_ternary_lut[16] __attribute__((aligned(16))) = {
    -1, 0, 1, 0,  -1, 0, 1, 0,  -1, 0, 1, 0,  -1, 0, 1, 0
};

/* ─── Scalar fallback ───────────────────────────────────────────────────── */
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

/* ─── SIMD signed path (SSE4.1: cvtepi8, phase-serial) ────────────────── */
#if defined(__SSSE3__) && defined(__SSE4_1__)

/* Process one nibble phase for 64 output columns.
 * Phase p: columns j = j0 + p + 4*i for i=0..15.
 * Uses int16 accumulators (2 regs = 16 int16 values per phase).
 * Widen to int32 at output, store at stride 4. */
static inline void bqsm_sse_phase_signed(const int8_t *x,
                                         const uint8_t *W_packed,
                                         int M, int N, int j0, int phase,
                                         int32_t *C) {
    __m128i lut   = _mm_load_si128((__m128i*)bqsm_ternary_lut);
    __m128i zero  = _mm_setzero_si128();
    __m128i m03   = _mm_set1_epi8(0x03);
    int shift     = phase * 2;

    /* int16 accumulators: 16 values → 2 XMM regs */
    __m128i acc_lo = zero, acc_hi = zero;

    for (int k = 0; k < M; k++) {
        __m128i act = _mm_set1_epi8(x[k]);
        __m128i p   = _mm_loadu_si128((__m128i*)&W_packed[k * (N/4) + j0/4]);

        /* Extract nibble phase, map to ternary via pshufb LUT */
        __m128i nb  = _mm_and_si128(_mm_srli_epi32(p, shift), m03);
        __m128i w   = _mm_shuffle_epi8(lut, nb);

        /* Element-wise signed multiply: sign_epi8(act, w) → int8 {-1,0,1} */
        __m128i pr  = _mm_sign_epi8(act, w);

        /* Widen int8 → int16 (SSE4.1). 5 ops total per iteration. */
        __m128i lo  = _mm_cvtepi8_epi16(pr);
        __m128i hi  = _mm_cvtepi8_epi16(_mm_srli_si128(pr, 8));

        acc_lo = _mm_add_epi16(acc_lo, lo);
        acc_hi = _mm_add_epi16(acc_hi, hi);
    }

    /* Widen int16 → int32 and store at stride-4 positions */
    int32_t tmp[16] __attribute__((aligned(16)));
    _mm_store_si128((__m128i*)&tmp[0],  _mm_cvtepi16_epi32(acc_lo));
    _mm_store_si128((__m128i*)&tmp[4],  _mm_cvtepi16_epi32(_mm_srli_si128(acc_lo, 8)));
    _mm_store_si128((__m128i*)&tmp[8],  _mm_cvtepi16_epi32(acc_hi));
    _mm_store_si128((__m128i*)&tmp[12], _mm_cvtepi16_epi32(_mm_srli_si128(acc_hi, 8)));

    int32_t *out = C + j0 + phase;
    for (int i = 0; i < 16; i++)
        out[i * 4] += tmp[i];
}

static inline void bqsm_matmul_vec_sse(const int8_t *x, const uint8_t *W_packed,
                                       int M, int N, int32_t *C) {
    memset(C, 0, N * sizeof(int32_t));

    for (int j0 = 0; j0 < N; j0 += 64) {
        int np = (j0 + 64 <= N) ? 64 : N - j0;
        if (np == 64) {
            for (int phase = 0; phase < 4; phase++)
                bqsm_sse_phase_signed(x, W_packed, M, N, j0, phase, C);
        } else {
            /* Scalar tail */
            for (int k = 0; k < M; k++) {
                int8_t a = x[k];
                for (int j = j0; j < N; j++) {
                    int bi = k * (N/4) + j/4;
                    int nb = (W_packed[bi] >> (2 * (j % 4))) & 3;
                    int8_t w = bqsm_ternary_lut[nb];
                    C[j] += (int)a * (int)w;
                }
            }
        }
    }
}
#endif

/* ─── Dispatch ──────────────────────────────────────────────────────────── */
static inline void bqsm_matmul_vec(const int8_t *x, const int8_t *W,
                                   int M, int N, int32_t *C) {
#if defined(__SSSE3__) && defined(__SSE4_1__)
    bqsm_matmul_vec_sse(x, (const uint8_t*)W, M, N, C);
#else
    bqsm_matmul_vec_scalar(x, W, M, N, C);
#endif
}

#endif /* BQSM_KERNEL_H */