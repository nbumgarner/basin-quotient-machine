/* ===========================================================================
 * bqsm_kernel.h — BQSM Matmul Kernels v4 (packed weights, in-register unpack)
 *
 *   Packed 2-bit weights: 4 values per byte. Unpack in-register via
 *   _mm_srli_epi32 + mask — no pshufb needed for unpacking.
 *   Accumulators held in permuted order (stride-4), un-permuted once at output.
 *   4× less memory traffic vs unpacked int8 arena.
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
#define BQSM_PANEL 2048

static int8_t bqsm_tt[16] __attribute__((aligned(16)));

static inline void bqsm_tt_init(void) {
    for (int a = 0; a <= BQSM_Q; a++)
        for (int b = 0; b <= BQSM_Q; b++) {
            int p = (a * b + 1) / 2;
            bqsm_tt[(a << 2) | b] = (int8_t)(p < 0 ? 0 : (p > BQSM_Q ? BQSM_Q : p));
        }
}

/* ─── Scalar fallback (unpacked arena, k-outer) ────────────────────────── */
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

/* ─── SIMD: packed weights, in-register unpack ─────────────────────────── */
#ifdef __SSSE3__

/* Un-permute: stride-4 → sequential.
 * Input: acc[p][i] corresponds to output j = p + 4*i.
 * Output: C[j] = acc[j%4][j/4].
 */
static inline void unpermute_4x16(const int32_t *acc, int32_t *C) {
    for (int i = 0; i < 16; i++) {
        C[4*i + 0] = acc[0*16 + i];
        C[4*i + 1] = acc[1*16 + i];
        C[4*i + 2] = acc[2*16 + i];
        C[4*i + 3] = acc[3*16 + i];
    }
}

/* Process 64 output columns (16 packed bytes = 64 weights) for all k.
 * Accumulators in permuted order, un-permuted once at output.
 * Returns pointer past processed columns. */
static inline void bqsm_sse_64cols(const int8_t *x, const uint8_t *W_packed,
                                   int M, int N, int j0,
                                   int32_t *C_panel) {
    __m128i tt    = _mm_load_si128((__m128i*)bqsm_tt);
    __m128i zero  = _mm_setzero_si128();
    __m128i m03   = _mm_set1_epi8(0x03);

    /* 64 int32 accumulators in permuted order: 4 phases × 16 values each.
     * Phase p: columns j = j0 + p + 4*i for i=0..15.
     * Total: 64 int32 = 16 XMM registers. Tight but fits on SSE. */
    __m128i acc[16] = {zero,zero,zero,zero, zero,zero,zero,zero,
                       zero,zero,zero,zero, zero,zero,zero,zero};

    for (int k = 0; k < M; k++) {
        __m128i a4  = _mm_set1_epi8((char)((int)x[k] << 2));
        /* Load 16 packed bytes = 64 weights for this row at j0..j0+63 */
        __m128i p   = _mm_loadu_si128((__m128i*)&W_packed[k * (N/4) + j0/4]);

        /* Unpack 4 nibble phases in-register */
        __m128i w0 = _mm_and_si128(p, m03);
        __m128i w1 = _mm_and_si128(_mm_srli_epi32(p, 2),  m03);
        __m128i w2 = _mm_and_si128(_mm_srli_epi32(p, 4),  m03);
        __m128i w3 = _mm_and_si128(_mm_srli_epi32(p, 6),  m03);

        /* TT lookup per phase: prod_p = pshufb(tt, a4 | w_p) */
        __m128i p0 = _mm_shuffle_epi8(tt, _mm_or_si128(a4, w0));
        __m128i p1 = _mm_shuffle_epi8(tt, _mm_or_si128(a4, w1));
        __m128i p2 = _mm_shuffle_epi8(tt, _mm_or_si128(a4, w2));
        __m128i p3 = _mm_shuffle_epi8(tt, _mm_or_si128(a4, w3));

        /* Zero-extend int8→int32 and accumulate. 16 acc regs.
         * Phase p, reg r (r=0..3): acc[p*4+r] += extend(prod_p, r*4..r*4+3) */
        #define ACCUM(p, prod) do { \
            __m128i lo16_0 = _mm_unpacklo_epi8(prod, zero); \
            __m128i hi16_0 = _mm_unpackhi_epi8(prod, zero); \
            acc[p*4+0] = _mm_add_epi32(acc[p*4+0], _mm_unpacklo_epi16(lo16_0, zero)); \
            acc[p*4+1] = _mm_add_epi32(acc[p*4+1], _mm_unpackhi_epi16(lo16_0, zero)); \
            acc[p*4+2] = _mm_add_epi32(acc[p*4+2], _mm_unpacklo_epi16(hi16_0, zero)); \
            acc[p*4+3] = _mm_add_epi32(acc[p*4+3], _mm_unpackhi_epi16(hi16_0, zero)); \
        } while(0)

        ACCUM(0, p0);
        ACCUM(1, p1);
        ACCUM(2, p2);
        ACCUM(3, p3);
        #undef ACCUM
    }

    /* Un-permute once: store to stack, then stride-4 → sequential */
    int32_t tmp[64] __attribute__((aligned(16)));
    for (int p = 0; p < 4; p++) {
        _mm_storeu_si128((__m128i*)&tmp[p*16 + 0],  acc[p*4+0]);
        _mm_storeu_si128((__m128i*)&tmp[p*16 + 4],  acc[p*4+1]);
        _mm_storeu_si128((__m128i*)&tmp[p*16 + 8],  acc[p*4+2]);
        _mm_storeu_si128((__m128i*)&tmp[p*16 + 12], acc[p*4+3]);
    }
    unpermute_4x16(tmp, C_panel);
}

/* Full vector matmul: panel-based N-blocking, packed weights. */
static inline void bqsm_matmul_vec_sse(const int8_t *x, const uint8_t *W_packed,
                                       int M, int N, int32_t *C) {
    memset(C, 0, N * sizeof(int32_t));

    /* Process 64 columns at a time (one 16-byte packed load = 64 weights per k) */
    for (int j0 = 0; j0 < N; j0 += 64) {
        int np = (j0 + 64 <= N) ? 64 : N - j0;
        if (np == 64) {
            bqsm_sse_64cols(x, W_packed, M, N, j0, C + j0);
        } else {
            /* Scalar tail for partial last block */
            for (int k = 0; k < M; k++) {
                int a = (int)x[k] << 2;
                for (int j = j0; j < N; j++) {
                    /* Extract from packed: j/4 byte, nibble j%4 */
                    int byte_off = k * (N/4) + j/4;
                    uint8_t b = W_packed[byte_off];
                    int w = (b >> (2 * (j % 4))) & 3;
                    C[j] += bqsm_tt[a | w];
                }
            }
        }
    }
}
#endif

/* ─── Dispatch ──────────────────────────────────────────────────────────── */
static inline void bqsm_matmul_vec(const int8_t *x, const int8_t *W,
                                   int M, int N, int32_t *C) {
#ifdef __SSSE3__
    bqsm_matmul_vec_sse(x, (const uint8_t*)W, M, N, C);
#else
    bqsm_matmul_vec_scalar(x, W, M, N, C);
#endif
}

#endif /* BQSM_KERNEL_H */