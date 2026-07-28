/* ===========================================================================
 * bqsm_kernel_3bit.h — Optimized 3-bit Matmul Kernels for Harpertown (X5472)
 *
 * Target: 96 GMACs/s on Xeon X5472 (3.0 GHz, SSE4.1 + SSSE3, 8 cores)
 * Approach: PSHUFB unpack + vectorized LUT + fused multiply-accumulate
 * ======================================================================== */

#ifndef BQSM_KERNEL_3BIT_H
#define BQSM_KERNEL_3BIT_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

#define BQSM_Q_3BIT 7  /* 3-bit: 0..7 -> -3.5..3.5 */

/* ─── 3-bit Signed LUT ───────────────────────────────────────────────── */
/* 0=-3.5, 1=-2.5, 2=-1.5, 3=-0.5, 4=0.5, 5=1.5, 6=2.5, 7=3.5 */
static const float bqsm_3bit_lut[8] __attribute__((aligned(16))) = {
    -3.5f, -2.5f, -1.5f, -0.5f,
     0.5f,  1.5f,  2.5f,  3.5f
};

/* ─── Scalar kernel: 3-bit matmul (baseline) ─────────────────────────── */
static inline void bqsm_matmul_vec_3bit_scalar(const int8_t *x, const uint8_t *W_packed,
                                                int M, int N, float *C) {
    memset(C, 0, N * sizeof(float));
    for (int i = 0; i < M; i++) {
        float a = (float)x[i];
        for (int j = 0; j < N; j++) {
            int bi = i * (N/2) + j/2;
            int shift = (j % 2) * 4;
            int nb = (W_packed[bi] >> shift) & 0x0F;
            nb = nb & 7;
            float w = bqsm_3bit_lut[nb];
            C[j] += a * w;
        }
    }
}

/* ─── SSE4.1 + SSSE3 kernel: 3-bit matmul (optimized for Harpertown) ─── */
/* Process 32 weights per iteration (16 bytes = 32 nibbles = 32 MACs)    */
/* Uses PSHUFB for fast nibble unpack + LUT lookup                       */
#if defined(__SSE4_1__) && defined(__SSSE3__)
static inline void bqsm_matmul_vec_3bit_sse(const int8_t *x, const uint8_t *W_packed,
                                             int M, int N, float *C) {
    memset(C, 0, N * sizeof(float));

    /* LUT: nibble (0-15) -> biased weight (w+8) as uint8.
     * 3-bit values 0-7 -> {-7,-5,-3,-1,1,3,5,7} -> bias +8 -> {1,3,5,7,9,11,13,15}
     * Nibbles 8-15 -> 0 (unused, masked out) */
    static const uint8_t nibble_to_w_biased[16] __attribute__((aligned(16))) = {
        1, 3, 5, 7, 9, 11, 13, 15,  0, 0, 0, 0, 0, 0, 0, 0
    };

    /* PSHUFB mask: expand 16 packed bytes -> 16 low nibbles */
    static const uint8_t mask_low[16] __attribute__((aligned(16))) = {
        0,0, 1,1, 2,2, 3,3, 4,4, 5,5, 6,6, 7,7
    };
    static const uint8_t mask_high[16] __attribute__((aligned(16))) = {
        8,8, 9,9, 10,10, 11,11, 12,12, 13,13, 14,14, 15,15
    };
    const __m128i shuffle_low = _mm_load_si128((const __m128i*)mask_low);
    const __m128i shuffle_high = _mm_load_si128((const __m128i*)mask_high);
    const __m128i nibble_lut = _mm_load_si128((const __m128i*)nibble_to_w_biased);
    const __m128 lut_lo = _mm_load_ps(bqsm_3bit_lut);       /* indices 0-3 */
    const __m128 lut_hi = _mm_load_ps(bqsm_3bit_lut + 4);   /* indices 4-7 */
    const __m128 bias8 = _mm_set1_ps(8.0f);

    for (int i = 0; i < M; i++) {
        const float a = (float)x[i];
        const __m128 a_vec = _mm_set1_ps(a);

        for (int j = 0; j < N; j += 32) {
            /* Load 16 packed bytes = 32 nibbles = 32 weights */
            __m128i packed = _mm_loadu_si128((const __m128i*)(W_packed + i * (N/2) + j/2));

            /* ─── Extract low 16 nibbles ──────────────────────────────── */
            __m128i low_nibbles = _mm_and_si128(packed, _mm_set1_epi8(0x0F));
            __m128i w_biased_low = _mm_shuffle_epi8(nibble_lut, low_nibbles);

            /* ─── Extract high 16 nibbles ─────────────────────────────── */
            __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(0x0F));
            __m128i w_biased_high = _mm_shuffle_epi8(nibble_lut, high_nibbles);

            /* ─── Load 16 inputs for low weights, 16 for high weights ─── */
            __m128i x_low = _mm_loadu_si128((const __m128i*)(x + j));
            __m128i x_high = _mm_loadu_si128((const __m128i*)(x + j + 16));

            /* ─── PMADDUBSW: (w_biased * x) = (w+8)*x = w*x + 8*x ────── */
            __m128i prod_low = _mm_maddubs_epi16(w_biased_low, x_low);
            __m128i prod_high = _mm_maddubs_epi16(w_biased_high, x_high);

            /* ─── Convert int16 products to float (8 products each = 16 floats) ─── */
            __m128i prod_low_lo = _mm_unpacklo_epi16(prod_low, _mm_setzero_si128());
            __m128i prod_low_hi = _mm_unpackhi_epi16(prod_low, _mm_setzero_si128());
            __m128i prod_high_lo = _mm_unpacklo_epi16(prod_high, _mm_setzero_si128());
            __m128i prod_high_hi = _mm_unpackhi_epi16(prod_high, _mm_setzero_si128());

            __m128 fprod_low_lo = _mm_cvtepi32_ps(prod_low_lo);
            __m128 fprod_low_hi = _mm_cvtepi32_ps(prod_low_hi);
            __m128 fprod_high_lo = _mm_cvtepi32_ps(prod_high_lo);
            __m128 fprod_high_hi = _mm_cvtepi32_ps(prod_high_hi);

            /* ─── Scale by input value 'a' ────────────────────────────── */
            fprod_low_lo = _mm_mul_ps(fprod_low_lo, a_vec);
            fprod_low_hi = _mm_mul_ps(fprod_low_hi, a_vec);
            fprod_high_lo = _mm_mul_ps(fprod_high_lo, a_vec);
            fprod_high_hi = _mm_mul_ps(fprod_high_hi, a_vec);

            /* ─── Subtract bias: 8 * x * a ────────────────────────────── */
            /* Convert x inputs to float */
            __m128i x_low_16 = _mm_cvtepi8_epi16(x_low);
            __m128i x_high_16 = _mm_cvtepi8_epi16(x_high);

            __m128i x_low_lo = _mm_unpacklo_epi16(x_low_16, _mm_setzero_si128());
            __m128i x_low_hi = _mm_unpackhi_epi16(x_low_16, _mm_setzero_si128());
            __m128i x_high_lo = _mm_unpacklo_epi16(x_high_16, _mm_setzero_si128());
            __m128i x_high_hi = _mm_unpackhi_epi16(x_high_16, _mm_setzero_si128());

            __m128 fx_low_lo = _mm_mul_ps(_mm_cvtepi32_ps(x_low_lo), a_vec);
            __m128 fx_low_hi = _mm_mul_ps(_mm_cvtepi32_ps(x_low_hi), a_vec);
            __m128 fx_high_lo = _mm_mul_ps(_mm_cvtepi32_ps(x_high_lo), a_vec);
            __m128 fx_high_hi = _mm_mul_ps(_mm_cvtepi32_ps(x_high_hi), a_vec);

            fx_low_lo = _mm_mul_ps(fx_low_lo, bias8);
            fx_low_hi = _mm_mul_ps(fx_low_hi, bias8);
            fx_high_lo = _mm_mul_ps(fx_high_lo, bias8);
            fx_high_hi = _mm_mul_ps(fx_high_hi, bias8);

            fprod_low_lo = _mm_sub_ps(fprod_low_lo, fx_low_lo);
            fprod_low_hi = _mm_sub_ps(fprod_low_hi, fx_low_hi);
            fprod_high_lo = _mm_sub_ps(fprod_high_lo, fx_high_lo);
            fprod_high_hi = _mm_sub_ps(fprod_high_hi, fx_high_hi);

            /* ─── Accumulate into C (16 floats per __m128) ────────────── */
            float *c_ptr = C + j;
            _mm_storeu_ps(c_ptr,     _mm_add_ps(_mm_loadu_ps(c_ptr),     fprod_low_lo));
            _mm_storeu_ps(c_ptr + 4, _mm_add_ps(_mm_loadu_ps(c_ptr + 4), fprod_low_hi));
            _mm_storeu_ps(c_ptr + 8, _mm_add_ps(_mm_loadu_ps(c_ptr + 8), fprod_high_lo));
            _mm_storeu_ps(c_ptr + 12,_mm_add_ps(_mm_loadu_ps(c_ptr + 12), fprod_high_hi));
        }
    }
}
#endif

/* ─── Main matmul function (auto-selects best kernel) ────────────────── */
static inline void bqsm_matmul_vec_3bit(const int8_t *x, const uint8_t *W_packed,
                                         int M, int N, float *C) {
#if defined(__SSE4_1__) && defined(__SSSE3__)
    bqsm_matmul_vec_3bit_sse(x, W_packed, M, N, C);
#else
    bqsm_matmul_vec_3bit_scalar(x, W_packed, M, N, C);
#endif
}

#endif /* BQSM_KERNEL_3BIT_H */