/* int8_q8.c (aarch64, needs the dot-product extension) — int8 weights x int8
 * activations with SDOT. EXPERIMENT: this changes the numerics.
 *
 *   y[o] = wscale[o] * sum_blocks  xscale[b] * ( sum_{i in block b} W[o,i] * xq[i] )
 *
 * Weights are the engine's existing int8 rows with a per-row scale. Activations
 * are quantised per call, in blocks of QB values with their own scale
 * (absmax / 127, round to nearest), so a massive-activation channel only crushes
 * its own block instead of the whole vector. Within a block the products are
 * exact int32 sums (vdotq_s32: 16 int8 multiply-adds per instruction, 4 lanes);
 * each block's 4-lane partial sum is converted to f32 once and scaled.
 *
 * Overflow: one vdotq lane adds 4 products per call, QB/16 calls per block,
 * so at most QB/4 * 127 * 127 per lane per block — 129,032 at QB = 32.
 *
 * cc -O3 -march=armv8.2-a+dotprod -fopenmp -shared -fPIC -o libint8q8.so int8_q8.c
 */
#include <arm_neon.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/* quantise n floats in blocks of qb: xq (int8), xs (per-block scale) */
static void quant_blocks(const float *x, int n, int qb, int8_t *xq, float *xs)
{
    for (int b = 0; b * qb < n; ++b) {
        const float *v = x + (size_t)b * qb;
        float amax = 0.0f;
        for (int k = 0; k < qb; ++k) { float a = fabsf(v[k]); if (a > amax) amax = a; }
        float s = amax / 127.0f, inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        xs[b] = s;
        for (int k = 0; k < qb; ++k) {
            float r = rintf(v[k] * inv);
            if (r > 127.0f) r = 127.0f;
            if (r < -127.0f) r = -127.0f;
            xq[(size_t)b * qb + k] = (int8_t)r;
        }
    }
}

static inline float row_dot(const int8_t *w, const int8_t *xq, const float *xs, int nin, int qb)
{
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    int nb = nin / qb, b = 0;
    for (; b + 2 <= nb; b += 2) {                      /* two blocks per iteration */
        int32x4_t d0 = vdupq_n_s32(0), d1 = vdupq_n_s32(0);
        const int8_t *w0 = w + (size_t)b * qb, *x0 = xq + (size_t)b * qb;
        const int8_t *w1 = w0 + qb, *x1 = x0 + qb;
        for (int k = 0; k < qb; k += 16) {
            d0 = vdotq_s32(d0, vld1q_s8(w0 + k), vld1q_s8(x0 + k));
            d1 = vdotq_s32(d1, vld1q_s8(w1 + k), vld1q_s8(x1 + k));
        }
        acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(d0), xs[b]);
        acc1 = vfmaq_n_f32(acc1, vcvtq_f32_s32(d1), xs[b + 1]);
    }
    for (; b < nb; ++b) {
        int32x4_t d0 = vdupq_n_s32(0);
        const int8_t *w0 = w + (size_t)b * qb, *x0 = xq + (size_t)b * qb;
        for (int k = 0; k < qb; k += 16) d0 = vdotq_s32(d0, vld1q_s8(w0 + k), vld1q_s8(x0 + k));
        acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(d0), xs[b]);
    }
    return vaddvq_f32(vaddq_f32(acc0, acc1));
}

/* Y[B, nout] = X[B, nin] @ W[nout, nin]^T with W int8 + per-row scale.
 * nin must be a multiple of qb, qb a multiple of 16. Returns 0, or -1 on bad shape. */
int int8_gemm_q8(const int8_t *W, const float *scale, const float *X, float *Y,
                 int nout, int nin, int B, int qb)
{
    if (qb < 16 || qb % 16 || nin % qb) return -1;
    int nb = nin / qb;
    int8_t *xq = malloc((size_t)B * nin);
    float *xs = malloc(sizeof(float) * (size_t)B * nb);
    if (!xq || !xs) { free(xq); free(xs); return -1; }
    for (int r = 0; r < B; ++r) quant_blocks(X + (size_t)r * nin, nin, qb, xq + (size_t)r * nin, xs + (size_t)r * nb);
#pragma omp parallel for schedule(static)
    for (int o = 0; o < nout; ++o) {
        const int8_t *w = W + (size_t)o * (size_t)nin;
        for (int r = 0; r < B; ++r)
            Y[(size_t)r * nout + o] = row_dot(w, xq + (size_t)r * nin, xs + (size_t)r * nb, nin, qb) * scale[o];
    }
    free(xq); free(xs);
    return 0;
}
