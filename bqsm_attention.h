/* ============================================================================
 * bqsm_attention.h — BQSM Attention v1.0 (scalar reference)
 *
 *   Gemma-2-2B specifics:
 *     head_dim = 256 (NOT d_model/n_heads = 288)
 *     GQA: 8 Q-heads / 4 KV-heads (kv_head = q_head / 2)
 *     Alternating local (sliding 4096) / global attention by layer
 *     Logit soft-capping: tanh(logit / cap) * cap (attn=50, final=30)
 *
 *   QKVO projections: weight matmuls (signed kernel, packed 2-bit)
 *   Q·K^T + attn·V: activation matmuls (scalar int8 dot + float softmax)
 *
 *   Numerical path:
 *     int8 (0-3) activations throughout
 *     Q·K^T in int32 dot products
 *     Scale by 1/sqrt(head_dim) = 1/16
 *     Float softmax (exp/div, memory-bound so float is fine)
 *     Weighted V sum in float → requantize to int8
 * ========================================================================== */

#ifndef BQSM_ATTENTION_H
#define BQSM_ATTENTION_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bqsm_model.h"
#include "bqsm_kernel.h"

/* ─── Gemma-2-2B architecture constants (hardcoded until model format carries them) */
#define ATT_HEAD_DIM      256    /* NOT d_model/n_heads! */
#define ATT_N_Q_HEADS       8
#define ATT_N_KV_HEADS      4
#define ATT_LOCAL_WINDOW 4096
#define ATT_LOGIT_CAP     50.0f
#define ATT_FINAL_CAP     30.0f

/* GQA mapping: each KV head serves 2 Q-heads */
static inline int att_kv_head(int q_head) { return q_head / 2; }

/* Alternating: even layers = local sliding window, odd = global */
static inline int att_is_local(int layer) { return (layer & 1) == 0; }

/* ─── KV Cache ─────────────────────────────────────────────────────────── */

typedef struct {
    int8_t *k;        /* [n_kv_heads * max_seq * head_dim] BQSM int8 states */
    int8_t *v;        /* [n_kv_heads * max_seq * head_dim] */
    int     seq_len;  /* tokens cached so far */
    int     max_seq;  /* allocated capacity */
    int     is_local; /* enforce sliding window */
} att_kv_cache_t;

static void att_kv_free(att_kv_cache_t *c, int n_layers);

static att_kv_cache_t *att_kv_alloc(int n_layers, int global_seq, int local_seq) {
    att_kv_cache_t *c = calloc((size_t)n_layers, sizeof(att_kv_cache_t));
    if (!c) { fprintf(stderr, "KV alloc: OOM for %d layer structs\n", n_layers); return NULL; }
    size_t total = 0;
    for (int L = 0; L < n_layers; L++) {
        int cap = att_is_local(L) ? local_seq : global_seq;
        size_t sz = (size_t)ATT_N_KV_HEADS * (size_t)cap * ATT_HEAD_DIM;
        c[L].k = calloc(1, sz);
        c[L].v = calloc(1, sz);
        if (!c[L].k || !c[L].v) {
            fprintf(stderr, "KV alloc: layer %d OOM (cap=%d, sz=%zu MB)\n",
                    L, cap, sz >> 20);
            att_kv_free(c, n_layers);
            return NULL;
        }
        c[L].max_seq = cap;
        c[L].seq_len = 0;
        c[L].is_local = att_is_local(L);
        total += sz * 2;
    }
    fprintf(stderr, "KV cache: %d layers, %.1f MB total (global=%d local=%d)\n",
            n_layers, total / (1024.0*1024.0), global_seq, local_seq);
    return c;
}

static void att_kv_free(att_kv_cache_t *c, int n_layers) {
    if (!c) return;
    for (int L = 0; L < n_layers; L++) {
        free(c[L].k);
        free(c[L].v);
    }
    free(c);
}

/* Store new K/V for a layer into the cache at position seq_len */
static void att_kv_store(att_kv_cache_t *c, int layer,
                         const int8_t *k_new, const int8_t *v_new, int head_dim) {
    if (layer < 0 || !c) return;
    att_kv_cache_t *kv = &c[layer];
    int pos = kv->seq_len;
    if (pos >= kv->max_seq) return;  /* cache full — should shift in prod */
    size_t row_sz = (size_t)ATT_N_KV_HEADS * (size_t)head_dim;
    memcpy(kv->k + pos * row_sz, k_new, row_sz);
    memcpy(kv->v + pos * row_sz, v_new, row_sz);
    kv->seq_len++;
}

/* ─── Softmax (float — memory-bound path, float is fine) ──────────────── */

static void att_softmax(float *scores, int n) {
    /* Find max for numerical stability */
    float mx = scores[0];
    for (int i = 1; i < n; i++)
        if (scores[i] > mx) mx = scores[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        scores[i] = expf(scores[i] - mx);
        sum += scores[i];
    }

    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int i = 0; i < n; i++)
            scores[i] *= inv;
    }
}

/* ─── Attention (scalar reference, single token, KV-cached) ───────────── */

/* Run one attention layer. x is [d_model] int8 activation (0-3).
 * q_w/k_w/v_w/o_w are packed 2-bit weight tensors from the model.
 * Output written to out[d_model] as int8 (0-3). */
static void bqsm_attention_layer(
    const int8_t *x, int d_model,
    const bqsm_tensor_t *q_w, const bqsm_tensor_t *k_w,
    const bqsm_tensor_t *v_w, const bqsm_tensor_t *o_w,
    int layer, att_kv_cache_t *kv_cache,
    int8_t *out, int32_t *o_raw)
{
    int hd      = ATT_HEAD_DIM;
    int n_qh    = ATT_N_Q_HEADS;
    int n_kvh   = ATT_N_KV_HEADS;
    int q_dim   = n_qh  * hd;  /* 2048 */
    int kv_dim  = n_kvh * hd;  /* 1024 */

    /* ── Q/K/V projections (weight matmuls — signed kernel, packed 2-bit) ── */
    int32_t *q_acc = calloc((size_t)q_dim,  sizeof(int32_t));
    int32_t *k_acc = calloc((size_t)kv_dim, sizeof(int32_t));
    int32_t *v_acc = calloc((size_t)kv_dim, sizeof(int32_t));
    int8_t  *q_out = calloc((size_t)q_dim, 1);
    int8_t  *k_out = calloc((size_t)kv_dim, 1);
    int8_t  *v_out = calloc((size_t)kv_dim, 1);

    if (!q_acc || !k_acc || !v_acc || !q_out || !k_out || !v_out) {
        free(q_acc); free(k_acc); free(v_acc);
        free(q_out); free(k_out); free(v_out);
        memcpy(out, x, (size_t)d_model); /* passthrough on OOM */
        return;
    }

    if (q_w) bqsm_matmul_vec(x, q_w->data, d_model, q_dim,  q_acc);
    if (k_w) bqsm_matmul_vec(x, k_w->data, d_model, kv_dim, k_acc);
    if (v_w) bqsm_matmul_vec(x, v_w->data, d_model, kv_dim, v_acc);

    /* Quantize Q/K/V to int8 (0-3) with gentle quantization */
    for (int i = 0; i < q_dim;  i++) {
        float v = (float)q_acc[i] / (float)(d_model * 2 + 1) + 1.5f;
        int iv = (int)(v < 0 ? 0 : (v > BQSM_Q ? BQSM_Q : v));
        q_out[i] = (int8_t)iv;
    }
    for (int i = 0; i < kv_dim; i++) {
        float v = (float)k_acc[i] / (float)(d_model * 2 + 1) + 1.5f;
        int iv = (int)(v < 0 ? 0 : (v > BQSM_Q ? BQSM_Q : v));
        k_out[i] = (int8_t)iv;
    }
    for (int i = 0; i < kv_dim; i++) {
        float v = (float)v_acc[i] / (float)(d_model * 2 + 1) + 1.5f;
        int iv = (int)(v < 0 ? 0 : (v > BQSM_Q ? BQSM_Q : v));
        v_out[i] = (int8_t)iv;
    }

    free(q_acc); free(k_acc); free(v_acc);

    /* ── Store K/V in cache ─────────────────────────────────────────── */
    att_kv_store(kv_cache, layer, k_out, v_out, hd);
    int seq_len = kv_cache[layer].seq_len;

    /* ── Attention: Q·K^T + softmax + weighted V ────────────────────── */
    /* Allocate scratch for one head's scores and attn-weighted output */
    float *scores = calloc((size_t)kv_cache[layer].max_seq, sizeof(float));
    float *head_out = calloc((size_t)hd, sizeof(float));
    float *attn_cat = calloc((size_t)q_dim, sizeof(float));

    if (!scores || !head_out || !attn_cat) {
        free(scores); free(head_out); free(attn_cat);
        free(q_out); free(k_out); free(v_out);
        memcpy(out, x, (size_t)d_model);
        return;
    }

    att_kv_cache_t *kv = &kv_cache[layer];
    int window = kv->is_local ? ATT_LOCAL_WINDOW : seq_len;
    float scale = 1.0f / sqrtf((float)hd);  /* 1/sqrt(256) = 1/16 */

    for (int qh = 0; qh < n_qh; qh++) {
        int kvh = att_kv_head(qh);
        const int8_t *q_head = q_out + qh * hd;

        /* Determine valid token range: causal + sliding window */
        int first = kv->is_local ? (seq_len > window ? seq_len - window : 0) : 0;
        int n_valid = 0;

        /* Compute Q·K scores for valid positions */
        for (int pos = first; pos < seq_len; pos++) {
            const int8_t *k_pos = kv->k + (pos * n_kvh + kvh) * hd;
            int32_t dot = 0;
            for (int d = 0; d < hd; d++)
                dot += (int32_t)q_head[d] * (int32_t)k_pos[d];
            /* Scale by 1/sqrt(hd) */
            float raw = (float)dot * scale;
            /* Logit soft-capping: capped = cap * tanh(raw / cap) */
            float capped = ATT_LOGIT_CAP * tanhf(raw / ATT_LOGIT_CAP);
            scores[n_valid++] = capped;
        }

        /* Softmax */
        att_softmax(scores, n_valid);

        /* Weighted V sum */
        memset(head_out, 0, (size_t)hd * sizeof(float));
        for (int p = 0; p < n_valid; p++) {
            int pos = first + p;
            const int8_t *v_pos = kv->v + (pos * n_kvh + kvh) * hd;
            float w = scores[p];
            for (int d = 0; d < hd; d++)
                head_out[d] += w * (float)v_pos[d];
        }

        /* Copy to concatenated output */
        memcpy(attn_cat + qh * hd, head_out, (size_t)hd * sizeof(float));
    }

    free(scores); free(head_out);
    free(q_out); free(k_out); free(v_out);

    /* ── Output projection: attn_cat [q_dim] → o_weight [d_model][q_dim] → out [d_model] ── */
    /* o_weight is stored as [d_model rows × q_dim cols] packed 2-bit.
     * We need: out[i] = sum_j attn_cat[j] * weight[i][j]
     * This is a transpose matmul — can't use bqsm_matmul_vec directly.
     * Scalar loop (one per layer per token — negligible).
     * Raw int32 accumulator written to o_raw for caller to scale+residual. */
    if (o_w) {
        const uint8_t *wp = (const uint8_t *)o_w->data;
        int n_out   = (int)o_w->shape[0];  /* d_model */
        int n_in    = (int)o_w->shape[1];  /* q_dim */
        for (int i = 0; i < n_out; i++) {
            int32_t sum = 0;
            for (int j = 0; j < n_in; j++) {
                int byte_idx = (i * n_in + j) / 4;
                int shift    = ((i * n_in + j) & 3) * 2;
                int nb       = (wp[byte_idx] >> shift) & 3;
                float wf = ((float)nb - 1.5f);  /* center nibble at 0 */
                sum += (int32_t)(attn_cat[j] * wf);
            }
            o_raw[i] = sum;
        }
        /* Also write a quantized int8 version to out for debug */
        for (int i = 0; i < n_out; i++) {
            float v = (float)o_raw[i] / (float)(n_in * 2) + 1.5f;
            int iv = (int)(v < 0 ? 0 : (v > BQSM_Q ? BQSM_Q : v));
            out[i] = (int8_t)iv;
        }
    } else {
        memset(o_raw, 0, (size_t)d_model * sizeof(int32_t));
        memcpy(out, x, (size_t)d_model);
    }

    free(attn_cat);
}

#endif /* BQSM_ATTENTION_H */
