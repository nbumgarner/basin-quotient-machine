/* ===========================================================================
 * bqsm_tokenizer.h — Minimal embedded tokenizer for BQSM Server
 *
 *   Character-level tokenization: each byte maps to a token ID.
 *   Vocabulary size: 256 tokens (0x00-0xFF).
 *   Embedding dimension: d_model (2304 for Gemma 2B).
 *
 *   For production: replace with SentencePiece BPE tokenizer.
 * ======================================================================== */
#ifndef BQSM_TOKENIZER_H
#define BQSM_TOKENIZER_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define BQSM_VOCAB_SIZE 256

/* Encode text to token IDs. Returns number of tokens. Caller frees *out. */
static inline int bqsm_encode(const char *text, int **out) {
    int len = (int)strlen(text);
    *out = (int*)malloc(len * sizeof(int));
    for (int i = 0; i < len; i++)
        (*out)[i] = (unsigned char)text[i] % BQSM_VOCAB_SIZE;
    return len;
}

/* Decode token IDs to text. Returns malloc'd string. */
static inline char *bqsm_decode(const int *tokens, int n) {
    char *s = (char*)malloc(n + 1);
    for (int i = 0; i < n; i++)
        s[i] = (char)(tokens[i] & 0xFF);
    s[n] = 0;
    return s;
}

/* Generate embedding vector from text (simple hash-based).
   Maps text → d_model-dimensional 4-state activation vector. */
static inline void bqsm_embed(const char *text, int d_model, int8_t *out) {
    int len = (int)strlen(text);
    memset(out, 2, d_model);  /* center at 2 */
    for (int i = 0; i < d_model; i++) {
        unsigned char c1 = (unsigned char)text[i % (len ? len : 1)];
        unsigned char c2 = (unsigned char)text[(i * 7 + 13) % (len ? len : 1)];
        out[i] = (int8_t)((c1 ^ c2) & 3);  /* 0-3 range */
    }
}

#endif /* BQSM_TOKENIZER_H */
