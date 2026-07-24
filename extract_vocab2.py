#!/usr/bin/env python3
"""extract_vocab2.py — Extract SentencePiece vocab, write bqsm_vocab.h
Sorted alphabetically for binary search. Format:
  sp_vocab_data: concatenated token bytes
  sp_vocab_index: sorted array of {uint32_t offset, uint16_t len, uint32_t id}
"""
from gguf import GGUFReader
import struct, sys

GGUF = "/home/compunerd/Downloads/gemma-2-2b-it-abliterated-Q4_K_M.gguf"
OUT_H = "bqsm_vocab.h"

reader = GGUFReader(GGUF)
tok = reader.fields['tokenizer.ggml.tokens']
parts = tok.parts
data = tok.data

# Extract tokens
tokens = []
for i in range(len(data)):
    idx = data[i]
    token_bytes = bytes(parts[idx])
    tokens.append((token_bytes, i))  # (bytes, original_id)

print(f"Extracted {len(tokens)} tokens, max_len={max(len(t) for t,_ in tokens)}")

# Sort by token bytes for binary search
tokens.sort(key=lambda x: x[0])

# Build concatenated data + index
data_bytes = b''
index = []  # (offset, length, token_id)
offset = 0
for tb, tid in tokens:
    index.append((offset, len(tb), tid))
    data_bytes += tb
    offset += len(tb)

# Write C header
with open(OUT_H, 'w') as f:
    f.write(f"""/* Auto-generated — SentencePiece vocab for Gemma 2B
 * {len(tokens)} tokens, {len(data_bytes):,} bytes concatenated.
 * Sorted alphabetically for binary search.
 * Greedy longest-match tokenizer uses sp_tokenize(). */
#ifndef BQSM_VOCAB_H
#define BQSM_VOCAB_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define SP_VOCAB_SIZE {len(tokens)}
#define SP_VOCAB_DATA_LEN {len(data_bytes)}
#define SP_MAX_TOKEN_LEN {max(len(t) for t,_ in tokens)}

""")
    
    # Token data
    f.write("static const unsigned char sp_vocab_data[] = {\n")
    for i in range(0, len(data_bytes), 16):
        chunk = data_bytes[i:i+16]
        f.write("  " + ",".join(str(b) for b in chunk) + ",\n")
    f.write("};\n\n")
    
    # Index: {uint32_t offset, uint16_t len, uint32_t id}
    f.write(f"static const struct {{ uint32_t off; uint16_t len; uint32_t id; }} sp_vocab_index[{len(tokens)}] = {{\n")
    for off, ln, tid in index:
        f.write(f"  {{{off}U, {ln}U, {tid}U}},\n")
    f.write("};\n\n")
    
    # Binary search function
    f.write("""/* Binary search: find token matching (str, slen). Returns token ID or -1. */
static int32_t sp_find_token(const unsigned char *str, int slen) {
    int lo = 0, hi = SP_VOCAB_SIZE - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int mlen = sp_vocab_index[mid].len;
        int off  = sp_vocab_index[mid].off;
        int cmp = slen - mlen;
        if (cmp == 0) {
            cmp = memcmp(str, sp_vocab_data + off, slen);
        }
        if (cmp < 0) hi = mid - 1;
        else if (cmp > 0) lo = mid + 1;
        else return (int32_t)sp_vocab_index[mid].id;
    }
    return -1;
}

/* Greedy longest-match tokenizer. Returns number of tokens (max max_tokens).
 * Caller frees *out. Returns -1 on allocation failure. */
static int sp_tokenize(const unsigned char *text, int text_len, int max_tokens, int32_t **out) {
    *out = (int32_t*)malloc((size_t)max_tokens * sizeof(int32_t));
    if (!*out) return -1;
    int n_tokens = 0;
    int pos = 0;
    
    while (pos < text_len && n_tokens < max_tokens) {
        int best_len = 0;
        int32_t best_id = 1;  /* <eos> = 1, fallback for unknown */
        int search_len = text_len - pos;
        if (search_len > SP_MAX_TOKEN_LEN) search_len = SP_MAX_TOKEN_LEN;
        
        for (int try_len = search_len; try_len >= 1; try_len--) {
            int32_t tid = sp_find_token(text + pos, try_len);
            if (tid >= 0) {
                best_len = try_len;
                best_id = tid;
                break;
            }
        }
        
        (*out)[n_tokens++] = best_id;
        pos += (best_len > 0 ? best_len : 1);
    }
    return n_tokens;
}

#endif /* BQSM_VOCAB_H */
""")

print(f"Wrote {OUT_H}")
print(f"Sample sorted tokens (first 10): {[(tokens[i][0][:20].decode('utf-8','replace'), tokens[i][1]) for i in range(10)]}")
print(f"Sample sorted tokens (around 'hello'):")
for i, (tb, tid) in enumerate(tokens):
    s = tb.decode('utf-8', errors='replace')
    if 'hello' in s.lower() or 'Hello' in s:
        print(f"  [{i}] id={tid}: {s}")
