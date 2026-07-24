/* ===========================================================================
 * bqsm_dequant.c — GGUF Q4_K/Q6_K nibble extractor → BQSM v3
 *
 *   Reads GGUF format directly. Extracts quantized nibbles from Q4_K/Q6_K
 *   tensors. Maps nibble value (0-15 for Q4_K, 0-63 for Q6_K) to BQSM
 *   4-state (0-3). No float math. No llama.cpp dependency.
 *
 *   GGUF layout:
 *     magic "GGUF" (4B) + version (u32) + n_tensors (u64) + n_kv (u64)
 *     + kv_pairs ... + tensor_infos ... + ALIGN + tensor_data ...
 *
 *   Each tensor_info: name(str) + ndim(u32) + dims(u64[])* + type(u32) + offset(u64)
 *
 *   Q4_K super-block (256 elements): 140 bytes
 *     d[0..1] = fp16 min, fp16 max (4B)
 *     scales[0..11] = 6-bit × 12 packed into 9 bytes
 *     q[0..127] = 4-bit values (128B)
 *
 *   Q6_K super-block (256 elements): 210 bytes
 *     d = fp16 (2B)
 *     ql[0..127] = 4-bit low (64B)
 *     qh[0..63]  = 2-bit high (16B)
 *     scales[0..15] = 8-bit (16B)
 *
 * BUILD: cc -O2 -std=c11 -o bqsm_dequant bqsm_dequant.c
 * ======================================================================== */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ─── GGUF types ──────────────────────────────────────────────────────── */
#define GGML_TYPE_F32  0
#define GGML_TYPE_Q4_K 12
#define GGML_TYPE_Q6_K 14

/* ─── Write helpers ───────────────────────────────────────────────────── */
static void w32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void w64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }
static uint32_t r32(FILE *f) { uint32_t v; fread(&v,4,1,f); return v; }
static uint64_t r64(FILE *f) { uint64_t v; fread(&v,8,1,f); return v; }
static void rbuf(FILE *f, void *buf, size_t n) { fread(buf,1,n,f); }

/* ─── Read GGUF string ───────────────────────────────────────────────── */
static void read_str(FILE *f, char *out, size_t max) {
    uint64_t len = r64(f);
    size_t n = len < max-1 ? len : max-1;
    rbuf(f, out, n); out[n] = 0;
    if (len > n) fseek(f, (long)(len - n), SEEK_CUR);
}

/* ─── Q4_K nibble extraction ─────────────────────────────────────────── */
/* Reads one super-block (256 elements) worth of nibbles into out[].
 * Returns number of nibbles written (256). */
static int extract_q4k_block(FILE *f, uint8_t *out) {
    uint8_t block[140];
    if (fread(block, 1, 140, f) != 140) return 0;

    /* block[4..131] = 128 bytes of 4-bit quantized values */
    /* Each byte has 2 nibbles: lo first, then hi */
    const uint8_t *q = block + 4;
    for (int i = 0; i < 128; i++) {
        out[i*2]     = q[i] & 0x0F;
        out[i*2 + 1] = q[i] >> 4;
    }
    return 256;
}

/* ─── Q6_K nibble extraction ─────────────────────────────────────────── */
/* Q6_K: 256 elements in 210 bytes.
 * Each value = ql[i] | (qh[i/4] >> (2*(i%4))) & 3) << 4
 * Range: 0-63 (6-bit) */
static int extract_q6k_block(FILE *f, uint8_t *out) {
    uint8_t block[210];
    if (fread(block, 1, 210, f) != 210) return 0;

    /* block[2..65] = ql (64 bytes, 4-bit low for pairs)
     * block[66..81] = qh (16 bytes, 2-bit high per element)
     * We want the full 6-bit values for max fidelity */
    const uint8_t *ql = block + 2;
    const uint8_t *qh = block + 66;

    for (int i = 0; i < 256; i++) {
        int lo = (i & 1) ? (ql[i/2] >> 4) : (ql[i/2] & 0x0F);
        int hi = (qh[i/4] >> (2 * (i % 4))) & 3;
        out[i] = (uint8_t)(lo | (hi << 4));  /* 0..63 */
    }
    return 256;
}

/* ─── Map nibble to BQSM 4-state ──────────────────────────────────────── */
/* Q4_K nibbles: 0-15 → mod 4 → 0-3      (uniform mapping)
 * Q6_K values: 0-63 → /16 → 0-3          (bucket mapping)
 * Both preserve signal structure from original quantization */
static uint8_t nibble_to_bqsm_q4(uint8_t n) { return n & 3; }
static uint8_t nibble_to_bqsm_q6(uint8_t n) { return n >> 4; }  /* 0..63 → 0..3 */

/* ─── Pack 4-state values ────────────────────────────────────────────── */
static size_t pack_4state(const uint8_t *vals, size_t n, uint8_t *out) {
    size_t pad = (4 - (n % 4)) % 4;
    size_t total = n + pad;
    for (size_t i = 0; i < total; i += 4) {
        uint8_t a = (i < n) ? vals[i] : 0;
        uint8_t b = (i+1 < n) ? vals[i+1] : 0;
        uint8_t c = (i+2 < n) ? vals[i+2] : 0;
        uint8_t d = (i+3 < n) ? vals[i+3] : 0;
        out[i/4] = (a & 3) | ((b & 3) << 2) | ((c & 3) << 4) | ((d & 3) << 6);
    }
    return total / 4;
}

/* ─── Main ────────────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    setbuf(stdout, NULL);  /* unbuffered for crash debugging */
    const char *inp  = "/home/compunerd/Downloads/gemma-2-2b-it-abliterated-Q4_K_M.gguf";
    const char *outf = "/home/compunerd/Downloads/gemma-2b-real.bqsm";
    if (argc > 1) inp = argv[1];
    if (argc > 2) outf = argv[2];

    FILE *fin = fopen(inp, "rb");
    if (!fin) { perror(inp); return 1; }

    /* ── GGUF header ── */
    char magic[5] = {0};
    rbuf(fin, magic, 4);
    if (memcmp(magic, "GGUF", 4)) { fprintf(stderr, "Not GGUF\n"); return 1; }
    uint32_t ver = r32(fin);
    uint64_t n_tensors = r64(fin);
    uint64_t n_kv = r64(fin);
    printf("GGUF v%u: %lu tensors, %lu kv pairs\n", ver,
           (unsigned long)n_tensors, (unsigned long)n_kv);

    /* ── Skip KV pairs ── */
    printf("Skipping %lu kv pairs...\n", (unsigned long)n_kv);
    for (uint64_t i = 0; i < n_kv; i++) {
        char key[256];
        read_str(fin, key, sizeof(key));
        uint32_t vtype = r32(fin);
        if (i < 3) printf("  kv[%lu]: key='%s' type=%u pos=%ld\n",
                          (unsigned long)i, key, vtype, ftell(fin));
        /* Skip value based on type */
        switch (vtype) {
            case 0: case 1: fseek(fin, 1, SEEK_CUR); break;          /* u8, i8 */
            case 2: case 3: fseek(fin, 2, SEEK_CUR); break;          /* u16, i16 */
            case 4: case 5: case 8: case 9: fseek(fin, 4, SEEK_CUR); break; /* u32,i32,f32,bool */
            case 6: case 7: case 10: fseek(fin, 8, SEEK_CUR); break; /* u64,i64,f64 */
            case 12: { /* string */
                uint64_t sl = r64(fin);
                fseek(fin, (long)sl, SEEK_CUR);
                break;
            }
            case 13: { /* array */
                uint32_t atype = r32(fin);
                uint64_t alen = r64(fin);
                /* Determine element size */
                size_t esz;
                switch (atype) {
                    case 0: case 1: esz = 1; break;
                    case 2: case 3: esz = 2; break;
                    case 4: case 5: case 8: case 9: esz = 4; break;
                    case 6: case 7: case 10: esz = 8; break;
                    case 12: esz = 8; break;  /* string in array = offset pointer */
                    default: esz = 4; break;
                }
                fseek(fin, (long)(alen * esz), SEEK_CUR);
                break;
            }
            default: break;
        }
    }
    printf("KV pairs skipped. Position: %ld\n", ftell(fin));

    /* ── Read tensor infos ── */
    typedef struct {
        char name[256];
        uint32_t ndim;
        uint64_t dims[8];
        uint32_t type;
        uint64_t offset;
        uint64_t nelems;
    } tensor_t;

    tensor_t *tensors = malloc((size_t)n_tensors * sizeof(tensor_t));

    /* Read all tensor info */
    for (uint64_t i = 0; i < n_tensors; i++) {
        tensor_t *t = &tensors[i];
        read_str(fin, t->name, sizeof(t->name));
        t->ndim = r32(fin);
        t->nelems = 1;
        for (uint32_t d = 0; d < t->ndim; d++) {
            t->dims[d] = r64(fin);
            t->nelems *= t->dims[d];
        }
        t->type = r32(fin);
        t->offset = r64(fin);
    }

    /* Show summary */
    int q4k_count=0, q6k_count=0, f32_count=0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        switch(tensors[i].type) {
            case GGML_TYPE_Q4_K: q4k_count++; break;
            case GGML_TYPE_Q6_K: q6k_count++; break;
            case GGML_TYPE_F32:  f32_count++; break;
        }
    }
    printf("Tensors: %d Q4_K, %d Q6_K, %d F32\n", q4k_count, q6k_count, f32_count);

    /* ── BQSM output ── */
    FILE *fout = fopen(outf, "wb");
    if (!fout) { perror(outf); return 1; }

    /* Architecture: hardcoded for Gemma 2B */
    int d_model=2304, ffn_dim=9216, n_layers=26, n_heads=8, n_kv_heads=4, vocab=256000;

    /* Write BQSM v3 header */
    fwrite("BQSM", 1, 4, fout);
    w32(fout, 3);  /* version */
    w32(fout, (uint32_t)d_model);
    w32(fout, (uint32_t)ffn_dim);
    w32(fout, (uint32_t)n_layers);
    w32(fout, (uint32_t)n_heads);
    w32(fout, (uint32_t)n_kv_heads);
    w32(fout, (uint32_t)vocab);
    w32(fout, (uint32_t)n_tensors);  /* tensor count */

    /* We'll write each tensor as we process it.
     * But first we need to write the header with correct packed sizes.
     * Strategy: two passes — first compute packed sizes, then write. */

    /* For now, process tensors in order and write directly.
     * We need to know packed sizes upfront for the BQSM header...
     * Actually, BQSM v3 format: each tensor header has orig_n + packed_n,
     * so we can write tensor headers first, then data.
     * But the data must follow immediately after header.
     *
     * Simplification: write header + all tensor metas, then all tensor data.
     * The BQSM format expects: [header] [tensor1 meta+data] [tensor2 meta+data]...
     *
     * To do this in one pass, we'd need to seek back to fill sizes.
     * Let's use a two-pass approach: compute sizes first, then write. */

    /* First pass: compute packed sizes */
    size_t *packed_sizes = malloc((size_t)n_tensors * sizeof(size_t));
    size_t total_packed = 0;

    for (uint64_t i = 0; i < n_tensors; i++) {
        tensor_t *t = &tensors[i];
        if (t->type == GGML_TYPE_F32) {
            /* F32 tensors (norms): we need to read and quantize to 4-state */
            /* Norm tensors are small (<2304 elements). Read to buffer. */
            size_t nf = (size_t)t->nelems;
            float *fbuf = malloc(nf * sizeof(float));
            fseek(fin, (long)t->offset, SEEK_SET);
            fread(fbuf, sizeof(float), nf, fin);

            /* Quantize to 4-state */
            uint8_t *qbuf = malloc(nf);
            for (size_t j = 0; j < nf; j++) {
                /* Map float to 0-3: center at 2 */
                float v = fbuf[j];
                int iv = (int)(v * 1.5f + 2.0f);
                if (iv < 0) iv = 0;
                if (iv > 3) iv = 3;
                qbuf[j] = (uint8_t)iv;
            }
            /* Pack */
            uint8_t *pkbuf = malloc((nf + 3) / 4);
            packed_sizes[i] = pack_4state(qbuf, nf, pkbuf);
            total_packed += packed_sizes[i];
            free(fbuf); free(qbuf); free(pkbuf);
        } else {
            /* Q4_K or Q6_K: nibbles → 4-state → pack */
            size_t n_blocks = (size_t)t->nelems / 256;
            size_t blk_size = (t->type == GGML_TYPE_Q4_K) ? 140 : 210;
            uint8_t *nibbles = malloc((size_t)t->nelems);
            uint8_t *qbuf = malloc((size_t)t->nelems);

            fseek(fin, (long)t->offset, SEEK_SET);
            for (size_t b = 0; b < n_blocks; b++) {
                uint8_t blk_buf[256];
                int n;
                if (t->type == GGML_TYPE_Q4_K)
                    n = extract_q4k_block(fin, blk_buf);
                else
                    n = extract_q6k_block(fin, blk_buf);
                for (int j = 0; j < n && b*256+j < (size_t)t->nelems; j++) {
                    uint8_t v = blk_buf[j];
                    uint8_t bq = (t->type == GGML_TYPE_Q4_K) ?
                        nibble_to_bqsm_q4(v) : nibble_to_bqsm_q6(v);
                    qbuf[b*256 + j] = bq;
                }
            }

            uint8_t *pkbuf = malloc(((size_t)t->nelems + 3) / 4);
            packed_sizes[i] = pack_4state(qbuf, (size_t)t->nelems, pkbuf);
            total_packed += packed_sizes[i];
            free(nibbles); free(qbuf); free(pkbuf);
        }
        if (i % 50 == 0) printf("  Pass 1: %lu/%lu tensors...\r", (unsigned long)i, (unsigned long)n_tensors);
    }
    printf("  Pass 1: %lu tensors done. Packed: %.1f MB\n",
           (unsigned long)n_tensors, total_packed / (1024.0*1024.0));

    /* Second pass: write everything */
    for (uint64_t i = 0; i < n_tensors; i++) {
        tensor_t *t = &tensors[i];

        /* Write tensor header */
        uint16_t nl = (uint16_t)strlen(t->name);
        fwrite(&nl, 2, 1, fout);
        fwrite(t->name, 1, nl, fout);

        uint8_t nd = (uint8_t)t->ndim;
        fwrite(&nd, 1, 1, fout);
        for (uint32_t d = 0; d < t->ndim; d++)
            w64(fout, t->dims[d]);

        w32(fout, (uint32_t)t->nelems);       /* original element count */
        w32(fout, (uint32_t)packed_sizes[i]); /* packed byte count */

        /* Read, convert, pack, write */
        if (t->type == GGML_TYPE_F32) {
            size_t nf = (size_t)t->nelems;
            float *fbuf = malloc(nf * sizeof(float));
            fseek(fin, (long)t->offset, SEEK_SET);
            fread(fbuf, sizeof(float), nf, fin);

            uint8_t *qbuf = malloc(nf);
            for (size_t j = 0; j < nf; j++) {
                float v = fbuf[j];
                int iv = (int)(v * 1.5f + 2.0f);
                if (iv < 0) iv = 0;
                if (iv > 3) iv = 3;
                qbuf[j] = (uint8_t)iv;
            }

            uint8_t *pkbuf = malloc(packed_sizes[i]);
            pack_4state(qbuf, nf, pkbuf);
            fwrite(pkbuf, 1, packed_sizes[i], fout);
            free(fbuf); free(qbuf); free(pkbuf);
        } else {
            size_t n_blocks = (size_t)t->nelems / 256;
            uint8_t *qbuf = malloc((size_t)t->nelems);

            fseek(fin, (long)t->offset, SEEK_SET);
            for (size_t b = 0; b < n_blocks; b++) {
                uint8_t blk_buf[256];
                int n;
                if (t->type == GGML_TYPE_Q4_K)
                    n = extract_q4k_block(fin, blk_buf);
                else
                    n = extract_q6k_block(fin, blk_buf);
                for (int j = 0; j < n && b*256+j < (size_t)t->nelems; j++) {
                    uint8_t v = blk_buf[j];
                    qbuf[b*256 + j] = (t->type == GGML_TYPE_Q4_K) ?
                        nibble_to_bqsm_q4(v) : nibble_to_bqsm_q6(v);
                }
            }

            uint8_t *pkbuf = malloc(packed_sizes[i]);
            pack_4state(qbuf, (size_t)t->nelems, pkbuf);
            fwrite(pkbuf, 1, packed_sizes[i], fout);
            free(qbuf); free(pkbuf);
        }
        if (i % 50 == 0) printf("  Pass 2: %lu/%lu...\r", (unsigned long)i, (unsigned long)n_tensors);
    }

    printf("  Pass 2: done.\n");

    fclose(fin);
    fclose(fout);
    free(tensors);
    free(packed_sizes);

    printf("\nWrote %s: %.1f MB\n", outf, total_packed / (1024.0*1024.0));
    return 0;
}
