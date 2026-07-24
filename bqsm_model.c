/* ===========================================================================
 * bqsm_model.c — BQSM Model Loader (v3 packed 2-bit)
 * ===========================================================================
 *   Reads .bqsm v3 files: architecture header + packed 4-state tensors.
 *   Unpacks on load: each byte → 4 int8 values.
 */
#include "bqsm_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Unpack 4-state (2-bit) packed bytes to int8 array.
 * Each packed byte yields 4 int8 values [0..3].
 * Returns number of unpacked values (= nelems). */
static size_t unpack_4state(const uint8_t *packed, size_t packed_len,
                            size_t nelems, int8_t *out) {
    size_t written = 0;
    for (size_t i = 0; i < packed_len && written < nelems; i++) {
        uint8_t b = packed[i];
        out[written++] = (int8_t)(b & 3);
        if (written >= nelems) break;
        out[written++] = (int8_t)((b >> 2) & 3);
        if (written >= nelems) break;
        out[written++] = (int8_t)((b >> 4) & 3);
        if (written >= nelems) break;
        out[written++] = (int8_t)((b >> 6) & 3);
    }
    return written;
}

int bqsm_model_load(bqsm_model_t *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    memset(m, 0, sizeof(*m));

    /* Header: magic + version */
    char magic[4];
    uint32_t ver;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "BQSM", 4)) {
        fprintf(stderr, "bqsm_model: bad magic\n"); fclose(f); return -1;
    }
    if (fread(&ver, 4, 1, f) != 1) {
        fprintf(stderr, "bqsm_model: bad version\n"); fclose(f); return -1;
    }

    int is_v3 = (ver >= 3);

    /* Architecture header (v2+) */
    if (ver >= 2) {
        uint32_t arch[6];
        if (fread(arch, 4, 6, f) != 6) {
            fprintf(stderr, "bqsm_model: bad arch header\n"); fclose(f); return -1;
        }
        m->d_model    = (int)arch[0];
        m->ffn_dim    = (int)arch[1];
        m->n_layers   = (int)arch[2];
        m->n_heads    = (int)arch[3];
        m->n_kv_heads = (int)arch[4];
        m->vocab_size = (int)arch[5];
    }

    uint32_t nt;
    if (fread(&nt, 4, 1, f) != 1) { fclose(f); return -1; }
    m->n_tensors = (int)nt;

    /* First pass: compute unpacked data size from tensor headers */
    size_t total_unpacked = 0;
    long   data_offsets[BQSM_MAX_TENSORS];
    size_t tensor_nelems[BQSM_MAX_TENSORS];
    size_t packed_lens[BQSM_MAX_TENSORS];

    for (int ti = 0; ti < m->n_tensors; ti++) {
        uint16_t nl; fread(&nl, 2, 1, f);
        fseek(f, nl, SEEK_CUR); /* skip name */
        uint8_t nd; fread(&nd, 1, 1, f);
        uint64_t shape[BQSM_MAX_DIMS];
        size_t ne = 1;
        for (int d = 0; d < nd; d++) {
            fread(&shape[d], 8, 1, f);
            ne *= (size_t)shape[d];
        }

        if (is_v3) {
            uint32_t orig_n, packed_n;
            fread(&orig_n, 4, 1, f);
            fread(&packed_n, 4, 1, f);
            tensor_nelems[ti] = (size_t)orig_n;
            packed_lens[ti]   = (size_t)packed_n;
            data_offsets[ti]  = (long)total_unpacked;
            total_unpacked += (size_t)orig_n;
            fseek(f, (long)packed_n, SEEK_CUR);
        } else {
            /* v1: raw int8 data */
            uint32_t dl; fread(&dl, 4, 1, f);
            tensor_nelems[ti] = (size_t)dl;
            packed_lens[ti]   = 0;
            data_offsets[ti]  = (long)total_unpacked;
            total_unpacked += (size_t)dl;
            fseek(f, (long)dl, SEEK_CUR);
        }
    }

    /* Allocate arena */
    m->arena = malloc(total_unpacked);
    if (!m->arena) {
        fprintf(stderr, "bqsm_model: alloc %zu failed\n", total_unpacked);
        fclose(f); return -1;
    }
    m->arena_size = total_unpacked;
    memset(m->arena, 0, total_unpacked);

    /* Second pass: read + unpack */
    long hdr_end = (ver >= 2) ? (4 + 4 + 24 + 4) : (4 + 4 + 4);
    fseek(f, hdr_end, SEEK_SET);

    for (int ti = 0; ti < m->n_tensors; ti++) {
        bqsm_tensor_t *t = &m->tensors[ti];
        uint16_t nl; fread(&nl, 2, 1, f);
        size_t ncopy = nl < BQSM_MAX_NAME-1 ? nl : BQSM_MAX_NAME-1;
        fread(t->name, 1, ncopy, f);
        t->name[ncopy] = 0;
        if (nl > ncopy) fseek(f, (long)(nl - ncopy), SEEK_CUR);

        uint8_t nd; fread(&nd, 1, 1, f);
        t->ndims = nd;
        t->nelems = 1;
        for (int d = 0; d < nd; d++) {
            fread(&t->shape[d], 8, 1, f);
            t->nelems *= (size_t)t->shape[d];
        }

        size_t nelems  = tensor_nelems[ti];
        size_t packed_n = packed_lens[ti];
        t->data = (int8_t*)((char*)m->arena + data_offsets[ti]);

        if (is_v3) {
            uint32_t orig_n, pn;
            fread(&orig_n, 4, 1, f);
            fread(&pn, 4, 1, f);
            /* Read packed data to temp, then unpack */
            uint8_t *tmp = (uint8_t*)malloc(packed_n);
            if (!tmp) { fclose(f); return -1; }
            fread(tmp, 1, packed_n, f);
            unpack_4state(tmp, packed_n, nelems, t->data);
            free(tmp);
        } else {
            uint32_t dl; fread(&dl, 4, 1, f);
            fread(t->data, 1, dl, f);
        }
    }
    fclose(f);
    return 0;
}

bqsm_tensor_t *bqsm_model_find(bqsm_model_t *m, const char *name) {
    for (int i = 0; i < m->n_tensors; i++)
        if (!strcmp(m->tensors[i].name, name))
            return &m->tensors[i];
    return NULL;
}

void bqsm_model_free(bqsm_model_t *m) {
    free(m->arena);
    memset(m, 0, sizeof(*m));
}

void bqsm_model_print(bqsm_model_t *m) {
    printf("BQSM Model v3: %d tensors\n", m->n_tensors);
    printf("  Arch: d=%d FFN=%d layers=%d heads=%d kv=%d vocab=%d\n",
           m->d_model, m->ffn_dim, m->n_layers, m->n_heads, m->n_kv_heads, m->vocab_size);
    printf("  Arena: %.1f MB\n", m->arena_size / (1024.0*1024.0));
    int show = m->n_tensors < 8 ? m->n_tensors : 5;
    for (int i = 0; i < show; i++) {
        bqsm_tensor_t *t = &m->tensors[i];
        printf("  %s: [", t->name);
        for (int d = 0; d < t->ndims; d++)
            printf("%s%lu", d?",":"", (unsigned long)t->shape[d]);
        printf("] (%zu elems)\n", t->nelems);
    }
    if (m->n_tensors > show) printf("  ... (%d more)\n", m->n_tensors - show);
}
