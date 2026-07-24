/* ===========================================================================
 * bqsm_model.c — BQSM Model Loader
 * =========================================================================== */
#include "bqsm_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int bqsm_model_load(bqsm_model_t *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    memset(m, 0, sizeof(*m));

    /* Header */
    char magic[4];
    uint32_t ver, nt;
    if (fread(magic,1,4,f)!=4 || memcmp(magic,"BQSM",4) ||
        fread(&ver,4,1,f)!=1 || fread(&nt,4,1,f)!=1) {
        fprintf(stderr,"bqsm_model: bad header\n"); fclose(f); return -1;
    }
    m->n_tensors = (int)nt;

    /* First pass: compute total data size */
    size_t total_data = 0;
    long data_offsets[BQSM_MAX_TENSORS];
    for (int ti = 0; ti < m->n_tensors; ti++) {
        uint16_t nl; fread(&nl,2,1,f);
        fseek(f, nl, SEEK_CUR); /* skip name */
        uint8_t nd; fread(&nd,1,1,f);
        for (int d=0;d<nd;d++) { uint64_t dim; fread(&dim,8,1,f); }
        uint32_t dl; fread(&dl,4,1,f);
        data_offsets[ti] = total_data;
        total_data += dl;
        fseek(f, dl, SEEK_CUR);
    }

    /* Allocate arena */
    m->arena = malloc(total_data);
    if (!m->arena) { fprintf(stderr,"bqsm_model: alloc %zu failed\n",total_data); fclose(f); return -1; }
    m->arena_size = total_data;

    /* Second pass: read tensor metadata + data */
    fseek(f, 4+4+4, SEEK_SET); /* rewind past header */
    for (int ti = 0; ti < m->n_tensors; ti++) {
        bqsm_tensor_t *t = &m->tensors[ti];
        uint16_t nl; fread(&nl,2,1,f);
        fread(t->name, 1, nl < BQSM_MAX_NAME-1 ? nl : BQSM_MAX_NAME-1, f);
        t->name[nl < BQSM_MAX_NAME-1 ? nl : BQSM_MAX_NAME-1] = 0;
        uint8_t nd; fread(&nd,1,1,f);
        t->ndims = nd;
        t->nelems = 1;
        for (int d=0;d<nd;d++) {
            fread(&t->shape[d],8,1,f);
            if (d>0) t->nelems *= (size_t)t->shape[d];
        }
        uint32_t dl; fread(&dl,4,1,f);
        t->data = (int8_t*)((char*)m->arena + data_offsets[ti]);
        fread(t->data, 1, dl, f);
    }
    fclose(f);

    /* Infer model metadata from tensor names */
    for (int ti = 0; ti < m->n_tensors; ti++) {
        const char *n = m->tensors[ti].name;
        if (n && m->tensors[ti].ndims >= 2) {
            if (strstr(n,"ffn_gate")||strstr(n,"ffn_up")) {
                if (!m->d_model) m->d_model = (int)m->tensors[ti].shape[0];
                if (!m->ffn_dim) m->ffn_dim = (int)m->tensors[ti].shape[1];
            }
            if (strstr(n,"ffn_down") && !m->d_model)
                m->d_model = (int)m->tensors[ti].shape[1];
        }
        /* Count layers from blk.N pattern */
        if (strstr(n,"blk.")) {
            const char *p = strstr(n,"blk.") + 4;
            int layer = atoi(p);
            if (layer >= m->n_layers) m->n_layers = layer + 1;
        }
    }

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
    printf("BQSM Model: %d tensors\n", m->n_tensors);
    printf("  d_model=%d ffn_dim=%d n_layers=%d\n",
           m->d_model, m->ffn_dim, m->n_layers);
    for (int i = 0; i < m->n_tensors && i < 5; i++) {
        bqsm_tensor_t *t = &m->tensors[i];
        printf("  %s: [", t->name);
        for (int d = 0; d < t->ndims; d++)
            printf("%s%lu", d?",":"", (unsigned long)t->shape[d]);
        printf("] (%zu elems)\n", t->nelems);
    }
    if (m->n_tensors > 5) printf("  ... (%d more)\n", m->n_tensors - 5);
}