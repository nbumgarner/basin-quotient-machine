/* ===========================================================================
 * bqsm_model.h — BQSM Model Loader (public interface)
 * ===========================================================================
 *   Loads .bqsm model files (7-state quantized weights).
 *   Provides tensor access by name. Single-header, no allocations after load.
 */

#ifndef BQSM_MODEL_H
#define BQSM_MODEL_H
#include <stdint.h>
#include <stddef.h>

#define BQSM_MODEL_MAGIC   0x4D535142  /* 'BQSM' */
#define BQSM_MODEL_VERSION 1
#define BQSM_MAX_TENSORS   512
#define BQSM_MAX_NAME      256
#define BQSM_MAX_DIMS      8

typedef struct {
    char     name[BQSM_MAX_NAME];
    uint64_t shape[BQSM_MAX_DIMS];
    uint8_t  ndims;
    size_t   nelems;
    int8_t  *data;        /* owned by model arena */
} bqsm_tensor_t;

typedef struct {
    int            n_tensors;
    bqsm_tensor_t  tensors[BQSM_MAX_TENSORS];
    void          *arena;       /* single contiguous allocation */
    size_t         arena_size;
    /* Model metadata */
    int d_model;
    int ffn_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int max_seq_len;
} bqsm_model_t;

/* Load a .bqsm model file. Returns 0 on success. Caller must call
 * bqsm_model_free() to release resources. */
int  bqsm_model_load(bqsm_model_t *m, const char *path);

/* Find a tensor by name. Returns NULL if not found. */
bqsm_tensor_t *bqsm_model_find(bqsm_model_t *m, const char *name);

/* Free all model resources. */
void bqsm_model_free(bqsm_model_t *m);

/* Print model summary to stdout. */
void bqsm_model_print(bqsm_model_t *m);

#endif /* BQSM_MODEL_H */