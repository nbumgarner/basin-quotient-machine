/* ===========================================================================
 * bqsm_model.h — BQSM Model Loader (public interface)
 * ===========================================================================
 *   v3: Packed 2-bit format, architecture header.
 *   Loads .bqsm model files, unpacks on load.
 */
#ifndef BQSM_MODEL_H
#define BQSM_MODEL_H
#include <stdint.h>
#include <stddef.h>

#define BQSM_MODEL_MAGIC   0x4D535142  /* 'BQSM' */
#define BQSM_MAX_TENSORS   512
#define BQSM_MAX_NAME      256
#define BQSM_MAX_DIMS      8

typedef struct {
    char     name[BQSM_MAX_NAME];
    uint64_t shape[BQSM_MAX_DIMS];
    uint8_t  ndims;
    size_t   nelems;     /* unpacked element count */
    int8_t  *data;       /* packed ternary data, owned by arena */
    float   *float_data; /* raw float32 data (norm weights), owned by arena */
    uint8_t  tensor_type; /* 0=packed ternary, 1=raw float32 */
} bqsm_tensor_t;

typedef struct {
    int            n_tensors;
    bqsm_tensor_t  tensors[BQSM_MAX_TENSORS];
    void          *arena;        /* single contiguous allocation */
    size_t         arena_size;
    /* Model metadata (from v2+ architecture header) */
    int d_model;
    int ffn_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
} bqsm_model_t;

int  bqsm_model_load(bqsm_model_t *m, const char *path);
bqsm_tensor_t *bqsm_model_find(bqsm_model_t *m, const char *name);
void bqsm_model_free(bqsm_model_t *m);
void bqsm_model_print(bqsm_model_t *m);

#endif
