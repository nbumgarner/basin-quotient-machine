/* bqsm_load_test.c — Quick model load test */
#include "bqsm_model.h"
#include <stdio.h>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/home/compunerd/Downloads/gemma-2b-full.bqsm";
    bqsm_model_t m;
    printf("Loading %s...\n", path);
    if (bqsm_model_load(&m, path) != 0) {
        fprintf(stderr, "FAILED\n");
        return 1;
    }
    bqsm_model_print(&m);

    /* Spot-check some tensors */
    const char *checks[] = {
        "token_embd.weight",
        "blk.0.ffn_up.weight",
        "blk.12.attn_q.weight",
        "blk.25.ffn_down.weight",
        "output_norm.weight",
        NULL
    };
    printf("\nSpot checks:\n");
    for (int i = 0; checks[i]; i++) {
        bqsm_tensor_t *t = bqsm_model_find(&m, checks[i]);
        if (t) {
            printf("  %s: ndims=%d nelems=%zu data[0..3]=[%d,%d,%d,%d]\n",
                   t->name, t->ndims, t->nelems,
                   t->data[0], t->data[1], t->data[2], t->data[3]);
        } else {
            printf("  %s: NOT FOUND\n", checks[i]);
        }
    }

    bqsm_model_free(&m);
    printf("\nOK\n");
    return 0;
}
