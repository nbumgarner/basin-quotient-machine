/* bqsm_load.c — Load .bqsm file, run BQSM matmul on real tensor shapes */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define QM 6  /* 7 states */
static int8_t TT[QM+1][QM+1];

static void init_tt(void) {
    for (int q = 0; q <= QM; q++)
        for (int w = 0; w <= QM; w++) {
            int p = (q * w * 2 + QM) / (QM * 2 / 3);
            TT[q][w] = (int8_t)(p < 0 ? 0 : (p > QM ? QM : p));
        }
}

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Load .bqsm file, run BQSM matmul on each 2D tensor, report tok/s */
int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/home/compunerd/Downloads/gemma-2b.bqsm";
    int L = argc > 2 ? atoi(argv[2]) : 64;  /* batch size */
    int cores = 1;
#ifdef _OPENMP
    cores = omp_get_max_threads();
#endif

    init_tt();
    printf("═══ BQSM Loader — %s ═══\n%d cores, L=%d batch\n\n", path, cores, L);

    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }

    char magic[5] = {0};
    uint32_t ver, n;
    fread(magic, 1, 4, f); fread(&ver, 4, 1, f); fread(&n, 4, 1, f);
    printf("BQSM v%d, %u tensors\n\n", ver, n);

    for (uint32_t ti = 0; ti < n; ti++) {
        uint16_t name_len; fread(&name_len, 2, 1, f);
        char name[256] = {0}; fread(name, 1, name_len < 255 ? name_len : 255, f);
        uint8_t ndims; fread(&ndims, 1, 1, f);
        uint64_t dims[8];
        size_t elems = 1;
        for (int d = 0; d < ndims; d++) {
            fread(&dims[d], 8, 1, f);
            if (d > 0) elems *= (size_t)dims[d];
        }
        /* For 2D tensors: first dim is K (input), second is N (output) */
        if (ndims < 2) { fseek(f, elems, SEEK_CUR); continue; }
        int K = (int)dims[0], N = (int)dims[1];

        uint32_t data_len; fread(&data_len, 4, 1, f);
        int8_t *B = malloc(data_len);
        if (!B) { printf("alloc fail\n"); fclose(f); return 1; }
        fread(B, 1, data_len, f);

        /* Generate random input batch (7-state) */
        srand(42);
        int8_t *A = malloc((size_t)L * K);
        int32_t *C = malloc((size_t)L * N * sizeof(int32_t));
        if (!A || !C) { printf("alloc fail\n"); free(B); fclose(f); return 1; }
        for (size_t i = 0; i < (size_t)L * K; i++)
            A[i] = (int8_t)(rand() % (QM+1));

        printf("%-45s %d×%d  ", name, K, N);

        /* BQSM matmul: A (L×K) × B (K×N) → C (L×N) */
        double t0 = now_sec();
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < N; j++) {
                int32_t acc = 0;
                for (int k = 0; k < K; k++)
                    acc += TT[(int)A[i*K + k]][(int)B[k*N + j]];
                C[i*N + j] = acc;
            }
        }
        double ms = (now_sec() - t0) * 1000;
        double macs = (double)L * K * N / (ms * 1e-3) / 1e9;
        printf("%8.0fms  %6.0fM MACs/s\n", ms, macs);

        free(B); free(A); free(C);
        if (ti >= 2) break;  /* just first 3 tensors */
    }

    fclose(f);
    return 0;
}