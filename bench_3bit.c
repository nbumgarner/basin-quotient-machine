/* ===========================================================================
 * bench_3bit.c — Benchmark 3-bit BQSM matmul kernel on Harpertown
 *
 * Compile: cc -O3 -march=core2 -msse4.1 -std=c11 bench_3bit.c -o bench_3bit -lm
 * Run:     ./bench_3bit <M> <K> <N> <trials>
 * ======================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define BQSM_KERNEL_IMPL
#include "bqsm_kernel_3bit.h"

static double now_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static void fill_random(int8_t *x, int n) {
    for (int i = 0; i < n; i++) x[i] = (int8_t)(rand() % 256 - 128);
}

static void fill_weights(uint8_t *w, int n) {
    for (int i = 0; i < n; i++) w[i] = (uint8_t)(rand() % 256);
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 512;
    int K = (argc > 2) ? atoi(argv[2]) : 512;
    int N = (argc > 3) ? atoi(argv[3]) : 512;
    int trials = (argc > 4) ? atoi(argv[4]) : 5;

    printf("═══ BQSM 3-bit Matmul Benchmark ═══\n");
    printf("M=%d  K=%d  N=%d  trials=%d\n", M, K, N, trials);
    printf("MACs per trial: %lld\n\n", (long long)M * N);

    /* Allocate aligned memory */
    int8_t *x = aligned_alloc(16, K * sizeof(int8_t));
    uint8_t *W_packed = aligned_alloc(16, M * (N/2) * sizeof(uint8_t));
    float *C = aligned_alloc(16, N * sizeof(float));

    fill_random(x, K);
    fill_weights(W_packed, M * (N/2));

    /* Warmup */
    bqsm_matmul_vec_3bit(x, W_packed, M, N, C);

    /* Benchmark */
    double best_ms = 1e9;
    double total_gmacs = 0;

    for (int t = 0; t < trials; t++) {
        memset(C, 0, N * sizeof(float));
        double t0 = now_sec();
        bqsm_matmul_vec_3bit(x, W_packed, M, N, C);
        double t1 = now_sec();
        double ms = (t1 - t0) * 1000;
        double gmacs = (M * (double)K * N) / (ms / 1000.0) / 1e9;
        printf("  Trial %d: %.2f ms  ->  %.2f GMACs/s\n", t+1, ms, gmacs);
        if (ms < best_ms) best_ms = ms;
        total_gmacs += gmacs;
    }

    double avg_gmacs = total_gmacs / trials;
    printf("\n  Best:   %.2f ms  ->  %.2f GMACs/s\n", best_ms, (M * (double)K * N) / (best_ms / 1000.0) / 1e9);
    printf("  Average:                    %.2f GMACs/s\n", avg_gmacs);

    /* Target: 96 GMACs/s */
    printf("\n  Target: 96 GMACs/s\n");
    printf("  Achieved: %.1f%% of target\n", avg_gmacs / 96.0 * 100);

    /* Projected tokens/sec for Phi-3 medium (32 layers, 2 FFN matmuls/layer) */
    double gmacs_per_token = 32 * 2 * (5120.0 * 14336.0) / 1e9;  /* ~4.7 GMACs/token */
    double tok_s = avg_gmacs / gmacs_per_token;
    printf("  Projected: %.2f tok/s (Phi-3 medium)\n", tok_s);

    free(x);
    free(W_packed);
    free(C);
    return 0;
}