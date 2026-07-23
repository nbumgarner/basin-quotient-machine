/* bqsm_batch.c — Batched BQSM Matmul with SoA layout
 * SoA: A_batch[(i*K + k)*L + l] — lanes contiguous, unit stride.
 * This is the ring_furnace.c layout. Inner loop auto-vectorizes. */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static const int8_t P[3][3] = {{1,0,-1},{0,0,0},{-1,0,1}};

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* ─── SoA Layout: A[(i*K + k)*L + l]  (lanes innermost) ──────────── */
static void matmul_soa(int L, int M, int N, int K,
                       const int8_t *A, const int8_t *B, int32_t *C)
{
    /* C[(i*N + j)*L + l] — also SoA for output */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t *c_row = &C[(i*N + j)*L];
            for (int l = 0; l < L; l++) c_row[l] = 0;

            for (int k = 0; k < K; k++) {
                int8_t b = B[k*N + j];
                int bi = (int)(b + 1);
                const int8_t *a_col = &A[(i*K + k)*L];  /* L lanes, unit stride */
                for (int l = 0; l < L; l++)
                    c_row[l] += P[(int)(a_col[l] + 1)][bi];
            }
        }
    }
}

/* ─── SoA OpenMP ──────────────────────────────────────────────────── */
static void matmul_soa_omp(int L, int M, int N, int K,
                           const int8_t *A, const int8_t *B, int32_t *C)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc[256] = {0};
            for (int k = 0; k < K; k++) {
                int8_t b = B[k*N + j];
                int bi = (int)(b + 1);
                const int8_t *a_col = &A[(i*K + k)*L];
                for (int l = 0; l < L; l++)
                    acc[l] += P[(int)(a_col[l] + 1)][bi];
            }
            for (int l = 0; l < L; l++)
                C[(i*N + j)*L + l] = acc[l];
        }
    }
}

/* ─── Non-batched reference ───────────────────────────────────────── */
static void matmul_ref(int M, int N, int K,
                       const int8_t *A, const int8_t *B, int32_t *C)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += P[(int)(A[i*K+k]+1)][(int)(B[k*N+j]+1)];
            C[i*N+j] = acc;
        }
}

static void fill_ternary(int8_t *buf, size_t n) {
    for (size_t i = 0; i < n; i++) buf[i] = (int8_t)((rand()%3)-1);
}

int main(int argc, char **argv) {
    int L = argc > 1 ? atoi(argv[1]) : 256;
    int M = argc > 2 ? atoi(argv[2]) : 256;
    int K = argc > 3 ? atoi(argv[3]) : 256;
    int N = argc > 4 ? atoi(argv[4]) : 256;
    int trials = argc > 5 ? atoi(argv[5]) : 10;

    int cores = 1;
#ifdef _OPENMP
    cores = omp_get_max_threads();
#endif

    printf("═══ BQSM SoA Matmul — L=%d lanes, %d×%d×%d, %d cores ═══\n\n",
           L, M, K, N, cores);

    size_t a_sz = (size_t)L * M * K;
    size_t b_sz = (size_t)K * N;
    size_t c_sz = (size_t)L * M * N * sizeof(int32_t);
    int8_t *A = malloc(a_sz), *B = malloc(b_sz);
    int32_t *C = malloc(c_sz);
    if (!A||!B||!C) { printf("alloc fail\n"); return 1; }
    srand(42);
    fill_ternary(A, a_sz); fill_ternary(B, b_sz);

    /* Non-batched: L individual matmuls */
    double t0 = now_sec();
    for (int t = 0; t < trials; t++)
        for (int l = 0; l < L; l++)
            matmul_ref(M, N, K, &A[l*M*K], B, &C[l*M*N]);
    double ref_t = (now_sec()-t0)/trials;
    double ref_tok = L / ref_t;
    double ref_macs = (double)L*M*N*K / ref_t / 1e9;
    printf("Ref (×%d indep):   %6.0fms  %6.0ftok/s  %.2f GMACs/s\n",
           L, ref_t*1e3, ref_tok, ref_macs);

    /* SoA batched */
    memset(C, 0, c_sz);
    t0 = now_sec();
    for (int t = 0; t < trials; t++)
        matmul_soa(L, M, N, K, A, B, C);
    double soa_t = (now_sec()-t0)/trials;
    double soa_tok = L / soa_t;
    double soa_macs = (double)L*M*N*K / soa_t / 1e9;
    printf("SoA (L=%d):        %6.0fms  %6.0ftok/s  %.2f GMACs/s  %.1f×\n",
           L, soa_t*1e3, soa_tok, soa_macs, soa_tok/ref_tok);

#ifdef _OPENMP
    memset(C, 0, c_sz);
    t0 = now_sec();
    for (int t = 0; t < trials; t++)
        matmul_soa_omp(L, M, N, K, A, B, C);
    double omp_t = (now_sec()-t0)/trials;
    double omp_tok = L / omp_t;
    double omp_macs = (double)L*M*N*K / omp_t / 1e9;
    printf("SoA OMP (L=%d):    %6.0fms  %6.0ftok/s  %.2f GMACs/s  %.1f×\n",
           L, omp_t*1e3, omp_tok, omp_macs, omp_tok/ref_tok);
#endif

    /* dolphin projection */
    double best_macs = soa_macs;
#ifdef _OPENMP
    best_macs = omp_macs;
#endif
    double ffn_macs = 2.0 * 5120.0 * 14336.0;
    double attn_macs = 4.0 * 5120.0 * 5120.0;
    double total = ffn_macs + attn_macs;
    double ms_layer = total / (best_macs*1e9) * 1000;
    double tok_s = 1000.0 / (ms_layer * 40);

    printf("\n─── dolphin 14B @ %.1f GMACs/s ───\n", best_macs);
    printf("  %d layers: %.0fms/token → %.2f tok/s\n", 40, ms_layer*40, tok_s);
    printf("  Weight reuse: %d× amortization per weight byte\n", L);
    printf("  If FFN-only (ternary, BQSM): %.0fms/layer\n", ffn_macs/(best_macs*1e9)*1000);

    free(A); free(B); free(C);
    return 0;
}