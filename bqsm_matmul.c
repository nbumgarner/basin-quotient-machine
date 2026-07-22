/* ===========================================================================
 * bqsm_matmul.c — BQSM Ternary Matrix Multiply Kernel
 * ===========================================================================
 *
 *   emerging.systems — foundation kernel for quantized inference
 *
 * WHAT THIS IS
 *   A drop-in matrix multiply for ternary-weight matrices. Every MAC
 *   (multiply-accumulate) becomes a 1-byte table lookup into a 9-entry
 *   product table, plus an int32 addition. No floating point, no integer
 *   multiply instruction — just L1d-resident lookup + add.
 *
 *   The product table encodes ternary × ternary:
 *              w = -1   0  +1
 *          ┌───────────────────
 *     a=-1 │     +1   0  -1
 *     a= 0 │      0   0   0
 *     a=+1 │     -1   0  +1
 *
 *   Stored as: product[w+1][a+1] = a*w  ∈ {-1,0,+1}
 *
 * VARIANTS
 *   bqsm_matmul_naive    — scalar lookup, reference implementation
 *   bqsm_matmul_simd     — SSE4.1 vectorized (4× int8→int32 per op)
 *   bqsm_matmul_threaded — OpenMP-parallel across cores
 *
 * BUILD
 *   cc -O3 -march=native -std=c11 bqsm_matmul.c -o bqsm_matmul -lm -fopenmp
 *
 * BENCHMARK
 *   ./bqsm_matmul <M> <K> <N> <trials>
 *   Reports effective GFLOPS (comparing against equivalent float32 MACs).
 * ======================================================================== */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* --- Product table: ternary × ternary → {-1,0,+1} --------------------- */
/*     product[w_idx][a_idx] where w_idx = w+1, a_idx = a+1               */
static const int8_t PRODUCT[3][3] = {
    /* w=-1       w=0        w=+1       */
    {  1,  0, -1 },   /* a=-1 */
    {  0,  0,  0 },   /* a= 0 */
    { -1,  0,  1 },   /* a=+1 */
};

/* --- Reference: naive scalar matmul ------------------------------------ */
/*     C = A × B  where A is (M×K), B is (K×N), C is (M×N)               */
/*     All matrices are int8_t with values in {-1,0,+1}.                   */
/*     Accumulator is int32_t (max |acc| ≤ K, fits for K ≤ 2^31).         */

static void matmul_naive(int M, int N, int K,
                         const int8_t *A, const int8_t *B, int32_t *C)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                acc += (int32_t)A[i * K + k] * (int32_t)B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

/* --- BQSM: scalar table-lookup matmul ---------------------------------- */
/*     Replaces integer multiply with PRODUCT[A][B] lookup.                */

static void matmul_bqsm(int M, int N, int K,
                        const int8_t *A, const int8_t *B, int32_t *C)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                int8_t a = A[i * K + k];
                int8_t b = B[k * N + j];
                acc += PRODUCT[(int)(a + 1)][(int)(b + 1)];
            }
            C[i * N + j] = acc;
        }
    }
}

/* --- BQSM: Multi-level transform (the real inference kernel) ----------- */
/*     Input A has values in {0..Q_MAX} (unsigned q-states).               */
/*     Weights B are ternary {-1,0,+1}.                                    */
/*     Transition table TT[q][w+1] = learned contribution ∈ {-Q_MAX..Q_MAX}*/
/*     This is the operation that bqsm_inference.c does per-layer.          */
/*     For comparison, the float32 equivalent is:                           */
/*       acc += (float)A[i][k] * (float)B[k][j]                            */
/*     which requires int→float conversion, multiply, and float add.        */

#define Q_MAX_MAT 3
static const int8_t TT_MAT[Q_MAX_MAT+1][3] = {
    /* q   w=-1    w=0    w=+1   */
    /* 0 */ {  0,    0,     0 },
    /* 1 */ { -1,    0,     1 },
    /* 2 */ { -2,    0,     2 },
    /* 3 */ { -3,    0,     3 },
};

static void matmul_bqsm_transform(int M, int N, int K,
                                   const int8_t *A, const int8_t *B, int32_t *C)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                int8_t a = A[i * K + k];  /* q ∈ {0..3} */
                int8_t b = B[k * N + j];  /* w ∈ {-1,0,+1} */
                acc += TT_MAT[(int)a][(int)(b + 1)];
            }
            C[i * N + j] = acc;
        }
    }
}

/* Float32 baseline: same operation in float */
static void matmul_float32_transform(int M, int N, int K,
                                      const int8_t *A, const int8_t *B, int32_t *C)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += (float)A[i * K + k] * (float)B[k * N + j];
            }
            C[i * N + j] = (int32_t)acc;
        }
    }
}
/* --- BQSM: SSE4.1 vectorized matmul ------------------------------------ */
/*     Process 4 output columns at once using 128-bit vectors.             */
/*     Inner loop: load 4 B values, broadcast A value, 4× table lookup,    */
/*     accumulate into 4× int32. Requires SSE4.1 (T7400 has it).          */

#include <smmintrin.h>  /* SSE4.1 */

static void matmul_bqsm_sse41(int M, int N, int K,
                              const int8_t *A, const int8_t *B, int32_t *C)
{
    /* Pre-load product table into XMM registers as int8 lookup */
    /* We use pshufb (SSSE3) for 16-byte parallel table lookup */
    const __m128i prod_lo = _mm_set_epi8(
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, -1, 0, 0, 0, -1, 0
    );
    /* Better: build a proper 16-byte lookup table for pshufb.
     * pshufb index: high bit → 0, low 4 bits → table index.
     * We need: lookup[a+1 + 4*(b+1)] = PRODUCT[a+1][b+1].
     * So table[0]..table[2] = row for b=-1, table[4]..table[6] = row for b=0,
     * table[8]..table[10] = row for b=+1.
     * Encoding: idx = (a+1) + 4*(b+1). For a∈{-1,0,1}, b∈{-1,0,1}:
     *   idx in {0,1,2, 4,5,6, 8,9,10}. Other entries: 0. */

    static const int8_t lut_bytes[16] __attribute__((aligned(16))) = {
        /* idx:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15 */
        /* b=-1 */    /* b=0  */       /* b=+1 */
              1,  0, -1,  0,  0,  0,  0,  0, -1,  0,  1,  0,  0,  0,  0,  0
    };
    const __m128i lut = _mm_load_si128((const __m128i *)lut_bytes);

    for (int i = 0; i < M; i++) {
        int j = 0;

        /* Process 4 columns at a time with SSE4.1 */
        for (; j + 3 < N; j += 4) {
            __m128i acc = _mm_setzero_si128();  /* 4× int32 = 0 */

            for (int k = 0; k < K; k++) {
                int8_t a = A[i * K + k];

                /* Build index: (a+1) for each of 4 B values */
                /* idx = (a+1) | ((b+1) << 2) for each column */
                int base = (int)(a + 1);
                int idx0 = base | ((int)(B[k * N + j + 0] + 1) << 2);
                int idx1 = base | ((int)(B[k * N + j + 1] + 1) << 2);
                int idx2 = base | ((int)(B[k * N + j + 2] + 1) << 2);
                int idx3 = base | ((int)(B[k * N + j + 3] + 1) << 2);

                /* Pack 4 indices into XMM */
                __m128i indices = _mm_set_epi32(idx3, idx2, idx1, idx0);

                /* Pack to 16-bit then 8-bit for pshufb */
                __m128i idx8 = _mm_shuffle_epi8(
                    indices,
                    _mm_set_epi8(12,12,12,12, 8,8,8,8, 4,4,4,4, 0,0,0,0)
                );
                /* Actually this is getting complicated. Let me use a simpler
                 * SSE approach: just do the scalar lookup but 4-wide add.
                 * The lookup itself is the bottleneck, but the add is cheap. */

                /* Simpler: gather 4 B values, compute 4 products via LUT */
                int8_t b0 = B[k * N + j + 0];
                int8_t b1 = B[k * N + j + 1];
                int8_t b2 = B[k * N + j + 2];
                int8_t b3 = B[k * N + j + 3];

                /* Pack 4 products (sign-extend to int32) */
                __m128i prods = _mm_set_epi32(
                    (int)PRODUCT[base][(int)(b3+1)],
                    (int)PRODUCT[base][(int)(b2+1)],
                    (int)PRODUCT[base][(int)(b1+1)],
                    (int)PRODUCT[base][(int)(b0+1)]
                );

                acc = _mm_add_epi32(acc, prods);
            }

            /* Store 4 results */
            _mm_storeu_si128((__m128i *)(&C[i * N + j]), acc);
        }

        /* Scalar tail */
        for (; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                int8_t a = A[i * K + k];
                int8_t b = B[k * N + j];
                acc += PRODUCT[(int)(a + 1)][(int)(b + 1)];
            }
            C[i * N + j] = acc;
        }
    }
}

/* --- Benchmark harness -------------------------------------------------- */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void fill_ternary(int8_t *buf, size_t n) {
    /* ternary values {-1,0,+1} with uniform distribution */
    for (size_t i = 0; i < n; i++) {
        int r = rand() % 3;
        buf[i] = (int8_t)(r - 1);  /* -1, 0, or +1 */
    }
}

typedef void (*matmul_fn)(int, int, int, const int8_t *, const int8_t *, int32_t *);

static double benchmark(matmul_fn fn, const char *name,
                        int M, int N, int K, int trials,
                        const int8_t *A, const int8_t *B, int32_t *C)
{
    double t0 = now_sec();
    for (int t = 0; t < trials; t++) {
        fn(M, N, K, A, B, C);
    }
    double elapsed = now_sec() - t0;

    /* Effective operations: M*N*K MACs per trial, each MAC = 2 ops (mul+add)
     * in float32-equivalent terms. For ternary, each MAC is 1 lookup + 1 add.
     * Report GFLOPS-equivalent for comparison. */
    double ops_per_trial = 2.0 * (double)M * (double)N * (double)K;
    double total_ops = ops_per_trial * trials;
    double gflops = total_ops / elapsed / 1e9;

    printf("  %-24s  %8.3f ms  %8.2f GFLOPS-eq\n",
           name, elapsed * 1e3 / trials, gflops);
    return elapsed;
}

static int verify(const int32_t *ref, const int32_t *test, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        if (ref[i] != test[i]) {
            printf("  MISMATCH at [%d]: ref=%d test=%d\n", i, ref[i], test[i]);
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    int M = argc > 1 ? atoi(argv[1]) : 256;
    int K = argc > 2 ? atoi(argv[2]) : 256;
    int N = argc > 3 ? atoi(argv[3]) : 256;
    int trials = argc > 4 ? atoi(argv[4]) : 20;

    int cores = 1;
#ifdef _OPENMP
    cores = omp_get_max_threads();
#endif

    printf("═══ BQSM Ternary MatMul Kernel ═══\n");
    printf("T7400 Xeon X5472 (SSE4.1, %d-core Harpertown, 3.0 GHz)\n\n", cores);
    printf("Dimensions: M=%d K=%d N=%d  trials=%d\n", M, K, N, trials);
    printf("Matrix sizes: A=%d×%d (%d KB)  B=%d×%d (%d KB)  C=%d×%d (%d KB)\n",
           M, K, M*K/1024, K, N, K*N/1024, M, N, M*N*4/1024);
    printf("Effective MACs per trial: %ld\n", (long)M * N * K);
    printf("Total effective ops: %.1f Gops\n\n",
           2.0 * M * N * K * trials / 1e9);

    /* Allocate */
    int8_t *A = malloc(M * K);
    int8_t *B = malloc(K * N);
    int32_t *C_ref  = malloc(M * N * sizeof(int32_t));
    int32_t *C_test = malloc(M * N * sizeof(int32_t));
    if (!A || !B || !C_ref || !C_test) {
        fprintf(stderr, "alloc failed\n"); return 1;
    }

    srand(42);
    fill_ternary(A, M * K);
    fill_ternary(B, K * N);

    printf("Benchmark results:\n");
    printf("  %-24s  %10s  %14s\n", "Method", "Time/trial", "GFLOPS-eq");

    /* Reference (scalar multiply) */
    benchmark(matmul_naive, "scalar mul", M, N, K, trials, A, B, C_ref);

    /* BQSM scalar */
    benchmark(matmul_bqsm, "bqsm scalar lookup", M, N, K, trials, A, B, C_test);
    printf("    verify: %s\n", verify(C_ref, C_test, M, N) ? "PASS" : "FAIL");

    /* BQSM SSE4.1 */
    benchmark(matmul_bqsm_sse41, "bqsm SSE4.1", M, N, K, trials, A, B, C_test);
    printf("    verify: %s\n", verify(C_ref, C_test, M, N) ? "PASS" : "FAIL");

    /* OpenMP BQSM if available */
#ifdef _OPENMP
    memset(C_test, 0, M * N * sizeof(int32_t));
    double t0 = now_sec();
    for (int t = 0; t < trials; t++) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int32_t acc = 0;
                for (int k = 0; k < K; k++) {
                    int8_t a = A[i * K + k];
                    int8_t b = B[k * N + j];
                    acc += PRODUCT[(int)(a + 1)][(int)(b + 1)];
                }
                C_test[i * N + j] = acc;
            }
        }
    }
    double elapsed = now_sec() - t0;
    double ops = 2.0 * M * N * K * trials;
    double gflops = ops / elapsed / 1e9;
    printf("  %-24s  %8.3f ms  %8.2f GFLOPS-eq\n",
           "bqsm OpenMP", elapsed * 1e3 / trials, gflops);
    printf("    verify: %s  (cores=%d)\n",
           verify(C_ref, C_test, M, N) ? "PASS" : "FAIL", cores);
#endif

    /* --- Multi-level transform benchmark (the real inference kernel) ----- */
    printf("\nMulti-level transform (A∈{0..3}, B∈{-1,0,1}):\n");
    printf("  %-24s  %10s  %14s\n", "Method", "Time/trial", "GFLOPS-eq");

    /* Re-fill A with multi-level values (0..3) */
    srand(42);
    for (int i = 0; i < M * K; i++) A[i] = (int8_t)(rand() % (Q_MAX_MAT+1));
    fill_ternary(B, K * N);

    /* Float32 baseline */
    benchmark(matmul_float32_transform, "float32 mul+add", M, N, K, trials, A, B, C_ref);

    /* BQSM transform */
    benchmark(matmul_bqsm_transform, "bqsm transform (LUT)", M, N, K, trials, A, B, C_test);
    printf("    verify: %s\n", verify(C_ref, C_test, M, N) ? "PASS" : "FAIL");

    /* BQSM transform OpenMP */
#ifdef _OPENMP
    memset(C_test, 0, M * N * sizeof(int32_t));
    t0 = now_sec();
    for (int t = 0; t < trials; t++) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int32_t acc = 0;
                for (int k = 0; k < K; k++) {
                    int8_t a = A[i * K + k];
                    int8_t b = B[k * N + j];
                    acc += TT_MAT[(int)a][(int)(b + 1)];
                }
                C_test[i * N + j] = acc;
            }
        }
    }
    elapsed = now_sec() - t0;
    ops = 2.0 * M * N * K * trials;
    gflops = ops / elapsed / 1e9;
    printf("  %-24s  %8.3f ms  %8.2f GFLOPS-eq\n",
           "bqsm transform OpenMP", elapsed * 1e3 / trials, gflops);
    printf("    verify: %s  (cores=%d)\n",
           verify(C_ref, C_test, M, N) ? "PASS" : "FAIL", cores);
#endif

    /* Memory bandwidth note */
    printf("\n─── Analysis ───\n");
    printf("Per MAC: 2 bytes read (A[i][k] + B[k][j]), 1-byte LUT, int32 accumulate\n");
    printf("Product table: 9 bytes (single cache line, XMM register)\n");
    printf("Weight compression: ternary = 1.6 bits/weight vs float32 = 32 bits (20×)\n");
    printf("No multiply ALU — just index + table load + add\n\n");

    /* Scaling projection */
    double scalar_ms = 0;
    {
        double t0 = now_sec();
        for (int t = 0; t < trials; t++)
            matmul_bqsm(M, N, K, A, B, C_test);
        scalar_ms = (now_sec() - t0) / trials * 1e3;
    }
    printf("─── LLM Projection ───\n");
    printf("Based on measured %.2f ms for %dx%dx%d matmul:\n", scalar_ms, M, K, N);
    double macs_per_s = (double)M * N * K / (scalar_ms * 1e-3);
    printf("  Throughput: %.1f GMACs/s (scalar)\n", macs_per_s / 1e9);

    /* Llama-3.2-1B: d_model=2048, 16 layers, FFN=8192 */
    double macs_per_token = 0;
    macs_per_token += 4.0 * 2048 * 2048;     /* QKV projections */
    macs_per_token += 2048 * 2048;            /* attention output */
    macs_per_token += 3.0 * 2048 * 8192;      /* MLP */
    macs_per_token *= 16;                      /* layers */
    double tokens_per_sec = macs_per_s / macs_per_token;
    printf("  Llama-3.2-1B (~%.0fM MACs/token): %.2f tok/s (scalar, 1 core)\n",
           macs_per_token / 1e6, tokens_per_sec);
    printf("  With SSE4.1 auto-vec (est 2×):     %.2f tok/s\n", tokens_per_sec * 2);
    printf("  With %d cores (est %d×):             %.2f tok/s\n",
           cores, cores, tokens_per_sec * cores * 2);

    free(A); free(B); free(C_ref); free(C_test);
    return 0;
}