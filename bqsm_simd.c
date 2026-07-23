/* bqsm_simd.c — pshufb-accelerated BQSM matmul, 16 MACs per instruction */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <tmmintrin.h>

#define QS 3
static int8_t TT[16] __attribute__((aligned(16)));

static void init_tt(void) {
    for (int a = 0; a <= QS; a++)
        for (int b = 0; b <= QS; b++) {
            int p = (a * b + 1) / 2;
            TT[(a << 2) | b] = (int8_t)(p < 0 ? 0 : (p > QS ? QS : p));
        }
}

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void matmul_ref(int M, int N, int K,
                       const int8_t *A, const int8_t *B, int32_t *C)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += TT[((int)A[i*K+k] << 2) | (int)B[k*N+j]];
            C[i*N+j] = acc;
        }
}

static void matmul_sse(int M, int N, int K,
                       const int8_t *A, const int8_t *B, int32_t *C)
{
    __m128i tt = _mm_load_si128((__m128i*)TT);
    __m128i zero = _mm_setzero_si128();

    for (int i = 0; i < M; i++) {
        int j = 0;
        for (; j + 15 < N; j += 16) {
            __m128i acc0 = zero, acc1 = zero, acc2 = zero, acc3 = zero;
            for (int k = 0; k < K; k++) {
                int a = (int)A[i*K + k];
                __m128i a4 = _mm_set1_epi8((char)(a << 2));
                __m128i b = _mm_loadu_si128((__m128i*)&B[k*N + j]);
                b = _mm_and_si128(b, _mm_set1_epi8(3));
                __m128i idx = _mm_or_si128(a4, b);
                __m128i prod = _mm_shuffle_epi8(tt, idx);

                /* Unpack int8 → int32 manually (SSSE3 doesn't have cvtepi8_epi32).
                 * punpcklbw/punpcklwd sign-extend lower half. */
                __m128i zero16 = _mm_setzero_si128();
                __m128i lo16 = _mm_unpacklo_epi8(prod, _mm_cmplt_epi8(prod, zero16)); /* sign extend */
                /* Actually use the classic: unpacklo + unpackhi → 2× int16, then unpack → int32 */
                __m128i prod_hi = _mm_unpackhi_epi8(prod, prod);
                __m128i prod_lo = _mm_unpacklo_epi8(prod, prod);
                prod_lo = _mm_srai_epi16(_mm_unpacklo_epi8(prod, prod), 8);
                /* Simpler: just use movsx via union or scalar. For now, use the fact that
                 * values are 0..3 (non-negative), so zero-extend is fine. */
                __m128i p0 = _mm_unpacklo_epi16(_mm_unpacklo_epi8(prod, zero16), zero16);
                __m128i p1 = _mm_unpackhi_epi16(_mm_unpacklo_epi8(prod, zero16), zero16);
                __m128i p2 = _mm_unpacklo_epi16(_mm_unpackhi_epi8(prod, zero16), zero16);
                __m128i p3 = _mm_unpackhi_epi16(_mm_unpackhi_epi8(prod, zero16), zero16);

                acc0 = _mm_add_epi32(acc0, p0);
                acc1 = _mm_add_epi32(acc1, p1);
                acc2 = _mm_add_epi32(acc2, p2);
                acc3 = _mm_add_epi32(acc3, p3);
            }
            _mm_storeu_si128((__m128i*)&C[i*N + j + 0],  acc0);
            _mm_storeu_si128((__m128i*)&C[i*N + j + 4],  acc1);
            _mm_storeu_si128((__m128i*)&C[i*N + j + 8],  acc2);
            _mm_storeu_si128((__m128i*)&C[i*N + j + 12], acc3);
        }
        for (; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += TT[((int)A[i*K+k] << 2) | (int)B[k*N+j]];
            C[i*N+j] = acc;
        }
    }
}

int main(void) {
    init_tt();
    int M=64, K=256, N=256, trials=20;
    printf("═══ BQSM pshufb Matmul — 4-state, 16-wide ═══\n%d×%d×%d, %d trials\n\n", M,K,N,trials);

    int8_t *A=malloc(M*K), *B=malloc(K*N);
    int32_t *Cr=malloc(M*N*4), *Cs=malloc(M*N*4);
    srand(42);
    for(int i=0;i<M*K;i++)A[i]=rand()%(QS+1);
    for(int i=0;i<K*N;i++)B[i]=rand()%(QS+1);

    double t0=now_sec();
    for(int t=0;t<trials;t++)matmul_ref(M,N,K,A,B,Cr);
    double rms=(now_sec()-t0)/trials*1e3;
    double rg=(double)M*N*K/(rms*1e-3)/1e9;

    t0=now_sec();
    for(int t=0;t<trials;t++)matmul_sse(M,N,K,A,B,Cs);
    double sms=(now_sec()-t0)/trials*1e3;
    double sg=(double)M*N*K/(sms*1e-3)/1e9;

    int ok=1;
    for(int i=0;i<M*N;i++)if(Cr[i]!=Cs[i]){ok=0;printf("MISMATCH %d: %d vs %d\n",i,Cr[i],Cs[i]);break;}

    printf("Ref: %7.1fms  %6.1f GMACs/s\n", rms, rg);
    printf("SSE: %7.1fms  %6.1f GMACs/s  %.1f×  %s\n", sms, sg, sg/rg, ok?"PASS":"FAIL");
    printf("\npshufb: 16 lookups/insn, %.1f cycles/MAC\n", 3.0e9/(sg*1e9));
    printf("16-wide SIMD on 4-state (2-bit) TT at 16 bytes\n");

    free(A);free(B);free(Cr);free(Cs);
    return ok?0:1;
}