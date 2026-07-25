/* ===========================================================================
 * encoding_scheme_experiment.c — Test New Encoding Scheme
 *
 *   Precision1's proposed encoding:
 *   a² → encode a into p=1 ONLY (b=0), measure |c₂|
 *   a·b → encode a into p=1, b into q=2, measure |c₃|
 *   b² → encode b into p=2 ONLY (a=0), measure |c₄|
 *
 *   Test if signals scale with products a², a*b, b²
 * ======================================================================== */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define N  16
#define NMASK (N-1)
#define NH  8
#define DT  0.50
#define T_CHUNK 60.0

typedef struct { double th[N], om[N], k1[N], k2[N], k3[N], k4[N], tmp[N]; } ring_t;

/* --- Core physics functions --- */
static void deriv(ring_t *r){
    for(int i=0;i<N;i++){
        int ip=(i+1)&NMASK;
        int im=(i-1)&NMASK;
        r->tmp[i]=r->om[i]+sin(r->th[ip]-r->th[i])+sin(r->th[im]-r->th[i]);
    }
}
static void rk4(ring_t *r){
    deriv(r);
    for(int i=0;i<N;i++)r->k1[i]=r->tmp[i];
    for(int i=0;i<N;i++)r->tmp[i]=r->th[i]+0.5*DT*r->k1[i];
    deriv(r);
    for(int i=0;i<N;i++)r->k2[i]=r->tmp[i];
    for(int i=0;i<N;i++)r->tmp[i]=r->th[i]+0.5*DT*r->k2[i];
    deriv(r);
    for(int i=0;i<N;i++)r->k3[i]=r->tmp[i];
    for(int i=0;i<N;i++)r->tmp[i]=r->th[i]+DT*r->k3[i];
    deriv(r);
    for(int i=0;i<N;i++)r->k4[i]=r->tmp[i];
    for(int i=0;i<N;i++)r->th[i]+=(DT/6.0)*(r->k1[i]+2.0*r->k2[i]+2.0*r->k3[i]+r->k4[i]);
}
static void settle(ring_t *r,int count){
    int steps=(int)(T_CHUNK/DT);
    for(int c=0;c<count;c++){
        for(int s=0;s<steps;s++){
            rk4(r);
        }
    }
}

static double ck_mag(ring_t *r,int k){
    double re=0,im=0;
    for(int j=0;j<N;j++){
        double a=-2.0*M_PI*k*j/N;
        re+=r->th[j]*cos(a);
        im+=r->th[j]*sin(a);
    }
    return sqrt(re*re+im*im)/N;
}

/* Encode SINGLE harmonic p with amplitude A (for a² or b²) */
static void encode_single(ring_t *r,int p,double A){
    memset(r->th,0,N*sizeof(double));
    memset(r->om,0,N*sizeof(double));
    for(int j=0;j<N;j++){
        r->th[j]=2.0*A*cos(2.0*M_PI*p*j/N);
    }
}

/* Encode TWO harmonics p and q with amplitudes A and B (for a·b) */
static void encode_pair(ring_t *r,int p,double A,int q,double B){
    memset(r->th,0,N*sizeof(double));
    memset(r->om,0,N*sizeof(double));
    for(int j=0;j<N;j++){
        r->th[j]=2.0*A*cos(2.0*M_PI*p*j/N)+2.0*B*cos(2.0*M_PI*q*j/N);
    }
}

/* Apply site0_±0.5 lens */
static void apply_site0_lens(ring_t *r,double sign){
    memset(r->om,0,N*sizeof(double));
    r->om[0]=sign*0.5;
}

/* Measure |c_k| with lens */
static double measure_with_lens(ring_t *r,int k_out,double sign,int chunks){
    settle(r,chunks);
    return ck_mag(r,k_out);
}

/* Baseline: measure |c_k| without lens */
static double measure_baseline(ring_t *r,int k_out,int chunks){
    memset(r->om,0,N*sizeof(double));
    settle(r,chunks);
    return ck_mag(r,k_out);
}

int main(){
    printf("═══ Encoding Scheme Experiment — Precision1's Proposal ═══\n");
    printf("Testing if coupling computes products with separate encoding\n\n");
    
    int chunks = 2;
    double lens_sign = 0.5;
    
    /* Test values */
    double a_vals[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double b_vals[] = {1.0, 1.5, 2.0, 2.5, 3.0};
    
    printf("TEST 1: a² encoding (p=1 ONLY, b=0)\n");
    printf("  Measure |c₂| for different 'a' values\n");
    printf("  If coupling computes a², signal should scale with a²\n\n");
    
    ring_t r;
    for(int i=0;i<5;i++){
        double a = a_vals[i];
        encode_single(&r, 1, a);  /* encode a into p=1 */
        apply_site0_lens(&r, lens_sign);
        
        double coupled = measure_with_lens(&r, 2, lens_sign, chunks);
        double baseline = measure_baseline(&r, 2, chunks);
        double signal = coupled - baseline;
        
        printf("  a=%.1f: coupled=%.4f, baseline=%.4f, signal=%.4f, expected a²=%.1f\n", 
               a, coupled, baseline, signal, a*a);
    }
    
    printf("\nTEST 2: a·b encoding (p=1, q=2)\n");
    printf("  Measure |c₃| for different (a,b) pairs\n");
    printf("  If coupling computes a·b, signal should scale with a*b\n\n");
    
    for(int i=0;i<5;i++){
        double a = a_vals[i];
        double b = b_vals[i];
        encode_pair(&r, 1, a, 2, b);  /* encode a into p=1, b into q=2 */
        apply_site0_lens(&r, lens_sign);
        
        double coupled = measure_with_lens(&r, 3, lens_sign, chunks);
        double baseline = measure_baseline(&r, 3, chunks);
        double signal = coupled - baseline;
        
        printf("  a=%.1f,b=%.1f: coupled=%.4f, baseline=%.4f, signal=%.4f, expected a·b=%.1f\n", 
               a, b, coupled, baseline, signal, a*b);
    }
    
    printf("\nTEST 3: b² encoding (p=2 ONLY, a=0)\n");
    printf("  Measure |c₄| for different 'b' values\n");
    printf("  If coupling computes b², signal should scale with b²\n\n");
    
    for(int i=0;i<5;i++){
        double b = b_vals[i];
        encode_single(&r, 2, b);  /* encode b into p=2 */
        apply_site0_lens(&r, lens_sign);
        
        double coupled = measure_with_lens(&r, 4, lens_sign, chunks);
        double baseline = measure_baseline(&r, 4, chunks);
        double signal = coupled - baseline;
        
        printf("  b=%.1f: coupled=%.4f, baseline=%.4f, signal=%.4f, expected b²=%.1f\n", 
               b, coupled, baseline, signal, b*b);
    }
    
    printf("\n═══ ANALYSIS ═══\n");
    printf("If signals scale with products (a², a·b, b²), coupling computes products.\n");
    printf("If signals are constant, coupling is fixed coefficient g(p,q).\n");
    printf("If signals scale but not exactly, coupling is approximate product detector.\n");
    
    return 0;
}