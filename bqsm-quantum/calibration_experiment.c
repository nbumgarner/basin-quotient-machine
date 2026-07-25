/* ===========================================================================
 * calibration_experiment.c — Calibrate Coupling Proportionality
 *
 *   Tests different (a,b) inputs to see if ratios converge to expected values.
 *   Expected ratios: a²/a·b = a/b, a²/b² = (a/b)²
 *   For a=3,b=2: expected a²/a·b = 1.5, a²/b² = 2.25
 *   For a=4,b=1: expected a²/a·b = 4.0, a²/b² = 16.0
 *
 *   Lens: site0_±0.5 (24.38× b² enhancement)
 *   Baseline subtraction for noise removal
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

/* Encode two values into harmonics p and q */
static void encode_pair(ring_t *r,int p,double A,int q,double B){
    memset(r->th,0,N*sizeof(double));
    memset(r->om,0,N*sizeof(double));
    for(int j=0;j<N;j++){
        r->th[j]=2.0*A*cos(2.0*M_PI*p*j/N)+2.0*B*cos(2.0*M_PI*q*j/N);
    }
}

/* Apply site0_±0.5 lens profile */
static void apply_site0_lens(ring_t *r,double sign){
    memset(r->om,0,N*sizeof(double));
    r->om[0]=sign*0.5;  /* ±0.5 at oscillator 0 */
}

/* Measure |c_k| with given lens */
static double measure_with_lens(ring_t *r,int p,double A,int q,double B,int k_out,double sign,int chunks){
    encode_pair(r,p,A,q,B);
    apply_site0_lens(r,sign);
    settle(r,chunks);
    return ck_mag(r,k_out);
}

/* Baseline: measure |c_k| without lens */
static double measure_baseline(ring_t *r,int p,double A,int q,double B,int k_out,int chunks){
    encode_pair(r,p,A,q,B);
    memset(r->om,0,N*sizeof(double));
    settle(r,chunks);
    return ck_mag(r,k_out);
}

int main(){
    printf("═══ Calibration Experiment: site0_±0.5 Lens Profile ═══\n");
    printf("Testing different (a,b) inputs to check ratio convergence\n\n");
    
    int chunks = 2;
    double lens_sign = 0.5;  /* +0.5 */
    
    /* Test cases */
    struct {double a, b; const char *desc;} test_cases[] = {
        {2.0, 1.5, "a=2.0, b=1.5 (previous)"},
        {3.0, 2.0, "a=3.0, b=2.0 (Precision1)"},
        {4.0, 1.0, "a=4.0, b=1.0 (Precision1)"},
        {1.0, 1.0, "a=1.0, b=1.0 (control)"},
        {5.0, 2.5, "a=5.0, b=2.5 (scaled)"},
        {0.0, 0.0, 0}
    };
    
    printf("Expected ratios: a²/a·b = a/b, a²/b² = (a/b)²\n");
    printf("For a=3,b=2: expected a²/a·b = 1.5, a²/b² = 2.25\n");
    printf("For a=4,b=1: expected a²/a·b = 4.0, a²/b² = 16.0\n\n");
    
    ring_t r;
    int idx = 0;
    
    while(test_cases[idx].a != 0.0){
        double a = test_cases[idx].a;
        double b = test_cases[idx].b;
        
        printf("TEST %d: %s\n", idx+1, test_cases[idx].desc);
        printf("  INPUT: a=%.1f, b=%.1f\n", a, b);
        
        /* Measure coupled signals */
        double c2_coupled = measure_with_lens(&r, 1, a, 2, b, 2, lens_sign, chunks);
        double c3_coupled = measure_with_lens(&r, 1, a, 2, b, 3, lens_sign, chunks);
        double c4_coupled = measure_with_lens(&r, 2, a, 2, b, 4, lens_sign, chunks);
        
        /* Baseline subtraction */
        double c2_baseline = measure_baseline(&r, 1, a, 2, b, 2, chunks);
        double c3_baseline = measure_baseline(&r, 1, a, 2, b, 3, chunks);
        double c4_baseline = measure_baseline(&r, 2, a, 2, b, 4, chunks);
        
        double c2_signal = c2_coupled - c2_baseline;
        double c3_signal = c3_coupled - c3_baseline;
        double c4_signal = c4_coupled - c4_baseline;
        
        printf("  COUPLED:    |c₂|=%.4f, |c₃|=%.4f, |c₄|=%.4f\n", c2_coupled, c3_coupled, c4_coupled);
        printf("  BASELINE:   |c₂|=%.4f, |c₃|=%.4f, |c₄|=%.4f\n", c2_baseline, c3_baseline, c4_baseline);
        printf("  SIGNALS:    a²=%.4f, a·b=%.4f, b²=%.4f\n", c2_signal, c3_signal, c4_signal);
        
        /* Ratios */
        double ratio_a2_ab = (c3_signal>1e-9) ? (c2_signal/c3_signal) : 0.0;
        double ratio_a2_b2 = (c4_signal>1e-9) ? (c2_signal/c4_signal) : 0.0;
        
        printf("  MEASURED RATIOS: a²/a·b = %.2f, a²/b² = %.2f\n", ratio_a2_ab, ratio_a2_b2);
        
        /* Expected ratios */
        double expected_a2_ab = (b>1e-9) ? (a/b) : 0.0;
        double expected_a2_b2 = (b>1e-9) ? (a*a/(b*b)) : 0.0;
        
        printf("  EXPECTED:    a²/a·b = %.2f, a²/b² = %.2f\n", expected_a2_ab, expected_a2_b2);
        
        /* Ratio errors */
        double error_a2_ab = (expected_a2_ab>1e-9) ? fabs(ratio_a2_ab - expected_a2_ab)/expected_a2_ab*100.0 : 0.0;
        double error_a2_b2 = (expected_a2_b2>1e-9) ? fabs(ratio_a2_b2 - expected_a2_b2)/expected_a2_b2*100.0 : 0.0;
        
        printf("  ERROR:       a²/a·b = %.1f%%, a²/b² = %.1f%%\n", error_a2_ab, error_a2_b2);
        printf("\n");
        
        idx++;
    }
    
    printf("═══ SUMMARY ═══\n");
    printf("Check if ratio errors improve with different (a,b) values.\n");
    printf("If ratios converge to expected values, coupling is mathematically accurate.\n");
    printf("If ratios stay distorted, coupling has nonlinear distortions.\n");
    
    return 0;
}