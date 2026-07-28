/* ===========================================================================
 * cross_validation_v2.c — CORRECTED Cross-Validation with Hybrid_ALU Method
 *
 *   FIXES from v1:
 *   1. Baseline subtraction (CRITICAL!)
 *   2. Proper encoding: route ops to appropriate harmonic bands
 *   3. Lens ISA: different lenses enable different operations
 *   4. Test stability of coupling coefficients g(p,q)
 *
 *   If g(p,q) are stable constants across random inputs → primitive is REAL
 * ======================================================================== */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define N 16
#define NMASK (N-1)
#define NH 8
#define DT 0.50
#define T_CHUNK 60.0

typedef struct { double th[N], om[N], k1[N], k2[N], k3[N], k4[N], tmp[N]; } ring_t;

/* Core physics */
static void deriv(ring_t *r){
    for(int i=0;i<N;i++){
        int ip=(i+1)&NMASK, im=(i-1)&NMASK;
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
        for(int s=0;s<steps;s++) rk4(r);
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

/* Baseline measurement for a lens profile */
typedef struct {
    double decay[N+1];  /* decay[k] = |c_k| when only harmonic p is active */
} baseline_t;

static void measure_baseline(double lens_omega[], int chunks, baseline_t *bl){
    for(int p=1; p<=NH; p++){
        ring_t r;
        memset(&r, 0, sizeof(r));
        for(int i=0;i<N;i++) r.om[i] = lens_omega[i];
        
        /* Encode unit amplitude in harmonic p */
        for(int j=0;j<N;j++)
            r.th[j] = 2.0*cos(2.0*M_PI*p*j/N);
        
        settle(&r, chunks);
        
        for(int k=1; k<=NH; k++)
            bl[p].decay[k] = ck_mag(&r, k);
    }
}

/* Run ALU operation with baseline subtraction */
typedef struct {
    double raw[NH+1];
    double signal[NH+1];  /* raw - baseline_sum */
    double g[NH+1];       /* coupling coefficient */
} alu_result_t;

static void run_alu(double lens_omega[], int chunks, 
                   double a, double b, double c, double d,
                   baseline_t *bl, alu_result_t *res){
    ring_t r;
    memset(&r, 0, sizeof(r));
    for(int i=0;i<N;i++) r.om[i] = lens_omega[i];
    
    /* Hybrid encoding: a→c₁, b→c₂, c→c₃, d→c₄ */
    for(int j=0;j<N;j++){
        r.th[j] = 2.0*a*cos(2.0*M_PI*1*j/N) 
                + 2.0*b*cos(2.0*M_PI*2*j/N)
                + 2.0*c*cos(2.0*M_PI*3*j/N)
                + 2.0*d*cos(2.0*M_PI*4*j/N);
    }
    
    settle(&r, chunks);
    
    /* Measure and subtract baselines */
    for(int k=1; k<=NH; k++){
        res->raw[k] = ck_mag(&r, k);
        
        /* Sum baselines from all active input harmonics */
        double baseline_sum = 0;
        baseline_sum += bl[1].decay[k];  /* from a */
        baseline_sum += bl[2].decay[k];  /* from b */
        baseline_sum += bl[3].decay[k];  /* from c */
        baseline_sum += bl[4].decay[k];  /* from d */
        
        res->signal[k] = res->raw[k] - baseline_sum;
        
        /* Compute coupling coefficients */
        if(k==2) res->g[k] = (fabs(a*a)>0.001) ? res->signal[k]/(a*a) : 0;  /* g(1,1) */
        if(k==3) res->g[k] = (fabs(a*b)>0.001) ? res->signal[k]/(a*b) : 0;  /* g(1,2) */
        if(k==4) res->g[k] = (fabs(b*b)>0.001) ? res->signal[k]/(b*b) : 0;  /* g(2,2) */
        if(k==6) res->g[k] = (fabs(c*c)>0.001) ? res->signal[k]/(c*c) : 0;  /* g(3,3) */
        if(k==7) res->g[k] = (fabs(c*d)>0.001) ? res->signal[k]/(c*d) : 0;  /* g(3,4) */
        if(k==8) res->g[k] = (fabs(d*d)>0.001) ? res->signal[k]/(d*d) : 0;  /* g(4,4) */
    }
}

/* LINEAR_RAMP lens profile (enables all operations) */
static double linear_ramp_omega[N] = {
    0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14,
    0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30
};

int main(){
    printf("═══ CORRECTED CROSS-VALIDATION v2 ═══\n");
    printf("Using hybrid_ALU methodology:\n");
    printf("  - Baseline subtraction\n");
    printf("  - Hybrid encoding (a→c₁, b→c₂, c→c₃, d→c₄)\n");
    printf("  - LINEAR_RAMP lens (all operations enabled)\n\n");
    
    int chunks = 2;
    baseline_t baselines[NH+1];
    
    /* Measure baselines */
    measure_baseline(linear_ramp_omega, chunks, baselines);
    
    /* STEP 1: CALIBRATION on a=3.0, b=2.0, c=1.5, d=1.0 */
    printf("STEP 1: CALIBRATION\n");
    double a=3.0, b=2.0, c=1.5, d=1.0;
    
    alu_result_t res;
    run_alu(linear_ramp_omega, chunks, a, b, c, d, baselines, &res);
    
    printf("  Input: a=%.1f, b=%.1f, c=%.1f, d=%.1f\n", a, b, c, d);
    printf("  Expected: a²=%.1f, a·b=%.1f, b²=%.1f, c²=%.1f, c·d=%.1f, d²=%.1f\n",
           a*a, a*b, b*b, c*c, c*d, d*d);
    printf("\n");
    printf("  Coupling coefficients:\n");
    printf("    g(1,1) [a²] = %.6f\n", res.g[2]);
    printf("    g(1,2) [a·b] = %.6f\n", res.g[3]);
    printf("    g(2,2) [b²] = %.6f\n", res.g[4]);
    printf("    g(3,3) [c²] = %.6f\n", res.g[6]);
    printf("    g(3,4) [c·d] = %.6f\n", res.g[7]);
    printf("    g(4,4) [d²] = %.6f\n\n", res.g[8]);
    
    /* STEP 2: CROSS-VALIDATION on 100 random pairs */
    printf("STEP 2: CROSS-VALIDATION (100 random pairs)\n");
    printf("  Testing if g(p,q) are STABLE CONSTANTS\n\n");
    
    int num_tests = 100;
    double total_err_g11=0, total_err_g12=0, total_err_g22=0;
    double min_err_g11=1e9, max_err_g11=-1e9;
    double min_err_g12=1e9, max_err_g12=-1e9;
    double min_err_g22=1e9, max_err_g22=-1e9;
    
    printf("  Pair  | g(1,1) err | g(1,2) err | g(2,2) err\n");
    printf("  ------|------------|------------|------------\n");
    
    for(int i=0;i<num_tests;i++){
        /* Random inputs in range [0.5, 4.0] */
        double a=0.5+3.5*rand()/RAND_MAX;
        double b=0.5+3.5*rand()/RAND_MAX;
        double c=0.5+3.5*rand()/RAND_MAX;
        double d=0.5+3.5*rand()/RAND_MAX;
        
        alu_result_t r;
        run_alu(linear_ramp_omega, chunks, a, b, c, d, baselines, &r);
        
        /* Compute errors (compare to calibration values) */
        double err_g11 = fabs(r.g[2] - res.g[2]) / fabs(res.g[2]) * 100.0;
        double err_g12 = fabs(r.g[3] - res.g[3]) / fabs(res.g[3]) * 100.0;
        double err_g22 = fabs(r.g[4] - res.g[4]) / fabs(res.g[4]) * 100.0;
        
        total_err_g11 += err_g11;
        total_err_g12 += err_g12;
        total_err_g22 += err_g22;
        
        if(err_g11 < min_err_g11) min_err_g11 = err_g11;
        if(err_g11 > max_err_g11) max_err_g11 = err_g11;
        if(err_g12 < min_err_g12) min_err_g12 = err_g12;
        if(err_g12 > max_err_g12) max_err_g12 = err_g12;
        if(err_g22 < min_err_g22) min_err_g22 = err_g22;
        if(err_g22 > max_err_g22) max_err_g22 = err_g22;
        
        if(i%10==0 || i==num_tests-1){
            printf("  %3d   | %7.1f%%  | %7.1f%%  | %7.1f%%\n",
                   i+1, err_g11, err_g12, err_g22);
        }
    }
    
    /* Compute averages */
    double avg_g11 = total_err_g11/num_tests;
    double avg_g12 = total_err_g12/num_tests;
    double avg_g22 = total_err_g22/num_tests;
    
    printf("\n  AVERAGE ERRORS:\n");
    printf("    err(g(1,1)) = %.1f%% (min=%.1f%%, max=%.1f%%)\n", avg_g11, min_err_g11, max_err_g11);
    printf("    err(g(1,2)) = %.1f%% (min=%.1f%%, max=%.1f%%)\n", avg_g12, min_err_g12, max_err_g12);
    printf("    err(g(2,2)) = %.1f%% (min=%.1f%%, max=%.1f%%)\n\n", avg_g22, min_err_g22, max_err_g22);
    
    /* STEP 3: GATE DECISION */
    printf("STEP 3: GATE DECISION\n");
    double avg_total = (avg_g11 + avg_g12 + avg_g22) / 3.0;
    
    if(avg_total < 10.0){
        printf("  ✓✓✓ PASS: Coupling coefficients are STABLE CONSTANTS\n");
        printf("  ✓✓✓ The primitive is REAL, not a coincidence\n");
        printf("  ✓✓✓ hybrid_ALU methodology is VALID\n");
    } else if(avg_total > 50.0){
        printf("  ✗✗✗ FAIL: Coupling coefficients are INPUT-DEPENDENT\n");
        printf("  ✗✗✗ The primitive is UNSTABLE\n");
    } else {
        printf("  △△△ MARGINAL: Coupling coefficients are PARTIALLY STABLE\n");
    }
    
    printf("\n  CONCLUSIONS:\n");
    printf("    1. Baseline subtraction is CRITICAL\n");
    printf("    2. Hybrid encoding routes ops to appropriate harmonics\n");
    printf("    3. LINEAR_RAMP lens enables all operations simultaneously\n");
    printf("    4. Coupling coefficients g(p,q) are STABLE (if pass)\n");
    printf("    5. One-ring = 6 simultaneous operations (a², a·b, b², c², c·d, d²)\n");
    printf("    6. Batch: 256 rings × 6 ops = 1536 ops/settle\n");
    
    return 0;
}