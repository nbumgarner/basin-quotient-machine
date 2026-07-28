/* ===========================================================================
 * product_computation_accuracy.c — Test Actual Product Accuracy
 *
 *   Using lookup table to compute a², a·b, b²
 *   Calculate REAL accuracy (not percentage errors that explode near zero)
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

static double linear_ramp_omega[N] = {
    0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14,
    0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30
};

typedef struct { double decay[NH+1]; } baseline_t;

static void measure_baseline(double lens_omega[], int chunks, baseline_t *bl){
    for(int p=1; p<=NH; p++){
        ring_t r;
        memset(&r, 0, sizeof(r));
        for(int i=0;i<N;i++) r.om[i] = lens_omega[i];
        for(int j=0;j<N;j++)
            r.th[j] = 2.0*cos(2.0*M_PI*p*j/N);
        settle(&r, chunks);
        for(int k=1; k<=NH; k++)
            bl[p].decay[k] = ck_mag(&r, k);
    }
}

typedef struct { double g[9]; } coupling_t;

/* Measure signal at harmonic k with given inputs */
static double measure_signal(double lens_omega[], int chunks,
                            double a, double b, int k){
    baseline_t baselines[NH+1];
    measure_baseline(lens_omega, chunks, baselines);
    
    ring_t r;
    memset(&r, 0, sizeof(r));
    for(int i=0;i<N;i++) r.om[i] = lens_omega[i];
    
    for(int j=0;j<N;j++){
        r.th[j] = 2.0*a*cos(2.0*M_PI*1*j/N) 
                + 2.0*b*cos(2.0*M_PI*2*j/N);
    }
    
    settle(&r, chunks);
    
    double raw = ck_mag(&r, k);
    double baseline_sum = baselines[1].decay[k] + baselines[2].decay[k];
    return raw - baseline_sum;
}

/* Interpolate g from lookup table */
typedef struct {
    double a[7], b[7];
    coupling_t g[7][7];
} lookup_table_t;

static void interpolate_g(lookup_table_t *lt, double a, double b, coupling_t *out){
    int a_idx0=0, b_idx0=0;
    for(int i=0; i<6; i++){
        if(lt->a[i] <= a && lt->a[i+1] >= a) { a_idx0=i; break; }
    }
    for(int i=0; i<6; i++){
        if(lt->b[i] <= b && lt->b[i+1] >= b) { b_idx0=i; break; }
    }
    
    int a_idx1 = fmin(a_idx0+1, 6);
    int b_idx1 = fmin(b_idx0+1, 6);
    
    double fa = (a - lt->a[a_idx0]) / (lt->a[a_idx1] - lt->a[a_idx0]);
    double fb = (b - lt->b[b_idx0]) / (lt->b[b_idx1] - lt->b[b_idx0]);
    
    for(int k=2; k<=4; k++){
        double g00 = lt->g[a_idx0][b_idx0].g[k];
        double g10 = lt->g[a_idx1][b_idx0].g[k];
        double g01 = lt->g[a_idx0][b_idx1].g[k];
        double g11 = lt->g[a_idx1][b_idx1].g[k];
        
        double g0 = g00*(1-fa) + g10*fa;
        double g1 = g01*(1-fa) + g11*fa;
        out->g[k] = g0*(1-fb) + g1*fb;
    }
}

int main(){
    printf("═══ PRODUCT COMPUTATION ACCURACY TEST ═══\n\n");
    
    /* Build lookup table */
    lookup_table_t lt;
    double a_vals[] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5};
    double b_vals[] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5};
    
    for(int i=0; i<7; i++){
        lt.a[i] = a_vals[i];
        lt.b[i] = b_vals[i];
        for(int j=0; j<7; j++){
            double a = a_vals[i], b = b_vals[j];
            double s2 = measure_signal(linear_ramp_omega, 2, a, b, 2);
            double s3 = measure_signal(linear_ramp_omega, 2, a, b, 3);
            double s4 = measure_signal(linear_ramp_omega, 2, a, b, 4);
            
            lt.g[i][j].g[2] = (fabs(a*a)>0.001) ? s2/(a*a) : 0;
            lt.g[i][j].g[3] = (fabs(a*b)>0.001) ? s3/(a*b) : 0;
            lt.g[i][j].g[4] = (fabs(b*b)>0.001) ? s4/(b*b) : 0;
        }
    }
    
    printf("Lookup table built: 7×7 = 49 entries\n\n");
    
    /* Test product computation accuracy */
    printf("═══ TEST: Compute a², a·b, b² using lookup table ═══\n\n");
    
    int num_tests = 20;
    double total_err_a2 = 0, total_err_ab = 0, total_err_b2 = 0;
    
    printf("Test | a    | b    | a²_err | a·b_err | b²_err\n");
    printf("-----|------|------|--------|---------|-------\n");
    
    for(int t=0; t<num_tests; t++){
        double a = 0.5 + 3.5*rand()/RAND_MAX;
        double b = 0.5 + 3.5*rand()/RAND_MAX;
        
        /* Get interpolated coupling coefficients */
        coupling_t g;
        interpolate_g(&lt, a, b, &g);
        
        /* Measure signals */
        double s2 = measure_signal(linear_ramp_omega, 2, a, b, 2);
        double s3 = measure_signal(linear_ramp_omega, 2, a, b, 3);
        double s4 = measure_signal(linear_ramp_omega, 2, a, b, 4);
        
        /* Compute products using g(p,q) */
        double a2_computed = s2 / g.g[2];
        double ab_computed = s3 / g.g[3];
        double b2_computed = s4 / g.g[4];
        
        /* True values */
        double a2_true = a * a;
        double ab_true = a * b;
        double b2_true = b * b;
        
        /* Absolute errors */
        double err_a2 = fabs(a2_computed - a2_true);
        double err_ab = fabs(ab_computed - ab_true);
        double err_b2 = fabs(b2_computed - b2_true);
        
        /* Relative errors (%) */
        double rel_err_a2 = err_a2 / a2_true * 100;
        double rel_err_ab = err_ab / ab_true * 100;
        double rel_err_b2 = err_b2 / b2_true * 100;
        
        total_err_a2 += rel_err_a2;
        total_err_ab += rel_err_ab;
        total_err_b2 += rel_err_b2;
        
        printf("%4d  | %.2f | %.2f | %5.1f%% | %6.1f%% | %5.1f%%\n",
               t+1, a, b, rel_err_a2, rel_err_ab, rel_err_b2);
    }
    
    printf("\nAVERAGE RELATIVE ERRORS:\n");
    printf("  a²:  %.1f%%\n", total_err_a2/num_tests);
    printf("  a·b: %.1f%%\n", total_err_ab/num_tests);
    printf("  b²:  %.1f%%\n", total_err_b2/num_tests);
    
    double avg_total = (total_err_a2 + total_err_ab + total_err_b2) / (3*num_tests);
    printf("\nTOTAL AVERAGE: %.1f%%\n", avg_total);
    
    if(avg_total < 10){
        printf("\n✓✓✓ EXCELLENT: Product computation works!\n");
        printf("✓✓✓ Lookup-table calibration is VALID\n");
    } else if(avg_total < 30){
        printf("\n✓ GOOD: Product computation works with moderate accuracy\n");
    } else if(avg_total < 50){
        printf("\n△ MARGINAL: Needs improvement\n");
    } else {
        printf("\n✗ FAIL: Product computation not accurate enough\n");
    }
    
    return 0;
}