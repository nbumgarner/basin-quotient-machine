/* ===========================================================================
 * lookup_table_calibration.c — Option 1: Lookup-Table Calibration
 *
 *   Build empirical map: (a,b) → (g(1,1), g(1,2), g(2,2))
 *   Use LINEAR_RAMP lens, hybrid encoding, baseline subtraction
 *
 *   Grid: a ∈ [0.5, 4.0], b ∈ [0.5, 4.0], step 0.5
 *   Result: 7 × 7 = 49 entries per coupling coefficient
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

/* LINEAR_RAMP lens */
static double linear_ramp_omega[N] = {
    0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14,
    0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30
};

/* Baseline structure */
typedef struct {
    double decay[NH+1];
} baseline_t;

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

/* Run ALU with baseline subtraction */
typedef struct {
    double g[9];  /* g[2]=g(1,1), g[3]=g(1,2), g[4]=g(2,2), ... */
} coupling_t;

static void run_alu_get_g(double lens_omega[], int chunks,
                         double a, double b, coupling_t *out){
    baseline_t baselines[NH+1];
    measure_baseline(lens_omega, chunks, baselines);
    
    ring_t r;
    memset(&r, 0, sizeof(r));
    for(int i=0;i<N;i++) r.om[i] = lens_omega[i];
    
    /* Hybrid encoding: a→c₁, b→c₂ */
    for(int j=0;j<N;j++){
        r.th[j] = 2.0*a*cos(2.0*M_PI*1*j/N) 
                + 2.0*b*cos(2.0*M_PI*2*j/N);
    }
    
    settle(&r, chunks);
    
    /* Measure and subtract baselines */
    for(int k=1; k<=NH; k++){
        double raw = ck_mag(&r, k);
        double baseline_sum = baselines[1].decay[k] + baselines[2].decay[k];
        double signal = raw - baseline_sum;
        
        /* Compute coupling coefficients */
        if(k==2) out->g[k] = (fabs(a*a)>0.001) ? signal/(a*a) : 0;
        if(k==3) out->g[k] = (fabs(a*b)>0.001) ? signal/(a*b) : 0;
        if(k==4) out->g[k] = (fabs(b*b)>0.001) ? signal/(b*b) : 0;
    }
}

/* Lookup table: 7 × 7 grid (a=0.5..4.0, b=0.5..4.0, step=0.5) */
#define GRID_SIZE 7
typedef struct {
    double a[GRID_SIZE];
    double b[GRID_SIZE];
    coupling_t g[GRID_SIZE][GRID_SIZE];  /* g[a_idx][b_idx] */
} lookup_table_t;

static void build_lookup_table(lookup_table_t *lt){
    printf("Building lookup table: a ∈ [0.5, 4.0], b ∈ [0.5, 4.0], step=0.5\n\n");
    
    double a_start=0.5, a_end=4.0, a_step=0.5;
    double b_start=0.5, b_end=4.0, b_step=0.5;
    
    int a_idx=0, b_idx=0;
    for(double a=a_start; a<=a_end+0.001; a+=a_step, a_idx++){
        for(double b=b_start; b<=b_end+0.001; b+=b_step, b_idx++){
            if(a_idx>=GRID_SIZE || b_idx>=GRID_SIZE) break;
            
            lt->a[a_idx] = a;
            lt->b[b_idx] = b;
            
            run_alu_get_g(linear_ramp_omega, 2, a, b, &lt->g[a_idx][b_idx]);
            
            printf("  (%.1f, %.1f): g(1,1)=%.4f, g(1,2)=%.4f, g(2,2)=%.4f\n",
                   a, b, lt->g[a_idx][b_idx].g[2], 
                   lt->g[a_idx][b_idx].g[3], lt->g[a_idx][b_idx].g[4]);
        }
        b_idx=0;
    }
    
    printf("\nLookup table built: %d × %d = %d entries\n", GRID_SIZE, GRID_SIZE, GRID_SIZE*GRID_SIZE);
}

/* Bilinear interpolation */
static void interpolate_g(lookup_table_t *lt, double a, double b, coupling_t *out){
    /* Find surrounding grid points */
    int a_idx0=0, b_idx0=0;
    for(int i=0; i<GRID_SIZE-1; i++){
        if(lt->a[i] <= a && lt->a[i+1] >= a){ a_idx0=i; break; }
    }
    for(int i=0; i<GRID_SIZE-1; i++){
        if(lt->b[i] <= b && lt->b[i+1] >= b){ b_idx0=i; break; }
    }
    
    int a_idx1 = a_idx0+1;
    int b_idx1 = b_idx0+1;
    
    /* Clamp to valid range */
    a_idx0 = fmax(0, fmin(GRID_SIZE-2, a_idx0));
    a_idx1 = fmax(1, fmin(GRID_SIZE-1, a_idx1));
    b_idx0 = fmax(0, fmin(GRID_SIZE-2, b_idx0));
    b_idx1 = fmax(1, fmin(GRID_SIZE-1, b_idx1));
    
    /* Bilinear interpolation for each coupling coefficient */
    double fa = (a - lt->a[a_idx0]) / (lt->a[a_idx1] - lt->a[a_idx0]);
    double fb = (b - lt->b[b_idx0]) / (lt->b[b_idx1] - lt->b[b_idx0]);
    
    for(int k=2; k<=4; k++){
        double g00 = lt->g[a_idx0][b_idx0].g[k];
        double g10 = lt->g[a_idx1][b_idx0].g[k];
        double g01 = lt->g[a_idx0][b_idx1].g[k];
        double g11 = lt->g[a_idx1][b_idx1].g[k];
        
        /* Bilinear interpolation */
        double g0 = g00*(1-fa) + g10*fa;
        double g1 = g01*(1-fa) + g11*fa;
        out->g[k] = g0*(1-fb) + g1*fb;
    }
}

int main(){
    printf("═══ LOOKUP-TABLE CALIBRATION — OPTION 1 ═══\n\n");
    
    lookup_table_t lt;
    build_lookup_table(&lt);
    
    /* Test interpolation */
    printf("\n═══ TEST INTERPOLATION ═══\n");
    double test_a = 2.25, test_b = 1.75;
    coupling_t interpolated;
    interpolate_g(&lt, test_a, test_b, &interpolated);
    
    printf("Input: a=%.2f, b=%.2f\n", test_a, test_b);
    printf("Interpolated: g(1,1)=%.4f, g(1,2)=%.4f, g(2,2)=%.4f\n",
           interpolated.g[2], interpolated.g[3], interpolated.g[4]);
    
    /* Compare to direct measurement */
    coupling_t direct;
    run_alu_get_g(linear_ramp_omega, 2, test_a, test_b, &direct);
    
    printf("Direct:     g(1,1)=%.4f, g(1,2)=%.4f, g(2,2)=%.4f\n",
           direct.g[2], direct.g[3], direct.g[4]);
    
    printf("Error:      g(1,1)=%.1f%%, g(1,2)=%.1f%%, g(2,2)=%.1f%%\n",
           fabs(interpolated.g[2]-direct.g[2])/fabs(direct.g[2])*100,
           fabs(interpolated.g[3]-direct.g[3])/fabs(direct.g[3])*100,
           fabs(interpolated.g[4]-direct.g[4])/fabs(direct.g[4])*100);
    
    printf("\nLOOKUP TABLE (g(1,2) values):\n");
    printf("      b=0.5  b=1.0  b=1.5  b=2.0  b=2.5  b=3.0  b=3.5  b=4.0\n");
    for(int i=0; i<GRID_SIZE; i++){
        printf("a=%.1f  ", lt.a[i]);
        for(int j=0; j<GRID_SIZE; j++){
            printf("%6.3f ", lt.g[i][j].g[3]);
        }
        printf("\n");
    }
    
    printf("\nCONCLUSION:\n");
    printf("  Lookup-table calibration is IMPLEMENTED.\n");
    printf("  Interpolation accuracy: %.1f%% average error.\n",
           (fabs(interpolated.g[2]-direct.g[2])/fabs(direct.g[2]) +
            fabs(interpolated.g[3]-direct.g[3])/fabs(direct.g[3]) +
            fabs(interpolated.g[4]-direct.g[4])/fabs(direct.g[4])) / 3.0 * 100);
    
    return 0;
}