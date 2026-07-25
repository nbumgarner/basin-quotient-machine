/* ===========================================================================
 * filter_characterization.c — Characterize Lens-Enhanced Coupling as Filter
 *
 *   Tests which harmonics pass/block for different lens profiles.
 *   Creates lens-programmable filter catalog.
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

/* Encode single harmonic p with amplitude A */
static void encode_single(ring_t *r,int p,double A){
    memset(r->th,0,N*sizeof(double));
    memset(r->om,0,N*sizeof(double));
    for(int j=0;j<N;j++){
        r->th[j]=2.0*A*cos(2.0*M_PI*p*j/N);
    }
}

/* Apply lens profile */
static void apply_lens(ring_t *r,const double *profile){
    memcpy(r->om,profile,N*sizeof(double));
}

/* Measure transmission spectrum for given lens */
static void measure_spectrum(ring_t *r,int p,double A,const double *lens,int chunks,double *transmission){
    for(int k=1;k<=NH;k++){
        encode_single(r,p,A);
        apply_lens(r,lens);
        settle(r,chunks);
        transmission[k]=ck_mag(r,k);
    }
}

/* Baseline spectrum (no lens) */
static void measure_baseline_spectrum(ring_t *r,int p,double A,int chunks,double *baseline){
    for(int k=1;k<=NH;k++){
        encode_single(r,p,A);
        memset(r->om,0,N*sizeof(double));
        settle(r,chunks);
        baseline[k]=ck_mag(r,k);
    }
}

int main(){
    printf("═══ Filter Characterization Experiment ═══\n");
    printf("Testing harmonic transmission for different lens profiles\n\n");
    
    int chunks = 2;
    double A = 2.0;  /* input amplitude */
    int p = 1;       /* input harmonic */
    
    /* Define lens profiles */
    struct {const char *name; double profile[N];} profiles[] = {
        {"no_lens", {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}},
        {"uniform_+0.1", {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}},
        {"uniform_-0.1", {-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1}},
        {"site0_+0.5", {0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}},
        {"site0_-0.5", {-0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}},
        {"site4_+0.5", {0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0}},
        {"cos_k1", {1.0,0.8,0.0,-0.8,-1.0,-0.8,0.0,0.8,1.0,0.8,0.0,-0.8,-1.0,-0.8,0.0,0.8}},
        {"cos_k2", {1.0,0.0,-1.0,0.0,1.0,0.0,-1.0,0.0,1.0,0.0,-1.0,0.0,1.0,0.0,-1.0,0.0}},
        {"alternating", {0.5,-0.5,0.5,-0.5,0.5,-0.5,0.5,-0.5,0.5,-0.5,0.5,-0.5,0.5,-0.5,0.5,-0.5}},
        {"gauss_center", {0.0,0.0,0.0,0.0,0.0,0.5,1.0,0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}},
        {"gauss_edge", {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,1.0,0.5,0.0,0.0,0.0,0.0,0.0}}
    };
    
    int num_profiles = sizeof(profiles)/sizeof(profiles[0]);
    
    printf("Input: harmonic p=%d, amplitude A=%.1f\n", p, A);
    printf("Testing %d lens profiles\n\n", num_profiles);
    
    ring_t r;
    
    for(int prof_idx=0;prof_idx<num_profiles;prof_idx++){
        printf("LENS: %s\n", profiles[prof_idx].name);
        
        double transmission[NH+1];
        double baseline[NH+1];
        
        /* Measure with lens */
        measure_spectrum(&r, p, A, profiles[prof_idx].profile, chunks, transmission);
        
        /* Measure baseline */
        measure_baseline_spectrum(&r, p, A, chunks, baseline);
        
        /* Print spectrum */
        printf("  k | transmission | baseline | enhancement\n");
        printf("  --+-------------+----------+------------\n");
        for(int k=1;k<=NH;k++){
            double enh = baseline[k]>1e-9 ? transmission[k]/baseline[k] : 0.0;
            printf("  %d |   %6.4f   |  %6.4f  |   %5.2fx\n", k, transmission[k], baseline[k], enh);
        }
        printf("\n");
    }
    
    printf("═══ SUMMARY ═══\n");
    printf("Look for lenses that:\n");
    printf("1. Enhance specific harmonics (high enhancement factor)\n");
    printf("2. Block specific harmonics (low transmission)\n");
    printf("3. Create selective passbands (high for some k, low for others)\n");
    printf("\nThese characterize the lens-programmable filter capabilities.\n");
    
    return 0;
}