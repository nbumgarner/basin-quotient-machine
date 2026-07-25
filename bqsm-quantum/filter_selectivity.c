/* ===========================================================================
 * filter_selectivity.c — Characterize Filter Q Factor and Bandwidth
 *
 *   Tests filter selectivity for different lens profiles.
 *   Measures which harmonics pass/block with different input harmonics.
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

static void encode_single(ring_t *r,int p,double A){
    memset(r->th,0,N*sizeof(double));
    memset(r->om,0,N*sizeof(double));
    for(int j=0;j<N;j++){
        r->th[j]=2.0*A*cos(2.0*M_PI*p*j/N);
    }
}

static void apply_lens(ring_t *r,const double *profile){
    memcpy(r->om,profile,N*sizeof(double));
}

int main(){
    printf("═══ Filter Selectivity Characterization ═══\n");
    printf("Testing different input harmonics with different lens profiles\n\n");
    
    int chunks = 2;
    double A = 2.0;
    
    /* Key lens profiles */
    struct {const char *name; double profile[N];} lenses[] = {
        {"site4_+0.5", {0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0}},
        {"cos_k1", {1.0,0.8,0.0,-0.8,-1.0,-0.8,0.0,0.8,1.0,0.8,0.0,-0.8,-1.0,-0.8,0.0,0.8}},
        {"gauss_center", {0.0,0.0,0.0,0.0,0.0,0.5,1.0,0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}},
        {"site0_+0.5", {0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}},
        {"cos_k2", {1.0,0.0,-1.0,0.0,1.0,0.0,-1.0,0.0,1.0,0.0,-1.0,0.0,1.0,0.0,-1.0,0.0}}
    };
    
    int num_lenses = sizeof(lenses)/sizeof(lenses[0]);
    int input_harmonics[] = {1, 2, 3, 4};
    int num_inputs = sizeof(input_harmonics)/sizeof(input_harmonics[0]);
    
    ring_t r;
    
    for(int l=0;l<num_lenses;l++){
        printf("LENS: %s\n", lenses[l].name);
        printf("Input | ", "");
        for(int k=1;k<=NH;k++) printf("k%-2d ", k);
        printf("\n");
        printf("------| ");
        for(int k=1;k<=NH;k++) printf("---- ", k);
        printf("\n");
        
        for(int i=0;i<num_inputs;i++){
            int p = input_harmonics[i];
            double transmission[NH+1];
            
            /* Measure spectrum */
            for(int k=1;k<=NH;k++){
                encode_single(&r, p, A);
                apply_lens(&r, lenses[l].profile);
                settle(&r, chunks);
                transmission[k]=ck_mag(&r, k);
            }
            
            /* Find peak harmonic */
            double max_val = 0;
            int max_k = 1;
            for(int k=1;k<=NH;k++){
                if(transmission[k] > max_val){
                    max_val = transmission[k];
                    max_k = k;
                }
            }
            
            printf("  p=%d  | ", p);
            for(int k=1;k<=NH;k++){
                if(transmission[k] > 0.01)
                    printf("%5.2f ", transmission[k]);
                else
                    printf("  .  ");
            }
            printf("  (peak: k=%d, val=%.2f)\n", max_k, max_val);
        }
        printf("\n");
    }
    
    printf("═══ ANALYSIS ═══\n");
    printf("Look for patterns:\n");
    printf("1. Which lens passes which input harmonic?\n");
    printf("2. What's the selectivity (peak vs sidebands)?\n");
    printf("3. Can we create harmonic-specific filters?\n");
    
    return 0;
}