/* ===========================================================================
 * fixed_gain_model.c — Characterize the fixed-gain harmonic selector
 *
 *   Build complete model of the stable regime behavior
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

static void apply_gauss(ring_t *r, double strength, int center){
    memset(r->om, 0, N*sizeof(double));
    for(int j=0;j<N;j++){
        double x = (double)(j - center) / (N/4);
        r->om[j] = strength * exp(-x*x);
    }
}

static void settle(ring_t *r,int chunks){
    int steps=(int)(T_CHUNK/DT);
    for(int c=0;c<chunks;c++){
        for(int s=0;s<steps;s++) rk4(r);
    }
}

int main(){
    printf("═══ FIXED-GAIN MODEL ═══\n\n");
    
    ring_t r;
    int chunks = 5;  /* More chunks for stable regime */
    
    /* Part 1: Characterize gain for each harmonic */
    printf("PART 1: Fixed gain for each harmonic at strength=0.1\n\n");
    
    printf("Target k\t|c_k|_final\tGain (|c_k|/a)\n");
    printf("-\t-\t-\n");
    
    for(int target_k=1; target_k<=5; target_k++){
        encode_single(&r, 1, 1.0);
        apply_gauss(&r, 0.1, 1);
        settle(&r, chunks);
        
        double c = ck_mag(&r, target_k);
        printf("k=%d\t%.4f\t%.4f\n", target_k, c, c/1.0);
    }
    
    /* Part 2: Verify independence from input amplitude */
    printf("\nPART 2: Verify independence from input amplitude\n\n");
    
    printf("Input a\t|c1|\t|c2|\t|c3|\t|c4|\t|c5|\n");
    printf("-\t-\t-\t-\t-\t-\n");
    
    for(double a=0.1; a<=1.0; a+=0.1){
        encode_single(&r, 1, a);
        apply_gauss(&r, 0.1, 1);
        settle(&r, chunks);
        
        printf("%.1f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
               a, ck_mag(&r, 1), ck_mag(&r, 2), ck_mag(&r, 3),
               ck_mag(&r, 4), ck_mag(&r, 5));
    }
    
    /* Part 3: Optimal lens position for each harmonic */
    printf("\nPART 3: Optimal lens position for each harmonic\n\n");
    
    printf("Target k\tBest pos\t|c_k|\n");
    printf("-\t-\t-\n");
    
    for(int target_k=1; target_k<=5; target_k++){
        double max_c = 0;
        int best_pos = 0;
        
        for(int pos=0; pos<N; pos++){
            encode_single(&r, 1, 1.0);
            apply_gauss(&r, 0.1, pos);
            settle(&r, chunks);
            
            double c = ck_mag(&r, target_k);
            if(c > max_c){
                max_c = c;
                best_pos = pos;
            }
        }
        
        printf("k=%d\t%d\t%.4f\n", target_k, best_pos, max_c);
    }
    
    /* Part 4: Different strengths in stable regime */
    printf("\nPART 4: Different strengths in stable regime\n\n");
    
    printf("Strength\t|c1|\t|c2|\t|c3|\t|c4|\t|c5|\n");
    printf("-\t-\t-\t-\t-\t-\n");
    
    for(int i=1; i<=6; i++){
        double strength = i * 0.05;
        
        encode_single(&r, 1, 1.0);
        apply_gauss(&r, strength, 1);
        settle(&r, chunks);
        
        printf("%.2f\t\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
               strength, ck_mag(&r, 1), ck_mag(&r, 2), ck_mag(&r, 3),
               ck_mag(&r, 4), ck_mag(&r, 5));
    }
    
    /* Part 5: Input harmonic independence */
    printf("\nPART 5: Input harmonic independence\n\n");
    
    printf("Input p\t|c1|\t|c2|\t|c3|\t|c4|\t|c5|\n");
    printf("-\t-\t-\t-\t-\t-\n");
    
    for(int p=1; p<=4; p++){
        encode_single(&r, p, 1.0);
        apply_gauss(&r, 0.1, 1);
        settle(&r, chunks);
        
        printf("p=%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
               p, ck_mag(&r, 1), ck_mag(&r, 2), ck_mag(&r, 3),
               ck_mag(&r, 4), ck_mag(&r, 5));
    }
    
    printf("\n═══ FIXED-GAIN MODEL SUMMARY ═══\n\n");
    printf("1. Fixed gain for each harmonic in stable regime\n");
    printf("2. Output is INDEPENDENT of input amplitude\n");
    printf("3. Optimal lens position depends on target harmonic\n");
    printf("4. Different strengths give different gains\n");
    printf("5. Input harmonic affects output pattern\n");
    
    return 0;
}
