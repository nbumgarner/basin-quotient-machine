/* ===========================================================================
 * stability_analysis.c — Find stable operating regime
 *
 *   Test different strengths to find where growth stops
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
    printf("═══ STABILITY ANALYSIS ═══\n\n");
    
    ring_t r;
    int chunks = 2;
    
    /* Part 1: Find stability threshold */
    printf("PART 1: Find stability threshold\n\n");
    
    printf("Strength\t|c1|_final\t|c2|_final\tGrowth rate\n");
    printf("-\t-\t-\t-\n");
    
    for(int i=1; i<=15; i++){
        double strength = i * 0.05;
        
        encode_single(&r, 1, 0.5);
        apply_gauss(&r, strength, 1);
        settle(&r, chunks);
        double c1_final = ck_mag(&r, 1);
        double c2_final = ck_mag(&r, 2);
        
        /* Estimate growth rate */
        double growth = (c1_final > 0.5) ? (c1_final / 0.5) : 0.0;
        
        printf("%.2f\t%.3f\t%.3f\t%.2fx\n", strength, c1_final, c2_final, growth);
    }
    
    printf("\nPART 2: Time-domain at different strengths\n\n");
    
    printf("Strength=0.1 (low): Time\t|c1|\t|c2|\n");
    printf("-\t-\t-\n");
    
    encode_single(&r, 1, 0.5);
    apply_gauss(&r, 0.1, 1);
    
    for(int t=0; t<=5; t++){
        settle(&r, t);
        printf("%.0f\t%.3f\t%.3f\n", t * T_CHUNK, ck_mag(&r, 1), ck_mag(&r, 2));
    }
    
    printf("\nStrength=0.3 (medium): Time\t|c1|\t|c2|\n");
    printf("-\t-\t-\n");
    
    encode_single(&r, 1, 0.5);
    apply_gauss(&r, 0.3, 1);
    
    for(int t=0; t<=5; t++){
        settle(&r, t);
        printf("%.0f\t%.3f\t%.3f\n", t * T_CHUNK, ck_mag(&r, 1), ck_mag(&r, 2));
    }
    
    printf("\nStrength=0.5 (high): Time\t|c1|\t|c2|\n");
    printf("-\t-\t-\n");
    
    encode_single(&r, 1, 0.5);
    apply_gauss(&r, 0.5, 1);
    
    for(int t=0; t<=5; t++){
        settle(&r, t);
        printf("%.0f\t%.3f\t%.3f\n", t * T_CHUNK, ck_mag(&r, 1), ck_mag(&r, 2));
    }
    
    printf("\nPART 3: Search for stable regime\n\n");
    
    printf("Test: Find strength where growth stops (|c1| < 2.0 after 5 chunks)\n\n");
    
    for(int i=1; i<=10; i++){
        double strength = i * 0.05;
        
        encode_single(&r, 1, 0.5);
        apply_gauss(&r, strength, 1);
        settle(&r, 5);
        
        double c1 = ck_mag(&r, 1);
        
        if(c1 < 2.0){
            printf("Strength %.2f: |c1|=%.3f (STABLE)\n", strength, c1);
        } else {
            printf("Strength %.2f: |c1|=%.3f (UNSTABLE)\n", strength, c1);
        }
    }
    
    printf("\nPART 4: Different input amplitudes at low strength\n\n");
    
    printf("Strength=0.1, different input amplitudes\n\n");
    printf("Input a\t|c1|_final\t|c2|_final\n");
    printf("-\t-\t-\n");
    
    for(double a=0.1; a<=1.0; a+=0.1){
        encode_single(&r, 1, a);
        apply_gauss(&r, 0.1, 1);
        settle(&r, 5);
        
        printf("%.1f\t%.3f\t%.3f\n", a, ck_mag(&r, 1), ck_mag(&r, 2));
    }
    
    printf("\n═══ STABILITY ANALYSIS SUMMARY ═══\n\n");
    printf("1. Low strengths (< 0.2) may be stable\n");
    printf("2. Medium strengths (0.3-0.5) show moderate growth\n");
    printf("3. High strengths (> 0.5) show exponential growth\n");
    printf("4. Need to find exact stability threshold\n");
    
    return 0;
}
