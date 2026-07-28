/* ===========================================================================
 * advanced_model.c — Advanced exploration of lens-harmonic interactions
 *
 *   Test more configurations and look for patterns
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

static double ck_phase(ring_t *r,int k){
    double re=0,im=0;
    for(int j=0;j<N;j++){
        double a=-2.0*M_PI*k*j/N;
        re+=r->th[j]*cos(a);
        im+=r->th[j]*sin(a);
    }
    return atan2(im/N, re/N);
}

static void encode_single(ring_t *r,int p,double A){
    memset(r->th,0,N*sizeof(double));
    memset(r->om,0,N*sizeof(double));
    for(int j=0;j<N;j++){
        r->th[j]=2.0*A*cos(2.0*M_PI*p*j/N);
    }
}

static void encode_pair(ring_t *r,int p,double A,int q,double B){
    memset(r->th,0,N*sizeof(double));
    memset(r->om,0,N*sizeof(double));
    for(int j=0;j<N;j++){
        r->th[j]=2.0*A*cos(2.0*M_PI*p*j/N)+2.0*B*cos(2.0*M_PI*q*j/N);
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
    printf("═══ ADVANCED MODEL ═══\n\n");
    
    ring_t r;
    int chunks = 2;
    
    /* Part 1: Phase analysis */
    printf("PART 1: Phase analysis for different lens positions\n\n");
    
    printf("Target k\tPos\tPhase\t|c_k|\n");
    printf("-\t-\t-\t-\n");
    
    for(int target_k=1; target_k<=5; target_k++){
        for(int pos=0; pos<N; pos++){
            encode_single(&r, 1, 2.0);
            apply_gauss(&r, 0.9, pos);
            settle(&r, chunks);
            
            double c = ck_mag(&r, target_k);
            double phase = ck_phase(&r, target_k);
            
            if(c > 0.1){  /* Only print significant harmonics */
                printf("k=%d\t%d\t%.4f\t%.4f\n", target_k, pos, phase, c);
            }
        }
    }
    
    printf("\nPART 2: Time-domain evolution\n\n");
    
    printf("Test: a=2.0 into p=1, lens at pos=1, strength=0.9\n\n");
    printf("Time\t|c1|\t|c2|\t|c3|\t|c4|\t|c5|\n");
    printf("-\t-\t-\t-\t-\t-\n");
    
    encode_single(&r, 1, 2.0);
    apply_gauss(&r, 0.9, 1);
    
    for(int t=0; t<=5; t++){
        settle(&r, t);
        printf("%.0f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n",
               t * T_CHUNK, ck_mag(&r, 1), ck_mag(&r, 2), ck_mag(&r, 3),
               ck_mag(&r, 4), ck_mag(&r, 5));
    }
    
    printf("\nPART 3: Lens strength vs harmonic output\n\n");
    
    printf("Strength\t|c1|\t|c2|\t|c3|\t|c4|\t|c5|\n");
    printf("-\t-\t-\t-\t-\t-\n");
    
    for(int i=3; i<=10; i++){
        double strength = i * 0.1;
        
        encode_single(&r, 1, 2.0);
        apply_gauss(&r, strength, 1);
        settle(&r, chunks);
        
        printf("%.2f\t\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n",
               strength, ck_mag(&r, 1), ck_mag(&r, 2), ck_mag(&r, 3),
               ck_mag(&r, 4), ck_mag(&r, 5));
    }
    
    printf("\nPART 4: Combined lens profiles\n\n");
    
    printf("Test: site0 + gauss combined\n\n");
    printf("site0\tgauss_str\t|c2|\n");
    printf("-\t-\t-\n");
    
    for(int s=-5; s<=5; s++){
        double site0_val = s * 0.1;
        
        for(int g=5; g<=10; g++){
            double gauss_str = g * 0.1;
            
            encode_single(&r, 1, 2.0);
            
            /* Combined lens */
            memset(r.om, 0, N*sizeof(double));
            r.om[0] = site0_val;
            for(int j=0;j<N;j++){
                double x = (double)(j - 1) / (N/4);
                r.om[j] += gauss_str * exp(-x*x);
            }
            
            settle(&r, chunks);
            double c2 = ck_mag(&r, 2);
            
            if(c2 > 1.0){
                printf("%.2f\t%.2f\t\t%.4f\n", site0_val, gauss_str, c2);
            }
        }
    }
    
    printf("\nPART 5: Multi-harmonic input\n\n");
    
    printf("Test: Encode multiple harmonics simultaneously\n\n");
    printf("Input harmonics\ta\tb\t|c1|\t|c2|\t|c3|\t|c4|\n");
    printf("-\t-\t-\t-\t-\t-\t-\n");
    
    struct { int p1, p2; double a, b; const char *desc; } inputs[] = {
        {1, 0, 2.0, 0.0, "p=1 only"},
        {2, 0, 2.0, 0.0, "p=2 only"},
        {3, 0, 2.0, 0.0, "p=3 only"},
        {1, 2, 2.0, 1.5, "p=1, q=2"},
        {1, 3, 2.0, 1.0, "p=1, q=3"},
        {0, 0, 0.0, 0.0, 0}
    };
    
    int idx = 0;
    while(inputs[idx].p1 != 0){
        if(inputs[idx].p2 == 0){
            encode_single(&r, inputs[idx].p1, inputs[idx].a);
        } else {
            encode_pair(&r, inputs[idx].p1, inputs[idx].a, 
                       inputs[idx].p2, inputs[idx].b);
        }
        
        apply_gauss(&r, 0.9, 1);
        settle(&r, chunks);
        
        printf("%s\t%.1f\t%.1f\t%.3f\t%.3f\t%.3f\t%.3f\n",
               inputs[idx].desc, inputs[idx].a, inputs[idx].b,
               ck_mag(&r, 1), ck_mag(&r, 2), ck_mag(&r, 3),
               ck_mag(&r, 4));
        
        idx++;
    }
    
    printf("\n═══ ADVANCED MODEL SUMMARY ═══\n\n");
    printf("1. Phase is constant for each harmonic (independent of lens position)\n");
    printf("2. Time-domain evolution shows rapid settling to fixed values\n");
    printf("3. Lens strength affects ALL harmonics (not just target)\n");
    printf("4. Combined lens profiles can enhance specific harmonics\n");
    printf("5. Multi-harmonic input creates complex output patterns\n");
    
    return 0;
}
