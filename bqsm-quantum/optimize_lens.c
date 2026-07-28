/* ===========================================================================
 * optimize_lens.c — Optimize lens profile for maximum enhancement
 *
 *   Test different strengths and positions to find optimal configuration
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

static void settle(ring_t *r,int chunks){
    int steps=(int)(T_CHUNK/DT);
    for(int c=0;c<chunks;c++){
        for(int s=0;s<steps;s++) rk4(r);
    }
}

int main(){
    printf("═══ OPTIMIZATION: Gauss lens strength ═══\n\n");
    
    ring_t r;
    int chunks = 2;
    
    printf("Test: a=2.0 into p=1, measure |c2| with different gauss strengths\n\n");
    printf("Strength\t|c2|\tEnhancement\n");
    printf("-\t-\t-\n");
    
    for(int i=1; i<=10; i++){
        double strength = i * 0.1;
        
        encode_single(&r, 1, 2.0);
        
        /* Apply gauss lens */
        memset(r.om, 0, N*sizeof(double));
        for(int j=0;j<N;j++){
            double x = (double)(j - N/2) / (N/4);
            r.om[j] = strength * exp(-x*x);
        }
        
        settle(&r, chunks);
        double c2 = ck_mag(&r, 2);
        
        printf("%.2f\t\t%.4f\n", strength, c2);
    }
    
    printf("\n═══ OPTIMIZATION: Gauss lens position ═══\n\n");
    
    printf("Test: a=2.0 into p=1, measure |c2| with gauss centered at different positions\n\n");
    printf("Center\t|c2|\n");
    printf("-\t-\n");
    
    for(int center=0; center<N; center++){
        encode_single(&r, 1, 2.0);
        
        /* Apply gauss lens centered at different position */
        memset(r.om, 0, N*sizeof(double));
        for(int j=0;j<N;j++){
            double x = (double)(j - center) / (N/4);
            r.om[j] = 1.0 * exp(-x*x);
        }
        
        settle(&r, chunks);
        double c2 = ck_mag(&r, 2);
        
        printf("pos %d\t%.4f\n", center, c2);
    }
    
    printf("\n═══ OPTIMIZATION: Combined lens profiles ═══\n\n");
    
    printf("Test: site0 + gauss combined\n\n");
    printf("site0\tgauss\t|c2|\n");
    printf("-\t-\t-\n");
    
    for(int s=-5; s<=5; s++){
        for(int g=1; g<=5; g++){
            double site0_val = s * 0.1;
            double gauss_str = g * 0.1;
            
            encode_single(&r, 1, 2.0);
            
            /* Apply combined lens */
            memset(r.om, 0, N*sizeof(double));
            r.om[0] = site0_val;
            for(int j=0;j<N;j++){
                double x = (double)(j - N/2) / (N/4);
                r.om[j] += gauss_str * exp(-x*x);
            }
            
            settle(&r, chunks);
            double c2 = ck_mag(&r, 2);
            
            if(c2 > 0.1){
                printf("%.2f\t%.2f\t%.4f\n", site0_val, gauss_str, c2);
            }
        }
    }
    
    printf("\n═══ EXPLORATION: Different input frequencies ═══\n\n");
    
    printf("Test: Different p values, measure |c_{2p}| with gauss lens\n\n");
    printf("p\ta\t|c_{2p}|\n");
    printf("-\t-\t-\n");
    
    for(int p=1; p<=4; p++){
        for(double a=1.0; a<=3.0; a+=1.0){
            encode_single(&r, p, a);
            
            /* Apply gauss lens */
            memset(r.om, 0, N*sizeof(double));
            for(int j=0;j<N;j++){
                double x = (double)(j - N/2) / (N/4);
                r.om[j] = 1.0 * exp(-x*x);
            }
            
            settle(&r, chunks);
            double c_2p = ck_mag(&r, 2*p);
            
            printf("%d\t%.1f\t%.4f\n", p, a, c_2p);
        }
    }
    
    return 0;
}
