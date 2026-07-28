/* ===========================================================================
 * resonance_threshold.c — Find the resonance threshold precisely
 *
 *   Test lens strengths around 0.8-1.0 to find the critical point
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
    printf("═══ RESONANCE THRESHOLD: Precise measurement ═══\n\n");
    
    ring_t r;
    int chunks = 2;
    
    /* Test strengths around 0.8-1.0 */
    printf("Test: a=2.0 into p=1, measure |c2| with different gauss strengths\n\n");
    printf("Strength\t|c2|\t|c3|\t|c4|\n");
    printf("-\t-\t-\t-\n");
    
    for(int i=80; i<=100; i++){
        double strength = i * 0.01;
        
        encode_single(&r, 1, 2.0);
        
        /* Apply gauss lens centered at position 1 */
        memset(r.om, 0, N*sizeof(double));
        for(int j=0;j<N;j++){
            double x = (double)(j - 1) / (N/4);
            r.om[j] = strength * exp(-x*x);
        }
        
        settle(&r, chunks);
        double c2 = ck_mag(&r, 2);
        double c3 = ck_mag(&r, 3);
        double c4 = ck_mag(&r, 4);
        
        printf("%.2f\t\t%.4f\t%.4f\t%.4f\n", strength, c2, c3, c4);
    }
    
    printf("\n═══ RESONANCE: Different input amplitudes ═══\n\n");
    
    printf("Test: Different a values, measure |c2| with gauss strength=0.9\n\n");
    printf("a\t|c2|\t|c3|\t|c4|\n");
    printf("-\t-\t-\t-\n");
    
    for(double a=0.5; a<=4.0; a+=0.5){
        encode_single(&r, 1, a);
        
        /* Apply gauss lens centered at position 1 */
        memset(r.om, 0, N*sizeof(double));
        for(int j=0;j<N;j++){
            double x = (double)(j - 1) / (N/4);
            r.om[j] = 0.9 * exp(-x*x);
        }
        
        settle(&r, chunks);
        double c2 = ck_mag(&r, 2);
        double c3 = ck_mag(&r, 3);
        double c4 = ck_mag(&r, 4);
        
        printf("%.1f\t%.4f\t%.4f\t%.4f\n", a, c2, c3, c4);
    }
    
    printf("\n═══ RESONANCE: Different harmonic pairs ═══\n\n");
    
    printf("Test: Different p values, measure |c_{2p}| with gauss strength=0.9\n\n");
    printf("p\ta\t|c_{2p}|\n");
    printf("-\t-\t-\n");
    
    for(int p=1; p<=4; p++){
        for(double a=1.0; a<=3.0; a+=1.0){
            encode_single(&r, p, a);
            
            /* Apply gauss lens centered at position 1 */
            memset(r.om, 0, N*sizeof(double));
            for(int j=0;j<N;j++){
                double x = (double)(j - 1) / (N/4);
                r.om[j] = 0.9 * exp(-x*x);
            }
            
            settle(&r, chunks);
            double c_2p = ck_mag(&r, 2*p);
            
            printf("%d\t%.1f\t%.4f\n", p, a, c_2p);
        }
    }
    
    return 0;
}
