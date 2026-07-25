/* ===========================================================================
 * time_domain_analysis.c — Analyze time-domain behavior of coupling
 *
 *   Measures phase and amplitude at different times to detect resonant behavior
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

static double ck_re(ring_t *r,int k){
    double re=0;
    for(int j=0;j<N;j++){
        double a=-2.0*M_PI*k*j/N;
        re+=r->th[j]*cos(a);
    }
    return re/N;
}

static double ck_im(ring_t *r,int k){
    double im=0;
    for(int j=0;j<N;j++){
        double a=-2.0*M_PI*k*j/N;
        im+=r->th[j]*sin(a);
    }
    return im/N;
}

static double ck_phase(ring_t *r,int k){
    return atan2(ck_im(r,k), ck_re(r,k));
}

static void encode_pair(ring_t *r,int p,double A,int q,double B){
    memset(r->th,0,N*sizeof(double));
    memset(r->om,0,N*sizeof(double));
    for(int j=0;j<N;j++){
        r->th[j]=2.0*A*cos(2.0*M_PI*p*j/N)+2.0*B*cos(2.0*M_PI*q*j/N);
    }
}

static void apply_site0_lens(ring_t *r,double sign){
    memset(r->om,0,N*sizeof(double));
    r->om[0]=sign*0.5;
}

static void rk4_steps(ring_t *r, int steps){
    for(int s=0;s<steps;s++) rk4(r);
}

int main(){
    printf("═══ Time-Domain Analysis ═══\n\n");
    
    ring_t r;
    int steps_per_chunk = (int)(T_CHUNK/DT);
    
    /* Test: a=2.0, b=1.5, p=1, q=2 */
    printf("Test: a=2.0, b=1.5, p=1, q=2, lens=+0.5\n\n");
    
    encode_pair(&r, 1, 2.0, 2, 1.5);
    apply_site0_lens(&r, 0.5);
    
    /* Measure at different times */
    printf("Time\t\t|c2| mag\t\tphase\t\t|c3| mag\t\tphase\n");
    printf("------\t\t------\t\t-----\t\t------\t\t-----\n");
    
    for(int chunk=0; chunk<=4; chunk++){
        rk4_steps(&r, chunk * steps_per_chunk);
        
        double mag2 = sqrt(ck_re(&r,2)*ck_re(&r,2) + ck_im(&r,2)*ck_im(&r,2));
        double phase2 = ck_phase(&r, 2);
        double mag3 = sqrt(ck_re(&r,3)*ck_re(&r,3) + ck_im(&r,3)*ck_im(&r,3));
        double phase3 = ck_phase(&r, 3);
        
        printf("%.1f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n", 
               chunk * T_CHUNK, mag2, phase2, mag3, phase3);
    }
    
    printf("\n=== CONCLUSION ===\n");
    printf("If phase is CONSTANT, coupling is magnitude-based (fixed coefficient).\n");
    printf("If phase VARIES, coupling is phase-based (resonant phenomenon).\n");
    
    return 0;
}
