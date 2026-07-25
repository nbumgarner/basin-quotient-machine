/* ===========================================================================
 * phase2_product_test.c — Test if we can compute products in harmonic space
 *
 *   Tests if c3 scales with a*b when encoding a into p=1, b into q=2
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

static void encode_pair(ring_t *r,int p,double A,int q,double B){
    memset(r->th,0,N*sizeof(double));
    memset(r->om,0,N*sizeof(double));
    for(int j=0;j<N;j++){
        r->th[j]=2.0*A*cos(2.0*M_PI*p*j/N)+2.0*B*cos(2.0*M_PI*q*j/N);
    }
}

static void apply_lens(ring_t *r,double sign){
    memset(r->om,0,N*sizeof(double));
    r->om[0]=sign*0.5;
}

static void settle(ring_t *r,int chunks){
    int steps=(int)(T_CHUNK/DT);
    for(int c=0;c<chunks;c++){
        for(int s=0;s<steps;s++) rk4(r);
    }
}

int main(){
    printf("═══ PHASE 2: PRODUCT COMPUTATION TEST ═══\n\n");
    printf("Testing if c3 scales with a*b\n\n");
    
    ring_t r;
    int chunks = 2;
    
    struct { double a, b; const char *desc; } test_cases[] = {
        {1.0, 1.0, "a=1.0, b=1.0 (a*b=1.0)"},
        {2.0, 1.5, "a=2.0, b=1.5 (a*b=3.0)"},
        {3.0, 2.0, "a=3.0, b=2.0 (a*b=6.0)"},
        {4.0, 2.5, "a=4.0, b=2.5 (a*b=10.0)"},
        {5.0, 3.0, "a=5.0, b=3.0 (a*b=15.0)"},
        {0.0, 0.0, 0}
    };
    
    printf("Test: a into p=1, b into q=2, measure |c3| with lens\n\n");
    printf("a\tb\ta*b\t|c3|\tRatio c3/(a*b)\n");
    printf("-\t-\t-\t-----\t-------------\n");
    
    int idx = 0;
    while(test_cases[idx].a != 0.0){
        double a = test_cases[idx].a;
        double b = test_cases[idx].b;
        double product = a * b;
        
        encode_pair(&r, 1, a, 2, b);
        apply_lens(&r, 0.5);
        settle(&r, chunks);
        
        double c3 = ck_mag(&r, 3);
        double ratio = (product > 1e-9) ? (c3 / product) : 0.0;
        
        printf("%.1f\t%.1f\t%.1f\t%.4f\t%.4f\n", a, b, product, c3, ratio);
        
        idx++;
    }
    
    printf("\n═══ CONCLUSION ═══\n");
    printf("If ratio c3/(a*b) is CONSTANT, then c3 scales with a*b.\n");
    printf("This means we CAN compute products in harmonic space!\n");
    
    return 0;
}
