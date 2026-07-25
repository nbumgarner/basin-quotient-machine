/* ===========================================================================
 * phase1_filter_test.c — Test lens-programmable filter
 *
 *   Tests if lens amplifies specific harmonics (c2, c3, c4)
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
    printf("═══ PHASE 1: LENS-PROGRAMMABLE FILTER TEST ═══\n\n");
    
    ring_t r;
    int chunks = 2;
    
    /* Test 1: Encode a=2.0 into p=1, measure |c2| with/without lens */
    printf("TEST 1: Encode a=2.0 into p=1, measure |c2|\n");
    encode_single(&r, 1, 2.0);
    
    /* Without lens */
    memset(r.om, 0, N*sizeof(double));
    settle(&r, chunks);
    double c2_no_lens = ck_mag(&r, 2);
    
    /* With lens */
    apply_lens(&r, 0.5);
    settle(&r, chunks);
    double c2_with_lens = ck_mag(&r, 2);
    
    printf("  |c2| without lens: %.4f\n", c2_no_lens);
    printf("  |c2| with lens:    %.4f\n", c2_with_lens);
    printf("  Enhancement:       %.2fx\n", c2_with_lens/c2_no_lens);
    printf("  Result: %s\n\n", (c2_with_lens > c2_no_lens) ? "AMPLIFIED" : "NOT AMPLIFIED");
    
    /* Test 2: Encode a=2.0 into p=1, b=1.5 into q=2, measure |c3| */
    printf("TEST 2: Encode a=2.0 (p=1), b=1.5 (q=2), measure |c3|\n");
    encode_pair(&r, 1, 2.0, 2, 1.5);
    
    /* Without lens */
    memset(r.om, 0, N*sizeof(double));
    settle(&r, chunks);
    double c3_no_lens = ck_mag(&r, 3);
    
    /* With lens */
    apply_lens(&r, 0.5);
    settle(&r, chunks);
    double c3_with_lens = ck_mag(&r, 3);
    
    printf("  |c3| without lens: %.4f\n", c3_no_lens);
    printf("  |c3| with lens:    %.4f\n", c3_with_lens);
    printf("  Enhancement:       %.2fx\n", c3_with_lens/c3_no_lens);
    printf("  Result: %s\n\n", (c3_with_lens > c3_no_lens) ? "AMPLIFIED" : "NOT AMPLIFIED");
    
    /* Test 3: Encode b=1.5 into p=2, measure |c4| */
    printf("TEST 3: Encode b=1.5 into p=2, measure |c4|\n");
    encode_single(&r, 2, 1.5);
    
    /* Without lens */
    memset(r.om, 0, N*sizeof(double));
    settle(&r, chunks);
    double c4_no_lens = ck_mag(&r, 4);
    
    /* With lens */
    apply_lens(&r, 0.5);
    settle(&r, chunks);
    double c4_with_lens = ck_mag(&r, 4);
    
    printf("  |c4| without lens: %.4f\n", c4_no_lens);
    printf("  |c4| with lens:    %.4f\n", c4_with_lens);
    printf("  Enhancement:       %.2fx\n", c4_with_lens/c4_no_lens);
    printf("  Result: %s\n\n", (c4_with_lens > c4_no_lens) ? "AMPLIFIED" : "NOT AMPLIFIED");
    
    printf("═══ CONCLUSION ═══\n");
    printf("If any harmonic is AMPLIFIED, the lens acts as a FILTER.\n");
    printf("This is the immediate utility of the fixed coupling.\n");
    
    return 0;
}
