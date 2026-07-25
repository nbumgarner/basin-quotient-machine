/* ===========================================================================
 * proactive_experiment.c — Test Different Configurations Proactively
 *
 *   Running experiments with varied parameters without waiting for direction.
 *   Testing: different amplitudes, lens strengths, input harmonics, ring sizes.
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
    printf("═══ PROACTIVE EXPERIMENT — Different Configurations ═══\n\n");
    
    ring_t r;
    
    /* Test 1: Different input amplitudes with site0_+0.5 lens */
    printf("TEST 1: Input amplitude scaling with site0_+0.5 lens\n");
    printf("Input harmonic p=1, measure |c₁| for different amplitudes\n");
    double lens_site0[] = {0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    
    for(double A = 0.5; A <= 3.0; A += 0.5){
        encode_single(&r, 1, A);
        apply_lens(&r, lens_site0);
        settle(&r, 2);
        double c1 = ck_mag(&r, 1);
        printf("  A=%.1f: |c₁|=%.4f, ratio=%.2f\n", A, c1, c1/A);
    }
    printf("  → If ratio constant, linear response. If varies, nonlinear.\n\n");
    
    /* Test 2: Different lens strengths */
    printf("TEST 2: Lens strength scaling with site0 lens\n");
    printf("Input A=2.0, p=1, vary lens strength\n");
    for(double strength = 0.1; strength <= 1.0; strength += 0.2){
        double lens[16] = {0};
        lens[0] = strength;
        encode_single(&r, 1, 2.0);
        apply_lens(&r, lens);
        settle(&r, 2);
        double c1 = ck_mag(&r, 1);
        printf("  strength=%.1f: |c₁|=%.4f, ratio=%.2f\n", strength, c1, c1/strength);
    }
    printf("  → If ratio constant, linear with lens. If varies, nonlinear.\n\n");
    
    /* Test 3: Different input harmonics with same lens */
    printf("TEST 3: Input harmonic scaling with site0_+0.5 lens\n");
    printf("Input A=2.0, lens=site0_+0.5, vary input harmonic p\n");
    for(int p = 1; p <= 4; p++){
        encode_single(&r, p, 2.0);
        apply_lens(&r, lens_site0);
        settle(&r, 2);
        double c1 = ck_mag(&r, 1);
        double c2 = ck_mag(&r, 2);
        double c3 = ck_mag(&r, 3);
        printf("  p=%d: |c₁|=%.4f, |c₂|=%.4f, |c₃|=%.4f\n", p, c1, c2, c3);
    }
    printf("  → See how input harmonic affects output spectrum.\n\n");
    
    /* Test 4: Different lens profiles with same input */
    printf("TEST 4: Lens profile comparison with input p=1, A=2.0\n");
    struct {const char *name; double profile[16];} lenses[] = {
        {"site0_+0.5", {0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}},
        {"site4_+0.5", {0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0}},
        {"site8_+0.5", {0,0,0,0,0,0,0,0,0.5,0,0,0,0,0,0,0}},
        {"uniform_+0.3", {0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3}}
    };
    
    for(int i=0;i<4;i++){
        encode_single(&r, 1, 2.0);
        apply_lens(&r, lenses[i].profile);
        settle(&r, 2);
        double c1 = ck_mag(&r, 1);
        double c2 = ck_mag(&r, 2);
        double c3 = ck_mag(&r, 3);
        printf("  %s: |c₁|=%.4f, |c₂|=%.4f, |c₃|=%.4f\n", lenses[i].name, c1, c2, c3);
    }
    printf("\n");
    
    /* Test 5: Multiple harmonics input (superposition) */
    printf("TEST 5: Superposition of harmonics p=1 and p=2\n");
    printf("Encode a=2.0 into p=1, b=1.0 into p=2, measure spectrum\n");
    memset(r.th,0,sizeof(r.th));
    memset(r.om,0,sizeof(r.om));
    for(int j=0;j<N;j++){
        r.th[j] = 2.0*2.0*cos(2.0*M_PI*1*j/N) + 2.0*1.0*cos(2.0*M_PI*2*j/N);
    }
    apply_lens(&r, lens_site0);
    settle(&r, 2);
    printf("  Superposition input: a=2.0 (p=1), b=1.0 (p=2)\n");
    for(int k=1;k<=4;k++){
        printf("  |c_%d|=%.4f\n", k, ck_mag(&r, k));
    }
    printf("  → Compare to single harmonic inputs to see interaction.\n\n");
    
    /* Test 6: Vary settle chunks */
    printf("TEST 6: Settle chunk variation with site0_+0.5 lens\n");
    printf("Input A=2.0, p=1, vary chunks\n");
    for(int chunks = 1; chunks <= 4; chunks++){
        encode_single(&r, 1, 2.0);
        apply_lens(&r, lens_site0);
        settle(&r, chunks);
        double c1 = ck_mag(&r, 1);
        double c2 = ck_mag(&r, 2);
        printf("  chunks=%d: |c₁|=%.4f, |c₂|=%.4f, ratio=%.2f\n", chunks, c1, c2, c1/c2);
    }
    printf("  → See convergence behavior.\n\n");
    
    printf("═══ CONCLUSIONS ═══\n");
    printf("Running these experiments reveals:\n");
    printf("1. Linear vs nonlinear response to amplitude\n");
    printf("2. Linear vs nonlinear response to lens strength\n");
    printf("3. Input harmonic dependence\n");
    printf("4. Lens profile selectivity\n");
    printf("5. Superposition behavior\n");
    printf("6. Convergence with settle time\n");
    
    return 0;
}