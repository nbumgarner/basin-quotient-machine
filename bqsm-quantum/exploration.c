/* ===========================================================================
 * exploration.c — Explore different lens profiles and configurations
 *
 *   Test various lens profiles to find optimal filter configurations
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

/* Different lens profiles */
static void apply_site0_lens(ring_t *r,double sign){
    memset(r->om,0,N*sizeof(double));
    r->om[0]=sign*0.5;
}

static void apply_gauss_lens(ring_t *r,double strength){
    memset(r->om,0,N*sizeof(double));
    for(int i=0;i<N;i++){
        double x = (double)(i - N/2) / (N/4);
        r->om[i] = strength * exp(-x*x);
    }
}

static void apply_edge_lens(ring_t *r,double strength){
    memset(r->om,0,N*sizeof(double));
    r->om[0] = strength * 0.5;
    r->om[N/2] = -strength * 0.5;
}

static void settle(ring_t *r,int chunks){
    int steps=(int)(T_CHUNK/DT);
    for(int c=0;c<chunks;c++){
        for(int s=0;s<steps;s++) rk4(r);
    }
}

int main(){
    printf("═══ EXPLORATION: Different lens profiles ═══\n\n");
    
    ring_t r;
    int chunks = 2;
    
    /* Test different lens profiles */
    struct {
        const char *name;
        void (*apply)(ring_t*, double);
        double strength;
    } lenses[] = {
        {"site0_+0.5", apply_site0_lens, 0.5},
        {"site0_-0.5", apply_site0_lens, -0.5},
        {"gauss_+1.0", apply_gauss_lens, 1.0},
        {"gauss_-1.0", apply_gauss_lens, -1.0},
        {"edge_+0.5", apply_edge_lens, 0.5},
        {"edge_-0.5", apply_edge_lens, -0.5},
    };
    
    int num_lenses = sizeof(lenses)/sizeof(lenses[0]);
    
    printf("Test: Encode a=2.0 into p=1, measure |c2| enhancement\n\n");
    printf("Lens profile\t\t|c2| no lens\t|c2| with lens\tEnhancement\n");
    printf("-\t\t-\t-\t-\n");
    
    for(int li=0; li<num_lenses; li++){
        encode_single(&r, 1, 2.0);
        
        /* Without lens */
        memset(r.om, 0, N*sizeof(double));
        settle(&r, chunks);
        double c2_no = ck_mag(&r, 2);
        
        /* With lens */
        lenses[li].apply(&r, lenses[li].strength);
        settle(&r, chunks);
        double c2_with = ck_mag(&r, 2);
        
        double enhancement = (c2_no > 1e-9) ? (c2_with / c2_no) : (c2_with > 1e-9 ? 1e9 : 0.0);
        
        printf("%-16s\t%.4f\t%.4f\t%.2fx\n", 
               lenses[li].name, c2_no, c2_with, enhancement);
    }
    
    printf("\n═══ EXPLORATION: Different input amplitudes ═══\n\n");
    
    /* Test different input amplitudes with site0 lens */
    printf("Test: a=1.0, 2.0, 3.0, 4.0, 5.0 into p=1, measure |c2| with lens\n\n");
    printf("a\t|c2| no lens\t|c2| with lens\tRatio\n");
    printf("-\t-\t-\t-\n");
    
    for(double a = 1.0; a <= 5.0; a += 1.0){
        encode_single(&r, 1, a);
        
        /* Without lens */
        memset(r.om, 0, N*sizeof(double));
        settle(&r, chunks);
        double c2_no = ck_mag(&r, 2);
        
        /* With lens */
        apply_site0_lens(&r, 0.5);
        settle(&r, chunks);
        double c2_with = ck_mag(&r, 2);
        
        printf("%.1f\t%.4f\t%.4f\t%.2f\n", a, c2_no, c2_with, c2_with/(a*a));
    }
    
    printf("\n═══ EXPLORATION: Different harmonic pairs ═══\n\n");
    
    /* Test different (p,q) pairs for a·b coupling */
    struct { int p, q; const char *desc; } pairs[] = {
        {1, 2, "p=1,q=2"},
        {1, 3, "p=1,q=3"},
        {2, 3, "p=2,q=3"},
        {1, 4, "p=1,q=4"},
    };
    
    printf("Test: a=2.0, b=1.5, measure |c_{p+q}| with lens\n\n");
    printf("Pair\t|c3| no lens\t|c3| with lens\tEnhancement\n");
    printf("-\t-\t-\t-\n");
    
    for(int pi=0; pi<4; pi++){
        int p = pairs[pi].p;
        int q = pairs[pi].q;
        int k_out = p + q;
        
        encode_pair(&r, p, 2.0, q, 1.5);
        
        /* Without lens */
        memset(r.om, 0, N*sizeof(double));
        settle(&r, chunks);
        double c_k_no = ck_mag(&r, k_out);
        
        /* With lens */
        apply_site0_lens(&r, 0.5);
        settle(&r, chunks);
        double c_k_with = ck_mag(&r, k_out);
        
        double enhancement = (c_k_no > 1e-9) ? (c_k_with / c_k_no) : (c_k_with > 1e-9 ? 1e9 : 0.0);
        
        printf("%s\t%.4f\t%.4f\t%.2fx\n", pairs[pi].desc, c_k_no, c_k_with, enhancement);
    }
    
    printf("\n═══ EXPLORATION: Time-domain behavior ═══\n\n");
    
    /* Measure at different times */
    encode_single(&r, 1, 2.0);
    apply_site0_lens(&r, 0.5);
    
    printf("Time\t|c2|\t|c3|\t|c4|\n");
    printf("-\t-\t-\t-\n");
    
    for(int t=0; t<=4; t++){
        settle(&r, t);
        printf("%.0f\t%.4f\t%.4f\t%.4f\n", 
               t * T_CHUNK, ck_mag(&r, 2), ck_mag(&r, 3), ck_mag(&r, 4));
    }
    
    return 0;
}
