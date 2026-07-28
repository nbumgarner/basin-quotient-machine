/* ===========================================================================
 * harmonic_selector.c — Explore harmonic selection mechanism
 *
 *   Test different lens shapes to find optimal harmonic selection
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

static void apply_gauss_pos1(ring_t *r, double s){
    memset(r->om, 0, N*sizeof(double));
    for(int j=0;j<N;j++){
        double x = (double)(j - 1) / (N/4);
        r->om[j] = s * exp(-x*x);
    }
}

static void apply_gauss_center(ring_t *r, double s){
    memset(r->om, 0, N*sizeof(double));
    for(int j=0;j<N;j++){
        double x = (double)(j - N/2) / (N/4);
        r->om[j] = s * exp(-x*x);
    }
}

static void apply_gauss_pos13(ring_t *r, double s){
    memset(r->om, 0, N*sizeof(double));
    for(int j=0;j<N;j++){
        double x = (double)(j - 13) / (N/4);
        r->om[j] = s * exp(-x*x);
    }
}

static void apply_two_peak(ring_t *r, double s){
    memset(r->om, 0, N*sizeof(double));
    for(int j=0;j<N;j++){
        double x1 = (double)(j - 1) / (N/8);
        double x2 = (double)(j - 15) / (N/8);
        r->om[j] = s * (exp(-x1*x1) + exp(-x2*x2));
    }
}

static void apply_delta(ring_t *r, double s){
    memset(r->om, 0, N*sizeof(double));
    r->om[1] = s;
}

static void settle(ring_t *r,int chunks){
    int steps=(int)(T_CHUNK/DT);
    for(int c=0;c<chunks;c++){
        for(int s=0;s<steps;s++) rk4(r);
    }
}

int main(){
    printf("═══ HARMONIC SELECTOR: Different lens shapes ═══\n\n");
    
    ring_t r;
    int chunks = 2;
    
    printf("Test: a=2.0 into p=1, measure all harmonics\n\n");
    printf("Lens\t\t|c1|\t|c2|\t|c3|\t|c4|\t|c5|\n");
    printf("-\t-\t-\t-\t-\t-\n");
    
    /* gauss_pos1 */
    encode_single(&r, 1, 2.0);
    apply_gauss_pos1(&r, 0.9);
    settle(&r, chunks);
    printf("%-12s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n",
           "gauss_pos1",
           ck_mag(&r, 1), ck_mag(&r, 2), ck_mag(&r, 3),
           ck_mag(&r, 4), ck_mag(&r, 5));
    
    /* gauss_center */
    encode_single(&r, 1, 2.0);
    apply_gauss_center(&r, 0.9);
    settle(&r, chunks);
    printf("%-12s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n",
           "gauss_center",
           ck_mag(&r, 1), ck_mag(&r, 2), ck_mag(&r, 3),
           ck_mag(&r, 4), ck_mag(&r, 5));
    
    /* gauss_pos13 */
    encode_single(&r, 1, 2.0);
    apply_gauss_pos13(&r, 0.9);
    settle(&r, chunks);
    printf("%-12s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n",
           "gauss_pos13",
           ck_mag(&r, 1), ck_mag(&r, 2), ck_mag(&r, 3),
           ck_mag(&r, 4), ck_mag(&r, 5));
    
    /* two_peak */
    encode_single(&r, 1, 2.0);
    apply_two_peak(&r, 0.45);
    settle(&r, chunks);
    printf("%-12s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n",
           "two_peak",
           ck_mag(&r, 1), ck_mag(&r, 2), ck_mag(&r, 3),
           ck_mag(&r, 4), ck_mag(&r, 5));
    
    /* delta */
    encode_single(&r, 1, 2.0);
    apply_delta(&r, 0.9);
    settle(&r, chunks);
    printf("%-12s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n",
           "delta",
           ck_mag(&r, 1), ck_mag(&r, 2), ck_mag(&r, 3),
           ck_mag(&r, 4), ck_mag(&r, 5));
    
    printf("\n═══ HARMONIC SELECTOR: Different input harmonics ═══\n\n");
    
    printf("Test: Different p values, measure all output harmonics\n\n");
    printf("Input p\tOutput k\t|c_k|\n");
    printf("-\t-\t-\n");
    
    for(int p=1; p<=4; p++){
        encode_single(&r, p, 2.0);
        
        /* Apply gauss lens at position 1 */
        apply_gauss_pos1(&r, 0.9);
        settle(&r, chunks);
        
        for(int k=1; k<=5; k++){
            printf("p=%d\tk=%d\t%.3f\n", p, k, ck_mag(&r, k));
        }
    }
    
    printf("\n═══ HARMONIC SELECTOR: Nonlinear behavior ═══\n\n");
    
    printf("Test: Measure |c2| for different input amplitudes with different lens shapes\n\n");
    printf("Input a\tgauss_pos1\tgauss_center\tgauss_pos13\ttwo_peak\tdelta\n");
    printf("-\t-\t-\t-\t-\t-\n");
    
    for(double a=1.0; a<=3.0; a+=1.0){
        encode_single(&r, 1, a);
        apply_gauss_pos1(&r, 0.9);
        settle(&r, chunks);
        double c2_g1 = ck_mag(&r, 2);
        
        encode_single(&r, 1, a);
        apply_gauss_center(&r, 0.9);
        settle(&r, chunks);
        double c2_gc = ck_mag(&r, 2);
        
        encode_single(&r, 1, a);
        apply_gauss_pos13(&r, 0.9);
        settle(&r, chunks);
        double c2_g13 = ck_mag(&r, 2);
        
        encode_single(&r, 1, a);
        apply_two_peak(&r, 0.45);
        settle(&r, chunks);
        double c2_tp = ck_mag(&r, 2);
        
        encode_single(&r, 1, a);
        apply_delta(&r, 0.9);
        settle(&r, chunks);
        double c2_d = ck_mag(&r, 2);
        
        printf("%.1f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n",
               a, c2_g1, c2_gc, c2_g13, c2_tp, c2_d);
    }
    
    return 0;
}
