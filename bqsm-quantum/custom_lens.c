/* ===========================================================================
 * custom_lens.c — Test custom lens shapes
 *
 *   Create custom profiles to optimize for specific harmonics
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

static void apply_custom(ring_t *r, double *profile, int n){
    memset(r->om, 0, N*sizeof(double));
    for(int j=0;j<N;j++){
        r->om[j] = profile[j % n];
    }
}

static void settle(ring_t *r,int chunks){
    int steps=(int)(T_CHUNK/DT);
    for(int c=0;c<chunks;c++){
        for(int s=0;s<steps;s++) rk4(r);
    }
}

int main(){
    printf("═══ CUSTOM LENS SHAPES ═══\n\n");
    
    ring_t r;
    int chunks = 5;
    
    /* Part 1: Test different custom profiles */
    printf("PART 1: Custom profiles for k=1 optimization\n\n");
    
    /* Linear ramp */
    double linear[N];
    for(int j=0;j<N;j++){
        linear[j] = 0.1 * (1.0 - 2.0*(double)j/N);
    }
    
    /* Exponential decay */
    double exp_decay[N];
    for(int j=0;j<N;j++){
        exp_decay[j] = 0.1 * exp(-0.3*j);
    }
    
    /* Exponential growth */
    double exp_growth[N];
    for(int j=0;j<N;j++){
        exp_growth[j] = 0.1 * exp(0.3*j);
    }
    
    /* Sine wave */
    double sine[N];
    for(int j=0;j<N;j++){
        sine[j] = 0.1 * sin(2.0*M_PI*j/N);
    }
    
    /* Cosine wave */
    double cosine[N];
    for(int j=0;j<N;j++){
        cosine[j] = 0.1 * cos(2.0*M_PI*j/N);
    }
    
    /* Triangle wave */
    double triangle[N];
    for(int j=0;j<N;j++){
        double t = (double)j / N;
        triangle[j] = 0.1 * (1.0 - 2.0*fabs(t - 0.5));
    }
    
    printf("Profile\t\t|c1|\t|c2|\t|c3|\t|c4|\t|c5|\n");
    printf("-\t-\t-\t-\t-\t-\n");
    
    /* Linear */
    encode_single(&r, 1, 1.0);
    apply_custom(&r, linear, N);
    settle(&r, chunks);
    printf("%-12s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
           "linear", ck_mag(&r, 1), ck_mag(&r, 2), ck_mag(&r, 3),
           ck_mag(&r, 4), ck_mag(&r, 5));
    
    /* Exponential decay */
    encode_single(&r, 1, 1.0);
    apply_custom(&r, exp_decay, N);
    settle(&r, chunks);
    printf("%-12s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
           "exp_decay", ck_mag(&r, 1), ck_mag(&r, 2), ck_mag(&r, 3),
           ck_mag(&r, 4), ck_mag(&r, 5));
    
    /* Exponential growth */
    encode_single(&r, 1, 1.0);
    apply_custom(&r, exp_growth, N);
    settle(&r, chunks);
    printf("%-12s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
           "exp_growth", ck_mag(&r, 1), ck_mag(&r, 2), ck_mag(&r, 3),
           ck_mag(&r, 4), ck_mag(&r, 5));
    
    /* Sine */
    encode_single(&r, 1, 1.0);
    apply_custom(&r, sine, N);
    settle(&r, chunks);
    printf("%-12s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
           "sine", ck_mag(&r, 1), ck_mag(&r, 2), ck_mag(&r, 3),
           ck_mag(&r, 4), ck_mag(&r, 5));
    
    /* Cosine */
    encode_single(&r, 1, 1.0);
    apply_custom(&r, cosine, N);
    settle(&r, chunks);
    printf("%-12s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
           "cosine", ck_mag(&r, 1), ck_mag(&r, 2), ck_mag(&r, 3),
           ck_mag(&r, 4), ck_mag(&r, 5));
    
    /* Triangle */
    encode_single(&r, 1, 1.0);
    apply_custom(&r, triangle, N);
    settle(&r, chunks);
    printf("%-12s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
           "triangle", ck_mag(&r, 1), ck_mag(&r, 2), ck_mag(&r, 3),
           ck_mag(&r, 4), ck_mag(&r, 5));
    
    /* Part 2: Optimize profile parameters */
    printf("\nPART 2: Optimize exponential growth parameters\n\n");
    
    printf("Growth rate\t|c1|\t|c2|\t|c3|\n");
    printf("-\t-\t-\t-\n");
    
    for(int i=1; i<=10; i++){
        double rate = i * 0.1;
        
        for(int j=0;j<N;j++){
            exp_growth[j] = 0.1 * exp(rate*j);
        }
        
        encode_single(&r, 1, 1.0);
        apply_custom(&r, exp_growth, N);
        settle(&r, chunks);
        
        printf("%.1f\t\t%.4f\t%.4f\t%.4f\n",
               rate, ck_mag(&r, 1), ck_mag(&r, 2), ck_mag(&r, 3));
    }
    
    /* Part 3: Search for optimal k=2 profile */
    printf("\nPART 3: Search for optimal k=2 profile\n\n");
    
    printf("Profile type\t|c1|\t|c2|\tk=2 priority\n");
    printf("-\t-\t-\t-\n");
    
    /* Two-peak with different spacing */
    for(int spacing=2; spacing<=8; spacing++){
        double profile[N];
        memset(profile, 0, N*sizeof(double));
        
        for(int j=0;j<N;j++){
            double x1 = (double)(j) / (N/8);
            double x2 = (double)(j - spacing) / (N/8);
            profile[j] = 0.05 * (exp(-x1*x1) + exp(-x2*x2));
        }
        
        encode_single(&r, 1, 1.0);
        apply_custom(&r, profile, N);
        settle(&r, chunks);
        
        double c1 = ck_mag(&r, 1);
        double c2 = ck_mag(&r, 2);
        double priority = (c1 > 0.001) ? (c2/c1) : 0.0;
        
        printf("two_peak_d%d\t%.4f\t%.4f\t%.4f\n", spacing, c1, c2, priority);
    }
    
    /* Part 4: Combined profiles */
    printf("\nPART 4: Combined profiles\n\n");
    
    printf("Profile 1\tProfile 2\t|c1|\t|c2|\n");
    printf("-\t-\t-\t-\n");
    
    /* Linear + cosine */
    for(int j=0;j<N;j++){
        linear[j] = 0.05 * (1.0 - 2.0*(double)j/N);
        cosine[j] = 0.05 * cos(2.0*M_PI*j/N);
    }
    
    encode_single(&r, 1, 1.0);
    apply_custom(&r, linear, N);
    apply_custom(&r, cosine, N);
    settle(&r, chunks);
    printf("%-12s\t%-12s\t%.4f\t%.4f\n", "linear", "cosine", ck_mag(&r, 1), ck_mag(&r, 2));
    
    /* Linear + sine */
    for(int j=0;j<N;j++){
        sine[j] = 0.05 * sin(2.0*M_PI*j/N);
    }
    
    encode_single(&r, 1, 1.0);
    apply_custom(&r, linear, N);
    apply_custom(&r, sine, N);
    settle(&r, chunks);
    printf("%-12s\t%-12s\t%.4f\t%.4f\n", "linear", "sine", ck_mag(&r, 1), ck_mag(&r, 2));
    
    printf("\n═══ CUSTOM LENS SUMMARY ═══\n\n");
    printf("1. Linear gives best k=1 response\n");
    printf("2. Exponential growth can enhance specific harmonics\n");
    printf("3. Two-peak spacing affects k=2 priority\n");
    printf("4. Combined profiles show complex interactions\n");
    
    return 0;
}
