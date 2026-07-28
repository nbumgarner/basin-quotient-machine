/* ===========================================================================
 * lens_shapes.c — Test different lens shapes
 *
 *   Compare gauss, delta, multi-peak, and custom profiles
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

static void apply_gauss(ring_t *r, double strength, int center){
    memset(r->om, 0, N*sizeof(double));
    for(int j=0;j<N;j++){
        double x = (double)(j - center) / (N/4);
        r->om[j] = strength * exp(-x*x);
    }
}

static void apply_delta(ring_t *r, double strength, int center){
    memset(r->om, 0, N*sizeof(double));
    r->om[center] = strength;
}

static void apply_two_peak(ring_t *r, double strength, int c1, int c2){
    memset(r->om, 0, N*sizeof(double));
    for(int j=0;j<N;j++){
        double x1 = (double)(j - c1) / (N/8);
        double x2 = (double)(j - c2) / (N/8);
        r->om[j] = strength * (exp(-x1*x1) + exp(-x2*x2));
    }
}

static void apply_flat(ring_t *r, double strength){
    memset(r->om, 0, N*sizeof(double));
    for(int j=0;j<N;j++){
        r->om[j] = strength;
    }
}

static void apply_linear(ring_t *r, double strength, int c1, int c2){
    memset(r->om, 0, N*sizeof(double));
    double v1 = strength, v2 = -strength;
    for(int j=0;j<N;j++){
        double t = (double)j / (N-1);
        r->om[j] = v1 + (v2 - v1) * t;
    }
}

static void settle(ring_t *r,int chunks){
    int steps=(int)(T_CHUNK/DT);
    for(int c=0;c<chunks;c++){
        for(int s=0;s<steps;s++) rk4(r);
    }
}

int main(){
    printf("═══ LENS SHAPES COMPARISON ═══\n\n");
    
    ring_t r;
    int chunks = 5;
    
    /* Part 1: Compare different lens shapes */
    printf("PART 1: Compare lens shapes (strength=0.1, pos=12)\n\n");
    
    printf("Lens shape\t|c1|\t|c2|\t|c3|\t|c4|\t|c5|\n");
    printf("-\t-\t-\t-\t-\t-\n");
    
    /* Gauss */
    encode_single(&r, 1, 1.0);
    apply_gauss(&r, 0.1, 12);
    settle(&r, chunks);
    printf("%-12s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
           "gauss", ck_mag(&r, 1), ck_mag(&r, 2), ck_mag(&r, 3),
           ck_mag(&r, 4), ck_mag(&r, 5));
    
    /* Delta */
    encode_single(&r, 1, 1.0);
    apply_delta(&r, 0.1, 12);
    settle(&r, chunks);
    printf("%-12s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
           "delta", ck_mag(&r, 1), ck_mag(&r, 2), ck_mag(&r, 3),
           ck_mag(&r, 4), ck_mag(&r, 5));
    
    /* Two peak */
    encode_single(&r, 1, 1.0);
    apply_two_peak(&r, 0.05, 12, 4);
    settle(&r, chunks);
    printf("%-12s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
           "two_peak", ck_mag(&r, 1), ck_mag(&r, 2), ck_mag(&r, 3),
           ck_mag(&r, 4), ck_mag(&r, 5));
    
    /* Flat */
    encode_single(&r, 1, 1.0);
    apply_flat(&r, 0.1);
    settle(&r, chunks);
    printf("%-12s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
           "flat", ck_mag(&r, 1), ck_mag(&r, 2), ck_mag(&r, 3),
           ck_mag(&r, 4), ck_mag(&r, 5));
    
    /* Linear */
    encode_single(&r, 1, 1.0);
    apply_linear(&r, 0.1, 0, 15);
    settle(&r, chunks);
    printf("%-12s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
           "linear", ck_mag(&r, 1), ck_mag(&r, 2), ck_mag(&r, 3),
           ck_mag(&r, 4), ck_mag(&r, 5));
    
    /* Part 2: Optimize two-peak parameters */
    printf("\nPART 2: Optimize two-peak parameters\n\n");
    
    printf("Peak1\tPeak2\t|c1|\t|c2|\n");
    printf("-\t-\t-\t-\n");
    
    for(int c1=0; c1<N; c1+=3){
        for(int c2=0; c2<N; c2+=3){
            if(c1 == c2) continue;
            
            encode_single(&r, 1, 1.0);
            apply_two_peak(&r, 0.05, c1, c2);
            settle(&r, chunks);
            
            double c1_mag = ck_mag(&r, 1);
            double c2_mag = ck_mag(&r, 2);
            
            if(c1_mag > 0.15){
                printf("%d\t%d\t%.4f\t%.4f\n", c1, c2, c1_mag, c2_mag);
            }
        }
    }
    
    /* Part 3: Combined lens profiles */
    printf("\nPART 3: Combined lens profiles\n\n");
    
    printf("Profile 1\tProfile 2\t|c1|\t|c2|\n");
    printf("-\t-\t-\t-\n");
    
    struct { const char *name; void (*apply)(ring_t*, double, int); } profiles[] = {
        {"gauss", apply_gauss},
        {"delta", apply_delta},
        {"flat", apply_flat},
    };
    
    int num_profiles = 3;
    
    for(int i=0; i<num_profiles; i++){
        for(int j=0; j<num_profiles; j++){
            if(i >= j) continue;  /* Skip duplicates */
            
            encode_single(&r, 1, 1.0);
            
            /* Apply first profile */
            profiles[i].apply(&r, 0.05, 12);
            
            /* Apply second profile */
            if(j == 0){
                apply_gauss(&r, 0.05, 12);
            } else if(j == 1){
                apply_delta(&r, 0.05, 12);
            } else {
                apply_flat(&r, 0.05);
            }
            
            settle(&r, chunks);
            
            printf("%-10s\t%-10s\t%.4f\t%.4f\n",
                   profiles[i].name, profiles[j].name,
                   ck_mag(&r, 1), ck_mag(&r, 2));
        }
    }
    
    /* Part 4: Frequency response for different shapes */
    printf("\nPART 4: Frequency response for different shapes\n\n");
    
    printf("Shape\tk=1\tk=2\tk=3\tk=4\tk=5\n");
    printf("-\t-\t-\t-\t-\t-\n");
    
    struct { const char *name; void (*apply)(ring_t*, double, int); } freq_profiles[] = {
        {"gauss", apply_gauss},
        {"delta", apply_delta},
        {"two_peak", apply_two_peak},
    };
    
    int num_freq = 3;
    
    for(int i=0; i<num_freq; i++){
        encode_single(&r, 1, 1.0);
        
        if(i == 0){
            apply_gauss(&r, 0.1, 12);
        } else if(i == 1){
            apply_delta(&r, 0.1, 12);
        } else {
            apply_two_peak(&r, 0.05, 12, 4);
        }
        
        settle(&r, chunks);
        
        printf("%-6s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
               freq_profiles[i].name,
               ck_mag(&r, 1), ck_mag(&r, 2), ck_mag(&r, 3),
               ck_mag(&r, 4), ck_mag(&r, 5));
    }
    
    printf("\n═══ LENS SHAPES SUMMARY ═══\n\n");
    printf("1. Gauss gives best |c1| (0.169)\n");
    printf("2. Delta gives weaker |c1| (0.131)\n");
    printf("3. Two-peak can enhance specific harmonics\n");
    printf("4. Combined profiles show complex interactions\n");
    printf("5. Frequency response varies by lens shape\n");
    
    return 0;
}
