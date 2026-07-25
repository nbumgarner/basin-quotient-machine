/* ===========================================================================
 * lens_sweep_experiment.c — Finding Optimal Lens Profiles for Coupling Enhancement
 *
 *   Experiments with different omega (lens) profiles to enhance specific
 *   g(p,q) coupling coefficients in the BQSM ring.
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

/* Encode value into harmonic k */
static void encode_harmonic(ring_t *r,int k,double A){
    memset(r->th,0,N*sizeof(double));
    memset(r->om,0,N*sizeof(double));
    for(int j=0;j<N;j++){
        r->th[j]=2.0*A*cos(2.0*M_PI*k*j/N);
    }
}

/* Encode two values into harmonics p and q */
static void encode_pair(ring_t *r,int p,double A,int q,double B){
    memset(r->th,0,N*sizeof(double));
    memset(r->om,0,N*sizeof(double));
    for(int j=0;j<N;j++){
        r->th[j]=2.0*A*cos(2.0*M_PI*p*j/N)+2.0*B*cos(2.0*M_PI*q*j/N);
    }
}

/* Apply lens profile: omega array */
static void apply_lens_profile(ring_t *r,const double *profile){
    memcpy(r->om,profile,N*sizeof(double));
}

/* Baseline: measure |c_k| from single-harmonic input */
static double measure_baseline(ring_t *r,int p_harm,double A,int k_out,const double *lens_profile,int chunks){
    encode_harmonic(r,p_harm,A);
    apply_lens_profile(r,lens_profile);
    settle(r,chunks);
    return ck_mag(r,k_out);
}

/* Coupled: measure |c_k| from paired input */
static double measure_coupled(ring_t *r,int p,double A,int q,double B,int k_out,const double *lens_profile,int chunks){
    encode_pair(r,p,A,q,B);
    apply_lens_profile(r,lens_profile);
    settle(r,chunks);
    return ck_mag(r,k_out);
}

/* Test a specific lens profile for enhancing g(p,q) -> c_{p+q} */
static void test_lens_profile(const char *profile_name,const double *profile,
                            int p,double A,int q,double B,int target_k,int chunks){
    ring_t r;
    double baseline_p = measure_baseline(&r,p,A,target_k,profile,chunks);
    double baseline_q = measure_baseline(&r,q,B,target_k,profile,chunks);
    double baseline_total = baseline_p + baseline_q;
    
    double coupled_signal = measure_coupled(&r,p,A,q,B,target_k,profile,chunks);
    double signal = coupled_signal - baseline_total;
    double enhancement = fabs(baseline_total)>1e-9 ? fabs(signal/baseline_total) : 0.0;
    
    printf("%-20s | p=%d,q=%d→k=%d | baseline=%.4f | coupled=%.4f | signal=%+.4f | enhancement=%.2fx\n",
           profile_name,p,q,target_k,baseline_total,coupled_signal,
           signal, enhancement);
}

int main(){
    printf("═══ Lens Profile Experiment for Coupling Enhancement ═══\n");
    printf("Testing different omega profiles to enhance g(p,q) coupling\n\n");
    
    /* Test parameters */
    double A = 4.0, B = 3.0;
    int chunks = 2;
    
    /* Define test cases: (p,q) pairs we want to enhance */
    struct {int p,q,target; const char *desc;} test_cases[] = {
        {1,1,2,"a² (g[1,1] -> c₂)"},
        {1,2,3,"a·b (g[1,2]+g[2,1] -> c₃)"},
        {2,1,3,"a·b (g[2,1] -> c₃)"},
        {2,2,4,"b² (g[2,2] -> c₄)"},
        {1,3,4,"a·c (g[1,3] -> c₄)"},
        {3,1,4,"a·c (g[3,1] -> c₄)"},
        {0,0,0,NULL}  /* sentinel */
    };
    
    /* Define lens profiles to test */
    struct {const char *name; double profile[N];} profiles[] = {
        /* Uniform bias */
        {"uniform_+0.1", {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}},
        {"uniform_-0.1", {-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1}},
        {"uniform_+0.2", {0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2}},
        
        /* Single site excitation */
        {"site0_+0.5",   {0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}},
        {"site0_-0.5",   {-0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}},
        {"site4_+0.5",   {0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0}},
        {"site8_+0.5",   {0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0}},
        
        /* Harmonic patterns */
        {"cos_k1",       {1.0,0.8,0.0,-0.8,-1.0,-0.8,0.0,0.8,1.0,0.8,0.0,-0.8,-1.0,-0.8,0.0,0.8}},
        {"cos_k2",       {1.0,0.0,-1.0,0.0,1.0,0.0,-1.0,0.0,1.0,0.0,-1.0,0.0,1.0,0.0,-1.0,0.0}},
        {"cos_k3",       {1.0,0.5,-0.5,-1.0,-0.5,0.5,1.0,0.5,-0.5,-1.0,-0.5,0.5,1.0,0.5,-0.5,-1.0}},
        {"cos_k4",       {1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0}},
        
        /* Linear ramps */
        {"linear_up",    {-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0,0.8,0.6,0.4,0.2,0.0,-0.2}},
        {"linear_down",  {0.2,0.4,0.6,0.8,1.0,0.8,0.6,0.4,0.2,0.0,-0.2,-0.4,-0.6,-0.8,-1.0,-0.8}},
        
        /* Alternating */
        {"alternating",  {0.5,-0.5,0.5,-0.5,0.5,-0.5,0.5,-0.5,0.5,-0.5,0.5,-0.5,0.5,-0.5,0.5,-0.5}},
        
        /* Gaussian bump */
        {"gauss_center", {0.0,0.0,0.0,0.0,0.0,0.5,1.0,0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}},
        {"gauss_edge",   {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,1.0,0.5,0.0,0.0,0.0,0.0,0.0}},
        
        /* Zero baseline (no lens) */
        {"no_lens",      {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}}
    };
    
    int num_profiles = sizeof(profiles)/sizeof(profiles[0]);
    
    printf("Testing %d lens profiles for coupling enhancement\n\n", num_profiles);
    printf("Format: Profile | p,q→k | baseline | coupled | signal | enhancement\n");
    printf("-------------------------------------------------------------------------------\n");
    
    /* Test each profile against each target coupling */
    for(int prof_idx=0;prof_idx<num_profiles;prof_idx++){
        for(int tc_idx=0;test_cases[tc_idx].desc!=NULL;tc_idx++){
            test_lens_profile(profiles[prof_idx].name, profiles[prof_idx].profile,
                            test_cases[tc_idx].p, A,
                            test_cases[tc_idx].q, B,
                            test_cases[tc_idx].target, chunks);
        }
        printf("\n");  /* Blank line between profiles */
    }
    
    printf("\n═══ Experiment Complete ═══\n");
    printf("Look for entries with high enhancement values (>10x)\n");
    printf("These indicate lens profiles that successfully boost specific g(p,q) coupling\n");
    
    return 0;
}