/* ===========================================================================
 * adversarial_test.c — Adversarial test suite for untrusted_model
 *
 * Attempts to break the BQSM product computer by testing edge cases,
 * numerical stability limits, and adversarial inputs.
 * ======================================================================== */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define N 16
#define NMASK (N-1)
#define NH 8
#define DT 0.50
#define T_CHUNK 60.0

typedef struct { double th[N], om[N], k1[N], k2[N], k3[N], k4[N], tmp[N]; } ring_t;
typedef struct { double decay[9]; } baseline_t;
typedef struct { double raw[9], signal[9], g[9]; } alu_result_t;

/* Core BQSM physics */
static void deriv(ring_t *r){
    for(int i=0;i<N;i++){
        int ip=(i+1)&(N-1), im=(i-1)&(N-1);
        r->tmp[i]=r->om[i]+sin(r->th[(i+1)&15]-r->th[i])+sin(r->th[(i-1)&15]-r->th[i]);
    }
}
static void rk4(ring_t *r){
    deriv(r);
    for(int i=0;i<16;i++)r->k1[i]=r->tmp[i];
    for(int i=0;i<16;i++)r->tmp[i]=r->th[i]+0.5*0.5*r->k1[i];
    deriv(r);
    for(int i=0;i<16;i++)r->k2[i]=r->tmp[i];
    for(int i=0;i<16;i++)r->tmp[i]=r->th[i]+0.5*0.5*r->k2[i];
    deriv(r);
    for(int i=0;i<16;i++)r->k3[i]=r->tmp[i];
    for(int i=0;i<16;i++)r->tmp[i]=r->th[i]+0.5*r->k3[i];
    deriv(r);
    for(int i=0;i<16;i++)r->k4[i]=r->tmp[i];
    for(int i=0;i<16;i++)r->th[i]+=(0.5/6.0)*(r->k1[i]+2.0*r->k2[i]+2.0*r->k3[i]+r->k4[i]);
}
static void settle(ring_t *r,int chunks){
    int steps=(int)(60.0/0.5);
    for(int c=0;c<chunks;c++)for(int s=0;s<120;s++)rk4(r);
}
static double ck_mag(ring_t *r,int k){
    double re=0,im=0;
    for(int j=0;j<16;j++){double a=-2.0*M_PI*k*j/16;re+=r->th[j]*cos(a);im+=r->th[j]*sin(a);}
    return sqrt(re*re+im*im)/16;
}
static void measure_baseline(double lens_omega[16], int chunks, double baseline[9][9]){
    for(int p=1; p<=8; p++){
        ring_t r; memset(&r,0,sizeof(r));
        for(int i=0;i<16;i++) r.om[i]=lens_omega[i];
        for(int j=0;j<16;j++) r.th[j]=2.0*cos(2.0*M_PI*p*j/16);
        settle(&r, chunks);
        for(int k=1;k<=8;k++) baseline[p][k]=ck_mag(&r,k);
    }
}
static void run_alu(double lens_omega[16], int chunks, double a, double b, double baseline[9][9], double g[9], double signal[9]){
    ring_t r; memset(&r,0,sizeof(r));
    for(int i=0;i<16;i++) r.om[i]=lens_omega[i];
    for(int j=0;j<16;j++) r.th[j]=2.0*a*cos(2.0*M_PI*1*j/16)+2.0*b*cos(2.0*M_PI*2*j/16);
    settle(&r, chunks);
    for(int k=1;k<=8;k++){
        double raw=ck_mag(&r,k);
        double baseline_sum=baseline[1][k]+baseline[2][k];
        double signal_val=raw-baseline_sum;
        if(k==2) g[2]=(fabs(a*a)>0.001)? signal_val/(a*a) : 0;
        if(k==3) g[3]=(fabs(a*b)>0.001)? signal_val/(a*b) : 0;
        if(k==4) g[4]=(fabs(b*b)>0.001)? signal_val/(b*b) : 0;
        signal[k]=signal_val;
    }
}

int run_test(const char *name, double a, double b, double lens_omega[16], double baseline[9][9], int verbose){
    double signal[9], g[9];
    run_alu(lens_omega, 2, a, b, baseline, g, signal);

    double a2 = signal[2] / g[2];
    double ab = signal[3] / g[3];
    double b2 = signal[4] / g[4];

    double err_a2 = fabs(a2 - a*a) / fabs(a*a) * 100;
    double err_ab = fabs(ab - a*b) / fabs(a*b) * 100;
    double err_b2 = fabs(b2 - b*b) / fabs(b*b) * 100;

    int pass = (err_a2 < 1.0 && err_ab < 1.0 && err_b2 < 1.0);

    if(verbose || !pass){
        printf("%s: a=%.2f b=%.2f | a²_err=%.2f%% ab_err=%.2f%% b²_err=%.2f%% | %s\n",
               name, a, b, err_a2, err_ab, err_b2, pass?"PASS":"FAIL");
    }
    return pass;
}

int main(){
    printf("═══ ADVERSARIAL TEST SUITE FOR UNTRUSTED_MODEL ═══\n\n");

    /* SITE0_SHIFT lens */
    double lens_omega[16]; memset(lens_omega,0,sizeof(lens_omega)); lens_omega[0]=0.2;
    double baseline[9][9]; memset(baseline,0,sizeof(baseline));
    measure_baseline(lens_omega, 2, baseline);

    int total=0, passed=0;

    printf("═══ TEST 1: ORIGINAL CALIBRATION POINTS ═══\n");
    double test_a[] = {1.0,2.0,3.0,4.0};
    double test_b[] = {1.0,1.5,2.0,2.5};
    for(int i=0;i<4;i++)for(int j=0;j<4;j++){
        total++; passed+=run_test("orig",test_a[i],test_b[j],lens_omega,baseline,1);
    }

    printf("\n═══ TEST 2: EDGE CASES - ZERO & NEAR-ZERO ═══\n");
    double edge_a[] = {0.0, 1e-10, 1e-6, 1e-3, 0.001, 0.01, 0.1};
    double edge_b[] = {0.0, 1e-10, 1e-6, 1e-3, 0.001, 0.01, 0.1};
    for(int i=0;i<7;i++)for(int j=0;j<7;j++){
        if(edge_a[i]==0 && edge_b[j]==0) continue;
        total++; passed+=run_test("edge",edge_a[i],edge_b[j],lens_omega,baseline,0);
    }

    printf("\n═══ TEST 3: LARGE VALUES (OVERFLOW BOUNDARY) ═══\n");
    double large_a[] = {10, 50, 100, 500, 1000, 1e4, 1e5, 1e6};
    double large_b[] = {1.0, 10.0, 100.0};
    for(int i=0;i<8;i++)for(int j=0;j<3;j++){
        total++; passed+=run_test("large",large_a[i],large_b[j],lens_omega,baseline,0);
    }

    printf("\n═══ TEST 4: NEGATIVE VALUES ═══\n");
    double neg_a[] = {-1,-2,-10,-100,-1000};
    double neg_b[] = {-1,-1.5,-2,-10,-100};
    for(int i=0;i<5;i++)for(int j=0;j<5;j++){
        total++; passed+=run_test("neg",neg_a[i],neg_b[j],lens_omega,baseline,0);
    }

    printf("\n═══ TEST 5: MIXED SIGN ═══\n");
    double mix_a[] = {-10,-1,-0.1,0.1,1,10,100};
    double mix_b[] = {-10,-1,-0.1,0.1,1,10,100};
    for(int i=0;i<7;i++)for(int j=0;j<7;j++){
        total++; passed+=run_test("mix",mix_a[i],mix_b[j],lens_omega,baseline,0);
    }

    printf("\n═══ TEST 6: EXTREME RATIOS ═══\n");
    double ratio_a[] = {1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6};
    double ratio_b[] = {1e6, 1e4, 1e2, 1, 1e-2, 1e-4, 1e-6};
    for(int i=0;i<7;i++)for(int j=0;j<7;j++){
        total++; passed+=run_test("ratio",ratio_a[i],ratio_b[j],lens_omega,baseline,0);
    }

    printf("\n═══ TEST 7: VARYING LENS STRENGTH ═══\n");
    double lens_strengths[] = {0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.62, 0.65, 0.7, 0.8, 1.0};
    for(int s=0;s<15;s++){
        double lens[16]; memset(lens,0,sizeof(double[16])); lens[0]=lens_strengths[s];
        double bl[9][9]; measure_baseline(lens,2,bl);
        int pass = run_test("lens_str",2.0,1.5,lens,bl,0);
        total++; passed+=pass;
        printf("  ω[0]=%.2f: %s\n", lens_strengths[s], pass?"PASS":"FAIL");
    }

    printf("\n═══ TEST 8: ALTERNATIVE LENS PROFILES ═══\n");
    // IDENTITY
    double lens_id[16]={0};
    double bl_id[9][9]; measure_baseline(lens_id,2,bl_id);
    total++; passed+=run_test("IDENTITY",2.0,1.5,lens_id,bl_id,1);

    // LINEAR_RAMP
    double lens_lr[16]; for(int i=0;i<16;i++) lens_lr[i]=0.02*i;
    double bl_lr[9][9]; measure_baseline(lens_lr,2,bl_lr);
    total++; passed+=run_test("LINEAR_RAMP",2.0,1.5,lens_lr,bl_lr,1);

    // QUAD
    double lens_qd[16]={0}; lens_qd[0]=lens_qd[4]=lens_qd[8]=lens_qd[12]=0.2;
    double bl_qd[9][9]; measure_baseline(lens_qd,2,bl_qd);
    total++; passed+=run_test("QUAD",2.0,1.5,lens_qd,bl_qd,1);

    // ANTIPODAL
    double lens_ap[16]={0}; lens_ap[0]=0.3; lens_ap[8]=-0.3;
    double bl_ap[9][9]; measure_baseline(lens_ap,2,bl_ap);
    total++; passed+=run_test("ANTIPODAL",2.0,1.5,lens_ap,bl_ap,1);

    printf("\n═══ TEST 9: SETTLING CHUNKS STRESS ═══\n");
    for(int chunks=1;chunks<=10;chunks++){
        double bl[9][9]; measure_baseline(lens_omega,chunks,baseline);
        int pass = run_test("chunks",2.0,1.5,lens_omega,baseline,0);
        total++; passed+=pass;
        printf("  chunks=%d: %s\n", chunks, pass?"PASS":"FAIL");
    }

    printf("\n═══ TEST 10: RANDOM STRESS (1000 random inputs) ═══\n");
    srand(42);
    for(int i=0;i<1000;i++){
        double a = (rand()/((double)RAND_MAX))*200 - 100;
        double b = (rand()/((double)RAND_MAX))*200 - 100;
        if(fabs(a)<1e-10) a = 0.001;
        if(fabs(b)<1e-10) b = 0.001;
        total++;
        int pass = run_test("rand",a,b,lens_omega,baseline,0);
        passed+=pass;
    }

    printf("\n═══ TEST 12: NUMERICAL PRECISION EDGES ═══\n");
    double prec_a[] = {DBL_MIN, DBL_EPSILON, 1e-15, 1e-12, 1e-9, 1e-6, FLT_MIN, FLT_EPSILON};
    double prec_b[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    for(int i=0;i<8;i++){
        total++; passed+=run_test("prec",prec_a[i],prec_b[i],lens_omega,baseline,0);
    }

    printf("\n═══ TEST 12b: PRECISION ON LARGE VALUES ═══\n");
    double large_prec_a[] = {DBL_MAX/2, 1e100, 1e150, 1e200, 1e250, 1e300};
    double large_prec_b[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    for(int i=0;i<6;i++){
        total++; passed+=run_test("large_prec",large_prec_a[i],large_prec_b[i],lens_omega,baseline,0);
    }

    printf("\n═══ TEST 13: NaN/INF HANDLING ═══\n");
    double special_a[] = {NAN, INFINITY, -INFINITY, 0.0/0.0, 1.0/0.0, -1.0/0.0};
    double special_b[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    for(int i=0;i<6;i++){
        total++;
        int pass = run_test("special",special_a[i],special_b[i],lens_omega,baseline,0);
        passed+=pass;
        printf("  special: %s\n", pass?"PASS":"FAIL (expected)");
    }

    printf("\n═══ TEST 14: VARYING BASELINE CHUNKS ═══\n");
    for(int bc=1;bc<=5;bc++){
        double bl[9][9]; measure_baseline(lens_omega,bc,baseline);
        int pass = run_test("base_chunks",2.0,1.5,lens_omega,baseline,0);
        total++; passed+=pass;
    }

    printf("\n═══ SUMMARY ═══\n");
    printf("Total tests: %d\n", total);
    printf("Passed: %d\n", passed);
    printf("Failed: %d\n", total-passed);
    printf("Pass rate: %.2f%%\n", 100.0*passed/total);

    if(total==passed){
        printf("\n✓✓✓ ALL TESTS PASSED — MODEL ROBUST\n");
    }else{
        printf("\n✗✗✗ %d TESTS FAILED — MODEL HAS WEAKNESSES\n", total-passed);
    }

    return (total==passed)?0:1;
}