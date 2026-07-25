/* ===========================================================================
 * verification_experiment.c — Verify Enhanced Coupling Accuracy
 *
 *   Tests site0_±0.5 lens profile for mathematical accuracy:
 *   - a² (|c₂|), a·b (|c₃|), b² (|c₄|)
 *   - Input: a=2.0, b=1.5
 *   - Expected: a²=4.0, a·b=3.0, b²=2.25
 *   - Lens: site0_±0.5 (24.38× b², 10.94-11.94× a·c enhancement)
 *   - Baseline subtraction for noise removal
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

/* Encode two values into harmonics p and q */
static void encode_pair(ring_t *r,int p,double A,int q,double B){
    memset(r->th,0,N*sizeof(double));
    memset(r->om,0,N*sizeof(double));
    for(int j=0;j<N;j++){
        r->th[j]=2.0*A*cos(2.0*M_PI*p*j/N)+2.0*B*cos(2.0*M_PI*q*j/N);
    }
}

/* Apply site0_±0.5 lens profile */
static void apply_site0_lens(ring_t *r,double sign){
    memset(r->om,0,N*sizeof(double));
    r->om[0]=sign*0.5;  /* ±0.5 at oscillator 0 */
}

/* Measure |c_k| with given lens */
static double measure_with_lens(ring_t *r,int p,double A,int q,double B,int k_out,double sign,int chunks){
    encode_pair(r,p,A,q,B);
    apply_site0_lens(r,sign);
    settle(r,chunks);
    return ck_mag(r,k_out);
}

/* Baseline: measure |c_k| without lens */
static double measure_baseline(ring_t *r,int p,double A,int q,double B,int k_out,int chunks){
    encode_pair(r,p,A,q,B);
    memset(r->om,0,N*sizeof(double));
    settle(r,chunks);
    return ck_mag(r,k_out);
}

int main(){
    printf("═══ Verification Experiment: site0_±0.5 Lens Profile ═══\n");
    printf("Testing mathematical accuracy of enhanced couplings\n\n");
    
    /* Test parameters */
    double a = 2.0, b = 1.5;
    int chunks = 2;
    double lens_sign = 0.5;  /* +0.5 */
    
    printf("INPUT: a=%.1f, b=%.1f\n", a, b);
    printf("EXPECTED: a²=%.2f, a·b=%.2f, b²=%.2f\n", a*a, a*b, b*b);
    printf("LENS: site0_±%.1f (24.38× b², 10.94-11.94× a·c enhancement)\n", lens_sign);
    printf("CHUNKS: %d\n\n", chunks);
    
    ring_t r;
    
    /* Measure coupled signals */
    double c2_coupled = measure_with_lens(&r, 1, a, 2, b, 2, lens_sign, chunks);
    double c3_coupled = measure_with_lens(&r, 1, a, 2, b, 3, lens_sign, chunks);
    double c4_coupled = measure_with_lens(&r, 2, a, 2, b, 4, lens_sign, chunks);
    
    printf("COUPLED MEASUREMENTS (with lens):\n");
    printf("  |c₂| (a² channel):  %.4f\n", c2_coupled);
    printf("  |c₃| (a·b channel): %.4f\n", c3_coupled);
    printf("  |c₄| (b² channel):  %.4f\n", c4_coupled);
    
    /* Baseline subtraction */
    double c2_baseline = measure_baseline(&r, 1, a, 2, b, 2, chunks);
    double c3_baseline = measure_baseline(&r, 1, a, 2, b, 3, chunks);
    double c4_baseline = measure_baseline(&r, 2, a, 2, b, 4, chunks);
    
    printf("\nBASELINE MEASUREMENTS (no lens):\n");
    printf("  |c₂| baseline: %.4f\n", c2_baseline);
    printf("  |c₃| baseline: %.4f\n", c3_baseline);
    printf("  |c₄| baseline: %.4f\n", c4_baseline);
    
    printf("\nBASELINE-SUBTRACTED SIGNALS:\n");
    double c2_signal = c2_coupled - c2_baseline;
    double c3_signal = c3_coupled - c3_baseline;
    double c4_signal = c4_coupled - c4_baseline;
    
    printf("  a² signal:    %.4f\n", c2_signal);
    printf("  a·b signal:   %.4f\n", c3_signal);
    printf("  b² signal:    %.4f\n", c4_signal);
    
    /* Calculate enhancement factors */
    double c2_enhancement = fabs(c2_baseline)>1e-9 ? fabs(c2_coupled/c2_baseline) : 0.0;
    double c3_enhancement = fabs(c3_baseline)>1e-9 ? fabs(c3_coupled/c3_baseline) : 0.0;
    double c4_enhancement = fabs(c4_baseline)>1e-9 ? fabs(c4_coupled/c4_baseline) : 0.0;
    
    printf("\nENHANCEMENT FACTORS:\n");
    printf("  a² enhancement:   %.2fx\n", c2_enhancement);
    printf("  a·b enhancement:  %.2fx\n", c3_enhancement);
    printf("  b² enhancement:   %.2fx\n", c4_enhancement);
    
    /* Compare to expected values */
    printf("\nACCURACY COMPARISON:\n");
    double a2_expected = a*a;
    double ab_expected = a*b;
    double b2_expected = b*b;
    
    double a2_error = fabs(c2_signal - a2_expected) / a2_expected * 100.0;
    double ab_error = fabs(c3_signal - ab_expected) / ab_expected * 100.0;
    double b2_error = fabs(c4_signal - b2_expected) / b2_expected * 100.0;
    
    printf("  a²: measured=%.4f, expected=%.2f, error=%.2f%%\n", c2_signal, a2_expected, a2_error);
    printf("  a·b: measured=%.4f, expected=%.2f, error=%.2f%%\n", c3_signal, ab_expected, ab_error);
    printf("  b²: measured=%.4f, expected=%.2f, error=%.2f%%\n", c4_signal, b2_expected, b2_error);
    
    /* Overall assessment */
    printf("\n═══ VERDICT ═══\n");
    if(a2_error < 5.0 && ab_error < 5.0 && b2_error < 5.0){
        printf("✅ PASS: All operations within 5%% error\n");
    } else if(a2_error < 10.0 && ab_error < 10.0 && b2_error < 10.0){
        printf("⚠️  MARGINAL: All operations within 10%% error\n");
    } else {
        printf("❌ FAIL: Errors exceed 10%%\n");
    }
    
    printf("\nNote: Coupling signals are proportional to products, not equal.\n");
    printf("Proportionality constant depends on lens strength and settle dynamics.\n");
    printf("Relative accuracy (a²:a·b:b² ratios) is the key metric.\n");
    
    double ratio_a2_ab = (ab_expected>1e-9) ? (a2_expected/ab_expected) : 0.0;
    double ratio_a2_b2 = (b2_expected>1e-9) ? (a2_expected/b2_expected) : 0.0;
    double ratio_measured_ab = (ab_expected>1e-9) ? (c2_signal/c3_signal) : 0.0;
    double ratio_measured_b2 = (b2_expected>1e-9) ? (c2_signal/c4_signal) : 0.0;
    
    printf("\nRATIO CHECK:\n");
    printf("  Expected a²:a·b:b² = %.2f:%.2f:%.2f\n", a2_expected, ab_expected, b2_expected);
    printf("  Measured ratios:    %.2f:%.2f:%.2f\n", c2_signal, c3_signal, c4_signal);
    printf("  Ratio a²/a·b: expected=%.2f, measured=%.2f\n", ratio_a2_ab, ratio_measured_ab);
    printf("  Ratio a²/b²:  expected=%.2f, measured=%.2f\n", ratio_a2_b2, ratio_measured_b2);
    
    return 0;
}