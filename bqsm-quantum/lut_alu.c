/* ===========================================================================
 * lut_alu.c — Lookup-Table Calibrated ALU
 *
 *   Strategy: Sweep (a,b) grid for each lens, measure g(p,q) at each point,
 *   store in LUT. At runtime, bilinear interpolate g(p,q) for arbitrary inputs.
 *
 *   This handles the input-dependence: g(p,q) = f(a,b) is a smooth surface.
 *
 * BUILD: cc -O3 -std=c11 lut_alu.c -o lut_alu -lm
 * ======================================================================== */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define N  16
#define NMASK (N-1)
#define NH  8
#define DT  0.50
#define T_CHUNK 60.0
#define CHUNKS 2

/* Grid resolution */
#define GRID_A  16   /* a ∈ [0.5, 4.5] */
#define GRID_B  16   /* b ∈ [0.5, 4.5] */
#define N_LENS  5

typedef struct { double th[N], om[N], k1[N], k2[N], k3[N], k4[N], tmp[N]; } ring_t;

static void deriv(ring_t *r) {
    for(int i=0;i<N;i++){int ip=(i+1)&NMASK,im=(i-1)&NMASK;
        r->tmp[i]=r->om[i]+sin(r->th[ip]-r->th[i])+sin(r->th[im]-r->th[i]);}
}
static void rk4(ring_t *r){deriv(r);memcpy(r->k1,r->tmp,N*sizeof(double));
    for(int i=0;i<N;i++)r->tmp[i]=r->th[i]+0.5*DT*r->k1[i];deriv(r);memcpy(r->k2,r->tmp,N*sizeof(double));
    for(int i=0;i<N;i++)r->tmp[i]=r->th[i]+0.5*DT*r->k2[i];deriv(r);memcpy(r->k3,r->tmp,N*sizeof(double));
    for(int i=0;i<N;i++)r->tmp[i]=r->th[i]+DT*r->k3[i];deriv(r);memcpy(r->k4,r->tmp,N*sizeof(double));
    for(int i=0;i<N;i++)r->th[i]+=(DT/6.0)*(r->k1[i]+2.0*r->k2[i]+2.0*r->k3[i]+r->k4[i]);}
static void settle(ring_t *r,int c){int s=(int)(T_CHUNK/DT);for(int i=0;i<c;i++)for(int j=0;j<s;j++)rk4(r);}

static double ck_mag(ring_t *r,int k){
    double re=0,im=0;
    for(int j=0;j<N;j++){double a=-2.0*M_PI*k*j/N;re+=r->th[j]*cos(a);im+=r->th[j]*sin(a);}
    return sqrt(re*re+im*im)/N;
}

/* ══════════════════════════════════════════════════════════════════════════
 * LENS PROFILES
 * ════════════════════════════════════════════════════════════════════════ */
typedef struct {
    const char *name;
    double omega[N];
} lens_profile_t;

static lens_profile_t lenses[] = {
    {"IDENTITY",     {0}},
    {"SITE0_SHIFT",  {0.2,0}},
    {"LINEAR_RAMP",  {0.00,0.02,0.04,0.06,0.08,0.10,0.12,0.14, 0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30}},
    {"QUAD",         {0.2,0,0,0, 0.2,0,0,0, -0.2,0,0,0, -0.2,0,0,0}},
    {"ANTIPODAL",    {0.3,0,0,0,0,0,0,0,-0.3,0,0,0,0,0,0,0}},
};

/* ══════════════════════════════════════════════════════════════════════════
 * LUT STRUCTURE: for each lens, grid of g(p,q) values
 * ════════════════════════════════════════════════════════════════════════ */
typedef struct {
    double a_vals[GRID_A];
    double b_vals[GRID_B];
    /* g11 = g(1,1), g12 = g(1,2), g22 = g(2,2) */
    double g11[GRID_A][GRID_B];
    double g12[GRID_A][GRID_B];
    double g22[GRID_A][GRID_B];
} lens_lut_t;

static lens_lut_t luts[N_LENS];

/* ══════════════════════════════════════════════════════════════════════════
 * BUILD LUT: sweep (a,b) grid, measure coupling at each point
 * ════════════════════════════════════════════════════════════════════════ */
static void build_lut(int lens_idx) {
    lens_profile_t *lens = &lenses[lens_idx];
    lens_lut_t *lut = &luts[lens_idx];
    
    printf("  Building LUT for %s... ", lens->name); fflush(stdout);
    
    /* Grid: a ∈ [0.5, 4.5], b ∈ [0.5, 4.5] */
    for(int ia=0; ia<GRID_A; ia++) {
        double a = 0.5 + ia * (4.0 / (GRID_A - 1));
        lut->a_vals[ia] = a;
        for(int ib=0; ib<GRID_B; ib++) {
            double b = 0.5 + ib * (4.0 / (GRID_B - 1));
            if (ia == 0) lut->b_vals[ib] = b;
            
            /* Baseline: single harmonics */
            double base_c[NH+1] = {0};
            for(int p=1; p<=2; p++) {
                ring_t r; memset(&r, 0, sizeof(r));
                for(int i=0;i<N;i++) r.om[i] = lens->omega[i];
                double A = (p==1) ? a : b;
                for(int j=0;j<N;j++)
                    r.th[j] = 2.0*A*cos(2.0*M_PI*p*j/N);
                settle(&r, CHUNKS);
                for(int k=1;k<=NH;k++)
                    base_c[k] += ck_mag(&r, k);
            }
            
            /* Paired: both harmonics */
            ring_t r; memset(&r, 0, sizeof(r));
            for(int i=0;i<N;i++) r.om[i] = lens->omega[i];
            for(int j=0;j<N;j++)
                r.th[j] = 2.0*a*cos(2.0*M_PI*1*j/N) + 2.0*b*cos(2.0*M_PI*2*j/N);
            settle(&r, CHUNKS);
            
            /* Coupling = raw - baseline */
            double raw_c2 = ck_mag(&r, 2);
            double raw_c3 = ck_mag(&r, 3);
            double raw_c4 = ck_mag(&r, 4);
            
            double sig2 = raw_c2 - base_c[2];
            double sig3 = raw_c3 - base_c[3];
            double sig4 = raw_c4 - base_c[4];
            
            /* g = signal / (input products) */
            lut->g11[ia][ib] = (fabs(a) > 0.001) ? sig2 / (a*a) : 0;
            lut->g12[ia][ib] = (fabs(a*b) > 0.001) ? sig3 / (a*b) : 0;
            lut->g22[ia][ib] = (fabs(b) > 0.001) ? sig4 / (b*b) : 0;
        }
    }
    printf("DONE\n");
}

/* ══════════════════════════════════════════════════════════════════════════
 * BILINEAR INTERPOLATION
 * ════════════════════════════════════════════════════════════════════════ */
static double bilinear_interp(const double grid[GRID_A][GRID_B],
                               const double a_vals[GRID_A],
                               const double b_vals[GRID_B],
                               double a, double b) {
    /* Clamp to grid bounds */
    if (a <= a_vals[0]) a = a_vals[0] + 1e-6;
    if (a >= a_vals[GRID_A-1]) a = a_vals[GRID_A-1] - 1e-6;
    if (b <= b_vals[0]) b = b_vals[0] + 1e-6;
    if (b >= b_vals[GRID_B-1]) b = b_vals[GRID_B-1] - 1e-6;
    
    /* Find cell indices */
    int ia = 0, ib = 0;
    while (ia < GRID_A-1 && a > a_vals[ia+1]) ia++;
    while (ib < GRID_B-1 && b > b_vals[ib+1]) ib++;
    
    /* Normalized position within cell [0,1] */
    double ta = (a - a_vals[ia]) / (a_vals[ia+1] - a_vals[ia]);
    double tb = (b - b_vals[ib]) / (b_vals[ib+1] - b_vals[ib]);
    
    /* Bilinear weights */
    double w00 = (1-ta)*(1-tb);
    double w10 = ta*(1-tb);
    double w01 = (1-ta)*tb;
    double w11 = ta*tb;
    
    return w00*grid[ia][ib] + w10*grid[ia+1][ib] + w01*grid[ia][ib+1] + w11*grid[ia+1][ib+1];
}

/* ══════════════════════════════════════════════════════════════════════════
 * DECODE: use interpolated g to compute results
 * ════════════════════════════════════════════════════════════════════════ */
static void decode_with_lut(int lens_idx, double a, double b,
                            double *a2_out, double *ab_out, double *b2_out) {
    lens_lut_t *lut = &luts[lens_idx];
    
    /* Get interpolated coupling coefficients */
    double g11 = bilinear_interp(lut->g11, lut->a_vals, lut->b_vals, a, b);
    double g12 = bilinear_interp(lut->g12, lut->a_vals, lut->b_vals, a, b);
    double g22 = bilinear_interp(lut->g22, lut->a_vals, lut->b_vals, a, b);
    
    /* Measure raw c_k */
    lens_profile_t *lens = &lenses[lens_idx];
    ring_t r; memset(&r, 0, sizeof(r));
    for(int i=0;i<N;i++) r.om[i] = lens->omega[i];
    for(int j=0;j<N;j++)
        r.th[j] = 2.0*a*cos(2.0*M_PI*1*j/N) + 2.0*b*cos(2.0*M_PI*2*j/N);
    settle(&r, CHUNKS);
    
    double raw_c2 = ck_mag(&r, 2);
    double raw_c3 = ck_mag(&r, 3);
    double raw_c4 = ck_mag(&r, 4);
    
    /* Baseline (single harmonics) - could also be interpolated from LUT */
    double base_c[NH+1] = {0};
    for(int p=1; p<=2; p++) {
        ring_t r1; memset(&r1, 0, sizeof(r1));
        for(int i=0;i<N;i++) r1.om[i] = lens->omega[i];
        double A = (p==1) ? a : b;
        for(int j=0;j<N;j++)
            r1.th[j] = 2.0*A*cos(2.0*M_PI*p*j/N);
        settle(&r1, CHUNKS);
        for(int k=1;k<=NH;k++)
            base_c[k] += ck_mag(&r1, k);
    }
    
    /* Signal = raw - baseline */
    double sig2 = raw_c2 - base_c[2];
    double sig3 = raw_c3 - base_c[3];
    double sig4 = raw_c4 - base_c[4];
    
    /* Decode: signal = g * input_product → input_product = signal / g */
    *a2_out = (fabs(g11) > 0.001) ? sig2 / g11 : 0;
    *ab_out = (fabs(g12) > 0.001) ? sig3 / g12 : 0;
    *b2_out = (fabs(g22) > 0.001) ? sig4 / g22 : 0;
}

/* ══════════════════════════════════════════════════════════════════════════
 * TEST: cross-validate LUT on random points not in grid
 * ════════════════════════════════════════════════════════════════════════ */
static void cross_validate(int lens_idx, int n_trials) {
    lens_profile_t *lens = &lenses[lens_idx];
    int n_ok = 0;
    double sum_e_a2=0, sum_e_ab=0, sum_e_b2=0;
    double max_e_a2=0, max_e_ab=0, max_e_b2=0;
    
    static uint64_t rng = 0xDEADBEEF;
    #define RNG() (rng = (rng * 0x9e3779b97f4a7c15ULL + 1), \
        (double)((rng ^ (rng >> 30)) >> 11) * (1.0/9007199254740992.0))
    
    for(int t=0; t<n_trials; t++) {
        double a = 0.75 + RNG() * 3.5;  /* [0.75, 4.25] - off-grid */
        double b = 0.75 + RNG() * 3.5;
        
        double a2_d, ab_d, b2_d;
        decode_with_lut(lens_idx, a, b, &a2_d, &ab_d, &b2_d);
        
        double e_a2 = fabs(a2_d - a*a)/fmax(fabs(a*a),0.01);
        double e_ab = fabs(ab_d - a*b)/fmax(fabs(a*b),0.01);
        double e_b2 = fabs(b2_d - b*b)/fmax(fabs(b*b),0.01);
        
        sum_e_a2 += e_a2; sum_e_ab += e_ab; sum_e_b2 += e_b2;
        if (e_a2 > max_e_a2) max_e_a2 = e_a2;
        if (e_ab > max_e_ab) max_e_ab = e_ab;
        if (e_b2 > max_e_b2) max_e_b2 = e_b2;
        
        if (e_a2 < 0.15 && e_ab < 0.15 && e_b2 < 0.15) n_ok++;
    }
    
    printf("  %s: %d/%d ok (<15%%)  RMS: a²=%.1f%% ab=%.1f%% b²=%.1f%%  max: a²=%.1f%% ab=%.1f%% b²=%.1f%%\n",
           lens->name, n_ok, n_trials,
           sum_e_a2/n_trials*100, sum_e_ab/n_trials*100, sum_e_b2/n_trials*100,
           max_e_a2*100, max_e_ab*100, max_e_b2*100);
}

/* ══════════════════════════════════════════════════════════════════════════
 * PRINT LUT SLICE
 * ════════════════════════════════════════════════════════════════════════ */
static void print_lut_slice(int lens_idx) {
    lens_lut_t *lut = &luts[lens_idx];
    printf("\n  LUT slice for %s (g11, g12, g22):\n", lenses[lens_idx].name);
    printf("  a\\b   ");
    for(int ib=0; ib<GRID_B; ib+=3)
        printf("  b=%.1f   ", lut->b_vals[ib]);
    printf("\n");
    for(int ia=0; ia<GRID_A; ia+=3) {
        printf("  a=%.1f ", lut->a_vals[ia]);
        for(int ib=0; ib<GRID_B; ib+=3) {
            printf(" %.3f/%.3f/%.3f",
                   lut->g11[ia][ib], lut->g12[ia][ib], lut->g22[ia][ib]);
        }
        printf("\n");
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ════════════════════════════════════════════════════════════════════════ */
int main() {
    printf("═══ BQSM LUT-Calibrated ALU ═══\n");
    printf("  Grid: %dx%d = %d points per lens\n", GRID_A, GRID_B, GRID_A*GRID_B);
    printf("  Domain: a,b ∈ [0.5, 4.5]\n");
    printf("  Interpolation: bilinear\n\n");
    
    /* Build all LUTs */
    printf("─── Building LUTs ───\n");
    for(int i=0; i<N_LENS; i++)
        build_lut(i);
    
    /* Print LUT slices */
    for(int i=0; i<N_LENS; i++)
        print_lut_slice(i);
    
    /* Cross-validate */
    printf("\n─── Cross-Validation (200 random off-grid points) ───\n");
    for(int i=0; i<N_LENS; i++)
        cross_validate(i, 200);
    
    /* Show specific test cases */
    printf("\n─── Specific Test Cases ───\n");
    double tests[][2] = {{1.0,1.0}, {2.0,1.5}, {2.5,2.0}, {3.0,2.5}, {3.5,1.5}, {4.0,3.0}};
    int n_tests = sizeof(tests)/sizeof(tests[0]);
    
    for(int ti=0; ti<n_tests; ti++) {
        double a = tests[ti][0], b = tests[ti][1];
        printf("\n  a=%.1f b=%.1f (gt: a²=%.1f ab=%.1f b²=%.1f):\n", a,b,a*a,a*b,b*b);
        for(int li=0; li<N_LENS; li++) {
            double a2,ab,b2;
            decode_with_lut(li, a, b, &a2, &ab, &b2);
            double e_a2=fabs(a2-a*a)/fmax(fabs(a*a),0.01)*100;
            double e_ab=fabs(ab-a*b)/fmax(fabs(a*b),0.01)*100;
            double e_b2=fabs(b2-b*b)/fmax(fabs(b*b),0.01)*100;
            printf("    %-12s a²=%.2f(%.1f%%) ab=%.2f(%.1f%%) b²=%.2f(%.1f%%)\n",
                   lenses[li].name, a2,e_a2, ab,e_ab, b2,e_b2);
        }
    }
    
    return 0;
}