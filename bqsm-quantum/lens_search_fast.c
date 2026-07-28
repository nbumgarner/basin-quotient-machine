/* ===========================================================================
 * lens_search_fast.c — Fast Lens Evaluation (no iterative search)
 *
 *   Evaluate a predefined set of lens profiles quickly, find the flattest g(p,q)
 *
 * BUILD: cc -O3 -std=c11 lens_search_fast.c -o lens_search_fast -lm
 * ======================================================================== */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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
#define GRID_A  7
#define GRID_B  7

typedef struct { double th[N], om[N], k1[N], k2[N], k3[N], k4[N], tmp[N]; } ring_t;

static void deriv(ring_t *r) {
    for(int i=0;i<N;i++){int ip=(i+1)&NMASK,im=(i-1)&NMASK;
        r->tmp[i]=r->om[i]+sin(r->th[ip]-r->th[i])+sin(r->th[im]-r->th[i]);}
}
static void rk4(ring_t *r){
    deriv(r); memcpy(r->k1,r->tmp,N*sizeof(double));
    for(int i=0;i<N;i++)r->tmp[i]=r->th[i]+0.5*DT*r->k1[i]; deriv(r); memcpy(r->k2,r->tmp,N*sizeof(double));
    for(int i=0;i<N;i++)r->tmp[i]=r->th[i]+0.5*DT*r->k2[i]; deriv(r); memcpy(r->k3,r->tmp,N*sizeof(double));
    for(int i=0;i<N;i++)r->tmp[i]=r->th[i]+DT*r->k3[i]; deriv(r); memcpy(r->k4,r->tmp,N*sizeof(double));
    for(int i=0;i<N;i++)r->th[i]+=(DT/6.0)*(r->k1[i]+2.0*r->k2[i]+2.0*r->k3[i]+r->k4[i]);
}
static void settle(ring_t *r,int c){int s=(int)(T_CHUNK/DT);for(int i=0;i<c;i++)for(int j=0;j<s;j++)rk4(r);}

static double ck_mag(ring_t *r,int k){
    double re=0,im=0;
    for(int j=0;j<N;j++){double a=-2.0*M_PI*k*j/N;re+=r->th[j]*cos(a);im+=r->th[j]*sin(a);}
    return sqrt(re*re+im*im)/N;
}

/* ══════════════════════════════════════════════════════════════════════════
 * G-SURFACE MEASUREMENT (single pass, no iteration)
 * ════════════════════════════════════════════════════════════════════════ */
typedef struct {
    double a_vals[GRID_A];
    double b_vals[GRID_B];
    double g11[GRID_A][GRID_B];
    double g12[GRID_A][GRID_B];
    double g22[GRID_A][GRID_B];
} g_surface_t;

static void measure_g_surface(const double omega[N], g_surface_t *surf) {
    for(int ia=0; ia<GRID_A; ia++) {
        double a = 1.0 + ia * (2.5 / (GRID_A - 1));
        surf->a_vals[ia] = a;
        for(int ib=0; ib<GRID_B; ib++) {
            double b = 1.0 + ib * (2.0 / (GRID_B - 1));
            if (ia == 0) surf->b_vals[ib] = b;
            
            double base_c[NH+1] = {0};
            for(int p=1; p<=2; p++) {
                ring_t r; memset(&r, 0, sizeof(r));
                for(int i=0;i<N;i++) r.om[i] = omega[i];
                double A = (p==1) ? a : b;
                for(int j=0;j<N;j++)
                    r.th[j] = 2.0*A*cos(2.0*M_PI*p*j/N);
                settle(&r, CHUNKS);
                for(int k=1;k<=NH;k++)
                    base_c[k] += ck_mag(&r, k);
            }
            
            ring_t r; memset(&r, 0, sizeof(r));
            for(int i=0;i<N;i++) r.om[i] = omega[i];
            for(int j=0;j<N;j++)
                r.th[j] = 2.0*a*cos(2.0*M_PI*1*j/N) + 2.0*b*cos(2.0*M_PI*2*j/N);
            settle(&r, CHUNKS);
            
            double sig2 = ck_mag(&r, 2) - base_c[2];
            double sig3 = ck_mag(&r, 3) - base_c[3];
            double sig4 = ck_mag(&r, 4) - base_c[4];
            
            surf->g11[ia][ib] = (fabs(a) > 0.001) ? sig2 / (a*a) : 0;
            surf->g12[ia][ib] = (fabs(a*b) > 0.001) ? sig3 / (a*b) : 0;
            surf->g22[ia][ib] = (fabs(b) > 0.001) ? sig4 / (b*b) : 0;
        }
    }
}

static double compute_fitness(const g_surface_t *surf) {
    double sum_grad2 = 0;
    int n_grad = 0;
    int zero_cross = 0;
    
    for(int ia=0; ia<GRID_A-1; ia++) {
        for(int ib=0; ib<GRID_B-1; ib++) {
            double dg11_da = surf->g11[ia+1][ib] - surf->g11[ia][ib];
            double dg11_db = surf->g11[ia][ib+1] - surf->g11[ia][ib];
            double dg12_da = surf->g12[ia+1][ib] - surf->g12[ia][ib];
            double dg12_db = surf->g12[ia][ib+1] - surf->g12[ia][ib];
            double dg22_da = surf->g22[ia+1][ib] - surf->g22[ia][ib];
            double dg22_db = surf->g22[ia][ib+1] - surf->g22[ia][ib];
            
            sum_grad2 += dg11_da*dg11_da + dg11_db*dg11_db
                       + dg12_da*dg12_da + dg12_db*dg12_db
                       + dg22_da*dg22_da + dg22_db*dg22_db;
            n_grad += 6;
            
            if (surf->g11[ia][ib] * surf->g11[ia+1][ib] < 0) zero_cross++;
            if (surf->g11[ia][ib] * surf->g11[ia][ib+1] < 0) zero_cross++;
            if (surf->g12[ia][ib] * surf->g12[ia+1][ib] < 0) zero_cross++;
            if (surf->g12[ia][ib] * surf->g12[ia][ib+1] < 0) zero_cross++;
            if (surf->g22[ia][ib] * surf->g22[ia+1][ib] < 0) zero_cross++;
            if (surf->g22[ia][ib] * surf->g22[ia][ib+1] < 0) zero_cross++;
        }
    }
    
    double rms_grad = sqrt(sum_grad2 / n_grad);
    return 1.0 / (1.0 + 100.0 * rms_grad + 10.0 * zero_cross);
}

/* ══════════════════════════════════════════════════════════════════════════
 * TEST LENS ACCURACY (off-grid, 20 random points)
 * ════════════════════════════════════════════════════════════════════════ */
static uint64_t rng_s = 0xDEADBEEF;
static double rng_unif(void) {
    rng_s = rng_s * 0x9e3779b97f4a7c15ULL + 0xbf58476d1ce4e5b9ULL;
    return (double)((rng_s ^ (rng_s >> 30)) >> 11) * (1.0/9007199254740992.0);
}

static void test_lens_accuracy(const double omega[N], const char *name) {
    int n_ok = 0, n_total = 20;
    double sum_e_a2=0, sum_e_ab=0, sum_e_b2=0;
    
    for(int t=0; t<n_total; t++) {
        double a = 1.5 + rng_unif() * 2.0;
        double b = 1.5 + rng_unif() * 1.5;
        
        double base_c[NH+1] = {0};
        for(int p=1; p<=2; p++) {
            ring_t r; memset(&r, 0, sizeof(r));
            for(int i=0;i<N;i++) r.om[i] = omega[i];
            double A = (p==1) ? a : b;
            for(int j=0;j<N;j++)
                r.th[j] = 2.0*A*cos(2.0*M_PI*p*j/N);
            settle(&r, CHUNKS);
            for(int k=1;k<=NH;k++)
                base_c[k] += ck_mag(&r, k);
        }
        
        ring_t r; memset(&r, 0, sizeof(r));
        for(int i=0;i<N;i++) r.om[i] = omega[i];
        for(int j=0;j<N;j++)
            r.th[j] = 2.0*a*cos(2.0*M_PI*1*j/N) + 2.0*b*cos(2.0*M_PI*2*j/N);
        settle(&r, CHUNKS);
        
        double sig2 = ck_mag(&r, 2) - base_c[2];
        double sig3 = ck_mag(&r, 3) - base_c[3];
        double sig4 = ck_mag(&r, 4) - base_c[4];
        
        double g11 = (fabs(a) > 0.001) ? sig2 / (a*a) : 0;
        double g12 = (fabs(a*b) > 0.001) ? sig3 / (a*b) : 0;
        double g22 = (fabs(b) > 0.001) ? sig4 / (b*b) : 0;
        
        double a2_d = (fabs(g11) > 0.001) ? sig2 / g11 : 0;
        double ab_d = (fabs(g12) > 0.001) ? sig3 / g12 : 0;
        double b2_d = (fabs(g22) > 0.001) ? sig4 / g22 : 0;
        
        double e_a2 = fabs(a2_d - a*a)/fmax(fabs(a*a),0.01);
        double e_ab = fabs(ab_d - a*b)/fmax(fabs(a*b),0.01);
        double e_b2 = fabs(b2_d - b*b)/fmax(fabs(b*b),0.01);
        
        sum_e_a2 += e_a2; sum_e_ab += e_ab; sum_e_b2 += e_b2;
        if (e_a2 < 0.15 && e_ab < 0.15 && e_b2 < 0.15) n_ok++;
    }
    
    printf("  %-20s %d/20 ok  RMS: a²=%.1f%% ab=%.1f%% b²=%.1f%%\n",
           name, n_ok,
           sum_e_a2/n_total*100, sum_e_ab/n_total*100, sum_e_b2/n_total*100);
}

/* ══════════════════════════════════════════════════════════════════════════
 * LENS CANDIDATES
 * ════════════════════════════════════════════════════════════════════════ */
typedef struct { const char *name; double omega[N]; } lens_t;

int main() {
    printf("═══ BQSM Fast Lens Evaluation ═══\n");
    printf("  Grid: %dx%d, Fitness = 1/(1+100*RMS_grad+10*zero_cross)\n\n", GRID_A, GRID_B);
    
    lens_t lenses[] = {
        {"IDENTITY",       {0}},
        {"SITE0_0.1",      {0.1,0}},
        {"SITE0_0.2",      {0.2,0}},
        {"SITE0_0.3",      {0.3,0}},
        {"SITE0_0.4",      {0.4,0}},
        {"SITE0_0.5",      {0.5,0}},
        {"LINEAR_RAMP",    {0.00,0.02,0.04,0.06,0.08,0.10,0.12,0.14, 0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30}},
        {"LINEAR_RAMP_HALF",{0.00,0.01,0.02,0.03,0.04,0.05,0.06,0.07, 0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15}},
        {"QUAD",           {0.2,0,0,0, 0.2,0,0,0, -0.2,0,0,0, -0.2,0,0,0}},
        {"ANTIPODAL",      {0.3,0,0,0,0,0,0,0,-0.3,0,0,0,0,0,0,0}},
        {"ANTIPODAL_HALF", {0.15,0,0,0,0,0,0,0,-0.15,0,0,0,0,0,0,0}},
        {"DUAL_PAIR",      {0.2,0.2,0,0,0,0,0,0,-0.2,-0.2,0,0,0,0,0,0}},
        {"TRIPLET",        {0.2,0.1,0.05,0,0,0,0,0,0,0,0,0,0,0,0,0}},
    };
    int n_lenses = sizeof(lenses)/sizeof(lenses[0]);
    
    printf("─── Lens Fitness (higher = flatter g surfaces) ───\n\n");
    
    double best_fit = 0;
    int best_idx = 0;
    
    for(int i=0; i<n_lenses; i++) {
        g_surface_t surf;
        measure_g_surface(lenses[i].omega, &surf);
        double fit = compute_fitness(&surf);
        
        printf("  %-20s fitness=%.6f", lenses[i].name, fit);
        if (fit > best_fit) {
            best_fit = fit; best_idx = i;
            printf(" ← BEST");
        }
        printf("\n");
        
        test_lens_accuracy(lenses[i].omega, lenses[i].name);
    }
    
    printf("\n═══ BEST LENS: %s (fitness=%.6f) ═══\n", lenses[best_idx].name, best_fit);
    printf("  ω = {");
    for(int i=0;i<N;i++) {
        if (fabs(lenses[best_idx].omega[i]) > 0.001)
            printf(" %.3f", lenses[best_idx].omega[i]);
        else
            printf(" 0");
        if (i < N-1) printf(",");
    }
    printf(" }\n");
    
    return 0;
}