/* ===========================================================================
 * hybrid_alu.c — Hybrid Harmonic ALU: route ops to bands by coupling strength
 *
 *   Strategy: 
 *     - Products (a·b) → low harmonics c₁,c₂ (strong g(1,2)=0.049)
 *     - Squares (a²,b²) → high harmonics c₃,c₄ (strong g(4,4)=0.162)
 *     - Lens profile selects coupling topology
 *     - Baseline subtraction isolates true coupling signal
 *
 *   Lens profiles:
 *     - IDENTITY:      g(1,2) active, g(1,1)=g(2,2)=0
 *     - SITE0_SHIFT:   activates g(1,1), g(2,2)  
 *     - LINEAR_RAMP:   inter-band coupling (c₃→c₁ flow)
 *     - QUAD:          high-harmonic squares boosted
 *
 * BUILD: cc -O3 -std=c11 hybrid_alu.c -o hybrid_alu -lm
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
 * LENS PROFILES — detuning vectors ω[0..15]
 * ════════════════════════════════════════════════════════════════════════ */
typedef struct {
    const char *name;
    const char *desc;
    double omega[N];
    /* Which operations this lens enables */
    int has_square_low;   /* a²,b² via g(1,1),g(2,2) in low band */
    int has_square_high;  /* a²,b² via g(3,3),g(4,4) in high band */
    int has_product_low;  /* a·b via g(1,2) in low band */
    int has_product_high; /* a·b via g(3,4) in high band */
    int has_cross;        /* cross terms a·c, b·d etc */
} lens_profile_t;

static lens_profile_t lenses[] = {
    {"IDENTITY",     "ω=0: baseline, only g(1,2) active",            {0},                0,0,1,0,0},
    {"SITE0_SHIFT",  "ω[0]=0.2: activates g(1,1),g(2,2)",           {0.2,0},           1,0,1,0,0},
    {"LINEAR_RAMP",  "ω[i]=0.02i: inter-band flow, high-harm boost", {0.00,0.02,0.04,0.06,0.08,0.10,0.12,0.14, 0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30}, 1,1,1,1,1},
    {"QUAD",         "ω[0,4,8,12]=0.2: high-harmonic squares",       {0.2,0,0,0, 0.2,0,0,0, -0.2,0,0,0, -0.2,0,0,0}, 0,1,0,1,1},
    {"ANTIPODAL",    "ω[0]=0.3, ω[8]=-0.3: asymmetric coupling",    {0.3,0,0,0,0,0,0,0,-0.3,0,0,0,0,0,0,0}, 1,1,1,0,1},
};
static const int n_lenses = sizeof(lenses)/sizeof(lenses[0]);

/* ══════════════════════════════════════════════════════════════════════════
 * ENCODING STRATEGIES
 * ════════════════════════════════════════════════════════════════════════ */
typedef enum {
    ENC_LOW,    /* a→c₁, b→c₂ — traditional, strong product coupling */
    ENC_HIGH,   /* a→c₃, b→c₄ — high harmonic, strong square coupling */
    ENC_HYBRID, /* a→c₁, b→c₂, c→c₃, d→c₄ — four inputs, 7 fused ops */
    ENC_QUAD    /* a²→c₁, b²→c₂, a·b→c₃, c·d→c₄ — pre-squared encoding */
} encoding_t;

static void encode_input(ring_t *r, encoding_t enc, 
                         double a, double b, double c, double d) {
    memset(r->th, 0, N*sizeof(double));
    memset(r->om, 0, N*sizeof(double));
    
    switch(enc) {
        case ENC_LOW:
            for(int j=0;j<N;j++)
                r->th[j] = 2.0*a*cos(2.0*M_PI*1*j/N) + 2.0*b*cos(2.0*M_PI*2*j/N);
            break;
        case ENC_HIGH:
            for(int j=0;j<N;j++)
                r->th[j] = 2.0*a*cos(2.0*M_PI*3*j/N) + 2.0*b*cos(2.0*M_PI*4*j/N);
            break;
        case ENC_HYBRID:
            for(int j=0;j<N;j++)
                r->th[j] = 2.0*a*cos(2.0*M_PI*1*j/N) + 2.0*b*cos(2.0*M_PI*2*j/N)
                         + 2.0*c*cos(2.0*M_PI*3*j/N) + 2.0*d*cos(2.0*M_PI*4*j/N);
            break;
        case ENC_QUAD:
            /* Pre-compute squares/products, encode results */
            for(int j=0;j<N;j++)
                r->th[j] = 2.0*a*a*cos(2.0*M_PI*1*j/N) + 2.0*b*b*cos(2.0*M_PI*2*j/N)
                         + 2.0*a*b*cos(2.0*M_PI*3*j/N) + 2.0*c*d*cos(2.0*M_PI*4*j/N);
            break;
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * BASELINE MEASUREMENT: single-harmonic decays for subtraction
 * ════════════════════════════════════════════════════════════════════════ */
typedef struct {
    double decay[NH+1];  /* |c_k| when only input harmonic p is active */
} baseline_t;

static void measure_baselines(lens_profile_t *lens, int chunks, 
                              baseline_t baselines[NH+1]) {
    for(int p=1; p<=NH; p++) {
        ring_t r; memset(&r, 0, sizeof(r));
        for(int i=0;i<N;i++) r.om[i] = lens->omega[i];
        
        /* Encode unit amplitude in harmonic p */
        double A = 1.0;
        for(int j=0;j<N;j++)
            r.th[j] = 2.0*A*cos(2.0*M_PI*p*j/N);
        
        settle(&r, chunks);
        
        for(int k=1; k<=NH; k++)
            baselines[p].decay[k] = ck_mag(&r, k);
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * RUN ONE ALU OPERATION: encode → settle → subtract baselines → decode
 * ════════════════════════════════════════════════════════════════════════ */
typedef struct {
    double raw[NH+1];
    double signal[NH+1];      /* baseline-subtracted */
    double coupling_g[NH+1];  /* signal / (input products) */
} alu_result_t;

static void run_alu(lens_profile_t *lens, int chunks,
                    encoding_t enc, 
                    double a, double b, double c, double d,
                    const baseline_t baselines[NH+1],
                    alu_result_t *out) {
    ring_t r; memset(&r, 0, sizeof(r));
    for(int i=0;i<N;i++) r.om[i] = lens->omega[i];
    
    encode_input(&r, enc, a, b, c, d);
    settle(&r, chunks);
    
    for(int k=1; k<=NH; k++) {
        out->raw[k] = ck_mag(&r, k);
        double baseline_sum = 0;
        for(int p=1; p<=NH; p++)
            baseline_sum += baselines[p].decay[k];
        out->signal[k] = out->raw[k] - baseline_sum;
    }
    
    /* Estimate coupling coefficients g(p,q) from signal */
    /* g(1,1) ≈ signal[2]/(a*a)  — a² in low band */
    out->coupling_g[2] = (fabs(a) > 0.001) ? out->signal[2] / (a*a) : 0;
    /* g(1,2) ≈ signal[3]/(a*b)  — a·b in low band */
    out->coupling_g[3] = (fabs(a*b) > 0.001) ? out->signal[3] / (a*b) : 0;
    /* g(2,2) ≈ signal[4]/(b*b)  — b² in low band */
    out->coupling_g[4] = (fabs(b) > 0.001) ? out->signal[4] / (b*b) : 0;
    /* g(3,3) ≈ signal[6]/(c*c)  — c² in high band */
    out->coupling_g[6] = (fabs(c) > 0.001) ? out->signal[6] / (c*c) : 0;
    /* g(3,4) ≈ signal[7]/(c*d)  — c·d in high band */
    out->coupling_g[7] = (fabs(c*d) > 0.001) ? out->signal[7] / (c*d) : 0;
    /* g(4,4) ≈ signal[8]/(d*d)  — d² in high band (Nyquist) */
    out->coupling_g[8] = (fabs(d) > 0.001) ? out->signal[8] / (d*d) : 0;
}

/* ══════════════════════════════════════════════════════════════════════════
 * PRINT ALU RESULT
 * ════════════════════════════════════════════════════════════════════════ */
static void print_result(const char *label, alu_result_t *res) {
    const char *ops[NH+1] = {"", "c₁", "a²", "a·b", "b²", "a·c+b·c", "c²", "c·d", "d²"};
    printf("  %s:\n", label);
    printf("    k  raw      signal    g(p,q)    op\n");
    for(int k=2; k<=8; k++) {
        if (fabs(res->signal[k]) > 0.0001 || fabs(res->coupling_g[k]) > 0.001)
            printf("    %d  %+.4f  %+.4f  %+.4f  %s\n",
                   k, res->raw[k], res->signal[k], res->coupling_g[k], ops[k]);
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ════════════════════════════════════════════════════════════════════════ */
int main() {
    printf("═══ BQSM Hybrid Harmonic ALU ═══\n");
    printf("  Route ops to harmonic bands by coupling strength\n");
    printf("  Low band (c₁,c₂): strong product coupling g(1,2)=0.049\n");
    printf("  High band (c₃,c₄): strong square coupling g(4,4)=0.162\n");
    printf("  Baseline subtraction isolates true coupling\n\n");
    
    const int CHUNKS = 2;
    double a=3.0, b=2.0, c=1.5, d=1.0;
    
    for(int li = 0; li < n_lenses; li++) {
        lens_profile_t *lens = &lenses[li];
        
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("LENS: %s (%s)\n", lens->name, lens->desc);
        printf("  Enables: square_low=%d square_high=%d product_low=%d product_high=%d cross=%d\n",
               lens->has_square_low, lens->has_square_high,
               lens->has_product_low, lens->has_product_high, lens->has_cross);
        
        /* Measure baselines for this lens */
        baseline_t baselines[NH+1];
        measure_baselines(lens, CHUNKS, baselines);
        
        printf("\n  Baselines (|c_k| after settle, unit input in c_p):\n");
        printf("    p  ");
        for(int k=1;k<=6;k++) printf("   c%d  ", k);
        printf("\n");
        for(int p=1; p<=4; p++) {
            printf("    %d  ", p);
            for(int k=1; k<=6; k++)
                printf(" %+.3f ", baselines[p].decay[k]);
            printf("\n");
        }
        
        /* Test different encodings */
        alu_result_t res;
        
        /* ENC_LOW: products in low band */
        run_alu(lens, CHUNKS, ENC_LOW, a, b, c, d, baselines, &res);
        print_result("  ENC_LOW (a→c₁, b→c₂)", &res);
        
        /* ENC_HIGH: squares in high band */
        run_alu(lens, CHUNKS, ENC_HIGH, a, b, c, d, baselines, &res);
        print_result("  ENC_HIGH (a→c₃, b→c₄)", &res);
        
        /* ENC_HYBRID: four inputs, 7 fused ops */
        run_alu(lens, CHUNKS, ENC_HYBRID, a, b, c, d, baselines, &res);
        print_result("  ENC_HYBRID (a,b,c,d→c₁,c₂,c₃,c₄)", &res);
        
        printf("\n");
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
     * PROGRAMMABLE ALU TABLE: same inputs, different lenses → different ops
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("════════════════════════════════════════════════════════════\n");
    printf("PROGRAMMABLE ALU TABLE: a=%.1f b=%.1f c=%.1f d=%.1f\n", a,b,c,d);
    printf("  Lens          | ENC_LOW signal  | ENC_HIGH signal | HYBRID signal\n");
    printf("                | a²  a·b  b²     | a²  a·b  b²     | a² a·b b² c² c·d d²\n");
    printf("  ────────────  ────────────────  ────────────────  ────────────────────\n");
    
    for(int li = 0; li < n_lenses; li++) {
        lens_profile_t *lens = &lenses[li];
        baseline_t baselines[NH+1];
        measure_baselines(lens, CHUNKS, baselines);
        alu_result_t rl, rh, rhy;
        
        run_alu(lens, CHUNKS, ENC_LOW, a, b, c, d, baselines, &rl);
        run_alu(lens, CHUNKS, ENC_HIGH, a, b, c, d, baselines, &rh);
        run_alu(lens, CHUNKS, ENC_HYBRID, a, b, c, d, baselines, &rhy);
        
        printf("  %-12s  %+.3f %+.3f %+.3f   %+.3f %+.3f %+.3f   %+.3f %+.3f %+.3f %+.3f %+.3f %+.3f\n",
               lens->name,
               rl.signal[2], rl.signal[3], rl.signal[4],
               rh.signal[6], rh.signal[7], rh.signal[8],
               rhy.signal[2], rhy.signal[3], rhy.signal[4], rhy.signal[6], rhy.signal[7], rhy.signal[8]);
    }
    
    printf("\n  Ground truth: a²=%.0f a·b=%.0f b²=%.0f c²=%.0f c·d=%.0f d²=%.0f\n",
           a*a, a*b, b*b, c*c, c*d, d*d);
    
    /* ═══════════════════════════════════════════════════════════════════════
     * LENS SWEEP: continuous lens tuning
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("\n\n════════════════════════════════════════════════════════════\n");
    printf("LENS SWEEP: ω[0] from 0→0.5, ENC_LOW, track g(1,1), g(1,2), g(2,2)\n");
    printf("  ω[0]  g(1,1)  g(1,2)  g(2,2)  ratio g(1,2)/g(1,1)\n");
    
    for(double w=0; w<=0.5; w+=0.05) {
        lens_profile_t tmp = *lenses;  /* copy IDENTITY */
        tmp.omega[0] = w;
        tmp.name = "sweep";
        baseline_t bl[NH+1];
        measure_baselines(&tmp, CHUNKS, bl);
        alu_result_t r;
        run_alu(&tmp, CHUNKS, ENC_LOW, 2.0, 1.5, c, d, bl, &r);
        printf("  %.2f  %.4f  %.4f  %.4f  %.2f\n",
               w, r.coupling_g[2], r.coupling_g[3], r.coupling_g[4],
               r.coupling_g[2] > 0.001 ? r.coupling_g[3]/r.coupling_g[2] : 0);
    }
    
    return 0;
}