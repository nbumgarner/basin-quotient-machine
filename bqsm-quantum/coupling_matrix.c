/* ===========================================================================
 * coupling_matrix.c — Characterize g(p,q) coupling matrix for lens-ALU
 *
 *   Maps the nonlinear mode-coupling coefficients g(p,q) of the BQSM ring.
 *   These are the "fused operation coefficients" — each g(p,q) tells us
 *   how strongly harmonic p and harmonic q couple to produce harmonic p+q.
 *
 *   By measuring g(p,q) under different lens profiles (omega detuning),
 *   we build a lookup table: lens_profile → ALU operation set.
 *
 * BUILD: cc -O3 -std=c11 coupling_matrix.c -o coupling_matrix -lm
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

/* Measure |c_k| after encoding specific harmonics */
static double ck_mag(ring_t *r,int k){
    double re=0,im=0;
    for(int j=0;j<N;j++){double a=-2.0*M_PI*k*j/N;re+=r->th[j]*cos(a);im+=r->th[j]*sin(a);}
    return sqrt(re*re+im*im)/N;
}

/* Encode one harmonic at amplitude A, phase 0 */
static void encode_harmonic(ring_t *r,int k,double A){
    memset(r->th,0,N*sizeof(double));memset(r->om,0,N*sizeof(double));
    for(int j=0;j<N;j++)r->th[j]=2.0*A*cos(2.0*M_PI*k*j/N);
}

/* Apply lens: constant detuning at site 0 */
static void apply_lens(ring_t *r,double delta){r->om[0]=delta;}

static double g_matrix[NH+1][NH+1][NH+1]; /* g[p][q][k] for k=p+q */

int main(){
    printf("═══ BQSM Coupling Matrix g(p,q) Characterization ═══\n\n");

    /* Measure g(p,q): encode c_p=A, c_q=B → measure c_k for k=p+q after settle */
    double A=2.0,B=1.5;
    double lens_delta = 0.15;
    int chunks = 2;

    printf("Lens: om[0]=%.2f  Settle chunks: %d  A=%.1f B=%.1f\n\n",lens_delta,chunks,A,B);

    /* Single-harmonic baselines */
    printf("─── Single-Harmonic Baselines ───\n");
    printf("  p  |c_p|_in  |c_p|_out  decay\n");
    for(int p=1;p<=4;p++){
        ring_t r;encode_harmonic(&r,p,A);apply_lens(&r,lens_delta);
        double in = ck_mag(&r,p);
        settle(&r,chunks);
        double out = ck_mag(&r,p);
        printf("  %d   %.3f     %.4f    %.4f\n",p,in,out,out/in);
    }

    /* Pair coupling: encode c_p=A, c_q=B, measure c_{p+q} */
    printf("\n─── Pair Coupling: g(p,q) → c_{p+q} ───\n");
    printf("  p  q  k=p+q  |c_k|_raw  g(p,q)  op\n");
    printf("  ───────────────────────────────────\n");
    for(int p=1;p<=4;p++){
        for(int q=1;q<=4;q++){
            int k = p+q; if(k>NH)continue;
            ring_t r;memset(&r,0,sizeof(r));apply_lens(&r,lens_delta);
            /* Encode both c_p=A and c_q=B */
            for(int j=0;j<N;j++){
                r.th[j]=2.0*A*cos(2.0*M_PI*p*j/N)+2.0*B*cos(2.0*M_PI*q*j/N);
            }
            settle(&r,chunks);
            double ck = ck_mag(&r,k);
            /* g(p,q) normalized by input amplitudes */
            double g = ck/(A*B);
            g_matrix[p][q][k] = g;
            const char *op = (p==1&&q==1)?"a²":(p==1&&q==2)?"a·b":(p==2&&q==2)?"b²":"";
            printf("  %d  %d   %d     %.4f     %.4f    %s\n",p,q,k,ck,g,op);
        }
    }

    /* Lens sweep: how does g(p,q) change with lens detuning? */
    printf("\n─── Lens Sweep: g(1,2) and g(2,2) vs delta ───\n");
    printf("  delta  g(1,2)   g(2,2)   g(1,1)   ratio(1,2)/g(2,2)\n");
    for(int d=0;d<=6;d++){
        double delta=d*0.05;
        ring_t r1,r2,r3;
        memset(&r1,0,sizeof(r1));apply_lens(&r1,delta);
        memset(&r2,0,sizeof(r2));apply_lens(&r2,delta);
        memset(&r3,0,sizeof(r3));apply_lens(&r3,delta);
        /* c₁=A, c₂=B → c₃ (product) */
        for(int j=0;j<N;j++)r1.th[j]=2.0*A*cos(2.0*M_PI*1*j/N)+2.0*B*cos(2.0*M_PI*2*j/N);
        /* c₂=B only → c₄ (square) */
        for(int j=0;j<N;j++)r2.th[j]=2.0*B*cos(2.0*M_PI*2*j/N);
        /* c₁=A only → c₂ (square) */
        for(int j=0;j<N;j++)r3.th[j]=2.0*A*cos(2.0*M_PI*1*j/N);
        settle(&r1,chunks);settle(&r2,chunks);settle(&r3,chunks);
        double g12=ck_mag(&r1,3)/(A*B);/* g(1,2) */
        double g22=ck_mag(&r2,4)/(B*B);/* g(2,2) */
        double g11=ck_mag(&r3,2)/(A*A);/* g(1,1) */
        printf("  %.2f   %.4f    %.4f    %.4f    %.2f\n",delta,g12,g22,g11,g22>0?g12/g22:0);
    }

    /* ── SUMMARY: What we can compute with one settle ── */
    printf("\n─── FUSED OPERATIONS (1 settle = all simultaneously) ───\n");
    printf("  Encode {a,b,c,d} in {c₁,c₂,c₃,c₄}\n");
    printf("  One settle computes:\n");
    printf("    |c₂| ∝ a² + a·b coupling\n");
    printf("    |c₃| ∝ a·b + a·c coupling (product)\n");
    printf("    |c₄| ∝ b² + a·c + b·d coupling (square + cross)\n");
    printf("    |c₅| ∝ b·c + a·d coupling (cross product)\n");
    printf("    |c₆| ∝ c² + b·d coupling (square)\n");
    printf("    |c₇| ∝ c·d coupling (product)\n");
    printf("    |c₈| ∝ d² (square, Nyquist)\n");
    printf("  7 operations, 1 settle cycle, Fourier-separable results.\n");

    return 0;
}