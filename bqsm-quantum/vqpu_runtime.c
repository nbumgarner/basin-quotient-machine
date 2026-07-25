/* ===========================================================================
 * vqpu_runtime.c — vQPU Runtime Implementation (CLEAN v2)
 * ======================================================================== */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "vqpu.h"

#define N 16
#define NMASK (N-1)
#define K 1.0
#define DT 0.50
#define T_CHUNK 60.0
#define FULL_CHUNKS (VQPU_DEFAULT_CHUNKS*4)

#ifdef TEST_VQPU
/* Standalone implementations that match vqpu_bridge.h declarations.
 * DO NOT redefine batch_t — the bridge header owns the typedef. */
static batch_t *g_batch = NULL;

static void deriv(batch_t *b) {
    for (int i=0;i<N;i++){int ip=(i+1)&NMASK,im=(i-1)&NMASK;
        for(int l=0;l<b->L;l++)b->tmp[i*b->L+l]=b->om[i*b->L+l]+K*(sin(b->th[ip*b->L+l]-b->th[i*b->L+l])+sin(b->th[im*b->L+l]-b->th[i*b->L+l]));}
}
static void rk4(batch_t *b) {
    deriv(b);for(int i=0;i<N;i++)for(int l=0;l<b->L;l++)b->k1[i*b->L+l]=b->tmp[i*b->L+l];
    for(int i=0;i<N;i++)for(int l=0;l<b->L;l++)b->tmp[i*b->L+l]=b->th[i*b->L+l]+0.5*DT*b->k1[i*b->L+l];deriv(b);
    for(int i=0;i<N;i++)for(int l=0;l<b->L;l++)b->k2[i*b->L+l]=b->tmp[i*b->L+l];
    for(int i=0;i<N;i++)for(int l=0;l<b->L;l++)b->tmp[i*b->L+l]=b->th[i*b->L+l]+0.5*DT*b->k2[i*b->L+l];deriv(b);
    for(int i=0;i<N;i++)for(int l=0;l<b->L;l++)b->k3[i*b->L+l]=b->tmp[i*b->L+l];
    for(int i=0;i<N;i++)for(int l=0;l<b->L;l++)b->tmp[i*b->L+l]=b->th[i*b->L+l]+DT*b->k3[i*b->L+l];deriv(b);
    for(int i=0;i<N;i++)for(int l=0;l<b->L;l++)b->k4[i*b->L+l]=b->tmp[i*b->L+l];
    for(int i=0;i<N;i++)for(int l=0;l<b->L;l++)b->th[i*b->L+l]+=(DT/6.0)*(b->k1[i*b->L+l]+2.0*b->k2[i*b->L+l]+2.0*b->k3[i*b->L+l]+b->k4[i*b->L+l]);
}
void settle_partial(batch_t *b,int mc){int s=(int)(T_CHUNK/DT);for(int c=0;c<mc;c++)for(int x=0;x<s;x++)rk4(b);}
int winding(const batch_t *b,int l){double s=0;for(int i=0;i<N;i++)s+=remainder(b->th[((i+1)&NMASK)*b->L+l]-b->th[i*b->L+l],2.0*M_PI);return(int)lround(s/(2.0*M_PI));}
void set_twisted(batch_t *b,int l,int q){for(int i=0;i<N;i++)b->th[i*b->L+l]=fmod(2.0*M_PI*q*i/N,2.0*M_PI);}
void apply_fourier_kick(batch_t *b,int l,int km,double a,double p){for(int j=0;j<N;j++)b->th[j*b->L+l]+=a*cos(2.0*M_PI*km*j/N+p);}
void apply_kick(batch_t *b,int l,int site,double amp){int w=3;for(int k=-(w/2);k<=w/2;k++)b->th[((site+k)&NMASK)*b->L+l]+=amp;}
void fft_snapshot(const batch_t *b,int l,double *cr,double *ci){for(int k=0;k<N;k++){double re=0,im=0;for(int j=0;j<N;j++){double a=-2.0*M_PI*k*j/N;re+=b->th[j*b->L+l]*cos(a);im+=b->th[j*b->L+l]*sin(a);}cr[k]=re/N;ci[k]=im/N;}}
batch_t *batch_new(int L){batch_t *b=calloc(1,sizeof(*b));b->L=L;size_t sz=(size_t)N*L*sizeof(double);b->th=calloc(1,sz);b->om=calloc(1,sz);b->k1=calloc(1,sz);b->k2=calloc(1,sz);b->k3=calloc(1,sz);b->k4=calloc(1,sz);b->tmp=calloc(1,sz);return b;}
void batch_free(batch_t *b){if(!b)return;free(b->th);free(b->om);free(b->k1);free(b->k2);free(b->k3);free(b->k4);free(b->tmp);free(b);}
static batch_t *get_batch(int n){if(!g_batch||g_batch->L<n){if(g_batch)batch_free(g_batch);g_batch=batch_new(n>0?n:1);}return g_batch;}
#endif

/* ─── Helpers ──────────────────────────────────────────────────────────── */
static void _sync_ring_to_batch(vring_t *r, batch_t *b, int l) {
    for (int j=0;j<VQPU_N;j++){b->th[j*b->L+l]=r->th[j];b->om[j*b->L+l]=r->om[j];}
}
static void _sync_batch_to_ring(vring_t *r, batch_t *b, int l) {
    for (int j=0;j<VQPU_N;j++){r->th[j]=b->th[j*b->L+l];r->om[j]=b->om[j*b->L+l];}
}

/* ══════════════════════════════════════════════════════════════════════════
 * GATE DISPATCH (single clean switch)
 * ══════════════════════════════════════════════════════════════════════════ */

int vqpu_gate_dispatch(vcircuit_t *c, vgate_t *gate) {
    int t = gate->target;
    if (t<0||t>=c->n_rings) return -1;
    vring_t *r = &c->rings[t];
    vparam_t *p = &gate->params;
    int chunks = p->settle_chunks>0 ? p->settle_chunks : VQPU_DEFAULT_CHUNKS;
    batch_t *b = get_batch(c->n_rings);

    /* Always sync ring→batch before gate */
    _sync_ring_to_batch(r, b, t);

    switch (gate->op) {
    case VOP_INIT:
        /* set_twisted to |q⟩ — bypass gradient flow entirely.
         * The only way to leave |0⟩. Uses p->amp as target winding. */
        {
            int q_target = (int)lround(p->amp);
            if (q_target<0) q_target=0;
            if (q_target>VQPU_Q_MAX) q_target=VQPU_Q_MAX;
            set_twisted(b, t, q_target);
            for (int j=0;j<VQPU_N;j++) r->th[j]=fmod(2.0*M_PI*q_target*j/VQPU_N,2.0*M_PI);
            r->q = q_target;
        }
        break;
    case VOP_HAD:
        apply_fourier_kick(b, t, 2, 5.0, 0.0);  /* k=2 escapes deep basins */
        settle_partial(b, chunks);
        break;
    case VOP_PHASE:
        for (int j=0;j<VQPU_N;j++) r->th[j] += p->phase;
        _sync_ring_to_batch(r, b, t);
        break;
    case VOP_RAMP:
        apply_fourier_kick(b, t, p->k_mode, p->amp, p->phase);
        settle_partial(b, chunks);
        break;
    case VOP_LENS:
        r->om[p->site] += p->delta;
        _sync_ring_to_batch(r, b, t);
        settle_partial(b, chunks);
        break;
    case VOP_ERASE:
        apply_kick(b, t, p->site, p->amp);
        settle_partial(b, chunks);
        break;
    case VOP_RESET:
        set_twisted(b, t, 0);
        for (int j=0;j<VQPU_N;j++) r->om[j]=0.0;
        for (int j=0;j<VQPU_N;j++) r->th[j]=0.0;
        break;
    case VOP_CNOT: {
        int ctrl = p->control_ring;
        if (ctrl<0||ctrl>=c->n_rings) break;
        _sync_ring_to_batch(&c->rings[ctrl], b, ctrl);
        settle_partial(b, FULL_CHUNKS);
        int q_c = winding(b, ctrl);
        if (q_c != 0) {
            int q_t = winding(b, t);
            if (q_t == 0) {
                /* |0⟩→|1⟩: must use set_twisted (q=0 is absorbing) */
                set_twisted(b, t, 2);
                for (int j=0;j<VQPU_N;j++) r->th[j]=fmod(2.0*M_PI*2.0*j/VQPU_N,2.0*M_PI);
            } else {
                /* |1⟩→|0⟩: perturbation works from non-ground state */
                apply_fourier_kick(b, t, 2, 5.0, M_PI);
                settle_partial(b, chunks);
            }
        }
        break;
    }
    case VOP_MEASURE:
        settle_partial(b, FULL_CHUNKS);
        r->q = winding(b, t);
        r->measured = 1;
        c->measures[t] = r->q;
        break;
    case VOP_FOURIER:
        apply_fourier_kick(b, t, p->k_mode, p->amp, p->phase);
        settle_partial(b, chunks);
        break;
    case VOP_SWEEP:
        vqpu_adiabatic_sweep(c, t, p->sweep_delta_start, p->sweep_delta_end, p->sweep_steps);
        return 0;
    default: break;
    }

    /* Read batch→ring */
    _sync_batch_to_ring(r, b, t);
    if (gate->op!=VOP_PHASE && gate->op!=VOP_MEASURE)
        r->q = winding(b, t);
    c->step_count += chunks;
    c->total_sim_time += (double)chunks*VQPU_T_CHUNK;
    return 0;
}

/* ─── Circuit run ──────────────────────────────────────────────────────── */
int vqpu_circuit_run(vcircuit_t *c) {
    if (!c->gap_table_loaded) vqpu_load_gap_table(c);
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    for (int g=0;g<c->n_gates;g++) vqpu_gate_dispatch(c,&c->gates[g]);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    c->total_wall_ms=(t1.tv_sec-t0.tv_sec)*1000.0+(t1.tv_nsec-t0.tv_nsec)/1e6;
    return 0;
}

/* ─── Lifecycle ────────────────────────────────────────────────────────── */
void vqpu_circuit_init(vcircuit_t *c,int nr){memset(c,0,sizeof(*c));c->n_rings=nr;c->rings=calloc(nr,sizeof(vring_t));c->gates=calloc(VQPU_MAX_GATES,sizeof(vgate_t));c->measures=calloc(nr,sizeof(int));for(int r=0;r<nr;r++){vqpu_ring_init(&c->rings[r]);c->measures[r]=-1;}}
void vqpu_circuit_free(vcircuit_t *c){free(c->rings);free(c->gates);free(c->measures);memset(c,0,sizeof(*c));}
void vqpu_circuit_reset(vcircuit_t *c){for(int r=0;r<c->n_rings;r++){vqpu_ring_init(&c->rings[r]);c->measures[r]=-1;}c->n_gates=0;c->step_count=0;}
void vqpu_circuit_add_gate(vcircuit_t *c,vopcode_t op,int t,vparam_t p){if(c->n_gates>=VQPU_MAX_GATES)return;c->gates[c->n_gates++]=(vgate_t){op,t,p};}

/* ─── Fourier state ────────────────────────────────────────────────────── */
void vqpu_fourier_snapshot(vcircuit_t *c,int r){if(r<0||r>=c->n_rings)return;batch_t *b=get_batch(c->n_rings);_sync_ring_to_batch(&c->rings[r],b,r);fft_snapshot(b,r,c->rings[r].c_real,c->rings[r].c_imag);double pw=0;for(int k=1;k<=VQPU_N_HALF;k++){double cr=c->rings[r].c_real[k],ci=(k==VQPU_N_HALF)?0.0:c->rings[r].c_imag[k];pw+=cr*cr+ci*ci;}c->rings[r].c_power=pw;}

/* ─── Gap table ────────────────────────────────────────────────────────── */
double vqpu_gap(double delta,int q){int aq=abs(q);if(aq>VQPU_Q_MAX)return 0.0;double dc=VQPU_DELTA_CRIT[aq];if(delta>=dc)return 0.0;return VQPU_GAP_0*(1.0-delta/dc);}
void vqpu_load_gap_table(vcircuit_t *c){for(int di=0;di<16;di++){double d=0.025*di;for(int q=0;q<=VQPU_Q_MAX;q++)c->gap_table[di][q]=vqpu_gap(d,q);}c->gap_table_loaded=1;}
double vqpu_adiabatic_step(double delta,double gm){return gm*gm*VQPU_MARGIN;}
int vqpu_adiabatic_sweep(vcircuit_t *c,int r,double ds,double de,int ms){if(r<0||r>=c->n_rings)return-1;if(!c->gap_table_loaded)vqpu_load_gap_table(c);int st=0;double d=ds;while(st<ms&&d<de){double gm=vqpu_gap(d,abs(c->rings[r].q));if(gm<1e-6)break;double dd=vqpu_adiabatic_step(d,gm);d+=dd;if(d>de)d=de;c->rings[r].om[0]+=dd;_sync_ring_to_batch(&c->rings[r],get_batch(c->n_rings),r);settle_partial(get_batch(c->n_rings),1);_sync_batch_to_ring(&c->rings[r],get_batch(c->n_rings),r);c->rings[r].q=winding(get_batch(c->n_rings),r);st++;}c->step_count+=st;return st;}

/* ─── Visibility ───────────────────────────────────────────────────────── */
double vqpu_interference_visibility(vcircuit_t *c,int r,int qi,double pa,int ps,int ch){
    if(r<0||r>=c->n_rings)return 0.0;
    int *qf=calloc(ps,sizeof(int));int qmi=999,qma=-999;
    batch_t *b=get_batch(c->n_rings);
    for(int i=0;i<ps;i++){
        double phi=2.0*M_PI*i/ps;
        /* Direct batch: twisted state + c_2 kick + settle */
        set_twisted(b,r,qi);
        apply_fourier_kick(b,r,2,pa,phi);
        settle_partial(b,ch);
        int q=winding(b,r);
        qf[i]=q;if(q<qmi)qmi=q;if(q>qma)qma=q;
    }
    double V=(double)(qma-qmi)/(2.0*VQPU_Q_MAX);
    c->rings[r].visibility=V;c->visibility=V;
    free(qf);return V;
}

/* ─── Diagnostics ──────────────────────────────────────────────────────── */
void vqpu_print_spectrum(vcircuit_t *c,int r){if(r<0||r>=c->n_rings)return;vqpu_fourier_snapshot(c,r);printf("─── Ring %d: q=%d power=%.4f ───\n",r,c->rings[r].q,c->rings[r].c_power);for(int k=1;k<=VQPU_N_HALF;k++){double cr=c->rings[r].c_real[k],ci=(k==VQPU_N_HALF)?0.0:c->rings[r].c_imag[k];printf("  k=%d |c|=%.4f re=%+.4f im=%+.4f\n",k,sqrt(cr*cr+ci*ci),cr,ci);}}
void vqpu_print_circuit_stats(vcircuit_t *c){printf("═══ vQPU: %d rings %d gates %d steps sim=%.1f wall=%.1fms V=%.4f gaps=%s ═══\n",c->n_rings,c->n_gates,c->step_count,c->total_sim_time,c->total_wall_ms,c->visibility,c->gap_table_loaded?"yes":"no");}
void vqpu_print_measures(vcircuit_t *c){printf("─── Measures ───\n");for(int r=0;r<c->n_rings;r++)printf("  R%d: q=%d\n",r,c->measures[r]);}

/* ══════════════════════════════════════════════════════════════════════════
 * TEST HARNESS
 * ══════════════════════════════════════════════════════════════════════════ */
#ifdef TEST_VQPU
int main(int argc,char **argv){
    int nr=(argc>1)?atoi(argv[1]):4;
    printf("═══ vQPU Runtime v2 ═══\n\n");

    /* Test 1: single INIT→HAD→MEASURE */
    printf("─── Single gate ───\n");
    vcircuit_t c0;vqpu_circuit_init(&c0,1);
    vqpu_circuit_add_gate(&c0,VOP_INIT,0,(vparam_t){.amp=2.0});
    vqpu_circuit_add_gate(&c0,VOP_HAD,0,(vparam_t){.settle_chunks=4});
    vqpu_circuit_add_gate(&c0,VOP_MEASURE,0,(vparam_t){0});
    vqpu_circuit_run(&c0);
    printf("  INIT(q=2)→HAD→MEASURE: q=%d\n",c0.measures[0]);
    vqpu_circuit_free(&c0);

    /* Test 2: Interference visibility */
    printf("\n─── Interference Visibility ───\n");
    vcircuit_t c1;vqpu_circuit_init(&c1,1);
    double V=vqpu_interference_visibility(&c1,0,2,5.0,64,4);
    printf("  V=%.4f %s\n",V,V>=0.50?"✓ VALIDATED":V>=0.17?"~ MARGINAL":"✗ INVALIDATED");
    vqpu_circuit_free(&c1);

    /* Test 3: GHZ — binary encoding |0⟩=q=0, |1⟩=q=2 */
    printf("\n─── GHZ Circuit (%d rings, |0⟩=q=0 |1⟩=q=2) ───\n",nr);
    vcircuit_t c2;vqpu_circuit_init(&c2,nr);
    /* All rings start at |0⟩ (q=0, default) */
    /* Set ring 0 to |1⟩ (q=2) for HAD target */
    vqpu_circuit_add_gate(&c2,VOP_INIT,0,(vparam_t){.amp=2.0});
    vqpu_circuit_add_gate(&c2,VOP_HAD,0,(vparam_t){.settle_chunks=4});
    /* CNOT chain: if R0 is non-zero, flip each target from |0⟩ to |1⟩ */
    for(int r=0;r<nr-1;r++)
        vqpu_circuit_add_gate(&c2,VOP_CNOT,r+1,(vparam_t){.control_ring=0,.settle_chunks=4});
    for(int r=0;r<nr;r++)
        vqpu_circuit_add_gate(&c2,VOP_MEASURE,r,(vparam_t){0});
    vqpu_circuit_run(&c2);
    vqpu_print_measures(&c2);
    printf("  Logical: ");
    for(int r=0;r<nr;r++) printf("%s ", c2.measures[r]!=0?"|1⟩":"|0⟩");
    printf("\n  All match: %s\n", 
        (c2.measures[0]!=0)==(c2.measures[1]!=0)&&
        (c2.measures[0]!=0)==(c2.measures[2]!=0)&&
        (c2.measures[0]!=0)==(c2.measures[3]!=0) ? "✓ YES":"✗ NO");
    vqpu_print_circuit_stats(&c2);
    vqpu_circuit_free(&c2);

    printf("\n═══ Done ═══\n");
#ifdef TEST_VQPU
    if(g_batch)batch_free(g_batch);
#endif
    return 0;
}
#endif