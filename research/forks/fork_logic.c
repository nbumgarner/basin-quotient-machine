/* fork_logic.c — Fork A on the real BQSM fabric.
 * Question: which 3-input Boolean functions can ONE N=16 ring compute when
 * the input bits arrive as a sequence of instructions (kicks), the state q is
 * the register, the lens (site 0, delta) is the program, and READ (winding)
 * is the output?
 *
 * Program  = (start state q0, for each input position k a pair of
 *            instructions (I_k^0, I_k^1) chosen by bit b_k, readout set R).
 * Instr    = NOP or kick(site s in 0..15, amp in {1.0, 2.2})  -> 33 symbols.
 * Output   = [ q_final in R ].
 * Transitions come from build_table() — the selftest-anchored table builder.
 * A physical spot-check re-simulates random programs kick-by-kick on the
 * furnace core and compares to the table-composed prediction (Markov test).
 *
 * Build: gcc -O3 -std=c11 fork_logic.c -o fork_logic -lm
 */
#define main rf_main
#include "../../ring_furnace.c"
#undef main

#define NS   (2*Q_MAX+1)        /* 7 state slots                       */
#define NK   (1 + N*2)          /* NOP + 32 kicks                      */
#define NI   (NK + N*2)         /* + 32 WRITE ramps (site x sign)      */
#define RAMP_W 8                /* WRITE ramp width (sites)            */
static int T[NS][NI];           /* next-state slot, or -1 invalid      */

static void apply_instr(batch_t *b, int l, int I){
    if (I == 0) return;
    if (I < NK) { int s=(I-1)/2, a=(I-1)%2; apply_kick(b,l,s,AMPS[a]); return; }
    int r=I-NK, s=r/2; double sg = (r%2)? -1.0 : 1.0;   /* WRITE: localized 2pi ramp */
    for (int k=0;k<N;k++){ double f = k<RAMP_W ? (double)k/RAMP_W : 1.0;
        b->th[((s+k)&NMASK)*b->L + l] += sg*2.0*M_PI*f; }
}
static void load_tables(double delta, int *alive){
    int tab[TBL_SZ]; build_table(0, delta, tab, alive);
    for (int qs = 0; qs < NS; qs++) {
        T[qs][0] = alive[qs] ? qs : -1;                   /* NOP           */
        for (int s = 0; s < N; s++) for (int a = 0; a < 2; a++) {
            int v = alive[qs] ? tab[TBL_IDX(qs - Q_MAX, s, a)] : DEAD;
            int ok = v > DEAD+1 && v >= -Q_MAX && v <= Q_MAX && alive[v + Q_MAX];
            T[qs][1 + s*2 + a] = ok ? v + Q_MAX : -1;
        }
    }
    /* WRITE rows: measured, not assumed — settle each (q, ramp) on the furnace */
    for (int qs = 0; qs < NS; qs++) {
        for (int I = NK; I < NI; I++) T[qs][I] = -1;
        if (!alive[qs]) continue;
        double att[N]; if (!settle_attractor(delta, qs - Q_MAX, att)) continue;
        int L = NI - NK; batch_t *b = batch_new(L);
        for (int l=0;l<L;l++){ for(int i=0;i<N;i++){ b->th[i*L+l]=att[i]; b->om[i*L+l]=0; }
            b->om[LENS_SITE*L+l]=delta; apply_instr(b,l,NK+l); }
        int lock[LMAX]; settle(b, lock);
        for (int l=0;l<L;l++){ int v=winding(b,l);
            T[qs][NK+l] = (lock[l] && v>=-Q_MAX && v<=Q_MAX && alive[v+Q_MAX]) ? v+Q_MAX : -1; }
        batch_free(b);
    }
}

/* open-addressing hash set of uint32 keys */
typedef struct { uint32_t *k; size_t cap, n; } hset;
static void hs_init(hset *h, size_t cap){ h->cap=cap; h->n=0; h->k=malloc(cap*4); memset(h->k,0xff,cap*4); }
static int hs_add(hset *h, uint32_t key){
    size_t i = (key * 2654435761u) & (h->cap-1);
    while (h->k[i] != 0xffffffffu) { if (h->k[i]==key) return 0; i=(i+1)&(h->cap-1); }
    h->k[i]=key; h->n++; return 1;
}
/* encode a map from (2^m inputs) -> state slot, base 8, 3 bits each */
static inline uint32_t enc_set(uint32_t code, int x, int st){ return (code & ~(7u<<(3*x))) | ((uint32_t)st<<(3*x)); }
static inline int enc_get(uint32_t code, int x){ return (code>>(3*x))&7; }

/* Robustness filter: each (q, instr) re-run R times with N(0,sigma) phase noise
 * on the attractor before the instruction; keep only entries with ALL R matching. */
static int ROBUST_R = 0; static double ROBUST_SIG = 0.15;
static int robust_filter(double delta, const int *alive){
    rng_t r={0xC0FFEEull ^ (uint64_t)(delta*1e6)}; int kept=0, total=0;
    for (int qs=0; qs<NS; qs++){ if(!alive[qs]) continue;
        double att[N]; if(!settle_attractor(delta, qs-Q_MAX, att)) continue;
        int instrs[NI], ni=0; for(int I=1;I<NI;I++) if(T[qs][I]>=0) instrs[ni++]=I;
        int per = LMAX / ROBUST_R;
        for (int i0=0;i0<ni;i0+=per){ int G = ni-i0<per? ni-i0: per; int L=G*ROBUST_R;
            batch_t *b=batch_new(L);
            for(int g=0;g<G;g++) for(int k=0;k<ROBUST_R;k++){ int l=g*ROBUST_R+k;
                for(int i=0;i<N;i++){ b->th[i*L+l]=att[i]+ROBUST_SIG*rng_gauss(&r); b->om[i*L+l]=0; }
                b->om[LENS_SITE*L+l]=delta; apply_instr(b,l,instrs[i0+g]); }
            int lock[LMAX]; settle(b,lock);
            for(int g=0;g<G;g++){ int I=instrs[i0+g], ok=1;
                for(int k=0;k<ROBUST_R;k++){ int l=g*ROBUST_R+k; if(!lock[l] || winding(b,l)+Q_MAX!=T[qs][I]) {ok=0;break;} }
                total++; if(ok) kept++; else T[qs][I]=-1; }
            batch_free(b);
        }
    }
    printf("   robust filter (R=%d, sigma=%.2f): kept %d/%d table entries\n",ROBUST_R,ROBUST_SIG,kept,total);
    return kept;
}
static int run_delta(double delta, FILE *csv){
    int alive[NS]; load_tables(delta, alive);
    if (ROBUST_R) robust_filter(delta, alive);
    int nalive=0; for(int i=0;i<NS;i++) nalive+=alive[i];
    /* level 0: constant maps (one input, the empty word) */
    hset cur; hs_init(&cur, 1<<4);
    for (int q=0;q<NS;q++) if(alive[q]) hs_add(&cur,(uint32_t)q);
    for (int m = 0; m < 3; m++) {          /* add input bit m */
        hset nxt; hs_init(&nxt, 1u<<24);
        int nin = 1<<m;
        for (size_t c=0;c<cur.cap;c++){ uint32_t code=cur.k[c]; if(code==0xffffffffu) continue;
            for (int i0=0;i0<NI;i0++) for (int i1=0;i1<NI;i1++) {
                uint32_t out=0; int ok=1;
                for (int x=0;x<nin && ok;x++){ int st=enc_get(code,x);
                    int a=T[st][i0], b=T[st][i1]; if(a<0||b<0){ok=0;break;}
                    out=enc_set(out,x,a); out=enc_set(out,x+nin,b); }
                if(ok) hs_add(&nxt,out);
            }
        }
        free(cur.k); cur=nxt;
    }
    /* functions: full readout (any subset R of states) and physical readouts */
    static int full[256], nz[256], pos[256];
    memset(full,0,sizeof full); memset(nz,0,sizeof nz); memset(pos,0,sizeof pos);
    int maxdistinct=0;
    for (size_t c=0;c<cur.cap;c++){ uint32_t code=cur.k[c]; if(code==0xffffffffu) continue;
        int used=0; for(int x=0;x<8;x++) used|=1<<enc_get(code,x);
        int d=__builtin_popcount(used); if(d>maxdistinct) maxdistinct=d;
        for (int R=0;R<128;R++){ if(R & ~used) continue;
            int f=0; for(int x=0;x<8;x++) if(R>>enc_get(code,x)&1) f|=1<<x; full[f]=1; }
        int fz=0,fp=0; for(int x=0;x<8;x++){ int q=enc_get(code,x)-Q_MAX; if(q!=0) fz|=1<<x; if(q>0) fp|=1<<x; }
        nz[fz]=1; pos[fp]=1;
    }
    int cf=0,cz=0,cp=0; for(int f=0;f<256;f++){cf+=full[f];cz+=nz[f];cp+=pos[f];}
    printf("delta=%.2f alive=%d  state-maps=%zu  max-distinct-final-states=%d  "
           "coverage: any-readout=%d/256  READ(q!=0)=%d/256  READ(q>0)=%d/256\n",
           delta,nalive,cur.n,maxdistinct,cf,cz,cp);
    if(csv){ fprintf(csv,"%.2f,%d,%zu,%d,%d,%d,%d\n",delta,nalive,cur.n,maxdistinct,cf,cz,cp); }
    /* which functions are missing under any-readout? */
    if(cf<256){ printf("   missing (any-readout):"); int shown=0;
        for(int f=0;f<256&&shown<24;f++) if(!full[f]){printf(" %02x",f);shown++;} printf("%s\n", cf<232?" ...":""); }
    free(cur.k);
    return 0;
}

/* Physical Markov spot-check: random 3-instruction words, kick-by-kick */
static int spotcheck(double delta, int trials, uint64_t seed){
    int alive[NS]; load_tables(delta, alive); if(ROBUST_R) robust_filter(delta, alive);
    rng_t r={seed}; int agree=0, done=0, unlocked=0;
    for(int t=0;t<trials;t++){
        int q0; do q0=(int)(rng_uniform(&r)*NS); while(!alive[q0]);
        int I[3]; int pred=q0, ok=1;
        for(int k=0;k<3;k++){ I[k]=1+(int)(rng_uniform(&r)*(NI-1)); pred=T[pred][I[k]]; if(pred<0){ok=0;break;} }
        if(!ok){ t--; continue; }
        double att[N]; if(!settle_attractor(delta,q0-Q_MAX,att)){ t--; continue; }
        batch_t *b=batch_new(1); for(int i=0;i<N;i++){b->th[i]=att[i]; b->om[i]=0;} b->om[LENS_SITE]=delta;
        int lock[LMAX]; int lk=1;
        double nz=getenv("NOISE")?atof(getenv("NOISE")):0.0;
        for(int k=0;k<3;k++){ for(int i=0;i<N;i++) b->th[i]+=nz*rng_gauss(&r); apply_instr(b,0,I[k]); settle(b,lock); if(!lock[0]) lk=0; }
        int qf=winding(b,0)+Q_MAX; batch_free(b);
        done++; if(!lk) unlocked++; else agree+=(qf==pred);
    }
    printf("   spot-check delta=%.2f: %d/%d physical 3-step runs match table composition (%d unlocked)\n",
           delta,agree,done-unlocked,unlocked);
    return 0;
}

int main(int argc,char**argv){
    if(getenv("ROBUST")) ROBUST_R=atoi(getenv("ROBUST")); if(getenv("RSIG")) ROBUST_SIG=atof(getenv("RSIG"));
    if(argc>1 && !strcmp(argv[1],"spot")){ spotcheck(atof(argv[2]),atoi(argv[3]),0xB0B5EEDull); return 0; }
    FILE *csv=fopen(ROBUST_R?"fork_logic_robust.csv":"fork_logic.csv","w"); fprintf(csv,"delta,alive,state_maps,max_distinct,any_readout,read_nonzero,read_positive\n");
    const double D[]={0.0,0.05,0.10,0.15,0.20,0.35,0.50,0.65};
    for(size_t i=0;i<sizeof D/sizeof*D;i++) run_delta(D[i],csv);
    fclose(csv); return 0;
}
