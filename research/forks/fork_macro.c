/* fork_macro.c — A2 on the macro-ring slice: two register rings A, B and an
 * output ring O (N = 16 each), coupled SITE-TO-SITE, chain A — O — B (O's two
 * neighbours in the macro ring; A and B are not neighbours).
 *
 * Question: the late-selector MUX (b2 ? u : v, selector last) is not robustly
 * computable on one N=16 ring with Nick's ISA (RESULTS_run03.md). With b0 held
 * in A and b1 in B, can one instruction on O, chosen by b2, copy A or B into O
 * — i.e. does site-to-site coupling give a conditional, addressable transfer?
 *
 * Dynamics, ring r, site i (furnace equations + one term):
 *   dθʳᵢ/dt = ωʳᵢ + K[sin(θʳᵢ₊₁−θʳᵢ) + sin(θʳᵢ₋₁−θʳᵢ)] + Σₛ εʳˢᵢ sin(θˢᵢ − θʳᵢ)
 * εʳˢᵢ is symmetric. GEOM=full couples O to A and to B at all 16 sites;
 * GEOM=half couples O to A on sites 0–7 and to B on sites 8–15. EPS sets ε.
 *
 * State = (q_A, q_B, q_O), |q| <= 2 per ring. READ = winding of O only.
 * Instructions (each targets one ring): NOP; ERASE = the furnace's 3-site kick,
 * 16 sites x {1.0, 2.2}; WRITE = apply_ramp() from bqsm_vm.c (+1 only), at
 * every site for each width in RAMP_WIDTHS (default 2..8 and N: the whole
 * localized range of RAMP_AT, so marginal writes are in the alphabet).
 *
 * Lock criterion: the furnace's velocity criterion ONLY (spread < LOCK_TOL
 * within PATIENCE chunks; spread over all 48 sites when coupled, so a locked
 * state means one common frequency). No early-exit: run 03 showed it can
 * declare drifting states locked once the fabric is no longer pristine.
 *
 * Trust anchor: `fork_macro xcheck` runs at ε = 0 and compares ring A's ERASE
 * rows with build_table() (expected IDENTICAL under -DREF_BUILD, whose
 * settle() uses the same velocity-only criterion).
 *
 * Build: gcc -O3 -std=c11 fork_macro.c -o fork_macro -lm   (+ -DREF_BUILD)
 */
#define main rf_main
#include "../../ring_furnace.c"
#undef main

#define NR     3                /* rings per system                          */
#define RA     0
#define RB     1
#define RO     2
#define NSITE  (NR*N)           /* 48 oscillators per system                 */
#define QR     2                /* register range |q| <= QR per ring         */
#define NQ     (2*QR+1)
#define NJ     (NQ*NQ*NQ)       /* 125 joint states                          */
#define NIMAX  512

static const char RNAME[NR] = {'A', 'B', 'O'};
static inline int jidx(int qa, int qb, int qo){ return ((qa+QR)*NQ + (qb+QR))*NQ + (qo+QR); }
static inline void jdec(int j, int *q){ q[RO] = j%NQ - QR; j /= NQ; q[RB] = j%NQ - QR; q[RA] = j/NQ - QR; }

/* ---------------------------------------------------------------------------
 * FABRIC: coupling, lens, batched integrator. SoA: site (r*N+i), lane l at
 * [(r*N+i)*L + l].
 * ------------------------------------------------------------------------- */
static double EPS[NR][NR][N];   /* εʳˢᵢ, symmetric                           */
static int    COUPLED = 0;
static double OMR[NR][N];       /* per-ring lens profile (default pristine)  */
static int    PATIENCE = MAXCHUNK;

typedef struct { int L; double *th, *om, *k1, *k2, *k3, *k4, *tmp; } mbatch;
static mbatch *mb_new(int L){
    mbatch *b = malloc(sizeof *b); b->L = L;
    size_t sz = (size_t)NSITE * L * sizeof(double);
    b->th = malloc(sz); b->om = calloc((size_t)NSITE * L, sizeof(double));
    b->k1 = malloc(sz); b->k2 = malloc(sz); b->k3 = malloc(sz); b->k4 = malloc(sz); b->tmp = malloc(sz);
    if (!b->th || !b->om || !b->k1 || !b->k2 || !b->k3 || !b->k4 || !b->tmp) { fprintf(stderr, "alloc failed\n"); exit(1); }
    return b;
}
static void mb_free(mbatch *b){ free(b->th); free(b->om); free(b->k1); free(b->k2); free(b->k3); free(b->k4); free(b->tmp); free(b); }

/* The furnace's edge-walk deriv per ring (same operation order, so ε = 0 is
 * bit-identical to deriv()), then the inter-ring site-to-site term.         */
static void mderiv(const double *th, const double *om, double *out, int L){
    int M = NSITE * L;
    for (int j = 0; j < M; j++) out[j] = om[j];
    for (int r = 0; r < NR; r++)
        for (int i = 0; i < N; i++) {
            int ii = (r*N + i) * L, jj = (r*N + ((i + 1) & NMASK)) * L;
            for (int l = 0; l < L; l++) {
                double s = KCPL * SIN(th[jj + l] - th[ii + l]);
                out[ii + l] += s; out[jj + l] -= s;
            }
        }
    for (int r = 0; r < NR; r++) for (int s = r + 1; s < NR; s++)
        for (int i = 0; i < N; i++) {
            double e = EPS[r][s][i]; if (e == 0.0) continue;
            int a = (r*N + i) * L, c = (s*N + i) * L;
            for (int l = 0; l < L; l++) {
                double v = e * SIN(th[c + l] - th[a + l]);
                out[a + l] += v; out[c + l] -= v;
            }
        }
}
static void mrk4(mbatch *b){
    int M = NSITE * b->L;
    mderiv(b->th, b->om, b->k1, b->L);
    for (int j = 0; j < M; j++) b->tmp[j] = b->th[j] + 0.5*DT*b->k1[j];
    mderiv(b->tmp, b->om, b->k2, b->L);
    for (int j = 0; j < M; j++) b->tmp[j] = b->th[j] + 0.5*DT*b->k2[j];
    mderiv(b->tmp, b->om, b->k3, b->L);
    for (int j = 0; j < M; j++) b->tmp[j] = b->th[j] +     DT*b->k3[j];
    mderiv(b->tmp, b->om, b->k4, b->L);
    for (int j = 0; j < M; j++) b->th[j] += (DT/6.0)*(b->k1[j] + 2*b->k2[j] + 2*b->k3[j] + b->k4[j]);
}
/* lane residual: velocity spread per ring (uncoupled) or over all 48 sites
 * (coupled: a locked state of a connected system has ONE frequency).        */
static void mspread(mbatch *b, double *sp){
    mderiv(b->th, b->om, b->k1, b->L);
    int L = b->L;
    for (int l = 0; l < L; l++) {
        double worst = 0, glo = 1e300, ghi = -1e300;
        for (int r = 0; r < NR; r++) {
            double lo = 1e300, hi = -1e300;
            for (int i = 0; i < N; i++) { double v = b->k1[(r*N + i)*L + l]; if (v < lo) lo = v; if (v > hi) hi = v; }
            if (hi - lo > worst) worst = hi - lo;
            if (lo < glo) glo = lo;
            if (hi > ghi) ghi = hi;
        }
        sp[l] = COUPLED ? ghi - glo : worst;
    }
}
static int LAST_CHUNKS = 0;      /* chunks the last msettle() ran (all locked or patience) */
static void msettle(mbatch *b, int *lock){
    int steps = (int)(T_CHUNK / DT), L = b->L;
    double *sp = malloc(L * sizeof *sp);
    for (int l = 0; l < L; l++) lock[l] = 0;
    LAST_CHUNKS = PATIENCE;
    for (int c = 0; c < PATIENCE; c++) {
        for (int s = 0; s < steps; s++) mrk4(b);
        mspread(b, sp); int all = 1;
        for (int l = 0; l < L; l++) { lock[l] = sp[l] < LOCK_TOL; all &= lock[l]; }
        if (all) { LAST_CHUNKS = c + 1; break; }
    }
    free(sp);
}
static int mwinding(const mbatch *b, int r, int l){
    double s = 0.0;
    for (int i = 0; i < N; i++) {
        double d = b->th[(r*N + ((i + 1) & NMASK))*b->L + l] - b->th[(r*N + i)*b->L + l];
        s += remainder(d, 2.0*M_PI);
    }
    return (int)lround(s / (2.0*M_PI));
}

/* ---------------------------------------------------------------------------
 * INSTRUCTIONS
 * ------------------------------------------------------------------------- */
enum { K_NOP, K_ERASE, K_WRITE };
typedef struct { int ring, kind, site, width; double amp; } minstr_t;
static minstr_t IS[NIMAX];
static int NI;

/* the furnace's apply_kick, on ring r */
static void kick_r(mbatch *b, int r, int l, int site, double amp){
    for (int k = -(KICK_W/2); k <= KICK_W/2; k++)
        b->th[(r*N + ((site + k) & NMASK))*b->L + l] += amp;
}
/* apply_ramp — bqsm_vm.c's WRITE_INC / RAMP_AT, verbatim logic, on ring r */
static void ramp_r(mbatch *b, int r, int l, int center_site, int ramp_width){
    if (ramp_width < 2) ramp_width = 2;
    for (int i = 0; i < N; i++) {
        int dist = (i - center_site) & NMASK;
        if (dist > N/2) dist -= N;
        double frac = (double)dist / ramp_width;
        if (frac >  0.5) frac =  0.5;
        if (frac < -0.5) frac = -0.5;
        b->th[(r*N + i)*b->L + l] += 2.0 * M_PI * frac;
    }
}
static void apply_m(mbatch *b, int l, int I){
    const minstr_t *x = &IS[I];
    if (x->kind == K_ERASE) kick_r(b, x->ring, l, x->site, x->amp);
    else if (x->kind == K_WRITE) ramp_r(b, x->ring, l, x->site, x->width);
}
static void iname(int I, char *buf, size_t n){
    const minstr_t *x = &IS[I];
    if (x->kind == K_NOP) snprintf(buf, n, "NOP");
    else if (x->kind == K_ERASE) snprintf(buf, n, "%c.ERASE(s=%d,a=%.1f)", RNAME[x->ring], x->site, x->amp);
    else snprintf(buf, n, "%c.WRITE+(s=%d,w=%d)", RNAME[x->ring], x->site, x->width);
}
static void build_alphabet(void){
    NI = 0; IS[NI++] = (minstr_t){0, K_NOP, 0, 0, 0};
    char buf[128]; const char *rw = getenv("RAMP_WIDTHS");
    snprintf(buf, sizeof buf, "%s", rw ? rw : "2,3,4,5,6,7,8,16");
    int widths[32], nw = 0;
    for (char *t = strtok(buf, ","); t && nw < 32; t = strtok(NULL, ",")) widths[nw++] = atoi(t);
    for (int r = 0; r < NR; r++) {
        for (int s = 0; s < N; s++) for (int a = 0; a < 2; a++) IS[NI++] = (minstr_t){r, K_ERASE, s, KICK_W, AMPS[a]};
        for (int w = 0; w < nw; w++) {
            if (widths[w] >= N) { IS[NI++] = (minstr_t){r, K_WRITE, 0, widths[w], 0}; continue; }
            for (int s = 0; s < N; s++) IS[NI++] = (minstr_t){r, K_WRITE, s, widths[w], 0};
        }
    }
    if (NI > NIMAX) { fprintf(stderr, "alphabet %d > NIMAX\n", NI); exit(2); }
}

/* ---------------------------------------------------------------------------
 * GEOMETRY
 * ------------------------------------------------------------------------- */
static double EPS_V = 0.0; static const char *GEOM = "half";
static void set_geometry(void){
    memset(EPS, 0, sizeof EPS);
    for (int i = 0; i < N; i++) {
        int toA = 1, toB = 1;
        if (!strcmp(GEOM, "half")) { toA = i < N/2; toB = i >= N/2; }
        if (toA) EPS[RA][RO][i] = EPS[RO][RA][i] = EPS_V;
        if (toB) EPS[RB][RO][i] = EPS[RO][RB][i] = EPS_V;
        if (getenv("EPS_AB")) EPS[RA][RB][i] = EPS[RB][RA][i] = atof(getenv("EPS_AB"));
    }
    COUPLED = 0;
    for (int r = 0; r < NR; r++) for (int s = 0; s < NR; s++) for (int i = 0; i < N; i++) COUPLED |= EPS[r][s][i] != 0.0;
}

/* ---------------------------------------------------------------------------
 * JOINT ATTRACTORS + TABLE
 * ------------------------------------------------------------------------- */
static int    ALIVE[NJ];
static double ATT[NJ][NSITE];
static int    T[NJ][NIMAX];
static int    JSEL[NJ], ISEL[NIMAX];   /* which (state, instr) entries measure() fills */

static void load_lane(mbatch *b, int l, const double *att){
    for (int k = 0; k < NSITE; k++) { b->th[k*b->L + l] = att[k]; b->om[k*b->L + l] = OMR[k / N][k % N]; }
}
static int joint_of(mbatch *b, int l){
    int q[NR];
    for (int r = 0; r < NR; r++) { q[r] = mwinding(b, r, l); if (q[r] < -QR || q[r] > QR) return -1; }
    return jidx(q[RA], q[RB], q[RO]);
}
static void settle_attractors(void){
    mbatch *b = mb_new(NJ);
    for (int j = 0; j < NJ; j++) { int q[NR]; jdec(j, q);
        for (int r = 0; r < NR; r++) for (int i = 0; i < N; i++) {
            b->th[(r*N + i)*NJ + j] = fmod(2.0*M_PI*q[r]*i/N, 2.0*M_PI);
            b->om[(r*N + i)*NJ + j] = OMR[r][i]; } }
    int *lock = malloc(NJ * sizeof *lock); msettle(b, lock);
    for (int j = 0; j < NJ; j++) {
        ALIVE[j] = lock[j] && joint_of(b, j) == j;
        for (int k = 0; k < NSITE; k++) ATT[j][k] = b->th[k*NJ + j];
    }
    free(lock); mb_free(b);
}
static void measure(void){
    settle_attractors();
    static int jq[NJ*NIMAX], ji[NJ*NIMAX]; int nj = 0;
    for (int j = 0; j < NJ; j++) { T[j][0] = ALIVE[j] ? j : -1;
        for (int I = 1; I < NI; I++) { T[j][I] = -1; if (ALIVE[j] && JSEL[j] && ISEL[I]) { jq[nj] = j; ji[nj] = I; nj++; } } }
    int lock[LMAX];
    for (int j0 = 0; j0 < nj; j0 += LMAX) {
        int L = nj - j0 < LMAX ? nj - j0 : LMAX;
        mbatch *b = mb_new(L);
        for (int l = 0; l < L; l++) { load_lane(b, l, ATT[jq[j0+l]]); apply_m(b, l, ji[j0+l]); }
        msettle(b, lock);
        for (int l = 0; l < L; l++) { int v = joint_of(b, l);
            T[jq[j0+l]][ji[j0+l]] = (lock[l] && v >= 0 && ALIVE[v]) ? v : -1; }
        mb_free(b);
    }
}

/* Robustness: each (state, instr) re-run under N(0, σ) phase noise on all 48
 * sites. Two stages: STAGE1 trials for every entry; entries that pass all of
 * them get the remaining R − STAGE1. PASS is exact for survivors and scaled
 * (passes·R/STAGE1) for stage-1 failures, which are non-robust either way.  */
static int ROBUST_R = 0, STAGE1 = 4; static double RSIG = 0.15;
static int PASS[NJ][NIMAX];
static void run_trials(int *jq, int *ji, int nj, int R, int *ok, rng_t *rg){
    int per = LMAX / R; int lock[LMAX];
    for (int j0 = 0; j0 < nj; j0 += per) {
        int G = nj - j0 < per ? nj - j0 : per, L = G * R;
        mbatch *b = mb_new(L);
        for (int g = 0; g < G; g++) for (int k = 0; k < R; k++) { int l = g*R + k;
            load_lane(b, l, ATT[jq[j0+g]]);
            for (int s = 0; s < NSITE; s++) b->th[s*L + l] += RSIG * rng_gauss(rg);
            apply_m(b, l, ji[j0+g]); }
        msettle(b, lock);
        for (int g = 0; g < G; g++) { int c = 0;
            for (int k = 0; k < R; k++) { int l = g*R + k; c += lock[l] && joint_of(b, l) == T[jq[j0+g]][ji[j0+g]]; }
            ok[j0+g] = c; }
        mb_free(b);
    }
}
static int robust_filter(int *total){
    rng_t rg = {0xA2A2ull ^ (uint64_t)llround(EPS_V*1e6) ^ (getenv("SEED") ? strtoull(getenv("SEED"), NULL, 0)*0x9E3779B97F4A7C15ull : 0)};
    static int jq[NJ*NIMAX], ji[NJ*NIMAX], ok[NJ*NIMAX], jq2[NJ*NIMAX], ji2[NJ*NIMAX], ok2[NJ*NIMAX];
    int nj = 0, n2 = 0, kept = 0;
    for (int j = 0; j < NJ; j++) for (int I = 1; I < NI; I++) { PASS[j][I] = 0; if (T[j][I] >= 0) { jq[nj] = j; ji[nj] = I; nj++; } }
    int s1 = STAGE1 < ROBUST_R ? STAGE1 : ROBUST_R;
    run_trials(jq, ji, nj, s1, ok, &rg);
    for (int k = 0; k < nj; k++) {
        if (ok[k] == s1 && ROBUST_R > s1) { jq2[n2] = jq[k]; ji2[n2] = ji[k]; n2++; }
        else PASS[jq[k]][ji[k]] = ok[k] * ROBUST_R / s1;
    }
    if (n2) run_trials(jq2, ji2, n2, ROBUST_R - s1, ok2, &rg);
    for (int k = 0; k < n2; k++) PASS[jq2[k]][ji2[k]] = s1 + ok2[k];
    for (int k = 0; k < nj; k++) { int j = jq[k], I = ji[k]; if (PASS[j][I] == ROBUST_R) kept++; }
    *total = nj; return kept;
}
static int RAW[NJ][NIMAX];
static void set_level(int k){
    for (int j = 0; j < NJ; j++) for (int I = 0; I < NI; I++)
        T[j][I] = (RAW[j][I] >= 0 && (I == 0 || k <= 0 || PASS[j][I] >= k)) ? RAW[j][I] : -1;
}

/* ---------------------------------------------------------------------------
 * GATES: the physics question directly. For every instruction on O, the
 * outcome q_O from each register context (q_A, q_B) ∈ {0,1}², O starting at
 * q_O = Q0. COPY-A: q_O' depends on q_A only and separates its values;
 * COPY-B likewise; MIX: depends on both. Registers must survive (A, B
 * unchanged) for the entry to count.
 * ------------------------------------------------------------------------- */
static void gates(int q0, int show_all){
    int nA = 0, nB = 0; char nm[64];
    printf("gates: GEOM=%s EPS=%.3f  O starts at q=%+d   (pass counts /%d where measured)\n", GEOM, EPS_V, q0, ROBUST_R);
    int ctx[4]; for (int c = 0; c < 4; c++) ctx[c] = jidx(c & 1, c >> 1, q0);
    printf("   contexts alive (qA,qB)=00,10,01,11: %d %d %d %d\n", ALIVE[ctx[0]], ALIVE[ctx[1]], ALIVE[ctx[2]], ALIVE[ctx[3]]);
    for (int I = 0; I < NI; I++) {
        if (IS[I].ring != RO && I) continue;
        int out[4], ok = 1, minp = ROBUST_R;
        for (int c = 0; c < 4; c++) { int j = ctx[c], v = ALIVE[j] ? RAW[j][I] : -1;
            if (v < 0) { ok = 0; break; }
            int q[NR]; jdec(v, q);
            if (q[RA] != (c & 1) || q[RB] != (c >> 1)) { ok = 0; break; }   /* registers must survive */
            out[c] = q[RO];
            if (I && ROBUST_R && PASS[j][I] < minp) minp = PASS[j][I]; }
        if (!ok) continue;
        int depA = out[0] != out[1] || out[2] != out[3], depB = out[0] != out[2] || out[1] != out[3];
        int copyA = out[0] == out[2] && out[1] == out[3] && out[0] != out[1];
        int copyB = out[0] == out[1] && out[2] == out[3] && out[0] != out[2];
        nA += copyA; nB += copyB;
        if (copyA || copyB || (show_all && (depA || depB))) {
            iname(I, nm, sizeof nm);
            printf("   %-24s qO'(00,10,01,11) = %+d %+d %+d %+d  %s", nm, out[0], out[1], out[2], out[3],
                   copyA ? "COPY-A" : copyB ? "COPY-B" : "MIX");
            if (ROBUST_R) printf("  min pass %d/%d", minp, ROBUST_R);
            printf("\n");
        }
    }
    printf("   => %d COPY-A and %d COPY-B instructions on O\n", nA, nB);
}

/* ---------------------------------------------------------------------------
 * ENUMERATION (run-02 program model on the joint state; READ = q_O).
 * Each level dedups by image vector; the final level projects onto q_O.
 * ------------------------------------------------------------------------- */
typedef struct { uint64_t *k; int32_t *v; size_t cap, n; } h64;
static void h_init(h64 *h, size_t cap){ h->cap = cap; h->n = 0; h->k = malloc(cap*8); h->v = malloc(cap*4); memset(h->k, 0xff, cap*8); }
static void h_free(h64 *h){ free(h->k); free(h->v); }
static inline size_t h_slot(uint64_t key, size_t cap){ return (size_t)((key * 0x9E3779B97F4A7C15ull) >> 17) & (cap - 1); }
static int h_get_or_put(h64 *h, uint64_t key, int32_t v){
    if ((h->n + 1) * 2 > h->cap) { h64 g; h_init(&g, h->cap * 2);
        for (size_t i = 0; i < h->cap; i++) if (h->k[i] != ~0ull) { size_t s = h_slot(h->k[i], g.cap);
            while (g.k[s] != ~0ull) s = (s + 1) & (g.cap - 1);
            g.k[s] = h->k[i]; g.v[s] = h->v[i]; g.n++; }
        h_free(h); *h = g; }
    size_t s = h_slot(key, h->cap);
    while (h->k[s] != ~0ull) { if (h->k[s] == key) return h->v[s]; s = (s + 1) & (h->cap - 1); }
    h->k[s] = key; h->v[s] = v; h->n++; return -1;
}
typedef struct { uint64_t *code; int32_t *par; int16_t *i0, *i1; size_t n, cap; } lev_t;
static void lv_push(lev_t *v, uint64_t c, int32_t p, int a, int b){
    if (v->n == v->cap) { v->cap = v->cap ? v->cap * 2 : 4096;
        v->code = realloc(v->code, v->cap*8); v->par = realloc(v->par, v->cap*4);
        v->i0 = realloc(v->i0, v->cap*2); v->i1 = realloc(v->i1, v->cap*2); }
    v->code[v->n] = c; v->par[v->n] = p; v->i0[v->n] = (int16_t)a; v->i1[v->n] = (int16_t)b; v->n++;
}
static void lv_clear(lev_t *v){ free(v->code); free(v->par); free(v->i0); free(v->i1); memset(v, 0, sizeof *v); }
static inline int cget(uint64_t c, int x){ return (int)((c >> (7*x)) & 127); }

static lev_t LV[3];
static uint8_t *FCOV;          /* [1<<24] final q_O code covered               */
static const int MUX8[8] = {0x35,0x3a,0x53,0x5c,0xa3,0xac,0xc5,0xca};
static int START_RESET = 0;    /* START=reset: programs start at (0,0,0)       */

/* distinct image vectors of a map (nin inputs) over valid instructions */
static int images(uint64_t code, int nin, uint64_t *img, int *rep){
    h64 h; h_init(&h, 1024); int n = 0;
    for (int I = 0; I < NI; I++) {
        uint64_t o = 0; int ok = 1;
        for (int x = 0; x < nin; x++) { int v = T[cget(code, x)][I]; if (v < 0) { ok = 0; break; } o |= (uint64_t)v << (7*x); }
        if (ok && h_get_or_put(&h, o, n) < 0) { img[n] = o; rep[n] = I; n++; }
    }
    h_free(&h); return n;
}
static inline int qo_of(int j){ return j % NQ; }     /* q_O + QR, 0..4 */
typedef struct { size_t l1, l2; int any, nz, pos, mux_any, mux_nz, mux_pos; } mcov_t;
static void codes_to_functions(int *full, int *nz, int *pos){
    for (int f = 0; f < 256; f++) full[f] = nz[f] = pos[f] = 0;
    for (uint32_t c = 0; c < (1u << 24); c++) { if (!FCOV[c]) continue;
        int used = 0, v[8]; for (int x = 0; x < 8; x++) { v[x] = (c >> (3*x)) & 7; used |= 1 << v[x]; }
        for (int R = 0; R < 32; R++) { if (R & ~used) continue;
            int f = 0; for (int x = 0; x < 8; x++) if (R >> v[x] & 1) f |= 1 << x; full[f] = 1; }
        int fz = 0, fp = 0; for (int x = 0; x < 8; x++) { if (v[x] != QR) fz |= 1 << x; if (v[x] > QR) fp |= 1 << x; }
        nz[fz] = 1; pos[fp] = 1; }
}
static mcov_t enumerate(void){
    mcov_t c = {0};
    for (int m = 0; m < 3; m++) lv_clear(&LV[m]);
    if (!FCOV) FCOV = malloc(1u << 24);
    memset(FCOV, 0, 1u << 24);
    for (int j = 0; j < NJ; j++) if (ALIVE[j] && (!START_RESET || j == jidx(0, 0, 0))) lv_push(&LV[0], (uint64_t)j, -1, 0, 0);
    static uint64_t img[NIMAX]; static int rep[NIMAX];
    for (int m = 0; m < 2; m++) {
        int nin = 1 << m; h64 h; h_init(&h, 1 << 16);
        for (size_t p = 0; p < LV[m].n; p++) {
            int n = images(LV[m].code[p], nin, img, rep);
            for (int a = 0; a < n; a++) for (int b2 = 0; b2 < n; b2++) {
                uint64_t ch = img[a] | (img[b2] << (7*nin));
                if (h_get_or_put(&h, ch, (int32_t)LV[m+1].n) < 0) lv_push(&LV[m+1], ch, (int32_t)p, rep[a], rep[b2]);
            }
        }
        h_free(&h);
    }
    c.l1 = LV[1].n; c.l2 = LV[2].n;
    /* final step projected onto q_O: per level-2 map, distinct 4-vectors */
    for (size_t p = 0; p < LV[2].n; p++) {
        uint32_t v[NIMAX]; int nv = 0; uint64_t code = LV[2].code[p];
        uint8_t seen[1 << 12]; memset(seen, 0, sizeof seen);
        for (int I = 0; I < NI; I++) { uint32_t o = 0; int ok = 1;
            for (int x = 0; x < 4; x++) { int t = T[cget(code, x)][I]; if (t < 0) { ok = 0; break; } o |= (uint32_t)qo_of(t) << (3*x); }
            if (ok && !seen[o]) { seen[o] = 1; v[nv++] = o; } }
        for (int a = 0; a < nv; a++) for (int b2 = 0; b2 < nv; b2++) FCOV[v[a] | (v[b2] << 12)] = 1;
    }
    int full[256], nz[256], pos[256]; codes_to_functions(full, nz, pos);
    for (int f = 0; f < 256; f++) { c.any += full[f]; c.nz += nz[f]; c.pos += pos[f]; }
    for (int k = 0; k < 8; k++) { c.mux_any += full[MUX8[k]]; c.mux_nz += nz[MUX8[k]]; c.mux_pos += pos[MUX8[k]]; }
    return c;
}

/* witness for target f: first (level-2 map, i0, i1, R over q_O) that computes
 * it — searched over each map's distinct q_O 4-vectors, not all NI² pairs.  */
typedef struct { int ok, j0, i0[3], i1[3], R; } mprog_t;
static mprog_t find_witness(int f){
    mprog_t pr = {0};
    static int vv[NIMAX][4], vrep[NIMAX];
    for (size_t p = 0; p < LV[2].n && !pr.ok; p++) {
        uint64_t code = LV[2].code[p]; int nv = 0; uint8_t seen[1 << 12]; memset(seen, 0, sizeof seen);
        for (int I = 0; I < NI; I++) { int v[4], ok = 1; uint32_t o = 0;
            for (int x = 0; x < 4; x++) { int t = T[cget(code, x)][I]; if (t < 0) { ok = 0; break; } v[x] = qo_of(t); o |= (uint32_t)v[x] << (3*x); }
            if (!ok || seen[o]) continue;
            seen[o] = 1;
            memcpy(vv[nv], v, sizeof v); vrep[nv] = I; nv++; }
        for (int a = 0; a < nv && !pr.ok; a++) for (int b2 = 0; b2 < nv && !pr.ok; b2++) {
            int one = 0, zero = 0;               /* q_O values that must read 1 / 0 */
            for (int x = 0; x < 8; x++) { int v = x < 4 ? vv[a][x] : vv[b2][x - 4];
                if (f >> x & 1) one |= 1 << v; else zero |= 1 << v; }
            if (one & zero) continue;
            pr.ok = 1; pr.R = one; pr.i0[2] = vrep[a]; pr.i1[2] = vrep[b2];
            int q2 = (int)p; pr.i0[1] = LV[2].i0[q2]; pr.i1[1] = LV[2].i1[q2];
            int q1 = LV[2].par[q2]; pr.i0[0] = LV[1].i0[q1]; pr.i1[0] = LV[1].i1[q1];
            pr.j0 = (int)LV[0].code[LV[1].par[q1]];
        }
    }
    return pr;
}
static void print_mprog(int f, const mprog_t *pr){
    char a[64], b[64]; int q[NR]; jdec(pr->j0, q);
    printf("   f=0x%02x  start (qA,qB,qO)=(%+d,%+d,%+d)  READ O in {", f, q[RA], q[RB], q[RO]);
    int first = 1; for (int v = 0; v < NQ; v++) if (pr->R >> v & 1) { printf("%s%+d", first ? "" : ",", v - QR); first = 0; }
    printf("}\n");
    for (int k = 0; k < 3; k++) { iname(pr->i0[k], a, sizeof a); iname(pr->i1[k], b, sizeof b);
        printf("      b%d: 0 -> %-26s 1 -> %s\n", k, a, b); }
}
/* physical run of a program, all 8 inputs x K trials, kick by kick */
static int verify(int f, const mprog_t *pr, int K, double sigma, uint64_t seed, int *unlocked){
    rng_t rg = {seed}; int correct = 0, total = 0; *unlocked = 0;
    int per = LMAX / 8; if (per > K) per = K; int lock[LMAX];
    for (int k0 = 0; k0 < K; k0 += per) {
        int G = K - k0 < per ? K - k0 : per, L = 8 * G;
        mbatch *b = mb_new(L); int lk[LMAX];
        for (int l = 0; l < L; l++) { load_lane(b, l, ATT[pr->j0]); lk[l] = 1; }
        for (int s = 0; s < 3; s++) {
            for (int l = 0; l < L; l++) { int x = l & 7;
                for (int i = 0; i < NSITE; i++) b->th[i*L + l] += sigma * rng_gauss(&rg);
                apply_m(b, l, (x >> s & 1) ? pr->i1[s] : pr->i0[s]); }
            msettle(b, lock); for (int l = 0; l < L; l++) lk[l] &= lock[l];
        }
        for (int l = 0; l < L; l++) { int x = l & 7, q = mwinding(b, RO, l);
            int out = (q >= -QR && q <= QR) ? (pr->R >> (q + QR) & 1) : -1;
            total++; if (!lk[l]) (*unlocked)++; correct += out == (f >> x & 1); }
        mb_free(b);
    }
    return correct * 1000 / total;
}

/* PROG: an explicit program, run physically. Clean trace of every input's
 * joint state after each step, then accuracy under N(0, σ) noise.           */
static int find_instr(const char *name){
    char nm[64];
    for (int I = 0; I < NI; I++) { iname(I, nm, sizeof nm); if (!strcmp(nm, name)) return I; }
    fprintf(stderr, "unknown instruction '%s'\n", name); exit(2);
}
static void trace_m(const mprog_t *pr){
    int lock[LMAX]; mbatch *b = mb_new(8);
    for (int l = 0; l < 8; l++) load_lane(b, l, ATT[pr->j0]);
    int lk[8] = {1,1,1,1,1,1,1,1}, st[3][8];
    for (int s = 0; s < 3; s++) {
        for (int l = 0; l < 8; l++) apply_m(b, l, (l >> s & 1) ? pr->i1[s] : pr->i0[s]);
        msettle(b, lock);
        for (int l = 0; l < 8; l++) { lk[l] &= lock[l]; st[s][l] = joint_of(b, l); }
    }
    for (int l = 0; l < 8; l++) {
        printf("   x=%d (b2b1b0=%d%d%d):", l, l >> 2 & 1, l >> 1 & 1, l & 1);
        for (int s = 0; s < 3; s++) { int q[NR];
            if (st[s][l] < 0) { printf("  s%d=(out of range)", s); continue; }
            jdec(st[s][l], q); printf("  s%d=(%+d,%+d,%+d)", s, q[RA], q[RB], q[RO]); }
        int q[NR]; if (st[2][l] >= 0) jdec(st[2][l], q); else q[RO] = 99;
        int out = (q[RO] >= -QR && q[RO] <= QR) ? (pr->R >> (q[RO] + QR) & 1) : -1;
        printf("  READ=%d%s\n", out, lk[l] ? "" : " (NOLOCK)");
    }
    mb_free(b);
}

/* MULTI: how many distinct attractors share one winding triple? Random
 * global phase per ring (φ_A, φ_B, φ_O ~ U[0,2π)) on the ideal twists,
 * settle, keep lanes that lock with the same windings, cluster by the
 * gauge-invariant distance (one common rotation of all 48 phases).          */
static double dist48(const double *a, const double *b){
    double best = 1e30;
    for (int k = 0; k < 256; k++) { double c = 2.0*M_PI*k/256.0, acc = 0;
        for (int i = 0; i < NSITE; i++) { double d = remainder(a[i] - b[i] - c, 2.0*M_PI); acc += d*d; }
        if (acc < best) best = acc; }
    return sqrt(best / NSITE);
}
#define MAXCL 64
static double CL[MAXCL][NSITE]; static int NCL, CLN[MAXCL];
static int classify(const double *th, double tol){
    for (int c = 0; c < NCL; c++) if (dist48(th, CL[c]) < tol) return c;
    return -1;
}
static void multi(int j, int M, double tol){
    rng_t rg = {0x3117ull + (uint64_t)j}; int q[NR]; jdec(j, q);
    NCL = 0; int lock[LMAX], kept = 0, lost = 0;
    for (int m0 = 0; m0 < M; m0 += LMAX) {
        int L = M - m0 < LMAX ? M - m0 : LMAX; mbatch *b = mb_new(L);
        for (int l = 0; l < L; l++) for (int r = 0; r < NR; r++) { double ph = 2.0*M_PI*rng_uniform(&rg);
            for (int i = 0; i < N; i++) { b->th[(r*N + i)*L + l] = 2.0*M_PI*q[r]*i/N + ph; b->om[(r*N + i)*L + l] = OMR[r][i]; } }
        msettle(b, lock);
        for (int l = 0; l < L; l++) {
            if (!lock[l] || joint_of(b, l) != j) { lost++; continue; }
            double th[NSITE]; for (int i = 0; i < NSITE; i++) th[i] = b->th[i*L + l];
            int c = classify(th, tol);
            if (c < 0 && NCL < MAXCL) { memcpy(CL[NCL], th, sizeof th); CLN[NCL] = 0; c = NCL++; }
            if (c >= 0) CLN[c]++;
            kept++;
        }
        mb_free(b);
    }
    printf("(%+d,%+d,%+d): %d/%d starts stay in the triple (%d leave or do not lock); %d distinct attractors:",
           q[RA], q[RB], q[RO], kept, M, lost, NCL);
    for (int c = 0; c < NCL; c++) printf(" %d", CLN[c]);
    printf("   reference ATT is cluster %d\n", ALIVE[j] ? classify(ATT[j], tol) : -9);
}

/* ===========================================================================
 * CONTINUOUS READOUT — O's phasor instead of O's winding.
 *   Z_k = (1/N) Σᵢ exp(i(θᴼᵢ − 2πki/N)), one complex number, mode k = PMODE.
 * Frame: ω = 0 everywhere and the coupling is antisymmetric, so Σθ is a
 * constant of the flow — the lab frame IS a reference clock at the common
 * natural frequency; kicks and noise move Σθ, which is part of the physics.
 * Decoder: ONE half-plane on that one phasor, fitted max-margin on the clean
 * points only. With 8 points in 2-D only a few dozen labellings are
 * half-plane separable, so a hit means the rings placed the points.
 * Contexts after the b0/b1 loads are PHYSICAL states (loads run kick by kick),
 * never reference attractors — a winding triple hides several attractors.
 * ========================================================================= */
static int PMODE = 0;
/* PREF=mean (default): O read against the array's collective phase
 * Θ = (1/48)Σθ. dΣθ/dt = Σω exactly (antisymmetric coupling), so Θ IS the
 * co-rotating frame: with any frequency disorder the whole array drifts at
 * Σω/48 and a lab-frame readout picks up a random rotation; Θ removes it.
 * With Σω = 0 and a flat start, Θ ≡ 0, so clean points are unchanged.
 * PREF=lab reads against the fixed lab frame (run 04's readout).           */
static int PREF_LAB = 0;
static void phasor(const mbatch *b, int l, int k, double *re, double *im){
    double x = 0, y = 0, th0 = 0;
    if (!PREF_LAB) { for (int i = 0; i < NSITE; i++) th0 += b->th[i*b->L + l]; th0 /= NSITE; }
    for (int i = 0; i < N; i++) { double a = b->th[(RO*N + i)*b->L + l] - th0 - 2.0*M_PI*k*i/N; x += cos(a); y += sin(a); }
    *re = x / N; *im = y / N;
}
/* max-margin half-plane for 8 labelled 2-D points; returns margin (<0 if not
 * separable), fills unit direction (ux,uy) and threshold t: label = u·z > t */
static double halfplane(const double *zr, const double *zi, int f, double *ux, double *uy, double *t){
    double best = -1e9;
    for (int a = 0; a < 1440; a++) {
        double c = cos(2.0*M_PI*a/1440), s2 = sin(2.0*M_PI*a/1440), lo1 = 1e9, hi0 = -1e9;
        for (int x = 0; x < 8; x++) { double p = c*zr[x] + s2*zi[x];
            if (f >> x & 1) { if (p < lo1) lo1 = p; } else { if (p > hi0) hi0 = p; } }
        double m = (lo1 - hi0) / 2;
        if (m > best) { best = m; *ux = c; *uy = s2; *t = (lo1 + hi0) / 2; }
    }
    return best;
}
/* physical contexts: start ATT[j0], loads b0 (NOP | A.WRITE), b1 (NOP | B.WRITE) */
static double CTX[4][NSITE];
static void run_loads(int j0, int ia, int ib, double sigma, rng_t *rg, mbatch *b, int lane0, int nl){
    int lock[LMAX];
    for (int l = 0; l < nl; l++) load_lane(b, lane0 + l, ATT[j0]);
    for (int s = 0; s < 2; s++) {
        for (int l = 0; l < nl; l++) { int c = l & 3;
            for (int i = 0; i < NSITE && sigma > 0; i++) b->th[i*b->L + lane0 + l] += sigma * rng_gauss(rg);
            int bit = s == 0 ? (c & 1) : (c >> 1 & 1);
            apply_m(b, lane0 + l, bit ? (s == 0 ? ia : ib) : 0); }
        msettle(b, lock);
    }
}
static void phasor_search(int q0){
    settle_attractors();
    int j0 = jidx(0, 0, q0);
    if (!ALIVE[j0]) { printf("start (0,0,%+d) not alive\n", q0); return; }
    int ia = find_instr("A.WRITE+(s=0,w=16)"), ib = find_instr("B.WRITE+(s=0,w=16)");
    /* clean contexts */
    { mbatch *b = mb_new(4); run_loads(j0, ia, ib, 0.0, NULL, b, 0, 4);
      for (int c = 0; c < 4; c++) { for (int i = 0; i < NSITE; i++) CTX[c][i] = b->th[i*4 + c];
          int q[NR]; for (int r = 0; r < NR; r++) q[r] = mwinding(b, r, c);
          printf("   context b1b0=%d%d -> windings (%+d,%+d,%+d)\n", c >> 1, c & 1, q[RA], q[RB], q[RO]); }
      mb_free(b); }
    /* every O-instruction (and NOP) from every context, clean */
    int oi[NIMAX], no = 0; for (int I = 0; I < NI; I++) if (I == 0 || IS[I].ring == RO) oi[no++] = I;
    static double ZR[NIMAX][4], ZI[NIMAX][4]; static int QO[NIMAX][4], LK[NIMAX][4];
    int lock[LMAX];
    for (int k0 = 0; k0 < no*4; k0 += LMAX) {
        int L = no*4 - k0 < LMAX ? no*4 - k0 : LMAX; mbatch *b = mb_new(L);
        for (int l = 0; l < L; l++) { int g = (k0 + l) / 4, c = (k0 + l) % 4;
            for (int i = 0; i < NSITE; i++) { b->th[i*L + l] = CTX[c][i]; b->om[i*L + l] = OMR[i / N][i % N]; }
            apply_m(b, l, oi[g]); }
        msettle(b, lock);
        for (int l = 0; l < L; l++) { int g = (k0 + l) / 4, c = (k0 + l) % 4;
            phasor(b, l, PMODE, &ZR[g][c], &ZI[g][c]); QO[g][c] = mwinding(b, RO, l); LK[g][c] = lock[l]; }
        mb_free(b);
    }
    /* best pair per MUX function: phasor half-plane vs winding readout */
    char n0[64], n1[64];
    for (int k = 0; k < 8; k++) {
        int f = MUX8[k]; double bm = -1e9; int b0 = -1, b1 = -1, wins = 0, w0 = -1, w1 = -1;
        for (int a = 0; a < no; a++) for (int c2 = 0; c2 < no; c2++) {
            double zr[8], zi[8]; int qo[8], allk = 1;
            for (int x = 0; x < 8; x++) { int g = (x >> 2) ? c2 : a, c = x & 3;
                zr[x] = ZR[g][c]; zi[x] = ZI[g][c]; qo[x] = QO[g][c]; allk &= LK[g][c]; }
            if (!allk) continue;
            double ux, uy, t, m = halfplane(zr, zi, f, &ux, &uy, &t);
            if (m > bm) { bm = m; b0 = a; b1 = c2; }
            int one = 0, zero = 0; for (int x = 0; x < 8; x++) { int v = qo[x] + 8; if (f >> x & 1) one |= 1 << v; else zero |= 1 << v; }
            if (!(one & zero)) { if (!wins) { w0 = a; w1 = c2; } wins++; }
        }
        if (b0 < 0) { printf("   f=0x%02x: no fully locked pair\n", f); continue; }
        iname(oi[b0], n0, sizeof n0); iname(oi[b1], n1, sizeof n1);
        printf("   f=0x%02x  phasor Z%+d best margin %.4f  b2: 0->%s 1->%s   | winding readout: %d pairs work",
               f, PMODE, bm, n0, n1, wins);
        if (wins) { iname(oi[w0], n0, sizeof n0); iname(oi[w1], n1, sizeof n1); printf(" e.g. 0->%s 1->%s", n0, n1); }
        printf("\n");
    }
}

/* noisy verification of one phasor program: decoder fitted on clean points */
static void phasor_verify(int q0, const char *s0, const char *s1, int f, int K){
    settle_attractors();
    int j0 = jidx(0, 0, q0), ia = find_instr("A.WRITE+(s=0,w=16)"), ib = find_instr("B.WRITE+(s=0,w=16)");
    int I2[2] = {find_instr(s0), find_instr(s1)};
    double sigs[] = {0.0, 0.05, 0.10, 0.15, 0.20, 0.30};
    double ux = 0, uy = 0, t = 0, margin = 0;
    for (int si = 0; si < 6; si++) {
        rng_t rg = {0xF00Dull + si}; int correct = 0, total = 0, unl = 0, wcorr = 0;
        double sum[8][2] = {{0}}, sq[8] = {0}; int cnt[8] = {0}; static int RW = -1;
        int per = LMAX / 8, KK = si == 0 ? 1 : K;
        for (int k0 = 0; k0 < KK; k0 += per) {
            int G = KK - k0 < per ? KK - k0 : per, L = 8 * G; mbatch *b = mb_new(L); int lock[LMAX];
            /* loads, noisy */
            for (int l = 0; l < L; l++) load_lane(b, l, ATT[j0]);
            for (int s = 0; s < 3; s++) {
                for (int l = 0; l < L; l++) { int x = l & 7;
                    for (int i = 0; i < NSITE; i++) b->th[i*L + l] += sigs[si] * rng_gauss(&rg);
                    int bit = x >> s & 1, I = s == 0 ? (bit ? ia : 0) : s == 1 ? (bit ? ib : 0) : I2[bit];
                    apply_m(b, l, I); }
                msettle(b, lock);
            }
            if (getenv("PDUMP") && si == atoi(getenv("PDUMP_SIG") ? getenv("PDUMP_SIG") : "1"))
                for (int l = 0; l < L; l++) if ((l & 7) == atoi(getenv("PDUMP"))) { double zr, zi; phasor(b, l, PMODE, &zr, &zi);
                    printf("DUMP %.5f %.5f %d %d %d %d\n", zr, zi, lock[l], mwinding(b, RA, l), mwinding(b, RB, l), mwinding(b, RO, l)); }
            for (int l = 0; l < L; l++) { int x = l & 7; double zr, zi; phasor(b, l, PMODE, &zr, &zi);
                unl += !lock[l]; sum[x][0] += zr; sum[x][1] += zi; sq[x] += zr*zr + zi*zi; cnt[x]++;
                int qo = mwinding(b, RO, l);
                if (si == 0) { if (x == 0) RW = 0; RW |= (f >> x & 1) << (qo + 8); RW |= (!(f >> x & 1)) << (qo + 8 + 16); continue; }
                int wout = (RW >= 0 && !((RW & 0xffff) & (RW >> 16))) ? ((RW >> (qo + 8)) & 1) : -1;
                wcorr += wout == (f >> x & 1);
                total++; correct += ((ux*zr + uy*zi > t) ? 1 : 0) == (f >> x & 1); }
            mb_free(b);
        }
        if (si == 0) {   /* clean run defines the decoder */
            double zr[8], zi[8]; for (int x = 0; x < 8; x++) { zr[x] = sum[x][0]; zi[x] = sum[x][1]; }
            margin = halfplane(zr, zi, f, &ux, &uy, &t);
            printf("   clean points (Re, Im) by x=b0+2b1+4b2:");
            for (int x = 0; x < 8; x++) printf(" (%.3f,%.3f)%s", zr[x], zi[x], f >> x & 1 ? "*" : "");
            printf("\n   decoder: u=(%.3f,%.3f) t=%.4f margin=%.4f  (* = f is 1)\n", ux, uy, t, margin);
            continue;
        }
        double spread = 0; for (int x = 0; x < 8; x++) { double mr = sum[x][0]/cnt[x], mi = sum[x][1]/cnt[x];
            spread += sqrt(fmax(0, sq[x]/cnt[x] - mr*mr - mi*mi)); }
        spread /= 8;
        int wsep = RW >= 0 && !((RW & 0xffff) & (RW >> 16));
        printf("   sigma=%.2f: phasor %d.%d%%  winding %s%d.%d%%  (n=%d)  noise radius %.4f  margin/radius %.2f  unlocked %d\n",
               sigs[si], correct*1000/total/10, correct*1000/total%10, wsep ? "" : "(not separable clean) ",
               wcorr*1000/total/10, wcorr*1000/total%10, total, spread, margin/spread, unl);
    }
}

/* ===========================================================================
 * CONTINUOUS INPUTS — persistent ω drives, one equilibrium, phasor readout.
 * Input k (bit b_k) switches on a zero-sum dipole lens: +U_k at (r1,s1) and
 * −U_k at (r2,s2), same ring or across rings. Σω = 0 keeps the lab frame a
 * valid reference (no co-rotation). Every input pattern must settle LOCKED
 * at windings (0,0,0): no slips, the operating point is the q = 0 basin.
 * With one equilibrium the final state cannot depend on input ORDER, so the
 * "late selector" is no longer a memory question: this asks whether ONE
 * settle computes the MUX (not linearly separable) from three drives,
 * read through ONE phasor and ONE half-plane.
 * ========================================================================= */
typedef struct { int r1, s1, r2, s2; double U; } drive_t;
static void set_drive_lane(mbatch *b, int l, const drive_t *d, int x, double jit, double disorder, rng_t *rg){
    for (int i = 0; i < NSITE; i++) b->om[i*b->L + l] = OMR[i / N][i % N] + (disorder > 0 ? disorder * rng_gauss(rg) : 0.0);
    for (int k = 0; k < 3; k++) if (x >> k & 1) {
        double U = d[k].U * (jit > 0 ? 1.0 + jit * rng_gauss(rg) : 1.0);
        b->om[(d[k].r1*N + d[k].s1)*b->L + l] += U;
        b->om[(d[k].r2*N + d[k].s2)*b->L + l] -= U;
    }
}
static const char RCH[3] = {'A', 'B', 'O'};
static void drive_str(const drive_t *d, char *buf, size_t n){
    size_t k = 0;
    for (int i = 0; i < 3; i++) k += snprintf(buf + k, n - k, "%s%c%d-%c%d:%.3f", i ? ";" : "",
                                               RCH[d[i].r1], d[i].s1, RCH[d[i].r2], d[i].s2, d[i].U);
}
static int parse_drives(const char *spec, drive_t *d){
    char buf[256]; snprintf(buf, sizeof buf, "%s", spec); int k = 0;
    for (char *t = strtok(buf, ";"); t && k < 3; t = strtok(NULL, ";"), k++) {
        char c1, c2; if (sscanf(t, "%c%d-%c%d:%lf", &c1, &d[k].s1, &c2, &d[k].s2, &d[k].U) != 5) return 0;
        d[k].r1 = c1 == 'A' ? RA : c1 == 'B' ? RB : RO; d[k].r2 = c2 == 'A' ? RA : c2 == 'B' ? RB : RO;
    }
    return k == 3;
}
/* settle nc configs x 8 patterns from the flat state; fill phasors + ok flags */
static void eval_drives(const drive_t *cfg, int nc, double (*zr)[8], double (*zi)[8], int (*ok)[8],
                        double sigma, double jit, double disorder, rng_t *rg){
    int lock[LMAX], per = LMAX / 8;
    for (int c0 = 0; c0 < nc; c0 += per) {
        int G = nc - c0 < per ? nc - c0 : per, L = 8 * G; mbatch *b = mb_new(L);
        for (int l = 0; l < L; l++) { int c = c0 + l / 8, x = l & 7;
            for (int i = 0; i < NSITE; i++) b->th[i*L + l] = sigma > 0 ? sigma * rng_gauss(rg) : 0.0;
            set_drive_lane(b, l, &cfg[3*c], x, jit, disorder, rg); }
        msettle(b, lock);
        for (int l = 0; l < L; l++) { int c = c0 + l / 8, x = l & 7;
            phasor(b, l, PMODE, &zr[c][x], &zi[c][x]);
            int w0 = mwinding(b, RA, l) == 0 && mwinding(b, RB, l) == 0 && mwinding(b, RO, l) == 0;
            ok[c][x] = lock[l] && w0 ? 1 : !w0 ? 0 : 2;   /* 1 locked in basin, 0 slipped, 2 in basin but not velocity-locked */ }
        mb_free(b);
    }
}
/* best MUX8 margin of one config's 8 clean points (−1e9 if any pattern slipped) */
static double mux_margin(const double *zr, const double *zi, const int *ok, int *fbest){
    for (int x = 0; x < 8; x++) if (ok[x] != 1) return -1e9;
    double best = -1e9, ux, uy, t;
    for (int k = 0; k < 8; k++) { double m = halfplane(zr, zi, MUX8[k], &ux, &uy, &t); if (m > best) { best = m; *fbest = MUX8[k]; } }
    return best;
}
static int count_separable(const double *zr, const double *zi, double minm){
    int n = 0; double ux, uy, t;
    for (int f = 0; f < 256; f++) n += halfplane(zr, zi, f, &ux, &uy, &t) > minm;
    return n;
}
static void random_drive(drive_t *d, rng_t *rg, double umax){
    d->r1 = (int)(rng_uniform(rg) * 3); d->r2 = (int)(rng_uniform(rg) * 3);
    d->s1 = (int)(rng_uniform(rg) * N);
    do d->s2 = (int)(rng_uniform(rg) * N); while (d->r2 == d->r1 && d->s2 == d->s1);
    d->U = 0.05 * (1 + (int)(rng_uniform(rg) * (umax / 0.05)));
}
static void dsearch(int M, double umax, uint64_t seed){
    rng_t rg = {seed};
    drive_t *cfg = calloc((size_t)3 * M, sizeof(drive_t));
    double (*zr)[8] = malloc(sizeof *zr * M), (*zi)[8] = malloc(sizeof *zi * M); int (*ok)[8] = malloc(sizeof *ok * M);
    for (int c = 0; c < M; c++) for (int k = 0; k < 3; k++) random_drive(&cfg[3*c + k], &rg, umax);
    eval_drives(cfg, M, zr, zi, ok, 0, 0, 0, NULL);
    int nok = 0, nmux = 0; double bm[16]; int bi[16], bf[16], nb = 0;
    for (int c = 0; c < M; c++) {
        int f = 0; double m = mux_margin(zr[c], zi[c], ok[c], &f);
        if (m > -1e8) nok++;
        if (m > 0) nmux++;
        if (m <= 0) continue;
        int pos = nb < 16 ? nb++ : 15; if (pos == 15 && m <= bm[15]) continue;
        bm[pos] = m; bi[pos] = c; bf[pos] = f;
        for (int q = pos; q > 0 && bm[q] > bm[q-1]; q--) { double tm = bm[q]; bm[q] = bm[q-1]; bm[q-1] = tm;
            int ti = bi[q]; bi[q] = bi[q-1]; bi[q-1] = ti; ti = bf[q]; bf[q] = bf[q-1]; bf[q-1] = ti; }
    }
    printf("dsearch: %d random configs (U<=%.2f, seed %llu): %d keep all 8 patterns in the q=0 basin, %d compute a MUX8 (margin>0)\n",
           M, umax, (unsigned long long)seed, nok, nmux);
    char buf[160];
    for (int q = 0; q < nb; q++) { drive_str(&cfg[3*bi[q]], buf, sizeof buf);
        printf("   margin %.4f  f=0x%02x  separable(256, m>0.02)=%3d  %s\n", bm[q], bf[q], count_separable(zr[bi[q]], zi[bi[q]], 0.02), buf); }
    free(cfg); free(zr); free(zi); free(ok);
}
/* coordinate hill-climb on MUX margin: one site +/-1, one amplitude +/-0.05, or one ring swap */
static void dclimb(drive_t *d, int steps){
    double zr[1][8], zi[1][8]; int ok[1][8], f = 0; char buf[160];
    eval_drives(d, 1, zr, zi, ok, 0, 0, 0, NULL);
    double cur = mux_margin(zr[0], zi[0], ok[0], &f);
    for (int st = 0; st < steps; st++) {
        drive_t cand[3*64]; int nc = 0;
        for (int k = 0; k < 3; k++) for (int v = 0; v < 8; v++) {
            drive_t e[3]; memcpy(e, d, sizeof e);
            switch (v) { case 0: e[k].s1 = (e[k].s1 + 1) & NMASK; break; case 1: e[k].s1 = (e[k].s1 + N - 1) & NMASK; break;
                         case 2: e[k].s2 = (e[k].s2 + 1) & NMASK; break; case 3: e[k].s2 = (e[k].s2 + N - 1) & NMASK; break;
                         case 4: e[k].U += 0.05; break; case 5: e[k].U -= 0.05; break;
                         case 6: e[k].r1 = (e[k].r1 + 1) % 3; break; case 7: e[k].r2 = (e[k].r2 + 1) % 3; break; }
            if (e[k].U <= 0 || (e[k].r1 == e[k].r2 && e[k].s1 == e[k].s2)) continue;
            memcpy(&cand[3*nc], e, sizeof e); nc++; }
        double (*czr)[8] = malloc(sizeof *czr * nc), (*czi)[8] = malloc(sizeof *czi * nc); int (*cok)[8] = malloc(sizeof *cok * nc);
        eval_drives(cand, nc, czr, czi, cok, 0, 0, 0, NULL);
        int bi = -1; double bmv = cur; int bfv = f;
        for (int c = 0; c < nc; c++) { int ff = 0; double m = mux_margin(czr[c], czi[c], cok[c], &ff); if (m > bmv + 1e-6) { bmv = m; bi = c; bfv = ff; } }
        free(czr); free(czi); free(cok);
        if (bi < 0) { printf("   climb: local optimum after %d steps\n", st); break; }
        memcpy(d, &cand[3*bi], sizeof(drive_t) * 3); cur = bmv; f = bfv;
        drive_str(d, buf, sizeof buf); printf("   climb %2d: margin %.4f f=0x%02x  %s\n", st, cur, f, buf);
    }
}
/* full verification of one drive program */
static void dverify(const drive_t *d, int K){
    double zr[1][8], zi[1][8]; int ok[1][8], f = 0; char buf[160];
    eval_drives(d, 1, zr, zi, ok, 0, 0, 0, NULL);
    double m = mux_margin(zr[0], zi[0], ok[0], &f), ux = 0, uy = 0, t = 0;
    drive_str(d, buf, sizeof buf);
    printf("drive program %s\n   GEOM=%s EPS=%.3f PATIENCE=%d DT=%.2f mode k=%d\n", buf, GEOM, EPS_V, PATIENCE, DT, PMODE);
    if (m < -1e8) { printf("   a pattern slipped out of the q=0 basin\n"); return; }
    halfplane(zr[0], zi[0], f, &ux, &uy, &t);
    printf("   best MUX f=0x%02x margin %.4f; 256-function half-plane coverage (m>0.02): %d\n", f, m, count_separable(zr[0], zi[0], 0.02));
    printf("   clean phasor by x=b0+2b1+4b2 (|Z|, angle deg):");
    for (int x = 0; x < 8; x++) printf(" (%.3f,%.1f)%s", hypot(zr[0][x], zi[0][x]), atan2(zi[0][x], zr[0][x])*180/M_PI, f >> x & 1 ? "*" : "");
    printf("\n   winding readout: every pattern is (0,0,0) by construction -> constant, computes nothing\n");
    struct { const char *what; double sig, jit, dis; } tests[] = {
        {"phase kicks sigma", 0.15, 0, 0}, {"phase kicks sigma", 0.30, 0, 0}, {"phase kicks sigma", 0.60, 0, 0}, {"phase kicks sigma", 1.00, 0, 0},
        {"omega disorder s", 0, 0, 0.005}, {"omega disorder s", 0, 0, 0.01}, {"omega disorder s", 0, 0, 0.02}, {"omega disorder s", 0, 0, 0.05},
        {"drive jitter", 0, 0.05, 0}, {"drive jitter", 0, 0.10, 0}, {"drive jitter", 0, 0.20, 0} };
    int nt = sizeof tests / sizeof *tests;
    drive_t *cfg = malloc(sizeof(drive_t) * 3 * K);
    for (int c = 0; c < K; c++) memcpy(&cfg[3*c], d, sizeof(drive_t) * 3);
    double (*nzr)[8] = malloc(sizeof *nzr * K), (*nzi)[8] = malloc(sizeof *nzi * K); int (*nok)[8] = malloc(sizeof *nok * K);
    for (int ti = 0; ti < nt; ti++) {
        rng_t rg = {0xD21Eull + ti};
        eval_drives(cfg, K, nzr, nzi, nok, tests[ti].sig, tests[ti].jit, tests[ti].dis, &rg);
        int corr = 0, slip = 0, slow = 0; double rad = 0;
        for (int c = 0; c < K; c++) for (int x = 0; x < 8; x++) {
            slip += nok[c][x] == 0; slow += nok[c][x] == 2;
            corr += ((ux*nzr[c][x] + uy*nzi[c][x] > t) ? 1 : 0) == (f >> x & 1);
            rad += hypot(nzr[c][x] - zr[0][x], nzi[c][x] - zi[0][x]); }
        rad /= 8.0 * K;
        printf("   %-18s %.3f: accuracy %5.1f%%  (n=%d)  mean displacement %.4f  margin/disp %s  slipped %d  not yet locked %d\n",
               tests[ti].what, tests[ti].sig + tests[ti].jit + tests[ti].dis, 100.0*corr/(8*K), 8*K, rad,
               rad > 1e-6 ? (snprintf(buf, sizeof buf, "%.1f", m/rad), buf) : "inf", slip, slow);
    }
    free(cfg); free(nzr); free(nzi); free(nok);
}

/* ===========================================================================
 * LANGEVIN NOISE DURING THE SETTLE
 *   dθ = F(θ) dt + √(2D) dW   on all 48 oscillators.
 * Integrator: the deterministic RK4 step, then the additive increment
 * √(2D·DT)·N(0,1) (Euler–Maruyama splitting; exact in law for additive noise
 * to O(DT) — REF_BUILD re-runs at DT = 0.02 as the check).
 * With noise the state never velocity-locks, so the clock is explicit:
 * T_SET = the time the NOISELESS run of this program needs to lock (measured,
 * all 8 patterns), then the phasor is averaged over a read window T_READ.
 * Decoder: the clean (noiseless, locked) half-plane, never refitted.
 * A lane counts as slipped if any ring's winding is not 0 at the end.
 * ========================================================================= */
static void langevin(const drive_t *d, int K, const double *Ds, int nD, const double *Wr, int nW){
    double zr[1][8], zi[1][8]; int ok[1][8], f = 0; char buf[160];
    eval_drives(d, 1, zr, zi, ok, 0, 0, 0, NULL);
    int tset_chunks = LAST_CHUNKS;
    double m = mux_margin(zr[0], zi[0], ok[0], &f), ux = 0, uy = 0, t = 0;
    if (m < -1e8) { printf("program leaves the basin noiselessly\n"); return; }
    halfplane(zr[0], zi[0], f, &ux, &uy, &t);
    drive_str(d, buf, sizeof buf);
    printf("langevin: %s  f=0x%02x margin %.4f\n   GEOM=%s EPS=%.3f DT=%.2f  T_SET = %d chunks = %.0f time units (measured noiseless lock time)\n",
           buf, f, m, GEOM, EPS_V, DT, tset_chunks, tset_chunks * T_CHUNK);
    int steps_chunk = (int)(T_CHUNK / DT), nset = tset_chunks * steps_chunk;
    double wmax = 0; for (int w = 0; w < nW; w++) if (Wr[w] > wmax) wmax = Wr[w];
    int nread = (int)(wmax / DT);
    for (int di = 0; di < nD; di++) {
        double D = Ds[di], amp = sqrt(2.0 * D * DT);
        rng_t rg = {0x1A9Eull + di};
        int per = LMAX / 8, L;
        int corr[16] = {0}, slip = 0, total = 0;
        double inst_sd = 0; int inst_n = 0;
        for (int k0 = 0; k0 < K; k0 += per) {
            int G = K - k0 < per ? K - k0 : per; L = 8 * G;
            mbatch *b = mb_new(L);
            for (int l = 0; l < L; l++) { for (int i = 0; i < NSITE; i++) b->th[i*L + l] = 0.0;
                set_drive_lane(b, l, d, l & 7, 0, 0, NULL); }
            for (int st = 0; st < nset; st++) { mrk4(b); for (int j = 0; j < NSITE*L; j++) b->th[j] += amp * rng_gauss(&rg); }
            /* read window: running sums of the phasor, kept per window length */
            double *ar = calloc(L * nW, sizeof *ar), *ai = calloc(L * nW, sizeof *ai);
            for (int st = 0; st < nread; st++) {
                mrk4(b); for (int j = 0; j < NSITE*L; j++) b->th[j] += amp * rng_gauss(&rg);
                double tleft = (nread - st) * DT;          /* samples inside the LAST Wr time units */
                for (int l = 0; l < L; l++) { double pr, pi; phasor(b, l, PMODE, &pr, &pi);
                    for (int w = 0; w < nW; w++) if (tleft <= Wr[w] + 1e-9) { ar[l*nW + w] += pr; ai[l*nW + w] += pi; }
                    if (st == nread - 1) { int x = l & 7; double dr = pr - zr[0][x], dm = pi - zi[0][x]; inst_sd += dr*dr + dm*dm; inst_n++; } }
            }
            for (int l = 0; l < L; l++) { int x = l & 7, want = f >> x & 1;
                int sl = mwinding(b, RA, l) || mwinding(b, RB, l) || mwinding(b, RO, l); slip += sl; total++;
                for (int w = 0; w < nW; w++) { double n = floor(Wr[w] / DT + 0.5); if (n < 1) n = 1;
                    double r = ar[l*nW + w] / n, i2 = ai[l*nW + w] / n;
                    corr[w] += ((ux*r + uy*i2 > t) ? 1 : 0) == want; } }
            free(ar); free(ai); mb_free(b);
        }
        printf("   D=%-8.5f instantaneous phasor scatter %.4f (margin %.4f)  slipped %d/%d  accuracy by read window:",
               D, sqrt(inst_sd / inst_n), m, slip, total);
        for (int w = 0; w < nW; w++) printf("  %g:%5.1f%%", Wr[w], 100.0 * corr[w] / total);
        printf("\n");
    }
}

static int mux_level(void){
    set_level(0); if (enumerate().mux_any == 0) return 0;
    for (int k = ROBUST_R; k >= 1; k--) { set_level(k); if (enumerate().mux_any > 0) return k; }
    return 0;
}

static void report(int verify_k){
    int kept = 0, total = 0, mk = -1;
    memcpy(RAW, T, sizeof T);
    int nalive = 0; for (int j = 0; j < NJ; j++) nalive += ALIVE[j];
    if (ROBUST_R) { kept = robust_filter(&total); mk = mux_level(); set_level(mk > 0 ? mk : ROBUST_R); }
    else set_level(0);
    mcov_t c = enumerate();
    printf("GEOM=%s EPS=%.3f alive=%d/%d NI=%d %s%d/%d  maps L1=%zu L2=%zu  any=%d nz=%d pos=%d  MUX8: any=%d nz=%d pos=%d  mux_k=%d\n",
           GEOM, EPS_V, nalive, NJ, NI, ROBUST_R ? "robust=" : "", kept, total, c.l1, c.l2, c.any, c.nz, c.pos,
           c.mux_any, c.mux_nz, c.mux_pos, mk);
    if (c.mux_any) {
        if (ROBUST_R) printf("   MUX programs below use entries passing >= %d/%d\n", mk, ROBUST_R);
        for (int k = 0; k < 8; k++) {
            mprog_t pr = find_witness(MUX8[k]); if (!pr.ok) continue;
            print_mprog(MUX8[k], &pr);
            if (verify_k) { int un;
                int cl = verify(MUX8[k], &pr, verify_k, 0.0, 0xA2ull + k, &un);
                int ns = verify(MUX8[k], &pr, verify_k, RSIG, 0x5EEDull + k, &un);
                printf("      physical: clean %d.%d%%  sigma=%.2f %d.%d%%  (%d unlocked)\n", cl/10, cl%10, RSIG, ns/10, ns%10, un); }
        }
    }
}

/* XCHECK at ε = 0: ring A's ERASE rows from (qA,0,0) vs build_table(0, 0.0) */
static int xcheck(void){
    EPS_V = 0.0; set_geometry(); measure();
    int tab[TBL_SZ], alive[2*Q_MAX+1]; build_table(0, 0.0, tab, alive);
    int diff = 0, n = 0;
    for (int qa = -QR; qa <= QR; qa++) { int j = jidx(qa, 0, 0);
        if (ALIVE[j] != alive[qa + Q_MAX]) { printf("   alive differs q=%+d\n", qa); diff++; }
        for (int I = 1; I < NI; I++) { if (IS[I].ring != RA || IS[I].kind != K_ERASE) continue;
            int a = IS[I].amp == AMPS[0] ? 0 : 1, ref = tab[TBL_IDX(qa, IS[I].site, a)];
            int refv = (ref > DEAD+1 && ref >= -QR && ref <= QR) ? jidx(ref, 0, 0) : -1;
            n++; if (refv != T[j][I]) { diff++; char nm[64]; iname(I, nm, sizeof nm);
                printf("   diff qA=%+d %-22s furnace=%d macro=%d\n", qa, nm, ref, T[j][I] < 0 ? -99 : T[j][I]); } } }
    printf("xcheck eps=0: %d/%d ring-A ERASE entries differ from build_table %s\n", diff, n, diff ? "" : "(IDENTICAL)");
    return diff != 0;
}

int main(int argc, char **argv){
    if (getenv("EPS"))      EPS_V = atof(getenv("EPS"));
    if (getenv("GEOM"))     GEOM = getenv("GEOM");
    if (getenv("ROBUST"))   ROBUST_R = atoi(getenv("ROBUST"));
    if (getenv("RSIG"))     RSIG = atof(getenv("RSIG"));
    if (getenv("STAGE1"))   STAGE1 = atoi(getenv("STAGE1"));
    if (getenv("PATIENCE")) PATIENCE = atoi(getenv("PATIENCE"));
    PREF_LAB = getenv("PREF") && !strcmp(getenv("PREF"), "lab");
    START_RESET = getenv("START") && !strcmp(getenv("START"), "reset");
    build_alphabet(); set_geometry();
    for (int j = 0; j < NJ; j++) JSEL[j] = 1;
    for (int I = 0; I < NI; I++) ISEL[I] = 1;
    const char *cmd = argc > 1 ? argv[1] : "";
    if (!strcmp(cmd, "xcheck")) return xcheck();
    if (!strcmp(cmd, "gates")) {               /* only the 4 contexts x O-instructions */
        int q0 = argc > 2 ? atoi(argv[2]) : 0;
        for (int j = 0; j < NJ; j++) JSEL[j] = 0;
        for (int c = 0; c < 4; c++) JSEL[jidx(c & 1, c >> 1, q0)] = 1;
        for (int I = 0; I < NI; I++) ISEL[I] = IS[I].ring == RO;
        measure(); memcpy(RAW, T, sizeof T);
        int tot; if (ROBUST_R) robust_filter(&tot);
        int na = 0; for (int j = 0; j < NJ; j++) na += ALIVE[j];
        printf("alive joint states: %d/%d  NI=%d\n", na, NJ, NI);
        gates(q0, getenv("SHOW_ALL") != NULL);
        return 0; }
    if (!strcmp(cmd, "prog")) {      /* prog qA,qB,qO I0 I1 I0 I1 I0 I1 readout(qO list, ';'-sep) f_hex K */
        if (argc < 12) { fprintf(stderr, "prog needs: start 6 instrs readout f K\n"); return 2; }
        settle_attractors();
        mprog_t pr = {0}; int qa, qb, qo;
        if (sscanf(argv[2], "%d,%d,%d", &qa, &qb, &qo) != 3) return 2;
        pr.j0 = jidx(qa, qb, qo);
        if (!ALIVE[pr.j0]) { printf("start state not alive\n"); return 1; }
        for (int k = 0; k < 3; k++) { pr.i0[k] = find_instr(argv[3 + 2*k]); pr.i1[k] = find_instr(argv[4 + 2*k]); }
        char rb[64]; snprintf(rb, sizeof rb, "%s", argv[9]);
        for (char *t = strtok(rb, ";"); t; t = strtok(NULL, ";")) pr.R |= 1 << (atoi(t) + QR);
        int f = (int)strtol(argv[10], NULL, 16), K = atoi(argv[11]);
        printf("GEOM=%s EPS=%.3f PATIENCE=%d DT=%.2f\n", GEOM, EPS_V, PATIENCE, DT);
        print_mprog(f, &pr); trace_m(&pr);
        int un, cl = verify(f, &pr, K, 0.0, 0xA2ull, &un);
        printf("   physical clean: %d.%d%% (%d unlocked)\n", cl/10, cl%10, un);
        double sigs[] = {0.05, 0.10, 0.15, 0.20, 0.30};
        for (int k = 0; k < 5; k++) { int ns = verify(f, &pr, K, sigs[k], 0x5EEDull + k, &un);
            printf("   physical sigma=%.2f: %d.%d%% (%d unlocked of %d)\n", sigs[k], ns/10, ns%10, un, 8*K); }
        return 0; }
    if (!strcmp(cmd, "multi")) {     /* multi M tol qa,qb,qo [...] */
        settle_attractors(); int M = atoi(argv[2]); double tol = atof(argv[3]);
        for (int k = 4; k < argc; k++) { int a, b2, c; if (sscanf(argv[k], "%d,%d,%d", &a, &b2, &c) == 3) multi(jidx(a, b2, c), M, tol); }
        return 0; }
    if (!strcmp(cmd, "phasor")) {    /* phasor q0 [mode] : search b2 pairs for MUX8 */
        if (argc > 3) PMODE = atoi(argv[3]);
        printf("phasor search: GEOM=%s EPS=%.3f PATIENCE=%d O starts q=%s mode k=%d\n", GEOM, EPS_V, PATIENCE, argv[2], PMODE);
        phasor_search(atoi(argv[2])); return 0; }
    if (!strcmp(cmd, "pverify")) {   /* pverify q0 mode I0 I1 f_hex K */
        PMODE = atoi(argv[3]);
        printf("phasor verify: GEOM=%s EPS=%.3f PATIENCE=%d DT=%.2f O starts q=%s mode k=%d f=0x%s  b2: 0->%s 1->%s\n",
               GEOM, EPS_V, PATIENCE, DT, argv[2], PMODE, argv[6], argv[4], argv[5]);
        phasor_verify(atoi(argv[2]), argv[4], argv[5], (int)strtol(argv[6], NULL, 16), atoi(argv[7])); return 0; }
    if (!strcmp(cmd, "dsearch")) {   /* dsearch M umax seed */
        dsearch(atoi(argv[2]), atof(argv[3]), strtoull(argv[4], NULL, 0)); return 0; }
    if (!strcmp(cmd, "dclimb")) {    /* dclimb "<drives>" steps */
        drive_t d[3]; if (!parse_drives(argv[2], d)) { fprintf(stderr, "bad drives\n"); return 2; }
        dclimb(d, atoi(argv[3])); return 0; }
    if (!strcmp(cmd, "dverify")) {   /* dverify "<drives>" K */
        drive_t d[3]; if (!parse_drives(argv[2], d)) { fprintf(stderr, "bad drives\n"); return 2; }
        dverify(d, atoi(argv[3])); return 0; }
    if (!strcmp(cmd, "langevin")) {  /* langevin "<drives>" K D1,D2,... W1,W2,... */
        drive_t d[3]; if (!parse_drives(argv[2], d)) { fprintf(stderr, "bad drives\n"); return 2; }
        double Ds[16], Ws[16]; int nD = 0, nW = 0; char b1[256], b2[256];
        snprintf(b1, sizeof b1, "%s", argv[4]); snprintf(b2, sizeof b2, "%s", argv[5]);
        for (char *t = strtok(b1, ","); t && nD < 16; t = strtok(NULL, ",")) Ds[nD++] = atof(t);
        for (char *t = strtok(b2, ","); t && nW < 16; t = strtok(NULL, ",")) Ws[nW++] = atof(t);
        langevin(d, atoi(argv[3]), Ds, nD, Ws, nW); return 0; }
    if (!strcmp(cmd, "run")) { measure(); report(argc > 2 ? atoi(argv[2]) : 0); return 0; }
    fprintf(stderr, "usage: fork_macro xcheck | gates [qO_start] | run [verify_trials]\n"
                    "env: GEOM=half|full EPS=e EPS_AB=e ROBUST=R RSIG=s STAGE1=n PATIENCE=c START=all|reset RAMP_WIDTHS=...\n");
    return 2;
}
