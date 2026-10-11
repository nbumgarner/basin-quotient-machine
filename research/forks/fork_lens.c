/* fork_lens.c — Fork A3 on the real fabric: multi-lens programs on ONE ring.
 *
 * Question (HANDOFF.md task 2): can lenses alone — a static ω profile with
 * lenses at 2–3 sites — close the late-selector MUX gap
 *   0x35 0x3a 0x53 0x5c 0xa3 0xac 0xc5 0xca   (b2 ? u : v, selector last)
 * on one N=16 ring using only noise-robust transitions?
 *
 * Generalizes fork_logic.c (run 02); everything not listed is identical:
 *   1. LENS: an ω profile OM[N] (any number of lens sites), not (site 0, Δ).
 *   2. WRITE: selectable. WRITE=vm (default) is apply_ramp() copied verbatim
 *      from bqsm_vm.c — Nick's WRITE_INC / RAMP_AT: centered, clamped to ±π,
 *      +1 only, width from RAMP_WIDTHS (default N = the VM's WRITE_INC).
 *      WRITE=handoff is fork_logic.c's 8-site start-anchored ±ramp, kept for
 *      the cross-check and for side-by-side numbers.
 *   3. WITNESSES: every reachable state map keeps a back-pointer, so any
 *      covered function prints its program (q0, instruction pairs, readout
 *      set), and `verify` runs that program physically on the furnace.
 *
 * Trust anchor: `fork_lens xcheck <delta>` builds the run-02 tables two ways —
 * build_table() + fork_logic's WRITE rows, and this file's generalized
 * builder (single lens at site 0, WRITE=handoff) — and diffs every entry.
 *
 * Build: gcc -O3 -std=c11 fork_lens.c -o fork_lens -lm      (+ -DREF_BUILD)
 */
#define main rf_main
#include "../../ring_furnace.c"
#undef main

#define NS     (2*Q_MAX+1)      /* 7 state slots                             */
#define NIMAX  160              /* instruction alphabet capacity             */
#define HO_RAMP_W 8             /* fork_logic.c RAMP_W (handoff WRITE width) */

/* ---------------------------------------------------------------------------
 * INSTRUCTION ALPHABET
 * ------------------------------------------------------------------------- */
enum { K_NOP, K_ERASE, K_WRITE_HO, K_WRITE_VM };
typedef struct { int kind, site, width; double amp, sign; } instr_t;
static instr_t IS[NIMAX];
static int NI;                  /* live alphabet size                        */
static int WRITE_VM = 1;        /* 1 = Nick's apply_ramp, 0 = handoff ramp   */

/* apply_ramp — VERBATIM from bqsm_vm.c (WRITE_INC / RAMP_AT). */
static void apply_ramp(batch_t *b, int l, int center_site, int ramp_width){
    if (ramp_width < 2) ramp_width = 2;  /* minimum viable ramp              */
    for (int i = 0; i < N; i++) {
        /* Distance from center_site to i around the ring, signed.
         * The ramp applies the full 2π gradient across the ramp window.      */
        int dist = (i - center_site) & NMASK;
        if (dist > N/2) dist -= N;       /* shortest signed distance         */
        /* Linear ramp: phase addition is 2π * dist / ramp_width clamped     */
        double frac = (double)dist / ramp_width;
        if (frac >  0.5) frac =  0.5;    /* clamp edges to half-ramp         */
        if (frac < -0.5) frac = -0.5;
        b->th[i * b->L + l] += 2.0 * M_PI * frac;
    }
}

/* apply_ramp with the phase negated (a −1 write). NOT in bqsm_vm.c's ISA —
 * WRITE_NEG=1 adds it only to separate ramp SHAPE from ramp SIGN.           */
static void apply_ramp_neg(batch_t *b, int l, int center_site, int ramp_width){
    if (ramp_width < 2) ramp_width = 2;
    for (int i = 0; i < N; i++) {
        int dist = (i - center_site) & NMASK;
        if (dist > N/2) dist -= N;
        double frac = (double)dist / ramp_width;
        if (frac >  0.5) frac =  0.5;
        if (frac < -0.5) frac = -0.5;
        b->th[i * b->L + l] -= 2.0 * M_PI * frac;
    }
}

/* The handoff WRITE — fork_logic.c apply_instr's WRITE branch, verbatim. */
static void apply_ramp_handoff(batch_t *b, int l, int s, double sg){
    for (int k=0;k<N;k++){ double f = k<HO_RAMP_W ? (double)k/HO_RAMP_W : 1.0;
        b->th[((s+k)&NMASK)*b->L + l] += sg*2.0*M_PI*f; }
}

static void apply_instr(batch_t *b, int l, int I){
    const instr_t *x = &IS[I];
    switch (x->kind) {
    case K_NOP:      return;
    case K_ERASE:    apply_kick(b, l, x->site, x->amp); return;
    case K_WRITE_HO: apply_ramp_handoff(b, l, x->site, x->sign); return;
    case K_WRITE_VM: if (x->sign > 0) apply_ramp(b, l, x->site, x->width);
                     else apply_ramp_neg(b, l, x->site, x->width);
                     return;
    }
}

static void instr_name(int I, char *buf, size_t n){
    const instr_t *x = &IS[I];
    switch (x->kind) {
    case K_NOP:      snprintf(buf, n, "NOP"); break;
    case K_ERASE:    snprintf(buf, n, "ERASE(s=%d,a=%.1f)", x->site, x->amp); break;
    case K_WRITE_HO: snprintf(buf, n, "WRITE%c(s=%d,w=%d,handoff)", x->sign > 0 ? '+' : '-', x->site, HO_RAMP_W); break;
    case K_WRITE_VM: snprintf(buf, n, "WRITE%c(s=%d,w=%d)", x->sign > 0 ? '+' : '-', x->site, x->width); break;
    }
}

/* Alphabet order matches fork_logic.c exactly for WRITE=handoff:
 * 0 = NOP, 1 + s*2 + a = ERASE(site s, AMPS[a]), then WRITEs.
 * WRITE=vm: width N is a uniform +1 twist — the center only adds a global
 * phase, so it is ONE instruction (site 0, as bqsm_vm.c's table builds it);
 * widths < N are site-dependent and enumerated at all 16 sites.             */
static void build_alphabet(void){
    NI = 0;
    IS[NI++] = (instr_t){K_NOP, 0, 0, 0, 0};
    for (int s = 0; s < N; s++) for (int a = 0; a < 2; a++)
        IS[NI++] = (instr_t){K_ERASE, s, KICK_W, AMPS[a], 0};
    /* EXTRA_AMPS: KICK_AT amplitudes beyond the furnace's {1.0, 2.2} (bqsm_vm.c
     * KICK_AT takes any amplitude). Diagnostic for the decrement hypothesis;
     * appended AFTER the run-02 ERASE block so its indices are unchanged.   */
    if (getenv("EXTRA_AMPS")) { char eb[128]; snprintf(eb, sizeof eb, "%s", getenv("EXTRA_AMPS"));
        for (char *t = strtok(eb, ","); t; t = strtok(NULL, ",")) { double a = atof(t);
            for (int s = 0; s < N; s++) IS[NI++] = (instr_t){K_ERASE, s, KICK_W, a, 0}; } }
    const char *w = getenv("WRITE");
    WRITE_VM = !(w && !strcmp(w, "handoff"));
    /* WRITE_NEG: 1 = include −1 writes. Default follows each family's own
     * definition: handoff had ±, bqsm_vm.c is +1 only.                      */
    int neg = getenv("WRITE_NEG") ? atoi(getenv("WRITE_NEG")) : !WRITE_VM;
    int nsg = neg ? 2 : 1;
    if (!WRITE_VM) {
        for (int s = 0; s < N; s++) for (int g = 0; g < nsg; g++)
            IS[NI++] = (instr_t){K_WRITE_HO, s, HO_RAMP_W, 0, g ? -1.0 : 1.0};
        return;
    }
    char buf[128]; const char *rw = getenv("RAMP_WIDTHS");
    snprintf(buf, sizeof buf, "%s", rw ? rw : "16");
    for (char *t = strtok(buf, ","); t; t = strtok(NULL, ",")) {
        int width = atoi(t);
        for (int g = 0; g < nsg; g++) {
            double sg = g ? -1.0 : 1.0;
            if (width >= N) { IS[NI++] = (instr_t){K_WRITE_VM, 0, width, 0, sg}; continue; }
            for (int s = 0; s < N; s++) IS[NI++] = (instr_t){K_WRITE_VM, s, width, 0, sg};
        }
    }
    if (NI > NIMAX) { fprintf(stderr, "alphabet %d > NIMAX\n", NI); exit(2); }
}

/* ---------------------------------------------------------------------------
 * LENS = ω profile. Spec string "site:delta,site:delta,..." ("" = pristine).
 * ------------------------------------------------------------------------- */
static double OM[N];
static void parse_lens(const char *spec){
    for (int i = 0; i < N; i++) OM[i] = 0.0;
    if (!spec || !*spec) return;
    char buf[256]; snprintf(buf, sizeof buf, "%s", spec);
    for (char *t = strtok(buf, ","); t; t = strtok(NULL, ",")) {
        int s; double d;
        if (sscanf(t, "%d:%lf", &s, &d) != 2) { fprintf(stderr, "bad lens term '%s'\n", t); exit(2); }
        OM[s & NMASK] += d;
    }
}
static void lens_str(char *buf, size_t n){
    size_t k = 0; buf[0] = 0;
    for (int i = 0; i < N; i++) if (OM[i] != 0.0)
        k += snprintf(buf + k, n - k, "%s%d:%.3f", k ? "," : "", i, OM[i]);
    if (!k) snprintf(buf, n, "pristine");
}
static void load_lane(batch_t *b, int l, const double *att){
    for (int i = 0; i < N; i++) { b->th[i*b->L + l] = att[i]; b->om[i*b->L + l] = OM[i]; }
}

/* STRICT settle. The fast build's early-exit lock (all wrapped gaps inside
 * (−π+0.1, π−0.1) on two checks) is rigorous only on the pristine ring (the
 * furnace says so); under a multi-lens ω profile a drifting state can pass
 * it and slip later — measured: q=−2 under 0:−0.40,1:0.55,5:0.60 "locks",
 * then slips to −1 on the next settle. STRICT=1 (default) keeps a lane
 * locked only if, continuing to integrate for up to CONFIRM chunks (default
 * MAXCHUNK, the furnace's own patience), its velocity spread falls below
 * LOCK_TOL — the furnace's lock criterion, the only one REF_BUILD uses — and
 * its winding never changes. Slow-converging true fixed points pass (the
 * dipole's marginal q=+3 needs ~12 chunks; a fixed 4-chunk check rejected
 * it); drifting states fail. STRICT=0 = plain settle().                     */
static int STRICT = 1;
static int CONFIRM = MAXCHUNK;          /* confirm patience, chunks; env      */
static void settle_s(batch_t *b, int *lock){
    settle(b, lock);
    if (!STRICT) return;
    int L = b->L, w0[LMAX], done[LMAX], open = 0;
    double spread[LMAX];
    lock_spread(b, spread);
    for (int l = 0; l < L; l++) {
        w0[l] = winding(b, l);
        done[l] = !lock[l] || spread[l] < LOCK_TOL;      /* unlocked: nothing to confirm */
        open += !done[l];
    }
    int steps = (int)(T_CHUNK / DT);
    for (int c = 0; c < CONFIRM && open; c++) {
        for (int s = 0; s < steps; s++) rk4_step(b);
        lock_spread(b, spread); open = 0;
        for (int l = 0; l < L; l++) {
            if (winding(b, l) != w0[l]) { lock[l] = 0; done[l] = 1; }
            if (!done[l]) { if (spread[l] < LOCK_TOL) done[l] = 1; else open++; }
        }
    }
    for (int l = 0; l < L; l++) {
        if (getenv("SPREAD_DEBUG"))
            fprintf(stderr, "lane %d lock=%d done=%d w0=%d w=%d spread=%.3e\n", l, lock[l], done[l], w0[l], winding(b, l), spread[l]);
        lock[l] = lock[l] && done[l] && winding(b, l) == w0[l];
    }
}

/* ---------------------------------------------------------------------------
 * TABLE — measured on the furnace under OM. Attractors settle as one batch of
 * 7 lanes, exactly like build_table's Phase A; every (q, instr) is one lane.
 * ------------------------------------------------------------------------- */
static int    T[NS][NIMAX];     /* next-state slot, or -1                    */
static int    ALIVE[NS];
static double ATT[NS][N];

static void settle_attractors(void){
    batch_t *b = batch_new(NS);
    for (int l = 0; l < NS; l++) {
        set_twisted(b, l, l - Q_MAX);
        for (int i = 0; i < N; i++) b->om[i*NS + l] = OM[i];
    }
    int lock[LMAX]; settle_s(b, lock);
    for (int l = 0; l < NS; l++) {
        ALIVE[l] = lock[l] && winding(b, l) == l - Q_MAX;
        for (int i = 0; i < N; i++) ATT[l][i] = b->th[i*NS + l];
    }
    batch_free(b);
}

/* settle a list of (state, instr) jobs, LMAX lanes at a time */
static void measure(void){
    settle_attractors();
    int jq[NS*NIMAX], ji[NS*NIMAX], nj = 0;
    for (int q = 0; q < NS; q++) {
        T[q][0] = ALIVE[q] ? q : -1;                     /* NOP              */
        for (int I = 1; I < NI; I++) {
            T[q][I] = -1;
            if (ALIVE[q]) { jq[nj] = q; ji[nj] = I; nj++; }
        }
    }
    for (int j0 = 0; j0 < nj; j0 += LMAX) {
        int L = nj - j0 < LMAX ? nj - j0 : LMAX;
        batch_t *b = batch_new(L);
        for (int l = 0; l < L; l++) { load_lane(b, l, ATT[jq[j0+l]]); apply_instr(b, l, ji[j0+l]); }
        int lock[LMAX]; settle_s(b, lock);
        for (int l = 0; l < L; l++) {
            int v = winding(b, l);
            T[jq[j0+l]][ji[j0+l]] = (lock[l] && v >= -Q_MAX && v <= Q_MAX && ALIVE[v+Q_MAX]) ? v + Q_MAX : -1;
        }
        batch_free(b);
    }
}

/* Robustness filter — fork_logic.c's rule: each (q, instr) re-run R times with
 * N(0, σ) phase noise on all sites before the instruction; keep only entries
 * whose R outcomes ALL match the table. PASS[q][I] keeps the pass count.     */
static int ROBUST_R = 0; static double ROBUST_SIG = 0.15;
static int PASS[NS][NIMAX];
static uint64_t lens_seed(void){
    uint64_t h = 0xC0FFEEull;
    for (int i = 0; i < N; i++) h = h*0x100000001B3ull ^ (uint64_t)(int64_t)llround(OM[i]*1e6) ^ (uint64_t)i;
    if (getenv("SEED")) h ^= strtoull(getenv("SEED"), NULL, 0) * 0x9E3779B97F4A7C15ull;  /* re-draw the noise */
    return h;
}
static int robust_filter(int *total_out){
    rng_t r = {lens_seed()};
    int jq[NS*NIMAX], ji[NS*NIMAX], nj = 0;
    for (int q = 0; q < NS; q++) for (int I = 1; I < NI; I++) {
        PASS[q][I] = 0;
        if (T[q][I] >= 0) { jq[nj] = q; ji[nj] = I; nj++; }
    }
    int per = LMAX / ROBUST_R, kept = 0;
    for (int j0 = 0; j0 < nj; j0 += per) {
        int G = nj - j0 < per ? nj - j0 : per, L = G * ROBUST_R;
        batch_t *b = batch_new(L);
        for (int g = 0; g < G; g++) for (int k = 0; k < ROBUST_R; k++) {
            int l = g*ROBUST_R + k;
            load_lane(b, l, ATT[jq[j0+g]]);
            for (int i = 0; i < N; i++) b->th[i*L + l] += ROBUST_SIG * rng_gauss(&r);
            apply_instr(b, l, ji[j0+g]);
        }
        int lock[LMAX]; settle_s(b, lock);
        for (int g = 0; g < G; g++) {
            int q = jq[j0+g], I = ji[j0+g], ok = 0;
            for (int k = 0; k < ROBUST_R; k++) { int l = g*ROBUST_R + k;
                ok += lock[l] && winding(b, l) + Q_MAX == T[q][I]; }
            PASS[q][I] = ok;
            if (ok == ROBUST_R) kept++; else T[q][I] = -1;
        }
        batch_free(b);
    }
    *total_out = nj;
    return kept;
}

/* ---------------------------------------------------------------------------
 * ENUMERATION with back-pointers. A state map assigns each of the 2^m input
 * words (x = b0 + 2 b1 + 4 b2; b0 arrives first) a state slot, base 8.
 * ------------------------------------------------------------------------- */
typedef struct { uint32_t *key; int32_t *idx; size_t cap, n; } hmap;
typedef struct { uint32_t *code; int32_t *par; uint8_t *i0, *i1; size_t n, cap; } level_t;
static void hm_init(hmap *h, size_t cap){ h->cap = cap; h->n = 0; h->key = malloc(cap*4); h->idx = malloc(cap*4); memset(h->key, 0xff, cap*4); }
static void hm_free(hmap *h){ free(h->key); free(h->idx); }
static void hm_put(hmap *h, uint32_t key, int32_t idx){
    size_t i = (key * 2654435761u) & (h->cap - 1);
    while (h->key[i] != 0xffffffffu) i = (i + 1) & (h->cap - 1);
    h->key[i] = key; h->idx[i] = idx; h->n++;
}
static int hm_find_or_add(hmap *h, uint32_t key, int32_t newidx){
    size_t i = (key * 2654435761u) & (h->cap - 1);
    while (h->key[i] != 0xffffffffu) { if (h->key[i] == key) return h->idx[i]; i = (i + 1) & (h->cap - 1); }
    h->key[i] = key; h->idx[i] = newidx; h->n++;
    if (h->n * 2 > h->cap) {                     /* grow at load 1/2 */
        hmap g; hm_init(&g, h->cap * 2);
        for (size_t j = 0; j < h->cap; j++) if (h->key[j] != 0xffffffffu) hm_put(&g, h->key[j], h->idx[j]);
        hm_free(h); *h = g;
    }
    return -1;
}
static void lv_push(level_t *v, uint32_t code, int32_t par, int i0, int i1){
    if (v->n == v->cap) { v->cap = v->cap ? v->cap*2 : 1024;
        v->code = realloc(v->code, v->cap*4); v->par = realloc(v->par, v->cap*4);
        v->i0 = realloc(v->i0, v->cap); v->i1 = realloc(v->i1, v->cap); }
    v->code[v->n] = code; v->par[v->n] = par; v->i0[v->n] = (uint8_t)i0; v->i1[v->n] = (uint8_t)i1; v->n++;
}
static void lv_free(level_t *v){ free(v->code); free(v->par); free(v->i0); free(v->i1); memset(v, 0, sizeof *v); }
static inline uint32_t enc_set(uint32_t code, int x, int st){ return (code & ~(7u<<(3*x))) | ((uint32_t)st<<(3*x)); }
static inline int enc_get(uint32_t code, int x){ return (code>>(3*x))&7; }

static level_t LV[4];
static int W_full[256], W_fullR[256], W_nz[256], W_pos[256]; /* witness final-map index (-1 = none) */

static const int MUX8[8] = {0x35,0x3a,0x53,0x5c,0xa3,0xac,0xc5,0xca};

typedef struct { int nalive, kept, total; size_t maps; int any, nz, pos, mux_any, mux_nz, mux_pos, maxd; } cov_t;

static cov_t enumerate(void){
    cov_t c = {0};
    for (int m = 0; m < 4; m++) lv_free(&LV[m]);
    for (int q = 0; q < NS; q++) if (ALIVE[q]) { lv_push(&LV[0], (uint32_t)q, -1, 0, 0); c.nalive++; }
    for (int m = 0; m < 3; m++) {
        hmap h; hm_init(&h, 1u << 12);
        int nin = 1 << m;
        for (size_t p = 0; p < LV[m].n; p++) {
            uint32_t code = LV[m].code[p];
            /* candidate instructions = those valid on every state this map uses */
            int used = 0; for (int x = 0; x < nin; x++) used |= 1 << enc_get(code, x);
            int cand[NIMAX], nc = 0;
            for (int I = 0; I < NI; I++) { int ok = 1;
                for (int q = 0; q < NS && ok; q++) if ((used >> q & 1) && T[q][I] < 0) ok = 0;
                if (ok) cand[nc++] = I; }
            for (int a = 0; a < nc; a++) for (int bb = 0; bb < nc; bb++) {
                int i0 = cand[a], i1 = cand[bb]; uint32_t out = 0;
                for (int x = 0; x < nin; x++) { int st = enc_get(code, x);
                    out = enc_set(out, x, T[st][i0]); out = enc_set(out, x + nin, T[st][i1]); }
                if (hm_find_or_add(&h, out, (int32_t)LV[m+1].n) < 0) lv_push(&LV[m+1], out, (int32_t)p, i0, i1);
            }
        }
        hm_free(&h);
    }
    for (int f = 0; f < 256; f++) W_full[f] = W_fullR[f] = W_nz[f] = W_pos[f] = -1;
    for (size_t p = 0; p < LV[3].n; p++) {
        uint32_t code = LV[3].code[p];
        int used = 0; for (int x = 0; x < 8; x++) used |= 1 << enc_get(code, x);
        int d = __builtin_popcount(used); if (d > c.maxd) c.maxd = d;
        for (int R = 0; R < 128; R++) { if (R & ~used) continue;
            int f = 0; for (int x = 0; x < 8; x++) if (R >> enc_get(code, x) & 1) f |= 1 << x;
            if (W_full[f] < 0) { W_full[f] = (int)p; W_fullR[f] = R; } }
        int fz = 0, fp = 0;
        for (int x = 0; x < 8; x++) { int q = enc_get(code, x) - Q_MAX; if (q != 0) fz |= 1 << x; if (q > 0) fp |= 1 << x; }
        if (W_nz[fz] < 0) W_nz[fz] = (int)p;
        if (W_pos[fp] < 0) W_pos[fp] = (int)p;
    }
    c.maps = LV[3].n;
    for (int f = 0; f < 256; f++) { c.any += W_full[f] >= 0; c.nz += W_nz[f] >= 0; c.pos += W_pos[f] >= 0; }
    for (int k = 0; k < 8; k++) { c.mux_any += W_full[MUX8[k]] >= 0; c.mux_nz += W_nz[MUX8[k]] >= 0; c.mux_pos += W_pos[MUX8[k]] >= 0; }
    return c;
}

/* program recovered from a final-level map index */
typedef struct { int q0, i0[3], i1[3]; } prog_t;
static prog_t recover(int p3){
    prog_t pr; int p = p3;
    for (int m = 3; m >= 1; m--) { pr.i0[m-1] = LV[m].i0[p]; pr.i1[m-1] = LV[m].i1[p]; p = LV[m].par[p]; }
    pr.q0 = (int)LV[0].code[p];
    return pr;
}
static void print_prog(int f, const prog_t *pr, int R, const char *readout){
    char a[64], b[64];
    printf("   f=0x%02x  q0=%+d  readout=%s", f, pr->q0 - Q_MAX, readout);
    if (R >= 0) { printf("{"); int first = 1; for (int q = 0; q < NS; q++) if (R >> q & 1) { printf("%s%+d", first ? "" : ",", q - Q_MAX); first = 0; } printf("}"); }
    printf("\n");
    for (int k = 0; k < 3; k++) { instr_name(pr->i0[k], a, sizeof a); instr_name(pr->i1[k], b, sizeof b);
        printf("      b%d: 0 -> %-28s 1 -> %s\n", k, a, b); }
}

/* ---------------------------------------------------------------------------
 * PHYSICAL VERIFICATION of a program: all 8 inputs × K trials, kick by kick on
 * the furnace, N(0,σ) phase noise on every site before each instruction.
 * ------------------------------------------------------------------------- */
static int verify_prog(int f, const prog_t *pr, int R, int K, double sigma, uint64_t seed, int *unlocked){
    rng_t r = {seed}; int correct = 0, total = 0; *unlocked = 0;
    int per = LMAX / 8; if (per > K) per = K;
    for (int k0 = 0; k0 < K; k0 += per) {
        int G = K - k0 < per ? K - k0 : per, L = 8 * G;
        batch_t *b = batch_new(L);
        for (int l = 0; l < L; l++) load_lane(b, l, ATT[pr->q0]);
        int lk[LMAX]; for (int l = 0; l < L; l++) lk[l] = 1;
        for (int s = 0; s < 3; s++) {
            for (int l = 0; l < L; l++) { int x = l & 7;
                for (int i = 0; i < N; i++) b->th[i*L + l] += sigma * rng_gauss(&r);
                apply_instr(b, l, (x >> s & 1) ? pr->i1[s] : pr->i0[s]); }
            int lock[LMAX]; settle_s(b, lock);
            for (int l = 0; l < L; l++) lk[l] &= lock[l];
        }
        for (int l = 0; l < L; l++) { int x = l & 7, q = winding(b, l), out;
            if (R >= 0) out = (q >= -Q_MAX && q <= Q_MAX) ? (R >> (q + Q_MAX) & 1) : -1;
            else if (R == -1) out = q != 0; else out = q > 0;
            total++; if (!lk[l]) (*unlocked)++;
            correct += out == (f >> x & 1); }
        batch_free(b);
    }
    return correct * 1000 / total;   /* per-mille */
}

/* TRACE: run one program clean, kick by kick, for all 8 inputs; at each step
 * print predicted q (table), physical q, and the gauge-invariant distance of
 * the settled state to the reference attractor ATT[q] (dist_orbit). A large
 * distance at the right q means a SECOND attractor with the same winding —
 * q is then not a complete state label and table composition can fail.     */
static void trace_prog(const prog_t *pr){
    for (int x = 0; x < 8; x++) {
        batch_t *b = batch_new(1); load_lane(b, 0, ATT[pr->q0]);
        int pred = pr->q0; printf("   x=%d (b2b1b0=%d%d%d) q0=%+d", x, x>>2&1, x>>1&1, x&1, pr->q0 - Q_MAX);
        for (int s = 0; s < 3; s++) {
            int I = (x >> s & 1) ? pr->i1[s] : pr->i0[s];
            apply_instr(b, 0, I); int lock[LMAX]; settle_s(b, lock);
            pred = pred >= 0 ? T[pred][I] : -1;
            int q = winding(b, 0); double d = -1;
            if (q >= -Q_MAX && q <= Q_MAX && ALIVE[q + Q_MAX]) {
                double th[N];
                for (int i = 0; i < N; i++) th[i] = b->th[i];
                d = dist_orbit(th, ATT[q + Q_MAX]);
            }
            printf("  | s%d pred=%+d phys=%+d%s d=%.3f", s, pred >= 0 ? pred - Q_MAX : -99, q, lock[0] ? "" : "(NOLOCK)", d);
        }
        printf("\n"); batch_free(b);
    }
}

/* ---------------------------------------------------------------------------
 * DIAGNOSIS: for the late-selector MUX, which transitions does every
 * non-robust witness need, and how robust is each? (robust table vs raw)
 * ------------------------------------------------------------------------- */
static int RAW[NS][NIMAX];

/* T := RAW entries whose noisy pass count is >= k (NOP exempt, as in run 02) */
static void set_level(int k){
    for (int q = 0; q < NS; q++) for (int I = 0; I < NI; I++)
        T[q][I] = (RAW[q][I] >= 0 && (I == 0 || PASS[q][I] >= k)) ? RAW[q][I] : -1;
}

/* MUX robustness level: the largest k such that the late-selector MUX is
 * computable using only entries that passed >= k of R noisy trials.
 * k = R is run 02's robust criterion; 0 = not computable even raw.          */
static int mux_level(void){
    set_level(1);
    if (enumerate().mux_any == 0) return 0;
    for (int k = ROBUST_R; k >= 1; k--) { set_level(k); if (enumerate().mux_any > 0) return k; }
    return 0;
}

static void show_mux(int verify_k){
    for (int k = 0; k < 8; k++) {
        int f = MUX8[k], p = W_full[f];
        if (p < 0) continue;
        prog_t pr = recover(p); print_prog(f, &pr, W_fullR[f], "R=");
        if (getenv("TRACE")) trace_prog(&pr);
        if (verify_k) { int un;
            int clean = verify_prog(f, &pr, W_fullR[f], verify_k, 0.0, 0xA3A3ull + k, &un);
            int noisy = verify_prog(f, &pr, W_fullR[f], verify_k, ROBUST_SIG, 0x5EEDull + k, &un);
            printf("      physical: clean %d.%d%%  sigma=%.2f %d.%d%%  (%d unlocked lanes)\n",
                   clean/10, clean%10, ROBUST_SIG, noisy/10, noisy%10, un); }
    }
}

static void run_config(const char *label, FILE *csv, int show, int verify_k){
    measure();
    memcpy(RAW, T, sizeof T);
    int kept = 0, total = 0, mk = -1, dec10 = -1;
    if (ROBUST_R) {
        kept = robust_filter(&total);
        mk = mux_level();
        for (int I = 1; I < NI; I++)                      /* best 1 -> 0 decrement */
            if (RAW[1+Q_MAX][I] == Q_MAX && PASS[1+Q_MAX][I] > dec10) dec10 = PASS[1+Q_MAX][I];
        if (mk > 0 && mk < ROBUST_R && show) {
            set_level(mk); enumerate();
            printf("   best MUX programs use entries passing >= %d/%d:\n", mk, ROBUST_R);
            show_mux(verify_k);
        }
        set_level(ROBUST_R);
    }
    cov_t c = enumerate();
    c.kept = kept; c.total = total;
    char ls[256]; lens_str(ls, sizeof ls);
    printf("%-34s alive=%d %s%d/%d maps=%zu any=%d nz=%d pos=%d  MUX8: any=%d nz=%d pos=%d  mux_k=%d dec10=%d\n",
           ls, c.nalive, ROBUST_R ? "robust=" : "entries=", ROBUST_R ? kept : 0, ROBUST_R ? total : 0,
           c.maps, c.any, c.nz, c.pos, c.mux_any, c.mux_nz, c.mux_pos, mk, dec10);
    if (csv) { fprintf(csv, "\"%s\",%s,%d,%d,%d,%zu,%d,%d,%d,%d,%d,%d,%d,%d\n", ls, label, c.nalive, kept, total,
                       c.maps, c.any, c.nz, c.pos, c.mux_any, c.mux_nz, c.mux_pos, mk, dec10); fflush(csv); }
    if (show && c.any < 256) { printf("   missing (any-readout):");
        for (int f = 0; f < 256; f++) if (W_full[f] < 0) printf(" %02x", f);
        printf("\n"); }
    if (show) show_mux(verify_k);
}

/* ---------------------------------------------------------------------------
 * XCHECK: generalized builder vs build_table() + fork_logic.c's WRITE rows.
 * ------------------------------------------------------------------------- */
static int xcheck(double delta){
    setenv("WRITE", "handoff", 1); build_alphabet();
    int strict_saved = STRICT; STRICT = 0;          /* compare like with like: plain settle() */
    int ref[NS][NIMAX], alive[NS];
    int tab[TBL_SZ]; build_table(0, delta, tab, alive);
    for (int qs = 0; qs < NS; qs++) {                     /* fork_logic load_tables, verbatim logic */
        ref[qs][0] = alive[qs] ? qs : -1;
        for (int s = 0; s < N; s++) for (int a = 0; a < 2; a++) {
            int v = alive[qs] ? tab[TBL_IDX(qs - Q_MAX, s, a)] : DEAD;
            int ok = v > DEAD+1 && v >= -Q_MAX && v <= Q_MAX && alive[v + Q_MAX];
            ref[qs][1 + s*2 + a] = ok ? v + Q_MAX : -1;
        }
        for (int I = 33; I < NI; I++) ref[qs][I] = -1;
        if (!alive[qs]) continue;
        double att[N]; if (!settle_attractor(delta, qs - Q_MAX, att)) continue;
        int L = NI - 33; batch_t *b = batch_new(L);
        for (int l = 0; l < L; l++) { for (int i = 0; i < N; i++) { b->th[i*L+l] = att[i]; b->om[i*L+l] = 0; }
            b->om[LENS_SITE*L+l] = delta; apply_instr(b, l, 33 + l); }
        int lock[LMAX]; settle(b, lock);
        for (int l = 0; l < L; l++) { int v = winding(b, l);
            ref[qs][33+l] = (lock[l] && v >= -Q_MAX && v <= Q_MAX && alive[v+Q_MAX]) ? v + Q_MAX : -1; }
        batch_free(b);
    }
    char spec[64]; snprintf(spec, sizeof spec, "0:%.6f", delta); parse_lens(spec);
    measure();
    int diff = 0, n = 0, adiff = 0;
    for (int q = 0; q < NS; q++) { adiff += alive[q] != ALIVE[q];
        for (int I = 0; I < NI; I++) { n++; if (ref[q][I] != T[q][I]) { diff++;
            char nm[64]; instr_name(I, nm, sizeof nm);
            printf("   diff q=%+d %-28s ref=%d gen=%d\n", q - Q_MAX, nm, ref[q][I] < 0 ? -99 : ref[q][I] - Q_MAX, T[q][I] < 0 ? -99 : T[q][I] - Q_MAX); } } }
    printf("xcheck delta=%.2f: %d/%d entries differ, alive differs in %d slots %s\n",
           delta, diff, n, adiff, diff || adiff ? "" : "(IDENTICAL)");
    STRICT = strict_saved;
    return diff || adiff;
}

int main(int argc, char **argv){
    if (getenv("ROBUST")) ROBUST_R = atoi(getenv("ROBUST"));
    if (getenv("RSIG"))   ROBUST_SIG = atof(getenv("RSIG"));
    if (getenv("STRICT")) STRICT = atoi(getenv("STRICT"));
    if (getenv("CONFIRM")) CONFIRM = atoi(getenv("CONFIRM"));
    build_alphabet();
    const char *cmd = argc > 1 ? argv[1] : "";
    if (!strcmp(cmd, "xcheck")) { int bad = 0;
        for (int k = 2; k < argc; k++) bad |= xcheck(atof(argv[k]));
        return bad; }
    if (!strcmp(cmd, "alphabet")) { char nm[64];
        for (int I = 0; I < NI; I++) { instr_name(I, nm, sizeof nm); printf("%3d %s\n", I, nm); } return 0; }
    if (!strcmp(cmd, "table")) {             /* fork_lens table "<lens spec>" : raw + pass counts */
        parse_lens(argc > 2 ? argv[2] : ""); measure(); memcpy(RAW, T, sizeof T);
        int tot = 0; if (ROBUST_R) robust_filter(&tot);
        char nm[64];
        for (int q = 0; q < NS; q++) { if (!ALIVE[q]) continue;
            for (int I = 0; I < NI; I++) { if (RAW[q][I] < 0) continue; instr_name(I, nm, sizeof nm);
                printf("q=%+d %-28s -> %+d  pass=%d/%d\n", q - Q_MAX, nm, RAW[q][I] - Q_MAX, I ? PASS[q][I] : ROBUST_R, ROBUST_R); } }
        return 0; }
    if (!strcmp(cmd, "run")) {               /* fork_lens run "<lens spec>" [verify_K] */
        parse_lens(argc > 2 ? argv[2] : "");
        run_config("run", NULL, 1, argc > 3 ? atoi(argv[3]) : 0);
        return 0; }
    if (!strcmp(cmd, "list")) {              /* fork_lens list <file of lens specs> <out.csv> : one config per line */
        FILE *in = fopen(argv[2], "r"); FILE *csv = fopen(argv[3], "a");
        if (!in || !csv) { perror("list"); return 1; }
        char line[256];
        while (fgets(line, sizeof line, in)) { line[strcspn(line, "\r\n")] = 0;
            parse_lens(!strcmp(line, "pristine") ? "" : line); run_config(getenv("LABEL") ? getenv("LABEL") : "list", csv, 0, 0); }
        fclose(in); fclose(csv); return 0; }
    fprintf(stderr, "usage: fork_lens xcheck <delta...>\n"
                    "       fork_lens alphabet\n"
                    "       fork_lens table \"<site:delta,...>\"\n"
                    "       fork_lens run   \"<site:delta,...>\" [verify_trials]\n"
                    "       fork_lens list  <specs.txt> <out.csv>\n"
                    "env: ROBUST=R RSIG=sigma WRITE=vm|handoff RAMP_WIDTHS=16[,4,6,8]\n");
    return 2;
}
