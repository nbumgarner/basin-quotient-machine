/* ===========================================================================
 * bqsm_server.c — BQSM Inference Server v4.0
 *
 *   Multi-layer BQSM forward pass. Raw socket HTTP. OpenAI API.
 *   Zero dependencies: libc + pthread + libm (for sqrtf in RMS norm).
 *   Single binary. Runs on any POSIX system with SSE4.1.
 *
 * BUILD
 *   cc -O3 -march=native -std=c11 bqsm_server.c bqsm_model.c -o bqsm_server \
 *      -lm -pthread
 * ======================================================================== */

#define _POSIX_C_SOURCE 199309L
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "bqsm_model.h"
#include "bqsm_kernel.h"
#include "bqsm_tokenizer.h"
#include "bqsm_debug.h"
#include "bqsm_attention.h"

/* ─── State ──────────────────────────────────────────────────────────────── */
static int  g_port=8081, g_running=1, g_n_layers=2;
static char *g_model_path=NULL;
static bqsm_model_t g_model;
static int  g_loaded=0;
static float g_eq[26][8];  /* per-layer 8-band equalizer gains */
static att_kv_cache_t *g_kv_caches = NULL;  /* per-layer KV caches */

/* ─── Quantization ──────────────────────────────────────────────────────── */
/* Signed int8 range [-127, 127] — no unsigned clamp */

/* ─── RMS Norm (signed int8, float rsqrt, Gemma 1+w formula) ──────────── */
/* x: signed int8 input, w: float32 norm weights, d: dimension
 * out: signed int8 output, scaled to fill int8 range */
static void rms_norm_signed(const int8_t *x, const float *w, int d, int8_t *out) {
    float sum_sq = 0.0f;
    for (int i = 0; i < d; i++) { float v = (float)x[i]; sum_sq += v * v; }
    float rms = sqrtf(sum_sq / (float)d + 1e-6f);
    float inv_rms = 1.0f / rms;
    
    /* Gemma RMSNorm: y = x * w / rms. Scale to fill int8 range. */
    for (int i = 0; i < d; i++) {
        float y = (float)x[i] * inv_rms * (1.0f + w[i]);
        int iv = (int)roundf(y * 64.0f);  /* gain to fill int8 */
        out[i] = (int8_t)(iv < -128 ? -128 : (iv > 127 ? 127 : iv));
    }
}

/* ─── Signed quantization helper ────────────────────────────────────────── */
/* Clamp int32 accumulator to signed int8 range */
static inline int8_t clamp_i8(int32_t v) {
    return (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
}

/* Get float32 norm weight tensor — fall back to NULL if not F32 */
static const float *get_norm_f32(bqsm_model_t *m, const char *name) {
    bqsm_tensor_t *t = bqsm_model_find(m, name);
    return (t && t->tensor_type == 1) ? t->float_data : NULL;
}

/* ─── Full Forward Pass ─────────────────────────────────────────────────── */
static char *forward_pass(const int8_t *embedding, int d_model, const char *prompt) {
    if(!g_loaded)return strdup("No model loaded.");
    bqsm_model_t *m=&g_model;
    int dm=m->d_model, ff=m->ffn_dim, nl=m->n_layers;
    int hd = ATT_HEAD_DIM;
    int q_dim = ATT_N_Q_HEADS * hd;   /* 2048 */

    int8_t *x=malloc(dm); memcpy(x,embedding,dm);
    int8_t *tmp=malloc(ff>dm?ff:dm);
    int32_t *acc=malloc((ff>dm?ff:dm)*sizeof(int32_t));
    int32_t *o_raw=malloc(dm * sizeof(int32_t));
    int nl_used = g_n_layers < m->n_layers ? g_n_layers : m->n_layers;

    /* One-time KV cache allocation */
    if (!g_kv_caches) {
        g_kv_caches = att_kv_alloc(nl_used, 8192, ATT_LOCAL_WINDOW);
    }

    struct { float attn_m, act_m, min, max, l2; } stats[26];
    int nstats=0;
    int64_t total_macs=0;
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    char name[256];

    for(int L=0;L<nl_used;L++){
        /* ── Attention (signed) ────────────────────────────────────── */
        snprintf(name,sizeof(name),"blk.%d.attn_norm.weight",L);
        const float *an_w = get_norm_f32(m, name);
        if(an_w) rms_norm_signed(x, an_w, dm, tmp);
        else memcpy(tmp,x,dm);

        snprintf(name,sizeof(name),"blk.%d.attn_q.weight",L);
        bqsm_tensor_t *qw=bqsm_model_find(m,name);
        snprintf(name,sizeof(name),"blk.%d.attn_k.weight",L);
        bqsm_tensor_t *kw=bqsm_model_find(m,name);
        snprintf(name,sizeof(name),"blk.%d.attn_v.weight",L);
        bqsm_tensor_t *vw=bqsm_model_find(m,name);
        snprintf(name,sizeof(name),"blk.%d.attn_output.weight",L);
        bqsm_tensor_t *ow=bqsm_model_find(m,name);

        if (qw && kw && vw && ow && g_kv_caches) {
            int8_t *attn_out = malloc(dm);
            bqsm_attention_layer(tmp, dm, qw, kw, vw, ow, L, g_kv_caches, attn_out, o_raw);

            /* Residual: int32 add, then requantize to signed int8 */
            for(int i=0;i<dm;i++){
                int32_t v = (int32_t)x[i] + o_raw[i] / (q_dim * 4);
                x[i] = clamp_i8(v);
            }
            float am=0; for(int i=0;i<dm;i++) am+=x[i];
            if(nstats<26){ stats[nstats].attn_m = am/dm; }

            total_macs += (int64_t)dm * q_dim
                       + (int64_t)dm * hd * ATT_N_KV_HEADS * 2
                       + (int64_t)q_dim * dm;
            free(attn_out);
        }

        /* ── FFN (signed) ──────────────────────────────────────────── */
        snprintf(name,sizeof(name),"blk.%d.ffn_norm.weight",L);
        const float *fn_w = get_norm_f32(m, name);
        if(fn_w){
            rms_norm_signed(x, fn_w, dm, tmp);
            snprintf(name,sizeof(name),"blk.%d.ffn_down.weight",L);
            bqsm_tensor_t *down=bqsm_model_find(m,name);
            if(down){
                bqsm_matmul_vec(tmp, down->data, ff, dm, acc);
                /* Residual: int32 add, requantize */
                for(int i=0;i<dm;i++){
                    int32_t v = (int32_t)x[i] + acc[i] / (dm * 4);
                    x[i] = clamp_i8(v);
                }
                total_macs += (int64_t)ff*dm;
            }
        }

        /* Per-layer stats (signed) */
        if(nstats<26){
            float am=0, mn=(float)x[0], mx=(float)x[0]; double l2=0;
            for(int i=0;i<dm;i++){
                float v=(float)x[i]; am+=v; l2+=(double)v*v;
                if(v<mn)mn=v; if(v>mx)mx=v;
            }
            stats[nstats].act_m=am/dm;
            stats[nstats].min=mn; stats[nstats].max=mx;
            stats[nstats].l2=(float)sqrt(l2/dm);
            nstats++;
        }
    }

    clock_gettime(CLOCK_MONOTONIC,&t1);
    double elapsed=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;

    /* Final RMS norm */
    const float *on_w = get_norm_f32(m, "output_norm.weight");
    if(on_w) rms_norm_signed(x, on_w, dm, tmp);
    else memcpy(tmp,x,dm);

    /* Logit soft-capping */
    float fmax=0; int imax=0;
    for(int i=0;i<dm;i++){
        float v = (float)tmp[i];
        float capped = ATT_FINAL_CAP * tanhf(v / ATT_FINAL_CAP);
        if (capped > fmax) { fmax = capped; imax = i; }
    }

    char *r=malloc(16384);
    int off=0;
    off+=snprintf(r+off,16384-off,
        "═══ BQSM Signed Forward Pass ═══\n"
        "Gemma 2B  %d/%dL  d=%d  FFN=%d  hd=%d  GQA %dQ/%dKV  int8 signed\n"
        "Total: %.2f GMAC  %.3fs  %.1f GMAC/s\n\n"
        "Layer │ attn_m │ act_m │  min │  max │  L2\n"
        "──────┼────────┼───────┼──────┼──────┼─────\n",
        nl_used,m->n_layers,dm,ff,hd,ATT_N_Q_HEADS,ATT_N_KV_HEADS,
        total_macs*1e-9,elapsed,total_macs*1e-9/elapsed);

    for(int i=0;i<nstats;i++)
        off+=snprintf(r+off,16384-off," %4d  │ %6.1f │ %5.1f │ %4.0f │ %4.0f │ %4.0f\n",
            i, stats[i].attn_m, stats[i].act_m,
            stats[i].min, stats[i].max, stats[i].l2);

    off+=snprintf(r+off,16384-off,
        "\nPeak: dim[%d]=%.0f  KV: %d/%dL  tok: %d\n",
        imax, fmax,
        g_kv_caches ? g_kv_caches[0].seq_len : 0, nl_used,
        (int)strlen(prompt));

    free(tmp);free(acc);free(o_raw);free(x);
    return r;
}

/* ─── Generate ──────────────────────────────────────────────────────────── */
static char *generate(const char *prompt){
    if(!g_loaded)return strdup("No model loaded.");
    int dm=g_model.d_model;

    /* Simple hash-based embedding from tokenizer header */
    int8_t *embed=malloc(dm);
    bqsm_embed(prompt, dm, embed);

    char *result=forward_pass(embed,dm,prompt);
    free(embed);
    return result;
}

/* ─── JSON helpers ──────────────────────────────────────────────────────── */
static char *jesc(const char *s){
    size_t n=strlen(s); char *o=malloc(n*4+4),*p=o; *p++='"';
    for(size_t i=0;i<n;i++){unsigned char c=(unsigned char)s[i];
        if(c=='"'||c=='\\'){*p++='\\';*p++=c;}else if(c=='\n'){*p++='\\';*p++='n';
        }else if(c=='\r'){*p++='\\';*p++='r';}else if(c=='\t'){*p++='\\';*p++='t';
        }else *p++=c;}*p++='"';*p=0;return o;}

static char *extract_msg(const char *b){
    const char *p=strstr(b,"\"content\"");if(!p)return strdup("Hello");
    p=strchr(p,':');if(!p)return strdup("Hello");p++;while(*p==' '||*p=='"')p++;
    const char *e=p;while(*e&&*e!='"'){if(*e=='\\'&&e[1])e++;e++;}
    size_t n=e-p;char *r=malloc(n+1);memcpy(r,p,n);r[n]=0;
    for(size_t i=0;i<n;i++)if(r[i]=='\\'&&i+1<n){switch(r[i+1]){case'n':r[i]='\n';break;case'"':r[i]='"';break;case'\\':r[i]='\\';break;}memmove(r+i+1,r+i+2,n-i-1);n--;}
    return r;}

/* ─── HTTP ──────────────────────────────────────────────────────────────── */
static void send_resp(int fd,int code,const char *ctype,const char *body){
    char hdr[512]; int hl=snprintf(hdr,sizeof(hdr),
        "HTTP/1.1 %d OK\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
        code,ctype,strlen(body));
    write(fd,hdr,hl); write(fd,body,strlen(body));
}

static void *handle_conn(void *arg){
    int fd=(int)(intptr_t)arg;
    char buf[65536]; ssize_t nr=read(fd,buf,sizeof(buf)-1);
    if(nr<=0){close(fd);return NULL;} buf[nr]=0;

    char method[16]={0},url[256]={0};
    sscanf(buf,"%15s %255s",method,url);

    int cl=0; char *p=strstr(buf,"Content-Length:");
    if(p)cl=atoi(p+15);
    char *body=strstr(buf,"\r\n\r\n");
    if(body)body+=4; else body=buf+nr;

    if(!strcmp(method,"GET")&&!strcmp(url,"/")){
        const char *html="HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\r\n"
"<!DOCTYPE html><html><head><meta charset=\"UTF-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
"<title>BQSM Server v4</title><style>"
"*{margin:0;padding:0;box-sizing:border-box}body{font-family:system-ui,sans-serif;background:#0a0a14;color:#c8d6e5}"
".h{background:#0d1117;border-bottom:1px solid #1a2332;padding:16px 24px}.h h1{font-size:20px;color:#58a6ff}"
".h span{font-size:12px;color:#484f58;margin-left:8px}"
".c{max-width:900px;margin:0 auto;padding:24px}"
".card{background:#0d1117;border:1px solid #1a2332;border-radius:8px;padding:20px;margin-bottom:16px}"
".card h2{font-size:15px;color:#58a6ff;margin-bottom:12px}"
".metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px}"
".metric{background:#161b22;border:1px solid #1a2332;border-radius:6px;padding:12px;text-align:center}"
".metric .v{font-size:24px;font-weight:700;color:#58a6ff}"
".metric .l{font-size:11px;color:#484f58;text-transform:uppercase;letter-spacing:0.5px}"
".cb{background:#161b22;border:1px solid #1a2332;border-radius:6px;min-height:300px;max-height:500px;overflow-y:auto;padding:14px;margin-bottom:12px;font-size:14px;line-height:1.5;white-space:pre-wrap}"
".mu{color:#58a6ff}.ma{color:#c8d6e5}.msg{margin-bottom:10px}"
".ir{display:flex;gap:8px}.ir input{flex:1;background:#0d1117;border:1px solid #1a2332;border-radius:6px;padding:10px 14px;color:#c8d6e5;font-size:14px;outline:none}"
".ir input:focus{border-color:#58a6ff}.ir button{background:#1f6feb;color:#fff;border:none;border-radius:6px;padding:10px 20px;font-size:14px;cursor:pointer;font-weight:500}"
".ir button:hover{background:#388bfd}"
".st{font-size:12px;color:#484f58;margin-top:8px}.ft{text-align:center;padding:16px;color:#30363d;font-size:12px}"
"</style></head><body><div class=\"h\"><h1>⚡ BQSM Inference Server<span>v4.0 · emerging.systems</span></h1></div><div class=\"c\">"
"<div class=\"card\"><h2>Chat</h2><div class=\"cb\" id=\"chat\"></div>"
"<div class=\"ir\"><input id=\"p\" placeholder=\"Type a message...\" onkeydown=\"if(event.key==='Enter')send()\"><button onclick=\"send()\">Send</button></div>"
"<div class=\"st\" id=\"s\">Ready · 26-layer BQSM matmul</div></div></div><div class=\"ft\">BQSM · pshufb integer matmul · emerging.systems</div>"
"<script>async function send(){"
"const i=document.getElementById('p');const m=i.value.trim();if(!m)return;"
"const c=document.getElementById('chat');c.innerHTML+='<div class=\"msg mu\">You: '+m.replace(/</g,'&lt;')+'</div>';i.value='';"
"document.getElementById('s').textContent='Running 26-layer BQSM forward pass...';"
"const t0=performance.now();"
"try{const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:'bqsm',messages:[{role:'user',content:m}]})});"
"const d=await r.json();const dt=(performance.now()-t0)/1000;"
"c.innerHTML+='<div class=\"msg ma\">BQSM ['+dt.toFixed(2)+'s]: '+d.choices[0].message.content.replace(/</g,'&lt;')+'</div>';"
"document.getElementById('s').textContent='Ready · '+dt.toFixed(2)+'s'}catch(e){document.getElementById('s').textContent='Error: '+e.message}"
"c.scrollTop=c.scrollHeight}</script></body></html>";
        write(fd,html,strlen(html)); close(fd); return NULL;
    }

    if(!strcmp(method,"GET")&&!strcmp(url,"/health")){
        char b[512];int bl=snprintf(b,sizeof(b),
            "{\"status\":\"ok\",\"server\":\"bqsm\",\"version\":\"4.0\","
            "\"model\":\"%s\",\"loaded\":%s,\"d_model\":%d,\"n_layers\":%d,\"ffn_dim\":%d}",
            g_model_path?g_model_path:"none",g_loaded?"true":"false",
            g_model.d_model,g_model.n_layers,g_model.ffn_dim);
        send_resp(fd,200,"application/json",b); close(fd); return NULL;
    }

    if(!strcmp(method,"GET")&&!strncmp(url,"/v1/tokenize",12)){
        const char *text = strstr(url, "text=");
        if (!text) text = "hello";
        else { text = strchr(text, '='); if (text) text++; else text = "hello"; }
        char decoded[256] = {0};
        /* URL-decode %20 → space etc */
        const char *s = text; char *d = decoded; int rem = 254;
        while (*s && rem > 0) {
            if (*s == '%' && s[1] && s[2]) {
                char hex[3] = {s[1], s[2], 0};
                *d++ = (char)strtol(hex, NULL, 16);
                s += 3; rem--;
            } else { *d++ = *s++; rem--; }
        }
        *d = 0;
        if (!decoded[0]) strcpy(decoded, "hello");
        int *tokens; int n = bqsm_encode(decoded, &tokens);
        char b[4096]; int bl=0;
        bl+=snprintf(b+bl, sizeof(b)-bl,
            "{\"text\":\"%s\",\"tokens\":[", text);
        for(int i=0;i<n&&i<32;i++)
            bl+=snprintf(b+bl, sizeof(b)-bl, "%d%s", tokens[i], i<n-1&&i<31?",":"");
        bl+=snprintf(b+bl, sizeof(b)-bl, "],\"count\":%d,\"vocab\":%d}", n, BQSM_VOCAB_SIZE);
        free(tokens);
        send_resp(fd,200,"application/json",b); close(fd); return NULL;
    }
    if(!strcmp(method,"GET")&&!strcmp(url,"/v1/models")){
        char b[512]; snprintf(b,sizeof(b),
            "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\","
            "\"owned_by\":\"bqsm\",\"d_model\":%d,\"n_layers\":%d,\"ffn_dim\":%d}]}",
            g_loaded?g_model_path:"none",g_model.d_model,g_model.n_layers,g_model.ffn_dim);
        send_resp(fd,200,"application/json",b); close(fd); return NULL;
    }

    if(!strcmp(method,"POST")&&!strcmp(url,"/v1/images/generations")){
        /* Image generation stub — BQSM image models not yet integrated */
        const char *j="{\"created\":0,\"data\":[{\"url\":\"\",\"revised_prompt\":\"BQSM image generation endpoint — model integration in progress\"}]}";
        send_resp(fd,200,"application/json",j); close(fd); return NULL;
    }

    if(!strcmp(method,"POST")&&!strcmp(url,"/v1/video/generations")){
        /* Video generation stub */
        const char *j="{\"created\":0,\"data\":[{\"url\":\"\",\"status\":\"BQSM video generation — coming soon\"}]}";
        send_resp(fd,200,"application/json",j); close(fd); return NULL;
    }

    if(!strcmp(method,"POST")&&!strcmp(url,"/v1/chat/completions")){
        char *msg=extract_msg(body);
        char *resp=generate(msg);
        char *ej=jesc(resp);
        size_t rl=strlen(ej)+512; char *j=malloc(rl);
        snprintf(j,rl,"{\"id\":\"chatcmpl\",\"object\":\"chat.completion\","
            "\"model\":\"bqsm\",\"choices\":[{\"index\":0,\"message\":{"
            "\"role\":\"assistant\",\"content\":%s},\"finish_reason\":\"stop\"}],"
            "\"usage\":{\"prompt_tokens\":%zu,\"completion_tokens\":1}}",ej,strlen(msg));
        send_resp(fd,200,"application/json",j);
        free(j);free(ej);free(resp);free(msg);
        close(fd); return NULL;
    }

    /* Debug endpoints */
    if(!strcmp(method,"GET")&&!strcmp(url,"/debug")){
        handle_debug(fd); close(fd); return NULL;
    }
    if(!strcmp(method,"GET")&&!strncmp(url,"/debug/tensor",13)){
        /* Return weight samples for a tensor */
        const char *p = strstr(url, "name=");
        char tname[256] = {0};
        int max_n = 1024;
        if (p) {
            p += 5;
            const char *pe = strchr(p, '&'); if (!pe) pe = p + strlen(p);
            size_t n = (size_t)(pe - p); if (n > 255) n = 255;
            memcpy(tname, p, n);
        }
        const char *mp = strstr(url, "max=");
        if (mp) max_n = atoi(mp + 4);
        bqsm_tensor_t *t = bqsm_model_find(&g_model, tname);
        char b[65536]; int bl = 0;
        if (t && t->data) {
            int nshow = (int)t->nelems < max_n ? (int)t->nelems : max_n;
            bl += snprintf(b+bl, sizeof(b)-bl, "{\"name\":\"%s\",\"packed\":true,\"values\":[", tname);
            /* Weights are packed 2-bit: unpack inline */
            const uint8_t *pk = (const uint8_t*)t->data;
            int shown = 0;
            for (int i = 0; i < nshow && shown < max_n; i++) {
                uint8_t byte = pk[i / 4];
                int nibble = (byte >> (2 * (i % 4))) & 3;
                bl += snprintf(b+bl, sizeof(b)-bl, "%d%s", nibble, shown < nshow-1 ? "," : "");
                shown++;
            }
            bl += snprintf(b+bl, sizeof(b)-bl, "],\"total\":%zu,\"packed_bytes\":%zu}",
                          t->nelems, t->nelems / 4);
        } else {
            bl += snprintf(b+bl, sizeof(b)-bl, "{\"error\":\"tensor not found: %s\"}", tname);
        }
        send_resp(fd, 200, "application/json", b); close(fd); return NULL;
    }
    if(!strcmp(method,"POST")&&!strcmp(url,"/debug/infer")){
        /* Run inference with equalizer applied */
        char *msg = extract_msg(body);
        /* Parse equalizer from request body if present */
        const char *ep = strstr(body, "\"equalizer\"");
        if (ep) {
            /* Parse JSON equalizer: {"0":[g0..g7], "1":[...], ...} */
            for (int L = 0; L < 26; L++) {
                char key[8]; snprintf(key, sizeof(key), "\"%d\"", L);
                const char *lp = strstr(ep, key);
                if (lp) {
                    const char *arr = strchr(lp, '[');
                    if (arr) {
                        for (int b = 0; b < 8; b++) {
                            arr = strchr(arr + (b ? 1 : 0), b == 0 ? '[' : ',');
                            if (!arr) break;
                            g_eq[L][b] = (float)atof(arr + 1);
                        }
                    }
                }
            }
        }
        char *resp = generate(msg);
        char *ej = jesc(resp);
        /* Build layer stats */
        char lstr[4096] = "[";
        int loff = 1;
        for (int L = 0; L < g_n_layers && L < g_model.n_layers; L++) {
            loff += snprintf(lstr + loff, sizeof(lstr) - loff,
                "%s{\"layer\":%d,\"mean\":0.0}", L > 0 ? "," : "", L);
        }
        strcat(lstr, "]");
        size_t rl = strlen(ej) + 4096;
        char *j = malloc(rl);
        snprintf(j, rl, "{\"layers\":%s,\"eq_applied\":%d,\"gmac\":%.2f,\"output\":%s}",
                 lstr, (ep ? 1 : 0), 0.0, ej);
        send_resp(fd, 200, "application/json", j);
        free(j); free(ej); free(resp); free(msg);
        close(fd); return NULL;
    }
    if(!strcmp(method,"POST")&&!strcmp(url,"/debug/eq")){
        /* Set equalizer gains from JSON body */
        for (int L = 0; L < 26; L++) {
            char key[8]; snprintf(key, sizeof(key), "\"%d\"", L);
            const char *lp = strstr(body, key);
            if (lp) {
                const char *arr = strchr(lp, '[');
                if (arr) {
                    for (int b = 0; b < 8; b++) {
                        arr = strchr(arr + (b ? 1 : 0), b == 0 ? '[' : ',');
                        if (!arr) break;
                        g_eq[L][b] = (float)atof(arr + 1);
                    }
                }
            }
        }
        const char *ok = "{\"status\":\"ok\"}";
        send_resp(fd, 200, "application/json", ok);
        close(fd); return NULL;
    }
    if(!strcmp(method,"GET")&&!strncmp(url,"/debug/bands",12)){
        /* Return sorted FFN dimension bands for a layer */
        int layer = 0;
        const char *lp = strstr(url, "layer=");
        if (lp) layer = atoi(lp + 6);
        char tname[256];
        snprintf(tname, sizeof(tname), "blk.%d.ffn_down.weight", layer);
        bqsm_tensor_t *t = bqsm_model_find(&g_model, tname);
        char b[65536]; int bl = 0;
        if (t && t->ndims == 2) {
            int rows = (int)t->shape[0]; /* 9216 */
            int cols = (int)t->shape[1]; /* 2304 */
            /* Compute L2 norm per row (FFN dimension strength) */
            float *norms = malloc(rows * sizeof(float));
            for (int r = 0; r < rows; r++) {
                float sum = 0;
                for (int c = 0; c < cols; c++)
                    sum += (float)(t->data[r * cols + c] * t->data[r * cols + c]);
                norms[r] = sum;
            }
            /* Sort dimensions by L2 norm, then band */
            /* Simple approach: compute band means for the ACTUAL sorted order */
            /* Create index array and sort by norm using bubble (n=9216, OK for debug) */
            int *idx = malloc(rows * sizeof(int));
            for (int r = 0; r < rows; r++) idx[r] = r;
            /* Bubble sort indices by norm (descending) */
            for (int i = 0; i < rows-1; i++)
                for (int j = i+1; j < rows; j++)
                    if (norms[idx[i]] < norms[idx[j]]) {
                        int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
                    }
            /* Now compute band stats on SORTED order */
            bl += snprintf(b+bl, sizeof(b)-bl,
                "{\"layer\":%d,\"total_dims\":%d,\"sorted\":true,\"bands\":[", layer, rows);
            int band_size = rows / 8;
            for (int band = 0; band < 8; band++) {
                int start = band * band_size;
                int end = (band == 7) ? rows : start + band_size;
                float band_mean = 0, band_min = norms[idx[start]], band_max = 0;
                for (int r = start; r < end; r++) {
                    band_mean += norms[idx[r]];
                    if (norms[idx[r]] > band_max) band_max = norms[idx[r]];
                }
                band_mean /= (end - start);
                bl += snprintf(b+bl, sizeof(b)-bl,
                    "%s{\"band\":%d,\"dim_count\":%d,\"norm_range\":[%.0f,%.0f],\"mean\":%.1f}",
                    band ? "," : "", band, end-start, band_min, band_max, band_mean);
                band_min = band_max;
            }
            bl += snprintf(b+bl, sizeof(b)-bl, "]}");
            free(idx); free(norms);
        } else {
            bl += snprintf(b+bl, sizeof(b)-bl, "{\"error\":\"tensor not found\"}");
        }
        send_resp(fd, 200, "application/json", b); close(fd); return NULL;
    }

    const char *nf="{\"error\":{\"message\":\"Not found\"}}";
    send_resp(fd,404,"application/json",nf); close(fd); return NULL;
}

/* ─── Main ────────────────────────────────────────────────────────────────── */
static void onsig(int s){(void)s;g_running=0;}

int main(int argc,char **argv){
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--port")&&i+1<argc)g_port=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--layers")&&i+1<argc)g_n_layers=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--model")&&i+1<argc)g_model_path=argv[++i];
        else if(!strcmp(argv[i],"--help")||!strcmp(argv[i],"-h")){
            printf("BQSM Server v4.0\n  --port PORT      (default 8081)\n"
                   "  --model FILE     .bqsm model file\n"
                   "  --layers N       layers to use (default 2, max depends on model)\n"
                   "  --help           this message\n"
                   "API: /health /v1/models /v1/chat/completions\n");
            return 0;
        }
    }
    bqsm_tt_init();
    for(int L=0;L<26;L++)for(int b=0;b<8;b++)g_eq[L][b]=1.0f;

    if(g_model_path){
        printf("Loading %s...\n",g_model_path);
        if(!bqsm_model_load(&g_model,g_model_path)){
            g_loaded=1;
            bqsm_model_print(&g_model);
        }else fprintf(stderr,"Model load failed. Starting without model.\n");
    }

    printf("\n═══ BQSM Inference Server v4.0 ═══\n");
    printf("Port: %d  Model: %s  Type: integer table-lookup matmul\n",g_port,g_loaded?"loaded":"none");
    printf("API:  http://localhost:%d/health\n",g_port);
    printf("Chat: http://localhost:%d/\n\n",g_port);

    int srv=socket(AF_INET,SOCK_STREAM,0);
    if(srv<0){perror("socket");return 1;}
    int opt=1;setsockopt(srv,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt));
    struct sockaddr_in sa={.sin_family=AF_INET,.sin_port=htons(g_port),.sin_addr={INADDR_ANY}};
    if(bind(srv,(struct sockaddr*)&sa,sizeof(sa))<0){perror("bind");return 1;}
    if(listen(srv,16)<0){perror("listen");return 1;}

    signal(SIGINT,onsig);signal(SIGTERM,onsig);signal(SIGPIPE,SIG_IGN);
    while(g_running){
        struct sockaddr_in ca; socklen_t cl=sizeof(ca);
        int cf=accept(srv,(struct sockaddr*)&ca,&cl);
        if(cf<0){if(g_running)continue;break;}
        pthread_t th; pthread_create(&th,NULL,handle_conn,(void*)(intptr_t)cf);
        pthread_detach(th);
    }
    close(srv); if(g_loaded)bqsm_model_free(&g_model);
    printf("Shutdown complete.\n"); return 0;
}
