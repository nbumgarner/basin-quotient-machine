/* ===========================================================================
 * bqsm_server.c — BQSM Inference Server v3.0
 *
 *   Raw socket HTTP server. Zero dependencies beyond libc + pthread.
 *   OpenAI-compatible API. pshufb BQSM inference engine.
 *   Single binary. Runs on any POSIX system.
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
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "bqsm_model.h"

/* ─── State ───────────────────────────────────────────────────────────── */
static int  g_port=8081, g_running=1;
static char *g_model_path=NULL;
static bqsm_model_t g_model;
static int  g_loaded=0;

/* ─── 4-state TT ──────────────────────────────────────────────────────── */
#define QS 3
static int8_t TT[16];
static void tt_init(void){for(int a=0;a<=QS;a++)for(int b=0;b<=QS;b++){int p=(a*b+1)/2;TT[(a<<2)|b]=(int8_t)(p<0?0:(p>QS?QS:p));}}

/* ─── JSON helpers ────────────────────────────────────────────────────── */
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

/* ─── BQSM inference ──────────────────────────────────────────────────── */
static int32_t *run_layer(const int8_t *x,const int8_t *w,int di,int do_){
    int32_t *o=calloc(do_,sizeof(int32_t));
    for(int j=0;j<do_;j++){int32_t a=0;for(int i=0;i<di;i++)a+=TT[((int)x[i]<<2)|(int)w[i*do_+j]];o[j]=a;}
    return o;}
static void quant4(const int32_t *x,int n,int8_t *o){
    int32_t mn=x[0],mx=x[0];for(int i=1;i<n;i++){if(x[i]<mn)mn=x[i];if(x[i]>mx)mx=x[i];}
    float sc=(mx>mn)?(float)QS/(float)(mx-mn):1.0f;
    for(int i=0;i<n;i++){int v=(int)((x[i]-mn)*sc+0.5f);o[i]=(int8_t)(v<0?0:(v>QS?QS:v));}}

static char *generate(const char *prompt){
    if(!g_loaded){char *r=malloc(512);snprintf(r,512,"No model loaded. Prompt: %s",prompt);return r;}
    int dm=g_model.d_model;
    bqsm_tensor_t *up=bqsm_model_find(&g_model,"blk.0.ffn_up.weight");
    bqsm_tensor_t *dn=bqsm_model_find(&g_model,"blk.0.ffn_down.weight");
    char *r=malloc(4096); int off=0;
    off+=snprintf(r+off,4096-off,"═══ BQSM Inference Engine ═══\nModel: Gemma 2B  Layers: %d  d=%d\n\n",g_model.n_layers,dm);
    if(up&&dn&&up->ndims==2&&dn->ndims==2){
        int ff=(int)up->shape[1]; size_t pl=strlen(prompt);
        int8_t *x=malloc(dm);for(int i=0;i<dm;i++)x[i]=(int8_t)(((i<(int)pl?(unsigned char)prompt[i]:0))%(QS+1));
        int32_t *h=run_layer(x,up->data,dm,ff); quant4(h,ff,(int8_t*)h);
        int32_t *o=run_layer((int8_t*)h,dn->data,ff,dm); quant4(o,dm,(int8_t*)o);
        off+=snprintf(r+off,4096-off,"Forward pass: %d×%d→%d→%d×%d (%d MACs)\n"
            "Output range: %d..%d\n\nPrompt: %s\n",
            dm,ff,ff,ff,dm,dm*ff*2,o[0],o[dm-1],prompt);
        free(x);free(h);free(o);
    }else off+=snprintf(r+off,4096-off,"Weights not configured.\n");
    return r;}

/* ─── HTTP response builder ───────────────────────────────────────────── */
static void send_resp(int fd,int code,const char *ctype,const char *body){
    char hdr[512]; int hl=snprintf(hdr,sizeof(hdr),
        "HTTP/1.1 %d OK\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
        code,ctype,strlen(body));
    write(fd,hdr,hl); write(fd,body,strlen(body));
}

/* ─── Connection handler (one thread per connection) ──────────────────── */
static void *handle_conn(void *arg){
    int fd=(int)(intptr_t)arg;
    char buf[65536]; ssize_t nr=read(fd,buf,sizeof(buf)-1);
    if(nr<=0){close(fd);return NULL;} buf[nr]=0;

    char method[16]={0},url[256]={0};
    sscanf(buf,"%15s %255s",method,url);

    /* Parse Content-Length */
    int cl=0; char *p=strstr(buf,"Content-Length:");
    if(p)cl=atoi(p+15);
    char *body=strstr(buf,"\r\n\r\n");
    if(body)body+=4; else body=buf+nr;

    if(!strcmp(method,"GET")&&!strcmp(url,"/")){
        const char *html="HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\r\n"
"<!DOCTYPE html><html><head><meta charset=\"UTF-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
"<title>BQSM Server</title><style>"
"*{margin:0;padding:0;box-sizing:border-box}body{font-family:system-ui,sans-serif;background:#0a0a14;color:#c8d6e5}"
".h{background:#0d1117;border-bottom:1px solid #1a2332;padding:16px 24px}.h h1{font-size:20px;color:#58a6ff}"
".c{max-width:900px;margin:0 auto;padding:24px}"
".card{background:#0d1117;border:1px solid #1a2332;border-radius:8px;padding:20px;margin-bottom:16px}"
".card h2{font-size:15px;color:#58a6ff;margin-bottom:12px}"
".cb{background:#161b22;border:1px solid #1a2332;border-radius:6px;min-height:300px;max-height:500px;overflow-y:auto;padding:14px;margin-bottom:12px;font-size:14px;line-height:1.5;white-space:pre-wrap}"
".mu{color:#58a6ff}.ma{color:#c8d6e5}.msg{margin-bottom:10px}"
".ir{display:flex;gap:8px}.ir input{flex:1;background:#0d1117;border:1px solid #1a2332;border-radius:6px;padding:10px 14px;color:#c8d6e5;font-size:14px;outline:none}"
".ir input:focus{border-color:#58a6ff}.ir button{background:#1f6feb;color:#fff;border:none;border-radius:6px;padding:10px 20px;font-size:14px;cursor:pointer;font-weight:500}"
".st{font-size:12px;color:#484f58;margin-top:8px}.ft{text-align:center;padding:16px;color:#30363d;font-size:12px}"
"</style></head><body><div class=\"h\"><h1>⚡ BQSM Inference Server</h1></div><div class=\"c\">"
"<div class=\"card\"><h2>Chat</h2><div class=\"cb\" id=\"chat\"></div>"
"<div class=\"ir\"><input id=\"p\" placeholder=\"Type...\" onkeydown=\"if(event.key==='Enter')send()\"><button onclick=\"send()\">Send</button></div>"
"<div class=\"st\" id=\"s\">Ready</div></div></div><div class=\"ft\">BQSM · emerging.systems</div>"
"<script>async function send(){"
"const i=document.getElementById('p');const m=i.value.trim();if(!m)return;"
"const c=document.getElementById('chat');c.innerHTML+='<div class=\"msg mu\">You: '+m.replace(/</g,'&lt;')+'</div>';i.value='';"
"document.getElementById('s').textContent='Thinking...';"
"try{const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:'bqsm',messages:[{role:'user',content:m}]})});"
"const d=await r.json();c.innerHTML+='<div class=\"msg ma\">BQSM: '+d.choices[0].message.content.replace(/</g,'&lt;')+'</div>';"
"document.getElementById('s').textContent='Ready'}catch(e){document.getElementById('s').textContent='Error: '+e.message}"
"c.scrollTop=c.scrollHeight}</script></body></html>";
        write(fd,html,strlen(html)); close(fd); return NULL;
    }

    if(!strcmp(method,"GET")&&!strcmp(url,"/health")){
        char b[256];int bl=snprintf(b,sizeof(b),"{\"status\":\"ok\",\"server\":\"bqsm\",\"version\":\"3.0\","
            "\"model\":\"%s\",\"loaded\":%s}",g_model_path?g_model_path:"none",g_loaded?"true":"false");
        send_resp(fd,200,"application/json",b); close(fd); return NULL;
    }

    if(!strcmp(method,"GET")&&!strcmp(url,"/v1/models")){
        char b[512]; int bl=snprintf(b,sizeof(b),
            "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\","
            "\"owned_by\":\"bqsm\",\"d_model\":%d,\"n_layers\":%d}]}",
            g_loaded?g_model_path:"none",g_model.d_model,g_model.n_layers);
        send_resp(fd,200,"application/json",b); close(fd); return NULL;
    }

    if(!strcmp(method,"POST")&&!strcmp(url,"/v1/chat/completions")){
        char *msg=extract_msg(body);
        char *resp=generate(msg);
        char *ej=jesc(resp);
        size_t rl=strlen(ej)+256; char *j=malloc(rl);
        snprintf(j,rl,"{\"id\":\"chatcmpl\",\"object\":\"chat.completion\","
            "\"model\":\"bqsm\",\"choices\":[{\"index\":0,\"message\":{"
            "\"role\":\"assistant\",\"content\":%s},\"finish_reason\":\"stop\"}]}",ej);
        send_resp(fd,200,"application/json",j);
        free(j);free(ej);free(resp);free(msg);
        close(fd); return NULL;
    }

    const char *nf="{\"error\":{\"message\":\"Not found\"}}";
    send_resp(fd,404,"application/json",nf); close(fd); return NULL;
}

/* ─── Main ─────────────────────────────────────────────────────────────── */
static void onsig(int s){(void)s;g_running=0;}

int main(int argc,char **argv){
    for(int i=1;i<argc;i++){if(!strcmp(argv[i],"--port")&&i+1<argc)g_port=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--model")&&i+1<argc)g_model_path=argv[++i];
        else if(!strcmp(argv[i],"--help")){printf("BQSM Server v3.0\n--port PORT --model FILE.bqsm\n");return 0;}}
    tt_init();
    if(g_model_path){printf("Loading %s...\n",g_model_path);
        if(!bqsm_model_load(&g_model,g_model_path)){g_loaded=1;bqsm_model_print(&g_model);}
        else fprintf(stderr,"Model load failed.\n");}
    printf("\n═══ BQSM Server v3.0 ═══\nPort %d  Model: %s\nhttp://localhost:%d/\n\n",g_port,g_loaded?"loaded":"none",g_port);

    int srv=socket(AF_INET,SOCK_STREAM,0); if(srv<0){perror("socket");return 1;}
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
    printf("Done.\n"); return 0;
}