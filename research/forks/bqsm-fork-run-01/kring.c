// kring.c — independent Kuramoto reference sim for BQSM fork tests.
// dtheta_i/dt = w_i + sum_j K_ij sin(theta_j - theta_i), RK4.
// Modes:
//   forkA <topo> <nprog> <seed>  : Boolean programmability (3-in/1-out), coverage of 256 truth tables
//   forkB <nIC> <seed>           : WS integrability breaking vs multistability along global->ring interpolation
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>

#define N 16
static double K[N][N];
static double W[N];

static uint64_t rs;
static double urand(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return (rs>>11)*(1.0/9007199254740992.0); }
static double nrand(void){ double u=urand()+1e-300,v=urand(); return sqrt(-2*log(u))*cos(2*M_PI*v); }

static void deriv(const double *th, double *d){
  for(int i=0;i<N;i++){ double s=W[i]; for(int j=0;j<N;j++) if(K[i][j]!=0.0) s+=K[i][j]*sin(th[j]-th[i]); d[i]=s; }
}
static void rk4(double *th, double dt){
  double k1[N],k2[N],k3[N],k4[N],t[N];
  deriv(th,k1); for(int i=0;i<N;i++) t[i]=th[i]+0.5*dt*k1[i];
  deriv(t,k2);  for(int i=0;i<N;i++) t[i]=th[i]+0.5*dt*k2[i];
  deriv(t,k3);  for(int i=0;i<N;i++) t[i]=th[i]+dt*k3[i];
  deriv(t,k4);  for(int i=0;i<N;i++) th[i]+=dt/6*(k1[i]+2*k2[i]+2*k3[i]+k4[i]);
}

/* ---------------- Fork A ---------------- */
// topo: 0 ring, 1 all-to-all, 2 ring + 4 random chords
static const int IN[3]={0,5,10}, OUT=13, REF=15;
static void build_program(int topo, double K0){
  memset(K,0,sizeof K);
  #define SETE(a,b,v) do{K[a][b]=v;K[b][a]=v;}while(0)
  if(topo==1){ for(int i=0;i<N;i++) for(int j=i+1;j<N;j++) SETE(i,j,(urand()<0.5?-1:1)*K0*4.0/N); }
  else {
    for(int i=0;i<N;i++) SETE(i,(i+1)%N,(urand()<0.5?-1:1)*K0);
    if(topo==2) for(int c=0;c<4;c++){ int a=(int)(urand()*N),b=(int)(urand()*N); if(a!=b) SETE(a,b,(urand()<0.5?-1:1)*K0); }
  }
  for(int i=0;i<N;i++) W[i]=(urand()-0.5)*0.4;
}
static double base_ic[N];
static int INMODE=0; static double NOISE=0.15, DELTA=0.3;
static int eval_bit(int in, double noise){
  double th[N]; for(int i=0;i<N;i++) th[i]=base_ic[i]+noise*nrand();
  double Wsave[N]; memcpy(Wsave,W,sizeof W);
  for(int b=0;b<3;b++){ int bit=in>>b&1;
    if(INMODE==0){ if(bit) th[IN[b]]+=M_PI; }
    else W[IN[b]]+= bit? DELTA : -DELTA; }
  const double dt=0.1; const int steps=1000, avg_from=700; double acc=0;
  for(int s=0;s<steps;s++){ rk4(th,dt); if(s>=avg_from) acc+=cos(th[OUT]-th[REF]); }
  memcpy(W,Wsave,sizeof W);
  return acc>0;
}
static int truth_table(double noise){ int tt=0; for(int in=0;in<8;in++) tt|=eval_bit(in,noise)<<in; return tt; }

static int forkA(int topo,int nprog,uint64_t seed){
  rs=seed?seed:88172645463325252ULL;
  for(int i=0;i<N;i++) base_ic[i]=(urand()-0.5)*0.6;   // fixed symmetry-breaking IC, same for all programs/inputs
  static int hit[256], robust[256]; memset(hit,0,sizeof hit); memset(robust,0,sizeof robust);
  int nrobust_prog=0;
  for(int p=0;p<nprog;p++){
    build_program(topo,1.0);
    int tt=truth_table(0.0); hit[tt]++;
    uint64_t save=rs; int ok=1;
    for(int r=0;r<3&&ok;r++) if(truth_table(NOISE)!=tt) ok=0;
    rs=save^0x9E3779B97F4A7C15ULL;
    if(ok){ robust[tt]++; nrobust_prog++; }
    if((p+1)%(nprog/4)==0){ int c=0,cr=0; for(int t=0;t<256;t++){c+=hit[t]>0;cr+=robust[t]>0;}
      printf("topo=%d progs=%d distinct=%d robust_distinct=%d robust_progs=%d\n",topo,p+1,c,cr,nrobust_prog); fflush(stdout);}
  }
  // dump histogram
  printf("HIST topo=%d:",topo); for(int t=0;t<256;t++) if(hit[t]) printf(" %d:%d/%d",t,hit[t],robust[t]); printf("\n");
  return 0;
}

/* ---------------- Fork B ---------------- */
// Identical oscillators, attractive coupling, K = (1-e)*global + e*ring. Measure:
//  (1) WS cross-ratio drift over t in [0,1] (global coupling conserves cross-ratios exactly)
//  (2) number of distinct attractors from random ICs (gauge-invariant signature: wrapped neighbor diffs -> winding q)
static double complex cr(const double *th,int a,int b,int c,int d){
  double complex za=cexp(I*th[a]),zb=cexp(I*th[b]),zc=cexp(I*th[c]),zd=cexp(I*th[d]);
  return (za-zc)*(zb-zd)/((za-zd)*(zb-zc));
}
static double wrap(double x){ while(x>M_PI)x-=2*M_PI; while(x<-M_PI)x+=2*M_PI; return x; }
static int forkB(int nIC,uint64_t seed){
  rs=seed?seed:1234567ULL;
  const double es[]={0.0,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
  const int Q[6][4]={{0,4,8,12},{1,5,9,13},{2,6,10,14},{3,7,11,15},{0,3,7,10},{2,5,11,14}};
  for(size_t ei=0;ei<sizeof es/sizeof*es;ei++){
    double e=es[ei]; memset(K,0,sizeof K);
    for(int i=0;i<N;i++){ W[i]=0; for(int j=0;j<N;j++) if(i!=j) K[i][j]=(1-e)*1.0/N; }
    for(int i=0;i<N;i++){ K[i][(i+1)%N]+=e*0.5; K[(i+1)%N][i]+=e*0.5; }
    int qcount[N+1]; memset(qcount,0,sizeof qcount); int unconv=0; double drift=0; int nd=0;
    for(int r=0;r<nIC;r++){
      double th[N]; for(int i=0;i<N;i++) th[i]=urand()*2*M_PI;
      double c0[6]; for(int q=0;q<6;q++) c0[q]=creal(cr(th,Q[q][0],Q[q][1],Q[q][2],Q[q][3]));
      const double dt=0.01;
      for(int s=0;s<100;s++) rk4(th,dt);           // t=1
      for(int q=0;q<6;q++){ double c1=creal(cr(th,Q[q][0],Q[q][1],Q[q][2],Q[q][3]));
        if(fabs(c0[q])>1e-3&&fabs(c0[q])<1e3){ drift+=fabs(c1-c0[q])/fabs(c0[q]); nd++; } }
      for(int s=0;s<6000;s++) rk4(th,0.05);        // t=301 settle
      double d[N]; deriv(th,d); double mx=0; for(int i=0;i<N;i++){ double a=fabs(d[i]-d[0]); if(a>mx)mx=a; }
      if(mx>1e-4) unconv++;
      double sum=0; for(int i=0;i<N;i++) sum+=wrap(th[(i+1)%N]-th[i]);
      int qw=(int)lround(sum/(2*M_PI)); int qi=qw<0?-qw:qw; if(qi>N) qi=N; qcount[qi]++;
    }
    printf("e=%.2f  ws_drift=%.3e  states:",e,drift/nd);
    int nstates=0; for(int q=0;q<=N;q++) if(qcount[q]){ printf(" |q|=%d:%d",q,qcount[q]); nstates+= q?2:1; }
    printf("  (~%d distinct incl. sign)  unconverged=%d\n",nstates,unconv); fflush(stdout);
  }
  return 0;
}

int main(int argc,char**argv){
  if(getenv("INMODE"))INMODE=atoi(getenv("INMODE")); if(getenv("NOISE"))NOISE=atof(getenv("NOISE")); if(getenv("DELTA"))DELTA=atof(getenv("DELTA"));
  if(argc>=2&&!strcmp(argv[1],"forkA")) return forkA(atoi(argv[2]),atoi(argv[3]),argc>4?strtoull(argv[4],0,10):0);
  if(argc>=2&&!strcmp(argv[1],"forkB")) return forkB(atoi(argv[2]),argc>3?strtoull(argv[3],0,10):0);
  fprintf(stderr,"usage\n"); return 1;
}
