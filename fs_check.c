/* fast_sin acceptance check: max error vs libm over 1e7 points, plus wrap path */
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#define FS_C1  9.99999999914541360e-01
#define FS_C3  -1.66666665577888368e-01
#define FS_C5  8.33332959535318385e-03
#define FS_C7  -1.98407312829215765e-04
#define FS_C9  2.75199467123275398e-06
#define FS_C11 -2.38101484600250895e-08
#define TWO_PI 6.283185307179586476925286766559
#define PI_    3.141592653589793238462643383279
static inline double fast_sin(double x){
    x -= TWO_PI * rint(x * (1.0/TWO_PI));       /* branchless wrap to [-pi,pi] */
    double s  = x < 0.0 ? -1.0 : 1.0;           /* odd symmetry               */
    double xa = fabs(x);                        /* work on [0, pi]            */
    xa = xa > PI_*0.5 ? PI_ - xa : xa;          /* THE REFLECTION: sin(pi-x)  */
    double x2 = xa*xa;                          /* Horner in x^2              */
    return s * xa * (FS_C1 + x2*(FS_C3 + x2*(FS_C5 + x2*(FS_C7 + x2*(FS_C9 + x2*FS_C11)))));
}
static inline uint64_t sm(uint64_t *s){ uint64_t z=(*s+=0x9e3779b97f4a7c15ULL);
    z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL; z=(z^(z>>27))*0x94d049bb133111ebULL; return z^(z>>31);}
int main(void){
    uint64_t st=1; double maxe=0, maxew=0;
    for(long i=0;i<10000000;i++){                       /* gate spec: 1e7 pts */
        double x=((double)(sm(&st)>>11)/9007199254740992.0-0.5)*TWO_PI;   /* [-pi,pi]  */
        double e=fabs(fast_sin(x)-sin(x)); if(e>maxe)maxe=e;
        double xw=x*3.7;                                 /* wrap path, |x|>pi */
        double ew=fabs(fast_sin(xw)-sin(xw)); if(ew>maxew)maxew=ew;
    }
    printf("max err [-pi,pi]      : %.3e  (budget 1e-7)\n", maxe);
    printf("max err wrapped range : %.3e\n", maxew);
    /* throughput sanity: fast_sin vs libm over 2e8 calls each */
    volatile double acc=0; clock_t t0=clock();
    for(long i=0;i<200000000;i++) acc+=fast_sin(1e-8*i-1.0);
    double tf=(double)(clock()-t0)/CLOCKS_PER_SEC; t0=clock();
    for(long i=0;i<200000000;i++) acc+=sin(1e-8*i-1.0);
    double tl=(double)(clock()-t0)/CLOCKS_PER_SEC;
    printf("scalar speedup vs libm: %.2fx  (%.2fs vs %.2fs; SIMD adds more)\n", tl/tf, tf, tl);
    return maxe < 1e-7 ? 0 : 1;
}
