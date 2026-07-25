/* chunk_sweep.c — find the continuous regime where mode coupling is measurable */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

extern void settle_partial(void *b, int max_chunks);
extern void fft_snapshot(const void *b, int l, double *c_real, double *c_imag);
extern void apply_fourier_kick(void *b, int l, int k_mode, double amp, double phase);
extern int  winding(const void *b, int l);
extern void set_twisted(void *b, int l, int q);
extern void *batch_new(int L);
extern void  batch_free(void *b);

#define N      16
#define N_HALF (N/2)

int main(void) {
    printf("%6s %8s %8s %8s %8s %6s\n",
           "chunks", "|c1|", "|c2|", "|c3|", "|c4|", "q");
    
    for (int chunks = 1; chunks <= 12; chunks++) {
        void *b = batch_new(1);
        set_twisted(b, 0, 0);  /* ground state */
        
        /* Kick: x=4.0 into c1, y=3.0 into c2 */
        apply_fourier_kick(b, 0, 1, 4.0, 0.0);
        apply_fourier_kick(b, 0, 2, 3.0, 0.0);
        
        settle_partial(b, chunks);
        
        double cr[N], ci[N];
        fft_snapshot(b, 0, cr, ci);
        int q = winding(b, 0);
        
        double c1 = sqrt(cr[1]*cr[1] + ci[1]*ci[1]);
        double c2 = sqrt(cr[2]*cr[2] + ci[2]*ci[2]);
        double c3 = sqrt(cr[3]*cr[3] + ci[3]*ci[3]);
        double c4 = sqrt(cr[4]*cr[4] + ci[4]*ci[4]);
        
        printf("%6d %8.4f %8.4f %8.4f %8.4f %6d\n",
               chunks, c1, c2, c3, c4, q);
        
        batch_free(b);
    }
    return 0;
}