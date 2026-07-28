/* Test just the sensor fusion demo */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define OSC_PER_MICRO 16
#define MICRO_PER_MACRO 16
#define MACRO_PER_SUPER 16
#define DT 0.5
#define SETTLE_STEPS 120
#define SETTLE_CHUNKS 3
#define MAX_VQPUS 16
#define MAX_PULSES 50000

/* ... (include necessary types and functions from cluster_validate.c) ... */

int main() {
    printf("Sensor fusion demo test\n");
    return 0;
}
