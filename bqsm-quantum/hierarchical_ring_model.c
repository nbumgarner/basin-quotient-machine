/* ===========================================================================
 * hierarchical_ring_model.c — 16³ = 4096 Oscillator Hierarchical Architecture
 *
 *   Models the hierarchical ring structure:
 *   Level 1: 16 oscillators (micro ring)
 *   Level 2: 16 micro rings (macro ring) = 256 oscillators
 *   Level 3: 16 macro rings (super macro ring) = 4096 oscillators
 *
 *   Explores:
 *   - Quantization configurations at each level
 *   - Programmability via lens profiles
 *   - Structural computability (what can be computed)
 *   - Hardware implications (processor encircling components)
 * ======================================================================== */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Hierarchical structure constants */
#define OSC_PER_MICRO_RING 16
#define MICRO_RINGS_PER_MACRO 16
#define MACRO_RINGS_PER_SUPER 16
#define TOTAL_OSCILLATORS (OSC_PER_MICRO_RING * MICRO_RINGS_PER_MACRO * MACRO_RINGS_PER_SUPER)

/* Level definitions */
typedef enum { LEVEL_MICRO = 1, LEVEL_MACRO = 2, LEVEL_SUPER = 3 } level_t;

/* Ring structure at each level */
typedef struct {
    double th[OSC_PER_MICRO_RING];  /* phases */
    double om[OSC_PER_MICRO_RING];  /* lens/omega */
    double k1[OSC_PER_MICRO_RING];
    double k2[OSC_PER_MICRO_RING];
    double k3[OSC_PER_MICRO_RING];
    double k4[OSC_PER_MICRO_RING];
    double tmp[OSC_PER_MICRO_RING];
} micro_ring_t;

typedef struct {
    micro_ring_t rings[MICRO_RINGS_PER_MACRO];
    /* Inter-ring coupling parameters */
    double coupling_strength;
    double coupling_phase;
} macro_ring_t;

typedef struct {
    macro_ring_t rings[MACRO_RINGS_PER_SUPER];
    /* Inter-macro-ring coupling parameters */
    double coupling_strength;
    double coupling_phase;
} super_macro_ring_t;

/* Core physics functions (same as before) */
static void deriv_micro(micro_ring_t *r){
    for(int i=0;i<OSC_PER_MICRO_RING;i++){
        int ip=(i+1)%(OSC_PER_MICRO_RING);
        int im=(i-1+OSC_PER_MICRO_RING)%(OSC_PER_MICRO_RING);
        r->tmp[i]=r->om[i]+sin(r->th[ip]-r->th[i])+sin(r->th[im]-r->th[i]);
    }
}
static void rk4_micro(micro_ring_t *r){
    deriv_micro(r);
    for(int i=0;i<OSC_PER_MICRO_RING;i++)r->k1[i]=r->tmp[i];
    for(int i=0;i<OSC_PER_MICRO_RING;i++)r->tmp[i]=r->th[i]+0.5*0.5*r->k1[i];
    deriv_micro(r);
    for(int i=0;i<OSC_PER_MICRO_RING;i++)r->k2[i]=r->tmp[i];
    for(int i=0;i<OSC_PER_MICRO_RING;i++)r->tmp[i]=r->th[i]+0.5*0.5*r->k2[i];
    deriv_micro(r);
    for(int i=0;i<OSC_PER_MICRO_RING;i++)r->k3[i]=r->tmp[i];
    for(int i=0;i<OSC_PER_MICRO_RING;i++)r->tmp[i]=r->th[i]+0.5*r->k3[i];
    deriv_micro(r);
    for(int i=0;i<OSC_PER_MICRO_RING;i++)r->k4[i]=r->tmp[i];
    for(int i=0;i<OSC_PER_MICRO_RING;i++)r->th[i]+=(0.5/6.0)*(r->k1[i]+2.0*r->k2[i]+2.0*r->k3[i]+r->k4[i]);
}
static void settle_micro(micro_ring_t *r,int count){
    for(int c=0;c<count;c++){
        for(int s=0;s<120;s++){  /* 60.0/0.5 = 120 steps per chunk */
            rk4_micro(r);
        }
    }
}

/* Compute harmonic magnitude for a micro ring */
static double micro_harmonic_mag(micro_ring_t *r,int k){
    double re=0,im=0;
    for(int j=0;j<OSC_PER_MICRO_RING;j++){
        double a=-2.0*M_PI*k*j/OSC_PER_MICRO_RING;
        re+=r->th[j]*cos(a);
        im+=r->th[j]*sin(a);
    }
    return sqrt(re*re+im*im)/OSC_PER_MICRO_RING;
}

/* Compute macro-ring harmonic (average of micro-ring harmonics) */
static double macro_harmonic_mag(macro_ring_t *mr,int k){
    double re=0,im=0;
    for(int i=0;i<MICRO_RINGS_PER_MACRO;i++){
        double mag = micro_harmonic_mag(&mr->rings[i], k);
        /* For simplicity, just sum magnitudes */
        re += mag;
    }
    return re/MICRO_RINGS_PER_MACRO;
}

/* Compute super macro-ring harmonic */
static double super_harmonic_mag(super_macro_ring_t *smr,int k){
    double re=0,im=0;
    for(int i=0;i<MACRO_RINGS_PER_SUPER;i++){
        double mag = macro_harmonic_mag(&smr->rings[i], k);
        re += mag;
    }
    return re/MACRO_RINGS_PER_SUPER;
}

/* Apply lens profile to micro ring */
static void apply_micro_lens(micro_ring_t *r,const double *profile){
    memcpy(r->om,profile,OSC_PER_MICRO_RING*sizeof(double));
}

/* Apply lens profile to macro ring (per micro ring) */
static void apply_macro_lens(macro_ring_t *mr,const double *profiles){
    for(int i=0;i<MICRO_RINGS_PER_MACRO;i++){
        apply_micro_lens(&mr->rings[i], &profiles[i*OSC_PER_MICRO_RING]);
    }
}

/* Quantization configurations */
typedef struct {
    const char *name;
    int bits;
    double resolution;
    double quantization_error;
    int num_levels;
} quant_config_t;

static quant_config_t quant_configs[] = {
    {"float64", 64, 1.0/18446744073709551615.0, 0.0, 18446744073709551616LL},
    {"float32", 32, 1.0/4294967295.0, 0.0, 4294967296LL},
    {"int16", 16, 1.0/32767.0, 0.0000305, 65536},
    {"int8", 8, 1.0/127.0, 0.00787, 256},
    {"int4", 4, 1.0/7.0, 0.0714, 16},
    {"int2", 2, 1.0/1.0, 0.5, 4},
    {"binary", 1, 1.0/0.0, 1.0, 2},
    {NULL, 0, 0.0, 0.0, 0}
};

/* Programmability configurations */
typedef struct {
    const char *name;
    int lens_profiles;
    double lens_resolution;
    int programmable_sites;
    const char *description;
} prog_config_t;

static prog_config_t prog_configs[] = {
    {"uniform", 1, 0.0, OSC_PER_MICRO_RING, "Single uniform lens for all sites"},
    {"site_specific", OSC_PER_MICRO_RING, 0.01, OSC_PER_MICRO_RING, "Independent lens per oscillator"},
    {"harmonic_patterns", 16, 0.01, OSC_PER_MICRO_RING, "16 harmonic lens patterns"},
    {"gaussian_bumps", 8, 0.05, OSC_PER_MICRO_RING, "8 Gaussian bump patterns"},
    {"hybrid", 64, 0.01, OSC_PER_MICRO_RING, "64 mixed patterns"},
    {NULL, 0, 0.0, 0, NULL}
};

/* Structural computability analysis */
typedef struct {
    const char *operation;
    int level_required;
    int micro_rings_needed;
    int macro_rings_needed;
    double complexity;
    const char *description;
} comp_operation_t;

static comp_operation_t comp_operations[] = {
    {"single_oscillator", 1, 1, 0, 1.0, "Individual oscillator control"},
    {"micro_ring_filter", 1, 1, 0, 1.0, "Micro ring as filter"},
    {"micro_ring_product", 1, 1, 0, 1.5, "Micro ring product computation"},
    {"macro_ring_filter", 2, 16, 1, 2.0, "Macro ring as filter"},
    {"macro_ring_product", 2, 16, 1, 2.5, "Macro ring product computation"},
    {"super_macro_filter", 3, 256, 16, 3.0, "Super macro ring as filter"},
    {"super_macro_product", 3, 256, 16, 3.5, "Super macro ring product computation"},
    {"cross_level_coupling", 3, 256, 16, 4.0, "Cross-level coupling operations"},
    {"hierarchical_attention", 3, 256, 16, 4.5, "Hierarchical attention mechanism"},
    {"global_computation", 3, 256, 16, 5.0, "Global computation on all 4096 oscillators"},
    {NULL, 0, 0, 0, 0.0, NULL}
};

int main(){
    printf("═══ HIERARCHICAL RING MODEL — 16³ = 4096 OSCILLATORS ═══\n\n");
    
    printf("ARCHITECTURE:\n");
    printf("  Level 1 (Micro):  16 oscillators per ring\n");
    printf("  Level 2 (Macro):  16 micro rings = 256 oscillators\n");
    printf("  Level 3 (Super):  16 macro rings = 4096 oscillators\n\n");
    
    /* Section 1: Quantization Configurations */
    printf("═══ SECTION 1: QUANTIZATION CONFIGURATIONS ═══\n\n");
    for(int i=0; quant_configs[i].name != NULL; i++){
        printf("  %-10s: %2d bits, resolution=%.2e, levels=%lld\n",
               quant_configs[i].name,
               quant_configs[i].bits,
               quant_configs[i].resolution,
               quant_configs[i].num_levels);
    }
    printf("\n");
    
    /* Section 2: Programmability Configurations */
    printf("═══ SECTION 2: PROGRAMMABILITY CONFIGURATIONS ═══\n\n");
    for(int i=0; prog_configs[i].name != NULL; i++){
        printf("  %-15s: %d profiles, resolution=%.2f, sites=%d\n",
               prog_configs[i].name,
               prog_configs[i].lens_profiles,
               prog_configs[i].lens_resolution,
               prog_configs[i].programmable_sites);
        printf("    %s\n", prog_configs[i].description);
    }
    printf("\n");
    
    /* Section 3: Structural Computability */
    printf("═══ SECTION 3: STRUCTURAL COMPUTABILITY ═══\n\n");
    for(int i=0; comp_operations[i].operation != NULL; i++){
        printf("  %-20s: level=%d, micro_rings=%d, macro_rings=%d, complexity=%.1f\n",
               comp_operations[i].operation,
               comp_operations[i].level_required,
               comp_operations[i].micro_rings_needed,
               comp_operations[i].macro_rings_needed,
               comp_operations[i].complexity);
        printf("    %s\n", comp_operations[i].description);
    }
    printf("\n");
    
    /* Section 4: Hardware Implications */
    printf("═══ SECTION 4: HARDWARE IMPLICATIONS ═══\n\n");
    printf("  Total oscillators: %d\n", TOTAL_OSCILLATORS);
    printf("  Micro rings: 256\n");
    printf("  Macro rings: 16\n");
    printf("  Super macro rings: 1\n");
    printf("\n");
    printf("  Hardware configuration:\n");
    printf("    - Processor encircles all components\n");
    printf("    - Lens control lines: %d (one per oscillator)\n", TOTAL_OSCILLATORS);
    printf("    - Phase readout lines: %d (one per oscillator)\n", TOTAL_OSCILLATORS);
    printf("    - Coupling lines: ~%d (nearest neighbor)\n", TOTAL_OSCILLATORS);
    printf("\n");
    
    /* Section 5: Configuration Space Size */
    printf("═══ SECTION 5: CONFIGURATION SPACE SIZE ═══\n\n");
    printf("  Quantization configurations: %d\n", (int)(sizeof(quant_configs)/sizeof(quant_configs[0])-1));
    printf("  Programmability configurations: %d\n", (int)(sizeof(prog_configs)/sizeof(prog_configs[0])-1));
    printf("  Computability configurations: %d\n", (int)(sizeof(comp_operations)/sizeof(comp_operations[0])-1));
    printf("\n");
    printf("  Total lens configurations per micro ring: %d\n", 1 << OSC_PER_MICRO_RING);
    printf("  Total lens configurations per macro ring: %d\n", 1 << (OSC_PER_MICRO_RING * MICRO_RINGS_PER_MACRO));
    printf("  Total lens configurations per super macro: %d\n", 1 << (OSC_PER_MICRO_RING * MICRO_RINGS_PER_MACRO * MACRO_RINGS_PER_SUPER));
    printf("\n");
    
    /* Section 6: Computational Capacity */
    printf("═══ SECTION 6: COMPUTATIONAL CAPACITY ═══\n\n");
    printf("  Micro ring capacity: 16 oscillators × %d lens states = %d states\n",
           1 << OSC_PER_MICRO_RING, 1 << OSC_PER_MICRO_RING);
    printf("  Macro ring capacity: 256 oscillators × %d lens states = %d states\n",
           1 << (OSC_PER_MICRO_RING * MICRO_RINGS_PER_MACRO), 1 << (OSC_PER_MICRO_RING * MICRO_RINGS_PER_MACRO));
    printf("  Super macro capacity: 4096 oscillators × %d lens states = %d states\n",
           1 << (OSC_PER_MICRO_RING * MICRO_RINGS_PER_MACRO * MACRO_RINGS_PER_SUPER),
           1 << (OSC_PER_MICRO_RING * MICRO_RINGS_PER_MACRO * MACRO_RINGS_PER_SUPER));
    printf("\n");
    
    /* Section 7: Practical Configurations */
    printf("═══ SECTION 7: PRACTICAL CONFIGURATIONS ═══\n\n");
    
    /* Configuration A: High precision, full programmability */
    printf("  Configuration A (High Precision):\n");
    printf("    - Quantization: float32\n");
    printf("    - Lens: site_specific (independent per oscillator)\n");
    printf("    - Resolution: 1e-8\n");
    printf("    - Lens states: 2^4096 (theoretically)\n");
    printf("    - Practical: 2^256 (macro ring level)\n");
    printf("\n");
    
    /* Configuration B: Balanced */
    printf("  Configuration B (Balanced):\n");
    printf("    - Quantization: int16\n");
    printf("    - Lens: harmonic_patterns (16 patterns)\n");
    printf("    - Resolution: 0.001\n");
    printf("    - Lens states: 16^256\n");
    printf("    - Practical: 2^1024\n");
    printf("\n");
    
    /* Configuration C: Low power */
    printf("  Configuration C (Low Power):\n");
    printf("    - Quantization: int4\n");
    printf("    - Lens: gaussian_bumps (8 patterns)\n");
    printf("    - Resolution: 0.1\n");
    printf("    - Lens states: 8^256\n");
    printf("    - Practical: 2^768\n");
    printf("\n");
    
    /* Configuration D: Ultra low power */
    printf("  Configuration D (Ultra Low Power):\n");
    printf("    - Quantization: int2\n");
    printf("    - Lens: uniform\n");
    printf("    - Resolution: 0.5\n");
    printf("    - Lens states: 4^256\n");
    printf("    - Practical: 2^512\n");
    printf("\n");
    
    /* Section 8: Summary */
    printf("═══ SUMMARY ═══\n\n");
    printf("  Total oscillators: %d\n", TOTAL_OSCILLATORS);
    printf("  Quantization options: %d (binary to float64)\n", (int)(sizeof(quant_configs)/sizeof(quant_configs[0])-1));
    printf("  Programmability options: %d (uniform to hybrid)\n", (int)(sizeof(prog_configs)/sizeof(prog_configs[0])-1));
    printf("  Computability options: %d (single oscillator to global)\n", (int)(sizeof(comp_operations)/sizeof(comp_operations[0])-1));
    printf("\n");
    printf("  Key insight: The hierarchical structure enables:\n");
    printf("    1. Multi-scale computation (micro, macro, super macro)\n");
    printf("    2. Programmable filters at each level\n");
    printf("    3. Cross-level coupling for complex operations\n");
    printf("    4. Hardware-efficient encircling processor design\n");
    printf("\n");
    printf("  Next: Simulate hierarchical dynamics and measure computational properties.\n");
    
    return 0;
}