/* ===========================================================================
 * bqsm_vm.h — BQSM Virtual Processor Core: public interface
 * ===========================================================================
 *
 * emerging.systems — BQSM instrument suite, virtual processor
 *
 * This header declares the public API for the Basin-Quotient N-ary Computer
 * virtual processor. The VM wraps the ring_furnace.c physics engine in a
 * register-machine architecture with an instruction set built from measured
 * topological operations on coupled-oscillator rings.
 *
 * DESIGN AUTHORITY: BQSM_ARCHITECTURE.txt — the blueprint this implements.
 * PHYSICS CORE:     ring_furnace.c — functions imported verbatim with
 *                   attribution (deriv, rk4_step, settle, winding).
 * ======================================================================= */

#ifndef BQSM_VM_H
#define BQSM_VM_H

#include <stdint.h>
#include <stddef.h>

/* --- Fabric constants (identical to ring_furnace.c) -------------------- */
#define BQSM_N        16               /* sites per ring (power of two)     */
#define BQSM_NMASK    (BQSM_N - 1)     /* ring-buffer index mask            */
#define BQSM_KCPL     1.0              /* coupling strength K               */
#define BQSM_Q_MAX    3                /* twisted states |q| ≤ 3 at N=16    */
#define BQSM_LMAX     256              /* max lanes (rings) per batch       */
#define BQSM_T_CHUNK  60.0             /* settling chunk length             */
#define BQSM_LOCK_TOL 1e-8             /* velocity-spread lock criterion    */
#define BQSM_MAXCHUNK 12               /* settling patience (chunks)        */
#define BQSM_KICK_W   3                /* kick window width                 */

/* --- Instruction set (BQSM_ARCHITECTURE.txt §2) ------------------------ */
typedef enum {
    BQSM_READ,          /* read winding number q → topological, free        */
    BQSM_WRITE_INC,     /* 2π phase ramp → exactly +1 quantum, else nil     */
    BQSM_ERASE,         /* scalar bump (3-site) → one-way funnel toward q=0 */
    BQSM_COMPARE,       /* conditional branch on q value                    */
    BQSM_RESET,         /* strong ERASE → guaranteed q=0                    */
    BQSM_KICK_AT,       /* general kick: site + amplitude + width           */
    BQSM_RAMP_AT,       /* general phase ramp at site with width            */
    BQSM_OPCODE_COUNT
} bqsm_opcode_t;

/* --- Encoded instruction (single-ring operation) ----------------------- */
typedef struct {
    bqsm_opcode_t opcode;               /* what to do                        */
    int           site;                 /* target oscillator site [0..N-1]   */
    double        amplitude;            /* kick amplitude (ERASE/KICK_AT)    */
    int           width;                /* window width (KICK_AT/RAMP_AT)    */
    int           target_ring;          /* which register this operates on   */
    int           branch_target;        /* for COMPARE: jump if q≠0          */
    int           branch_q;             /* for COMPARE: threshold q value    */
} bqsm_instruction_t;

/* --- Radix control (BQSM_ARCHITECTURE.txt §5) ------------------------- */
typedef enum {
    BQSM_RADIX_BINARY  = 2,             /* q∈{0,1}     1 bit/ring           */
    BQSM_RADIX_TERNARY = 3,             /* q∈{0,1,2}   ~1.6 bits/ring       */
    BQSM_RADIX_QUAD    = 4,             /* q∈{0,1,2,3} 2 bits/ring          */
    BQSM_RADIX_MAX     = 7              /* q∈{0..6}    2.8 bits/ring (N=16) */
} bqsm_radix_t;

/* --- Transition table entry (precomputed instruction outcomes) --------- */
typedef struct {
    int result_q;                       /* winding number after settle       */
    int locked;                         /* did the ring lock?                */
    int alive;                          /* did start state survive?          */
} bqsm_transition_t;

/* Opaque VM handle ------------------------------------------------------ */
typedef struct bqsm_vm_s bqsm_vm_t;

/* =========================================================================
 * PUBLIC API
 * ========================================================================= */

/* Create a virtual processor with `num_rings` registers and the given radix.
 * Radix caps the admissible winding numbers to [0, radix-1]; values outside
 * are treated as overflow/trap states.  Returns NULL on failure.           */
bqsm_vm_t *bqsm_vm_create(int num_rings, bqsm_radix_t radix);

/* Destroy the VM and free all resources.                                   */
void bqsm_vm_destroy(bqsm_vm_t *vm);

/* --- Register operations ----------------------------------------------- */

/* Initialize ring r to the ideal twisted state q (q=0 is ground).          */
void bqsm_ring_set_twisted(bqsm_vm_t *vm, int r, int q);

/* Read the current winding number of ring r. Returns DEAD (-99) if the
 * ring is not in a locked state.                                            */
int bqsm_ring_read(bqsm_vm_t *vm, int r);

/* Check if ring r is currently in a locked (settled) state.                */
int bqsm_ring_is_locked(bqsm_vm_t *vm, int r);

/* --- Instruction execution --------------------------------------------- */

/* Execute a single instruction on the VM. Returns the new winding number
 * of the target ring, or -99 if the ring failed to lock.                   */
int bqsm_execute_instruction(bqsm_vm_t *vm, const bqsm_instruction_t *inst);

/* --- Program execution ------------------------------------------------- */

/* Load a program: an array of `count` instructions into the VM's program
 * memory. Overwrites any previous program.                                 */
void bqsm_program_load(bqsm_vm_t *vm, const bqsm_instruction_t *program, int count);

/* Run the loaded program from start to finish. Returns the number of
 * instructions successfully executed, or -1 if a trap occurred.
 * Stops at COMPARE instructions that branch (non-zero q).                   */
int bqsm_program_run(bqsm_vm_t *vm);

/* Step the loaded program by one instruction, starting at PC. Returns the
 * next PC, or -1 if program finished/trapped. Fills *result with the
 * target ring's new winding number.                                         */
int bqsm_program_step(bqsm_vm_t *vm, int *result);

/* --- Transition table (precomputed) ------------------------------------ */

/* Precompute the full transition table T[q][opcode] for the current lens
 * configuration. Call after setting lens parameters (bqsm_vm_set_lens).
 * Table is cached internally; subsequent instruction execution can use the
 * table for O(1) lookup instead of O(settle) integration.                  */
void bqsm_transition_table_compute(bqsm_vm_t *vm);

/* Look up a transition: given start state q and opcode, what is the
 * resulting winding number? Returns -98 if not computed, -99 if dead.      */
int bqsm_transition_lookup(bqsm_vm_t *vm, int q, bqsm_opcode_t opcode,
                           int site, double amplitude);

/* --- Lens control (instruction reprogramming) -------------------------- */

/* Set the lens detuning at a specific site. delta=0 is pristine.
 * This changes the omega profile and thus DEFORMS the transition table.
 * Requires re-computation of the transition table.                          */
void bqsm_vm_set_lens(bqsm_vm_t *vm, int site, double delta);

/* --- Diagnostics ------------------------------------------------------- */

/* Return the number of rings currently alive (locked with valid q).        */
int bqsm_vm_alive_count(bqsm_vm_t *vm);

/* Print a summary of all ring states.                                       */
void bqsm_vm_dump(bqsm_vm_t *vm);

/* --- Program counter access -------------------------------------------- */
int  bqsm_vm_get_pc(bqsm_vm_t *vm);
void bqsm_vm_set_pc(bqsm_vm_t *vm, int pc);
int  bqsm_vm_get_program_length(bqsm_vm_t *vm);

#endif /* BQSM_VM_H */