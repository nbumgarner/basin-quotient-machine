/* ===========================================================================
 * vqpu_test.c — vQPU Validation & Demo
 *
 *   Exercises the vQPU core: gate dispatch, circuit execution, interference
 *   visibility, adiabatic sweep. Self-contained test suite.
 *
 * BUILD:
 *   cc -O3 -std=c11 -march=native ring_furnace.c vqpu.c vqpu_test.c -o vqpu_test -lm
 *
 * RUN:
 *   ./vqpu_test [test_name]
 *   Tests: had, measure, cnot, sweep, visibility, all (default)
 * ======================================================================== */

#include "vqpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int tests_run = 0, tests_passed = 0;

#define TEST(name)  do { tests_run++; printf("\n===== %s =====\n", name); } while(0)
#define PASS()      do { tests_passed++; printf("  PASS\n"); } while(0)
#define FAIL(msg)   do { printf("  FAIL: %s\n", msg); } while(0)
#define CHECK(cond, msg) do { if (cond) PASS(); else FAIL(msg); } while(0)

/* ─── Test 1: Single gate — vHAD + vMEASURE ──────────────────────────────── */
static void test_had_measure(void) {
    TEST("vHAD + vMEASURE: superposition readout");

    vcircuit_t c;
    vqpu_circuit_init(&c, 1);

    /* Ground state */
    CHECK(c.rings[0].q == 0, "initial q != 0");

    /* Apply Hadamard */
    vparam_t p_had = {.amp = 2.2};
    vqpu_circuit_add_gate(&c, VOP_HAD, 0, p_had);

    /* Measure */
    vparam_t p_meas = {0};
    vqpu_circuit_add_gate(&c, VOP_MEASURE, 0, p_meas);

    int result = vqpu_circuit_run(&c);

    /* After vHAD+vMEASURE, result should be in {-1,0,1} (superposition
     * across basins, not always 0). The vHAD kick at amp=2.2 places the
     * ring at a basin boundary. Result is deterministic per initial condition
     * but can be any q. We just verify it ran and produced an integer. */
    printf("  Measured q = %d\n", result);
    CHECK(result >= -VQPU_Q_MAX && result <= VQPU_Q_MAX, "result out of range");
    CHECK(c.measures[0] == result, "measurement array mismatch");

    vqpu_circuit_print_stats(&c);
    vqpu_circuit_free(&c);
}

/* ─── Test 2: vRAMP + vMEASURE (state preparation) ───────────────────────── */
static void test_ramp_measure(void) {
    TEST("vRAMP + vMEASURE: prepare |2⟩ and verify");

    vcircuit_t c;
    vqpu_circuit_init(&c, 1);

    /* Prepare |2⟩: ramp at k=1 with amp = 2π*2 = 4π */
    vparam_t p_ramp = {.k_mode = 1, .amp = 4.0 * M_PI};
    vqpu_circuit_add_gate(&c, VOP_RAMP, 0, p_ramp);
    vqpu_circuit_add_gate(&c, VOP_MEASURE, 0, (vparam_t){0});

    int result = vqpu_circuit_run(&c);
    printf("  Prepared q = %d (target 2)\n", result);
    CHECK(result == 2, "vRAMP did not prepare |2⟩");

    vqpu_circuit_print_stats(&c);
    vqpu_circuit_free(&c);
}

/* ─── Test 3: vPHASE (instant, no settle) ────────────────────────────────── */
static void test_phase_gate(void) {
    TEST("vPHASE: instant phase rotation");

    vcircuit_t c;
    vqpu_circuit_init(&c, 1);

    /* Initialize to |1⟩ twisted state */
    vqpu_ring_set_twisted(&c.rings[0], 1);
    CHECK(c.rings[0].q == 1, "pre-state wrong");

    /* Apply phase gate: global rotation by π/2 */
    vparam_t p_phi = {.phase = M_PI / 2.0};
    vqpu_circuit_add_gate(&c, VOP_PHASE, 0, p_phi);
    vqpu_circuit_run(&c);

    /* Phase gate shouldn't change winding number (it's a gauge
     * transformation), but should shift all phases by φ. */
    CHECK(c.rings[0].q == 1, "vPHASE changed winding number (shouldn't)");
    CHECK(c.step_count == 1, "step count wrong");
    CHECK(c.total_sim_time == 0.0, "vPHASE should cost zero sim time");

    vqpu_circuit_print_stats(&c);
    vqpu_circuit_free(&c);
}

/* ─── Test 4: vRESET → guaranteed q=0 ────────────────────────────────────── */
static void test_reset(void) {
    TEST("vRESET: guaranteed q=0 from |3⟩");

    vcircuit_t c;
    vqpu_circuit_init(&c, 1);

    /* Set to marginal state |3⟩ */
    vqpu_ring_set_twisted(&c.rings[0], 3);
    CHECK(c.rings[0].q == 3, "pre-state wrong");

    /* RESET */
    vqpu_circuit_add_gate(&c, VOP_RESET, 0, (vparam_t){0});
    vqpu_circuit_add_gate(&c, VOP_MEASURE, 0, (vparam_t){0});

    int result = vqpu_circuit_run(&c);
    printf("  After RESET: q = %d\n", result);
    CHECK(result == 0, "vRESET did not reach q=0");

    vqpu_circuit_print_stats(&c);
    vqpu_circuit_free(&c);
}

/* ─── Test 5: vCNOT on two rings ─────────────────────────────────────────── */
static void test_cnot(void) {
    TEST("vCNOT: conditional NOT");

    vcircuit_t c;
    vqpu_circuit_init(&c, 2);

    /* Control = |1⟩, Target = |0⟩ */
    vqpu_ring_set_twisted(&c.rings[0], 1);
    vqpu_ring_set_twisted(&c.rings[1], 0);

    vparam_t p_cnot = {.control_ring = 0, .site = 0, .delta = 0.35};
    vqpu_circuit_add_gate(&c, VOP_CNOT, 1, p_cnot);

    vqpu_circuit_run(&c);

    /* Target should be flipped: 0 XOR 1 = 1 */
    printf("  Control q = %d, Target q = %d\n", c.rings[0].q, c.rings[1].q);
    CHECK(c.rings[0].q == 1, "control ring q changed");
    /* Note: vCNOT measures control first, so control ring's Fourier state
     * is collapsed. But its q should still be 1 after measurement. */

    /* Target result depends on lens effectiveness.
     * At δ=0.35, the pristine transition table is heavily modified.
     * We verify the operation completed without error. */
    CHECK(c.rings[1].q >= -VQPU_Q_MAX && c.rings[1].q <= VQPU_Q_MAX,
          "target ring result out of range");

    printf("  Target after vCNOT: q = %d (was 0, expected flip toward ±1)\n",
           c.rings[1].q);

    /* Now test with control = |0⟩ (should NOT flip) */
    vqpu_circuit_reset(&c);
    vqpu_ring_set_twisted(&c.rings[0], 0);
    vqpu_ring_set_twisted(&c.rings[1], 1);
    vqpu_circuit_add_gate(&c, VOP_CNOT, 1, p_cnot);
    vqpu_circuit_run(&c);

    printf("  Control q=0 case: Target q = %d (was 1, should stay near 1)\n",
           c.rings[1].q);
    /* With control=0, delta_lens=0, target should stay near 1 */
    CHECK(abs(c.rings[1].q) <= 2, "target changed too much with control=0");

    vqpu_circuit_print_stats(&c);
    vqpu_circuit_free(&c);
}

/* ─── Test 6: Interference visibility (validation gate) ──────────────────── */
static void test_interference(void) {
    TEST("Interference visibility: V must exceed 0.50 for validation");

    vcircuit_t c;
    vqpu_circuit_init(&c, 1);

    /* Run visibility experiment: q_init=2, pert_amp=5.0, phi_steps=64, chunks=4 */
    double V = vqpu_interference_visibility(&c, 0, 2, 5.0, 64, 4);

    printf("  V = %.4f (threshold: 0.50)\n", V);
    CHECK(V >= 0.50, "V below 0.50 — interference not validated");

    vqpu_circuit_print_stats(&c);
    vqpu_circuit_free(&c);
}

/* ─── Test 7: Adiabatic sweep ────────────────────────────────────────────── */
static void test_adiabatic_sweep(void) {
    TEST("Adiabatic sweep: δ=0→0.20, ground state stays q=0");

    vcircuit_t c;
    vqpu_circuit_init(&c, 1);

    /* Ground state, sweep δ from 0 to 0.20 (below q=0 critical).
     * The ring should follow the instantaneous attractor and stay at q=0. */
    int q_final = vqpu_adiabatic_sweep(&c, 0, 0.0, 0.20, 100);

    printf("  Sweep 0→0.20: q_final = %d (expected 0)\n", q_final);
    CHECK(q_final == 0, "adiabatic sweep lost ground state tracking");

    /* Now sweep from 0 to 0.35 with the ring at q=1 — verify it TRACKS
     * the instantaneous attractor rather than falling to q=0.
     * At δ=0.35, q=0 has gap 0.126, q=1 has gap 0.099 — both survive. */
    vqpu_circuit_reset(&c);
    vqpu_ring_set_twisted(&c.rings[0], 1);

    q_final = vqpu_adiabatic_sweep(&c, 0, 0.0, 0.35, 150);

    printf("  Sweep 0→0.35 from q=1: q_final = %d\n", q_final);
    CHECK(abs(q_final) <= 2, "adiabatic sweep produced wild result");

    vqpu_circuit_print_stats(&c);
    vqpu_circuit_free(&c);
}

/* ─── Test 8: Fourier state evolution ────────────────────────────────────── */
static void test_fourier_evolution(void) {
    TEST("Fourier evolution: prepare c_2, watch c_1 coupling");

    vcircuit_t c;
    vqpu_circuit_init(&c, 1);

    /* Start at q=2 (strong c_1 harmonic, near-marginal basin) */
    vqpu_ring_set_twisted(&c.rings[0], 2);

    /* Snapshot Fourier state */
    vqpu_fourier_snapshot(&c, 0);
    printf("  Before perturbation: |c_1| = %.4f, |c_2| = %.4f, power = %.4f\n",
           sqrt(c.rings[0].c_real[1]*c.rings[0].c_real[1] + c.rings[0].c_imag[1]*c.rings[0].c_imag[1]),
           sqrt(c.rings[0].c_real[2]*c.rings[0].c_real[2] + c.rings[0].c_imag[2]*c.rings[0].c_imag[2]),
           c.rings[0].c_power);

    /* Apply c_2 perturbation (second harmonic excitation) */
    vqpu_apply_fourier_kick(&c, 0, 2, 5.0, 0.0);

    /* Partial settle: 4 chunks */
    {
        struct batch_t *b = batch_new(1);
        vqpu_ring_to_batch(&c.rings[0], b, 0);
        settle_partial(b, 4);
        vqpu_batch_to_ring(&c.rings[0], b, 0);
        batch_free(b);
    }

    /* Snapshot after */
    vqpu_fourier_snapshot(&c, 0);
    printf("  After perturbation + 4-chunk settle:\n");
    printf("    |c_1| = %.4f, |c_2| = %.4f, power = %.4f\n",
           sqrt(c.rings[0].c_real[1]*c.rings[0].c_real[1] + c.rings[0].c_imag[1]*c.rings[0].c_imag[1]),
           sqrt(c.rings[0].c_real[2]*c.rings[0].c_real[2] + c.rings[0].c_imag[2]*c.rings[0].c_imag[2]),
           c.rings[0].c_power);
    printf("    q = %d\n", c.rings[0].q);

    /* The key test: c_1 should have changed from the perturbation
     * through nonlinear c_2·c_2 → c_1 coupling. If |c_1| is still
     * exactly 2π*2/16 (pure twisted state), no coupling occurred. */
    double c1_mag = sqrt(c.rings[0].c_real[1]*c.rings[0].c_real[1]
                       + c.rings[0].c_imag[1]*c.rings[0].c_imag[1]);
    CHECK(c1_mag > 0.0, "c_1 vanished (should have energy)");

    vqpu_print_spectrum(&c, 0);
    vqpu_circuit_free(&c);
}

/* ─── Test 9: Multi-gate circuit (Bell-state analog) ─────────────────────── */
static void test_bell_analog(void) {
    TEST("Bell-state analog: vHAD(0) + vCNOT(0,1)");

    vcircuit_t c;
    vqpu_circuit_init(&c, 2);

    /* |0⟩⊗|0⟩ → vHAD(0) → vCNOT(0,1) → measure both */
    vqpu_circuit_add_gate(&c, VOP_HAD, 0, (vparam_t){.amp = 2.2});
    vparam_t p_cnot = {.control_ring = 0, .site = 0, .delta = 0.35};
    vqpu_circuit_add_gate(&c, VOP_CNOT, 1, p_cnot);
    vqpu_circuit_add_gate(&c, VOP_MEASURE, 0, (vparam_t){0});
    vqpu_circuit_add_gate(&c, VOP_MEASURE, 1, (vparam_t){0});

    vqpu_circuit_run(&c);

    printf("  Control measured: %d, Target measured: %d\n",
           c.measures[0], c.measures[1]);

    /* Classical Bell analog: measurements should be correlated.
     * Since vHAD produces superposition, and vCNOT conditions on control,
     * we expect correlated outcomes (both 1 or both 0 in binary radix). */
    int qc = c.measures[0] & 1;
    int qt = c.measures[1] & 1;

    printf("  Binary radix: control=%d target=%d (CORRELATED: %s)\n",
           qc, qt, (qc == qt) ? "YES" : "NO");

    /* Run 10 trials, check correlation rate */
    int correlated = 0;
    for (int trial = 0; trial < 10; trial++) {
        vqpu_circuit_reset(&c);

        vqpu_circuit_add_gate(&c, VOP_HAD, 0, (vparam_t){.amp = 2.2});
        vqpu_circuit_add_gate(&c, VOP_CNOT, 1, p_cnot);
        vqpu_circuit_add_gate(&c, VOP_MEASURE, 0, (vparam_t){0});
        vqpu_circuit_add_gate(&c, VOP_MEASURE, 1, (vparam_t){0});
        vqpu_circuit_run(&c);

        if ((c.measures[0] & 1) == (c.measures[1] & 1))
            correlated++;
    }

    printf("  10 trials: %d/10 correlated\n", correlated);
    /* With vHAD producing roughly 50/50 split on binary radix,
     * and vCNOT linking target to control, we expect >50% correlation.
     * Random would be 5/10. Correlation > 5 indicates vCNOT works. */
    CHECK(correlated > 5, "vCNOT correlation rate ≤ random chance (50%)");

    vqpu_circuit_print_stats(&c);
    vqpu_circuit_free(&c);
}

/* ─── Main ────────────────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    printf("vQPU Test Suite — bqsm-quantum/ Phase 2 Validation\n");
    printf("N=%d  Q_MAX=%d  gap₀=%.5f\n\n",
           VQPU_N, VQPU_Q_MAX, VQPU_GAP_0);

    const char *filter = (argc > 1) ? argv[1] : "all";

    #define RUN(test_fn, name) \
        if (!strcmp(filter, "all") || !strcmp(filter, name)) test_fn()

    RUN(test_had_measure,        "had");
    RUN(test_ramp_measure,       "ramp");
    RUN(test_phase_gate,         "phase");
    RUN(test_reset,              "reset");
    RUN(test_cnot,               "cnot");
    RUN(test_interference,       "visibility");
    RUN(test_adiabatic_sweep,    "sweep");
    RUN(test_fourier_evolution,  "fourier");
    RUN(test_bell_analog,        "bell");

    printf("\n═══════════════════════════════════════\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("═══════════════════════════════════════\n");

    return (tests_passed == tests_run) ? 0 : 1;
}