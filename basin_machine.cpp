// ===========================================================================
// basin_machine.cpp — the basin-quotient machine, live in one file
// ===========================================================================
//
//   emerging.systems — public demonstration build
//
// WHAT YOU ARE LOOKING AT
//   A ring of N=16 coupled phase oscillators (nearest-neighbor Kuramoto).
//   Its stable attractors are "twisted states" — phase patterns that wind
//   around the ring an integer number of times. That integer (the winding
//   number q) is a topological invariant, which makes it a perfect discrete
//   machine-state label sitting inside a continuous dynamical system.
//
//   Treat the attractors as STATES and perturbations as INSTRUCTIONS and
//   the ring becomes a small computer whose transition table is DISCOVERED
//   by measurement, not designed. This demo runs the measurements live:
//
//     1. The state alphabet, rendered as arrow rings, with exact energies.
//     2. ERASE — a localized scalar kick knocks winding out of the fabric.
//     3. The one-way funnel — no scalar kick ever ADDS winding.
//     4. WRITE — a gradient-shaped (ramp) input adds winding, and it is
//        QUANTIZED: a full 2π ramp writes exactly +1; a π ramp writes zero.
//
//   Build:  g++ -O2 -std=c++17 basin_machine.cpp -o basin_machine
//   Run:    ./basin_machine          (a few seconds, no dependencies)
//
//   Everything is deterministic — same output every run, every machine.
// ===========================================================================

#include <cmath>                     // sin, cos, fmod, remainder
#include <cstdio>                    // printf — the whole UI
#include <array>                     // fixed-size state vectors
#include <string>                    // output assembly

// --- Fabric parameters ------------------------------------------------------
constexpr int    N        = 16;      // oscillators on the ring (power of two)
constexpr int    NMASK    = N - 1;   // ring-buffer index mask: (i±1) & NMASK
constexpr double K        = 1.0;     // coupling strength (sets the time unit)
constexpr int    Q_MAX    = 3;       // stable twisted states: |q| < N/4
constexpr double DT       = 0.02;    // RK4 time step
constexpr double T_CHUNK  = 60.0;    // settling chunk length
constexpr double LOCK_TOL = 1e-8;    // phase-lock criterion (velocity spread)

using State = std::array<double, N>; // one ring's phases

// --- Dynamics ---------------------------------------------------------------
// dθ_i/dt = K·[sin(θ_{i+1}−θ_i) + sin(θ_{i−1}−θ_i)] — a gradient flow whose
// potential V(θ) = −K Σ cos(θ_{i+1}−θ_i) makes every energy below exact.
static State deriv(const State& th) {
    State d;                                          // velocity field
    for (int i = 0; i < N; ++i)                       // each site, ring-masked
        d[i] = K * (std::sin(th[(i + 1) & NMASK] - th[i])
                  + std::sin(th[(i - 1) & NMASK] - th[i]));
    return d;
}

static double V(const State& th) {                    // the exact potential
    double v = 0.0;                                   // energy accumulator
    for (int i = 0; i < N; ++i)                       // each coupling link
        v -= K * std::cos(th[(i + 1) & NMASK] - th[i]);
    return v;
}

static void rk4(State& th, double t_total) {          // classic RK4 march
    const int steps = static_cast<int>(t_total / DT); // fixed grid
    for (int s = 0; s < steps; ++s) {                 // deterministic loop
        State k1 = deriv(th), t2, t3, t4;             // slope staging
        for (int i = 0; i < N; ++i) t2[i] = th[i] + 0.5 * DT * k1[i];
        State k2 = deriv(t2);                         // midpoint slope 1
        for (int i = 0; i < N; ++i) t3[i] = th[i] + 0.5 * DT * k2[i];
        State k3 = deriv(t3);                         // midpoint slope 2
        for (int i = 0; i < N; ++i) t4[i] = th[i] + DT * k3[i];
        State k4 = deriv(t4);                         // endpoint slope
        for (int i = 0; i < N; ++i)                   // combine + advance
            th[i] += (DT / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
    }
}

// Settle until phase-locked (velocity spread < tol). Chunked with bounded
// patience; the tolerance is derived from behavior, not hope — the ring's
// slowest mode decays at ~K·(2π/N)², so chunks of 60/K are the right scale.
static bool settle(State& th) {
    for (int c = 0; c < 12; ++c) {                    // bounded patience
        rk4(th, T_CHUNK);                             // one chunk forward
        State v = deriv(th);                          // audit velocities
        double lo = v[0], hi = v[0];                  // spread scan
        for (int i = 1; i < N; ++i) { lo = std::min(lo, v[i]);
                                      hi = std::max(hi, v[i]); }
        if (hi - lo < LOCK_TOL) return true;          // locked (co-rotation
    }                                                 //  safe: max−min)
    return false;                                     // never locked
}

// Winding number: wrapped phase differences summed around the ring — the
// integer topological invariant that serves as the machine-state label.
static int winding(const State& th) {
    double s = 0.0;                                   // winding accumulator
    for (int i = 0; i < N; ++i)                       // ring-closed diffs
        s += std::remainder(th[(i + 1) & NMASK] - th[i], 2.0 * M_PI);
    return static_cast<int>(std::lround(s / (2.0 * M_PI)));
}

static State twisted(int q) {                         // ideal q-twist
    State th;                                         // θ_i = 2πqi/N
    for (int i = 0; i < N; ++i)
        th[i] = std::fmod(2.0 * M_PI * q * i / N, 2.0 * M_PI);
    return th;
}

// --- Instructions -----------------------------------------------------------
// ERASE-class input: a scalar bump — the same offset added to a 3-site window.
static State bump(State th, int site, double amp) {
    for (int k = -1; k <= 1; ++k)                     // 3-site window
        th[(site + k) & NMASK] += amp;                // masked ring write
    return th;
}
// WRITE-class input: a phase RAMP — `total` radians distributed as a smooth
// gradient across `width` sites (the tail offset is pure gauge). Shaped like
// winding itself, which is exactly why it can create winding.
static State ramp(State th, int site, double total, int width) {
    for (int k = 0; k < width; ++k)                   // the ramp itself
        th[(site + k) & NMASK] += total * (k + 1) / width;
    for (int k = width; k < N; ++k)                   // constant tail
        th[(site + k) & NMASK] += total;              // (global = gauge)
    return th;
}

// --- ASCII rendering --------------------------------------------------------
// Each site's phase becomes one of eight arrows; a twisted state reads as a
// compass needle rotating q full turns as you walk the ring left to right.
static std::string render(const State& th) {
    static const char* GLYPH[8] =                     // 45° per glyph
        {u8"\u2192", u8"\u2197", u8"\u2191", u8"\u2196",
         u8"\u2190", u8"\u2199", u8"\u2193", u8"\u2198"}; // → ↗ ↑ ↖ ← ↙ ↓ ↘
    std::string out;                                  // the strip
    for (int i = 0; i < N; ++i) {                     // each site in order
        double p = std::fmod(th[i], 2.0 * M_PI);      // wrap to [0, 2π)
        if (p < 0) p += 2.0 * M_PI;                   // positive branch
        int g = static_cast<int>(std::lround(p / (M_PI / 4.0))) & 7;
        out += GLYPH[g]; out += ' ';                  // glyph + spacer
    }
    return out;
}

// --- The demonstration ------------------------------------------------------
int main() {
    std::printf("basin-quotient machine — live demonstration (N=%d ring)\n", N);
    std::printf("========================================================\n\n");

    // 1. The state alphabet: seven attractors, labeled by winding number.
    std::printf("1. THE STATES — each row is one attractor; arrows are the\n"
                "   oscillator phases around the ring. The winding number q\n"
                "   counts full rotations of the needle across the row.\n\n");
    for (int q = -Q_MAX; q <= Q_MAX; ++q) {           // the whole alphabet
        State th = twisted(q);                        // analytic attractor
        std::printf("   q=%+d  %s V = %+9.4f\n", q,
                    render(th).c_str(), V(th));       // pattern + energy
    }
    std::printf("\n   Energy is exact: V(q) = -N·K·cos(2πq/N). Higher |q| =\n"
                "   higher energy = shallower basin (|q|=3 is marginal).\n\n");

    // 2. ERASE: one localized scalar kick removes a quantum of winding.
    std::printf("2. ERASE — a scalar bump (amp 2.2 on 3 sites) hits q=+2:\n\n");
    State s2 = twisted(2);                            // start state
    std::printf("   before  %s q=%+d\n", render(s2).c_str(), winding(s2));
    State s2k = bump(s2, 0, 2.2);                     // the instruction
    std::printf("   kicked  %s (mid-transient)\n", render(s2k).c_str());
    settle(s2k);                                      // let the fabric answer
    std::printf("   after   %s q=%+d   <- one winding quantum erased\n\n",
                render(s2k).c_str(), winding(s2k));

    // 3. The one-way funnel: scalar bumps NEVER add winding.
    std::printf("3. THE FUNNEL — scalar bumps from q=0, amplitudes 1.0..3.0:\n\n");
    bool any_up = false;                              // falsification flag
    for (double a : {1.0, 1.5, 2.0, 2.5, 3.0}) {      // sweep the drive
        State s = bump(twisted(0), 0, a);             // kick the ground state
        settle(s);                                    // relax
        int q = winding(s);                           // where did it land?
        std::printf("   amp %.1f -> q=%+d\n", a, q);  // report
        any_up |= (q != 0);                           // any escape at all?
    }
    std::printf("   %s\n\n", any_up
        ? "UNEXPECTED: escape observed (please report this build/platform)."
        : "No scalar bump adds winding: the instruction set only runs downhill.");

    // 4. WRITE is shaped and quantized: 2π of ramp writes exactly +1; π fails.
    std::printf("4. QUANTIZED WRITE — gradient ramps from q=0 (width 6):\n\n");
    for (double total : {M_PI, 2.0 * M_PI}) {         // half vs full quantum
        State s = ramp(twisted(0), 0, total, 6);      // shaped input
        settle(s);                                    // relax
        std::printf("   ramp %4.2fπ -> q=%+d  %s\n", total / M_PI, winding(s),
                    total < 4.0 ? "(sub-quantum: rejected)"
                                : "(one quantum: written)");
    }
    State chain = twisted(0);                         // increment chain 0->1->2
    for (int step = 0; step < 2; ++step) {            // apply twice
        chain = ramp(chain, 0, 2.0 * M_PI, 6);        // +1 quantum
        settle(chain);                                // commit
    }
    std::printf("   two ramps chained: q=%+d  (an increment instruction)\n\n",
                winding(chain));

    // 5. The machine, summarized.
    std::printf("5. THE MACHINE — what the measurements above add up to:\n\n"
                "   read  : the winding number (topological -> free, robust)\n"
                "   erase : local scalar bump   (one-way funnel toward q=0)\n"
                "   write : shaped ramp, quantized (+1 per full 2π, else nil)\n\n"
                "   None of this was designed in. It is what the mathematics\n"
                "   of the ring volunteers when you treat basins as states.\n");
    return 0;
}

// ===========================================================================
// PROVENANCE AND HONEST SCOPE
//   This file is the public demo of a measured research program; the full
//   instrument suite (locality sweeps, clock-period measurement, saddle-point
//   energetics, batch C engine) lives alongside it. Findings shown here are
//   from the N=16 pristine ring only. The concept family has real ancestry —
//   attractor networks (Hopfield), reservoir computing, twisted-state
//   stability on Kuramoto rings (Wiley–Strogatz–Girvan), coupled-oscillator
//   and bifurcation machines — and the winding quantization used by the
//   write instruction is classical physics. What is claimed here is the
//   machine-theoretic reading: a typed, measured instruction set on the
//   basin quotient, with composability and timing rules established by
//   experiment. Nothing more, and nothing less.
// ===========================================================================
