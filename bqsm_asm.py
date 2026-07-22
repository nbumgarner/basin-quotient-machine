#!/usr/bin/env python3
# ===========================================================================
# bqsm_asm.py — BQSM Assembler and Programming Model (Fork B)
# ===========================================================================
#   emerging.systems — Basin-Quotient State Machine research fork
#
# A human-readable assembly language for the basin-quotient machine, an
# assembler that parses it into an executable Program, and a reference VM
# that executes programs against the measured transition table.
#
# DESIGN PRINCIPLE
#   The instruction set is TYPED by measurement, not designed:
#     READ  = topological invariant (free, no energy cost)
#     WRITE = quantized increment (+1 per full 2π ramp)
#     ERASE = one-way funnel (localized scalar bump toward q=0)
#     RESET = strong ERASE (guaranteed q=0 from any state)
#     KICK  = generic scalar perturbation (amplitude-dependent)
#
#   This is not an arbitrary ISA — the types are physical constraints.
#   WRITE from |q|=3 destroys the machine (overflow into instability).
#   ERASE never increases |q|. RESET always reaches the ground state.
#
# PHYSICS CONTEXT (pristine N=16 Kuramoto ring)
#   N=16 oscillators, Q_MAX=3 (7 stable states: -3,-2,-1,0,1,2,3)
#   Radix 1 (binary): q ∈ {0,1}
#   Radix 2 (ternary): q ∈ {0,1,2}
#   Radix 3 (quad):    q ∈ {0,1,2,3}
#
#   Transition table (measured, not designed):
#                q=0    q=1    q=2    q=3
#     ERASE(2.2) q=0    q=0    q=1    q=2
#     WRITE      q=1    q=2    q=3    DEAD
#     RESET      q=0    q=0    q=0    q=0
#     KICK(1.0)  q=0    q=1    q=1    q=2
#     KICK(2.2)  q=0    q=0    q=1    q=2
#
#   Energy model: V(q) = -16 · cos(2π·q/16)
#     q=0: -16.00 (ground, deepest)
#     q=1: -14.78
#     q=2: -11.31
#     q=3:  -6.12 (marginal, shallowest)
#
# DESIGN NOTES
#   - Structural pattern matching (match/case) throughout
#   - Dataclasses for all data models
#   - stdlib only — no external dependencies
#   - Reference VM executes against the transition table
#   - Energy costs annotated on every instruction
#   - Assembly source: mnemonics, labels, numeric operands, ; comments
# ===========================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Union
import math
import sys
import os


# ═══════════════════════════════════════════════════════════════════════════
# Physical constants (match the instrument suite)
# ═══════════════════════════════════════════════════════════════════════════

N_OSC  = 16             # oscillators on the ring
Q_MAX  = 3              # maximum stable winding |q| < N/4
K      = 1.0            # coupling strength (sets energy scale)


def potential_energy(q: int) -> float:
    """Exact potential energy of twisted state q.
    V(q) = -K·N·cos(2πq/N). For the pristine symmetric ring this is a
    gradient flow: dθ/dt = −∇V, so energies are physically meaningful."""
    return float(-K * N_OSC * math.cos(2 * math.pi * q / N_OSC))


# Precomputed energy table for annotation
ENERGY_TABLE: dict[int, float] = {q: potential_energy(q) for q in range(Q_MAX + 1)}
# q=0: -16.0000 (ground state, deepest basin)
# q=1: -14.7821
# q=2: -11.3137
# q=3:  -6.1229 (marginal: near stability boundary |q| >= N/4=4)


# ═══════════════════════════════════════════════════════════════════════════
# State model
# ═══════════════════════════════════════════════════════════════════════════

class Radix(Enum):
    """Operating mode — constrains the active state alphabet.
    Lower radix = fewer states available but more robust against overflow."""
    BINARY  = 1    # q ∈ {0, 1}
    TERNARY = 2    # q ∈ {0, 1, 2}
    QUAD    = 3    # q ∈ {0, 1, 2, 3}  (full Q_MAX used)

    @property
    def max_q(self) -> int:
        """Maximum |q| allowed in this mode."""
        return self.value

    @property
    def alphabet(self) -> tuple[int, ...]:
        """All valid state labels in this mode."""
        return tuple(range(self.max_q + 1))

    @classmethod
    def parse(cls, value: Union[int, str]) -> Radix:
        """Parse a radix from an integer or string name."""
        match value:
            case 1 | "BINARY" | "BIN":
                return cls.BINARY
            case 2 | "TERNARY" | "TER":
                return cls.TERNARY
            case 3 | "QUAD" | "QUAD":
                return cls.QUAD
            case _:
                raise ValueError(f"unknown radix: {value!r}")


@dataclass(frozen=True)
class MachineState:
    """A discrete state of the BQSM: the winding number q.
    Labels a stable attractor in the ring's phase space. q is a topological
    invariant — it survives small parameter changes and is read for free."""
    q: int

    def is_valid(self, radix: Radix) -> bool:
        """True if this state is within the current radix alphabet."""
        return 0 <= self.q <= radix.max_q

    @property
    def energy(self) -> float:
        """Potential energy of this attractor basin."""
        return potential_energy(self.q)

    @property
    def is_ground(self) -> bool:
        """Is this the ground state (q=0, deepest basin)?"""
        return self.q == 0

    @property
    def is_marginal(self) -> bool:
        """Marginal states (|q|=Q_MAX) are near the stability boundary.
        They are the first to die under a lens/perturbation."""
        return self.q == Q_MAX

    @property
    def is_robust(self) -> bool:
        """Robust states (|q| < Q_MAX) are well inside the stability region."""
        return not self.is_marginal

    def __str__(self) -> str:
        return f"q={self.q:+d}" if self.q != 0 else "q=0"

    def __repr__(self) -> str:
        return f"MachineState(q={self.q})"


# ═══════════════════════════════════════════════════════════════════════════
# Instruction Set Architecture
# ═══════════════════════════════════════════════════════════════════════════

class Opcode(Enum):
    """BQSM instruction opcodes — typed by physical measurement."""

    # ── Physical instructions (modify the machine state) ────────────────

    WRITE = auto()     # quantized increment: +1 per 2π ramp; |q|=3 → DEAD
    ERASE = auto()     # one-way funnel: localized scalar bump toward q=0
    RESET = auto()     # strong erase: guaranteed to reach q=0 from any state
    KICK  = auto()     # generic scalar kick with specified amplitude
    READ  = auto()     # topological readout: free, no energy cost

    # ── Control flow (conventional, for programmability) ────────────────

    NOP = auto()       # no operation (one clock cycle, no state change)
    HLT = auto()       # halt execution
    JMP = auto()       # unconditional jump to label
    JEQ = auto()       # jump to label if current state equals expected value
    JNE = auto()       # jump to label if current state not equal to expected

    # ── Assembler directives (compile-time) ─────────────────────────────

    RADIX = auto()     # set operating mode (BINARY / TERNARY / QUAD)
    ASSERT = auto()    # runtime assertion: verify current state matches expected

    # ── Internal (assembler-only, not emitted) ──────────────────────────

    LABEL = auto()     # label definition point — not an executable instruction

    def is_physical(self) -> bool:
        """True for instructions that physically act on the ring."""
        return self in (Opcode.WRITE, Opcode.ERASE, Opcode.RESET,
                        Opcode.KICK, Opcode.READ)

    def is_control(self) -> bool:
        """True for control-flow instructions (VM overhead)."""
        return self in (Opcode.NOP, Opcode.HLT, Opcode.JMP,
                        Opcode.JEQ, Opcode.JNE)

    def is_directive(self) -> bool:
        """True for assembler directives (not executed)."""
        return self in (Opcode.RADIX, Opcode.ASSERT, Opcode.LABEL)


# ── Mnemonic table for parsing ─────────────────────────────────────────────

MNEMONIC_TABLE: dict[str, Opcode] = {
    "WRITE":  Opcode.WRITE,
    "ERASE":  Opcode.ERASE,
    "RESET":  Opcode.RESET,
    "KICK":   Opcode.KICK,
    "READ":   Opcode.READ,
    "NOP":    Opcode.NOP,
    "HLT":    Opcode.HLT,
    "HALT":   Opcode.HLT,      # alias
    "JMP":    Opcode.JMP,
    "JUMP":   Opcode.JMP,      # alias
    "JEQ":    Opcode.JEQ,
    "JNE":    Opcode.JNE,
    "RADIX":  Opcode.RADIX,
    "ASSERT": Opcode.ASSERT,
}


@dataclass(frozen=True)
class Instruction:
    """A single assembled instruction in the BQSM program.

    Fields:
        opcode:      The operation to perform.
        operand:     Numeric argument (KICK amplitude, ASSERT expected q)
                     or string label name (JMP/JEQ/JNE target, before resolution)
                     or None for zero-operand instructions.
        expected_q:  For JEQ/JNE: the state value to compare against.
        line:        Source line number (for error reporting).
        comment:     Trailing comment text preserved from source.
        energy_cost: Energy change for this instruction (annotated post-assembly).
    """
    opcode: Opcode
    operand: Optional[Union[int, float, str]] = None
    expected_q: Optional[int] = None
    line: int = 0
    comment: str = ""
    energy_cost: float = 0.0

    @property
    def is_conditional(self) -> bool:
        """True for conditional jump instructions."""
        return self.opcode in (Opcode.JEQ, Opcode.JNE)

    @property
    def is_jump(self) -> bool:
        """True for any jump instruction (conditional or unconditional)."""
        return self.opcode in (Opcode.JMP, Opcode.JEQ, Opcode.JNE)

    @property
    def target_address(self) -> Optional[int]:
        """Resolved jump target address, or None if not yet resolved."""
        if self.is_jump and isinstance(self.operand, int):
            return self.operand
        return None

    def __str__(self) -> str:
        parts = [self.opcode.name]
        if self.operand is not None:
            parts.append(str(self.operand))
        if self.expected_q is not None:
            parts.append(str(self.expected_q))
        return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Transition Table — the measured machine behavior
# ═══════════════════════════════════════════════════════════════════════════

class TransitionFn:
    """The pristine N=16 ring's measured transition function.

    Encodes the outcome of every physical instruction on every state.
    The table is the result of measurement (integrate + winding readout),
    not a designed mapping. It is deterministic and reproduces exactly
    across all runs on the same pristine fabric.

    The key finding: this is a genuine state machine — the outcome depends
    only on (current state label, instruction), not on history or site.
    """

    # Internal table: {(instruction_key, q_from) → q_to | None}
    # None represents DEAD (machine destroyed: |q|=3 + WRITE overflow).
    _table: dict[tuple[str, int], Optional[int]]

    def __init__(self):
        t: dict[tuple[str, int], Optional[int]] = {}

        # ── ERASE (amp=2.2 scalar bump at site 0) ──────────────────
        # One-way funnel: never increases |q|. q=0 is absorbing.
        t[("ERASE", 0)] = 0
        t[("ERASE", 1)] = 0
        t[("ERASE", 2)] = 1
        t[("ERASE", 3)] = 2

        # ── WRITE (2π phase ramp, width 6) ─────────────────────────
        # Quantized: exactly +1 per full 2π. |q|=3 overflows → DEAD.
        t[("WRITE", 0)] = 1
        t[("WRITE", 1)] = 2
        t[("WRITE", 2)] = 3
        t[("WRITE", 3)] = None   # DEAD: machine destroyed

        # ── RESET (strong ERASE) ───────────────────────────────────
        # Guaranteed to reach q=0 from any state.
        for q in range(4):
            t[("RESET", q)] = 0

        # ── KICK(1.0) (sub-critical) ───────────────────────────────
        t[("KICK_1.0", 0)] = 0
        t[("KICK_1.0", 1)] = 1
        t[("KICK_1.0", 2)] = 1
        t[("KICK_1.0", 3)] = 2

        # ── KICK(2.2) (erase-regime) ───────────────────────────────
        t[("KICK_2.2", 0)] = 0
        t[("KICK_2.2", 1)] = 0
        t[("KICK_2.2", 2)] = 1
        t[("KICK_2.2", 3)] = 2

        self._table = t

    def _key(self, instr: Instruction) -> str:
        """Map an instruction to the table lookup key."""
        match instr.opcode:
            case Opcode.ERASE:
                return "ERASE"
            case Opcode.WRITE:
                return "WRITE"
            case Opcode.RESET:
                return "RESET"
            case Opcode.KICK:
                amp = float(instr.operand) if instr.operand is not None else 1.0
                # Snap to known amplitude bins
                if abs(amp - 1.0) < 0.05:
                    return "KICK_1.0"
                elif abs(amp - 2.2) < 0.05:
                    return "KICK_2.2"
                else:
                    # Unknown amplitude: use interpolation heuristic
                    return f"KICK_{amp:.1f}"
            case _:
                return instr.opcode.name

    def lookup(self, instr: Instruction, q: int) -> Optional[int]:
        """Look up the outcome of an instruction from state q.
        Returns the new state label, or None for DEAD."""
        key = self._key(instr)
        result = self._table.get((key, q))

        if result is not None:
            return result

        # Fallback: unknown KICK amplitude — use linear interpolation
        # between the known sub-critical (1.0) and erase-regime (2.2) behaviors
        if instr.opcode == Opcode.KICK and instr.operand is not None:
            amp = float(instr.operand)
            r1 = self._table.get(("KICK_1.0", q))
            r2 = self._table.get(("KICK_2.2", q))
            if r1 is not None and r2 is not None:
                if 1.0 <= amp <= 2.2:
                    # Funnel deepens with amplitude
                    return r2  # conservative: use the deeper-erase result
                elif amp < 1.0:
                    return r1  # weaker than sub-critical: no effect
                else:
                    # Stronger than 2.2: extrapolate toward RESET
                    return max(0, (r2 or q) - 1)

        return None  # DEAD or unknown

    def apply(self, instr: Instruction, state: MachineState) -> Optional[MachineState]:
        """Apply instruction to state. Returns new state or None (DEAD)."""
        q_out = self.lookup(instr, state.q)
        return MachineState(q=q_out) if q_out is not None else None

    def energy_delta(self, instr: Instruction, q_from: int) -> float:
        """Compute the potential energy change for this instruction.

        READ is free (topological invariant).
        Control-flow instructions are VM overhead (zero physical energy).
        Physical instructions cost the difference in basin energies.
        """
        match instr.opcode:
            case Opcode.READ:
                return 0.0       # topological readout: no energy cost
            case Opcode.NOP | Opcode.HLT:
                return 0.0       # control flow is simulator overhead
            case Opcode.JMP | Opcode.JEQ | Opcode.JNE:
                return 0.0
            case Opcode.RADIX | Opcode.ASSERT | Opcode.LABEL:
                return 0.0       # compile-time directives
            case _:
                q_to = self.lookup(instr, q_from)
                if q_to is None:
                    return 0.0   # DEAD: catastrophic, energy undefined
                return potential_energy(q_to) - potential_energy(q_from)

    def display(self) -> str:
        """Pretty-print the transition table."""
        lines = []
        lines.append("  Transition Table (pristine N=16 ring)")
        lines.append("  " + "─" * 48)
        header = f"  {'':>12}  " + "  ".join(f"q={q:<4}" for q in range(4))
        lines.append(header)
        for key_name in ("ERASE", "WRITE", "RESET", "KICK_1.0", "KICK_2.2"):
            row = f"  {key_name:<12}  "
            for q in range(4):
                val = self._table.get((key_name, q))
                row += f"  {str(val):>4}"
            lines.append(row)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Assembler: Lexer + Parser + Resolver
# ═══════════════════════════════════════════════════════════════════════════

class AsmError(Exception):
    """Assembly error with source location."""
    def __init__(self, msg: str, line: int = 0, col: int = 0):
        self.line = line
        self.col = col
        loc = f"line {line}" if line else ""
        prefix = f"{loc}: " if loc else ""
        super().__init__(f"{prefix}{msg}")


# ── Lexer ──────────────────────────────────────────────────────────────────

@dataclass
class Token:
    """A single token from the assembly source."""
    kind: str           # MNEMONIC, NUMBER, COLON, IDENT, STRING, COMMENT, EOL
    value: str
    line: int
    col: int

    def __repr__(self) -> str:
        return f"Token({self.kind}, {self.value!r}, L{self.line})"


def lex(source: str) -> list[Token]:
    """Tokenize BQSM assembly source into a flat token list.

    Syntax:
        MNEMONIC [operand] [; comment]
        :label
        ; comment line

    The language is line-oriented; labels end with ':' and stand alone
    on their line. Comments begin with ';' and run to end of line.
    """
    tokens: list[Token] = []
    for lineno, raw_line in enumerate(source.splitlines(), start=1):
        # Strip comments
        comment_text = ""
        if ";" in raw_line:
            raw_line, comment_text = raw_line.split(";", 1)
            comment_text = comment_text.strip()

        stripped = raw_line.strip()
        if not stripped and not comment_text:
            if comment_text:
                tokens.append(Token("COMMENT", comment_text, lineno, 0))
            continue

        # Split into parts
        parts = stripped.split()

        for part in parts:
            # Find column position for error reporting
            col_offset = raw_line.find(part) if raw_line else 0

            if part.endswith(":"):
                # Label definition: the colon is part of the label token
                label_name = part[:-1]
                if not label_name:
                    raise AsmError("empty label name", lineno, col_offset)
                tokens.append(Token("COLON", label_name, lineno, col_offset))
            else:
                # Try numeric, then mnemonic, then identifier
                try:
                    if "." in part:
                        float(part)  # validate
                        tokens.append(Token("NUMBER", part, lineno, col_offset))
                    else:
                        int(part)    # validate
                        tokens.append(Token("NUMBER", part, lineno, col_offset))
                except ValueError:
                    tokens.append(Token("MNEMONIC", part.upper(), lineno, col_offset))

        if comment_text:
            tokens.append(Token("COMMENT", comment_text, lineno, len(raw_line)))

        tokens.append(Token("EOL", "", lineno, 0))

    return tokens


# ── Parser ─────────────────────────────────────────────────────────────────

@dataclass
class Program:
    """A fully assembled BQSM program, ready for execution.

    After assembly, all labels are resolved to instruction indices,
    the radix is determined, and energy costs are annotated.
    """
    instructions: list[Instruction] = field(default_factory=list)
    labels: dict[str, int] = field(default_factory=dict)
    radix: Radix = Radix.QUAD
    source_lines: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def total_energy(self) -> float:
        """Sum of all physical instruction energy costs (lower bound)."""
        return sum(i.energy_cost for i in self.instructions
                   if i.opcode.is_physical())

    def annotate_energies(self, tfn: TransitionFn):
        """Compute and set energy_cost on every instruction."""
        # We need to simulate the state to know q_from for each instruction.
        # Start from q=0 (ground) and walk the program path.
        q = 0
        for instr in self.instructions:
            match instr.opcode:
                case Opcode.RADIX:
                    pass  # no state change, energy already 0
                case Opcode.ASSERT:
                    pass
                case Opcode.READ:
                    pass
                case _:
                    delta = tfn.energy_delta(instr, q)
                    # Use object.__setattr__ to update the frozen dataclass
                    object.__setattr__(instr, 'energy_cost', delta)
                    q_out = tfn.lookup(instr, q)
                    if q_out is not None:
                        q = q_out
                    # For control flow, q doesn't change

    def disassemble(self) -> str:
        """Pretty-print the assembled program with addresses and annotations."""
        lines: list[str] = []
        lines.append(f"; BQSM program  radix={self.radix.name}  "
                     f"instructions={len(self.instructions)}  "
                     f"energy_bound={self.total_energy:.2f}")
        lines.append("")

        # Build reverse label map
        rev_labels: dict[int, str] = {}
        for name, addr in self.labels.items():
            rev_labels.setdefault(addr, name)

        for i, instr in enumerate(self.instructions):
            label = rev_labels.get(i, "")
            lbl_str = f"{label}:".ljust(14) if label else " " * 14

            cost_str = ""
            if instr.energy_cost != 0.0:
                cost_str = f"  ; ΔE={instr.energy_cost:+.2f}"

            comment_str = f"  ; {instr.comment}" if instr.comment else ""

            lines.append(f"  {i:>3d}: {lbl_str}{instr}{cost_str}{comment_str}")

        return "\n".join(lines)

    def summary(self) -> str:
        """One-line program summary."""
        phys = sum(1 for i in self.instructions if i.opcode.is_physical())
        ctrl = sum(1 for i in self.instructions if i.opcode.is_control())
        return (f"Program: {len(self.instructions)} instrs "
                f"({phys} physical, {ctrl} control), "
                f"radix={self.radix.name}, "
                f"energy={self.total_energy:.2f}")


def parse(tokens: list[Token]) -> Program:
    """Parse a token stream into a Program.

    Two-pass assembly:
      1. Parse mnemonics + operands into raw instructions, collecting labels.
      2. For JEQ/JNE, parse the expected_q from a second operand.
      3. Resolve label references in JMP/JEQ/JNE to instruction indices.
      4. Determine radix from RADIX directives.
    """
    prog = Program()
    labels: dict[str, int] = {}
    raw_instrs: list[Instruction] = []

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        match tok.kind:
            case "EOL":
                i += 1
                continue

            case "COMMENT":
                # Standalone comment line: skip
                i += 1
                continue

            case "MNEMONIC":
                mnemonic = tok.value
                if mnemonic not in MNEMONIC_TABLE:
                    prog.errors.append(
                        f"line {tok.line}: unknown mnemonic '{mnemonic}'")
                    i += 1
                    continue

                opcode = MNEMONIC_TABLE[mnemonic]
                operand: Optional[Union[int, float, str]] = None
                expected_q: Optional[int] = None
                i += 1

                # Read first operand
                if i < len(tokens) and tokens[i].kind not in ("EOL", "COMMENT", "COLON"):
                    op_tok = tokens[i]
                    match op_tok.kind:
                        case "NUMBER":
                            operand = (float(op_tok.value) if "." in op_tok.value
                                       else int(op_tok.value))
                        case "MNEMONIC":
                            # Label reference for JMP/JEQ/JNE
                            operand = op_tok.value
                        case _:
                            prog.errors.append(
                                f"line {tok.line}: unexpected token "
                                f"'{op_tok.value}' after {mnemonic}")
                    i += 1

                # For JEQ/JNE, read expected_q as second numeric operand
                if opcode in (Opcode.JEQ, Opcode.JNE):
                    if i < len(tokens) and tokens[i].kind == "NUMBER":
                        expected_q = int(tokens[i].value)
                        i += 1

                # Consume trailing comment
                comment = ""
                if i < len(tokens) and tokens[i].kind == "COMMENT":
                    comment = tokens[i].value
                    i += 1

                raw_instrs.append(Instruction(
                    opcode=opcode,
                    operand=operand,
                    expected_q=expected_q,
                    line=tok.line,
                    comment=comment,
                ))

            case "COLON":
                # Label definition: <name>:
                label_name = tok.value
                if label_name in labels:
                    prog.errors.append(
                        f"line {tok.line}: duplicate label '{label_name}'")
                else:
                    labels[label_name] = len(raw_instrs)
                i += 1

            case "NUMBER":
                prog.errors.append(
                    f"line {tok.line}: unexpected number '{tok.value}' "
                    f"— did you forget a mnemonic?")
                i += 1

            case _:
                i += 1

    # ── Pass 2: resolve label references ──────────────────────────────────

    instructions: list[Instruction] = []
    for instr in raw_instrs:
        match instr.opcode:
            case Opcode.JMP | Opcode.JEQ | Opcode.JNE:
                if isinstance(instr.operand, str):
                    label_name = instr.operand
                    if label_name not in labels:
                        prog.errors.append(
                            f"line {instr.line}: undefined label '{label_name}'")
                        instructions.append(instr)
                    else:
                        instructions.append(Instruction(
                            opcode=instr.opcode,
                            operand=labels[label_name],
                            expected_q=instr.expected_q,
                            line=instr.line,
                            comment=instr.comment,
                        ))
                elif instr.operand is None:
                    prog.errors.append(
                        f"line {instr.line}: {instr.opcode.name} requires "
                        f"a label operand")
                    instructions.append(instr)
                else:
                    instructions.append(instr)
            case _:
                instructions.append(instr)

    # ── Pass 3: determine radix from RADIX directives ─────────────────────

    radix = Radix.QUAD
    for instr in instructions:
        if instr.opcode == Opcode.RADIX:
            try:
                match instr.operand:
                    case int() | float():
                        radix = Radix.parse(int(instr.operand))
                    case str():
                        radix = Radix.parse(instr.operand)
                    case None:
                        prog.warnings.append(
                            f"line {instr.line}: RADIX without operand, "
                            f"defaulting to QUAD")
            except ValueError as e:
                prog.warnings.append(
                    f"line {instr.line}: {e}, defaulting to QUAD")

    prog.instructions = instructions
    prog.labels = labels
    prog.radix = radix

    return prog


def assemble(source: str) -> Program:
    """Full assembly pipeline: source text → assembled Program.

    After assembly, call program.annotate_energies(tfn) to add energy
    annotations, and use VM(program) to execute.

    Raises AsmError if there are assembly errors.
    """
    tokens = lex(source)
    prog = parse(tokens)
    if prog.has_errors:
        raise AsmError(
            f"Assembly failed with {len(prog.errors)} error(s):\n" +
            "\n".join(f"  {e}" for e in prog.errors))
    tfn = TransitionFn()
    prog.annotate_energies(tfn)
    return prog


def assemble_file(path: str) -> Program:
    """Assemble a .asm source file. Returns assembled Program."""
    if not os.path.exists(path):
        raise AsmError(f"file not found: {path}")
    with open(path) as f:
        source = f.read()
    return assemble(source)


# ═══════════════════════════════════════════════════════════════════════════
# Reference Virtual Machine
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VMState:
    """Full VM snapshot after execution completes."""
    pc: int                        # final program counter
    state: MachineState            # final machine state
    cycles: int                    # total clock cycles executed
    halted: bool = False           # normal termination?
    dead: bool = False             # machine destroyed? (|q|=3 + WRITE)
    history: list[tuple[int, str, int]] = field(default_factory=list)
    # history entries: (pc, description, q_after)
    assertions_checked: int = 0
    assertions_failed: int = 0

    @property
    def is_ok(self) -> bool:
        """Execution completed without error."""
        return self.halted and not self.dead

    @property
    def is_dead(self) -> bool:
        """Machine was destroyed by overflow."""
        return self.dead


class VM:
    """Reference VM for the BQSM.

    Executes an assembled Program against the measured TransitionFn.
    The VM is a clock-cycle-level simulator: each physical instruction
    corresponds to one kick-and-settle operation on the ring.

    Usage:
        prog = assemble(source)
        vm = VM(prog, init_q=0)
        result = vm.run()
        print(vm.trace())
    """

    def __init__(self, program: Program, init_q: int = 0):
        if init_q < 0:
            raise ValueError(f"init_q={init_q} must be non-negative: "
                             "use positive basins only")
        if init_q > program.radix.max_q:
            raise ValueError(f"init_q={init_q} exceeds radix max "
                             f"({program.radix.max_q})")

        self.prog = program
        self.tfn = TransitionFn()
        self.state = VMState(
            pc=0,
            state=MachineState(q=init_q),
            cycles=0,
        )

    def step(self) -> bool:
        """Execute one instruction. Returns False if execution terminated."""
        if self.state.halted or self.state.dead:
            return False
        if self.state.pc >= len(self.prog.instructions):
            self.state.halted = True
            return False

        instr = self.prog.instructions[self.state.pc]
        self.state.cycles += 1
        q_cur = self.state.state.q
        radix = self.prog.radix

        match instr.opcode:

            # ── Physical instructions ────────────────────────────────────

            case Opcode.WRITE:
                if q_cur >= radix.max_q:
                    # Overflow: |q| at radix max + WRITE → machine destroyed
                    self.state.dead = True
                    self.state.history.append(
                        (self.state.pc, "WRITE→DEAD", -1))
                    return False
                new_q = q_cur + 1
                self.state.state = MachineState(q=new_q)
                self.state.history.append(
                    (self.state.pc, f"WRITE", new_q))
                self.state.pc += 1

            case Opcode.ERASE:
                q_out = self.tfn.lookup(instr, q_cur)
                if q_out is None:
                    self.state.dead = True
                    self.state.history.append(
                        (self.state.pc, "ERASE→DEAD", -1))
                    return False
                self.state.state = MachineState(q=q_out)
                self.state.history.append(
                    (self.state.pc, f"ERASE", q_out))
                self.state.pc += 1

            case Opcode.RESET:
                # Guaranteed q=0 from any state
                self.state.state = MachineState(q=0)
                self.state.history.append(
                    (self.state.pc, "RESET→0", 0))
                self.state.pc += 1

            case Opcode.KICK:
                amp = float(instr.operand) if instr.operand else 1.0
                q_out = self.tfn.lookup(instr, q_cur)
                if q_out is None:
                    self.state.dead = True
                    self.state.history.append(
                        (self.state.pc, f"KICK({amp:.1f})→DEAD", -1))
                    return False
                self.state.state = MachineState(q=q_out)
                self.state.history.append(
                    (self.state.pc, f"KICK({amp:.1f})", q_out))
                self.state.pc += 1

            case Opcode.READ:
                # Topological readout: observe without changing state
                self.state.history.append(
                    (self.state.pc, f"READ", q_cur))
                self.state.pc += 1

            # ── Control flow ────────────────────────────────────────────

            case Opcode.NOP:
                self.state.history.append(
                    (self.state.pc, "NOP", q_cur))
                self.state.pc += 1

            case Opcode.HLT:
                self.state.halted = True
                self.state.history.append(
                    (self.state.pc, "HLT", q_cur))
                return False

            case Opcode.JMP:
                match instr.operand:
                    case int():
                        target = instr.operand
                        if 0 <= target < len(self.prog.instructions):
                            self.state.history.append(
                                (self.state.pc, f"JMP→{target}", q_cur))
                            self.state.pc = target
                        else:
                            self.state.halted = True
                            self.state.history.append(
                                (self.state.pc, f"JMP→OOB:{target}", q_cur))
                            return False
                    case _:
                        self.state.halted = True
                        return False

            case Opcode.JEQ:
                match instr.operand:
                    case int():
                        target = instr.operand
                        expected = instr.expected_q
                        if (expected is None or q_cur == expected):
                            if 0 <= target < len(self.prog.instructions):
                                taken = "taken" if expected == q_cur else "bare"
                                self.state.history.append(
                                    (self.state.pc,
                                     f"JEQ→{target} ({taken})", q_cur))
                                self.state.pc = target
                            else:
                                self.state.halted = True
                                return False
                        else:
                            self.state.history.append(
                                (self.state.pc,
                                 f"JEQ✗ (q={q_cur}≠{expected})", q_cur))
                            self.state.pc += 1
                    case _:
                        self.state.halted = True
                        return False

            case Opcode.JNE:
                match instr.operand:
                    case int():
                        target = instr.operand
                        expected = instr.expected_q
                        if expected is not None and q_cur != expected:
                            if 0 <= target < len(self.prog.instructions):
                                self.state.history.append(
                                    (self.state.pc,
                                     f"JNE→{target} (q={q_cur}≠{expected})",
                                     q_cur))
                                self.state.pc = target
                            else:
                                self.state.halted = True
                                return False
                        else:
                            self.state.history.append(
                                (self.state.pc,
                                 f"JNE✗ (q={q_cur})", q_cur))
                            self.state.pc += 1
                    case _:
                        self.state.halted = True
                        return False

            # ── Directives ──────────────────────────────────────────────

            case Opcode.RADIX:
                # Already handled at assembly time; no-op at runtime
                self.state.pc += 1

            case Opcode.ASSERT:
                expected = instr.operand
                if isinstance(expected, (int, float)):
                    exp_q = int(expected)
                    if q_cur != exp_q:
                        self.state.assertions_failed += 1
                        self.state.history.append(
                            (self.state.pc,
                             f"ASSERT FAIL (q={q_cur}≠{exp_q})", q_cur))
                    else:
                        self.state.assertions_checked += 1
                        self.state.history.append(
                            (self.state.pc, f"ASSERT OK q={q_cur}", q_cur))
                self.state.pc += 1

            case Opcode.LABEL:
                self.state.pc += 1

        return not self.state.halted and not self.state.dead

    def run(self, max_steps: int = 10_000) -> VMState:
        """Execute until termination or max_steps exhausted."""
        for _ in range(max_steps):
            if not self.step():
                break
        return self.state

    def trace(self) -> str:
        """Return formatted execution trace."""
        lines: list[str] = []
        lines.append(f"Execution trace ({len(self.state.history)} steps):")
        lines.append(f"  {'PC':>4}  {'Instruction':<24}  q_after")
        lines.append(f"  {'─'*4}  {'─'*24}  {'─'*6}")
        for pc, desc, q in self.state.history:
            lines.append(f"  {pc:>4d}  {desc:<24}  {q:+d}")
        lines.append("")
        lines.append(f"Final state: {self.state.state}  "
                     f"cycles={self.state.cycles}  "
                     f"halted={self.state.halted}  dead={self.state.dead}")
        if self.state.dead:
            lines.append("*** MACHINE DESTROYED: |q|=radix_max + WRITE → overflow ***")
        if self.state.assertions_failed:
            lines.append(f"Assertions: {self.state.assertions_checked} OK, "
                         f"{self.state.assertions_failed} FAILED")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Example programs — demonstrating the programming model
# ═══════════════════════════════════════════════════════════════════════════

EXAMPLE_INCREMENT = """\
; increment.asm — WRITE from q=0 to q=1 to q=2 in quad mode.
; Demonstrates the quantized write: exactly +1 per full 2π ramp.
RADIX QUAD
WRITE        ; q=0 → q=1  (one quantum of winding)
READ         ; observe q=1
WRITE        ; q=1 → q=2  (second quantum)
READ         ; observe q=2
HLT
"""

EXAMPLE_ERASE_FUNNEL = """\
; funnel.asm — ERASE always moves toward q=0, never away.
; Demonstrates the one-way funnel property: ERASE never increases |q|.
; Run with init_q=2
RADIX QUAD
ERASE        ; q=2 → q=1  (funnel one step)
ERASE        ; q=1 → q=0  (funnel another)
ERASE        ; q=0 → q=0  (absorbing: q=0 is the funnel's sink)
READ         ; observe q=0
HLT
"""

EXAMPLE_RESET = """\
; reset.asm — RESET is strong ERASE: guaranteed q=0 from any state.
; Run with init_q=3 (marginal state)
RADIX QUAD
RESET        ; q=3 → q=0  guaranteed by the funnel theorem
READ         ; observe q=0
HLT
"""

EXAMPLE_OVERFLOW = """\
; overflow.asm — WRITE from |q|=3 destroys the machine.
; Demonstrates the DEAD state and the radix safety boundary.
RADIX QUAD
WRITE        ; q=0 → q=1
WRITE        ; q=1 → q=2
WRITE        ; q=2 → q=3  (now at the marginal edge)
WRITE        ; q=3 → DEAD (overflow: machine destroyed)
READ         ; never reached
HLT
"""

EXAMPLE_TERNARY = """\
; ternary.asm — operate in ternary mode (q ∈ {0,1,2})
; WRITE at radix max = 2 → DEAD
RADIX TERNARY
WRITE        ; q=0 → q=1
WRITE        ; q=1 → q=2
WRITE        ; q=2 → DEAD  (radix max reached)
HLT
"""

EXAMPLE_BINARY = """\
; binary.asm — operate in binary mode (q ∈ {0,1})
; This is a single-bit register. WRITE toggles 0↔1.
; WRITE from q=1 in binary mode → DEAD
RADIX BINARY
WRITE        ; q=0 → q=1
READ
HLT
"""

EXAMPLE_KICKS = """\
; kicks.asm — sub-critical vs erase-regime kicks.
; Demonstrates the amplitude-dependent kick behavior.
; Run with init_q=2 for full effect.
RADIX QUAD
KICK 1.0     ; sub-critical: q=2 → q=1  (gentle nudge)
READ         ; observe q=1
KICK 2.2     ; erase-regime: q=1 → q=0  (deeper funnel)
READ         ; observe q=0
HLT
"""

EXAMPLE_LOOP = """\
; countdown.asm — WRITE then ERASE loop: a toggle program.
; Demonstrates control flow + the typed instruction set.
RADIX QUAD
:loop
READ         ; observe current state
WRITE        ; increment (q=0→1, q=1→2, q=2→3)
ERASE        ; funnel down one step
READ
JMP loop     ; repeat forever
; (never reaches HLT — use the step limit or Ctrl-C)
HLT
"""

EXAMPLE_CONDITIONAL = """\
; conditional.asm — branch on state after ERASE.
; Demonstrates JEQ/JNE for state-dependent control flow.
; Run with init_q=2
RADIX QUAD
ERASE        ; q=2 → q=1
JEQ zero 0   ; jump to :zero if q==0 (not taken: q=1)
READ         ; we land here: q=1
ERASE        ; q=1 → q=0
READ
JMP done     ; skip over the zero-label section
:zero
READ         ; only reached if q was already 0
:done
HLT
"""

EXAMPLE_KICK_STRENGTH_SWEEP = """\
; kicksweep.asm — survey the full kick amplitude range.
; Demonstrates how different amplitudes map to different funnel depths.
RADIX QUAD
; Start at q=3 (marginal), test each kick regime
KICK 1.0     ; sub-critical: q=3 → q=2  (barely nudged)
READ
KICK 2.2     ; erase-regime: q=2 → q=1  (funnel one step)
READ
KICK 3.0     ; beyond table: q=1 → q=0  (extrapolated stronger erase)
READ
HLT
"""

EXAMPLE_WRITE_ERASE_CYCLE = """\
; cycle.asm — WRITE/ERASE cycle: a two-instruction toggle.
; Demonstrates the simplest nontrivial BQSM program.
; WRITE pushes up, ERASE pulls down — the basic computation primitives.
RADIX QUAD
; Start at q=0
WRITE        ; q=0 → q=1
ASSERT 1     ; verify we incremented
ERASE        ; q=1 → q=0  (funnel absorbs it back)
ASSERT 0     ; verify we returned to ground
HLT
"""


# ═══════════════════════════════════════════════════════════════════════════
# Built-in example catalog
# ═══════════════════════════════════════════════════════════════════════════

BUILTIN_EXAMPLES: dict[str, tuple[str, str, int]] = {
    "increment":   (EXAMPLE_INCREMENT,   "WRITE increment: q=0→1→2", 0),
    "funnel":      (EXAMPLE_ERASE_FUNNEL, "ERASE funnel: q=2→1→0", 2),
    "reset":       (EXAMPLE_RESET,       "RESET: q=3→0 (guaranteed)", 3),
    "overflow":    (EXAMPLE_OVERFLOW,    "WRITE overflow: q=0→1→2→3→DEAD", 0),
    "ternary":     (EXAMPLE_TERNARY,     "Ternary mode: q=0→1→2→DEAD", 0),
    "binary":      (EXAMPLE_BINARY,      "Binary mode: single-bit register", 0),
    "kicks":       (EXAMPLE_KICKS,       "KICK amplitudes: sub-critical→erase", 2),
    "loop":        (EXAMPLE_LOOP,        "WRITE/ERASE loop (Ctrl-C to stop)", 0),
    "conditional": (EXAMPLE_CONDITIONAL, "JEQ/JNE branching on state", 2),
    "kicksweep":   (EXAMPLE_KICK_STRENGTH_SWEEP, "KICK amplitude sweep", 3),
    "cycle":       (EXAMPLE_WRITE_ERASE_CYCLE, "WRITE/ERASE cycle with ASSERT", 0),
}


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def print_header():
    print("=" * 70)
    print("  BQSM Assembler & Programming Model  —  Fork B")
    print("  Basin-Quotient State Machine  ·  emerging.systems")
    print("=" * 70)


def cmd_assemble(args: list[str]) -> int:
    """Assemble a .asm file and print the disassembly."""
    if len(args) < 2:
        print("usage: bqsm_asm.py asm <file.asm> [init_q]")
        return 1
    path = args[1]
    init_q = int(args[2]) if len(args) > 2 else 0

    try:
        prog = assemble_file(path)
    except AsmError as e:
        print(f"Assembly error: {e}", file=sys.stderr)
        return 1

    print(prog.disassemble())
    print()
    print(prog.summary())
    print()

    if not prog.instructions:
        print("(empty program)")
        return 0

    vm = VM(prog, init_q=init_q)
    result = vm.run()
    print(vm.trace())
    return 0 if result.is_ok else 1


def cmd_examples(_args: list[str]) -> int:
    """List all built-in example programs."""
    print("Built-in examples:")
    print()
    for name, (_, desc, init_q) in BUILTIN_EXAMPLES.items():
        print(f"  {name:<14} init_q={init_q}  {desc}")
    print()
    print("Run with: bqsm_asm.py run <name>")
    return 0


def cmd_run(args: list[str]) -> int:
    """Run a built-in example by name."""
    if len(args) < 2:
        print("usage: bqsm_asm.py run <example_name>")
        print("       bqsm_asm.py examples  (to list names)")
        return 1

    name = args[1]
    entry = BUILTIN_EXAMPLES.get(name)
    if entry is None:
        print(f"Unknown example: '{name}'")
        print(f"Available: {', '.join(BUILTIN_EXAMPLES)}")
        return 1

    src, desc, init_q = entry

    try:
        prog = assemble(src)
    except AsmError as e:
        print(f"Internal assembly error: {e}", file=sys.stderr)
        return 1

    print(prog.disassemble())
    print()
    print(prog.summary())
    print()
    print(f"Running with init_q={init_q}...")
    print()

    vm = VM(prog, init_q=init_q)
    result = vm.run()
    print(vm.trace())
    return 0 if result.is_ok else 1


def cmd_demo(_args: list[str]) -> int:
    """Run a demo suite: all examples with output."""
    print_header()
    print()

    tfn = TransitionFn()
    print(tfn.display())
    print()

    print("Energy Model:")
    print("  " + "─" * 45)
    for q in range(Q_MAX + 1):
        energy = ENERGY_TABLE[q]
        label = ""
        if q == 0:
            label = "  (ground)"
        elif q == Q_MAX:
            label = "  (marginal)"
        print(f"  q={q}: V = {energy:+.4f}{label}")
    print()

    for name, (src, desc, init_q) in BUILTIN_EXAMPLES.items():
        print("─" * 70)
        print(f"  EXAMPLE: {name}  —  {desc}")
        print("─" * 70)

        try:
            prog = assemble(src)
        except AsmError as e:
            print(f"  Assembly error: {e}")
            continue

        print(prog.disassemble())
        print()
        print(f"  Running with init_q={init_q}...")
        print()

        # Loop example runs forever — cap it
        max_steps = 100 if name == "loop" else 10_000
        vm = VM(prog, init_q=init_q)
        result = vm.run(max_steps=max_steps)
        print(vm.trace())
        print()

    print("─" * 70)
    print("  Demo complete.")
    print("─" * 70)
    return 0


def cmd_vm(args: list[str]) -> int:
    """Low-level VM operations on an assembled program."""
    if len(args) < 2:
        print("usage: bqsm_asm.py vm <file.asm> [init_q]")
        return 1

    path = args[1]
    init_q = int(args[2]) if len(args) > 2 else 0

    try:
        prog = assemble_file(path)
    except AsmError as e:
        print(f"Assembly error: {e}", file=sys.stderr)
        return 1

    vm = VM(prog, init_q=init_q)
    result = vm.run()
    print(vm.trace())
    return 0 if result.is_ok else 1


def cmd_check(args: list[str]) -> int:
    """Check an assembly file for errors without running."""
    if len(args) < 2:
        print("usage: bqsm_asm.py check <file.asm>")
        return 1

    path = args[1]
    try:
        prog = assemble_file(path)
    except AsmError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    print(f"OK: {prog.summary()}")
    return 0


# ── Main dispatcher ────────────────────────────────────────────────────────

USAGE = """\
usage: bqsm_asm.py <command> [args...]

commands:
  asm <file> [init_q]   Assemble a .asm file, disassemble, and run
  check <file>          Check a .asm file for errors (no execution)
  run <example>         Run a built-in example program
  examples              List all built-in example programs
  vm <file> [init_q]    Run an assembled program in the VM
  demo                  Run the full demo suite (all examples)

Assembly language reference:
  WRITE                 quantized increment (+1); |q|=radix_max → DEAD
  ERASE                 one-way funnel toward q=0
  RESET                 strong erase to q=0 (guaranteed)
  KICK <amplitude>      scalar kick (1.0=sub-critical, 2.2=erase-regime)
  READ                  topological readout (free, no energy cost)
  NOP                   no operation
  HLT                   halt execution
  JMP <label>           unconditional jump
  JEQ <label> <q>       jump if current state equals q
  JNE <label> <q>       jump if current state not equal to q
  RADIX <mode>          set mode: BINARY(1), TERNARY(2), or QUAD(3)
  ASSERT <q>            assert current state equals q (runtime check)
  :label                define a label (jump target)
  ; comment             line comment (to end of line)

Physics: N=16 oscillators, Q_MAX=3, pristine Kuramoto ring.
The transition table is measured, not designed — see README.

Built with stdlib only.  Structural pattern matching + dataclasses.
"""


def main() -> int:
    if len(sys.argv) < 2:
        print(USAGE)
        return 0

    cmd = sys.argv[1]
    rest = sys.argv[1:]  # include command name for sub-dispatch

    match cmd:
        case "asm" | "assemble":
            return cmd_assemble(rest)
        case "check":
            return cmd_check(rest)
        case "run":
            return cmd_run(rest)
        case "examples" | "list":
            return cmd_examples(rest)
        case "vm":
            return cmd_vm(rest)
        case "demo":
            return cmd_demo(rest)
        case "-h" | "--help" | "help":
            print(USAGE)
            return 0
        case _:
            print(f"Unknown command: '{cmd}'", file=sys.stderr)
            print(f"Try: bqsm_asm.py --help", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main())