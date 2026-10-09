//! Built-in Panda-8 programs (assembly source plus assembled images).

use crate::asm::assemble;

/// A named built-in program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Program {
    /// Short identifier.
    pub name: &'static str,
    /// What it does.
    pub description: &'static str,
    /// Assembly source (see [`crate::asm`]).
    pub source: &'static str,
    /// `true` if the program ends with `HALT`.
    pub halts: bool,
}

impl Program {
    /// Assemble the source (built-in programs always assemble).
    pub fn image(&self) -> Vec<u8> {
        assemble(self.source).unwrap_or_else(|e| panic!("built-in program '{}': {e}", self.name))
    }
}

/// Fibonacci numbers `0, 1, 1, 2, …, 233` — every one that fits in 8 bits — then `HALT`.
pub const FIBONACCI: Program = Program {
    name: "fibonacci",
    description: "outputs Fibonacci numbers until the next one overflows 8 bits, then halts",
    halts: true,
    source: "\
; r0 = a, r1 = b
        LDI  r0, 0
        LDI  r1, 1
loop:   OUT  r0
        MOV  r2, r0
        ADD  r2, r1      ; r2 = a + b, C on overflow
        JC   last
        MOV  r0, r1
        MOV  r1, r2
        JMP  loop
last:   OUT  r1          ; b still fits: emit it
        HALT
",
};

/// Counts down `10, 9, …, 1, 0`, then `HALT`.
pub const COUNTDOWN: Program = Program {
    name: "countdown",
    description: "counts down from 10 to 0 on the output port, then halts",
    halts: true,
    source: "\
        .equ START, 10
        LDI  r0, START
loop:   OUT  r0
        DEC  r0
        JNZ  loop
        OUT  r0          ; 0
        HALT
",
};

/// Multiplies by repeated addition: outputs the running products `13·k` for `k = 1..=11`
/// (the last is `143`), then the products `200·2 mod 256 = 144` with carry detection, then `HALT`.
pub const MULTIPLY: Program = Program {
    name: "multiply",
    description: "multiplies 13 x 11 by repeated addition (emitting partial products), then 200 x 2 with overflow flag",
    halts: true,
    source: "\
; r0 = multiplicand, r1 = multiplier, r2 = product
        LDI  r0, 13
        LDI  r1, 11
        CLR  r2
mul:    ADD  r2, r0
        OUT  r2          ; partial product
        DEC  r1
        JNZ  mul
; second product, with overflow detection into r3
        LDI  r0, 200
        LDI  r1, 2
        CLR  r2
        CLR  r3
mul2:   ADD  r2, r0
        JNC  nc
        INC  r3          ; count carries
nc:     DEC  r1
        JNZ  mul2
        OUT  r2          ; 400 mod 256 = 144
        OUT  r3          ; 1 overflow
; exercise the remaining ALU ops: AND, OR, XOR, SUB, CMP, CMPI
        LDI  r0, 0xF0
        LDI  r1, 0x3C
        MOV  r2, r0
        AND  r2, r1
        OUT  r2          ; 0x30
        MOV  r2, r0
        OR   r2, r1
        OUT  r2          ; 0xFC
        MOV  r2, r0
        XOR  r2, r1
        OUT  r2          ; 0xCC
        MOV  r2, r1
        SUB  r2, r0      ; 0x3C - 0xF0 borrows
        JC   bad
        OUT  r2          ; 0x4C
        CMP  r0, r1
        JNC  bad         ; 0xF0 >= 0x3C: no borrow
        CMPI r0, 0xF0
        JNZ  bad
        LDI  r3, 'K'
        OUT  r3
        HALT
bad:    LDI  r3, 0xEE
        OUT  r3
        HALT
",
};

/// Never halts: an outer counter incremented every 256 inner-loop iterations (2 instructions × 2 cycles × 256 = 1024 cycles, plus 8),
/// output after each increment (one `OUT` every 1032 cycles). Used for benchmarking.
pub const BENCH_LOOP: Program = Program {
    name: "bench_loop",
    description: "endless nested loop; outputs an incrementing counter every 256 inner iterations",
    halts: false,
    source: "\
        CLR  r0
outer:  CLR  r1
inner:  INC  r1
        JNZ  inner       ; 256 iterations
        INC  r0
        OUT  r0
        JMP  outer
",
};

/// Every built-in program.
pub const ALL: [Program; 4] = [FIBONACCI, COUNTDOWN, MULTIPLY, BENCH_LOOP];

/// Look up a built-in program by name.
pub fn by_name(name: &str) -> Option<Program> {
    ALL.iter().copied().find(|p| p.name == name)
}
