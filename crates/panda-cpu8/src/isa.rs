//! The Panda-8 instruction set.
//!
//! # Programmer's model
//!
//! * Four 8-bit general registers `r0..r3`.
//! * An 8-bit program counter `PC` addressing a 256-byte program ROM (wraps modulo 256).
//! * Two flags: `Z` (zero) and `C` (carry).
//! * An 8-bit output port `OUT` with a strobe pin `OUT_STROBE` that is high for exactly one clock
//!   cycle after every `OUT` instruction.
//! * A `HALTED` status pin.
//!
//! There is no data memory; programs compute in registers.
//!
//! # Encoding
//!
//! Every instruction is **two bytes**: an opcode byte followed by an operand byte.
//!
//! ```text
//!   byte 0 (at PC):    7 6 5 4 | 3 2 | 1 0        byte 1 (at PC+1):  7 ........ 0
//!                      opcode  |  rd |  rs                           imm8 / addr8
//! ```
//!
//! Fields that an instruction does not use are ignored by the hardware (the assembler emits them
//! as 0). For `Jcc` the `rd` field holds the condition code (see [`Cond`]).
//!
//! | op | mnemonic        | effect                                  | flags        |
//! |----|-----------------|-----------------------------------------|--------------|
//! | 0  | `NOP`           | –                                       | –            |
//! | 1  | `LDI rd, imm`   | `rd ← imm`                              | –            |
//! | 2  | `MOV rd, rs`    | `rd ← rs`                               | –            |
//! | 3  | `ADD rd, rs`    | `rd ← rd + rs`                          | Z, C = carry |
//! | 4  | `SUB rd, rs`    | `rd ← rd - rs`                          | Z, C = ¬borrow |
//! | 5  | `AND rd, rs`    | `rd ← rd & rs`                          | Z, C ← 0     |
//! | 6  | `OR rd, rs`     | `rd ← rd \| rs`                         | Z, C ← 0     |
//! | 7  | `XOR rd, rs`    | `rd ← rd ^ rs`                          | Z, C ← 0     |
//! | 8  | `ADDI rd, imm`  | `rd ← rd + imm`                         | Z, C = carry |
//! | 9  | `SUBI rd, imm`  | `rd ← rd - imm`                         | Z, C = ¬borrow |
//! | A  | `CMP rd, rs`    | flags of `rd - rs`                      | Z, C = ¬borrow |
//! | B  | `CMPI rd, imm`  | flags of `rd - imm`                     | Z, C = ¬borrow |
//! | C  | `OUT rs`        | `OUT ← rs`, pulse `OUT_STROBE`          | –            |
//! | D  | `JMP addr`      | `PC ← addr`                             | –            |
//! | E  | `Jcc addr`      | if `cc` then `PC ← addr` (`JZ JNZ JC JNC`) | –         |
//! | F  | `HALT`          | stop (until reset)                      | –            |
//!
//! Subtraction is computed as `rd + ¬x + 1`, so `C` is the adder's carry out: `C = 1` means
//! *no borrow* (`rd ≥ x` unsigned), as on ARM/6502.
//!
//! # Timing
//!
//! Every instruction takes exactly two clock cycles:
//!
//! 1. **Fetch** (`PHASE = 0`): `IR ← ROM[PC]`, `PC ← PC + 1`.
//! 2. **Execute** (`PHASE = 1`): the operand is `ROM[PC]`; registers, flags and `OUT` are written;
//!    `PC ← taken ? operand : PC + 1`.
//!
//! `HALT` freezes all architectural state (the final `PC` points past the `HALT`).

use std::fmt;

/// A general register `r0..r3`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Reg {
    /// Register 0.
    R0 = 0,
    /// Register 1.
    R1 = 1,
    /// Register 2.
    R2 = 2,
    /// Register 3.
    R3 = 3,
}

impl Reg {
    /// All registers in index order.
    pub const ALL: [Reg; 4] = [Reg::R0, Reg::R1, Reg::R2, Reg::R3];

    /// Register with index `i & 3`.
    pub fn from_bits(i: u8) -> Reg {
        Reg::ALL[usize::from(i & 3)]
    }

    /// Register index (0..=3).
    pub fn index(self) -> usize {
        self as usize
    }
}

impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "r{}", *self as u8)
    }
}

/// Branch condition of a `Jcc` instruction (stored in the `rd` field).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cond {
    /// Jump if `Z = 1`.
    Z = 0,
    /// Jump if `Z = 0`.
    Nz = 1,
    /// Jump if `C = 1`.
    C = 2,
    /// Jump if `C = 0`.
    Nc = 3,
}

impl Cond {
    /// Condition with code `i & 3`.
    pub fn from_bits(i: u8) -> Cond {
        [Cond::Z, Cond::Nz, Cond::C, Cond::Nc][usize::from(i & 3)]
    }

    /// Evaluate the condition on the flags.
    pub fn holds(self, z: bool, c: bool) -> bool {
        match self {
            Cond::Z => z,
            Cond::Nz => !z,
            Cond::C => c,
            Cond::Nc => !c,
        }
    }

    /// Assembler mnemonic (`JZ`, `JNZ`, `JC`, `JNC`).
    pub fn mnemonic(self) -> &'static str {
        match self {
            Cond::Z => "JZ",
            Cond::Nz => "JNZ",
            Cond::C => "JC",
            Cond::Nc => "JNC",
        }
    }
}

/// Opcode numbers (the high nibble of the opcode byte).
pub mod opcode {
    /// `NOP`
    pub const NOP: u8 = 0x0;
    /// `LDI rd, imm`
    pub const LDI: u8 = 0x1;
    /// `MOV rd, rs`
    pub const MOV: u8 = 0x2;
    /// `ADD rd, rs`
    pub const ADD: u8 = 0x3;
    /// `SUB rd, rs`
    pub const SUB: u8 = 0x4;
    /// `AND rd, rs`
    pub const AND: u8 = 0x5;
    /// `OR rd, rs`
    pub const OR: u8 = 0x6;
    /// `XOR rd, rs`
    pub const XOR: u8 = 0x7;
    /// `ADDI rd, imm`
    pub const ADDI: u8 = 0x8;
    /// `SUBI rd, imm`
    pub const SUBI: u8 = 0x9;
    /// `CMP rd, rs`
    pub const CMP: u8 = 0xA;
    /// `CMPI rd, imm`
    pub const CMPI: u8 = 0xB;
    /// `OUT rs`
    pub const OUT: u8 = 0xC;
    /// `JMP addr`
    pub const JMP: u8 = 0xD;
    /// `Jcc addr`
    pub const JCC: u8 = 0xE;
    /// `HALT`
    pub const HALT: u8 = 0xF;
}

/// A decoded instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Instr {
    /// No operation.
    Nop,
    /// `rd ← imm`.
    Ldi(Reg, u8),
    /// `rd ← rs`.
    Mov(Reg, Reg),
    /// `rd ← rd + rs`.
    Add(Reg, Reg),
    /// `rd ← rd - rs`.
    Sub(Reg, Reg),
    /// `rd ← rd & rs`.
    And(Reg, Reg),
    /// `rd ← rd | rs`.
    Or(Reg, Reg),
    /// `rd ← rd ^ rs`.
    Xor(Reg, Reg),
    /// `rd ← rd + imm`.
    Addi(Reg, u8),
    /// `rd ← rd - imm`.
    Subi(Reg, u8),
    /// Flags of `rd - rs`.
    Cmp(Reg, Reg),
    /// Flags of `rd - imm`.
    Cmpi(Reg, u8),
    /// `OUT ← rs`.
    Out(Reg),
    /// `PC ← addr`.
    Jmp(u8),
    /// Conditional jump.
    Jcc(Cond, u8),
    /// Stop.
    Halt,
}

impl Instr {
    /// Encode as `[opcode byte, operand byte]`; unused fields are 0.
    pub fn encode(self) -> [u8; 2] {
        use opcode::*;
        let op = |o: u8, rd: u8, rs: u8, imm: u8| [o << 4 | (rd & 3) << 2 | (rs & 3), imm];
        match self {
            Instr::Nop => op(NOP, 0, 0, 0),
            Instr::Ldi(d, i) => op(LDI, d as u8, 0, i),
            Instr::Mov(d, s) => op(MOV, d as u8, s as u8, 0),
            Instr::Add(d, s) => op(ADD, d as u8, s as u8, 0),
            Instr::Sub(d, s) => op(SUB, d as u8, s as u8, 0),
            Instr::And(d, s) => op(AND, d as u8, s as u8, 0),
            Instr::Or(d, s) => op(OR, d as u8, s as u8, 0),
            Instr::Xor(d, s) => op(XOR, d as u8, s as u8, 0),
            Instr::Addi(d, i) => op(ADDI, d as u8, 0, i),
            Instr::Subi(d, i) => op(SUBI, d as u8, 0, i),
            Instr::Cmp(d, s) => op(CMP, d as u8, s as u8, 0),
            Instr::Cmpi(d, i) => op(CMPI, d as u8, 0, i),
            Instr::Out(s) => op(OUT, 0, s as u8, 0),
            Instr::Jmp(a) => op(JMP, 0, 0, a),
            Instr::Jcc(c, a) => op(JCC, c as u8, 0, a),
            Instr::Halt => op(HALT, 0, 0, 0),
        }
    }

    /// Decode an opcode byte and its operand byte. Every byte pair decodes (unused fields are
    /// ignored, exactly as the hardware does).
    pub fn decode(op_byte: u8, operand: u8) -> Instr {
        use opcode::*;
        let rd = Reg::from_bits(op_byte >> 2);
        let rs = Reg::from_bits(op_byte);
        match op_byte >> 4 {
            NOP => Instr::Nop,
            LDI => Instr::Ldi(rd, operand),
            MOV => Instr::Mov(rd, rs),
            ADD => Instr::Add(rd, rs),
            SUB => Instr::Sub(rd, rs),
            AND => Instr::And(rd, rs),
            OR => Instr::Or(rd, rs),
            XOR => Instr::Xor(rd, rs),
            ADDI => Instr::Addi(rd, operand),
            SUBI => Instr::Subi(rd, operand),
            CMP => Instr::Cmp(rd, rs),
            CMPI => Instr::Cmpi(rd, operand),
            OUT => Instr::Out(rs),
            JMP => Instr::Jmp(operand),
            JCC => Instr::Jcc(Cond::from_bits(op_byte >> 2), operand),
            _ => Instr::Halt,
        }
    }
}

impl fmt::Display for Instr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Instr::Nop => write!(f, "NOP"),
            Instr::Ldi(d, i) => write!(f, "LDI {d}, {i:#04x}"),
            Instr::Mov(d, s) => write!(f, "MOV {d}, {s}"),
            Instr::Add(d, s) => write!(f, "ADD {d}, {s}"),
            Instr::Sub(d, s) => write!(f, "SUB {d}, {s}"),
            Instr::And(d, s) => write!(f, "AND {d}, {s}"),
            Instr::Or(d, s) => write!(f, "OR {d}, {s}"),
            Instr::Xor(d, s) => write!(f, "XOR {d}, {s}"),
            Instr::Addi(d, i) => write!(f, "ADDI {d}, {i:#04x}"),
            Instr::Subi(d, i) => write!(f, "SUBI {d}, {i:#04x}"),
            Instr::Cmp(d, s) => write!(f, "CMP {d}, {s}"),
            Instr::Cmpi(d, i) => write!(f, "CMPI {d}, {i:#04x}"),
            Instr::Out(s) => write!(f, "OUT {s}"),
            Instr::Jmp(a) => write!(f, "JMP {a:#04x}"),
            Instr::Jcc(c, a) => write!(f, "{} {a:#04x}", c.mnemonic()),
            Instr::Halt => write!(f, "HALT"),
        }
    }
}

/// Disassemble a program image (two bytes per line, starting at address 0) into
/// `"addr: bytes  mnemonic"` lines. A trailing odd byte is shown as data.
pub fn disassemble(program: &[u8]) -> String {
    let mut out = String::new();
    for (i, pair) in program.chunks(2).enumerate() {
        let addr = i * 2;
        match *pair {
            [a, b] => out.push_str(&format!("{addr:02x}: {a:02x} {b:02x}  {}\n", Instr::decode(a, b))),
            [a] => out.push_str(&format!("{addr:02x}: {a:02x}     .byte {a:#04x}\n")),
            _ => unreachable!("chunks(2) yields 1 or 2 bytes"),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip_all_byte_pairs() {
        // Every canonical encoding re-encodes to itself after decoding.
        for op in 0..=255u8 {
            for operand in [0u8, 1, 0x7f, 0xff] {
                let i = Instr::decode(op, operand);
                assert_eq!(Instr::decode(i.encode()[0], i.encode()[1]), i);
            }
        }
    }

    #[test]
    fn field_layout() {
        assert_eq!(Instr::Add(Reg::R2, Reg::R3).encode(), [0x3B, 0]);
        assert_eq!(Instr::Ldi(Reg::R1, 0xAB).encode(), [0x14, 0xAB]);
        assert_eq!(Instr::Jcc(Cond::Nc, 0x10).encode(), [0xEC, 0x10]);
        assert_eq!(Instr::Out(Reg::R3).encode(), [0xC3, 0]);
        assert_eq!(Instr::Halt.encode(), [0xF0, 0]);
    }

    #[test]
    fn conditions() {
        assert!(Cond::Z.holds(true, false) && !Cond::Z.holds(false, true));
        assert!(Cond::Nz.holds(false, false) && !Cond::Nz.holds(true, false));
        assert!(Cond::C.holds(false, true) && !Cond::C.holds(true, false));
        assert!(Cond::Nc.holds(true, false) && !Cond::Nc.holds(false, true));
    }

    #[test]
    fn disassembly() {
        let text = disassemble(&[0x14, 0x05, 0xF0, 0x00, 0x42]);
        assert_eq!(text, "00: 14 05  LDI r1, 0x05\n02: f0 00  HALT\n04: 42     .byte 0x42\n");
    }
}
