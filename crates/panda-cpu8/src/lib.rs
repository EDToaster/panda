//! Panda-8 — the gate-only 8-bit reference CPU benchmark (SPEC §12).
//!
//! * [`isa`] — the instruction set (2-byte instructions, 4 registers, Z/C flags, `OUT` port).
//! * [`asm`] — a two-pass assembler with labels.
//! * [`emu`] — a cycle-level behavioral emulator (the reference for gate-level runs).
//! * [`generate`] — the gate-level design generator ([`panda8`]) with the SPEC §12 variants.
//! * [`programs`] — built-in programs (Fibonacci, countdown, multiply, benchmark loop).
//! * [`sim`] — run a generated design in the timing engine and sample its state per cycle.
//!
//! ```
//! use panda_cpu8::{asm::assemble, emu::Emulator};
//!
//! let program = assemble("LDI r0, 3\nloop: OUT r0\nDEC r0\nJNZ loop\nHALT").unwrap();
//! assert_eq!(Emulator::new(&program).run(1_000), vec![3, 2, 1]);
//! ```

pub mod asm;
pub mod emu;
pub mod generate;
pub mod isa;
pub mod programs;
pub mod sim;

pub use generate::{Options, RomStyle, Variant, panda8, panda8_with};
