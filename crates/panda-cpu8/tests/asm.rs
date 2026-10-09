//! Assembler and ISA integration tests.

use panda_cpu8::asm::{AsmError, assemble, assemble_with_symbols};
use panda_cpu8::emu::Emulator;
use panda_cpu8::isa::{Cond, Instr, Reg, disassemble};
use panda_cpu8::programs;

fn err(src: &str) -> AsmError {
    assemble(src).expect_err("should fail")
}

#[test]
fn encodes_every_mnemonic() {
    let src = "
        nop
        ldi r0, 1
        mov r1, r0
        add r2, r3
        sub r3, r2
        and r0, r1
        or  r1, r2
        xor r2, r3
        addi r3, 0x10
        subi r0, 2
        cmp r1, r2
        cmpi r2, 3
        out r3
        jmp 0
        jz 2
        jnz 4
        jc 6
        jnc 8
        halt
        inc r1
        dec r2
        clr r3
    ";
    use Instr::*;
    use Reg::*;
    let want = [
        Nop,
        Ldi(R0, 1),
        Mov(R1, R0),
        Add(R2, R3),
        Sub(R3, R2),
        And(R0, R1),
        Or(R1, R2),
        Xor(R2, R3),
        Addi(R3, 0x10),
        Subi(R0, 2),
        Cmp(R1, R2),
        Cmpi(R2, 3),
        Out(R3),
        Jmp(0),
        Jcc(Cond::Z, 2),
        Jcc(Cond::Nz, 4),
        Jcc(Cond::C, 6),
        Jcc(Cond::Nc, 8),
        Halt,
        Addi(R1, 1),
        Subi(R2, 1),
        Ldi(R3, 0),
    ];
    let bytes: Vec<u8> = want.iter().flat_map(|i| i.encode()).collect();
    assert_eq!(assemble(src).unwrap(), bytes);
}

#[test]
fn labels_directives_and_expressions() {
    let src = "
        .equ  K, 5          ; constant
start:  JMP   main          // forward reference
data:   .byte 1, 'A', -1, K+1
        .org  0x10
main:   LDI   r0, data      # label as a value
        LDI   r1, end-main
        LDI   r2, K - 2
loop: end: HALT             ; two labels on one line
    ";
    let (img, syms) = assemble_with_symbols(src).unwrap();
    assert_eq!(syms["start"], 0);
    assert_eq!(syms["data"], 2);
    assert_eq!(syms["main"], 0x10);
    assert_eq!(syms["end"], 0x16);
    assert_eq!(syms["loop"], 0x16);
    assert_eq!(syms["K"], 5);
    let mut want = vec![0xD0, 0x10, 1, b'A', 0xFF, 6];
    want.resize(0x10, 0);
    want.extend([0x10, 2, 0x14, 6, 0x18, 3, 0xF0, 0]);
    assert_eq!(img, want);
}

#[test]
fn case_insensitive_mnemonics_and_registers() {
    assert_eq!(assemble("LdI R2, 0X1f").unwrap(), assemble("ldi r2, 31").unwrap());
}

#[test]
fn errors_name_the_line() {
    let e = err("NOP\nNOP\nFROB r0");
    assert_eq!(e.line, 3);
    assert!(e.message.contains("unknown mnemonic"), "{e}");
    assert!(e.to_string().starts_with("line 3:"));

    assert!(err("JMP nowhere").message.contains("undefined symbol"));
    assert!(err("a: NOP\na: NOP").message.contains("defined twice"));
    assert!(err("LDI r4, 1").message.contains("expected a register"));
    assert!(err("LDI r0").message.contains("expects 2 operand"));
    assert!(err("ADD r0, 5").message.contains("expected a register"));
    assert!(err("LDI r0, 256").message.contains("does not fit"));
    assert!(err("LDI r0, r1").message.contains("register"));
    assert!(err(".org 4\nNOP\n.org 2").message.contains("backwards"));
    assert!(err(".bogus 1").message.contains("unknown directive"));
    assert!(err("r0: NOP").message.contains("invalid label"));
    assert!(err("LDI r0, 'ab'").message.contains("character literal"));
    assert!(err(".equ X, Y\n.equ Y, 1").message.contains("undefined symbol"), "no forward .equ references");
    let too_long = "NOP\n".repeat(129);
    assert!(err(&too_long).message.contains("256"));
    assert_eq!(assemble(&"NOP\n".repeat(128)).unwrap().len(), 256);
}

#[test]
fn builtin_programs_assemble_and_disassemble_round_trip() {
    for p in programs::ALL {
        let img = p.image();
        assert!(img.len() % 2 == 0 && !img.is_empty(), "{}", p.name);
        // Strip "addr: bytes  " from each disassembly line; the rest is valid assembly.
        let text: String = disassemble(&img).lines().map(|l| format!("{}\n", l.split_once("  ").unwrap().1)).collect();
        assert_eq!(assemble(&text).unwrap(), img, "{}:\n{text}", p.name);
        assert_eq!(programs::by_name(p.name), Some(p));
    }
    assert_eq!(programs::by_name("nope"), None);
}

#[test]
fn builtin_programs_on_the_emulator() {
    let fib = Emulator::new(&programs::FIBONACCI.image()).run(10_000);
    assert_eq!(fib, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]);
    let cd = Emulator::new(&programs::COUNTDOWN.image()).run(10_000);
    assert_eq!(cd, (0..=10).rev().collect::<Vec<u8>>());
    let mut e = Emulator::new(&programs::BENCH_LOOP.image());
    let outs = e.run(100_000);
    assert!(!e.state().halted);
    assert_eq!(outs, (1..=96).collect::<Vec<u8>>(), "one OUT per 1032 cycles");
}

#[test]
fn emulator_trace_and_reset() {
    let p = programs::COUNTDOWN.image();
    let mut e = Emulator::new(&p);
    let t = e.trace(4);
    assert_eq!(t.len(), 4);
    assert_eq!(t[1].regs[0], 10, "LDI executes at the second edge");
    assert_eq!(e.cycles(), 4);
    e.reset();
    assert_eq!(e.cycles(), 0);
    assert_eq!(*e.state(), panda_cpu8::emu::ArchState::reset());
    assert_eq!(&e.rom()[..p.len()], &p[..]);
}
