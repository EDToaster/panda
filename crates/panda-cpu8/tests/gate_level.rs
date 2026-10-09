//! Gate-level Panda-8 in the timing engine vs. the behavioral emulator, cycle by cycle.
//!
//! Clock period: [`DEFAULT_PERIOD`] = 128 ticks; the measured worst settle time is 47 ticks after a
//! rising edge and 6 after a falling edge (`settle_time_fits_default_period` re-measures it).

mod common;

use common::{Engine, Rng, lockstep, lockstep_with};
use panda_cpu8::emu::Emulator;
use panda_cpu8::generate::DEFAULT_PERIOD;
use panda_cpu8::asm::assemble;
use panda_cpu8::programs::{self, Program};
use panda_cpu8::sim::GateCpu;
use panda_cpu8::{Options, RomStyle, Variant, panda8, panda8_with};

const FIB: [u8; 14] = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233];

fn expected_outputs(p: &Program) -> Vec<u8> {
    match p.name {
        "fibonacci" => FIB.to_vec(),
        "countdown" => (0..=10).rev().collect(),
        "multiply" => {
            let mut v: Vec<u8> = (1..=11).map(|k| 13 * k).collect();
            v.extend([144, 1, 0x30, 0xFC, 0xCC, 0x4C, b'K']);
            v
        }
        "bench_loop" => vec![1, 2, 3],
        other => panic!("no expectation for {other}"),
    }
}

/// Lockstep-check a built-in program; halting programs run to HALT (+10 cycles), the benchmark
/// loop for 3200 cycles (three outer iterations of 1032 cycles).
fn check(p: Program, opts: Options) {
    let max = if p.halts { 10_000 } else { 3_200 };
    let r = lockstep(&p.image(), &opts, max, 10);
    assert_eq!(r.halted, p.halts, "{}: halted", p.name);
    assert_eq!(r.gate_outs, r.emu_outs, "{}: OUT sequence", p.name);
    assert_eq!(r.emu_outs, expected_outputs(&p), "{}: expected outputs", p.name);
}

macro_rules! program_variant_tests {
    ($($name:ident: $prog:ident, $variant:ident;)*) => {$(
        #[test]
        fn $name() {
            check(programs::$prog, Options::new(Variant::$variant));
        }
    )*};
}

program_variant_tests! {
    fibonacci_hierarchical: FIBONACCI, Hierarchical;
    fibonacci_flat: FIBONACCI, Flat;
    fibonacci_gated: FIBONACCI, GatedClockRegfile;
    countdown_hierarchical: COUNTDOWN, Hierarchical;
    countdown_flat: COUNTDOWN, Flat;
    countdown_gated: COUNTDOWN, GatedClockRegfile;
    multiply_hierarchical: MULTIPLY, Hierarchical;
    multiply_flat: MULTIPLY, Flat;
    multiply_gated: MULTIPLY, GatedClockRegfile;
    bench_loop_hierarchical: BENCH_LOOP, Hierarchical;
    bench_loop_flat: BENCH_LOOP, Flat;
    bench_loop_gated: BENCH_LOOP, GatedClockRegfile;
}

#[test]
fn mux_tree_rom_all_programs_all_variants() {
    for v in Variant::ALL {
        for p in programs::ALL {
            check(p, Options::new(v).with_rom(RomStyle::MuxTree));
        }
    }
}

#[test]
fn decoder_rom_all_programs_all_variants() {
    for v in Variant::ALL {
        for p in programs::ALL {
            check(p, Options::new(v).with_rom(RomStyle::Decoder));
        }
    }
}

#[test]
fn other_clock_periods_behave_identically() {
    for period in [96, 256] {
        check(programs::MULTIPLY, Options::new(Variant::Hierarchical).with_period(period));
        check(programs::FIBONACCI, Options::new(Variant::GatedClockRegfile).with_period(period));
    }
}

#[test]
fn longer_reset_behaves_identically() {
    let mut o = Options::new(Variant::GatedClockRegfile);
    o.reset_cycles = 5;
    check(programs::COUNTDOWN, o);
}

/// The harness must be able to see a timing failure: at a period far below the settle time the
/// gate-level outputs differ from the emulator (or are not fully known).
#[test]
fn too_short_period_is_detected() {
    let p = programs::COUNTDOWN.image();
    let d = panda8_with(&p, &Options::new(Variant::Hierarchical).with_period(16));
    let mut gate = GateCpu::new(&d).unwrap();
    let want = Emulator::new(&p).run(1_000);
    let got = gate.run_outputs(200);
    assert_ne!(got.ok(), Some(want));
}

#[test]
fn settle_time_fits_default_period() {
    let mut worst = (0, 0);
    for rom in [RomStyle::SumOfProducts, RomStyle::MuxTree, RomStyle::Decoder] {
        for v in Variant::ALL {
            for p in programs::ALL {
                // Measure with a generous period so rising- and falling-edge activity never overlap.
                let d = panda8_with(&p.image(), &Options::new(v).with_rom(rom).with_period(512));
                let mut gate = GateCpu::new(&d).unwrap();
                let st = gate.measure_settle(400);
                worst.0 = worst.0.max(st.after_rise);
                worst.1 = worst.1.max(st.after_fall);
            }
        }
    }
    println!("worst settle: {} ticks after rising edge, {} after falling edge", worst.0, worst.1);
    assert!(worst.0 > 0 && worst.1 > 0, "measurement saw no activity");
    // At least 2x margin before the next rising edge, 4x within the low half period.
    assert!(worst.0 * 2 <= u64::from(DEFAULT_PERIOD), "rising-edge settle {} too slow", worst.0);
    assert!(worst.1 * 4 <= u64::from(DEFAULT_PERIOD / 2), "falling-edge settle {} too slow", worst.1);
}

#[test]
fn halted_state_is_frozen() {
    let p = programs::COUNTDOWN.image();
    for v in Variant::ALL {
        let r = lockstep(&p, &Options::new(v), 10_000, 50);
        assert!(r.halted);
    }
}

#[test]
fn every_instruction_and_condition() {
    // Hits every opcode, every condition both taken and not taken, and every register as rd/rs.
    let src = "
        NOP
        LDI  r0, 0x81
        LDI  r1, 0x7F
        LDI  r2, 0x01
        MOV  r3, r0
        ADD  r3, r1       ; 0x00, C=1 Z=1
        JNZ  bad          ; not taken
        JNC  bad          ; not taken
        OUT  r3           ; 0x00
        SUB  r2, r1       ; 0x01 - 0x7F borrows: 0x82, C=0 Z=0
        JZ   bad          ; not taken
        JC   bad          ; not taken
        OUT  r2           ; 0x82
        AND  r2, r0       ; 0x80
        OR   r2, r1       ; 0xFF
        XOR  r2, r0       ; 0x7E
        OUT  r2           ; 0x7E
        ADDI r2, 0x82     ; 0x00, C=1
        JC   t1           ; taken
        JMP  bad
t1:     SUBI r0, 0x81     ; 0, Z=1, C=1
        JZ   t2           ; taken
        JMP  bad
t2:     CMP  r1, r0       ; 0x7F - 0: C=1, Z=0
        JNZ  t3           ; taken
        JMP  bad
t3:     CMPI r1, 0x80     ; 0x7F - 0x80 borrows: C=0
        JNC  t4           ; taken
        JMP  bad
t4:     XOR  r3, r3       ; Z=1, C=0
        JNC  t5           ; taken
        JMP  bad
t5:     OUT  r1           ; 0x7F
        HALT
bad:    LDI  r3, 0xEE
        OUT  r3
        HALT
    ";
    let program = assemble(src).unwrap();
    let mut emu = Emulator::new(&program);
    let outs = emu.run(10_000);
    assert_eq!(outs, vec![0x00, 0x82, 0x7E, 0x7F]);
    assert!(emu.state().halted);
    for v in Variant::ALL {
        let r = lockstep(&program, &Options::new(v), 10_000, 4);
        assert_eq!(r.gate_outs, outs, "{v:?}");
    }
}

/// The same programs with the timing engine's default options (`p2_limit = 4`): the state must
/// still match the emulator exactly; only `Metastable` warnings from combinational glitch trains
/// are tolerated (and reported).
#[test]
fn default_engine_options_match_emulator() {
    for v in Variant::ALL {
        for p in programs::ALL {
            let max = if p.halts { 10_000 } else { 3_200 };
            let r = lockstep_with(&p.image(), &Options::new(v), max, 10, Engine::Default);
            assert_eq!(r.gate_outs, expected_outputs(&p), "{} {v:?}", p.name);
            println!("{v:?} {}: {} cycles, {} Metastable warnings", p.name, r.cycles, r.metastable);
        }
    }
}

/// Random instruction streams (random opcodes, registers, immediates and in-range jump targets),
/// compared state-for-state against the emulator.
#[test]
fn random_programs_match_emulator() {
    let mut rng = Rng(0x5eed_cafe_f00d_1234);
    for case in 0..12 {
        let len = 8 + rng.below(56) as usize; // instructions
        let mut program = Vec::with_capacity(len * 2);
        for _ in 0..len {
            let op = rng.below(16) as u8;
            // Keep HALT rare so programs run for a while.
            let op = if op == 0xF && rng.below(4) != 0 { 0x8 } else { op };
            let fields = rng.below(16) as u8;
            let operand = match op {
                0xD | 0xE => (rng.below(len as u64) * 2) as u8, // instruction-aligned target
                _ => rng.next() as u8,
            };
            program.push(op << 4 | fields);
            program.push(operand);
        }
        let v = Variant::ALL[case % 3];
        let rom = [RomStyle::SumOfProducts, RomStyle::Decoder, RomStyle::MuxTree][case / 3 % 3];
        let r = lockstep(&program, &Options::new(v).with_rom(rom), 600, 2);
        assert_eq!(r.gate_outs, r.emu_outs, "case {case}");
    }
}

#[test]
fn default_constructor_matches_options() {
    let p = programs::FIBONACCI.image();
    for v in Variant::ALL {
        assert_eq!(panda8(&p, v), panda8_with(&p, &Options::new(v)));
    }
}
