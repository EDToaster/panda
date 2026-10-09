//! Panda-8 (SPEC §12): every variant and ROM style in FastRef vs Timing mode at the STA minimum
//! period, compared on every net at every phase for ≥ 2000 cycles; FastRef's `OUT` sequence must
//! match the behavioral emulator.

use panda_cpu8::{RomStyle, Variant, programs};
use panda_difftest::cpu8::{Cpu8Config, run_cpu8};

const ROMS: [RomStyle; 3] = [
    RomStyle::SumOfProducts,
    RomStyle::MuxTree,
    RomStyle::Decoder,
];

fn check(program: &programs::Program, cfg: &Cpu8Config) {
    let image = program.image();
    for variant in Variant::ALL {
        for rom in ROMS {
            let r = run_cpu8(variant, rom, &image, cfg).expect("run_cpu8");
            println!(
                "{} {variant:?} {rom:?}: min period {} period {} compared {} outs {:?}",
                program.name, r.min_period, r.period, r.diff.compared, r.outs
            );
            assert!(
                r.sta_errors.is_empty(),
                "{variant:?} {rom:?}: STA errors {:?}",
                r.sta_errors
            );
            assert!(
                r.diff.equal(),
                "{variant:?} {rom:?}: FastRef != Timing: {:?}",
                r.diff.first().map(ToString::to_string)
            );
            assert!(
                r.diff.compared >= 2 * cfg.cycles as usize,
                "{variant:?} {rom:?}: only {} phases",
                r.diff.compared
            );
            assert_eq!(r.outs, r.golden, "{variant:?} {rom:?}: OUT sequence");
            assert!(!r.golden.is_empty(), "program produced no output");
        }
    }
}

#[test]
fn panda8_every_variant_and_rom_fastref_equals_timing_2000_cycles() {
    check(&programs::FIBONACCI, &Cpu8Config::default());
}

#[test]
#[ignore = "long: every program, plus ideal-block Timing mode"]
fn panda8_every_program_and_ideal_blocks() {
    for p in programs::ALL {
        check(
            &p,
            &Cpu8Config {
                cycles: 3000,
                ..Cpu8Config::default()
            },
        );
    }
    // Ideal blocks: STA analyzes for ideal blocks too (each block output changes its full latency
    // after an input change), so the STA minimum period is used directly.
    for p in [programs::FIBONACCI, programs::MULTIPLY] {
        let image = p.image();
        for variant in Variant::ALL {
            for rom in ROMS {
                let r = run_cpu8(
                    variant,
                    rom,
                    &image,
                    &Cpu8Config {
                        cycles: 2000,
                        period: None,
                        ideal_blocks: true,
                    },
                )
                .unwrap();
                println!(
                    "{} {variant:?} {rom:?} ideal blocks: min period {} compared {}",
                    p.name, r.min_period, r.diff.compared
                );
                assert!(
                    r.ok(),
                    "{} {variant:?} {rom:?} ideal blocks @ {}: STA {:?}, {:?}",
                    p.name,
                    r.period,
                    r.sta_errors,
                    r.diff.first().map(ToString::to_string)
                );
            }
        }
    }
}
