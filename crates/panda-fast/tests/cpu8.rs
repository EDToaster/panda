//! FastSim = FastRef on Panda-8 (every Variant × RomStyle) for 3200 clock cycles of the
//! benchmark loop, compared after every event; the OUT sequence also matches the emulator.

mod common;

use std::sync::Arc;

use common::{Coverage, check, pair};
use panda_compile::FastOptions;
use panda_cpu8::emu::Emulator;
use panda_cpu8::programs::{self, Program};
use panda_cpu8::{Options, RomStyle, Variant, panda8_with};
use panda_fast::CompileOptions;
use panda_netlist::flatten;

fn lockstep(p: &Program, variant: Variant, rom: RomStyle, cycles: u64) {
    let opts = Options::new(variant).with_rom(rom);
    let flat = Arc::new(flatten(&panda8_with(&p.image(), &opts)).expect("flatten"));
    let (mut fr, mut fs) = pair(&flat, &FastOptions::default(), &CompileOptions::default());
    let mut cov = Coverage::default();
    let ctx = format!("{} {variant:?} {rom:?}", p.name);
    check(
        &mut fr,
        &mut fs,
        &flat,
        &format!("{ctx} power-on"),
        &mut cov,
    );
    // Two events per cycle (rising + falling edge) plus the reset release.
    let reset_cycles = u64::from(opts.reset_cycles) + 2;
    let events = 2 * (cycles + reset_cycles) + 1;
    let mut outs = Vec::new();
    let mut prev_strobe_tick = None;
    for e in 0..events {
        let a = fr.step_event();
        let b = fs.step_event();
        assert_eq!(a, b, "{ctx} event {e}");
        check(
            &mut fr,
            &mut fs,
            &flat,
            &format!("{ctx} event {e}"),
            &mut cov,
        );
        // Sample OUT once per strobed cycle (just before each rising edge).
        let next = fs.next_event_tick().unwrap();
        let rising = fs.netlist().sources.iter().any(|s| {
            matches!(s.kind, panda_netlist::flat::SourceKind::Clock { period, phase, .. }
                if (next + u64::from(period) - u64::from(phase) % u64::from(period)) % u64::from(period) == 0)
        });
        if rising
            && next > u64::from(opts.reset_ticks())
            && prev_strobe_tick != Some(next)
            && fs.read_pin("OUT_STROBE").and_then(|v| v.to_u64()) == Some(1)
        {
            prev_strobe_tick = Some(next);
            outs.push(fs.read_pin("OUT").and_then(|v| v.to_u64()).unwrap() as u8);
        }
    }
    assert!(cov.diag_kinds.is_empty(), "{ctx}: diagnostics {cov:?}");
    let mut emu = Emulator::new(&p.image());
    let golden = emu.run(cycles);
    assert!(!outs.is_empty(), "{ctx}: no output");
    assert!(
        golden.starts_with(&outs) && outs.len() + 1 >= golden.len(),
        "{ctx}: OUT {outs:?} vs emulator {golden:?}"
    );
}

fn all(rom: RomStyle) {
    std::thread::scope(|s| {
        for v in Variant::ALL {
            s.spawn(move || lockstep(&programs::BENCH_LOOP, v, rom, 3_200));
        }
    });
}

#[test]
fn panda8_sum_of_products_all_variants() {
    all(RomStyle::SumOfProducts);
}

#[test]
fn panda8_mux_tree_all_variants() {
    all(RomStyle::MuxTree);
}

#[test]
fn panda8_decoder_all_variants() {
    all(RomStyle::Decoder);
}

#[test]
fn panda8_other_programs() {
    for p in [programs::FIBONACCI, programs::MULTIPLY, programs::COUNTDOWN] {
        lockstep(&p, Variant::Hierarchical, RomStyle::SumOfProducts, 600);
    }
}
