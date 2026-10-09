//! Panda-8 through the facade: both modes against the emulator, mode switches mid-run.

mod common;

use std::time::Instant;

use common::*;
use panda_cpu8::programs;
use panda_cpu8::{RomStyle, Variant};
use panda_kernel::{EngineKind, FastBackend, Kernel, Mode, StepUnit};

fn check_cycles(k: &mut Kernel, want: &[panda_cpu8::emu::ArchState], from: usize, to: usize) {
    for (c, w) in want.iter().enumerate().take(to + 1).skip(from) {
        if c > from {
            k.step(1, StepUnit::Cycle);
        }
        assert_eq!(
            arch(k),
            *w,
            "cycle {c}, tick {}, mode {:?}",
            k.tick(),
            k.mode()
        );
    }
}

fn run_mode(mode: Mode, variant: Variant, rom: RomStyle, cycles: usize) {
    let prog = programs::MULTIPLY.image();
    let (design, opts) = cpu_design(&prog, variant, rom);
    let mut k = Kernel::new(quiet_opts());
    k.set_mode(mode);
    let t = Instant::now();
    k.load(design).unwrap();
    k.wait_ready().unwrap();
    let ready = t.elapsed();
    if mode == Mode::Fast {
        assert_eq!(k.engine(), EngineKind::Fast(FastBackend::default()));
    }
    let c0 = cycle0_tick(&opts);
    k.run_to(c0);
    assert_eq!(k.tick(), c0);
    let want = trace(&prog, cycles);
    let t = Instant::now();
    check_cycles(&mut k, &want, 0, cycles);
    let el = t.elapsed();
    eprintln!(
        "{mode:?} {variant:?} {rom:?}: ready in {ready:?}, {cycles} cycles in {el:?} ({:.0} cycles/s)",
        cycles as f64 / el.as_secs_f64()
    );
}

#[test]
fn timing_matches_emulator() {
    run_mode(
        Mode::Timing,
        Variant::Hierarchical,
        RomStyle::SumOfProducts,
        300,
    );
}

#[test]
fn fast_matches_emulator() {
    run_mode(
        Mode::Fast,
        Variant::Hierarchical,
        RomStyle::SumOfProducts,
        300,
    );
}

#[test]
fn fast_matches_emulator_muxtree_flat() {
    run_mode(Mode::Fast, Variant::Flat, RomStyle::MuxTree, 150);
}

#[test]
fn timing_matches_emulator_gated() {
    run_mode(
        Mode::Timing,
        Variant::GatedClockRegfile,
        RomStyle::MuxTree,
        150,
    );
}

#[test]
fn mode_switches_mid_run() {
    let prog = programs::MULTIPLY.image();
    let (design, opts) = cpu_design(&prog, Variant::Hierarchical, RomStyle::SumOfProducts);
    let want = trace(&prog, 320);
    let mut k = Kernel::new(quiet_opts());
    k.load(design).unwrap();
    k.run_to(cycle0_tick(&opts));
    check_cycles(&mut k, &want, 0, 40);
    // Timing → Fast at a phase end (the sampling point is one), compiled in the background.
    k.set_mode(Mode::Fast);
    assert_eq!(k.mode(), Mode::Fast);
    // While compiling, the Timing engine serves Fast mode at phase granularity.
    check_cycles(&mut k, &want, 40, 45);
    k.wait_ready().unwrap();
    assert_eq!(k.engine(), EngineKind::Fast(FastBackend::default()));
    check_cycles(&mut k, &want, 45, 120);
    // Fast → Timing.
    k.set_mode(Mode::Timing);
    assert_eq!(k.engine(), EngineKind::Timing);
    check_cycles(&mut k, &want, 120, 180);
    // And back: the compiled engine is kept, so the switch is immediate.
    k.set_mode(Mode::Fast);
    assert_eq!(k.engine(), EngineKind::Fast(FastBackend::default()));
    check_cycles(&mut k, &want, 180, 230);
    // Switch in the middle of a phase: Timing → Fast waits for the phase end.
    k.set_mode(Mode::Timing);
    k.step(1, StepUnit::Phase);
    k.step(7, StepUnit::Tick);
    k.set_mode(Mode::Fast);
    let t = k.tick();
    k.run_to(t);
    // Re-align to the sampling point and keep checking.
    let c0 = cycle0_tick(&opts);
    let p = u64::from(opts.period);
    let cyc = ((k.tick() - c0) / p + 1) as usize;
    k.run_to(c0 + cyc as u64 * p);
    check_cycles(&mut k, &want, cyc, cyc + 40);
    let bad: Vec<_> = k
        .diagnostics()
        .iter()
        .filter(|d| d.severity == panda_netlist::Severity::Error)
        .collect();
    assert!(bad.is_empty(), "{bad:?}");
}

#[test]
fn phase_steps_agree_between_modes() {
    let prog = programs::FIBONACCI.image();
    let (design, _) = cpu_design(&prog, Variant::Hierarchical, RomStyle::SumOfProducts);
    let mut t = Kernel::new(quiet_opts());
    t.load(design.clone()).unwrap();
    let mut f = Kernel::new(quiet_opts());
    f.set_mode(Mode::Fast);
    f.load(design).unwrap();
    f.wait_ready().unwrap();
    // Fast starts at the end of phase 0; Timing gets there with one Phase step.
    t.step(1, StepUnit::Phase);
    assert_eq!(t.tick(), f.tick());
    let names: Vec<String> = f.netlist().unwrap().net_names.clone();
    for i in 0..60 {
        t.step(1, StepUnit::Phase);
        f.step(1, StepUnit::Phase);
        assert_eq!(t.tick(), f.tick(), "phase {i}");
        if t.tick() > 300 {
            for n in &names {
                assert_eq!(t.read(n), f.read(n), "net {n} at tick {}", t.tick());
            }
        }
    }
}

/// Latencies of the facade operations on Panda-8 (printed; loose sanity bounds only).
#[test]
fn operation_latencies() {
    for rom in [RomStyle::SumOfProducts, RomStyle::MuxTree] {
        let prog = programs::BENCH_LOOP.image();
        let (design, _) = cpu_design(&prog, Variant::Hierarchical, rom);
        let mut k = Kernel::new(quiet_opts());
        let t = Instant::now();
        k.load(design.clone()).unwrap();
        let load = t.elapsed();
        k.step(30, StepUnit::Cycle);
        let t = Instant::now();
        k.precompile();
        k.wait_ready().unwrap();
        let compile = t.elapsed();
        let t = Instant::now();
        k.set_mode(Mode::Fast);
        let to_fast = t.elapsed();
        assert_eq!(k.engine(), EngineKind::Fast(FastBackend::default()));
        k.step(30, StepUnit::Cycle);
        let t = Instant::now();
        k.set_mode(Mode::Timing);
        let to_timing = t.elapsed();
        let t = Instant::now();
        let s = k.snapshot();
        let snap = t.elapsed();
        let t = Instant::now();
        k.restore(&s).unwrap();
        let restore = t.elapsed();
        // An edit that changes one atom (an output pin buffer would do; flip a constant of the ROM).
        let top = k.design().unwrap().top.clone();
        let t = Instant::now();
        k.edit(&[panda_kernel::EditOp::SetTermination {
            def: top,
            net: "rom_d".into(),
            termination: panda_netlist::Termination::PullDown,
        }])
        .unwrap();
        let edit = t.elapsed();
        let t = Instant::now();
        k.set_mode(Mode::Fast);
        k.wait_ready().unwrap();
        let recompile = t.elapsed();
        eprintln!(
            "{rom:?}: load {load:?}, compile (kernels+tables+engine) {compile:?}, Timing→Fast {to_fast:?}, \
             Fast→Timing {to_timing:?}, snapshot {snap:?}, restore {restore:?}, edit (Timing) {edit:?}, \
             recompile after edit (cached tables) {recompile:?}"
        );
        assert!(
            to_fast.as_millis() < 500 && to_timing.as_millis() < 500 && edit.as_millis() < 2000
        );
    }
}
