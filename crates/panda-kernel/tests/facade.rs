//! run(budget), diagnostics flow, timing report, misc API behaviour.

mod common;

use std::time::Duration;

use common::*;
use panda_circuits::{ms_dff, ring_oscillator, sync_counter, testbench};
use panda_cpu8::programs;
use panda_cpu8::{RomStyle, Variant};
use panda_kernel::{DiagSource, EngineKind, Kernel, KernelOptions, Mode, StepUnit};
use panda_netlist::{DefBuilder, DesignBuilder, DiagKind, Severity, Value};

fn cpu_kernel(mode: Mode) -> Kernel {
    let (design, _) = cpu_design(
        &programs::BENCH_LOOP.image(),
        Variant::Hierarchical,
        RomStyle::SumOfProducts,
    );
    let mut k = Kernel::new(quiet_opts());
    k.set_mode(mode);
    k.load(design).unwrap();
    k.wait_ready().unwrap();
    k
}

fn run_budget(mode: Mode) {
    let mut k = cpu_kernel(mode);
    k.step(4, StepUnit::Cycle);
    for ms in [5u64, 40, 120] {
        let budget = Duration::from_millis(ms);
        let t0 = k.tick();
        let s = k.run(budget);
        eprintln!(
            "{mode:?} run({budget:?}): elapsed {:?}, {} cycles, {} ticks ({:.0} cycles/s)",
            s.elapsed,
            s.cycles,
            s.ticks,
            s.cycles as f64 / s.elapsed.as_secs_f64()
        );
        assert!(s.elapsed >= budget, "{s:?}");
        assert!(
            s.elapsed < budget + Duration::from_millis(15) + budget / 5,
            "overran: {s:?}"
        );
        assert!(!s.idle);
        assert!(s.cycles > 0);
        assert_eq!(s.ticks, k.tick() - t0);
        assert_eq!(s.cycles, s.ticks / 128, "{s:?}");
        assert_eq!(s.mode, mode);
    }
}

#[test]
fn run_respects_budget_timing() {
    run_budget(Mode::Timing);
}

#[test]
fn run_respects_budget_fast() {
    run_budget(Mode::Fast);
}

#[test]
fn run_idles_without_activity() {
    let mut db = DesignBuilder::new();
    let mut b = DefBuilder::new("inv");
    let a = b.input("A", 1);
    let y = b.not(a);
    b.output("Y", y);
    let top = db.add(b.finish());
    let design = db.build(&top);
    for mode in [Mode::Timing, Mode::Fast] {
        let mut k = Kernel::default();
        k.set_mode(mode);
        k.load(design.clone()).unwrap();
        k.wait_ready().unwrap();
        let s = k.run(Duration::from_secs(5));
        assert!(s.idle && s.elapsed < Duration::from_secs(1), "{s:?}");
        k.set_input("A", &Value::from_bool(true)).unwrap();
        k.step(1, StepUnit::Phase);
        assert_eq!(k.read_u64("Y"), Some(0), "{mode:?}");
        k.set_input("A", &Value::from_bool(false)).unwrap();
        k.step(1, StepUnit::Cycle); // no clock: behaves like Phase
        assert_eq!(k.read_u64("Y"), Some(1));
        assert!(k.set_input("A", &Value::zeros(2)).is_err());
        assert!(k.set_input("B", &Value::zeros(1)).is_err());
        assert_eq!(k.input("A"), Some(Value::from_bool(false)));
    }
}

#[test]
fn reads_inside_packages_in_both_modes() {
    let mut t = cpu_kernel(Mode::Timing);
    let mut f = cpu_kernel(Mode::Fast);
    t.step(40, StepUnit::Cycle);
    f.run_to(t.tick());
    assert_eq!(t.tick(), f.tick());
    for name in [
        "cpu/pc",
        "cpu/ROM_D",
        "cpu/alu/Y",
        "rom/A",
        "rom_d",
        "cpu/rf/CLK",
        "clk",
        "PC",
        "R0",
    ] {
        let a = t.read(name);
        assert!(a.is_some(), "{name}");
        assert_eq!(a, f.read(name), "{name}");
    }
    assert!(t.read("cpu/nope").is_none() && f.read("cpu/nope").is_none());
    assert_eq!(t.read("cpu/pc").unwrap().width(), 8);
}

#[test]
fn diagnostics_flow() {
    // Load errors are returned and leave the kernel empty.
    let mut k = Kernel::default();
    let mut db = DesignBuilder::new();
    let mut b = DefBuilder::new("bad");
    let a = b.input("A", 2);
    let y = b.not(a);
    b.output("Y", y);
    let top = db.add(b.finish());
    let mut bad = db.build(&top);
    bad.defs[0].nets[0].width = 3; // width mismatch
    let errs = k.load(bad).unwrap_err();
    assert!(
        errs.iter()
            .any(|d| d.severity == Severity::Error && d.kind == DiagKind::WidthMismatch),
        "{errs:?}"
    );
    assert_eq!(k.engine(), EngineKind::None);
    assert!(k.design().is_none());

    // Contention between a clock buffer and a constant.
    let mut db = DesignBuilder::new();
    let mut b = DefBuilder::new("noisy");
    let clk = b.clock(8, 4, 4);
    let shared = b.buf(clk);
    let zero = b.lit(1, 0);
    b.connect(shared, zero);
    b.output("S", shared);
    let top = db.add(b.finish());
    let noisy = db.build(&top);
    // An enabled ring oscillator (no clock, so Timing mode sees uninterrupted activity).
    let mut db = DesignBuilder::new();
    let osc = ring_oscillator(&mut db, 3, true);
    let mut b = DefBuilder::new("osc_top");
    let en = b.input("EN", 1);
    let o = b.inst_named("osc", &osc, &[en]);
    b.output("O", o[0]);
    let top = db.add(b.finish());
    let oscillating = db.build(&top);
    for mode in [Mode::Timing, Mode::Fast] {
        let mut k = Kernel::new(KernelOptions {
            timing: panda_timing::TimingOptions::default().with_osc_limit(200),
            ..Default::default()
        });
        k.set_mode(mode);
        k.load(noisy.clone()).unwrap();
        k.wait_ready().unwrap();
        // Validation warning (two non-tri-state drivers).
        assert!(
            k.diagnostic_entries()
                .iter()
                .any(|e| e.source == DiagSource::Design
                    && e.diagnostic.kind == DiagKind::MultipleDrivers),
            "{:?}",
            k.diagnostics()
        );
        k.step(40, StepUnit::Cycle);
        let src = if mode == Mode::Timing {
            DiagSource::Timing
        } else {
            DiagSource::Fast
        };
        let cont: Vec<_> = k
            .diagnostic_entries()
            .iter()
            .filter(|e| e.diagnostic.kind == DiagKind::Contention && e.source == src)
            .collect();
        assert_eq!(cont.len(), 1, "deduplicated: {:?}", k.diagnostic_entries());
        assert!(
            cont[0].count > 1,
            "repeated contention episodes are counted: {cont:?}"
        );
        assert!(k.diagnostics_at_least(Severity::Error).is_empty());
        assert!(!k.diagnostics_at_least(Severity::Warning).is_empty());

        // Loading another design replaces the design diagnostics, keeps the simulation ones.
        k.load(oscillating.clone()).unwrap();
        k.wait_ready().unwrap();
        assert!(
            !k.diagnostics()
                .iter()
                .any(|d| d.kind == DiagKind::MultipleDrivers)
        );
        assert!(
            k.diagnostics()
                .iter()
                .any(|d| d.kind == DiagKind::Contention)
        );
        k.step(5, StepUnit::Tick); // EN = 0: the ring settles
        k.set_input("EN", &Value::from_bool(true)).unwrap();
        k.step(1000, StepUnit::Tick);
        assert!(
            k.diagnostics()
                .iter()
                .any(|d| d.kind == DiagKind::Oscillating),
            "{mode:?}: {:?}",
            k.diagnostics()
        );
        // STA flags the oscillator as an error and keeps its findings in the unified list.
        let rep = k.timing_report();
        assert!(!rep.is_clean());
        assert!(
            k.diagnostic_entries()
                .iter()
                .any(|e| e.source == DiagSource::Sta && e.diagnostic.severity == Severity::Error)
        );
        assert!(!k.diagnostics_at_least(Severity::Error).is_empty());
        k.clear_diagnostics();
        assert!(k.diagnostics().is_empty());
    }
}

#[test]
fn timing_report_and_period_suggestion() {
    let mut k = cpu_kernel(Mode::Timing);
    let rep = k.timing_report();
    assert_eq!(rep.clocks.len(), 1);
    let c = &rep.clocks[0];
    eprintln!(
        "Panda-8 clock {}: period {}, STA min {:?}, suggested {:?}",
        c.net, c.period, c.min_period, c.suggested_period
    );
    assert_eq!(c.period, 128);
    assert!(c.meets_timing);
    let min = c.suggested_period.unwrap();
    assert!((2..=128).contains(&min) && min.is_multiple_of(2));
    assert_eq!(rep.suggested_period(&c.net), Some(min));
    assert!(
        rep.report.is_clean(),
        "{:?}",
        rep.report.errors().collect::<Vec<_>>()
    );
    assert!(rep.warnings.is_empty());

    // A counter clocked too fast: setup warning with the suggested period.
    let mut db = DesignBuilder::new();
    let cnt = sync_counter(&mut db, 6);
    let tb = testbench(&mut db, &cnt, 4);
    let mut k = Kernel::default();
    k.load(db.build(&tb)).unwrap();
    let rep = k.timing_report();
    let c = &rep.clocks[0];
    assert!(!c.meets_timing, "{c:?}");
    assert!(c.suggested_period.unwrap() > 4);
    assert!(!rep.is_clean());
    assert!(
        k.diagnostics()
            .iter()
            .any(|d| d.kind == DiagKind::SetupViolation && d.severity == Severity::Warning)
    );
    // Cached until the design changes.
    assert_eq!(k.timing_report(), rep);
}

#[test]
fn ms_dff_tick_level_reads_and_fast_overshoot() {
    let mut db = DesignBuilder::new();
    let ff = ms_dff(&mut db);
    let tb = testbench(&mut db, &ff, 20);
    let design = db.build(&tb);
    let mut t = Kernel::default();
    t.load(design.clone()).unwrap();
    t.set_input("D", &Value::from_bool(true)).unwrap();
    t.step(25, StepUnit::Tick);
    assert_eq!(t.tick(), 25);
    // The first rising edge is at 10: Q = 1 (sampled D = 1) settles within a few ticks.
    assert_eq!(t.read_u64("Q"), Some(1));
    // Fast mode cannot stop inside a phase: Tick steps end at the end of the phase.
    let mut f = Kernel::default();
    f.set_mode(Mode::Fast);
    f.load(design).unwrap();
    f.wait_ready().unwrap();
    // After power-on the Fast engine is at the end of phase [0, 10).
    assert_eq!(f.tick(), 9);
    f.set_input("D", &Value::from_bool(true)).unwrap();
    f.step(25, StepUnit::Tick);
    assert_eq!(
        f.tick(),
        39,
        "target 34 lies in phase [30, 40), which ends at 39"
    );
    assert_eq!(f.read_u64("Q"), Some(1));
    t.run_to(39);
    assert_eq!(t.read("dut/m"), f.read("dut/m"));
    assert_eq!(t.read("Q"), f.read("Q"));
}

#[test]
fn no_design() {
    let mut k = Kernel::default();
    k.step(10, StepUnit::Cycle);
    k.set_mode(Mode::Fast);
    assert_eq!(k.tick(), 0);
    assert!(k.read("x").is_none());
    assert!(k.set_input("A", &Value::zeros(1)).is_err());
    assert!(k.edit(&[]).is_err());
    assert!(k.wait_ready().is_ok());
    let s = k.run(Duration::from_millis(10));
    assert!(s.idle);
}

#[test]
fn kernel_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Kernel>();
    assert_send::<panda_kernel::State>();
}
