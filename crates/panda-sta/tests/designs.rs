//! STA on small designs: clean synchronous logic, setup / hold violations, guard hazards, derived
//! clocks, transparent loops. Every verdict is cross-checked against FastRef vs Timing mode.

mod common;

use common::*;
use panda_circuits::{d_latch, ripple_counter, sync_counter, testbench, toggle_ff};
use panda_netlist::{DefBuilder, DesignBuilder, DiagKind, Severity, Value};
use panda_sta::{EventKind, StaOptions, analyze_with};
use panda_timing::{TimingOptions, TimingSim};
use std::sync::Arc;

fn counter(period: u32) -> panda_netlist::Design {
    let mut db = DesignBuilder::new();
    let c = sync_counter(&mut db, 8);
    let tb = testbench(&mut db, &c, period);
    db.build(&tb)
}

fn en_on() -> [(&'static str, Value); 1] {
    [("EN", Value::from_bool(true))]
}

#[test]
fn ms_dff_counter_is_clean_with_tight_min_period() {
    // sync_counter = incrementer + register of d_latch-based master-slave flip-flops.
    let flat = flat_of(&counter(64));
    let r = sta(&flat);
    dump("counter@64", &r);
    assert!(r.is_clean(), "{:?}", r.errors().collect::<Vec<_>>());
    let min = r.min_period_of("clk").unwrap();
    assert!((16..=64).contains(&min), "min period {min}");
    // Events: both edges, the reset release, UI input changes in both clock phases.
    let kinds: Vec<_> = r.phase_settle.iter().map(|p| p.kind).collect();
    assert_eq!(kinds.iter().filter(|&&k| k == EventKind::Clock).count(), 2);
    assert_eq!(kinds.iter().filter(|&&k| k == EventKind::Reset).count(), 1);
    assert_eq!(kinds.iter().filter(|&&k| k == EventKind::Inputs).count(), 2);
    assert!(
        r.critical_path.first().is_some_and(|n| n == "clk"),
        "{:?}",
        r.critical_path
    );

    // At the minimum period the counter counts in Timing mode, and FastRef agrees on every net.
    let flat_min = flat_of(&counter(min));
    let rmin = sta(&flat_min);
    assert!(rmin.is_clean(), "{:?}", rmin.errors().collect::<Vec<_>>());
    let mut sim = TimingSim::new(Arc::new(flat_min.clone()), TimingOptions::default());
    sim.set_input("EN", &Value::from_bool(true)).unwrap();
    let tb = panda_circuits::TestbenchOptions::new(min);
    let first = tb.first_edge_after_reset();
    let mut expect = 0u64;
    for k in first..first + 300 {
        let t = tb.rising_edge(k + 1) - 1;
        sim.run(t - sim.tick());
        expect = (expect + 1) % 256;
        assert_eq!(
            sim.read_pin("Q").unwrap().to_u64(),
            Some(expect),
            "cycle {k}"
        );
    }
    let from = u64::from(tb.reset_ticks);
    assert!(differential_with(&flat_min, 800, from, &en_on()).expect("FastRef == Timing") > 700);
}

#[test]
fn setup_violation_below_min_period_is_real() {
    let min = sta(&flat_of(&counter(64))).min_period_of("clk").unwrap();
    for p in [min - 4, min / 2] {
        let flat = flat_of(&counter(p));
        let r = sta(&flat);
        let setup: Vec<_> = r.of_kind(DiagKind::SetupViolation).collect();
        assert!(!setup.is_empty(), "period {p}");
        assert!(
            setup
                .iter()
                .all(|d| d.severity == Severity::Error && d.location.is_some())
        );
        // The bound is tight for this design: Timing mode really diverges from FastRef.
        let from = u64::from(panda_circuits::TestbenchOptions::new(p).reset_ticks);
        assert!(
            differential_with(&flat, 4000, from, &en_on()).is_err(),
            "period {p}"
        );
    }
}

#[test]
fn hold_violation_from_delayed_clock_is_flagged_and_real() {
    // No skew: clean, and FastRef equals Timing.
    // (Period 64: the reset release a quarter period after a falling edge leaves the delayed
    // flip-flop's falling-edge activity enough time.)
    let flat = flat_of(&hold_design(64, 0));
    let r = sta(&flat);
    assert!(r.is_clean(), "{:?}", r.errors().collect::<Vec<_>>());
    assert!(differential(&flat, 200, 200).is_ok());
    // ff2's clock arrives 4 ticks late; ff1.Q changes 2 ticks after the edge.
    let flat = flat_of(&hold_design(64, 4));
    let r = sta(&flat);
    dump("hold", &r);
    let holds: Vec<_> = r.of_kind(DiagKind::HoldViolation).collect();
    assert!(!holds.is_empty());
    assert!(
        holds
            .iter()
            .all(|d| d.location.as_deref().is_some_and(|l| l.starts_with("ff2/"))),
        "{holds:?}"
    );
    assert!(holds[0].message.contains("Q1"), "{}", holds[0].message);
    assert_eq!(count(&r, DiagKind::SetupViolation), 0);
    // It is a real equivalence failure.
    assert!(differential(&flat, 200, 200).is_err());
}

#[test]
fn gated_clock_with_glitchy_enable_is_a_guard_hazard() {
    // en = s·1 + ¬s·1 (static-1 hazard when s changes), s toggles at every rising edge, and
    // gclk = clk · en is shared by two flip-flops.
    let d = shared_clock_design("glitchy_gate", 64, 2, |b, clk, rst, lib| {
        let one = b.lit(1, 1);
        let s = b.inst_named("sel", &lib.tff, &[one, rst, clk])[0];
        let ns = b.not(s);
        let a1 = b.and(&[one, s]);
        let a2 = b.and(&[one, ns]);
        let en = b.or(&[a1, a2]);
        b.and(&[clk, en])
    });
    let flat = flat_of(&d);
    let r = sta(&flat);
    dump("glitchy gate", &r);
    let hz: Vec<_> = r.of_kind(DiagKind::GuardHazard).collect();
    assert!(!hz.is_empty());
    assert!(hz.iter().any(|d| d.message.contains("gclk")), "{hz:?}");
    assert!(differential(&flat, 200, 200).is_err());
}

#[test]
fn pulse_generated_clock_is_a_guard_hazard() {
    // pclk = clk · ¬delay(clk): a 3-tick pulse at every rising edge (SPEC §9.3).
    let d = shared_clock_design("pulse_clock", 64, 2, |b, clk, _rst, _| {
        let d1 = b.buf(clk);
        let d2 = b.buf(d1);
        let nd = b.not(d2);
        b.and(&[clk, nd])
    });
    let flat = flat_of(&d);
    let r = sta(&flat);
    dump("pulse clock", &r);
    assert!(count(&r, DiagKind::GuardHazard) > 0);
    assert!(differential(&flat, 200, 200).is_err());
}

#[test]
fn latch_based_clock_gate_is_clean() {
    // The standard clock-gating cell: the enable passes a latch transparent while clk is low, so
    // gclk = clk · en_l cannot glitch. Enable = a toggling flip-flop.
    let d = shared_clock_design("icg", 64, 3, |b, clk, rst, lib| {
        let one = b.lit(1, 1);
        let s = b.inst_named("sel", &lib.tff, &[one, rst, clk])[0];
        let en = b.or(&[s, rst]);
        let nclk = b.not(clk);
        let en_l = b.inst_named("icg", &lib.latch, &[en, nclk])[0];
        b.and(&[clk, en_l])
    });
    let flat = flat_of(&d);
    let r = sta(&flat);
    dump("icg", &r);
    assert!(r.is_clean(), "{:?}", r.errors().collect::<Vec<_>>());
    assert!(differential(&flat, 300, 200).expect("FastRef == Timing") > 200);
}

#[test]
fn transparent_loop_never_settles() {
    // A latch transparent while clk is high, with D = NOT Q: oscillates in the high phase.
    let mut db = DesignBuilder::new();
    let l = d_latch(&mut db);
    let mut b = DefBuilder::new("loop_top");
    let clk = b.clock(64, 32, 32);
    let qw = b.wire(1);
    let nq = b.not(qw);
    let q = b.inst_named("l", &l, &[nq, clk])[0];
    b.connect(q, qw);
    b.output("Q", q);
    let top = db.add(b.finish());
    let r = sta(&flat_of(&db.build(&top)));
    dump("transparent loop", &r);
    assert!(!r.is_clean());
    assert!(count(&r, DiagKind::Oscillating) > 0);
    assert!(count(&r, DiagKind::SetupViolation) > 0);
}

#[test]
fn transparent_loop_through_two_kernels_is_unbounded() {
    // Two latches, both transparent while clk is high, in a loop with an inverter: the rounds
    // never end (each kernel is a plain latch; the loop goes through combinational logic).
    let mut db = DesignBuilder::new();
    let l = d_latch(&mut db);
    let mut b = DefBuilder::new("loop2_top");
    let clk = b.clock(64, 32, 32);
    let q2w = b.wire(1);
    let n2 = b.not(q2w);
    let b1 = b.buf(n2);
    let q1 = b.inst_named("l1", &l, &[b1, clk])[0];
    let b2 = b.buf(q1);
    let b3 = b.buf(b2);
    let q2 = b.inst_named("l2", &l, &[b3, clk])[0];
    b.connect(q2, q2w);
    b.output("Q", q2);
    // Make the intermediate nets non-private so the latches stay separate kernels.
    b.output("A", b1);
    b.output("B", b3);
    let top = db.add(b.finish());
    let flat = flat_of(&db.build(&top));
    let r = sta(&flat);
    dump("loop2", &r);
    let rise = r
        .phase_settle
        .iter()
        .find(|p| p.event.ends_with('↑'))
        .unwrap();
    assert_eq!(rise.settle, None, "{rise:?}");
    assert!(r.min_period.iter().all(|&(_, p)| p == u32::MAX));
    assert!(
        r.of_kind(DiagKind::SetupViolation)
            .any(|d| d.message.contains("may not settle"))
    );
}

#[test]
fn ripple_counter_derived_clocks() {
    // Stage i+1 is clocked by stage i's Q_n (SPEC §8.5.4): several rounds per edge.
    let mut db = DesignBuilder::new();
    let rc = ripple_counter(&mut db, 4);
    let tb = testbench(&mut db, &rc, 64);
    let flat = flat_of(&db.build(&tb));
    let r = sta(&flat);
    dump("ripple", &r);
    assert!(r.is_clean(), "{:?}", r.errors().collect::<Vec<_>>());
    let rise = r.phase_settle.iter().find(|p| p.event == "clk↑").unwrap();
    assert!(rise.rounds >= 4, "{rise:?}");
    assert!(differential(&flat, 400, 200).expect("FastRef == Timing") > 300);
}

fn two_clock_design(pa: u32, pb: u32, reset: u32) -> panda_netlist::Design {
    let mut db = DesignBuilder::new();
    let tff = toggle_ff(&mut db);
    let mut b = DefBuilder::new(&format!("two_clocks_{pa}_{pb}"));
    let c1 = b.clock(pa, pa / 2, 0);
    b.name(c1, "ca");
    let c2 = b.clock(pb, pb / 2, 0);
    b.name(c2, "cb");
    let rst = b.power_on_reset(reset);
    let one = b.lit(1, 1);
    let q1 = b.inst_named("a", &tff, &[one, rst, c1])[0];
    let q2 = b.inst_named("b", &tff, &[one, rst, c2])[0];
    b.output("QA", q1);
    b.output("QB", q2);
    let top = db.add(b.finish());
    db.build(&top)
}

#[test]
fn two_clock_domains() {
    // Periods 48 and 80 (both rising at 0): merged timeline 0, 24, 40, 48, 72, 80, 96, 120, …;
    // the smallest gap between events of different clocks is 8 ticks.
    // Reset release at 330: 10 ticks after cb↑ (320), 6 before ca↑ (336).
    let flat = flat_of(&two_clock_design(48, 80, 330));
    let r = sta(&flat);
    dump("two clocks 48/80", &r);
    assert!(r.is_clean(), "{:?}", r.errors().collect::<Vec<_>>());
    assert_eq!(r.min_period.len(), 2);
    assert!(r.min_period_of("ca").is_some() && r.min_period_of("cb").is_some());
    let clock_events = r
        .phase_settle
        .iter()
        .filter(|p| p.kind == EventKind::Clock)
        .count();
    assert!(clock_events > 4, "{clock_events}");
    assert!(
        r.phase_settle.iter().any(|p| p.event == "ca↑ cb↑"),
        "simultaneous edges form one event"
    );
    let min_gap = r
        .phase_settle
        .iter()
        .filter(|p| p.kind == EventKind::Clock)
        .filter_map(|p| p.gap)
        .min();
    assert_eq!(min_gap, Some(8));
    assert!(differential(&flat, 300, 400).expect("FastRef == Timing") > 200);

    // Periods 24 and 40: edges of the two clocks only 4 ticks apart. Each domain is fine on its
    // own (min periods are unchanged), but a phase boundary falls inside the other domain's
    // activity: a real setup violation.
    let flat = flat_of(&two_clock_design(24, 40, 330));
    let r = sta(&flat);
    dump("two clocks 24/40", &r);
    assert!(count(&r, DiagKind::SetupViolation) > 0);
    assert!(r.min_period_of("ca").unwrap() <= 24 && r.min_period_of("cb").unwrap() <= 40);
    assert!(differential(&flat, 300, 400).is_err());
}

#[test]
fn options_disable_input_events_and_advisories() {
    let flat = flat_of(&counter(64));
    let (ks, tables) = panda_compile::compile_kernels(&flat, &Default::default());
    let opts = StaOptions {
        input_events: false,
        advisory: false,
        ..StaOptions::default()
    };
    let r = analyze_with(&flat, &ks, &tables, &opts);
    assert!(r.phase_settle.iter().all(|p| p.kind != EventKind::Inputs));
    assert_eq!(count(&r, DiagKind::DelaySensitive), 0);
    assert!(r.is_clean());
}
