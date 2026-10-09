//! Self-tests of the harness: the comparison detects a real disagreement (a clock period below
//! the STA minimum), agrees on a clean design, and the minimizer shrinks failing designs.

use std::sync::Arc;

use panda_circuits::sync_counter;
use panda_difftest::driver::{DriverOptions, run_netlist};
use panda_difftest::harness::{Class, HarnessConfig, plan_from_report, run_seed};
use panda_difftest::minimize::{evaluate, minimize};
use panda_netlist::{DefBuilder, DesignBuilder, DiagKind, flatten};

/// A 6-bit counter whose enable is `EN AND NOT HOLD` (two top-level inputs), synchronous reset.
fn counter(period: u32) -> Arc<panda_netlist::FlatNetlist> {
    let mut db = DesignBuilder::new();
    let c = sync_counter(&mut db, 6);
    let mut b = DefBuilder::new("counter_top");
    let clk = b.clock(period, period / 2, period / 2);
    b.name(clk, "clk");
    let rst = b.power_on_reset(2 * period + period / 4);
    let en = b.input("EN", 1);
    let hold = b.input("HOLD", 1);
    let nh = b.not(hold);
    let e = b.and(&[en, nh]);
    let ne = b.not(e);
    // Counts when the inputs say so; with the inputs at their default (0) it counts anyway.
    let en_or_default = b.nand(&[ne, nh]);
    let q = b.inst(&c, &[clk, rst, en_or_default])[0];
    b.output("Q", q);
    let top = db.add(b.finish());
    Arc::new(flatten(&db.build(&top)).expect("flatten"))
}

const RESET_PHASES: u64 = 3;

#[test]
fn period_below_sta_minimum_is_flagged_and_detected() {
    let min = panda_sta::analyze_netlist(&counter(128))
        .min_period_of("clk")
        .expect("min period");
    assert!(min > 8, "min period {min}");
    // Every period below the minimum is flagged; short enough ones really break the counter, and
    // the comparison must see it.
    let opts = DriverOptions {
        phases: 200,
        input_pct: 0,
        ..DriverOptions::default()
    };
    let mut detected = Vec::new();
    for p in (4..min).step_by(2) {
        let flat = counter(p);
        let r = panda_sta::analyze_netlist(&flat);
        assert!(
            r.of_kind(DiagKind::SetupViolation).next().is_some(),
            "STA must flag period {p} < {min}"
        );
        let mut plan = plan_from_report(&r, &flat, RESET_PHASES * u64::from(p), 1, 8);
        plan.inputs.clear();
        let d = run_netlist(&flat, &plan, &opts).expect("FastRef");
        if !d.equal() {
            let m = d.first().unwrap();
            assert!(m.count > 0 && !m.nets.is_empty());
            detected.push(p);
        }
    }
    assert!(
        !detected.is_empty(),
        "no period below {min} produced a detected disagreement"
    );
}

#[test]
fn clean_counter_at_min_period_agrees_with_inputs() {
    let min = panda_sta::analyze_netlist(&counter(128))
        .min_period_of("clk")
        .expect("min period");
    // Room for input changes: the counter's input event settles quickly.
    let p = (min * 4).next_multiple_of(2);
    let flat = counter(p);
    let r = panda_sta::analyze_netlist(&flat);
    assert!(r.is_clean(), "{:?}", r.errors().collect::<Vec<_>>());
    let opts = DriverOptions {
        phases: 200,
        input_pct: 70,
        ..DriverOptions::default()
    };
    let plan = plan_from_report(&r, &flat, 3 * u64::from(p), 1, 8);
    assert!(
        plan.input_margin + plan.input_settle + 1 < u64::from(p / 2),
        "{plan:?}"
    );
    let d = run_netlist(&flat, &plan, &opts).expect("FastRef");
    assert!(d.equal(), "{:?}", d.first().map(ToString::to_string));
    assert!(
        d.compared > 150 && d.input_events > 20,
        "{} compared, {} input events",
        d.compared,
        d.input_events
    );
}

#[test]
fn every_seed_builds_and_runs_deterministically() {
    let cfg = HarnessConfig::default();
    for seed in [3u64, 17, 42] {
        let a = run_seed(seed, &cfg);
        let b = run_seed(seed, &cfg);
        assert_ne!(a.class, Class::Skipped, "{:?}", a.skip_reason);
        assert_eq!(a.class, b.class);
        assert_eq!(a.scale, b.scale);
        assert_eq!(
            a.diff.as_ref().map(|d| d.compared),
            b.diff.as_ref().map(|d| d.compared)
        );
    }
}

#[test]
fn minimizer_shrinks_a_mismatching_design() {
    let cfg = HarnessConfig::default();
    let o = (0..60u64)
        .map(|s| run_seed(s, &cfg))
        .find(|o| {
            // Not an ideal-block run: the minimizer inlines the design, which removes the
            // packages Timing mode would evaluate as ideal blocks.
            o.class == Class::FlaggedMismatch
                && o.runtime_flags.is_empty()
                && o.explained.is_empty()
                && !o.features.contains(&"ideal_blocks")
        })
        .expect("a flagged-and-mismatching design");
    let g = panda_difftest::generate(o.seed, &cfg.generator, o.scale);
    let keep = |o: &panda_difftest::Outcome| o.diff.as_ref().is_some_and(|d| !d.equal());
    let (m, evals) = minimize(&g, &cfg, &keep, 400);
    let before = flatten(&g.design).unwrap().gates.len();
    let after = flatten(&m.design).unwrap().gates.len();
    assert!(evals > 1);
    assert!(after < before, "seed {}: {before} -> {after} gates", o.seed);
    assert!(
        keep(&evaluate(&g, m.design.clone(), &cfg)),
        "the reduced design must still mismatch"
    );
}
