//! Behavioral tests of latches, flip-flops and registers (Timing mode).

mod common;

use common::*;
use panda_circuits::*;
use panda_netlist::{DesignBuilder, Logic};
use panda_timing::TimingSim;

use Logic::{One, X, Zero};

fn qs(s: &TimingSim) -> (Logic, Logic) {
    (bit(s, "Q"), bit(s, "Q_n"))
}

#[test]
fn nand_sr_latch_truth_table() {
    let mut db = DesignBuilder::new();
    let top = nand_sr_latch(&mut db);
    let mut s = sim_top(db, &top);
    set(&mut s, "S_n", 1);
    set(&mut s, "R_n", 1);
    s.reset();
    settle(&mut s);
    assert_eq!(qs(&s), (X, X), "power-on hold is X");
    for (sn, rn, want) in [
        (0, 1, (One, Zero)),
        (1, 1, (One, Zero)),
        (1, 0, (Zero, One)),
        (1, 1, (Zero, One)),
        (0, 0, (One, One)),
        (0, 1, (One, Zero)),
    ] {
        set(&mut s, "S_n", sn);
        set(&mut s, "R_n", rn);
        settle(&mut s);
        assert_eq!(qs(&s), want, "S_n={sn} R_n={rn}");
    }
    assert_clean(&mut s);
}

#[test]
fn nor_sr_latch_truth_table() {
    let mut db = DesignBuilder::new();
    let top = nor_sr_latch(&mut db);
    let mut s = sim_top(db, &top);
    settle(&mut s);
    assert_eq!(qs(&s), (X, X), "power-on hold is X");
    for (sv, rv, want) in [
        (1, 0, (One, Zero)),
        (0, 0, (One, Zero)),
        (0, 1, (Zero, One)),
        (0, 0, (Zero, One)),
        (1, 1, (Zero, Zero)),
        (1, 0, (One, Zero)),
    ] {
        set(&mut s, "S", sv);
        set(&mut s, "R", rv);
        settle(&mut s);
        assert_eq!(qs(&s), want, "S={sv} R={rv}");
    }
    assert_clean(&mut s);
}

#[test]
fn d_latch_transparent_and_hold() {
    let mut db = DesignBuilder::new();
    let top = d_latch(&mut db);
    let mut s = sim_top(db, &top);
    settle(&mut s);
    assert_eq!(qs(&s), (X, X));
    set(&mut s, "EN", 1);
    for d in [1, 0, 1, 1, 0] {
        set(&mut s, "D", d);
        settle(&mut s);
        let want = if d == 1 { (One, Zero) } else { (Zero, One) };
        assert_eq!(qs(&s), want, "transparent, D={d}");
    }
    // Close with D = 0, then wiggle D: held.
    set(&mut s, "EN", 0);
    settle(&mut s);
    for d in [1, 0, 1] {
        set(&mut s, "D", d);
        settle(&mut s);
        assert_eq!(qs(&s), (Zero, One), "holding, D={d}");
    }
    // Reopen: follows D again.
    set(&mut s, "EN", 1);
    settle(&mut s);
    assert_eq!(qs(&s), (One, Zero));
    assert_clean(&mut s);
}

#[test]
fn d_latch_clr_clears_asynchronously() {
    let mut db = DesignBuilder::new();
    let top = d_latch_clr(&mut db);
    let mut s = sim_top(db, &top);
    set(&mut s, "EN", 1);
    set(&mut s, "D", 1);
    settle(&mut s);
    assert_eq!(qs(&s), (One, Zero));
    set(&mut s, "EN", 0);
    settle(&mut s);
    set(&mut s, "CLR", 1);
    settle(&mut s);
    assert_eq!(qs(&s), (Zero, One), "clear while holding");
    set(&mut s, "EN", 1);
    settle(&mut s);
    assert_eq!(qs(&s), (Zero, One), "clear dominates a transparent latch");
    set(&mut s, "EN", 0);
    settle(&mut s);
    set(&mut s, "CLR", 0);
    settle(&mut s);
    assert_eq!(qs(&s), (Zero, One), "holds 0 after release");
    set(&mut s, "EN", 1);
    settle(&mut s);
    assert_eq!(qs(&s), (One, Zero), "transparent again after release");
    assert_clean(&mut s);
}

const P: u32 = 32;

#[test]
fn ms_dff_samples_on_rising_edge_only() {
    let mut db = DesignBuilder::new();
    let ff = ms_dff(&mut db);
    let opts = TestbenchOptions::new(P);
    let tb = testbench_with(&mut db, &ff, &opts);
    let mut s = sim_top(db, &tb);
    // Before any edge the state is unknown.
    run_to(&mut s, opts.rising_edge(0) - u64::from(P / 4) - 1);
    assert_eq!(qs(&s), (X, X));
    let pattern = [1, 0, 0, 1, 1, 0, 1, 0, 1, 1];
    let mut prev: Option<u64> = None;
    for (k, &d) in pattern.iter().enumerate() {
        let k = k as u64;
        let q = u64::from(P / 4);
        run_to(&mut s, opts.rising_edge(k) - q);
        set(&mut s, "D", d);
        // Until the edge the old value is held.
        run_to(&mut s, opts.rising_edge(k));
        if let Some(p) = prev {
            assert_eq!(get(&s, "Q"), p, "before edge {k}");
        }
        // A D change during the high phase must not be captured.
        run_to(&mut s, opts.rising_edge(k) + q);
        set(&mut s, "D", 1 - d);
        run_to(&mut s, opts.rising_edge(k + 1) - q - 1);
        assert_eq!(get(&s, "Q"), d, "after edge {k}");
        assert_eq!(get(&s, "Q_n"), 1 - d, "after edge {k}");
        prev = Some(d);
    }
    assert_clean(&mut s);
}

#[test]
fn dff_rst_resets_synchronously() {
    let mut db = DesignBuilder::new();
    let ff = dff_rst(&mut db);
    let opts = TestbenchOptions::new(P);
    let tb = testbench_with(&mut db, &ff, &opts);
    let mut s = sim_top(db, &tb);
    set(&mut s, "D", 1);
    let k0 = opts.first_edge_after_reset();
    // Reset is asserted for the edges before k0 (D = 1 is ignored).
    run_to(&mut s, opts.rising_edge(k0) - u64::from(P / 4) - 1);
    assert_eq!(qs(&s), (Zero, One));
    for (i, d) in [1, 1, 0, 1, 0, 0].into_iter().enumerate() {
        cycle(&mut s, &opts, k0 + i as u64, &[("D", d)]);
        assert_eq!(get(&s, "Q"), d);
    }
    assert_clean(&mut s);
}

#[test]
fn dff_en_holds_without_enable() {
    let mut db = DesignBuilder::new();
    let ff = dff_en(&mut db);
    let opts = TestbenchOptions::new(P);
    let tb = testbench_with(&mut db, &ff, &opts);
    let mut s = sim_top(db, &tb);
    let mut model: Option<u64> = None;
    let seq = [(1, 1), (0, 0), (0, 1), (1, 0), (1, 1), (0, 0), (0, 1), (1, 1)];
    for (k, (d, en)) in seq.into_iter().enumerate() {
        cycle(&mut s, &opts, k as u64, &[("D", d), ("EN", en)]);
        if en == 1 {
            model = Some(d);
        }
        assert_eq!(get(&s, "Q"), model.unwrap(), "cycle {k}");
    }
    assert_clean(&mut s);
}

#[test]
fn dff_en_power_on_hold_stays_x() {
    let mut db = DesignBuilder::new();
    let ff = dff_en(&mut db);
    let opts = TestbenchOptions::new(P);
    let tb = testbench_with(&mut db, &ff, &opts);
    let mut s = sim_top(db, &tb);
    run_to(&mut s, opts.rising_edge(3));
    assert_eq!(bit(&s, "Q"), X);
}

#[test]
fn dff_arst_clears_without_clock_edge() {
    let mut db = DesignBuilder::new();
    let ff = dff_arst(&mut db);
    // CLR is a user input here (no reset source).
    let opts = TestbenchOptions { reset_pin: "none".into(), ..TestbenchOptions::new(P) };
    let tb = testbench_with(&mut db, &ff, &opts);
    let mut s = sim_top(db, &tb);
    cycle(&mut s, &opts, 0, &[("D", 1)]);
    assert_eq!(get(&s, "Q"), 1);
    // Assert CLR in the middle of the high phase; Q drops within a few ticks.
    run_to(&mut s, opts.rising_edge(1) + u64::from(P / 8));
    set(&mut s, "CLR", 1);
    s.run(6);
    assert_eq!(get(&s, "Q"), 0);
    // Held at 0 across edges while CLR = 1, even with D = 1.
    cycle(&mut s, &opts, 2, &[]);
    assert_eq!(get(&s, "Q"), 0);
    cycle(&mut s, &opts, 3, &[("CLR", 0)]);
    assert_eq!(get(&s, "Q"), 1, "captures D again after release");
    assert_clean(&mut s);
}

#[test]
fn toggle_ff_toggles() {
    let mut db = DesignBuilder::new();
    let ff = toggle_ff(&mut db);
    let opts = TestbenchOptions::new(P);
    let tb = testbench_with(&mut db, &ff, &opts);
    let mut s = sim_top(db, &tb);
    set(&mut s, "T", 1);
    let k0 = opts.first_edge_after_reset();
    run_to(&mut s, opts.rising_edge(k0) - u64::from(P / 4) - 1);
    assert_eq!(get(&s, "Q"), 0);
    let mut q = 0;
    for (i, t) in [1, 1, 1, 0, 0, 1, 0, 1, 1].into_iter().enumerate() {
        cycle(&mut s, &opts, k0 + i as u64, &[("T", t)]);
        q ^= t;
        assert_eq!(get(&s, "Q"), q, "cycle {i}");
    }
    assert_clean(&mut s);
}

/// Drive a register-like DUT (pins D, LOAD, RST, CLK → Q) with a random sequence and compare with
/// the behavioral model.
fn check_register(gated: bool) {
    let n = 8;
    let mut db = DesignBuilder::new();
    let reg = if gated { register_gated(&mut db, n) } else { register(&mut db, n) };
    assert_eq!(reg.id, if gated { "register_gated_8" } else { "register_8" });
    let opts = TestbenchOptions { reset_pin: "none".into(), ..TestbenchOptions::new(P) };
    let tb = testbench_with(&mut db, &reg, &opts);
    let mut s = sim_top(db, &tb);
    let mut rng = Rng::new(if gated { 11 } else { 12 });
    let mut model = 0u64;
    // First cycle: reset.
    cycle(&mut s, &opts, 0, &[("RST", 1), ("LOAD", 0), ("D", 0xa5)]);
    assert_eq!(get(&s, "Q"), 0);
    for k in 1..80 {
        let d = rng.below(256);
        let load = rng.below(2);
        let rst = u64::from(rng.below(10) == 0);
        cycle(&mut s, &opts, k, &[("D", d), ("LOAD", load), ("RST", rst)]);
        if rst == 1 {
            model = 0;
        } else if load == 1 {
            model = d;
        }
        assert_eq!(get(&s, "Q"), model, "cycle {k}: D={d:#x} LOAD={load} RST={rst}");
    }
    assert_clean(&mut s);
}

#[test]
fn register_load_and_reset() {
    check_register(false);
}

#[test]
fn register_gated_load_and_reset() {
    check_register(true);
}

#[test]
fn register_internal_nets_are_probeable() {
    let mut db = DesignBuilder::new();
    let reg = register(&mut db, 4);
    let opts = TestbenchOptions::new(P);
    let tb = testbench_with(&mut db, &reg, &opts);
    let mut s = sim_top(db, &tb);
    let k0 = opts.first_edge_after_reset();
    cycle(&mut s, &opts, k0, &[("D", 9), ("LOAD", 1)]);
    assert_eq!(get(&s, "dut/q"), 9);
    assert_eq!(get(&s, "dut/d"), 9);
    assert_eq!(bit(&s, "dut/ff0/m"), One);
}

#[test]
fn register_gated_load_rising_in_high_phase_is_a_spurious_edge() {
    // Documents the gating constraint of `register_gated`: LOAD rising while CLK is high makes
    // `gclk` rise mid-phase, and the register captures D although no CLK edge occurred.
    let mut db = DesignBuilder::new();
    let reg = register_gated(&mut db, 4);
    let opts = TestbenchOptions::new(P);
    let tb = testbench_with(&mut db, &reg, &opts);
    let mut s = sim_top(db, &tb);
    let k0 = opts.first_edge_after_reset();
    cycle(&mut s, &opts, k0, &[("D", 5), ("LOAD", 1)]);
    cycle(&mut s, &opts, k0 + 1, &[("D", 9), ("LOAD", 0)]);
    assert_eq!(get(&s, "Q"), 5);
    // Raise LOAD in the middle of the high phase after edge k0+2.
    run_to(&mut s, opts.rising_edge(k0 + 2) + u64::from(P / 4));
    set(&mut s, "LOAD", 1);
    s.run(8);
    assert_eq!(get(&s, "Q"), 9, "spurious capture through the gated clock");
}
