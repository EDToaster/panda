//! Golden sequential circuits (SPEC §13 M1 exit criterion): latches, flip-flops, counters.

mod common;

use common::*;
use panda_netlist::{DefBuilder, DefRef, DiagKind, Logic, Value};
use panda_timing::TimingSim;

use Logic::{One, Zero, X};

fn kinds(sim: &mut TimingSim) -> Vec<DiagKind> {
    sim.take_diagnostics().into_iter().map(|d| d.kind).collect()
}

fn nand_sr_top() -> TimingSim {
    sim(nand_sr("sr"), vec![])
}

#[test]
fn nand_sr_latch_power_on_is_x_when_holding() {
    let mut s = nand_sr_top();
    set(&mut s, "S_n", true);
    set(&mut s, "R_n", true);
    s.reset();
    assert!(s.settle(100));
    assert_eq!((bit(&s, "Q"), bit(&s, "Q_n")), (X, X), "power-on state of a holding latch is X");
    assert!(kinds(&mut s).is_empty());
}

#[test]
fn nand_sr_latch_set_reset_hold() {
    let mut s = nand_sr_top();
    // Default inputs are 0/0: both outputs high.
    assert!(s.settle(100));
    assert_eq!((bit(&s, "Q"), bit(&s, "Q_n")), (One, One));

    // Set (S_n=0, R_n=1).
    set(&mut s, "R_n", true);
    assert!(s.settle(100));
    assert_eq!((bit(&s, "Q"), bit(&s, "Q_n")), (One, Zero));
    // Hold.
    set(&mut s, "S_n", true);
    assert!(s.settle(100));
    assert_eq!((bit(&s, "Q"), bit(&s, "Q_n")), (One, Zero));
    // Reset.
    set(&mut s, "R_n", false);
    assert!(s.settle(100));
    assert_eq!((bit(&s, "Q"), bit(&s, "Q_n")), (Zero, One));
    // Hold.
    set(&mut s, "R_n", true);
    assert!(s.settle(100));
    assert_eq!((bit(&s, "Q"), bit(&s, "Q_n")), (Zero, One));
    // Set again.
    set(&mut s, "S_n", false);
    assert!(s.settle(100));
    assert_eq!((bit(&s, "Q"), bit(&s, "Q_n")), (One, Zero));
    assert!(kinds(&mut s).is_empty());
}

#[test]
fn nand_sr_latch_set_timing() {
    let mut s = nand_sr_top();
    set(&mut s, "R_n", true);
    s.settle(100);
    set(&mut s, "S_n", true);
    s.settle(100);
    // Q=1, Qn=0. Reset: R_n falls at tick t0+1, Qn rises at t0+2, Q falls at t0+3.
    let t0 = s.tick();
    set(&mut s, "R_n", false);
    s.step();
    assert_eq!(s.tick(), t0 + 1);
    assert_eq!(bit(&s, "R_n"), Zero);
    assert_eq!((bit(&s, "Q"), bit(&s, "Q_n")), (One, Zero));
    s.step();
    assert_eq!((bit(&s, "Q"), bit(&s, "Q_n")), (One, One));
    s.step();
    assert_eq!((bit(&s, "Q"), bit(&s, "Q_n")), (Zero, One));
    assert!(s.settle(10));
}

#[test]
fn nand_sr_latch_simultaneous_release_collapses_to_x() {
    let mut s = nand_sr_top();
    assert!(s.settle(100));
    assert_eq!((bit(&s, "Q"), bit(&s, "Q_n")), (One, One));
    kinds(&mut s);

    // Release 00 -> 11 on the same tick.
    s.set_input("S_n", &Value::from_bool(true)).unwrap();
    s.set_input("R_n", &Value::from_bool(true)).unwrap();
    s.step();
    let release = s.tick();
    // Period-2 oscillation for 2 * p2_limit = 8 ticks, then collapse.
    let mut seen = Vec::new();
    for _ in 0..7 {
        s.step();
        let q = bit(&s, "Q");
        assert_eq!(q, bit(&s, "Q_n"), "symmetric oscillation");
        assert!(q.is_known());
        seen.push(q);
    }
    assert_eq!(seen, vec![Zero, One, Zero, One, Zero, One, Zero]);
    assert!(s.take_diagnostics().is_empty());
    s.step();
    assert_eq!(s.tick(), release + 8);
    assert_eq!((bit(&s, "Q"), bit(&s, "Q_n")), (X, X));
    let d = s.take_diagnostics();
    assert_eq!(d.len(), 1, "{d:?}");
    assert_eq!(d[0].kind, DiagKind::Metastable);

    // Stable at X/X afterwards, no further diagnostics.
    assert!(s.settle(10));
    s.run(100);
    assert_eq!((bit(&s, "Q"), bit(&s, "Q_n")), (X, X));
    assert!(kinds(&mut s).is_empty());

    // A clean set recovers the latch.
    set(&mut s, "S_n", false);
    assert!(s.settle(100));
    assert_eq!((bit(&s, "Q"), bit(&s, "Q_n")), (One, Zero));
    assert!(kinds(&mut s).is_empty());
}

#[test]
fn p2_limit_controls_collapse_delay() {
    for limit in [1u32, 2, 4, 7] {
        let mut s = sim_with(nand_sr("sr"), vec![], panda_timing::TimingOptions::default().with_p2_limit(limit));
        s.settle(100);
        set(&mut s, "S_n", true);
        set(&mut s, "R_n", true);
        s.step();
        let release = s.tick();
        while bit(&s, "Q") != X {
            s.step();
            assert!(s.tick() < release + 100);
        }
        assert_eq!(s.tick(), release + 2 * u64::from(limit), "p2_limit {limit}");
    }
    // p2_limit = 0 disables collapse: the latch oscillates forever.
    let opts = panda_timing::TimingOptions::default().with_p2_limit(0).with_osc_limit(50);
    let mut s = sim_with(nand_sr("sr"), vec![], opts);
    s.settle(100);
    set(&mut s, "S_n", true);
    set(&mut s, "R_n", true);
    s.run(200);
    assert!(bit(&s, "Q").is_known());
    assert_eq!(kinds(&mut s), vec![DiagKind::Oscillating]);
}

#[test]
fn nor_sr_latch() {
    let mut s = sim(nor_sr("nsr"), vec![]);
    // S=R=0 from power-on: hold X.
    assert!(s.settle(100));
    assert_eq!((bit(&s, "Q"), bit(&s, "Q_n")), (X, X));
    set(&mut s, "S", true);
    assert!(s.settle(100));
    assert_eq!((bit(&s, "Q"), bit(&s, "Q_n")), (One, Zero));
    set(&mut s, "S", false);
    assert!(s.settle(100));
    assert_eq!((bit(&s, "Q"), bit(&s, "Q_n")), (One, Zero));
    set(&mut s, "R", true);
    assert!(s.settle(100));
    assert_eq!((bit(&s, "Q"), bit(&s, "Q_n")), (Zero, One));
    set(&mut s, "R", false);
    assert!(s.settle(100));
    assert_eq!((bit(&s, "Q"), bit(&s, "Q_n")), (Zero, One));
    // S=R=1 forces both low; simultaneous release collapses.
    set(&mut s, "S", true);
    set(&mut s, "R", true);
    assert!(s.settle(100));
    assert_eq!((bit(&s, "Q"), bit(&s, "Q_n")), (Zero, Zero));
    assert!(kinds(&mut s).is_empty());
    set(&mut s, "S", false);
    set(&mut s, "R", false);
    assert!(s.settle(100));
    assert_eq!((bit(&s, "Q"), bit(&s, "Q_n")), (X, X));
    assert_eq!(kinds(&mut s), vec![DiagKind::Metastable]);
}

#[test]
fn gated_d_latch() {
    let mut s = sim(d_latch(), vec![]);
    // E=0 at power-on: opaque, X.
    assert!(s.settle(100));
    assert_eq!(bit(&s, "Q"), X);
    // Transparent: follows D.
    set(&mut s, "E", true);
    assert!(s.settle(100));
    assert_eq!((bit(&s, "Q"), bit(&s, "Q_n")), (Zero, One));
    set(&mut s, "D", true);
    assert!(s.settle(100));
    assert_eq!((bit(&s, "Q"), bit(&s, "Q_n")), (One, Zero));
    set(&mut s, "D", false);
    assert!(s.settle(100));
    assert_eq!(bit(&s, "Q"), Zero);
    set(&mut s, "D", true);
    assert!(s.settle(100));
    // Close, then D changes are ignored.
    set(&mut s, "E", false);
    assert!(s.settle(100));
    assert_eq!(bit(&s, "Q"), One);
    for d in [false, true, false] {
        set(&mut s, "D", d);
        assert!(s.settle(100));
        assert_eq!((bit(&s, "Q"), bit(&s, "Q_n")), (One, Zero));
    }
    // Re-open with D=0.
    set(&mut s, "E", true);
    assert!(s.settle(100));
    assert_eq!(bit(&s, "Q"), Zero);
    assert!(kinds(&mut s).is_empty());
}

/// Top: MS-DFF with a clock (period 20, high 10, rising at ticks 5, 25, 45, ...) and input `D`.
fn dff_top() -> TimingSim {
    let dff = DefRef::of(&ms_dff());
    let mut b = DefBuilder::new("top");
    let d = b.input("D", 1);
    let clk = b.clock(20, 10, 5);
    b.name(clk, "clk");
    let o = b.inst_named("ff", &dff, &[d, clk]);
    b.output("Q", o[0]);
    b.output("Q_n", o[1]);
    sim(b.finish(), dff_defs())
}

#[test]
fn ms_dff_captures_on_rising_edge() {
    let mut s = dff_top();
    // Clock: rising at 5 + 20k, falling at 15 + 20k.
    s.run(3);
    assert_eq!(bit(&s, "clk"), Zero);
    // No rising edge seen yet with defined data... the edge at tick 5 captures D=0.
    s.run(10 - s.tick()); // tick 10
    assert_eq!(bit(&s, "Q"), Zero);
    assert_eq!(bit(&s, "Q_n"), One);

    // Pattern of D values, each applied mid-low-phase (tick 20k + 20), sampled before the next edge.
    let pattern = [true, true, false, true, false, false, true];
    for (k, &d) in pattern.iter().enumerate() {
        let k = k as u64;
        let t_set = 20 * k + 19; // clock low (low phase is 15..25 mod 20)
        s.run(t_set - s.tick());
        let before = bit(&s, "Q");
        set(&mut s, "D", d);
        // Still before the edge at 20k+25: Q unchanged.
        s.run(20 * k + 25 - s.tick());
        assert_eq!(bit(&s, "clk"), One);
        assert_eq!(bit(&s, "Q"), before, "Q must not change before the edge");
        // A few ticks after the edge, Q = D.
        s.run(5);
        assert_eq!(bit(&s, "Q"), Logic::from_bool(d), "edge {k}");
        assert_eq!(bit(&s, "Q_n"), Logic::from_bool(!d), "edge {k}");
        // D changes while the clock is high are ignored.
        set(&mut s, "D", !d);
        s.run(5);
        assert_eq!(bit(&s, "Q"), Logic::from_bool(d), "edge {k}: high-phase D change ignored");
        set(&mut s, "D", d);
    }
    assert!(kinds(&mut s).is_empty());
}

/// Top: a [`counter4`] clocked with period 20 (rising at 0, 20, 40, ...), reset by PowerOnReset(35).
fn counter_top() -> TimingSim {
    let ctr = DefRef::of(&counter4());
    let mut b = DefBuilder::new("top");
    let clk = b.clock(20, 10, 0);
    b.name(clk, "clk");
    let rst = b.power_on_reset(35);
    b.name(rst, "rst");
    let o = b.inst_named("ctr", &ctr, &[clk, rst]);
    b.output("COUNT", o[0]);
    let mut subs = dff_defs();
    subs.push(counter4());
    sim(b.finish(), subs)
}

#[test]
fn counter_counts_after_power_on_reset() {
    let mut s = counter_top();
    // Before the first rising edge (tick 20) the state is X.
    s.run(15);
    assert!(!s.read_pin("COUNT").unwrap().is_fully_known());
    // The edge at 20 loads 0 (reset asserted until tick 35).
    s.run(35 - s.tick());
    assert_eq!(s.read_pin("COUNT").unwrap().to_u64(), Some(0));
    assert_eq!(bit(&s, "rst"), Zero);
    // Then counts once per rising edge at 40, 60, ... and wraps modulo 16.
    for k in 2u64..40 {
        let t = 20 * k + 15;
        s.run(t - s.tick());
        let v = s.read_pin("COUNT").unwrap();
        assert_eq!(v.to_u64(), Some((k - 1) % 16), "at tick {t}: {v}");
        // Hierarchical reads agree with the pin.
        assert_eq!(s.read("ctr/Q").unwrap(), v);
        assert_eq!(s.read("COUNT").unwrap(), v);
        assert_eq!(s.read("ctr/next").unwrap().to_u64(), Some(k % 16));
        assert_eq!(s.read("ctr/ff0/Q").unwrap().get(0), v.get(0));
    }
    let d = s.take_diagnostics();
    assert!(d.is_empty(), "{d:?}");
}

#[test]
fn counter_reset_restarts() {
    let mut s = counter_top();
    s.run(20 * 7 + 15);
    assert_eq!(s.read_pin("COUNT").unwrap().to_u64(), Some(6));
    s.reset();
    assert_eq!(s.tick(), 0);
    assert!(!s.read_pin("COUNT").unwrap().is_fully_known());
    s.run(20 * 7 + 15);
    assert_eq!(s.read_pin("COUNT").unwrap().to_u64(), Some(6));
}

#[test]
fn unreset_toggle_ff_stays_x() {
    let dff = DefRef::of(&ms_dff());
    let mut b = DefBuilder::new("top");
    let clk = b.clock(16, 8, 7);
    let qn = b.wire(1);
    let o = b.inst_named("tff", &dff, &[qn, clk]);
    b.connect(o[1], qn);
    b.output("Q", o[0]);
    let mut s = sim(b.finish(), dff_defs());
    for _ in 0..50 {
        s.run(16);
        assert_eq!(bit(&s, "Q"), X);
        assert_eq!(bit(&s, "tff/Q_n"), X);
    }
    assert!(s.take_diagnostics().is_empty());

    // Forcing the slave state to a known value makes it toggle (power-on randomization use case).
    // After one tick the slave's steering NANDs are known (clock low), so the forced state holds.
    let nl = s.netlist().clone();
    s.reset();
    s.step();
    for (name, v) in [("tff/slave/q", Zero), ("tff/slave/qn", One)] {
        s.force_net(nl.net_by_name(name).unwrap(), v);
    }
    // Clock rises at 7, 23, 39, ...; sample mid-cycle.
    s.run(14);
    let mut last = bit(&s, "Q");
    assert!(last.is_known(), "forced state propagates");
    for _ in 0..10 {
        s.run(16);
        let q = bit(&s, "Q");
        assert!(q.is_known());
        assert_ne!(q, last, "toggles every cycle");
        last = q;
    }
}
