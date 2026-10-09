//! Regression tests for bugs found in the adversarial review of the timing engine.

mod common;

use std::sync::Arc;

use common::*;
use panda_netlist::flat::{FlatGate, FlatNetlist, FlatOp, Source, SourceKind};
use panda_netlist::{ComponentDef, DefBuilder, DefRef, DiagKind, Logic, Termination};
use panda_timing::{TimingOptions, TimingSim};

use Logic::{One, X, Z, Zero};

fn kinds(sim: &mut TimingSim) -> Vec<DiagKind> {
    sim.take_diagnostics().into_iter().map(|d| d.kind).collect()
}

fn net(sim: &TimingSim, name: &str) -> u32 {
    sim.netlist().net_by_name(name).unwrap_or_else(|| panic!("no net {name}"))
}

fn not_loop() -> ComponentDef {
    let mut b = DefBuilder::new("top");
    let y = b.wire(1);
    b.not_to(y, y);
    b.name(y, "y");
    b.output("Y", y);
    b.finish()
}

/// Bug: the per-net history packed `tick << 16`, silently dropping the high bits of ticks
/// ≥ 2^48 (reachable through `run`'s fast-forward). Collapse then never fired and
/// `last_changed` could list a net twice.
#[test]
fn collapse_still_works_beyond_2_pow_48_ticks() {
    let mut s = sim(not_loop(), vec![]);
    s.run((1u64 << 48) + 12_345);
    assert_eq!(s.tick(), (1u64 << 48) + 12_345);
    let y = net(&s, "y");
    s.force_net(y, Zero);
    for _ in 0..7 {
        s.step();
        assert!(bit(&s, "y").is_known());
        assert_eq!(s.last_changed(), &[y]);
    }
    s.step();
    assert_eq!(bit(&s, "y"), X, "collapses at the 8th alternation");
    assert_eq!(kinds(&mut s), vec![DiagKind::Metastable]);
}

/// Bug: a clock with `high == 0` or `high >= period` is constant, but was treated as running
/// forever, so `settle` and `run`'s fast-forward never applied. (Such clocks are rejected by
/// validation, so this uses a raw flat netlist.)
#[test]
fn constant_clocks_are_quiescent() {
    for (period, high) in [(4, 0), (4, 4), (4, 9), (1, 1), (0, 0)] {
        let nl = FlatNetlist {
            num_nets: 2,
            gates: vec![FlatGate { op: FlatOp::Not, inputs: vec![0], output: 1, owner: 0, group: 0, bit: 0 }],
            sources: vec![Source { kind: SourceKind::Clock { period, high, phase: 1 }, output: 0 }],
            net_names: vec!["c".into(), "n".into()],
            ..Default::default()
        };
        let mut s = TimingSim::new(Arc::new(nl), TimingOptions::default());
        assert!(s.settle(10), "clock({period}, {high}) is constant");
        let expect = if high == 0 { One } else { Zero };
        assert_eq!(s.get(1), expect);
        s.run(1 << 40);
        assert_eq!(s.tick(), 1 + (1 << 40));
        assert_eq!(s.get(1), expect);
    }
}

/// Advancing past `u64::MAX` panics instead of wrapping.
#[test]
#[should_panic(expected = "tick counter overflow")]
fn tick_overflow_panics() {
    let mut s = sim(not_loop(), vec![]);
    s.run(u64::MAX);
    assert_eq!(s.tick(), u64::MAX);
    s.step();
}

/// Bug (false positive): the parity of a 3-inverter ring's taps alternates every tick, but that
/// is a real oscillation of period 6, not metastability. It used to be collapsed to X every
/// 8 ticks, with a `Metastable` diagnostic each episode.
#[test]
fn parity_of_ring_oscillator_is_not_collapsed() {
    let mut b = DefBuilder::new("top");
    let a = b.wire(1);
    let bb = b.not(a);
    let c = b.not(bb);
    b.not_to(c, a);
    b.name(a, "a");
    let y = b.xor(&[a, bb, c]);
    b.name(y, "y");
    b.output("Y", y);
    let opts = TimingOptions::default().with_osc_limit(50);
    let mut s = sim_with(b.finish(), vec![], opts);
    assert!(s.settle(10), "power-on X ring is quiescent");
    let a = net(&s, "a");
    s.force_net(a, Zero);
    s.run(10);
    let mut prev = bit(&s, "y");
    for _ in 0..200 {
        s.step();
        let v = bit(&s, "y");
        assert!(v.is_known(), "y must never be forced to X");
        assert_ne!(v, prev, "y alternates every tick");
        prev = v;
    }
    assert_eq!(kinds(&mut s), vec![DiagKind::Oscillating], "oscillating, not metastable");
}

/// Bug (false positive): a glitch train passing through a transparent D latch made the latch's
/// loop nets alternate for 8+ ticks; they were collapsed to X (and recovered), raising a spurious
/// `Metastable`.
#[test]
fn glitch_train_through_transparent_latch_is_not_collapsed() {
    let latch = DefRef::of(&d_latch());
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 1);
    let e = b.input("E", 1);
    let mut taps = vec![a];
    for _ in 0..13 {
        let last = *taps.last().unwrap();
        taps.push(b.buf(last));
    }
    let d = b.xor(&taps);
    let o = b.inst_named("l", &latch, &[d, e]);
    b.output("Q", o[0]);
    let mut s = sim(b.finish(), vec![d_latch()]);
    set(&mut s, "E", true);
    assert!(s.settle(100));
    assert_eq!(bit(&s, "Q"), Zero);
    set(&mut s, "A", true);
    let mut ones = 0;
    while !s.is_quiescent() {
        s.step();
        let q = bit(&s, "Q");
        assert!(q.is_known(), "Q must not be collapsed at tick {}", s.tick());
        ones += usize::from(q == One);
    }
    assert!(ones >= 5, "the glitch train reaches Q");
    assert_eq!(bit(&s, "Q"), Zero);
    assert!(kinds(&mut s).is_empty());
}

/// Genuine metastability inside a design that also has unrelated, non-alternating activity
/// elsewhere: the latch still collapses (the closure only looks at the latch's own fan-in).
#[test]
fn metastable_latch_collapses_next_to_unrelated_activity() {
    let sr = DefRef::of(&nand_sr("sr"));
    let mut b = DefBuilder::new("top");
    let s_n = b.input("S_n", 1);
    let r_n = b.input("R_n", 1);
    let o = b.inst_named("l", &sr, &[s_n, r_n]);
    b.output("Q", o[0]);
    // An independent 5-inverter ring (period 10), started by forcing.
    let a = b.wire(1);
    let mut cur = a;
    for _ in 0..4 {
        cur = b.not(cur);
    }
    b.not_to(cur, a);
    b.name(a, "ring");
    let mut s = sim(b.finish(), vec![nand_sr("sr")]);
    set(&mut s, "S_n", false);
    set(&mut s, "R_n", false);
    assert!(s.settle(20));
    let ring = net(&s, "ring");
    s.force_net(ring, One);
    s.run(20);
    set(&mut s, "S_n", true);
    set(&mut s, "R_n", true);
    s.run(9);
    assert_eq!(bit(&s, "Q"), X);
    assert!(kinds(&mut s).contains(&DiagKind::Metastable));
}

/// A self-sustained loop that collapsed recovers as soon as an input change breaks it.
#[test]
fn collapse_then_recovery_on_input_change() {
    let mut b = DefBuilder::new("top");
    let en = b.input("EN", 1);
    let y = b.wire(1);
    b.nand_to(&[en, y], y); // EN=1: NOT self-loop; EN=0: y=1
    b.name(y, "y");
    let mut s = sim(b.finish(), vec![]);
    assert!(s.settle(10));
    assert_eq!(bit(&s, "y"), One);
    set(&mut s, "EN", true);
    s.run(30);
    assert_eq!(bit(&s, "y"), X);
    assert_eq!(kinds(&mut s), vec![DiagKind::Metastable]);
    set(&mut s, "EN", false);
    assert!(s.settle(10));
    assert_eq!(bit(&s, "y"), One);
    // And again: a new episode reports again.
    set(&mut s, "EN", true);
    s.run(30);
    assert_eq!(bit(&s, "y"), X);
    assert_eq!(kinds(&mut s), vec![DiagKind::Metastable]);
}

/// Bug: `force_net` did not count as a source change, did not record history (so an earlier
/// alternation run continued across the force) and left `last_changed` inconsistent; forced
/// constant / input nets kept their forced value until an unrelated input changed.
#[test]
fn force_net_semantics() {
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 1);
    let k = b.lit(1, 1);
    let na = b.not(a);
    b.name(na, "na");
    let nk = b.not(k);
    b.name(nk, "nk");
    let bus = b.wire(1);
    b.tri_to(a, k, bus);
    b.terminate(bus, Termination::PullDown);
    b.name(bus, "bus");
    let nb = b.not(bus);
    b.name(nb, "nb");
    let mut s = sim(b.finish(), vec![]);
    assert!(s.settle(10));
    let k_net = s
        .netlist()
        .sources
        .iter()
        .find(|src| matches!(src.kind, SourceKind::Constant(One)))
        .expect("constant source")
        .output;
    let (a_net, na_net, bus_net) = (net(&s, "A"), net(&s, "na"), net(&s, "bus"));
    assert_eq!(s.get(k_net), One);

    // Gate output: persists until the gate is re-evaluated.
    s.force_net(na_net, Zero);
    assert!(s.last_changed().contains(&na_net), "the forced change is listed");
    assert!(s.settle(10));
    assert_eq!(bit(&s, "na"), Zero, "forced gate output persists");
    set(&mut s, "A", true);
    s.run(2);
    assert_eq!(bit(&s, "na"), Zero, "re-evaluated: NOT(1)");
    set(&mut s, "A", false);
    s.run(2);
    assert_eq!(bit(&s, "na"), One);

    // Source nets get their source value back at the next tick.
    s.force_net(k_net, Zero);
    assert!(!s.is_quiescent(), "a forced source will be restored");
    s.step();
    assert_eq!(s.get(k_net), One, "constant restored");
    s.force_net(a_net, One);
    s.step();
    assert_eq!(s.get(a_net), Zero, "input restored");
    assert!(s.settle(10));
    assert_eq!(bit(&s, "nk"), Zero);
    assert_eq!(bit(&s, "na"), One);

    // Resolver output: persists until a driver changes; resolvers reading a forced net recompute
    // at once.
    s.force_net(bus_net, Z);
    assert!(s.settle(10));
    assert_eq!(bit(&s, "bus"), Z);
    assert_eq!(bit(&s, "nb"), X);
    set(&mut s, "A", true);
    s.run(3);
    assert_eq!(bit(&s, "bus"), One);
    assert_eq!(bit(&s, "nb"), Zero);
    assert!(kinds(&mut s).is_empty());
}

/// Forcing is a source change: an alternation run that started before a force does not carry
/// over, and the collapse window restarts.
#[test]
fn force_restarts_the_collapse_window() {
    let mut s = sim(not_loop(), vec![]);
    s.step();
    let y = net(&s, "y");
    s.force_net(y, Zero);
    s.run(6); // 6 alternations
    let flipped = panda_netlist::eval::not1(bit(&s, "y"));
    s.force_net(y, flipped);
    let t0 = s.tick();
    while bit(&s, "y") != X {
        s.step();
        assert!(s.tick() <= t0 + 8);
    }
    assert_eq!(s.tick(), t0 + 8, "a full window after the (last) force");
}

/// `run`'s fast-forward ends the activity episode exactly like stepping does (it used not to;
/// combined with the old `force_net`, which was not a source change, an oscillation restarted
/// by forcing after a fast-forward could go unreported).
#[test]
fn oscillation_reported_again_after_fast_forward() {
    let ring = || {
        let mut b = DefBuilder::new("top");
        let a = b.wire(1);
        let bb = b.not(a);
        let c = b.not(bb);
        b.not_to(c, a);
        b.name(a, "a");
        b.name(bb, "bb");
        b.name(c, "c");
        b.finish()
    };
    let opts = TimingOptions::default().with_osc_limit(40);
    for fast in [false, true] {
        let mut s = sim_with(ring(), vec![], opts.clone());
        let a = net(&s, "a");
        assert!(s.settle(10));
        s.force_net(a, Zero);
        s.run(100);
        assert_eq!(kinds(&mut s), vec![DiagKind::Oscillating]);
        // Kill the ring with X everywhere; it becomes quiescent.
        for n in ["a", "bb", "c"] {
            let n = net(&s, n);
            s.force_net(n, X);
        }
        if fast {
            s.run(1000);
        } else {
            for _ in 0..1000 {
                s.step();
            }
        }
        assert!(s.is_quiescent());
        s.force_net(a, One);
        s.run(100);
        assert_eq!(kinds(&mut s), vec![DiagKind::Oscillating], "fast-forward = {fast}");
    }
}

/// Bug: a mis-sized NOT/BUF/TRI compiled to an all-zero table and was then misread as a wide
/// gate (out-of-bounds index or garbage). It is now rejected up front.
#[test]
#[should_panic(expected = "Not with 2 inputs")]
fn malformed_gate_arity_is_rejected() {
    let nl = FlatNetlist {
        num_nets: 3,
        gates: vec![FlatGate { op: FlatOp::Not, inputs: vec![0, 1], output: 2, owner: 0, group: 0, bit: 0 }],
        net_names: vec!["a".into(), "b".into(), "c".into()],
        ..Default::default()
    };
    let _ = TimingSim::new(Arc::new(nl), TimingOptions::default());
}

/// A clocked design whose clock has period 2 toggles nets every tick, but sources change every
/// tick, so nothing collapses (and nothing is reported).
#[test]
fn clocked_divider_with_fast_clock_does_not_collapse() {
    let dff = DefRef::of(&ms_dff());
    let mut b = DefBuilder::new("top");
    let clk = b.clock(2, 1, 0);
    let rst = b.power_on_reset(6);
    let q = b.wire(1);
    let nq = b.not(q);
    let nrst = b.not(rst);
    let d = b.and(&[nq, nrst]);
    let o = b.inst_named("ff", &dff, &[d, clk]);
    b.connect(o[0], q);
    b.output("Q", q);
    let mut s = sim(b.finish(), dff_defs());
    s.run(200);
    assert!(!kinds(&mut s).contains(&DiagKind::Metastable));
}

/// A slower clock (period 40) driving a toggle flip-flop: between edges there are long quiet
/// windows, and the divider still works without any collapse.
#[test]
fn toggle_flip_flop_with_slow_clock_divides_by_two() {
    let dff = DefRef::of(&ms_dff());
    let mut b = DefBuilder::new("top");
    let clk = b.clock(40, 20, 0);
    let rst = b.power_on_reset(50);
    let q = b.wire(1);
    let nq = b.not(q);
    let nrst = b.not(rst);
    let d = b.and(&[nq, nrst]);
    let o = b.inst_named("ff", &dff, &[d, clk]);
    b.connect(o[0], q);
    b.output("Q", q);
    let mut s = sim(b.finish(), dff_defs());
    s.run(100);
    let mut wave = Vec::new();
    for _ in 0..8 {
        s.run(40);
        wave.push(bit(&s, "Q"));
    }
    assert!(wave.iter().all(|v| v.is_known()), "{wave:?}");
    for w in wave.windows(2) {
        assert_ne!(w[0], w[1], "toggles every clock period: {wave:?}");
    }
    assert!(kinds(&mut s).is_empty());
}
