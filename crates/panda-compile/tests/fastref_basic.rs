//! FastRef (SPEC §6.2) on small golden circuits, plus equivalence with Timing mode (SPEC §6.3).

use std::sync::Arc;

use panda_circuits::{
    TestbenchOptions, d_latch, d_latch_clr, ms_dff, nand_sr_latch, register, register_gated,
    ring_oscillator, ripple_counter, sync_counter, testbench, testbench_with, toggle_ff,
};
use panda_compile::{FastOptions, FastRef};
use panda_netlist::flat::FlatNetlist;
use panda_netlist::{DefBuilder, DefRef, DesignBuilder, DiagKind, Logic, Value, flatten};
use panda_timing::{TimingOptions, TimingSim};

fn flat_of(db: DesignBuilder, top: &DefRef) -> Arc<FlatNetlist> {
    Arc::new(flatten(&db.build(top)).expect("flatten"))
}

fn engine(flat: &Arc<FlatNetlist>) -> FastRef {
    FastRef::new(flat.clone(), FastOptions::default()).expect("FastRef::new")
}

fn bit(v: bool) -> Value {
    Value::from_bool(v)
}

fn pin_u64(fr: &FastRef, pin: &str) -> Option<u64> {
    fr.read_pin(pin).expect("pin").to_u64()
}

fn pin_logic(fr: &FastRef, pin: &str) -> Logic {
    let v = fr.read_pin(pin).expect("pin");
    assert_eq!(v.width(), 1);
    v.get(0)
}

/// Apply input changes and process them as one event.
fn apply(fr: &mut FastRef, changes: &[(&str, bool)]) {
    for &(p, v) in changes {
        fr.set_input(p, &bit(v)).unwrap();
    }
    let t = fr.tick();
    assert_eq!(
        fr.step_event(),
        Some(t),
        "input event is processed at the current tick"
    );
}

fn diag_kinds(fr: &mut FastRef) -> Vec<DiagKind> {
    fr.take_diagnostics().iter().map(|d| d.kind).collect()
}

// ---------------------------------------------------------------------------------------------
// Latches driven by inputs (no clock: input events only)
// ---------------------------------------------------------------------------------------------

#[test]
fn sr_latch_set_reset_hold() {
    let mut db = DesignBuilder::new();
    let top = nand_sr_latch(&mut db);
    let flat = flat_of(db, &top);
    let mut fr = engine(&flat);
    assert_eq!(fr.kernels().kernels.len(), 1);
    // Power-on inputs 0,0: both outputs 1.
    assert_eq!(pin_logic(&fr, "Q"), Logic::One);
    assert_eq!(pin_logic(&fr, "Q_n"), Logic::One);
    assert_eq!(fr.next_event_tick(), None, "no clock: nothing scheduled");
    assert_eq!(fr.step_event(), None);
    // Release reset first (S_n still 0): set.
    apply(&mut fr, &[("R_n", true)]);
    assert_eq!(
        (pin_logic(&fr, "Q"), pin_logic(&fr, "Q_n")),
        (Logic::One, Logic::Zero)
    );
    // Hold.
    apply(&mut fr, &[("S_n", true)]);
    assert_eq!(
        (pin_logic(&fr, "Q"), pin_logic(&fr, "Q_n")),
        (Logic::One, Logic::Zero)
    );
    // Reset.
    apply(&mut fr, &[("R_n", false)]);
    assert_eq!(
        (pin_logic(&fr, "Q"), pin_logic(&fr, "Q_n")),
        (Logic::Zero, Logic::One)
    );
    // Hold.
    apply(&mut fr, &[("R_n", true)]);
    assert_eq!(
        (pin_logic(&fr, "Q"), pin_logic(&fr, "Q_n")),
        (Logic::Zero, Logic::One)
    );
    // Set again.
    apply(&mut fr, &[("S_n", false)]);
    assert_eq!(
        (pin_logic(&fr, "Q"), pin_logic(&fr, "Q_n")),
        (Logic::One, Logic::Zero)
    );
    assert!(fr.take_diagnostics().is_empty());
    let st = fr.stats();
    assert!(
        st.table_hits >= 4,
        "binary transitions use the table: {st:?}"
    );
    // Internal nets are exposed too.
    assert_eq!(fr.read("q").unwrap().get(0), Logic::One);
    assert_eq!(fr.read("qn").unwrap().get(0), Logic::Zero);
}

#[test]
fn sr_latch_simultaneous_release_is_metastable() {
    let mut db = DesignBuilder::new();
    let top = nand_sr_latch(&mut db);
    let flat = flat_of(db, &top);
    let mut fr = engine(&flat);
    fr.take_diagnostics();
    // 00 → 11 at once: period-2 oscillation in Timing mode → collapse to X.
    apply(&mut fr, &[("S_n", true), ("R_n", true)]);
    assert_eq!(pin_logic(&fr, "Q"), Logic::X);
    assert_eq!(pin_logic(&fr, "Q_n"), Logic::X);
    let kinds = diag_kinds(&mut fr);
    assert!(kinds.contains(&DiagKind::Metastable), "{kinds:?}");
    // Recovers on a set.
    apply(&mut fr, &[("S_n", false)]);
    assert_eq!(
        (pin_logic(&fr, "Q"), pin_logic(&fr, "Q_n")),
        (Logic::One, Logic::Zero)
    );
}

#[test]
fn d_latch_transparent_and_opaque() {
    let mut db = DesignBuilder::new();
    let top = d_latch(&mut db);
    let flat = flat_of(db, &top);
    let mut fr = engine(&flat);
    // EN = 0 at power-on: the latch holds X.
    assert_eq!(pin_logic(&fr, "Q"), Logic::X);
    apply(&mut fr, &[("EN", true)]);
    assert_eq!(pin_logic(&fr, "Q"), Logic::Zero);
    assert_eq!(pin_logic(&fr, "Q_n"), Logic::One);
    apply(&mut fr, &[("D", true)]);
    assert_eq!(pin_logic(&fr, "Q"), Logic::One);
    apply(&mut fr, &[("EN", false)]);
    apply(&mut fr, &[("D", false)]);
    assert_eq!(pin_logic(&fr, "Q"), Logic::One, "opaque: holds");
    apply(&mut fr, &[("D", true)]);
    apply(&mut fr, &[("D", false)]);
    assert_eq!(pin_logic(&fr, "Q"), Logic::One);
    apply(&mut fr, &[("EN", true)]);
    assert_eq!(pin_logic(&fr, "Q"), Logic::Zero, "transparent again");
    assert_eq!(pin_logic(&fr, "Q_n"), Logic::One);
}

// ---------------------------------------------------------------------------------------------
// Clocked circuits
// ---------------------------------------------------------------------------------------------

/// Run to one tick before rising edge `k` (the values Timing mode settles to before that edge).
fn before_edge(fr: &mut FastRef, tb: &TestbenchOptions, k: u64) {
    fr.run_until(tb.rising_edge(k) - 1);
    assert!(fr.next_event_tick().is_none_or(|t| t >= tb.rising_edge(k)));
}

#[test]
fn ms_dff_with_clock() {
    let mut db = DesignBuilder::new();
    let ff = ms_dff(&mut db);
    let period = 16;
    let top = testbench(&mut db, &ff, period);
    let tb = TestbenchOptions::new(period);
    let flat = flat_of(db, &top);
    let mut fr = engine(&flat);
    assert_eq!(fr.kernels().kernels.len(), 2);
    assert_eq!(fr.next_event_tick(), Some(tb.rising_edge(0)));
    assert_eq!(pin_logic(&fr, "Q"), Logic::X, "before any edge");
    // D = 1 (queued, processed at tick 0).
    apply(&mut fr, &[("D", true)]);
    before_edge(&mut fr, &tb, 0);
    assert_eq!(pin_logic(&fr, "Q"), Logic::X);
    let data = [true, false, false, true, true, false, true];
    for (k, &d) in data.iter().enumerate() {
        // Sample the edge-k value of D = data[k-1] (set during the previous low phase).
        let k = k as u64 + 1;
        before_edge(&mut fr, &tb, k);
        let want = if k == 1 { true } else { data[k as usize - 2] };
        assert_eq!(
            pin_logic(&fr, "Q"),
            Logic::from_bool(want),
            "after edge {}",
            k - 1
        );
        assert_eq!(pin_logic(&fr, "Q_n"), Logic::from_bool(!want));
        // Change D in the low phase (just after the falling edge processed last).
        fr.set_input("D", &bit(d)).unwrap();
    }
    assert!(fr.take_diagnostics().is_empty());
    let st = fr.stats();
    assert!(
        st.max_rounds <= 2,
        "a master-slave FF costs one round per edge: {st:?}"
    );
}

#[test]
fn register8_load_and_hold() {
    let mut db = DesignBuilder::new();
    let reg = register(&mut db, 8);
    let period = 32;
    let top = testbench(&mut db, &reg, period);
    let tb = TestbenchOptions::new(period);
    let flat = flat_of(db, &top);
    let mut fr = engine(&flat);
    assert_eq!(fr.kernels().kernels.len(), 16);
    let first = tb.first_edge_after_reset();
    before_edge(&mut fr, &tb, first);
    assert_eq!(pin_u64(&fr, "Q"), Some(0), "synchronous reset");
    // Load 0xA5.
    fr.set_input("D", &Value::from_u64(8, 0xA5)).unwrap();
    fr.set_input("LOAD", &bit(true)).unwrap();
    before_edge(&mut fr, &tb, first + 1);
    assert_eq!(pin_u64(&fr, "Q"), Some(0xA5));
    // Hold while D changes.
    fr.set_input("LOAD", &bit(false)).unwrap();
    fr.set_input("D", &Value::from_u64(8, 0x3C)).unwrap();
    for k in 2..6 {
        before_edge(&mut fr, &tb, first + k);
        assert_eq!(pin_u64(&fr, "Q"), Some(0xA5));
    }
    fr.set_input("LOAD", &bit(true)).unwrap();
    before_edge(&mut fr, &tb, first + 6);
    assert_eq!(pin_u64(&fr, "Q"), Some(0x3C));
    assert!(fr.take_diagnostics().is_empty());
    let st = fr.stats();
    assert!(st.table_hits > st.sim_fallbacks, "{st:?}");
}

#[test]
fn sync_counter_counts() {
    let mut db = DesignBuilder::new();
    let c = sync_counter(&mut db, 4);
    let period = 32;
    let top = testbench(&mut db, &c, period);
    let tb = TestbenchOptions::new(period);
    let flat = flat_of(db, &top);
    let mut fr = engine(&flat);
    fr.set_input("EN", &bit(true)).unwrap();
    let first = tb.first_edge_after_reset();
    for k in first..first + 40 {
        before_edge(&mut fr, &tb, k);
        // Edges before `first` reset to 0; each later edge increments.
        let want = (k - first) % 16;
        assert_eq!(pin_u64(&fr, "Q"), Some(want), "before edge {k}");
    }
    // Disable counting.
    fr.set_input("EN", &bit(false)).unwrap();
    before_edge(&mut fr, &tb, first + 41);
    let held = pin_u64(&fr, "Q");
    before_edge(&mut fr, &tb, first + 45);
    assert_eq!(pin_u64(&fr, "Q"), held);
    assert!(fr.take_diagnostics().is_empty());
}

#[test]
fn ripple_counter_derived_clocks() {
    let mut db = DesignBuilder::new();
    let c = ripple_counter(&mut db, 5);
    let period = 64;
    let top = testbench(&mut db, &c, period);
    let tb = TestbenchOptions::new(period);
    let flat = flat_of(db, &top);
    let mut fr = engine(&flat);
    let first = tb.first_edge_after_reset();
    before_edge(&mut fr, &tb, first);
    assert_eq!(pin_u64(&fr, "Q"), Some(0), "asynchronously cleared");
    for k in first + 1..first + 70 {
        before_edge(&mut fr, &tb, k);
        assert_eq!(pin_u64(&fr, "Q"), Some((k - first) % 32), "before edge {k}");
    }
    let st = fr.stats();
    assert!(
        st.max_rounds >= 5,
        "carry ripples through derived clocks over several rounds: {st:?}"
    );
    assert!(fr.take_diagnostics().is_empty());
}

#[test]
fn toggle_ff_without_reset_stays_x() {
    let mut db = DesignBuilder::new();
    let t = toggle_ff(&mut db);
    let mut opts = TestbenchOptions::new(16);
    // No PowerOnReset: RST becomes a plain input (default 0).
    opts.reset_pin = "NO_RESET".into();
    let top = testbench_with(&mut db, &t, &opts);
    let flat = flat_of(db, &top);
    let mut fr = engine(&flat);
    fr.set_input("T", &bit(true)).unwrap();
    for k in 0..20 {
        before_edge(&mut fr, &opts, k);
        assert_eq!(pin_logic(&fr, "Q"), Logic::X, "before edge {k}");
    }
    // A synchronous reset fixes it; afterwards it toggles.
    fr.set_input("RST", &bit(true)).unwrap();
    before_edge(&mut fr, &opts, 21);
    assert_eq!(pin_logic(&fr, "Q"), Logic::Zero);
    fr.set_input("RST", &bit(false)).unwrap();
    for k in 22..30u64 {
        before_edge(&mut fr, &opts, k);
        assert_eq!(
            pin_logic(&fr, "Q"),
            Logic::from_bool((k - 21) % 2 == 1),
            "edge {k}"
        );
    }
}

#[test]
fn register_gated_clock() {
    let mut db = DesignBuilder::new();
    let reg = register_gated(&mut db, 8);
    let period = 32;
    let top = testbench(&mut db, &reg, period);
    let tb = TestbenchOptions::new(period);
    let flat = flat_of(db, &top);
    let mut fr = engine(&flat);
    let first = tb.first_edge_after_reset();
    before_edge(&mut fr, &tb, first);
    assert_eq!(pin_u64(&fr, "Q"), Some(0));
    let mut want = 0u64;
    let script = [
        (true, 0x11),
        (false, 0x22),
        (false, 0x33),
        (true, 0x44),
        (true, 0x55),
        (false, 0x66),
        (true, 0x77),
    ];
    for (i, &(load, d)) in script.iter().enumerate() {
        // Inputs change in the low phase (right after the falling edge).
        fr.set_input("LOAD", &bit(load)).unwrap();
        fr.set_input("D", &Value::from_u64(8, d)).unwrap();
        before_edge(&mut fr, &tb, first + 1 + i as u64);
        if load {
            want = d;
        }
        assert_eq!(pin_u64(&fr, "Q"), Some(want), "step {i}");
    }
    assert!(fr.take_diagnostics().is_empty());
}

// ---------------------------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------------------------

#[test]
fn ring_oscillator_goes_x_with_oscillating() {
    let mut db = DesignBuilder::new();
    let top = ring_oscillator(&mut db, 3, true);
    let flat = flat_of(db, &top);
    let mut fr = engine(&flat);
    fr.take_diagnostics();
    let en = flat
        .top_inputs
        .first()
        .map(|p| p.name.clone())
        .expect("enable pin");
    apply(&mut fr, &[(&en, true)]);
    let kinds = diag_kinds(&mut fr);
    assert!(kinds.contains(&DiagKind::Oscillating), "{kinds:?}");
    let k = &fr.kernels().kernels[0];
    assert!(k.nets.iter().all(|&n| fr.get(n) == Logic::X));
}

/// Two transparent latches in a loop with one inversion, through shared (non-absorbable) buffer
/// taps: the loop passes through two kernels, so it oscillates across rounds.
#[test]
fn oscillation_across_kernels_hits_delta_limit() {
    let mut db = DesignBuilder::new();
    let l1 = d_latch_clr(&mut db);
    let l2 = d_latch(&mut db);
    let mut b = DefBuilder::new("latch_loop");
    let en = b.input("EN", 1);
    let clr = b.input("CLR", 1);
    let q2 = b.wire(1);
    let q1 = b.inst_named("l1", &l1, &[q2, en, clr])[0];
    let mut x = q1;
    for i in 0..6 {
        x = b.buf(x);
        let tap = b.not(x);
        b.output(&format!("tap{i}"), tap);
    }
    let d2 = b.not(x);
    let o2 = b.inst_named("l2", &l2, &[d2, en]);
    b.connect(o2[0], q2);
    b.output("Q1", q1);
    let top = db.add(b.finish());
    let flat = flat_of(db, &top);
    let opts = FastOptions {
        delta_limit: 10,
        ..FastOptions::default()
    };
    let mut fr = FastRef::new(flat.clone(), opts).unwrap();
    assert_eq!(fr.kernels().kernels.len(), 2, "{:?}", fr.kernels().kernels);
    apply(&mut fr, &[("EN", true), ("CLR", true)]);
    assert_eq!(pin_logic(&fr, "Q1"), Logic::Zero);
    fr.take_diagnostics();
    apply(&mut fr, &[("CLR", false)]);
    let kinds = diag_kinds(&mut fr);
    assert!(kinds.contains(&DiagKind::IterationLimit), "{kinds:?}");
    assert_eq!(pin_logic(&fr, "Q1"), Logic::X);
    // The event terminated and later events still work: clear again.
    apply(&mut fr, &[("CLR", true)]);
    assert_eq!(pin_logic(&fr, "Q1"), Logic::Zero);
}

#[test]
fn contention_on_combinational_resolver() {
    let mut db = DesignBuilder::new();
    let mut b = DefBuilder::new("bus");
    let a = b.input("A", 1);
    let ea = b.input("EA", 1);
    let c = b.input("C", 1);
    let ec = b.input("EC", 1);
    let bus = b.wire(1);
    b.tri_to(a, ea, bus);
    b.tri_to(c, ec, bus);
    b.output("BUS", bus);
    let top = db.add(b.finish());
    let flat = flat_of(db, &top);
    let mut fr = engine(&flat);
    assert!(fr.kernels().kernels.is_empty());
    assert_eq!(pin_logic(&fr, "BUS"), Logic::Z);
    apply(&mut fr, &[("A", true), ("EA", true)]);
    assert_eq!(pin_logic(&fr, "BUS"), Logic::One);
    assert!(fr.take_diagnostics().is_empty());
    apply(&mut fr, &[("EC", true)]);
    assert_eq!(pin_logic(&fr, "BUS"), Logic::X);
    let kinds = diag_kinds(&mut fr);
    assert_eq!(kinds, vec![DiagKind::Contention]);
    apply(&mut fr, &[("C", true)]);
    assert_eq!(pin_logic(&fr, "BUS"), Logic::One);
    assert!(fr.take_diagnostics().is_empty());
}

#[test]
fn set_input_errors() {
    let mut db = DesignBuilder::new();
    let top = d_latch(&mut db);
    let flat = flat_of(db, &top);
    let mut fr = engine(&flat);
    assert!(fr.set_input("nope", &bit(true)).is_err());
    assert!(fr.set_input("D", &Value::from_u64(2, 1)).is_err());
    // Setting the same value queues nothing.
    fr.set_input("D", &bit(false)).unwrap();
    assert!(!fr.has_pending_inputs());
    fr.set_input("D", &bit(true)).unwrap();
    assert!(fr.has_pending_inputs());
    // Reset folds pending inputs into the tick-0 event and keeps them.
    fr.reset();
    assert!(!fr.has_pending_inputs());
    assert_eq!(fr.input_value("D"), Some(&bit(true)));
    assert_eq!(fr.tick(), 0);
}

// ---------------------------------------------------------------------------------------------
// Equivalence with Timing mode (SPEC §6.3)
// ---------------------------------------------------------------------------------------------

/// Run `events` FastRef events and compare every net with Timing mode at `next_event_tick - 1`.
/// `inputs` are applied (to both engines) before power-on.
fn assert_equivalent(flat: &Arc<FlatNetlist>, inputs: &[(&str, Value)], events: usize) {
    let mut ts = TimingSim::new(flat.clone(), TimingOptions::default());
    let mut fr = engine(flat);
    for (p, v) in inputs {
        ts.set_input(p, v).unwrap();
        fr.set_input(p, v).unwrap();
    }
    ts.reset();
    fr.reset();
    for _ in 0..events {
        let next = fr.next_event_tick().expect("clocked design");
        let target = next - 1;
        assert!(ts.tick() <= target);
        ts.run(target - ts.tick());
        assert_eq!(ts.tick(), target);
        let diffs: Vec<String> = (0..flat.num_nets as usize)
            .filter(|&n| ts.values()[n] != fr.values()[n])
            .map(|n| {
                format!(
                    "{}: timing {} fast {}",
                    flat.net_names[n],
                    ts.values()[n],
                    fr.values()[n]
                )
            })
            .collect();
        assert!(
            diffs.is_empty(),
            "mismatch at tick {target} (event at {}): {diffs:#?}",
            fr.tick()
        );
        fr.step_event();
    }
}

#[test]
fn equivalence_sync_counter() {
    let mut db = DesignBuilder::new();
    let c = sync_counter(&mut db, 4);
    let top = testbench(&mut db, &c, 32);
    let flat = flat_of(db, &top);
    assert_equivalent(&flat, &[("EN", bit(true))], 80);
}

#[test]
fn equivalence_ripple_counter() {
    let mut db = DesignBuilder::new();
    let c = ripple_counter(&mut db, 4);
    let top = testbench(&mut db, &c, 64);
    let flat = flat_of(db, &top);
    assert_equivalent(&flat, &[], 80);
}

#[test]
fn equivalence_register_gated() {
    let mut db = DesignBuilder::new();
    let r = register_gated(&mut db, 4);
    let top = testbench(&mut db, &r, 32);
    let flat = flat_of(db, &top);
    assert_equivalent(
        &flat,
        &[("LOAD", bit(true)), ("D", Value::from_u64(4, 0b1010))],
        30,
    );
}
