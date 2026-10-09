//! Engine semantics: oscillation and collapse rules, resolvers, sources, naming, settle/reset.

mod common;

use common::*;
use panda_netlist::{ComponentDef, DefBuilder, DiagKind, Logic, Termination, Value};
use panda_timing::{TimingOptions, TimingSim};

use Logic::{One, X, Z, Zero};

fn kinds(sim: &mut TimingSim) -> Vec<DiagKind> {
    sim.take_diagnostics().into_iter().map(|d| d.kind).collect()
}

/// Ring oscillator: `a = NAND(EN, last)` followed by `n_not` inverters (`a`, `b1`, `b2`, ...).
fn ring(n_not: usize) -> ComponentDef {
    let mut b = DefBuilder::new("ring");
    let en = b.input("EN", 1);
    let fb = b.wire(1);
    let a = b.nand(&[en, fb]);
    b.name(a, "a");
    let mut cur = a;
    for i in 0..n_not {
        let next = if i + 1 == n_not { fb } else { b.wire(1) };
        b.not_to(cur, next);
        b.name(next, &format!("b{}", i + 1));
        cur = next;
    }
    b.output("OUT", a);
    b.finish()
}

#[test]
fn ring_oscillator_three_stages_oscillates_without_collapse() {
    let opts = TimingOptions::default().with_osc_limit(100);
    let mut s = sim_with(ring(2), vec![], opts);
    // Disabled: settles at a=1.
    assert!(s.settle(100));
    assert_eq!(bit(&s, "a"), One);
    set(&mut s, "EN", true);
    s.run(10);
    // Period 6: every net repeats every 6 ticks and toggles every 3 ticks; never X.
    let mut wave = Vec::new();
    for _ in 0..60 {
        s.step();
        wave.push(bit(&s, "a"));
    }
    assert!(wave.iter().all(|v| v.is_known()), "{wave:?}");
    for i in 6..wave.len() {
        assert_eq!(wave[i], wave[i - 6]);
        assert_ne!(wave[i], wave[i - 3]);
    }
    s.run(200);
    assert!(bit(&s, "a").is_known());
    assert!(!s.settle(50), "an oscillator never settles");
    let d = s.take_diagnostics();
    assert_eq!(d.len(), 1, "exactly one Oscillating per episode: {d:?}");
    assert_eq!(d[0].kind, DiagKind::Oscillating);
    assert!(d[0].message.contains('a') || d[0].message.contains('b'));

    // Stop: the ring settles; restart: a new episode reports again.
    set(&mut s, "EN", false);
    assert!(s.settle(100));
    assert_eq!(bit(&s, "a"), One);
    assert!(kinds(&mut s).is_empty());
    set(&mut s, "EN", true);
    s.run(150);
    assert_eq!(kinds(&mut s), vec![DiagKind::Oscillating]);
}

#[test]
fn ring_oscillator_five_stages_has_period_ten() {
    let opts = TimingOptions::default().with_osc_limit(1_000);
    let mut s = sim_with(ring(4), vec![], opts);
    assert!(s.settle(100));
    set(&mut s, "EN", true);
    s.run(20);
    let mut wave = Vec::new();
    for _ in 0..100 {
        s.step();
        wave.push(bit(&s, "b2"));
    }
    for i in 10..wave.len() {
        assert_eq!(wave[i], wave[i - 10]);
        assert_ne!(wave[i], wave[i - 5]);
    }
    assert!(kinds(&mut s).is_empty(), "below osc_limit, no collapse");
    s.run(1_000);
    assert_eq!(kinds(&mut s), vec![DiagKind::Oscillating]);
}

#[test]
fn ring_oscillator_starts_quiescent_at_power_on_x() {
    // Enabled from power-on: NAND(1, X) = X, so the ring sits at X (SPEC §14.4).
    let mut b = DefBuilder::new("top");
    let en = b.lit(1, 1);
    let fb = b.wire(1);
    let a = b.nand(&[en, fb]);
    let c = b.not(a);
    b.not_to(c, fb);
    b.output("OUT", a);
    let mut s = sim(b.finish(), vec![]);
    assert!(s.settle(10));
    assert_eq!(bit(&s, "OUT"), X);
}

fn not_loop() -> ComponentDef {
    let mut b = DefBuilder::new("top");
    let y = b.wire(1);
    b.not_to(y, y);
    b.name(y, "y");
    b.output("Y", y);
    b.finish()
}

#[test]
fn not_self_loop_power_on_x_is_stable() {
    let mut s = sim(not_loop(), vec![]);
    assert!(s.settle(10));
    assert_eq!(bit(&s, "y"), X);
    assert!(kinds(&mut s).is_empty());
}

#[test]
fn not_self_loop_collapses() {
    let mut s = sim(not_loop(), vec![]);
    s.run(20);
    let y = s.netlist().net_by_name("y").unwrap();
    s.force_net(y, Zero);
    let t0 = s.tick();
    let mut wave = Vec::new();
    while s.tick() < t0 + 20 {
        s.step();
        wave.push(bit(&s, "y"));
    }
    // Toggles for 2 * p2_limit - 1 ticks, collapses at the 8th change, then stays X.
    assert_eq!(&wave[..7], &[One, Zero, One, Zero, One, Zero, One]);
    assert!(wave[7..].iter().all(|&v| v == X), "{wave:?}");
    assert!(s.settle(10));
    let d = s.take_diagnostics();
    assert_eq!(d.len(), 1);
    assert_eq!(d[0].kind, DiagKind::Metastable);
    assert_eq!(d[0].location.as_deref(), Some("Y"), "canonical name prefers the pin");
}

#[test]
fn nor_self_loop_released_collapses() {
    // y = NOR(EN, y). EN=1 holds y=0; releasing EN starts a period-2 oscillation.
    let mut b = DefBuilder::new("top");
    let en = b.input("EN", 1);
    let y = b.wire(1);
    b.nor_to(&[en, y], y);
    b.name(y, "y");
    let z = b.not(y);
    b.name(z, "z");
    let mut s = sim(b.finish(), vec![]);
    set(&mut s, "EN", true);
    s.reset();
    assert!(s.settle(10));
    assert_eq!(bit(&s, "y"), Zero);
    set(&mut s, "EN", false);
    assert!(s.settle(30));
    assert_eq!(bit(&s, "y"), X);
    assert_eq!(bit(&s, "z"), X, "X propagates downstream");
    // z also toggled every tick, but it became X through propagation: one diagnostic only.
    let d = s.take_diagnostics();
    assert_eq!(d.len(), 1, "{d:?}");
    assert_eq!(d[0].kind, DiagKind::Metastable);
    assert_eq!(d[0].location.as_deref(), Some("y"));
    // Re-asserting EN recovers.
    set(&mut s, "EN", true);
    assert!(s.settle(10));
    assert_eq!((bit(&s, "y"), bit(&s, "z")), (Zero, One));
}

#[test]
fn clock_activity_suppresses_collapse() {
    // A NOT self-loop next to an unrelated clock: sources change every 3 ticks, so the
    // period-2 rule never fires and the loop keeps toggling.
    let mut b = DefBuilder::new("top");
    let clk = b.clock(6, 3, 0);
    b.output("CLK", clk);
    let y = b.wire(1);
    b.not_to(y, y);
    b.name(y, "y");
    let opts = TimingOptions::default().with_osc_limit(10);
    let mut s = sim_with(b.finish(), vec![], opts);
    s.step();
    let y = s.netlist().net_by_name("y").unwrap();
    s.force_net(y, One);
    let mut prev = s.get(y);
    for _ in 0..500 {
        s.step();
        let v = s.get(y);
        assert!(v.is_known());
        assert_ne!(v, prev);
        prev = v;
    }
    assert!(kinds(&mut s).is_empty(), "no Metastable / Oscillating while sources change");
}

#[test]
fn clock_with_period_two_toggles_every_tick_without_collapse() {
    let mut b = DefBuilder::new("top");
    let clk = b.clock(2, 1, 0);
    let nclk = b.not(clk);
    b.name(nclk, "nclk");
    b.output("CLK", clk);
    let mut s = sim(b.finish(), vec![]);
    s.run(3);
    for _ in 0..100 {
        s.step();
        assert_eq!(bit(&s, "nclk"), eval_not(bit(&s, "CLK")).flip());
    }
    assert!(kinds(&mut s).is_empty());
}

trait Flip {
    fn flip(self) -> Self;
}
impl Flip for Logic {
    /// The value one tick earlier on a period-2 waveform.
    fn flip(self) -> Self {
        panda_netlist::eval::not1(self)
    }
}
fn eval_not(v: Logic) -> Logic {
    panda_netlist::eval::not1(v)
}

#[test]
fn long_glitch_train_is_not_collapsed() {
    // y = XOR of A delayed by 0..=9 ticks: after A toggles, y toggles on 10 consecutive ticks
    // (a glitch train in acyclic logic). That alternation is not self-sustained (it is driven by
    // taps that each change once), so the period-2 rule must not touch it.
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 1);
    let mut taps = vec![a];
    for _ in 0..9 {
        let last = *taps.last().unwrap();
        taps.push(b.buf(last));
    }
    let y = b.xor(&taps);
    b.name(y, "y");
    let mut s = sim(b.finish(), vec![]);
    assert!(s.settle(50));
    assert_eq!(bit(&s, "y"), Zero);
    set(&mut s, "A", true);
    let mut wave = Vec::new();
    while !s.is_quiescent() {
        s.step();
        wave.push(bit(&s, "y"));
    }
    assert_eq!(wave.iter().filter(|v| **v == One).count(), 5, "{wave:?}");
    assert!(wave.iter().all(|v| v.is_known()), "no X blip: {wave:?}");
    assert_eq!(bit(&s, "y"), Zero);
    assert!(kinds(&mut s).is_empty());
}

/// Two tri-state drivers on one bus with the given termination; `INV = NOT(bus)`.
fn bus(term: Termination) -> TimingSim {
    let mut b = DefBuilder::new("top");
    let d1 = b.input("D1", 1);
    let e1 = b.input("E1", 1);
    let d2 = b.input("D2", 1);
    let e2 = b.input("E2", 1);
    let bus = b.wire(1);
    b.tri_to(d1, e1, bus);
    b.tri_to(d2, e2, bus);
    b.name(bus, "bus");
    b.terminate(bus, term);
    let inv = b.not(bus);
    b.output("BUS", bus);
    b.output("INV", inv);
    sim(b.finish(), vec![])
}

#[test]
fn tri_state_bus_contention() {
    let mut s = bus(Termination::None);
    assert!(s.settle(10));
    assert_eq!(bit(&s, "bus"), Z, "no driver enabled: Z");
    assert_eq!(bit(&s, "INV"), X, "a gate reads Z as X");

    set(&mut s, "D1", true);
    set(&mut s, "E1", true);
    assert!(s.settle(10));
    assert_eq!((bit(&s, "BUS"), bit(&s, "INV")), (One, Zero));

    // Second driver agrees: no contention.
    set(&mut s, "D2", true);
    set(&mut s, "E2", true);
    assert!(s.settle(10));
    assert_eq!(bit(&s, "bus"), One);
    assert!(kinds(&mut s).is_empty());

    // Disagree: X + one Contention diagnostic.
    set(&mut s, "D2", false);
    assert!(s.settle(10));
    assert_eq!(bit(&s, "bus"), X);
    let d = s.take_diagnostics();
    assert_eq!(d.len(), 1, "{d:?}");
    assert_eq!(d[0].kind, DiagKind::Contention);
    assert_eq!(d[0].location.as_deref(), Some("BUS"));
    // Persisting contention is not re-reported.
    set(&mut s, "D1", false);
    set(&mut s, "D2", true);
    assert!(s.settle(10));
    assert_eq!(bit(&s, "bus"), X);
    assert!(kinds(&mut s).is_empty());

    // Resolve, then contend again: new episode.
    set(&mut s, "E1", false);
    assert!(s.settle(10));
    assert_eq!(bit(&s, "bus"), One);
    set(&mut s, "E1", true);
    assert!(s.settle(10));
    assert_eq!(kinds(&mut s), vec![DiagKind::Contention]);

    // Enable X on one driver: that driver outputs X -> contention X.
    s.set_input("E1", &Value::from_logic(X)).unwrap();
    set(&mut s, "E2", false);
    assert!(s.settle(10));
    assert_eq!(bit(&s, "bus"), X);
}

#[test]
fn tri_state_bus_resolver_is_zero_delay() {
    let mut s = bus(Termination::None);
    set(&mut s, "D1", true);
    s.settle(10);
    set(&mut s, "E1", true);
    s.step(); // tick t: E1 = 1
    assert_eq!(bit(&s, "bus"), Z);
    s.step(); // t+1: TRI output (delay 1) and bus (resolver, delay 0)
    assert_eq!(bit(&s, "bus"), One);
    assert_eq!(bit(&s, "INV"), X);
    s.step();
    assert_eq!(bit(&s, "INV"), Zero);
}

#[test]
fn tri_state_bus_pull_up_and_pull_down() {
    let mut s = bus(Termination::PullUp);
    assert!(s.settle(10));
    assert_eq!((bit(&s, "bus"), bit(&s, "INV")), (One, Zero));
    set(&mut s, "E2", true);
    assert!(s.settle(10));
    assert_eq!(bit(&s, "bus"), Zero);
    set(&mut s, "E2", false);
    assert!(s.settle(10));
    assert_eq!(bit(&s, "bus"), One);

    let mut s = bus(Termination::PullDown);
    assert!(s.settle(10));
    assert_eq!(bit(&s, "bus"), Zero);
    assert!(kinds(&mut s).is_empty());
}

#[test]
fn tri_state_bus_keeper() {
    let mut s = bus(Termination::Keeper);
    assert!(s.settle(10));
    assert_eq!(bit(&s, "bus"), X, "keeper powers on at X");
    set(&mut s, "D1", true);
    set(&mut s, "E1", true);
    assert!(s.settle(10));
    assert_eq!(bit(&s, "bus"), One);
    set(&mut s, "E1", false);
    assert!(s.settle(10));
    assert_eq!(bit(&s, "bus"), One, "keeper holds the last driven value");
    set(&mut s, "E2", true);
    assert!(s.settle(10));
    assert_eq!(bit(&s, "bus"), Zero);
    set(&mut s, "E2", false);
    set(&mut s, "D2", true);
    assert!(s.settle(10));
    assert_eq!(bit(&s, "bus"), Zero, "still holding 0");
    s.reset();
    assert!(s.settle(10));
    assert_eq!(bit(&s, "bus"), X, "reset clears keeper state");
    assert!(kinds(&mut s).is_empty());
}

#[test]
fn undriven_terminated_net() {
    let mut b = DefBuilder::new("top");
    let w = b.wire(1);
    b.terminate(w, Termination::PullUp);
    let n = b.not(w);
    b.output("N", n);
    let floating = b.wire(1);
    let m = b.not(floating);
    b.output("M", m);
    let mut s = sim(b.finish(), vec![]);
    assert!(s.settle(10));
    assert_eq!(bit(&s, "N"), Zero);
    assert_eq!(bit(&s, "M"), X);
}

#[test]
fn clock_waveform_phase_and_high() {
    let mut b = DefBuilder::new("top");
    let c1 = b.clock(6, 2, 1);
    b.output("C1", c1);
    let c2 = b.clock(4, 3, 0);
    b.output("C2", c2);
    let c3 = b.clock(5, 1, 13); // phase larger than the period: phase % period = 3
    b.output("C3", c3);
    let mut s = sim(b.finish(), vec![]);
    let mut w1 = String::new();
    let mut w2 = String::new();
    let mut w3 = String::new();
    for _ in 0..13 {
        w1.push(bit(&s, "C1").to_char());
        w2.push(bit(&s, "C2").to_char());
        w3.push(bit(&s, "C3").to_char());
        s.step();
    }
    assert_eq!(w1, "0110000110000");
    assert_eq!(w2, "1110111011101");
    assert_eq!(w3, "0001000010000");
}

#[test]
fn power_on_reset_waveform() {
    let mut b = DefBuilder::new("top");
    let r = b.power_on_reset(3);
    b.output("R", r);
    let mut s = sim(b.finish(), vec![]);
    let mut w = String::new();
    for _ in 0..6 {
        w.push(bit(&s, "R").to_char());
        s.step();
    }
    assert_eq!(w, "111000");
    s.reset();
    assert_eq!(bit(&s, "R"), One);
}

fn named_design() -> TimingSim {
    let mut sub = DefBuilder::new("sub");
    let a = sub.input("A", 4);
    let c = sub.not(a);
    sub.name(c, "c");
    let flag = sub.reduce(panda_netlist::GateOp::And, c);
    sub.name(flag, "flag");
    sub.output("F", flag);
    sub.output("C", c);
    let sub = sub.finish();
    let sref = panda_netlist::DefRef::of(&sub);

    let mut b = DefBuilder::new("top");
    let a = b.input("A", 4);
    let o = b.inst_named("u", &sref, &[a]);
    b.output("F", o[0]);
    b.output("C", o[1]);
    let x = b.xor(&[a, o[1]]);
    b.name(x, "x");
    sim(b.finish(), vec![sub])
}

#[test]
fn read_and_read_pin_naming() {
    let mut s = named_design();
    s.set_input("A", &Value::from_u64(4, 0b0101)).unwrap();
    assert!(s.settle(10));
    // Top pins.
    assert_eq!(s.read_pin("A").unwrap().to_u64(), Some(0b0101));
    assert_eq!(s.read_pin("C").unwrap().to_u64(), Some(0b1010));
    assert_eq!(s.read_pin("F").unwrap().to_u64(), Some(0));
    assert_eq!(s.read_pin("u/c"), None, "read_pin is top-level only");
    assert_eq!(s.read_pin("nope"), None);
    // Exact flat names and hierarchical aliases.
    assert_eq!(s.read("u/c[1]").unwrap(), Value::from_bool(true));
    assert_eq!(s.read("u/c[0]").unwrap(), Value::from_bool(false));
    assert_eq!(s.read("u/flag").unwrap(), Value::from_bool(false));
    assert_eq!(s.read("u/F").unwrap(), Value::from_bool(false));
    assert_eq!(s.read("A[2]").unwrap(), Value::from_bool(true));
    // Multi-bit base names.
    assert_eq!(s.read("u/c").unwrap().to_u64(), Some(0b1010));
    assert_eq!(s.read("u/A").unwrap().to_u64(), Some(0b0101));
    assert_eq!(s.read("x").unwrap().to_u64(), Some(0b1111));
    assert_eq!(s.read("A").unwrap().to_u64(), Some(0b0101));
    assert_eq!(s.read("C").unwrap().to_u64(), Some(0b1010));
    assert_eq!(s.read("u/c[4]"), None);
    assert_eq!(s.read("nope"), None);

    s.set_input("A", &Value::from_u64(4, 0)).unwrap();
    assert!(s.settle(10));
    assert_eq!(s.read("u/flag").unwrap(), Value::from_bool(true));
    assert_eq!(s.read_pin("F").unwrap(), Value::from_bool(true));
}

#[test]
fn set_input_multi_bit_and_errors() {
    let mut s = named_design();
    assert_eq!(s.read_pin("A").unwrap().to_u64(), Some(0), "inputs default to 0");
    let v = Value::parse(4, "0b1x0z").unwrap();
    s.set_input("A", &v).unwrap();
    assert_eq!(s.read_pin("A").unwrap().to_u64(), Some(0), "takes effect at the next tick");
    s.step();
    assert_eq!(s.read_pin("A").unwrap(), v);
    assert!(s.settle(10));
    // NOT: 1->0, x->x, 0->1, z->x
    assert_eq!(s.read("u/c").unwrap(), Value::parse(4, "0b0x1x").unwrap());
    assert_eq!(s.input_value("A"), Some(&v));

    assert!(s.set_input("A", &Value::from_u64(3, 1)).is_err(), "width mismatch");
    assert!(s.set_input("B", &Value::from_u64(4, 1)).is_err(), "unknown pin");
    assert!(s.set_input("C", &Value::from_u64(4, 1)).is_err(), "outputs are not inputs");
}

#[test]
fn reset_restores_power_on_and_keeps_inputs() {
    let mut s = named_design();
    s.set_input("A", &Value::from_u64(4, 9)).unwrap();
    assert!(s.settle(10));
    assert_eq!(s.read_pin("C").unwrap().to_u64(), Some(6));
    assert!(s.stats().gate_evals > 0);
    s.reset();
    assert_eq!(s.tick(), 0);
    assert_eq!(s.stats(), Default::default());
    assert_eq!(s.read_pin("A").unwrap().to_u64(), Some(9), "inputs survive reset");
    assert_eq!(s.read_pin("C").unwrap(), Value::xs(4), "gate outputs are X at power-on");
    assert_eq!(s.pending_events(), s.netlist().gates.len(), "every gate evaluated at tick 1");
    s.step();
    assert_eq!(s.read_pin("C").unwrap().to_u64(), Some(6));
    assert!(s.settle(10));
    assert_eq!(s.read_pin("F").unwrap().to_u64(), Some(0));
}

#[test]
fn settle_semantics() {
    // A chain of 7 buffers settles exactly 7 ticks after an input change.
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 1);
    let mut cur = a;
    for _ in 0..7 {
        cur = b.buf(cur);
    }
    b.output("Y", cur);
    let mut s = sim(b.finish(), vec![]);
    assert!(s.settle(100));
    assert_eq!(s.tick(), 7, "power-on X needs 7 ticks to flush");
    assert!(s.settle(100));
    assert_eq!(s.tick(), 7, "settle on a quiescent circuit does not advance time");
    set(&mut s, "A", true);
    assert!(!s.is_quiescent(), "a pending input change is not quiescent");
    assert!(!s.settle(5), "not settled within the budget");
    assert_eq!(s.tick(), 12);
    assert!(s.settle(100));
    assert_eq!(s.tick(), 15, "input at 8, output at 15");
    assert_eq!(bit(&s, "Y"), One);
    // Setting the same value again is not a change.
    set(&mut s, "A", true);
    assert!(s.is_quiescent());
    // run() fast-forwards a quiescent circuit.
    s.run(1_000_000);
    assert_eq!(s.tick(), 1_000_015);
    assert_eq!(s.stats().ticks, 1_000_015);
    assert!(s.last_changed().is_empty());

    // A clock is never quiescent.
    let mut b = DefBuilder::new("top");
    let c = b.clock(10, 5, 0);
    b.output("C", c);
    let mut s = sim(b.finish(), vec![]);
    assert!(!s.settle(25));
    assert_eq!(s.tick(), 25);

    // A PowerOnReset is not quiescent until released.
    let mut b = DefBuilder::new("top");
    let r = b.power_on_reset(20);
    let n = b.not(r);
    b.output("N", n);
    let mut s = sim(b.finish(), vec![]);
    assert!(s.settle(100));
    assert_eq!(s.tick(), 21, "released at 20, NOT output at 21");
    assert_eq!(bit(&s, "N"), One);
}

#[test]
fn last_changed_lists_this_ticks_changes() {
    let mut s = bus(Termination::None);
    s.settle(10);
    set(&mut s, "D1", true);
    set(&mut s, "E1", true);
    s.step();
    let nl = s.netlist().clone();
    let mut names: Vec<&str> = s.last_changed().iter().map(|&n| nl.net_names[n as usize].as_str()).collect();
    names.sort();
    assert_eq!(names, vec!["D1", "E1"]);
    s.step();
    let names: Vec<&str> = s.last_changed().iter().map(|&n| nl.net_names[n as usize].as_str()).collect();
    assert!(names.contains(&"BUS"), "{names:?}");
    // Every reported net really differs from its value one tick earlier.
    let before = s.values().to_vec();
    s.step();
    for &n in s.last_changed() {
        assert_ne!(before[n as usize], s.get(n));
    }
    for (i, (&b0, &b1)) in before.iter().zip(s.values()).enumerate() {
        if b0 != b1 {
            assert!(s.last_changed().contains(&(i as u32)));
        }
    }
}

#[test]
fn ms_dff_same_tick_data_and_clock_is_deterministic() {
    // D and the clock edge change on the same tick (a hold violation, SPEC §9.2). Whatever the
    // outcome, it must be reproducible and must not hang; any oscillation collapses to X.
    let dff = panda_netlist::DefRef::of(&ms_dff());
    let mut b = DefBuilder::new("top");
    let d = b.input("D", 1);
    let c = b.input("C", 1);
    let o = b.inst_named("ff", &dff, &[d, c]);
    b.output("Q", o[0]);
    let top = b.finish();
    let mut outcomes = Vec::new();
    for _ in 0..2 {
        let mut s = sim(top.clone(), dff_defs());
        assert!(s.settle(100));
        set(&mut s, "D", true);
        assert!(s.settle(100));
        set(&mut s, "D", false);
        set(&mut s, "C", true);
        assert!(s.settle(100), "must settle");
        outcomes.push((bit(&s, "Q"), kinds(&mut s)));
    }
    assert_eq!(outcomes[0], outcomes[1]);
}
