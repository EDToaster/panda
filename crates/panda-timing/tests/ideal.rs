//! Ideal blocks (SPEC §5.2): selection, latency, transport semantics, glass-box override, internal
//! net reads, and differential equivalence with gate-level ("glass-box") simulation.

mod common;

use std::sync::Arc;

use common::Rng;
use panda_circuits::{bus_port, decoder, mux, ripple_adder, rom, testbench};
use panda_netlist::flat::FlatNetlist;
use panda_netlist::{DefBuilder, DefRef, DesignBuilder, Logic, Termination, Value, flatten};
use panda_timing::{TimingOptions, TimingSim};

fn ideal_opts() -> TimingOptions {
    TimingOptions::default().with_ideal_blocks(true)
}

fn flat(db: DesignBuilder, top: &DefRef) -> Arc<FlatNetlist> {
    Arc::new(flatten(&db.build(top)).unwrap_or_else(|e| panic!("flatten failed: {e:#?}")))
}

/// Wrap a combinational DUT (instance `dut`) with top-level pins mirroring its pins.
fn wrapped(mut db: DesignBuilder, dut: &DefRef) -> Arc<FlatNetlist> {
    let top = testbench(&mut db, dut, 4);
    flat(db, &top)
}

fn random_value(rng: &mut Rng, width: u32, x_prob: f64) -> Value {
    let mut v = Value::zeros(width);
    for i in 0..width {
        let l = if rng.chance(x_prob) {
            if rng.chance(0.5) { Logic::X } else { Logic::Z }
        } else {
            Logic::from_bool(rng.chance(0.5))
        };
        v.set(i, l);
    }
    v
}

fn set_all(sims: &mut [&mut TimingSim], pin: &str, v: &Value) {
    for s in sims.iter_mut() {
        s.set_input(pin, v).unwrap();
    }
}

fn max_latency(s: &TimingSim) -> u64 {
    s.ideal_blocks().iter().map(|b| u64::from(b.max_latency())).max().unwrap_or(0)
}

/// Differential test: random input vectors (with optional X/Z bits), sometimes changed again
/// before settling (glitchy input bursts), always finally held for `max latency + 1` ticks; then
/// every net — outputs *and* internal nets of ideal blocks — must equal the glass-box value.
/// Runs twice: with LUT evaluation (default) and with gate evaluation only.
fn differential(nl: Arc<FlatNetlist>, vectors: usize, seed: u64, x_prob: f64) {
    for lut_bits in [12, 0] {
        differential_lut(nl.clone(), vectors, seed, x_prob, lut_bits);
    }
}

fn differential_lut(nl: Arc<FlatNetlist>, vectors: usize, seed: u64, x_prob: f64, lut_bits: u32) {
    let mut id = TimingSim::new(nl.clone(), ideal_opts().with_ideal_lut_bits(lut_bits));
    let mut gb = TimingSim::new(nl.clone(), TimingOptions::default());
    assert!(!id.ideal_blocks().is_empty(), "no ideal blocks selected");
    let hold = max_latency(&id) + 1;
    let mut rng = Rng::new(seed);
    let pins: Vec<(String, u32)> = nl.top_inputs.iter().map(|p| (p.name.clone(), p.nets.len() as u32)).collect();
    for k in 0..vectors {
        // Optional burst of short-lived vectors (shorter than the latency).
        if rng.chance(0.3) {
            for _ in 0..1 + rng.below(4) {
                for (name, w) in &pins {
                    let v = random_value(&mut rng, *w, x_prob);
                    set_all(&mut [&mut id, &mut gb], name, &v);
                }
                let dt = 1 + rng.below(hold as usize) as u64;
                id.run(dt);
                gb.run(dt);
            }
        }
        for (name, w) in &pins {
            let v = random_value(&mut rng, *w, x_prob);
            set_all(&mut [&mut id, &mut gb], name, &v);
        }
        id.run(hold);
        gb.run(hold);
        if id.ideal_blocks().len() == 1 {
            // A single block settles exactly within `max latency + 1` ticks, as do its gates.
            assert!(gb.is_quiescent() && id.is_quiescent(), "not settled after {hold} ticks (vector {k})");
        }
        // Chains of blocks (e.g. through an outside resolver) take longer.
        assert!(gb.settle(10 * hold + 100), "glass-box not settled (vector {k})");
        assert!(id.settle(10 * hold + 100), "ideal not settled (vector {k})");
        // `get` is exact without a refresh (internal nets computed on demand after LUT hits).
        for n in 0..nl.num_nets {
            assert_eq!(
                id.get(n),
                gb.get(n),
                "net {} differs after vector {k} (lut {lut_bits})",
                nl.net_names[n as usize]
            );
        }
        // `values` after a refresh (which must not disturb the simulation).
        id.refresh_ideal_internal();
        assert!(id.is_quiescent());
        for n in 0..nl.num_nets as usize {
            assert_eq!(id.values()[n], gb.values()[n], "net {} differs after vector {k} (values)", nl.net_names[n]);
        }
    }
}

#[test]
fn differential_ripple_adder() {
    let mut db = DesignBuilder::new();
    let dut = ripple_adder(&mut db, 8);
    let nl = wrapped(db, &dut);
    differential(nl.clone(), 300, 1, 0.0);
    differential(nl, 200, 2, 0.1);
}

#[test]
fn differential_mux() {
    let mut db = DesignBuilder::new();
    let dut = mux(&mut db, 3, 4);
    let nl = wrapped(db, &dut);
    differential(nl.clone(), 300, 3, 0.0);
    differential(nl, 200, 4, 0.1);
}

#[test]
fn differential_decoder() {
    let mut db = DesignBuilder::new();
    let dut = decoder(&mut db, 4);
    let nl = wrapped(db, &dut);
    differential(nl.clone(), 200, 5, 0.0);
    differential(nl, 200, 6, 0.15);
}

#[test]
fn differential_rom() {
    let mut rng = Rng::new(7);
    let words: Vec<u64> = (0..32).map(|_| rng.next_u64() & 0xFF).collect();
    let mut db = DesignBuilder::new();
    let dut = rom(&mut db, "t", 5, 8, &words);
    let nl = wrapped(db, &dut);
    // Constants inside the ROM are block inputs, not sources that disqualify it.
    let s = TimingSim::new(nl.clone(), ideal_opts());
    assert_eq!(s.ideal_blocks().len(), 1);
    assert_eq!(s.ideal_blocks()[0].path, "dut");
    differential(nl.clone(), 200, 8, 0.0);
    differential(nl.clone(), 100, 9, 0.1);
    // And the ROM reads its contents.
    let mut s = TimingSim::new(nl, ideal_opts());
    for (a, &w) in words.iter().enumerate() {
        s.set_input("A", &Value::from_u64(5, a as u64)).unwrap();
        s.run(max_latency(&s) + 1);
        assert_eq!(s.read_pin("D").unwrap().to_u64(), Some(w), "word {a}");
    }
}

/// A package with two `TRI`s onto one internal wire (a resolver inside the block).
fn trimux(db: &mut DesignBuilder) -> DefRef {
    let mut b = DefBuilder::new("trimux");
    let d0 = b.input("D0", 4);
    let d1 = b.input("D1", 4);
    let s = b.input("S", 1);
    let ns = b.not(s);
    let y = b.wire(4);
    b.tri_to(d0, ns, y);
    b.tri_to(d1, s, y);
    b.terminate(y, Termination::PullDown);
    b.name(y, "y");
    let o = b.buf(y);
    b.output("Y", o);
    db.add_once(b.finish())
}

#[test]
fn differential_internal_resolver() {
    let mut db = DesignBuilder::new();
    let dut = trimux(&mut db);
    let nl = wrapped(db, &dut);
    let s = TimingSim::new(nl.clone(), ideal_opts());
    let b = &s.ideal_blocks()[0];
    assert_eq!(b.resolvers.len(), 4, "the 4 bus-bit resolvers are inside the block");
    differential(nl.clone(), 300, 10, 0.0);
    differential(nl, 300, 11, 0.2);
}

#[test]
fn differential_shared_bus_resolver_outside_blocks() {
    // Two bus ports on one top-level bus: the resolver has drivers in two blocks, so it stays
    // outside; each port's TRI output is a block output.
    let mut db = DesignBuilder::new();
    let bp = bus_port(&mut db, 2);
    let mut b = DefBuilder::new("top");
    let d0 = b.input("D0", 2);
    let d1 = b.input("D1", 2);
    let oe0 = b.input("OE0", 1);
    let oe1 = b.input("OE1", 1);
    let bus = b.wire(2);
    b.terminate(bus, Termination::PullUp);
    b.name(bus, "bus");
    let q0 = b.inst_map("p0", &bp, &[("D", d0), ("OE", oe0), ("BUS", bus)]);
    let q1 = b.inst_map("p1", &bp, &[("D", d1), ("OE", oe1), ("BUS", bus)]);
    b.output("Q0", q0["Q"]);
    b.output("Q1", q1["Q"]);
    let top = db.add(b.finish());
    let nl = flat(db, &top);
    let s = TimingSim::new(nl.clone(), ideal_opts());
    let paths: Vec<&str> = s.ideal_blocks().iter().map(|b| b.path.as_str()).collect();
    assert_eq!(paths, ["p0", "p1"]);
    assert!(s.ideal_blocks().iter().all(|b| b.resolvers.is_empty()));
    differential(nl, 300, 12, 0.05);
}

#[test]
fn selection_and_latencies_ripple_adder() {
    let mut db = DesignBuilder::new();
    let dut = ripple_adder(&mut db, 8);
    let nl = wrapped(db, &dut);
    let s = TimingSim::new(nl.clone(), ideal_opts());
    let blocks = s.ideal_blocks();
    assert_eq!(blocks.len(), 1, "the outermost acyclic instance is the block");
    let b = &blocks[0];
    assert_eq!(b.path, "dut");
    assert_eq!(b.gates.len(), 8 * 5);
    let lat = |name: &str| {
        let net = nl.net_by_name(name).unwrap();
        b.latency[b.outputs.iter().position(|&o| o == net).unwrap()]
    };
    // S0 = (A^B)^CIN: depth 2. Carry into bit i (i ≥ 1): 3 + 2(i-1). S_i = carry_in(i) + 1.
    assert_eq!(lat("S[0]"), 2);
    for i in 1..8u32 {
        assert_eq!(lat(&format!("S[{i}]")), 2 * i + 2, "S[{i}]");
    }
    assert_eq!(lat("COUT"), 17);
    assert_eq!(b.max_latency(), 17);
    // Default options: no ideal blocks at all.
    assert!(TimingSim::new(nl, TimingOptions::default()).ideal_blocks().is_empty());
}

/// Output `j` changes exactly at `t + L_j`, where `t` is the tick at which the input changed — even
/// when the actual sensitized path is shorter (CIN → COUT is 16 gates, L = 17).
#[test]
fn latency_is_exact() {
    let mut db = DesignBuilder::new();
    let dut = ripple_adder(&mut db, 8);
    let nl = wrapped(db, &dut);
    let mut s = TimingSim::new(nl.clone(), ideal_opts());
    let b = s.ideal_blocks()[0].clone();
    s.set_input("A", &Value::from_u64(8, 0xFF)).unwrap();
    s.set_input("B", &Value::from_u64(8, 0)).unwrap();
    s.run(30);
    assert_eq!(s.read_pin("S").unwrap().to_u64(), Some(0xFF));
    assert_eq!(s.read_pin("COUT").unwrap().to_u64(), Some(0));
    s.set_input("CIN", &Value::from_bool(true)).unwrap();
    let t_in = s.tick() + 1; // the input source changes at the next tick
    let mut first_change = vec![None; b.outputs.len()];
    let before: Vec<Logic> = b.outputs.iter().map(|&n| s.get(n)).collect();
    for _ in 0..30 {
        s.step();
        for (j, &n) in b.outputs.iter().enumerate() {
            if first_change[j].is_none() && s.get(n) != before[j] {
                first_change[j] = Some(s.tick());
            }
        }
    }
    for (j, &n) in b.outputs.iter().enumerate() {
        assert_eq!(first_change[j], Some(t_in + u64::from(b.latency[j])), "output {}", nl.net_names[n as usize]);
    }
    assert_eq!(s.read_pin("S").unwrap().to_u64(), Some(0));
    assert_eq!(s.read_pin("COUT").unwrap().to_u64(), Some(1));
}

/// Transport delay: inputs changing every tick (pulses much shorter than L) reach the outputs as
/// `out_j(t + L_j) = f_j(in(t))`, every pending change preserved.
#[test]
fn transport_delay_keeps_every_pending_change() {
    let mut db = DesignBuilder::new();
    let dut = ripple_adder(&mut db, 8);
    let nl = wrapped(db, &dut);
    let mut s = TimingSim::new(nl.clone(), ideal_opts());
    // Oracle for f: a glass-box sim settled per vector.
    let mut oracle = TimingSim::new(nl.clone(), TimingOptions::default());
    let b = s.ideal_blocks()[0].clone();
    let mut rng = Rng::new(42);
    // in_at[t] = (A, B, CIN) at tick t.
    let mut history: Vec<(u64, Value, Value, Value)> = Vec::new();
    let mut outs: Vec<(u64, Vec<Logic>)> = Vec::new();
    for _ in 0..80 {
        let a = Value::from_u64(8, rng.next_u64() & 0xFF);
        let bb = Value::from_u64(8, rng.next_u64() & 0xFF);
        let c = Value::from_bool(rng.chance(0.5));
        s.set_input("A", &a).unwrap();
        s.set_input("B", &bb).unwrap();
        s.set_input("CIN", &c).unwrap();
        s.step();
        history.push((s.tick(), a, bb, c));
        outs.push((s.tick(), b.outputs.iter().map(|&n| s.get(n)).collect()));
    }
    for _ in 0..20 {
        s.step();
        outs.push((s.tick(), b.outputs.iter().map(|&n| s.get(n)).collect()));
    }
    for (t, a, bb, c) in &history {
        oracle.set_input("A", a).unwrap();
        oracle.set_input("B", bb).unwrap();
        oracle.set_input("CIN", c).unwrap();
        assert!(oracle.settle(100));
        for (j, &n) in b.outputs.iter().enumerate() {
            let due = t + u64::from(b.latency[j]);
            let got = outs.iter().find(|(tt, _)| *tt == due).map(|(_, v)| v[j]);
            if let Some(got) = got {
                assert_eq!(got, oracle.get(n), "output {} at tick {due}", nl.net_names[n as usize]);
            }
        }
    }
}

/// A block where every input→output path has the same length behaves exactly like its gates.
#[test]
fn balanced_block_is_tick_exact() {
    let mut db = DesignBuilder::new();
    let mut b = DefBuilder::new("inv3");
    let a = b.input("A", 1);
    let x = b.not(a);
    let y = b.not(x);
    let z = b.not(y);
    b.output("Y", z);
    let inv3 = db.add(b.finish());
    let nl = wrapped(db, &inv3);
    let mut id = TimingSim::new(nl.clone(), ideal_opts());
    let mut gb = TimingSim::new(nl.clone(), TimingOptions::default());
    assert_eq!(id.ideal_blocks()[0].latency, vec![3]);
    let mut rng = Rng::new(5);
    for _ in 0..200 {
        if rng.chance(0.5) {
            let v = Value::from_bool(rng.chance(0.5));
            set_all(&mut [&mut id, &mut gb], "A", &v);
        }
        id.step();
        gb.step();
        assert_eq!(id.read_pin("Y"), gb.read_pin("Y"), "tick {}", id.tick());
    }
}

/// A 1-gate package.
fn nand_pkg(db: &mut DesignBuilder) -> DefRef {
    let mut b = DefBuilder::new("nand_pkg");
    let a = b.input("A", 1);
    let bb = b.input("B", 1);
    let y = b.nand(&[a, bb]);
    b.output("Y", y);
    db.add_once(b.finish())
}

/// SR latch from two `nand_pkg` instances: the feedback loop runs through the package pins, each
/// instance is an acyclic L = 1 block, and the behaviour (including period-2 collapse on the
/// simultaneous release) is identical to plain gates, tick by tick.
#[test]
fn sr_latch_from_nand_packages_matches_gates() {
    let mut db = DesignBuilder::new();
    let np = nand_pkg(&mut db);
    let mut b = DefBuilder::new("top");
    let s_n = b.input("S_n", 1);
    let r_n = b.input("R_n", 1);
    let qn = b.wire(1);
    let q = b.inst_named("u1", &np, &[s_n, qn])[0];
    let qn2 = b.inst_named("u2", &np, &[r_n, q])[0];
    b.connect(qn2, qn);
    b.output("Q", q);
    b.output("Q_n", qn);
    let top = db.add(b.finish());
    let nl = flat(db, &top);
    let mut id = TimingSim::new(nl.clone(), ideal_opts());
    let mut gb = TimingSim::new(nl.clone(), TimingOptions::default());
    let paths: Vec<&str> = id.ideal_blocks().iter().map(|b| b.path.as_str()).collect();
    assert_eq!(paths, ["u1", "u2"]);
    assert!(id.ideal_blocks().iter().all(|b| b.latency == [1]));
    // set, hold, reset, hold, both low, simultaneous release (metastable), set, ...
    let seq = [(0, 1), (1, 1), (1, 0), (1, 1), (0, 0), (1, 1), (0, 1), (1, 1), (0, 0), (1, 1)];
    let mut diags_id = Vec::new();
    let mut diags_gb = Vec::new();
    for &(s, r) in &seq {
        set_all(&mut [&mut id, &mut gb], "S_n", &Value::from_bool(s == 1));
        set_all(&mut [&mut id, &mut gb], "R_n", &Value::from_bool(r == 1));
        for _ in 0..30 {
            id.step();
            gb.step();
            assert_eq!(id.values(), gb.values(), "tick {}", id.tick());
            assert_eq!(id.last_changed(), gb.last_changed(), "tick {}", id.tick());
        }
        diags_id.extend(id.take_diagnostics());
        diags_gb.extend(gb.take_diagnostics());
    }
    assert_eq!(diags_id, diags_gb);
    assert!(diags_id.iter().any(|d| d.kind == panda_netlist::DiagKind::Metastable), "{diags_id:?}");
}

#[test]
fn glass_box_override() {
    let mut db = DesignBuilder::new();
    let dut = ripple_adder(&mut db, 4);
    let nl = wrapped(db, &dut);
    // Glass-boxing the whole adder: no ideal blocks; identical to the default engine tick by tick.
    let mut gbx = TimingSim::new(nl.clone(), ideal_opts().with_glass_box("dut"));
    let mut plain = TimingSim::new(nl.clone(), TimingOptions::default());
    assert!(gbx.ideal_blocks().is_empty());
    let mut rng = Rng::new(9);
    for _ in 0..300 {
        if rng.chance(0.3) {
            let a = Value::from_u64(4, rng.next_u64() & 15);
            set_all(&mut [&mut gbx, &mut plain], "A", &a);
        }
        gbx.step();
        plain.step();
        assert_eq!(gbx.values(), plain.values());
    }
    // Glass-boxing one full adder: the adder itself is no longer a block (it would swallow fa2),
    // but the other full adders are.
    let s = TimingSim::new(nl.clone(), ideal_opts().with_glass_box("dut/fa2"));
    let paths: Vec<&str> = s.ideal_blocks().iter().map(|b| b.path.as_str()).collect();
    assert_eq!(paths, ["dut/fa0", "dut/fa1", "dut/fa3"]);
    differential_with(nl, ideal_opts().with_glass_box("dut/fa2"), 200, 13);
}

/// [`differential`] with explicit ideal-side options.
fn differential_with(nl: Arc<FlatNetlist>, opts: TimingOptions, vectors: usize, seed: u64) {
    let mut id = TimingSim::new(nl.clone(), opts);
    let mut gb = TimingSim::new(nl.clone(), TimingOptions::default());
    let mut rng = Rng::new(seed);
    let pins: Vec<(String, u32)> = nl.top_inputs.iter().map(|p| (p.name.clone(), p.nets.len() as u32)).collect();
    for _ in 0..vectors {
        for (name, w) in &pins {
            let v = random_value(&mut rng, *w, 0.0);
            set_all(&mut [&mut id, &mut gb], name, &v);
        }
        assert!(id.settle(1000));
        assert!(gb.settle(1000));
        id.refresh_ideal_internal();
        assert_eq!(id.values(), gb.values());
    }
}

/// Internal nets of an ideal block read as evaluated from the block's *current* inputs — already
/// in the tick the inputs change, while the outputs still show the old (delayed) values.
#[test]
fn internal_net_reads() {
    let mut db = DesignBuilder::new();
    let dut = ripple_adder(&mut db, 8);
    let nl = wrapped(db, &dut);
    let mut s = TimingSim::new(nl.clone(), ideal_opts());
    let carries = nl.net_by_name("dut/c[4]").unwrap();
    assert!(s.is_ideal_internal(carries));
    assert!(!s.is_ideal_internal(nl.net_by_name("COUT").unwrap()));
    // Power-on: internal nets are already computed from the tick-0 inputs (all 0); c[8] = COUT is
    // a block output and still X.
    let c = s.read("dut/c").unwrap();
    assert_eq!(c.slice(0, 8).to_u64(), Some(0));
    assert_eq!(c.get(8), Logic::X);
    assert_eq!(s.read_pin("S").unwrap().get(0), Logic::X, "outputs are still X at tick 0");
    s.run(20);
    s.set_input("A", &Value::from_u64(8, 0x0F)).unwrap();
    s.set_input("B", &Value::from_u64(8, 0x01)).unwrap();
    s.step(); // inputs change at this tick
    // c[i] = carry into bit i; 0x0F + 0x01: carries into bits 1..=4.
    assert_eq!(s.read("dut/c").unwrap().to_u64(), Some(0b1_1110));
    assert_eq!(s.read("dut/fa3/p").unwrap().to_u64(), Some(1));
    assert_eq!(s.get(nl.net_by_name("dut/fa0/g").unwrap()), Logic::One);
    assert_eq!(s.read_pin("S").unwrap().to_u64(), Some(0), "outputs not updated yet");
    // values() agrees with get() for internal nets.
    assert_eq!(s.values()[carries as usize], Logic::One);
    s.run(20);
    assert_eq!(s.read_pin("S").unwrap().to_u64(), Some(0x10));
    // Forcing an internal net is ignored.
    s.force_net(carries, Logic::Zero);
    assert_eq!(s.get(carries), Logic::One);
    // Glass-box values agree once settled.
    let mut gb = TimingSim::new(nl.clone(), TimingOptions::default());
    gb.set_input("A", &Value::from_u64(8, 0x0F)).unwrap();
    gb.set_input("B", &Value::from_u64(8, 0x01)).unwrap();
    assert!(gb.settle(100));
    assert_eq!(gb.read("dut/c"), s.read("dut/c"));
}

#[test]
fn reset_restarts_blocks() {
    let mut db = DesignBuilder::new();
    let dut = ripple_adder(&mut db, 8);
    let nl = wrapped(db, &dut);
    let mut s = TimingSim::new(nl, ideal_opts());
    s.set_input("A", &Value::from_u64(8, 3)).unwrap();
    s.set_input("B", &Value::from_u64(8, 4)).unwrap();
    s.step();
    assert!(!s.is_quiescent(), "output changes pending");
    s.reset();
    assert_eq!(s.read_pin("S").unwrap().get(0), Logic::X);
    assert!(!s.is_quiescent());
    s.run(15);
    assert_eq!(s.read_pin("S").unwrap().get(7), Logic::X, "S[7] has latency 16 from tick 0");
    assert_eq!(s.read_pin("S").unwrap().get(0), Logic::One, "S[0] has latency 2");
    s.step();
    assert_eq!(s.read_pin("S").unwrap().to_u64(), Some(7));
    assert!(s.settle(100));
    assert!(s.stats().block_evals > 0);
    assert_eq!(s.stats().gate_evals, 0, "every gate is inside the block");
}

/// The cross-package feedback loop with a longer latency: a ring of an L = 3 inverter package and
/// an L = 2 buffer package oscillates with period 2·(3+2) like the gates would.
#[test]
fn feedback_through_blocks_oscillates_like_gates() {
    let mut db = DesignBuilder::new();
    let mut b = DefBuilder::new("inv3");
    let a = b.input("A", 1);
    let x = b.not(a);
    let y = b.not(x);
    let z = b.not(y);
    b.output("Y", z);
    let inv3 = db.add(b.finish());
    let mut b = DefBuilder::new("nand_buf");
    let a = b.input("A", 1);
    let en = b.input("EN", 1);
    let x = b.nand(&[a, en]);
    let y = b.not(x);
    b.output("Y", y);
    let nb = db.add(b.finish());
    let mut b = DefBuilder::new("top");
    let en = b.input("EN", 1);
    let fb = b.wire(1);
    let o1 = b.inst_named("i", &inv3, &[fb])[0];
    let o2 = b.inst_named("g", &nb, &[o1, en])[0];
    b.connect(o2, fb);
    b.output("OUT", fb);
    let top = db.add(b.finish());
    let nl = flat(db, &top);
    let mut id = TimingSim::new(nl.clone(), ideal_opts().with_p2_limit(0));
    let mut gb = TimingSim::new(nl.clone(), TimingOptions::default().with_p2_limit(0));
    assert_eq!(id.ideal_blocks().len(), 2);
    set_all(&mut [&mut id, &mut gb], "EN", &Value::from_bool(false));
    id.run(10);
    gb.run(10);
    set_all(&mut [&mut id, &mut gb], "EN", &Value::from_bool(true));
    for _ in 0..100 {
        id.step();
        gb.step();
        assert_eq!(id.read_pin("OUT"), gb.read_pin("OUT"), "tick {}", id.tick());
    }
    assert!(!id.is_quiescent());
}

/// Forcing a constant input of a LUT-evaluated block (a ROM word) takes effect (the LUT is
/// bypassed until the next reset), and after a reset the ROM reads its real contents again.
#[test]
fn forcing_a_rom_constant_bypasses_the_lut() {
    let words = [0x11u64, 0x22, 0x33, 0x44];
    let mut db = DesignBuilder::new();
    let dut = rom(&mut db, "f", 2, 8, &words);
    let nl = wrapped(db, &dut);
    let mut s = TimingSim::new(nl.clone(), ideal_opts());
    let hold = max_latency(&s) + 1;
    let read = |s: &mut TimingSim, a: u64| {
        s.set_input("A", &Value::from_u64(2, a)).unwrap();
        s.run(hold);
        s.read_pin("D").unwrap().to_u64()
    };
    for (a, &w) in words.iter().enumerate() {
        assert_eq!(read(&mut s, a as u64), Some(w));
    }
    assert!(s.stats().lut_hits == 0, "rows are filled on first use");
    for (a, &w) in words.iter().enumerate() {
        assert_eq!(read(&mut s, a as u64), Some(w));
    }
    assert!(s.stats().lut_hits >= 4, "second pass is answered by the LUT");
    // Find the constant net carrying bit 0 of word 0x11 (= 1) and force it to 0 while A = 0.
    let src = nl
        .sources
        .iter()
        .find(|src| {
            src.kind == panda_netlist::flat::SourceKind::Constant(Logic::One)
                && s.ideal_blocks()[0].inputs.contains(&src.output)
        })
        .map(|src| src.output)
        .unwrap();
    assert_eq!(read(&mut s, 0), Some(0x11));
    s.force_net(src, Logic::Zero);
    // The forced source reverts at the next tick: the output shows a one-tick pulse L ticks later.
    let mut seen = Vec::new();
    for _ in 0..hold + 2 {
        seen.push(s.read_pin("D").unwrap().to_u64());
        s.step();
    }
    assert!(seen.contains(&Some(0x10)), "the forced constant reached the output: {seen:?}");
    assert_eq!(seen.iter().filter(|v| **v == Some(0x10)).count(), 1, "{seen:?}");
    assert_eq!(s.read_pin("D").unwrap().to_u64(), Some(0x11));
    s.reset();
    for (a, &w) in words.iter().enumerate() {
        assert_eq!(read(&mut s, a as u64), Some(w));
    }
}

#[test]
fn contention_inside_a_block_is_reported() {
    let mut db = DesignBuilder::new();
    let dut = trimux(&mut db);
    let nl = wrapped(db, &dut);
    for lut in [12, 0] {
        let mut s = TimingSim::new(nl.clone(), ideal_opts().with_ideal_lut_bits(lut));
        s.set_input("D0", &Value::from_u64(4, 0b0000)).unwrap();
        s.set_input("D1", &Value::from_u64(4, 0b1111)).unwrap();
        s.set_input("S", &Value::from_logic(Logic::X)).unwrap();
        assert!(s.settle(100));
        // S = X enables both TRIs as X: X on the bus.
        assert_eq!(s.read_pin("Y").unwrap().get(0), Logic::X);
        let d = s.take_diagnostics();
        assert!(
            d.iter().any(|d| d.kind == panda_netlist::DiagKind::Contention && d.message.contains("ideal block")),
            "lut {lut}: {d:?}"
        );
    }
}
