//! `inline_all` produces an atoms-only design that simulates identically to the original.

mod common;

use std::sync::Arc;

use common::*;
use panda_circuits::transform::{PATH_SEP, try_inline_all};
use panda_circuits::*;
use panda_netlist::{DefBuilder, DefRef, Design, DesignBuilder, InstKind, Value, flatten};
use panda_timing::{TimingOptions, TimingSim};

/// Counter + adder: `Q` counts, `SUM = Q + B`.
fn counter_adder(db: &mut DesignBuilder) -> DefRef {
    let cnt = sync_counter(db, 4);
    let add = ripple_adder(db, 4);
    let mut b = DefBuilder::new("counter_adder");
    let clk = b.input("CLK", 1);
    let rst = b.input("RST", 1);
    let en = b.input("EN", 1);
    let bb = b.input("B", 4);
    let q = b.inst_named("cnt", &cnt, &[clk, rst, en])[0];
    let zero = b.lit(1, 0);
    let o = b.inst_named("add", &add, &[q, bb, zero]);
    b.output("Q", q);
    b.output("SUM", o[0]);
    b.output("COUT", o[1]);
    db.add_once(b.finish())
}

fn assert_atoms_only(d: &Design) {
    assert_eq!(d.defs.len(), 1);
    let top = d.top_def().unwrap();
    assert!(top.instances.iter().all(|i| matches!(i.kind, InstKind::Atom(_))));
}

/// Run both designs in lockstep, applying `stimulus(tick)` (pin, value) changes to both and
/// comparing every top-level pin and every net that exists under both naming schemes.
fn lockstep(
    orig: &Design,
    inl: &Design,
    ticks: u64,
    mut stimulus: impl FnMut(u64) -> Vec<(&'static str, u64)>,
) -> usize {
    let fo = flatten(orig).unwrap();
    let fi = flatten(inl).unwrap();
    assert_eq!(fo.gates.len(), fi.gates.len(), "gate count");
    assert_eq!(fo.sources.len(), fi.sources.len(), "source count");
    assert_eq!(fo.resolvers.len(), fi.resolvers.len(), "resolver count");
    assert_eq!(fo.num_nets, fi.num_nets, "net count");
    // Net pairs: inlined `a.b.c` ↔ original `a/b/c`.
    let pairs: Vec<(u32, u32)> = fi
        .net_names
        .iter()
        .enumerate()
        .filter_map(|(i, name)| {
            let o = fo.net_by_name(&name.replace(PATH_SEP, "/"))?;
            Some((o, i as u32))
        })
        .collect();
    assert!(
        pairs.len() * 2 > fi.net_names.len(),
        "most nets keep recognizable names ({} of {})",
        pairs.len(),
        fi.net_names.len()
    );
    let pins: Vec<String> = fo.top_inputs.iter().chain(&fo.top_outputs).map(|p| p.name.clone()).collect();
    let mut so = TimingSim::new(Arc::new(fo), TimingOptions::default());
    let mut si = TimingSim::new(Arc::new(fi), TimingOptions::default());
    for t in 0..ticks {
        for (pin, v) in stimulus(t) {
            let w = so.input_value(pin).unwrap().width();
            so.set_input(pin, &Value::from_u64(w, v)).unwrap();
            si.set_input(pin, &Value::from_u64(w, v)).unwrap();
        }
        so.step();
        si.step();
        for p in &pins {
            assert_eq!(so.read_pin(p), si.read_pin(p), "pin {p} at tick {}", so.tick());
        }
        for &(o, i) in &pairs {
            assert_eq!(so.get(o), si.get(i), "net {} at tick {}", si.netlist().net_names[i as usize], so.tick());
        }
    }
    assert_eq!(so.take_diagnostics().len(), si.take_diagnostics().len());
    pairs.len()
}

#[test]
fn inline_counter_adder_simulates_identically() {
    let mut db = DesignBuilder::new();
    let dut = counter_adder(&mut db);
    let opts = TestbenchOptions::new(48);
    let tb = testbench_with(&mut db, &dut, &opts);
    let orig = db.build(&tb);
    let inl = inline_all(&orig);
    assert_atoms_only(&inl);
    assert_eq!(inl.top, orig.top);
    assert_eq!(inl.top_def().unwrap().pins, orig.top_def().unwrap().pins);
    // Valid with no warnings.
    let mut s = sim_design(&inl);
    // Hierarchical-ish names survive.
    assert!(s.read("dut.cnt.next").is_some());
    assert!(s.read("dut.cnt.reg.ff0.m").is_some());
    assert!(s.read("dut.cnt.reg.ff0.master.s").is_some());
    assert!(s.read("dut.add.c").is_some());
    // And it counts.
    set(&mut s, "EN", 1);
    set(&mut s, "B", 3);
    let k0 = opts.first_edge_after_reset();
    for i in 0..20u64 {
        cycle(&mut s, &opts, k0 + i, &[]);
        let q = (i + 1) % 16;
        assert_eq!(get(&s, "Q"), q);
        assert_eq!(get(&s, "SUM"), (q + 3) % 16);
        assert_eq!(get(&s, "COUT"), (q + 3) / 16);
    }

    let mut rng = Rng::new(9);
    let n = lockstep(&orig, &inl, 4000, |t| {
        let mut v = Vec::new();
        if t % 37 == 5 {
            v.push(("B", rng.below(16)));
        }
        if t % 101 == 50 {
            v.push(("EN", rng.below(2)));
        }
        v
    });
    assert!(n > 50, "only {n} nets compared");
}

#[test]
fn inline_gated_regfile_and_oscillator_simulate_identically() {
    let mut db = DesignBuilder::new();
    let rf = register_file(&mut db, 4, 4, RegFileStyle::GatedClock);
    let opts = TestbenchOptions::new(64);
    let tb = testbench_with(&mut db, &rf, &opts);
    let orig = db.build(&tb);
    let inl = inline_all(&orig);
    assert_atoms_only(&inl);
    let mut rng = Rng::new(4);
    let p = u64::from(opts.period);
    lockstep(&orig, &inl, 3000, |t| {
        // Change inputs only during the low phase (a quarter period before each edge).
        if t >= p && t % p == p / 4 {
            vec![("WA", rng.below(4)), ("WE", rng.below(2)), ("WD", rng.below(16)), ("RA", rng.below(4))]
        } else {
            vec![]
        }
    });

    let mut db = DesignBuilder::new();
    let r = ring_oscillator(&mut db, 5, true);
    let orig = db.build(&r);
    let inl = inline_all(&orig);
    lockstep(&orig, &inl, 300, |t| {
        if t == 10 {
            vec![("EN", 1)]
        } else if t == 200 {
            vec![("EN", 0)]
        } else {
            vec![]
        }
    });
}

#[test]
fn inline_bus_design_with_inout_pins() {
    let mut db = DesignBuilder::new();
    let port = bus_port(&mut db, 2);
    let xcvr = transceiver(&mut db, 2);
    let mut b = DefBuilder::new("bus_top");
    let da = b.input("DA", 2);
    let oea = b.input("OEA", 1);
    let dir = b.input("DIR", 1);
    let oe = b.input("OE", 1);
    let a = b.wire(2);
    let bb = b.wire(2);
    b.terminate(bb, panda_netlist::Termination::PullDown);
    let qa = b.inst_named("pa", &port, &[da, oea, a])[0];
    b.inst_named("x", &xcvr, &[a, bb, dir, oe]);
    b.output("QA", qa);
    b.output("B", bb);
    let top = db.add(b.finish());
    let orig = db.build(&top);
    let inl = inline_all(&orig);
    assert_atoms_only(&inl);
    sim_design(&inl);
    lockstep(&orig, &inl, 200, |t| match t {
        5 => vec![("DA", 2), ("OEA", 1)],
        20 => vec![("DIR", 1)],
        30 => vec![("OE", 1)],
        60 => vec![("DA", 1)],
        90 => vec![("OE", 0)],
        _ => vec![],
    });
}

#[test]
fn inline_is_idempotent_on_flat_designs() {
    let mut db = DesignBuilder::new();
    let a = ripple_adder(&mut db, 3);
    let orig = db.build(&a);
    let once = inline_all(&orig);
    let twice = inline_all(&once);
    assert_eq!(once, twice);
}

#[test]
fn try_inline_reports_malformed_designs() {
    let mut db = DesignBuilder::new();
    let ha = half_adder(&mut db);
    let mut d = db.build(&ha);
    d.top = "nope".into();
    assert!(try_inline_all(&d).unwrap_err().contains("unknown top"));

    // Recursive def.
    let mut b = DefBuilder::new("rec");
    let x = b.input("X", 1);
    let r = DefRef {
        id: "rec".into(),
        pins: vec![panda_netlist::PinDecl { name: "X".into(), dir: panda_netlist::Dir::In, width: 1 }],
    };
    b.inst(&r, &[x]);
    let def = b.finish();
    let d = Design { format: panda_netlist::FORMAT.into(), defs: vec![def], top: "rec".into() };
    assert!(try_inline_all(&d).unwrap_err().contains("recursive"));
}
