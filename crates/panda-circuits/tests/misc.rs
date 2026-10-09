//! Ring oscillator, tri-state bus blocks and testbench plumbing (Timing mode).

mod common;

use common::*;
use panda_circuits::*;
use panda_netlist::{DefBuilder, DesignBuilder, DiagKind, Dir, Logic, Termination};

#[test]
fn ring_oscillator_with_enable() {
    for n in [3u32, 5, 7] {
        let mut db = DesignBuilder::new();
        let r = ring_oscillator(&mut db, n, true);
        assert_eq!(r.id, format!("ring_osc_en_{n}"));
        let mut s = sim_top(db, &r);
        settle(&mut s);
        assert_eq!(bit(&s, "OUT"), Logic::One, "disabled ring holds OUT = 1");
        set(&mut s, "EN", 1);
        let mut changes = Vec::new();
        let mut last = bit(&s, "OUT");
        for _ in 0..20 * n {
            s.step();
            let v = bit(&s, "OUT");
            assert!(v.is_known());
            if v != last {
                changes.push(s.tick());
                last = v;
            }
        }
        assert!(changes.len() >= 10, "n={n}: oscillates");
        for w in changes.windows(2) {
            assert_eq!(w[1] - w[0], u64::from(n), "n={n}: half period = n ticks");
        }
        // Disable again: stops.
        set(&mut s, "EN", 0);
        settle(&mut s);
        assert_eq!(bit(&s, "OUT"), Logic::One);
        assert_clean(&mut s);
    }
}

#[test]
fn ring_oscillator_without_enable_stays_x() {
    let mut db = DesignBuilder::new();
    let r = ring_oscillator(&mut db, 5, false);
    assert!(r.pins.iter().all(|p| p.dir == Dir::Out));
    let mut s = sim_top(db, &r);
    s.run(100);
    assert_eq!(bit(&s, "OUT"), Logic::X);
}

#[test]
fn bus_ports_share_a_pulled_down_bus() {
    let mut db = DesignBuilder::new();
    let port = bus_port(&mut db, 4);
    let mut b = DefBuilder::new("bus_top");
    let d0 = b.input("D0", 4);
    let oe0 = b.input("OE0", 1);
    let d1 = b.input("D1", 4);
    let oe1 = b.input("OE1", 1);
    let bus = b.wire(4);
    b.terminate(bus, Termination::PullDown);
    let q0 = b.inst_named("p0", &port, &[d0, oe0, bus])[0];
    let q1 = b.inst_named("p1", &port, &[d1, oe1, bus])[0];
    b.output("Q0", q0);
    b.output("Q1", q1);
    b.output("BUS", bus);
    let top = db.add(b.finish());
    let mut s = sim_top(db, &top);
    set(&mut s, "D0", 0xa);
    set(&mut s, "D1", 0x5);
    settle(&mut s);
    assert_eq!(get(&s, "BUS"), 0, "released bus reads the pull-down");
    set(&mut s, "OE0", 1);
    settle(&mut s);
    assert_eq!((get(&s, "BUS"), get(&s, "Q0"), get(&s, "Q1")), (0xa, 0xa, 0xa));
    set(&mut s, "OE0", 0);
    set(&mut s, "OE1", 1);
    settle(&mut s);
    assert_eq!((get(&s, "BUS"), get(&s, "Q0")), (0x5, 0x5));
    assert_clean(&mut s);
    // Both drive different values: contention → X.
    set(&mut s, "OE0", 1);
    settle(&mut s);
    assert!(!s.read("BUS").unwrap().is_fully_known());
    let kinds: Vec<_> = s.take_diagnostics().into_iter().map(|d| d.kind).collect();
    assert!(kinds.contains(&DiagKind::Contention), "{kinds:?}");
    // Both drive the same value: fine.
    set(&mut s, "D1", 0xa);
    settle(&mut s);
    assert_eq!(get(&s, "BUS"), 0xa);
}

#[test]
fn transceiver_both_directions() {
    let w = 8;
    let mut db = DesignBuilder::new();
    let port = bus_port(&mut db, w);
    let xcvr = transceiver(&mut db, w);
    let names: Vec<_> = xcvr.pins.iter().map(|p| (p.name.as_str(), p.dir)).collect();
    assert_eq!(names, [("A", Dir::InOut), ("B", Dir::InOut), ("DIR", Dir::In), ("OE", Dir::In)]);
    let mut b = DefBuilder::new("xcvr_top");
    let da = b.input("DA", w);
    let oea = b.input("OEA", 1);
    let dbv = b.input("DB", w);
    let oeb = b.input("OEB", 1);
    let dir = b.input("DIR", 1);
    let oe = b.input("OE", 1);
    let a = b.wire(w);
    let bb = b.wire(w);
    b.terminate(bb, Termination::PullUp);
    let qa = b.inst_named("pa", &port, &[da, oea, a])[0];
    let qb = b.inst_named("pb", &port, &[dbv, oeb, bb])[0];
    b.inst_named("x", &xcvr, &[a, bb, dir, oe]);
    b.output("QA", qa);
    b.output("QB", qb);
    let top = db.add(b.finish());
    let mut s = sim_top(db, &top);
    // OE = 0: B side floats to the pull-up.
    set(&mut s, "DA", 0x3c);
    set(&mut s, "OEA", 1);
    settle(&mut s);
    assert_eq!(get(&s, "QA"), 0x3c);
    assert_eq!(get(&s, "QB"), 0xff);
    // A → B (DIR is set up before OE, as on a real 74245).
    set(&mut s, "DIR", 1);
    settle(&mut s);
    set(&mut s, "OE", 1);
    settle(&mut s);
    assert_eq!(get(&s, "QB"), 0x3c);
    // Bus turnaround without contention: release everything, then reverse, then enable.
    set(&mut s, "OE", 0);
    set(&mut s, "OEA", 0);
    settle(&mut s);
    set(&mut s, "DIR", 0);
    set(&mut s, "DB", 0xc5);
    set(&mut s, "OEB", 1);
    settle(&mut s);
    set(&mut s, "OE", 1);
    settle(&mut s);
    // B → A.
    assert_eq!(get(&s, "QA"), 0xc5);
    assert_eq!(get(&s, "QB"), 0xc5);
    assert_clean(&mut s);
}

#[test]
fn testbench_maps_pins() {
    let mut db = DesignBuilder::new();
    let rf = register_file(&mut db, 4, 8, RegFileStyle::EnableMux);
    let tb = testbench(&mut db, &rf, 40);
    // CLK and RST are sources; the other inputs and the output are re-exported in order.
    let names: Vec<_> = tb.pins.iter().map(|p| (p.name.as_str(), p.dir, p.width)).collect();
    assert_eq!(
        names,
        [("WD", Dir::In, 8), ("WA", Dir::In, 2), ("WE", Dir::In, 1), ("RA", Dir::In, 2), ("RD", Dir::Out, 8),]
    );
    // Same parameters → same def.
    assert_eq!(testbench(&mut db, &rf, 40), tb);
    let design = db.build(&tb);
    let top = design.top_def().unwrap();
    let clocks = top
        .instances
        .iter()
        .filter(|i| {
            matches!(
                &i.kind,
                panda_netlist::InstKind::Atom(panda_netlist::Atom::Clock { period: 40, high: 20, phase: 20 })
            )
        })
        .count();
    assert_eq!(clocks, 1);
}

#[test]
fn testbench_options_edges() {
    let o = TestbenchOptions::new(40);
    assert_eq!(o.rising_edge(0), 20);
    assert_eq!(o.rising_edge(3), 140);
    assert_eq!(o.reset_ticks, 90);
    assert_eq!(o.first_edge_after_reset(), 2);
    let o2 = TestbenchOptions { reset_ticks: 100, ..o.clone() };
    assert_eq!(o2.first_edge_after_reset(), 3, "an edge at the release tick still sees reset");
    let o3 = TestbenchOptions { reset_ticks: 5, ..o };
    assert_eq!(o3.first_edge_after_reset(), 0);
}

#[test]
fn testbench_inout_pins_pass_through() {
    let mut db = DesignBuilder::new();
    let port = bus_port(&mut db, 2);
    let tb = testbench(&mut db, &port, 8);
    let names: Vec<_> = tb.pins.iter().map(|p| (p.name.as_str(), p.dir)).collect();
    assert_eq!(names, [("D", Dir::In), ("OE", Dir::In), ("BUS", Dir::InOut), ("Q", Dir::Out)]);
    let mut s = sim_top(db, &tb);
    set(&mut s, "D", 2);
    set(&mut s, "OE", 1);
    settle(&mut s);
    assert_eq!(get(&s, "BUS"), 2);
    assert_eq!(get(&s, "Q"), 2);
}
