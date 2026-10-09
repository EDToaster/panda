//! Regression tests for bugs found in the adversarial review of panda-circuits.

mod common;

use common::*;
use panda_circuits::transform::try_inline_all;
use panda_circuits::*;
use panda_netlist::{DefBuilder, DesignBuilder, Termination, flatten};

/// Bug: the testbench id ignored `clock_pin` / `reset_pin`, so a second testbench of the same DUT
/// with different pin choices silently returned the first one's def (with the wrong wiring).
#[test]
fn testbench_id_includes_non_default_pin_choices() {
    let mut db = DesignBuilder::new();
    let ff = dff_rst(&mut db);
    let a = testbench(&mut db, &ff, 8);
    let opts = TestbenchOptions { reset_pin: "NONE".into(), ..TestbenchOptions::new(8) };
    let b = testbench_with(&mut db, &ff, &opts);
    assert_ne!(a.id, b.id);
    // Default pins keep the historical id.
    assert_eq!(a.id, "tb_dff_rst_p8_h4_ph4_r18");
    // With no reset source, RST is a user input of the second testbench.
    assert!(a.pins.iter().all(|p| p.name != "RST"));
    assert!(b.pins.iter().any(|p| p.name == "RST"));
    // And it really behaves that way.
    let mut design = db.build(&b);
    design.defs.retain(|d| d.id != a.id);
    let mut s = sim_design(&design);
    set(&mut s, "RST", 1);
    set(&mut s, "D", 1);
    run_to(&mut s, opts.rising_edge(1) + 6);
    assert_eq!(get(&s, "Q"), 0);
    set(&mut s, "RST", 0);
    run_to(&mut s, opts.rising_edge(3) + 6);
    assert_eq!(get(&s, "Q"), 1);
}

/// Bug: `inline_all` silently merged nets with conflicting terminations across a pin (keeping
/// the parent's), producing a valid design from one that flattening rejects.
#[test]
fn inline_rejects_conflicting_terminations_across_pins() {
    let mut db = DesignBuilder::new();
    let mut c = DefBuilder::new("child");
    let io = c.inout("IO", 1);
    c.terminate(io, Termination::PullUp);
    let y = c.not(io);
    c.output("Y", y);
    let child = db.add(c.finish());
    let mut t = DefBuilder::new("top");
    let w = t.wire(1);
    t.terminate(w, Termination::PullDown);
    let y = t.inst(&child, &[w])[0];
    t.output("Y", y);
    let top = db.add(t.finish());
    let d = db.build(&top);
    assert!(flatten(&d).is_err());
    let e = try_inline_all(&d).unwrap_err();
    assert!(e.contains("conflicting terminations"), "{e}");

    // Equal terminations merge fine.
    let mut d2 = d.clone();
    let td = d2.defs.iter_mut().find(|x| x.id == "top").unwrap();
    for n in &mut td.nets {
        if n.termination == Termination::PullDown {
            n.termination = Termination::PullUp;
        }
    }
    let inl = try_inline_all(&d2).unwrap();
    assert!(inl.defs[0].nets.iter().any(|n| n.termination == Termination::PullUp));
}

/// Bug: `inline_all` ignored connections to pins a child def does not have (and width
/// mismatches at pins that are on no internal net) instead of reporting a malformed design.
#[test]
fn inline_rejects_unknown_or_mis_sized_child_pins() {
    let mut db = DesignBuilder::new();
    let ha = half_adder(&mut db);
    let mut t = DefBuilder::new("top");
    let a = t.input("A", 1);
    let o = t.inst_named("u", &ha, &[a, a]);
    t.output("S", o[0]);
    let top = db.add(t.finish());
    let d = db.build(&top);
    assert!(try_inline_all(&d).is_ok());

    let mut bad = d.clone();
    let td = bad.defs.iter_mut().find(|x| x.id == "top").unwrap();
    for n in &mut td.nets {
        for e in &mut n.ends {
            if let panda_netlist::Endpoint::Port { port, .. } = e
                && port == "B"
            {
                *port = "NOPE".into();
            }
        }
    }
    assert!(try_inline_all(&bad).unwrap_err().contains("no pin 'NOPE'"));

    // A child pin that is on no internal net, connected with the wrong width.
    let mut d3 = d.clone();
    let hd = d3.defs.iter_mut().find(|x| x.id == "half_adder").unwrap();
    hd.pins.push(panda_netlist::PinDecl { name: "W".into(), dir: panda_netlist::Dir::In, width: 2 });
    let td = d3.defs.iter_mut().find(|x| x.id == "top").unwrap();
    let a_net = td.nets.iter_mut().find(|n| n.id == "A").unwrap();
    a_net.ends.push(panda_netlist::Endpoint::Port { inst: "u".into(), port: "W".into() });
    assert!(try_inline_all(&d3).unwrap_err().contains("width mismatch"));
}
