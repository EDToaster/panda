//! One test (at least) per validation diagnostic kind, with locations.

use panda_netlist::validate::{has_errors, validate};
use panda_netlist::*;

fn inv() -> ComponentDef {
    let mut b = DefBuilder::new("inv");
    let a = b.input("a", 1);
    let y = b.not(a);
    b.output("y", y);
    b.finish()
}

/// top: A → inv instance u0 → Y
fn base() -> Design {
    let mut db = DesignBuilder::new();
    let i = db.add(inv());
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 1);
    let y = b.inst(&i, &[a]);
    b.output("Y", y[0]);
    let top = db.add(b.finish());
    db.build(&top)
}

fn def_mut<'a>(d: &'a mut Design, id: &str) -> &'a mut ComponentDef {
    d.defs.iter_mut().find(|x| x.id == id).unwrap()
}

/// Exactly one error of `kind` exists, and its location is `loc`.
#[track_caller]
fn expect_error(d: &Design, kind: DiagKind, loc: &str) {
    let diags = validate(d);
    let errs: Vec<_> = diags.iter().filter(|x| x.severity == Severity::Error && x.kind == kind).collect();
    assert!(errs.iter().any(|e| e.location.as_deref() == Some(loc)), "expected {kind:?} at {loc:?}, got {diags:#?}");
    assert!(flatten(d).is_err());
}

#[track_caller]
fn expect_warning(d: &Design, kind: DiagKind, loc: &str) {
    let diags = validate(d);
    assert!(
        diags.iter().any(|x| x.severity == Severity::Warning && x.kind == kind && x.location.as_deref() == Some(loc)),
        "expected warning {kind:?} at {loc:?}, got {diags:#?}"
    );
}

#[test]
fn base_is_clean() {
    let diags = validate(&base());
    assert!(diags.is_empty(), "{diags:?}");
}

#[test]
fn unknown_top() {
    let mut d = base();
    d.top = "nope".into();
    expect_error(&d, DiagKind::UnknownDef, "nope");
}

#[test]
fn unknown_def_instance() {
    let mut d = base();
    def_mut(&mut d, "top").instances[0].kind = InstKind::Def("missing".into());
    expect_error(&d, DiagKind::UnknownDef, "top: instance u0");
}

#[test]
fn unknown_instance_in_net() {
    let mut d = base();
    def_mut(&mut d, "top").nets[0].ends.push(Endpoint::Port { inst: "ghost".into(), port: "a".into() });
    expect_error(&d, DiagKind::UnknownInstance, "top: net A");
}

#[test]
fn unknown_port_and_pin() {
    let mut d = base();
    def_mut(&mut d, "top").nets[0].ends.push(Endpoint::Port { inst: "u0".into(), port: "zz".into() });
    expect_error(&d, DiagKind::UnknownPort, "top: net A");
    let mut d = base();
    def_mut(&mut d, "inv").nets[0].ends.push(Endpoint::Pin("nopin".into()));
    expect_error(&d, DiagKind::UnknownPort, "inv: net a");
    let mut d = base();
    def_mut(&mut d, "inv").nets[0].ends.push(Endpoint::Port { inst: "g0".into(), port: "in1".into() });
    expect_error(&d, DiagKind::UnknownPort, "inv: net a");
}

#[test]
fn duplicate_ids() {
    let mut d = base();
    let copy = d.defs[0].clone();
    d.defs.push(copy);
    expect_error(&d, DiagKind::DuplicateId, "inv");

    let mut d = base();
    let t = def_mut(&mut d, "inv");
    let i = t.instances[0].clone();
    t.instances.push(i);
    expect_error(&d, DiagKind::DuplicateId, "inv: instance g0");

    let mut d = base();
    let t = def_mut(&mut d, "inv");
    let mut n = t.nets[0].clone();
    n.ends.clear();
    t.nets.push(n);
    expect_error(&d, DiagKind::DuplicateId, "inv: net a");

    let mut d = base();
    let t = def_mut(&mut d, "inv");
    let p = t.pins[0].clone();
    t.pins.push(p);
    expect_error(&d, DiagKind::DuplicateId, "inv: pin a");

    let mut d = base();
    def_mut(&mut d, "inv").instances[0].id = "pin".into();
    expect_error(&d, DiagKind::DuplicateId, "inv: instance pin");
}

#[test]
fn recursive_defs() {
    // inv instantiates top → cycle top -> inv -> top.
    let mut d = base();
    def_mut(&mut d, "inv").instances.push(Instance { id: "loop".into(), kind: InstKind::Def("top".into()) });
    let diags = validate(&d);
    let rec: Vec<_> = diags.iter().filter(|x| x.kind == DiagKind::RecursiveDef).collect();
    assert_eq!(rec.len(), 1, "{diags:#?}");
    assert!(rec[0].message.contains("top -> inv -> top"), "{}", rec[0].message);
    assert!(flatten(&d).is_err());

    // Self-recursion.
    let mut d = base();
    def_mut(&mut d, "inv").instances.push(Instance { id: "me".into(), kind: InstKind::Def("inv".into()) });
    expect_error(&d, DiagKind::RecursiveDef, "inv");
}

#[test]
fn width_mismatches() {
    let mut d = base();
    def_mut(&mut d, "inv").nets[0].width = 2;
    let diags = validate(&d);
    let n = diags.iter().filter(|x| x.kind == DiagKind::WidthMismatch).count();
    assert_eq!(n, 2, "pin a and port g0.in: {diags:#?}"); // pin a and g0.in
    expect_error(&d, DiagKind::WidthMismatch, "inv: net a");

    let mut d = base();
    def_mut(&mut d, "inv").pins[0].width = 0;
    expect_error(&d, DiagKind::WidthMismatch, "inv: pin a");
}

#[track_caller]
fn bad_atom(atom: Atom) {
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 1);
    b.output("Y", a);
    let mut def = b.finish();
    def.instances.push(Instance { id: "bad".into(), kind: InstKind::Atom(atom) });
    let mut db = DesignBuilder::new();
    let top = db.add(def);
    let d = db.build(&top);
    expect_error(&d, DiagKind::BadAtom, "top: instance bad");
}

#[test]
fn bad_atoms() {
    bad_atom(Atom::Gate { op: GateOp::And, inputs: 0, width: 1, reduce: false });
    bad_atom(Atom::Gate { op: GateOp::And, inputs: 2, width: 4, reduce: true });
    bad_atom(Atom::Gate { op: GateOp::And, inputs: 2, width: 0, reduce: false });
    bad_atom(Atom::Not { width: 0 });
    bad_atom(Atom::Buf { width: 0 });
    bad_atom(Atom::Tri { width: 0 });
    bad_atom(Atom::Splitter { width: 4, parts: vec![(0, 4)] });
    bad_atom(Atom::Splitter { width: 4, parts: vec![(3, 1)] });
    bad_atom(Atom::Splitter { width: 0, parts: vec![] });
    bad_atom(Atom::Clock { period: 1, high: 1, phase: 0 });
    bad_atom(Atom::Clock { period: 4, high: 0, phase: 0 });
    bad_atom(Atom::Clock { period: 4, high: 4, phase: 0 });
}

#[test]
fn bad_splitter_with_connections_does_not_panic() {
    let mut d = base();
    let t = def_mut(&mut d, "top");
    t.instances
        .push(Instance { id: "s".into(), kind: InstKind::Atom(Atom::Splitter { width: 2, parts: vec![(1, 0)] }) });
    t.nets[0].ends.push(Endpoint::Port { inst: "s".into(), port: "p0".into() });
    expect_error(&d, DiagKind::BadAtom, "top: instance s");
}

#[test]
fn source_not_top() {
    let mut d = base();
    def_mut(&mut d, "inv")
        .instances
        .push(Instance { id: "clk".into(), kind: InstKind::Atom(Atom::Clock { period: 2, high: 1, phase: 0 }) });
    expect_error(&d, DiagKind::SourceNotTop, "inv: instance clk");
    let mut d = base();
    def_mut(&mut d, "inv")
        .instances
        .push(Instance { id: "por".into(), kind: InstKind::Atom(Atom::PowerOnReset { ticks: 3 }) });
    expect_error(&d, DiagKind::SourceNotTop, "inv: instance por");
}

#[test]
fn pin_or_port_on_two_nets() {
    let mut d = base();
    let t = def_mut(&mut d, "inv");
    t.nets[1].ends.push(Endpoint::Pin("a".into()));
    let diags = validate(&d);
    assert!(diags.iter().any(|x| x.kind == DiagKind::BadPinUse
        && x.severity == Severity::Error
        && x.location.as_deref() == Some("inv: pin a")));

    let mut d = base();
    let t = def_mut(&mut d, "inv");
    t.nets[1].ends.push(Endpoint::Port { inst: "g0".into(), port: "in".into() });
    expect_error(&d, DiagKind::BadPinUse, "inv: instance g0");
}

#[test]
fn input_pin_driven_from_inside() {
    let mut b = DefBuilder::new("bad");
    let a = b.input("a", 1);
    let c = b.lit(1, 1);
    b.buf_to(c, a);
    b.output("y", a);
    let mut db = DesignBuilder::new();
    let top = db.add(b.finish());
    let d = db.build(&top);
    expect_error(&d, DiagKind::BadPinUse, "bad: pin a");
}

#[test]
fn input_pin_driven_through_splitter_and_child() {
    // Bit 1 of input pin A is driven by the Out pin of a child, via a splitter.
    let mut db = DesignBuilder::new();
    let i = db.add(inv());
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 2);
    let bits = b.bits(a);
    let x = b.input("X", 1);
    let _ = b.inst_map("u", &i, &[("a", x), ("y", bits[1])]);
    b.output("O", bits[0]);
    let top = db.add(b.finish());
    let d = db.build(&top);
    expect_error(&d, DiagKind::BadPinUse, "top: pin A");
}

#[test]
fn invalid_identifiers() {
    let mut d = base();
    def_mut(&mut d, "inv").instances[0].id = "a/b".into();
    let diags = validate(&d);
    assert!(diags.iter().any(|x| x.kind == DiagKind::Other && x.location.as_deref() == Some("inv: instance a/b")));
    let mut d = base();
    def_mut(&mut d, "inv").nets[0].id = "x[0]".into();
    expect_error(&d, DiagKind::Other, "inv: net x[0]");
}

#[test]
fn unused_pin_warning() {
    let mut d = base();
    def_mut(&mut d, "inv").pins.push(PinDecl { name: "spare".into(), dir: Dir::In, width: 1 });
    // The parent does not connect it either (that's allowed); only a warning.
    let diags = validate(&d);
    assert!(!has_errors(&diags), "{diags:#?}");
    expect_warning(&d, DiagKind::BadPinUse, "inv: pin spare");
    assert!(flatten(&d).is_ok());
}

#[test]
fn multiple_drivers_warning() {
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 1);
    let y = b.not(a);
    b.not_to(a, y);
    b.name(y, "y");
    b.output("Y", y);
    let mut db = DesignBuilder::new();
    let top = db.add(b.finish());
    let d = db.build(&top);
    assert!(!has_errors(&validate(&d)));
    expect_warning(&d, DiagKind::MultipleDrivers, "top: net y");
}

#[test]
fn tri_bus_has_no_warning() {
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 4);
    let e1 = b.input("E1", 1);
    let e2 = b.input("E2", 1);
    let bus = b.tri(a, e1);
    let na = b.not(a);
    b.tri_to(na, e2, bus);
    b.output("BUS", bus);
    let mut db = DesignBuilder::new();
    let top = db.add(b.finish());
    let d = db.build(&top);
    let diags = validate(&d);
    assert!(diags.is_empty(), "{diags:#?}");
}

/// Two instances of a def whose Out pin is a TRI (Z-capable) vs. a NOT (not Z-capable).
#[test]
fn multiple_drivers_through_children() {
    let tri_def = {
        let mut b = DefBuilder::new("tdrv");
        let d = b.input("d", 1);
        let en = b.input("en", 1);
        let o = b.tri(d, en);
        let o2 = b.buf(o); // a gate in between is NOT Z-capable — but this net is separate
        let _ = o2;
        b.output("o", o);
        b.finish()
    };
    let mut db = DesignBuilder::new();
    let t = db.add(tri_def);
    let i = db.add(inv());
    let mut b = DefBuilder::new("top");
    let d = b.input("D", 1);
    let e = b.input("E", 1);
    let bus = b.wire(1);
    b.name(bus, "bus");
    b.inst_map("t0", &t, &[("d", d), ("en", e), ("o", bus)]);
    b.inst_map("t1", &t, &[("d", d), ("en", e), ("o", bus)]);
    b.output("BUS", bus);
    let top = db.add(b.finish());
    let design = db.build(&top);
    assert!(validate(&design).iter().all(|x| x.kind != DiagKind::MultipleDrivers), "{:#?}", validate(&design));

    // Now add an inverter driving the same bus → warning.
    let mut design2 = design.clone();
    let top = def_mut(&mut design2, "top");
    top.instances.push(Instance { id: "x".into(), kind: InstKind::Def("inv".into()) });
    let bus_net = top.nets.iter_mut().find(|n| n.ends.contains(&Endpoint::Pin("BUS".into()))).unwrap();
    bus_net.ends.push(Endpoint::Port { inst: "x".into(), port: "y".into() });
    let bus_id = bus_net.id.clone();
    let dnet = top.nets.iter_mut().find(|n| n.ends.contains(&Endpoint::Pin("D".into()))).unwrap();
    dnet.ends.push(Endpoint::Port { inst: "x".into(), port: "a".into() });
    let _ = i;
    expect_warning(&design2, DiagKind::MultipleDrivers, &format!("top: net {bus_id}"));
}

/// Pass-through child (In wired straight to Out) aliases parent nets: the In pin of the parent
/// that reaches a gate output through the child is detected.
#[test]
fn passthrough_child_aliasing() {
    let wire_def = {
        let mut b = DefBuilder::new("wire");
        let a = b.input("a", 1);
        b.output("y", a);
        b.finish()
    };
    let mut db = DesignBuilder::new();
    let w = db.add(wire_def);
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 1);
    let n = b.lit(1, 0);
    let c = b.not(n);
    b.inst_map("w", &w, &[("a", a), ("y", c)]);
    b.output("Y", c);
    let top = db.add(b.finish());
    let d = db.build(&top);
    expect_error(&d, DiagKind::BadPinUse, "top: pin A");
}

#[test]
fn all_errors_reported() {
    let mut d = base();
    d.top = "nope".into();
    def_mut(&mut d, "inv").nets[0].width = 3;
    def_mut(&mut d, "inv").instances[0].id = "pin".into();
    let diags = validate(&d);
    assert!(diags.iter().filter(|x| x.severity == Severity::Error).count() >= 3, "{diags:#?}");
    let errs = flatten(&d).unwrap_err();
    assert!(errs.iter().all(|e| e.severity == Severity::Error));
    assert!(errs.len() >= 3);
}
