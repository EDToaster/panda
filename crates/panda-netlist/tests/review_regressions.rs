//! Regression tests for bugs found in the adversarial review of validation and flattening, plus
//! explicit checks of flattening corner cases from the `flat.rs` contract.

use panda_netlist::flat::{Driver, FlatNetlist, SourceKind};
use panda_netlist::validate::{has_errors, validate};
use panda_netlist::*;

fn errors(d: &Design) -> Vec<Diagnostic> {
    validate(d).into_iter().filter(|x| x.severity == Severity::Error).collect()
}

fn flat(d: &Design) -> FlatNetlist {
    flatten(d).unwrap_or_else(|e| panic!("flatten failed: {e:#?}"))
}

/// Remove the endpoint `inst.port` from every net (leaves that instance pin unconnected).
fn unconnect(mut def: ComponentDef, inst: &str, port: &str) -> ComponentDef {
    let e = Endpoint::Port { inst: inst.into(), port: port.into() };
    for n in &mut def.nets {
        n.ends.retain(|x| *x != e);
    }
    def.nets.retain(|n| !n.ends.is_empty());
    def
}

fn net(f: &FlatNetlist, name: &str) -> u32 {
    f.net_by_name(name).unwrap_or_else(|| panic!("no net '{name}'"))
}

/// Child with three inout pins `X`, `Y`, `Z` shorted together on one internal net.
fn short3() -> ComponentDef {
    ComponentDef {
        id: "short3".into(),
        name: String::new(),
        version: 0,
        pins: ["X", "Y", "Z"].iter().map(|n| PinDecl { name: (*n).into(), dir: Dir::InOut, width: 1 }).collect(),
        instances: vec![],
        nets: vec![Net {
            id: "s".into(),
            width: 1,
            termination: Termination::None,
            ends: vec![Endpoint::Pin("X".into()), Endpoint::Pin("Y".into()), Endpoint::Pin("Z".into())],
        }],
        editor: serde_json::Value::Null,
    }
}

/// Bug: validation joined child pins only through the class representative (the first pin of the
/// internal alias class). With the representative (`X`) unconnected at the instance, `Y` and `Z`
/// were not joined, so an `In` pin driven through the child went unreported (and multi-driver
/// warnings through such children were missed).
#[test]
fn input_pin_driven_through_child_with_unconnected_representative() {
    let mut db = DesignBuilder::new();
    let s = db.add(short3());
    let mut b = DefBuilder::new("pkg");
    let p = b.input("P", 1);
    let c = b.constant(Value::from_u64(1, 1));
    let x = b.wire(1);
    b.inst_map("u", &s, &[("X", x), ("Y", p), ("Z", c)]);
    b.output("O", p);
    let pkg = db.add(unconnect(b.finish(), "u", "X"));
    let mut t = DefBuilder::new("top");
    let i = t.input("I", 1);
    let o = t.inst(&pkg, &[i])[0];
    t.output("O", o);
    let top = db.add(t.finish());
    let d = db.build(&top);
    let e = errors(&d);
    assert!(
        e.iter().any(|x| x.kind == DiagKind::BadPinUse && x.location.as_deref() == Some("pkg: pin P")),
        "{e:#?}"
    );
}

#[test]
fn multiple_drivers_through_child_with_unconnected_representative() {
    let mut db = DesignBuilder::new();
    let s = db.add(short3());
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 1);
    let y = b.not(a);
    let z = b.buf(a);
    let x = b.wire(1);
    b.inst_map("u", &s, &[("X", x), ("Y", y), ("Z", z)]);
    b.output("Y", y);
    let top = db.add(unconnect(b.finish(), "u", "X"));
    let d = db.build(&top);
    let diags = validate(&d);
    assert!(diags.iter().any(|x| x.kind == DiagKind::MultipleDrivers), "{diags:#?}");
    // Flattening agrees: one resolver with both gates.
    let f = flat(&d);
    assert_eq!(f.resolvers.len(), 1);
    assert_eq!(f.resolvers[0].drivers.len(), 2);
    assert_eq!(net(&f, "u/X"), net(&f, "Y"));
}

/// Missing check: conflicting terminations were only detected by `flatten`, so a design could
/// validate cleanly and still fail to flatten.
#[test]
fn termination_conflict_through_splitter_is_a_validation_error() {
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 2);
    let w = b.wire(2);
    b.terminate(w, Termination::PullUp);
    let lo = b.slice(w, 0, 1);
    b.terminate(lo, Termination::PullDown);
    let t = b.tri(a, lo);
    b.connect(t, w);
    b.output("W", w);
    let mut db = DesignBuilder::new();
    let top = db.add(b.finish());
    let d = db.build(&top);
    let e = errors(&d);
    assert_eq!(e.len(), 1, "{e:#?}");
    assert!(e[0].message.contains("conflicting terminations"));
    assert!(flatten(&d).is_err());
}

#[test]
fn termination_conflict_between_two_children_is_a_validation_error() {
    let child = |id: &str, t: Termination| {
        let mut b = DefBuilder::new(id);
        let io = b.inout("IO", 4);
        b.terminate(io, t);
        let y = b.not(io);
        b.output("Y", y);
        b.finish()
    };
    let mut db = DesignBuilder::new();
    let up = db.add(child("up", Termination::PullUp));
    let down = db.add(child("down", Termination::PullDown));
    let mut b = DefBuilder::new("top");
    let bus = b.wire(4);
    b.name(bus, "bus");
    let y1 = b.inst(&up, &[bus])[0];
    let y2 = b.inst(&down, &[bus])[0];
    b.output("Y1", y1);
    b.output("Y2", y2);
    let top = db.add(b.finish());
    let d = db.build(&top);
    let e = errors(&d);
    assert_eq!(e.len(), 1, "one error per net, not per bit: {e:#?}");
    assert_eq!(e[0].location.as_deref(), Some("top: net bus"));
    assert!(flatten(&d).is_err());

    // The child's termination is visible two levels up (through a pass-through wrapper).
    let mut db = DesignBuilder::new();
    let keep = db.add(child("keep", Termination::Keeper));
    let mut w = DefBuilder::new("wrap");
    let io = w.inout("IO", 4);
    let y = w.inst(&keep, &[io])[0];
    w.output("Y", y);
    let wrap = db.add(w.finish());
    let mut b = DefBuilder::new("top2");
    let bus = b.wire(4);
    b.name(bus, "bus");
    b.terminate(bus, Termination::PullDown);
    let y = b.inst(&wrap, &[bus])[0];
    b.output("Y", y);
    let top2 = db.add(b.finish());
    let d = db.build(&top2);
    assert!(has_errors(&validate(&d)));
    // Same termination: fine.
    let mut d = d;
    let t2 = d.defs.iter_mut().find(|x| x.id == "top2").unwrap();
    t2.nets.iter_mut().find(|n| n.id == "bus").unwrap().termination = Termination::Keeper;
    assert!(!has_errors(&validate(&d)));
    let f = flat(&d);
    assert_eq!(f.undriven_termination.get(&net(&f, "bus[2]")), Some(&Termination::Keeper));
}

/// Bug: private resolver nets are named `<canonical>~<k>`, but identifiers may contain `~`. With a
/// real net called `OUT~0`, the private net's name was shadowed in `name_index` (it resolved to
/// the other net).
#[test]
fn private_resolver_names_never_shadowed() {
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 1);
    let o = b.wire(1);
    b.tri_to(a, a, o);
    let na = b.not(a);
    b.tri_to(na, na, o);
    b.output("OUT", o);
    let x = b.buf(a);
    b.name(x, "OUT~0");
    b.output("X", x);
    let mut db = DesignBuilder::new();
    let top = db.add(b.finish());
    let f = flat(&db.build(&top));
    let r = &f.resolvers[0];
    for &p in &r.drivers {
        assert_eq!(f.net_by_name(&f.net_names[p as usize]), Some(p));
        assert!(f.net_names[p as usize].starts_with("OUT~"));
    }
    assert_eq!(f.net_names[net(&f, "OUT~0") as usize], "X");
    let mut names = f.net_names.clone();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), f.net_names.len(), "net names are unique");
}

#[test]
fn zero_instance_designs() {
    // Pins only: an In pin wired straight to an Out pin, plus an unconnected Out pin.
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 70);
    b.output("Y", a);
    let mut def = b.finish();
    def.pins.push(PinDecl { name: "Z".into(), dir: Dir::Out, width: 3 });
    let d = Design { format: FORMAT.into(), defs: vec![def], top: "top".into() };
    let f = flat(&d);
    assert_eq!(f.gates.len(), 0);
    assert_eq!(f.num_nets, 73);
    assert_eq!(f.sources.len(), 70);
    assert_eq!(f.top_outputs[0].nets, f.top_inputs[0].nets);
    let z = &f.top_outputs[1];
    let drv = f.drivers();
    assert!(z.nets.iter().all(|&n| drv[n as usize] == Driver::None));
    assert_eq!(f.net_names[z.nets[2] as usize], "Z[2]");
    // Completely empty top.
    let empty = Design {
        format: FORMAT.into(),
        defs: vec![ComponentDef {
            id: "e".into(),
            name: String::new(),
            version: 0,
            pins: vec![],
            instances: vec![],
            nets: vec![],
            editor: serde_json::Value::Null,
        }],
        top: "e".into(),
    };
    let f = flat(&empty);
    assert_eq!((f.num_nets, f.instances.len()), (0, 1));
}

#[test]
fn unconnected_child_input_is_undriven_inside() {
    // Child: Y = NOT A, with a pulled-up In pin B feeding Q = BUF B. A and B left unconnected.
    let mut c = DefBuilder::new("child");
    let a = c.input("A", 2);
    let bb = c.input("B", 1);
    c.terminate(bb, Termination::PullUp);
    let y = c.not(a);
    let q = c.buf(bb);
    c.output("Y", y);
    c.output("Q", q);
    let mut db = DesignBuilder::new();
    let child = db.add(c.finish());
    let mut t = DefBuilder::new("top");
    let (wa, wb) = (t.wire(2), t.wire(1));
    let m = t.inst_map("u", &child, &[("A", wa), ("B", wb)]);
    t.output("Y", m["Y"]);
    t.output("Q", m["Q"]);
    let top = db.add(unconnect(unconnect(t.finish(), "u", "A"), "u", "B"));
    let d = db.build(&top);
    let f = flat(&d);
    let drv = f.drivers();
    for name in ["u/A[0]", "u/A[1]", "u/B"] {
        assert_eq!(drv[net(&f, name) as usize], Driver::None, "{name}");
    }
    assert!(!f.undriven_termination.contains_key(&net(&f, "u/A[0]")));
    assert_eq!(f.undriven_termination.get(&net(&f, "u/B")), Some(&Termination::PullUp));
    // The instance pins point at the internal nets.
    let u = f.instance_by_path("u").unwrap() as usize;
    assert_eq!(f.instances[u].pins[0].nets, vec![net(&f, "u/A[0]"), net(&f, "u/A[1]")]);
}

#[test]
fn two_child_outputs_and_a_constant_on_one_net() {
    let mut c = DefBuilder::new("drv");
    let a = c.input("A", 1);
    let y = c.not(a);
    c.output("Y", y);
    let mut db = DesignBuilder::new();
    let drv = db.add(c.finish());
    let mut t = DefBuilder::new("top");
    let a = t.input("A", 1);
    let w = t.wire(1);
    t.name(w, "w");
    t.terminate(w, Termination::PullDown);
    t.inst_map("d0", &drv, &[("A", a), ("Y", w)]);
    t.inst_map("d1", &drv, &[("A", a), ("Y", w)]);
    let k = t.constant(Value::parse(1, "z").unwrap());
    t.connect(k, w);
    t.output("W", w);
    let top = db.add(t.finish());
    let d = db.build(&top);
    let f = flat(&d);
    assert_eq!(f.resolvers.len(), 1);
    let r = &f.resolvers[0];
    assert_eq!(r.output, net(&f, "w"));
    assert_eq!(r.termination, Termination::PullDown);
    assert_eq!(r.drivers.len(), 3);
    let drvs = f.drivers();
    let kinds: Vec<_> = r.drivers.iter().map(|&p| drvs[p as usize]).collect();
    assert_eq!(kinds.iter().filter(|k| matches!(k, Driver::Gate(_))).count(), 2);
    let src = kinds.iter().find_map(|k| if let Driver::Source(s) = k { Some(*s) } else { None }).unwrap();
    assert_eq!(f.sources[src as usize].kind, SourceKind::Constant(Logic::Z));
    // Aliases of the child pins resolve to the resolved net, not to the private ones.
    assert_eq!(net(&f, "d0/Y"), r.output);
    assert_eq!(net(&f, "d1/Y"), r.output);
}

#[test]
fn wide_net_through_splitters_both_directions_and_levels() {
    // Child splits a 100-bit inout bus into halves and swaps them onto an Out pin through BUFs;
    // parent builds the bus from two 50-bit constants via a joiner.
    let mut c = DefBuilder::new("swap");
    let bus = c.inout("BUS", 100);
    let lo = c.slice(bus, 0, 50);
    let hi = c.slice(bus, 50, 50);
    let lo_b = c.buf(lo);
    let hi_b = c.buf(hi);
    let y = c.concat(&[hi_b, lo_b]);
    c.output("Y", y);
    let mut db = DesignBuilder::new();
    let swap = db.add(c.finish());
    let mut t = DefBuilder::new("top");
    let a = t.input("A", 50);
    let k = t.constant(Value::ones(50));
    let bus = t.concat(&[a, k]);
    let y = t.inst_named("s", &swap, &[bus])[0];
    t.output("Y", y);
    let top = db.add(t.finish());
    let f = flat(&db.build(&top));
    for j in 0..50 {
        assert_eq!(net(&f, &format!("s/BUS[{j}]")), f.top_inputs[0].nets[j as usize]);
        assert_eq!(net(&f, &format!("A[{j}]")), net(&f, &format!("s/BUS[{j}]")));
    }
    // Output bit 50+j comes from a BUF reading input bit j.
    for j in 0..50usize {
        let o = f.top_outputs[0].nets[50 + j];
        let g = f.gates.iter().find(|g| g.output == o).unwrap();
        assert_eq!(g.inputs, vec![f.top_inputs[0].nets[j]]);
    }
    assert_eq!(f.groups.len(), 2);
    assert!(f.groups.iter().all(|g| g.gates.len() == 50 && g.path.starts_with("s/")));
}

#[test]
fn reduce_gates_in_child_and_one_input_gates() {
    let mut c = DefBuilder::new("red");
    let a = c.input("A", 66);
    let r = c.reduce(GateOp::Xor, a);
    let one = c.gate(GateOp::Nand, &[a]);
    let r1 = c.reduce(GateOp::And, r);
    c.output("R", r1);
    c.output("N", one);
    let mut db = DesignBuilder::new();
    let red = db.add(c.finish());
    let mut t = DefBuilder::new("top");
    let a = t.input("A", 66);
    let o = t.inst_named("u", &red, &[a]);
    t.output("R", o[0]);
    t.output("N", o[1]);
    let top = db.add(t.finish());
    let f = flat(&db.build(&top));
    let red_g = f.gates.iter().find(|g| g.inputs.len() == 66).unwrap();
    assert_eq!(red_g.inputs, f.top_inputs[0].nets);
    assert_eq!(red_g.bit, 0);
    assert_eq!(f.groups[red_g.group as usize].gates.len(), 1);
    assert_eq!(f.instances[red_g.owner as usize].path, "u");
    // A width-1 reduce is a 1-input gate.
    let r1 = f.gates.iter().find(|g| g.output == f.top_outputs[0].nets[0]).unwrap();
    assert_eq!(r1.inputs, vec![red_g.output]);
    // A 1-input non-reduce gate is width-polymorphic: 66 one-input gates.
    let nand = f.groups.iter().find(|g| g.gates.len() == 66).unwrap();
    for (j, &gi) in nand.gates.iter().enumerate() {
        let g = &f.gates[gi as usize];
        assert_eq!((g.inputs.len(), g.bit), (1, j as u32));
    }
}

#[test]
fn clock_inside_nested_package_is_rejected() {
    let mut c = DefBuilder::new("inner");
    let k = c.clock(4, 2, 0);
    c.output("C", k);
    let mut db = DesignBuilder::new();
    let inner = db.add(c.finish());
    let mut m = DefBuilder::new("mid");
    let o = m.inst(&inner, &[])[0];
    m.output("C", o);
    let mid = db.add(m.finish());
    let mut t = DefBuilder::new("top");
    let o = t.inst(&mid, &[])[0];
    let r = t.power_on_reset(3);
    t.output("C", o);
    t.output("R", r);
    let top = db.add(t.finish());
    let d = db.build(&top);
    let e = errors(&d);
    assert_eq!(e.len(), 1, "{e:#?}");
    assert_eq!(e[0].kind, DiagKind::SourceNotTop);
    assert_eq!(e[0].location.as_deref(), Some("inner: instance g0"));
}

#[test]
fn recursion_variants() {
    let mk = |id: &str, child: Option<&str>| ComponentDef {
        id: id.into(),
        name: String::new(),
        version: 0,
        pins: vec![],
        instances: child.map(|c| Instance { id: "u".into(), kind: InstKind::Def(c.into()) }).into_iter().collect(),
        nets: vec![],
        editor: serde_json::Value::Null,
    };
    // Self-recursion, a 3-cycle below the top, and a cycle not reachable from the top.
    let cases: Vec<(Vec<ComponentDef>, usize)> = vec![
        (vec![mk("top", Some("top"))], 1),
        (vec![mk("top", Some("a")), mk("a", Some("b")), mk("b", Some("c")), mk("c", Some("a"))], 1),
        (vec![mk("top", None), mk("x", Some("y")), mk("y", Some("x"))], 1),
        (vec![mk("top", Some("a")), mk("a", Some("b")), mk("b", None)], 0),
    ];
    for (defs, n) in cases {
        let d = Design { format: FORMAT.into(), defs, top: "top".into() };
        let rec = validate(&d).into_iter().filter(|x| x.kind == DiagKind::RecursiveDef).count();
        assert_eq!(rec, n, "{d:?}");
        assert_eq!(flatten(&d).is_err(), n > 0);
    }
}
