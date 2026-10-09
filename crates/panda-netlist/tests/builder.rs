//! Builder behaviour and JSON round-trips.

use panda_netlist::validate::{has_errors, validate};
use panda_netlist::*;

fn sr_latch() -> ComponentDef {
    let mut b = DefBuilder::new("sr_latch");
    let s_n = b.input("S_n", 1);
    let r_n = b.input("R_n", 1);
    let qn = b.wire(1);
    let q = b.nand(&[s_n, qn]);
    b.nand_to(&[r_n, q], qn);
    b.output("Q", q);
    b.output("Q_n", qn);
    b.finish()
}

fn full_adder() -> ComponentDef {
    let mut b = DefBuilder::new("fa");
    let a = b.input("A", 1);
    let bb = b.input("B", 1);
    let c = b.input("Cin", 1);
    let x = b.xor(&[a, bb]);
    b.name(x, "axb");
    let s = b.xor(&[x, c]);
    let t1 = b.and(&[a, bb]);
    let t2 = b.and(&[x, c]);
    let co = b.or(&[t1, t2]);
    b.output("S", s);
    b.output("Cout", co);
    b.finish()
}

fn adder4() -> Design {
    let mut db = DesignBuilder::new();
    let fa = db.add_once(full_adder());
    let fa2 = db.add_once(full_adder());
    assert_eq!(fa, fa2);
    let mut b = DefBuilder::new("add4");
    let a = b.input("A", 4);
    let bb = b.input("B", 4);
    let mut c = b.input("Cin", 1);
    let abits = b.bits(a);
    let bbits = b.bits(bb);
    let mut sums = Vec::new();
    for i in 0..4 {
        let o = b.inst(&fa, &[abits[i], bbits[i], c]);
        sums.push(o[0]);
        c = o[1];
    }
    let s = b.concat(&sums);
    b.output("S", s);
    b.output("Cout", c);
    let k = b.clock(4, 2, 0);
    let _ = k;
    let top = db.add(b.finish());
    db.build(&top)
}

#[test]
fn sr_latch_structure() {
    let d = sr_latch();
    assert_eq!(d.instances.len(), 2);
    assert_eq!(d.instances[0].id, "g0");
    assert_eq!(d.instances[1].id, "g1");
    let ids: Vec<_> = d.nets.iter().map(|n| n.id.as_str()).collect();
    assert_eq!(ids, ["S_n", "R_n", "Q_n", "Q"]);
    let q = d.nets.iter().find(|n| n.id == "Q").unwrap();
    assert!(q.ends.contains(&Endpoint::Pin("Q".into())));
    assert!(q.ends.contains(&Endpoint::Port { inst: "g0".into(), port: "out".into() }));
    assert!(q.ends.contains(&Endpoint::Port { inst: "g1".into(), port: "in1".into() }));
}

#[test]
fn json_roundtrip_builder_designs() {
    for design in [adder4(), {
        let mut db = DesignBuilder::new();
        let top = db.add(sr_latch());
        db.build(&top)
    }] {
        let diags = validate(&design);
        assert!(!has_errors(&diags), "{diags:?}");
        let json = design.to_json();
        let back = Design::from_json(&json).unwrap();
        assert_eq!(back, design);
        assert_eq!(back.to_json(), json);
    }
}

#[test]
fn json_roundtrip_all_atoms() {
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 3);
    let en = b.input("EN", 1);
    let io = b.inout("IO", 3);
    let t = b.tri(a, en);
    b.connect(t, io);
    let n = b.not(a);
    let bf = b.buf(n);
    let r = b.reduce(GateOp::Xnor, bf);
    let c = b.constant(Value::parse(3, "0b1xz").unwrap());
    let w = b.wire(3);
    b.terminate(w, Termination::Keeper);
    b.connect(w, c);
    let p = b.power_on_reset(5);
    let k = b.clock(10, 3, 7);
    let all = b.concat(&[r, p, k]);
    b.output("O", all);
    let mut db = DesignBuilder::new();
    let top = db.add(b.finish());
    let d = db.build(&top);
    assert!(!has_errors(&validate(&d)));
    let back = Design::from_json(&d.to_json()).unwrap();
    assert_eq!(back, d);
    assert!(d.to_json().contains("\"keeper\""));
}

#[test]
fn auto_ids_skip_named() {
    let mut b = DefBuilder::new("x");
    let a = b.input("A", 1);
    let lib = DefRef { id: "leaf".into(), pins: vec![PinDecl { name: "i".into(), dir: Dir::In, width: 1 }] };
    b.inst_named("u0", &lib, &[a]);
    b.inst(&lib, &[a]);
    let _ = b.not(a);
    let d = b.finish();
    let ids: Vec<_> = d.instances.iter().map(|i| i.id.as_str()).collect();
    assert_eq!(ids, ["u0", "u1", "g0"]);
}

#[test]
fn net_naming_rules() {
    let mut b = DefBuilder::new("x");
    let a = b.input("A", 1);
    let n1 = b.not(a);
    b.name(n1, "foo");
    let n2 = b.not(n1);
    b.name(n2, "foo"); // collides → foo_1
    let n3 = b.not(n2);
    b.name(n3, "A"); // collides with pin A of another net → A_1
    let n4 = b.not(n3);
    b.name(n4, "n0"); // explicit name that looks auto
    let n5 = b.not(n4); // auto: skips n0
    b.name(n5, "first");
    b.name(n5, "second"); // last call wins
    let n6 = b.not(n5);
    b.output("Y", n6);
    let _unused = b.wire(4); // dropped: no endpoints
    let d = b.finish();
    let ids: Vec<_> = d.nets.iter().map(|n| n.id.as_str()).collect();
    assert_eq!(ids, ["A", "foo", "foo_1", "A_1", "n0", "second", "Y"]);
}

#[test]
fn connect_merges_names_and_terminations() {
    let mut b = DefBuilder::new("x");
    let w1 = b.wire(2);
    let w2 = b.wire(2);
    b.name(w1, "old");
    b.name(w2, "new");
    b.terminate(w2, Termination::PullDown);
    b.connect(w1, w2);
    b.output("O", w1);
    let d = b.finish();
    assert_eq!(d.nets.len(), 1);
    assert_eq!(d.nets[0].id, "new");
    assert_eq!(d.nets[0].termination, Termination::PullDown);
    assert_eq!(d.nets[0].width, 2);
}

#[test]
fn pin_name_used_when_no_explicit_name() {
    let mut b = DefBuilder::new("x");
    let a = b.input("A", 1);
    b.output("Y", a); // pass-through: class has pins A and Y
    let d = b.finish();
    assert_eq!(d.nets.len(), 1);
    assert_eq!(d.nets[0].id, "A");
    assert_eq!(d.nets[0].ends.len(), 2);
}

#[test]
fn inst_map_by_name() {
    let mut db = DesignBuilder::new();
    let fa = db.add(full_adder());
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 1);
    let s = b.wire(1);
    let m = b.inst_map("adder", &fa, &[("A", a), ("B", a), ("Cin", a), ("S", s)]);
    assert_eq!(m.len(), 5);
    assert_eq!(m["S"], s);
    assert_eq!(m["A"], a);
    b.output("S", m["S"]);
    b.output("C", m["Cout"]);
    let top = db.add(b.finish());
    let d = db.build(&top);
    assert!(!has_errors(&validate(&d)), "{:?}", validate(&d));
}

#[test]
#[should_panic(expected = "must be connected")]
fn inst_map_missing_input_panics() {
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 1);
    let fa = DefRef::of(&full_adder());
    b.inst_map("adder", &fa, &[("A", a)]);
}

#[test]
#[should_panic(expected = "expected width")]
fn width_mismatch_panics() {
    let mut b = DefBuilder::new("x");
    let a = b.input("A", 1);
    let c = b.input("C", 2);
    b.and(&[a, c]);
}

#[test]
#[should_panic(expected = "different builder")]
fn foreign_sig_panics() {
    let mut b1 = DefBuilder::new("x");
    let mut b2 = DefBuilder::new("y");
    let a = b1.input("A", 1);
    let _ = b2.input("B", 1);
    b2.not(a);
}

#[test]
#[should_panic(expected = "conflicting terminations")]
fn conflicting_termination_panics() {
    let mut b = DefBuilder::new("x");
    let w1 = b.wire(1);
    let w2 = b.wire(1);
    b.terminate(w1, Termination::PullUp);
    b.terminate(w2, Termination::PullDown);
    b.connect(w1, w2);
}

#[test]
#[should_panic(expected = "out of range")]
fn slice_out_of_range_panics() {
    let mut b = DefBuilder::new("x");
    let a = b.input("A", 4);
    b.slice(a, 2, 3);
}

#[test]
fn design_builder_merge_and_get() {
    let mut db = DesignBuilder::new();
    let fa = db.add(full_adder());
    assert!(db.contains("fa"));
    assert_eq!(db.get("fa"), Some(fa.clone()));
    assert_eq!(db.get("nope"), None);
    let other = adder4();
    db.merge(other.clone()); // identical "fa" is fine
    assert!(db.contains("add4"));
    let d = db.build(&DefRef::of(other.def("add4").unwrap()));
    assert_eq!(d.defs.len(), 2);
    assert_eq!(d.top, "add4");
    assert_eq!(d.format, FORMAT);
}

#[test]
#[should_panic(expected = "conflicting definitions")]
fn design_builder_merge_conflict_panics() {
    let mut db = DesignBuilder::new();
    db.add(full_adder());
    let mut other = adder4();
    other.defs.iter_mut().find(|d| d.id == "fa").unwrap().name = "changed".into();
    db.merge(other);
}

#[test]
#[should_panic(expected = "already exists")]
fn design_builder_add_duplicate_panics() {
    let mut db = DesignBuilder::new();
    db.add(full_adder());
    db.add(full_adder());
}

#[test]
fn splitter_atoms() {
    let mut b = DefBuilder::new("x");
    let a = b.input("A", 8);
    let s = b.slice(a, 2, 3);
    let bits = b.bits(s);
    let c = b.concat(&[bits[2], bits[1], bits[0], a]);
    b.output("O", c);
    let d = b.finish();
    let parts: Vec<_> = d
        .instances
        .iter()
        .map(|i| match &i.kind {
            InstKind::Atom(Atom::Splitter { width, parts }) => (*width, parts.clone()),
            k => panic!("unexpected {k:?}"),
        })
        .collect();
    assert_eq!(parts[0], (8, vec![(2, 4)]));
    assert_eq!(parts[1], (3, vec![(0, 0), (1, 1), (2, 2)]));
    assert_eq!(parts[2], (11, vec![(0, 0), (1, 1), (2, 2), (3, 10)]));
    assert_eq!(d.pins[1].width, 11);
}
