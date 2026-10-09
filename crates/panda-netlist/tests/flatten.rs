//! Flattening tests.

use panda_netlist::flat::*;
use panda_netlist::*;

fn design_of(top: ComponentDef, subs: Vec<ComponentDef>) -> Design {
    let mut db = DesignBuilder::new();
    for s in subs {
        db.add(s);
    }
    let t = db.add(top);
    db.build(&t)
}

fn flat(d: &Design) -> FlatNetlist {
    let f = flatten(d).unwrap_or_else(|e| panic!("flatten failed: {e:#?}"));
    check_invariants(&f);
    f
}

/// Structural invariants every flat netlist must satisfy.
fn check_invariants(f: &FlatNetlist) {
    let n = f.num_nets as usize;
    assert_eq!(f.net_names.len(), n);
    let mut drivers = vec![0u32; n];
    for g in &f.gates {
        assert!(g.output < f.num_nets);
        drivers[g.output as usize] += 1;
        assert!(!g.inputs.is_empty());
        for &i in &g.inputs {
            assert!(i < f.num_nets);
        }
        let grp = &f.groups[g.group as usize];
        assert_eq!(grp.gates[g.bit as usize] as usize, f.gates.iter().position(|x| std::ptr::eq(x, g)).unwrap());
        assert_eq!(grp.owner, g.owner);
    }
    for s in &f.sources {
        drivers[s.output as usize] += 1;
    }
    for r in &f.resolvers {
        drivers[r.output as usize] += 1;
        for &d in &r.drivers {
            assert_eq!(drivers[d as usize], 1, "resolver input must be driven by exactly one driver");
        }
    }
    assert!(drivers.iter().all(|&c| c <= 1), "at most one driver per flat net");
    for (name, &net) in &f.name_index {
        assert!(net < f.num_nets, "{name}");
    }
    for (i, name) in f.net_names.iter().enumerate() {
        assert_eq!(f.name_index.get(name), Some(&(i as u32)), "canonical name {name} must index its net");
    }
    for (&net, t) in &f.undriven_termination {
        assert_eq!(drivers[net as usize], 0);
        assert_ne!(*t, Termination::None);
    }
    for (i, inst) in f.instances.iter().enumerate() {
        for &c in &inst.children {
            assert_eq!(f.instances[c as usize].parent, Some(i as u32));
        }
    }
}

fn net(f: &FlatNetlist, name: &str) -> NetIdx {
    f.net_by_name(name).unwrap_or_else(|| {
        let mut names: Vec<_> = f.name_index.keys().collect();
        names.sort();
        panic!("no net named {name:?}; names: {names:?}")
    })
}

#[test]
fn nand_sr_latch() {
    let mut b = DefBuilder::new("sr");
    let s_n = b.input("S_n", 1);
    let r_n = b.input("R_n", 1);
    let qn = b.wire(1);
    let q = b.nand(&[s_n, qn]);
    b.nand_to(&[r_n, q], qn);
    b.output("Q", q);
    b.output("Q_n", qn);
    let f = flat(&design_of(b.finish(), vec![]));
    assert_eq!(f.num_nets, 4);
    assert_eq!(f.gates.len(), 2);
    assert!(f.resolvers.is_empty());
    let (sn, rn, q, qn) = (net(&f, "S_n"), net(&f, "R_n"), net(&f, "Q"), net(&f, "Q_n"));
    assert_eq!(f.gates[0].op, FlatOp::Nand);
    assert_eq!(f.gates[0].inputs, vec![sn, qn]);
    assert_eq!(f.gates[0].output, q);
    assert_eq!(f.gates[1].inputs, vec![rn, q]);
    assert_eq!(f.gates[1].output, qn);
    assert_eq!(f.sources.len(), 2);
    assert_eq!(f.sources[0].kind, SourceKind::TopInput { pin: 0, bit: 0 });
    assert_eq!(f.sources[0].output, sn);
    assert_eq!(f.sources[1].kind, SourceKind::TopInput { pin: 1, bit: 0 });
    assert_eq!(f.top_inputs.len(), 2);
    assert_eq!(f.top_outputs.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(), ["Q", "Q_n"]);
    assert_eq!(f.top_outputs[0].nets, vec![q]);
    assert_eq!(f.net_names[q as usize], "Q");
    assert_eq!(f.instances.len(), 1);
    assert_eq!(f.instances[0].path, "");
    assert_eq!(f.groups.len(), 2);
    assert_eq!(f.groups[0].path, "g0");
    let drv = f.drivers();
    assert_eq!(drv[q as usize], Driver::Gate(0));
    assert_eq!(drv[sn as usize], Driver::Source(0));
}

#[test]
fn bus_with_splitters_and_concat() {
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 8);
    let bits = b.bits(a);
    let rev: Vec<Sig> = bits.iter().rev().copied().collect();
    let r = b.concat(&rev);
    b.output("R", r);
    let mid = b.slice(a, 2, 4); // A[2..6]
    b.name(mid, "mid");
    let nm = b.not(mid);
    let hi = b.slice(a, 6, 2);
    let lo = b.slice(a, 0, 2);
    let j = b.concat(&[lo, nm, hi]);
    b.output("J", j);
    let f = flat(&design_of(b.finish(), vec![]));
    // No gates for wiring: only the 4-bit NOT.
    assert_eq!(f.gates.len(), 4);
    assert_eq!(f.groups.len(), 1);
    for i in 0..8 {
        assert_eq!(net(&f, &format!("R[{i}]")), net(&f, &format!("A[{}]", 7 - i)));
        assert_eq!(f.net_names[net(&f, &format!("A[{i}]")) as usize], format!("A[{i}]"));
    }
    for i in 0..4 {
        assert_eq!(net(&f, &format!("mid[{i}]")), net(&f, &format!("A[{}]", i + 2)));
        let g = &f.gates[i];
        assert_eq!(g.op, FlatOp::Not);
        assert_eq!(g.bit, i as u32);
        assert_eq!(g.inputs, vec![net(&f, &format!("A[{}]", i + 2))]);
        assert_eq!(g.output, net(&f, &format!("J[{}]", i + 2)));
    }
    assert_eq!(net(&f, "J[0]"), net(&f, "A[0]"));
    assert_eq!(net(&f, "J[7]"), net(&f, "A[7]"));
    // 8 input bits + 4 NOT outputs.
    assert_eq!(f.num_nets, 12);
    assert_eq!(f.top_inputs[0].nets.len(), 8);
    assert_eq!(f.sources.len(), 8);
}

#[test]
fn overlapping_splitter_parts_alias_through_whole() {
    // Splitter with an unconnected whole and overlapping parts: p0 = bits 0..=2, p1 = bits 1..=3.
    let mut b = DefBuilder::new("top");
    let x = b.input("X", 3);
    b.output("Y", x);
    let mut def = b.finish();
    def.pins.push(PinDecl { name: "Z".into(), dir: Dir::Out, width: 3 });
    def.instances.push(Instance {
        id: "s".into(),
        kind: InstKind::Atom(Atom::Splitter { width: 4, parts: vec![(0, 2), (1, 3)] }),
    });
    def.nets[0].ends.push(Endpoint::Port { inst: "s".into(), port: "p0".into() });
    def.nets.push(Net {
        id: "z".into(),
        width: 3,
        termination: Termination::None,
        ends: vec![Endpoint::Pin("Z".into()), Endpoint::Port { inst: "s".into(), port: "p1".into() }],
    });
    let f = flat(&design_of(def, vec![]));
    // whole[1] = X[1] = Z[0]; whole[2] = X[2] = Z[1]
    assert_eq!(net(&f, "Z[0]"), net(&f, "X[1]"));
    assert_eq!(net(&f, "Z[1]"), net(&f, "X[2]"));
    assert_eq!(net(&f, "s.whole[1]"), net(&f, "X[1]"));
    assert_ne!(net(&f, "Z[2]"), net(&f, "X[0]"));
    assert_eq!(f.net_names[net(&f, "Z[2]") as usize], "Z[2]");
    assert_eq!(net(&f, "s.whole[3]"), net(&f, "Z[2]"));
}

fn inv() -> ComponentDef {
    let mut b = DefBuilder::new("inv");
    let a = b.input("a", 1);
    let y = b.not(a);
    b.output("y", y);
    b.finish()
}

fn pair(inv: &DefRef) -> ComponentDef {
    let mut b = DefBuilder::new("pair");
    let x = b.input("x", 1);
    let m = b.inst_named("i0", inv, &[x])[0];
    b.name(m, "mid");
    let z = b.inst_named("i1", inv, &[m])[0];
    b.output("z", z);
    b.finish()
}

fn quad(pair: &DefRef) -> ComponentDef {
    let mut b = DefBuilder::new("quad");
    let x = b.input("x", 1);
    let l = b.inst_named("p0", pair, &[x])[0];
    b.name(l, "link");
    let z = b.inst_named("p1", pair, &[l])[0];
    b.output("z", z);
    b.finish()
}

#[test]
fn nested_three_levels() {
    let mut db = DesignBuilder::new();
    let i = db.add(inv());
    let p = db.add(pair(&i));
    let q = db.add(quad(&p));
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 1);
    let y = b.inst_named("q", &q, &[a])[0];
    b.output("Y", y);
    let t = db.add(b.finish());
    let f = flat(&db.build(&t));

    assert_eq!(f.gates.len(), 4);
    // Aliases of the top input at every level.
    let a = net(&f, "A");
    for alias in ["q/x", "q/p0/x", "q/p0/i0/a"] {
        assert_eq!(net(&f, alias), a, "{alias}");
    }
    assert_eq!(f.net_names[a as usize], "A");
    // The inner "mid" net of p0.
    let mid = net(&f, "q/p0/mid");
    assert_eq!(net(&f, "q/p0/i0/y"), mid);
    assert_eq!(net(&f, "q/p0/i1/a"), mid);
    assert_eq!(f.net_names[mid as usize], "q/p0/mid");
    // "link" spans p0 and p1.
    let link = net(&f, "q/link");
    for alias in ["q/p0/z", "q/p1/x", "q/p0/i1/y", "q/p1/i0/a"] {
        assert_eq!(net(&f, alias), link, "{alias}");
    }
    assert_eq!(f.net_names[link as usize], "q/link");
    let y = net(&f, "Y");
    assert_eq!(net(&f, "q/z"), y);
    assert_eq!(net(&f, "q/p1/i1/y"), y);
    assert_eq!(f.net_names[y as usize], "Y");
    assert_eq!(f.num_nets, 5);

    // Instance tree (pre-order).
    let paths: Vec<_> = f.instances.iter().map(|x| x.path.as_str()).collect();
    assert_eq!(paths, ["", "q", "q/p0", "q/p0/i0", "q/p0/i1", "q/p1", "q/p1/i0", "q/p1/i1"]);
    let defs: Vec<_> = f.instances.iter().map(|x| x.def.as_str()).collect();
    assert_eq!(defs, ["top", "quad", "pair", "inv", "inv", "pair", "inv", "inv"]);
    let qi = f.instance_by_path("q").unwrap();
    let p0 = f.instance_by_path("q/p0").unwrap();
    let p1 = f.instance_by_path("q/p1").unwrap();
    let p0i1 = f.instance_by_path("q/p0/i1").unwrap();
    assert_eq!(f.instances[qi as usize].children, vec![p0, p1]);
    assert!(f.is_within(p0i1, qi));
    assert!(f.is_within(p0i1, p0));
    assert!(f.is_within(p0i1, 0));
    assert!(f.is_within(p0i1, p0i1));
    assert!(!f.is_within(p0i1, p1));
    assert!(!f.is_within(qi, p0));
    // Instance pins.
    assert_eq!(f.instances[p0 as usize].pins[0].name, "x");
    assert_eq!(f.instances[p0 as usize].pins[0].nets, vec![a]);
    assert_eq!(f.instances[p0 as usize].pins[1].nets, vec![link]);
    // Gate owners / groups.
    assert_eq!(f.gates[1].owner, p0i1);
    assert_eq!(f.groups[1].path, "q/p0/i1/g0");
    assert_eq!(f.gates[1].inputs, vec![mid]);
    assert_eq!(f.gates[1].output, link);
}

#[test]
fn canonical_name_tie_breaks() {
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 1);
    let foo = b.wire(1);
    b.name(foo, "foo");
    // pin A and net foo share a flat net via a full-width splitter: pin wins.
    let s = b.slice(a, 0, 1);
    b.connect(s, foo);
    let x = b.not(a); // auto id
    let named = b.slice(x, 0, 1);
    b.name(named, "nice");
    b.output("O", named);
    let nb = b.not(x);
    let auto2 = b.slice(nb, 0, 1);
    let _ = b.not(auto2);
    let f = flat(&design_of(b.finish(), vec![]));
    let an = net(&f, "A");
    assert_eq!(net(&f, "foo"), an);
    assert_eq!(f.net_names[an as usize], "A");
    // Pin "O" beats explicit "nice" which beats auto "n0".
    let o = net(&f, "O");
    assert_eq!(net(&f, "n0"), o);
    assert_eq!(net(&f, "nice"), o);
    assert_eq!(f.net_names[o as usize], "O");
    // explicit vs auto: n1 (not output) vs n2 (slice) — both auto: first seen wins.
    let n1 = net(&f, "n1");
    assert_eq!(net(&f, "n2"), n1);
    assert_eq!(f.net_names[n1 as usize], "n1");
}

#[test]
fn named_net_beats_auto() {
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 1);
    let x = b.not(a); // n0
    let s = b.slice(x, 0, 1);
    b.name(s, "zed");
    let _ = b.not(s);
    let f = flat(&design_of(b.finish(), vec![]));
    let z = net(&f, "zed");
    assert_eq!(net(&f, "n0"), z);
    assert_eq!(f.net_names[z as usize], "zed");
}

fn tri_driver() -> ComponentDef {
    let mut b = DefBuilder::new("drv");
    let d = b.input("d", 1);
    let en = b.input("en", 1);
    let bus = b.inout("bus", 1);
    b.tri_to(d, en, bus);
    b.finish()
}

fn wrap(drv: &DefRef) -> ComponentDef {
    let mut b = DefBuilder::new("wrap");
    let d = b.input("d", 1);
    let en = b.input("en", 1);
    let bus = b.inout("bus", 1);
    b.inst_named("u0", drv, &[d, en, bus]);
    b.finish()
}

#[test]
fn inout_across_two_levels() {
    let mut db = DesignBuilder::new();
    let drv = db.add(tri_driver());
    let w = db.add(wrap(&drv));
    let mut b = DefBuilder::new("top");
    let d = b.input("D", 1);
    let e = b.input("E", 1);
    let bus = b.inout("BUS", 1);
    b.inst_named("w0", &w, &[d, e, bus]);
    let r = b.not(bus);
    b.output("R", r);
    let t = db.add(b.finish());
    let f = flat(&db.build(&t));
    let busn = net(&f, "BUS");
    assert_eq!(net(&f, "w0/bus"), busn);
    assert_eq!(net(&f, "w0/u0/bus"), busn);
    assert_eq!(f.net_names[busn as usize], "BUS");
    assert!(f.resolvers.is_empty());
    let tri = f.gates.iter().find(|g| g.op == FlatOp::Tri).unwrap();
    assert_eq!(tri.output, busn);
    assert_eq!(tri.inputs, vec![net(&f, "D"), net(&f, "E")]);
    assert_eq!(tri.owner, f.instance_by_path("w0/u0").unwrap());
    // Top InOut pins are outputs, not sources.
    assert_eq!(f.top_outputs.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(), ["BUS", "R"]);
    assert_eq!(f.sources.len(), 2);
    let not = f.gates.iter().find(|g| g.op == FlatOp::Not).unwrap();
    assert_eq!(not.inputs, vec![busn]);
}

#[test]
fn tri_state_bus_gets_resolver() {
    let mut db = DesignBuilder::new();
    let drv = db.add(tri_driver());
    let w = db.add(wrap(&drv));
    let mut b = DefBuilder::new("top");
    let d0 = b.input("D0", 1);
    let d1 = b.input("D1", 1);
    let e0 = b.input("E0", 1);
    let e1 = b.input("E1", 1);
    let bus = b.wire(1);
    b.name(bus, "bus");
    b.inst_named("w0", &w, &[d0, e0, bus]);
    b.inst_named("w1", &w, &[d1, e1, bus]);
    b.output("OUT", bus);
    let t = db.add(b.finish());
    let design = db.build(&t);
    assert!(validate::validate(&design).is_empty(), "{:#?}", validate::validate(&design));
    let f = flat(&design);
    let busn = net(&f, "OUT");
    assert_eq!(net(&f, "bus"), busn);
    assert_eq!(f.net_names[busn as usize], "OUT");
    assert_eq!(f.resolvers.len(), 1);
    let r = &f.resolvers[0];
    assert_eq!(r.output, busn);
    assert_eq!(r.termination, Termination::None);
    assert_eq!(r.drivers.len(), 2);
    let tris: Vec<&FlatGate> = f.gates.iter().filter(|g| g.op == FlatOp::Tri).collect();
    assert_eq!(tris.len(), 2);
    for (k, g) in tris.iter().enumerate() {
        assert_eq!(g.output, r.drivers[k]);
        assert_ne!(g.output, busn);
        assert_eq!(f.net_names[g.output as usize], format!("OUT~{k}"));
        assert_eq!(net(&f, &format!("OUT~{k}")), g.output);
    }
    assert_eq!(f.drivers()[busn as usize], Driver::Resolver(0));
    let fo = f.fanout();
    assert_eq!(fo.resolvers[r.drivers[0] as usize], vec![0]);
    assert_eq!(f.num_nets, 4 + 1 + 2);
}

#[test]
fn pulled_up_tri_bus_and_single_tri() {
    let mut b = DefBuilder::new("top");
    let d = b.input("D", 2);
    let e = b.input("E", 1);
    let bus = b.tri(d, e);
    b.terminate(bus, Termination::PullUp);
    b.name(bus, "bus");
    b.output("O", bus);
    let f = flat(&design_of(b.finish(), vec![]));
    // A single Z-capable driver on a terminated net still needs resolution.
    assert_eq!(f.resolvers.len(), 2);
    for (i, r) in f.resolvers.iter().enumerate() {
        assert_eq!(r.termination, Termination::PullUp);
        assert_eq!(r.drivers.len(), 1);
        assert_eq!(r.output, net(&f, &format!("O[{i}]")));
    }
    assert!(f.undriven_termination.is_empty());
}

#[test]
fn single_gate_on_terminated_net_needs_no_resolver() {
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 1);
    let n = b.not(a);
    b.terminate(n, Termination::PullDown);
    b.output("O", n);
    let f = flat(&design_of(b.finish(), vec![]));
    assert!(f.resolvers.is_empty());
}

#[test]
fn pull_up_undriven_net() {
    let mut b = DefBuilder::new("top");
    let w = b.wire(1);
    b.name(w, "w");
    b.terminate(w, Termination::PullUp);
    let k = b.wire(2);
    b.name(k, "k");
    b.terminate(k, Termination::Keeper);
    let r = b.not(w);
    let rk = b.not(k);
    b.output("O", r);
    b.output("OK", rk);
    let f = flat(&design_of(b.finish(), vec![]));
    let wn = net(&f, "w");
    assert_eq!(f.undriven_termination.get(&wn), Some(&Termination::PullUp));
    assert_eq!(f.undriven_termination.get(&net(&f, "k[1]")), Some(&Termination::Keeper));
    assert_eq!(f.undriven_termination.len(), 3);
    assert!(f.resolvers.is_empty());
    assert_eq!(f.drivers()[wn as usize], Driver::None);
    assert_eq!(f.gates[0].inputs, vec![wn]);
}

#[test]
fn termination_from_child_and_conflicts() {
    // Child pulls up its inout pin net; parent leaves it unterminated → merged pull-up.
    let child = {
        let mut b = DefBuilder::new("child");
        let io = b.inout("io", 1);
        b.terminate(io, Termination::PullUp);
        let y = b.not(io);
        b.output("y", y);
        b.finish()
    };
    let mut db = DesignBuilder::new();
    let c = db.add(child);
    let mut b = DefBuilder::new("top");
    let w = b.wire(1);
    b.name(w, "w");
    let y = b.inst_named("c", &c, &[w])[0];
    b.output("Y", y);
    let t = db.add(b.finish());
    let design = db.build(&t);
    let f = flat(&design);
    assert_eq!(f.undriven_termination.get(&net(&f, "w")), Some(&Termination::PullUp));
    assert_eq!(f.net_names[net(&f, "c/io") as usize], "w");

    // Same termination in parent: fine.
    let mut d2 = design.clone();
    let top = d2.defs.iter_mut().find(|d| d.id == "top").unwrap();
    top.nets.iter_mut().find(|n| n.id == "w").unwrap().termination = Termination::PullUp;
    assert!(flatten(&d2).is_ok());

    // Conflicting termination in parent: error.
    let mut d3 = design.clone();
    let top = d3.defs.iter_mut().find(|d| d.id == "top").unwrap();
    top.nets.iter_mut().find(|n| n.id == "w").unwrap().termination = Termination::PullDown;
    let errs = flatten(&d3).unwrap_err();
    assert_eq!(errs.len(), 1, "{errs:#?}");
    assert!(errs[0].message.contains("conflicting terminations"), "{}", errs[0].message);
    assert!(errs[0].location.as_deref().unwrap().contains("net w"));
}

#[test]
fn gate_groups_8bit_and() {
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 8);
    let bb = b.input("B", 8);
    let y = b.and(&[a, bb]);
    b.output("Y", y);
    let c = b.input("C", 1);
    let n = b.not(c);
    b.output("N", n);
    let f = flat(&design_of(b.finish(), vec![]));
    assert_eq!(f.gates.len(), 9);
    assert_eq!(f.groups.len(), 2);
    let g = &f.groups[0];
    assert_eq!(g.path, "g0");
    assert_eq!(g.owner, 0);
    assert_eq!(g.gates, (0..8).collect::<Vec<_>>());
    for i in 0..8u32 {
        let gate = &f.gates[i as usize];
        assert_eq!(gate.op, FlatOp::And);
        assert_eq!(gate.group, 0);
        assert_eq!(gate.bit, i);
        assert_eq!(gate.inputs, vec![net(&f, &format!("A[{i}]")), net(&f, &format!("B[{i}]"))]);
        assert_eq!(gate.output, net(&f, &format!("Y[{i}]")));
    }
    assert_eq!(f.groups[1].gates, vec![8]);
    assert_eq!(f.gates[8].group, 1);
    assert_eq!(f.gates[8].bit, 0);
    assert_eq!(f.top_inputs.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(), ["A", "B", "C"]);
    assert_eq!(f.sources[8].kind, SourceKind::TopInput { pin: 1, bit: 0 });
    assert_eq!(f.sources[16].kind, SourceKind::TopInput { pin: 2, bit: 0 });
}

#[test]
fn reduce_gate() {
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 8);
    let p = b.reduce(GateOp::Xor, a);
    b.output("P", p);
    let f = flat(&design_of(b.finish(), vec![]));
    assert_eq!(f.gates.len(), 1);
    let g = &f.gates[0];
    assert_eq!(g.op, FlatOp::Xor);
    assert_eq!(g.inputs, (0..8).map(|i| net(&f, &format!("A[{i}]"))).collect::<Vec<_>>());
    assert_eq!(g.output, net(&f, "P"));
    assert_eq!(f.groups.len(), 1);
    assert_eq!(f.groups[0].gates, vec![0]);
}

#[test]
fn sources_constants_clock_reset() {
    let mut b = DefBuilder::new("top");
    let c = b.constant(Value::parse(4, "0b1z0x").unwrap());
    let k = b.clock(8, 3, 1);
    let r = b.power_on_reset(4);
    b.output("C", c);
    b.output("K", k);
    b.output("R", r);
    let f = flat(&design_of(b.finish(), vec![]));
    let kinds: Vec<_> = f.sources.iter().map(|s| s.kind.clone()).collect();
    assert_eq!(
        kinds,
        vec![
            SourceKind::Constant(Logic::X),
            SourceKind::Constant(Logic::Zero),
            SourceKind::Constant(Logic::Z),
            SourceKind::Constant(Logic::One),
            SourceKind::Clock { period: 8, high: 3, phase: 1 },
            SourceKind::PowerOnReset { ticks: 4 },
        ]
    );
    assert_eq!(f.sources[2].output, net(&f, "C[2]"));
    assert_eq!(f.sources[4].output, net(&f, "K"));
    assert!(f.gates.is_empty());
    assert!(f.top_inputs.is_empty());
}

#[test]
fn buf_is_a_gate() {
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 2);
    let y = b.buf(a);
    b.output("Y", y);
    let f = flat(&design_of(b.finish(), vec![]));
    assert_eq!(f.gates.len(), 2);
    assert!(f.gates.iter().all(|g| g.op == FlatOp::Buf));
    assert_ne!(net(&f, "Y[0]"), net(&f, "A[0]"));
}

#[test]
fn non_tri_multiple_drivers_resolved() {
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 1);
    let y = b.not(a);
    b.buf_to(a, y);
    b.output("Y", y);
    let d = design_of(b.finish(), vec![]);
    let f = flat(&d);
    assert_eq!(f.resolvers.len(), 1);
    assert_eq!(f.resolvers[0].drivers.len(), 2);
    assert_eq!(f.resolvers[0].output, net(&f, "Y"));
}

#[test]
fn unconnected_ports_and_pins_get_nets() {
    // AND gate with in1 unconnected; a sub-def with an unconnected input pin.
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 1);
    b.output("Y", a);
    let mut def = b.finish();
    def.instances.push(Instance {
        id: "g".into(),
        kind: InstKind::Atom(Atom::Gate { op: GateOp::And, inputs: 2, width: 2, reduce: false }),
    });
    def.instances.push(Instance { id: "u".into(), kind: InstKind::Def("inv".into()) });
    let f = flat(&design_of(def, vec![inv()]));
    assert_eq!(f.gates.len(), 3);
    let g0 = &f.gates[0];
    assert_eq!(g0.inputs[0], net(&f, "g.in0[0]"));
    assert_eq!(g0.inputs[1], net(&f, "g.in1[0]"));
    assert_eq!(g0.output, net(&f, "g.out[0]"));
    assert_eq!(f.net_names[g0.inputs[1] as usize], "g.in1[0]");
    let u = f.instance_by_path("u").unwrap();
    let ua = f.instances[u as usize].pins[0].nets[0];
    assert_eq!(f.net_names[ua as usize], "u/a");
    assert_eq!(f.drivers()[ua as usize], Driver::None);
}

#[test]
fn json_design_flattens() {
    let json = r#"{
      "format": "panda/1",
      "defs": [{
        "id": "top", "pins": [{"name":"A","dir":"in","width":2},{"name":"Y","dir":"out","width":1}],
        "instances": [{"id":"g1","kind":{"atom":{"atom":"gate","op":"nand","inputs":1,"width":2,"reduce":true}}}],
        "nets": [
          {"id":"a","width":2,"ends":[["pin","A"],["g1","in0"]]},
          {"id":"y","width":1,"termination":"pull-up","ends":[["pin","Y"],["g1","out"]]}
        ]
      }],
      "top": "top"
    }"#;
    let d = Design::from_json(json).unwrap();
    let f = flat(&d);
    assert_eq!(f.gates.len(), 1);
    assert_eq!(f.gates[0].op, FlatOp::Nand);
    assert_eq!(f.gates[0].inputs.len(), 2);
    assert_eq!(net(&f, "a[1]"), net(&f, "A[1]"));
    assert_eq!(net(&f, "y"), net(&f, "Y"));
    assert_eq!(f.net_names[net(&f, "y") as usize], "Y");
}

#[test]
fn shared_subdef_instances_are_independent() {
    let mut db = DesignBuilder::new();
    let i = db.add(inv());
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 4);
    let bits = b.bits(a);
    let outs: Vec<Sig> = bits.iter().map(|&x| b.inst(&i, &[x])[0]).collect();
    let y = b.concat(&outs);
    b.output("Y", y);
    let t = db.add(b.finish());
    let f = flat(&db.build(&t));
    assert_eq!(f.gates.len(), 4);
    assert_eq!(f.instances.len(), 5);
    for k in 0..4 {
        let g = &f.gates[k];
        assert_eq!(g.inputs, vec![net(&f, &format!("A[{k}]"))]);
        assert_eq!(g.output, net(&f, &format!("Y[{k}]")));
        assert_eq!(net(&f, &format!("u{k}/y")), g.output);
        assert_eq!(f.groups[k].path, format!("u{k}/g0"));
    }
}

#[test]
fn deep_hierarchy_no_recursion() {
    const DEPTH: usize = 3000;
    let mut db = DesignBuilder::new();
    let mut prev = db.add(inv());
    for k in 0..DEPTH {
        let mut b = DefBuilder::new(&format!("lvl{k}"));
        let a = b.input("a", 1);
        let y = b.inst_named("c", &prev, &[a])[0];
        b.output("y", y);
        prev = db.add(b.finish());
    }
    let d = db.build(&prev);
    let f = flatten(&d).unwrap();
    assert_eq!(f.gates.len(), 1);
    assert_eq!(f.instances.len(), DEPTH + 1);
    assert_eq!(f.net_names[f.gates[0].inputs[0] as usize], "a");
    let deepest = format!("{}/a", vec!["c"; DEPTH].join("/"));
    assert_eq!(f.net_by_name(&deepest), Some(f.gates[0].inputs[0]));
}

#[test]
fn fanout_dedups_repeated_inputs() {
    let mut b = DefBuilder::new("top");
    let a = b.input("A", 1);
    let x = b.and(&[a, a, a]);
    let y = b.or(&[a, x]);
    b.output("Y", y);
    let f = flat(&design_of(b.finish(), vec![]));
    let fo = f.fanout();
    assert_eq!(fo.gates[net(&f, "A") as usize], vec![0, 1]);
}
