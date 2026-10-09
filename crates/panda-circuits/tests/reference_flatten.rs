//! Independent oracle for `panda_netlist::flatten`: a deliberately naive, recursive, string-keyed
//! reference flattener computes the bit-level connectivity of random hierarchical designs, and the
//! real flat netlist must match it exactly:
//! * one flat net per reference connectivity class (a bijection);
//! * every gate (per atom path and bit, incl. reduce gates and group/bit assignment) reads and
//!   drives the right classes;
//! * every class has the right drivers; multi-driver classes (and single Z-capable drivers on a
//!   terminated class) get a resolver with the merged termination, undriven terminated classes
//!   appear in `undriven_termination`; conflicting terminations are rejected;
//! * instance-tree pins, top inputs/outputs and every hierarchical alias in `name_index` point at
//!   the right class, and every canonical name follows the documented rank rule.

mod common;

use std::collections::{BTreeMap, HashMap, HashSet};

use common::random::random_design;
use panda_netlist::flat::{Driver, FlatNetlist, FlatOp, SourceKind};
use panda_netlist::{Atom, ComponentDef, Design, Dir, Endpoint, GateOp, InstKind, Logic, Termination, flatten};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum DrvKind {
    /// Atom path and bit.
    Gate(String, u32),
    Const(String),
    Clock(u32, u32, u32),
    Por(u32),
    Top(u32, u32),
}

struct RefGate {
    path: String,
    bit: u32,
    op: FlatOp,
    inputs: Vec<usize>,
    output: usize,
}

/// Rank of an alias for canonical naming: (depth, kind) with kind 0 = pin, 1 = named net,
/// 2 = auto net id, 3 = unconnected atom port.
type Rank = (u32, u8);

#[derive(Default)]
struct Ref {
    keys: HashMap<String, usize>,
    parent: Vec<usize>,
    gates: Vec<RefGate>,
    drivers: Vec<(usize, DrvKind, bool)>,
    terms: Vec<(usize, Termination)>,
    /// (key, alias name, rank)
    aliases: Vec<(usize, String, Rank)>,
    /// (instance path, pin name, key per bit)
    pins: Vec<(String, String, Dir, Vec<usize>)>,
}

fn bit_name(prefix: &str, name: &str, width: u32, j: u32) -> String {
    let base = if prefix.is_empty() { name.to_string() } else { format!("{prefix}/{name}") };
    if width == 1 { base } else { format!("{base}[{j}]") }
}

fn is_auto(s: &str) -> bool {
    s.len() > 1 && s.starts_with('n') && s[1..].bytes().all(|b| b.is_ascii_digit())
}

fn flat_op(op: GateOp) -> FlatOp {
    match op {
        GateOp::And => FlatOp::And,
        GateOp::Or => FlatOp::Or,
        GateOp::Nand => FlatOp::Nand,
        GateOp::Nor => FlatOp::Nor,
        GateOp::Xor => FlatOp::Xor,
        GateOp::Xnor => FlatOp::Xnor,
    }
}

impl Ref {
    fn key(&mut self, s: String) -> usize {
        if let Some(&k) = self.keys.get(&s) {
            return k;
        }
        let k = self.parent.len();
        self.parent.push(k);
        self.keys.insert(s, k);
        k
    }
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            x = self.parent[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            self.parent[a] = b;
        }
    }

    fn expand(&mut self, design: &Design, def: &ComponentDef, path: &str, depth: u32) {
        let p = if path.is_empty() { String::new() } else { format!("{path}/") };
        let pin_key = |r: &mut Ref, pin: &str, j: u32| r.key(format!("{p}#pin:{pin}[{j}]"));
        let child_def = |id: &str| design.def(id).expect("known def");
        // Pins of this instance.
        for pin in &def.pins {
            let ks: Vec<usize> = (0..pin.width).map(|j| pin_key(self, &pin.name, j)).collect();
            for (j, &k) in ks.iter().enumerate() {
                self.aliases.push((k, bit_name(path, &pin.name, pin.width, j as u32), (depth, 0)));
            }
            self.pins.push((path.to_string(), pin.name.clone(), pin.dir, ks));
        }
        let mut on_net: HashSet<(String, String)> = HashSet::new();
        for net in &def.nets {
            for j in 0..net.width {
                let kn = self.key(format!("{p}#net:{}[{j}]", net.id));
                let kind = if is_auto(&net.id) { 2 } else { 1 };
                self.aliases.push((kn, bit_name(path, &net.id, net.width, j), (depth, kind)));
                if net.termination != Termination::None {
                    self.terms.push((kn, net.termination));
                }
                for e in &net.ends {
                    let ke = match e {
                        Endpoint::Pin(name) => pin_key(self, name, j),
                        Endpoint::Port { inst, port } => {
                            on_net.insert((inst.clone(), port.clone()));
                            let i = def.instances.iter().find(|i| &i.id == inst).expect("instance");
                            match i.kind {
                                InstKind::Atom(_) => self.key(format!("{p}{inst}.{port}[{j}]")),
                                InstKind::Def(_) => self.key(format!("{p}{inst}/#pin:{port}[{j}]")),
                            }
                        }
                    };
                    self.union(kn, ke);
                }
            }
        }
        for inst in &def.instances {
            match &inst.kind {
                InstKind::Def(c) => {
                    let sub = if path.is_empty() { inst.id.clone() } else { format!("{path}/{}", inst.id) };
                    self.expand(design, child_def(c), &sub, depth + 1);
                }
                InstKind::Atom(atom) => {
                    let apath = format!("{p}{}", inst.id);
                    let mut port_keys: HashMap<String, Vec<usize>> = HashMap::new();
                    for pd in atom.ports() {
                        let ks: Vec<usize> =
                            (0..pd.width).map(|j| self.key(format!("{apath}.{}[{j}]", pd.name))).collect();
                        if !on_net.contains(&(inst.id.clone(), pd.name.clone())) {
                            for (j, &k) in ks.iter().enumerate() {
                                let nm = bit_name(path, &format!("{}.{}", inst.id, pd.name), pd.width, j as u32);
                                self.aliases.push((k, nm, (depth, 3)));
                            }
                        }
                        port_keys.insert(pd.name.clone(), ks);
                    }
                    let pk = |n: &str| port_keys[n].clone();
                    let gate = |r: &mut Ref, op, inputs, output, bit| {
                        r.gates.push(RefGate { path: apath.clone(), bit, op, inputs, output })
                    };
                    match atom {
                        Atom::Gate { op, inputs, width, reduce } => {
                            let ins: Vec<Vec<usize>> = (0..*inputs).map(|i| pk(&format!("in{i}"))).collect();
                            let out = pk("out");
                            if *reduce {
                                gate(self, flat_op(*op), ins[0].clone(), out[0], 0);
                            } else {
                                for j in 0..*width as usize {
                                    gate(self, flat_op(*op), ins.iter().map(|v| v[j]).collect(), out[j], j as u32);
                                }
                            }
                        }
                        Atom::Not { width } | Atom::Buf { width } => {
                            let op = if matches!(atom, Atom::Not { .. }) { FlatOp::Not } else { FlatOp::Buf };
                            let (i, o) = (pk("in"), pk("out"));
                            for j in 0..*width as usize {
                                gate(self, op, vec![i[j]], o[j], j as u32);
                            }
                        }
                        Atom::Tri { width } => {
                            let (d, en, o) = (pk("d"), pk("en"), pk("out"));
                            for j in 0..*width as usize {
                                gate(self, FlatOp::Tri, vec![d[j], en[0]], o[j], j as u32);
                            }
                        }
                        Atom::Splitter { parts, .. } => {
                            let whole = pk("whole");
                            for (k, &(lo, hi)) in parts.iter().enumerate() {
                                let part = pk(&format!("p{k}"));
                                for j in 0..=(hi - lo) {
                                    self.union(whole[(lo + j) as usize], part[j as usize]);
                                }
                            }
                        }
                        Atom::Constant { value } => {
                            for (j, (b, &k)) in value.bits().zip(&pk("out")).enumerate() {
                                let _ = j;
                                self.drivers.push((k, DrvKind::Const(format!("{b:?}")), b == Logic::Z));
                            }
                        }
                        Atom::Clock { period, high, phase } => {
                            self.drivers.push((pk("out")[0], DrvKind::Clock(*period, *high, *phase), false));
                        }
                        Atom::PowerOnReset { ticks } => {
                            self.drivers.push((pk("out")[0], DrvKind::Por(*ticks), false));
                        }
                    }
                }
            }
        }
    }
}

/// Merged termination per class, or `Err` on a conflict.
fn merged_terms(r: &mut Ref) -> Result<HashMap<usize, Termination>, ()> {
    let mut m: HashMap<usize, Termination> = HashMap::new();
    for (k, t) in r.terms.clone() {
        let c = r.find(k);
        match m.get(&c) {
            Some(&old) if old != t => return Err(()),
            _ => {
                m.insert(c, t);
            }
        }
    }
    Ok(m)
}

/// Bidirectional binding between flat nets and reference classes.
#[derive(Default)]
struct Binding {
    f2r: HashMap<u32, usize>,
    r2f: HashMap<usize, u32>,
}

impl Binding {
    fn bind(&mut self, f: u32, r: usize, what: &dyn Fn() -> String) {
        let a = *self.f2r.entry(f).or_insert(r);
        let b = *self.r2f.entry(r).or_insert(f);
        assert!(a == r && b == f, "connectivity mismatch at {}: flat net {f} vs class {r}", what());
    }
}

fn check_against_reference(design: &Design, ctx: &dyn Fn() -> String) {
    let mut r = Ref::default();
    let top = design.top_def().unwrap();
    r.expand(design, top, "", 0);
    // Top inputs drive.
    for (pi, p) in top.pins.iter().filter(|p| p.dir == Dir::In).enumerate() {
        for j in 0..p.width {
            let k = r.key(format!("#pin:{}[{j}]", p.name));
            r.drivers.push((k, DrvKind::Top(pi as u32, j), true));
        }
    }
    let flat = flatten(design);
    // A design validates without errors iff it flattens.
    let vd = panda_netlist::validate::validate(design);
    assert_eq!(
        panda_netlist::validate::has_errors(&vd),
        flat.is_err(),
        "validate/flatten disagree: {vd:#?}\n{}",
        ctx()
    );
    let terms = match merged_terms(&mut r) {
        Ok(t) => t,
        Err(()) => {
            let e = flat.err().unwrap_or_else(|| panic!("termination conflict not rejected\n{}", ctx()));
            assert!(e.iter().any(|d| d.message.contains("termination")), "{e:#?}");
            return;
        }
    };
    let f: FlatNetlist = flat.unwrap_or_else(|e| panic!("flatten failed: {e:#?}\n{}", ctx()));
    let n_keys = r.parent.len();
    let classes: HashSet<usize> = (0..n_keys).map(|k| r.find(k)).collect();
    let private: HashSet<u32> = f.resolvers.iter().flat_map(|x| x.drivers.iter().copied()).collect();
    let orig = f.num_nets as usize - private.len();
    assert_eq!(classes.len(), orig, "number of flat nets\n{}", ctx());
    for &p in &private {
        assert!(p as usize >= orig, "private resolver nets come after all real nets");
    }
    let resolved = |n: u32| -> u32 {
        if private.contains(&n) {
            f.resolvers.iter().find(|x| x.drivers.contains(&n)).unwrap().output
        } else {
            n
        }
    };

    let mut bnd = Binding::default();
    // Gates by (group path, bit).
    let mut by_path: HashMap<(String, u32), usize> = HashMap::new();
    for (gi, g) in f.gates.iter().enumerate() {
        let grp = &f.groups[g.group as usize];
        assert!(grp.gates.contains(&(gi as u32)), "group membership");
        assert_eq!(grp.gates[g.bit as usize], gi as u32, "bit = position in group");
        assert_eq!(grp.owner, g.owner);
        assert!(by_path.insert((grp.path.clone(), g.bit), gi).is_none(), "duplicate gate {}", grp.path);
    }
    assert_eq!(by_path.len(), r.gates.len(), "gate count\n{}", ctx());
    let rgates = std::mem::take(&mut r.gates);
    for rg in &rgates {
        let gi = *by_path
            .get(&(rg.path.clone(), rg.bit))
            .unwrap_or_else(|| panic!("missing gate {}[{}]\n{}", rg.path, rg.bit, ctx()));
        let g = &f.gates[gi];
        assert_eq!(g.op, rg.op, "op of {}", rg.path);
        assert_eq!(g.inputs.len(), rg.inputs.len(), "arity of {}", rg.path);
        // Owner: the instance whose path is the atom path's parent.
        let owner_path = rg.path.rsplit_once('/').map_or("", |(a, _)| a);
        assert_eq!(f.instances[g.owner as usize].path, owner_path, "owner of {}", rg.path);
        for (&fi, &ri) in g.inputs.iter().zip(&rg.inputs) {
            let c = r.find(ri);
            bnd.bind(fi, c, &|| format!("input of {}[{}]", rg.path, rg.bit));
        }
        let c = r.find(rg.output);
        bnd.bind(resolved(g.output), c, &|| format!("output of {}[{}]", rg.path, rg.bit));
    }
    // Instance pins and top pins.
    for (path, name, dir, ks) in r.pins.clone() {
        let ii = f.instance_by_path(&path).unwrap_or_else(|| panic!("instance '{path}' missing"));
        let fp = f.instances[ii as usize].pins.iter().find(|p| p.name == name).expect("pin");
        assert_eq!(fp.dir, dir);
        assert_eq!(fp.nets.len(), ks.len());
        for (&fnet, &k) in fp.nets.iter().zip(&ks) {
            let c = r.find(k);
            bnd.bind(fnet, c, &|| format!("pin {path}/{name}"));
        }
    }
    let top_in: Vec<&str> = top.pins.iter().filter(|p| p.dir == Dir::In).map(|p| p.name.as_str()).collect();
    let top_out: Vec<&str> = top.pins.iter().filter(|p| p.dir != Dir::In).map(|p| p.name.as_str()).collect();
    assert_eq!(f.top_inputs.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(), top_in);
    assert_eq!(f.top_outputs.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(), top_out);
    for p in f.top_inputs.iter().chain(&f.top_outputs) {
        assert_eq!(p.nets, f.instances[0].pins.iter().find(|q| q.name == p.name).unwrap().nets);
    }
    // Aliases: every alias is in name_index and points at its class.
    let aliases = std::mem::take(&mut r.aliases);
    let mut best: HashMap<usize, Rank> = HashMap::new();
    for (k, name, rank) in &aliases {
        let c = r.find(*k);
        let n = f.net_by_name(name).unwrap_or_else(|| panic!("alias '{name}' missing from name_index\n{}", ctx()));
        bnd.bind(n, c, &|| format!("alias {name}"));
        let b = best.entry(c).or_insert(*rank);
        *b = (*b).min(*rank);
    }
    assert_eq!(bnd.r2f.len(), classes.len(), "every class is reachable by some alias");
    // Canonical names: an alias of its own net, of minimal rank.
    for (c, &n) in &bnd.r2f {
        let cname = &f.net_names[n as usize];
        assert!(
            aliases.iter().any(|(k, nm, rank)| nm == cname && r.find(*k) == *c && *rank == best[c]),
            "canonical name '{cname}' is not a best-ranked alias of its net (best {:?})\n{}",
            best[c],
            ctx()
        );
    }
    // Private nets.
    for res in &f.resolvers {
        for (i, &d) in res.drivers.iter().enumerate() {
            assert!(f.net_names[d as usize].starts_with(&format!("{}~{i}", f.net_names[res.output as usize])));
            assert_eq!(f.net_by_name(&f.net_names[d as usize]), Some(d));
        }
    }

    // Drivers per class.
    let mut ref_drv: BTreeMap<u32, Vec<DrvKind>> = BTreeMap::new();
    let mut zcap: HashMap<u32, bool> = HashMap::new();
    for (k, kind, z) in r.drivers.clone() {
        let c = r.find(k);
        let n = bnd.r2f[&c];
        ref_drv.entry(n).or_default().push(kind);
        *zcap.entry(n).or_default() |= z;
    }
    for rg in &rgates {
        let c = r.find(rg.output);
        let n = bnd.r2f[&c];
        ref_drv.entry(n).or_default().push(DrvKind::Gate(rg.path.clone(), rg.bit));
        *zcap.entry(n).or_default() |= rg.op == FlatOp::Tri;
    }
    let mut flat_drv: BTreeMap<u32, Vec<DrvKind>> = BTreeMap::new();
    let kind_of_source = |k: &SourceKind| match k {
        SourceKind::Constant(l) => DrvKind::Const(format!("{l:?}")),
        SourceKind::Clock { period, high, phase } => DrvKind::Clock(*period, *high, *phase),
        SourceKind::PowerOnReset { ticks } => DrvKind::Por(*ticks),
        SourceKind::TopInput { pin, bit } => DrvKind::Top(*pin, *bit),
    };
    for g in &f.gates {
        let grp = &f.groups[g.group as usize];
        flat_drv.entry(resolved(g.output)).or_default().push(DrvKind::Gate(grp.path.clone(), g.bit));
    }
    for s in &f.sources {
        flat_drv.entry(resolved(s.output)).or_default().push(kind_of_source(&s.kind));
    }
    for v in ref_drv.values_mut().chain(flat_drv.values_mut()) {
        v.sort();
    }
    assert_eq!(ref_drv, flat_drv, "drivers per net\n{}", ctx());
    // Top input sources are exactly on the top input pin nets.
    for s in &f.sources {
        if let SourceKind::TopInput { pin, bit } = s.kind {
            assert_eq!(resolved(s.output), f.top_inputs[pin as usize].nets[bit as usize]);
        }
    }
    // Resolvers and terminations.
    let drivers = f.drivers();
    for n in 0..orig as u32 {
        let c = *bnd.f2r.get(&n).unwrap_or_else(|| panic!("flat net {n} unbound"));
        let t = terms.get(&c).copied().unwrap_or(Termination::None);
        let cnt = ref_drv.get(&n).map_or(0, Vec::len);
        let z = zcap.get(&n).copied().unwrap_or(false);
        let needs = cnt >= 2 || (cnt == 1 && z && t != Termination::None);
        match drivers[n as usize] {
            Driver::Resolver(ri) => {
                assert!(needs, "unneeded resolver on {}", f.net_names[n as usize]);
                let res = &f.resolvers[ri as usize];
                assert_eq!(res.termination, t, "resolver termination on {}", f.net_names[n as usize]);
                assert_eq!(res.drivers.len(), cnt);
            }
            Driver::None => {
                assert_eq!(cnt, 0);
                assert_eq!(f.undriven_termination.get(&n).copied().unwrap_or(Termination::None), t);
            }
            _ => {
                assert!(!needs && cnt == 1, "net {} needs a resolver", f.net_names[n as usize]);
                assert!(!f.undriven_termination.contains_key(&n));
            }
        }
    }
    for p in &private {
        assert!(matches!(drivers[*p as usize], Driver::Gate(_) | Driver::Source(_)), "private net has one driver");
    }
    // Instance tree.
    for (i, inst) in f.instances.iter().enumerate() {
        for &c in &inst.children {
            assert_eq!(f.instances[c as usize].parent, Some(i as u32));
            let cp = &f.instances[c as usize].path;
            let expect_prefix = if inst.path.is_empty() { String::new() } else { format!("{}/", inst.path) };
            assert!(cp.starts_with(&expect_prefix) && !cp[expect_prefix.len()..].contains('/'));
        }
        if let Some(p) = inst.parent {
            assert!(f.instances[p as usize].children.contains(&(i as u32)));
        } else {
            assert_eq!(i, 0);
        }
    }
}

#[test]
fn flatten_matches_reference_on_random_designs() {
    for seed in 0..400 {
        let design = random_design(seed);
        check_against_reference(&design, &|| format!("seed {seed}\n{}", design.to_json()));
    }
}

#[test]
#[ignore = "long-running fuzz; run with --ignored"]
fn flatten_matches_reference_on_random_designs_long() {
    for seed in 10_000..20_000 {
        let design = random_design(seed);
        check_against_reference(&design, &|| format!("seed {seed}\n{}", design.to_json()));
    }
}

#[test]
fn flatten_matches_reference_on_library_blocks() {
    use panda_circuits::*;
    use panda_netlist::DesignBuilder;
    let mut db = DesignBuilder::new();
    let tops = vec![
        register_file(&mut db, 5, 3, RegFileStyle::EnableMux),
        register_file(&mut db, 4, 2, RegFileStyle::GatedClock),
        sync_counter(&mut db, 4),
        ripple_counter(&mut db, 3),
        rom(&mut db, "t", 3, 64, &[1, 2, u64::MAX]),
        transceiver(&mut db, 3),
        bus_port(&mut db, 4),
        dff_en(&mut db),
        dff_arst(&mut db),
        toggle_ff(&mut db),
        ring_oscillator(&mut db, 5, true),
        equality(&mut db, 5),
        decoder(&mut db, 3),
        mux(&mut db, 2, 3),
    ];
    let cnt = sync_counter(&mut db, 3);
    let tb = testbench(&mut db, &cnt, 8);
    let all = db.build(&tb);
    for t in tops.iter().chain([&tb]) {
        // Keep only the defs reachable from `t` (the testbench def holds sources).
        let mut d = all.clone();
        d.top = t.id.clone();
        if t.id != tb.id {
            d.defs.retain(|x| x.id != tb.id);
        }
        check_against_reference(&d, &|| format!("top {}", t.id));
    }
}
