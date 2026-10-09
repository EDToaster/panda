//! Implementation of [`crate::flat::flatten`].
//!
//! Algorithm (all passes linear in the size of the expanded design, no recursion):
//! 1. validate; any error aborts;
//! 2. build a per-def [`Template`] (local bit layout of nets, pins and atom ports);
//! 3. expand the instance tree depth-first with an explicit stack, giving every instance a range
//!    of global bits; union child pin bits with the parent's port bits and apply splitter aliasing
//!    (union-find whose roots are class minima, so one forward pass numbers the flat nets);
//! 4. merge terminations per flat net (conflicts are errors);
//! 5. emit gates (one [`GateGroup`] per gate-like atom instance), sources and top inputs;
//! 6. give every flat net at most one driver: nets with ≥2 drivers — or with one Z-capable driver
//!    (TRI, Constant `Z` bit, top input) and a termination — get a [`Resolver`] whose inputs are
//!    fresh private nets, one per original driver;
//! 7. names: every alias goes into `name_index`; the canonical name is the alias with the fewest
//!    `/` (highest level), ties broken by pin names, then explicitly named nets, then auto ids
//!    (`n<digits>`), then unconnected atom ports (`"<path>/<inst>.<port>[b]"`), then first seen.
//!    Private resolver input nets are named `"<canonical>~<k>"` (k = driver position; further `~`
//!    are appended in the unlikely case that this collides with an existing alias).

use std::collections::HashMap;
use std::fmt::Write as _;

use crate::flat::*;
use crate::model::{Atom, Design, Dir, GateOp, InstKind, Termination};
use crate::structure::{DesignIndex, Template, UnionFind};
use crate::validate::{has_errors, validate};
use crate::value::Logic;
use crate::{DiagKind, Diagnostic, Severity};

/// One expanded def instance (during flattening).
struct Exp {
    def: usize,
    base: u32,
    depth: u32,
}

/// Rank of an alias for canonical-name selection (lower is better).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Rank {
    depth: u32,
    kind: u8,
}

/// What a canonical alias refers to.
#[derive(Clone, Copy)]
enum Alias {
    Pin { inst: u32, pin: u32, bit: u32 },
    Net { inst: u32, net: u32, bit: u32 },
    Port { inst: u32, atom: u32, port: u32, bit: u32 },
}

const KIND_PIN: u8 = 0;
const KIND_NAMED: u8 = 1;
const KIND_AUTO: u8 = 2;
const KIND_PORT: u8 = 3;

fn is_auto_id(s: &str) -> bool {
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

/// Append `name` with an optional bit suffix (`[b]` unless the item is 1 bit wide).
fn push_name(buf: &mut String, prefix: &str, name: &str, width: u32, bit: u32) {
    buf.clear();
    if !prefix.is_empty() {
        buf.push_str(prefix);
        buf.push('/');
    }
    buf.push_str(name);
    if width != 1 {
        let _ = write!(buf, "[{bit}]");
    }
}

pub(crate) fn flatten(design: &Design) -> Result<FlatNetlist, Vec<Diagnostic>> {
    let diags = validate(design);
    if has_errors(&diags) {
        return Err(diags.into_iter().filter(|d| d.severity == Severity::Error).collect());
    }
    let ix = DesignIndex::new(design);
    let too_large = || vec![Diagnostic::error(DiagKind::Other, "design too large to flatten (> 2^32 bits)")];
    let top = ix.def_by_id[design.top.as_str()];

    // ---- 2. templates (only for reachable defs) ----
    let mut templates: Vec<Option<Template>> = (0..ix.defs.len()).map(|_| None).collect();
    {
        let mut stack = vec![top];
        while let Some(d) = stack.pop() {
            if templates[d].is_some() {
                continue;
            }
            templates[d] = Some(Template::build(&ix, d).map_err(|_| too_large())?);
            stack.extend(ix.defs[d].child_def.iter().flatten().copied().filter(|&c| templates[c].is_none()));
        }
    }
    let tpl = |d: usize| templates[d].as_ref().expect("template built for reachable def");

    // ---- 3. expand instance tree ----
    let mut exps: Vec<Exp> = Vec::new();
    let mut instances: Vec<FlatInstance> = Vec::new();
    let mut uf = UnionFind::new(0);
    let mut total: u32 = 0;
    // (def, parent instance, index of the instance within the parent def)
    let mut stack: Vec<(usize, Option<u32>, usize)> = vec![(top, None, 0)];
    while let Some((d, parent, ii)) = stack.pop() {
        let t = tpl(d);
        let idx = exps.len() as u32;
        let base = total;
        total = total.checked_add(t.bits).ok_or_else(too_large)?;
        uf.grow(t.bits);
        for &(a, b) in &t.unions {
            uf.union(base + a, base + b);
        }
        let (path, depth) = match parent {
            None => (String::new(), 0),
            Some(p) => {
                let pe = &exps[p as usize];
                let pt = tpl(pe.def);
                for (pi, pin) in ix.defs[d].def.pins.iter().enumerate() {
                    if let Some(q) = pt.inst_ports[ii][pi] {
                        let c0 = base + t.pin_off[pi];
                        let p0 = pe.base + q;
                        for j in 0..pin.width {
                            uf.union(c0 + j, p0 + j);
                        }
                    }
                }
                let id = &ix.defs[pe.def].def.instances[ii].id;
                let pp = &instances[p as usize].path;
                let path = if pp.is_empty() { id.clone() } else { format!("{pp}/{id}") };
                instances[p as usize].children.push(idx);
                (path, pe.depth + 1)
            }
        };
        exps.push(Exp { def: d, base, depth });
        instances.push(FlatInstance {
            path,
            def: ix.defs[d].def.id.clone(),
            parent,
            children: Vec::new(),
            pins: Vec::new(),
        });
        for (ci, c) in ix.defs[d].child_def.iter().enumerate().rev() {
            if let Some(c) = c {
                stack.push((*c, Some(idx), ci));
            }
        }
    }
    debug_assert_eq!(uf.len(), total as usize);

    // Number flat nets: roots are class minima, so a forward pass sees each root first.
    let mut flat_of: Vec<u32> = vec![0; total as usize];
    let mut num_nets: u32 = 0;
    for i in 0..total {
        let r = uf.find(i);
        if r == i {
            flat_of[i as usize] = num_nets;
            num_nets += 1;
        } else {
            flat_of[i as usize] = flat_of[r as usize];
        }
    }
    drop(uf);
    let fnet = |e: &Exp, off: u32| flat_of[(e.base + off) as usize];

    // Instance pins.
    for (ei, e) in exps.iter().enumerate() {
        let t = tpl(e.def);
        instances[ei].pins = ix.defs[e.def]
            .def
            .pins
            .iter()
            .enumerate()
            .map(|(pi, p)| FlatPin {
                name: p.name.clone(),
                dir: p.dir,
                nets: (0..p.width).map(|j| fnet(e, t.pin_off[pi] + j)).collect(),
            })
            .collect();
    }

    // ---- 4. terminations ----
    let mut term = vec![Termination::None; num_nets as usize];
    let mut conflicts: HashMap<u32, (Termination, Termination, u32)> = HashMap::new();
    for (ei, e) in exps.iter().enumerate() {
        let t = tpl(e.def);
        for (ni, n) in ix.defs[e.def].def.nets.iter().enumerate() {
            if n.termination == Termination::None {
                continue;
            }
            for j in 0..n.width {
                let f = fnet(e, t.net_off[ni] + j);
                let cur = term[f as usize];
                if cur == Termination::None {
                    term[f as usize] = n.termination;
                } else if cur != n.termination {
                    conflicts.entry(f).or_insert((cur, n.termination, ei as u32));
                }
            }
        }
    }

    // ---- 5. gates, groups, sources ----
    let mut gates: Vec<FlatGate> = Vec::new();
    let mut groups: Vec<GateGroup> = Vec::new();
    let mut sources: Vec<Source> = Vec::new();
    let mut top_inputs = Vec::new();
    let mut top_outputs = Vec::new();
    for p in &instances[0].pins {
        match p.dir {
            Dir::In => {
                let pin = top_inputs.len() as u32;
                for (bit, &net) in p.nets.iter().enumerate() {
                    sources.push(Source { kind: SourceKind::TopInput { pin, bit: bit as u32 }, output: net });
                }
                top_inputs.push(p.clone());
            }
            Dir::Out | Dir::InOut => top_outputs.push(p.clone()),
        }
    }
    for (ei, e) in exps.iter().enumerate() {
        let t = tpl(e.def);
        let def = ix.defs[e.def].def;
        for (ii, inst) in def.instances.iter().enumerate() {
            let InstKind::Atom(atom) = &inst.kind else { continue };
            let port = |p: usize| t.inst_ports[ii][p].expect("atom ports always have bits");
            let mut new_group = |gates: &mut Vec<FlatGate>, specs: Vec<(FlatOp, Vec<NetIdx>, NetIdx)>| {
                let group = groups.len() as GroupIdx;
                let path = &instances[ei].path;
                let path = if path.is_empty() { inst.id.clone() } else { format!("{path}/{}", inst.id) };
                let mut members = Vec::with_capacity(specs.len());
                for (bit, (op, inputs, output)) in specs.into_iter().enumerate() {
                    members.push(gates.len() as GateIdx);
                    gates.push(FlatGate { op, inputs, output, owner: ei as InstIdx, group, bit: bit as u32 });
                }
                groups.push(GateGroup { owner: ei as InstIdx, path, gates: members });
            };
            match atom {
                Atom::Gate { op, inputs, width, reduce } => {
                    let op = flat_op(*op);
                    let out = port(*inputs as usize);
                    if *reduce {
                        let i0 = port(0);
                        let ins = (0..*width).map(|j| fnet(e, i0 + j)).collect();
                        new_group(&mut gates, vec![(op, ins, fnet(e, out))]);
                    } else {
                        let in_offs: Vec<u32> = (0..*inputs as usize).map(port).collect();
                        let specs = (0..*width)
                            .map(|j| (op, in_offs.iter().map(|&o| fnet(e, o + j)).collect(), fnet(e, out + j)))
                            .collect();
                        new_group(&mut gates, specs);
                    }
                }
                Atom::Not { width } | Atom::Buf { width } => {
                    let op = if matches!(atom, Atom::Not { .. }) { FlatOp::Not } else { FlatOp::Buf };
                    let (i, o) = (port(0), port(1));
                    let specs = (0..*width).map(|j| (op, vec![fnet(e, i + j)], fnet(e, o + j))).collect();
                    new_group(&mut gates, specs);
                }
                Atom::Tri { width } => {
                    let (d, en, o) = (port(0), port(1), port(2));
                    let en = fnet(e, en);
                    let specs = (0..*width).map(|j| (FlatOp::Tri, vec![fnet(e, d + j), en], fnet(e, o + j))).collect();
                    new_group(&mut gates, specs);
                }
                Atom::Splitter { .. } => {}
                Atom::Constant { value } => {
                    let o = port(0);
                    for (j, b) in value.bits().enumerate() {
                        sources.push(Source { kind: SourceKind::Constant(b), output: fnet(e, o + j as u32) });
                    }
                }
                Atom::Clock { period, high, phase } => sources.push(Source {
                    kind: SourceKind::Clock { period: *period, high: *high, phase: *phase },
                    output: fnet(e, port(0)),
                }),
                Atom::PowerOnReset { ticks } => {
                    sources.push(Source { kind: SourceKind::PowerOnReset { ticks: *ticks }, output: fnet(e, port(0)) })
                }
            }
        }
    }

    // ---- 6. drivers & resolvers ----
    let mut count = vec![0u32; num_nets as usize];
    let mut zcap = vec![false; num_nets as usize];
    for g in &gates {
        count[g.output as usize] += 1;
        zcap[g.output as usize] |= g.op == FlatOp::Tri;
    }
    for s in &sources {
        count[s.output as usize] += 1;
        zcap[s.output as usize] |= matches!(s.kind, SourceKind::Constant(Logic::Z) | SourceKind::TopInput { .. });
    }
    let needs_resolver = |n: usize| count[n] >= 2 || (count[n] == 1 && zcap[n] && term[n] != Termination::None);
    let mut resolver_of = vec![u32::MAX; num_nets as usize];
    let mut resolvers: Vec<Resolver> = Vec::new();
    let mut private: Vec<(NetIdx, u32)> = Vec::new(); // (resolved net, driver position) per private net
    let mut reroute = |out: &mut NetIdx, num_nets: &mut u32| {
        let n = *out as usize;
        if !needs_resolver(n) {
            return;
        }
        if resolver_of[n] == u32::MAX {
            resolver_of[n] = resolvers.len() as u32;
            resolvers.push(Resolver { drivers: Vec::new(), termination: term[n], output: *out });
        }
        let r = &mut resolvers[resolver_of[n] as usize];
        let p = *num_nets;
        *num_nets += 1;
        private.push((*out, r.drivers.len() as u32));
        r.drivers.push(p);
        *out = p;
    };
    let orig_nets = num_nets;
    for g in &mut gates {
        reroute(&mut g.output, &mut num_nets);
    }
    for s in &mut sources {
        reroute(&mut s.output, &mut num_nets);
    }
    let undriven_termination: HashMap<NetIdx, Termination> = (0..orig_nets as usize)
        .filter(|&n| count[n] == 0 && term[n] != Termination::None)
        .map(|n| (n as NetIdx, term[n]))
        .collect();

    // ---- 7. names ----
    let mut best: Vec<Option<(Rank, Alias)>> = vec![None; orig_nets as usize];
    let mut name_index: HashMap<String, NetIdx> = HashMap::with_capacity(total as usize);
    let mut buf = String::new();
    let mut consider = |name: &str, net: NetIdx, rank: Rank, alias: Alias, name_index: &mut HashMap<String, NetIdx>| {
        name_index.entry(name.to_owned()).or_insert(net);
        let b = &mut best[net as usize];
        if b.is_none_or(|(r, _)| rank < r) {
            *b = Some((rank, alias));
        }
    };
    for (ei, e) in exps.iter().enumerate() {
        let t = tpl(e.def);
        let def = ix.defs[e.def].def;
        let path = &instances[ei].path;
        let inst = ei as u32;
        for (pi, p) in def.pins.iter().enumerate() {
            let rank = Rank { depth: e.depth, kind: KIND_PIN };
            for j in 0..p.width {
                push_name(&mut buf, path, &p.name, p.width, j);
                let alias = Alias::Pin { inst, pin: pi as u32, bit: j };
                consider(&buf, fnet(e, t.pin_off[pi] + j), rank, alias, &mut name_index);
            }
        }
        for (ni, n) in def.nets.iter().enumerate() {
            let kind = if is_auto_id(&n.id) { KIND_AUTO } else { KIND_NAMED };
            let rank = Rank { depth: e.depth, kind };
            for j in 0..n.width {
                push_name(&mut buf, path, &n.id, n.width, j);
                let alias = Alias::Net { inst, net: ni as u32, bit: j };
                consider(&buf, fnet(e, t.net_off[ni] + j), rank, alias, &mut name_index);
            }
        }
        for (ii, a) in def.instances.iter().enumerate() {
            let InstKind::Atom(atom) = &a.kind else { continue };
            for (pi, &virt) in t.port_virtual[ii].iter().enumerate() {
                if !virt {
                    continue;
                }
                let (_, w) = crate::structure::atom_port(atom, pi);
                let pname = &atom.ports()[pi].name;
                let rank = Rank { depth: e.depth, kind: KIND_PORT };
                let off = t.inst_ports[ii][pi].expect("atom port");
                for j in 0..w {
                    push_name(&mut buf, path, &format!("{}.{pname}", a.id), w, j);
                    let alias = Alias::Port { inst, atom: ii as u32, port: pi as u32, bit: j };
                    consider(&buf, fnet(e, off + j), rank, alias, &mut name_index);
                }
            }
        }
    }
    let mut net_names: Vec<String> = Vec::with_capacity(num_nets as usize);
    for b in &best {
        let (_, alias) = b.expect("every flat net has at least one alias");
        match alias {
            Alias::Pin { inst, pin, bit } => {
                let p = &ix.defs[exps[inst as usize].def].def.pins[pin as usize];
                push_name(&mut buf, &instances[inst as usize].path, &p.name, p.width, bit);
            }
            Alias::Net { inst, net, bit } => {
                let n = &ix.defs[exps[inst as usize].def].def.nets[net as usize];
                push_name(&mut buf, &instances[inst as usize].path, &n.id, n.width, bit);
            }
            Alias::Port { inst, atom, port, bit } => {
                let a = &ix.defs[exps[inst as usize].def].def.instances[atom as usize];
                let InstKind::Atom(at) = &a.kind else { unreachable!("port alias of an atom") };
                let (_, w) = crate::structure::atom_port(at, port as usize);
                let pname = &at.ports()[port as usize].name;
                push_name(&mut buf, &instances[inst as usize].path, &format!("{}.{pname}", a.id), w, bit);
            }
        }
        net_names.push(buf.clone());
    }
    for &(net, k) in &private {
        // Identifiers may contain `~`, so `<canonical>~<k>` can already be a real alias; append
        // further `~` until the name is free so that every private net stays addressable.
        let mut name = format!("{}~{k}", net_names[net as usize]);
        while name_index.contains_key(&name) {
            name.push('~');
        }
        name_index.insert(name.clone(), net_names.len() as NetIdx);
        net_names.push(name);
    }
    debug_assert_eq!(net_names.len(), num_nets as usize);

    if !conflicts.is_empty() {
        let mut errs: Vec<(u32, Diagnostic)> = conflicts
            .into_iter()
            .map(|(f, (a, b, ei))| {
                let d = Diagnostic::error(
                    DiagKind::Other,
                    format!("conflicting terminations {a:?} and {b:?} on net '{}'", net_names[f as usize]),
                )
                .at(format!("{}: net {}", instances[ei as usize].def, net_names[f as usize]));
                (f, d)
            })
            .collect();
        errs.sort_by_key(|(f, _)| *f);
        return Err(errs.into_iter().map(|(_, d)| d).collect());
    }

    Ok(FlatNetlist {
        num_nets,
        gates,
        sources,
        resolvers,
        groups,
        instances,
        undriven_termination,
        net_names,
        name_index,
        top_inputs,
        top_outputs,
    })
}
