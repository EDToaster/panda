//! Structural validation of a [`Design`] (SPEC §4). Returns every problem found, not just the first.
//!
//! Errors: unknown top/def/instance/port, duplicate ids (defs, instances within a def, nets within a
//! def, pins within a def), recursive def dependency, net endpoint width ≠ net width, atom parameter
//! errors (gate inputs < 1, reduce with inputs ≠ 1, width 0, splitter parts out of range, clock
//! period < 2 or high ∉ (0, period)), Clock/PowerOnReset outside the top def, a pin or port used by
//! more than one net, an `In` pin of the def driven from inside (a gate output on the same net),
//! conflicting terminations on nets joined through splitters or sub-def pins (kind `Other`; the
//! same check flattening applies, so a design that validates without errors always flattens).
//! Warnings: unused pins, nets with ≥2 drivers where some driver is not Z-capable (`MultipleDrivers`).
//!
//! Additional checks (beyond the list above):
//! * identifiers (pin names, instance ids, net ids) must be non-empty and must not contain `/`,
//!   `[` or `]` (they are used to build hierarchical net names) — error, kind `Other`;
//! * the instance id `"pin"` is reserved (it would serialize like a pin endpoint) — error,
//!   kind `DuplicateId`;
//! * pin and net widths must be ≥ 1 — error, kind `WidthMismatch`;
//! * the same endpoint listed twice in one net — warning, kind `BadPinUse`;
//! * a net whose id equals the name of a pin that is not on that net (ambiguous hierarchical
//!   name) — warning, kind `Other`.
//!
//! Driver analysis is bit-exact: it follows splitter aliasing and pin-to-pin connections through
//! sub-defs. A driver is an atom output (gate/NOT/BUF/TRI/Constant/Clock/PowerOnReset) or a sub-def
//! `Out`/`InOut` port that is driven inside the sub-def. A driver is Z-capable if it is a TRI, a
//! Constant bit equal to `Z`, or a sub-def port whose internal drivers are all Z-capable.
//!
//! Locations have the form `"<def>"`, `"<def>: instance <id>"`, `"<def>: net <id>"` or
//! `"<def>: pin <name>"`.

use std::collections::{HashMap, HashSet};

use crate::model::{Atom, ComponentDef, Design, Dir, Endpoint, InstKind, Termination};
use crate::structure::{DesignIndex, Template, UnionFind, atom_port_count, splitter_ok};
use crate::value::Logic;
use crate::{DiagKind, Diagnostic};

/// Validate a design; returns all errors and warnings found.
pub fn validate(design: &Design) -> Vec<Diagnostic> {
    let mut v = Validator { design, ix: DesignIndex::new(design), diags: Vec::new() };
    v.run();
    v.diags
}

/// True if any diagnostic is an error.
pub fn has_errors(diags: &[Diagnostic]) -> bool {
    diags.iter().any(|d| d.severity == crate::Severity::Error)
}

fn loc_inst(def: &str, inst: &str) -> String {
    format!("{def}: instance {inst}")
}
fn loc_net(def: &str, net: &str) -> String {
    format!("{def}: net {net}")
}
fn loc_pin(def: &str, pin: &str) -> String {
    format!("{def}: pin {pin}")
}

fn bad_ident(s: &str) -> bool {
    s.is_empty() || s.contains(['/', '[', ']'])
}

fn endpoint_str(e: &Endpoint) -> String {
    match e {
        Endpoint::Pin(p) => format!("pin {p}"),
        Endpoint::Port { inst, port } => format!("{inst}.{port}"),
    }
}

/// Internal drive state of each bit of each pin of a def, as seen from its parent.
struct PinDrive {
    /// Per pin, per bit: `Some(zcap)` if driven inside the def (all internal drivers Z-capable).
    drive: Vec<Vec<Option<bool>>>,
    /// Per pin, per bit: index of a representative `(pin, bit)` of its internal alias class.
    alias: Vec<Vec<(u32, u32)>>,
    /// Per pin, per bit: termination of its internal alias class (`None` if unterminated).
    term: Vec<Vec<Termination>>,
}

struct Validator<'a> {
    design: &'a Design,
    ix: DesignIndex<'a>,
    diags: Vec<Diagnostic>,
}

impl<'a> Validator<'a> {
    fn err(&mut self, kind: DiagKind, msg: String, loc: String) {
        self.diags.push(Diagnostic::error(kind, msg).at(loc));
    }
    fn warn(&mut self, kind: DiagKind, msg: String, loc: String) {
        self.diags.push(Diagnostic::warning(kind, msg).at(loc));
    }

    fn run(&mut self) {
        let design = self.design;
        let mut seen = HashSet::new();
        for d in &design.defs {
            if !seen.insert(d.id.as_str()) {
                self.err(DiagKind::DuplicateId, format!("duplicate def id '{}'", d.id), d.id.clone());
            }
        }
        if !self.ix.def_by_id.contains_key(design.top.as_str()) {
            self.err(DiagKind::UnknownDef, format!("top def '{}' does not exist", design.top), design.top.clone());
        }
        for (d, def) in design.defs.iter().enumerate() {
            // Only the first def with a given id is indexed; duplicates are still checked locally.
            self.check_def(d, def);
        }
        let order = self.check_recursion();
        self.check_drivers(&order);
    }

    fn check_def(&mut self, d: usize, def: &'a ComponentDef) {
        let id = def.id.as_str();
        let is_top = id == self.design.top;
        // Pins.
        let mut pins = HashSet::new();
        for p in &def.pins {
            if !pins.insert(p.name.as_str()) {
                self.err(DiagKind::DuplicateId, format!("duplicate pin '{}'", p.name), loc_pin(id, &p.name));
            }
            if bad_ident(&p.name) {
                self.err(DiagKind::Other, format!("invalid pin name '{}'", p.name), loc_pin(id, &p.name));
            }
            if p.width == 0 {
                self.err(DiagKind::WidthMismatch, format!("pin '{}' has width 0", p.name), loc_pin(id, &p.name));
            }
        }
        // Instances.
        let mut insts = HashSet::new();
        for inst in &def.instances {
            let loc = loc_inst(id, &inst.id);
            if !insts.insert(inst.id.as_str()) {
                self.err(DiagKind::DuplicateId, format!("duplicate instance id '{}'", inst.id), loc.clone());
            }
            if inst.id == "pin" {
                self.err(DiagKind::DuplicateId, "instance id 'pin' is reserved".into(), loc.clone());
            } else if bad_ident(&inst.id) {
                self.err(DiagKind::Other, format!("invalid instance id '{}'", inst.id), loc.clone());
            }
            match &inst.kind {
                InstKind::Def(child) => {
                    if !self.ix.def_by_id.contains_key(child.as_str()) {
                        self.err(DiagKind::UnknownDef, format!("unknown def '{child}'"), loc);
                    }
                }
                InstKind::Atom(atom) => {
                    if !is_top && matches!(atom, Atom::Clock { .. } | Atom::PowerOnReset { .. }) {
                        self.err(
                            DiagKind::SourceNotTop,
                            "Clock/PowerOnReset may only appear in the top def".into(),
                            loc.clone(),
                        );
                    }
                    for m in atom_errors(atom) {
                        self.err(DiagKind::BadAtom, m, loc.clone());
                    }
                }
            }
        }
        // Nets.
        let mut nets = HashSet::new();
        let mut pin_uses: HashMap<&str, Vec<usize>> = HashMap::new();
        let mut port_uses: HashMap<(&str, &str), Vec<usize>> = HashMap::new();
        for (ni, n) in def.nets.iter().enumerate() {
            let loc = loc_net(id, &n.id);
            if !nets.insert(n.id.as_str()) {
                self.err(DiagKind::DuplicateId, format!("duplicate net id '{}'", n.id), loc.clone());
            }
            if bad_ident(&n.id) {
                self.err(DiagKind::Other, format!("invalid net id '{}'", n.id), loc.clone());
            }
            if n.width == 0 {
                self.err(DiagKind::WidthMismatch, format!("net '{}' has width 0", n.id), loc.clone());
            }
            let mut in_net = HashSet::new();
            for e in &n.ends {
                if !in_net.insert(e) {
                    self.warn(
                        DiagKind::BadPinUse,
                        format!("endpoint {} listed twice in net '{}'", endpoint_str(e), n.id),
                        loc.clone(),
                    );
                    continue;
                }
                match e {
                    Endpoint::Pin(p) => {
                        pin_uses.entry(p.as_str()).or_default().push(ni);
                        match self.ix.defs[d].pin_index.get(p.as_str()) {
                            None => self.err(DiagKind::UnknownPort, format!("unknown pin '{p}'"), loc.clone()),
                            Some(&pi) => {
                                let w = def.pins[pi].width;
                                if w != n.width {
                                    self.err(
                                        DiagKind::WidthMismatch,
                                        format!("pin '{p}' has width {w} but net '{}' has width {}", n.id, n.width),
                                        loc.clone(),
                                    );
                                }
                            }
                        }
                    }
                    Endpoint::Port { inst, port } => {
                        port_uses.entry((inst.as_str(), port.as_str())).or_default().push(ni);
                        let Some(&ii) = self.ix.defs[d].inst_index.get(inst.as_str()) else {
                            self.err(DiagKind::UnknownInstance, format!("unknown instance '{inst}'"), loc.clone());
                            continue;
                        };
                        let known = match &def.instances[ii].kind {
                            InstKind::Def(_) => self.ix.defs[d].child_def[ii].is_some(),
                            InstKind::Atom(Atom::Splitter { width, parts }) => splitter_ok(*width, parts),
                            InstKind::Atom(_) => true,
                        };
                        if !known {
                            continue; // already reported (unknown def / bad splitter)
                        }
                        match self.ix.port(d, ii, port) {
                            None => self.err(
                                DiagKind::UnknownPort,
                                format!("instance '{inst}' has no port '{port}'"),
                                loc.clone(),
                            ),
                            Some((_, _, w)) if w != n.width => self.err(
                                DiagKind::WidthMismatch,
                                format!("port {inst}.{port} has width {w} but net '{}' has width {}", n.id, n.width),
                                loc.clone(),
                            ),
                            Some(_) => {}
                        }
                    }
                }
            }
        }
        // Net ids that shadow a pin name of another net.
        for n in &def.nets {
            if pins.contains(n.id.as_str()) && !n.ends.contains(&Endpoint::Pin(n.id.clone())) {
                self.warn(
                    DiagKind::Other,
                    format!("net id '{}' equals a pin name but the net is not connected to that pin", n.id),
                    loc_net(id, &n.id),
                );
            }
        }
        for p in &def.pins {
            match pin_uses.get(p.name.as_str()) {
                None => self.warn(DiagKind::BadPinUse, format!("pin '{}' is unused", p.name), loc_pin(id, &p.name)),
                Some(v) if v.len() > 1 => {
                    let names: Vec<_> = v.iter().map(|&i| def.nets[i].id.as_str()).collect();
                    self.err(
                        DiagKind::BadPinUse,
                        format!("pin '{}' is used by {} nets: {}", p.name, v.len(), names.join(", ")),
                        loc_pin(id, &p.name),
                    );
                }
                _ => {}
            }
        }
        let mut multi: Vec<_> = port_uses.into_iter().filter(|(_, v)| v.len() > 1).collect();
        multi.sort();
        for ((inst, port), v) in multi {
            let names: Vec<_> = v.iter().map(|&i| def.nets[i].id.as_str()).collect();
            self.err(
                DiagKind::BadPinUse,
                format!("port {inst}.{port} is used by {} nets: {}", v.len(), names.join(", ")),
                loc_inst(id, inst),
            );
        }
    }

    /// Detects cycles in the def dependency graph; returns a children-first order of all defs that
    /// are not part of (or above) a cycle.
    fn check_recursion(&mut self) -> Vec<usize> {
        let n = self.ix.defs.len();
        // Only the first def with each id participates (duplicates are already errors).
        let canonical: Vec<bool> = (0..n).map(|d| self.ix.def_by_id[self.ix.defs[d].def.id.as_str()] == d).collect();
        let children: Vec<Vec<usize>> = (0..n)
            .map(|d| {
                let mut c: Vec<usize> = self.ix.defs[d].child_def.iter().flatten().copied().collect();
                c.sort_unstable();
                c.dedup();
                c
            })
            .collect();
        // Iterative DFS with colours: 0 = white, 1 = on stack, 2 = done.
        let mut color = vec![0u8; n];
        let mut bad = vec![false; n];
        let mut order = Vec::with_capacity(n);
        let mut reported = HashSet::new();
        // Start from the top def so cycles are reported top-down.
        let top = self.ix.def_by_id.get(self.design.top.as_str()).copied();
        for start in top.into_iter().chain(0..n) {
            if color[start] != 0 || !canonical[start] {
                continue;
            }
            let mut stack: Vec<(usize, usize)> = vec![(start, 0)];
            color[start] = 1;
            while let Some(&mut (d, ref mut next)) = stack.last_mut() {
                if *next < children[d].len() {
                    let c = children[d][*next];
                    *next += 1;
                    match color[c] {
                        0 => {
                            color[c] = 1;
                            stack.push((c, 0));
                        }
                        1 => {
                            // Cycle: stack from c to d.
                            let pos = stack.iter().position(|&(x, _)| x == c).unwrap_or(0);
                            let mut cyc: Vec<usize> = stack[pos..].iter().map(|&(x, _)| x).collect();
                            for &x in &cyc {
                                bad[x] = true;
                            }
                            let mut key = cyc.clone();
                            key.sort_unstable();
                            if reported.insert(key) {
                                cyc.push(c);
                                let path: Vec<_> = cyc.iter().map(|&x| self.ix.defs[x].def.id.as_str()).collect();
                                let msg = format!("recursive def dependency: {}", path.join(" -> "));
                                let loc = self.ix.defs[c].def.id.clone();
                                self.err(DiagKind::RecursiveDef, msg, loc);
                            }
                        }
                        _ => {
                            if bad[c] {
                                bad[d] = true;
                            }
                        }
                    }
                } else {
                    color[d] = 2;
                    stack.pop();
                    if children[d].iter().any(|&c| bad[c]) {
                        bad[d] = true;
                    }
                    if !bad[d] {
                        order.push(d);
                    }
                    if let Some(&(p, _)) = stack.last()
                        && bad[d]
                    {
                        bad[p] = true;
                    }
                }
            }
        }
        order
    }

    /// Bit-level driver analysis in children-first order.
    fn check_drivers(&mut self, order: &[usize]) {
        let mut info: Vec<Option<PinDrive>> = (0..self.ix.defs.len()).map(|_| None).collect();
        for &d in order {
            let def = self.ix.defs[d].def;
            let Ok(t) = Template::build(&self.ix, d) else {
                self.err(DiagKind::Other, "def too large".into(), def.id.clone());
                continue;
            };
            let mut uf = UnionFind::new(t.bits as usize);
            for &(a, b) in &t.unions {
                uf.union(a, b);
            }
            // Pin-to-pin aliasing through children: every connected port bit is joined with the
            // first connected port bit of the same internal alias class. (The class representative
            // itself may be a pin that is left unconnected at this instance.)
            for (ii, _) in def.instances.iter().enumerate() {
                let Some(c) = self.ix.defs[d].child_def[ii] else { continue };
                let Some(ci) = info[c].as_ref() else { continue };
                let mut first: HashMap<(u32, u32), u32> = HashMap::new();
                for (pi, bits) in ci.alias.iter().enumerate() {
                    let Some(q) = t.inst_ports[ii][pi] else { continue };
                    for (j, &rep) in bits.iter().enumerate() {
                        let b = q + j as u32;
                        match first.get(&rep) {
                            Some(&f) => uf.union(b, f),
                            None => {
                                first.insert(rep, b);
                            }
                        }
                    }
                }
            }
            // Drivers: (local bit, zcap).
            let mut drivers: Vec<(u32, bool)> = Vec::new();
            for (ii, inst) in def.instances.iter().enumerate() {
                match &inst.kind {
                    InstKind::Atom(atom) => {
                        let zbits: Option<Vec<bool>> = match atom {
                            Atom::Gate { .. } | Atom::Not { .. } | Atom::Buf { .. } => None,
                            Atom::Clock { .. } | Atom::PowerOnReset { .. } => None,
                            Atom::Tri { width } => Some(vec![true; *width as usize]),
                            Atom::Constant { value } => Some(value.bits().map(|b| b == Logic::Z).collect()),
                            Atom::Splitter { .. } => continue,
                        };
                        let out = atom_port_count(atom) - 1;
                        let (_, w) = crate::structure::atom_port(atom, out);
                        let Some(o) = t.inst_ports[ii][out] else { continue };
                        for j in 0..w {
                            let z = zbits.as_ref().is_some_and(|z| z[j as usize]);
                            drivers.push((o + j, z));
                        }
                    }
                    InstKind::Def(_) => {
                        let Some(c) = self.ix.defs[d].child_def[ii] else { continue };
                        let Some(ci) = info[c].as_ref() else { continue };
                        let cdef = self.ix.defs[c].def;
                        // Count each internal alias class once per instance.
                        let mut counted = HashSet::new();
                        for (pi, p) in cdef.pins.iter().enumerate() {
                            if p.dir == Dir::In {
                                continue;
                            }
                            let Some(q) = t.inst_ports[ii][pi] else { continue };
                            for j in 0..p.width {
                                if let Some(z) = ci.drive[pi][j as usize]
                                    && counted.insert(ci.alias[pi][j as usize])
                                {
                                    drivers.push((q + j, z));
                                }
                            }
                        }
                    }
                }
            }
            // Per class: (count, all zcap).
            let mut class: HashMap<u32, (u32, bool)> = HashMap::new();
            for &(b, z) in &drivers {
                let e = class.entry(uf.find(b)).or_insert((0, true));
                e.0 += 1;
                e.1 &= z;
            }
            // Terminations per class: nets of this def plus the internal classes of child pins.
            // Conflicting (different, non-`none`) terminations on one class are errors, reported
            // once per net at the nets containing the conflicting bits.
            let mut class_term: HashMap<u32, Termination> = HashMap::new();
            let mut term_conflict: HashMap<u32, (Termination, Termination)> = HashMap::new();
            let mut add_term = |uf: &mut UnionFind, bit: u32, tm: Termination| {
                if tm == Termination::None {
                    return;
                }
                let r = uf.find(bit);
                match class_term.get(&r) {
                    None => {
                        class_term.insert(r, tm);
                    }
                    Some(&old) if old != tm => {
                        term_conflict.entry(r).or_insert((old, tm));
                    }
                    Some(_) => {}
                }
            };
            for (ni, n) in def.nets.iter().enumerate() {
                for j in 0..n.width {
                    add_term(&mut uf, t.net_off[ni] + j, n.termination);
                }
            }
            for (ii, _) in def.instances.iter().enumerate() {
                let Some(c) = self.ix.defs[d].child_def[ii] else { continue };
                let Some(ci) = info[c].as_ref() else { continue };
                for (pi, bits) in ci.term.iter().enumerate() {
                    let Some(q) = t.inst_ports[ii][pi] else { continue };
                    for (j, &tm) in bits.iter().enumerate() {
                        add_term(&mut uf, q + j as u32, tm);
                    }
                }
            }
            if !term_conflict.is_empty() {
                let mut done = HashSet::new();
                for (ni, n) in def.nets.iter().enumerate() {
                    let mut pairs: Vec<(Termination, Termination)> = Vec::new();
                    for j in 0..n.width {
                        let r = uf.find(t.net_off[ni] + j);
                        if let Some(&pair) = term_conflict.get(&r)
                            && done.insert(r)
                            && !pairs.contains(&pair)
                        {
                            pairs.push(pair);
                        }
                    }
                    for (a, b) in pairs {
                        self.err(
                            DiagKind::Other,
                            format!("conflicting terminations {a:?} and {b:?} on net '{}'", n.id),
                            loc_net(&def.id, &n.id),
                        );
                    }
                }
            }
            // In pins driven from inside.
            for (pi, p) in def.pins.iter().enumerate() {
                if p.dir != Dir::In || t.pin_virtual[pi] {
                    continue;
                }
                let driven: Vec<u32> =
                    (0..p.width).filter(|&j| class.contains_key(&uf.find(t.pin_off[pi] + j))).collect();
                if !driven.is_empty() {
                    self.err(
                        DiagKind::BadPinUse,
                        format!("input pin '{}' is driven from inside the def (bits {:?})", p.name, driven),
                        loc_pin(&def.id, &p.name),
                    );
                }
            }
            // Multiple drivers, reported once per class at the first net containing it.
            let mut done = HashSet::new();
            for (ni, n) in def.nets.iter().enumerate() {
                let mut bad_bits = Vec::new();
                for j in 0..n.width {
                    let r = uf.find(t.net_off[ni] + j);
                    if let Some(&(cnt, z)) = class.get(&r)
                        && cnt >= 2
                        && !z
                        && done.insert(r)
                    {
                        bad_bits.push((j, cnt));
                    }
                }
                if !bad_bits.is_empty() {
                    let desc: Vec<_> = bad_bits.iter().map(|(j, c)| format!("bit {j}: {c} drivers")).collect();
                    self.warn(
                        DiagKind::MultipleDrivers,
                        format!("net '{}' has multiple drivers, not all Z-capable ({})", n.id, desc.join(", ")),
                        loc_net(&def.id, &n.id),
                    );
                }
            }
            // Export pin drive state and aliasing.
            let mut rep: HashMap<u32, (u32, u32)> = HashMap::new();
            let mut drive = Vec::with_capacity(def.pins.len());
            let mut alias = Vec::with_capacity(def.pins.len());
            let mut term = Vec::with_capacity(def.pins.len());
            for (pi, p) in def.pins.iter().enumerate() {
                let mut dv = Vec::with_capacity(p.width as usize);
                let mut av = Vec::with_capacity(p.width as usize);
                let mut tv = Vec::with_capacity(p.width as usize);
                for j in 0..p.width {
                    let r = uf.find(t.pin_off[pi] + j);
                    dv.push(class.get(&r).map(|&(_, z)| z));
                    av.push(*rep.entry(r).or_insert((pi as u32, j)));
                    tv.push(class_term.get(&r).copied().unwrap_or_default());
                }
                drive.push(dv);
                alias.push(av);
                term.push(tv);
            }
            info[d] = Some(PinDrive { drive, alias, term });
        }
    }
}

/// Parameter errors of an atom.
fn atom_errors(atom: &Atom) -> Vec<String> {
    let mut e = Vec::new();
    match atom {
        Atom::Gate { inputs, width, reduce, .. } => {
            if *inputs < 1 {
                e.push("gate must have at least 1 input".into());
            }
            if *reduce && *inputs != 1 {
                e.push(format!("reduce gate must have exactly 1 input (has {inputs})"));
            }
            if *width == 0 {
                e.push("gate width must be ≥ 1".into());
            }
        }
        Atom::Not { width } | Atom::Buf { width } | Atom::Tri { width } => {
            if *width == 0 {
                e.push("width must be ≥ 1".into());
            }
        }
        Atom::Splitter { width, parts } => {
            if *width == 0 {
                e.push("splitter width must be ≥ 1".into());
            }
            for (i, &(lo, hi)) in parts.iter().enumerate() {
                if lo > hi || hi >= *width {
                    e.push(format!("splitter part p{i} = ({lo}, {hi}) out of range for width {width}"));
                }
            }
        }
        Atom::Constant { value } => {
            if value.width() == 0 {
                e.push("constant width must be ≥ 1".into());
            }
        }
        Atom::Clock { period, high, .. } => {
            if *period < 2 {
                e.push(format!("clock period must be ≥ 2 (is {period})"));
            }
            if *high == 0 || *high >= *period {
                e.push(format!("clock high must be in (0, period) (high {high}, period {period})"));
            }
        }
        Atom::PowerOnReset { .. } => {}
    }
    e
}
