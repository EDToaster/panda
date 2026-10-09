//! Ergonomic programmatic construction of designs — used by tests, circuit generators
//! (`panda-circuits`, `panda-cpu8`) and the CLI.
//!
//! ```ignore
//! let mut b = DefBuilder::new("sr_latch");
//! let s_n = b.input("S_n", 1);
//! let r_n = b.input("R_n", 1);
//! let qn = b.wire(1);                 // forward reference for the feedback loop
//! let q = b.nand(&[s_n, qn]);
//! b.nand_to(&[r_n, q], qn);           // drive the previously created wire
//! b.output("Q", q);
//! b.output("Q_n", qn);
//! let def = b.finish();
//! ```
//!
//! Semantics:
//! * A [`Sig`] is a handle to a builder-local net of a fixed width.
//! * `*_to` variants drive an existing net instead of creating a new output net.
//! * [`DefBuilder::connect`] merges two nets (union). Merging nets of different widths panics.
//! * Instance ids are auto-generated (`g0`, `g1`, ... for atoms, `u0`, `u1`, ... for defs) unless
//!   the `_named` variants are used. Net ids are auto-generated (`n0`, ...) unless [`DefBuilder::name`]
//!   is called; pin nets are named after the pin.
//! * Builder misuse (width mismatch, unknown pin) panics with a descriptive message — the builder is
//!   for trusted programmatic construction; untrusted JSON goes through [`crate::validate`].
//!
//! Details of [`DefBuilder::finish`]:
//! * one [`Net`] is emitted per union class that has at least one endpoint (nets that were created
//!   but never connected to anything are dropped);
//! * a class's id is, in priority order: its explicit name (the most recent [`DefBuilder::name`]
//!   call on any member), the name of its first pin (declaration order), or an auto id `n<k>`;
//! * ids are kept unique: an explicit name that collides with another net's id or with the name of
//!   a pin that is not on that net gets a suffix `_1`, `_2`, ...; auto ids skip taken names;
//! * auto instance ids skip ids already taken by explicitly named instances.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};

use crate::model::*;
use crate::value::Value;

/// Handle to a builder-local net.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sig {
    pub(crate) net: u32,
    /// Width of the net in bits.
    pub width: u32,
    /// Identity of the [`DefBuilder`] that created this handle (catches cross-builder misuse).
    pub(crate) owner: u32,
}

/// Lightweight reference to a def's interface, used to instantiate it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DefRef {
    /// The def's id.
    pub id: DefId,
    /// The def's pins in declaration order.
    pub pins: Vec<PinDecl>,
}

impl DefRef {
    /// Interface reference of an existing def.
    pub fn of(def: &ComponentDef) -> DefRef {
        DefRef { id: def.id.clone(), pins: def.pins.clone() }
    }
}

/// Collects defs into a [`Design`].
#[derive(Default)]
pub struct DesignBuilder {
    defs: Vec<ComponentDef>,
    index: HashMap<DefId, usize>,
}

impl DesignBuilder {
    /// Empty design builder.
    pub fn new() -> Self {
        Self::default()
    }
    /// Add a def (replacing an existing one with the same id is a panic). Returns its ref.
    pub fn add(&mut self, def: ComponentDef) -> DefRef {
        assert!(!self.index.contains_key(&def.id), "DesignBuilder::add: def '{}' already exists", def.id);
        let r = DefRef::of(&def);
        self.index.insert(def.id.clone(), self.defs.len());
        self.defs.push(def);
        r
    }
    /// Add the def if no def with that id exists yet; return the ref either way.
    /// Generators use this to share sub-defs: `let fa = db.add_once(full_adder());`
    pub fn add_once(&mut self, def: ComponentDef) -> DefRef {
        match self.index.get(&def.id) {
            Some(&i) => DefRef::of(&self.defs[i]),
            None => self.add(def),
        }
    }
    /// Interface of the def with this id, if present.
    pub fn get(&self, id: &str) -> Option<DefRef> {
        self.index.get(id).map(|&i| DefRef::of(&self.defs[i]))
    }
    /// True if a def with this id has been added.
    pub fn contains(&self, id: &str) -> bool {
        self.index.contains_key(id)
    }
    /// Merge all defs of another design (dedup by id; identical duplicates allowed, conflicting panic).
    pub fn merge(&mut self, other: Design) {
        for def in other.defs {
            match self.index.get(&def.id) {
                Some(&i) => {
                    assert!(self.defs[i] == def, "DesignBuilder::merge: conflicting definitions for def '{}'", def.id)
                }
                None => {
                    self.add(def);
                }
            }
        }
    }
    /// Finish the design with `top` as the top def. Panics if `top` was never added.
    pub fn build(self, top: &DefRef) -> Design {
        assert!(self.index.contains_key(&top.id), "DesignBuilder::build: top def '{}' was not added", top.id);
        Design { format: FORMAT.to_string(), defs: self.defs, top: top.id.clone() }
    }
}

static NEXT_BUILDER: AtomicU32 = AtomicU32::new(0);

/// Builds one [`ComponentDef`].
pub struct DefBuilder {
    id: String,
    name: String,
    owner: u32,
    pins: Vec<PinDecl>,
    /// Builder net of each pin.
    pin_nets: Vec<u32>,
    /// Union-find parent per builder net.
    parent: Vec<u32>,
    widths: Vec<u32>,
    /// Explicit name with its sequence number (per class root; the latest call wins).
    names: Vec<Option<(u64, String)>>,
    /// Termination per class root.
    terms: Vec<Termination>,
    name_seq: u64,
    instances: Vec<Instance>,
    /// `(builder net, endpoint)` in insertion order.
    ends: Vec<(u32, Endpoint)>,
    inst_ids: HashSet<String>,
    next_atom: u32,
    next_def: u32,
}

impl DefBuilder {
    /// Start building a def with the given id (the display name defaults to the id).
    pub fn new(id: &str) -> Self {
        DefBuilder {
            id: id.to_string(),
            name: id.to_string(),
            owner: NEXT_BUILDER.fetch_add(1, Ordering::Relaxed),
            pins: Vec::new(),
            pin_nets: Vec::new(),
            parent: Vec::new(),
            widths: Vec::new(),
            names: Vec::new(),
            terms: Vec::new(),
            name_seq: 0,
            instances: Vec::new(),
            ends: Vec::new(),
            inst_ids: HashSet::new(),
            next_atom: 0,
            next_def: 0,
        }
    }
    /// Set the human-readable name of the def.
    pub fn set_name(&mut self, name: &str) {
        self.name = name.to_string();
    }

    // ---- internals ----

    fn new_net(&mut self, width: u32) -> Sig {
        assert!(width > 0, "DefBuilder '{}': net width must be ≥ 1", self.id);
        let net = self.parent.len() as u32;
        self.parent.push(net);
        self.widths.push(width);
        self.names.push(None);
        self.terms.push(Termination::None);
        Sig { net, width, owner: self.owner }
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let p = self.parent[x as usize];
            self.parent[x as usize] = self.parent[p as usize];
            x = p;
        }
        x
    }

    fn check(&self, s: Sig, what: &str) {
        assert!(
            s.owner == self.owner && (s.net as usize) < self.parent.len(),
            "DefBuilder '{}': {what}: signal belongs to a different builder",
            self.id
        );
    }

    fn check_width(&self, s: Sig, width: u32, what: &str) {
        self.check(s, what);
        assert!(s.width == width, "DefBuilder '{}': {what}: expected width {width}, got {}", self.id, s.width);
    }

    fn attach(&mut self, s: Sig, e: Endpoint) {
        self.ends.push((s.net, e));
    }

    fn fresh_inst_id(&mut self, def: bool) -> String {
        loop {
            let id = if def {
                self.next_def += 1;
                format!("u{}", self.next_def - 1)
            } else {
                self.next_atom += 1;
                format!("g{}", self.next_atom - 1)
            };
            if !self.inst_ids.contains(&id) {
                return id;
            }
        }
    }

    fn add_inst(&mut self, id: String, kind: InstKind) -> String {
        assert!(self.inst_ids.insert(id.clone()), "DefBuilder '{}': duplicate instance id '{id}'", self.id);
        self.instances.push(Instance { id: id.clone(), kind });
        id
    }

    /// Add an atom with auto id; connect `(port, sig)` pairs.
    fn atom(&mut self, atom: Atom, conns: &[(String, Sig)]) {
        let id = self.fresh_inst_id(false);
        let id = self.add_inst(id, InstKind::Atom(atom));
        for (port, s) in conns {
            self.attach(*s, Endpoint::Port { inst: id.clone(), port: port.clone() });
        }
    }

    fn add_pin(&mut self, name: &str, dir: Dir, width: u32) {
        assert!(!self.pins.iter().any(|p| p.name == name), "DefBuilder '{}': duplicate pin '{name}'", self.id);
        self.pins.push(PinDecl { name: name.to_string(), dir, width });
    }

    // ---- pins ----
    /// Declare an `In` pin; returns its net.
    pub fn input(&mut self, name: &str, width: u32) -> Sig {
        self.add_pin(name, Dir::In, width);
        let s = self.new_net(width);
        self.pin_nets.push(s.net);
        self.attach(s, Endpoint::Pin(name.to_string()));
        s
    }
    /// Declare an `Out` pin connected to `sig`.
    pub fn output(&mut self, name: &str, sig: Sig) {
        self.check(sig, "output");
        self.add_pin(name, Dir::Out, sig.width);
        self.pin_nets.push(sig.net);
        self.attach(sig, Endpoint::Pin(name.to_string()));
    }
    /// Declare an `InOut` pin; returns its net.
    pub fn inout(&mut self, name: &str, width: u32) -> Sig {
        self.add_pin(name, Dir::InOut, width);
        let s = self.new_net(width);
        self.pin_nets.push(s.net);
        self.attach(s, Endpoint::Pin(name.to_string()));
        s
    }

    // ---- nets ----
    /// Fresh undriven net (forward reference for feedback).
    pub fn wire(&mut self, width: u32) -> Sig {
        self.new_net(width)
    }
    /// Give the net a readable id (used in net names / probing). Last call wins.
    pub fn name(&mut self, sig: Sig, name: &str) -> Sig {
        self.check(sig, "name");
        let r = self.find(sig.net);
        self.name_seq += 1;
        self.names[r as usize] = Some((self.name_seq, name.to_string()));
        sig
    }
    /// Set the termination of the net (SPEC §3.3). Conflicting terminations on merge panic.
    pub fn terminate(&mut self, sig: Sig, t: Termination) {
        self.check(sig, "terminate");
        let r = self.find(sig.net);
        self.terms[r as usize] = t;
    }
    /// Merge two nets of equal width.
    pub fn connect(&mut self, a: Sig, b: Sig) {
        self.check(a, "connect");
        self.check_width(b, a.width, "connect");
        let (ra, rb) = (self.find(a.net), self.find(b.net));
        if ra == rb {
            return;
        }
        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
        let (tl, th) = (self.terms[lo as usize], self.terms[hi as usize]);
        assert!(
            tl == Termination::None || th == Termination::None || tl == th,
            "DefBuilder '{}': connect: conflicting terminations {tl:?} and {th:?}",
            self.id
        );
        if tl == Termination::None {
            self.terms[lo as usize] = th;
        }
        let name_hi = self.names[hi as usize].take();
        let newer = match (&self.names[lo as usize], &name_hi) {
            (Some((sl, _)), Some((sh, _))) => sh > sl,
            (None, Some(_)) => true,
            _ => false,
        };
        if newer {
            self.names[lo as usize] = name_hi;
        }
        self.parent[hi as usize] = lo;
    }

    // ---- gates ----
    /// N-input bitwise gate over equal-width inputs; returns the new output net.
    pub fn gate(&mut self, op: GateOp, ins: &[Sig]) -> Sig {
        assert!(!ins.is_empty(), "DefBuilder '{}': gate needs at least one input", self.id);
        let out = self.new_net(ins[0].width);
        self.gate_to(op, ins, out);
        out
    }
    /// N-input bitwise gate driving the existing net `out`.
    pub fn gate_to(&mut self, op: GateOp, ins: &[Sig], out: Sig) {
        assert!(!ins.is_empty(), "DefBuilder '{}': gate needs at least one input", self.id);
        let w = ins[0].width;
        for s in ins {
            self.check_width(*s, w, "gate input");
        }
        self.check_width(out, w, "gate output");
        let mut conns: Vec<(String, Sig)> = ins.iter().enumerate().map(|(i, s)| (format!("in{i}"), *s)).collect();
        conns.push(("out".into(), out));
        self.atom(Atom::Gate { op, inputs: ins.len() as u32, width: w, reduce: false }, &conns);
    }
    /// AND gate.
    pub fn and(&mut self, ins: &[Sig]) -> Sig {
        self.gate(GateOp::And, ins)
    }
    /// OR gate.
    pub fn or(&mut self, ins: &[Sig]) -> Sig {
        self.gate(GateOp::Or, ins)
    }
    /// NAND gate.
    pub fn nand(&mut self, ins: &[Sig]) -> Sig {
        self.gate(GateOp::Nand, ins)
    }
    /// NOR gate.
    pub fn nor(&mut self, ins: &[Sig]) -> Sig {
        self.gate(GateOp::Nor, ins)
    }
    /// XOR gate.
    pub fn xor(&mut self, ins: &[Sig]) -> Sig {
        self.gate(GateOp::Xor, ins)
    }
    /// XNOR gate.
    pub fn xnor(&mut self, ins: &[Sig]) -> Sig {
        self.gate(GateOp::Xnor, ins)
    }
    /// NAND gate driving `out`.
    pub fn nand_to(&mut self, ins: &[Sig], out: Sig) {
        self.gate_to(GateOp::Nand, ins, out)
    }
    /// NOR gate driving `out`.
    pub fn nor_to(&mut self, ins: &[Sig], out: Sig) {
        self.gate_to(GateOp::Nor, ins, out)
    }
    /// Reduce gate: W-bit → 1-bit.
    pub fn reduce(&mut self, op: GateOp, a: Sig) -> Sig {
        self.check(a, "reduce");
        let out = self.new_net(1);
        self.atom(
            Atom::Gate { op, inputs: 1, width: a.width, reduce: true },
            &[("in0".into(), a), ("out".into(), out)],
        );
        out
    }
    /// Inverter.
    pub fn not(&mut self, a: Sig) -> Sig {
        let out = self.new_net(a.width);
        self.not_to(a, out);
        out
    }
    /// Inverter driving `out`.
    pub fn not_to(&mut self, a: Sig, out: Sig) {
        self.check(a, "not");
        self.check_width(out, a.width, "not output");
        self.atom(Atom::Not { width: a.width }, &[("in".into(), a), ("out".into(), out)]);
    }
    /// Buffer (a gate with delay 1).
    pub fn buf(&mut self, a: Sig) -> Sig {
        let out = self.new_net(a.width);
        self.buf_to(a, out);
        out
    }
    /// Buffer driving `out`.
    pub fn buf_to(&mut self, a: Sig, out: Sig) {
        self.check(a, "buf");
        self.check_width(out, a.width, "buf output");
        self.atom(Atom::Buf { width: a.width }, &[("in".into(), a), ("out".into(), out)]);
    }
    /// Tri-state buffer (`en` must be 1 bit).
    pub fn tri(&mut self, d: Sig, en: Sig) -> Sig {
        let out = self.new_net(d.width);
        self.tri_to(d, en, out);
        out
    }
    /// Tri-state buffer driving `out`.
    pub fn tri_to(&mut self, d: Sig, en: Sig, out: Sig) {
        self.check(d, "tri");
        self.check_width(en, 1, "tri enable");
        self.check_width(out, d.width, "tri output");
        self.atom(Atom::Tri { width: d.width }, &[("d".into(), d), ("en".into(), en), ("out".into(), out)]);
    }

    // ---- sources ----
    /// Constant source (may contain X/Z bits).
    pub fn constant(&mut self, v: Value) -> Sig {
        let out = self.new_net(v.width());
        self.atom(Atom::Constant { value: v }, &[("out".into(), out)]);
        out
    }
    /// Known constant `v` of `width` bits.
    pub fn lit(&mut self, width: u32, v: u64) -> Sig {
        self.constant(Value::from_u64(width, v))
    }
    /// Clock source (only valid in the top def).
    pub fn clock(&mut self, period: u32, high: u32, phase: u32) -> Sig {
        let out = self.new_net(1);
        self.atom(Atom::Clock { period, high, phase }, &[("out".into(), out)]);
        out
    }
    /// PowerOnReset source (only valid in the top def).
    pub fn power_on_reset(&mut self, ticks: u32) -> Sig {
        let out = self.new_net(1);
        self.atom(Atom::PowerOnReset { ticks }, &[("out".into(), out)]);
        out
    }

    // ---- bus wiring (Splitter atoms) ----
    /// Bits `lo..lo+width` of `s`.
    pub fn slice(&mut self, s: Sig, lo: u32, width: u32) -> Sig {
        self.check(s, "slice");
        assert!(
            width > 0 && lo.checked_add(width).is_some_and(|e| e <= s.width),
            "DefBuilder '{}': slice {lo}+{width} out of range for width {}",
            self.id,
            s.width
        );
        let out = self.new_net(width);
        self.atom(
            Atom::Splitter { width: s.width, parts: vec![(lo, lo + width - 1)] },
            &[("whole".into(), s), ("p0".into(), out)],
        );
        out
    }
    /// Bit `i` of `s`.
    pub fn bit(&mut self, s: Sig, i: u32) -> Sig {
        self.slice(s, i, 1)
    }
    /// All bits of `s`, LSB first (one Splitter).
    pub fn bits(&mut self, s: Sig) -> Vec<Sig> {
        self.check(s, "bits");
        let outs: Vec<Sig> = (0..s.width).map(|_| self.new_net(1)).collect();
        let mut conns = vec![("whole".to_string(), s)];
        conns.extend(outs.iter().enumerate().map(|(i, o)| (format!("p{i}"), *o)));
        self.atom(Atom::Splitter { width: s.width, parts: (0..s.width).map(|i| (i, i)).collect() }, &conns);
        outs
    }
    /// General splitter: one part net per `(lo, hi)` bit range of `whole` (inclusive, LSB first).
    /// Ranges may overlap and may come in any order; overlapping parts alias the shared bits.
    pub fn splitter(&mut self, whole: Sig, parts: &[(u32, u32)]) -> Vec<Sig> {
        self.check(whole, "splitter");
        for &(lo, hi) in parts {
            assert!(
                lo <= hi && hi < whole.width,
                "DefBuilder '{}': splitter part ({lo}, {hi}) out of range for width {}",
                self.id,
                whole.width
            );
        }
        let outs: Vec<Sig> = parts.iter().map(|&(lo, hi)| self.new_net(hi - lo + 1)).collect();
        let mut conns = vec![("whole".to_string(), whole)];
        conns.extend(outs.iter().enumerate().map(|(i, o)| (format!("p{i}"), *o)));
        self.atom(Atom::Splitter { width: whole.width, parts: parts.to_vec() }, &conns);
        outs
    }
    /// Concatenate, LSB first: `concat(&[lo, hi])`.
    pub fn concat(&mut self, parts: &[Sig]) -> Sig {
        assert!(!parts.is_empty(), "DefBuilder '{}': concat needs at least one part", self.id);
        let mut ranges = Vec::with_capacity(parts.len());
        let mut lo = 0u32;
        for p in parts {
            self.check(*p, "concat");
            ranges.push((lo, lo + p.width - 1));
            lo += p.width;
        }
        let out = self.new_net(lo);
        let mut conns = vec![("whole".to_string(), out)];
        conns.extend(parts.iter().enumerate().map(|(i, p)| (format!("p{i}"), *p)));
        self.atom(Atom::Splitter { width: lo, parts: ranges }, &conns);
        out
    }

    // ---- instances ----
    /// Instantiate a def. `inputs` are the def's `In` and `InOut` pins **in pin-declaration order**;
    /// returns the `Out` pins in declaration order. Widths must match.
    pub fn inst(&mut self, def: &DefRef, inputs: &[Sig]) -> Vec<Sig> {
        let id = self.fresh_inst_id(true);
        self.inst_named(&id, def, inputs)
    }
    /// Like [`Self::inst`] with an explicit instance id.
    pub fn inst_named(&mut self, id: &str, def: &DefRef, inputs: &[Sig]) -> Vec<Sig> {
        let n_in = def.pins.iter().filter(|p| p.dir != Dir::Out).count();
        assert!(
            inputs.len() == n_in,
            "DefBuilder '{}': instance '{id}' of '{}' expects {n_in} In/InOut signals, got {}",
            self.id,
            def.id,
            inputs.len()
        );
        let id = self.add_inst(id.to_string(), InstKind::Def(def.id.clone()));
        let mut it = inputs.iter();
        let mut outs = Vec::new();
        for p in &def.pins {
            let s = if p.dir == Dir::Out {
                let s = self.new_net(p.width);
                outs.push(s);
                s
            } else {
                let s = *it.next().expect("counted above");
                self.check_width(s, p.width, &format!("instance '{id}' pin '{}'", p.name));
                s
            };
            self.attach(s, Endpoint::Port { inst: id.clone(), port: p.name.clone() });
        }
        outs
    }
    /// Instantiate connecting pins by name. Every `In`/`InOut` pin must be given; `Out` pins may be
    /// given (then that net is driven by the pin) or omitted (fresh net created). Returns every pin's net.
    pub fn inst_map(&mut self, id: &str, def: &DefRef, conns: &[(&str, Sig)]) -> HashMap<String, Sig> {
        let mut given: HashMap<&str, Sig> = HashMap::new();
        for (name, s) in conns {
            assert!(
                def.pins.iter().any(|p| p.name == *name),
                "DefBuilder '{}': instance '{id}': def '{}' has no pin '{name}'",
                self.id,
                def.id
            );
            assert!(
                given.insert(name, *s).is_none(),
                "DefBuilder '{}': instance '{id}': pin '{name}' connected twice",
                self.id
            );
        }
        let id = self.add_inst(id.to_string(), InstKind::Def(def.id.clone()));
        let mut out = HashMap::with_capacity(def.pins.len());
        for p in &def.pins {
            let s = match given.get(p.name.as_str()) {
                Some(&s) => {
                    self.check_width(s, p.width, &format!("instance '{id}' pin '{}'", p.name));
                    s
                }
                None => {
                    assert!(
                        p.dir == Dir::Out,
                        "DefBuilder '{}': instance '{id}': {:?} pin '{}' of '{}' must be connected",
                        self.id,
                        p.dir,
                        p.name,
                        def.id
                    );
                    self.new_net(p.width)
                }
            };
            self.attach(s, Endpoint::Port { inst: id.clone(), port: p.name.clone() });
            out.insert(p.name.clone(), s);
        }
        out
    }

    /// Emit the [`ComponentDef`] (see the module docs for net naming rules).
    pub fn finish(mut self) -> ComponentDef {
        let n = self.parent.len();
        let roots: Vec<u32> = (0..n as u32).map(|i| self.find(i)).collect();
        // Classes with endpoints, in order of their root (= smallest member).
        let mut class_of_root: Vec<u32> = vec![u32::MAX; n];
        let mut class_root: Vec<u32> = Vec::new();
        let mut class_ends: Vec<Vec<Endpoint>> = Vec::new();
        let mut used_roots: Vec<u32> = self.ends.iter().map(|(net, _)| roots[*net as usize]).collect();
        used_roots.sort_unstable();
        used_roots.dedup();
        for r in used_roots {
            class_of_root[r as usize] = class_root.len() as u32;
            class_root.push(r);
            class_ends.push(Vec::new());
        }
        for (net, e) in std::mem::take(&mut self.ends) {
            class_ends[class_of_root[roots[net as usize] as usize] as usize].push(e);
        }
        // Pin-derived names: first pin (declaration order) of each class.
        let mut pin_name: Vec<Option<&str>> = vec![None; class_root.len()];
        for (p, &net) in self.pins.iter().zip(&self.pin_nets) {
            let c = class_of_root[roots[net as usize] as usize] as usize;
            pin_name[c].get_or_insert(p.name.as_str());
        }
        let pin_names: HashSet<&str> = self.pins.iter().map(|p| p.name.as_str()).collect();
        let mut taken: HashSet<String> = HashSet::new();
        let mut ids: Vec<Option<String>> = vec![None; class_root.len()];
        // 1. Pin names (unique by construction).
        for (c, pn) in pin_name.iter().enumerate() {
            if self.names[class_root[c] as usize].is_none()
                && let Some(pn) = pn
            {
                taken.insert(pn.to_string());
                ids[c] = Some(pn.to_string());
            }
        }
        // 2. Explicit names, in order of naming.
        let mut explicit: Vec<(u64, usize, &str)> = (0..class_root.len())
            .filter_map(|c| self.names[class_root[c] as usize].as_ref().map(|(s, nm)| (*s, c, nm.as_str())))
            .collect();
        explicit.sort_unstable();
        for (_, c, nm) in explicit {
            let own_pins: HashSet<&str> = class_ends[c]
                .iter()
                .filter_map(|e| if let Endpoint::Pin(p) = e { Some(p.as_str()) } else { None })
                .collect();
            let free = |s: &str, taken: &HashSet<String>| {
                !taken.contains(s) && (!pin_names.contains(s) || own_pins.contains(s))
            };
            let mut id = nm.to_string();
            let mut k = 1;
            while !free(&id, &taken) {
                id = format!("{nm}_{k}");
                k += 1;
            }
            taken.insert(id.clone());
            ids[c] = Some(id);
        }
        // 3. Auto ids.
        let mut next = 0u32;
        let mut nets = Vec::with_capacity(class_root.len());
        for (c, ends) in class_ends.into_iter().enumerate() {
            let id = match ids[c].take() {
                Some(id) => id,
                None => loop {
                    let id = format!("n{next}");
                    next += 1;
                    if !taken.contains(&id) && !pin_names.contains(id.as_str()) {
                        break id;
                    }
                },
            };
            let r = class_root[c] as usize;
            nets.push(Net { id, width: self.widths[r], termination: self.terms[r], ends });
        }
        ComponentDef {
            id: self.id,
            name: self.name,
            version: 0,
            pins: self.pins,
            instances: self.instances,
            nets,
            editor: serde_json::Value::Null,
        }
    }
}
