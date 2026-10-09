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

use std::collections::HashMap;

use crate::model::*;
use crate::value::Value;

/// Handle to a builder-local net.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sig {
    pub(crate) net: u32,
    pub width: u32,
}

/// Lightweight reference to a def's interface, used to instantiate it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DefRef {
    pub id: DefId,
    pub pins: Vec<PinDecl>,
}

impl DefRef {
    pub fn of(def: &ComponentDef) -> DefRef {
        DefRef { id: def.id.clone(), pins: def.pins.clone() }
    }
}

/// Collects defs into a [`Design`].
#[derive(Default)]
pub struct DesignBuilder {
    defs: Vec<ComponentDef>,
}

impl DesignBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    /// Add a def (replacing an existing one with the same id is a panic). Returns its ref.
    pub fn add(&mut self, def: ComponentDef) -> DefRef {
        todo!()
    }
    /// Add the def if no def with that id exists yet; return the ref either way.
    /// Generators use this to share sub-defs: `let fa = db.add_once(full_adder());`
    pub fn add_once(&mut self, def: ComponentDef) -> DefRef {
        todo!()
    }
    pub fn get(&self, id: &str) -> Option<DefRef> {
        todo!()
    }
    pub fn contains(&self, id: &str) -> bool {
        todo!()
    }
    /// Merge all defs of another design (dedup by id; identical duplicates allowed, conflicting panic).
    pub fn merge(&mut self, other: Design) {
        todo!()
    }
    pub fn build(self, top: &DefRef) -> Design {
        todo!()
    }
}

/// Builds one [`ComponentDef`].
pub struct DefBuilder {
    // implementation-defined
}

impl DefBuilder {
    pub fn new(id: &str) -> Self {
        todo!()
    }
    pub fn set_name(&mut self, name: &str) {
        todo!()
    }

    // ---- pins ----
    /// Declare an `In` pin; returns its net.
    pub fn input(&mut self, name: &str, width: u32) -> Sig {
        todo!()
    }
    /// Declare an `Out` pin connected to `sig`.
    pub fn output(&mut self, name: &str, sig: Sig) {
        todo!()
    }
    /// Declare an `InOut` pin; returns its net.
    pub fn inout(&mut self, name: &str, width: u32) -> Sig {
        todo!()
    }

    // ---- nets ----
    /// Fresh undriven net (forward reference for feedback).
    pub fn wire(&mut self, width: u32) -> Sig {
        todo!()
    }
    /// Give the net a readable id (used in net names / probing). Last call wins.
    pub fn name(&mut self, sig: Sig, name: &str) -> Sig {
        todo!()
    }
    pub fn terminate(&mut self, sig: Sig, t: Termination) {
        todo!()
    }
    /// Merge two nets of equal width.
    pub fn connect(&mut self, a: Sig, b: Sig) {
        todo!()
    }

    // ---- gates ----
    pub fn gate(&mut self, op: GateOp, ins: &[Sig]) -> Sig {
        todo!()
    }
    pub fn gate_to(&mut self, op: GateOp, ins: &[Sig], out: Sig) {
        todo!()
    }
    pub fn and(&mut self, ins: &[Sig]) -> Sig {
        self.gate(GateOp::And, ins)
    }
    pub fn or(&mut self, ins: &[Sig]) -> Sig {
        self.gate(GateOp::Or, ins)
    }
    pub fn nand(&mut self, ins: &[Sig]) -> Sig {
        self.gate(GateOp::Nand, ins)
    }
    pub fn nor(&mut self, ins: &[Sig]) -> Sig {
        self.gate(GateOp::Nor, ins)
    }
    pub fn xor(&mut self, ins: &[Sig]) -> Sig {
        self.gate(GateOp::Xor, ins)
    }
    pub fn xnor(&mut self, ins: &[Sig]) -> Sig {
        self.gate(GateOp::Xnor, ins)
    }
    pub fn nand_to(&mut self, ins: &[Sig], out: Sig) {
        self.gate_to(GateOp::Nand, ins, out)
    }
    pub fn nor_to(&mut self, ins: &[Sig], out: Sig) {
        self.gate_to(GateOp::Nor, ins, out)
    }
    /// Reduce gate: W-bit → 1-bit.
    pub fn reduce(&mut self, op: GateOp, a: Sig) -> Sig {
        todo!()
    }
    pub fn not(&mut self, a: Sig) -> Sig {
        todo!()
    }
    pub fn not_to(&mut self, a: Sig, out: Sig) {
        todo!()
    }
    pub fn buf(&mut self, a: Sig) -> Sig {
        todo!()
    }
    pub fn buf_to(&mut self, a: Sig, out: Sig) {
        todo!()
    }
    /// Tri-state buffer (`en` must be 1 bit).
    pub fn tri(&mut self, d: Sig, en: Sig) -> Sig {
        todo!()
    }
    pub fn tri_to(&mut self, d: Sig, en: Sig, out: Sig) {
        todo!()
    }

    // ---- sources ----
    pub fn constant(&mut self, v: Value) -> Sig {
        todo!()
    }
    /// Known constant `v` of `width` bits.
    pub fn lit(&mut self, width: u32, v: u64) -> Sig {
        self.constant(Value::from_u64(width, v))
    }
    /// Clock source (only valid in the top def).
    pub fn clock(&mut self, period: u32, high: u32, phase: u32) -> Sig {
        todo!()
    }
    /// PowerOnReset source (only valid in the top def).
    pub fn power_on_reset(&mut self, ticks: u32) -> Sig {
        todo!()
    }

    // ---- bus wiring (Splitter atoms) ----
    /// Bits `lo..lo+width` of `s`.
    pub fn slice(&mut self, s: Sig, lo: u32, width: u32) -> Sig {
        todo!()
    }
    pub fn bit(&mut self, s: Sig, i: u32) -> Sig {
        self.slice(s, i, 1)
    }
    /// All bits of `s`, LSB first (one Splitter).
    pub fn bits(&mut self, s: Sig) -> Vec<Sig> {
        todo!()
    }
    /// Concatenate, LSB first: `concat(&[lo, hi])`.
    pub fn concat(&mut self, parts: &[Sig]) -> Sig {
        todo!()
    }

    // ---- instances ----
    /// Instantiate a def. `inputs` are the def's `In` and `InOut` pins **in pin-declaration order**;
    /// returns the `Out` pins in declaration order. Widths must match.
    pub fn inst(&mut self, def: &DefRef, inputs: &[Sig]) -> Vec<Sig> {
        todo!()
    }
    /// Like [`Self::inst`] with an explicit instance id.
    pub fn inst_named(&mut self, id: &str, def: &DefRef, inputs: &[Sig]) -> Vec<Sig> {
        todo!()
    }
    /// Instantiate connecting pins by name. Every `In`/`InOut` pin must be given; `Out` pins may be
    /// given (then that net is driven by the pin) or omitted (fresh net created). Returns every pin's net.
    pub fn inst_map(&mut self, id: &str, def: &DefRef, conns: &[(&str, Sig)]) -> HashMap<String, Sig> {
        todo!()
    }

    pub fn finish(self) -> ComponentDef {
        todo!()
    }
}
