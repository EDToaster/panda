//! The inspector (right panel) for the selection:
//!
//! * **one atom**: gate op / inputs / width / reduce, NOT/BUF/TRI width, clock period / high /
//!   phase, constant value (with a radix selector), splitter parts (row editor with coverage
//!   hints), PowerOnReset ticks;
//! * **one instance** (atom or package): its id (rename), package pins, "Enter package";
//! * **one net** (wire segments of one net): name (rename), width, termination, connections,
//!   "Add to waveform";
//! * **several items / nets**: the fields every selected atom shares (shown as *mixed* when
//!   they differ) and a common termination for the selected nets, applied as **one** undoable
//!   action.
//!
//! Every edit is validated before it is applied ([`AtomDraft::to_atom`],
//! [`super::actions::plan_set_atom`], [`check_new_id`], [`plan_multi`]): errors and width
//! conflicts with connected nets are shown inline and disable "Apply" ("Apply & disconnect"
//! drops the mismatched connections). Changes go through the edit core
//! ([`Editor::set_atom`], [`Editor::set_termination`], [`History::commit`]); the batch and
//! rename actions below are thin wrappers over it ([`set_atoms`], [`rename_instance`],
//! [`rename_net`], [`set_terminations`]).
//!
//! [`History::commit`]: super::history::History::commit

use std::collections::BTreeMap;

use egui::{Color32, RichText, Ui};
use panda_kernel::EditOp;
use panda_netlist::{
    Atom, ComponentDef, Dir, Endpoint, GateOp, InstKind, Logic, Termination, Value,
};

use super::SharedEditor;
use super::actions::{Editor, WireSeg, atom_label, check_atom, ep_label, plan_set_atom};
use super::history::EditTarget;
use crate::app::{AppState, Panel, Slot};
use crate::doc::{Document, ItemRef};
use crate::simhost::SimEvent;

// =====================================================================================
// Values and radix
// =====================================================================================

/// Number base of the constant value field.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Radix {
    /// Hexadecimal (`2a`).
    #[default]
    Hex,
    /// Decimal (`42`).
    Dec,
    /// Binary (`101010`, may contain `x` / `z`).
    Bin,
}

impl Radix {
    /// Every radix, in menu order.
    pub const ALL: [Radix; 3] = [Radix::Hex, Radix::Dec, Radix::Bin];

    /// Short label (`hex`, `dec`, `bin`).
    pub fn label(self) -> &'static str {
        match self {
            Radix::Hex => "hex",
            Radix::Dec => "dec",
            Radix::Bin => "bin",
        }
    }
}

/// Text of value `v` in radix `r` without a prefix (`"2a"`, `"42"`, `"00101010"`).
///
/// Values with unknown bits can only be written in binary: for hex / decimal they come back
/// as `"x"` / `"z"` when uniform, otherwise as a `0b…` literal (which [`parse_value`] accepts
/// in any radix).
pub fn format_value(v: &Value, r: Radix) -> String {
    let w = v.width();
    let bin = || -> String { (0..w).rev().map(|i| v.get(i).to_char()).collect() };
    if r == Radix::Bin {
        return bin();
    }
    if !v.is_fully_known() {
        if v.bits().all(|b| b == Logic::X) {
            return "x".into();
        }
        if v.bits().all(|b| b == Logic::Z) {
            return "z".into();
        }
        return format!("0b{}", bin());
    }
    match r {
        Radix::Hex => {
            let digits = w.div_ceil(4);
            (0..digits)
                .rev()
                .map(|d| {
                    let n = (0..4)
                        .filter(|k| d * 4 + k < w && v.get(d * 4 + k) == Logic::One)
                        .fold(0u32, |n, k| n | (1 << k));
                    char::from_digit(n, 16).expect("nibble")
                })
                .collect()
        }
        Radix::Dec if w <= 128 => {
            let n = (0..w)
                .filter(|&i| v.get(i) == Logic::One)
                .fold(0u128, |n, i| n | (1 << i));
            n.to_string()
        }
        _ => format!("0b{}", bin()),
    }
}

/// Parse a constant value typed in radix `r`. A `0x` / `0b` prefix or a lone `x` / `z`
/// overrides the radix; `_` separators are ignored. Hex works for any width.
pub fn parse_value(width: u32, text: &str, r: Radix) -> Result<Value, String> {
    let t: String = text.trim().chars().filter(|&c| c != '_').collect();
    let lower = t.to_ascii_lowercase();
    if lower.is_empty() {
        return Err("enter a value".into());
    }
    let (radix, digits) = if let Some(h) = lower.strip_prefix("0x") {
        (Radix::Hex, h.to_string())
    } else if let Some(b) = lower.strip_prefix("0b") {
        (Radix::Bin, b.to_string())
    } else if lower == "x" || lower == "z" {
        return Value::parse(width, &lower);
    } else {
        (r, lower)
    };
    if digits.is_empty() {
        return Err(format!("'{t}' has no digits"));
    }
    match radix {
        Radix::Dec => Value::parse(width, &digits),
        Radix::Bin => {
            let bits = strip_zeros(&digits);
            if let Some(c) = bits.chars().find(|c| Logic::from_char(*c).is_none()) {
                return Err(format!("'{c}' is not a binary digit (0, 1, x, z)"));
            }
            if bits.len() as u32 > width {
                return Err(format!("{t} does not fit in {width} bit(s)"));
            }
            Value::parse(width, &format!("0b{bits}"))
        }
        Radix::Hex => {
            let mut bits = String::with_capacity(digits.len() * 4);
            for c in digits.chars() {
                let n = c
                    .to_digit(16)
                    .ok_or_else(|| format!("'{c}' is not a hex digit"))?;
                bits.push_str(&format!("{n:04b}"));
            }
            let bits = strip_zeros(&bits);
            if bits.len() as u32 > width {
                return Err(format!("0x{digits} does not fit in {width} bit(s)"));
            }
            Value::parse(width, &format!("0b{bits}"))
        }
    }
}

fn strip_zeros(s: &str) -> &str {
    let t = s.trim_start_matches('0');
    if t.is_empty() && !s.is_empty() {
        "0"
    } else {
        t
    }
}

/// `value` resized to `width` bits (truncated, or zero-extended).
pub fn resize_value(value: &Value, width: u32) -> Value {
    let mut v = Value::zeros(width);
    for i in 0..width.min(value.width()) {
        v.set(i, value.get(i));
    }
    v
}

/// Legacy one-line form of a constant value: `0`/`1`/`x`/`z` for one bit, `0x…` if fully
/// known, else `0b…`.
pub fn value_text(v: &Value) -> String {
    if v.width() == 1 {
        return v.get(0).to_char().to_string();
    }
    match format_value(v, Radix::Hex) {
        s if s.starts_with("0b") || s == "x" || s == "z" => s,
        s => format!("0x{s}"),
    }
}

// =====================================================================================
// Splitter parts
// =====================================================================================

/// Text form of splitter parts (`"7:4, 3:0"`; a one-bit part is just its index).
pub fn parts_text(parts: &[(u32, u32)]) -> String {
    parts
        .iter()
        .map(|&(lo, hi)| {
            if lo == hi {
                lo.to_string()
            } else {
                format!("{hi}:{lo}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Parse splitter parts (`"7:4, 3:0"`, `"0, 1, 2"`) into `(lo, hi)` pairs.
pub fn parse_parts(s: &str) -> Result<Vec<(u32, u32)>, String> {
    let mut out = Vec::new();
    for (i, p) in s.split(',').enumerate() {
        let p = p.trim();
        if p.is_empty() {
            continue;
        }
        let num = |t: &str| {
            t.trim()
                .parse::<u32>()
                .map_err(|_| format!("part {i}: '{}' is not a bit index", t.trim()))
        };
        let (lo, hi) = match p.split_once(':') {
            Some((a, b)) => {
                let (a, b) = (num(a)?, num(b)?);
                (a.min(b), a.max(b))
            }
            None => {
                let b = num(p)?;
                (b, b)
            }
        };
        out.push((lo, hi));
    }
    if out.is_empty() {
        return Err("give at least one part, e.g. 7:4, 3:0".into());
    }
    Ok(out)
}

/// `width` bits split into consecutive parts of `size` bits, lowest part first (the last part
/// takes the remainder).
pub fn even_parts(width: u32, size: u32) -> Vec<(u32, u32)> {
    let size = size.clamp(1, width.max(1));
    (0..width.div_ceil(size))
        .map(|k| (k * size, ((k + 1) * size).min(width) - 1))
        .collect()
}

/// Informational notes about splitter parts (not errors): bits of the whole that no part
/// covers, and bits covered by more than one part.
pub fn splitter_notes(width: u32, parts: &[(u32, u32)]) -> Vec<String> {
    let mut count = vec![0u32; width as usize];
    for &(lo, hi) in parts {
        for b in lo..=hi.min(width.saturating_sub(1)) {
            count[b as usize] += 1;
        }
    }
    let ranges = |pred: &dyn Fn(u32) -> bool| -> Vec<String> {
        let mut out = Vec::new();
        let mut b = 0;
        while b < width {
            if pred(count[b as usize]) {
                let lo = b;
                while b + 1 < width && pred(count[b as usize + 1]) {
                    b += 1;
                }
                out.push(if lo == b {
                    lo.to_string()
                } else {
                    format!("{b}:{lo}")
                });
            }
            b += 1;
        }
        out
    };
    let mut notes = Vec::new();
    let unused = ranges(&|c| c == 0);
    if !unused.is_empty() {
        notes.push(format!("bit(s) {} not in any part", unused.join(", ")));
    }
    let shared = ranges(&|c| c > 1);
    if !shared.is_empty() {
        notes.push(format!("bit(s) {} in several parts", shared.join(", ")));
    }
    notes
}

// =====================================================================================
// Atom drafts
// =====================================================================================

/// Kind of atom being edited.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtomKind {
    /// [`Atom::Gate`].
    Gate,
    /// [`Atom::Not`].
    Not,
    /// [`Atom::Buf`].
    Buf,
    /// [`Atom::Tri`].
    Tri,
    /// [`Atom::Splitter`].
    Splitter,
    /// [`Atom::Constant`].
    Constant,
    /// [`Atom::Clock`].
    Clock,
    /// [`Atom::PowerOnReset`].
    PowerOnReset,
}

impl AtomKind {
    /// Kind of an atom.
    pub fn of(a: &Atom) -> AtomKind {
        match a {
            Atom::Gate { .. } => AtomKind::Gate,
            Atom::Not { .. } => AtomKind::Not,
            Atom::Buf { .. } => AtomKind::Buf,
            Atom::Tri { .. } => AtomKind::Tri,
            Atom::Splitter { .. } => AtomKind::Splitter,
            Atom::Constant { .. } => AtomKind::Constant,
            Atom::Clock { .. } => AtomKind::Clock,
            Atom::PowerOnReset { .. } => AtomKind::PowerOnReset,
        }
    }
}

/// Gate operations in menu order.
pub const GATE_OPS: [GateOp; 6] = [
    GateOp::And,
    GateOp::Or,
    GateOp::Nand,
    GateOp::Nor,
    GateOp::Xor,
    GateOp::Xnor,
];

fn op_label(op: GateOp) -> String {
    format!("{op:?}").to_uppercase()
}

/// Editable form of an atom's parameters (text fields stay as typed until applied).
#[derive(Clone, Debug, PartialEq)]
pub struct AtomDraft {
    /// Atom kind (fixed).
    pub kind: AtomKind,
    /// Gate operation.
    pub op: GateOp,
    /// Gate inputs.
    pub inputs: u32,
    /// Width (gates, NOT/BUF/TRI, splitter whole, constant).
    pub width: u32,
    /// Gate reduce mode.
    pub reduce: bool,
    /// Splitter parts as `(lo, hi)`, part 0 first.
    pub parts: Vec<(u32, u32)>,
    /// Quick-entry text for the splitter parts (`"7:4, 3:0"`).
    pub parts_entry: String,
    /// Constant value as typed (in [`AtomDraft::radix`]).
    pub value: String,
    /// Radix of [`AtomDraft::value`].
    pub radix: Radix,
    /// Clock period (ticks).
    pub period: u32,
    /// Clock high time (ticks).
    pub high: u32,
    /// Clock phase (ticks).
    pub phase: u32,
    /// PowerOnReset duration (ticks).
    pub ticks: u32,
}

impl AtomDraft {
    /// Draft of an existing atom.
    pub fn from_atom(a: &Atom) -> AtomDraft {
        let mut d = AtomDraft {
            kind: AtomKind::of(a),
            op: GateOp::And,
            inputs: 2,
            width: 1,
            reduce: false,
            parts: Vec::new(),
            parts_entry: String::new(),
            value: String::new(),
            radix: Radix::Hex,
            period: 16,
            high: 8,
            phase: 0,
            ticks: 4,
        };
        match a {
            Atom::Gate {
                op,
                inputs,
                width,
                reduce,
            } => {
                d.op = *op;
                d.inputs = *inputs;
                d.width = *width;
                d.reduce = *reduce;
            }
            Atom::Not { width } | Atom::Buf { width } | Atom::Tri { width } => d.width = *width,
            Atom::Splitter { width, parts } => {
                d.width = *width;
                d.parts = parts.clone();
                d.parts_entry = parts_text(parts);
            }
            Atom::Constant { value } => {
                d.width = value.width();
                d.radix = if value.width() <= 4 || !value.is_fully_known() {
                    Radix::Bin
                } else {
                    Radix::Hex
                };
                d.value = format_value(value, d.radix);
            }
            Atom::Clock {
                period,
                high,
                phase,
            } => {
                d.period = *period;
                d.high = *high;
                d.phase = *phase;
            }
            Atom::PowerOnReset { ticks } => d.ticks = *ticks,
        }
        d
    }

    /// Switch the constant value's radix, converting the typed text if it parses.
    pub fn set_radix(&mut self, r: Radix) {
        if let Ok(v) = parse_value(self.width.max(1), &self.value, self.radix) {
            self.value = format_value(&v, r);
        }
        self.radix = r;
    }

    /// The atom described by the draft, or why it is invalid.
    pub fn to_atom(&self) -> Result<Atom, String> {
        if self.width == 0 {
            return Err("width must be ≥ 1".into());
        }
        let a = match self.kind {
            AtomKind::Gate => Atom::Gate {
                op: self.op,
                inputs: if self.reduce { 1 } else { self.inputs },
                width: self.width,
                reduce: self.reduce,
            },
            AtomKind::Not => Atom::Not { width: self.width },
            AtomKind::Buf => Atom::Buf { width: self.width },
            AtomKind::Tri => Atom::Tri { width: self.width },
            AtomKind::Splitter => Atom::Splitter {
                width: self.width,
                parts: self.parts.clone(),
            },
            AtomKind::Constant => Atom::Constant {
                value: parse_value(self.width, &self.value, self.radix)?,
            },
            AtomKind::Clock => Atom::Clock {
                period: self.period,
                high: self.high,
                phase: self.phase,
            },
            AtomKind::PowerOnReset => Atom::PowerOnReset { ticks: self.ticks },
        };
        check_atom(&a)?;
        Ok(a)
    }

    /// Draw the parameter fields. Returns true if Enter was pressed in a text field.
    fn ui(&mut self, ui: &mut Ui) -> bool {
        let mut enter = false;
        let enter_in = |ui: &Ui, r: &egui::Response| {
            r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))
        };
        egui::Grid::new("panda_edit_atom")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| match self.kind {
                AtomKind::Gate => {
                    ui.label("operation");
                    egui::ComboBox::from_id_salt("panda_edit_gate_op")
                        .selected_text(op_label(self.op))
                        .show_ui(ui, |ui| {
                            for op in GATE_OPS {
                                ui.selectable_value(&mut self.op, op, op_label(op));
                            }
                        });
                    ui.end_row();
                    ui.label("inputs");
                    ui.add_enabled(
                        !self.reduce,
                        egui::DragValue::new(&mut self.inputs).range(1..=64),
                    );
                    ui.end_row();
                    ui.label("width");
                    ui.add(egui::DragValue::new(&mut self.width).range(1..=1024));
                    ui.end_row();
                    ui.label("reduce");
                    if ui
                        .checkbox(&mut self.reduce, "across bits")
                        .on_hover_text("one bus input, 1-bit output")
                        .changed()
                        && self.reduce
                    {
                        self.inputs = 1;
                    }
                    ui.end_row();
                }
                AtomKind::Not | AtomKind::Buf | AtomKind::Tri => {
                    ui.label("width");
                    ui.add(egui::DragValue::new(&mut self.width).range(1..=1024));
                    ui.end_row();
                }
                AtomKind::Splitter => {
                    ui.label("width");
                    ui.add(egui::DragValue::new(&mut self.width).range(1..=1024));
                    ui.end_row();
                    ui.label("quick entry");
                    let r = ui.add(
                        egui::TextEdit::singleline(&mut self.parts_entry)
                            .hint_text("7:4, 3:0")
                            .desired_width(110.0),
                    );
                    if r.changed()
                        && let Ok(p) = parse_parts(&self.parts_entry)
                    {
                        self.parts = p;
                    }
                    enter |= enter_in(ui, &r);
                    ui.end_row();
                }
                AtomKind::Constant => {
                    ui.label("width");
                    ui.add(egui::DragValue::new(&mut self.width).range(1..=1024));
                    ui.end_row();
                    ui.label("radix");
                    ui.horizontal(|ui| {
                        for r in Radix::ALL {
                            if ui.selectable_label(self.radix == r, r.label()).clicked() {
                                self.set_radix(r);
                            }
                        }
                    });
                    ui.end_row();
                    ui.label("value");
                    let r = ui.add(
                        egui::TextEdit::singleline(&mut self.value)
                            .hint_text(match self.radix {
                                Radix::Hex => "2a",
                                Radix::Dec => "42",
                                Radix::Bin => "1010",
                            })
                            .font(egui::TextStyle::Monospace)
                            .desired_width(120.0),
                    );
                    enter |= enter_in(ui, &r);
                    ui.end_row();
                }
                AtomKind::Clock => {
                    ui.label("period");
                    ui.add(egui::DragValue::new(&mut self.period).range(2..=1_000_000));
                    ui.end_row();
                    ui.label("high");
                    ui.add(egui::DragValue::new(&mut self.high).range(1..=1_000_000));
                    ui.end_row();
                    ui.label("phase");
                    ui.add(egui::DragValue::new(&mut self.phase).range(0..=1_000_000));
                    ui.end_row();
                }
                AtomKind::PowerOnReset => {
                    ui.label("ticks");
                    ui.add(egui::DragValue::new(&mut self.ticks).range(0..=1_000_000));
                    ui.end_row();
                }
            });
        if self.kind == AtomKind::Splitter {
            self.parts_ui(ui);
        }
        enter
    }

    /// The splitter part rows: `p<i>  hi : lo  [up] [remove]`, plus "add" and "split evenly".
    fn parts_ui(&mut self, ui: &mut Ui) {
        let before = self.parts.clone();
        let max = self.width.saturating_sub(1).max(1023);
        let mut remove = None;
        let mut up = None;
        ui.add_space(4.0);
        ui.label(RichText::new("parts (hi : lo)").weak());
        egui::Grid::new("panda_edit_parts")
            .num_columns(4)
            .spacing([6.0, 2.0])
            .show(ui, |ui| {
                let n = self.parts.len();
                for (i, (lo, hi)) in self.parts.iter_mut().enumerate() {
                    ui.label(format!("p{i}"));
                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(hi).range(0..=max));
                        ui.label(":");
                        ui.add(egui::DragValue::new(lo).range(0..=max));
                    });
                    ui.label(RichText::new(format!("{}b", hi.saturating_sub(*lo) + 1)).weak());
                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(i > 0, egui::Button::new("⬆").small())
                            .on_hover_text("move up")
                            .clicked()
                        {
                            up = Some(i);
                        }
                        if ui
                            .add_enabled(n > 1, egui::Button::new("✖").small())
                            .on_hover_text("remove part")
                            .clicked()
                        {
                            remove = Some(i);
                        }
                    });
                    ui.end_row();
                }
            });
        if let Some(i) = up {
            self.parts.swap(i, i - 1);
        }
        if let Some(i) = remove {
            self.parts.remove(i);
        }
        ui.horizontal(|ui| {
            if ui.small_button("+ part").clicked() {
                let next = self
                    .parts
                    .iter()
                    .map(|&(_, hi)| hi + 1)
                    .max()
                    .unwrap_or(0)
                    .min(self.width.saturating_sub(1));
                self.parts.push((next, next));
            }
            ui.menu_button(RichText::new("split evenly").small(), |ui| {
                for size in [1, 2, 4, 8, 16] {
                    if size <= self.width && ui.button(format!("{size}-bit parts")).clicked() {
                        self.parts = even_parts(self.width, size);
                        ui.close();
                    }
                }
            });
        });
        for note in splitter_notes(self.width, &self.parts) {
            ui.label(RichText::new(format!("ⓘ {note}")).small().weak());
        }
        if self.parts != before {
            self.parts_entry = parts_text(&self.parts);
        }
    }
}

// =====================================================================================
// Common fields of a multi-selection
// =====================================================================================

/// A parameter that several atoms can share.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Field {
    /// Gate operation.
    Op,
    /// Gate inputs.
    Inputs,
    /// Width (gates, NOT/BUF/TRI, splitter whole, constant: resized).
    Width,
    /// Gate reduce mode.
    Reduce,
    /// Clock period.
    Period,
    /// Clock high time.
    High,
    /// Clock phase.
    Phase,
    /// PowerOnReset ticks.
    Ticks,
}

impl Field {
    /// Every field in display order.
    pub const ALL: [Field; 8] = [
        Field::Op,
        Field::Inputs,
        Field::Width,
        Field::Reduce,
        Field::Period,
        Field::High,
        Field::Phase,
        Field::Ticks,
    ];

    /// Display name.
    pub fn label(self) -> &'static str {
        match self {
            Field::Op => "operation",
            Field::Inputs => "inputs",
            Field::Width => "width",
            Field::Reduce => "reduce",
            Field::Period => "period",
            Field::High => "high",
            Field::Phase => "phase",
            Field::Ticks => "ticks",
        }
    }
}

/// Value of a [`Field`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldValue {
    /// A number.
    Num(u32),
    /// A gate operation.
    Op(GateOp),
    /// A flag.
    Flag(bool),
}

/// Value of field `f` of atom `a`, if the atom has it.
pub fn field(a: &Atom, f: Field) -> Option<FieldValue> {
    use FieldValue::{Flag, Num, Op};
    match (a, f) {
        (Atom::Gate { op, .. }, Field::Op) => Some(Op(*op)),
        (Atom::Gate { inputs, .. }, Field::Inputs) => Some(Num(*inputs)),
        (Atom::Gate { reduce, .. }, Field::Reduce) => Some(Flag(*reduce)),
        (
            Atom::Gate { width, .. }
            | Atom::Not { width }
            | Atom::Buf { width }
            | Atom::Tri { width }
            | Atom::Splitter { width, .. },
            Field::Width,
        ) => Some(Num(*width)),
        (Atom::Constant { value }, Field::Width) => Some(Num(value.width())),
        (Atom::Clock { period, .. }, Field::Period) => Some(Num(*period)),
        (Atom::Clock { high, .. }, Field::High) => Some(Num(*high)),
        (Atom::Clock { phase, .. }, Field::Phase) => Some(Num(*phase)),
        (Atom::PowerOnReset { ticks }, Field::Ticks) => Some(Num(*ticks)),
        _ => None,
    }
}

/// `a` with field `f` set to `v` (`None` if the atom has no such field or the value has the
/// wrong type). Setting a gate's `reduce` also sets its inputs to 1.
pub fn with_field(a: &Atom, f: Field, v: FieldValue) -> Option<Atom> {
    use FieldValue::{Flag, Num, Op};
    let mut a = a.clone();
    match (&mut a, f, v) {
        (Atom::Gate { op, .. }, Field::Op, Op(x)) => *op = x,
        (Atom::Gate { inputs, .. }, Field::Inputs, Num(x)) => *inputs = x,
        (Atom::Gate { reduce, inputs, .. }, Field::Reduce, Flag(x)) => {
            *reduce = x;
            if x {
                *inputs = 1;
            }
        }
        (
            Atom::Gate { width, .. }
            | Atom::Not { width }
            | Atom::Buf { width }
            | Atom::Tri { width }
            | Atom::Splitter { width, .. },
            Field::Width,
            Num(x),
        ) => *width = x,
        (Atom::Constant { value }, Field::Width, Num(x)) => *value = resize_value(value, x),
        (Atom::Clock { period, .. }, Field::Period, Num(x)) => *period = x,
        (Atom::Clock { high, .. }, Field::High, Num(x)) => *high = x,
        (Atom::Clock { phase, .. }, Field::Phase, Num(x)) => *phase = x,
        (Atom::PowerOnReset { ticks }, Field::Ticks, Num(x)) => *ticks = x,
        _ => return None,
    }
    Some(a)
}

/// The fields every atom of `atoms` has, with their common value (`None`: values differ).
pub fn common_fields(atoms: &[&Atom]) -> Vec<(Field, Option<FieldValue>)> {
    if atoms.is_empty() {
        return Vec::new();
    }
    Field::ALL
        .into_iter()
        .filter_map(|f| {
            let vals: Vec<FieldValue> = atoms.iter().map(|a| field(a, f)).collect::<Option<_>>()?;
            let first = vals[0];
            Some((f, vals.iter().all(|v| *v == first).then_some(first)))
        })
        .collect()
}

/// What applying common-field changes to several atoms would do (see [`plan_multi`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MultiPlan {
    /// `(instance, new atom)` for every atom that changes.
    pub changes: Vec<(String, Atom)>,
    /// Invalid parameters (`"g1: high must be …"`); nothing can be applied.
    pub errors: Vec<String>,
    /// Width conflicts with connected nets ("Apply & disconnect" drops those connections).
    pub conflicts: Vec<String>,
    /// Connections that will be dropped because their port disappears.
    pub vanished: Vec<String>,
}

/// Plan setting fields `pending` on atom instances `insts` of `def_id`.
pub fn plan_multi(
    doc: &Document,
    def_id: &str,
    insts: &[String],
    pending: &BTreeMap<Field, FieldValue>,
) -> MultiPlan {
    let mut plan = MultiPlan::default();
    let Some(def) = doc.def(def_id) else {
        plan.errors.push(format!("unknown def '{def_id}'"));
        return plan;
    };
    for id in insts {
        let Some(InstKind::Atom(old)) = def.instances.iter().find(|i| &i.id == id).map(|i| &i.kind)
        else {
            continue;
        };
        let mut new = old.clone();
        for (&f, &v) in pending {
            if let Some(a) = with_field(&new, f, v) {
                new = a;
            }
        }
        if &new == old {
            continue;
        }
        match plan_set_atom(doc, def_id, id, &new) {
            Err(e) => plan.errors.push(format!("{id}: {e}")),
            Ok(p) => {
                plan.conflicts.extend(p.conflicts);
                plan.vanished.extend(
                    p.vanished
                        .iter()
                        .map(|(n, e)| format!("{} from '{n}'", ep_label(e))),
                );
                plan.changes.push((id.clone(), new));
            }
        }
    }
    plan
}

// =====================================================================================
// Actions (wrappers over the edit core)
// =====================================================================================

/// Set several atoms at once as **one** undoable action: the edit core's
/// [`Editor::set_atom`] runs for each change on a scratch copy of the document, and the
/// resulting def (structure + geometry) is committed with one `ReplaceDef`.
pub fn set_atoms(
    ed: &mut Editor,
    t: &mut dyn EditTarget,
    def_id: &str,
    changes: &[(String, Atom)],
    disconnect_mismatched: bool,
) -> Result<(), String> {
    let r = (|| {
        match changes {
            [] => return Ok(()),
            [(inst, atom)] => return ed.set_atom(t, def_id, inst, atom, disconnect_mismatched),
            _ => {}
        }
        let mut scratch = t.doc().clone();
        let mut se = Editor::new();
        for (inst, atom) in changes {
            se.set_atom(&mut scratch, def_id, inst, atom, disconnect_mismatched)
                .map_err(|e| format!("{inst}: {e}"))?;
        }
        let def = scratch
            .def(def_id)
            .cloned()
            .ok_or_else(|| format!("unknown def '{def_id}'"))?;
        let geom = scratch.editors.remove(def_id);
        let label = format!("edit {} atoms", changes.len());
        let id = def_id.to_string();
        let defs = [id.clone()];
        ed.history
            .commit(t, label, &defs, vec![EditOp::ReplaceDef(def)], move |doc| {
                if let Some(g) = geom {
                    doc.editors.insert(id, g);
                }
            })
    })();
    report(ed, r)
}

/// Set the termination of several nets as one undoable action.
pub fn set_terminations(
    ed: &mut Editor,
    t: &mut dyn EditTarget,
    def_id: &str,
    nets: &[String],
    termination: Termination,
) -> Result<(), String> {
    let ops: Vec<EditOp> = nets
        .iter()
        .filter(|n| {
            t.doc()
                .def(def_id)
                .and_then(|d| d.nets.iter().find(|x| &x.id == *n))
                .is_some_and(|x| x.termination != termination)
        })
        .map(|n| EditOp::SetTermination {
            def: def_id.to_string(),
            net: n.clone(),
            termination,
        })
        .collect();
    if ops.is_empty() {
        return Ok(());
    }
    let label = match nets {
        [one] => format!("termination of {one}"),
        _ => format!("termination of {} nets", ops.len()),
    };
    let r = ed
        .history
        .commit(t, label, &[def_id.to_string()], ops, |_| {});
    report(ed, r)
}

/// Kind of id being changed (see [`check_new_id`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdKind {
    /// An instance id.
    Instance,
    /// A net id.
    Net,
}

/// Check that `new` can replace id `old` in `def` (the validator's rules: not empty, no `/`,
/// `[` or `]`, unique within the def; `pin` is reserved for instances).
pub fn check_new_id(def: &ComponentDef, kind: IdKind, old: &str, new: &str) -> Result<(), String> {
    if new.is_empty() {
        return Err("the name cannot be empty".into());
    }
    if new.chars().any(char::is_whitespace) {
        return Err("the name cannot contain spaces".into());
    }
    if let Some(c) = new.chars().find(|c| matches!(c, '/' | '[' | ']')) {
        return Err(format!("the name cannot contain '{c}'"));
    }
    if new == old {
        return Ok(());
    }
    let taken = match kind {
        IdKind::Instance => {
            if new == "pin" {
                return Err("'pin' is reserved".into());
            }
            def.instances.iter().any(|i| i.id == new)
        }
        IdKind::Net => def.nets.iter().any(|n| n.id == new),
    };
    if taken {
        return Err(format!("'{new}' is already used in {}", def.id));
    }
    Ok(())
}

/// Rename instance `old` of `def_id` to `new` (connections, geometry and selection follow).
pub fn rename_instance(
    ed: &mut Editor,
    t: &mut dyn EditTarget,
    def_id: &str,
    old: &str,
    new: &str,
) -> Result<(), String> {
    let r = (|| {
        let def = t
            .doc()
            .def(def_id)
            .ok_or_else(|| format!("unknown def '{def_id}'"))?;
        check_new_id(def, IdKind::Instance, old, new)?;
        if old == new {
            return Ok(());
        }
        let mut d = def.clone();
        let inst = d
            .instances
            .iter_mut()
            .find(|i| i.id == old)
            .ok_or_else(|| format!("no instance '{old}'"))?;
        inst.id = new.to_string();
        for n in &mut d.nets {
            for e in &mut n.ends {
                if let Endpoint::Port { inst, .. } = e
                    && inst == old
                {
                    *inst = new.to_string();
                }
            }
        }
        let mut geom = t.doc().editor(def_id).cloned();
        if let Some(g) = &mut geom
            && let Some(ig) = g.instances.remove(old)
        {
            g.instances.insert(new.to_string(), ig);
        }
        let id = def_id.to_string();
        let defs = [id.clone()];
        ed.history.commit(
            t,
            format!("rename {old} to {new}"),
            &defs,
            vec![EditOp::ReplaceDef(d)],
            move |doc| {
                if let Some(g) = geom {
                    doc.editors.insert(id, g);
                }
            },
        )?;
        let (o, n) = (ItemRef::Instance(old.into()), ItemRef::Instance(new.into()));
        if ed.selection.items.remove(&o) {
            ed.selection.items.insert(n);
        }
        Ok(())
    })();
    report(ed, r)
}

/// Rename net `old` of `def_id` to `new` (its wires and selected segments follow).
pub fn rename_net(
    ed: &mut Editor,
    t: &mut dyn EditTarget,
    def_id: &str,
    old: &str,
    new: &str,
) -> Result<(), String> {
    let r = (|| {
        let def = t
            .doc()
            .def(def_id)
            .ok_or_else(|| format!("unknown def '{def_id}'"))?;
        check_new_id(def, IdKind::Net, old, new)?;
        if old == new {
            return Ok(());
        }
        let mut d = def.clone();
        d.nets
            .iter_mut()
            .find(|n| n.id == old)
            .ok_or_else(|| format!("no net '{old}'"))?
            .id = new.to_string();
        let mut geom = t.doc().editor(def_id).cloned();
        if let Some(g) = &mut geom
            && let Some(w) = g.wires.remove(old)
        {
            g.wires.insert(new.to_string(), w);
        }
        let id = def_id.to_string();
        let defs = [id.clone()];
        ed.history.commit(
            t,
            format!("rename {old} to {new}"),
            &defs,
            vec![EditOp::ReplaceDef(d)],
            move |doc| {
                if let Some(g) = geom {
                    doc.editors.insert(id, g);
                }
            },
        )?;
        ed.selection.wires = std::mem::take(&mut ed.selection.wires)
            .into_iter()
            .map(|w| {
                if w.net == old {
                    WireSeg {
                        net: new.into(),
                        ..w
                    }
                } else {
                    w
                }
            })
            .collect();
        Ok(())
    })();
    report(ed, r)
}

/// Record the outcome in the editor's message line (as the edit core's actions do).
fn report(ed: &mut Editor, r: Result<(), String>) -> Result<(), String> {
    match &r {
        Ok(()) => ed.message = None,
        Err(e) => ed.message = Some(e.clone()),
    }
    r
}

// =====================================================================================
// The panel
// =====================================================================================

/// Colour of inline errors.
pub const ERROR: Color32 = Color32::from_rgb(230, 80, 70);
/// Colour of inline warnings (conflicts that "Apply & disconnect" resolves).
pub const WARN: Color32 = Color32::from_rgb(225, 165, 50);

/// Key of a draft: `(def, selection key, document revision)`; a draft is reloaded when the
/// selection or the document changes.
type DraftKey = (String, String, u64);

/// The inspector panel.
pub struct Inspector {
    ed: SharedEditor,
    atom: Option<(DraftKey, AtomDraft)>,
    /// Rename field: key and text.
    name: Option<(DraftKey, String)>,
    /// Multi-selection pending field values.
    multi: Option<(DraftKey, BTreeMap<Field, FieldValue>)>,
    /// The selected part of the def ([`selected_part`]) for `(def, revision, selection)`,
    /// rebuilt only when one of them changes, and a counter bumped on every rebuild (keys the
    /// multi-selection draft).
    part: Option<(String, u64, super::actions::Selection, ComponentDef)>,
    part_gen: u64,
}

impl Inspector {
    /// Inspector over the shared editor.
    pub fn new(ed: SharedEditor) -> Inspector {
        Inspector {
            ed,
            atom: None,
            name: None,
            multi: None,
            part: None,
            part_gen: 0,
        }
    }
}

/// Display name of a termination.
pub fn term_label(t: Termination) -> &'static str {
    match t {
        Termination::None => "none",
        Termination::PullUp => "pull-up",
        Termination::PullDown => "pull-down",
        Termination::Keeper => "keeper",
    }
}

/// Every termination in menu order.
pub const TERMINATIONS: [Termination; 4] = [
    Termination::None,
    Termination::PullUp,
    Termination::PullDown,
    Termination::Keeper,
];

fn dir_label(d: Dir) -> &'static str {
    match d {
        Dir::In => "input",
        Dir::Out => "output",
        Dir::InOut => "in/out",
    }
}

/// Termination combo box; `None` current value shows "mixed". Returns the chosen value.
fn term_combo(ui: &mut Ui, salt: &str, cur: Option<Termination>) -> Option<Termination> {
    let mut chosen = None;
    egui::ComboBox::from_id_salt(salt)
        .selected_text(cur.map_or("mixed", term_label))
        .show_ui(ui, |ui| {
            for t in TERMINATIONS {
                if ui.selectable_label(cur == Some(t), term_label(t)).clicked() {
                    chosen = Some(t);
                }
            }
        });
    chosen.filter(|t| Some(*t) != cur)
}

/// Waveform button for a probe (pin or net) through the sim UX hook.
fn wave_button(ui: &mut Ui, app: &mut AppState, target: &super::menus::Target) {
    if let Some(name) = super::menus::wave_name(app, target) {
        match super::menus::wave_index(app, &name) {
            None => {
                if ui.button("Add to waveform").clicked() {
                    super::menus::add_to_waveform(app, &name);
                }
            }
            Some(i) => {
                if ui.button("Remove from waveform").clicked() {
                    super::menus::remove_from_waveform(app, i);
                }
            }
        }
    }
}

impl Inspector {
    /// Editable id: a text field with inline validation, applied on Enter or "Rename".
    fn name_ui(
        &mut self,
        ui: &mut Ui,
        ed: &mut Editor,
        app: &mut AppState,
        kind: IdKind,
        id: &str,
    ) {
        let def_id = app.view.def.clone();
        let key = (def_id.clone(), format!("{kind:?}:{id}"), app.doc.revision);
        if self.name.as_ref().is_none_or(|(k, _)| *k != key) {
            self.name = Some((key, id.to_string()));
        }
        let (_, text) = self.name.as_mut().expect("set");
        let check = app.doc.def(&def_id).map_or(Err("no def".to_string()), |d| {
            check_new_id(d, kind, id, text.trim())
        });
        let changed = text.trim() != id;
        let mut apply = false;
        ui.horizontal(|ui| {
            ui.label(match kind {
                IdKind::Instance => "id",
                IdKind::Net => "name",
            });
            let r = ui.add(
                egui::TextEdit::singleline(text)
                    .desired_width(120.0)
                    .font(egui::TextStyle::Monospace),
            );
            apply = r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if changed
                && ui
                    .add_enabled(check.is_ok(), egui::Button::new("Rename"))
                    .clicked()
            {
                apply = true;
            }
        });
        if changed && let Err(e) = &check {
            ui.colored_label(ERROR, format!("✖ {e}"));
        }
        let apply = apply && changed && check.is_ok();
        if apply {
            let new = text.trim().to_string();
            let _ = match kind {
                IdKind::Instance => rename_instance(ed, app, &def_id, id, &new),
                IdKind::Net => rename_net(ed, app, &def_id, id, &new),
            };
        }
    }

    fn atom_ui(
        &mut self,
        ui: &mut Ui,
        ed: &mut Editor,
        app: &mut AppState,
        inst: &str,
        atom: &Atom,
    ) {
        let def_id = app.view.def.clone();
        let key = (def_id.clone(), inst.to_string(), app.doc.revision);
        if self.atom.as_ref().is_none_or(|(k, _)| *k != key) {
            self.atom = Some((key, AtomDraft::from_atom(atom)));
        }
        let (_, draft) = self.atom.as_mut().expect("set");
        let enter = draft.ui(ui);
        ui.add_space(6.0);
        let checked = draft
            .to_atom()
            .and_then(|new| plan_set_atom(&app.doc, &def_id, inst, &new).map(|p| (new, p)));
        let (new, plan) = match checked {
            Err(e) => {
                ui.colored_label(ERROR, format!("✖ {e}"));
                if ui.button("Revert").clicked() {
                    self.atom = None;
                }
                return;
            }
            Ok((new, _)) if &new == atom => return,
            Ok(x) => x,
        };
        for c in &plan.conflicts {
            ui.colored_label(WARN, format!("⚠ {c}"));
        }
        for (net, ep) in &plan.vanished {
            ui.label(
                RichText::new(format!(
                    "{} will be disconnected from '{net}'",
                    ep_label(ep)
                ))
                .weak(),
            );
        }
        let mut revert = false;
        ui.horizontal(|ui| {
            let ok = plan.mismatched.is_empty();
            if ui.add_enabled(ok, egui::Button::new("Apply")).clicked() || (ok && enter) {
                let _ = ed.set_atom(app, &def_id, inst, &new, false);
            }
            if !ok && ui.button("Apply & disconnect").clicked() {
                let _ = ed.set_atom(app, &def_id, inst, &new, true);
            }
            revert = ui.button("Revert").clicked();
        });
        if revert {
            self.atom = None;
        }
    }

    fn net_ui(
        &mut self,
        ui: &mut Ui,
        ed: &mut Editor,
        app: &mut AppState,
        def: &ComponentDef,
        net_id: &str,
    ) {
        let Some(net) = def.nets.iter().find(|n| n.id == net_id) else {
            return;
        };
        let def_id = def.id.clone();
        ui.heading(format!("net {net_id}"));
        ui.label(
            RichText::new(format!(
                "{} bit(s) · {} connection(s)",
                net.width,
                net.ends.len()
            ))
            .weak(),
        );
        ui.separator();
        self.name_ui(ui, ed, app, IdKind::Net, net_id);
        ui.horizontal(|ui| {
            ui.label("termination");
            if let Some(t) = term_combo(ui, "panda_edit_term", Some(net.termination)) {
                let _ = ed.set_termination(app, &def_id, net_id, t);
            }
        });
        wave_button(ui, app, &super::menus::Target::Net(net_id.to_string()));
        ui.add_space(4.0);
        ui.label(RichText::new("connected to").weak());
        for e in &net.ends {
            ui.label(RichText::new(format!("  {}", ep_label(e))).monospace());
        }
    }

    fn multi_ui(&mut self, ui: &mut Ui, ed: &mut Editor, app: &mut AppState, def: &ComponentDef) {
        let sel = ed.selection.clone();
        let def_id = def.id.clone();
        let atoms: Vec<(String, &Atom)> = sel
            .items
            .iter()
            .filter_map(|it| match it {
                ItemRef::Instance(id) => def.instances.iter().find(|i| &i.id == id),
                ItemRef::Pin(_) => None,
            })
            .filter_map(|i| match &i.kind {
                InstKind::Atom(a) => Some((i.id.clone(), a)),
                InstKind::Def(_) => None,
            })
            .collect();
        let packages = sel
            .items
            .iter()
            .filter(|it| {
                matches!(it, ItemRef::Instance(id)
                    if def.instances.iter().any(|i| &i.id == id && matches!(i.kind, InstKind::Def(_))))
            })
            .count();
        let pins = sel
            .items
            .iter()
            .filter(|it| matches!(it, ItemRef::Pin(_)))
            .count();
        let mut nets: Vec<String> = sel.wires.iter().map(|w| w.net.clone()).collect();
        nets.dedup();
        ui.heading(format!("{} selected", sel.items.len() + nets.len()));
        let mut parts = Vec::new();
        if !atoms.is_empty() {
            parts.push(format!("{} atom(s)", atoms.len()));
        }
        if packages > 0 {
            parts.push(format!("{packages} package(s)"));
        }
        if pins > 0 {
            parts.push(format!("{pins} pin(s)"));
        }
        if !nets.is_empty() {
            parts.push(format!(
                "{} net(s), {} segment(s)",
                nets.len(),
                sel.wires.len()
            ));
        }
        ui.label(RichText::new(parts.join(" · ")).weak());
        ui.separator();

        // Common atom fields.
        let refs: Vec<&Atom> = atoms.iter().map(|(_, a)| *a).collect();
        let common = common_fields(&refs);
        let mut kinds: Vec<String> = refs.iter().map(|a| atom_label(a)).collect();
        kinds.sort();
        kinds.dedup();
        if !refs.is_empty() {
            ui.label(RichText::new(format!("common fields of {}", kinds.join(", "))).weak());
        }
        let key = (
            def_id.clone(),
            format!("sel{}", self.part_gen),
            app.doc.revision,
        );
        if self.multi.as_ref().is_none_or(|(k, _)| *k != key) {
            self.multi = Some((key, BTreeMap::new()));
        }
        let (_, pending) = self.multi.as_mut().expect("set");
        if !refs.is_empty() && common.is_empty() {
            ui.label(RichText::new("(no shared parameters)").weak().italics());
        }
        egui::Grid::new("panda_edit_multi")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                for (f, cur) in &common {
                    let shown = pending.get(f).copied().or(*cur);
                    let edited = pending.contains_key(f);
                    ui.label(if edited {
                        RichText::new(f.label()).strong()
                    } else {
                        RichText::new(f.label())
                    });
                    match shown {
                        Some(FieldValue::Op(op)) => {
                            let mut v = op;
                            egui::ComboBox::from_id_salt("panda_edit_multi_op")
                                .selected_text(op_label(v))
                                .show_ui(ui, |ui| {
                                    for o in GATE_OPS {
                                        ui.selectable_value(&mut v, o, op_label(o));
                                    }
                                });
                            if v != op {
                                pending.insert(*f, FieldValue::Op(v));
                            }
                        }
                        Some(FieldValue::Flag(b)) => {
                            let mut v = b;
                            if ui.checkbox(&mut v, "").changed() {
                                pending.insert(*f, FieldValue::Flag(v));
                            }
                        }
                        Some(FieldValue::Num(n)) => {
                            let mut v = n;
                            if ui
                                .add(egui::DragValue::new(&mut v).range(0..=1_000_000))
                                .changed()
                            {
                                pending.insert(*f, FieldValue::Num(v));
                            }
                        }
                        None => {
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("mixed").italics().weak());
                                if ui.small_button("set…").clicked() {
                                    let first = refs.iter().find_map(|a| field(a, *f));
                                    if let Some(v) = first {
                                        pending.insert(*f, v);
                                    }
                                }
                            });
                        }
                    }
                    ui.end_row();
                }
            });
        if !pending.is_empty() {
            let insts: Vec<String> = atoms.iter().map(|(i, _)| i.clone()).collect();
            let plan = plan_multi(&app.doc, &def_id, &insts, pending);
            for e in &plan.errors {
                ui.colored_label(ERROR, format!("✖ {e}"));
            }
            for c in &plan.conflicts {
                ui.colored_label(WARN, format!("⚠ {c}"));
            }
            for v in &plan.vanished {
                ui.label(RichText::new(format!("{v} will be disconnected")).weak());
            }
            let mut clear = false;
            ui.horizontal(|ui| {
                let ok = plan.errors.is_empty() && !plan.changes.is_empty();
                let clean = plan.conflicts.is_empty();
                let label = format!("Apply to {}", plan.changes.len());
                if ui
                    .add_enabled(ok && clean, egui::Button::new(label))
                    .clicked()
                {
                    clear = set_atoms(ed, app, &def_id, &plan.changes, false).is_ok();
                }
                if ok && !clean && ui.button("Apply & disconnect").clicked() {
                    clear = set_atoms(ed, app, &def_id, &plan.changes, true).is_ok();
                }
                if ui.button("Revert").clicked() {
                    clear = true;
                }
            });
            if clear {
                self.multi = None;
            }
        }

        // Common termination of the selected nets.
        if !nets.is_empty() {
            ui.add_space(6.0);
            let terms: Vec<Termination> = nets
                .iter()
                .filter_map(|n| def.nets.iter().find(|x| &x.id == n))
                .map(|n| n.termination)
                .collect();
            let cur = terms
                .first()
                .copied()
                .filter(|t| terms.iter().all(|x| x == t));
            ui.horizontal(|ui| {
                ui.label("termination");
                if let Some(t) = term_combo(ui, "panda_edit_multi_term", cur) {
                    let _ = set_terminations(ed, app, &def_id, &nets, t);
                }
            });
        }
    }
}

/// The part of `def` the inspector needs for selection `sel`: its pins, the selected
/// instances, and the nets of selected wires or touching selected pins.
pub fn selected_part(def: &ComponentDef, sel: &super::actions::Selection) -> ComponentDef {
    let inst = |id: &str| sel.items.contains(&ItemRef::Instance(id.to_string()));
    ComponentDef {
        id: def.id.clone(),
        name: def.name.clone(),
        version: def.version,
        pins: def.pins.clone(),
        instances: def.instances.iter().filter(|i| inst(&i.id)).cloned().collect(),
        nets: def
            .nets
            .iter()
            .filter(|n| {
                sel.wires.iter().any(|w| w.net == n.id)
                    || n.ends.iter().any(|e| {
                        matches!(e, Endpoint::Pin(p) if sel.items.contains(&ItemRef::Pin(p.clone())))
                    })
            })
            .cloned()
            .collect(),
        editor: serde_json::Value::Null,
    }
}

impl Panel for Inspector {
    fn id(&self) -> &'static str {
        "edit.inspector"
    }

    fn title(&self) -> String {
        "Inspector".into()
    }

    fn slot(&self) -> Slot {
        Slot::Right
    }

    fn on_event(&mut self, ev: &SimEvent, app: &mut AppState) {
        if let SimEvent::Edited { id, result: Err(d) } = ev {
            let why = d.first().map_or(String::new(), ToString::to_string);
            self.ed.borrow_mut().kernel_rejected(app, *id, &why);
        }
    }

    fn ui(&mut self, ui: &mut Ui, app: &mut AppState) {
        let shared = self.ed.clone();
        let mut ed = shared.borrow_mut();
        let def_id = app.view.def.clone();
        ed.sync(&app.doc, &app.view.path, &def_id);
        let sel = ed.selection.clone();
        // Only the selected part of the def is copied (defs can be large; this runs every
        // frame, and an action may change the document while the panel draws).
        let fresh = self
            .part
            .as_ref()
            .is_some_and(|(d, rev, s, _)| *d == def_id && *rev == app.doc.revision && *s == sel);
        if !fresh {
            let Some(d) = app.doc.def(&def_id) else {
                ui.label("No def");
                return;
            };
            self.part = Some((
                def_id.clone(),
                app.doc.revision,
                sel.clone(),
                selected_part(d, &sel),
            ));
            self.part_gen += 1;
        }
        // Moved out for the frame (the panel methods borrow `self` mutably), put back below.
        let part = self.part.take().expect("set");
        let def = &part.3;
        egui::ScrollArea::vertical()
            .auto_shrink([false, true])
            .show(ui, |ui| {
                if sel.is_empty() {
                    ui.label(RichText::new("Nothing selected").weak());
                    ui.add_space(6.0);
                    ui.label(
                        RichText::new(
                            "Click to select, drag a box, Shift toggles.\n\
                             Right-click an item for its menu.\n\
                             Drag from a pin to draw a wire (W: wire tool).\n\
                             R rotate · F flip · Del delete\n\
                             Ctrl+C/X/V/D · Ctrl+Z / Ctrl+Shift+Z",
                        )
                        .small()
                        .weak(),
                    );
                    return;
                }
                if let Some(inst_id) = sel.single_instance() {
                    let Some(inst) = def.instances.iter().find(|i| i.id == inst_id) else {
                        return;
                    };
                    match &inst.kind {
                        InstKind::Atom(atom) => {
                            ui.heading(atom_label(atom));
                            ui.label(RichText::new(format!("atom instance {inst_id}")).weak());
                            ui.separator();
                            self.name_ui(ui, &mut ed, app, IdKind::Instance, inst_id);
                            ui.add_space(4.0);
                            self.atom_ui(ui, &mut ed, app, inst_id, atom);
                        }
                        InstKind::Def(d) => {
                            ui.heading(d);
                            ui.label(RichText::new(format!("package instance {inst_id}")).weak());
                            ui.separator();
                            self.name_ui(ui, &mut ed, app, IdKind::Instance, inst_id);
                            if ui.button("Enter package").clicked() {
                                super::menus::enter_package(app, inst_id);
                            }
                            ui.add_space(4.0);
                            if let Some(cd) = app.doc.def(d) {
                                ui.label(RichText::new("pins").weak());
                                egui::Grid::new("panda_edit_pkg_pins").num_columns(3).show(
                                    ui,
                                    |ui| {
                                        for p in &cd.pins {
                                            ui.label(RichText::new(&p.name).monospace());
                                            ui.label(dir_label(p.dir));
                                            ui.label(format!("{}b", p.width));
                                            ui.end_row();
                                        }
                                    },
                                );
                            }
                        }
                    }
                } else if let Some(net_id) = sel.single_net() {
                    let net_id = net_id.to_string();
                    self.net_ui(ui, &mut ed, app, def, &net_id);
                } else if sel.items.len() == 1
                    && sel.wires.is_empty()
                    && let Some(ItemRef::Pin(name)) = sel.items.iter().next()
                {
                    if let Some(p) = def.pins.iter().find(|p| &p.name == name) {
                        ui.heading(format!("pin {name}"));
                        ui.label(format!("{} · {} bit(s)", dir_label(p.dir), p.width));
                        let net = def
                            .nets
                            .iter()
                            .find(|n| n.ends.contains(&Endpoint::Pin(name.clone())));
                        ui.label(
                            RichText::new(match net {
                                Some(n) => format!("on net {}", n.id),
                                None => "not connected".into(),
                            })
                            .weak(),
                        );
                        wave_button(ui, app, &super::menus::Target::Pin(name.clone()));
                    }
                } else {
                    self.multi_ui(ui, &mut ed, app, def);
                }
            });
        self.part = Some(part);
        if let Some(m) = &ed.message {
            ui.add_space(8.0);
            ui.colored_label(ERROR, m);
        }
        ui.add_space(8.0);
        ui.separator();
        let undo = ed.history.undo_label().map(str::to_string);
        let redo = ed.history.redo_label().map(str::to_string);
        ui.label(
            RichText::new(format!(
                "undo: {}   redo: {}",
                undo.as_deref().unwrap_or("—"),
                redo.as_deref().unwrap_or("—")
            ))
            .small()
            .weak(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drafts_round_trip_every_atom_kind() {
        let atoms = [
            Atom::Gate {
                op: GateOp::Nor,
                inputs: 3,
                width: 4,
                reduce: false,
            },
            Atom::Gate {
                op: GateOp::Xor,
                inputs: 1,
                width: 8,
                reduce: true,
            },
            Atom::Not { width: 2 },
            Atom::Buf { width: 1 },
            Atom::Tri { width: 8 },
            Atom::Splitter {
                width: 8,
                parts: vec![(4, 7), (0, 3)],
            },
            Atom::Splitter {
                width: 3,
                parts: vec![(0, 0), (1, 1), (2, 2)],
            },
            Atom::Constant {
                value: Value::from_u64(8, 0x2a),
            },
            Atom::Constant {
                value: Value::from_bool(true),
            },
            Atom::Constant {
                value: Value::parse(4, "0b1x0z").unwrap(),
            },
            Atom::Constant {
                value: Value::xs(16),
            },
            Atom::Clock {
                period: 10,
                high: 3,
                phase: 2,
            },
            Atom::PowerOnReset { ticks: 7 },
        ];
        for a in atoms {
            let d = AtomDraft::from_atom(&a);
            assert_eq!(d.to_atom().as_ref(), Ok(&a), "{d:?}");
        }
    }

    #[test]
    fn draft_errors_are_reported() {
        let mut d = AtomDraft::from_atom(&Atom::Clock {
            period: 4,
            high: 2,
            phase: 0,
        });
        d.high = 4;
        assert!(d.to_atom().unwrap_err().contains("high"));
        let mut d = AtomDraft::from_atom(&Atom::Splitter {
            width: 4,
            parts: vec![(0, 3)],
        });
        d.parts = vec![(4, 7)];
        assert!(d.to_atom().unwrap_err().contains("out of range"));
        let mut d = AtomDraft::from_atom(&Atom::Constant {
            value: Value::from_u64(8, 3),
        });
        assert_eq!(d.radix, Radix::Hex);
        d.value = "1ff".into();
        assert!(d.to_atom().unwrap_err().contains("fit"));
        d.value = "9".into();
        assert_eq!(
            d.to_atom(),
            Ok(Atom::Constant {
                value: Value::from_u64(8, 9)
            })
        );
        d.width = 0;
        assert!(d.to_atom().unwrap_err().contains("width"));
    }

    #[test]
    fn radix_formatting_and_parsing() {
        let v = Value::from_u64(12, 0xa5c);
        assert_eq!(format_value(&v, Radix::Hex), "a5c");
        assert_eq!(format_value(&v, Radix::Dec), "2652");
        assert_eq!(format_value(&v, Radix::Bin), "101001011100");
        for r in Radix::ALL {
            assert_eq!(
                parse_value(12, &format_value(&v, r), r),
                Ok(v.clone()),
                "{r:?}"
            );
        }
        // Prefixes override the radix; underscores are ignored.
        assert_eq!(
            parse_value(12, "0b1010_0101_1100", Radix::Dec),
            Ok(v.clone())
        );
        assert_eq!(parse_value(12, "0xA5C", Radix::Bin), Ok(v.clone()));
        // Errors.
        assert!(
            parse_value(4, "12", Radix::Bin)
                .unwrap_err()
                .contains("binary digit")
        );
        assert!(
            parse_value(4, "g", Radix::Hex)
                .unwrap_err()
                .contains("hex digit")
        );
        assert!(
            parse_value(4, "11111", Radix::Bin)
                .unwrap_err()
                .contains("fit")
        );
        assert!(parse_value(4, "", Radix::Hex).is_err());
        // Leading zeros do not count against the width; hex is not limited to 128 bits.
        assert_eq!(
            parse_value(4, "000f", Radix::Hex),
            Ok(Value::from_u64(4, 15))
        );
        let wide = parse_value(200, &"f".repeat(50), Radix::Hex).unwrap();
        assert!(wide.bits().all(|b| b == Logic::One));
        assert_eq!(format_value(&wide, Radix::Hex), "f".repeat(50));
        // Unknown bits: binary only, uniform x/z abbreviated.
        let xz = Value::parse(4, "0b1x0z").unwrap();
        assert_eq!(format_value(&xz, Radix::Hex), "0b1x0z");
        assert_eq!(format_value(&Value::zs(8), Radix::Dec), "z");
        assert_eq!(parse_value(8, "x", Radix::Dec), Ok(Value::xs(8)));
        // Radix switching converts the text.
        let mut d = AtomDraft::from_atom(&Atom::Constant { value: v.clone() });
        d.set_radix(Radix::Dec);
        assert_eq!(d.value, "2652");
        d.set_radix(Radix::Bin);
        assert_eq!(d.value, "101001011100");
        assert_eq!(value_text(&Value::from_u64(8, 42)), "0x2a");
        assert_eq!(value_text(&Value::from_bool(false)), "0");
    }

    #[test]
    fn splitter_helpers() {
        assert_eq!(parse_parts("7:4, 3:0"), Ok(vec![(4, 7), (0, 3)]));
        assert_eq!(parse_parts("0,1, 2"), Ok(vec![(0, 0), (1, 1), (2, 2)]));
        assert_eq!(parts_text(&[(4, 7), (0, 0)]), "7:4, 0");
        assert!(parse_parts("a:b").is_err());
        assert!(parse_parts(" , ").is_err());
        assert_eq!(even_parts(8, 4), vec![(0, 3), (4, 7)]);
        assert_eq!(even_parts(10, 4), vec![(0, 3), (4, 7), (8, 9)]);
        assert_eq!(even_parts(3, 1), vec![(0, 0), (1, 1), (2, 2)]);
        assert!(splitter_notes(8, &[(0, 3), (4, 7)]).is_empty());
        assert_eq!(
            splitter_notes(8, &[(0, 3), (3, 5)]),
            vec!["bit(s) 7:6 not in any part", "bit(s) 3 in several parts"]
        );
    }

    #[test]
    fn common_fields_of_mixed_atoms() {
        let and4 = Atom::Gate {
            op: GateOp::And,
            inputs: 2,
            width: 4,
            reduce: false,
        };
        let or4 = Atom::Gate {
            op: GateOp::Or,
            inputs: 2,
            width: 4,
            reduce: false,
        };
        let not8 = Atom::Not { width: 8 };
        let clk = Atom::Clock {
            period: 4,
            high: 2,
            phase: 0,
        };
        let c = common_fields(&[&and4, &or4]);
        assert_eq!(
            c,
            vec![
                (Field::Op, None),
                (Field::Inputs, Some(FieldValue::Num(2))),
                (Field::Width, Some(FieldValue::Num(4))),
                (Field::Reduce, Some(FieldValue::Flag(false))),
            ]
        );
        assert_eq!(common_fields(&[&and4, &not8]), vec![(Field::Width, None)]);
        assert!(common_fields(&[&and4, &clk]).is_empty());
        assert_eq!(
            with_field(&not8, Field::Width, FieldValue::Num(2)),
            Some(Atom::Not { width: 2 })
        );
        assert_eq!(with_field(&not8, Field::Period, FieldValue::Num(2)), None);
        let k = Atom::Constant {
            value: Value::from_u64(8, 0x5a),
        };
        assert_eq!(
            with_field(&k, Field::Width, FieldValue::Num(4)),
            Some(Atom::Constant {
                value: Value::from_u64(4, 0xa)
            })
        );
        let r = with_field(&and4, Field::Reduce, FieldValue::Flag(true)).unwrap();
        assert!(matches!(
            r,
            Atom::Gate {
                inputs: 1,
                reduce: true,
                ..
            }
        ));
    }
}
