//! The inspector (right panel): parameters of the selected atom (gate op / inputs / width /
//! reduce, splitter parts, constant value, clock period / high / phase, PowerOnReset ticks),
//! net termination, and a summary of everything else.
//!
//! Atom parameters are edited in an [`AtomDraft`] (pure, parsed and checked without egui) and
//! applied as one undoable `SetAtom` action. Width conflicts with connected nets are shown
//! inline; "Apply & disconnect" drops the mismatched connections.

use egui::{Color32, RichText, Ui};
use panda_netlist::{Atom, Dir, GateOp, InstKind, Termination, Value};

use super::SharedEditor;
use super::actions::{atom_label, ep_label, plan_set_atom};
use crate::app::{AppState, Panel, Slot};
use crate::doc::ItemRef;
use crate::simhost::SimEvent;

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
    /// Splitter parts, `hi:lo` or `bit`, comma separated, part 0 first.
    pub parts: String,
    /// Constant value (`0x2a`, `0b10x1`, `42`, `x`, `z`).
    pub value: String,
    /// Clock period (ticks).
    pub period: u32,
    /// Clock high time (ticks).
    pub high: u32,
    /// Clock phase (ticks).
    pub phase: u32,
    /// PowerOnReset duration (ticks).
    pub ticks: u32,
}

/// Text form of a constant value: `0`/`1`/`x`/`z` for one bit, hex if fully known, else
/// binary.
pub fn value_text(v: &Value) -> String {
    let w = v.width();
    if w == 1 {
        return v.get(0).to_char().to_string();
    }
    match v.to_u64() {
        Some(n) if w <= 64 => format!("{n:#x}"),
        _ => {
            let bits: String = (0..w).rev().map(|i| v.get(i).to_char()).collect();
            format!("0b{bits}")
        }
    }
}

/// Text form of splitter parts.
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

/// Parse splitter parts (`"7:4, 3:0"`, `"0, 1, 2"`).
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
                .map_err(|_| format!("part {i}: '{t}' is not a bit index"))
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

impl AtomDraft {
    /// Draft of an existing atom.
    pub fn from_atom(a: &Atom) -> AtomDraft {
        let mut d = AtomDraft {
            kind: AtomKind::Gate,
            op: GateOp::And,
            inputs: 2,
            width: 1,
            reduce: false,
            parts: String::new(),
            value: String::new(),
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
            Atom::Not { width } => {
                d.kind = AtomKind::Not;
                d.width = *width;
            }
            Atom::Buf { width } => {
                d.kind = AtomKind::Buf;
                d.width = *width;
            }
            Atom::Tri { width } => {
                d.kind = AtomKind::Tri;
                d.width = *width;
            }
            Atom::Splitter { width, parts } => {
                d.kind = AtomKind::Splitter;
                d.width = *width;
                d.parts = parts_text(parts);
            }
            Atom::Constant { value } => {
                d.kind = AtomKind::Constant;
                d.width = value.width();
                d.value = value_text(value);
            }
            Atom::Clock {
                period,
                high,
                phase,
            } => {
                d.kind = AtomKind::Clock;
                d.period = *period;
                d.high = *high;
                d.phase = *phase;
            }
            Atom::PowerOnReset { ticks } => {
                d.kind = AtomKind::PowerOnReset;
                d.ticks = *ticks;
            }
        }
        d
    }

    /// The atom described by the draft, or why it is invalid.
    pub fn to_atom(&self) -> Result<Atom, String> {
        let a = match self.kind {
            AtomKind::Gate => Atom::Gate {
                op: self.op,
                inputs: self.inputs,
                width: self.width,
                reduce: self.reduce,
            },
            AtomKind::Not => Atom::Not { width: self.width },
            AtomKind::Buf => Atom::Buf { width: self.width },
            AtomKind::Tri => Atom::Tri { width: self.width },
            AtomKind::Splitter => Atom::Splitter {
                width: self.width,
                parts: parse_parts(&self.parts)?,
            },
            AtomKind::Constant => Atom::Constant {
                value: Value::parse(self.width.max(1), &self.value)?,
            },
            AtomKind::Clock => Atom::Clock {
                period: self.period,
                high: self.high,
                phase: self.phase,
            },
            AtomKind::PowerOnReset => Atom::PowerOnReset { ticks: self.ticks },
        };
        if self.width == 0 {
            return Err("width must be ≥ 1".into());
        }
        super::actions::check_atom(&a)?;
        Ok(a)
    }

    /// Draw the parameter fields. Returns true if Enter was pressed in a text field.
    fn ui(&mut self, ui: &mut Ui) -> bool {
        let mut enter = false;
        egui::Grid::new("panda_edit_atom")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| match self.kind {
                AtomKind::Gate => {
                    ui.label("operation");
                    egui::ComboBox::from_id_salt("panda_edit_gate_op")
                        .selected_text(format!("{:?}", self.op).to_uppercase())
                        .show_ui(ui, |ui| {
                            for op in [
                                GateOp::And,
                                GateOp::Or,
                                GateOp::Nand,
                                GateOp::Nor,
                                GateOp::Xor,
                                GateOp::Xnor,
                            ] {
                                let t = format!("{op:?}").to_uppercase();
                                ui.selectable_value(&mut self.op, op, t);
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
                    ui.label("parts");
                    let r = ui.add(
                        egui::TextEdit::singleline(&mut self.parts)
                            .hint_text("7:4, 3:0")
                            .desired_width(120.0),
                    );
                    enter |= r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    ui.end_row();
                }
                AtomKind::Constant => {
                    ui.label("width");
                    ui.add(egui::DragValue::new(&mut self.width).range(1..=1024));
                    ui.end_row();
                    ui.label("value");
                    let r = ui.add(
                        egui::TextEdit::singleline(&mut self.value)
                            .hint_text("0x2a / 0b1010 / 42")
                            .desired_width(120.0),
                    );
                    enter |= r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    ui.end_row();
                }
                AtomKind::Clock => {
                    ui.label("period");
                    ui.add(egui::DragValue::new(&mut self.period).range(2..=1_000_000));
                    ui.end_row();
                    ui.label("high");
                    ui.add(
                        egui::DragValue::new(&mut self.high)
                            .range(1..=self.period.saturating_sub(1).max(1)),
                    );
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
        enter
    }
}

/// Colour of inline errors.
pub const ERROR: Color32 = Color32::from_rgb(230, 80, 70);

/// The inspector panel.
pub struct Inspector {
    ed: SharedEditor,
    /// `(def, instance, revision)` the draft was loaded from, and the draft.
    draft: Option<((String, String, u64), AtomDraft)>,
}

impl Inspector {
    /// Inspector over the shared editor.
    pub fn new(ed: SharedEditor) -> Inspector {
        Inspector { ed, draft: None }
    }
}

fn term_label(t: Termination) -> &'static str {
    match t {
        Termination::None => "none",
        Termination::PullUp => "pull-up",
        Termination::PullDown => "pull-down",
        Termination::Keeper => "keeper",
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
        let ed = self.ed.clone();
        let mut ed = ed.borrow_mut();
        let def_id = app.view.def.clone();
        ed.sync(&app.doc, &app.view.path, &def_id);
        // Only the selected element is cloned (defs can be large; this runs every frame).
        let Some(def) = app.doc.def(&def_id) else {
            ui.label("No def");
            return;
        };
        let sel = ed.selection.clone();
        if let Some(inst_id) = sel.single_instance() {
            let Some(inst) = def.instances.iter().find(|i| i.id == inst_id).cloned() else {
                return;
            };
            match &inst.kind {
                InstKind::Atom(atom) => {
                    ui.heading(atom_label(atom));
                    ui.label(RichText::new(format!("instance {inst_id}")).weak());
                    ui.separator();
                    let key = (def_id.clone(), inst_id.to_string(), app.doc.revision);
                    if self.draft.as_ref().is_none_or(|(k, _)| *k != key) {
                        self.draft = Some((key, AtomDraft::from_atom(atom)));
                    }
                    let (_, draft) = self.draft.as_mut().expect("set");
                    let enter = draft.ui(ui);
                    let parsed = draft.to_atom();
                    ui.add_space(6.0);
                    match parsed {
                        Err(e) => {
                            ui.colored_label(ERROR, e);
                        }
                        Ok(new) if &new != atom => {
                            let plan = plan_set_atom(&app.doc, &def_id, inst_id, &new);
                            match plan {
                                Err(e) => {
                                    ui.colored_label(ERROR, e);
                                }
                                Ok(plan) => {
                                    for c in &plan.conflicts {
                                        ui.colored_label(ERROR, format!("⚠ {c}"));
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
                                    ui.horizontal(|ui| {
                                        let ok = plan.mismatched.is_empty();
                                        if ui.add_enabled(ok, egui::Button::new("Apply")).clicked()
                                            || (ok && enter)
                                        {
                                            let _ = ed.set_atom(app, &def_id, inst_id, &new, false);
                                        }
                                        if !ok && ui.button("Apply & disconnect").clicked() {
                                            let _ = ed.set_atom(app, &def_id, inst_id, &new, true);
                                        }
                                        if ui.button("Revert").clicked() {
                                            self.draft = None;
                                        }
                                    });
                                }
                            }
                        }
                        Ok(_) => {}
                    }
                }
                InstKind::Def(d) => {
                    ui.heading(d);
                    ui.label(RichText::new(format!("instance {inst_id} (package)")).weak());
                    ui.separator();
                    if let Some(cd) = app.doc.def(d) {
                        for p in &cd.pins {
                            ui.label(format!("{:?} {} [{}]", p.dir, p.name, p.width));
                        }
                    }
                }
            }
        } else if let Some(net_id) = sel.single_net() {
            if let Some(net) = def.nets.iter().find(|n| n.id == net_id).cloned() {
                ui.heading(format!("net {net_id}"));
                ui.label(format!("width {}", net.width));
                ui.separator();
                let mut term = net.termination;
                egui::ComboBox::from_label("termination")
                    .selected_text(term_label(term))
                    .show_ui(ui, |ui| {
                        for t in [
                            Termination::None,
                            Termination::PullUp,
                            Termination::PullDown,
                            Termination::Keeper,
                        ] {
                            ui.selectable_value(&mut term, t, term_label(t));
                        }
                    });
                if term != net.termination {
                    let _ = ed.set_termination(app, &def_id, net_id, term);
                }
                ui.add_space(4.0);
                ui.label(RichText::new("connected to").weak());
                for e in &net.ends {
                    ui.label(format!("  {}", ep_label(e)));
                }
            }
        } else if sel.items.len() == 1
            && sel.wires.is_empty()
            && let Some(ItemRef::Pin(name)) = sel.items.iter().next()
        {
            if let Some(p) = def.pins.iter().find(|p| &p.name == name).cloned() {
                ui.heading(format!("pin {name}"));
                ui.label(format!(
                    "{} · {} bit(s)",
                    match p.dir {
                        Dir::In => "input",
                        Dir::Out => "output",
                        Dir::InOut => "in/out",
                    },
                    p.width
                ));
            }
        } else if sel.is_empty() {
            ui.label(RichText::new("Nothing selected").weak());
            ui.add_space(6.0);
            ui.label(
                RichText::new(
                    "Click to select, drag a box, Shift toggles.\n\
                     Drag from a pin to draw a wire (W: wire tool).\n\
                     R rotate · F flip · Del delete\n\
                     Ctrl+C/X/V/D · Ctrl+Z / Ctrl+Shift+Z",
                )
                .small()
                .weak(),
            );
        } else {
            ui.heading(format!(
                "{} item(s), {} wire segment(s)",
                sel.items.len(),
                sel.wires.len()
            ));
        }
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
        d.parts = "7:4".into();
        assert!(d.to_atom().unwrap_err().contains("out of range"));
        d.parts = "a:b".into();
        assert!(d.to_atom().is_err());
        let mut d = AtomDraft::from_atom(&Atom::Constant {
            value: Value::from_u64(4, 3),
        });
        d.value = "0x1ff".into();
        assert!(d.to_atom().unwrap_err().contains("fit"));
        d.value = "0x9".into();
        assert_eq!(
            d.to_atom(),
            Ok(Atom::Constant {
                value: Value::from_u64(4, 9)
            })
        );
    }
}
