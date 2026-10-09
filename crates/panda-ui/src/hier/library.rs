//! The component library (pure model): atoms, the document's own defs ("user packages") and
//! parameterized panda-circuits blocks, and placing one of them into the viewed def as an
//! undoable [`DocChange`].
//!
//! Placing a block adds the defs it needs that the document does not have yet (generator ids
//! encode every parameter, so an existing def with the same id is the same block) and one
//! instance. A def cannot be placed into itself or into a def it (transitively) instantiates,
//! and `Clock` / `PowerOnReset` only into the top def.

use panda_circuits as pc;
use panda_kernel::EditOp;
use panda_netlist::{
    Atom, ComponentDef, DefId, DefRef, Design, DesignBuilder, GateOp, InstKind, Instance, Value,
};

use super::change::DocChange;
use super::classify::closure;
use super::package::fresh_instance_id;
use crate::doc::{Document, EditorData, InstGeom};
use crate::geom::{GridPt, Orient};

/// Library section.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Category {
    /// Gates, wiring, constants, sources.
    Atoms,
    /// Defs of the open document.
    User,
    /// panda-circuits arithmetic blocks.
    Arithmetic,
    /// panda-circuits latches, flip-flops, registers, counters, register files.
    Sequential,
    /// Tri-state helpers and oscillators.
    Misc,
}

impl Category {
    /// Section title.
    pub fn title(self) -> &'static str {
        match self {
            Category::Atoms => "Atoms",
            Category::User => "Packages in this design",
            Category::Arithmetic => "Arithmetic",
            Category::Sequential => "Sequential",
            Category::Misc => "Bus & misc",
        }
    }
}

/// An integer parameter of an entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Param {
    /// Name shown in the palette.
    pub name: &'static str,
    /// Current value.
    pub value: u32,
    /// Minimum.
    pub min: u32,
    /// Maximum.
    pub max: u32,
}

const fn param(name: &'static str, value: u32, min: u32, max: u32) -> Param {
    Param {
        name,
        value,
        min,
        max,
    }
}

/// How an entry produces what gets placed.
#[derive(Clone, Debug)]
pub enum EntryKind {
    /// An atom from the parameter values.
    Atom(fn(&[u32]) -> Atom),
    /// A panda-circuits generator from the parameter values.
    Block(fn(&mut DesignBuilder, &[u32]) -> DefRef),
    /// A def of the document.
    User(DefId),
}

/// One palette entry.
#[derive(Clone, Debug)]
pub struct LibEntry {
    /// Display name.
    pub name: String,
    /// Section.
    pub category: Category,
    /// Parameters (edited in the palette).
    pub params: Vec<Param>,
    /// Producer.
    pub kind: EntryKind,
}

/// What [`LibEntry::resolve`] yields: an atom, or a def plus the defs to add for it.
#[derive(Clone, Debug, PartialEq)]
pub enum Placeable {
    /// An atom instance.
    Atom(Atom),
    /// A def instance; `new_defs` are added first (dependency order).
    Def {
        /// The def to instantiate.
        id: DefId,
        /// Defs missing from the document.
        new_defs: Vec<ComponentDef>,
    },
}

impl Placeable {
    /// Short label (def id or atom name).
    pub fn label(&self) -> String {
        match self {
            Placeable::Atom(a) => atom_label(a),
            Placeable::Def { id, .. } => id.clone(),
        }
    }
}

fn atom_label(a: &Atom) -> String {
    match a {
        Atom::Gate { op, .. } => format!("{op:?}").to_uppercase(),
        Atom::Not { .. } => "NOT".into(),
        Atom::Buf { .. } => "BUF".into(),
        Atom::Tri { .. } => "TRI".into(),
        Atom::Splitter { .. } => "Splitter".into(),
        Atom::Constant { .. } => "Constant".into(),
        Atom::Clock { .. } => "Clock".into(),
        Atom::PowerOnReset { .. } => "POR".into(),
    }
}

impl LibEntry {
    fn new(name: &str, category: Category, params: &[Param], kind: EntryKind) -> LibEntry {
        LibEntry {
            name: name.to_string(),
            category,
            params: params.to_vec(),
            kind,
        }
    }

    /// Parameter values (clamped to their ranges).
    pub fn values(&self) -> Vec<u32> {
        self.params
            .iter()
            .map(|p| p.value.clamp(p.min, p.max))
            .collect()
    }

    /// Build what this entry places, given the document's design.
    pub fn resolve(&self, design: &Design) -> Result<Placeable, String> {
        let v = self.values();
        match &self.kind {
            EntryKind::Atom(f) => Ok(Placeable::Atom(f(&v))),
            EntryKind::User(id) => design
                .def(id)
                .map(|_| Placeable::Def {
                    id: id.clone(),
                    new_defs: Vec::new(),
                })
                .ok_or_else(|| format!("unknown def '{id}'")),
            EntryKind::Block(f) => {
                let mut db = DesignBuilder::new();
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&mut db, &v)))
                    .map_err(|_| format!("{}: invalid parameters {v:?}", self.name))?;
                let built = db.build(&r);
                Ok(Placeable::Def {
                    id: r.id,
                    new_defs: built
                        .defs
                        .into_iter()
                        .filter(|d| design.def(&d.id).is_none())
                        .collect(),
                })
            }
        }
    }

    /// True if this entry can be placed into def `into` of `design`.
    pub fn allowed_in(&self, design: &Design, into: &str) -> bool {
        match &self.kind {
            EntryKind::Atom(f) => {
                let a = f(&self.values());
                design.top == into || !matches!(a, Atom::Clock { .. } | Atom::PowerOnReset { .. })
            }
            EntryKind::User(id) => !closure(design, id).iter().any(|d| d == into),
            EntryKind::Block(_) => true,
        }
    }
}

fn gate(op: GateOp) -> fn(&[u32]) -> Atom {
    // Non-capturing closures per op (fn pointers cannot capture).
    match op {
        GateOp::And => |v| g(GateOp::And, v),
        GateOp::Or => |v| g(GateOp::Or, v),
        GateOp::Nand => |v| g(GateOp::Nand, v),
        GateOp::Nor => |v| g(GateOp::Nor, v),
        GateOp::Xor => |v| g(GateOp::Xor, v),
        GateOp::Xnor => |v| g(GateOp::Xnor, v),
    }
}

fn g(op: GateOp, v: &[u32]) -> Atom {
    Atom::Gate {
        op,
        inputs: v[0],
        width: v[1],
        reduce: false,
    }
}

/// The built-in entries (atoms and panda-circuits blocks) with default parameters.
pub fn builtin_entries() -> Vec<LibEntry> {
    use Category::*;
    use EntryKind::{Atom as A, Block as B};
    let gi = [param("inputs", 2, 1, 16), param("width", 1, 1, 64)];
    let w = [param("width", 1, 1, 64)];
    let mut v = Vec::new();
    for (name, op) in [
        ("AND", GateOp::And),
        ("OR", GateOp::Or),
        ("NAND", GateOp::Nand),
        ("NOR", GateOp::Nor),
        ("XOR", GateOp::Xor),
        ("XNOR", GateOp::Xnor),
    ] {
        v.push(LibEntry::new(name, Atoms, &gi, A(gate(op))));
    }
    v.extend([
        LibEntry::new("NOT", Atoms, &w, A(|v| Atom::Not { width: v[0] })),
        LibEntry::new("BUF", Atoms, &w, A(|v| Atom::Buf { width: v[0] })),
        LibEntry::new("TRI", Atoms, &w, A(|v| Atom::Tri { width: v[0] })),
        LibEntry::new(
            "Splitter",
            Atoms,
            &[param("width", 8, 2, 64), param("parts", 8, 2, 64)],
            A(|v| {
                let (w, k) = (v[0], v[1].min(v[0]));
                // `k` near-equal parts, LSB first.
                let mut parts = Vec::new();
                let mut lo = 0;
                for i in 0..k {
                    let n = w / k + u32::from(i < w % k);
                    parts.push((lo, lo + n - 1));
                    lo += n;
                }
                Atom::Splitter { width: w, parts }
            }),
        ),
        LibEntry::new(
            "Constant",
            Atoms,
            &[param("width", 1, 1, 64), param("value", 0, 0, u32::MAX)],
            A(|v| Atom::Constant {
                value: Value::from_u64(v[0], u64::from(v[1]) & mask(v[0])),
            }),
        ),
        LibEntry::new(
            "Clock",
            Atoms,
            &[param("period", 16, 2, 1 << 20), param("high", 8, 1, 1 << 20)],
            A(|v| Atom::Clock {
                period: v[0],
                high: v[1].clamp(1, v[0] - 1),
                phase: 0,
            }),
        ),
        LibEntry::new(
            "Power-on reset",
            Atoms,
            &[param("ticks", 8, 1, 1 << 20)],
            A(|v| Atom::PowerOnReset { ticks: v[0] }),
        ),
    ]);
    let n = |d| [param("bits", d, 1, 32)];
    v.extend([
        LibEntry::new("half adder", Arithmetic, &[], B(|db, _| pc::half_adder(db))),
        LibEntry::new("full adder", Arithmetic, &[], B(|db, _| pc::full_adder(db))),
        LibEntry::new("ripple adder", Arithmetic, &n(8), B(|db, v| pc::ripple_adder(db, v[0]))),
        LibEntry::new("incrementer", Arithmetic, &n(8), B(|db, v| pc::incrementer(db, v[0]))),
        LibEntry::new("mux2", Arithmetic, &[param("width", 8, 1, 64)], B(|db, v| pc::mux2(db, v[0]))),
        LibEntry::new(
            "mux",
            Arithmetic,
            &[param("select bits", 2, 1, 6), param("width", 8, 1, 64)],
            B(|db, v| pc::mux(db, v[0], v[1])),
        ),
        LibEntry::new("decoder", Arithmetic, &[param("bits", 3, 1, 8)], B(|db, v| pc::decoder(db, v[0]))),
        LibEntry::new("equality", Arithmetic, &n(8), B(|db, v| pc::equality(db, v[0]))),
        LibEntry::new("NAND SR latch", Sequential, &[], B(|db, _| pc::nand_sr_latch(db))),
        LibEntry::new("NOR SR latch", Sequential, &[], B(|db, _| pc::nor_sr_latch(db))),
        LibEntry::new("D latch", Sequential, &[], B(|db, _| pc::d_latch(db))),
        LibEntry::new("D flip-flop", Sequential, &[], B(|db, _| pc::ms_dff(db))),
        LibEntry::new("DFF + sync reset", Sequential, &[], B(|db, _| pc::dff_rst(db))),
        LibEntry::new("DFF + async clear", Sequential, &[], B(|db, _| pc::dff_arst(db))),
        LibEntry::new("DFF + enable", Sequential, &[], B(|db, _| pc::dff_en(db))),
        LibEntry::new("T flip-flop", Sequential, &[], B(|db, _| pc::toggle_ff(db))),
        LibEntry::new("register", Sequential, &n(8), B(|db, v| pc::register(db, v[0]))),
        LibEntry::new(
            "register (gated clock)",
            Sequential,
            &n(8),
            B(|db, v| pc::register_gated(db, v[0])),
        ),
        LibEntry::new("sync counter", Sequential, &n(4), B(|db, v| pc::sync_counter(db, v[0]))),
        LibEntry::new("ripple counter", Sequential, &n(4), B(|db, v| pc::ripple_counter(db, v[0]))),
        LibEntry::new(
            "register file",
            Sequential,
            &[param("words", 4, 2, 64), param("width", 8, 1, 64)],
            B(|db, v| pc::register_file(db, v[0] as usize, v[1], pc::RegFileStyle::EnableMux)),
        ),
        LibEntry::new("bus port", Misc, &[param("width", 8, 1, 64)], B(|db, v| pc::bus_port(db, v[0]))),
        LibEntry::new(
            "transceiver",
            Misc,
            &[param("width", 8, 1, 64)],
            B(|db, v| pc::transceiver(db, v[0])),
        ),
        LibEntry::new(
            "ring oscillator",
            Misc,
            &[param("stages", 3, 1, 31)],
            B(|db, v| pc::ring_oscillator(db, v[0] | 1, true)),
        ),
    ]);
    v
}

fn mask(w: u32) -> u64 {
    if w >= 64 { u64::MAX } else { (1 << w) - 1 }
}

/// Entries for the document's own defs (every def except the top), sorted by id.
pub fn user_entries(design: &Design) -> Vec<LibEntry> {
    let mut v: Vec<LibEntry> = design
        .defs
        .iter()
        .filter(|d| d.id != design.top)
        .map(|d| {
            LibEntry::new(
                &d.id,
                Category::User,
                &[],
                EntryKind::User(d.id.clone()),
            )
        })
        .collect();
    v.sort_by(|a, b| a.name.cmp(&b.name));
    v
}

/// A change adding `new_defs` (with auto-laid-out geometry) to `doc`; `None` if empty.
pub fn add_defs_change(doc: &Document, new_defs: &[ComponentDef], label: &str) -> Option<DocChange> {
    if new_defs.is_empty() {
        return None;
    }
    let mut d2 = doc.design.clone();
    d2.defs.extend(new_defs.iter().cloned());
    let ids: Vec<DefId> = new_defs.iter().map(|d| d.id.clone()).collect();
    let lay = crate::layout::layout_defs(&d2, &doc.editors, &ids);
    Some(DocChange {
        label: label.to_string(),
        redo: new_defs.iter().cloned().map(EditOp::AddDef).collect(),
        undo: ids.iter().rev().map(|i| EditOp::RemoveDef(i.clone())).collect(),
        geom_before: ids.iter().map(|i| (i.clone(), None)).collect(),
        geom_after: ids.iter().map(|i| (i.clone(), lay.get(i).cloned())).collect(),
    })
}

/// Plan placing `what` into def `into` with its symbol's top-left corner at `pos`.
pub fn place(
    doc: &Document,
    into: &str,
    what: &Placeable,
    pos: GridPt,
) -> Result<(DocChange, String), String> {
    let def = doc
        .def(into)
        .ok_or_else(|| format!("unknown def '{into}'"))?;
    let (kind, prefix, new_defs) = match what {
        Placeable::Atom(a) => {
            if into != doc.design.top && matches!(a, Atom::Clock { .. } | Atom::PowerOnReset { .. }) {
                return Err("clock and reset sources may only be placed in the top def".into());
            }
            (InstKind::Atom(a.clone()), "g", Vec::new())
        }
        Placeable::Def { id, new_defs } => {
            if id == into || closure(&doc.design, id).iter().any(|d| d == into) {
                return Err(format!("'{id}' contains '{into}': placing it would make a cycle"));
            }
            (InstKind::Def(id.clone()), "u", new_defs.clone())
        }
    };
    let iid = fresh_instance_id(def, prefix);
    let mut redo: Vec<EditOp> = new_defs.iter().cloned().map(EditOp::AddDef).collect();
    redo.push(EditOp::AddInstance {
        def: into.to_string(),
        instance: Instance {
            id: iid.clone(),
            kind,
        },
    });
    let mut undo = vec![EditOp::RemoveInstance {
        def: into.to_string(),
        id: iid.clone(),
    }];
    undo.extend(new_defs.iter().rev().map(|d| EditOp::RemoveDef(d.id.clone())));
    let before = doc.editor(into).cloned();
    let mut after = before.clone().unwrap_or_else(|| EditorData {
        version: crate::doc::EDITOR_VERSION,
        ..EditorData::default()
    });
    after.instances.insert(
        iid.clone(),
        InstGeom {
            pos,
            orient: Orient::default(),
        },
    );
    after.auto = false;
    let mut geom_before = vec![(into.to_string(), before)];
    let mut geom_after = vec![(into.to_string(), Some(after))];
    // New defs get auto-laid-out geometry so they can be entered right away.
    if !new_defs.is_empty() {
        let mut d2 = doc.design.clone();
        d2.defs.extend(new_defs.iter().cloned());
        let ids: Vec<DefId> = new_defs.iter().map(|d| d.id.clone()).collect();
        let lay = crate::layout::layout_defs(&d2, &doc.editors, &ids);
        for d in &new_defs {
            geom_before.push((d.id.clone(), None));
            geom_after.push((d.id.clone(), lay.get(&d.id).cloned()));
        }
    }
    Ok((
        DocChange {
            label: format!("Place {}", what.label()),
            redo,
            undo,
            geom_before,
            geom_after,
        },
        iid,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples;

    #[test]
    fn every_builtin_entry_resolves_and_places() {
        let mut doc = Document::new(examples::counter());
        doc.layout_missing();
        let top = doc.design.top.clone();
        for (k, e) in builtin_entries().iter().enumerate() {
            let p = e.resolve(&doc.design).unwrap_or_else(|err| panic!("{}: {err}", e.name));
            let (c, iid) = place(&doc, &top, &p, GridPt { x: 100, y: 4 * k as i32 })
                .unwrap_or_else(|err| panic!("{}: {err}", e.name));
            let mut d2 = doc.clone();
            c.apply_doc(&mut d2).unwrap_or_else(|err| panic!("{}: {err}", e.name));
            assert!(d2.editor(&top).unwrap().instances.contains_key(&iid));
            // Structurally valid as a whole (dangling ports are fine).
            panda_netlist::flatten(&d2.design).unwrap_or_else(|err| panic!("{}: {err:?}", e.name));
            c.revert_doc(&mut d2).unwrap();
            assert_eq!(d2.design, doc.design, "{}", e.name);
        }
    }

    #[test]
    fn cycles_and_sources_are_rejected() {
        let doc = Document::new(examples::counter());
        // sync_counter_4 contains register_4: placing the counter into the register is a cycle.
        let p = Placeable::Def {
            id: "sync_counter_4".into(),
            new_defs: vec![],
        };
        assert!(place(&doc, "register_4", &p, GridPt::default()).is_err());
        let user = user_entries(&doc.design);
        let e = user.iter().find(|e| e.name == "sync_counter_4").unwrap();
        assert!(!e.allowed_in(&doc.design, "register_4"));
        assert!(e.allowed_in(&doc.design, &doc.design.top));
        let clk = Placeable::Atom(Atom::Clock {
            period: 4,
            high: 2,
            phase: 0,
        });
        assert!(place(&doc, "sync_counter_4", &clk, GridPt::default()).is_err());
    }
}
