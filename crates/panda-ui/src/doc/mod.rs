//! The document model: a [`panda_netlist::Design`] plus editor geometry per def.
//!
//! Geometry lives in each def's `editor` JSON field ([`ComponentDef::editor`], ignored by the
//! kernel) as an [`EditorData`]: instance placement ([`InstGeom`]), wire polylines with junction
//! points per net ([`NetGeom`]), placement of the def's own pin markers, free text labels and an
//! optional package symbol layout ([`SymbolLayout`]). [`Document`] keeps the parsed geometry
//! next to the design and writes it back on save, so a design round-trips through the
//! `panda/1` JSON format unchanged.
//!
//! Defs without geometry (generated designs: panda-circuits blocks, Panda-8) are laid out by
//! [`crate::layout`] ([`Document::layout_missing`]).
//!
//! [`values`] maps (view instance path, net id, bit) to flat net indices for live values.

pub mod values;

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use panda_kernel::EditOp;
use panda_netlist::{ComponentDef, DefId, Design, Diagnostic, InstKind};
use serde::{Deserialize, Serialize};

use crate::geom::{
    GridPt, Orient, Symbol, SymbolLayout, instance_symbols, package_symbol, pin_symbol,
};

/// Current [`EditorData::version`].
pub const EDITOR_VERSION: u32 = 1;

/// Placement of an instance (or of a pin marker).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstGeom {
    /// Top-left corner of the (rotated) symbol box.
    pub pos: GridPt,
    /// Orientation.
    #[serde(default, flatten)]
    pub orient: Orient,
}

/// Wire geometry of one net: axis-aligned polylines and junction dots.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetGeom {
    /// Polylines (each ≥ 2 points; consecutive points differ in one coordinate).
    #[serde(default)]
    pub polylines: Vec<Vec<GridPt>>,
    /// Junction points (where three or more wire ends / segments meet).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub junctions: Vec<GridPt>,
}

impl NetGeom {
    /// Every segment `(a, b)` of every polyline.
    pub fn segments(&self) -> impl Iterator<Item = (GridPt, GridPt)> + '_ {
        self.polylines
            .iter()
            .flat_map(|l| l.windows(2).map(|w| (w[0], w[1])))
    }
}

/// A free text label.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Label {
    /// Anchor (top-left).
    pub pos: GridPt,
    /// Text.
    pub text: String,
}

/// Editor geometry of one def (stored in [`ComponentDef::editor`]).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EditorData {
    /// Format version ([`EDITOR_VERSION`]).
    #[serde(default)]
    pub version: u32,
    /// Placement by instance id.
    #[serde(default)]
    pub instances: BTreeMap<String, InstGeom>,
    /// Placement of the def's own pin markers, by pin name.
    #[serde(default)]
    pub pins: BTreeMap<String, InstGeom>,
    /// Wires by net id.
    #[serde(default)]
    pub wires: BTreeMap<String, NetGeom>,
    /// Free labels.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<Label>,
    /// Package symbol layout (how instances of this def look); automatic if absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub symbol: Option<SymbolLayout>,
    /// True if the geometry was produced by auto-layout (not edited by hand yet).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub auto: bool,
}

impl EditorData {
    /// Parse a def's `editor` field (`null` or unparsable → `None`).
    pub fn from_json(v: &serde_json::Value) -> Option<EditorData> {
        if v.is_null() {
            return None;
        }
        serde_json::from_value(v.clone()).ok()
    }

    /// Serialize into a def's `editor` field.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("editor data serializes")
    }

    /// True if every instance and pin of `def` has a placement.
    pub fn covers(&self, def: &ComponentDef) -> bool {
        def.instances
            .iter()
            .all(|i| self.instances.contains_key(&i.id))
            && def.pins.iter().all(|p| self.pins.contains_key(&p.name))
    }
}

/// One placed item of a def's schematic (what the canvas draws and tools hit-test).
#[derive(Clone, Debug, PartialEq)]
pub struct Placed {
    /// What it is.
    pub item: ItemRef,
    /// Its symbol (local coordinates).
    pub symbol: Symbol,
    /// Placement.
    pub geom: InstGeom,
}

impl Placed {
    /// World position of a port.
    pub fn port_pos(&self, port: &str) -> Option<GridPt> {
        let p = self.symbol.port(port)?;
        Some(self.geom.pos + self.geom.orient.apply(p.pos, self.symbol.size))
    }

    /// World bounding box.
    pub fn rect(&self) -> crate::geom::GridRect {
        crate::geom::GridRect::from_pos_size(self.geom.pos, self.geom.orient.size(self.symbol.size))
    }
}

/// Reference to a placeable item of a def.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ItemRef {
    /// Instance by id.
    Instance(String),
    /// The def's own pin by name.
    Pin(String),
}

/// The open document.
#[derive(Clone, Debug)]
pub struct Document {
    /// The design (its defs' `editor` fields are stale; [`Document::editors`] is authoritative
    /// until [`Document::to_design`]).
    pub design: Design,
    /// Parsed geometry per def.
    pub editors: HashMap<DefId, EditorData>,
    /// File the document was loaded from / saved to.
    pub path: Option<PathBuf>,
    /// Bumped on every change (structure or geometry); caches key on it.
    pub revision: u64,
    /// Changed since the last save.
    pub dirty: bool,
}

impl Document {
    /// Wrap a design, parsing the geometry of every def that has some. Does not lay out.
    pub fn new(design: Design) -> Document {
        let editors = design
            .defs
            .iter()
            .filter_map(|d| EditorData::from_json(&d.editor).map(|e| (d.id.clone(), e)))
            .collect();
        Document {
            design,
            editors,
            path: None,
            revision: 1,
            dirty: false,
        }
    }

    /// Parse a `panda/1` JSON document.
    pub fn from_json(s: &str) -> Result<Document, String> {
        Ok(Document::new(Design::from_json(s)?))
    }

    /// The design with every def's geometry written into its `editor` field.
    pub fn to_design(&self) -> Design {
        let mut d = self.design.clone();
        for def in &mut d.defs {
            def.editor = match self.editors.get(&def.id) {
                Some(e) => e.to_json(),
                None => serde_json::Value::Null,
            };
        }
        d
    }

    /// Serialize to `panda/1` JSON (with geometry).
    pub fn to_json(&self) -> String {
        self.to_design().to_json()
    }

    /// Def by id.
    pub fn def(&self, id: &str) -> Option<&ComponentDef> {
        self.design.def(id)
    }

    /// Geometry of a def.
    pub fn editor(&self, id: &str) -> Option<&EditorData> {
        self.editors.get(id)
    }

    /// Defs reachable from the top that lack geometry (or whose geometry does not cover every
    /// instance and pin).
    pub fn defs_missing_layout(&self) -> Vec<DefId> {
        reachable_defs(&self.design)
            .into_iter()
            .filter(|id| {
                let def = self.design.def(id).expect("reachable def exists");
                self.editors.get(id).is_none_or(|e| !e.covers(def))
            })
            .collect()
    }

    /// Auto-lay out every def without (complete) geometry. Returns the laid-out def ids.
    pub fn layout_missing(&mut self) -> Vec<DefId> {
        let missing = self.defs_missing_layout();
        let out = crate::layout::layout_defs(&self.design, &self.editors, &missing);
        self.install_layouts(out)
    }

    /// Install geometry computed by [`crate::layout::layout_defs`] (e.g. on a worker thread).
    pub fn install_layouts(&mut self, layouts: HashMap<DefId, EditorData>) -> Vec<DefId> {
        let mut ids: Vec<DefId> = layouts.keys().cloned().collect();
        ids.sort();
        for (id, e) in layouts {
            self.editors.insert(id, e);
        }
        if !ids.is_empty() {
            self.revision += 1;
        }
        ids
    }

    /// Package symbol layout of a def (from its geometry), if set.
    pub fn symbol_layout(&self, def: &str) -> Option<&SymbolLayout> {
        self.editors.get(def).and_then(|e| e.symbol.as_ref())
    }

    /// Every placed item of `def` that has a placement: pin markers first, then instances in
    /// instance order.
    pub fn placed(&self, def_id: &str) -> Vec<Placed> {
        let Some(def) = self.design.def(def_id) else {
            return Vec::new();
        };
        let ed = self.editors.get(def_id);
        let mut out = Vec::new();
        for p in &def.pins {
            if let Some(g) = ed.and_then(|e| e.pins.get(&p.name)) {
                out.push(Placed {
                    item: ItemRef::Pin(p.name.clone()),
                    symbol: pin_symbol(p),
                    geom: *g,
                });
            }
        }
        let syms = instance_symbols(&self.design, def, |id| self.symbol_layout(id));
        for (inst, sym) in def.instances.iter().zip(syms) {
            if let (Some(sym), Some(g)) = (sym, ed.and_then(|e| e.instances.get(&inst.id))) {
                out.push(Placed {
                    item: ItemRef::Instance(inst.id.clone()),
                    symbol: sym,
                    geom: *g,
                });
            }
        }
        out
    }

    /// Symbol of instances of `def` (as packages).
    pub fn package_symbol(&self, def: &str) -> Option<Symbol> {
        self.design
            .def(def)
            .map(|d| package_symbol(d, self.symbol_layout(def)))
    }

    /// Apply structural edits (the same [`EditOp`]s that are sent to the kernel) to the
    /// document's design. Atomic: on error nothing changes. Geometry of removed instances, pins,
    /// nets and defs is dropped; new items get no geometry (the caller places them).
    pub fn apply_ops(&mut self, ops: &[EditOp]) -> Result<(), Diagnostic> {
        let mut d = self.design.clone();
        panda_kernel::apply_edits(&mut d, ops)?;
        self.design = d;
        self.prune_geometry();
        self.touch();
        Ok(())
    }

    /// Mark changed (bump revision, set dirty).
    pub fn touch(&mut self) {
        self.revision += 1;
        self.dirty = true;
    }

    /// Drop geometry entries that no longer refer to anything.
    fn prune_geometry(&mut self) {
        let design = &self.design;
        self.editors.retain(|id, _| design.def(id).is_some());
        for (id, e) in &mut self.editors {
            let def = design.def(id).expect("retained");
            e.instances
                .retain(|k, _| def.instances.iter().any(|i| &i.id == k));
            e.pins.retain(|k, _| def.pins.iter().any(|p| &p.name == k));
            e.wires.retain(|k, _| def.nets.iter().any(|n| &n.id == k));
        }
    }
}

/// Def ids reachable from the top (top first, then depth-first in instance order).
pub fn reachable_defs(design: &Design) -> Vec<DefId> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    let mut stack = vec![design.top.clone()];
    while let Some(id) = stack.pop() {
        if !seen.insert(id.clone()) {
            continue;
        }
        let Some(d) = design.def(&id) else { continue };
        out.push(id);
        for i in d.instances.iter().rev() {
            if let InstKind::Def(c) = &i.kind
                && !seen.contains(c)
            {
                stack.push(c.clone());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples;

    #[test]
    fn editor_data_round_trips_through_design_json() {
        let mut doc = Document::new(examples::counter());
        doc.layout_missing();
        assert!(doc.defs_missing_layout().is_empty());
        let json = doc.to_json();
        let back = Document::from_json(&json).unwrap();
        assert_eq!(back.editors, doc.editors);
        assert!(back.defs_missing_layout().is_empty());
        // The structure is untouched by geometry.
        let mut a = back.design.clone();
        let mut b = examples::counter();
        for d in a.defs.iter_mut().chain(b.defs.iter_mut()) {
            d.editor = serde_json::Value::Null;
        }
        assert_eq!(a, b);
        // And serializing again is stable.
        assert_eq!(back.to_json(), json);
    }

    #[test]
    fn hand_written_geometry_parses() {
        let v = serde_json::json!({
            "version": 1,
            "instances": {"g1": {"pos": [4, 6], "rot": 1, "flip": true}},
            "pins": {"A": {"pos": [0, 0]}},
            "wires": {"n1": {"polylines": [[[0, 1], [4, 1], [4, 7]]], "junctions": [[4, 1]]}},
            "labels": [{"pos": [1, 1], "text": "hello"}],
            "symbol": {"size": [6, 6], "pins": {"A": {"side": "left", "offset": 3}}}
        });
        let e = EditorData::from_json(&v).unwrap();
        assert_eq!(e.instances["g1"].orient, Orient { rot: 1, flip: true });
        assert_eq!(e.wires["n1"].segments().count(), 2);
        assert_eq!(EditorData::from_json(&e.to_json()).unwrap(), e);
    }

    #[test]
    fn apply_ops_prunes_geometry() {
        let mut doc = Document::new(examples::counter());
        doc.layout_missing();
        let top = doc.design.top.clone();
        let def = doc.def(&top).unwrap().clone();
        let inst = def.instances[0].id.clone();
        assert!(doc.editor(&top).unwrap().instances.contains_key(&inst));
        let rev = doc.revision;
        // Removing an instance that something depends on may leave nets dangling, which
        // apply_ops does not validate; the geometry is pruned either way.
        doc.apply_ops(&[EditOp::RemoveInstance {
            def: top.clone(),
            id: inst.clone(),
        }])
        .unwrap();
        assert!(!doc.editor(&top).unwrap().instances.contains_key(&inst));
        assert!(doc.revision > rev && doc.dirty);
        assert!(
            doc.apply_ops(&[EditOp::RemoveInstance {
                def: top,
                id: "nope".into()
            }])
            .is_err()
        );
    }
}
