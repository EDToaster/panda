//! The editing controller: selection, clipboard, history and every editing action as a pure
//! function of the [`Document`] (no egui, no GPU).
//!
//! Each action computes the structural [`EditOp`]s (submitted through an [`EditTarget`]:
//! document + kernel) and the geometry change, and records one undoable [`Txn`]
//! (see [`History`]). Connectivity rules for wires:
//!
//! * a wire's two ends are classified as a **port** (instance port or the def's own pin), a
//!   point on an **existing wire** of some net, or **free** space ([`anchor_at`]);
//! * a free port and nothing else → `AddNet`; a port onto a net → `Connect`; two different nets
//!   → the smaller one is merged into the larger (`RemoveNet` + `Connect` of its ends);
//! * every net and port involved must have the same width (else an inline error, no change);
//! * deleting wire segments or items splits a net whose wires fall apart into pieces that each
//!   reach endpoints (`Disconnect` + `AddNet` for every piece but the biggest).

use std::collections::{BTreeSet, HashMap, HashSet};

use panda_kernel::EditOp;
use panda_netlist::{
    Atom, ComponentDef, DefId, Design, Dir, Endpoint, InstKind, Instance, Net, PinDecl, Termination,
};

use super::history::{EditTarget, History};
use super::route::{self, Seg, norm, on_segment, rebuild, stretch};
use crate::doc::{Document, EDITOR_VERSION, EditorData, InstGeom, ItemRef, NetGeom, Placed};
use crate::geom::{GridPt, GridRect, Orient, Side, atom_symbol, gp, pin_symbol};

/// What the palette asks to place (see [`crate::edit::request_place`]).
#[derive(Clone, Debug, PartialEq)]
pub enum PlaceRequest {
    /// An atom with these parameters.
    Atom(Atom),
    /// An instance of a def (package) by id.
    Def(DefId),
    /// A new pin of the viewed def (an input or output of the def's interface).
    Pin {
        /// Direction.
        dir: Dir,
        /// Width.
        width: u32,
    },
}

impl PlaceRequest {
    /// Short label ("AND", "NOT", "register_4", "input", …).
    pub fn label(&self) -> String {
        match self {
            PlaceRequest::Atom(a) => atom_label(a),
            PlaceRequest::Def(d) => d.clone(),
            PlaceRequest::Pin { dir: Dir::In, .. } => "input pin".into(),
            PlaceRequest::Pin { .. } => "output pin".into(),
        }
    }
}

/// Display name of an atom kind.
pub fn atom_label(a: &Atom) -> String {
    match a {
        Atom::Gate { op, .. } => format!("{op:?}").to_uppercase(),
        Atom::Not { .. } => "NOT".into(),
        Atom::Buf { .. } => "BUF".into(),
        Atom::Tri { .. } => "TRI".into(),
        Atom::Splitter { .. } => "Splitter".into(),
        Atom::Constant { .. } => "Constant".into(),
        Atom::Clock { .. } => "Clock".into(),
        Atom::PowerOnReset { .. } => "PowerOnReset".into(),
    }
}

/// One selected wire segment (a segment of a net's polyline, points ordered).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WireSeg {
    /// Net id.
    pub net: String,
    /// First point (`a <= b`).
    pub a: GridPt,
    /// Second point.
    pub b: GridPt,
}

impl WireSeg {
    /// Segment of `net` between `a` and `b` (any order).
    pub fn new(net: impl Into<String>, a: GridPt, b: GridPt) -> WireSeg {
        let (a, b) = norm(a, b);
        WireSeg {
            net: net.into(),
            a,
            b,
        }
    }
}

/// The selection of the viewed def.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Selection {
    /// Selected items (instances, pin markers).
    pub items: BTreeSet<ItemRef>,
    /// Selected wire segments.
    pub wires: BTreeSet<WireSeg>,
}

impl Selection {
    /// True if nothing is selected.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty() && self.wires.is_empty()
    }

    /// Select nothing.
    pub fn clear(&mut self) {
        self.items.clear();
        self.wires.clear();
    }

    /// The single selected instance id, if exactly one item (an instance) and no wire is
    /// selected.
    pub fn single_instance(&self) -> Option<&str> {
        match (self.items.len(), self.wires.is_empty()) {
            (1, true) => match self.items.iter().next() {
                Some(ItemRef::Instance(id)) => Some(id),
                _ => None,
            },
            _ => None,
        }
    }

    /// The net if only wire segments of one net are selected.
    pub fn single_net(&self) -> Option<&str> {
        let first = self.wires.iter().next()?;
        (self.items.is_empty() && self.wires.iter().all(|w| w.net == first.net))
            .then_some(first.net.as_str())
    }
}

/// A port of a placed item in world coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct PortInfo {
    /// The item.
    pub item: ItemRef,
    /// Net endpoint naming the port.
    pub ep: Endpoint,
    /// World position.
    pub pos: GridPt,
    /// Boundary side (after orientation).
    pub side: Side,
    /// Direction.
    pub dir: Dir,
    /// Width.
    pub width: u32,
}

/// Ports of a placed item.
pub fn item_ports(p: &Placed) -> Vec<PortInfo> {
    p.symbol
        .ports
        .iter()
        .map(|port| PortInfo {
            item: p.item.clone(),
            ep: match &p.item {
                ItemRef::Pin(n) => Endpoint::Pin(n.clone()),
                ItemRef::Instance(i) => Endpoint::Port {
                    inst: i.clone(),
                    port: port.name.clone(),
                },
            },
            pos: p.geom.pos + p.geom.orient.apply(port.pos, p.symbol.size),
            side: p.geom.orient.side(port.side),
            dir: port.dir,
            width: port.width,
        })
        .collect()
}

/// Every port of every placed item of `def`.
pub fn ports(doc: &Document, def: &str) -> Vec<PortInfo> {
    doc.placed(def).iter().flat_map(item_ports).collect()
}

/// Human-readable endpoint (`pin A`, `g1.in0`).
pub fn ep_label(e: &Endpoint) -> String {
    match e {
        Endpoint::Pin(p) => format!("pin {p}"),
        Endpoint::Port { inst, port } => format!("{inst}.{port}"),
    }
}

/// The net an endpoint is connected to.
pub fn net_of<'a>(def: &'a ComponentDef, e: &Endpoint) -> Option<&'a Net> {
    def.nets.iter().find(|n| n.ends.contains(e))
}

/// What a wire end touches.
#[derive(Clone, Debug, PartialEq)]
pub enum Anchor {
    /// A port, and the net it is connected to (if any).
    Port {
        /// The port.
        port: PortInfo,
        /// Its net.
        net: Option<String>,
    },
    /// A point on a wire of a net.
    Wire {
        /// The net.
        net: String,
    },
    /// Nothing.
    Free,
}

impl Anchor {
    /// The net this anchor belongs to, if any.
    pub fn net(&self) -> Option<&str> {
        match self {
            Anchor::Port { net, .. } => net.as_deref(),
            Anchor::Wire { net } => Some(net),
            Anchor::Free => None,
        }
    }

    /// Width of what the anchor touches (`None` for free space).
    pub fn width(&self, def: &ComponentDef) -> Option<u32> {
        match self {
            Anchor::Port { port, .. } => Some(port.width),
            Anchor::Wire { net } => def.nets.iter().find(|n| &n.id == net).map(|n| n.width),
            Anchor::Free => None,
        }
    }

    /// Wire leg direction at this anchor (ports: perpendicular to their side).
    pub fn axis(&self) -> Option<route::Axis> {
        match self {
            Anchor::Port { port, .. } => Some(route::Axis::of_side(port.side)),
            _ => None,
        }
    }
}

/// Classify grid point `p` of `def`: a port there, else a wire passing through, else free.
pub fn anchor_at(doc: &Document, def_id: &str, p: GridPt) -> Anchor {
    let Some(def) = doc.def(def_id) else {
        return Anchor::Free;
    };
    for placed in doc.placed(def_id) {
        if !placed.rect().contains(p) {
            continue;
        }
        if let Some(port) = item_ports(&placed).into_iter().find(|q| q.pos == p) {
            let net = net_of(def, &port.ep).map(|n| n.id.clone());
            return Anchor::Port { port, net };
        }
    }
    if let Some(ed) = doc.editor(def_id) {
        for n in &def.nets {
            if let Some(g) = ed.wires.get(&n.id)
                && g.segments().any(|s| on_segment(p, s))
            {
                return Anchor::Wire { net: n.id.clone() };
            }
        }
    }
    Anchor::Free
}

/// Check atom parameters (the kernel's validation rules for atoms).
pub fn check_atom(a: &Atom) -> Result<(), String> {
    match a {
        Atom::Gate {
            inputs,
            width,
            reduce,
            ..
        } => {
            if *inputs < 1 {
                return Err("a gate needs at least 1 input".into());
            }
            if *reduce && *inputs != 1 {
                return Err(format!("a reduce gate has exactly 1 input (not {inputs})"));
            }
            if *width == 0 {
                return Err("width must be ≥ 1".into());
            }
        }
        Atom::Not { width } | Atom::Buf { width } | Atom::Tri { width } => {
            if *width == 0 {
                return Err("width must be ≥ 1".into());
            }
        }
        Atom::Splitter { width, parts } => {
            if *width == 0 {
                return Err("width must be ≥ 1".into());
            }
            if parts.is_empty() {
                return Err("a splitter needs at least one part".into());
            }
            for (i, &(lo, hi)) in parts.iter().enumerate() {
                if lo > hi || hi >= *width {
                    return Err(format!(
                        "part p{i} = {hi}:{lo} is out of range for width {width}"
                    ));
                }
            }
        }
        Atom::Constant { value } => {
            if value.width() == 0 {
                return Err("width must be ≥ 1".into());
            }
        }
        Atom::Clock { period, high, .. } => {
            if *period < 2 {
                return Err(format!("period must be ≥ 2 (is {period})"));
            }
            if *high == 0 || *high >= *period {
                return Err(format!("high must be in 1..{period} (is {high})"));
            }
        }
        Atom::PowerOnReset { .. } => {}
    }
    Ok(())
}

fn is_source(a: &Atom) -> bool {
    matches!(a, Atom::Clock { .. } | Atom::PowerOnReset { .. })
}

/// True if def `from` is `target` or (transitively) instantiates it.
pub fn instantiates(design: &Design, from: &str, target: &str) -> bool {
    let mut stack = vec![from.to_string()];
    let mut seen = HashSet::new();
    while let Some(id) = stack.pop() {
        if id == target {
            return true;
        }
        if !seen.insert(id.clone()) {
            continue;
        }
        if let Some(d) = design.def(&id) {
            for i in &d.instances {
                if let InstKind::Def(c) = &i.kind {
                    stack.push(c.clone());
                }
            }
        }
    }
    false
}

/// `base` + the smallest number ≥ 1 such that `taken` is false.
fn fresh_id(base: &str, taken: impl Fn(&str) -> bool) -> String {
    (1..)
        .map(|n| format!("{base}{n}"))
        .find(|s| !taken(s))
        .expect("unbounded")
}

/// Instance id prefix for an atom.
fn atom_base(a: &Atom) -> String {
    match a {
        Atom::Gate { op, .. } => format!("{op:?}").to_lowercase(),
        Atom::Not { .. } => "not".into(),
        Atom::Buf { .. } => "buf".into(),
        Atom::Tri { .. } => "tri".into(),
        Atom::Splitter { .. } => "split".into(),
        Atom::Constant { .. } => "const".into(),
        Atom::Clock { .. } => "clk".into(),
        Atom::PowerOnReset { .. } => "por".into(),
    }
}

/// A valid identifier derived from `s` (no `/`, `[`, `]`), without trailing digits.
fn id_base(s: &str) -> String {
    let t: String = s
        .chars()
        .map(|c| if matches!(c, '/' | '[' | ']') { '_' } else { c })
        .collect();
    let t = t.trim_end_matches(|c: char| c.is_ascii_digit());
    if t.is_empty() || t == "pin" {
        "u".into()
    } else {
        t.to_string()
    }
}

/// Ids handed out within one action (not yet in the def).
#[derive(Default)]
struct Ids {
    inst: HashSet<String>,
    net: HashSet<String>,
}

impl Ids {
    fn instance(&mut self, def: &ComponentDef, base: &str) -> String {
        let id = fresh_id(base, |s| {
            s == "pin" || self.inst.contains(s) || def.instances.iter().any(|i| i.id == s)
        });
        self.inst.insert(id.clone());
        id
    }

    fn net(&mut self, def: &ComponentDef) -> String {
        let id = fresh_id("n", |s| {
            self.net.contains(s)
                || def.nets.iter().any(|n| n.id == s)
                || def.pins.iter().any(|p| p.name == s)
        });
        self.net.insert(id.clone());
        id
    }
}

fn def_of<'a>(doc: &'a Document, id: &str) -> Result<&'a ComponentDef, String> {
    doc.def(id).ok_or_else(|| format!("unknown def '{id}'"))
}

/// Geometry of `def` for a hand edit (created if missing; no longer "auto-layout").
fn editor_mut<'a>(doc: &'a mut Document, def: &str) -> &'a mut EditorData {
    let e = doc
        .editors
        .entry(def.to_string())
        .or_insert_with(|| EditorData {
            version: EDITOR_VERSION,
            ..EditorData::default()
        });
    e.auto = false;
    e
}

fn segs_of(doc: &Document, def: &str, net: &str) -> Vec<Seg> {
    doc.editor(def)
        .and_then(|e| e.wires.get(net))
        .map(|g| g.segments().collect())
        .unwrap_or_default()
}

/// Where endpoints sit (world).
type PortMap = HashMap<Endpoint, GridPt>;

fn port_map(doc: &Document, def: &str) -> PortMap {
    ports(doc, def).into_iter().map(|p| (p.ep, p.pos)).collect()
}

fn end_points(ends: &[Endpoint], pm: &PortMap) -> Vec<GridPt> {
    ends.iter().filter_map(|e| pm.get(e).copied()).collect()
}

/// Where an action that needs a cursor position puts things when there is none.
pub const PASTE_OFFSET: GridPt = gp(2, 2);

/// Items copied by [`Editor::copy`].
#[derive(Clone, Debug, PartialEq)]
pub struct Clipboard {
    /// Instances with their placement.
    pub items: Vec<(Instance, InstGeom)>,
    /// Nets whose every endpoint is on a copied instance, with their wires.
    pub nets: Vec<(Net, NetGeom)>,
    /// Top-left of the copied items' bounding box.
    pub origin: GridPt,
}

/// The current editing tool.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Tool {
    /// Select, move, box-select; dragging from a port draws a wire.
    #[default]
    Select,
    /// Draw wires (drag from anywhere on a port or wire).
    Wire,
    /// Primary drag pans the view (no editing).
    Pan,
    /// Place an item at the next click / drop.
    Place(PlaceRequest),
}

/// Editing state and actions (pure: works on any [`EditTarget`]).
#[derive(Debug, Default)]
pub struct Editor {
    /// Current tool.
    pub tool: Tool,
    /// Selection in the viewed def.
    pub selection: Selection,
    /// Undo/redo.
    pub history: History,
    /// Copied items.
    pub clipboard: Option<Clipboard>,
    /// Last message for the user (errors from actions), newest wins.
    pub message: Option<String>,
    /// One-shot request (F2) to focus the inspector's name field with the whole name selected,
    /// ready to type a new one. Taken by the inspector when it draws the field.
    pub focus_name: bool,
    view: Option<(String, DefId)>,
    revision: u64,
    /// [`Document::id`] of the document the history and selection belong to.
    doc_id: u64,
}

impl Editor {
    /// A fresh editor.
    pub fn new() -> Editor {
        Editor::default()
    }

    /// Keep the selection consistent with the view (`path`, `def`) and the document: clears it
    /// when the view changes, drops items that no longer exist after other changes. Clears the
    /// history too when a different document was opened.
    pub fn sync(&mut self, doc: &Document, path: &str, def: &str) {
        // Another document was opened: its history and selection are not this one's.
        if doc.id != self.doc_id {
            if self.doc_id != 0 {
                self.history.clear();
                self.selection.clear();
                self.view = None;
            }
            self.doc_id = doc.id;
        }
        let view = (path.to_string(), def.to_string());
        if self.view.as_ref() != Some(&view) {
            self.view = Some(view);
            self.selection.clear();
            self.revision = doc.revision;
            return;
        }
        if self.revision == doc.revision {
            return;
        }
        self.revision = doc.revision;
        let Some(d) = doc.def(def) else {
            self.selection.clear();
            return;
        };
        self.selection.items.retain(|it| match it {
            ItemRef::Instance(id) => d.instances.iter().any(|i| &i.id == id),
            ItemRef::Pin(n) => d.pins.iter().any(|p| &p.name == n),
        });
        let ed = doc.editor(def);
        self.selection.wires.retain(|w| {
            ed.and_then(|e| e.wires.get(&w.net))
                .is_some_and(|g| g.segments().any(|(a, b)| norm(a, b) == (w.a, w.b)))
        });
    }

    /// Record an error / info for the user and return it as `Err`.
    fn fail<T>(&mut self, e: String) -> Result<T, String> {
        self.message = Some(e.clone());
        Err(e)
    }

    fn done(&mut self, r: Result<(), String>) -> Result<(), String> {
        match r {
            Ok(()) => {
                self.message = None;
                Ok(())
            }
            Err(e) => self.fail(e),
        }
    }

    // ----------------------------------------------------------------------------------
    // Selection
    // ----------------------------------------------------------------------------------

    /// Click selection: `item` / `wire` under the pointer (item wins). Without `shift` the
    /// selection becomes just that thing (or empty); with `shift` it is toggled.
    pub fn click_select(&mut self, item: Option<ItemRef>, wire: Option<WireSeg>, shift: bool) {
        if !shift {
            self.selection.clear();
        }
        if let Some(it) = item {
            if !self.selection.items.remove(&it) {
                self.selection.items.insert(it);
            }
        } else if let Some(w) = wire
            && !self.selection.wires.remove(&w)
        {
            self.selection.wires.insert(w);
        }
    }

    /// Box selection: items whose box and wire segments that lie entirely inside `r`.
    pub fn box_select(&mut self, doc: &Document, def: &str, r: GridRect, shift: bool) {
        if !shift {
            self.selection.clear();
        }
        for p in doc.placed(def) {
            let b = p.rect();
            if r.contains(b.min) && r.contains(b.max) {
                self.selection.items.insert(p.item.clone());
            }
        }
        if let Some(ed) = doc.editor(def) {
            for (net, g) in &ed.wires {
                for (a, b) in g.segments() {
                    if r.contains(a) && r.contains(b) {
                        self.selection.wires.insert(WireSeg::new(net.clone(), a, b));
                    }
                }
            }
        }
    }

    /// Select every item and wire of `def`.
    pub fn select_all(&mut self, doc: &Document, def: &str) {
        let r = GridRect {
            min: gp(i32::MIN / 2, i32::MIN / 2),
            max: gp(i32::MAX / 2, i32::MAX / 2),
        };
        self.box_select(doc, def, r, false);
    }

    // ----------------------------------------------------------------------------------
    // Place
    // ----------------------------------------------------------------------------------

    /// Place `req` centred at grid point `at` and select it.
    pub fn place(
        &mut self,
        t: &mut dyn EditTarget,
        def_id: &str,
        req: &PlaceRequest,
        at: GridPt,
    ) -> Result<ItemRef, String> {
        let r = self.place_inner(t, def_id, req, at);
        match r {
            Ok(it) => {
                self.message = None;
                self.selection.clear();
                self.selection.items.insert(it.clone());
                Ok(it)
            }
            Err(e) => self.fail(e),
        }
    }

    fn place_inner(
        &mut self,
        t: &mut dyn EditTarget,
        def_id: &str,
        req: &PlaceRequest,
        at: GridPt,
    ) -> Result<ItemRef, String> {
        let doc = t.doc();
        let def = def_of(doc, def_id)?;
        let mut ids = Ids::default();
        let (op, item, size) = match req {
            PlaceRequest::Atom(a) => {
                check_atom(a)?;
                if is_source(a) && doc.design.top != def_id {
                    return Err(format!(
                        "{} can only be placed in the top def",
                        atom_label(a)
                    ));
                }
                let id = ids.instance(def, &atom_base(a));
                let size = atom_symbol(a).size;
                (
                    EditOp::AddInstance {
                        def: def_id.to_string(),
                        instance: Instance {
                            id: id.clone(),
                            kind: InstKind::Atom(a.clone()),
                        },
                    },
                    ItemRef::Instance(id),
                    size,
                )
            }
            PlaceRequest::Def(d) => {
                let target = def_of(doc, d)?;
                if instantiates(&doc.design, d, def_id) {
                    return Err(format!(
                        "'{d}' contains '{def_id}': placing it here would be recursive"
                    ));
                }
                let base = id_base(if target.name.is_empty() {
                    &target.id
                } else {
                    &target.name
                });
                let id = ids.instance(def, &base);
                let size = doc.package_symbol(d).expect("def exists").size;
                (
                    EditOp::AddInstance {
                        def: def_id.to_string(),
                        instance: Instance {
                            id: id.clone(),
                            kind: InstKind::Def(d.clone()),
                        },
                    },
                    ItemRef::Instance(id),
                    size,
                )
            }
            PlaceRequest::Pin { dir, width } => {
                if *width == 0 {
                    return Err("width must be ≥ 1".into());
                }
                let base = if *dir == Dir::In { "in" } else { "out" };
                let name = fresh_id(base, |s| {
                    def.pins.iter().any(|p| p.name == s) || def.nets.iter().any(|n| n.id == s)
                });
                let pin = PinDecl {
                    name: name.clone(),
                    dir: *dir,
                    width: *width,
                };
                let size = pin_symbol(&pin).size;
                (
                    EditOp::AddPin {
                        def: def_id.to_string(),
                        pin,
                    },
                    ItemRef::Pin(name),
                    size,
                )
            }
        };
        let pos = at - gp(size.x / 2, size.y / 2);
        let label = format!("place {}", req.label());
        let it = item.clone();
        self.history
            .commit(t, label, &[def_id.to_string()], vec![op], |doc| {
                let ed = editor_mut(doc, def_id);
                let g = InstGeom {
                    pos,
                    orient: Orient::default(),
                };
                match &it {
                    ItemRef::Instance(id) => ed.instances.insert(id.clone(), g),
                    ItemRef::Pin(n) => ed.pins.insert(n.clone(), g),
                };
            })?;
        Ok(item)
    }

    // ----------------------------------------------------------------------------------
    // Wires
    // ----------------------------------------------------------------------------------

    /// Draw a wire along the orthogonal polyline `pts` and connect what its two ends touch.
    pub fn draw_wire(
        &mut self,
        t: &mut dyn EditTarget,
        def_id: &str,
        pts: &[GridPt],
    ) -> Result<(), String> {
        let r = self.draw_wire_inner(t, def_id, pts);
        self.done(r)
    }

    fn draw_wire_inner(
        &mut self,
        t: &mut dyn EditTarget,
        def_id: &str,
        pts: &[GridPt],
    ) -> Result<(), String> {
        let pts = route::simplify(pts);
        if pts.len() < 2 {
            return Err("the wire is too short".into());
        }
        if pts.windows(2).any(|w| w[0].x != w[1].x && w[0].y != w[1].y) {
            return Err("wires must be horizontal / vertical".into());
        }
        let doc = t.doc();
        let def = def_of(doc, def_id)?;
        let ends = [
            anchor_at(doc, def_id, pts[0]),
            anchor_at(doc, def_id, *pts.last().expect("≥ 2 points")),
        ];
        let plan = plan_connect(def, &ends)?;
        let pm = port_map(doc, def_id);
        let mut segs: Vec<Seg> = route::segments_of(&pts);
        for n in std::iter::once(&plan.keep).chain(&plan.absorbed) {
            segs.extend(segs_of(doc, def_id, n));
        }
        // Endpoints of the resulting net (for junction dots at ports).
        let mut ends_after: Vec<Endpoint> = def
            .nets
            .iter()
            .filter(|n| n.id == plan.keep || plan.absorbed.contains(&n.id))
            .flat_map(|n| n.ends.iter().cloned())
            .collect();
        ends_after.extend(plan.connected.iter().cloned());
        let geom = rebuild(segs, &end_points(&ends_after, &pm));
        let keep = plan.keep.clone();
        let absorbed = plan.absorbed.clone();
        self.history
            .commit(t, plan.label, &[def_id.to_string()], plan.ops, |doc| {
                let ed = editor_mut(doc, def_id);
                for a in &absorbed {
                    ed.wires.remove(a);
                }
                ed.wires.insert(keep, geom);
            })
    }

    // ----------------------------------------------------------------------------------
    // Delete
    // ----------------------------------------------------------------------------------

    /// Delete the selected items and wire segments. Nets whose wires fall apart into pieces
    /// that each reach endpoints are split; nets left without endpoints and wires are removed.
    pub fn delete_selection(&mut self, t: &mut dyn EditTarget, def_id: &str) -> Result<(), String> {
        if self.selection.is_empty() {
            return Ok(());
        }
        let sel = self.selection.clone();
        let r = self.delete_inner(t, def_id, &sel);
        if r.is_ok() {
            self.selection.clear();
        }
        self.done(r)
    }

    fn delete_inner(
        &mut self,
        t: &mut dyn EditTarget,
        def_id: &str,
        sel: &Selection,
    ) -> Result<(), String> {
        let doc = t.doc();
        let def = def_of(doc, def_id)?;
        let mut ops = Vec::new();
        let mut touched = vec![def_id.to_string()];
        let mut gone_insts: HashSet<&str> = HashSet::new();
        let mut gone_pins: HashSet<&str> = HashSet::new();
        for it in &sel.items {
            match it {
                ItemRef::Instance(id) => {
                    if def.instances.iter().any(|i| &i.id == id) {
                        gone_insts.insert(id);
                        ops.push(EditOp::RemoveInstance {
                            def: def_id.to_string(),
                            id: id.clone(),
                        });
                    }
                }
                ItemRef::Pin(name) => {
                    if def.pins.iter().any(|p| &p.name == name) {
                        gone_pins.insert(name);
                        ops.push(EditOp::RemovePin {
                            def: def_id.to_string(),
                            name: name.clone(),
                        });
                        // Uses of the pin on instances of this def elsewhere.
                        for parent in &doc.design.defs {
                            let mut hit = false;
                            for inst in &parent.instances {
                                if !matches!(&inst.kind, InstKind::Def(d) if d == def_id) {
                                    continue;
                                }
                                let ep = Endpoint::Port {
                                    inst: inst.id.clone(),
                                    port: name.clone(),
                                };
                                if let Some(n) = net_of(parent, &ep) {
                                    hit = true;
                                    ops.push(EditOp::Disconnect {
                                        def: parent.id.clone(),
                                        net: n.id.clone(),
                                        endpoint: ep,
                                    });
                                }
                            }
                            if hit && !touched.contains(&parent.id) {
                                touched.push(parent.id.clone());
                            }
                        }
                    }
                }
            }
        }
        let mut seg_rm: HashMap<&str, HashSet<Seg>> = HashMap::new();
        for w in &sel.wires {
            seg_rm.entry(&w.net).or_default().insert((w.a, w.b));
        }
        let pm = port_map(doc, def_id);
        let mut ids = Ids::default();
        let mut wires: Vec<(String, Option<NetGeom>)> = Vec::new();
        for n in &def.nets {
            let ends: Vec<Endpoint> = n
                .ends
                .iter()
                .filter(|e| match e {
                    Endpoint::Pin(p) => !gone_pins.contains(p.as_str()),
                    Endpoint::Port { inst, .. } => !gone_insts.contains(inst.as_str()),
                })
                .cloned()
                .collect();
            let rm = seg_rm.get(n.id.as_str());
            if rm.is_none() && ends.len() == n.ends.len() {
                continue;
            }
            let segs: Vec<Seg> = segs_of(doc, def_id, &n.id)
                .into_iter()
                .filter(|&(a, b)| rm.is_none_or(|r| !r.contains(&norm(a, b))))
                .collect();
            if ends.is_empty() && segs.is_empty() {
                ops.push(EditOp::RemoveNet {
                    def: def_id.to_string(),
                    id: n.id.clone(),
                });
                wires.push((n.id.clone(), None));
                continue;
            }
            // Which pieces of wire reach which endpoints.
            let comps = route::components(&segs);
            let mut comp_ends: Vec<Vec<Endpoint>> = vec![Vec::new(); comps.len()];
            for e in &ends {
                let Some(&p) = pm.get(e) else { continue };
                if let Some(k) = comps
                    .iter()
                    .position(|c| c.iter().any(|&i| on_segment(p, segs[i])))
                {
                    comp_ends[k].push(e.clone());
                }
            }
            let with_ends: Vec<usize> = (0..comps.len())
                .filter(|&k| !comp_ends[k].is_empty())
                .collect();
            if with_ends.len() <= 1 {
                wires.push((n.id.clone(), Some(rebuild(segs, &end_points(&ends, &pm)))));
                continue;
            }
            let keep = *with_ends
                .iter()
                .max_by_key(|&&k| (comp_ends[k].len(), std::cmp::Reverse(k)))
                .expect("non-empty");
            let mut keep_segs = Vec::new();
            let mut keep_ends: Vec<Endpoint> = ends.clone();
            for (k, comp) in comps.iter().enumerate() {
                let csegs: Vec<Seg> = comp.iter().map(|&i| segs[i]).collect();
                if k == keep || comp_ends[k].is_empty() {
                    keep_segs.extend(csegs);
                    continue;
                }
                let id = ids.net(def);
                for e in &comp_ends[k] {
                    keep_ends.retain(|x| x != e);
                    ops.push(EditOp::Disconnect {
                        def: def_id.to_string(),
                        net: n.id.clone(),
                        endpoint: e.clone(),
                    });
                }
                ops.push(EditOp::AddNet {
                    def: def_id.to_string(),
                    net: Net {
                        id: id.clone(),
                        width: n.width,
                        termination: n.termination,
                        ends: comp_ends[k].clone(),
                    },
                });
                wires.push((id, Some(rebuild(csegs, &end_points(&comp_ends[k], &pm)))));
            }
            wires.push((
                n.id.clone(),
                Some(rebuild(keep_segs, &end_points(&keep_ends, &pm))),
            ));
        }
        let n = sel.items.len() + sel.wires.len();
        let label = if n == 1 {
            "delete".to_string()
        } else {
            format!("delete {n} things")
        };
        let def_owned = def_id.to_string();
        self.history.commit(t, label, &touched, ops, move |doc| {
            let ed = editor_mut(doc, &def_owned);
            for (id, g) in wires {
                match g {
                    Some(g) if !g.polylines.is_empty() => {
                        ed.wires.insert(id, g);
                    }
                    _ => {
                        ed.wires.remove(&id);
                    }
                }
            }
        })
    }

    // ----------------------------------------------------------------------------------
    // Move / rotate / flip
    // ----------------------------------------------------------------------------------

    /// Move the selection by `delta` grid units. Wires attached to moved ports follow
    /// (rubber band, kept orthogonal); selected wire segments move along.
    pub fn move_selection(
        &mut self,
        t: &mut dyn EditTarget,
        def_id: &str,
        delta: GridPt,
    ) -> Result<(), String> {
        if delta == gp(0, 0) || self.selection.is_empty() {
            return Ok(());
        }
        let sel = self.selection.clone();
        let doc = t.doc();
        let placed = doc.placed(def_id);
        let new_geom: HashMap<ItemRef, InstGeom> = placed
            .iter()
            .filter(|p| sel.items.contains(&p.item))
            .map(|p| {
                (
                    p.item.clone(),
                    InstGeom {
                        pos: p.geom.pos + delta,
                        orient: p.geom.orient,
                    },
                )
            })
            .collect();
        let mut extra: HashMap<String, Vec<(GridPt, GridPt)>> = HashMap::new();
        for w in &sel.wires {
            let e = extra.entry(w.net.clone()).or_default();
            e.push((w.a, w.a + delta));
            e.push((w.b, w.b + delta));
        }
        let n = sel.items.len() + sel.wires.len();
        let label = if n == 1 {
            "move".to_string()
        } else {
            format!("move {n} things")
        };
        let r = self.regeometry(t, def_id, &placed, &new_geom, extra, label);
        if r.is_ok() {
            // Keep the moved segments selected.
            self.selection.wires = sel
                .wires
                .iter()
                .map(|w| WireSeg::new(w.net.clone(), w.a + delta, w.b + delta))
                .collect();
            self.revision = t.doc().revision;
        }
        self.done(r)
    }

    /// Rotate the selected items by a quarter turn (clockwise or counter-clockwise) about
    /// their own centres.
    pub fn rotate_selection(
        &mut self,
        t: &mut dyn EditTarget,
        def_id: &str,
        clockwise: bool,
    ) -> Result<(), String> {
        self.reorient(t, def_id, "rotate", |o| Orient {
            rot: (o.rot + if clockwise { 1 } else { 3 }) % 4,
            flip: o.flip,
        })
    }

    /// Mirror the selected items horizontally about their own centres.
    pub fn flip_selection(&mut self, t: &mut dyn EditTarget, def_id: &str) -> Result<(), String> {
        // Mirroring the rotated symbol: M·R^k·F^f = R^-k·F^(f+1).
        self.reorient(t, def_id, "flip", |o| Orient {
            rot: (4 - o.rot % 4) % 4,
            flip: !o.flip,
        })
    }

    fn reorient(
        &mut self,
        t: &mut dyn EditTarget,
        def_id: &str,
        label: &str,
        f: impl Fn(Orient) -> Orient,
    ) -> Result<(), String> {
        if self.selection.items.is_empty() {
            return Ok(());
        }
        let doc = t.doc();
        let placed = doc.placed(def_id);
        let new_geom: HashMap<ItemRef, InstGeom> = placed
            .iter()
            .filter(|p| self.selection.items.contains(&p.item))
            .map(|p| {
                let o = f(p.geom.orient);
                let (s0, s1) = (p.geom.orient.size(p.symbol.size), o.size(p.symbol.size));
                let pos = p.geom.pos + gp((s0.x - s1.x) / 2, (s0.y - s1.y) / 2);
                (p.item.clone(), InstGeom { pos, orient: o })
            })
            .collect();
        let r = self.regeometry(t, def_id, &placed, &new_geom, HashMap::new(), label.into());
        self.done(r)
    }

    /// Commit new placements (`new_geom`) and rubber-band the wires attached to ports that
    /// moved; `extra` adds more moved wire points per net.
    fn regeometry(
        &mut self,
        t: &mut dyn EditTarget,
        def_id: &str,
        placed: &[Placed],
        new_geom: &HashMap<ItemRef, InstGeom>,
        extra: HashMap<String, Vec<(GridPt, GridPt)>>,
        label: String,
    ) -> Result<(), String> {
        let doc = t.doc();
        let def = def_of(doc, def_id)?;
        // Port positions before and after.
        let mut pm_old = PortMap::new();
        let mut pm_new = PortMap::new();
        for p in placed {
            let old = item_ports(p);
            match new_geom.get(&p.item) {
                Some(g) => {
                    let q = Placed {
                        geom: *g,
                        ..p.clone()
                    };
                    for (a, b) in old.into_iter().zip(item_ports(&q)) {
                        pm_old.insert(a.ep.clone(), a.pos);
                        pm_new.insert(b.ep, b.pos);
                    }
                }
                None => {
                    for a in old {
                        pm_old.insert(a.ep.clone(), a.pos);
                        pm_new.insert(a.ep, a.pos);
                    }
                }
            }
        }
        let mut moved: HashMap<String, HashMap<GridPt, GridPt>> = HashMap::new();
        for n in &def.nets {
            for e in &n.ends {
                if let (Some(&a), Some(&b)) = (pm_old.get(e), pm_new.get(e))
                    && a != b
                {
                    moved.entry(n.id.clone()).or_default().insert(a, b);
                }
            }
        }
        // Moved wire points sitting on a port that stays put get a link back to the port, so
        // the wire stays attached.
        let mut links: HashMap<String, Vec<Seg>> = HashMap::new();
        for (net, pts) in extra {
            if let Some(n) = def.nets.iter().find(|n| n.id == net) {
                let fixed: HashSet<GridPt> = n
                    .ends
                    .iter()
                    .filter_map(|e| pm_old.get(e).filter(|&p| pm_new.get(e) == Some(p)))
                    .copied()
                    .collect();
                for &(a, b) in &pts {
                    if fixed.contains(&a) && a != b {
                        links
                            .entry(net.clone())
                            .or_default()
                            .extend(route::segments_of(&route::route(a, None, b, None)));
                    }
                }
            }
            moved.entry(net).or_default().extend(pts);
        }
        let mut wires: Vec<(String, NetGeom)> = Vec::new();
        for n in &def.nets {
            let Some(m) = moved.get(&n.id) else { continue };
            let segs = segs_of(doc, def_id, &n.id);
            if segs.is_empty() {
                continue;
            }
            let mut segs = stretch(&segs, m);
            segs.extend(links.remove(&n.id).unwrap_or_default());
            wires.push((n.id.clone(), rebuild(segs, &end_points(&n.ends, &pm_new))));
        }
        let new_geom = new_geom.clone();
        let def_owned = def_id.to_string();
        self.history
            .commit(t, label, &[def_id.to_string()], Vec::new(), move |doc| {
                let ed = editor_mut(doc, &def_owned);
                for (it, g) in new_geom {
                    match it {
                        ItemRef::Instance(id) => ed.instances.insert(id, g),
                        ItemRef::Pin(n) => ed.pins.insert(n, g),
                    };
                }
                for (id, g) in wires {
                    ed.wires.insert(id, g);
                }
            })
    }

    // ----------------------------------------------------------------------------------
    // Clipboard
    // ----------------------------------------------------------------------------------

    /// Copy the selected instances (pins are not copied) and the nets entirely between them.
    /// Returns the number of copied instances.
    pub fn copy(&mut self, doc: &Document, def_id: &str) -> usize {
        let Some(def) = doc.def(def_id) else { return 0 };
        let ed = doc.editor(def_id);
        let insts: HashSet<&str> = self
            .selection
            .items
            .iter()
            .filter_map(|it| match it {
                ItemRef::Instance(id) => Some(id.as_str()),
                ItemRef::Pin(_) => None,
            })
            .collect();
        if insts.is_empty() {
            return 0;
        }
        let placed = doc.placed(def_id);
        let mut items = Vec::new();
        let mut origin: Option<GridPt> = None;
        for inst in &def.instances {
            if !insts.contains(inst.id.as_str()) {
                continue;
            }
            let Some(p) = placed
                .iter()
                .find(|p| p.item == ItemRef::Instance(inst.id.clone()))
            else {
                continue;
            };
            let r = p.rect();
            origin = Some(origin.map_or(r.min, |o| gp(o.x.min(r.min.x), o.y.min(r.min.y))));
            items.push((inst.clone(), p.geom));
        }
        let nets = def
            .nets
            .iter()
            .filter(|n| {
                !n.ends.is_empty()
                    && n.ends.iter().all(|e| {
                        matches!(e, Endpoint::Port { inst, .. } if insts.contains(inst.as_str()))
                    })
            })
            .map(|n| {
                let g = ed
                    .and_then(|e| e.wires.get(&n.id))
                    .cloned()
                    .unwrap_or_default();
                (n.clone(), g)
            })
            .collect();
        let count = items.len();
        self.clipboard = Some(Clipboard {
            items,
            nets,
            origin: origin.unwrap_or_default(),
        });
        count
    }

    /// Copy, then delete the selection.
    pub fn cut(&mut self, t: &mut dyn EditTarget, def_id: &str) -> Result<(), String> {
        self.copy(t.doc(), def_id);
        self.delete_selection(t, def_id)
    }

    /// Paste the clipboard with its top-left at `at` (or offset from the original position),
    /// with fresh ids, and select the pasted items.
    pub fn paste(
        &mut self,
        t: &mut dyn EditTarget,
        def_id: &str,
        at: Option<GridPt>,
    ) -> Result<(), String> {
        let Some(clip) = self.clipboard.clone() else {
            return Ok(());
        };
        let r = self.paste_inner(t, def_id, &clip, at);
        match r {
            Ok(items) => {
                self.message = None;
                self.selection.clear();
                self.selection.items.extend(items);
                Ok(())
            }
            Err(e) => self.fail(e),
        }
    }

    fn paste_inner(
        &mut self,
        t: &mut dyn EditTarget,
        def_id: &str,
        clip: &Clipboard,
        at: Option<GridPt>,
    ) -> Result<Vec<ItemRef>, String> {
        let doc = t.doc();
        let def = def_of(doc, def_id)?;
        let offset = at.map_or(PASTE_OFFSET, |a| a - clip.origin);
        let mut ids = Ids::default();
        let mut rename: HashMap<String, String> = HashMap::new();
        let mut ops = Vec::new();
        let mut geoms = Vec::new();
        for (inst, g) in &clip.items {
            match &inst.kind {
                InstKind::Atom(a) if is_source(a) && doc.design.top != def_id => {
                    return Err(format!(
                        "{} can only be placed in the top def",
                        atom_label(a)
                    ));
                }
                InstKind::Def(d) if instantiates(&doc.design, d, def_id) => {
                    return Err(format!("pasting '{d}' into '{def_id}' would be recursive"));
                }
                InstKind::Def(d) if doc.def(d).is_none() => {
                    return Err(format!("unknown def '{d}'"));
                }
                _ => {}
            }
            let id = ids.instance(def, &id_base(&inst.id));
            rename.insert(inst.id.clone(), id.clone());
            ops.push(EditOp::AddInstance {
                def: def_id.to_string(),
                instance: Instance {
                    id: id.clone(),
                    kind: inst.kind.clone(),
                },
            });
            geoms.push((
                id,
                InstGeom {
                    pos: g.pos + offset,
                    orient: g.orient,
                },
            ));
        }
        let mut wires = Vec::new();
        for (net, g) in &clip.nets {
            let id = ids.net(def);
            let ends = net
                .ends
                .iter()
                .map(|e| match e {
                    Endpoint::Port { inst, port } => Endpoint::Port {
                        inst: rename[inst].clone(),
                        port: port.clone(),
                    },
                    Endpoint::Pin(p) => Endpoint::Pin(p.clone()),
                })
                .collect();
            ops.push(EditOp::AddNet {
                def: def_id.to_string(),
                net: Net {
                    id: id.clone(),
                    width: net.width,
                    termination: net.termination,
                    ends,
                },
            });
            let g = NetGeom {
                polylines: g
                    .polylines
                    .iter()
                    .map(|l| l.iter().map(|&p| p + offset).collect())
                    .collect(),
                junctions: g.junctions.iter().map(|&p| p + offset).collect(),
            };
            wires.push((id, g));
        }
        let items: Vec<ItemRef> = geoms
            .iter()
            .map(|(id, _)| ItemRef::Instance(id.clone()))
            .collect();
        let label = match geoms.len() {
            1 => "paste 1 item".to_string(),
            n => format!("paste {n} items"),
        };
        let def_owned = def_id.to_string();
        self.history
            .commit(t, label, &[def_id.to_string()], ops, move |doc| {
                let ed = editor_mut(doc, &def_owned);
                for (id, g) in geoms {
                    ed.instances.insert(id, g);
                }
                for (id, g) in wires {
                    ed.wires.insert(id, g);
                }
            })?;
        Ok(items)
    }

    /// Copy and paste right away (offset by [`PASTE_OFFSET`]).
    pub fn duplicate(&mut self, t: &mut dyn EditTarget, def_id: &str) -> Result<(), String> {
        let saved = self.clipboard.take();
        let r = if self.copy(t.doc(), def_id) > 0 {
            let at = self.clipboard.as_ref().map(|c| c.origin + PASTE_OFFSET);
            self.paste(t, def_id, at)
        } else {
            Ok(())
        };
        self.clipboard = saved.or(self.clipboard.take());
        r
    }

    // ----------------------------------------------------------------------------------
    // Inspector actions
    // ----------------------------------------------------------------------------------

    /// Change the parameters of atom instance `inst`. Connections to ports that no longer
    /// exist are dropped. Connections whose net width no longer matches the port are an error
    /// unless `disconnect_mismatched`, which drops them too. Wires follow moved ports.
    pub fn set_atom(
        &mut self,
        t: &mut dyn EditTarget,
        def_id: &str,
        inst: &str,
        atom: &Atom,
        disconnect_mismatched: bool,
    ) -> Result<(), String> {
        let r = self.set_atom_inner(t, def_id, inst, atom, disconnect_mismatched);
        self.done(r)
    }

    fn set_atom_inner(
        &mut self,
        t: &mut dyn EditTarget,
        def_id: &str,
        inst: &str,
        atom: &Atom,
        disconnect_mismatched: bool,
    ) -> Result<(), String> {
        let doc = t.doc();
        let def = def_of(doc, def_id)?;
        let plan = plan_set_atom(doc, def_id, inst, atom)?;
        if !plan.mismatched.is_empty() && !disconnect_mismatched {
            return Err(plan.conflict_message());
        }
        let mut ops: Vec<EditOp> = plan
            .vanished
            .iter()
            .chain(&plan.mismatched)
            .map(|(net, ep)| EditOp::Disconnect {
                def: def_id.to_string(),
                net: net.clone(),
                endpoint: ep.clone(),
            })
            .collect();
        ops.push(EditOp::SetAtom {
            def: def_id.to_string(),
            instance: inst.to_string(),
            atom: atom.clone(),
        });
        // Wires attached to ports that keep their connection follow the new port positions.
        let placed = doc.placed(def_id);
        let me = ItemRef::Instance(inst.to_string());
        let mut wires = Vec::new();
        if let Some(p) = placed.iter().find(|p| p.item == me) {
            let new = Placed {
                symbol: atom_symbol(atom),
                ..p.clone()
            };
            let old_pos: HashMap<Endpoint, GridPt> =
                item_ports(p).into_iter().map(|q| (q.ep, q.pos)).collect();
            let new_pos: HashMap<Endpoint, GridPt> = item_ports(&new)
                .into_iter()
                .map(|q| (q.ep, q.pos))
                .collect();
            let mut pm = port_map(doc, def_id);
            pm.retain(|e, _| !old_pos.contains_key(e));
            pm.extend(new_pos.iter().map(|(e, &p)| (e.clone(), p)));
            let dropped: HashSet<&Endpoint> = plan
                .vanished
                .iter()
                .chain(&plan.mismatched)
                .map(|(_, e)| e)
                .collect();
            for n in &def.nets {
                let mut m = HashMap::new();
                for e in &n.ends {
                    if dropped.contains(e) {
                        continue;
                    }
                    if let (Some(&a), Some(&b)) = (old_pos.get(e), new_pos.get(e))
                        && a != b
                    {
                        m.insert(a, b);
                    }
                }
                if m.is_empty() {
                    continue;
                }
                let ends: Vec<Endpoint> = n
                    .ends
                    .iter()
                    .filter(|e| !dropped.contains(e))
                    .cloned()
                    .collect();
                wires.push((
                    n.id.clone(),
                    rebuild(
                        stretch(&segs_of(doc, def_id, &n.id), &m),
                        &end_points(&ends, &pm),
                    ),
                ));
            }
        }
        let label = format!("edit {inst}");
        let def_owned = def_id.to_string();
        self.history
            .commit(t, label, &[def_id.to_string()], ops, move |doc| {
                let ed = editor_mut(doc, &def_owned);
                for (id, g) in wires {
                    ed.wires.insert(id, g);
                }
            })
    }

    /// Change a net's termination.
    pub fn set_termination(
        &mut self,
        t: &mut dyn EditTarget,
        def_id: &str,
        net: &str,
        termination: Termination,
    ) -> Result<(), String> {
        let op = EditOp::SetTermination {
            def: def_id.to_string(),
            net: net.to_string(),
            termination,
        };
        let r = self.history.commit(
            t,
            format!("termination of {net}"),
            &[def_id.to_string()],
            vec![op],
            |_| {},
        );
        self.done(r)
    }

    // ----------------------------------------------------------------------------------
    // History
    // ----------------------------------------------------------------------------------

    /// Undo the last action.
    pub fn undo(&mut self, t: &mut dyn EditTarget) -> Result<Option<String>, String> {
        let r = self.history.undo(t);
        self.after_history(t, r)
    }

    /// Redo the last undone action.
    pub fn redo(&mut self, t: &mut dyn EditTarget) -> Result<Option<String>, String> {
        let r = self.history.redo(t);
        self.after_history(t, r)
    }

    fn after_history(
        &mut self,
        t: &mut dyn EditTarget,
        r: Result<Option<String>, String>,
    ) -> Result<Option<String>, String> {
        // Force a selection prune on the next sync.
        self.revision = t.doc().revision.wrapping_sub(1);
        match r {
            Ok(l) => {
                self.message = None;
                Ok(l)
            }
            Err(e) => self.fail(e),
        }
    }

    /// The kernel rejected edit command `id` (validation error): roll back the action that
    /// sent it. Returns true if it was one of ours.
    pub fn kernel_rejected(&mut self, t: &mut dyn EditTarget, id: u64, why: &str) -> bool {
        match self.history.rejected(t, id) {
            Some(label) => {
                self.message = Some(format!("'{label}' was rejected and undone: {why}"));
                self.revision = t.doc().revision.wrapping_sub(1);
                true
            }
            None => false,
        }
    }
}

/// Result of [`plan_set_atom`]: connections affected by new atom parameters.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AtomPlan {
    /// `(net, endpoint)` of connections to ports that no longer exist.
    pub vanished: Vec<(String, Endpoint)>,
    /// `(net, endpoint)` of connections whose net width differs from the new port width.
    pub mismatched: Vec<(String, Endpoint)>,
    /// Description of each mismatch.
    pub conflicts: Vec<String>,
}

impl AtomPlan {
    /// One-line description of the width conflicts.
    pub fn conflict_message(&self) -> String {
        format!("width mismatch: {}", self.conflicts.join("; "))
    }
}

/// Check new parameters for atom instance `inst` against its connections.
pub fn plan_set_atom(
    doc: &Document,
    def_id: &str,
    inst: &str,
    atom: &Atom,
) -> Result<AtomPlan, String> {
    check_atom(atom)?;
    let def = def_of(doc, def_id)?;
    let i = def
        .instances
        .iter()
        .find(|i| i.id == inst)
        .ok_or_else(|| format!("no instance '{inst}'"))?;
    if !matches!(i.kind, InstKind::Atom(_)) {
        return Err(format!("'{inst}' is not an atom"));
    }
    if is_source(atom) && doc.design.top != def_id {
        return Err(format!(
            "{} can only be used in the top def",
            atom_label(atom)
        ));
    }
    let widths: HashMap<String, u32> = atom
        .ports()
        .into_iter()
        .map(|p| (p.name, p.width))
        .collect();
    let mut plan = AtomPlan::default();
    for n in &def.nets {
        for e in &n.ends {
            let Endpoint::Port { inst: ei, port } = e else {
                continue;
            };
            if ei != inst {
                continue;
            }
            match widths.get(port) {
                None => plan.vanished.push((n.id.clone(), e.clone())),
                Some(&w) if w != n.width => {
                    plan.conflicts.push(format!(
                        "{} would be {w} bit(s) but net '{}' is {} bit(s)",
                        ep_label(e),
                        n.id,
                        n.width
                    ));
                    plan.mismatched.push((n.id.clone(), e.clone()));
                }
                Some(_) => {}
            }
        }
    }
    Ok(plan)
}

/// Structural plan of a wire between two anchors.
#[derive(Clone, Debug, PartialEq)]
pub struct ConnectPlan {
    /// The ops.
    pub ops: Vec<EditOp>,
    /// The resulting net.
    pub keep: String,
    /// Nets merged into `keep` (removed).
    pub absorbed: Vec<String>,
    /// Ports newly connected to `keep`.
    pub connected: Vec<Endpoint>,
    /// Undo label.
    pub label: String,
}

/// Plan the connectivity change of a wire whose ends touch `anchors`; checks widths.
pub fn plan_connect(def: &ComponentDef, anchors: &[Anchor]) -> Result<ConnectPlan, String> {
    let mut nets: Vec<&Net> = Vec::new();
    let mut free: Vec<&PortInfo> = Vec::new();
    for a in anchors {
        match a {
            Anchor::Port { port, net: None } => {
                if !free.iter().any(|p| p.ep == port.ep) {
                    free.push(port);
                }
            }
            Anchor::Port { net: Some(n), .. } | Anchor::Wire { net: n } => {
                let net = def
                    .nets
                    .iter()
                    .find(|x| &x.id == n)
                    .ok_or_else(|| format!("no net '{n}'"))?;
                if !nets.iter().any(|x| x.id == net.id) {
                    nets.push(net);
                }
            }
            Anchor::Free => {}
        }
    }
    if nets.is_empty() && free.is_empty() {
        return Err("a wire has to start or end on a pin or an existing wire".into());
    }
    let mut widths: Vec<(String, u32)> = nets
        .iter()
        .map(|n| (format!("net '{}'", n.id), n.width))
        .collect();
    widths.extend(free.iter().map(|p| (ep_label(&p.ep), p.width)));
    if let Some((d1, w1)) = widths.first()
        && let Some((d2, w2)) = widths.iter().find(|(_, w)| w != w1)
    {
        return Err(format!(
            "width mismatch: {d1} is {w1} bit(s) but {d2} is {w2} bit(s)"
        ));
    }
    let width = widths[0].1;
    let def_id = def.id.clone();
    let connected: Vec<Endpoint> = free.iter().map(|p| p.ep.clone()).collect();
    let connect = |net: &str, eps: &[Endpoint]| -> Vec<EditOp> {
        eps.iter()
            .map(|e| EditOp::Connect {
                def: def_id.clone(),
                net: net.to_string(),
                endpoint: e.clone(),
            })
            .collect()
    };
    match nets.as_slice() {
        [] => {
            let mut ids = Ids::default();
            let id = ids.net(def);
            Ok(ConnectPlan {
                ops: vec![EditOp::AddNet {
                    def: def_id.clone(),
                    net: Net {
                        id: id.clone(),
                        width,
                        termination: Termination::None,
                        ends: connected.clone(),
                    },
                }],
                keep: id,
                absorbed: Vec::new(),
                connected,
                label: "wire".into(),
            })
        }
        [n] => Ok(ConnectPlan {
            ops: connect(&n.id, &connected),
            keep: n.id.clone(),
            absorbed: Vec::new(),
            label: if connected.is_empty() {
                "wire".into()
            } else {
                format!("connect to {}", n.id)
            },
            connected,
        }),
        [a, b] => {
            let (keep, other) = if b.ends.len() > a.ends.len() {
                (*b, *a)
            } else {
                (*a, *b)
            };
            let mut ops = vec![EditOp::RemoveNet {
                def: def_id.clone(),
                id: other.id.clone(),
            }];
            ops.extend(connect(&keep.id, &other.ends));
            ops.extend(connect(&keep.id, &connected));
            match (keep.termination, other.termination) {
                (k, o) if k == o || o == Termination::None => {}
                (Termination::None, o) => ops.push(EditOp::SetTermination {
                    def: def_id.clone(),
                    net: keep.id.clone(),
                    termination: o,
                }),
                (k, o) => {
                    return Err(format!(
                        "cannot merge net '{}' ({k:?}) with net '{}' ({o:?})",
                        keep.id, other.id
                    ));
                }
            }
            Ok(ConnectPlan {
                ops,
                keep: keep.id.clone(),
                absorbed: vec![other.id.clone()],
                connected,
                label: format!("merge {} into {}", other.id, keep.id),
            })
        }
        _ => unreachable!("a wire has two ends"),
    }
}
