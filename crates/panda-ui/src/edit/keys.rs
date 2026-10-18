//! Keyboard-first editing: the pieces behind the canvas shortcuts that let a circuit be built
//! without the mouse, and the contextual key hints.
//!
//! * [`next_item`]: Tab / Shift+Tab cycle the selection through the chips and pins of the view
//!   (left to right, then top to bottom).
//! * [`plan_quick_wire`]: `C` wires a driver of one item to the next free input of another
//!   (the two selected items, or the previously selected item and the selected one).
//! * [`quick_candidates`] / [`fuzzy_score`]: `A` opens a search box over everything that can be
//!   placed; Enter places the choice next to the selection ([`place_spot`]).
//! * [`hints`] / [`SHEET`]: the hint bar along the bottom of the canvas (keys for the current
//!   selection and tool) and the full shortcut sheet (`?`).

use panda_netlist::{Atom, Dir};

use super::actions::{PortInfo, Selection, Tool, instantiates, item_ports, net_of};
use super::{PlaceRequest, basic_palette};
use crate::app::AppState;
use crate::doc::{Document, ItemRef, Placed};
use crate::geom::{GridPt, GridRect, gp};

/// Every shortcut, grouped (the `?` sheet).
pub const SHEET: &[(&str, &[(&str, &str)])] = &[
    (
        "Select & move",
        &[
            ("Tab / Shift+Tab", "select next / previous chip or pin"),
            ("Esc", "clear selection, cancel tool"),
            ("Ctrl+A", "select all"),
            ("Arrows", "move selection 1 step (Shift: 5)"),
            ("Arrows, nothing selected", "pan the view"),
        ],
    ),
    (
        "Build",
        &[
            ("A", "add: search chips, pins, packages"),
            (
                "C",
                "connect: wire the previous / other selected item to this one",
            ),
            ("W", "wire tool (drag from a port)"),
            ("S", "select tool"),
            ("N", "select the wires on this chip (repeat: next wire)"),
            ("F2", "rename the selected wire or chip"),
        ],
    ),
    (
        "Edit",
        &[
            ("R / Shift+R", "rotate clockwise / counter-clockwise"),
            ("F", "flip"),
            ("Del / Backspace", "delete"),
            ("Ctrl+C / X / V / D", "copy / cut / paste / duplicate"),
            ("Ctrl+Z / Ctrl+Shift+Z", "undo / redo"),
        ],
    ),
    (
        "Packages",
        &[
            ("Ctrl+G", "package the selection"),
            ("Enter", "open the selected package"),
            ("Alt+Up", "up one level"),
        ],
    ),
    (
        "Simulate",
        &[
            ("Space", "toggle the selected input pin"),
            ("F5", "run / pause"),
            ("F10", "step"),
        ],
    ),
    (
        "View & file",
        &[
            ("Home / 0", "fit the view"),
            ("+ / -", "zoom in / out"),
            ("Ctrl+N / O / S", "new / open / save"),
            ("?", "show / hide this sheet"),
        ],
    ),
];

/// What the keyboard can act on right now (input to [`hints`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HintCtx {
    /// The quick-add box is open.
    QuickAdd,
    /// Placing (place tool armed).
    Placing,
    /// Wire tool.
    Wiring,
    /// Nothing selected.
    Nothing,
    /// One wire (net) selected.
    Net,
    /// One item selected.
    Item {
        /// It is a top-level input pin (Space toggles it).
        input: bool,
        /// It is a package instance (Enter opens it).
        package: bool,
        /// It can be renamed (instances).
        renamable: bool,
        /// `C` would wire the previous item to it.
        connect: bool,
    },
    /// Two items selected.
    Two,
    /// More than two items (or items and wires).
    Many,
}

/// Classify the current state for [`hints`].
pub fn hint_ctx(
    app: &AppState,
    sel: &Selection,
    tool: &Tool,
    quick_open: bool,
    prev: Option<&ItemRef>,
) -> HintCtx {
    if quick_open {
        return HintCtx::QuickAdd;
    }
    match tool {
        Tool::Place(_) => return HintCtx::Placing,
        Tool::Wire => return HintCtx::Wiring,
        _ => {}
    }
    if sel.is_empty() {
        return HintCtx::Nothing;
    }
    if sel.single_net().is_some() {
        return HintCtx::Net;
    }
    if sel.wires.is_empty() && sel.items.len() == 1 {
        let it = sel.items.iter().next().expect("one item");
        let def = app.doc.def(&app.view.def);
        let (input, package, renamable) = match it {
            ItemRef::Pin(p) => (
                app.view.path.is_empty()
                    && def.is_some_and(|d| d.pins.iter().any(|q| &q.name == p && q.dir == Dir::In)),
                false,
                false,
            ),
            ItemRef::Instance(id) => (
                false,
                def.and_then(|d| d.instances.iter().find(|i| &i.id == id))
                    .is_some_and(|i| matches!(i.kind, panda_netlist::InstKind::Def(_))),
                true,
            ),
        };
        let connect = prev.is_some_and(|p| p != it);
        return HintCtx::Item {
            input,
            package,
            renamable,
            connect,
        };
    }
    if sel.wires.is_empty() && sel.items.len() == 2 {
        return HintCtx::Two;
    }
    HintCtx::Many
}

/// Keys worth showing for `ctx`: (keys, action).
pub fn hints(ctx: &HintCtx) -> Vec<(&'static str, &'static str)> {
    let mut v: Vec<(&str, &str)> = Vec::new();
    match ctx {
        HintCtx::QuickAdd => {
            v.extend([("Up/Down", "choose"), ("Enter", "place"), ("Esc", "close")]);
            return v;
        }
        HintCtx::Placing => {
            v.extend([
                ("click", "place"),
                ("Shift+click", "place more"),
                ("Esc", "cancel"),
            ]);
        }
        HintCtx::Wiring => {
            v.extend([
                ("drag", "from a port"),
                ("S", "select tool"),
                ("Esc", "cancel"),
            ]);
        }
        HintCtx::Nothing => {
            v.extend([
                ("A", "add"),
                ("Tab", "select"),
                ("Arrows", "pan"),
                ("Home", "fit"),
                ("Ctrl+A", "select all"),
            ]);
        }
        HintCtx::Net => {
            v.extend([
                ("F2", "rename"),
                ("N", "next wire"),
                ("Tab", "chips"),
                ("Del", "delete"),
                ("Esc", "deselect"),
            ]);
        }
        HintCtx::Item {
            input,
            package,
            renamable,
            connect,
        } => {
            if *connect {
                v.push(("C", "connect from previous"));
            }
            if *input {
                v.push(("Space", "toggle"));
            }
            if *package {
                v.push(("Enter", "open"));
            }
            if *renamable {
                v.push(("F2", "rename"));
            }
            v.extend([
                ("A", "add next"),
                ("N", "select wire"),
                ("Tab", "next"),
                ("Arrows", "move"),
                ("R", "rotate"),
                ("Del", "delete"),
            ]);
        }
        HintCtx::Two => {
            v.extend([
                ("C", "connect"),
                ("Ctrl+G", "package"),
                ("Arrows", "move"),
                ("Del", "delete"),
            ]);
        }
        HintCtx::Many => {
            v.extend([
                ("Ctrl+G", "package"),
                ("Arrows", "move"),
                ("Ctrl+D", "duplicate"),
                ("Del", "delete"),
            ]);
        }
    }
    v.push(("?", "all shortcuts"));
    v
}

/// The item after (or before, `back`) `current` in reading order (left to right, then top to
/// bottom by top-left corner), wrapping around; the first (last) item without `current`.
pub fn next_item(
    doc: &Document,
    def: &str,
    current: Option<&ItemRef>,
    back: bool,
) -> Option<ItemRef> {
    let mut items: Vec<(GridPt, ItemRef)> = doc
        .placed(def)
        .into_iter()
        .map(|p| (p.rect().min, p.item))
        .collect();
    if items.is_empty() {
        return None;
    }
    items.sort_by_key(|(p, it)| (p.x, p.y, it.clone()));
    let n = items.len();
    let k = match current.and_then(|c| items.iter().position(|(_, it)| it == c)) {
        Some(i) if back => (i + n - 1) % n,
        Some(i) => (i + 1) % n,
        None if back => n - 1,
        None => 0,
    };
    Some(items[k].1.clone())
}

/// The net after `current` among the nets connected to `item` (drivers first, then inputs,
/// in port order), wrapping; the first one without `current`.
pub fn next_net_of(
    doc: &Document,
    def_id: &str,
    item: &ItemRef,
    current: Option<&str>,
) -> Option<String> {
    let def = doc.def(def_id)?;
    let mut ports: Vec<PortInfo> = doc
        .placed(def_id)
        .iter()
        .filter(|p| &p.item == item)
        .flat_map(item_ports)
        .collect();
    ports.sort_by_key(|p| !drives(p));
    let mut nets: Vec<String> = Vec::new();
    for p in &ports {
        if let Some(n) = net_of(def, &p.ep)
            && !nets.contains(&n.id)
        {
            nets.push(n.id.clone());
        }
    }
    let k = current
        .and_then(|c| nets.iter().position(|n| n == c))
        .map_or(0, |i| (i + 1) % nets.len().max(1));
    nets.get(k).cloned()
}

/// True if the port drives its net (atom / package outputs, input-pin markers).
fn drives(p: &PortInfo) -> bool {
    match p.item {
        ItemRef::Pin(_) => p.dir == Dir::In,
        ItemRef::Instance(_) => p.dir != Dir::In,
    }
}

/// The wire `C` would draw from item `from` to item `to`: the first driving port of `from`
/// and the first unconnected input of `to` of the same width (tried in both directions, `from`
/// → `to` first). Errors say why nothing can be connected.
pub fn plan_quick_wire(
    doc: &Document,
    def_id: &str,
    from: &ItemRef,
    to: &ItemRef,
) -> Result<(PortInfo, PortInfo), String> {
    let def = doc.def(def_id).ok_or("no such def")?;
    let placed = doc.placed(def_id);
    let ports_of = |it: &ItemRef| -> Vec<PortInfo> {
        placed
            .iter()
            .filter(|p| &p.item == it)
            .flat_map(item_ports)
            .collect()
    };
    let (pf, pt) = (ports_of(from), ports_of(to));
    let attempt = |src: &[PortInfo], dst: &[PortInfo]| -> Option<(PortInfo, PortInfo)> {
        for d in dst
            .iter()
            .filter(|d| !drives(d) && net_of(def, &d.ep).is_none())
        {
            if let Some(s) = src.iter().find(|s| drives(s) && s.width == d.width) {
                return Some((s.clone(), d.clone()));
            }
        }
        None
    };
    attempt(&pf, &pt)
        .or_else(|| attempt(&pt, &pf))
        .ok_or_else(|| "nothing to connect: no free input with a matching driver".to_string())
}

/// What `A` offers in the current view: the basic palette (sources only at the top level, so
/// they can be driven by the simulator) and every package that does not contain this view.
pub fn quick_candidates(app: &AppState) -> Vec<(String, PlaceRequest)> {
    let def = &app.view.def;
    let top = app.view.path.is_empty();
    let mut v: Vec<(String, PlaceRequest)> = basic_palette()
        .into_iter()
        .filter(|r| {
            top || !matches!(
                r,
                PlaceRequest::Atom(Atom::Clock { .. } | Atom::PowerOnReset { .. })
            )
        })
        .map(|r| (quick_label(&r), r))
        .collect();
    for d in &app.doc.design.defs {
        if &d.id != def && d.id != app.doc.design.top && !instantiates(&app.doc.design, &d.id, def)
        {
            v.push((d.id.clone(), PlaceRequest::Def(d.id.clone())));
        }
    }
    v
}

/// Display name in the quick-add list.
fn quick_label(r: &PlaceRequest) -> String {
    match r {
        PlaceRequest::Pin { dir: Dir::In, .. } => "input pin".into(),
        PlaceRequest::Pin { .. } => "output pin".into(),
        PlaceRequest::Atom(Atom::Gate { op, .. }) => format!("{op:?}").to_uppercase(),
        PlaceRequest::Atom(Atom::Tri { .. }) => "tri-state buffer".into(),
        PlaceRequest::Atom(Atom::Splitter { .. }) => "split / merge".into(),
        PlaceRequest::Atom(Atom::Constant { .. }) => "constant".into(),
        PlaceRequest::Atom(Atom::Clock { .. }) => "clock".into(),
        PlaceRequest::Atom(Atom::PowerOnReset { .. }) => "power-on reset".into(),
        PlaceRequest::Atom(a) => super::actions::atom_label(a),
        PlaceRequest::Def(d) => d.clone(),
    }
}

/// Fuzzy match score of `query` against `label` (case-insensitive subsequence; higher is
/// better, `None` if it does not match). Prefix and word-start matches rank first.
pub fn fuzzy_score(label: &str, query: &str) -> Option<i32> {
    let q: Vec<char> = query
        .to_lowercase()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    if q.is_empty() {
        return Some(0);
    }
    let l: Vec<char> = label.to_lowercase().chars().collect();
    let mut score = 0;
    let mut qi = 0;
    let mut prev_match = false;
    for (i, &c) in l.iter().enumerate() {
        if qi < q.len() && c == q[qi] {
            let word_start = i == 0 || !l[i - 1].is_alphanumeric();
            score += 1 + if word_start { 6 } else { 0 } + if prev_match { 3 } else { 0 };
            if i == 0 {
                score += 10;
            }
            qi += 1;
            prev_match = true;
        } else {
            prev_match = false;
        }
    }
    (qi == q.len()).then_some(score - l.len() as i32 / 4)
}

/// The quick-add candidates matching `query`, best first.
pub fn quick_matches(cands: &[(String, PlaceRequest)], query: &str) -> Vec<usize> {
    let mut v: Vec<(i32, usize)> = cands
        .iter()
        .enumerate()
        .filter_map(|(i, (l, _))| fuzzy_score(l, query).map(|s| (s, i)))
        .collect();
    v.sort_by_key(|&(s, i)| (-s, i));
    v.into_iter().map(|(_, i)| i).collect()
}

/// Where a quick-added symbol of `size` goes (its center): to the right of the selected items,
/// vertically centred on them; else at `fallback` (the pointer or the view centre). Moved
/// further right while it would overlap another item.
pub fn place_spot(placed: &[Placed], sel: &Selection, size: GridPt, fallback: GridPt) -> GridPt {
    const GAP: i32 = 6;
    let mut bounds: Option<GridRect> = None;
    for p in placed.iter().filter(|p| sel.items.contains(&p.item)) {
        let r = p.rect();
        bounds = Some(match bounds {
            None => r,
            Some(b) => GridRect {
                min: gp(b.min.x.min(r.min.x), b.min.y.min(r.min.y)),
                max: gp(b.max.x.max(r.max.x), b.max.y.max(r.max.y)),
            },
        });
    }
    let mut at = match bounds {
        Some(b) => gp(b.max.x + GAP + size.x / 2, (b.min.y + b.max.y) / 2),
        None => fallback,
    };
    for _ in 0..50 {
        let r = GridRect {
            min: at - gp(size.x / 2 + 1, size.y / 2 + 1),
            max: at - gp(size.x / 2, size.y / 2) + size + gp(1, 1),
        };
        let hit = placed.iter().find(|p| {
            let q = p.rect();
            q.min.x < r.max.x && r.min.x < q.max.x && q.min.y < r.max.y && r.min.y < q.max.y
        });
        match hit {
            Some(p) => at.x = p.rect().max.x + GAP / 2 + size.x / 2 + 1,
            None => break,
        }
    }
    at
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuzzy_ranks_prefixes_first() {
        assert!(fuzzy_score("NAND", "nd").is_some());
        assert!(fuzzy_score("NAND", "x").is_none());
        let cands: Vec<(String, PlaceRequest)> = basic_palette()
            .into_iter()
            .map(|r| (quick_label(&r), r))
            .collect();
        let m = quick_matches(&cands, "in");
        assert_eq!(cands[m[0]].0, "input pin");
        let m = quick_matches(&cands, "nand");
        assert_eq!(cands[m[0]].0, "NAND");
        let m = quick_matches(&cands, "out");
        assert_eq!(cands[m[0]].0, "output pin");
        assert_eq!(quick_matches(&cands, "").len(), cands.len());
    }

    #[test]
    fn hints_follow_the_context() {
        let keys = |c: HintCtx| hints(&c).into_iter().map(|(k, _)| k).collect::<Vec<_>>();
        assert!(keys(HintCtx::Nothing).contains(&"A"));
        assert!(keys(HintCtx::Net).contains(&"F2"));
        let item = HintCtx::Item {
            input: true,
            package: false,
            renamable: false,
            connect: true,
        };
        let k = keys(item);
        assert!(k.contains(&"Space") && k.contains(&"C") && !k.contains(&"F2"));
        assert_eq!(keys(HintCtx::QuickAdd), ["Up/Down", "Enter", "Esc"]);
        assert!(keys(HintCtx::Two).contains(&"C"));
    }
}
