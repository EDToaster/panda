//! Create a package from a selection (pure model code, no egui).
//!
//! [`plan_package`] turns a set of instances of one def into a new def (the *package*) and
//! replaces them by one instance of it:
//!
//! * nets whose endpoints are all inside the selection move into the package;
//! * nets with endpoints inside and outside (*crossing* nets; the def's own pins count as
//!   outside) become package pins: inside, the net keeps its id and gains the pin endpoint;
//!   outside, it stays in the parent and connects to the new instance's port;
//! * a pin's direction follows the drivers of the net's bits (through splitter aliases): driven
//!   only from inside → `Out`, only from outside → `In`, from both → `InOut`, undriven → `In`;
//! * pin names: the net id unless it is an automatic one (`n<k>`), else the name of a parent pin
//!   on the net, else `in<k>` / `out<k>` / `io<k>`; pins are ordered `In`, `Out`, `InOut`, each
//!   by the vertical position of their inside endpoints;
//! * geometry: inside, the selected instances keep their relative placement, the moved nets
//!   keep their wires, pin markers are stacked left (`In`) and right (others) with one routing
//!   lane each; outside, the new instance sits at the centre of the selection and each crossing
//!   net keeps its wires outside the selection box plus a short Manhattan stub to the new port.
//!   The package symbol gets the automatic pin layout ([`crate::geom::auto_symbol_layout`]).
//!
//! The structural change is two [`EditOp`]s ([`PackagePlan::ops`]): `AddDef(package)` and
//! `ReplaceDef(parent)`; undo is `ReplaceDef(old parent)` + `RemoveDef(package)`
//! ([`PackagePlan::undo_ops`]). The simulated behaviour is unchanged (pins are pure aliases).

use std::collections::{BTreeSet, HashMap, HashSet};

use panda_kernel::EditOp;
use panda_netlist::{Atom, ComponentDef, Design, Dir, Endpoint, InstKind, Instance, Net, PinDecl};

use crate::doc::{Document, EditorData, InstGeom, ItemRef, Placed};
use crate::geom::{GridPt, GridRect, auto_symbol_layout, gp, package_symbol, pin_symbol};
use crate::layout::net_geom;

/// One pin of the new package.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackagePin {
    /// Pin declaration.
    pub decl: PinDecl,
    /// The crossing net (same id inside the package and in the parent).
    pub net: String,
}

/// Everything needed to apply (and undo) "create package from selection".
#[derive(Clone, Debug, PartialEq)]
pub struct PackagePlan {
    /// The new def.
    pub package: ComponentDef,
    /// The parent def before the change.
    pub parent_before: ComponentDef,
    /// The parent def after the change (selection replaced by one instance).
    pub parent_after: ComponentDef,
    /// Id of the new instance in the parent.
    pub instance: String,
    /// Pins of the package (declaration order) with their nets.
    pub pins: Vec<PackagePin>,
    /// Ids of the nets that moved into the package entirely.
    pub moved_nets: Vec<String>,
    /// Geometry of the package def.
    pub package_editor: EditorData,
    /// Geometry of the parent before (if it had any).
    pub parent_editor_before: Option<EditorData>,
    /// Geometry of the parent after.
    pub parent_editor_after: Option<EditorData>,
}

impl PackagePlan {
    /// The structural edit: add the package, replace the parent.
    pub fn ops(&self) -> Vec<EditOp> {
        vec![
            EditOp::AddDef(self.package.clone()),
            EditOp::ReplaceDef(self.parent_after.clone()),
        ]
    }

    /// The inverse edit: restore the parent, remove the package.
    pub fn undo_ops(&self) -> Vec<EditOp> {
        vec![
            EditOp::ReplaceDef(self.parent_before.clone()),
            EditOp::RemoveDef(self.package.id.clone()),
        ]
    }

    /// Install the geometry after [`PackagePlan::ops`] were applied to `doc`.
    pub fn install_geometry(&self, doc: &mut Document) {
        doc.editors
            .insert(self.package.id.clone(), self.package_editor.clone());
        set_editor(doc, &self.parent_after.id, self.parent_editor_after.clone());
        doc.touch();
    }

    /// Restore the geometry after [`PackagePlan::undo_ops`] were applied to `doc`.
    pub fn restore_geometry(&self, doc: &mut Document) {
        doc.editors.remove(&self.package.id);
        set_editor(
            doc,
            &self.parent_before.id,
            self.parent_editor_before.clone(),
        );
        doc.touch();
    }
}

fn set_editor(doc: &mut Document, def: &str, e: Option<EditorData>) {
    match e {
        Some(e) => {
            doc.editors.insert(def.to_string(), e);
        }
        None => {
            doc.editors.remove(def);
        }
    }
}

/// True for ids usable as def, pin, net or instance names (non-empty, no `/`, `[`, `]`,
/// no whitespace).
pub fn valid_ident(s: &str) -> bool {
    !s.is_empty() && !s.contains(['/', '[', ']']) && !s.chars().any(char::is_whitespace)
}

/// A def id not used in `design`, derived from `base` (`base`, `base_1`, …).
pub fn fresh_def_id(design: &Design, base: &str) -> String {
    let base = if valid_ident(base) { base } else { "pkg" };
    unique(base, |s| design.def(s).is_some())
}

/// `base`, or `base_1`, `base_2`, … — the first one for which `taken` is false.
pub fn unique(base: &str, taken: impl Fn(&str) -> bool) -> String {
    if !taken(base) {
        return base.to_string();
    }
    (1..)
        .map(|k| format!("{base}_{k}"))
        .find(|s| !taken(s))
        .expect("infinite")
}

/// An instance id not used in `def`: `prefix<k>` with the smallest free `k`.
pub fn fresh_instance_id(def: &ComponentDef, prefix: &str) -> String {
    let used: HashSet<&str> = def.instances.iter().map(|i| i.id.as_str()).collect();
    (0..)
        .map(|k| format!("{prefix}{k}"))
        .find(|s| !used.contains(s.as_str()))
        .expect("infinite")
}

fn is_auto_net_id(id: &str) -> bool {
    id.strip_prefix('n')
        .is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
}

/// Direction of the port `port` of instance `inst` (`None` if unknown).
fn port_dir(design: &Design, inst: &Instance, port: &str) -> Option<Dir> {
    match &inst.kind {
        InstKind::Atom(a) => a
            .ports()
            .into_iter()
            .find(|p| p.name == port)
            .map(|p| p.dir),
        InstKind::Def(d) => design
            .def(d)?
            .pins
            .iter()
            .find(|p| p.name == port)
            .map(|p| p.dir),
    }
}

/// Union-find over `(net, bit)` of one def, aliasing splitter bits.
struct BitClasses {
    base: Vec<usize>,
    parent: Vec<usize>,
}

impl BitClasses {
    fn new(def: &ComponentDef) -> BitClasses {
        let mut base = Vec::with_capacity(def.nets.len());
        let mut n = 0;
        for net in &def.nets {
            base.push(n);
            n += net.width as usize;
        }
        let mut s = BitClasses {
            base,
            parent: (0..n).collect(),
        };
        // Endpoint → net index.
        let mut net_of: HashMap<(&str, &str), usize> = HashMap::new();
        for (ni, net) in def.nets.iter().enumerate() {
            for e in &net.ends {
                if let Endpoint::Port { inst, port } = e {
                    net_of.insert((inst, port), ni);
                }
            }
        }
        for inst in &def.instances {
            let InstKind::Atom(Atom::Splitter { parts, .. }) = &inst.kind else {
                continue;
            };
            let Some(&whole) = net_of.get(&(inst.id.as_str(), "whole")) else {
                continue;
            };
            for (i, &(lo, hi)) in parts.iter().enumerate() {
                let pname = format!("p{i}");
                let Some(&part) = net_of.get(&(inst.id.as_str(), pname.as_str())) else {
                    continue;
                };
                for b in 0..=(hi - lo) {
                    let a = s.base[whole] + (lo + b) as usize;
                    let c = s.base[part] + b as usize;
                    s.union(a, c);
                }
            }
        }
        s
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            self.parent[a] = b;
        }
    }

    fn bit(&mut self, net: usize, bit: u32) -> usize {
        let i = self.base[net] + bit as usize;
        self.find(i)
    }
}

/// Plan packaging the instances `selection` of def `def_id` into a new def `new_def`.
///
/// Errors: unknown def or instance, empty selection, a `Clock` / `PowerOnReset` in the selection
/// (they may only live in the top def), an invalid or taken def id.
pub fn plan_package(
    doc: &Document,
    def_id: &str,
    selection: &[String],
    new_def: &str,
) -> Result<PackagePlan, String> {
    let design = &doc.design;
    let def = design
        .def(def_id)
        .ok_or_else(|| format!("unknown def '{def_id}'"))?;
    if !valid_ident(new_def) {
        return Err(format!("'{new_def}' is not a valid def id"));
    }
    if design.def(new_def).is_some() {
        return Err(format!("a def '{new_def}' already exists"));
    }
    let sel: BTreeSet<&str> = selection.iter().map(String::as_str).collect();
    if sel.is_empty() {
        return Err("nothing selected: select the instances to package".into());
    }
    for id in &sel {
        let inst = def
            .instances
            .iter()
            .find(|i| i.id == *id)
            .ok_or_else(|| format!("def '{def_id}' has no instance '{id}'"))?;
        if matches!(
            inst.kind,
            InstKind::Atom(Atom::Clock { .. } | Atom::PowerOnReset { .. })
        ) {
            return Err(format!(
                "'{id}' is a clock / reset source; those must stay in the top def"
            ));
        }
    }
    let inside =
        |e: &Endpoint| matches!(e, Endpoint::Port { inst, .. } if sel.contains(inst.as_str()));

    // ---- drivers per bit class ----
    let mut classes = BitClasses::new(def);
    let inst_by_id: HashMap<&str, &Instance> =
        def.instances.iter().map(|i| (i.id.as_str(), i)).collect();
    // class → (driven from inside, driven from outside)
    let mut drv: HashMap<usize, (bool, bool)> = HashMap::new();
    for (ni, net) in def.nets.iter().enumerate() {
        for e in &net.ends {
            let drives = match e {
                Endpoint::Pin(p) => def
                    .pins
                    .iter()
                    .any(|d| &d.name == p && matches!(d.dir, Dir::In | Dir::InOut)),
                Endpoint::Port { inst, port } => {
                    let Some(i) = inst_by_id.get(inst.as_str()) else {
                        continue;
                    };
                    let splitter = matches!(i.kind, InstKind::Atom(Atom::Splitter { .. }));
                    !splitter && matches!(port_dir(design, i, port), Some(Dir::Out | Dir::InOut))
                }
            };
            if !drives {
                continue;
            }
            let ins = inside(e);
            for b in 0..net.width {
                let c = classes.bit(ni, b);
                let d = drv.entry(c).or_default();
                if ins {
                    d.0 = true;
                } else {
                    d.1 = true;
                }
            }
        }
    }

    // ---- classify nets ----
    let placed: Vec<Placed> = doc.placed(def_id);
    let placed_by: HashMap<&ItemRef, &Placed> = placed.iter().map(|p| (&p.item, p)).collect();
    let end_pos = |e: &Endpoint| -> Option<GridPt> {
        match e {
            Endpoint::Pin(n) => placed_by.get(&ItemRef::Pin(n.clone()))?.port_pos(n),
            Endpoint::Port { inst, port } => placed_by
                .get(&ItemRef::Instance(inst.clone()))?
                .port_pos(port),
        }
    };
    let mut moved: Vec<&Net> = Vec::new();
    let mut crossing: Vec<(usize, &Net, Dir, f64)> = Vec::new();
    for (ni, net) in def.nets.iter().enumerate() {
        let n_in = net.ends.iter().filter(|e| inside(e)).count();
        if n_in == 0 {
            continue;
        }
        if n_in == net.ends.len() {
            moved.push(net);
            continue;
        }
        let (mut di, mut dout) = (false, false);
        for b in 0..net.width {
            let c = classes.bit(ni, b);
            if let Some(&(a, o)) = drv.get(&c) {
                di |= a;
                dout |= o;
            }
        }
        let dir = match (di, dout) {
            (true, false) => Dir::Out,
            (true, true) => Dir::InOut,
            _ => Dir::In,
        };
        let ys: Vec<i32> = net
            .ends
            .iter()
            .filter(|e| inside(e))
            .filter_map(|e| end_pos(e).map(|p| p.y))
            .collect();
        let y = if ys.is_empty() {
            ni as f64
        } else {
            ys.iter().map(|&y| f64::from(y)).sum::<f64>() / ys.len() as f64
        };
        crossing.push((ni, net, dir, y));
    }
    let rank = |d: Dir| match d {
        Dir::In => 0,
        Dir::Out => 1,
        Dir::InOut => 2,
    };
    crossing.sort_by(|a, b| {
        rank(a.2)
            .cmp(&rank(b.2))
            .then(a.3.total_cmp(&b.3))
            .then(a.0.cmp(&b.0))
    });

    // ---- pin names ----
    let mut taken: HashSet<String> = HashSet::new();
    let mut counters: HashMap<&str, usize> = HashMap::new();
    let mut pins: Vec<PackagePin> = Vec::new();
    for &(_, net, dir, _) in &crossing {
        let parent_pin = net.ends.iter().find_map(|e| match e {
            Endpoint::Pin(p) => Some(p.clone()),
            _ => None,
        });
        let base = if !is_auto_net_id(&net.id) {
            net.id.clone()
        } else if let Some(p) = parent_pin.filter(|p| !taken.contains(p)) {
            p
        } else {
            let prefix = match dir {
                Dir::In => "in",
                Dir::Out => "out",
                Dir::InOut => "io",
            };
            let k = counters.entry(prefix).or_default();
            loop {
                let s = format!("{prefix}{k}");
                *k += 1;
                if !taken.contains(&s) {
                    break s;
                }
            }
        };
        let name = unique(&base, |s| taken.contains(s));
        taken.insert(name.clone());
        pins.push(PackagePin {
            decl: PinDecl {
                name,
                dir,
                width: net.width,
            },
            net: net.id.clone(),
        });
    }

    // ---- the package def ----
    let pin_of_net: HashMap<&str, &str> = pins
        .iter()
        .map(|p| (p.net.as_str(), p.decl.name.as_str()))
        .collect();
    let mut pkg_nets: Vec<Net> = Vec::new();
    for net in &def.nets {
        let crosses = pin_of_net.get(net.id.as_str());
        let is_moved = moved.iter().any(|m| m.id == net.id);
        if !is_moved && crosses.is_none() {
            continue;
        }
        let mut ends: Vec<Endpoint> = net.ends.iter().filter(|e| inside(e)).cloned().collect();
        if let Some(p) = crosses {
            ends.insert(0, Endpoint::Pin((*p).to_string()));
        }
        pkg_nets.push(Net {
            id: net.id.clone(),
            width: net.width,
            termination: net.termination,
            ends,
        });
    }
    let package = ComponentDef {
        id: new_def.to_string(),
        name: new_def.to_string(),
        version: 0,
        pins: pins.iter().map(|p| p.decl.clone()).collect(),
        instances: def
            .instances
            .iter()
            .filter(|i| sel.contains(i.id.as_str()))
            .cloned()
            .collect(),
        nets: pkg_nets,
        editor: serde_json::Value::Null,
    };

    // ---- the new parent ----
    let base_inst = if valid_ident(new_def) && !def.instances.iter().any(|i| i.id == new_def) {
        new_def.to_string()
    } else {
        fresh_instance_id(def, "u")
    };
    let instance = base_inst;
    let mut parent_after = def.clone();
    parent_after
        .instances
        .retain(|i| !sel.contains(i.id.as_str()));
    parent_after.instances.push(Instance {
        id: instance.clone(),
        kind: InstKind::Def(new_def.to_string()),
    });
    parent_after
        .nets
        .retain(|n| !moved.iter().any(|m| m.id == n.id));
    for net in &mut parent_after.nets {
        let crosses = pin_of_net.get(net.id.as_str()).copied();
        net.ends.retain(|e| !inside(e));
        if let Some(p) = crosses {
            net.ends.push(Endpoint::Port {
                inst: instance.clone(),
                port: p.to_string(),
            });
        }
    }

    // ---- geometry ----
    let parent_editor_before = doc.editor(def_id).cloned();
    let sel_placed: Vec<&Placed> = placed
        .iter()
        .filter(|p| matches!(&p.item, ItemRef::Instance(i) if sel.contains(i.as_str())))
        .collect();
    let (package_editor, parent_editor_after) = match (&parent_editor_before, sel_placed.first()) {
        (Some(ped), Some(first)) => {
            let bbox = sel_placed
                .iter()
                .fold(first.rect(), |r, p| r.union(&p.rect()));
            let pkg_ed = package_geometry(&package, &pins, &moved, ped, &sel_placed, bbox);
            let mut par = parent_geometry(
                &parent_after,
                &package,
                &pkg_ed,
                &instance,
                &pins,
                ped,
                bbox,
                &placed,
                &sel,
            );
            par.auto = false;
            (pkg_ed, Some(par))
        }
        _ => {
            // No geometry to preserve: auto-lay out the package.
            let mut d2 = design.clone();
            d2.defs.push(package.clone());
            let mut e = crate::layout::layout_def(&d2, &package, |_| None);
            e.symbol = Some(auto_symbol_layout(&package));
            (e, None)
        }
    };

    Ok(PackagePlan {
        package,
        parent_before: def.clone(),
        parent_after,
        instance,
        pins,
        moved_nets: moved.iter().map(|n| n.id.clone()).collect(),
        package_editor,
        parent_editor_before,
        parent_editor_after,
    })
}

/// Geometry inside the new package (see the module docs).
fn package_geometry(
    package: &ComponentDef,
    pins: &[PackagePin],
    moved: &[&Net],
    ped: &EditorData,
    sel_placed: &[&Placed],
    bbox: GridRect,
) -> EditorData {
    let ins: Vec<&PackagePin> = pins.iter().filter(|p| p.decl.dir == Dir::In).collect();
    let outs: Vec<&PackagePin> = pins.iter().filter(|p| p.decl.dir != Dir::In).collect();
    let marker_w = |p: &PackagePin| pin_symbol(&p.decl).size.x;
    let wl = ins.iter().map(|p| marker_w(p)).max().unwrap_or(0);
    // Lanes: one per pin between the markers and the instances.
    let x0 = wl + ins.len() as i32 + 2;
    let y0 = 2;
    let delta = gp(x0, y0) - bbox.min;
    let mut e = EditorData {
        version: crate::doc::EDITOR_VERSION,
        ..EditorData::default()
    };
    let mut ports: HashMap<(&str, &str), GridPt> = HashMap::new();
    for p in sel_placed {
        let ItemRef::Instance(id) = &p.item else {
            continue;
        };
        e.instances.insert(
            id.clone(),
            InstGeom {
                pos: p.geom.pos + delta,
                orient: p.geom.orient,
            },
        );
        for port in &p.symbol.ports {
            if let Some(pos) = p.port_pos(&port.name) {
                ports.insert((id.as_str(), port.name.as_str()), pos + delta);
            }
        }
    }
    for net in moved {
        if let Some(g) = ped.wires.get(&net.id) {
            let segs = g.segments().map(|(a, b)| (a + delta, b + delta)).collect();
            e.wires.insert(net.id.clone(), net_geom(segs));
        }
    }
    let net_ends = |net: &str| -> Vec<GridPt> {
        package
            .nets
            .iter()
            .find(|n| n.id == net)
            .map(|n| {
                n.ends
                    .iter()
                    .filter_map(|e| match e {
                        Endpoint::Port { inst, port } => {
                            ports.get(&(inst.as_str(), port.as_str())).copied()
                        }
                        Endpoint::Pin(_) => None,
                    })
                    .collect()
            })
            .unwrap_or_default()
    };
    let bw = bbox.max.x - bbox.min.x;
    let place_column = |list: &[&PackagePin], e: &mut EditorData, left: bool| {
        let mut next_y = y0;
        for (k, p) in list.iter().enumerate() {
            let ends = net_ends(&p.net);
            let avg = if ends.is_empty() {
                next_y + 1
            } else {
                ends.iter().map(|q| q.y).sum::<i32>() / ends.len() as i32
            };
            let my = (avg - 1).max(next_y);
            next_y = my + 2;
            let sym = pin_symbol(&p.decl);
            let w = sym.size.x;
            let (mx, lane) = if left {
                (wl - w, wl + 1 + k as i32)
            } else {
                let lane = x0 + bw + 1 + k as i32;
                (x0 + bw + outs.len() as i32 + 2, lane)
            };
            let marker = gp(mx, my);
            e.pins.insert(
                p.decl.name.clone(),
                InstGeom {
                    pos: marker,
                    orient: Default::default(),
                },
            );
            let port = marker + sym.ports[0].pos;
            let mut segs = vec![(port, gp(lane, port.y))];
            let (mut ymin, mut ymax) = (port.y, port.y);
            for q in &ends {
                segs.push((gp(lane, q.y), *q));
                ymin = ymin.min(q.y);
                ymax = ymax.max(q.y);
            }
            segs.push((gp(lane, ymin), gp(lane, ymax)));
            e.wires.insert(p.net.clone(), net_geom(segs));
        }
    };
    place_column(&ins, &mut e, true);
    place_column(&outs, &mut e, false);
    e.symbol = Some(auto_symbol_layout(package));
    e
}

/// Geometry of the parent after packaging (see the module docs).
#[allow(clippy::too_many_arguments)]
fn parent_geometry(
    parent_after: &ComponentDef,
    package: &ComponentDef,
    pkg_ed: &EditorData,
    instance: &str,
    pins: &[PackagePin],
    ped: &EditorData,
    bbox: GridRect,
    placed: &[Placed],
    sel: &BTreeSet<&str>,
) -> EditorData {
    let mut e = ped.clone();
    e.instances.retain(|k, _| !sel.contains(k.as_str()));
    let sym = package_symbol(package, pkg_ed.symbol.as_ref());
    // The free spot nearest to the selection's centre (other items keep a 1-unit margin).
    let obstacles: Vec<GridRect> = placed
        .iter()
        .filter(|p| !matches!(&p.item, ItemRef::Instance(i) if sel.contains(i.as_str())))
        .map(Placed::rect)
        .collect();
    let centre = gp((bbox.min.x + bbox.max.x) / 2, (bbox.min.y + bbox.max.y) / 2);
    let pos = free_spot(
        centre - gp(sym.size.x / 2, sym.size.y / 2),
        sym.size,
        &obstacles,
    );
    let geom = InstGeom {
        pos,
        orient: Default::default(),
    };
    e.instances.insert(instance.to_string(), geom);
    let new_placed = Placed {
        item: ItemRef::Instance(instance.to_string()),
        symbol: sym,
        geom,
    };
    // Wires inside the selection's box are dropped (clipped at its boundary).
    let cut = bbox;
    let strictly_inside =
        |p: GridPt| p.x > cut.min.x && p.x < cut.max.x && p.y > cut.min.y && p.y < cut.max.y;
    e.wires
        .retain(|k, _| parent_after.nets.iter().any(|n| &n.id == k));
    for p in pins {
        let Some(port) = new_placed.port_pos(&p.decl.name) else {
            continue;
        };
        let side = new_placed
            .symbol
            .port(&p.decl.name)
            .map(|q| q.side)
            .unwrap_or(crate::geom::Side::Left);
        let stub = port + side.outward();
        let old = e.wires.remove(&p.net).unwrap_or_default();
        let mut anchors = outside_ends(parent_after, &p.net, placed, sel);
        let clipped: Vec<(GridPt, GridPt)> = old
            .segments()
            .filter_map(|(a, b)| clip_outside(a, b, &cut, &strictly_inside))
            .collect();
        // Attach to the nearest point of wire still reaching an outside endpoint, else to the
        // nearest outside endpoint itself.
        let mut segs = live_wire(clipped, &anchors);
        let dist = |q: &GridPt| (q.x - stub.x).abs() + (q.y - stub.y).abs();
        let target = if segs.is_empty() {
            anchors.iter().copied().min_by_key(dist)
        } else {
            segs.iter()
                .map(|&(a, b)| closest_on_seg(stub, a, b))
                .min_by_key(dist)
        };
        segs.push((port, stub));
        if let Some(t) = target {
            segs.push((stub, gp(t.x, stub.y)));
            segs.push((gp(t.x, stub.y), t));
        }
        anchors.push(port);
        // Normalize (split at the attach point), then drop the dangling remainder.
        let g = net_geom(segs);
        let segs = prune_dead_ends(g.segments().collect(), &anchors);
        e.wires.insert(p.net.clone(), net_geom(segs));
    }
    e
}

/// Top-left position for a `size` box nearest to `want` that keeps one grid unit away from
/// every obstacle (searching square rings outwards; `want` itself if nothing is free nearby).
fn free_spot(want: GridPt, size: GridPt, obstacles: &[GridRect]) -> GridPt {
    let fits = |p: GridPt| {
        let r = GridRect::from_pos_size(p - gp(1, 1), size + gp(2, 2));
        !obstacles.iter().any(|o| o.overlaps(&r))
    };
    for radius in 0..=60i32 {
        let mut best: Option<GridPt> = None;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx.abs().max(dy.abs()) != radius {
                    continue;
                }
                let p = want + gp(dx, dy);
                let d = dx.abs() + dy.abs();
                if fits(p) && best.is_none_or(|b| (b.x - want.x).abs() + (b.y - want.y).abs() > d) {
                    best = Some(p);
                }
            }
        }
        if let Some(b) = best {
            return b;
        }
    }
    want
}

fn touches(p: GridPt, (a, b): (GridPt, GridPt)) -> bool {
    p == a
        || p == b
        || (a.y == b.y && p.y == a.y && p.x > a.x.min(b.x) && p.x < a.x.max(b.x))
        || (a.x == b.x && p.x == a.x && p.y > a.y.min(b.y) && p.y < a.y.max(b.y))
}

/// The segments connected (through shared or touching ends) to at least one anchor.
fn live_wire(segs: Vec<(GridPt, GridPt)>, anchors: &[GridPt]) -> Vec<(GridPt, GridPt)> {
    let n = segs.len();
    let mut live = vec![false; n];
    let mut stack: Vec<usize> = (0..n)
        .filter(|&i| anchors.iter().any(|&p| touches(p, segs[i])))
        .collect();
    for &i in &stack {
        live[i] = true;
    }
    while let Some(i) = stack.pop() {
        let (a, b) = segs[i];
        for j in 0..n {
            if !live[j]
                && (touches(a, segs[j])
                    || touches(b, segs[j])
                    || touches(segs[j].0, segs[i])
                    || touches(segs[j].1, segs[i]))
            {
                live[j] = true;
                stack.push(j);
            }
        }
    }
    segs.into_iter()
        .zip(live)
        .filter_map(|(s, l)| l.then_some(s))
        .collect()
}

/// Remove wire segments ending in a dead end (an end touching no other segment and not at an
/// anchor), repeatedly.
fn prune_dead_ends(mut segs: Vec<(GridPt, GridPt)>, anchors: &[GridPt]) -> Vec<(GridPt, GridPt)> {
    loop {
        let mut deg: HashMap<GridPt, u32> = HashMap::new();
        for &(a, b) in &segs {
            *deg.entry(a).or_default() += 1;
            *deg.entry(b).or_default() += 1;
        }
        // An end lying on another segment's interior is connected too.
        let on_interior = |p: GridPt, segs: &[(GridPt, GridPt)]| {
            segs.iter().any(|&(a, b)| {
                (a.y == b.y && p.y == a.y && p.x > a.x.min(b.x) && p.x < a.x.max(b.x))
                    || (a.x == b.x && p.x == a.x && p.y > a.y.min(b.y) && p.y < a.y.max(b.y))
            })
        };
        let dead = |p: GridPt| deg[&p] == 1 && !anchors.contains(&p) && !on_interior(p, &segs);
        let before = segs.len();
        let keep: Vec<(GridPt, GridPt)> = segs
            .iter()
            .copied()
            .filter(|&(a, b)| !dead(a) && !dead(b))
            .collect();
        segs = keep;
        if segs.len() == before {
            return segs;
        }
    }
}

/// World positions of the endpoints of `net` outside the selection (parent pins included).
fn outside_ends(
    parent_after: &ComponentDef,
    net: &str,
    placed: &[Placed],
    sel: &BTreeSet<&str>,
) -> Vec<GridPt> {
    let Some(n) = parent_after.nets.iter().find(|n| n.id == net) else {
        return Vec::new();
    };
    n.ends
        .iter()
        .filter_map(|e| {
            let (item, port) = match e {
                Endpoint::Pin(p) => (ItemRef::Pin(p.clone()), p.as_str()),
                Endpoint::Port { inst, port } if !sel.contains(inst.as_str()) => {
                    (ItemRef::Instance(inst.clone()), port.as_str())
                }
                Endpoint::Port { .. } => return None,
            };
            placed.iter().find(|p| p.item == item)?.port_pos(port)
        })
        .collect()
}

/// The part of segment `a`–`b` outside box `r`: unchanged if neither end is strictly inside,
/// `None` if both are, else clipped at the box boundary (from the outside end).
fn clip_outside(
    a: GridPt,
    b: GridPt,
    r: &GridRect,
    strictly_inside: &impl Fn(GridPt) -> bool,
) -> Option<(GridPt, GridPt)> {
    match (strictly_inside(a), strictly_inside(b)) {
        (false, false) => Some((a, b)),
        (true, true) => None,
        (ia, _) => {
            let (out, inn) = if ia { (b, a) } else { (a, b) };
            let edge = if out.y == inn.y {
                gp(if out.x < inn.x { r.min.x } else { r.max.x }, out.y)
            } else {
                gp(out.x, if out.y < inn.y { r.min.y } else { r.max.y })
            };
            (edge != out).then_some((out, edge))
        }
    }
}

fn closest_on_seg(p: GridPt, a: GridPt, b: GridPt) -> GridPt {
    gp(
        p.x.clamp(a.x.min(b.x), a.x.max(b.x)),
        p.y.clamp(a.y.min(b.y), a.y.max(b.y)),
    )
}

/// Package geometry is consistent: every instance and pin placed, every net wired.
pub fn geometry_complete(def: &ComponentDef, e: &EditorData) -> bool {
    e.covers(def)
        && def
            .nets
            .iter()
            .all(|n| e.wires.contains_key(&n.id) || n.ends.len() < 2)
}
