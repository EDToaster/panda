//! Layout invariants ([`check`]) and readability measures ([`metrics`]).

use std::collections::{HashMap, HashSet};

use panda_netlist::Endpoint;

use super::{net_geom, on_interior, on_segment};
use crate::doc::{Document, EditorData, ItemRef, Placed};
use crate::geom::{GridPt, GridRect, gp};

/// Axis-parallel wire pieces of every net of a def, indexed by line (`h`: by y, `v`: by x).
struct WireIndex<'a> {
    /// Net ids by index.
    names: Vec<&'a str>,
    /// Merged segments per net.
    segs: Vec<Vec<(GridPt, GridPt)>>,
    /// Horizontal pieces by y: `(x0, x1, net)`, sorted.
    h: HashMap<i32, Vec<(i32, i32, u32)>>,
    /// Vertical pieces by x: `(y0, y1, net)`, sorted.
    v: HashMap<i32, Vec<(i32, i32, u32)>>,
}

impl<'a> WireIndex<'a> {
    fn new(ed: &'a EditorData) -> WireIndex<'a> {
        let mut ix = WireIndex {
            names: Vec::with_capacity(ed.wires.len()),
            segs: Vec::with_capacity(ed.wires.len()),
            h: HashMap::new(),
            v: HashMap::new(),
        };
        for (ni, (id, g)) in ed.wires.iter().enumerate() {
            let ni = ni as u32;
            let segs: Vec<(GridPt, GridPt)> = net_geom(g.segments().collect()).segments().collect();
            for &(a, b) in &segs {
                if a.y == b.y {
                    ix.h.entry(a.y)
                        .or_default()
                        .push((a.x.min(b.x), a.x.max(b.x), ni));
                } else {
                    ix.v.entry(a.x)
                        .or_default()
                        .push((a.y.min(b.y), a.y.max(b.y), ni));
                }
            }
            ix.names.push(id);
            ix.segs.push(segs);
        }
        for iv in ix.h.values_mut().chain(ix.v.values_mut()) {
            iv.sort_unstable();
        }
        ix
    }

    /// Nets (other than `except`) whose wires contain point `p`.
    fn nets_at(&self, p: GridPt, except: Option<u32>) -> impl Iterator<Item = u32> + '_ {
        let hs = self.h.get(&p.y).into_iter().flatten().filter(move |s| s.0 <= p.x && p.x <= s.1);
        let vs = self.v.get(&p.x).into_iter().flatten().filter(move |s| s.0 <= p.y && p.y <= s.1);
        hs.chain(vs).map(|s| s.2).filter(move |&n| Some(n) != except)
    }

    /// Collinear overlaps (sharing more than a point) between pieces of different nets.
    fn overlaps(&self) -> Vec<(u32, u32, GridPt)> {
        let mut out = Vec::new();
        for (horizontal, lines) in [(true, &self.h), (false, &self.v)] {
            for (&k, iv) in lines {
                for i in 0..iv.len() {
                    for j in i + 1..iv.len() {
                        if iv[j].0 >= iv[i].1 {
                            break;
                        }
                        if iv[j].2 != iv[i].2 {
                            let at = if horizontal {
                                gp(iv[j].0, k)
                            } else {
                                gp(k, iv[j].0)
                            };
                            out.push((iv[i].2, iv[j].2, at));
                        }
                    }
                }
            }
        }
        out
    }

    /// Wire nodes (ends, corners, tees) of each net lying on another net's wire.
    fn touches(&self) -> Vec<(u32, u32, GridPt)> {
        let mut out = Vec::new();
        for (ni, segs) in self.segs.iter().enumerate() {
            let ni = ni as u32;
            // Directions present at every vertex: bit 0 left, 1 right, 2 up, 3 down.
            let mut dirs: HashMap<GridPt, u8> = HashMap::new();
            for &(a, b) in segs {
                let (lo, hi) = if (a.x, a.y) < (b.x, b.y) { (a, b) } else { (b, a) };
                let (dl, dh) = if a.y == b.y { (2u8, 1u8) } else { (8, 4) };
                *dirs.entry(lo).or_default() |= dl;
                *dirs.entry(hi).or_default() |= dh;
            }
            for (&p, &d) in &dirs {
                let straight = d == 0b0011 || d == 0b1100;
                if straight || segs.iter().any(|&(a, b)| on_interior(p, a, b)) {
                    continue;
                }
                if let Some(other) = self.nets_at(p, Some(ni)).next() {
                    out.push((ni, other, p));
                }
            }
        }
        out
    }
}

/// Invariant violations of a def's geometry (empty = fine): overlapping symbols, wires passing
/// through a symbol's interior, a net end not on its wires, a net whose wires are disconnected,
/// wires of different nets overlapping or touching, a wire over a port of another net.
pub fn check(doc: &Document, def_id: &str) -> Vec<String> {
    let mut errs = Vec::new();
    let Some(def) = doc.def(def_id) else {
        return vec![format!("no def {def_id}")];
    };
    let Some(ed) = doc.editor(def_id) else {
        return vec![format!("no geometry for {def_id}")];
    };
    let placed = doc.placed(def_id);
    if placed.len() != def.instances.len() + def.pins.len() {
        errs.push(format!(
            "{def_id}: {} of {} items placed",
            placed.len(),
            def.instances.len() + def.pins.len()
        ));
    }
    // Overlaps (sweep by x).
    let mut rects: Vec<(GridRect, &Placed)> = placed.iter().map(|p| (p.rect(), p)).collect();
    rects.sort_by_key(|r| r.0.min.x);
    for i in 0..rects.len() {
        for j in i + 1..rects.len() {
            if rects[j].0.min.x >= rects[i].0.max.x {
                break;
            }
            if rects[i].0.overlaps(&rects[j].0) {
                errs.push(format!(
                    "{def_id}: {:?} overlaps {:?}",
                    rects[i].1.item, rects[j].1.item
                ));
            }
        }
    }
    // Symbol interiors bucketed by 16-unit cells for the wire test.
    const CELL: i32 = 16;
    let mut cells: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
    for (i, (r, _)) in rects.iter().enumerate() {
        for cx in r.min.x.div_euclid(CELL)..=r.max.x.div_euclid(CELL) {
            for cy in r.min.y.div_euclid(CELL)..=r.max.y.div_euclid(CELL) {
                cells.entry((cx, cy)).or_default().push(i);
            }
        }
    }
    let by_item: HashMap<&ItemRef, &Placed> = placed.iter().map(|p| (&p.item, p)).collect();
    let ix = WireIndex::new(ed);
    let net_ix: HashMap<&str, u32> = ix.names.iter().enumerate().map(|(i, n)| (*n, i as u32)).collect();
    // Port positions with the net (index into `ix`) they belong to, if wired.
    let mut port_net: HashMap<GridPt, Option<u32>> = HashMap::new();
    for net in &def.nets {
        let ends: Vec<GridPt> = net
            .ends
            .iter()
            .filter_map(|e| {
                let (item, port) = match e {
                    Endpoint::Pin(n) => (ItemRef::Pin(n.clone()), n.as_str()),
                    Endpoint::Port { inst, port } => {
                        (ItemRef::Instance(inst.clone()), port.as_str())
                    }
                };
                by_item.get(&item)?.port_pos(port)
            })
            .collect();
        for &p in &ends {
            port_net.insert(p, net_ix.get(net.id.as_str()).copied());
        }
        if ends.len() < 2 {
            continue;
        }
        let Some(g) = ed.wires.get(&net.id) else {
            errs.push(format!("{def_id}: net {} has no wires", net.id));
            continue;
        };
        let segs: Vec<(GridPt, GridPt)> = g.segments().collect();
        for &p in &ends {
            if !segs.iter().any(|&(a, b)| on_segment(p, a, b)) {
                errs.push(format!("{def_id}: net {}: end {p:?} not on a wire", net.id));
            }
        }
        if pieces(&segs) > 1 {
            errs.push(format!(
                "{def_id}: net {} wires form {} pieces",
                net.id,
                pieces(&segs)
            ));
        }
        // Wires outside symbol interiors.
        let mut seen: HashSet<usize> = HashSet::new();
        for &(a, b) in &segs {
            let lo = gp(a.x.min(b.x), a.y.min(b.y));
            let hi = gp(a.x.max(b.x), a.y.max(b.y));
            seen.clear();
            for cx in lo.x.div_euclid(CELL)..=hi.x.div_euclid(CELL) {
                for cy in lo.y.div_euclid(CELL)..=hi.y.div_euclid(CELL) {
                    for &i in cells.get(&(cx, cy)).into_iter().flatten() {
                        if !seen.insert(i) {
                            continue;
                        }
                        let (r, p) = &rects[i];
                        let inside = if a.y == b.y {
                            a.y > r.min.y && a.y < r.max.y && lo.x < r.max.x && hi.x > r.min.x
                        } else {
                            a.x > r.min.x && a.x < r.max.x && lo.y < r.max.y && hi.y > r.min.y
                        };
                        if inside {
                            errs.push(format!(
                                "{def_id}: net {} wire {a:?}-{b:?} crosses {:?}",
                                net.id, p.item
                            ));
                        }
                    }
                }
            }
        }
    }
    // Every port of every item, wired or not: no foreign wire on it.
    for p in &placed {
        for port in &p.symbol.ports {
            let at = p.geom.pos + p.geom.orient.apply(port.pos, p.symbol.size);
            let own = port_net.get(&at).copied().flatten();
            if let Some(n) = ix.nets_at(at, own).next() {
                errs.push(format!(
                    "{def_id}: net {} runs over port {}.{} at {at:?}",
                    ix.names[n as usize], item_name(p), port.name
                ));
            }
        }
    }
    for (a, b, at) in ix.overlaps() {
        errs.push(format!(
            "{def_id}: nets {} and {} overlap at {at:?}",
            ix.names[a as usize], ix.names[b as usize]
        ));
    }
    for (a, b, at) in ix.touches() {
        errs.push(format!(
            "{def_id}: net {} touches net {} at {at:?}",
            ix.names[a as usize], ix.names[b as usize]
        ));
    }
    errs
}

/// Connected pieces of a set of segments (union-find over touching segments).
fn pieces(segs: &[(GridPt, GridPt)]) -> usize {
    let mut parent: Vec<usize> = (0..segs.len()).collect();
    fn find(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    for i in 0..segs.len() {
        for j in i + 1..segs.len() {
            let (a, b) = segs[i];
            let (c, d) = segs[j];
            if on_segment(c, a, b) || on_segment(d, a, b) || on_segment(a, c, d) || on_segment(b, c, d)
            {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                parent[ri] = rj;
            }
        }
    }
    let roots: HashSet<usize> = (0..segs.len()).map(|i| find(&mut parent, i)).collect();
    roots.len()
}

/// Readability measures of a def's geometry (see [`metrics`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Metrics {
    /// Bounding box width (grid units).
    pub width: i32,
    /// Bounding box height (grid units).
    pub height: i32,
    /// Total wire length (grid units).
    pub wire_len: i64,
    /// Wire segments (after merging collinear pieces).
    pub segments: usize,
    /// Bends: vertices where a net's wire turns (corners, not tees).
    pub bends: usize,
    /// Proper crossings between wires of different nets.
    pub crossings: usize,
    /// Collinear overlaps between wires of different nets (visual shorts; [`check`] rejects).
    pub overlaps: usize,
    /// Wire nodes of one net lying on another net's wire (look connected; [`check`] rejects).
    pub touches: usize,
}

/// Measure a def's geometry: size, wire length, bends, crossings, and the two wire faults
/// [`check`] rejects (overlapping nets, touching nets).
pub fn metrics(doc: &Document, def_id: &str) -> Metrics {
    let mut m = Metrics::default();
    let Some(ed) = doc.editor(def_id) else {
        return m;
    };
    let ix = WireIndex::new(ed);
    let mut bb: Option<GridRect> = doc
        .placed(def_id)
        .iter()
        .map(Placed::rect)
        .reduce(|a, b| a.union(&b));
    for segs in &ix.segs {
        let mut deg: HashMap<GridPt, (u8, u8)> = HashMap::new();
        for &(a, b) in segs {
            let r = GridRect {
                min: gp(a.x.min(b.x), a.y.min(b.y)),
                max: gp(a.x.max(b.x), a.y.max(b.y)),
            };
            bb = Some(bb.map_or(r, |x| x.union(&r)));
            m.segments += 1;
            m.wire_len += i64::from((a.x - b.x).abs() + (a.y - b.y).abs());
            for p in [a, b] {
                let d = deg.entry(p).or_default();
                if a.y == b.y {
                    d.0 += 1;
                } else {
                    d.1 += 1;
                }
            }
        }
        m.bends += deg
            .iter()
            .filter(|&(&p, &(h, v))| {
                h > 0 && v > 0 && !segs.iter().any(|&(a, b)| on_interior(p, a, b))
            })
            .count();
    }
    if let Some(b) = bb {
        m.width = b.max.x - b.min.x;
        m.height = b.max.y - b.min.y;
    }
    // Crossings: every horizontal's interior x against the verticals there.
    for (&y, iv) in &ix.h {
        for &(x0, x1, n) in iv {
            for x in x0 + 1..x1 {
                if let Some(vv) = ix.v.get(&x) {
                    m.crossings += vv
                        .iter()
                        .filter(|&&(y0, y1, k)| k != n && y > y0 && y < y1)
                        .count();
                }
            }
        }
    }
    m.overlaps = ix.overlaps().len();
    m.touches = ix.touches().len();
    m
}

/// Display name of an item (instance id or pin name).
fn item_name(p: &Placed) -> &str {
    match &p.item {
        ItemRef::Instance(s) | ItemRef::Pin(s) => s,
    }
}
