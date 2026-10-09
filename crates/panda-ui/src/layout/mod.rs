//! Automatic layered layout and orthogonal routing for defs without geometry.
//!
//! A Sugiyama-style pipeline, sized for generated designs (Panda-8's 256-input ROM OR, decoder
//! fan-outs, latch feedback, the 1.8k-instance flat CPU):
//!
//! 0. **Groups.** In flat designs instance ids carry the hierarchy they were inlined from
//!    (`cpu.alu.add.g3`, separator `.`). Instances sharing a prefix are laid out on their own,
//!    recursively, and the result is placed as one rectangular *block* in the enclosing layout;
//!    nets crossing a block's boundary enter on its left (driven outside) or leave on its right
//!    (driven inside). Defs without dotted ids are one scope.
//! 1. **Graph.** Nodes are pin markers, instances and blocks. For every net, driver ends (`Out`
//!    ports, the def's `In` pins; for undriven nets the `InOut` ends of splitters fed from
//!    elsewhere) get an edge to every other end.
//! 2. **Layers.** Cycles (feedback) are broken by DFS back edges; longest-path layering on the
//!    rest, then nodes with more successors than predecessors are pulled right toward them
//!    (shorter nets, fewer lanes). `In` pins form column 0, `Out`/`InOut` pins the last column,
//!    sources without inputs (constants, clocks) sit right next to their first consumer.
//! 3. **Lanes.** A net whose ends lie in channels `c0..=c1` (the gaps left of/right of the
//!    columns) gets one *lane* (a horizontal slot) in every column it crosses, so wires never
//!    pass through symbols, also for feedback going right-to-left.
//! 4. **Order.** Crossing reduction ([`order`]): barycenter sweeps with port offsets plus
//!    adjacent-swap transposes, keeping the order with the fewest crossings.
//! 5. **Y.** Order-preserving 1-D placement that minimizes the squared distance of every item to
//!    its desired position (aligned with the net it connects to on the left, else on the right),
//!    solved exactly by pool-adjacent-violators; three alternating passes.
//! 6. **Channels.** Per channel, every net with ends at different heights gets a vertical track
//!    ([`tracks`]): no overlapping stubs (doglegs break constraint cycles), bundle-friendly track
//!    order (parallel buses route as crossing-free staircases), left-edge packing. Column x
//!    positions follow from channel widths.
//!
//! The result is an [`EditorData`] with `auto = true`. [`check`] verifies the invariants (no
//! overlapping symbols, wires outside symbols, every net's ends connected by its wires, no wires
//! of different nets overlapping or touching); [`metrics`] measures readability (size, wire
//! length, bends, crossings).

mod check;
mod order;
mod tracks;

use std::collections::{BTreeMap, HashMap, HashSet};

use panda_netlist::{ComponentDef, DefId, Design, Dir, Endpoint, InstKind};

pub use check::{Metrics, check, metrics};

#[cfg(test)]
use crate::doc::{Document, ItemRef};
use crate::doc::{EditorData, InstGeom, NetGeom};
use crate::geom::{
    GridPt, GridRect, Orient, Side, Symbol, SymbolKind, SymbolLayout, atom_symbol, gp,
    package_symbol, pin_symbol,
};
use order::{Ctx, Item};
use tracks::{ChanPt, Piece, route_channel};

/// Vertical gap between two symbols in a column.
const NODE_GAP: i32 = 2;
/// Vertical gap next to a lane.
const LANE_GAP: i32 = 1;
/// Minimum channel width.
const MIN_CHANNEL: i32 = 3;
/// Smallest prefix group laid out as a block.
const MIN_GROUP: usize = 3;
/// Scopes with more instances than this lay out their blocks on several threads.
const PARALLEL_SCOPE: usize = 600;
/// Height of the name row above a block.
const LABEL_ROW: i32 = 2;
/// Approximate width of a label character (grid units, canvas text at size 1).
const LABEL_CHAR_W: f32 = 0.5;
/// Separator of hierarchical instance ids in flat designs (`panda_circuits::PATH_SEP`).
const PATH_SEP: char = '.';

/// Lay out the defs `ids` of `design` (in parallel). Package symbols of instantiated defs come
/// from `editors` (their `symbol`), else the automatic symbol.
pub fn layout_defs(
    design: &Design,
    editors: &HashMap<DefId, EditorData>,
    ids: &[DefId],
) -> HashMap<DefId, EditorData> {
    let symbols = |id: &str| editors.get(id).and_then(|e| e.symbol.as_ref());
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(ids.len().max(1));
    // Big defs first, dealt round-robin, so threads get similar loads.
    let mut order: Vec<&DefId> = ids.iter().collect();
    let size = |id: &DefId| design.def(id).map_or(0, |d| d.instances.len());
    order.sort_by_key(|id| std::cmp::Reverse(size(id)));
    let mut chunks: Vec<Vec<&DefId>> = vec![Vec::new(); threads];
    for (i, id) in order.into_iter().enumerate() {
        chunks[i % threads].push(id);
    }
    std::thread::scope(|s| {
        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                s.spawn(move || {
                    chunk
                        .into_iter()
                        .filter_map(|id| {
                            let def = design.def(id)?;
                            let mut e = layout_def(design, def, symbols);
                            // Keep a hand-made symbol layout.
                            e.symbol = editors.get(id).and_then(|x| x.symbol.clone());
                            Some((id.clone(), e))
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("layout thread"))
            .collect()
    })
}

/// What a layout node stands for.
#[derive(Clone, Debug)]
enum What {
    /// The def's own pin (by index).
    Pin(usize),
    /// An instance (by index).
    Inst(usize),
    /// A block's boundary pin for a def net, `In` (driven outside) or `Out`.
    VPin(Dir, usize),
    /// A laid-out group (index into the scope's blocks).
    Block(usize),
}

/// One node of the layout graph.
struct Node {
    what: What,
    sym: Symbol,
    orient: Orient,
    layer: usize,
    /// Left x (set when the columns are placed).
    x: i32,
    /// Top y.
    y: i32,
    /// Extra height below the symbol (bottom ports route one unit down).
    extra: i32,
}

impl Node {
    fn new(what: What, sym: Symbol) -> Node {
        Node {
            what,
            sym,
            orient: Orient::default(),
            layer: 0,
            x: 0,
            y: 0,
            extra: 0,
        }
    }
    fn size(&self) -> GridPt {
        self.orient.size(self.sym.size)
    }
    fn height(&self) -> i32 {
        self.size().y + self.extra
    }
    /// Local (rotated) position and side of port `p`.
    fn port(&self, p: usize) -> (GridPt, Side) {
        let pg = &self.sym.ports[p];
        (
            self.orient.apply(pg.pos, self.sym.size),
            self.orient.side(pg.side),
        )
    }
    /// A pin marker on the input side (column 0).
    fn is_in_pin(&self) -> bool {
        match self.what {
            What::Pin(_) => self.sym.ports[0].dir == Dir::In,
            What::VPin(d, _) => d == Dir::In,
            _ => false,
        }
    }
    /// A pin marker on the output side (last column).
    fn is_out_pin(&self) -> bool {
        match self.what {
            What::Pin(_) => self.sym.ports[0].dir != Dir::In,
            What::VPin(d, _) => d != Dir::In,
            _ => false,
        }
    }
    /// A pin marker (either side).
    fn is_pin_marker(&self) -> bool {
        matches!(self.what, What::Pin(_) | What::VPin(..))
    }
    /// Role of port `p` for the net it is on: `Out` drives it.
    fn role(&self, p: usize) -> Dir {
        let d = self.sym.ports[p].dir;
        match self.what {
            // The def's own pins drive the inside (In) or are driven (Out); so do block pins.
            What::Pin(_) | What::VPin(..) => match d {
                Dir::In => Dir::Out,
                Dir::Out => Dir::In,
                Dir::InOut => Dir::InOut,
            },
            _ => d,
        }
    }
}

/// End of a net at a node port.
#[derive(Clone, Copy)]
struct End {
    node: usize,
    port: usize,
}

/// An end of a def net.
#[derive(Clone, Copy, Debug)]
enum NetEnd {
    /// The def's pin (index).
    Pin(usize),
    /// Instance (index) and port (index into its symbol's ports).
    Port(usize, usize),
}

/// Per-def data shared by all scopes.
struct DefCtx<'a> {
    def: &'a ComponentDef,
    /// Symbol per instance (`None`: a def use whose def is missing; not laid out).
    syms: Vec<Option<Symbol>>,
    /// Id path components per instance.
    path: Vec<Vec<&'a str>>,
    /// Ends of every def net.
    net_ends: Vec<Vec<NetEnd>>,
    /// Nets per instance.
    inst_nets: Vec<Vec<usize>>,
}

/// A laid-out scope (local coordinates).
#[derive(Default)]
struct ScopeOut {
    /// Instance placements.
    insts: Vec<(usize, InstGeom)>,
    /// Def pin placements (top scope only).
    pins: Vec<(usize, InstGeom)>,
    /// Wire segments by def net.
    wires: HashMap<usize, Vec<(GridPt, GridPt)>>,
    /// Boundary pins: net, direction, port position.
    vports: Vec<(usize, Dir, GridPt)>,
    /// Group name labels (top-left anchor, text).
    labels: Vec<(GridPt, String)>,
    /// Bounding box of everything.
    bounds: Option<GridRect>,
}

impl ScopeOut {
    fn grow(&mut self, r: GridRect) {
        self.bounds = Some(self.bounds.map_or(r, |b| b.union(&r)));
    }

    /// Move everything by `d`.
    fn translate(&mut self, d: GridPt) {
        for (_, g) in self.insts.iter_mut().chain(self.pins.iter_mut()) {
            g.pos = g.pos + d;
        }
        for segs in self.wires.values_mut() {
            for s in segs {
                s.0 = s.0 + d;
                s.1 = s.1 + d;
            }
        }
        for v in &mut self.vports {
            v.2 = v.2 + d;
        }
        for l in &mut self.labels {
            l.0 = l.0 + d;
        }
        if let Some(b) = &mut self.bounds {
            b.min = b.min + d;
            b.max = b.max + d;
        }
    }

    /// The block symbol of this scope: its box, `In` boundary pins on the left edge, the others
    /// on the right. Port order is `vports` order.
    fn block_symbol(&self) -> Symbol {
        let b = self.bounds.unwrap_or(GridRect {
            min: gp(0, 0),
            max: gp(0, 0),
        });
        let ports = self
            .vports
            .iter()
            .map(|&(net, dir, p)| {
                let side = if dir == Dir::In {
                    Side::Left
                } else {
                    Side::Right
                };
                debug_assert_eq!(p.x, if side == Side::Left { b.min.x } else { b.max.x });
                crate::geom::PortGeom {
                    name: net.to_string(),
                    pos: p - b.min,
                    side,
                    dir,
                    width: 1,
                }
            })
            .collect();
        Symbol {
            kind: SymbolKind::Package {
                def: String::new(),
                title: String::new(),
            },
            size: b.max - b.min,
            ports,
        }
    }
}

/// Lay out one def. `symbols` gives package symbol layouts by def id.
pub fn layout_def<'a>(
    design: &'a Design,
    def: &'a ComponentDef,
    symbols: impl Fn(&str) -> Option<&'a SymbolLayout>,
) -> EditorData {
    let syms: Vec<Option<Symbol>> = def
        .instances
        .iter()
        .map(|inst| match &inst.kind {
            InstKind::Atom(a) => Some(atom_symbol(a)),
            InstKind::Def(id) => design.def(id).map(|d| package_symbol(d, symbols(id))),
        })
        .collect();
    let inst_ix: HashMap<&str, usize> = def
        .instances
        .iter()
        .enumerate()
        .map(|(i, x)| (x.id.as_str(), i))
        .collect();
    let pin_ix: HashMap<&str, usize> = def
        .pins
        .iter()
        .enumerate()
        .map(|(i, p)| (p.name.as_str(), i))
        .collect();
    let mut inst_nets: Vec<Vec<usize>> = vec![Vec::new(); def.instances.len()];
    let net_ends: Vec<Vec<NetEnd>> = def
        .nets
        .iter()
        .enumerate()
        .map(|(ni, net)| {
            net.ends
                .iter()
                .filter_map(|e| match e {
                    Endpoint::Pin(name) => pin_ix.get(name.as_str()).map(|&p| NetEnd::Pin(p)),
                    Endpoint::Port { inst, port } => {
                        let i = *inst_ix.get(inst.as_str())?;
                        let p = syms[i].as_ref()?.ports.iter().position(|p| p.name == *port)?;
                        if inst_nets[i].last() != Some(&ni) {
                            inst_nets[i].push(ni);
                        }
                        Some(NetEnd::Port(i, p))
                    }
                })
                .collect()
        })
        .collect();
    let cx = DefCtx {
        def,
        path: def
            .instances
            .iter()
            .map(|i| i.id.split(PATH_SEP).collect())
            .collect(),
        syms,
        net_ends,
        inst_nets,
    };
    let all: Vec<usize> = (0..def.instances.len())
        .filter(|&i| cx.syms[i].is_some())
        .collect();
    let mut scope = layout_scope(&cx, &all, 0, None, "");
    // Normalize: content starts at (0, 0) like a hand-drawn schematic would.
    if let Some(b) = scope.bounds {
        scope.translate(gp(-b.min.x, -b.min.y));
    }
    let mut out = EditorData {
        version: crate::doc::EDITOR_VERSION,
        auto: true,
        ..EditorData::default()
    };
    for (i, g) in scope.insts {
        out.instances.insert(def.instances[i].id.clone(), g);
    }
    for (p, g) in scope.pins {
        out.pins.insert(def.pins[p].name.clone(), g);
    }
    for (ni, segs) in scope.wires {
        if !segs.is_empty() {
            out.wires.insert(def.nets[ni].id.clone(), net_geom(segs));
        }
    }
    out.labels = scope
        .labels
        .into_iter()
        .map(|(pos, text)| crate::doc::Label { pos, text })
        .collect();
    out
}

/// Split `insts` (sharing their first `depth` id components) into prefix groups and leaves.
/// Descends while everything is in one group. Returns `(depth, groups, leaves)`.
fn split_groups(
    cx: &DefCtx,
    insts: &[usize],
    mut depth: usize,
) -> (usize, Vec<Vec<usize>>, Vec<usize>) {
    loop {
        let mut groups: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        let mut leaves = Vec::new();
        for &i in insts {
            let p = &cx.path[i];
            if p.len() > depth + 1 {
                groups.entry(p[depth]).or_default().push(i);
            } else {
                leaves.push(i);
            }
        }
        if leaves.is_empty() && groups.len() == 1 {
            depth += 1;
            continue;
        }
        let mut out = Vec::new();
        for (_, g) in groups {
            if g.len() >= MIN_GROUP && g.len() < insts.len() {
                out.push(g);
            } else {
                leaves.extend(g);
            }
        }
        leaves.sort_unstable();
        return (depth, out, leaves);
    }
}

/// Lay out the instances `insts` (sharing their first `depth` id components). `boundary` lists
/// the nets crossing the scope's edge with their direction (`None`: the def's top scope, whose
/// boundary is the def's pins). A block (`boundary` given) gets a label row with `name` on top.
fn layout_scope(
    cx: &DefCtx,
    insts: &[usize],
    depth: usize,
    boundary: Option<&[(usize, Dir)]>,
    name: &str,
) -> ScopeOut {
    let (depth, groups, leaves) = split_groups(cx, insts, depth);
    // ---- blocks (recursively, in parallel for big scopes) ----
    let mut scope_of = HashMap::with_capacity(insts.len());
    for (g, members) in groups.iter().enumerate() {
        for &i in members {
            scope_of.insert(i, g);
        }
    }
    let bounds: Vec<Vec<(usize, Dir)>> = groups
        .iter()
        .enumerate()
        .map(|(g, members)| group_boundary(cx, members, |i| scope_of.get(&i) == Some(&g)))
        .collect();
    let names: Vec<String> = groups
        .iter()
        .map(|m| cx.path[m[0]][..=depth].join(&PATH_SEP.to_string()))
        .collect();
    let blocks: Vec<ScopeOut> = if insts.len() > PARALLEL_SCOPE && groups.len() > 1 {
        std::thread::scope(|s| {
            let hs: Vec<_> = groups
                .iter()
                .zip(&bounds)
                .zip(&names)
                .map(|((m, b), n)| s.spawn(move || layout_scope(cx, m, depth + 1, Some(b), n)))
                .collect();
            hs.into_iter()
                .map(|h| h.join().expect("layout thread"))
                .collect()
        })
    } else {
        groups
            .iter()
            .zip(&bounds)
            .zip(&names)
            .map(|((m, b), n)| layout_scope(cx, m, depth + 1, Some(b), n))
            .collect()
    };

    // ---- nodes ----
    let mut nodes: Vec<Node> = Vec::new();
    // Node of each def pin / boundary net, leaf instance and block.
    let mut pin_node: HashMap<usize, usize> = HashMap::new();
    let mut vpin_node: HashMap<usize, usize> = HashMap::new();
    let mut inst_node: HashMap<usize, usize> = HashMap::new();
    match boundary {
        None => {
            for (pi, p) in cx.def.pins.iter().enumerate() {
                pin_node.insert(pi, nodes.len());
                nodes.push(Node::new(What::Pin(pi), pin_symbol(p)));
            }
        }
        Some(b) => {
            for &(ni, dir) in b {
                vpin_node.insert(ni, nodes.len());
                nodes.push(Node::new(What::VPin(dir, ni), vpin_symbol(dir)));
            }
        }
    }
    for &i in &leaves {
        inst_node.insert(i, nodes.len());
        let sym = cx.syms[i].clone().expect("laid-out instances have symbols");
        nodes.push(Node::new(What::Inst(i), sym));
    }
    // Block port index by (block, net).
    let mut block_port: HashMap<(usize, usize), usize> = HashMap::new();
    let mut block_node = Vec::with_capacity(blocks.len());
    for (bi, b) in blocks.iter().enumerate() {
        for (pi, &(ni, _, _)) in b.vports.iter().enumerate() {
            block_port.insert((bi, ni), pi);
        }
        block_node.push(nodes.len());
        nodes.push(Node::new(What::Block(bi), b.block_symbol()));
    }

    // ---- nets of this scope ----
    let mut scope_nets: Vec<usize> = insts
        .iter()
        .flat_map(|&i| cx.inst_nets[i].iter().copied())
        .collect();
    if boundary.is_none() {
        for (ni, ends) in cx.net_ends.iter().enumerate() {
            if ends.iter().any(|e| matches!(e, NetEnd::Pin(_))) {
                scope_nets.push(ni);
            }
        }
    }
    scope_nets.sort_unstable();
    scope_nets.dedup();
    let mut nets: Vec<Vec<End>> = Vec::with_capacity(scope_nets.len());
    for &ni in &scope_nets {
        let mut ends = Vec::new();
        let mut seen_blocks: HashSet<usize> = HashSet::new();
        for e in &cx.net_ends[ni] {
            match *e {
                NetEnd::Pin(p) => {
                    if let Some(&n) = pin_node.get(&p) {
                        ends.push(End { node: n, port: 0 });
                    }
                }
                NetEnd::Port(i, p) => {
                    if let Some(&n) = inst_node.get(&i) {
                        ends.push(End { node: n, port: p });
                    } else if let Some(&g) = scope_of.get(&i)
                        && seen_blocks.insert(g)
                        && let Some(&bp) = block_port.get(&(g, ni))
                    {
                        ends.push(End {
                            node: block_node[g],
                            port: bp,
                        });
                    }
                }
            }
        }
        if let Some(&n) = vpin_node.get(&ni) {
            ends.push(End { node: n, port: 0 });
        }
        nets.push(ends);
    }

    // ---- solve ----
    let segs = solve(&mut nodes, &nets);

    // ---- emit ----
    let mut out = ScopeOut::default();
    for (k, s) in segs.into_iter().enumerate() {
        for &(a, b) in &s {
            out.grow(GridRect {
                min: gp(a.x.min(b.x), a.y.min(b.y)),
                max: gp(a.x.max(b.x), a.y.max(b.y)),
            });
        }
        if !s.is_empty() {
            out.wires.entry(scope_nets[k]).or_default().extend(s);
        }
    }
    let mut blocks: Vec<Option<ScopeOut>> = blocks.into_iter().map(Some).collect();
    for n in &nodes {
        let pos = gp(n.x, n.y);
        let rect = GridRect::from_pos_size(pos, n.size());
        match n.what {
            What::Pin(p) => {
                out.grow(rect);
                out.pins.push((
                    p,
                    InstGeom {
                        pos,
                        orient: n.orient,
                    },
                ));
            }
            What::Inst(i) => {
                out.grow(rect);
                out.insts.push((
                    i,
                    InstGeom {
                        pos,
                        orient: n.orient,
                    },
                ));
            }
            What::VPin(dir, net) => {
                out.grow(rect);
                out.vports.push((net, dir, pos + n.port(0).0));
            }
            What::Block(bi) => {
                let mut b = blocks[bi].take().expect("block placed once");
                let min = b.bounds.map_or(gp(0, 0), |r| r.min);
                b.translate(pos - min);
                out.grow(rect);
                out.insts.extend(b.insts);
                out.labels.extend(b.labels);
                for (ni, s) in b.wires {
                    out.wires.entry(ni).or_default().extend(s);
                }
            }
        }
    }
    // Boundary pins in the order given (block port order).
    if let Some(b) = boundary {
        let rank: HashMap<usize, usize> = b.iter().enumerate().map(|(i, &(n, _))| (n, i)).collect();
        out.vports.sort_by_key(|v| rank[&v.0]);
        // The name row above the block's content.
        if let Some(r) = out.bounds {
            let at = gp(r.min.x, r.min.y - LABEL_ROW);
            out.grow(GridRect { min: at, max: at });
            out.labels.push((at, fit_label(name, r.max.x - r.min.x)));
        }
    }
    out
}

/// `name`, shortened from the left (whole components, `…` prefix) to fit `width` grid units.
fn fit_label(name: &str, width: i32) -> String {
    let room = (width.max(0) as f32 / LABEL_CHAR_W) as usize;
    if name.chars().count() <= room {
        return name.to_string();
    }
    let mut parts: Vec<&str> = name.split(PATH_SEP).collect();
    while parts.len() > 1 {
        parts.remove(0);
        let s = format!("…{}", parts.join(&PATH_SEP.to_string()));
        if s.chars().count() <= room {
            return s;
        }
    }
    parts[0].chars().take(room).collect()
}

/// Nets of the group `members` that also have ends outside it (or on a def pin), with their
/// direction: `Out` if driven inside the group.
fn group_boundary(
    cx: &DefCtx,
    members: &[usize],
    inside: impl Fn(usize) -> bool,
) -> Vec<(usize, Dir)> {
    let mut nets: Vec<usize> = members
        .iter()
        .flat_map(|&i| cx.inst_nets[i].iter().copied())
        .collect();
    nets.sort_unstable();
    nets.dedup();
    let dir_of = |e: &NetEnd| match *e {
        NetEnd::Pin(_) => None,
        NetEnd::Port(i, p) => cx.syms[i].as_ref().map(|s| s.ports[p].dir),
    };
    nets.into_iter()
        .filter_map(|ni| {
            let ends = &cx.net_ends[ni];
            let crosses = ends.iter().any(|e| match *e {
                NetEnd::Pin(_) => true,
                NetEnd::Port(i, _) => !inside(i),
            });
            if !crosses {
                return None;
            }
            let is_in = |e: &NetEnd| matches!(*e, NetEnd::Port(i, _) if inside(i));
            // Def `In` pins drive the net from outside; `Out` ports drive it where they are.
            let driven_inside = ends.iter().any(|e| is_in(e) && dir_of(e) == Some(Dir::Out));
            let driven = ends.iter().any(|e| match *e {
                NetEnd::Pin(p) => cx.def.pins[p].dir == Dir::In,
                _ => dir_of(e) == Some(Dir::Out),
            });
            // Undriven (splitter-only) nets: the group holding the first end drives it.
            let first_inside = ends.first().is_some_and(is_in);
            let dir = if driven_inside || (!driven && first_inside) {
                Dir::Out
            } else {
                Dir::In
            };
            Some((ni, dir))
        })
        .collect()
}

/// Symbol of a block boundary pin: a zero-width marker with its port at mid height.
fn vpin_symbol(dir: Dir) -> Symbol {
    let side = if dir == Dir::In {
        Side::Right
    } else {
        Side::Left
    };
    Symbol {
        kind: SymbolKind::Pin {
            dir,
            name: String::new(),
        },
        size: gp(0, 2),
        ports: vec![crate::geom::PortGeom {
            name: String::new(),
            pos: gp(0, 1),
            side,
            dir,
            width: 1,
        }],
    }
}


/// Place `nodes` (sets `layer`, `x`, `y`, `orient`) and route `nets`; returns the wire segments
/// of every net (same indices as `nets`).
fn solve(nodes: &mut Vec<Node>, nets: &[Vec<End>]) -> Vec<Vec<(GridPt, GridPt)>> {
    let n_nodes = nodes.len();
    // ---- roles: drivers of each net ----
    let role = |nodes: &[Node], e: &End| nodes[e.node].role(e.port);
    let has_real_driver: Vec<bool> = nets
        .iter()
        .map(|ends| ends.iter().any(|e| role(nodes, e) == Dir::Out))
        .collect();
    // Splitters fed through some port from a really driven net.
    let mut fed = vec![false; n_nodes];
    for (ni, ends) in nets.iter().enumerate() {
        if has_real_driver[ni] {
            for e in ends {
                if role(nodes, e) == Dir::InOut {
                    fed[e.node] = true;
                }
            }
        }
    }
    let drivers: Vec<Vec<usize>> = nets
        .iter()
        .enumerate()
        .map(|(ni, ends)| {
            let real: Vec<usize> = (0..ends.len())
                .filter(|&i| role(nodes, &ends[i]) == Dir::Out)
                .collect();
            if !real.is_empty() || has_real_driver[ni] {
                return real;
            }
            let fed_io: Vec<usize> = (0..ends.len())
                .filter(|&i| role(nodes, &ends[i]) == Dir::InOut && fed[ends[i].node])
                .collect();
            if !fed_io.is_empty() {
                return fed_io;
            }
            (0..ends.len())
                .filter(|&i| role(nodes, &ends[i]) == Dir::InOut)
                .take(1)
                .collect()
        })
        .collect();

    // A splitter whose `whole` drives its net is a joiner: mirror it (whole on the right).
    for (ni, ends) in nets.iter().enumerate() {
        for &d in &drivers[ni] {
            let e = ends[d];
            let n = &mut nodes[e.node];
            if matches!(n.sym.kind, SymbolKind::Splitter { .. }) && n.sym.ports[e.port].name == "whole"
            {
                n.orient.flip = true;
            }
        }
    }

    // ---- graph ----
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    for (ni, ends) in nets.iter().enumerate() {
        for &d in &drivers[ni] {
            let a = ends[d].node;
            for (i, e) in ends.iter().enumerate() {
                if i != d && e.node != a && !drivers[ni].contains(&i) {
                    succ[a].push(e.node);
                }
            }
        }
    }
    for s in &mut succ {
        s.sort_unstable();
        s.dedup();
    }
    let dag = break_cycles(&succ);
    let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    for (a, ss) in dag.iter().enumerate() {
        for &b in ss {
            pred[b].push(a);
        }
    }

    // ---- layers ----
    let topo = topo_order(&dag, &pred);
    let is_in_pin: Vec<bool> = nodes.iter().map(Node::is_in_pin).collect();
    let is_out_pin: Vec<bool> = nodes.iter().map(Node::is_out_pin).collect();
    let mut layer = vec![1usize; n_nodes];
    for &v in &topo {
        if is_in_pin[v] {
            layer[v] = 0;
            continue;
        }
        if is_out_pin[v] {
            continue;
        }
        layer[v] = pred[v]
            .iter()
            .filter(|&&u| !is_out_pin[u])
            .map(|&u| layer[u] + 1)
            .max()
            .unwrap_or(1)
            .max(1);
    }
    // Pull sources (no predecessors) right next to their first consumer.
    for &v in topo.iter().rev() {
        if pred[v].is_empty() && !is_in_pin[v] && !is_out_pin[v] {
            let m = dag[v]
                .iter()
                .filter(|&&s| !is_out_pin[s])
                .map(|&s| layer[s])
                .min();
            if let Some(m) = m {
                layer[v] = m.saturating_sub(1).max(1);
            }
        }
    }
    let last_inner = (0..n_nodes)
        .filter(|&i| !is_out_pin[i])
        .map(|i| layer[i])
        .max()
        .unwrap_or(0);
    // Balance: a node with more successors than predecessors moves right toward them (as far
    // as its successors allow), which shortens its nets.
    for &v in topo.iter().rev() {
        if is_in_pin[v] || is_out_pin[v] || pred[v].is_empty() {
            continue;
        }
        let ns = dag[v].len();
        if ns <= pred[v].len() {
            continue;
        }
        let hi = dag[v]
            .iter()
            .map(|&s| if is_out_pin[s] { last_inner } else { layer[s] - 1 })
            .min()
            .unwrap_or(layer[v]);
        if hi > layer[v] {
            layer[v] = hi;
        }
    }
    for (i, l) in layer.iter_mut().enumerate() {
        if is_out_pin[i] {
            *l = last_inner + 1;
        }
    }
    let n_cols = layer.iter().copied().max().map_or(1, |m| m + 1);
    for (i, n) in nodes.iter_mut().enumerate() {
        n.layer = layer[i];
        n.extra = i32::from(
            n.sym
                .ports
                .iter()
                .any(|p| n.orient.side(p.side) == Side::Bottom),
        );
    }

    // ---- channels touched by each end; lanes ----
    // Channel c lies left of column c; a right-side port of column k is in channel k + 1.
    let end_channel = |e: &End, nodes: &[Node]| -> usize {
        let n = &nodes[e.node];
        match n.port(e.port).1 {
            Side::Right => n.layer + 1,
            _ => n.layer,
        }
    };
    let mut lanes: Vec<(usize, usize)> = Vec::new(); // (net, column)
    for (ni, ends) in nets.iter().enumerate() {
        if ends.len() < 2 {
            continue;
        }
        let cs = ends.iter().map(|e| end_channel(e, nodes));
        let (c0, c1) = cs.fold((usize::MAX, 0), |(a, b), c| (a.min(c), b.max(c)));
        for col in c0..c1 {
            lanes.push((ni, col));
        }
    }

    // ---- columns, order, y ----
    let mut cols: Vec<Vec<Item>> = vec![Vec::new(); n_cols];
    for (i, n) in nodes.iter().enumerate() {
        cols[n.layer].push(Item::Node(i));
    }
    for (li, &(_, col)) in lanes.iter().enumerate() {
        cols[col].push(Item::Lane(li));
    }
    let mut node_nets: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_nodes];
    for (ni, ends) in nets.iter().enumerate() {
        if ends.len() < 2 {
            continue;
        }
        for e in ends {
            node_nets[e.node].push((ni, e.port));
        }
    }
    let mut ctx = Ctx {
        nodes: std::mem::take(nodes),
        lanes,
        node_nets,
        cols,
        ys: Vec::new(),
    };
    ctx.order();
    ctx.place(3);
    let Ctx {
        nodes: placed,
        lanes,
        ys: lane_y,
        ..
    } = ctx;
    *nodes = placed;

    // ---- channel points ----
    // Left points of channel c: right ports of column c-1 and lane ends of column c-1 (at
    // x = col_right(c-1)). Right points: left/bottom ports of column c and lane starts.
    let n_chan = n_cols + 1;
    let mut chan_pts: Vec<Vec<ChanPt>> = vec![Vec::new(); n_chan];
    for (ni, ends) in nets.iter().enumerate() {
        if ends.len() < 2 {
            continue;
        }
        for e in ends {
            let n = &nodes[e.node];
            let (p, side) = n.port(e.port);
            let (chan, y, left) = match side {
                Side::Right => (n.layer + 1, n.y + p.y, true),
                Side::Bottom => (n.layer, n.y + p.y + 1, false),
                Side::Top => (n.layer, n.y + p.y - 1, false),
                Side::Left => (n.layer, n.y + p.y, false),
            };
            chan_pts[chan].push(ChanPt { net: ni, y, left });
        }
    }
    for (li, &(ni, col)) in lanes.iter().enumerate() {
        chan_pts[col].push(ChanPt {
            net: ni,
            y: lane_y[li],
            left: false,
        });
        chan_pts[col + 1].push(ChanPt {
            net: ni,
            y: lane_y[li],
            left: true,
        });
    }

    // ---- tracks and x ----
    let routes: Vec<tracks::ChannelRoute> = chan_pts.iter().map(|p| route_channel(p)).collect();
    let chan_w: Vec<i32> = routes
        .iter()
        .map(|r| (r.tracks + 2).max(MIN_CHANNEL))
        .collect();
    let mut col_w = vec![0i32; n_cols];
    for n in nodes.iter() {
        col_w[n.layer] = col_w[n.layer].max(n.size().x);
    }
    let mut chan_x = vec![0i32; n_chan];
    let mut col_x = vec![0i32; n_cols];
    let mut x = 0;
    for c in 0..n_cols {
        chan_x[c] = x;
        x += chan_w[c];
        col_x[c] = x;
        x += col_w[c];
    }
    chan_x[n_cols] = x;
    let col_right = |c: usize| col_x[c] + col_w[c];
    for n in nodes.iter_mut() {
        n.x = col_x[n.layer];
    }

    // ---- wires ----
    let mut segs: Vec<Vec<(GridPt, GridPt)>> = vec![Vec::new(); nets.len()];
    // Stubs inside columns.
    for (ni, ends) in nets.iter().enumerate() {
        if ends.len() < 2 {
            continue;
        }
        for e in ends {
            let n = &nodes[e.node];
            let (p, side) = n.port(e.port);
            let at = gp(n.x + p.x, n.y + p.y);
            match side {
                Side::Right => {
                    let r = col_right(n.layer);
                    if at.x < r {
                        segs[ni].push((at, gp(r, at.y)));
                    }
                }
                Side::Bottom | Side::Top => {
                    let y = if side == Side::Bottom {
                        at.y + 1
                    } else {
                        at.y - 1
                    };
                    segs[ni].push((at, gp(at.x, y)));
                    segs[ni].push((gp(at.x, y), gp(n.x, y)));
                }
                Side::Left => {}
            }
        }
    }
    for (li, &(ni, col)) in lanes.iter().enumerate() {
        let y = lane_y[li];
        segs[ni].push((gp(col_x[col], y), gp(col_right(col), y)));
    }
    // Channels.
    for (c, r) in routes.iter().enumerate() {
        let left_x = chan_x[c];
        let right_x = if c < n_cols {
            col_x[c]
        } else {
            chan_x[c] + chan_w[c]
        };
        let tx = |t: i32| left_x + 1 + t;
        for p in &r.pieces {
            let (net, a, b) = match *p {
                Piece::Across { net, y } => (net, gp(left_x, y), gp(right_x, y)),
                Piece::Vertical { net, t, y0, y1 } => (net, gp(tx(t), y0), gp(tx(t), y1)),
                Piece::LeftStub { net, t, y } => (net, gp(left_x, y), gp(tx(t), y)),
                Piece::RightStub { net, t, y } => (net, gp(tx(t), y), gp(right_x, y)),
                Piece::Jog { net, t0, t1, y } => (net, gp(tx(t0), y), gp(tx(t1), y)),
            };
            segs[net].push((a, b));
        }
    }
    segs
}

/// Build a [`NetGeom`] from segments: drop degenerate ones, merge collinear overlapping or
/// touching segments (keeping the original end points as polyline vertices, so every
/// connection point is a vertex), and compute junctions.
pub fn net_geom(segs: Vec<(GridPt, GridPt)>) -> NetGeom {
    let mut h: HashMap<i32, Vec<(i32, i32)>> = HashMap::new();
    let mut v: HashMap<i32, Vec<(i32, i32)>> = HashMap::new();
    let mut ends: HashSet<GridPt> = HashSet::new();
    for (a, b) in segs {
        if a == b {
            continue;
        }
        ends.insert(a);
        ends.insert(b);
        if a.y == b.y {
            h.entry(a.y).or_default().push((a.x.min(b.x), a.x.max(b.x)));
        } else {
            debug_assert_eq!(a.x, b.x, "non-orthogonal segment");
            v.entry(a.x).or_default().push((a.y.min(b.y), a.y.max(b.y)));
        }
    }
    let merge = |m: HashMap<i32, Vec<(i32, i32)>>| -> Vec<(i32, i32, i32)> {
        let mut out = Vec::new();
        for (k, mut iv) in m {
            iv.sort_unstable();
            let mut cur = iv[0];
            for &(a, b) in &iv[1..] {
                if a <= cur.1 {
                    cur.1 = cur.1.max(b);
                } else {
                    out.push((k, cur.0, cur.1));
                    cur = (a, b);
                }
            }
            out.push((k, cur.0, cur.1));
        }
        out.sort_unstable();
        out
    };
    // Vertices of a merged line: its ends plus every original end point (of any segment of the
    // net) lying on it, i.e. every place another wire connects.
    let line = |p0: GridPt, p1: GridPt, horizontal: bool| -> Vec<GridPt> {
        let mut pts: Vec<GridPt> = ends
            .iter()
            .copied()
            .filter(|&p| {
                if horizontal {
                    p.y == p0.y && p.x > p0.x && p.x < p1.x
                } else {
                    p.x == p0.x && p.y > p0.y && p.y < p1.y
                }
            })
            .collect();
        pts.push(p0);
        pts.push(p1);
        pts.sort_unstable_by_key(|p| if horizontal { p.x } else { p.y });
        pts
    };
    let mut polylines: Vec<Vec<GridPt>> = Vec::new();
    polylines.extend(
        merge(h)
            .into_iter()
            .map(|(y, a, b)| line(gp(a, y), gp(b, y), true)),
    );
    polylines.extend(
        merge(v)
            .into_iter()
            .map(|(x, a, b)| line(gp(x, a), gp(x, b), false)),
    );
    let mut g = NetGeom {
        polylines,
        junctions: Vec::new(),
    };
    g.junctions = junctions(&g);
    g
}

/// Junction points of a net's wires: points where three or more wire directions meet (an end
/// on another segment's interior counts twice for that segment).
pub fn junctions(g: &NetGeom) -> Vec<GridPt> {
    let segs: Vec<(GridPt, GridPt)> = g.segments().collect();
    let mut deg: HashMap<GridPt, u32> = HashMap::new();
    for &(a, b) in &segs {
        *deg.entry(a).or_default() += 1;
        *deg.entry(b).or_default() += 1;
    }
    let mut out: Vec<GridPt> = Vec::new();
    for (&p, &d) in &deg {
        let interior = segs.iter().filter(|&&(a, b)| on_interior(p, a, b)).count() as u32;
        if d + 2 * interior >= 3 {
            out.push(p);
        }
    }
    out.sort_unstable();
    out
}

pub(crate) fn on_interior(p: GridPt, a: GridPt, b: GridPt) -> bool {
    if a.y == b.y && p.y == a.y {
        p.x > a.x.min(b.x) && p.x < a.x.max(b.x)
    } else if a.x == b.x && p.x == a.x {
        p.y > a.y.min(b.y) && p.y < a.y.max(b.y)
    } else {
        false
    }
}

pub(crate) fn on_segment(p: GridPt, a: GridPt, b: GridPt) -> bool {
    p == a || p == b || on_interior(p, a, b)
}

/// Order-preserving placement: minimize Σ (y_i − d_i)² over items with a desired position `d_i`
/// subject to `y_i ≥ y_{i−1} + gap_i`. With `z_i = y_i − off_i` (`off` = prefix sums of the gaps)
/// this is isotonic regression of `d_i − off_i`, solved by pool-adjacent-violators. Items without
/// a desired position follow their predecessor tightly (leading ones: their successor); with no
/// desired position at all, `current` is kept (made feasible).
fn pav_place(desired: &[Option<f64>], gap: &[i32], current: &[f64]) -> Vec<i32> {
    let n = desired.len();
    let mut off = vec![0i64; n];
    for i in 1..n {
        off[i] = off[i - 1] + i64::from(gap[i]);
    }
    // Blocks of weighted items: (mean target, weight, first item, last item).
    let mut blocks: Vec<(f64, f64, usize, usize)> = Vec::new();
    for i in 0..n {
        let Some(d) = desired[i] else { continue };
        let mut b = (d - off[i] as f64, 1.0, i, i);
        while let Some(&p) = blocks.last() {
            if p.0 <= b.0 {
                break;
            }
            blocks.pop();
            let w = p.1 + b.1;
            b = ((p.0 * p.1 + b.0 * b.1) / w, w, p.2, b.3);
        }
        blocks.push(b);
    }
    let mut z: Vec<Option<f64>> = vec![None; n];
    for &(v, _, a, b) in &blocks {
        for zi in &mut z[a..=b] {
            *zi = Some(v);
        }
    }
    // Fill unweighted items: previous value, else the first value after.
    let first = z.iter().flatten().next().copied();
    let mut out = vec![0i32; n];
    let mut prev: Option<f64> = None;
    for i in 0..n {
        let zi = match (z[i], desired[i]) {
            (Some(v), Some(_)) => v,
            _ => match (prev, first) {
                (Some(p), _) => p,
                (None, Some(f)) => f,
                (None, None) => current[i] - off[i] as f64,
            },
        };
        let zi = prev.map_or(zi, |p| zi.max(p));
        prev = Some(zi);
        out[i] = (zi.round() as i64 + off[i]) as i32;
    }
    out
}

/// Remove DFS back edges from `succ` (iterative DFS from nodes in index order, sources first).
fn break_cycles(succ: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = succ.len();
    let mut indeg = vec![0usize; n];
    for ss in succ {
        for &b in ss {
            indeg[b] += 1;
        }
    }
    let mut state = vec![0u8; n]; // 0 new, 1 on stack, 2 done
    let mut dag: Vec<Vec<usize>> = vec![Vec::new(); n];
    let roots = (0..n).filter(|&i| indeg[i] == 0).chain(0..n);
    for r in roots {
        if state[r] != 0 {
            continue;
        }
        let mut stack: Vec<(usize, usize)> = vec![(r, 0)];
        state[r] = 1;
        while let Some(&mut (v, ref mut i)) = stack.last_mut() {
            if *i < succ[v].len() {
                let w = succ[v][*i];
                *i += 1;
                match state[w] {
                    0 => {
                        dag[v].push(w);
                        state[w] = 1;
                        stack.push((w, 0));
                    }
                    1 => {} // back edge: dropped
                    _ => dag[v].push(w),
                }
            } else {
                state[v] = 2;
                stack.pop();
            }
        }
    }
    dag
}

/// Kahn topological order of a DAG.
fn topo_order(dag: &[Vec<usize>], pred: &[Vec<usize>]) -> Vec<usize> {
    let n = dag.len();
    let mut indeg: Vec<usize> = pred.iter().map(Vec::len).collect();
    let mut q: std::collections::VecDeque<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    let mut out = Vec::with_capacity(n);
    while let Some(v) = q.pop_front() {
        out.push(v);
        for &w in &dag[v] {
            indeg[w] -= 1;
            if indeg[w] == 0 {
                q.push_back(w);
            }
        }
    }
    debug_assert_eq!(out.len(), n, "DAG");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples;
    use std::time::Instant;

    fn check_all(doc: &Document) {
        let mut errs: Vec<String> = Vec::new();
        for id in crate::doc::reachable_defs(&doc.design) {
            errs.extend(check(doc, &id));
        }
        assert!(
            errs.is_empty(),
            "{} problems, first: {:#?}",
            errs.len(),
            &errs[..errs.len().min(20)]
        );
    }

    #[test]
    fn counter_layout_is_valid() {
        let mut doc = Document::new(examples::counter());
        let ids = doc.layout_missing();
        assert!(!ids.is_empty());
        check_all(&doc);
    }

    #[test]
    fn panda8_layout_is_valid_and_fast() {
        let mut doc = Document::new(examples::panda8());
        let t0 = Instant::now();
        let ids = doc.layout_missing();
        let dt = t0.elapsed();
        eprintln!("Panda-8: {} defs laid out in {dt:?}", ids.len());
        assert!(dt.as_millis() < 1000, "layout took {dt:?}");
        check_all(&doc);
    }

    #[test]
    fn flat_panda8_layout_is_valid() {
        let design = panda_cpu8::panda8(
            &panda_cpu8::programs::BENCH_LOOP.image(),
            panda_cpu8::Variant::Flat,
        );
        let mut doc = Document::new(design);
        let t0 = Instant::now();
        doc.layout_missing();
        eprintln!("flat Panda-8 laid out in {:?}", t0.elapsed());
        check_all(&doc);
        // Prefix groups become disjoint blocks, nested ones inside their parents, each with a
        // name label.
        let top = doc.design.top.clone();
        let ed = &doc.editors[&top];
        let placed = doc.placed(&top);
        let group_box = |prefix: &str| {
            placed
                .iter()
                .filter(|p| matches!(&p.item, ItemRef::Instance(id) if id.starts_with(prefix)))
                .map(|p| p.rect())
                .reduce(|a, b| a.union(&b))
                .expect("group members")
        };
        let level1 = ["cpu.alu.", "cpu.ctl.", "cpu.rf.", "cpu.pc.", "cpu.ir."];
        for (i, a) in level1.iter().enumerate() {
            assert!(group_box("cpu.").contains(group_box(a).min), "{a}");
            for b in &level1[i + 1..] {
                assert!(!group_box(a).overlaps(&group_box(b)), "{a} overlaps {b}");
            }
        }
        assert!(!group_box("cpu.").overlaps(&group_box("rom.")));
        for name in ["cpu", "rom", "cpu.alu", "cpu.rf"] {
            assert!(ed.labels.iter().any(|l| l.text == name), "label {name}");
        }
    }

    #[test]
    fn layout_is_deterministic() {
        let design = panda_cpu8::panda8(
            &panda_cpu8::programs::BENCH_LOOP.image(),
            panda_cpu8::Variant::Flat,
        );
        let a = layout_defs(&design, &HashMap::new(), std::slice::from_ref(&design.top));
        let b = layout_defs(&design, &HashMap::new(), std::slice::from_ref(&design.top));
        assert!(a == b);
    }

    #[test]
    fn labels_shorten_from_the_left() {
        assert_eq!(fit_label("cpu.alu", 20), "cpu.alu");
        assert_eq!(fit_label("cpu.rf.w0.ff1.master", 4), "…master");
        assert_eq!(fit_label("cpu.rf.w0", 3), "…rf.w0");
        assert_eq!(fit_label("cpu.rf.w0", 1), "w0");
    }

    #[test]
    fn pav_respects_order_and_gaps() {
        let d = [Some(10.0), Some(0.0), None, Some(30.0)];
        let gaps = [0, 4, 2, 2];
        let y = pav_place(&d, &gaps, &[0.0; 4]);
        for i in 1..4 {
            assert!(y[i] >= y[i - 1] + gaps[i], "{y:?}");
        }
        // The first two conflict and meet in the middle; the last one is free.
        assert_eq!(y[3], 30);
        assert_eq!(y[0], 3);
    }
}
