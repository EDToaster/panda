//! Automatic layered layout and orthogonal routing for defs without geometry.
//!
//! A Sugiyama-style pipeline, sized for generated designs (Panda-8's 256-input ROM OR, decoder
//! fan-outs, latch feedback):
//!
//! 1. **Graph.** Nodes are the def's pin markers and instances. For every net, driver ends
//!    (`Out` ports, the def's `In` pins; for undriven nets the `InOut` ends of splitters fed from
//!    elsewhere) get an edge to every other end.
//! 2. **Layers.** Cycles (feedback) are broken by DFS back edges; longest-path layering on the
//!    rest. `In` pins form column 0, `Out`/`InOut` pins the last column, sources without inputs
//!    (constants, clocks) are pulled right next to their first consumer.
//! 3. **Lanes.** A net whose ends lie in channels `c0..=c1` (the gaps left of/right of the
//!    columns) gets one *lane* (a horizontal slot) in every column it crosses, so wires never
//!    pass through symbols, also for feedback going right-to-left.
//! 4. **Order.** Barycenter sweeps over nodes and lanes, per net centroid.
//! 5. **Y.** Order-preserving 1-D placement that minimizes the squared distance of every item to
//!    its desired position (aligned with the net it connects to on the left, else on the right),
//!    solved exactly by pool-adjacent-violators; three alternating passes.
//! 6. **Channels.** Per channel, every net with ends at different heights gets a vertical track;
//!    tracks are assigned left-edge first under the constraints that keep stubs of different nets
//!    at the same height from overlapping. Column x positions follow from channel widths.
//!
//! The result is an [`EditorData`] with `auto = true`. [`check`] verifies the invariants (no
//! overlapping symbols, wires outside symbols, every net's ends connected by its wires); the
//! tests run it over every def of Panda-8.

use std::collections::{HashMap, HashSet};

use panda_netlist::{ComponentDef, DefId, Design, Dir, Endpoint, InstKind};

use crate::doc::{Document, EditorData, InstGeom, ItemRef, NetGeom, Placed};
use crate::geom::{
    GridPt, GridRect, Orient, Side, Symbol, SymbolKind, SymbolLayout, atom_symbol, gp,
    package_symbol, pin_symbol,
};

/// Vertical gap between two symbols in a column.
const NODE_GAP: i32 = 2;
/// Vertical gap next to a lane.
const LANE_GAP: i32 = 1;
/// Minimum channel width.
const MIN_CHANNEL: i32 = 3;
/// Barycenter iterations.
const ORDER_SWEEPS: usize = 8;

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
    let chunks: Vec<&[DefId]> = ids.chunks(ids.len().div_ceil(threads).max(1)).collect();
    std::thread::scope(|s| {
        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                s.spawn(move || {
                    chunk
                        .iter()
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

/// One node of the layout graph.
struct Node {
    item: ItemRef,
    sym: Symbol,
    orient: Orient,
    layer: usize,
    /// Top y.
    y: i32,
    /// Extra height below the symbol (bottom ports route one unit down).
    extra: i32,
}

impl Node {
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
}

/// End of a net at a node port.
#[derive(Clone, Copy)]
struct End {
    node: usize,
    port: usize,
}

/// An item of a column: a node or a lane of a net.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Item {
    Node(usize),
    Lane(usize),
}

/// Lay out one def. `symbols` gives package symbol layouts by def id.
pub fn layout_def<'a>(
    design: &'a Design,
    def: &'a ComponentDef,
    symbols: impl Fn(&str) -> Option<&'a SymbolLayout>,
) -> EditorData {
    // ---- nodes ----
    let mut nodes: Vec<Node> = Vec::new();
    let mut pin_node: HashMap<&str, usize> = HashMap::new();
    let mut inst_node: HashMap<&str, usize> = HashMap::new();
    for p in &def.pins {
        pin_node.insert(&p.name, nodes.len());
        nodes.push(Node {
            item: ItemRef::Pin(p.name.clone()),
            sym: pin_symbol(p),
            orient: Orient::default(),
            layer: 0,
            y: 0,
            extra: 0,
        });
    }
    for inst in &def.instances {
        let sym = match &inst.kind {
            InstKind::Atom(a) => atom_symbol(a),
            InstKind::Def(id) => match design.def(id) {
                Some(d) => package_symbol(d, symbols(id)),
                None => continue,
            },
        };
        inst_node.insert(&inst.id, nodes.len());
        nodes.push(Node {
            item: ItemRef::Instance(inst.id.clone()),
            sym,
            orient: Orient::default(),
            layer: 0,
            y: 0,
            extra: 0,
        });
    }
    let n_nodes = nodes.len();

    // ---- net ends ----
    let nets: Vec<Vec<End>> = def
        .nets
        .iter()
        .map(|net| {
            net.ends
                .iter()
                .filter_map(|e| {
                    let (node, port) = match e {
                        Endpoint::Pin(name) => (*pin_node.get(name.as_str())?, name.as_str()),
                        Endpoint::Port { inst, port } => {
                            (*inst_node.get(inst.as_str())?, port.as_str())
                        }
                    };
                    let port = nodes[node].sym.ports.iter().position(|p| p.name == port)?;
                    Some(End { node, port })
                })
                .collect()
        })
        .collect();

    // ---- roles: drivers of each net ----
    let role = |e: &End| -> Dir {
        let p = &nodes[e.node].sym.ports[e.port];
        match (&nodes[e.node].item, p.dir) {
            // The def's own pins drive the inside (In) or are driven (Out).
            (ItemRef::Pin(_), Dir::In) => Dir::Out,
            (ItemRef::Pin(_), Dir::Out) => Dir::In,
            (_, d) => d,
        }
    };
    let has_real_driver: Vec<bool> = nets
        .iter()
        .map(|ends| ends.iter().any(|e| role(e) == Dir::Out))
        .collect();
    // Splitters fed through some port from a really driven net.
    let mut fed = vec![false; n_nodes];
    for (ni, ends) in nets.iter().enumerate() {
        if has_real_driver[ni] {
            for e in ends {
                if role(e) == Dir::InOut {
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
                .filter(|&i| role(&ends[i]) == Dir::Out)
                .collect();
            if !real.is_empty() || has_real_driver[ni] {
                return real;
            }
            let fed_io: Vec<usize> = (0..ends.len())
                .filter(|&i| role(&ends[i]) == Dir::InOut && fed[ends[i].node])
                .collect();
            if !fed_io.is_empty() {
                return fed_io;
            }
            (0..ends.len())
                .filter(|&i| role(&ends[i]) == Dir::InOut)
                .take(1)
                .collect()
        })
        .collect();

    // A splitter whose `whole` drives its net is a joiner: mirror it (whole on the right).
    for (ni, ends) in nets.iter().enumerate() {
        for &d in &drivers[ni] {
            let e = ends[d];
            if matches!(nodes[e.node].sym.kind, SymbolKind::Splitter { .. })
                && nodes[e.node].sym.ports[e.port].name == "whole"
            {
                nodes[e.node].orient.flip = true;
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
    let is_in_pin = |i: usize| {
        matches!(&nodes[i].item, ItemRef::Pin(_)) && nodes[i].sym.ports[0].dir == Dir::In
    };
    let is_out_pin = |i: usize| {
        matches!(&nodes[i].item, ItemRef::Pin(_)) && nodes[i].sym.ports[0].dir != Dir::In
    };
    let mut layer = vec![1usize; n_nodes];
    for &v in &topo {
        if is_in_pin(v) {
            layer[v] = 0;
            continue;
        }
        if is_out_pin(v) {
            continue;
        }
        layer[v] = pred[v]
            .iter()
            .filter(|&&u| !is_out_pin(u))
            .map(|&u| layer[u] + 1)
            .max()
            .unwrap_or(1)
            .max(1);
    }
    // Pull sources (no predecessors) right next to their first consumer.
    for &v in topo.iter().rev() {
        if pred[v].is_empty() && !is_in_pin(v) && !is_out_pin(v) {
            let m = dag[v]
                .iter()
                .filter(|&&s| !is_out_pin(s))
                .map(|&s| layer[s])
                .min();
            if let Some(m) = m {
                layer[v] = m.saturating_sub(1).max(1);
            }
        }
    }
    let last_inner = (0..n_nodes)
        .filter(|&i| !is_out_pin(i))
        .map(|i| layer[i])
        .max()
        .unwrap_or(0);
    for (i, l) in layer.iter_mut().enumerate() {
        if is_out_pin(i) {
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
    let mut lane_of: HashMap<(usize, usize), usize> = HashMap::new();
    for (ni, ends) in nets.iter().enumerate() {
        if ends.len() < 2 {
            continue;
        }
        let cs = ends.iter().map(|e| end_channel(e, &nodes));
        let (c0, c1) = cs.fold((usize::MAX, 0), |(a, b), c| (a.min(c), b.max(c)));
        for col in c0..c1 {
            lane_of.insert((ni, col), lanes.len());
            lanes.push((ni, col));
        }
    }

    // ---- columns and initial order ----
    let mut cols: Vec<Vec<Item>> = vec![Vec::new(); n_cols];
    for (i, n) in nodes.iter().enumerate() {
        cols[n.layer].push(Item::Node(i));
    }
    for (li, &(_, col)) in lanes.iter().enumerate() {
        cols[col].push(Item::Lane(li));
    }
    // Nets of each node (for barycenters / anchors), with the port.
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
        nodes,
        lanes,
        nets: &nets,
        node_nets,
        cols,
        ys: Vec::new(),
    };
    ctx.pack_initial();
    for it in 0..ORDER_SWEEPS {
        if it % 2 == 0 {
            for c in 1..n_cols {
                ctx.reorder(c, c - 1);
            }
        } else {
            for c in (0..n_cols.saturating_sub(1)).rev() {
                ctx.reorder(c, c + 1);
            }
        }
    }
    // ---- y placement ----
    for pass in 0..3 {
        if pass % 2 == 0 {
            for c in 0..n_cols {
                ctx.place_column(c, true);
            }
        } else {
            for c in (0..n_cols).rev() {
                ctx.place_column(c, false);
            }
        }
    }
    let Ctx {
        nodes,
        lanes,
        cols: _,
        ys: lane_y,
        ..
    } = ctx;
    // Normalize so the topmost item is at y = 0.
    let min_y = nodes
        .iter()
        .map(|n| n.y)
        .chain(lane_y.iter().copied())
        .min()
        .unwrap_or(0);
    let nodes: Vec<Node> = nodes
        .into_iter()
        .map(|mut n| {
            n.y -= min_y;
            n
        })
        .collect();
    let lane_y: Vec<i32> = lane_y.iter().map(|y| y - min_y).collect();

    // ---- channel points ----
    // Left points of channel c: right ports of column c-1 and lane ends of column c-1 (at
    // x = col_right(c-1)). Right points: left/bottom ports of column c and lane starts.
    #[derive(Clone, Copy)]
    struct Pt {
        net: usize,
        y: i32,
        left: bool,
    }
    let n_chan = n_cols + 1;
    let mut chan_pts: Vec<Vec<Pt>> = vec![Vec::new(); n_chan];
    for (ni, ends) in nets.iter().enumerate() {
        if ends.len() < 2 {
            continue;
        }
        for e in ends {
            let n = &nodes[e.node];
            let (p, side) = n.port(e.port);
            match side {
                Side::Right => chan_pts[n.layer + 1].push(Pt {
                    net: ni,
                    y: n.y + p.y,
                    left: true,
                }),
                Side::Bottom | Side::Top => {
                    let y = if side == Side::Bottom {
                        n.y + p.y + 1
                    } else {
                        n.y + p.y - 1
                    };
                    chan_pts[n.layer].push(Pt {
                        net: ni,
                        y,
                        left: false,
                    });
                }
                Side::Left => chan_pts[n.layer].push(Pt {
                    net: ni,
                    y: n.y + p.y,
                    left: false,
                }),
            }
        }
    }
    for (li, &(ni, col)) in lanes.iter().enumerate() {
        chan_pts[col].push(Pt {
            net: ni,
            y: lane_y[li],
            left: false,
        });
        chan_pts[col + 1].push(Pt {
            net: ni,
            y: lane_y[li],
            left: true,
        });
    }

    // ---- tracks ----
    let mut tracks: Vec<HashMap<usize, i32>> = Vec::with_capacity(n_chan);
    let mut chan_w = vec![MIN_CHANNEL; n_chan];
    for (c, pts) in chan_pts.iter().enumerate() {
        let t = assign_tracks(pts.iter().map(|p| (p.net, p.y, p.left)));
        let n_tracks = t.values().copied().max().map_or(0, |m| m + 1);
        chan_w[c] = (n_tracks + 2).max(MIN_CHANNEL);
        tracks.push(t);
    }
    let col_w: Vec<i32> = (0..n_cols)
        .map(|c| {
            nodes
                .iter()
                .filter(|n| n.layer == c)
                .map(|n| n.size().x)
                .max()
                .unwrap_or(0)
        })
        .collect();
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

    // ---- emit placements ----
    let mut out = EditorData {
        version: crate::doc::EDITOR_VERSION,
        auto: true,
        ..EditorData::default()
    };
    for n in &nodes {
        let g = InstGeom {
            pos: gp(col_x[n.layer], n.y),
            orient: n.orient,
        };
        match &n.item {
            ItemRef::Pin(p) => out.pins.insert(p.clone(), g),
            ItemRef::Instance(i) => out.instances.insert(i.clone(), g),
        };
    }

    // ---- emit wires ----
    let mut segs: Vec<Vec<(GridPt, GridPt)>> = vec![Vec::new(); nets.len()];
    // Stubs inside columns.
    for (ni, ends) in nets.iter().enumerate() {
        if ends.len() < 2 {
            continue;
        }
        for e in ends {
            let n = &nodes[e.node];
            let (p, side) = n.port(e.port);
            let at = gp(col_x[n.layer] + p.x, n.y + p.y);
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
                    segs[ni].push((gp(at.x, y), gp(col_x[n.layer], y)));
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
    for (c, pts) in chan_pts.iter().enumerate() {
        let left_x = chan_x[c];
        let right_x = if c < n_cols {
            col_x[c]
        } else {
            chan_x[c] + chan_w[c]
        };
        let mut by_net: HashMap<usize, Vec<Pt>> = HashMap::new();
        for p in pts {
            by_net.entry(p.net).or_default().push(*p);
        }
        for (ni, ps) in by_net {
            match tracks[c].get(&ni) {
                None => {
                    // All at one height: one straight segment across.
                    let y = ps[0].y;
                    let has_l = ps.iter().any(|p| p.left);
                    let has_r = ps.iter().any(|p| !p.left);
                    if has_l && has_r {
                        segs[ni].push((gp(left_x, y), gp(right_x, y)));
                    }
                }
                Some(&t) => {
                    let tx = left_x + 1 + t;
                    let y0 = ps.iter().map(|p| p.y).min().expect("points");
                    let y1 = ps.iter().map(|p| p.y).max().expect("points");
                    segs[ni].push((gp(tx, y0), gp(tx, y1)));
                    for p in ps {
                        if p.left {
                            segs[ni].push((gp(left_x, p.y), gp(tx, p.y)));
                        } else {
                            segs[ni].push((gp(tx, p.y), gp(right_x, p.y)));
                        }
                    }
                }
            }
        }
    }
    for (ni, s) in segs.into_iter().enumerate() {
        if s.is_empty() {
            continue;
        }
        out.wires.insert(def.nets[ni].id.clone(), net_geom(s));
    }
    out
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

fn on_interior(p: GridPt, a: GridPt, b: GridPt) -> bool {
    if a.y == b.y && p.y == a.y {
        p.x > a.x.min(b.x) && p.x < a.x.max(b.x)
    } else if a.x == b.x && p.x == a.x {
        p.y > a.y.min(b.y) && p.y < a.y.max(b.y)
    } else {
        false
    }
}

fn on_segment(p: GridPt, a: GridPt, b: GridPt) -> bool {
    p == a || p == b || on_interior(p, a, b)
}

/// Layout state for ordering and y placement.
struct Ctx<'a> {
    nodes: Vec<Node>,
    lanes: Vec<(usize, usize)>,
    nets: &'a [Vec<End>],
    node_nets: Vec<Vec<(usize, usize)>>,
    cols: Vec<Vec<Item>>,
    /// Lane y by lane index.
    ys: Vec<i32>,
}

impl Ctx<'_> {
    fn height(&self, it: Item) -> i32 {
        match it {
            Item::Node(n) => self.nodes[n].height(),
            Item::Lane(_) => 0,
        }
    }

    fn gap(&self, a: Item, b: Item) -> i32 {
        match (a, b) {
            (Item::Node(_), Item::Node(_)) => NODE_GAP,
            _ => LANE_GAP,
        }
    }

    fn top(&self, it: Item) -> i32 {
        match it {
            Item::Node(n) => self.nodes[n].y,
            Item::Lane(l) => self.ys[l],
        }
    }

    fn set_top(&mut self, it: Item, y: i32) {
        match it {
            Item::Node(n) => self.nodes[n].y = y,
            Item::Lane(l) => self.ys[l] = y,
        }
    }

    /// Stack every column top-down in its current order.
    fn pack_initial(&mut self) {
        self.ys = vec![0; self.lanes.len()];
        for c in 0..self.cols.len() {
            self.pack(c);
        }
    }

    fn pack(&mut self, c: usize) {
        let mut y = 0;
        let mut prev: Option<Item> = None;
        for i in 0..self.cols[c].len() {
            let it = self.cols[c][i];
            if let Some(p) = prev {
                y += self.height(p) + self.gap(p, it);
            }
            self.set_top(it, y);
            prev = Some(it);
        }
    }

    /// Mean y of the points of each net in channel `chan` contributed by column `col` (its
    /// right side if `col < chan`... i.e. the side facing the channel).
    fn net_anchors(&self, col: usize, facing_right: bool) -> HashMap<usize, (f64, u32)> {
        let mut acc: HashMap<usize, (f64, u32)> = HashMap::new();
        for &it in &self.cols[col] {
            match it {
                Item::Node(n) => {
                    let node = &self.nodes[n];
                    for &(ni, p) in &self.node_nets[n] {
                        let (pos, side) = node.port(p);
                        let faces_right = side == Side::Right;
                        if faces_right == facing_right {
                            let e = acc.entry(ni).or_default();
                            e.0 += f64::from(node.y + pos.y);
                            e.1 += 1;
                        }
                    }
                }
                Item::Lane(l) => {
                    let e = acc.entry(self.lanes[l].0).or_default();
                    e.0 += f64::from(self.ys[l]);
                    e.1 += 1;
                }
            }
        }
        acc
    }

    /// Desired top of item `it` from the anchors of a neighbouring column (`from_left`: the
    /// anchors are on the item's left).
    fn desired(
        &self,
        it: Item,
        anchors: &HashMap<usize, (f64, u32)>,
        from_left: bool,
    ) -> Option<f64> {
        match it {
            Item::Lane(l) => anchors
                .get(&self.lanes[l].0)
                .map(|&(s, n)| s / f64::from(n)),
            Item::Node(n) => {
                let node = &self.nodes[n];
                let (mut s, mut k) = (0.0, 0u32);
                for &(ni, p) in &self.node_nets[n] {
                    let (pos, side) = node.port(p);
                    let faces_left = side != Side::Right;
                    if faces_left == from_left
                        && let Some(&(sum, cnt)) = anchors.get(&ni)
                    {
                        s += sum / f64::from(cnt) - f64::from(pos.y);
                        k += 1;
                    }
                }
                (k > 0).then(|| s / f64::from(k))
            }
        }
    }

    /// Barycenter reorder of column `c` against column `other`.
    fn reorder(&mut self, c: usize, other: usize) {
        let from_left = other < c;
        let anchors = self.net_anchors(other, from_left);
        let items = self.cols[c].clone();
        let mut keyed: Vec<(f64, usize, Item)> = items
            .iter()
            .enumerate()
            .map(|(i, &it)| {
                let center = f64::from(self.top(it)) + f64::from(self.height(it)) / 2.0;
                let k = self
                    .desired(it, &anchors, from_left)
                    .map_or(center, |d| d + f64::from(self.height(it)) / 2.0);
                (k, i, it)
            })
            .collect();
        keyed.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        self.cols[c] = keyed.into_iter().map(|k| k.2).collect();
        self.pack(c);
        // Unpacked spacing would bias the next column; keep the packed positions centered on
        // the previous ones.
        let _ = self.nets;
    }

    /// Place column `c` by order-preserving least squares toward the desired positions
    /// (anchors on the left if `left_pass`, falling back to the other side).
    fn place_column(&mut self, c: usize, left_pass: bool) {
        let n_cols = self.cols.len();
        let empty = HashMap::new();
        let left = if c > 0 {
            self.net_anchors(c - 1, true)
        } else {
            empty.clone()
        };
        let right = if c + 1 < n_cols {
            self.net_anchors(c + 1, false)
        } else {
            empty
        };
        let items = self.cols[c].clone();
        let desired: Vec<Option<f64>> = items
            .iter()
            .map(|&it| {
                let (a, b) = if left_pass {
                    (&left, &right)
                } else {
                    (&right, &left)
                };
                let (fa, fb) = if left_pass {
                    (true, false)
                } else {
                    (false, true)
                };
                self.desired(it, a, fa).or_else(|| self.desired(it, b, fb))
            })
            .collect();
        // Offsets of each item if packed tightly.
        let mut gaps = Vec::with_capacity(items.len());
        for i in 0..items.len() {
            gaps.push(if i == 0 {
                0
            } else {
                self.height(items[i - 1]) + self.gap(items[i - 1], items[i])
            });
        }
        let tops = pav_place(
            &desired,
            &gaps,
            &items
                .iter()
                .map(|&it| f64::from(self.top(it)))
                .collect::<Vec<_>>(),
        );
        for (it, y) in items.into_iter().zip(tops) {
            self.set_top(it, y);
        }
    }
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

/// Assign tracks in one channel. Input: `(net, y, left)` points (`left`: the point is on the
/// channel's left side). Nets whose points all share one height need no track. Returns the
/// track index per net.
fn assign_tracks(pts: impl Iterator<Item = (usize, i32, bool)>) -> HashMap<usize, i32> {
    let mut span: HashMap<usize, (i32, i32)> = HashMap::new();
    let mut left_at: HashMap<i32, Vec<usize>> = HashMap::new();
    let mut right_at: HashMap<i32, Vec<usize>> = HashMap::new();
    for (net, y, left) in pts {
        let e = span.entry(net).or_insert((y, y));
        e.0 = e.0.min(y);
        e.1 = e.1.max(y);
        if left {
            left_at.entry(y).or_default().push(net)
        } else {
            right_at.entry(y).or_default().push(net)
        }
    }
    let mut need: Vec<usize> = span
        .iter()
        .filter(|(_, s)| s.0 != s.1)
        .map(|(&n, _)| n)
        .collect();
    need.sort_by_key(|n| (span[n].0, span[n].1, *n));
    let needs: HashSet<usize> = need.iter().copied().collect();
    // Constraint a → b: a's track left of b's (a has a left point and b a right point at the
    // same height). Straight nets (no track) do not constrain.
    let mut after: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut indeg: HashMap<usize, usize> = need.iter().map(|&n| (n, 0)).collect();
    for (y, ls) in &left_at {
        if let Some(rs) = right_at.get(y) {
            for &a in ls {
                for &b in rs {
                    if a != b && needs.contains(&a) && needs.contains(&b) {
                        after.entry(a).or_default().push(b);
                        *indeg.get_mut(&b).expect("needs") += 1;
                    }
                }
            }
        }
    }
    let mut track: HashMap<usize, i32> = HashMap::new();
    let mut used: Vec<Vec<(i32, i32)>> = Vec::new();
    let mut done: HashSet<usize> = HashSet::new();
    let mut min_track: HashMap<usize, i32> = HashMap::new();
    while done.len() < need.len() {
        // Next ready net in span order (or, on a cycle, the first pending one).
        let pick = need
            .iter()
            .copied()
            .find(|n| !done.contains(n) && indeg[n] == 0)
            .or_else(|| need.iter().copied().find(|n| !done.contains(n)))
            .expect("pending");
        let (a, b) = span[&pick];
        let mut t = *min_track.get(&pick).unwrap_or(&0);
        loop {
            if used.len() <= t as usize {
                used.push(Vec::new());
            }
            if used[t as usize].iter().all(|&(c, d)| b < c || a > d) {
                break;
            }
            t += 1;
        }
        used[t as usize].push((a, b));
        track.insert(pick, t);
        done.insert(pick);
        for &s in after.get(&pick).map_or(&[][..], Vec::as_slice) {
            if let Some(d) = indeg.get_mut(&s) {
                *d = d.saturating_sub(1);
            }
            let m = min_track.entry(s).or_insert(0);
            *m = (*m).max(t + 1);
        }
    }
    track
}

/// Invariant violations of a def's geometry (empty = fine): overlapping symbols, wires passing
/// through a symbol's interior, a net end not on its wires, a net whose wires are disconnected.
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
    let by_item: HashMap<&ItemRef, &Placed> = placed.iter().map(|p| (&p.item, p)).collect();
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
        // Connectivity: union-find over segments that touch.
        let mut parent: Vec<usize> = (0..segs.len()).collect();
        fn find(p: &mut [usize], i: usize) -> usize {
            let mut i = i;
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
                if on_segment(c, a, b)
                    || on_segment(d, a, b)
                    || on_segment(a, c, d)
                    || on_segment(b, c, d)
                {
                    let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                    parent[ri] = rj;
                }
            }
        }
        let roots: HashSet<usize> = (0..segs.len()).map(|i| find(&mut parent, i)).collect();
        if roots.len() > 1 {
            errs.push(format!(
                "{def_id}: net {} wires form {} pieces",
                net.id,
                roots.len()
            ));
        }
        // Wires outside symbol interiors.
        for &(a, b) in &segs {
            for (r, p) in &rects {
                let lo = gp(a.x.min(b.x), a.y.min(b.y));
                let hi = gp(a.x.max(b.x), a.y.max(b.y));
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
    errs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples;
    use std::time::Instant;

    fn check_all(doc: &Document) {
        let mut errs = Vec::new();
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

    #[test]
    fn tracks_avoid_stub_overlaps() {
        // Net 0: left at y=0, right at y=4. Net 1: left at y=4, right at y=8.
        // Net 0's right stub at y=4 must start right of net 1's track.
        let t =
            assign_tracks([(0, 0, true), (0, 4, false), (1, 4, true), (1, 8, false)].into_iter());
        assert!(t[&1] < t[&0], "{t:?}");
    }
}
