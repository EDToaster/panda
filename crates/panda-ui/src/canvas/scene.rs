//! The drawable scene of one def: placed symbols, wires and text, built once per document
//! revision and reused every frame (hit-testing for tools lives here too).

use std::collections::HashMap;

use egui::{Align2, Pos2, Rect, pos2};
use panda_netlist::Endpoint;

use crate::doc::{Document, ItemRef, Placed};
use crate::geom::{GridPt, GridRect, gp};

/// Text drawn with a symbol, in local grid coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct SymText {
    /// Anchor (local, unrotated).
    pub pos: [f32; 2],
    /// Text.
    pub text: String,
    /// Height in grid units.
    pub size: f32,
    /// Alignment at the anchor.
    pub align: Align2,
    /// Dimmed (pin labels).
    pub dim: bool,
    /// Drawn on a chip body (light text) rather than on the background / a light fill.
    pub on_chip: bool,
    /// Runs along the local y axis, reading upward (else along x, reading rightward).
    pub vertical: bool,
}

/// A rounded wire bend.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Corner {
    /// The bend point.
    pub at: GridPt,
    /// Unit directions from the bend along its two segments.
    pub dirs: [[f32; 2]; 2],
    /// Largest usable radius (half the shorter segment), in grid units.
    pub max_r: f32,
}

/// Corners, per-segment corner ends and caps of a net's segments (see [`SceneNet`]).
pub fn wire_topology(
    segs: &[(GridPt, GridPt)],
    junctions: &[GridPt],
) -> (Vec<[Option<u32>; 2]>, Vec<Corner>, Vec<GridPt>) {
    // Point -> incident segment ends (segment index, which end, direction away from the point).
    let mut ends: HashMap<GridPt, Vec<(usize, usize, [i32; 2])>> = HashMap::new();
    for (i, &(a, b)) in segs.iter().enumerate() {
        if a == b {
            continue;
        }
        let d = [(b.x - a.x).signum(), (b.y - a.y).signum()];
        ends.entry(a).or_default().push((i, 0, d));
        ends.entry(b).or_default().push((i, 1, [-d[0], -d[1]]));
    }
    let len = |i: usize| {
        let (a, b) = segs[i];
        ((b.x - a.x).abs() + (b.y - a.y).abs()) as f32
    };
    let mut seg_corners = vec![[None; 2]; segs.len()];
    let mut corners = Vec::new();
    let mut caps = Vec::new();
    let mut pts: Vec<&GridPt> = ends.keys().collect();
    pts.sort_unstable_by_key(|p| (p.y, p.x));
    for p in pts {
        let e = &ends[p];
        match e.as_slice() {
            [(i, ei, d0), (j, ej, d1)]
                if d0[0] * d1[0] + d0[1] * d1[1] == 0 && !junctions.contains(p) =>
            {
                let k = corners.len() as u32;
                corners.push(Corner {
                    at: *p,
                    dirs: [[d0[0] as f32, d0[1] as f32], [d1[0] as f32, d1[1] as f32]],
                    max_r: len(*i).min(len(*j)) / 2.0,
                });
                seg_corners[*i][*ei] = Some(k);
                seg_corners[*j][*ej] = Some(k);
            }
            [(_, _, d0), (_, _, d1)] if d0[0] == -d1[0] && d0[1] == -d1[1] => {}
            _ => caps.push(*p),
        }
    }
    (seg_corners, corners, caps)
}

/// One net's wires.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneNet {
    /// Net id.
    pub id: String,
    /// Bit width.
    pub width: u32,
    /// Axis-aligned segments.
    pub segs: Vec<(GridPt, GridPt)>,
    /// Per segment: the corner (index into `corners`) at its first / second end, if any.
    pub seg_corners: Vec<[Option<u32>; 2]>,
    /// Bends where exactly two perpendicular segments meet (drawn rounded).
    pub corners: Vec<Corner>,
    /// Free wire ends (and other non-corner, non-straight joints), drawn with round caps.
    pub caps: Vec<GridPt>,
    /// Junction dots.
    pub junctions: Vec<GridPt>,
    /// Bounding box of the wires.
    pub bounds: GridRect,
    /// Where a bus value label goes (midpoint of the longest horizontal segment).
    pub label_at: Option<GridPt>,
}

/// The scene of one def at one document revision.
#[derive(Clone, Debug)]
pub struct Scene {
    /// Def id.
    pub def: String,
    /// Document revision it was built from.
    pub revision: u64,
    /// Placed items.
    pub items: Vec<Placed>,
    /// World rectangles of `items`.
    pub rects: Vec<GridRect>,
    /// Text per item.
    pub texts: Vec<Vec<SymText>>,
    /// For pin markers: the id of the net connected to the pin (its live value lights the
    /// marker's indicator).
    pub pin_nets: Vec<Option<String>>,
    /// Nets with wires.
    pub nets: Vec<SceneNet>,
    /// Net index by id.
    pub net_index: HashMap<String, usize>,
    /// Bounding box of everything.
    pub bounds: GridRect,
    /// Free text labels (e.g. group names from auto-layout).
    pub labels: Vec<crate::doc::Label>,
    /// Occupancy pyramid for overview zoom levels (built on first use).
    overview: std::sync::OnceLock<Overview>,
}

impl Scene {
    /// Build the scene of `def` (empty if the def is unknown or has no geometry).
    pub fn build(doc: &Document, def_id: &str) -> Scene {
        let items = doc.placed(def_id);
        let rects: Vec<GridRect> = items.iter().map(Placed::rect).collect();
        let texts = items.iter().map(symbol_texts).collect();
        let mut nets = Vec::new();
        if let (Some(def), Some(ed)) = (doc.def(def_id), doc.editor(def_id)) {
            for n in &def.nets {
                let Some(g) = ed.wires.get(&n.id) else {
                    continue;
                };
                let segs: Vec<(GridPt, GridPt)> = g.segments().collect();
                if segs.is_empty() {
                    continue;
                }
                let mut b = GridRect {
                    min: segs[0].0,
                    max: segs[0].0,
                };
                for &(a, c) in &segs {
                    b = b.union(&GridRect {
                        min: gp(a.x.min(c.x), a.y.min(c.y)),
                        max: gp(a.x.max(c.x), a.y.max(c.y)),
                    });
                }
                let label_at = segs
                    .iter()
                    .filter(|(a, c)| a.y == c.y)
                    .max_by_key(|(a, c)| (a.x - c.x).abs())
                    .map(|&(a, c)| gp((a.x + c.x) / 2, a.y));
                let (seg_corners, corners, caps) = wire_topology(&segs, &g.junctions);
                nets.push(SceneNet {
                    id: n.id.clone(),
                    width: n.width,
                    segs,
                    seg_corners,
                    corners,
                    caps,
                    junctions: g.junctions.clone(),
                    bounds: b,
                    label_at,
                });
            }
        }
        let net_index = nets
            .iter()
            .enumerate()
            .map(|(i, n)| (n.id.clone(), i))
            .collect();
        let bounds = rects
            .iter()
            .chain(nets.iter().map(|n| &n.bounds))
            .copied()
            .reduce(|a, b| a.union(&b))
            .unwrap_or(GridRect {
                min: gp(0, 0),
                max: gp(40, 30),
            });
        let labels = doc
            .editor(def_id)
            .map_or_else(Vec::new, |e| e.labels.clone());
        let pin_nets = items
            .iter()
            .map(|p| {
                let ItemRef::Pin(name) = &p.item else {
                    return None;
                };
                let def = doc.def(def_id)?;
                def.nets
                    .iter()
                    .find(|n| {
                        n.ends
                            .iter()
                            .any(|e| matches!(e, Endpoint::Pin(pn) if pn == name))
                    })
                    .map(|n| n.id.clone())
            })
            .collect();
        Scene {
            def: def_id.to_string(),
            revision: doc.revision,
            items,
            rects,
            texts,
            pin_nets,
            nets,
            net_index,
            bounds,
            labels,
            overview: std::sync::OnceLock::new(),
        }
    }

    /// The occupancy pyramid (built once per scene).
    pub fn overview(&self) -> &Overview {
        self.overview.get_or_init(|| Overview::build(self))
    }

    /// World bounds as an egui rect (grid units).
    pub fn bounds_rect(&self) -> Rect {
        Rect::from_min_max(
            pos2(self.bounds.min.x as f32, self.bounds.min.y as f32),
            pos2(self.bounds.max.x as f32, self.bounds.max.y as f32),
        )
    }

    /// Topmost item whose box contains world point `w`.
    pub fn hit_item(&self, w: Pos2) -> Option<&Placed> {
        self.rects
            .iter()
            .zip(&self.items)
            .rev()
            .find_map(|(r, it)| {
                (w.x >= r.min.x as f32
                    && w.x <= r.max.x as f32
                    && w.y >= r.min.y as f32
                    && w.y <= r.max.y as f32)
                    .then_some(it)
            })
    }

    /// Net whose wire passes within `tol` grid units of world point `w`.
    pub fn hit_net(&self, w: Pos2, tol: f32) -> Option<&SceneNet> {
        self.nets.iter().find(|n| {
            n.segs.iter().any(|&(a, b)| {
                let (x0, x1) = (a.x.min(b.x) as f32 - tol, a.x.max(b.x) as f32 + tol);
                let (y0, y1) = (a.y.min(b.y) as f32 - tol, a.y.max(b.y) as f32 + tol);
                w.x >= x0 && w.x <= x1 && w.y >= y0 && w.y <= y1
            })
        })
    }

    /// Item by reference.
    pub fn item(&self, r: &ItemRef) -> Option<&Placed> {
        self.items.iter().find(|p| &p.item == r)
    }

    /// Nets connected to an endpoint (for tools): world position of an endpoint.
    pub fn endpoint_pos(&self, e: &Endpoint) -> Option<GridPt> {
        match e {
            Endpoint::Pin(n) => self.item(&ItemRef::Pin(n.clone()))?.port_pos(n),
            Endpoint::Port { inst, port } => {
                self.item(&ItemRef::Instance(inst.clone()))?.port_pos(port)
            }
        }
    }
}

/// Text shown with a symbol (local coordinates; see [`super::shapes::symbol_texts`]).
fn symbol_texts(p: &Placed) -> Vec<SymText> {
    super::shapes::symbol_texts(&p.symbol)
}

/// One level of an [`Overview`]: a bitmap of `cell`-unit squares.
#[derive(Clone, Debug)]
pub struct OverviewLevel {
    /// Cell size (grid units).
    pub cell: i32,
    /// Cells per row.
    pub w: usize,
    /// Rows.
    pub h: usize,
    /// Occupied cells, row-major.
    pub bits: Vec<bool>,
}

/// Occupancy pyramid of a scene (symbols and wires), for drawing far-out zoom levels at a cost
/// proportional to the screen instead of the design.
#[derive(Clone, Debug)]
pub struct Overview {
    /// World position of cell (0, 0).
    pub origin: GridPt,
    /// Levels, finest first; each coarser level has 4× the cell size.
    pub levels: Vec<OverviewLevel>,
}

impl Overview {
    /// Cells of the finest level are capped at about this many.
    const MAX_CELLS: f64 = 8.0e6;

    fn build(scene: &Scene) -> Overview {
        let b = scene.bounds;
        let (ww, wh) = (
            f64::from(b.max.x - b.min.x + 1),
            f64::from(b.max.y - b.min.y + 1),
        );
        let mut cell = 2i32;
        while ww / f64::from(cell) * (wh / f64::from(cell)) > Self::MAX_CELLS {
            cell *= 2;
        }
        let w = ((ww / f64::from(cell)).ceil() as usize).max(1);
        let h = ((wh / f64::from(cell)).ceil() as usize).max(1);
        let mut bits = vec![false; w * h];
        let mut mark = |r: GridRect| {
            let x0 = ((r.min.x - b.min.x) / cell).max(0) as usize;
            let y0 = ((r.min.y - b.min.y) / cell).max(0) as usize;
            let x1 = (((r.max.x - b.min.x) / cell).max(0) as usize).min(w - 1);
            let y1 = (((r.max.y - b.min.y) / cell).max(0) as usize).min(h - 1);
            for y in y0..=y1 {
                bits[y * w + x0..=y * w + x1].fill(true);
            }
        };
        for r in &scene.rects {
            mark(*r);
        }
        for n in &scene.nets {
            for &(a, c) in &n.segs {
                mark(GridRect {
                    min: gp(a.x.min(c.x), a.y.min(c.y)),
                    max: gp(a.x.max(c.x), a.y.max(c.y)),
                });
            }
        }
        let mut levels = vec![OverviewLevel { cell, w, h, bits }];
        while let Some(l) = levels.last() {
            if l.w <= 1 && l.h <= 1 {
                break;
            }
            let (nw, nh) = (l.w.div_ceil(4), l.h.div_ceil(4));
            let mut nb = vec![false; nw * nh];
            for y in 0..l.h {
                for x in 0..l.w {
                    if l.bits[y * l.w + x] {
                        nb[(y / 4) * nw + x / 4] = true;
                    }
                }
            }
            let next = OverviewLevel {
                cell: l.cell * 4,
                w: nw,
                h: nh,
                bits: nb,
            };
            levels.push(next);
        }
        Overview {
            origin: b.min,
            levels,
        }
    }

    /// The finest level whose cells are at least `min_px` points at `zoom`.
    pub fn level_for(&self, zoom: f32, min_px: f32) -> &OverviewLevel {
        self.levels
            .iter()
            .find(|l| l.cell as f32 * zoom >= min_px)
            .unwrap_or_else(|| self.levels.last().expect("at least one level"))
    }
}

#[cfg(test)]
mod topology_tests {
    use super::*;

    #[test]
    fn bends_become_corners_and_free_ends_caps() {
        // An L (corner at (4, 0)) with a straight split at (2, 0) and a T at (4, 3).
        let segs = [
            (gp(0, 0), gp(2, 0)),
            (gp(2, 0), gp(4, 0)),
            (gp(4, 0), gp(4, 3)),
            (gp(4, 3), gp(4, 6)),
            (gp(4, 3), gp(7, 3)),
        ];
        let (sc, corners, caps) = wire_topology(&segs, &[gp(4, 3)]);
        assert_eq!(corners.len(), 1);
        assert_eq!(corners[0].at, gp(4, 0));
        assert_eq!(corners[0].max_r, 1.0);
        assert_eq!(sc[1], [None, Some(0)]);
        assert_eq!(sc[2], [Some(0), None]);
        assert_eq!(caps, vec![gp(0, 0), gp(4, 3), gp(7, 3), gp(4, 6)]);
    }
}
