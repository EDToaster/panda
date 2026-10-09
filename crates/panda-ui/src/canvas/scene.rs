//! The drawable scene of one def: placed symbols, wires and text, built once per document
//! revision and reused every frame (hit-testing for tools lives here too).

use std::collections::HashMap;

use egui::{Align2, Pos2, Rect, pos2};
use panda_netlist::{Dir, Endpoint};

use crate::doc::{Document, ItemRef, Placed};
use crate::geom::{GridPt, GridRect, SymbolKind, gp};

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
    /// Nets with wires.
    pub nets: Vec<SceneNet>,
    /// Net index by id.
    pub net_index: HashMap<String, usize>,
    /// Bounding box of everything.
    pub bounds: GridRect,
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
                nets.push(SceneNet {
                    id: n.id.clone(),
                    width: n.width,
                    segs,
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
        Scene {
            def: def_id.to_string(),
            revision: doc.revision,
            items,
            rects,
            texts,
            nets,
            net_index,
            bounds,
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

/// Text shown with a symbol (local coordinates).
fn symbol_texts(p: &Placed) -> Vec<SymText> {
    let s = &p.symbol;
    let (w, h) = (s.size.x as f32, s.size.y as f32);
    let t = |pos: [f32; 2], text: &str, size: f32, align: Align2, dim: bool| SymText {
        pos,
        text: text.to_string(),
        size,
        align,
        dim,
    };
    match &s.kind {
        SymbolKind::Constant { text } => {
            vec![t([w / 2.0, 1.0], text, 1.1, Align2::CENTER_CENTER, false)]
        }
        SymbolKind::PowerOnReset => {
            vec![t([w / 2.0, 1.0], "POR", 1.1, Align2::CENTER_CENTER, false)]
        }
        SymbolKind::Pin { name, dir } => {
            let x = if *dir == Dir::In {
                (w - 1.0) / 2.0
            } else {
                (w + 1.0) / 2.0
            };
            vec![t([x, 1.0], name, 1.1, Align2::CENTER_CENTER, false)]
        }
        SymbolKind::Package { title, .. } => {
            let mut v = vec![t([w / 2.0, 1.0], title, 1.15, Align2::CENTER_CENTER, false)];
            for port in &s.ports {
                let (pos, align) = match port.side {
                    crate::geom::Side::Left => ([0.6, port.pos.y as f32], Align2::LEFT_CENTER),
                    crate::geom::Side::Right => {
                        ([w - 0.6, port.pos.y as f32], Align2::RIGHT_CENTER)
                    }
                    crate::geom::Side::Top => ([port.pos.x as f32, 0.6], Align2::CENTER_TOP),
                    crate::geom::Side::Bottom => {
                        ([port.pos.x as f32, h - 0.6], Align2::CENTER_BOTTOM)
                    }
                };
                v.push(t(pos, &port.name, 1.0, align, true));
            }
            v
        }
        SymbolKind::Splitter { parts } => parts
            .iter()
            .enumerate()
            .map(|(i, (lo, hi))| {
                let label = if lo == hi {
                    format!("{lo}")
                } else {
                    format!("{hi}:{lo}")
                };
                t(
                    [1.25, 1.0 + 2.0 * i as f32 - 0.15],
                    &label,
                    0.75,
                    Align2::LEFT_BOTTOM,
                    true,
                )
            })
            .collect(),
        _ => Vec::new(),
    }
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
