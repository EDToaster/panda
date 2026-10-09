//! Symbol glyphs and symbol text.
//!
//! * Glyphs: outlines in local grid coordinates, tessellated once per zoom into pixel-space
//!   template meshes that are copied (rotated / mirrored in 90° steps, which keeps anti-aliasing
//!   exact) for every visible instance. Package boxes show a tick at every pin.
//! * Text ([`symbol_texts`], [`place_text`]): labels are laid out in the symbol's local frame
//!   (pin names inside the box next to their pin, sized to the room they have) and follow the
//!   symbol's orientation like the glyph does, except that they are never drawn upside down or
//!   mirrored: text that would read right-to-left reads left-to-right, text that would read
//!   downward reads upward, with the alignment mirrored so a label stays attached to its pin.

use std::collections::HashMap;

use egui::epaint::{PathShape, PathStroke, TessellationOptions, Tessellator, TextShape};
use egui::{Align2, Color32, FontId, Mesh, Painter, Pos2, Shape, Stroke, pos2};
use panda_netlist::{Dir, GateOp};

use crate::geom::{GridPt, Orient, Side, Symbol, SymbolKind};

use super::scene::SymText;
use super::style::Palette;

type P = [f32; 2];

/// Glyph geometry in local grid coordinates.
#[derive(Clone, Debug, Default)]
pub struct Glyph {
    /// Filled star-shaped polygons: (outline, fan center, use the header colour).
    pub fills: Vec<(Vec<P>, P, bool)>,
    /// Stroked polylines: (points, closed, width multiplier).
    pub strokes: Vec<(Vec<P>, bool, f32)>,
    /// Discs: (center, radius) filled with the fill colour and stroked.
    pub discs: Vec<(P, f32)>,
}

/// Template cache key: everything that changes a glyph's geometry.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TemplateKey {
    /// Gate (op, inputs, reduce, height).
    Gate(GateOp, u32, bool, i32),
    /// Inverter.
    Not,
    /// Buffer.
    Buf,
    /// Tri-state buffer.
    Tri,
    /// Splitter with this many parts.
    Splitter(usize),
    /// Constant box of this width.
    Constant(i32),
    /// Clock.
    Clock,
    /// Power-on reset.
    Por,
    /// Pin marker (direction, width).
    Pin(Dir, i32),
    /// Package box: size, and per pin its side, position along the side and whether it is a
    /// bus.
    Package(i32, i32, Vec<(Side, i32, bool)>),
}

/// Template key of a symbol.
pub fn template_key(s: &Symbol) -> TemplateKey {
    match &s.kind {
        SymbolKind::Gate { op, inputs, reduce } => {
            TemplateKey::Gate(*op, *inputs, *reduce, s.size.y)
        }
        SymbolKind::Not => TemplateKey::Not,
        SymbolKind::Buf => TemplateKey::Buf,
        SymbolKind::Tri => TemplateKey::Tri,
        SymbolKind::Splitter { parts } => TemplateKey::Splitter(parts.len()),
        SymbolKind::Constant { .. } => TemplateKey::Constant(s.size.x),
        SymbolKind::Clock => TemplateKey::Clock,
        SymbolKind::PowerOnReset => TemplateKey::Por,
        SymbolKind::Pin { dir, .. } => TemplateKey::Pin(*dir, s.size.x),
        SymbolKind::Package { .. } => TemplateKey::Package(
            s.size.x,
            s.size.y,
            s.ports
                .iter()
                .map(|p| {
                    let along = match p.side {
                        Side::Left | Side::Right => p.pos.y,
                        Side::Top | Side::Bottom => p.pos.x,
                    };
                    (p.side, along, p.width > 1)
                })
                .collect(),
        ),
    }
}

fn quad_bezier(a: P, c: P, b: P, n: usize, out: &mut Vec<P>) {
    for i in 1..=n {
        let t = i as f32 / n as f32;
        let u = 1.0 - t;
        out.push([
            u * u * a[0] + 2.0 * u * t * c[0] + t * t * b[0],
            u * u * a[1] + 2.0 * u * t * c[1] + t * t * b[1],
        ]);
    }
}

fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<P> {
    vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
}

/// x of the OR back curve (quadratic from (x0, y0) to (x0, y1) bulging to x0 + bulge) at `y`.
fn or_back_x(x0: f32, bulge: f32, y0: f32, y1: f32, y: f32) -> f32 {
    let t = ((y - y0) / (y1 - y0)).clamp(0.0, 1.0);
    // Control point at 2*bulge gives a peak of `bulge` at t = 0.5.
    x0 + 2.0 * bulge * 2.0 * t * (1.0 - t)
}

/// The glyph of a symbol.
pub fn glyph(s: &Symbol) -> Glyph {
    let (w, h) = (s.size.x as f32, s.size.y as f32);
    let mut g = Glyph::default();
    match &s.kind {
        SymbolKind::Gate { op, .. } => {
            let bubble = matches!(op, GateOp::Nand | GateOp::Nor | GateOp::Xnor);
            let mid = h / 2.0;
            let bh = h.min(8.0);
            let (y0, y1) = (mid - bh / 2.0, mid + bh / 2.0);
            let bw = if bubble { 3.4 } else { 4.0 };
            let in_ys: Vec<f32> = s
                .ports
                .iter()
                .filter(|p| p.dir == Dir::In)
                .map(|p| p.pos.y as f32)
                .collect();
            let (ymin, ymax) = in_ys
                .iter()
                .fold((y0, y1), |(a, b), &y| (a.min(y), b.max(y)));
            let ors = matches!(op, GateOp::Or | GateOp::Nor | GateOp::Xor | GateOp::Xnor);
            let xors = matches!(op, GateOp::Xor | GateOp::Xnor);
            let x0 = if xors { 0.6 } else { 0.0 };
            let bulge = 0.45;
            if ors {
                let mut pts: Vec<P> = vec![[x0, y0]];
                quad_bezier([x0, y0], [x0 + bw * 0.55, y0], [bw, mid], 12, &mut pts);
                quad_bezier([bw, mid], [x0 + bw * 0.55, y1], [x0, y1], 12, &mut pts);
                quad_bezier([x0, y1], [x0 + 2.0 * bulge, mid], [x0, y0], 8, &mut pts);
                pts.pop();
                g.fills
                    .push((pts.clone(), [x0 + (bw - x0) * 0.45, mid], false));
                g.strokes.push((pts, true, 1.0));
                if xors {
                    let mut back = vec![[0.0, y1]];
                    quad_bezier([0.0, y1], [2.0 * bulge, mid], [0.0, y0], 8, &mut back);
                    g.strokes.push((back, false, 1.0));
                }
                // Input stubs from the pin to the back curve.
                for &y in &in_ys {
                    if y > y0 && y < y1 {
                        let xb = or_back_x(x0, bulge, y0, y1, y);
                        g.strokes.push((vec![[0.0, y], [xb, y]], false, 1.0));
                    }
                }
            } else {
                let xs = bw * 0.45;
                let mut pts: Vec<P> = vec![[0.0, y0], [xs, y0]];
                let (rx, ry) = (bw - xs, bh / 2.0);
                for i in 1..16 {
                    let a = -std::f32::consts::FRAC_PI_2 + std::f32::consts::PI * i as f32 / 16.0;
                    pts.push([xs + rx * a.cos(), mid + ry * a.sin()]);
                }
                pts.push([xs, y1]);
                pts.push([0.0, y1]);
                g.fills.push((pts.clone(), [bw * 0.4, mid], false));
                g.strokes.push((pts, true, 1.0));
            }
            // Extension bar for inputs beyond the body (wide gates).
            if ymin < y0 {
                g.strokes.push((vec![[0.0, ymin], [0.0, y0]], false, 1.0));
            }
            if ymax > y1 {
                g.strokes.push((vec![[0.0, y1], [0.0, ymax]], false, 1.0));
            }
            if bubble {
                g.discs.push(([bw + 0.3, mid], 0.3));
            }
        }
        SymbolKind::Not | SymbolKind::Buf => {
            let not = matches!(s.kind, SymbolKind::Not);
            let bw = if not { 3.4 } else { 4.0 };
            let pts = vec![[0.0, 0.05], [bw, 1.0], [0.0, 1.95]];
            g.fills.push((pts.clone(), [bw * 0.33, 1.0], false));
            g.strokes.push((pts, true, 1.0));
            if not {
                g.discs.push(([3.7, 1.0], 0.3));
            }
        }
        SymbolKind::Tri => {
            let pts = vec![[0.5, 0.5], [3.5, 2.0], [0.5, 3.5]];
            g.fills.push((pts.clone(), [1.5, 2.0], false));
            g.strokes.push((pts, true, 1.0));
            g.strokes.push((vec![[0.0, 2.0], [0.5, 2.0]], false, 1.0));
            g.strokes.push((vec![[3.5, 2.0], [4.0, 2.0]], false, 1.0));
            g.strokes.push((vec![[2.0, 4.0], [2.0, 2.75]], false, 1.0));
        }
        SymbolKind::Splitter { parts } => {
            g.strokes
                .push((vec![[1.0, 0.4], [1.0, h - 0.4]], false, 2.5));
            g.strokes
                .push((vec![[0.0, h / 2.0], [1.0, h / 2.0]], false, 1.6));
            for i in 0..parts.len() {
                let y = 1.0 + 2.0 * i as f32;
                g.strokes.push((vec![[1.0, y], [2.0, y]], false, 1.0));
            }
        }
        SymbolKind::Constant { .. } => {
            let r = rect(0.0, 0.2, w, 1.8);
            g.fills.push((r.clone(), [w / 2.0, 1.0], false));
            g.strokes.push((r, true, 1.0));
        }
        SymbolKind::Clock => {
            let r = rect(0.0, 0.1, w, 1.9);
            g.fills.push((r.clone(), [w / 2.0, 1.0], false));
            g.strokes.push((r, true, 1.0));
            g.strokes.push((
                vec![
                    [0.5, 1.5],
                    [1.1, 1.5],
                    [1.1, 0.5],
                    [2.0, 0.5],
                    [2.0, 1.5],
                    [2.9, 1.5],
                    [2.9, 0.5],
                    [3.5, 0.5],
                ],
                false,
                1.0,
            ));
        }
        SymbolKind::PowerOnReset => {
            let r = rect(0.0, 0.1, w, 1.9);
            g.fills.push((r.clone(), [w / 2.0, 1.0], false));
            g.strokes.push((r, true, 1.0));
        }
        SymbolKind::Pin { dir, .. } => {
            let pts = match dir {
                Dir::In => vec![
                    [0.0, 0.2],
                    [w - 1.0, 0.2],
                    [w, 1.0],
                    [w - 1.0, 1.8],
                    [0.0, 1.8],
                ],
                _ => vec![[0.0, 1.0], [1.0, 0.2], [w, 0.2], [w, 1.8], [1.0, 1.8]],
            };
            g.fills.push((pts.clone(), [w / 2.0, 1.0], true));
            g.strokes.push((pts, true, 1.0));
        }
        SymbolKind::Package { .. } => {
            let body = rect(0.0, 0.0, w, h);
            g.fills.push((body.clone(), [w / 2.0, h / 2.0], false));
            // The title band; with pins on the top edge the title moves into the body.
            if has_header(s) {
                let header = rect(0.0, 0.0, w, 2.0);
                g.fills.push((header, [w / 2.0, 1.0], true));
                g.strokes.push((vec![[0.0, 2.0], [w, 2.0]], false, 0.6));
            }
            // A short tick inside the box at every pin (thicker for buses).
            for p in &s.ports {
                let (x, y) = (p.pos.x as f32, p.pos.y as f32);
                let (dx, dy) = match p.side {
                    Side::Left => (PIN_TICK, 0.0),
                    Side::Right => (-PIN_TICK, 0.0),
                    Side::Top => (0.0, PIN_TICK),
                    Side::Bottom => (0.0, -PIN_TICK),
                };
                let wm = if p.width > 1 { 2.2 } else { 1.2 };
                g.strokes.push((vec![[x, y], [x + dx, y + dy]], false, wm));
            }
            g.strokes.push((body, true, 1.2));
        }
    }
    g
}

/// Length of the pin ticks inside package boxes (grid units).
const PIN_TICK: f32 = 0.45;
/// Inset of pin labels from the box edge (grid units).
const LABEL_INSET: f32 = 0.7;
/// Width of one character per unit of [`SymText::size`] (grid units; the canvas draws text at
/// 0.8 × size with a proportional font averaging ~0.55 em per character).
const CHAR_W: f32 = 0.47;
/// Smallest text size labels shrink to before they are allowed to overflow.
const MIN_TEXT: f32 = 0.55;

/// True if a package symbol draws its title band (no pins on its top edge).
fn has_header(s: &Symbol) -> bool {
    !s.ports.iter().any(|p| p.side == Side::Top)
}

/// Text size ≤ `max` at which `text` fits into `room` grid units.
fn fit(text: &str, room: f32, max: f32) -> f32 {
    let n = text.chars().count().max(1) as f32;
    (room / (CHAR_W * n)).clamp(MIN_TEXT, max)
}

/// The texts of a symbol, in its local (unoriented) frame.
///
/// Package pin names sit inside the box next to their pin: left/right pins horizontally, top /
/// bottom pins vertically (reading away from the edge), each shrunk to fit the room it has
/// (half the box when a label faces another one on the opposite side). The title fills the
/// title band, or the body when pins occupy the top edge.
pub fn symbol_texts(s: &Symbol) -> Vec<SymText> {
    let (w, h) = (s.size.x as f32, s.size.y as f32);
    let t = |pos: [f32; 2], text: &str, size: f32, align: Align2, dim: bool| SymText {
        pos,
        text: text.to_string(),
        size,
        align,
        dim,
        vertical: false,
    };
    match &s.kind {
        SymbolKind::Constant { text } => {
            vec![t([w / 2.0, 1.0], text, fit(text, w - 0.4, 1.1), Align2::CENTER_CENTER, false)]
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
            vec![t([x, 1.0], name, fit(name, w - 1.2, 1.1), Align2::CENTER_CENTER, false)]
        }
        SymbolKind::Package { title, .. } => {
            let mut v = Vec::with_capacity(s.ports.len() + 1);
            let header = has_header(s);
            // Rows / columns used by pins on each side (for the room left to each label).
            let rows = |side: Side| -> Vec<i32> {
                s.ports
                    .iter()
                    .filter(|p| p.side == side)
                    .map(|p| p.pos.y)
                    .collect()
            };
            let cols = |side: Side| -> Vec<i32> {
                s.ports
                    .iter()
                    .filter(|p| p.side == side)
                    .map(|p| p.pos.x)
                    .collect()
            };
            let (left_rows, right_rows) = (rows(Side::Left), rows(Side::Right));
            let (top_cols, bottom_cols) = (cols(Side::Top), cols(Side::Bottom));
            let faces = |others: &[i32], at: i32| others.iter().any(|&o| (o - at).abs() < 2);
            for port in &s.ports {
                let name = port.name.as_str();
                let (x, y) = (port.pos.x as f32, port.pos.y as f32);
                let mut label = match port.side {
                    Side::Left | Side::Right => {
                        let shared = if faces(
                            if port.side == Side::Left {
                                &right_rows
                            } else {
                                &left_rows
                            },
                            port.pos.y,
                        ) {
                            0.5
                        } else {
                            1.0
                        };
                        let room = (w - 2.0 * LABEL_INSET) * shared - 0.3;
                        let size = fit(name, room, 1.0);
                        if port.side == Side::Left {
                            t([LABEL_INSET, y], name, size, Align2::LEFT_CENTER, true)
                        } else {
                            t([w - LABEL_INSET, y], name, size, Align2::RIGHT_CENTER, true)
                        }
                    }
                    Side::Top | Side::Bottom => {
                        let shared = if faces(
                            if port.side == Side::Top {
                                &bottom_cols
                            } else {
                                &top_cols
                            },
                            port.pos.x,
                        ) {
                            0.5
                        } else {
                            1.0
                        };
                        let room = (h - 2.0 * LABEL_INSET) * shared - 0.3;
                        let size = fit(name, room, 1.0);
                        // Vertical text reads upward: a top label ends at the top edge, a bottom
                        // label starts at the bottom edge.
                        if port.side == Side::Top {
                            t([x, LABEL_INSET], name, size, Align2::RIGHT_CENTER, true)
                        } else {
                            t([x, h - LABEL_INSET], name, size, Align2::LEFT_CENTER, true)
                        }
                    }
                };
                label.vertical = matches!(port.side, Side::Top | Side::Bottom);
                v.push(label);
            }
            if header {
                v.push(t([w / 2.0, 1.0], title, fit(title, w - 0.8, 1.15), Align2::CENTER_CENTER, false));
            } else {
                // Between the left and right label columns, below the top labels.
                let room = w - 2.0 * LABEL_INSET - 2.0;
                v.push(t([w / 2.0, h / 2.0], title, fit(title, room, 1.15), Align2::CENTER_CENTER, false));
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

/// A symbol text placed for an orientation: anchor in the oriented symbol's frame (grid units
/// from its top-left corner), alignment at the anchor and rotation (radians clockwise; 0 or
/// −90°, i.e. reading upward).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextPlacement {
    /// Anchor (oriented local grid coordinates).
    pub pos: [f32; 2],
    /// Alignment of the (unrotated) text box at the anchor.
    pub align: Align2,
    /// Clockwise rotation in radians.
    pub angle: f32,
}

/// Place text `t` of a symbol of local size `size` drawn with orientation `o`: the text follows
/// the orientation but always reads left-to-right or bottom-to-top.
pub fn place_text(t: &SymText, o: Orient, size: GridPt) -> TextPlacement {
    let origin = o.apply_f([0.0, 0.0], size);
    let map = |v: [f32; 2]| {
        let p = o.apply_f(v, size);
        [(p[0] - origin[0]).round() as i32, (p[1] - origin[1]).round() as i32]
    };
    // Local reading direction and the local "down" of the text.
    let (b_local, n_local) = if t.vertical {
        ([0.0, -1.0], [1.0, 0.0])
    } else {
        ([1.0, 0.0], [0.0, 1.0])
    };
    let d = map(b_local);
    let u = map(n_local);
    // Readable direction: rightward, or upward for vertical text.
    let b = if d == [1, 0] || d == [0, -1] { d } else { [-d[0], -d[1]] };
    let n = if b == [1, 0] { [0, 1] } else { [1, 0] };
    let mut align = t.align;
    if b != d {
        align = Align2([align.x().flip(), align.y()]);
    }
    if n != u {
        align = Align2([align.x(), align.y().flip()]);
    }
    TextPlacement {
        pos: o.apply_f(t.pos, size),
        align,
        angle: if b == [1, 0] {
            0.0
        } else {
            -std::f32::consts::FRAC_PI_2
        },
    }
}

/// Paint `text` anchored at `pos` with `align` (of the unrotated text box), rotated by `angle`
/// radians clockwise about the anchor.
pub fn paint_text(
    painter: &Painter,
    pos: Pos2,
    align: Align2,
    angle: f32,
    text: String,
    font: FontId,
    color: Color32,
) {
    let galley = painter.layout_no_wrap(text, font, color);
    let rect = align.anchor_size(pos, galley.size());
    if angle == 0.0 {
        painter.galley(rect.min, galley, color);
    } else {
        painter.add(TextShape::new(rect.min, galley, color).with_angle_and_anchor(angle, align));
    }
}

/// Tessellated template meshes for the current zoom and palette.
#[derive(Default)]
pub struct Templates {
    zoom: f32,
    ppp: f32,
    palette: Option<Palette>,
    meshes: HashMap<TemplateKey, Mesh>,
    /// Template for junction dots.
    pub dot: Mesh,
}

impl Templates {
    /// Drop the cache if zoom, pixels-per-point or palette changed.
    pub fn prepare(&mut self, zoom: f32, ppp: f32, palette: &Palette) {
        if self.zoom != zoom || self.ppp != ppp || self.palette.as_ref() != Some(palette) {
            self.meshes.clear();
            self.zoom = zoom;
            self.ppp = ppp;
            self.palette = Some(*palette);
            let mut t = tessellator(ppp);
            self.dot = Mesh::default();
            let r = (zoom * 0.28).max(1.6);
            t.tessellate_shape(
                Shape::circle_filled(Pos2::ZERO, r, Color32::WHITE),
                &mut self.dot,
            );
        }
    }

    /// Stroke width (points) at the current zoom.
    pub fn stroke_width(&self) -> f32 {
        (self.zoom * 0.11).clamp(0.6, 3.0)
    }

    /// The template of symbol `s` (local pixel coordinates, unrotated).
    pub fn get(&mut self, s: &Symbol) -> &Mesh {
        let key = template_key(s);
        let (zoom, ppp, pal) = (self.zoom, self.ppp, self.palette.expect("prepared"));
        let sw = self.stroke_width();
        self.meshes
            .entry(key)
            .or_insert_with(|| build_template(&glyph(s), zoom, ppp, &pal, sw))
    }

    /// Number of cached templates.
    pub fn len(&self) -> usize {
        self.meshes.len()
    }

    /// True if no template is cached.
    pub fn is_empty(&self) -> bool {
        self.meshes.is_empty()
    }
}

fn tessellator(ppp: f32) -> Tessellator {
    Tessellator::new(ppp, TessellationOptions::default(), [1, 1], Vec::new())
}

fn build_template(g: &Glyph, zoom: f32, ppp: f32, pal: &Palette, sw: f32) -> Mesh {
    let mut mesh = Mesh::default();
    let px = |p: P| pos2(p[0] * zoom, p[1] * zoom);
    for (pts, c, header) in &g.fills {
        let color = if *header {
            pal.package_header
        } else {
            pal.fill
        };
        let base = mesh.vertices.len() as u32;
        mesh.colored_vertex(px(*c), color);
        for &p in pts {
            mesh.colored_vertex(px(p), color);
        }
        let n = pts.len() as u32;
        for i in 0..n {
            mesh.add_triangle(base, base + 1 + i, base + 1 + (i + 1) % n);
        }
    }
    let mut t = tessellator(ppp);
    for (pts, closed, wm) in &g.strokes {
        let points: Vec<Pos2> = pts.iter().map(|&p| px(p)).collect();
        let stroke = PathStroke::new(sw * wm, pal.stroke);
        let shape = if *closed {
            PathShape::closed_line(points, stroke)
        } else {
            PathShape::line(points, stroke)
        };
        t.tessellate_path(&shape, &mut mesh);
    }
    for &(c, r) in &g.discs {
        t.tessellate_shape(
            Shape::Circle(egui::epaint::CircleShape {
                center: px(c),
                radius: r * zoom,
                fill: pal.fill,
                stroke: Stroke::new(sw, pal.stroke),
            }),
            &mut mesh,
        );
    }
    mesh
}

/// Append `template` (local pixel coordinates of a symbol of local size `size`) to `out`,
/// oriented by `orient` and placed with its top-left at `at` (screen).
pub fn append_oriented(
    out: &mut Mesh,
    template: &Mesh,
    size: GridPt,
    zoom: f32,
    orient: Orient,
    at: Pos2,
) {
    let base = out.vertices.len() as u32;
    let (w, h) = (size.x as f32 * zoom, size.y as f32 * zoom);
    out.vertices.reserve(template.vertices.len());
    if orient == Orient::default() {
        out.vertices.extend(template.vertices.iter().map(|v| {
            let mut v = *v;
            v.pos = at + v.pos.to_vec2();
            v
        }));
    } else {
        out.vertices.extend(template.vertices.iter().map(|v| {
            let mut v = *v;
            let p = [v.pos.x, v.pos.y];
            let p = if orient.flip { [w - p[0], p[1]] } else { p };
            let q = match orient.rot % 4 {
                0 => p,
                1 => [h - p[1], p[0]],
                2 => [w - p[0], h - p[1]],
                _ => [p[1], w - p[0]],
            };
            v.pos = pos2(at.x + q[0], at.y + q[1]);
            v
        }));
    }
    out.indices
        .extend(template.indices.iter().map(|i| i + base));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::atom_symbol;
    use panda_netlist::Atom;

    #[test]
    fn every_atom_glyph_tessellates_inside_its_box() {
        let atoms = [
            Atom::Gate {
                op: GateOp::And,
                inputs: 2,
                width: 1,
                reduce: false,
            },
            Atom::Gate {
                op: GateOp::Xnor,
                inputs: 3,
                width: 1,
                reduce: false,
            },
            Atom::Gate {
                op: GateOp::Or,
                inputs: 256,
                width: 8,
                reduce: false,
            },
            Atom::Not { width: 1 },
            Atom::Tri { width: 4 },
            Atom::Splitter {
                width: 8,
                parts: vec![(0, 3), (4, 7)],
            },
            Atom::Constant {
                value: panda_netlist::Value::from_u64(8, 0xab),
            },
            Atom::Clock {
                period: 8,
                high: 4,
                phase: 0,
            },
            Atom::PowerOnReset { ticks: 3 },
        ];
        let mut t = Templates::default();
        t.prepare(10.0, 1.0, &Palette::LIGHT);
        for a in &atoms {
            let s = atom_symbol(a);
            let m = t.get(&s).clone();
            assert!(!m.is_empty(), "{a:?}");
            let (w, h) = (s.size.x as f32 * 10.0, s.size.y as f32 * 10.0);
            for v in &m.vertices {
                assert!(
                    v.pos.x > -3.0 && v.pos.y > -3.0 && v.pos.x < w + 3.0 && v.pos.y < h + 3.0,
                    "{a:?}: {:?}",
                    v.pos
                );
            }
        }
        // Rotation keeps the vertices inside the rotated box.
        let s = atom_symbol(&atoms[4]);
        let m = t.get(&s).clone();
        let mut out = Mesh::default();
        append_oriented(
            &mut out,
            &m,
            s.size,
            10.0,
            Orient {
                rot: 1,
                flip: false,
            },
            pos2(100.0, 100.0),
        );
        assert_eq!(out.vertices.len(), m.vertices.len());
        for v in &out.vertices {
            assert!(v.pos.x > 96.0 && v.pos.x < 100.0 + 40.0 + 4.0);
        }
    }

    /// World-frame (oriented local) bounding box of a placed text of `chars` characters.
    fn text_box(t: &SymText, tp: &TextPlacement) -> egui::Rect {
        let len = CHAR_W * t.size * t.text.chars().count() as f32;
        let r = tp.align.anchor_size(pos2(tp.pos[0], tp.pos[1]), egui::vec2(len, 0.8 * t.size));
        if tp.angle == 0.0 {
            return r;
        }
        // −90° about the anchor: (vx, vy) → (vy, −vx).
        let a = pos2(tp.pos[0], tp.pos[1]);
        let rot = |p: Pos2| pos2(a.x + (p.y - a.y), a.y - (p.x - a.x));
        egui::Rect::from_two_pos(rot(r.min), rot(r.max))
    }

    #[test]
    fn package_labels_stay_with_their_pins_in_every_orientation() {
        use crate::geom::{SymbolLayout, SymbolPin, gp, package_symbol};
        use panda_netlist::DefBuilder;
        let mut b = DefBuilder::new("pkg");
        let ins: Vec<_> = ["A", "BUS", "SEL", "EN"].iter().map(|n| b.input(n, 1)).collect();
        let y = b.and(&ins);
        b.output("Y", y);
        b.output("OVF", ins[0]);
        let def = b.finish();
        let pin = |side, offset| SymbolPin { side, offset };
        let layout = SymbolLayout {
            size: gp(12, 10),
            pins: [
                ("A".to_string(), pin(Side::Left, 3)),
                ("BUS".to_string(), pin(Side::Left, 5)),
                ("SEL".to_string(), pin(Side::Top, 4)),
                ("EN".to_string(), pin(Side::Bottom, 6)),
                ("Y".to_string(), pin(Side::Right, 3)),
                ("OVF".to_string(), pin(Side::Right, 7)),
            ]
            .into_iter()
            .collect(),
        };
        let s = package_symbol(&def, Some(&layout));
        let texts = symbol_texts(&s);
        assert_eq!(texts.len(), s.ports.len() + 1);
        for rot in 0..4u8 {
            for flip in [false, true] {
                let o = Orient { rot, flip };
                let size = o.size(s.size);
                for (t, port) in texts.iter().zip(&s.ports) {
                    let tp = place_text(t, o, s.size);
                    // Readable: horizontal or reading upward.
                    assert!(tp.angle == 0.0 || tp.angle == -std::f32::consts::FRAC_PI_2);
                    let r = text_box(t, &tp);
                    let tol = 0.05;
                    assert!(
                        r.min.x >= -tol
                            && r.min.y >= -tol
                            && r.max.x <= size.x as f32 + tol
                            && r.max.y <= size.y as f32 + tol,
                        "{o:?} {}: {r:?} outside {size:?}",
                        t.text
                    );
                    // Attached to its pin: the box starts within an inset of the pin.
                    let pp = o.apply_f([port.pos.x as f32, port.pos.y as f32], s.size);
                    let dx = (pp[0] - r.center().x).abs() - r.width() / 2.0;
                    let dy = (pp[1] - r.center().y).abs() - r.height() / 2.0;
                    assert!(
                        dx.max(0.0) + dy.max(0.0) <= LABEL_INSET + tol,
                        "{o:?} {}: {r:?} far from pin {pp:?}",
                        t.text
                    );
                    // Labels run perpendicular to the edge their pin is on.
                    let edge_vertical = pp[0] < 0.01 || pp[0] > size.x as f32 - 0.01;
                    assert_eq!(tp.angle == 0.0, edge_vertical, "{o:?} {}", t.text);
                }
            }
        }
    }
}
