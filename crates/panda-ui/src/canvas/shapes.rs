//! Symbol glyphs: outlines in local grid coordinates, tessellated once per zoom into pixel-space
//! template meshes that are copied (rotated / mirrored in 90° steps, which keeps anti-aliasing
//! exact) for every visible instance.

use std::collections::HashMap;

use egui::epaint::{PathShape, PathStroke, TessellationOptions, Tessellator};
use egui::{Color32, Mesh, Pos2, Shape, Stroke, pos2};
use panda_netlist::{Dir, GateOp};

use crate::geom::{GridPt, Orient, Symbol, SymbolKind};

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
    /// Package box (size).
    Package(i32, i32),
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
        SymbolKind::Package { .. } => TemplateKey::Package(s.size.x, s.size.y),
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
            let header = rect(0.0, 0.0, w, 2.0);
            g.fills.push((header, [w / 2.0, 1.0], true));
            g.strokes.push((vec![[0.0, 2.0], [w, 2.0]], false, 0.6));
            g.strokes.push((body, true, 1.2));
        }
    }
    g
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
}
