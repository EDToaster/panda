//! The schematic canvas: a read-only view of one def with live values.
//!
//! * Pan (middle drag; primary drag on empty space when no tool claims it), cursor-anchored zoom
//!   (wheel / pinch), fit-to-content on entering a def.
//! * Symbols are drawn from per-zoom tessellated templates ([`shapes`]) copied into one mesh per
//!   frame; wires are anti-aliased axis-aligned quads in the same mesh. Items and wires outside
//!   the viewport are culled; below a few points per grid unit symbols degrade to boxes and text
//!   is skipped, so 10k+ gate defs stay at 60 fps.
//! * Wire colour follows the live value ([`style::wire_look`]): 0 / 1 / X / Z / contention,
//!   thicker buses with a hex label.
//!
//! Tools (editing, wave probes) do not draw into the canvas: they get the [`CanvasResponse`]
//! (screen rect, transform, hovered world point) and paint overlays with `ui.painter_at(rect)`;
//! selection and highlights are passed in through [`Highlights`].

pub mod scene;
pub mod shapes;
pub mod style;
pub mod view;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use egui::{
    Align2, Color32, FontId, Mesh, PointerButton, Pos2, Rect, Response, Sense, Ui, pos2, vec2,
};
use panda_netlist::Logic;

use crate::doc::values::ViewMap;
use crate::doc::{Document, ItemRef};

pub use scene::Scene;
pub use style::Palette;
pub use view::ViewXf;

/// Below this zoom (points per grid unit) symbols are drawn as plain boxes.
pub const LOD_BOXES: f32 = 2.5;
/// Below this zoom everything is drawn into a coarse occupancy raster (overview).
pub const LOD_DENSITY: f32 = 1.0;
/// Below this zoom no text is drawn.
pub const LOD_TEXT: f32 = 5.0;

/// Live values for the canvas: the view's mapping and the snapshot values aligned with
/// [`ViewMap::watch`].
#[derive(Clone, Copy)]
pub struct LiveValues<'a> {
    /// Mapping of the viewed def.
    pub map: &'a ViewMap,
    /// Values aligned with `map.watch`.
    pub values: &'a [Logic],
}

/// Things to emphasize (selection, hovered net, critical path, diagnostics).
#[derive(Clone, Debug, Default)]
pub struct Highlights {
    /// Items outlined in the accent colour.
    pub items: HashSet<ItemRef>,
    /// Nets drawn with an accent underlay.
    pub nets: HashSet<String>,
    /// Nets drawn in the contention colour (e.g. from `Contention` diagnostics).
    pub contention: HashSet<String>,
}

/// What the canvas reports back each frame.
pub struct CanvasResponse {
    /// The canvas widget response (click / drag senses).
    pub response: Response,
    /// Screen rectangle.
    pub rect: Rect,
    /// Transform used this frame.
    pub xf: ViewXf,
    /// World position (grid units) under the pointer.
    pub hover_world: Option<Pos2>,
}

/// A text queued while building the mesh: position, text, size, alignment, colour, optional
/// background box.
type CanvasText = (Pos2, String, f32, Align2, Color32, Option<Color32>);

/// Per-frame drawing statistics.
#[derive(Clone, Copy, Debug, Default)]
pub struct CanvasStats {
    /// Items drawn (after culling).
    pub items: usize,
    /// Wire segments drawn.
    pub segments: usize,
    /// Mesh vertices.
    pub vertices: usize,
    /// Texts drawn.
    pub texts: usize,
    /// Time to build the frame's mesh (ms).
    pub build_ms: f64,
}

/// Canvas state (one per editor view).
#[derive(Default)]
pub struct Canvas {
    /// View transform.
    pub xf: ViewXf,
    scene: Option<Arc<Scene>>,
    templates: shapes::Templates,
    fit_pending: bool,
    /// Primary-button drags pan the view (set false while a tool handles the primary button).
    pub primary_pans: bool,
    /// Statistics of the last frame.
    pub stats: CanvasStats,
}

impl Canvas {
    /// A canvas that fits the first def it shows.
    pub fn new() -> Canvas {
        Canvas {
            fit_pending: true,
            primary_pans: true,
            ..Canvas::default()
        }
    }

    /// The current scene, if built.
    pub fn scene(&self) -> Option<&Arc<Scene>> {
        self.scene.as_ref()
    }

    /// Rebuild the scene if the def or the document revision changed.
    pub fn ensure_scene(&mut self, doc: &Document, def: &str) -> &Arc<Scene> {
        let stale = self
            .scene
            .as_ref()
            .is_none_or(|s| s.def != def || s.revision != doc.revision);
        if stale {
            if self.scene.as_ref().is_none_or(|s| s.def != def) {
                self.fit_pending = true;
            }
            self.scene = Some(Arc::new(Scene::build(doc, def)));
        }
        self.scene.as_ref().expect("built")
    }

    /// Fit the content on the next frame.
    pub fn request_fit(&mut self) {
        self.fit_pending = true;
    }

    /// Show the canvas filling the available space.
    pub fn show(
        &mut self,
        ui: &mut Ui,
        doc: &Document,
        def: &str,
        live: Option<LiveValues<'_>>,
        hl: &Highlights,
    ) -> CanvasResponse {
        let rect = ui.available_rect_before_wrap();
        let response = ui.allocate_rect(rect, Sense::click_and_drag());
        let scene = self.ensure_scene(doc, def).clone();
        if self.fit_pending && rect.width() > 10.0 {
            self.xf.fit(scene.bounds_rect(), rect, 24.0);
            self.fit_pending = false;
        }
        // Navigation.
        if response.hovered() {
            let (scroll, pinch, hover) =
                ui.input(|i| (i.smooth_scroll_delta, i.zoom_delta(), i.pointer.hover_pos()));
            let factor = (scroll.y * 0.0015).exp() * pinch;
            if let Some(p) = hover
                && (factor - 1.0).abs() > 1e-4
            {
                self.xf.zoom_at(p, factor);
            }
            if scroll.x != 0.0 {
                self.xf.pan(vec2(scroll.x, 0.0));
            }
        }
        if response.dragged_by(PointerButton::Middle)
            || (self.primary_pans && response.dragged_by(PointerButton::Primary))
        {
            self.xf.pan(response.drag_delta());
        }
        let hover_world = response.hover_pos().map(|p| self.xf.to_world(p));
        let painter = ui.painter_at(rect);
        let pal = Palette::for_dark_mode(ui.visuals().dark_mode);
        painter.rect_filled(rect, 0.0, pal.bg);
        let t0 = Instant::now();
        let ppp = ui.ctx().pixels_per_point();
        let mut texts: Vec<CanvasText> = Vec::new();
        let mesh = self.build_mesh(&scene, rect, &pal, ppp, live, hl, &mut texts);
        self.stats.vertices = mesh.vertices.len();
        self.stats.build_ms = t0.elapsed().as_secs_f64() * 1e3;
        painter.add(egui::Shape::mesh(mesh));
        self.stats.texts = texts.len();
        for (pos, text, size, align, color, bg) in texts {
            if let Some(bg) = bg {
                let galley = painter.layout_no_wrap(text, FontId::monospace(size), color);
                let r = align.anchor_size(pos, galley.size()).expand(1.5);
                painter.rect_filled(r, 2.0, bg);
                painter.galley(r.min + vec2(1.5, 1.5), galley, color);
            } else {
                painter.text(pos, align, text, FontId::proportional(size), color);
            }
        }
        CanvasResponse {
            response,
            rect,
            xf: self.xf,
            hover_world,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_mesh(
        &mut self,
        scene: &Scene,
        rect: Rect,
        pal: &Palette,
        ppp: f32,
        live: Option<LiveValues<'_>>,
        hl: &Highlights,
        texts: &mut Vec<CanvasText>,
    ) -> Mesh {
        let xf = self.xf;
        let z = xf.zoom;
        let mut mesh = Mesh::default();
        let vis = xf.visible_world(rect).expand(1.0);
        let in_view = |r: &crate::geom::GridRect| {
            (r.max.x as f32) >= vis.min.x
                && (r.min.x as f32) <= vis.max.x
                && (r.max.y as f32) >= vis.min.y
                && (r.min.y as f32) <= vis.max.y
        };
        // Grid.
        if z >= 4.0 {
            let step = if z >= 10.0 { 1 } else { 5 };
            let (x0, x1) = (vis.min.x.floor() as i32, vis.max.x.ceil() as i32);
            let (y0, y1) = (vis.min.y.floor() as i32, vis.max.y.ceil() as i32);
            for x in (x0..=x1).filter(|x| x.rem_euclid(step) == 0) {
                let c = if x.rem_euclid(10) == 0 {
                    pal.grid_major
                } else {
                    pal.grid
                };
                let sx = xf.to_screen(pos2(x as f32, 0.0)).x;
                aa_rect(
                    &mut mesh,
                    Rect::from_min_max(pos2(sx - 0.5, rect.min.y), pos2(sx + 0.5, rect.max.y)),
                    c,
                    false,
                );
            }
            for y in (y0..=y1).filter(|y| y.rem_euclid(step) == 0) {
                let c = if y.rem_euclid(10) == 0 {
                    pal.grid_major
                } else {
                    pal.grid
                };
                let sy = xf.to_screen(pos2(0.0, y as f32)).y;
                aa_rect(
                    &mut mesh,
                    Rect::from_min_max(pos2(rect.min.x, sy - 0.5), pos2(rect.max.x, sy + 0.5)),
                    c,
                    false,
                );
            }
        }
        // Symbols.
        self.templates.prepare(z, ppp, pal);
        let text_ok = z >= LOD_TEXT;
        let mut n_items = 0;
        // Things smaller than a couple of points go into a coarse occupancy raster instead.
        let mut density = Density::new(rect);
        if z < LOD_DENSITY {
            draw_overview(
                &mut mesh,
                scene.overview(),
                &xf,
                rect,
                pal.stroke.gamma_multiply(0.45),
            );
        }
        let items_iter = if z < LOD_DENSITY {
            &scene.items[..0]
        } else {
            &scene.items[..]
        };
        for (i, (p, r)) in items_iter.iter().zip(&scene.rects).enumerate() {
            if !in_view(r) {
                continue;
            }
            n_items += 1;
            let at = xf.grid_to_screen(p.geom.pos);
            if z < LOD_BOXES {
                let sr = xf.rect_to_screen(r);
                if z < LOD_DENSITY {
                    // Drawn by the overview pyramid below.
                } else if sr.width().max(sr.height()) < Density::CELL {
                    density.mark(sr.center());
                } else {
                    aa_rect(&mut mesh, sr, pal.stroke.gamma_multiply(0.55), false);
                }
            } else {
                let t = self.templates.get(&p.symbol);
                shapes::append_oriented(&mut mesh, t, p.symbol.size, z, p.geom.orient, at);
            }
            if hl.items.contains(&p.item) {
                outline(&mut mesh, xf.rect_to_screen(r).expand(2.0), pal.accent, 2.0);
            }
            if text_ok {
                for t in &scene.texts[i] {
                    let size = t.size * z;
                    if size < 6.0 {
                        continue;
                    }
                    let lp = p.geom.orient.apply_f(t.pos, p.symbol.size);
                    let pos = at + vec2(lp[0], lp[1]) * z;
                    let color = if t.dim { pal.text_dim } else { pal.text };
                    texts.push((
                        pos,
                        t.text.clone(),
                        size * 0.8,
                        oriented_align(t.align, p.geom.orient),
                        color,
                        None,
                    ));
                }
            }
        }
        // Wires.
        let base_w = (z * 0.13).clamp(1.0, 3.5);
        // Screen rects of placed bus labels (labels that would overlap one are skipped).
        let mut label_rects: Vec<Rect> = Vec::new();
        let mut n_segs = 0;
        let nets_iter = if z < LOD_DENSITY {
            &scene.nets[..0]
        } else {
            &scene.nets[..]
        };
        for net in nets_iter {
            if !in_view(&net.bounds) {
                continue;
            }
            // Overview zoom: drawn by the overview pyramid (no live colours at this scale).
            if z < LOD_DENSITY {
                continue;
            }
            // Whole net below the raster resolution: one density mark.
            let nb = xf.rect_to_screen(&net.bounds);
            if nb.width().max(nb.height()) < Density::CELL {
                density.mark(nb.center());
                continue;
            }
            let value = live.and_then(|l| l.map.net_value(&net.id, l.values));
            let look = style::wire_look(
                pal,
                net.width,
                value.as_ref(),
                hl.contention.contains(&net.id),
            );
            let w = (base_w * look.width).max(1.0);
            let accent = hl.nets.contains(&net.id);
            for &(a, b) in &net.segs {
                let sa = xf.grid_to_screen(a);
                let sb = xf.grid_to_screen(b);
                let r = Rect::from_two_pos(sa, sb);
                if !r.expand(w).intersects(rect) {
                    continue;
                }
                if z < LOD_BOXES && r.width().max(r.height()) < Density::CELL {
                    density.mark(r.center());
                    continue;
                }
                n_segs += 1;
                if accent {
                    aa_rect(
                        &mut mesh,
                        r.expand(w / 2.0 + 2.5),
                        pal.accent.gamma_multiply(0.6),
                        true,
                    );
                }
                aa_rect(&mut mesh, r.expand(w / 2.0), look.color, true);
            }
            if z >= LOD_BOXES {
                for &j in &net.junctions {
                    let c = xf.grid_to_screen(j);
                    if rect.contains(c) {
                        append_tinted(&mut mesh, &self.templates.dot, c, look.color);
                    }
                }
            }
            if text_ok && let (Some(label), Some(at)) = (look.label, net.label_at) {
                let p = xf.grid_to_screen(at) - vec2(0.0, w / 2.0 + 1.0);
                let size = (0.9 * z).clamp(8.0, 16.0);
                // Approximate monospace extent (0.6 em per char).
                let r = Align2::CENTER_BOTTOM
                    .anchor_size(p, vec2(size * 0.62 * label.len() as f32, size * 1.1))
                    .expand(2.0);
                if rect.contains(p)
                    && !label_rects.iter().any(|o| o.intersects(r))
                    && label_rects.len() < 400
                {
                    label_rects.push(r);
                    texts.push((
                        p,
                        label,
                        size,
                        Align2::CENTER_BOTTOM,
                        look.color,
                        Some(pal.bg),
                    ));
                }
            }
        }
        density.draw(&mut mesh, pal.stroke.gamma_multiply(0.45));
        self.stats.items = n_items;
        self.stats.segments = n_segs;
        mesh
    }
}

/// Draw the visible occupied cells of the overview level matching the zoom, merged into
/// horizontal runs.
fn draw_overview(mesh: &mut Mesh, ov: &scene::Overview, xf: &ViewXf, rect: Rect, color: Color32) {
    let l = ov.level_for(xf.zoom, 1.5);
    let vis = xf.visible_world(rect);
    let c = l.cell as f32;
    let (ox, oy) = (ov.origin.x as f32, ov.origin.y as f32);
    let x0 = (((vis.min.x - ox) / c).floor().max(0.0) as usize).min(l.w);
    let x1 = (((vis.max.x - ox) / c).ceil().max(0.0) as usize).min(l.w);
    let y0 = (((vis.min.y - oy) / c).floor().max(0.0) as usize).min(l.h);
    let y1 = (((vis.max.y - oy) / c).ceil().max(0.0) as usize).min(l.h);
    for y in y0..y1 {
        let row = &l.bits[y * l.w..(y + 1) * l.w];
        let mut x = x0;
        while x < x1 {
            if !row[x] {
                x += 1;
                continue;
            }
            let s = x;
            while x < x1 && row[x] {
                x += 1;
            }
            let min = xf.to_screen(pos2(ox + s as f32 * c, oy + y as f32 * c));
            let max = xf.to_screen(pos2(ox + x as f32 * c, oy + (y + 1) as f32 * c));
            aa_rect(mesh, Rect::from_min_max(min, max), color, false);
        }
    }
}

/// Coarse occupancy raster for detail below the screen resolution (zoomed far out): marked
/// cells are drawn as merged horizontal runs.
struct Density {
    origin: Pos2,
    w: usize,
    h: usize,
    cells: Vec<bool>,
    any: bool,
}

impl Density {
    /// Cell size (points).
    const CELL: f32 = 2.0;

    fn new(rect: Rect) -> Density {
        let w = (rect.width() / Self::CELL).ceil().max(1.0) as usize;
        let h = (rect.height() / Self::CELL).ceil().max(1.0) as usize;
        Density {
            origin: rect.min,
            w,
            h,
            cells: Vec::new(),
            any: false,
        }
    }

    fn mark(&mut self, p: Pos2) {
        let x = ((p.x - self.origin.x) / Self::CELL).floor();
        let y = ((p.y - self.origin.y) / Self::CELL).floor();
        if x < 0.0 || y < 0.0 || x >= self.w as f32 || y >= self.h as f32 {
            return;
        }
        if !self.any {
            self.cells = vec![false; self.w * self.h];
            self.any = true;
        }
        self.cells[y as usize * self.w + x as usize] = true;
    }

    fn draw(&self, mesh: &mut Mesh, color: Color32) {
        if !self.any {
            return;
        }
        for y in 0..self.h {
            let row = &self.cells[y * self.w..(y + 1) * self.w];
            let mut x = 0;
            while x < self.w {
                if !row[x] {
                    x += 1;
                    continue;
                }
                let x0 = x;
                while x < self.w && row[x] {
                    x += 1;
                }
                let min = self.origin + vec2(x0 as f32, y as f32) * Self::CELL;
                let max = self.origin + vec2(x as f32, (y + 1) as f32) * Self::CELL;
                aa_rect(mesh, Rect::from_min_max(min, max), color, false);
            }
        }
    }
}

/// Alignment of symbol text after orienting the symbol (left/right swap when mirrored).
fn oriented_align(a: Align2, o: crate::geom::Orient) -> Align2 {
    let mirrored = o.flip ^ (o.rot % 4 == 2);
    match o.rot % 2 {
        1 => Align2::CENTER_CENTER,
        _ if mirrored => Align2([a.x().flip(), a.y()]),
        _ => a,
    }
}

/// Anti-aliased axis-aligned filled rectangle (solid core, 1-point feather ring).
fn aa_rect(mesh: &mut Mesh, r: Rect, color: Color32, feather: bool) {
    let f: f32 = if feather { 0.5 } else { 0.0 };
    let inner = r.shrink(f.min(r.width() / 2.0).min(r.height() / 2.0));
    let base = mesh.vertices.len() as u32;
    for p in [
        inner.left_top(),
        inner.right_top(),
        inner.right_bottom(),
        inner.left_bottom(),
    ] {
        mesh.colored_vertex(p, color);
    }
    mesh.add_triangle(base, base + 1, base + 2);
    mesh.add_triangle(base, base + 2, base + 3);
    if f > 0.0 {
        let outer = r.expand(f);
        for p in [
            outer.left_top(),
            outer.right_top(),
            outer.right_bottom(),
            outer.left_bottom(),
        ] {
            mesh.colored_vertex(p, Color32::TRANSPARENT);
        }
        for i in 0..4 {
            let (a, b) = (base + i, base + (i + 1) % 4);
            let (c, d) = (base + 4 + i, base + 4 + (i + 1) % 4);
            mesh.add_triangle(a, c, d);
            mesh.add_triangle(a, d, b);
        }
    }
}

/// Rectangle outline of width `w`.
fn outline(mesh: &mut Mesh, r: Rect, color: Color32, w: f32) {
    aa_rect(
        mesh,
        Rect::from_min_max(r.left_top(), pos2(r.right(), r.top() + w)),
        color,
        true,
    );
    aa_rect(
        mesh,
        Rect::from_min_max(pos2(r.left(), r.bottom() - w), r.right_bottom()),
        color,
        true,
    );
    aa_rect(
        mesh,
        Rect::from_min_max(r.left_top(), pos2(r.left() + w, r.bottom())),
        color,
        true,
    );
    aa_rect(
        mesh,
        Rect::from_min_max(pos2(r.right() - w, r.top()), r.right_bottom()),
        color,
        true,
    );
}

/// Append a white template mesh translated to `at` and tinted with `color`.
fn append_tinted(out: &mut Mesh, t: &Mesh, at: Pos2, color: Color32) {
    let base = out.vertices.len() as u32;
    out.vertices.extend(t.vertices.iter().map(|v| {
        let mut v = *v;
        v.pos = at + v.pos.to_vec2();
        v.color = tint(v.color, color);
        v
    }));
    out.indices.extend(t.indices.iter().map(|i| i + base));
}

/// Multiply a premultiplied white-ish colour by `color`.
fn tint(white: Color32, color: Color32) -> Color32 {
    let a = white.a() as f32 / 255.0;
    color.gamma_multiply(a)
}
