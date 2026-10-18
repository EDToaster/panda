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
/// background box, clockwise rotation about the anchor (radians).
type CanvasText = (Pos2, String, f32, Align2, Color32, Option<Color32>, f32);

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
        crate::theme::apply(ui.ctx());
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
        for (pos, text, size, align, color, bg, angle) in texts {
            if let Some(bg) = bg {
                let galley = painter.layout_no_wrap(text, FontId::monospace(size), color);
                let pad = vec2(galley.size().y * 0.45, 1.5);
                let r = align.anchor_size(pos, galley.size()).expand2(pad);
                painter.rect_filled(r, r.height() / 2.0, bg);
                painter.galley(r.min + pad, galley, color);
            } else {
                shapes::paint_text(
                    &painter,
                    pos,
                    align,
                    angle,
                    text,
                    FontId::proportional(size),
                    color,
                );
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
        // Grid: a dot at every grid point (every 5th when zoomed out), stronger every 10.
        if z >= 4.0 {
            let step = if z >= 9.0 { 1 } else { 5 };
            let (x0, x1) = (vis.min.x.floor() as i32, vis.max.x.ceil() as i32);
            let (y0, y1) = (vis.min.y.floor() as i32, vis.max.y.ceil() as i32);
            let d = (z * 0.09).clamp(0.9, 1.6);
            for y in (y0..=y1).filter(|y| y.rem_euclid(step) == 0) {
                for x in (x0..=x1).filter(|x| x.rem_euclid(step) == 0) {
                    let major = x.rem_euclid(10) == 0 && y.rem_euclid(10) == 0;
                    let c = xf.to_screen(pos2(x as f32, y as f32));
                    let (r, col) = if major {
                        (d * 1.4, pal.grid_major)
                    } else {
                        (d, pal.grid)
                    };
                    aa_rect(
                        &mut mesh,
                        Rect::from_center_size(c, vec2(r, r) * 2.0),
                        col,
                        false,
                    );
                }
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
            // Pin markers light up with the value on their net.
            if z >= LOD_BOXES
                && let crate::geom::SymbolKind::Pin { dir, .. } = &p.symbol.kind
            {
                let width = p.symbol.ports.first().map_or(1, |q| q.width);
                let value = live.and_then(|l| {
                    let net = scene.pin_nets.get(i)?.as_ref()?;
                    l.map.net_value(net, l.values)
                });
                let color = if value.is_some() {
                    style::wire_look(pal, width, value.as_ref(), false).color
                } else {
                    pal.wire
                };
                let (c, rad) = shapes::pin_indicator(*dir, p.symbol.size.x as f32);
                let c = p.geom.orient.apply_f(c, p.symbol.size);
                disc(&mut mesh, at + vec2(c[0], c[1]) * z, rad * z, color);
            }
            if hl.items.contains(&p.item) {
                rounded_outline(
                    &mut mesh,
                    xf.rect_to_screen(r).expand(3.0),
                    (z * 0.6).clamp(3.0, 10.0),
                    pal.accent,
                    2.0,
                    ppp,
                );
            }
            if text_ok {
                for t in &scene.texts[i] {
                    let size = t.size * z;
                    if size < 6.0 {
                        continue;
                    }
                    let tp = shapes::place_text(t, p.geom.orient, p.symbol.size);
                    let pos = at + vec2(tp.pos[0], tp.pos[1]) * z;
                    let color = match (t.on_chip, t.dim) {
                        (true, false) => pal.chip_text,
                        (true, true) => pal.chip_text_dim,
                        (false, false) => pal.text,
                        (false, true) => pal.text_dim,
                    };
                    texts.push((
                        pos,
                        t.text.clone(),
                        size * 0.8,
                        tp.align,
                        color,
                        None,
                        tp.angle,
                    ));
                }
            }
        }
        // Free labels (group names): outer groups stay legible further out.
        {
            for l in &scene.labels {
                // Outer groups stay legible further out.
                let size = match l.text.matches('.').count() {
                    0 => (z * 1.4).max(14.0),
                    1 if z >= 0.6 => (z * 1.4).max(11.0),
                    2 if z >= 2.0 => (z * 1.4).max(9.0),
                    _ => z * 1.4,
                }
                .min(20.0);
                let p = xf.grid_to_screen(l.pos + crate::geom::gp(0, 2));
                if size >= 8.0 && rect.expand(300.0).contains(p) {
                    texts.push((
                        p,
                        l.text.clone(),
                        size,
                        Align2::LEFT_BOTTOM,
                        pal.text_dim,
                        Some(pal.bg),
                        0.0,
                    ));
                }
            }
        }
        // Wires: grid-snapped polylines drawn with rounded corners and round ends; buses get a
        // dark casing around a thinner value-coloured core and a value badge.
        let base_w = (z * 0.24).clamp(1.5, 7.0);
        let corner = z * 0.75;
        // Screen rects of placed bus labels (labels that would overlap one are skipped).
        let mut label_rects: Vec<Rect> = Vec::new();
        let mut n_segs = 0;
        let mut seg_vis: Vec<usize> = Vec::new();
        let nets_iter = if z < LOD_DENSITY {
            &scene.nets[..0]
        } else {
            &scene.nets[..]
        };
        for net in nets_iter {
            if !in_view(&net.bounds) {
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
            let bus = net.width > 1;
            let (core_w, case_w) = if bus {
                (base_w * 1.3, base_w * 1.3 + 2.0 * (base_w * 0.45).max(1.2))
            } else {
                (base_w, 0.0)
            };
            let outer_w = core_w.max(case_w);
            let accent = hl.nets.contains(&net.id);
            let fine = z >= LOD_BOXES;
            // Visible segments (culled per segment; tiny ones become density marks).
            seg_vis.clear();
            for (k, &(a, b)) in net.segs.iter().enumerate() {
                let r = Rect::from_two_pos(xf.grid_to_screen(a), xf.grid_to_screen(b));
                if !r.expand(outer_w + 3.0).intersects(rect) {
                    continue;
                }
                n_segs += 1;
                if r.width().max(r.height()) < Density::CELL * 0.5 && !fine {
                    density.mark(r.center());
                    continue;
                }
                seg_vis.push(k);
            }
            if !fine {
                for &k in &seg_vis {
                    let (a, b) = net.segs[k];
                    let r = Rect::from_two_pos(xf.grid_to_screen(a), xf.grid_to_screen(b));
                    aa_rect(&mut mesh, r.expand(outer_w / 2.0), look.color, true);
                }
                continue;
            }
            let mut passes: Vec<(f32, Color32)> = Vec::with_capacity(3);
            if accent {
                passes.push((outer_w + 5.0, pal.accent.gamma_multiply(0.7)));
            }
            if bus {
                passes.push((case_w, pal.bus_casing));
            }
            passes.push((core_w, look.color));
            for (w, color) in passes {
                draw_net_pass(&mut mesh, net, &seg_vis, &xf, rect, w, corner, color);
            }
            for &j in &net.junctions {
                let c = xf.grid_to_screen(j);
                if rect.contains(c) {
                    if bus {
                        disc(&mut mesh, c, case_w * 0.75, pal.bus_casing);
                        disc(&mut mesh, c, core_w * 0.85, look.color);
                    } else {
                        disc(&mut mesh, c, (core_w * 1.1).max(z * 0.22), look.color);
                    }
                }
            }
            if text_ok && let (Some(label), Some(at)) = (look.label, net.label_at) {
                let p = xf.grid_to_screen(at);
                let size = (0.8 * z).clamp(8.0, 15.0);
                // Approximate monospace extent (0.6 em per char).
                let r = Align2::CENTER_CENTER
                    .anchor_size(p, vec2(size * 0.62 * label.len() as f32, size * 1.1))
                    .expand(4.0);
                if rect.contains(p)
                    && !label_rects.iter().any(|o| o.intersects(r))
                    && label_rects.len() < 400
                {
                    label_rects.push(r);
                    texts.push((
                        p,
                        label,
                        size,
                        Align2::CENTER_CENTER,
                        pal.bg,
                        Some(pal.bus_casing),
                        0.0,
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

/// Rounded rectangle outline of width `w` (corner radius `radius`).
fn rounded_outline(mesh: &mut Mesh, r: Rect, radius: f32, color: Color32, w: f32, ppp: f32) {
    let mut t = egui::epaint::Tessellator::new(
        ppp,
        egui::epaint::TessellationOptions::default(),
        [1, 1],
        Vec::new(),
    );
    t.tessellate_shape(
        egui::Shape::rect_stroke(
            r,
            radius,
            egui::Stroke::new(w, color),
            egui::StrokeKind::Outside,
        ),
        mesh,
    );
}

/// Anti-aliased filled disc.
fn disc(mesh: &mut Mesh, c: Pos2, r: f32, color: Color32) {
    let n = ((r * 1.2) as usize).clamp(10, 40);
    let base = mesh.vertices.len() as u32;
    mesh.colored_vertex(c, color);
    let (ri, ro) = ((r - 0.5).max(0.0), r + 0.5);
    for i in 0..n {
        let a = std::f32::consts::TAU * i as f32 / n as f32;
        let u = vec2(a.cos(), a.sin());
        mesh.colored_vertex(c + u * ri, color);
        mesh.colored_vertex(c + u * ro, Color32::TRANSPARENT);
    }
    for i in 0..n as u32 {
        let j = (i + 1) % n as u32;
        let (a, b) = (base + 1 + 2 * i, base + 1 + 2 * j);
        mesh.add_triangle(base, a, b);
        mesh.add_triangle(a, a + 1, b + 1);
        mesh.add_triangle(a, b + 1, b);
    }
}

/// Anti-aliased band of an arc: centre `c`, centre-line radius `r`, half width `hw`, from angle
/// `a0` sweeping `sweep` radians.
fn arc_band(mesh: &mut Mesh, c: Pos2, r: f32, hw: f32, a0: f32, sweep: f32, color: Color32) {
    let n = ((r * sweep.abs() / 3.0) as usize).clamp(4, 16);
    let radii = [
        (r - hw - 0.5).max(0.0),
        (r - hw + 0.5).max(0.0),
        r + hw - 0.5,
        r + hw + 0.5,
    ];
    let cols = [Color32::TRANSPARENT, color, color, Color32::TRANSPARENT];
    let base = mesh.vertices.len() as u32;
    for i in 0..=n {
        let a = a0 + sweep * i as f32 / n as f32;
        let u = vec2(a.cos(), a.sin());
        for k in 0..4 {
            mesh.colored_vertex(c + u * radii[k], cols[k]);
        }
    }
    for i in 0..n as u32 {
        let (p, q) = (base + 4 * i, base + 4 * (i + 1));
        for k in 0..3 {
            mesh.add_triangle(p + k, p + k + 1, q + k + 1);
            mesh.add_triangle(p + k, q + k + 1, q + k);
        }
    }
}

/// One stroke pass (width `w`, colour `color`) over a net's visible segments `vis`: segments
/// are trimmed at bends, which get quarter-circle arcs of radius up to `radius`; free ends get
/// round caps.
#[allow(clippy::too_many_arguments)]
fn draw_net_pass(
    mesh: &mut Mesh,
    net: &scene::SceneNet,
    vis: &[usize],
    xf: &ViewXf,
    clip: Rect,
    w: f32,
    radius: f32,
    color: Color32,
) {
    let hw = w / 2.0;
    let z = xf.zoom;
    let corner_r = |c: u32| radius.min(net.corners[c as usize].max_r * z);
    for &k in vis {
        let (a, b) = net.segs[k];
        let (a, b) = (xf.grid_to_screen(a), xf.grid_to_screen(b));
        let d = (b - a).normalized();
        let [ca, cb] = net.seg_corners[k];
        let a = a + d * ca.map_or(0.0, corner_r);
        let b = b - d * cb.map_or(0.0, corner_r);
        let r = if (a.y - b.y).abs() < 1e-3 {
            Rect::from_min_max(pos2(a.x.min(b.x), a.y - hw), pos2(a.x.max(b.x), a.y + hw))
        } else {
            Rect::from_min_max(pos2(a.x - hw, a.y.min(b.y)), pos2(a.x + hw, a.y.max(b.y)))
        };
        aa_rect(mesh, r, color, true);
    }
    for (i, c) in net.corners.iter().enumerate() {
        let at = xf.grid_to_screen(c.at);
        let r = corner_r(i as u32);
        if !clip.expand(r + w).contains(at) {
            continue;
        }
        let (d0, d1) = (
            vec2(c.dirs[0][0], c.dirs[0][1]),
            vec2(c.dirs[1][0], c.dirs[1][1]),
        );
        let center = at + (d0 + d1) * r;
        let a0 = (-d1).angle();
        let mut sweep = (-d0).angle() - a0;
        while sweep > std::f32::consts::PI {
            sweep -= std::f32::consts::TAU;
        }
        while sweep < -std::f32::consts::PI {
            sweep += std::f32::consts::TAU;
        }
        arc_band(mesh, center, r, hw, a0, sweep, color);
    }
    for &p in &net.caps {
        let c = xf.grid_to_screen(p);
        if clip.expand(w).contains(c) {
            disc(mesh, c, hw, color);
        }
    }
}
