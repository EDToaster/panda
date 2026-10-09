//! Canvas interaction (the editing [`CanvasTool`]) and the edit toolbar.
//!
//! Gestures (Select tool): click selects (Shift toggles), drag on an item moves the selection
//! (snapped to the grid), drag on empty space box-selects, drag from a port or from a wire
//! draws a wire (Manhattan route previewed live, invalid targets in red with the reason). The
//! Wire tool draws wires from anywhere; Place puts the requested item at the click / drop
//! position. Overlays: selected wire segments, hovered port, rubber-band wire, selection box,
//! ghosts of moved / placed items, and the last error as a toast.
//!
//! Per-frame cost is a hover hit test over the scene's items (only those whose box contains the
//! pointer are examined further); nothing here touches the sim thread except through submitted
//! edits.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use egui::{
    Align2, Color32, FontId, Key, Mesh, Modifiers, PointerButton, Pos2, Rect, Stroke, StrokeKind,
    Ui, pos2, vec2,
};

use super::SharedEditor;
use super::actions::{
    Anchor, PlaceRequest, Selection, Tool, WireSeg, item_ports, net_of, plan_connect,
};
use super::route::{self, nearest_on};
use crate::app::{AppState, CanvasTool};
use crate::canvas::shapes::{Templates, append_oriented};
use crate::canvas::{CanvasResponse, Palette, Scene, ViewXf};
use crate::doc::{InstGeom, ItemRef, Placed};
use crate::geom::{GridPt, GridRect, Orient, atom_symbol, gp, pin_symbol};
use panda_netlist::PinDecl;

/// Pick tolerance for ports and wires (grid units).
const PICK: f32 = 0.45;
/// How long an error toast stays.
const TOAST: Duration = Duration::from_secs(6);

/// A gesture in progress.
#[derive(Clone, Debug, Default)]
enum Gesture {
    #[default]
    None,
    /// Box selection from a world point.
    Box { start: Pos2 },
    /// Moving the selection; drag started at a world point.
    Move { start: Pos2 },
    /// Drawing a wire from a grid point.
    Wire { start: GridPt, anchor: Anchor },
}

/// The editing canvas tool.
pub struct EditTool {
    ed: SharedEditor,
    gesture: Gesture,
    templates: Templates,
    /// Items this tool put into `AppState::highlights.items`.
    highlighted: HashSet<ItemRef>,
    /// Selection last exported to [`super::current_selection`].
    published: Selection,
    toast: Option<(String, Instant)>,
    /// Handle under which [`super::shared_editor`] finds `ed`.
    handle: u64,
}

impl EditTool {
    /// Tool over the shared editor.
    pub fn new(ed: SharedEditor) -> EditTool {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        EditTool {
            handle: NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            ed,
            gesture: Gesture::None,
            templates: Templates::default(),
            highlighted: HashSet::new(),
            published: Selection::default(),
            toast: None,
        }
    }
}

/// The port of the scene within [`PICK`] of world point `w`.
fn hit_port(scene: &Scene, w: Pos2) -> Option<super::actions::PortInfo> {
    for (p, r) in scene.items.iter().zip(&scene.rects).rev() {
        let inside = w.x >= r.min.x as f32 - PICK
            && w.x <= r.max.x as f32 + PICK
            && w.y >= r.min.y as f32 - PICK
            && w.y <= r.max.y as f32 + PICK;
        if !inside {
            continue;
        }
        if let Some(port) = item_ports(p)
            .into_iter()
            .find(|q| (q.pos.x as f32 - w.x).abs() <= PICK && (q.pos.y as f32 - w.y).abs() <= PICK)
        {
            return Some(port);
        }
    }
    None
}

/// The wire segment nearest to `w` within [`PICK`].
fn hit_wire(scene: &Scene, w: Pos2) -> Option<WireSeg> {
    let mut best: Option<(f32, WireSeg)> = None;
    for n in &scene.nets {
        let b = &n.bounds;
        if w.x < b.min.x as f32 - PICK
            || w.x > b.max.x as f32 + PICK
            || w.y < b.min.y as f32 - PICK
            || w.y > b.max.y as f32 + PICK
        {
            continue;
        }
        for &s in &n.segs {
            let d = route::distance(w.x, w.y, s);
            if d <= PICK && best.as_ref().is_none_or(|(bd, _)| d < *bd) {
                best = Some((d, WireSeg::new(n.id.clone(), s.0, s.1)));
            }
        }
    }
    best.map(|(_, s)| s)
}

/// Classify grid point `p` using the scene (cheap version of [`super::actions::anchor_at`]).
fn scene_anchor(app: &AppState, scene: &Scene, p: GridPt) -> Anchor {
    let Some(def) = app.doc.def(&scene.def) else {
        return Anchor::Free;
    };
    let w = pos2(p.x as f32, p.y as f32);
    if let Some(port) = hit_port(scene, w).filter(|q| q.pos == p) {
        let net = net_of(def, &port.ep).map(|n| n.id.clone());
        return Anchor::Port { port, net };
    }
    for n in &scene.nets {
        if n.segs.iter().any(|&s| route::on_segment(p, s)) {
            return Anchor::Wire { net: n.id.clone() };
        }
    }
    Anchor::Free
}

/// Symbol and size of what a place request would put down.
fn request_symbol(app: &AppState, req: &PlaceRequest) -> Option<crate::geom::Symbol> {
    match req {
        PlaceRequest::Atom(a) => Some(atom_symbol(a)),
        PlaceRequest::Def(d) => app.doc.package_symbol(d),
        PlaceRequest::Pin { dir, width } => Some(pin_symbol(&PinDecl {
            name: "pin".into(),
            dir: *dir,
            width: *width,
        })),
    }
}

impl EditTool {
    /// Draw `p` with its placement replaced by `g`, faded (ghost).
    fn ghost(
        &mut self,
        painter: &egui::Painter,
        xf: &ViewXf,
        pal: &Palette,
        ppp: f32,
        sym: &crate::geom::Symbol,
        g: InstGeom,
    ) {
        let z = xf.zoom;
        let r = GridRect::from_pos_size(g.pos, g.orient.size(sym.size));
        let sr = xf.rect_to_screen(&r);
        if z >= crate::canvas::LOD_BOXES {
            self.templates.prepare(z, ppp, pal);
            let t = self.templates.get(sym).clone();
            let mut mesh = Mesh::default();
            append_oriented(&mut mesh, &t, sym.size, z, g.orient, sr.min);
            for v in &mut mesh.vertices {
                v.color = v.color.gamma_multiply(0.55);
            }
            painter.add(egui::Shape::mesh(mesh));
        }
        painter.rect_stroke(
            sr.expand(2.0),
            2.0,
            Stroke::new(1.5, pal.accent.gamma_multiply(0.8)),
            StrokeKind::Outside,
        );
    }

    fn keyboard(&mut self, ui: &Ui, app: &mut AppState, hover: Option<GridPt>) {
        if ui.ctx().egui_wants_keyboard_input() {
            return;
        }
        let def = app.view.def.clone();
        let mut ed = self.ed.borrow_mut();
        let pressed = |m: Modifiers, k: Key| ui.ctx().input_mut(|i| i.consume_key(m, k));
        if pressed(Modifiers::NONE, Key::Escape) {
            if !matches!(self.gesture, Gesture::None) {
                self.gesture = Gesture::None;
            } else if ed.tool != Tool::Select {
                ed.tool = Tool::Select;
            } else {
                ed.selection.clear();
            }
        }
        if pressed(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z)
            || pressed(Modifiers::COMMAND, Key::Y)
        {
            let _ = ed.redo(app);
        }
        if pressed(Modifiers::COMMAND, Key::Z) {
            let _ = ed.undo(app);
        }
        if pressed(Modifiers::COMMAND, Key::A) {
            ed.select_all(&app.doc, &def);
        }
        if pressed(Modifiers::COMMAND, Key::C) {
            ed.copy(&app.doc, &def);
        }
        if pressed(Modifiers::COMMAND, Key::X) {
            let _ = ed.cut(app, &def);
        }
        if pressed(Modifiers::COMMAND, Key::V) {
            let _ = ed.paste(app, &def, hover);
        }
        if pressed(Modifiers::COMMAND, Key::D) {
            let _ = ed.duplicate(app, &def);
        }
        if pressed(Modifiers::NONE, Key::Delete) || pressed(Modifiers::NONE, Key::Backspace) {
            let _ = ed.delete_selection(app, &def);
        }
        if pressed(Modifiers::SHIFT, Key::R) {
            let _ = ed.rotate_selection(app, &def, false);
        }
        if pressed(Modifiers::NONE, Key::R) {
            let _ = ed.rotate_selection(app, &def, true);
        }
        if pressed(Modifiers::NONE, Key::F) {
            let _ = ed.flip_selection(app, &def);
        }
        if pressed(Modifiers::NONE, Key::W) {
            ed.tool = Tool::Wire;
        }
        if pressed(Modifiers::NONE, Key::S) {
            ed.tool = Tool::Select;
        }
    }
}

impl CanvasTool for EditTool {
    fn id(&self) -> &'static str {
        "edit"
    }

    fn claims_primary(&self, _app: &AppState) -> bool {
        self.ed.borrow().tool != Tool::Pan
    }

    fn canvas(&mut self, ui: &mut Ui, canvas: &CanvasResponse, scene: &Scene, app: &mut AppState) {
        super::announce_editor(ui.ctx(), self.handle, &self.ed);
        let def = app.view.def.clone();
        self.ed.borrow_mut().sync(&app.doc, &app.view.path, &def);
        if let Some(req) = super::take_place_request(ui.ctx()) {
            self.ed.borrow_mut().tool = Tool::Place(req);
            self.gesture = Gesture::None;
        }
        let xf = canvas.xf;
        let resp = &canvas.response;
        let pointer = ui.input(|i| i.pointer.latest_pos());
        let over = resp.contains_pointer();
        let hover_w = pointer.filter(|_| over).map(|p| xf.to_world(p));
        let hover_g = hover_w.map(|w| gp(w.x.round() as i32, w.y.round() as i32));
        if over || resp.has_focus() || matches!(self.gesture, Gesture::None) {
            self.keyboard(ui, app, hover_g);
        }
        let shift = ui.input(|i| i.modifiers.shift);
        let tool = self.ed.borrow().tool.clone();
        let painter = ui.painter_at(canvas.rect);
        let pal = Palette::for_dark_mode(ui.visuals().dark_mode);
        let ppp = ui.ctx().pixels_per_point();

        // ---- input ------------------------------------------------------------------
        match &tool {
            Tool::Pan => {}
            Tool::Place(req) => {
                let released = ui.input(|i| i.pointer.primary_released());
                if released && let Some(g) = hover_g {
                    let r = self.ed.borrow_mut().place(app, &def, req, g);
                    if r.is_ok() && !shift {
                        self.ed.borrow_mut().tool = Tool::Select;
                    }
                }
                if resp.secondary_clicked() {
                    self.ed.borrow_mut().tool = Tool::Select;
                }
            }
            Tool::Select | Tool::Wire => {
                if resp.drag_started_by(PointerButton::Primary)
                    && let Some(origin) = ui.input(|i| i.pointer.press_origin())
                {
                    let w0 = xf.to_world(origin);
                    let g0 = gp(w0.x.round() as i32, w0.y.round() as i32);
                    let mut ed = self.ed.borrow_mut();
                    self.gesture = if let Some(port) = hit_port(scene, w0) {
                        let start = port.pos;
                        Gesture::Wire {
                            start,
                            anchor: scene_anchor(app, scene, start),
                        }
                    } else if tool == Tool::Wire {
                        let start =
                            hit_wire(scene, w0).map_or(g0, |s| nearest_on(w0.x, w0.y, (s.a, s.b)));
                        Gesture::Wire {
                            start,
                            anchor: scene_anchor(app, scene, start),
                        }
                    } else if let Some(it) = scene.hit_item(w0).map(|p| p.item.clone()) {
                        if !ed.selection.items.contains(&it) {
                            if !shift {
                                ed.selection.clear();
                            }
                            ed.selection.items.insert(it);
                        }
                        Gesture::Move { start: w0 }
                    } else if let Some(s) = hit_wire(scene, w0) {
                        if ed.selection.wires.contains(&s) {
                            Gesture::Move { start: w0 }
                        } else {
                            let start = nearest_on(w0.x, w0.y, (s.a, s.b));
                            Gesture::Wire {
                                start,
                                anchor: Anchor::Wire { net: s.net },
                            }
                        }
                    } else {
                        Gesture::Box { start: w0 }
                    };
                }
                if resp.drag_stopped_by(PointerButton::Primary) {
                    let end_w = pointer.map(|p| xf.to_world(p));
                    let gesture = std::mem::take(&mut self.gesture);
                    if let Some(w) = end_w {
                        let g = gp(w.x.round() as i32, w.y.round() as i32);
                        let mut ed = self.ed.borrow_mut();
                        match gesture {
                            Gesture::Wire { start, anchor } => {
                                let end = scene_anchor(app, scene, g);
                                let pts = route::route(start, anchor.axis(), g, end.axis());
                                if pts.len() >= 2 {
                                    let _ = ed.draw_wire(app, &def, &pts);
                                }
                            }
                            Gesture::Move { start } => {
                                let d = w - start;
                                let delta = gp(d.x.round() as i32, d.y.round() as i32);
                                let _ = ed.move_selection(app, &def, delta);
                            }
                            Gesture::Box { start } => {
                                let r = GridRect {
                                    min: gp(
                                        start.x.min(w.x).floor() as i32,
                                        start.y.min(w.y).floor() as i32,
                                    ),
                                    max: gp(
                                        start.x.max(w.x).ceil() as i32,
                                        start.y.max(w.y).ceil() as i32,
                                    ),
                                };
                                ed.box_select(&app.doc, &def, r, shift);
                            }
                            Gesture::None => {}
                        }
                    }
                }
                if resp.clicked_by(PointerButton::Primary)
                    && tool == Tool::Select
                    && let Some(p) = resp.interact_pointer_pos()
                {
                    let w = xf.to_world(p);
                    let item = scene.hit_item(w).map(|p| p.item.clone());
                    let wire = if item.is_none() {
                        hit_wire(scene, w)
                    } else {
                        None
                    };
                    self.ed.borrow_mut().click_select(item, wire, shift);
                }
                if resp.secondary_clicked() {
                    self.gesture = Gesture::None;
                    if tool == Tool::Wire {
                        self.ed.borrow_mut().tool = Tool::Select;
                    }
                }
            }
        }

        // ---- overlays -----------------------------------------------------------------
        let ed = self.ed.borrow();
        let wire_w = (xf.zoom * 0.13).clamp(1.0, 3.5);
        for w in &ed.selection.wires {
            let r = Rect::from_two_pos(xf.grid_to_screen(w.a), xf.grid_to_screen(w.b));
            painter.rect_filled(
                r.expand(wire_w / 2.0 + 2.5),
                1.0,
                pal.accent.gamma_multiply(0.7),
            );
        }
        let gesture = self.gesture.clone();
        let sel_items = ed.selection.items.clone();
        let current_tool = ed.tool.clone();
        drop(ed);
        match gesture {
            Gesture::Box { start } => {
                if let Some(w) = hover_w.or(pointer.map(|p| xf.to_world(p))) {
                    let r = Rect::from_two_pos(xf.to_screen(start), xf.to_screen(w));
                    painter.rect_filled(r, 0.0, pal.accent.gamma_multiply(0.12));
                    painter.rect_stroke(r, 0.0, Stroke::new(1.0, pal.accent), StrokeKind::Inside);
                }
            }
            Gesture::Move { start } => {
                if let Some(p) = pointer {
                    let d = xf.to_world(p) - start;
                    let delta = gp(d.x.round() as i32, d.y.round() as i32);
                    let moved: Vec<Placed> = scene
                        .items
                        .iter()
                        .filter(|p| sel_items.contains(&p.item))
                        .cloned()
                        .collect();
                    for p in moved {
                        let g = InstGeom {
                            pos: p.geom.pos + delta,
                            orient: p.geom.orient,
                        };
                        self.ghost(&painter, &xf, &pal, ppp, &p.symbol, g);
                    }
                }
            }
            Gesture::Wire { start, anchor } => {
                if let Some(p) = pointer {
                    let w = xf.to_world(p);
                    let g = gp(w.x.round() as i32, w.y.round() as i32);
                    let end = scene_anchor(app, scene, g);
                    let pts = route::route(start, anchor.axis(), g, end.axis());
                    let check = app
                        .doc
                        .def(&def)
                        .map(|d| plan_connect(d, &[anchor.clone(), end.clone()]));
                    let (color, err) = match check {
                        Some(Err(e)) if pts.len() >= 2 => (pal.x, Some(e)),
                        _ => (pal.accent, None),
                    };
                    let screen: Vec<Pos2> = pts.iter().map(|&q| xf.grid_to_screen(q)).collect();
                    painter.add(egui::Shape::line(screen, Stroke::new(wire_w + 0.5, color)));
                    let r = (xf.zoom * 0.3).max(3.0);
                    painter.circle_stroke(xf.grid_to_screen(start), r, Stroke::new(1.5, color));
                    if !matches!(end, Anchor::Free) {
                        painter.circle_stroke(xf.grid_to_screen(g), r, Stroke::new(1.5, color));
                    }
                    if let Some(e) = err {
                        let at = xf.grid_to_screen(g) + vec2(10.0, -10.0);
                        let galley =
                            painter.layout_no_wrap(e, FontId::proportional(13.0), Color32::WHITE);
                        let mut box_ = Align2::LEFT_BOTTOM
                            .anchor_size(at, galley.size())
                            .expand(4.0);
                        // Keep the label inside the canvas.
                        let over = box_.right() - (canvas.rect.right() - 4.0);
                        if over > 0.0 {
                            box_ = box_
                                .translate(vec2(-over.min(box_.left() - canvas.rect.left()), 0.0));
                        }
                        if box_.top() < canvas.rect.top() {
                            box_ = box_.translate(vec2(0.0, canvas.rect.top() - box_.top() + 2.0));
                        }
                        painter.rect_filled(box_, 3.0, pal.x.gamma_multiply(0.9));
                        painter.galley(box_.min + vec2(4.0, 4.0), galley, Color32::WHITE);
                    }
                }
            }
            Gesture::None => {
                if let Tool::Place(req) = &current_tool {
                    if let (Some(g), Some(sym)) = (hover_g, request_symbol(app, req)) {
                        let pos = g - gp(sym.size.x / 2, sym.size.y / 2);
                        let ig = InstGeom {
                            pos,
                            orient: Orient::default(),
                        };
                        self.ghost(&painter, &xf, &pal, ppp, &sym, ig);
                    }
                } else if current_tool != Tool::Pan
                    && let Some(w) = hover_w
                    && let Some(port) = hit_port(scene, w)
                {
                    let r = (xf.zoom * 0.3).max(3.0);
                    painter.circle_stroke(
                        xf.grid_to_screen(port.pos),
                        r,
                        Stroke::new(1.5, pal.accent),
                    );
                }
            }
        }

        // ---- toast, highlights, selection export ---------------------------------------
        let ed = self.ed.borrow();
        match (&ed.message, &self.toast) {
            (Some(m), Some((t, _))) if m == t => {}
            (Some(m), _) => self.toast = Some((m.clone(), Instant::now())),
            (None, _) => self.toast = None,
        }
        if let Some((m, t0)) = &self.toast
            && t0.elapsed() < TOAST
        {
            let at = canvas.rect.left_bottom() + vec2(10.0, -10.0);
            let galley = painter.layout(
                m.clone(),
                FontId::proportional(13.0),
                Color32::WHITE,
                canvas.rect.width() - 40.0,
            );
            let box_ = Align2::LEFT_BOTTOM
                .anchor_size(at, galley.size())
                .expand(5.0);
            painter.rect_filled(box_, 4.0, pal.x.gamma_multiply(0.9));
            painter.galley(box_.min + vec2(5.0, 5.0), galley, Color32::WHITE);
            ui.ctx().request_repaint_after(Duration::from_millis(250));
        }
        if self.published != ed.selection {
            for it in self.highlighted.drain() {
                app.highlights.items.remove(&it);
            }
            for it in &ed.selection.items {
                app.highlights.items.insert(it.clone());
                self.highlighted.insert(it.clone());
            }
            self.published = ed.selection.clone();
            super::publish_selection(ui.ctx(), &ed.selection);
        }
        if !matches!(self.gesture, Gesture::None) || matches!(current_tool, Tool::Place(_)) {
            ui.ctx().request_repaint();
        }
    }
}
