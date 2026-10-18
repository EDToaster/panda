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
    Align2, FontId, Key, Mesh, Modifiers, PointerButton, Pos2, Rect, Stroke, StrokeKind, Ui, pos2,
    vec2,
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
    /// The quick-add box (`A`), if open.
    quick: Option<QuickAdd>,
    /// The shortcut sheet (`?`) is shown.
    sheet: bool,
    /// The single selected item, and the one selected before it (`C` wires prev → current).
    cur_item: Option<ItemRef>,
    prev_item: Option<ItemRef>,
    /// The item whose wires `N` cycles through.
    net_anchor: Option<ItemRef>,
    /// Frame whose keys were handled (egui may run a frame twice with the same input).
    keys_frame: u64,
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
            quick: None,
            sheet: false,
            cur_item: None,
            prev_item: None,
            net_anchor: None,
            keys_frame: u64::MAX,
        }
    }
}

/// State of the quick-add box.
struct QuickAdd {
    query: String,
    /// Highlighted row.
    row: usize,
    /// Where to place when nothing is selected (pointer or view centre at opening).
    fallback: GridPt,
    /// Frame it was opened in (the shortcut's own text is dropped during that frame).
    opened: u64,
}

/// The single selected item (no wires), if any.
fn single_item(sel: &Selection) -> Option<ItemRef> {
    (sel.items.len() == 1 && sel.wires.is_empty())
        .then(|| sel.items.iter().next().cloned())
        .flatten()
}

/// Pan the view so `item` is visible (centred if it is outside the canvas).
fn reveal(app: &mut AppState, rect: Rect, item: &ItemRef) {
    let def = app.view.def.clone();
    let Some(p) = app.doc.placed(&def).into_iter().find(|p| &p.item == item) else {
        return;
    };
    let r = p.rect();
    let xf = &mut app.canvas.xf;
    let sr = Rect::from_two_pos(xf.grid_to_screen(r.min), xf.grid_to_screen(r.max));
    if !rect.shrink(20.0).contains_rect(sr) {
        xf.origin += rect.center() - sr.center();
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

    fn keyboard(&mut self, ui: &Ui, app: &mut AppState, hover: Option<GridPt>, rect: Rect) {
        let frame = ui.ctx().cumulative_frame_nr();
        if ui.ctx().egui_wants_keyboard_input() || self.quick.is_some() || self.keys_frame == frame
        {
            return;
        }
        self.keys_frame = frame;
        let def = app.view.def.clone();
        let mut ed = self.ed.borrow_mut();
        let pressed = |m: Modifiers, k: Key| ui.ctx().input_mut(|i| i.consume_key(m, k));
        // ? toggles the shortcut sheet; Esc closes it first.
        if pressed(Modifiers::SHIFT, Key::Slash)
            || pressed(Modifiers::NONE, Key::Questionmark)
            || pressed(Modifiers::SHIFT, Key::Questionmark)
        {
            self.sheet = !self.sheet;
        }
        if self.sheet && pressed(Modifiers::NONE, Key::Escape) {
            self.sheet = false;
        }
        // A: quick add.
        if pressed(Modifiers::NONE, Key::A) {
            let center = app.canvas.xf.to_world(rect.center());
            self.quick = Some(QuickAdd {
                query: String::new(),
                row: 0,
                fallback: hover.unwrap_or(gp(center.x.round() as i32, center.y.round() as i32)),
                opened: frame,
            });
            // Swallow the "a" typed with the shortcut so it does not land in the search box.
            ui.ctx().input_mut(|i| {
                i.events
                    .retain(|e| !matches!(e, egui::Event::Text(t) if t == "a" || t == "A"))
            });
            self.sheet = false;
            ui.ctx().request_repaint();
            return;
        }
        // Tab / Shift+Tab: cycle the selection through chips and pins.
        let back = pressed(Modifiers::SHIFT, Key::Tab);
        if back || pressed(Modifiers::NONE, Key::Tab) {
            let cur = single_item(&ed.selection);
            if let Some(it) = super::keys::next_item(&app.doc, &def, cur.as_ref(), back) {
                ed.selection.clear();
                ed.selection.items.insert(it.clone());
                reveal(app, rect, &it);
            }
        }
        // Arrows: nudge the selection (Shift: 5 steps), or pan with nothing selected.
        for (k, d) in [
            (Key::ArrowLeft, (-1, 0)),
            (Key::ArrowRight, (1, 0)),
            (Key::ArrowUp, (0, -1)),
            (Key::ArrowDown, (0, 1)),
        ] {
            let big = pressed(Modifiers::SHIFT, k);
            if !big && !pressed(Modifiers::NONE, k) {
                continue;
            }
            let n = if big { 5 } else { 1 };
            if ed.selection.is_empty() {
                app.canvas.xf.origin -= vec2(d.0 as f32, d.1 as f32) * 60.0 * n as f32;
            } else {
                let _ = ed.move_selection(app, &def, gp(d.0 * n, d.1 * n));
            }
        }
        // View: fit, zoom.
        if pressed(Modifiers::NONE, Key::Home) || pressed(Modifiers::NONE, Key::Num0) {
            app.canvas.request_fit();
        }
        if pressed(Modifiers::NONE, Key::Plus)
            || pressed(Modifiers::NONE, Key::Equals)
            || pressed(Modifiers::SHIFT, Key::Equals)
        {
            app.canvas.xf.zoom_at(rect.center(), 1.25);
        }
        if pressed(Modifiers::NONE, Key::Minus) {
            app.canvas.xf.zoom_at(rect.center(), 0.8);
        }
        // C: wire the two selected items (left to right), or the previous item to this one.
        if pressed(Modifiers::NONE, Key::C) {
            let pair = if ed.selection.wires.is_empty() && ed.selection.items.len() == 2 {
                let placed = app.doc.placed(&def);
                let mut two: Vec<(i32, ItemRef)> = ed
                    .selection
                    .items
                    .iter()
                    .map(|it| {
                        let x = placed
                            .iter()
                            .find(|p| &p.item == it)
                            .map_or(0, |p| p.rect().min.x);
                        (x, it.clone())
                    })
                    .collect();
                two.sort();
                Some((two[0].1.clone(), two[1].1.clone()))
            } else {
                match (&self.prev_item, single_item(&ed.selection)) {
                    (Some(p), Some(c)) if *p != c => Some((p.clone(), c)),
                    _ => None,
                }
            };
            match pair {
                Some((from, to)) => {
                    match super::keys::plan_quick_wire(&app.doc, &def, &from, &to) {
                        Ok((a, b)) => {
                            let pts = route::route(
                                a.pos,
                                Some(route::Axis::of_side(a.side)),
                                b.pos,
                                Some(route::Axis::of_side(b.side)),
                            );
                            let sel = ed.selection.clone();
                            if ed.draw_wire(app, &def, &pts).is_ok() {
                                ed.selection = sel;
                            }
                        }
                        Err(e) => ed.message = Some(e),
                    }
                }
                None => {
                    ed.message = Some(
                        "C connects two selected items, or the previously selected item to \
                         this one"
                            .into(),
                    );
                }
            }
        }
        // N: select the wires on the selected item (repeat: the next one).
        if pressed(Modifiers::NONE, Key::N) {
            let anchor = single_item(&ed.selection).or_else(|| self.net_anchor.clone());
            if let Some(it) = anchor
                && let Some(next) =
                    super::keys::next_net_of(&app.doc, &def, &it, ed.selection.single_net())
            {
                let segs: Vec<WireSeg> = app
                    .doc
                    .editor(&def)
                    .and_then(|e| e.wires.get(&next))
                    .map(|g| {
                        g.segments()
                            .map(|(a, b)| WireSeg::new(&next, a, b))
                            .collect()
                    })
                    .unwrap_or_default();
                if !segs.is_empty() {
                    ed.selection.clear();
                    ed.selection.wires.extend(segs);
                    self.net_anchor = Some(it);
                }
            }
        }
        // Enter: open the selected package.
        if pressed(Modifiers::NONE, Key::Enter)
            && let Some(id) = ed.selection.single_instance().map(str::to_string)
        {
            drop(ed);
            super::menus::enter_package(app, &id);
            return;
        }
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
        // F2: rename the selected net (or instance) in the inspector.
        if pressed(Modifiers::NONE, Key::F2)
            && (ed.selection.single_net().is_some() || ed.selection.single_instance().is_some())
        {
            ed.focus_name = true;
            ui.ctx().request_repaint();
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
        let cur = single_item(&self.ed.borrow().selection);
        if cur != self.cur_item {
            if self.cur_item.is_some() {
                self.prev_item = self.cur_item.take();
            }
            self.cur_item = cur;
        }
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
            self.keyboard(ui, app, hover_g, canvas.rect);
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
        let wire_w = (xf.zoom * 0.2).clamp(1.5, 6.0);
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
                        let galley = painter.layout_no_wrap(
                            e,
                            FontId::proportional(13.0),
                            crate::theme::NAVY,
                        );
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
                        painter.galley(box_.min + vec2(4.0, 4.0), galley, crate::theme::NAVY);
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
            let at = canvas.rect.left_bottom() + vec2(10.0, -HINT_BAR - 8.0);
            let galley = painter.layout(
                m.clone(),
                FontId::proportional(13.0),
                crate::theme::NAVY,
                canvas.rect.width() - 40.0,
            );
            let box_ = Align2::LEFT_BOTTOM
                .anchor_size(at, galley.size())
                .expand(5.0);
            painter.rect_filled(box_, 4.0, pal.x.gamma_multiply(0.9));
            painter.galley(box_.min + vec2(5.0, 5.0), galley, crate::theme::NAVY);
            ui.ctx().request_repaint_after(Duration::from_millis(250));
        }
        // ---- keyboard: quick add, hint bar, shortcut sheet ------------------------------
        let ctx = super::keys::hint_ctx(
            app,
            &ed.selection,
            &ed.tool,
            self.quick.is_some(),
            self.prev_item.as_ref(),
        );
        drop(ed);
        draw_hint_bar(&painter, canvas.rect, &super::keys::hints(&ctx));
        self.quick_ui(ui, app, canvas.rect);
        if self.sheet {
            sheet_ui(ui, canvas.rect, &mut self.sheet);
        }
        let ed = self.ed.borrow();
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

/// Height of the hint bar along the bottom of the canvas.
const HINT_BAR: f32 = 26.0;

/// The hint bar: key caps and what they do, for the current context.
fn draw_hint_bar(painter: &egui::Painter, rect: Rect, hints: &[(&str, &str)]) {
    use crate::theme;
    let bar = Rect::from_min_max(pos2(rect.left(), rect.bottom() - HINT_BAR), rect.max);
    painter.rect_filled(bar, 0.0, theme::SAND.gamma_multiply(0.92));
    painter.line_segment(
        [bar.left_top(), bar.right_top()],
        Stroke::new(1.0, theme::DUNE),
    );
    let y = bar.center().y;
    let cap = |k: &str, what: &str, x: f32, right: f32, draw: bool| -> Option<f32> {
        let kg = painter.layout_no_wrap(k.to_string(), FontId::monospace(11.5), theme::EGGSHELL);
        let wg = painter.layout_no_wrap(what.to_string(), FontId::proportional(12.0), theme::SLATE);
        let w = kg.size().x + 10.0 + 5.0 + wg.size().x;
        if x + w > right {
            return None;
        }
        if draw {
            let r = Rect::from_min_size(
                pos2(x, y - kg.size().y / 2.0 - 2.0),
                kg.size() + vec2(10.0, 4.0),
            );
            painter.rect_filled(r, 4.0, theme::NAVY);
            painter.galley(r.min + vec2(5.0, 2.0), kg, theme::EGGSHELL);
            painter.galley(
                pos2(r.right() + 5.0, y - wg.size().y / 2.0),
                wg,
                theme::SLATE,
            );
        }
        Some(w)
    };
    // "? all shortcuts" sits at the right end; the others fill from the left while they fit.
    let mut right = bar.right() - 8.0;
    let (rest, last) = match hints.split_last() {
        Some((l, r)) if l.0 == "?" => (r, Some(*l)),
        _ => (hints, None),
    };
    if let Some((k, what)) = last
        && let Some(w) = cap(k, what, f32::NEG_INFINITY, f32::INFINITY, false)
    {
        let _ = cap(k, what, right - w, f32::INFINITY, true);
        right -= w + 14.0;
    }
    let mut x = bar.left() + 8.0;
    for (k, what) in rest {
        match cap(k, what, x, right, true) {
            Some(w) => x += w + 14.0,
            None => break,
        }
    }
}

/// The shortcut sheet over the canvas (closes on its button or `?` / Esc).
fn sheet_ui(ui: &Ui, rect: Rect, open: &mut bool) {
    use crate::theme;
    egui::Area::new(egui::Id::new("panda-shortcut-sheet"))
        .order(egui::Order::Foreground)
        .pivot(Align2::CENTER_CENTER)
        .fixed_pos(rect.center())
        .show(ui.ctx(), |ui| {
            egui::Frame::popup(ui.style())
                .fill(theme::EGGSHELL)
                .inner_margin(16.0)
                .corner_radius(10.0)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading("Keyboard shortcuts");
                        ui.add_space(12.0);
                        if ui.small_button("close (Esc)").clicked() {
                            *open = false;
                        }
                    });
                    ui.add_space(6.0);
                    let groups = super::keys::SHEET;
                    let half = groups.len().div_ceil(2);
                    ui.horizontal_top(|ui| {
                        for col in [&groups[..half], &groups[half..]] {
                            ui.vertical(|ui| {
                                ui.set_width(380.0);
                                egui::Grid::new(("sheet", col[0].0))
                                    .num_columns(2)
                                    .spacing([12.0, 3.0])
                                    .show(ui, |ui| {
                                        for (gi, (title, rows)) in col.iter().enumerate() {
                                            if gi > 0 {
                                                ui.label("");
                                                ui.end_row();
                                            }
                                            ui.label(
                                                egui::RichText::new(*title)
                                                    .strong()
                                                    .color(theme::MAROON),
                                            );
                                            ui.end_row();
                                            for (k, what) in *rows {
                                                ui.label(
                                                    egui::RichText::new(*k)
                                                        .monospace()
                                                        .background_color(theme::NAVY)
                                                        .color(theme::EGGSHELL),
                                                );
                                                ui.label(*what);
                                                ui.end_row();
                                            }
                                        }
                                    });
                            });
                        }
                    });
                });
        });
}

impl EditTool {
    /// The quick-add box: type to filter, Up/Down to choose, Enter to place, Esc to close.
    fn quick_ui(&mut self, ui: &Ui, app: &mut AppState, rect: Rect) {
        let Some(q) = self.quick.as_mut() else {
            return;
        };
        // The "a" typed by the shortcut may arrive a frame or two after the key: drop it.
        if q.query.is_empty() && ui.ctx().cumulative_frame_nr() <= q.opened + 3 {
            ui.ctx().input_mut(|i| {
                i.events
                    .retain(|e| !matches!(e, egui::Event::Text(t) if t == "a" || t == "A"))
            });
        }
        let cands = super::keys::quick_candidates(app);
        let matches = super::keys::quick_matches(&cands, &q.query);
        let (up, down, enter, esc) = ui.ctx().input_mut(|i| {
            (
                i.consume_key(Modifiers::NONE, Key::ArrowUp),
                i.consume_key(Modifiers::NONE, Key::ArrowDown)
                    || i.consume_key(Modifiers::NONE, Key::Tab),
                i.consume_key(Modifiers::NONE, Key::Enter),
                i.consume_key(Modifiers::NONE, Key::Escape),
            )
        });
        let shown = matches.len().min(8);
        if up && q.row > 0 {
            q.row -= 1;
        }
        if down && q.row + 1 < shown {
            q.row += 1;
        }
        q.row = q.row.min(shown.saturating_sub(1));
        let mut choose = enter.then_some(q.row);
        let before = q.query.clone();
        egui::Area::new(egui::Id::new("panda-quick-add"))
            .order(egui::Order::Foreground)
            .pivot(Align2::CENTER_TOP)
            .fixed_pos(rect.center_top() + vec2(0.0, 40.0))
            .show(ui.ctx(), |ui| {
                egui::Frame::popup(ui.style())
                    .fill(crate::theme::EGGSHELL)
                    .inner_margin(8.0)
                    .corner_radius(10.0)
                    .show(ui, |ui| {
                        ui.set_width(300.0);
                        let r = ui.add(
                            egui::TextEdit::singleline(&mut q.query)
                                .hint_text("add: type to search")
                                .desired_width(f32::INFINITY),
                        );
                        r.request_focus();
                        ui.add_space(4.0);
                        if matches.is_empty() {
                            ui.weak("no match");
                        }
                        for (row, &i) in matches.iter().take(8).enumerate() {
                            if ui
                                .add_sized(
                                    [ui.available_width(), 20.0],
                                    egui::Button::selectable(row == q.row, cands[i].0.as_str()),
                                )
                                .clicked()
                            {
                                choose = Some(row);
                            }
                        }
                    });
            });
        if q.query != before {
            q.row = 0;
        }
        if esc {
            self.quick = None;
            return;
        }
        let Some(row) = choose else {
            ui.ctx().request_repaint();
            return;
        };
        let fallback = q.fallback;
        self.quick = None;
        let Some(&i) = matches.get(row) else {
            return;
        };
        let req = cands[i].1.clone();
        let Some(sym) = request_symbol(app, &req) else {
            return;
        };
        let def = app.view.def.clone();
        let placed = app.doc.placed(&def);
        let mut ed = self.ed.borrow_mut();
        let at = super::keys::place_spot(&placed, &ed.selection, sym.size, fallback);
        if let Ok(it) = ed.place(app, &def, &req, at) {
            ed.tool = Tool::Select;
            drop(ed);
            reveal(app, rect, &it);
        }
        ui.ctx().request_repaint();
    }
}
