//! The sim UX canvas tool: clickable top-level inputs, live value labels on hover, the
//! "Add to waveform" context menu, and centring the view on jump targets.
//!
//! * Click a top-level input pin (top view): a 1-bit input toggles (`1 → 0`, else `→ 1`); a
//!   multi-bit input opens a small editor (decimal, `0x…`, `0b…` with `x`/`z`, negative
//!   decimal). The command is sent without waiting; [`InputModel`] bridges the publication
//!   delay so fast double clicks toggle twice.
//!   While paused, a sent value applies at the next step; until then the pin carries a badge
//!   with the pending value.
//! * Hover a wire or pin: its name and live value (hex and decimal for buses).
//! * The canvas context menu (owned by [`crate::edit::menus`], which also has "Add to
//!   waveform") gets a "Set input value…" entry for top-level inputs.

use std::cell::RefCell;

use egui::{Color32, FontId, Id, Key, Order, Pos2, Rect, Ui, pos2, vec2};
use panda_netlist::{Dir, Endpoint, PinDecl, Value};

use crate::app::{AppState, CanvasTool};
use crate::canvas::{CanvasResponse, Palette, Scene};
use crate::doc::ItemRef;
use crate::edit::menus::{ContextEntry, Target, add_context_entry, context_menu_open};
use crate::simhost::Packed;

use super::SharedRef;
use super::control::{InputModel, parse_input};
use super::locate::Mark;
use super::wave_view::{Radix, format_value};

/// What the pointer is over.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Probe {
    /// A net of the viewed def (by id).
    Net(String),
    /// A pin of the viewed def (by name).
    Pin(String),
}

/// Hierarchical name for the waveform viewer of a probe in the view at `path`. A net that is
/// wired to a pin of the same width is named after the pin (`"PC"` rather than `"n17"`).
pub fn wave_name(app: &AppState, path: &str, probe: &Probe) -> String {
    let id = match probe {
        Probe::Pin(p) => p.clone(),
        Probe::Net(n) => app
            .doc
            .def(&app.view.def)
            .and_then(|d| {
                let net = d.nets.iter().find(|x| &x.id == n)?;
                net.ends.iter().find_map(|e| match e {
                    Endpoint::Pin(p)
                        if d.pins.iter().any(|q| &q.name == p && q.width == net.width) =>
                    {
                        Some(p.clone())
                    }
                    _ => None,
                })
            })
            .unwrap_or_else(|| n.clone()),
    };
    if path.is_empty() {
        id
    } else {
        format!("{path}/{id}")
    }
}

/// Live value text: `0`/`1`/`x`/`z`, or `0x2a (42)` for buses.
pub fn value_text(v: &Value) -> String {
    let bits: Vec<_> = v.bits().collect();
    let p = Packed::from_bits(&bits);
    let w = v.width().min(64);
    if w == 1 {
        return format_value(p, 1, Radix::Hex);
    }
    let hex = format_value(p, w, Radix::Hex);
    let dec = format_value(p, w, Radix::Dec);
    if dec.starts_with(['x', 'z', '?']) {
        format!("0x{hex}")
    } else {
        format!("0x{hex} ({dec})")
    }
}

/// Text of the hover label: `"Q = 0x5 (5)"`, `"EN = 1   · click to toggle"`; `input_width` is
/// set for a clickable top-level input, `pending` is a value sent to it that the simulation has
/// not applied yet (inputs take effect at the next tick / step, also while paused).
pub fn hover_text(
    name: &str,
    value: Option<&Value>,
    input_width: Option<u32>,
    pending: Option<&Value>,
) -> String {
    let mut text = match value {
        Some(v) => format!("{name} = {}", value_text(v)),
        None => format!("{name}  (no live value)"),
    };
    if let Some(p) = pending {
        text.push_str(&format!(" → {} at the next step", value_text(p)));
    }
    match input_width {
        Some(1) => text.push_str("   · click to toggle"),
        Some(_) => text.push_str("   · click to edit"),
        None => {}
    }
    text
}

/// A top-level input pin of the viewed def (only in the top view).
fn top_input<'a>(app: &'a AppState, name: &str) -> Option<&'a PinDecl> {
    if !app.view.path.is_empty() {
        return None;
    }
    app.doc
        .def(&app.view.def)?
        .pins
        .iter()
        .find(|p| p.name == name && p.dir == Dir::In)
}

struct InputEdit {
    pin: String,
    width: u32,
    text: String,
    pos: Pos2,
    error: Option<String>,
    focus: bool,
}

thread_local! {
    /// Input pins whose editor the context-menu entry asked for (the entry's `run` is a plain
    /// `fn` without access to the tool; UI thread only).
    static EDIT_REQUESTS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// The "Set input value…" entry of the canvas context menu (owned by the editing module).
fn set_value_entry() -> ContextEntry {
    ContextEntry {
        key: "simux.set_input",
        label: "Set input value…".into(),
        applies: |app, t| matches!(t, Target::Pin(p) if top_input(app, p).is_some()),
        run: |_, t| {
            if let Target::Pin(p) = t {
                EDIT_REQUESTS.with(|r| r.borrow_mut().push(p.clone()));
            }
        },
    }
}

/// The canvas tool.
pub struct SimTool {
    shared: SharedRef,
    inputs: InputModel,
    edit: Option<InputEdit>,
    /// Screen position of the last right click (where a requested editor opens).
    menu_pos: Option<Pos2>,
    entry_added: bool,
    design_gen: u64,
}

impl SimTool {
    /// A tool sharing `shared` with the module's panels.
    pub fn new(shared: SharedRef) -> SimTool {
        SimTool {
            shared,
            inputs: InputModel::default(),
            edit: None,
            menu_pos: None,
            entry_added: false,
            design_gen: 0,
        }
    }

    fn live_pin(app: &AppState, pin: &str) -> Option<Value> {
        let l = app.live.values()?;
        l.map.pin_value(pin, l.values)
    }

    fn live_net(app: &AppState, net: &str) -> Option<Value> {
        let l = app.live.values()?;
        l.map.net_value(net, l.values)
    }

    fn probe_at(scene: &Scene, w: Pos2, zoom: f32) -> Option<Probe> {
        if let Some(p) = scene.hit_item(w)
            && let ItemRef::Pin(name) = &p.item
        {
            return Some(Probe::Pin(name.clone()));
        }
        let tol = (6.0 / zoom).clamp(0.15, 2.0);
        scene.hit_net(w, tol).map(|n| Probe::Net(n.id.clone()))
    }

    /// Click on an input pin: toggle or open the editor.
    fn click_input(&mut self, app: &mut AppState, pin: &str, at: Pos2) {
        let Some(decl) = top_input(app, pin) else {
            return;
        };
        let width = decl.width;
        let live = Self::live_pin(app, pin);
        if width == 1 {
            let cmd = self.inputs.toggle(pin, live.as_ref());
            app.sim.send(cmd);
        } else {
            self.open_editor(app, pin, width, at);
        }
    }

    /// Open the value editor of input `pin` at screen position `at`.
    fn open_editor(&mut self, app: &AppState, pin: &str, width: u32, at: Pos2) {
        let live = Self::live_pin(app, pin);
        let cur = self.inputs.current(pin, live.as_ref());
        self.edit = Some(InputEdit {
            pin: pin.to_string(),
            width,
            text: cur
                .and_then(|v| v.to_u64())
                .map_or(String::new(), |v| format!("0x{v:x}")),
            pos: at,
            error: None,
            focus: true,
        });
    }

    fn apply_focus(&self, app: &mut AppState, canvas: &CanvasResponse, scene: &Scene) {
        let mut sh = self.shared.borrow_mut();
        let Some(f) = &sh.focus else { return };
        if scene.def != f.def || app.view.def != f.def {
            return;
        }
        let bounds = match &f.mark {
            Mark::Net(id) => scene.net_index.get(id).map(|&i| scene.nets[i].bounds),
            Mark::Item(r) => scene.item(r).map(|p| p.rect()),
            Mark::None => None,
        };
        sh.focus = None;
        let Some(b) = bounds else { return };
        let world = Rect::from_min_max(
            pos2(b.min.x as f32, b.min.y as f32),
            pos2(b.max.x as f32, b.max.y as f32),
        )
        .expand(4.0);
        let xf = &mut app.canvas.xf;
        xf.fit(world, canvas.rect, 24.0);
        xf.zoom = xf.zoom.clamp(4.0, 24.0);
        xf.origin = canvas.rect.center() - world.center().to_vec2() * xf.zoom;
    }

    /// The hover label: name and live value next to the pointer (a non-interactive tooltip
    /// layer, so it is also visible to accessibility / tests).
    fn hover_label(&self, ui: &Ui, app: &AppState, canvas: &CanvasResponse, probe: &Probe) {
        let Some(p) = canvas.response.hover_pos() else {
            return;
        };
        let (name, value) = match probe {
            Probe::Net(n) => (n.as_str(), Self::live_net(app, n)),
            Probe::Pin(n) => (n.as_str(), Self::live_pin(app, n)),
        };
        let (width, pending) = match probe {
            Probe::Pin(n) => (
                top_input(app, n).map(|d| d.width),
                self.inputs.pending(n, value.as_ref()),
            ),
            Probe::Net(_) => (None, None),
        };
        let text = hover_text(name, value.as_ref(), width, pending);
        let pal = Palette::for_dark_mode(ui.visuals().dark_mode);
        let font = FontId::monospace(12.0);
        // Keep the label inside the canvas (flip to the other side of the pointer at the edges).
        let size = ui
            .painter()
            .layout_no_wrap(text.clone(), font.clone(), pal.text)
            .size()
            + vec2(12.0, 10.0);
        let mut at = p + vec2(14.0, 14.0);
        if at.x + size.x > canvas.rect.right() {
            at.x -= size.x + 28.0;
        }
        if at.y + size.y > canvas.rect.bottom() {
            at.y -= size.y + 28.0;
        }
        egui::Area::new(Id::new("simux_hover_value"))
            .order(Order::Tooltip)
            .interactable(false)
            .fixed_pos(at)
            .show(ui.ctx(), |ui| {
                egui::Frame::new()
                    .fill(pal.fill.gamma_multiply(0.95))
                    .stroke(egui::Stroke::new(1.0, pal.accent.gamma_multiply(0.7)))
                    .corner_radius(3.0)
                    .inner_margin(egui::Margin::symmetric(6, 5))
                    .show(ui, |ui| {
                        ui.add(
                            egui::Label::new(egui::RichText::new(text).font(font).color(pal.text))
                                .extend(),
                        );
                    });
            });
    }

    /// Mark inputs whose sent value is not applied yet (paused simulation): a badge with the
    /// pending value next to the pin.
    fn paint_pending(&self, ui: &Ui, app: &AppState, canvas: &CanvasResponse, scene: &Scene) {
        if self.inputs.is_empty() || !app.view.path.is_empty() {
            return;
        }
        let pal = Palette::for_dark_mode(ui.visuals().dark_mode);
        let painter = ui.painter_at(canvas.rect);
        let font = FontId::monospace(11.0);
        for (pin, v) in self.inputs.sent() {
            let live = Self::live_pin(app, pin);
            if self.inputs.pending(pin, live.as_ref()).is_none() {
                continue;
            }
            let Some(p) = scene.item(&ItemRef::Pin(pin.to_string())) else {
                continue;
            };
            let r = canvas.xf.rect_to_screen(&p.rect());
            if !canvas.rect.intersects(r) {
                continue;
            }
            let g = painter.layout_no_wrap(format!("→{}", value_text(v)), font.clone(), pal.fill);
            let badge = Rect::from_min_size(r.left_top() - vec2(0.0, g.size().y + 6.0), g.size())
                .expand2(vec2(4.0, 2.0));
            painter.rect_filled(badge, 3.0, pal.accent);
            painter.galley(badge.min + vec2(4.0, 2.0), g, pal.fill);
        }
    }

    fn show_editor(&mut self, ui: &mut Ui, app: &mut AppState) {
        let Some(e) = &mut self.edit else { return };
        let mut submit = false;
        let mut cancel = false;
        egui::Area::new(Id::new("simux_input_editor"))
            .order(Order::Foreground)
            .fixed_pos(e.pos + vec2(8.0, 8.0))
            .show(ui.ctx(), |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.label(format!("Set input {} [{}]", e.pin, e.width));
                    let r = ui.add(
                        egui::TextEdit::singleline(&mut e.text)
                            .desired_width(140.0)
                            .hint_text("42, 0x2a, 0b1010, -1"),
                    );
                    if e.focus {
                        r.request_focus();
                        e.focus = false;
                    }
                    if r.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
                        submit = true;
                    }
                    if let Some(err) = &e.error {
                        ui.colored_label(Color32::from_rgb(240, 80, 80), err);
                    }
                    ui.horizontal(|ui| {
                        submit |= ui.button("Set").clicked();
                        cancel |= ui.button("Cancel").clicked();
                    });
                });
            });
        if ui.input(|i| i.key_pressed(Key::Escape)) {
            cancel = true;
        }
        if submit {
            match parse_input(e.width, &e.text) {
                Ok(v) => {
                    let cmd = self.inputs.set(&e.pin, v);
                    app.sim.send(cmd);
                    self.edit = None;
                }
                Err(msg) => {
                    e.error = Some(msg);
                    e.focus = true;
                }
            }
        } else if cancel {
            self.edit = None;
        }
    }
}

impl CanvasTool for SimTool {
    fn id(&self) -> &'static str {
        "sim_tool"
    }

    fn canvas(&mut self, ui: &mut Ui, canvas: &CanvasResponse, scene: &Scene, app: &mut AppState) {
        if app.live.design_gen != self.design_gen {
            self.design_gen = app.live.design_gen;
            self.edit = None;
        }
        if !self.entry_added {
            add_context_entry(ui.ctx(), set_value_entry());
            self.entry_added = true;
        }
        let requested = EDIT_REQUESTS.with(|r| r.borrow_mut().pop());
        if let Some(pin) = requested
            && let Some(width) = top_input(app, &pin).map(|d| d.width)
        {
            let at = self.menu_pos.unwrap_or(canvas.rect.center());
            self.open_editor(app, &pin, width, at);
        }
        // Forget sent input values the snapshot confirms.
        if !self.inputs.is_empty()
            && app.view.path.is_empty()
            && let Some(def) = app.doc.def(&app.view.def)
        {
            for p in def.pins.iter().filter(|p| p.dir == Dir::In) {
                let live = Self::live_pin(app, &p.name);
                self.inputs.confirm(&p.name, live.as_ref());
            }
        }
        self.apply_focus(app, canvas, scene);
        let xf = canvas.xf;
        let probe = canvas
            .hover_world
            .and_then(|w| Self::probe_at(scene, w, xf.zoom));
        let r = &canvas.response;
        if r.clicked()
            && let Some(p) = r.interact_pointer_pos()
            && let Some(Probe::Pin(pin)) = Self::probe_at(scene, xf.to_world(p), xf.zoom)
        {
            self.click_input(app, &pin, p);
        }
        if r.secondary_clicked() {
            self.menu_pos = r.interact_pointer_pos();
        }
        if self.edit.is_none()
            && !context_menu_open(ui.ctx())
            && let Some(pr) = &probe
        {
            self.hover_label(ui, app, canvas, pr);
        }
        self.paint_pending(ui, app, canvas, scene);
        self.show_editor(ui, app);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_texts() {
        assert_eq!(value_text(&Value::from_bool(true)), "1");
        assert_eq!(value_text(&Value::from_u64(8, 42)), "0x2a (42)");
        assert_eq!(value_text(&Value::xs(8)), "0xxx");
        assert_eq!(value_text(&Value::xs(1)), "x");
        assert_eq!(
            hover_text("EN", Some(&Value::from_bool(true)), Some(1), None),
            "EN = 1   · click to toggle"
        );
        assert_eq!(
            hover_text("WD", Some(&Value::from_u64(8, 90)), Some(8), None),
            "WD = 0x5a (90)   · click to edit"
        );
        assert_eq!(hover_text("n3", None, None, None), "n3  (no live value)");
        assert_eq!(
            hover_text(
                "EN",
                Some(&Value::from_bool(false)),
                Some(1),
                Some(&Value::from_bool(true))
            ),
            "EN = 0 → 1 at the next step   · click to toggle"
        );
    }
}
