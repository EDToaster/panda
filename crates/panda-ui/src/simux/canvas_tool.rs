//! The sim UX canvas tool: clickable top-level inputs, live value labels on hover, the
//! "Add to waveform" context menu, and centring the view on jump targets.
//!
//! * Click a top-level input pin (top view): a 1-bit input toggles (`1 → 0`, else `→ 1`); a
//!   multi-bit input opens a small editor (decimal, `0x…`, `0b…` with `x`/`z`, negative
//!   decimal). The command is sent without waiting; [`InputModel`] bridges the publication
//!   delay so fast double clicks toggle twice.
//! * Hover a wire or pin: its name and live value (hex and decimal for buses).
//! * Right-click a wire or pin: add it to / remove it from the waveform viewer, set an input.

use egui::{Color32, FontId, Id, Key, Order, Pos2, Rect, Ui, pos2, vec2};
use panda_netlist::{Dir, Endpoint, PinDecl, Value};

use crate::app::{AppState, CanvasTool};
use crate::canvas::{CanvasResponse, Palette, Scene};
use crate::doc::ItemRef;
use crate::simhost::Packed;

use super::control::{InputModel, parse_input};
use super::locate::Mark;
use super::wave_view::{Radix, format_value};
use super::waveform::{add_signal, remove_signal, signal_names};
use super::SharedRef;

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

struct Menu {
    pos: Pos2,
    probe: Probe,
}

/// The canvas tool.
pub struct SimTool {
    shared: SharedRef,
    inputs: InputModel,
    edit: Option<InputEdit>,
    menu: Option<Menu>,
    design_gen: u64,
}

impl SimTool {
    /// A tool sharing `shared` with the module's panels.
    pub fn new(shared: SharedRef) -> SimTool {
        SimTool {
            shared,
            inputs: InputModel::default(),
            edit: None,
            menu: None,
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

    fn hover_label(&self, ui: &Ui, app: &AppState, canvas: &CanvasResponse, probe: &Probe) {
        let Some(p) = canvas.response.hover_pos() else {
            return;
        };
        let (name, value) = match probe {
            Probe::Net(n) => (n.clone(), Self::live_net(app, n)),
            Probe::Pin(n) => (n.clone(), Self::live_pin(app, n)),
        };
        let mut text = match &value {
            Some(v) => format!("{name} = {}", value_text(v)),
            None => format!("{name}  (no live value)"),
        };
        if let Probe::Pin(n) = probe
            && top_input(app, n).is_some()
        {
            text.push_str("   · click to set");
        }
        let pal = Palette::for_dark_mode(ui.visuals().dark_mode);
        let painter = ui.painter_at(canvas.rect);
        let g = painter.layout_no_wrap(text, FontId::monospace(12.0), pal.text);
        let mut r = Rect::from_min_size(p + vec2(14.0, 14.0), g.size()).expand(4.0);
        if r.right() > canvas.rect.right() {
            r = r.translate(vec2(-(r.width() + 28.0), 0.0));
        }
        if r.bottom() > canvas.rect.bottom() {
            r = r.translate(vec2(0.0, -(r.height() + 28.0)));
        }
        painter.rect_filled(r, 3.0, pal.fill.gamma_multiply(0.95));
        painter.rect_stroke(
            r,
            3.0,
            egui::Stroke::new(1.0, pal.accent.gamma_multiply(0.7)),
            egui::StrokeKind::Inside,
        );
        painter.galley(r.min + vec2(4.0, 4.0), g, pal.text);
    }

    fn show_menu(&mut self, ui: &mut Ui, app: &mut AppState) {
        let Some(m) = &self.menu else { return };
        let path = app.view.path.clone();
        let name = wave_name(app, &path, &m.probe);
        let names = signal_names(app);
        let shown = names.iter().position(|n| n == &name);
        let input = match &m.probe {
            Probe::Pin(p) => top_input(app, p).map(|d| (d.name.clone(), d.width)),
            _ => None,
        };
        let mut close = false;
        let mut action: Option<Box<dyn FnOnce(&mut SimTool, &mut AppState)>> = None;
        let resp = egui::Area::new(Id::new("simux_canvas_menu"))
            .order(Order::Foreground)
            .fixed_pos(m.pos)
            .show(ui.ctx(), |ui| {
                egui::Frame::menu(ui.style()).show(ui, |ui| {
                    ui.set_min_width(180.0);
                    ui.weak(&name);
                    match shown {
                        None => {
                            if ui.button("Add to waveform").clicked() {
                                let n = name.clone();
                                action = Some(Box::new(move |_, app| {
                                    if !add_signal(app, &n) {
                                        app.log.push(format!("cannot watch '{n}'"));
                                    }
                                }));
                                close = true;
                            }
                        }
                        Some(i) => {
                            if ui.button("Remove from waveform").clicked() {
                                action = Some(Box::new(move |_, app| remove_signal(app, i)));
                                close = true;
                            }
                        }
                    }
                    if let Some((pin, width)) = &input {
                        let (pin, width) = (pin.clone(), *width);
                        if ui.button("Set value…").clicked() {
                            let pos = m.pos;
                            action = Some(Box::new(move |t, app| {
                                let live = SimTool::live_pin(app, &pin);
                                let cur = t.inputs.current(&pin, live.as_ref());
                                t.edit = Some(InputEdit {
                                    text: cur
                                        .and_then(|v| v.to_u64())
                                        .map_or(String::new(), |v| format!("0x{v:x}")),
                                    pin,
                                    width,
                                    pos,
                                    error: None,
                                    focus: true,
                                });
                            }));
                            close = true;
                        }
                    }
                });
            });
        let outside = ui.input(|i| {
            i.pointer.any_pressed()
                && i.pointer
                    .interact_pos()
                    .is_some_and(|p| !resp.response.rect.contains(p))
        });
        if close || outside || ui.input(|i| i.key_pressed(Key::Escape)) {
            self.menu = None;
        }
        if let Some(a) = action {
            a(self, app);
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
                Err(msg) => e.error = Some(msg),
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
            self.menu = None;
        }
        // Forget sent input values the snapshot confirms.
        if app.view.path.is_empty()
            && let Some(def) = app.doc.def(&app.view.def)
        {
            let pins: Vec<String> = def
                .pins
                .iter()
                .filter(|p| p.dir == Dir::In)
                .map(|p| p.name.clone())
                .collect();
            for p in pins {
                let live = Self::live_pin(app, &p);
                self.inputs.confirm(&p, live.as_ref());
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
        if r.secondary_clicked()
            && let Some(p) = r.interact_pointer_pos()
        {
            self.menu = Self::probe_at(scene, xf.to_world(p), xf.zoom).map(|probe| Menu { pos: p, probe });
        }
        if self.menu.is_none()
            && self.edit.is_none()
            && let Some(pr) = &probe
        {
            self.hover_label(ui, app, canvas, pr);
        }
        self.show_menu(ui, app);
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
    }
}
