//! egui panels and the canvas tool of the hierarchy feature. All behaviour lives in
//! [`super::actions`]; this file only draws and maps input to those actions.
//!
//! * [`Breadcrumbs`] (top bar): `top › cpu › alu` buttons, an *Up* button, the viewed def's
//!   classification badge; it also hosts the package-name prompt and the keyboard shortcuts
//!   (`Alt+↑` up, `Ctrl+G` package the selection), and rolls back kernel-rejected changes.
//! * [`HierMenu`] (the *Hierarchy* menu).
//! * [`LibraryTab`] / [`TreeTab`] (left panel tabs *Library* and *Instances*).
//! * [`HierTool`] (canvas): double-click a package instance to enter it.
//!
//! Placing library items uses the edit module's place tool ([`crate::edit::request_place`]:
//! click an entry then the canvas, or drag the entry onto the canvas); the selection to package
//! is the edit module's ([`crate::edit::current_selection`]).

use std::cell::RefCell;
use std::rc::Rc;

use egui::{Color32, Key, KeyboardShortcut, Modifiers, RichText, Ui};
use panda_kernel::ClassKind;

use super::actions::{self, HierState};
use super::classify::{Badge, BadgeState};
use super::library::{Category, LibEntry, user_entries};
use super::nav::{child_path, crumbs, package_instances};
use crate::app::{AppState, CanvasTool, Panel, Slot};
use crate::canvas::{CanvasResponse, Scene};
use crate::doc::ItemRef;
use crate::geom::SymbolKind;
use crate::simhost::SimEvent;

/// Shared state handle (single UI thread).
pub type Shared = Rc<RefCell<HierState>>;

const UP: KeyboardShortcut = KeyboardShortcut::new(Modifiers::ALT, Key::ArrowUp);
const PACKAGE: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::G);

fn badge_color(kind: ClassKind, dark: bool) -> Color32 {
    match (kind, dark) {
        (ClassKind::Combinational, true) => Color32::from_rgb(110, 200, 120),
        (ClassKind::Combinational, false) => Color32::from_rgb(30, 130, 50),
        (ClassKind::Sequential, true) => Color32::from_rgb(120, 170, 255),
        (ClassKind::Sequential, false) => Color32::from_rgb(30, 80, 190),
        (ClassKind::Oscillator, true) => Color32::from_rgb(255, 170, 80),
        (ClassKind::Oscillator, false) => Color32::from_rgb(190, 100, 0),
    }
}

/// Draw a badge (`short` text in a coloured frame, `long` text as tooltip).
pub fn badge_ui(ui: &mut Ui, state: Option<&BadgeState>, long: bool) {
    let dark = ui.visuals().dark_mode;
    match state {
        Some(BadgeState::Ready(b)) => {
            let c = badge_color(b.kind, dark);
            let text = if long { b.long() } else { b.short() };
            egui::Frame::new()
                .stroke(egui::Stroke::new(1.0, c))
                .corner_radius(4.0)
                .inner_margin(egui::Margin::symmetric(4, 0))
                .show(ui, |ui| ui.label(RichText::new(text).small().color(c)))
                .response
                .on_hover_text(badge_tooltip(b));
        }
        Some(BadgeState::Failed(e)) => {
            ui.label(RichText::new("?").small().weak()).on_hover_text(e);
        }
        Some(BadgeState::Pending) => {
            ui.label(RichText::new("…").small().weak())
                .on_hover_text("classifying");
        }
        None => {}
    }
}

fn badge_tooltip(b: &Badge) -> String {
    format!("Classification: {}", b.long())
}

/// Instance ids of the current selection: the edit module's, else the canvas highlights.
fn selection(ctx: &egui::Context, app: &AppState) -> Vec<String> {
    let sel = crate::edit::current_selection(ctx);
    let v = actions::instance_ids(&sel.items);
    if v.is_empty() {
        actions::selected_instances(app)
    } else {
        v
    }
}

fn ask_package_name(ctx: &egui::Context, hs: &mut HierState, app: &AppState) {
    let sel = selection(ctx, app);
    if sel.is_empty() {
        hs.status = Some("select the instances to package first".into());
        return;
    }
    hs.ask_package(app, sel);
}

// ---------------------------------------------------------------- breadcrumbs

/// Top-bar breadcrumbs, badge, prompt host and shortcuts.
pub struct Breadcrumbs(pub Shared);

impl Panel for Breadcrumbs {
    fn id(&self) -> &'static str {
        "hier.breadcrumbs"
    }
    fn slot(&self) -> Slot {
        Slot::TopBar
    }

    fn on_frame(&mut self, ctx: &egui::Context, app: &mut AppState) {
        let hs = &mut *self.0.borrow_mut();
        hs.sync(app);
        hs.classifier.sync(&app.doc);
        hs.classifier.poll();
        // Fit when the view changed by any means.
        let view = (app.view.path.clone(), app.view.def.clone());
        if hs.last_view.as_ref() != Some(&view) {
            if hs.last_view.is_some() {
                app.canvas.request_fit();
            }
            hs.last_view = Some(view);
        }
        if hs.prompt.is_none() {
            let (up, pkg) =
                ctx.input_mut(|i| (i.consume_shortcut(&UP), i.consume_shortcut(&PACKAGE)));
            if up {
                actions::exit(app);
            }
            if pkg {
                ask_package_name(ctx, hs, app);
            }
        }
        prompt_modal(ctx, hs, app);
    }

    fn on_event(&mut self, ev: &SimEvent, app: &mut AppState) {
        if let SimEvent::Edited { id, result } = ev {
            self.0.borrow_mut().edited(app, *id, result.is_ok());
        }
    }

    fn ui(&mut self, ui: &mut Ui, app: &mut AppState) {
        let hs = &mut *self.0.borrow_mut();
        let cs = crumbs(&app.doc.design, &app.view.path);
        let mut go: Option<String> = None;
        ui.add_enabled_ui(!app.view.path.is_empty(), |ui| {
            if ui
                .small_button("⏶")
                .on_hover_text("Up to the enclosing package (Alt+↑)")
                .clicked()
            {
                go = crate::hier::nav::parent_path(&app.view.path);
            }
        });
        for (k, c) in cs.iter().enumerate() {
            if k > 0 {
                ui.label(RichText::new("›").weak());
            }
            let last = k + 1 == cs.len();
            let text = if last {
                RichText::new(&c.label).strong()
            } else {
                RichText::new(&c.label)
            };
            if ui
                .add(egui::Button::new(text).frame(false))
                .on_hover_text(format!(
                    "{} ({})",
                    if c.path.is_empty() { "/" } else { &c.path },
                    c.def
                ))
                .clicked()
                && !last
            {
                go = Some(c.path.clone());
            }
        }
        ui.label(RichText::new(format!("({})", app.view.def)).weak().small());
        let def = app.view.def.clone();
        let st = hs.classifier.get(&def).clone();
        badge_ui(ui, Some(&st), true);
        if let Some(p) = go {
            app.enter(&p);
        }
    }
}

fn prompt_modal(ctx: &egui::Context, hs: &mut HierState, app: &mut AppState) {
    let Some(p) = hs.prompt.as_mut() else { return };
    let mut submit = false;
    let mut cancel = false;
    let r = egui::Modal::new(egui::Id::new("hier.prompt")).show(ctx, |ui| {
        ui.set_width(420.0);
        ui.heading("Create package from selection");
        ui.label(format!(
            "{} instance(s): {}",
            p.selection.len(),
            p.selection.join(", ")
        ));
        ui.label("Package (def) name");
        let te = ui.add(egui::TextEdit::singleline(&mut p.name).desired_width(400.0));
        te.request_focus();
        if te.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
            submit = true;
        }
        if let Some(e) = &p.error {
            ui.colored_label(ui.visuals().error_fg_color, e);
        }
        ui.horizontal(|ui| {
            if ui.button("Create").clicked() {
                submit = true;
            }
            if ui.button("Cancel").clicked() {
                cancel = true;
            }
        });
    });
    if r.should_close() {
        cancel = true;
    }
    if cancel {
        hs.prompt = None;
    } else if submit {
        hs.submit_prompt(crate::edit::shared_editor(ctx).as_ref(), app);
    }
}

// ---------------------------------------------------------------- menus

/// The *Hierarchy* menu.
pub struct HierMenu(pub Shared);

impl Panel for HierMenu {
    fn id(&self) -> &'static str {
        "hier.menu"
    }
    fn title(&self) -> String {
        "Hierarchy".into()
    }
    fn slot(&self) -> Slot {
        Slot::Menu
    }
    fn ui(&mut self, ui: &mut Ui, app: &mut AppState) {
        let hs = &mut *self.0.borrow_mut();
        let sel = selection(ui.ctx(), app);
        let pkg_sel: Option<String> = match sel.as_slice() {
            [one] => app
                .doc
                .def(&app.view.def)
                .and_then(|d| d.instances.iter().find(|i| &i.id == one))
                .filter(|i| matches!(i.kind, panda_netlist::InstKind::Def(_)))
                .map(|i| i.id.clone()),
            _ => None,
        };
        if ui
            .add_enabled(
                pkg_sel.is_some(),
                egui::Button::new("Enter selected package"),
            )
            .clicked()
        {
            if let Some(i) = &pkg_sel {
                actions::enter_instance(app, i);
            }
            ui.close();
        }
        if ui
            .add_enabled(
                !app.view.path.is_empty(),
                egui::Button::new("Up one level").shortcut_text(ui.ctx().format_shortcut(&UP)),
            )
            .clicked()
        {
            actions::exit(app);
            ui.close();
        }
        if ui
            .add_enabled(!app.view.path.is_empty(), egui::Button::new("Go to top"))
            .clicked()
        {
            app.enter("");
            ui.close();
        }
        ui.separator();
        if ui
            .add_enabled(
                !sel.is_empty(),
                egui::Button::new(format!("Create package from selection ({})", sel.len()))
                    .shortcut_text(ui.ctx().format_shortcut(&PACKAGE)),
            )
            .clicked()
        {
            ask_package_name(ui.ctx(), hs, app);
            ui.close();
        }
    }
}

// ---------------------------------------------------------------- library

/// The library palette (left panel tab).
pub struct LibraryTab(pub Shared);

fn entry_matches(e: &LibEntry, filter: &str) -> bool {
    filter.is_empty() || e.name.to_lowercase().contains(&filter.to_lowercase())
}

impl Panel for LibraryTab {
    fn id(&self) -> &'static str {
        "hier.library"
    }
    fn title(&self) -> String {
        "Library".into()
    }
    fn slot(&self) -> Slot {
        Slot::Left
    }
    fn ui(&mut self, ui: &mut Ui, app: &mut AppState) {
        let hs = &mut *self.0.borrow_mut();
        ui.horizontal(|ui| {
            ui.label("🔍");
            ui.add(egui::TextEdit::singleline(&mut hs.filter).hint_text("filter"));
        });
        ui.label(
            RichText::new("Click to place (then click the canvas) or drag onto the canvas.")
                .small()
                .weak(),
        );
        ui.separator();
        let into = app.view.def.clone();
        let user = user_entries(&app.doc.design);
        let mut arm: Option<LibEntry> = None;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let cats = [
                    Category::Atoms,
                    Category::User,
                    Category::Arithmetic,
                    Category::Sequential,
                    Category::Misc,
                ];
                for cat in cats {
                    let list: Vec<usize> = if cat == Category::User {
                        (0..user.len()).collect()
                    } else {
                        (0..hs.builtin.len())
                            .filter(|&i| hs.builtin[i].category == cat)
                            .collect()
                    };
                    let shown: Vec<usize> = list
                        .into_iter()
                        .filter(|&i| {
                            let e = if cat == Category::User {
                                &user[i]
                            } else {
                                &hs.builtin[i]
                            };
                            entry_matches(e, &hs.filter)
                        })
                        .collect();
                    if shown.is_empty() {
                        continue;
                    }
                    egui::CollapsingHeader::new(cat.title())
                        .default_open(cat != Category::Misc)
                        .show(ui, |ui| {
                            for i in shown {
                                let e = if cat == Category::User {
                                    user[i].clone()
                                } else {
                                    hs.builtin[i].clone()
                                };
                                let allowed = e.allowed_in(&app.doc.design, &into);
                                let selected = hs.selected_entry.as_deref() == Some(&e.name);
                                ui.horizontal(|ui| {
                                    let r = ui.add_enabled(
                                        allowed,
                                        egui::Button::selectable(selected, &e.name)
                                            .sense(egui::Sense::click_and_drag()),
                                    );
                                    if r.clicked() || r.drag_started() {
                                        hs.selected_entry = Some(e.name.clone());
                                        arm = Some(e.clone());
                                    }
                                    if cat == Category::User {
                                        let st = hs.classifier.get(&e.name).clone();
                                        badge_ui(ui, Some(&st), false);
                                    }
                                });
                                if selected && !e.params.is_empty() && cat != Category::User {
                                    // Parameters apply to the next placement of this entry.
                                    let entry = &mut hs.builtin[i];
                                    ui.indent(("params", &e.name), |ui| {
                                        for p in &mut entry.params {
                                            ui.horizontal(|ui| {
                                                ui.label(RichText::new(p.name).small());
                                                ui.add(
                                                    egui::DragValue::new(&mut p.value)
                                                        .range(p.min..=p.max),
                                                );
                                            });
                                        }
                                    });
                                }
                            }
                        });
                }
            });
        if let Some(e) = arm {
            let ed = crate::edit::shared_editor(ui.ctx());
            match actions::arm_entry(hs, ed.as_ref(), app, &e) {
                Ok(req) => crate::edit::request_place(ui.ctx(), req),
                Err(err) => hs.status = Some(err),
            }
        }
        if let Some(s) = &hs.status {
            ui.separator();
            ui.label(RichText::new(s).small());
        }
    }
}

// ---------------------------------------------------------------- hierarchy tree

/// The instance hierarchy (left panel tab): click an instance to view it.
pub struct TreeTab(pub Shared);

impl Panel for TreeTab {
    fn id(&self) -> &'static str {
        "hier.tree"
    }
    fn title(&self) -> String {
        "Instances".into()
    }
    fn slot(&self) -> Slot {
        Slot::Left
    }
    fn ui(&mut self, ui: &mut Ui, app: &mut AppState) {
        let hs = &mut *self.0.borrow_mut();
        let mut go: Option<String> = None;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let top = app.doc.design.top.clone();
                tree_node(ui, hs, app, "", &top, &top, &mut go);
            });
        if let Some(p) = go {
            app.enter(&p);
        }
    }
}

fn tree_node(
    ui: &mut Ui,
    hs: &mut HierState,
    app: &AppState,
    path: &str,
    label: &str,
    def: &str,
    go: &mut Option<String>,
) {
    let children = package_instances(&app.doc.design, def);
    let current = app.view.path == path;
    let row = |ui: &mut Ui, hs: &mut HierState, go: &mut Option<String>| {
        let r = ui.selectable_label(current, label).on_hover_text(def);
        if r.clicked() {
            *go = Some(path.to_string());
        }
        let st = hs.classifier.peek(def).cloned();
        badge_ui(ui, st.as_ref(), false);
        if label != def {
            ui.label(RichText::new(def).small().weak());
        }
    };
    if children.is_empty() {
        ui.horizontal(|ui| row(ui, hs, go));
        return;
    }
    let id = ui.make_persistent_id(("hier.tree", path));
    let open_default = path.is_empty() || app.view.path.starts_with(path);
    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, open_default)
        .show_header(ui, |ui| row(ui, hs, go))
        .body(|ui| {
            for (inst, cdef) in children {
                let p = child_path(path, &inst);
                tree_node(ui, hs, app, &p, &inst, &cdef, go);
            }
        });
}

// ---------------------------------------------------------------- canvas tool

/// Canvas input: double-click a package instance to enter it.
pub struct HierTool(pub Shared);

impl CanvasTool for HierTool {
    fn id(&self) -> &'static str {
        "hier.tool"
    }

    fn canvas(&mut self, _ui: &mut Ui, canvas: &CanvasResponse, scene: &Scene, app: &mut AppState) {
        if canvas.response.double_clicked()
            && let Some(w) = canvas.hover_world
            && let Some(item) = scene.hit_item(w)
            && let ItemRef::Instance(id) = &item.item
            && matches!(item.symbol.kind, SymbolKind::Package { .. })
        {
            let id = id.clone();
            actions::enter_instance(app, &id);
        }
    }
}
