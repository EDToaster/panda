//! Menus: the "Edit" and "Place" menus of the top bar (tools, undo/redo, clipboard,
//! rotate/flip/delete, placing atoms, pins and packages) and the **context menu** of canvas
//! items ([`ContextMenu`], a canvas tool).
//!
//! # Context menu
//!
//! A right click on an instance, pin or wire selects it (unless it already is part of the
//! selection, which is then kept) and opens a menu acting on the selection: Inspect, Enter
//! package, Add to / Remove from waveform, Rotate, Flip, Cut / Copy / Duplicate, Select net,
//! Termination, Delete. On empty canvas: Paste here, Select all, Fit view. The entries are
//! computed by the pure [`entries`] and run by [`run`] / [`run_edit`].
//!
//! Hooks into other modules (thin wrappers, so the call sites stay in one place):
//! [`wave_name`] / [`wave_index`] / [`add_to_waveform`] / [`remove_from_waveform`] (sim UX
//! waveform viewer) and [`enter_package`] (hierarchy navigation). Other modules can add their
//! own entries with [`add_context_entry`]; [`context_menu_open`] tells a canvas tool that runs
//! later in the frame that the right click was taken.

use egui::{Id, Order, Pos2, Ui, vec2};
use panda_netlist::{Atom, InstKind, Termination};

use super::SharedEditor;
use super::actions::{Editor, PlaceRequest, Selection, Tool, WireSeg, instantiates};
use super::history::EditTarget;
use super::inspector::{TERMINATIONS, term_label};
use super::route;
use crate::app::{AppState, CanvasTool, Panel, Slot};
use crate::canvas::{CanvasResponse, Scene};
use crate::doc::{Document, ItemRef};
use crate::geom::{GridPt, gp};

/// The "Edit" menu.
pub struct EditMenu {
    ed: SharedEditor,
}

impl EditMenu {
    /// Menu over the shared editor.
    pub fn new(ed: SharedEditor) -> EditMenu {
        EditMenu { ed }
    }
}

impl Panel for EditMenu {
    fn id(&self) -> &'static str {
        "edit.menu"
    }

    fn title(&self) -> String {
        "Edit".into()
    }

    fn slot(&self) -> Slot {
        Slot::Menu
    }

    fn ui(&mut self, ui: &mut Ui, app: &mut AppState) {
        let def = app.view.def.clone();
        let ed = self.ed.clone();
        let mut ed = ed.borrow_mut();
        let tool = ed.tool.clone();
        for (t, label) in [
            (Tool::Select, "Select tool  (S)"),
            (Tool::Wire, "Wire tool  (W)"),
            (Tool::Pan, "Pan tool"),
        ] {
            if ui.selectable_label(tool == t, label).clicked() {
                ed.tool = t;
                ui.close();
            }
        }
        ui.separator();
        let undo = ed
            .history
            .undo_label()
            .map(|l| format!("Undo {l}  (Ctrl+Z)"));
        if ui
            .add_enabled(
                undo.is_some(),
                egui::Button::new(undo.unwrap_or("Undo".into())),
            )
            .clicked()
        {
            let _ = ed.undo(app);
            ui.close();
        }
        let redo = ed
            .history
            .redo_label()
            .map(|l| format!("Redo {l}  (Ctrl+Shift+Z)"));
        if ui
            .add_enabled(
                redo.is_some(),
                egui::Button::new(redo.unwrap_or("Redo".into())),
            )
            .clicked()
        {
            let _ = ed.redo(app);
            ui.close();
        }
        ui.separator();
        let has_items = !ed.selection.items.is_empty();
        let has_sel = !ed.selection.is_empty();
        let has_clip = ed.clipboard.is_some();
        if ui
            .add_enabled(has_items, egui::Button::new("Cut  (Ctrl+X)"))
            .clicked()
        {
            let _ = ed.cut(app, &def);
            ui.close();
        }
        if ui
            .add_enabled(has_items, egui::Button::new("Copy  (Ctrl+C)"))
            .clicked()
        {
            ed.copy(&app.doc, &def);
            ui.close();
        }
        if ui
            .add_enabled(has_clip, egui::Button::new("Paste  (Ctrl+V)"))
            .clicked()
        {
            let _ = ed.paste(app, &def, None);
            ui.close();
        }
        if ui
            .add_enabled(has_items, egui::Button::new("Duplicate  (Ctrl+D)"))
            .clicked()
        {
            let _ = ed.duplicate(app, &def);
            ui.close();
        }
        if ui
            .add_enabled(has_sel, egui::Button::new("Delete  (Del)"))
            .clicked()
        {
            let _ = ed.delete_selection(app, &def);
            ui.close();
        }
        ui.separator();
        if ui
            .add_enabled(has_items, egui::Button::new("Rotate  (R)"))
            .clicked()
        {
            let _ = ed.rotate_selection(app, &def, true);
            ui.close();
        }
        if ui
            .add_enabled(has_items, egui::Button::new("Rotate back  (Shift+R)"))
            .clicked()
        {
            let _ = ed.rotate_selection(app, &def, false);
            ui.close();
        }
        if ui
            .add_enabled(has_items, egui::Button::new("Flip  (F)"))
            .clicked()
        {
            let _ = ed.flip_selection(app, &def);
            ui.close();
        }
        if ui.button("Select all  (Ctrl+A)").clicked() {
            ed.select_all(&app.doc, &def);
            ui.close();
        }
    }
}

/// The "Place" menu: basic atoms and pins, and packages (defs) that can be instantiated here.
pub struct PlaceMenu {
    ed: SharedEditor,
}

impl PlaceMenu {
    /// Menu over the shared editor.
    pub fn new(ed: SharedEditor) -> PlaceMenu {
        PlaceMenu { ed }
    }
}

impl Panel for PlaceMenu {
    fn id(&self) -> &'static str {
        "edit.place"
    }

    fn title(&self) -> String {
        "Place".into()
    }

    fn slot(&self) -> Slot {
        Slot::Menu
    }

    fn ui(&mut self, ui: &mut Ui, app: &mut AppState) {
        let def = app.view.def.clone();
        let is_top = app.doc.design.top == def;
        let mut chosen = None;
        for req in super::basic_palette() {
            let source = matches!(
                &req,
                PlaceRequest::Atom(Atom::Clock { .. } | Atom::PowerOnReset { .. })
            );
            let r = ui
                .add_enabled(!source || is_top, egui::Button::new(req.label()))
                .on_disabled_hover_text("only in the top def");
            if r.clicked() {
                chosen = Some(req);
            }
        }
        let defs: Vec<String> = app
            .doc
            .design
            .defs
            .iter()
            .filter(|d| !instantiates(&app.doc.design, &d.id, &def))
            .map(|d| d.id.clone())
            .collect();
        if !defs.is_empty() {
            ui.separator();
            ui.menu_button("Package", |ui| {
                egui::ScrollArea::vertical()
                    .max_height(320.0)
                    .show(ui, |ui| {
                        for d in defs {
                            if ui.button(&d).clicked() {
                                chosen = Some(PlaceRequest::Def(d));
                            }
                        }
                    });
            });
        }
        if let Some(req) = chosen {
            self.ed.borrow_mut().tool = Tool::Place(req);
            ui.close();
        }
    }
}

// =====================================================================================
// Context menu: pure part
// =====================================================================================

/// What a right click hit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    /// An instance (atom or package) by id.
    Instance(String),
    /// A pin marker of the viewed def by name.
    Pin(String),
    /// A wire of a net by id.
    Net(String),
    /// Empty canvas.
    Canvas,
}

/// One command of the context menu.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MenuAction {
    /// Show the selection in the inspector (selects only the target).
    Inspect,
    /// Enter the package instance.
    EnterPackage(String),
    /// Add the hierarchical net name to the waveform viewer.
    AddToWaveform(String),
    /// Remove waveform signal `i`.
    RemoveFromWaveform(usize),
    /// Rotate the selected items clockwise.
    RotateCw,
    /// Rotate the selected items counter-clockwise.
    RotateCcw,
    /// Mirror the selected items.
    Flip,
    /// Cut the selected items.
    Cut,
    /// Copy the selected items.
    Copy,
    /// Paste the clipboard at a grid point.
    Paste(GridPt),
    /// Duplicate the selected items.
    Duplicate,
    /// Delete the selection.
    Delete,
    /// Select every wire segment of the net.
    SelectNet(String),
    /// Set the termination of the net.
    Termination(String, Termination),
    /// Select everything in the def.
    SelectAll,
    /// Fit the view to the def.
    FitView,
}

/// A context menu entry; `None` entries in an [`entries`] list are separators.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// The command.
    pub action: MenuAction,
    /// Label (with the keyboard shortcut, if any).
    pub label: String,
    /// False: shown greyed out.
    pub enabled: bool,
    /// Checked (radio state, for terminations).
    pub checked: bool,
}

fn entry(action: MenuAction, label: impl Into<String>) -> Option<Entry> {
    Some(Entry {
        action,
        label: label.into(),
        enabled: true,
        checked: false,
    })
}

/// Waveform state of a probed signal: its hierarchical name and its index in the viewer.
pub type WaveState = (String, Option<usize>);

/// The context menu for `target` with the (already updated) selection `sel` of `def_id`.
/// `wave` is the waveform state of the target (pins and nets; `None` hides the entry, e.g.
/// when another menu already offers it), `clipboard` whether there is something to paste and
/// `at` the clicked grid point.
pub fn entries(
    doc: &Document,
    def_id: &str,
    target: &Target,
    sel: &Selection,
    wave: Option<&WaveState>,
    clipboard: bool,
    at: GridPt,
) -> Vec<Option<Entry>> {
    let mut v = Vec::new();
    let def = doc.def(def_id);
    if *target == Target::Canvas {
        v.push(Some(Entry {
            enabled: clipboard,
            ..entry(MenuAction::Paste(at), "Paste here  (Ctrl+V)").expect("entry")
        }));
        v.push(entry(MenuAction::SelectAll, "Select all  (Ctrl+A)"));
        v.push(entry(MenuAction::FitView, "Fit view"));
        if !sel.is_empty() {
            v.push(None);
            v.push(entry(MenuAction::Delete, "Delete selection  (Del)"));
        }
        return v;
    }
    let n = sel.items.len() + sel.wires.len();
    v.push(entry(
        MenuAction::Inspect,
        if n > 1 {
            "Inspect only this"
        } else {
            "Inspect"
        },
    ));
    if let Target::Instance(id) = target
        && def
            .and_then(|d| d.instances.iter().find(|i| &i.id == id))
            .is_some_and(|i| matches!(i.kind, InstKind::Def(_)))
    {
        v.push(entry(MenuAction::EnterPackage(id.clone()), "Enter package"));
    }
    if let Some((name, idx)) = wave {
        v.push(match idx {
            None => entry(MenuAction::AddToWaveform(name.clone()), "Add to waveform"),
            Some(i) => entry(MenuAction::RemoveFromWaveform(*i), "Remove from waveform"),
        });
    }
    if let Target::Net(net) = target {
        v.push(None);
        v.push(entry(
            MenuAction::SelectNet(net.clone()),
            "Select whole net",
        ));
        let cur = def
            .and_then(|d| d.nets.iter().find(|x| &x.id == net))
            .map(|x| x.termination);
        for t in TERMINATIONS {
            v.push(Some(Entry {
                checked: cur == Some(t),
                ..entry(MenuAction::Termination(net.clone(), t), term_label(t)).expect("entry")
            }));
        }
    }
    if !sel.items.is_empty() {
        v.push(None);
        v.push(entry(MenuAction::RotateCw, "Rotate  (R)"));
        v.push(entry(MenuAction::RotateCcw, "Rotate back  (Shift+R)"));
        v.push(entry(MenuAction::Flip, "Flip  (F)"));
        v.push(None);
        v.push(entry(MenuAction::Cut, "Cut  (Ctrl+X)"));
        v.push(entry(MenuAction::Copy, "Copy  (Ctrl+C)"));
        v.push(entry(MenuAction::Duplicate, "Duplicate  (Ctrl+D)"));
    }
    v.push(None);
    v.push(entry(
        MenuAction::Delete,
        if n > 1 {
            format!("Delete {n} selected  (Del)")
        } else {
            "Delete  (Del)".to_string()
        },
    ));
    v
}

/// Update the selection for a right click on `target` (wire segment `seg` for a net): an
/// unselected target becomes the selection; a selected one keeps the selection as it is.
pub fn select_for_context(sel: &mut Selection, target: &Target, seg: Option<&WireSeg>) {
    match target {
        Target::Instance(id) => {
            let it = ItemRef::Instance(id.clone());
            if !sel.items.contains(&it) {
                sel.clear();
                sel.items.insert(it);
            }
        }
        Target::Pin(p) => {
            let it = ItemRef::Pin(p.clone());
            if !sel.items.contains(&it) {
                sel.clear();
                sel.items.insert(it);
            }
        }
        Target::Net(net) => {
            let selected = match seg {
                Some(s) => sel.wires.contains(s),
                None => sel.wires.iter().any(|w| &w.net == net),
            };
            if !selected {
                sel.clear();
                if let Some(s) = seg {
                    sel.wires.insert(s.clone());
                }
            }
        }
        Target::Canvas => {}
    }
}

/// Every wire segment of `net` in `def_id`.
pub fn net_segments(doc: &Document, def_id: &str, net: &str) -> Vec<WireSeg> {
    doc.editor(def_id)
        .and_then(|e| e.wires.get(net))
        .map(|g| g.segments().map(|(a, b)| WireSeg::new(net, a, b)).collect())
        .unwrap_or_default()
}

/// Run an editing command on any [`EditTarget`]. Returns `None` for commands that need the
/// app ([`MenuAction::EnterPackage`], waveform, [`MenuAction::FitView`]; see [`run`]).
pub fn run_edit(
    action: &MenuAction,
    ed: &mut Editor,
    t: &mut dyn EditTarget,
    def_id: &str,
    target: &Target,
) -> Option<Result<(), String>> {
    Some(match action {
        MenuAction::Inspect => {
            let mut only = Selection::default();
            match target {
                Target::Instance(id) => {
                    only.items.insert(ItemRef::Instance(id.clone()));
                }
                Target::Pin(p) => {
                    only.items.insert(ItemRef::Pin(p.clone()));
                }
                Target::Net(n) => {
                    only.wires = net_segments(t.doc(), def_id, n).into_iter().collect()
                }
                Target::Canvas => {}
            }
            ed.selection = only;
            Ok(())
        }
        MenuAction::RotateCw => ed.rotate_selection(t, def_id, true),
        MenuAction::RotateCcw => ed.rotate_selection(t, def_id, false),
        MenuAction::Flip => ed.flip_selection(t, def_id),
        MenuAction::Cut => ed.cut(t, def_id),
        MenuAction::Copy => {
            ed.copy(t.doc(), def_id);
            Ok(())
        }
        MenuAction::Paste(at) => ed.paste(t, def_id, Some(*at)),
        MenuAction::Duplicate => ed.duplicate(t, def_id),
        MenuAction::Delete => ed.delete_selection(t, def_id),
        MenuAction::SelectNet(n) => {
            ed.selection.clear();
            ed.selection.wires = net_segments(t.doc(), def_id, n).into_iter().collect();
            Ok(())
        }
        MenuAction::Termination(n, term) => ed.set_termination(t, def_id, n, *term),
        MenuAction::SelectAll => {
            ed.select_all(t.doc(), def_id);
            Ok(())
        }
        MenuAction::EnterPackage(_)
        | MenuAction::AddToWaveform(_)
        | MenuAction::RemoveFromWaveform(_)
        | MenuAction::FitView => return None,
    })
}

/// Run a context menu command in the app.
pub fn run(action: &MenuAction, ed: &mut Editor, app: &mut AppState, target: &Target) {
    let def = app.view.def.clone();
    if run_edit(action, ed, app, &def, target).is_some() {
        return;
    }
    match action {
        MenuAction::EnterPackage(inst) if !enter_package(app, inst) => {
            ed.message = Some(format!("cannot enter '{inst}'"));
        }
        MenuAction::AddToWaveform(name) => add_to_waveform(app, name),
        MenuAction::RemoveFromWaveform(i) => remove_from_waveform(app, *i),
        MenuAction::FitView => app.canvas.request_fit(),
        _ => {}
    }
}

// =====================================================================================
// Hooks into other modules
// =====================================================================================

/// Hierarchical waveform name of a pin / net target in the current view (sim UX naming).
pub fn wave_name(app: &AppState, target: &Target) -> Option<String> {
    use crate::simux::canvas_tool::Probe;
    let probe = match target {
        Target::Pin(p) => Probe::Pin(p.clone()),
        Target::Net(n) => Probe::Net(n.clone()),
        _ => return None,
    };
    Some(crate::simux::canvas_tool::wave_name(
        app,
        &app.view.path,
        &probe,
    ))
}

/// Index of signal `name` in the waveform viewer.
pub fn wave_index(app: &AppState, name: &str) -> Option<usize> {
    crate::simux::waveform::signal_names(app)
        .iter()
        .position(|n| n == name)
}

/// Add signal `name` to the waveform viewer (logs if it cannot be watched).
pub fn add_to_waveform(app: &mut AppState, name: &str) {
    if !crate::simux::waveform::add_signal(app, name) {
        app.log.push(format!("cannot watch '{name}'"));
    }
}

/// Remove waveform signal `i`.
pub fn remove_from_waveform(app: &mut AppState, i: usize) {
    crate::simux::waveform::remove_signal(app, i);
}

/// Enter package instance `inst` of the current view (hierarchy navigation). False if it is
/// not a package instance.
pub fn enter_package(app: &mut AppState, inst: &str) -> bool {
    crate::hier::actions::enter_instance(app, inst)
}

/// An extra context menu entry contributed by another module ([`add_context_entry`]).
#[derive(Clone, Debug)]
pub struct ContextEntry {
    /// Unique key (re-adding replaces the entry).
    pub key: &'static str,
    /// Label.
    pub label: String,
    /// Whether the entry is shown for a target.
    pub applies: fn(&AppState, &Target) -> bool,
    /// Run it (after the selection was updated for the target).
    pub run: fn(&mut AppState, &Target),
}

#[derive(Clone, Debug, Default)]
struct ContextEntries(Vec<ContextEntry>);

fn entries_id() -> Id {
    Id::new("panda.edit.context_entries")
}

fn open_id() -> Id {
    Id::new("panda.edit.context_open")
}

/// Add (or replace, by key) an entry of the canvas context menu.
pub fn add_context_entry(ctx: &egui::Context, e: ContextEntry) {
    ctx.data_mut(|d| {
        let v = d.get_temp_mut_or_default::<ContextEntries>(entries_id());
        v.0.retain(|x| x.key != e.key);
        v.0.push(e);
    });
}

/// True while the edit context menu is open (set before later canvas tools run in the frame
/// the right click happened, so they can skip their own menu).
pub fn context_menu_open(ctx: &egui::Context) -> bool {
    ctx.data(|d| d.get_temp::<bool>(open_id())).unwrap_or(false)
}

// =====================================================================================
// Context menu: the canvas tool
// =====================================================================================

/// An open context menu.
#[derive(Clone, Debug)]
struct Open {
    /// Screen position of the click.
    pos: Pos2,
    /// Grid point of the click.
    at: GridPt,
    target: Target,
}

/// The canvas context menu (a [`CanvasTool`], registered after the edit tool).
pub struct ContextMenu {
    ed: SharedEditor,
    open: Option<Open>,
    /// Tool of the previous frame: a right click that cancelled a wire or a placement does not
    /// open the menu.
    last_tool: Tool,
}

impl ContextMenu {
    /// Context menu over the shared editor.
    pub fn new(ed: SharedEditor) -> ContextMenu {
        ContextMenu {
            ed,
            open: None,
            last_tool: Tool::Select,
        }
    }
}

/// The nearest wire segment within `tol` of world point `w`.
fn hit_wire(scene: &Scene, w: Pos2, tol: f32) -> Option<WireSeg> {
    let mut best: Option<(f32, WireSeg)> = None;
    for n in &scene.nets {
        let b = &n.bounds;
        if w.x < b.min.x as f32 - tol
            || w.x > b.max.x as f32 + tol
            || w.y < b.min.y as f32 - tol
            || w.y > b.max.y as f32 + tol
        {
            continue;
        }
        for &s in &n.segs {
            let d = route::distance(w.x, w.y, s);
            if d <= tol && best.as_ref().is_none_or(|(bd, _)| d < *bd) {
                best = Some((d, WireSeg::new(n.id.clone(), s.0, s.1)));
            }
        }
    }
    best.map(|(_, s)| s)
}

/// What is under world point `w` (items win over wires).
pub fn target_at(scene: &Scene, w: Pos2, zoom: f32) -> (Target, Option<WireSeg>) {
    if let Some(p) = scene.hit_item(w) {
        return (
            match &p.item {
                ItemRef::Instance(i) => Target::Instance(i.clone()),
                ItemRef::Pin(n) => Target::Pin(n.clone()),
            },
            None,
        );
    }
    let tol = (6.0 / zoom).clamp(0.15, 0.6);
    match hit_wire(scene, w, tol) {
        Some(s) => (Target::Net(s.net.clone()), Some(s)),
        None => (Target::Canvas, None),
    }
}

impl CanvasTool for ContextMenu {
    fn id(&self) -> &'static str {
        "edit.context_menu"
    }

    fn canvas(&mut self, ui: &mut Ui, canvas: &CanvasResponse, scene: &Scene, app: &mut AppState) {
        let ctx = ui.ctx().clone();
        let tool = self.ed.borrow().tool.clone();
        let was_select = self.last_tool == Tool::Select;
        self.last_tool = tool.clone();
        let resp = &canvas.response;
        if resp.secondary_clicked()
            && was_select
            && tool == Tool::Select
            && let Some(p) = resp.interact_pointer_pos()
        {
            let w = canvas.xf.to_world(p);
            let (target, seg) = target_at(scene, w, canvas.xf.zoom);
            let mut ed = self.ed.borrow_mut();
            select_for_context(&mut ed.selection, &target, seg.as_ref());
            super::publish_selection(&ctx, &ed.selection);
            self.open = Some(Open {
                pos: p,
                at: gp(w.x.round() as i32, w.y.round() as i32),
                target,
            });
        }
        // The view changed under the menu (entered a package, opened a file): close it.
        if self.open.is_some() && scene.def != app.view.def {
            self.open = None;
        }
        ctx.data_mut(|d| d.insert_temp(open_id(), self.open.is_some()));
        let Some(open) = self.open.clone() else {
            return;
        };

        let pos = open.pos + vec2(2.0, 2.0);
        let def = app.view.def.clone();
        let mut ed = self.ed.borrow_mut();
        let wave = wave_name(app, &open.target).map(|n| {
            let i = wave_index(app, &n);
            (n, i)
        });
        let list = entries(
            &app.doc,
            &def,
            &open.target,
            &ed.selection,
            wave.as_ref(),
            ed.clipboard.is_some(),
            open.at,
        );
        let extra: Vec<ContextEntry> = ctx
            .data(|d| d.get_temp::<ContextEntries>(entries_id()))
            .map(|e| e.0)
            .unwrap_or_default()
            .into_iter()
            .filter(|e| (e.applies)(app, &open.target))
            .collect();
        let title = match &open.target {
            Target::Instance(i) => i.clone(),
            Target::Pin(p) => format!("pin {p}"),
            Target::Net(n) => format!("net {n}"),
            Target::Canvas => def.clone(),
        };
        let mut chosen: Option<MenuAction> = None;
        let mut extra_chosen: Option<ContextEntry> = None;
        let area = egui::Area::new(Id::new("panda_edit_context_menu"))
            .order(Order::Foreground)
            .fixed_pos(pos)
            .constrain(true)
            .show(&ctx, |ui| {
                egui::Frame::menu(ui.style()).show(ui, |ui| {
                    ui.set_min_width(180.0);
                    ui.weak(title);
                    let terms: Vec<&Entry> = list
                        .iter()
                        .flatten()
                        .filter(|e| matches!(e.action, MenuAction::Termination(..)))
                        .collect();
                    let mut terms_shown = false;
                    for e in &list {
                        match e {
                            None => {
                                ui.separator();
                            }
                            Some(e) if matches!(e.action, MenuAction::Termination(..)) => {
                                if terms_shown {
                                    continue;
                                }
                                terms_shown = true;
                                ui.label(egui::RichText::new("Termination").weak());
                                ui.horizontal(|ui| {
                                    for t in &terms {
                                        if ui.selectable_label(t.checked, &t.label).clicked() {
                                            chosen = Some(t.action.clone());
                                        }
                                    }
                                });
                            }
                            Some(e) => {
                                if ui
                                    .add_enabled(e.enabled, egui::Button::new(&e.label))
                                    .clicked()
                                {
                                    chosen = Some(e.action.clone());
                                }
                            }
                        }
                    }
                    if !extra.is_empty() {
                        ui.separator();
                        for e in &extra {
                            if ui.button(&e.label).clicked() {
                                extra_chosen = Some(e.clone());
                            }
                        }
                    }
                });
            });
        // Clicks on other foreground layers (e.g. a popup of the sim UX menu) are not outside.
        let outside = ui.input(|i| {
            i.pointer.any_pressed()
                && i.pointer.interact_pos().is_some_and(|p| {
                    !area.response.rect.contains(p)
                        && ctx
                            .layer_id_at(p)
                            .is_none_or(|l| l.order != Order::Foreground)
                })
        });
        // The edit tool (earlier in the frame) consumes Escape; look at the raw events.
        let escape = ui.input(|i| {
            i.raw.events.iter().any(|e| {
                matches!(
                    e,
                    egui::Event::Key {
                        key: egui::Key::Escape,
                        pressed: true,
                        ..
                    }
                )
            })
        });
        if chosen.is_some() || extra_chosen.is_some() || outside || escape {
            self.open = None;
            ctx.data_mut(|d| d.insert_temp(open_id(), false));
        }
        if let Some(a) = chosen {
            run(&a, &mut ed, app, &open.target);
            super::publish_selection(&ctx, &ed.selection);
        }
        if let Some(e) = extra_chosen {
            drop(ed);
            (e.run)(app, &open.target);
        }
    }
}
