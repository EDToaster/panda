//! The "Edit" and "Place" menus (top bar): tools, undo/redo, clipboard, rotate/flip/delete,
//! and placing atoms, pins and packages.

use egui::Ui;
use panda_netlist::Atom;

use super::SharedEditor;
use super::actions::{PlaceRequest, Tool, instantiates};
use crate::app::{AppState, Panel, Slot};

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
