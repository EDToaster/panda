//! The egui side of the *File* menu: [`FileMenu`] (a [`Slot::Menu`] panel), its keyboard
//! shortcuts (`Ctrl+N`, `Ctrl+O`, `Ctrl+S`, `Ctrl+Shift+S`), the unsaved-changes / error / path
//! modals, the window title (`*` when dirty), close-request interception, and persistence of the
//! recent-files list in egui's persisted memory (which eframe writes to its storage with the
//! `persistence` feature). All behaviour is in [`FileSession`].

use std::path::PathBuf;

use egui::{Context, Id, Key, KeyboardShortcut, Modifiers, RichText, Ui, ViewportCommand};

use super::RecentFiles;
use super::session::{Confirm, FileAction, FileSession};
use crate::app::{AppState, Panel, Slot};
use crate::edit::{PlaceRequest, request_place};
use crate::examples;

const NEW: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::N);
const OPEN: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::O);
const SAVE: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::S);
const SAVE_AS: KeyboardShortcut =
    KeyboardShortcut::new(Modifiers::COMMAND.plus(Modifiers::SHIFT), Key::S);

/// Key of the recent-files list in egui's persisted memory.
pub fn recent_id() -> Id {
    Id::new("panda.files.recent")
}

/// Read the persisted recent-files list from `ctx` (empty if none).
pub fn load_recent(ctx: &Context) -> RecentFiles {
    let paths = ctx
        .data_mut(|d| d.get_persisted::<Vec<PathBuf>>(recent_id()))
        .unwrap_or_default();
    RecentFiles::with_paths(paths)
}

/// Store the recent-files list in `ctx`'s persisted memory.
pub fn store_recent(ctx: &Context, r: &RecentFiles) {
    ctx.data_mut(|d| d.insert_persisted(recent_id(), r.paths.clone()));
}

/// The *File* menu panel.
pub struct FileMenu {
    /// The controller (public for tests and headless drivers).
    pub session: FileSession,
    recent_loaded: bool,
    title: String,
}

impl FileMenu {
    /// A menu driving `session`. The recent-files list is loaded from egui's persisted memory
    /// on the first frame (entries already in `session.recent` are kept in front).
    pub fn new(session: FileSession) -> FileMenu {
        FileMenu {
            session,
            recent_loaded: false,
            title: String::new(),
        }
    }

    fn shortcuts(&mut self, ctx: &Context, app: &mut AppState) {
        if self.session.confirm.is_some() || self.session.picker_mut().prompt().is_some() {
            return;
        }
        let (new, open, save_as, save) = ctx.input_mut(|i| {
            (
                i.consume_shortcut(&NEW),
                i.consume_shortcut(&OPEN),
                // Before Ctrl+S: `consume_shortcut` ignores extra Shift.
                i.consume_shortcut(&SAVE_AS),
                i.consume_shortcut(&SAVE),
            )
        });
        if new {
            self.session.request(app, FileAction::New);
        }
        if open {
            self.session.open_dialog(app);
        }
        if save_as {
            self.session.save_as_dialog(app);
        } else if save {
            self.session.save(app);
        }
    }

    fn modals(&mut self, ctx: &Context, app: &mut AppState) {
        let s = &mut self.session;
        if s.confirm.is_some() {
            let mut answer = None;
            let what = match &s.confirm {
                Some(FileAction::Quit) => "before closing",
                _ => "before continuing",
            };
            let r = egui::Modal::new(Id::new("panda.files.confirm")).show(ctx, |ui| {
                ui.set_width(360.0);
                ui.heading("Unsaved changes");
                ui.label(format!(
                    "Save the changes to '{}' {what}?",
                    app.doc
                        .path
                        .as_ref()
                        .map_or_else(|| app.doc.design.top.clone(), |p| p.display().to_string())
                ));
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() {
                        answer = Some(Confirm::Save);
                    }
                    if ui.button("Don't save").clicked() {
                        answer = Some(Confirm::Discard);
                    }
                    if ui.button("Cancel").clicked() {
                        answer = Some(Confirm::Cancel);
                    }
                });
            });
            if r.should_close() && answer.is_none() {
                answer = Some(Confirm::Cancel);
            }
            if let Some(a) = answer {
                s.answer(app, a);
            }
        }
        if let Some(p) = s.picker_mut().prompt() {
            let r = egui::Modal::new(Id::new("panda.files.prompt")).show(ctx, |ui| {
                ui.set_width(460.0);
                ui.heading(p.req.purpose.title());
                ui.horizontal(|ui| {
                    ui.label("Path");
                    let t = ui
                        .add(egui::TextEdit::singleline(&mut p.text).desired_width(f32::INFINITY));
                    if t.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
                        p.answer = Some(true);
                    } else if !t.has_focus() && p.answer.is_none() {
                        t.request_focus();
                    }
                });
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("OK").clicked() {
                        p.answer = Some(true);
                    }
                    if ui.button("Cancel").clicked() {
                        p.answer = Some(false);
                    }
                });
            });
            if r.should_close() && p.answer.is_none() {
                p.answer = Some(false);
            }
        }
        if let Some(e) = s.error.clone() {
            let r = egui::Modal::new(Id::new("panda.files.error")).show(ctx, |ui| {
                ui.set_width(420.0);
                ui.heading("File error");
                ui.label(RichText::new(e).monospace());
                ui.add_space(8.0);
                ui.button("OK").clicked()
            });
            if r.inner || r.should_close() {
                s.error = None;
            }
        }
    }
}

impl Panel for FileMenu {
    fn id(&self) -> &'static str {
        "files.menu"
    }

    fn title(&self) -> String {
        "File".into()
    }

    fn slot(&self) -> Slot {
        Slot::Menu
    }

    fn ui(&mut self, ui: &mut Ui, app: &mut AppState) {
        let ctx = ui.ctx().clone();
        let s = &mut self.session;
        let sc = |k: &KeyboardShortcut| ctx.format_shortcut(k);
        if ui
            .add(egui::Button::new("New").shortcut_text(sc(&NEW)))
            .clicked()
        {
            s.request(app, FileAction::New);
            ui.close();
        }
        if ui
            .add(egui::Button::new("Open…").shortcut_text(sc(&OPEN)))
            .clicked()
        {
            s.open_dialog(app);
            ui.close();
        }
        ui.menu_button("Open recent", |ui| {
            if s.recent.paths.is_empty() {
                ui.label(RichText::new("(none)").weak());
            }
            for p in s.recent.paths.clone() {
                let label = p.file_name().map_or_else(
                    || p.display().to_string(),
                    |n| n.to_string_lossy().into_owned(),
                );
                if ui
                    .button(label)
                    .on_hover_text(p.display().to_string())
                    .clicked()
                {
                    s.request(app, FileAction::Open(p));
                    ui.close();
                }
            }
            if !s.recent.paths.is_empty() {
                ui.separator();
                if ui.button("Clear list").clicked() {
                    s.recent.paths.clear();
                    s.recent_changed = true;
                    ui.close();
                }
            }
        });
        ui.separator();
        if ui
            .add(egui::Button::new("Save").shortcut_text(sc(&SAVE)))
            .clicked()
        {
            s.save(app);
            ui.close();
        }
        if ui
            .add(egui::Button::new("Save as…").shortcut_text(sc(&SAVE_AS)))
            .clicked()
        {
            s.save_as_dialog(app);
            ui.close();
        }
        ui.separator();
        if ui.button("Import ROM from hex…").clicked() {
            s.import_rom_dialog(app);
            ui.close();
        }
        ui.menu_button("Examples", |ui| {
            for (i, e) in examples::all().iter().enumerate() {
                if ui.button(e.name).on_hover_text(e.description).clicked() {
                    s.request(app, FileAction::Example(i));
                    ui.close();
                }
            }
        });
        ui.separator();
        if ui.button("Quit").clicked() {
            s.request(app, FileAction::Quit);
            ui.close();
        }
    }

    fn on_frame(&mut self, ctx: &Context, app: &mut AppState) {
        if !self.recent_loaded {
            self.recent_loaded = true;
            let stored = load_recent(ctx);
            let mine = std::mem::take(&mut self.session.recent.paths);
            self.session.recent = RecentFiles::with_paths(mine.into_iter().chain(stored.paths));
        }
        self.shortcuts(ctx, app);
        if self.session.poll(app) {
            ctx.request_repaint();
        }
        if self.session.picking() || self.session.rom_layout_pending() {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
        if let Some(def) = self.session.pending_place.take() {
            request_place(ctx, PlaceRequest::Def(def));
        }
        if ctx.input(|i| i.viewport().close_requested()) && !self.session.close_now {
            if app.doc.dirty {
                ctx.send_viewport_cmd(ViewportCommand::CancelClose);
                self.session.confirm = Some(FileAction::Quit);
            } else {
                self.session.close_now = true;
            }
        }
        self.modals(ctx, app);
        if self.session.close_now {
            ctx.send_viewport_cmd(ViewportCommand::Close);
        }
        if self.session.recent_changed {
            self.session.recent_changed = false;
            store_recent(ctx, &self.session.recent);
        }
        let title = FileSession::title(app);
        if title != self.title {
            ctx.send_viewport_cmd(ViewportCommand::Title(title.clone()));
            self.title = title;
        }
    }
}
