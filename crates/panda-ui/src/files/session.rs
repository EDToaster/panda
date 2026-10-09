//! The GPU-free controller of the *File* menu: [`FileSession`].
//!
//! Every menu entry maps to one method here; the egui code in [`super::menu`] only draws and
//! forwards. Actions that would drop the open document ([`FileAction`]: new, open, example,
//! quit) first ask about unsaved changes when [`crate::doc::Document::dirty`] is set
//! ([`FileSession::confirm`], answered with [`FileSession::answer`]). *Save* on a document
//! without a path chains into *Save as*, and an action waiting for that save continues after it.
//!
//! Paths come from a [`PathPicker`] (native dialog, in-app prompt or a test script), answered
//! asynchronously and collected by [`FileSession::poll`] once per frame.
//!
//! ROM import builds the ROM def ([`super::build_rom`]), adds it with
//! [`AppState::submit_edit`] (`EditOp::AddDef`, applied by the sim thread between run chunks),
//! lays the new defs out on a worker thread and leaves [`FileSession::pending_place`] for the
//! menu to hand to the edit module's place tool ([`crate::edit::request_place`]); the placement
//! itself is an ordinary undoable edit.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, TryRecvError};

use panda_kernel::EditOp;
use panda_netlist::DefId;

use super::dialog::{PathPicker, PickPurpose, PickRequest};
use super::{
    RecentFiles, build_rom, new_design, read_document, rom_name, with_extension, write_document,
};
use crate::app::AppState;
use crate::doc::{Document, EditorData};
use crate::examples;

/// An action that replaces (or closes) the open document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FileAction {
    /// A new empty design.
    New,
    /// Open the document at this path.
    Open(PathBuf),
    /// Open built-in example `i` of [`examples::all`].
    Example(usize),
    /// Close the editor.
    Quit,
}

/// Answer to the unsaved-changes prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Confirm {
    /// Save first (Save as if the document has no path), then continue.
    Save,
    /// Drop the changes and continue.
    Discard,
    /// Keep the document; do nothing.
    Cancel,
}

/// State of the *File* menu (see the module docs).
pub struct FileSession {
    /// Recent documents (newest first).
    pub recent: RecentFiles,
    /// Action waiting for the unsaved-changes answer.
    pub confirm: Option<FileAction>,
    /// Error to show (cleared by the UI).
    pub error: Option<String>,
    /// Last message for the user.
    pub status: Option<String>,
    /// ROM def to hand to the place tool on the next frame.
    pub pending_place: Option<DefId>,
    /// Set once quitting is confirmed: the UI closes the window.
    pub close_now: bool,
    /// Set when [`FileSession::recent`] changed (the UI persists it and clears the flag).
    pub recent_changed: bool,
    picker: Box<dyn PathPicker>,
    after_save_as: Option<FileAction>,
    rom_layout: Option<Receiver<HashMap<DefId, EditorData>>>,
}

impl FileSession {
    /// A session asking `picker` for paths, with an empty recent-files list.
    pub fn new(picker: Box<dyn PathPicker>) -> FileSession {
        FileSession {
            recent: RecentFiles::in_memory(),
            confirm: None,
            error: None,
            status: None,
            pending_place: None,
            close_now: false,
            recent_changed: false,
            picker,
            after_save_as: None,
            rom_layout: None,
        }
    }

    /// The path picker (the UI draws its prompt, if any).
    pub fn picker_mut(&mut self) -> &mut dyn PathPicker {
        &mut *self.picker
    }

    /// True while a path is being asked for.
    pub fn picking(&self) -> bool {
        self.picker.busy()
    }

    /// True while a ROM layout runs on its worker thread.
    pub fn rom_layout_pending(&self) -> bool {
        self.rom_layout.is_some()
    }

    fn report(&mut self, app: &mut AppState, msg: String) {
        app.log.push(msg.clone());
        self.status = Some(msg);
    }

    fn fail(&mut self, app: &mut AppState, msg: String) {
        app.log.push(msg.clone());
        self.status = Some(msg.clone());
        self.error = Some(msg);
    }

    fn remember(&mut self, path: &Path) {
        self.recent.push(path);
        self.recent_changed = true;
    }

    /// Run `action`, or first ask about unsaved changes ([`FileSession::confirm`]).
    pub fn request(&mut self, app: &mut AppState, action: FileAction) {
        if app.doc.dirty {
            self.confirm = Some(action);
        } else {
            self.perform(app, action);
        }
    }

    /// Answer the unsaved-changes prompt.
    pub fn answer(&mut self, app: &mut AppState, c: Confirm) {
        let Some(action) = self.confirm.take() else {
            return;
        };
        match c {
            Confirm::Cancel => {}
            Confirm::Discard => self.perform(app, action),
            Confirm::Save => {
                if app.doc.path.is_some() {
                    if self.save(app) {
                        self.perform(app, action);
                    }
                } else {
                    self.after_save_as = Some(action);
                    self.save_as_dialog(app);
                }
            }
        }
    }

    /// Run `action` now, dropping unsaved changes.
    pub fn perform(&mut self, app: &mut AppState, action: FileAction) {
        match action {
            FileAction::New => {
                let mut doc = Document::new(new_design());
                doc.layout_missing();
                self.open(app, doc);
                self.report(app, "new design".into());
            }
            FileAction::Open(path) => {
                if let Err(e) = self.open_file(app, &path) {
                    self.fail(app, e);
                }
            }
            FileAction::Example(i) => {
                let all = examples::all();
                if let Some(e) = all.get(i) {
                    self.open(app, Document::new((e.build)()));
                    self.report(app, format!("opened example '{}'", e.name));
                }
            }
            FileAction::Quit => self.close_now = true,
        }
    }

    fn open(&mut self, app: &mut AppState, doc: Document) {
        self.pending_place = None;
        self.rom_layout = None;
        app.open_document(doc);
    }

    /// Open the document at `path` now (no unsaved-changes check) and remember it.
    pub fn open_file(&mut self, app: &mut AppState, path: &Path) -> Result<(), String> {
        match read_document(path) {
            Ok(doc) => {
                self.open(app, doc);
                self.remember(path);
                self.report(app, format!("opened {}", path.display()));
                Ok(())
            }
            Err(e) => {
                if !path.exists() && self.recent.paths.iter().any(|p| p == path) {
                    self.recent.remove(path);
                    self.recent_changed = true;
                }
                Err(e)
            }
        }
    }

    /// Write the document to `path` (gets [`super::EXTENSION`] unless it ends in `.json`) and
    /// remember it; clears the dirty flag.
    pub fn save_to(&mut self, app: &mut AppState, path: &Path) -> Result<PathBuf, String> {
        let path = with_extension(&path.to_string_lossy());
        write_document(&mut app.doc, &path)?;
        self.remember(&path);
        self.report(app, format!("saved {}", path.display()));
        Ok(path)
    }

    /// *Save*: to the document's path, or ask for one (*Save as*). True if it was written now.
    pub fn save(&mut self, app: &mut AppState) -> bool {
        let Some(path) = app.doc.path.clone() else {
            self.save_as_dialog(app);
            return false;
        };
        match self.save_to(app, &path) {
            Ok(_) => true,
            Err(e) => {
                self.fail(app, e);
                false
            }
        }
    }

    fn suggested_dir(&self, app: &AppState) -> Option<PathBuf> {
        app.doc
            .path
            .as_deref()
            .or(self.recent.paths.first().map(PathBuf::as_path))
            .and_then(Path::parent)
            .map(Path::to_path_buf)
    }

    /// *Open…*: ask for a path (the unsaved-changes check runs once it is chosen).
    pub fn open_dialog(&mut self, app: &AppState) {
        let suggested = self.suggested_dir(app);
        self.picker.start(PickRequest {
            purpose: PickPurpose::Open,
            suggested,
        });
    }

    /// *Save as…*: ask for a path, suggesting the current one (or `<top>.panda.json`).
    pub fn save_as_dialog(&mut self, app: &AppState) {
        let suggested = app.doc.path.clone().or_else(|| {
            let name = format!("{}{}", app.doc.design.top, super::EXTENSION);
            Some(
                self.suggested_dir(app)
                    .map_or(PathBuf::from(&name), |d| d.join(&name)),
            )
        });
        self.picker.start(PickRequest {
            purpose: PickPurpose::SaveAs,
            suggested,
        });
    }

    /// *Import ROM from hex…*: ask for a hex file.
    pub fn import_rom_dialog(&mut self, app: &AppState) {
        let suggested = self.suggested_dir(app);
        self.picker.start(PickRequest {
            purpose: PickPurpose::ImportRom,
            suggested,
        });
    }

    /// Import a ROM from hex text: add its defs to the document and the kernel, lay them out in
    /// the background and request its placement ([`FileSession::pending_place`]). Returns the
    /// ROM def id.
    pub fn import_rom(
        &mut self,
        app: &mut AppState,
        hex: &str,
        name: &str,
    ) -> Result<DefId, String> {
        let mut b = build_rom(hex, name, None)?;
        b.defs.retain(|d| app.doc.def(&d.id).is_none());
        if !b.defs.is_empty() {
            let ops = b.defs.iter().cloned().map(EditOp::AddDef).collect();
            app.submit_edit(ops).map_err(|d| d.to_string())?;
            let (tx, rx) = std::sync::mpsc::channel();
            let design = app.doc.design.clone();
            let editors = app.doc.editors.clone();
            let ids: Vec<DefId> = b.defs.iter().map(|d| d.id.clone()).collect();
            std::thread::Builder::new()
                .name("panda-rom-layout".into())
                .spawn(move || {
                    let _ = tx.send(crate::layout::layout_defs(&design, &editors, &ids));
                })
                .map_err(|e| e.to_string())?;
            self.rom_layout = Some(rx);
        }
        self.pending_place = Some(b.def.clone());
        let msg = format!(
            "imported ROM '{}' ({} words, {} address x {} data bits): click the canvas to place it",
            b.def, b.words, b.addr_bits, b.data_bits
        );
        self.report(app, msg);
        Ok(b.def)
    }

    /// [`FileSession::import_rom`] from a hex file (the ROM is named after the file).
    pub fn import_rom_file(&mut self, app: &mut AppState, path: &Path) -> Result<DefId, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        self.import_rom(app, &text, &rom_name(path))
            .map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Block until a background ROM layout finished and install it (tests, headless tools).
    pub fn wait_rom_layout(&mut self, app: &mut AppState) {
        if let Some(rx) = self.rom_layout.take()
            && let Ok(l) = rx.recv()
        {
            install_new(app, l);
        }
    }

    /// Per-frame: collect picker answers and finished ROM layouts. True if anything happened.
    pub fn poll(&mut self, app: &mut AppState) -> bool {
        let mut any = false;
        if let Some(rx) = &self.rom_layout {
            match rx.try_recv() {
                Ok(l) => {
                    install_new(app, l);
                    self.rom_layout = None;
                    any = true;
                }
                Err(TryRecvError::Disconnected) => self.rom_layout = None,
                Err(TryRecvError::Empty) => {}
            }
        }
        if let Some((req, path)) = self.picker.poll() {
            any = true;
            self.picked(app, req.purpose, path);
        }
        any
    }

    fn picked(&mut self, app: &mut AppState, purpose: PickPurpose, path: Option<PathBuf>) {
        let Some(path) = path else {
            // Cancelled: an action waiting for Save as is dropped too.
            self.after_save_as = None;
            return;
        };
        match purpose {
            PickPurpose::Open => self.request(app, FileAction::Open(path)),
            PickPurpose::SaveAs => match self.save_to(app, &path) {
                Ok(_) => {
                    if let Some(a) = self.after_save_as.take() {
                        self.perform(app, a);
                    }
                }
                Err(e) => {
                    self.after_save_as = None;
                    self.fail(app, e);
                }
            },
            PickPurpose::ImportRom => {
                if let Err(e) = self.import_rom_file(app, &path) {
                    self.fail(app, e);
                }
            }
        }
    }

    /// Window title: `Panda — name` with `*` when there are unsaved changes.
    pub fn title(app: &AppState) -> String {
        let name = app
            .doc
            .path
            .as_deref()
            .and_then(Path::file_name)
            .map_or_else(
                || app.doc.design.top.clone(),
                |n| n.to_string_lossy().into_owned(),
            );
        format!("Panda — {name}{}", if app.doc.dirty { " *" } else { "" })
    }
}

/// Install layouts for defs that still exist and have no geometry (the document may have been
/// replaced or edited while the worker ran).
fn install_new(app: &mut AppState, l: HashMap<DefId, EditorData>) {
    let keep: HashMap<DefId, EditorData> = l
        .into_iter()
        .filter(|(id, _)| app.doc.def(id).is_some() && app.doc.editor(id).is_none())
        .collect();
    app.doc.install_layouts(keep);
}
