//! The hierarchy/files actions on [`AppState`] (GPU-free; the panels in [`super::ui`] only call
//! these). Shared state between the panels and tools is [`HierState`].

use std::path::Path;

use panda_netlist::DefId;

use super::change::History;
use super::classify::Classifier;
use super::library::{EntryKind, LibEntry, Placeable, add_defs_change, builtin_entries};
use super::nav::{child_path, parent_path};
use super::package::{fresh_def_id, plan_package};
use crate::app::AppState;
use crate::doc::{Document, ItemRef};
use crate::edit::PlaceRequest;
use crate::examples::Example;
use crate::files::{self, RecentFiles};

/// What an open path prompt will do with the text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptKind {
    /// Open a document.
    Open,
    /// Save under a new path.
    SaveAs,
    /// Import a ROM from a hex file.
    ImportRom,
    /// Name of a package created from the selection.
    PackageName,
}

impl PromptKind {
    /// Dialog title.
    pub fn title(self) -> &'static str {
        match self {
            PromptKind::Open => "Open design",
            PromptKind::SaveAs => "Save design as",
            PromptKind::ImportRom => "Import ROM from hex file",
            PromptKind::PackageName => "Create package from selection",
        }
    }

    /// Label of the text field.
    pub fn field(self) -> &'static str {
        match self {
            PromptKind::PackageName => "Package (def) name",
            PromptKind::ImportRom => "Hex file",
            _ => "Path",
        }
    }
}

/// A modal text prompt (paths, package names).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prompt {
    /// Purpose.
    pub kind: PromptKind,
    /// Current text.
    pub text: String,
    /// Error of the last attempt.
    pub error: Option<String>,
}

/// State shared by the hierarchy/files panels and tools.
pub struct HierState {
    /// Undo/redo of hierarchy changes.
    pub history: History,
    /// Classification badges.
    pub classifier: Classifier,
    /// Recent documents.
    pub recent: RecentFiles,
    /// Built-in library entries (parameters edited in the palette).
    pub builtin: Vec<LibEntry>,
    /// Placement to hand to the edit module's place tool ([`crate::edit::request_place`]) on
    /// the next frame (set by GPU-free actions such as ROM import).
    pub pending_place: Option<PlaceRequest>,
    /// Selection captured when the package-name prompt opened.
    pub package_selection: Vec<String>,
    /// Open prompt.
    pub prompt: Option<Prompt>,
    /// Library search text.
    pub filter: String,
    /// Selected library entry (name), whose parameters are shown.
    pub selected_entry: Option<String>,
    /// Last view seen (to fit the canvas when it changes).
    pub last_view: Option<(String, DefId)>,
    /// Last message for the user.
    pub status: Option<String>,
}

impl HierState {
    /// State with the recent-files list stored in `recent`.
    pub fn new(recent: RecentFiles) -> HierState {
        HierState {
            history: History::default(),
            classifier: Classifier::new(),
            recent,
            builtin: builtin_entries(),
            pending_place: None,
            package_selection: Vec::new(),
            prompt: None,
            filter: String::new(),
            selected_entry: None,
            last_view: None,
            status: None,
        }
    }

    /// State using the user's recent-files list ([`RecentFiles::default_store`]).
    pub fn with_default_store() -> HierState {
        HierState::new(
            RecentFiles::default_store()
                .map(RecentFiles::load)
                .unwrap_or_default(),
        )
    }

    fn report(&mut self, app: &mut AppState, msg: String) {
        app.log.push(msg.clone());
        self.status = Some(msg);
    }

    /// Open a prompt.
    pub fn ask(&mut self, kind: PromptKind, text: String) {
        self.prompt = Some(Prompt {
            kind,
            text,
            error: None,
        });
    }
}

/// Instance ids among selected items (pin markers are ignored), sorted.
pub fn instance_ids<'a>(items: impl IntoIterator<Item = &'a ItemRef>) -> Vec<String> {
    let mut v: Vec<String> = items
        .into_iter()
        .filter_map(|i| match i {
            ItemRef::Instance(id) => Some(id.clone()),
            ItemRef::Pin(_) => None,
        })
        .collect();
    v.sort();
    v.dedup();
    v
}

/// Instance ids among the canvas highlights ([`crate::app::AppState::highlights`] items; the
/// UI prefers the edit module's selection, [`crate::edit::current_selection`]).
pub fn selected_instances(app: &AppState) -> Vec<String> {
    instance_ids(&app.highlights.items)
}

/// Enter package instance `inst` of the current view. False if it is not a package instance.
pub fn enter_instance(app: &mut AppState, inst: &str) -> bool {
    let path = child_path(&app.view.path, inst);
    let ok = app.enter(&path);
    if ok {
        app.canvas.request_fit();
    }
    ok
}

/// Go to the enclosing instance; the instance just left is selected there. False at the top.
pub fn exit(app: &mut AppState) -> bool {
    let Some(parent) = parent_path(&app.view.path) else {
        return false;
    };
    let left = app
        .view
        .path
        .rsplit('/')
        .next()
        .unwrap_or_default()
        .to_string();
    let ok = app.enter(&parent);
    if ok {
        app.highlights.items.insert(ItemRef::Instance(left));
        app.canvas.request_fit();
    }
    ok
}

/// Create a package named `name` from the instances `sel` of the viewed def and select the
/// new instance. Returns the instance id.
pub fn create_package(
    hs: &mut HierState,
    app: &mut AppState,
    sel: &[String],
    name: &str,
) -> Result<String, String> {
    let def = app.view.def.clone();
    let plan = plan_package(&app.doc, &def, sel, name.trim())?;
    let inst = plan.instance.clone();
    hs.history
        .apply(app, (&plan).into())
        .map_err(|d| d.to_string())?;
    app.highlights.items.clear();
    app.highlights.items.insert(ItemRef::Instance(inst.clone()));
    let msg = format!(
        "created package '{}' ({} pins, {} instances) as '{inst}'",
        plan.package.id,
        plan.pins.len(),
        plan.package.instances.len()
    );
    hs.report(app, msg);
    Ok(inst)
}

/// A free default name for a new package.
pub fn default_package_name(app: &AppState) -> String {
    fresh_def_id(&app.doc.design, "pkg")
}

/// Prepare placing library entry `e` into the viewed def: blocks first get their defs added
/// (an undoable change), then the edit module's place tool takes over. Returns the request
/// for [`crate::edit::request_place`].
pub fn arm_entry(hs: &mut HierState, app: &mut AppState, e: &LibEntry) -> Result<PlaceRequest, String> {
    if !e.allowed_in(&app.doc.design, &app.view.def) {
        return Err(format!("'{}' cannot be placed into '{}'", e.name, app.view.def));
    }
    match e.resolve(&app.doc.design)? {
        Placeable::Atom(a) => Ok(PlaceRequest::Atom(a)),
        Placeable::Def { id, new_defs } => {
            if let Some(c) = add_defs_change(&app.doc, &new_defs, &format!("Add {}", e.name)) {
                hs.history.apply(app, c).map_err(|d| d.to_string())?;
            }
            if matches!(e.kind, EntryKind::Block(_)) {
                hs.status = Some(format!("placing {id}"));
            }
            Ok(PlaceRequest::Def(id))
        }
    }
}

/// Undo the last hierarchy change.
pub fn undo(hs: &mut HierState, app: &mut AppState) {
    if let Some(r) = hs.history.undo(app) {
        let msg = match r {
            Ok(l) => format!("undid: {l}"),
            Err(e) => format!("cannot undo: {e}"),
        };
        hs.report(app, msg);
    }
}

/// Redo the last undone hierarchy change.
pub fn redo(hs: &mut HierState, app: &mut AppState) {
    if let Some(r) = hs.history.redo(app) {
        let msg = match r {
            Ok(l) => format!("redid: {l}"),
            Err(e) => format!("cannot redo: {e}"),
        };
        hs.report(app, msg);
    }
}

fn open_doc(hs: &mut HierState, app: &mut AppState, doc: Document) {
    hs.history.clear();
    hs.pending_place = None;
    hs.last_view = None;
    app.open_document(doc);
}

/// New empty document.
pub fn new_document(hs: &mut HierState, app: &mut AppState) {
    let mut doc = Document::new(files::new_design());
    doc.layout_missing();
    open_doc(hs, app, doc);
    hs.report(app, "new design".into());
}

/// Open a built-in example (laid out on a worker thread by the shell).
pub fn open_example(hs: &mut HierState, app: &mut AppState, e: &Example) {
    open_doc(hs, app, Document::new((e.build)()));
    hs.report(app, format!("opened example '{}'", e.name));
}

/// Open the document at `path` and remember it in the recent files.
pub fn open_file(hs: &mut HierState, app: &mut AppState, path: &Path) -> Result<(), String> {
    match files::read_document(path) {
        Ok(doc) => {
            open_doc(hs, app, doc);
            hs.recent.push(path);
            hs.report(app, format!("opened {}", path.display()));
            Ok(())
        }
        Err(e) => {
            if !path.exists() {
                hs.recent.remove(path);
            }
            Err(e)
        }
    }
}

/// Save to `path` (or the document's own path). Errors if there is neither.
pub fn save(hs: &mut HierState, app: &mut AppState, path: Option<&Path>) -> Result<(), String> {
    let path = path
        .map(Path::to_path_buf)
        .or_else(|| app.doc.path.clone())
        .ok_or_else(|| "no file name yet: use Save As".to_string())?;
    files::write_document(&mut app.doc, &path)?;
    hs.recent.push(&path);
    hs.report(app, format!("saved {}", path.display()));
    Ok(())
}

/// Import a ROM from a hex file as a new def and request its placement.
pub fn import_rom_file(hs: &mut HierState, app: &mut AppState, path: &Path) -> Result<DefId, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let imp = files::import_rom(&app.doc, &text, &files::rom_name(path), None)?;
    hs.history
        .apply(app, imp.change.clone())
        .map_err(|d| d.to_string())?;
    hs.pending_place = Some(PlaceRequest::Def(imp.def.clone()));
    let msg = format!(
        "imported ROM '{}' ({} words, {}x{} bits): click the canvas to place it",
        imp.def, imp.words, imp.addr_bits, imp.data_bits
    );
    hs.report(app, msg);
    Ok(imp.def)
}

/// Carry out the open prompt with its text. On error the prompt stays open with the message.
pub fn submit_prompt(hs: &mut HierState, app: &mut AppState) {
    let Some(p) = hs.prompt.take() else { return };
    let text = p.text.trim().to_string();
    let r = match p.kind {
        PromptKind::Open => open_file(hs, app, Path::new(&text)),
        PromptKind::SaveAs => save(hs, app, Some(&files::with_extension(&text))),
        PromptKind::ImportRom => import_rom_file(hs, app, Path::new(&text)).map(|_| ()),
        PromptKind::PackageName => {
            let sel = std::mem::take(&mut hs.package_selection);
            let r = create_package(hs, app, &sel, &text).map(|_| ());
            if r.is_err() {
                hs.package_selection = sel;
            }
            r
        }
    };
    if let Err(e) = r {
        hs.prompt = Some(Prompt {
            error: Some(e),
            ..p
        });
    }
}
