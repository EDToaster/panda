//! The hierarchy actions on [`AppState`] (GPU-free; the panels in [`super::ui`] only call
//! these). Shared state between the panels and the canvas tool is [`HierState`].

use panda_netlist::DefId;

use super::change::DocChange;
use super::classify::Classifier;
use super::library::{LibEntry, Placeable, add_defs_change, builtin_entries};
use super::nav::{child_path, parent_path, valid_view};
use super::package::{fresh_def_id, plan_package};
use crate::app::AppState;
use crate::doc::ItemRef;
use crate::edit::{EditTarget, History, PlaceRequest, SharedEditor};

/// The open "create package" prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackagePrompt {
    /// Package (def) name being typed.
    pub name: String,
    /// Instances to package (captured when the prompt opened).
    pub selection: Vec<String>,
    /// Error of the last attempt.
    pub error: Option<String>,
    /// Select the whole name on the next frame (just opened: typing replaces the default).
    pub select_all: bool,
}

/// The open "rename package" prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenamePrompt {
    /// Package being renamed.
    pub old: DefId,
    /// New name being typed.
    pub name: String,
    /// Error of the last attempt.
    pub error: Option<String>,
    /// Select the whole name on the next frame.
    pub select_all: bool,
}

/// State shared by the hierarchy panels and tools.
///
/// Changes are recorded in the editor-wide undo history ([`crate::edit::shared_editor`]) when
/// one is given (the app: Ctrl+Z covers them in order with the editing actions), else in
/// [`HierState::history`] (headless use).
pub struct HierState {
    /// Fallback undo/redo when there is no shared editor (same type as the editor's).
    pub history: History,
    /// Kernel command ids of fallback-history changes not yet answered by the sim thread.
    pub in_flight: Vec<u64>,
    /// Classification badges.
    pub classifier: Classifier,
    /// Built-in library entries (parameters edited in the palette).
    pub builtin: Vec<LibEntry>,
    /// Open package prompt.
    pub prompt: Option<PackagePrompt>,
    /// Open rename-package prompt.
    pub rename: Option<RenamePrompt>,
    /// Library search text.
    pub filter: String,
    /// Selected library entry (name), whose parameters are shown.
    pub selected_entry: Option<String>,
    /// Last view seen (to fit the canvas when it changes).
    pub last_view: Option<(String, DefId)>,
    /// `(top def, revision)` of the document last seen (a new document clears the history).
    pub last_doc: Option<(DefId, u64)>,
    /// Last message for the user.
    pub status: Option<String>,
}

impl Default for HierState {
    fn default() -> Self {
        HierState::new()
    }
}

impl HierState {
    /// Fresh state (starts the classification worker).
    pub fn new() -> HierState {
        HierState {
            history: History::default(),
            in_flight: Vec::new(),
            classifier: Classifier::new(),
            builtin: builtin_entries(),
            prompt: None,
            rename: None,
            filter: String::new(),
            selected_entry: None,
            last_view: None,
            last_doc: None,
            status: None,
        }
    }

    fn report(&mut self, app: &mut AppState, msg: String) {
        app.log.push(msg.clone());
        self.status = Some(msg);
    }

    /// Record `c` (apply it to the document and kernel) as one undoable action of `ed`'s
    /// history, or of the fallback history without an editor.
    pub fn commit(
        &mut self,
        ed: Option<&SharedEditor>,
        app: &mut AppState,
        c: &DocChange,
    ) -> Result<(), String> {
        match ed {
            // The edit module rolls back kernel rejections of its own history's actions.
            Some(ed) => c.commit(&mut ed.borrow_mut().history, app).map(|_| ()),
            None => {
                if let Some(id) = c.commit(&mut self.history, app)? {
                    self.in_flight.push(id);
                }
                Ok(())
            }
        }
    }

    /// Per-frame bookkeeping: forget the history when another document was opened, keep the
    /// view on an existing def after structural changes (e.g. undoing the package the view
    /// was in).
    pub fn sync(&mut self, app: &mut AppState) {
        let top = app.doc.design.top.clone();
        let rev = app.doc.revision;
        let replaced = self
            .last_doc
            .as_ref()
            .is_some_and(|(t, r)| *t != top || rev < *r);
        if replaced {
            self.history.clear();
            self.in_flight.clear();
            self.prompt = None;
        }
        self.last_doc = Some((top, rev));
        if let Some(path) = valid_view(&app.doc.design, &app.view.path, &app.view.def) {
            app.enter(&path);
        }
    }

    /// The kernel answered command `id`. A rejected change of the fallback history is rolled
    /// back (document and kernel). Returns the label of a rolled-back change.
    pub fn edited(&mut self, app: &mut AppState, id: u64, ok: bool) -> Option<String> {
        let pos = self.in_flight.iter().position(|&x| x == id)?;
        self.in_flight.remove(pos);
        if ok {
            return None;
        }
        let label = self.history.rejected(app, id)?;
        self.report(
            app,
            format!("'{label}' was rejected by the kernel and rolled back"),
        );
        Some(label)
    }

    /// Open the package prompt for `selection` with a free default name.
    pub fn ask_package(&mut self, app: &AppState, selection: Vec<String>) {
        self.prompt = Some(PackagePrompt {
            name: default_package_name(app),
            selection,
            error: None,
            select_all: true,
        });
    }

    /// Open the rename prompt for package `id` (its current name, selected).
    pub fn ask_rename(&mut self, app: &AppState, id: &str) {
        if app.doc.def(id).is_none() || id == app.doc.design.top {
            self.status = Some(format!("'{id}' is not a package of this design"));
            return;
        }
        self.rename = Some(RenamePrompt {
            old: id.to_string(),
            name: id.to_string(),
            error: None,
            select_all: true,
        });
    }

    /// Apply the open rename prompt. On error the prompt stays open with the message.
    pub fn submit_rename(&mut self, ed: Option<&SharedEditor>, app: &mut AppState) {
        let Some(mut p) = self.rename.take() else {
            return;
        };
        let new = p.name.trim().to_string();
        if new == p.old {
            return;
        }
        let r = super::manage::plan_rename_def(EditTarget::doc(app), &p.old, &new)
            .and_then(|c| self.commit(ed, app, &c));
        match r {
            Ok(()) => {
                if app.view.def == p.old {
                    app.view.def = new.clone();
                    self.last_view = Some((app.view.path.clone(), new.clone()));
                }
                if self.selected_entry.as_deref() == Some(p.old.as_str()) {
                    self.selected_entry = Some(new.clone());
                }
                self.report(app, format!("renamed package '{}' to '{new}'", p.old));
            }
            Err(e) => {
                p.error = Some(e);
                self.rename = Some(p);
            }
        }
    }

    /// Delete package `id` (refused, with the reason in the status line, while it is used).
    pub fn delete_def(&mut self, ed: Option<&SharedEditor>, app: &mut AppState, id: &str) {
        let r = super::manage::plan_delete_def(EditTarget::doc(app), id)
            .and_then(|c| self.commit(ed, app, &c));
        match r {
            Ok(()) => {
                if self.selected_entry.as_deref() == Some(id) {
                    self.selected_entry = None;
                }
                self.report(app, format!("deleted package '{id}' (Ctrl+Z to undo)"));
            }
            Err(e) => self.status = Some(e),
        }
    }

    /// Create the package of the open prompt. On error the prompt stays open with the message.
    pub fn submit_prompt(&mut self, ed: Option<&SharedEditor>, app: &mut AppState) {
        let Some(mut p) = self.prompt.take() else {
            return;
        };
        if let Err(e) = create_package(self, ed, app, &p.selection, &p.name) {
            p.error = Some(e);
            self.prompt = Some(p);
        }
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

/// Go to the enclosing instance; the instance just left is highlighted there. False at the top.
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

/// Create a package named `name` from the instances `sel` of the viewed def (one undoable
/// action) and highlight the new instance. Returns the instance id.
pub fn create_package(
    hs: &mut HierState,
    ed: Option<&SharedEditor>,
    app: &mut AppState,
    sel: &[String],
    name: &str,
) -> Result<String, String> {
    let def = app.view.def.clone();
    let plan = plan_package(EditTarget::doc(app), &def, sel, name.trim())?;
    let inst = plan.instance.clone();
    hs.commit(ed, app, &DocChange::from(&plan))?;
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

/// Prepare placing library entry `e` into the viewed def: blocks first get their missing defs
/// added (an undoable action), then the edit module's place tool takes over. Returns the
/// request for [`crate::edit::request_place`].
pub fn arm_entry(
    hs: &mut HierState,
    ed: Option<&SharedEditor>,
    app: &mut AppState,
    e: &LibEntry,
) -> Result<PlaceRequest, String> {
    if !e.allowed_in(&app.doc.design, &app.view.def) {
        return Err(format!(
            "'{}' cannot be placed into '{}'",
            e.name, app.view.def
        ));
    }
    match e.resolve(&app.doc.design)? {
        Placeable::Atom(a) => Ok(PlaceRequest::Atom(a)),
        Placeable::Def { id, new_defs } => {
            if let Some(c) = add_defs_change(&app.doc, &new_defs, &format!("Add {id}")) {
                hs.commit(ed, app, &c)?;
            }
            hs.status = Some(format!("placing {id}: click the canvas"));
            Ok(PlaceRequest::Def(id))
        }
    }
}

/// Undo the last change of the fallback history (in the app, the Edit menu / Ctrl+Z undo
/// hierarchy changes together with the editing actions).
pub fn undo(hs: &mut HierState, app: &mut AppState) {
    let msg = match hs.history.undo(app) {
        Ok(Some(l)) => format!("undid: {l}"),
        Ok(None) => return,
        Err(e) => format!("cannot undo: {e}"),
    };
    track(hs);
    hs.report(app, msg);
}

/// Redo the last undone change of the fallback history.
pub fn redo(hs: &mut HierState, app: &mut AppState) {
    let msg = match hs.history.redo(app) {
        Ok(Some(l)) => format!("redid: {l}"),
        Ok(None) => return,
        Err(e) => format!("cannot redo: {e}"),
    };
    track(hs);
    hs.report(app, msg);
}

fn track(hs: &mut HierState) {
    if let Some(id) = hs.history.last_kernel_id()
        && !hs.in_flight.contains(&id)
    {
        hs.in_flight.push(id);
    }
}
