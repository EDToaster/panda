//! Undoable document changes made by the hierarchy/files features.
//!
//! A [`DocChange`] is a structural edit as kernel [`EditOp`]s in both directions plus the
//! geometry of the touched defs before and after. Applying it goes through
//! [`AppState::submit_edit`] (document and sim thread in one step, the sim thread applies it
//! between run chunks), so it never blocks on compilation.
//!
//! **Undo hook.** [`History`] is this module's undo/redo stack (the *Hierarchy* menu's
//! Undo/Redo). An editor-wide undo stack can adopt these changes as they are: undo =
//! [`DocChange::revert`], redo = [`DocChange::apply`]; every change made here is also offered
//! through [`History::take_new`] so such a stack can pick them up.

use panda_kernel::EditOp;
use panda_netlist::{DefId, Diagnostic};

use crate::app::AppState;
use crate::doc::{Document, EditorData};

/// One undoable change.
#[derive(Clone, Debug, PartialEq)]
pub struct DocChange {
    /// Human-readable description ("Create package alu", "Place mux2_8").
    pub label: String,
    /// Forward edit.
    pub redo: Vec<EditOp>,
    /// Inverse edit.
    pub undo: Vec<EditOp>,
    /// Geometry of the touched defs before (`None` = no geometry).
    pub geom_before: Vec<(DefId, Option<EditorData>)>,
    /// Geometry of the touched defs after.
    pub geom_after: Vec<(DefId, Option<EditorData>)>,
}

impl DocChange {
    /// Apply forward to the app (document + kernel).
    pub fn apply(&self, app: &mut AppState) -> Result<u64, Diagnostic> {
        let id = app.submit_edit(self.redo.clone())?;
        install(&mut app.doc, &self.geom_after);
        Ok(id)
    }

    /// Apply backward to the app (document + kernel).
    pub fn revert(&self, app: &mut AppState) -> Result<u64, Diagnostic> {
        let id = app.submit_edit(self.undo.clone())?;
        install(&mut app.doc, &self.geom_before);
        Ok(id)
    }

    /// Apply forward to a bare document (tests, headless tools).
    pub fn apply_doc(&self, doc: &mut Document) -> Result<(), Diagnostic> {
        doc.apply_ops(&self.redo)?;
        install(doc, &self.geom_after);
        Ok(())
    }

    /// Apply backward to a bare document.
    pub fn revert_doc(&self, doc: &mut Document) -> Result<(), Diagnostic> {
        doc.apply_ops(&self.undo)?;
        install(doc, &self.geom_before);
        Ok(())
    }
}

fn install(doc: &mut Document, geom: &[(DefId, Option<EditorData>)]) {
    for (id, e) in geom {
        match e {
            Some(e) => {
                doc.editors.insert(id.clone(), e.clone());
            }
            None => {
                doc.editors.remove(id);
            }
        }
    }
    doc.touch();
}

impl From<&super::package::PackagePlan> for DocChange {
    fn from(p: &super::package::PackagePlan) -> DocChange {
        DocChange {
            label: format!("Create package {}", p.package.id),
            redo: p.ops(),
            undo: p.undo_ops(),
            geom_before: vec![
                (p.parent_before.id.clone(), p.parent_editor_before.clone()),
                (p.package.id.clone(), None),
            ],
            geom_after: vec![
                (p.parent_after.id.clone(), p.parent_editor_after.clone()),
                (p.package.id.clone(), Some(p.package_editor.clone())),
            ],
        }
    }
}

/// Undo/redo stack of [`DocChange`]s.
#[derive(Default)]
pub struct History {
    undo: Vec<DocChange>,
    redo: Vec<DocChange>,
    fresh: Vec<DocChange>,
}

impl History {
    /// Apply `c` and record it. On error nothing is recorded.
    pub fn apply(&mut self, app: &mut AppState, c: DocChange) -> Result<(), Diagnostic> {
        c.apply(app)?;
        self.redo.clear();
        if self.fresh.len() >= 64 {
            self.fresh.remove(0);
        }
        self.fresh.push(c.clone());
        self.undo.push(c);
        Ok(())
    }

    /// Undo the last change. Returns its label.
    pub fn undo(&mut self, app: &mut AppState) -> Option<Result<String, Diagnostic>> {
        let c = self.undo.pop()?;
        Some(match c.revert(app) {
            Ok(_) => {
                let l = c.label.clone();
                self.redo.push(c);
                Ok(l)
            }
            Err(e) => {
                // The document moved on in a way this change cannot be undone over.
                self.undo.clear();
                Err(e)
            }
        })
    }

    /// Redo the last undone change. Returns its label.
    pub fn redo(&mut self, app: &mut AppState) -> Option<Result<String, Diagnostic>> {
        let c = self.redo.pop()?;
        Some(match c.apply(app) {
            Ok(_) => {
                let l = c.label.clone();
                self.undo.push(c);
                Ok(l)
            }
            Err(e) => {
                self.redo.clear();
                Err(e)
            }
        })
    }

    /// Label of the change [`History::undo`] would revert.
    pub fn undo_label(&self) -> Option<&str> {
        self.undo.last().map(|c| c.label.as_str())
    }

    /// Label of the change [`History::redo`] would apply.
    pub fn redo_label(&self) -> Option<&str> {
        self.redo.last().map(|c| c.label.as_str())
    }

    /// Changes recorded since the last call (for an editor-wide undo stack).
    pub fn take_new(&mut self) -> Vec<DocChange> {
        std::mem::take(&mut self.fresh)
    }

    /// Forget everything (a new document was opened).
    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
        self.fresh.clear();
    }
}
