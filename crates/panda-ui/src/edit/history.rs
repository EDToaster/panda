//! Undo/redo: every editing action is one [`Txn`] recording the touched defs (structure and
//! geometry) before and after.
//!
//! The forward action applies fine-grained [`EditOp`]s (`AddNet`, `Connect`, …) through an
//! [`EditTarget`]; undo and redo restore the recorded states with one `ReplaceDef` per changed
//! def plus the recorded geometry, which is exact whatever the forward ops were.

use panda_kernel::EditOp;
use panda_netlist::{ComponentDef, DefId, Diagnostic};

use crate::app::AppState;
use crate::doc::{Document, EditorData};

/// Where edits go: the document, and (in the app) the kernel on the sim thread.
pub trait EditTarget {
    /// The document.
    fn doc(&self) -> &Document;
    /// The document, mutable (geometry edits; structure only through [`EditTarget::submit`]).
    fn doc_mut(&mut self) -> &mut Document;
    /// Apply structural edits to the document and send them to the kernel without waiting.
    /// Returns the kernel command id (if there is a kernel).
    fn submit(&mut self, ops: Vec<EditOp>) -> Result<Option<u64>, Diagnostic>;
}

impl EditTarget for Document {
    fn doc(&self) -> &Document {
        self
    }
    fn doc_mut(&mut self) -> &mut Document {
        self
    }
    fn submit(&mut self, ops: Vec<EditOp>) -> Result<Option<u64>, Diagnostic> {
        self.apply_ops(&ops).map(|()| None)
    }
}

impl EditTarget for AppState {
    fn doc(&self) -> &Document {
        &self.doc
    }
    fn doc_mut(&mut self) -> &mut Document {
        &mut self.doc
    }
    fn submit(&mut self, ops: Vec<EditOp>) -> Result<Option<u64>, Diagnostic> {
        self.submit_edit(ops).map(Some)
    }
}

/// Recorded state of one def.
#[derive(Clone, Debug, PartialEq)]
pub struct DefState {
    /// Structure (`None`: the def did not exist).
    pub def: Option<ComponentDef>,
    /// Geometry.
    pub geom: Option<EditorData>,
}

impl DefState {
    fn capture(doc: &Document, id: &str) -> DefState {
        DefState {
            def: doc.def(id).cloned(),
            geom: doc.editor(id).cloned(),
        }
    }
}

/// One undoable action.
#[derive(Clone, Debug)]
pub struct Txn {
    /// Short description ("wire", "move 3 items", …).
    pub label: String,
    /// `(def, before, after)` for every touched def.
    pub defs: Vec<(DefId, DefState, DefState)>,
    /// Kernel command id of the last submission of this transaction's structure (forward or
    /// redo); a kernel rejection of that id rolls the transaction back.
    pub kernel_id: Option<u64>,
}

impl Txn {
    /// True if the transaction changed structure (not only geometry).
    pub fn structural(&self) -> bool {
        self.defs.iter().any(|(_, b, a)| b.def != a.def)
    }
}

/// Undo and redo stacks.
#[derive(Debug)]
pub struct History {
    undo: Vec<Txn>,
    redo: Vec<Txn>,
    last_id: Option<u64>,
    /// Maximum number of undo steps kept.
    pub limit: usize,
}

impl Default for History {
    fn default() -> Self {
        History {
            undo: Vec::new(),
            redo: Vec::new(),
            last_id: None,
            limit: 500,
        }
    }
}

impl History {
    /// Undo steps available.
    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }

    /// Redo steps available.
    pub fn redo_len(&self) -> usize {
        self.redo.len()
    }

    /// Label of the action [`History::undo`] would revert.
    pub fn undo_label(&self) -> Option<&str> {
        self.undo.last().map(|t| t.label.as_str())
    }

    /// Label of the action [`History::redo`] would re-apply.
    pub fn redo_label(&self) -> Option<&str> {
        self.redo.last().map(|t| t.label.as_str())
    }

    /// Kernel command id of the most recent structural submission (action, undo or redo).
    pub fn last_kernel_id(&self) -> Option<u64> {
        self.last_id
    }

    /// Forget everything (e.g. a new document was opened).
    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }

    /// Run one action: snapshot `defs`, submit `ops` (structure), then apply `geom` to the
    /// document's geometry, and record the transaction. Nothing is recorded if nothing changed.
    /// On a local error nothing changes.
    pub fn commit(
        &mut self,
        t: &mut dyn EditTarget,
        label: impl Into<String>,
        defs: &[DefId],
        ops: Vec<EditOp>,
        geom: impl FnOnce(&mut Document),
    ) -> Result<(), String> {
        let before: Vec<DefState> = defs.iter().map(|d| DefState::capture(t.doc(), d)).collect();
        let kernel_id = if ops.is_empty() {
            None
        } else {
            t.submit(ops).map_err(|d| d.message)?
        };
        if kernel_id.is_some() {
            self.last_id = kernel_id;
        }
        geom(t.doc_mut());
        t.doc_mut().touch();
        let after: Vec<DefState> = defs.iter().map(|d| DefState::capture(t.doc(), d)).collect();
        if before == after {
            return Ok(());
        }
        self.undo.push(Txn {
            label: label.into(),
            defs: defs
                .iter()
                .cloned()
                .zip(before)
                .zip(after)
                .map(|((d, b), a)| (d, b, a))
                .collect(),
            kernel_id,
        });
        if self.undo.len() > self.limit {
            self.undo.remove(0);
        }
        self.redo.clear();
        Ok(())
    }

    /// Revert the last action. Returns its label.
    pub fn undo(&mut self, t: &mut dyn EditTarget) -> Result<Option<String>, String> {
        let Some(mut txn) = self.undo.pop() else {
            return Ok(None);
        };
        let states: Vec<(DefId, DefState)> = txn
            .defs
            .iter()
            .map(|(d, b, _)| (d.clone(), b.clone()))
            .collect();
        match restore(t, &states) {
            Ok(id) => {
                txn.kernel_id = id;
                if id.is_some() {
                    self.last_id = id;
                }
                let label = txn.label.clone();
                self.redo.push(txn);
                Ok(Some(label))
            }
            Err(e) => {
                self.undo.push(txn);
                Err(e)
            }
        }
    }

    /// Re-apply the last undone action. Returns its label.
    pub fn redo(&mut self, t: &mut dyn EditTarget) -> Result<Option<String>, String> {
        let Some(mut txn) = self.redo.pop() else {
            return Ok(None);
        };
        let states: Vec<(DefId, DefState)> = txn
            .defs
            .iter()
            .map(|(d, _, a)| (d.clone(), a.clone()))
            .collect();
        match restore(t, &states) {
            Ok(id) => {
                txn.kernel_id = id;
                if id.is_some() {
                    self.last_id = id;
                }
                let label = txn.label.clone();
                self.undo.push(txn);
                Ok(Some(label))
            }
            Err(e) => {
                self.redo.push(txn);
                Err(e)
            }
        }
    }

    /// The kernel rejected command `id`. If it belongs to a recorded action, roll the document
    /// back to the state before that action (undoing later actions too) and drop it from the
    /// history. Returns the label of the dropped action.
    pub fn rejected(&mut self, t: &mut dyn EditTarget, id: u64) -> Option<String> {
        let pos = self.undo.iter().position(|x| x.kernel_id == Some(id))?;
        while self.undo.len() > pos + 1 {
            self.undo(t).ok()?;
        }
        let txn = self.undo.pop()?;
        let states: Vec<(DefId, DefState)> = txn
            .defs
            .iter()
            .map(|(d, b, _)| (d.clone(), b.clone()))
            .collect();
        // The kernel never applied it, but later edits it accepted may depend on it: replacing
        // the defs re-synchronizes both sides.
        restore(t, &states).ok();
        self.redo.clear();
        Some(txn.label)
    }
}

/// Make the document (and kernel) match `states`: `ReplaceDef` for changed structure, then the
/// geometry.
fn restore(t: &mut dyn EditTarget, states: &[(DefId, DefState)]) -> Result<Option<u64>, String> {
    let mut ops = Vec::new();
    for (id, s) in states {
        let cur = t.doc().def(id);
        match (&s.def, cur) {
            (Some(want), Some(cur)) if want != cur => ops.push(EditOp::ReplaceDef(want.clone())),
            (Some(want), None) => ops.push(EditOp::AddDef(want.clone())),
            (None, Some(_)) => ops.push(EditOp::RemoveDef(id.clone())),
            _ => {}
        }
    }
    let kernel_id = if ops.is_empty() {
        None
    } else {
        t.submit(ops).map_err(|d| d.message)?
    };
    let doc = t.doc_mut();
    for (id, s) in states {
        match &s.geom {
            Some(g) => {
                doc.editors.insert(id.clone(), g.clone());
            }
            None => {
                doc.editors.remove(id);
            }
        }
    }
    doc.touch();
    Ok(kernel_id)
}
