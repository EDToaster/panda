//! Undoable document changes made by the hierarchy features (packaging, adding library defs).
//!
//! A [`DocChange`] is a structural edit as kernel [`EditOp`]s in both directions plus the
//! geometry of the touched defs before and after. In the app it is recorded through the edit
//! module's undo API ([`crate::edit::History::commit`], see [`DocChange::commit`]): the
//! structure goes through [`crate::edit::EditTarget::submit`] (document at once, kernel on the
//! sim thread between run chunks, so nothing waits for compilation) and undo / redo restore the
//! recorded def states exactly. [`DocChange::apply_doc`] / [`DocChange::revert_doc`] apply it to
//! a bare [`Document`] (tests, headless tools).

use panda_kernel::EditOp;
use panda_netlist::{DefId, Diagnostic};

use crate::doc::{Document, EditorData};
use crate::edit::{EditTarget, History};

/// One undoable change.
#[derive(Clone, Debug, PartialEq)]
pub struct DocChange {
    /// Human-readable description ("Create package alu", "Add mux2_8").
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
    /// Ids of every def the change touches (structure or geometry), first-seen order.
    pub fn touched(&self) -> Vec<DefId> {
        let mut v: Vec<DefId> = Vec::new();
        for (id, _) in self.geom_before.iter().chain(&self.geom_after) {
            if !v.contains(id) {
                v.push(id.clone());
            }
        }
        v
    }

    /// Apply forward to `t` and record it as one undoable action in `history` (the edit
    /// module's undo/redo). Returns the kernel command id of the submission, if any.
    pub fn commit(
        &self,
        history: &mut History,
        t: &mut dyn EditTarget,
    ) -> Result<Option<u64>, String> {
        let before = history.undo_len();
        history.commit(
            t,
            self.label.clone(),
            &self.touched(),
            self.redo.clone(),
            |doc| {
                install(doc, &self.geom_after);
            },
        )?;
        Ok(if history.undo_len() > before {
            history.last_kernel_id()
        } else {
            None
        })
    }

    /// Apply forward to a bare document (tests, headless tools).
    pub fn apply_doc(&self, doc: &mut Document) -> Result<(), Diagnostic> {
        doc.apply_ops(&self.redo)?;
        install(doc, &self.geom_after);
        doc.touch();
        Ok(())
    }

    /// Apply backward to a bare document.
    pub fn revert_doc(&self, doc: &mut Document) -> Result<(), Diagnostic> {
        doc.apply_ops(&self.undo)?;
        install(doc, &self.geom_before);
        doc.touch();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples;

    #[test]
    fn commit_undo_redo_through_edit_history() {
        let mut doc = Document::new(examples::counter());
        doc.layout_missing();
        let orig = doc.clone();
        let sel: Vec<String> = doc
            .def("sync_counter_4")
            .unwrap()
            .instances
            .iter()
            .filter(|i| i.id != "reg")
            .map(|i| i.id.clone())
            .collect();
        let plan =
            super::super::package::plan_package(&doc, "sync_counter_4", &sel, "incr").unwrap();
        let c = DocChange::from(&plan);
        assert_eq!(
            c.touched(),
            vec!["sync_counter_4".to_string(), "incr".to_string()]
        );
        let mut h = History::default();
        c.commit(&mut h, &mut doc).unwrap();
        assert_eq!(h.undo_label(), Some("Create package incr"));
        assert!(doc.def("incr").is_some() && doc.editor("incr").is_some());
        assert_eq!(
            h.undo(&mut doc).unwrap().as_deref(),
            Some("Create package incr")
        );
        assert_eq!(doc.design, orig.design);
        assert_eq!(doc.editors, orig.editors);
        h.redo(&mut doc).unwrap();
        assert_eq!(doc.def("incr"), Some(&plan.package));
        assert_eq!(doc.editor("incr"), Some(&plan.package_editor));
    }
}
