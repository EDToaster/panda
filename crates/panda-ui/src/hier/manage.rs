//! Managing the packages (user defs) of a design from the library: delete an unused package,
//! rename a package everywhere it is used. Both are plain [`DocChange`]s, so they are one
//! undoable action each.

use panda_kernel::EditOp;
use panda_netlist::{DefId, Design, InstKind};

use super::change::DocChange;
use super::package::valid_ident;
use crate::doc::Document;

/// Every placed instance of def `id`: `(parent def, instance id)`, in design order.
pub fn users(design: &Design, id: &str) -> Vec<(DefId, String)> {
    design
        .defs
        .iter()
        .flat_map(|d| {
            d.instances
                .iter()
                .filter(|i| matches!(&i.kind, InstKind::Def(x) if x == id))
                .map(|i| (d.id.clone(), i.id.clone()))
        })
        .collect()
}

/// Short human description of [`users`] ("top (not1, not2) and alu (n)").
pub fn describe_users(users: &[(DefId, String)]) -> String {
    let mut parents: Vec<(&str, Vec<&str>)> = Vec::new();
    for (p, i) in users {
        match parents.iter_mut().find(|(q, _)| q == p) {
            Some((_, v)) => v.push(i),
            None => parents.push((p, vec![i])),
        }
    }
    parents
        .iter()
        .map(|(p, v)| format!("{p} ({})", v.join(", ")))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Delete package `id`. Refused for the top def and while any instance of it is placed (the
/// message names them), so nothing in the design changes behaviour.
pub fn plan_delete_def(doc: &Document, id: &str) -> Result<DocChange, String> {
    let design = &doc.design;
    let def = design
        .def(id)
        .ok_or_else(|| format!("unknown package '{id}'"))?;
    if id == design.top {
        return Err("the top design cannot be deleted".into());
    }
    let used = users(design, id);
    if !used.is_empty() {
        return Err(format!(
            "'{id}' is still used by {}; delete those chips first",
            describe_users(&used)
        ));
    }
    let geom = doc.editors.get(id).cloned();
    Ok(DocChange {
        label: format!("Delete package {id}"),
        redo: vec![EditOp::RemoveDef(id.to_string())],
        undo: vec![EditOp::AddDef(def.clone())],
        geom_before: vec![(id.to_string(), geom)],
        geom_after: vec![(id.to_string(), None)],
    })
}

/// Rename package `old` to `new`: the def gets the new id and name, and every instance of it
/// (in every parent def) refers to the new id. Instance ids, connections and geometry stay.
pub fn plan_rename_def(doc: &Document, old: &str, new: &str) -> Result<DocChange, String> {
    let design = &doc.design;
    let def = design
        .def(old)
        .ok_or_else(|| format!("unknown package '{old}'"))?;
    if old == design.top {
        return Err("the top design cannot be renamed".into());
    }
    if new.is_empty() {
        return Err("the name cannot be empty".into());
    }
    if !valid_ident(new) {
        return Err("the name cannot contain spaces, '/', '[' or ']'".into());
    }
    if new == old {
        return Err("that is already its name".into());
    }
    if design.def(new).is_some() {
        return Err(format!("a package named '{new}' already exists"));
    }
    let mut renamed = def.clone();
    renamed.id = new.to_string();
    renamed.name = new.to_string();
    let mut redo = vec![EditOp::AddDef(renamed)];
    let mut undo = vec![EditOp::AddDef(def.clone())];
    let mut geom_before = vec![(old.to_string(), doc.editors.get(old).cloned())];
    let mut geom_after = vec![(old.to_string(), None)];
    for parent in &design.defs {
        if parent.id == old
            || !parent
                .instances
                .iter()
                .any(|i| matches!(&i.kind, InstKind::Def(x) if x == old))
        {
            continue;
        }
        let mut p = parent.clone();
        for i in &mut p.instances {
            if matches!(&i.kind, InstKind::Def(x) if x == old) {
                i.kind = InstKind::Def(new.to_string());
            }
        }
        redo.push(EditOp::ReplaceDef(p));
        undo.push(EditOp::ReplaceDef(parent.clone()));
        let g = doc.editors.get(&parent.id).cloned();
        geom_before.push((parent.id.clone(), g.clone()));
        geom_after.push((parent.id.clone(), g));
    }
    redo.push(EditOp::RemoveDef(old.to_string()));
    undo.push(EditOp::RemoveDef(new.to_string()));
    geom_before.push((new.to_string(), None));
    geom_after.push((new.to_string(), doc.editors.get(old).cloned()));
    Ok(DocChange {
        label: format!("Rename package {old} to {new}"),
        redo,
        undo,
        geom_before,
        geom_after,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hier::package::plan_package;

    /// The SR latch with one of its NANDs packaged as `not` (any name will do).
    fn doc_with_package() -> Document {
        let mut doc = Document::new(crate::examples::sr_latch());
        doc.layout_missing();
        let top = doc.design.top.clone();
        let insts: Vec<String> = doc
            .def(&top)
            .unwrap()
            .instances
            .iter()
            .map(|i| i.id.clone())
            .collect();
        let plan = plan_package(&doc, &top, &insts[..1], "not").expect("package");
        DocChange::from(&plan).apply_doc(&mut doc).expect("apply");
        doc
    }

    #[test]
    fn delete_is_refused_while_used_and_undoable_when_not() {
        let mut doc = doc_with_package();
        let top = doc.design.top.clone();
        let err = plan_delete_def(&doc, "not").unwrap_err();
        assert!(err.contains(&format!("still used by {top}")), "{err}");
        assert!(plan_delete_def(&doc, &top).is_err());
        // Remove the instance, then the package can go.
        let inst = users(&doc.design, "not")[0].1.clone();
        let mut t = doc.def(&top).unwrap().clone();
        t.instances.retain(|i| i.id != inst);
        for n in &mut t.nets {
            n.ends.retain(
                |e| !matches!(e, panda_netlist::Endpoint::Port { inst: i, .. } if *i == inst),
            );
        }
        t.nets.retain(|n| !n.ends.is_empty());
        doc.apply_ops(&[EditOp::ReplaceDef(t)])
            .expect("remove instance");
        let c = plan_delete_def(&doc, "not").expect("unused");
        c.apply_doc(&mut doc).expect("delete");
        assert!(doc.def("not").is_none() && !doc.editors.contains_key("not"));
        c.revert_doc(&mut doc).expect("undo");
        assert!(doc.def("not").is_some() && doc.editors.contains_key("not"));
    }

    #[test]
    fn rename_updates_every_use_and_undoes() {
        let mut doc = doc_with_package();
        let before = doc.design.clone();
        let c = plan_rename_def(&doc, "not", "inverter").expect("plan");
        c.apply_doc(&mut doc).expect("apply");
        assert!(doc.def("not").is_none());
        assert_eq!(doc.def("inverter").unwrap().name, "inverter");
        assert!(users(&doc.design, "not").is_empty());
        assert_eq!(users(&doc.design, "inverter").len(), 1);
        assert!(doc.editors.contains_key("inverter") && !doc.editors.contains_key("not"));
        c.revert_doc(&mut doc).expect("undo");
        assert_eq!(users(&doc.design, "not").len(), 1);
        assert!(doc.def("inverter").is_none());
        let mut ids: Vec<_> = doc.design.defs.iter().map(|d| d.id.clone()).collect();
        let mut want: Vec<_> = before.defs.iter().map(|d| d.id.clone()).collect();
        ids.sort();
        want.sort();
        assert_eq!(ids, want);
    }

    #[test]
    fn rename_rejects_bad_names() {
        let doc = doc_with_package();
        let top = doc.design.top.clone();
        assert!(plan_rename_def(&doc, "not", "").is_err());
        assert!(plan_rename_def(&doc, "not", "a b").is_err());
        assert!(plan_rename_def(&doc, "not", &top).is_err());
        assert!(plan_rename_def(&doc, &top, "x").is_err());
    }

    #[test]
    fn renaming_a_package_pin_rewires_its_chips() {
        let mut doc = doc_with_package();
        let top = doc.design.top.clone();
        let old = doc.def("not").unwrap().pins[0].name.clone();
        let (inst_parent, inst) = users(&doc.design, "not")[0].clone();
        assert_eq!(inst_parent, top);
        let (ops, defs, geom) =
            crate::edit::inspector::plan_rename_pin(&doc, "not", &old, "renamed_pin")
                .expect("plan");
        assert_eq!(defs, vec!["not".to_string(), top.clone()]);
        doc.apply_ops(&ops).expect("valid");
        if let Some((id, g)) = geom {
            assert!(g.pins.contains_key("renamed_pin") && !g.pins.contains_key(&old));
            doc.editors.insert(id, g);
        }
        let ends: Vec<_> = doc
            .def(&top)
            .unwrap()
            .nets
            .iter()
            .flat_map(|n| n.ends.clone())
            .collect();
        let port = |p: &str| panda_netlist::Endpoint::Port {
            inst: inst.clone(),
            port: p.into(),
        };
        assert!(ends.contains(&port("renamed_pin")) && !ends.contains(&port(&old)));
        assert!(
            crate::edit::inspector::plan_rename_pin(&doc, "not", "renamed_pin", "renamed_pin")
                .unwrap()
                .0
                .is_empty()
        );
        let other = doc.def("not").unwrap().pins[1].name.clone();
        assert!(
            crate::edit::inspector::plan_rename_pin(&doc, "not", "renamed_pin", &other).is_err()
        );
    }
}
