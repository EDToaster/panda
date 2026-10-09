//! Hierarchy navigation (pure): breadcrumbs of an instance path, parent / child paths and the
//! package instances of a def.
//!
//! The view itself is [`crate::app::ViewState`]; [`crate::app::AppState::enter`] switches it,
//! after which the shell's watch set follows the viewed instance's flat nets (SPEC §11 names,
//! [`crate::doc::values::ViewMap`]), so live values inside a package are those of that very
//! instance.

use panda_netlist::{DefId, Design, InstKind};

/// One breadcrumb.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Crumb {
    /// Display text: the top def's name, then instance ids.
    pub label: String,
    /// Instance path it navigates to (`""` = top).
    pub path: String,
    /// Def viewed there.
    pub def: DefId,
}

/// Breadcrumbs from the top to `path` (empty if the path does not exist).
pub fn crumbs(design: &Design, path: &str) -> Vec<Crumb> {
    let Some(mut def) = design.top_def() else {
        return Vec::new();
    };
    let mut out = vec![Crumb {
        label: if def.name.is_empty() {
            def.id.clone()
        } else {
            def.name.clone()
        },
        path: String::new(),
        def: def.id.clone(),
    }];
    if path.is_empty() {
        return out;
    }
    let mut cur = String::new();
    for id in path.split('/') {
        let Some(InstKind::Def(d)) = def.instances.iter().find(|i| i.id == id).map(|i| &i.kind)
        else {
            return Vec::new();
        };
        let Some(next) = design.def(d) else {
            return Vec::new();
        };
        cur = child_path(&cur, id);
        out.push(Crumb {
            label: id.to_string(),
            path: cur.clone(),
            def: next.id.clone(),
        });
        def = next;
    }
    out
}

/// Path of the enclosing instance (`None` at the top).
pub fn parent_path(path: &str) -> Option<String> {
    if path.is_empty() {
        return None;
    }
    Some(path.rsplit_once('/').map_or(String::new(), |(p, _)| p.to_string()))
}

/// Path of instance `inst` inside the instance at `path`.
pub fn child_path(path: &str, inst: &str) -> String {
    if path.is_empty() {
        inst.to_string()
    } else {
        format!("{path}/{inst}")
    }
}

/// Def-use instances of `def`: `(instance id, def id)` in instance order.
pub fn package_instances(design: &Design, def: &str) -> Vec<(String, DefId)> {
    design
        .def(def)
        .map(|d| {
            d.instances
                .iter()
                .filter_map(|i| match &i.kind {
                    InstKind::Def(c) => Some((i.id.clone(), c.clone())),
                    InstKind::Atom(_) => None,
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples;

    #[test]
    fn crumbs_follow_paths() {
        let d = examples::panda8();
        let c = crumbs(&d, "cpu/alu");
        assert_eq!(c.len(), 3);
        assert_eq!(c[0].path, "");
        assert_eq!(c[1].label, "cpu");
        assert_eq!(c[2].path, "cpu/alu");
        assert_eq!(
            c[2].def,
            crate::app::def_at_path(&d, "cpu/alu").unwrap()
        );
        assert!(crumbs(&d, "cpu/nope").is_empty());
        assert_eq!(parent_path("cpu/alu").as_deref(), Some("cpu"));
        assert_eq!(parent_path("cpu").as_deref(), Some(""));
        assert_eq!(parent_path(""), None);
        assert_eq!(child_path("", "cpu"), "cpu");
        assert!(!package_instances(&d, &d.top).is_empty());
    }
}
