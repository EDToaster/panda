//! Mapping diagnostic / STA locations to a place on the canvas (pure, no egui).
//!
//! Locations come in several shapes:
//! * validation: `"<def>: net <id>"`, `"<def>: instance <id>"`, `"<def>: pin <name>"`;
//! * simulation / STA: hierarchical flat-net names, `"cpu/alu/c[3]"` (SPEC §11 names; any alias
//!   of the net);
//! * instance paths (`"cpu/alu/g12"`: the atom `g12` inside instance `cpu/alu`) or a bare def id.
//!
//! [`resolve_location`] turns them into a [`JumpTarget`]: the instance path to view and what to
//! mark there. [`JumpTarget::crumbs`] describes it with the hierarchy navigation's breadcrumbs
//! ([`crate::hier::nav`]) and [`mark_bounds`] finds the marked thing on a built canvas
//! [`Scene`] (for centring and the jump overlay).

use panda_netlist::flat::FlatNetlist;
use panda_netlist::{ComponentDef, DefId, Design, InstKind};

use crate::app::def_at_path;
use crate::canvas::Scene;
use crate::doc::ItemRef;
use crate::geom::GridRect;
use crate::hier::nav;

/// What to emphasize in the target view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mark {
    /// A net of the viewed def (by net id).
    Net(String),
    /// An instance or pin of the viewed def.
    Item(ItemRef),
    /// Nothing specific (the whole view).
    None,
}

/// Where a location points.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JumpTarget {
    /// Instance path to view (`""` = top).
    pub path: String,
    /// Def of that instance.
    pub def: DefId,
    /// What to mark.
    pub mark: Mark,
}

impl JumpTarget {
    /// Breadcrumbs from the top to the target's instance ([`nav::crumbs`]); empty if the path
    /// no longer exists in `design`.
    pub fn crumbs(&self, design: &Design) -> Vec<nav::Crumb> {
        nav::crumbs(design, &self.path)
    }

    /// One-line description, e.g. `"Panda-8 › cpu › alu: net c"`.
    pub fn describe(&self, design: &Design) -> String {
        let crumbs: Vec<String> = self.crumbs(design).into_iter().map(|c| c.label).collect();
        let at = if crumbs.is_empty() {
            format!("/{}", self.path)
        } else {
            crumbs.join(" › ")
        };
        match &self.mark {
            Mark::Net(n) => format!("{at}: net {n}"),
            Mark::Item(ItemRef::Instance(i)) => format!("{at}: instance {i}"),
            Mark::Item(ItemRef::Pin(p)) => format!("{at}: pin {p}"),
            Mark::None => at,
        }
    }
}

/// Grid bounds of `mark` on `scene` (`None` for [`Mark::None`] or when it is not drawn).
pub fn mark_bounds(scene: &Scene, mark: &Mark) -> Option<GridRect> {
    match mark {
        Mark::Net(id) => scene.net_index.get(id).map(|&i| scene.nets[i].bounds),
        Mark::Item(r) => scene.item(r).map(|p| p.rect()),
        Mark::None => None,
    }
}

/// The first instance path (depth-first, in instance order) whose def is `def`.
pub fn first_path_of_def(design: &Design, def: &str) -> Option<String> {
    fn walk(
        design: &Design,
        d: &ComponentDef,
        path: &str,
        want: &str,
        depth: u32,
    ) -> Option<String> {
        if d.id == want {
            return Some(path.to_string());
        }
        if depth > 64 {
            return None;
        }
        for i in &d.instances {
            if let InstKind::Def(sub) = &i.kind
                && let Some(sd) = design.def(sub)
            {
                let p = if path.is_empty() {
                    i.id.clone()
                } else {
                    format!("{path}/{}", i.id)
                };
                if let Some(r) = walk(design, sd, &p, want, depth + 1) {
                    return Some(r);
                }
            }
        }
        None
    }
    walk(design, design.top_def()?, "", def, 0)
}

/// The net named by a critical-path entry: STA writes a sequential kernel (latch / flip-flop
/// cluster) on the path as `"[kernel <net>]"` (its first output net); plain entries are net
/// names. Returns the net name and whether the entry was a kernel.
pub fn path_entry_net(entry: &str) -> (&str, bool) {
    match entry
        .strip_prefix("[kernel ")
        .and_then(|r| r.strip_suffix(']'))
    {
        Some(n) => (n, true),
        None => (entry, false),
    }
}

/// Strip a trailing `[<bit>]`.
fn strip_bit(s: &str) -> &str {
    if s.ends_with(']')
        && let Some(i) = s.rfind('[')
        && s[i + 1..s.len() - 1].bytes().all(|b| b.is_ascii_digit())
    {
        return &s[..i];
    }
    s
}

fn split_last(s: &str) -> (&str, &str) {
    match s.rfind('/') {
        Some(i) => (&s[..i], &s[i + 1..]),
        None => ("", s),
    }
}

/// Resolve a hierarchical name (`"path/id"`, optionally `[bit]`) against the design.
fn resolve_hier(design: &Design, name: &str) -> Option<JumpTarget> {
    let base = strip_bit(name);
    let (parent, id) = split_last(base);
    if let Some(def_id) = def_at_path(design, parent)
        && let Some(def) = design.def(&def_id)
    {
        let mark = if def.nets.iter().any(|n| n.id == id) {
            Some(Mark::Net(id.to_string()))
        } else if def.pins.iter().any(|p| p.name == id) {
            Some(Mark::Item(ItemRef::Pin(id.to_string())))
        } else if def.instances.iter().any(|i| i.id == id) {
            Some(Mark::Item(ItemRef::Instance(id.to_string())))
        } else {
            None
        };
        if let Some(mark) = mark {
            return Some(JumpTarget {
                path: parent.to_string(),
                def: def_id,
                mark,
            });
        }
    }
    None
}

/// Resolve `loc` (see the module docs). `flat` lets net aliases resolve when the given name is
/// not itself a `path/net` of the design.
pub fn resolve_location(
    design: &Design,
    flat: Option<&FlatNetlist>,
    loc: &str,
) -> Option<JumpTarget> {
    let loc = path_entry_net(loc.trim()).0;
    if loc.is_empty() {
        return None;
    }
    // Validation form: "<def>: <what> <id>".
    if let Some((def, rest)) = loc.split_once(": ") {
        let path = first_path_of_def(design, def)?;
        let mark = match rest.split_once(' ') {
            Some(("net", id)) => Mark::Net(id.to_string()),
            Some(("instance", id)) => Mark::Item(ItemRef::Instance(id.to_string())),
            Some(("pin", id)) => Mark::Item(ItemRef::Pin(id.to_string())),
            _ => Mark::None,
        };
        return Some(JumpTarget {
            path,
            def: def.to_string(),
            mark,
        });
    }
    if let Some(t) = resolve_hier(design, loc) {
        return Some(t);
    }
    // An instance path or a def id.
    if let Some(def) = def_at_path(design, loc) {
        return Some(JumpTarget {
            path: loc.to_string(),
            def,
            mark: Mark::None,
        });
    }
    if design.def(loc).is_some()
        && let Some(path) = first_path_of_def(design, loc)
    {
        return Some(JumpTarget {
            path,
            def: loc.to_string(),
            mark: Mark::None,
        });
    }
    // Another alias of the same flat net (shallowest first).
    let flat = flat?;
    let net = flat.net_by_name(loc)?;
    let mut aliases: Vec<&String> = flat
        .name_index
        .iter()
        .filter(|&(n, &i)| i == net && n.as_str() != loc)
        .map(|(n, _)| n)
        .collect();
    aliases.sort_by_key(|a| (a.matches('/').count(), a.len(), (*a).clone()));
    aliases.into_iter().find_map(|a| resolve_hier(design, a))
}

/// Local net ids of the view mapped by `map` that carry any of the flat nets named in `names`
/// (any alias, e.g. a critical path given as canonical flat-net names).
pub fn nets_in_view(
    flat: &FlatNetlist,
    map: &crate::doc::values::ViewMap,
    names: &[String],
) -> Vec<String> {
    use std::collections::HashSet;
    let wanted: HashSet<u32> = names
        .iter()
        .filter_map(|n| flat.net_by_name(path_entry_net(n).0))
        .collect();
    if wanted.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<String> = map
        .nets
        .iter()
        .filter(|(_, bits)| {
            bits.iter().flatten().any(|&w| {
                map.watch
                    .get(w as usize)
                    .is_some_and(|n| wanted.contains(n))
            })
        })
        .map(|(id, _)| id.clone())
        .collect();
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use panda_netlist::{DefBuilder, DesignBuilder};

    /// top { u0: inner { g: not } } with nets named.
    fn design() -> Design {
        let mut db = DesignBuilder::new();
        let mut b = DefBuilder::new("inner");
        let a = b.input("A", 1);
        let y = b.not(a);
        let y = b.name(y, "ny");
        b.output("Y", y);
        let inner = db.add(b.finish());
        let mut b = DefBuilder::new("top");
        let x = b.input("X", 1);
        let o = b.inst_named("u0", &inner, &[x]);
        let o = b.name(o[0], "res");
        b.output("O", o);
        let top = db.add(b.finish());
        db.build(&top)
    }

    fn not_inst(d: &Design) -> String {
        d.def("inner").unwrap().instances[0].id.clone()
    }

    #[test]
    fn validation_locations() {
        let d = design();
        assert_eq!(first_path_of_def(&d, "inner").as_deref(), Some("u0"));
        assert_eq!(first_path_of_def(&d, "top").as_deref(), Some(""));
        let t = resolve_location(&d, None, "inner: net ny").unwrap();
        assert_eq!((t.path.as_str(), t.def.as_str()), ("u0", "inner"));
        assert_eq!(t.mark, Mark::Net("ny".into()));
        let t = resolve_location(&d, None, "top: instance u0").unwrap();
        assert_eq!(t.path, "");
        assert_eq!(t.mark, Mark::Item(ItemRef::Instance("u0".into())));
        let t = resolve_location(&d, None, "inner: pin A").unwrap();
        assert_eq!(t.mark, Mark::Item(ItemRef::Pin("A".into())));
        assert!(resolve_location(&d, None, "edit op #3").is_none());
        assert!(resolve_location(&d, None, "nowhere: net q").is_none());
    }

    #[test]
    fn hierarchical_names_and_instances() {
        let d = design();
        let t = resolve_location(&d, None, "u0/ny").unwrap();
        assert_eq!((t.path.as_str(), t.mark), ("u0", Mark::Net("ny".into())));
        let t = resolve_location(&d, None, "res").unwrap();
        assert_eq!((t.path.as_str(), t.mark), ("", Mark::Net("res".into())));
        let g = not_inst(&d);
        let t = resolve_location(&d, None, &format!("u0/{g}")).unwrap();
        assert_eq!(t.mark, Mark::Item(ItemRef::Instance(g)));
        // Instance path / def id: the whole view.
        let t = resolve_location(&d, None, "inner").unwrap();
        assert_eq!((t.path.as_str(), t.mark), ("u0", Mark::None));
        // Bit suffix (the builder names an input's net after the pin; nets win over pins).
        let t = resolve_location(&d, None, "u0/A[0]").unwrap();
        assert_eq!((t.path.as_str(), t.mark), ("u0", Mark::Net("A".into())));
    }

    #[test]
    fn aliases_resolve_through_the_flat_netlist() {
        let d = design();
        let flat = panda_netlist::flat::flatten(&d).expect("flattens");
        // Every alias of the inner output net resolves to some marked place.
        let n = flat.net_by_name("u0/ny").unwrap();
        for (name, &i) in &flat.name_index {
            if i == n {
                let t = resolve_location(&d, Some(&flat), name);
                assert!(t.is_some_and(|t| t.mark != Mark::None), "{name}");
            }
        }
        // nets_in_view maps flat names onto the local nets of a view.
        let top = d.def("top").unwrap();
        let map = crate::doc::values::ViewMap::build(&flat, "", top);
        assert_eq!(
            nets_in_view(&flat, &map, &["u0/ny".to_string()]),
            vec!["res"]
        );
        assert_eq!(
            nets_in_view(&flat, &map, &["[kernel u0/ny]".to_string()]),
            vec!["res"]
        );
        assert_eq!(path_entry_net("[kernel a/b]"), ("a/b", true));
        assert_eq!(path_entry_net("a/b[3]"), ("a/b[3]", false));
        let t = resolve_location(&d, Some(&flat), "[kernel u0/ny]").unwrap();
        assert_eq!((t.path.as_str(), t.mark), ("u0", Mark::Net("ny".into())));
        assert!(nets_in_view(&flat, &map, &["nope".to_string()]).is_empty());
    }
}
