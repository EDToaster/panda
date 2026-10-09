//! Design-level transforms.

use std::collections::{HashMap, HashSet};

use panda_netlist::{Atom, ComponentDef, Design, Endpoint, FORMAT, InstKind, Net, Termination};

/// Separator between hierarchy levels in the ids produced by [`inline_all`]. (`/` is reserved by
/// the netlist for real hierarchy paths and is not a legal id character.)
pub const PATH_SEP: char = '.';

/// Fully inline a design: the result has a single def — the top, with the same id, name and pins —
/// whose body contains only atoms. Behavior is identical to the original (flattening either design
/// yields the same gates, sources, resolvers and connectivity), but there are no hierarchy hints
/// left for the kernel to exploit (SPEC §12 variant b).
///
/// Naming: an atom or net with id `x` inside instance path `u0/ff3` gets the id `u0.ff3.x`
/// (see [`PATH_SEP`]), so flat net names stay recognizable. A net connected to a child pin is
/// merged with the parent's net and keeps the shallowest id (the parent's). Colliding ids get a
/// `_1`, `_2`, … suffix. Editor data is dropped.
///
/// Panics if the design is malformed (unknown top/def/instance/pin, recursive defs, width mismatch
/// at a pin, conflicting terminations on nets joined through a pin); use [`try_inline_all`] to get an error instead.
pub fn inline_all(design: &Design) -> Design {
    try_inline_all(design).unwrap_or_else(|e| panic!("inline_all: {e}"))
}

/// Fallible form of [`inline_all`].
pub fn try_inline_all(design: &Design) -> Result<Design, String> {
    let defs: HashMap<&str, &ComponentDef> = design.defs.iter().map(|d| (d.id.as_str(), d)).collect();
    let top = *defs.get(design.top.as_str()).ok_or_else(|| format!("unknown top def '{}'", design.top))?;
    let mut ctx = Ctx { defs, ..Ctx::default() };
    let mut stack = Vec::new();
    ctx.expand(top, "", None, &mut stack)?;
    let def = ctx.finish(top);
    Ok(Design { format: FORMAT.to_string(), defs: vec![def], top: design.top.clone() })
}

/// Where an inlined endpoint attaches.
enum End {
    Pin(String),
    /// Global atom index, port name.
    Port(usize, String),
}

#[derive(Default)]
struct Ctx<'a> {
    defs: HashMap<&'a str, &'a ComponentDef>,
    net_ids: Vec<String>,
    net_widths: Vec<u32>,
    net_terms: Vec<Termination>,
    parent: Vec<usize>,
    atoms: Vec<(String, Atom)>,
    ends: Vec<(usize, End)>,
}

impl<'a> Ctx<'a> {
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    /// Union keeping the smaller index (the shallower net) as the root. Fails if the two nets
    /// carry different non-`none` terminations (flattening rejects such designs too).
    fn union(&mut self, a: usize, b: usize) -> Result<(), String> {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return Ok(());
        }
        let (lo, hi) = (ra.min(rb), ra.max(rb));
        let (tl, th) = (self.net_terms[lo], self.net_terms[hi]);
        if tl == Termination::None {
            self.net_terms[lo] = th;
        } else if th != Termination::None && th != tl {
            return Err(format!(
                "conflicting terminations {tl:?} and {th:?} on nets '{}' and '{}'",
                self.net_ids[lo], self.net_ids[hi]
            ));
        }
        self.parent[hi] = lo;
        Ok(())
    }

    /// Inline `def` with id prefix `prefix`. `bindings` maps the def's pin names to the parent's
    /// global nets (`None` for the top def, whose pins stay pins).
    fn expand(
        &mut self,
        def: &'a ComponentDef,
        prefix: &str,
        bindings: Option<&HashMap<&str, usize>>,
        stack: &mut Vec<&'a str>,
    ) -> Result<(), String> {
        if stack.contains(&def.id.as_str()) {
            return Err(format!("recursive def '{}'", def.id));
        }
        stack.push(&def.id);
        // Local instance id → atom index or child def.
        enum Local<'a> {
            Atom(usize),
            Def(&'a ComponentDef),
        }
        let mut locals: HashMap<&str, Local<'a>> = HashMap::with_capacity(def.instances.len());
        for inst in &def.instances {
            let l = match &inst.kind {
                InstKind::Atom(a) => {
                    self.atoms.push((format!("{prefix}{}", inst.id), a.clone()));
                    Local::Atom(self.atoms.len() - 1)
                }
                InstKind::Def(child) => Local::Def(
                    self.defs
                        .get(child.as_str())
                        .copied()
                        .ok_or_else(|| format!("unknown def '{child}' (instance '{}' in '{}')", inst.id, def.id))?,
                ),
            };
            locals.insert(inst.id.as_str(), l);
        }
        let mut child_bind: HashMap<&str, HashMap<&str, usize>> = HashMap::new();
        for net in &def.nets {
            let g = self.parent.len();
            self.parent.push(g);
            self.net_ids.push(format!("{prefix}{}", net.id));
            self.net_widths.push(net.width);
            self.net_terms.push(net.termination);
            for e in &net.ends {
                match e {
                    Endpoint::Pin(p) => match bindings {
                        None => self.ends.push((g, End::Pin(p.clone()))),
                        Some(b) => {
                            if let Some(&pg) = b.get(p.as_str()) {
                                if self.net_widths[pg] != net.width {
                                    return Err(format!(
                                        "width mismatch at pin '{p}' of '{}' ({} vs {})",
                                        def.id, net.width, self.net_widths[pg]
                                    ));
                                }
                                self.union(g, pg)?;
                            }
                        }
                    },
                    Endpoint::Port { inst, port } => match locals.get(inst.as_str()) {
                        Some(Local::Atom(ai)) => self.ends.push((g, End::Port(*ai, port.clone()))),
                        Some(Local::Def(child)) => {
                            let pin = child.pins.iter().find(|p| p.name == *port).ok_or_else(|| {
                                format!("def '{}' has no pin '{port}' (instance '{inst}' in '{}')", child.id, def.id)
                            })?;
                            if pin.width != net.width {
                                return Err(format!(
                                    "width mismatch at {inst}.{port} in '{}' ({} vs {})",
                                    def.id, net.width, pin.width
                                ));
                            }
                            child_bind.entry(inst.as_str()).or_default().insert(port.as_str(), g);
                        }
                        None => return Err(format!("unknown instance '{inst}' in '{}'", def.id)),
                    },
                }
            }
        }
        for inst in &def.instances {
            if let Some(Local::Def(child)) = locals.get(inst.id.as_str()) {
                let empty = HashMap::new();
                let b = child_bind.get(inst.id.as_str()).unwrap_or(&empty);
                let p = format!("{prefix}{}{PATH_SEP}", inst.id);
                self.expand(child, &p, Some(b), stack)?;
            }
        }
        stack.pop();
        Ok(())
    }

    fn finish(mut self, top: &ComponentDef) -> ComponentDef {
        // Atom ids (unique; collisions only possible with odd user ids containing PATH_SEP).
        let mut taken: HashSet<String> = HashSet::new();
        let atom_ids: Vec<String> = self.atoms.iter().map(|(id, _)| unique(id, &mut taken)).collect();
        // Group endpoints per class root, in order of first appearance of the root index.
        let n = self.parent.len();
        let roots: Vec<usize> = (0..n).map(|i| self.find(i)).collect();
        let mut class_ends: Vec<Vec<Endpoint>> = (0..n).map(|_| Vec::new()).collect();
        for (net, e) in std::mem::take(&mut self.ends) {
            let ep = match e {
                End::Pin(p) => Endpoint::Pin(p),
                End::Port(ai, port) => Endpoint::Port { inst: atom_ids[ai].clone(), port },
            };
            class_ends[roots[net]].push(ep);
        }
        let pin_names: HashSet<&str> = top.pins.iter().map(|p| p.name.as_str()).collect();
        let mut net_taken: HashSet<String> = HashSet::new();
        // Reserve ids of nets that carry their own pin name first, so they keep them.
        for (r, ends) in class_ends.iter().enumerate() {
            if !ends.is_empty() && ends.contains(&Endpoint::Pin(self.net_ids[r].clone())) {
                net_taken.insert(self.net_ids[r].clone());
            }
        }
        let mut nets = Vec::new();
        for (r, ends) in class_ends.into_iter().enumerate() {
            if ends.is_empty() {
                continue;
            }
            let own = &self.net_ids[r];
            let id = if ends.contains(&Endpoint::Pin(own.clone())) {
                own.clone()
            } else {
                // Avoid other nets' ids and pin names of other nets.
                let mut id = own.clone();
                let mut k = 1;
                while net_taken.contains(&id) || pin_names.contains(id.as_str()) {
                    id = format!("{own}_{k}");
                    k += 1;
                }
                net_taken.insert(id.clone());
                id
            };
            nets.push(Net { id, width: self.net_widths[r], termination: self.net_terms[r], ends });
        }
        let instances = self
            .atoms
            .into_iter()
            .zip(atom_ids)
            .map(|((_, atom), id)| panda_netlist::Instance { id, kind: InstKind::Atom(atom) })
            .collect();
        ComponentDef {
            id: top.id.clone(),
            name: top.name.clone(),
            version: top.version,
            pins: top.pins.clone(),
            instances,
            nets,
            editor: serde_json::Value::Null,
        }
    }
}

fn unique(id: &str, taken: &mut HashSet<String>) -> String {
    let mut out = id.to_string();
    let mut k = 1;
    while taken.contains(&out) {
        out = format!("{id}_{k}");
        k += 1;
    }
    taken.insert(out.clone());
    out
}
