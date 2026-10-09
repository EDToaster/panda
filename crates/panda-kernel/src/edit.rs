//! Edit operations on the hierarchical design ([`EditOp`]).

use panda_netlist::{
    Atom, ComponentDef, DefId, DiagKind, Diagnostic, Endpoint, InstKind, Instance, Net, PinDecl,
};
use panda_netlist::{Design, Termination};

/// One incremental edit of the loaded [`Design`]. A batch of operations passed to
/// [`Kernel::edit`](crate::Kernel::edit) is applied in order and then validated as a whole; any
/// error rolls the whole batch back.
#[derive(Clone, Debug, PartialEq)]
pub enum EditOp {
    /// Add a new def (its id must not exist yet).
    AddDef(ComponentDef),
    /// Replace the def with the same id (live link: every instance sees the new content).
    ReplaceDef(ComponentDef),
    /// Remove a def (validation fails if it is still instantiated or is the top).
    RemoveDef(DefId),
    /// Add a pin to a def's interface.
    AddPin {
        /// Def to edit.
        def: DefId,
        /// The new pin.
        pin: PinDecl,
    },
    /// Remove a pin from a def's interface, together with the endpoints naming it in that def.
    RemovePin {
        /// Def to edit.
        def: DefId,
        /// Pin name.
        name: String,
    },
    /// Add an instance (atom or def use) to a def.
    AddInstance {
        /// Def to edit.
        def: DefId,
        /// The new instance.
        instance: Instance,
    },
    /// Remove an instance and every net endpoint referring to it.
    RemoveInstance {
        /// Def to edit.
        def: DefId,
        /// Instance id.
        id: String,
    },
    /// Add a net to a def.
    AddNet {
        /// Def to edit.
        def: DefId,
        /// The new net.
        net: Net,
    },
    /// Remove a net (its endpoints become unconnected).
    RemoveNet {
        /// Def to edit.
        def: DefId,
        /// Net id.
        id: String,
    },
    /// Connect an endpoint to a net.
    Connect {
        /// Def to edit.
        def: DefId,
        /// Net id.
        net: String,
        /// Endpoint to add to the net.
        endpoint: Endpoint,
    },
    /// Disconnect an endpoint from a net.
    Disconnect {
        /// Def to edit.
        def: DefId,
        /// Net id.
        net: String,
        /// Endpoint to remove from the net.
        endpoint: Endpoint,
    },
    /// Change the parameters of an atom instance (e.g. a gate's op or input count, a constant's
    /// value, a clock's period).
    SetAtom {
        /// Def to edit.
        def: DefId,
        /// Instance id (must be an atom instance).
        instance: String,
        /// New atom.
        atom: Atom,
    },
    /// Point a def-use instance at another def.
    SetInstanceDef {
        /// Def to edit.
        def: DefId,
        /// Instance id (must be a def use).
        instance: String,
        /// The def it instantiates from now on.
        target: DefId,
    },
    /// Make another def the top.
    SetTop(DefId),
    /// Change a net's termination.
    SetTermination {
        /// Def to edit.
        def: DefId,
        /// Net id.
        net: String,
        /// New termination.
        termination: Termination,
    },
}

fn err(kind: DiagKind, i: usize, msg: String) -> Diagnostic {
    Diagnostic::error(kind, msg).at(format!("edit op #{i}"))
}

fn def_mut<'a>(d: &'a mut Design, id: &str, i: usize) -> Result<&'a mut ComponentDef, Diagnostic> {
    d.defs
        .iter_mut()
        .find(|x| x.id == id)
        .ok_or_else(|| err(DiagKind::UnknownDef, i, format!("unknown def '{id}'")))
}

fn net_mut<'a>(def: &'a mut ComponentDef, id: &str, i: usize) -> Result<&'a mut Net, Diagnostic> {
    let did = def.id.clone();
    def.nets
        .iter_mut()
        .find(|n| n.id == id)
        .ok_or_else(|| err(DiagKind::Other, i, format!("def '{did}' has no net '{id}'")))
}

fn inst_mut<'a>(
    def: &'a mut ComponentDef,
    id: &str,
    i: usize,
) -> Result<&'a mut Instance, Diagnostic> {
    let did = def.id.clone();
    def.instances
        .iter_mut()
        .find(|n| n.id == id)
        .ok_or_else(|| {
            err(
                DiagKind::UnknownInstance,
                i,
                format!("def '{did}' has no instance '{id}'"),
            )
        })
}

/// Apply `ops` to `design` in order. Stops at the first operation that cannot be applied
/// (unknown def / instance / net, duplicate id); the design is then partially edited, so callers
/// apply edits to a copy. Structural validity is checked separately (validation).
pub(crate) fn apply(design: &mut Design, ops: &[EditOp]) -> Result<(), Diagnostic> {
    for (i, op) in ops.iter().enumerate() {
        apply_one(design, i, op)?;
    }
    Ok(())
}

fn apply_one(design: &mut Design, i: usize, op: &EditOp) -> Result<(), Diagnostic> {
    match op {
        EditOp::AddDef(def) => {
            if design.def(&def.id).is_some() {
                return Err(err(
                    DiagKind::DuplicateId,
                    i,
                    format!("def '{}' already exists", def.id),
                ));
            }
            design.defs.push(def.clone());
        }
        EditOp::ReplaceDef(def) => {
            *def_mut(design, &def.id, i)? = def.clone();
        }
        EditOp::RemoveDef(id) => {
            let pos = design
                .defs
                .iter()
                .position(|d| d.id == *id)
                .ok_or_else(|| err(DiagKind::UnknownDef, i, format!("unknown def '{id}'")))?;
            design.defs.remove(pos);
        }
        EditOp::AddPin { def, pin } => {
            let d = def_mut(design, def, i)?;
            if d.pins.iter().any(|p| p.name == pin.name) {
                return Err(err(
                    DiagKind::DuplicateId,
                    i,
                    format!("def '{def}' already has pin '{}'", pin.name),
                ));
            }
            d.pins.push(pin.clone());
        }
        EditOp::RemovePin { def, name } => {
            let d = def_mut(design, def, i)?;
            let pos = d.pins.iter().position(|p| p.name == *name).ok_or_else(|| {
                err(
                    DiagKind::UnknownPort,
                    i,
                    format!("def '{def}' has no pin '{name}'"),
                )
            })?;
            d.pins.remove(pos);
            for n in &mut d.nets {
                n.ends
                    .retain(|e| !matches!(e, Endpoint::Pin(p) if p == name));
            }
        }
        EditOp::AddInstance { def, instance } => {
            let d = def_mut(design, def, i)?;
            if d.instances.iter().any(|x| x.id == instance.id) {
                return Err(err(
                    DiagKind::DuplicateId,
                    i,
                    format!("def '{def}' already has instance '{}'", instance.id),
                ));
            }
            d.instances.push(instance.clone());
        }
        EditOp::RemoveInstance { def, id } => {
            let d = def_mut(design, def, i)?;
            let pos = d
                .instances
                .iter()
                .position(|x| x.id == *id)
                .ok_or_else(|| {
                    err(
                        DiagKind::UnknownInstance,
                        i,
                        format!("def '{def}' has no instance '{id}'"),
                    )
                })?;
            d.instances.remove(pos);
            for n in &mut d.nets {
                n.ends
                    .retain(|e| !matches!(e, Endpoint::Port { inst, .. } if inst == id));
            }
        }
        EditOp::AddNet { def, net } => {
            let d = def_mut(design, def, i)?;
            if d.nets.iter().any(|n| n.id == net.id) {
                return Err(err(
                    DiagKind::DuplicateId,
                    i,
                    format!("def '{def}' already has net '{}'", net.id),
                ));
            }
            d.nets.push(net.clone());
        }
        EditOp::RemoveNet { def, id } => {
            let d = def_mut(design, def, i)?;
            let pos =
                d.nets.iter().position(|n| n.id == *id).ok_or_else(|| {
                    err(DiagKind::Other, i, format!("def '{def}' has no net '{id}'"))
                })?;
            d.nets.remove(pos);
        }
        EditOp::Connect { def, net, endpoint } => {
            let n = net_mut(def_mut(design, def, i)?, net, i)?;
            if !n.ends.contains(endpoint) {
                n.ends.push(endpoint.clone());
            }
        }
        EditOp::Disconnect { def, net, endpoint } => {
            let n = net_mut(def_mut(design, def, i)?, net, i)?;
            let before = n.ends.len();
            n.ends.retain(|e| e != endpoint);
            if n.ends.len() == before {
                return Err(err(
                    DiagKind::UnknownPort,
                    i,
                    format!("net '{net}' of def '{def}' has no endpoint {endpoint:?}"),
                ));
            }
        }
        EditOp::SetAtom {
            def,
            instance,
            atom,
        } => {
            let x = inst_mut(def_mut(design, def, i)?, instance, i)?;
            match &mut x.kind {
                InstKind::Atom(a) => *a = atom.clone(),
                InstKind::Def(_) => {
                    return Err(err(
                        DiagKind::BadAtom,
                        i,
                        format!("instance '{instance}' of def '{def}' is not an atom"),
                    ));
                }
            }
        }
        EditOp::SetInstanceDef {
            def,
            instance,
            target,
        } => {
            let x = inst_mut(def_mut(design, def, i)?, instance, i)?;
            match &mut x.kind {
                InstKind::Def(d) => *d = target.clone(),
                InstKind::Atom(_) => {
                    return Err(err(
                        DiagKind::BadAtom,
                        i,
                        format!("instance '{instance}' of def '{def}' is an atom"),
                    ));
                }
            }
        }
        EditOp::SetTop(id) => {
            if design.def(id).is_none() {
                return Err(err(DiagKind::UnknownDef, i, format!("unknown def '{id}'")));
            }
            design.top = id.clone();
        }
        EditOp::SetTermination {
            def,
            net,
            termination,
        } => {
            net_mut(def_mut(design, def, i)?, net, i)?.termination = *termination;
        }
    }
    Ok(())
}
