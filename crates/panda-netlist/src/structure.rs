//! Private helpers shared by validation and flattening: non-panicking atom port lookup and the
//! per-def bit-level "template" (local bit layout of nets, pins and instance ports).
//!
//! A template assigns every bit of every net of a def a *local bit offset*. Pins that are not on
//! any net and atom ports that are not on any net get private ("virtual") bit ranges so that every
//! pin and every atom port has bits. Splitter aliasing is recorded as local bit pairs.
//!
//! Template construction is robust against invalid designs (unknown endpoints, width mismatches,
//! bad splitter parts): offending items are simply skipped. Validation reports them separately.

use std::collections::HashMap;

use crate::model::{Atom, ComponentDef, Design, Dir, Endpoint, InstKind};

/// Number of ports of an atom (see [`Atom::ports`] for the order).
pub(crate) fn atom_port_count(atom: &Atom) -> usize {
    match atom {
        Atom::Gate { inputs, .. } => *inputs as usize + 1,
        Atom::Not { .. } | Atom::Buf { .. } => 2,
        Atom::Tri { .. } => 3,
        Atom::Splitter { parts, .. } => parts.len() + 1,
        Atom::Constant { .. } | Atom::Clock { .. } | Atom::PowerOnReset { .. } => 1,
    }
}

/// Direction and width of port `idx` (in [`Atom::ports`] order). Never panics for indices below
/// [`atom_port_count`]; invalid splitter parts report width 0.
pub(crate) fn atom_port(atom: &Atom, idx: usize) -> (Dir, u32) {
    match atom {
        Atom::Gate { inputs, width, reduce, .. } => {
            if idx < *inputs as usize {
                (Dir::In, *width)
            } else {
                (Dir::Out, if *reduce { 1 } else { *width })
            }
        }
        Atom::Not { width } | Atom::Buf { width } => {
            if idx == 0 {
                (Dir::In, *width)
            } else {
                (Dir::Out, *width)
            }
        }
        Atom::Tri { width } => match idx {
            0 => (Dir::In, *width),
            1 => (Dir::In, 1),
            _ => (Dir::Out, *width),
        },
        Atom::Splitter { width, parts } => {
            if idx == 0 {
                (Dir::InOut, *width)
            } else {
                let (lo, hi) = parts[idx - 1];
                (Dir::InOut, hi.checked_sub(lo).map_or(0, |d| d.saturating_add(1)))
            }
        }
        Atom::Constant { value } => (Dir::Out, value.width()),
        Atom::Clock { .. } | Atom::PowerOnReset { .. } => (Dir::Out, 1),
    }
}

/// Parse `<prefix><n>` where `n` is a canonical decimal (no sign, no leading zeros).
fn indexed(name: &str, prefix: &str) -> Option<usize> {
    let digits = name.strip_prefix(prefix)?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) || (digits.len() > 1 && digits.starts_with('0'))
    {
        return None;
    }
    digits.parse().ok()
}

/// Index of the port called `name`, if the atom has one.
pub(crate) fn atom_port_index(atom: &Atom, name: &str) -> Option<usize> {
    match atom {
        Atom::Gate { inputs, .. } => {
            if name == "out" {
                Some(*inputs as usize)
            } else {
                indexed(name, "in").filter(|&i| i < *inputs as usize)
            }
        }
        Atom::Not { .. } | Atom::Buf { .. } => match name {
            "in" => Some(0),
            "out" => Some(1),
            _ => None,
        },
        Atom::Tri { .. } => match name {
            "d" => Some(0),
            "en" => Some(1),
            "out" => Some(2),
            _ => None,
        },
        Atom::Splitter { parts, .. } => {
            if name == "whole" {
                Some(0)
            } else {
                indexed(name, "p").filter(|&i| i < parts.len()).map(|i| i + 1)
            }
        }
        Atom::Constant { .. } | Atom::Clock { .. } | Atom::PowerOnReset { .. } => (name == "out").then_some(0),
    }
}

/// True if the splitter's parameters are valid (so its port widths are meaningful).
pub(crate) fn splitter_ok(width: u32, parts: &[(u32, u32)]) -> bool {
    parts.iter().all(|&(lo, hi)| lo <= hi && hi < width)
}

/// Name lookups for one def.
pub(crate) struct DefIndex<'a> {
    pub def: &'a ComponentDef,
    pub pin_index: HashMap<&'a str, usize>,
    pub inst_index: HashMap<&'a str, usize>,
    /// For each instance: the index of the referenced def (if it is a known def instance).
    pub child_def: Vec<Option<usize>>,
}

/// Name lookups for a whole design.
pub(crate) struct DesignIndex<'a> {
    pub defs: Vec<DefIndex<'a>>,
    pub def_by_id: HashMap<&'a str, usize>,
}

impl<'a> DesignIndex<'a> {
    pub fn new(design: &'a Design) -> Self {
        let mut def_by_id = HashMap::with_capacity(design.defs.len());
        for (i, d) in design.defs.iter().enumerate() {
            def_by_id.entry(d.id.as_str()).or_insert(i);
        }
        let defs = design
            .defs
            .iter()
            .map(|def| {
                let mut pin_index = HashMap::with_capacity(def.pins.len());
                for (i, p) in def.pins.iter().enumerate() {
                    pin_index.entry(p.name.as_str()).or_insert(i);
                }
                let mut inst_index = HashMap::with_capacity(def.instances.len());
                for (i, inst) in def.instances.iter().enumerate() {
                    inst_index.entry(inst.id.as_str()).or_insert(i);
                }
                let child_def = def
                    .instances
                    .iter()
                    .map(|inst| match &inst.kind {
                        InstKind::Def(id) => def_by_id.get(id.as_str()).copied(),
                        InstKind::Atom(_) => None,
                    })
                    .collect();
                DefIndex { def, pin_index, inst_index, child_def }
            })
            .collect();
        DesignIndex { defs, def_by_id }
    }

    /// Port index and (dir, width) of `port` on instance `inst` of def `d`.
    pub fn port(&self, d: usize, inst: usize, port: &str) -> Option<(usize, Dir, u32)> {
        let di = &self.defs[d];
        match &di.def.instances[inst].kind {
            InstKind::Atom(a) => {
                let idx = atom_port_index(a, port)?;
                let (dir, w) = atom_port(a, idx);
                Some((idx, dir, w))
            }
            InstKind::Def(_) => {
                let c = di.child_def[inst]?;
                let pi = *self.defs[c].pin_index.get(port)?;
                let p = &self.defs[c].def.pins[pi];
                Some((pi, p.dir, p.width))
            }
        }
    }

    /// Number of ports of instance `inst` of def `d` (0 for unknown defs).
    pub fn port_count(&self, d: usize, inst: usize) -> usize {
        let di = &self.defs[d];
        match &di.def.instances[inst].kind {
            InstKind::Atom(a) => atom_port_count(a),
            InstKind::Def(_) => di.child_def[inst].map_or(0, |c| self.defs[c].def.pins.len()),
        }
    }
}

/// Per-def local bit layout.
#[derive(Debug, Default)]
pub(crate) struct Template {
    /// Local bit offset of each net (index into `def.nets`).
    pub net_off: Vec<u32>,
    /// Local bit offset of each pin.
    pub pin_off: Vec<u32>,
    /// True if the pin is on no net (its bits are private).
    pub pin_virtual: Vec<bool>,
    /// Per instance, per port: local bit offset. Atom ports are always `Some` (virtual bits are
    /// allocated for unconnected ones); def-instance ports are `None` when unconnected.
    pub inst_ports: Vec<Vec<Option<u32>>>,
    /// Per instance, per port: true if the port's bits are virtual (not on any net).
    pub port_virtual: Vec<Vec<bool>>,
    /// Splitter aliasing as local bit pairs.
    pub unions: Vec<(u32, u32)>,
    /// Total local bits.
    pub bits: u32,
}

/// Error raised when the local bit space overflows `u32`.
#[derive(Debug)]
pub(crate) struct TooLarge;

impl Template {
    /// Build the template for def `d`. Robust: invalid endpoints are skipped.
    pub fn build(ix: &DesignIndex<'_>, d: usize) -> Result<Template, TooLarge> {
        let di = &ix.defs[d];
        let def = di.def;
        let mut bits: u32 = 0;
        let mut alloc = |w: u32| -> Result<u32, TooLarge> {
            let off = bits;
            bits = bits.checked_add(w).ok_or(TooLarge)?;
            Ok(off)
        };
        let mut net_off = Vec::with_capacity(def.nets.len());
        for n in &def.nets {
            net_off.push(alloc(n.width)?);
        }
        let mut pin_off: Vec<Option<u32>> = vec![None; def.pins.len()];
        let mut inst_ports: Vec<Vec<Option<u32>>> =
            (0..def.instances.len()).map(|i| vec![None; ix.port_count(d, i)]).collect();
        for (ni, n) in def.nets.iter().enumerate() {
            for e in &n.ends {
                match e {
                    Endpoint::Pin(p) => {
                        if let Some(&pi) = di.pin_index.get(p.as_str())
                            && def.pins[pi].width == n.width
                            && pin_off[pi].is_none()
                        {
                            pin_off[pi] = Some(net_off[ni]);
                        }
                    }
                    Endpoint::Port { inst, port } => {
                        if let Some(&ii) = di.inst_index.get(inst.as_str())
                            && let Some((pi, _, w)) = ix.port(d, ii, port)
                            && w == n.width
                            && inst_ports[ii][pi].is_none()
                        {
                            inst_ports[ii][pi] = Some(net_off[ni]);
                        }
                    }
                }
            }
        }
        let mut pin_virtual = vec![false; def.pins.len()];
        let mut pin_out = Vec::with_capacity(def.pins.len());
        for (pi, p) in def.pins.iter().enumerate() {
            match pin_off[pi] {
                Some(o) => pin_out.push(o),
                None => {
                    pin_virtual[pi] = true;
                    pin_out.push(alloc(p.width)?);
                }
            }
        }
        let mut port_virtual: Vec<Vec<bool>> = inst_ports.iter().map(|v| vec![false; v.len()]).collect();
        let mut unions = Vec::new();
        for (ii, inst) in def.instances.iter().enumerate() {
            let InstKind::Atom(atom) = &inst.kind else { continue };
            for pi in 0..inst_ports[ii].len() {
                if inst_ports[ii][pi].is_none() {
                    let (_, w) = atom_port(atom, pi);
                    inst_ports[ii][pi] = Some(alloc(w)?);
                    port_virtual[ii][pi] = true;
                }
            }
            if let Atom::Splitter { width, parts } = atom
                && splitter_ok(*width, parts)
                && let Some(whole) = inst_ports[ii][0]
            {
                for (k, &(lo, hi)) in parts.iter().enumerate() {
                    let Some(part) = inst_ports[ii][k + 1] else { continue };
                    for j in 0..=(hi - lo) {
                        unions.push((whole + lo + j, part + j));
                    }
                }
            }
        }
        Ok(Template { net_off, pin_off: pin_out, pin_virtual, inst_ports, port_virtual, unions, bits })
    }
}

/// Minimal union-find over `u32` ids. The root of every class is its smallest member.
#[derive(Debug, Default)]
pub(crate) struct UnionFind {
    parent: Vec<u32>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        UnionFind { parent: (0..n as u32).collect() }
    }
    pub fn len(&self) -> usize {
        self.parent.len()
    }
    /// Append `n` singleton elements.
    pub fn grow(&mut self, n: u32) {
        let start = self.parent.len() as u32;
        self.parent.extend(start..start + n);
    }
    #[inline]
    pub fn find(&mut self, mut x: u32) -> u32 {
        // Path halving.
        while self.parent[x as usize] != x {
            let p = self.parent[x as usize];
            let gp = self.parent[p as usize];
            self.parent[x as usize] = gp;
            x = gp;
        }
        x
    }
    #[inline]
    pub fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            self.parent[hi as usize] = lo;
        }
    }
}
