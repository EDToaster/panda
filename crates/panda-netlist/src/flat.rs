//! Flattened, bit-level netlist — the input to every engine and to the compiler.
//!
//! Flattening (see [`flatten`]):
//! * expands the hierarchy, recording an instance tree so engines/compilers can still find
//!   package boundaries (ideal blocks, def-level caching) and map nets back to paths;
//! * splits every W-bit net into W 1-bit nets and every W-bit gate into W 1-bit gates
//!   (a reduce gate becomes a single N-input gate); the W gates share a [`GroupIdx`] so the
//!   compiler can re-pack them into word ops (slice packing, SPEC §8.7);
//! * resolves Splitters, cross-boundary pins and inout pins by aliasing bits (union-find);
//! * gives every flat net **at most one driver**: a net with ≥2 drivers gets a [`Resolver`]
//!   whose inputs are private per-driver nets;
//! * Constant / Clock / PowerOnReset / top-level input pins become [`Source`]s.
//!
//! After flattening, the only delay-carrying elements are [`FlatGate`]s (delay 1 tick each).
//! Sources and resolvers are delay 0.

use std::collections::HashMap;

use crate::model::{DefId, Dir, Termination};
use crate::value::Logic;

pub type NetIdx = u32;
pub type GateIdx = u32;
pub type InstIdx = u32;
pub type GroupIdx = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FlatOp {
    And,
    Or,
    Nand,
    Nor,
    Xor,
    Xnor,
    Not,
    Buf,
    /// inputs = `[d, en]`
    Tri,
}

#[derive(Clone, Debug)]
pub struct FlatGate {
    pub op: FlatOp,
    /// ≥1 input nets (Not/Buf: 1; Tri: 2 = [d, en]; n-input gates: n).
    pub inputs: Vec<NetIdx>,
    pub output: NetIdx,
    /// Innermost def instance containing this gate (0 = top).
    pub owner: InstIdx,
    /// Gates created from the same multi-bit atom share a group; `bit` is the position in it.
    pub group: GroupIdx,
    pub bit: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SourceKind {
    Constant(Logic),
    Clock { period: u32, high: u32, phase: u32 },
    PowerOnReset { ticks: u32 },
    /// Bit `bit` of top-level input pin index `pin` (index into `FlatNetlist::top_inputs`).
    TopInput { pin: u32, bit: u32 },
}

#[derive(Clone, Debug)]
pub struct Source {
    pub kind: SourceKind,
    pub output: NetIdx,
}

/// Resolution of a multi-driver net (SPEC §3.3). Delay 0.
/// Per bit: drop Z drivers; none left → termination; all agree & known → value; else X (+contention).
/// `Keeper` holds the last resolved non-Z value (initially X).
#[derive(Clone, Debug)]
pub struct Resolver {
    pub drivers: Vec<NetIdx>,
    pub termination: Termination,
    pub output: NetIdx,
}

#[derive(Clone, Debug)]
pub struct FlatPin {
    pub name: String,
    pub dir: Dir,
    /// One flat net per bit, LSB first.
    pub nets: Vec<NetIdx>,
}

/// A def instance in the expanded hierarchy. Index 0 is the top def.
#[derive(Clone, Debug)]
pub struct FlatInstance {
    /// `""` for top, otherwise instance ids joined by `/`, e.g. `"cpu/alu/fa3"`.
    pub path: String,
    pub def: DefId,
    pub parent: Option<InstIdx>,
    pub children: Vec<InstIdx>,
    pub pins: Vec<FlatPin>,
}

#[derive(Clone, Debug)]
pub struct GateGroup {
    pub owner: InstIdx,
    /// Path of the atom instance, e.g. `"cpu/alu/g12"`.
    pub path: String,
    pub gates: Vec<GateIdx>,
}

#[derive(Clone, Debug, Default)]
pub struct FlatNetlist {
    pub num_nets: u32,
    pub gates: Vec<FlatGate>,
    pub sources: Vec<Source>,
    pub resolvers: Vec<Resolver>,
    pub groups: Vec<GateGroup>,
    pub instances: Vec<FlatInstance>,
    /// Termination for undriven single-driver-less nets (pull-up/down on a net with no drivers).
    /// Nets absent from this map and without a driver read `Z`.
    pub undriven_termination: HashMap<NetIdx, Termination>,
    /// Canonical name per flat net: `"<inst path>/<net id>[<bit>]"` (top-level: `"<net id>[<bit>]"`;
    /// 1-bit nets omit `[0]`).
    pub net_names: Vec<String>,
    /// Every alias of every net (all hierarchical names that refer to it) → net. Includes pin names
    /// like `"cpu/alu/A[3]"` and multi-bit-less forms for 1-bit nets.
    pub name_index: HashMap<String, NetIdx>,
    /// Top-level In pins (clickable inputs). Order = order in the top def's pin list.
    pub top_inputs: Vec<FlatPin>,
    /// Top-level Out (and InOut) pins.
    pub top_outputs: Vec<FlatPin>,
}

/// What drives a flat net.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Driver {
    None,
    Gate(GateIdx),
    Source(u32),
    Resolver(u32),
}

impl FlatNetlist {
    /// Driver of every net (index = NetIdx).
    pub fn drivers(&self) -> Vec<Driver> {
        let mut d = vec![Driver::None; self.num_nets as usize];
        for (i, g) in self.gates.iter().enumerate() {
            d[g.output as usize] = Driver::Gate(i as GateIdx);
        }
        for (i, s) in self.sources.iter().enumerate() {
            d[s.output as usize] = Driver::Source(i as u32);
        }
        for (i, r) in self.resolvers.iter().enumerate() {
            d[r.output as usize] = Driver::Resolver(i as u32);
        }
        d
    }

    /// For every net, the gates and resolvers that read it.
    pub fn fanout(&self) -> Fanout {
        let n = self.num_nets as usize;
        let mut gates = vec![Vec::new(); n];
        let mut resolvers = vec![Vec::new(); n];
        for (i, g) in self.gates.iter().enumerate() {
            for &inp in &g.inputs {
                // Gates are visited in increasing order, so a duplicate (a gate reading the same
                // net twice) can only be the last entry: O(1) dedup instead of `contains`.
                if gates[inp as usize].last() != Some(&(i as GateIdx)) {
                    gates[inp as usize].push(i as GateIdx);
                }
            }
        }
        for (i, r) in self.resolvers.iter().enumerate() {
            for &inp in &r.drivers {
                resolvers[inp as usize].push(i as u32);
            }
        }
        Fanout { gates, resolvers }
    }

    pub fn net_by_name(&self, name: &str) -> Option<NetIdx> {
        self.name_index.get(name).copied()
    }

    /// Instance by path (`""` = top).
    pub fn instance_by_path(&self, path: &str) -> Option<InstIdx> {
        self.instances.iter().position(|i| i.path == path).map(|i| i as InstIdx)
    }

    /// True if instance `inst` is `ancestor` or nested inside it.
    pub fn is_within(&self, inst: InstIdx, ancestor: InstIdx) -> bool {
        let mut cur = Some(inst);
        while let Some(c) = cur {
            if c == ancestor {
                return true;
            }
            cur = self.instances[c as usize].parent;
        }
        false
    }
}

#[derive(Clone, Debug, Default)]
pub struct Fanout {
    pub gates: Vec<Vec<GateIdx>>,
    pub resolvers: Vec<Vec<u32>>,
}

/// Flatten a design. Validates first (see [`crate::validate`]); returns all errors found.
pub fn flatten(design: &crate::model::Design) -> Result<FlatNetlist, Vec<crate::Diagnostic>> {
    crate::flatten_impl::flatten(design)
}
