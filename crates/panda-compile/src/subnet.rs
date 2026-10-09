//! Standalone sub-netlist of one kernel, simulated by [`panda_timing::TimingSim`] to compute the
//! kernel's transitions with exact Timing-mode semantics (docs/M3-CONTRACT.md).
//!
//! Layout of the sub-netlist (all indices local):
//! * nets `0..k` are the kernel inputs ([`Kernel::inputs`] order), each driven by a
//!   [`SourceKind::TopInput`] source, bit `i` of the single top-level input pin [`INPUT_PIN`]
//!   (absent when `k == 0`);
//! * nets `k..k+m` are the kernel nets ([`Kernel::nets`] order) — the state vector;
//! * the kernel's gates and resolvers, in [`Kernel::gates`] / [`Kernel::resolvers`] order;
//! * nothing else: no outputs, no other sources.

use std::collections::HashMap;

use panda_netlist::flat::{
    FlatGate, FlatInstance, FlatNetlist, FlatOp, FlatPin, GateGroup, NetIdx, Resolver, Source,
    SourceKind,
};
use panda_netlist::{Dir, Termination};

use crate::kernels::Kernel;

/// Name of the sub-netlist's single top-level input pin (bit `i` = kernel input `i`).
pub const INPUT_PIN: &str = "in";

/// Correspondence between a kernel sub-netlist and the original flat netlist.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubnetMap {
    /// Flat net of sub-net `i` for `i < k` (= [`Kernel::inputs`]).
    pub inputs: Vec<NetIdx>,
    /// Flat net of sub-net `k + j` (= [`Kernel::nets`]).
    pub nets: Vec<NetIdx>,
}

impl SubnetMap {
    /// Number of kernel inputs `k`.
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Number of state nets `m`.
    pub fn num_state(&self) -> usize {
        self.nets.len()
    }

    /// Sub-net index of state net `j`.
    pub fn state_net(&self, j: usize) -> NetIdx {
        (self.inputs.len() + j) as NetIdx
    }

    /// Flat net of a sub-net.
    pub fn to_flat(&self, sub: NetIdx) -> NetIdx {
        let s = sub as usize;
        if s < self.inputs.len() {
            self.inputs[s]
        } else {
            self.nets[s - self.inputs.len()]
        }
    }
}

/// Build the standalone sub-netlist of `kernel` (see the module docs for its layout).
pub fn kernel_subnet(flat: &FlatNetlist, kernel: &Kernel) -> (FlatNetlist, SubnetMap) {
    let k = kernel.inputs.len();
    let mut local: HashMap<NetIdx, NetIdx> = HashMap::with_capacity(k + kernel.nets.len());
    for (i, &net) in kernel.inputs.iter().chain(&kernel.nets).enumerate() {
        local.insert(net, i as NetIdx);
    }
    let map_net = |net: NetIdx| -> NetIdx {
        *local
            .get(&net)
            .unwrap_or_else(|| panic!("kernel {}: net {net} is neither input nor state", kernel.id))
    };
    let mut gates = Vec::with_capacity(kernel.gates.len());
    let mut groups = Vec::with_capacity(kernel.gates.len());
    for (i, &gi) in kernel.gates.iter().enumerate() {
        let g = &flat.gates[gi as usize];
        gates.push(FlatGate {
            op: g.op,
            inputs: g.inputs.iter().map(|&n| map_net(n)).collect(),
            output: map_net(g.output),
            owner: 0,
            group: i as u32,
            bit: 0,
        });
        let path = flat
            .groups
            .get(g.group as usize)
            .map(|gr| gr.path.clone())
            .unwrap_or_default();
        groups.push(GateGroup {
            owner: 0,
            path,
            gates: vec![i as u32],
        });
    }
    let resolvers = kernel
        .resolvers
        .iter()
        .map(|&ri| {
            let r = &flat.resolvers[ri as usize];
            Resolver {
                drivers: r.drivers.iter().map(|&n| map_net(n)).collect(),
                termination: r.termination,
                output: map_net(r.output),
            }
        })
        .collect();
    let sources = (0..k)
        .map(|i| Source {
            kind: SourceKind::TopInput {
                pin: 0,
                bit: i as u32,
            },
            output: i as NetIdx,
        })
        .collect();
    let top_inputs = if k > 0 {
        vec![FlatPin {
            name: INPUT_PIN.to_string(),
            dir: Dir::In,
            nets: (0..k as NetIdx).collect(),
        }]
    } else {
        Vec::new()
    };
    let net_names = kernel
        .inputs
        .iter()
        .chain(&kernel.nets)
        .map(|&n| {
            flat.net_names
                .get(n as usize)
                .cloned()
                .unwrap_or_else(|| format!("n{n}"))
        })
        .collect();
    let sub = FlatNetlist {
        num_nets: (k + kernel.nets.len()) as u32,
        gates,
        sources,
        resolvers,
        groups,
        instances: vec![FlatInstance {
            path: String::new(),
            def: format!("kernel{}", kernel.id),
            parent: None,
            children: Vec::new(),
            pins: Vec::new(),
        }],
        undriven_termination: HashMap::new(),
        net_names,
        name_index: HashMap::new(),
        top_inputs,
        top_outputs: Vec::new(),
    };
    (
        sub,
        SubnetMap {
            inputs: kernel.inputs.clone(),
            nets: kernel.nets.clone(),
        },
    )
}

fn op_code(op: FlatOp) -> u32 {
    match op {
        FlatOp::And => 0,
        FlatOp::Or => 1,
        FlatOp::Nand => 2,
        FlatOp::Nor => 3,
        FlatOp::Xor => 4,
        FlatOp::Xnor => 5,
        FlatOp::Not => 6,
        FlatOp::Buf => 7,
        FlatOp::Tri => 8,
    }
}

fn term_code(t: Termination) -> u32 {
    match t {
        Termination::None => 0,
        Termination::PullUp => 1,
        Termination::PullDown => 2,
        Termination::Keeper => 3,
    }
}

/// Canonical structural key of a kernel sub-netlist (built by [`kernel_subnet`]): input count,
/// net count, and every gate/resolver with its local net indices. Names are ignored. Two kernels
/// with equal keys have identical transition tables (in their own input/state orders), which is
/// what [`crate::charz::TableCache`] relies on.
pub fn structural_key(sub: &FlatNetlist) -> Vec<u32> {
    let k = sub.top_inputs.first().map_or(0, |p| p.nets.len());
    let mut key = Vec::with_capacity(4 + sub.gates.len() * 5 + sub.resolvers.len() * 5);
    key.extend([
        k as u32,
        sub.num_nets,
        sub.gates.len() as u32,
        sub.resolvers.len() as u32,
    ]);
    for g in &sub.gates {
        key.push(op_code(g.op));
        key.push(g.inputs.len() as u32);
        key.extend_from_slice(&g.inputs);
        key.push(g.output);
    }
    for r in &sub.resolvers {
        key.push(term_code(r.termination));
        key.push(r.drivers.len() as u32);
        key.extend_from_slice(&r.drivers);
        key.push(r.output);
    }
    key
}
