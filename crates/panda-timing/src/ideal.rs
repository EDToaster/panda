//! Ideal blocks: packaged combinational components evaluated as `out_j(t + L_j) = f_j(in(t))`
//! (SPEC §5.2).
//!
//! # Selection ([`select_ideal_blocks`])
//!
//! An instance of the flattened hierarchy is an ideal block if
//! * it is not the top instance, it is not listed in the glass-box set and contains no listed
//!   instance (a glass-boxed instance is simulated at gate level *including its whole subtree*;
//!   its ancestors are therefore not blocks either, while its siblings still may be);
//! * its subtree contains at least one gate;
//! * the gate graph of its subtree (gates whose owner lies in the subtree, plus the resolvers whose
//!   drivers are all driven by such gates) is acyclic;
//! * none of those resolvers has `keeper` termination (a keeper holds state);
//! * it is the outermost instance with these properties.
//!
//! Sources never belong to a block: the validator keeps `Clock` / `PowerOnReset` / top inputs at
//! top level, and `Constant`s inside a package (e.g. ROM words) are simply block inputs that never
//! change. A resolver with a driver outside the subtree (an `inout` bus shared across the
//! boundary) stays outside the block; the block's private driver nets are then block outputs.
//!
//! * **Inputs**: nets read by the block's elements and not driven by them.
//! * **Outputs**: nets driven inside and read outside the block (by a gate or resolver that is not
//!   in the block) or connected to a top-level output pin.
//! * **Internal nets**: all other nets driven inside.
//! * **Latency** `L_j` of output `j`: the maximum gate depth from the block inputs (depth 0) to the
//!   output; every gate (including `BUF` / `TRI`) counts 1, resolvers count 0. `L_j ≥ 1`.
//!
//! # Evaluation
//!
//! The block keeps a private zero-delay copy of all its nets (inputs and internal nets). When an
//! input net changes during tick `t` (compared at the end of the tick, so a change that is undone
//! within the same tick is not an event), the affected elements are re-evaluated in topological
//! (rank) order with the exact [`panda_netlist::eval`] semantics, and every output whose zero-delay
//! value changed is scheduled to take that value at tick `t + L_j` (transport delay: a timing
//! wheel holds any number of pending changes per output).
//!
//! Blocks without internal resolvers and with at most [`crate::TimingOptions::ideal_lut_bits`]
//! *variable* inputs (inputs not driven by a `Constant` and not undriven) are evaluated through a lazily filled 4-state LUT
//! (SPEC §7.2): with all variable inputs binary, the outputs are looked up; a missing row or an
//! `X`/`Z` variable input falls back to gate evaluation (filling the row when the index is
//! binary). Forcing a constant input of such a block disables its LUT until the next reset.
//!
//! # Internal nets
//!
//! Internal nets are not simulated. [`crate::TimingSim::get`] and `read` always return them
//! evaluated from the block's *current* input values (SPEC §5.2 probing). After a gate-evaluated
//! update they are also written into the simulator's value array; after a LUT hit the block is
//! *stale* and `get`/`read` compute them on demand, while [`crate::TimingSim::values`] may show
//! older internal values until [`crate::TimingSim::refresh_ideal_internal`] is called. Output nets
//! always show their simulated (delayed) value.

use std::collections::{HashMap, HashSet};

use panda_netlist::eval;
use panda_netlist::flat::{Driver, FlatNetlist, GateIdx, InstIdx, NetIdx, SourceKind};
use panda_netlist::{Logic, Termination};

use crate::{GateRec, NONE, Tables, WideGate, eval_rec};

/// One selected ideal block (see the module docs for the definitions).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IdealBlock {
    /// Instance index in [`FlatNetlist::instances`].
    pub inst: InstIdx,
    /// Instance path, e.g. `"cpu/alu"`.
    pub path: String,
    /// Gates inside the block (sorted).
    pub gates: Vec<GateIdx>,
    /// Resolvers inside the block (sorted).
    pub resolvers: Vec<u32>,
    /// Input nets (sorted).
    pub inputs: Vec<NetIdx>,
    /// Output nets (sorted).
    pub outputs: Vec<NetIdx>,
    /// Latency of each output (parallel to `outputs`), in ticks.
    pub latency: Vec<u32>,
    /// Nets driven inside the block that are not outputs (sorted).
    pub internal: Vec<NetIdx>,
}

impl IdealBlock {
    /// The largest output latency.
    pub fn max_latency(&self) -> u32 {
        self.latency.iter().copied().max().unwrap_or(0)
    }
}

/// An element of a block: a gate or a resolver.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ElemRef {
    Gate(GateIdx),
    Res(u32),
}

/// Select the ideal blocks of `nl` (module docs, "Selection"). `glass_box` lists instance paths
/// that must be simulated at gate level (with their subtrees). Blocks are returned in instance
/// order.
pub fn select_ideal_blocks(nl: &FlatNetlist, glass_box: &HashSet<String>) -> Vec<IdealBlock> {
    let ni = nl.instances.len();
    if ni <= 1 {
        return Vec::new();
    }
    let drivers = nl.drivers();
    let fo = nl.fanout();
    // Gates and resolvers per owning instance. A resolver is owned by the deepest common ancestor
    // of its drivers' gates, if all of its drivers are gate-driven.
    let mut gates_of: Vec<Vec<GateIdx>> = vec![Vec::new(); ni];
    for (gi, g) in nl.gates.iter().enumerate() {
        gates_of[g.owner as usize].push(gi as GateIdx);
    }
    let depth_of = |mut i: InstIdx| {
        let mut d = 0usize;
        while let Some(p) = nl.instances[i as usize].parent {
            i = p;
            d += 1;
        }
        d
    };
    let lca = |a: InstIdx, b: InstIdx| {
        let (mut a, mut b) = (a, b);
        let (mut da, mut db) = (depth_of(a), depth_of(b));
        while da > db {
            a = nl.instances[a as usize].parent.unwrap_or(0);
            da -= 1;
        }
        while db > da {
            b = nl.instances[b as usize].parent.unwrap_or(0);
            db -= 1;
        }
        while a != b {
            a = nl.instances[a as usize].parent.unwrap_or(0);
            b = nl.instances[b as usize].parent.unwrap_or(0);
        }
        a
    };
    let mut res_of: Vec<Vec<u32>> = vec![Vec::new(); ni];
    for (ri, r) in nl.resolvers.iter().enumerate() {
        let mut owner: Option<InstIdx> = None;
        let mut ok = !r.drivers.is_empty();
        for &d in &r.drivers {
            match drivers[d as usize] {
                panda_netlist::flat::Driver::Gate(g) => {
                    let o = nl.gates[g as usize].owner;
                    owner = Some(owner.map_or(o, |cur| lca(cur, o)));
                }
                _ => ok = false,
            }
        }
        if ok && let Some(o) = owner {
            res_of[o as usize].push(ri as u32);
        }
    }

    let top_out: HashSet<NetIdx> = nl.top_outputs.iter().flat_map(|p| p.nets.iter().copied()).collect();
    // `glassy[i]`: instance `i` or one of its descendants is glass-boxed.
    let mut glassy = vec![false; ni];
    for (i, fi) in nl.instances.iter().enumerate() {
        if glass_box.contains(&fi.path) {
            let mut cur = Some(i as InstIdx);
            while let Some(c) = cur {
                glassy[c as usize] = true;
                cur = nl.instances[c as usize].parent;
            }
        }
    }
    let mut out = Vec::new();
    let mut stack: Vec<InstIdx> = nl.instances[0].children.iter().rev().copied().collect();
    while let Some(inst) = stack.pop() {
        let fi = &nl.instances[inst as usize];
        if glass_box.contains(&fi.path) {
            continue;
        }
        if glassy[inst as usize] {
            stack.extend(fi.children.iter().rev().copied());
            continue;
        }
        // Collect the subtree.
        let mut gates = Vec::new();
        let mut res = Vec::new();
        let mut sub = vec![inst];
        while let Some(i) = sub.pop() {
            gates.extend_from_slice(&gates_of[i as usize]);
            res.extend_from_slice(&res_of[i as usize]);
            sub.extend_from_slice(&nl.instances[i as usize].children);
        }
        if gates.is_empty() {
            continue;
        }
        gates.sort_unstable();
        res.sort_unstable();
        let keeper = res.iter().any(|&r| nl.resolvers[r as usize].termination == Termination::Keeper);
        let order = if keeper { None } else { topo_order(nl, &drivers, &gates, &res) };
        match order {
            Some(_) => {
                out.push(IdealBlock {
                    inst,
                    path: fi.path.clone(),
                    gates,
                    resolvers: res,
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    latency: Vec::new(),
                    internal: Vec::new(),
                });
            }
            None => stack.extend(fi.children.iter().rev().copied()),
        }
    }
    // Interfaces and latencies (needs the final membership to classify outputs).
    for b in &mut out {
        let mut driven: HashMap<NetIdx, ElemRef> = HashMap::new();
        for &g in &b.gates {
            driven.insert(nl.gates[g as usize].output, ElemRef::Gate(g));
        }
        for &r in &b.resolvers {
            driven.insert(nl.resolvers[r as usize].output, ElemRef::Res(r));
        }
        let mut inputs: HashSet<NetIdx> = HashSet::new();
        for &g in &b.gates {
            inputs.extend(nl.gates[g as usize].inputs.iter().filter(|n| !driven.contains_key(n)));
        }
        for &r in &b.resolvers {
            inputs.extend(nl.resolvers[r as usize].drivers.iter().filter(|n| !driven.contains_key(n)));
        }
        b.inputs = inputs.into_iter().collect();
        b.inputs.sort_unstable();
        let order = topo_order(nl, &drivers, &b.gates, &b.resolvers).expect("block is acyclic");
        let mut depth: HashMap<NetIdx, u32> = HashMap::new();
        for e in order {
            match e {
                ElemRef::Gate(g) => {
                    let fg = &nl.gates[g as usize];
                    let d = fg.inputs.iter().map(|n| depth.get(n).copied().unwrap_or(0)).max().unwrap_or(0);
                    depth.insert(fg.output, d + 1);
                }
                ElemRef::Res(r) => {
                    let fr = &nl.resolvers[r as usize];
                    let d = fr.drivers.iter().map(|n| depth.get(n).copied().unwrap_or(0)).max().unwrap_or(0);
                    depth.insert(fr.output, d);
                }
            }
        }
        let mut nets: Vec<NetIdx> = driven.keys().copied().collect();
        nets.sort_unstable();
        for net in nets {
            let read_outside = fo.gates[net as usize].iter().any(|g| b.gates.binary_search(g).is_err())
                || fo.resolvers[net as usize].iter().any(|r| b.resolvers.binary_search(r).is_err());
            if read_outside || top_out.contains(&net) {
                b.outputs.push(net);
                b.latency.push(depth[&net]);
            } else {
                b.internal.push(net);
            }
        }
    }
    out
}

/// Topological order of the given elements (drivers before readers), or `None` if they contain a
/// cycle.
fn topo_order(
    nl: &FlatNetlist,
    drivers: &[panda_netlist::flat::Driver],
    gates: &[GateIdx],
    res: &[u32],
) -> Option<Vec<ElemRef>> {
    let elems: Vec<ElemRef> =
        gates.iter().map(|&g| ElemRef::Gate(g)).chain(res.iter().map(|&r| ElemRef::Res(r))).collect();
    let local: HashMap<ElemRef, usize> = elems.iter().enumerate().map(|(i, &e)| (e, i)).collect();
    let driver_local = |net: NetIdx| -> Option<usize> {
        let e = match drivers[net as usize] {
            panda_netlist::flat::Driver::Gate(g) => ElemRef::Gate(g),
            panda_netlist::flat::Driver::Resolver(r) => ElemRef::Res(r),
            _ => return None,
        };
        local.get(&e).copied()
    };
    let inputs_of = |e: ElemRef| -> &[NetIdx] {
        match e {
            ElemRef::Gate(g) => &nl.gates[g as usize].inputs,
            ElemRef::Res(r) => &nl.resolvers[r as usize].drivers,
        }
    };
    let mut indeg = vec![0u32; elems.len()];
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); elems.len()];
    for (i, &e) in elems.iter().enumerate() {
        for &n in inputs_of(e) {
            if let Some(d) = driver_local(n) {
                succ[d].push(i);
                indeg[i] += 1;
            }
        }
    }
    let mut ready: Vec<usize> = (0..elems.len()).filter(|&i| indeg[i] == 0).collect();
    let mut order = Vec::with_capacity(elems.len());
    while let Some(i) = ready.pop() {
        order.push(elems[i]);
        for &s in &succ[i] {
            indeg[s] -= 1;
            if indeg[s] == 0 {
                ready.push(s);
            }
        }
    }
    (order.len() == elems.len()).then_some(order)
}

/// A compiled block element. Gate records index [`Blocks::slot_val`].
#[derive(Clone, Copy, Debug)]
enum Elem {
    Gate(GateRec),
    /// Index into [`Blocks::res`].
    Res(u32),
}

/// A compiled resolver inside a block.
#[derive(Clone, Copy, Debug)]
struct BRes {
    start: u32,
    end: u32,
    term: Termination,
    out: u32,
}

/// Static layout of one block: its slots, elements and outputs are contiguous ranges.
#[derive(Clone, Copy, Debug, Default)]
struct Layout {
    slot_start: u32,
    /// Input slots are `slot_start..inputs_end`.
    inputs_end: u32,
    slot_end: u32,
    /// Elements `elem_start..elem_end`, in topological order.
    elem_start: u32,
    elem_end: u32,
    out_start: u32,
    out_end: u32,
    /// LUT index bits: `lut_vars[var_start..var_end]` (bit `i` = the `i`-th variable input).
    var_start: u32,
    var_end: u32,
    /// The block can use a LUT.
    lut_capable: bool,
    /// First LUT row (row `r` of this block is global row `row_base + r`).
    row_base: usize,
    /// First LUT cell (row `r` occupies cells `cell_base + r·n_out ..`).
    cell_base: usize,
}

/// Cap on LUT cells (rows × outputs) per block.
const LUT_MAX_CELLS: usize = 1 << 20;

/// Runtime state of all ideal blocks of a simulator. Empty when ideal blocks are disabled.
///
/// # LUT evaluation
///
/// A block without internal resolvers whose *variable* inputs (inputs not driven by a `Constant`
/// source and not undriven) number at most `lut_bits` gets a lazily filled 4-state LUT (SPEC §7.2): when all variable inputs
/// are binary, the output row for that input vector is looked up; a missing row, or any `X`/`Z`
/// variable input, falls back to gate evaluation (which fills the row when the index is binary).
/// A LUT hit does not evaluate the internal nets: the block is then *stale* and its internal nets
/// are computed on demand ([`Blocks::lazy_value`]) or by [`Blocks::refresh`]. Forcing a constant
/// input of a LUT block disables its LUT until the next reset.
#[derive(Default)]
pub(crate) struct Blocks {
    /// Public description of every block.
    pub(crate) info: Vec<IdealBlock>,
    layout: Vec<Layout>,
    // ---- slots: a block-private copy of every input / internal / output net ----
    slot_net: Vec<NetIdx>,
    slot_val: Vec<Logic>,
    /// Output index of the slot, or `NONE`.
    slot_out: Vec<u32>,
    /// Element driving the slot, or `NONE` for input slots.
    slot_elem: Vec<u32>,
    /// CSR: elements reading slot `s` are `slot_rd[slot_rd_start[s]..slot_rd_start[s+1]]`.
    slot_rd_start: Vec<u32>,
    slot_rd: Vec<u32>,
    // ---- elements ----
    elems: Vec<Elem>,
    elem_rank: Vec<u32>,
    elem_marked: Vec<bool>,
    wide: Vec<WideGate>,
    wide_inputs: Vec<u32>,
    not_tab: u32,
    res: Vec<BRes>,
    r_drivers: Vec<u32>,
    res_contending: Vec<bool>,
    res_buf: Vec<Logic>,
    // ---- outputs ----
    out_net: Vec<NetIdx>,
    out_lat: Vec<u32>,
    out_slot: Vec<u32>,
    out_block: Vec<u32>,
    /// Last value scheduled for each output (LUT-capable blocks only; others use the slot).
    out_last: Vec<Logic>,
    /// CSR: block input nets in the cone of output `j` (for period-2 collapse).
    out_cone_start: Vec<u32>,
    out_cone: Vec<NetIdx>,
    // ---- LUTs ----
    lut_vars: Vec<NetIdx>,
    /// Row-major cells: row `r` of block `b` holds its outputs in order.
    lut_cells: Vec<Logic>,
    /// Per global row: the row has been computed.
    lut_filled: Vec<bool>,
    // ---- per block runtime flags ----
    lut_on: Vec<bool>,
    dirty: Vec<bool>,
    force_full: Vec<bool>,
    stale: Vec<bool>,
    dirty_list: Vec<u32>,
    post: Vec<(u32, u32)>,
    // ---- per net (empty when there are no blocks) ----
    /// Output index driving the net, or `NONE`.
    pub(crate) net_out: Vec<u32>,
    /// Block owning an internal net (driven inside a block, not an output), or `NONE`.
    net_iblock: Vec<u32>,
    /// Owning block of each gate, or `NONE`.
    pub(crate) gate_block: Vec<u32>,
    /// Owning block of each resolver, or `NONE`.
    pub(crate) res_block: Vec<u32>,
    // ---- input entries (fanout tags) ----
    /// Input entries per net: `(net, entry)` pairs, sorted by net.
    pub(crate) net_entries: Vec<(NetIdx, u32)>,
    entry_slot: Vec<u32>,
    entry_block: Vec<u32>,
    /// The entry's net is constant (Constant source or undriven): not a LUT index bit.
    entry_const: Vec<bool>,
    pub(crate) entry_marked: Vec<bool>,
    pub(crate) changed: Vec<u32>,
    // ---- scheduling ----
    buckets: Vec<Vec<u32>>,
    max_touched: usize,
    wheel: Vec<Vec<(NetIdx, Logic)>>,
    mask: u64,
    /// Output changes scheduled and not yet applied.
    pub(crate) pending: usize,
    /// Element evaluations since the last reset.
    pub(crate) evals: u64,
    /// LUT hits since the last reset.
    pub(crate) lut_hits: u64,
}

impl Blocks {
    /// Compile the ideal blocks of `nl` (none if `enabled` is false). Blocks with at most
    /// `lut_bits` variable inputs use LUTs (`0` disables them).
    pub(crate) fn new(
        nl: &FlatNetlist,
        enabled: bool,
        glass_box: &HashSet<String>,
        lut_bits: u32,
        tabs: &Tables,
    ) -> Blocks {
        let mut b = Blocks { not_tab: tabs.not_tab, ..Default::default() };
        if !enabled {
            return b;
        }
        b.info = select_ideal_blocks(nl, glass_box);
        if b.info.is_empty() {
            return b;
        }
        let n = nl.num_nets as usize;
        b.net_out = vec![NONE; n];
        b.net_iblock = vec![NONE; n];
        b.gate_block = vec![NONE; nl.gates.len()];
        b.res_block = vec![NONE; nl.resolvers.len()];
        let drivers = nl.drivers();
        let is_const = |net: NetIdx| match drivers[net as usize] {
            Driver::None => true,
            Driver::Source(s) => matches!(nl.sources[s as usize].kind, SourceKind::Constant(_)),
            _ => false,
        };
        let mut net_slot: HashMap<NetIdx, u32> = HashMap::new();
        let mut readers: Vec<Vec<u32>> = Vec::new();
        let mut max_rank = 0u32;
        let mut max_lat = 0u32;
        let mut rows = 0usize;
        let mut cells = 0usize;
        let infos = std::mem::take(&mut b.info);
        for (bi, info) in infos.iter().enumerate() {
            let bi32 = bi as u32;
            net_slot.clear();
            for &g in &info.gates {
                b.gate_block[g as usize] = bi32;
            }
            for &r in &info.resolvers {
                b.res_block[r as usize] = bi32;
            }
            let mut lay = Layout { slot_start: b.slot_net.len() as u32, ..Default::default() };
            let new_slot = |b: &mut Blocks, readers: &mut Vec<Vec<u32>>, net: NetIdx| -> u32 {
                let s = b.slot_net.len() as u32;
                b.slot_net.push(net);
                b.slot_val.push(Logic::X);
                b.slot_out.push(NONE);
                b.slot_elem.push(NONE);
                readers.push(Vec::new());
                s
            };
            lay.var_start = b.lut_vars.len() as u32;
            for &net in &info.inputs {
                let s = new_slot(&mut b, &mut readers, net);
                net_slot.insert(net, s);
                let e = b.entry_slot.len() as u32;
                b.entry_slot.push(s);
                b.entry_block.push(bi32);
                let c = is_const(net);
                b.entry_const.push(c);
                if !c {
                    b.lut_vars.push(net);
                }
                b.net_entries.push((net, e));
            }
            lay.var_end = b.lut_vars.len() as u32;
            lay.inputs_end = b.slot_net.len() as u32;
            for &net in info.outputs.iter().chain(&info.internal) {
                let s = new_slot(&mut b, &mut readers, net);
                net_slot.insert(net, s);
            }
            lay.slot_end = b.slot_net.len() as u32;
            for &net in &info.internal {
                b.net_iblock[net as usize] = bi32;
            }
            // Elements in topological order; ranks from the drivers' ranks.
            lay.elem_start = b.elems.len() as u32;
            let order = topo_order(nl, &drivers, &info.gates, &info.resolvers).expect("block is acyclic");
            let mut slot_rank: HashMap<u32, u32> = HashMap::new();
            for e in order {
                let el = b.elems.len() as u32;
                let (ins, out_net): (Vec<u32>, NetIdx) = match e {
                    ElemRef::Gate(g) => {
                        let fg = &nl.gates[g as usize];
                        (fg.inputs.iter().map(|n| net_slot[n]).collect(), fg.output)
                    }
                    ElemRef::Res(r) => {
                        let fr = &nl.resolvers[r as usize];
                        (fr.drivers.iter().map(|n| net_slot[n]).collect(), fr.output)
                    }
                };
                let out = net_slot[&out_net];
                let rank = 1 + ins.iter().map(|s| slot_rank.get(s).copied().unwrap_or(0)).max().unwrap_or(0);
                max_rank = max_rank.max(rank);
                slot_rank.insert(out, rank);
                b.slot_elem[out as usize] = el;
                let mut seen: Vec<u32> = ins.clone();
                seen.sort_unstable();
                seen.dedup();
                for s in seen {
                    readers[s as usize].push(el);
                }
                let elem = match e {
                    ElemRef::Gate(g) => {
                        let op = nl.gates[g as usize].op;
                        Elem::Gate(tabs.compile(op, &ins, out, &mut b.wide, &mut b.wide_inputs, g))
                    }
                    ElemRef::Res(r) => {
                        let start = b.r_drivers.len() as u32;
                        b.r_drivers.extend_from_slice(&ins);
                        b.res.push(BRes {
                            start,
                            end: b.r_drivers.len() as u32,
                            term: nl.resolvers[r as usize].termination,
                            out,
                        });
                        b.res_contending.push(false);
                        Elem::Res(b.res.len() as u32 - 1)
                    }
                };
                b.elems.push(elem);
                b.elem_rank.push(rank);
            }
            lay.elem_end = b.elems.len() as u32;
            // Outputs and their input cones.
            lay.out_start = b.out_net.len() as u32;
            for (j, &net) in info.outputs.iter().enumerate() {
                let oi = b.out_net.len() as u32;
                let s = net_slot[&net];
                b.slot_out[s as usize] = oi;
                b.net_out[net as usize] = oi;
                b.out_net.push(net);
                b.out_lat.push(info.latency[j]);
                b.out_block.push(bi32);
                max_lat = max_lat.max(info.latency[j]);
                b.out_slot.push(s);
                if b.out_cone_start.is_empty() {
                    b.out_cone_start.push(0);
                }
                let mut seen: HashSet<u32> = HashSet::new();
                let mut work = vec![s];
                let mut cone: Vec<NetIdx> = Vec::new();
                while let Some(s) = work.pop() {
                    if !seen.insert(s) {
                        continue;
                    }
                    let el = b.slot_elem[s as usize];
                    if el == NONE {
                        cone.push(b.slot_net[s as usize]);
                    } else {
                        work.extend(b.elem_inputs(el));
                    }
                }
                cone.sort_unstable();
                b.out_cone.extend(cone);
                b.out_cone_start.push(b.out_cone.len() as u32);
            }
            lay.out_end = b.out_net.len() as u32;
            let k = lay.var_end - lay.var_start;
            let n_out = (lay.out_end - lay.out_start) as usize;
            // Blocks with internal resolvers are always gate-evaluated, so that `Contention`
            // episodes are reported exactly as without LUTs.
            lay.lut_capable = lut_bits > 0
                && n_out > 0
                && info.resolvers.is_empty()
                && k <= lut_bits
                && k < usize::BITS
                && n_out.checked_shl(k).is_some_and(|c| c <= LUT_MAX_CELLS);
            if lay.lut_capable {
                lay.row_base = rows;
                lay.cell_base = cells;
                rows += 1 << k;
                cells += n_out << k;
            }
            b.layout.push(lay);
        }
        b.info = infos;
        b.slot_rd_start.push(0);
        for r in &readers {
            b.slot_rd.extend_from_slice(r);
            b.slot_rd_start.push(b.slot_rd.len() as u32);
        }
        b.net_entries.sort_unstable();
        b.elem_marked = vec![false; b.elems.len()];
        b.entry_marked = vec![false; b.entry_slot.len()];
        b.buckets = vec![Vec::new(); max_rank as usize + 1];
        let w = (u64::from(max_lat) + 1).next_power_of_two();
        b.wheel = vec![Vec::new(); w as usize];
        b.mask = w - 1;
        b.out_last = vec![Logic::X; b.out_net.len()];
        let nb = b.layout.len();
        b.lut_on = b.layout.iter().map(|l| l.lut_capable).collect();
        b.dirty = vec![false; nb];
        b.force_full = vec![false; nb];
        b.stale = vec![false; nb];
        b.lut_cells = vec![Logic::X; cells];
        b.lut_filled = vec![false; rows];
        b
    }

    /// Slots read by element `el`.
    fn elem_inputs(&self, el: u32) -> Vec<u32> {
        match self.elems[el as usize] {
            Elem::Gate(rec) => {
                if rec.tab != 0 {
                    vec![rec.a, rec.b]
                } else {
                    let w = self.wide[rec.a as usize];
                    self.wide_inputs[w.start as usize..w.end as usize].to_vec()
                }
            }
            Elem::Res(r) => {
                let r = self.res[r as usize];
                self.r_drivers[r.start as usize..r.end as usize].to_vec()
            }
        }
    }

    /// Evaluate element `el` reading slot values through `get` (cold path; non-keeper resolvers).
    fn eval_elem_with(&self, el: u32, get: impl Fn(u32) -> Logic) -> (u32, Logic) {
        match self.elems[el as usize] {
            Elem::Gate(rec) => {
                let v = if rec.tab != 0 {
                    eval::lookup2(rec.tab, get(rec.a), get(rec.b))
                } else {
                    let w = self.wide[rec.a as usize];
                    let ins = &self.wide_inputs[w.start as usize..w.end as usize];
                    let mut acc = eval::lookup2(w.tab, get(ins[0]), get(ins[1]));
                    for &i in &ins[2..] {
                        acc = eval::lookup2(w.tab, acc, get(i));
                    }
                    if w.invert { eval::lookup2(self.not_tab, acc, acc) } else { acc }
                };
                (rec.out, v)
            }
            Elem::Res(ri) => {
                let br = self.res[ri as usize];
                let ds: Vec<Logic> =
                    self.r_drivers[br.start as usize..br.end as usize].iter().map(|&d| get(d)).collect();
                (br.out, eval::resolve(&ds, br.term, Logic::X).value)
            }
        }
    }

    /// Zero-delay values of all slots of block `b` computed from the input values in `values`
    /// (index = slot - slot_start). Cold path.
    fn eval_block_local(&self, b: usize, values: &[Logic]) -> Vec<Logic> {
        let l = self.layout[b];
        let base = l.slot_start;
        let mut local: Vec<Logic> = vec![Logic::X; (l.slot_end - base) as usize];
        for s in base..l.inputs_end {
            local[(s - base) as usize] = values[self.slot_net[s as usize] as usize];
        }
        for el in l.elem_start..l.elem_end {
            let (out, v) = self.eval_elem_with(el, |s| local[(s - base) as usize]);
            local[(out - base) as usize] = v;
        }
        local
    }

    /// True if there are no blocks.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.info.is_empty()
    }

    /// True if `net` is an internal net of some block.
    pub(crate) fn is_internal(&self, net: NetIdx) -> bool {
        self.net_iblock.get(net as usize).is_some_and(|&b| b != NONE)
    }

    /// The value of internal net `net` computed from its block's current inputs, if `net` is an
    /// internal net of a *stale* block (otherwise `values[net]` is already exact).
    pub(crate) fn lazy_value(&self, net: NetIdx, values: &[Logic]) -> Option<Logic> {
        let b = *self.net_iblock.get(net as usize)?;
        if b == NONE || !self.stale[b as usize] {
            return None;
        }
        let local = self.eval_block_local(b as usize, values);
        let l = self.layout[b as usize];
        let s = (l.inputs_end..l.slot_end).find(|&s| self.slot_net[s as usize] == net)?;
        Some(local[(s - l.slot_start) as usize])
    }

    /// True if some block's internal nets in the simulator's value array are out of date.
    pub(crate) fn any_stale(&self) -> bool {
        self.stale.iter().any(|&s| s)
    }

    /// Write the exact internal-net values of every stale block into `values` (a copy of the
    /// simulator's values; the block state is not changed).
    pub(crate) fn fill_stale(&self, values: &mut [Logic]) {
        for b in 0..self.layout.len() {
            if self.stale[b] {
                let local = self.eval_block_local(b, values);
                let l = self.layout[b];
                for s in l.inputs_end..l.slot_end {
                    let net = self.slot_net[s as usize];
                    if self.net_iblock[net as usize] != NONE {
                        values[net as usize] = local[(s - l.slot_start) as usize];
                    }
                }
            }
        }
    }

    /// Bring the internal nets of every stale block up to date in `values` (gate evaluation from
    /// the current inputs). Outputs are unaffected (they already have their LUT values).
    pub(crate) fn refresh(&mut self, values: &mut [Logic], t: u64, contention: &mut Vec<(NetIdx, Vec<Logic>)>) {
        for b in 0..self.layout.len() as u32 {
            if self.stale[b as usize] {
                self.make_dirty(b, true);
            }
        }
        self.process(values, t, contention);
    }

    /// Block input nets in the cone of the block output driving `net`, if `net` is one.
    pub(crate) fn output_cone(&self, net: NetIdx) -> Option<&[NetIdx]> {
        self.output_cone_lat(net).map(|(c, _)| c)
    }

    /// [`Blocks::output_cone`] and the output's latency.
    pub(crate) fn output_cone_lat(&self, net: NetIdx) -> Option<(&[NetIdx], u32)> {
        let j = *self.net_out.get(net as usize)?;
        if j == NONE {
            return None;
        }
        let j = j as usize;
        Some((
            &self.out_cone[self.out_cone_start[j] as usize..self.out_cone_start[j + 1] as usize],
            self.out_lat[j],
        ))
    }

    /// `net` (a block input) is being forced: if it is a constant input of a LUT block, that
    /// block's LUT no longer describes the circuit — disable it until the next reset.
    pub(crate) fn forcing(&mut self, net: NetIdx) {
        let lo = self.net_entries.partition_point(|&(n, _)| n < net);
        for i in lo..self.net_entries.len() {
            let (n, e) = self.net_entries[i];
            if n != net {
                break;
            }
            if self.entry_const[e as usize] {
                let b = self.entry_block[e as usize] as usize;
                if self.lut_on[b] {
                    self.lut_on[b] = false;
                    // The block now runs on its slots: bring them up to date first.
                    self.make_dirty(b as u32, true);
                }
            }
        }
    }

    /// Power-on: forget all pending output changes and schedule a full evaluation of every block
    /// (run [`Blocks::process`] afterwards). Internal nets must already be X in `values`.
    pub(crate) fn reset(&mut self) {
        if self.is_empty() {
            return;
        }
        for w in &mut self.wheel {
            w.clear();
        }
        self.pending = 0;
        self.evals = 0;
        self.lut_hits = 0;
        self.slot_val.fill(Logic::X);
        self.out_last.fill(Logic::X);
        self.res_contending.fill(false);
        self.stale.fill(false);
        for (b, l) in self.layout.iter().enumerate() {
            if l.lut_capable && !self.lut_on[b] {
                // Disabled by forcing a constant: rows filled since then are suspect.
                let rows = 1usize << (l.var_end - l.var_start);
                self.lut_filled[l.row_base..l.row_base + rows].fill(false);
                self.lut_on[b] = true;
            }
        }
        for &e in &self.changed {
            self.entry_marked[e as usize] = false;
        }
        self.changed.clear();
        for &b in &self.dirty_list {
            self.dirty[b as usize] = false;
            self.force_full[b as usize] = false;
        }
        self.dirty_list.clear();
        // Every non-LUT input is compared afresh; every LUT block is fully evaluated; every
        // element is evaluated.
        for e in 0..self.entry_slot.len() as u32 {
            self.entry_marked[e as usize] = true;
            self.changed.push(e);
        }
        for b in 0..self.layout.len() as u32 {
            if self.lut_on[b as usize] {
                self.make_dirty(b, true);
            }
        }
        for b in &mut self.buckets {
            b.clear();
        }
        for (el, &rank) in self.elem_rank.iter().enumerate() {
            self.elem_marked[el] = true;
            self.buckets[rank as usize].push(el as u32);
        }
        self.max_touched = self.buckets.len() - 1;
    }

    /// The scheduled view of output `net` was invalidated (period-2 collapse forced it to X):
    /// re-evaluate its driver, so its value is scheduled again (the block analogue of
    /// re-scheduling a gate).
    ///
    /// Changes of the output still in flight (scheduled from older input values) are dropped:
    /// in the block's gates they would be carried by internal nets that alternate with the output
    /// and collapse with it, so they must not land after the collapse.
    pub(crate) fn reschedule_output(&mut self, net: NetIdx) {
        if self.pending > 0 {
            for slot in &mut self.wheel {
                let before = slot.len();
                slot.retain(|&(n, _)| n != net);
                self.pending -= before - slot.len();
            }
        }
        let j = self.net_out[net as usize] as usize;
        let b = self.out_block[j];
        if self.lut_on[b as usize] {
            self.out_last[j] = Logic::X;
            self.make_dirty(b, false);
        } else {
            let s = self.out_slot[j] as usize;
            self.slot_val[s] = Logic::X;
            let el = self.slot_elem[s];
            self.mark(el);
        }
    }

    #[inline]
    fn make_dirty(&mut self, b: u32, force: bool) {
        let bi = b as usize;
        self.force_full[bi] |= force;
        if !self.dirty[bi] {
            self.dirty[bi] = true;
            self.dirty_list.push(b);
        }
    }

    #[inline]
    fn mark(&mut self, el: u32) {
        if !self.elem_marked[el as usize] {
            self.elem_marked[el as usize] = true;
            let r = self.elem_rank[el as usize] as usize;
            self.buckets[r].push(el);
            self.max_touched = self.max_touched.max(r);
        }
    }

    #[inline]
    fn mark_readers(&mut self, s: u32) {
        let s = s as usize;
        for i in self.slot_rd_start[s]..self.slot_rd_start[s + 1] {
            let el = self.slot_rd[i as usize];
            self.mark(el);
        }
    }

    /// Copy input slot `s` from `values`; mark its readers if it changed.
    #[inline]
    fn sync_input(&mut self, s: u32, values: &[Logic]) {
        let v = values[self.slot_net[s as usize] as usize];
        if self.slot_val[s as usize] != v {
            self.slot_val[s as usize] = v;
            self.mark_readers(s);
        }
    }

    /// True if any block has work to do at the end of the current tick.
    #[inline]
    pub(crate) fn has_work(&self) -> bool {
        !self.changed.is_empty() || self.max_touched > 0 || !self.dirty_list.is_empty()
    }

    /// LUT row index of block `b` for the current values, or `None` if a variable input is not
    /// binary.
    #[inline]
    fn lut_index(&self, b: usize, values: &[Logic]) -> Option<usize> {
        let l = self.layout[b];
        let mut idx = 0usize;
        for (i, &net) in self.lut_vars[l.var_start as usize..l.var_end as usize].iter().enumerate() {
            match values[net as usize] {
                Logic::Zero => {}
                Logic::One => idx |= 1 << i,
                _ => return None,
            }
        }
        Some(idx)
    }

    #[inline]
    fn schedule(&mut self, j: usize, v: Logic, t: u64) {
        let at = (t + u64::from(self.out_lat[j])) & self.mask;
        self.wheel[at as usize].push((self.out_net[j], v));
        self.pending += 1;
    }

    /// Apply this tick's input changes and re-evaluate the affected blocks; schedule output
    /// changes at `t + L_j`, write internal nets into `values` (except for LUT hits, which leave
    /// the block stale). Contention on internal resolvers is reported through `contention` as
    /// `(net, driver values)`.
    pub(crate) fn process(&mut self, values: &mut [Logic], t: u64, contention: &mut Vec<(NetIdx, Vec<Logic>)>) {
        // 1. Input changes: LUT blocks become dirty, other blocks update their input slots.
        let changed = std::mem::take(&mut self.changed);
        for &e in &changed {
            self.entry_marked[e as usize] = false;
            let b = self.entry_block[e as usize];
            if self.lut_on[b as usize] {
                self.make_dirty(b, false);
            } else {
                self.sync_input(self.entry_slot[e as usize], values);
            }
        }
        let mut changed = changed;
        changed.clear();
        self.changed = changed;

        // 2. Dirty blocks: LUT hit, or fall back to gate evaluation.
        let dirty = std::mem::take(&mut self.dirty_list);
        for &b in &dirty {
            let bi = b as usize;
            self.dirty[bi] = false;
            let force = std::mem::replace(&mut self.force_full[bi], false);
            let l = self.layout[bi];
            let idx = if self.lut_on[bi] { self.lut_index(bi, values) } else { None };
            if let Some(i) = idx {
                let n_out = (l.out_end - l.out_start) as usize;
                let row = l.cell_base + i * n_out;
                if !force && self.lut_filled[l.row_base + i] {
                    self.lut_hits += 1;
                    for k in 0..n_out {
                        let j = l.out_start as usize + k;
                        let v = self.lut_cells[row + k];
                        if v != self.out_last[j] {
                            self.out_last[j] = v;
                            self.schedule(j, v, t);
                        }
                    }
                    self.stale[bi] = true;
                    continue;
                }
            }
            for s in l.slot_start..l.inputs_end {
                self.sync_input(s, values);
            }
            if self.lut_on[bi] || force {
                self.post.push((b, idx.map_or(NONE, |i| i as u32)));
            }
        }
        let mut dirty = dirty;
        dirty.clear();
        self.dirty_list = dirty;

        // 3. Gate evaluation in rank order (all blocks at once: dependencies are block-local).
        let mut r = 1;
        while r <= self.max_touched {
            let list = std::mem::take(&mut self.buckets[r]);
            self.evals += list.len() as u64;
            for &el in &list {
                self.elem_marked[el as usize] = false;
                let (out, v) = match self.elems[el as usize] {
                    Elem::Gate(rec) => {
                        (rec.out, eval_rec(rec, &self.slot_val, &self.wide, &self.wide_inputs, self.not_tab))
                    }
                    Elem::Res(ri) => {
                        let br = self.res[ri as usize];
                        self.res_buf.clear();
                        let ds = &self.r_drivers[br.start as usize..br.end as usize];
                        self.res_buf.extend(ds.iter().map(|&d| self.slot_val[d as usize]));
                        let res = eval::resolve(&self.res_buf, br.term, self.slot_val[br.out as usize]);
                        let c = res.contention && t > 0;
                        if c && !self.res_contending[ri as usize] {
                            contention.push((self.slot_net[br.out as usize], self.res_buf.clone()));
                        }
                        self.res_contending[ri as usize] = c;
                        (br.out, res.value)
                    }
                };
                let o = out as usize;
                if self.slot_val[o] == v {
                    continue;
                }
                self.slot_val[o] = v;
                let j = self.slot_out[o];
                if j == NONE {
                    values[self.slot_net[o] as usize] = v;
                } else if !self.lut_on[self.out_block[j as usize] as usize] {
                    self.schedule(j as usize, v, t);
                }
                self.mark_readers(out);
            }
            let mut list = list;
            list.clear();
            self.buckets[r] = list;
            r += 1;
        }
        self.max_touched = 0;

        // 4. LUT (and forced) blocks evaluated by gates: fill the row, schedule changed outputs.
        let post = std::mem::take(&mut self.post);
        for &(b, idx) in &post {
            let bi = b as usize;
            self.stale[bi] = false;
            let l = self.layout[bi];
            let n_out = (l.out_end - l.out_start) as usize;
            if !self.lut_on[bi] {
                // A block whose LUT was just disabled: its outputs are scheduled from the slots
                // from now on; reconcile the last LUT-scheduled values once.
                for j in l.out_start as usize..l.out_end as usize {
                    let v = self.slot_val[self.out_slot[j] as usize];
                    if v != self.out_last[j] {
                        self.out_last[j] = v;
                        self.schedule(j, v, t);
                    }
                }
                continue;
            }
            if idx != NONE {
                self.lut_filled[l.row_base + idx as usize] = true;
            }
            for k in 0..n_out {
                let j = l.out_start as usize + k;
                let v = self.slot_val[self.out_slot[j] as usize];
                if idx != NONE {
                    self.lut_cells[l.cell_base + idx as usize * n_out + k] = v;
                }
                if v != self.out_last[j] {
                    self.out_last[j] = v;
                    self.schedule(j, v, t);
                }
            }
        }
        let mut post = post;
        post.clear();
        self.post = post;
    }

    /// Take the output changes due at tick `t` (the caller applies them and gives the buffer back
    /// with [`Blocks::restore_due`]).
    #[inline]
    pub(crate) fn take_due(&mut self, t: u64) -> Vec<(NetIdx, Logic)> {
        if self.pending == 0 {
            return Vec::new();
        }
        let list = std::mem::take(&mut self.wheel[(t & self.mask) as usize]);
        self.pending -= list.len();
        list
    }

    /// Return the buffer taken by [`Blocks::take_due`] (keeps its capacity).
    #[inline]
    pub(crate) fn restore_due(&mut self, t: u64, mut list: Vec<(NetIdx, Logic)>) {
        if list.capacity() == 0 {
            return;
        }
        list.clear();
        let slot = &mut self.wheel[(t & self.mask) as usize];
        if slot.is_empty() {
            *slot = list;
        }
    }
}
