//! LUT extraction (SPEC §7.1, docs/M4-CONTRACT.md "LUTs").
//!
//! A LUT replaces a multi-output cone of 2-state gates (AND/OR/XOR families, NOT, BUF; no
//! tri-states or resolvers, whose outputs are not binary for binary inputs) by one table indexed
//! by the cone's binary *support* (the nets it reads that are driven outside it). Constant
//! sources and undriven nets with a binary reset value are folded into the table.
//!
//! Cone selection (MFFC-style, disjoint cones):
//!
//! 1. Per combinational node, its support set (truncated at [`LutParams::max_bits`]).
//! 2. *Seeds*: small-support nodes read by a kernel, by a node outside the small-support region,
//!    or by nothing (top-level outputs / dead logic).
//! 3. Seeds are grouped greedily by support (a group's support union stays ≤ `max_bits`), so
//!    e.g. all data bits of a ROM end up in one LUT.
//! 4. Each group's cone is the fan-in of its seeds through unclaimed small-support nodes. Every
//!    cone node with a reader outside the cone becomes an output (≤ 32); the others are internal
//!    and are materialized lazily. Cones with fewer than [`LutParams::min_gates`] gates are
//!    dropped, as are cones whose replacement by a single op would close a cycle (a path leaving
//!    and re-entering the cone through a tri-state or resolver).
//!
//! Tables are computed bit-parallel (64 support vectors per word).

use std::collections::{BinaryHeap, HashMap};

use panda_compile::{KernelSet, Node};
use panda_netlist::eval::terminate;
use panda_netlist::flat::{FlatNetlist, FlatOp, SourceKind};
use panda_netlist::{Logic, Termination};

use crate::ir::NONE;

/// LUT extraction parameters.
#[derive(Clone, Copy, Debug)]
pub struct LutParams {
    /// Maximum support bits (≤ 16). 0 disables LUT extraction.
    pub max_bits: usize,
    /// Minimum gates absorbed per LUT.
    pub min_gates: usize,
    /// Total entry budget over all LUTs (memory/cache footprint).
    pub max_total_entries: usize,
    /// A seed bucket merges into a group whose support contains its own only if the group's
    /// support is at most this many bits larger (merging small-support logic into a wide LUT
    /// makes it re-evaluate on every change of the wider support).
    pub merge_slack: usize,
}

/// A combinational item of the evaluation order: a gate/resolver or a LUT.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Item {
    /// One gate or resolver.
    Node(Node),
    /// LUT `i` of [`crate::compile::CompileCtx::luts`].
    Lut(u32),
}

/// A LUT under construction.
#[derive(Clone, Debug)]
pub struct LutBuild {
    /// Support nets (bit `i` of the table index = `support[i]`).
    pub support: Vec<u32>,
    /// Output nets (bit `j` of an entry = `outputs[j]`).
    pub outputs: Vec<u32>,
    /// The cone's gates in topological order.
    pub cone: Vec<Node>,
    /// `2^support.len()` entries.
    pub entries: Vec<u32>,
}

/// Value of every net that never changes after reset and is binary (constant sources, undriven
/// nets with a binary termination), else `None`. Also returns the set of such nets that are
/// non-binary.
fn fixed_nets(flat: &FlatNetlist) -> (Vec<Option<Logic>>, Vec<bool>) {
    let n = flat.num_nets as usize;
    let mut driven = vec![false; n];
    for g in &flat.gates {
        driven[g.output as usize] = true;
    }
    for r in &flat.resolvers {
        driven[r.output as usize] = true;
    }
    let mut fixed: Vec<Option<Logic>> = vec![None; n];
    let mut fixed_nb = vec![false; n];
    for s in &flat.sources {
        driven[s.output as usize] = true;
        if let SourceKind::Constant(v) = s.kind {
            if v.is_known() {
                fixed[s.output as usize] = Some(v);
            } else {
                fixed_nb[s.output as usize] = true;
            }
        }
    }
    for net in 0..n {
        if driven[net] {
            continue;
        }
        let term = flat
            .undriven_termination
            .get(&(net as u32))
            .copied()
            .unwrap_or(Termination::None);
        // Same as `FastSim::reset`: Z, or the termination's value.
        let v = match term {
            Termination::None => Logic::Z,
            t => terminate(t, Logic::X),
        };
        if v.is_known() {
            fixed[net] = Some(v);
        } else {
            fixed_nb[net] = true;
        }
    }
    (fixed, fixed_nb)
}

fn is_two_state_op(op: FlatOp) -> bool {
    !matches!(op, FlatOp::Tri)
}

/// Merge two sorted, deduplicated nets lists; `None` if the result exceeds `max`.
fn union_small(a: &[u32], b: &[u32], max: usize) -> Option<Vec<u32>> {
    let mut out = Vec::with_capacity((a.len() + b.len()).min(max + 1));
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        let x = match (a.get(i), b.get(j)) {
            (Some(&x), Some(&y)) if x == y => {
                i += 1;
                j += 1;
                x
            }
            (Some(&x), Some(&y)) if x < y => {
                i += 1;
                x
            }
            (Some(_), Some(&y)) => {
                j += 1;
                y
            }
            (Some(&x), None) => {
                i += 1;
                x
            }
            (None, Some(&y)) => {
                j += 1;
                y
            }
            (None, None) => unreachable!(),
        };
        out.push(x);
        if out.len() > max {
            return None;
        }
    }
    Some(out)
}

/// Find LUTs among the combinational nodes `comb` (topological order). Returns the LUTs and the
/// new evaluation order (topological over nodes and LUTs).
pub fn extract_luts(
    flat: &FlatNetlist,
    ks: &KernelSet,
    comb: &[Node],
    p: &LutParams,
) -> (Vec<LutBuild>, Vec<Item>) {
    let max = p.max_bits.min(16);
    let all_nodes = || comb.iter().map(|&n| Item::Node(n)).collect::<Vec<_>>();
    if max == 0 || comb.is_empty() {
        return (Vec::new(), all_nodes());
    }
    let n = flat.num_nets as usize;
    let (fixed, fixed_nb) = fixed_nets(flat);
    let inputs = |node: Node| -> &[u32] {
        match node {
            Node::Gate(g) => &flat.gates[g as usize].inputs,
            Node::Resolver(r) => &flat.resolvers[r as usize].drivers,
        }
    };
    let output = |node: Node| -> u32 {
        match node {
            Node::Gate(g) => flat.gates[g as usize].output,
            Node::Resolver(r) => flat.resolvers[r as usize].output,
        }
    };
    let mut drv = vec![NONE; n];
    for (i, &node) in comb.iter().enumerate() {
        drv[output(node) as usize] = i as u32;
    }
    let eligible: Vec<bool> = comb
        .iter()
        .map(|&node| match node {
            Node::Gate(g) => {
                let gate = &flat.gates[g as usize];
                is_two_state_op(gate.op)
                    && !gate.inputs.is_empty()
                    && gate.inputs.iter().all(|&x| !fixed_nb[x as usize])
            }
            Node::Resolver(_) => false,
        })
        .collect();
    // Support sets (None = too wide or ineligible).
    let mut supp: Vec<Option<Vec<u32>>> = Vec::with_capacity(comb.len());
    for (i, &node) in comb.iter().enumerate() {
        if !eligible[i] {
            supp.push(None);
            continue;
        }
        let mut s: Option<Vec<u32>> = Some(Vec::new());
        for &x in inputs(node) {
            if fixed[x as usize].is_some() {
                continue;
            }
            let d = drv[x as usize];
            let add: Option<Vec<u32>> = if d != NONE && eligible[d as usize] {
                supp[d as usize].clone()
            } else {
                Some(vec![x])
            };
            s = match (s, add) {
                (Some(a), Some(b)) => union_small(&a, &b, max),
                _ => None,
            };
            if s.is_none() {
                break;
            }
        }
        supp.push(s);
    }
    // Readers.
    let mut kernel_read = vec![false; n];
    for k in &ks.kernels {
        for &x in &k.inputs {
            kernel_read[x as usize] = true;
        }
    }
    let mut readers: Vec<Vec<u32>> = vec![Vec::new(); comb.len()];
    for (i, &node) in comb.iter().enumerate() {
        for &x in inputs(node) {
            let d = drv[x as usize];
            if d != NONE {
                readers[d as usize].push(i as u32);
            }
        }
    }
    let inlut = |i: usize| supp[i].is_some();
    let must_out = |i: usize| kernel_read[output(comb[i]) as usize] || readers[i].is_empty();
    let seeds: Vec<usize> = (0..comb.len())
        .filter(|&i| inlut(i) && (must_out(i) || readers[i].iter().any(|&r| !inlut(r as usize))))
        .collect();

    // Group seeds: exact-support buckets, then each bucket merges into the first larger bucket
    // whose support contains it (no support growth, so tables do not grow).
    let mut buckets: HashMap<Vec<u32>, Vec<usize>> = HashMap::new();
    for &sd in &seeds {
        buckets
            .entry(supp[sd].clone().unwrap())
            .or_default()
            .push(sd);
    }
    let mut blist: Vec<(Vec<u32>, Vec<usize>)> = buckets.into_iter().collect();
    blist.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.1[0].cmp(&b.1[0])));
    let mut groups: Vec<(Vec<u32>, Vec<Vec<usize>>)> = Vec::new();
    for (sp, sds) in blist {
        let target = groups.iter().position(|(gs, parts)| {
            parts.iter().map(Vec::len).sum::<usize>() + sds.len() <= 32
                && gs.len() <= sp.len().saturating_add(p.merge_slack)
                && sp.iter().all(|x| gs.binary_search(x).is_ok())
        });
        match target {
            Some(g) => groups[g].1.push(sds),
            None => groups.push((sp, vec![sds])),
        }
    }
    // Attempts in order: each group whole; on rejection its buckets one by one; on rejection
    // a bucket in chunks of 8 seeds.
    // Narrow supports claim their cones first.
    let mut order: Vec<usize> = (0..groups.len()).collect();
    order.sort_by_key(|&g| (groups[g].0.len(), std::cmp::Reverse(groups[g].1.len())));
    let attempts: Vec<(Vec<usize>, u8)> =
        order.iter().map(|&g| (groups[g].1.concat(), 0u8)).collect();
    let mut queue: std::collections::VecDeque<(Vec<usize>, u8)> = attempts.into();
    let group_parts: HashMap<usize, Vec<Vec<usize>>> = groups
        .iter()
        .map(|(_, parts)| (parts[0][0], parts.clone()))
        .collect();

    // Cones (disjoint).
    let mut claimed = vec![NONE; comb.len()];
    let mut in_cone = vec![false; comb.len()];
    let mut luts: Vec<LutBuild> = Vec::new();
    let mut lut_cones: Vec<Vec<usize>> = Vec::new();
    let mut total_entries = 0usize;
    let mut stack = Vec::new();
    while let Some((sds, level)) = queue.pop_front() {
        let mut cone = Vec::new();
        for &sd in &sds {
            if claimed[sd] == NONE && !in_cone[sd] {
                in_cone[sd] = true;
                stack.push(sd);
            }
        }
        while let Some(i) = stack.pop() {
            cone.push(i);
            for &x in inputs(comb[i]) {
                let d = drv[x as usize];
                if d != NONE
                    && inlut(d as usize)
                    && claimed[d as usize] == NONE
                    && !in_cone[d as usize]
                    && fixed[x as usize].is_none()
                {
                    in_cone[d as usize] = true;
                    stack.push(d as usize);
                }
            }
        }
        cone.sort_unstable();
        let mut support: Vec<u32> = Vec::new();
        for &i in &cone {
            for &x in inputs(comb[i]) {
                if fixed[x as usize].is_some() {
                    continue;
                }
                let d = drv[x as usize];
                if d == NONE || !in_cone[d as usize] {
                    support.push(x);
                }
            }
        }
        support.sort_unstable();
        support.dedup();
        let outputs: Vec<u32> = cone
            .iter()
            .filter(|&&i| must_out(i) || readers[i].iter().any(|&r| !in_cone[r as usize]))
            .map(|&i| output(comb[i]))
            .collect();
        let entries = 1usize << support.len().min(31);
        let shape_ok = support.len() <= max && !outputs.is_empty() && outputs.len() <= 32;
        let ok = shape_ok
            && cone.len() >= p.min_gates.max(1)
            && total_entries + entries <= p.max_total_entries;
        if !ok {
            for &i in &cone {
                in_cone[i] = false;
            }
            if !shape_ok && cone.len() >= p.min_gates.max(1) {
                // Retry smaller seed sets.
                match level {
                    0 => {
                        if let Some(parts) = group_parts.get(&sds[0])
                            && parts.len() > 1
                        {
                            for part in parts {
                                queue.push_back((part.clone(), 1));
                            }
                            continue;
                        }
                        if sds.len() > 1 {
                            for c in sds.chunks(8) {
                                queue.push_back((c.to_vec(), 2));
                            }
                        }
                    }
                    1 if sds.len() > 1 => {
                        for c in sds.chunks(8) {
                            queue.push_back((c.to_vec(), 2));
                        }
                    }
                    2 if sds.len() > 1 => {
                        for &sd in &sds {
                            queue.push_back((vec![sd], 3));
                        }
                    }
                    _ => {}
                }
            }
            continue;
        }
        total_entries += entries;
        let id = luts.len() as u32;
        for &i in &cone {
            claimed[i] = id;
            in_cone[i] = false;
        }
        luts.push(LutBuild {
            support,
            outputs,
            cone: cone.iter().map(|&i| comb[i]).collect(),
            entries: Vec::new(),
        });
        lut_cones.push(cone);
    }

    // Topological order over the remaining nodes and the LUTs; drop LUTs on cycles.
    let mut alive = vec![true; luts.len()];
    let items = loop {
        match topo_items(comb, &drv, &claimed, &alive, &lut_cones, &luts, &inputs) {
            Ok(items) => break items,
            Err(cyclic) => {
                for l in cyclic {
                    alive[l as usize] = false;
                    for &i in &lut_cones[l as usize] {
                        claimed[i] = NONE;
                    }
                }
            }
        }
    };
    // Renumber the surviving LUTs.
    let mut new_id = vec![NONE; luts.len()];
    let mut kept = Vec::new();
    for (l, lut) in luts.into_iter().enumerate() {
        if alive[l] {
            new_id[l] = kept.len() as u32;
            kept.push(lut);
        }
    }
    let items = items
        .into_iter()
        .map(|it| match it {
            Item::Lut(l) => Item::Lut(new_id[l as usize]),
            it => it,
        })
        .collect();
    for lut in &mut kept {
        lut.entries = lut_table(flat, lut, &fixed);
    }
    (kept, items)
}

/// Kahn's algorithm over the unclaimed nodes and the alive LUTs (priority: original position).
/// `Err` lists the LUTs left on a cycle.
#[allow(clippy::too_many_arguments)]
fn topo_items<'a>(
    comb: &[Node],
    drv: &[u32],
    claimed: &[u32],
    alive: &[bool],
    lut_cones: &[Vec<usize>],
    luts: &[LutBuild],
    inputs: &dyn Fn(Node) -> &'a [u32],
) -> Result<Vec<Item>, Vec<u32>> {
    // Item ids: comb index for nodes, comb.len() + l for LUTs. Driver item of a comb position.
    let nl = luts.len();
    let ni = comb.len() + nl;
    let item_of_pos = |i: usize| -> usize {
        if claimed[i] != NONE {
            comb.len() + claimed[i] as usize
        } else {
            i
        }
    };
    let present = |it: usize| -> bool {
        if it < comb.len() {
            claimed[it] == NONE
        } else {
            alive[it - comb.len()]
        }
    };
    let item_inputs = |it: usize| -> Vec<u32> {
        if it < comb.len() {
            inputs(comb[it]).to_vec()
        } else {
            luts[it - comb.len()].support.clone()
        }
    };
    let mut indeg = vec![0u32; ni];
    let mut succ: Vec<Vec<u32>> = vec![Vec::new(); ni];
    #[allow(clippy::needless_range_loop)] // `it` is an item id, not just an index of `indeg`
    for it in 0..ni {
        if !present(it) {
            continue;
        }
        let mut deps: Vec<usize> = item_inputs(it)
            .iter()
            .filter_map(|&x| {
                let d = drv[x as usize];
                (d != NONE).then(|| item_of_pos(d as usize))
            })
            .filter(|&d| d != it)
            .collect();
        deps.sort_unstable();
        deps.dedup();
        indeg[it] = deps.len() as u32;
        for d in deps {
            succ[d].push(it as u32);
        }
    }
    let prio = |it: usize| -> usize {
        if it < comb.len() {
            it
        } else {
            *lut_cones[it - comb.len()].last().unwrap()
        }
    };
    let mut heap = BinaryHeap::new();
    for (it, &d) in indeg.iter().enumerate() {
        if present(it) && d == 0 {
            heap.push(std::cmp::Reverse((prio(it), it)));
        }
    }
    let mut out = Vec::new();
    while let Some(std::cmp::Reverse((_, it))) = heap.pop() {
        out.push(if it < comb.len() {
            Item::Node(comb[it])
        } else {
            Item::Lut((it - comb.len()) as u32)
        });
        for &s in &succ[it] {
            indeg[s as usize] -= 1;
            if indeg[s as usize] == 0 {
                heap.push(std::cmp::Reverse((prio(s as usize), s as usize)));
            }
        }
    }
    let expected = (0..ni).filter(|&it| present(it)).count();
    if out.len() == expected {
        Ok(out)
    } else {
        Err((0..nl)
            .filter(|&l| alive[l] && indeg[comb.len() + l] > 0)
            .map(|l| l as u32)
            .collect())
    }
}

/// Bit-parallel evaluation of a LUT's cone for every binary support vector.
fn lut_table(flat: &FlatNetlist, lut: &LutBuild, fixed: &[Option<Logic>]) -> Vec<u32> {
    let s = lut.support.len();
    let entries = 1usize << s;
    let words = entries.div_ceil(64);
    let mut slot: HashMap<u32, usize> = HashMap::new();
    for (i, &x) in lut.support.iter().enumerate() {
        slot.insert(x, i);
    }
    let base = s;
    for (j, node) in lut.cone.iter().enumerate() {
        let Node::Gate(g) = *node else {
            unreachable!("resolvers are never in a LUT")
        };
        slot.insert(flat.gates[g as usize].output, base + j);
    }
    let nslots = base + lut.cone.len();
    let mut vals = vec![0u64; nslots * words];
    const PAT: [u64; 6] = [
        0xAAAA_AAAA_AAAA_AAAA,
        0xCCCC_CCCC_CCCC_CCCC,
        0xF0F0_F0F0_F0F0_F0F0,
        0xFF00_FF00_FF00_FF00,
        0xFFFF_0000_FFFF_0000,
        0xFFFF_FFFF_0000_0000,
    ];
    for i in 0..s {
        for w in 0..words {
            vals[i * words + w] = if i < 6 {
                PAT[i]
            } else if (w >> (i - 6)) & 1 == 1 {
                !0
            } else {
                0
            };
        }
    }
    let read = |vals: &[u64], x: u32, w: usize| -> u64 {
        match fixed[x as usize] {
            Some(Logic::One) => !0,
            Some(_) => 0,
            None => vals[slot[&x] * words + w],
        }
    };
    for (j, node) in lut.cone.iter().enumerate() {
        let Node::Gate(g) = *node else { unreachable!() };
        let gate = &flat.gates[g as usize];
        for w in 0..words {
            let mut it = gate.inputs.iter().map(|&x| read(&vals, x, w));
            let v = match gate.op {
                FlatOp::And => it.fold(!0, |a, b| a & b),
                FlatOp::Nand => !it.fold(!0, |a, b| a & b),
                FlatOp::Or => it.fold(0, |a, b| a | b),
                FlatOp::Nor => !it.fold(0, |a, b| a | b),
                FlatOp::Xor => it.fold(0, |a, b| a ^ b),
                FlatOp::Xnor => !it.fold(0, |a, b| a ^ b),
                FlatOp::Not => !it.next().unwrap(),
                FlatOp::Buf => it.next().unwrap(),
                FlatOp::Tri => unreachable!("tri-states are never in a LUT"),
            };
            vals[(base + j) * words + w] = v;
        }
    }
    let outs: Vec<usize> = lut.outputs.iter().map(|x| slot[x]).collect();
    (0..entries)
        .map(|a| {
            let (w, b) = (a / 64, a % 64);
            outs.iter().enumerate().fold(0u32, |e, (j, &sl)| {
                e | (((vals[sl * words + w] >> b) & 1) as u32) << j
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn union_small_merges_and_limits() {
        assert_eq!(
            union_small(&[1, 3, 5], &[2, 3, 6], 8),
            Some(vec![1, 2, 3, 5, 6])
        );
        assert_eq!(union_small(&[1, 3, 5], &[2, 3, 6], 4), None);
        assert_eq!(union_small(&[], &[7], 1), Some(vec![7]));
    }
}
