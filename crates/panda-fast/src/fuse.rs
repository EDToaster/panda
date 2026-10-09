//! The interpreter encoding of the combinational ops ([`crate::ir::HOp`]) with gate fusion.
//!
//! Within a partition, a gate whose output is read by exactly one op — another gate of the same
//! partition — and by nothing else (no later partition, no kernel: the op does not notify) is
//! *absorbed* into its reader, as long as the fused cluster reads at most [`MAX_FUSE_INPUTS`]
//! distinct nets. A cluster of two or more gates (or a single N-input gate with at most
//! [`MAX_FUSE_INPUTS`] inputs) becomes one *fused* op: a 4-state truth table over its input
//! nets (256 entries of 2-bit codes, eight `u64` words), computed by evaluating the cluster's
//! gates with the reference gate semantics for every input combination, so it is exact for
//! `X`/`Z` by construction. Measured on Panda-8 this replaces the 2:1 muxes and the adder's
//! carry logic by one op each (a gate op costs ~20 instructions in the interpreter, a fused op
//! ~25).
//!
//! The absorbed gates' output nets are not written by the hot path; they are internal to their
//! cluster (no other reader) and are recomputed on demand from [`Lowered::fcold`] when the
//! values are read (like LUT-internal nets).
//!
//! [`lower`] is deterministic: `VerifyPass` recomputes it and checks that the program holds
//! exactly its result (so the encoding provably matches [`Program::ops`]).

use std::collections::HashMap;

use panda_netlist::Logic;
use panda_netlist::eval::lookup2;

use crate::ir::{HOp, NONE, Op, Partition, Program};

/// Most distinct input nets of a fused op.
pub const MAX_FUSE_INPUTS: usize = 6;

/// Words of a fused truth table with up to 4 inputs (256 entries × 2 bits).
pub const FTABLE_WORDS: usize = 8;

/// Words of a wide fused truth table (5–6 inputs: 4096 entries × 2 bits).
pub const FTABLE_WIDE_WORDS: usize = 128;

/// Result of [`lower`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Lowered {
    /// The interpreter ops (a fused op takes two slots).
    pub hops: Vec<HOp>,
    /// Per hop slot: the change-propagation record of the slot's output, or [`NONE`].
    pub hrec: Vec<u32>,
    /// Per partition: its hop range.
    pub hpart: Vec<Partition>,
    /// Fused truth tables ([`FTABLE_WORDS`] or, wide, [`FTABLE_WIDE_WORDS`] words each; entry `i`
    /// = bits `2i..2i+2` of word `i / 32`, index `i = c0 | c1 << 2 | c2 << 4 | ...` over the input
    /// codes).
    pub ftables: Vec<u64>,
    /// The absorbed gates (topological order): recompute the internal nets on demand.
    pub fcold: Vec<Op>,
}

/// Input nets of a gate op (deduplicated, first-occurrence order); `None` for other ops.
fn gate_inputs(prog: &Program, op: &Op) -> Option<Vec<u32>> {
    let mut v: Vec<u32> = match *op {
        Op::Gate2 { a, b, .. } => vec![a, b],
        Op::GateN { args, n, .. } => prog.args[args as usize..(args + n) as usize].to_vec(),
        _ => return None,
    };
    let mut seen = Vec::with_capacity(v.len());
    v.retain(|x| {
        if seen.contains(x) {
            false
        } else {
            seen.push(*x);
            true
        }
    });
    Some(v)
}

/// Every net read by an op (with repeats).
fn op_reads(prog: &Program, op: &Op) -> Vec<u32> {
    match *op {
        Op::Gate2 { a, b, .. } => vec![a, b],
        Op::GateN { args, n, .. } => prog.args[args as usize..(args + n) as usize].to_vec(),
        Op::Resolve { res, .. } => {
            let r = prog.resolvers[res as usize];
            prog.args[r.args as usize..(r.args + r.n) as usize].to_vec()
        }
        Op::Lut { lut } => prog.lut_support(&prog.luts[lut as usize]).to_vec(),
    }
}

/// Code of a gate op's output given a lookup of its input codes.
fn eval_gate(prog: &Program, op: &Op, val: &dyn Fn(u32) -> u8) -> u8 {
    match *op {
        Op::Gate2 { t, a, b, .. } => lookup2(
            prog.tables[t as usize],
            Logic::from_code(val(a)),
            Logic::from_code(val(b)),
        )
        .code(),
        Op::GateN {
            fold,
            invert,
            args,
            n,
            ..
        } => {
            // Fold with the 2-input tables (the reference semantics of the N-input families).
            let mut acc = match fold {
                crate::ir::table::AND => Logic::One,
                _ => Logic::Zero,
            };
            for &x in &prog.args[args as usize..(args + n) as usize] {
                acc = lookup2(prog.tables[fold as usize], acc, Logic::from_code(val(x)));
            }
            if invert {
                acc = panda_netlist::eval::not1(acc);
            }
            acc.code()
        }
        _ => unreachable!("only gates are fused"),
    }
}

/// Build the interpreter encoding of `prog`'s partitions, fusing clusters of up to `max_inputs`
/// inputs (`< 2`: no fusion; capped at [`MAX_FUSE_INPUTS`]).
pub fn lower(prog: &Program, max_inputs: usize) -> Result<Lowered, String> {
    let max_inputs = max_inputs.min(MAX_FUSE_INPUTS);
    let fuse = max_inputs >= 2;
    let mut out = Lowered::default();
    let mut tables: HashMap<Vec<u64>, u32> = HashMap::new();
    for part in &prog.partitions {
        let (s, e) = (part.start as usize, part.end as usize);
        let ops = &prog.ops[s..e];
        // Readers of each net inside the partition, and the op driving it.
        let mut readers: HashMap<u32, (u32, usize)> = HashMap::new();
        let mut driver: HashMap<u32, usize> = HashMap::new();
        for (i, op) in ops.iter().enumerate() {
            let mut reads = op_reads(prog, op);
            reads.sort_unstable();
            reads.dedup();
            for x in reads {
                let r = readers.entry(x).or_insert((0, i));
                r.0 += 1;
                r.1 = i;
            }
            if !matches!(op, Op::Lut { .. }) {
                driver.insert(op.output(), i);
            }
        }
        // Greedy clustering in topological order.
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); ops.len()];
        let mut inputs: Vec<Vec<u32>> = vec![Vec::new(); ops.len()];
        let mut absorbed = vec![false; ops.len()];
        for (i, op) in ops.iter().enumerate() {
            let Some(ins) = gate_inputs(prog, op) else {
                continue;
            };
            members[i] = vec![i];
            inputs[i] = ins;
            if !fuse || inputs[i].len() > max_inputs {
                continue;
            }
            for x in inputs[i].clone() {
                let Some(&d) = driver.get(&x) else {
                    continue;
                };
                let dop = &ops[d];
                let absorbable = d < i
                    && !absorbed[d]
                    && !members[d].is_empty()
                    && !dop.notifies()
                    && readers.get(&x).is_some_and(|&(c, r)| c == 1 && r == i)
                    && inputs[d].len() <= max_inputs;
                if !absorbable {
                    continue;
                }
                let mut union: Vec<u32> = inputs[i].iter().copied().filter(|&y| y != x).collect();
                for &y in &inputs[d] {
                    if !union.contains(&y) {
                        union.push(y);
                    }
                }
                if union.len() > max_inputs {
                    continue;
                }
                absorbed[d] = true;
                let m = std::mem::take(&mut members[d]);
                members[i].extend(m);
                inputs[i] = union;
            }
            members[i].sort_unstable();
        }
        let hstart = out.hops.len() as u32;
        for (i, op) in ops.iter().enumerate() {
            if absorbed[i] {
                out.fcold.push(*op);
                continue;
            }
            let rec = prog.op_rec[s + i];
            let notify = if op.notifies() { HOp::NOTIFY } else { 0 };
            let check = |net: u32| -> Result<u32, String> {
                if net > HOp::NET {
                    Err("net index too large for the interpreter encoding".into())
                } else {
                    Ok(net)
                }
            };
            let fused_kind = fuse
                && match *op {
                    Op::Gate2 { .. } => members[i].len() >= 2,
                    Op::GateN { .. } => !members[i].is_empty() && inputs[i].len() <= max_inputs,
                    _ => false,
                };
            match *op {
                Op::Gate2 {
                    t, out: o, a, b, ..
                } if !fused_kind => {
                    out.hops.push(HOp {
                        table: prog.tables[t as usize],
                        out: check(o)? | notify,
                        a: check(a)?,
                        b: check(b)?,
                    });
                    out.hrec.push(rec);
                }
                _ if fused_kind => {
                    let ins = &inputs[i];
                    let wide = ins.len() > 4;
                    let table = fused_table(prog, ops, &members[i], ins, wide);
                    let off = out.ftables.len() as u32;
                    let off = *tables.entry(table.clone()).or_insert_with(|| {
                        out.ftables.extend_from_slice(&table);
                        off
                    });
                    let pad = |j: usize| ins.get(j).copied().unwrap_or(ins[0]);
                    let flags = HOp::SPECIAL | HOp::FUSED | if wide { HOp::WIDE } else { 0 };
                    out.hops.push(HOp {
                        table: off,
                        out: check(op.output())? | flags | notify,
                        a: check(pad(0))?,
                        b: check(pad(1))?,
                    });
                    out.hops.push(HOp {
                        table: if wide { check(pad(4))? } else { 0 },
                        out: if wide { check(pad(5))? } else { 0 },
                        a: check(pad(2))?,
                        b: check(pad(3))?,
                    });
                    out.hrec.extend([rec, NONE]);
                }
                _ => {
                    out.hops.push(HOp {
                        table: 0,
                        out: HOp::SPECIAL,
                        a: (s + i) as u32,
                        b: 0,
                    });
                    out.hrec.push(NONE);
                }
            }
        }
        out.hpart.push(Partition {
            start: hstart,
            end: out.hops.len() as u32,
        });
    }
    Ok(out)
}

/// Truth table of a cluster (`members`: op positions in `ops`, topological; the last is the
/// root) over its input nets `ins` (padded inputs are don't-cares).
fn fused_table(prog: &Program, ops: &[Op], members: &[usize], ins: &[u32], wide: bool) -> Vec<u64> {
    let words = if wide {
        FTABLE_WIDE_WORDS
    } else {
        FTABLE_WORDS
    };
    let mut table = vec![0u64; words];
    let root = *members.last().expect("non-empty cluster");
    let mut local: Vec<(u32, u8)> = Vec::with_capacity(members.len() + ins.len());
    for idx in 0..words * 32 {
        local.clear();
        for (j, &x) in ins.iter().enumerate() {
            local.push((x, ((idx >> (2 * j)) & 3) as u8));
        }
        let mut code = 0u8;
        for &m in members {
            let op = &ops[m];
            let val = |net: u32| -> u8 {
                local
                    .iter()
                    .rev()
                    .find(|&&(n, _)| n == net)
                    .map(|&(_, c)| c)
                    .expect("cluster input")
            };
            code = eval_gate(prog, op, &val);
            local.push((op.output(), code));
            if m == root {
                break;
            }
        }
        table[idx / 32] |= u64::from(code) << (2 * (idx % 32));
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_index_layout() {
        // Entry i lives in word i / 32 at bits 2 * (i % 32).
        let mut t = [0u64; FTABLE_WORDS];
        let i = 0b11_10_01_00usize;
        t[i / 32] |= 3 << (2 * (i % 32));
        assert_eq!((t[i >> 5] >> ((i & 31) * 2)) & 3, 3);
    }
}
