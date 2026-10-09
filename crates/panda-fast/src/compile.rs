//! The compiler: `(FlatNetlist, KernelSet, kernel tables)` → [`Program`].
//!
//! Compilation is a pipeline of [`Pass`]es over a shared [`CompileCtx`]. The default pipeline
//! ([`default_pipeline`]) is:
//!
//! 1. [`OrderPass`] — choose the evaluation order of the combinational nodes (any topological
//!    order gives the same settled values; [`OrderStrategy`] picks a cone-clustered DFS order or
//!    the extraction order).
//! 2. [`LutPass`] — replace small-support, many-gate cones (the ROM, decoders) by LUT items
//!    ([`crate::lut`]); the order stays topological over nodes and LUTs.
//! 3. [`PartitionPass`] — cut the order into contiguous partitions of about
//!    [`CompileOptions::partition_ops`] op weights (a LUT weighs [`CompileOptions::lut_weight`]).
//! 4. [`LevelizePass`] — sort each partition's items by logic level (instruction-level
//!    parallelism; still topological).
//! 5. [`LowerPass`] — emit one [`Op`] per gate/resolver/LUT (scalar 2-bit-code table ops, N-input
//!    folds, resolvers, LUT ops with their cold gate code).
//! 6. [`NotifyPass`] — static change propagation: per net the reader partitions (driver's own
//!    excluded) and reader kernels; `notify` flags on ops.
//! 7. [`KernelPass`] — one [`KernelOp`] per kernel (`KTable` when a table exists, else `KSim`).
//! 8. [`crate::kspec::KernelSpecPass`] — kernel phase specialization (reconstruction tables,
//!    KOpaque / KTransparent per clock tuple, per-vector actions).
//! 9. [`LayoutPass`] — the interpreter's data: per-net change-propagation records (word masks
//!    of reader partitions / kernels plus kernel input-index entries) with the record offsets of
//!    every op, LUT and kernel output, the flip-tracked kernel set, and the interpreter encoding
//!    of the ops with gate fusion ([`crate::fuse::lower`]).
//! 10. [`VerifyPass`] (always appended by [`compile_with`]) — structural checks (topological op
//!     order, every combinational node lowered exactly once, LUT cold code well-formed, record
//!     and table bounds, the interpreter encoding equal to a fresh lowering of the ops), which
//!     the interpreter relies on for unchecked indexing.
//!
//! Slice packing (SPEC §8.7) is not a pass: measured on Panda-8's multi-bit gate groups,
//! gathering bits from the byte-per-net state into words and scattering the results costs
//! ~3× the scalar table ops (see `tests/bench.rs::slice_packing_microbench`).

use std::collections::HashMap;
use std::sync::Arc;

use panda_compile::{KernelSet, KernelTable, Node};
use panda_netlist::flat::{FlatNetlist, FlatOp};

use crate::ir::{
    KIN_DATA, KIN_KERNEL_MASK, KIN_POS_SHIFT, KIN_TRACK, KTableData, KernelOp, KernelOpKind,
    LutDesc, MaskCsr, NONE, NetCsr, Op, Partition, Program, ResolverDesc, gate_tables, table,
};
pub use crate::lut::{Item, LutBuild};
use crate::lut::{LutParams, extract_luts};

/// Evaluation order of the combinational nodes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OrderStrategy {
    /// `KernelSet::comb_order` as produced by kernel extraction.
    KernelSet,
    /// Post-order DFS over fan-in from every sink: each output cone is (mostly) contiguous. The
    /// default: with LUTs and small partitions it skips best (Panda-8 bench_loop, 8-op
    /// partitions: ~460k vs ~440k cycles/s with `KernelSet`).
    #[default]
    ConeDfs,
}

/// Compiler tunables (performance only; every setting gives the same observable behaviour).
#[derive(Clone, Debug)]
pub struct CompileOptions {
    /// Target number of ops per partition (≥ 1). Default 32 (measured on Panda-8 with gate
    /// fusion: 32 to one partition best, 8 is ~7% slower; larger partitions fuse more).
    pub partition_ops: usize,
    /// Combinational evaluation order.
    pub order: OrderStrategy,
    /// LUT support limit (`LUT_MAX_BITS`, ≤ 16); 0 disables LUT extraction. Default 10
    /// (measured on Panda-8 bench_loop: 10–11 beat 12 by ~10%, which merges the ROM into a
    /// 12-input, 16 KB table that re-evaluates on more changes; < 8 cannot hold the ROM).
    pub lut_max_bits: usize,
    /// Minimum gates a LUT must absorb (`LUT_MIN_GATES`). Default 16 (16–24 measured best).
    pub lut_min_gates: usize,
    /// Total LUT entries over the program (cache footprint budget). Default 65536.
    pub lut_max_entries: usize,
    /// Partition weight of one LUT op, in gate ops. Default 4.
    pub lut_weight: usize,
    /// LUT grouping: how many support bits a group may exceed a merged seed bucket's support.
    /// Default 16 (unrestricted subset merging: measured ~10% faster than 0–1).
    pub lut_merge_slack: usize,
    /// Phase-specialize kernels (KOpaque / KTransparent, SPEC §8.5). Default true.
    pub specialize: bool,
    /// Skip the runtime order check for data transitions of Opaque/Transparent phases, which are
    /// provably order-insensitive (see [`crate::kspec`]). Default true; false keeps FastRef's
    /// check on every transition the characterization left open (ablation / safety switch).
    pub data_order_lemma: bool,
    /// Fuse fanout-1 gate chains with at most this many inputs (0: off, else 2–6) into exact
    /// 4-state truth-table ops in the interpreter encoding ([`crate::fuse`]). Default 4.
    pub fuse_inputs: usize,
}

impl Default for CompileOptions {
    fn default() -> Self {
        CompileOptions {
            partition_ops: 32,
            order: OrderStrategy::default(),
            lut_max_bits: 10,
            lut_min_gates: 16,
            lut_max_entries: 1 << 16,
            lut_weight: 4,
            lut_merge_slack: 16,
            specialize: true,
            data_order_lemma: true,
            fuse_inputs: 6,
        }
    }
}

/// State shared by the passes.
pub struct CompileCtx<'a> {
    /// The netlist.
    pub flat: &'a FlatNetlist,
    /// Its kernels.
    pub ks: &'a KernelSet,
    /// Kernel tables (`tables[i]` belongs to kernel `i`).
    pub tables: &'a [Option<Arc<KernelTable>>],
    /// Options.
    pub opts: &'a CompileOptions,
    /// Combinational items in evaluation order (set by [`OrderPass`], rewritten by
    /// [`LutPass`]).
    pub order: Vec<Item>,
    /// LUTs referenced by [`Item::Lut`] (set by [`LutPass`]).
    pub luts: Vec<LutBuild>,
    /// Partition `i` = `order[bounds[i]..bounds[i + 1]]` (set by [`PartitionPass`]).
    pub bounds: Vec<usize>,
    /// Partition of the op driving each net, or [`NONE`] (set by [`LowerPass`]).
    pub driver_part: Vec<u32>,
    /// The program under construction.
    pub program: Program,
}

/// One compiler pass.
pub trait Pass {
    /// Short name (recorded in [`Program::passes`]).
    fn name(&self) -> &'static str;
    /// Transform the context.
    fn run(&self, cx: &mut CompileCtx<'_>) -> Result<(), String>;
}

/// The baseline pass pipeline (see the module docs; [`VerifyPass`] is appended by
/// [`compile_with`]).
pub fn default_pipeline() -> Vec<Box<dyn Pass>> {
    vec![
        Box::new(OrderPass),
        Box::new(LutPass),
        Box::new(PartitionPass),
        Box::new(LevelizePass),
        Box::new(LowerPass),
        Box::new(NotifyPass),
        Box::new(KernelPass),
        Box::new(crate::kspec::KernelSpecPass),
        Box::new(LayoutPass),
    ]
}

/// Compile with [`default_pipeline`].
pub fn compile(
    flat: &FlatNetlist,
    ks: &KernelSet,
    tables: &[Option<Arc<KernelTable>>],
    opts: &CompileOptions,
) -> Result<Program, String> {
    compile_with(flat, ks, tables, opts, &default_pipeline())
}

/// Compile with an explicit pass pipeline ([`VerifyPass`] always runs after it).
pub fn compile_with(
    flat: &FlatNetlist,
    ks: &KernelSet,
    tables: &[Option<Arc<KernelTable>>],
    opts: &CompileOptions,
    passes: &[Box<dyn Pass>],
) -> Result<Program, String> {
    if tables.len() != ks.kernels.len() {
        return Err(format!(
            "{} tables for {} kernels",
            tables.len(),
            ks.kernels.len()
        ));
    }
    let n = flat.num_nets as usize;
    let mut cx = CompileCtx {
        flat,
        ks,
        tables,
        opts,
        order: Vec::new(),
        luts: Vec::new(),
        bounds: Vec::new(),
        driver_part: vec![NONE; n],
        program: Program {
            num_nets: flat.num_nets,
            tables: gate_tables(),
            ops: Vec::new(),
            hops: Vec::new(),
            hrec: Vec::new(),
            hpart: Vec::new(),
            ftables: Vec::new(),
            fcold: Vec::new(),
            args: Vec::new(),
            resolvers: Vec::new(),
            partitions: Vec::new(),
            net_parts: NetCsr::default(),
            net_kernels: NetCsr::default(),
            net_kin: NetCsr::default(),
            net_kpend: MaskCsr::default(),
            notify_off: Vec::new(),
            tracked: Vec::new(),
            tracked_list: Vec::new(),
            notify_rec: Vec::new(),
            op_rec: Vec::new(),
            cold_rec: Vec::new(),
            lut_rec: Vec::new(),
            kernel_out_rec: Vec::new(),
            kout1: Vec::new(),
            net_log: vec![NONE; n],
            num_log: 0,
            kernel_kin_mask: ks
                .kernels
                .iter()
                .map(|k| match k.inputs.len() {
                    n if n >= 32 => u32::MAX,
                    n => (1u32 << n) - 1,
                })
                .collect(),
            kernel_ops: Vec::new(),
            kernel_nets: Vec::new(),
            kernel_outs: Vec::new(),
            kernel_out_nets: Vec::new(),
            ktables: Vec::new(),
            kernel_inputs_total: 0,
            luts: Vec::new(),
            lut_nets: Vec::new(),
            lut_entries: Vec::new(),
            cold_ops: Vec::new(),
            lut_gates: 0,
            lazy_owner: vec![NONE; n],
            kclasses: Vec::new(),
            kernel_class: vec![NONE; ks.kernels.len()],
            timed_nets: Vec::new(),
            num_tuples: 0,
            kernel_words: ks.kernels.len().div_ceil(64) as u32,
            tuple_skip: Vec::new(),
            tuple_fast: Vec::new(),
            tuple_check: Vec::new(),
            tuple_copaque: Vec::new(),
            vfast: vec![0; ks.kernels.len().div_ceil(64)],
            kernel_fast: Vec::new(),
            ktout: Vec::new(),
            kpure_mask: Vec::new(),
            tmasks: Vec::new(),
            touts8: Vec::new(),
            touts2: Vec::new(),
            kpure: Vec::new(),
            vfast_used: false,
            kernel_phase: Vec::new(),
            touts: Vec::new(),
            passes: Vec::new(),
        },
    };
    // `VerifyPass` always runs last: the interpreter relies on its checks (net indices and table
    // selectors in range) for unchecked indexing.
    for p in passes
        .iter()
        .map(Box::as_ref)
        .chain([&VerifyPass as &dyn Pass])
    {
        p.run(&mut cx)
            .map_err(|e| format!("pass {}: {e}", p.name()))?;
        cx.program.passes.push(p.name());
    }
    Ok(cx.program)
}

/// Nets read by an item.
fn item_inputs<'a>(flat: &'a FlatNetlist, luts: &'a [LutBuild], it: Item) -> &'a [u32] {
    match it {
        Item::Node(n) => node_inputs(flat, n),
        Item::Lut(l) => &luts[l as usize].support,
    }
}

/// Nets written by an item.
fn item_outputs<'a>(flat: &'a FlatNetlist, luts: &'a [LutBuild], it: Item) -> &'a [u32] {
    match it {
        Item::Node(Node::Gate(g)) => std::slice::from_ref(&flat.gates[g as usize].output),
        Item::Node(Node::Resolver(r)) => std::slice::from_ref(&flat.resolvers[r as usize].output),
        Item::Lut(l) => &luts[l as usize].outputs,
    }
}

fn node_inputs(flat: &FlatNetlist, node: Node) -> &[u32] {
    match node {
        Node::Gate(g) => &flat.gates[g as usize].inputs,
        Node::Resolver(r) => &flat.resolvers[r as usize].drivers,
    }
}

fn node_output(flat: &FlatNetlist, node: Node) -> u32 {
    match node {
        Node::Gate(g) => flat.gates[g as usize].output,
        Node::Resolver(r) => flat.resolvers[r as usize].output,
    }
}

/// Chooses [`CompileCtx::order`] ([`CompileOptions::order`]).
pub struct OrderPass;

impl Pass for OrderPass {
    fn name(&self) -> &'static str {
        "order"
    }

    fn run(&self, cx: &mut CompileCtx<'_>) -> Result<(), String> {
        let comb = &cx.ks.comb_order;
        let nodes = match cx.opts.order {
            OrderStrategy::KernelSet => comb.clone(),
            OrderStrategy::ConeDfs => cone_dfs_order(cx.flat, comb),
        };
        cx.order = nodes.into_iter().map(Item::Node).collect();
        Ok(())
    }
}

/// Replaces small-support, many-gate cones by LUTs ([`crate::lut`]).
pub struct LutPass;

impl Pass for LutPass {
    fn name(&self) -> &'static str {
        "lut"
    }

    fn run(&self, cx: &mut CompileCtx<'_>) -> Result<(), String> {
        if !cx.luts.is_empty() {
            return Err("LUTs already extracted".into());
        }
        let nodes: Vec<Node> = cx
            .order
            .iter()
            .map(|it| match *it {
                Item::Node(n) => Ok(n),
                Item::Lut(_) => Err("unexpected LUT item".to_string()),
            })
            .collect::<Result<_, _>>()?;
        let params = LutParams {
            max_bits: cx.opts.lut_max_bits,
            min_gates: cx.opts.lut_min_gates,
            max_total_entries: cx.opts.lut_max_entries,
            merge_slack: cx.opts.lut_merge_slack,
        };
        let (luts, order) = extract_luts(cx.flat, cx.ks, &nodes, &params);
        cx.luts = luts;
        cx.order = order;
        Ok(())
    }
}

/// Post-order DFS over combinational fan-in, rooted at every sink (a node no combinational node
/// reads) in `comb` order, then at any node left over. The result is topological.
fn cone_dfs_order(flat: &FlatNetlist, comb: &[Node]) -> Vec<Node> {
    let n = flat.num_nets as usize;
    let mut drv = vec![NONE; n];
    for (i, &node) in comb.iter().enumerate() {
        drv[node_output(flat, node) as usize] = i as u32;
    }
    let mut has_succ = vec![false; comb.len()];
    for &node in comb {
        for &net in node_inputs(flat, node) {
            let d = drv[net as usize];
            if d != NONE {
                has_succ[d as usize] = true;
            }
        }
    }
    let mut state = vec![0u8; comb.len()]; // 0 new, 1 on stack, 2 done
    let mut order = Vec::with_capacity(comb.len());
    let mut stack: Vec<(u32, usize)> = Vec::new();
    let roots = (0..comb.len())
        .filter(|&i| !has_succ[i])
        .chain(0..comb.len())
        .collect::<Vec<_>>();
    for root in roots {
        if state[root] != 0 {
            continue;
        }
        state[root] = 1;
        stack.push((root as u32, 0));
        while let Some(top) = stack.last_mut() {
            let v = top.0;
            let ins = node_inputs(flat, comb[v as usize]);
            let mut child = None;
            while top.1 < ins.len() {
                let d = drv[ins[top.1] as usize];
                top.1 += 1;
                if d != NONE && state[d as usize] == 0 {
                    child = Some(d);
                    break;
                }
            }
            match child {
                Some(d) => {
                    state[d as usize] = 1;
                    stack.push((d, 0));
                }
                None => {
                    state[v as usize] = 2;
                    order.push(comb[v as usize]);
                    stack.pop();
                }
            }
        }
    }
    order
}

/// Cuts the order into partitions of about [`CompileOptions::partition_ops`] nodes.
pub struct PartitionPass;

impl Pass for PartitionPass {
    fn name(&self) -> &'static str {
        "partition"
    }

    fn run(&self, cx: &mut CompileCtx<'_>) -> Result<(), String> {
        let size = cx.opts.partition_ops.max(1);
        let lut_w = cx.opts.lut_weight.max(1);
        let len = cx.order.len();
        cx.bounds = vec![0];
        let mut w = 0usize;
        for (i, it) in cx.order.iter().enumerate() {
            // Saturating: any `lut_weight` is valid (debug builds would panic on overflow).
            w = w.saturating_add(match it {
                Item::Node(_) => 1,
                Item::Lut(_) => lut_w,
            });
            if w >= size && i + 1 < len {
                cx.bounds.push(i + 1);
                w = 0;
            }
        }
        cx.bounds.push(len);
        if len == 0 {
            cx.bounds = vec![0];
        }
        Ok(())
    }
}

/// Reorders the nodes inside each partition by logic level (longest combinational path from a
/// source / kernel net), keeping the partition contents. Still topological (levels strictly
/// increase along edges), but consecutive ops are mostly independent, which lets the CPU overlap
/// them instead of waiting on a store→load chain per op.
pub struct LevelizePass;

impl Pass for LevelizePass {
    fn name(&self) -> &'static str {
        "levelize"
    }

    fn run(&self, cx: &mut CompileCtx<'_>) -> Result<(), String> {
        let flat = cx.flat;
        let luts = &cx.luts;
        let mut level = vec![0u32; flat.num_nets as usize];
        let mut node_level = Vec::with_capacity(cx.order.len());
        for &it in &cx.order {
            let l = item_inputs(flat, luts, it)
                .iter()
                .map(|&n| level[n as usize] + 1)
                .max()
                .unwrap_or(1);
            for &o in item_outputs(flat, luts, it) {
                level[o as usize] = l;
            }
            node_level.push(l);
        }
        for w in cx.bounds.windows(2) {
            let mut keyed: Vec<(u32, Item)> =
                (w[0]..w[1]).map(|i| (node_level[i], cx.order[i])).collect();
            keyed.sort_by_key(|&(l, _)| l);
            for (i, (_, node)) in keyed.into_iter().enumerate() {
                cx.order[w[0] + i] = node;
            }
        }
        Ok(())
    }
}

/// Emits one op per combinational node, partition by partition.
pub struct LowerPass;

impl Pass for LowerPass {
    fn name(&self) -> &'static str {
        "lower"
    }

    fn run(&self, cx: &mut CompileCtx<'_>) -> Result<(), String> {
        let flat = cx.flat;
        let prog = &mut cx.program;
        for (pi, w) in cx.bounds.windows(2).enumerate() {
            let start = prog.ops.len() as u32;
            for &item in &cx.order[w[0]..w[1]] {
                let node = match item {
                    Item::Node(n) => n,
                    Item::Lut(l) => {
                        let lut = &cx.luts[l as usize];
                        let id = prog.luts.len() as u32;
                        let support = prog.lut_nets.len() as u32;
                        prog.lut_nets.extend_from_slice(&lut.support);
                        let outputs = prog.lut_nets.len() as u32;
                        prog.lut_nets.extend_from_slice(&lut.outputs);
                        let table = prog.lut_entries.len() as u32;
                        prog.lut_entries.extend_from_slice(&lut.entries);
                        let cold_start = prog.cold_ops.len() as u32;
                        for &cn in &lut.cone {
                            let Node::Gate(g) = cn else {
                                return Err("resolver in a LUT cone".into());
                            };
                            let gate = &flat.gates[g as usize];
                            let op = lower_gate(&mut prog.args, gate)?;
                            prog.cold_ops.push(op);
                            prog.lazy_owner[gate.output as usize] = 2 * id;
                        }
                        for &o in &lut.outputs {
                            prog.lazy_owner[o as usize] = NONE;
                            cx.driver_part[o as usize] = pi as u32;
                        }
                        prog.lut_gates += lut.cone.len() as u32;
                        prog.luts.push(LutDesc {
                            support,
                            num_support: lut.support.len() as u32,
                            outputs,
                            num_outputs: lut.outputs.len() as u32,
                            table,
                            cold: Partition {
                                start: cold_start,
                                end: prog.cold_ops.len() as u32,
                            },
                        });
                        prog.ops.push(Op::Lut { lut: id });
                        continue;
                    }
                };
                let op = match node {
                    Node::Gate(g) => lower_gate(&mut prog.args, &flat.gates[g as usize])?,
                    Node::Resolver(r) => {
                        let res = &flat.resolvers[r as usize];
                        let args = prog.args.len() as u32;
                        prog.args.extend_from_slice(&res.drivers);
                        prog.resolvers.push(ResolverDesc {
                            args,
                            n: res.drivers.len() as u32,
                            termination: res.termination,
                            flat_index: r,
                        });
                        Op::Resolve {
                            notify: false,
                            out: res.output,
                            res: prog.resolvers.len() as u32 - 1,
                        }
                    }
                };
                cx.driver_part[op.output() as usize] = pi as u32;
                prog.ops.push(op);
            }
            prog.partitions.push(Partition {
                start,
                end: prog.ops.len() as u32,
            });
        }
        Ok(())
    }
}

/// Lower one gate (exactly `panda_netlist::eval::eval_gate`).
fn lower_gate(args: &mut Vec<u32>, g: &panda_netlist::flat::FlatGate) -> Result<Op, String> {
    let ins = &g.inputs;
    let out = g.output;
    let two = |t: u8, a: u32, b: u32| Op::Gate2 {
        t,
        notify: false,
        out,
        a,
        b,
    };
    let (fold, invert) = match g.op {
        FlatOp::Not | FlatOp::Buf => {
            let Some(&a) = ins.first() else {
                return Err(format!("{:?} gate without input", g.op));
            };
            let t = if g.op == FlatOp::Not {
                table::NOT
            } else {
                table::BUF
            };
            return Ok(two(t, a, a));
        }
        FlatOp::Tri => {
            let [d, en] = ins[..] else {
                return Err("Tri gate needs 2 inputs".into());
            };
            return Ok(two(table::TRI, d, en));
        }
        FlatOp::And => (table::AND, false),
        FlatOp::Nand => (table::AND, true),
        FlatOp::Or => (table::OR, false),
        FlatOp::Nor => (table::OR, true),
        FlatOp::Xor => (table::XOR, false),
        FlatOp::Xnor => (table::XOR, true),
    };
    Ok(match ins.len() {
        0 => return Err(format!("{:?} gate without inputs", g.op)),
        // The fold of one input with the identity is its normalization.
        1 => two(if invert { table::NOT } else { table::BUF }, ins[0], ins[0]),
        2 => {
            let t = match (fold, invert) {
                (table::AND, false) => table::AND,
                (table::AND, true) => table::NAND,
                (table::OR, false) => table::OR,
                (table::OR, true) => table::NOR,
                (table::XOR, false) => table::XOR,
                _ => table::XNOR,
            };
            two(t, ins[0], ins[1])
        }
        n => {
            let off = args.len() as u32;
            args.extend_from_slice(ins);
            Op::GateN {
                fold,
                invert,
                notify: false,
                out,
                args: off,
                n: n as u32,
            }
        }
    })
}

/// Builds the static change propagation lists and the `notify` flags.
pub struct NotifyPass;

impl Pass for NotifyPass {
    fn name(&self) -> &'static str {
        "notify"
    }

    fn run(&self, cx: &mut CompileCtx<'_>) -> Result<(), String> {
        let n = cx.flat.num_nets as usize;
        let prog = &mut cx.program;
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        for (pi, part) in prog.partitions.iter().enumerate() {
            for op in &prog.ops[part.start as usize..part.end as usize] {
                let mut add = |net: u32| {
                    if cx.driver_part[net as usize] != pi as u32 {
                        pairs.push((net, pi as u32));
                    }
                };
                match *op {
                    Op::Gate2 { a, b, .. } => {
                        add(a);
                        add(b);
                    }
                    Op::GateN { args, n, .. } => {
                        for &x in &prog.args[args as usize..(args + n) as usize] {
                            add(x);
                        }
                    }
                    Op::Resolve { res, .. } => {
                        let r = prog.resolvers[res as usize];
                        for &x in &prog.args[r.args as usize..(r.args + r.n) as usize] {
                            add(x);
                        }
                    }
                    Op::Lut { lut } => {
                        let l = prog.luts[lut as usize];
                        for &x in prog.lut_support(&l) {
                            add(x);
                        }
                    }
                }
            }
        }
        prog.net_parts = NetCsr::from_pairs(n, &mut pairs);
        let mut kpairs: Vec<(u32, u32)> = Vec::new();
        for k in &cx.ks.kernels {
            for &net in &k.inputs {
                kpairs.push((net, k.id));
            }
        }
        prog.net_kernels = NetCsr::from_pairs(n, &mut kpairs);
        if cx.ks.kernels.len() > KIN_KERNEL_MASK as usize {
            return Err("too many kernels".into());
        }
        let mut kin: Vec<(u32, u32)> = Vec::new();
        for k in &cx.ks.kernels {
            for (i, &net) in k.inputs.iter().enumerate() {
                kin.push((net, k.id | (i.min(32) as u32) << KIN_POS_SHIFT | KIN_DATA));
            }
        }
        prog.net_kin = NetCsr::from_pairs(n, &mut kin);
        let has_readers =
            |out: u32| !prog.net_parts.row(out).is_empty() || !prog.net_kernels.row(out).is_empty();
        for op in prog.ops.iter_mut().chain(prog.cold_ops.iter_mut()) {
            if let Op::Lut { .. } = op {
                continue;
            }
            let notify = has_readers(op.output());
            op.set_notify(notify);
        }
        Ok(())
    }
}

/// One transition op per kernel.
pub struct KernelPass;

impl Pass for KernelPass {
    fn name(&self) -> &'static str {
        "kernels"
    }

    fn run(&self, cx: &mut CompileCtx<'_>) -> Result<(), String> {
        let prog = &mut cx.program;
        if prog.net_parts.start.len() != prog.num_nets as usize + 1 {
            return Err("requires the notify pass".into());
        }
        let mut table_ids: HashMap<*const KernelTable, u32> = HashMap::new();
        let mut last_in = 0u32;
        for (i, k) in cx.ks.kernels.iter().enumerate() {
            if k.id as usize != i {
                return Err(format!("kernel {i} has id {}", k.id));
            }
            let kind = match &cx.tables[i] {
                Some(t) if t.num_inputs == k.inputs.len() && k.inputs.len() <= 31 => {
                    let key = Arc::as_ptr(t);
                    let id = *table_ids.entry(key).or_insert_with(|| {
                        prog.ktables.push(KTableData::new(t.clone(), k.nets.len()));
                        prog.ktables.len() as u32 - 1
                    });
                    if prog.ktables[id as usize].num_nets as usize != k.nets.len() {
                        return Err(format!("table of kernel {i} does not match its nets"));
                    }
                    KernelOpKind::KTable { table: id }
                }
                Some(_) => return Err(format!("table of kernel {i} does not match its inputs")),
                None => KernelOpKind::KSim,
            };
            let inputs = prog.kernel_nets.len() as u32;
            prog.kernel_nets.extend_from_slice(&k.inputs);
            let nets = prog.kernel_nets.len() as u32;
            prog.kernel_nets.extend_from_slice(&k.nets);
            // Requires NotifyPass (reader lists) to have run.
            let outs = prog.kernel_outs.len() as u32;
            for (j, &net) in k.nets.iter().enumerate() {
                if !prog.net_parts.row(net).is_empty() || !prog.net_kernels.row(net).is_empty() {
                    prog.kernel_outs.push(j as u32);
                    prog.kernel_out_nets.push(net);
                }
            }
            let num_outs = prog.kernel_outs.len() as u32 - outs;
            prog.kernel_ops.push(KernelOp {
                kernel: k.id,
                kind,
                inputs,
                num_inputs: k.inputs.len() as u32,
                nets,
                num_nets: k.nets.len() as u32,
                outs,
                num_outs,
                last_in,
            });
            last_in += k.inputs.len() as u32;
        }
        prog.kernel_inputs_total = last_in;
        Ok(())
    }
}

/// Builds the interpreter's combined per-net change-propagation records
/// ([`Program::notify_rec`]); runs after every
/// pass that edits the reader lists. Readers without a kernel class are reduced to pending-mark
/// word masks (they never use the input index or the transparent-skip flags).
pub struct LayoutPass;

impl Pass for LayoutPass {
    fn name(&self) -> &'static str {
        "layout"
    }

    fn run(&self, cx: &mut CompileCtx<'_>) -> Result<(), String> {
        let prog = &mut cx.program;
        let n = prog.num_nets as usize;
        prog.notify_off = vec![NONE; n];
        prog.notify_rec = Vec::new();
        prog.net_log = vec![NONE; n];
        prog.num_log = 0;
        let mut parts: Vec<(u64, u64)> = Vec::new();
        // Per kernel word: (pending mask, data-input reader mask).
        let mut groups: Vec<(u64, u64, u64)> = Vec::new();
        let mut kin: Vec<u32> = Vec::new();
        for net in 0..n as u32 {
            parts.clear();
            for &p in prog.net_parts.row(net) {
                let w = u64::from(p / 64);
                match parts.iter_mut().find(|(pw, _)| *pw == w) {
                    Some((_, m)) => *m |= 1 << (p % 64),
                    None => parts.push((w, 1 << (p % 64))),
                }
            }
            groups.clear();
            kin.clear();
            fn group(groups: &mut Vec<(u64, u64, u64)>, w: u64) -> &mut (u64, u64, u64) {
                let i = match groups.iter().position(|g| g.0 == w) {
                    Some(i) => i,
                    None => {
                        groups.push((w, 0, 0));
                        groups.len() - 1
                    }
                };
                &mut groups[i]
            }
            for &(w, m) in prog.net_kpend.row(net) {
                group(&mut groups, u64::from(w)).1 |= m;
            }
            for &e in prog.net_kin.row(net) {
                let k = e & KIN_KERNEL_MASK;
                let g = group(&mut groups, u64::from(k / 64));
                g.1 |= 1 << (k % 64);
                // Kernels without a class never take a fast path: they only need the pending
                // mark.
                if prog.kernel_class[k as usize] == NONE {
                    continue;
                }
                if e & KIN_DATA != 0 {
                    g.2 |= 1 << (k % 64);
                }
                kin.push(e & !KIN_DATA);
            }
            if parts.is_empty() && groups.is_empty() {
                continue;
            }
            let log = if groups.is_empty() {
                NONE
            } else {
                prog.num_log += 1;
                prog.num_log - 1
            };
            prog.net_log[net as usize] = log;
            if prog.notify_rec.len() >= 1 << 30 {
                return Err("too many change-propagation records".into());
            }
            prog.notify_off[net as usize] = prog.notify_rec.len() as u32;
            let rec = &mut prog.notify_rec;
            // Compact header when the counts fit, else the escape plus full-word counts (a net
            // may be read by any number of partitions / kernels, e.g. a clock read by every
            // latch of a large register file).
            let (np, ng, nkin) = (parts.len(), groups.len(), kin.len());
            if np < REC_ESCAPE as usize && ng <= 0xFF && nkin <= 0xFFFF {
                rec.extend([np as u32 | (ng as u32) << 8 | (nkin as u32) << 16, net, log]);
            } else {
                let count = |c: usize| {
                    u32::try_from(c).map_err(|_| format!("net {net} has too many readers"))
                };
                rec.extend([REC_ESCAPE, net, log, count(np)?, count(ng)?, count(nkin)?]);
            }
            for &(w, m) in &parts {
                rec.extend([w as u32, m as u32, (m >> 32) as u32]);
            }
            for &(w, pend, data) in &groups {
                rec.extend([
                    w as u32,
                    pend as u32,
                    (pend >> 32) as u32,
                    data as u32,
                    (data >> 32) as u32,
                ]);
            }
            rec.extend_from_slice(&kin);
        }
        let off = |net: u32| prog.notify_off[net as usize];
        let op_rec: Vec<u32> = prog
            .ops
            .iter()
            .map(|op| {
                if op.notifies() && !matches!(op, Op::Lut { .. }) {
                    off(op.output())
                } else {
                    NONE
                }
            })
            .collect();
        let cold_rec: Vec<u32> = prog
            .cold_ops
            .iter()
            .map(|op| {
                if op.notifies() {
                    off(op.output())
                } else {
                    NONE
                }
            })
            .collect();
        let lut_rec: Vec<u32> = prog.lut_nets.iter().map(|&n| off(n)).collect();
        let kernel_out_rec: Vec<u32> = prog.kernel_out_nets.iter().map(|&n| off(n)).collect();
        prog.op_rec = op_rec;
        prog.cold_rec = cold_rec;
        prog.lut_rec = lut_rec;
        prog.kernel_out_rec = kernel_out_rec;
        prog.kout1 = prog
            .kernel_ops
            .iter()
            .map(|k| {
                if k.num_outs == 1 {
                    (
                        prog.kernel_out_nets[k.outs as usize],
                        prog.kernel_out_rec[k.outs as usize],
                    )
                } else {
                    (NONE, NONE)
                }
            })
            .collect();
        prog.tracked = vec![0; prog.kernel_ops.len().div_ceil(64)];
        for &e in &prog.net_kin.items {
            if e & KIN_TRACK != 0 {
                let k = e & KIN_KERNEL_MASK;
                prog.tracked[k as usize / 64] |= 1 << (k % 64);
            }
        }
        prog.tracked_list = (0..prog.kernel_ops.len() as u32)
            .filter(|&k| prog.tracked[k as usize / 64] & (1 << (k % 64)) != 0)
            .collect();
        let low = crate::fuse::lower(prog, cx.opts.fuse_inputs)?;
        for op in &low.fcold {
            prog.lazy_owner[op.output() as usize] = crate::ir::LAZY_FUSED;
        }
        prog.hops = low.hops;
        prog.hrec = low.hrec;
        prog.hpart = low.hpart;
        prog.ftables = low.ftables;
        prog.fcold = low.fcold;
        Ok(())
    }
}

/// Bounds of the interpreter's per-tuple kernel tables (indexed unchecked).
fn verify_kfast(prog: &Program) -> Result<(), String> {
    let nk = prog.kernel_ops.len();
    let nt = prog.num_tuples as usize;
    let words = nk.div_ceil(64);
    if prog.ktout.len() != nk * nt
        || prog.kpure.len() != nk * nt
        || prog.tmasks.len() != nt * words
        || prog.tracked.len() != words
        || (!prog.touts8.is_empty() && prog.touts8.len() != prog.touts.len())
        || (!prog.touts2.is_empty() && prog.touts2.len() != prog.touts.len().div_ceil(32))
    {
        return Err("per-tuple kernel tables have the wrong size".into());
    }
    for t in 0..nt {
        for (k, kop) in prog.kernel_ops.iter().enumerate() {
            let (tout, pure) = (prog.ktout[t * nk + k], prog.kpure[t * nk + k]);
            let size = 1u64 << kop.num_inputs.min(32);
            if u64::from(pure) >= size
                || u64::from(prog.kernel_kin_mask[k]) >= size
                || (tout != NONE && u64::from(tout) + size > prog.touts.len() as u64)
            {
                return Err(format!("kernel {k} tuple {t}: output table out of range"));
            }
        }
    }
    Ok(())
}

/// First word of a change-propagation record ([`Program::notify_rec`]) whose counts do not fit
/// the compact header: the counts follow the log slot as three full words.
pub(crate) const REC_ESCAPE: u32 = 0xFF;

/// Bounds of the change-propagation records (the interpreter indexes them unchecked).
fn verify_notify(prog: &Program) -> Result<(), String> {
    let n = prog.num_nets as usize;
    if prog.notify_off.len() != n || prog.net_log.len() != n {
        return Err("notify records missing (layout pass not run?)".into());
    }
    let pwords = prog.partitions.len().div_ceil(64) as u32;
    let kwords = prog.kernel_ops.len().div_ceil(64) as u32;
    for (net, &off) in prog.notify_off.iter().enumerate() {
        if off == NONE {
            continue;
        }
        let r = prog
            .notify_rec
            .get(off as usize..)
            .ok_or_else(|| format!("net {net}: record offset out of range"))?;
        if r.len() < 3 || r[1] as usize != net || off >= 1 << 30 {
            return Err(format!("net {net}: record header broken"));
        }
        let h = r[0];
        let (np, nk, nkin, header) = if h & 0xFF != REC_ESCAPE & 0xFF {
            (
                (h & 0xFF) as usize,
                ((h >> 8) & 0xFF) as usize,
                (h >> 16) as usize,
                3,
            )
        } else if h == REC_ESCAPE && r.len() >= 6 {
            (r[3] as usize, r[4] as usize, r[5] as usize, 6)
        } else {
            return Err(format!("net {net}: record header broken"));
        };
        let len = np
            .checked_mul(3)
            .and_then(|a| nk.checked_mul(5).and_then(|b| a.checked_add(b)))
            .and_then(|a| a.checked_add(nkin))
            .and_then(|a| a.checked_add(header))
            .ok_or_else(|| format!("net {net}: record size overflows"))?;
        if r.len() < len {
            return Err(format!("net {net}: record truncated"));
        }
        if r[2] != NONE && r[2] >= prog.num_log {
            return Err(format!("net {net}: log slot out of range"));
        }
        if nk == 0 && nkin != 0 {
            return Err(format!(
                "net {net}: input-index entries without kernel groups"
            ));
        }
        let (p0, k0) = (header, header + 3 * np);
        let nparts = prog.partitions.len() as u64;
        let nkern = prog.kernel_ops.len() as u64;
        // Bits beyond the last partition / kernel would make the sweep or the round loop name
        // a nonexistent one.
        let in_range = |w: u32, lo: u32, hi: u32, count: u64| {
            let m = u64::from(lo) | u64::from(hi) << 32;
            m == 0 || u64::from(w) * 64 + 63 - u64::from(m.leading_zeros()) < count
        };
        for pm in r[p0..k0].chunks_exact(3) {
            if pm[0] >= pwords || !in_range(pm[0], pm[1], pm[2], nparts) {
                return Err(format!("net {net}: partition word out of range"));
            }
        }
        for g in r[k0..k0 + 5 * nk].chunks_exact(5) {
            if g[0] >= kwords
                || !in_range(g[0], g[1], g[2], nkern)
                || !in_range(g[0], g[3], g[4], nkern)
            {
                return Err(format!("net {net}: kernel group out of range"));
            }
        }
        for &e in &r[k0 + 5 * nk..len] {
            let k = (e & crate::ir::KIN_KERNEL_MASK) as usize;
            let Some(kop) = prog.kernel_ops.get(k) else {
                return Err(format!("net {net}: kernel entry out of range"));
            };
            // The interpreter sets bit `pos` of the kernel's input index (position 32: no bit),
            // and indexes the transparent output tables with it: `pos` must name an input.
            let pos = (e >> KIN_POS_SHIFT) & crate::ir::KIN_POS_MASK;
            let ok = if pos == 32 {
                kop.num_inputs > 32
            } else {
                pos < 32 && pos < kop.num_inputs
            };
            if !ok {
                return Err(format!(
                    "net {net}: kernel input position {pos} out of range"
                ));
            }
        }
    }
    // The record arrays name exactly the records of the nets they stand for.
    let expect = |net: u32, notify: bool| {
        if notify {
            prog.notify_off[net as usize]
        } else {
            NONE
        }
    };
    let lut_out = |l: &crate::ir::LutDesc, i: usize| -> bool {
        i >= l.outputs as usize && i < (l.outputs + l.num_outputs) as usize
    };
    let ok = prog.op_rec.len() == prog.ops.len()
        && prog.cold_rec.len() == prog.cold_ops.len()
        && prog.lut_rec.len() == prog.lut_nets.len()
        && prog.kernel_out_rec.len() == prog.kernel_out_nets.len()
        && prog.ops.iter().zip(&prog.op_rec).all(|(op, &r)| match op {
            Op::Lut { .. } => r == NONE,
            _ => r == expect(op.output(), op.notifies()),
        })
        && prog
            .cold_ops
            .iter()
            .zip(&prog.cold_rec)
            .all(|(op, &r)| r == expect(op.output(), op.notifies()))
        && prog
            .lut_nets
            .iter()
            .zip(&prog.lut_rec)
            .enumerate()
            .all(|(i, (&net, &r))| {
                r == prog.notify_off[net as usize] || !prog.luts.iter().any(|l| lut_out(l, i))
            })
        && prog
            .kernel_out_nets
            .iter()
            .zip(&prog.kernel_out_rec)
            .all(|(&net, &r)| r == prog.notify_off[net as usize]);
    let ok = ok
        && prog.kout1.len() == prog.kernel_ops.len()
        && prog
            .kernel_ops
            .iter()
            .zip(&prog.kout1)
            .all(|(k, &(net, rec))| {
                if k.num_outs == 1 {
                    net == prog.kernel_out_nets[k.outs as usize]
                        && rec == prog.kernel_out_rec[k.outs as usize]
                } else {
                    net == NONE && rec == NONE
                }
            });
    if !ok {
        return Err("record offset arrays disagree with the records".into());
    }
    // Every notifying op has a record (it has readers).
    if prog.ops.iter().chain(&prog.cold_ops).any(|op| {
        !matches!(op, Op::Lut { .. })
            && op.notifies()
            && prog.notify_off[op.output() as usize] == NONE
    }) {
        return Err("notifying op without a record".into());
    }
    Ok(())
}

/// Structural checks of the lowered program.
pub struct VerifyPass;

impl Pass for VerifyPass {
    fn name(&self) -> &'static str {
        "verify"
    }

    fn run(&self, cx: &mut CompileCtx<'_>) -> Result<(), String> {
        let prog = &cx.program;
        let n = prog.num_nets as usize;
        let n_luts = prog
            .ops
            .iter()
            .filter(|o| matches!(o, Op::Lut { .. }))
            .count();
        if n_luts != prog.luts.len() {
            return Err(format!("{n_luts} LUT ops for {} LUTs", prog.luts.len()));
        }
        if prog.ops.len() - n_luts + prog.cold_ops.len() != cx.ks.comb_order.len() {
            return Err(format!(
                "{} ops + {} LUT gates for {} combinational nodes",
                prog.ops.len() - n_luts,
                prog.cold_ops.len(),
                cx.ks.comb_order.len()
            ));
        }
        // Position of the op driving each net (a LUT's internal nets: the LUT op).
        let mut pos = vec![NONE; n];
        fn set_pos(pos: &mut [u32], o: u32, i: usize) -> Result<(), String> {
            let o = o as usize;
            if o >= pos.len() || pos[o] != NONE {
                return Err(format!("net {o} driven by several ops or out of range"));
            }
            pos[o] = i as u32;
            Ok(())
        }
        for (i, op) in prog.ops.iter().enumerate() {
            match *op {
                Op::Lut { lut } => {
                    let l = prog.luts[lut as usize];
                    if l.num_support > 16
                        || l.num_outputs > 32
                        || l.num_outputs == 0
                        || (l.table as usize + (1usize << l.num_support)) > prog.lut_entries.len()
                        || l.cold.end as usize > prog.cold_ops.len()
                    {
                        return Err(format!("LUT {lut}: descriptor out of range"));
                    }
                    for c in &prog.cold_ops[l.cold.start as usize..l.cold.end as usize] {
                        if matches!(c, Op::Lut { .. } | Op::Resolve { .. }) {
                            return Err(format!("LUT {lut}: cold code holds a {c:?}"));
                        }
                        set_pos(&mut pos, c.output(), i)?;
                    }
                    for &o in prog.lut_outputs(&l) {
                        if o as usize >= n || pos[o as usize] != i as u32 {
                            return Err(format!("LUT {lut}: output {o} not computed by its cone"));
                        }
                    }
                }
                _ => set_pos(&mut pos, op.output(), i)?,
            }
        }
        let check = |i: usize, net: u32| -> Result<(), String> {
            if net as usize >= n {
                return Err(format!("op {i} reads net {net} out of range"));
            }
            let d = pos[net as usize];
            if d != NONE && d as usize >= i {
                return Err(format!("op {i} reads net {net} before it is computed"));
            }
            Ok(())
        };
        let check_op = |i: usize, op: &Op| -> Result<(), String> {
            match *op {
                Op::Gate2 { t, a, b, .. } => {
                    if t as usize >= table::COUNT {
                        return Err(format!("op {i}: table selector {t} out of range"));
                    }
                    check(i, a)?;
                    check(i, b)?;
                }
                Op::GateN { fold, args, n, .. } => {
                    if ![table::AND, table::OR, table::XOR].contains(&fold) {
                        return Err(format!("op {i}: fold table {fold} is not AND/OR/XOR"));
                    }
                    if (args + n) as usize > prog.args.len() {
                        return Err(format!("op {i}: operand range out of bounds"));
                    }
                    for &x in &prog.args[args as usize..(args + n) as usize] {
                        check(i, x)?;
                    }
                }
                Op::Resolve { res, .. } => {
                    let r = prog.resolvers[res as usize];
                    for &x in &prog.args[r.args as usize..(r.args + r.n) as usize] {
                        check(i, x)?;
                    }
                }
                Op::Lut { lut } => {
                    let l = prog.luts[lut as usize];
                    for &x in prog.lut_support(&l) {
                        check(i, x)?;
                    }
                }
            }
            Ok(())
        };
        for (i, op) in prog.ops.iter().enumerate() {
            check_op(i, op)?;
            if let Op::Lut { lut } = *op {
                // The cold code reads the support and earlier cone nets only.
                let l = prog.luts[lut as usize];
                let cold = &prog.cold_ops[l.cold.start as usize..l.cold.end as usize];
                let mut seen: Vec<u32> = Vec::new();
                for c in cold {
                    let ins: Vec<u32> = match *c {
                        Op::Gate2 { a, b, .. } => vec![a, b],
                        Op::GateN { args, n, .. } => {
                            prog.args[args as usize..(args + n) as usize].to_vec()
                        }
                        _ => unreachable!(),
                    };
                    for x in ins {
                        if x as usize >= n {
                            return Err(format!("LUT {lut}: cold op reads net {x} out of range"));
                        }
                        let d = pos[x as usize];
                        if d == i as u32 && !seen.contains(&x) {
                            return Err(format!("LUT {lut}: cold op reads net {x} too early"));
                        }
                        if d != NONE && d != i as u32 && d as usize >= i {
                            return Err(format!(
                                "LUT {lut}: cold op reads net {x} before it is computed"
                            ));
                        }
                    }
                    seen.push(c.output());
                }
            }
        }
        for k in &cx.ks.kernels {
            for &net in &k.nets {
                if pos[net as usize] != NONE {
                    return Err(format!("kernel net {net} is also driven by an op"));
                }
            }
        }
        verify_notify(prog)?;
        verify_kfast(prog)?;
        // The interpreter encoding is exactly the deterministic lowering of the ops (so it
        // computes the same functions; nets, records and table offsets are in range).
        let low = crate::fuse::lower(prog, cx.opts.fuse_inputs)?;
        if low.hops != prog.hops
            || low.hrec != prog.hrec
            || low.hpart != prog.hpart
            || low.ftables != prog.ftables
            || low.fcold != prog.fcold
        {
            return Err("interpreter encoding differs from the ops".into());
        }
        for op in &prog.fcold {
            if prog.lazy_owner[op.output() as usize] != crate::ir::LAZY_FUSED {
                return Err("fused-internal net not marked lazy".into());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use panda_compile::{KernelOptions, compile_kernels};
    use panda_netlist::Dir;
    use panda_netlist::flat::{FlatGate, FlatInstance, FlatPin, GateGroup, Source, SourceKind};

    /// Two NAND D latches sharing their data and enable inputs (both inputs read by kernels
    /// with a class).
    fn latches() -> FlatNetlist {
        let mut f = FlatNetlist::default();
        f.instances.push(FlatInstance {
            path: String::new(),
            def: "top".into(),
            parent: None,
            children: Vec::new(),
            pins: Vec::new(),
        });
        let net = |f: &mut FlatNetlist, name: &str| {
            let n = f.num_nets;
            f.num_nets += 1;
            f.net_names.push(name.into());
            f.name_index.insert(name.into(), n);
            n
        };
        let d = net(&mut f, "d");
        let en = net(&mut f, "en");
        for (pin, n) in [d, en].into_iter().enumerate() {
            f.sources.push(Source {
                kind: SourceKind::TopInput {
                    pin: pin as u32,
                    bit: 0,
                },
                output: n,
            });
            f.top_inputs.push(FlatPin {
                name: f.net_names[n as usize].clone(),
                dir: Dir::In,
                nets: vec![n],
            });
        }
        for l in 0..2 {
            let dn = net(&mut f, &format!("dn{l}"));
            let s = net(&mut f, &format!("s{l}"));
            let r = net(&mut f, &format!("r{l}"));
            let q = net(&mut f, &format!("q{l}"));
            let qn = net(&mut f, &format!("qn{l}"));
            for (op, ins, out) in [
                (FlatOp::Not, vec![d], dn),
                (FlatOp::Nand, vec![d, en], s),
                (FlatOp::Nand, vec![dn, en], r),
                (FlatOp::Nand, vec![s, qn], q),
                (FlatOp::Nand, vec![r, q], qn),
            ] {
                let gi = f.gates.len() as u32;
                f.gates.push(FlatGate {
                    op,
                    inputs: ins,
                    output: out,
                    owner: 0,
                    group: gi,
                    bit: 0,
                });
                f.groups.push(GateGroup {
                    owner: 0,
                    path: format!("g{gi}"),
                    gates: vec![gi],
                });
            }
            f.top_outputs.push(FlatPin {
                name: format!("q{l}"),
                dir: Dir::Out,
                nets: vec![q],
            });
        }
        f
    }

    fn program() -> Program {
        let flat = latches();
        let (ks, tables) = compile_kernels(&flat, &KernelOptions::default());
        compile(&flat, &ks, &tables, &CompileOptions::default()).expect("compiles")
    }

    /// Offset of the first kernel input-index entry in `d`'s record.
    fn kin_entry(prog: &Program) -> usize {
        let off = prog.notify_off[0] as usize;
        let h = prog.notify_rec[off];
        assert_ne!(h, REC_ESCAPE);
        let (np, nk, nkin) = (h & 0xFF, (h >> 8) & 0xFF, h >> 16);
        assert!(nkin > 0, "d is read by kernels with a class");
        off + 3 + 3 * np as usize + 5 * nk as usize
    }

    #[test]
    fn verify_notify_accepts_compiled_records() {
        verify_notify(&program()).expect("valid records");
    }

    #[test]
    fn verify_notify_rejects_input_positions_beyond_the_kernel() {
        // A position past the kernel's inputs would index the transparent output tables out of
        // range in the interpreter.
        let mut prog = program();
        let i = kin_entry(&prog);
        let e = prog.notify_rec[i];
        let k = e & KIN_KERNEL_MASK;
        let n = prog.kernel_ops[k as usize].num_inputs;
        let pos_mask = crate::ir::KIN_POS_MASK << KIN_POS_SHIFT;
        prog.notify_rec[i] = (e & !pos_mask) | n << KIN_POS_SHIFT;
        assert!(verify_notify(&prog).is_err());
        prog.notify_rec[i] = (e & !pos_mask) | 32 << KIN_POS_SHIFT;
        assert!(verify_notify(&prog).is_err());
    }

    #[test]
    fn verify_notify_rejects_masks_beyond_the_last_partition_or_kernel() {
        let prog = program();
        let off = prog.notify_off[0] as usize;
        let h = prog.notify_rec[off];
        let (np, nk) = ((h & 0xFF) as usize, ((h >> 8) & 0xFF) as usize);
        assert!(nk > 0);
        // Kernel pending mask: bit 63 of word 0 (only 2 kernels exist).
        let mut bad = prog.clone();
        bad.notify_rec[off + 3 + 3 * np + 2] |= 1 << 31;
        assert!(verify_notify(&bad).is_err());
        if np > 0 {
            let mut bad = prog.clone();
            bad.notify_rec[off + 3 + 2] |= 1 << 31;
            assert!(verify_notify(&bad).is_err());
        }
    }
}
