//! Kernel extraction (SPEC §8.3): find the small state-holding gate clusters of a flat netlist.
//!
//! Algorithm ([`extract`]):
//! 1. **Regions**: SCCs of the full gate/resolver graph. Only cyclic regions are searched.
//! 2. **Seed loops**: every simple cycle of length ≤ [`ExtractOptions::max_loop`] (enumerated per
//!    root with [`CycleSearch`], pruned by a backward BFS so the work stays local). A root whose
//!    search exceeds [`ExtractOptions::max_seed_work_per_root`] is skipped (its cycles found so far
//!    are kept), a region whose total search work exceeds
//!    [`ExtractOptions::max_seed_work_per_region`] stops being searched (cycles found so far are
//!    kept), and a region with more
//!    than [`ExtractOptions::max_cycles_per_region`] cycles is not seeded at all; the loops lost that
//!    way land in step 5 as Sim kernels (correct, just slower).
//! 3. **Merge**: cycles sharing a node form one kernel core (union-find).
//! 4. **Absorb private fan-in**: a gate/resolver outside every kernel whose readers all lie in one
//!    kernel joins it; repeated to a fixpoint. An absorption that would push the kernel's
//!    `inputs + cut` bits past [`ExtractOptions::max_bits`] is refused unless it does not increase
//!    that count (e.g. an inverter of an existing input).
//! 5. **Residual cycles**: SCCs of the graph with all kernel nodes *removed* (i.e. cycles that pass
//!    through no kernel) become additional kernels, then fan-in absorption runs again.
//! 6. **Keepers**: every keeper-terminated resolver not inside a kernel becomes a 1-node Keeper
//!    kernel (it holds state).
//! 7. **Kinds**: a kernel with `inputs + cut ≤ max_bits` is a Table kernel (characterized,
//!    [`crate::charz`]); larger ones are Sim kernels (transition computed on the fly).
//!
//! After extraction every cycle of the netlist passes through a kernel, so the remaining
//! combinational nodes are acyclic; [`KernelSet::comb_order`] is a topological order of them.

use std::collections::{HashMap, HashSet, VecDeque};

use panda_netlist::Termination;
use panda_netlist::flat::{FlatNetlist, GateIdx, InstIdx, NetIdx};

use crate::graph::{
    CycleSearch, Graph, NONE, Node, SearchOutcome, UnionFind, feedback_vertex_set, scc, topo_order,
};

/// How a kernel's transition function is evaluated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelKind {
    /// Small enough to characterize into a [`crate::charz::KernelTable`].
    Table,
    /// Transition computed on the fly by unit-delay simulation ([`crate::charz::KernelSim`]).
    Sim,
    /// A lone keeper-terminated resolver (1 state bit). Its inputs can legitimately be `Z`, which a
    /// table cannot index, so it is always simulated on the fly.
    Keeper,
}

/// One kernel: a set of gates/resolvers whose nets are the kernel state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Kernel {
    /// Index into [`KernelSet::kernels`].
    pub id: u32,
    /// Evaluation strategy.
    pub kind: KernelKind,
    /// Gates inside (including absorbed private fan-in), ascending.
    pub gates: Vec<GateIdx>,
    /// Resolvers inside, ascending.
    pub resolvers: Vec<u32>,
    /// Nets read by the kernel and driven outside it (sorted, deduplicated). Includes source-driven
    /// and undriven nets.
    pub inputs: Vec<NetIdx>,
    /// Kernel nets read outside the kernel, or that are top-level output nets (sorted).
    pub outputs: Vec<NetIdx>,
    /// All nets driven inside the kernel = the state (sorted).
    pub nets: Vec<NetIdx>,
    /// A feedback cut of the kernel (subset of [`Kernel::nets`], sorted): with these nets fixed,
    /// the kernel is acyclic. Always contains keeper-terminated resolver outputs. Complete for
    /// Table and Keeper kernels; empty for Sim kernels (not computed).
    pub cut: Vec<NetIdx>,
    /// Deepest instance containing every gate (for caching); 0 (top) for gate-less kernels.
    pub owner: InstIdx,
}

impl Kernel {
    /// `inputs + cut` bits (the characterization size measure, SPEC §8.3 step 6).
    pub fn bits(&self) -> usize {
        self.inputs.len() + self.cut.len()
    }

    /// The kernel's nodes (gates, then resolvers).
    pub fn nodes(&self) -> impl Iterator<Item = Node> + '_ {
        self.gates
            .iter()
            .map(|&g| Node::Gate(g))
            .chain(self.resolvers.iter().map(|&r| Node::Resolver(r)))
    }
}

/// The result of [`extract`].
#[derive(Clone, Debug, Default)]
pub struct KernelSet {
    /// All kernels, ordered by their smallest node.
    pub kernels: Vec<Kernel>,
    /// Kernel of every gate (per [`GateIdx`]).
    pub kernel_of_gate: Vec<Option<u32>>,
    /// Kernel of every resolver.
    pub kernel_of_resolver: Vec<Option<u32>>,
    /// Combinational gates/resolvers (not in a kernel) in topological order (kernel outputs and
    /// sources are the roots). Guaranteed acyclic.
    pub comb_order: Vec<Node>,
}

/// Kernel count per kind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KindHistogram {
    /// Number of [`KernelKind::Table`] kernels.
    pub table: usize,
    /// Number of [`KernelKind::Sim`] kernels.
    pub sim: usize,
    /// Number of [`KernelKind::Keeper`] kernels.
    pub keeper: usize,
}

impl KernelSet {
    /// Kernel containing `node`, if any.
    pub fn kernel_of(&self, node: Node) -> Option<u32> {
        match node {
            Node::Gate(g) => self.kernel_of_gate[g as usize],
            Node::Resolver(r) => self.kernel_of_resolver[r as usize],
        }
    }

    /// Number of kernels of each kind.
    pub fn histogram(&self) -> KindHistogram {
        let mut h = KindHistogram::default();
        for k in &self.kernels {
            match k.kind {
                KernelKind::Table => h.table += 1,
                KernelKind::Sim => h.sim += 1,
                KernelKind::Keeper => h.keeper += 1,
            }
        }
        h
    }

    /// Histogram of kernel sizes: `(gates + resolvers, count)`, ascending by size.
    pub fn size_histogram(&self) -> Vec<(usize, usize)> {
        let mut m: HashMap<usize, usize> = HashMap::new();
        for k in &self.kernels {
            *m.entry(k.gates.len() + k.resolvers.len()).or_default() += 1;
        }
        let mut v: Vec<_> = m.into_iter().collect();
        v.sort_unstable();
        v
    }

    /// Check every structural invariant against `flat`: the kernels partition their nodes, the
    /// per-kernel net lists are consistent, `comb_order` lists every non-kernel node exactly once
    /// in topological order, and the non-kernel graph is acyclic.
    pub fn verify(&self, flat: &FlatNetlist) -> Result<(), String> {
        let g = Graph::new(flat);
        let n = g.num_nodes();
        if self.kernel_of_gate.len() != flat.gates.len()
            || self.kernel_of_resolver.len() != flat.resolvers.len()
        {
            return Err("kernel_of arrays have the wrong length".into());
        }
        let mut owner = vec![NONE; n];
        for (ki, k) in self.kernels.iter().enumerate() {
            if k.id as usize != ki {
                return Err(format!("kernel {ki} has id {}", k.id));
            }
            if k.gates.is_empty() && k.resolvers.is_empty() {
                return Err(format!("kernel {ki} is empty"));
            }
            let mut nets = Vec::new();
            for node in k.nodes() {
                let id = g.id(node);
                if owner[id as usize] != NONE {
                    return Err(format!("{node:?} is in two kernels"));
                }
                owner[id as usize] = ki as u32;
                if self.kernel_of(node) != Some(ki as u32) {
                    return Err(format!("kernel_of({node:?}) != {ki}"));
                }
                nets.push(g.out_net(id));
            }
            nets.sort_unstable();
            if nets != k.nets {
                return Err(format!("kernel {ki}: nets mismatch"));
            }
            let mut inputs: Vec<NetIdx> = k
                .nodes()
                .flat_map(|nd| g.in_nets(g.id(nd)).to_vec())
                .filter(|&net| {
                    let d = g.driver(net);
                    d == NONE || owner[d as usize] != ki as u32
                })
                .collect();
            inputs.sort_unstable();
            inputs.dedup();
            if inputs != k.inputs {
                return Err(format!("kernel {ki}: inputs mismatch"));
            }
        }
        for v in 0..n as u32 {
            let node = g.node(v);
            let expect = if owner[v as usize] == NONE {
                None
            } else {
                Some(owner[v as usize])
            };
            if self.kernel_of(node) != expect {
                return Err(format!("kernel_of({node:?}) inconsistent"));
            }
        }
        let mut pos = vec![NONE; n];
        for (i, &node) in self.comb_order.iter().enumerate() {
            let id = g.id(node);
            if owner[id as usize] != NONE {
                return Err(format!("{node:?} is in a kernel and in comb_order"));
            }
            if pos[id as usize] != NONE {
                return Err(format!("{node:?} twice in comb_order"));
            }
            pos[id as usize] = i as u32;
        }
        for v in 0..n as u32 {
            if owner[v as usize] == NONE && pos[v as usize] == NONE {
                return Err(format!("{:?} missing from comb_order", g.node(v)));
            }
        }
        for v in 0..n as u32 {
            if owner[v as usize] != NONE {
                continue;
            }
            for &w in g.succs(v) {
                if owner[w as usize] == NONE && pos[w as usize] <= pos[v as usize] {
                    return Err(format!(
                        "comb_order not topological: {:?} -> {:?}",
                        g.node(v),
                        g.node(w)
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Tunables of [`extract`] (SPEC §8.3, [OPEN-1]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtractOptions {
    /// `KERNEL_MAX_LOOP`: longest seed cycle (in gates/resolvers). Default 6.
    pub max_loop: usize,
    /// `KERNEL_MAX_BITS`: max `inputs + cut` bits of a Table kernel. Default 12.
    pub max_bits: usize,
    /// Explosion guard: a region (SCC of the raw graph) with more short cycles than this is not
    /// seeded; its loops become Sim kernels. Default 1,000,000.
    pub max_cycles_per_region: usize,
    /// Explosion guard: work units (visited nodes / explored edges) one root's cycle search may
    /// spend before that root is skipped. Default 65,536.
    pub max_seed_work_per_root: u64,
    /// Explosion guard: total work units the cycle searches of one region may spend; the
    /// remaining roots of the region are not searched (cycles found so far are kept). Bounds
    /// seeding to `O(max_seed_work_per_region)` per region even when every root exhausts its own
    /// budget. Default 2^27.
    pub max_seed_work_per_region: u64,
}

impl Default for ExtractOptions {
    fn default() -> Self {
        ExtractOptions {
            max_loop: 6,
            max_bits: 12,
            max_cycles_per_region: 1_000_000,
            max_seed_work_per_root: 1 << 16,
            max_seed_work_per_region: 1 << 27,
        }
    }
}

/// Extraction counters (diagnostics / tuning).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExtractStats {
    /// Short cycles found (in regions that were seeded).
    pub cycles: u64,
    /// Roots skipped because their search exceeded the per-root budget.
    pub roots_skipped: u64,
    /// Regions whose search stopped early because they exceeded the per-region work budget.
    pub regions_truncated: u64,
    /// Regions not seeded because they exceeded the per-region cycle cap.
    pub regions_skipped: u64,
    /// Kernels created from residual SCCs (step 5).
    pub residual_kernels: u64,
    /// Nodes added by private fan-in absorption.
    pub absorbed: u64,
}

/// A kernel under construction.
struct Work {
    nodes: Vec<u32>,
    inputs: HashSet<NetIdx>,
    /// Cached cut size (`None` = not computed / stale).
    cut: Option<usize>,
}

struct Extractor<'a> {
    flat: &'a FlatNetlist,
    g: Graph,
    opts: &'a ExtractOptions,
    kernel_of: Vec<u32>,
    works: Vec<Work>,
    stats: ExtractStats,
}

/// Extract the kernels of `flat` (SPEC §8.3). See the module docs.
pub fn extract(flat: &FlatNetlist, opts: &ExtractOptions) -> KernelSet {
    extract_with_stats(flat, opts).0
}

/// [`extract`], also returning counters.
pub fn extract_with_stats(flat: &FlatNetlist, opts: &ExtractOptions) -> (KernelSet, ExtractStats) {
    let g = Graph::new(flat);
    let n = g.num_nodes();
    let mut ex = Extractor {
        flat,
        g,
        opts,
        kernel_of: vec![NONE; n],
        works: Vec::new(),
        stats: ExtractStats::default(),
    };
    ex.seed();
    ex.absorb_all();
    ex.residual();
    ex.absorb_all();
    ex.keepers();
    let ks = ex.finish();
    (ks, ex.stats)
}

impl Extractor<'_> {
    fn is_keeper(&self, v: u32) -> bool {
        self.g.is_resolver(v)
            && self.flat.resolvers[(v as usize) - self.g.num_gates()].termination
                == Termination::Keeper
    }

    /// Steps 1–3: seed cycles, merge into cores.
    fn seed(&mut self) {
        let n = self.g.num_nodes();
        let succ = self.g.succ_csr();
        let pred = self.g.pred_csr();
        let regions = scc(succ, |_| true);
        let sizes = regions.sizes();
        let mut uf = UnionFind::new(n);
        let mut seeded = vec![false; n];
        let mut search = CycleSearch::new(n, self.opts.max_loop);
        // Bucket nodes by region (ascending within each region).
        let members = regions.members();
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        for (c, mem) in members.iter().enumerate() {
            let cyclic = sizes[c] > 1 || succ.row(mem[0]).contains(&mem[0]);
            if !cyclic {
                continue;
            }
            pairs.clear();
            let mut cycles = 0u64;
            let mut failed = false;
            let comp = &regions.comp;
            let mut region_budget = self.opts.max_seed_work_per_region;
            for &r in mem {
                if region_budget == 0 {
                    self.stats.regions_truncated += 1;
                    break;
                }
                let start = self.opts.max_seed_work_per_root.min(region_budget);
                let mut budget = start;
                let outcome = search.cycles_through(
                    succ,
                    pred,
                    r,
                    |w| comp[w as usize] == c as u32,
                    &mut budget,
                    |cyc| {
                        cycles += 1;
                        for &v in cyc {
                            pairs.push((r, v));
                        }
                    },
                );
                region_budget -= start - budget;
                if outcome == SearchOutcome::BudgetExhausted {
                    // Partial results of this root are still genuine cycles; keep them.
                    self.stats.roots_skipped += 1;
                }
                if cycles > self.opts.max_cycles_per_region as u64 {
                    failed = true;
                    break;
                }
            }
            if failed {
                self.stats.regions_skipped += 1;
                continue;
            }
            self.stats.cycles += cycles;
            for &(a, b) in &pairs {
                uf.union(a, b);
                seeded[a as usize] = true;
                seeded[b as usize] = true;
            }
        }
        let mut root_kernel: HashMap<u32, u32> = HashMap::new();
        for v in 0..n as u32 {
            if !seeded[v as usize] {
                continue;
            }
            let r = uf.find(v);
            let k = *root_kernel.entry(r).or_insert_with(|| {
                self.works.push(Work {
                    nodes: Vec::new(),
                    inputs: HashSet::new(),
                    cut: None,
                });
                (self.works.len() - 1) as u32
            });
            self.kernel_of[v as usize] = k;
            self.works[k as usize].nodes.push(v);
        }
        for k in 0..self.works.len() {
            self.recompute_inputs(k as u32);
        }
    }

    fn recompute_inputs(&mut self, k: u32) {
        let mut inputs = HashSet::new();
        for &v in &self.works[k as usize].nodes {
            for &net in self.g.in_nets(v) {
                let d = self.g.driver(net);
                if d == NONE || self.kernel_of[d as usize] != k {
                    inputs.insert(net);
                }
            }
        }
        self.works[k as usize].inputs = inputs;
        self.works[k as usize].cut = None;
    }

    fn cut_size(&mut self, k: u32, extra: Option<u32>) -> usize {
        let limit = self.opts.max_bits + 1;
        if extra.is_none()
            && let Some(c) = self.works[k as usize].cut
        {
            return c;
        }
        let mut nodes = self.works[k as usize].nodes.clone();
        nodes.extend(extra);
        let c = feedback_vertex_set(self.g.succ_csr(), &nodes, |v| self.is_keeper(v), limit)
            .map_or(limit + 1, |c| c.len());
        if extra.is_none() {
            self.works[k as usize].cut = Some(c);
        }
        c
    }

    /// Step 4: absorb private fan-in into every live kernel, to a fixpoint.
    fn absorb_all(&mut self) {
        loop {
            let mut queue: VecDeque<(u32, u32)> = VecDeque::new();
            for k in 0..self.works.len() {
                let mut ins: Vec<NetIdx> = self.works[k].inputs.iter().copied().collect();
                ins.sort_unstable();
                for net in ins {
                    let d = self.g.driver(net);
                    if d != NONE && self.kernel_of[d as usize] == NONE {
                        queue.push_back((d, k as u32));
                    }
                }
            }
            let mut changed = false;
            while let Some((d, k)) = queue.pop_front() {
                if self.try_absorb(d, k) {
                    changed = true;
                    for &net in self.g.in_nets(d) {
                        let dd = self.g.driver(net);
                        if dd != NONE && self.kernel_of[dd as usize] == NONE {
                            queue.push_back((dd, k));
                        }
                    }
                }
            }
            // An absorption elsewhere may have shrunk a kernel's input count so that a refused
            // candidate now fits: iterate until nothing changes.
            if !changed {
                break;
            }
        }
    }

    fn try_absorb(&mut self, d: u32, k: u32) -> bool {
        if self.kernel_of[d as usize] != NONE {
            return false;
        }
        let out = self.g.out_net(d);
        let readers = self.g.readers(out);
        if readers.is_empty() || readers.iter().any(|&r| self.kernel_of[r as usize] != k) {
            return false;
        }
        let mut new_inputs = 0usize;
        let mut reads_self = false;
        {
            let w = &self.works[k as usize];
            for &net in self.g.in_nets(d) {
                let dr = self.g.driver(net);
                if dr != NONE && self.kernel_of[dr as usize] == k {
                    reads_self = true;
                } else if !w.inputs.contains(&net) {
                    new_inputs += 1;
                }
            }
        }
        let old_inputs = self.works[k as usize].inputs.len();
        // `out` is currently an input (all its readers are in k); it stops being one.
        let new_in_total = old_inputs + new_inputs - 1;
        if new_in_total > old_inputs || reads_self || self.is_keeper(d) {
            let old_cut = self.cut_size(k, None);
            // A kernel whose cut already exceeds the budget stays over it (adding a node never
            // removes a cycle), so skip the feedback-vertex-set computation: on a large Sim
            // kernel it costs O(edges) per candidate and made absorption quadratic.
            let new_cut = if (reads_self || self.is_keeper(d)) && old_cut <= self.opts.max_bits {
                self.cut_size(k, Some(d))
            } else {
                old_cut
            };
            let old_bits = old_inputs + old_cut;
            let new_bits = new_in_total + new_cut;
            if new_bits > self.opts.max_bits.max(old_bits) {
                return false;
            }
            self.works[k as usize].cut = Some(new_cut);
        }
        self.kernel_of[d as usize] = k;
        let w = &mut self.works[k as usize];
        w.nodes.push(d);
        w.inputs.remove(&out);
        for &net in self.g.in_nets(d) {
            let dr = self.g.driver(net);
            if dr == NONE || self.kernel_of[dr as usize] != k {
                w.inputs.insert(net);
            }
        }
        self.stats.absorbed += 1;
        true
    }

    /// Step 5: cycles avoiding every kernel become kernels.
    fn residual(&mut self) {
        let succ = self.g.succ_csr();
        let kernel_of = &self.kernel_of;
        let s = scc(succ, |v| kernel_of[v as usize] == NONE);
        let sizes = s.sizes();
        let members = s.members();
        let mut new = Vec::new();
        for (c, mem) in members.iter().enumerate() {
            if mem.is_empty() {
                continue;
            }
            let cyclic = sizes[c] > 1 || succ.row(mem[0]).contains(&mem[0]);
            if cyclic {
                new.push(mem.clone());
            }
        }
        for mem in new {
            let k = self.works.len() as u32;
            for &v in &mem {
                self.kernel_of[v as usize] = k;
            }
            self.works.push(Work {
                nodes: mem,
                inputs: HashSet::new(),
                cut: None,
            });
            self.recompute_inputs(k);
            self.stats.residual_kernels += 1;
        }
    }

    /// Step 6: lone keeper resolvers.
    fn keepers(&mut self) {
        for r in 0..self.flat.resolvers.len() {
            let v = (self.g.num_gates() + r) as u32;
            if self.kernel_of[v as usize] == NONE && self.is_keeper(v) {
                let k = self.works.len() as u32;
                self.kernel_of[v as usize] = k;
                self.works.push(Work {
                    nodes: vec![v],
                    inputs: HashSet::new(),
                    cut: None,
                });
                self.recompute_inputs(k);
            }
        }
    }

    fn lca(&self, a: InstIdx, b: InstIdx) -> InstIdx {
        let insts = &self.flat.instances;
        if insts.is_empty() {
            return 0;
        }
        let depth = |mut x: InstIdx| {
            let mut d = 0;
            while let Some(p) = insts[x as usize].parent {
                x = p;
                d += 1;
            }
            d
        };
        let (mut a, mut b) = (a, b);
        let (mut da, mut db) = (depth(a), depth(b));
        while da > db {
            a = insts[a as usize].parent.unwrap_or(0);
            da -= 1;
        }
        while db > da {
            b = insts[b as usize].parent.unwrap_or(0);
            db -= 1;
        }
        while a != b {
            a = insts[a as usize].parent.unwrap_or(0);
            b = insts[b as usize].parent.unwrap_or(0);
        }
        a
    }

    /// Step 7 and assembly of the public result.
    fn finish(&mut self) -> KernelSet {
        let g = &self.g;
        let n = g.num_nodes();
        let ng = g.num_gates();
        // Renumber live kernels by smallest node.
        let mut live: Vec<(u32, u32)> = self
            .works
            .iter()
            .enumerate()
            .filter(|(_, w)| !w.nodes.is_empty())
            .map(|(i, w)| (*w.nodes.iter().min().expect("non-empty"), i as u32))
            .collect();
        live.sort_unstable();
        let mut remap = vec![NONE; self.works.len()];
        for (new, &(_, old)) in live.iter().enumerate() {
            remap[old as usize] = new as u32;
        }
        let kernel_of: Vec<u32> = self
            .kernel_of
            .iter()
            .map(|&k| if k == NONE { NONE } else { remap[k as usize] })
            .collect();
        let mut top_out = vec![false; self.flat.num_nets as usize];
        for p in &self.flat.top_outputs {
            for &net in &p.nets {
                top_out[net as usize] = true;
            }
        }
        let mut kernels = Vec::with_capacity(live.len());
        for (new, &(_, old)) in live.iter().enumerate() {
            let k = new as u32;
            let mut nodes = self.works[old as usize].nodes.clone();
            nodes.sort_unstable();
            let gates: Vec<GateIdx> = nodes
                .iter()
                .copied()
                .filter(|&v| (v as usize) < ng)
                .collect();
            let resolvers: Vec<u32> = nodes
                .iter()
                .copied()
                .filter(|&v| (v as usize) >= ng)
                .map(|v| v - ng as u32)
                .collect();
            let mut nets: Vec<NetIdx> = nodes.iter().map(|&v| g.out_net(v)).collect();
            nets.sort_unstable();
            let mut inputs: Vec<NetIdx> = Vec::new();
            for &v in &nodes {
                for &net in g.in_nets(v) {
                    let d = g.driver(net);
                    if d == NONE || kernel_of[d as usize] != k {
                        inputs.push(net);
                    }
                }
            }
            inputs.sort_unstable();
            inputs.dedup();
            let outputs: Vec<NetIdx> = nets
                .iter()
                .copied()
                .filter(|&net| {
                    top_out[net as usize]
                        || g.readers(net).iter().any(|&r| kernel_of[r as usize] != k)
                })
                .collect();
            let lone_keeper = nodes.len() == 1 && self.is_keeper(nodes[0]);
            let (kind, cut) = if lone_keeper {
                (KernelKind::Keeper, vec![g.out_net(nodes[0])])
            } else if inputs.len() > self.opts.max_bits {
                (KernelKind::Sim, Vec::new())
            } else {
                let limit = self.opts.max_bits - inputs.len();
                match feedback_vertex_set(g.succ_csr(), &nodes, |v| self.is_keeper(v), limit) {
                    Some(c) => {
                        let mut cut: Vec<NetIdx> = c.iter().map(|&v| g.out_net(v)).collect();
                        cut.sort_unstable();
                        (KernelKind::Table, cut)
                    }
                    None => (KernelKind::Sim, Vec::new()),
                }
            };
            let owner = gates
                .iter()
                .map(|&gi| self.flat.gates[gi as usize].owner)
                .reduce(|a, b| self.lca(a, b))
                .unwrap_or(0);
            kernels.push(Kernel {
                id: k,
                kind,
                gates,
                resolvers,
                inputs,
                outputs,
                nets,
                cut,
                owner,
            });
        }
        let order = topo_order(g.succ_csr(), |v| kernel_of[v as usize] == NONE);
        let comb_order: Vec<Node> = match order {
            Ok(o) => o.into_iter().map(|v| g.node(v)).collect(),
            Err(cyc) => panic!(
                "kernel extraction left {} combinational nodes on cycles",
                cyc.len()
            ),
        };
        let to_opt = |k: u32| if k == NONE { None } else { Some(k) };
        let ks = KernelSet {
            kernels,
            kernel_of_gate: kernel_of[..ng].iter().map(|&k| to_opt(k)).collect(),
            kernel_of_resolver: kernel_of[ng..n].iter().map(|&k| to_opt(k)).collect(),
            comb_order,
        };
        debug_assert_eq!(ks.verify(self.flat), Ok(()));
        ks
    }
}
