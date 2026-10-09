//! The gate/resolver graph of a [`FlatNetlist`] and generic graph algorithms.
//!
//! A [`Node`] is a gate or a resolver (the only elements that drive nets besides sources). There is
//! an edge `a → b` when `b` reads the net driven by `a`. Internally nodes have dense ids: gate `g`
//! is id `g`, resolver `r` is id `num_gates + r`.
//!
//! Algorithms ([`scc`], [`topo_order`], [`CycleSearch`]) work on any [`Csr`] adjacency structure and
//! are iterative (no recursion), so they handle graphs of millions of nodes.

use std::collections::VecDeque;

use panda_netlist::flat::{FlatNetlist, GateIdx, NetIdx};

/// Sentinel for "no node / no index".
pub const NONE: u32 = u32::MAX;

/// A gate or a resolver of a [`FlatNetlist`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Node {
    /// Index into [`FlatNetlist::gates`].
    Gate(GateIdx),
    /// Index into [`FlatNetlist::resolvers`].
    Resolver(u32),
}

/// Compressed sparse rows: row `i` is `adj[start[i]..start[i + 1]]`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Csr {
    start: Vec<u32>,
    adj: Vec<u32>,
}

impl Csr {
    /// Build from a row generator: `fill(i, buf)` appends the entries of row `i` to `buf`.
    pub fn from_fn(n: usize, mut fill: impl FnMut(u32, &mut Vec<u32>)) -> Csr {
        let mut start = Vec::with_capacity(n + 1);
        let mut adj = Vec::new();
        start.push(0);
        for i in 0..n {
            fill(i as u32, &mut adj);
            start.push(adj.len() as u32);
        }
        Csr { start, adj }
    }

    /// Build from an edge list (`(from, to)`, `from < n`). Rows keep the edge-list order.
    pub fn from_edges(n: usize, edges: &[(u32, u32)]) -> Csr {
        let mut start = vec![0u32; n + 1];
        for &(a, _) in edges {
            start[a as usize + 1] += 1;
        }
        for i in 0..n {
            start[i + 1] += start[i];
        }
        let mut pos = start.clone();
        let mut adj = vec![0u32; edges.len()];
        for &(a, b) in edges {
            adj[pos[a as usize] as usize] = b;
            pos[a as usize] += 1;
        }
        Csr { start, adj }
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.start.len().saturating_sub(1)
    }

    /// True if there are no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Entries of row `i`.
    #[inline]
    pub fn row(&self, i: u32) -> &[u32] {
        &self.adj[self.start[i as usize] as usize..self.start[i as usize + 1] as usize]
    }

    /// The transposed structure (edge `a → b` becomes `b → a`), with `n` rows.
    pub fn transpose(&self) -> Csr {
        let n = self.len();
        let mut edges = Vec::with_capacity(self.adj.len());
        for a in 0..n as u32 {
            for &b in self.row(a) {
                edges.push((b, a));
            }
        }
        Csr::from_edges(n, &edges)
    }
}

/// The gate/resolver graph of a flat netlist, with net ↔ node incidence.
#[derive(Clone, Debug)]
pub struct Graph {
    num_gates: u32,
    num_resolvers: u32,
    succ: Csr,
    pred: Csr,
    /// Output net of every node.
    out_net: Vec<NetIdx>,
    /// Input nets of every node (deduplicated, first-occurrence order).
    in_nets: Csr,
    /// Driving node of every net, or [`NONE`] (source-driven or undriven).
    driver: Vec<u32>,
    /// Reader nodes of every net (deduplicated, ascending).
    readers: Csr,
}

impl Graph {
    /// Build the graph of `flat`.
    pub fn new(flat: &FlatNetlist) -> Graph {
        let ng = flat.gates.len();
        let nr = flat.resolvers.len();
        let n = ng + nr;
        assert!(n < NONE as usize, "netlist too large");
        let nn = flat.num_nets as usize;
        let mut out_net = Vec::with_capacity(n);
        let mut driver = vec![NONE; nn];
        for (i, g) in flat.gates.iter().enumerate() {
            out_net.push(g.output);
            driver[g.output as usize] = i as u32;
        }
        for (i, r) in flat.resolvers.iter().enumerate() {
            out_net.push(r.output);
            driver[r.output as usize] = (ng + i) as u32;
        }
        let in_nets = Csr::from_fn(n, |v, buf| {
            let s = buf.len();
            let ins: &[NetIdx] = if (v as usize) < ng {
                &flat.gates[v as usize].inputs
            } else {
                &flat.resolvers[v as usize - ng].drivers
            };
            for &i in ins {
                if !buf[s..].contains(&i) {
                    buf.push(i);
                }
            }
        });
        // readers: transpose of node → input nets.
        let mut edges = Vec::with_capacity(in_nets.adj.len());
        for v in 0..n as u32 {
            for &net in in_nets.row(v) {
                edges.push((net, v));
            }
        }
        let readers = Csr::from_edges(nn, &edges);
        let succ = Csr::from_fn(n, |v, buf| {
            buf.extend_from_slice(readers.row(out_net[v as usize]))
        });
        let pred = Csr::from_fn(n, |v, buf| {
            for &net in in_nets.row(v) {
                let d = driver[net as usize];
                if d != NONE {
                    buf.push(d);
                }
            }
        });
        Graph {
            num_gates: ng as u32,
            num_resolvers: nr as u32,
            succ,
            pred,
            out_net,
            in_nets,
            driver,
            readers,
        }
    }

    /// Number of nodes (gates + resolvers).
    pub fn num_nodes(&self) -> usize {
        (self.num_gates + self.num_resolvers) as usize
    }

    /// Number of gates (node ids `0..num_gates` are gates).
    pub fn num_gates(&self) -> usize {
        self.num_gates as usize
    }

    /// Dense id of a node.
    #[inline]
    pub fn id(&self, node: Node) -> u32 {
        match node {
            Node::Gate(g) => g,
            Node::Resolver(r) => self.num_gates + r,
        }
    }

    /// Node of a dense id.
    #[inline]
    pub fn node(&self, id: u32) -> Node {
        if id < self.num_gates {
            Node::Gate(id)
        } else {
            Node::Resolver(id - self.num_gates)
        }
    }

    /// True if node `id` is a resolver.
    #[inline]
    pub fn is_resolver(&self, id: u32) -> bool {
        id >= self.num_gates
    }

    /// Successors (readers of the node's output net), deduplicated.
    #[inline]
    pub fn succs(&self, id: u32) -> &[u32] {
        self.succ.row(id)
    }

    /// Predecessors (drivers of the node's input nets that are nodes), deduplicated.
    #[inline]
    pub fn preds(&self, id: u32) -> &[u32] {
        self.pred.row(id)
    }

    /// Successor structure.
    pub fn succ_csr(&self) -> &Csr {
        &self.succ
    }

    /// Predecessor structure.
    pub fn pred_csr(&self) -> &Csr {
        &self.pred
    }

    /// Net driven by node `id`.
    #[inline]
    pub fn out_net(&self, id: u32) -> NetIdx {
        self.out_net[id as usize]
    }

    /// Distinct input nets of node `id`.
    #[inline]
    pub fn in_nets(&self, id: u32) -> &[NetIdx] {
        self.in_nets.row(id)
    }

    /// Node driving `net`, or [`NONE`] if it is source-driven or undriven.
    #[inline]
    pub fn driver(&self, net: NetIdx) -> u32 {
        self.driver[net as usize]
    }

    /// Nodes reading `net` (ascending, deduplicated).
    #[inline]
    pub fn readers(&self, net: NetIdx) -> &[u32] {
        self.readers.row(net)
    }
}

/// Strongly connected components.
#[derive(Clone, Debug)]
pub struct Sccs {
    /// Component of every node ([`NONE`] for nodes excluded by the filter). Components are
    /// numbered in completion order, which is a *reverse* topological order of the condensation:
    /// if there is an edge from component `a` to component `b != a`, then `b < a`.
    pub comp: Vec<u32>,
    /// Number of components.
    pub count: u32,
}

impl Sccs {
    /// Size of every component.
    pub fn sizes(&self) -> Vec<u32> {
        let mut s = vec![0u32; self.count as usize];
        for &c in &self.comp {
            if c != NONE {
                s[c as usize] += 1;
            }
        }
        s
    }

    /// Members of every component, ascending node order within each.
    pub fn members(&self) -> Vec<Vec<u32>> {
        let mut m = vec![Vec::new(); self.count as usize];
        for (v, &c) in self.comp.iter().enumerate() {
            if c != NONE {
                m[c as usize].push(v as u32);
            }
        }
        m
    }

    /// True if the component of `v` contains a cycle: it has ≥ 2 nodes or `v` has a self-loop.
    pub fn is_cyclic(&self, g: &Csr, sizes: &[u32], v: u32) -> bool {
        let c = self.comp[v as usize];
        c != NONE && (sizes[c as usize] > 1 || g.row(v).contains(&v))
    }
}

/// Tarjan's SCC algorithm (iterative) over the nodes `v` with `alive(v)`; edges to dead nodes are
/// ignored.
pub fn scc(g: &Csr, alive: impl Fn(u32) -> bool) -> Sccs {
    let n = g.len();
    let mut index = vec![NONE; n];
    let mut low = vec![0u32; n];
    let mut on_stack = vec![false; n];
    let mut stack: Vec<u32> = Vec::new();
    let mut comp = vec![NONE; n];
    let mut count = 0u32;
    let mut call: Vec<(u32, u32)> = Vec::new();
    let mut next = 0u32;
    for root in 0..n as u32 {
        if index[root as usize] != NONE || !alive(root) {
            continue;
        }
        index[root as usize] = next;
        low[root as usize] = next;
        next += 1;
        stack.push(root);
        on_stack[root as usize] = true;
        call.push((root, 0));
        while let Some(top) = call.last_mut() {
            let v = top.0;
            let row = g.row(v);
            if (top.1 as usize) < row.len() {
                let w = row[top.1 as usize];
                top.1 += 1;
                if !alive(w) {
                    continue;
                }
                if index[w as usize] == NONE {
                    index[w as usize] = next;
                    low[w as usize] = next;
                    next += 1;
                    stack.push(w);
                    on_stack[w as usize] = true;
                    call.push((w, 0));
                } else if on_stack[w as usize] {
                    low[v as usize] = low[v as usize].min(index[w as usize]);
                }
            } else {
                call.pop();
                if let Some(&(p, _)) = call.last() {
                    low[p as usize] = low[p as usize].min(low[v as usize]);
                }
                if low[v as usize] == index[v as usize] {
                    loop {
                        let w = stack.pop().expect("tarjan stack");
                        on_stack[w as usize] = false;
                        comp[w as usize] = count;
                        if w == v {
                            break;
                        }
                    }
                    count += 1;
                }
            }
        }
    }
    Sccs { comp, count }
}

/// Topological order (Kahn) of the nodes with `alive(v)`, considering only edges between alive
/// nodes. Ties are broken by ascending node id among the initially ready nodes and FIFO order
/// afterwards. Returns `Err(nodes left on cycles)` if the alive subgraph is cyclic.
pub fn topo_order(g: &Csr, alive: impl Fn(u32) -> bool) -> Result<Vec<u32>, Vec<u32>> {
    let n = g.len();
    let mut indeg = vec![0u32; n];
    let mut total = 0usize;
    for v in 0..n as u32 {
        if !alive(v) {
            continue;
        }
        total += 1;
        for &w in g.row(v) {
            if alive(w) {
                indeg[w as usize] += 1;
            }
        }
    }
    let mut queue: VecDeque<u32> = (0..n as u32)
        .filter(|&v| alive(v) && indeg[v as usize] == 0)
        .collect();
    let mut order = Vec::with_capacity(total);
    while let Some(v) = queue.pop_front() {
        order.push(v);
        for &w in g.row(v) {
            if alive(w) {
                indeg[w as usize] -= 1;
                if indeg[w as usize] == 0 {
                    queue.push_back(w);
                }
            }
        }
    }
    if order.len() == total {
        Ok(order)
    } else {
        Err((0..n as u32)
            .filter(|&v| alive(v) && indeg[v as usize] > 0)
            .collect())
    }
}

/// Outcome of a bounded search ([`CycleSearch::cycles_through`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchOutcome {
    /// Every qualifying cycle was reported.
    Complete,
    /// The work budget ran out; some cycles may be missing.
    BudgetExhausted,
}

/// Bounded simple-cycle enumeration with reusable scratch space.
///
/// [`CycleSearch::cycles_through`] reports every simple cycle of length ≤ `max_len` whose
/// *smallest* node is `root` (so iterating `root` over all nodes reports every short cycle exactly
/// once). The search first runs a backward BFS from `root` (depth `max_len - 1`) and then a forward
/// DFS that only enters nodes from which `root` is still reachable within the remaining length, so
/// the work is proportional to the local neighbourhood, not to the graph.
#[derive(Clone, Debug, Default)]
pub struct CycleSearch {
    /// Backward distance to the current root, valid where `stamp == cur`.
    dist: Vec<u32>,
    stamp: Vec<u32>,
    on_path: Vec<bool>,
    cur: u32,
    bfs: Vec<u32>,
    path: Vec<u32>,
    frames: Vec<(u32, u32)>,
    max_len: usize,
}

impl CycleSearch {
    /// Scratch space for graphs of `n` nodes, reporting cycles of at most `max_len` nodes.
    pub fn new(n: usize, max_len: usize) -> CycleSearch {
        CycleSearch {
            max_len,
            dist: vec![0; n],
            stamp: vec![0; n],
            on_path: vec![false; n],
            cur: 0,
            bfs: Vec::new(),
            path: Vec::new(),
            frames: Vec::new(),
        }
    }

    /// Enumerate the simple cycles of length ≤ `max_len` (in nodes, see [`CycleSearch::new`])
    /// through `root` that use only
    /// nodes `v ≥ root` with `allowed(v)`; `f` receives each cycle as a node list starting at
    /// `root`. Every visited node or explored edge costs one unit of `budget`.
    pub fn cycles_through(
        &mut self,
        succ: &Csr,
        pred: &Csr,
        root: u32,
        allowed: impl Fn(u32) -> bool,
        budget: &mut u64,
        mut f: impl FnMut(&[u32]),
    ) -> SearchOutcome {
        let max_len = self.max_len;
        if max_len == 0 {
            return SearchOutcome::Complete;
        }
        self.cur = self.cur.wrapping_add(1);
        if self.cur == 0 {
            self.stamp.fill(0);
            self.cur = 1;
        }
        let cur = self.cur;
        let max_len = max_len.min(u32::MAX as usize - 1) as u32;
        // Backward BFS: dist[w] = shortest path length w → root (edges), ≤ max_len - 1.
        self.bfs.clear();
        self.bfs.push(root);
        self.stamp[root as usize] = cur;
        self.dist[root as usize] = 0;
        let mut head = 0;
        while head < self.bfs.len() {
            let v = self.bfs[head];
            head += 1;
            let d = self.dist[v as usize];
            if d + 1 >= max_len {
                continue;
            }
            for &w in pred.row(v) {
                if w < root || self.stamp[w as usize] == cur || !allowed(w) {
                    continue;
                }
                if *budget == 0 {
                    return SearchOutcome::BudgetExhausted;
                }
                *budget -= 1;
                self.stamp[w as usize] = cur;
                self.dist[w as usize] = d + 1;
                self.bfs.push(w);
            }
        }
        // Forward DFS over simple paths from root.
        self.path.clear();
        self.frames.clear();
        self.path.push(root);
        self.on_path[root as usize] = true;
        self.frames.push((root, 0));
        let mut outcome = SearchOutcome::Complete;
        while let Some(top) = self.frames.last_mut() {
            let v = top.0;
            let row = succ.row(v);
            if (top.1 as usize) >= row.len() {
                self.frames.pop();
                self.path.pop();
                self.on_path[v as usize] = false;
                continue;
            }
            let w = row[top.1 as usize];
            top.1 += 1;
            if *budget == 0 {
                outcome = SearchOutcome::BudgetExhausted;
                break;
            }
            *budget -= 1;
            if w == root {
                f(&self.path);
                continue;
            }
            if w < root || self.stamp[w as usize] != cur || self.on_path[w as usize] {
                continue;
            }
            // Cycle nodes if we continue through w: path + w + (dist[w] - 1) more.
            if self.path.len() as u32 + self.dist[w as usize] > max_len {
                continue;
            }
            self.path.push(w);
            self.on_path[w as usize] = true;
            self.frames.push((w, 0));
        }
        for &v in &self.path {
            self.on_path[v as usize] = false;
        }
        self.path.clear();
        self.frames.clear();
        outcome
    }
}

/// Union-find with path halving and union by size.
#[derive(Clone, Debug)]
pub struct UnionFind {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl UnionFind {
    /// `n` singleton sets.
    pub fn new(n: usize) -> UnionFind {
        UnionFind {
            parent: (0..n as u32).collect(),
            size: vec![1; n],
        }
    }

    /// Representative of `x`'s set.
    pub fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let p = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = p;
            x = p;
        }
        x
    }

    /// Merge the sets of `a` and `b`.
    pub fn union(&mut self, a: u32, b: u32) {
        let (mut a, mut b) = (self.find(a), self.find(b));
        if a == b {
            return;
        }
        if self.size[a as usize] < self.size[b as usize] {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent[b as usize] = a;
        self.size[a as usize] += self.size[b as usize];
    }
}

/// Greedy feedback vertex set of the subgraph induced by `nodes` (any order, distinct): a set of
/// nodes whose removal leaves the subgraph acyclic. Nodes for which `forced(v)` holds are always
/// included (e.g. keeper resolvers, which depend on their own previous value). Repeatedly removes,
/// from each cyclic SCC, the node with the largest `in × out` degree inside the SCC (ties: lowest
/// id). Returns `None` as soon as the set would exceed `limit` nodes. The result is sorted.
pub fn feedback_vertex_set(
    succ: &Csr,
    nodes: &[u32],
    forced: impl Fn(u32) -> bool,
    limit: usize,
) -> Option<Vec<u32>> {
    let mut sorted: Vec<u32> = nodes.to_vec();
    sorted.sort_unstable();
    let local = |v: u32| sorted.binary_search(&v).ok().map(|i| i as u32);
    let m = sorted.len();
    // Local successor structure.
    let lsucc = Csr::from_fn(m, |i, buf| {
        for &w in succ.row(sorted[i as usize]) {
            if let Some(j) = local(w) {
                buf.push(j);
            }
        }
    });
    let mut removed = vec![false; m];
    let mut cut: Vec<u32> = Vec::new();
    for i in 0..m {
        if forced(sorted[i]) {
            removed[i] = true;
            cut.push(sorted[i]);
        }
    }
    if cut.len() > limit {
        return None;
    }
    loop {
        let s = scc(&lsucc, |i| !removed[i as usize]);
        let sizes = s.sizes();
        let members = s.members();
        let mut progress = false;
        for (c, mem) in members.iter().enumerate() {
            let cyclic = sizes[c] > 1 || lsucc.row(mem[0]).contains(&mem[0]);
            if !cyclic {
                continue;
            }
            // Pick the node with max in×out degree within the component.
            let mut indeg = vec![0u32; mem.len()];
            let mut outdeg = vec![0u32; mem.len()];
            for (a, &v) in mem.iter().enumerate() {
                for &w in lsucc.row(v) {
                    if s.comp[w as usize] == c as u32 {
                        outdeg[a] += 1;
                        if let Ok(b) = mem.binary_search(&w) {
                            indeg[b] += 1;
                        }
                    }
                }
            }
            let mut best = 0usize;
            let mut best_score = 0u64;
            for a in 0..mem.len() {
                let sc = u64::from(indeg[a]) * u64::from(outdeg[a]);
                if sc > best_score {
                    best_score = sc;
                    best = a;
                }
            }
            removed[mem[best] as usize] = true;
            cut.push(sorted[mem[best] as usize]);
            progress = true;
            if cut.len() > limit {
                return None;
            }
        }
        if !progress {
            break;
        }
    }
    cut.sort_unstable();
    Some(cut)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn csr(n: usize, e: &[(u32, u32)]) -> Csr {
        Csr::from_edges(n, e)
    }

    #[test]
    fn scc_basic() {
        // 0→1→2→0, 2→3, 3→3, 4
        let g = csr(5, &[(0, 1), (1, 2), (2, 0), (2, 3), (3, 3)]);
        let s = scc(&g, |_| true);
        assert_eq!(s.count, 3);
        assert_eq!(s.comp[0], s.comp[1]);
        assert_eq!(s.comp[1], s.comp[2]);
        assert_ne!(s.comp[3], s.comp[0]);
        // reverse topological numbering: 3 is reachable from 0, so comp(3) < comp(0)
        assert!(s.comp[3] < s.comp[0]);
        let sizes = s.sizes();
        assert!(s.is_cyclic(&g, &sizes, 3));
        assert!(!s.is_cyclic(&g, &sizes, 4));
        // filtered: removing node 1 breaks the 3-cycle
        let s2 = scc(&g, |v| v != 1);
        assert_eq!(s2.comp[1], NONE);
        assert_ne!(s2.comp[0], s2.comp[2]);
    }

    #[test]
    fn scc_deep_chain_no_recursion() {
        let n = 200_000u32;
        let mut e: Vec<(u32, u32)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        e.push((n - 1, 0));
        let s = scc(&csr(n as usize, &e), |_| true);
        assert_eq!(s.count, 1);
    }

    #[test]
    fn topo() {
        let g = csr(4, &[(0, 1), (1, 2), (0, 2), (3, 0)]);
        let o = topo_order(&g, |_| true).unwrap();
        let pos: Vec<usize> = (0..4)
            .map(|v| o.iter().position(|&x| x == v).unwrap())
            .collect();
        assert!(pos[3] < pos[0] && pos[0] < pos[1] && pos[1] < pos[2]);
        let c = csr(3, &[(0, 1), (1, 0), (1, 2)]);
        assert_eq!(topo_order(&c, |_| true).unwrap_err(), vec![0, 1, 2]);
        assert!(topo_order(&c, |v| v != 0).is_ok());
    }

    fn all_cycles(g: &Csr, max_len: usize) -> Vec<Vec<u32>> {
        let pred = g.transpose();
        let mut cs = CycleSearch::new(g.len(), max_len);
        let mut out = Vec::new();
        for r in 0..g.len() as u32 {
            let mut budget = u64::MAX;
            let o = cs.cycles_through(g, &pred, r, |_| true, &mut budget, |c| out.push(c.to_vec()));
            assert_eq!(o, SearchOutcome::Complete);
        }
        out.sort();
        out
    }

    #[test]
    fn bounded_cycles() {
        // 0⇄1, 1→2→3→1 (len 3), 3→3 (self loop), 0→4→5→6→7→0 (len 5)
        let g = csr(
            8,
            &[
                (0, 1),
                (1, 0),
                (1, 2),
                (2, 3),
                (3, 1),
                (3, 3),
                (0, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 0),
            ],
        );
        assert_eq!(
            all_cycles(&g, 6),
            vec![vec![0, 1], vec![0, 4, 5, 6, 7], vec![1, 2, 3], vec![3]]
        );
        assert_eq!(all_cycles(&g, 3), vec![vec![0, 1], vec![1, 2, 3], vec![3]]);
        assert_eq!(all_cycles(&g, 1), vec![vec![3]]);
    }

    #[test]
    fn bounded_cycles_complete_graph_count() {
        // K4 has 4·3/2 = 6 two-cycles, 8 three-cycles, 6 four-cycles.
        let mut e = Vec::new();
        for a in 0..4 {
            for b in 0..4 {
                if a != b {
                    e.push((a, b));
                }
            }
        }
        let g = csr(4, &e);
        assert_eq!(all_cycles(&g, 2).len(), 6);
        assert_eq!(all_cycles(&g, 3).len(), 14);
        assert_eq!(all_cycles(&g, 4).len(), 20);
    }

    #[test]
    fn budget_exhaustion() {
        let mut e = Vec::new();
        for a in 0..10 {
            for b in 0..10 {
                if a != b {
                    e.push((a, b));
                }
            }
        }
        let g = csr(10, &e);
        let pred = g.transpose();
        let mut cs = CycleSearch::new(10, 6);
        let mut budget = 50;
        let o = cs.cycles_through(&g, &pred, 0, |_| true, &mut budget, |_| {});
        assert_eq!(o, SearchOutcome::BudgetExhausted);
        // scratch state is reusable afterwards
        let mut budget = u64::MAX;
        let mut n = 0;
        let mut cs2 = CycleSearch::new(10, 2);
        cs2.cycles_through(&g, &pred, 8, |_| true, &mut budget, |_| n += 1);
        cs.cycles_through(&g, &pred, 9, |_| true, &mut budget, |_| n += 100);
        assert_eq!(n, 1); // only 8⇄9 has min node 8; nothing has min node 9
    }

    #[test]
    fn fvs() {
        let g = csr(5, &[(0, 1), (1, 0), (1, 2), (2, 3), (3, 1), (4, 4)]);
        let cut = feedback_vertex_set(&g, &[0, 1, 2, 3, 4], |_| false, 10).unwrap();
        assert_eq!(cut, vec![1, 4]);
        let s = scc(&g, |v| !cut.contains(&v));
        assert_eq!(s.count, 3);
        assert!(feedback_vertex_set(&g, &[0, 1, 2, 3, 4], |_| false, 1).is_none());
        assert_eq!(
            feedback_vertex_set(&g, &[2, 3], |v| v == 2, 4).unwrap(),
            vec![2]
        );
    }
}
