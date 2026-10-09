//! Analysis of one event: arrival windows per Fast round, then setup / hold / hazard checks.
//!
//! # Model
//!
//! Time is measured in ticks after the event. For every net the analysis computes a list of
//! [`Win`]dows, one per Fast *round label* `r` (SPEC §6.2): label 0 = changes caused by the
//! event's sources through combinational logic only; a kernel transition triggered by inputs of
//! label `r` produces output changes of label `r + 1`. A window says: the net may change (possibly
//! several times if `multi`) at ticks `min..=max`, and only if the path condition `cond` holds.
//!
//! * **Combinational gates** add their delay (1; resolvers 0). A gate whose stable input has a
//!   known controlling value never changes. A change on input `a` is *blocked* when another input
//!   reaches a known controlling final value strictly before `a` can change. Stable inputs with
//!   statically unknown value contribute literals (they must be non-controlling for anything to
//!   pass). Contributions with contradictory conditions are dropped. The output may glitch
//!   (`multi`) if two compatible contributions of the same label arrive at different ticks, or one
//!   of them glitches.
//! * **Kernels** are queried per input label through [`Model::round`] (see `kernel_round`);
//!   non-exact bursts go through the timed check ([`Model::timed`]) first. Rounds of one kernel
//!   whose bursts overlap in time are grouped and accepted only if the merged burst passes the
//!   timed check or is burst-safe (see `kernel_eval`).
//!
//! The hold condition is exact at equality: if a later burst reaches the kernel's inputs at tick
//! `t ≥ S`, where `S` is the tick of the last kernel-net change of the earlier transition, then at
//! `t` the kernel holds its stable state and the gates reading the new inputs evaluate from exactly
//! the values the table transition starts from.
//!
//! The analysis iterates to a fixpoint (everything grows monotonically; kernel output windows are
//! joined with those of the previous iteration, since a burst may pass the timed check with
//! narrow input windows and need the coarser analysis with wider ones). Labels beyond the delta
//! limit mean the event may never settle (e.g. a loop through transparent kernels).

use std::collections::BTreeSet;

use panda_compile::{KernelSet, Node};
use panda_netlist::eval::{eval_gate, resolve, terminate};
use panda_netlist::flat::{Driver, FlatNetlist, FlatOp, NetIdx};
use panda_netlist::{DiagKind, Diagnostic, Logic, Severity, Termination};

use crate::cond::Cond;
use crate::model::{Arrival, Burst, INF, Model, ModelLimits, RoundSig, Timed, Transient};
use crate::schedule::EventSpec;

/// Largest number of kernel inputs a [`RoundSig`] describes (bit masks are `u32`); kernels with
/// more inputs are analyzed fully conservatively.
const MAX_SIG_INPUTS: usize = 31;

/// No predecessor.
const NO_PRED: (NetIdx, u16) = (u32::MAX, 0);

/// Possible changes of a net with one round label.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Win {
    pub label: u16,
    pub min: u32,
    pub max: u32,
    pub multi: bool,
    pub cond: Cond,
    /// Net and label that determined `max` (critical-path reconstruction).
    pub pred: (NetIdx, u16),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Gate(FlatOp),
    Resolver(Termination),
    /// Output of an ideal block (Timing mode with ideal blocks, SPEC §5.2): changes exactly this
    /// many ticks after an input in its cone (transport delay, no internal glitches).
    Ideal(u32),
}

/// A combinational node, in topological order.
#[derive(Clone, Debug)]
struct CombNode {
    out: NetIdx,
    /// Distinct input nets (gates: first-occurrence order; `Tri`: `[d, en]`).
    ins: Vec<NetIdx>,
    /// All inputs in order (for static evaluation).
    raw: Vec<NetIdx>,
    op: Op,
}

/// Static, event-independent context.
pub(crate) struct Ctx<'a, 't> {
    pub flat: &'a FlatNetlist,
    pub ks: &'a KernelSet,
    pub models: Vec<Model<'t>>,
    pub kernel_model: Vec<usize>,
    /// Kernels analyzed conservatively (an input may carry `Z`).
    pub kernel_conservative: Vec<bool>,
    /// Position of each kernel output in `Kernel::nets`.
    kernel_out_pos: Vec<Vec<usize>>,
    /// Root net and polarity of every net (through combinational NOT/BUF chains).
    pub roots: Vec<(NetIdx, bool)>,
    comb: Vec<CombNode>,
    /// The nodes whose change windows are computed, in topological order: `comb`, except that
    /// the gates and resolvers of every ideal block are replaced by one [`Op::Ideal`] node per
    /// block output (inputs = the block inputs in its cone).
    tcomb: Vec<CombNode>,
    /// Kernels with an ideal-block element on one of their cycles: `(kernel, block path)`.
    pub ideal_conflicts: Vec<(usize, String)>,
    /// Per kernel, per input: extra latency from absorbed ideal-block elements reachable from
    /// that input (empty = none). Rounds in which such an input changes are analyzed
    /// delay-independently (ternary burst safety even for single clean changes).
    ideal_extra: Vec<Vec<u32>>,
    /// Kernels reading each net.
    kernel_readers: Vec<Vec<u32>>,
    /// Static value of undriven nets (`None` = driven).
    undriven: Vec<Option<Logic>>,
    pub limits: ModelLimits,
    pub delta_limit: u16,
    pub max_iterations: usize,
    /// Window bound: a change later than this means the event may never settle.
    pub max_window: u32,
    pub advisory: bool,
    /// Element evaluations left for timed burst checks in the current event (beyond: the
    /// delay-independent ternary analysis is used).
    timed_budget: u64,
}

/// Element evaluations of timed burst checks per event.
const TIMED_EVENT_BUDGET: u64 = 1 << 22;

/// Result of one event.
#[derive(Clone, Debug, Default)]
pub(crate) struct EventOutcome {
    /// Ticks after the event until the last change anywhere (`None` = may never settle).
    pub settle: Option<u32>,
    /// Number of Fast rounds (highest label + 1).
    pub rounds: u32,
    pub diags: Vec<Diagnostic>,
    /// Net names along the path to the last change.
    pub path: Vec<String>,
}

/// Outcome of one kernel round.
#[derive(Clone, Debug, Default)]
struct RoundOutcome {
    outs: Vec<(NetIdx, Win)>,
    settle: u32,
    min_arr: u32,
    max_arr: u32,
    exact: bool,
    safe: bool,
    c_inputs: Vec<NetIdx>,
    multi_inputs: Vec<NetIdx>,
    order_sensitive: bool,
    x_result: bool,
    /// Some feasible transition may never settle (see `RoundResult::may_not_settle`).
    may_not_settle: bool,
    pred: (NetIdx, u16),
}

fn known(v: Logic) -> Option<bool> {
    match v {
        Logic::Zero => Some(false),
        Logic::One => Some(true),
        _ => None,
    }
}

impl<'a, 't> Ctx<'a, 't> {
    /// Build the context. `models`/`kernel_model` come from the caller; `ideal` lists the ideal
    /// blocks Timing mode uses (empty for glass-box analysis).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        flat: &'a FlatNetlist,
        ks: &'a KernelSet,
        models: Vec<Model<'t>>,
        kernel_model: Vec<usize>,
        limits: ModelLimits,
        delta_limit: u16,
        advisory: bool,
        ideal: &[panda_timing::IdealBlock],
    ) -> Ctx<'a, 't> {
        let n = flat.num_nets as usize;
        let drivers = flat.drivers();
        let mut comb = Vec::with_capacity(ks.comb_order.len());
        for &node in &ks.comb_order {
            match node {
                Node::Gate(g) => {
                    let g = &flat.gates[g as usize];
                    let mut ins: Vec<NetIdx> = Vec::with_capacity(g.inputs.len());
                    for &i in &g.inputs {
                        if !ins.contains(&i) {
                            ins.push(i);
                        }
                    }
                    if g.op == FlatOp::Tri {
                        ins = g.inputs.clone();
                    }
                    comb.push(CombNode {
                        out: g.output,
                        ins,
                        raw: g.inputs.clone(),
                        op: Op::Gate(g.op),
                    });
                }
                Node::Resolver(r) => {
                    let r = &flat.resolvers[r as usize];
                    let mut ins = r.drivers.clone();
                    ins.dedup();
                    comb.push(CombNode {
                        out: r.output,
                        ins,
                        raw: r.drivers.clone(),
                        op: Op::Resolver(r.termination),
                    });
                }
            }
        }
        let mut roots: Vec<(NetIdx, bool)> = (0..n as NetIdx).map(|i| (i, false)).collect();
        for c in &comb {
            if let Op::Gate(op @ (FlatOp::Not | FlatOp::Buf)) = c.op {
                let (r, p) = roots[c.raw[0] as usize];
                roots[c.out as usize] = (r, p ^ (op == FlatOp::Not));
            }
        }
        let mut kernel_readers = vec![Vec::new(); n];
        for (ki, k) in ks.kernels.iter().enumerate() {
            for &i in &k.inputs {
                kernel_readers[i as usize].push(ki as u32);
            }
        }
        let undriven: Vec<Option<Logic>> = (0..n)
            .map(|i| {
                (drivers[i] == Driver::None).then(|| {
                    flat.undriven_termination
                        .get(&(i as NetIdx))
                        .map_or(Logic::Z, |&t| terminate(t, Logic::X))
                })
            })
            .collect();
        // Kernels whose inputs may carry Z or X in steady state (tri-state, resolvers, undriven
        // nets, Z/X constants) are analyzed conservatively: tables index binary inputs only.
        let mut may_z = vec![false; n];
        for (i, d) in drivers.iter().enumerate() {
            may_z[i] = match *d {
                Driver::None => !undriven[i].is_some_and(Logic::is_known),
                Driver::Gate(g) => flat.gates[g as usize].op == FlatOp::Tri,
                Driver::Resolver(_) => true,
                Driver::Source(s) => {
                    matches!(
                        flat.sources[s as usize].kind,
                        panda_netlist::flat::SourceKind::Constant(Logic::Z | Logic::X)
                    )
                }
            };
        }
        let kernel_conservative = ks
            .kernels
            .iter()
            .map(|k| k.inputs.iter().any(|&i| may_z[i as usize]))
            .collect();
        let kernel_out_pos = ks
            .kernels
            .iter()
            .map(|k| {
                k.outputs
                    .iter()
                    .map(|o| k.nets.binary_search(o).expect("output is a kernel net"))
                    .collect()
            })
            .collect();
        let max_window = {
            let lat = models
                .iter()
                .map(|m| m.max_latency.max(m.bound))
                .max()
                .unwrap_or(0);
            let per_round = (flat.gates.len() as u64 + u64::from(lat) + 1).saturating_mul(2);
            (u64::from(delta_limit) + 1)
                .saturating_mul(per_round)
                .min(u64::from(u32::MAX / 2)) as u32
        };
        let IdealNodes {
            tcomb,
            conflicts: ideal_conflicts,
            extra: ideal_extra,
        } = timing_nodes(flat, ks, &comb, ideal);
        Ctx {
            flat,
            ks,
            models,
            kernel_model,
            kernel_conservative,
            kernel_out_pos,
            roots,
            comb,
            tcomb,
            ideal_conflicts,
            ideal_extra,
            kernel_readers,
            undriven,
            limits,
            delta_limit,
            max_iterations: 20_000,
            max_window,
            advisory,
            timed_budget: TIMED_EVENT_BUDGET,
        }
    }

    /// Extra ideal-block latency of a burst at kernel `kid` with the given changing inputs.
    fn round_extra(&self, kid: usize, inputs: impl Iterator<Item = usize>) -> u32 {
        let e = &self.ideal_extra[kid];
        if e.is_empty() {
            return 0;
        }
        inputs.map(|j| e[j]).max().unwrap_or(0)
    }

    fn name(&self, n: NetIdx) -> String {
        self.flat
            .net_names
            .get(n as usize)
            .cloned()
            .unwrap_or_else(|| format!("n{n}"))
    }

    /// Kernel location string: its first output (or first net).
    pub(crate) fn kernel_name(&self, kid: usize) -> String {
        let k = &self.ks.kernels[kid];
        self.name(*k.outputs.first().or(k.nets.first()).unwrap_or(&0))
    }

    /// Ternary evaluation of all combinational logic with the given source values (kernel nets
    /// unknown).
    fn static_eval(&self, src: &[Logic]) -> Vec<Logic> {
        let n = self.flat.num_nets as usize;
        let mut v = vec![Logic::X; n];
        for (i, u) in self.undriven.iter().enumerate() {
            if let Some(u) = u {
                v[i] = *u;
            }
        }
        for (s, &val) in self.flat.sources.iter().zip(src) {
            v[s.output as usize] = val;
        }
        let mut buf = Vec::new();
        for c in &self.comb {
            buf.clear();
            buf.extend(c.raw.iter().map(|&i| v[i as usize]));
            v[c.out as usize] = match c.op {
                Op::Gate(op) => eval_gate(op, &buf),
                Op::Resolver(t) => resolve(&buf, t, Logic::X).value,
                Op::Ideal(_) => unreachable!("ideal nodes are not statically evaluated"),
            };
        }
        v
    }

    /// Analyze one event.
    pub(crate) fn run_event(&mut self, spec: &EventSpec) -> EventOutcome {
        self.timed_budget = TIMED_EVENT_BUDGET;
        let pre_src: Vec<Logic> = spec.src.iter().map(|s| s.pre).collect();
        let post_src: Vec<Logic> = spec.src.iter().map(|s| s.post).collect();
        let n = self.flat.num_nets as usize;
        let mut ev = Ev {
            pre: self.static_eval(&pre_src),
            post: self.static_eval(&post_src),
            act: vec![Vec::new(); n],
            unbounded: false,
        };
        let mut dirty = vec![false; n];
        for (s, st) in self.flat.sources.iter().zip(&spec.src) {
            if st.changes {
                let o = s.output as usize;
                ev.act[o] = vec![Win {
                    label: 0,
                    min: 0,
                    max: 0,
                    multi: false,
                    cond: Cond::top(),
                    pred: NO_PRED,
                }];
                dirty[o] = true;
            }
        }
        let mut kernel_dirty = vec![false; self.ks.kernels.len()];
        let mut iterations = 0;
        loop {
            iterations += 1;
            if iterations > self.max_iterations {
                ev.unbounded = true;
                break;
            }
            for ci in 0..self.tcomb.len() {
                let c = &self.tcomb[ci];
                if !c.ins.iter().any(|&i| dirty[i as usize]) {
                    continue;
                }
                let w = self.comb_eval(c, &ev);
                let o = c.out as usize;
                if w != ev.act[o] {
                    ev.act[o] = w;
                    dirty[o] = true;
                }
            }
            let mut todo = Vec::new();
            for (net, d) in dirty.iter_mut().enumerate() {
                if *d {
                    for &k in &self.kernel_readers[net] {
                        if !kernel_dirty[k as usize] {
                            kernel_dirty[k as usize] = true;
                            todo.push(k as usize);
                        }
                    }
                    *d = false;
                }
            }
            if todo.is_empty() {
                break;
            }
            todo.sort_unstable();
            for kid in todo {
                kernel_dirty[kid] = false;
                let outs = self.kernel_outputs(kid, &mut ev);
                for (o, w) in outs {
                    // Kernel output windows only grow from one iteration to the next (a burst may
                    // pass the timed check with narrow input windows and fall back to the coarser
                    // analysis with wider ones): the fixpoint iteration stays monotone.
                    let w = join_wins(&ev.act[o as usize], w);
                    if w != ev.act[o as usize] {
                        if w.iter().any(|x| x.max > self.max_window) {
                            ev.unbounded = true;
                        }
                        ev.act[o as usize] = w;
                        dirty[o as usize] = true;
                    }
                }
            }
            if ev.unbounded {
                // The event is reported as never settling; further rounds add nothing.
                break;
            }
        }
        self.finish_event(spec, &mut ev)
    }

    /// Windows of a combinational node's output.
    fn comb_eval(&self, c: &CombNode, ev: &Ev) -> Vec<Win> {
        let stable = |i: NetIdx| ev.act[i as usize].is_empty();
        let stat = |i: NetIdx| {
            let p = ev.pre[i as usize];
            if p.is_known() { p } else { ev.post[i as usize] }
        };
        let last_change = |i: NetIdx| ev.act[i as usize].iter().map(|w| w.max).max().unwrap_or(0);
        let (delay, ctrl) = match c.op {
            Op::Gate(FlatOp::And | FlatOp::Nand) => (1, Some(Logic::Zero)),
            Op::Gate(FlatOp::Or | FlatOp::Nor) => (1, Some(Logic::One)),
            Op::Gate(_) => (1, None),
            Op::Resolver(_) => (0, None),
            Op::Ideal(l) => (l, None),
        };
        let is_tri = c.op == Op::Gate(FlatOp::Tri);
        let mut gate_cond = Cond::top();
        // `Tri`: literals that apply to changes coming from `d` only.
        let mut d_cond = Cond::top();
        if let Some(cv) = ctrl {
            let mut lits = Vec::new();
            for &i in &c.ins {
                if stable(i) {
                    let s = stat(i);
                    if s == cv {
                        return Vec::new();
                    }
                    if !s.is_known() {
                        let (r, p) = self.roots[i as usize];
                        lits.push((r, (cv == Logic::Zero) ^ p));
                    }
                }
            }
            match Cond::from_lits(lits) {
                Some(cnd) => gate_cond = cnd,
                None => return Vec::new(),
            }
        }
        if is_tri {
            let en = c.ins[1];
            if stable(en) {
                match stat(en) {
                    Logic::Zero => return Vec::new(),
                    Logic::One => {}
                    _ => {
                        let (r, p) = self.roots[en as usize];
                        d_cond = Cond::from_lits(vec![(r, !p)]).expect("single literal");
                    }
                }
            }
        }
        // (label, min, max, multi, cond, src)
        let mut contribs: Vec<(u16, u32, u32, bool, Cond, NetIdx)> = Vec::new();
        for (idx, &j) in c.ins.iter().enumerate() {
            for w in &ev.act[j as usize] {
                if let Some(cv) = ctrl {
                    let blocked = c.ins.iter().any(|&b| {
                        b != j && !stable(b) && ev.post[b as usize] == cv && last_change(b) < w.min
                    });
                    if blocked {
                        continue;
                    }
                }
                let extra = if is_tri && idx == 0 {
                    let en = c.ins[1];
                    if !stable(en) && ev.post[en as usize] == Logic::Zero && last_change(en) < w.min
                    {
                        continue;
                    }
                    &d_cond
                } else {
                    &gate_cond
                };
                let Some(cond) = w.cond.union(extra) else {
                    continue;
                };
                contribs.push((w.label, w.min, w.max, w.multi, cond, j));
            }
        }
        contribs.sort_by_key(|x| x.0);
        let mut out = Vec::new();
        let mut i = 0;
        while i < contribs.len() {
            let mut j = i;
            while j < contribs.len() && contribs[j].0 == contribs[i].0 {
                j += 1;
            }
            let g = &contribs[i..j];
            let min = g.iter().map(|x| x.1).min().expect("nonempty");
            let (max, src) = g
                .iter()
                .map(|x| (x.2, x.5))
                .max_by_key(|x| x.0)
                .expect("nonempty");
            let identical = g.iter().all(|x| x.1 == x.2 && x.1 == g[0].1);
            let mut multi = g.iter().any(|x| x.3);
            if !multi && g.len() >= 2 && !identical {
                'outer: for a in 0..g.len() {
                    for b in a + 1..g.len() {
                        if g[a].4.compatible(&g[b].4) {
                            multi = true;
                            break 'outer;
                        }
                    }
                }
            }
            let mut cond = g[0].4.clone();
            for x in &g[1..] {
                cond = cond.intersect(&x.4);
            }
            out.push(Win {
                label: g[0].0,
                min: min + delay,
                max: max + delay,
                multi,
                cond,
                pred: (src, g[0].0),
            });
            i = j;
        }
        out
    }

    /// Output windows of kernel `kid` for all input labels.
    fn kernel_outputs(&mut self, kid: usize, ev: &mut Ev) -> Vec<(NetIdx, Vec<Win>)> {
        let ke = self.kernel_eval(kid, ev);
        if ke.unbounded {
            ev.unbounded = true;
        }
        ke.outs
    }

    fn kernel_labels(&self, kid: usize, ev: &Ev) -> Vec<u16> {
        let mut labels = BTreeSet::new();
        for &i in &self.ks.kernels[kid].inputs {
            for w in &ev.act[i as usize] {
                labels.insert(w.label);
            }
        }
        labels.into_iter().collect()
    }

    /// Round query for the burst of kernel `kid` made of all input changes with labels in
    /// `lo..=hi` (one round when `lo == hi`; several overlapping rounds otherwise).
    fn build_sig<'e>(&self, kid: usize, lo: u16, hi: u16, ev: &'e Ev) -> SigBuild<'e> {
        const PRE: u8 = 1;
        const FINAL: u8 = 2;
        let ks = self.ks;
        let kernel = &ks.kernels[kid];
        let k = kernel.inputs.len();
        let mut sig = RoundSig {
            rmask: vec![0; k],
            rval: vec![0; k],
            ..RoundSig::default()
        };
        let mut cwins: Vec<(usize, &'e Win)> = Vec::new();
        let mut vflags = vec![0u8; k];
        let mut conds: Vec<Option<Cond>> = vec![None; k];
        for (j, &net) in kernel.inputs.iter().enumerate() {
            let wins = &ev.act[net as usize];
            let has_lt = wins.iter().any(|w| w.label < lo);
            let has_gt = wins.iter().any(|w| w.label > hi);
            let pre = known(ev.pre[net as usize]);
            let post = known(ev.post[net as usize]);
            let bit = 1u32 << j;
            let in_range: Vec<&Win> = wins
                .iter()
                .filter(|w| (lo..=hi).contains(&w.label))
                .collect();
            let (uf, vf, u_known, v_known);
            if !in_range.is_empty() {
                let mut c = in_range[0].cond.clone();
                for w in &in_range[1..] {
                    c = c.intersect(&w.cond);
                }
                conds[j] = Some(c);
                cwins.extend(in_range.iter().map(|w| (j, *w)));
                sig.cmask |= bit;
                uf = if has_lt { 0 } else { PRE };
                vf = if has_gt { 0 } else { FINAL };
                u_known = if uf == PRE { pre } else { None };
                v_known = if vf == FINAL { post } else { None };
            } else {
                let f = match (wins.is_empty(), has_lt, has_gt) {
                    (true, _, _) => PRE | FINAL,
                    (false, true, false) => FINAL,
                    (false, false, true) => PRE,
                    _ => 0,
                };
                uf = f;
                vf = f;
                u_known = if f & PRE != 0 && pre.is_some() {
                    pre
                } else if f & FINAL != 0 {
                    post
                } else {
                    None
                };
                v_known = u_known;
            }
            if let Some(b) = u_known {
                sig.u_fix_mask |= bit;
                sig.u_fix_val |= u32::from(b) << j;
            }
            if sig.cmask & bit != 0
                && let Some(b) = v_known
            {
                sig.v_fix_mask |= bit;
                sig.v_fix_val |= u32::from(b) << j;
            }
            if uf & PRE != 0 {
                sig.u_pre_mask |= bit;
            }
            if vf & FINAL != 0 {
                sig.v_fin_mask |= bit;
                if v_known.is_none() {
                    sig.v_lit_mask |= bit;
                }
            }
            vflags[j] = vf;
        }
        // Correlated inputs (same root net).
        let in_roots: Vec<(NetIdx, bool)> = kernel
            .inputs
            .iter()
            .map(|&i| self.roots[i as usize])
            .collect();
        let mut done = 0u32;
        for j in 0..k {
            if done >> j & 1 == 1 {
                continue;
            }
            let mut members = 0u32;
            let mut pol = 0u32;
            for (jj, rt) in in_roots.iter().enumerate().skip(j) {
                if rt.0 == in_roots[j].0 {
                    members |= 1 << jj;
                    pol |= u32::from(rt.1) << jj;
                }
            }
            done |= members;
            if members.count_ones() >= 2 {
                sig.groups.push((members, pol));
            }
        }
        // Path-condition restrictions on final-valued kernel inputs.
        for (i, c) in conds.iter().enumerate() {
            let Some(c) = c else { continue };
            let mut rm = 0u32;
            let mut rv = 0u32;
            for &(root, b) in c.lits() {
                for (j, rt) in in_roots.iter().enumerate() {
                    if rt.0 == root && vflags[j] & FINAL != 0 {
                        let val = b ^ rt.1;
                        if rm >> j & 1 == 1 && (rv >> j & 1 == 1) != val {
                            sig.rnever |= 1 << i;
                        }
                        rm |= 1 << j;
                        rv |= u32::from(val) << j;
                    }
                }
            }
            sig.rmask[i] = rm;
            sig.rval[i] = rv & rm;
        }
        sig.finish();
        SigBuild {
            sig,
            cwins,
            in_roots,
        }
    }

    /// Analyze the round-`r` burst of kernel `kid`.
    fn kernel_round(&mut self, kid: usize, r: u16, ev: &Ev) -> RoundOutcome {
        if self.ks.kernels[kid].inputs.len() > MAX_SIG_INPUTS {
            return self.wide_round(kid, r, ev);
        }
        let SigBuild {
            sig,
            cwins,
            in_roots,
        } = self.build_sig(kid, r, r, ev);
        if cwins.is_empty() {
            return RoundOutcome::default();
        }
        let ks = self.ks;
        let kernel = &ks.kernels[kid];
        let k = kernel.inputs.len();
        // Kernels with absorbed ideal-block elements: Timing mode evaluates those elements as an
        // ideal block (other delays, no glitches), so only delay-independent (ternary-safe)
        // bursts are accepted, with the blocks' latency added.
        let extra = self.round_extra(kid, cwins.iter().map(|&(j, _)| j));
        let exact = extra == 0 && {
            let first = cwins[0].1;
            cwins.iter().all(|(_, w)| !w.multi)
                && (cwins.len() == 1
                    || cwins.iter().all(|(_, w)| {
                        w.min == w.max && w.min == first.min && first.min == first.max
                    }))
        };
        let conservative = self.kernel_conservative[kid];
        let mi = self.kernel_model[kid];
        let model = &mut self.models[mi];
        let mut rr = model.round(
            &sig,
            &self.limits,
            exact && !conservative,
            !exact && !conservative,
        );
        if conservative {
            make_conservative(&mut rr, sig.cmask, model.bound);
        }
        let m = model.m;
        let burst = if exact {
            None
        } else {
            Some(
                rr.burst
                    .clone()
                    .unwrap_or_else(|| Burst::unsafe_with(rr.lat_all, m)),
            )
        };
        // Timed check of a non-exact burst: exact unit-delay simulation outside the inputs'
        // arrival windows (see `Model::timed`).
        let timed = if !exact && !conservative && extra == 0 && rr.entries > 0 {
            let (arr, t0) = arrivals(&cwins, k, r);
            model
                .timed(&sig, &arr, 1, &self.limits, &mut self.timed_budget)
                .filter(|t| t.safe)
                .map(|t| (t, t0))
        } else {
            None
        };
        let timed = timed.map(|(td, t0)| {
            (
                timed_windows(&td, t0, &cwins, &in_roots),
                t0.saturating_add(td.settle),
            )
        });
        let (safe, bsteps) = burst.as_ref().map_or((true, 0), |b| (b.safe, b.steps));
        // Transiently changing state nets: from unit-delay simulation for exact bursts, from the
        // ternary A states otherwise.
        let transient = match &burst {
            None => rr.glitch.clone().unwrap_or_else(|| Transient::all(m)),
            Some(b) => b.transient.clone(),
        };
        let (max_arr, pred_j) = cwins
            .iter()
            .map(|&(j, w)| (w.max, j))
            .max_by_key(|x| x.0)
            .expect("nonempty");
        let min_arr = cwins.iter().map(|(_, w)| w.min).min().expect("nonempty");
        let pred = (kernel.inputs[pred_j], r);
        let lat_extra = if exact {
            0
        } else {
            bsteps.saturating_add(extra)
        };
        let mut outs = Vec::new();
        if r + 1 < self.delta_limit && rr.entries > 0 {
            for (oi, &o) in kernel.outputs.iter().enumerate() {
                let p = self.kernel_out_pos[kid][oi];
                let pi = rr.pos[p];
                let pulse = transient.has(p);
                if !pi.changes && !pulse {
                    continue;
                }
                // Inputs whose change can be involved; entry-derived literals common to every
                // entry in which the net changes (finally or transiently).
                let mut contrib = 0u32;
                let mut lit: Option<(u32, u32)> = None;
                let mut lat = 0u32;
                if pi.changes {
                    contrib |= pi.contrib;
                    lit = Some((pi.lit_mask, pi.lit_val));
                    lat = lat.max(pi.lat);
                }
                if pulse {
                    contrib |= sig.cmask;
                    let (tm, tv) = if p < 64 {
                        (transient.lit_mask[p], transient.lit_val[p])
                    } else {
                        (0, 0)
                    };
                    lit = Some(match lit {
                        None => (tm, tv),
                        Some((lm, lv)) => {
                            let mm = lm & tm & !(lv ^ tv);
                            (mm, lv & mm)
                        }
                    });
                    lat = lat.max(rr.lat_all);
                }
                let (lit_mask, lit_val) = lit.unwrap_or((0, 0));
                let mut min = INF;
                let mut cond: Option<Cond> = None;
                for &(j, w) in &cwins {
                    if contrib >> j & 1 == 0 {
                        continue;
                    }
                    let d = model.dist[j * m + p];
                    if d != INF {
                        min = min.min(w.min.saturating_add(d));
                    }
                    cond = Some(match cond {
                        None => w.cond.clone(),
                        Some(c) => c.intersect(&w.cond),
                    });
                }
                if min == INF {
                    min = min_arr;
                }
                // Exact burst: the unit-delay simulation of every feasible entry gives the ticks
                // (after the burst's change) at which this net changes.
                let sim_span = if exact {
                    rr.span.as_ref().and_then(|sp| sp[p])
                } else {
                    None
                };
                if let Some((f, _)) = sim_span {
                    min = min.max(min_arr.saturating_add(f));
                }
                let mut cond = cond.unwrap_or_default();
                let lits: Vec<(NetIdx, bool)> = (0..k)
                    .filter(|&j| lit_mask >> j & 1 == 1)
                    .map(|j| (in_roots[j].0, (lit_val >> j & 1 == 1) ^ in_roots[j].1))
                    .collect();
                if let Some(u) = Cond::from_lits(lits).and_then(|l| cond.union(&l)) {
                    cond = u;
                }
                let max = match sim_span {
                    Some((_, l)) => max_arr.saturating_add(l),
                    None => max_arr.saturating_add(lat.saturating_add(extra).max(lat_extra)),
                }
                .max(min);
                outs.push((
                    o,
                    Win {
                        label: r + 1,
                        min,
                        max,
                        multi: !exact || pulse,
                        cond,
                        pred,
                    },
                ));
            }
        }
        let mut settle = max_arr.saturating_add(if rr.entries > 0 {
            rr.lat_all.saturating_add(extra).max(lat_extra)
        } else {
            0
        });
        let mut safe = safe || rr.entries == 0;
        if let Some((wins, tsettle)) = timed {
            // The burst passed the timed check: its windows are sound. If the ternary check passed
            // too, both window sets are, and so is their intersection.
            let mut touts = Vec::new();
            if r + 1 < self.delta_limit {
                for (oi, &o) in kernel.outputs.iter().enumerate() {
                    let p = self.kernel_out_pos[kid][oi];
                    let Some((tmin, tmax, tmulti, tcond)) = &wins[p] else {
                        continue;
                    };
                    let mut w = Win {
                        label: r + 1,
                        min: *tmin,
                        max: *tmax,
                        multi: *tmulti,
                        cond: tcond.clone(),
                        pred,
                    };
                    if safe {
                        let Some((_, ow)) = outs.iter().find(|(n, _)| *n == o) else {
                            continue;
                        };
                        w.min = w.min.max(ow.min);
                        w.max = w.max.min(ow.max).max(w.min);
                        w.multi &= ow.multi;
                        if let Some(c) = w.cond.union(&ow.cond) {
                            w.cond = c;
                        }
                    }
                    touts.push((o, w));
                }
            }
            outs = touts;
            settle = if safe { settle.min(tsettle) } else { tsettle };
            safe = true;
        }
        RoundOutcome {
            outs,
            settle,
            min_arr,
            max_arr,
            exact,
            safe,
            c_inputs: cwins.iter().map(|&(j, _)| kernel.inputs[j]).collect(),
            multi_inputs: cwins
                .iter()
                .filter(|(_, w)| w.multi)
                .map(|&(j, _)| kernel.inputs[j])
                .collect(),
            order_sensitive: rr.order_sensitive,
            x_result: rr.x_result,
            may_not_settle: rr.may_not_settle,
            pred,
        }
    }

    /// Round of a kernel with more inputs than a [`RoundSig`] can describe: fully conservative
    /// (every output may change, glitching, until the settle bound; settling is unproven).
    fn wide_round(&mut self, kid: usize, r: u16, ev: &Ev) -> RoundOutcome {
        let kernel = &self.ks.kernels[kid];
        let cwins: Vec<(usize, &Win)> = kernel
            .inputs
            .iter()
            .enumerate()
            .flat_map(|(j, &net)| {
                ev.act[net as usize]
                    .iter()
                    .filter(move |w| w.label == r)
                    .map(move |w| (j, w))
            })
            .collect();
        if cwins.is_empty() {
            return RoundOutcome::default();
        }
        let model = &self.models[self.kernel_model[kid]];
        let m = model.m;
        let lat = model
            .max_latency
            .max(model.bound)
            .saturating_add(self.round_extra(kid, cwins.iter().map(|&(j, _)| j)));
        let first = cwins[0].1;
        let exact = cwins.iter().all(|(_, w)| !w.multi)
            && (cwins.len() == 1
                || cwins
                    .iter()
                    .all(|(_, w)| w.min == w.max && w.min == first.min && first.min == first.max));
        let (max_arr, pred_j) = cwins
            .iter()
            .map(|&(j, w)| (w.max, j))
            .max_by_key(|x| x.0)
            .expect("nonempty");
        let min_arr = cwins.iter().map(|(_, w)| w.min).min().expect("nonempty");
        let pred = (kernel.inputs[pred_j], r);
        let mut cond = cwins[0].1.cond.clone();
        for (_, w) in &cwins[1..] {
            cond = cond.intersect(&w.cond);
        }
        let settle = max_arr.saturating_add(lat);
        let mut outs = Vec::new();
        if r + 1 < self.delta_limit {
            for (oi, &o) in kernel.outputs.iter().enumerate() {
                let p = self.kernel_out_pos[kid][oi];
                let min = cwins
                    .iter()
                    .filter_map(|&(j, w)| {
                        let d = model.dist[j * m + p];
                        (d != INF).then(|| w.min.saturating_add(d))
                    })
                    .min()
                    .unwrap_or(min_arr);
                outs.push((
                    o,
                    Win {
                        label: r + 1,
                        min,
                        max: settle.max(min),
                        multi: true,
                        cond: cond.clone(),
                        pred,
                    },
                ));
            }
        }
        RoundOutcome {
            outs,
            settle,
            min_arr,
            max_arr,
            exact,
            safe: exact,
            c_inputs: cwins.iter().map(|&(j, _)| kernel.inputs[j]).collect(),
            multi_inputs: cwins
                .iter()
                .filter(|(_, w)| w.multi)
                .map(|&(j, _)| kernel.inputs[j])
                .collect(),
            order_sensitive: false,
            x_result: false,
            may_not_settle: !model.is_keeper(),
            pred,
        }
    }

    /// All rounds of kernel `kid`, with overlapping rounds grouped (see [`KernelEval`]).
    fn kernel_eval(&mut self, kid: usize, ev: &Ev) -> KernelEval {
        let labels = self.kernel_labels(kid, ev);
        let mut ke = KernelEval::default();
        let mut rounds: Vec<(u16, RoundOutcome)> = Vec::with_capacity(labels.len());
        for r in labels {
            if r >= self.delta_limit || r + 1 >= self.delta_limit {
                ke.unbounded = true;
            }
            if r < self.delta_limit {
                rounds.push((r, self.kernel_round(kid, r, ev)));
            }
        }
        // Group rounds whose bursts overlap in time (a later round arrives before the earlier one
        // settled). A group is acceptable if the timed check of the merged burst passes (rounds
        // are added one at a time while the group's exact settle time still overlaps the next
        // round), or else if the merged burst is burst-safe (ternary, any delays).
        let mut i = 0;
        while i < rounds.len() {
            let mut j = i;
            let mut gsettle = rounds[i].1.settle;
            let mut timed: Option<(Timed, u32, Vec<Option<TimedWin>>)> = None;
            let mut fallback = false;
            while j + 1 < rounds.len() && rounds[j + 1].1.min_arr < gsettle {
                j += 1;
                match self.timed_group(kid, rounds[i].0, rounds[j].0, ev) {
                    Some(tg) => {
                        gsettle = tg.1.saturating_add(tg.0.settle);
                        timed = Some(tg);
                    }
                    None => {
                        timed = None;
                        fallback = true;
                        break;
                    }
                }
            }
            if fallback {
                self.group_fallback(kid, ev, &mut rounds, i, &mut j, &mut ke);
            } else if let Some((_, _, wins)) = timed {
                self.apply_timed_group(kid, &mut rounds[i..=j], gsettle, &wins);
            }
            i = j + 1;
        }
        let ks = self.ks;
        ke.outs = ks.kernels[kid]
            .outputs
            .iter()
            .map(|&o| (o, Vec::new()))
            .collect();
        for (_, ro) in &rounds {
            for (net, w) in &ro.outs {
                if let Some(slot) = ke.outs.iter_mut().find(|(o, _)| o == net) {
                    slot.1.push(w.clone());
                }
            }
        }
        ke.rounds = rounds;
        ke
    }

    /// Timed check of the merged burst of labels `lo..=hi` at kernel `kid` (see `Model::timed`):
    /// `Some((result, burst start tick, per state net window))` if it passes.
    fn timed_group(
        &mut self,
        kid: usize,
        lo: u16,
        hi: u16,
        ev: &Ev,
    ) -> Option<(Timed, u32, Vec<Option<TimedWin>>)> {
        let k = self.ks.kernels[kid].inputs.len();
        if k > MAX_SIG_INPUTS || self.kernel_conservative[kid] {
            return None;
        }
        let sb = self.build_sig(kid, lo, hi, ev);
        if sb.cwins.is_empty() || self.round_extra(kid, sb.cwins.iter().map(|&(j, _)| j)) != 0 {
            return None;
        }
        let (arr, t0) = arrivals(&sb.cwins, k, lo);
        let mi = self.kernel_model[kid];
        let td = self.models[mi]
            .timed(
                &sb.sig,
                &arr,
                hi - lo + 1,
                &self.limits,
                &mut self.timed_budget,
            )
            .filter(|t| t.safe)?;
        let wins = timed_windows(&td, t0, &sb.cwins, &sb.in_roots);
        Some((td, t0, wins))
    }

    /// Accept a group of overlapping rounds whose merged burst passed the timed check: every
    /// round settles at `gsettle`; each output's windows in the group all get the same tick range
    /// (covering the timed simulation's changes and every round's earliest change) and are
    /// marked glitchy, so later kernels see them as one overlapping burst; an output that changes
    /// in Timing mode without a round window gets one in the group's first round.
    fn apply_timed_group(
        &self,
        kid: usize,
        group: &mut [(u16, RoundOutcome)],
        gsettle: u32,
        wins: &[Option<TimedWin>],
    ) {
        let (lo, pred) = (group[0].0, group[0].1.pred);
        for x in group.iter_mut() {
            x.1.settle = gsettle;
            x.1.safe = true;
        }
        let kernel = &self.ks.kernels[kid];
        for (oi, &o) in kernel.outputs.iter().enumerate() {
            let p = self.kernel_out_pos[kid][oi];
            let tw = &wins[p];
            let mut hmin = tw.as_ref().map_or(INF, |w| w.0);
            let mut hmax = tw.as_ref().map_or(0, |w| w.1);
            let mut any = false;
            for x in group.iter() {
                for (n, w) in &x.1.outs {
                    if *n == o {
                        hmin = hmin.min(w.min);
                        any = true;
                    }
                }
            }
            if !any && tw.is_none() {
                continue;
            }
            hmax = hmax.max(hmin);
            for x in group.iter_mut() {
                for (n, w) in &mut x.1.outs {
                    if *n == o {
                        w.min = hmin;
                        w.max = hmax;
                        w.multi = true;
                        if let Some(tw) = tw {
                            w.cond = w.cond.intersect(&tw.3);
                        }
                    }
                }
            }
            if !any
                && lo + 1 < self.delta_limit
                && let Some(tw) = tw
            {
                group[0].1.outs.push((
                    o,
                    Win {
                        label: lo + 1,
                        min: hmin,
                        max: hmax,
                        multi: true,
                        cond: tw.3.clone(),
                        pred,
                    },
                ));
            }
        }
    }

    /// The ternary merged-burst check of the group of overlapping rounds `i..=*j`: extends the
    /// group while rounds overlap and accepts it if the merged burst is burst-safe (outputs get
    /// the merged settle bound and are marked glitchy), otherwise records a [`HoldIssue`].
    fn group_fallback(
        &mut self,
        kid: usize,
        ev: &Ev,
        rounds: &mut [(u16, RoundOutcome)],
        i: usize,
        j: &mut usize,
        ke: &mut KernelEval,
    ) {
        let mut gsettle = rounds[i..=*j].iter().map(|x| x.1.settle).max().unwrap_or(0);
        let mut checked_upto = i;
        let mut merged_safe = true;
        let mut merged: Option<MergedBurst> = None;
        loop {
            while *j + 1 < rounds.len() && rounds[*j + 1].1.min_arr < gsettle {
                *j += 1;
                gsettle = gsettle.max(rounds[*j].1.settle);
            }
            if *j == checked_upto {
                break;
            }
            checked_upto = *j;
            let (lo, hi) = (rounds[i].0, rounds[*j].0);
            let mb = self.merged_burst(kid, lo, hi, ev);
            let (safe, bsteps) = (mb.safe, mb.steps);
            merged = Some(mb);
            let max_arr = rounds[i..=*j]
                .iter()
                .map(|x| x.1.max_arr)
                .max()
                .unwrap_or(0);
            if !safe {
                merged_safe = false;
                // Report the first overlapping pair.
                let mut s = rounds[i].1.settle;
                for x in i + 1..=*j {
                    if rounds[x].1.min_arr < s {
                        let (r1, r2) = (rounds[x - 1].0, rounds[x].0);
                        let early = rounds[x]
                            .1
                            .c_inputs
                            .iter()
                            .copied()
                            .min_by_key(|&n| {
                                ev.act[n as usize]
                                    .iter()
                                    .find(|w| w.label == r2)
                                    .map_or(INF, |w| w.min)
                            })
                            .expect("nonempty");
                        ke.hold.push(HoldIssue {
                            early_round: r2,
                            early_net: early,
                            early_tick: rounds[x].1.min_arr,
                            round: r1,
                            inputs: rounds[x - 1].1.c_inputs.clone(),
                            settle: s,
                        });
                        break;
                    }
                    s = s.max(rounds[x].1.settle);
                }
                break;
            }
            gsettle = gsettle.max(
                max_arr
                    .saturating_add(bsteps)
                    .saturating_add(self.round_extra(kid, 0..self.ks.kernels[kid].inputs.len())),
            );
        }
        if *j > i && merged_safe {
            for x in &mut rounds[i..=*j] {
                x.1.settle = gsettle;
                for (_, w) in &mut x.1.outs {
                    w.max = w.max.max(gsettle);
                    w.multi = true;
                }
            }
            // Nets that may change only under the merged (interleaved) burst: give them a
            // window in the group's first round, covering the whole group.
            if let Some(mb) = &merged {
                let (lo, pred) = (rounds[i].0, rounds[i].1.pred);
                let ks = self.ks;
                let kernel = &ks.kernels[kid];
                if lo + 1 < self.delta_limit {
                    for (oi, &o) in kernel.outputs.iter().enumerate() {
                        let p = self.kernel_out_pos[kid][oi];
                        let Some(c) = &mb.changes[p] else { continue };
                        let gmin = mb.earliest[p];
                        let outs = &mut rounds[i].1.outs;
                        match outs.iter_mut().find(|(n, _)| *n == o) {
                            Some((_, w)) => {
                                w.min = w.min.min(gmin);
                                w.cond = w.cond.intersect(c);
                            }
                            None => outs.push((
                                o,
                                Win {
                                    label: lo + 1,
                                    min: gmin,
                                    max: gsettle,
                                    multi: true,
                                    cond: c.clone(),
                                    pred,
                                },
                            )),
                        }
                    }
                }
            }
        }
    }

    /// Burst analysis of the merged burst of labels `lo..=hi` at kernel `kid`: safety, settle
    /// bound, and for every state net that may change during it, the literal condition (over
    /// root nets) common to the entries in which it does (`None` = cannot change).
    fn merged_burst(&mut self, kid: usize, lo: u16, hi: u16, ev: &Ev) -> MergedBurst {
        let mi = self.kernel_model[kid];
        let m = self.models[mi].m;
        if self.ks.kernels[kid].inputs.len() > MAX_SIG_INPUTS {
            // Never accepted (unsafe): the earliest ticks are not used.
            return MergedBurst {
                safe: false,
                steps: 0,
                changes: vec![Some(Cond::top()); m],
                earliest: vec![0; m],
            };
        }
        let sb = self.build_sig(kid, lo, hi, ev);
        // Earliest possible change of each state net: input arrival + gate distance.
        let model = &self.models[mi];
        let earliest: Vec<u32> = (0..m)
            .map(|p| {
                let d = sb
                    .cwins
                    .iter()
                    .filter_map(|&(j, w)| {
                        let d = model.dist[j * m + p];
                        (d != INF).then(|| w.min.saturating_add(d))
                    })
                    .min();
                d.unwrap_or_else(|| sb.cwins.iter().map(|(_, w)| w.min).min().unwrap_or(0))
            })
            .collect();
        if self.kernel_conservative[kid] {
            return MergedBurst {
                safe: false,
                steps: 0,
                changes: vec![Some(Cond::top()); m],
                earliest,
            };
        }
        let rr = self.models[mi].round(&sb.sig, &self.limits, false, true);
        if rr.entries == 0 {
            return MergedBurst {
                safe: true,
                steps: 0,
                changes: vec![None; m],
                earliest,
            };
        }
        let b = rr
            .burst
            .clone()
            .unwrap_or_else(|| Burst::unsafe_with(rr.lat_all, m));
        let to_cond = |mask: u32, val: u32| -> Cond {
            let lits: Vec<(NetIdx, bool)> = (0..sb.in_roots.len())
                .filter(|&j| mask >> j & 1 == 1)
                .map(|j| (sb.in_roots[j].0, (val >> j & 1 == 1) ^ sb.in_roots[j].1))
                .collect();
            Cond::from_lits(lits).unwrap_or_default()
        };
        let changes = (0..m)
            .map(|p| {
                let pi = rr.pos[p];
                let fin = pi.changes.then(|| to_cond(pi.lit_mask, pi.lit_val));
                let tra = b.transient.has(p).then(|| {
                    if p < 64 {
                        to_cond(b.transient.lit_mask[p], b.transient.lit_val[p])
                    } else {
                        Cond::top()
                    }
                });
                match (fin, tra) {
                    (Some(a), Some(c)) => Some(a.intersect(&c)),
                    (a, c) => a.or(c),
                }
            })
            .collect();
        MergedBurst {
            safe: b.safe,
            steps: b.steps,
            changes,
            earliest,
        }
    }

    /// Settle time, critical path and diagnostics of an analyzed event.
    fn finish_event(&mut self, spec: &EventSpec, ev: &mut Ev) -> EventOutcome {
        let mut out = EventOutcome::default();
        let mut best: (u32, Option<(NetIdx, u16)>, Option<usize>) = (0, None, None);
        let mut max_label = 0u16;
        for (n, wins) in ev.act.iter().enumerate() {
            for w in wins {
                max_label = max_label.max(w.label);
                if w.max > best.0 || best.1.is_none() && best.2.is_none() {
                    best = (w.max, Some((n as NetIdx, w.label)), None);
                }
            }
        }
        for kid in 0..self.ks.kernels.len() {
            if self.ks.kernels[kid]
                .inputs
                .iter()
                .all(|&i| ev.act[i as usize].is_empty())
            {
                continue;
            }
            let ke = self.kernel_eval(kid, ev);
            let kname = self.kernel_name(kid);
            let names = |v: &[NetIdx]| {
                v.iter()
                    .map(|&n| self.name(n))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            // One diagnostic per kernel and problem (the earliest round).
            let mut seen = [false; 5];
            let mut first = |tag: usize| !std::mem::replace(&mut seen[tag], true);
            for (r, ro) in &ke.rounds {
                if ro.settle > best.0 {
                    best = (ro.settle, Some(ro.pred), Some(kid));
                }
                if !ro.exact && !ro.safe {
                    if !ro.multi_inputs.is_empty() {
                        if first(0) {
                            out.diags.push(
                            Diagnostic::error(
                                DiagKind::GuardHazard,
                                format!(
                                    "{}: kernel input(s) {} may glitch in round {r} and the kernel's \
                                     state depends on the glitch (not Fast-mode compatible)",
                                    spec.name,
                                    names(&ro.multi_inputs)
                                ),
                            )
                            .at(kname.clone()),
                        );
                        }
                    } else if first(1) {
                        out.diags.push(
                            Diagnostic::error(
                                DiagKind::HoldViolation,
                                format!(
                                    "{}: inputs {} change in the same round ({r}) at different ticks \
                                     ({}..{}) and the kernel's result depends on their arrival order",
                                    spec.name,
                                    names(&ro.c_inputs),
                                    ro.min_arr,
                                    ro.max_arr
                                ),
                            )
                            .at(kname.clone()),
                        );
                    }
                }
                if ro.may_not_settle && first(4) {
                    out.diags.push(
                        Diagnostic::error(
                            DiagKind::Oscillating,
                            format!(
                                "{}: a reachable round-{r} transition of this kernel may never \
                                 settle (oscillation): Fast mode yields X where Timing mode keeps \
                                 oscillating (SPEC §6.3 condition 5)",
                                spec.name
                            ),
                        )
                        .at(kname.clone()),
                    );
                }
                if ro.x_result && first(2) {
                    out.diags.push(
                        Diagnostic::warning(
                            DiagKind::Metastable,
                            format!(
                                "{}: a reachable round-{r} transition ends in a non-binary state \
                                 (collapse to X or oscillation)",
                                spec.name
                            ),
                        )
                        .at(kname.clone()),
                    );
                }
                if self.advisory && ro.order_sensitive && first(3) {
                    out.diags.push(Diagnostic {
                        kind: DiagKind::DelaySensitive,
                        severity: Severity::Info,
                        message: format!(
                            "{}: a reachable table entry with several inputs changing in round {r} is \
                             order-sensitive (advisory)",
                            spec.name
                        ),
                        location: Some(kname.clone()),
                    });
                }
            }
            if let Some(h) = ke.hold.first() {
                out.diags.push(
                    Diagnostic::error(
                        DiagKind::HoldViolation,
                        format!(
                            "{}: round-{} change of {} can arrive at tick {} but the round-{} transition \
                             (inputs {}) settles only at tick {}",
                            spec.name,
                            h.early_round,
                            self.name(h.early_net),
                            h.early_tick,
                            h.round,
                            names(&h.inputs),
                            h.settle
                        ),
                    )
                    .at(kname.clone()),
                );
            }
        }
        out.rounds = if ev.act.iter().any(|w| !w.is_empty()) {
            u32::from(max_label) + 1
        } else {
            0
        };
        if ev.unbounded {
            out.settle = None;
            out.diags.push(Diagnostic::error(
                DiagKind::SetupViolation,
                format!(
                    "{}: the event may not settle (more than {} Fast rounds, or kernel bursts that keep \
                     overlapping; e.g. a loop through transparent kernels)",
                    spec.name, self.delta_limit
                ),
            ));
        } else {
            out.settle = Some(best.0);
        }
        // Critical path.
        let mut path = Vec::new();
        if let Some(kid) = best.2 {
            path.push(format!("[kernel {}]", self.kernel_name(kid)));
        }
        let mut cur = best.1;
        let mut guard = 0;
        while let Some((net, label)) = cur {
            if net == NO_PRED.0 || guard > 100_000 {
                break;
            }
            guard += 1;
            path.push(self.name(net));
            cur = ev.act[net as usize]
                .iter()
                .find(|w| w.label == label)
                .map(|w| w.pred)
                .filter(|p| p.0 != NO_PRED.0);
        }
        path.reverse();
        out.path = path;
        out
    }
}

/// Ideal-block handling (see [`Ctx::tcomb`]).
struct IdealNodes {
    tcomb: Vec<CombNode>,
    /// `(kernel, block path)`: a block element lies on a cycle of the kernel.
    conflicts: Vec<(usize, String)>,
    /// Per kernel, per input (empty = no absorbed block): summed max latency of the ideal blocks
    /// with elements in the kernel reachable from that input.
    extra: Vec<Vec<u32>>,
}

/// The window nodes and the kernels that absorbed ideal-block elements.
///
/// A block output driven by combinational logic gets one [`Op::Ideal`] node; internal nets of a
/// block are not simulated in Timing mode (dropped), except in blocks of which a kernel absorbed
/// elements as private fan-in: there, the internal gate-level nodes are kept (their windows are
/// lower bounds of what the kernel sees; the kernel is analyzed delay-independently with extra
/// latency, see `kernel_round`). A block element on a cycle of a kernel is a conflict: the
/// kernel's transitions differ in kind between gate-level and ideal-block evaluation.
fn timing_nodes(
    flat: &FlatNetlist,
    ks: &KernelSet,
    comb: &[CombNode],
    ideal: &[panda_timing::IdealBlock],
) -> IdealNodes {
    let mut extra: Vec<Vec<u32>> = vec![Vec::new(); ks.kernels.len()];
    if ideal.is_empty() {
        return IdealNodes {
            tcomb: comb.to_vec(),
            conflicts: Vec::new(),
            extra,
        };
    }
    use std::collections::{HashMap, HashSet};
    let mut conflicts = Vec::new();
    // Per comb output net: the ideal node replacing it (`None` = internal, dropped).
    let mut replace: HashMap<NetIdx, Option<CombNode>> = HashMap::new();
    let drivers = flat.drivers();
    let fo = flat.fanout();
    for b in ideal {
        let in_block = |n: Node| match n {
            Node::Gate(g) => b.gates.binary_search(&g).is_ok(),
            Node::Resolver(r) => b.resolvers.binary_search(&r).is_ok(),
        };
        let mut absorbed = false;
        for (kid, k) in ks.kernels.iter().enumerate() {
            let mine: Vec<Node> = k
                .gates
                .iter()
                .map(|&g| Node::Gate(g))
                .chain(k.resolvers.iter().map(|&r| Node::Resolver(r)))
                .filter(|&n| in_block(n))
                .collect();
            if mine.is_empty() {
                continue;
            }
            absorbed = true;
            if extra[kid].is_empty() {
                extra[kid] = vec![0; k.inputs.len()];
            }
            for (j, &i) in k.inputs.iter().enumerate() {
                if reaches_any(flat, &fo, ks, kid, i, &mine) {
                    extra[kid][j] = extra[kid][j].saturating_add(b.max_latency());
                }
            }
            if mine.iter().any(|&n| on_kernel_cycle(flat, &fo, ks, kid, n)) {
                conflicts.push((kid, b.path.clone()));
            }
        }
        let inputs: HashSet<NetIdx> = b.inputs.iter().copied().collect();
        for (&out, &lat) in b.outputs.iter().zip(&b.latency) {
            // Block inputs in the cone of `out`.
            let mut cone = Vec::new();
            let mut seen = HashSet::new();
            let mut work = vec![out];
            while let Some(n) = work.pop() {
                if !seen.insert(n) {
                    continue;
                }
                if inputs.contains(&n) {
                    cone.push(n);
                    continue;
                }
                match drivers[n as usize] {
                    Driver::Gate(g) => work.extend(&flat.gates[g as usize].inputs),
                    Driver::Resolver(r) => work.extend(&flat.resolvers[r as usize].drivers),
                    _ => cone.push(n),
                }
            }
            cone.sort_unstable();
            replace.insert(
                out,
                Some(CombNode {
                    out,
                    ins: cone.clone(),
                    raw: cone,
                    op: Op::Ideal(lat),
                }),
            );
        }
        if !absorbed {
            for &n in &b.internal {
                replace.insert(n, None);
            }
        }
    }
    // The node of a block output takes the place of the output's driver, which comes after every
    // element of its cone (and so after the drivers of the cone's inputs) and before its readers.
    let mut tcomb = Vec::with_capacity(comb.len());
    for c in comb {
        match replace.get(&c.out) {
            None => tcomb.push(c.clone()),
            Some(Some(node)) => tcomb.push(node.clone()),
            Some(None) => {}
        }
    }
    IdealNodes {
        tcomb,
        conflicts,
        extra,
    }
}

/// True if a change of net `from` can reach one of `targets` through elements of kernel `kid`.
fn reaches_any(
    flat: &FlatNetlist,
    fo: &panda_netlist::flat::Fanout,
    ks: &KernelSet,
    kid: usize,
    from: NetIdx,
    targets: &[Node],
) -> bool {
    let out = |n: Node| match n {
        Node::Gate(g) => flat.gates[g as usize].output,
        Node::Resolver(r) => flat.resolvers[r as usize].output,
    };
    let mut seen = std::collections::HashSet::new();
    let mut work = vec![from];
    while let Some(net) = work.pop() {
        let net = net as usize;
        let readers = fo.gates[net]
            .iter()
            .map(|&g| Node::Gate(g))
            .chain(fo.resolvers[net].iter().map(|&r| Node::Resolver(r)));
        for r in readers {
            if ks.kernel_of(r) != Some(kid as u32) || !seen.insert(r) {
                continue;
            }
            if targets.contains(&r) {
                return true;
            }
            work.push(out(r));
        }
    }
    false
}

/// True if element `start` of kernel `kid` lies on a cycle of the kernel's elements.
fn on_kernel_cycle(
    flat: &FlatNetlist,
    fo: &panda_netlist::flat::Fanout,
    ks: &KernelSet,
    kid: usize,
    start: Node,
) -> bool {
    let out = |n: Node| match n {
        Node::Gate(g) => flat.gates[g as usize].output,
        Node::Resolver(r) => flat.resolvers[r as usize].output,
    };
    let in_kernel = |n: Node| ks.kernel_of(n) == Some(kid as u32);
    let mut seen = std::collections::HashSet::new();
    let mut work = vec![start];
    while let Some(n) = work.pop() {
        let net = out(n) as usize;
        let readers = fo.gates[net]
            .iter()
            .map(|&g| Node::Gate(g))
            .chain(fo.resolvers[net].iter().map(|&r| Node::Resolver(r)));
        for r in readers {
            if !in_kernel(r) {
                continue;
            }
            if r == start {
                return true;
            }
            if seen.insert(r) {
                work.push(r);
            }
        }
    }
    false
}

/// Make a round answer fully conservative (kernels whose inputs may carry `Z`).
fn make_conservative(rr: &mut crate::model::RoundResult, cmask: u32, bound: u32) {
    let lat = rr.lat_all.max(bound);
    for p in &mut rr.pos {
        *p = crate::model::PosInfo {
            changes: true,
            contrib: cmask,
            lit_mask: 0,
            lit_val: 0,
            lat,
        };
    }
    rr.lat_all = lat;
    let m = rr.pos.len();
    rr.glitch = Some(Transient::all(m));
    rr.span = None;
    rr.burst = Some(Burst::unsafe_with(lat, m));
    rr.entries = rr.entries.max(1);
}

/// A round query under construction.
struct SigBuild<'e> {
    sig: RoundSig,
    /// `(input index, window)` of every changing input (one entry per window in the range).
    cwins: Vec<(usize, &'e Win)>,
    /// Root and polarity of every kernel input.
    in_roots: Vec<(NetIdx, bool)>,
}

/// Result of [`Ctx::merged_burst`].
#[derive(Clone, Debug)]
struct MergedBurst {
    safe: bool,
    steps: u32,
    /// Per state net: `Some(condition)` if it may change during the merged burst.
    changes: Vec<Option<Cond>>,
    /// Per state net: earliest tick at which it can change.
    earliest: Vec<u32>,
}

/// A hold violation between two rounds at one kernel.
#[derive(Clone, Debug)]
struct HoldIssue {
    early_round: u16,
    early_net: NetIdx,
    early_tick: u32,
    round: u16,
    inputs: Vec<NetIdx>,
    settle: u32,
}

/// All rounds of one kernel in one event. Rounds whose bursts overlap in time are accepted when
/// the merged burst is burst-safe (their outputs then get the merged settle bound and are marked
/// glitchy); otherwise a [`HoldIssue`] is recorded.
#[derive(Clone, Debug, Default)]
struct KernelEval {
    rounds: Vec<(u16, RoundOutcome)>,
    outs: Vec<(NetIdx, Vec<Win>)>,
    hold: Vec<HoldIssue>,
    unbounded: bool,
}

/// Per-event mutable state.
struct Ev {
    pre: Vec<Logic>,
    post: Vec<Logic>,
    act: Vec<Vec<Win>>,
    unbounded: bool,
}

/// A state net's window from a timed burst: `(min, max, multi, cond)` in absolute ticks.
type TimedWin = (u32, u32, bool, Cond);

/// Arrivals of the changing inputs of a burst (`cwins`, labels from `lo`), relative to the
/// burst's first arrival, and that first arrival tick.
fn arrivals(cwins: &[(usize, &Win)], k: usize, lo: u16) -> (Vec<Option<Arrival>>, u32) {
    let t0 = cwins.iter().map(|(_, w)| w.min).min().unwrap_or(0);
    let mut arr: Vec<Option<Arrival>> = vec![None; k];
    let mut count = vec![0u32; k];
    for &(j, w) in cwins {
        count[j] += 1;
        let l = w.label - lo;
        arr[j] = Some(match arr[j] {
            None => Arrival {
                min: w.min - t0,
                max: w.max - t0,
                clean: !w.multi,
                first_label: l,
                last_label: l,
            },
            Some(a) => Arrival {
                min: a.min.min(w.min - t0),
                max: a.max.max(w.max - t0),
                clean: false,
                first_label: a.first_label.min(l),
                last_label: a.last_label.max(l),
            },
        });
    }
    (arr, t0)
}

/// Per state net, the window of a passed timed burst starting at `t0`: the condition is the
/// weakest one of the changes of the inputs with activity in the entries where the net changes
/// (one of them must happen), plus the entry-derived literals.
fn timed_windows(
    td: &Timed,
    t0: u32,
    cwins: &[(usize, &Win)],
    in_roots: &[(NetIdx, bool)],
) -> Vec<Option<TimedWin>> {
    td.nets
        .iter()
        .map(|tn| {
            if !tn.active {
                return None;
            }
            let mut base: Option<Cond> = None;
            for (j, w) in cwins {
                if tn.contrib >> j & 1 == 1 {
                    base = Some(match base {
                        None => w.cond.clone(),
                        Some(c) => c.intersect(&w.cond),
                    });
                }
            }
            // No input had activity: the net changes without any (cannot happen in a passed
            // check, but stay weak).
            let base = base.unwrap_or_default();
            let lits: Vec<(NetIdx, bool)> = (0..in_roots.len())
                .filter(|&j| tn.lit_mask >> j & 1 == 1)
                .map(|j| (in_roots[j].0, (tn.lit_val >> j & 1 == 1) ^ in_roots[j].1))
                .collect();
            let cond = Cond::from_lits(lits)
                .and_then(|l| base.union(&l))
                .unwrap_or_else(|| base.clone());
            Some((
                t0.saturating_add(tn.first),
                t0.saturating_add(tn.last),
                tn.multi,
                cond,
            ))
        })
        .collect()
}

/// Per label, the hull of the windows of `old` and `new` (conditions intersected, `multi` if
/// either is). Each describes the possible changes soundly, so the hull does too.
fn join_wins(old: &[Win], mut new: Vec<Win>) -> Vec<Win> {
    for o in old {
        match new.iter_mut().find(|w| w.label == o.label) {
            None => new.push(o.clone()),
            Some(w) => {
                if w == o {
                    continue;
                }
                w.min = w.min.min(o.min);
                if o.max > w.max {
                    w.max = o.max;
                    w.pred = o.pred;
                }
                w.multi |= o.multi;
                w.cond = w.cond.intersect(&o.cond);
            }
        }
    }
    new.sort_by_key(|w| w.label);
    new
}
