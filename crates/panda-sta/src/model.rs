//! Per-kernel-structure behavior model used by the event analysis.
//!
//! A [`Model`] is shared by all kernels with the same sub-netlist structure (and table). It answers
//! *round queries*: given what is known about every kernel input during one Fast round (which
//! inputs change, their values before/after when statically known, correlations and path-condition
//! restrictions — a [`RoundSig`]), which state nets can change, how late, under which
//! entry-derived literals ([`RoundResult`]). Answers come from enumerating the feasible
//! characterized table entries `(stable state s, inputs u → v)`; kernels without a table get a
//! conservative answer.
//!
//! Two extra, lazily computed facts per query:
//! * **burst safety** — Eichelberger ternary simulation (algorithm A with the changing inputs at
//!   `X`, then synchronous algorithm B with the new inputs): if it ends in the binary state the
//!   table predicts for every feasible entry, the result is independent of the arrival order,
//!   skew and glitches of the changing inputs, under arbitrary delays. The number of B steps bounds
//!   the settle time after the last input change.
//! * **output glitches** — unit-delay simulation ([`KernelSim`]) of every feasible entry, recording
//!   state nets that change more than once during the transition, and the first and last tick at
//!   which each state net changes.
//!
//! A third check is a separate query, [`Model::timed`]: unit-delay ternary simulation of every
//! feasible entry with the changing inputs at `X` only inside their arrival windows, compared
//! with Fast mode's result (the table transition, or the chain of transitions of several
//! overlapping rounds).

use std::collections::HashMap;
use std::sync::Arc;

use panda_compile::charz::input_vector;
use panda_compile::{CharzOptions, KernelSim, KernelTable, SubnetMap, Transition};
use panda_netlist::Logic;
use panda_netlist::eval::{eval_gate, resolve};
use panda_netlist::flat::FlatNetlist;

/// "Unreachable" distance.
pub(crate) const INF: u32 = u32::MAX;

/// Everything known about the inputs of one kernel during one round. See the module docs.
#[derive(Clone, Debug, Default)]
pub(crate) struct RoundSig {
    /// Cache key (encodes every field below).
    pub key: Vec<u32>,
    /// Inputs changing in this round.
    pub cmask: u32,
    /// Inputs whose value before the burst is statically known, and those values.
    pub u_fix_mask: u32,
    pub u_fix_val: u32,
    /// Changing inputs whose value after the burst is statically known, and those values.
    pub v_fix_mask: u32,
    pub v_fix_val: u32,
    /// Inputs whose value before the burst is their pre-event value (for correlations).
    pub u_pre_mask: u32,
    /// Inputs whose value after the burst is their final value (for correlations, restrictions).
    pub v_fin_mask: u32,
    /// Final-valued inputs with statically unknown value: entry-derived literals.
    pub v_lit_mask: u32,
    /// Inputs driven by the same root net: `(members, polarity)` (≥ 2 members).
    pub groups: Vec<(u32, u32)>,
    /// Per input `i`: if `i` actually changes, `(v & rmask[i]) == rval[i]` must hold.
    pub rmask: Vec<u32>,
    pub rval: Vec<u32>,
    /// Inputs whose path condition contradicts itself on kernel inputs: they cannot change.
    pub rnever: u32,
}

impl RoundSig {
    /// Fill [`RoundSig::key`] from the other fields.
    pub(crate) fn finish(&mut self) {
        let mut key = vec![
            self.cmask,
            self.u_fix_mask,
            self.u_fix_val,
            self.v_fix_mask,
            self.v_fix_val,
            self.u_pre_mask,
            self.v_fin_mask,
            self.v_lit_mask,
            self.rnever,
        ];
        for (i, (&m, &v)) in self.rmask.iter().zip(&self.rval).enumerate() {
            if m != 0 {
                key.extend([i as u32, m, v]);
            }
        }
        key.push(u32::MAX);
        for &(m, p) in &self.groups {
            key.extend([m, p]);
        }
        self.key = key;
    }

    fn group_ok(&self, x: u32, sel: u32) -> bool {
        self.groups.iter().all(|&(members, pol)| {
            let mm = members & sel;
            let y = (x ^ pol) & mm;
            y == 0 || y == mm
        })
    }

    /// Inputs (any, not only changing ones) whose path-condition restriction holds for `v`.
    fn restr_sat(&self, v: u32) -> u32 {
        let mut sat = 0u32;
        for (i, (&m, &val)) in self.rmask.iter().zip(&self.rval).enumerate() {
            if self.rnever >> i & 1 == 0 && v & m == val {
                sat |= 1 << i;
            }
        }
        sat
    }

    fn u_ok(&self, u: u32) -> bool {
        u & self.u_fix_mask == self.u_fix_val && self.group_ok(u, self.u_pre_mask)
    }

    fn v_ok(&self, u: u32, v: u32) -> bool {
        if v & self.v_fix_mask != self.v_fix_val || !self.group_ok(v, self.v_fin_mask) {
            return false;
        }
        let mut d = u ^ v;
        if d & self.rnever != 0 {
            return false;
        }
        while d != 0 {
            let i = d.trailing_zeros() as usize;
            d &= d - 1;
            if v & self.rmask[i] != self.rval[i] {
                return false;
            }
        }
        true
    }
}

/// State nets that may change *transiently* during a round (glitch, or a pulse that returns to
/// the initial value), with the entry-derived literals common to the entries in which they do.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Transient {
    /// Bit `p`: state net `p` may change transiently (all ones = unknown).
    pub mask: u64,
    /// Per state net: literal inputs (mask) and values common to those entries.
    pub lit_mask: Vec<u32>,
    pub lit_val: Vec<u32>,
}

impl Transient {
    fn none(m: usize) -> Transient {
        Transient {
            mask: 0,
            lit_mask: vec![0; m],
            lit_val: vec![0; m],
        }
    }

    /// Anything may change transiently; nothing is implied.
    pub(crate) fn all(m: usize) -> Transient {
        Transient {
            mask: u64::MAX,
            lit_mask: vec![0; m],
            lit_val: vec![0; m],
        }
    }

    /// True if state net `p` may change transiently.
    pub(crate) fn has(&self, p: usize) -> bool {
        p >= 64 || self.mask >> p & 1 == 1
    }

    /// Record that state net `p` changes transiently in an entry with literals `(e_mask, e_val)`.
    fn add(&mut self, p: usize, e_mask: u32, e_val: u32) {
        if p >= 64 {
            return;
        }
        if self.mask >> p & 1 == 1 {
            self.lit_mask[p] &= e_mask & !(self.lit_val[p] ^ e_val);
            self.lit_val[p] &= self.lit_mask[p];
        } else {
            self.mask |= 1 << p;
            self.lit_mask[p] = e_mask;
            self.lit_val[p] = e_val & e_mask;
        }
    }
}

/// Ternary burst analysis of a round (see the module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Burst {
    /// Every feasible entry ends, under arbitrary input skew and glitches, in the table's state.
    pub safe: bool,
    /// Settle bound after the last input change (B steps), meaningful when `safe`.
    pub steps: u32,
    /// State nets that may change transiently during the burst (`X` in some A state).
    pub transient: Transient,
}

impl Burst {
    /// Nothing is known.
    pub(crate) fn unsafe_with(steps: u32, m: usize) -> Burst {
        Burst {
            safe: false,
            steps,
            transient: Transient::all(m),
        }
    }
}

/// When one changing kernel input of a timed burst may change (see [`Model::timed`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Arrival {
    /// First and last tick, relative to the start of the burst, at which the input may change.
    /// Before `min` it has its initial value, from `max` on its final value.
    pub min: u32,
    pub max: u32,
    /// The input changes at most once (one clean window): it keeps its value in entries where the
    /// final value equals the initial one.
    pub clean: bool,
    /// Fast round labels (relative to the burst's first round) of its first and last window.
    pub first_label: u16,
    pub last_label: u16,
}

/// What one state net does in a timed burst, over all feasible entries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TimedNet {
    /// It may change (finally or transiently) in some entry.
    pub active: bool,
    /// First and last tick (relative to the burst start) at which it may change.
    pub first: u32,
    pub last: u32,
    /// It may change more than once in some entry.
    pub multi: bool,
    /// Entry-derived literals common to the entries in which it may change.
    pub lit_mask: u32,
    pub lit_val: u32,
    /// Union of the inputs with activity in those entries.
    pub contrib: u32,
}

/// Result of a timed burst check (see [`Model::timed`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Timed {
    /// Every feasible entry ends, in Timing mode, in the state Fast mode computes.
    pub safe: bool,
    /// Last tick (relative to the burst start) at which an input or a state net may change.
    pub settle: u32,
    /// Per state net (meaningful when `safe`).
    pub nets: Vec<TimedNet>,
}

/// Largest number of inputs with a free intermediate value in one round of the Fast round chain
/// of a multi-round timed burst.
const MAX_CHAIN_FREE: u32 = 10;

/// What one state net (position in `Kernel::nets`) does in a round.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PosInfo {
    /// Some feasible entry changes it.
    pub changes: bool,
    /// Union of the changed-input sets of those entries.
    pub contrib: u32,
    /// Entry-derived literals common to all those entries: inputs (mask) and values.
    pub lit_mask: u32,
    pub lit_val: u32,
    /// Max latency of those entries.
    pub lat: u32,
}

/// Answer to a round query.
#[derive(Clone, Debug, Default)]
pub(crate) struct RoundResult {
    /// Number of feasible entries.
    pub entries: u64,
    /// Per state net.
    pub pos: Vec<PosInfo>,
    /// Max latency over all feasible entries.
    pub lat_all: u32,
    /// A feasible entry with ≥ 2 changing inputs is order-sensitive.
    pub order_sensitive: bool,
    /// A feasible entry with a changing input ends in a non-binary state (collapse to `X` or no
    /// settling within the bound).
    pub x_result: bool,
    /// State nets that may glitch (change more than once, possibly returning to their initial
    /// value) during a single exact transition (lazily; bit `p` = state net `p`, all ones if
    /// unknown).
    pub glitch: Option<Transient>,
    /// Computed with `glitch`: per state net, the first and last tick (after the input change) at
    /// which it changes in some feasible entry (`None` inside = never; `None` = unknown).
    pub span: Option<Spans>,
    /// Burst safety, settle bound and transient nets (lazily).
    pub burst: Option<Burst>,
    /// Some feasible transition may never settle (an oscillation with period > 2 that the table
    /// records as a non-binary result, or a kernel without a table whose settling could not be
    /// proven). Timing mode then keeps oscillating where Fast mode yields `X`.
    pub may_not_settle: bool,
    /// Feasible entries with a non-binary result `(s, u, v)`, one per `(s, v)` (checked for
    /// settling by [`Model::round`]); `x_over` = too many to check.
    x_entries: Vec<(u32, u32, u32)>,
    x_over: bool,
    /// Conservative answer (no enumeration happened).
    approx: bool,
}

/// Tunables of model queries.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ModelLimits {
    pub max_enum_work: u64,
    pub max_glitch_sims: usize,
    pub max_ternary: usize,
}

/// The behavior model of one kernel structure.
pub(crate) struct Model<'t> {
    pub table: Option<&'t KernelTable>,
    sub: Arc<FlatNetlist>,
    map: SubnetMap,
    charz: CharzOptions,
    /// Number of kernel inputs.
    pub k: usize,
    /// Number of state nets.
    pub m: usize,
    /// Shortest gate distance from input `i` to state net `p`: `dist[i * m + p]` ([`INF`] if none).
    pub dist: Vec<u32>,
    /// Settle bound used when nothing better is known.
    pub bound: u32,
    /// Max latency over the whole table (or the settle bound without a table).
    pub max_latency: u32,
    sim: Option<KernelSim>,
    /// Unit-delay simulation of `(s, v)`: nets changing ≥ 2 times, per-net change span.
    glitch_cache: HashMap<(u32, u32), Sim1>,
    /// Settling of non-binary table entries `(s, v)`: true = quiescent within the bound.
    settle_cache: HashMap<(u32, u32), bool>,
    /// Kernels without a table: settling of `(state, new inputs)` transitions.
    sim_settle_cache: HashMap<(Vec<Logic>, u32), bool>,
    /// Kernels without a table: a feedback cut (state positions) and the evaluation order of the
    /// other elements (lazily; `None` inside = could not be built).
    cut_plan: Option<Option<CutPlan>>,
    rounds: HashMap<Vec<u32>, RoundResult>,
    /// Timed burst checks by query key and arrivals (`None` = too much work).
    timed_cache: HashMap<Vec<u32>, Option<Timed>>,
}

impl<'t> Model<'t> {
    /// Model of a kernel sub-netlist. `table` must belong to that structure; `keeper` marks a
    /// Keeper kernel (zero-delay resolver).
    pub(crate) fn new(
        sub: FlatNetlist,
        map: SubnetMap,
        table: Option<&'t KernelTable>,
        charz: &CharzOptions,
    ) -> Model<'t> {
        let k = map.num_inputs();
        let m = map.num_state();
        let usable =
            table.filter(|t| k <= 24 && t.num_inputs == k && t.states.iter().all(|s| s.len() == m));
        let n = sub.num_nets as usize;
        // 0-1 BFS: gates cost 1, resolvers 0.
        let mut readers: Vec<Vec<(usize, u32)>> = vec![Vec::new(); n];
        for g in &sub.gates {
            for &i in &g.inputs {
                readers[i as usize].push((g.output as usize, 1));
            }
        }
        for r in &sub.resolvers {
            for &i in &r.drivers {
                readers[i as usize].push((r.output as usize, 0));
            }
        }
        let mut dist = vec![INF; k * m];
        for i in 0..k {
            let mut d = vec![INF; n];
            let mut dq = std::collections::VecDeque::new();
            d[i] = 0;
            dq.push_back(i);
            while let Some(x) = dq.pop_front() {
                for &(y, w) in &readers[x] {
                    let nd = d[x] + w;
                    if nd < d[y] {
                        d[y] = nd;
                        if w == 0 {
                            dq.push_front(y)
                        } else {
                            dq.push_back(y)
                        }
                    }
                }
            }
            for p in 0..m {
                dist[i * m + p] = d[k + p];
            }
        }
        let bound = if sub.gates.is_empty() {
            0
        } else {
            charz.bound_for(m).min(u64::from(u32::MAX / 4)) as u32
        };
        let max_latency = usable.map_or(bound, |t| {
            t.transitions.iter().map(|x| x.latency).max().unwrap_or(0)
        });
        Model {
            table: usable,
            sub: Arc::new(sub),
            map,
            charz: charz.clone(),
            k,
            m,
            dist,
            bound,
            max_latency,
            sim: None,
            glitch_cache: HashMap::new(),
            settle_cache: HashMap::new(),
            sim_settle_cache: HashMap::new(),
            cut_plan: None,
            rounds: HashMap::new(),
            timed_cache: HashMap::new(),
        }
    }

    /// True for a kernel without gates (a keeper resolver: zero delay, always settles).
    pub(crate) fn is_keeper(&self) -> bool {
        self.sub.gates.is_empty()
    }

    /// Answer a round query (cached). `need_glitch` / `need_burst` request the lazy facts.
    pub(crate) fn round(
        &mut self,
        sig: &RoundSig,
        lim: &ModelLimits,
        need_glitch: bool,
        need_burst: bool,
    ) -> RoundResult {
        if !self.rounds.contains_key(&sig.key) {
            let mut r = self.compute_base(sig, lim);
            r.may_not_settle = match self.table {
                Some(t) if r.approx => t.transitions.iter().any(|x| x.next.is_none()),
                Some(t) => {
                    let mut x = std::mem::take(&mut r.x_entries);
                    x.sort_unstable_by_key(|&(s, _, v)| (s, v));
                    x.dedup_by_key(|e| (e.0, e.2));
                    r.x_over || x.iter().any(|&(s, u, v)| !self.entry_settles(t, s, u, v))
                }
                None => sig.cmask != 0 && !self.sim_settles(sig, lim),
            };
            self.rounds.insert(sig.key.clone(), r);
        }
        if need_glitch && self.rounds[&sig.key].glitch.is_none() {
            let (g, span) = self.compute_glitch(sig, lim);
            let r = self.rounds.get_mut(&sig.key).expect("cached");
            r.glitch = Some(g);
            r.span = span;
        }
        if need_burst && self.rounds[&sig.key].burst.is_none() {
            let b = self.compute_burst(sig, lim);
            self.rounds.get_mut(&sig.key).expect("cached").burst = Some(b);
        }
        self.rounds[&sig.key].clone()
    }

    /// The kernel simulator (created on first use).
    fn kernel_sim(&mut self) -> &mut KernelSim {
        let (sub, map, charz) = (&self.sub, &self.map, &self.charz);
        self.sim
            .get_or_insert_with(|| KernelSim::from_subnet(sub.clone(), map.clone(), charz))
    }

    /// True if the table entry `(s, u → v)` (recorded as non-binary) becomes quiescent within the
    /// settle bound, i.e. it collapsed to `X` rather than oscillating.
    fn entry_settles(&mut self, t: &KernelTable, s: u32, u: u32, v: u32) -> bool {
        if let Some(&q) = self.settle_cache.get(&(s, v)) {
            return q;
        }
        let k = self.k;
        let (_, st) = self.kernel_sim().transition(
            &t.states[s as usize],
            &input_vector(u, k),
            &input_vector(v, k),
        );
        self.settle_cache.insert((s, v), st.quiescent);
        st.quiescent
    }

    /// A feedback cut and evaluation order for stable-state enumeration of a kernel without a
    /// table (greedy feedback vertex set; keepers are always cut).
    fn cut_plan(&mut self) -> Option<&CutPlan> {
        if self.cut_plan.is_none() {
            self.cut_plan = Some(CutPlan::build(&self.sub, self.k));
        }
        self.cut_plan.as_ref().and_then(Option::as_ref)
    }

    /// Kernel without a table: prove that every feasible transition of the round (from every
    /// binary stable state, enumerated through a feedback cut) settles within the bound. `false`
    /// if some transition oscillates or the enumeration is too large.
    ///
    /// A unit-delay transition depends only on the state and the new input vector (every gate
    /// reads the state nets and the new inputs at the tick they are applied), so the work is one
    /// simulation per distinct `(stable state, new inputs)`; the new input vectors of a state are
    /// over-approximated by `old inputs` (any it is stable under) with the changing bits free
    /// (respecting statically known final values).
    fn sim_settles(&mut self, sig: &RoundSig, lim: &ModelLimits) -> bool {
        let k = self.k;
        if self.sub.gates.is_empty() {
            return true; // a keeper: zero-delay resolver only
        }
        if k > 31 {
            return false;
        }
        let Some(plan) = self.cut_plan().cloned() else {
            return false;
        };
        let all = (1u32 << k) - 1;
        let free = all & !sig.u_fix_mask;
        if free.count_ones() as usize + plan.cut.len() > SIM_PROOF_BITS {
            return false;
        }
        let budget = lim.max_glitch_sims.saturating_mul(16);
        // Stable states and the non-changing part of the inputs they are stable under.
        let mut bases: HashMap<Vec<Logic>, Vec<u32>> = HashMap::new();
        let mut fsub = 0u32;
        loop {
            let u = (sig.u_fix_val & sig.u_fix_mask) | fsub;
            if sig.u_ok(u) {
                let iu = input_vector(u, k);
                for c in 0..(1u32 << plan.cut.len()) {
                    let state = plan.candidate(&self.sub, &iu, c, self.m);
                    if state.contains(&Logic::X) || !self.kernel_sim().is_fixed_point(&state, &iu) {
                        continue;
                    }
                    let b = bases.entry(state).or_default();
                    if !b.contains(&u) {
                        b.push(u);
                    }
                }
            }
            if fsub == free {
                break;
            }
            fsub = fsub.wrapping_sub(free) & free;
        }
        let per_base = 1usize << sig.cmask.count_ones();
        let mut work = 0usize;
        for us in bases.values_mut() {
            us.sort_unstable_by_key(|&u| u & !sig.cmask);
            us.dedup_by_key(|u| *u & !sig.cmask);
            work = work.saturating_add(us.len().saturating_mul(per_base));
        }
        if work > budget {
            return false;
        }
        let vmask = sig.v_fix_mask & sig.cmask;
        for (state, us) in &bases {
            for &u in us {
                let iu = input_vector(u, k);
                let mut vs = 0u32;
                loop {
                    let v = (u & !sig.cmask) | vs;
                    if v != u && v & vmask == sig.v_fix_val & vmask {
                        let key = (state.clone(), v);
                        let ok = match self.sim_settle_cache.get(&key) {
                            Some(&ok) => ok,
                            None => {
                                let (_, st) =
                                    self.kernel_sim()
                                        .transition(state, &iu, &input_vector(v, k));
                                self.sim_settle_cache.insert(key, st.quiescent);
                                st.quiescent
                            }
                        };
                        if !ok {
                            return false;
                        }
                    }
                    if vs == sig.cmask {
                        break;
                    }
                    vs = vs.wrapping_sub(sig.cmask) & sig.cmask;
                }
            }
        }
        true
    }

    fn conservative(&self, sig: &RoundSig) -> RoundResult {
        let changes = sig.cmask != 0;
        RoundResult {
            entries: u64::from(changes),
            pos: vec![
                PosInfo {
                    changes,
                    contrib: sig.cmask,
                    lit_mask: 0,
                    lit_val: 0,
                    lat: self.max_latency
                };
                self.m
            ],
            lat_all: self.max_latency,
            order_sensitive: false,
            x_result: false,
            glitch: Some(Transient::all(self.m)),
            burst: Some(Burst::unsafe_with(self.max_latency, self.m)),
            approx: true,
            ..RoundResult::default()
        }
    }

    /// Visit every feasible entry `(s, u, v, transition)`.
    fn for_each_entry(
        t: &KernelTable,
        sig: &RoundSig,
        mut f: impl FnMut(u32, u32, u32, &Transition),
    ) {
        let k = t.num_inputs;
        for u in 0..(1u32 << k) {
            if !sig.u_ok(u) {
                continue;
            }
            let states = &t.states_by_input[u as usize];
            if states.is_empty() {
                continue;
            }
            let mut sub = 0u32;
            loop {
                let v = (u & !sig.cmask) | sub;
                if sig.v_ok(u, v) {
                    for &s in states {
                        f(s, u, v, t.transition(s, v));
                    }
                }
                if sub == sig.cmask {
                    break;
                }
                sub = sub.wrapping_sub(sig.cmask) & sig.cmask;
            }
        }
    }

    fn compute_base(&self, sig: &RoundSig, lim: &ModelLimits) -> RoundResult {
        let Some(t) = self.table else {
            return self.conservative(sig);
        };
        let k = self.k;
        let mut work: u64 = 0;
        for u in 0..(1u32 << k) {
            if sig.u_ok(u) {
                work += t.states_by_input[u as usize].len() as u64;
            }
        }
        work = work.saturating_mul(1u64 << sig.cmask.count_ones());
        if work > lim.max_enum_work {
            return self.conservative(sig);
        }
        let m = self.m;
        let mut res = RoundResult {
            pos: vec![PosInfo::default(); m],
            ..RoundResult::default()
        };
        let mut seen = vec![false; m];
        Self::for_each_entry(t, sig, |s, u, v, tr| {
            res.entries += 1;
            res.lat_all = res.lat_all.max(tr.latency);
            let d = u ^ v;
            if d.count_ones() >= 2 && tr.order_sensitive {
                res.order_sensitive = true;
            }
            if d != 0 && tr.next.is_none() {
                res.x_result = true;
                if res.x_entries.len() <= lim.max_glitch_sims {
                    res.x_entries.push((s, u, v));
                } else {
                    res.x_over = true;
                }
            }
            let sv = &t.states[s as usize];
            let e_mask = sig.v_lit_mask;
            let e_val = v & e_mask;
            for p in 0..m {
                let changed = match tr.next {
                    None => true,
                    Some(n) => t.states[n as usize][p] != sv[p],
                };
                if !changed {
                    continue;
                }
                let pi = &mut res.pos[p];
                pi.changes = true;
                pi.contrib |= d;
                pi.lat = pi.lat.max(tr.latency);
                if seen[p] {
                    pi.lit_mask &= e_mask & !(pi.lit_val ^ e_val);
                    pi.lit_val &= pi.lit_mask;
                } else {
                    seen[p] = true;
                    pi.lit_mask = e_mask;
                    pi.lit_val = e_val;
                }
            }
        });
        res
    }

    /// Glitching state nets of the feasible entries (unit-delay simulation of each), and the span
    /// of ticks at which each state net changes (`None` if not simulated).
    fn compute_glitch(&mut self, sig: &RoundSig, lim: &ModelLimits) -> (Transient, Option<Spans>) {
        let m = self.m;
        let Some(t) = self.table else {
            return (Transient::all(m), None);
        };
        let mut todo: Vec<(u32, u32, u32)> = Vec::new();
        let mut over = false;
        Self::for_each_entry(t, sig, |s, u, v, tr| {
            // Also entries that return to the same state: an output may pulse on the way.
            if over || u == v {
                return;
            }
            if tr.next.is_none() {
                over = true; // X result: the outputs are X anyway; treat as glitchy
                return;
            }
            todo.push((s, u, v));
            if todo.len() > lim.max_glitch_sims * 4 {
                over = true;
            }
        });
        if over {
            return (Transient::all(m), None);
        }
        todo.sort_unstable_by_key(|&(s, _, v)| (s, v));
        todo.dedup_by_key(|e| (e.0, e.2));
        let fresh = todo
            .iter()
            .filter(|&&(s, _, v)| !self.glitch_cache.contains_key(&(s, v)))
            .count();
        if fresh > lim.max_glitch_sims {
            return (Transient::all(m), None);
        }
        let mut tr = Transient::none(m);
        let mut span: Option<Spans> = Some(vec![None; m]);
        for (s, u, v) in todo {
            if !self.glitch_cache.contains_key(&(s, v)) {
                let g = self.simulate_glitches(t, s, u, v);
                self.glitch_cache.insert((s, v), g);
            }
            let g = &self.glitch_cache[&(s, v)];
            let mut mask = g.multi;
            while mask != 0 {
                let p = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                tr.add(p, sig.v_lit_mask, v & sig.v_lit_mask);
            }
            match (&mut span, &g.span) {
                (Some(acc), Some(sp)) => {
                    for (a, &x) in acc.iter_mut().zip(sp.iter()) {
                        if let Some((f, l)) = x {
                            *a = Some(a.map_or((f, l), |(af, al)| (af.min(f), al.max(l))));
                        }
                    }
                }
                _ => span = None,
            }
        }
        (tr, span)
    }

    /// Unit-delay simulation of entry `(s, u → v)`: mask of state nets that change ≥ 2 times
    /// (a glitch, or a pulse back to the initial value), and per state net the first and last tick
    /// (after the input change) at which it changes.
    fn simulate_glitches(&mut self, t: &KernelTable, s: u32, u: u32, v: u32) -> Sim1 {
        let k = self.k;
        let m = self.m;
        let (sub, map, charz) = (&self.sub, &self.map, &self.charz);
        let ks = self
            .sim
            .get_or_insert_with(|| KernelSim::from_subnet(sub.clone(), map.clone(), charz));
        let bound = ks.default_bound();
        ks.load_state(&t.states[s as usize], &input_vector(u, k));
        ks.apply_inputs(&input_vector(v, k));
        let sim = ks.sim_mut();
        // The inputs change at the next tick.
        let start = sim.tick() + 1;
        let mut counts = vec![0u32; m];
        let mut span: Vec<Option<(u32, u32)>> = vec![None; m];
        let mut quiet = false;
        for _ in 0..bound {
            if sim.is_quiescent() {
                quiet = true;
                break;
            }
            sim.step();
            let off = sim.tick().saturating_sub(start).min(u64::from(u32::MAX)) as u32;
            for &n in sim.last_changed() {
                if n as usize >= k {
                    let p = n as usize - k;
                    counts[p] += 1;
                    span[p] = Some(span[p].map_or((off, off), |(f, _)| (f, off)));
                }
            }
        }
        sim.take_diagnostics();
        if !quiet && !sim.is_quiescent() || m > 64 {
            return Sim1 {
                multi: u64::MAX,
                span: None,
            };
        }
        Sim1 {
            multi: counts
                .iter()
                .enumerate()
                .filter(|&(_, &c)| c >= 2)
                .fold(0u64, |acc, (p, _)| acc | 1 << p),
            span: Some(span),
        }
    }

    fn compute_burst(&self, sig: &RoundSig, lim: &ModelLimits) -> Burst {
        let m = self.m;
        let Some(t) = self.table else {
            return Burst::unsafe_with(self.max_latency, m);
        };
        let mut transient = Transient::none(m);
        // State nets that are X in each A state.
        let mut a_x: Vec<Vec<usize>> = Vec::new();
        let k = self.k;
        let cmask = sig.cmask;
        let max_steps = 4 * self.sub.num_nets + 8;
        let mut a_ids: HashMap<(u32, u32, u32), usize> = HashMap::new();
        let mut a_index: HashMap<Vec<Logic>, usize> = HashMap::new();
        let mut a_states: Vec<Vec<Logic>> = Vec::new();
        // (A-state id, new inputs) → B result (state, steps) or None (not binary / not stable).
        let mut b_cache: HashMap<(usize, u32), BResult> = HashMap::new();
        let mut safe = true;
        let mut steps_max = 0u32;
        let mut buf = Vec::new();
        let sub = &*self.sub;
        Self::for_each_entry(t, sig, |s, u, v, tr| {
            if !safe {
                return;
            }
            let Some(next) = tr.next else {
                safe = false;
                return;
            };
            // Inputs that may change in this entry (even glitch-only): those that do change, and
            // every other changing input whose path condition can hold. An input whose
            // condition fails has no activity at all and keeps its value.
            let xs = (u ^ v) | (sig.restr_sat(v) & cmask);
            let akey = (s, u & !xs, xs);
            let aid = *a_ids.entry(akey).or_insert_with(|| {
                let mut vals = vec![Logic::X; sub.num_nets as usize];
                for (i, val) in vals.iter_mut().enumerate().take(k) {
                    if xs >> i & 1 == 0 {
                        *val = Logic::from_bool(u >> i & 1 == 1);
                    }
                }
                vals[k..].copy_from_slice(&t.states[s as usize]);
                ternary_a(sub, &mut vals, &mut buf);
                let n = a_states.len();
                *a_index.entry(vals.clone()).or_insert_with(|| {
                    a_x.push((0..m).filter(|&p| vals[k + p] == Logic::X).collect());
                    a_states.push(vals);
                    n
                })
            });
            for &p in &a_x[aid] {
                transient.add(p, sig.v_lit_mask, v & sig.v_lit_mask);
            }
            if b_cache.len() > lim.max_ternary {
                safe = false;
                return;
            }
            let b = b_cache.entry((aid, v)).or_insert_with(|| {
                let mut vals = a_states[aid].clone();
                for (i, val) in vals.iter_mut().enumerate().take(k) {
                    *val = Logic::from_bool(v >> i & 1 == 1);
                }
                ternary_b(sub, &mut vals, max_steps, &mut buf).map(|st| (vals, st))
            });
            match b {
                Some((vals, st)) if vals[k..] == t.states[next as usize][..] => {
                    steps_max = steps_max.max(*st);
                }
                _ => safe = false,
            }
        });
        if !safe {
            return Burst::unsafe_with(steps_max, m);
        }
        Burst {
            safe,
            steps: steps_max,
            transient,
        }
    }

    /// Timed burst check (cached). Every feasible entry `(s, u → v)` is simulated from the stable
    /// state `s` with Timing mode's unit delays in ternary logic, where a changing input `j` has
    /// the value `u_j` before `arr[j].min`, `X` from `arr[j].min` to `arr[j].max - 1` (unless it
    /// cannot change in that entry: a clean input whose final value equals its initial one, or
    /// one whose path condition fails) and `v_j` from `arr[j].max` on. The simulation covers
    /// every Timing-mode behavior compatible with the arrival windows (ternary simulation is
    /// monotone). It must end in a binary stable state equal to Fast mode's result: the table
    /// transition `T(s, v)` for a one-round burst (`stages == 1`), otherwise the chain of table
    /// transitions of rounds `0..stages`, where an input changing in several rounds takes every
    /// binary value between its first and last round. Period-2 collapse is accounted for: nets
    /// that could be collapsed (a possibly self-sustained alternation of `2 × p2_limit` ticks) are
    /// set to `X` in the simulation.
    ///
    /// `arr[j]` is `Some` exactly for the changing inputs (`sig.cmask`). `None` if the kernel has
    /// no table or the work exceeds the limits (per query, and the caller's `budget` of element
    /// evaluations, which is reduced by the work done).
    pub(crate) fn timed(
        &mut self,
        sig: &RoundSig,
        arr: &[Option<Arrival>],
        stages: u16,
        lim: &ModelLimits,
        budget: &mut u64,
    ) -> Option<Timed> {
        let mut key = sig.key.clone();
        key.push(u32::from(stages));
        for a in arr {
            match a {
                None => key.push(u32::MAX),
                Some(a) => key.extend([
                    a.min,
                    a.max,
                    u32::from(a.clean),
                    u32::from(a.first_label) << 16 | u32::from(a.last_label),
                ]),
            }
        }
        if let Some(r) = self.timed_cache.get(&key) {
            return r.clone();
        }
        let r = self.compute_timed(sig, arr, stages, lim, budget);
        self.timed_cache.insert(key, r.clone());
        r
    }

    fn compute_timed(
        &self,
        sig: &RoundSig,
        arr: &[Option<Arrival>],
        stages: u16,
        lim: &ModelLimits,
        total: &mut u64,
    ) -> Option<Timed> {
        let t = self.table?;
        let k = self.k;
        let m = self.m;
        if stages == 0 || stages > 32 || arr.len() != k {
            return None;
        }
        let mut work: u64 = 0;
        for u in 0..(1u32 << k) {
            if sig.u_ok(u) {
                work += t.states_by_input[u as usize].len() as u64;
            }
        }
        work = work.saturating_mul(1u64 << sig.cmask.count_ones());
        let t_in = arr.iter().flatten().map(|a| a.max).max().unwrap_or(0);
        let max_steps = self.bound.max(4 * self.sub.num_nets + 8);
        if work > lim.max_glitch_sims as u64 * 4 {
            return None;
        }
        let start = TIMED_MAX_EVALS.min(*total);
        let mut budget = start;
        let mut over = false;
        let l2 = 2 * self.charz.timing.p2_limit;
        let sim = TimedSim::new(&self.sub, k, m);
        let mut res = Timed {
            safe: true,
            settle: t_in,
            nets: vec![TimedNet::default(); m],
        };
        Self::for_each_entry(t, sig, |s, u, v, tr| {
            if !res.safe {
                return;
            }
            let fast = if stages == 1 {
                tr.next
            } else {
                match fast_chain(t, arr, stages, s, u, v, &mut budget) {
                    Ok(f) => f,
                    Err(()) => {
                        over = true;
                        None
                    }
                }
            };
            let Some(fast) = fast else {
                res.safe = false;
                return;
            };
            // Inputs with activity in this entry.
            let rs = sig.restr_sat(v);
            let mut act = 0u32;
            for (j, a) in arr.iter().enumerate() {
                if let Some(a) = a
                    && ((u ^ v) >> j & 1 == 1 || (!a.clean && rs >> j & 1 == 1))
                {
                    act |= 1 << j;
                }
            }
            let input_at = |j: usize, tick: u32| -> Logic {
                let bit = |x: u32| Logic::from_bool(x >> j & 1 == 1);
                match arr[j] {
                    Some(a) if act >> j & 1 == 1 => {
                        if tick < a.min {
                            bit(u)
                        } else if tick >= a.max {
                            bit(v)
                        } else {
                            Logic::X
                        }
                    }
                    _ => bit(u),
                }
            };
            let mut bounds: Vec<u32> = arr
                .iter()
                .enumerate()
                .filter(|&(j, _)| act >> j & 1 == 1)
                .flat_map(|(_, a)| a.map(|a| [a.min, a.max]).unwrap_or_default())
                .collect();
            bounds.sort_unstable();
            bounds.dedup();
            let tr = match sim.run(
                &t.states[s as usize],
                u,
                &input_at,
                &bounds,
                t_in,
                max_steps,
                l2,
                &mut budget,
            ) {
                Ok(Some(tr)) => tr,
                Ok(None) => {
                    res.safe = false;
                    return;
                }
                Err(()) => {
                    over = true;
                    res.safe = false;
                    return;
                }
            };
            if tr.fin[..] != t.states[fast as usize][..] {
                res.safe = false;
                return;
            }
            res.settle = res.settle.max(tr.settle);
            let e_mask = sig.v_lit_mask;
            let e_val = v & e_mask;
            for (p, tn) in res.nets.iter_mut().enumerate() {
                let Some((first, last)) = tr.span[p] else {
                    continue;
                };
                if tn.active {
                    tn.first = tn.first.min(first);
                    tn.last = tn.last.max(last);
                    tn.lit_mask &= e_mask & !(tn.lit_val ^ e_val);
                    tn.lit_val &= tn.lit_mask;
                    tn.contrib |= act;
                } else {
                    *tn = TimedNet {
                        active: true,
                        first,
                        last,
                        multi: false,
                        lit_mask: e_mask,
                        lit_val: e_val,
                        contrib: act,
                    };
                }
                tn.multi |= tr.multi[p];
            }
        });
        *total -= start - budget;
        (!over).then_some(res)
    }
}

/// Work limit of one timed burst check, in element evaluations.
const TIMED_MAX_EVALS: u64 = 1 << 21;

/// The Fast-mode result of a multi-round burst for entry `(s, u → v)`: the chain of table
/// transitions of rounds `0..stages`. After round `l`, a changing input has its initial value
/// before its first round, its final value from its last round on, and any value in between.
/// The reachable `(state, input vector)` pairs are propagated round by round; all must end in the
/// same state. `Ok(None)` if a transition is non-binary, the results differ, or a round has too
/// many free inputs; `Err` if `budget` (table lookups) runs out.
fn fast_chain(
    t: &KernelTable,
    arr: &[Option<Arrival>],
    stages: u16,
    s: u32,
    u: u32,
    v: u32,
    budget: &mut u64,
) -> Result<Option<u32>, ()> {
    let mut set: Vec<(u32, u32)> = vec![(s, u)];
    for l in 0..stages {
        // Inputs fixed after round `l`, and the free ones.
        let mut base = u;
        let mut free: Vec<u32> = Vec::new();
        for (j, a) in arr.iter().enumerate() {
            let Some(a) = a else { continue };
            let bit = 1u32 << j;
            if l >= a.last_label {
                base = (base & !bit) | (v & bit);
            } else if l >= a.first_label {
                base &= !bit;
                free.push(bit);
            }
        }
        if free.len() as u32 > MAX_CHAIN_FREE {
            return Ok(None);
        }
        let mut next: Vec<(u32, u32)> = Vec::with_capacity(set.len() << free.len());
        for &(state, prev) in &set {
            for assign in 0..(1u32 << free.len()) {
                let mut vec = base;
                for (i, &bit) in free.iter().enumerate() {
                    if assign >> i & 1 == 1 {
                        vec |= bit;
                    }
                }
                let st = if vec == prev {
                    state
                } else {
                    *budget = budget.checked_sub(1).ok_or(())?;
                    match t.transition(state, vec).next {
                        Some(n) => n,
                        None => return Ok(None),
                    }
                };
                next.push((st, vec));
            }
        }
        next.sort_unstable();
        next.dedup();
        set = next;
    }
    let mut result: Option<u32> = None;
    for &(state, prev) in &set {
        let fin = if prev == v {
            state
        } else {
            match t.transition(state, v).next {
                Some(n) => n,
                None => return Ok(None),
            }
        };
        match result {
            None => result = Some(fin),
            Some(r) if r != fin => return Ok(None),
            Some(_) => {}
        }
    }
    Ok(result)
}

/// Trace of one timed simulation.
struct TimedTrace {
    /// Final (binary, stable) state.
    fin: Vec<Logic>,
    /// Per state net: first and last tick at which it may change (`None` = never).
    span: Vec<Option<(u32, u32)>>,
    /// Per state net: it may change more than once.
    multi: Vec<bool>,
    /// Last tick at which an input or a state net may change.
    settle: u32,
}

/// Unit-delay ternary simulation of a kernel sub-netlist with timed inputs.
struct TimedSim<'s> {
    sub: &'s FlatNetlist,
    k: usize,
    m: usize,
    /// Per state net: the nets its driver reads.
    drv_in: Vec<Vec<usize>>,
    /// Per state net: its driver can output `Z` (a tri-state gate or an unterminated resolver).
    may_z: Vec<bool>,
}

/// Index of a definite value for the change-count DP (`X` has none).
fn val_idx(v: Logic) -> Option<usize> {
    match v {
        Logic::Zero => Some(0),
        Logic::One => Some(1),
        Logic::Z => Some(2),
        Logic::X => None,
    }
}

impl<'s> TimedSim<'s> {
    fn new(sub: &'s FlatNetlist, k: usize, m: usize) -> TimedSim<'s> {
        let mut drv_in = vec![Vec::new(); m];
        let mut may_z = vec![false; m];
        for g in &sub.gates {
            let o = g.output as usize;
            if o >= k {
                drv_in[o - k] = g.inputs.iter().map(|&i| i as usize).collect();
                may_z[o - k] = g.op == panda_netlist::flat::FlatOp::Tri;
            }
        }
        for r in &sub.resolvers {
            let o = r.output as usize;
            if o >= k {
                drv_in[o - k] = r.drivers.iter().map(|&i| i as usize).collect();
                may_z[o - k] = r.termination == panda_netlist::Termination::None;
            }
        }
        TimedSim {
            sub,
            k,
            m,
            drv_in,
            may_z,
        }
    }

    /// Resolvers of tick `t` (zero delay) from the tick-`t` driver values in `next`, keepers
    /// holding their tick-`t - 1` value from `prev`.
    fn resolvers(&self, next: &mut [Logic], prev: &[Logic], buf: &mut Vec<Logic>) {
        for _ in 0..=self.sub.resolvers.len() {
            let mut ch = false;
            for r in &self.sub.resolvers {
                buf.clear();
                buf.extend(r.drivers.iter().map(|&n| next[n as usize]));
                let o = r.output as usize;
                let nv = resolve(buf, r.termination, prev[o]).value;
                if next[o] != nv {
                    next[o] = nv;
                    ch = true;
                }
            }
            if !ch {
                break;
            }
        }
    }

    /// Simulate from stable state `state` (stable under the initial input vector `init`) until
    /// stable with the final inputs (`input_at(j, t)`: value of input `j` at tick `t`, final for
    /// `t ≥ t_in`; `bounds`: the sorted ticks at which an input may change value). `None` if the
    /// result is not binary or not stable within `t_in + max_steps` (a possible period-2 collapse,
    /// `l2` = window, 0 = none, sets the nets concerned to `X`); `Err` if the evaluation `budget` ran out. Ticks at which nothing can change (a fixed point
    /// with unchanged inputs) cost no gate evaluations.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        state: &[Logic],
        init: u32,
        input_at: &dyn Fn(usize, u32) -> Logic,
        bounds: &[u32],
        t_in: u32,
        max_steps: u32,
        l2: u32,
        budget: &mut u64,
    ) -> Result<Option<TimedTrace>, ()> {
        let (k, m) = (self.k, self.m);
        let n = self.sub.num_nets as usize;
        let mut buf = Vec::new();
        let mut span: Vec<Option<(u32, u32)>> = vec![None; m];
        // Change-count DP per state net (max changes, capped at 2) and possible-alternation runs.
        const NEG: i8 = i8::MIN / 2;
        let mut dp: Vec<[i8; 3]> = state
            .iter()
            .map(|&v| {
                let mut d = [NEG; 3];
                if let Some(i) = val_idx(v) {
                    d[i] = 0;
                }
                d
            })
            .collect();
        let mut alt = vec![0u32; m];
        // Last tick of a definite change (two different definite values) per net.
        let mut last_def = vec![None::<u32>; n];
        let mut hist2: Vec<Logic> = state.to_vec(); // state at tick t-2
        let mut hist1: Vec<Logic> = state.to_vec(); // state at tick t-1
        let mut settle = t_in;
        let mut tick: u32 = 0;
        // Tick -1: the stable state with the initial inputs. Tick 0: inputs of tick 0, gate
        // outputs unchanged (they were stable), resolvers recomputed.
        let mut base = vec![Logic::X; n];
        for (j, b) in base.iter_mut().enumerate().take(k) {
            *b = Logic::from_bool(init >> j & 1 == 1);
        }
        base[k..].copy_from_slice(state);
        let mut prev_in: Vec<Logic> = base[..k].to_vec();
        let mut cur = base.clone();
        for (j, c) in cur.iter_mut().enumerate().take(k) {
            *c = input_at(j, 0);
        }
        self.resolvers(&mut cur, &base, &mut buf);
        let mut next = cur.clone();
        // `next == cur` after the last evaluation: with unchanged inputs nothing changes.
        let mut fixed = false;
        let cost = (self.sub.gates.len() + self.sub.resolvers.len() + n + 1) as u64;
        loop {
            // Record tick `tick` (values in `cur`, predecessors in `prev_in` / `hist1`).
            for j in 0..k {
                let (a, b) = (prev_in[j], cur[j]);
                if a != Logic::X && b != Logic::X && a != b {
                    last_def[j] = Some(tick);
                }
                prev_in[j] = b;
            }
            // Alternation runs and definite changes (from the values before a collapse).
            for p in 0..m {
                let (a, b) = (hist1[p], cur[k + p]);
                if a == Logic::X || b == Logic::X || a != b {
                    if a != Logic::X && b != Logic::X {
                        last_def[k + p] = Some(tick);
                    }
                    let compat2 = |x: Logic, y: Logic| x == Logic::X || y == Logic::X || x == y;
                    alt[p] = if alt[p] >= 1 && compat2(b, hist2[p]) {
                        alt[p] + 1
                    } else {
                        1
                    };
                } else {
                    alt[p] = 0;
                }
            }
            // Period-2 collapse: candidates with an alternation-compatible run of `l2` ticks,
            // reduced to a self-sustained set. A candidate is dropped only if a driver input
            // outside the set *definitely* changed in the window or none is in the set, so the set
            // contains every set Timing mode could collapse here. Its binary-valued nets are set to `X`, which
            // covers both outcomes (collapsed or not); their runs restart as in Timing mode.
            if l2 > 0 && alt.iter().any(|&a| a >= l2) {
                let lo = (tick + 1).saturating_sub(l2);
                let quiet = |x: usize| last_def[x].is_none_or(|t| t < lo);
                let mut inset: Vec<bool> = alt.iter().map(|&a| a >= l2).collect();
                loop {
                    let mut ch = false;
                    for p in 0..m {
                        let ins = |x: usize| x >= k && inset[x - k];
                        // Dropped: a driver input outside the set definitely changed, or no driver
                        // input is in the set (with its other inputs constant, the net is constant:
                        // Timing mode's set has a driver in the set for every net).
                        if inset[p]
                            && (self.drv_in[p].iter().any(|&x| !(quiet(x) || ins(x)))
                                || !self.drv_in[p].iter().any(|&x| ins(x)))
                        {
                            inset[p] = false;
                            ch = true;
                        }
                    }
                    if !ch {
                        break;
                    }
                }
                let mut forced = false;
                for p in 0..m {
                    if inset[p] && cur[k + p] != Logic::X {
                        cur[k + p] = Logic::X;
                        alt[p] = 1;
                        forced = true;
                    }
                }
                if forced {
                    // Resolvers see the collapsed values at the same tick (keepers hold their
                    // tick-(t-1) value).
                    let mut prev_t = cur.clone();
                    prev_t[k..].copy_from_slice(&hist1);
                    self.resolvers(&mut cur, &prev_t, &mut buf);
                    fixed = false;
                }
            }
            // Possible changes at this tick (final values) and the change-count DP.
            for p in 0..m {
                let (a, b) = (hist1[p], cur[k + p]);
                if a == Logic::X || b == Logic::X || a != b {
                    span[p] = Some(span[p].map_or((tick, tick), |(f, _)| (f, tick)));
                    settle = settle.max(tick);
                }
                let d = dp[p];
                let mut nd = [NEG; 3];
                for (c, slot) in nd.iter_mut().enumerate() {
                    // `X` is `0` or `1`, or also `Z` at a net whose driver can output it.
                    let allowed = match val_idx(b) {
                        Some(i) => i == c,
                        None => c < 2 || self.may_z[p],
                    };
                    if allowed {
                        let mut best = d[c];
                        for (bb, &x) in d.iter().enumerate() {
                            if bb != c {
                                best = best.max(x.saturating_add(1));
                            }
                        }
                        *slot = best.min(2);
                    }
                }
                dp[p] = nd;
            }
            // Next tick.
            let mut same_inputs = (0..k).all(|j| input_at(j, tick + 1) == cur[j]);
            if fixed && same_inputs && tick < t_in {
                // Nothing changes until the next input boundary: record the idle ticks
                // `tick + 1 ..= target - 1` in bulk (only `X` nets may change there, and never
                // definitely).
                let target = bounds
                    .iter()
                    .copied()
                    .find(|&b| b > tick + 1)
                    .unwrap_or(t_in)
                    .min(t_in);
                if target > tick + 1 {
                    let (from, to) = (tick + 1, target - 1);
                    let d = to - from + 1;
                    for p in 0..m {
                        if cur[k + p] != Logic::X {
                            alt[p] = 0;
                            continue;
                        }
                        span[p] = Some(span[p].map_or((from, to), |(f, _)| (f, to)));
                        settle = settle.max(to);
                        alt[p] = alt[p].saturating_add(d);
                        for _ in 0..d.min(2) {
                            let x = dp[p][0].max(dp[p][1]).max(dp[p][2]);
                            let y = x.saturating_add(1).min(2);
                            dp[p] = [y, y, if self.may_z[p] { y } else { NEG }];
                        }
                    }
                    hist1.copy_from_slice(&cur[k..]);
                    hist2.copy_from_slice(&cur[k..]);
                    tick = to;
                    same_inputs = (0..k).all(|j| input_at(j, tick + 1) == cur[j]);
                }
            }
            if !(fixed && same_inputs) {
                *budget = budget.checked_sub(cost).ok_or(())?;
                next.copy_from_slice(&cur);
                for g in &self.sub.gates {
                    buf.clear();
                    buf.extend(g.inputs.iter().map(|&i| cur[i as usize]));
                    next[g.output as usize] = eval_gate(g.op, &buf);
                }
                for (j, x) in next.iter_mut().enumerate().take(k) {
                    *x = input_at(j, tick + 1);
                }
                self.resolvers(&mut next, &cur, &mut buf);
                fixed = next == cur;
            }
            if tick >= t_in && fixed {
                break;
            }
            if tick > t_in.saturating_add(max_steps) {
                return Ok(None);
            }
            hist2.copy_from_slice(&hist1);
            hist1.copy_from_slice(&cur[k..]);
            std::mem::swap(&mut cur, &mut next);
            tick += 1;
        }
        let fin = cur[k..].to_vec();
        if fin.contains(&Logic::X) {
            return Ok(None);
        }
        let multi = dp
            .iter()
            .map(|d| d.iter().copied().max().unwrap_or(0) >= 2)
            .collect();
        Ok(Some(TimedTrace {
            fin,
            span,
            multi,
            settle,
        }))
    }
}

/// Per state net: first and last change tick (`None` = never).
pub(crate) type Spans = Vec<Option<(u32, u32)>>;

/// Result of one unit-delay entry simulation (see [`Model::simulate_glitches`]).
#[derive(Clone, Debug)]
struct Sim1 {
    /// State nets changing ≥ 2 times (all ones = unknown).
    multi: u64,
    /// Per state net: first and last change tick after the input change (`None` = unknown).
    span: Option<Spans>,
}

/// Largest `free input bits + cut bits` enumerated to prove that a kernel without a table settles.
const SIM_PROOF_BITS: usize = 16;

/// Element of a kernel sub-netlist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Elem {
    Gate(usize),
    Res(usize),
}

/// Stable-state enumeration plan of a kernel sub-netlist: with the `cut` nets fixed, the other
/// state nets follow from the inputs in `order` (zero-delay evaluation).
#[derive(Clone, Debug)]
struct CutPlan {
    /// Cut state positions (index into the state vector).
    cut: Vec<usize>,
    /// Non-cut elements in topological order.
    order: Vec<Elem>,
}

impl CutPlan {
    /// Greedy feedback vertex set (keepers always in the cut). `None` if a state net has no
    /// driver in the sub-netlist.
    fn build(sub: &FlatNetlist, k: usize) -> Option<CutPlan> {
        let n = sub.num_nets as usize;
        let m = n - k;
        let mut driver: Vec<Option<Elem>> = vec![None; n];
        for (i, g) in sub.gates.iter().enumerate() {
            driver[g.output as usize] = Some(Elem::Gate(i));
        }
        for (i, r) in sub.resolvers.iter().enumerate() {
            driver[r.output as usize] = Some(Elem::Res(i));
        }
        if driver[k..].iter().any(Option::is_none) {
            return None;
        }
        let ins = |e: Elem| -> &[u32] {
            match e {
                Elem::Gate(i) => &sub.gates[i].inputs,
                Elem::Res(i) => &sub.resolvers[i].drivers,
            }
        };
        // Node p = state position p (its driver element).
        let elem = |p: usize| driver[k + p].expect("driven");
        let mut succ: Vec<Vec<usize>> = vec![Vec::new(); m];
        let mut indeg = vec![0usize; m];
        #[allow(clippy::needless_range_loop)] // `p` indexes both `indeg` and the driver table
        for p in 0..m {
            let mut srcs: Vec<usize> = ins(elem(p))
                .iter()
                .filter(|&&x| x as usize >= k)
                .map(|&x| x as usize - k)
                .collect();
            srcs.sort_unstable();
            srcs.dedup();
            indeg[p] = srcs.len();
            for q in srcs {
                succ[q].push(p);
            }
        }
        let mut done = vec![false; m];
        let mut cut = Vec::new();
        let mut order = Vec::with_capacity(m);
        let remove =
            |p: usize, indeg: &mut Vec<usize>, done: &mut Vec<bool>, ready: &mut Vec<usize>| {
                done[p] = true;
                for &s in &succ[p] {
                    indeg[s] -= 1;
                    if indeg[s] == 0 && !done[s] {
                        ready.push(s);
                    }
                }
            };
        let mut ready: Vec<usize> = Vec::new();
        for p in 0..m {
            let keeper = matches!(elem(p), Elem::Res(i)
                if sub.resolvers[i].termination == panda_netlist::Termination::Keeper);
            if keeper {
                cut.push(p);
                remove(p, &mut indeg, &mut done, &mut ready);
            }
        }
        ready.extend((0..m).filter(|&p| !done[p] && indeg[p] == 0));
        let mut left = m - cut.len();
        while left > 0 {
            if let Some(p) = ready.pop() {
                if done[p] {
                    continue;
                }
                order.push(elem(p));
                remove(p, &mut indeg, &mut done, &mut ready);
            } else {
                // Stuck on cycles: cut the remaining node with the most remaining edges.
                let p = (0..m)
                    .filter(|&p| !done[p])
                    .max_by_key(|&p| indeg[p] + succ[p].iter().filter(|&&s| !done[s]).count())
                    .expect("nodes left");
                cut.push(p);
                remove(p, &mut indeg, &mut done, &mut ready);
            }
            left -= 1;
        }
        cut.sort_unstable();
        Some(CutPlan { cut, order })
    }

    /// The candidate state for inputs `inputs` and cut assignment `c` (bit `i` = `cut[i]`).
    fn candidate(&self, sub: &FlatNetlist, inputs: &[Logic], c: u32, m: usize) -> Vec<Logic> {
        let k = inputs.len();
        let mut vals = vec![Logic::X; k + m];
        vals[..k].copy_from_slice(inputs);
        for (i, &p) in self.cut.iter().enumerate() {
            vals[k + p] = Logic::from_bool(c >> i & 1 == 1);
        }
        let mut buf = Vec::new();
        for &e in &self.order {
            let (out, v) = match e {
                Elem::Gate(i) => {
                    let g = &sub.gates[i];
                    buf.clear();
                    buf.extend(g.inputs.iter().map(|&x| vals[x as usize]));
                    (g.output, eval_gate(g.op, &buf))
                }
                Elem::Res(i) => {
                    let r = &sub.resolvers[i];
                    buf.clear();
                    buf.extend(r.drivers.iter().map(|&x| vals[x as usize]));
                    (r.output, resolve(&buf, r.termination, Logic::X).value)
                }
            };
            vals[out as usize] = v;
        }
        vals.split_off(k)
    }
}

/// Result of algorithm B: the final state and the number of changing steps, if it is binary.
type BResult = Option<(Vec<Logic>, u32)>;

/// Eichelberger algorithm A: raise every net whose driver disagrees with it to `X`, to a fixpoint.
fn ternary_a(sub: &FlatNetlist, vals: &mut [Logic], buf: &mut Vec<Logic>) {
    loop {
        let mut changed = false;
        for g in &sub.gates {
            buf.clear();
            buf.extend(g.inputs.iter().map(|&n| vals[n as usize]));
            let nv = eval_gate(g.op, buf);
            let o = g.output as usize;
            if nv != vals[o] && vals[o] != Logic::X {
                vals[o] = Logic::X;
                changed = true;
            }
        }
        for r in &sub.resolvers {
            buf.clear();
            buf.extend(r.drivers.iter().map(|&n| vals[n as usize]));
            let o = r.output as usize;
            let nv = resolve(buf, r.termination, vals[o]).value;
            if nv != vals[o] && vals[o] != Logic::X {
                vals[o] = Logic::X;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

/// Synchronous ternary simulation (algorithm B, one tick per step, resolvers zero-delay) until
/// stable. Returns the number of steps that changed something, or `None` if not stable within
/// `max_steps` or the result contains `X`.
fn ternary_b(
    sub: &FlatNetlist,
    vals: &mut Vec<Logic>,
    max_steps: u32,
    buf: &mut Vec<Logic>,
) -> Option<u32> {
    let mut next = vals.clone();
    for step in 0..=max_steps {
        next.copy_from_slice(vals);
        for g in &sub.gates {
            buf.clear();
            buf.extend(g.inputs.iter().map(|&n| vals[n as usize]));
            next[g.output as usize] = eval_gate(g.op, buf);
        }
        for _ in 0..=sub.resolvers.len() {
            let mut ch = false;
            for r in &sub.resolvers {
                buf.clear();
                buf.extend(r.drivers.iter().map(|&n| next[n as usize]));
                let o = r.output as usize;
                let nv = resolve(buf, r.termination, vals[o]).value;
                if next[o] != nv {
                    next[o] = nv;
                    ch = true;
                }
            }
            if !ch {
                break;
            }
        }
        if next == *vals {
            return if vals.contains(&Logic::X) {
                None
            } else {
                Some(step)
            };
        }
        std::mem::swap(vals, &mut next);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use panda_compile::{KernelOptions, compile_kernels, kernel_subnet};
    use panda_netlist::{DefBuilder, DesignBuilder, flatten};

    const LIM: ModelLimits = ModelLimits {
        max_enum_work: 1 << 22,
        max_glitch_sims: 4096,
        max_ternary: 1 << 16,
    };

    /// A lone d_latch with top-level inputs `D`, `EN`: its model and the input indices of D / EN.
    fn latch_model(table: &mut Option<Arc<KernelTable>>) -> (FlatNetlist, usize, usize) {
        let mut db = DesignBuilder::new();
        let l = panda_circuits::d_latch(&mut db);
        let mut b = DefBuilder::new("top");
        let d = b.input("D", 1);
        let en = b.input("EN", 1);
        let q = b.inst_named("l", &l, &[d, en]);
        b.output("Q", q[0]);
        b.output("QN", q[1]);
        let top = db.add(b.finish());
        let flat = flatten(&db.build(&top)).unwrap();
        let (ks, tables) = compile_kernels(&flat, &KernelOptions::default());
        assert_eq!(ks.kernels.len(), 1);
        *table = tables[0].clone();
        let k = &ks.kernels[0];
        let di = k
            .inputs
            .iter()
            .position(|&n| n == flat.net_by_name("D").unwrap())
            .unwrap();
        let ei = k
            .inputs
            .iter()
            .position(|&n| n == flat.net_by_name("EN").unwrap())
            .unwrap();
        (flat, di, ei)
    }

    fn sig(cmask: u32, u_fix: &[(usize, bool)], v_fix: &[(usize, bool)], k: usize) -> RoundSig {
        let mut s = RoundSig {
            cmask,
            rmask: vec![0; k],
            rval: vec![0; k],
            ..RoundSig::default()
        };
        for &(j, b) in u_fix {
            s.u_fix_mask |= 1 << j;
            s.u_fix_val |= u32::from(b) << j;
        }
        for &(j, b) in v_fix {
            s.v_fix_mask |= 1 << j;
            s.v_fix_val |= u32::from(b) << j;
        }
        s.u_pre_mask = (1 << k) - 1;
        s.v_fin_mask = (1 << k) - 1;
        s.v_lit_mask = s.v_fin_mask & !s.v_fix_mask & !(s.u_fix_mask & !cmask);
        s.finish();
        s
    }

    #[test]
    fn d_latch_opaque_transparent_and_bursts() {
        let mut table = None;
        let (flat, di, ei) = latch_model(&mut table);
        let (ks, _) = compile_kernels(&flat, &KernelOptions::default());
        let kernel = &ks.kernels[0];
        let (sub, map) = kernel_subnet(&flat, kernel);
        let t = table.as_deref();
        let mut m = Model::new(sub, map, t, &CharzOptions::default());
        assert!(m.table.is_some());
        let q = kernel
            .nets
            .binary_search(&flat.net_by_name("Q").unwrap())
            .unwrap();
        // EN → s/r NAND → q: two gates; D → NOT/NAND → ... → q: at least two.
        assert_eq!(m.dist[ei * m.m + q], 2);
        assert!(m.dist[di * m.m + q] >= 2);

        // Opaque (EN = 0 throughout): D changes, q never changes; any D glitch is harmless.
        let s = sig(1 << di, &[(ei, false)], &[], 2);
        let r = m.round(&s, &LIM, false, true);
        assert!(r.entries > 0);
        assert!(!r.pos[q].changes);
        assert_eq!(r.burst.map(|b| b.safe), Some(true));

        // Transparent (EN = 1): q follows D; glitches on D still end in the right state.
        let s = sig(1 << di, &[(ei, true)], &[], 2);
        let r = m.round(&s, &LIM, true, true);
        assert!(r.pos[q].changes);
        assert_eq!(r.pos[q].contrib, 1 << di);
        assert!(r.pos[q].lat >= 2);
        assert_eq!(r.burst.map(|b| b.safe), Some(true));

        // EN opens (0 → 1) with D stable but unknown: q may change; the change requires the
        // (final) D value to differ from q — no single D value is implied.
        let s = sig(1 << ei, &[(ei, false)], &[(ei, true)], 2);
        let r = m.round(&s, &LIM, true, true);
        assert!(r.pos[q].changes);
        assert_eq!(r.pos[q].contrib, 1 << ei);
        // Glitches on an opening EN are harmless: the latch ends transparent.
        assert_eq!(r.burst.map(|b| b.safe), Some(true));

        // EN should stay 0 but may glitch: the latch can capture a D that differs from q.
        let s = sig(1 << ei, &[(ei, false)], &[(ei, false)], 2);
        let r = m.round(&s, &LIM, false, true);
        assert!(
            !r.pos[q].changes,
            "no entry changes anything without an actual change"
        );
        assert_eq!(r.burst.map(|b| b.safe), Some(false));

        // EN closes (1 → 0): nothing but the steering nets change.
        let s = sig(1 << ei, &[(ei, true)], &[(ei, false)], 2);
        let r = m.round(&s, &LIM, true, false);
        assert!(!r.pos[q].changes);
        assert!(r.lat_all >= 1);
        assert_eq!(r.glitch.map(|g| g.mask), Some(0));

        // A path condition that contradicts itself means the input cannot change.
        let mut s = sig(1 << di, &[(ei, true)], &[], 2);
        s.rnever = 1 << di;
        s.finish();
        assert!(!m.round(&s, &LIM, false, false).pos[q].changes);
    }

    #[test]
    fn timed_bursts_on_a_closing_latch() {
        let mut table = None;
        let (flat, di, ei) = latch_model(&mut table);
        let (ks, _) = compile_kernels(&flat, &KernelOptions::default());
        let (sub, map) = kernel_subnet(&flat, &ks.kernels[0]);
        let mut m = Model::new(sub, map, table.as_deref(), &CharzOptions::default());
        let q = ks.kernels[0]
            .nets
            .binary_search(&flat.net_by_name("Q").unwrap())
            .unwrap();
        // EN closes (1 → 0) cleanly at tick 0; D may glitch during its window.
        let s = sig((1 << di) | (1 << ei), &[(ei, true)], &[(ei, false)], 2);
        let arr = |en: u32, dmin: u32, dmax: u32, dl: u16| {
            let mut a = vec![None; 2];
            a[ei] = Some(Arrival {
                min: en,
                max: en,
                clean: true,
                first_label: 0,
                last_label: 0,
            });
            a[di] = Some(Arrival {
                min: dmin,
                max: dmax,
                clean: false,
                first_label: dl,
                last_label: dl,
            });
            a
        };
        let mut budget = u64::MAX;
        // The ternary (any-delay) check rejects this burst: D may glitch while EN closes.
        let r = m.round(&s, &LIM, false, true);
        assert_eq!(r.burst.map(|b| b.safe), Some(false));
        // D's activity starts after the closing EN has reached the latch: Q keeps its value.
        let t = m.timed(&s, &arr(0, 4, 8, 0), 1, &LIM, &mut budget).unwrap();
        assert!(t.safe);
        assert!(t.settle >= 8);
        assert!(!t.nets[q].active, "{:?}", t.nets[q]);
        // The same with D changing one Fast round later (the chain T(T(s, EN=0), D)).
        let t = m.timed(&s, &arr(0, 4, 8, 1), 2, &LIM, &mut budget).unwrap();
        assert!(t.safe);
        // Even a D glitch starting with the EN change is harmless: the gates reading D see the
        // closed EN at the same tick.
        let t = m.timed(&s, &arr(0, 0, 3, 0), 1, &LIM, &mut budget).unwrap();
        assert!(t.safe);
        // D glitching while the latch is still open (EN closes at tick 2): it may capture the
        // glitch.
        let t = m.timed(&s, &arr(2, 0, 4, 0), 1, &LIM, &mut budget).unwrap();
        assert!(!t.safe);
        // An exhausted budget gives no answer.
        let mut none = 0u64;
        assert!(m.timed(&s, &arr(0, 5, 9, 0), 1, &LIM, &mut none).is_none());
    }

    #[test]
    fn model_without_table_is_conservative() {
        let mut table = None;
        let (flat, di, _) = latch_model(&mut table);
        let (ks, _) = compile_kernels(&flat, &KernelOptions::default());
        let (sub, map) = kernel_subnet(&flat, &ks.kernels[0]);
        let mut m = Model::new(sub, map, None, &CharzOptions::default());
        let s = sig(1 << di, &[], &[], 2);
        let r = m.round(&s, &LIM, true, true);
        assert!(r.pos.iter().all(|p| p.changes && p.lat == m.bound));
        assert_eq!(r.burst, Some(Burst::unsafe_with(m.bound, m.m)));
        assert_eq!(r.glitch, Some(Transient::all(m.m)));
    }

    #[test]
    fn ternary_algorithms() {
        // NAND SR latch (q = NAND(s, qn), qn = NAND(r, q)) as a sub-netlist: inputs s, r.
        let mut db = DesignBuilder::new();
        let sr = panda_circuits::nand_sr_latch(&mut db);
        let mut b = DefBuilder::new("top");
        let s_n = b.input("S", 1);
        let r_n = b.input("R", 1);
        let q = b.inst_named("l", &sr, &[s_n, r_n]);
        b.output("Q", q[0]);
        b.output("QN", q[1]);
        let top = db.add(b.finish());
        let flat = flatten(&db.build(&top)).unwrap();
        let (ks, _) = compile_kernels(&flat, &KernelOptions::default());
        let (sub, map) = kernel_subnet(&flat, &ks.kernels[0]);
        let k = map.num_inputs();
        let mut buf = Vec::new();
        // Holding (S = R = 1) with q = 1: an X on S (a glitch on the set input) keeps q = 1
        // only if q was already 1.
        let si = map
            .inputs
            .iter()
            .position(|&n| n == flat.net_by_name("S").unwrap())
            .unwrap();
        let qi = map
            .nets
            .iter()
            .position(|&n| n == flat.net_by_name("Q").unwrap())
            .unwrap();
        let mut vals = vec![Logic::One; sub.num_nets as usize];
        vals[k + qi] = Logic::One;
        let qni = 1 - qi;
        vals[k + qni] = Logic::Zero;
        vals[si] = Logic::X;
        ternary_a(&sub, &mut vals, &mut buf);
        assert_eq!(vals[k + qi], Logic::One);
        assert_eq!(vals[k + qni], Logic::Zero);
        // q = 0: the same glitch may set the latch.
        let mut vals = vec![Logic::One; sub.num_nets as usize];
        vals[k + qi] = Logic::Zero;
        vals[k + qni] = Logic::One;
        vals[si] = Logic::X;
        ternary_a(&sub, &mut vals, &mut buf);
        assert_eq!(vals[k + qi], Logic::X);
        // Algorithm B with S = 0 (set): resolves to q = 1 within a few steps.
        vals[si] = Logic::Zero;
        let steps = ternary_b(&sub, &mut vals, 20, &mut buf).expect("binary");
        assert_eq!(vals[k + qi], Logic::One);
        assert!((1..=3).contains(&steps), "{steps}");
        // With S back to 1 after the X, the result stays X (metastable under arbitrary delays).
        let mut vals2 = vals.clone();
        vals2[k + qi] = Logic::X;
        vals2[k + qni] = Logic::X;
        vals2[si] = Logic::One;
        assert_eq!(ternary_b(&sub, &mut vals2, 20, &mut buf), None);
    }
}
