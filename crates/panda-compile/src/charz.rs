//! Kernel characterization — Latch Tables (SPEC §8.4).
//!
//! Every transition has exactly the semantics of running [`TimingSim`] on the kernel's
//! sub-netlist ([`crate::subnet`]), so tables have Timing-mode semantics, including period-2
//! collapse (§5.3). The reference implementation does literally that
//! ([`characterize_subnet_reference`]); [`characterize`] / [`characterize_subnet`] use an exact
//! specialized re-implementation (`fast` module: bit-parallel transitions, compact order walks,
//! optional threads, see [`characterize_subnet_with`]) that produces bit-identical tables and
//! falls back to the reference for sub-netlists outside its preconditions.
//!
//! [`TableCache`] holds tables by structural key; it is internally synchronized, characterizes
//! distinct kernels in parallel ([`TableCache::characterize_all`]), offers non-blocking lookups
//! ([`TableCache::try_tables`]) and can be saved and restored ([`TableCache::snapshot`],
//! [`TableCache::save`] / [`TableCache::load`]).
//!
//! * **State** = the values of all kernel nets ([`Kernel::nets`] order).
//! * **Stable states**: for every binary input vector `u` and every binary assignment of the cut
//!   nets ([`Kernel::cut`]), the remaining kernel nets are computed in topological order (the cut
//!   makes the kernel acyclic); that candidate state is loaded into the simulator under `u` and
//!   settled. Every quiescent result without `X` that is a fixed point under `u` is a stable
//!   state. This finds *every* stable state: a fixed point is determined by its cut values.
//! * **Transitions** `T(s, v)`: load `s` (under an input vector it is stable for), apply `v` all at
//!   once, settle (bound [`CharzOptions::settle_bound`]). `next = None` if the result contains `X`
//!   or did not settle (oscillation). `latency` = ticks from the input change to the last change
//!   of a kernel net.
//! * **Order sensitivity**: for every stable state `s`, every input vector `u` it is stable under,
//!   and every `v` differing from `u` in `j ≥ 2` bits, the `j` changes are also applied one at a
//!   time (each settling before the next) in every order (`j ≤ exhaustive_order_max`, default 4)
//!   or in `order_samples` pseudo-random orders. The entry `(s, v)` is order-sensitive if any
//!   order ends in a different state than the simultaneous change *from some* `u`;
//!   [`KernelTable::sensitive_from`] records the exact `(s, u, v)` combinations. The pass is
//!   bounded by [`CharzOptions::max_order_work`]; combinations it could not check are flagged
//!   conservatively and can be decided exactly on demand ([`KernelTable::check_order_sensitive`]).
//! * **Size guards**: kernels whose table would exceed [`CharzOptions::max_table_entries`] (or
//!   [`MAX_TABLE_INPUTS`] inputs) are not characterized; they run on the fly.
//!
//! "Binary" state values: a kernel net may also hold `Z` (a `Tri` output that is off, or an
//! undriven resolver); only `X` disqualifies a state. Input vectors are always binary.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use panda_netlist::eval::{eval_gate, resolve};
use panda_netlist::flat::{FlatNetlist, NetIdx};
use panda_netlist::{Diagnostic, Logic, Termination, Value};
use panda_timing::{TimingOptions, TimingSim};

use crate::kernels::{Kernel, KernelKind};
use crate::subnet::{INPUT_PIN, SubnetMap, kernel_subnet};

mod cache;
mod fast;
mod lanes;

pub use cache::{
    CacheSnapshot, OptionsTag, SNAPSHOT_FORMAT, SNAPSHOT_VERSION, SnapshotEntry, TableCache,
    TableData,
};

/// Tunables of characterization.
#[derive(Clone, Debug)]
pub struct CharzOptions {
    /// Options of the simulator used for every transition (period-2 collapse threshold etc.).
    pub timing: TimingOptions,
    /// Settle bound in ticks. `None` = `4 × nets + 8 + 2 × p2_limit` (SPEC §8.3: `4 × net count +
    /// 8`, plus room for the collapse window on 1-net kernels).
    pub settle_bound: Option<u64>,
    /// Largest number of simultaneous changes whose orders are checked exhaustively. Default 4.
    pub exhaustive_order_max: usize,
    /// Number of sampled orders when more inputs change. Default 16.
    pub order_samples: usize,
    /// Work budget for the order-sensitivity pass of one kernel, in sequential table steps (an
    /// on-the-fly step from a non-binary intermediate state costs [`SIM_STEP_COST`]). If the
    /// full check (every input vector each state is stable under) is estimated to exceed it, only
    /// one canonical input vector per state is used and [`KernelTable::order_check_partial`] is
    /// set. The budget is also enforced: once spent, every remaining multi-input entry is
    /// conservatively marked order-sensitive (and `order_check_partial` is set), so
    /// characterization time stays bounded. Default 4,000,000.
    pub max_order_work: u64,
    /// Size guard: a kernel whose table would have more than this many entries
    /// (`stable states × 2^inputs`, each costing one kernel simulation) is not characterized
    /// ([`characterize`] returns `None`; the kernel's transitions are then computed on the fly).
    /// Default 2^22.
    pub max_table_entries: u64,
}

impl Default for CharzOptions {
    fn default() -> Self {
        CharzOptions {
            timing: TimingOptions::default(),
            settle_bound: None,
            exhaustive_order_max: 4,
            order_samples: 16,
            max_order_work: 4_000_000,
            max_table_entries: 1 << 22,
        }
    }
}

impl CharzOptions {
    /// Settle bound for a kernel with `nets` state nets.
    pub fn bound_for(&self, nets: usize) -> u64 {
        self.settle_bound
            .unwrap_or(4 * nets as u64 + 8 + 2 * u64::from(self.timing.p2_limit))
    }
}

/// One table entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Transition {
    /// Next stable state id, or `None` for a non-binary result (collapsed to `X`, or did not
    /// settle within the bound).
    pub next: Option<u32>,
    /// Ticks from the input change until the last kernel-net change (0 = nothing changed).
    pub latency: u32,
    /// Applying the changed inputs one at a time in some order gives a different result
    /// (SPEC §6.3 condition 4).
    pub order_sensitive: bool,
}

/// The characterized transition function of a Table kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelTable {
    /// Number of kernel inputs `k`.
    pub num_inputs: usize,
    /// Stable states (values of [`Kernel::nets`]); no `X`.
    pub states: Vec<Vec<Logic>>,
    /// State → id.
    pub state_index: HashMap<Vec<Logic>, u32>,
    /// `transitions[state * 2^k + input_vector]` (input_vector bit `i` = [`Kernel::inputs`]`[i]`).
    pub transitions: Vec<Transition>,
    /// For every binary input vector, the stable states under it (ascending ids). An empty list
    /// means the kernel has no stable state for that input (oscillator, SPEC §4.4).
    pub states_by_input: Vec<Vec<u32>>,
    /// For every state, one input vector it is stable under (the smallest).
    pub home_input: Vec<u32>,
    /// The order-sensitivity check used only one input vector per state (budget exceeded).
    pub order_check_partial: bool,
    /// Exactly which checked `(state, old input vector u, new input vector v)` combinations were
    /// found order-sensitive. [`Transition::order_sensitive`] is the union over every `u` the
    /// state is stable under; this set tells them apart (see
    /// [`KernelTable::order_sensitivity_known`]).
    pub sensitive_from: HashSet<(u32, u32, u32)>,
    /// The order-check work budget ran out at this `(state, u, v)`: it and every later
    /// combination (lexicographic order) were not checked; their entries are conservatively
    /// flagged [`Transition::order_sensitive`].
    pub order_check_stop: Option<(u32, u32, u32)>,
}

impl KernelTable {
    /// Number of stable states.
    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    /// The entry for `(state, input_vector)`.
    pub fn transition(&self, state: u32, input: u32) -> &Transition {
        &self.transitions[((state as usize) << self.num_inputs) | input as usize]
    }

    /// True if the transition from `state` (reached under input vector `u`) to input vector `v`
    /// depends on the arrival order of the changed inputs (SPEC §6.3 condition 4). Exact when the
    /// full order check ran; with [`KernelTable::order_check_partial`] it falls back to the
    /// per-entry flag (conservative for the input vectors that were not checked).
    pub fn is_order_sensitive(&self, state: u32, u: u32, v: u32) -> bool {
        self.order_sensitivity_known(state, u, v)
            .unwrap_or_else(|| self.transition(state, v).order_sensitive)
    }

    /// The characterization's exact answer for `(state, u, v)` (see
    /// [`KernelTable::is_order_sensitive`]), or `None` if that combination was not checked
    /// (partial check or exhausted budget); [`KernelTable::check_order_sensitive`] computes it.
    pub fn order_sensitivity_known(&self, state: u32, u: u32, v: u32) -> Option<bool> {
        if (u ^ v).count_ones() < 2 {
            return Some(false);
        }
        if !self.transition(state, v).order_sensitive {
            // Never flagged: either checked and safe from every `u`, or (partial check) the
            // canonical input vector only — treated as safe, as documented.
            if !self.order_check_partial || self.home_input[state as usize] == u {
                return Some(false);
            }
            return None;
        }
        let checked = (!self.order_check_partial || self.home_input[state as usize] == u)
            && self
                .order_check_stop
                .is_none_or(|stop| (state, u, v) < stop);
        checked.then(|| self.sensitive_from.contains(&(state, u, v)))
    }

    /// Compute the order sensitivity of `(state, u, v)` exactly, with the same orders as
    /// characterization, using `sim` (the kernel's on-the-fly simulator; any simulator of a
    /// structurally identical kernel) and the `opts` the table was built with. Used for
    /// combinations [`KernelTable::order_sensitivity_known`] cannot answer.
    pub fn check_order_sensitive(
        &self,
        sim: &mut KernelSim,
        opts: &CharzOptions,
        state: u32,
        u: u32,
        v: u32,
    ) -> bool {
        OrderChecker::new(opts).entry_sensitive(self, sim, None, state, u, v)
    }

    /// Id of a state vector, if it is a stable state.
    pub fn lookup(&self, state: &[Logic]) -> Option<u32> {
        self.state_index.get(state).copied()
    }

    /// True if some binary input vector leaves the kernel without any stable state (SPEC §4.4
    /// "Oscillator" flag).
    pub fn is_oscillator(&self) -> bool {
        self.states_by_input.iter().any(Vec::is_empty)
    }
}

/// Pack a binary input vector (bit `i` = `inputs[i]`). `None` if any value is not 0/1 or there are
/// more than 31 inputs.
pub fn input_index(inputs: &[Logic]) -> Option<u32> {
    if inputs.len() > 31 {
        return None;
    }
    let mut idx = 0u32;
    for (i, &v) in inputs.iter().enumerate() {
        match v {
            Logic::Zero => {}
            Logic::One => idx |= 1 << i,
            _ => return None,
        }
    }
    Some(idx)
}

/// Unpack an input vector index into `k` logic values.
pub fn input_vector(idx: u32, k: usize) -> Vec<Logic> {
    (0..k)
        .map(|i| Logic::from_bool(idx >> i & 1 == 1))
        .collect()
}

/// Result of [`KernelSim::settle`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Settled {
    /// The kernel became quiescent within the bound.
    pub quiescent: bool,
    /// Ticks from the input change (or the load) to the last kernel-net change.
    pub latency: u32,
}

/// On-the-fly transition function of one kernel: a [`TimingSim`] over its sub-netlist. Used for
/// Sim and Keeper kernels and for Table kernels whose state or inputs contain `X`/`Z`.
pub struct KernelSim {
    sim: TimingSim,
    map: SubnetMap,
    /// State indices of the gate-driven kernel nets (forced by [`KernelSim::load_state`]).
    gate_nets: Vec<usize>,
    /// Tick at which the most recent input change takes effect.
    change_tick: u64,
    bound: u64,
    buf: Vec<Logic>,
    /// Diagnostics raised by the simulator during the most recent [`KernelSim::settle`].
    last_diags: Vec<Diagnostic>,
}

impl KernelSim {
    /// Simulator for `kernel` of `flat`.
    pub fn new(flat: &FlatNetlist, kernel: &Kernel, opts: &CharzOptions) -> KernelSim {
        let (sub, map) = kernel_subnet(flat, kernel);
        KernelSim::from_subnet(Arc::new(sub), map, opts)
    }

    /// Simulator over an already built sub-netlist (see [`kernel_subnet`]).
    pub fn from_subnet(sub: Arc<FlatNetlist>, map: SubnetMap, opts: &CharzOptions) -> KernelSim {
        let k = map.num_inputs();
        let m = map.num_state();
        let mut is_res = vec![false; m];
        for r in &sub.resolvers {
            is_res[r.output as usize - k] = true;
        }
        let gate_nets: Vec<usize> = (0..m).filter(|&j| !is_res[j]).collect();
        let bound = opts.bound_for(m);
        let sim = TimingSim::new(sub, opts.timing.clone());
        KernelSim {
            sim,
            map,
            gate_nets,
            change_tick: 0,
            bound,
            buf: Vec::new(),
            last_diags: Vec::new(),
        }
    }

    /// The sub-netlist mapping.
    pub fn map(&self) -> &SubnetMap {
        &self.map
    }

    /// The default settle bound for this kernel ([`CharzOptions::bound_for`]).
    pub fn default_bound(&self) -> u64 {
        self.bound
    }

    /// The underlying simulator (e.g. to inspect diagnostics).
    pub fn sim_mut(&mut self) -> &mut TimingSim {
        &mut self.sim
    }

    fn set_inputs(&mut self, inputs: &[Logic]) {
        assert_eq!(inputs.len(), self.map.num_inputs(), "input vector width");
        if !inputs.is_empty() {
            self.sim
                .set_input(INPUT_PIN, &Value::from_bits(inputs))
                .expect("kernel input pin");
        }
    }

    /// Power the sub-netlist on with `inputs` (the inputs the state was reached under) and load
    /// `state`: every gate-driven kernel net is forced to its value; resolver outputs are zero-delay
    /// functions of their drivers and are *recomputed* from the loaded values (a keeper uses its
    /// `state` value as the held value). Nothing is simulated yet; a consistent (quiescent) state
    /// stays put when simulation starts, and an inconsistent one (e.g. the all-`X` state of a
    /// kernel that oscillated) cannot leave a resolver stuck at a value its drivers contradict.
    pub fn load_state(&mut self, state: &[Logic], inputs: &[Logic]) {
        assert_eq!(state.len(), self.map.num_state(), "state vector width");
        self.set_inputs(inputs);
        self.sim.reset();
        let k = self.map.num_inputs();
        for &j in &self.gate_nets {
            self.sim.force_net((k + j) as NetIdx, state[j]);
        }
        // Resolvers (in sub-netlist order). Non-keepers were recomputed by the simulator when
        // their drivers were forced (or at reset); keepers get `resolve(drivers, held = state)`.
        let net = self.sim.netlist().clone();
        for r in &net.resolvers {
            if r.termination != Termination::Keeper {
                continue;
            }
            self.buf.clear();
            let vals = self.sim.values();
            self.buf.extend(r.drivers.iter().map(|&n| vals[n as usize]));
            let j = r.output as usize - k;
            let v = resolve(&self.buf, r.termination, state[j]).value;
            self.sim.force_net(r.output, v);
        }
        self.sim.take_diagnostics();
        self.change_tick = self.sim.tick() + 1;
    }

    /// Apply new input values; they take effect at the next tick (all at once).
    pub fn apply_inputs(&mut self, inputs: &[Logic]) {
        self.set_inputs(inputs);
        self.change_tick = self.sim.tick() + 1;
    }

    /// Simulate until quiescent or `bound` ticks elapse.
    pub fn settle(&mut self, bound: u64) -> Settled {
        let k = self.map.num_inputs() as NetIdx;
        let start = self.change_tick;
        let mut last = start;
        let mut quiescent = false;
        for _ in 0..bound {
            if self.sim.is_quiescent() {
                quiescent = true;
                break;
            }
            self.sim.step();
            if self.sim.last_changed().iter().any(|&n| n >= k) {
                last = self.sim.tick();
            }
        }
        quiescent = quiescent || self.sim.is_quiescent();
        self.last_diags = self.sim.take_diagnostics();
        Settled {
            quiescent,
            latency: last.saturating_sub(start).min(u64::from(u32::MAX)) as u32,
        }
    }

    /// Diagnostics the simulator raised during the most recent [`KernelSim::settle`] (e.g.
    /// `Metastable` for a period-2 collapse, `Contention` on a kernel resolver). Locations are
    /// flat net names.
    pub fn last_diagnostics(&self) -> &[Diagnostic] {
        &self.last_diags
    }

    /// Sub-netlist index (see [`crate::subnet`]: `< k` = input `i`, else state net `i - k`) of
    /// the net a diagnostic location names. Simulators are shared by structurally identical
    /// kernels, so callers map this index through *their* kernel's input/net lists.
    pub fn local_net_by_name(&self, name: &str) -> Option<usize> {
        self.sim.netlist().net_names.iter().position(|n| n == name)
    }

    /// Current kernel state.
    pub fn read_state(&self) -> Vec<Logic> {
        let k = self.map.num_inputs();
        self.sim.values()[k..k + self.map.num_state()].to_vec()
    }

    /// Write the current kernel state into `out`.
    pub fn read_state_into(&self, out: &mut Vec<Logic>) {
        let k = self.map.num_inputs();
        out.clear();
        out.extend_from_slice(&self.sim.values()[k..k + self.map.num_state()]);
    }

    /// `T(state, new_inputs)`: load `state` (quiescent under `old_inputs`), apply `new_inputs`,
    /// settle with the default bound. Returns the new state and the settle result.
    pub fn transition(
        &mut self,
        state: &[Logic],
        old_inputs: &[Logic],
        new_inputs: &[Logic],
    ) -> (Vec<Logic>, Settled) {
        self.load_state(state, old_inputs);
        self.apply_inputs(new_inputs);
        let s = self.settle(self.bound);
        (self.read_state(), s)
    }

    /// True if `state` is a fixed point of every kernel gate/resolver under `inputs` (each net
    /// equals its driver's function of the current values; keepers hold their own value).
    pub fn is_fixed_point(&mut self, state: &[Logic], inputs: &[Logic]) -> bool {
        let k = self.map.num_inputs();
        let net = self.sim.netlist().clone();
        let val = |n: NetIdx| -> Logic {
            if (n as usize) < k {
                inputs[n as usize]
            } else {
                state[n as usize - k]
            }
        };
        for g in &net.gates {
            self.buf.clear();
            self.buf.extend(g.inputs.iter().map(|&n| val(n)));
            if eval_gate(g.op, &self.buf) != val(g.output) {
                return false;
            }
        }
        for r in &net.resolvers {
            self.buf.clear();
            self.buf.extend(r.drivers.iter().map(|&n| val(n)));
            if resolve(&self.buf, r.termination, val(r.output)).value != val(r.output) {
                return false;
            }
        }
        true
    }
}

/// Characterize a kernel (SPEC §8.4). `None` unless the kernel is a [`KernelKind::Table`] kernel
/// whose table fits the size guards ([`CharzOptions::max_table_entries`], at most
/// [`MAX_TABLE_INPUTS`] inputs).
pub fn characterize(
    flat: &FlatNetlist,
    kernel: &Kernel,
    opts: &CharzOptions,
) -> Option<KernelTable> {
    if kernel.kind != KernelKind::Table {
        return None;
    }
    let (sub, map) = kernel_subnet(flat, kernel);
    characterize_subnet(Arc::new(sub), map, opts)
}

/// Cost, in [`CharzOptions::max_order_work`] units, of one on-the-fly kernel simulation during
/// the order-sensitivity pass (a table step costs 1).
pub const SIM_STEP_COST: u64 = 64;

/// Largest number of kernel inputs a table is built for (tables are indexed by `2^inputs`).
pub const MAX_TABLE_INPUTS: usize = 24;

/// Characterize an already built kernel sub-netlist. The cut is recomputed locally (any feedback
/// cut gives the same table). `None` if the table would exceed the size guards (see
/// [`characterize`]).
pub fn characterize_subnet(
    sub: Arc<FlatNetlist>,
    map: SubnetMap,
    opts: &CharzOptions,
) -> Option<KernelTable> {
    if map.num_inputs() > MAX_TABLE_INPUTS {
        return None;
    }
    characterize_subnet_with(sub, map, opts, default_threads())
}

/// The default number of threads characterization may use: the available parallelism.
pub fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// [`characterize_subnet`] using at most `threads` threads (`0` is treated as 1). Small kernels
/// never spawn threads. The table does not depend on `threads`.
pub fn characterize_subnet_with(
    sub: Arc<FlatNetlist>,
    map: SubnetMap,
    opts: &CharzOptions,
    threads: usize,
) -> Option<KernelTable> {
    if map.num_inputs() > MAX_TABLE_INPUTS {
        return None;
    }
    let (cut, order) = cut_and_order(&sub, map.num_inputs(), map.num_state());
    if let Some(table) = fast::characterize(&sub, &map, opts, &cut, &order, threads.max(1)) {
        return table;
    }
    Characterizer::with_cut(sub, map, opts, cut, order).run()
}

/// The reference characterization: every transition runs through [`KernelSim`] (a full
/// [`TimingSim`] reset, load and settle). [`characterize_subnet`] gives bit-identical tables
/// (its fast path re-implements exactly the simulation semantics used here); this entry point
/// exists for differential tests and as the documentation of record.
#[doc(hidden)]
pub fn characterize_subnet_reference(
    sub: Arc<FlatNetlist>,
    map: SubnetMap,
    opts: &CharzOptions,
) -> Option<KernelTable> {
    if map.num_inputs() > MAX_TABLE_INPUTS {
        return None;
    }
    let (cut, order) = cut_and_order(&sub, map.num_inputs(), map.num_state());
    Characterizer::with_cut(sub, map, opts, cut, order).run()
}

/// True if [`characterize_subnet`] uses its fast path for this sub-netlist (the reference path
/// otherwise; both give identical tables). For tests and diagnostics.
#[doc(hidden)]
pub fn uses_fast_path(sub: &FlatNetlist, map: &SubnetMap, opts: &CharzOptions) -> bool {
    fast::KSim::new(sub, map, opts).is_some()
}

/// The fast path's `T(state, old, new)` = `(next state, quiescent, latency)`, or `None` if the
/// sub-netlist does not qualify for the fast path. Must equal [`KernelSim::transition`] exactly
/// (differential tests).
#[doc(hidden)]
pub fn fast_transition(
    sub: &FlatNetlist,
    map: &SubnetMap,
    opts: &CharzOptions,
    state: &[Logic],
    old: u32,
    new: u32,
) -> Option<(Vec<Logic>, bool, u32)> {
    let mut sim = fast::KSim::new(sub, map, opts)?;
    Some(fast::sim_transition(&mut sim, state, old, new))
}

/// Hit counts of the fast path's rarely taken exact-fallback branches, process-wide and
/// monotonic: `[period-2 collapses, resolver evaluations in simulated ticks, order-pass entries
/// on the sequential path, on-the-fly steps simulated, streamed exhaustive-order entries,
/// bit-parallel batches with lanes handed to the scalar simulator]`. For coverage tests.
#[doc(hidden)]
pub fn fast_path_coverage() -> [u64; 6] {
    fast::cov::snapshot()
}

/// Feedback cut of the kernel's state nets (always containing keeper outputs) and the
/// evaluation order of the remaining state nets (topological once the cut is removed).
fn cut_and_order(sub: &FlatNetlist, k: usize, m: usize) -> (Vec<usize>, Vec<usize>) {
    let mut driver: Vec<Option<Drv>> = (0..m).map(|_| None).collect();
    for (i, g) in sub.gates.iter().enumerate() {
        driver[g.output as usize - k] = Some(Drv::Gate(i));
    }
    for (i, r) in sub.resolvers.iter().enumerate() {
        driver[r.output as usize - k] = Some(Drv::Res(i));
    }
    let deps = |j: usize| -> Vec<usize> {
        let ins: &[NetIdx] = match driver[j].as_ref().expect("state net has a driver") {
            Drv::Gate(i) => &sub.gates[*i].inputs,
            Drv::Res(i) => &sub.resolvers[*i].drivers,
        };
        ins.iter()
            .filter(|&&n| n as usize >= k)
            .map(|&n| n as usize - k)
            .collect()
    };
    let succ = crate::graph::Csr::from_fn(m, |j, buf| {
        // edges dep → j are built below via transpose; here rows are deps of j
        buf.extend(deps(j as usize).iter().map(|&d| d as u32));
    })
    .transpose();
    let is_keeper = |j: u32| matches!(driver[j as usize], Some(Drv::Res(i)) if sub.resolvers[i].termination == Termination::Keeper);
    let all: Vec<u32> = (0..m as u32).collect();
    let cut: Vec<usize> = crate::graph::feedback_vertex_set(&succ, &all, is_keeper, m)
        .expect("limit m always suffices")
        .into_iter()
        .map(|j| j as usize)
        .collect();
    let mut is_cut = vec![false; m];
    for &j in &cut {
        is_cut[j] = true;
    }
    let order = crate::graph::topo_order(&succ, |j| !is_cut[j as usize])
        .expect("cut makes the kernel acyclic")
        .into_iter()
        .map(|j| j as usize)
        .collect();
    (cut, order)
}

/// How a node computes its state net from the others (local evaluation during enumeration).
enum Drv {
    Gate(usize),
    Res(usize),
}

struct Characterizer<'a> {
    sub: Arc<FlatNetlist>,
    ks: KernelSim,
    opts: &'a CharzOptions,
    k: usize,
    m: usize,
    /// Cut state indices.
    cut: Vec<usize>,
    /// Non-cut state indices in evaluation order, with their drivers.
    order: Vec<(usize, Drv)>,
    bound: u64,
}

/// Settled results of the non-binary table entries, keyed by `(state, input vector)`
/// (`None` = did not settle).
type RawResults = HashMap<(u32, u32), Option<Vec<Logic>>>;

/// One step of a sequential (one input at a time) application.
#[derive(Clone)]
enum Cur {
    Id(u32),
    /// Settled but non-binary state (reached under the previous input vector).
    Raw(Vec<Logic>),
    /// Did not settle: the outcome is non-binary.
    Lost,
}

impl<'a> Characterizer<'a> {
    fn with_cut(
        sub: Arc<FlatNetlist>,
        map: SubnetMap,
        opts: &'a CharzOptions,
        cut: Vec<usize>,
        order: Vec<usize>,
    ) -> Self {
        let k = map.num_inputs();
        let m = map.num_state();
        let ks = KernelSim::from_subnet(sub.clone(), map, opts);
        let bound = ks.default_bound();
        let mut driver: Vec<Option<Drv>> = (0..m).map(|_| None).collect();
        for (i, g) in sub.gates.iter().enumerate() {
            driver[g.output as usize - k] = Some(Drv::Gate(i));
        }
        for (i, r) in sub.resolvers.iter().enumerate() {
            driver[r.output as usize - k] = Some(Drv::Res(i));
        }
        let order = order
            .into_iter()
            .map(|j| (j, driver[j].take().expect("driver")))
            .collect();
        Characterizer {
            sub,
            ks,
            opts,
            k,
            m,
            cut,
            order,
            bound,
        }
    }

    /// Candidate state for input vector `u` and cut assignment `c` (acyclic evaluation).
    fn candidate(&self, inputs: &[Logic], c: u32) -> Vec<Logic> {
        let k = self.k;
        let mut st = vec![Logic::X; self.m];
        for (b, &j) in self.cut.iter().enumerate() {
            st[j] = Logic::from_bool(c >> b & 1 == 1);
        }
        let mut buf = Vec::new();
        for (j, drv) in &self.order {
            buf.clear();
            let ins: &[NetIdx] = match drv {
                Drv::Gate(i) => &self.sub.gates[*i].inputs,
                Drv::Res(i) => &self.sub.resolvers[*i].drivers,
            };
            buf.extend(ins.iter().map(|&n| {
                if (n as usize) < k {
                    inputs[n as usize]
                } else {
                    st[n as usize - k]
                }
            }));
            st[*j] = match drv {
                Drv::Gate(i) => eval_gate(self.sub.gates[*i].op, &buf),
                Drv::Res(i) => resolve(&buf, self.sub.resolvers[*i].termination, Logic::X).value,
            };
        }
        st
    }

    fn run(mut self) -> Option<KernelTable> {
        let k = self.k;
        debug_assert!(k <= MAX_TABLE_INPUTS);
        // Stable-state enumeration costs 2^(inputs + cut) simulations.
        if k + self.cut.len() > 2 * MAX_TABLE_INPUTS
            || (1u64 << (k + self.cut.len())) > self.opts.max_table_entries.max(1 << 12)
        {
            return None;
        }
        let nv = 1u32 << k;
        let mut states: Vec<Vec<Logic>> = Vec::new();
        let mut state_index: HashMap<Vec<Logic>, u32> = HashMap::new();
        let mut states_by_input: Vec<Vec<u32>> = vec![Vec::new(); nv as usize];
        let mut home_input: Vec<u32> = Vec::new();
        let ncut = 1u32 << self.cut.len();
        for u in 0..nv {
            let uv = input_vector(u, k);
            for c in 0..ncut {
                let cand = self.candidate(&uv, c);
                self.ks.load_state(&cand, &uv);
                let s = self.ks.settle(self.bound);
                if !s.quiescent {
                    continue;
                }
                let st = self.ks.read_state();
                if st.contains(&Logic::X) || !self.ks.is_fixed_point(&st, &uv) {
                    continue;
                }
                let id = match state_index.get(&st) {
                    Some(&id) => id,
                    None => {
                        let id = states.len() as u32;
                        state_index.insert(st.clone(), id);
                        states.push(st);
                        home_input.push(u);
                        id
                    }
                };
                let list = &mut states_by_input[u as usize];
                if !list.contains(&id) {
                    list.push(id);
                }
            }
            states_by_input[u as usize].sort_unstable();
        }
        // Simultaneous transitions.
        let ns = states.len();
        if (ns as u64) << k > self.opts.max_table_entries {
            return None;
        }
        let mut transitions = Vec::with_capacity(ns << k);
        // Raw results of non-binary entries (needed by the order pass): None = did not settle.
        let mut raw: RawResults = HashMap::new();
        for (s, st) in states.iter().enumerate() {
            let home = input_vector(home_input[s], k);
            for v in 0..nv {
                let (next_st, settled) = self.ks.transition(st, &home, &input_vector(v, k));
                let next = if settled.quiescent {
                    state_index.get(&next_st).copied()
                } else {
                    None
                };
                if next.is_none() {
                    raw.insert((s as u32, v), settled.quiescent.then_some(next_st));
                }
                transitions.push(Transition {
                    next,
                    latency: settled.latency,
                    order_sensitive: false,
                });
            }
        }
        let mut table = KernelTable {
            num_inputs: k,
            states,
            state_index,
            transitions,
            states_by_input,
            home_input,
            order_check_partial: false,
            sensitive_from: HashSet::new(),
            order_check_stop: None,
        };
        self.order_pass(&mut table, &raw);
        Some(table)
    }

    fn order_pass(&mut self, table: &mut KernelTable, raw: &RawResults) {
        let k = self.k;
        let nv = 1u32 << k;
        if k < 2 || table.states.is_empty() {
            return;
        }
        let mut oc = OrderChecker::new(self.opts);
        // Work estimate for the full check: Σ_u |states(u)| × Σ_{v: j≥2} orders(j) × j.
        let per_home: u64 = (0..=k)
            .filter(|&j| j >= 2)
            .map(|j| binomial(k, j) * oc.orders_for(j) * j as u64)
            .sum();
        let homes_total: u64 = table.states_by_input.iter().map(|l| l.len() as u64).sum();
        let full = homes_total.saturating_mul(per_home) <= self.opts.max_order_work;
        table.order_check_partial = !full;
        let mut homes: Vec<Vec<u32>> = vec![Vec::new(); table.states.len()];
        if full {
            for (u, list) in table.states_by_input.iter().enumerate() {
                for &s in list {
                    homes[s as usize].push(u as u32);
                }
            }
        } else {
            for (s, &u) in table.home_input.iter().enumerate() {
                homes[s].push(u);
            }
        }
        for s in 0..table.states.len() as u32 {
            for &u in &homes[s as usize] {
                for v in 0..nv {
                    if (u ^ v).count_ones() < 2 {
                        continue;
                    }
                    let idx = ((s as usize) << k) | v as usize;
                    if oc.work > self.opts.max_order_work {
                        // Budget spent: unknown, so conservatively flagged; the exact answer is
                        // computed on demand ([`KernelTable::check_order_sensitive`]).
                        table.order_check_stop.get_or_insert((s, u, v));
                        table.transitions[idx].order_sensitive = true;
                        continue;
                    }
                    if oc.entry_sensitive(table, &mut self.ks, Some(raw), s, u, v) {
                        table.transitions[idx].order_sensitive = true;
                        table.sensitive_from.insert((s, u, v));
                    }
                }
            }
        }
    }
}

/// The order-sensitivity check of one table entry (SPEC §8.4), with a memo of on-the-fly steps.
struct OrderChecker<'a> {
    opts: &'a CharzOptions,
    /// Work spent ([`CharzOptions::max_order_work`] units).
    work: u64,
    /// `(non-binary state, old inputs, new inputs)` → outcome of the on-the-fly step.
    memo: HashMap<(Vec<Logic>, u32, u32), Cur>,
    perm: Vec<usize>,
}

impl<'a> OrderChecker<'a> {
    fn new(opts: &'a CharzOptions) -> Self {
        OrderChecker {
            opts,
            work: 0,
            memo: HashMap::new(),
            perm: Vec::new(),
        }
    }

    /// Number of orders checked for `j` simultaneous changes.
    fn orders_for(&self, j: usize) -> u64 {
        if j <= self.opts.exhaustive_order_max {
            (1..=j as u64).product()
        } else {
            self.opts.order_samples as u64
        }
    }

    /// True if applying the changes `u → v` one at a time from state `s` (stable under `u`) in
    /// some order (all orders for `j ≤ exhaustive_order_max` changes, else `order_samples`
    /// deterministic pseudo-random ones) ends differently from the simultaneous change.
    /// `raw`: the settled non-binary results of non-binary entries, if known.
    fn entry_sensitive(
        &mut self,
        table: &KernelTable,
        sim: &mut KernelSim,
        raw: Option<&RawResults>,
        s: u32,
        u: u32,
        v: u32,
    ) -> bool {
        let k = table.num_inputs;
        let diff = u ^ v;
        let j = diff.count_ones() as usize;
        if j < 2 {
            return false;
        }
        let simul = table.transition(s, v).next;
        let bits: Vec<usize> = (0..k).filter(|&b| diff >> b & 1 == 1).collect();
        if j <= self.opts.exhaustive_order_max {
            let mut perm = std::mem::take(&mut self.perm);
            perm.clear();
            perm.extend(0..j);
            let mut sensitive = false;
            loop {
                let order: Vec<usize> = perm.iter().map(|&p| bits[p]).collect();
                if self.sequential(table, sim, raw, s, u, &order) != simul {
                    sensitive = true;
                    break;
                }
                if !next_permutation(&mut perm) {
                    break;
                }
            }
            self.perm = perm;
            sensitive
        } else {
            let mut rng = XorShift::new(
                0x9E37_79B9_7F4A_7C15 ^ (u64::from(s) << 40) ^ (u64::from(u) << 20) ^ u64::from(v),
            );
            let mut order = bits;
            for _ in 0..self.opts.order_samples {
                for i in (1..order.len()).rev() {
                    let r = (rng.next() % (i as u64 + 1)) as usize;
                    order.swap(i, r);
                }
                if self.sequential(table, sim, raw, s, u, &order) != simul {
                    return true;
                }
            }
            false
        }
    }

    /// Final state id after applying the input changes `order` (bit positions) one at a time from
    /// state `s` under inputs `u`, settling after each. `None` = non-binary.
    fn sequential(
        &mut self,
        table: &KernelTable,
        sim: &mut KernelSim,
        raw: Option<&RawResults>,
        s: u32,
        u: u32,
        order: &[usize],
    ) -> Option<u32> {
        let mut cur = Cur::Id(s);
        let mut w = u;
        for &b in order {
            let prev = w;
            w ^= 1 << b;
            self.work += 1;
            cur = match cur {
                Cur::Id(x) => match table.transition(x, w).next {
                    Some(y) => Cur::Id(y),
                    None => match raw {
                        Some(raw) => match raw.get(&(x, w)) {
                            Some(Some(st)) => Cur::Raw(st.clone()),
                            _ => Cur::Lost,
                        },
                        // Recompute the non-binary entry (T does not depend on which input
                        // vector the stable state is loaded under).
                        None => self.step(table, sim, table.states[x as usize].clone(), prev, w),
                    },
                },
                Cur::Raw(st) => self.step(table, sim, st, prev, w),
                Cur::Lost => Cur::Lost,
            };
        }
        match cur {
            Cur::Id(x) => Some(x),
            _ => None,
        }
    }

    /// One memoized on-the-fly step from `st` (settled under `prev`) to inputs `w`.
    fn step(
        &mut self,
        table: &KernelTable,
        sim: &mut KernelSim,
        st: Vec<Logic>,
        prev: u32,
        w: u32,
    ) -> Cur {
        let k = table.num_inputs;
        let key = (st, prev, w);
        if let Some(c) = self.memo.get(&key) {
            return c.clone();
        }
        self.work += SIM_STEP_COST;
        let (nst, settled) = sim.transition(&key.0, &input_vector(prev, k), &input_vector(w, k));
        let c = if !settled.quiescent {
            Cur::Lost
        } else if let Some(&id) = table.state_index.get(&nst) {
            Cur::Id(id)
        } else {
            Cur::Raw(nst)
        };
        self.memo.insert(key, c.clone());
        c
    }
}

fn binomial(n: usize, r: usize) -> u64 {
    let mut acc = 1u64;
    for i in 0..r as u64 {
        acc = acc * (n as u64 - i) / (i + 1);
    }
    acc
}

/// Lexicographic next permutation; false when `p` was the last one.
fn next_permutation(p: &mut [usize]) -> bool {
    if p.len() < 2 {
        return false;
    }
    let mut i = p.len() - 1;
    while i > 0 && p[i - 1] >= p[i] {
        i -= 1;
    }
    if i == 0 {
        return false;
    }
    let mut j = p.len() - 1;
    while p[j] <= p[i - 1] {
        j -= 1;
    }
    p.swap(i - 1, j);
    p[i..].reverse();
    true
}

/// Small deterministic PRNG for order sampling.
struct XorShift(u64);

impl XorShift {
    fn new(seed: u64) -> Self {
        XorShift(seed | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutations() {
        let mut p = vec![0, 1, 2];
        let mut n = 1;
        while next_permutation(&mut p) {
            n += 1;
        }
        assert_eq!(n, 6);
        assert_eq!(p, vec![2, 1, 0]);
    }

    #[test]
    fn binom() {
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(12, 6), 924);
    }

    #[test]
    fn input_packing() {
        let v = input_vector(0b101, 3);
        assert_eq!(v, vec![Logic::One, Logic::Zero, Logic::One]);
        assert_eq!(input_index(&v), Some(0b101));
        assert_eq!(input_index(&[Logic::X]), None);
    }
}
