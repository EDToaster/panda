//! FastRef — the reference Fast-mode engine (SPEC §6.2), the M3 oracle.
//!
//! FastRef is a direct interpreter of the kernel-delta algorithm. Every optimized Fast-mode engine
//! must produce exactly its results.
//!
//! # Model
//!
//! * **State** = the value of every net. Kernel nets ([`Kernel::nets`]) are the kernel states;
//!   every other gate/resolver output is combinational and is a function of sources and kernel
//!   nets, evaluated with zero delay in [`KernelSet::comb_order`].
//! * **Events** ([`FastRef::step_event`]): tick 0 (processed by [`FastRef::reset`]), every tick at
//!   which some `Clock` output differs from its value at the previous tick, and every
//!   `PowerOnReset` release tick. Simultaneous source changes form one event. Input changes made
//!   with [`FastRef::set_input`] are queued and processed by the next [`FastRef::step_event`] as one
//!   event at the *current* tick (before the next scheduled source event), mirroring "inputs
//!   change after the edge".
//!
//! # Per-event algorithm (SPEC §6.2)
//!
//! 1. **Round 0**: every source takes its value at the event tick (top-level inputs: the values
//!    given to [`FastRef::set_input`], default 0); the combinational logic downstream of changed
//!    sources is re-evaluated (dirty-driven, in topological order), kernel outputs taken from the
//!    current kernel states.
//! 2. **Rounds 1, 2, …**: every kernel whose input vector differs from the vector of its last
//!    transition transitions — all of them at once, reading the values settled by the previous
//!    round — `state ← T(state, inputs)`; then the combinational logic downstream of changed kernel
//!    nets is re-evaluated. Repeat until no kernel input vector changed.
//! 3. **Transition `T`**: if the kernel has a [`KernelTable`], its state is a table state and its
//!    inputs are binary, the table entry is used (a *table hit*). Otherwise (state or inputs with
//!    `X`/`Z`, Sim/Keeper kernels, or a non-binary table entry) `T` is computed on the fly by a
//!    [`KernelSim`] (a *sim fallback*), which yields the exact, possibly `X`-carrying state.
//!    A kernel whose on-the-fly transition does not settle within the bound
//!    ([`CharzOptions::bound_for`]) goes to all-`X` and raises `Oscillating`.
//! 4. **Delta limit**: if kernels still need to transition after [`FastOptions::delta_limit`]
//!    rounds, every such kernel's nets are set to `X` and the kernel is *frozen* for the rest of the
//!    event (it does not transition again until the next event); an `IterationLimit` diagnostic
//!    is raised per frozen kernel. The rounds then continue (with a fresh budget) so that the
//!    `X` propagates; freezing is monotone, so every event terminates.
//! 5. **Publish**: [`FastRef::values`] holds every net at the settled point.
//!
//! Power-on (tick 0): every gate and resolver output is `X`, undriven nets read their termination
//! (`Z`, pull-up `1`, pull-down `0`, keeper `X`), sources take their tick-0 values. Every kernel
//! transitions once at tick 0 (from its all-`X` state), and all combinational logic is evaluated.
//!
//! # Diagnostics
//!
//! * `DelaySensitive` (warning): a table entry was used for a transition in which ≥ 2 kernel
//!   inputs changed at once and whose result depends on their arrival order from the kernel's
//!   actual previous input vector ([`KernelTable::is_order_sensitive`], SPEC §6.3 condition 4).
//!   Once per kernel per reset.
//! * `Oscillating` (warning): an on-the-fly transition did not settle. Once per kernel per reset.
//! * `Metastable` (warning): an on-the-fly transition went through a period-2 collapse inside the
//!   kernel (as reported by the kernel's simulator). Once per kernel per reset. (Table entries
//!   whose result is non-binary are always recomputed on the fly, so a table hit never hides one.)
//! * `IterationLimit` (warning): see step 4. Once per kernel per event.
//! * `Contention` (warning): a combinational resolver's settled value at the end of an event has
//!   contention ([`resolve`]). Reported again only after the resolver settled once without
//!   contention.
//!
//! * `Contention` on a resolver *inside* a kernel: raised when an on-the-fly transition saw
//!   contention, located at the resolver's net; re-armed by a transition without contention.
//!   Contention that only occurs transiently during a table transition is not reported.
//!
//! Other diagnostics raised inside kernel simulations are not forwarded.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use panda_netlist::eval::{eval_gate, resolve, terminate};
use panda_netlist::flat::{FlatNetlist, NetIdx, SourceKind};
use panda_netlist::{DiagKind, Diagnostic, Logic, Value};

use crate::charz::{CharzOptions, KernelSim, KernelTable, TableCache};
use crate::graph::Node;
use crate::kernels::{ExtractOptions, Kernel, KernelSet};
use crate::subnet::{kernel_subnet, structural_key};
use crate::{KernelOptions, compile_kernels_with_cache};

/// Sentinel for "no index".
const NONE: u32 = u32::MAX;

/// Tunables of [`FastRef`].
#[derive(Clone, Debug)]
pub struct FastOptions {
    /// `DELTA_LIMIT`: kernel rounds per event before the still-changing kernels are forced to `X`
    /// (with `IterationLimit`). Default 64.
    pub delta_limit: u32,
    /// Kernel extraction tunables.
    pub extract: ExtractOptions,
    /// Characterization / on-the-fly transition tunables.
    pub charz: CharzOptions,
}

impl Default for FastOptions {
    fn default() -> Self {
        FastOptions {
            delta_limit: 64,
            extract: ExtractOptions::default(),
            charz: CharzOptions::default(),
        }
    }
}

/// Counters since the last [`FastRef::reset`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FastRefStats {
    /// Events processed (including the tick-0 event).
    pub events: u64,
    /// Kernel rounds executed (rounds in which at least one kernel transitioned).
    pub rounds: u64,
    /// Most rounds needed by a single event.
    pub max_rounds: u32,
    /// Kernel transitions computed.
    pub transitions: u64,
    /// Transitions answered by a table entry.
    pub table_hits: u64,
    /// Transitions computed on the fly ([`KernelSim`]).
    pub sim_fallbacks: u64,
    /// Combinational gate/resolver evaluations.
    pub comb_evals: u64,
}

/// A time-varying source.
#[derive(Clone, Copy, Debug)]
enum Timed {
    Clock { period: u64, high: u64, phase: u64 },
    PowerOnReset { ticks: u64 },
}

impl Timed {
    /// Value at tick `t` (same formula as the timing engine).
    fn value_at(self, t: u64) -> Logic {
        match self {
            Timed::Clock {
                period,
                high,
                phase,
            } => {
                if period == 0 {
                    return Logic::from_bool(high > 0);
                }
                Logic::from_bool((t % period + period - phase % period) % period < high)
            }
            Timed::PowerOnReset { ticks } => Logic::from_bool(t < ticks),
        }
    }

    /// First tick `> t` at which the value differs from the value at the previous tick.
    fn next_change_after(self, t: u64) -> Option<u64> {
        match self {
            Timed::Clock {
                period,
                high,
                phase,
            } => {
                if period == 0 || high == 0 || high >= period {
                    return None;
                }
                // Position of tick t within the cycle (0 = rising edge, `high` = falling edge).
                let r = (t % period + period - phase % period) % period;
                let dist = |c: u64| (c + period - r - 1) % period + 1;
                t.checked_add(dist(0).min(dist(high)))
            }
            Timed::PowerOnReset { ticks } => (ticks > t).then_some(ticks),
        }
    }
}

/// What a kernel transitions to in the current round.
enum NewState {
    /// A table state id.
    Table(u32),
    /// An explicit state vector (from [`KernelSim`]).
    Explicit(Vec<Logic>),
    /// All nets `X`.
    AllX,
}

/// Per-kernel static data and runtime bookkeeping.
struct KState {
    /// Offset of this kernel's inputs in [`FastRef::last_in`].
    in_off: usize,
    /// The kernel has transitioned since the last reset (`last_in` is valid).
    ever: bool,
    /// Current table state id, or [`NONE`] (unknown / not a table state).
    state_id: u32,
    /// Index into [`FastRef::sims`] once the on-the-fly simulator exists.
    sim: u32,
    /// Frozen at `X` for the rest of the current event (delta limit).
    frozen: bool,
    delay_reported: bool,
    osc_reported: bool,
    meta_reported: bool,
    /// The kernel's last on-the-fly transition saw contention (re-armed by one without).
    contention_active: bool,
}

/// The reference Fast-mode engine (SPEC §6.2). See the module docs.
pub struct FastRef {
    flat: Arc<FlatNetlist>,
    opts: FastOptions,
    ks: KernelSet,
    tables: Vec<Option<Arc<KernelTable>>>,

    // ---- static structure ----
    /// Position of each comb node in `ks.comb_order`, per net it drives (NONE otherwise).
    /// Per net: comb readers (positions in comb_order), CSR.
    comb_rd_start: Vec<u32>,
    comb_rd: Vec<u32>,
    /// Per net: kernels reading it as an input, CSR.
    kern_rd_start: Vec<u32>,
    kern_rd: Vec<u32>,
    /// Time-varying sources.
    timed: Vec<(NetIdx, Timed)>,
    /// Top-level input sources: (net, pin, bit).
    input_srcs: Vec<(NetIdx, u32, u32)>,
    /// On-the-fly simulators, shared by structurally identical kernels.
    sims: Vec<KernelSim>,
    sim_by_key: HashMap<Vec<u32>, u32>,
    /// Order sensitivities computed on demand: (table address, state, u, v) → sensitive.
    order_memo: HashMap<(usize, u32, u32, u32), bool>,

    // ---- runtime ----
    values: Vec<Logic>,
    kst: Vec<KState>,
    last_in: Vec<Logic>,
    input_values: Vec<Value>,
    inputs_pending: bool,
    tick: u64,
    /// Comb positions scheduled for evaluation.
    heap: BinaryHeap<Reverse<u32>>,
    scheduled: Vec<bool>,
    /// Kernels whose inputs may have changed.
    cand: Vec<u32>,
    is_cand: Vec<bool>,
    /// Comb resolvers evaluated during the current event (positions in comb_order).
    res_touched: Vec<u32>,
    res_touched_mark: Vec<bool>,
    res_contending: Vec<bool>,
    res_reported: Vec<bool>,
    frozen_list: Vec<u32>,
    diags: Vec<Diagnostic>,
    stats: FastRefStats,
    buf: Vec<Logic>,
}

impl FastRef {
    /// Compile `flat` (kernel extraction + characterization) and power it on (processes the
    /// tick-0 event). Errors if the extracted kernel set is inconsistent (an internal error).
    pub fn new(flat: Arc<FlatNetlist>, opts: FastOptions) -> Result<FastRef, Vec<Diagnostic>> {
        FastRef::with_cache(flat, opts, &TableCache::new())
    }

    /// [`FastRef::new`] with a caller-provided table cache (reusable across netlists compiled
    /// with the same [`CharzOptions`]; internally synchronized, `&mut` is not required).
    pub fn with_cache(
        flat: Arc<FlatNetlist>,
        opts: FastOptions,
        cache: &TableCache,
    ) -> Result<FastRef, Vec<Diagnostic>> {
        let kopts = KernelOptions {
            extract: opts.extract.clone(),
            charz: opts.charz.clone(),
        };
        let (ks, tables) = compile_kernels_with_cache(&flat, &kopts, cache);
        FastRef::from_parts(flat, opts, ks, tables)
    }

    /// Build the engine from an already extracted kernel set and its tables (`tables[i]` belongs
    /// to `ks.kernels[i]`; `None` = always on the fly). The tables must have been characterized
    /// with `opts.charz`. Powers on (processes the tick-0 event).
    pub fn from_parts(
        flat: Arc<FlatNetlist>,
        opts: FastOptions,
        ks: KernelSet,
        tables: Vec<Option<Arc<KernelTable>>>,
    ) -> Result<FastRef, Vec<Diagnostic>> {
        if let Err(e) = ks.verify(&flat) {
            return Err(vec![Diagnostic::error(
                DiagKind::Other,
                format!("inconsistent kernel set: {e}"),
            )]);
        }
        if tables.len() != ks.kernels.len() {
            return Err(vec![Diagnostic::error(
                DiagKind::Other,
                format!("{} tables for {} kernels", tables.len(), ks.kernels.len()),
            )]);
        }
        for (k, t) in ks.kernels.iter().zip(&tables) {
            if let Some(t) = t
                && t.num_inputs != k.inputs.len()
            {
                return Err(vec![Diagnostic::error(
                    DiagKind::Other,
                    format!("table of kernel {} does not match its inputs", k.id),
                )]);
            }
        }
        let n = flat.num_nets as usize;

        // Comb readers per net.
        let nodes_inputs = |node: Node| -> &[NetIdx] {
            match node {
                Node::Gate(g) => &flat.gates[g as usize].inputs,
                Node::Resolver(r) => &flat.resolvers[r as usize].drivers,
            }
        };
        let mut pairs: Vec<(NetIdx, u32)> = Vec::new();
        for (pos, &node) in ks.comb_order.iter().enumerate() {
            for &net in nodes_inputs(node) {
                pairs.push((net, pos as u32));
            }
        }
        let (comb_rd_start, comb_rd) = csr(n, &mut pairs);
        let mut kpairs: Vec<(NetIdx, u32)> = Vec::new();
        for k in &ks.kernels {
            for &net in &k.inputs {
                kpairs.push((net, k.id));
            }
        }
        let (kern_rd_start, kern_rd) = csr(n, &mut kpairs);

        let mut timed = Vec::new();
        let mut input_srcs = Vec::new();
        for s in &flat.sources {
            match s.kind {
                SourceKind::Clock {
                    period,
                    high,
                    phase,
                } => timed.push((
                    s.output,
                    Timed::Clock {
                        period: u64::from(period),
                        high: u64::from(high),
                        phase: u64::from(phase),
                    },
                )),
                SourceKind::PowerOnReset { ticks } => timed.push((
                    s.output,
                    Timed::PowerOnReset {
                        ticks: u64::from(ticks),
                    },
                )),
                SourceKind::TopInput { pin, bit } => input_srcs.push((s.output, pin, bit)),
                SourceKind::Constant(_) => {}
            }
        }

        let mut off = 0usize;
        let kst = ks
            .kernels
            .iter()
            .map(|k| {
                let in_off = off;
                off += k.inputs.len();
                KState {
                    in_off,
                    ever: false,
                    state_id: NONE,
                    sim: NONE,
                    frozen: false,
                    delay_reported: false,
                    osc_reported: false,
                    meta_reported: false,
                    contention_active: false,
                }
            })
            .collect();
        let nc = ks.comb_order.len();
        let nk = ks.kernels.len();
        let input_values = flat
            .top_inputs
            .iter()
            .map(|p| Value::zeros(p.nets.len() as u32))
            .collect();
        let mut fr = FastRef {
            flat,
            opts,
            ks,
            tables,
            comb_rd_start,
            comb_rd,
            kern_rd_start,
            kern_rd,
            timed,
            input_srcs,
            sims: Vec::new(),
            sim_by_key: HashMap::new(),
            order_memo: HashMap::new(),
            values: vec![Logic::X; n],
            kst,
            last_in: vec![Logic::X; off],
            input_values,
            inputs_pending: false,
            tick: 0,
            heap: BinaryHeap::new(),
            scheduled: vec![false; nc],
            cand: Vec::new(),
            is_cand: vec![false; nk],
            res_touched: Vec::new(),
            res_touched_mark: vec![false; nc],
            res_contending: vec![false; nc],
            res_reported: vec![false; nc],
            frozen_list: Vec::new(),
            diags: Vec::new(),
            stats: FastRefStats::default(),
            buf: Vec::new(),
        };
        fr.reset();
        Ok(fr)
    }

    /// The netlist.
    pub fn netlist(&self) -> &Arc<FlatNetlist> {
        &self.flat
    }

    /// The options this engine was created with.
    pub fn options(&self) -> &FastOptions {
        &self.opts
    }

    /// Back to tick 0: every gate/resolver output `X` (kernel states all `X`), then the tick-0
    /// event is processed. Input pin values set so far are kept (a pending input change is folded
    /// into the tick-0 event); untaken diagnostics are kept; statistics restart.
    pub fn reset(&mut self) {
        let flat = self.flat.clone();
        self.values.fill(Logic::Z);
        for g in &flat.gates {
            self.values[g.output as usize] = Logic::X;
        }
        for r in &flat.resolvers {
            self.values[r.output as usize] = Logic::X;
        }
        for (&net, &term) in &flat.undriven_termination {
            if (net as usize) < self.values.len() && self.values[net as usize] == Logic::Z {
                self.values[net as usize] = terminate(term, Logic::X);
            }
        }
        for s in &flat.sources {
            if let SourceKind::Constant(v) = s.kind {
                self.values[s.output as usize] = v;
            }
        }
        for &(net, t) in &self.timed {
            self.values[net as usize] = t.value_at(0);
        }
        for &(net, pin, bit) in &self.input_srcs {
            self.values[net as usize] = self.input_values[pin as usize].get(bit);
        }
        for k in &mut self.kst {
            k.ever = false;
            k.state_id = NONE;
            k.frozen = false;
            k.delay_reported = false;
            k.osc_reported = false;
            k.meta_reported = false;
            k.contention_active = false;
        }
        self.last_in.fill(Logic::X);
        self.inputs_pending = false;
        self.tick = 0;
        self.stats = FastRefStats::default();
        self.heap.clear();
        self.scheduled.fill(true);
        for pos in 0..self.ks.comb_order.len() as u32 {
            self.heap.push(Reverse(pos));
        }
        self.cand.clear();
        self.is_cand.fill(true);
        self.cand.extend(0..self.ks.kernels.len() as u32);
        self.res_touched.clear();
        self.res_touched_mark.fill(false);
        self.res_contending.fill(false);
        self.res_reported.fill(false);
        self.process_event();
    }

    /// Set a top-level input pin (by name). The width must match. The change is queued and
    /// processed by the next [`FastRef::step_event`] as an event at the current tick.
    pub fn set_input(&mut self, pin: &str, value: &Value) -> Result<(), String> {
        let idx = self
            .flat
            .top_inputs
            .iter()
            .position(|p| p.name == pin)
            .ok_or_else(|| format!("no top-level input pin '{pin}'"))?;
        let width = self.flat.top_inputs[idx].nets.len() as u32;
        if value.width() != width {
            return Err(format!(
                "input pin '{pin}' has width {width}, got a {}-bit value",
                value.width()
            ));
        }
        if self.input_values[idx] != *value {
            self.input_values[idx] = value.clone();
            self.inputs_pending = true;
        }
        Ok(())
    }

    /// The value most recently given to a top-level input pin (default 0), if the pin exists.
    pub fn input_value(&self, pin: &str) -> Option<&Value> {
        let idx = self.flat.top_inputs.iter().position(|p| p.name == pin)?;
        Some(&self.input_values[idx])
    }

    /// True if input changes are queued (the next [`FastRef::step_event`] processes them at the
    /// current tick).
    pub fn has_pending_inputs(&self) -> bool {
        self.inputs_pending
    }

    /// Process one event: the pending input changes if any (as one event at the current tick),
    /// else the next scheduled source event (clock edges / `PowerOnReset` release). Returns the
    /// event tick, or `None` if there is nothing to do (no pending inputs and no future source
    /// change).
    pub fn step_event(&mut self) -> Option<u64> {
        if self.inputs_pending {
            self.inputs_pending = false;
            self.apply_sources();
            self.process_event();
            return Some(self.tick);
        }
        let t = self.next_event_tick()?;
        self.tick = t;
        self.apply_sources();
        self.process_event();
        Some(t)
    }

    /// Process up to `n` events (stops early when there is nothing to do).
    pub fn run_events(&mut self, n: u64) {
        for _ in 0..n {
            if self.step_event().is_none() {
                break;
            }
        }
    }

    /// Process every event up to and including tick `t` (pending inputs first). Afterwards the
    /// values correspond to Timing mode at tick `t` provided the next event is after `t`.
    pub fn run_until(&mut self, t: u64) {
        if self.inputs_pending {
            self.step_event();
        }
        while self.next_event_tick().is_some_and(|e| e <= t) {
            self.step_event();
        }
    }

    /// Tick of the most recently processed event. Values correspond to Timing mode at
    /// `next_event_tick - 1` under the SPEC §6.3 conditions.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Tick of the next scheduled source event (strictly after [`FastRef::tick`]), or `None` if no
    /// source will change again. Pending input changes are not included (see
    /// [`FastRef::has_pending_inputs`]).
    pub fn next_event_tick(&self) -> Option<u64> {
        self.timed
            .iter()
            .filter_map(|&(_, t)| t.next_change_after(self.tick))
            .min()
    }

    /// Value of a net at the settled point.
    pub fn get(&self, net: NetIdx) -> Logic {
        self.values[net as usize]
    }

    /// Values of all nets at the settled point (index = [`NetIdx`]).
    pub fn values(&self) -> &[Logic] {
        &self.values
    }

    /// Read a top-level pin (input or output) by name.
    pub fn read_pin(&self, pin: &str) -> Option<Value> {
        let nl = &self.flat;
        let p = nl
            .top_inputs
            .iter()
            .chain(&nl.top_outputs)
            .find(|p| p.name == pin)?;
        Some(self.gather(&p.nets))
    }

    /// Read by hierarchical name: an exact flat net name (`"cpu/alu/c[3]"`, `"q"`), or a
    /// multi-bit net base name (`"cpu/alu/c"` gathers `c[0]..`), or a top-level pin name.
    pub fn read(&self, name: &str) -> Option<Value> {
        let nl = &self.flat;
        if let Some(net) = nl.net_by_name(name) {
            return Some(Value::from_logic(self.get(net)));
        }
        let mut nets = Vec::new();
        while let Some(net) = nl.net_by_name(&format!("{name}[{}]", nets.len())) {
            nets.push(net);
        }
        if !nets.is_empty() {
            return Some(self.gather(&nets));
        }
        self.read_pin(name)
    }

    /// Drain the diagnostics raised so far.
    pub fn take_diagnostics(&mut self) -> Vec<Diagnostic> {
        std::mem::take(&mut self.diags)
    }

    /// The diagnostics raised so far (not drained).
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diags
    }

    /// The kernel set.
    pub fn kernels(&self) -> &KernelSet {
        &self.ks
    }

    /// The kernel tables (`tables()[i]` belongs to kernel `i`).
    pub fn tables(&self) -> &[Option<Arc<KernelTable>>] {
        &self.tables
    }

    /// Counters since the last reset.
    pub fn stats(&self) -> FastRefStats {
        self.stats
    }

    // -----------------------------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------------------------

    fn gather(&self, nets: &[NetIdx]) -> Value {
        let bits: Vec<Logic> = nets.iter().map(|&n| self.get(n)).collect();
        Value::from_bits(&bits)
    }

    fn emit(&mut self, d: Diagnostic) {
        if self.diags.len() < panda_timing::MAX_PENDING_DIAGNOSTICS {
            self.diags.push(d);
        }
    }

    fn kernel_location(&self, k: u32) -> String {
        let kern = &self.ks.kernels[k as usize];
        let net = kern.outputs.first().or(kern.nets.first());
        match net {
            Some(&n) => self.flat.net_names[n as usize].clone(),
            None => format!("kernel {k}"),
        }
    }

    /// Give every time-varying source and top-level input its value at the current tick.
    fn apply_sources(&mut self) {
        let t = self.tick;
        for i in 0..self.timed.len() {
            let (net, src) = self.timed[i];
            self.set_net(net, src.value_at(t));
        }
        for i in 0..self.input_srcs.len() {
            let (net, pin, bit) = self.input_srcs[i];
            let v = self.input_values[pin as usize].get(bit);
            self.set_net(net, v);
        }
    }

    /// Write a net; on change schedule its readers.
    #[inline]
    fn set_net(&mut self, net: NetIdx, v: Logic) {
        let n = net as usize;
        if self.values[n] == v {
            return;
        }
        self.values[n] = v;
        let (a, b) = (
            self.comb_rd_start[n] as usize,
            self.comb_rd_start[n + 1] as usize,
        );
        for i in a..b {
            let pos = self.comb_rd[i];
            if !self.scheduled[pos as usize] {
                self.scheduled[pos as usize] = true;
                self.heap.push(Reverse(pos));
            }
        }
        let (a, b) = (
            self.kern_rd_start[n] as usize,
            self.kern_rd_start[n + 1] as usize,
        );
        for i in a..b {
            let k = self.kern_rd[i];
            if !self.is_cand[k as usize] {
                self.is_cand[k as usize] = true;
                self.cand.push(k);
            }
        }
    }

    /// Evaluate scheduled combinational nodes in topological order until none is left.
    fn eval_comb(&mut self) {
        let flat = self.flat.clone();
        while let Some(Reverse(pos)) = self.heap.pop() {
            let p = pos as usize;
            self.scheduled[p] = false;
            self.stats.comb_evals += 1;
            match self.ks.comb_order[p] {
                Node::Gate(g) => {
                    let gate = &flat.gates[g as usize];
                    let v = match gate.inputs.as_slice() {
                        [a] => eval_gate(gate.op, &[self.values[*a as usize]]),
                        [a, b] => eval_gate(
                            gate.op,
                            &[self.values[*a as usize], self.values[*b as usize]],
                        ),
                        ins => {
                            self.buf.clear();
                            self.buf
                                .extend(ins.iter().map(|&n| self.values[n as usize]));
                            eval_gate(gate.op, &self.buf)
                        }
                    };
                    self.set_net(gate.output, v);
                }
                Node::Resolver(r) => {
                    let res = &flat.resolvers[r as usize];
                    self.buf.clear();
                    self.buf
                        .extend(res.drivers.iter().map(|&n| self.values[n as usize]));
                    let out = self.values[res.output as usize];
                    let rv = resolve(&self.buf, res.termination, out);
                    self.res_contending[p] = rv.contention;
                    if !self.res_touched_mark[p] {
                        self.res_touched_mark[p] = true;
                        self.res_touched.push(pos);
                    }
                    self.set_net(res.output, rv.value);
                }
            }
        }
    }

    /// Rounds 0.. of SPEC §6.2 for the source values already applied.
    fn process_event(&mut self) {
        self.stats.events += 1;
        self.eval_comb();
        let mut rounds_since_freeze = 0u32;
        let mut event_rounds = 0u32;
        let mut pending: Vec<(u32, NewState)> = Vec::new();
        let mut todo: Vec<u32> = Vec::new();
        loop {
            // Kernels whose input vector differs from the one of their last transition.
            todo.clear();
            let mut cand = std::mem::take(&mut self.cand);
            cand.sort_unstable();
            for &k in &cand {
                self.is_cand[k as usize] = false;
                if !self.kst[k as usize].frozen && self.inputs_changed(k) {
                    todo.push(k);
                }
            }
            cand.clear();
            self.cand = cand;
            if todo.is_empty() {
                break;
            }
            if rounds_since_freeze >= self.opts.delta_limit {
                // Delta limit: force the still-changing kernels to X and freeze them.
                for &k in &todo {
                    self.kst[k as usize].frozen = true;
                    self.frozen_list.push(k);
                    self.record_inputs(k);
                    let loc = self.kernel_location(k);
                    self.emit(
                        Diagnostic::warning(
                            DiagKind::IterationLimit,
                            format!(
                                "kernel {k} still changing after {} rounds at tick {}; forced to X",
                                self.opts.delta_limit, self.tick
                            ),
                        )
                        .at(loc),
                    );
                    pending.push((k, NewState::AllX));
                }
                rounds_since_freeze = 0;
            } else {
                for &k in &todo {
                    let ns = self.transition(k);
                    pending.push((k, ns));
                }
                rounds_since_freeze += 1;
                event_rounds += 1;
                self.stats.rounds += 1;
            }
            for (k, ns) in pending.drain(..) {
                self.write_state(k, ns);
            }
            self.eval_comb();
        }
        self.stats.max_rounds = self.stats.max_rounds.max(event_rounds);
        for k in self.frozen_list.drain(..) {
            self.kst[k as usize].frozen = false;
        }
        self.check_contention();
    }

    /// Report contention of combinational resolvers evaluated during this event (settled values).
    fn check_contention(&mut self) {
        let touched = std::mem::take(&mut self.res_touched);
        for &pos in &touched {
            let p = pos as usize;
            self.res_touched_mark[p] = false;
            if self.res_contending[p] {
                if !self.res_reported[p] {
                    self.res_reported[p] = true;
                    let Node::Resolver(r) = self.ks.comb_order[p] else {
                        continue;
                    };
                    let net = self.flat.resolvers[r as usize].output;
                    let name = self.flat.net_names[net as usize].clone();
                    let msg = format!("contention on net {name} at tick {}", self.tick);
                    self.emit(Diagnostic::warning(DiagKind::Contention, msg).at(name));
                }
            } else {
                self.res_reported[p] = false;
            }
        }
        self.res_touched = touched;
        self.res_touched.clear();
    }

    /// True if kernel `k`'s current inputs differ from those of its last transition.
    fn inputs_changed(&self, k: u32) -> bool {
        let ks = &self.kst[k as usize];
        if !ks.ever {
            return true;
        }
        let ins = &self.ks.kernels[k as usize].inputs;
        ins.iter()
            .enumerate()
            .any(|(i, &n)| self.values[n as usize] != self.last_in[ks.in_off + i])
    }

    fn record_inputs(&mut self, k: u32) {
        let off = self.kst[k as usize].in_off;
        let ins = &self.ks.kernels[k as usize].inputs;
        for (i, &n) in ins.iter().enumerate() {
            self.last_in[off + i] = self.values[n as usize];
        }
        self.kst[k as usize].ever = true;
    }

    /// Binary input index of kernel `k` from the current values (`None` if any input is not 0/1).
    fn current_input_index(&self, k: u32) -> Option<u32> {
        let ins = &self.ks.kernels[k as usize].inputs;
        if ins.len() > 31 {
            return None;
        }
        let mut idx = 0u32;
        for (i, &n) in ins.iter().enumerate() {
            match self.values[n as usize] {
                Logic::Zero => {}
                Logic::One => idx |= 1 << i,
                _ => return None,
            }
        }
        Some(idx)
    }

    /// Binary index of the inputs of kernel `k`'s last transition.
    fn last_input_index(&self, k: u32) -> Option<u32> {
        let ks = &self.kst[k as usize];
        if !ks.ever {
            return None;
        }
        let n = self.ks.kernels[k as usize].inputs.len();
        crate::charz::input_index(&self.last_in[ks.in_off..ks.in_off + n])
    }

    fn state_vec(&self, k: u32) -> Vec<Logic> {
        self.ks.kernels[k as usize]
            .nets
            .iter()
            .map(|&n| self.values[n as usize])
            .collect()
    }

    /// Compute `T(state, inputs)` for kernel `k` from the current values; records the inputs as
    /// the kernel's last-transition inputs. Does not write the state.
    fn transition(&mut self, k: u32) -> NewState {
        self.stats.transitions += 1;
        let ki = k as usize;
        let new_idx = self.current_input_index(k);
        let old_idx = self.last_input_index(k);
        let mut entry = None;
        if let (Some(table), Some(v)) = (self.tables[ki].as_deref(), new_idx) {
            if self.kst[ki].state_id == NONE
                && let Some(id) = table.lookup(&self.state_vec(k))
            {
                self.kst[ki].state_id = id;
            }
            let sid = self.kst[ki].state_id;
            if sid != NONE {
                entry = Some((v, *table.transition(sid, v)));
            }
        }
        if let Some((v, tr)) = entry {
            let sid = self.kst[ki].state_id;
            if !self.kst[ki].delay_reported
                && let Some(u) = old_idx
                && self.order_sensitive(k, sid, u, v)
            {
                self.kst[ki].delay_reported = true;
                let loc = self.kernel_location(k);
                self.emit(
                    Diagnostic::warning(
                        DiagKind::DelaySensitive,
                        format!(
                            "kernel {k}: several inputs changed at once at tick {} and the result \
                             depends on their arrival order",
                            self.tick
                        ),
                    )
                    .at(loc),
                );
            }
            if let Some(next) = tr.next {
                self.stats.table_hits += 1;
                self.record_inputs(k);
                return NewState::Table(next);
            }
        }
        // On the fly.
        self.stats.sim_fallbacks += 1;
        let state = self.state_vec(k);
        let n_in = self.ks.kernels[ki].inputs.len();
        let new_in: Vec<Logic> = self.ks.kernels[ki]
            .inputs
            .iter()
            .map(|&n| self.values[n as usize])
            .collect();
        // First transition after power-on: the inputs change from X (power-on) to their current
        // values, so that every kernel element reading them — in particular a resolver whose
        // drivers are all kernel inputs, e.g. a keeper net driven by constants — is evaluated, as
        // Timing mode resolves every resolver at power-on. (With `old = new` nothing would
        // change and the forced all-X state would stick forever.)
        let old_in: Vec<Logic> = if self.kst[ki].ever {
            let off = self.kst[ki].in_off;
            self.last_in[off..off + n_in].to_vec()
        } else {
            vec![Logic::X; n_in]
        };
        let si = self.sim_index(k);
        let (next, settled) = self.sims[si].transition(&state, &old_in, &new_in);
        self.record_inputs(k);
        if !settled.quiescent {
            if !self.kst[ki].osc_reported {
                self.kst[ki].osc_reported = true;
                let loc = self.kernel_location(k);
                self.emit(
                    Diagnostic::warning(
                        DiagKind::Oscillating,
                        format!(
                            "kernel {k} does not settle at tick {}; its nets are set to X",
                            self.tick
                        ),
                    )
                    .at(loc),
                );
            }
            return NewState::AllX;
        }
        // Forward what the kernel simulation observed: a period-2 collapse (`Metastable`) and
        // contention on a kernel resolver. (An `X` result alone is not evidence of
        // metastability: contention inside the kernel also yields `X`.)
        let (mut meta, mut contention) = (false, None);
        for d in self.sims[si].last_diagnostics() {
            match d.kind {
                DiagKind::Metastable => meta = true,
                DiagKind::Contention if contention.is_none() => {
                    // The simulator may be shared with a structurally identical kernel: map
                    // the location through this kernel's net lists.
                    let kern = &self.ks.kernels[ki];
                    let net = d
                        .location
                        .as_deref()
                        .and_then(|l| self.sims[si].local_net_by_name(l))
                        .map(|i| {
                            if i < n_in {
                                kern.inputs[i]
                            } else {
                                kern.nets[i - n_in]
                            }
                        });
                    contention = Some(match net {
                        Some(n) => self.flat.net_names[n as usize].clone(),
                        None => self.kernel_location(k),
                    });
                }
                _ => {}
            }
        }
        match contention {
            Some(net) if !self.kst[ki].contention_active => {
                self.kst[ki].contention_active = true;
                self.emit(
                    Diagnostic::warning(
                        DiagKind::Contention,
                        format!("contention on net {net} (kernel {k}) at tick {}", self.tick),
                    )
                    .at(net),
                );
            }
            Some(_) => {}
            None => self.kst[ki].contention_active = false,
        }
        if meta && !self.kst[ki].meta_reported {
            self.kst[ki].meta_reported = true;
            let loc = self.kernel_location(k);
            self.emit(
                Diagnostic::warning(
                    DiagKind::Metastable,
                    format!(
                        "kernel {k} collapsed to X (period-2 oscillation) at tick {}",
                        self.tick
                    ),
                )
                .at(loc),
            );
        }
        NewState::Explicit(next)
    }

    /// Order sensitivity (SPEC §6.3 condition 4) of kernel `k`'s table entry from state `sid`
    /// (reached under input vector `u`) to `v`: the table's answer when characterization checked
    /// that combination, else computed exactly on demand and memoized.
    fn order_sensitive(&mut self, k: u32, sid: u32, u: u32, v: u32) -> bool {
        let ki = k as usize;
        let Some(table) = self.tables[ki].clone() else {
            return false;
        };
        if let Some(b) = table.order_sensitivity_known(sid, u, v) {
            return b;
        }
        let key = (Arc::as_ptr(&table) as usize, sid, u, v);
        if let Some(&b) = self.order_memo.get(&key) {
            return b;
        }
        let si = self.sim_index(k);
        let b = table.check_order_sensitive(&mut self.sims[si], &self.opts.charz, sid, u, v);
        self.order_memo.insert(key, b);
        b
    }

    /// Index of the on-the-fly simulator of kernel `k`, creating it on first use.
    fn sim_index(&mut self, k: u32) -> usize {
        let ki = k as usize;
        if self.kst[ki].sim == NONE {
            let kern: &Kernel = &self.ks.kernels[ki];
            let (sub, map) = kernel_subnet(&self.flat, kern);
            let key = structural_key(&sub);
            let idx = match self.sim_by_key.get(&key) {
                Some(&i) => i,
                None => {
                    let i = self.sims.len() as u32;
                    self.sims
                        .push(KernelSim::from_subnet(Arc::new(sub), map, &self.opts.charz));
                    self.sim_by_key.insert(key, i);
                    i
                }
            };
            self.kst[ki].sim = idx;
        }
        self.kst[ki].sim as usize
    }

    /// Write a kernel's new state into the net values (scheduling readers of changed nets).
    fn write_state(&mut self, k: u32, ns: NewState) {
        let ki = k as usize;
        let m = self.ks.kernels[ki].nets.len();
        match ns {
            NewState::Table(id) => {
                for j in 0..m {
                    let v = match &self.tables[ki] {
                        Some(t) => t.states[id as usize][j],
                        None => unreachable!("table state without table"),
                    };
                    self.set_net(self.ks.kernels[ki].nets[j], v);
                }
                self.kst[ki].state_id = id;
            }
            NewState::Explicit(st) => {
                for (j, &v) in st.iter().enumerate() {
                    self.set_net(self.ks.kernels[ki].nets[j], v);
                }
                self.kst[ki].state_id = match &self.tables[ki] {
                    Some(t) => t.lookup(&st).unwrap_or(NONE),
                    None => NONE,
                };
            }
            NewState::AllX => {
                for j in 0..m {
                    self.set_net(self.ks.kernels[ki].nets[j], Logic::X);
                }
                self.kst[ki].state_id = NONE;
            }
        }
    }
}

/// Build a CSR (per net) from `(net, value)` pairs; values within a row are deduplicated.
fn csr(n: usize, pairs: &mut [(NetIdx, u32)]) -> (Vec<u32>, Vec<u32>) {
    pairs.sort_unstable();
    let mut start = vec![0u32; n + 1];
    let mut adj = Vec::with_capacity(pairs.len());
    let mut prev: Option<(NetIdx, u32)> = None;
    for &(net, v) in pairs.iter() {
        if prev == Some((net, v)) {
            continue;
        }
        prev = Some((net, v));
        adj.push(v);
        start[net as usize + 1] += 1;
    }
    for i in 0..n {
        start[i + 1] += start[i];
    }
    (start, adj)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_change_schedule() {
        let c = Timed::Clock {
            period: 10,
            high: 5,
            phase: 5,
        };
        // value_at: high for ticks 5..10, 15..20, ...
        assert_eq!(c.value_at(0), Logic::Zero);
        assert_eq!(c.value_at(5), Logic::One);
        assert_eq!(c.value_at(10), Logic::Zero);
        assert_eq!(c.next_change_after(0), Some(5));
        assert_eq!(c.next_change_after(4), Some(5));
        assert_eq!(c.next_change_after(5), Some(10));
        assert_eq!(c.next_change_after(9), Some(10));
        assert_eq!(c.next_change_after(10), Some(15));
        // Brute force agreement for odd parameters.
        for (period, high, phase) in [(7u64, 3u64, 2u64), (4, 1, 0), (6, 5, 13), (2, 1, 1)] {
            let c = Timed::Clock {
                period,
                high,
                phase,
            };
            for t in 0..50u64 {
                let want = (t + 1..t + 100).find(|&u| c.value_at(u) != c.value_at(u - 1));
                assert_eq!(
                    c.next_change_after(t),
                    want,
                    "{period} {high} {phase} t={t}"
                );
            }
        }
        let flat = Timed::Clock {
            period: 8,
            high: 0,
            phase: 0,
        };
        assert_eq!(flat.next_change_after(3), None);
    }

    #[test]
    fn por_schedule() {
        let p = Timed::PowerOnReset { ticks: 20 };
        assert_eq!(p.next_change_after(0), Some(20));
        assert_eq!(p.next_change_after(19), Some(20));
        assert_eq!(p.next_change_after(20), None);
        assert_eq!(Timed::PowerOnReset { ticks: 0 }.next_change_after(0), None);
    }

    #[test]
    fn csr_dedup() {
        let mut pairs = vec![(2, 7), (0, 1), (2, 7), (2, 3)];
        let (s, a) = csr(4, &mut pairs);
        assert_eq!(s, vec![0, 1, 1, 3, 3]);
        assert_eq!(a, vec![1, 3, 7]);
    }
}
