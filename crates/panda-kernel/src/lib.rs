//! panda-kernel — the facade of the Panda simulator kernel (SPEC §10).
//!
//! A [`Kernel`] owns a loaded [`Design`], its flattened netlist and the engines that simulate it:
//!
//! * **Timing mode** ([`Mode::Timing`]): the unit-delay reference engine
//!   ([`panda_timing::TimingSim`]).
//! * **Fast mode** ([`Mode::Fast`]): the kernel-delta engine of SPEC §6 behind an internal engine
//!   trait. The backend is chosen with [`KernelOptions::backend`]: [`FastBackend::Jit`]
//!   (default: `panda_fast::FastSim` with its Cranelift tier from `panda_jit`),
//!   [`FastBackend::Compiled`] (the `FastSim` interpreter only) or [`FastBackend::Reference`]
//!   (FastRef semantics, the oracle). All are bit-identical after every event; with `Compiled`
//!   and `Jit`, a Reference engine built from the same kernel tables serves while the FastSim
//!   program compiles, and with `Jit` the interpreting FastSim hot-swaps to native code at an
//!   event boundary once the JIT has compiled its program (tiers Reference → interpreter →
//!   JIT; see [`Kernel::jit_active`]).
//!
//! # Time
//!
//! The kernel has one clock, [`Kernel::tick`]. In Timing mode it is the engine's tick. In Fast
//! mode the engine is always at the end of a phase: after processing the event at tick `e`, its
//! values equal Timing mode's values at `next_event − 1` (SPEC §6.3), and that is the tick
//! reported. Stepping ([`Kernel::step`]) is defined on this common clock:
//!
//! * [`StepUnit::Tick`]: advance `n` ticks. Fast mode cannot stop inside a phase: it processes
//!   every event up to the target tick and ends at the end of the phase containing it.
//! * [`StepUnit::Phase`]: advance to the end of the next phase, i.e. to the tick before the first
//!   source change after `tick + 1` (from the middle of a phase: to its end).
//! * [`StepUnit::Cycle`]: advance to the tick before the first rising edge of the reference clock
//!   ([`KernelOptions::cycle_clock`], default: the first `Clock`) after `tick + 1`. At this point a
//!   whole clock cycle has settled (the sampling point of `panda_cpu8::sim::GateCpu`).
//!
//! Without any future source change, `Phase` and `Cycle` let pending input changes settle.
//!
//! Input changes ([`Kernel::set_input`]) follow each mode's semantics: Timing mode applies them at
//! the next tick (at a phase end that is the tick of the next clock edge, i.e. simultaneous with
//! it — a hold race, exactly as in the timing engine); Fast mode processes them as their own event
//! after the last processed edge, at the start of the next step (`step(0, StepUnit::Tick)` applies
//! them without advancing).
//!
//! # Example
//!
//! ```
//! use panda_circuits::{sync_counter, testbench};
//! use panda_kernel::{Kernel, Mode, StepUnit};
//! use panda_netlist::{DesignBuilder, Value};
//!
//! let mut db = DesignBuilder::new();
//! let counter = sync_counter(&mut db, 4);
//! let top = testbench(&mut db, &counter, 16);
//! let mut k = Kernel::default();
//! k.load(db.build(&top)).unwrap();
//! k.set_input("EN", &Value::from_bool(true)).unwrap();
//! k.step(5, StepUnit::Cycle); // Timing mode
//! k.set_mode(Mode::Fast); // compiles in the background; Timing serves until ready
//! k.wait_ready().unwrap();
//! let before = k.read_u64("Q").unwrap();
//! k.step(3, StepUnit::Cycle);
//! assert_eq!(k.read_u64("Q"), Some((before + 3) % 16));
//! assert_eq!(k.read("dut/reg/Q").unwrap().width(), 4); // nets inside packages
//! ```
//!
//! # Mode switches and state transfer
//!
//! * Timing → Fast happens at a phase end: the Timing engine is first advanced to the end of the
//!   current phase (no-op if it is at one). The Fast engine then loads every net value. SPEC §10
//!   makes the switch legal at quiescence; if the Timing engine still has gate events pending at
//!   that phase end (the phase is shorter than the settling time, or the design oscillates), the
//!   switch still happens and a warning says the state was not quiescent. Defined behaviour of
//!   the load ([`Kernel`]'s Fast engines share it): combinational logic is re-evaluated from the
//!   kernel nets and sources, and every kernel whose state is inconsistent with its inputs (caught
//!   mid-transition, or poked through a [`State`]) *finishes* that transition with Timing-mode
//!   semantics under its current inputs; only a kernel that does not settle restarts from `X`.
//!   A warning reports either case.
//! * Fast → Timing loads every net value into a fresh Timing engine at the same tick (sources are
//!   aligned to the tick, every non-source net is forced, inconsistent gates are re-evaluated).
//! * Fast mode needs compiled kernels. Compilation runs on a worker thread
//!   ([`KernelOptions::background`]); until it is done, Fast-mode requests are served by the
//!   Timing engine at phase granularity (SPEC §10), and the state is transferred when the engine
//!   is ready (checked at every API call; [`Kernel::wait_ready`] blocks for it). With
//!   [`FastBackend::Compiled`] a Reference engine takes over as soon as the kernel tables exist
//!   and is hot-swapped for the `FastSim` once its program is compiled. A compiled engine is kept
//!   while the design does not change, so switching back and forth is cheap. Results of a
//!   compilation started for an older design are never used (every load / effective edit starts
//!   a new generation; stale workers stop at their next stage boundary).
//! * While the Timing engine stands in for Fast mode, input changes follow Timing semantics
//!   (applied at the next tick, i.e. at a phase end simultaneously with the next edge), not Fast
//!   semantics (their own event before the next edge). On hold-clean designs both agree.
//!
//! # Snapshots
//!
//! [`Kernel::snapshot`] records every net value, the applied and the pending input values, the
//! tick and the mode. [`Kernel::restore`] matches nets by name (so snapshots survive edits) and
//! works across modes: a Fast-mode snapshot (a settled phase end) loads directly into a Fast
//! engine; a Timing-mode snapshot restored in Fast mode is loaded into the Timing engine, which
//! finishes the phase, and then transferred like a mode switch.
//!
//! # Edits
//!
//! [`Kernel::edit`] applies a batch of [`EditOp`]s to a copy of the design, validates and
//! flattens it, and rolls everything back on error. Otherwise net values are carried across by
//! net name (nets that still exist keep their values, new nets start `X`, gates whose output
//! disagrees with their inputs are re-evaluated; input pins whose width changed are reset to 0).
//! Input changes given in Timing mode and not applied yet still take effect at the next tick.
//! Timing mode resumes immediately; Fast mode runs on the Timing engine until the new program is
//! compiled (kernel tables are cached by structure, so unchanged defs are not re-characterized),
//! then rebuilds its kernel states from the carried values.
//!
//! # Diagnostics
//!
//! Every producer (validation, engines, compiler, STA, the facade) feeds one deduplicated list
//! ([`Kernel::diagnostics`], [`Kernel::diagnostic_entries`] with occurrence counts and producers).

mod classify;
mod diag;
mod edit;
mod engine;
mod refengine;
mod schedule;
mod timing;

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use panda_compile::{FastOptions, KernelOptions as CompileOptions, TableCache};
use panda_netlist::flat::{FlatNetlist, SourceKind};
use panda_netlist::validate::{has_errors, validate};
use panda_netlist::{DiagKind, Diagnostic, Logic, Severity, Value, flatten};
use panda_sta::StaOptions;
use panda_timing::TimingOptions;

pub use classify::{ClassKind, Classification, Edge, KernelSummary, MAX_EXPLORE_INPUT_BITS};
pub use diag::{DiagEntry, DiagSource};
pub use edit::EditOp;
pub use engine::FastBackend;
pub use panda_fast::CompileOptions as FastCompileOptions;
pub use panda_netlist::Design;
pub use panda_sta::TimingReport;

use engine::{Artifacts, FastEngine, JitTier, LoadReport, build_engine};
use schedule::Schedule;
use timing::TimingEngine;

/// Simulation mode (SPEC D3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Mode {
    /// Unit-delay, event-driven reference semantics (SPEC §5).
    #[default]
    Timing,
    /// Kernel-delta semantics, phase granularity (SPEC §6).
    Fast,
}

/// Granularity of [`Kernel::step`] (see the crate docs, "Time").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StepUnit {
    /// One tick.
    Tick,
    /// To the end of the next phase.
    Phase,
    /// To the end of the next clock cycle of the reference clock.
    Cycle,
}

/// Which engine currently serves requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EngineKind {
    /// No design loaded.
    None,
    /// The Timing engine in Timing mode.
    Timing,
    /// The Timing engine standing in for Fast mode (at phase granularity) while compiling.
    TimingFallback,
    /// A Fast-mode engine.
    Fast(FastBackend),
}

/// Facade options.
#[derive(Clone, Debug)]
pub struct KernelOptions {
    /// Timing-engine options.
    pub timing: TimingOptions,
    /// Fast-mode (kernel extraction, characterization, delta limit) options.
    pub fast: FastOptions,
    /// Fast-mode engine.
    pub backend: FastBackend,
    /// Compiler tunables of [`FastBackend::Compiled`] (performance only).
    pub compiled: FastCompileOptions,
    /// Compile Fast-mode programs on a worker thread (default `true`). With `false`, compilation
    /// happens synchronously inside the call that needs it.
    pub background: bool,
    /// Static timing analysis options ([`Kernel::timing_report`]).
    pub sta: StaOptions,
    /// Clock net defining [`StepUnit::Cycle`] (default: the first `Clock` source).
    pub cycle_clock: Option<String>,
    /// Bound (ticks) for letting a design without future source changes settle. Default 100 000.
    pub max_settle_ticks: u64,
}

impl Default for KernelOptions {
    fn default() -> Self {
        KernelOptions {
            timing: TimingOptions::default(),
            fast: FastOptions::default(),
            backend: FastBackend::default(),
            compiled: FastCompileOptions::default(),
            background: true,
            sta: StaOptions::default(),
            cycle_clock: None,
            max_settle_ticks: 100_000,
        }
    }
}

/// Result of [`Kernel::run`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunStats {
    /// Wall-clock time spent.
    pub elapsed: Duration,
    /// Ticks advanced.
    pub ticks: u64,
    /// Rising edges of the reference clock passed.
    pub cycles: u64,
    /// Steps taken (cycles, or phases for designs without a clock).
    pub steps: u64,
    /// Mode at the end of the run.
    pub mode: Mode,
    /// Stopped before the budget because nothing could change any more.
    pub idle: bool,
}

/// A snapshot of the simulation state ([`Kernel::snapshot`]). Restorable into the same design
/// or, by net name, into an edited one; in either mode.
#[derive(Clone, Debug)]
pub struct State {
    tick: u64,
    mode: Mode,
    flat: Arc<FlatNetlist>,
    values: Vec<Logic>,
    /// Input values the values were computed with.
    inputs: Vec<(String, Value)>,
    /// Input changes given but not applied yet (re-applied with `set_input` on restore).
    pending: Vec<(String, Value)>,
}

impl State {
    /// Tick of the snapshot.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Mode the snapshot was taken in.
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Value of a net (any hierarchical alias) in the snapshot.
    pub fn get(&self, name: &str) -> Option<Logic> {
        self.flat.net_by_name(name).map(|n| self.values[n as usize])
    }

    /// Input pin values the snapshot's net values were computed with.
    pub fn inputs(&self) -> &[(String, Value)] {
        &self.inputs
    }

    /// Input changes that were given but not yet applied when the snapshot was taken.
    pub fn pending_inputs(&self) -> &[(String, Value)] {
        &self.pending
    }

    /// Every net value (index = net of the snapshot's netlist).
    pub fn values(&self) -> &[Logic] {
        &self.values
    }

    /// Overwrite the value of a net (any hierarchical alias) in the snapshot, e.g. to poke a
    /// state before [`Kernel::restore`]. Returns false if no net has that name.
    pub fn set(&mut self, name: &str, value: Logic) -> bool {
        match self.flat.net_by_name(name) {
            Some(n) => {
                self.values[n as usize] = value;
                true
            }
            None => false,
        }
    }
}

/// Per-clock timing information of a [`TimingSummary`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClockTiming {
    /// Clock net name.
    pub net: String,
    /// Configured period (ticks).
    pub period: u32,
    /// STA minimum period (`u32::MAX` if its events never settle).
    pub min_period: Option<u32>,
    /// Suggested default period: the STA minimum (even, ≥ 2), SPEC §9.4.
    pub suggested_period: Option<u32>,
    /// `period ≥ min_period`.
    pub meets_timing: bool,
}

/// Result of [`Kernel::timing_report`]: the STA report plus per-clock period checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimingSummary {
    /// The full report of `panda_sta`.
    pub report: TimingReport,
    /// Per running clock.
    pub clocks: Vec<ClockTiming>,
    /// Warnings for clocks configured below their minimum period (also in
    /// [`Kernel::diagnostics`]).
    pub warnings: Vec<Diagnostic>,
}

impl TimingSummary {
    /// The suggested period of a clock net.
    pub fn suggested_period(&self, clock: &str) -> Option<u32> {
        self.clocks
            .iter()
            .find(|c| c.net == clock)
            .and_then(|c| c.suggested_period)
    }

    /// True if STA found no error and every clock meets its minimum period: Fast mode equals
    /// Timing mode at phase boundaries.
    pub fn is_clean(&self) -> bool {
        self.report.is_clean() && self.clocks.iter().all(|c| c.meets_timing)
    }
}

/// The loaded design and everything derived from it.
struct Loaded {
    design: Design,
    flat: Arc<FlatNetlist>,
    schedule: Schedule,
    hash: u64,
    /// Kernels and tables of `flat`, once compiled.
    art: Option<Artifacts>,
    sta: Option<TimingSummary>,
}

enum Active {
    Timing(Box<TimingEngine>),
    Fast {
        engine: Box<dyn FastEngine>,
        backend: FastBackend,
    },
}

type CompileResult = Result<(Box<dyn FastEngine>, Artifacts), Vec<Diagnostic>>;

struct WorkerMsg {
    generation: u64,
    backend: FastBackend,
    last: bool,
    result: CompileResult,
}

/// The simulator facade (SPEC §10). See the crate documentation.
pub struct Kernel {
    opts: KernelOptions,
    loaded: Option<Loaded>,
    mode: Mode,
    active: Option<Active>,
    /// A compiled engine for the current design that is not in use (Timing mode).
    standby: Option<(Box<dyn FastEngine>, FastBackend)>,
    /// Input pin values by name (kept across edits and mode switches).
    inputs: HashMap<String, Value>,
    /// Input values the active engine has applied (`inputs` minus pending changes).
    applied: HashMap<String, Value>,
    /// Input changes queued in the Fast engine and not processed yet.
    fast_pending: bool,
    /// Nothing happened since `load` (a fresh Fast engine can be used as is).
    pristine: bool,
    diags: DiagStore,
    cache: Arc<Mutex<TableCache>>,
    /// The JIT tier of [`FastBackend::Jit`] (its function cache persists across edits).
    jit: JitTier,
    generation: u64,
    /// `generation`, shared with the compiler threads (stale jobs stop early).
    current: Arc<AtomicU64>,
    /// Generation whose final compile stage is outstanding.
    compiling: Option<u64>,
    tx: Sender<WorkerMsg>,
    rx: Receiver<WorkerMsg>,
}

use diag::DiagStore;

impl Default for Kernel {
    fn default() -> Self {
        Kernel::new(KernelOptions::default())
    }
}

fn lock(cache: &Mutex<TableCache>) -> MutexGuard<'_, TableCache> {
    cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Content hash of a design, ignoring editor data.
fn design_hash(d: &Design) -> u64 {
    let mut d = d.clone();
    for def in &mut d.defs {
        def.editor = serde_json::Value::Null;
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    serde_json::to_string(&d).unwrap_or_default().hash(&mut h);
    h.finish()
}

/// The defs reachable from `top` (in their original order), with `top` as the top.
fn subdesign(design: &Design, top: &str) -> Design {
    let mut keep = std::collections::HashSet::new();
    let mut stack = vec![top.to_string()];
    while let Some(id) = stack.pop() {
        if !keep.insert(id.clone()) {
            continue;
        }
        if let Some(d) = design.def(&id) {
            for i in &d.instances {
                if let panda_netlist::InstKind::Def(c) = &i.kind {
                    stack.push(c.clone());
                }
            }
        }
    }
    Design {
        format: design.format.clone(),
        defs: design
            .defs
            .iter()
            .filter(|d| keep.contains(&d.id))
            .cloned()
            .collect(),
        top: top.to_string(),
    }
}

/// Values of `new` nets carried from `old` by net name (canonical names first, then aliases).
fn carry_values(old: &FlatNetlist, old_values: &[Logic], new: &FlatNetlist) -> Vec<Option<Logic>> {
    let mut out: Vec<Option<Logic>> = vec![None; new.num_nets as usize];
    for (n, name) in new.net_names.iter().enumerate() {
        if let Some(o) = old.net_by_name(name) {
            out[n] = Some(old_values[o as usize]);
        }
    }
    for (name, &n) in &new.name_index {
        if out[n as usize].is_none()
            && let Some(o) = old.net_by_name(name)
        {
            out[n as usize] = Some(old_values[o as usize]);
        }
    }
    out
}

/// One background compilation.
struct CompileJob {
    flat: Arc<FlatNetlist>,
    fopts: FastOptions,
    copts: panda_fast::CompileOptions,
    backend: FastBackend,
    cache: Arc<Mutex<TableCache>>,
    art: Option<Artifacts>,
    jit: JitTier,
    /// The job's generation and the kernel's current one: once they differ, the result can never
    /// be used, so the job stops at the next stage boundary without sending anything.
    generation: u64,
    current: Arc<AtomicU64>,
}

impl CompileJob {
    fn stale(&self) -> bool {
        self.current.load(Ordering::Acquire) != self.generation
    }

    /// Compile kernels (unless given) and build the engines of `backend` (Reference first).
    /// Returns early (without emitting) when the job became stale.
    fn run(self, mut emit: impl FnMut(FastBackend, bool, CompileResult)) {
        if self.stale() {
            return;
        }
        let art = match self.art.clone() {
            Some(a) => a,
            None => {
                let kopts = CompileOptions {
                    extract: self.fopts.extract.clone(),
                    charz: self.fopts.charz.clone(),
                };
                let (ks, tables) = {
                    let cache = lock(&self.cache);
                    // Checked again under the lock: stale jobs queued behind a long
                    // characterization do not repeat it.
                    if self.stale() {
                        return;
                    }
                    panda_compile::compile_kernels_with_cache(&self.flat, &kopts, &cache)
                };
                Artifacts {
                    kernels: Arc::new(ks),
                    tables: Arc::new(tables),
                }
            }
        };
        if self.stale() {
            return;
        }
        let last = self.backend == FastBackend::Reference;
        let r = build_engine(
            FastBackend::Reference,
            self.flat.clone(),
            &self.fopts,
            &self.copts,
            &art,
            &self.jit,
        )
        .map(|e| (e, art.clone()));
        let failed = r.is_err();
        emit(FastBackend::Reference, last || failed, r);
        if last || failed || self.stale() {
            return;
        }
        let r = build_engine(
            self.backend,
            self.flat,
            &self.fopts,
            &self.copts,
            &art,
            &self.jit,
        )
        .map(|e| (e, art));
        emit(self.backend, true, r);
    }
}

impl Kernel {
    /// A kernel without a design.
    pub fn new(opts: KernelOptions) -> Kernel {
        let (tx, rx) = channel();
        let jit = JitTier::new(opts.background);
        Kernel {
            jit,
            opts,
            loaded: None,
            mode: Mode::Timing,
            active: None,
            standby: None,
            inputs: HashMap::new(),
            applied: HashMap::new(),
            fast_pending: false,
            pristine: true,
            diags: DiagStore::default(),
            cache: Arc::new(Mutex::new(TableCache::new())),
            generation: 0,
            current: Arc::new(AtomicU64::new(0)),
            compiling: None,
            tx,
            rx,
        }
    }

    /// The options.
    pub fn options(&self) -> &KernelOptions {
        &self.opts
    }

    // -----------------------------------------------------------------------------------------
    // Loading and editing
    // -----------------------------------------------------------------------------------------

    /// Load a design (validated and flattened; on error nothing changes). Powers on at tick 0 in
    /// the current mode. Input pin values set earlier are kept where the pin still exists.
    pub fn load(&mut self, design: Design) -> Result<(), Vec<Diagnostic>> {
        let flat = self.check(&design)?;
        self.install(design, flat);
        self.pristine = true;
        let inputs = self.input_vec();
        let eng = TimingEngine::power_on(self.flat().clone(), &self.opts.timing, &inputs);
        self.applied = self.inputs.clone();
        self.active = Some(Active::Timing(Box::new(eng)));
        if self.mode == Mode::Fast {
            self.start_compile();
            self.poll(false).ok();
        }
        Ok(())
    }

    /// Apply a batch of edits (SPEC §10 incremental edits). All operations are applied to a copy
    /// of the design, which is then validated and flattened; any error rolls the whole batch back
    /// and is returned (the simulation is untouched). On success the simulation continues at the
    /// same tick with the net values carried over by name (see the crate docs, "Edits").
    pub fn edit(&mut self, ops: &[EditOp]) -> Result<(), Vec<Diagnostic>> {
        let Some(loaded) = &self.loaded else {
            return Err(vec![Diagnostic::error(DiagKind::Other, "no design loaded")]);
        };
        let mut design = loaded.design.clone();
        edit::apply(&mut design, ops).map_err(|e| vec![e])?;
        let old_hash = loaded.hash;
        let flat = self.check(&design)?;
        if design_hash(&design) == old_hash {
            self.loaded.as_mut().expect("loaded").design = design;
            return Ok(());
        }
        self.flush_fast_inputs();
        let tick = self.tick();
        let old_flat = self.flat().clone();
        let old_values = self.current_values();
        self.drain_engine();
        let carried = carry_values(&old_flat, &old_values, &flat);
        self.install(design, flat);
        self.pristine = false;
        let flat = self.flat().clone();
        // The carried values were computed with the applied inputs; input changes given but not
        // applied yet (Timing mode) still take effect at the next tick, as they would have.
        let applied = self.input_vec_from(&self.applied);
        let pending = self.pending_inputs();
        let sched = &self.loaded.as_ref().expect("loaded").schedule;
        let mut eng = TimingEngine::load(flat, &self.opts.timing, sched, tick, &carried, &applied);
        for (pin, v) in &pending {
            let _ = eng.sim.set_input(pin, v);
        }
        self.fast_pending = false;
        self.active = Some(Active::Timing(Box::new(eng)));
        if self.mode == Mode::Fast {
            self.start_compile();
            self.poll(false).ok();
        }
        Ok(())
    }

    /// Validate and flatten; errors are returned, warnings go to the diagnostics.
    fn check(&mut self, design: &Design) -> Result<Arc<FlatNetlist>, Vec<Diagnostic>> {
        let diags = validate(design);
        if has_errors(&diags) {
            return Err(diags
                .into_iter()
                .filter(|d| d.severity == Severity::Error)
                .collect());
        }
        let flat = flatten(design)?;
        let tick = self.tick();
        self.diags.clear_source(DiagSource::Design);
        self.diags.extend(DiagSource::Design, tick, diags);
        Ok(Arc::new(flat))
    }

    /// Make `design` / `flat` current; drops everything derived from the previous design.
    fn install(&mut self, design: Design, flat: Arc<FlatNetlist>) {
        let schedule = Schedule::new(&flat, self.opts.cycle_clock.as_deref());
        let hash = design_hash(&design);
        self.loaded = Some(Loaded {
            design,
            flat,
            schedule,
            hash,
            art: None,
            sta: None,
        });
        self.generation += 1;
        self.current.store(self.generation, Ordering::Release);
        self.compiling = None;
        self.standby = None;
        self.fast_pending = false;
        self.diags.clear_source(DiagSource::Sta);
        // Compiler errors of an earlier design are stale (a new compilation reports its own).
        self.diags.clear_source(DiagSource::Compile);
        let flat = self.flat().clone();
        let keep = |name: &String, v: &mut Value| {
            flat.top_inputs
                .iter()
                .any(|p| p.name == *name && p.nets.len() as u32 == v.width())
        };
        self.inputs.retain(keep);
        self.applied.retain(keep);
    }

    /// Input changes given but not applied by the active engine yet (sorted by pin name).
    fn pending_inputs(&self) -> Vec<(String, Value)> {
        let mut pending: Vec<(String, Value)> = self
            .inputs
            .iter()
            .filter(|(k, v)| self.applied.get(*k) != Some(*v))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        pending.sort_by(|a, b| a.0.cmp(&b.0));
        pending
    }

    /// The loaded design.
    pub fn design(&self) -> Option<&Design> {
        self.loaded.as_ref().map(|l| &l.design)
    }

    /// The flattened netlist of the loaded design.
    pub fn netlist(&self) -> Option<&Arc<FlatNetlist>> {
        self.loaded.as_ref().map(|l| &l.flat)
    }

    fn flat(&self) -> &Arc<FlatNetlist> {
        &self.loaded.as_ref().expect("design loaded").flat
    }

    fn schedule(&self) -> &Schedule {
        &self.loaded.as_ref().expect("design loaded").schedule
    }

    // -----------------------------------------------------------------------------------------
    // Modes and background compilation
    // -----------------------------------------------------------------------------------------

    /// The current mode.
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// The engine currently serving requests.
    pub fn engine(&self) -> EngineKind {
        match &self.active {
            None => EngineKind::None,
            Some(Active::Timing(_)) if self.mode == Mode::Fast => EngineKind::TimingFallback,
            Some(Active::Timing(_)) => EngineKind::Timing,
            Some(Active::Fast { backend, .. }) => EngineKind::Fast(*backend),
        }
    }

    /// Switch modes, transferring the state (see the crate docs, "Mode switches").
    pub fn set_mode(&mut self, mode: Mode) {
        if mode == self.mode {
            return;
        }
        self.mode = mode;
        if self.loaded.is_none() {
            return;
        }
        match mode {
            Mode::Fast => {
                self.advance_to_phase_end();
                if let Some((engine, backend)) = self.standby.take() {
                    self.timing_to_fast(engine, backend);
                } else if self.compiling.is_none() {
                    self.start_compile();
                }
                self.poll(false).ok();
            }
            Mode::Timing => {
                if matches!(self.active, Some(Active::Fast { .. })) {
                    self.fast_to_timing();
                }
            }
        }
    }

    /// True if Fast mode is served by a Fast engine (not by the Timing fallback), or the mode is
    /// Timing.
    pub fn is_ready(&mut self) -> bool {
        self.poll(false).ok();
        !(self.mode == Mode::Fast && matches!(self.active, Some(Active::Timing(_))))
    }

    /// Block until the outstanding background compilation (if any) is finished and its engine
    /// is in use (Fast mode) or ready (Timing mode). Returns the compiler's errors, if it failed.
    pub fn wait_ready(&mut self) -> Result<(), Vec<Diagnostic>> {
        self.poll(true)
    }

    /// True if the active Fast engine runs native (JIT) code ([`FastBackend::Jit`] after its code
    /// was hot-swapped in). Installs code that became ready since the last event.
    pub fn jit_active(&mut self) -> bool {
        self.poll(false).ok();
        match &mut self.active {
            Some(Active::Fast { engine, .. }) => engine.native_active(),
            _ => false,
        }
    }

    /// Wait (up to `timeout`) until the Fast engine runs native code ([`Kernel::wait_ready`]
    /// first); false if it does not by then (other backend, Timing mode, or JIT failure).
    pub fn wait_jit(&mut self, timeout: Duration) -> bool {
        if self.wait_ready().is_err() {
            return false;
        }
        let t0 = Instant::now();
        loop {
            if self.jit_active() {
                return true;
            }
            let pending = match &mut self.active {
                Some(Active::Fast { engine, .. }) => engine.native_pending(),
                _ => false,
            };
            if !pending {
                // Finished meanwhile (installed by `native_pending`), failed, or not started.
                return self.jit_active();
            }
            if self.opts.backend != FastBackend::Jit || t0.elapsed() >= timeout {
                return false;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Compile the Fast-mode program for the current design in the background now (so a later
    /// switch to Fast mode is immediate).
    pub fn precompile(&mut self) {
        if self.loaded.is_some()
            && self.compiling.is_none()
            && self.standby.is_none()
            && self.mode == Mode::Timing
        {
            self.start_compile();
        }
    }

    fn start_compile(&mut self) {
        let loaded = self.loaded.as_ref().expect("design loaded");
        let generation = self.generation;
        self.compiling = Some(generation);
        let backend = self.opts.backend;
        let compile = CompileJob {
            flat: loaded.flat.clone(),
            fopts: self.opts.fast.clone(),
            copts: self.opts.compiled.clone(),
            backend,
            cache: self.cache.clone(),
            art: loaded.art.clone(),
            jit: self.jit.clone(),
            generation,
            current: self.current.clone(),
        };
        let tx = self.tx.clone();
        let job = move || {
            let sent_last = std::cell::Cell::new(false);
            let run = std::panic::AssertUnwindSafe(|| {
                compile.run(|backend, last, result| {
                    sent_last.set(last);
                    let _ = tx.send(WorkerMsg {
                        generation,
                        backend,
                        last,
                        result,
                    });
                });
            });
            if std::panic::catch_unwind(run).is_err() && !sent_last.get() {
                // Never leave `wait_ready` waiting for a result that will not come.
                let err = Diagnostic::error(DiagKind::Other, "the Fast-mode compiler panicked");
                let _ = tx.send(WorkerMsg {
                    generation,
                    backend,
                    last: true,
                    result: Err(vec![err]),
                });
            }
        };
        if self.opts.background {
            std::thread::Builder::new()
                .name("panda-compile".into())
                .spawn(job)
                .expect("spawn the compiler thread");
        } else {
            job();
        }
    }

    fn loaded_art(&self) -> Option<Artifacts> {
        self.loaded.as_ref().and_then(|l| l.art.clone())
    }

    /// Receive compiler results; with `block`, wait for the final stage of the current
    /// generation.
    fn poll(&mut self, block: bool) -> Result<(), Vec<Diagnostic>> {
        let mut result = Ok(());
        loop {
            let msg = if block && self.compiling == Some(self.generation) {
                match self.rx.recv() {
                    Ok(m) => m,
                    Err(_) => break,
                }
            } else {
                match self.rx.try_recv() {
                    Ok(m) => m,
                    Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
                }
            };
            if msg.generation != self.generation || self.loaded.is_none() {
                continue;
            }
            if msg.last {
                self.compiling = None;
            }
            match msg.result {
                Ok((engine, art)) => {
                    self.loaded.as_mut().expect("loaded").art = Some(art);
                    self.adopt(engine, msg.backend);
                }
                Err(errs) => {
                    let tick = self.tick();
                    self.diags.extend(DiagSource::Compile, tick, errs.clone());
                    result = Err(errs);
                }
            }
        }
        result
    }

    /// A freshly compiled engine (power-on state) for the current design arrived.
    fn adopt(&mut self, engine: Box<dyn FastEngine>, backend: FastBackend) {
        if self.mode == Mode::Timing {
            self.standby = Some((engine, backend));
            return;
        }
        match self.active.take() {
            Some(Active::Timing(t)) => {
                self.active = Some(Active::Timing(t));
                let default_inputs = self.inputs.values().all(|v| *v == Value::zeros(v.width()));
                if self.pristine && default_inputs {
                    // Nothing happened since load: the engine's own power-on state is exact.
                    self.drain_engine();
                    self.active = Some(Active::Fast { engine, backend });
                    self.fast_pending = false;
                    self.applied = self.inputs.clone();
                } else {
                    self.advance_to_phase_end();
                    self.timing_to_fast(engine, backend);
                }
            }
            Some(Active::Fast {
                engine: mut old,
                backend: old_backend,
            }) => {
                // Hot swap at the current phase end.
                if self.fast_pending {
                    old.step_event();
                    self.fast_pending = false;
                    self.applied = self.inputs.clone();
                }
                let tick = self.tick_of_fast(old.as_ref());
                let values = old.values().to_vec();
                let t = self.tick();
                self.diags
                    .extend(DiagSource::Fast, t, old.take_diagnostics());
                let mut engine = engine;
                let inputs = self.input_vec();
                // The new engine's power-on diagnostics belong to tick 0, not to this design state.
                engine.take_diagnostics();
                let r = engine.load_state(tick, &values, &inputs);
                self.report_load("engine swap", tick, r);
                let _ = old_backend;
                self.active = Some(Active::Fast { engine, backend });
            }
            None => {}
        }
    }

    /// Advance the Timing engine to the end of the current phase (no-op at a phase end, or if
    /// the active engine is a Fast engine).
    fn advance_to_phase_end(&mut self) {
        let Some(Active::Timing(_)) = &self.active else {
            return;
        };
        let now = self.tick();
        let target = match self.schedule().next_change_after(now) {
            Some(e) => Target::Tick(e - 1),
            None => Target::Settle,
        };
        self.advance_timing(target);
    }

    /// Timing → Fast state transfer at the current tick (a phase end).
    fn timing_to_fast(&mut self, mut engine: Box<dyn FastEngine>, backend: FastBackend) {
        let tick = self.tick();
        if let Some(Active::Timing(t)) = &self.active
            && t.busy()
        {
            // SPEC §10: Timing → Fast is legal at quiescence. Defined behaviour otherwise: the
            // switch happens at the phase end and kernels caught mid-transition finish it
            // (`load_state`); the user is told the state was not quiescent.
            self.diags.push(
                DiagSource::Kernel,
                tick,
                Diagnostic::warning(
                    DiagKind::Other,
                    "Timing → Fast: the Timing state at the phase end is not quiescent (gate events \
                     pending: the phase is shorter than the settling time, or the design \
                     oscillates); Fast mode settles it with kernel-delta semantics",
                ),
            );
        }
        let values = self.current_values();
        self.drain_engine();
        let pending = self.pending_inputs();
        let applied = self.input_vec_from(&self.applied);
        // A fresh engine's power-on diagnostics belong to tick 0, not to this state.
        engine.take_diagnostics();
        let r = engine.load_state(tick, &values, &applied);
        self.report_load("Timing → Fast", tick, r);
        self.fast_pending = false;
        for (pin, v) in pending {
            if engine.set_input(&pin, &v).is_ok() {
                self.fast_pending = true;
            }
        }
        self.active = Some(Active::Fast { engine, backend });
    }

    /// Let the Fast engine process queued input changes (as their own event at its current tick).
    fn flush_fast_inputs(&mut self) {
        if let Some(Active::Fast { engine, .. }) = &mut self.active
            && self.fast_pending
        {
            engine.step_event();
            self.fast_pending = false;
            self.applied = self.inputs.clone();
        }
    }

    /// Report kernels that did not load as they were. The messages carry no tick (the entry
    /// records first/last tick and a count), so repeated occurrences deduplicate.
    fn report_load(&mut self, what: &str, tick: u64, r: LoadReport) {
        let settled = r.inconsistent - r.fallen;
        if settled > 0 {
            self.diags.push(
                DiagSource::Kernel,
                tick,
                Diagnostic::warning(
                    DiagKind::Other,
                    format!(
                        "{what}: kernel state(s) inconsistent with their inputs (a state taken before \
                         quiescence, or poked) finished their transition under the current inputs"
                    ),
                ),
            );
        }
        if r.fallen > 0 {
            self.diags.push(
                DiagSource::Kernel,
                tick,
                Diagnostic::warning(
                    DiagKind::Other,
                    format!(
                        "{what}: kernel state(s) inconsistent with their inputs did not settle and restarted from X"
                    ),
                ),
            );
        }
    }

    /// Fast → Timing state transfer; the Fast engine goes to standby.
    fn fast_to_timing(&mut self) {
        self.flush_fast_inputs();
        let Some(Active::Fast {
            mut engine,
            backend,
        }) = self.active.take()
        else {
            return;
        };
        let tick = self.tick_of_fast(engine.as_ref());
        let t = tick;
        self.diags
            .extend(DiagSource::Fast, t, engine.take_diagnostics());
        let values: Vec<Option<Logic>> = engine.values().iter().map(|&v| Some(v)).collect();
        let inputs = self.input_vec();
        let eng = TimingEngine::load(
            self.flat().clone(),
            &self.opts.timing,
            self.schedule(),
            tick,
            &values,
            &inputs,
        );
        self.applied = self.inputs.clone();
        self.active = Some(Active::Timing(Box::new(eng)));
        self.standby = Some((engine, backend));
    }

    fn tick_of_fast(&self, e: &dyn FastEngine) -> u64 {
        match e.next_event_tick() {
            Some(n) => (n - 1).max(e.tick()),
            None => e.tick(),
        }
    }

    // -----------------------------------------------------------------------------------------
    // Inputs, stepping, reading
    // -----------------------------------------------------------------------------------------

    fn input_vec_from(&self, map: &HashMap<String, Value>) -> Vec<Value> {
        self.flat()
            .top_inputs
            .iter()
            .map(|p| {
                let w = p.nets.len() as u32;
                map.get(&p.name)
                    .filter(|v| v.width() == w)
                    .cloned()
                    .unwrap_or_else(|| Value::zeros(w))
            })
            .collect()
    }

    fn input_vec(&self) -> Vec<Value> {
        self.input_vec_from(&self.inputs)
    }

    /// Set a top-level input pin. Timing mode applies it at the next tick; Fast mode processes it
    /// as its own event at the next step (SPEC §6.2).
    pub fn set_input(&mut self, pin: &str, value: &Value) -> Result<(), String> {
        let Some(loaded) = &self.loaded else {
            return Err("no design loaded".into());
        };
        let p = loaded
            .flat
            .top_inputs
            .iter()
            .find(|p| p.name == pin)
            .ok_or_else(|| format!("no top-level input pin '{pin}'"))?;
        if p.nets.len() as u32 != value.width() {
            return Err(format!(
                "input pin '{pin}' has width {}, got a {}-bit value",
                p.nets.len(),
                value.width()
            ));
        }
        let current = self
            .inputs
            .get(pin)
            .cloned()
            .unwrap_or_else(|| Value::zeros(value.width()));
        if current == *value {
            return Ok(());
        }
        self.pristine = false;
        self.inputs.insert(pin.to_string(), value.clone());
        match &mut self.active {
            Some(Active::Timing(t)) => t.sim.set_input(pin, value)?,
            Some(Active::Fast { engine, .. }) => {
                engine.set_input(pin, value)?;
                self.fast_pending = true;
            }
            None => {}
        }
        Ok(())
    }

    /// The value last given to an input pin (default 0).
    pub fn input(&self, pin: &str) -> Option<Value> {
        let p = self
            .loaded
            .as_ref()?
            .flat
            .top_inputs
            .iter()
            .find(|p| p.name == pin)?;
        Some(
            self.inputs
                .get(pin)
                .cloned()
                .unwrap_or_else(|| Value::zeros(p.nets.len() as u32)),
        )
    }

    /// Current tick (see the crate docs, "Time"). 0 without a design.
    pub fn tick(&self) -> u64 {
        match &self.active {
            None => 0,
            Some(Active::Timing(t)) => t.tick(),
            Some(Active::Fast { engine, .. }) => self.tick_of_fast(engine.as_ref()),
        }
    }

    /// Advance the simulation by `n` units (see the crate docs, "Time").
    pub fn step(&mut self, n: u64, unit: StepUnit) {
        self.step_counted(n, unit);
    }

    /// [`Kernel::step`], returning the number of units that advanced the time (`Phase` /
    /// `Cycle`; for `Tick`: the ticks advanced).
    fn step_counted(&mut self, n: u64, unit: StepUnit) -> u64 {
        if self.active.is_none() {
            return 0;
        }
        self.poll(false).ok();
        let start = self.tick();
        let mut done = 0;
        match unit {
            StepUnit::Tick => {
                let t = start.saturating_add(n);
                self.advance(Target::Tick(t));
                done = self.tick() - start;
            }
            StepUnit::Phase | StepUnit::Cycle => {
                for _ in 0..n {
                    let before = self.tick();
                    let target = self.unit_target(unit);
                    self.advance(target);
                    if self.tick() > before {
                        done += 1;
                    } else if target == Target::Settle {
                        break;
                    }
                }
            }
        }
        self.poll(false).ok();
        done
    }

    /// Advance to facade tick `tick` (Fast mode: to the end of the phase containing it).
    pub fn run_to(&mut self, tick: u64) {
        if self.active.is_none() {
            return;
        }
        self.poll(false).ok();
        self.advance(Target::Tick(tick));
        self.poll(false).ok();
    }

    fn unit_target(&self, unit: StepUnit) -> Target {
        let now = self.tick();
        let s = self.schedule();
        let edge = match unit {
            StepUnit::Cycle => s
                .next_cycle_edge_after(now + 1)
                .or_else(|| s.next_change_after(now + 1)),
            _ => s.next_change_after(now + 1),
        };
        match edge {
            Some(e) => Target::Tick(e - 1),
            None => Target::Settle,
        }
    }

    /// Run until `budget` expires (or nothing can change any more), in steps of whole cycles
    /// (phases without a clock), sized so the budget is not overrun by more than about one step.
    pub fn run(&mut self, budget: Duration) -> RunStats {
        let start = Instant::now();
        let mut stats = RunStats {
            elapsed: Duration::ZERO,
            ticks: 0,
            cycles: 0,
            steps: 0,
            mode: self.mode,
            idle: false,
        };
        if self.active.is_none() {
            stats.idle = true;
            return stats;
        }
        let unit = if self.schedule().cycle_clock.is_some() {
            StepUnit::Cycle
        } else {
            StepUnit::Phase
        };
        let t0 = self.tick();
        let mut chunk: u64 = 1;
        loop {
            let elapsed = start.elapsed();
            if elapsed >= budget {
                break;
            }
            let remaining = budget - elapsed;
            let before = self.tick();
            let c0 = Instant::now();
            stats.steps += self.step_counted(chunk, unit);
            let dt = c0.elapsed();
            if self.tick() == before {
                stats.idle = true;
                break;
            }
            // Size the next chunk to about half the remaining time.
            let per = dt.as_nanos().max(1) / u128::from(chunk);
            let left = remaining.saturating_sub(dt).as_nanos() / 2;
            chunk = u64::try_from(left / per)
                .unwrap_or(u64::MAX)
                .clamp(1, chunk.saturating_mul(2));
        }
        let t1 = self.tick();
        stats.ticks = t1 - t0;
        stats.cycles = self.schedule().cycle_edges_between(t0, t1);
        stats.mode = self.mode;
        stats.elapsed = start.elapsed();
        stats
    }

    fn advance(&mut self, target: Target) {
        match &self.active {
            Some(Active::Timing(_)) => {
                let target = if self.mode == Mode::Fast {
                    // Fallback for Fast mode: phase granularity.
                    match target {
                        Target::Tick(t) => match self.schedule().next_change_after(t) {
                            Some(e) => Target::Tick(e - 1),
                            None => Target::TickThenSettle(t),
                        },
                        Target::Settle | Target::TickThenSettle(_) => target,
                    }
                } else {
                    target
                };
                self.advance_timing(target);
            }
            Some(Active::Fast { .. }) => self.advance_fast(target),
            None => {}
        }
        self.pristine = false;
    }

    fn advance_timing(&mut self, target: Target) {
        let max = self.opts.max_settle_ticks;
        let Some(Active::Timing(t)) = &mut self.active else {
            return;
        };
        let before = t.tick();
        match target {
            Target::Tick(tick) => t.run_to(tick),
            Target::Settle => {
                t.sim.settle(max);
            }
            Target::TickThenSettle(tick) => {
                t.run_to(tick);
                t.sim.settle(max);
            }
        }
        if t.tick() > before {
            // Input changes take effect at the first step.
            self.applied = self.inputs.clone();
        }
        self.drain_engine();
    }

    fn advance_fast(&mut self, target: Target) {
        self.flush_fast_inputs();
        let Some(Active::Fast { engine, .. }) = &mut self.active else {
            return;
        };
        if let Target::Tick(t) | Target::TickThenSettle(t) = target {
            while engine.next_event_tick().is_some_and(|e| e <= t) {
                engine.step_event();
            }
        }
        self.drain_engine();
    }

    /// Move the active engine's diagnostics into the unified list.
    fn drain_engine(&mut self) {
        let tick = self.tick();
        match &mut self.active {
            Some(Active::Timing(t)) => {
                let d = t.take_diagnostics();
                self.diags.extend(DiagSource::Timing, tick, d);
            }
            Some(Active::Fast { engine, .. }) => {
                let d = engine.take_diagnostics();
                self.diags.extend(DiagSource::Fast, tick, d);
            }
            None => {}
        }
    }

    /// Read a net by hierarchical name (works inside packages, e.g. `"cpu/alu/c[3]"`), a
    /// multi-bit net by base name (`"cpu/pc"`), a package pin (`"cpu/alu/A"`) or a top-level pin.
    /// `None` if nothing has that name.
    pub fn read(&self, net: &str) -> Option<Value> {
        match &self.active {
            Some(Active::Timing(t)) => t.sim.read(net),
            Some(Active::Fast { engine, .. }) => engine.read(net),
            None => None,
        }
    }

    /// [`Kernel::read`] as a number (`None` if unknown or not fully binary).
    pub fn read_u64(&self, net: &str) -> Option<u64> {
        self.read(net)?.to_u64()
    }

    /// Every net value of the active engine (index = net of [`Kernel::netlist`]).
    fn current_values(&self) -> Vec<Logic> {
        match &self.active {
            Some(Active::Timing(t)) => (0..t.sim.netlist().num_nets)
                .map(|n| t.sim.get(n))
                .collect(),
            Some(Active::Fast { engine, .. }) => engine.values().to_vec(),
            None => Vec::new(),
        }
    }

    // -----------------------------------------------------------------------------------------
    // Snapshots
    // -----------------------------------------------------------------------------------------

    /// Snapshot the current state (every net value, the input pins, the tick).
    ///
    /// # Panics
    ///
    /// If no design is loaded.
    pub fn snapshot(&self) -> State {
        let flat = self.flat().clone();
        let mut inputs: Vec<(String, Value)> = self
            .applied
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        inputs.sort_by(|a, b| a.0.cmp(&b.0));
        let pending = self.pending_inputs();
        State {
            tick: self.tick(),
            mode: self.mode,
            flat,
            values: self.current_values(),
            inputs,
            pending,
        }
    }

    /// Restore a snapshot (into the current mode). Nets are matched by name, so a snapshot of an
    /// earlier version of the design restores what still exists (other nets start `X`). A
    /// Timing-mode snapshot restored in Fast mode first finishes its phase in the Timing engine
    /// (see the crate docs, "Snapshots"), so the tick afterwards is that phase's end.
    pub fn restore(&mut self, s: &State) -> Result<(), Vec<Diagnostic>> {
        if self.loaded.is_none() {
            return Err(vec![Diagnostic::error(DiagKind::Other, "no design loaded")]);
        }
        self.drain_engine();
        self.poll(false).ok();
        let flat = self.flat().clone();
        let carried: Vec<Option<Logic>> = if Arc::ptr_eq(&flat, &s.flat) {
            s.values.iter().map(|&v| Some(v)).collect()
        } else {
            carry_values(&s.flat, &s.values, &flat)
        };
        self.inputs.clear();
        for (k, v) in &s.inputs {
            if flat
                .top_inputs
                .iter()
                .any(|p| p.name == *k && p.nets.len() as u32 == v.width())
            {
                self.inputs.insert(k.clone(), v.clone());
            }
        }
        self.applied = self.inputs.clone();
        let inputs = self.input_vec();
        self.pristine = false;
        self.fast_pending = false;
        let engine = match self.active.take() {
            Some(Active::Fast { engine, backend }) => Some((engine, backend)),
            _ if self.mode == Mode::Fast => self.standby.take(),
            _ => None,
        };
        match engine {
            // A Fast-mode snapshot is a settled phase end: load it directly.
            Some((mut engine, backend)) if self.mode == Mode::Fast && s.mode == Mode::Fast => {
                let values: Vec<Logic> = carried.iter().map(|v| v.unwrap_or(Logic::X)).collect();
                engine.take_diagnostics();
                let r = engine.load_state(s.tick, &values, &inputs);
                self.report_load("restore", s.tick, r);
                self.active = Some(Active::Fast { engine, backend });
                for (pin, v) in &s.pending {
                    // Pins that no longer exist (or changed width) are skipped.
                    let _ = self.set_input(pin, v);
                }
            }
            other => {
                let fast_engine = if self.mode == Mode::Fast {
                    other
                } else {
                    if let Some(e) = other {
                        self.standby = Some(e);
                    }
                    None
                };
                let eng = TimingEngine::load(
                    flat,
                    &self.opts.timing,
                    self.schedule(),
                    s.tick,
                    &carried,
                    &inputs,
                );
                self.active = Some(Active::Timing(Box::new(eng)));
                for (pin, v) in &s.pending {
                    let _ = self.set_input(pin, v);
                }
                if self.mode == Mode::Fast {
                    // A Timing snapshot may be anywhere inside a phase, not settled: Timing
                    // semantics finish the phase, then the state transfers as in a mode switch
                    // (or when the compiler is done).
                    self.advance_to_phase_end();
                    if let Some((engine, backend)) = fast_engine {
                        self.timing_to_fast(engine, backend);
                    }
                }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------------------------
    // Diagnostics, classification, timing
    // -----------------------------------------------------------------------------------------

    /// Unified, deduplicated diagnostics (first occurrence of each kind × location × producer).
    pub fn diagnostics(&self) -> &[Diagnostic] {
        self.diags.diagnostics()
    }

    /// [`Kernel::diagnostics`] with producers and occurrence counts.
    pub fn diagnostic_entries(&self) -> &[DiagEntry] {
        self.diags.entries()
    }

    /// Diagnostics of at least the given severity (`Error` < `Warning` < `Info` in verbosity).
    pub fn diagnostics_at_least(&self, severity: Severity) -> Vec<&Diagnostic> {
        let rank = |s: Severity| match s {
            Severity::Error => 0,
            Severity::Warning => 1,
            Severity::Info => 2,
        };
        self.diags
            .diagnostics()
            .iter()
            .filter(|d| rank(d.severity) <= rank(severity))
            .collect()
    }

    /// Forget all diagnostics.
    pub fn clear_diagnostics(&mut self) {
        self.diags.clear();
    }

    /// Classify a def of the loaded design (SPEC §4.4): the def is flattened on its own (as if
    /// it were the top) and analyzed.
    pub fn classify(&mut self, def: &str) -> Result<Classification, Vec<Diagnostic>> {
        let Some(loaded) = &self.loaded else {
            return Err(vec![Diagnostic::error(DiagKind::Other, "no design loaded")]);
        };
        if loaded.design.def(def).is_none() {
            return Err(vec![Diagnostic::error(
                DiagKind::UnknownDef,
                format!("unknown def '{def}'"),
            )]);
        }
        let d = subdesign(&loaded.design, def);
        let flat = Arc::new(flatten(&d)?);
        let mut cache = lock(&self.cache);
        Ok(classify::classify(def, flat, &self.opts.fast, &mut cache))
    }

    /// Static timing analysis of the loaded design (SPEC §9) with per-clock period checks: the
    /// suggested default period of each clock is its STA minimum; a clock configured below it
    /// raises a warning. Results are cached until the design changes; the report's diagnostics
    /// are added to [`Kernel::diagnostics`].
    ///
    /// # Panics
    ///
    /// If no design is loaded.
    pub fn timing_report(&mut self) -> TimingSummary {
        self.poll(false).ok();
        if let Some(s) = &self.loaded.as_ref().expect("design loaded").sta {
            return s.clone();
        }
        let flat = self.flat().clone();
        let art = match self.loaded_art() {
            Some(a) => a,
            None => {
                let kopts = CompileOptions {
                    extract: self.opts.fast.extract.clone(),
                    charz: self.opts.fast.charz.clone(),
                };
                let (ks, tables) =
                    panda_compile::compile_kernels_with_cache(&flat, &kopts, &lock(&self.cache));
                let a = Artifacts {
                    kernels: Arc::new(ks),
                    tables: Arc::new(tables),
                };
                self.loaded.as_mut().expect("loaded").art = Some(a.clone());
                a
            }
        };
        let mut sta_opts = self.opts.sta.clone();
        sta_opts.ideal_blocks = self.opts.timing.ideal_blocks;
        sta_opts.glass_box = self.opts.timing.glass_box.clone();
        let report = panda_sta::analyze_with(&flat, &art.kernels, &art.tables, &sta_opts);
        let mut clocks = Vec::new();
        let mut warnings = Vec::new();
        for s in &flat.sources {
            let SourceKind::Clock { period, high, .. } = s.kind else {
                continue;
            };
            if high == 0 || high >= period {
                continue;
            }
            let net = flat.net_names[s.output as usize].clone();
            let min_period = report.min_period_of(&net);
            let suggested_period = min_period
                .filter(|&m| m != u32::MAX)
                .map(|m| m.max(2).next_multiple_of(2));
            let meets_timing = min_period.is_none_or(|m| period >= m);
            if !meets_timing {
                let min = min_period.unwrap_or(u32::MAX);
                warnings.push(
                    Diagnostic::warning(
                        DiagKind::SetupViolation,
                        format!(
                            "clock {net}: period {period} is below the STA minimum {min}: Fast mode assumes timing \
                             closure; Timing mode will differ"
                        ),
                    )
                    .at(net.clone()),
                );
            }
            clocks.push(ClockTiming {
                net,
                period,
                min_period,
                suggested_period,
                meets_timing,
            });
        }
        let summary = TimingSummary {
            report,
            clocks,
            warnings,
        };
        let tick = self.tick();
        self.diags.clear_source(DiagSource::Sta);
        self.diags.extend(
            DiagSource::Sta,
            tick,
            summary.report.diagnostics.iter().cloned(),
        );
        self.diags
            .extend(DiagSource::Sta, tick, summary.warnings.iter().cloned());
        self.loaded.as_mut().expect("loaded").sta = Some(summary.clone());
        summary
    }
}

/// Where `Kernel::advance` goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Target {
    /// To this tick.
    Tick(u64),
    /// Until nothing changes any more (bounded).
    Settle,
    /// To this tick, then until nothing changes.
    TickThenSettle(u64),
}
