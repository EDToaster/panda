//! [`FastSim`] — the compiled Fast-mode engine (round-faithful executor).
//!
//! # Kernel paths
//!
//! At each round boundary every pending (input-changed, not frozen) kernel transitions once with
//! its current inputs, exactly as in FastRef. How the transition is computed:
//!
//! * **KOpaque** ([`PhaseClass::Opaque`] in the current clock tuple): nothing observable changes;
//!   the kernel's state is implied by its outputs and inputs (lazy). Word-level bitset ops only.
//! * **KTransparent**: the outputs are a table lookup over the input vector; the state is lazy.
//! * **General**: the table entry (state from the reconstruction table if lazy), or the
//!   on-the-fly simulator for `X`/`Z` and missing tables — FastRef's code path.
//!
//! The specialized paths need the kernel to be *fast-ok*: it transitioned before, its last
//! transition used binary inputs equal to its inputs at the previous boundary ("synced"), its
//! state is a binary table state, and its inputs are binary (the interpreter clears the bit when
//! an input becomes `X`/`Z`). The phase classes are proven from the tables for every transition
//! a fast-ok kernel can make in that phase ([`crate::kspec`]), so every single transition gives
//! exactly the table's result, and the statistics count exactly what FastRef counts.
//!
//! Order checks (the `DelaySensitive` diagnostic): a fast-path transition of a kernel whose phase
//! is not *certain* runs FastRef's exact check when two or more inputs flipped (flips are the
//! difference between the kernel's input index and its snapshot at the last boundary); answers
//! are cached per kernel (16 ways) and per table entry.
//!
//! # Which kernels are pending
//!
//! Each net changes at most once between two round boundaries (each op runs at most once per
//! sweep; kernel nets are written once per boundary), so for a synced kernel "an input net
//! changed since the last boundary" is the same as "its input vector differs from its last
//! transition's". Only kernels whose last-transition inputs are recorded explicitly (never
//! transitioned, or frozen at the delta limit) compare against the record, like FastRef. The
//! last-transition inputs of a synced kernel are its inputs at the previous boundary, recovered
//! from the boundary log ([`exec::Marks::boundary_value`]).
//!
//! # Native tier
//!
//! With native code installed ([`FastSim::install_native`], [`crate::native`]), sweeps call the
//! native partition functions, and a scheduled or input event whose clock tuple has a native
//! event function runs its timed-source writes, sweeps and round loop natively until a round
//! boundary needs the general machinery; the loop above then continues from that decision
//! point. Native code works on the same state and has the same effects, so every statistic
//! except `native_*` is identical to the interpreter's.
//!
//! # Lazy nets
//!
//! LUT-internal nets, the outputs of gates fused into other ops ([`crate::fuse`]) and the
//! non-output state nets of lazy kernels are not kept up to date by the hot path.
//! [`FastSim::values`] / [`FastSim::get`] materialize them on demand (from the cone's / fused
//! gates' code and the reconstruction tables) into the same buffer, so `values()` is always exactly
//! FastRef's. The buffer lives in an `UnsafeCell` (written through `&self` only while no borrow
//! of it can exist), which makes `FastSim` `Send` but not `Sync`.

use std::cell::{Cell, UnsafeCell};
use std::collections::HashMap;
use std::sync::Arc;

use panda_compile::subnet::{kernel_subnet, structural_key};
use panda_compile::{
    FastOptions, KernelOptions, KernelSet, KernelSim, KernelTable, TableCache,
    compile_kernels_with_cache,
};
use panda_netlist::eval::terminate;
use panda_netlist::flat::{FlatNetlist, NetIdx, SourceKind};
use panda_netlist::{DiagKind, Diagnostic, Logic, Value};

use crate::compile::{CompileOptions, compile};
use crate::exec::{self, Rt, clear_bit, for_bits, set_bit, test_bit};
use crate::fxhash::FxHashMap;
use crate::ir::{KTableData, KernelOpKind, NONE, Program, VACT_OPAQUE};
use crate::native::{Env, EventCtx, NativeCode, NativeSlot};
use crate::probe;
use crate::schedule::Timed;

#[cfg(doc)]
use crate::ir::PhaseClass;

/// Tunables of [`FastSim`].
#[derive(Clone, Debug, Default)]
pub struct FastSimOptions {
    /// Semantics-relevant options shared with FastRef (delta limit, extraction, characterization).
    pub fast: FastOptions,
    /// Compiler tunables (performance only).
    pub compile: CompileOptions,
}

impl From<FastOptions> for FastSimOptions {
    fn from(fast: FastOptions) -> Self {
        FastSimOptions {
            fast,
            compile: CompileOptions::default(),
        }
    }
}

/// Counters since the last [`FastSim::reset`]. The first six fields count exactly what the
/// corresponding `FastRefStats` fields count, so they are equal to FastRef's for the same run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FastSimStats {
    /// Events processed (including the tick-0 event).
    pub events: u64,
    /// Kernel rounds executed.
    pub rounds: u64,
    /// Most rounds needed by a single event.
    pub max_rounds: u32,
    /// Kernel transitions computed.
    pub transitions: u64,
    /// Transitions answered by a table entry.
    pub table_hits: u64,
    /// Transitions computed on the fly.
    pub sim_fallbacks: u64,
    /// Partitions evaluated.
    pub partitions: u64,
    /// Combinational ops executed (a LUT counts as one).
    pub ops: u64,
    /// Transitions done by the KOpaque path (no work).
    pub opaque_skips: u64,
    /// Transitions done by the KTransparent path.
    pub transparent: u64,
    /// Fast-path transitions that needed per-kernel work (the rest are word-level skips).
    pub fast_evals: u64,
    /// Round boundaries handled by native event functions (part of `rounds`).
    pub native_rounds: u64,
    /// Native event functions that stopped at a round decision point for the interpreter to
    /// continue (a kernel needed the general path, an order check, explicit inputs, or the delta
    /// limit was reached).
    pub native_bails: u64,
}

/// What a kernel transitions to in the current round.
enum NewState {
    /// Table state `id`, reached under binary input vector `v`.
    Table { id: u32, v: u32 },
    /// Explicit state values; `binary_in`: the transition's inputs were binary.
    Explicit { st: Vec<Logic>, binary_in: bool },
    /// All `X`.
    AllX,
}

/// Per-kernel runtime bookkeeping (mirrors FastRef).
#[derive(Clone, Copy)]
struct KState {
    ever: bool,
    /// Explicit table state id (valid when the kernel is not lazy), or [`NONE`].
    state_id: u32,
    sim: u32,
    delay_reported: bool,
    osc_reported: bool,
    meta_reported: bool,
    contention_active: bool,
}

impl KState {
    const POWER_ON: KState = KState {
        ever: false,
        state_id: NONE,
        sim: NONE,
        delay_reported: false,
        osc_reported: false,
        meta_reported: false,
        contention_active: false,
    };
}

/// The compiled Fast-mode engine. Observable behaviour is bit-identical to
/// `panda_compile::FastRef` after every event (values of all nets, tick, schedule, diagnostic
/// kinds); see the module docs for the execution model.
pub struct FastSim {
    flat: Arc<FlatNetlist>,
    opts: FastSimOptions,
    ks: KernelSet,
    tables: Vec<Option<Arc<KernelTable>>>,
    prog: Arc<Program>,
    /// Installed native code (tier 2) for `prog`, if any.
    native: Option<Arc<NativeCode>>,
    /// Mailbox of a background native compiler (polled at every event boundary).
    native_slot: Option<Arc<NativeSlot>>,
    /// Scratch of the native event functions (`EventCtx::pending`): two words per kernel.
    native_pending: Vec<u32>,
    /// Prebuilt contexts of the native event functions.
    native_cache: Option<Box<NativeCache>>,

    timed: Vec<(NetIdx, Timed)>,
    input_srcs: Vec<(NetIdx, u32, u32)>,
    /// Nets without any driver and their (constant) value: `Z`, or their termination's.
    undriven: Vec<(NetIdx, Logic)>,
    sims: Vec<KernelSim>,
    sim_by_key: HashMap<Vec<u32>, u32>,
    /// Order-sensitivity answers `(table, state, u, v)` (the characterization's or computed).
    order_memo: FxHashMap<(u32, u32, u32, u32), bool>,

    /// Net values (see the module docs on lazy nets).
    values: UnsafeCell<Vec<Logic>>,
    /// The lazy nets in `values` are materialized.
    clean: Cell<bool>,
    rt: Rt,
    kst: Vec<KState>,
    /// Kernels whose non-output state nets are stale (state = reconstruction table entry).
    lazy: Vec<u64>,
    /// Kernels whose last-transition inputs are in `last_in` (frozen at the delta limit since)
    /// or that never transitioned.
    unsynced: Vec<u64>,
    /// Kernels frozen for the rest of the current event.
    frozen: Vec<u64>,
    /// Per kernel: its output bits as last written (binary outputs), or [`NONE`].
    kob: Vec<u32>,
    /// Per kernel: the binary input index of its last general-path transition (or [`NONE`]);
    /// the last-transition inputs of a synced kernel without a class.
    glast: Vec<u32>,
    /// Per tuple (`kernel_words` words each): kernels whose last transparent evaluation was in
    /// that tuple.
    tlast: Vec<u64>,
    /// Explicit last-transition inputs (only meaningful for unsynced kernels).
    last_in: Vec<Logic>,
    todo: Vec<u64>,
    general: Vec<u32>,
    /// Per kernel, [`CHECK_WAYS`] direct-mapped slots of fast-path order checks found
    /// insensitive, keyed `(state key, u, v)` (see [`check_key`]).
    check_cache: Vec<(u32, u32, u32)>,
    /// Order checks queued by the fast paths: `(kernel, u, v, state id, state key)`.
    checks: Vec<(u32, u32, u32, u32, u32)>,
    pending: Vec<(u32, NewState)>,
    fast_pending: Vec<(u32, u32)>,
    frozen_list: Vec<u32>,
    input_values: Vec<Value>,
    inputs_pending: bool,
    tick: u64,
    /// Cached [`FastSim::next_event_tick`] (a function of `tick` and the schedule).
    next_tick: Option<u64>,
    /// Per timed source: its next change tick (`u64::MAX`: none).
    tnext: Vec<u64>,
    /// Per timed source: its bit in the clock tuple (`Program::timed_nets`), or [`NONE`].
    timed_bit: Vec<u32>,
    res_reported: Vec<bool>,
    diags: Vec<Diagnostic>,
    stats: FastSimStats,
}

impl FastSim {
    /// Compile `flat` (kernel extraction, characterization, program) and power it on (processes
    /// the tick-0 event).
    pub fn new(flat: Arc<FlatNetlist>, opts: FastSimOptions) -> Result<FastSim, Vec<Diagnostic>> {
        let mut cache = TableCache::new();
        FastSim::with_cache(flat, opts, &mut cache)
    }

    /// [`FastSim::new`] with a caller-provided table cache.
    pub fn with_cache(
        flat: Arc<FlatNetlist>,
        opts: FastSimOptions,
        cache: &mut TableCache,
    ) -> Result<FastSim, Vec<Diagnostic>> {
        let kopts = KernelOptions {
            extract: opts.fast.extract.clone(),
            charz: opts.fast.charz.clone(),
        };
        let (ks, tables) = compile_kernels_with_cache(&flat, &kopts, cache);
        FastSim::from_parts(flat, opts, ks, tables)
    }

    /// Build from an already extracted kernel set and its tables (characterized with
    /// `opts.fast.charz`). Powers on (processes the tick-0 event).
    pub fn from_parts(
        flat: Arc<FlatNetlist>,
        opts: FastSimOptions,
        ks: KernelSet,
        tables: Vec<Option<Arc<KernelTable>>>,
    ) -> Result<FastSim, Vec<Diagnostic>> {
        let err = |m: String| vec![Diagnostic::error(DiagKind::Other, m)];
        ks.verify(&flat)
            .map_err(|e| err(format!("inconsistent kernel set: {e}")))?;
        let prog = compile(&flat, &ks, &tables, &opts.compile).map_err(err)?;

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
        let mut driven = vec![false; flat.num_nets as usize];
        for g in &flat.gates {
            driven[g.output as usize] = true;
        }
        for r in &flat.resolvers {
            driven[r.output as usize] = true;
        }
        for src in &flat.sources {
            driven[src.output as usize] = true;
        }
        let undriven = (0..flat.num_nets)
            .filter(|&n| !driven[n as usize])
            .map(|n| {
                let v = match flat.undriven_termination.get(&n) {
                    Some(&term) => terminate(term, Logic::X),
                    None => Logic::Z,
                };
                (n, v)
            })
            .collect();
        let input_values = flat
            .top_inputs
            .iter()
            .map(|p| Value::zeros(p.nets.len() as u32))
            .collect();
        let nk = ks.kernels.len();
        let kw = nk.div_ceil(64);
        let timed_bit = timed
            .iter()
            .map(|&(net, _)| {
                prog.timed_nets
                    .iter()
                    .position(|&n| n == net)
                    .map_or(NONE, |b| b as u32)
            })
            .collect();
        let mut fs = FastSim {
            rt: Rt::new(&prog),
            values: UnsafeCell::new(vec![Logic::X; prog.num_nets as usize]),
            clean: Cell::new(false),
            kst: vec![KState::POWER_ON; nk],
            lazy: vec![0; kw],
            unsynced: vec![0; kw],
            frozen: vec![0; kw],
            kob: vec![NONE; nk],
            glast: vec![NONE; nk],
            tlast: vec![0; kw * prog.num_tuples as usize],
            last_in: vec![Logic::X; prog.kernel_inputs_total as usize],
            todo: vec![0; kw],
            general: Vec::new(),
            checks: Vec::new(),
            check_cache: vec![(NONE, NONE, NONE); nk * CHECK_WAYS],
            res_reported: vec![false; prog.resolvers.len()],
            flat,
            opts,
            ks,
            tables,
            prog: Arc::new(prog),
            native: None,
            native_slot: None,
            native_pending: vec![0; 2 * nk],
            native_cache: None,
            timed,
            input_srcs,
            undriven,
            sims: Vec::new(),
            sim_by_key: HashMap::new(),
            order_memo: FxHashMap::default(),
            input_values,
            inputs_pending: false,
            tick: 0,
            next_tick: None,
            tnext: Vec::new(),
            timed_bit,
            pending: Vec::new(),
            fast_pending: Vec::new(),
            frozen_list: Vec::new(),
            diags: Vec::new(),
            stats: FastSimStats::default(),
        };
        fs.reset();
        Ok(fs)
    }

    /// The netlist.
    pub fn netlist(&self) -> &Arc<FlatNetlist> {
        &self.flat
    }

    /// The options this engine was created with.
    pub fn options(&self) -> &FastSimOptions {
        &self.opts
    }

    /// The compiled program (introspection: ops, partitions, LUTs, kernel classes).
    pub fn program(&self) -> &Program {
        &self.prog
    }

    /// The compiled program, shared (what a native backend compiles; see
    /// [`FastSim::install_native`]).
    pub fn program_arc(&self) -> &Arc<Program> {
        &self.prog
    }

    /// Install native code (tier 2) for this engine's program; it is used from the next sweep
    /// on. The code must have been compiled for exactly this program (checked by identity).
    pub fn install_native(&mut self, code: Arc<NativeCode>) -> Result<(), String> {
        if !Arc::ptr_eq(code.program(), &self.prog) {
            return Err("native code was compiled for a different program".into());
        }
        self.native = Some(code);
        Ok(())
    }

    /// Remove the installed native code (back to the interpreter).
    pub fn remove_native(&mut self) -> Option<Arc<NativeCode>> {
        self.native.take()
    }

    /// The installed native code, if any.
    pub fn native(&self) -> Option<&Arc<NativeCode>> {
        self.native.as_ref()
    }

    /// Attach a mailbox from which native code is installed at the next event boundary once a
    /// background compiler publishes it (hot swap). Code for another program is ignored.
    pub fn set_native_slot(&mut self, slot: Arc<NativeSlot>) {
        self.native_slot = Some(slot);
        self.poll_native();
    }

    /// Install native code waiting in the native slot now (normally done at the next event
    /// boundary); returns true if native code is installed.
    pub fn poll_native_slot(&mut self) -> bool {
        self.poll_native();
        self.native.is_some()
    }

    /// Install native code waiting in the slot, if any.
    #[inline]
    fn poll_native(&mut self) {
        if let Some(slot) = &self.native_slot
            && slot.is_ready()
            && let Some(code) = slot.take()
        {
            let _ = self.install_native(code);
        }
    }

    /// Evaluate the dirty partitions (native code where installed).
    #[inline]
    fn sweep(&mut self) {
        match &self.native {
            Some(code) => {
                crate::native::sweep_native(&self.prog, self.values.get_mut(), &mut self.rt, code)
            }
            None => exec::sweep(&self.prog, self.values.get_mut(), &mut self.rt),
        }
    }

    /// The kernel set.
    pub fn kernels(&self) -> &KernelSet {
        &self.ks
    }

    /// The kernel tables (`tables()[i]` belongs to kernel `i`).
    pub fn tables(&self) -> &[Option<Arc<KernelTable>>] {
        &self.tables
    }

    /// Back to tick 0: every gate/resolver output `X`, then the tick-0 event is processed. Input
    /// pin values set so far are kept (a pending change is folded into the tick-0 event);
    /// untaken diagnostics are kept; statistics restart.
    pub fn reset(&mut self) {
        let flat = self.flat.clone();
        let values = self.values.get_mut();
        values.fill(Logic::Z);
        for g in &flat.gates {
            values[g.output as usize] = Logic::X;
        }
        for r in &flat.resolvers {
            values[r.output as usize] = Logic::X;
        }
        for (&net, &term) in &flat.undriven_termination {
            if (net as usize) < values.len() && values[net as usize] == Logic::Z {
                values[net as usize] = terminate(term, Logic::X);
            }
        }
        for s in &flat.sources {
            if let SourceKind::Constant(v) = s.kind {
                values[s.output as usize] = v;
            }
        }
        for &(net, t) in &self.timed {
            values[net as usize] = t.value_at(0);
        }
        for &(net, pin, bit) in &self.input_srcs {
            values[net as usize] = self.input_values[pin as usize].get(bit);
        }
        for k in &mut self.kst {
            let sim = k.sim;
            *k = KState {
                sim,
                ..KState::POWER_ON
            };
        }
        let nk = self.prog.kernel_ops.len();
        self.lazy.fill(0);
        self.frozen.fill(0);
        self.kob.fill(NONE);
        fill_bits(&mut self.unsynced, nk);
        self.last_in.fill(Logic::X);
        self.inputs_pending = false;
        self.tick = 0;
        self.tnext = self
            .timed
            .iter()
            .map(|&(_, t)| t.next_change_after(0).unwrap_or(u64::MAX))
            .collect();
        self.next_tick = self.compute_next_tick();
        self.stats = FastSimStats::default();
        self.rt.parts_run = 0;
        self.rt.ops_run = 0;
        let m = &mut self.rt.marks;
        fill_bits(&mut m.dirty, self.prog.partitions.len());
        fill_bits(&mut m.kpending, nk);
        m.fastok.fill(0);
        fill_bits(&mut m.ddirty, nk);
        self.tlast.fill(0);
        m.bepoch.fill(u32::MAX);
        m.epoch = 0;
        m.init_kin(&self.prog, self.values.get_mut());
        self.rt.lut_stale.fill(0);
        self.rt.lut_last.fill(NONE);
        for &r in &self.rt.res_touched {
            self.rt.res_touched_mark[r as usize] = false;
        }
        self.rt.res_touched.clear();
        self.rt.res_contending.fill(false);
        self.res_reported.fill(false);
        self.process_event();
    }

    /// Load a state (the facade's `load_state`, same semantics as `panda_kernel`'s reference
    /// engine):
    ///
    /// * The engine's time becomes `tick`: the next event is the first source change after
    ///   `tick`, and the loaded values are taken as the settled values of the phase containing
    ///   `tick`.
    /// * `inputs[i]` becomes the value of top-level input pin `i` (pins whose width does not match
    ///   keep their value); no input change is pending afterwards.
    /// * `values` (one per net) is loaded; sources then take their values at `tick`, nets without
    ///   a driver their constant value (`Z`, or that of their termination), and all
    ///   combinational logic is re-evaluated from the kernel nets and the sources.
    /// * Every kernel whose loaded state is consistent with its current inputs (a table state that
    ///   the table maps to itself under the binary input vector, or else a fixed point of the
    ///   kernel's gates and resolvers) keeps it and records the current inputs as those of its
    ///   last transition. Every other kernel falls back to all-`X` and transitions from there
    ///   (exactly like at power-on) as part of a settling event processed at `tick`.
    ///
    /// Statistics and the "already reported" state of diagnostics are kept. Returns the number of
    /// kernels that fell back to `X`.
    ///
    /// # Panics
    ///
    /// If `values.len()` differs from the number of nets.
    pub fn load_state(&mut self, tick: u64, values: &[Logic], inputs: &[Value]) -> usize {
        assert_eq!(
            values.len(),
            self.prog.num_nets as usize,
            "load_state: one value per net"
        );
        for (slot, v) in self.input_values.iter_mut().zip(inputs) {
            if slot.width() == v.width() {
                *slot = v.clone();
            }
        }
        self.inputs_pending = false;
        self.tick = tick;
        self.tnext = self
            .timed
            .iter()
            .map(|&(_, t)| t.next_change_after(tick).unwrap_or(u64::MAX))
            .collect();
        self.next_tick = self.compute_next_tick();
        let flat = self.flat.clone();
        {
            let vals = self.values.get_mut();
            vals.copy_from_slice(values);
            // Undriven nets are constants (`Z` or their termination's value) like constant
            // sources: nothing re-evaluates them, and the compiled program relies on their value
            // (pull-up/-down nets are folded into LUTs and phase classes as constants).
            for &(net, v) in &self.undriven {
                vals[net as usize] = v;
            }
            for s in &flat.sources {
                if let SourceKind::Constant(v) = s.kind {
                    vals[s.output as usize] = v;
                }
            }
            for &(net, t) in &self.timed {
                vals[net as usize] = t.value_at(tick);
            }
            for &(net, pin, bit) in &self.input_srcs {
                vals[net as usize] = self.input_values[pin as usize].get(bit);
            }
        }
        self.clean.set(false);
        // Every kernel is explicit (its nets were loaded); no derived runtime state survives.
        let nk = self.prog.kernel_ops.len();
        self.lazy.fill(0);
        self.frozen.fill(0);
        self.frozen_list.clear();
        self.kob.fill(NONE);
        self.rt.lut_stale.fill(0);
        self.rt.lut_last.fill(NONE);
        let m = &mut self.rt.marks;
        m.fastok.fill(0);
        fill_bits(&mut m.ddirty, nk);
        self.tlast.fill(0);
        // Combinational logic from the loaded kernel nets and sources; its marks are dropped
        // (FastRef clears the candidates after this evaluation).
        fill_bits(&mut m.dirty, self.prog.partitions.len());
        self.sweep();
        let m = &mut self.rt.marks;
        m.kpending.fill(0);
        m.bepoch.fill(u32::MAX);
        m.next_epoch(&self.prog);
        m.init_kin(&self.prog, self.values.get_mut());
        // Kernel consistency.
        let mut fallen = 0;
        for k in 0..nk as u32 {
            let ki = k as usize;
            let state = self.state_vec(k, None);
            let v = self.current_index(k);
            let table_id = self.tables[ki].as_deref().and_then(|t| t.lookup(&state));
            let consistent = match (self.tables[ki].as_deref(), table_id, v) {
                (Some(t), Some(id), Some(v)) => t.transition(id, v).next == Some(id),
                _ => {
                    let vals = self.values.get_mut();
                    let inputs: Vec<Logic> = self.ks.kernels[ki]
                        .inputs
                        .iter()
                        .map(|&n| vals[n as usize])
                        .collect();
                    let si = self.sim_index(k);
                    self.sims[si].is_fixed_point(&state, &inputs)
                }
            };
            self.kst[ki].state_id = NONE;
            if consistent {
                self.kst[ki].ever = true;
                clear_bit(&mut self.unsynced, k);
                self.glast[ki] = v.unwrap_or(NONE);
                // A table state of a kernel with a class, reached under binary inputs.
                if let (KernelOpKind::KTable { table }, Some(_)) =
                    (self.prog.kernel_ops[ki].kind, v)
                {
                    self.kst[ki].state_id = self.prog.ktables[table as usize]
                        .table
                        .lookup(&state)
                        .unwrap_or(NONE);
                    if self.prog.kernel_class[ki] != NONE && self.kst[ki].state_id != NONE {
                        set_bit(&mut self.rt.marks.fastok, k);
                    }
                }
                let kop = self.prog.kernel_ops[ki];
                let vals = self.values.get_mut();
                let outs = self.prog.kernel_out_nets(&kop);
                // Kept for kernels with a class only (as `write_state` does): the others never
                // update it.
                self.kob[ki] = if self.prog.kernel_class[ki] != NONE
                    && outs.len() <= 31
                    && outs.iter().all(|&n| vals[n as usize].is_known())
                {
                    current_outputs(&self.prog, vals, k)
                } else {
                    NONE
                };
            } else {
                fallen += 1;
                self.kst[ki].ever = false;
                set_bit(&mut self.unsynced, k);
                let kop = self.prog.kernel_ops[ki];
                let vals = self.values.get_mut();
                for &net in self.prog.kernel_state_nets(&kop) {
                    exec::set_net(&self.prog, vals, &mut self.rt.marks, net, Logic::X);
                }
                set_bit(&mut self.rt.marks.kpending, k);
            }
        }
        self.process_event();
        fallen
    }

    /// Set a top-level input pin (by name). The width must match. The change is queued and
    /// processed by the next [`FastSim::step_event`] as an event at the current tick.
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

    /// True if input changes are queued.
    pub fn has_pending_inputs(&self) -> bool {
        self.inputs_pending
    }

    /// Process one event: the pending input changes if any (as one event at the current tick),
    /// else the next scheduled source event. Returns the event tick, or `None` if there is nothing
    /// to do.
    pub fn step_event(&mut self) -> Option<u64> {
        self.poll_native();
        if self.inputs_pending {
            self.inputs_pending = false;
            // Same tick: the timed sources keep their values.
            self.apply_inputs();
            self.process_event();
            return Some(self.tick);
        }
        let t = self.next_event_tick()?;
        self.tick = t;
        if !self.native_timed_event() {
            let ta = probe::start();
            self.apply_timed();
            probe::stop(probe::TIMED, ta);
            self.process_event();
        }
        Some(t)
    }

    /// A scheduled event at `self.tick` processed by a native event function, which also writes
    /// the timed-source changes (see `native::EventCtx`); false (nothing done) if there is no
    /// event function for the event's clock tuple.
    #[inline]
    fn native_timed_event(&mut self) -> bool {
        let Some(code) = self.native.as_ref() else {
            return false;
        };
        if !code.has_events() {
            return false;
        }
        let t = self.tick;
        let values = self.values.get_mut();
        let mut tuple = self
            .prog
            .timed_nets
            .iter()
            .enumerate()
            .fold(0u32, |a, (i, &n)| {
                a | u32::from(values[n as usize] == Logic::One) << i
            });
        let mut changed = 0u32;
        for (i, &(net, _)) in self.timed.iter().enumerate() {
            if self.tnext[i] != t {
                continue;
            }
            let bit = self.timed_bit[i];
            if bit == NONE {
                return false;
            }
            // Clocks flip; a power-on reset falls to 0 (its only change).
            let one = match self.timed[i].1 {
                Timed::Clock { .. } => values[net as usize] != Logic::One,
                Timed::PowerOnReset { .. } => false,
            };
            changed |= 1 << bit;
            tuple = tuple & !(1 << bit) | u32::from(one) << bit;
        }
        if code.event(tuple).is_none() {
            return false;
        }
        for (i, &(net, src)) in self.timed.iter().enumerate() {
            if self.tnext[i] != t {
                continue;
            }
            let next = match src {
                Timed::Clock { period, high, .. } => {
                    let rising = values[net as usize] != Logic::One;
                    t.saturating_add(if rising { high } else { period - high })
                }
                Timed::PowerOnReset { .. } => u64::MAX,
            };
            debug_assert_eq!(
                Some(next).filter(|&x| x != u64::MAX),
                src.next_change_after(t)
            );
            self.tnext[i] = next;
        }
        self.next_tick = self.compute_next_tick();
        self.process_event_with(Some((tuple, changed)));
        true
    }

    /// Process up to `n` events (stops early when there is nothing to do).
    pub fn run_events(&mut self, n: u64) {
        for _ in 0..n {
            if self.step_event().is_none() {
                break;
            }
        }
    }

    /// Process every event up to and including tick `t` (pending inputs first).
    pub fn run_until(&mut self, t: u64) {
        if self.inputs_pending {
            self.step_event();
        }
        while self.next_event_tick().is_some_and(|e| e <= t) {
            self.step_event();
        }
    }

    /// Run `cycles` clock cycles of the clock whose output net is named `clock_pin_or_net`:
    /// processes pending inputs, then every event before the `cycles + 1`-th rising edge strictly
    /// after the current tick (so the values are those one tick before a rising edge, the usual
    /// cycle sample). Panics if the name is not a running clock (see [`FastSim::try_run_cycles`]).
    pub fn run_cycles(&mut self, clock_pin_or_net: &str, cycles: u64) {
        if let Err(e) = self.try_run_cycles(clock_pin_or_net, cycles) {
            panic!("run_cycles: {e}");
        }
    }

    /// [`FastSim::run_cycles`] returning an error for an unknown or stopped clock.
    pub fn try_run_cycles(&mut self, clock_pin_or_net: &str, cycles: u64) -> Result<(), String> {
        let net = self
            .flat
            .net_by_name(clock_pin_or_net)
            .ok_or_else(|| format!("no net '{clock_pin_or_net}'"))?;
        let clk = self
            .timed
            .iter()
            .find(|&&(n, t)| n == net && matches!(t, Timed::Clock { .. }))
            .map(|&(_, t)| t)
            .ok_or_else(|| format!("'{clock_pin_or_net}' is not driven by a Clock"))?;
        if cycles == 0 {
            return Ok(());
        }
        let first = clk
            .next_rising_after(self.tick)
            .ok_or_else(|| format!("clock '{clock_pin_or_net}' does not run"))?;
        let Timed::Clock { period, .. } = clk else {
            unreachable!()
        };
        let end = first.saturating_add(cycles.saturating_mul(period));
        self.run_until(end - 1);
        Ok(())
    }

    /// Tick of the most recently processed event.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Tick of the next scheduled source event (strictly after [`FastSim::tick`]), or `None`.
    pub fn next_event_tick(&self) -> Option<u64> {
        self.next_tick
    }

    /// The next scheduled source event after `tick` (cached in `next_tick`), from the per-source
    /// next changes.
    fn compute_next_tick(&self) -> Option<u64> {
        self.tnext.iter().copied().min().filter(|&t| t != u64::MAX)
    }

    /// Value of a net at the settled point.
    pub fn get(&self, net: NetIdx) -> Logic {
        if self.prog.lazy_owner[net as usize] != NONE {
            self.materialize();
        }
        self.values_ref()[net as usize]
    }

    /// Values of all nets at the settled point (index = [`NetIdx`]); exactly FastRef's.
    pub fn values(&self) -> &[Logic] {
        self.materialize();
        self.values_ref()
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

    /// Read by hierarchical name: exact flat net name, multi-bit base name, or top-level pin.
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

    /// Counters since the last reset.
    pub fn stats(&self) -> FastSimStats {
        FastSimStats {
            partitions: self.rt.parts_run,
            ops: self.rt.ops_run,
            ..self.stats
        }
    }

    /// Test hook: check the engine's internal invariants at the settled point (between
    /// events); returns the first violation. Used by the differential tests to catch internal
    /// state corruption before it becomes observable.
    ///
    /// * every lazy kernel has a class and a reconstructible state (outputs, current inputs);
    /// * every fast-ok kernel has a class, binary inputs, and a table state (lazy or explicit);
    /// * recorded output bits (`kob`) equal the kernel's output values;
    /// * every classed kernel's input bits (`kin`) equal its inputs' values;
    /// * a LUT whose last evaluation was a table read holds that entry on its outputs.
    #[doc(hidden)]
    pub fn check_invariants(&self) -> Result<(), String> {
        let values = self.values_ref();
        let prog = &self.prog;
        let m = &self.rt.marks;
        for k in 0..prog.kernel_ops.len() as u32 {
            let ki = k as usize;
            let kop = &prog.kernel_ops[ki];
            let class = prog.kernel_class[ki];
            if test_bit(&self.lazy, k)
                && (class == NONE || lazy_state(prog, values, k, None) == NONE)
            {
                return Err(format!("lazy kernel {k} has no reconstructible state"));
            }
            if test_bit(&m.fastok, k) {
                let binary = prog
                    .kernel_inputs(kop)
                    .iter()
                    .all(|&n| values[n as usize].is_known());
                let table_state = test_bit(&self.lazy, k) || self.kst[ki].state_id != NONE;
                if class == NONE || !binary || !table_state {
                    return Err(format!(
                        "fast-ok kernel {k}: class {class}, binary inputs {binary}, table state \
                         {table_state}"
                    ));
                }
            }
            if self.kob[ki] != NONE && self.kob[ki] != current_outputs(prog, values, k) {
                return Err(format!("kernel {k}: recorded output bits are stale"));
            }
            let kin = prog
                .kernel_inputs(kop)
                .iter()
                .take(32)
                .enumerate()
                .fold(0u32, |a, (i, &n)| {
                    a | u32::from(values[n as usize] == Logic::One) << i
                })
                & prog.kernel_kin_mask[ki];
            // Only kernels with a class have input-index entries (the others only use the
            // pending marks).
            if class != NONE && m.kin[ki] != kin {
                return Err(format!(
                    "kernel {k}: input bits {:b}, inputs {kin:b}",
                    m.kin[ki]
                ));
            }
        }
        for (l, desc) in prog.luts.iter().enumerate() {
            let e = self.rt.lut_last[l];
            if e == NONE {
                continue;
            }
            for (j, &o) in prog.lut_outputs(desc).iter().enumerate() {
                if values[o as usize] != Logic::from_bool((e >> j) & 1 == 1) {
                    return Err(format!("LUT {l}: output {j} does not hold its last entry"));
                }
            }
        }
        Ok(())
    }

    /// Benchmark hook: mark every partition dirty and sweep (a full evaluation of the
    /// combinational logic). Values do not change (they are already settled), so this does not
    /// affect the simulation.
    #[doc(hidden)]
    pub fn bench_full_sweep(&mut self) {
        fill_bits(&mut self.rt.marks.dirty, self.prog.partitions.len());
        self.sweep();
    }

    /// Test hook: the engine's internal runtime state between events, by component (raw net
    /// values except the outputs of fused-away gates, which only native code may store;
    /// change-propagation marks; the boundary log; LUT and kernel bookkeeping). Two engines
    /// over the same program that took the same steps (in any mix of tiers) have equal states:
    /// the differential tests compare them to catch divergences before they become observable.
    #[doc(hidden)]
    pub fn debug_runtime_state(&self) -> Vec<(&'static str, Vec<u64>)> {
        let m = &self.rt.marks;
        let w32 = |v: &[u32]| v.iter().map(|&x| u64::from(x)).collect::<Vec<u64>>();
        let values = self
            .values_ref()
            .iter()
            .zip(&self.prog.lazy_owner)
            .map(|(v, &o)| {
                if o == crate::ir::LAZY_FUSED {
                    u64::MAX
                } else {
                    u64::from(v.code())
                }
            })
            .collect();
        vec![
            ("values", values),
            ("dirty", m.dirty.clone()),
            ("kpending", m.kpending.clone()),
            ("fastok", m.fastok.clone()),
            ("ddirty", m.ddirty.clone()),
            ("bval", m.bval.iter().map(|v| u64::from(v.code())).collect()),
            ("bepoch", w32(&m.bepoch)),
            ("epoch", vec![u64::from(m.epoch)]),
            ("kin", w32(&m.kin)),
            ("kin_prev", w32(&m.kin_prev)),
            ("lut_stale", self.rt.lut_stale.clone()),
            ("lut_last", w32(&self.rt.lut_last)),
            (
                "res_contending",
                self.rt
                    .res_contending
                    .iter()
                    .map(|&b| u64::from(b))
                    .collect(),
            ),
            ("lazy", self.lazy.clone()),
            ("unsynced", self.unsynced.clone()),
            ("kob", w32(&self.kob)),
            ("tlast", self.tlast.clone()),
        ]
    }

    /// Test hook: jump the boundary log's interval counter to `epoch` (to exercise its
    /// wrap-around at `u32::MAX` in every tier). Between events no log entry belongs to the
    /// current interval (every event ends by starting a new one), so a jump forward to an
    /// epoch above every logged one does not change the simulation.
    #[doc(hidden)]
    pub fn set_epoch_for_test(&mut self, epoch: u32) {
        assert!(
            self.rt
                .marks
                .bepoch
                .iter()
                .all(|&e| e == u32::MAX || e < epoch),
            "set_epoch_for_test: only forward jumps"
        );
        self.rt.marks.epoch = epoch;
    }

    // -----------------------------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------------------------

    /// Shared view of the value buffer.
    fn values_ref(&self) -> &[Logic] {
        // SAFETY: the buffer is only mutated through `&mut self` (via `get_mut`) or by
        // `materialize` while `clean` is false. `clean` is only reset by `&mut self` methods, so
        // when `materialize` writes, every borrow handed out by an earlier `&self` call has
        // ended; within one `&self` call, `materialize` runs before any borrow is created.
        // `FastSim` is `!Sync` (UnsafeCell), so no other thread can observe the buffer.
        unsafe { &*self.values.get() }
    }

    /// Bring the lazy nets (LUT-internal nets, non-output nets of lazy kernels) up to date.
    fn materialize(&self) {
        if self.clean.get() {
            return;
        }
        // SAFETY: see `values_ref`: `clean` is false, so no borrow of the buffer exists.
        let values = unsafe { &mut *self.values.get() };
        let prog = &self.prog;
        for (wi, &w) in self.lazy.iter().enumerate() {
            for_bits(w, wi, |k| {
                let sid = lazy_state(prog, values, k, None);
                debug_assert_ne!(sid, NONE, "lazy kernel {k} without a state");
                let kop = &prog.kernel_ops[k as usize];
                let cl = &prog.kclasses[prog.kernel_class[k as usize] as usize];
                let st = prog.ktables[cl.table as usize].state(sid);
                for (&net, &v) in prog.kernel_state_nets(kop).iter().zip(st) {
                    values[net as usize] = v;
                }
            });
        }
        for (wi, &w) in self.rt.lut_stale.iter().enumerate() {
            for_bits(w, wi, |l| {
                exec::run_cold_silent(prog, values, &prog.luts[l as usize]);
            });
        }
        exec::run_fcold_silent(prog, values);
        self.clean.set(true);
    }

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
        match kern.outputs.first().or(kern.nets.first()) {
            Some(&n) => self.flat.net_names[n as usize].clone(),
            None => format!("kernel {k}"),
        }
    }

    /// Write the top-level input values (an input event).
    fn apply_inputs(&mut self) {
        let values = self.values.get_mut();
        for &(net, pin, bit) in &self.input_srcs {
            let v = self.input_values[pin as usize].get(bit);
            exec::set_net(&self.prog, values, &mut self.rt.marks, net, v);
        }
    }

    /// The timed sources changing at `tick` (a scheduled event; the inputs cannot have changed,
    /// pending input changes are processed as their own event first). Incremental: a clock that
    /// changes at `t` flips and next changes `high` (after rising) or `period - high` (after
    /// falling) ticks later; a power-on reset changes once.
    fn apply_timed(&mut self) {
        let t = self.tick;
        let values = self.values.get_mut();
        for (i, &(net, src)) in self.timed.iter().enumerate() {
            if self.tnext[i] != t {
                continue;
            }
            let (v, next) = match src {
                Timed::Clock { period, high, .. } => {
                    let v = if values[net as usize] == Logic::One {
                        Logic::Zero
                    } else {
                        Logic::One
                    };
                    let d = if v == Logic::One { high } else { period - high };
                    (v, t.saturating_add(d))
                }
                Timed::PowerOnReset { .. } => (Logic::Zero, u64::MAX),
            };
            debug_assert_eq!(v, src.value_at(t));
            debug_assert_eq!(
                Some(next).filter(|&x| x != u64::MAX),
                src.next_change_after(t)
            );
            self.tnext[i] = next;
            let off = self.prog.notify_off[net as usize];
            exec::set_net_deferred(values, &mut self.rt.chg, net, off, v);
        }
        exec::flush_deferred(&self.prog, values, &mut self.rt);
        self.next_tick = self.compute_next_tick();
    }

    /// Index of the current clock tuple, or [`NONE`] if phase specialization is off.
    fn current_tuple(&mut self) -> u32 {
        if self.prog.num_tuples == 0 {
            return NONE;
        }
        let values = self.values.get_mut();
        self.prog
            .timed_nets
            .iter()
            .enumerate()
            .fold(0u32, |a, (i, &n)| {
                a | u32::from(values[n as usize] == Logic::One) << i
            })
    }

    /// Rounds 0.. of SPEC §6.2 for the source values already applied (round-faithful).
    fn process_event(&mut self) {
        self.process_event_with(None);
    }

    /// [`FastSim::process_event`]; with `Some((tuple, timed))`, the event's clock tuple is
    /// `tuple` and the timed sources in `timed` (tuple bits) are not written yet: the native
    /// event function of `tuple` (which must exist) writes them.
    fn process_event_with(&mut self, native: Option<(u32, u32)>) {
        self.clean.set(false);
        self.stats.events += 1;
        // The clock tuple only depends on sources (already written).
        let (tuple, timed) = native.unwrap_or_else(|| (self.current_tuple(), 0));
        let mut rounds_since_freeze = 0u32;
        let mut event_rounds = 0u32;
        match self.native_event(tuple, timed) {
            Some((rounds, true)) => {
                self.stats.max_rounds = self.stats.max_rounds.max(rounds);
                self.check_contention();
                return;
            }
            Some((rounds, false)) => {
                // Continue at the round decision point where native code stopped.
                rounds_since_freeze = rounds;
                event_rounds = rounds;
            }
            None => {
                let t0 = probe::start();
                self.sweep();
                probe::stop(probe::SWEEP, t0);
            }
        }
        let words = self.prog.kernel_words as usize;
        let mut pending = std::mem::take(&mut self.pending);
        let mut fast_pending = std::mem::take(&mut self.fast_pending);
        let mut general = std::mem::take(&mut self.general);
        let mut checks = std::mem::take(&mut self.checks);
        loop {
            let t1 = probe::start();
            // Pending kernels whose input vector differs from their last one.
            let mut any = 0u64;
            for w in 0..words {
                let mut bits = std::mem::take(&mut self.rt.marks.kpending[w]) & !self.frozen[w];
                let special = bits & self.unsynced[w];
                for_bits(special, w, |k| {
                    if !self.inputs_changed_explicit(k) {
                        bits &= !(1u64 << (k % 64));
                    }
                });
                self.todo[w] = bits;
                any |= bits;
            }
            if any == 0 {
                self.rt.marks.next_epoch(&self.prog);
                probe::stop(probe::DECIDE, t1);
                break;
            }
            if rounds_since_freeze >= self.opts.fast.delta_limit {
                for w in 0..words {
                    for_bits(self.todo[w], w, |k| general.push(k));
                }
                for &k in &general {
                    set_bit(&mut self.frozen, k);
                    self.frozen_list.push(k);
                    self.record_inputs_explicit(k);
                    let loc = self.kernel_location(k);
                    self.emit(
                        Diagnostic::warning(
                            DiagKind::IterationLimit,
                            format!(
                                "kernel {k} still changing after {} rounds at tick {}; forced to X",
                                self.opts.fast.delta_limit, self.tick
                            ),
                        )
                        .at(loc),
                    );
                    pending.push((k, NewState::AllX));
                }
                general.clear();
                rounds_since_freeze = 0;
            } else {
                if self.all_skip(tuple) {
                    // Every pending kernel is fast-ok and opaque without a possible diagnostic in
                    // this phase: nothing but the lazy marks and the counts.
                    let mut n = 0u64;
                    for w in 0..words {
                        n += u64::from(self.todo[w].count_ones());
                        self.lazy[w] |= self.todo[w];
                        self.todo[w] = 0;
                    }
                    self.stats.transitions += n;
                    self.stats.table_hits += n;
                    self.stats.opaque_skips += n;
                } else {
                    let mut cnt = FastCounts::default();
                    let tf = probe::start();
                    fast_round(
                        &self.prog,
                        self.values.get_mut(),
                        &mut self.rt.marks,
                        &self.kst,
                        &self.kob,
                        &mut self.lazy,
                        &mut self.todo,
                        &mut self.tlast,
                        tuple,
                        &mut fast_pending,
                        &mut checks,
                        &self.check_cache,
                        &mut cnt,
                    );
                    probe::stop(probe::FAST, tf);
                    let n = cnt.opaque + cnt.transparent;
                    self.stats.transitions += n;
                    self.stats.table_hits += n;
                    self.stats.opaque_skips += cnt.opaque;
                    self.stats.transparent += cnt.transparent;
                    self.stats.fast_evals += cnt.evals;
                    for (k, u, v, sid, key) in checks.drain(..) {
                        self.fast_order_check(k, u, v, sid, key);
                    }
                }
                for w in 0..words {
                    for_bits(self.todo[w], w, |k| general.push(k));
                }
                let tg = probe::start();
                for &k in &general {
                    let ns = self.transition(k);
                    pending.push((k, ns));
                }
                probe::stop(probe::GENERAL, tg);
                general.clear();
                rounds_since_freeze += 1;
                event_rounds += 1;
                self.stats.rounds += 1;
            }
            // Round boundary: the transitions were computed from the boundary values; now a new
            // interval starts and the new states are written.
            probe::stop(probe::DECIDE, t1);
            let t2 = probe::start();
            self.rt.marks.next_epoch(&self.prog);
            // Fast-ok bits first: a write below may make an input of a kernel non-binary, which
            // clears its bit again (the interpreter's notify).
            for (k, ns) in &pending {
                // Kernels without a class never have the bit.
                if self.prog.kernel_class[*k as usize] == NONE {
                    continue;
                }
                if self.fast_ok_after(*k, ns) {
                    set_bit(&mut self.rt.marks.fastok, *k);
                } else {
                    clear_bit(&mut self.rt.marks.fastok, *k);
                }
            }
            if !fast_pending.is_empty() {
                let tw = probe::start();
                write_fast(
                    &self.prog,
                    self.values.get_mut(),
                    &mut self.rt,
                    &mut self.kob,
                    &fast_pending,
                );
                probe::stop(probe::WFAST, tw);
                fast_pending.clear();
            }
            let wrote = !pending.is_empty() || self.rt.marks.kpending.iter().any(|&w| w != 0);
            for (k, ns) in pending.drain(..) {
                self.write_state(k, ns);
            }
            probe::stop(probe::WRITE, t2);
            if !wrote && self.rt.marks.dirty.iter().all(|&w| w == 0) {
                // Nothing changed at this boundary: no kernel can be pending in the next round
                // (that iteration would only advance the interval and stop).
                self.rt.marks.next_epoch(&self.prog);
                break;
            }
            let t3 = probe::start();
            self.sweep();
            probe::stop(probe::SWEEP, t3);
        }
        self.pending = pending;
        self.fast_pending = fast_pending;
        self.general = general;
        self.checks = checks;
        self.stats.max_rounds = self.stats.max_rounds.max(event_rounds);
        for k in self.frozen_list.drain(..) {
            clear_bit(&mut self.frozen, k);
        }
        self.check_contention();
    }

    /// Run the native event function of `tuple` (if installed): `Some((rounds, done))` with the
    /// rounds it completed and whether it finished the event (else it stopped at a round
    /// decision point, after the initial sweep; see `native::EventCtx`), or `None`.
    #[inline]
    fn native_event(&mut self, tuple: u32, timed: u32) -> Option<(u32, bool)> {
        if tuple == NONE {
            return None;
        }
        let code = self.native.as_ref()?;
        let f = code.event(tuple)?;
        let parts = code.table();
        debug_assert!(self.frozen.iter().all(|&w| w == 0));
        // Contexts are cached (rebuilding them costs ~9% of the JIT's event throughput) and
        // rebuilt when the engine moved or its native code changed.
        let me: *const FastSim = self;
        if self
            .native_cache
            .as_ref()
            .is_none_or(|c| c.owner != me || c.parts != parts)
        {
            let mut c = Box::new(self.native_contexts(parts));
            c.ctx.env = (&raw mut c.env).cast();
            self.native_cache = Some(c);
        }
        let epoch = self.rt.marks.epoch;
        let c = self.native_cache.as_mut().expect("built above");
        c.ctx.epoch = epoch;
        c.ectx.timed = timed;
        c.ectx.rounds = 0;
        c.ectx.opaque = 0;
        c.ectx.transparent = 0;
        c.ectx.evals = 0;
        // SAFETY: `f` was compiled for `self.prog` (checked when installed) and implements the
        // `EventCtx` contract; the cached contexts point into this engine at its current
        // address (rebuilt above otherwise) with the program's sizes (`parts` has one entry per
        // partition), and the state is not otherwise accessed during the call.
        let tp = probe::start();
        let r = unsafe { f(&mut c.ctx, &mut c.ectx) };
        probe::stop(probe::NATIVE, tp);
        let ec = &c.ectx;
        let n = ec.opaque + ec.transparent;
        self.stats.transitions += n;
        self.stats.table_hits += n;
        self.stats.opaque_skips += ec.opaque;
        self.stats.transparent += ec.transparent;
        self.stats.fast_evals += ec.evals;
        self.stats.rounds += ec.rounds;
        self.stats.native_rounds += ec.rounds;
        let done = match r {
            crate::native::EVENT_DONE => true,
            crate::native::EVENT_BAIL => {
                self.stats.native_bails += 1;
                false
            }
            _ => panic!("native event function of tuple {tuple}: bad result {r}"),
        };
        Some((ec.rounds as u32, done))
    }

    /// The native contexts for this engine at its current address (`ctx.env` is set by the
    /// caller once the value has its final address).
    fn native_contexts(&mut self, parts: *const crate::native::PartitionFn) -> NativeCache {
        let prog: *const Program = &*self.prog;
        let values = self.values.get_mut();
        let len = values.len();
        let vp = values.as_mut_ptr();
        let rtp: *mut Rt = &mut self.rt;
        // SAFETY: `rtp` is valid; only raw pointers into it are taken.
        let (kin_prev, epoch) = unsafe {
            (
                (*rtp).marks.kin_prev.as_mut_ptr(),
                &raw mut (*rtp).marks.epoch,
            )
        };
        NativeCache {
            owner: self,
            parts,
            env: Env {
                prog,
                rt: rtp,
                values: vp,
                len,
            },
            // `env` is set by the caller.
            ctx: crate::native::make_ctx(&self.prog, vp, len, rtp, std::ptr::null_mut()),
            ectx: EventCtx {
                lazy: self.lazy.as_mut_ptr(),
                tlast: self.tlast.as_mut_ptr(),
                kin_prev,
                kob: self.kob.as_mut_ptr(),
                unsynced: self.unsynced.as_mut_ptr(),
                pending: self.native_pending.as_mut_ptr(),
                epoch,
                epoch_wrap: crate::native::epoch_wrap_cb,
                parts,
                delta_limit: self.opts.fast.delta_limit,
                timed: 0,
                n_tuples: self.prog.num_tuples,
                n_parts: self.prog.partitions.len() as u32,
                rounds: 0,
                opaque: 0,
                transparent: 0,
                evals: 0,
            },
        }
    }

    /// True if every pending kernel (`todo`) is fast-ok and skipped in phase `tuple` without a
    /// possible diagnostic ([`Program::tuple_skip`]).
    #[inline]
    fn all_skip(&self, tuple: u32) -> bool {
        if tuple == NONE {
            return false;
        }
        let words = self.todo.len();
        let base = tuple as usize * words;
        (0..words).all(|w| {
            let t = self.todo[w];
            t & !(self.rt.marks.fastok[w] & self.prog.tmasks[base + w][0]) == 0
        })
    }

    /// Report contention of combinational resolvers evaluated during this event.
    fn check_contention(&mut self) {
        if self.rt.res_touched.is_empty() {
            return;
        }
        let touched = std::mem::take(&mut self.rt.res_touched);
        for &r in &touched {
            let ri = r as usize;
            self.rt.res_touched_mark[ri] = false;
            if self.rt.res_contending[ri] {
                if !self.res_reported[ri] {
                    self.res_reported[ri] = true;
                    let fi = self.prog.resolvers[ri].flat_index;
                    let net = self.flat.resolvers[fi as usize].output;
                    let name = self.flat.net_names[net as usize].clone();
                    let msg = format!("contention on net {name} at tick {}", self.tick);
                    self.emit(Diagnostic::warning(DiagKind::Contention, msg).at(name));
                }
            } else {
                self.res_reported[ri] = false;
            }
        }
        self.rt.res_touched = touched;
        self.rt.res_touched.clear();
    }

    /// FastRef's order check for a fast-path transition of kernel `k` from input vector `u` to
    /// `v` (two or more bits differ) in table state `sid`; emits `DelaySensitive` if sensitive.
    #[cold]
    fn fast_order_check(&mut self, k: u32, u: u32, v: u32, sid: u32, key: u32) {
        let ki = k as usize;
        debug_assert_eq!(Some(u), self.last_index(k));
        let KernelOpKind::KTable { table } = self.prog.kernel_ops[ki].kind else {
            unreachable!("a class requires a table");
        };
        debug_assert_ne!(sid, NONE, "fast-ok kernel {k} without a table state");
        let td = &self.prog.ktables[table as usize];
        let e = td.entry(sid, v);
        if !((e & KTableData::ORDER_FLAG != 0 || td.order_check_partial)
            && self.order_sensitive(k, table, sid, u, v))
        {
            self.check_cache[ki * CHECK_WAYS + check_slot(key, u, v)] = (key, u, v);
        } else {
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
    }

    /// Write kernel `k`'s output nets from output bits `ob` (bit `j` = output `j`).
    fn write_outputs(&mut self, k: u32, ob: u32) {
        let prog = &self.prog;
        let kop = &prog.kernel_ops[k as usize];
        let values = self.values.get_mut();
        for (j, &net) in prog.kernel_out_nets(kop).iter().enumerate() {
            let v = Logic::from_bool((ob >> j) & 1 == 1);
            exec::set_net(prog, values, &mut self.rt.marks, net, v);
        }
        self.kob[k as usize] = ob;
    }

    /// True if kernel `k`'s current inputs differ from its explicitly recorded last-transition
    /// inputs (always true before its first transition).
    fn inputs_changed_explicit(&mut self, k: u32) -> bool {
        let ki = k as usize;
        if !self.kst[ki].ever {
            return true;
        }
        let kop = &self.prog.kernel_ops[ki];
        let off = kop.last_in as usize;
        let values = self.values.get_mut();
        self.prog
            .kernel_inputs(kop)
            .iter()
            .zip(&self.last_in[off..off + kop.num_inputs as usize])
            .any(|(&n, &l)| values[n as usize] != l)
    }

    /// Record kernel `k`'s current inputs explicitly (freezing at the delta limit) and mark it
    /// unsynced.
    fn record_inputs_explicit(&mut self, k: u32) {
        let ki = k as usize;
        let kop = &self.prog.kernel_ops[ki];
        let off = kop.last_in as usize;
        let values = self.values.get_mut();
        for (i, &n) in self.prog.kernel_inputs(kop).iter().enumerate() {
            self.last_in[off + i] = values[n as usize];
        }
        self.kst[ki].ever = true;
        set_bit(&mut self.unsynced, k);
    }

    /// The inputs of kernel `k`'s last transition (all `X` before the first one).
    fn last_inputs(&mut self, k: u32) -> Vec<Logic> {
        let ki = k as usize;
        let kop = &self.prog.kernel_ops[ki];
        let n = kop.num_inputs as usize;
        if !self.kst[ki].ever {
            vec![Logic::X; n]
        } else if test_bit(&self.unsynced, k) {
            let off = kop.last_in as usize;
            self.last_in[off..off + n].to_vec()
        } else {
            let values = self.values.get_mut();
            self.prog
                .kernel_inputs(kop)
                .iter()
                .map(|&net| self.rt.marks.boundary_value(&self.prog, values, net))
                .collect()
        }
    }

    /// Binary index of kernel `k`'s last-transition inputs (`None` before the first transition or
    /// if not binary); allocation-free [`FastSim::last_inputs`].
    fn last_index(&mut self, k: u32) -> Option<u32> {
        let ki = k as usize;
        let kop = &self.prog.kernel_ops[ki];
        if !self.kst[ki].ever || kop.num_inputs > 31 {
            return None;
        }
        let synced = !test_bit(&self.unsynced, k);
        if synced && self.prog.kernel_class[ki] == NONE {
            // Only the general path transitions it, which records the index.
            let g = self.glast[ki];
            return (g != NONE).then_some(g);
        }
        let ins = self.prog.kernel_inputs(kop);
        let mut idx = 0u32;
        if !synced {
            let off = kop.last_in as usize;
            for (i, &l) in self.last_in[off..off + ins.len()].iter().enumerate() {
                match l {
                    Logic::Zero => {}
                    Logic::One => idx |= 1 << i,
                    _ => return None,
                }
            }
        } else {
            let values = self.values.get_mut();
            for (i, &net) in ins.iter().enumerate() {
                match self.rt.marks.boundary_value(&self.prog, values, net) {
                    Logic::Zero => {}
                    Logic::One => idx |= 1 << i,
                    _ => return None,
                }
            }
        }
        Some(idx)
    }

    /// Current state vector of kernel `k` (from the reconstruction table if lazy, given its
    /// last-transition input index `u`).
    fn state_vec(&mut self, k: u32, u: Option<u32>) -> Vec<Logic> {
        let prog = &self.prog;
        let kop = &prog.kernel_ops[k as usize];
        let values = self.values.get_mut();
        if test_bit(&self.lazy, k) {
            let sid = lazy_state(prog, values, k, u);
            let cl = &prog.kclasses[prog.kernel_class[k as usize] as usize];
            return prog.ktables[cl.table as usize].state(sid).to_vec();
        }
        prog.kernel_state_nets(kop)
            .iter()
            .map(|&n| values[n as usize])
            .collect()
    }

    /// Binary index of the current inputs of kernel `k`, or `None`.
    fn current_index(&mut self, k: u32) -> Option<u32> {
        let kop = &self.prog.kernel_ops[k as usize];
        if kop.num_inputs > 31 {
            return None;
        }
        let values = self.values.get_mut();
        let mut idx = 0u32;
        for (i, &n) in self.prog.kernel_inputs(kop).iter().enumerate() {
            match values[n as usize] {
                Logic::Zero => {}
                Logic::One => idx |= 1 << i,
                _ => return None,
            }
        }
        Some(idx)
    }

    /// General path: `T(state, inputs)` of kernel `k` from the current values (FastRef's
    /// `transition`). Does not write the state.
    fn transition(&mut self, k: u32) -> NewState {
        self.stats.transitions += 1;
        let ki = k as usize;
        let kop = self.prog.kernel_ops[ki];
        let v = self.current_index(k);
        let ns = self.transition_with(k, kop, v);
        // Recorded after the transition read the previous one.
        self.glast[ki] = v.unwrap_or(NONE);
        ns
    }

    /// [`FastSim::transition`] given the current input index `v`.
    fn transition_with(&mut self, k: u32, kop: crate::ir::KernelOp, v: Option<u32>) -> NewState {
        let ki = k as usize;
        if let KernelOpKind::KTable { table } = kop.kind
            && let Some(v) = v
        {
            let u = self.last_index(k);
            let sid = if test_bit(&self.lazy, k) {
                let values = self.values.get_mut();
                lazy_state(&self.prog, values, k, u)
            } else {
                if self.kst[ki].state_id == NONE {
                    let st = self.state_vec(k, u);
                    if let Some(id) = self.prog.ktables[table as usize].table.lookup(&st) {
                        self.kst[ki].state_id = id;
                    }
                }
                self.kst[ki].state_id
            };
            if sid != NONE {
                let td = &self.prog.ktables[table as usize];
                let e = td.entry(sid, v);
                let partial = td.order_check_partial;
                // Fast exits of `KernelTable::order_sensitivity_known` (fewer than 2 inputs
                // changed; never flagged with a complete check), else the exact check.
                if !self.kst[ki].delay_reported
                    && let Some(u) = u
                    && (u ^ v).count_ones() >= 2
                    && (e & KTableData::ORDER_FLAG != 0 || partial)
                    && self.order_sensitive(k, table, sid, u, v)
                {
                    self.kst[ki].delay_reported = true;
                    let loc = self.kernel_location(k);
                    self.emit(
                        Diagnostic::warning(
                            DiagKind::DelaySensitive,
                            format!(
                                "kernel {k}: several inputs changed at once at tick {} and the \
                                 result depends on their arrival order",
                                self.tick
                            ),
                        )
                        .at(loc),
                    );
                }
                let next = e & !KTableData::ORDER_FLAG;
                if next != KTableData::NON_BINARY {
                    self.stats.table_hits += 1;
                    return NewState::Table { id: next, v };
                }
                // Non-binary entry: the simulator needs the state explicitly.
                if test_bit(&self.lazy, k) {
                    self.kst[ki].state_id = sid;
                }
            }
        }
        self.sim_transition(k, v.is_some())
    }

    /// On-the-fly transition (exactly FastRef's fallback path).
    fn sim_transition(&mut self, k: u32, binary_in: bool) -> NewState {
        self.stats.sim_fallbacks += 1;
        let ki = k as usize;
        let kop = self.prog.kernel_ops[ki];
        // First transition after power-on: the inputs change from X to their current values
        // (FastRef: every kernel element reading them is evaluated, as Timing mode resolves
        // every resolver at power-on).
        let old_in = self.last_inputs(k);
        let u = if self.kst[ki].ever && kop.num_inputs <= 31 {
            panda_compile::charz::input_index(&old_in)
        } else {
            None
        };
        let state = self.state_vec(k, u);
        let n_in = kop.num_inputs as usize;
        let values = self.values.get_mut();
        let new_in: Vec<Logic> = self
            .prog
            .kernel_inputs(&kop)
            .iter()
            .map(|&n| values[n as usize])
            .collect();
        let si = self.sim_index(k);
        let (next, settled) = self.sims[si].transition(&state, &old_in, &new_in);
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
        let (mut meta, mut contention) = (false, None);
        for d in self.sims[si].last_diagnostics() {
            match d.kind {
                DiagKind::Metastable => meta = true,
                DiagKind::Contention if contention.is_none() => {
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
        NewState::Explicit {
            st: next,
            binary_in,
        }
    }

    /// Order sensitivity of kernel `k`'s entry `(sid, u → v)` (FastRef's `order_sensitive`).
    fn order_sensitive(&mut self, k: u32, table: u32, sid: u32, u: u32, v: u32) -> bool {
        let key = (table, sid, u, v);
        if let Some(&b) = self.order_memo.get(&key) {
            return b;
        }
        let t = self.prog.ktables[table as usize].table.clone();
        let b = match t.order_sensitivity_known(sid, u, v) {
            Some(b) => b,
            None => {
                let si = self.sim_index(k);
                t.check_order_sensitive(&mut self.sims[si], &self.opts.fast.charz, sid, u, v)
            }
        };
        self.order_memo.insert(key, b);
        b
    }

    /// Index of kernel `k`'s on-the-fly simulator (shared by structurally identical kernels).
    fn sim_index(&mut self, k: u32) -> usize {
        let ki = k as usize;
        if self.kst[ki].sim == NONE {
            let (sub, map) = kernel_subnet(&self.flat, &self.ks.kernels[ki]);
            let key = structural_key(&sub);
            let idx = match self.sim_by_key.get(&key) {
                Some(&i) => i,
                None => {
                    let i = self.sims.len() as u32;
                    self.sims.push(KernelSim::from_subnet(
                        Arc::new(sub),
                        map,
                        &self.opts.fast.charz,
                    ));
                    self.sim_by_key.insert(key, i);
                    i
                }
            };
            self.kst[ki].sim = idx;
        }
        self.kst[ki].sim as usize
    }

    /// Whether kernel `k` may take the specialized paths after its transition to `ns` (see the
    /// module docs): it has a class, is not frozen, its inputs were binary and its new state is a
    /// table state.
    fn fast_ok_after(&self, k: u32, ns: &NewState) -> bool {
        let ki = k as usize;
        if self.prog.kernel_class[ki] == NONE || test_bit(&self.frozen, k) {
            return false;
        }
        match ns {
            NewState::Table { .. } => true,
            NewState::Explicit { st, binary_in } => {
                *binary_in
                    && match self.prog.kernel_ops[ki].kind {
                        KernelOpKind::KTable { table } => {
                            self.prog.ktables[table as usize].table.lookup(st).is_some()
                        }
                        KernelOpKind::KSim => false,
                    }
            }
            NewState::AllX => false,
        }
    }

    /// Write a kernel's new state (marking readers of changed nets) after a transition (or a
    /// freeze, which leaves the kernel unsynced). The fast-ok bit was set before
    /// ([`FastSim::fast_ok_after`]).
    fn write_state(&mut self, k: u32, ns: NewState) {
        let ki = k as usize;
        let kop = self.prog.kernel_ops[ki];
        let nets = kop.nets as usize..(kop.nets + kop.num_nets) as usize;
        let class = self.prog.kernel_class[ki];
        let frozen = test_bit(&self.frozen, k);
        let was_lazy = test_bit(&self.lazy, k);
        let mut lazy_now = false;
        if class != NONE {
            // The outputs may change outside a transparent evaluation.
            set_bit(&mut self.rt.marks.ddirty, k);
        }
        match ns {
            NewState::Table { id, v } => {
                let KernelOpKind::KTable { table } = kop.kind else {
                    unreachable!("table state without table");
                };
                if class != NONE {
                    let cl = &self.prog.kclasses[class as usize];
                    let ob = cl.out_bits[id as usize];
                    if cl.recon(ob, v) == id {
                        self.write_outputs(k, ob);
                        set_bit(&mut self.lazy, k);
                        self.kst[ki].state_id = NONE;
                        lazy_now = true;
                    }
                }
                if !lazy_now {
                    let td = &self.prog.ktables[table as usize];
                    if !was_lazy && id == self.kst[ki].state_id {
                        // Invariant: the kernel nets hold `states[state_id]`; nothing changes.
                        debug_assert!(
                            self.prog.kernel_nets[nets.clone()]
                                .iter()
                                .zip(td.state(id))
                                .all(|(&n, &v)| self.values.get_mut()[n as usize] == v)
                        );
                    } else {
                        write_kernel_state(
                            &self.prog,
                            self.values.get_mut(),
                            &mut self.rt.marks,
                            &self.prog.kernel_nets[nets],
                            self.prog.kernel_out_indices(&kop),
                            td.state(id),
                        );
                    }
                    clear_bit(&mut self.lazy, k);
                    self.kst[ki].state_id = id;
                }
            }
            NewState::Explicit { st, binary_in } => {
                write_kernel_state(
                    &self.prog,
                    self.values.get_mut(),
                    &mut self.rt.marks,
                    &self.prog.kernel_nets[nets],
                    self.prog.kernel_out_indices(&kop),
                    &st,
                );
                clear_bit(&mut self.lazy, k);
                self.kst[ki].state_id = match kop.kind {
                    KernelOpKind::KTable { table } => self.prog.ktables[table as usize]
                        .table
                        .lookup(&st)
                        .unwrap_or(NONE),
                    KernelOpKind::KSim => match &self.tables[ki] {
                        Some(t) => t.lookup(&st).unwrap_or(NONE),
                        None => NONE,
                    },
                };
                let _ = binary_in;
            }
            NewState::AllX => {
                let values = self.values.get_mut();
                for &net in &self.prog.kernel_nets[nets] {
                    exec::set_net(&self.prog, values, &mut self.rt.marks, net, Logic::X);
                }
                clear_bit(&mut self.lazy, k);
                self.kst[ki].state_id = NONE;
            }
        }
        if !frozen {
            self.kst[ki].ever = true;
            clear_bit(&mut self.unsynced, k);
        }
        if class != NONE {
            // Output bits as written (`NONE` unless all outputs are binary).
            let values = self.values.get_mut();
            let outs = self.prog.kernel_out_nets(&kop);
            self.kob[ki] =
                if outs.len() <= 31 && outs.iter().all(|&n| values[n as usize].is_known()) {
                    current_outputs(&self.prog, values, k)
                } else {
                    NONE
                };
        }
    }
}

/// The contexts handed to native event functions, prebuilt: they point into the owning
/// [`FastSim`] (its runtime vectors, which never reallocate, and its fields), so they are
/// rebuilt when the engine moved (`owner`) or its native code changed (`parts`; a new table
/// at the address of a dropped one is equally valid).
///
/// Note: the raw pointers are derived once and reused across calls while the interpreter
/// accesses the same state through `&mut` borrows in between. Machine code cannot observe the
/// difference (the pointers only reach opaque native calls), but Rust's formal aliasing
/// models (Stacked/Tree Borrows) would consider the later uses invalidated.
struct NativeCache {
    owner: *const FastSim,
    parts: *const crate::native::PartitionFn,
    env: Env,
    ctx: crate::native::NativeCtx,
    ectx: EventCtx,
}

// SAFETY: the pointers refer to the owning `FastSim` (moved together with it, and checked
// against its address before every use) and to its native code (kept alive by it); they are
// only dereferenced by `&mut FastSim` methods.
unsafe impl Send for NativeCache {}

/// Transitions done by [`fast_round`].
#[derive(Default)]
struct FastCounts {
    opaque: u64,
    transparent: u64,
    evals: u64,
}

/// The specialized paths of one round boundary in phase `tuple` for the pending kernels in
/// `todo`; every kernel handled is removed from `todo` (the rest take the general path):
///
/// * fast-ok kernels in [`Program::tuple_skip`] (and checked-opaque kernels with fewer than two
///   flipped inputs): nothing to do but mark them lazy (word ops);
/// * fast-ok KTransparent kernels in [`Program::tuple_fast`] whose last transparent evaluation
///   was in this tuple and that saw no data-input change nor output write since (`tlast`,
///   `ddirty`): the same input vector, so the same entry, nothing changes (word ops);
/// * the other fast-ok kernels in [`Program::tuple_fast`]: KOpaque marks the kernel lazy,
///   KTransparent looks its output bits up from its input index (kept up to date by the
///   interpreter) and queues them in `fast_pending` if they differ from the current outputs.
///   A transparent kernel whose table has no proven entry for the input vector is left to the
///   general path. With [`crate::ir::KFast::check_order`], a transition with two or more
///   changed inputs queues FastRef's order check in `checks` (unless `check_cache` holds the
///   entry as insensitive).
#[allow(clippy::too_many_arguments)]
#[inline]
fn fast_round(
    prog: &Program,
    values: &[Logic],
    marks: &mut exec::Marks,
    kst: &[KState],
    kob: &[u32],
    lazy: &mut [u64],
    todo: &mut [u64],
    tlast: &mut [u64],
    tuple: u32,
    fast_pending: &mut Vec<(u32, u32)>,
    checks: &mut Vec<(u32, u32, u32, u32, u32)>,
    check_cache: &[(u32, u32, u32)],
    cnt: &mut FastCounts,
) {
    let words = todo.len();
    let nt = prog.num_tuples as usize;
    let nk = prog.kernel_ops.len();
    if tuple != NONE {
        let t = tuple as usize;
        let base = t * words;
        let tmasks = &prog.tmasks[base..base + words];
        let ktout = &prog.ktout[t * nk..(t + 1) * nk];
        let kpure = &prog.kpure[t * nk..(t + 1) * nk];
        // One-time length checks for the unchecked word accesses below.
        assert!(
            marks.fastok.len() == words
                && marks.ddirty.len() == words
                && lazy.len() == words
                && tlast.len() == nt * words
                && marks.kin.len() == nk
                && marks.kin_prev.len() == nk
                && kob.len() == nk
                && prog.tracked.len() == words
        );
        for w in 0..words {
            // SAFETY: `w < words` and the lengths were checked above.
            unsafe {
                let ok = *todo.get_unchecked(w) & *marks.fastok.get_unchecked(w);
                if ok == 0 {
                    continue;
                }
                let [t_skip, t_fast, t_check, t_copaque] = *tmasks.get_unchecked(w);
                // Order checks are only needed when two or more inputs flipped.
                let mut multi = 0u64;
                let mut tr = ok & (t_copaque | t_check) & *prog.tracked.get_unchecked(w);
                while tr != 0 {
                    let b = tr.trailing_zeros();
                    tr &= tr - 1;
                    let ki = w * 64 + b as usize;
                    let fl = *marks.kin.get_unchecked(ki) ^ *marks.kin_prev.get_unchecked(ki);
                    multi |= u64::from(fl & fl.wrapping_sub(1) != 0) << b;
                }
                let skip = ok & (t_skip | (t_copaque & !multi));
                let fast = ok & t_fast & !skip;
                let checked = fast & t_check & multi;
                let tl = tlast.get_unchecked_mut(base + w);
                let same = fast & *tl & !*marks.ddirty.get_unchecked(w) & !checked;
                let todo_f = fast & !same;
                let (n_trans, opaque, general) = if !prog.touts2.is_empty() {
                    let t2 = Packed2(&prog.touts2);
                    eval_fast(ktout, kpure, &t2, &marks.kin, kob, todo_f, w, fast_pending)
                } else if !prog.touts8.is_empty() {
                    eval_fast(
                        ktout,
                        kpure,
                        &prog.touts8[..],
                        &marks.kin,
                        kob,
                        todo_f,
                        w,
                        fast_pending,
                    )
                } else {
                    eval_fast(
                        ktout,
                        kpure,
                        &prog.touts[..],
                        &marks.kin,
                        kob,
                        todo_f,
                        w,
                        fast_pending,
                    )
                };
                let evaluated = fast & !same & !general;
                let mut chk = checked & evaluated;
                while chk != 0 {
                    let k = (w * 64) as u32 + chk.trailing_zeros();
                    chk &= chk - 1;
                    if !kst[k as usize].delay_reported {
                        let v = marks.kin[k as usize] | kpure[k as usize];
                        queue_check(
                            prog,
                            values,
                            marks,
                            kst,
                            kob,
                            lazy,
                            k,
                            v,
                            checks,
                            check_cache,
                        );
                    }
                }
                // Transparent evaluations in this tuple: the unchanged skip applies until the
                // next data change.
                let transp = evaluated & !opaque;
                if transp != 0 {
                    for tt in 0..nt {
                        *tlast.get_unchecked_mut(tt * words + w) &= !transp;
                    }
                    *tlast.get_unchecked_mut(base + w) |= transp;
                    *marks.ddirty.get_unchecked_mut(w) &= !transp;
                }
                cnt.opaque += u64::from((skip | (evaluated & opaque)).count_ones());
                cnt.transparent += u64::from(same.count_ones()) + n_trans;
                cnt.evals += u64::from(evaluated.count_ones());
                let done = skip | same | evaluated;
                *lazy.get_unchecked_mut(w) |= done;
                *todo.get_unchecked_mut(w) &= !done;
            }
        }
    }
    if !prog.vfast_used {
        return;
    }
    // Per-vector actions for the kernels without a tuple class here: exactly one input changed
    // since the last boundary (tracked), so `u = v ^ flipped`.
    for w in 0..words {
        let ok = todo[w] & marks.fastok[w];
        let mut g = ok & prog.vfast[w];
        if tuple != NONE {
            let base = tuple as usize * words;
            g &= !(prog.tuple_skip[base + w] | prog.tuple_fast[base + w]);
        }
        let mut done = 0u64;
        while g != 0 {
            let b = g.trailing_zeros();
            g &= g - 1;
            let k = (w * 64) as u32 + b;
            let ki = k as usize;
            if marks.flipped(k).count_ones() != 1 {
                continue;
            }
            let v = marks.kin[ki];
            debug_assert_eq!(Some(v), input_index_of(prog, values, k));
            let act = prog.kclasses[prog.kernel_class[ki] as usize].vact[v as usize];
            if act == NONE {
                continue;
            }
            let bit = 1u64 << b;
            cnt.evals += 1;
            if act == VACT_OPAQUE {
                cnt.opaque += 1;
            } else {
                cnt.transparent += 1;
                debug_assert!(kob[ki] == NONE || kob[ki] == current_outputs(prog, values, k));
                if kob[ki] != act {
                    fast_pending.push((k, act));
                }
                // Not a tuple-transparent evaluation: invalidate the unchanged skip.
                marks.ddirty[w] |= bit;
            }
            done |= bit;
        }
        lazy[w] |= done;
        todo[w] &= !done;
    }
}

/// Per-kernel fast paths of the kernels in `f` (bitset word `w`) with the current tuple's
/// output tables `ktout` / pure bits `kpure`: returns (transparent evaluations, opaque kernels,
/// kernels left to the general path). Queues changed transparent outputs in `fast_pending`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn eval_fast<T: ToutTable + ?Sized>(
    ktout: &[u32],
    kpure: &[u32],
    touts: &T,
    kin: &[u32],
    kob: &[u32],
    mut f: u64,
    w: usize,
    fast_pending: &mut Vec<(u32, u32)>,
) -> (u64, u64, u64) {
    let (mut n_trans, mut opaque, mut general) = (0u64, 0u64, 0u64);
    debug_assert!(ktout.len() == kin.len() && kpure.len() == kin.len() && kob.len() == kin.len());
    while f != 0 {
        let b = f.trailing_zeros();
        f &= f - 1;
        let ki = w * 64 + b as usize;
        debug_assert!(ki < kin.len());
        // SAFETY: `f` holds kernels of word `w` (< the kernel count, the length of `ktout`,
        // `kpure`, `kin` and `kob`, checked by the caller); `VerifyPass` checked that every
        // proven table `tout` has `2^inputs` entries in `touts`, and `kin | kpure` only has bits
        // below the kernel's input count.
        unsafe {
            let tout = *ktout.get_unchecked(ki);
            if tout == NONE {
                opaque |= 1 << b;
                continue;
            }
            // Fast-ok: the inputs are binary, so the tracked bits plus the tuple's pure bits are
            // the input index.
            let v = *kin.get_unchecked(ki) | *kpure.get_unchecked(ki);
            debug_assert!((tout as usize + v as usize) < touts.entries());
            let Some(ob) = touts.get_unchecked(tout as usize + v as usize) else {
                general |= 1 << b;
                continue;
            };
            n_trans += 1;
            if *kob.get_unchecked(ki) != ob {
                fast_pending.push((ki as u32, ob));
            }
        }
    }
    (n_trans, opaque, general)
}

/// Ways of the per-kernel order-check cache ([`FastSim`]'s `check_cache`).
const CHECK_WAYS: usize = 16;

/// Slot of `(key, u, v)` in a kernel's order-check cache.
#[inline]
fn check_slot(key: u32, u: u32, v: u32) -> usize {
    let h = (u64::from(key) ^ u64::from(u) << 21 ^ u64::from(v) << 42)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (h >> 60) as usize & (CHECK_WAYS - 1)
}

/// The state key of a kernel's order-check cache entry: its output bits (tagged) when lazy and
/// binary (the state is then `recon(outputs, u)`, a function of the key and `u`), else its
/// table state id.
#[inline]
fn check_key(lazy: bool, ob: u32, sid: impl FnOnce() -> u32) -> u32 {
    if lazy && ob != NONE && ob < 1 << 31 {
        1 << 31 | ob
    } else {
        sid()
    }
}

/// A transparent output table ([`Program::touts`], [`Program::touts8`] or
/// [`Program::touts2`]).
trait ToutTable {
    /// Number of entries.
    fn entries(&self) -> usize;
    /// Entry `i`: the output bits, or `None` if not proven.
    ///
    /// # Safety
    ///
    /// `i < self.entries()`.
    unsafe fn get_unchecked(&self, i: usize) -> Option<u32>;
}

impl ToutTable for [u32] {
    fn entries(&self) -> usize {
        self.len()
    }
    #[inline(always)]
    unsafe fn get_unchecked(&self, i: usize) -> Option<u32> {
        // SAFETY: the caller guarantees `i < self.len()`.
        let t = unsafe { *<[u32]>::get_unchecked(self, i) };
        (t != NONE).then_some(t)
    }
}

impl ToutTable for [u8] {
    fn entries(&self) -> usize {
        self.len()
    }
    #[inline(always)]
    unsafe fn get_unchecked(&self, i: usize) -> Option<u32> {
        // SAFETY: the caller guarantees `i < self.len()`.
        let t = unsafe { *<[u8]>::get_unchecked(self, i) };
        (t != 0xFF).then_some(u32::from(t))
    }
}

/// [`Program::touts2`] (2-bit entries).
struct Packed2<'a>(&'a [u64]);

impl ToutTable for Packed2<'_> {
    fn entries(&self) -> usize {
        self.0.len() * 32
    }
    #[inline(always)]
    unsafe fn get_unchecked(&self, i: usize) -> Option<u32> {
        // SAFETY: the caller guarantees `i < 32 * self.0.len()`.
        let c = (unsafe { *self.0.get_unchecked(i / 32) } >> (2 * (i % 32))) & 3;
        (c != 2).then_some(c as u32)
    }
}

/// Queue FastRef's order check for the fast-path transition of kernel `k` to input vector `v`
/// if two or more inputs flipped (its state is read before `lazy` changes) and the cache does
/// not hold the entry as insensitive.
#[cold]
#[allow(clippy::too_many_arguments)]
fn queue_check(
    prog: &Program,
    values: &[Logic],
    marks: &exec::Marks,
    kst: &[KState],
    kob: &[u32],
    lazy: &[u64],
    k: u32,
    v: u32,
    checks: &mut Vec<(u32, u32, u32, u32, u32)>,
    check_cache: &[(u32, u32, u32)],
) {
    let fl = marks.flipped(k);
    if fl.count_ones() < 2 {
        return;
    }
    let u = v ^ fl;
    let is_lazy = test_bit(lazy, k);
    let sid = || {
        if is_lazy {
            lazy_state(prog, values, k, Some(u))
        } else {
            kst[k as usize].state_id
        }
    };
    let key = check_key(is_lazy, kob[k as usize], sid);
    if check_cache[k as usize * CHECK_WAYS + check_slot(key, u, v)] == (key, u, v) {
        return;
    }
    let sid = if key >> 31 == 1 { sid() } else { key };
    checks.push((k, u, v, sid, key));
}

/// Write the queued transparent outputs `(kernel, output bits)` and propagate the changes in one
/// batch.
fn write_fast(
    prog: &Program,
    values: &mut [Logic],
    rt: &mut exec::Rt,
    kob: &mut [u32],
    fast_pending: &[(u32, u32)],
) {
    for &(k, ob) in fast_pending {
        let (net, rec) = prog.kout1[k as usize];
        if net != NONE {
            // One output: `ob` is 0 or 1.
            kob[k as usize] = ob;
            exec::set_net_deferred(values, &mut rt.chg, net, rec, Logic::from_bool(ob & 1 == 1));
            continue;
        }
        let kop = &prog.kernel_ops[k as usize];
        let outs = prog.kernel_out_nets(kop);
        let recs = &prog.kernel_out_rec[kop.outs as usize..(kop.outs + kop.num_outs) as usize];
        let last = kob[k as usize];
        // Kernels with a class have at most 31 outputs. `kob` (if binary) holds the current
        // outputs: only the differing bits change.
        debug_assert!(outs.len() <= 31);
        let mut diff = if last == NONE {
            (1u32 << outs.len()) - 1
        } else {
            ob ^ last
        };
        while diff != 0 {
            let j = diff.trailing_zeros();
            diff &= diff - 1;
            let v = Logic::from_bool((ob >> j) & 1 == 1);
            let j = j as usize;
            exec::set_net_deferred(values, &mut rt.chg, outs[j], recs[j], v);
        }
        kob[k as usize] = ob;
    }
    exec::flush_deferred(prog, values, rt);
}

/// Output bits of kernel `k` from the current values.
#[inline]
fn current_outputs(prog: &Program, values: &[Logic], k: u32) -> u32 {
    let kop = &prog.kernel_ops[k as usize];
    prog.kernel_out_nets(kop)
        .iter()
        .enumerate()
        .fold(0u32, |a, (j, &n)| {
            a | u32::from(values[n as usize] == Logic::One) << j
        })
}

/// Binary index of kernel `k`'s current inputs (debug checks).
fn input_index_of(prog: &Program, values: &[Logic], k: u32) -> Option<u32> {
    let kop = &prog.kernel_ops[k as usize];
    panda_compile::charz::input_index(
        &prog
            .kernel_inputs(kop)
            .iter()
            .map(|&n| values[n as usize])
            .collect::<Vec<_>>(),
    )
}

/// State id of lazy kernel `k`: the reconstruction entry for its current outputs and input
/// vector `u` (its last-transition inputs; `None` = its current inputs).
fn lazy_state(prog: &Program, values: &[Logic], k: u32, u: Option<u32>) -> u32 {
    let kop = &prog.kernel_ops[k as usize];
    let cl = &prog.kclasses[prog.kernel_class[k as usize] as usize];
    let nets = prog.kernel_state_nets(kop);
    let mut o = 0u32;
    for (j, &idx) in prog.kernel_out_indices(kop).iter().enumerate() {
        o |= u32::from(values[nets[idx as usize] as usize] == Logic::One) << j;
    }
    let v = u.unwrap_or_else(|| {
        prog.kernel_inputs(kop)
            .iter()
            .enumerate()
            .fold(0u32, |a, (i, &n)| {
                a | u32::from(values[n as usize] == Logic::One) << i
            })
    });
    cl.recon(o, v)
}

/// Write a kernel's state nets (`nets[j] = state[j]`), propagating changes of the nets listed in
/// `outs` (the only ones with readers).
fn write_kernel_state(
    prog: &Program,
    values: &mut [Logic],
    marks: &mut exec::Marks,
    nets: &[u32],
    outs: &[u32],
    state: &[Logic],
) {
    debug_assert_eq!(nets.len(), state.len());
    for &j in outs {
        let net = nets[j as usize];
        let old = values[net as usize];
        let new = state[j as usize];
        if old != new {
            values[net as usize] = new;
            marks.notify(prog, values, net, old);
        }
    }
    for (&net, &v) in nets.iter().zip(state) {
        values[net as usize] = v;
    }
}

/// Set the first `n` bits of a bitset.
fn fill_bits(bits: &mut [u64], n: usize) {
    bits.fill(0);
    for i in 0..n {
        bits[i / 64] |= 1 << (i % 64);
    }
}
