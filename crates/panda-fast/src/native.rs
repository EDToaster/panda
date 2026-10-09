//! The native-code backend hook (tier 2, SPEC §8.9).
//!
//! A native backend (the Cranelift JIT in `panda-jit`) compiles parts of a [`Program`] to
//! machine code that runs over the **same** flat runtime state as the interpreter, so the tiers
//! can be mixed freely and swapped at any event boundary:
//!
//! * **Partitions** ([`PartitionFn`]): evaluate every op of one partition exactly as
//!   `exec::run_partition` does — same values, same change propagation (reader partitions dirty,
//!   reader kernels pending / data-dirty / not fast-ok, boundary log, kernel input indices), same
//!   LUT bookkeeping, same counters. Partitions without native code run in the interpreter.
//! * **Events** ([`EventFn`], optional, per clock tuple): the round loop of
//!   `FastSim::process_event` for an event whose clock tuple is that tuple, as long as every
//!   round boundary is handled by the specialized kernel paths (see [`EventCtx`]); at the first
//!   round boundary that needs anything else, the function returns at that decision point and
//!   the interpreter continues the event from there.
//!
//! Native code receives a [`NativeCtx`] (`#[repr(C)]`, field offsets in [`ctx_offsets`]) with
//! base pointers of the runtime arrays, plus host callbacks for the rare, complex cases (a LUT
//! whose support is not binary, a resolver, an interpreted partition, the boundary-log epoch
//! wrap-around), which run the interpreter's code.
//!
//! The contexts also carry the lengths of the arrays behind their pointers and a fault callback
//! ([`FaultFn`]), for native code compiled with runtime checks (`panda-jit`'s paranoid mode).
//!
//! [`NativeCode`] bundles the functions with the [`Program`] they were compiled for (checked by
//! identity when installed, see [`crate::FastSim::install_native`]) and keeps the code memory
//! alive. A [`NativeSlot`] lets a background compiler hand finished code to a running
//! [`crate::FastSim`], which picks it up at its next event boundary (hot swap).

use std::any::Any;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use panda_netlist::Logic;

use crate::exec::{self, Rt};
#[cfg(doc)]
use crate::ir::NONE;
use crate::ir::{KIN_KERNEL_MASK, KIN_POS_MASK, KIN_POS_SHIFT, Program};

/// Native evaluation of partition `p` (second argument; see the module docs), including the
/// counters (`NativeCtx::parts_run += 1`, `NativeCtx::ops_run += ` the partition's op count).
///
/// # Safety
///
/// Called with a [`NativeCtx`] whose pointers are valid for the program the code was compiled
/// for; the function may only access the state through them.
pub type PartitionFn = unsafe extern "C" fn(*mut NativeCtx, u32);

/// Native event processing for one clock tuple (see [`EventCtx`]): returns
/// [`EVENT_DONE`] or [`EVENT_BAIL`].
///
/// # Safety
///
/// As [`PartitionFn`], with a valid [`EventCtx`].
pub type EventFn = unsafe extern "C" fn(*mut NativeCtx, *mut EventCtx) -> u32;

/// [`EventFn`] result: the event is fully processed.
pub const EVENT_DONE: u32 = 1;
/// [`EventFn`] result: stopped at a round-boundary decision point (pending kernels not taken
/// yet); the host continues the event's round loop from there.
pub const EVENT_BAIL: u32 = 0;

/// Host callback for a LUT op whose support is not binary: runs the LUT's cold gate code with
/// change propagation (the interpreter's path), argument = LUT index.
pub type LutColdFn = unsafe extern "C" fn(*mut NativeCtx, u32);

/// Host callback for a resolver op: argument = index of the op in [`Program::ops`].
pub type ResolveFn = unsafe extern "C" fn(*mut NativeCtx, u32);

/// Host callback of paranoid native code (see [`NativeCtx::fault`]): a runtime check in the
/// generated code failed (argument: a check code). Never returns (aborts the process: the
/// state may be corrupt and unwinding through native frames is impossible).
pub type FaultFn = unsafe extern "C" fn(*mut NativeCtx, u32);

/// Host callback: the boundary-log epoch wrapped around (`*EventCtx::epoch` reached
/// `u32::MAX`): reset the log (`exec::Marks::next_epoch`'s wrap: every slot unlogged, epoch 0).
pub type EpochWrapFn = unsafe extern "C" fn(*mut NativeCtx);

/// The runtime state handed to native code. All pointers are base pointers of the interpreter's
/// arrays (see `exec::Marks` / `exec::Rt`); bitsets are `u64` words.
#[repr(C)]
pub struct NativeCtx {
    /// Net values (one `Logic` code byte per net).
    pub values: *mut u8,
    /// Dirty partitions.
    pub dirty: *mut u64,
    /// Pending kernels.
    pub kpending: *mut u64,
    /// Data-dirty kernels.
    pub ddirty: *mut u64,
    /// Fast-ok kernels.
    pub fastok: *mut u64,
    /// Per kernel: input index bits.
    pub kin: *mut u32,
    /// Boundary log values (per log slot).
    pub bval: *mut u8,
    /// Boundary log epochs (per log slot).
    pub bepoch: *mut u32,
    /// Per LUT: entry held by its outputs, or [`NONE`].
    pub lut_last: *mut u32,
    /// LUTs with stale internal nets.
    pub lut_stale: *mut u64,
    /// [`Program::lut_entries`].
    pub lut_entries: *const u32,
    /// Partitions evaluated (counter).
    pub parts_run: *mut u64,
    /// Ops executed (counter).
    pub ops_run: *mut u64,
    /// Opaque host environment for the callbacks.
    pub env: *mut c_void,
    /// See [`LutColdFn`].
    pub lut_cold: LutColdFn,
    /// See [`ResolveFn`].
    pub resolve: ResolveFn,
    /// See [`FaultFn`]: called by native code compiled with runtime checks (`panda-jit`'s
    /// paranoid mode) when a check fails.
    pub fault: FaultFn,
    /// Current inter-boundary interval (`exec::Marks::epoch`).
    pub epoch: u32,
    /// Length of `values` (nets). This and the following lengths describe the arrays behind
    /// the pointers, for the runtime checks of paranoid native code.
    pub n_values: u32,
    /// Length of `dirty` (words).
    pub n_part_words: u32,
    /// Length of `kpending`, `ddirty` and `fastok` (words).
    pub n_kernel_words: u32,
    /// Length of `kin` (kernels).
    pub n_kernels: u32,
    /// Length of `bval` and `bepoch` (log slots).
    pub n_log: u32,
    /// Length of `lut_last` (LUTs; `lut_stale` has `n_luts.div_ceil(64)` words).
    pub n_luts: u32,
    /// Length of `lut_entries`.
    pub n_lut_entries: u32,
}

/// Byte offsets of the [`NativeCtx`] and [`EventCtx`] fields (for code generators).
pub mod ctx_offsets {
    use super::{EventCtx, NativeCtx};
    use std::mem::offset_of;
    /// [`NativeCtx::values`].
    pub const VALUES: i32 = offset_of!(NativeCtx, values) as i32;
    /// [`NativeCtx::dirty`].
    pub const DIRTY: i32 = offset_of!(NativeCtx, dirty) as i32;
    /// [`NativeCtx::kpending`].
    pub const KPENDING: i32 = offset_of!(NativeCtx, kpending) as i32;
    /// [`NativeCtx::ddirty`].
    pub const DDIRTY: i32 = offset_of!(NativeCtx, ddirty) as i32;
    /// [`NativeCtx::fastok`].
    pub const FASTOK: i32 = offset_of!(NativeCtx, fastok) as i32;
    /// [`NativeCtx::kin`].
    pub const KIN: i32 = offset_of!(NativeCtx, kin) as i32;
    /// [`NativeCtx::bval`].
    pub const BVAL: i32 = offset_of!(NativeCtx, bval) as i32;
    /// [`NativeCtx::bepoch`].
    pub const BEPOCH: i32 = offset_of!(NativeCtx, bepoch) as i32;
    /// [`NativeCtx::lut_last`].
    pub const LUT_LAST: i32 = offset_of!(NativeCtx, lut_last) as i32;
    /// [`NativeCtx::lut_stale`].
    pub const LUT_STALE: i32 = offset_of!(NativeCtx, lut_stale) as i32;
    /// [`NativeCtx::lut_entries`].
    pub const LUT_ENTRIES: i32 = offset_of!(NativeCtx, lut_entries) as i32;
    /// [`NativeCtx::parts_run`].
    pub const PARTS_RUN: i32 = offset_of!(NativeCtx, parts_run) as i32;
    /// [`NativeCtx::ops_run`].
    pub const OPS_RUN: i32 = offset_of!(NativeCtx, ops_run) as i32;
    /// [`NativeCtx::lut_cold`].
    pub const LUT_COLD: i32 = offset_of!(NativeCtx, lut_cold) as i32;
    /// [`NativeCtx::resolve`].
    pub const RESOLVE: i32 = offset_of!(NativeCtx, resolve) as i32;
    /// [`NativeCtx::epoch`].
    pub const EPOCH: i32 = offset_of!(NativeCtx, epoch) as i32;
    /// [`NativeCtx::fault`].
    pub const FAULT: i32 = offset_of!(NativeCtx, fault) as i32;
    /// [`NativeCtx::n_values`].
    pub const N_VALUES: i32 = offset_of!(NativeCtx, n_values) as i32;
    /// [`NativeCtx::n_part_words`].
    pub const N_PART_WORDS: i32 = offset_of!(NativeCtx, n_part_words) as i32;
    /// [`NativeCtx::n_kernel_words`].
    pub const N_KERNEL_WORDS: i32 = offset_of!(NativeCtx, n_kernel_words) as i32;
    /// [`NativeCtx::n_kernels`].
    pub const N_KERNELS: i32 = offset_of!(NativeCtx, n_kernels) as i32;
    /// [`NativeCtx::n_log`].
    pub const N_LOG: i32 = offset_of!(NativeCtx, n_log) as i32;
    /// [`NativeCtx::n_luts`].
    pub const N_LUTS: i32 = offset_of!(NativeCtx, n_luts) as i32;
    /// [`NativeCtx::n_lut_entries`].
    pub const N_LUT_ENTRIES: i32 = offset_of!(NativeCtx, n_lut_entries) as i32;

    /// [`EventCtx::lazy`].
    pub const E_LAZY: i32 = offset_of!(EventCtx, lazy) as i32;
    /// [`EventCtx::tlast`].
    pub const E_TLAST: i32 = offset_of!(EventCtx, tlast) as i32;
    /// [`EventCtx::kin_prev`].
    pub const E_KIN_PREV: i32 = offset_of!(EventCtx, kin_prev) as i32;
    /// [`EventCtx::kob`].
    pub const E_KOB: i32 = offset_of!(EventCtx, kob) as i32;
    /// [`EventCtx::unsynced`].
    pub const E_UNSYNCED: i32 = offset_of!(EventCtx, unsynced) as i32;
    /// [`EventCtx::pending`].
    pub const E_PENDING: i32 = offset_of!(EventCtx, pending) as i32;
    /// [`EventCtx::epoch`].
    pub const E_EPOCH: i32 = offset_of!(EventCtx, epoch) as i32;
    /// [`EventCtx::epoch_wrap`].
    pub const E_EPOCH_WRAP: i32 = offset_of!(EventCtx, epoch_wrap) as i32;
    /// [`EventCtx::parts`].
    pub const E_PARTS: i32 = offset_of!(EventCtx, parts) as i32;
    /// [`EventCtx::delta_limit`].
    pub const E_DELTA_LIMIT: i32 = offset_of!(EventCtx, delta_limit) as i32;
    /// [`EventCtx::n_tuples`].
    pub const E_N_TUPLES: i32 = offset_of!(EventCtx, n_tuples) as i32;
    /// [`EventCtx::n_parts`].
    pub const E_N_PARTS: i32 = offset_of!(EventCtx, n_parts) as i32;
    /// [`EventCtx::timed`].
    pub const E_TIMED: i32 = offset_of!(EventCtx, timed) as i32;
    /// [`EventCtx::rounds`].
    pub const E_ROUNDS: i32 = offset_of!(EventCtx, rounds) as i32;
    /// [`EventCtx::opaque`].
    pub const E_OPAQUE: i32 = offset_of!(EventCtx, opaque) as i32;
    /// [`EventCtx::transparent`].
    pub const E_TRANSPARENT: i32 = offset_of!(EventCtx, transparent) as i32;
    /// [`EventCtx::evals`].
    pub const E_EVALS: i32 = offset_of!(EventCtx, evals) as i32;
}

/// The extra state of an [`EventFn`] (bitsets of `Program::kernel_words` words).
///
/// An event function for clock tuple `t` is called with no kernel frozen, either right after an
/// input event's source changes were written (`timed` = 0), or for a scheduled event before its
/// timed-source changes were written: it first writes the timed nets in `timed` (bit `i`: net
/// `Program::timed_nets[i]` becomes bit `i` of `t`) with change propagation, exactly as
/// `FastSim::apply_timed`. It then performs `FastSim::process_event`'s initial sweep and its
/// round loop, where each round boundary must be one that `fast_round`'s
/// specialized paths (tuple part and per-vector part) handle completely without an order check:
///
/// 1. **decide** without side effects: the pending kernels are `kpending` (none may be
///    `unsynced`); with none, start the next interval (`Marks::next_epoch`) and return
///    [`EVENT_DONE`]; if `rounds` reached `delta_limit`, or some pending kernel is not fast-ok or
///    would need the general path or an order check, return [`EVENT_BAIL`] (nothing changed
///    since the previous boundary's sweep);
/// 2. **commit**: take `kpending`, apply `fast_round`'s effects (`lazy`, `tlast`, `ddirty`) and
///    count (`opaque`, `transparent`, `evals`, `rounds`);
/// 3. start the next interval (`kin_prev` of the tracked kernels, `*epoch + 1` with
///    [`EventCtx::epoch_wrap`] at `u32::MAX`, `NativeCtx::epoch`);
/// 4. write the changed transparent outputs with change propagation and update `kob`
///    (`write_fast`);
/// 5. if nothing is pending nor dirty: next interval, [`EVENT_DONE`]; else sweep (partition `p`
///    by `parts[p]`) and continue at 1.
#[repr(C)]
pub struct EventCtx {
    /// Lazy kernels.
    pub lazy: *mut u64,
    /// Per tuple: kernels whose last transparent evaluation was in that tuple.
    pub tlast: *mut u64,
    /// Per kernel: `kin` at the last round boundary.
    pub kin_prev: *mut u32,
    /// Per kernel: output bits as last written, or [`NONE`].
    pub kob: *mut u32,
    /// Kernels whose last-transition inputs are recorded explicitly.
    pub unsynced: *mut u64,
    /// Scratch: room for one `(kernel, output bits)` pair per kernel.
    pub pending: *mut u32,
    /// `exec::Marks::epoch`.
    pub epoch: *mut u32,
    /// See [`EpochWrapFn`].
    pub epoch_wrap: EpochWrapFn,
    /// Per partition: the function evaluating it (native, or the interpreter trampoline).
    pub parts: *const PartitionFn,
    /// `FastOptions::delta_limit`.
    pub delta_limit: u32,
    /// Timed sources changing at this event (bit `i` = `Program::timed_nets[i]`): written first
    /// (to their value in the tuple), with change propagation.
    pub timed: u32,
    /// Number of clock tuples (`tlast` has `n_tuples * Program::kernel_words` words; for the
    /// runtime checks of paranoid native code, like the following lengths).
    pub n_tuples: u32,
    /// Length of `parts` (partitions); `lazy` and `unsynced` have
    /// `NativeCtx::n_kernel_words` words, `kin_prev` and `kob` `NativeCtx::n_kernels`
    /// entries, `pending` room for `NativeCtx::n_kernels` pairs.
    pub n_parts: u32,
    /// Rounds completed (output).
    pub rounds: u64,
    /// Opaque transitions (output).
    pub opaque: u64,
    /// Transparent transitions (output).
    pub transparent: u64,
    /// Per-kernel fast evaluations (output).
    pub evals: u64,
}

/// A decoded change-propagation record ([`Program::notify_rec`]).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct NotifyRecord {
    /// The net.
    pub net: u32,
    /// Boundary-log slot, or [`NONE`].
    pub log: u32,
    /// `(partition word, mask)`: OR into the dirty set.
    pub parts: Vec<(u32, u64)>,
    /// `(kernel word, pending mask, data-input reader mask)`.
    pub kernels: Vec<(u32, u64, u64)>,
    /// `(kernel, input position)` of the kernel input-index entries with a position below 32
    /// (entries with position 32 are untracked and have no effect).
    pub kin: Vec<(u32, u32)>,
}

impl Program {
    /// Decode the change-propagation record at offset `off` of [`Program::notify_rec`].
    pub fn record(&self, off: u32) -> NotifyRecord {
        let r = &self.notify_rec[off as usize..];
        let h = r[0];
        let (np, nk, nkin, mut i) = if h != crate::compile::REC_ESCAPE {
            (
                (h & 0xFF) as usize,
                ((h >> 8) & 0xFF) as usize,
                (h >> 16) as usize,
                3,
            )
        } else {
            (r[3] as usize, r[4] as usize, r[5] as usize, 6)
        };
        let m64 = |lo: u32, hi: u32| u64::from(lo) | u64::from(hi) << 32;
        let mut rec = NotifyRecord {
            net: r[1],
            log: r[2],
            ..NotifyRecord::default()
        };
        for _ in 0..np {
            rec.parts.push((r[i], m64(r[i + 1], r[i + 2])));
            i += 3;
        }
        for _ in 0..nk {
            rec.kernels
                .push((r[i], m64(r[i + 1], r[i + 2]), m64(r[i + 3], r[i + 4])));
            i += 5;
        }
        // The interpreter applies the input-index entries only with kernel groups.
        if nk > 0 {
            for &e in &r[i..i + nkin] {
                let pos = (e >> KIN_POS_SHIFT) & KIN_POS_MASK;
                if pos < 32 {
                    rec.kin.push((e & KIN_KERNEL_MASK, pos));
                }
            }
        }
        rec
    }
}

/// Native code for (parts of) one [`Program`].
pub struct NativeCode {
    program: Arc<Program>,
    parts: Vec<Option<PartitionFn>>,
    /// `parts` with the interpreter trampoline in the gaps ([`EventCtx::parts`]).
    table: Vec<PartitionFn>,
    events: Vec<Option<EventFn>>,
    /// Some event function exists.
    any_events: bool,
    keep: Vec<Arc<dyn Any + Send + Sync>>,
}

impl std::fmt::Debug for NativeCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeCode")
            .field("partitions", &self.native_partitions())
            .field("events", &self.native_events())
            .finish()
    }
}

impl NativeCode {
    /// Bundle native functions for `program`: `parts[p]` evaluates partition `p` (`None`: the
    /// interpreter does), `events[t]` is the event function of clock tuple `t` (may be empty
    /// or shorter than the tuple count). `keep` holds whatever owns the code memory.
    ///
    /// # Safety
    ///
    /// Every function must implement exactly the semantics documented on [`PartitionFn`] /
    /// [`EventCtx`] for `program`, accessing memory only through the context pointers, and stay
    /// valid while `keep` is alive.
    pub unsafe fn new(
        program: Arc<Program>,
        parts: Vec<Option<PartitionFn>>,
        events: Vec<Option<EventFn>>,
        keep: Vec<Arc<dyn Any + Send + Sync>>,
    ) -> Result<NativeCode, String> {
        if parts.len() != program.partitions.len() {
            return Err(format!(
                "{} partition functions for {} partitions",
                parts.len(),
                program.partitions.len()
            ));
        }
        if events.len() > program.num_tuples as usize {
            return Err(format!(
                "{} event functions for {} clock tuples",
                events.len(),
                program.num_tuples
            ));
        }
        #[allow(unused_mut)]
        let mut table: Vec<PartitionFn> = parts
            .iter()
            .map(|p| p.unwrap_or(interp_partition_cb as PartitionFn))
            .collect();
        #[cfg(feature = "prof")]
        if std::env::var_os("PANDA_PROF_PARTS").is_some() {
            // Profiling: time every partition call (probe slot `LUT`) through a trampoline.
            let real: &'static [PartitionFn] = Box::leak(table.clone().into_boxed_slice());
            PROF_PARTS.store(real.as_ptr().cast_mut(), Ordering::Release);
            table.fill(prof_partition_cb);
        }
        let any_events = events.iter().any(Option::is_some);
        Ok(NativeCode {
            program,
            parts,
            table,
            events,
            any_events,
            keep,
        })
    }

    /// The same code with only the partitions `keep_part(p)` and the event functions
    /// `keep_event(t)` (the rest falls back to the interpreter); for testing mixed tiers.
    pub fn subset(
        &self,
        keep_part: impl Fn(usize) -> bool,
        keep_event: impl Fn(usize) -> bool,
    ) -> NativeCode {
        let parts: Vec<Option<PartitionFn>> = self
            .parts
            .iter()
            .enumerate()
            .map(|(p, f)| f.filter(|_| keep_part(p)))
            .collect();
        let events = self
            .events
            .iter()
            .enumerate()
            .map(|(t, f)| f.filter(|_| keep_event(t)))
            .collect();
        // SAFETY: a subset of valid functions for the same program (the interpreter implements
        // the removed partitions and events), with the same code owners.
        unsafe { NativeCode::new(self.program.clone(), parts, events, self.keep.clone()) }
            .expect("same shape")
    }

    /// The program this code was compiled for.
    pub fn program(&self) -> &Arc<Program> {
        &self.program
    }

    /// Number of partitions with native code.
    pub fn native_partitions(&self) -> usize {
        self.parts.iter().filter(|p| p.is_some()).count()
    }

    /// Number of clock tuples with a native event function.
    pub fn native_events(&self) -> usize {
        self.events.iter().filter(|r| r.is_some()).count()
    }

    /// True if some clock tuple has an event function.
    #[inline]
    pub fn has_events(&self) -> bool {
        self.any_events
    }

    #[inline]
    pub(crate) fn event(&self, tuple: u32) -> Option<EventFn> {
        self.events.get(tuple as usize).copied().flatten()
    }

    #[inline]
    pub(crate) fn table(&self) -> *const PartitionFn {
        self.table.as_ptr()
    }
}

/// A mailbox through which a background compiler hands [`NativeCode`] to a running
/// [`crate::FastSim`] ([`crate::FastSim::set_native_slot`]): the engine checks it (one atomic
/// load) at every event boundary and installs the code it finds there.
#[derive(Debug, Default)]
pub struct NativeSlot {
    ready: AtomicBool,
    code: Mutex<Option<Arc<NativeCode>>>,
}

impl NativeSlot {
    /// An empty slot.
    pub fn new() -> Arc<NativeSlot> {
        Arc::new(NativeSlot::default())
    }

    /// Hand `code` over (replacing code not picked up yet).
    pub fn publish(&self, code: Arc<NativeCode>) {
        *self.code.lock().unwrap_or_else(|e| e.into_inner()) = Some(code);
        self.ready.store(true, Ordering::Release);
    }

    /// True if code is waiting to be picked up.
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    /// Take the waiting code, if any.
    pub fn take(&self) -> Option<Arc<NativeCode>> {
        if !self.is_ready() {
            return None;
        }
        let mut g = self.code.lock().unwrap_or_else(|e| e.into_inner());
        self.ready.store(false, Ordering::Release);
        g.take()
    }
}

/// Host environment of the callbacks (pointed to by [`NativeCtx::env`]).
pub(crate) struct Env {
    pub prog: *const Program,
    pub rt: *mut Rt,
    pub values: *mut Logic,
    pub len: usize,
}

impl Env {
    /// The state behind a context.
    ///
    /// # Safety
    ///
    /// `ctx.env` points to a live `Env` whose pointers are valid and not otherwise borrowed for
    /// the returned lifetime.
    unsafe fn state<'a>(ctx: *mut NativeCtx) -> (&'a Program, &'a mut Rt, &'a mut [Logic]) {
        // SAFETY: see the function contract.
        unsafe {
            let env = &*((*ctx).env as *const Env);
            (
                &*env.prog,
                &mut *env.rt,
                std::slice::from_raw_parts_mut(env.values, env.len),
            )
        }
    }
}

/// [`LutColdFn`]: the interpreter's cold LUT path.
unsafe extern "C" fn lut_cold_cb(ctx: *mut NativeCtx, lut: u32) {
    // SAFETY: called by native code during `sweep_native` / `FastSim::native_event`, whose
    // `Env` is live and whose state is only accessed through it meanwhile.
    let (prog, rt, values) = unsafe { Env::state(ctx) };
    exec::run_lut_cold(prog, values, rt, lut);
}

/// [`ResolveFn`]: the interpreter's resolver op.
unsafe extern "C" fn resolve_cb(ctx: *mut NativeCtx, op: u32) {
    // SAFETY: as `lut_cold_cb`.
    let (prog, rt, values) = unsafe { Env::state(ctx) };
    exec::run_resolve_alone(prog, values, rt, op);
}

/// [`PartitionFn`] of partitions without native code: the interpreter.
unsafe extern "C" fn interp_partition_cb(ctx: *mut NativeCtx, p: u32) {
    // SAFETY: as `lut_cold_cb`.
    let (prog, rt, values) = unsafe { Env::state(ctx) };
    exec::run_partition_alone(prog, values, rt, p as usize);
}

/// The real partition functions behind [`prof_partition_cb`] (profiling only).
#[cfg(feature = "prof")]
static PROF_PARTS: std::sync::atomic::AtomicPtr<PartitionFn> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// Profiling trampoline: partition `p` timed into probe slot `LUT`.
#[cfg(feature = "prof")]
unsafe extern "C" fn prof_partition_cb(ctx: *mut NativeCtx, p: u32) {
    let t = crate::probe::start();
    // SAFETY: set to a leaked table with an entry per partition before this trampoline is
    // installed (profiling of a single program).
    unsafe { (*PROF_PARTS.load(Ordering::Acquire).add(p as usize))(ctx, p) };
    crate::probe::stop(crate::probe::LUT, t);
}

/// [`FaultFn`]: a runtime check of paranoid native code failed.
unsafe extern "C" fn fault_cb(_ctx: *mut NativeCtx, code: u32) {
    eprintln!("panda: a runtime check of native code failed (check {code}); aborting");
    std::process::abort();
}

/// [`EpochWrapFn`]: `Marks::next_epoch`'s wrap-around.
pub(crate) unsafe extern "C" fn epoch_wrap_cb(ctx: *mut NativeCtx) {
    // SAFETY: as `lut_cold_cb`.
    let (_, rt, _) = unsafe { Env::state(ctx) };
    rt.marks.bepoch.fill(u32::MAX);
    rt.marks.epoch = 0;
}

/// Build the context for `rt` / `values` (the pointers stay valid while neither is moved or
/// resized; the vectors have fixed sizes).
pub(crate) fn make_ctx(
    prog: &Program,
    values: *mut Logic,
    len: usize,
    rt: *mut Rt,
    env: *mut Env,
) -> NativeCtx {
    let len32 = |n: usize| u32::try_from(n).expect("runtime array length fits in u32");
    // SAFETY: `rt` is valid (caller); only raw pointers to its fields and buffers are taken.
    unsafe {
        let m = &mut (*rt).marks;
        NativeCtx {
            values: values.cast(),
            dirty: m.dirty.as_mut_ptr(),
            kpending: m.kpending.as_mut_ptr(),
            ddirty: m.ddirty.as_mut_ptr(),
            fastok: m.fastok.as_mut_ptr(),
            kin: m.kin.as_mut_ptr(),
            bval: m.bval.as_mut_ptr().cast(),
            bepoch: m.bepoch.as_mut_ptr(),
            epoch: m.epoch,
            n_values: len32(len),
            n_part_words: len32(m.dirty.len()),
            n_kernel_words: len32(m.kpending.len()),
            n_kernels: len32(m.kin.len()),
            n_log: len32(m.bval.len()),
            n_luts: len32((*rt).lut_last.len()),
            n_lut_entries: len32(prog.lut_entries.len()),
            fault: fault_cb,
            lut_last: (*rt).lut_last.as_mut_ptr(),
            lut_stale: (*rt).lut_stale.as_mut_ptr(),
            lut_entries: prog.lut_entries.as_ptr(),
            parts_run: &raw mut (*rt).parts_run,
            ops_run: &raw mut (*rt).ops_run,
            env: env.cast(),
            lut_cold: lut_cold_cb,
            resolve: resolve_cb,
        }
    }
}

/// [`exec::sweep`] with native partitions where `code` has them.
pub(crate) fn sweep_native(prog: &Program, values: &mut [Logic], rt: &mut Rt, code: &NativeCode) {
    debug_assert!(std::ptr::eq(prog, &*code.program));
    let rtp: *mut Rt = rt;
    let len = values.len();
    let vp = values.as_mut_ptr();
    let mut env = Env {
        prog,
        rt: rtp,
        values: vp,
        len,
    };
    let mut ctx = make_ctx(prog, vp, len, rtp, &mut env);
    // SAFETY: while native code runs, the runtime state is only accessed through `ctx` / `env`
    // (raw pointers derived from `rt` and `values`); `table` has one entry per partition.
    unsafe {
        let nw = (*rtp).marks.dirty.len();
        let dirty = ctx.dirty;
        let table = code.table();
        for w in 0..nw {
            loop {
                let bits = *dirty.add(w);
                if bits == 0 {
                    break;
                }
                *dirty.add(w) = bits & (bits - 1);
                let p = w * 64 + bits.trailing_zeros() as usize;
                (*table.add(p))(&mut ctx, p as u32);
            }
        }
    }
}
