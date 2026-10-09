//! The simulation host: a dedicated thread that owns the [`Kernel`], and the UI-side handle.
//!
//! ```text
//!  UI thread                                   sim thread (owns Kernel)
//!  SimHost::send(Command) ── bounded channel ─▶ drained at chunk boundaries (one atomic
//!                                               "pending" flag checked per chunk)
//!                                               loop { kernel.run(chunk ≈ 1.5 ms) ;
//!                                                      every ≥ 1/rate s: publish }
//!  SimHost::snapshot() ◀── triple buffer ────── Snapshot: tick, mode, engine, stats, values of
//!                         (lock-free, latest)    the watched nets only
//!  SimHost::poll_events() ◀── bounded channel ── SimEvent: load/edit results, diagnostics,
//!                                               waveform chunks (never blocks the sim)
//! ```
//!
//! The UI never waits on the sim thread and vice versa: commands are queued locally when the
//! channel is full, publication is a lock-free buffer swap, and events that do not fit are kept
//! in a sim-side outbox (waveform chunks there are dropped first, marked as gaps).
//!
//! Net indices in [`WatchSet`] / [`WaveConfig`] refer to the netlist of a *design generation*
//! ([`Snapshot::design_gen`], bumped on every successful load/edit/reset and announced by
//! [`SimEvent::Netlist`]); watch sets for another generation are ignored.
//!
//! # Waveforms
//!
//! [`Command::SetWave`] selects signals; changes reach the UI as [`SimEvent::Wave`] chunks
//! ([`WaveStore`] keeps the history). With the default [`WaveCapture::Shadow`] the sim thread
//! does **not** sample: it hands a checkpoint (kernel snapshot + input changes) to a shadow
//! thread at each publication, and the shadow replays the interval on its own kernel with the
//! observer attached (module `shadow`). Measured on Panda-8 (Fast/JIT), sampling on the sim
//! thread ([`WaveCapture::Inline`]) costs ≈ 0.45 % throughput per sampled net (≈ 17 % for
//! 8 bus signals); shadow capture costs one snapshot per publication (≈ 15 µs at 120 Hz). Clock
//! nets are never sampled: [`WaveStore`] synthesizes them.
//!
//! Extension point: [`Command::Custom`] runs a closure on the sim thread with the kernel (e.g.
//! timing reports, classification) and can answer with a [`SimEvent::Custom`] payload.

mod shadow;
mod thread;
pub mod triple;
pub mod wave;

use std::any::Any;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::thread::JoinHandle;
use std::time::Duration;

use panda_kernel::{DiagEntry, EditOp, EngineKind, Kernel, KernelOptions, Mode, StepUnit};
use panda_netlist::flat::{FlatNetlist, NetIdx};
use panda_netlist::{Design, Diagnostic, Logic, Value};

pub use wave::{
    ClockSpec, Packed, WaveChange, WaveChunk, WaveConfig, WaveRecorder, WaveSignal, WaveStore,
};

/// Simulation speed.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Speed {
    /// As fast as possible.
    #[default]
    Unlimited,
    /// Target clock cycles (phases for clockless designs) per second.
    CyclesPerSec(f64),
}

/// The flat nets whose values every snapshot carries (the visible view's nets).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WatchSet {
    /// Caller-chosen id, echoed as [`Snapshot::watch_id`].
    pub id: u64,
    /// Design generation the indices refer to.
    pub design_gen: u64,
    /// Flat nets; [`Snapshot::values`] is aligned with this list.
    pub nets: Vec<NetIdx>,
}

/// A closure run on the sim thread by [`Command::Custom`].
pub type CustomFn = Box<dyn FnOnce(&mut Kernel) -> Option<SimEvent> + Send>;

/// A command to the sim thread.
pub enum Command {
    /// Load a design (replaces the current one; the mode is kept).
    Load(Box<Design>),
    /// Apply structural edits; answered by [`SimEvent::Edited`] with the same id.
    Edit {
        /// Caller-chosen id.
        id: u64,
        /// The batch (atomic).
        ops: Vec<EditOp>,
    },
    /// Set a top-level input pin.
    SetInput {
        /// Pin name.
        pin: String,
        /// New value.
        value: Value,
    },
    /// Switch mode (Fast mode compiles in the background).
    SetMode(Mode),
    /// Run continuously.
    Run,
    /// Stop running.
    Pause,
    /// Advance by `n` units (also while paused).
    Step {
        /// Count.
        n: u64,
        /// Unit.
        unit: StepUnit,
    },
    /// Speed limit.
    SetSpeed(Speed),
    /// Power-on reset: reload the current design (inputs keep their values).
    Reset,
    /// Replace the watched view nets.
    SetWatch(WatchSet),
    /// Replace the waveform signals (empty = no observer, zero cost).
    SetWave(WaveConfig),
    /// Snapshot publication rate cap (Hz).
    SetPublishRate(f64),
    /// Run chunk length (between command checks).
    SetChunk(Duration),
    /// Run a closure on the sim thread (extension point for other panels).
    Custom(CustomFn),
    /// Stop the thread.
    Quit,
}

impl std::fmt::Debug for Command {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Command::Load(d) => write!(f, "Load({})", d.top),
            Command::Edit { id, ops } => write!(f, "Edit {{ id: {id}, ops: {} }}", ops.len()),
            Command::SetInput { pin, value } => write!(f, "SetInput({pin}, {value:?})"),
            Command::SetMode(m) => write!(f, "SetMode({m:?})"),
            Command::Run => write!(f, "Run"),
            Command::Pause => write!(f, "Pause"),
            Command::Step { n, unit } => write!(f, "Step({n}, {unit:?})"),
            Command::SetSpeed(s) => write!(f, "SetSpeed({s:?})"),
            Command::Reset => write!(f, "Reset"),
            Command::SetWatch(w) => write!(f, "SetWatch(id {}, {} nets)", w.id, w.nets.len()),
            Command::SetWave(w) => write!(f, "SetWave(id {}, {} signals)", w.id, w.signals.len()),
            Command::SetPublishRate(r) => write!(f, "SetPublishRate({r})"),
            Command::SetChunk(d) => write!(f, "SetChunk({d:?})"),
            Command::Custom(_) => write!(f, "Custom"),
            Command::Quit => write!(f, "Quit"),
        }
    }
}

/// Something the sim thread tells the UI (not lossy, except [`SimEvent::Wave`] under overload).
pub enum SimEvent {
    /// A new netlist is active (after load, edit or reset): net indices of `design_gen`.
    Netlist {
        /// The generation.
        design_gen: u64,
        /// Its flat netlist.
        flat: Arc<FlatNetlist>,
    },
    /// A load failed.
    LoadFailed(Vec<Diagnostic>),
    /// Result of [`Command::Edit`].
    Edited {
        /// The command's id.
        id: u64,
        /// Ok, or the validation errors (the edit was rolled back).
        result: Result<(), Vec<Diagnostic>>,
    },
    /// A command could not be carried out (e.g. unknown input pin).
    Error(String),
    /// The kernel's deduplicated diagnostics changed (the full list).
    Diagnostics(Arc<Vec<DiagEntry>>),
    /// Recorded waveform changes.
    Wave(WaveChunk),
    /// Answer of a [`Command::Custom`].
    Custom(Box<dyn Any + Send>),
}

impl std::fmt::Debug for SimEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimEvent::Netlist { design_gen, flat } => {
                write!(f, "Netlist(gen {design_gen}, {} nets)", flat.num_nets)
            }
            SimEvent::LoadFailed(d) => write!(f, "LoadFailed({d:?})"),
            SimEvent::Edited { id, result } => write!(f, "Edited({id}, {result:?})"),
            SimEvent::Error(e) => write!(f, "Error({e})"),
            SimEvent::Diagnostics(d) => write!(f, "Diagnostics({})", d.len()),
            SimEvent::Wave(w) => write!(f, "Wave({} changes)", w.changes.len()),
            SimEvent::Custom(_) => write!(f, "Custom"),
        }
    }
}

/// Simulation statistics in a [`Snapshot`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SimStats {
    /// Clock cycles simulated since the last load/reset.
    pub cycles: u64,
    /// Measured cycles per second (over roughly the last half second of running).
    pub cycles_per_sec: f64,
    /// Measured ticks per second.
    pub ticks_per_sec: f64,
    /// Fraction of wall time spent inside `Kernel::run` / `step` (last window).
    pub busy: f64,
    /// Snapshots published so far.
    pub publications: u64,
    /// Mean cost of one publication (µs).
    pub publish_us: f64,
}

/// What the UI sees of the simulation (published by the sim thread, lock-free).
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    /// Publication counter (0 = nothing published yet).
    pub seq: u64,
    /// Design generation (0 = no design).
    pub design_gen: u64,
    /// [`WatchSet::id`] the values belong to (0 = none).
    pub watch_id: u64,
    /// Kernel tick.
    pub tick: u64,
    /// Mode.
    pub mode: Mode,
    /// Serving engine.
    pub engine: Option<EngineKind>,
    /// The JIT tier is active.
    pub jit: bool,
    /// Running continuously.
    pub running: bool,
    /// The last run made no progress (nothing can change any more).
    pub idle: bool,
    /// Speed limit.
    pub speed: Speed,
    /// Statistics.
    pub stats: SimStats,
    /// Values of the watched nets (aligned with the [`WatchSet`] `watch_id`).
    pub values: Vec<Logic>,
    /// Number of deduplicated diagnostics and how many are errors / warnings.
    pub diag_count: usize,
    /// Error-severity diagnostics.
    pub diag_errors: usize,
    /// Warning-severity diagnostics.
    pub diag_warnings: usize,
}

impl Snapshot {
    /// Short name of the serving engine tier: `"—"` (no design), `"Timing"`,
    /// `"Timing (compiling)"`, `"Reference"`, `"FastSim"`, `"JIT"`.
    pub fn tier_label(&self) -> &'static str {
        use panda_kernel::FastBackend;
        match self.engine {
            None | Some(EngineKind::None) => "—",
            Some(EngineKind::Timing) => "Timing",
            Some(EngineKind::TimingFallback) => "Timing (compiling)",
            Some(EngineKind::Fast(FastBackend::Reference)) => "Reference",
            Some(EngineKind::Fast(_)) if self.jit => "JIT",
            Some(EngineKind::Fast(_)) => "FastSim",
        }
    }
}

/// Where waveform samples are taken.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WaveCapture {
    /// On a shadow thread replaying the main run from checkpoints (≈ zero cost for the main
    /// simulation; may skip intervals while the sim runs flat out and the shadow cannot keep up,
    /// reported as gaps; complete while stepping, speed-limited or paused).
    #[default]
    Shadow,
    /// On the sim thread after every event (lossless; costs ≈ 0.45 % of Panda-8 throughput per
    /// sampled net).
    Inline,
}

/// Options of the sim thread.
#[derive(Clone, Debug)]
pub struct SimOptions {
    /// Kernel options.
    pub kernel: KernelOptions,
    /// Snapshot publication cap (Hz).
    pub publish_hz: f64,
    /// Run chunk between command checks.
    pub chunk: Duration,
    /// Command channel capacity.
    pub command_capacity: usize,
    /// Event channel capacity.
    pub event_capacity: usize,
    /// Waveform capture strategy.
    pub wave_capture: WaveCapture,
}

impl Default for SimOptions {
    fn default() -> Self {
        SimOptions {
            kernel: KernelOptions::default(),
            publish_hz: 120.0,
            chunk: Duration::from_micros(1500),
            command_capacity: 1024,
            event_capacity: 256,
            wave_capture: WaveCapture::default(),
        }
    }
}

/// UI-side handle of the sim thread. Dropping it stops the thread.
pub struct SimHost {
    tx: SyncSender<Command>,
    pending: Arc<AtomicBool>,
    snap: triple::Reader<Snapshot>,
    events: Receiver<SimEvent>,
    backlog: VecDeque<Command>,
    thread: Option<JoinHandle<()>>,
}

impl SimHost {
    /// Start the sim thread.
    pub fn spawn(opts: SimOptions) -> SimHost {
        let (tx, rx) = sync_channel(opts.command_capacity.max(1));
        let (etx, erx) = sync_channel(opts.event_capacity.max(1));
        let pending = Arc::new(AtomicBool::new(false));
        let (w, r) = triple::triple_buffer(Snapshot::default());
        let p = pending.clone();
        let thread = std::thread::Builder::new()
            .name("panda-sim".into())
            .spawn(move || thread::SimThread::new(opts, rx, p, w, etx).run())
            .expect("spawn sim thread");
        SimHost {
            tx,
            pending,
            snap: r,
            events: erx,
            backlog: VecDeque::new(),
            thread: Some(thread),
        }
    }

    /// Send a command. Never blocks: if the channel is full the command is queued locally and
    /// retried on the next `send` / [`SimHost::flush`].
    pub fn send(&mut self, cmd: Command) {
        self.backlog.push_back(cmd);
        self.flush();
    }

    /// Retry queued commands. Returns the number still queued.
    pub fn flush(&mut self) -> usize {
        let mut sent = false;
        while let Some(cmd) = self.backlog.pop_front() {
            match self.tx.try_send(cmd) {
                Ok(()) => sent = true,
                Err(TrySendError::Full(cmd)) => {
                    self.backlog.push_front(cmd);
                    break;
                }
                Err(TrySendError::Disconnected(_)) => {
                    self.backlog.clear();
                    break;
                }
            }
        }
        if sent {
            self.pending.store(true, Ordering::Release);
        }
        self.backlog.len()
    }

    /// The latest snapshot (never blocks).
    pub fn snapshot(&mut self) -> &Snapshot {
        self.snap.read()
    }

    /// True if a snapshot was published since the last [`SimHost::snapshot`].
    pub fn has_new_snapshot(&self) -> bool {
        self.snap.has_fresh()
    }

    /// Drain the events that arrived (never blocks).
    pub fn poll_events(&mut self) -> Vec<SimEvent> {
        self.flush();
        self.events.try_iter().collect()
    }

    /// Wait up to `timeout` for the next event (for tests and headless tools; the UI uses
    /// [`SimHost::poll_events`]).
    pub fn wait_event(&mut self, timeout: Duration) -> Option<SimEvent> {
        self.flush();
        self.events.recv_timeout(timeout).ok()
    }

    /// Stop the thread and wait for it.
    pub fn shutdown(&mut self) {
        if let Some(t) = self.thread.take() {
            self.backlog.clear();
            // The channel may be full; keep trying while the thread drains it.
            let mut cmd = Command::Quit;
            loop {
                match self.tx.try_send(cmd) {
                    Ok(()) => break,
                    Err(TrySendError::Full(c)) => {
                        cmd = c;
                        self.pending.store(true, Ordering::Release);
                        while self.events.try_recv().is_ok() {}
                        std::thread::yield_now();
                    }
                    Err(TrySendError::Disconnected(_)) => break,
                }
            }
            self.pending.store(true, Ordering::Release);
            // Unblock a sim thread waiting to hand over events.
            while !t.is_finished() {
                while self.events.try_recv().is_ok() {}
                std::thread::sleep(Duration::from_millis(1));
            }
            let _ = t.join();
        }
    }
}

impl Drop for SimHost {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests;
