//! Shadow waveform capture: per-event sampling without slowing the main simulation.
//!
//! Sampling on the sim thread costs one gather of the watched nets per event, measured at
//! ≈ 0.45 % of Panda-8 throughput per sampled net (an event takes only ~150 ns). Instead, the
//! sim thread sends cheap **checkpoints** ([`Checkpoint`]: a `Kernel::snapshot` plus the input
//! changes since the previous one) at its publication rate, and this thread *replays* each
//! interval on its own [`Kernel`] with the [`WaveRecorder`] attached. The simulation is
//! deterministic, so the replay reproduces every event of the main run exactly.
//!
//! * Continuous intervals are replayed back to back without reloading state; a checkpoint
//!   marked `reload` (after a mode switch, a custom command or a new waveform config) or one
//!   following a lost interval is loaded with `Kernel::restore`.
//! * When the shadow falls behind (it runs the same simulation plus sampling on another core),
//!   it skips to the newest checkpoints and reports a gap; with a speed limit, while stepping or
//!   once paused it catches up, so the history up to the current tick is complete.
//! * Wave chunks go straight to the UI's event channel.

use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};
use std::thread::JoinHandle;

use panda_kernel::{EventObserver, Kernel, KernelOptions, State, StepUnit};
use panda_netlist::{Design, Logic, Value};

use super::{SimEvent, WaveChunk, WaveConfig, WaveRecorder};

/// Checkpoints kept when the shadow is behind (older ones are skipped).
const MAX_BACKLOG: usize = 3;
/// Wave chunks kept while the UI does not drain its channel.
const MAX_OUTBOX: usize = 64;

/// One interval of the main simulation, as the shadow needs it.
pub(super) struct Checkpoint {
    /// Design generation.
    pub gen_: u64,
    /// Consecutive number (a missing number = a lost interval).
    pub seq: u64,
    /// Input changes since the previous checkpoint, with the main kernel tick when each was given.
    pub inputs: Vec<(u64, String, Value)>,
    /// Replay the interval from the previous checkpoint up to this tick.
    pub replay_to: u64,
    /// The main state after the interval.
    pub state: State,
    /// The state is not the end of the replayed interval (a command changed it): load it.
    pub reload: bool,
}

/// Messages to the shadow thread.
pub(super) enum ShadowMsg {
    /// A new design generation (reloads the shadow kernel).
    Design { gen_: u64, design: Box<Design> },
    /// New waveform signals (empty: idle).
    Config(WaveConfig),
    /// An interval to replay.
    Checkpoint(Box<Checkpoint>),
    /// Stop.
    Quit,
}

/// Handle owned by the sim thread.
pub(super) struct Shadow {
    tx: SyncSender<ShadowMsg>,
    thread: Option<JoinHandle<()>>,
}

impl Shadow {
    /// Start the shadow thread; wave chunks go to `etx`.
    pub(super) fn spawn(kernel: KernelOptions, etx: SyncSender<SimEvent>) -> Shadow {
        let (tx, rx) = std::sync::mpsc::sync_channel(64);
        let thread = std::thread::Builder::new()
            .name("panda-wave-shadow".into())
            .spawn(move || ShadowThread::new(kernel, rx, etx).run())
            .expect("spawn shadow thread");
        Shadow {
            tx,
            thread: Some(thread),
        }
    }

    /// Send without blocking; false if the queue is full (the message is dropped).
    pub(super) fn send(&self, msg: ShadowMsg) -> bool {
        match self.tx.try_send(msg) {
            Ok(()) => true,
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => false,
        }
    }

    /// Send, waiting for room (for control messages that must not be lost).
    pub(super) fn send_blocking(&self, msg: ShadowMsg) {
        let _ = self.tx.send(msg);
    }
}

impl Drop for Shadow {
    fn drop(&mut self) {
        let _ = self.tx.send(ShadowMsg::Quit);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

struct ShadowThread {
    opts: KernelOptions,
    rx: Receiver<ShadowMsg>,
    etx: SyncSender<SimEvent>,
    outbox: VecDeque<SimEvent>,
    kernel: Option<Kernel>,
    gen_: u64,
    recorder: WaveRecorder,
    /// Sequence number of the checkpoint the shadow kernel is synchronized with.
    synced: Option<u64>,
    gap: bool,
}

impl ShadowThread {
    fn new(
        mut opts: KernelOptions,
        rx: Receiver<ShadowMsg>,
        etx: SyncSender<SimEvent>,
    ) -> ShadowThread {
        // Compile synchronously: this thread may block, the main sim never waits for it.
        opts.background = false;
        ShadowThread {
            opts,
            rx,
            etx,
            outbox: VecDeque::new(),
            kernel: None,
            gen_: 0,
            recorder: WaveRecorder::default(),
            synced: None,
            gap: false,
        }
    }

    fn run(mut self) {
        loop {
            let first = match self.rx.recv_timeout(std::time::Duration::from_millis(50)) {
                Ok(m) => m,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    self.flush_outbox();
                    continue;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return,
            };
            let mut batch = vec![first];
            batch.extend(self.rx.try_iter());
            // Keep only the newest checkpoints after the last control message.
            let last_ctl = batch
                .iter()
                .rposition(|m| !matches!(m, ShadowMsg::Checkpoint(_)));
            let start_cp = last_ctl.map_or(0, |i| i + 1);
            let n_cp = batch.len() - start_cp;
            let skip = n_cp.saturating_sub(MAX_BACKLOG);
            for (i, msg) in batch.into_iter().enumerate() {
                match msg {
                    ShadowMsg::Quit => return,
                    ShadowMsg::Design { gen_, design } => self.load(gen_, *design),
                    ShadowMsg::Config(cfg) => {
                        self.flush_recorder();
                        self.recorder = WaveRecorder::new(cfg);
                        self.synced = None;
                    }
                    ShadowMsg::Checkpoint(c) => {
                        if i >= start_cp && i - start_cp < skip {
                            self.synced = None; // skipped: the next one reloads
                            self.gap = true;
                            continue;
                        }
                        self.checkpoint(*c);
                    }
                }
            }
            self.flush_recorder();
            self.flush_outbox();
        }
    }

    fn load(&mut self, gen_: u64, design: Design) {
        self.flush_recorder();
        self.gen_ = gen_;
        self.synced = None;
        self.recorder = WaveRecorder::default();
        let mut k = Kernel::new(self.opts.clone());
        self.kernel = k.load(design).is_ok().then_some(k);
    }

    fn checkpoint(&mut self, c: Checkpoint) {
        if c.gen_ != self.gen_ || self.recorder.is_empty() {
            return;
        }
        let Some(k) = self.kernel.as_mut() else {
            return;
        };
        let continuous = self.synced.is_some_and(|s| s + 1 == c.seq);
        if continuous {
            for (t, pin, v) in &c.inputs {
                if *t > k.tick() {
                    k.step_observed(t - k.tick(), StepUnit::Tick, &mut self.recorder);
                }
                let _ = k.set_input(pin, v);
            }
            if c.replay_to > k.tick() {
                k.step_observed(c.replay_to - k.tick(), StepUnit::Tick, &mut self.recorder);
            }
        } else if self.synced.is_some() || self.gap {
            self.gap = true;
        }
        if !continuous || c.reload {
            k.set_mode(c.state.mode());
            let _ = k.wait_ready();
            if k.restore(&c.state).is_err() {
                self.synced = None;
                return;
            }
            let nets = self.recorder.nets().to_vec();
            let mut v = vec![Logic::X; nets.len()];
            k.copy_values(&nets, &mut v);
            self.recorder.prime(k.tick(), &v);
        }
        self.synced = Some(c.seq);
    }

    fn flush_recorder(&mut self) {
        if let Some(mut chunk) = self.recorder.take_chunk() {
            chunk.gap |= std::mem::take(&mut self.gap);
            self.emit(chunk);
        }
    }

    fn emit(&mut self, chunk: WaveChunk) {
        if self.outbox.len() >= MAX_OUTBOX {
            self.gap = true;
            return;
        }
        self.outbox.push_back(SimEvent::Wave(chunk));
        self.flush_outbox();
    }

    fn flush_outbox(&mut self) {
        while let Some(ev) = self.outbox.pop_front() {
            match self.etx.try_send(ev) {
                Ok(()) => {}
                Err(TrySendError::Full(ev)) => {
                    self.outbox.push_front(ev);
                    break;
                }
                Err(TrySendError::Disconnected(_)) => {
                    self.outbox.clear();
                    break;
                }
            }
        }
    }
}
