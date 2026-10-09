//! Shadow waveform capture: per-event sampling without slowing the main simulation.
//!
//! Sampling on the sim thread costs one gather of the watched nets per event, measured at
//! ≈ 0.45 % of Panda-8 throughput per sampled net (an event takes only ~150 ns). Instead, the
//! sim thread sends cheap **checkpoints** ([`Checkpoint`]: a `Kernel::snapshot_into` a pooled
//! [`State`] plus the input changes since the previous checkpoint) at its publication rate, and
//! shadow threads *replay* each interval on their own [`Kernel`] with a [`WaveRecorder`]
//! attached. The simulation is deterministic, so the replay reproduces every event of the main
//! run exactly.
//!
//! ```text
//!  sim thread ── Checkpoint i (state S_i) ──▶ coordinator ── Job(S_{i-1} → tick_i) ──▶ worker 0..N
//!       ▲                                       │  ▲                                    │
//!       └──── State buffers (pool) ◀── drop ────┘  └────── Done(job, chunk) ────────────┘
//!                                               │ reorders, removes redundant samples
//!                                               ▼
//!                                      UI event channel (SimEvent::Wave)
//! ```
//!
//! * Every interval `S_{i-1} → S_i` is an independent **job**: a worker restores `S_{i-1}`,
//!   records the values there and replays the input changes up to the interval's end. Jobs run
//!   on up to [`super::SimOptions::shadow_workers`] workers **in parallel** (workers are spawned
//!   on demand), so capture keeps up with a flat-out main run although a replay with sampling is
//!   slower than the main simulation. The coordinator emits results strictly in job order and
//!   drops samples that repeat the previously emitted value (interval boundaries).
//! * When every worker is busy and the backlog is full, the oldest queued job is dropped and the
//!   next chunk is flagged as a gap; while paused, stepping or speed-limited capture is complete.
//! * Nothing here ever blocks the sim thread: all channels are unbounded and the sim thread
//!   bounds the checkpoints in flight with one atomic counter ([`Shadow::send_checkpoint`]).
//!   State buffers return to the sim thread through a pool, so steady-state checkpoints do not
//!   allocate.
//! * Shutdown: workers check a quit flag between replay slices, so dropping the [`Shadow`] never
//!   waits for a long replay.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, SyncSender, TrySendError, channel};
use std::thread::JoinHandle;
use std::time::Duration;

use panda_kernel::{EventObserver, Kernel, KernelOptions, State, StepUnit};
use panda_netlist::{Design, Logic, Value};

use super::{Packed, SimEvent, WaveChunk, WaveConfig, WaveRecorder};

/// Checkpoints the sim thread may have in flight to the coordinator (beyond it they are skipped,
/// which the coordinator sees as a missing sequence number).
const MAX_INFLIGHT: usize = 32;
/// Jobs queued per worker while every worker is busy (beyond it the oldest is dropped: a gap).
const BACKLOG_PER_WORKER: usize = 2;
/// Wave chunks kept while the UI does not drain its channel.
const MAX_OUTBOX: usize = 64;
/// How long the coordinator waits for its workers at shutdown before detaching them.
const WORKER_JOIN_WAIT: Duration = Duration::from_millis(500);
/// Replay slices between quit-flag checks.
const SLICES: u64 = 8;

/// A pooled [`State`]: returns its buffers to the sim thread when the last reference drops.
pub(super) struct PooledState {
    state: Option<State>,
    home: Sender<State>,
}

impl PooledState {
    fn get(&self) -> &State {
        self.state.as_ref().expect("state present until drop")
    }
}

impl Drop for PooledState {
    fn drop(&mut self) {
        if let Some(s) = self.state.take() {
            // The sim thread may be gone (shutdown): then the buffer is simply freed.
            let _ = self.home.send(s);
        }
    }
}

/// One interval of the main simulation, as the shadow needs it.
pub(super) struct Checkpoint {
    /// Design generation.
    pub gen_: u64,
    /// Consecutive number (a missing number = a lost interval).
    pub seq: u64,
    /// Input changes since the previous checkpoint, with the main kernel tick when each was given.
    pub inputs: Vec<(u64, String, Value)>,
    /// The interval since the previous checkpoint ends at this tick.
    pub replay_to: u64,
    /// The main state now (the start of the next interval).
    pub state: Arc<PooledState>,
}

/// Messages to the coordinator.
pub(super) enum ShadowMsg {
    /// A new design generation (reloads the workers' kernels).
    Design { gen_: u64, design: Box<Design> },
    /// New waveform signals (empty: idle).
    Config(WaveConfig),
    /// An interval to replay.
    Checkpoint(Checkpoint),
    /// A worker finished a job.
    Done {
        worker: usize,
        job: u64,
        chunk: Option<WaveChunk>,
    },
    /// Stop.
    Quit,
}

/// Handle owned by the sim thread.
pub(super) struct Shadow {
    tx: Sender<ShadowMsg>,
    inflight: Arc<AtomicUsize>,
    pool: Receiver<State>,
    pool_home: Sender<State>,
    quit: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    /// Checkpoints skipped because the coordinator was behind.
    pub skipped: u64,
}

impl Shadow {
    /// Start the coordinator (workers start on demand, at most `workers`); wave chunks go to
    /// `etx`; a crash of a shadow thread is reported there as [`SimEvent::Error`].
    pub(super) fn spawn(
        kernel: KernelOptions,
        etx: SyncSender<SimEvent>,
        workers: usize,
    ) -> Shadow {
        let (tx, rx) = channel();
        let (pool_home, pool) = channel();
        let inflight = Arc::new(AtomicUsize::new(0));
        let quit = Arc::new(AtomicBool::new(false));
        let coord = Coordinator {
            opts: kernel,
            rx,
            tx: tx.clone(),
            etx,
            inflight: inflight.clone(),
            quit: quit.clone(),
            max_workers: workers.max(1),
            workers: Vec::new(),
            design: None,
            config: None,
            prev: None,
            next_job: 0,
            epoch: 0,
            queue: VecDeque::new(),
            done: BTreeMap::new(),
            next_emit: 0,
            last: Vec::new(),
            gap: false,
            outbox: VecDeque::new(),
        };
        let thread = std::thread::Builder::new()
            .name("panda-wave-shadow".into())
            .spawn(move || coord.run())
            .expect("spawn shadow thread");
        Shadow {
            tx,
            inflight,
            pool,
            pool_home,
            quit,
            thread: Some(thread),
            skipped: 0,
        }
    }

    /// Send a control message (never blocks).
    pub(super) fn send(&self, msg: ShadowMsg) {
        let _ = self.tx.send(msg);
    }

    /// Snapshot `kernel` into a pooled state buffer (no allocation once the pool is warm).
    pub(super) fn snapshot(&self, kernel: &Kernel) -> Arc<PooledState> {
        let state = match self.pool.try_recv() {
            Ok(mut s) => {
                kernel.snapshot_into(&mut s);
                s
            }
            Err(_) => kernel.snapshot(),
        };
        Arc::new(PooledState {
            state: Some(state),
            home: self.pool_home.clone(),
        })
    }

    /// True if a checkpoint can be sent now (fewer than [`MAX_INFLIGHT`] unprocessed).
    pub(super) fn has_room(&self) -> bool {
        self.inflight.load(Ordering::Relaxed) < MAX_INFLIGHT
    }

    /// Send a checkpoint (never blocks; see [`Shadow::has_room`]).
    pub(super) fn send_checkpoint(&self, c: Checkpoint) {
        self.inflight.fetch_add(1, Ordering::Relaxed);
        let _ = self.tx.send(ShadowMsg::Checkpoint(c));
    }
}

impl Drop for Shadow {
    fn drop(&mut self) {
        self.quit.store(true, Ordering::Relaxed);
        let _ = self.tx.send(ShadowMsg::Quit);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// An interval to replay: restore `start`, record, apply `inputs`, run to `end`.
struct Job {
    id: u64,
    gen_: u64,
    config: u64,
    start: Arc<PooledState>,
    inputs: Vec<(u64, String, Value)>,
    end: u64,
}

enum WorkerMsg {
    Design { gen_: u64, design: Arc<Design> },
    Config(Arc<WaveConfig>),
    Job(Box<Job>),
}

struct WorkerHandle {
    tx: Sender<WorkerMsg>,
    /// Jobs sent and not finished.
    busy: usize,
    thread: Option<JoinHandle<()>>,
}

struct Coordinator {
    opts: KernelOptions,
    rx: Receiver<ShadowMsg>,
    /// For the workers' results.
    tx: Sender<ShadowMsg>,
    etx: SyncSender<SimEvent>,
    inflight: Arc<AtomicUsize>,
    quit: Arc<AtomicBool>,
    max_workers: usize,
    workers: Vec<WorkerHandle>,
    design: Option<(u64, Arc<Design>)>,
    config: Option<Arc<WaveConfig>>,
    /// The last checkpoint (start of the next interval).
    prev: Option<(u64, Arc<PooledState>)>,
    next_job: u64,
    /// First job id of the current design / config (older results are discarded).
    epoch: u64,
    /// Jobs waiting for a worker.
    queue: VecDeque<Box<Job>>,
    /// Finished (or dropped: `None`) jobs not emitted yet.
    done: BTreeMap<u64, Option<WaveChunk>>,
    next_emit: u64,
    /// Last emitted value per signal (removes repeated samples at interval boundaries).
    last: Vec<Option<Packed>>,
    gap: bool,
    outbox: VecDeque<SimEvent>,
}

impl Coordinator {
    fn run(mut self) {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.serve()));
        if let Err(p) = r {
            let msg = super::panic_message(p.as_ref());
            let _ = self.etx.try_send(SimEvent::Error(format!(
                "waveform capture thread panicked: {msg}"
            )));
        }
        self.quit.store(true, Ordering::Relaxed);
        for w in &mut self.workers {
            // Closing the channel stops the worker.
            let (dead, _) = channel();
            w.tx = dead;
        }
        // Workers stop at their next quit check; one still compiling a design (synchronous) is
        // detached rather than waited for, so the sim thread's shutdown stays fast.
        let deadline = std::time::Instant::now() + WORKER_JOIN_WAIT;
        for w in &mut self.workers {
            if let Some(t) = w.thread.take() {
                while !t.is_finished() && std::time::Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(1));
                }
                if t.is_finished() {
                    let _ = t.join();
                }
            }
        }
    }

    fn serve(&mut self) {
        loop {
            let msg = match self.rx.recv_timeout(Duration::from_millis(50)) {
                Ok(m) => m,
                Err(RecvTimeoutError::Timeout) => {
                    self.flush_outbox();
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => return,
            };
            match msg {
                ShadowMsg::Quit => return,
                ShadowMsg::Design { gen_, design } => {
                    let design: Arc<Design> = Arc::from(design);
                    self.design = Some((gen_, design.clone()));
                    self.config = None;
                    self.new_epoch();
                    for w in &self.workers {
                        let _ = w.tx.send(WorkerMsg::Design {
                            gen_,
                            design: design.clone(),
                        });
                    }
                }
                ShadowMsg::Config(cfg) => {
                    let cfg = Arc::new(cfg);
                    self.new_epoch();
                    self.last = vec![None; cfg.signals.len()];
                    for w in &self.workers {
                        let _ = w.tx.send(WorkerMsg::Config(cfg.clone()));
                    }
                    self.config = Some(cfg);
                }
                ShadowMsg::Checkpoint(c) => {
                    self.inflight.fetch_sub(1, Ordering::Relaxed);
                    self.checkpoint(c);
                }
                ShadowMsg::Done { worker, job, chunk } => {
                    if let Some(w) = self.workers.get_mut(worker) {
                        w.busy = w.busy.saturating_sub(1);
                    }
                    if job >= self.epoch {
                        self.done.insert(job, chunk);
                    }
                    self.dispatch();
                    self.emit_ready();
                }
            }
            self.flush_outbox();
        }
    }

    /// Forget queued jobs and pending results (a new design or config).
    fn new_epoch(&mut self) {
        self.prev = None;
        self.queue.clear();
        self.done.clear();
        self.epoch = self.next_job;
        self.next_emit = self.next_job;
        self.gap = false;
    }

    /// True if the current config samples something (workers are worth spawning).
    fn active(&self) -> bool {
        self.design.is_some()
            && self.config.as_ref().is_some_and(|c| {
                c.signals
                    .iter()
                    .any(|s| s.clock.is_none() && !s.nets.is_empty())
            })
    }

    fn checkpoint(&mut self, c: Checkpoint) {
        let Some((gen_, _)) = &self.design else {
            return;
        };
        if c.gen_ != *gen_ || self.config.is_none() {
            return;
        }
        let config = self.config.as_ref().map_or(0, |c| c.id);
        let (start, inputs, end) = match self.prev.take() {
            Some((seq, start)) if seq + 1 == c.seq => (start, c.inputs, c.replay_to),
            prev => {
                // First checkpoint, or intervals were lost: record the values at this state.
                if prev.is_some() {
                    self.gap = true;
                }
                let tick = c.state.get().tick();
                (c.state.clone(), Vec::new(), tick)
            }
        };
        self.prev = Some((c.seq, c.state));
        let id = self.next_job;
        self.next_job += 1;
        self.queue.push_back(Box::new(Job {
            id,
            gen_: c.gen_,
            config,
            start,
            inputs,
            end,
        }));
        self.dispatch();
        // Every worker busy and the backlog full: drop the oldest queued interval.
        let cap = BACKLOG_PER_WORKER * self.max_workers;
        while self.queue.len() > cap {
            let j = self.queue.pop_front().expect("non-empty");
            self.done.insert(j.id, None);
        }
        self.emit_ready();
    }

    /// Hand queued jobs to idle workers (spawning workers on demand).
    fn dispatch(&mut self) {
        while !self.queue.is_empty() {
            let idle = self.workers.iter().position(|w| w.busy == 0);
            let w = match idle {
                Some(w) => w,
                None if self.workers.len() < self.max_workers && self.active() => {
                    self.spawn_worker()
                }
                // Pipeline one more job per worker while all are busy.
                None => match self.workers.iter().position(|w| w.busy < 2) {
                    Some(w) => w,
                    None => return,
                },
            };
            let job = self.queue.pop_front().expect("non-empty");
            self.workers[w].busy += 1;
            if self.workers[w].tx.send(WorkerMsg::Job(job)).is_err() {
                // A dead worker: its jobs never finish; treat as lost.
                self.workers[w].busy = usize::MAX / 2;
            }
        }
    }

    fn spawn_worker(&mut self) -> usize {
        let index = self.workers.len();
        let (tx, rx) = channel();
        if let Some((gen_, d)) = &self.design {
            let _ = tx.send(WorkerMsg::Design {
                gen_: *gen_,
                design: d.clone(),
            });
        }
        if let Some(c) = &self.config {
            let _ = tx.send(WorkerMsg::Config(c.clone()));
        }
        let worker = Worker {
            index,
            opts: self.opts.clone(),
            rx,
            tx: self.tx.clone(),
            quit: self.quit.clone(),
            kernel: None,
            gen_: 0,
            recorder: WaveRecorder::default(),
        };
        let etx = self.etx.clone();
        let thread = std::thread::Builder::new()
            .name(format!("panda-wave-replay-{index}"))
            .spawn(move || {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| worker.run()));
                if let Err(p) = r {
                    let msg = super::panic_message(p.as_ref());
                    let _ = etx.try_send(SimEvent::Error(format!(
                        "waveform replay thread panicked: {msg}"
                    )));
                }
            })
            .expect("spawn replay thread");
        self.workers.push(WorkerHandle {
            tx,
            busy: 0,
            thread: Some(thread),
        });
        index
    }

    /// Emit finished jobs in order.
    fn emit_ready(&mut self) {
        let stuck = 4 * BACKLOG_PER_WORKER * self.max_workers + 8;
        while let Some((&first, _)) = self.done.first_key_value() {
            if first != self.next_emit {
                // A dead worker never reports: do not wait forever behind it.
                if self.done.len() > stuck {
                    self.next_emit = first;
                    self.gap = true;
                    continue;
                }
                return;
            }
            let chunk = self.done.remove(&first).expect("present");
            self.next_emit += 1;
            match chunk {
                None => self.gap = true,
                Some(mut chunk) => {
                    let last = &mut self.last;
                    chunk.changes.retain(|c| {
                        let Some(l) = last.get_mut(c.signal as usize) else {
                            return false;
                        };
                        let keep = *l != Some(c.value);
                        *l = Some(c.value);
                        keep
                    });
                    chunk.gap |= std::mem::take(&mut self.gap);
                    if !chunk.changes.is_empty() || chunk.gap {
                        self.emit(chunk);
                    }
                }
            }
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

/// A replay thread: one kernel per design generation.
struct Worker {
    index: usize,
    opts: KernelOptions,
    rx: Receiver<WorkerMsg>,
    tx: Sender<ShadowMsg>,
    quit: Arc<AtomicBool>,
    kernel: Option<Kernel>,
    gen_: u64,
    recorder: WaveRecorder,
}

impl Worker {
    fn run(mut self) {
        // Compile synchronously: this thread may block, the main sim never waits for it.
        self.opts.background = false;
        while let Ok(msg) = self.rx.recv() {
            if self.quit.load(Ordering::Relaxed) {
                return;
            }
            match msg {
                WorkerMsg::Design { gen_, design } => {
                    self.gen_ = gen_;
                    self.recorder = WaveRecorder::default();
                    let mut k = Kernel::new(self.opts.clone());
                    if k.load((*design).clone()).is_ok() {
                        // Compile the Fast engine now (the usual mode), not at the first job.
                        k.set_mode(panda_kernel::Mode::Fast);
                        let _ = k.wait_ready();
                        self.kernel = Some(k);
                    } else {
                        self.kernel = None;
                    }
                }
                WorkerMsg::Config(cfg) => self.recorder = WaveRecorder::new((*cfg).clone()),
                WorkerMsg::Job(job) => {
                    let id = job.id;
                    let chunk = self.replay(&job);
                    drop(job); // return the state buffer before reporting
                    let done = ShadowMsg::Done {
                        worker: self.index,
                        job: id,
                        chunk,
                    };
                    if self.tx.send(done).is_err() {
                        return;
                    }
                }
            }
        }
    }

    fn replay(&mut self, job: &Job) -> Option<WaveChunk> {
        if job.gen_ != self.gen_
            || job.config != self.recorder.config().id
            || self.recorder.is_empty()
        {
            return None;
        }
        let k = self.kernel.as_mut()?;
        let start = job.start.get();
        k.set_mode(start.mode());
        let _ = k.wait_ready();
        k.restore(start).ok()?;
        // Discard anything left over, then record every signal at the start.
        let _ = self.recorder.take_chunk();
        let nets = self.recorder.nets().to_vec();
        let mut v = vec![Logic::X; nets.len()];
        k.copy_values(&nets, &mut v);
        self.recorder.prime(k.tick(), &v);
        for (t, pin, value) in &job.inputs {
            replay_to(k, &mut self.recorder, *t, &self.quit);
            let _ = k.set_input(pin, value);
        }
        replay_to(k, &mut self.recorder, job.end, &self.quit);
        let mut chunk = self.recorder.take_chunk()?;
        chunk.config = job.config;
        Some(chunk)
    }
}

/// Step `k` with sampling up to tick `to`, in slices (checking `quit` between them).
fn replay_to(k: &mut Kernel, rec: &mut WaveRecorder, to: u64, quit: &AtomicBool) {
    let from = k.tick();
    if to <= from {
        return;
    }
    let slice = ((to - from) / SLICES).max(1);
    while k.tick() < to && !quit.load(Ordering::Relaxed) {
        let before = k.tick();
        k.step_observed(slice.min(to - before), StepUnit::Tick, rec);
        if k.tick() == before {
            break;
        }
    }
}
