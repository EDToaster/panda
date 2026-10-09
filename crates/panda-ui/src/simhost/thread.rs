//! The sim thread's loop.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError};
use std::time::{Duration, Instant};

use panda_kernel::{Kernel, StepUnit};
use panda_netlist::{Logic, Severity};

use super::shadow::{Checkpoint, Shadow, ShadowMsg};
use super::triple::Writer;
use super::{
    Command, SimEvent, SimOptions, Snapshot, Speed, WatchSet, WaveCapture, WaveConfig, WaveRecorder,
};
use panda_netlist::Value;

/// Max events kept in the outbox while the UI does not drain the channel; beyond it, waveform
/// chunks are dropped (and flagged as gaps).
const OUTBOX_WAVE_LIMIT: usize = 64;
/// Minimum interval between diagnostics lists.
const DIAG_INTERVAL: Duration = Duration::from_millis(100);
/// Window of the rate measurement.
const RATE_WINDOW: Duration = Duration::from_millis(500);
/// Publication interval while paused and nothing changed.
const PAUSED_PUBLISH: Duration = Duration::from_millis(100);
/// Sleep after an idle run (nothing can change) while "running".
const IDLE_SLEEP: Duration = Duration::from_millis(5);

pub(super) struct SimThread {
    opts: SimOptions,
    kernel: Kernel,
    rx: Receiver<Command>,
    pending: Arc<AtomicBool>,
    out: Writer<Snapshot>,
    etx: SyncSender<SimEvent>,
    outbox: VecDeque<SimEvent>,
    running: bool,
    idle: bool,
    speed: Speed,
    design_gen: u64,
    watch: WatchSet,
    wave: WaveRecorder,
    wave_gap: bool,
    // Shadow capture (see `shadow`).
    shadow: Option<Shadow>,
    shadow_gen: u64,
    shadow_active: bool,
    cp_seq: u64,
    cp_dirty: bool,
    input_log: Vec<(u64, String, Value)>,
    seq: u64,
    publish_every: Duration,
    next_publish: Instant,
    last_publish: Instant,
    dirty: bool,
    // Diagnostics change detection.
    diag_len: usize,
    diag_total: u64,
    last_diag: Instant,
    // Statistics.
    cycles: u64,
    win_start: Instant,
    win_cycles: u64,
    win_ticks: u64,
    win_busy: Duration,
    cps: f64,
    tps: f64,
    busy: f64,
    publish_total: Duration,
    // Speed limiting.
    pace_start: Instant,
    pace_cycles: u64,
}

impl SimThread {
    pub(super) fn new(
        opts: SimOptions,
        rx: Receiver<Command>,
        pending: Arc<AtomicBool>,
        out: Writer<Snapshot>,
        etx: SyncSender<SimEvent>,
    ) -> SimThread {
        let now = Instant::now();
        let publish_every = Duration::from_secs_f64(1.0 / opts.publish_hz.max(1.0));
        SimThread {
            kernel: Kernel::new(opts.kernel.clone()),
            opts,
            rx,
            pending,
            out,
            etx,
            outbox: VecDeque::new(),
            running: false,
            idle: false,
            speed: Speed::Unlimited,
            design_gen: 0,
            watch: WatchSet::default(),
            wave: WaveRecorder::default(),
            wave_gap: false,
            shadow: None,
            shadow_gen: 0,
            shadow_active: false,
            cp_seq: 0,
            cp_dirty: false,
            input_log: Vec::new(),
            seq: 0,
            publish_every,
            next_publish: now,
            last_publish: now,
            dirty: true,
            diag_len: 0,
            diag_total: 0,
            last_diag: now,
            cycles: 0,
            win_start: now,
            win_cycles: 0,
            win_ticks: 0,
            win_busy: Duration::ZERO,
            cps: 0.0,
            tps: 0.0,
            busy: 0.0,
            publish_total: Duration::ZERO,
            pace_start: now,
            pace_cycles: 0,
        }
    }

    pub(super) fn run(mut self) {
        loop {
            // Commands: one atomic load per chunk while running; block while paused.
            if self.running && !self.idle {
                if self.pending.load(Ordering::Acquire) {
                    self.pending.store(false, Ordering::Release);
                    if !self.drain() {
                        return;
                    }
                }
            } else {
                self.pending.store(false, Ordering::Release);
                let wait = if self.running {
                    IDLE_SLEEP
                } else {
                    self.next_publish
                        .saturating_duration_since(Instant::now())
                        .max(Duration::from_millis(1))
                        .min(Duration::from_millis(50))
                };
                match self.rx.recv_timeout(wait) {
                    Ok(cmd) => {
                        if !self.handle(cmd) || !self.drain() {
                            return;
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => return,
                }
                if self.running && self.idle {
                    // Re-check whether something can change (e.g. after an input change).
                    self.idle = false;
                }
            }
            if self.running && !self.idle && self.kernel.design().is_some() {
                self.run_chunk();
            }
            let now = Instant::now();
            // While paused, still publish at a low rate: the kernel's background compilation
            // advances tiers (and reports them) only when asked.
            let stale = now.duration_since(self.last_publish) >= PAUSED_PUBLISH;
            if now >= self.next_publish && (self.dirty || self.running || stale) {
                self.publish(now);
                self.last_publish = now;
                self.next_publish = now + self.publish_every;
            }
            self.flush_outbox();
        }
    }

    /// Drain every queued command. False on `Quit` / disconnect.
    fn drain(&mut self) -> bool {
        loop {
            match self.rx.try_recv() {
                Ok(cmd) => {
                    if !self.handle(cmd) {
                        return false;
                    }
                }
                Err(TryRecvError::Empty) => return true,
                Err(TryRecvError::Disconnected) => return false,
            }
        }
    }

    fn handle(&mut self, cmd: Command) -> bool {
        self.dirty = true;
        match cmd {
            Command::Quit => return false,
            Command::Load(design) => match self.kernel.load(*design) {
                Ok(()) => self.new_generation(),
                Err(d) => self.emit(SimEvent::LoadFailed(d)),
            },
            Command::Edit { id, ops } => {
                let result = self.kernel.edit(&ops);
                let ok = result.is_ok();
                self.emit(SimEvent::Edited { id, result });
                if ok {
                    self.new_generation_keep_stats();
                }
            }
            Command::SetInput { pin, value } => {
                let tick = self.kernel.tick();
                match self.kernel.set_input(&pin, &value) {
                    Err(e) => self.emit(SimEvent::Error(e)),
                    Ok(()) if self.shadow_active => {
                        self.input_log.push((tick, pin, value));
                        self.cp_dirty = true;
                    }
                    Ok(()) => {}
                }
                self.idle = false;
            }
            Command::SetMode(m) => {
                let pre = self.kernel.tick();
                self.kernel.set_mode(m);
                self.idle = false;
                self.checkpoint(true, pre);
            }
            Command::Run => {
                self.running = true;
                self.idle = false;
                self.reset_pace();
            }
            Command::Pause => self.running = false,
            Command::Step { n, unit } => {
                let t0 = Instant::now();
                if self.wave.is_empty() {
                    self.kernel.step(n, unit);
                } else {
                    self.kernel.step_observed(n, unit, &mut self.wave);
                }
                if unit == StepUnit::Cycle {
                    self.cycles += n;
                }
                self.win_busy += t0.elapsed();
                self.idle = false;
                self.cp_dirty = true;
            }
            Command::SetSpeed(s) => {
                self.speed = s;
                self.reset_pace();
            }
            Command::Reset => {
                if let Some(d) = self.kernel.design().cloned() {
                    match self.kernel.load(d) {
                        Ok(()) => self.new_generation(),
                        Err(d) => self.emit(SimEvent::LoadFailed(d)),
                    }
                }
            }
            Command::SetWatch(w) => {
                if w.design_gen == self.design_gen {
                    self.watch = w;
                }
            }
            Command::SetWave(cfg) => self.set_wave(cfg),
            Command::SetPublishRate(hz) => {
                self.publish_every = Duration::from_secs_f64(1.0 / hz.max(1.0));
            }
            Command::SetChunk(d) => self.opts.chunk = d.max(Duration::from_micros(50)),
            Command::Custom(f) => {
                let pre = self.kernel.tick();
                if let Some(ev) = f(&mut self.kernel) {
                    self.emit(ev);
                }
                self.idle = false;
                // The closure may have changed anything: the shadow reloads.
                self.checkpoint(true, pre);
            }
        }
        true
    }

    fn set_wave(&mut self, cfg: WaveConfig) {
        if cfg.design_gen != self.design_gen && !cfg.signals.is_empty() {
            return;
        }
        if let Some(c) = self.wave.take_chunk() {
            self.emit_wave(c);
        }
        if self.opts.wave_capture == WaveCapture::Shadow {
            self.wave = WaveRecorder::default();
            self.set_shadow_wave(cfg);
            return;
        }
        self.wave = WaveRecorder::new(cfg);
        if !self.wave.is_empty() {
            // Initial values.
            let nets = panda_kernel::EventObserver::nets(&self.wave).to_vec();
            let mut v = vec![Logic::X; nets.len()];
            self.kernel.copy_values(&nets, &mut v);
            let tick = self.kernel.tick();
            self.wave.prime(tick, &v);
        }
    }

    /// Shadow capture: hand the config to the shadow thread and give it a starting point.
    fn set_shadow_wave(&mut self, cfg: WaveConfig) {
        let active = !WaveRecorder::new(cfg.clone()).is_empty();
        if self.shadow.is_none() {
            if !active {
                return;
            }
            self.shadow = Some(Shadow::spawn(self.opts.kernel.clone(), self.etx.clone()));
        }
        let sh = self.shadow.as_ref().expect("spawned");
        if self.shadow_gen != self.design_gen
            && let Some(d) = self.kernel.design().cloned()
        {
            sh.send_blocking(ShadowMsg::Design {
                gen_: self.design_gen,
                design: Box::new(d),
            });
            self.shadow_gen = self.design_gen;
        }
        sh.send_blocking(ShadowMsg::Config(cfg));
        self.shadow_active = active;
        self.input_log.clear();
        let tick = self.kernel.tick();
        self.checkpoint(true, tick);
    }

    /// Send the shadow a checkpoint: the interval since the previous one ends at `replay_to`;
    /// `reload`: the current state is not that interval's end (a command changed it).
    fn checkpoint(&mut self, reload: bool, replay_to: u64) {
        if !self.shadow_active || self.kernel.design().is_none() {
            return;
        }
        let Some(sh) = &self.shadow else { return };
        self.cp_seq += 1;
        let c = Checkpoint {
            gen_: self.design_gen,
            seq: self.cp_seq,
            inputs: std::mem::take(&mut self.input_log),
            replay_to,
            state: self.kernel.snapshot(),
            reload,
        };
        // Dropped when the shadow's queue is full: it notices the missing number and reloads.
        sh.send(ShadowMsg::Checkpoint(Box::new(c)));
        self.cp_dirty = false;
    }

    fn new_generation(&mut self) {
        self.cycles = 0;
        self.new_generation_keep_stats();
    }

    fn new_generation_keep_stats(&mut self) {
        self.design_gen += 1;
        self.idle = false;
        // Net indices changed: the old watch / wave sets are meaningless.
        self.watch = WatchSet {
            id: 0,
            design_gen: self.design_gen,
            nets: Vec::new(),
        };
        if !self.wave.is_empty() {
            if let Some(c) = self.wave.take_chunk() {
                self.emit_wave(c);
            }
            self.wave = WaveRecorder::default();
        }
        if let Some(sh) = &self.shadow {
            if let Some(d) = self.kernel.design().cloned() {
                sh.send_blocking(ShadowMsg::Design {
                    gen_: self.design_gen,
                    design: Box::new(d),
                });
                self.shadow_gen = self.design_gen;
            }
            sh.send_blocking(ShadowMsg::Config(WaveConfig::default()));
        }
        self.shadow_active = false;
        self.input_log.clear();
        if let Some(flat) = self.kernel.netlist().cloned() {
            self.emit(SimEvent::Netlist {
                design_gen: self.design_gen,
                flat,
            });
        }
        self.reset_pace();
    }

    fn reset_pace(&mut self) {
        self.pace_start = Instant::now();
        self.pace_cycles = 0;
    }

    fn run_chunk(&mut self) {
        let t0 = Instant::now();
        let tick0 = self.kernel.tick();
        let (cycles, idle) = match self.speed {
            Speed::Unlimited => {
                let s = if self.wave.is_empty() {
                    self.kernel.run(self.opts.chunk)
                } else {
                    self.kernel.run_observed(self.opts.chunk, &mut self.wave)
                };
                (s.cycles.max(s.steps), s.idle)
            }
            Speed::CyclesPerSec(rate) => {
                let due = (self.pace_start.elapsed().as_secs_f64() * rate) as u64;
                let n = due.saturating_sub(self.pace_cycles);
                if n == 0 {
                    // Ahead of schedule: sleep until the next cycle is due (bounded).
                    let wait = Duration::from_secs_f64((1.0 / rate.max(1e-3)).min(0.01));
                    std::thread::sleep(
                        wait.min(self.next_publish.saturating_duration_since(Instant::now())),
                    );
                    (0, false)
                } else {
                    // Cap a burst at the chunk length's worth at the measured rate.
                    let cap = if self.cps > 0.0 {
                        ((self.cps * self.opts.chunk.as_secs_f64()) as u64).max(1)
                    } else {
                        1
                    };
                    let n = n.min(cap);
                    let before = self.kernel.tick();
                    if self.wave.is_empty() {
                        self.kernel.step(n, StepUnit::Cycle);
                    } else {
                        self.kernel
                            .step_observed(n, StepUnit::Cycle, &mut self.wave);
                    }
                    self.pace_cycles += n;
                    (n, self.kernel.tick() == before)
                }
            }
        };
        let dt = t0.elapsed();
        self.cycles += cycles;
        self.win_cycles += cycles;
        self.win_ticks += self.kernel.tick().saturating_sub(tick0);
        self.win_busy += dt;
        self.idle = idle;
        self.dirty = true;
        self.cp_dirty |= cycles > 0;
    }

    fn publish(&mut self, now: Instant) {
        let t0 = Instant::now();
        // Rates over the window.
        let w = now.duration_since(self.win_start);
        if w >= RATE_WINDOW {
            let s = w.as_secs_f64();
            self.cps = self.win_cycles as f64 / s;
            self.tps = self.win_ticks as f64 / s;
            self.busy = self.win_busy.as_secs_f64() / s;
            self.win_start = now;
            self.win_cycles = 0;
            self.win_ticks = 0;
            self.win_busy = Duration::ZERO;
        }
        self.seq += 1;
        let k = &mut self.kernel;
        let has_design = k.design().is_some();
        let jit = has_design && k.jit_active();
        let engine = has_design.then(|| k.engine());
        let slot = self.out.slot();
        slot.seq = self.seq;
        slot.design_gen = self.design_gen;
        slot.tick = k.tick();
        slot.mode = k.mode();
        slot.engine = engine;
        slot.jit = jit;
        slot.running = self.running;
        slot.idle = self.idle;
        slot.speed = self.speed;
        slot.watch_id = self.watch.id;
        slot.values.clear();
        slot.values.resize(self.watch.nets.len(), Logic::X);
        if has_design && !self.watch.nets.is_empty() {
            k.copy_values(&self.watch.nets, &mut slot.values);
        }
        let entries = k.diagnostic_entries();
        slot.diag_count = entries.len();
        slot.diag_errors = entries
            .iter()
            .filter(|e| e.diagnostic.severity == Severity::Error)
            .count();
        slot.diag_warnings = entries
            .iter()
            .filter(|e| e.diagnostic.severity == Severity::Warning)
            .count();
        let total: u64 = entries.iter().map(|e| e.count).sum();
        let diag_changed = entries.len() != self.diag_len || total != self.diag_total;
        slot.stats.cycles = self.cycles;
        slot.stats.cycles_per_sec = self.cps;
        slot.stats.ticks_per_sec = self.tps;
        slot.stats.busy = self.busy;
        slot.stats.publications = self.seq;
        slot.stats.publish_us = if self.seq > 1 {
            self.publish_total.as_secs_f64() * 1e6 / (self.seq - 1) as f64
        } else {
            0.0
        };
        self.out.publish();
        if diag_changed && now.duration_since(self.last_diag) >= DIAG_INTERVAL {
            let list = Arc::new(self.kernel.diagnostic_entries().to_vec());
            self.diag_len = list.len();
            self.diag_total = total;
            self.last_diag = now;
            self.emit(SimEvent::Diagnostics(list));
        }
        if let Some(c) = self.wave.take_chunk() {
            self.emit_wave(c);
        }
        if self.cp_dirty {
            let tick = self.kernel.tick();
            self.checkpoint(false, tick);
        }
        self.dirty = false;
        self.publish_total += t0.elapsed();
    }

    fn emit(&mut self, ev: SimEvent) {
        self.outbox.push_back(ev);
        self.flush_outbox();
    }

    fn emit_wave(&mut self, mut c: super::WaveChunk) {
        let waves = self
            .outbox
            .iter()
            .filter(|e| matches!(e, SimEvent::Wave(_)))
            .count();
        if waves >= OUTBOX_WAVE_LIMIT {
            // The UI is not draining: drop this chunk, flag the next one.
            self.wave_gap = true;
            return;
        }
        c.gap |= std::mem::take(&mut self.wave_gap);
        self.emit(SimEvent::Wave(c));
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
