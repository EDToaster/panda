//! Command / snapshot semantics of the sim host.

use std::time::{Duration, Instant};

use panda_kernel::{EventObserver, Kernel, KernelOptions, Mode, StepUnit};
use panda_netlist::{Logic, Value};

use super::*;
use crate::examples;

/// Wait until `pred` holds for a fresh snapshot (or panic after `secs`).
fn wait_snapshot(h: &mut SimHost, secs: f64, mut pred: impl FnMut(&Snapshot) -> bool) -> Snapshot {
    let t0 = Instant::now();
    loop {
        let s = h.snapshot().clone();
        if pred(&s) {
            return s;
        }
        assert!(
            t0.elapsed().as_secs_f64() < secs,
            "timed out; last snapshot {s:?}"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// Wait until the Fast-mode engine serves (not the Timing fallback used while compiling: its
/// sub-phase behaviour differs, and a loaded machine compiles slowly).
fn wait_fast_engine(h: &mut SimHost) {
    wait_snapshot(h, 60.0, |s| {
        matches!(s.engine, Some(panda_kernel::EngineKind::Fast(_)))
    });
}

/// Wait until the JIT tier serves.
fn wait_jit(h: &mut SimHost) {
    wait_snapshot(h, 120.0, |s| s.jit);
}

fn netlist_event(h: &mut SimHost) -> (u64, Arc<FlatNetlist>) {
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_secs(10) {
        if let Some(SimEvent::Netlist { design_gen, flat }) =
            h.wait_event(Duration::from_millis(100))
        {
            return (design_gen, flat);
        }
    }
    panic!("no netlist event");
}

fn q_watch(flat: &FlatNetlist, gen_: u64) -> WatchSet {
    WatchSet {
        id: 42,
        design_gen: gen_,
        nets: (0..4)
            .map(|b| flat.net_by_name(&format!("Q[{b}]")).unwrap())
            .collect(),
    }
}

fn q_of(values: &[Logic]) -> Option<u64> {
    Value::from_bits(values).to_u64()
}

#[test]
fn load_watch_step_publishes_values() {
    let mut h = SimHost::spawn(SimOptions::default());
    assert_eq!(h.snapshot().seq, 0);
    h.send(Command::Load(Box::new(examples::counter())));
    let (g, flat) = netlist_event(&mut h);
    assert_eq!(g, 1);
    h.send(Command::SetWatch(q_watch(&flat, g)));
    h.send(Command::SetInput {
        pin: "EN".into(),
        value: Value::from_bool(true),
    });
    h.send(Command::Step {
        n: 5,
        unit: StepUnit::Cycle,
    });
    let s = wait_snapshot(&mut h, 5.0, |s| s.watch_id == 42 && s.tick > 0);
    assert_eq!(s.design_gen, 1);
    assert_eq!(s.values.len(), 4);
    assert!(!s.running);
    let q = q_of(&s.values).expect("binary Q");
    // Paused: values stay put.
    std::thread::sleep(Duration::from_millis(30));
    let s2 = h.snapshot().clone();
    assert_eq!(s2.tick, s.tick);
    assert_eq!(q_of(&s2.values), Some(q));
    // One more cycle increments the counter.
    h.send(Command::Step {
        n: 1,
        unit: StepUnit::Cycle,
    });
    let s3 = wait_snapshot(&mut h, 5.0, |x| x.tick > s.tick);
    assert_eq!(q_of(&s3.values), Some((q + 1) % 16));
}

#[test]
fn run_pause_and_rate_limit() {
    let mut h = SimHost::spawn(SimOptions::default());
    h.send(Command::Load(Box::new(examples::counter())));
    let _ = netlist_event(&mut h);
    h.send(Command::SetSpeed(Speed::CyclesPerSec(1000.0)));
    h.send(Command::Run);
    let s0 = wait_snapshot(&mut h, 5.0, |s| s.running && s.stats.cycles > 0);
    std::thread::sleep(Duration::from_millis(600));
    let s1 = h.snapshot().clone();
    let rate = (s1.stats.cycles - s0.stats.cycles) as f64 / 0.6;
    // Paced to 1000/s (lower bound generous: the test machine may be loaded).
    assert!((250.0..1600.0).contains(&rate), "rate {rate}");
    h.send(Command::Pause);
    let p = wait_snapshot(&mut h, 5.0, |s| !s.running);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(h.snapshot().tick, p.tick, "paused sim advanced");
    // Unlimited runs far faster.
    h.send(Command::SetSpeed(Speed::Unlimited));
    h.send(Command::Run);
    std::thread::sleep(Duration::from_millis(300));
    let s = h.snapshot().clone();
    assert!(s.stats.cycles - p.stats.cycles > 5_000, "{:?}", s.stats);
    // Publication rate cap: ~120 Hz → about 36 publications in 300 ms.
    let pubs = s.seq - p.seq;
    assert!(pubs <= 50, "{pubs} publications in 300 ms");
}

#[test]
fn watch_sets_of_other_generations_are_ignored() {
    let mut h = SimHost::spawn(SimOptions::default());
    h.send(Command::Load(Box::new(examples::counter())));
    let (g, flat) = netlist_event(&mut h);
    h.send(Command::SetWatch(WatchSet {
        design_gen: g + 5,
        ..q_watch(&flat, g)
    }));
    h.send(Command::Step {
        n: 1,
        unit: StepUnit::Cycle,
    });
    let s = wait_snapshot(&mut h, 5.0, |s| s.tick > 0);
    assert_eq!(s.watch_id, 0);
    assert!(s.values.is_empty());
    // Reset starts a new generation (old indices invalid).
    h.send(Command::Reset);
    let (g2, _) = netlist_event(&mut h);
    assert_eq!(g2, g + 1);
}

#[test]
fn edit_round_trip_and_custom_command() {
    let mut h = SimHost::spawn(SimOptions::default());
    let design = examples::counter();
    let top = design.top.clone();
    h.send(Command::Load(Box::new(design)));
    let (g, _) = netlist_event(&mut h);
    // An invalid edit is rejected and the generation stays.
    h.send(Command::Edit {
        id: 9,
        ops: vec![panda_kernel::EditOp::RemoveDef(top.clone())],
    });
    let t0 = Instant::now();
    let mut got = None;
    while got.is_none() && t0.elapsed() < Duration::from_secs(5) {
        if let Some(SimEvent::Edited { id, result }) = h.wait_event(Duration::from_millis(50)) {
            got = Some((id, result.is_err()));
        }
    }
    assert_eq!(got, Some((9, true)));
    h.send(Command::Custom(Box::new(|k| {
        Some(SimEvent::Custom(Box::new(k.tick() + 1000)))
    })));
    let t0 = Instant::now();
    loop {
        assert!(t0.elapsed() < Duration::from_secs(5));
        if let Some(SimEvent::Custom(b)) = h.wait_event(Duration::from_millis(50)) {
            assert!(*b.downcast::<u64>().unwrap() >= 1000);
            break;
        }
    }
    let s = wait_snapshot(&mut h, 5.0, |s| s.seq > 0 && s.design_gen >= g);
    assert_eq!(s.design_gen, g);
}

#[test]
fn waveform_chunks_follow_the_counter() {
    let mut h = SimHost::spawn(SimOptions::default());
    h.send(Command::Load(Box::new(examples::counter())));
    let (g, flat) = netlist_event(&mut h);
    h.send(Command::SetMode(Mode::Fast));
    h.send(Command::SetInput {
        pin: "EN".into(),
        value: Value::from_bool(true),
    });
    let q: Vec<NetIdx> = (0..4)
        .map(|b| flat.net_by_name(&format!("Q[{b}]")).unwrap())
        .collect();
    let cfg = WaveConfig {
        id: 3,
        design_gen: g,
        signals: vec![WaveSignal {
            name: "Q".into(),
            nets: q,
            clock: None,
        }],
    };
    let mut store = WaveStore::new(cfg.clone());
    h.send(Command::SetWave(cfg));
    h.send(Command::Step {
        n: 40,
        unit: StepUnit::Cycle,
    });
    let t0 = Instant::now();
    while store.history[0].len() < 20 && t0.elapsed() < Duration::from_secs(10) {
        if let Some(SimEvent::Wave(c)) = h.wait_event(Duration::from_millis(50)) {
            store.append(&c);
        }
    }
    let h0 = &store.history[0];
    assert!(h0.len() >= 20, "{} changes", h0.len());
    assert!(h0.iter().zip(h0.iter().skip(1)).all(|(a, b)| a.0 <= b.0));
    // After reset the counter counts up by one per change.
    let vals: Vec<u64> = h0
        .iter()
        .filter_map(|(_, v)| v.to_value(4).to_u64())
        .collect();
    let steps = vals.windows(2).filter(|w| w[1] == (w[0] + 1) % 16).count();
    assert!(steps >= 10, "{vals:?}");
}

#[test]
fn drop_stops_the_thread_even_when_events_are_not_drained() {
    let mut h = SimHost::spawn(SimOptions {
        event_capacity: 1,
        ..SimOptions::default()
    });
    for _ in 0..5 {
        h.send(Command::Load(Box::new(examples::counter())));
    }
    h.send(Command::Run);
    std::thread::sleep(Duration::from_millis(50));
    let t0 = Instant::now();
    drop(h);
    assert!(t0.elapsed() < Duration::from_secs(2));
}

/// Collect wave chunks into a store until none arrived for `quiet`.
fn drain_waves(h: &mut SimHost, store: &mut WaveStore, quiet: Duration) {
    let mut last = Instant::now();
    let t0 = Instant::now();
    while last.elapsed() < quiet && t0.elapsed() < Duration::from_secs(20) {
        if let Some(SimEvent::Wave(c)) = h.wait_event(Duration::from_millis(10)) {
            store.append(&c);
            last = Instant::now();
        }
    }
}

fn counter_history(capture: WaveCapture, mode: Mode) -> Vec<Vec<(u64, Packed)>> {
    let mut h = SimHost::spawn(SimOptions {
        wave_capture: capture,
        ..SimOptions::default()
    });
    h.send(Command::Load(Box::new(examples::counter())));
    let (g, flat) = netlist_event(&mut h);
    h.send(Command::SetMode(mode));
    if mode == Mode::Fast {
        wait_fast_engine(&mut h);
    }
    h.send(Command::Step {
        n: 1,
        unit: StepUnit::Cycle,
    });
    let q: Vec<NetIdx> = (0..4)
        .map(|b| flat.net_by_name(&format!("Q[{b}]")).unwrap())
        .collect();
    let en = vec![flat.net_by_name("EN").unwrap()];
    let cfg = WaveConfig {
        id: 5,
        design_gen: g,
        signals: vec![
            WaveSignal::new(&flat, "Q", q),
            WaveSignal::new(&flat, "EN", en),
        ],
    };
    let mut store = WaveStore::new(cfg.clone());
    h.send(Command::SetWave(cfg));
    for i in 0..12 {
        if i == 3 {
            h.send(Command::SetInput {
                pin: "EN".into(),
                value: Value::from_bool(true),
            });
        }
        if i == 9 {
            h.send(Command::SetInput {
                pin: "EN".into(),
                value: Value::from_bool(false),
            });
        }
        h.send(Command::Step {
            n: 3,
            unit: StepUnit::Cycle,
        });
        // Let some steps land in separate checkpoints.
        std::thread::sleep(Duration::from_millis(if i % 3 == 0 { 20 } else { 0 }));
    }
    drain_waves(&mut h, &mut store, Duration::from_millis(400));
    assert_eq!(store.gaps, 0);
    store
        .history
        .iter()
        .map(|q| q.iter().copied().collect())
        .collect()
}

#[test]
fn shadow_capture_equals_inline_capture() {
    for mode in [Mode::Timing, Mode::Fast] {
        let inline = counter_history(WaveCapture::Inline, mode);
        let shadow = counter_history(WaveCapture::Shadow, mode);
        assert!(
            inline[0].len() > 10,
            "{mode:?}: {} Q changes",
            inline[0].len()
        );
        assert_eq!(inline, shadow, "{mode:?}");
    }
}

#[test]
fn shadow_catches_up_after_a_flat_out_run() {
    let mut h = SimHost::spawn(SimOptions::default());
    h.send(Command::Load(Box::new(examples::panda8())));
    let (g, flat) = netlist_event(&mut h);
    h.send(Command::SetMode(Mode::Fast));
    wait_jit(&mut h);
    let pc: Vec<NetIdx> = (0..8)
        .map(|b| flat.net_by_name(&format!("PC[{b}]")).unwrap())
        .collect();
    let cfg = WaveConfig {
        id: 9,
        design_gen: g,
        signals: vec![WaveSignal::new(&flat, "PC", pc.clone())],
    };
    let mut store = WaveStore::new(cfg.clone());
    h.send(Command::SetWave(cfg));
    h.send(Command::SetWatch(WatchSet {
        id: 1,
        design_gen: g,
        nets: pc,
    }));
    h.send(Command::Run);
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_millis(500) {
        if let Some(SimEvent::Wave(c)) = h.wait_event(Duration::from_millis(5)) {
            store.append(&c);
        }
    }
    h.send(Command::Pause);
    let s = wait_snapshot(&mut h, 5.0, |s| !s.running && s.watch_id == 1);
    drain_waves(&mut h, &mut store, Duration::from_millis(1500));
    assert!(s.stats.cycles > 50_000, "{:?}", s.stats);
    // The history reaches the paused tick and agrees with the live value there.
    let last = store.history[0].back().copied().expect("changes");
    assert!(last.0 <= s.tick);
    assert_eq!(
        store.value_at(0, s.tick).unwrap().to_value(8),
        Value::from_bits(&s.values)
    );
    // PC changes every other cycle in the bench loop: most of the run was captured.
    eprintln!(
        "{} PC changes for {} cycles, {} gap chunks",
        store.history[0].len(),
        s.stats.cycles,
        store.gaps
    );
    assert!(store.history[0].len() as u64 > s.stats.cycles / 8);
}

#[test]
fn sim_thread_panic_is_reported_and_drop_does_not_hang() {
    let mut h = SimHost::spawn(SimOptions::default());
    h.send(Command::Load(Box::new(examples::counter())));
    let _ = netlist_event(&mut h);
    assert!(h.crashed().is_none());
    h.send(Command::Crash);
    let t0 = Instant::now();
    let mut reported = false;
    while !reported && t0.elapsed() < Duration::from_secs(5) {
        reported = h
            .poll_events()
            .iter()
            .any(|e| matches!(e, SimEvent::Error(m) if m.contains("simulation thread panicked")));
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(reported, "crash not reported");
    assert_eq!(h.crashed(), Some("requested crash"));
    assert!(!h.is_alive());
    // Reported once; further sends are harmless.
    assert!(h.poll_events().is_empty());
    h.send(Command::Run);
    let t0 = Instant::now();
    drop(h);
    assert!(t0.elapsed() < Duration::from_secs(1));
}

#[test]
fn panicking_custom_command_does_not_stop_the_simulation() {
    let mut h = SimHost::spawn(SimOptions::default());
    h.send(Command::Load(Box::new(examples::counter())));
    let _ = netlist_event(&mut h);
    h.send(Command::Custom(Box::new(|_| panic!("panel bug"))));
    let t0 = Instant::now();
    let mut got = false;
    while !got && t0.elapsed() < Duration::from_secs(5) {
        if let Some(SimEvent::Error(m)) = h.wait_event(Duration::from_millis(20)) {
            got = m.contains("panel bug");
        }
    }
    assert!(got);
    h.send(Command::Step {
        n: 3,
        unit: StepUnit::Cycle,
    });
    let s = wait_snapshot(&mut h, 5.0, |s| s.tick > 0);
    assert!(h.crashed().is_none() && h.is_alive(), "{s:?}");
}

#[test]
fn long_step_is_interruptible_and_keeps_publishing() {
    let mut h = SimHost::spawn(SimOptions::default());
    h.send(Command::Load(Box::new(examples::counter())));
    let (g, flat) = netlist_event(&mut h);
    h.send(Command::SetWatch(q_watch(&flat, g)));
    h.send(Command::Step {
        n: u64::MAX / 4,
        unit: StepUnit::Cycle,
    });
    // An input change queued behind the step waits for it (order is kept).
    h.send(Command::SetInput {
        pin: "EN".into(),
        value: Value::from_bool(true),
    });
    let a = wait_snapshot(&mut h, 5.0, |s| s.tick > 1000);
    let b = wait_snapshot(&mut h, 5.0, |s| s.seq > a.seq + 3 && s.tick > a.tick);
    assert!(!b.running);
    // The view still updates mid-step.
    h.send(Command::SetWatch(WatchSet {
        id: 77,
        ..q_watch(&flat, g)
    }));
    let _ = wait_snapshot(&mut h, 5.0, |s| s.watch_id == 77);
    // EN is still 0 (deferred), so Q holds.
    let q = q_of(&h.snapshot().values.clone());
    h.send(Command::Pause);
    let p = wait_snapshot(&mut h, 5.0, |s| s.seq > b.seq + 1);
    std::thread::sleep(Duration::from_millis(50));
    let p2 = wait_snapshot(&mut h, 5.0, |s| s.seq > p.seq);
    assert_eq!(p2.tick, h.snapshot().tick);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(h.snapshot().tick, p2.tick, "paused: the step was aborted");
    // The deferred input change was applied after the abort: one cycle now counts.
    h.send(Command::Step {
        n: 2,
        unit: StepUnit::Cycle,
    });
    let s = wait_snapshot(&mut h, 5.0, |s| s.tick > p2.tick);
    assert_ne!(q_of(&s.values), q, "EN applied after the aborted step");
    let t0 = Instant::now();
    h.send(Command::Step {
        n: u64::MAX / 4,
        unit: StepUnit::Tick,
    });
    std::thread::sleep(Duration::from_millis(50));
    drop(h);
    assert!(
        t0.elapsed() < Duration::from_secs(2),
        "drop waited for the step"
    );
}

#[test]
fn chunked_steps_match_direct_kernel_steps() {
    // Same command sequence through the host and directly on a kernel.
    let mut h = SimHost::spawn(SimOptions {
        chunk: Duration::from_micros(50),
        ..SimOptions::default()
    });
    h.send(Command::Load(Box::new(examples::counter())));
    let (g, flat) = netlist_event(&mut h);
    h.send(Command::SetWatch(q_watch(&flat, g)));
    let mut k = Kernel::new(KernelOptions::default());
    k.load(examples::counter()).unwrap();
    let script: [(u64, bool); 5] = [(37, true), (5, false), (101, true), (3, true), (64, false)];
    for (n, en) in script {
        h.send(Command::SetInput {
            pin: "EN".into(),
            value: Value::from_bool(en),
        });
        h.send(Command::Step {
            n,
            unit: StepUnit::Tick,
        });
        k.set_input("EN", &Value::from_bool(en)).unwrap();
        k.step(n, StepUnit::Tick);
    }
    let want_tick = k.tick();
    let s = wait_snapshot(&mut h, 5.0, |s| s.tick == want_tick && s.watch_id == 42);
    let nets = q_watch(&flat, g).nets;
    let mut v = vec![Logic::X; nets.len()];
    k.copy_values(&nets, &mut v);
    assert_eq!(s.values, v);
}

/// Shadow capture during a flat-out Panda-8 run reproduces exactly what an inline recorder on a
/// reference kernel records from the same start state (every recorded change is a real one; with
/// no gaps the histories are identical).
#[test]
fn parallel_shadow_replay_is_exact() {
    let mut h = SimHost::spawn(SimOptions {
        shadow_workers: 3,
        ..SimOptions::default()
    });
    let design = examples::panda8();
    h.send(Command::Load(Box::new(design.clone())));
    let (g, flat) = netlist_event(&mut h);
    h.send(Command::SetMode(Mode::Fast));
    wait_jit(&mut h);
    let names = [
        "PC",
        "IR",
        "OUT",
        "R0",
        "FLAGS",
        "PHASE",
        "OUT_STROBE",
        "clk",
    ];
    let signals: Vec<WaveSignal> = names
        .iter()
        .map(|n| {
            let nets = crate::app::resolve_net(&flat, n).unwrap();
            WaveSignal::new(&flat, *n, nets)
        })
        .collect();
    let cfg = WaveConfig {
        id: 11,
        design_gen: g,
        signals,
    };
    let mut store = WaveStore::new(cfg.clone());
    h.send(Command::SetWave(cfg.clone()));
    h.send(Command::Custom(Box::new(|k| {
        Some(SimEvent::Custom(Box::new(k.snapshot())))
    })));
    h.send(Command::Run);
    let mut start: Option<panda_kernel::State> = None;
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_millis(700) {
        match h.wait_event(Duration::from_millis(5)) {
            Some(SimEvent::Wave(c)) => store.append(&c),
            Some(SimEvent::Custom(b)) => start = Some(*b.downcast().unwrap()),
            _ => {}
        }
    }
    h.send(Command::Pause);
    let s = wait_snapshot(&mut h, 5.0, |s| !s.running);
    // Drain until the capture is quiet (replay threads compile the design first).
    let mut last = Instant::now();
    let t1 = Instant::now();
    while last.elapsed() < Duration::from_millis(1500) && t1.elapsed() < Duration::from_secs(60) {
        match h.wait_event(Duration::from_millis(10)) {
            Some(SimEvent::Wave(c)) => {
                store.append(&c);
                last = Instant::now();
            }
            Some(SimEvent::Custom(b)) => start = Some(*b.downcast().unwrap()),
            _ => {}
        }
    }
    let start = start.expect("start state");
    // Reference: the same interval recorded inline.
    let mut k = Kernel::new(KernelOptions {
        background: false,
        ..KernelOptions::default()
    });
    k.set_mode(Mode::Fast);
    k.load(design).unwrap();
    k.wait_ready().unwrap();
    k.restore(&start).unwrap();
    let mut rec = WaveRecorder::new(cfg.clone());
    let nets = rec.nets().to_vec();
    let mut v = vec![Logic::X; nets.len()];
    k.copy_values(&nets, &mut v);
    rec.prime(k.tick(), &v);
    // Record the reference in slices: one recorder chunk holds at most
    // `WaveRecorder::DEFAULT_CAP` changes, and a truncated reference would make `value_at`
    // report stale values for every later tick.
    let mut reference = WaveStore::new(cfg);
    reference.cap = usize::MAX;
    while k.tick() < s.tick {
        let n = (s.tick - k.tick()).min(1 << 16);
        k.step_observed(n, StepUnit::Tick, &mut rec);
        if let Some(c) = rec.take_chunk() {
            assert!(!c.gap, "reference recording lost changes");
            reference.append(&c);
        }
    }
    assert_eq!(k.tick(), s.tick);
    let total: usize = store.history.iter().map(|q| q.len()).sum();
    eprintln!(
        "{} cycles, {} recorded changes ({} in the reference), {} gap chunks",
        s.stats.cycles,
        total,
        reference.history.iter().map(|q| q.len()).sum::<usize>(),
        store.gaps
    );
    assert!(total > 10_000, "{total} changes");
    for (sig, hist) in store.history.iter().enumerate() {
        // No repeated values.
        assert!(
            hist.iter()
                .zip(hist.iter().skip(1))
                .all(|(a, b)| a.0 < b.0 && a.1 != b.1)
        );
        // Every recorded change is a real one.
        for &(t, val) in hist {
            assert_eq!(reference.value_at(sig, t), Some(val), "signal {sig} at {t}");
        }
    }
    if store.gaps == 0 {
        assert_eq!(store.history, reference.history);
    }
}
