//! Command / snapshot semantics of the sim host.

use std::time::{Duration, Instant};

use panda_kernel::{Mode, StepUnit};
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
    drain_waves(&mut h, &mut store, Duration::from_millis(500));
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
