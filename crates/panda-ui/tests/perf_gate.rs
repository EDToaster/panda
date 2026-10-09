//! The M6 perf gate (docs/M6-UI.md): with the editor open, Panda-8 (SumOfProducts,
//! Hierarchical) running in Fast mode (JIT), the canvas showing live values at 60 fps and 8
//! waveform signals watched, simulation throughput must be ≥ 95 % of headless `Kernel::run`.
//!
//! ```text
//! cargo test --release -p panda-ui --test perf_gate -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Rounds alternate headless and full-stack measurements (best of each is compared, medians are
//! reported too). The full stack is the real app: `SimHost` sim thread + `PandaApp` rendered
//! offscreen by egui_kittest/wgpu at 60 Hz (step + GPU render + readback every frame).
//! Environment: `PERF_SECS` (seconds per measurement, default 3), `PERF_ROUNDS` (default 3).

use std::time::{Duration, Instant};

use egui_kittest::Harness;
use panda_kernel::{EventObserver, Kernel, KernelOptions, Mode};
use panda_netlist::Logic;
use panda_netlist::flat::NetIdx;
use panda_ui::app::{AppState, PandaApp, resolve_net};
use panda_ui::examples;
use panda_ui::simhost::{Command, SimOptions};

/// The gate's "8 nets in the waveform viewer" (docs/M6-UI.md): eight 1-bit nets.
const NETS8: [&str; 8] = [
    "clk",
    "OUT_STROBE",
    "PHASE",
    "HALTED",
    "FLAGS[0]",
    "FLAGS[1]",
    "PC[0]",
    "IR[0]",
];
/// Eight signals including 8-bit buses (37 nets): reported, not gated.
const BUSES8: [&str; 8] = [
    "clk",
    "PC",
    "IR",
    "OUT",
    "OUT_STROBE",
    "PHASE",
    "FLAGS",
    "R0",
];

fn env_or(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn fast_kernel() -> Kernel {
    let mut k = Kernel::new(KernelOptions::default());
    k.set_mode(Mode::Fast);
    k.load(examples::panda8()).unwrap();
    k.wait_ready().unwrap();
    assert!(k.wait_jit(Duration::from_secs(10)), "JIT did not activate");
    k.run(Duration::from_millis(300)); // warm up
    k
}

/// Headless `Kernel::run(secs)`: cycles per second.
fn headless(secs: f64) -> f64 {
    let mut k = fast_kernel();
    let s = k.run(Duration::from_secs_f64(secs));
    s.cycles as f64 / s.elapsed.as_secs_f64()
}

/// Headless `Kernel::run` in 1.5 ms chunks (what the sim thread does), no UI.
fn headless_chunked(secs: f64) -> f64 {
    let mut k = fast_kernel();
    let t0 = Instant::now();
    let mut cycles = 0;
    while t0.elapsed().as_secs_f64() < secs {
        cycles += k.run(Duration::from_micros(1500)).cycles;
    }
    cycles as f64 / t0.elapsed().as_secs_f64()
}

/// Minimal observer gathering the 8 signals' nets per event (kernel-level observer cost).
struct Sink {
    nets: Vec<NetIdx>,
    n: u64,
    last: Vec<Logic>,
}

impl EventObserver for Sink {
    fn nets(&self) -> &[NetIdx] {
        &self.nets
    }
    fn observe(&mut self, _tick: u64, values: &[Logic]) {
        self.n += 1;
        if values != self.last.as_slice() {
            self.last.copy_from_slice(values);
        }
    }
}

fn headless_observed(secs: f64, signals: &[&str]) -> (f64, usize) {
    let mut k = fast_kernel();
    let flat = k.netlist().unwrap().clone();
    let nets: Vec<NetIdx> = signals
        .iter()
        .flat_map(|s| resolve_net(&flat, s).expect(s))
        .collect();
    let n = nets.len();
    let mut sink = Sink {
        last: vec![Logic::X; n],
        nets,
        n: 0,
    };
    let s = k.run_observed(Duration::from_secs_f64(secs), &mut sink);
    assert!(sink.n >= s.cycles);
    (s.cycles as f64 / s.elapsed.as_secs_f64(), n)
}

/// Full stack result.
struct Full {
    rate: f64,
    frames: usize,
    frame_ms: Vec<f64>,
    ui_ms: Vec<f64>,
    wave_changes: usize,
    /// Wave chunks that reported skipped intervals (shadow capture fell behind).
    wave_gaps: u64,
    busy: f64,
    publish_us: f64,
}

fn full_stack(secs: f64, signals: &[&str]) -> Full {
    let mut state = AppState::new(examples::panda8(), SimOptions::default());
    state.wait_layout();
    let app = PandaApp::new(state);
    let mut h: Harness<'static, PandaApp> = Harness::builder()
        .with_size(egui::vec2(1600.0, 1000.0))
        .wgpu()
        .build_ui_state(|ui, app: &mut PandaApp| app.show(ui), app);
    let wait =
        |h: &mut Harness<'static, PandaApp>, what: &str, pred: &dyn Fn(&PandaApp) -> bool| {
            let t0 = Instant::now();
            while !pred(h.state()) {
                assert!(
                    t0.elapsed() < Duration::from_secs(20),
                    "timed out waiting for {what}"
                );
                h.step();
                std::thread::sleep(Duration::from_millis(5));
            }
        };
    wait(&mut h, "live values", &|a| a.state.live.values().is_some());
    h.state_mut().state.sim.send(Command::SetMode(Mode::Fast));
    wait(&mut h, "JIT", &|a| {
        a.state.live.snapshot.tier_label() == "JIT"
    });
    let n = h.state_mut().state.set_wave_signals(signals);
    assert_eq!(n, signals.len(), "waveform signals resolved");
    h.state_mut().state.sim.send(Command::Run);
    // Warm up: running, top def visible with live values.
    wait(&mut h, "running", &|a| {
        a.state.live.snapshot.running && a.state.live.snapshot.stats.cycles > 10_000
    });
    let t_end = Instant::now() + Duration::from_millis(300);
    while Instant::now() < t_end {
        h.step();
        h.render().unwrap();
    }
    assert!(
        h.state().state.live.values().is_some(),
        "canvas has live values"
    );
    assert_eq!(h.state().state.view.path, "", "top def visible");

    let c0 = h.state().state.live.snapshot.stats.cycles;
    let w0: usize = h.state().state.wave.history.iter().map(|q| q.len()).sum();
    let t0 = Instant::now();
    let period = Duration::from_secs_f64(1.0 / 60.0);
    let mut next = t0;
    let mut frame_ms = Vec::new();
    let mut ui_ms = Vec::new();
    while t0.elapsed().as_secs_f64() < secs {
        let f0 = Instant::now();
        h.step();
        h.render().unwrap();
        frame_ms.push(f0.elapsed().as_secs_f64() * 1e3);
        ui_ms.push(h.state().state.frame.last_ms);
        next += period;
        if let Some(d) = next.checked_duration_since(Instant::now()) {
            std::thread::sleep(d);
        } else {
            next = Instant::now();
        }
    }
    // Let one more publication arrive, then read the counter.
    std::thread::sleep(Duration::from_millis(20));
    h.step();
    let elapsed = t0.elapsed().as_secs_f64();
    let s = h.state().state.live.snapshot.clone();
    let c1 = s.stats.cycles;
    let w1: usize = h.state().state.wave.history.iter().map(|q| q.len()).sum();
    Full {
        rate: (c1 - c0) as f64 / elapsed,
        frames: frame_ms.len(),
        frame_ms,
        ui_ms,
        wave_changes: w1 - w0,
        wave_gaps: h.state().state.wave.gaps,
        busy: s.stats.busy,
        publish_us: s.stats.publish_us,
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() as f64 - 1.0) * p).round() as usize]
}

#[test]
#[ignore = "perf gate: run with --release --ignored --nocapture --test-threads=1"]
fn perf_gate() {
    let secs = env_or("PERF_SECS", 3.0);
    let rounds = env_or("PERF_ROUNDS", 3.0) as usize;
    let configs: [(&str, &[&str]); 3] = [
        ("no waves", &[]),
        ("8 nets", &NETS8),
        ("8 bus signals", &BUSES8),
    ];
    // Warm up the GPU driver (device creation, pipeline compilation) outside the measurements.
    drop(full_stack(0.5, &[]));
    let mut hs = Vec::new();
    let mut fs: Vec<Vec<f64>> = vec![Vec::new(); configs.len()];
    let mut frames = Vec::new();
    let mut uis = Vec::new();
    for r in 0..rounds {
        let h = headless(secs);
        eprintln!("round {r}: headless {:.3} M cycles/s", h / 1e6);
        hs.push(h);
        for (ci, (name, sigs)) in configs.iter().enumerate() {
            let f = full_stack(secs, sigs);
            eprintln!(
                "round {r}: full stack, {name}: {:.3} M cycles/s ({:.1} % of this round's headless), {} frames ({:.1} fps), \
                 wave changes {} ({} gap chunks), sim busy {:.1} %, publish {:.1} µs",
                f.rate / 1e6,
                100.0 * f.rate / h,
                f.frames,
                f.frames as f64 / secs,
                f.wave_changes,
                f.wave_gaps,
                100.0 * f.busy,
                f.publish_us
            );
            fs[ci].push(f.rate);
            frames.extend(f.frame_ms);
            uis.extend(f.ui_ms);
        }
    }
    let chunked = headless_chunked(secs);
    let best = |v: &[f64]| v.iter().copied().fold(0.0, f64::max);
    let best_h = best(&hs);
    let med_h = median(&mut hs.clone());
    eprintln!(
        "---- perf gate (Panda-8 SoP/Hierarchical, Fast/JIT, {secs} s per run, {rounds} rounds) ----"
    );
    eprintln!(
        "headless Kernel::run:              best {:.3} M/s, median {:.3} M/s",
        best_h / 1e6,
        med_h / 1e6
    );
    eprintln!(
        "headless run(1.5 ms) chunks:       {:.3} M/s ({:.1} %)",
        chunked / 1e6,
        100.0 * chunked / best_h
    );
    for sigs in [&NETS8[..], &BUSES8[..]] {
        let (observed, nbits) = headless_observed(secs, sigs);
        eprintln!(
            "headless run_observed, {nbits:2} nets:  {:.3} M/s ({:.1} %)  (kernel observer only, no clock synthesis)",
            observed / 1e6,
            100.0 * observed / best_h
        );
    }
    let mut ratios = Vec::new();
    for (ci, (name, _)) in configs.iter().enumerate() {
        let r = best(&fs[ci]) / best_h;
        ratios.push(r);
        eprintln!(
            "full stack (UI 60 Hz), {name:<14} best {:.3} M/s, median {:.3} M/s -> ratio best/best {:.1} %, median/median {:.1} %",
            best(&fs[ci]) / 1e6,
            median(&mut fs[ci].clone()) / 1e6,
            100.0 * r,
            100.0 * median(&mut fs[ci].clone()) / med_h
        );
    }
    eprintln!(
        "frame (step + GPU render + readback): mean {:.2} ms, p95 {:.2} ms, max {:.2} ms; UI code: mean {:.2} ms, p95 {:.2} ms",
        frames.iter().sum::<f64>() / frames.len() as f64,
        pct(&mut frames, 0.95),
        pct(&mut frames, 1.0),
        uis.iter().sum::<f64>() / uis.len() as f64,
        pct(&mut uis, 0.95)
    );
    assert!(
        ratios[0] >= 0.95,
        "no waves: full-stack throughput {:.1} % of headless (< 95 %)",
        100.0 * ratios[0]
    );
    assert!(
        ratios[1] >= 0.95,
        "8 nets: full-stack throughput {:.1} % of headless (< 95 %)",
        100.0 * ratios[1]
    );
    assert!(pct(&mut frames, 0.95) < 16.0, "frame time p95 >= 16 ms");
}
