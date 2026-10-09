//! Waveform viewer (`panda_ui::simux::{wave_view, waveform}`): pure axis / radix / decimation /
//! gap logic, and offscreen (egui_kittest + wgpu) renders of Panda-8's clock, PC and OUT.
//!
//! Writes `docs/screenshots/sim-wave-*.png` (inspect them after visual changes).

use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use egui_kittest::Harness;
use panda_kernel::{Mode, StepUnit};
use panda_ui::app::{AppState, Panel};
use panda_ui::examples;
use panda_ui::simhost::{ClockSpec, Command, Packed, SimOptions, WaveChange, WaveChunk};
use panda_ui::simux::wave_view::{
    GapTracker, MAX_TPP, MIN_TPP, Radix, RunKind, TimeUnit, WaveView, axis_marks, cycle_of,
    filter_candidates, format_time, format_value, rising_edge, runs, signal_candidates,
};
use panda_ui::simux::waveform::{WaveformPanel, add_signal, remove_signal, signal_names};

// ---------------------------------------------------------------------------------------------
// Pure logic
// ---------------------------------------------------------------------------------------------

fn p(v: u64) -> Packed {
    Packed { a: v, b: 0 }
}

fn view(start: f64, tpp: f64) -> WaveView {
    WaveView {
        start,
        tpp,
        follow: false,
        cursor: None,
    }
}

#[test]
fn zoom_is_anchored_and_invertible() {
    let mut v = view(1000.0, 3.0);
    for x in [0.0, 17.5, 400.0, 799.0] {
        let before = v.tick_at(x);
        v.zoom_at(x, 800.0, 1.7);
        assert!((v.tick_at(x) - before).abs() < 1e-6, "anchor at {x}");
        v.zoom_at(x, 800.0, 1.0 / 1.7);
        assert!((v.tick_at(x) - before).abs() < 1e-6);
        assert!((v.tpp - 3.0).abs() < 1e-9);
    }
    // Invalid factors are ignored; clamps hold.
    v.zoom_at(10.0, 800.0, 0.0);
    v.zoom_at(10.0, 800.0, f64::NAN);
    assert!((v.tpp - 3.0).abs() < 1e-9);
    v.zoom_at(10.0, 800.0, 1e30);
    assert_eq!(v.tpp, MIN_TPP);
    v.zoom_at(10.0, 800.0, 1e-30);
    assert_eq!(v.tpp, MAX_TPP);
    // x_of / tick_at are inverses.
    let v = view(-12.5, 0.25);
    for t in [-12.5, 0.0, 3.75, 1e6] {
        assert!((v.tick_at(v.x_of(t)) - t).abs() < 1e-6);
    }
}

#[test]
fn scroll_and_follow() {
    let mut v = WaveView::default();
    assert!(v.follow);
    v.follow_to(10_000, 500.0);
    assert_eq!(v.range(500.0), (9_500.0, 10_000.0));
    // Zooming while following keeps the newest tick at the right edge.
    v.zoom_at(3.0, 500.0, 2.0);
    assert_eq!(v.range(500.0).1, 10_000.0);
    // Scrolling (drag right = earlier) stops following.
    v.scroll(100.0);
    assert!(!v.follow);
    assert_eq!(v.range(500.0).1, 10_000.0 - 50.0);
    v.follow_to(20_000, 500.0);
    assert_eq!(v.range(500.0).1, 9_950.0);
    v.scroll(0.0);
    v.fit(100, 1100, 500.0);
    assert_eq!((v.start, v.tpp), (100.0, 2.0));
    v.set_cursor_at(-1000.0);
    assert_eq!(v.cursor, Some(0), "cursor never negative");
}

/// Brute-force reference for [`runs`]: per pixel column, the number of changes inside it and the
/// value at its end.
fn brute_columns(
    h: &[(u64, Packed)],
    v: &WaveView,
    width: usize,
    horizon: u64,
) -> Vec<(usize, Option<Packed>)> {
    (0..width)
        .map(|c| {
            let a = v.tick_at(c as f64);
            let b = v.tick_at(c as f64 + 1.0);
            let n = h
                .iter()
                // A change exactly at the left edge is the value shown there, not a transition.
                .filter(|&&(t, _)| {
                    let t = t as f64;
                    (t > a || (t == a && c > 0)) && t < b && t <= horizon as f64
                })
                .count();
            let end = h
                .iter()
                .rev()
                .find(|&&(t, _)| (t as f64) < b)
                .map(|&(_, v)| v);
            (n, end)
        })
        .collect()
}

#[test]
fn decimation_matches_brute_force() {
    // Deterministic pseudo-random histories at several zoom levels.
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for case in 0..60 {
        let n = 1 + (rnd() % 400) as usize;
        let mut t = rnd() % 50;
        let mut h = Vec::with_capacity(n);
        for _ in 0..n {
            t += 1 + rnd() % [2, 10, 200][case % 3];
            h.push((t, p(rnd() % 16)));
        }
        let last = h.last().unwrap().0;
        let tpp = [0.25, 1.0, 3.0, 17.0, 250.0][case % 5];
        let v = view((rnd() % 200) as f64, tpp);
        let width = 300usize;
        let horizon = last + 30;
        let dq: VecDeque<_> = h.iter().copied().collect();
        let rs = runs(&dq, &v, width as f64, horizon);
        let cols = brute_columns(&h, &v, width, horizon);
        let x_end = v.x_of(horizon as f64).min(width as f64);
        // Runs are ordered, non-overlapping and inside the plot.
        for w in rs.windows(2) {
            assert!(w[0].x1 <= w[1].x0 + 1e-9, "case {case}: {w:?}");
        }
        for r in &rs {
            assert!(
                r.x0 >= 0.0 && r.x1 <= x_end + 1e-9 && r.x0 < r.x1,
                "case {case}: {r:?}"
            );
        }
        assert!(rs.len() <= 2 * width + 2, "bounded by the plot width");
        for (c, &(changes, end)) in cols.iter().enumerate() {
            let mid = c as f64 + 0.5;
            if mid >= x_end {
                break;
            }
            let covering: Vec<_> = rs.iter().filter(|r| r.x0 <= mid && r.x1 > mid).collect();
            if changes >= 2 {
                assert!(
                    covering.iter().all(|r| r.kind == RunKind::Dense) && !covering.is_empty(),
                    "case {case} col {c}: {changes} changes but {covering:?}"
                );
            } else if changes == 0 {
                match end {
                    // Before the first change nothing is drawn.
                    None => assert!(covering.is_empty(), "case {case} col {c}"),
                    Some(val) => assert!(
                        covering
                            .iter()
                            .all(|r| r.kind == RunKind::Value(val) || r.kind == RunKind::Dense),
                        "case {case} col {c}: expected {val:?}, got {covering:?}"
                    ),
                }
            }
        }
    }
}

#[test]
fn decimation_of_a_million_changes_is_bounded() {
    let h: VecDeque<_> = (0..2_000_000u64).map(|t| (t * 3, p(t & 0xff))).collect();
    let v = view(0.0, 6_000_000.0 / 1000.0);
    let t0 = Instant::now();
    let rs = runs(&h, &v, 1000.0, 6_000_000);
    let el = t0.elapsed();
    assert!(rs.len() <= 3, "{} runs", rs.len());
    assert!(rs.iter().all(|r| r.kind == RunKind::Dense));
    eprintln!("decimate 2M changes to 1000 px: {el:?}");
    // Partially zoomed: per-column binary searches, never a walk over every change.
    let v = view(1_000_000.0, 0.5);
    let rs = runs(&h, &v, 1000.0, 6_000_000);
    assert!(rs.iter().all(|r| matches!(r.kind, RunKind::Value(_))));
    assert!(rs.len() >= 160 && rs.len() <= 170, "{}", rs.len());
}

#[test]
fn radix_formatting_all_widths() {
    assert_eq!(format_value(p(0xbeef), 16, Radix::Hex), "beef");
    assert_eq!(format_value(p(0xbeef), 16, Radix::Dec), "48879");
    assert_eq!(format_value(p(0xbeef), 16, Radix::Signed), "-16657");
    assert_eq!(format_value(p(5), 4, Radix::Bin), "0101");
    assert_eq!(format_value(p(0x1f), 5, Radix::Hex), "1f");
    assert_eq!(format_value(p(0x10), 5, Radix::Signed), "-16");
    assert_eq!(format_value(p(0x0f), 5, Radix::Signed), "15");
    assert_eq!(format_value(p(0), 64, Radix::Hex), "0".repeat(16));
    assert_eq!(
        format_value(p(1 << 63), 64, Radix::Signed),
        i64::MIN.to_string()
    );
    // Bits above the width are ignored.
    assert_eq!(format_value(p(0x1ff), 8, Radix::Dec), "255");
    // 4-state.
    let z8 = Packed { a: 0, b: 0xff };
    let x8 = Packed { a: 0xff, b: 0xff };
    assert_eq!(format_value(z8, 8, Radix::Hex), "zz");
    assert_eq!(format_value(x8, 8, Radix::Hex), "xx");
    assert_eq!(format_value(x8, 8, Radix::Signed), "x");
    assert_eq!(format_value(z8, 8, Radix::Dec), "z");
    assert_eq!(
        format_value(Packed { a: 0x0f, b: 0xf0 }, 8, Radix::Hex),
        "zf"
    );
    assert_eq!(format_value(Packed { a: 0, b: 1 }, 1, Radix::Bin), "z");
    assert_eq!(format_value(p(1), 1, Radix::Signed), "1");
}

#[test]
fn time_axis_in_ticks_and_cycles() {
    let clk = ClockSpec {
        period: 40,
        high: 20,
        phase: 5,
    };
    assert_eq!(rising_edge(&clk, 0), 5);
    assert_eq!(rising_edge(&clk, 3), 125);
    assert_eq!(cycle_of(&clk, 125.0), 3.0);
    assert_eq!(format_time(125, TimeUnit::Cycles, Some(&clk)), "c3");
    assert_eq!(format_time(131, TimeUnit::Cycles, Some(&clk)), "c3+6");
    assert_eq!(format_time(2, TimeUnit::Cycles, Some(&clk)), "c-1+37");
    assert_eq!(format_time(131, TimeUnit::Ticks, Some(&clk)), "131");
    assert_eq!(format_time(131, TimeUnit::Cycles, None), "131");

    // Ticks: nice multiples, ≥ min_px apart, inside the range.
    let v = view(3.0, 2.5);
    let m = axis_marks(&v, 800.0, 90.0, TimeUnit::Ticks, None);
    assert!(!m.is_empty());
    let (t0, t1) = v.range(800.0);
    for w in m.windows(2) {
        assert!(v.x_of(w[1].tick as f64) - v.x_of(w[0].tick as f64) >= 90.0 - 1e-9);
    }
    assert!(
        m.iter()
            .all(|k| (k.tick as f64) >= t0 && (k.tick as f64) <= t1)
    );
    assert_eq!(m[0].label, m[0].tick.to_string());
    assert_eq!(m[0].tick % 500, 0);

    // Cycles: on rising edges, whole nice cycle counts.
    let v = view(0.0, 4.0); // 10 px per cycle
    let m = axis_marks(&v, 1000.0, 90.0, TimeUnit::Cycles, Some(&clk));
    assert_eq!(m[0].tick, 5);
    assert_eq!(m[0].label, "c0");
    assert_eq!(m[1].label, "c10");
    assert_eq!(m[1].tick, rising_edge(&clk, 10));
    // Zoomed far out: still bounded.
    let v = view(0.0, 1e9);
    assert!(axis_marks(&v, 1000.0, 90.0, TimeUnit::Cycles, Some(&clk)).len() <= 13);
    assert!(axis_marks(&v, 1000.0, 90.0, TimeUnit::Ticks, None).len() <= 13);
}

fn chunk(config: u64, gap: bool, ticks: &[u64]) -> WaveChunk {
    WaveChunk {
        config,
        gap,
        changes: ticks
            .iter()
            .map(|&tick| WaveChange {
                tick,
                signal: 0,
                value: p(tick),
            })
            .collect(),
    }
}

#[test]
fn gap_tracker_locates_lost_intervals() {
    let mut g = GapTracker::default();
    g.on_chunk(&chunk(4, false, &[10, 20, 30]));
    assert!(g.gaps.is_empty());
    g.on_chunk(&chunk(4, true, &[500, 510]));
    assert_eq!(g.gaps, vec![(30, 500)]);
    // A gap flag on an empty chunk stays open until data arrives.
    g.on_chunk(&chunk(4, true, &[]));
    assert_eq!(g.open, Some(510));
    g.on_chunk(&chunk(4, false, &[900]));
    assert_eq!(g.gaps, vec![(30, 500), (510, 900)]);
    assert_eq!(g.visible(0.0, 100.0).collect::<Vec<_>>(), vec![(30, 500)]);
    assert_eq!(
        g.visible(505.0, 600.0).collect::<Vec<_>>(),
        vec![(510, 900)]
    );
    assert_eq!(g.visible(950.0, 1e9).count(), 0);
    // A new config starts afresh.
    g.on_chunk(&chunk(5, false, &[1]));
    assert_eq!((g.config, g.gaps.len()), (5, 0));
}

#[test]
fn picker_candidates() {
    let names = [
        "cpu/alu/c[1]",
        "PC[0]",
        "PC[1]",
        "clk",
        "cpu/alu/c[0]",
        "cpu/x",
        "IR[3]",
    ];
    let c = signal_candidates(names.iter().copied());
    assert_eq!(c, vec!["IR", "PC", "clk", "cpu/x", "cpu/alu/c"]);
    assert_eq!(filter_candidates(&c, "ALU c", 10), vec!["cpu/alu/c"]);
    assert_eq!(filter_candidates(&c, "", 2), vec!["IR", "PC"]);
    assert!(filter_candidates(&c, "nope", 10).is_empty());
}

// ---------------------------------------------------------------------------------------------
// Offscreen renders
// ---------------------------------------------------------------------------------------------

/// The waveform panel alone, driven like the shell drives it.
struct WaveApp {
    app: AppState,
    panel: WaveformPanel,
}

fn harness(w: WaveApp, width: f32, height: f32) -> Harness<'static, WaveApp> {
    Harness::builder()
        .with_size(egui::vec2(width, height))
        .wgpu()
        .build_ui_state(
            |ui, s: &mut WaveApp| {
                let events = s.app.update();
                for ev in &events {
                    s.panel.on_event(ev, &mut s.app);
                }
                egui::Frame::central_panel(ui.style()).show(ui, |ui| {
                    s.panel.ui(ui, &mut s.app);
                });
            },
            w,
        )
}

fn step_until(
    h: &mut Harness<'_, WaveApp>,
    secs: f64,
    what: &str,
    mut pred: impl FnMut(&WaveApp) -> bool,
) {
    let t0 = Instant::now();
    while !pred(h.state()) {
        assert!(
            t0.elapsed().as_secs_f64() < secs,
            "timed out waiting for {what}"
        );
        h.step();
        std::thread::sleep(Duration::from_millis(5));
    }
    h.step();
}

fn save(h: &mut Harness<'_, WaveApp>, name: &str) {
    let img = h.render().expect("render");
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/screenshots")
        .join(format!("sim-wave-{name}.png"));
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    img.save(&p).unwrap();
}

/// Panda-8 with `signals` in the viewer, stepped `cycles` cycles in `mode`, capture complete.
fn panda8(
    mode: Mode,
    cycles: u64,
    signals: &[&str],
    w: f32,
    hgt: f32,
) -> Harness<'static, WaveApp> {
    let app = AppState::new(examples::panda8(), SimOptions::default());
    let mut h = harness(
        WaveApp {
            app,
            panel: WaveformPanel::new(),
        },
        w,
        hgt,
    );
    step_until(&mut h, 30.0, "netlist", |s| s.app.live.flat.is_some());
    for name in signals {
        assert!(add_signal(&mut h.state_mut().app, name), "{name}");
    }
    assert_eq!(signal_names(&h.state().app), signals);
    if mode == Mode::Fast {
        h.state_mut().app.sim.send(Command::SetMode(Mode::Fast));
        step_until(&mut h, 30.0, "fast mode", |s| {
            s.app.live.snapshot.mode == Mode::Fast
        });
    }
    let t_before = h.state().app.live.snapshot.tick;
    h.state_mut().app.sim.send(Command::Step {
        n: cycles,
        unit: StepUnit::Cycle,
    });
    step_until(&mut h, 60.0, "stepping", |s| {
        let sn = &s.app.live.snapshot;
        sn.tick > t_before && !sn.running
    });
    // Capture is complete once paused: wait until the history stops growing.
    let mut last = usize::MAX;
    let mut stable = 0;
    step_until(&mut h, 30.0, "capture", |s| {
        let n: usize = s.app.wave.history.iter().map(VecDeque::len).sum();
        if n == last && n > 0 {
            stable += 1;
        } else {
            stable = 0;
        }
        last = n;
        std::thread::sleep(Duration::from_millis(20));
        stable >= 10
    });
    h
}

/// Fit the whole recorded range (the "Fit" button does the same).
fn fit(h: &mut Harness<'_, WaveApp>) {
    let s = h.state_mut();
    let t0 = s
        .app
        .wave
        .history
        .iter()
        .filter_map(|h| h.front().map(|x| x.0))
        .min()
        .unwrap_or(0);
    let t1 = s.app.live.snapshot.tick;
    s.panel.view.follow = false;
    let w = 1100.0 - 150.0 - 96.0 - 16.0;
    s.panel.view.fit(t0, t1, w);
    h.step();
    h.step();
}

#[test]
fn panda8_timing_waves() {
    let mut h = panda8(
        Mode::Timing,
        24,
        &["clk", "PC", "OUT", "OUT_STROBE", "IR"],
        1100.0,
        220.0,
    );
    let s = h.state();
    let pc = &s.app.wave.history[1];
    eprintln!(
        "Timing: tick {}, PC changes {}, OUT changes {}, gaps {}",
        s.app.live.snapshot.tick,
        pc.len(),
        s.app.wave.history[2].len(),
        s.app.wave.gaps
    );
    assert!(pc.len() > 5, "PC changes {}", pc.len());
    assert!(
        s.app.wave.config.signals[0].clock.is_some(),
        "clk is a clock"
    );
    fit(&mut h);
    // Cursor somewhere in the middle; OUT in unsigned decimal.
    let s = h.state_mut();
    let mid = s.app.live.snapshot.tick / 2;
    s.panel.view.cursor = Some(mid);
    s.panel.radix.insert("OUT".into(), Radix::Dec);
    h.step();
    h.step();
    save(&mut h, "timing");
}

#[test]
fn panda8_fast_cycles_axis_and_gap() {
    let mut h = panda8(
        Mode::Fast,
        400,
        &["clk", "PC", "OUT", "OUT_STROBE"],
        1100.0,
        200.0,
    );
    let s = h.state();
    eprintln!(
        "Fast: tick {}, PC changes {}, gaps {}",
        s.app.live.snapshot.tick,
        s.app.wave.history[1].len(),
        s.app.wave.gaps
    );
    assert!(s.app.wave.history[1].len() > 100);
    // Default unit in Fast mode: cycles of clk.
    fit(&mut h);
    save(&mut h, "fast-fit");
    // Zoom in on ~40 cycles, signed OUT, with a (synthetic) capture gap in view.
    let s = h.state_mut();
    let period = f64::from(s.app.wave.config.signals[0].clock.unwrap().period);
    let w = 1100.0 - 150.0 - 96.0 - 16.0;
    s.panel.view.tpp = period * 40.0 / w;
    s.panel.view.start = period * 100.0;
    s.panel.view.cursor = Some((period * 112.0) as u64);
    s.panel.radix.insert("PC".into(), Radix::Bin);
    s.panel.radix.insert("OUT".into(), Radix::Signed);
    let id = s.app.wave.config.id;
    s.panel.gaps = GapTracker {
        config: id,
        last: None,
        gaps: vec![((period * 126.0) as u64, (period * 131.0) as u64)],
        open: None,
    };
    h.step();
    h.step();
    save(&mut h, "fast-zoom-gap");
    // Ticks unit on request.
    h.state_mut().panel.unit = Some(TimeUnit::Ticks);
    h.step();
    h.step();
    // Remove a signal through the public hook.
    remove_signal(&mut h.state_mut().app, 3);
    h.step();
    assert_eq!(signal_names(&h.state().app), ["clk", "PC", "OUT"]);
}

#[test]
fn panda8_running_flat_out_with_follow() {
    let app = AppState::new(examples::panda8(), SimOptions::default());
    let mut h = harness(
        WaveApp {
            app,
            panel: WaveformPanel::new(),
        },
        1100.0,
        200.0,
    );
    step_until(&mut h, 30.0, "netlist", |s| s.app.live.flat.is_some());
    for name in ["clk", "PC", "OUT", "IR"] {
        assert!(add_signal(&mut h.state_mut().app, name));
    }
    h.state_mut().app.sim.send(Command::SetMode(Mode::Fast));
    step_until(&mut h, 30.0, "fast", |s| {
        s.app.live.snapshot.mode == Mode::Fast
    });
    h.state_mut().app.sim.send(Command::Run);
    step_until(&mut h, 10.0, "running", |s| s.app.live.snapshot.running);
    // Frames keep coming while the sim runs; the viewer follows the capture horizon. (The
    // shadow capture compiles its own Fast-mode kernels first, so data may start late.)
    let t0 = Instant::now();
    let mut frames = 0u32;
    let mut worst = Duration::ZERO;
    loop {
        let f = Instant::now();
        h.step();
        worst = worst.max(f.elapsed());
        frames += 1;
        let changes: usize = h.state().app.wave.history.iter().map(VecDeque::len).sum();
        if t0.elapsed() > Duration::from_millis(1500) && changes > 5000 {
            break;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(60),
            "no waveform data while running"
        );
    }
    let el = t0.elapsed();
    let s = h.state();
    let cps = s.app.live.snapshot.stats.cycles_per_sec;
    let changes: usize = s.app.wave.history.iter().map(VecDeque::len).sum();
    eprintln!(
        "running: {frames} frames in {el:.1?} (worst {worst:?}), {cps:.0} cycles/s, {changes} \
         changes, gaps {} (tracked {})",
        s.app.wave.gaps,
        s.panel.gaps.gaps.len()
    );
    assert!(changes > 1000);
    assert!(s.panel.view.follow);
    // Zoom out to a million-sample overview (decimated) and render.
    let s = h.state_mut();
    s.panel.view.tpp = 2000.0;
    h.step();
    save(&mut h, "running");
    h.state_mut().app.sim.send(Command::Pause);
    step_until(&mut h, 10.0, "paused", |s| !s.app.live.snapshot.running);
}

fn press(h: &mut Harness<'_, WaveApp>, pos: egui::Pos2, pressed: bool) {
    h.event(egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    });
    h.step();
}

fn wheel(h: &mut Harness<'_, WaveApp>, pos: egui::Pos2, dy: f32, modifiers: egui::Modifiers) {
    h.hover_at(pos);
    h.step();
    h.event(egui::Event::MouseWheel {
        unit: egui::MouseWheelUnit::Point,
        delta: egui::vec2(0.0, dy),
        phase: egui::TouchPhase::Move,
        modifiers,
    });
    for _ in 0..20 {
        h.step(); // smooth scrolling is spread over frames
    }
}

#[test]
fn plot_mouse_navigation_and_picker() {
    use egui_kittest::kittest::Queryable;
    let mut h = panda8(Mode::Timing, 6, &["clk", "PC"], 1100.0, 300.0);
    fit(&mut h);
    let v0 = h.state().panel.view;
    let at = egui::pos2(700.0, 80.0); // inside the plot, on a row
    // Click: cursor at the tick under the pointer.
    h.hover_at(at);
    h.step();
    press(&mut h, at, true);
    press(&mut h, at, false);
    h.step();
    let c = h.state().panel.view.cursor.expect("cursor placed");
    let v = h.state().panel.view;
    assert!((v.start - v0.start).abs() < 1e-9, "a click does not scroll");
    // Ctrl+wheel zooms in at the pointer: the tick under it stays put.
    let under = v.tick_at(f64::from(at.x) - 262.0);
    wheel(&mut h, at, 120.0, egui::Modifiers::CTRL);
    let v1 = h.state().panel.view;
    assert!(v1.tpp < v.tpp * 0.9, "zoomed in: {} -> {}", v.tpp, v1.tpp);
    let under1 = v1.tick_at(f64::from(at.x) - 262.0);
    assert!(
        (under1 - under).abs() <= 2.0 * v.tpp,
        "anchor drifted: {under} -> {under1} (cursor {c})"
    );
    // Plain wheel scrolls in time.
    wheel(&mut h, at, -200.0, egui::Modifiers::NONE);
    let v2 = h.state().panel.view;
    assert!(v2.start > v1.start, "wheel down → later: {} -> {}", v1.start, v2.start);
    assert!((v2.tpp - v1.tpp).abs() < 1e-12);
    // Drag right by 100 px → 100 px earlier.
    h.hover_at(at);
    h.step();
    press(&mut h, at, true);
    for k in 1..=10 {
        h.hover_at(at + egui::vec2(10.0 * k as f32, 0.0));
        h.step();
    }
    press(&mut h, at + egui::vec2(100.0, 0.0), false);
    let v3 = h.state().panel.view;
    assert!(!v3.follow);
    let moved = (v2.start - v3.start) / v3.tpp;
    assert!((moved - 100.0).abs() < 12.0, "dragged {moved} px");
    // Picker: filter and add a net.
    h.query_by_label("+ Add signal").expect("picker button").click();
    h.step();
    h.step();
    h.get_by_role(egui::accesskit::Role::TextInput).type_text("out_str");
    h.step();
    h.step();
    save(&mut h, "picker");
    h.query_by_label("OUT_STROBE").expect("filtered entry").click();
    h.step();
    h.step();
    assert_eq!(signal_names(&h.state().app), ["clk", "PC", "OUT_STROBE"]);
}
