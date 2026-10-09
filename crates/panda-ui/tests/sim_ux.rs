//! Offscreen (egui_kittest + wgpu) tests of the run controls of the simulation UX
//! (`panda_ui::simux`): toolbar → sim-thread commands, speed-limit pacing accuracy, clickable /
//! editable top-level inputs, hover values, the perf HUD, and the "Set input value…" context
//! entry. (Waveforms: `sim_wave.rs`; diagnostics and STA: `sim_diag.rs`.)
//!
//! Writes `docs/screenshots/sim-controls-*.png` (inspect them after visual changes).

use std::path::PathBuf;
use std::time::{Duration, Instant};

use egui::{Event, Key, Modifiers, PointerButton, Pos2};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use panda_kernel::Mode;
use panda_netlist::Logic;
use panda_ui::app::{AppState, PandaApp};
use panda_ui::doc::ItemRef;
use panda_ui::examples;
use panda_ui::simhost::{Command, SimHost, SimOptions, Snapshot, Speed};

fn screenshot_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/screenshots")
        .join(format!("sim-{name}.png"))
}

fn harness(app: PandaApp, w: f32, h: f32) -> Harness<'static, PandaApp> {
    Harness::builder()
        .with_size(egui::vec2(w, h))
        .wgpu()
        .build_ui_state(|ui, app: &mut PandaApp| app.show(ui), app)
}

fn step_until(
    h: &mut Harness<'_, PandaApp>,
    secs: f64,
    what: &str,
    mut pred: impl FnMut(&PandaApp) -> bool,
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

fn frames(h: &mut Harness<'_, PandaApp>, n: usize) {
    for _ in 0..n {
        h.step();
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn save(h: &mut Harness<'_, PandaApp>, name: &str) {
    let img = h.render().expect("render");
    let p = screenshot_path(name);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    img.save(&p).unwrap();
}

/// Click the first widget labelled exactly `label` (after letting the layout settle, so the
/// accessibility tree matches the frame that handles the click).
fn click(h: &mut Harness<'_, PandaApp>, label: &str) {
    h.step();
    h.query_all_by_label(label)
        .next()
        .unwrap_or_else(|| panic!("no widget labelled {label:?}"))
        .click();
    h.step();
    h.step();
}

fn pointer_click(h: &mut Harness<'_, PandaApp>, pos: Pos2, button: PointerButton) {
    h.hover_at(pos);
    h.step();
    for pressed in [true, false] {
        h.event(Event::PointerButton {
            pos,
            button,
            pressed,
            modifiers: Modifiers::NONE,
        });
        h.step();
    }
}

/// Screen centre of an item of the viewed def.
fn item_center(app: &PandaApp, item: &ItemRef) -> Pos2 {
    let s = &app.state;
    let scene = s.canvas.scene().expect("scene");
    let p = scene.item(item).expect("item on canvas");
    s.canvas.xf.rect_to_screen(&p.rect()).center()
}

fn ready(state: AppState, w: f32, hgt: f32) -> Harness<'static, PandaApp> {
    let mut h = harness(PandaApp::new(state), w, hgt);
    step_until(&mut h, 20.0, "live values", |a| {
        a.state.live.values().is_some()
    });
    h
}

#[test]
fn counter_input_click_toggle_and_hover() {
    let mut state = AppState::new(examples::counter(), SimOptions::default());
    state.wait_layout();
    let mut h = ready(state, 1500.0, 700.0);
    let en = ItemRef::Pin("EN".into());
    let pos = item_center(h.state(), &en);
    let en_value = |a: &PandaApp| {
        a.state
            .live
            .values()
            .and_then(|l| l.map.pin_value("EN", l.values))
            .map(|v| v.get(0))
    };
    let before = en_value(h.state());
    assert_ne!(before, Some(Logic::One));
    // While paused an input change applies at the next step; the pin shows it as pending.
    pointer_click(&mut h, pos, PointerButton::Primary);
    frames(&mut h, 3);
    assert!(
        h.query_by_label_contains("→ 1 at the next step").is_some(),
        "pending value in the hover label"
    );
    save(&mut h, "controls-counter-pending");
    click(&mut h, "tick");
    step_until(&mut h, 10.0, "EN = 1", |a| en_value(a) == Some(Logic::One));
    // Two quick clicks within one publication toggle twice (1 → 0 → 1).
    pointer_click(&mut h, pos, PointerButton::Primary);
    pointer_click(&mut h, pos, PointerButton::Primary);
    click(&mut h, "tick");
    frames(&mut h, 5);
    assert_eq!(en_value(h.state()), Some(Logic::One));
    pointer_click(&mut h, pos, PointerButton::Primary);
    click(&mut h, "tick");
    step_until(&mut h, 10.0, "EN = 0", |a| en_value(a) == Some(Logic::Zero));
    pointer_click(&mut h, pos, PointerButton::Primary);
    click(&mut h, "tick");
    step_until(&mut h, 10.0, "EN = 1 again", |a| {
        en_value(a) == Some(Logic::One)
    });
    // Run a little so Q counts, then pause.
    click(&mut h, "Limit");
    click(&mut h, "▶ Run");
    std::thread::sleep(Duration::from_millis(400));
    click(&mut h, "⏸ Pause");
    step_until(&mut h, 10.0, "paused", |a| !a.state.live.snapshot.running);
    // Hover the EN pin: live value label.
    h.hover_at(pos);
    frames(&mut h, 3);
    assert!(
        h.query_by_label_contains("EN = 1").is_some(),
        "hover label shows the live value"
    );
    save(&mut h, "controls-counter-hover");
}

// ---- run controls --------------------------------------------------------------------------

fn snap<'a>(h: &'a Harness<'_, PandaApp>) -> &'a Snapshot {
    &h.state().state.live.snapshot
}

/// Step frames for `secs` (the UI keeps rendering; nothing waits on the sim).
fn run_frames(h: &mut Harness<'_, PandaApp>, secs: f64) {
    let t0 = Instant::now();
    while t0.elapsed().as_secs_f64() < secs {
        h.step();
        std::thread::sleep(Duration::from_millis(4));
    }
}

/// Every toolbar gesture reaches the sim thread as the right command.
#[test]
fn toolbar_drives_the_sim_thread() {
    let mut state = AppState::new(examples::counter(), SimOptions::default());
    state.wait_layout();
    let mut h = ready(state, 1500.0, 640.0);
    assert_eq!(snap(&h).mode, Mode::Timing);
    assert!(!snap(&h).running);
    // Step tick / phase / cycle (Timing mode; counter clock period 16, high 8).
    let t0 = snap(&h).tick;
    click(&mut h, "tick");
    step_until(&mut h, 10.0, "one tick", |a| {
        a.state.live.snapshot.tick == t0 + 1
    });
    let t1 = t0 + 1;
    click(&mut h, "phase");
    step_until(&mut h, 10.0, "one phase", |a| {
        a.state.live.snapshot.tick > t1
    });
    let t2 = snap(&h).tick;
    assert!(t2 - t1 <= 8, "a phase is at most 8 ticks: {t1} → {t2}");
    let c0 = snap(&h).stats.cycles;
    click(&mut h, "cycle");
    step_until(&mut h, 10.0, "one cycle", |a| {
        a.state.live.snapshot.stats.cycles == c0 + 1
    });
    let t3 = snap(&h).tick;
    assert!(
        (9..=16).contains(&(t3 - t2)),
        "a cycle ends at the next cycle end: {t2} → {t3}"
    );
    // F10 repeats the last unit (cycle), F5 toggles run.
    h.key_press(Key::F10);
    step_until(&mut h, 10.0, "F10 cycle", |a| {
        a.state.live.snapshot.stats.cycles == c0 + 2
    });
    assert_eq!(snap(&h).tick, t3 + 16);
    h.key_press(Key::F5);
    step_until(&mut h, 10.0, "F5 run", |a| a.state.live.snapshot.running);
    h.key_press(Key::F5);
    step_until(&mut h, 10.0, "F5 pause", |a| !a.state.live.snapshot.running);
    // Mode.
    click(&mut h, "Fast");
    step_until(&mut h, 10.0, "Fast mode", |a| {
        a.state.live.snapshot.mode == Mode::Fast
    });
    // Speed: the limit checkbox, then a preset.
    click(&mut h, "Limit");
    step_until(&mut h, 10.0, "limited", |a| {
        a.state.live.snapshot.speed == Speed::CyclesPerSec(10.0)
    });
    click(&mut h, "⏷");
    save(&mut h, "controls-speed-presets");
    click(&mut h, "1000 cyc/s");
    step_until(&mut h, 10.0, "1000 cyc/s", |a| {
        a.state.live.snapshot.speed == Speed::CyclesPerSec(1000.0)
    });
    // Run paced: the sim thread paces itself while the UI keeps rendering frames.
    click(&mut h, "▶ Run");
    step_until(&mut h, 10.0, "running", |a| a.state.live.snapshot.running);
    run_frames(&mut h, 0.3);
    let (w0, k0) = (Instant::now(), snap(&h).stats.cycles);
    run_frames(&mut h, 1.2);
    let (w1, k1) = (Instant::now(), snap(&h).stats.cycles);
    let rate = (k1 - k0) as f64 / (w1 - w0).as_secs_f64();
    eprintln!("toolbar-paced rate: {rate:.0} cycles/s (target 1000)");
    assert!((700.0..=1100.0).contains(&rate), "paced rate {rate}");
    assert!(
        h.query_by_label_contains("/ 1000 cyc/s").is_some(),
        "HUD shows the measured and target rate"
    );
    save(&mut h, "controls-toolbar-limited");
    // Unlimited again, then pause.
    click(&mut h, "Limit");
    step_until(&mut h, 10.0, "unlimited", |a| {
        a.state.live.snapshot.speed == Speed::Unlimited
    });
    run_frames(&mut h, 0.6);
    let fast = snap(&h).stats.cycles_per_sec;
    eprintln!("unlimited counter: {fast:.0} cycles/s");
    assert!(fast > 20_000.0, "{fast}");
    click(&mut h, "⏸ Pause");
    step_until(&mut h, 10.0, "paused", |a| !a.state.live.snapshot.running);
    // Reset reloads the design: a new generation, cycle count back to 0.
    let g = snap(&h).design_gen;
    click(&mut h, "⟲ Reset");
    step_until(&mut h, 10.0, "reset", |a| {
        let s = &a.state.live.snapshot;
        s.design_gen > g && s.stats.cycles == 0 && s.tick == 0
    });
    assert_eq!(snap(&h).mode, Mode::Fast, "reset keeps the mode");
}

/// Wait until a snapshot satisfies `pred` (headless host).
fn wait_host(h: &mut SimHost, secs: f64, mut pred: impl FnMut(&Snapshot) -> bool) -> Snapshot {
    let t0 = Instant::now();
    loop {
        let _ = h.poll_events();
        let s = h.snapshot();
        if pred(s) {
            return s.clone();
        }
        assert!(
            t0.elapsed().as_secs_f64() < secs,
            "timed out: {:?}",
            h.snapshot()
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// Measured cycle rate of a running host over `secs` of wall time.
fn measure(h: &mut SimHost, secs: f64) -> f64 {
    let _ = h.poll_events();
    // Start right after a fresh publication so both ends have the same staleness.
    let seq = h.snapshot().seq;
    let s0 = wait_host(h, 5.0, |s| s.seq > seq);
    let t0 = Instant::now();
    std::thread::sleep(Duration::from_secs_f64(secs));
    let _ = h.poll_events();
    let seq = h.snapshot().seq;
    let s1 = wait_host(h, 5.0, |s| s.seq > seq);
    (s1.stats.cycles - s0.stats.cycles) as f64 / t0.elapsed().as_secs_f64()
}

/// The sim thread paces a target rate itself (never by blocking on the UI), accurately across
/// four decades, in both modes; UI-side sends never block.
#[test]
fn speed_limit_pacing_accuracy() {
    let mut h = SimHost::spawn(SimOptions::default());
    h.send(Command::Load(Box::new(examples::counter())));
    wait_host(&mut h, 10.0, |s| s.design_gen > 0);
    let mut report = Vec::new();
    for (mode, rates) in [
        (Mode::Fast, &[5.0, 200.0, 5_000.0, 100_000.0][..]),
        (Mode::Timing, &[1_000.0][..]),
    ] {
        h.send(Command::SetMode(mode));
        wait_host(&mut h, 10.0, |s| s.mode == mode);
        for &rate in rates {
            h.send(Command::SetSpeed(Speed::CyclesPerSec(rate)));
            h.send(Command::Run);
            wait_host(&mut h, 5.0, |s| {
                s.running && s.speed == Speed::CyclesPerSec(rate)
            });
            let secs = if rate < 50.0 { 2.0 } else { 1.0 };
            let got = measure(&mut h, secs);
            report.push(format!(
                "{mode:?} {rate}/s → {got:.1}/s ({:+.1} %)",
                (got / rate - 1.0) * 100.0
            ));
            // Never faster than asked (one cycle of slack for the window edges); slower only
            // when the machine is loaded (other builds run concurrently).
            let slack = 1.5 / secs;
            assert!(
                got <= rate * 1.03 + slack,
                "{mode:?} {rate}: {got} (too fast)"
            );
            assert!(
                got >= rate * 0.8 - slack,
                "{mode:?} {rate}: {got} (too slow)"
            );
            h.send(Command::Pause);
            wait_host(&mut h, 5.0, |s| !s.running);
        }
    }
    eprintln!("pacing: {}", report.join("; "));
    // Sends are non-blocking even when the sim thread is busy running flat out and the command
    // channel overflows (commands queue on the UI side and are retried).
    h.send(Command::SetSpeed(Speed::Unlimited));
    h.send(Command::Run);
    wait_host(&mut h, 5.0, |s| s.running && s.speed == Speed::Unlimited);
    let mut worst = Duration::ZERO;
    for i in 0..5000u64 {
        let t = Instant::now();
        h.send(Command::SetInput {
            pin: "EN".into(),
            value: panda_netlist::Value::from_bool(i % 2 == 0),
        });
        worst = worst.max(t.elapsed());
    }
    eprintln!("worst UI-side send: {worst:?}");
    assert!(
        worst < Duration::from_millis(5),
        "send blocked for {worst:?}"
    );
}

/// A multi-bit top-level input opens an editor; bad text shows an error, good text sets it.
#[test]
fn multi_bit_input_editor() {
    let mut state = AppState::new(examples::register_file_4x8(), SimOptions::default());
    state.wait_layout();
    let mut h = ready(state, 1500.0, 760.0);
    let wd = item_center(h.state(), &ItemRef::Pin("WD".into()));
    pointer_click(&mut h, wd, PointerButton::Primary);
    frames(&mut h, 2);
    assert!(
        h.query_by_label_contains("Set input WD [8]").is_some(),
        "editor opened"
    );
    let clear = |h: &mut Harness<'_, PandaApp>| {
        for _ in 0..12 {
            h.key_press(Key::Backspace);
        }
    };
    clear(&mut h);
    h.event(Event::Text("300".into()));
    h.step();
    h.key_press(Key::Enter);
    frames(&mut h, 2);
    assert!(
        h.query_by_label_contains("Set input WD").is_some(),
        "still open after an error"
    );
    save(&mut h, "controls-input-editor-error");
    // The field keeps the focus after an error: replace the text, press "Set".
    clear(&mut h);
    h.event(Event::Text("0x5a".into()));
    frames(&mut h, 2);
    save(&mut h, "controls-input-editor");
    click(&mut h, "Set");
    click(&mut h, "tick");
    let wd_value = |a: &PandaApp| {
        a.state
            .live
            .values()
            .and_then(|l| l.map.pin_value("WD", l.values))
            .and_then(|v| v.to_u64())
    };
    step_until(&mut h, 10.0, "WD = 0x5a", |a| wd_value(a) == Some(0x5a));
    assert!(
        h.query_by_label_contains("Set input WD").is_none(),
        "editor closed"
    );
    // Hovering the pin shows the bus value in hex and decimal.
    h.hover_at(wd);
    frames(&mut h, 3);
    assert!(h.query_by_label_contains("WD = 0x5a (90)").is_some());
    save(&mut h, "controls-input-hover");
}

/// Panda-8 flat out in Fast mode: the HUD reports the JIT tier and the UI frames stay cheap.
#[test]
fn panda8_flat_out_hud() {
    let mut state = AppState::new(examples::panda8(), SimOptions::default());
    state.wait_layout();
    let mut h = ready(state, 1400.0, 860.0);
    click(&mut h, "Fast");
    click(&mut h, "▶ Run");
    step_until(&mut h, 60.0, "JIT tier", |a| {
        let s = &a.state.live.snapshot;
        s.running && s.jit
    });
    run_frames(&mut h, 1.5);
    let s = snap(&h);
    let (cps, ms) = (s.stats.cycles_per_sec, h.state().state.frame.avg_ms);
    eprintln!(
        "Panda-8 flat out: {cps:.0} cycles/s, UI code {ms:.2} ms/frame, busy {:.2}",
        s.stats.busy
    );
    assert!(cps > 10_000.0, "{cps}");
    assert!(ms < 16.0, "UI frame code {ms} ms");
    assert!(h.query_by_label("JIT").is_some(), "HUD shows the tier");
    assert!(h.query_by_label_contains("cyc/s").is_some());
    save(&mut h, "controls-panda8-running");
    click(&mut h, "⏸ Pause");
}

/// The canvas context menu (editing module) has "Add to waveform" and, for top-level inputs,
/// this module's "Set input value…" entry.
#[test]
fn context_menu_wave_and_set_value() {
    let mut state = AppState::new(examples::counter(), SimOptions::default());
    state.wait_layout();
    let mut h = ready(state, 1500.0, 700.0);
    // Right-click the Q output pin → add to waveform.
    let q = item_center(h.state(), &ItemRef::Pin("Q".into()));
    pointer_click(&mut h, q, PointerButton::Secondary);
    frames(&mut h, 1);
    click(&mut h, "Add to waveform");
    frames(&mut h, 2);
    let names: Vec<String> = h
        .state()
        .state
        .wave
        .config
        .signals
        .iter()
        .map(|s| s.name.clone())
        .collect();
    assert_eq!(names, ["Q"]);
    // Right-click the EN input → "Set input value…" opens the editor.
    let en = item_center(h.state(), &ItemRef::Pin("EN".into()));
    pointer_click(&mut h, en, PointerButton::Secondary);
    frames(&mut h, 1);
    save(&mut h, "controls-context-menu");
    click(&mut h, "Set input value…");
    frames(&mut h, 2);
    assert!(
        h.query_by_label_contains("Set input EN [1]").is_some(),
        "editor opened"
    );
    h.event(Event::Text("1".into()));
    h.key_press(Key::Enter);
    click(&mut h, "tick");
    step_until(&mut h, 10.0, "EN = 1", |a| {
        a.state
            .live
            .values()
            .and_then(|l| l.map.pin_value("EN", l.values))
            .is_some_and(|v| v.get(0) == Logic::One)
    });
}
