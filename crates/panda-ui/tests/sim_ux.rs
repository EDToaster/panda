//! Offscreen (egui_kittest + wgpu) tests of the simulation UX (`panda_ui::simux`): toolbar,
//! clickable inputs, waveform viewer, diagnostics jump, STA panel.
//!
//! Writes `docs/screenshots/sim-*.png` (inspect them after visual changes).

use std::path::PathBuf;
use std::time::{Duration, Instant};

use egui::{Event, Modifiers, PointerButton, Pos2};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use panda_kernel::{Mode, StepUnit};
use panda_netlist::{Atom, Design, DesignBuilder, DefBuilder, InstKind, Logic};
use panda_ui::app::{AppState, PandaApp};
use panda_ui::doc::ItemRef;
use panda_ui::examples;
use panda_ui::simhost::{Command, SimOptions};
use panda_ui::simux::waveform::add_signal;

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

fn step_until(h: &mut Harness<'_, PandaApp>, secs: f64, what: &str, mut pred: impl FnMut(&PandaApp) -> bool) {
    let t0 = Instant::now();
    while !pred(h.state()) {
        assert!(t0.elapsed().as_secs_f64() < secs, "timed out waiting for {what}");
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

fn click_contains(h: &mut Harness<'_, PandaApp>, label: &str) {
    h.step();
    h.query_all_by_label_contains(label)
        .next()
        .unwrap_or_else(|| panic!("no widget with label containing {label:?}"))
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

/// Make the bottom panel taller by dragging its top edge up by `dy` points.
fn grow_bottom(h: &mut Harness<'_, PandaApp>, dy: f32) {
    h.step();
    let tab = h
        .query_all_by_label("Waveforms")
        .next()
        .expect("waveform tab")
        .rect();
    let x = tab.center().x + 200.0;
    // The panel edge sits a few points above the tab row; probe until a drag takes.
    for off in [6.0, 8.0, 10.0, 4.0, 12.0] {
        let y = tab.top() - off;
        let before = h.query_all_by_label("Waveforms").next().unwrap().rect().top();
        h.hover_at(Pos2::new(x, y));
        h.step();
        h.event(Event::PointerButton {
            pos: Pos2::new(x, y),
            button: PointerButton::Primary,
            pressed: true,
            modifiers: Modifiers::NONE,
        });
        h.step();
        for k in 1..=8 {
            h.hover_at(Pos2::new(x, y - dy * k as f32 / 8.0));
            h.step();
        }
        h.event(Event::PointerButton {
            pos: Pos2::new(x, y - dy),
            button: PointerButton::Primary,
            pressed: false,
            modifiers: Modifiers::NONE,
        });
        h.step();
        h.step();
        let after = h.query_all_by_label("Waveforms").next().unwrap().rect().top();
        if after < before - dy / 2.0 {
            return;
        }
    }
    panic!("could not resize the bottom panel");
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
    step_until(&mut h, 20.0, "live values", |a| a.state.live.values().is_some());
    h
}

#[test]
fn panda8_running_with_waveforms() {
    let mut state = AppState::new(examples::panda8(), SimOptions::default());
    state.wait_layout();
    let mut h = ready(state, 1400.0, 900.0);
    for name in ["clk", "PC", "OUT", "OUT_STROBE", "IR", "PHASE"] {
        assert!(add_signal(&mut h.state_mut().state, name), "{name}");
    }
    // Toolbar → sim thread: Fast mode, run.
    click(&mut h, "Fast");
    step_until(&mut h, 10.0, "Fast mode", |a| a.state.live.snapshot.mode == Mode::Fast);
    click(&mut h, "▶ Run");
    step_until(&mut h, 10.0, "running", |a| a.state.live.snapshot.running);
    // The UI keeps rendering frames while the simulation runs flat out.
    let t0 = Instant::now();
    let mut n = 0;
    while t0.elapsed() < Duration::from_millis(600) {
        h.step();
        n += 1;
    }
    let cps = h.state().state.live.snapshot.stats.cycles_per_sec;
    let tier = h.state().state.live.snapshot.tier_label();
    eprintln!("ui frames in 0.6 s while running: {n}; sim {cps:.0} cycles/s ({tier})");
    assert!(cps > 1000.0, "{cps}");
    click(&mut h, "⏸ Pause");
    step_until(&mut h, 10.0, "paused", |a| !a.state.live.snapshot.running);
    // Shadow capture catches up while paused.
    let pc = 1;
    step_until(&mut h, 20.0, "waveform history", |a| {
        let s = &a.state;
        let tick = s.live.snapshot.tick;
        s.wave
            .history
            .get(pc)
            .and_then(|h| h.back())
            .is_some_and(|&(t, _)| t + 2000 >= tick)
    });
    let changes = h.state().state.wave.history[pc].len();
    eprintln!("PC changes recorded: {changes}, gaps {}", h.state().state.wave.gaps);
    assert!(changes > 100);
    // Bring the waveform tab forward and zoom so a couple of dozen cycles are visible.
    click(&mut h, "Waveforms");
    let period = h.state().state.wave.config.signals[0]
        .clock
        .expect("clk is a clock")
        .period;
    grow_bottom(&mut h, 160.0);
    let want_tpp = f64::from(period) * 16.0 / 600.0;
    let clicks = want_tpp.log2().round() as i32;
    for _ in 0..clicks.max(0) {
        click(&mut h, "−");
    }
    frames(&mut h, 3);
    save(&mut h, "panda8-waves");
}

/// A testbench whose package drives one net from a clock buffer and a constant (contention).
fn contention_design() -> Design {
    let mut db = DesignBuilder::new();
    let mut b = DefBuilder::new("noisy_core");
    let clk = b.input("CLK", 1);
    let shared = b.buf(clk);
    let shared = b.name(shared, "bus");
    let zero = b.lit(1, 0);
    b.connect(shared, zero);
    let q = b.not(shared);
    b.output("S", q);
    let core = db.add(b.finish());
    let mut b = DefBuilder::new("top");
    let clk = b.clock(8, 4, 4);
    let clk = b.name(clk, "clk");
    let o = b.inst_named("u0", &core, &[clk]);
    b.output("S", o[0]);
    let top = db.add(b.finish());
    db.build(&top)
}

#[test]
fn diagnostics_contention_jump() {
    let mut state = AppState::new(contention_design(), SimOptions::default());
    state.wait_layout();
    let mut h = ready(state, 1300.0, 820.0);
    click(&mut h, "▶ Run");
    step_until(&mut h, 20.0, "contention diagnostic", |a| {
        a.state
            .diagnostics
            .iter()
            .any(|e| e.diagnostic.kind == panda_netlist::DiagKind::Contention)
    });
    click(&mut h, "⏸ Pause");
    click(&mut h, "Diagnostics");
    grow_bottom(&mut h, 120.0);
    let loc = h
        .state()
        .state
        .diagnostics
        .iter()
        .find(|e| e.diagnostic.kind == panda_netlist::DiagKind::Contention)
        .and_then(|e| e.diagnostic.location.clone())
        .expect("located");
    eprintln!("contention at {loc}");
    click(&mut h, &loc);
    frames(&mut h, 4);
    let s = &h.state().state;
    assert_eq!(s.view.path, "u0", "jumped into the package");
    assert!(s.highlights.contention.contains("bus"), "{:?}", s.highlights);
    save(&mut h, "diagnostics-contention");
    // The validation warning's location ("noisy_core: net bus") jumps too.
    click(&mut h, "noisy_core: net bus");
    assert_eq!(h.state().state.view.path, "u0");
}

#[test]
fn counter_input_click_hover_and_wave_menu() {
    let mut state = AppState::new(examples::counter(), SimOptions::default());
    state.wait_layout();
    let mut h = ready(state, 1100.0, 700.0);
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
    pointer_click(&mut h, pos, PointerButton::Primary);
    step_until(&mut h, 10.0, "EN = 1", |a| en_value(a) == Some(Logic::One));
    assert_ne!(before, Some(Logic::One));
    pointer_click(&mut h, pos, PointerButton::Primary);
    step_until(&mut h, 10.0, "EN = 0", |a| en_value(a) == Some(Logic::Zero));
    pointer_click(&mut h, pos, PointerButton::Primary);
    step_until(&mut h, 10.0, "EN = 1 again", |a| en_value(a) == Some(Logic::One));
    // Run a little so Q counts, then pause.
    click(&mut h, "Limit");
    click(&mut h, "▶ Run");
    std::thread::sleep(Duration::from_millis(400));
    click(&mut h, "⏸ Pause");
    step_until(&mut h, 10.0, "paused", |a| !a.state.live.snapshot.running);
    // Right-click the Q output pin → add to waveform.
    let q = item_center(h.state(), &ItemRef::Pin("Q".into()));
    pointer_click(&mut h, q, PointerButton::Secondary);
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
    // Hover the EN pin: live value label.
    h.hover_at(pos);
    frames(&mut h, 3);
    save(&mut h, "counter-input-hover");
}

/// Counter with a too-fast clock: STA suggests a longer period; "Apply" edits the clock.
#[test]
fn sta_apply_suggested_period() {
    let mut db = DesignBuilder::new();
    let c = panda_circuits::sync_counter(&mut db, 4);
    let top = panda_circuits::testbench(&mut db, &c, 4);
    let design = db.build(&top);
    let mut state = AppState::new(design, SimOptions::default());
    state.wait_layout();
    let mut h = ready(state, 1300.0, 820.0);
    click(&mut h, "Timing (STA)");
    click(&mut h, "Run STA");
    let t0 = Instant::now();
    while h.query_by_label_contains("analyzed in").is_none() {
        assert!(t0.elapsed() < Duration::from_secs(60), "STA timed out");
        h.step();
        std::thread::sleep(Duration::from_millis(10));
    }
    frames(&mut h, 2);
    assert!(h.query_by_label("too fast").is_some(), "period 4 must be too fast");
    click_contains(&mut h, "Apply ");
    frames(&mut h, 2);
    let s = &h.state().state;
    let top = s.doc.design.top_def().unwrap();
    let period = top
        .instances
        .iter()
        .find_map(|i| match &i.kind {
            InstKind::Atom(Atom::Clock { period, .. }) => Some(*period),
            _ => None,
        })
        .unwrap();
    assert!(period > 4, "clock period applied: {period}");
    // The re-analysis after the edit meets timing.
    let t0 = Instant::now();
    loop {
        h.step();
        if h.query_by_label("meets").is_some() && h.query_by_label_contains("analyzed in").is_some() {
            break;
        }
        assert!(t0.elapsed() < Duration::from_secs(60), "re-analysis timed out");
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn sta_panel_panda8() {
    let mut state = AppState::new(examples::panda8(), SimOptions::default());
    state.wait_layout();
    let mut h = ready(state, 1400.0, 900.0);
    assert!(h.state_mut().state.enter("cpu"));
    h.state_mut().state.canvas.request_fit();
    step_until(&mut h, 10.0, "cpu view", |a| {
        a.state.live.values().is_some_and(|l| l.map.path == "cpu")
    });
    // Some live values first (wires are X at tick 0).
    h.state_mut().state.sim.send(Command::Step {
        n: 40,
        unit: StepUnit::Cycle,
    });
    step_until(&mut h, 10.0, "stepped", |a| a.state.live.snapshot.tick > 1000);
    click(&mut h, "Timing (STA)");
    grow_bottom(&mut h, 230.0);
    click(&mut h, "Run STA");
    let t0 = Instant::now();
    while h.query_by_label_contains("analyzed in").is_none() {
        assert!(t0.elapsed() < Duration::from_secs(120), "STA timed out");
        h.step();
        std::thread::sleep(Duration::from_millis(10));
    }
    eprintln!("STA on Panda-8 took {:.1} s (wall, incl. compile)", t0.elapsed().as_secs_f64());
    frames(&mut h, 3);
    let hl = h.state().state.highlights.nets.len();
    eprintln!("critical-path nets highlighted in cpu view: {hl}");
    assert!(hl > 0);
    save(&mut h, "sta-panda8");
}
