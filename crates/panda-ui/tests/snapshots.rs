//! Offscreen rendering (egui_kittest + wgpu) of the app shell and canvas.
//!
//! Writes `docs/screenshots/foundation-*.png` (inspect them after visual changes).

use std::path::PathBuf;
use std::time::{Duration, Instant};

use egui_kittest::Harness;
use panda_kernel::{Mode, StepUnit};
use panda_netlist::Value;
use panda_ui::app::{AppState, PandaApp};
use panda_ui::examples;
use panda_ui::simhost::{Command, SimOptions};

fn screenshot_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/screenshots")
        .join(format!("foundation-{name}.png"))
}

fn harness(app: PandaApp, w: f32, h: f32) -> Harness<'static, PandaApp> {
    Harness::builder()
        .with_size(egui::vec2(w, h))
        .wgpu()
        .build_ui_state(|ui, app: &mut PandaApp| app.show(ui), app)
}

/// Step frames until `pred` holds (live values arrived, etc.).
fn step_until(h: &mut Harness<'_, PandaApp>, secs: f64, mut pred: impl FnMut(&PandaApp) -> bool) {
    let t0 = Instant::now();
    while !pred(h.state()) {
        assert!(t0.elapsed().as_secs_f64() < secs, "timed out");
        h.step();
        std::thread::sleep(Duration::from_millis(5));
    }
    h.step();
}

fn save(h: &mut Harness<'_, PandaApp>, name: &str) -> image::RgbaImage {
    let img = h.render().expect("render");
    let p = screenshot_path(name);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    img.save(&p).unwrap();
    img
}

/// Distinct colours in an image region (a crude "something was drawn" check).
fn colours(img: &image::RgbaImage) -> usize {
    let mut set = std::collections::HashSet::new();
    for p in img.pixels().step_by(7) {
        set.insert(p.0);
    }
    set.len()
}

#[test]
fn counter_top_level() {
    let mut state = AppState::new(examples::counter(), SimOptions::default());
    state.wait_layout();
    let mut h = harness(PandaApp::new(state), 1100.0, 640.0);
    step_until(&mut h, 10.0, |a| a.state.live.values().is_some());
    let s = &mut h.state_mut().state;
    s.sim.send(Command::SetInput {
        pin: "EN".into(),
        value: Value::from_bool(true),
    });
    s.sim.send(Command::Step {
        n: 11,
        unit: StepUnit::Cycle,
    });
    let tick0 = h.state().state.live.snapshot.tick;
    step_until(&mut h, 10.0, |a| {
        a.state.live.snapshot.tick > tick0 && a.state.live.values().is_some()
    });
    let img = save(&mut h, "counter-top");
    assert!(colours(&img) > 20);
    // Inside the counter package.
    assert!(h.state_mut().state.enter("dut"));
    h.state_mut().state.canvas.request_fit();
    step_until(&mut h, 10.0, |a| {
        a.state.live.values().is_some_and(|l| l.map.path == "dut")
    });
    save(&mut h, "counter-dut");
}

#[test]
fn panda8_top_level_and_core() {
    let mut state = AppState::new(examples::panda8(), SimOptions::default());
    state.wait_layout();
    let mut h = harness(PandaApp::new(state), 1400.0, 860.0);
    step_until(&mut h, 10.0, |a| a.state.live.values().is_some());
    {
        let s = &mut h.state_mut().state;
        s.sim.send(Command::SetMode(Mode::Fast));
        s.sim.send(Command::Run);
    }
    std::thread::sleep(Duration::from_millis(300));
    h.state_mut().state.sim.send(Command::Pause);
    step_until(&mut h, 10.0, |a| {
        !a.state.live.snapshot.running && a.state.live.snapshot.stats.cycles > 1000
    });
    let img = save(&mut h, "panda8-top");
    assert!(colours(&img) > 20);
    assert!(h.state_mut().state.enter("cpu"));
    h.state_mut().state.canvas.request_fit();
    step_until(&mut h, 10.0, |a| {
        a.state.live.values().is_some_and(|l| l.map.path == "cpu")
    });
    save(&mut h, "panda8-cpu");
    // A zoomed-in look at the ALU.
    assert!(h.state_mut().state.enter("cpu/alu"));
    h.state_mut().state.canvas.request_fit();
    step_until(&mut h, 10.0, |a| {
        a.state
            .live
            .values()
            .is_some_and(|l| l.map.path == "cpu/alu")
    });
    {
        let c = &mut h.state_mut().state.canvas;
        let center = egui::pos2(500.0, 400.0);
        c.xf.zoom_at(center, 3.0);
    }
    h.step();
    save(&mut h, "panda8-alu-zoom");
}
