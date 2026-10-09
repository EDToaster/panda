//! Offscreen renders (egui_kittest + wgpu) of the hierarchy/files UI: breadcrumbs and badges,
//! the library palette, the instance tree, packaging and its prompt, the File menu.
//!
//! Writes `docs/screenshots/hier-*.png` (inspect them after visual changes).

use std::path::PathBuf;
use std::time::{Duration, Instant};

use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use panda_kernel::{Mode, StepUnit};
use panda_netlist::Value;
use panda_ui::app::{AppState, PandaApp, Registry};
use panda_ui::examples;
use panda_ui::files::RecentFiles;
use panda_ui::hier::classify::BadgeState;
use panda_ui::hier::ui::Shared;
use panda_ui::hier::{HierState, actions};
use panda_ui::simhost::{Command, SimOptions};

fn shot(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/screenshots")
        .join(format!("hier-{name}.png"))
}

/// The full app with an in-memory recent-files list.
fn app(state: AppState) -> (PandaApp, Shared) {
    let mut reg = Registry::default();
    panda_ui::edit::register(&mut reg);
    let hs = panda_ui::hier::register_with(
        &mut reg,
        HierState::new(RecentFiles {
            paths: vec!["/home/user/designs/alu.panda.json".into()],
            store: None,
        }),
    );
    panda_ui::simui::register(&mut reg);
    panda_ui::simux::register(&mut reg);
    (PandaApp::with_registry(state, reg), hs)
}

fn harness(app: PandaApp, w: f32, h: f32) -> Harness<'static, PandaApp> {
    Harness::builder()
        .with_size(egui::vec2(w, h))
        .wgpu()
        .build_ui_state(|ui, app: &mut PandaApp| app.show(ui), app)
}

fn step_until(h: &mut Harness<'_, PandaApp>, secs: f64, mut pred: impl FnMut(&PandaApp) -> bool) {
    let t0 = Instant::now();
    while !pred(h.state()) {
        assert!(t0.elapsed().as_secs_f64() < secs, "timed out");
        h.step();
        std::thread::sleep(Duration::from_millis(5));
    }
    h.step();
    h.step();
}

fn save(h: &mut Harness<'_, PandaApp>, name: &str) {
    let img = h.render().expect("render");
    let p = shot(name);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    img.save(&p).unwrap();
}

fn badge_ready(hs: &Shared, def: &str) -> bool {
    matches!(
        hs.borrow().classifier.peek(def),
        Some(BadgeState::Ready(_) | BadgeState::Failed(_))
    )
}

#[test]
fn panda8_breadcrumbs_badge_and_library() {
    let mut state = AppState::new(examples::panda8(), SimOptions::default());
    state.wait_layout();
    let (app, hs) = app(state);
    let mut h = harness(app, 1400.0, 860.0);
    step_until(&mut h, 20.0, |a| a.state.live.values().is_some());
    {
        let s = &mut h.state_mut().state;
        s.sim.send(Command::SetMode(Mode::Fast));
        s.sim.send(Command::Run);
    }
    std::thread::sleep(Duration::from_millis(300));
    h.state_mut().state.sim.send(Command::Pause);
    assert!(h.state_mut().state.enter("cpu/alu"));
    let def = h.state().state.view.def.clone();
    step_until(&mut h, 20.0, |a| {
        a.state.live.values().is_some_and(|l| l.map.path == "cpu/alu")
            && !a.state.live.snapshot.running
            && badge_ready(&hs, &def)
    });
    // Classify the user defs listed in the library too.
    step_until(&mut h, 30.0, |a| {
        a.state
            .doc
            .design
            .defs
            .iter()
            .filter(|d| d.id != a.state.doc.design.top)
            .all(|d| badge_ready(&hs, &d.id))
    });
    hs.borrow_mut().selected_entry = Some("ripple adder".into());
    h.step();
    save(&mut h, "panda8-alu-breadcrumbs");
    // The instance tree tab.
    h.get_by_label("Instances").click();
    h.step();
    h.step();
    save(&mut h, "panda8-instances");
    // Breadcrumb click goes back to `cpu`.
    // (The first "cpu" button is the breadcrumb, the second the instance tree row.)
    h.query_all_by_label("cpu").next().unwrap().click();
    step_until(&mut h, 10.0, |a| a.state.view.path == "cpu");
    assert_eq!(h.state().state.view.path, "cpu");
}

#[test]
fn package_counter_and_enter_it() {
    let mut state = AppState::new(examples::counter(), SimOptions::default());
    state.wait_layout();
    let (app, hs) = app(state);
    let mut h = harness(app, 1200.0, 700.0);
    step_until(&mut h, 20.0, |a| a.state.live.values().is_some());
    {
        let s = &mut h.state_mut().state;
        s.sim.send(Command::SetInput {
            pin: "EN".into(),
            value: Value::from_bool(true),
        });
        s.sim.send(Command::Step {
            n: 6,
            unit: StepUnit::Cycle,
        });
    }
    assert!(h.state_mut().state.enter("dut"));
    step_until(&mut h, 20.0, |a| {
        a.state.live.values().is_some_and(|l| l.map.path == "dut") && a.state.live.snapshot.tick > 50
    });
    // Select everything except the register, then open the package prompt.
    let sel: Vec<String> = {
        let s = &h.state().state;
        s.doc
            .def(&s.view.def)
            .unwrap()
            .instances
            .iter()
            .filter(|i| i.id != "reg")
            .map(|i| i.id.clone())
            .collect()
    };
    {
        let s = &mut h.state_mut().state;
        for id in &sel {
            s.highlights
                .items
                .insert(panda_ui::doc::ItemRef::Instance(id.clone()));
        }
    }
    h.step();
    save(&mut h, "counter-selection");
    {
        let mut b = hs.borrow_mut();
        b.package_selection = sel.clone();
        b.ask(actions::PromptKind::PackageName, "incr".into());
    }
    h.step();
    h.step();
    save(&mut h, "package-prompt");
    h.get_by_label("OK").click();
    step_until(&mut h, 20.0, |a| a.state.doc.def("incr").is_some());
    step_until(&mut h, 20.0, |a| {
        a.state.live.values().is_some_and(|l| {
            l.map.path == "dut" && l.map.nets.len() == a.state.doc.def(&a.state.view.def).unwrap().nets.len()
        })
    });
    save(&mut h, "counter-packaged");
    assert!(actions::enter_instance(&mut h.state_mut().state, "incr"));
    step_until(&mut h, 20.0, |a| {
        a.state.live.values().is_some_and(|l| l.map.path == "dut/incr") && badge_ready(&hs, "incr")
    });
    save(&mut h, "inside-new-package");
}

#[test]
fn file_menu_and_examples() {
    let mut state = AppState::new(examples::sr_latch(), SimOptions::default());
    state.wait_layout();
    let (app, hs) = app(state);
    let mut h = harness(app, 1000.0, 600.0);
    step_until(&mut h, 20.0, |a| a.state.live.values().is_some());
    h.get_by_label("File").click();
    h.step();
    h.step();
    h.get_by_label_contains("Examples").click();
    h.step();
    h.step();
    save(&mut h, "file-menu");
    h.get_by_label("Register file").click();
    step_until(&mut h, 20.0, |a| {
        a.state.doc.design.top.starts_with("tb_regfile") && a.state.live.values().is_some()
    });
    // Badges are computed for the new document (not stale ones of the previous document).
    step_until(&mut h, 30.0, |a| {
        a.state.doc.design.defs.iter().all(|d| badge_ready(&hs, &d.id))
    });
    for d in &h.state().state.doc.design.defs {
        assert!(
            matches!(hs.borrow().classifier.peek(&d.id), Some(BadgeState::Ready(_))),
            "{}: {:?}",
            d.id,
            hs.borrow().classifier.peek(&d.id)
        );
    }
    save(&mut h, "example-register-file");
}

#[test]
fn library_block_click_then_canvas_click_places_it() {
    let mut state = AppState::new(examples::counter(), SimOptions::default());
    state.wait_layout();
    let (app, hs) = app(state);
    let mut h = harness(app, 1200.0, 700.0);
    step_until(&mut h, 20.0, |a| a.state.live.values().is_some());
    hs.borrow_mut().filter = "adder".into();
    h.step();
    h.get_by_label("ripple adder").click();
    h.step();
    h.step();
    // The block's defs were added (undoable), the edit module's place tool is armed.
    assert!(h.state().state.doc.def("ripple_adder_8").is_some());
    assert!(h.state().state.doc.def("full_adder").is_some());
    let top = h.state().state.doc.design.top.clone();
    let before = h.state().state.doc.def(&top).unwrap().instances.len();
    // Move over an empty part of the canvas and click.
    let p = egui::pos2(700.0, 520.0);
    h.event(egui::Event::PointerMoved(p));
    h.step();
    save(&mut h, "library-place-ghost");
    for pressed in [true, false] {
        h.event(egui::Event::PointerButton {
            pos: p,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        });
        h.step();
    }
    h.step();
    let def = h.state().state.doc.def(&top).unwrap().clone();
    assert_eq!(def.instances.len(), before + 1, "nothing placed");
    assert!(def
        .instances
        .iter()
        .any(|i| matches!(&i.kind, panda_netlist::InstKind::Def(d) if d == "ripple_adder_8")));
    save(&mut h, "library-placed");
}
