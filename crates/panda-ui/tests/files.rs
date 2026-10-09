//! Files: save → open round trips (structure + geometry), the unsaved-changes flow, recent
//! files, ROM import and the built-in examples (load, lay out, simulate). Paths are injected
//! with [`ScriptedPicker`] (no native dialogs). The last tests render the File menu offscreen
//! (`docs/screenshots/files-*.png`).

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use panda_kernel::{Kernel, Mode, StepUnit};
use panda_netlist::Value;
use panda_ui::app::{AppState, PandaApp, Registry};
use panda_ui::doc::{Document, Label};
use panda_ui::examples;
use panda_ui::files::dialog::{PickPurpose, ScriptedPicker};
use panda_ui::files::menu::{FileMenu, load_recent, store_recent};
use panda_ui::files::session::{Confirm, FileAction, FileSession};
use panda_ui::files::{self, EXTENSION, MAX_RECENT, RecentFiles};
use panda_ui::geom::gp;
use panda_ui::simhost::{Command, SimOptions};

fn tmp_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("panda-files-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn session() -> (FileSession, ScriptedPicker) {
    let p = ScriptedPicker::default();
    (FileSession::new(Box::new(p.clone())), p)
}

fn example(name: &str) -> (usize, examples::Example) {
    examples::all()
        .into_iter()
        .enumerate()
        .find(|(_, e)| e.name == name)
        .unwrap_or_else(|| panic!("no example {name}"))
}

/// Laid-out document of an example with some hand-made geometry on top.
fn edited_doc(design: panda_netlist::Design) -> Document {
    let mut doc = Document::new(design);
    doc.layout_missing();
    let top = doc.design.top.clone();
    let ed = doc.editors.get_mut(&top).unwrap();
    ed.auto = false;
    if let Some(g) = ed.instances.values_mut().next() {
        g.pos = gp(g.pos.x + 3, g.pos.y - 2);
        g.orient.rot = (g.orient.rot + 1) % 4;
        g.orient.flip = !g.orient.flip;
    }
    ed.labels.push(Label {
        pos: gp(-4, -4),
        text: "hand-made label".into(),
    });
    doc.touch();
    doc
}

#[test]
fn save_open_round_trip_is_identical() {
    let dir = tmp_dir("rt");
    for e in examples::all().iter().take(4) {
        let mut doc = edited_doc((e.build)());
        let p = dir.join(format!("{}{EXTENSION}", e.name.replace([' ', ':'], "_")));
        files::write_document(&mut doc, &p).unwrap();
        assert!(!doc.dirty);
        let back = files::read_document(&p).unwrap();
        assert_eq!(back.design.top, doc.design.top);
        assert_eq!(back.to_design(), doc.to_design(), "{}: structure", e.name);
        assert_eq!(back.editors, doc.editors, "{}: geometry", e.name);
        assert!(back.defs_missing_layout().is_empty(), "{}", e.name);
        assert!(!back.dirty);
        // Saving again gives the same bytes.
        assert_eq!(back.to_json(), std::fs::read_to_string(&p).unwrap());
        // The kernel accepts the saved file (geometry is ignored by it).
        let mut k = Kernel::default();
        k.load(back.to_design()).unwrap();
    }
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn open_rejects_garbage_and_forgets_missing_recent() {
    let dir = tmp_dir("bad");
    let mut app = AppState::new(examples::sr_latch(), SimOptions::default());
    app.wait_layout();
    let (mut s, _) = session();
    let bad = dir.join("bad.panda.json");
    std::fs::write(&bad, "{ not json").unwrap();
    s.perform(&mut app, FileAction::Open(bad.clone()));
    assert!(s.error.take().is_some());
    assert_eq!(
        app.doc.design.top,
        examples::sr_latch().top,
        "document kept"
    );
    let gone = dir.join("gone.panda.json");
    s.recent = RecentFiles::with_paths([gone.clone()]);
    s.perform(&mut app, FileAction::Open(gone));
    assert!(s.error.is_some() && s.recent.paths.is_empty());
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn session_save_as_open_and_dirty_prompt() {
    let dir = tmp_dir("session");
    let mut app = AppState::new(examples::counter(), SimOptions::default());
    app.wait_layout();
    let (mut s, picker) = session();
    assert!(!app.doc.dirty, "auto-layout does not dirty the document");

    // Save without a path → Save as (extension appended).
    app.doc.touch();
    assert!(FileSession::title(&app).ends_with('*'));
    assert!(!s.save(&mut app));
    assert_eq!(
        picker.requests.borrow().last().unwrap().purpose,
        PickPurpose::SaveAs
    );
    picker.push(Some(dir.join("counter")));
    assert!(s.poll(&mut app));
    let saved = dir.join(format!("counter{EXTENSION}"));
    assert_eq!(app.doc.path.as_deref(), Some(saved.as_path()));
    assert!(!app.doc.dirty && saved.exists());
    assert!(!FileSession::title(&app).ends_with('*'));
    assert!(s.recent.paths[0].ends_with(format!("counter{EXTENSION}")));
    let geometry = app.doc.editors.clone();

    // Dirty + New → prompt; Cancel keeps everything.
    app.doc.touch();
    s.request(&mut app, FileAction::New);
    assert_eq!(s.confirm, Some(FileAction::New));
    s.answer(&mut app, Confirm::Cancel);
    assert!(s.confirm.is_none() && app.doc.dirty);
    assert_eq!(app.doc.path.as_deref(), Some(saved.as_path()));

    // Save (has a path) → written, then the action runs.
    s.request(&mut app, FileAction::New);
    s.answer(&mut app, Confirm::Save);
    assert_eq!(app.doc.design, files::new_design());
    assert!(app.doc.path.is_none() && !app.doc.dirty);

    // Open the saved file through the dialog: identical structure and geometry.
    s.open_dialog(&app);
    picker.push(Some(saved.clone()));
    s.poll(&mut app);
    assert_eq!(app.doc.design.top, examples::counter().top);
    assert_eq!(app.doc.editors, geometry);
    assert_eq!(app.doc.path.as_deref(), Some(saved.as_path()));

    // Dirty + example, Discard → example opens, file untouched.
    let before = std::fs::read_to_string(&saved).unwrap();
    app.doc.touch();
    let (i, _) = example("SR latch");
    s.request(&mut app, FileAction::Example(i));
    s.answer(&mut app, Confirm::Discard);
    assert_eq!(app.doc.design, examples::sr_latch());
    assert_eq!(std::fs::read_to_string(&saved).unwrap(), before);

    // Dirty, no path, Save → Save as; cancelling the dialog drops the pending action.
    app.wait_layout();
    app.doc.touch();
    s.request(&mut app, FileAction::Quit);
    s.answer(&mut app, Confirm::Save);
    picker.push(None);
    s.poll(&mut app);
    assert!(!s.close_now && app.doc.dirty);
    // … and choosing a path saves, then quits.
    s.request(&mut app, FileAction::Quit);
    s.answer(&mut app, Confirm::Save);
    picker.push(Some(dir.join("latch.json")));
    s.poll(&mut app);
    assert!(s.close_now && !app.doc.dirty && dir.join("latch.json").exists());
    assert_eq!(s.recent.paths.len(), 2);
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn recent_files_dedup_cap_and_persist_in_egui_memory() {
    let dir = tmp_dir("recent");
    let mut r = RecentFiles::in_memory();
    for i in 0..14 {
        r.push(&dir.join(format!("f{i}.panda.json")));
    }
    r.push(&dir.join("f3.panda.json"));
    assert_eq!(r.paths.len(), MAX_RECENT);
    assert!(r.paths[0].ends_with("f3.panda.json"));
    assert_eq!(
        r.paths
            .iter()
            .filter(|p| p.ends_with("f3.panda.json"))
            .count(),
        1
    );
    let ctx = egui::Context::default();
    assert!(load_recent(&ctx).paths.is_empty());
    store_recent(&ctx, &r);
    assert_eq!(load_recent(&ctx).paths, r.paths);
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn rom_import_adds_def_kernel_and_layout() {
    let dir = tmp_dir("rom");
    let hex = dir.join("01-prog.hex");
    std::fs::write(&hex, "// image\n12 34 @4 ab\n0x7f\n").unwrap();
    let mut app = AppState::new(files::new_design(), SimOptions::default());
    app.wait_layout();
    let (mut s, picker) = session();
    s.import_rom_dialog(&app);
    picker.push(Some(hex.clone()));
    s.poll(&mut app);
    assert!(s.error.is_none(), "{:?}", s.error);
    let def = s.pending_place.clone().expect("placement requested");
    assert!(def.starts_with("rom_rom_01_prog_3x8_"), "{def}");
    assert!(app.doc.def(&def).is_some() && app.doc.dirty);
    s.wait_rom_layout(&mut app);
    assert!(app.doc.editor(&def).is_some(), "ROM def laid out");
    // Same file again: same content-addressed def, nothing new added.
    let n = app.doc.design.defs.len();
    assert_eq!(s.import_rom_file(&mut app, &hex).unwrap(), def);
    assert_eq!(app.doc.design.defs.len(), n);
    // Contents, read through a kernel with the ROM as the top.
    let mut d = app.doc.to_design();
    d.top = def.clone();
    let mut k = Kernel::default();
    k.load(d).unwrap();
    for (a, want) in [
        (0u64, 0x12u64),
        (1, 0x34),
        (2, 0),
        (3, 0),
        (4, 0xab),
        (5, 0x7f),
        (7, 0),
    ] {
        k.set_input("A", &Value::from_u64(3, a)).unwrap();
        k.step(200, StepUnit::Tick);
        assert_eq!(k.read_u64("D"), Some(want), "A={a}");
    }
    // Errors are reported, nothing changes.
    std::fs::write(&hex, "zz").unwrap();
    s.import_rom_dialog(&app);
    picker.push(Some(hex));
    s.poll(&mut app);
    assert!(s.error.is_some());
    assert_eq!(app.doc.design.defs.len(), n);
    std::fs::remove_dir_all(dir).ok();
}

fn settle(k: &mut Kernel) {
    k.step(64, StepUnit::Tick);
}

#[test]
fn examples_load_lay_out_and_simulate() {
    let all = examples::all();
    assert_eq!(all.len(), 3 + panda_cpu8::programs::ALL.len());
    for e in &all {
        let t0 = Instant::now();
        let mut doc = Document::new((e.build)());
        doc.layout_missing();
        assert!(doc.defs_missing_layout().is_empty(), "{}", e.name);
        let mut k = Kernel::default();
        k.load(doc.to_design())
            .unwrap_or_else(|d| panic!("{}: {d:?}", e.name));
        eprintln!(
            "{}: {} defs, laid out + loaded in {:?}",
            e.name,
            doc.design.defs.len(),
            t0.elapsed()
        );
    }

    // SR latch: set, hold, reset.
    let mut k = Kernel::default();
    k.load(examples::sr_latch()).unwrap();
    let one = Value::from_bool(true);
    let zero = Value::from_bool(false);
    k.set_input("S_n", &zero).unwrap();
    k.set_input("R_n", &one).unwrap();
    settle(&mut k);
    assert_eq!((k.read_u64("Q"), k.read_u64("Q_n")), (Some(1), Some(0)));
    k.set_input("S_n", &one).unwrap();
    settle(&mut k);
    assert_eq!(k.read_u64("Q"), Some(1), "holds");
    k.set_input("R_n", &zero).unwrap();
    settle(&mut k);
    assert_eq!((k.read_u64("Q"), k.read_u64("Q_n")), (Some(0), Some(1)));

    // Counter: counts while enabled.
    let mut k = Kernel::default();
    k.load(examples::counter()).unwrap();
    k.set_input("EN", &one).unwrap();
    k.step(4, StepUnit::Cycle);
    let a = k.read_u64("Q").expect("Q known after reset");
    k.step(5, StepUnit::Cycle);
    assert_eq!(k.read_u64("Q"), Some((a + 5) % 16));

    // Register file: write 0xa5 to r2 and 0x3c to r1, read both back.
    let mut k = Kernel::default();
    k.load(examples::register_file_4x8()).unwrap();
    k.step(4, StepUnit::Cycle);
    for (r, v) in [(2u64, 0xa5u64), (1, 0x3c)] {
        k.set_input("WA", &Value::from_u64(2, r)).unwrap();
        k.set_input("WD", &Value::from_u64(8, v)).unwrap();
        k.set_input("WE", &one).unwrap();
        k.step(1, StepUnit::Cycle);
    }
    k.set_input("WE", &zero).unwrap();
    for (r, v) in [(2u64, 0xa5u64), (1, 0x3c)] {
        k.set_input("RA", &Value::from_u64(2, r)).unwrap();
        k.step(1, StepUnit::Cycle);
        assert_eq!(k.read_u64("RD"), Some(v), "r{r}");
    }

    // Panda-8: every halting program halts (Fast mode); bench_loop keeps running.
    for p in panda_cpu8::programs::ALL {
        let mut k = Kernel::default();
        k.load(examples::panda8_program(&p)).unwrap();
        k.set_mode(Mode::Fast);
        let mut cycles = 0;
        while k.read_u64("HALTED") != Some(1) && cycles < 4000 {
            k.step(50, StepUnit::Cycle);
            cycles += 50;
        }
        assert_eq!(
            k.read_u64("HALTED") == Some(1),
            p.halts,
            "{}: halted after {cycles}",
            p.name
        );
        assert!(k.read_u64("PC").is_some(), "{}", p.name);
    }
}

// ---------------------------------------------------------------- offscreen UI

fn screenshot(h: &mut Harness<'_, PandaApp>, name: &str) {
    let img = h.render().expect("render");
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/screenshots")
        .join(format!("files-{name}.png"));
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    img.save(&p).unwrap();
}

fn step_until(h: &mut Harness<'_, PandaApp>, secs: f64, mut pred: impl FnMut(&PandaApp) -> bool) {
    let t0 = Instant::now();
    while !pred(h.state()) {
        assert!(t0.elapsed().as_secs_f64() < secs, "timed out");
        h.step();
        std::thread::sleep(Duration::from_millis(5));
    }
    h.step();
}

/// The full app, but with this module's File menu driven by a scripted picker.
fn app_with_files(state: AppState, picker: ScriptedPicker) -> PandaApp {
    let mut reg = Registry::default();
    panda_ui::app::panel::register_all(&mut reg);
    reg.panels
        .retain(|p| p.id() != "files.menu" && p.title() != "File");
    reg.panels.insert(
        0,
        Box::new(FileMenu::new(FileSession::new(Box::new(picker)))),
    );
    PandaApp::with_registry(state, reg)
}

fn harness(app: PandaApp) -> Harness<'static, PandaApp> {
    Harness::builder()
        .with_size(egui::vec2(1200.0, 760.0))
        .wgpu()
        .build_ui_state(|ui, app: &mut PandaApp| app.show(ui), app)
}

#[test]
fn examples_menu_opens_panda8_and_prompts_when_dirty() {
    let mut state = AppState::new(examples::sr_latch(), SimOptions::default());
    state.wait_layout();
    let mut h = harness(app_with_files(state, ScriptedPicker::default()));
    step_until(&mut h, 10.0, |a| a.state.live.values().is_some());
    h.get_by_label("File").click();
    h.run_steps(3);
    h.get_by_label_contains("Examples").hover();
    h.run_steps(3);
    screenshot(&mut h, "examples-menu");
    h.get_by_label("Panda-8: fibonacci").click();
    h.run_steps(2);
    step_until(&mut h, 60.0, |a| {
        !a.state.layout_pending()
            && a.state.live.values().is_some()
            && a.state.doc.design.top.contains("panda8")
    });
    h.state_mut().state.sim.send(Command::Step {
        n: 120,
        unit: StepUnit::Cycle,
    });
    step_until(&mut h, 60.0, |a| {
        a.state.live.snapshot.tick > 10_000 && !a.state.live.snapshot.running
    });
    h.state_mut().state.canvas.request_fit();
    h.run_steps(3);
    assert!(!h.state().state.doc.dirty);
    screenshot(&mut h, "panda8-opened");

    // Dirty document + New (Ctrl+N) → unsaved-changes modal; "Don't save" proceeds.
    h.state_mut().state.doc.touch();
    h.key_press_modifiers(egui::Modifiers::COMMAND, egui::Key::N);
    h.run_steps(3);
    h.get_by_label("Unsaved changes");
    screenshot(&mut h, "unsaved-prompt");
    h.get_by_label("Don't save").click();
    h.run_steps(3);
    assert_eq!(h.state().state.doc.design, files::new_design());
}

#[test]
fn save_shortcut_writes_through_scripted_dialog() {
    let dir = tmp_dir("ui-save");
    let picker = ScriptedPicker::default();
    let mut state = AppState::new(examples::counter(), SimOptions::default());
    state.wait_layout();
    let mut h = harness(app_with_files(state, picker.clone()));
    h.run_steps(3);
    picker.push(Some(dir.join("ui")));
    h.key_press_modifiers(egui::Modifiers::COMMAND, egui::Key::S);
    h.run_steps(3);
    let p = dir.join(format!("ui{EXTENSION}"));
    assert!(Path::new(&p).exists());
    let back = files::read_document(&p).unwrap();
    assert_eq!(back.editors, h.state().state.doc.editors);
    std::fs::remove_dir_all(dir).ok();
}
