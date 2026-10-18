//! Managing packages without the mouse and from the library, on the real app rendered
//! offscreen: rename a pin (F2), package the selection (Ctrl+G, type, Enter / Esc), rename a
//! package (Shift+F2 or the library's right-click menu) so every chip of it follows, rename a
//! package's pin so the chips' connections follow, and delete an unused package (blocked while
//! it is placed; Ctrl+Z restores it).
//!
//! Writes `docs/screenshots/pkg-*.png`.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use egui::{Event, Key, Modifiers};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use panda_netlist::{ComponentDef, Design, Endpoint, InstKind};
use panda_ui::app::{AppState, PandaApp};
use panda_ui::doc::{Document, ItemRef};
use panda_ui::edit::current_selection;
use panda_ui::simhost::SimOptions;

fn screenshot(h: &mut Harness<'_, PandaApp>, name: &str) {
    let img = h.render().expect("render");
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/screenshots")
        .join(format!("pkg-{name}.png"));
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    img.save(&p).unwrap();
}

/// An empty top def.
fn doc() -> Document {
    let design = Design {
        format: panda_netlist::FORMAT.into(),
        defs: vec![ComponentDef {
            id: "top".into(),
            name: "top".into(),
            version: 0,
            pins: vec![],
            instances: vec![],
            nets: vec![],
            editor: serde_json::Value::Null,
        }],
        top: "top".into(),
    };
    let mut doc = Document::new(design);
    doc.layout_missing();
    doc
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

/// Press and release `key` with `modifiers` (plus the text a real keyboard would type).
fn key(h: &mut Harness<'_, PandaApp>, key: Key, modifiers: Modifiers, text: Option<&str>) {
    h.event(Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers,
    });
    if let Some(t) = text {
        h.event(Event::Text(t.into()));
    }
    h.event(Event::Key {
        key,
        physical_key: None,
        pressed: false,
        repeat: false,
        modifiers,
    });
    h.step();
    h.step();
}

fn tap(h: &mut Harness<'_, PandaApp>, k: Key) {
    key(h, k, Modifiers::NONE, None);
}

fn type_text(h: &mut Harness<'_, PandaApp>, t: &str) {
    h.event(Event::Text(t.into()));
    h.step();
    h.step();
}

fn selected(h: &Harness<'_, PandaApp>) -> Vec<ItemRef> {
    current_selection(&h.ctx).items.into_iter().collect()
}

fn ends_of(h: &Harness<'_, PandaApp>, e: &Endpoint) -> Vec<Endpoint> {
    ends_in(h, "top", e)
}

fn ends_in(h: &Harness<'_, PandaApp>, def: &str, e: &Endpoint) -> Vec<Endpoint> {
    let def = h.state().state.doc.def(def).unwrap();
    let mut v = def
        .nets
        .iter()
        .find(|n| n.ends.contains(e))
        .map(|n| n.ends.clone())
        .unwrap_or_default();
    v.sort_by_key(|e| format!("{e:?}"));
    v
}

fn port(i: &str, p: &str) -> Endpoint {
    Endpoint::Port {
        inst: i.into(),
        port: p.into(),
    }
}

fn has_def(h: &Harness<'_, PandaApp>, id: &str) -> bool {
    h.state().state.doc.def(id).is_some()
}

fn insts_of(h: &Harness<'_, PandaApp>, def: &str) -> Vec<(String, String)> {
    h.state()
        .state
        .doc
        .design
        .defs
        .iter()
        .flat_map(|d| {
            d.instances
                .iter()
                .filter(|i| matches!(&i.kind, InstKind::Def(x) if x == def))
                .map(|i| (d.id.clone(), i.id.clone()))
        })
        .collect()
}

/// The library row of package `id` (the left panel's selectable button).
fn library_row<'a>(h: &'a Harness<'_, PandaApp>, id: &'a str) -> egui_kittest::Node<'a> {
    h.query_all_by_label(id)
        .find(|n| n.rect().left() < 220.0)
        .unwrap_or_else(|| panic!("library row {id}"))
}

#[test]
fn rename_package_and_delete_packages_from_the_keyboard_and_library() {
    let state = AppState::from_document(doc(), SimOptions::default());
    let mut h = Harness::builder()
        .with_size(egui::vec2(1100.0, 640.0))
        .wgpu()
        .build_ui_state(|ui, app: &mut PandaApp| app.show(ui), PandaApp::new(state));
    step_until(&mut h, 10.0, |a| a.state.live.values().is_some());

    // Build in1 → NAND (both inputs) → out1 from the keyboard (see keyboard.rs).
    for (q, want) in [
        ("in", ItemRef::Pin("in1".into())),
        ("nand", ItemRef::Instance("nand1".into())),
    ] {
        key(&mut h, Key::A, Modifiers::NONE, Some("a"));
        type_text(&mut h, q);
        tap(&mut h, Key::Enter);
        assert_eq!(selected(&h), [want]);
    }
    tap(&mut h, Key::C);
    tap(&mut h, Key::C);
    key(&mut h, Key::A, Modifiers::NONE, Some("a"));
    type_text(&mut h, "out");
    tap(&mut h, Key::Enter);
    tap(&mut h, Key::C);
    assert!(ends_of(&h, &Endpoint::Pin("out1".into())).contains(&port("nand1", "out")));

    // Tab to in1, F2: the pin's name is focused and selected; typing replaces it.
    tap(&mut h, Key::Tab);
    assert_eq!(selected(&h), [ItemRef::Pin("in1".into())]);
    tap(&mut h, Key::F2);
    type_text(&mut h, "a");
    screenshot(&mut h, "rename-pin");
    tap(&mut h, Key::Enter);
    let top = h.state().state.doc.def("top").unwrap();
    assert!(top.pins.iter().any(|p| p.name == "a") && top.pins.iter().all(|p| p.name != "in1"));
    assert!(ends_of(&h, &Endpoint::Pin("a".into())).contains(&port("nand1", "in0")));
    assert_eq!(
        selected(&h),
        [ItemRef::Pin("a".into())],
        "selection follows"
    );

    // Tab to the NAND, Ctrl+G: the name prompt opens focused with the default selected.
    tap(&mut h, Key::Tab);
    assert_eq!(selected(&h), [ItemRef::Instance("nand1".into())]);
    key(&mut h, Key::G, Modifiers::COMMAND, None);
    assert!(
        h.query_by_label("Create package from selection").is_some(),
        "prompt open"
    );
    screenshot(&mut h, "package-prompt");
    // Esc cancels.
    tap(&mut h, Key::Escape);
    assert!(
        h.query_by_label("Create package from selection").is_none(),
        "prompt closed"
    );
    assert_eq!(h.state().state.doc.design.defs.len(), 1);
    // Again: type a name (replacing the default) and Enter creates it.
    key(&mut h, Key::G, Modifiers::COMMAND, None);
    type_text(&mut h, "not");
    tap(&mut h, Key::Enter);
    assert!(has_def(&h, "not"), "package created from the keyboard");
    assert!(h.query_by_label("Create package from selection").is_none());
    let users = insts_of(&h, "not");
    assert_eq!(users.len(), 1);
    let chip = users[0].1.clone();

    // Shift+F2 with the package chip selected: rename the package.
    step_until(&mut h, 5.0, |a| {
        let _ = a;
        true
    });
    if selected(&h) != [ItemRef::Instance(chip.clone())] {
        // Select it from the keyboard.
        for _ in 0..4 {
            if selected(&h) == [ItemRef::Instance(chip.clone())] {
                break;
            }
            tap(&mut h, Key::Tab);
        }
    }
    assert_eq!(selected(&h), [ItemRef::Instance(chip.clone())]);
    key(&mut h, Key::F2, Modifiers::SHIFT, None);
    assert!(
        h.query_by_label("Rename package not").is_some(),
        "rename prompt open"
    );
    type_text(&mut h, "inverter");
    screenshot(&mut h, "rename-package");
    tap(&mut h, Key::Enter);
    assert!(
        has_def(&h, "inverter") && !has_def(&h, "not"),
        "package renamed"
    );
    assert_eq!(
        insts_of(&h, "inverter"),
        [("top".to_string(), chip.clone())]
    );

    // Enter opens it; rename one of its pins: the chip's connections in top follow.
    tap(&mut h, Key::Enter);
    step_until(&mut h, 5.0, |a| a.state.view.def == "inverter");
    tap(&mut h, Key::Tab);
    let pin = match selected(&h).as_slice() {
        [ItemRef::Pin(p)] => p.clone(),
        [ItemRef::Instance(_)] => {
            tap(&mut h, Key::Tab);
            match selected(&h).as_slice() {
                [ItemRef::Pin(p)] => p.clone(),
                other => panic!("expected a pin, got {other:?}"),
            }
        }
        other => panic!("expected a pin, got {other:?}"),
    };
    tap(&mut h, Key::F2);
    type_text(&mut h, "x");
    tap(&mut h, Key::Enter);
    let inv = h.state().state.doc.def("inverter").unwrap();
    assert!(inv.pins.iter().any(|p| p.name == "x"), "inner pin renamed");
    let top = h.state().state.doc.def("top").unwrap();
    let ports: Vec<&str> = top
        .nets
        .iter()
        .flat_map(|n| &n.ends)
        .filter_map(|e| match e {
            Endpoint::Port { inst, port } if *inst == chip => Some(port.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        ports.contains(&"x") && !ports.contains(&pin.as_str()),
        "{ports:?}"
    );
    key(&mut h, Key::ArrowUp, Modifiers::ALT, None);
    step_until(&mut h, 5.0, |a| a.state.view.def == "top");
    tap(&mut h, Key::Escape);
    screenshot(&mut h, "renamed-everywhere");

    // Library right-click: Delete is disabled while the chip is placed.
    library_row(&h, "inverter").click_secondary();
    h.step();
    h.step();
    assert!(h.query_by_label_contains("Rename…").is_some(), "menu open");
    screenshot(&mut h, "library-menu-in-use");
    tap(&mut h, Key::Escape);
    h.step();

    // Delete the chip, then Delete in the library removes the package; Ctrl+Z brings it back.
    for _ in 0..4 {
        if selected(&h) == [ItemRef::Instance(chip.clone())] {
            break;
        }
        tap(&mut h, Key::Tab);
    }
    tap(&mut h, Key::Delete);
    assert!(insts_of(&h, "inverter").is_empty(), "chip deleted");
    library_row(&h, "inverter").click_secondary();
    h.step();
    h.step();
    screenshot(&mut h, "library-menu-unused");
    h.get_by_label("Delete").click();
    h.step();
    h.step();
    assert!(!has_def(&h, "inverter"), "package deleted");
    screenshot(&mut h, "deleted");
    key(&mut h, Key::Z, Modifiers::COMMAND, None);
    assert!(has_def(&h, "inverter"), "undo restores the package");
}
