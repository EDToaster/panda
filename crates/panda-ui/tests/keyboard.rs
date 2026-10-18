//! Keyboard-only editing on the real app, rendered offscreen (egui_kittest + wgpu): build a NOT
//! gate out of one NAND from an empty design without touching the mouse (A quick add, C connect,
//! Tab select, N select wire, F2 rename, Space toggle), with the contextual hint bar and the `?`
//! shortcut sheet.
//!
//! Writes `docs/screenshots/keys-*.png`.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use egui::{Event, Key, Modifiers};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use panda_netlist::{ComponentDef, Design, Endpoint};
use panda_ui::app::{AppState, PandaApp};
use panda_ui::doc::{Document, ItemRef};
use panda_ui::edit::current_selection;
use panda_ui::simhost::SimOptions;

fn screenshot(h: &mut Harness<'_, PandaApp>, name: &str) {
    let img = h.render().expect("render");
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/screenshots")
        .join(format!("keys-{name}.png"));
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
    let def = h.state().state.doc.def("top").unwrap();
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

#[test]
fn build_a_not_gate_without_the_mouse() {
    let state = AppState::from_document(doc(), SimOptions::default());
    let mut h = Harness::builder()
        .with_size(egui::vec2(1100.0, 640.0))
        .wgpu()
        .build_ui_state(|ui, app: &mut PandaApp| app.show(ui), PandaApp::new(state));
    step_until(&mut h, 10.0, |a| a.state.live.values().is_some());
    h.step();
    screenshot(&mut h, "empty-hints");

    // ? opens the shortcut sheet, Esc closes it.
    key(&mut h, Key::Slash, Modifiers::SHIFT, Some("?"));
    assert!(
        h.query_by_label("Keyboard shortcuts").is_some(),
        "sheet open"
    );
    screenshot(&mut h, "sheet");
    tap(&mut h, Key::Escape);
    assert!(
        h.query_by_label("Keyboard shortcuts").is_none(),
        "sheet closed"
    );

    // A, "in", Enter: an input pin at the view centre.
    key(&mut h, Key::A, Modifiers::NONE, Some("a"));
    type_text(&mut h, "in");
    screenshot(&mut h, "quick-add");
    tap(&mut h, Key::Enter);
    assert_eq!(selected(&h), [ItemRef::Pin("in1".into())]);

    // A, "nand", Enter: a NAND to the right of the pin, selected.
    key(&mut h, Key::A, Modifiers::NONE, Some("a"));
    type_text(&mut h, "nand");
    tap(&mut h, Key::Enter);
    assert_eq!(selected(&h), [ItemRef::Instance("nand1".into())]);

    // C twice: in1 drives both NAND inputs.
    tap(&mut h, Key::C);
    tap(&mut h, Key::C);
    assert_eq!(
        ends_of(&h, &Endpoint::Pin("in1".into())),
        {
            let mut v = vec![
                Endpoint::Pin("in1".into()),
                port("nand1", "in0"),
                port("nand1", "in1"),
            ];
            v.sort_by_key(|e| format!("{e:?}"));
            v
        },
        "C wired the previous item to the next free inputs"
    );
    screenshot(&mut h, "connect");

    // A, "out", Enter, C: the NAND output drives a new output pin.
    key(&mut h, Key::A, Modifiers::NONE, Some("a"));
    type_text(&mut h, "out");
    tap(&mut h, Key::Enter);
    assert_eq!(selected(&h), [ItemRef::Pin("out1".into())]);
    tap(&mut h, Key::C);
    assert!(
        ends_of(&h, &Endpoint::Pin("out1".into())).contains(&port("nand1", "out")),
        "nand1.out → out1"
    );

    // Tab wraps from the rightmost item (out1) to the leftmost (in1).
    tap(&mut h, Key::Tab);
    assert_eq!(selected(&h), [ItemRef::Pin("in1".into())]);
    // Shift+Tab goes back.
    key(&mut h, Key::Tab, Modifiers::SHIFT, None);
    assert_eq!(selected(&h), [ItemRef::Pin("out1".into())]);
    tap(&mut h, Key::Tab);

    // Space toggles the selected input (in1 = 1) and F10 steps: out1 = 0.
    tap(&mut h, Key::Space);
    // F10 steps the (paused) simulation.
    for _ in 0..4 {
        tap(&mut h, Key::F10);
    }
    step_until(&mut h, 10.0, |a| {
        a.state.live.values().is_some_and(|l| {
            l.map.pin_value("in1", l.values).and_then(|v| v.to_u64()) == Some(1)
                && l.map.pin_value("out1", l.values).and_then(|v| v.to_u64()) == Some(0)
        })
    });
    screenshot(&mut h, "not-built");

    // N selects in1's wire, F2 focuses its name with the text selected, typing replaces it.
    tap(&mut h, Key::N);
    let net = current_selection(&h.ctx)
        .single_net()
        .map(str::to_string)
        .expect("a net selected");
    tap(&mut h, Key::F2);
    type_text(&mut h, "A");
    screenshot(&mut h, "rename");
    tap(&mut h, Key::Enter);
    let def = h.state().state.doc.def("top").unwrap();
    assert!(def.nets.iter().any(|n| n.id == "A"), "{net} renamed to A");
    assert!(def.nets.iter().all(|n| n.id != net));
}
