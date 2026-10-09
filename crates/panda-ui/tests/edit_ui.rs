//! Editing interactions on the real app, rendered offscreen (egui_kittest + wgpu): drag an
//! item onto the canvas, draw wires between pins, live values, Shift-click selection toggling,
//! rotate + undo, a width-mismatch wire, box selection.
//!
//! Writes `docs/screenshots/edit-*.png`.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use egui::{Event, Key, Modifiers, PointerButton, Pos2, pos2};
use egui_kittest::Harness;
use panda_kernel::StepUnit;
use panda_netlist::{Atom, ComponentDef, Design, Dir, Endpoint, PinDecl, Value};
use panda_ui::app::{AppState, PandaApp};
use panda_ui::canvas::ViewXf;
use panda_ui::doc::{Document, InstGeom, ItemRef};
use panda_ui::edit::{PlaceRequest, current_selection, request_place};
use panda_ui::geom::{GridPt, gp};
use panda_ui::simhost::{Command, SimOptions};

fn screenshot(h: &mut Harness<'_, PandaApp>, name: &str) {
    let img = h.render().expect("render");
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/screenshots")
        .join(format!("edit-{name}.png"));
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    img.save(&p).unwrap();
}

/// A top def with input `a` at (0, 0) and output `y` at (24, 0), nothing else.
fn doc() -> Document {
    let design = Design {
        format: panda_netlist::FORMAT.into(),
        defs: vec![ComponentDef {
            id: "top".into(),
            name: "top".into(),
            version: 0,
            pins: vec![
                PinDecl {
                    name: "a".into(),
                    dir: Dir::In,
                    width: 1,
                },
                PinDecl {
                    name: "y".into(),
                    dir: Dir::Out,
                    width: 1,
                },
            ],
            instances: vec![],
            nets: vec![],
            editor: serde_json::Value::Null,
        }],
        top: "top".into(),
    };
    let mut doc = Document::new(design);
    doc.layout_missing();
    let e = doc.editors.get_mut("top").unwrap();
    e.pins.insert(
        "a".into(),
        InstGeom {
            pos: gp(0, 0),
            ..Default::default()
        },
    );
    e.pins.insert(
        "y".into(),
        InstGeom {
            pos: gp(24, 0),
            ..Default::default()
        },
    );
    doc
}

const XF: ViewXf = ViewXf {
    origin: Pos2 { x: 300.0, y: 250.0 },
    zoom: 20.0,
};

fn sp(p: GridPt) -> Pos2 {
    XF.grid_to_screen(p)
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

/// Modifiers currently held (as egui sees them).
fn mods(h: &Harness<'_, PandaApp>) -> Modifiers {
    h.ctx.input(|i| i.modifiers)
}

fn set_mods(h: &mut Harness<'_, PandaApp>, m: Modifiers) {
    h.event(Event::ModifiersChanged(m));
    h.step();
}

fn press(h: &mut Harness<'_, PandaApp>, p: Pos2) {
    let modifiers = mods(h);
    h.event(Event::PointerMoved(p));
    h.event(Event::PointerButton {
        pos: p,
        button: PointerButton::Primary,
        pressed: true,
        modifiers,
    });
    h.step();
}

fn move_to(h: &mut Harness<'_, PandaApp>, from: Pos2, to: Pos2) {
    for k in 1..=6 {
        let t = k as f32 / 6.0;
        h.event(Event::PointerMoved(from + (to - from) * t));
        h.step();
    }
}

fn release(h: &mut Harness<'_, PandaApp>, p: Pos2) {
    let modifiers = mods(h);
    h.event(Event::PointerButton {
        pos: p,
        button: PointerButton::Primary,
        pressed: false,
        modifiers,
    });
    h.step();
    h.step();
}

fn drag(h: &mut Harness<'_, PandaApp>, from: Pos2, to: Pos2) {
    press(h, from);
    move_to(h, from, to);
    release(h, to);
}

fn click(h: &mut Harness<'_, PandaApp>, p: Pos2) {
    press(h, p);
    release(h, p);
}

fn net_ends(app: &PandaApp, e: &Endpoint) -> Option<Vec<Endpoint>> {
    app.state
        .doc
        .def("top")?
        .nets
        .iter()
        .find(|n| n.ends.contains(e))
        .map(|n| n.ends.clone())
}

fn pin(n: &str) -> Endpoint {
    Endpoint::Pin(n.into())
}

fn port(i: &str, p: &str) -> Endpoint {
    Endpoint::Port {
        inst: i.into(),
        port: p.into(),
    }
}

#[test]
fn place_wire_toggle_rotate_and_errors() {
    let state = AppState::from_document(doc(), SimOptions::default());
    let mut h = Harness::builder()
        .with_size(egui::vec2(1100.0, 640.0))
        .wgpu()
        .build_ui_state(|ui, app: &mut PandaApp| app.show(ui), PandaApp::new(state));
    step_until(&mut h, 10.0, |a| a.state.live.values().is_some());
    h.state_mut().state.canvas.xf = XF;
    h.step();

    // 1. Drag a NOT from outside the canvas (the status bar) onto it.
    request_place(&h.ctx, PlaceRequest::Atom(Atom::Not { width: 1 }));
    let start = pos2(560.0, 630.0);
    let target = sp(gp(14, 1));
    press(&mut h, start);
    move_to(&mut h, start, target);
    screenshot(&mut h, "place-drag");
    release(&mut h, target);
    let g = h.state().state.doc.editor("top").unwrap().instances["not1"];
    assert_eq!(g.pos, gp(12, 0), "placed centred under the drop point");

    // 2. Draw wires a → not1.in and not1.out → y by dragging between the pins.
    let (a_port, not_in, not_out, y_port) = (gp(4, 1), gp(12, 1), gp(16, 1), gp(24, 1));
    press(&mut h, sp(a_port));
    move_to(&mut h, sp(a_port), sp(not_in));
    screenshot(&mut h, "wire-drag");
    release(&mut h, sp(not_in));
    assert_eq!(
        net_ends(h.state(), &pin("a")),
        Some(vec![pin("a"), port("not1", "in")])
    );
    drag(&mut h, sp(not_out), sp(y_port));
    assert_eq!(
        net_ends(h.state(), &pin("y")),
        Some(vec![port("not1", "out"), pin("y")])
    );

    // 3. The kernel got the edits: drive a = 1 and see y = 0 live.
    {
        let s = &mut h.state_mut().state;
        s.sim.send(Command::SetInput {
            pin: "a".into(),
            value: Value::from_bool(true),
        });
        s.sim.send(Command::Step {
            n: 10,
            unit: StepUnit::Tick,
        });
    }
    step_until(&mut h, 10.0, |a| {
        a.state.live.values().is_some_and(|l| {
            l.map.pin_value("y", l.values).and_then(|v| v.to_u64()) == Some(0)
                && l.map.pin_value("a", l.values).and_then(|v| v.to_u64()) == Some(1)
        })
    });
    h.event(Event::PointerGone);
    h.step();
    screenshot(&mut h, "wired-live");

    // 4. Click selects; Shift-click toggles.
    click(&mut h, sp(gp(14, 1)));
    let sel = current_selection(&h.ctx);
    assert_eq!(
        sel.items.iter().cloned().collect::<Vec<_>>(),
        vec![ItemRef::Instance("not1".into())]
    );
    set_mods(&mut h, Modifiers::SHIFT);
    click(&mut h, sp(gp(26, 1)));
    assert_eq!(current_selection(&h.ctx).items.len(), 2);
    click(&mut h, sp(gp(26, 1)));
    set_mods(&mut h, Modifiers::NONE);
    assert_eq!(current_selection(&h.ctx).items.len(), 1);

    // 5. R rotates the selected NOT (the wires follow), Ctrl+Z undoes it.
    h.event(Event::PointerMoved(sp(gp(14, 6))));
    h.key_press(Key::R);
    h.step();
    let g = h.state().state.doc.editor("top").unwrap().instances["not1"];
    assert_eq!(g.orient.rot, 1);
    screenshot(&mut h, "rotated-selected");
    h.key_press_modifiers(Modifiers::COMMAND, Key::Z);
    h.step();
    let g = h.state().state.doc.editor("top").unwrap().instances["not1"];
    assert_eq!(g.orient.rot, 0);

    // 6. An 8-bit constant wired to the 1-bit net: red preview with the reason, then refused.
    request_place(
        &h.ctx,
        PlaceRequest::Atom(Atom::Constant {
            value: Value::from_u64(8, 0x2a),
        }),
    );
    h.step();
    click(&mut h, sp(gp(8, 8)));
    let c = h.state().state.doc.editor("top").unwrap().instances["const1"];
    let sym = panda_ui::geom::atom_symbol(&Atom::Constant {
        value: Value::from_u64(8, 0x2a),
    });
    let c_out = c.pos + sym.ports[0].pos;
    let nets_before = h.state().state.doc.def("top").unwrap().nets.len();
    press(&mut h, sp(c_out));
    move_to(&mut h, sp(c_out), sp(gp(20, 1)));
    screenshot(&mut h, "width-error");
    release(&mut h, sp(gp(20, 1)));
    assert_eq!(
        h.state().state.doc.def("top").unwrap().nets.len(),
        nets_before
    );
    h.event(Event::PointerGone);
    h.step();
    screenshot(&mut h, "error-toast");

    // 7. Box selection from empty space.
    let (b0, b1) = (sp(gp(-1, -3)), sp(gp(19, 4)));
    press(&mut h, b0);
    move_to(&mut h, b0, b1);
    screenshot(&mut h, "box-select");
    release(&mut h, b1);
    let sel = current_selection(&h.ctx);
    assert!(sel.items.contains(&ItemRef::Pin("a".into())));
    assert!(sel.items.contains(&ItemRef::Instance("not1".into())));
    assert!(!sel.items.contains(&ItemRef::Pin("y".into())));
    assert!(!sel.wires.is_empty());

    // 8. Drag the selection down by 4: snapped, wires to y follow.
    drag(&mut h, sp(gp(14, 1)), sp(gp(14, 5)) + egui::vec2(3.0, 4.0));
    let g = h.state().state.doc.editor("top").unwrap().instances["not1"];
    assert_eq!(g.pos, gp(12, 4));
    h.event(Event::PointerGone);
    h.step();
    screenshot(&mut h, "moved");

    // 9. Clicking a wire selects that segment; the inspector shows its net (termination).
    click(&mut h, sp(gp(8, 5)));
    let sel = current_selection(&h.ctx);
    assert!(sel.items.is_empty());
    assert_eq!(sel.wires.len(), 1);
    h.event(Event::PointerGone);
    h.step();
    screenshot(&mut h, "net-inspector");
}

/// UI time per frame (ms) of the app showing Panda-8's `cpu` def with the pointer moving over
/// the canvas, with and without the editing module registered. Run with
/// `cargo test --release -p panda-ui --test edit_ui -- --ignored --nocapture`.
#[test]
#[ignore]
fn edit_tool_frame_cost_panda8() {
    use panda_ui::app::Registry;
    let mut results = Vec::new();
    for with_edit in [false, true, false, true] {
        let mut state = AppState::new(panda_ui::examples::panda8(), SimOptions::default());
        state.wait_layout();
        let mut reg = Registry::default();
        if with_edit {
            panda_ui::edit::register(&mut reg);
        }
        let mut h = Harness::builder()
            .with_size(egui::vec2(1400.0, 860.0))
            .build_ui_state(
                |ui, app: &mut PandaApp| app.show(ui),
                PandaApp::with_registry(state, reg),
            );
        step_until(&mut h, 20.0, |a| a.state.live.values().is_some());
        assert!(h.state_mut().state.enter("cpu"));
        h.state_mut().state.canvas.request_fit();
        step_until(&mut h, 20.0, |a| {
            a.state.live.values().is_some_and(|l| l.map.path == "cpu")
        });
        let mut ms = Vec::new();
        for k in 0..400 {
            let t = k as f32 * 0.05;
            let p = pos2(700.0 + 500.0 * t.sin(), 430.0 + 300.0 * (t * 1.3).cos());
            h.event(Event::PointerMoved(p));
            h.step();
            if k >= 50 {
                ms.push(h.state().state.frame.last_ms);
            }
        }
        ms.sort_by(f64::total_cmp);
        let mean = ms.iter().sum::<f64>() / ms.len() as f64;
        let p95 = ms[ms.len() * 95 / 100];
        println!("edit registered: {with_edit:5}  UI frame mean {mean:.3} ms  p95 {p95:.3} ms");
        results.push((with_edit, mean, p95));
    }
    let best = |w: bool| {
        results
            .iter()
            .filter(|r| r.0 == w)
            .map(|r| r.1)
            .fold(f64::INFINITY, f64::min)
    };
    println!(
        "best mean without edit {:.3} ms, with edit {:.3} ms (+{:.3} ms)",
        best(false),
        best(true),
        best(true) - best(false)
    );
}
