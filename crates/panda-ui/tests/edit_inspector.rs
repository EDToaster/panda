//! Inspector and context menus: the batch / rename / termination actions and the context menu
//! model on a plain [`Document`] (no GPU), then the real app rendered offscreen
//! (egui_kittest + wgpu): inspector panels for every kind of selection, inline validation, and
//! the canvas context menu.
//!
//! Writes `docs/screenshots/edit-inspector-*.png`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use egui::{Event, Modifiers, PointerButton, Pos2};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use panda_netlist::validate::{has_errors, validate};
use panda_netlist::{
    Atom, ComponentDef, Design, Dir, Endpoint, GateOp, InstKind, Instance, Net, PinDecl,
    Termination, Value,
};
use panda_ui::app::{AppState, PandaApp};
use panda_ui::canvas::ViewXf;
use panda_ui::doc::{Document, EDITOR_VERSION, EditorData, InstGeom, ItemRef};
use panda_ui::edit::actions::ports;
use panda_ui::edit::inspector::{
    Field, FieldValue, IdKind, check_new_id, plan_multi, rename_instance, rename_net, set_atoms,
    set_terminations,
};
use panda_ui::edit::menus::{
    MenuAction, Target, entries, net_segments, run_edit, select_for_context,
};
use panda_ui::edit::route::route;
use panda_ui::edit::{Editor, PlaceRequest, Selection, WireSeg, current_selection};
use panda_ui::geom::{GridPt, gp};
use panda_ui::simhost::SimOptions;

// =====================================================================================
// Fixture
// =====================================================================================

fn pin_decl(name: &str, dir: Dir, width: u32) -> PinDecl {
    PinDecl {
        name: name.into(),
        dir,
        width,
    }
}

fn port_ep(i: &str, p: &str) -> Endpoint {
    Endpoint::Port {
        inst: i.into(),
        port: p.into(),
    }
}

/// `sub`: o = NOT i.
fn sub_def() -> ComponentDef {
    ComponentDef {
        id: "sub".into(),
        name: "sub".into(),
        version: 0,
        pins: vec![pin_decl("i", Dir::In, 1), pin_decl("o", Dir::Out, 1)],
        instances: vec![Instance {
            id: "n".into(),
            kind: InstKind::Atom(Atom::Not { width: 1 }),
        }],
        nets: vec![
            Net {
                id: "ni".into(),
                width: 1,
                termination: Termination::None,
                ends: vec![Endpoint::Pin("i".into()), port_ep("n", "in")],
            },
            Net {
                id: "no".into(),
                width: 1,
                termination: Termination::None,
                ends: vec![port_ep("n", "out"), Endpoint::Pin("o".into())],
            },
        ],
        editor: serde_json::Value::Null,
    }
}

/// Instance ids of the fixture.
struct Ids {
    and1: String,
    and2: String,
    not1: String,
    konst: String,
    split: String,
    clock: String,
    pkg: String,
}

fn item_id(r: ItemRef) -> String {
    match r {
        ItemRef::Instance(i) | ItemRef::Pin(i) => i,
    }
}

/// Wire two ports (`"a"` for a pin, `"g1.in0"` for an instance port).
fn wire(ed: &mut Editor, doc: &mut Document, from: &str, to: &str) {
    let pos = |doc: &Document, s: &str| {
        let ep = match s.split_once('.') {
            Some((i, p)) => port_ep(i, p),
            None => Endpoint::Pin(s.into()),
        };
        ports(doc, "top")
            .into_iter()
            .find(|p| p.ep == ep)
            .unwrap_or_else(|| panic!("no port {s}"))
            .pos
    };
    let (a, b) = (pos(doc, from), pos(doc, to));
    ed.draw_wire(doc, "top", &route(a, None, b, None))
        .unwrap_or_else(|e| panic!("wire {from} → {to}: {e}"));
}

/// Top def: inputs `a`, `b`, output `y`; `y = NOT(a AND b)` plus a second AND on `a`, an 8-bit
/// constant into a splitter, a clock, and a `sub` package instance.
fn fixture() -> (Document, Ids) {
    let design = Design {
        format: panda_netlist::FORMAT.into(),
        defs: vec![
            ComponentDef {
                id: "top".into(),
                name: "top".into(),
                version: 0,
                pins: vec![
                    pin_decl("a", Dir::In, 1),
                    pin_decl("b", Dir::In, 1),
                    pin_decl("y", Dir::Out, 1),
                ],
                instances: vec![],
                nets: vec![],
                editor: serde_json::Value::Null,
            },
            sub_def(),
        ],
        top: "top".into(),
    };
    let mut doc = Document::new(design);
    let mut e = EditorData {
        version: EDITOR_VERSION,
        ..EditorData::default()
    };
    for (p, at) in [("a", gp(0, 0)), ("b", gp(0, 4)), ("y", gp(34, 2))] {
        e.pins.insert(
            p.into(),
            InstGeom {
                pos: at,
                ..Default::default()
            },
        );
    }
    doc.editors.insert("top".into(), e);
    doc.layout_missing();
    let mut ed = Editor::new();
    let mut place = |doc: &mut Document, req: PlaceRequest, at: GridPt| {
        item_id(ed.place(doc, "top", &req, at).expect("place"))
    };
    let gate = |op| {
        PlaceRequest::Atom(Atom::Gate {
            op,
            inputs: 2,
            width: 1,
            reduce: false,
        })
    };
    let and1 = place(&mut doc, gate(GateOp::And), gp(14, 3));
    let and2 = place(&mut doc, gate(GateOp::And), gp(14, 10));
    let not1 = place(
        &mut doc,
        PlaceRequest::Atom(Atom::Not { width: 1 }),
        gp(24, 3),
    );
    let konst = place(
        &mut doc,
        PlaceRequest::Atom(Atom::Constant {
            value: Value::from_u64(8, 0x2a),
        }),
        gp(4, 14),
    );
    let split = place(
        &mut doc,
        PlaceRequest::Atom(Atom::Splitter {
            width: 8,
            parts: vec![(4, 7), (0, 3)],
        }),
        gp(16, 14),
    );
    let clock = place(
        &mut doc,
        PlaceRequest::Atom(Atom::Clock {
            period: 16,
            high: 8,
            phase: 0,
        }),
        gp(4, 20),
    );
    let pkg = place(&mut doc, PlaceRequest::Def("sub".into()), gp(26, 12));
    let mut ed = Editor::new();
    wire(&mut ed, &mut doc, "a", &format!("{and1}.in0"));
    wire(&mut ed, &mut doc, "b", &format!("{and1}.in1"));
    wire(
        &mut ed,
        &mut doc,
        &format!("{and1}.out"),
        &format!("{not1}.in"),
    );
    wire(&mut ed, &mut doc, &format!("{not1}.out"), "y");
    wire(
        &mut ed,
        &mut doc,
        &format!("{konst}.out"),
        &format!("{split}.whole"),
    );
    wire(
        &mut ed,
        &mut doc,
        &format!("{and2}.out"),
        &format!("{pkg}.i"),
    );
    let ids = Ids {
        and1,
        and2,
        not1,
        konst,
        split,
        clock,
        pkg,
    };
    (doc, ids)
}

fn top(doc: &Document) -> &ComponentDef {
    doc.def("top").expect("top")
}

fn atom<'a>(doc: &'a Document, id: &str) -> &'a Atom {
    match &top(doc)
        .instances
        .iter()
        .find(|i| i.id == id)
        .expect(id)
        .kind
    {
        InstKind::Atom(a) => a,
        InstKind::Def(_) => panic!("{id} is a package"),
    }
}

fn net_of(doc: &Document, e: &Endpoint) -> Option<String> {
    top(doc)
        .nets
        .iter()
        .find(|n| n.ends.contains(e))
        .map(|n| n.id.clone())
}

fn assert_valid(doc: &Document) {
    let diags = validate(&doc.to_design());
    assert!(!has_errors(&diags), "{diags:?}");
}

// =====================================================================================
// Pure tests
// =====================================================================================

#[test]
fn fixture_is_valid() {
    let (doc, ids) = fixture();
    assert_valid(&doc);
    assert!(net_of(&doc, &port_ep(&ids.and1, "out")).is_some());
    assert!(net_of(&doc, &port_ep(&ids.pkg, "i")).is_some());
}

#[test]
fn multi_edit_is_one_undo_step() {
    let (mut doc, ids) = fixture();
    let mut ed = Editor::new();
    let insts = vec![ids.and1.clone(), ids.and2.clone()];
    let mut pending = BTreeMap::new();
    pending.insert(Field::Op, FieldValue::Op(GateOp::Nor));
    pending.insert(Field::Inputs, FieldValue::Num(3));
    let plan = plan_multi(&doc, "top", &insts, &pending);
    assert!(
        plan.errors.is_empty() && plan.conflicts.is_empty(),
        "{plan:?}"
    );
    assert_eq!(plan.changes.len(), 2);
    let before = ed.history.undo_len();
    set_atoms(&mut ed, &mut doc, "top", &plan.changes, false).unwrap();
    assert_eq!(ed.history.undo_len(), before + 1, "one transaction");
    for id in &insts {
        assert!(matches!(
            atom(&doc, id),
            Atom::Gate {
                op: GateOp::Nor,
                inputs: 3,
                ..
            }
        ));
    }
    // The existing connections survive (in0 / in1 still exist).
    assert!(net_of(&doc, &port_ep(&ids.and1, "in1")).is_some());
    assert_valid(&doc);
    ed.undo(&mut doc).unwrap();
    for id in &insts {
        assert!(matches!(
            atom(&doc, id),
            Atom::Gate {
                op: GateOp::And,
                inputs: 2,
                ..
            }
        ));
    }
    ed.redo(&mut doc).unwrap();
    assert!(matches!(
        atom(&doc, &ids.and2),
        Atom::Gate {
            op: GateOp::Nor,
            ..
        }
    ));
}

#[test]
fn multi_edit_width_conflicts_need_disconnect() {
    let (mut doc, ids) = fixture();
    let mut ed = Editor::new();
    let insts = vec![ids.and1.clone(), ids.and2.clone(), ids.not1.clone()];
    let mut pending = BTreeMap::new();
    pending.insert(Field::Width, FieldValue::Num(4));
    let plan = plan_multi(&doc, "top", &insts, &pending);
    assert!(plan.errors.is_empty());
    assert_eq!(plan.changes.len(), 3);
    // and1: in0, in1, out; and2: out; not1: in, out.
    assert_eq!(plan.conflicts.len(), 6, "{:?}", plan.conflicts);
    let rev = doc.revision;
    assert!(
        set_atoms(&mut ed, &mut doc, "top", &plan.changes, false)
            .unwrap_err()
            .contains("width mismatch")
    );
    assert_eq!(doc.revision, rev, "nothing changed");
    assert!(ed.message.is_some(), "the error is shown");
    set_atoms(&mut ed, &mut doc, "top", &plan.changes, true).unwrap();
    assert!(net_of(&doc, &port_ep(&ids.and1, "out")).is_none());
    assert!(matches!(atom(&doc, &ids.not1), Atom::Not { width: 4 }));
    assert_valid(&doc);
    ed.undo(&mut doc).unwrap();
    assert!(net_of(&doc, &port_ep(&ids.and1, "out")).is_some());

    // Invalid values are errors, not conflicts.
    let mut pending = BTreeMap::new();
    pending.insert(Field::Inputs, FieldValue::Num(0));
    let plan = plan_multi(&doc, "top", &insts, &pending);
    assert_eq!(plan.errors.len(), 2, "{plan:?}");
}

#[test]
fn rename_instance_and_net() {
    let (mut doc, ids) = fixture();
    let mut ed = Editor::new();
    let def = top(&doc).clone();
    // Validation.
    let bad = [
        ("", "empty"),
        ("a b", "spaces"),
        ("x/y", "'/'"),
        ("pin", "reserved"),
        (ids.and2.as_str(), "already used"),
    ];
    for (name, why) in bad {
        let e = check_new_id(&def, IdKind::Instance, &ids.and1, name).unwrap_err();
        assert!(e.contains(why), "{name}: {e}");
    }
    assert!(check_new_id(&def, IdKind::Instance, &ids.and1, &ids.and1).is_ok());
    assert!(check_new_id(&def, IdKind::Net, "x", "pin").is_ok());

    ed.selection
        .items
        .insert(ItemRef::Instance(ids.and1.clone()));
    let n_out = net_of(&doc, &port_ep(&ids.and1, "out")).unwrap();
    let g = doc.editor("top").unwrap().instances[&ids.and1];
    rename_instance(&mut ed, &mut doc, "top", &ids.and1, "gate_a").unwrap();
    assert_eq!(net_of(&doc, &port_ep("gate_a", "out")), Some(n_out.clone()));
    assert!(net_of(&doc, &port_ep(&ids.and1, "out")).is_none());
    assert_eq!(doc.editor("top").unwrap().instances["gate_a"], g);
    assert!(
        ed.selection
            .items
            .contains(&ItemRef::Instance("gate_a".into()))
    );
    assert_valid(&doc);
    assert!(rename_instance(&mut ed, &mut doc, "top", "gate_a", &ids.not1).is_err());

    // Nets: wires and selected segments follow.
    let segs = net_segments(&doc, "top", &n_out);
    assert!(!segs.is_empty());
    ed.selection.clear();
    ed.selection.wires.insert(segs[0].clone());
    rename_net(&mut ed, &mut doc, "top", &n_out, "and_out").unwrap();
    assert_eq!(
        net_of(&doc, &port_ep("gate_a", "out")).as_deref(),
        Some("and_out")
    );
    assert_eq!(net_segments(&doc, "top", "and_out").len(), segs.len());
    assert!(ed.selection.wires.iter().all(|w| w.net == "and_out"));
    assert_valid(&doc);

    ed.undo(&mut doc).unwrap();
    ed.undo(&mut doc).unwrap();
    assert_eq!(net_of(&doc, &port_ep(&ids.and1, "out")), Some(n_out));
    assert_eq!(doc.editor("top").unwrap().instances[&ids.and1], g);
}

#[test]
fn terminations_of_several_nets_in_one_step() {
    let (mut doc, ids) = fixture();
    let mut ed = Editor::new();
    let nets = vec![
        net_of(&doc, &Endpoint::Pin("a".into())).unwrap(),
        net_of(&doc, &port_ep(&ids.konst, "out")).unwrap(),
    ];
    set_terminations(&mut ed, &mut doc, "top", &nets, Termination::PullDown).unwrap();
    assert_eq!(ed.history.undo_len(), 1);
    assert_eq!(ed.history.undo_label(), Some("termination of 2 nets"));
    for n in &nets {
        let t = top(&doc)
            .nets
            .iter()
            .find(|x| &x.id == n)
            .unwrap()
            .termination;
        assert_eq!(t, Termination::PullDown);
    }
    // Nothing to change → no transaction.
    set_terminations(&mut ed, &mut doc, "top", &nets, Termination::PullDown).unwrap();
    assert_eq!(ed.history.undo_len(), 1);
    ed.undo(&mut doc).unwrap();
    assert!(
        top(&doc)
            .nets
            .iter()
            .all(|n| n.termination == Termination::None)
    );
}

fn actions(v: &[Option<panda_ui::edit::menus::Entry>]) -> Vec<MenuAction> {
    v.iter().flatten().map(|e| e.action.clone()).collect()
}

#[test]
fn context_menu_entries() {
    let (doc, ids) = fixture();
    let mut sel = Selection::default();
    let t = Target::Instance(ids.pkg.clone());
    select_for_context(&mut sel, &t, None);
    let a = actions(&entries(&doc, "top", &t, &sel, None, false, gp(0, 0)));
    assert_eq!(a[0], MenuAction::Inspect);
    assert!(a.contains(&MenuAction::EnterPackage(ids.pkg.clone())));
    assert!(a.contains(&MenuAction::RotateCw) && a.contains(&MenuAction::Delete));
    // Atoms cannot be entered.
    let t = Target::Instance(ids.and1.clone());
    select_for_context(&mut sel, &t, None);
    let a = actions(&entries(&doc, "top", &t, &sel, None, false, gp(0, 0)));
    assert!(!a.iter().any(|x| matches!(x, MenuAction::EnterPackage(_))));

    // A net: waveform, select net, the four terminations (current one checked), delete.
    let net = net_of(&doc, &Endpoint::Pin("a".into())).unwrap();
    let seg = net_segments(&doc, "top", &net)[0].clone();
    let t = Target::Net(net.clone());
    select_for_context(&mut sel, &t, Some(&seg));
    assert_eq!(sel.wires.len(), 1);
    assert!(sel.items.is_empty());
    let wave = ("a".to_string(), None);
    let list = entries(&doc, "top", &t, &sel, Some(&wave), false, gp(0, 0));
    let a = actions(&list);
    assert!(a.contains(&MenuAction::AddToWaveform("a".into())));
    assert!(a.contains(&MenuAction::SelectNet(net.clone())));
    let terms: Vec<_> = list
        .iter()
        .flatten()
        .filter(|e| matches!(e.action, MenuAction::Termination(..)))
        .collect();
    assert_eq!(terms.len(), 4);
    assert!(terms[0].checked && !terms[1].checked);
    assert!(!a.contains(&MenuAction::RotateCw), "no items selected");
    let wave = ("a".to_string(), Some(3));
    let a = actions(&entries(
        &doc,
        "top",
        &t,
        &sel,
        Some(&wave),
        false,
        gp(0, 0),
    ));
    assert!(a.contains(&MenuAction::RemoveFromWaveform(3)));

    // Empty canvas: paste (disabled without a clipboard), select all, fit.
    let list = entries(
        &doc,
        "top",
        &Target::Canvas,
        &Selection::default(),
        None,
        false,
        gp(5, 6),
    );
    let paste = list.iter().flatten().next().unwrap();
    assert_eq!(paste.action, MenuAction::Paste(gp(5, 6)));
    assert!(!paste.enabled);
    assert!(actions(&list).contains(&MenuAction::FitView));
}

#[test]
fn right_click_keeps_a_selection_that_contains_the_target() {
    let (_, ids) = fixture();
    let mut sel = Selection::default();
    sel.items.insert(ItemRef::Instance(ids.and1.clone()));
    sel.items.insert(ItemRef::Instance(ids.and2.clone()));
    select_for_context(&mut sel, &Target::Instance(ids.and2.clone()), None);
    assert_eq!(sel.items.len(), 2, "kept");
    select_for_context(&mut sel, &Target::Instance(ids.not1.clone()), None);
    assert_eq!(
        sel.items.iter().cloned().collect::<Vec<_>>(),
        vec![ItemRef::Instance(ids.not1.clone())]
    );
    select_for_context(&mut sel, &Target::Pin("a".into()), None);
    assert_eq!(sel.items.len(), 1);
    assert!(sel.items.contains(&ItemRef::Pin("a".into())));
    let seg = WireSeg::new("n", gp(0, 0), gp(4, 0));
    select_for_context(&mut sel, &Target::Net("n".into()), Some(&seg));
    assert!(sel.items.is_empty() && sel.wires.contains(&seg));
    select_for_context(&mut sel, &Target::Canvas, None);
    assert!(sel.wires.contains(&seg), "canvas keeps the selection");
}

#[test]
fn context_menu_commands_edit_the_document() {
    let (mut doc, ids) = fixture();
    let mut ed = Editor::new();
    // Rotate a selected gate, then delete it (its connections go), then undo both.
    let t = Target::Instance(ids.not1.clone());
    select_for_context(&mut ed.selection, &t, None);
    run_edit(&MenuAction::RotateCw, &mut ed, &mut doc, "top", &t)
        .unwrap()
        .unwrap();
    assert_eq!(
        doc.editor("top").unwrap().instances[&ids.not1].orient.rot,
        1
    );
    run_edit(&MenuAction::Delete, &mut ed, &mut doc, "top", &t)
        .unwrap()
        .unwrap();
    assert!(!top(&doc).instances.iter().any(|i| i.id == ids.not1));
    assert_valid(&doc);
    ed.undo(&mut doc).unwrap();
    ed.undo(&mut doc).unwrap();
    assert_eq!(
        doc.editor("top").unwrap().instances[&ids.not1].orient.rot,
        0
    );

    // Select net / inspect / termination.
    let net = net_of(&doc, &port_ep(&ids.konst, "out")).unwrap();
    let t = Target::Net(net.clone());
    run_edit(
        &MenuAction::SelectNet(net.clone()),
        &mut ed,
        &mut doc,
        "top",
        &t,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        ed.selection.wires.len(),
        net_segments(&doc, "top", &net).len()
    );
    assert_eq!(ed.selection.single_net(), Some(net.as_str()));
    run_edit(
        &MenuAction::Termination(net.clone(), Termination::Keeper),
        &mut ed,
        &mut doc,
        "top",
        &t,
    )
    .unwrap()
    .unwrap();
    let term = top(&doc)
        .nets
        .iter()
        .find(|n| n.id == net)
        .unwrap()
        .termination;
    assert_eq!(term, Termination::Keeper);
    let t = Target::Instance(ids.clock.clone());
    run_edit(&MenuAction::Inspect, &mut ed, &mut doc, "top", &t)
        .unwrap()
        .unwrap();
    assert_eq!(ed.selection.single_instance(), Some(ids.clock.as_str()));

    // Copy + paste here.
    run_edit(&MenuAction::Copy, &mut ed, &mut doc, "top", &t)
        .unwrap()
        .unwrap();
    let n = top(&doc).instances.len();
    run_edit(
        &MenuAction::Paste(gp(40, 30)),
        &mut ed,
        &mut doc,
        "top",
        &Target::Canvas,
    )
    .unwrap()
    .unwrap();
    assert_eq!(top(&doc).instances.len(), n + 1);
    // App-only commands are not edits.
    assert!(
        run_edit(
            &MenuAction::FitView,
            &mut ed,
            &mut doc,
            "top",
            &Target::Canvas
        )
        .is_none()
    );
}

// =====================================================================================
// Offscreen UI
// =====================================================================================

fn screenshot(h: &mut Harness<'_, PandaApp>, name: &str) {
    let img = h.render().expect("render");
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/screenshots")
        .join(format!("edit-inspector-{name}.png"));
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    img.save(&p).unwrap();
}

const XF: ViewXf = ViewXf {
    origin: Pos2 { x: 260.0, y: 120.0 },
    zoom: 18.0,
};

fn step_until(h: &mut Harness<'_, PandaApp>, secs: f64, mut pred: impl FnMut(&PandaApp) -> bool) {
    let t0 = Instant::now();
    while !pred(h.state()) {
        assert!(t0.elapsed().as_secs_f64() < secs, "timed out");
        h.step();
        std::thread::sleep(Duration::from_millis(5));
    }
    h.step();
}

fn button(h: &mut Harness<'_, PandaApp>, p: Pos2, b: PointerButton) {
    let modifiers = h.ctx.input(|i| i.modifiers);
    h.event(Event::PointerMoved(p));
    h.step();
    for pressed in [true, false] {
        h.event(Event::PointerButton {
            pos: p,
            button: b,
            pressed,
            modifiers,
        });
        h.step();
    }
    h.step();
}

/// Screen centre of an item.
fn centre(h: &Harness<'_, PandaApp>, item: &ItemRef) -> Pos2 {
    let placed = h.state().state.doc.placed("top");
    let p = placed.iter().find(|p| &p.item == item).expect("placed");
    let r = p.rect();
    XF.to_screen(egui::pos2(
        (r.min.x + r.max.x) as f32 / 2.0,
        (r.min.y + r.max.y) as f32 / 2.0,
    ))
}

fn click_item(h: &mut Harness<'_, PandaApp>, id: &str) {
    let p = centre(h, &ItemRef::Instance(id.into()));
    button(h, p, PointerButton::Primary);
}

fn app(doc: Document) -> Harness<'static, PandaApp> {
    let state = AppState::from_document(doc, SimOptions::default());
    let mut h = Harness::builder()
        .with_size(egui::vec2(1180.0, 660.0))
        .wgpu()
        .build_ui_state(|ui, app: &mut PandaApp| app.show(ui), PandaApp::new(state));
    step_until(&mut h, 20.0, |a| a.state.live.values().is_some());
    h.state_mut().state.canvas.xf = XF;
    h.step();
    h
}

#[test]
fn inspector_and_context_menu_offscreen() {
    let (doc, ids) = fixture();
    let mut h = app(doc);

    // 1. Constant: radix switch and an inline error before applying.
    click_item(&mut h, &ids.konst);
    assert_eq!(
        current_selection(&h.ctx).single_instance(),
        Some(ids.konst.as_str())
    );
    h.get_by_label("dec").click();
    h.step();
    h.step();
    let field = |h: &Harness<'_, PandaApp>, v: &str| {
        h.get_all_by_role(egui::accesskit::Role::TextInput)
            .find(|n| n.value().as_deref() == Some(v))
            .map(|n| n.rect())
    };
    let r = field(&h, "42").expect("0x2a shown in decimal");
    button(&mut h, r.center(), PointerButton::Primary);
    h.event(Event::Text("9".into()));
    h.step();
    h.step();
    h.step();
    assert!(
        h.query_all_by_label_contains("does not fit in 8 bit")
            .next()
            .is_some(),
        "inline validation of 429"
    );
    screenshot(&mut h, "constant-error");
    h.get_by_label("Revert").click();
    h.step();
    h.step();
    h.get_by_label("bin").click();
    h.step();
    h.step();
    screenshot(&mut h, "constant");

    // 2. Splitter parts editor.
    click_item(&mut h, &ids.split);
    h.step();
    screenshot(&mut h, "splitter");

    // 3. Clock.
    click_item(&mut h, &ids.clock);
    screenshot(&mut h, "clock");

    // 3b. Rename the clock from the id field (Enter applies); an invalid name is refused inline.
    let r = field(&h, &ids.clock).expect("id field");
    button(&mut h, r.center(), PointerButton::Primary);
    h.key_press_modifiers(Modifiers::COMMAND, egui::Key::A);
    h.event(Event::Text("osc".into()));
    h.key_press(egui::Key::Enter);
    h.step();
    h.step();
    assert!(
        top(&h.state().state.doc)
            .instances
            .iter()
            .any(|i| i.id == "osc")
    );
    assert_eq!(current_selection(&h.ctx).single_instance(), Some("osc"));
    let r = field(&h, "osc").expect("renamed id field");
    button(&mut h, r.center(), PointerButton::Primary);
    h.key_press_modifiers(Modifiers::COMMAND, egui::Key::A);
    h.event(Event::Text(ids.and1.clone()));
    h.step();
    h.step();
    assert!(
        h.query_all_by_label_contains("is already used")
            .next()
            .is_some()
    );
    screenshot(&mut h, "rename-error");
    h.key_press(egui::Key::Enter);
    h.step();
    assert!(
        top(&h.state().state.doc)
            .instances
            .iter()
            .any(|i| i.id == "osc"),
        "refused"
    );

    // 4. A width change that conflicts with the connected nets: warnings, Apply disabled,
    //    "Apply & disconnect" offered.
    click_item(&mut h, &ids.not1);
    let r = h
        .get_all_by_role(egui::accesskit::Role::SpinButton)
        .map(|n| n.rect())
        .find(|r| r.left() > 900.0)
        .expect("width field in the inspector");
    button(&mut h, r.center(), PointerButton::Primary);
    h.key_press_modifiers(Modifiers::COMMAND, egui::Key::A);
    h.event(Event::Text("4".into()));
    h.key_press(egui::Key::Enter);
    h.step();
    h.step();
    assert!(
        h.query_all_by_label_contains("would be 4 bit(s)")
            .next()
            .is_some(),
        "width conflict shown inline"
    );
    assert!(h.query_by_label("Apply & disconnect").is_some());
    screenshot(&mut h, "width-conflict");
    h.get_by_label("Revert").click();
    h.step();

    // 5. Multi-selection: the common fields of two AND gates.
    click_item(&mut h, &ids.and1);
    h.event(Event::ModifiersChanged(Modifiers::SHIFT));
    h.step();
    click_item(&mut h, &ids.and2);
    h.event(Event::ModifiersChanged(Modifiers::NONE));
    h.step();
    assert_eq!(current_selection(&h.ctx).items.len(), 2);
    assert!(
        h.query_all_by_label_contains("common fields of AND")
            .next()
            .is_some()
    );
    screenshot(&mut h, "multi");

    // 6. A net: name, termination, connections.
    let net = net_of(&h.state().state.doc, &port_ep(&ids.and1, "out")).unwrap();
    let seg = net_segments(&h.state().state.doc, "top", &net)
        .into_iter()
        .max_by_key(|s| (s.b.x - s.a.x).abs() + (s.b.y - s.a.y).abs())
        .unwrap();
    let mid = XF.to_screen(egui::pos2(
        (seg.a.x + seg.b.x) as f32 / 2.0,
        (seg.a.y + seg.b.y) as f32 / 2.0,
    ));
    button(&mut h, mid, PointerButton::Primary);
    assert_eq!(current_selection(&h.ctx).single_net(), Some(net.as_str()));
    assert!(h.query_by_label(&format!("net {net}")).is_some());
    screenshot(&mut h, "net");

    // 7. Context menu on the wire: termination sub-menu.
    button(&mut h, mid, PointerButton::Secondary);
    assert!(h.query_by_label("Select whole net").is_some());
    assert!(h.query_by_label("Termination").is_some());
    screenshot(&mut h, "menu-net");
    h.get_by_label("pull-up").click();
    h.step();
    h.step();
    let term = top(&h.state().state.doc)
        .nets
        .iter()
        .find(|n| n.id == net)
        .unwrap()
        .termination;
    assert_eq!(term, Termination::PullUp);

    // 8. Context menu on a gate: Rotate.
    let p = centre(&h, &ItemRef::Instance(ids.not1.clone()));
    button(&mut h, p, PointerButton::Secondary);
    assert_eq!(
        current_selection(&h.ctx).single_instance(),
        Some(ids.not1.as_str()),
        "right click selects"
    );
    screenshot(&mut h, "menu-gate");
    h.get_by_label("Rotate  (R)").click();
    h.step();
    h.step();
    let rot = h.state().state.doc.editor("top").unwrap().instances[&ids.not1]
        .orient
        .rot;
    assert_eq!(rot, 1);
    assert!(h.query_by_label("Rotate  (R)").is_none(), "menu closed");

    // 9. Empty canvas menu; Escape closes it.
    let empty = XF.grid_to_screen(gp(30, 19));
    button(&mut h, empty, PointerButton::Secondary);
    assert!(h.query_by_label("Fit view").is_some());
    screenshot(&mut h, "menu-canvas");
    h.key_press(egui::Key::Escape);
    h.step();
    h.step();
    assert!(h.query_by_label("Fit view").is_none());

    // 10. A pin: the sim UX menu (waveform, set value) and this menu stack into one column.
    let p = centre(&h, &ItemRef::Pin("a".into()));
    button(&mut h, p, PointerButton::Secondary);
    h.step();
    screenshot(&mut h, "menu-pin");
    let menu_items = h
        .query_all_by_label("Add to waveform")
        .filter(|n| n.rect().left() < 900.0)
        .count();
    assert_eq!(menu_items, 1, "one waveform entry between both menus");
    assert!(h.query_by_label("Inspect").is_some());
    h.key_press(egui::Key::Escape);
    h.step();
    h.step();

    // 11. Package: inspector, then "Enter package" from its context menu.
    click_item(&mut h, &ids.pkg);
    assert!(
        h.query_by_label(&format!("package instance {}", ids.pkg))
            .is_some()
    );
    screenshot(&mut h, "package");
    let p = centre(&h, &ItemRef::Instance(ids.pkg.clone()));
    button(&mut h, p, PointerButton::Secondary);
    screenshot(&mut h, "menu-package");
    h.query_all_by_label("Enter package")
        .find(|n| n.rect().left() < 900.0)
        .expect("the menu's Enter package")
        .click();
    h.step();
    h.step();
    assert_eq!(h.state().state.view.def, "sub");
}
