//! Controller tests (no GPU): every action, its `EditOp`s, undo/redo round trips, and that the
//! edited design validates, flattens and simulates as expected (also through the real sim
//! thread via [`AppState`]).

use std::time::{Duration, Instant};

use panda_kernel::{EditOp, Kernel, KernelOptions, StepUnit};
use panda_netlist::validate::{has_errors, validate};
use panda_netlist::{
    Atom, ComponentDef, Design, Dir, Endpoint, GateOp, Net, Termination, Value, flatten,
};

use super::actions::{Anchor, PortInfo, anchor_at, ports};
use super::route::{Axis, route};
use super::*;
use crate::app::AppState;
use crate::doc::{Document, EDITOR_VERSION, EditorData, ItemRef};
use crate::geom::{GridPt, GridRect, Orient, gp};
use crate::simhost::{Command, SimEvent, SimOptions};

/// Document target that records every submitted op batch.
struct Rec {
    doc: Document,
    log: Vec<Vec<EditOp>>,
}

impl EditTarget for Rec {
    fn doc(&self) -> &Document {
        &self.doc
    }
    fn doc_mut(&mut self) -> &mut Document {
        &mut self.doc
    }
    fn submit(&mut self, ops: Vec<EditOp>) -> Result<Option<u64>, panda_netlist::Diagnostic> {
        self.log.push(ops.clone());
        self.doc.submit(ops)
    }
}

fn empty_design() -> Design {
    Design {
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
    }
}

fn empty() -> Rec {
    let mut doc = Document::new(empty_design());
    doc.editors.insert(
        "top".into(),
        EditorData {
            version: EDITOR_VERSION,
            ..EditorData::default()
        },
    );
    Rec {
        doc,
        log: Vec::new(),
    }
}

fn port(doc: &Document, ep: &str) -> PortInfo {
    let want = match ep.split_once('.') {
        Some((i, p)) => Endpoint::Port {
            inst: i.into(),
            port: p.into(),
        },
        None => Endpoint::Pin(ep.into()),
    };
    ports(doc, "top")
        .into_iter()
        .find(|p| p.ep == want)
        .unwrap_or_else(|| panic!("no port {ep}"))
}

fn ep(s: &str) -> Endpoint {
    match s.split_once('.') {
        Some((i, p)) => Endpoint::Port {
            inst: i.into(),
            port: p.into(),
        },
        None => Endpoint::Pin(s.into()),
    }
}

/// Wire between two ports, routed like the canvas does.
fn wire(ed: &mut Editor, t: &mut Rec, a: &str, b: &str) -> Result<(), String> {
    let (pa, pb) = (port(&t.doc, a), port(&t.doc, b));
    let pts = route(
        pa.pos,
        Some(Axis::of_side(pa.side)),
        pb.pos,
        Some(Axis::of_side(pb.side)),
    );
    ed.draw_wire(t, "top", &pts)
}

fn assert_valid(doc: &Document) {
    let d = validate(&doc.design);
    assert!(!has_errors(&d), "validation errors: {d:#?}");
    flatten(&doc.design).expect("flattens");
}

fn top(doc: &Document) -> &ComponentDef {
    doc.def("top").unwrap()
}

fn net_with<'a>(doc: &'a Document, e: &str) -> Option<&'a Net> {
    top(doc).nets.iter().find(|n| n.ends.contains(&ep(e)))
}

/// Simulate the document's design: apply inputs, settle, read outputs.
fn simulate(doc: &Document, inputs: &[(&str, bool)], outputs: &[&str]) -> Vec<Option<u64>> {
    let mut k = Kernel::new(KernelOptions::default());
    k.load(doc.design.clone()).expect("loads");
    for (pin, v) in inputs {
        k.set_input(pin, &Value::from_bool(*v)).unwrap();
    }
    k.step(20, StepUnit::Tick);
    outputs.iter().map(|o| k.read_u64(o)).collect()
}

/// State of the document that undo/redo must restore exactly.
fn state(doc: &Document) -> (Design, EditorData) {
    (
        doc.design.clone(),
        doc.editor("top").cloned().unwrap_or_default(),
    )
}

/// The fixture used by most tests:
///
/// ```text
///  in1 ──────────────────── not1 ── out1        (net A, y = 0)
///            or1 ─────────── and1 ── out2       (net B, y = 8: or1.in0 – and1.in0, undriven)
///                 const1 ─── and1.in1           (constant 1)
/// ```
fn fixture() -> (Editor, Rec) {
    let mut ed = Editor::new();
    let mut t = empty();
    let place = |ed: &mut Editor, t: &mut Rec, r: PlaceRequest, at: GridPt| {
        ed.place(t, "top", &r, at).unwrap()
    };
    let pin_in = PlaceRequest::Pin {
        dir: Dir::In,
        width: 1,
    };
    let pin_out = PlaceRequest::Pin {
        dir: Dir::Out,
        width: 1,
    };
    let gate = |op| {
        PlaceRequest::Atom(Atom::Gate {
            op,
            inputs: 2,
            width: 1,
            reduce: false,
        })
    };
    assert_eq!(
        place(&mut ed, &mut t, pin_in, gp(2, 0)),
        ItemRef::Pin("in1".into())
    );
    place(
        &mut ed,
        &mut t,
        PlaceRequest::Atom(Atom::Not { width: 1 }),
        gp(26, 0),
    );
    place(&mut ed, &mut t, pin_out.clone(), gp(36, 0));
    place(&mut ed, &mut t, gate(GateOp::Or), gp(16, 9));
    place(&mut ed, &mut t, gate(GateOp::And), gp(30, 9));
    place(
        &mut ed,
        &mut t,
        PlaceRequest::Atom(Atom::Constant {
            value: Value::from_bool(true),
        }),
        gp(21, 10),
    );
    place(&mut ed, &mut t, pin_out, gp(38, 9));
    assert_eq!(port(&t.doc, "in1").pos, gp(4, 0));
    assert_eq!(port(&t.doc, "not1.in").pos, gp(24, 0));
    assert_eq!(port(&t.doc, "or1.in0").pos, gp(14, 8));
    assert_eq!(port(&t.doc, "and1.in0").pos, gp(28, 8));
    assert_eq!(port(&t.doc, "const1.out").pos, gp(22, 10));
    wire(&mut ed, &mut t, "in1", "not1.in").unwrap();
    wire(&mut ed, &mut t, "not1.out", "out1").unwrap();
    wire(&mut ed, &mut t, "or1.in0", "and1.in0").unwrap();
    wire(&mut ed, &mut t, "const1.out", "and1.in1").unwrap();
    wire(&mut ed, &mut t, "and1.out", "out2").unwrap();
    (ed, t)
}

#[test]
fn place_and_wire_emit_add_ops_and_simulate() {
    let mut ed = Editor::new();
    let mut t = empty();
    ed.place(
        &mut t,
        "top",
        &PlaceRequest::Pin {
            dir: Dir::In,
            width: 1,
        },
        gp(2, 0),
    )
    .unwrap();
    assert_eq!(
        t.log.last().unwrap(),
        &vec![EditOp::AddPin {
            def: "top".into(),
            pin: panda_netlist::PinDecl {
                name: "in1".into(),
                dir: Dir::In,
                width: 1
            }
        }]
    );
    let r = ed
        .place(
            &mut t,
            "top",
            &PlaceRequest::Atom(Atom::Not { width: 1 }),
            gp(12, 0),
        )
        .unwrap();
    assert_eq!(r, ItemRef::Instance("not1".into()));
    assert!(matches!(
        &t.log.last().unwrap()[0],
        EditOp::AddInstance { def, instance } if def == "top" && instance.id == "not1"
    ));
    // Placed centred on the cursor and selected.
    assert_eq!(
        t.doc.editor("top").unwrap().instances["not1"].pos,
        gp(10, -1)
    );
    assert!(ed.selection.items.contains(&r));
    ed.place(
        &mut t,
        "top",
        &PlaceRequest::Pin {
            dir: Dir::Out,
            width: 1,
        },
        gp(22, 0),
    )
    .unwrap();
    wire(&mut ed, &mut t, "in1", "not1.in").unwrap();
    assert_eq!(
        t.log.last().unwrap(),
        &vec![EditOp::AddNet {
            def: "top".into(),
            net: Net {
                id: "n1".into(),
                width: 1,
                termination: Termination::None,
                ends: vec![ep("in1"), ep("not1.in")],
            }
        }]
    );
    wire(&mut ed, &mut t, "out1", "not1.out").unwrap();
    assert_valid(&t.doc);
    assert_eq!(
        simulate(&t.doc, &[("in1", false)], &["out1"]),
        vec![Some(1)]
    );
    assert_eq!(simulate(&t.doc, &[("in1", true)], &["out1"]), vec![Some(0)]);
    // The wires end exactly on the ports.
    let g = &t.doc.editor("top").unwrap().wires["n1"];
    let pts: Vec<GridPt> = g.polylines.iter().flatten().copied().collect();
    assert!(pts.contains(&gp(4, 0)) && pts.contains(&gp(10, 0)));
}

#[test]
fn fixture_validates_and_simulates() {
    let (_, t) = fixture();
    assert_valid(&t.doc);
    // Net B is undriven: and1 sees Z → X.
    let r = simulate(&t.doc, &[("in1", true)], &["out1", "out2"]);
    assert_eq!(r, vec![Some(0), None]);
}

#[test]
fn wire_onto_existing_wire_connects_and_adds_junction() {
    let (mut ed, mut t) = fixture();
    // From or1.in1 (14, 10) up onto net A's wire at (10, 0)? Route: port leaves horizontally.
    let p = port(&t.doc, "or1.in1").pos;
    assert_eq!(p, gp(14, 10));
    let n_a = net_with(&t.doc, "in1").unwrap().id.clone();
    let pts = route(p, Some(Axis::Horizontal), gp(10, 0), None);
    assert_eq!(*pts.last().unwrap(), gp(10, 0));
    assert_eq!(
        anchor_at(&t.doc, "top", gp(10, 0)),
        Anchor::Wire { net: n_a.clone() }
    );
    ed.draw_wire(&mut t, "top", &pts).unwrap();
    assert_eq!(
        t.log.last().unwrap(),
        &vec![EditOp::Connect {
            def: "top".into(),
            net: n_a.clone(),
            endpoint: ep("or1.in1"),
        }]
    );
    let g = &t.doc.editor("top").unwrap().wires[&n_a];
    assert_eq!(g.junctions, vec![gp(10, 0)]);
    assert_valid(&t.doc);
}

#[test]
fn wire_between_nets_merges_them() {
    let (mut ed, mut t) = fixture();
    let before = state(&t.doc);
    let a = net_with(&t.doc, "in1").unwrap().id.clone();
    let b = net_with(&t.doc, "or1.in0").unwrap().id.clone();
    ed.draw_wire(&mut t, "top", &[gp(20, 0), gp(20, 8)])
        .unwrap();
    assert_eq!(
        t.log.last().unwrap(),
        &vec![
            EditOp::RemoveNet {
                def: "top".into(),
                id: b.clone()
            },
            EditOp::Connect {
                def: "top".into(),
                net: a.clone(),
                endpoint: ep("or1.in0")
            },
            EditOp::Connect {
                def: "top".into(),
                net: a.clone(),
                endpoint: ep("and1.in0")
            },
        ]
    );
    let g = &t.doc.editor("top").unwrap().wires[&a];
    assert_eq!(g.junctions, vec![gp(20, 0), gp(20, 8)]);
    assert!(!t.doc.editor("top").unwrap().wires.contains_key(&b));
    assert_valid(&t.doc);
    // in1 now drives and1 (with the constant 1): out2 = in1.
    for v in [false, true] {
        assert_eq!(
            simulate(&t.doc, &[("in1", v)], &["out2"]),
            vec![Some(u64::from(v))]
        );
    }
    // Undo restores both nets and their wires exactly; redo merges again.
    let after = state(&t.doc);
    ed.undo(&mut t).unwrap();
    assert_eq!(state(&t.doc), before);
    assert_eq!(
        t.log.last().unwrap(),
        &vec![EditOp::ReplaceDef(before.0.defs[0].clone())]
    );
    ed.redo(&mut t).unwrap();
    assert_eq!(state(&t.doc), after);
}

#[test]
fn deleting_the_bridge_splits_the_net() {
    let (mut ed, mut t) = fixture();
    ed.draw_wire(&mut t, "top", &[gp(20, 0), gp(20, 8)])
        .unwrap();
    let merged = state(&t.doc);
    let a = net_with(&t.doc, "in1").unwrap().id.clone();
    ed.click_select(
        None,
        Some(WireSeg::new(a.clone(), gp(20, 0), gp(20, 8))),
        false,
    );
    ed.delete_selection(&mut t, "top").unwrap();
    let ops = t.log.last().unwrap().clone();
    assert_eq!(ops.len(), 3, "{ops:#?}");
    let disconnected: Vec<&Endpoint> = ops
        .iter()
        .filter_map(|o| match o {
            EditOp::Disconnect { endpoint, net, .. } if *net == a => Some(endpoint),
            _ => None,
        })
        .collect();
    let EditOp::AddNet { net, .. } = &ops[2] else {
        panic!("{ops:#?}")
    };
    assert_eq!(disconnected.len(), 2);
    assert_eq!(net.ends.len(), 2);
    for e in &net.ends {
        assert!(disconnected.contains(&e));
    }
    // Two nets again, with the original partition of endpoints.
    let na = net_with(&t.doc, "in1").unwrap();
    assert!(na.ends.contains(&ep("not1.in")) && na.ends.len() == 2);
    let nb = net_with(&t.doc, "or1.in0").unwrap();
    assert!(nb.ends.contains(&ep("and1.in0")) && nb.ends.len() == 2);
    assert_ne!(na.id, nb.id);
    assert_valid(&t.doc);
    assert_eq!(simulate(&t.doc, &[("in1", true)], &["out2"]), vec![None]);
    // Each piece has wires reaching its endpoints, no junction dots left at the cut.
    let ed_data = t.doc.editor("top").unwrap();
    for n in [&na.id, &nb.id] {
        let g = &ed_data.wires[n];
        assert!(!g.junctions.contains(&gp(20, 0)) && !g.junctions.contains(&gp(20, 8)));
    }
    ed.undo(&mut t).unwrap();
    assert_eq!(state(&t.doc), merged);
}

#[test]
fn width_mismatch_is_rejected_without_change() {
    let (mut ed, mut t) = fixture();
    ed.place(
        &mut t,
        "top",
        &PlaceRequest::Atom(Atom::Buf { width: 8 }),
        gp(10, 20),
    )
    .unwrap();
    let before = state(&t.doc);
    let undo = ed.history.undo_len();
    let e = wire(&mut ed, &mut t, "or1.out", "buf1.in").unwrap_err();
    assert!(e.contains("width mismatch"), "{e}");
    assert_eq!(ed.message.as_deref(), Some(e.as_str()));
    assert_eq!(state(&t.doc), before);
    assert_eq!(ed.history.undo_len(), undo);
    // A wire from nowhere to nowhere is refused too.
    assert!(
        ed.draw_wire(&mut t, "top", &[gp(-30, -30), gp(-20, -30)])
            .is_err()
    );
}

#[test]
fn move_rubber_bands_wires_and_undoes() {
    let (mut ed, mut t) = fixture();
    let before = state(&t.doc);
    ed.click_select(Some(ItemRef::Instance("not1".into())), None, false);
    ed.move_selection(&mut t, "top", gp(3, 4)).unwrap();
    // Geometry only: no kernel ops.
    assert_eq!(t.doc.design, before.0);
    let g = &t.doc.editor("top").unwrap().instances["not1"];
    assert_eq!(g.pos, gp(27, 3));
    let new_in = port(&t.doc, "not1.in").pos;
    assert_eq!(new_in, gp(27, 4));
    for id in [
        net_with(&t.doc, "in1").unwrap().id.clone(),
        net_with(&t.doc, "out1").unwrap().id.clone(),
    ] {
        let w = &t.doc.editor("top").unwrap().wires[&id];
        for l in &w.polylines {
            assert!(l.windows(2).all(|s| s[0].x == s[1].x || s[0].y == s[1].y));
        }
    }
    let a = &t.doc.editor("top").unwrap().wires[&net_with(&t.doc, "in1").unwrap().id];
    let segs: Vec<_> = a.segments().collect();
    assert!(segs.iter().any(|&s| route::on_segment(gp(4, 0), s)));
    assert!(segs.iter().any(|&s| route::on_segment(new_in, s)));
    let after = state(&t.doc);
    ed.undo(&mut t).unwrap();
    assert_eq!(state(&t.doc), before);
    ed.redo(&mut t).unwrap();
    assert_eq!(state(&t.doc), after);
    // Moving a wire segment with the selection.
    ed.click_select(
        None,
        Some(WireSeg::new(
            net_with(&t.doc, "or1.in0").unwrap().id.clone(),
            gp(14, 8),
            gp(28, 8),
        )),
        false,
    );
    ed.move_selection(&mut t, "top", gp(0, -2)).unwrap();
    let b = &t.doc.editor("top").unwrap().wires[&net_with(&t.doc, "or1.in0").unwrap().id];
    let segs: Vec<_> = b.segments().collect();
    // The segment moved to y = 6; links keep it attached to the ports.
    assert!(segs.iter().any(|&s| route::on_segment(gp(14, 8), s)));
    assert!(segs.iter().any(|&s| route::on_segment(gp(28, 8), s)));
    assert!(segs.iter().any(|&(a, b)| a.y == 6 && b.y == 6));
    assert_eq!(ed.selection.wires.len(), 1);
}

#[test]
fn rotate_and_flip_follow_ports() {
    let (mut ed, mut t) = fixture();
    let design = t.doc.design.clone();
    let geom0 = t.doc.editor("top").unwrap().clone();
    ed.click_select(Some(ItemRef::Instance("not1".into())), None, false);
    ed.rotate_selection(&mut t, "top", true).unwrap();
    let o = t.doc.editor("top").unwrap().instances["not1"].orient;
    assert_eq!(
        o,
        Orient {
            rot: 1,
            flip: false
        }
    );
    // Rotated about its centre: the 4×2 box becomes 2×4 around the same centre.
    assert_eq!(
        t.doc.editor("top").unwrap().instances["not1"].pos,
        gp(25, -2)
    );
    let p_in = port(&t.doc, "not1.in").pos;
    let p_out = port(&t.doc, "not1.out").pos;
    assert_eq!((p_in, p_out), (gp(26, -2), gp(26, 2)));
    let wires = &t.doc.editor("top").unwrap().wires;
    let a = &wires[&net_with(&t.doc, "in1").unwrap().id];
    assert!(a.segments().any(|s| route::on_segment(p_in, s)));
    let o1 = &wires[&net_with(&t.doc, "out1").unwrap().id];
    assert!(o1.segments().any(|s| route::on_segment(p_out, s)));
    assert_eq!(t.doc.design, design, "rotation is geometry only");
    for _ in 0..3 {
        ed.rotate_selection(&mut t, "top", true).unwrap();
    }
    let g = t.doc.editor("top").unwrap().instances["not1"];
    assert_eq!(
        (g.orient, g.pos),
        (Orient::default(), geom0.instances["not1"].pos)
    );
    ed.flip_selection(&mut t, "top").unwrap();
    assert_eq!(
        t.doc.editor("top").unwrap().instances["not1"].orient,
        Orient { rot: 0, flip: true }
    );
    // Mirrored: in on the right now.
    assert_eq!(port(&t.doc, "not1.in").pos, gp(28, 0));
    ed.rotate_selection(&mut t, "top", true).unwrap();
    ed.flip_selection(&mut t, "top").unwrap();
    // F·R·F = R⁻¹.
    assert_eq!(
        t.doc.editor("top").unwrap().instances["not1"].orient,
        Orient {
            rot: 3,
            flip: false
        }
    );
    // Undo everything back to the start.
    for _ in 0..7 {
        ed.undo(&mut t).unwrap();
    }
    assert_eq!(t.doc.editor("top").unwrap(), &geom0);
}

#[test]
fn copy_paste_creates_fresh_ids_and_internal_nets() {
    let (mut ed, mut t) = fixture();
    for id in ["or1", "and1", "const1"] {
        ed.click_select(Some(ItemRef::Instance(id.into())), None, true);
    }
    assert_eq!(ed.copy(&t.doc, "top"), 3);
    let clip = ed.clipboard.clone().unwrap();
    // Nets entirely between copied items: or1.in0–and1.in0 and const1–and1.in1.
    assert_eq!(clip.nets.len(), 2);
    assert_eq!(clip.origin, gp(14, 7));
    ed.paste(&mut t, "top", Some(gp(14, 20))).unwrap();
    let ops = t.log.last().unwrap();
    assert_eq!(ops.len(), 5);
    let added: Vec<&str> = ops
        .iter()
        .filter_map(|o| match o {
            EditOp::AddInstance { instance, .. } => Some(instance.id.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(added, vec!["or2", "and2", "const2"]);
    let nets: Vec<&Net> = ops
        .iter()
        .filter_map(|o| match o {
            EditOp::AddNet { net, .. } => Some(net),
            _ => None,
        })
        .collect();
    assert!(
        nets.iter()
            .any(|n| n.ends == vec![ep("or2.in0"), ep("and2.in0")])
    );
    assert!(
        nets.iter()
            .any(|n| n.ends == vec![ep("const2.out"), ep("and2.in1")])
    );
    let e = t.doc.editor("top").unwrap();
    assert_eq!(e.instances["or2"].pos, gp(14, 20));
    assert_eq!(e.instances["and2"].pos, gp(28, 20));
    // The pasted items are selected.
    assert_eq!(ed.selection.items.len(), 3);
    assert!(
        ed.selection
            .items
            .contains(&ItemRef::Instance("and2".into()))
    );
    assert_valid(&t.doc);
    // Duplicate (copy + paste with an offset) keeps the clipboard.
    ed.duplicate(&mut t, "top").unwrap();
    assert!(top(&t.doc).instances.iter().any(|i| i.id == "and3"));
    assert_eq!(ed.clipboard.as_ref().unwrap(), &clip);
    // Cut removes the selection and fills the clipboard.
    ed.cut(&mut t, "top").unwrap();
    assert!(!top(&t.doc).instances.iter().any(|i| i.id == "and3"));
    assert_eq!(ed.clipboard.as_ref().unwrap().items.len(), 3);
    assert_valid(&t.doc);
}

#[test]
fn delete_instance_and_pin_and_undo() {
    let (mut ed, mut t) = fixture();
    let before = state(&t.doc);
    ed.click_select(Some(ItemRef::Instance("not1".into())), None, false);
    ed.click_select(Some(ItemRef::Pin("out2".into())), None, true);
    ed.delete_selection(&mut t, "top").unwrap();
    let ops = t.log.last().unwrap();
    assert!(ops.contains(&EditOp::RemoveInstance {
        def: "top".into(),
        id: "not1".into()
    }));
    assert!(ops.contains(&EditOp::RemovePin {
        def: "top".into(),
        name: "out2".into()
    }));
    assert!(ed.selection.is_empty());
    assert!(!top(&t.doc).instances.iter().any(|i| i.id == "not1"));
    // The wire of net A stays (dangling); its net keeps in1.
    let a = net_with(&t.doc, "in1").unwrap();
    assert_eq!(a.ends, vec![ep("in1")]);
    assert!(t.doc.editor("top").unwrap().wires.contains_key(&a.id));
    assert_valid(&t.doc);
    ed.undo(&mut t).unwrap();
    assert_eq!(state(&t.doc), before);
    // Deleting every segment of a net with no endpoints left removes the net.
    ed.click_select(Some(ItemRef::Instance("const1".into())), None, false);
    let n = net_with(&t.doc, "const1.out").unwrap().id.clone();
    ed.click_select(Some(ItemRef::Instance("and1".into())), None, true);
    ed.click_select(Some(ItemRef::Instance("or1".into())), None, true);
    for (a, b) in t.doc.editor("top").unwrap().wires[&n].segments() {
        ed.selection.wires.insert(WireSeg::new(n.clone(), a, b));
    }
    ed.delete_selection(&mut t, "top").unwrap();
    assert!(top(&t.doc).nets.iter().all(|x| x.id != n));
    assert!(t.log.last().unwrap().contains(&EditOp::RemoveNet {
        def: "top".into(),
        id: n.clone()
    }));
    assert_valid(&t.doc);
}

#[test]
fn set_atom_moves_wires_and_checks_widths() {
    let (mut ed, mut t) = fixture();
    let out_before = port(&t.doc, "and1.out").pos;
    // 2 → 3 inputs: the output moves down one unit; its wire follows.
    let three = Atom::Gate {
        op: GateOp::And,
        inputs: 3,
        width: 1,
        reduce: false,
    };
    ed.set_atom(&mut t, "top", "and1", &three, false).unwrap();
    assert_eq!(
        t.log.last().unwrap(),
        &vec![EditOp::SetAtom {
            def: "top".into(),
            instance: "and1".into(),
            atom: three.clone()
        }]
    );
    let out_after = port(&t.doc, "and1.out").pos;
    assert_eq!(out_after, out_before + gp(0, 1));
    let w = &t.doc.editor("top").unwrap().wires[&net_with(&t.doc, "out2").unwrap().id];
    assert!(w.segments().any(|s| route::on_segment(out_after, s)));
    assert_valid(&t.doc);
    // Width 8 conflicts with the 1-bit nets: refused, with the reason …
    let wide = Atom::Gate {
        op: GateOp::And,
        inputs: 3,
        width: 8,
        reduce: false,
    };
    let e = ed
        .set_atom(&mut t, "top", "and1", &wide, false)
        .unwrap_err();
    assert!(
        e.contains("and1.in0") && e.contains("width mismatch"),
        "{e}"
    );
    // … unless the mismatched connections are dropped.
    ed.set_atom(&mut t, "top", "and1", &wide, true).unwrap();
    let ops = t.log.last().unwrap();
    assert_eq!(
        ops.iter()
            .filter(|o| matches!(o, EditOp::Disconnect { .. }))
            .count(),
        3
    );
    assert!(top(&t.doc).nets.iter().all(|n| {
        n.ends
            .iter()
            .all(|e| !matches!(e, Endpoint::Port { inst, .. } if inst == "and1"))
    }));
    assert_valid(&t.doc);
    ed.undo(&mut t).unwrap();
    // 3 → 1 input: in1 and in2 vanish, their connections are dropped automatically.
    let one = Atom::Gate {
        op: GateOp::And,
        inputs: 1,
        width: 1,
        reduce: false,
    };
    ed.set_atom(&mut t, "top", "and1", &one, false).unwrap();
    assert_eq!(
        t.log.last().unwrap()[0],
        EditOp::Disconnect {
            def: "top".into(),
            net: net_with(&t.doc, "const1.out").unwrap().id.clone(),
            endpoint: ep("and1.in1"),
        }
    );
    assert_valid(&t.doc);
    // Constant value, clock and power-on reset parameters.
    ed.set_atom(
        &mut t,
        "top",
        "const1",
        &Atom::Constant {
            value: Value::from_bool(false),
        },
        false,
    )
    .unwrap();
    let clk = ed
        .place(
            &mut t,
            "top",
            &PlaceRequest::Atom(Atom::Clock {
                period: 16,
                high: 8,
                phase: 0,
            }),
            gp(0, 30),
        )
        .unwrap();
    let ItemRef::Instance(clk) = clk else {
        panic!()
    };
    let c2 = Atom::Clock {
        period: 10,
        high: 3,
        phase: 1,
    };
    ed.set_atom(&mut t, "top", &clk, &c2, false).unwrap();
    assert!(
        top(&t.doc)
            .instances
            .iter()
            .any(|i| i.id == clk && i.kind == panda_netlist::InstKind::Atom(c2.clone()))
    );
    let bad = Atom::Clock {
        period: 10,
        high: 10,
        phase: 0,
    };
    assert!(ed.set_atom(&mut t, "top", &clk, &bad, false).is_err());
    assert_valid(&t.doc);
}

#[test]
fn termination_and_undo() {
    let (mut ed, mut t) = fixture();
    let n = net_with(&t.doc, "or1.in0").unwrap().id.clone();
    ed.set_termination(&mut t, "top", &n, Termination::PullUp)
        .unwrap();
    assert_eq!(
        t.log.last().unwrap(),
        &vec![EditOp::SetTermination {
            def: "top".into(),
            net: n.clone(),
            termination: Termination::PullUp
        }]
    );
    assert_valid(&t.doc);
    // The pulled-up net now reads 1 at and1.in0: out2 = 1.
    assert_eq!(
        simulate(&t.doc, &[("in1", false)], &["out2"]),
        vec![Some(1)]
    );
    ed.undo(&mut t).unwrap();
    assert_eq!(
        net_with(&t.doc, "or1.in0").unwrap().termination,
        Termination::None
    );
    // Merging a pulled-up net into a plain one carries the termination over.
    ed.redo(&mut t).unwrap();
    ed.draw_wire(&mut t, "top", &[gp(20, 0), gp(20, 8)])
        .unwrap();
    assert_eq!(
        net_with(&t.doc, "in1").unwrap().termination,
        Termination::PullUp
    );
}

#[test]
fn selection_click_box_and_prune() {
    let (mut ed, t) = fixture();
    ed.sync(&t.doc, "", "top");
    ed.click_select(Some(ItemRef::Instance("or1".into())), None, false);
    ed.click_select(Some(ItemRef::Instance("and1".into())), None, true);
    assert_eq!(ed.selection.items.len(), 2);
    // Shift-click toggles off.
    ed.click_select(Some(ItemRef::Instance("or1".into())), None, true);
    assert_eq!(ed.selection.items.len(), 1);
    // Plain click on empty space clears.
    ed.click_select(None, None, false);
    assert!(ed.selection.is_empty());
    // Box around or1, and1, const1 and the wires between them.
    ed.box_select(
        &t.doc,
        "top",
        GridRect {
            min: gp(13, 6),
            max: gp(33, 12),
        },
        false,
    );
    let items: Vec<String> = ed
        .selection
        .items
        .iter()
        .map(|i| format!("{i:?}"))
        .collect();
    assert_eq!(items.len(), 3, "{items:?}");
    assert!(!ed.selection.wires.is_empty());
    // Changing the view clears the selection.
    ed.sync(&t.doc, "sub", "top");
    assert!(ed.selection.is_empty());
}

#[test]
fn placing_sources_and_defs_is_checked() {
    let mut ed = Editor::new();
    let mut doc = Document::new(crate::examples::counter());
    doc.layout_missing();
    let top_id = doc.design.top.clone();
    let inner = doc
        .def(&top_id)
        .unwrap()
        .instances
        .iter()
        .find_map(|i| match &i.kind {
            panda_netlist::InstKind::Def(d) => Some(d.clone()),
            _ => None,
        })
        .unwrap();
    // A clock inside a package is refused.
    let e = ed
        .place(
            &mut doc,
            &inner,
            &PlaceRequest::Atom(Atom::Clock {
                period: 4,
                high: 2,
                phase: 0,
            }),
            gp(0, 0),
        )
        .unwrap_err();
    assert!(e.contains("top def"), "{e}");
    // The top def cannot be placed inside the counter (recursion).
    let e = ed
        .place(
            &mut doc,
            &inner,
            &PlaceRequest::Def(top_id.clone()),
            gp(0, 0),
        )
        .unwrap_err();
    assert!(e.contains("recursive"), "{e}");
    // Another counter in the top is fine and validates.
    let r = ed
        .place(
            &mut doc,
            &top_id,
            &PlaceRequest::Def(inner.clone()),
            gp(0, 40),
        )
        .unwrap();
    assert!(matches!(r, ItemRef::Instance(_)));
    assert!(!has_errors(&validate(&doc.design)));
}

// ---------------------------------------------------------------------------------------
// Through the real sim thread
// ---------------------------------------------------------------------------------------

/// Pump app events until the kernel answered the newest edit (rejections go to the editor).
fn pump(app: &mut AppState, ed: &mut Editor) -> bool {
    let id = last_kernel_id(ed);
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_secs(10) {
        for ev in app.update() {
            if let SimEvent::Edited { id: got, result } = ev {
                if let Err(d) = &result {
                    let why = d.first().map_or(String::new(), ToString::to_string);
                    ed.kernel_rejected(app, got, &why);
                }
                if got == id {
                    return result.is_ok();
                }
            }
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    panic!("no answer for edit {id}");
}

fn last_kernel_id(ed: &Editor) -> u64 {
    // The id of the newest submitted edit is the app's latest id; the history records it.
    ed.history.last_kernel_id().expect("structural action")
}

fn app_port(app: &AppState, e: &str) -> PortInfo {
    let want = ep(e);
    ports(&app.doc, &app.view.def)
        .into_iter()
        .find(|p| p.ep == want)
        .unwrap()
}

fn app_wire(ed: &mut Editor, app: &mut AppState, a: &str, b: &str) -> Result<(), String> {
    let (pa, pb) = (app_port(app, a), app_port(app, b));
    let pts = route(
        pa.pos,
        Some(Axis::of_side(pa.side)),
        pb.pos,
        Some(Axis::of_side(pb.side)),
    );
    let def = app.view.def.clone();
    ed.draw_wire(app, &def, &pts)
}

fn live_pin(app: &mut AppState, pin: &str, secs: f64) -> Option<u64> {
    let t0 = Instant::now();
    loop {
        app.update();
        if let Some(l) = app.live.values()
            && let Some(v) = l.map.pin_value(pin, l.values)
            && let Some(n) = v.to_u64()
        {
            return Some(n);
        }
        if t0.elapsed().as_secs_f64() > secs {
            return None;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn edits_reach_the_kernel_and_simulate_live() {
    let mut design = empty_design();
    // Start with one input so the design is not empty.
    design.defs[0].pins.push(panda_netlist::PinDecl {
        name: "a".into(),
        dir: Dir::In,
        width: 1,
    });
    let mut doc = Document::new(design);
    doc.layout_missing();
    let mut app = AppState::from_document(doc, SimOptions::default());
    let mut ed = Editor::new();
    let def = app.view.def.clone();
    let at_a = app_port(&app, "a").pos;
    ed.place(
        &mut app,
        &def,
        &PlaceRequest::Atom(Atom::Not { width: 1 }),
        at_a + gp(10, 0),
    )
    .unwrap();
    assert!(pump(&mut app, &mut ed));
    ed.place(
        &mut app,
        &def,
        &PlaceRequest::Pin {
            dir: Dir::Out,
            width: 1,
        },
        at_a + gp(20, 0),
    )
    .unwrap();
    assert!(pump(&mut app, &mut ed));
    app_wire(&mut ed, &mut app, "a", "not1.in").unwrap();
    assert!(pump(&mut app, &mut ed));
    app_wire(&mut ed, &mut app, "not1.out", "out1").unwrap();
    assert!(pump(&mut app, &mut ed));
    app.sim.send(Command::SetInput {
        pin: "a".into(),
        value: Value::from_bool(false),
    });
    app.sim.send(Command::Step {
        n: 5,
        unit: StepUnit::Tick,
    });
    assert_eq!(live_pin(&mut app, "out1", 10.0), Some(1));
    app.sim.send(Command::SetInput {
        pin: "a".into(),
        value: Value::from_bool(true),
    });
    app.sim.send(Command::Step {
        n: 5,
        unit: StepUnit::Tick,
    });
    let t0 = Instant::now();
    while live_pin(&mut app, "out1", 10.0) != Some(0) {
        assert!(
            t0.elapsed() < Duration::from_secs(10),
            "output never became 0"
        );
    }
    // Undo the output wire: the kernel follows (ReplaceDef → new netlist generation) and the
    // output no longer follows the input.
    let gen_before = app.live.design_gen;
    ed.undo(&mut app).unwrap();
    assert!(pump(&mut app, &mut ed));
    let t0 = Instant::now();
    while app.live.design_gen == gen_before {
        app.update();
        assert!(t0.elapsed() < Duration::from_secs(10), "no new netlist");
        std::thread::sleep(Duration::from_millis(2));
    }
    app.sim.send(Command::SetInput {
        pin: "a".into(),
        value: Value::from_bool(false),
    });
    app.sim.send(Command::Step {
        n: 5,
        unit: StepUnit::Tick,
    });
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_millis(300) {
        app.update();
        if let Some(l) = app.live.values() {
            let v = l.map.pin_value("out1", l.values).unwrap();
            assert_ne!(
                v.to_u64(),
                Some(1),
                "disconnected output still follows the input"
            );
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn kernel_rejection_rolls_the_action_back() {
    let mut design = empty_design();
    design.defs[0].pins.push(panda_netlist::PinDecl {
        name: "a".into(),
        dir: Dir::In,
        width: 1,
    });
    let mut doc = Document::new(design);
    doc.layout_missing();
    let mut app = AppState::from_document(doc, SimOptions::default());
    let mut ed = Editor::new();
    let def = app.view.def.clone();
    let at_a = app_port(&app, "a").pos;
    ed.place(
        &mut app,
        &def,
        &PlaceRequest::Atom(Atom::Not { width: 1 }),
        at_a + gp(10, 4),
    )
    .unwrap();
    assert!(pump(&mut app, &mut ed));
    let before = app.doc.design.clone();
    // Driving the top-level input pin from a gate output passes the local checks but the
    // kernel's validation rejects it ("input pin driven from inside").
    app_wire(&mut ed, &mut app, "not1.out", "a").unwrap();
    assert_eq!(top(&app.doc).nets.len(), 1);
    assert!(!pump(&mut app, &mut ed));
    assert_eq!(app.doc.design, before, "rolled back");
    assert_eq!(ed.history.undo_len(), 1);
    assert_eq!(ed.history.redo_len(), 0);
    assert!(
        ed.message.as_deref().unwrap().contains("rejected"),
        "{:?}",
        ed.message
    );
}
