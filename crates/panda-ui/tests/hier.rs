//! Hierarchy through the real app state and sim thread (no GPU): enter/exit value mapping,
//! packaging in the running app (with undo/redo through the edit module's history), library
//! placement through the edit module's placement API, and every built-in example loading and
//! simulating.

use std::time::{Duration, Instant};

use panda_kernel::StepUnit;
use panda_netlist::Value;
use panda_ui::app::AppState;
use panda_ui::examples;
use panda_ui::hier::{HierState, actions};
use panda_ui::simhost::{Command, SimEvent, SimOptions};

/// Pump `app.update()` until `pred`; returns the events seen.
fn pump(
    app: &mut AppState,
    secs: f64,
    mut pred: impl FnMut(&AppState, &[SimEvent]) -> bool,
) -> Vec<SimEvent> {
    let t0 = Instant::now();
    let mut all = Vec::new();
    loop {
        let ev = app.update();
        let done = pred(app, &ev);
        all.extend(ev);
        if done {
            return all;
        }
        assert!(
            t0.elapsed().as_secs_f64() < secs,
            "timed out; log: {:?}",
            app.log
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// Read nets by name on the sim thread (the kernel's own view).
fn kernel_read(app: &mut AppState, names: Vec<String>) -> Vec<Option<Value>> {
    app.sim.send(Command::Custom(Box::new(move |k| {
        let v: Vec<Option<Value>> = names.iter().map(|n| k.read(n)).collect();
        Some(SimEvent::Custom(Box::new(v)))
    })));
    let mut out = None;
    pump(app, 20.0, |_, ev| {
        for e in ev {
            if let SimEvent::Custom(b) = e
                && let Some(v) = b.downcast_ref::<Vec<Option<Value>>>()
            {
                out = Some(v.clone());
            }
        }
        out.is_some()
    });
    out.unwrap()
}

/// Every net of the viewed def: live value (snapshot) == kernel read at the same tick.
fn check_view(app: &mut AppState, path: &str) -> usize {
    let tick = app.live.snapshot.tick;
    pump(app, 20.0, |a, _| {
        a.live.values().is_some_and(|l| l.map.path == path) && a.live.snapshot.tick == tick
    });
    let def = app.doc.def(&app.view.def).unwrap().clone();
    let names: Vec<String> = def
        .nets
        .iter()
        .map(|n| {
            if path.is_empty() {
                n.id.clone()
            } else {
                format!("{path}/{}", n.id)
            }
        })
        .collect();
    let live: Vec<Option<Value>> = {
        let l = app.live.values().unwrap();
        def.nets
            .iter()
            .map(|n| l.map.net_value(&n.id, l.values))
            .collect()
    };
    let want = kernel_read(app, names.clone());
    for ((n, l), w) in names.iter().zip(&live).zip(&want) {
        assert!(w.is_some(), "{n} unknown to the kernel");
        assert_eq!(l, w, "{n} at tick {tick}");
    }
    names.len()
}

fn started(design: panda_netlist::Design) -> AppState {
    let mut app = AppState::new(design, SimOptions::default());
    app.wait_layout();
    pump(&mut app, 30.0, |a, _| a.live.flat.is_some());
    app
}

fn step_paused(app: &mut AppState, n: u64, unit: StepUnit) {
    let t = app.live.snapshot.tick;
    app.sim.send(Command::Step { n, unit });
    pump(app, 60.0, |a, _| {
        a.live.snapshot.tick > t && !a.live.snapshot.running
    });
    // Let a final publication land.
    let seq = app.live.snapshot.seq;
    pump(app, 5.0, |a, _| {
        a.live.snapshot.seq > seq || a.live.snapshot.seq > 0
    });
}

#[test]
fn enter_exit_live_values_match_the_instance() {
    let mut app = started(examples::counter());
    app.sim.send(Command::SetInput {
        pin: "EN".into(),
        value: Value::from_bool(true),
    });
    step_paused(&mut app, 9, StepUnit::Cycle);
    let mut checked = 0;
    for path in ["", "dut", "dut/reg", "dut/inc", "dut/reg/b2"] {
        if panda_ui::app::def_at_path(&app.doc.design, path).is_none() {
            continue;
        }
        assert!(app.enter(path), "{path}");
        checked += check_view(&mut app, path);
    }
    assert!(checked > 20, "{checked}");
    // Exit goes to the parent and selects the instance just left.
    assert!(app.enter("dut/reg"));
    assert!(actions::exit(&mut app));
    assert_eq!(app.view.path, "dut");
    assert!(
        app.highlights
            .items
            .contains(&panda_ui::doc::ItemRef::Instance("reg".into()))
    );
    assert!(actions::exit(&mut app));
    assert_eq!(app.view.path, "");
    assert!(!actions::exit(&mut app));
    // Enter by instance id of the current view.
    assert!(actions::enter_instance(&mut app, "dut"));
    assert_eq!(app.view.path, "dut");
    assert!(!actions::enter_instance(&mut app, "nope"));
}

#[test]
fn package_in_running_app_then_undo() {
    let mut app = started(examples::counter());
    let mut hs = HierState::new();
    let original = app.doc.design.clone();
    app.sim.send(Command::SetInput {
        pin: "EN".into(),
        value: Value::from_bool(true),
    });
    step_paused(&mut app, 3, StepUnit::Cycle);
    assert!(app.enter("dut"));
    let sel: Vec<String> = app
        .doc
        .def(&app.view.def)
        .unwrap()
        .instances
        .iter()
        .filter(|i| i.id != "reg")
        .map(|i| i.id.clone())
        .collect();
    let inst = actions::create_package(&mut hs, None, &mut app, &sel, "incr").unwrap();
    assert_eq!(inst, "incr");
    let ev = pump(&mut app, 20.0, |_, ev| {
        ev.iter().any(|e| matches!(e, SimEvent::Edited { .. }))
    });
    assert!(
        ev.iter()
            .any(|e| matches!(e, SimEvent::Edited { result: Ok(()), .. })),
        "{ev:?}"
    );
    // The kernel keeps simulating the edited design; values inside the new package are its own.
    step_paused(&mut app, 4, StepUnit::Cycle);
    assert!(actions::enter_instance(&mut app, "incr"));
    assert_eq!(app.view.path, "dut/incr");
    assert!(check_view(&mut app, "dut/incr") >= 2);
    assert!(app.doc.editor("incr").is_some());
    // Undo restores the original structure (document and kernel).
    app.enter("dut");
    actions::undo(&mut hs, &mut app);
    let ev = pump(&mut app, 20.0, |_, ev| {
        ev.iter().any(|e| matches!(e, SimEvent::Edited { .. }))
    });
    assert!(
        ev.iter()
            .any(|e| matches!(e, SimEvent::Edited { result: Ok(()), .. }))
    );
    assert_eq!(app.doc.design, original);
    assert!(app.doc.editor("incr").is_none());
    actions::redo(&mut hs, &mut app);
    assert!(app.doc.def("incr").is_some());
    let ev = pump(&mut app, 20.0, |_, ev| {
        ev.iter().any(|e| matches!(e, SimEvent::Edited { .. }))
    });
    assert!(
        ev.iter()
            .any(|e| matches!(e, SimEvent::Edited { result: Ok(()), .. }))
    );
    step_paused(&mut app, 2, StepUnit::Cycle);
    assert!(app.enter("dut/incr"));
    assert!(check_view(&mut app, "dut/incr") >= 2);
}

#[test]
fn library_entries_place_through_the_edit_module() {
    use panda_ui::edit::{Editor, PlaceRequest};
    use panda_ui::geom::gp;
    let mut app = started(examples::counter());
    let mut hs = HierState::new();
    let top = app.doc.design.top.clone();
    let n0 = app.doc.def(&top).unwrap().instances.len();
    let mut ed = Editor::new();
    // A parameterized block: its defs are added (one undoable action), then placed.
    let mut entry = panda_ui::hier::library::catalog_entries()
        .into_iter()
        .find(|e| e.name == "mux")
        .unwrap();
    entry.params[0].value = 1;
    entry.params[1].value = 4;
    let req = actions::arm_entry(&mut hs, None, &mut app, &entry).unwrap();
    let PlaceRequest::Def(id) = &req else {
        panic!("{req:?}")
    };
    assert!(app.doc.def(id).is_some(), "{id}");
    ed.place(&mut app, &top, &req, gp(60, 4)).unwrap();
    // An atom with parameters, and a user def of the document.
    let mut and = hs
        .builtin
        .iter()
        .find(|e| e.name == "NAND")
        .unwrap()
        .clone();
    and.params[0].value = 3;
    and.params[1].value = 8;
    let req = actions::arm_entry(&mut hs, None, &mut app, &and).unwrap();
    assert!(matches!(
        &req,
        PlaceRequest::Atom(panda_netlist::Atom::Gate {
            inputs: 3,
            width: 8,
            ..
        })
    ));
    ed.place(&mut app, &top, &req, gp(60, 20)).unwrap();
    let user = panda_ui::hier::library::user_entries(&app.doc.design);
    let reg = user.iter().find(|e| e.name == "register_4").unwrap();
    let req = actions::arm_entry(&mut hs, None, &mut app, reg).unwrap();
    ed.place(&mut app, &top, &req, gp(60, 30)).unwrap();
    let def = app.doc.def(&top).unwrap();
    assert_eq!(def.instances.len(), n0 + 3);
    // Sources only go into the top def; nothing may be placed into itself.
    assert!(app.enter("dut"));
    let clk = hs
        .builtin
        .iter()
        .find(|e| e.name == "Clock")
        .unwrap()
        .clone();
    assert!(actions::arm_entry(&mut hs, None, &mut app, &clk).is_err());
    let user = panda_ui::hier::library::user_entries(&app.doc.design);
    let me = user
        .iter()
        .find(|e| e.name == app.view.def)
        .unwrap()
        .clone();
    assert!(actions::arm_entry(&mut hs, None, &mut app, &me).is_err());
    // The kernel accepted all of it and keeps simulating.
    let mut answered = 0;
    pump(&mut app, 20.0, |_, ev| {
        for e in ev {
            if let SimEvent::Edited { result, .. } = e {
                assert!(result.is_ok(), "{result:?}");
                answered += 1;
            }
        }
        answered >= 4
    });
    app.enter("");
    step_paused(&mut app, 2, StepUnit::Cycle);
    assert!(check_view(&mut app, "") > 0);
    // Undo of the block's defs (fallback history) after the placements were undone (editor).
    ed.undo(&mut app).unwrap();
    ed.undo(&mut app).unwrap();
    ed.undo(&mut app).unwrap();
    assert_eq!(app.doc.def(&top).unwrap().instances.len(), n0);
    actions::undo(&mut hs, &mut app);
    assert!(app.doc.def(id).is_none());
}

#[test]
fn every_example_loads_lays_out_and_simulates() {
    for e in examples::all() {
        let t0 = Instant::now();
        let mut app = started((e.build)());
        assert!(app.doc.defs_missing_layout().is_empty(), "{}", e.name);
        let laid = t0.elapsed();
        let top = app.doc.design.top_def().unwrap().clone();
        let unit = if e.name.starts_with("SR") {
            StepUnit::Tick
        } else {
            StepUnit::Cycle
        };
        step_paused(&mut app, 20, unit);
        assert!(app.live.snapshot.tick > 0, "{}", e.name);
        assert!(
            !app.log.iter().any(|l| l.contains("failed")),
            "{}: {:?}",
            e.name,
            app.log
        );
        // Outputs are readable through the live view.
        pump(&mut app, 20.0, |a, _| a.live.values().is_some());
        let l = app.live.values().unwrap();
        for p in top.pins.iter().filter(|p| p.dir == panda_netlist::Dir::Out) {
            assert!(
                l.map.pin_value(&p.name, l.values).is_some(),
                "{}: {}",
                e.name,
                p.name
            );
        }
        eprintln!(
            "{}: {} defs, loaded + laid out in {:.0} ms, tick {}",
            e.name,
            app.doc.design.defs.len(),
            laid.as_secs_f64() * 1e3,
            app.live.snapshot.tick
        );
    }
}

#[test]
fn panda8_programs_run_their_program() {
    // fibonacci halts after outputting 233; the output pin must reach it.
    let mut app = started(examples::panda8_program(&panda_cpu8::programs::FIBONACCI));
    app.sim.send(Command::SetMode(panda_kernel::Mode::Fast));
    step_paused(&mut app, 400, StepUnit::Cycle);
    let top = app.doc.design.top_def().unwrap().clone();
    let outs: Vec<String> = top
        .pins
        .iter()
        .filter(|p| p.dir == panda_netlist::Dir::Out)
        .map(|p| p.name.clone())
        .collect();
    let vals = kernel_read(&mut app, outs.clone());
    let any_233 = vals
        .iter()
        .any(|v| v.as_ref().and_then(Value::to_u64) == Some(233));
    assert!(any_233, "{outs:?} = {vals:?}");
}
