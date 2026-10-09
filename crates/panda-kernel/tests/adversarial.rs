//! Adversarial facade tests: edit atomicity, state carry-over, mode switches at awkward points,
//! background-compile races, snapshots across modes and backends, diagnostics dedupe.

mod common;

use std::time::Duration;

use common::*;
use panda_circuits::{TestbenchOptions, sync_counter, testbench, testbench_with};
use panda_cpu8::programs;
use panda_cpu8::{RomStyle, Variant};
use panda_difftest::Rng;
use panda_kernel::{
    DiagSource, EditOp, EngineKind, FastBackend, Kernel, KernelOptions, Mode, StepUnit,
};
use panda_netlist::{
    Atom, DefBuilder, Design, DesignBuilder, DiagKind, Dir, Endpoint, InstKind, Instance, Logic,
    Net, PinDecl, Termination, Value,
};

const PERIOD: u32 = 16;

fn counter_design_p(period: u32) -> Design {
    let mut db = DesignBuilder::new();
    let c = sync_counter(&mut db, 4);
    let tb = testbench(&mut db, &c, period);
    db.build(&tb)
}

fn counter_design() -> Design {
    counter_design_p(PERIOD)
}

fn opts(backend: FastBackend, background: bool) -> KernelOptions {
    KernelOptions {
        backend,
        background,
        ..Default::default()
    }
}

fn port(inst: &str, port: &str) -> Endpoint {
    Endpoint::Port {
        inst: inst.into(),
        port: port.into(),
    }
}

fn net(id: &str, width: u32, ends: Vec<Endpoint>) -> Net {
    Net {
        id: id.into(),
        width,
        termination: Termination::None,
        ends,
    }
}

fn net_with(design: &Design, def: &str, ep: &Endpoint) -> String {
    design
        .def(def)
        .unwrap()
        .nets
        .iter()
        .find(|n| n.ends.contains(ep))
        .unwrap()
        .id
        .clone()
}

/// Ops adding output pin `name` = NOT(Q[0]) to the top (ids suffixed with `name`).
fn probe_ops(design: &Design, name: &str) -> Vec<EditOp> {
    let top = design.top.clone();
    let qnet = net_with(design, &top, &Endpoint::Pin("Q".into()));
    let (sp, g, b, o) = (
        format!("sp_{name}"),
        format!("g_{name}"),
        format!("b_{name}"),
        format!("o_{name}"),
    );
    vec![
        EditOp::AddInstance {
            def: top.clone(),
            instance: Instance {
                id: sp.clone(),
                kind: InstKind::Atom(Atom::Splitter {
                    width: 4,
                    parts: vec![(0, 0), (1, 3)],
                }),
            },
        },
        EditOp::Connect {
            def: top.clone(),
            net: qnet,
            endpoint: port(&sp, "whole"),
        },
        EditOp::AddInstance {
            def: top.clone(),
            instance: Instance {
                id: g.clone(),
                kind: InstKind::Atom(Atom::Not { width: 1 }),
            },
        },
        EditOp::AddNet {
            def: top.clone(),
            net: net(&b, 1, vec![port(&sp, "p0"), port(&g, "in")]),
        },
        EditOp::AddPin {
            def: top.clone(),
            pin: PinDecl {
                name: name.into(),
                dir: Dir::Out,
                width: 1,
            },
        },
        EditOp::AddNet {
            def: top,
            net: net(&o, 1, vec![port(&g, "out"), Endpoint::Pin(name.into())]),
        },
    ]
}

/// Ops removing what `probe_ops(name)` added.
fn unprobe_ops(design: &Design, name: &str) -> Vec<EditOp> {
    let top = design.top.clone();
    vec![
        EditOp::RemoveNet {
            def: top.clone(),
            id: format!("o_{name}"),
        },
        EditOp::RemovePin {
            def: top.clone(),
            name: name.into(),
        },
        EditOp::RemoveNet {
            def: top.clone(),
            id: format!("b_{name}"),
        },
        EditOp::RemoveInstance {
            def: top.clone(),
            id: format!("g_{name}"),
        },
        EditOp::RemoveInstance {
            def: top,
            id: format!("sp_{name}"),
        },
    ]
}

fn q(k: &Kernel) -> u64 {
    k.read_u64("Q")
        .unwrap_or_else(|| panic!("Q = {:?} at tick {}", k.read("Q"), k.tick()))
}

fn all_values(k: &Kernel) -> Vec<Option<Value>> {
    let names = k.netlist().unwrap().net_names.clone();
    names.iter().map(|n| k.read(n)).collect()
}

const BACKENDS: [FastBackend; 2] = [FastBackend::Reference, FastBackend::Compiled];

// ---------------------------------------------------------------------------------------------
// Edit atomicity
// ---------------------------------------------------------------------------------------------

/// A failing batch (an op that cannot be applied, or a batch that applies but does not validate)
/// leaves design, time, values, engine and diagnostics untouched, in both modes and while a
/// background compilation is outstanding.
#[test]
fn failed_edits_roll_back_everything() {
    for mode in [Mode::Timing, Mode::Fast] {
        for wait in [true, false] {
            let mut k = Kernel::default();
            k.set_mode(mode);
            k.load(counter_design()).unwrap();
            if wait {
                k.wait_ready().unwrap();
            }
            k.set_input("EN", &Value::from_bool(true)).unwrap();
            k.step(5, StepUnit::Cycle);
            let design = k.design().unwrap().clone();
            let top = design.top.clone();
            let tick = k.tick();
            let values = all_values(&k);
            let engine = k.engine();
            let diags = k.diagnostic_entries().to_vec();

            let mut bad1 = probe_ops(&design, "P");
            bad1.push(EditOp::Connect {
                def: top.clone(),
                net: "no_such_net".into(),
                endpoint: port("g_P", "out"),
            });
            // Applies, but drives a 1-bit port from a 4-bit net (width mismatch).
            let qnet = net_with(&design, &top, &Endpoint::Pin("Q".into()));
            let mut bad2 = probe_ops(&design, "P");
            bad2.push(EditOp::Connect {
                def: top.clone(),
                net: qnet,
                endpoint: port("g_P", "in"),
            });
            // Removing a def that is still instantiated.
            let bad3 = vec![EditOp::RemoveDef("sync_counter_4".into())];
            // Duplicate instance id inside one batch.
            let mut bad4 = probe_ops(&design, "P");
            bad4.extend(probe_ops(&design, "P"));
            for (i, bad) in [bad1, bad2, bad3, bad4].iter().enumerate() {
                let r = k.edit(bad);
                assert!(r.is_err(), "{mode:?} batch {i} should fail");
                assert_eq!(k.design().unwrap(), &design, "{mode:?} batch {i}");
                assert_eq!(k.tick(), tick);
                assert_eq!(all_values(&k), values, "{mode:?} batch {i}");
                assert_eq!(k.engine(), engine);
                assert_eq!(k.diagnostic_entries(), &diags[..], "{mode:?} batch {i}");
                assert!(k.read("P").is_none());
            }
            // The kernel keeps working, and a later good edit still applies.
            k.wait_ready().unwrap();
            k.edit(&probe_ops(&design, "P")).unwrap();
            k.wait_ready().unwrap();
            let q0 = q(&k);
            k.step(3, StepUnit::Cycle);
            assert_eq!(q(&k), (q0 + 3) % 16, "{mode:?}");
            assert_eq!(k.read_u64("P"), Some(!q(&k) & 1));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Background compilation races
// ---------------------------------------------------------------------------------------------

/// Rapid edits while compiling: the engine finally serving Fast mode is always built for the
/// newest design (stale results are dropped), and the count is never disturbed.
#[test]
fn edits_while_compiling_never_apply_stale_programs() {
    for backend in BACKENDS {
        let mut k = Kernel::new(opts(backend, true));
        k.set_mode(Mode::Fast);
        k.load(counter_design()).unwrap();
        k.set_input("EN", &Value::from_bool(true)).unwrap();
        k.step(4, StepUnit::Cycle);
        let mut q_expect = q(&k);
        let mut present: Vec<String> = Vec::new();
        let mut rng = Rng::new(7);
        for round in 0..24 {
            let design = k.design().unwrap().clone();
            if !present.is_empty() && rng.below(3) == 0 {
                let name = present.remove(rng.below(present.len()));
                k.edit(&unprobe_ops(&design, &name)).unwrap();
            } else {
                let name = format!("P{round}");
                k.edit(&probe_ops(&design, &name)).unwrap();
                present.push(name);
            }
            // Sometimes wait, sometimes step on whatever engine serves right now.
            match rng.below(4) {
                0 => k.wait_ready().unwrap(),
                1 => {
                    k.set_mode(Mode::Timing);
                    k.set_mode(Mode::Fast);
                }
                _ => {}
            }
            k.step(1, StepUnit::Cycle);
            q_expect = (q_expect + 1) % 16;
            assert_eq!(q(&k), q_expect, "{backend:?} round {round}");
            for p in &present {
                assert_eq!(
                    k.read_u64(p),
                    Some(!q_expect & 1),
                    "{backend:?} round {round} probe {p} engine {:?}",
                    k.engine()
                );
            }
        }
        k.wait_ready().unwrap();
        assert_eq!(k.engine(), EngineKind::Fast(backend));
        // The serving engine has exactly the current design's pins.
        let flat = k.netlist().unwrap().clone();
        for p in &flat.top_outputs {
            assert!(k.read(&p.name).is_some(), "{}", p.name);
        }
        for round in 0..24 {
            let name = format!("P{round}");
            assert_eq!(k.read(&name).is_some(), present.contains(&name), "{name}");
        }
        k.step(2, StepUnit::Cycle);
        assert_eq!(q(&k), (q_expect + 2) % 16);
        // Carrying a settled state across edits that only add/remove logic outside the kernels
        // is exact: no facade warnings (in particular no "not quiescent" for the gates an edit
        // schedules).
        let notes: Vec<_> = k
            .diagnostic_entries()
            .iter()
            .filter(|e| e.source == DiagSource::Kernel)
            .collect();
        assert!(notes.is_empty(), "{backend:?}: {notes:#?}");
    }
}

/// Edit to B and back to A (same content hash as the first design) while compiling: the result
/// of the first compilation of A must not be taken for the newest A (other generation).
#[test]
fn abab_edits_while_compiling() {
    let (design, opts_cpu) = cpu_design(
        &programs::BENCH_LOOP.image(),
        Variant::Hierarchical,
        RomStyle::SumOfProducts,
    );
    let mut k = Kernel::new(quiet_opts());
    k.set_mode(Mode::Fast);
    k.load(design.clone()).unwrap();
    let c0 = cycle0_tick(&opts_cpu);
    k.run_to(c0);
    let want = trace(&programs::BENCH_LOOP.image(), 60);
    let ops = probe_ops_cpu(&design);
    for i in 0..3 {
        k.edit(&ops).unwrap();
        assert!(k.read("PROBE").is_some(), "iteration {i}");
        k.edit(&unprobe_cpu(&design)).unwrap();
        assert!(k.read("PROBE").is_none());
    }
    assert_eq!(k.design().unwrap(), &design);
    k.wait_ready().unwrap();
    assert_eq!(k.engine(), EngineKind::Fast(FastBackend::Compiled));
    assert!(k.read("PROBE").is_none());
    assert_eq!(arch(&k), want[0]);
    for (c, w) in want.iter().enumerate().skip(1) {
        k.step(1, StepUnit::Cycle);
        assert_eq!(arch(&k), *w, "cycle {c}");
    }
}

/// `PROBE` = NOT(clk) on the CPU testbench top.
fn probe_ops_cpu(design: &Design) -> Vec<EditOp> {
    let top = design.top.clone();
    let clk = design
        .def(&top)
        .unwrap()
        .nets
        .iter()
        .find(|n| {
            n.id == "clk"
                || n.ends
                    .iter()
                    .any(|e| matches!(e, Endpoint::Port{port, ..} if port == "CLK"))
        })
        .expect("clock net")
        .id
        .clone();
    vec![
        EditOp::AddInstance {
            def: top.clone(),
            instance: Instance {
                id: "g_probe".into(),
                kind: InstKind::Atom(Atom::Not { width: 1 }),
            },
        },
        EditOp::Connect {
            def: top.clone(),
            net: clk,
            endpoint: port("g_probe", "in"),
        },
        EditOp::AddPin {
            def: top.clone(),
            pin: PinDecl {
                name: "PROBE".into(),
                dir: Dir::Out,
                width: 1,
            },
        },
        EditOp::AddNet {
            def: top,
            net: net(
                "probe",
                1,
                vec![port("g_probe", "out"), Endpoint::Pin("PROBE".into())],
            ),
        },
    ]
}

fn unprobe_cpu(design: &Design) -> Vec<EditOp> {
    let top = design.top.clone();
    vec![
        EditOp::RemoveNet {
            def: top.clone(),
            id: "probe".into(),
        },
        EditOp::RemovePin {
            def: top.clone(),
            name: "PROBE".into(),
        },
        EditOp::RemoveInstance {
            def: top,
            id: "g_probe".into(),
        },
    ]
}

// ---------------------------------------------------------------------------------------------
// Timing-mode input semantics across edits
// ---------------------------------------------------------------------------------------------

fn inverter_chain() -> Design {
    let mut db = DesignBuilder::new();
    let mut b = DefBuilder::new("inv2");
    let a = b.input("A", 1);
    let n = b.not(a);
    let y = b.not(n);
    b.output("Y", y);
    let top = db.add(b.finish());
    db.build(&top)
}

/// An input change given in Timing mode but not applied yet (it applies at the next tick) keeps
/// that timing across an edit: the edited design sees it at the same tick as the original.
#[test]
fn pending_timing_input_survives_edit_with_same_timing() {
    let design = inverter_chain();
    let mut a = Kernel::default();
    let mut b = Kernel::default();
    a.load(design.clone()).unwrap();
    b.load(design.clone()).unwrap();
    a.step(10, StepUnit::Tick);
    b.step(10, StepUnit::Tick);
    a.set_input("A", &Value::from_bool(true)).unwrap();
    b.set_input("A", &Value::from_bool(true)).unwrap();
    let snap = b.snapshot();
    assert_eq!(snap.pending_inputs().len(), 1);
    // A no-op-ish edit (adds an unconnected net) on b.
    b.edit(&[EditOp::AddNet {
        def: "inv2".into(),
        net: net("spare", 1, vec![]),
    }])
    .unwrap();
    for t in 0..5 {
        assert_eq!(a.read("Y"), b.read("Y"), "tick +{t}");
        assert_eq!(a.tick(), b.tick());
        a.step(1, StepUnit::Tick);
        b.step(1, StepUnit::Tick);
    }
    assert_eq!(b.read_u64("Y"), Some(1));
}

// ---------------------------------------------------------------------------------------------
// Width changes and renames
// ---------------------------------------------------------------------------------------------

fn not_design(width: u32) -> Design {
    let mut db = DesignBuilder::new();
    let mut b = DefBuilder::new("top");
    let a = b.input("A", width);
    let y = b.not(a);
    let y = b.name(y, "y");
    b.output("Y", y);
    let top = db.add(b.finish());
    db.build(&top)
}

/// Replacing the top with a wider version: the input of the old width is dropped (reads 0), the
/// new nets compute correctly in both modes and backends, and nothing panics.
#[test]
fn width_change_edit() {
    for backend in BACKENDS {
        for mode in [Mode::Timing, Mode::Fast] {
            let mut k = Kernel::new(opts(backend, false));
            k.set_mode(mode);
            k.load(not_design(4)).unwrap();
            k.set_input("A", &Value::from_u64(4, 5)).unwrap();
            k.step(1, StepUnit::Phase);
            assert_eq!(k.read_u64("Y"), Some(0b1010), "{mode:?}");
            let snap4 = k.snapshot();
            let wide = not_design(8)
                .defs
                .into_iter()
                .find(|d| d.id == "top")
                .unwrap();
            k.edit(&[EditOp::ReplaceDef(wide)]).unwrap();
            assert_eq!(k.input("A"), Some(Value::zeros(8)), "{mode:?}");
            k.step(1, StepUnit::Phase);
            assert_eq!(k.read_u64("Y"), Some(0xFF), "{backend:?} {mode:?}");
            k.set_input("A", &Value::from_u64(8, 0x0F)).unwrap();
            k.step(1, StepUnit::Phase);
            assert_eq!(k.read_u64("Y"), Some(0xF0), "{backend:?} {mode:?}");
            // A snapshot of the 4-bit design restores into the 8-bit one by name: the input of
            // the wrong width is dropped, carried bits keep their values, then everything
            // settles from the inputs.
            k.restore(&snap4).unwrap();
            k.step(1, StepUnit::Phase);
            assert_eq!(k.read_u64("Y"), Some(0xFF), "{backend:?} {mode:?}");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Mode switches at awkward points
// ---------------------------------------------------------------------------------------------

/// Timing → Fast where the Timing engine is not quiescent at the phase end (clock far below the
/// STA minimum): the switch happens (defined behaviour: Fast re-settles from the net values,
/// inconsistent kernels restart from X) and a warning says the Timing state was not quiescent.
#[test]
fn non_quiescent_timing_to_fast_warns() {
    for backend in BACKENDS {
        let mut k = Kernel::new(opts(backend, false));
        k.load(counter_design_p(4)).unwrap();
        k.set_input("EN", &Value::from_bool(true)).unwrap();
        k.step(30, StepUnit::Tick);
        k.set_mode(Mode::Fast);
        assert_eq!(k.engine(), EngineKind::Fast(backend));
        let warned = k.diagnostic_entries().iter().any(|e| {
            e.source == DiagSource::Kernel && e.diagnostic.message.contains("not quiescent")
        });
        assert!(warned, "{:#?}", k.diagnostic_entries());
        // Still usable.
        k.step(4, StepUnit::Cycle);
        k.set_mode(Mode::Timing);
        k.step(4, StepUnit::Cycle);
    }
}

/// A Timing snapshot taken in the middle of a phase (right after a clock edge, while the edge is
/// still propagating) restored in Fast mode: Fast mode cannot load a non-settled state, so the
/// facade lets the Timing semantics finish the phase first. The result equals a Timing kernel
/// that restored the same snapshot and stepped to the phase end, for every net (period 64: every
/// phase settles, so the phase-end state is quiescent).
#[test]
fn mid_phase_timing_snapshot_restored_in_fast_mode() {
    for backend in BACKENDS {
        let mut t = Kernel::new(opts(backend, false));
        t.load(counter_design_p(64)).unwrap();
        t.set_input("EN", &Value::from_bool(true)).unwrap();
        t.step(5, StepUnit::Cycle);
        for d in 1..6 {
            let mut tt = Kernel::new(opts(backend, false));
            tt.load(counter_design_p(64)).unwrap();
            tt.restore(&t.snapshot()).unwrap();
            tt.step(d, StepUnit::Tick); // d ticks past the rising edge
            let snap = tt.snapshot();
            let mut f = Kernel::new(opts(backend, false));
            f.set_mode(Mode::Fast);
            f.load(counter_design_p(64)).unwrap();
            f.restore(&snap).unwrap();
            tt.step(1, StepUnit::Phase);
            assert_eq!(f.tick(), tt.tick(), "{backend:?} +{d}");
            let names = f.netlist().unwrap().net_names.clone();
            let diff: Vec<_> = names
                .iter()
                .filter(|n| f.read(n) != tt.read(n))
                .map(|n| (n.clone(), f.read(n), tt.read(n)))
                .collect();
            assert!(
                diff.is_empty(),
                "{backend:?} +{d}: {diff:?} {:#?}",
                f.diagnostics()
            );
            // Every phase settles at period 64: nothing to report.
            let notes: Vec<_> = f
                .diagnostic_entries()
                .iter()
                .filter(|e| e.source == DiagSource::Kernel)
                .collect();
            assert!(notes.is_empty(), "{backend:?} +{d}: {notes:#?}");
            // And they keep agreeing.
            f.step(3, StepUnit::Cycle);
            tt.step(3, StepUnit::Cycle);
            assert_eq!(q(&f), q(&tt));
        }
    }
}

/// A design with contention only during reset (two tri-state drivers, one enabled by the
/// power-on reset). A Fast engine built at tick 0 sees the contention at its power-on event; when
/// it is adopted later (after the reset) that stale power-on diagnostic must not appear as a
/// Fast-mode diagnostic of the current tick.
#[test]
fn stale_power_on_diagnostics_of_new_engines_are_dropped() {
    let mut db = DesignBuilder::new();
    let mut b = DefBuilder::new("por_bus");
    let r = b.power_on_reset(20);
    let one = b.lit(1, 1);
    let zero = b.lit(1, 0);
    let en = b.lit(1, 1);
    let bus = b.tri(zero, en);
    b.tri_to(one, r, bus);
    let bus = b.name(bus, "bus");
    let clk = b.clock(8, 4, 0);
    let _ = b.name(clk, "clk");
    b.output("BUS", bus);
    let top = db.add(b.finish());
    let design = db.build(&top);
    for backend in BACKENDS {
        // Fast engine powered on at tick 0 (contention), adopted at tick > 20.
        let mut k = Kernel::new(opts(backend, true));
        k.load(design.clone()).unwrap();
        k.precompile();
        k.step(10, StepUnit::Cycle);
        k.wait_ready().unwrap();
        assert!(k.tick() > 30);
        assert!(
            k.diagnostic_entries().iter().any(
                |e| e.diagnostic.kind == DiagKind::Contention && e.source == DiagSource::Timing
            ),
            "the Timing engine sees the reset-time contention"
        );
        k.set_mode(Mode::Fast);
        assert_eq!(k.engine(), EngineKind::Fast(backend));
        k.step(3, StepUnit::Cycle);
        let stale: Vec<_> = k
            .diagnostic_entries()
            .iter()
            .filter(|e| e.diagnostic.kind == DiagKind::Contention && e.source == DiagSource::Fast)
            .collect();
        assert!(stale.is_empty(), "{backend:?}: {stale:#?}");
        assert_eq!(k.read_u64("BUS"), Some(0));
    }
}

/// Facade warnings that repeat (kernels restarting from X on every restore of a poked snapshot)
/// are one entry with a count, not one entry per tick.
#[test]
fn repeated_facade_warnings_are_deduplicated() {
    for backend in BACKENDS {
        let mut k = Kernel::new(opts(backend, false));
        k.set_mode(Mode::Fast);
        k.load(counter_design()).unwrap();
        k.set_input("EN", &Value::from_bool(true)).unwrap();
        k.step(3, StepUnit::Cycle);
        for i in 0..5 {
            k.step(1 + i, StepUnit::Cycle);
            let mut s = k.snapshot();
            // Poke an internal latch net into a state inconsistent with its inputs.
            let name = k
                .netlist()
                .unwrap()
                .net_names
                .iter()
                .find(|n| n.starts_with("dut/reg/"))
                .unwrap()
                .clone();
            let v = s.get(&name).unwrap();
            s.set(
                &name,
                if v == Logic::One {
                    Logic::Zero
                } else {
                    Logic::One
                },
            );
            k.restore(&s).unwrap();
        }
        let entries: Vec<_> = k
            .diagnostic_entries()
            .iter()
            .filter(|e| e.source == DiagSource::Kernel)
            .collect();
        assert!(entries.len() <= 1, "{backend:?}: {entries:#?}");
    }
}

// ---------------------------------------------------------------------------------------------
// Backends
// ---------------------------------------------------------------------------------------------

/// The Compiled and Reference backends behave identically through the facade (all net values
/// after every cycle, through mode switches, edits and snapshot restores) on Panda-8.
#[test]
fn backends_agree_through_the_facade() {
    let prog = programs::MULTIPLY.image();
    for (variant, rom) in [
        (Variant::Hierarchical, RomStyle::SumOfProducts),
        (Variant::Flat, RomStyle::MuxTree),
    ] {
        let (design, copts) = cpu_design(&prog, variant, rom);
        let mk = |backend| {
            let mut k = Kernel::new(KernelOptions {
                backend,
                background: false,
                ..quiet_opts()
            });
            k.set_mode(Mode::Fast);
            k.load(design.clone()).unwrap();
            k.run_to(cycle0_tick(&copts));
            k
        };
        let mut r = mk(FastBackend::Reference);
        let mut c = mk(FastBackend::Compiled);
        assert_eq!(c.engine(), EngineKind::Fast(FastBackend::Compiled));
        let want = trace(&prog, 200);
        let mut snap = None;
        for cyc in 1..=120usize {
            for k in [&mut r, &mut c] {
                k.step(1, StepUnit::Cycle);
                match cyc {
                    20 => {
                        k.set_mode(Mode::Timing);
                    }
                    30 => k.set_mode(Mode::Fast),
                    50 => {
                        let d = k.design().unwrap().clone();
                        k.edit(&probe_ops_cpu(&d)).unwrap();
                    }
                    _ => {}
                }
            }
            if cyc == 60 {
                snap = Some((c.snapshot(), r.snapshot()));
            }
            if cyc == 90 {
                let (sc, sr) = snap.take().unwrap();
                // Cross restore: each backend restores the other one's snapshot.
                c.restore(&sr).unwrap();
                r.restore(&sc).unwrap();
            }
            let wi = if cyc >= 90 { cyc - 30 } else { cyc };
            assert_eq!(arch(&c), want[wi], "{variant:?} cycle {cyc}");
            assert_eq!(c.tick(), r.tick());
            assert_eq!(all_values(&c), all_values(&r), "{variant:?} cycle {cyc}");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Random operation sequences against a counter model
// ---------------------------------------------------------------------------------------------

/// Random sequences of mode switches, edits, snapshots/restores, waits and steps on a counter
/// (STA-clean, so both modes count identically). Model: Q advances by one per cycle while
/// EN = 1; a restore goes back to the snapshot's count.
#[test]
fn random_operation_sequences() {
    for seed in 0..12u64 {
        let mut rng = Rng::new(seed);
        let backend = BACKENDS[(seed % 2) as usize];
        let background = seed % 3 != 0;
        let mut k = Kernel::new(opts(backend, background));
        if rng.below(2) == 0 {
            k.set_mode(Mode::Fast);
        }
        k.load(counter_design()).unwrap();
        k.set_input("EN", &Value::from_bool(true)).unwrap();
        k.step(3, StepUnit::Cycle);
        let mut model = q(&k);
        let mut snaps: Vec<(panda_kernel::State, u64)> = Vec::new();
        let mut probes: Vec<String> = Vec::new();
        for op in 0..80 {
            match rng.below(9) {
                0 => {
                    let m = if k.mode() == Mode::Fast {
                        Mode::Timing
                    } else {
                        Mode::Fast
                    };
                    k.set_mode(m);
                }
                1 => {
                    let d = k.design().unwrap().clone();
                    if !probes.is_empty() && rng.below(2) == 0 {
                        let p = probes.remove(rng.below(probes.len()));
                        k.edit(&unprobe_ops(&d, &p)).unwrap();
                    } else {
                        let p = format!("P{op}");
                        k.edit(&probe_ops(&d, &p)).unwrap();
                        probes.push(p);
                    }
                }
                2 => snaps.push((k.snapshot(), model)),
                3 if !snaps.is_empty() => {
                    let (s, m) = &snaps[rng.below(snaps.len())];
                    k.restore(s).unwrap();
                    model = *m;
                }
                4 => k.wait_ready().unwrap(),
                5 => k.precompile(),
                6 => {
                    let n = 1 + rng.below(4) as u64;
                    k.step(n, StepUnit::Cycle);
                    model = (model + n) % 16;
                }
                7 => {
                    // A whole cycle in phases.
                    k.step(2, StepUnit::Phase);
                    model = (model + 1) % 16;
                }
                _ => {
                    k.step(1, StepUnit::Cycle);
                    model = (model + 1) % 16;
                }
            }
            assert_eq!(
                q(&k),
                model,
                "seed {seed} op {op} mode {:?} engine {:?} tick {}",
                k.mode(),
                k.engine(),
                k.tick()
            );
            for p in &probes {
                if let Some(v) = k.read_u64(p) {
                    assert_eq!(v, !model & 1, "seed {seed} op {op} probe {p}");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// run(budget)
// ---------------------------------------------------------------------------------------------

/// `run` with a zero budget does nothing; `steps` counts only steps that advanced time.
#[test]
fn run_zero_budget_and_step_count() {
    let mut k = Kernel::default();
    k.load(counter_design()).unwrap();
    let t0 = k.tick();
    let s = k.run(Duration::ZERO);
    assert_eq!((s.ticks, s.cycles, s.steps), (0, 0, 0));
    assert_eq!(k.tick(), t0);
    let mut k = Kernel::default();
    k.load(inverter_chain()).unwrap();
    k.step(1, StepUnit::Phase);
    let s = k.run(Duration::from_millis(50));
    assert!(s.idle);
    assert_eq!(s.steps, 0, "{s:?}");
}

/// A clock with a phase offset and a POR: `run` cycles are counted from rising edges, and a run
/// started mid-phase is still exact.
#[test]
fn run_counts_cycles_from_edges() {
    let mut db = DesignBuilder::new();
    let c = sync_counter(&mut db, 4);
    let tb = testbench_with(
        &mut db,
        &c,
        &TestbenchOptions {
            phase: 3,
            high: 5,
            ..TestbenchOptions::new(12)
        },
    );
    let design = db.build(&tb);
    for mode in [Mode::Timing, Mode::Fast] {
        let mut k = Kernel::new(opts(FastBackend::Compiled, false));
        k.set_mode(mode);
        k.load(design.clone()).unwrap();
        k.set_input("EN", &Value::from_bool(true)).unwrap();
        k.step(4, StepUnit::Cycle);
        if mode == Mode::Timing {
            k.step(5, StepUnit::Tick);
        }
        k.step(1, StepUnit::Cycle);
        let q0 = q(&k);
        let s = k.run(Duration::from_millis(30));
        assert!(s.cycles > 0);
        assert_eq!(q(&k), (q0 + s.cycles) % 16, "{mode:?} {s:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------------------------

fn classify_kernel(build: impl FnOnce(&mut DesignBuilder) -> panda_netlist::DefRef) -> Kernel {
    let mut db = DesignBuilder::new();
    let top = build(&mut db);
    let mut k = Kernel::default();
    k.load(db.build(&top)).unwrap();
    k
}

/// A flip-flop on the inverted clock is falling-edge triggered.
#[test]
fn classify_falling_edge_ff() {
    let mut k = classify_kernel(|db| {
        let ff = panda_circuits::ms_dff(db);
        let mut b = DefBuilder::new("negff");
        let d = b.input("D", 1);
        let clk = b.input("CLK", 1);
        let n = b.not(clk);
        let o = b.inst_named("ff", &ff, &[d, n]);
        b.output("Q", o[0]);
        db.add(b.finish())
    });
    let c = k.classify("negff").unwrap();
    assert_eq!(c.kind, panda_kernel::ClassKind::Sequential);
    assert_eq!(c.clock_like_pins, vec!["CLK".to_string()], "{c:?}");
    assert_eq!(
        c.edge_triggered,
        Some(("CLK".to_string(), panda_kernel::Edge::Falling)),
        "{c:?}"
    );
}

/// A bus with a keeper (no gate cycle) holds state: sequential, one state bit, and the enable
/// is not clock-like (the data input matters while it is 1).
#[test]
fn classify_keeper_bus() {
    let mut k = classify_kernel(|db| {
        let mut b = DefBuilder::new("kbus");
        let d = b.input("D", 1);
        let en = b.input("EN", 1);
        let bus = b.tri(d, en);
        b.terminate(bus, Termination::Keeper);
        b.output("Y", bus);
        db.add(b.finish())
    });
    let c = k.classify("kbus").unwrap();
    assert_eq!(c.kind, panda_kernel::ClassKind::Sequential, "{c:?}");
    assert!(c.cyclic);
    assert_eq!(c.state_bits, 1, "{c:?}");
    assert!(c.clock_like_pins.is_empty(), "{c:?}");
}

/// A ring oscillator whose kernel has too many inputs to be characterized (28 > the table
/// limit) is still an oscillator, not merely "sequential".
#[test]
fn classify_wide_uncharacterized_oscillator() {
    let mut k = classify_kernel(|db| {
        let mut b = DefBuilder::new("wide_ring");
        let en = b.input("EN", 28);
        let bits = b.bits(en);
        let n1 = b.wire(1);
        let n2 = b.wire(1);
        let n3 = b.wire(1);
        let mut i1 = vec![n3];
        i1.extend_from_slice(&bits[0..10]);
        let mut i2 = vec![n1];
        i2.extend_from_slice(&bits[10..19]);
        let mut i3 = vec![n2];
        i3.extend_from_slice(&bits[19..28]);
        b.nand_to(&i1, n1);
        b.nand_to(&i2, n2);
        b.nand_to(&i3, n3);
        b.output("Y", n3);
        db.add(b.finish())
    });
    let c = k.classify("wide_ring").unwrap();
    assert_eq!(c.kind, panda_kernel::ClassKind::Oscillator, "{c:?}");
}

/// Throughput of `run` per backend on Panda-8 (bench_loop) and compile latency (`--ignored`,
/// release build).
#[test]
#[ignore]
fn facade_throughput() {
    for (variant, rom) in [
        (Variant::Hierarchical, RomStyle::SumOfProducts),
        (Variant::Hierarchical, RomStyle::MuxTree),
        (Variant::Flat, RomStyle::SumOfProducts),
    ] {
        let (design, copts) = cpu_design(&programs::BENCH_LOOP.image(), variant, rom);
        for backend in BACKENDS {
            let mut k = Kernel::new(KernelOptions {
                backend,
                ..quiet_opts()
            });
            k.set_mode(Mode::Fast);
            let t = std::time::Instant::now();
            k.load(design.clone()).unwrap();
            k.wait_ready().unwrap();
            let ready = t.elapsed();
            k.run_to(cycle0_tick(&copts));
            let s = k.run(Duration::from_millis(500));
            eprintln!(
                "{variant:?} {rom:?} {backend:?}: ready {ready:?}; run {:?}: {} cycles, {:.0} cycles/s",
                s.elapsed,
                s.cycles,
                s.cycles as f64 / s.elapsed.as_secs_f64()
            );
        }
        let mut k = Kernel::new(quiet_opts());
        k.load(design.clone()).unwrap();
        k.run_to(cycle0_tick(&copts));
        let s = k.run(Duration::from_millis(500));
        eprintln!(
            "{variant:?} {rom:?} Timing: {} cycles, {:.0} cycles/s",
            s.cycles,
            s.cycles as f64 / s.elapsed.as_secs_f64()
        );
    }
}
