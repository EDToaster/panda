//! Incremental edits with state carry-over, in both modes.

mod common;

use common::*;
use panda_circuits::{sync_counter, testbench};
use panda_cpu8::asm::assemble;
use panda_cpu8::{RomStyle, Variant};
use panda_kernel::{EditOp, EngineKind, FastBackend, Kernel, Mode, StepUnit};
use panda_netlist::{
    Atom, Design, DesignBuilder, Endpoint, GateOp, InstKind, Instance, Logic, Net, PinDecl,
    Severity, Value,
};
use panda_netlist::{Dir, Termination};

const PERIOD: u32 = 16;

fn counter_design() -> Design {
    let mut db = DesignBuilder::new();
    let c = sync_counter(&mut db, 4);
    let tb = testbench(&mut db, &c, PERIOD);
    db.build(&tb)
}

/// Id of the net of `def` that has endpoint `ep`.
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

/// Ops adding `qand = Q[0] & Q[1]` (gate `g_and`, splitter `qs`, output pin `AND01`) to the top.
fn add_gate_ops(design: &Design) -> Vec<EditOp> {
    let top = design.top.clone();
    let qnet = net_with(design, &top, &Endpoint::Pin("Q".into()));
    vec![
        EditOp::AddInstance {
            def: top.clone(),
            instance: Instance {
                id: "qs".into(),
                kind: InstKind::Atom(Atom::Splitter {
                    width: 4,
                    parts: vec![(0, 0), (1, 1), (2, 2), (3, 3)],
                }),
            },
        },
        EditOp::Connect {
            def: top.clone(),
            net: qnet,
            endpoint: port("qs", "whole"),
        },
        EditOp::AddInstance {
            def: top.clone(),
            instance: Instance {
                id: "g_and".into(),
                kind: InstKind::Atom(Atom::Gate {
                    op: GateOp::And,
                    inputs: 2,
                    width: 1,
                    reduce: false,
                }),
            },
        },
        EditOp::AddNet {
            def: top.clone(),
            net: net("b0", 1, vec![port("qs", "p0"), port("g_and", "in0")]),
        },
        EditOp::AddNet {
            def: top.clone(),
            net: net("b1", 1, vec![port("qs", "p1"), port("g_and", "in1")]),
        },
        EditOp::AddNet {
            def: top.clone(),
            net: net("b2", 1, vec![port("qs", "p2")]),
        },
        EditOp::AddPin {
            def: top.clone(),
            pin: PinDecl {
                name: "AND01".into(),
                dir: Dir::Out,
                width: 1,
            },
        },
        EditOp::AddNet {
            def: top,
            net: net(
                "qand",
                1,
                vec![port("g_and", "out"), Endpoint::Pin("AND01".into())],
            ),
        },
    ]
}

fn counter_kernel(mode: Mode) -> Kernel {
    let mut k = Kernel::new(Default::default());
    k.set_mode(mode);
    k.load(counter_design()).unwrap();
    k.wait_ready().unwrap();
    k.set_input("EN", &Value::from_bool(true)).unwrap();
    k
}

fn q(k: &Kernel) -> u64 {
    k.read_u64("Q")
        .unwrap_or_else(|| panic!("Q = {:?}", k.read("Q")))
}

fn edits_with_carry_over(mode: Mode) {
    let mut k = counter_kernel(mode);
    k.step(10, StepUnit::Cycle);
    let q0 = q(&k);
    let t0 = k.tick();
    let design = k.design().unwrap().clone();

    // Add a gate: state is kept, the new gate computes from the carried values.
    k.edit(&add_gate_ops(&design)).unwrap();
    assert_eq!(k.tick(), t0, "edit keeps the time");
    assert_eq!(q(&k), q0, "counter state carried over by name");
    if mode == Mode::Fast {
        assert_eq!(k.engine(), EngineKind::TimingFallback);
        k.wait_ready().unwrap();
        assert_eq!(k.engine(), EngineKind::Fast(FastBackend::default()));
        assert_eq!(q(&k), q0);
    }
    for i in 1..=6 {
        k.step(1, StepUnit::Cycle);
        let v = q(&k);
        assert_eq!(v, (q0 + i) % 16, "counter keeps counting after the edit");
        assert_eq!(
            k.read_u64("AND01"),
            Some(v & (v >> 1) & 1),
            "new gate output, Q={v}"
        );
        assert_eq!(k.read_u64("qand"), Some(v & (v >> 1) & 1));
    }

    // Rewire: g_and.in1 from Q[1] to Q[2].
    let top = design.top.clone();
    let q1 = q(&k);
    k.edit(&[
        EditOp::Disconnect {
            def: top.clone(),
            net: "b1".into(),
            endpoint: port("g_and", "in1"),
        },
        EditOp::Connect {
            def: top.clone(),
            net: "b2".into(),
            endpoint: port("g_and", "in1"),
        },
    ])
    .unwrap();
    assert_eq!(q(&k), q1);
    for i in 1..=6 {
        k.step(1, StepUnit::Cycle);
        let v = q(&k);
        assert_eq!(v, (q1 + i) % 16);
        assert_eq!(
            k.read_u64("AND01"),
            Some(v & (v >> 2) & 1),
            "rewired gate, Q={v}"
        );
    }

    // Change atom parameters: AND → XOR.
    k.edit(&[EditOp::SetAtom {
        def: top.clone(),
        instance: "g_and".into(),
        atom: Atom::Gate {
            op: GateOp::Xor,
            inputs: 2,
            width: 1,
            reduce: false,
        },
    }])
    .unwrap();
    k.step(1, StepUnit::Cycle);
    let v = q(&k);
    assert_eq!(k.read_u64("AND01"), Some((v ^ (v >> 2)) & 1));

    // A failing batch rolls back atomically (the first op alone would be fine).
    let before = k.design().unwrap().clone();
    let (tick, val) = (k.tick(), q(&k));
    let err = k
        .edit(&[
            EditOp::RemoveNet {
                def: top.clone(),
                id: "qand".into(),
            },
            EditOp::Connect {
                def: top.clone(),
                net: "b2".into(),
                endpoint: Endpoint::Pin("Q".into()),
            },
        ])
        .unwrap_err();
    assert!(
        err.iter().all(|d| d.severity == Severity::Error) && !err.is_empty(),
        "{err:?}"
    );
    assert_eq!(k.design().unwrap(), &before);
    assert_eq!((k.tick(), q(&k)), (tick, val));
    assert!(k.read("qand").is_some());
    // Unknown targets are reported, nothing changes.
    assert!(
        k.edit(&[EditOp::RemoveInstance {
            def: top.clone(),
            id: "nope".into()
        }])
        .is_err()
    );
    assert!(k.edit(&[EditOp::SetTop("nope".into())]).is_err());
    assert_eq!(k.design().unwrap(), &before);

    // Removing the gate again: its nets disappear, the rest keeps running.
    k.edit(&[
        EditOp::RemoveInstance {
            def: top.clone(),
            id: "g_and".into(),
        },
        EditOp::RemoveNet {
            def: top.clone(),
            id: "qand".into(),
        },
        EditOp::RemovePin {
            def: top.clone(),
            name: "AND01".into(),
        },
    ])
    .unwrap();
    k.wait_ready().unwrap();
    assert!(k.read("qand").is_none());
    let v = q(&k);
    k.step(1, StepUnit::Cycle);
    assert_eq!(q(&k), (v + 1) % 16);
}

#[test]
fn edits_timing() {
    edits_with_carry_over(Mode::Timing);
}

#[test]
fn edits_fast() {
    edits_with_carry_over(Mode::Fast);
}

#[test]
fn fast_mode_runs_on_timing_until_compiled() {
    let mut k = counter_kernel(Mode::Fast);
    k.step(5, StepUnit::Cycle);
    let q0 = q(&k);
    let design = k.design().unwrap().clone();
    k.edit(&add_gate_ops(&design)).unwrap();
    // Steps before the compiler is done run on the Timing engine at phase granularity.
    for i in 1..=3 {
        k.step(1, StepUnit::Cycle);
        assert_eq!(q(&k), (q0 + i) % 16);
    }
    // An edit while compiling: the stale result is dropped.
    k.edit(&[EditOp::SetAtom {
        def: design.top.clone(),
        instance: "g_and".into(),
        atom: Atom::Gate {
            op: GateOp::Or,
            inputs: 2,
            width: 1,
            reduce: false,
        },
    }])
    .unwrap();
    k.wait_ready().unwrap();
    assert_eq!(k.engine(), EngineKind::Fast(FastBackend::default()));
    k.step(1, StepUnit::Cycle);
    let v = q(&k);
    assert_eq!(v, (q0 + 4) % 16);
    assert_eq!(k.read_u64("AND01"), Some((v | (v >> 1)) & 1));
}

#[test]
fn synchronous_compilation() {
    let mut k = Kernel::new(panda_kernel::KernelOptions {
        background: false,
        ..Default::default()
    });
    k.load(counter_design()).unwrap();
    k.set_mode(Mode::Fast);
    assert_eq!(k.engine(), EngineKind::Fast(FastBackend::default()));
    k.set_input("EN", &Value::from_bool(true)).unwrap();
    k.step(12, StepUnit::Cycle);
    let design = k.design().unwrap().clone();
    let q0 = q(&k);
    k.edit(&add_gate_ops(&design)).unwrap();
    assert_eq!(k.engine(), EngineKind::Fast(FastBackend::default()));
    assert_eq!(q(&k), q0);
    assert_eq!(k.read_u64("AND01"), Some(q0 & (q0 >> 1) & 1));
}

#[test]
fn new_nets_start_x_and_kernels_fall_back() {
    // Replace the counter's register def by a structurally identical copy under new instance
    // names: the nets inside get new names, so their state is lost (X) — Fast mode falls back to
    // X for the inconsistent kernels and reports it.
    let mut k = counter_kernel(Mode::Fast);
    k.step(6, StepUnit::Cycle);
    assert!(k.read_u64("Q").is_some());
    let design = k.design().unwrap().clone();
    let def = design.def("sync_counter_4").unwrap().clone();
    let mut renamed = def.clone();
    for i in &mut renamed.instances {
        if i.id == "reg" {
            i.id = "reg2".into();
        }
    }
    for n in &mut renamed.nets {
        for e in &mut n.ends {
            if let Endpoint::Port { inst, .. } = e
                && inst == "reg"
            {
                *inst = "reg2".into();
            }
        }
    }
    k.edit(&[EditOp::ReplaceDef(renamed)]).unwrap();
    k.wait_ready().unwrap();
    // The register outputs are X now (Q pins of dut/reg2 are new nets), and stay X: EN=1 counts
    // from X. A reset is the fix.
    assert_eq!(
        k.read("dut/reg2/Q")
            .map(|v| v.bits().all(|b| b == Logic::X)),
        Some(true)
    );
    k.step(2, StepUnit::Cycle);
    assert!(k.read_u64("Q").is_none());
}

fn rom_def_of(design: &Design) -> panda_netlist::ComponentDef {
    design
        .defs
        .iter()
        .find(|d| d.id.starts_with("panda8_sop_rom_"))
        .unwrap()
        .clone()
}

#[test]
fn rom_change_in_both_modes() {
    let a = assemble("LDI r0, 5\nloop: OUT r0\nJMP loop\n").unwrap();
    let b = assemble("LDI r0, 5\nloop: OUT r0\nINC r0\nJMP loop\n").unwrap();
    let (da, opts) = cpu_design(&a, Variant::Hierarchical, RomStyle::SumOfProducts);
    let (db, _) = cpu_design(&b, Variant::Hierarchical, RomStyle::SumOfProducts);
    let rom_a = rom_def_of(&da);
    let mut rom_b = rom_def_of(&db);
    rom_b.id = rom_a.id.clone();

    let mut kernels: Vec<Kernel> = [Mode::Timing, Mode::Fast]
        .into_iter()
        .map(|m| {
            let mut k = Kernel::new(quiet_opts());
            k.set_mode(m);
            k.load(da.clone()).unwrap();
            k.wait_ready().unwrap();
            k.run_to(cycle0_tick(&opts));
            k
        })
        .collect();
    // Run program A until a fetch at PC = 2 (the OUT), well after reset.
    let mut cycles = 0;
    loop {
        for k in &mut kernels {
            k.step(1, StepUnit::Cycle);
        }
        cycles += 1;
        let s = arch(&kernels[0]);
        assert_eq!(s, arch(&kernels[1]));
        if cycles > 10 && s.pc == 2 && !s.execute {
            assert_eq!(s.regs[0], 5);
            break;
        }
    }
    for k in &mut kernels {
        k.edit(&[EditOp::ReplaceDef(rom_b.clone())]).unwrap();
        k.wait_ready().unwrap();
        assert_eq!(arch(k).pc, 2, "CPU state carried over");
    }
    // Program B from PC = 2 with r0 = 5 outputs 5, 6, 7, ...
    let mut outs = vec![Vec::new(), Vec::new()];
    for _ in 0..60 {
        for (k, o) in kernels.iter_mut().zip(&mut outs) {
            k.step(1, StepUnit::Cycle);
            let s = arch(k);
            if s.out_strobe {
                o.push(s.out);
            }
        }
        assert_eq!(arch(&kernels[0]), arch(&kernels[1]));
    }
    assert_eq!(outs[0], outs[1]);
    assert!(outs[0].len() >= 8, "{outs:?}");
    for (i, &v) in outs[0].iter().enumerate() {
        assert_eq!(v, 5 + i as u8, "{:?}", outs[0]);
    }
}
