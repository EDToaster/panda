//! The JIT tier through the facade ([`FastBackend::Jit`]): the native code is hot-swapped in
//! and every tier agrees with the interpreter and the reference engine — per cycle, on every net,
//! across edits (incremental recompilation), snapshots and mode switches.

#![cfg(feature = "jit")]

mod common;

use std::time::Duration;

use common::*;
use panda_cpu8::programs;
use panda_cpu8::{RomStyle, Variant};
use panda_kernel::{EditOp, EngineKind, FastBackend, Kernel, KernelOptions, Mode, StepUnit};
use panda_netlist::{Atom, Dir, Endpoint, InstKind, Instance, Net, PinDecl, Termination, Value};

fn kernel(backend: FastBackend, background: bool) -> Kernel {
    Kernel::new(KernelOptions {
        backend,
        background,
        ..quiet_opts()
    })
}

fn all_values(k: &Kernel) -> Vec<Option<Value>> {
    let names = k.netlist().unwrap().net_names.clone();
    names.iter().map(|n| k.read(n)).collect()
}

#[test]
fn jit_tier_swaps_in_and_matches_every_backend() {
    let prog = programs::MULTIPLY.image();
    for (variant, rom) in [
        (Variant::Hierarchical, RomStyle::SumOfProducts),
        (Variant::GatedClockRegfile, RomStyle::MuxTree),
    ] {
        let (design, opts) = cpu_design(&prog, variant, rom);
        let want = trace(&prog, 200);
        let mut ks: Vec<Kernel> = [
            FastBackend::Reference,
            FastBackend::Compiled,
            FastBackend::Jit,
        ]
        .into_iter()
        .map(|b| {
            let mut k = kernel(b, true);
            k.set_mode(Mode::Fast);
            k.load(design.clone()).unwrap();
            k.wait_ready().unwrap();
            assert_eq!(k.engine(), EngineKind::Fast(b));
            k
        })
        .collect();
        // Run part of the way on the interpreter tier, then wait for the native code.
        let c0 = cycle0_tick(&opts);
        for k in &mut ks {
            k.run_to(c0);
        }
        assert!(
            ks[2].wait_jit(Duration::from_secs(60)),
            "JIT never swapped in"
        );
        assert!(!ks[1].jit_active() && !ks[0].jit_active());
        for (c, w) in want.iter().enumerate() {
            if c > 0 {
                for k in &mut ks {
                    k.step(1, StepUnit::Cycle);
                }
            }
            for k in &ks {
                assert_eq!(arch(k), *w, "{variant:?} cycle {c}");
            }
            if c % 50 == 0 {
                let v = all_values(&ks[0]);
                assert_eq!(all_values(&ks[1]), v, "cycle {c}: compiled vs reference");
                assert_eq!(all_values(&ks[2]), v, "cycle {c}: JIT vs reference");
            }
        }
    }
}

#[test]
fn synchronous_jit_is_active_when_ready() {
    let prog = programs::FIBONACCI.image();
    let (design, opts) = cpu_design(&prog, Variant::Flat, RomStyle::SumOfProducts);
    let mut k = kernel(FastBackend::Jit, false);
    k.set_mode(Mode::Fast);
    k.load(design).unwrap();
    k.wait_ready().unwrap();
    assert!(k.jit_active());
    k.run_to(cycle0_tick(&opts));
    let want = trace(&prog, 100);
    for (c, w) in want.iter().enumerate() {
        if c > 0 {
            k.step(1, StepUnit::Cycle);
        }
        assert_eq!(arch(&k), *w, "cycle {c}");
    }
}

#[test]
fn jit_survives_edits_snapshots_and_mode_switches() {
    let prog = programs::MULTIPLY.image();
    let (design, opts) = cpu_design(&prog, Variant::Hierarchical, RomStyle::SumOfProducts);
    let want = trace(&prog, 160);
    let mut j = kernel(FastBackend::Jit, true);
    let mut r = kernel(FastBackend::Reference, true);
    for k in [&mut j, &mut r] {
        k.set_mode(Mode::Fast);
        k.load(design.clone()).unwrap();
        k.wait_ready().unwrap();
        k.run_to(cycle0_tick(&opts));
    }
    assert!(j.wait_jit(Duration::from_secs(60)));
    let step = |j: &mut Kernel, r: &mut Kernel, from: usize, to: usize| {
        for (c, w) in want.iter().enumerate().take(to + 1).skip(from + 1) {
            j.step(1, StepUnit::Cycle);
            r.step(1, StepUnit::Cycle);
            assert_eq!(arch(j), *w, "cycle {c}");
            assert_eq!(arch(r), *w, "cycle {c}");
        }
        assert_eq!(all_values(j), all_values(r), "cycle {to}");
    };
    step(&mut j, &mut r, 0, 30);
    // Snapshot / restore round trip on the native engine.
    let snap = j.snapshot();
    step(&mut j, &mut r, 30, 40);
    j.restore(&snap).unwrap();
    r.restore(&snap).unwrap();
    assert_eq!(all_values(&j), all_values(&r));
    step(&mut j, &mut r, 30, 60);
    // Mode switch away and back (the compiled engine, with its native code, is kept).
    for k in [&mut j, &mut r] {
        k.set_mode(Mode::Timing);
    }
    step(&mut j, &mut r, 60, 70);
    for k in [&mut j, &mut r] {
        k.set_mode(Mode::Fast);
        k.wait_ready().unwrap();
    }
    step(&mut j, &mut r, 70, 90);
    assert!(j.jit_active());
    // An edit that does not change the architectural state (an inverter probing the clock, on
    // a new output pin): a new program, compiled and swapped in again.
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
                    .any(|e| matches!(e, Endpoint::Port { port, .. } if port == "CLK"))
        })
        .expect("clock net")
        .id
        .clone();
    let ops = vec![
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
            endpoint: Endpoint::Port {
                inst: "g_probe".into(),
                port: "in".into(),
            },
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
            def: top.clone(),
            net: Net {
                id: "probe".into(),
                width: 1,
                termination: Termination::None,
                ends: vec![
                    Endpoint::Port {
                        inst: "g_probe".into(),
                        port: "out".into(),
                    },
                    Endpoint::Pin("PROBE".into()),
                ],
            },
        },
    ];
    for k in [&mut j, &mut r] {
        k.edit(&ops).unwrap();
        k.wait_ready().unwrap();
    }
    assert!(
        j.wait_jit(Duration::from_secs(60)),
        "JIT not swapped in after the edit"
    );
    assert!(j.read("PROBE").is_some());
    step(&mut j, &mut r, 90, 160);
}

/// Random gate-function edits on random designs: after every edit the JIT engine (compiled
/// synchronously, with the function cache shared across edits so unchanged partitions are
/// reused) matches the reference engine on every net, phase by phase.
#[test]
fn random_edits_recompile_exactly() {
    use panda_difftest::{GenConfig, Rng, generate};
    use panda_kernel::EditOp;
    use panda_netlist::{Atom, GateOp};
    let ops = [
        GateOp::And,
        GateOp::Or,
        GateOp::Nand,
        GateOp::Nor,
        GateOp::Xor,
        GateOp::Xnor,
    ];
    let mut active = 0;
    for seed in 0..16u64 {
        let g = generate(seed, &GenConfig::default(), 4);
        let mut design = g.design.clone();
        let mut rng = Rng::new(seed ^ 0xED17);
        let mut j = kernel(FastBackend::Jit, false);
        let mut r = kernel(FastBackend::Reference, false);
        for k in [&mut j, &mut r] {
            k.set_mode(Mode::Fast);
            k.load(design.clone()).unwrap();
            k.wait_ready().unwrap();
        }
        for edit in 0..5 {
            active += usize::from(j.jit_active());
            for ph in 0..24 {
                if rng.chance(30) {
                    let flat = j.netlist().unwrap().clone();
                    if !flat.top_inputs.is_empty() {
                        let p = &flat.top_inputs[rng.below(flat.top_inputs.len())];
                        let w = p.nets.len() as u32;
                        let v = Value::from_u64(w, rng.next_u64() & ((1u64 << w.min(63)) - 1));
                        j.set_input(&p.name, &v).unwrap();
                        r.set_input(&p.name, &v).unwrap();
                    }
                }
                j.step(1, StepUnit::Phase);
                r.step(1, StepUnit::Phase);
                assert_eq!(j.tick(), r.tick());
                assert_eq!(
                    all_values(&j),
                    all_values(&r),
                    "seed {seed} edit {edit} phase {ph}"
                );
            }
            // Change the function of a random gate.
            let gates: Vec<(usize, usize)> = design
                .defs
                .iter()
                .enumerate()
                .flat_map(|(d, def)| {
                    def.instances
                        .iter()
                        .enumerate()
                        .filter_map(move |(i, inst)| {
                            matches!(inst.kind, InstKind::Atom(Atom::Gate { .. })).then_some((d, i))
                        })
                })
                .collect();
            if gates.is_empty() {
                break;
            }
            let (d, i) = gates[rng.below(gates.len())];
            if let InstKind::Atom(Atom::Gate { op, .. }) = &mut design.defs[d].instances[i].kind {
                let mut n = *rng.pick(&ops);
                while n == *op {
                    n = *rng.pick(&ops);
                }
                *op = n;
            }
            let replace = [EditOp::ReplaceDef(design.defs[d].clone())];
            for k in [&mut j, &mut r] {
                k.edit(&replace).unwrap();
                k.wait_ready().unwrap();
            }
            assert_eq!(
                all_values(&j),
                all_values(&r),
                "seed {seed} right after edit {edit}"
            );
        }
    }
    assert!(
        active > 20,
        "the JIT was active in only {active} configurations"
    );
}
