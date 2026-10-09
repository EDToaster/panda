//! Op semantics (exhaustive against `eval_gate` / `resolve` over all 4-state inputs), program
//! structure invariants, and the convenience API (`run_cycles`, `read`, `set_input`).

mod common;

/// Native-code hook of `common::pair` (none here: the interpreter; the panda-jit crate runs
/// this suite with the JIT installed through it).
fn native_hook(_: &mut panda_fast::FastSim) {}

use std::sync::Arc;

use panda_compile::{FastOptions, FastRef};
use panda_cpu8::programs;
use panda_cpu8::{Options, RomStyle, Variant, panda8_with};
use panda_fast::ir::{NONE, Op};
use panda_fast::{CompileOptions, FastSim, FastSimOptions, OrderStrategy};
use panda_netlist::eval::{eval_gate, resolve};
use panda_netlist::flat::{
    FlatGate, FlatInstance, FlatNetlist, FlatOp, FlatPin, GateGroup, NetIdx, Resolver, Source,
    SourceKind,
};
use panda_netlist::{Dir, Logic, Termination, Value, flatten};

const ALL: [Logic; 4] = [Logic::Zero, Logic::One, Logic::X, Logic::Z];

/// Inputs `i0..i3`; one gate per (op, arity 1..=4) reading `i0..`, plus Tri and one resolver per
/// termination over the 4 inputs.
fn gate_zoo() -> (Arc<FlatNetlist>, Vec<(String, FlatOp, usize)>) {
    let mut f = FlatNetlist::default();
    f.instances.push(FlatInstance {
        path: String::new(),
        def: "top".into(),
        parent: None,
        children: Vec::new(),
        pins: Vec::new(),
    });
    let net = |f: &mut FlatNetlist, name: &str| -> NetIdx {
        let n = f.num_nets;
        f.num_nets += 1;
        f.net_names.push(name.into());
        f.name_index.insert(name.into(), n);
        n
    };
    let ins: Vec<NetIdx> = (0..4)
        .map(|i| {
            let n = net(&mut f, &format!("i{i}"));
            f.sources.push(Source {
                kind: SourceKind::TopInput { pin: i, bit: 0 },
                output: n,
            });
            f.top_inputs.push(FlatPin {
                name: format!("i{i}"),
                dir: Dir::In,
                nets: vec![n],
            });
            n
        })
        .collect();
    let mut gates = Vec::new();
    let ops = [
        FlatOp::And,
        FlatOp::Or,
        FlatOp::Nand,
        FlatOp::Nor,
        FlatOp::Xor,
        FlatOp::Xnor,
        FlatOp::Not,
        FlatOp::Buf,
        FlatOp::Tri,
    ];
    for op in ops {
        let arities: &[usize] = match op {
            FlatOp::Not | FlatOp::Buf => &[1],
            FlatOp::Tri => &[2],
            _ => &[1, 2, 3, 4],
        };
        for &a in arities {
            let name = format!("{op:?}{a}");
            let out = net(&mut f, &name);
            let gi = f.gates.len() as u32;
            f.gates.push(FlatGate {
                op,
                inputs: ins[..a].to_vec(),
                output: out,
                owner: 0,
                group: gi,
                bit: 0,
            });
            f.groups.push(GateGroup {
                owner: 0,
                path: name.clone(),
                gates: vec![gi],
            });
            gates.push((name, op, a));
        }
    }
    for (ti, term) in [
        Termination::None,
        Termination::PullUp,
        Termination::PullDown,
    ]
    .into_iter()
    .enumerate()
    {
        let out = net(&mut f, &format!("res{ti}"));
        f.resolvers.push(Resolver {
            drivers: ins.clone(),
            termination: term,
            output: out,
        });
    }
    (Arc::new(f), gates)
}

#[test]
fn ops_match_reference_semantics_exhaustively() {
    let (flat, gates) = gate_zoo();
    // Plain ops, and every 2-state gate absorbed into LUTs (table path for binary inputs, cold
    // gate code for X/Z).
    for (partition_ops, lut_min_gates) in [(1, 24), (4, 24), (64, 24), (1, 1), (64, 1)] {
        let opts = FastSimOptions {
            compile: CompileOptions {
                partition_ops,
                lut_min_gates,
                lut_max_bits: 4,
                ..CompileOptions::default()
            },
            ..FastSimOptions::default()
        };
        let mut fs = FastSim::new(flat.clone(), opts).unwrap();
        native_hook(&mut fs);
        if lut_min_gates == 1 {
            let st = fs.program().stats();
            assert!(st.luts >= 1 && st.lut_gates == 26, "{st:?}");
        }
        let mut fr = FastRef::new(flat.clone(), FastOptions::default()).unwrap();
        let mut cov = common::Coverage::default();
        for code in 0..256u32 {
            let vals: Vec<Logic> = (0..4)
                .map(|i| ALL[(code >> (2 * i) & 3) as usize])
                .collect();
            for (i, &v) in vals.iter().enumerate() {
                fs.set_input(&format!("i{i}"), &Value::from_logic(v))
                    .unwrap();
                fr.set_input(&format!("i{i}"), &Value::from_logic(v))
                    .unwrap();
            }
            fs.step_event();
            fr.step_event();
            for (name, op, a) in &gates {
                let want = eval_gate(*op, &vals[..*a]);
                let got = fs.read(name).unwrap().get(0);
                assert_eq!(got, want, "{name} on {vals:?}");
            }
            for (ti, term) in [
                Termination::None,
                Termination::PullUp,
                Termination::PullDown,
            ]
            .into_iter()
            .enumerate()
            {
                let want = resolve(&vals, term, Logic::X).value;
                assert_eq!(fs.read(&format!("res{ti}")).unwrap().get(0), want);
            }
            common::check(&mut fr, &mut fs, &flat, &format!("code {code}"), &mut cov);
        }
        assert!(cov.diag_kinds.contains_key("Contention"));
    }
}

fn panda8(rom: RomStyle, variant: Variant) -> Arc<FlatNetlist> {
    let opts = Options::new(variant).with_rom(rom);
    Arc::new(flatten(&panda8_with(&programs::BENCH_LOOP.image(), &opts)).unwrap())
}

#[test]
fn program_structure_invariants() {
    let flat = panda8(RomStyle::SumOfProducts, Variant::Hierarchical);
    for order in [OrderStrategy::ConeDfs, OrderStrategy::KernelSet] {
        for partition_ops in [1, 50, 1000] {
            let opts = FastSimOptions {
                compile: CompileOptions {
                    partition_ops,
                    order,
                    ..CompileOptions::default()
                },
                ..FastSimOptions::default()
            };
            let fs = FastSim::new(flat.clone(), opts).unwrap();
            let p = fs.program();
            let st = p.stats();
            assert_eq!(
                st.ops - st.luts + st.lut_gates,
                fs.kernels().comb_order.len()
            );
            assert_eq!(st.kernels, fs.kernels().kernels.len());
            assert!(st.max_partition <= partition_ops);
            assert_eq!(
                p.passes,
                [
                    "order",
                    "lut",
                    "partition",
                    "levelize",
                    "lower",
                    "notify",
                    "kernels",
                    "kspec",
                    "layout",
                    "verify"
                ]
            );
            // Partitions tile the op array.
            let mut next = 0;
            for part in &p.partitions {
                assert_eq!(part.start, next);
                next = part.end;
            }
            assert_eq!(next as usize, p.ops.len());
            // Driver partition of every op output.
            let mut drv = vec![NONE; p.num_nets as usize];
            for (pi, part) in p.partitions.iter().enumerate() {
                for op in &p.ops[part.start as usize..part.end as usize] {
                    match *op {
                        Op::Lut { lut } => {
                            for &o in p.lut_outputs(&p.luts[lut as usize]) {
                                drv[o as usize] = pi as u32;
                            }
                        }
                        _ => drv[op.output() as usize] = pi as u32,
                    }
                }
            }
            // Every cross-partition read is in the notify list, and notify flags are exact.
            for (pi, part) in p.partitions.iter().enumerate() {
                for op in &p.ops[part.start as usize..part.end as usize] {
                    let reads: Vec<u32> = match *op {
                        Op::Gate2 { a, b, .. } => vec![a, b],
                        Op::GateN { args, n, .. } => {
                            p.args[args as usize..(args + n) as usize].to_vec()
                        }
                        Op::Resolve { res, .. } => {
                            let r = p.resolvers[res as usize];
                            p.args[r.args as usize..(r.args + r.n) as usize].to_vec()
                        }
                        Op::Lut { lut } => p.lut_support(&p.luts[lut as usize]).to_vec(),
                    };
                    for net in reads {
                        if drv[net as usize] != pi as u32 {
                            assert!(p.net_parts.row(net).contains(&(pi as u32)));
                        }
                        if drv[net as usize] != NONE {
                            assert!(drv[net as usize] <= pi as u32, "not topological");
                        }
                    }
                    if let Op::Lut { .. } = op {
                        continue;
                    }
                    let out = op.output();
                    assert_eq!(
                        op.notifies(),
                        !p.net_parts.row(out).is_empty() || !p.net_kernels.row(out).is_empty()
                    );
                }
            }
            for k in &fs.kernels().kernels {
                for &net in &k.inputs {
                    assert!(p.net_kernels.row(net).contains(&k.id));
                }
                // Exactly the state nets with readers are change-propagated.
                let kop = &p.kernel_ops[k.id as usize];
                let nets = p.kernel_state_nets(kop);
                assert_eq!(nets, &k.nets[..]);
                let want: Vec<u32> = (0..nets.len() as u32)
                    .filter(|&j| {
                        let n = nets[j as usize];
                        !p.net_parts.row(n).is_empty() || !p.net_kernels.row(n).is_empty()
                    })
                    .collect();
                assert_eq!(p.kernel_out_indices(kop), &want[..]);
                for &n in &k.outputs {
                    if !p.net_parts.row(n).is_empty() || !p.net_kernels.row(n).is_empty() {
                        let j = k.nets.iter().position(|&x| x == n).unwrap() as u32;
                        assert!(want.contains(&j));
                    }
                }
            }
        }
    }
}

#[test]
fn run_cycles_matches_run_until() {
    let flat = panda8(RomStyle::SumOfProducts, Variant::Flat);
    let mut fs = FastSim::new(flat.clone(), FastSimOptions::default()).unwrap();
    native_hook(&mut fs);
    let mut fr = FastRef::new(flat.clone(), FastOptions::default()).unwrap();
    let (period, phase) = flat
        .sources
        .iter()
        .find_map(|s| match s.kind {
            SourceKind::Clock { period, phase, .. } => Some((u64::from(period), u64::from(phase))),
            _ => None,
        })
        .unwrap();
    let clk = flat
        .sources
        .iter()
        .find(|s| matches!(s.kind, SourceKind::Clock { .. }))
        .map(|s| flat.net_names[s.output as usize].clone())
        .unwrap();
    let mut cov = common::Coverage::default();
    for n in [1u64, 1, 7, 100, 3] {
        fs.run_cycles(&clk, n);
        // Rising edges are at phase + k·period; we must stop one tick before one.
        let next = fs.next_event_tick().unwrap();
        assert_eq!(
            (next + period - phase % period) % period,
            0,
            "stops before a rising edge"
        );
        fr.run_until(next - 1);
        common::check(
            &mut fr,
            &mut fs,
            &flat,
            &format!("run_cycles {n}"),
            &mut cov,
        );
    }
    // 1 cycle = 2 events (rising + falling edge) in steady state.
    let e0 = fs.stats().events;
    fs.run_cycles(&clk, 10);
    assert_eq!(fs.stats().events - e0, 20);
    assert!(fs.try_run_cycles("no_such_net", 1).is_err());
    assert!(fs.try_run_cycles("OUT_STROBE", 1).is_err());
}

/// The facade runs engines on a worker thread: `FastSim` must be `Send`.
#[test]
fn fastsim_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<FastSim>();
}

#[test]
fn api_mirrors_fastref() {
    let flat = panda8(RomStyle::Decoder, Variant::Hierarchical);
    let mut fs = FastSim::new(flat.clone(), FastSimOptions::default()).unwrap();
    native_hook(&mut fs);
    let mut fr = FastRef::new(flat.clone(), FastOptions::default()).unwrap();
    fs.run_events(500);
    fr.run_events(500);
    for name in ["PC", "IR", "R0", "R1", "OUT", "FLAGS", "OUT_STROBE"] {
        assert_eq!(fs.read_pin(name), fr.read_pin(name), "{name}");
        assert_eq!(fs.read(name), fr.read(name), "{name}");
    }
    for n in 0..flat.num_nets {
        assert_eq!(fs.get(n), fr.get(n));
    }
    assert!(fs.set_input("nope", &Value::zeros(1)).is_err());
    assert_eq!(fs.input_value("nope"), None);
    assert!(fs.diagnostics().is_empty());
    fs.reset();
    fr.reset();
    assert_eq!(fs.values(), fr.values());
    assert_eq!(fs.stats().events, 1);
}

/// Every RomStyle × Variant: all ROM gates (net names `rom.*` / `rom/*`) end up in one single LUT
/// (SPEC §7.4), and kernels are specialized (opaque and transparent phases).
#[test]
fn panda8_rom_is_one_lut() {
    for rom in [
        RomStyle::SumOfProducts,
        RomStyle::MuxTree,
        RomStyle::Decoder,
    ] {
        for variant in Variant::ALL {
            let flat = panda8(rom, variant);
            let fs = FastSim::new(flat.clone(), FastSimOptions::default()).unwrap();
            let p = fs.program();
            let mut lut_of = vec![NONE; p.num_nets as usize];
            for (l, d) in p.luts.iter().enumerate() {
                for op in &p.cold_ops[d.cold.start as usize..d.cold.end as usize] {
                    lut_of[op.output() as usize] = l as u32;
                }
            }
            let rom_gates: Vec<u32> = flat
                .gates
                .iter()
                .map(|g| g.output)
                .filter(|&o| {
                    let n = &flat.net_names[o as usize];
                    n.starts_with("rom.") || n.starts_with("rom/")
                })
                .collect();
            assert!(
                !rom_gates.is_empty(),
                "{rom:?} {variant:?}: no ROM gates found"
            );
            let mut ids: Vec<u32> = rom_gates.iter().map(|&o| lut_of[o as usize]).collect();
            ids.sort_unstable();
            ids.dedup();
            assert_eq!(ids.len(), 1, "{rom:?} {variant:?}: ROM spread over {ids:?}");
            assert_ne!(ids[0], NONE, "{rom:?} {variant:?}: ROM gates outside LUTs");
            let st = p.stats();
            assert!(
                st.phase_classes[0] > 0 && st.phase_classes[1] > 0,
                "{rom:?} {variant:?}: {st:?}"
            );
        }
    }
}

/// Lazy nets (LUT-internal nets, non-output nets of lazy kernels) read through `get`/`read`
/// without a prior `values()` call equal FastRef's, after every event.
#[test]
fn lazy_nets_materialize_on_get() {
    for rom in [RomStyle::SumOfProducts, RomStyle::MuxTree] {
        let flat = panda8(rom, Variant::Hierarchical);
        let (mut fr, mut fs) =
            common::pair(&flat, &FastOptions::default(), &CompileOptions::default());
        let p = fs.program();
        let lazy: Vec<u32> = (0..p.num_nets)
            .filter(|&n| p.lazy_owner[n as usize] != NONE)
            .collect();
        let (lut_nets, kernel_nets) = (
            lazy.iter()
                .filter(|&&n| p.lazy_owner[n as usize] % 2 == 0)
                .count(),
            lazy.iter()
                .filter(|&&n| p.lazy_owner[n as usize] % 2 == 1)
                .count(),
        );
        assert!(
            lut_nets > 1000 && kernel_nets > 200,
            "{lut_nets} {kernel_nets}"
        );
        for e in 0..400u32 {
            fr.step_event();
            fs.step_event();
            // A different lazy net first each event (materializes on demand), then all of them.
            let first = lazy[(e as usize * 7919) % lazy.len()];
            assert_eq!(
                fs.get(first),
                fr.get(first),
                "event {e}: net {}",
                flat.net_names[first as usize]
            );
            let name = &flat.net_names[first as usize];
            assert_eq!(fs.read(name), fr.read(name));
            for &n in &lazy {
                assert_eq!(
                    fs.get(n),
                    fr.get(n),
                    "event {e}: net {}",
                    flat.net_names[n as usize]
                );
            }
            assert_eq!(fs.values(), fr.values(), "event {e}");
        }
    }
}
