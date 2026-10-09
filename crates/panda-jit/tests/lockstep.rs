//! JIT == interpreter, bit for bit: two `FastSim`s over the same program, one interpreting and
//! one running native code, driven identically and compared after every event — all net
//! values, tick, schedule, diagnostics and *every* statistic (partitions evaluated, ops, the
//! fast-path counters), plus the engines' internal invariants. Also covers mixed tiers, the
//! background hot swap, `load_state` into a native engine, and incremental recompilation
//! through the function cache.

use std::sync::Arc;
use std::time::{Duration, Instant};

use panda_compile::{FastOptions, KernelOptions, compile_kernels};
use panda_cpu8::programs;
use panda_cpu8::{Options, RomStyle, Variant, panda8_with};
use panda_difftest::{GenConfig, Rng, generate};
use panda_fast::{CompileOptions, FastSim, FastSimOptions, FastSimStats};
use panda_jit::{Jit, JitOptions};
use panda_netlist::flat::FlatNetlist;
use panda_netlist::{Diagnostic, Logic, Value, flatten};

/// Two engines over the same kernels and options.
fn twins(flat: &Arc<FlatNetlist>, opts: &FastSimOptions) -> (FastSim, FastSim) {
    let kopts = KernelOptions {
        extract: opts.fast.extract.clone(),
        charz: opts.fast.charz.clone(),
    };
    let (ks, tables) = compile_kernels(flat, &kopts);
    let a = FastSim::from_parts(flat.clone(), opts.clone(), ks.clone(), tables.clone()).unwrap();
    let b = FastSim::from_parts(flat.clone(), opts.clone(), ks, tables).unwrap();
    (a, b)
}

/// Statistics without the native-only counters.
fn core_stats(s: FastSimStats) -> FastSimStats {
    FastSimStats {
        native_rounds: 0,
        native_bails: 0,
        ..s
    }
}

fn kinds(d: &[Diagnostic]) -> Vec<String> {
    d.iter()
        .map(|d| format!("{:?} {:?}", d.kind, d.location))
        .collect()
}

/// Compare everything observable (and the statistics); drains diagnostics.
fn same(a: &mut FastSim, b: &mut FastSim, ctx: &str) {
    if let Err(m) = a.check_invariants() {
        panic!("{ctx}: interpreter invariant: {m}");
    }
    if let Err(m) = b.check_invariants() {
        panic!("{ctx}: JIT invariant: {m}");
    }
    // Internal runtime state first (raw, before materialization): divergences show here
    // before they become observable.
    for ((name, x), (_, y)) in a.debug_runtime_state().iter().zip(b.debug_runtime_state()) {
        if *x != y {
            let at: Vec<String> = x
                .iter()
                .zip(&y)
                .enumerate()
                .filter(|(_, (p, q))| p != q)
                .take(8)
                .map(|(i, (p, q))| format!("[{i}] interp {p:#x} jit {q:#x}"))
                .collect();
            panic!(
                "{ctx}: runtime state `{name}` differs at tick {}: {at:?}",
                a.tick()
            );
        }
    }
    if a.values() != b.values() {
        let flat = a.netlist().clone();
        let diffs: Vec<String> = (0..flat.num_nets as usize)
            .filter(|&n| a.values()[n] != b.values()[n])
            .take(10)
            .map(|n| {
                format!(
                    "{}: interp {} jit {}",
                    flat.net_names[n],
                    a.values()[n],
                    b.values()[n]
                )
            })
            .collect();
        panic!("{ctx}: values differ at tick {}: {diffs:?}", a.tick());
    }
    assert_eq!(a.tick(), b.tick(), "{ctx}: tick");
    assert_eq!(
        a.next_event_tick(),
        b.next_event_tick(),
        "{ctx}: next event"
    );
    assert_eq!(
        core_stats(a.stats()),
        core_stats(b.stats()),
        "{ctx}: statistics"
    );
    let (da, db) = (a.take_diagnostics(), b.take_diagnostics());
    assert_eq!(kinds(&da), kinds(&db), "{ctx}: diagnostics");
}

fn random_value(rng: &mut Rng, width: u32, xz_pct: u32) -> Value {
    if rng.chance(xz_pct) {
        let bits: Vec<Logic> = (0..width)
            .map(|_| *rng.pick(&[Logic::Zero, Logic::One, Logic::X, Logic::Z]))
            .collect();
        Value::from_bits(&bits)
    } else {
        let v = rng.next_u64();
        Value::from_u64(
            width,
            if width >= 64 {
                v
            } else {
                v & ((1 << width) - 1)
            },
        )
    }
}

/// Drive both engines with the same random actions, comparing after each.
fn drive(a: &mut FastSim, b: &mut FastSim, rng: &mut Rng, actions: usize, ctx: &str) {
    let pins: Vec<(String, u32)> = a
        .netlist()
        .top_inputs
        .iter()
        .map(|p| (p.name.clone(), p.nets.len() as u32))
        .collect();
    same(a, b, &format!("{ctx} start"));
    for i in 0..actions {
        let roll = rng.below(100);
        if roll < 30 && !pins.is_empty() {
            for _ in 0..1 + rng.below(2) {
                let (name, w) = rng.pick(&pins).clone();
                let v = random_value(rng, w, 15);
                a.set_input(&name, &v).unwrap();
                b.set_input(&name, &v).unwrap();
            }
            assert_eq!(a.step_event(), b.step_event(), "{ctx} action {i}");
        } else if roll < 31 {
            a.reset();
            b.reset();
        } else {
            assert_eq!(a.step_event(), b.step_event(), "{ctx} action {i}");
        }
        same(a, b, &format!("{ctx} action {i}"));
    }
}

/// Random compile options (partition sizes, LUTs, specialization, fusion).
fn random_compile(rng: &mut Rng) -> CompileOptions {
    let mut c = CompileOptions {
        partition_ops: *rng.pick(&[1, 3, 8, 32, 1 << 20]),
        ..CompileOptions::default()
    };
    match rng.below(4) {
        0 => {}
        1 => c.lut_max_bits = 0,
        _ => {
            c.lut_max_bits = *rng.pick(&[1, 2, 3, 4, 6, 8]);
            c.lut_min_gates = *rng.pick(&[1, 2, 3, 5]);
        }
    }
    c.specialize = !rng.chance(15);
    c.fuse_inputs = *rng.pick(&[0, 2, 4, 6]);
    c.data_order_lemma = !rng.chance(20);
    c
}

#[test]
fn random_designs_jit_equals_interpreter() {
    let jit = Jit::new(JitOptions::default());
    let (mut rounds, mut bails, mut events) = (0u64, 0u64, 0usize);
    // `PANDA_JIT_SEEDS` for soaks.
    let seeds: u64 = std::env::var("PANDA_JIT_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
    for seed in 0..seeds {
        let mut rng = Rng::new(seed ^ 0x5EED_0717);
        let g = generate(seed, &GenConfig::default(), 2 * (1 + rng.below(3) as u32));
        let flat = Arc::new(flatten(&g.design).expect("flatten"));
        let mut fast = FastOptions::default();
        if rng.chance(15) {
            fast.delta_limit = *rng.pick(&[1, 2, 3, 8]);
        }
        let opts = FastSimOptions {
            fast,
            compile: random_compile(&mut rng),
        };
        let (mut a, mut b) = twins(&flat, &opts);
        let (code, stats) = jit.compile(b.program_arc()).expect("jit");
        events += stats.events;
        // Mixed tiers for some seeds.
        let code = match seed % 5 {
            1 => Arc::new(code.subset(|p| p % 2 == 1, |_| true)),
            3 => Arc::new(code.subset(|_| true, |t| t % 2 == 0)),
            _ => code,
        };
        b.install_native(code).unwrap();
        drive(&mut a, &mut b, &mut rng, 200, &format!("seed {seed}"));
        rounds += b.stats().native_rounds;
        bails += b.stats().native_bails;
    }
    // Event functions must run, complete rounds, and bail out to the interpreter.
    assert!(
        events > 0 && rounds > 0 && bails > 0,
        "{events} {rounds} {bails}"
    );
}

fn panda8(rom: RomStyle, v: Variant, prog: &programs::Program) -> Arc<FlatNetlist> {
    Arc::new(flatten(&panda8_with(&prog.image(), &Options::new(v).with_rom(rom))).unwrap())
}

#[test]
fn panda8_jit_equals_interpreter() {
    std::thread::scope(|s| {
        for rom in [
            RomStyle::SumOfProducts,
            RomStyle::MuxTree,
            RomStyle::Decoder,
        ] {
            for v in Variant::ALL {
                s.spawn(move || {
                    let flat = panda8(rom, v, &programs::BENCH_LOOP);
                    let (mut a, mut b) = twins(&flat, &FastSimOptions::default());
                    let stats = Jit::default().install(&mut b).unwrap();
                    assert_eq!(stats.compiled, stats.partitions);
                    let ctx = format!("{rom:?} {v:?}");
                    for e in 0..3000 {
                        assert_eq!(a.step_event(), b.step_event());
                        same(&mut a, &mut b, &format!("{ctx} event {e}"));
                    }
                    let st = b.stats();
                    assert!(st.native_rounds > 0, "{ctx}: {st:?}");
                });
            }
        }
    });
}

#[test]
fn hot_swap_in_background_mid_run() {
    let flat = panda8(
        RomStyle::SumOfProducts,
        Variant::Hierarchical,
        &programs::BENCH_LOOP,
    );
    let (mut a, mut b) = twins(&flat, &FastSimOptions::default());
    let jit = Jit::default();
    for e in 0..50 {
        a.step_event();
        b.step_event();
        same(&mut a, &mut b, &format!("before attach, event {e}"));
    }
    let handle = jit.attach(&mut b);
    let t0 = Instant::now();
    let mut e = 0;
    // Keep stepping (interpreting) until the code is installed at an event boundary.
    while b.native().is_none() {
        assert!(
            t0.elapsed() < Duration::from_secs(60),
            "JIT never installed"
        );
        a.step_event();
        b.step_event();
        same(&mut a, &mut b, &format!("while compiling, event {e}"));
        e += 1;
    }
    let stats = handle.join().unwrap().unwrap();
    assert_eq!(stats.compiled, stats.partitions);
    for i in 0..2000 {
        a.step_event();
        b.step_event();
        same(&mut a, &mut b, &format!("after swap, event {i}"));
    }
    assert!(b.stats().native_rounds > 0);
    // Code for another program is refused.
    let other = panda8(RomStyle::MuxTree, Variant::Flat, &programs::BENCH_LOOP);
    let c = FastSim::new(other, FastSimOptions::default()).unwrap();
    let (code, _) = jit.compile(c.program_arc()).unwrap();
    assert!(b.install_native(code).is_err());
    // Back to the interpreter at any time.
    b.remove_native();
    for i in 0..200 {
        a.step_event();
        b.step_event();
        same(&mut a, &mut b, &format!("after removal, event {i}"));
    }
}

#[test]
fn load_state_into_native_engine() {
    let flat = panda8(
        RomStyle::SumOfProducts,
        Variant::GatedClockRegfile,
        &programs::FIBONACCI,
    );
    let (mut a, mut b) = twins(&flat, &FastSimOptions::default());
    Jit::default().install(&mut b).unwrap();
    for e in 0..300 {
        a.step_event();
        b.step_event();
        same(&mut a, &mut b, &format!("event {e}"));
    }
    // A third engine's state loaded into both.
    let mut c = FastSim::new(flat.clone(), FastSimOptions::default()).unwrap();
    c.run_events(777);
    let vals = c.values().to_vec();
    let inputs: Vec<Value> = flat
        .top_inputs
        .iter()
        .map(|p| c.input_value(&p.name).unwrap().clone())
        .collect();
    assert_eq!(
        a.load_state(c.tick(), &vals, &inputs),
        b.load_state(c.tick(), &vals, &inputs)
    );
    same(&mut a, &mut b, "after load");
    for e in 0..1000 {
        a.step_event();
        b.step_event();
        same(&mut a, &mut b, &format!("after load, event {e}"));
    }
}

#[test]
fn incremental_recompile_reuses_unchanged_partitions() {
    let jit = Jit::default();
    let flat = panda8(
        RomStyle::SumOfProducts,
        Variant::Hierarchical,
        &programs::BENCH_LOOP,
    );
    let a = FastSim::new(flat.clone(), FastSimOptions::default()).unwrap();
    let s1 = jit.compile(a.program_arc()).unwrap().1;
    assert_eq!((s1.compiled, s1.reused), (s1.partitions, 0));
    // The same program again: everything from the cache.
    let s2 = jit.compile(a.program_arc()).unwrap().1;
    assert_eq!(
        (s2.compiled, s2.reused, s2.events_compiled),
        (0, s2.partitions, 0)
    );
    // An edit that keeps the net numbering: one combinational AND becomes an OR. Only the
    // partition holding it is recompiled.
    let prog = a.program();
    let target = prog
        .ops
        .iter()
        .find_map(|op| match *op {
            panda_fast::Op::Gate2 { t, out, a, b, .. }
                if t == panda_fast::ir::table::AND && a != b =>
            {
                Some(out)
            }
            _ => None,
        })
        .expect("an AND gate op");
    let mut edited = (*flat).clone();
    let g = edited
        .gates
        .iter_mut()
        .find(|g| g.output == target)
        .expect("its gate");
    assert_eq!(g.op, panda_netlist::flat::FlatOp::And);
    g.op = panda_netlist::flat::FlatOp::Or;
    let edited = Arc::new(edited);
    let (mut x, mut y) = twins(&edited, &FastSimOptions::default());
    let (code, s3) = jit.compile(y.program_arc()).unwrap();
    assert_eq!(s3.compiled, 1, "{s3:?}");
    assert_eq!(s3.reused, s3.partitions - 1, "{s3:?}");
    y.install_native(code).unwrap();
    for e in 0..2000 {
        x.step_event();
        y.step_event();
        same(&mut x, &mut y, &format!("recompiled, event {e}"));
    }
    assert!(!jit.cache().is_empty());
}

#[test]
fn delta_limit_and_oscillators_bail_correctly() {
    // Random designs with tiny delta limits: event functions must stop at the limit.
    let jit = Jit::default();
    for seed in 1000..1100u64 {
        let mut rng = Rng::new(seed);
        let g = generate(seed, &GenConfig::default(), 4);
        let flat = Arc::new(flatten(&g.design).expect("flatten"));
        let opts = FastSimOptions {
            fast: FastOptions {
                delta_limit: 1 + rng.below(2) as u32,
                ..FastOptions::default()
            },
            compile: CompileOptions::default(),
        };
        let (mut a, mut b) = twins(&flat, &opts);
        jit.install(&mut b).unwrap();
        drive(&mut a, &mut b, &mut rng, 150, &format!("seed {seed}"));
    }
}

#[test]
fn boundary_log_epoch_wraps_around_identically() {
    // The interval counter wraps at `u32::MAX` (the native event functions call back into the
    // host, partitions log with the context's copy of the epoch): jump both engines close to
    // the wrap repeatedly and compare.
    let jit = Jit::default();
    let mut native = 0u64;
    let mut check = |flat: &Arc<FlatNetlist>, opts: &FastSimOptions, seed: u64, ctx: &str| {
        let (mut a, mut b) = twins(flat, opts);
        jit.install(&mut b).unwrap();
        let mut rng = Rng::new(seed);
        for round in 0..12u32 {
            let gap = 1 + rng.below(4) as u32;
            a.set_epoch_for_test(u32::MAX - gap);
            b.set_epoch_for_test(u32::MAX - gap);
            drive(&mut a, &mut b, &mut rng, 40, &format!("{ctx} wrap {round}"));
            native += b.stats().native_rounds;
        }
    };
    for (i, v) in [Variant::Hierarchical, Variant::GatedClockRegfile]
        .into_iter()
        .enumerate()
    {
        let flat = panda8(RomStyle::SumOfProducts, v, &programs::FIBONACCI);
        check(
            &flat,
            &FastSimOptions::default(),
            i as u64,
            &format!("{v:?}"),
        );
    }
    for seed in 0..40u64 {
        let g = generate(seed, &GenConfig::default(), 4);
        let flat = Arc::new(flatten(&g.design).expect("flatten"));
        let mut rng = Rng::new(seed ^ 0xE90C);
        let opts = FastSimOptions {
            fast: FastOptions::default(),
            compile: random_compile(&mut rng),
        };
        check(&flat, &opts, seed, &format!("seed {seed}"));
    }
    assert!(native > 0, "native event functions never ran");
}

#[test]
fn wide_rom_luts_with_all_ones_entries() {
    // ROM LUTs up to the 32-output limit (an all-ones entry of a 32-output LUT equals the
    // "no entry held" marker), random contents, X/Z addresses (the cold path).
    let jit = Jit::default();
    let mut widest = 0;
    for (i, (addr_bits, data_bits)) in [(3, 31), (3, 32), (5, 32), (4, 33), (2, 64), (6, 8)]
        .into_iter()
        .enumerate()
    {
        let mut rng = Rng::new(0x0B0E + i as u64);
        let mask = if data_bits == 64 {
            u64::MAX
        } else {
            (1u64 << data_bits) - 1
        };
        let contents: Vec<u64> = (0..1u64 << addr_bits)
            .map(|w| match w % 4 {
                0 => mask,
                1 => 0,
                _ => rng.next_u64() & mask,
            })
            .collect();
        let mut db = panda_netlist::DesignBuilder::new();
        let r = panda_circuits::rom(&mut db, "r", addr_bits, data_bits, &contents);
        let flat = Arc::new(flatten(&db.build(&r)).expect("flatten"));
        for lut_bits in [0, 4, 8] {
            let opts = FastSimOptions {
                fast: FastOptions::default(),
                compile: CompileOptions {
                    lut_max_bits: lut_bits,
                    lut_min_gates: 1,
                    ..CompileOptions::default()
                },
            };
            let (mut a, mut b) = twins(&flat, &opts);
            jit.install(&mut b).unwrap();
            widest = b
                .program()
                .luts
                .iter()
                .map(|l| l.num_outputs)
                .fold(widest, u32::max);
            let ctx = format!("rom {addr_bits}x{data_bits} lut_max_bits {lut_bits}");
            drive(&mut a, &mut b, &mut rng, 300, &ctx);
        }
    }
    assert_eq!(widest, 32, "no 32-output LUT was formed");
}

#[test]
fn cranelift_failures_fall_back_to_the_interpreter() {
    use panda_jit::{Func, Inject};
    fn some_fail(f: Func) -> Inject {
        match f {
            Func::Partition(p) if p % 3 == 1 => Inject::Error,
            Func::Partition(p) if p % 5 == 2 => Inject::Panic,
            Func::Event(t) if t % 2 == 1 => Inject::Panic,
            Func::Event(2) => Inject::Error,
            _ => Inject::None,
        }
    }
    let flat = panda8(
        RomStyle::MuxTree,
        Variant::GatedClockRegfile,
        &programs::BENCH_LOOP,
    );
    let jit = Jit::new(JitOptions {
        inject: Some(some_fail),
        ..JitOptions::default()
    });
    let (mut a, mut b) = twins(&flat, &FastSimOptions::default());
    let np = b.program().partitions.len();
    let nt = b.program().num_tuples as usize;
    let (code, stats) = jit.compile(b.program_arc()).unwrap();
    let bad_parts = (0..np).filter(|p| p % 3 == 1 || p % 5 == 2).count();
    let bad_events = (0..nt).filter(|t| t % 2 == 1 || *t == 2).count();
    assert_eq!(stats.failed, bad_parts + bad_events, "{stats:?}");
    assert_eq!(code.native_partitions(), np - bad_parts, "{stats:?}");
    assert_eq!(code.native_events(), nt - bad_events, "{stats:?}");
    assert_eq!(stats.compiled, np - bad_parts);
    assert!(stats.errors.iter().any(|e| e.contains("injected panic")));
    b.install_native(code).unwrap();
    for e in 0..1500 {
        assert_eq!(a.step_event(), b.step_event());
        same(&mut a, &mut b, &format!("partly compiled, event {e}"));
    }
    assert!(b.stats().native_rounds > 0);

    // Only the successful functions were cached: compiling again without failures compiles
    // exactly the failed ones.
    let retry = jit.with_options(JitOptions::default());
    let s1 = retry.compile(b.program_arc()).unwrap().1;
    assert_eq!(
        (s1.failed, s1.compiled, s1.reused),
        (0, bad_parts, np - bad_parts),
        "{s1:?}"
    );
    assert_eq!(s1.events_compiled, bad_events, "{s1:?}");

    // Nothing compiles: an error, and a background attach leaves the engine interpreting.
    let none = Jit::new(JitOptions {
        inject: Some(|_| Inject::Error),
        ..JitOptions::default()
    });
    assert!(none.compile(b.program_arc()).is_err());
    let (mut x, mut y) = twins(&flat, &FastSimOptions::default());
    let h = none.attach(&mut y);
    assert!(h.join().unwrap().is_err());
    for e in 0..300 {
        assert_eq!(x.step_event(), y.step_event());
        same(&mut x, &mut y, &format!("failed JIT, event {e}"));
    }
    assert!(y.native().is_none());
}

#[test]
fn background_compile_outliving_its_engine() {
    // Engines dropped (or given other code) while their background compilation runs: the code
    // is published into an orphaned slot or not compiled at all; nothing dangles.
    let flat = panda8(
        RomStyle::SumOfProducts,
        Variant::Hierarchical,
        &programs::BENCH_LOOP,
    );
    let jit = Jit::default();
    let mut handles = Vec::new();
    for i in 0..6 {
        let mut s = FastSim::new(flat.clone(), FastSimOptions::default()).unwrap();
        handles.push(jit.attach(&mut s));
        if i % 2 == 0 {
            s.run_events(5);
        }
        drop(s);
    }
    for h in handles {
        match h.join().unwrap() {
            Ok(_) => {}
            Err(e) => assert!(e.contains("abandoned"), "{e}"),
        }
    }
    // Hot swap racing with synchronous installs and removal on a live engine.
    let (mut a, mut b) = twins(&flat, &FastSimOptions::default());
    let h = jit.attach(&mut b);
    for e in 0..400 {
        if e == 100 {
            jit.install(&mut b).unwrap();
        }
        if e == 200 {
            b.remove_native();
        }
        assert_eq!(a.step_event(), b.step_event());
        same(&mut a, &mut b, &format!("racing installs, event {e}"));
    }
    h.join().unwrap().unwrap();
}

#[test]
fn large_random_designs_multiword_bitsets() {
    // More than 64 kernels and partitions: multi-word kernel / partition bitsets, jump-table
    // dispatch over many kernels per word, long sweeps.
    let cfg = GenConfig {
        regs: (60, 160),
        comb: (150, 500),
        inputs: (2, 6),
        ..GenConfig::default()
    };
    let jit = Jit::default();
    let (mut multi_k, mut multi_p, mut rounds) = (0, 0, 0u64);
    let seeds: u64 = std::env::var("PANDA_JIT_LARGE_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    for seed in 0..seeds {
        let mut rng = Rng::new(seed ^ 0x1A46E);
        let g = generate(seed, &cfg, 2 * (1 + rng.below(3) as u32));
        let flat = Arc::new(flatten(&g.design).expect("flatten"));
        let mut fast = FastOptions::default();
        if rng.chance(10) {
            fast.delta_limit = *rng.pick(&[2, 3, 8]);
        }
        let opts = FastSimOptions {
            fast,
            compile: random_compile(&mut rng),
        };
        let (mut a, mut b) = twins(&flat, &opts);
        let (code, _) = jit.compile(b.program_arc()).expect("jit");
        let code = match seed % 4 {
            1 => Arc::new(code.subset(|p| p % 3 != 0, |_| true)),
            _ => code,
        };
        multi_k += usize::from(b.program().kernel_words > 1);
        multi_p += usize::from(b.program().partitions.len() > 64);
        b.install_native(code).unwrap();
        drive(&mut a, &mut b, &mut rng, 150, &format!("large seed {seed}"));
        rounds += b.stats().native_rounds;
    }
    assert!(
        multi_k > seeds as usize / 2,
        "{multi_k} designs with >64 kernels"
    );
    assert!(multi_p > 0 && rounds > 0, "{multi_p} {rounds}");
}

#[test]
fn compilation_is_deterministic() {
    // Fresh JITs (empty caches) emit the same code for the same program, and engines running
    // it evolve identically.
    let flat = panda8(RomStyle::Decoder, Variant::Flat, &programs::FIBONACCI);
    let compile = || {
        let mut s = FastSim::new(flat.clone(), FastSimOptions::default()).unwrap();
        let stats = Jit::default().install(&mut s).unwrap();
        (s, stats)
    };
    let (mut a, sa) = compile();
    let (mut b, sb) = compile();
    let strip = |s: panda_jit::JitStats| panda_jit::JitStats {
        compile_time: Duration::ZERO,
        ..s
    };
    assert_eq!(strip(sa.clone()), strip(sb));
    assert!(sa.code_bytes > 0 && sa.failed == 0, "{sa:?}");
    for e in 0..500 {
        assert_eq!(a.step_event(), b.step_event());
        assert_eq!(a.values(), b.values(), "event {e}");
        assert_eq!(a.stats(), b.stats(), "event {e}");
    }
}
