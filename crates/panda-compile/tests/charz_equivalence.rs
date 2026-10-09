//! The fast characterization path ([`characterize_subnet_with`], any thread count) gives tables
//! bit-identical to the reference path ([`characterize_subnet_reference`]: every transition
//! through `TimingSim`, the implementation used before the fast path existed), and the fast
//! simulator's transitions equal [`KernelSim::transition`] exactly. Also: thread-count
//! independence of [`compile_kernels_with`], concurrent use of a shared [`TableCache`], and the
//! cache snapshot round trip.

use std::collections::HashSet;
use std::sync::Arc;

use panda_compile::charz::{
    CharzOptions, KernelSim, TableCache, characterize_subnet_reference, characterize_subnet_with,
    fast_path_coverage, fast_transition, input_vector, uses_fast_path,
};
use panda_compile::kernels::{ExtractOptions, KernelKind, extract};
use panda_compile::subnet::{INPUT_PIN, SubnetMap, structural_key};
use panda_compile::{KernelOptions, compile_kernels_with, kernel_subnet};
use panda_cpu8::{Options, RomStyle, Variant, panda8_with, programs};
use panda_netlist::flat::{
    FlatGate, FlatInstance, FlatNetlist, FlatOp, FlatPin, GateGroup, NetIdx, Resolver, Source,
    SourceKind,
};
use panda_netlist::{Dir, Logic, Termination, flatten};
use panda_timing::TimingOptions;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn chance(&mut self, pct: usize) -> bool {
        self.below(100) < pct
    }
}

/// Compare fast (1 and 4 threads) with the reference on one sub-netlist. Returns true if the
/// fast path applied and a table was built.
fn compare(sub: &Arc<FlatNetlist>, map: &SubnetMap, opts: &CharzOptions, what: &str) -> bool {
    let reference = characterize_subnet_reference(sub.clone(), map.clone(), opts);
    let one = characterize_subnet_with(sub.clone(), map.clone(), opts, 1);
    assert_eq!(
        one, reference,
        "{what}: fast (1 thread) != reference ({opts:?})"
    );
    let four = characterize_subnet_with(sub.clone(), map.clone(), opts, 4);
    assert_eq!(four, reference, "{what}: fast (4 threads) != reference");
    uses_fast_path(sub, map, opts) && reference.is_some()
}

/// A random sub-netlist in the [`kernel_subnet`] layout: `k` inputs, `m` state nets, each
/// driven by a random gate (feedback likely) or, sometimes, a resolver over tri-state drivers.
fn random_subnet(rng: &mut Rng, max_k: usize, max_m: usize) -> (FlatNetlist, SubnetMap) {
    let k = rng.below(max_k + 1);
    let m = 1 + rng.below(max_m);
    let n = k + m;
    let ops = [
        FlatOp::Nand,
        FlatOp::Nor,
        FlatOp::And,
        FlatOp::Or,
        FlatOp::Xor,
        FlatOp::Xnor,
        FlatOp::Not,
        FlatOp::Buf,
        FlatOp::Tri,
        FlatOp::Nand,
        FlatOp::Nor,
    ];
    let mut gates = Vec::new();
    let mut groups = Vec::new();
    let mut resolvers = Vec::new();
    // Decide the drivers first: resolvers only where there are other nets to drive them.
    let is_res: Vec<bool> = (0..n).map(|j| m >= 3 && j > k && rng.chance(15)).collect();
    for j in k..n {
        if is_res[j] {
            let nd = 1 + rng.below(3);
            let mut drivers = Vec::new();
            for _ in 0..nd {
                // Usually gate outputs (often Tri), occasionally inputs or (rarely) another
                // resolver's output (exercises the reference fallback).
                let d = rng.below(n);
                if d != j && (!is_res[d] || rng.chance(10)) {
                    drivers.push(d as NetIdx);
                }
            }
            if drivers.is_empty() {
                drivers.push(if k > 0 {
                    0
                } else {
                    k as NetIdx + u32::from(j == k)
                });
            }
            let term = [
                Termination::None,
                Termination::PullUp,
                Termination::PullDown,
                Termination::Keeper,
            ][rng.below(4)];
            resolvers.push(Resolver {
                drivers,
                termination: term,
                output: j as NetIdx,
            });
        } else {
            let op = ops[rng.below(ops.len())];
            let arity = match op {
                FlatOp::Not | FlatOp::Buf => 1,
                FlatOp::Tri => 2,
                _ => 1 + rng.below(4),
            };
            let inputs = (0..arity).map(|_| rng.below(n) as NetIdx).collect();
            let gi = gates.len() as u32;
            gates.push(FlatGate {
                op,
                inputs,
                output: j as NetIdx,
                owner: 0,
                group: gi,
                bit: 0,
            });
            groups.push(GateGroup {
                owner: 0,
                path: String::new(),
                gates: vec![gi],
            });
        }
    }
    let sources = (0..k)
        .map(|i| Source {
            kind: SourceKind::TopInput {
                pin: 0,
                bit: i as u32,
            },
            output: i as NetIdx,
        })
        .collect();
    let top_inputs = if k > 0 {
        vec![FlatPin {
            name: INPUT_PIN.into(),
            dir: Dir::In,
            nets: (0..k as NetIdx).collect(),
        }]
    } else {
        Vec::new()
    };
    let sub = FlatNetlist {
        num_nets: n as u32,
        gates,
        sources,
        resolvers,
        groups,
        instances: vec![FlatInstance {
            path: String::new(),
            def: "kernel".into(),
            parent: None,
            children: Vec::new(),
            pins: Vec::new(),
        }],
        net_names: (0..n).map(|i| format!("n{i}")).collect(),
        top_inputs,
        ..Default::default()
    };
    let map = SubnetMap {
        inputs: (0..k as NetIdx).collect(),
        nets: (k as NetIdx..n as NetIdx).collect(),
    };
    (sub, map)
}

/// Random characterization options covering the corner cases of the fast path (collapse off /
/// short windows, tiny settle bounds, budget stops, partial checks, streamed exhaustive orders,
/// size guards).
fn random_opts(rng: &mut Rng) -> CharzOptions {
    CharzOptions {
        timing: TimingOptions::default().with_p2_limit([0, 1, 2, 3, 4, 4, 4][rng.below(7)]),
        settle_bound: if rng.chance(25) {
            Some(rng.below(14) as u64)
        } else {
            None
        },
        exhaustive_order_max: [0, 1, 2, 3, 4, 4, 5, 6, 7][rng.below(9)],
        order_samples: [1, 2, 5, 16, 16][rng.below(5)],
        max_order_work: [0, 10, 300, 5_000, 60_000, 4_000_000][rng.below(6)],
        max_table_entries: if rng.chance(10) { 1 << 9 } else { 1 << 22 },
    }
}

#[test]
fn random_subnets_default_options() {
    let mut rng = Rng(0x00C0_FFEE_1234_5677);
    let opts = CharzOptions::default();
    let mut tables = 0;
    let mut fast = 0;
    for case in 0..1500 {
        let (sub, map) = random_subnet(&mut rng, 6, 8);
        let sub = Arc::new(sub);
        fast += usize::from(uses_fast_path(&sub, &map, &opts));
        tables += usize::from(compare(&sub, &map, &opts, &format!("case {case}")));
    }
    println!("{tables} fast-path tables, {fast} of 1500 on the fast path");
    assert!(tables > 500, "{tables}");
}

#[test]
fn random_subnets_varied_options() {
    let mut rng = Rng(0x5EED_0000_ABCD_0001);
    let mut tables = 0;
    let mut stops = 0;
    let mut partial = 0;
    for case in 0..1500 {
        let opts = random_opts(&mut rng);
        let (sub, map) = random_subnet(&mut rng, 7, 7);
        let sub = Arc::new(sub);
        if compare(&sub, &map, &opts, &format!("case {case}")) {
            tables += 1;
            let t = characterize_subnet_with(sub, map, &opts, 1).expect("table");
            stops += usize::from(t.order_check_stop.is_some());
            partial += usize::from(t.order_check_partial);
        }
    }
    println!("{tables} fast-path tables ({stops} budget stops, {partial} partial checks)");
    assert!(
        tables > 400 && stops > 20 && partial > 20,
        "{tables} {stops} {partial}"
    );
}

/// Larger random kernels (up to 9 inputs) whose transitions go through the bit-parallel
/// engine in several 64-lane batches and whose order passes use worker threads.
#[test]
fn random_larger_subnets() {
    let mut rng = Rng(0x0BAD_CAFE_F00D_0003);
    let mut tables = 0;
    for case in 0..120 {
        let opts = if case % 3 == 0 {
            random_opts(&mut rng)
        } else {
            CharzOptions::default()
        };
        let (sub, map) = random_subnet(&mut rng, 9, 12);
        let sub = Arc::new(sub);
        let reference = characterize_subnet_reference(sub.clone(), map.clone(), &opts);
        for threads in [1, 3, 16] {
            let t = characterize_subnet_with(sub.clone(), map.clone(), &opts, threads);
            assert_eq!(t, reference, "case {case}, {threads} threads");
        }
        tables += usize::from(reference.is_some());
    }
    assert!(tables > 60, "{tables}");
}

/// The fast simulator's `T(state, old, new)` equals `KernelSim::transition` for arbitrary
/// (also non-stable, `X`/`Z`-containing) states.
#[test]
fn fast_transition_matches_kernel_sim() {
    let mut rng = Rng(0x7A11_5EED_9876_5431);
    let vals = [Logic::Zero, Logic::One, Logic::X, Logic::Z];
    let mut checked = 0;
    for case in 0..1200 {
        let opts = if case % 2 == 0 {
            CharzOptions::default()
        } else {
            random_opts(&mut rng)
        };
        let (sub, map) = random_subnet(&mut rng, 5, 9);
        if !uses_fast_path(&sub, &map, &opts) {
            continue;
        }
        let k = map.num_inputs();
        let m = map.num_state();
        let sub = Arc::new(sub);
        let mut ks = KernelSim::from_subnet(sub.clone(), map.clone(), &opts);
        for _ in 0..12 {
            let state: Vec<Logic> = (0..m)
                .map(|_| {
                    if rng.chance(70) {
                        vals[rng.below(2)]
                    } else {
                        vals[rng.below(4)]
                    }
                })
                .collect();
            let old = rng.below(1 << k) as u32;
            let new = if rng.chance(20) {
                old
            } else {
                rng.below(1 << k) as u32
            };
            let (want, settled) =
                ks.transition(&state, &input_vector(old, k), &input_vector(new, k));
            let got = fast_transition(&sub, &map, &opts, &state, old, new).expect("fast path");
            assert_eq!(
                got,
                (want, settled.quiescent, settled.latency),
                "case {case}: state {state:?}, {old:b} -> {new:b}"
            );
            checked += 1;
        }
    }
    assert!(checked > 5000, "{checked}");
}

fn panda8(variant: Variant, rom: RomStyle) -> FlatNetlist {
    flatten(&panda8_with(
        &programs::BENCH_LOOP.image(),
        &Options::new(variant).with_rom(rom),
    ))
    .expect("flatten")
}

/// Every distinct Table kernel of `flat` (under `xopts`): fast == reference.
fn compare_design(
    flat: &FlatNetlist,
    xopts: &ExtractOptions,
    opts: &CharzOptions,
    what: &str,
) -> usize {
    let ks = extract(flat, xopts);
    let mut seen = HashSet::new();
    let mut n = 0;
    for k in &ks.kernels {
        if k.kind != KernelKind::Table {
            continue;
        }
        let (sub, map) = kernel_subnet(flat, k);
        if !seen.insert(structural_key(&sub)) {
            continue;
        }
        compare(
            &Arc::new(sub),
            &map,
            opts,
            &format!("{what}, kernel {}", k.id),
        );
        n += 1;
    }
    n
}

#[test]
fn panda8_all_variants_tables_identical() {
    let tight = CharzOptions {
        max_order_work: 20_000,
        ..CharzOptions::default()
    };
    for rom in [RomStyle::SumOfProducts, RomStyle::MuxTree] {
        for variant in Variant::ALL {
            let flat = panda8(variant, rom);
            let what = format!("panda8 {variant:?} {rom:?}");
            let n = compare_design(
                &flat,
                &ExtractOptions::default(),
                &CharzOptions::default(),
                &what,
            );
            assert!(n >= 10, "{what}: {n} distinct kernels");
            compare_design(&flat, &ExtractOptions::default(), &tight, &what);
        }
    }
}

/// Designs of the differential fuzzer (incl. asynchronous structures, gated clocks, ring
/// oscillators, wide kernels), with default and varied extraction options.
#[test]
fn difftest_designs_tables_identical() {
    use panda_difftest::generator::{GenConfig, generate};
    let cfg = GenConfig::default();
    let mut kernels = 0;
    let seeds: Vec<u64> = (0..120).chain([290]).collect();
    for seed in seeds {
        let g = generate(seed, &cfg, 2);
        let Ok(flat) = flatten(&g.design) else {
            continue;
        };
        let what = format!("difftest seed {seed}");
        kernels += compare_design(
            &flat,
            &ExtractOptions::default(),
            &CharzOptions::default(),
            &what,
        );
        let xopts = ExtractOptions {
            max_loop: [2, 3, 4, 8][seed as usize % 4],
            max_bits: [4, 6, 8, 10][seed as usize / 4 % 4],
            ..ExtractOptions::default()
        };
        kernels += compare_design(&flat, &xopts, &CharzOptions::default(), &what);
    }
    println!("{kernels} distinct kernels compared");
    assert!(kernels > 300, "{kernels}");
}

#[test]
fn compile_kernels_independent_of_thread_count() {
    for (variant, rom) in [
        (Variant::Hierarchical, RomStyle::SumOfProducts),
        (Variant::GatedClockRegfile, RomStyle::MuxTree),
    ] {
        let flat = panda8(variant, rom);
        let opts = KernelOptions::default();
        let c1 = TableCache::new();
        let (ks1, t1) = compile_kernels_with(&flat, &opts, &c1, 1);
        for threads in [2, 5, 16] {
            let cn = TableCache::new();
            let (ksn, tn) = compile_kernels_with(&flat, &opts, &cn, threads);
            assert_eq!(ks1.kernels.len(), ksn.kernels.len());
            assert_eq!(t1, tn, "{variant:?} {rom:?}: {threads} threads");
            assert_eq!((c1.hits(), c1.misses()), (cn.hits(), cn.misses()));
            assert_eq!(c1.len(), cn.len());
        }
        // A second compile hits only.
        let (_, again) = compile_kernels_with(&flat, &opts, &c1, 4);
        assert_eq!(again, t1);
        assert_eq!(c1.misses() as usize, c1.len());
    }
}

#[test]
fn shared_cache_concurrent_compiles() {
    let flat = Arc::new(panda8(Variant::Flat, RomStyle::SumOfProducts));
    let reference = {
        let c = TableCache::new();
        compile_kernels_with(&flat, &KernelOptions::default(), &c, 1).1
    };
    let cache = Arc::new(TableCache::new());
    let results: Vec<_> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..6)
            .map(|i| {
                let (flat, cache) = (flat.clone(), cache.clone());
                s.spawn(move || {
                    compile_kernels_with(&flat, &KernelOptions::default(), &cache, 1 + i % 3).1
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("join")).collect()
    });
    for r in results {
        assert_eq!(r, reference);
    }
    // Every distinct kernel was characterized exactly once.
    assert_eq!(cache.misses() as usize, cache.len());
    let ks = extract(&flat, &ExtractOptions::default());
    assert!(cache.try_tables(&flat, &ks.kernels).is_some());
}

#[test]
fn snapshot_round_trip() {
    let flat = panda8(Variant::GatedClockRegfile, RomStyle::SumOfProducts);
    let opts = KernelOptions::default();
    let cache = TableCache::new();
    let (ks, tables) = compile_kernels_with(&flat, &opts, &cache, 4);
    let json = cache.to_json(&opts.charz);
    let loaded = TableCache::from_json(&json, &opts.charz).expect("load");
    assert_eq!(loaded.len(), cache.len());
    // Everything is available without characterizing.
    assert_eq!(loaded.try_tables(&flat, &ks.kernels), Some(tables.clone()));
    let (_, again) = compile_kernels_with(&flat, &opts, &loaded, 4);
    assert_eq!(again, tables);
    assert_eq!(loaded.misses(), 0);
    // Deterministic serialization.
    assert_eq!(loaded.to_json(&opts.charz), json);
    // File round trip.
    let dir = std::env::temp_dir().join(format!("panda-charz-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let path = dir.join("tables.json");
    cache.save(&path, &opts.charz).expect("save");
    let from_file = TableCache::load(&path, &opts.charz).expect("load file");
    assert_eq!(from_file.try_tables(&flat, &ks.kernels), Some(tables));
    let _ = std::fs::remove_dir_all(&dir);
    // Other options are refused; malformed input is an error, not a panic.
    let other = CharzOptions {
        max_order_work: 7,
        ..CharzOptions::default()
    };
    assert!(TableCache::from_json(&json, &other).is_err());
    assert!(TableCache::from_json("{}", &opts.charz).is_err());
    let mut snap = cache.snapshot(&opts.charz);
    if let Some(t) = snap.entries.iter_mut().find_map(|e| e.table.as_mut()) {
        t.next[0] = 1_000_000;
    }
    assert!(TableCache::new().restore(&snap, &opts.charz).is_err());
}

#[test]
fn non_blocking_lookup() {
    let flat = panda8(Variant::Hierarchical, RomStyle::SumOfProducts);
    let ks = extract(&flat, &ExtractOptions::default());
    let cache = TableCache::new();
    assert!(cache.try_tables(&flat, &ks.kernels).is_none());
    let table_kernel = ks
        .kernels
        .iter()
        .find(|k| k.kind == KernelKind::Table)
        .expect("table kernel");
    assert_eq!(cache.try_get(&flat, table_kernel), None);
    let t = cache.get_or_characterize(&flat, table_kernel, &CharzOptions::default());
    assert!(t.is_some());
    assert_eq!(cache.try_get(&flat, table_kernel), Some(t));
    assert_eq!((cache.hits(), cache.misses()), (0, 1));
}

/// The exact-fallback branches of the fast path are exercised by random kernels (so the
/// equivalence tests above cover them, not only the common paths).
#[test]
fn fallback_paths_are_exercised() {
    let before = fast_path_coverage();
    let mut rng = Rng(0xC0DE_C0DE_0000_0007);
    for case in 0..800 {
        let mut opts = random_opts(&mut rng);
        if case % 4 == 0 {
            // Streamed exhaustive orders: 7 simultaneous changes need 7 inputs.
            opts.exhaustive_order_max = 7;
            opts.max_order_work = 4_000_000;
        }
        let (sub, map) = random_subnet(&mut rng, 7, 8);
        compare(&Arc::new(sub), &map, &opts, &format!("case {case}"));
    }
    let after = fast_path_coverage();
    let names = [
        "collapse",
        "resolver tick",
        "slow entry",
        "sim step",
        "streamed exhaustive",
        "lane fallback",
    ];
    for i in 0..6 {
        println!("{}: {}", names[i], after[i] - before[i]);
        assert!(after[i] > before[i], "{} never exercised", names[i]);
    }
}

/// Kernels with more than 64 / 128 state nets (multi-word packed states): a small random
/// feedback core plus a long acyclic fan-out of absorbed gates (so the cut stays small).
#[test]
fn wide_state_kernels() {
    let mut rng = Rng(0x1D1D_2E2E_3F3F_4A4A);
    let mut tables = 0;
    for case in 0..40 {
        let (mut sub, mut map) = random_subnet(&mut rng, 4, 5);
        let extra = [70usize, 100, 140, 200][case % 4];
        let k = map.num_inputs();
        for e in 0..extra {
            let n = sub.num_nets as usize;
            let op = [FlatOp::Buf, FlatOp::Not, FlatOp::Nand, FlatOp::Xor][rng.below(4)];
            let arity = if matches!(op, FlatOp::Buf | FlatOp::Not) {
                1
            } else {
                2
            };
            // Read the core, inputs or earlier extra nets only (acyclic).
            let inputs = (0..arity).map(|_| rng.below(n) as NetIdx).collect();
            let gi = sub.gates.len() as u32;
            sub.gates.push(FlatGate {
                op,
                inputs,
                output: n as NetIdx,
                owner: 0,
                group: gi,
                bit: 0,
            });
            sub.groups.push(GateGroup {
                owner: 0,
                path: String::new(),
                gates: vec![gi],
            });
            sub.net_names.push(format!("x{e}"));
            sub.num_nets += 1;
            map.nets.push(n as NetIdx);
        }
        assert_eq!(map.num_state() + k, sub.num_nets as usize);
        let opts = if case % 2 == 0 {
            CharzOptions::default()
        } else {
            random_opts(&mut rng)
        };
        let sub = Arc::new(sub);
        tables += usize::from(compare(&sub, &map, &opts, &format!("wide case {case}")));
    }
    assert!(tables > 20, "{tables}");
}
