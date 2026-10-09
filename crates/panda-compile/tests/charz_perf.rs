//! Compile-latency benchmark of kernel extraction + characterization (SPEC §8.3–§8.4).
//!
//! `cargo test --release -p panda-compile --test charz_perf -- --ignored --nocapture`

use std::time::{Duration, Instant};

use panda_compile::charz::{CharzOptions, TableCache, characterize, characterize_subnet_with};

/// Threads for the parallel measurements (`PANDA_THREADS`, default: available parallelism).
fn threads() -> usize {
    std::env::var("PANDA_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(panda_compile::charz::default_threads)
}
use panda_compile::kernels::{ExtractOptions, extract};
use panda_compile::{KernelOptions, compile_kernels_with};
use panda_cpu8::{Options, RomStyle, Variant, panda8_with, programs};
use panda_netlist::flat::{FlatNetlist, SourceKind};
use panda_netlist::flatten;

/// Disjoint union of `n` copies of `flat` (indices offset; names not unique).
fn replicate(flat: &FlatNetlist, n: usize) -> FlatNetlist {
    let mut out = FlatNetlist::default();
    for c in 0..n {
        let no = out.num_nets;
        let io = out.instances.len() as u32;
        let go = out.gates.len() as u32;
        let grp = out.groups.len() as u32;
        let pin_o = out.top_inputs.len() as u32;
        for g in &flat.gates {
            let mut g = g.clone();
            g.inputs.iter_mut().for_each(|i| *i += no);
            g.output += no;
            g.owner += io;
            g.group += grp;
            out.gates.push(g);
        }
        for gr in &flat.groups {
            let mut gr = gr.clone();
            gr.owner += io;
            gr.gates.iter_mut().for_each(|x| *x += go);
            out.groups.push(gr);
        }
        for r in &flat.resolvers {
            let mut r = r.clone();
            r.drivers.iter_mut().for_each(|i| *i += no);
            r.output += no;
            out.resolvers.push(r);
        }
        for s in &flat.sources {
            let mut s = s.clone();
            s.output += no;
            if let SourceKind::TopInput { pin, .. } = &mut s.kind {
                *pin += pin_o;
            }
            out.sources.push(s);
        }
        for i in &flat.instances {
            let mut i = i.clone();
            i.parent = i.parent.map(|p| p + io);
            i.children.iter_mut().for_each(|x| *x += io);
            i.path = format!("c{c}/{}", i.path);
            out.instances.push(i);
        }
        for p in &flat.top_inputs {
            let mut p = p.clone();
            p.name = format!("c{c}_{}", p.name);
            p.nets.iter_mut().for_each(|x| *x += no);
            out.top_inputs.push(p);
        }
        for p in &flat.top_outputs {
            let mut p = p.clone();
            p.name = format!("c{c}_{}", p.name);
            p.nets.iter_mut().for_each(|x| *x += no);
            out.top_outputs.push(p);
        }
        for (k, v) in &flat.undriven_termination {
            out.undriven_termination.insert(k + no, *v);
        }
        out.net_names
            .extend(flat.net_names.iter().map(|s| format!("c{c}/{s}")));
        out.num_nets += flat.num_nets;
    }
    out
}

fn panda8(variant: Variant, rom: RomStyle) -> FlatNetlist {
    flatten(&panda8_with(
        &programs::BENCH_LOOP.image(),
        &Options::new(variant).with_rom(rom),
    ))
    .expect("flatten")
}

/// Best of `reps` runs of `f`.
fn best<T>(reps: usize, mut f: impl FnMut() -> T) -> (Duration, T) {
    let mut out = None;
    let mut best = Duration::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        let r = f();
        best = best.min(t.elapsed());
        out = Some(r);
    }
    (best, out.expect("reps > 0"))
}

/// Reference characterization (the pre-fast-path algorithm, one thread) of every distinct
/// kernel, with a structural-key cache like the old `TableCache`.
fn reference_all(flat: &FlatNetlist, ks: &panda_compile::KernelSet) -> usize {
    let mut seen = std::collections::HashMap::new();
    for k in &ks.kernels {
        if k.kind != panda_compile::KernelKind::Table {
            continue;
        }
        let (sub, map) = panda_compile::kernel_subnet(flat, k);
        let key = panda_compile::subnet::structural_key(&sub);
        seen.entry(key).or_insert_with(|| {
            panda_compile::charz::characterize_subnet_reference(
                std::sync::Arc::new(sub),
                map,
                &CharzOptions::default(),
            )
        });
    }
    seen.len()
}

fn measure(name: &str, flat: &FlatNetlist, with_reference: bool) {
    let opts = KernelOptions::default();
    let (dx, ks) = best(3, || extract(flat, &ExtractOptions::default()));
    let (dr, distinct) = if with_reference {
        best(1, || reference_all(flat, &ks))
    } else {
        (Duration::ZERO, 0)
    };
    let (d1, _) = best(3, || {
        compile_kernels_with(flat, &opts, &TableCache::new(), 1)
    });
    let (dn, _) = best(3, || {
        compile_kernels_with(flat, &opts, &TableCache::new(), threads())
    });
    let warm = TableCache::new();
    compile_kernels_with(flat, &opts, &warm, threads());
    let (dw, _) = best(3, || compile_kernels_with(flat, &opts, &warm, threads()));
    println!(
        "{name}: {} gates, {} kernels ({distinct} distinct): extract {dx:.2?}; extract+charz: \
         reference {:.1?}, new 1 thread {d1:.1?}, new {} threads {dn:.1?}, warm cache {dw:.2?}",
        flat.gates.len(),
        ks.kernels.len(),
        dr + dx,
        threads()
    );
}

#[test]
#[ignore = "benchmark"]
fn charz_latency_panda8() {
    for rom in [RomStyle::SumOfProducts, RomStyle::MuxTree] {
        for variant in Variant::ALL {
            measure(
                &format!("panda8 {variant:?} {rom:?}"),
                &panda8(variant, rom),
                true,
            );
        }
    }
    let one = panda8(Variant::Hierarchical, RomStyle::SumOfProducts);
    measure(
        "26x panda8 Hierarchical SumOfProducts",
        &replicate(&one, 26),
        true,
    );
}

/// Difftest designs: the slowest seeds of `0..PANDA_SEEDS` (default 400) by new-path time, then
/// reference vs new on those.
#[test]
#[ignore = "benchmark"]
fn charz_latency_difftest() {
    use panda_difftest::generator::{GenConfig, generate};
    let n: u64 = std::env::var("PANDA_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(400);
    let cfg = GenConfig::default();
    let mut rows = Vec::new();
    let mut total = Duration::ZERO;
    for seed in 0..n {
        let Ok(flat) = flatten(&generate(seed, &cfg, 2).design) else {
            continue;
        };
        let t = Instant::now();
        compile_kernels_with(&flat, &KernelOptions::default(), &TableCache::new(), 1);
        let d = t.elapsed();
        total += d;
        rows.push((d, seed, flat));
    }
    let mut total_n = Duration::ZERO;
    let mut total_ref = Duration::ZERO;
    for (_, _, flat) in &rows {
        let t = Instant::now();
        compile_kernels_with(
            flat,
            &KernelOptions::default(),
            &TableCache::new(),
            threads(),
        );
        total_n += t.elapsed();
        if std::env::var("PANDA_REF_TOTAL").is_ok() {
            let t = Instant::now();
            let ks = extract(flat, &ExtractOptions::default());
            reference_all(flat, &ks);
            total_ref += t.elapsed();
        }
    }
    println!(
        "{} seeds: extract+charz total: new 1 thread {total:.1?}, new {} threads {total_n:.1?}, \
         reference {total_ref:.1?} (PANDA_REF_TOTAL to measure)",
        rows.len(),
        threads()
    );
    rows.sort_by_key(|r| std::cmp::Reverse(r.0));
    let mut seeds: Vec<u64> = rows.iter().take(6).map(|r| r.1).collect();
    if !seeds.contains(&290) && n > 290 {
        seeds.push(290);
    }
    for seed in seeds {
        let flat = &rows.iter().find(|r| r.1 == seed).expect("seed").2;
        let ks = extract(flat, &ExtractOptions::default());
        let bits = ks.kernels.iter().map(|k| k.bits()).max().unwrap_or(0);
        measure(
            &format!("difftest seed {seed} (max kernel bits {bits})"),
            flat,
            true,
        );
    }
}

/// Per-kernel characterization cost of the slowest distinct kernels of Panda-8.
#[test]
#[ignore = "benchmark"]
fn charz_slowest_kernels_panda8() {
    let flat = panda8(Variant::Flat, RomStyle::SumOfProducts);
    let ks = extract(&flat, &ExtractOptions::default());
    let mut seen = std::collections::HashSet::new();
    let mut rows = Vec::new();
    for k in &ks.kernels {
        let (sub, _) = panda_compile::kernel_subnet(&flat, k);
        if !seen.insert(panda_compile::subnet::structural_key(&sub)) {
            continue;
        }
        let (sub, map) = panda_compile::kernel_subnet(&flat, k);
        let sub = std::sync::Arc::new(sub);
        let t1 = Instant::now();
        let tab = characterize_subnet_with(sub.clone(), map.clone(), &CharzOptions::default(), 1);
        let d1 = t1.elapsed();
        let t = Instant::now();
        let _ = characterize_subnet_with(sub, map, &CharzOptions::default(), threads());
        let dn = t.elapsed();
        rows.push((
            d1,
            dn,
            k.inputs.len(),
            k.cut.len(),
            k.nets.len(),
            tab.map_or(0, |t| t.num_states()),
        ));
    }
    rows.sort_by_key(|r| std::cmp::Reverse(r.0));
    let total: Duration = rows.iter().map(|r| r.0).sum();
    let total_n: Duration = rows.iter().map(|r| r.1).sum();
    println!(
        "{} distinct kernels, sequential sum: 1 thread {total:?}, {} threads {total_n:?}",
        rows.len(),
        threads()
    );
    for (d, dn, k, c, m, s) in rows.iter().take(12) {
        println!(
            "  {d:?} (1 thread) / {dn:?} ({} threads): {k} inputs, {c} cut, {m} nets, {s} states",
            threads()
        );
    }
}

#[test]
#[ignore = "profiling loop"]
fn charz_profile_loop() {
    let flat = panda8(Variant::Flat, RomStyle::SumOfProducts);
    let ks = extract(&flat, &ExtractOptions::default());
    let big: Vec<_> = ks.kernels.iter().filter(|k| k.inputs.len() >= 9).collect();
    let t = Instant::now();
    while t.elapsed().as_secs() < 20 {
        for k in &big {
            std::hint::black_box(characterize(&flat, k, &CharzOptions::default()));
        }
    }
}

/// Per-kernel diagnosis of difftest seeds (`PANDA_SEED_LIST`, comma-separated).
#[test]
#[ignore = "diagnostic"]
fn charz_difftest_kernels() {
    use panda_difftest::generator::{GenConfig, generate};
    let list = std::env::var("PANDA_SEED_LIST").unwrap_or_else(|_| "290,395,303".into());
    for seed in list.split(',').filter_map(|s| s.trim().parse::<u64>().ok()) {
        let flat = flatten(&generate(seed, &GenConfig::default(), 2).design).expect("flatten");
        let ks = extract(&flat, &ExtractOptions::default());
        let mut seen = std::collections::HashSet::new();
        println!("seed {seed}:");
        for k in &ks.kernels {
            if k.kind != panda_compile::KernelKind::Table {
                continue;
            }
            let (sub, map) = panda_compile::kernel_subnet(&flat, k);
            if !seen.insert(panda_compile::subnet::structural_key(&sub)) {
                continue;
            }
            let opts = CharzOptions::default();
            let fast = panda_compile::charz::uses_fast_path(&sub, &map, &opts);
            let sub = std::sync::Arc::new(sub);
            let c0 = panda_compile::charz::fast_path_coverage();
            let t = Instant::now();
            let tab = characterize_subnet_with(sub.clone(), map.clone(), &opts, 1);
            let d1 = t.elapsed();
            let c1 = panda_compile::charz::fast_path_coverage();
            let cov: Vec<u64> = c1.iter().zip(&c0).map(|(a, b)| a - b).collect();
            let t = Instant::now();
            let _ = characterize_subnet_with(sub.clone(), map.clone(), &opts, threads());
            let dn = t.elapsed();
            println!(
                "  kernel {}: {} inputs, {} nets, {} cut, {} gates, {} resolvers, fast {fast}: \
                 1 thread {d1:.1?}, {} threads {dn:.1?}; states {:?}, partial {:?}, stop {:?}, \
                 coverage {cov:?}",
                k.id,
                k.inputs.len(),
                k.nets.len(),
                k.cut.len(),
                sub.gates.len(),
                sub.resolvers.len(),
                threads(),
                tab.as_ref().map(|t| t.num_states()),
                tab.as_ref().map(|t| t.order_check_partial),
                tab.as_ref().map(|t| t.order_check_stop.is_some()),
            );
        }
    }
}
