//! `difftest [start] [count] [threads]` — run the differential fuzzer over a seed range and print
//! the summary (and details of every clean-but-mismatch design).
//!
//! `difftest show <seed>` — print the STA report and comparison of one design.

use panda_difftest::harness::{Class, HarnessConfig, default_threads, run_seed, run_seeds};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cfg = HarnessConfig::default();
    if args.get(1).map(String::as_str) == Some("show") {
        let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).expect("seed");
        let o = run_seed(seed, &cfg);
        println!("{o:#?}");
        let g = panda_difftest::generate(seed, &cfg.generator, o.scale);
        let flat = panda_netlist::flatten(&g.design).expect("flatten");
        let r = panda_sta::analyze_netlist(&flat);
        println!("min_period {:?}", r.min_period);
        for p in &r.phase_settle {
            println!(
                "  {:?} {}: settle {:?} gap {:?} rounds {} path {:?}",
                p.kind, p.event, p.settle, p.gap, p.rounds, p.critical_path
            );
        }
        for d in &r.diagnostics {
            println!("  {d}");
        }
        if std::env::var("JSON").is_ok() {
            println!("{}", g.design.to_json());
        }
        if std::env::var("DUMP").is_ok() {
            print!("{}", panda_difftest::minimize::describe(&flat));
        }
        return;
    }
    if args.get(1).map(String::as_str) == Some("cpu8") {
        // difftest cpu8 [cycles]
        let cycles: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2000);
        let prog = args.get(3).map_or("fibonacci", String::as_str);
        let image = panda_cpu8::programs::by_name(prog)
            .expect("program")
            .image();
        let period = std::env::var("PERIOD").ok().and_then(|p| p.parse().ok());
        let c = panda_difftest::cpu8::Cpu8Config {
            cycles,
            ideal_blocks: std::env::var("IDEAL").is_ok(),
            period,
        };
        for variant in panda_cpu8::Variant::ALL {
            for rom in [
                panda_cpu8::RomStyle::SumOfProducts,
                panda_cpu8::RomStyle::MuxTree,
                panda_cpu8::RomStyle::Decoder,
            ] {
                let t = std::time::Instant::now();
                let r = panda_difftest::cpu8::run_cpu8(variant, rom, &image, &c).expect("cpu8");
                println!(
                    "{variant:?} {rom:?}: min {} period {} sta_errors {} compared {} mismatches {} outs {} golden {} ok {} ({:.1}s)",
                    r.min_period,
                    r.period,
                    r.sta_errors.len(),
                    r.diff.compared,
                    r.diff.mismatch_count,
                    r.outs.len(),
                    r.golden.len(),
                    r.ok(),
                    t.elapsed().as_secs_f64()
                );
                for m in &r.diff.mismatches {
                    println!("    {m}");
                }
            }
        }
        return;
    }
    if args.get(1).map(String::as_str) == Some("hunt") {
        // difftest hunt <start> <count> <tries>: mutate flagged-and-mismatching designs looking
        // for clean-but-mismatch variants.
        let num = |i: usize| args.get(i).and_then(|s| s.parse::<u64>().ok());
        let (start, count, tries) = (
            num(2).unwrap_or(0),
            num(3).unwrap_or(100),
            num(4).unwrap_or(20) as usize,
        );
        let (_, outcomes) = run_seeds(start..start + count, &cfg, default_threads());
        let todo: Vec<_> = outcomes
            .iter()
            .filter(|o| o.class == Class::FlaggedMismatch)
            .map(|o| (o.seed, o.scale))
            .collect();
        println!("{} flagged-and-mismatching designs", todo.len());
        let next = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|sc| {
            for _ in 0..default_threads() {
                sc.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&(seed, scale)) = todo.get(i) else {
                            break;
                        };
                        let g = panda_difftest::generate(seed, &cfg.generator, scale);
                        if let Some((_, o)) = panda_difftest::minimize::hunt(&g, &cfg, tries, seed)
                        {
                            println!(
                                "FOUND seed {seed} scale {scale}: {:?}",
                                o.diff
                                    .as_ref()
                                    .and_then(|d| d.first())
                                    .map(|m| m.to_string())
                            );
                        }
                    }
                });
            }
        });
        return;
    }
    if args.get(1).map(String::as_str) == Some("huntmin") {
        // difftest huntmin <seed> <scale> <tries>: re-run the hunt for one design and minimize the
        // variant found.
        let seed: u64 = args[2].parse().expect("seed");
        let scale: u32 = args[3].parse().expect("scale");
        let tries: usize = args[4].parse().expect("tries");
        let g = panda_difftest::generate(seed, &cfg.generator, scale);
        let (v, _) =
            panda_difftest::minimize::hunt(&g, &cfg, tries, seed).expect("no variant found");
        let keep = |o: &panda_difftest::Outcome| o.class == Class::CleanMismatch;
        let (m, evals) = panda_difftest::minimize::minimize(&v, &cfg, &keep, 20_000);
        let flat = panda_netlist::flatten(&m.design).expect("flatten");
        println!("{evals} evaluations; {} gates left", flat.gates.len());
        print!("{}", panda_difftest::minimize::describe(&flat));
        let o = panda_difftest::minimize::evaluate(&g, m.design.clone(), &cfg);
        println!(
            "{:?} {:?} {:?}",
            o.class,
            o.runtime_flags,
            o.diff
                .as_ref()
                .and_then(|d| d.first())
                .map(|m| m.to_string())
        );
        if let Some(path) = args.get(5) {
            std::fs::write(path, m.design.to_json()).expect("write");
        }
        return;
    }
    if args.get(1).map(String::as_str) == Some("min") {
        // difftest min <seed> [any]: minimize a clean-but-mismatch design ("any": any mismatch
        // without STA errors).
        let seed: u64 = args[2].parse().expect("seed");
        let any = args.get(3).is_some_and(|a| a == "any");
        let o = run_seed(seed, &cfg);
        let g = panda_difftest::generate(seed, &cfg.generator, o.scale);
        let keep = |o: &panda_difftest::Outcome| {
            if any {
                o.sta_errors.is_empty() && o.diff.as_ref().is_some_and(|d| !d.equal())
            } else {
                o.class == Class::CleanMismatch
            }
        };
        let (m, evals) = panda_difftest::minimize::minimize(&g, &cfg, &keep, 5000);
        let flat = panda_netlist::flatten(&m.design).expect("flatten");
        println!("{evals} evaluations; {} gates left", flat.gates.len());
        print!("{}", panda_difftest::minimize::describe(&flat));
        let o = panda_difftest::minimize::evaluate(&g, m.design.clone(), &cfg);
        println!(
            "{:?} {:?} {:?}",
            o.class,
            o.runtime_flags,
            o.diff
                .as_ref()
                .and_then(|d| d.first())
                .map(|m| m.to_string())
        );
        if std::env::var("JSON").is_ok() {
            println!("{}", m.design.to_json());
        }
        return;
    }
    if args.get(1).map(String::as_str) == Some("trace") {
        // difftest trace <seed> <net,net,...> [from_tick]
        let seed: u64 = args[2].parse().expect("seed");
        // DESIGN=<json> SCALE=<n>: trace a (minimized) design saved by `huntmin` instead.
        let p = if let Ok(path) = std::env::var("DESIGN") {
            let scale: u32 = std::env::var("SCALE")
                .expect("SCALE")
                .parse()
                .expect("SCALE");
            let mut g = panda_difftest::generate(seed, &cfg.generator, scale);
            g.design =
                panda_netlist::Design::from_json(&std::fs::read_to_string(path).expect("read"))
                    .expect("json");
            panda_difftest::harness::prepare_with(seed, &cfg, true, &|_| g.clone())
                .unwrap_or_else(|o| panic!("{o:?}"))
        } else {
            panda_difftest::harness::prepare(seed, &cfg).unwrap_or_else(|o| panic!("{o:?}"))
        };
        let names: Vec<String> = args[3].split(',').map(String::from).collect();
        let from: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        let nets: Vec<u32> = names
            .iter()
            .map(|n| {
                p.flat
                    .net_by_name(n)
                    .unwrap_or_else(|| panic!("no net {n}"))
            })
            .collect();
        let mut c = cfg.clone();
        c.driver.stop_at_first = false;
        c.driver.phases = args
            .get(5)
            .and_then(|s| s.parse().ok())
            .unwrap_or(c.driver.phases);
        println!(
            "{:>8} {:>8} {:12} {}",
            "event",
            "sample",
            "point",
            names.join(" ")
        );
        let o = panda_difftest::harness::finish(p, &c, &mut |f, t, point, _next| {
            if t.tick() < from {
                return;
            }
            let vals: Vec<String> = nets
                .iter()
                .map(|&n| {
                    format!(
                        "{}{}{}",
                        f.get(n).to_char(),
                        t.get(n).to_char(),
                        if f.get(n) == t.get(n) { ' ' } else { '!' }
                    )
                })
                .collect();
            println!(
                "{:>8} {:>8} {:12} {}",
                f.tick(),
                t.tick(),
                format!("{point:?}"),
                vals.join(" ")
            );
        });
        println!("{:?}", o.class);
        return;
    }
    let num = |i: usize, d: u64| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let start = num(1, 0);
    let count = num(2, 500);
    let threads = num(3, default_threads() as u64) as usize;
    let t = std::time::Instant::now();
    let (summary, outcomes) = run_seeds(start..start + count, &cfg, threads);
    for o in &outcomes {
        if o.class == Class::CleanMismatch
            || o.class == Class::Skipped
            || std::env::var("VERBOSE").is_ok()
        {
            println!(
                "seed {} {:?} scale {} gates {} kernels {} sta {:?} rt {:?} features {:?} skip {:?}",
                o.seed,
                o.class,
                o.scale,
                o.gates,
                o.kernels,
                o.sta_errors,
                o.runtime_flags,
                o.features,
                o.skip_reason
            );
            if let Some(d) = &o.diff {
                for m in &d.mismatches {
                    println!("    {m}");
                }
            }
        }
    }
    let mut slow: Vec<_> = outcomes
        .iter()
        .map(|o| (o.millis, o.seed, o.gates, o.features.clone()))
        .collect();
    slow.sort_by(|a, b| b.cmp(a));
    for s in slow.iter().take(5) {
        println!("slow: {} ms seed {} gates {} {:?}", s.0, s.1, s.2, s.3);
    }
    println!("{summary}");
    println!("elapsed {:.1}s", t.elapsed().as_secs_f64());
}
