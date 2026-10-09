//! Panda-8 `bench_loop` throughput, FastRef vs FastSim (docs/M4-CONTRACT.md "Benchmarks"), and
//! profiling tools (hardware counters, probe breakdowns, configuration sweeps).
//!
//! `cargo test --release -p panda-fast --test bench -- --ignored --nocapture panda8_bench_loop`
//! (`PANDA_BENCH_CYCLES`, `PANDA_REPS` = best of N FastSim runs, default 3).

use std::sync::Arc;
use std::time::Instant;

use panda_compile::{FastOptions, FastRef, KernelOptions, compile_kernels};
use panda_cpu8::programs;
use panda_cpu8::{Options, RomStyle, Variant, panda8_with};
use panda_fast::{CompileOptions, FastSim, FastSimOptions, OrderStrategy};
use panda_netlist::flat::SourceKind;
use panda_netlist::flatten;

/// FastSim options for the benchmark, overridable through the environment:
/// `PANDA_LUT_BITS` (0 disables LUTs), `PANDA_LUT_MIN`, `PANDA_LUT_SLACK`, `PANDA_SPEC=0` (no
/// kernel specialization), `PANDA_PART` (partition ops), `PANDA_ORDER=kernelset|conedfs`,
/// `PANDA_FUSE` (gate fusion inputs, 0 = off), `PANDA_LEMMA=0` (keep runtime order checks on
/// O/T data transitions).
fn bench_options() -> FastSimOptions {
    let env = |k: &str| std::env::var(k).ok().and_then(|s| s.parse::<usize>().ok());
    let mut o = FastSimOptions::default();
    if let Some(b) = env("PANDA_LUT_BITS") {
        o.compile.lut_max_bits = b;
    }
    if let Some(s) = env("PANDA_SPEC") {
        o.compile.specialize = s != 0;
    }
    if let Some(f) = env("PANDA_FUSE") {
        o.compile.fuse_inputs = f;
    }
    if let Some(l) = env("PANDA_LEMMA") {
        o.compile.data_order_lemma = l != 0;
    }
    if let Some(p) = env("PANDA_PART") {
        o.compile.partition_ops = p;
    }
    if let Some(g) = env("PANDA_LUT_MIN") {
        o.compile.lut_min_gates = g;
    }
    if let Some(g) = env("PANDA_LUT_SLACK") {
        o.compile.lut_merge_slack = g;
    }
    match std::env::var("PANDA_ORDER").as_deref() {
        Ok("kernelset") => o.compile.order = OrderStrategy::KernelSet,
        Ok("conedfs") => o.compile.order = OrderStrategy::ConeDfs,
        _ => {}
    }
    o
}

#[test]
#[ignore = "benchmark; run in release with --ignored --nocapture"]
fn panda8_bench_loop_cycles_per_second() {
    let cycles: u64 = std::env::var("PANDA_BENCH_CYCLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let reps: usize = std::env::var("PANDA_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    println!(
        "{:<14} {:<18} {:>7} {:>11} {:>9} {:>8} {:>9} {:>9} {:>8} {:>8}",
        "rom",
        "variant",
        "engine",
        "cycles/s",
        "ns/cycle",
        "parts/c",
        "ops/c",
        "evals/c",
        "trans/c",
        "rounds/c"
    );
    for rom in [RomStyle::SumOfProducts, RomStyle::MuxTree] {
        for v in Variant::ALL {
            let opts = Options::new(v).with_rom(rom);
            let flat = Arc::new(
                flatten(&panda8_with(&programs::BENCH_LOOP.image(), &opts)).expect("flatten"),
            );
            let (period, phase, clk) = flat
                .sources
                .iter()
                .find_map(|s| match s.kind {
                    SourceKind::Clock { period, phase, .. } => Some((
                        u64::from(period),
                        u64::from(phase),
                        flat.net_names[s.output as usize].clone(),
                    )),
                    _ => None,
                })
                .unwrap();
            let t0 = Instant::now();
            let (ks, tables) = compile_kernels(&flat, &KernelOptions::default());
            let t_kern = t0.elapsed();
            let t0 = Instant::now();
            let mut fs =
                FastSim::from_parts(flat.clone(), bench_options(), ks.clone(), tables.clone())
                    .unwrap();
            let t_fs = t0.elapsed();
            let mut fr =
                FastRef::from_parts(flat.clone(), FastOptions::default(), ks, tables).unwrap();
            // Past the reset, to just before a rising edge.
            let start = u64::from(opts.reset_ticks()) + 4 * period;
            let start = start - (start + period - phase % period) % period + period - 1;
            fr.run_until(start);
            fs.run_until(start);
            assert_eq!(fr.values(), fs.values());

            let s0 = fr.stats();
            let t = Instant::now();
            fr.run_until(start + cycles * period);
            let fr_secs = t.elapsed().as_secs_f64();
            let s1 = fr.stats();

            let f0 = fs.stats();
            let t = Instant::now();
            fs.run_cycles(&clk, cycles);
            let mut fs_secs = t.elapsed().as_secs_f64();
            let f1 = fs.stats();
            assert_eq!(fs.tick(), fr.tick());
            assert_eq!(fr.values(), fs.values(), "engines diverged");
            // Best of `PANDA_REPS` runs (robust against load from other processes; the loop is
            // periodic, so later runs do the same work).
            for _ in 1..reps {
                let t = Instant::now();
                fs.run_cycles(&clk, cycles);
                fs_secs = fs_secs.min(t.elapsed().as_secs_f64());
            }

            let c = cycles as f64;
            let per = |a: u64, b: u64| (a - b) as f64 / c;
            println!(
                "{:<14} {:<18} {:>7} {:>11.0} {:>9.1} {:>8} {:>9} {:>9.0} {:>8.1} {:>8.2}",
                format!("{rom:?}"),
                format!("{v:?}"),
                "FastRef",
                c / fr_secs,
                fr_secs * 1e9 / c,
                "-",
                "-",
                per(s1.comb_evals, s0.comb_evals),
                per(s1.transitions, s0.transitions),
                per(s1.rounds, s0.rounds),
            );
            println!(
                "{:<14} {:<18} {:>7} {:>11.0} {:>9.1} {:>8.1} {:>9.0} {:>9} {:>8.1} {:>8.2}",
                "",
                "",
                "FastSim",
                c / fs_secs,
                fs_secs * 1e9 / c,
                per(f1.partitions, f0.partitions),
                per(f1.ops, f0.ops),
                "-",
                per(f1.transitions, f0.transitions),
                per(f1.rounds, f0.rounds),
            );
            let st = fs.program().stats();
            println!(
                "    {} gates, {} kernels ({} table), {} ops in {} partitions (max {}), {} notifying, \
                 {} LUTs absorbing {} gates ({} entries); kernels+tables {t_kern:?}, FastSim \
                 compile+reset {t_fs:?}; per cycle {:.2} table hits ({:.1} opaque, {:.1} \
                 transparent), {:.3} sim fallbacks; phase classes O/T/G {:?}; speedup {:.2}x",
                flat.gates.len(),
                st.kernels,
                st.table_kernels,
                st.ops,
                st.partitions,
                st.max_partition,
                st.notifying,
                st.luts,
                st.lut_gates,
                st.lut_entries,
                per(f1.table_hits, f0.table_hits),
                per(f1.opaque_skips, f0.opaque_skips),
                per(f1.transparent, f0.transparent),
                per(f1.sim_fallbacks, f0.sim_fallbacks),
                st.phase_classes,
                fr_secs / fs_secs,
            );
        }
    }
}

#[test]
#[ignore = "benchmark; run in release with --ignored --nocapture"]
fn full_sweep_ns_per_op() {
    for (rom, order) in [RomStyle::SumOfProducts, RomStyle::MuxTree]
        .into_iter()
        .flat_map(|r| [(r, OrderStrategy::ConeDfs), (r, OrderStrategy::KernelSet)])
    {
        let opts = Options::new(Variant::Hierarchical).with_rom(rom);
        let flat =
            Arc::new(flatten(&panda8_with(&programs::BENCH_LOOP.image(), &opts)).expect("flatten"));
        let fopts = FastSimOptions {
            compile: CompileOptions {
                order,
                ..CompileOptions::default()
            },
            ..FastSimOptions::default()
        };
        let mut fs = FastSim::new(flat, fopts).unwrap();
        fs.run_events(100);
        let n = 2000;
        let o0 = fs.stats().ops;
        let t = Instant::now();
        for _ in 0..n {
            fs.bench_full_sweep();
        }
        let secs = t.elapsed().as_secs_f64();
        let ops = fs.stats().ops - o0;
        let st = fs.program().stats();
        println!(
            "{rom:?} {order:?}: full sweep {:.2} us ({} ops: {} gate2, {} gateN, {} args), {:.2} ns/op",
            secs * 1e6 / n as f64,
            st.ops,
            st.gate2,
            st.gate_n,
            fs.program().args.len(),
            secs * 1e9 / ops as f64
        );
    }
}

#[test]
#[ignore = "info"]
fn table_sizes() {
    let opts = Options::new(Variant::Hierarchical);
    let flat = Arc::new(flatten(&panda8_with(&programs::BENCH_LOOP.image(), &opts)).unwrap());
    let fs = FastSim::new(flat, FastSimOptions::default()).unwrap();
    let p = fs.program();
    let mut users = vec![0; p.ktables.len()];
    for k in &p.kernel_ops {
        if let panda_fast::ir::KernelOpKind::KTable { table } = k.kind {
            users[table as usize] += 1;
        }
    }
    for (i, t) in p.ktables.iter().enumerate() {
        println!(
            "table {i}: k={} m={} states={} entries={} ({} KB) used by {} kernels",
            t.num_inputs,
            t.num_nets,
            t.table.num_states(),
            t.entries.len(),
            t.entries.len() * 4 / 1024,
            users[i]
        );
    }
}

#[test]
#[ignore = "benchmark: partition size / order sweep"]
fn partition_size_sweep() {
    let cycles: u64 = std::env::var("PANDA_BENCH_CYCLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);
    let sizes: Vec<usize> = std::env::var("PANDA_SIZES")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 3, 4, 6, 8, 12, 16, 32, 64]);
    for rom in [RomStyle::SumOfProducts, RomStyle::MuxTree] {
        for v in Variant::ALL {
            let opts = Options::new(v).with_rom(rom);
            let flat = Arc::new(
                flatten(&panda8_with(&programs::BENCH_LOOP.image(), &opts)).expect("flatten"),
            );
            let (ks, tables) = compile_kernels(&flat, &KernelOptions::default());
            let clk = flat
                .sources
                .iter()
                .find(|s| matches!(s.kind, SourceKind::Clock { .. }))
                .map(|s| flat.net_names[s.output as usize].clone())
                .unwrap();
            for order in [OrderStrategy::ConeDfs, OrderStrategy::KernelSet] {
                let mut line = format!("{rom:?} {v:?} {order:?}:");
                for &partition_ops in &sizes {
                    let mut fopts = bench_options();
                    fopts.compile.partition_ops = partition_ops;
                    fopts.compile.order = order;
                    let mut fs =
                        FastSim::from_parts(flat.clone(), fopts, ks.clone(), tables.clone())
                            .unwrap();
                    fs.run_cycles(&clk, 50);
                    // Best of 3 runs (robust against load spikes).
                    let mut best = f64::MAX;
                    for _ in 0..3 {
                        let t = Instant::now();
                        fs.run_cycles(&clk, cycles);
                        best = best.min(t.elapsed().as_secs_f64());
                    }
                    line += &format!(" {partition_ops}:{:.0}k", cycles as f64 / best / 1e3);
                }
                println!("{line}");
            }
        }
    }
}

/// Slice packing feasibility (SPEC §8.7): evaluate Panda-8's multi-bit gate groups (2-input
/// AND/OR/XOR families outside LUTs and kernels, ≥ 4 bits) as scalar table ops vs. as
/// gather → two-plane word op → scatter over the byte-per-net value array.
#[test]
#[ignore = "benchmark: slice packing microbenchmark"]
fn slice_packing_microbench() {
    use panda_netlist::Logic;
    use panda_netlist::eval::{lookup2, tables, w_and, w_not, w_or, w_xor};
    use panda_netlist::flat::FlatOp;
    let opts = Options::new(Variant::Hierarchical);
    let flat = Arc::new(flatten(&panda8_with(&programs::BENCH_LOOP.image(), &opts)).unwrap());
    let mut fs = FastSim::new(flat.clone(), FastSimOptions::default()).unwrap();
    fs.run_events(200);
    let values: Vec<Logic> = fs.values().to_vec();
    let ks = fs.kernels();
    let p = fs.program();
    let in_lut: std::collections::HashSet<u32> = p.cold_ops.iter().map(|o| o.output()).collect();
    type Group = (FlatOp, Vec<(u32, u32, u32)>);
    let mut groups: Vec<Group> = Vec::new();
    for g in &flat.groups {
        let gates: Vec<_> = g
            .gates
            .iter()
            .map(|&i| (i, &flat.gates[i as usize]))
            .filter(|(i, gt)| {
                ks.kernel_of_gate[*i as usize].is_none()
                    && !in_lut.contains(&gt.output)
                    && gt.inputs.len() == 2
                    && !matches!(gt.op, FlatOp::Tri | FlatOp::Not | FlatOp::Buf)
            })
            .collect();
        if gates.len() >= 4 && gates.iter().all(|(_, gt)| gt.op == gates[0].1.op) {
            groups.push((
                gates[0].1.op,
                gates
                    .iter()
                    .map(|(_, gt)| (gt.inputs[0], gt.inputs[1], gt.output))
                    .collect(),
            ));
        }
    }
    let bits: usize = groups.iter().map(|g| g.1.len()).sum();
    let t = tables();
    let sel = |op: FlatOp| match op {
        FlatOp::And => (t[0], 0u8),
        FlatOp::Or => (t[1], 1),
        FlatOp::Xor => (t[2], 2),
        FlatOp::Nand => (t[3], 3),
        FlatOp::Nor => (t[4], 4),
        _ => (t[5], 5),
    };
    let reps = 200_000;
    let mut vals = values.clone();
    let tm = Instant::now();
    for _ in 0..reps {
        for (op, gates) in &groups {
            let (tab, _) = sel(*op);
            for &(a, b, o) in gates {
                let v = lookup2(tab, vals[a as usize], vals[b as usize]);
                let slot = &mut vals[o as usize];
                if *slot != v {
                    *slot = v;
                }
            }
        }
        std::hint::black_box(&mut vals);
    }
    let scalar = tm.elapsed().as_secs_f64() / reps as f64;
    let mut vals2 = values.clone();
    let tm = Instant::now();
    for _ in 0..reps {
        for (op, gates) in &groups {
            let (_, kind) = sel(*op);
            let (mut a1, mut b1, mut a2, mut b2) = (0u64, 0u64, 0u64, 0u64);
            for (i, &(a, b, _)) in gates.iter().enumerate() {
                let (pa, pb) = vals2[a as usize].planes();
                let (qa, qb) = vals2[b as usize].planes();
                a1 |= u64::from(pa) << i;
                b1 |= u64::from(pb) << i;
                a2 |= u64::from(qa) << i;
                b2 |= u64::from(qb) << i;
            }
            let (ra, rb) = match kind {
                0 => w_and(a1, b1, a2, b2),
                1 => w_or(a1, b1, a2, b2),
                2 => w_xor(a1, b1, a2, b2),
                3 => {
                    let (x, y) = w_and(a1, b1, a2, b2);
                    w_not(x, y)
                }
                4 => {
                    let (x, y) = w_or(a1, b1, a2, b2);
                    w_not(x, y)
                }
                _ => {
                    let (x, y) = w_xor(a1, b1, a2, b2);
                    w_not(x, y)
                }
            };
            for (i, &(_, _, o)) in gates.iter().enumerate() {
                let v = Logic::from_planes((ra >> i) & 1 == 1, (rb >> i) & 1 == 1);
                let slot = &mut vals2[o as usize];
                if *slot != v {
                    *slot = v;
                }
            }
        }
        std::hint::black_box(&mut vals2);
    }
    let word = tm.elapsed().as_secs_f64() / reps as f64;
    assert_eq!(vals, vals2, "word ops must equal scalar ops");
    println!(
        "slice packing: {} groups, {bits} gates (of {} comb ops): scalar {:.1} ns ({:.2} ns/gate), \
         gather+word+scatter {:.1} ns ({:.2} ns/gate), ratio {:.2}",
        groups.len(),
        p.ops.len(),
        scalar * 1e9,
        scalar * 1e9 / bits as f64,
        word * 1e9,
        word * 1e9 / bits as f64,
        word / scalar
    );
}

/// Compare compiler configurations robustly: configs (from `PANDA_CONFIGS`, `;`-separated
/// `lut_bits,lut_min,slack,partition_ops`) run interleaved, best of `PANDA_REPS` (default 5)
/// runs of `PANDA_BENCH_CYCLES` cycles each, on Panda-8 SumOfProducts Hierarchical and
/// GatedClockRegfile.
#[test]
#[ignore = "benchmark: configuration comparison"]
fn config_compare() {
    let cycles: u64 = std::env::var("PANDA_BENCH_CYCLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);
    let reps: usize = std::env::var("PANDA_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let configs: Vec<Vec<usize>> = std::env::var("PANDA_CONFIGS")
        .unwrap_or_else(|_| "12,24,1,8;10,4,1,8".into())
        .split(';')
        .map(|c| c.split(',').map(|x| x.trim().parse().unwrap()).collect())
        .collect();
    for v in [Variant::Hierarchical, Variant::GatedClockRegfile] {
        let opts = Options::new(v);
        let flat =
            Arc::new(flatten(&panda8_with(&programs::BENCH_LOOP.image(), &opts)).expect("flatten"));
        let (ks, tables) = compile_kernels(&flat, &KernelOptions::default());
        let mut sims: Vec<FastSim> = configs
            .iter()
            .map(|c| {
                let mut o = FastSimOptions::default();
                o.compile.lut_max_bits = c[0];
                o.compile.lut_min_gates = c[1];
                o.compile.lut_merge_slack = c[2];
                o.compile.partition_ops = c[3];
                let mut fs =
                    FastSim::from_parts(flat.clone(), o, ks.clone(), tables.clone()).unwrap();
                fs.run_cycles("clk", 50);
                fs
            })
            .collect();
        let mut best = vec![f64::MAX; sims.len()];
        for _ in 0..reps {
            for (i, fs) in sims.iter_mut().enumerate() {
                let t = Instant::now();
                fs.run_cycles("clk", cycles);
                best[i] = best[i].min(t.elapsed().as_secs_f64());
            }
        }
        for (c, b) in configs.iter().zip(&best) {
            println!("{v:?} {c:?}: {:.0} cycles/s", cycles as f64 / b);
        }
    }
}

/// Minimal `perf_event_open` access (raw syscalls; x86_64 Linux) for counting this thread's
/// user-mode hardware events without an external profiler.
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod hwc {
    use std::fs::File;
    use std::io::Read;
    use std::os::fd::{AsRawFd, FromRawFd};

    /// `PERF_TYPE_HARDWARE` configs.
    pub const CYCLES: u64 = 0;
    /// Retired instructions.
    pub const INSTRUCTIONS: u64 = 1;
    /// Retired branches.
    pub const BRANCHES: u64 = 4;
    /// Mispredicted branches.
    pub const BRANCH_MISSES: u64 = 5;

    /// Open a user-mode counter of `(type, config)` for this thread, or `None` if unavailable.
    pub fn open(typ: u32, config: u64) -> Option<File> {
        let mut attr = [0u8; 128];
        attr[0..4].copy_from_slice(&typ.to_le_bytes());
        attr[4..8].copy_from_slice(&64u32.to_le_bytes()); // PERF_ATTR_SIZE_VER0
        attr[8..16].copy_from_slice(&config.to_le_bytes());
        let flags: u64 = (1 << 5) | (1 << 6); // exclude_kernel, exclude_hv
        attr[40..48].copy_from_slice(&flags.to_le_bytes());
        let ret: i64;
        // SAFETY: perf_event_open(&attr, pid 0, cpu -1, group -1, flags 0); `attr` outlives the
        // call and has the declared size.
        unsafe {
            std::arch::asm!("syscall", inlateout("rax") 298i64 => ret, in("rdi") attr.as_ptr(),
                in("rsi") 0i64, in("rdx") -1i64, in("r10") -1i64, in("r8") 0i64,
                lateout("rcx") _, lateout("r11") _, options(nostack));
        }
        // SAFETY: a fresh file descriptor owned by us.
        (ret >= 0).then(|| unsafe { File::from_raw_fd(ret as i32) })
    }

    /// Current count.
    pub fn read(f: &mut File) -> u64 {
        let mut b = [0u8; 8];
        f.read_exact(&mut b).expect("read counter");
        u64::from_le_bytes(b)
    }

    /// Map counter `f` into the process (enables user-space `rdpmc` for it) and return
    /// `1 +` its hardware counter index, or `None`.
    #[allow(dead_code)]
    pub fn rdpmc_index(f: &File) -> Option<u32> {
        let ret: i64;
        // SAFETY: mmap(NULL, 4096, PROT_READ, MAP_SHARED, fd, 0); the mapping is leaked (it
        // must stay valid while the probes read the counter).
        unsafe {
            std::arch::asm!("syscall", inlateout("rax") 9i64 => ret, in("rdi") 0i64,
                in("rsi") 4096i64, in("rdx") 1i64, in("r10") 1i64,
                in("r8") f.as_raw_fd() as i64, in("r9") 0i64,
                lateout("rcx") _, lateout("r11") _, options(nostack));
        }
        if ret <= 0 {
            return None;
        }
        // SAFETY: the mapped `perf_event_mmap_page`: `index` at offset 12.
        let index = unsafe { std::ptr::read_volatile((ret + 12) as *const u32) };
        (index > 0).then_some(index)
    }
}

/// Hardware counters per simulated cycle (Panda-8 bench_loop; `PANDA_VARIANT=gated|flat`,
/// `PANDA_ROM=mux`): cycles, instructions, IPC, branches, branch misses.
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
#[ignore = "benchmark: hardware counters"]
fn hw_counters() {
    let v = match std::env::var("PANDA_VARIANT").as_deref() {
        Ok("gated") => Variant::GatedClockRegfile,
        Ok("flat") => Variant::Flat,
        _ => Variant::Hierarchical,
    };
    let rom = match std::env::var("PANDA_ROM").as_deref() {
        Ok("mux") => RomStyle::MuxTree,
        _ => RomStyle::SumOfProducts,
    };
    let flat = Arc::new(
        flatten(&panda8_with(
            &programs::BENCH_LOOP.image(),
            &Options::new(v).with_rom(rom),
        ))
        .expect("flatten"),
    );
    let mut fs = FastSim::new(flat, bench_options()).unwrap();
    fs.run_cycles("clk", 1000);
    let events = [
        ("cycles", hwc::CYCLES),
        ("instructions", hwc::INSTRUCTIONS),
        ("branches", hwc::BRANCHES),
        ("branch misses", hwc::BRANCH_MISSES),
    ];
    let mut fds: Vec<_> = events.iter().map(|&(_, c)| hwc::open(0, c)).collect();
    if fds.iter().any(Option::is_none) {
        println!("perf events unavailable");
        return;
    }
    let n = 200_000u64;
    let before: Vec<u64> = fds
        .iter_mut()
        .map(|f| hwc::read(f.as_mut().unwrap()))
        .collect();
    let t = Instant::now();
    fs.run_cycles("clk", n);
    let secs = t.elapsed().as_secs_f64();
    let after: Vec<u64> = fds
        .iter_mut()
        .map(|f| hwc::read(f.as_mut().unwrap()))
        .collect();
    let per = |i: usize| (after[i] - before[i]) as f64 / n as f64;
    println!(
        "{v:?} {rom:?}: {:.0} cycles/s ({:.0} ns/cycle); per simulated cycle: {:.0} CPU cycles, \
         {:.0} instructions (IPC {:.2}), {:.0} branches, {:.1} branch misses",
        n as f64 / secs,
        secs * 1e9 / n as f64,
        per(0),
        per(1),
        per(1) / per(0),
        per(2),
        per(3)
    );
}

/// Breakdown of the event loop from the probes (`--features prof`):
/// `cargo test --release -p panda-fast --features prof --test bench -- --ignored --nocapture
/// profile_breakdown` (`PANDA_VARIANT=gated|flat`). `PANDA_PMC=instructions` counts retired
/// instructions (exact; via `rdpmc`) instead of time-stamp-counter cycles. The probes add
/// ~10–15 instructions each.
#[cfg(all(feature = "prof", target_arch = "x86_64", target_os = "linux"))]
#[test]
#[ignore = "profiling (feature prof)"]
fn profile_breakdown() {
    use panda_fast::probe;
    use std::sync::atomic::Ordering;
    let v = match std::env::var("PANDA_VARIANT").as_deref() {
        Ok("gated") => Variant::GatedClockRegfile,
        Ok("flat") => Variant::Flat,
        _ => Variant::Hierarchical,
    };
    let flat = Arc::new(
        flatten(&panda8_with(
            &programs::BENCH_LOOP.image(),
            &Options::new(v),
        ))
        .expect("flatten"),
    );
    let mut fs = FastSim::new(flat, bench_options()).unwrap();
    fs.run_cycles("clk", 100);
    let instructions = std::env::var("PANDA_PMC").as_deref() == Ok("instructions");
    let mut _keep = None;
    if instructions {
        let f = hwc::open(0, hwc::INSTRUCTIONS).expect("instructions counter");
        let idx = hwc::rdpmc_index(&f).expect("rdpmc");
        probe::PMC.store(idx, Ordering::Relaxed);
        _keep = Some(f);
    }
    probe::reset();
    let n = 50_000u64;
    let s0 = fs.stats();
    let c0 = probe::start();
    let t = Instant::now();
    fs.run_cycles("clk", n);
    let secs = t.elapsed().as_secs_f64();
    let total = probe::start().wrapping_sub(c0) as f64 / n as f64;
    let s1 = fs.stats();
    probe::PMC.store(0, Ordering::Relaxed);
    let per = |a: u64, b: u64| (a - b) as f64 / n as f64;
    println!(
        "{v:?}: {:.0} ns/cycle; per cycle {:.1} ops, {:.1} partitions, {:.2} rounds, {:.1} \
         transitions ({:.1} opaque, {:.1} transparent, {:.1} per-kernel fast evals)",
        secs * 1e9 / n as f64,
        per(s1.ops, s0.ops),
        per(s1.partitions, s0.partitions),
        per(s1.rounds, s0.rounds),
        per(s1.transitions, s0.transitions),
        per(s1.opaque_skips, s0.opaque_skips),
        per(s1.transparent, s0.transparent),
        per(s1.fast_evals, s0.fast_evals),
    );
    let unit = if instructions {
        "instructions"
    } else {
        "TSC cycles"
    };
    println!("  total: {total:.0} {unit}/cycle");
    for (slot, name) in [
        (probe::TIMED, "apply timed sources"),
        (probe::SWEEP, "sweeps"),
        (probe::LUT, "  of which LUT ops"),
        (probe::DECIDE, "round-boundary decisions"),
        (probe::GENERAL, "  of which general transitions"),
        (probe::WRITE, "state writes"),
        (probe::FLUSH, "change propagation (inside the above)"),
    ] {
        let c = probe::COUNTERS[slot].load(Ordering::Relaxed) as f64 / n as f64;
        println!("  {name}: {c:.0} {unit}/cycle");
    }
    println!(
        "  net changes propagated: {:.1}/cycle",
        probe::COUNTERS[probe::NOTIFIES].load(Ordering::Relaxed) as f64 / n as f64
    );
}

/// Steady-state throughput of one configuration for `PANDA_SECS` seconds (default 2;
/// `PANDA_VARIANT=gated|flat`, `PANDA_ROM=mux`, plus the [`bench_options`] variables).
#[test]
#[ignore = "benchmark: steady-state spin"]
fn spin() {
    let v = match std::env::var("PANDA_VARIANT").as_deref() {
        Ok("gated") => Variant::GatedClockRegfile,
        Ok("flat") => Variant::Flat,
        _ => Variant::Hierarchical,
    };
    let rom = match std::env::var("PANDA_ROM").as_deref() {
        Ok("mux") => RomStyle::MuxTree,
        _ => RomStyle::SumOfProducts,
    };
    let flat = Arc::new(
        flatten(&panda8_with(
            &programs::BENCH_LOOP.image(),
            &Options::new(v).with_rom(rom),
        ))
        .expect("flatten"),
    );
    let _fr = std::env::var_os("PANDA_WITH_FASTREF").map(|_| {
        let (ks, tables) = compile_kernels(&flat, &KernelOptions::default());
        let mut fr = FastRef::from_parts(flat.clone(), FastOptions::default(), ks, tables).unwrap();
        fr.run_until(2000);
        fr
    });
    let mut fs = if std::env::var_os("PANDA_FROM_PARTS").is_some() {
        let (ks, tables) = compile_kernels(&flat, &KernelOptions::default());
        FastSim::from_parts(flat, bench_options(), ks, tables).unwrap()
    } else {
        FastSim::new(flat, bench_options()).unwrap()
    };
    fs.run_cycles("clk", 100);
    let secs: f64 = std::env::var("PANDA_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2.0);
    let chunk: u64 = std::env::var("PANDA_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);
    let t = Instant::now();
    let mut n = 0u64;
    while t.elapsed().as_secs_f64() < secs {
        fs.run_cycles("clk", chunk);
        n += chunk;
    }
    println!(
        "{v:?} {rom:?}: {:.0} cycles/s",
        n as f64 / t.elapsed().as_secs_f64()
    );
}
