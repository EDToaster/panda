//! Panda-8 `bench_loop` throughput: interpreter vs JIT (SPEC §13 M5: ≥ 2× over the
//! interpreter), plus JIT compile times.
//!
//! `cargo test --release -p panda-jit --test bench -- --ignored --nocapture`
//! (`PANDA_BENCH_CYCLES`, default 200000; `PANDA_REPS` = best of N, default 5).

use std::sync::Arc;
use std::time::Instant;

use panda_cpu8::programs;
use panda_cpu8::{Options, RomStyle, Variant, panda8_with};
use panda_fast::{FastSim, FastSimOptions};
use panda_jit::{Jit, JitOptions};
use panda_netlist::flatten;

fn env(k: &str, d: u64) -> u64 {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

/// Best-of-`reps` seconds for `cycles` cycles.
fn time(fs: &mut FastSim, cycles: u64, reps: u64) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        fs.run_cycles("clk", cycles);
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

#[test]
#[ignore = "benchmark; run in release with --ignored --nocapture"]
fn panda8_bench_loop_interpreter_vs_jit() {
    let cycles = env("PANDA_BENCH_CYCLES", 200_000);
    let reps = env("PANDA_REPS", 5);
    println!(
        "{:<14} {:<18} {:>12} {:>12} {:>12} {:>9} {:>9} {:>9} {:>8}",
        "rom",
        "variant",
        "interp c/s",
        "jit-part c/s",
        "jit c/s",
        "ns/c int",
        "ns/c jit",
        "compile",
        "speedup"
    );
    for rom in [RomStyle::SumOfProducts, RomStyle::MuxTree] {
        for v in Variant::ALL {
            let opts = Options::new(v).with_rom(rom);
            let flat = Arc::new(
                flatten(&panda8_with(&programs::BENCH_LOOP.image(), &opts)).expect("flatten"),
            );
            let mut a = FastSim::new(flat.clone(), FastSimOptions::default()).unwrap();
            let mut b = FastSim::new(flat.clone(), FastSimOptions::default()).unwrap();
            let mut c = FastSim::new(flat, FastSimOptions::default()).unwrap();
            let jit = Jit::new(JitOptions::default());
            let stats = jit.install(&mut b).expect("jit");
            let part_only = Jit::new(JitOptions {
                events: false,
                ..JitOptions::default()
            });
            part_only.install(&mut c).expect("jit");
            for s in [&mut a, &mut b, &mut c] {
                s.run_cycles("clk", 1000);
            }
            let ti = time(&mut a, cycles, reps);
            let tj = time(&mut b, cycles, reps);
            let tp = time(&mut c, cycles, reps);
            assert_eq!(a.values(), b.values(), "JIT diverged");
            assert_eq!(a.values(), c.values(), "JIT diverged");
            let cyc = cycles as f64;
            println!(
                "{:<14} {:<18} {:>12.0} {:>12.0} {:>12.0} {:>9.1} {:>9.1} {:>7.1}ms {:>7.2}x",
                format!("{rom:?}"),
                format!("{v:?}"),
                cyc / ti,
                cyc / tp,
                cyc / tj,
                ti * 1e9 / cyc,
                tj * 1e9 / cyc,
                stats.compile_time.as_secs_f64() * 1e3,
                ti / tj
            );
            println!("    jit: {stats:?}");
            println!("    engine stats per cycle: {:?}", b.stats());
        }
    }
}

/// Probe breakdown of the event loop with the JIT installed (`--features prof`):
/// `cargo test --release -p panda-jit --features prof --test bench -- --ignored --nocapture
/// profile_jit` (`PANDA_JIT_ROUNDS=0` for partitions only, `PANDA_NO_JIT=1` for the interpreter).
#[cfg(all(feature = "prof", target_arch = "x86_64"))]
#[test]
#[ignore = "profiling (feature prof)"]
fn profile_jit() {
    use panda_fast::probe;
    use std::sync::atomic::Ordering;
    let flat = Arc::new(
        flatten(&panda8_with(
            &programs::BENCH_LOOP.image(),
            &Options::new(Variant::Hierarchical),
        ))
        .expect("flatten"),
    );
    let mut fs = FastSim::new(flat, FastSimOptions::default()).unwrap();
    if std::env::var_os("PANDA_NO_JIT").is_none() {
        let jit = Jit::new(JitOptions {
            events: env("PANDA_JIT_EVENTS", 1) != 0,
            ..JitOptions::default()
        });
        jit.install(&mut fs).unwrap();
    }
    fs.run_cycles("clk", 1000);
    let mut _keep = None;
    if std::env::var("PANDA_PMC").as_deref() == Ok("instructions") {
        let f = hwc::open(0, hwc::INSTRUCTIONS).expect("instructions counter");
        let idx = hwc::rdpmc_index(&f).expect("rdpmc");
        probe::PMC.store(idx, Ordering::Relaxed);
        _keep = Some(f);
    }
    probe::reset();
    let n = 100_000u64;
    let c0 = probe::start();
    let t = Instant::now();
    fs.run_cycles("clk", n);
    let secs = t.elapsed().as_secs_f64();
    let total = probe::start().wrapping_sub(c0) as f64 / n as f64;
    probe::PMC.store(0, Ordering::Relaxed);
    println!(
        "{:.0} ns/cycle; total {total:.0} units/cycle",
        secs * 1e9 / n as f64
    );
    for (slot, name) in [
        (probe::TIMED, "apply timed sources"),
        (probe::SWEEP, "sweeps"),
        (probe::LUT, "  of which LUT ops"),
        (probe::DECIDE, "round-boundary decisions"),
        (probe::GENERAL, "  of which general transitions"),
        (probe::FAST, "  of which fast paths"),
        (probe::WRITE, "state writes"),
        (probe::WFAST, "  of which fast-path output writes"),
        (
            probe::FLUSH,
            "record-driven change propagation (inside the above)",
        ),
        (probe::NATIVE, "native event functions"),
    ] {
        let c = probe::COUNTERS[slot].load(Ordering::Relaxed) as f64 / n as f64;
        println!("  {name}: {c:.0}");
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

/// Hardware counters per simulated cycle for the interpreter, the partition-only JIT and the
/// full JIT (Panda-8 bench_loop, Hierarchical; `PANDA_VARIANT=gated` for GatedClockRegfile).
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
#[ignore = "benchmark: hardware counters"]
fn hw_counters_jit() {
    let v = match std::env::var("PANDA_VARIANT").as_deref() {
        Ok("gated") => Variant::GatedClockRegfile,
        _ => Variant::Hierarchical,
    };
    let flat = Arc::new(
        flatten(&panda8_with(
            &programs::BENCH_LOOP.image(),
            &Options::new(v),
        ))
        .expect("flatten"),
    );
    let mut fopts = FastSimOptions::default();
    if let Some(p) = std::env::var("PANDA_PART")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        fopts.compile.partition_ops = p;
    }
    if let Some(p) = std::env::var("PANDA_FUSE")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        fopts.compile.fuse_inputs = p;
    }
    for (name, jit) in [
        ("interpreter", None),
        (
            "jit partitions",
            Some(JitOptions {
                events: false,
                ..JitOptions::default()
            }),
        ),
        (
            "jit events only",
            Some(JitOptions {
                partitions: false,
                ..JitOptions::default()
            }),
        ),
        ("jit", Some(JitOptions::default())),
    ] {
        let mut fs = FastSim::new(flat.clone(), fopts.clone()).unwrap();
        if let Some(o) = jit {
            Jit::new(o).install(&mut fs).unwrap();
        }
        fs.run_cycles("clk", 1000);
        let n = 100_000u64;
        // (type, config): cycles, instructions, branches, branch misses, L1I misses.
        let events = [
            (0, hwc::CYCLES),
            (0, hwc::INSTRUCTIONS),
            (0, hwc::BRANCHES),
            (0, hwc::BRANCH_MISSES),
            (3, 0x1_0001),
        ];
        let mut vals = Vec::new();
        for &(t, e) in &events {
            let Some(mut f) = hwc::open(t, e) else {
                vals.push(f64::NAN);
                continue;
            };
            let a = hwc::read(&mut f);
            fs.run_cycles("clk", n);
            vals.push((hwc::read(&mut f) - a) as f64 / n as f64);
        }
        println!(
            "{name:<16} cycles {:>7.0} instr {:>7.0} IPC {:>4.2} branches {:>6.0} misses {:>5.1} \
             L1I misses {:>6.1}",
            vals[0],
            vals[1],
            vals[1] / vals[0],
            vals[2],
            vals[3],
            vals[4]
        );
    }
}

/// Full combinational sweeps (every partition dirty): ns per op, interpreter vs JIT.
#[test]
#[ignore = "benchmark: full sweeps"]
fn full_sweep_jit() {
    let flat = Arc::new(
        flatten(&panda8_with(
            &programs::BENCH_LOOP.image(),
            &Options::new(Variant::Hierarchical),
        ))
        .expect("flatten"),
    );
    for jit in [false, true] {
        let mut fs = FastSim::new(flat.clone(), FastSimOptions::default()).unwrap();
        if jit {
            Jit::new(JitOptions::default()).install(&mut fs).unwrap();
        }
        fs.run_cycles("clk", 100);
        let ops = fs.program().ops.len() as f64;
        let n = 200_000;
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            for _ in 0..n {
                fs.bench_full_sweep();
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        {
            let mut v = Vec::new();
            for e in [hwc::CYCLES, hwc::INSTRUCTIONS] {
                if let Some(mut f) = hwc::open(0, e) {
                    let a = hwc::read(&mut f);
                    for _ in 0..n {
                        fs.bench_full_sweep();
                    }
                    v.push((hwc::read(&mut f) - a) as f64 / n as f64);
                }
            }
            if let [c, i] = v[..] {
                println!(
                    "  per sweep: {c:.0} cycles, {i:.0} instructions, IPC {:.2}",
                    i / c
                );
            }
        }
        println!(
            "{}: {:.0} ns/sweep, {:.2} ns/op",
            if jit { "jit" } else { "interpreter" },
            best * 1e9 / n as f64,
            best * 1e9 / n as f64 / ops
        );
    }
}
