//! Panda-8 throughput of every engine of the stack, plus compile times (numbers for README.md).
//!
//! ```text
//! cargo run --release -p panda-difftest --example engines [cycles] [reps]
//! ```
//!
//! Runs `bench_loop` on Panda-8 (each ROM style × variant) at its STA minimum clock period and
//! prints, per engine, the best-of-`reps` throughput in clock cycles per second:
//!
//! * `timing`  — Timing mode (`panda_timing::TimingSim`, glass-box, unit delay, event-driven);
//! * `fastref` — FastRef (`panda_compile::FastRef`, the §6.2 reference interpreter);
//! * `fastsim` — the compiled interpreter (`panda_fast::FastSim`);
//! * `jit`     — `FastSim` with the Cranelift JIT installed (`panda_jit`);
//! * `facade`  — `panda_kernel::Kernel` in Fast mode (default `FastBackend::Jit`), `Kernel::run`.
//!
//! Compile columns: kernel extraction + characterization, FastSim program compilation, JIT
//! compilation, and the facade's load-to-Fast-ready time (synchronous compilation).
//! Timing and FastRef run fewer cycles (`cycles / 50`, `cycles / 10`): they are much slower.

use std::sync::Arc;
use std::time::{Duration, Instant};

use panda_compile::{FastOptions, FastRef, KernelOptions as CompileOptions, compile_kernels};
use panda_cpu8::{Options, RomStyle, Variant, panda8_with, programs};
use panda_fast::{FastSim, FastSimOptions};
use panda_jit::Jit;
use panda_kernel::{Kernel, KernelOptions, Mode, StepUnit};
use panda_netlist::{FlatNetlist, flatten};
use panda_timing::{TimingOptions, TimingSim};

/// Best of `reps` runs of `f` (which simulates `cycles` cycles): cycles per second.
fn best(reps: u32, cycles: u64, mut f: impl FnMut()) -> f64 {
    let mut t = f64::MAX;
    for _ in 0..reps {
        let t0 = Instant::now();
        f();
        t = t.min(t0.elapsed().as_secs_f64());
    }
    cycles as f64 / t
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn fmt_rate(r: f64) -> String {
    if r >= 1e6 {
        format!("{:.2} M", r / 1e6)
    } else if r >= 1e3 {
        format!("{:.1} k", r / 1e3)
    } else {
        format!("{r:.0}")
    }
}

/// STA minimum period of `variant`/`rom`, rounded up to an even value ≥ 8.
fn min_period(image: &[u8], base: &Options) -> u32 {
    let flat = flatten(&panda8_with(image, base)).expect("flatten");
    let (ks, tables) = compile_kernels(&flat, &CompileOptions::default());
    panda_sta::analyze(&flat, &ks, &tables)
        .max_min_period()
        .expect("min period")
        .max(8)
        .next_multiple_of(2)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cycles: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let reps: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let image = programs::BENCH_LOOP.image();
    println!(
        "Panda-8 bench_loop, best of {reps}; cycles: fast engines {cycles}, FastRef {}, Timing {}",
        cycles / 10,
        cycles / 50
    );
    println!(
        "{:<14} {:<18} {:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8} {:>8} {:>8}",
        "rom",
        "variant",
        "gates",
        "period",
        "timing",
        "fastref",
        "fastsim",
        "jit",
        "facade",
        "kern ms",
        "prog ms",
        "jit ms",
        "ready ms"
    );
    for rom in [RomStyle::SumOfProducts, RomStyle::MuxTree] {
        for variant in Variant::ALL {
            let base = Options::new(variant).with_rom(rom);
            let period = min_period(&image, &base);
            let opts = base.with_period(period);
            let design = panda8_with(&image, &opts);
            let flat: Arc<FlatNetlist> = Arc::new(flatten(&design).expect("flatten"));
            let p = u64::from(period);

            // Timing mode: `cycles / 50` cycles from power-on.
            let tc = (cycles / 50).max(1);
            let timing = best(reps, tc, || {
                let mut t = TimingSim::new(
                    flat.clone(),
                    TimingOptions::default().with_stuck_x_periods(0),
                );
                t.reset();
                t.run(tc * p);
            });

            // Compile times.
            let t0 = Instant::now();
            let (ks, tables) = compile_kernels(&flat, &CompileOptions::default());
            let kern = t0.elapsed();
            let t0 = Instant::now();
            let mut sim = FastSim::from_parts(
                flat.clone(),
                FastSimOptions::default(),
                ks.clone(),
                tables.clone(),
            )
            .expect("FastSim");
            let prog = t0.elapsed();
            let mut jsim = FastSim::from_parts(
                flat.clone(),
                FastSimOptions::default(),
                ks.clone(),
                tables.clone(),
            )
            .expect("FastSim");
            let jit_stats = Jit::default().install(&mut jsim).expect("JIT");

            // FastRef: `cycles / 10` cycles (two events per cycle).
            let rc = (cycles / 10).max(1);
            let mut fr = FastRef::from_parts(flat.clone(), FastOptions::default(), ks, tables)
                .expect("FastRef");
            let fastref = best(reps, rc, || fr.run_events(2 * rc));
            let fastsim = best(reps, cycles, || sim.run_cycles("clk", cycles));
            let jit = best(reps, cycles, || jsim.run_cycles("clk", cycles));

            // Facade: load → Fast mode ready (synchronous compile with a cold table cache), then
            // `Kernel::step` by cycles.
            let t0 = Instant::now();
            let mut k = Kernel::new(KernelOptions {
                background: false,
                ..KernelOptions::default()
            });
            k.set_mode(Mode::Fast);
            k.load(design.clone()).expect("load");
            k.wait_ready().expect("ready");
            let ready = t0.elapsed();
            assert!(k.jit_active(), "facade JIT active");
            let facade = best(reps, cycles, || k.step(cycles, StepUnit::Cycle));

            println!(
                "{:<14} {:<18} {:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8.1} {:>8.1} {:>8.1} {:>8.1}",
                format!("{rom:?}"),
                format!("{variant:?}"),
                flat.gates.len(),
                period,
                fmt_rate(timing),
                fmt_rate(fastref),
                fmt_rate(fastsim),
                fmt_rate(jit),
                fmt_rate(facade),
                ms(kern),
                ms(prog),
                ms(jit_stats.compile_time),
                ms(ready)
            );
        }
    }
}
