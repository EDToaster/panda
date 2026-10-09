//! Timing-mode throughput benchmark:
//! `cargo test --release -p panda-cpu8 --test bench -- --ignored --nocapture`

use std::time::Instant;

use panda_cpu8::emu::Emulator;
use panda_cpu8::programs;
use panda_cpu8::sim::GateCpu;
use panda_cpu8::{Options, RomStyle, Variant, panda8_with};

#[test]
#[ignore = "benchmark; run in release with --ignored --nocapture"]
fn timing_mode_cycles_per_second() {
    let program = programs::BENCH_LOOP.image();
    let cycles = 20_000u64;
    for rom in [RomStyle::SumOfProducts, RomStyle::MuxTree] {
        for v in Variant::ALL {
            let opts = Options::new(v).with_rom(rom);
            let d = panda8_with(&program, &opts);
            let mut cpu = GateCpu::new(&d).unwrap();
            let gates = cpu.sim().netlist().gates.len();
            let evals0 = cpu.sim().stats().gate_evals;
            let t = Instant::now();
            for _ in 0..cycles {
                cpu.step();
            }
            let secs = t.elapsed().as_secs_f64();
            let evals = cpu.sim().stats().gate_evals - evals0;
            // Sanity: still in lockstep with the emulator.
            let mut emu = Emulator::new(&program);
            for _ in 0..cycles {
                emu.step();
            }
            assert_eq!(cpu.state().unwrap(), *emu.state());
            println!(
                "{v:?} {rom:?} ({gates} gates, period {}): {:.1} k cycles/s, {:.0} gate evals/cycle ({:.1}% of gates), {:.1} M evals/s",
                cpu.period(),
                cycles as f64 / secs / 1e3,
                evals as f64 / cycles as f64,
                100.0 * evals as f64 / (cycles as f64 * gates as f64),
                evals as f64 / secs / 1e6,
            );
        }
    }
    let t = Instant::now();
    let mut emu = Emulator::new(&program);
    for _ in 0..10_000_000u64 {
        emu.step();
    }
    println!("behavioral emulator: {:.1} M cycles/s", 10.0 / t.elapsed().as_secs_f64());
}
