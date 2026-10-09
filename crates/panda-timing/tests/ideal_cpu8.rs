//! Panda-8 in Timing mode with ideal blocks (SPEC §5.2) against the behavioral emulator, and an
//! ignored throughput comparison:
//! `cargo test --release -p panda-timing --test ideal_cpu8 -- --ignored --nocapture`.

use std::sync::Arc;
use std::time::Instant;

use panda_cpu8::emu::Emulator;
use panda_cpu8::programs::{self, Program};
use panda_cpu8::sim::{GateCpu, QUIET_P2_LIMIT};
use panda_cpu8::{Options, RomStyle, Variant, panda8_with};
use panda_netlist::{DiagKind, flatten};
use panda_timing::TimingOptions;

fn cpu(program: &[u8], opts: &Options, ideal: bool) -> GateCpu {
    let flat = Arc::new(flatten(&panda8_with(program, opts)).expect("flatten"));
    let topts = TimingOptions::default().with_p2_limit(QUIET_P2_LIMIT).with_ideal_blocks(ideal);
    GateCpu::from_netlist(flat, topts).expect("GateCpu")
}

/// Cycle-by-cycle architectural state and the OUT sequence must match the emulator.
fn lockstep(p: &Program, opts: &Options, max_cycles: u64) -> usize {
    let image = p.image();
    let mut g = cpu(&image, opts, true);
    let blocks = g.sim().ideal_blocks().len();
    let mut emu = Emulator::new(&image);
    assert_eq!(g.state().unwrap(), *emu.state(), "{}: reset state", p.name);
    let mut gate_outs = Vec::new();
    let mut emu_outs = Vec::new();
    for c in 0..max_cycles {
        g.step();
        let info = emu.step();
        emu_outs.extend(info.out);
        let st = g.state().unwrap_or_else(|e| panic!("{} {opts:?}: cycle {c}: {e}", p.name));
        assert_eq!(st, *emu.state(), "{} {opts:?}: cycle {c}", p.name);
        if st.out_strobe {
            gate_outs.push(st.out);
        }
        if st.halted {
            break;
        }
    }
    assert_eq!(gate_outs, emu_outs, "{}: OUT sequence", p.name);
    let diags = g.sim_mut().take_diagnostics();
    assert!(!diags.iter().any(|d| matches!(d.kind, DiagKind::StuckX | DiagKind::Contention)), "{diags:?}");
    blocks
}

#[test]
fn panda8_ideal_blocks_match_emulator() {
    for rom in [RomStyle::SumOfProducts, RomStyle::MuxTree] {
        for v in [Variant::Hierarchical, Variant::GatedClockRegfile] {
            let opts = Options::new(v).with_rom(rom);
            for p in [programs::FIBONACCI, programs::MULTIPLY] {
                let blocks = lockstep(&p, &opts, 3_000);
                assert!(blocks > 10, "{v:?} {rom:?}: only {blocks} ideal blocks");
            }
            lockstep(&programs::BENCH_LOOP, &opts, 1_100);
        }
    }
}

#[test]
fn panda8_flat_variant_has_no_blocks_and_still_runs() {
    let opts = Options::new(Variant::Flat);
    assert_eq!(lockstep(&programs::COUNTDOWN, &opts, 2_000), 0);
}

#[test]
fn panda8_block_selection() {
    let image = programs::FIBONACCI.image();
    for rom in [RomStyle::SumOfProducts, RomStyle::MuxTree] {
        let g = cpu(&image, &Options::new(Variant::Hierarchical).with_rom(rom), true);
        let blocks = g.sim().ideal_blocks();
        let paths: Vec<&str> = blocks.iter().map(|b| b.path.as_str()).collect();
        // The ROM, ALU, control, PC incrementer and the operand / next-PC muxes are whole blocks;
        // sequential instances (core, register file, registers) are not.
        for want in ["rom", "cpu/ctl", "cpu/alu", "cpu/pcinc", "cpu/bsel", "cpu/pcsel"] {
            assert!(paths.contains(&want), "{rom:?}: {want} missing from {paths:?}");
        }
        assert!(!paths.iter().any(|p| *p == "cpu" || *p == "cpu/rf" || *p == "cpu/pc"));
        let max = blocks.iter().map(|b| b.max_latency()).max().unwrap();
        let rom_lat = blocks.iter().find(|b| b.path == "rom").unwrap().max_latency();
        println!("{rom:?}: {} blocks, max latency {max}, rom latency {rom_lat}", blocks.len());
        assert!(u64::from(max) < g.period() / 2);
    }
}

#[test]
#[ignore = "benchmark; run in release with --ignored --nocapture"]
fn bench_panda8_ideal_vs_gates() {
    let program = programs::BENCH_LOOP.image();
    let cycles = 20_000u64;
    for rom in [RomStyle::SumOfProducts, RomStyle::MuxTree] {
        for v in [Variant::Hierarchical, Variant::GatedClockRegfile] {
            let opts = Options::new(v).with_rom(rom);
            let mut rates = Vec::new();
            for ideal in [false, true] {
                let mut g = cpu(&program, &opts, ideal);
                let t = Instant::now();
                for _ in 0..cycles {
                    g.step();
                }
                let secs = t.elapsed().as_secs_f64();
                let mut emu = Emulator::new(&program);
                for _ in 0..cycles {
                    emu.step();
                }
                assert_eq!(g.state().unwrap(), *emu.state());
                let st = g.sim().stats();
                println!(
                    "{v:?} {rom:?} ideal={ideal}: {:.1} k cycles/s ({} blocks; {:.0} gate evals + {:.0} block evals per cycle)",
                    cycles as f64 / secs / 1e3,
                    g.sim().ideal_blocks().len(),
                    st.gate_evals as f64 / cycles as f64,
                    st.block_evals as f64 / cycles as f64,
                );
                rates.push(cycles as f64 / secs);
            }
            println!("  speedup {:.2}x", rates[1] / rates[0]);
        }
    }
}
