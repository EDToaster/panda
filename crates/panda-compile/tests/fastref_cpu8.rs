//! Panda-8 in FastRef vs. the behavioral emulator (SPEC §6, §12).
//!
//! Benchmark: `cargo test --release -p panda-compile --test fastref_cpu8 -- --ignored --nocapture`

use std::sync::Arc;
use std::time::Instant;

use panda_compile::{FastOptions, FastRef};
use panda_cpu8::emu::{ArchState, Emulator};
use panda_cpu8::programs::{self, Program};
use panda_cpu8::{Options, RomStyle, Variant, panda8_with};
use panda_netlist::flat::{FlatNetlist, SourceKind};
use panda_netlist::flatten;

/// A Panda-8 design running in FastRef at clock-cycle granularity (like `panda_cpu8::sim::GateCpu`):
/// cycle `k` = values one tick before rising edge `first + k`.
struct FastCpu {
    fr: FastRef,
    period: u64,
    phase: u64,
    next_edge: u64,
    cycles: u64,
}

impl FastCpu {
    fn new(flat: Arc<FlatNetlist>) -> FastCpu {
        let mut clock = None;
        let mut reset = None;
        for s in &flat.sources {
            match s.kind {
                SourceKind::Clock { period, phase, .. } => clock = Some((period, phase)),
                SourceKind::PowerOnReset { ticks } => reset = Some(ticks),
                _ => {}
            }
        }
        let (period, phase) = clock.expect("clock");
        let (period, phase) = (u64::from(period), u64::from(phase % period));
        let reset = u64::from(reset.expect("reset"));
        let first = if reset < phase {
            0
        } else {
            (reset - phase) / period + 1
        };
        let fr = FastRef::new(flat, FastOptions::default()).expect("FastRef::new");
        let mut cpu = FastCpu {
            fr,
            period,
            phase,
            next_edge: first,
            cycles: 0,
        };
        cpu.fr.run_until(cpu.edge_tick(first) - 1);
        cpu
    }

    fn edge_tick(&self, k: u64) -> u64 {
        self.phase + k * self.period
    }

    fn step(&mut self) {
        self.next_edge += 1;
        self.cycles += 1;
        self.fr.run_until(self.edge_tick(self.next_edge) - 1);
    }

    fn pin(&self, name: &str) -> Result<u64, String> {
        let v = self
            .fr
            .read_pin(name)
            .ok_or_else(|| format!("no pin {name}"))?;
        v.to_u64()
            .ok_or_else(|| format!("pin {name} = {v:?} not known at cycle {}", self.cycles))
    }

    fn state(&self) -> Result<ArchState, String> {
        let byte = |n: &str| self.pin(n).map(|v| v as u8);
        let bit = |n: &str| self.pin(n).map(|v| v != 0);
        let flags = self.pin("FLAGS")?;
        Ok(ArchState {
            pc: byte("PC")?,
            ir: byte("IR")?,
            execute: bit("PHASE")?,
            regs: [byte("R0")?, byte("R1")?, byte("R2")?, byte("R3")?],
            z: flags & 1 != 0,
            c: flags & 2 != 0,
            out: byte("OUT")?,
            out_strobe: bit("OUT_STROBE")?,
            halted: bit("HALTED")?,
        })
    }
}

fn flat_for(p: &Program, opts: &Options) -> Arc<FlatNetlist> {
    Arc::new(flatten(&panda8_with(&p.image(), opts)).expect("flatten"))
}

/// Run `p` in lockstep with the emulator (full state every cycle) until it halts (+5 cycles) or
/// `max` cycles; check the OUT sequence. No diagnostic at all is allowed. Returns the cycles run.
fn lockstep(p: &Program, opts: &Options, max: u64) -> u64 {
    let flat = flat_for(p, opts);
    let mut cpu = FastCpu::new(flat);
    let reset_diags = cpu.fr.take_diagnostics();
    assert!(
        reset_diags.is_empty(),
        "power-on diagnostics: {reset_diags:#?}"
    );
    let ctx = |c: u64| format!("{} {opts:?} cycle {c}", p.name);
    assert_eq!(
        cpu.state().unwrap_or_else(|e| panic!("{e} ({})", ctx(0))),
        ArchState::reset(),
        "reset state ({})",
        ctx(0)
    );
    let image = p.image();
    let mut emu = Emulator::new(&image);
    let mut emu_outs = Vec::new();
    let mut fast_outs = Vec::new();
    let mut after_halt = 0;
    while emu.cycles() < max {
        let info = emu.step();
        cpu.step();
        emu_outs.extend(info.out);
        let got = cpu
            .state()
            .unwrap_or_else(|e| panic!("{e} ({})", ctx(emu.cycles())));
        assert_eq!(got, *emu.state(), "state mismatch ({})", ctx(emu.cycles()));
        if got.out_strobe {
            fast_outs.push(got.out);
        }
        let diags = cpu.fr.take_diagnostics();
        assert!(
            diags.is_empty(),
            "diagnostics ({}): {diags:#?}",
            ctx(emu.cycles())
        );
        if got.halted {
            after_halt += 1;
            if after_halt > 5 {
                break;
            }
        }
    }
    assert_eq!(fast_outs, emu_outs, "OUT sequence ({})", ctx(emu.cycles()));
    assert_eq!(emu.state().halted, p.halts, "{}", p.name);
    let mut want = Emulator::new(&image);
    assert_eq!(emu_outs, want.run(max), "emulator OUT sequence");
    emu.cycles()
}

fn check_all_variants(rom: RomStyle) {
    for v in Variant::ALL {
        let opts = Options::new(v).with_rom(rom);
        lockstep(&programs::FIBONACCI, &opts, 5_000);
        lockstep(&programs::MULTIPLY, &opts, 5_000);
    }
}

#[test]
fn panda8_sum_of_products_all_variants() {
    check_all_variants(RomStyle::SumOfProducts);
}

#[test]
fn panda8_mux_tree_all_variants() {
    check_all_variants(RomStyle::MuxTree);
}

#[test]
fn panda8_countdown_and_bench_loop() {
    let opts = Options::new(Variant::Hierarchical);
    lockstep(&programs::COUNTDOWN, &opts, 5_000);
    // Three outer iterations of the benchmark loop (≈ 3100 cycles).
    let cycles = lockstep(&programs::BENCH_LOOP, &Options::new(Variant::Flat), 3_200);
    assert_eq!(cycles, 3_200);
}

#[test]
fn panda8_uses_tables_after_reset() {
    let flat = flat_for(&programs::FIBONACCI, &Options::new(Variant::Hierarchical));
    let mut cpu = FastCpu::new(flat);
    let before = cpu.fr.stats();
    for _ in 0..200 {
        cpu.step();
    }
    let after = cpu.fr.stats();
    let hits = after.table_hits - before.table_hits;
    let sims = after.sim_fallbacks - before.sim_fallbacks;
    assert!(
        sims * 20 <= hits,
        "steady state should be table-driven: {hits} hits, {sims} sim fallbacks"
    );
}

#[test]
#[ignore = "benchmark; run in release with --ignored --nocapture"]
fn fastref_cycles_per_second() {
    let p = programs::BENCH_LOOP;
    let image = p.image();
    let cycles = 20_000u64;
    for rom in [RomStyle::SumOfProducts, RomStyle::MuxTree] {
        for v in Variant::ALL {
            let opts = Options::new(v).with_rom(rom);
            let flat = flat_for(&p, &opts);
            let gates = flat.gates.len();
            let t0 = Instant::now();
            let mut cpu = FastCpu::new(flat);
            let compile = t0.elapsed();
            let s0 = cpu.fr.stats();
            let t = Instant::now();
            for _ in 0..cycles {
                cpu.step();
            }
            let secs = t.elapsed().as_secs_f64();
            let s1 = cpu.fr.stats();
            let mut emu = Emulator::new(&image);
            for _ in 0..cycles {
                emu.step();
            }
            assert_eq!(cpu.state().unwrap(), *emu.state());
            let per = |a: u64, b: u64| (a - b) as f64 / cycles as f64;
            println!(
                "FastRef {v:?} {rom:?} ({gates} gates, {} kernels): {:.1} k cycles/s; compile+reset {compile:?}; \
                 per cycle: {:.1} events, {:.1} rounds, {:.1} transitions ({:.1} table, {:.2} sim), {:.0} comb evals",
                cpu.fr.kernels().kernels.len(),
                cycles as f64 / secs / 1e3,
                per(s1.events, s0.events),
                per(s1.rounds, s0.rounds),
                per(s1.transitions, s0.transitions),
                per(s1.table_hits, s0.table_hits),
                per(s1.sim_fallbacks, s0.sim_fallbacks),
                per(s1.comb_evals, s0.comb_evals),
            );
        }
    }
}
