//! Panda-8 (SPEC §12) in FastRef vs Timing mode: every variant and ROM style, at the STA minimum
//! period, compared on every net at every phase for thousands of cycles; FastRef's `OUT` sequence
//! is also checked against the behavioral emulator.

use std::sync::Arc;

use panda_compile::{FastOptions, FastRef, KernelOptions, compile_kernels};
use panda_cpu8::emu::Emulator;
use panda_cpu8::{Options, RomStyle, Variant, panda8_with};
use panda_netlist::flatten;
use panda_sta::{StaOptions, analyze_with};

use crate::driver::{self, DiffResult, DriverOptions, Plan};

/// Result of [`run_cpu8`].
#[derive(Clone, Debug)]
pub struct Cpu8Report {
    /// Variant.
    pub variant: Variant,
    /// ROM style.
    pub rom: RomStyle,
    /// STA minimum period at the default period.
    pub min_period: u32,
    /// Period used.
    pub period: u32,
    /// STA error messages at the period used (empty = clean).
    pub sta_errors: Vec<String>,
    /// Comparison result.
    pub diff: DiffResult,
    /// `OUT` values strobed in FastRef.
    pub outs: Vec<u8>,
    /// The emulator's `OUT` sequence over the same number of cycles.
    pub golden: Vec<u8>,
}

impl Cpu8Report {
    /// STA clean, FastRef = Timing on every compared phase, and FastRef's outputs match the
    /// emulator.
    pub fn ok(&self) -> bool {
        self.sta_errors.is_empty() && self.diff.equal() && self.outs == self.golden
    }
}

/// Even period ≥ 8 at least `p`.
fn usable_period(p: u32) -> u32 {
    p.max(8).next_multiple_of(2)
}

/// Options of [`run_cpu8`].
#[derive(Clone, Debug)]
pub struct Cpu8Config {
    /// Clock cycles after the reset release.
    pub cycles: u64,
    /// Clock period; `None` = the STA minimum period (rounded up to an even value ≥ 8).
    pub period: Option<u32>,
    /// Run Timing mode with ideal blocks (SPEC §5.2); STA then analyzes for ideal blocks too.
    pub ideal_blocks: bool,
}

impl Default for Cpu8Config {
    fn default() -> Self {
        Cpu8Config {
            cycles: 2000,
            period: None,
            ideal_blocks: false,
        }
    }
}

/// Run `program` on Panda-8 (`variant`, `rom`) in both engines (see [`Cpu8Config`]).
pub fn run_cpu8(
    variant: Variant,
    rom: RomStyle,
    program: &[u8],
    cfg: &Cpu8Config,
) -> Result<Cpu8Report, String> {
    let (cycles, period) = (cfg.cycles, cfg.period);
    let timing = DriverOptions::default()
        .timing
        .with_ideal_blocks(cfg.ideal_blocks);
    let sta = StaOptions {
        ideal_blocks: timing.ideal_blocks,
        glass_box: timing.glass_box.clone(),
        ..StaOptions::default()
    };
    let base = Options::new(variant).with_rom(rom);
    let flat0 = flatten(&panda8_with(program, &base)).map_err(|e| format!("{e:?}"))?;
    let (ks0, t0) = compile_kernels(&flat0, &KernelOptions::default());
    let r0 = analyze_with(&flat0, &ks0, &t0, &sta);
    let min_period = r0.max_min_period().ok_or("no min period")?;
    if min_period == u32::MAX {
        return Err("unbounded min period".into());
    }
    let p = period.unwrap_or_else(|| usable_period(min_period));
    let opts = base.with_period(p);
    let flat = Arc::new(flatten(&panda8_with(program, &opts)).map_err(|e| format!("{e:?}"))?);
    let (ks, tables) = compile_kernels(&flat, &KernelOptions::default());
    let r = analyze_with(&flat, &ks, &tables, &sta);
    let sta_errors = r.errors().map(|d| d.to_string()).collect();
    let fast = FastRef::from_parts(flat.clone(), FastOptions::default(), ks, tables)
        .map_err(|e| format!("{e:?}"))?;
    let reset = u64::from(opts.reset_ticks());
    let plan = Plan {
        input_margin: 1,
        input_settle: 0,
        compare_from: reset,
        inputs: Vec::new(),
        seed: 0,
    };
    // Two events per cycle plus the reset phases.
    let reset_phases = 2 * (reset / u64::from(p) + 2) as usize;
    let dopts = DriverOptions {
        phases: reset_phases + 2 * cycles as usize,
        stop_at_first: true,
        timing,
        ..DriverOptions::default()
    };
    let (period64, phase64) = (u64::from(p), u64::from(p / 2));
    let mut outs = Vec::new();
    let mut counted = 0u64;
    let diff = driver::run_observed(
        &flat,
        fast,
        &plan,
        &dopts,
        &mut |fr: &FastRef, _: &panda_timing::TimingSim, point, next: u64| {
            if point != driver::Point::PhaseEnd {
                return;
            }
            // A cycle sample = the values one tick before a rising edge after the reset release.
            let rising = next >= phase64 && (next - phase64).is_multiple_of(period64);
            if !rising || next <= reset || counted >= cycles {
                return;
            }
            counted += 1;
            let strobe = fr.read_pin("OUT_STROBE").and_then(|v| v.to_u64());
            if strobe == Some(1)
                && let Some(o) = fr.read_pin("OUT").and_then(|v| v.to_u64())
            {
                outs.push(o as u8);
            }
        },
    );
    let mut emu = Emulator::new(program);
    let mut golden = Vec::new();
    for _ in 0..counted {
        if let Some(o) = emu.step().out {
            golden.push(o);
        }
    }
    Ok(Cpu8Report {
        variant,
        rom,
        min_period,
        period: p,
        sta_errors,
        diff,
        outs,
        golden,
    })
}
