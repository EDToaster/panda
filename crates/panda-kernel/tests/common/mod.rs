//! Shared helpers for the facade tests.
#![allow(dead_code)]

use panda_cpu8::emu::{ArchState, Emulator};
use panda_cpu8::{Options, RomStyle, Variant, panda8_with};
use panda_kernel::{Kernel, KernelOptions};
use panda_netlist::Design;
use panda_timing::TimingOptions;

/// Kernel options with the quiet period-2 threshold used for Panda-8 (see `panda_cpu8::sim`).
pub fn quiet_opts() -> KernelOptions {
    KernelOptions {
        timing: TimingOptions::default().with_p2_limit(panda_cpu8::sim::QUIET_P2_LIMIT),
        ..Default::default()
    }
}

pub fn cpu_design(program: &[u8], variant: Variant, rom: RomStyle) -> (Design, Options) {
    let opts = Options::new(variant).with_rom(rom);
    (panda8_with(program, &opts), opts)
}

/// Tick of cycle 0 (one tick before the first rising edge after reset), as `GateCpu`.
pub fn cycle0_tick(opts: &Options) -> u64 {
    let p = u64::from(opts.period);
    let phase = p / 2;
    let r = u64::from(opts.reset_ticks());
    let first = if r < phase { 0 } else { (r - phase) / p + 1 };
    phase + first * p - 1
}

pub fn pin(k: &Kernel, name: &str) -> u64 {
    k.read_u64(name)
        .unwrap_or_else(|| panic!("pin {name} = {:?} at tick {}", k.read(name), k.tick()))
}

pub fn arch(k: &Kernel) -> ArchState {
    let flags = pin(k, "FLAGS");
    ArchState {
        pc: pin(k, "PC") as u8,
        ir: pin(k, "IR") as u8,
        execute: pin(k, "PHASE") != 0,
        regs: [
            pin(k, "R0") as u8,
            pin(k, "R1") as u8,
            pin(k, "R2") as u8,
            pin(k, "R3") as u8,
        ],
        z: flags & 1 != 0,
        c: flags & 2 != 0,
        out: pin(k, "OUT") as u8,
        out_strobe: pin(k, "OUT_STROBE") != 0,
        halted: pin(k, "HALTED") != 0,
    }
}

/// Emulator trace: states after 0..=n steps.
pub fn trace(program: &[u8], n: usize) -> Vec<ArchState> {
    let mut e = Emulator::new(program);
    let mut v = vec![*e.state()];
    for _ in 0..n {
        e.step();
        v.push(*e.state());
    }
    v
}
