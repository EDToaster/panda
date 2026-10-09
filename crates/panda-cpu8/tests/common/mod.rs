//! Shared helpers: run a program on the emulator and on the gate-level design in lockstep.
#![allow(dead_code)]

use panda_cpu8::emu::{ArchState, Emulator};
use panda_cpu8::sim::{GateCpu, QUIET_P2_LIMIT};
use panda_cpu8::{Options, panda8_with};
use panda_netlist::DiagKind;
use panda_timing::TimingOptions;

/// Result of a lockstep run.
pub struct Lockstep {
    /// OUT values seen by the emulator.
    pub emu_outs: Vec<u8>,
    /// OUT values sampled from the gate-level design (OUT while OUT_STROBE is high).
    pub gate_outs: Vec<u8>,
    /// Cycles executed.
    pub cycles: u64,
    /// Whether the program halted.
    pub halted: bool,
    /// `Metastable` diagnostics raised after reset (only possible with the default engine options).
    pub metastable: usize,
}

/// Timing-engine settings for a lockstep run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    /// Engine defaults (`p2_limit = 4`): combinational glitch trains may raise `Metastable`
    /// warnings (counted); any other diagnostic fails the run.
    Default,
    /// `p2_limit = QUIET_P2_LIMIT`: no diagnostic at all is allowed.
    Quiet,
}

/// [`lockstep_with`] using [`Engine::Quiet`].
pub fn lockstep(program: &[u8], opts: &Options, max_cycles: u64, extra: u64) -> Lockstep {
    lockstep_with(program, opts, max_cycles, extra, Engine::Quiet)
}

/// Run `program` for up to `max_cycles` (stopping `extra` cycles after HALT), comparing the full
/// architectural state every cycle. Panics with context on the first mismatch, on X, and on any
/// timing diagnostic raised after reset that `engine` does not allow.
pub fn lockstep_with(program: &[u8], opts: &Options, max_cycles: u64, extra: u64, engine: Engine) -> Lockstep {
    let design = panda8_with(program, opts);
    let topts = match engine {
        Engine::Default => TimingOptions::default(),
        Engine::Quiet => TimingOptions::default().with_p2_limit(QUIET_P2_LIMIT),
    };
    let mut gate = GateCpu::with_options(&design, topts).expect("design flattens");
    let mut metastable = 0;
    // Power-on diagnostics (before reset completes) are expected to be harmless; drop them.
    gate.sim_mut().take_diagnostics();
    let mut emu = Emulator::new(program);
    let ctx = |emu: &Emulator| format!("{opts:?}, cycle {}", emu.cycles());
    assert_eq!(gate.state().unwrap_or_else(|e| panic!("{e} ({})", ctx(&emu))), ArchState::reset(), "reset state");
    let mut emu_outs = Vec::new();
    let mut gate_outs = Vec::new();
    let mut after_halt = 0;
    while emu.cycles() < max_cycles {
        let info = emu.step();
        gate.step();
        emu_outs.extend(info.out);
        let g = gate.state().unwrap_or_else(|e| panic!("{e} ({})", ctx(&emu)));
        let want = *emu.state();
        assert_eq!(g, want, "state mismatch ({}), last instruction {:?}", ctx(&emu), info.executed);
        if g.out_strobe {
            gate_outs.push(g.out);
        }
        let mut diags = gate.sim_mut().take_diagnostics();
        if engine == Engine::Default {
            let before = diags.len();
            diags.retain(|d| d.kind != DiagKind::Metastable);
            metastable += before - diags.len();
        }
        assert!(diags.is_empty(), "timing diagnostics ({}): {diags:#?}", ctx(&emu));
        if want.halted {
            after_halt += 1;
            if after_halt > extra {
                break;
            }
        }
    }
    Lockstep { emu_outs, gate_outs, cycles: emu.cycles(), halted: emu.state().halted, metastable }
}

/// Tiny deterministic PRNG (xorshift64*), so tests need no extra dependency.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}
