//! Run a generated Panda-8 design in Timing mode ([`panda_timing`]) at clock-cycle granularity.
//!
//! [`GateCpu`] reads the `Clock` / `PowerOnReset` parameters from the flattened netlist, runs the
//! timing engine to one tick before each rising edge (when the previous cycle has settled) and
//! samples the debug pins into an [`ArchState`]. Cycle `k` of a [`GateCpu`] corresponds to the
//! state of an [`Emulator`](crate::emu::Emulator) after `k` steps.

use std::sync::Arc;

use panda_netlist::flat::{FlatNetlist, SourceKind};
use panda_netlist::{Design, Diagnostic, flatten};
use panda_timing::{TimingOptions, TimingSim};

use crate::emu::ArchState;

/// A period-2 collapse threshold (in cycles, i.e. `2 × 32 = 64` consecutive changes) above the
/// longest combinational path of every Panda-8 variant (≤ 47 ticks): acyclic logic cannot
/// alternate that long, so only genuine latch-loop metastability is collapsed.
pub const QUIET_P2_LIMIT: u32 = 32;

/// A Panda-8 design running in the timing engine.
pub struct GateCpu {
    sim: TimingSim,
    period: u64,
    phase: u64,
    /// Index of the next rising edge (absolute edge number since tick 0).
    next_edge: u64,
    /// Edges since reset release.
    cycles: u64,
}

/// Settle times observed by [`GateCpu::measure_settle`], in ticks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SettleTimes {
    /// Longest time from a rising clock edge to the last net change before the falling edge.
    pub after_rise: u64,
    /// Longest time from a falling clock edge to the last net change before the next rising edge.
    pub after_fall: u64,
}

impl GateCpu {
    /// Flatten `design` and power it on with the engine's default [`TimingOptions`]; runs through
    /// reset, stopping one tick before the first rising edge after reset (the reset state,
    /// cycle 0).
    ///
    /// Note: with the default period-2 collapse threshold (`p2_limit = 4`) the engine reports
    /// occasional `Metastable` warnings on *combinational* nets of the ALU: glitch trains caused by
    /// staggered input arrival that alternate for 8+ consecutive ticks. The collapsed nets recover
    /// one tick later and the settled state is unaffected. Use [`GateCpu::with_options`] with
    /// [`QUIET_P2_LIMIT`] to suppress them.
    pub fn new(design: &Design) -> Result<Self, String> {
        Self::with_options(design, TimingOptions::default())
    }

    /// Like [`GateCpu::new`] with explicit timing-engine options.
    pub fn with_options(design: &Design, opts: TimingOptions) -> Result<Self, String> {
        let flat = flatten(design).map_err(|e| {
            let msgs: Vec<String> = e.iter().map(Diagnostic::to_string).collect();
            format!("flatten failed: {}", msgs.join("; "))
        })?;
        Self::from_netlist(Arc::new(flat), opts)
    }

    /// Like [`GateCpu::with_options`] for an already flattened netlist.
    pub fn from_netlist(netlist: Arc<FlatNetlist>, opts: TimingOptions) -> Result<Self, String> {
        let mut clock = None;
        let mut reset = None;
        for s in &netlist.sources {
            match s.kind {
                SourceKind::Clock { .. } if clock.is_some() => return Err("design has more than one Clock".into()),
                SourceKind::Clock { period, high, phase } => clock = Some((period, high, phase)),
                SourceKind::PowerOnReset { .. } if reset.is_some() => {
                    return Err("design has more than one PowerOnReset".into());
                }
                SourceKind::PowerOnReset { ticks } => reset = Some(ticks),
                _ => {}
            }
        }
        let (period, _high, phase) = clock.ok_or("design has no Clock")?;
        let reset_ticks = u64::from(reset.ok_or("design has no PowerOnReset")?);
        let (period, phase) = (u64::from(period), u64::from(phase % period));
        // First rising edge strictly after the reset release tick.
        let first = if reset_ticks < phase { 0 } else { (reset_ticks - phase) / period + 1 };
        let sim = TimingSim::new(netlist, opts);
        let mut cpu = GateCpu { sim, period, phase, next_edge: first, cycles: 0 };
        let target = cpu.edge_tick(first) - 1;
        cpu.sim.run(target);
        Ok(cpu)
    }

    fn edge_tick(&self, k: u64) -> u64 {
        self.phase + k * self.period
    }

    /// Clock period in ticks.
    pub fn period(&self) -> u64 {
        self.period
    }

    /// Clock edges executed since reset.
    pub fn cycles(&self) -> u64 {
        self.cycles
    }

    /// The underlying timing simulator.
    pub fn sim(&self) -> &TimingSim {
        &self.sim
    }

    /// Mutable access to the underlying timing simulator.
    pub fn sim_mut(&mut self) -> &mut TimingSim {
        &mut self.sim
    }

    /// Advance one clock cycle (through the next rising edge, to one tick before the following one).
    pub fn step(&mut self) {
        self.sim.run(self.period);
        self.next_edge += 1;
        self.cycles += 1;
    }

    /// Read an 8-bit-or-narrower output pin as a number; error if it contains X/Z.
    pub fn pin(&self, name: &str) -> Result<u64, String> {
        let v = self.sim.read_pin(name).ok_or_else(|| format!("no pin '{name}'"))?;
        v.to_u64().ok_or_else(|| format!("pin {name} = {v:?} is not fully known at cycle {}", self.cycles))
    }

    /// Sample the architectural state from the debug pins.
    pub fn state(&self) -> Result<ArchState, String> {
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

    /// Run until `HALTED` (or `max_cycles`), collecting the `OUT` value at every cycle in which
    /// `OUT_STROBE` is high.
    pub fn run_outputs(&mut self, max_cycles: u64) -> Result<Vec<u8>, String> {
        let mut outs = Vec::new();
        for _ in 0..max_cycles {
            self.step();
            if self.pin("OUT_STROBE")? != 0 {
                outs.push(self.pin("OUT")? as u8);
            }
            if self.pin("HALTED")? != 0 {
                break;
            }
        }
        Ok(outs)
    }

    /// Run `cycles` clock cycles tick by tick and report the longest settle times after rising
    /// and falling edges. The clock period must leave room for them: a design is timing-clean when
    /// `after_rise < period` and `after_fall < period / 2` (with margin).
    pub fn measure_settle(&mut self, cycles: u64) -> SettleTimes {
        let mut st = SettleTimes::default();
        let half = self.period / 2;
        for _ in 0..cycles {
            let rise = self.edge_tick(self.next_edge);
            let fall = rise + half;
            let next = rise + self.period;
            // Step to the rising edge, then observe each half period.
            while self.sim.tick() < rise {
                self.sim.step();
            }
            let mut last_rise = 0;
            let mut last_fall = 0;
            while self.sim.tick() < next - 1 {
                self.sim.step();
                let t = self.sim.tick();
                if t != fall && !self.sim.last_changed().is_empty() {
                    if t < fall {
                        last_rise = t - rise;
                    } else {
                        last_fall = t - fall;
                    }
                }
            }
            st.after_rise = st.after_rise.max(last_rise);
            st.after_fall = st.after_fall.max(last_fall);
            self.next_edge += 1;
            self.cycles += 1;
        }
        st
    }
}
