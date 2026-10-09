//! The reference clock of a design and clock-cycle stepping of the Timing engine.
//!
//! A *cycle* has the meaning of `panda_kernel::StepUnit::Cycle`: advancing one cycle moves to the
//! tick just before the first rising edge of the reference clock after `tick + 1`, so a whole
//! clock cycle has settled at the sample point. The reference clock is the `Clock` source driving
//! the net given by the user (`--clock`), or else the first running `Clock` source of the
//! netlist. The edge formula is that of `Atom::Clock`: a clock `(period, high, phase)` is 1 at
//! tick `t` iff `(t + period - phase % period) % period < high`.

use std::io::{self, Write};

use panda_netlist::flat::{FlatNetlist, NetIdx, SourceKind};
use panda_timing::TimingSim;

use crate::vcd::{self, VcdWriter};

/// A running `Clock` source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RefClock {
    /// Net driven by the clock.
    pub net: NetIdx,
    /// Canonical name of that net.
    pub name: String,
    /// Period (ticks).
    pub period: u64,
    /// High ticks per period.
    pub high: u64,
    /// Tick of the first rising edge (modulo the period).
    pub phase: u64,
}

impl RefClock {
    /// The reference clock of `nl`: the clock driving net `name` if given (an error if that net is
    /// not driven by a running clock), else the first running clock (`Ok(None)` if there is none).
    pub fn find(nl: &FlatNetlist, name: Option<&str>) -> Result<Option<RefClock>, String> {
        let wanted = match name {
            Some(n) => Some(nl.net_by_name(n).ok_or_else(|| format!("no net named '{n}'"))?),
            None => None,
        };
        let found = nl.sources.iter().find_map(|s| match s.kind {
            SourceKind::Clock { period, high, phase }
                if high > 0 && high < period && wanted.is_none_or(|w| w == s.output) =>
            {
                Some(RefClock {
                    net: s.output,
                    name: nl.net_names[s.output as usize].clone(),
                    period: u64::from(period),
                    high: u64::from(high),
                    phase: u64::from(phase),
                })
            }
            _ => None,
        });
        match (found, name) {
            (None, Some(n)) => Err(format!("'{n}' is not driven by a running Clock")),
            (f, _) => Ok(f),
        }
    }

    /// First rising edge strictly after tick `t`.
    pub fn next_rise_after(&self, t: u64) -> u64 {
        let p = self.period;
        let r = (t % p + p - self.phase % p) % p;
        t + (p - r - 1) % p + 1
    }

    /// The tick a cycle step from tick `t` ends at: one tick before the first rising edge after
    /// `t + 1`.
    pub fn cycle_end_after(&self, t: u64) -> u64 {
        self.next_rise_after(t + 1) - 1
    }

    /// Number of cycle steps from power-on that end at or before tick `t` (the cycle ends are
    /// `cycles_end(0, k)` for `k ≥ 1`).
    pub fn steps_done(&self, t: u64) -> u64 {
        let first = self.cycle_end_after(0);
        if t < first { 0 } else { (t - first) / self.period + 1 }
    }

    /// The tick reached after `n` cycle steps from tick `t`.
    pub fn cycles_end(&self, t: u64, n: u64) -> u64 {
        if n == 0 {
            return t;
        }
        let first = self.cycle_end_after(t);
        first + (n - 1) * self.period
    }
}

/// Advance a Timing engine by `n` clock cycles of `clock` (see the module docs), sampling every
/// tick into `vcd` if given.
pub fn run_cycles<W: Write>(
    sim: &mut TimingSim,
    clock: &RefClock,
    n: u64,
    vcd: Option<&mut VcdWriter<W>>,
) -> io::Result<()> {
    let end = clock.cycles_end(sim.tick(), n);
    vcd::advance(sim, end - sim.tick(), vcd)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clk(period: u64, high: u64, phase: u64) -> RefClock {
        RefClock { net: 0, name: "clk".into(), period, high, phase }
    }

    #[test]
    fn edges() {
        // Rising edges at 10, 30, 50, ...
        let c = clk(20, 10, 10);
        assert_eq!(c.next_rise_after(0), 10);
        assert_eq!(c.next_rise_after(9), 10);
        assert_eq!(c.next_rise_after(10), 30);
        assert_eq!(c.cycle_end_after(0), 9);
        assert_eq!(c.cycle_end_after(8), 9);
        assert_eq!(c.cycle_end_after(9), 29, "from a cycle end, the next cycle end");
        assert_eq!(c.cycles_end(0, 3), 49);
        assert_eq!(c.cycles_end(9, 2), 49);
        assert_eq!(c.cycles_end(5, 0), 5);
        assert_eq!(c.steps_done(8), 0);
        assert_eq!(c.steps_done(9), 1);
        assert_eq!(c.steps_done(48), 2);
        assert_eq!(c.steps_done(49), 3);
        // Phase 0: rising edges at 0, 8, 16, ...
        let c = clk(8, 4, 0);
        assert_eq!(c.next_rise_after(0), 8);
        assert_eq!(c.cycle_end_after(0), 7);
        assert_eq!(c.cycles_end(7, 1), 15);
    }
}
