//! Timing mode — the reference semantics (SPEC §5).
//!
//! * Discrete ticks. Every [`FlatGate`](panda_netlist::flat::FlatGate) has delay 1; sources and
//!   resolvers have delay 0.
//! * Synchronous update: `V(t+1)[gate.out] = eval_gate(gate, V(t)[gate.inputs])`. Then sources take
//!   their tick-(t+1) values, then resolvers recompute from their (tick t+1) driver nets.
//! * Event-driven: only gates whose inputs changed at tick t are evaluated for t+1.
//! * Gate semantics come exclusively from [`panda_netlist::eval`].
//! * Reset (tick 0): every gate output = X; sources = their tick-0 values (top inputs default to 0
//!   unless set); resolvers computed; keeper state X.
//! * Period-2 collapse (§5.3): if, with no source change during the window, a net changed value on
//!   each of the last `2 * p2_limit` consecutive ticks, it is forced to X (and a `Metastable`
//!   diagnostic is raised once per episode). X then propagates normally.
//! * Oscillation (§5.3): if gate events continue for `osc_limit` consecutive ticks with no source
//!   change, an `Oscillating` diagnostic is raised (once per episode) naming some oscillating nets.
//!   The simulation keeps running faithfully.
//! * Contention on a resolver raises a `Contention` diagnostic (once per net per episode).
//! * Input changes via [`TimingSim::set_input`] are applied as source changes at the next tick.

use std::sync::Arc;

use panda_netlist::flat::{FlatNetlist, NetIdx};
use panda_netlist::{Diagnostic, Logic, Value};

#[derive(Clone, Debug)]
pub struct TimingOptions {
    /// Period-2 collapse threshold (cycles). Default 4.
    pub p2_limit: u32,
    /// Ticks of continuous activity without source change before `Oscillating`. Default 10_000.
    pub osc_limit: u64,
}

impl Default for TimingOptions {
    fn default() -> Self {
        TimingOptions { p2_limit: 4, osc_limit: 10_000 }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TimingStats {
    pub ticks: u64,
    pub gate_evals: u64,
}

pub struct TimingSim {
    // implementation-defined
}

impl TimingSim {
    pub fn new(netlist: Arc<FlatNetlist>, opts: TimingOptions) -> Self {
        todo!()
    }
    pub fn netlist(&self) -> &Arc<FlatNetlist> {
        todo!()
    }
    /// Back to tick 0 (power-on: all gate outputs X). Input pin values set so far are kept.
    pub fn reset(&mut self) {
        todo!()
    }
    /// Set a top-level input pin (by name). Width must match. Takes effect at the next tick.
    pub fn set_input(&mut self, pin: &str, value: &Value) -> Result<(), String> {
        todo!()
    }
    /// Force a net's current value (testing / power-on randomization). Schedules its fanout.
    pub fn force_net(&mut self, net: NetIdx, value: Logic) {
        todo!()
    }
    /// Advance one tick.
    pub fn step(&mut self) {
        todo!()
    }
    /// Advance `ticks` ticks.
    pub fn run(&mut self, ticks: u64) {
        todo!()
    }
    /// Run until no gate events are pending and no source changes at the next tick
    /// (clocks make this never true), or `max_ticks` elapse. Returns true if quiescent.
    pub fn settle(&mut self, max_ticks: u64) -> bool {
        todo!()
    }
    /// Current tick.
    pub fn tick(&self) -> u64 {
        todo!()
    }
    pub fn get(&self, net: NetIdx) -> Logic {
        todo!()
    }
    /// All net values (index = NetIdx).
    pub fn values(&self) -> &[Logic] {
        todo!()
    }
    /// Nets whose value changed during the most recent `step` (for VCD dumping).
    pub fn last_changed(&self) -> &[NetIdx] {
        todo!()
    }
    /// Read a top-level pin (input or output) by name.
    pub fn read_pin(&self, pin: &str) -> Option<Value> {
        todo!()
    }
    /// Read by hierarchical name: an exact flat net name (`"cpu/alu/c[3]"`, `"q"`), or a multi-bit
    /// net base name (`"cpu/alu/c"` gathers `c[0]..`), or a top-level pin name.
    pub fn read(&self, name: &str) -> Option<Value> {
        todo!()
    }
    pub fn take_diagnostics(&mut self) -> Vec<Diagnostic> {
        todo!()
    }
    pub fn stats(&self) -> TimingStats {
        todo!()
    }
}
