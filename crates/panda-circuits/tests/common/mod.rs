//! Shared helpers for the behavioral tests.
#![allow(dead_code)]

pub mod random;

use std::sync::Arc;

use panda_circuits::TestbenchOptions;
use panda_netlist::validate::validate;
use panda_netlist::{DefRef, Design, DesignBuilder, Logic, Severity, Value, flatten};
use panda_timing::{TimingOptions, TimingSim};

/// Validate (no errors, no warnings) and flatten `design`, then create a simulator.
pub fn sim_design(design: &Design) -> TimingSim {
    let diags = validate(design);
    let bad: Vec<_> = diags.iter().filter(|d| d.severity != Severity::Info).collect();
    assert!(bad.is_empty(), "validation diagnostics: {bad:#?}");
    let flat = flatten(design).unwrap_or_else(|e| panic!("flatten failed: {e:#?}"));
    TimingSim::new(Arc::new(flat), TimingOptions::default())
}

/// Build the design with `top` as top and simulate it.
pub fn sim_top(db: DesignBuilder, top: &DefRef) -> TimingSim {
    sim_design(&db.build(top))
}

/// Set a top-level input to an integer value.
pub fn set(sim: &mut TimingSim, pin: &str, v: u64) {
    let w = sim.input_value(pin).unwrap_or_else(|| panic!("no input {pin}")).width();
    sim.set_input(pin, &Value::from_u64(w, v)).unwrap();
}

/// Read a signal as an integer; panics (with the raw value) if it is not fully known.
pub fn get(sim: &TimingSim, name: &str) -> u64 {
    let v = sim.read(name).unwrap_or_else(|| panic!("no signal {name}"));
    v.to_u64().unwrap_or_else(|| panic!("{name} = {v} is not fully known at tick {}", sim.tick()))
}

/// Read a 1-bit signal.
pub fn bit(sim: &TimingSim, name: &str) -> Logic {
    let v = sim.read(name).unwrap_or_else(|| panic!("no signal {name}"));
    assert_eq!(v.width(), 1, "{name} is not 1 bit");
    v.get(0)
}

/// Settle a combinational circuit (panics if it does not settle).
pub fn settle(sim: &mut TimingSim) {
    assert!(sim.settle(10_000), "did not settle");
}

/// Run until tick `t` (must not be in the past).
pub fn run_to(sim: &mut TimingSim, t: u64) {
    assert!(t >= sim.tick(), "run_to({t}) but already at {}", sim.tick());
    sim.run(t - sim.tick());
}

/// Synchronous test driver: for clock cycle `k` (rising edge index), apply `inputs` a quarter
/// period before the edge (during the low phase), then run until just before the next input
/// change point (3/4 period after the edge), where the post-edge state is checked.
pub fn cycle(sim: &mut TimingSim, opts: &TestbenchOptions, k: u64, inputs: &[(&str, u64)]) {
    let q = u64::from(opts.period) / 4;
    run_to(sim, opts.rising_edge(k) - q);
    for &(pin, v) in inputs {
        set(sim, pin, v);
    }
    run_to(sim, opts.rising_edge(k + 1) - q - 1);
}

/// Assert that no diagnostics other than the allowed kinds were raised.
pub fn assert_clean(sim: &mut TimingSim) {
    let d = sim.take_diagnostics();
    assert!(d.is_empty(), "unexpected diagnostics: {d:#?}");
}

/// Small deterministic RNG (xorshift64*).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform-ish in `0..n`.
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}
