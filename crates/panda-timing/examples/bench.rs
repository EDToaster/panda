//! Throughput benchmark of the timing engine on a random 5k-gate netlist.
//!
//! `cargo run --release -p panda-timing --example bench [gates] [ticks] [seed]`

#[path = "../tests/common/mod.rs"]
mod common;

use std::time::Instant;

use common::{random_design, sim, RandomSpec, Rng};
use panda_netlist::Value;

fn main() {
    let args: Vec<u64> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let gates = args.first().copied().unwrap_or(5_000) as usize;
    let ticks = args.get(1).copied().unwrap_or(200_000);
    let seed = args.get(2).copied().unwrap_or(1);
    let r = run(gates, ticks, seed);
    println!(
        "{gates} gates, {} ticks: {} evals in {:.3}s = {:.1} M evals/s ({:.1}% activity, {:.0} k ticks/s, {} diagnostics)",
        r.ticks,
        r.evals,
        r.secs,
        r.evals as f64 / r.secs / 1e6,
        100.0 * r.evals as f64 / (r.ticks as f64 * gates as f64),
        r.ticks as f64 / r.secs / 1e3,
        r.diags,
    );
}

/// Result of one benchmark run.
pub struct BenchResult {
    pub ticks: u64,
    pub evals: u64,
    pub secs: f64,
    pub diags: usize,
}

/// Simulate a random netlist (clocked, some feedback, inputs toggled) for `ticks` ticks.
pub fn run(gates: usize, ticks: u64, seed: u64) -> BenchResult {
    let mut rng = Rng::new(seed);
    let spec = RandomSpec { inputs: 16, gates, feedback: 0.01, buses: 8, clocks: 4, locality: 64 };
    let mut s = sim(random_design(&mut rng, &spec), vec![]);
    let names: Vec<String> = s.netlist().top_inputs.iter().map(|p| p.name.clone()).collect();
    let start = Instant::now();
    for t in 0..ticks {
        if t % 7 == 0 {
            let pin = &names[rng.below(names.len())];
            s.set_input(pin, &Value::from_bool(rng.chance(0.5))).unwrap();
        }
        s.step();
    }
    let secs = start.elapsed().as_secs_f64();
    BenchResult { ticks, evals: s.stats().gate_evals, secs, diags: s.take_diagnostics().len() }
}
