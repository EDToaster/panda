//! Time the pipeline stages of one seed: `cargo run --release --example stages -- <seed>`.
use std::time::Instant;

use panda_compile::{KernelOptions, compile_kernels};
use panda_difftest::harness::HarnessConfig;

fn main() {
    let seed: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("seed");
    let cfg = HarnessConfig::default();
    let t = Instant::now();
    let p = panda_difftest::harness::prepare(seed, &cfg).unwrap_or_else(|o| panic!("{o:?}"));
    println!(
        "prepare (incl. STA loop): {:?}, scale {}",
        t.elapsed(),
        p.outcome.scale
    );
    let t = Instant::now();
    let (ks, tables) = compile_kernels(&p.flat, &KernelOptions::default());
    println!(
        "compile_kernels: {:?} ({} kernels)",
        t.elapsed(),
        ks.kernels.len()
    );
    let t = Instant::now();
    let ks2 = panda_compile::extract(&p.flat, &Default::default());
    println!("extract: {:?}", t.elapsed());
    for k in &ks2.kernels {
        let t = Instant::now();
        let tab = panda_compile::characterize(&p.flat, k, &Default::default());
        let el = t.elapsed();
        if el.as_millis() > 50 {
            println!(
                "  kernel {} {:?}: {} gates {} inputs {} nets: {:?} table states {:?}",
                k.id,
                k.kind,
                k.gates.len(),
                k.inputs.len(),
                k.nets.len(),
                el,
                tab.as_ref().map(|t| t.states.len())
            );
        }
    }
    let t = Instant::now();
    let _ = panda_sta::analyze(&p.flat, &ks, &tables);
    println!("sta: {:?}", t.elapsed());
    let t = Instant::now();
    let o = panda_difftest::harness::finish(p, &cfg, &mut |_, _, _, _| {});
    println!("compare: {:?} {:?}", t.elapsed(), o.class);
}
