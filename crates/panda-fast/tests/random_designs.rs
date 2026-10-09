//! FastSim = FastRef on panda-difftest random designs (registers, latches, gated clocks, async
//! structures, tri-state buses, ring oscillators, …) with random input events (including X/Z
//! input values), X power-on, resets and kernel-extraction variants.
//!
//! Soak: `cargo test --release -p panda-fast --test random_designs -- --ignored --nocapture`

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use common::{Coverage, pair, random_compile, random_lockstep};
use panda_compile::FastOptions;
use panda_difftest::{GenConfig, Rng, generate};
use panda_netlist::flatten;

/// Run one seed; returns the coverage.
fn run_seed(seed: u64, actions: usize) -> Coverage {
    let mut rng = Rng::new(seed ^ 0xFA57_51E0);
    let scale = 2 * (1 + rng.below(3) as u32);
    let g = generate(seed, &GenConfig::default(), scale);
    let flat = Arc::new(flatten(&g.design).expect("flatten"));
    let mut fast = FastOptions::default();
    if rng.chance(30) {
        fast.extract.max_loop = *rng.pick(&[0, 2, 3, 4, 8]);
        fast.extract.max_bits = *rng.pick(&[1, 4, 6, 8, 10]);
    }
    if rng.chance(15) {
        fast.delta_limit = *rng.pick(&[1, 2, 3, 8]);
    }
    let compile = random_compile(&mut rng, &[1, 3, 8, 64, 1 << 20]);
    // A separate stream, so the other choices of a seed do not depend on it.
    let mut crng = Rng::new(seed ^ 0xC4A2_0000);
    if crng.chance(40) {
        common::randomize_charz(&mut crng, &mut fast);
    }
    let (mut fr, mut fs) = pair(&flat, &fast, &compile);
    let mut cov = Coverage::default();
    let ctx = format!("seed {seed} (scale {scale}, {fast:?}, {compile:?})");
    random_lockstep(
        &mut fr, &mut fs, &flat, &mut rng, actions, 15, &ctx, &mut cov,
    );
    cov
}

fn run_range(seeds: std::ops::Range<u64>, actions: usize, threads: usize) -> Coverage {
    let next = AtomicU64::new(seeds.start);
    let total = std::sync::Mutex::new(Coverage::default());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let seed = next.fetch_add(1, Ordering::Relaxed);
                    if seed >= seeds.end {
                        break;
                    }
                    let c = run_seed(seed, actions);
                    total.lock().unwrap().add(&c);
                }
            });
        }
    });
    total.into_inner().unwrap()
}

fn threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

#[test]
fn random_designs_500_seeds() {
    let cov = run_range(0..500, 150, threads());
    eprintln!("random designs: {cov:?}");
    assert!(cov.events > 500 * 100);
    assert!(cov.table_hits > 0 && cov.sim_fallbacks > 0);
    // The specialized paths and LUTs must be exercised.
    assert!(
        cov.fast_opaque > 0 && cov.fast_transparent > 0 && cov.fast_evals > 0 && cov.luts > 0,
        "{cov:?}"
    );
    // The suite must exercise the diagnostic paths.
    for kind in ["DelaySensitive", "Oscillating", "Contention"] {
        assert!(
            cov.diag_kinds.contains_key(kind),
            "no {kind} diagnostic in 500 seeds: {cov:?}"
        );
    }
}

#[test]
#[ignore = "soak: 5000 seeds"]
fn random_designs_soak_5000() {
    let cov = run_range(500..5500, 300, threads());
    eprintln!("random designs soak: {cov:?}");
}

/// Debug helper: `PANDA_SEED=<n> cargo test --release -p panda-fast --test random_designs one_seed -- --ignored --nocapture`.
#[test]
#[ignore = "debug helper"]
fn one_seed() {
    let seed: u64 = std::env::var("PANDA_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let cov = run_seed(seed, 300);
    eprintln!("seed {seed}: {cov:?}");
}
