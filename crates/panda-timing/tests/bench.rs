//! Ignored throughput benchmark: `cargo test --release -p panda-timing --test bench -- --ignored --nocapture`.

#[path = "../examples/bench.rs"]
#[allow(dead_code)]
mod bench_main;

#[test]
#[ignore = "benchmark; run in release with --ignored --nocapture"]
fn bench_random_5k() {
    let r = bench_main::run(5_000, 100_000, 1);
    let rate = r.evals as f64 / r.secs;
    println!(
        "random 5k-gate netlist: {} ticks, {} evals, {:.3}s => {:.1} M evals/s ({:.1}% activity)",
        r.ticks,
        r.evals,
        r.secs,
        rate / 1e6,
        100.0 * r.evals as f64 / (r.ticks as f64 * 5_000.0)
    );
}
