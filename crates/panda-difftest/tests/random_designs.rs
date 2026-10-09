//! SPEC §13 M3 acceptance: on random designs, FastRef = Timing mode at every phase boundary
//! whenever STA is clean; every mismatch is explained by a reported diagnostic.
//!
//! The quick test runs a few hundred seeds; the soak (`--ignored`) runs 5000 more.
//! Reproduce a seed with `cargo run --release -p panda-difftest -- show <seed>`.

use panda_difftest::harness::{Class, HarnessConfig, default_threads, run_seeds};

fn check(range: std::ops::Range<u64>, min_clean: usize, min_flagged_mismatch: usize) {
    let cfg = HarnessConfig::default();
    let (summary, outcomes) = run_seeds(range, &cfg, default_threads());
    println!("{summary}");
    let bugs: Vec<String> = outcomes
        .iter()
        .filter(|o| o.class == Class::CleanMismatch)
        .map(|o| {
            format!(
                "seed {}: {:?}",
                o.seed,
                o.diff
                    .as_ref()
                    .and_then(|d| d.first())
                    .map(ToString::to_string)
            )
        })
        .collect();
    assert!(
        bugs.is_empty(),
        "STA clean but FastRef != Timing:\n{}",
        bugs.join("\n")
    );
    let skipped: Vec<_> = outcomes
        .iter()
        .filter(|o| o.class == Class::Skipped)
        .map(|o| (o.seed, o.skip_reason.clone()))
        .collect();
    assert!(skipped.is_empty(), "skipped designs: {skipped:?}");
    assert!(
        summary.count(Class::CleanEqual) >= min_clean,
        "too few clean designs: {}",
        summary.count(Class::CleanEqual)
    );
    assert!(
        summary.count(Class::FlaggedMismatch) >= min_flagged_mismatch,
        "too few detected (flagged) mismatches — is the comparison working? {}",
        summary.count(Class::FlaggedMismatch)
    );
}

#[test]
fn clean_random_designs_fastref_equals_timing() {
    check(0..300, 40, 40);
}

#[test]
#[ignore = "soak: 5000 designs (a few minutes in release mode)"]
fn soak_5000_random_designs() {
    check(100_000..105_000, 700, 700);
}
