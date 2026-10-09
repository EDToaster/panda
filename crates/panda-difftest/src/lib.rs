//! panda-difftest — the M3 acceptance harness (SPEC §13 M3): differential fuzzing of FastRef
//! (SPEC §6.2) against Timing mode (SPEC §5) on random designs, gated by STA (SPEC §9).
//!
//! Acceptance criterion: FastRef = Timing at phase boundaries on STA-clean designs; every
//! mismatch is explained by a reported diagnostic.
//!
//! * [`generator`] — random design generator: registers and latches from `panda-circuits` and
//!   custom gate-level latches (mux, long-loop, keeper), two-phase latch pipelines, random logic
//!   (gates, muxes, adders, decoders, comparators, tri-state buses), feedback, inputs, one or two
//!   clocks, synchronous POR reset; asynchronous structures and clockless designs at lower weight.
//! * [`driver`] — runs both engines side by side and compares every net at every phase boundary
//!   (and just before every input change), with random input changes placed safely inside phases.
//! * [`harness`] — per-seed pipeline (clock periods from STA `min_period`, phases long enough for
//!   input events), ideal-block and kernel-extraction variants, and the classification into
//!   clean-and-equal / clean-but-mismatch (a bug) / flagged-and-mismatch / flagged-and-equal (STA
//!   pessimism).
//! * [`minimize`] — delta debugging of failing designs, a mutation-based hunt for clean variants of
//!   flagged designs, and a readable netlist dump.
//! * [`cpu8`] — Panda-8 (every variant and ROM style) in FastRef vs Timing mode.
//!
//! The `difftest` binary runs seed ranges and debugging aids:
//!
//! ```text
//! cargo run --release -p panda-difftest -- <start> <count> [threads]   # fuzz + summary
//! cargo run --release -p panda-difftest -- show <seed>                 # one design, STA report
//! cargo run --release -p panda-difftest -- trace <seed> <net,net,..> [from] [phases]
//! cargo run --release -p panda-difftest -- min <seed> [any]            # delta debugging
//! cargo run --release -p panda-difftest -- hunt <start> <count> <tries>
//! cargo run --release -p panda-difftest -- cpu8 [cycles] [program]
//! ```

pub mod cpu8;
pub mod driver;
pub mod generator;
pub mod harness;
pub mod minimize;
pub mod rng;

pub use driver::{DiffResult, DriverOptions, Mismatch, Plan, Point};
pub use generator::{GenConfig, Generated, generate};
pub use harness::{Class, HarnessConfig, Outcome, Summary, run_seed, run_seeds};
pub use rng::Rng;
