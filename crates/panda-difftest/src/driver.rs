//! The comparison driver: FastRef and Timing mode side by side, compared on every net at every
//! phase boundary.
//!
//! # Protocol
//!
//! Both engines are reset with the same initial input values. Then, for every phase (event `E` of
//! FastRef, next event `N`):
//!
//! 1. Optionally (probability [`DriverOptions::input_pct`], and only when the phase is long enough)
//!    random input changes are applied *safely inside the phase*: Timing mode is run to tick
//!    `E + margin - 1` (FastRef's clock-event values must equal Timing there too: a *mid-phase*
//!    comparison), the new values are given to both engines — Timing applies them at tick
//!    `E + margin`, FastRef processes them as an input event right after `E` (SPEC §6.2 "inputs
//!    change after the edge"). `margin` is the STA settle bound of clock events + 1 (after hold
//!    windows), and the phase must leave room for the input event to settle as well
//!    (`margin + input_settle < N - E`).
//! 2. Timing mode is run to tick `N - 1` and every net is compared with FastRef (*phase-end*
//!    comparison).
//! 3. FastRef processes event `N`.

use std::sync::Arc;

use panda_compile::{FastOptions, FastRef};
use panda_netlist::flat::FlatNetlist;
use panda_netlist::{Diagnostic, Logic, Value};
use panda_timing::{TimingOptions, TimingSim};

use crate::rng::Rng;

/// Where a comparison happened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Point {
    /// Tick `next_event - 1`.
    PhaseEnd,
    /// Tick `event + margin - 1`, just before an input change.
    BeforeInput,
    /// Timing mode settled after an input event with no scheduled event left (clockless).
    Settled,
}

/// One disagreeing net.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetDiff {
    /// Net index.
    pub net: u32,
    /// Canonical net name.
    pub name: String,
    /// FastRef value.
    pub fast: Logic,
    /// Timing-mode value.
    pub timing: Logic,
}

/// A comparison point with disagreeing nets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mismatch {
    /// Phase number (0 = the tick-0 phase).
    pub phase: usize,
    /// Tick of the FastRef event the phase starts with.
    pub event_tick: u64,
    /// Timing tick compared.
    pub sample_tick: u64,
    /// Comparison point.
    pub point: Point,
    /// Number of disagreeing nets.
    pub count: usize,
    /// The first few disagreeing nets.
    pub nets: Vec<NetDiff>,
    /// Every disagreement (not only the kept ones) involves an `X`/`Z` on at least one side.
    pub only_unknown: bool,
}

impl std::fmt::Display for Mismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "phase {} (event @{}, {:?} sample @{}): {} nets differ:",
            self.phase, self.event_tick, self.point, self.sample_tick, self.count
        )?;
        for d in &self.nets {
            write!(
                f,
                " {}=fast {}/timing {}",
                d.name,
                d.fast.to_char(),
                d.timing.to_char()
            )?;
        }
        Ok(())
    }
}

/// Timing parameters of a run (from STA, see [`crate::harness`]).
#[derive(Clone, Debug)]
pub struct Plan {
    /// Ticks after a clock/reset event before inputs may change (> settle of the event).
    pub input_margin: u64,
    /// Settle bound of an input event (ticks).
    pub input_settle: u64,
    /// Comparisons before this tick are not counted as mismatches (power-on).
    pub compare_from: u64,
    /// Input pins that may be changed: (name, width).
    pub inputs: Vec<(String, u32)>,
    /// Seed of the input stimulus.
    pub seed: u64,
}

/// Tunables of [`run`].
#[derive(Clone, Debug)]
pub struct DriverOptions {
    /// Phases to simulate.
    pub phases: usize,
    /// Percent of (long enough) phases with an input change.
    pub input_pct: u32,
    /// Stop at the first mismatch.
    pub stop_at_first: bool,
    /// Disagreeing nets kept per mismatch.
    pub keep_nets: usize,
    /// Mismatches kept.
    pub keep_mismatches: usize,
    /// Timing-mode options.
    pub timing: TimingOptions,
    /// Most ticks Timing mode may take to settle after an input event when no scheduled event
    /// is left.
    pub settle_bound: u64,
}

impl Default for DriverOptions {
    fn default() -> Self {
        DriverOptions {
            phases: 120,
            input_pct: 50,
            stop_at_first: true,
            keep_nets: 6,
            keep_mismatches: 4,
            timing: TimingOptions::default().with_stuck_x_periods(0),
            settle_bound: 50_000,
        }
    }
}

/// Result of [`run`].
#[derive(Clone, Debug, Default)]
pub struct DiffResult {
    /// Phases simulated.
    pub phases: usize,
    /// Comparisons made at or after `compare_from`.
    pub compared: usize,
    /// Input events applied.
    pub input_events: usize,
    /// Comparisons with disagreeing nets (at or after `compare_from`).
    pub mismatch_count: usize,
    /// The first mismatches.
    pub mismatches: Vec<Mismatch>,
    /// Mismatches before `compare_from` (power-on transients; not counted).
    pub early_mismatches: usize,
    /// Diagnostics raised by FastRef.
    pub fast_diags: Vec<Diagnostic>,
    /// Diagnostics raised by Timing mode.
    pub timing_diags: Vec<Diagnostic>,
}

impl DiffResult {
    /// True if no counted mismatch was found.
    pub fn equal(&self) -> bool {
        self.mismatch_count == 0
    }

    /// The first counted mismatch.
    pub fn first(&self) -> Option<&Mismatch> {
        self.mismatches.first()
    }
}

fn random_value(rng: &mut Rng, width: u32) -> Value {
    let mask = if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    Value::from_u64(width, rng.next_u64() & mask)
}

fn diff_nets(
    flat: &FlatNetlist,
    fast: &[Logic],
    timing: &[Logic],
    keep: usize,
) -> (usize, Vec<NetDiff>, bool) {
    let mut count = 0;
    let mut nets = Vec::new();
    let mut only_unknown = true;
    for (n, (&f, &t)) in fast.iter().zip(timing).enumerate() {
        if f != t {
            count += 1;
            only_unknown &= !f.is_known() || !t.is_known();
            if nets.len() < keep {
                nets.push(NetDiff {
                    net: n as u32,
                    name: flat.net_names[n].clone(),
                    fast: f,
                    timing: t,
                });
            }
        }
    }
    (count, nets, only_unknown)
}

/// Run FastRef (`fast`, freshly built or reset) against a new Timing-mode simulator of the same
/// netlist under `plan`. See the module docs.
pub fn run(
    flat: &Arc<FlatNetlist>,
    fast: FastRef,
    plan: &Plan,
    opts: &DriverOptions,
) -> DiffResult {
    run_observed(flat, fast, plan, opts, &mut |_, _, _, _| {})
}

/// [`run`], calling `observe(fast, timing, point, next_event_tick)` after every comparison (at a
/// phase end FastRef holds the values of the phase ending at `next_event_tick - 1`).
pub fn run_observed(
    flat: &Arc<FlatNetlist>,
    mut fast: FastRef,
    plan: &Plan,
    opts: &DriverOptions,
    observe: &mut dyn FnMut(&FastRef, &TimingSim, Point, u64),
) -> DiffResult {
    let mut rng = Rng::new(plan.seed ^ 0x1234_5678_9ABC_DEF0);
    let mut sim = TimingSim::new(flat.clone(), opts.timing.clone());
    for (name, w) in &plan.inputs {
        let v = random_value(&mut rng, *w);
        fast.set_input(name, &v).expect("fast input");
        sim.set_input(name, &v).expect("timing input");
    }
    fast.reset();
    sim.reset();
    let mut res = DiffResult::default();

    let record = |res: &mut DiffResult,
                  phase,
                  event_tick,
                  sample_tick,
                  point,
                  fast: &FastRef,
                  sim: &TimingSim|
     -> bool {
        let (count, nets, only_unknown) =
            diff_nets(flat, fast.values(), sim.values(), opts.keep_nets);
        if count == 0 {
            if sample_tick >= plan.compare_from {
                res.compared += 1;
            }
            return false;
        }
        if sample_tick < plan.compare_from {
            res.early_mismatches += 1;
            return false;
        }
        res.compared += 1;
        res.mismatch_count += 1;
        if res.mismatches.len() < opts.keep_mismatches {
            res.mismatches.push(Mismatch {
                phase,
                event_tick,
                sample_tick,
                point,
                count,
                nets,
                only_unknown,
            });
        }
        true
    };

    for phase in 0..opts.phases {
        let e = fast.tick();
        let Some(next) = fast.next_event_tick() else {
            // No scheduled source event left (clockless design after the reset release): settle
            // Timing mode, compare, then apply an input event.
            if plan.inputs.is_empty() {
                break;
            }
            res.phases += 1;
            sim.settle(opts.settle_bound);
            sim.refresh_ideal_internal();
            let bad = record(&mut res, phase, e, sim.tick(), Point::Settled, &fast, &sim);
            observe(&fast, &sim, Point::Settled, e);
            if bad && opts.stop_at_first {
                break;
            }
            loop {
                for (name, w) in &plan.inputs {
                    if rng.chance(50) {
                        let v = random_value(&mut rng, *w);
                        fast.set_input(name, &v).expect("fast input");
                        sim.set_input(name, &v).expect("timing input");
                    }
                }
                if fast.has_pending_inputs() {
                    break;
                }
            }
            res.input_events += 1;
            fast.step_event();
            continue;
        };
        res.phases += 1;
        let room = next - e;
        if !plan.inputs.is_empty()
            && room > plan.input_margin + plan.input_settle + 1
            && rng.chance(opts.input_pct)
        {
            // Randomize the margin within the safe window.
            let slack = room - (plan.input_margin + plan.input_settle + 1);
            let margin = plan.input_margin + rng.next_u64() % (slack + 1);
            let at = e + margin - 1;
            if at > sim.tick() {
                sim.run(at - sim.tick());
            }
            sim.refresh_ideal_internal();
            let bad = record(
                &mut res,
                phase,
                e,
                sim.tick(),
                Point::BeforeInput,
                &fast,
                &sim,
            );
            observe(&fast, &sim, Point::BeforeInput, next);
            if bad && opts.stop_at_first {
                break;
            }
            let mut changed = false;
            for (name, w) in &plan.inputs {
                if rng.chance(50) {
                    let v = random_value(&mut rng, *w);
                    fast.set_input(name, &v).expect("fast input");
                    sim.set_input(name, &v).expect("timing input");
                    changed = true;
                }
            }
            if changed && fast.has_pending_inputs() {
                res.input_events += 1;
                fast.step_event();
            }
        }
        let target = next - 1;
        if target > sim.tick() {
            sim.run(target - sim.tick());
        }
        sim.refresh_ideal_internal();
        let bad = record(&mut res, phase, e, sim.tick(), Point::PhaseEnd, &fast, &sim);
        observe(&fast, &sim, Point::PhaseEnd, next);
        if bad && opts.stop_at_first {
            break;
        }
        fast.step_event();
    }
    if fast.next_event_tick().is_none() && !(opts.stop_at_first && res.mismatch_count > 0) {
        sim.settle(opts.settle_bound);
        sim.refresh_ideal_internal();
        let e = fast.tick();
        record(
            &mut res,
            opts.phases,
            e,
            sim.tick(),
            Point::Settled,
            &fast,
            &sim,
        );
        observe(&fast, &sim, Point::Settled, e);
    }
    res.fast_diags = fast.take_diagnostics();
    res.timing_diags = sim.take_diagnostics();
    res
}

/// Convenience: build FastRef with default options and [`run`].
pub fn run_netlist(
    flat: &Arc<FlatNetlist>,
    plan: &Plan,
    opts: &DriverOptions,
) -> Result<DiffResult, Vec<Diagnostic>> {
    let fast = FastRef::new(flat.clone(), FastOptions::default())?;
    Ok(run(flat, fast, plan, opts))
}
