//! Shared helpers for the panda-sta integration tests.
#![allow(dead_code)]

use std::sync::Arc;

use panda_circuits::{d_latch, ms_dff, toggle_ff};
use panda_compile::{FastOptions, FastRef, KernelOptions, compile_kernels};
use panda_netlist::flat::FlatNetlist;
use panda_netlist::{
    DefBuilder, DefRef, Design, DesignBuilder, DiagKind, Logic, Sig, Value, flatten,
};
use panda_sta::{TimingReport, analyze};
use panda_timing::{TimingOptions, TimingSim};

/// Flatten (panics with the diagnostics on failure).
pub fn flat_of(d: &Design) -> FlatNetlist {
    flatten(d).unwrap_or_else(|e| panic!("flatten failed: {e:?}"))
}

/// Compile the kernels with default options and analyze.
pub fn sta(flat: &FlatNetlist) -> TimingReport {
    let (ks, tables) = compile_kernels(flat, &KernelOptions::default());
    analyze(flat, &ks, &tables)
}

/// Pretty-print a report (shown with `--nocapture`).
pub fn dump(name: &str, r: &TimingReport) {
    println!("== {name}: min_period {:?}", r.min_period);
    for p in &r.phase_settle {
        println!(
            "  {:?} {}: settle {:?}, gap {:?}, rounds {}",
            p.kind, p.event, p.settle, p.gap, p.rounds
        );
    }
    for d in r.diagnostics.iter().take(20) {
        println!("  {d}");
    }
    if r.diagnostics.len() > 20 {
        println!("  ... {} diagnostics", r.diagnostics.len());
    }
}

/// Number of diagnostics of `kind`.
pub fn count(r: &TimingReport, kind: DiagKind) -> usize {
    r.of_kind(kind).count()
}

/// First disagreement between FastRef and Timing mode at phase boundaries.
#[derive(Debug)]
pub struct Mismatch {
    pub event_tick: u64,
    pub sample_tick: u64,
    pub net: String,
    pub fast: Logic,
    pub timing: Logic,
}

/// Run FastRef and TimingSim side by side for up to `events` events and compare **every net** at
/// every phase boundary (FastRef after an event at `t` = Timing at `next_event - 1`), starting with
/// the first event at or after tick `from`. Returns the number of compared phases or the first
/// mismatch.
pub fn differential(flat: &FlatNetlist, events: usize, from: u64) -> Result<usize, Mismatch> {
    differential_with(flat, events, from, &[])
}

/// [`differential`] with top-level inputs set before reset (both engines).
pub fn differential_with(
    flat: &FlatNetlist,
    events: usize,
    from: u64,
    inputs: &[(&str, Value)],
) -> Result<usize, Mismatch> {
    differential_opts(flat, events, from, inputs, TimingOptions::default())
}

/// [`differential_with`] with explicit Timing-mode options (e.g. ideal blocks).
pub fn differential_opts(
    flat: &FlatNetlist,
    events: usize,
    from: u64,
    inputs: &[(&str, Value)],
    topts: TimingOptions,
) -> Result<usize, Mismatch> {
    let flat = Arc::new(flat.clone());
    let mut fast = FastRef::new(flat.clone(), FastOptions::default()).expect("FastRef");
    let mut sim = TimingSim::new(flat.clone(), topts);
    for (pin, v) in inputs {
        fast.set_input(pin, v).expect("fast input");
        sim.set_input(pin, v).expect("timing input");
    }
    fast.reset();
    sim.reset();
    let mut compared = 0;
    for _ in 0..events {
        let Some(next) = fast.next_event_tick() else {
            break;
        };
        let target = next - 1;
        if target > sim.tick() {
            sim.run(target - sim.tick());
        }
        if fast.tick() >= from {
            sim.refresh_ideal_internal();
            let fv = fast.values();
            let tv = sim.values();
            if let Some(n) = (0..fv.len()).find(|&n| fv[n] != tv[n]) {
                return Err(Mismatch {
                    event_tick: fast.tick(),
                    sample_tick: sim.tick(),
                    net: flat.net_names[n].clone(),
                    fast: fv[n],
                    timing: tv[n],
                });
            }
            compared += 1;
        }
        fast.step_event();
    }
    Ok(compared)
}

/// Two flip-flops with synchronous reset: `ff1` (clock `clk`, `D = NOT Q2`) and `ff2` whose clock
/// is `clk` delayed by `delay` buffers and whose `D` is `ff1.Q` with no logic in between — a
/// classic hold violation once the delay exceeds the clock-to-output time of `ff1`.
pub fn hold_design(period: u32, delay: usize) -> Design {
    let mut db = DesignBuilder::new();
    let ff = ms_dff(&mut db);
    let mut b = DefBuilder::new(&format!("hold_top_{delay}"));
    let clk = b.clock(period, period / 2, period / 2);
    b.name(clk, "clk");
    let rst = b.power_on_reset(2 * period + period / 4);
    b.name(rst, "rst");
    let nrst = b.not(rst);
    let q2w = b.wire(1);
    let nq2 = b.not(q2w);
    let d1 = b.and(&[nq2, nrst]);
    let q1 = b.inst_named("ff1", &ff, &[d1, clk])[0];
    let mut c = clk;
    for _ in 0..delay {
        c = b.buf(c);
    }
    let q2 = b.inst_named("ff2", &ff, &[q1, c])[0];
    b.connect(q2, q2w);
    b.output("Q1", q1);
    b.output("Q2", q2);
    let top = db.add(b.finish());
    db.build(&top)
}

/// Defs available to the clock builder of [`shared_clock_design`].
pub struct Lib {
    /// `toggle_ff` (`T`, `RST`, `CLK` → `Q`, `Q_n`).
    pub tff: DefRef,
    /// `d_latch` (`D`, `EN` → `Q`, `Q_n`).
    pub latch: DefRef,
}

/// `n` toggle flip-flops (synchronous reset, `T = 1`) clocked by one shared derived clock `gclk`
/// built by `make_clock(builder, clk, rst, lib)`; outputs `Q0..`. Sharing keeps the clock logic
/// combinational (not absorbed into a single kernel as private fan-in).
pub fn shared_clock_design(
    id: &str,
    period: u32,
    n: usize,
    make_clock: impl FnOnce(&mut DefBuilder, Sig, Sig, &Lib) -> Sig,
) -> Design {
    let mut db = DesignBuilder::new();
    let lib = Lib {
        tff: toggle_ff(&mut db),
        latch: d_latch(&mut db),
    };
    let mut b = DefBuilder::new(id);
    let clk = b.clock(period, period / 2, period / 2);
    b.name(clk, "clk");
    let rst = b.power_on_reset(2 * period + period / 4);
    b.name(rst, "rst");
    let gclk = make_clock(&mut b, clk, rst, &lib);
    b.name(gclk, "gclk");
    let one = b.lit(1, 1);
    for i in 0..n {
        let q = b.inst_named(&format!("t{i}"), &lib.tff, &[one, rst, gclk])[0];
        b.output(&format!("Q{i}"), q);
    }
    let top = db.add(b.finish());
    db.build(&top)
}

/// [`differential`] with random UI input changes placed inside phases the way the M3 harness
/// does it: Timing mode runs to `event + margin - 1` (compared there too), both engines get the
/// new values (FastRef processes them as an input event), and the phase must leave room for the
/// input event to settle (`margin + input_settle < phase length`). `margin` / `input_settle` come
/// from the STA report (clock-event and input-event settle times).
pub fn differential_inputs(
    flat: &FlatNetlist,
    report: &TimingReport,
    events: usize,
    from: u64,
    pins: &[&str],
    seed: u64,
) -> Result<usize, Mismatch> {
    use panda_sta::EventKind;
    let settle = |k: EventKind| {
        report
            .phase_settle
            .iter()
            .filter(|p| p.kind == k)
            .map(|p| u64::from(p.settle.unwrap_or(u32::MAX / 4)))
            .max()
            .unwrap_or(0)
    };
    let margin = settle(EventKind::Clock).max(settle(EventKind::Reset)) + 1;
    let input_settle = settle(EventKind::Inputs);
    let flat = Arc::new(flat.clone());
    let mut fast = FastRef::new(flat.clone(), FastOptions::default()).expect("FastRef");
    let mut sim = TimingSim::new(flat.clone(), TimingOptions::default());
    fast.reset();
    sim.reset();
    let mut x = seed | 1;
    let mut rnd = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let compare = |fast: &FastRef, sim: &TimingSim, e: u64| -> Result<(), Mismatch> {
        let fv = fast.values();
        let tv = sim.values();
        match (0..fv.len()).find(|&n| fv[n] != tv[n]) {
            Some(n) => Err(Mismatch {
                event_tick: e,
                sample_tick: sim.tick(),
                net: flat.net_names[n].clone(),
                fast: fv[n],
                timing: tv[n],
            }),
            None => Ok(()),
        }
    };
    let mut compared = 0;
    for _ in 0..events {
        let e = fast.tick();
        let Some(next) = fast.next_event_tick() else {
            break;
        };
        let room = next - e;
        if room > margin + input_settle + 1 && rnd() % 2 == 0 {
            let at = e + margin - 1;
            if at > sim.tick() {
                sim.run(at - sim.tick());
            }
            if e >= from {
                compare(&fast, &sim, e)?;
                compared += 1;
            }
            for pin in pins {
                if rnd() % 2 == 0 {
                    let v = Value::from_bool(rnd() % 2 == 0);
                    fast.set_input(pin, &v).expect("fast input");
                    sim.set_input(pin, &v).expect("timing input");
                }
            }
            if fast.has_pending_inputs() {
                fast.step_event();
            }
        }
        if next - 1 > sim.tick() {
            sim.run(next - 1 - sim.tick());
        }
        if e >= from {
            compare(&fast, &sim, e)?;
            compared += 1;
        }
        fast.step_event();
    }
    Ok(compared)
}
