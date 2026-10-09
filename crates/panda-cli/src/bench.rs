//! `panda bench`: clock-cycle throughput of the engines on one design.
//!
//! Engines (`--engines`, comma-separated; default `timing,fastref,fastsim,jit`):
//!
//! * `timing` — the unit-delay reference engine ([`TimingSim`]);
//! * `fastref` — the Fast-mode oracle ([`FastRef`], SPEC §6.2);
//! * `fastsim` — the compiled Fast-mode engine ([`FastSim`], bytecode interpreter);
//! * `jit` — [`FastSim`] with its program compiled to native code by the Cranelift JIT
//!   ([`panda_jit`], SPEC §8.9) and installed before the run (the setup time includes the JIT
//!   compilation);
//! * `kernel` — the `panda_kernel` facade in Fast mode with its default backend
//!   ([`FastBackend::Jit`](panda_kernel::FastBackend::Jit): FastSim plus the JIT, plus the
//!   facade's per-step bookkeeping).
//!
//! A *cycle* is one period of the reference clock (`--clock`, default the first running clock);
//! every engine is advanced to the same sample points (one tick before a rising edge). With
//! `--cycles N` every engine runs exactly `N` cycles from power-on and the top-level outputs are
//! compared at the end (exit code 1 on a mismatch); otherwise each engine runs for `--seconds S`
//! (default 1) after a one-cycle warm-up, in chunks of cycles sized to the remaining time.
//! Kernel extraction and characterization are done once and shared by `fastref`, `fastsim` and
//! `jit`.

use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};

use panda_compile::{FastOptions, FastRef, KernelOptions as CompileOptions};
use panda_fast::{FastSim, FastSimOptions};
use panda_kernel::{Kernel, Mode};
use panda_netlist::flat::FlatNetlist;
use panda_netlist::{Logic, Value};
use panda_timing::TimingSim;

use crate::CliError;
use crate::args::Args;
use crate::clock::RefClock;
use crate::commands::{load_netlist, out_err, timing_options};
use crate::fastmode::{self, KernelSettings, dur, rate, si};

/// Engine names accepted by `--engines`.
pub const ENGINES: &[&str] = &["timing", "fastref", "fastsim", "jit", "kernel"];

/// Engines run when `--engines` is not given.
pub const DEFAULT_ENGINES: &[&str] = &["timing", "fastref", "fastsim", "jit"];

/// One benchmarked engine.
enum Engine {
    Timing(Box<TimingSim>),
    FastRef(Box<FastRef>),
    FastSim(Box<FastSim>),
    /// A [`FastSim`] with JIT-compiled native code installed.
    Jit(Box<FastSim>),
    Kernel(Box<Kernel>),
}

/// Work counters (cumulative) of an engine.
#[derive(Clone, Copy, Default)]
struct Work {
    a: u64,
    b: u64,
    c: u64,
}

impl Engine {
    fn tick(&self) -> u64 {
        match self {
            Engine::Timing(s) => s.tick(),
            // The fast engines' values at a sample point are those of Timing mode at
            // `next_event - 1`; run_to targets are always such points.
            Engine::FastRef(s) => s.next_event_tick().map_or(s.tick(), |e| e - 1),
            Engine::FastSim(s) | Engine::Jit(s) => s.next_event_tick().map_or(s.tick(), |e| e - 1),
            Engine::Kernel(k) => k.tick(),
        }
    }

    fn run_to(&mut self, t: u64) {
        match self {
            Engine::Timing(s) => {
                let now = s.tick();
                if t > now {
                    s.run(t - now);
                }
            }
            Engine::FastRef(s) => s.run_until(t),
            Engine::FastSim(s) | Engine::Jit(s) => s.run_until(t),
            Engine::Kernel(k) => k.run_to(t),
        }
    }

    fn values(&self) -> Vec<Logic> {
        match self {
            Engine::Timing(s) => s.values().to_vec(),
            Engine::FastRef(s) => s.values().to_vec(),
            Engine::FastSim(s) | Engine::Jit(s) => s.values().to_vec(),
            Engine::Kernel(k) => fastmode::values(k),
        }
    }

    fn work(&self) -> Work {
        match self {
            Engine::Timing(s) => {
                let st = s.stats();
                Work { a: st.gate_evals, b: st.block_evals, c: 0 }
            }
            Engine::FastRef(s) => {
                let st = s.stats();
                Work { a: st.comb_evals, b: st.transitions, c: st.events }
            }
            Engine::FastSim(s) => {
                let st = s.stats();
                Work { a: st.ops, b: st.partitions, c: st.transitions }
            }
            Engine::Jit(s) => {
                let st = s.stats();
                Work { a: st.rounds, b: st.native_rounds, c: st.native_bails }
            }
            Engine::Kernel(_) => Work::default(),
        }
    }

    /// Per-cycle work description from counter deltas.
    fn work_str(&self, before: Work, after: Work, cycles: u64) -> String {
        let per = |x: u64, y: u64| (y - x) as f64 / cycles.max(1) as f64;
        match self {
            Engine::Timing(_) => format!(
                "{:.0} gate evals/cycle{}",
                per(before.a, after.a),
                if after.b > before.b { format!(", {:.0} block evals/cycle", per(before.b, after.b)) } else { String::new() }
            ),
            Engine::FastRef(_) => format!(
                "{:.0} comb evals, {:.1} kernel transitions, {:.2} events per cycle",
                per(before.a, after.a),
                per(before.b, after.b),
                per(before.c, after.c)
            ),
            Engine::FastSim(_) => format!(
                "{:.0} ops, {:.1} partitions, {:.1} kernel transitions per cycle",
                per(before.a, after.a),
                per(before.b, after.b),
                per(before.c, after.c)
            ),
            Engine::Jit(_) => format!(
                "{:.1} kernel rounds ({:.1} native), {:.2} native bails per cycle",
                per(before.a, after.a),
                per(before.b, after.b),
                per(before.c, after.c)
            ),
            Engine::Kernel(_) => "facade, default backend".into(),
        }
    }
}

/// Measured result of one engine.
struct Measured {
    name: &'static str,
    setup: Duration,
    cycles: u64,
    elapsed: Duration,
    work: String,
    values: Vec<Logic>,
    tick: u64,
}

/// `panda bench`.
pub(crate) fn cmd_bench(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> Result<i32, CliError> {
    let a = Args::parse(args, &["cycles", "seconds", "engines", "clock", "set", "p2-limit", "osc-limit"], &[], &[])?;
    let path = a.positionals("bench", &["design.json"])?[0].clone();
    let cycles = a.u64("cycles")?;
    let seconds = a.positive_f64("seconds")?;
    if cycles.is_some() && seconds.is_some() {
        return Err(CliError::Usage("give either --cycles or --seconds, not both".into()));
    }
    if cycles == Some(0) {
        return Err(CliError::Usage("--cycles must be ≥ 1".into()));
    }
    let budget = Duration::from_secs_f64(seconds.unwrap_or(1.0));
    let engines: Vec<&'static str> = match a.last("engines") {
        Some(list) => list
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|e| {
                ENGINES.iter().copied().find(|x| x.eq_ignore_ascii_case(e)).ok_or_else(|| {
                    CliError::Usage(format!("--engines: unknown engine '{e}' (available: {})", ENGINES.join(", ")))
                })
            })
            .collect::<Result<_, _>>()?,
        None => DEFAULT_ENGINES.to_vec(),
    };
    if engines.is_empty() {
        return Err(CliError::Usage("--engines: no engine given".into()));
    }
    let topts = timing_options(&a)?;
    let (design, nl) = load_netlist(&path, err)?;
    let clock = RefClock::find(&nl, a.last("clock"))
        .map_err(|e| CliError::Usage(format!("--clock: {e}")))?
        .ok_or_else(|| CliError::Usage("'panda bench' needs a design with a running Clock".into()))?;
    let mut sets = Vec::new();
    for item in a.all("set").flat_map(|s| s.split(',')).map(str::trim).filter(|s| !s.is_empty()) {
        let (pin, val) =
            item.split_once('=').ok_or_else(|| CliError::Usage(format!("--set: expected PIN=VALUE, got '{item}'")))?;
        let p = nl
            .top_inputs
            .iter()
            .find(|p| p.name == pin)
            .ok_or_else(|| CliError::Usage(format!("--set: '{pin}' is not a top-level input pin")))?;
        sets.push((p.name.clone(), Value::parse(p.nets.len() as u32, val).map_err(CliError::Usage)?));
    }

    writeln!(
        out,
        "bench: {path}: {} gates, {} nets; clock {} (period {}); {}",
        nl.gates.len(),
        nl.num_nets,
        clock.name,
        clock.period,
        match cycles {
            Some(n) => format!("{n} cycles from power-on per engine"),
            None => format!("{} per engine", dur(budget)),
        }
    )
    .map_err(out_err)?;

    // Kernel tables, shared by FastRef and FastSim.
    let needs_tables = engines.iter().any(|e| matches!(*e, "fastref" | "fastsim" | "jit"));
    let fopts = FastOptions::default();
    let parts = if needs_tables {
        let t0 = Instant::now();
        let kopts = CompileOptions { extract: fopts.extract.clone(), charz: fopts.charz.clone() };
        let (ks, tables) = panda_compile::compile_kernels(&nl, &kopts);
        let h = ks.histogram();
        writeln!(
            out,
            "kernels: {} (table {}, sim {}, keeper {}), extracted and characterized in {}",
            ks.kernels.len(),
            h.table,
            h.sim,
            h.keeper,
            dur(t0.elapsed())
        )
        .map_err(out_err)?;
        Some((ks, tables))
    } else {
        None
    };

    let mut results: Vec<Measured> = Vec::new();
    for name in engines {
        let t0 = Instant::now();
        let mut eng = build(name, &nl, &design, &topts, &fopts, parts.as_ref(), &sets, &clock)?;
        let setup = t0.elapsed();
        let (n, elapsed, work) = match cycles {
            Some(n) => {
                let end = clock.cycles_end(0, n);
                let w0 = eng.work();
                let t = Instant::now();
                eng.run_to(end);
                let el = t.elapsed();
                (n, el, eng.work_str(w0, eng.work(), n))
            }
            None => {
                // Warm up one cycle (power-on, compilation stand-ins), then run in chunks.
                let warm = clock.cycles_end(eng.tick(), 1);
                eng.run_to(warm);
                let w0 = eng.work();
                let t = Instant::now();
                let mut done = 0u64;
                let mut chunk = 1u64;
                loop {
                    let elapsed = t.elapsed();
                    if elapsed >= budget {
                        break;
                    }
                    let end = clock.cycles_end(eng.tick(), chunk);
                    eng.run_to(end);
                    done += chunk;
                    // Next chunk: at most double, sized to about the remaining time.
                    let spent = t.elapsed();
                    let per = spent.as_secs_f64() / done as f64;
                    let left = budget.saturating_sub(spent).as_secs_f64();
                    chunk = ((left / per.max(1e-12)) as u64).clamp(1, chunk.saturating_mul(2));
                }
                let el = t.elapsed();
                (done, el, eng.work_str(w0, eng.work(), done))
            }
        };
        let m = Measured { name, setup, cycles: n, elapsed, work, values: eng.values(), tick: eng.tick() };
        writeln!(
            out,
            "  {:<8} setup {:>9}   {:>9} cycles in {:>9} = {:>7} cycles/s ({:>9}/cycle)   {}",
            m.name,
            dur(m.setup),
            m.cycles,
            dur(m.elapsed),
            si(rate(m.cycles, m.elapsed)),
            dur(m.elapsed / u32::try_from(m.cycles.max(1)).unwrap_or(u32::MAX)),
            m.work
        )
        .map_err(out_err)?;
        results.push(m);
    }

    if let Some(base) = results.iter().find(|m| m.name == "timing") {
        let base_rate = rate(base.cycles, base.elapsed);
        let others: Vec<String> = results
            .iter()
            .filter(|m| m.name != "timing")
            .map(|m| format!("{} {:.1}×", m.name, rate(m.cycles, m.elapsed) / base_rate))
            .collect();
        if !others.is_empty() {
            writeln!(out, "speedup vs timing: {}", others.join(", ")).map_err(out_err)?;
        }
    }

    let mut code = 0;
    if cycles.is_some() && results.len() > 1 {
        let first = &results[0];
        let mut mismatches = Vec::new();
        for m in &results[1..] {
            if m.tick != first.tick {
                mismatches.push(format!("{}: at tick {} vs {}", m.name, m.tick, first.tick));
                continue;
            }
            for p in &nl.top_outputs {
                let get = |v: &[Logic]| Value::from_bits(&p.nets.iter().map(|&n| v[n as usize]).collect::<Vec<_>>());
                let (x, y) = (get(&first.values), get(&m.values));
                if x != y {
                    mismatches.push(format!("{}: {} = {y} vs {} {x}", m.name, p.name, first.name));
                }
            }
        }
        if mismatches.is_empty() {
            writeln!(out, "outputs agree at tick {} across {} engines", first.tick, results.len()).map_err(out_err)?;
        } else {
            code = 1;
            writeln!(out, "outputs DIFFER:").map_err(out_err)?;
            for m in mismatches {
                writeln!(out, "  {m}").map_err(out_err)?;
            }
        }
    }
    Ok(code)
}

type Parts = (panda_compile::KernelSet, Vec<Option<Arc<panda_compile::KernelTable>>>);

#[allow(clippy::too_many_arguments)]
fn build(
    name: &str,
    nl: &Arc<FlatNetlist>,
    design: &panda_netlist::Design,
    topts: &panda_timing::TimingOptions,
    fopts: &FastOptions,
    parts: Option<&Parts>,
    sets: &[(String, Value)],
    clock: &RefClock,
) -> Result<Engine, CliError> {
    let parts = || parts.cloned().expect("tables compiled");
    Ok(match name {
        "timing" => {
            let mut s = TimingSim::new(nl.clone(), topts.clone());
            for (p, v) in sets {
                s.set_input(p, v).map_err(CliError::Other)?;
            }
            if !sets.is_empty() {
                s.reset();
            }
            Engine::Timing(Box::new(s))
        }
        "fastref" => {
            let (ks, tables) = parts();
            let mut s = FastRef::from_parts(nl.clone(), fopts.clone(), ks, tables).map_err(CliError::Design)?;
            for (p, v) in sets {
                s.set_input(p, v).map_err(CliError::Other)?;
            }
            Engine::FastRef(Box::new(s))
        }
        "fastsim" => {
            let (ks, tables) = parts();
            let mut s =
                FastSim::from_parts(nl.clone(), FastSimOptions::from(fopts.clone()), ks, tables).map_err(CliError::Design)?;
            for (p, v) in sets {
                s.set_input(p, v).map_err(CliError::Other)?;
            }
            Engine::FastSim(Box::new(s))
        }
        "jit" => {
            let (ks, tables) = parts();
            let mut s =
                FastSim::from_parts(nl.clone(), FastSimOptions::from(fopts.clone()), ks, tables).map_err(CliError::Design)?;
            panda_jit::Jit::new(panda_jit::JitOptions::default())
                .install(&mut s)
                .map_err(|e| CliError::Other(format!("JIT compilation failed: {e}")))?;
            for (p, v) in sets {
                s.set_input(p, v).map_err(CliError::Other)?;
            }
            Engine::Jit(Box::new(s))
        }
        "kernel" => {
            let settings = KernelSettings {
                timing: topts.clone(),
                backend: panda_kernel::FastBackend::default(),
                cycle_clock: Some(clock.name.clone()),
                max_settle_ticks: crate::vectors::DEFAULT_SETTLE_MAX,
            };
            let mut k = fastmode::new_kernel(design, &settings, sets)?;
            fastmode::enter_mode(&mut k, Mode::Fast).map_err(CliError::Other)?;
            Engine::Kernel(Box::new(k))
        }
        other => unreachable!("engine {other} validated"),
    })
}
