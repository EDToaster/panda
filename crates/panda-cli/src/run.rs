//! `panda run` and `panda test`: simulate a design in Timing or Fast mode.
//!
//! Timing mode without `mode` directives runs directly on [`TimingSim`] (the M1 behaviour). Fast
//! mode, and vector files that switch modes, run through the `panda_kernel` facade with the
//! backend chosen by `--backend` (default: the compiled `FastSim`).

use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use panda_kernel::{FastBackend, Kernel, Mode, StepUnit};
use panda_netlist::flat::FlatNetlist;
use panda_netlist::{Design, Value};
use panda_timing::TimingSim;

use crate::args::Args;
use crate::clock::{self, RefClock};
use crate::commands::{io_err, load_netlist, open_vcd, out_err, signal_lists, timing_options};
use crate::fastmode::{self, KernelSettings, backend_name, dur, engine_name, rate, si};
use crate::signals::{self, Signal};
use crate::vcd::{self, VcdWriter};
use crate::vectors::{self, Program};
use crate::{CliError, format_value, read_file, write_diagnostics};

/// `--mode` and `--backend`.
fn mode_and_backend(a: &Args) -> Result<(Option<Mode>, Option<FastBackend>), CliError> {
    let mode = a.choice("mode", &["timing", "fast"])?.and_then(fastmode::parse_mode);
    let backend = match a.last("backend") {
        Some(b) => Some(fastmode::parse_backend(b).ok_or_else(|| {
            CliError::Usage(format!("--backend: expected jit, compiled or reference, got '{b}'"))
        })?),
        None => None,
    };
    Ok((mode, backend))
}

/// `--set PIN=VALUE[,PIN=VALUE...]` (repeatable), checked against the top-level inputs.
fn parse_sets(nl: &FlatNetlist, a: &Args) -> Result<Vec<(String, Value)>, CliError> {
    let mut sets = Vec::new();
    for item in a.all("set") {
        for item in item.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let (pin, val) =
                item.split_once('=').ok_or_else(|| CliError::Usage(format!("--set: expected PIN=VALUE, got '{item}'")))?;
            let p = nl
                .top_inputs
                .iter()
                .find(|p| p.name == pin.trim())
                .ok_or_else(|| CliError::Usage(format!("--set: '{pin}' is not a top-level input pin")))?;
            let v = Value::parse(p.nets.len() as u32, val).map_err(|e| CliError::Usage(format!("--set {pin}: {e}")))?;
            sets.push((p.name.clone(), v));
        }
    }
    Ok(sets)
}

/// The reference clock (`--clock`, else the first running clock).
fn find_clock(nl: &FlatNetlist, a: &Args) -> Result<Option<RefClock>, CliError> {
    RefClock::find(nl, a.last("clock")).map_err(|e| CliError::Usage(format!("--clock: {e}")))
}

/// What `panda run` was asked to simulate.
#[derive(Clone, Copy, Debug)]
enum Span {
    Ticks(u64),
    Cycles(u64),
    Settle(u64),
}

/// Everything `panda run` needs after argument parsing.
struct RunJob {
    design: Design,
    nl: Arc<FlatNetlist>,
    sets: Vec<(String, Value)>,
    watch: Vec<Signal>,
    vcd: Option<(String, VcdWriter<std::io::BufWriter<std::fs::File>>)>,
    span: Span,
    clock: Option<RefClock>,
    stats: bool,
}

impl RunJob {
    /// Cycle steps requested with `--cycles`.
    fn cycles(&self) -> Option<u64> {
        match self.span {
            Span::Cycles(n) => Some(n),
            _ => None,
        }
    }
}

/// `panda run`.
pub(crate) fn cmd_run(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> Result<i32, CliError> {
    let a = Args::parse(
        args,
        &[
            "ticks", "max-ticks", "cycles", "set", "watch", "vcd", "trace", "p2-limit", "osc-limit", "mode", "backend",
            "clock",
        ],
        &["stats"],
        &[],
    )?;
    let path = a.positionals("run", &["design.json"])?[0].clone();
    let (mode, backend) = mode_and_backend(&a)?;
    let mode = mode.unwrap_or(Mode::Timing);
    if backend.is_some() && mode != Mode::Fast {
        return Err(CliError::Usage("--backend needs --mode fast".into()));
    }
    let ticks = a.u64("ticks")?;
    let cycles = a.u64("cycles")?;
    let max_ticks = a.u64("max-ticks")?.unwrap_or(vectors::DEFAULT_SETTLE_MAX);
    let span = match (ticks, cycles) {
        (Some(_), Some(_)) => return Err(CliError::Usage("give either --ticks or --cycles, not both".into())),
        (Some(t), None) => Span::Ticks(t),
        (None, Some(c)) => Span::Cycles(c),
        (None, None) => Span::Settle(max_ticks),
    };
    let opts = timing_options(&a)?;
    let (design, nl) = load_netlist(&path, err)?;
    let sets = parse_sets(&nl, &a)?;
    let clock = find_clock(&nl, &a)?;
    if matches!(span, Span::Cycles(_)) && clock.is_none() {
        return Err(CliError::Usage("--cycles needs a design with a running Clock".into()));
    }
    let watch = signal_lists(&nl, &a, "watch", || {
        let outs: Vec<Signal> = nl
            .top_outputs
            .iter()
            .filter(|p| !p.nets.is_empty())
            .map(|p| Signal { name: p.name.clone(), nets: p.nets.clone() })
            .collect();
        if outs.is_empty() { signals::default_selection(&nl) } else { outs }
    })?;
    let vcd = match a.last("vcd") {
        Some(vpath) => {
            let sigs = signal_lists(&nl, &a, "trace", || signals::default_selection(&nl))?;
            Some((vpath.to_string(), open_vcd(vpath, &design, &nl, sigs)?))
        }
        None => None,
    };
    let job = RunJob { design, nl, sets, watch, vcd, span, clock, stats: a.flag("stats") };
    match mode {
        Mode::Timing => run_timing(job, opts, out),
        Mode::Fast => {
            let settings = KernelSettings {
                timing: opts,
                backend: backend.unwrap_or_default(),
                cycle_clock: job.clock.as_ref().map(|c| c.name.clone()),
                max_settle_ticks: vectors::DEFAULT_SETTLE_MAX,
            };
            run_fast(job, &settings, out)
        }
    }
}

/// Write the watched signals.
fn write_watch(out: &mut dyn Write, watch: &[Signal], get: impl Fn(&Signal) -> Value) -> Result<(), CliError> {
    let width = watch.iter().map(|s| s.name.len()).max().unwrap_or(0);
    for s in watch {
        writeln!(out, "  {:<width$} = {}", s.name, format_value(&get(s))).map_err(out_err)?;
    }
    Ok(())
}

/// Write the throughput line.
///
/// `cycles`: the number of cycle steps taken (`--cycles`); otherwise the rising edges of the
/// reference clock passed in `(t0, t1]` are counted.
fn write_throughput(
    out: &mut dyn Write,
    clock: Option<&RefClock>,
    cycles: Option<u64>,
    t0: u64,
    t1: u64,
    elapsed: Duration,
) -> Result<(), CliError> {
    let ticks = t1 - t0;
    let mut line = format!("throughput: {ticks} ticks in {}", dur(elapsed));
    if let Some(c) = clock {
        let first = c.next_rise_after(t0);
        let edges = if first > t1 { 0 } else { (t1 - first) / c.period + 1 };
        let cycles = cycles.unwrap_or(edges);
        line.push_str(&format!(", {cycles} cycles of {} = {} cycles/s", c.name, si(rate(cycles, elapsed))));
    }
    line.push_str(&format!(" ({} ticks/s)", si(rate(ticks, elapsed))));
    writeln!(out, "{line}").map_err(out_err)
}

fn run_timing(mut job: RunJob, opts: panda_timing::TimingOptions, out: &mut dyn Write) -> Result<i32, CliError> {
    let mut sim = TimingSim::new(job.nl.clone(), opts);
    for (pin, v) in &job.sets {
        sim.set_input(pin, v).map_err(CliError::Other)?;
    }
    // Inputs given with --set are part of the power-on state.
    if !job.sets.is_empty() {
        sim.reset();
    }
    let vpath = job.vcd.as_ref().map(|(p, _)| p.clone()).unwrap_or_default();
    if let Some((_, w)) = job.vcd.as_mut() {
        w.sample(&sim).map_err(io_err(&vpath))?;
    }
    let w = job.vcd.as_mut().map(|(_, w)| w);
    let t0 = sim.tick();
    let start = Instant::now();
    let status = match job.span {
        Span::Ticks(n) => {
            vcd::advance(&mut sim, n, w).map_err(io_err(&vpath))?;
            format!("ran {n} ticks{}", if sim.is_quiescent() { " (quiescent)" } else { "" })
        }
        Span::Cycles(n) => {
            let c = job.clock.as_ref().expect("checked");
            clock::run_cycles(&mut sim, c, n, w).map_err(io_err(&vpath))?;
            format!("ran {n} cycles of {}", c.name)
        }
        Span::Settle(max) => {
            if vcd::settle(&mut sim, max, w).map_err(io_err(&vpath))? {
                "settled".to_string()
            } else {
                format!("not quiescent after {max} ticks")
            }
        }
    };
    let elapsed = start.elapsed();
    if let Some((p, w)) = job.vcd.take() {
        w.finish(&sim).map_err(io_err(&p))?;
    }

    writeln!(out, "tick {}: {status}", sim.tick()).map_err(out_err)?;
    write_watch(out, &job.watch, |s| vectors::gather(&sim, s))?;
    if job.stats || matches!(job.span, Span::Cycles(_)) {
        writeln!(out, "mode: timing").map_err(out_err)?;
        write_throughput(out, job.clock.as_ref(), job.cycles(), t0, sim.tick(), elapsed)?;
    }
    let diags = sim.take_diagnostics();
    write_diagnostics(out, "note: ", &diags).map_err(out_err)?;
    if sim.suppressed_diagnostics() > 0 {
        writeln!(out, "note: {} further diagnostics suppressed", sim.suppressed_diagnostics()).map_err(out_err)?;
    }
    Ok(0)
}

/// Advance a Fast-mode kernel phase by phase until `target`, sampling into `vcd`.
fn fast_phases_to(
    k: &mut Kernel,
    target: u64,
    vcd: &mut VcdWriter<std::io::BufWriter<std::fs::File>>,
    vpath: &str,
) -> Result<(), CliError> {
    while k.tick() < target {
        let before = k.tick();
        k.step(1, StepUnit::Phase);
        if k.tick() == before {
            break;
        }
        vcd.sample_values(k.tick(), &fastmode::values(k)).map_err(io_err(vpath))?;
    }
    Ok(())
}

fn run_fast(mut job: RunJob, settings: &KernelSettings, out: &mut dyn Write) -> Result<i32, CliError> {
    let mut k = fastmode::new_kernel(&job.design, settings, &job.sets)?;
    let compile = fastmode::enter_mode(&mut k, Mode::Fast).map_err(CliError::Other)?;
    let kn = k.netlist().expect("loaded").clone();
    if kn.num_nets != job.nl.num_nets {
        return Err(CliError::Other("internal: the facade flattened the design differently".into()));
    }
    let vpath = job.vcd.as_ref().map(|(p, _)| p.clone()).unwrap_or_default();
    if let Some((_, w)) = job.vcd.as_mut() {
        w.sample_values(k.tick(), &fastmode::values(&k)).map_err(io_err(&vpath))?;
    }
    let t0 = k.tick();
    let start = Instant::now();
    // Targets count from power-on, exactly as in Timing mode: the Fast engine starts at the end of
    // the first phase and ends at the end of the phase containing the target.
    let status = match job.span {
        Span::Ticks(n) => {
            match job.vcd.as_mut() {
                Some((_, w)) => {
                    k.step(0, StepUnit::Tick);
                    fast_phases_to(&mut k, n, w, &vpath)?;
                }
                None => k.run_to(n),
            }
            format!("ran to the end of the phase containing tick {n}")
        }
        Span::Cycles(n) => {
            let c = job.clock.as_ref().expect("checked");
            let target = c.cycles_end(0, n);
            match job.vcd.as_mut() {
                Some((_, w)) => {
                    k.step(0, StepUnit::Tick);
                    fast_phases_to(&mut k, target, w, &vpath)?;
                }
                None => k.run_to(target),
            }
            format!("ran {n} cycles of {}", c.name)
        }
        Span::Settle(max) => {
            let mut settled = false;
            loop {
                let before = k.tick();
                k.step(1, StepUnit::Phase);
                if let Some((_, w)) = job.vcd.as_mut() {
                    w.sample_values(k.tick(), &fastmode::values(&k)).map_err(io_err(&vpath))?;
                }
                if k.tick() == before {
                    settled = true;
                    break;
                }
                if k.tick() - t0 >= max {
                    break;
                }
            }
            if settled { "settled".to_string() } else { format!("still changing after {} ticks", k.tick() - t0) }
        }
    };
    let elapsed = start.elapsed();
    if let Some((p, w)) = job.vcd.take() {
        w.finish_at(k.tick()).map_err(io_err(&p))?;
    }

    writeln!(out, "tick {}: {status}", k.tick()).map_err(out_err)?;
    let values = fastmode::values(&k);
    write_watch(out, &job.watch, |s| vectors::gather_values(&values, s))?;
    writeln!(
        out,
        "mode: fast ({} backend; engine {}; compiled in {})",
        backend_name(settings.backend),
        engine_name(k.engine()),
        dur(compile)
    )
    .map_err(out_err)?;
    if job.stats || job.clock.is_some() || k.tick() > t0 {
        // Cycle steps actually simulated here (the first phase ran before the clock started).
        let steps = job.cycles().zip(job.clock.as_ref()).map(|(n, c)| n.saturating_sub(c.steps_done(t0)));
        write_throughput(out, job.clock.as_ref(), steps, t0, k.tick(), elapsed)?;
    }
    fastmode::write_entries(out, "note: ", k.diagnostic_entries()).map_err(out_err)?;
    Ok(0)
}

/// `panda test`.
pub(crate) fn cmd_test(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> Result<i32, CliError> {
    let a = Args::parse(
        args,
        &["vcd", "trace", "p2-limit", "osc-limit", "mode", "backend", "clock"],
        &["verbose"],
        &[("v", "verbose")],
    )?;
    let pos = a.positionals("test", &["design.json", "vectors.txt"])?;
    let (path, vpath) = (pos[0].clone(), pos[1].clone());
    let (mode, backend) = mode_and_backend(&a)?;
    let opts = timing_options(&a)?;
    let (design, nl) = load_netlist(&path, err)?;
    let clock = find_clock(&nl, &a)?;
    let src = read_file(Path::new(&vpath))?;
    let mut program = match Program::parse(&src, &nl) {
        Ok(p) => p,
        Err(errors) => {
            for e in &errors {
                writeln!(err, "{vpath}:{}: error: {}", e.line, e.message).map_err(out_err)?;
            }
            writeln!(err, "{vpath}: {} error(s); nothing was run", errors.len()).map_err(out_err)?;
            return Ok(1);
        }
    };
    if clock.is_some() {
        program.clock = clock;
    }
    let use_kernel = program.uses_modes() || mode == Some(Mode::Fast) || backend.is_some();
    let vcd = match a.last("vcd") {
        Some(dump) => {
            let sigs = signal_lists(&nl, &a, "trace", || signals::default_selection(&nl))?;
            Some((dump.to_string(), open_vcd(dump, &design, &nl, sigs)?))
        }
        None => None,
    };
    let report = if use_kernel {
        let settings = KernelSettings {
            timing: opts,
            backend: backend.unwrap_or_default(),
            cycle_clock: program.clock.as_ref().map(|c| c.name.clone()),
            max_settle_ticks: program.max_settle(),
        };
        let mut k = fastmode::new_kernel(&design, &settings, &[])?;
        if mode == Some(Mode::Fast) {
            fastmode::enter_mode(&mut k, Mode::Fast).map_err(CliError::Other)?;
        }
        match vcd {
            Some((dump, mut w)) => {
                let r = program.run_kernel(&mut k, Some(&mut w)).map_err(io_err(&dump))?;
                w.finish_at(k.tick()).map_err(io_err(&dump))?;
                r
            }
            None => program.run_kernel::<std::io::Sink>(&mut k, None).map_err(out_err)?,
        }
    } else {
        let mut sim = TimingSim::new(nl.clone(), opts);
        match vcd {
            Some((dump, mut w)) => {
                let r = program.run_traced(&mut sim, &mut w).map_err(io_err(&dump))?;
                w.finish(&sim).map_err(io_err(&dump))?;
                r
            }
            None => program.run(&mut sim),
        }
    };
    report.write(out, &vpath, a.flag("verbose")).map_err(out_err)?;
    Ok(if report.passed() { 0 } else { 1 })
}
