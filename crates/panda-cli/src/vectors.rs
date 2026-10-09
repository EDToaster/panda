//! Test vectors: a line-based text format and its runner.
//!
//! # Format
//!
//! One command per line. `#` starts a comment (anywhere on a line); blank lines are ignored.
//! Keywords are case-sensitive. Signal names are resolved with [`crate::signals::resolve`]
//! (top-level pins, any flat net name, bus base names, `base[hi:lo]` ranges).
//!
//! | command | meaning |
//! |---|---|
//! | `set P=V [P=V ...]` | set top-level **input** pins. Takes effect at the next tick. |
//! | `run N` / `tick N` | advance exactly `N` ticks (default `1`). |
//! | `settle [MAX]` | run until quiescent, at most `MAX` ticks (default 100000). Not settling is a failure. |
//! | `pulse P [N]` | `N` times (default 1): `set P=1`, settle, `set P=0`, settle. `P` must be a 1-bit input. A pulse that does not settle is a failure. |
//! | `expect S=V [S=V ...]` | check signals now. Each `S=V` is one check. |
//! | `print S [S ...]` | print current values (for debugging; never fails). |
//! | `reset` | power-on reset (tick 0; all gate outputs `X`). Input values set so far are kept. |
//! | `cycle [N]` | advance `N` clock cycles (default `1`) of the reference clock: to the tick just before the `N`-th rising edge after the next tick (see [`crate::clock`]). Needs a running `Clock`. |
//! | `mode timing\|fast` | switch the simulation mode (state is carried over by the `panda_kernel` facade). |
//!
//! ## Values
//!
//! Values for `set` use [`Value::parse`] at the pin's width: decimal `5`, hex `0x3f`,
//! binary `0b10x1` (may contain `x`/`z`), or a lone `x` / `z` filling every bit. Literals shorter
//! than the width are zero-extended; wider ones are an error.
//!
//! Expected values for `expect` accept the same forms, plus `-` as a **don't-care** bit in binary
//! literals (`0b1-0-`) and a lone `-` meaning "anything". `x` and `z` are **exact**: an expected
//! `x` bit passes only if the bit is `X`, and `z` only if it is `Z`. Binary literals shorter than
//! the signal are zero-extended (`0b1` on a 4-bit signal means `0b0001`).
//!
//! ## Example
//!
//! ```text
//! # NAND SR latch
//! set S_n=1 R_n=1
//! settle
//! expect Q=x Q_n=x      # never set: unknown
//! set S_n=0
//! settle
//! expect Q=1 Q_n=0
//! ```
//!
//! # Modes
//!
//! A file without `mode` lines runs on a plain [`TimingSim`] ([`Program::run`]). A file with
//! `mode` lines (or `panda test --mode fast`) runs on a [`panda_kernel::Kernel`]
//! ([`Program::run_kernel`]), starting in Timing mode unless the runner says otherwise. `mode fast`
//! compiles the design (synchronously) and continues in Fast mode (SPEC §6): time advances in
//! whole phases, so in Fast mode
//!
//! * `run N` ends at the end of the phase containing the target tick. Targets accumulate from
//!   power-on exactly as in Timing mode (`run 14` then `run 36` targets tick 50, whatever phase
//!   end the first `run` stopped at), so a file written for Timing mode samples the same phases;
//! * `cycle N` is the natural step for clocked designs (identical sample points in both modes,
//!   also counted from power-on: the first `cycle` ends one tick before the first rising edge);
//! * `settle` processes pending input changes; on a design without running clocks it also lets
//!   every remaining source change (a `PowerOnReset` release) happen. It fails only if the design
//!   does not settle within the bound (largest `settle` argument of the file);
//! * input changes are their own event after the last processed clock edge.
//! * `reset` powers on again and ends at the end of the first phase (Fast mode has no
//!   mid-phase sample points).
//!
//! On the kernel runner in Timing mode, `settle` on a design with running clocks advances `MAX`
//! ticks and fails (as on the plain runner); without running clocks it settles phase by phase.
//!
//! # Running
//!
//! [`Program::parse`] checks syntax and resolves every name and literal against the netlist up
//! front, reporting **all** errors with line numbers. [`Program::run`] executes it on a
//! [`TimingSim`] and returns a [`Report`] with one [`Check`] per expectation (and failures for
//! `settle`/`pulse` that did not settle); [`Report::passed`] is false if any check failed.
//! Simulator diagnostics (metastability, oscillation, contention) are collected with the tick and
//! line at which they were taken.

use std::fmt;
use std::io::{self, Write};

use panda_kernel::{Kernel, Mode, StepUnit};
use panda_netlist::flat::FlatNetlist;
use panda_netlist::{Diagnostic, Logic, Value};
use panda_timing::TimingSim;

use crate::clock::{self, RefClock};
use crate::fastmode;
use crate::signals::{Signal, resolve};
use crate::vcd::{self, VcdWriter};

/// Default limit for `settle` without an argument (and for each half of a `pulse`).
pub const DEFAULT_SETTLE_MAX: u64 = 100_000;

/// An error in a vector file, with its 1-based line number.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VectorError {
    /// 1-based line number.
    pub line: usize,
    /// What is wrong.
    pub message: String,
}

impl fmt::Display for VectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

/// An expected value: per bit (LSB first) `Some(l)` = must equal `l`, `None` = don't care.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expected {
    bits: Vec<Option<Logic>>,
}

impl Expected {
    /// Parse an expected value of `width` bits (see the module docs).
    pub fn parse(width: u32, s: &str) -> Result<Expected, String> {
        let t: String = s.trim().chars().filter(|&c| c != '_').collect();
        if t == "-" {
            return Ok(Expected { bits: vec![None; width as usize] });
        }
        let lower = t.to_ascii_lowercase();
        if let Some(bin) = lower.strip_prefix("0b") {
            if bin.is_empty() {
                return Err(format!("empty binary literal '{s}'"));
            }
            if bin.len() as u32 > width {
                return Err(format!("binary literal '{s}' wider than {width} bits"));
            }
            let mut bits = vec![Some(Logic::Zero); width as usize];
            for (i, c) in bin.chars().rev().enumerate() {
                bits[i] = match c {
                    '-' => None,
                    c => Some(Logic::from_char(c).ok_or_else(|| format!("bad binary digit '{c}' in '{s}'"))?),
                };
            }
            return Ok(Expected { bits });
        }
        let v = Value::parse(width, &t)?;
        Ok(Expected { bits: v.bits().map(Some).collect() })
    }

    /// Width in bits.
    pub fn width(&self) -> u32 {
        self.bits.len() as u32
    }

    /// True if `actual` matches (same width, every cared-about bit equal).
    pub fn matches(&self, actual: &Value) -> bool {
        actual.width() == self.width()
            && self.bits.iter().enumerate().all(|(i, e)| e.is_none_or(|l| actual.get(i as u32) == l))
    }
}

impl fmt::Display for Expected {
    /// Binary MSB first with `-` for don't-care bits, e.g. `0b1-x0`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0b")?;
        for b in self.bits.iter().rev() {
            write!(f, "{}", b.map_or('-', Logic::to_char))?;
        }
        Ok(())
    }
}

/// A resolved command.
#[derive(Clone, Debug)]
pub enum Command {
    /// Set top-level inputs.
    Set(Vec<(String, Value)>),
    /// Advance exactly N ticks.
    Run(u64),
    /// Settle with a tick limit.
    Settle(u64),
    /// Pulse a 1-bit input N times.
    Pulse {
        /// Input pin name.
        pin: String,
        /// Number of pulses.
        count: u64,
    },
    /// Check signals.
    Expect(Vec<(Signal, Expected)>),
    /// Print signals.
    Print(Vec<Signal>),
    /// Power-on reset.
    Reset,
    /// Advance N cycles of the reference clock.
    Cycle(u64),
    /// Switch the simulation mode.
    Mode(Mode),
}

/// A command with its source line.
#[derive(Clone, Debug)]
pub struct Step {
    /// 1-based line number.
    pub line: usize,
    /// The command.
    pub command: Command,
}

/// The result of one expectation (or of a `settle`/`pulse` that may fail).
#[derive(Clone, Debug)]
pub struct Check {
    /// 1-based line number.
    pub line: usize,
    /// Tick at which the check ran.
    pub tick: u64,
    /// What was checked (signal name, or `settle` / `pulse <pin>`).
    pub what: String,
    /// Expected value (rendered), or `quiescent`.
    pub expected: String,
    /// Actual value (rendered), or `still active`.
    pub actual: String,
    /// Whether it passed.
    pub passed: bool,
}

/// One `print` output line.
#[derive(Clone, Debug)]
pub struct Printed {
    /// 1-based line number.
    pub line: usize,
    /// Tick at which it was printed.
    pub tick: u64,
    /// Signal name.
    pub name: String,
    /// Value at that time.
    pub value: Value,
}

/// A simulator diagnostic, with when it was observed.
#[derive(Clone, Debug)]
pub struct TimedDiagnostic {
    /// Line of the command during which it was raised.
    pub line: usize,
    /// Simulator tick when it was collected (end of that command).
    pub tick: u64,
    /// The diagnostic.
    pub diagnostic: Diagnostic,
}

/// Outcome of running a [`Program`].
#[derive(Clone, Debug, Default)]
pub struct Report {
    /// Every check in execution order.
    pub checks: Vec<Check>,
    /// Output of `print` commands.
    pub printed: Vec<Printed>,
    /// Diagnostics raised by the simulator.
    pub diagnostics: Vec<TimedDiagnostic>,
}

impl Report {
    /// True if every check passed.
    pub fn passed(&self) -> bool {
        self.checks.iter().all(|c| c.passed)
    }

    /// Number of failed checks.
    pub fn failures(&self) -> usize {
        self.checks.iter().filter(|c| !c.passed).count()
    }

    /// Human-readable report. With `verbose`, passing checks are listed too.
    pub fn write(&self, w: &mut dyn Write, source_name: &str, verbose: bool) -> io::Result<()> {
        // Commands run in line order, so a stable sort by line interleaves the three kinds of
        // entries in execution order.
        let mut lines: Vec<(usize, String)> = Vec::new();
        for p in &self.printed {
            lines.push((p.line, format!("{source_name}:{}: [t={}] {} = {}", p.line, p.tick, p.name, crate::format_value(&p.value))));
        }
        for c in self.checks.iter().filter(|c| verbose || !c.passed) {
            let status = if c.passed { "ok  " } else { "FAIL" };
            lines.push((
                c.line,
                format!("{status} {source_name}:{}: [t={}] {}: expected {}, got {}", c.line, c.tick, c.what, c.expected, c.actual),
            ));
        }
        for d in &self.diagnostics {
            lines.push((d.line, format!("note {source_name}:{}: [t={}] {}", d.line, d.tick, d.diagnostic)));
        }
        lines.sort_by_key(|(l, _)| *l);
        for (_, text) in lines {
            writeln!(w, "{text}")?;
        }
        let total = self.checks.len();
        let failed = self.failures();
        writeln!(
            w,
            "{}: {} of {total} checks passed{}",
            if failed == 0 { "PASS" } else { "FAIL" },
            total - failed,
            if failed == 0 { String::new() } else { format!(", {failed} failed") }
        )
    }
}

/// A parsed and resolved vector file.
#[derive(Clone, Debug)]
pub struct Program {
    /// The commands in order.
    pub steps: Vec<Step>,
    /// Reference clock of `cycle` commands (default: the first running `Clock` of the netlist;
    /// may be replaced before running, e.g. by `panda test --clock`).
    pub clock: Option<RefClock>,
}

fn parse_count(arg: Option<&str>, default: u64, what: &str) -> Result<u64, String> {
    match arg {
        None => Ok(default),
        Some(a) => a.replace('_', "").parse::<u64>().map_err(|_| format!("{what}: expected a tick count, got '{a}'")),
    }
}

fn split_assign(item: &str) -> Result<(&str, &str), String> {
    match item.split_once('=') {
        Some((n, v)) if !n.trim().is_empty() && !v.trim().is_empty() => Ok((n.trim(), v.trim())),
        _ => Err(format!("expected NAME=VALUE, got '{item}'")),
    }
}

fn input_width(nl: &FlatNetlist, pin: &str) -> Result<u32, String> {
    nl.top_inputs
        .iter()
        .find(|p| p.name == pin)
        .map(|p| p.nets.len() as u32)
        .ok_or_else(|| format!("'{pin}' is not a top-level input pin"))
}

fn parse_line(nl: &FlatNetlist, clock: Option<&RefClock>, words: &[&str]) -> Result<Command, Vec<String>> {
    let (kw, args) = (words[0], &words[1..]);
    let one = |e: String| vec![e];
    let max_args = |n: usize| {
        if args.len() > n { Err(one(format!("'{kw}' takes at most {n} argument(s), got {}", args.len()))) } else { Ok(()) }
    };
    match kw {
        "set" | "expect" => {
            if args.is_empty() {
                return Err(one(format!("'{kw}' needs at least one NAME=VALUE")));
            }
            let mut errs = Vec::new();
            let mut sets = Vec::new();
            let mut exps = Vec::new();
            for item in args {
                let r = split_assign(item).and_then(|(name, val)| {
                    if kw == "set" {
                        let w = input_width(nl, name)?;
                        let v = Value::parse(w, val).map_err(|e| format!("{name}: {e}"))?;
                        sets.push((name.to_string(), v));
                    } else {
                        let s = resolve(nl, name)?;
                        let e = Expected::parse(s.width(), val).map_err(|e| format!("{name}: {e}"))?;
                        exps.push((s, e));
                    }
                    Ok(())
                });
                if let Err(e) = r {
                    errs.push(e);
                }
            }
            if !errs.is_empty() {
                return Err(errs);
            }
            Ok(if kw == "set" { Command::Set(sets) } else { Command::Expect(exps) })
        }
        "run" | "tick" => {
            max_args(1)?;
            Ok(Command::Run(parse_count(args.first().copied(), 1, kw).map_err(one)?))
        }
        "settle" => {
            max_args(1)?;
            Ok(Command::Settle(parse_count(args.first().copied(), DEFAULT_SETTLE_MAX, kw).map_err(one)?))
        }
        "pulse" => {
            max_args(2)?;
            let pin = args.first().ok_or_else(|| one("'pulse' needs a pin name".into()))?;
            let w = input_width(nl, pin).map_err(one)?;
            if w != 1 {
                return Err(one(format!("'pulse' needs a 1-bit input, '{pin}' has width {w}")));
            }
            let count = parse_count(args.get(1).copied(), 1, kw).map_err(one)?;
            Ok(Command::Pulse { pin: pin.to_string(), count })
        }
        "print" => {
            if args.is_empty() {
                return Err(one("'print' needs at least one signal name".into()));
            }
            let mut errs = Vec::new();
            let mut sigs = Vec::new();
            for a in args {
                match resolve(nl, a) {
                    Ok(s) => sigs.push(s),
                    Err(e) => errs.push(e),
                }
            }
            if errs.is_empty() { Ok(Command::Print(sigs)) } else { Err(errs) }
        }
        "reset" => {
            max_args(0)?;
            Ok(Command::Reset)
        }
        "cycle" => {
            max_args(1)?;
            if clock.is_none() {
                return Err(one("'cycle' needs a design with a running Clock".into()));
            }
            Ok(Command::Cycle(parse_count(args.first().copied(), 1, kw).map_err(one)?))
        }
        "mode" => {
            max_args(1)?;
            let m = args.first().ok_or_else(|| one("'mode' needs 'timing' or 'fast'".into()))?;
            Ok(Command::Mode(
                fastmode::parse_mode(m).ok_or_else(|| one(format!("unknown mode '{m}' (expected timing or fast)")))?,
            ))
        }
        _ => Err(one(format!(
            "unknown command '{kw}' (expected set, run, tick, settle, pulse, cycle, expect, print, reset, mode)"
        ))),
    }
}

impl Program {
    /// Parse `src` and resolve it against `nl`. Returns every error found.
    pub fn parse(src: &str, nl: &FlatNetlist) -> Result<Program, Vec<VectorError>> {
        let mut steps = Vec::new();
        let mut errors = Vec::new();
        let clock = RefClock::find(nl, None).ok().flatten();
        for (i, raw) in src.lines().enumerate() {
            let line = i + 1;
            let text = raw.split('#').next().unwrap_or("");
            let words: Vec<&str> = text.split_whitespace().collect();
            if words.is_empty() {
                continue;
            }
            match parse_line(nl, clock.as_ref(), &words) {
                Ok(command) => steps.push(Step { line, command }),
                Err(es) => errors.extend(es.into_iter().map(|message| VectorError { line, message })),
            }
        }
        if errors.is_empty() { Ok(Program { steps, clock }) } else { Err(errors) }
    }

    /// Number of `expect` checks in the program.
    pub fn expect_count(&self) -> usize {
        self.steps
            .iter()
            .map(|s| if let Command::Expect(e) = &s.command { e.len() } else { 0 })
            .sum()
    }

    /// True if the program has `mode` lines (so it needs [`Program::run_kernel`]).
    pub fn uses_modes(&self) -> bool {
        self.steps.iter().any(|s| matches!(s.command, Command::Mode(_)))
    }

    /// The largest `settle` bound of the program (at least the default), used as the kernel
    /// runner's settle bound.
    pub fn max_settle(&self) -> u64 {
        self.steps
            .iter()
            .filter_map(|s| if let Command::Settle(m) = s.command { Some(m) } else { None })
            .fold(DEFAULT_SETTLE_MAX, u64::max)
    }

    /// Run on `sim` (which is used as-is: it is *not* reset first). `mode timing` lines are
    /// no-ops; a `mode fast` line is a failed check (Fast mode needs [`Program::run_kernel`]).
    pub fn run(&self, sim: &mut TimingSim) -> Report {
        self.run_inner::<io::Sink>(sim, None).expect("no I/O without a VCD writer")
    }

    /// Run on `sim`, sampling every simulated tick into `vcd`.
    pub fn run_traced<W: Write>(&self, sim: &mut TimingSim, vcd: &mut VcdWriter<W>) -> io::Result<Report> {
        self.run_inner(sim, Some(vcd))
    }

    fn run_inner<W: Write>(&self, sim: &mut TimingSim, mut vcd: Option<&mut VcdWriter<W>>) -> io::Result<Report> {
        let mut report = Report::default();
        if let Some(w) = vcd.as_deref_mut() {
            w.sample(sim)?;
        }
        for step in &self.steps {
            let line = step.line;
            let settle_check = |report: &mut Report, what: String, ok: bool, tick: u64| {
                report.checks.push(Check {
                    line,
                    tick,
                    what,
                    expected: "quiescent".into(),
                    actual: if ok { "quiescent".into() } else { "still active".into() },
                    passed: ok,
                });
            };
            match &step.command {
                Command::Set(items) => {
                    for (pin, v) in items {
                        sim.set_input(pin, v).expect("pin and width checked at parse time");
                    }
                }
                Command::Run(n) => vcd::advance(sim, *n, vcd.as_deref_mut())?,
                Command::Settle(max) => {
                    let ok = vcd::settle(sim, *max, vcd.as_deref_mut())?;
                    if !ok {
                        settle_check(&mut report, format!("settle {max}"), false, sim.tick());
                    }
                }
                Command::Pulse { pin, count } => {
                    let (one, zero) = (Value::from_bool(true), Value::from_bool(false));
                    for _ in 0..*count {
                        let mut ok = true;
                        for v in [&one, &zero] {
                            sim.set_input(pin, v).expect("pin checked at parse time");
                            ok &= vcd::settle(sim, DEFAULT_SETTLE_MAX, vcd.as_deref_mut())?;
                        }
                        if !ok {
                            settle_check(&mut report, format!("pulse {pin}"), false, sim.tick());
                            break;
                        }
                    }
                }
                Command::Expect(items) => {
                    for (sig, exp) in items {
                        let actual = gather(sim, sig);
                        report.checks.push(Check {
                            line,
                            tick: sim.tick(),
                            what: sig.name.clone(),
                            expected: exp.to_string(),
                            actual: actual.to_string(),
                            passed: exp.matches(&actual),
                        });
                    }
                }
                Command::Print(sigs) => {
                    for sig in sigs {
                        report.printed.push(Printed {
                            line,
                            tick: sim.tick(),
                            name: sig.name.clone(),
                            value: gather(sim, sig),
                        });
                    }
                }
                Command::Reset => {
                    sim.reset();
                    if let Some(w) = vcd.as_deref_mut() {
                        w.sample(sim)?;
                    }
                }
                Command::Cycle(n) => {
                    let clk = self.clock.as_ref().expect("checked at parse time");
                    clock::run_cycles(sim, clk, *n, vcd.as_deref_mut())?;
                }
                Command::Mode(Mode::Timing) => {}
                Command::Mode(Mode::Fast) => report.checks.push(Check {
                    line,
                    tick: sim.tick(),
                    what: "mode fast".into(),
                    expected: "Fast mode".into(),
                    actual: "not supported by the Timing-only runner".into(),
                    passed: false,
                }),
            }
            for d in sim.take_diagnostics() {
                report.diagnostics.push(TimedDiagnostic { line, tick: sim.tick(), diagnostic: d });
            }
        }
        Ok(report)
    }
}

impl Program {
    /// Run on a loaded `kernel` (used as-is, in its current mode), switching modes at `mode`
    /// lines (see the module docs, "Modes"). A kernel passed in Fast mode is assumed to be freshly
    /// powered on (its program time starts at tick 0, whatever phase end the engine is at). With `vcd`, values are sampled after every tick in
    /// Timing mode (`run`, `cycle`), after every phase in Fast mode, and after each other command.
    pub fn run_kernel<W: Write>(&self, kernel: &mut Kernel, mut vcd: Option<&mut VcdWriter<W>>) -> io::Result<Report> {
        let mut r = KernelRun { k: kernel, clock: self.clock.as_ref(), report: Report::default(), logical: 0 };
        // A kernel in Fast mode is at the end of a phase; the program's time starts at the
        // power-on tick (0) it stands for (see the module docs, "Modes").
        r.logical = if r.k.mode() == Mode::Fast { 0 } else { r.k.tick() };
        r.sample(vcd.as_deref_mut())?;
        for step in &self.steps {
            let line = step.line;
            if r.k.mode() == Mode::Timing {
                r.logical = r.k.tick();
            }
            match &step.command {
                Command::Set(items) => {
                    for (pin, v) in items {
                        r.k.set_input(pin, v).expect("pin and width checked at parse time");
                    }
                }
                Command::Run(n) => r.advance(Advance::Ticks(*n), vcd.as_deref_mut())?,
                Command::Cycle(n) => r.advance(Advance::Cycles(*n), vcd.as_deref_mut())?,
                Command::Settle(max) => {
                    if !r.settle(*max, vcd.as_deref_mut())? {
                        r.failed_settle(line, format!("settle {max}"));
                    }
                }
                Command::Pulse { pin, count } => {
                    let (one, zero) = (Value::from_bool(true), Value::from_bool(false));
                    for _ in 0..*count {
                        let mut ok = true;
                        for v in [&one, &zero] {
                            r.k.set_input(pin, v).expect("pin checked at parse time");
                            ok &= r.settle(DEFAULT_SETTLE_MAX, vcd.as_deref_mut())?;
                        }
                        if !ok {
                            r.failed_settle(line, format!("pulse {pin}"));
                            break;
                        }
                    }
                }
                Command::Expect(items) => {
                    let values = fastmode::values(r.k);
                    for (sig, exp) in items {
                        let actual = gather_values(&values, sig);
                        r.report.checks.push(Check {
                            line,
                            tick: r.k.tick(),
                            what: sig.name.clone(),
                            expected: exp.to_string(),
                            actual: actual.to_string(),
                            passed: exp.matches(&actual),
                        });
                    }
                }
                Command::Print(sigs) => {
                    let values = fastmode::values(r.k);
                    for sig in sigs {
                        r.report.printed.push(Printed {
                            line,
                            tick: r.k.tick(),
                            name: sig.name.clone(),
                            value: gather_values(&values, sig),
                        });
                    }
                }
                Command::Reset => {
                    let design = r.k.design().cloned().expect("kernel has a design");
                    r.k.load(design).expect("the loaded design reloads");
                    if r.k.mode() == Mode::Fast {
                        r.k.wait_ready().ok();
                    }
                    r.logical = 0;
                    r.sample(vcd.as_deref_mut())?;
                }
                Command::Mode(m) => {
                    if let Err(e) = fastmode::enter_mode(r.k, *m) {
                        r.report.checks.push(Check {
                            line,
                            tick: r.k.tick(),
                            what: format!("mode {}", if *m == Mode::Fast { "fast" } else { "timing" }),
                            expected: "ready".into(),
                            actual: e,
                            passed: false,
                        });
                    }
                    r.sample(vcd.as_deref_mut())?;
                }
            }
            let tick = r.k.tick();
            for d in r.k.diagnostics().to_vec() {
                r.report.diagnostics.push(TimedDiagnostic { line, tick, diagnostic: d });
            }
            r.k.clear_diagnostics();
        }
        Ok(r.report)
    }
}

/// How far [`KernelRun::advance`] goes.
#[derive(Clone, Copy, Debug)]
enum Advance {
    Ticks(u64),
    Cycles(u64),
}

/// State of one [`Program::run_kernel`].
struct KernelRun<'a> {
    k: &'a mut Kernel,
    clock: Option<&'a RefClock>,
    report: Report,
    /// The tick the program has asked for so far: Timing mode reaches it exactly, Fast mode
    /// ends at the end of the phase containing it (so `run`/`cycle` targets accumulate the same
    /// way in both modes, counted from power-on).
    logical: u64,
}

impl KernelRun<'_> {
    fn sample<W: Write>(&self, vcd: Option<&mut VcdWriter<W>>) -> io::Result<()> {
        match vcd {
            Some(w) => w.sample_values(self.k.tick(), &fastmode::values(self.k)),
            None => Ok(()),
        }
    }

    fn failed_settle(&mut self, line: usize, what: String) {
        self.report.checks.push(Check {
            line,
            tick: self.k.tick(),
            what,
            expected: "quiescent".into(),
            actual: "still active".into(),
            passed: false,
        });
    }

    fn advance<W: Write>(&mut self, how: Advance, vcd: Option<&mut VcdWriter<W>>) -> io::Result<()> {
        let base = self.logical;
        let target = match how {
            Advance::Ticks(n) => base.saturating_add(n),
            Advance::Cycles(n) => self.clock.expect("checked at parse time").cycles_end(base, n),
        };
        self.logical = target;
        // Apply pending input changes even when no time passes.
        self.k.step(0, StepUnit::Tick);
        let Some(w) = vcd else {
            self.k.run_to(target);
            return Ok(());
        };
        while self.k.tick() < target {
            let before = self.k.tick();
            let unit = if self.k.mode() == Mode::Timing { StepUnit::Tick } else { StepUnit::Phase };
            self.k.step(1, unit);
            if self.k.tick() == before {
                break;
            }
            w.sample_values(self.k.tick(), &fastmode::values(self.k))?;
        }
        Ok(())
    }

    /// See the module docs ("Modes"). Returns true if settled.
    fn settle<W: Write>(&mut self, max: u64, mut vcd: Option<&mut VcdWriter<W>>) -> io::Result<bool> {
        if self.clock.is_some() {
            if self.k.mode() == Mode::Fast {
                self.k.step(0, StepUnit::Tick);
                self.sample(vcd)?;
                return Ok(true);
            }
            self.advance(Advance::Ticks(max), vcd)?;
            return Ok(false);
        }
        let start = self.k.tick();
        let settled = loop {
            let before = self.k.tick();
            self.k.step(1, StepUnit::Phase);
            self.sample(vcd.as_deref_mut())?;
            if self.k.tick() == before {
                break true;
            }
            if self.k.tick() - start >= max {
                break false;
            }
        };
        self.logical = self.logical.max(self.k.tick());
        Ok(settled)
    }
}

/// Value of a signal from a full net-value array.
pub fn gather_values(values: &[Logic], sig: &Signal) -> Value {
    Value::from_bits(&sig.nets.iter().map(|&n| values[n as usize]).collect::<Vec<_>>())
}

/// Current value of a signal.
pub fn gather(sim: &TimingSim, sig: &Signal) -> Value {
    Value::from_bits(&sig.nets.iter().map(|&n| sim.get(n)).collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_parsing() {
        let e = Expected::parse(4, "0b1-x0").unwrap();
        assert_eq!(e.to_string(), "0b1-x0");
        assert!(e.matches(&Value::parse(4, "0b11x0").unwrap()));
        assert!(e.matches(&Value::parse(4, "0b10x0").unwrap()));
        assert!(!e.matches(&Value::parse(4, "0b1100").unwrap()), "x is exact");
        assert!(!e.matches(&Value::parse(4, "0b11z0").unwrap()), "x does not match z");
        assert!(!e.matches(&Value::parse(3, "0b1x0").unwrap()), "width must match");

        let e = Expected::parse(4, "0b1").unwrap();
        assert_eq!(e.to_string(), "0b0001");
        let e = Expected::parse(8, "0x3f").unwrap();
        assert!(e.matches(&Value::from_u64(8, 0x3f)));
        let e = Expected::parse(3, "-").unwrap();
        assert!(e.matches(&Value::xs(3)));
        assert!(e.matches(&Value::zeros(3)));
        let e = Expected::parse(2, "x").unwrap();
        assert!(e.matches(&Value::xs(2)));
        assert!(!e.matches(&Value::zs(2)));
        let e = Expected::parse(1, "z").unwrap();
        assert!(e.matches(&Value::zs(1)));

        assert!(Expected::parse(2, "0b101").is_err());
        assert!(Expected::parse(2, "0b").is_err());
        assert!(Expected::parse(2, "0b1q").is_err());
        assert!(Expected::parse(2, "4").is_err());
        assert!(Expected::parse(2, "zz").is_err());
    }
}
