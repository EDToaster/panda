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

use panda_netlist::flat::FlatNetlist;
use panda_netlist::{Diagnostic, Logic, Value};
use panda_timing::TimingSim;

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

fn parse_line(nl: &FlatNetlist, words: &[&str]) -> Result<Command, Vec<String>> {
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
        _ => Err(one(format!("unknown command '{kw}' (expected set, run, tick, settle, pulse, expect, print, reset)"))),
    }
}

impl Program {
    /// Parse `src` and resolve it against `nl`. Returns every error found.
    pub fn parse(src: &str, nl: &FlatNetlist) -> Result<Program, Vec<VectorError>> {
        let mut steps = Vec::new();
        let mut errors = Vec::new();
        for (i, raw) in src.lines().enumerate() {
            let line = i + 1;
            let text = raw.split('#').next().unwrap_or("");
            let words: Vec<&str> = text.split_whitespace().collect();
            if words.is_empty() {
                continue;
            }
            match parse_line(nl, &words) {
                Ok(command) => steps.push(Step { line, command }),
                Err(es) => errors.extend(es.into_iter().map(|message| VectorError { line, message })),
            }
        }
        if errors.is_empty() { Ok(Program { steps }) } else { Err(errors) }
    }

    /// Number of `expect` checks in the program.
    pub fn expect_count(&self) -> usize {
        self.steps
            .iter()
            .map(|s| if let Command::Expect(e) = &s.command { e.len() } else { 0 })
            .sum()
    }

    /// Run on `sim` (which is used as-is: it is *not* reset first).
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
            }
            for d in sim.take_diagnostics() {
                report.diagnostics.push(TimedDiagnostic { line, tick: sim.tick(), diagnostic: d });
            }
        }
        Ok(report)
    }
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
