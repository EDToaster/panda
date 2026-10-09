//! Command-line entry points: argument parsing and the `run`, `test`, `info`, `flat` and
//! `example` commands. See [`USAGE`].
//!
//! Every command writes its normal output to `out` and warnings / errors to `err`, so they can be
//! driven from tests without spawning a process.

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use panda_netlist::flat::{FlatNetlist, FlatOp, SourceKind};
use panda_netlist::validate::validate;
use panda_netlist::{Atom, Design, Diagnostic, InstKind, Severity, Value};
use panda_timing::{TimingOptions, TimingSim};

use crate::signals::{self, Signal};
use crate::vcd::{self, VcdWriter};
use crate::vectors::{self, Program};
use crate::{CliError, flatten_design, format_value, load_design, read_file, write_diagnostics};

/// Help text.
pub const USAGE: &str = "\
panda — gate-level digital logic simulator (Timing mode)

USAGE:
    panda run <design.json> [--ticks N] [--max-ticks N] [--set PIN=VALUE]... [--watch NAMES]...
                            [--vcd FILE] [--trace NAMES]... [--p2-limit N] [--osc-limit N]
    panda test <design.json> <vectors.txt> [--vcd FILE] [--trace NAMES]... [-v|--verbose]
                            [--p2-limit N] [--osc-limit N]
    panda info <design.json>
    panda flat <design.json> [--json]
    panda example [NAME]
    panda help | --help | -h
    panda --version

COMMANDS:
    run      Simulate. Without --ticks, run until quiescent (at most --max-ticks, default 100000).
             --set applies top-level input values before the first tick. Prints the watched
             signals at the end (default: top-level outputs) and simulator diagnostics.
    test     Run a test-vector file (format: see the panda_cli::vectors docs). Exit code 1 if any
             check fails.
    info     Defs, instance counts, flat gate/net counts, sources, validation diagnostics.
    flat     Flat-netlist statistics; --json dumps nets, pins, gates, sources and resolvers.
    example  List the built-in example designs, or print one as JSON.

NAMES are comma-separated signal names: top-level pins, flat net names (\"cnt/ff0/m\", \"c[2]\"),
bus base names (\"carry\") or bit ranges (\"carry[3:0]\"). VCD files use 1 tick = 1ns and dump
--trace signals (default: top-level pins and named top-level nets).

EXIT CODES: 0 ok, 1 failure, 2 usage error.
";

/// Parsed command-line arguments of one subcommand.
#[derive(Debug, Default)]
struct Args {
    positional: Vec<String>,
    /// `(option name without dashes, value)` in order.
    values: Vec<(String, String)>,
    flags: HashSet<String>,
}

impl Args {
    /// Parse `args`. `valued` options take a value (`--opt V` or `--opt=V`); `flags` do not.
    /// Short aliases are given as `("v", "verbose")` pairs.
    fn parse(args: &[String], valued: &[&str], flags: &[&str], short: &[(&str, &str)]) -> Result<Args, CliError> {
        let mut out = Args::default();
        let mut it = args.iter();
        let mut only_positional = false;
        while let Some(a) = it.next() {
            if only_positional || a == "-" || !a.starts_with('-') {
                out.positional.push(a.clone());
                continue;
            }
            if a == "--" {
                only_positional = true;
                continue;
            }
            let body = if let Some(long) = a.strip_prefix("--") {
                long.to_string()
            } else {
                let s = &a[1..];
                short
                    .iter()
                    .find(|(k, _)| *k == s)
                    .map(|(_, long)| long.to_string())
                    .ok_or_else(|| CliError::Usage(format!("unknown option '{a}'")))?
            };
            let (name, inline) = match body.split_once('=') {
                Some((n, v)) => (n.to_string(), Some(v.to_string())),
                None => (body, None),
            };
            if valued.contains(&name.as_str()) {
                let v = match inline {
                    Some(v) => v,
                    None => it.next().cloned().ok_or_else(|| CliError::Usage(format!("option '--{name}' needs a value")))?,
                };
                out.values.push((name, v));
            } else if flags.contains(&name.as_str()) {
                if inline.is_some() {
                    return Err(CliError::Usage(format!("option '--{name}' takes no value")));
                }
                out.flags.insert(name);
            } else {
                return Err(CliError::Usage(format!("unknown option '{a}'")));
            }
        }
        Ok(out)
    }

    fn all(&self, name: &str) -> impl Iterator<Item = &str> {
        self.values.iter().filter(move |(n, _)| n == name).map(|(_, v)| v.as_str())
    }

    fn last(&self, name: &str) -> Option<&str> {
        self.all(name).last()
    }

    fn u64(&self, name: &str) -> Result<Option<u64>, CliError> {
        self.last(name)
            .map(|v| {
                v.replace('_', "")
                    .parse::<u64>()
                    .map_err(|_| CliError::Usage(format!("--{name}: expected a non-negative integer, got '{v}'")))
            })
            .transpose()
    }

    fn flag(&self, name: &str) -> bool {
        self.flags.contains(name)
    }

    /// Exactly `names.len()` positionals.
    fn positionals(&self, cmd: &str, names: &[&str]) -> Result<&[String], CliError> {
        if self.positional.len() != names.len() {
            let want = names.iter().map(|n| format!("<{n}>")).collect::<Vec<_>>().join(" ");
            return Err(CliError::Usage(format!(
                "'panda {cmd}' expects {want}, got {} argument(s) (see 'panda help')",
                self.positional.len()
            )));
        }
        Ok(&self.positional)
    }
}

/// Run the CLI with `args` (excluding the program name). Returns the process exit code.
pub fn main_with_args(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    match dispatch(args, out, err) {
        Ok(code) => code,
        Err(e) => {
            let _ = writeln!(err, "error: {e}");
            if matches!(e, CliError::Usage(_)) {
                let _ = writeln!(err, "run 'panda help' for usage");
            }
            e.exit_code()
        }
    }
}

fn io_err(path: &str) -> impl Fn(std::io::Error) -> CliError + '_ {
    move |error| CliError::Io { path: path.to_string(), error }
}

fn out_err(error: std::io::Error) -> CliError {
    CliError::Io { path: "<output>".into(), error }
}

fn dispatch(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> Result<i32, CliError> {
    let Some((cmd, rest)) = args.split_first() else {
        write!(err, "{USAGE}").map_err(out_err)?;
        return Ok(2);
    };
    match cmd.as_str() {
        "help" | "--help" | "-h" => {
            write!(out, "{USAGE}").map_err(out_err)?;
            Ok(0)
        }
        "--version" | "-V" => {
            writeln!(out, "panda {}", env!("CARGO_PKG_VERSION")).map_err(out_err)?;
            Ok(0)
        }
        "run" => cmd_run(rest, out, err),
        "test" => cmd_test(rest, out, err),
        "info" => cmd_info(rest, out, err),
        "flat" => cmd_flat(rest, out, err),
        "example" => cmd_example(rest, out),
        other => Err(CliError::Usage(format!("unknown command '{other}'"))),
    }
}

/// Load, validate and flatten; print warnings to `err`.
fn load_netlist(path: &str, err: &mut dyn Write) -> Result<(Design, Arc<FlatNetlist>), CliError> {
    let design = load_design(Path::new(path))?;
    let (nl, warnings) = flatten_design(&design)?;
    write_diagnostics(err, &format!("{path}: "), &warnings).map_err(out_err)?;
    Ok((design, Arc::new(nl)))
}

fn timing_options(a: &Args) -> Result<TimingOptions, CliError> {
    let mut o = TimingOptions::default();
    if let Some(p) = a.u64("p2-limit")? {
        o.p2_limit = u32::try_from(p).map_err(|_| CliError::Usage("--p2-limit too large".into()))?;
    }
    if let Some(l) = a.u64("osc-limit")? {
        o.osc_limit = l;
    }
    Ok(o)
}

/// Signals for `--trace` / `--watch` lists (may be repeated), or `default` if none given.
fn signal_lists(
    nl: &FlatNetlist,
    a: &Args,
    opt: &str,
    default: impl FnOnce() -> Vec<Signal>,
) -> Result<Vec<Signal>, CliError> {
    let mut sigs = Vec::new();
    let mut any = false;
    for list in a.all(opt) {
        any = true;
        sigs.extend(signals::resolve_list(nl, list).map_err(|e| CliError::Usage(format!("--{opt}: {e}")))?);
    }
    Ok(if any { sigs } else { default() })
}

fn open_vcd(path: &str, design: &Design, nl: &FlatNetlist, sigs: Vec<Signal>) -> Result<VcdWriter<BufWriter<File>>, CliError> {
    let f = File::create(path).map_err(io_err(path))?;
    VcdWriter::new(BufWriter::new(f), nl, &design.top, sigs).map_err(io_err(path))
}

fn cmd_run(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> Result<i32, CliError> {
    let a = Args::parse(
        args,
        &["ticks", "max-ticks", "set", "watch", "vcd", "trace", "p2-limit", "osc-limit"],
        &[],
        &[],
    )?;
    let path = a.positionals("run", &["design.json"])?[0].clone();
    let ticks = a.u64("ticks")?;
    let max_ticks = a.u64("max-ticks")?.unwrap_or(vectors::DEFAULT_SETTLE_MAX);
    let opts = timing_options(&a)?;
    let (design, nl) = load_netlist(&path, err)?;

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
    let watch = signal_lists(&nl, &a, "watch", || {
        let outs: Vec<Signal> = nl
            .top_outputs
            .iter()
            .filter(|p| !p.nets.is_empty())
            .map(|p| Signal { name: p.name.clone(), nets: p.nets.clone() })
            .collect();
        if outs.is_empty() { signals::default_selection(&nl) } else { outs }
    })?;
    let mut vcd_writer = match a.last("vcd") {
        Some(vpath) => {
            let sigs = signal_lists(&nl, &a, "trace", || signals::default_selection(&nl))?;
            Some((vpath.to_string(), open_vcd(vpath, &design, &nl, sigs)?))
        }
        None => None,
    };

    let mut sim = TimingSim::new(nl.clone(), opts);
    for (pin, v) in &sets {
        sim.set_input(pin, v).map_err(CliError::Other)?;
    }
    // Inputs given with --set are part of the power-on state.
    if !sets.is_empty() {
        sim.reset();
    }
    let vpath = vcd_writer.as_ref().map(|(p, _)| p.clone()).unwrap_or_default();
    if let Some((_, w)) = vcd_writer.as_mut() {
        w.sample(&sim).map_err(io_err(&vpath))?;
    }
    let w = vcd_writer.as_mut().map(|(_, w)| w);
    let status = if let Some(n) = ticks {
        vcd::advance(&mut sim, n, w).map_err(io_err(&vpath))?;
        format!("ran {n} ticks{}", if sim.is_quiescent() { " (quiescent)" } else { "" })
    } else if vcd::settle(&mut sim, max_ticks, w).map_err(io_err(&vpath))? {
        "settled".to_string()
    } else {
        format!("not quiescent after {max_ticks} ticks")
    };
    if let Some((p, w)) = vcd_writer {
        w.finish(&sim).map_err(io_err(&p))?;
    }

    writeln!(out, "tick {}: {status}", sim.tick()).map_err(out_err)?;
    let width = watch.iter().map(|s| s.name.len()).max().unwrap_or(0);
    for s in &watch {
        writeln!(out, "  {:<width$} = {}", s.name, format_value(&vectors::gather(&sim, s))).map_err(out_err)?;
    }
    let diags = sim.take_diagnostics();
    write_diagnostics(out, "note: ", &diags).map_err(out_err)?;
    if sim.suppressed_diagnostics() > 0 {
        writeln!(out, "note: {} further diagnostics suppressed", sim.suppressed_diagnostics()).map_err(out_err)?;
    }
    Ok(0)
}

fn cmd_test(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> Result<i32, CliError> {
    let a = Args::parse(args, &["vcd", "trace", "p2-limit", "osc-limit"], &["verbose"], &[("v", "verbose")])?;
    let pos = a.positionals("test", &["design.json", "vectors.txt"])?;
    let (path, vpath) = (pos[0].clone(), pos[1].clone());
    let opts = timing_options(&a)?;
    let (design, nl) = load_netlist(&path, err)?;
    let src = read_file(Path::new(&vpath))?;
    let program = match Program::parse(&src, &nl) {
        Ok(p) => p,
        Err(errors) => {
            for e in &errors {
                writeln!(err, "{vpath}:{}: error: {}", e.line, e.message).map_err(out_err)?;
            }
            writeln!(err, "{vpath}: {} error(s); nothing was run", errors.len()).map_err(out_err)?;
            return Ok(1);
        }
    };
    let mut sim = TimingSim::new(nl.clone(), opts);
    let report = match a.last("vcd") {
        Some(dump) => {
            let sigs = signal_lists(&nl, &a, "trace", || signals::default_selection(&nl))?;
            let mut w = open_vcd(dump, &design, &nl, sigs)?;
            let r = program.run_traced(&mut sim, &mut w).map_err(io_err(dump))?;
            w.finish(&sim).map_err(io_err(dump))?;
            r
        }
        None => program.run(&mut sim),
    };
    report.write(out, &vpath, a.flag("verbose")).map_err(out_err)?;
    Ok(if report.passed() { 0 } else { 1 })
}

fn atom_kind(a: &Atom) -> String {
    match a {
        Atom::Gate { op, reduce, .. } => {
            format!("{}{}", format!("{op:?}").to_lowercase(), if *reduce { "-reduce" } else { "" })
        }
        Atom::Not { .. } => "not".into(),
        Atom::Buf { .. } => "buf".into(),
        Atom::Tri { .. } => "tri".into(),
        Atom::Splitter { .. } => "splitter".into(),
        Atom::Constant { .. } => "constant".into(),
        Atom::Clock { .. } => "clock".into(),
        Atom::PowerOnReset { .. } => "power-on-reset".into(),
    }
}

fn op_name(op: FlatOp) -> &'static str {
    match op {
        FlatOp::And => "and",
        FlatOp::Or => "or",
        FlatOp::Nand => "nand",
        FlatOp::Nor => "nor",
        FlatOp::Xor => "xor",
        FlatOp::Xnor => "xnor",
        FlatOp::Not => "not",
        FlatOp::Buf => "buf",
        FlatOp::Tri => "tri",
    }
}

fn counts_str(m: &BTreeMap<String, usize>) -> String {
    if m.is_empty() {
        return "none".into();
    }
    m.iter().map(|(k, v)| format!("{k} {v}")).collect::<Vec<_>>().join(", ")
}

fn source_counts(nl: &FlatNetlist) -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    for s in &nl.sources {
        let k = match s.kind {
            SourceKind::Constant(_) => "constant",
            SourceKind::Clock { .. } => "clock",
            SourceKind::PowerOnReset { .. } => "power-on-reset",
            SourceKind::TopInput { .. } => "top-input",
        };
        *m.entry(k.to_string()).or_default() += 1;
    }
    m
}

fn gate_counts(nl: &FlatNetlist) -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    for g in &nl.gates {
        *m.entry(op_name(g.op).to_string()).or_default() += 1;
    }
    m
}

fn write_flat_stats(out: &mut dyn Write, nl: &FlatNetlist) -> std::io::Result<()> {
    let depth = nl
        .instances
        .iter()
        .map(|i| if i.path.is_empty() { 0 } else { i.path.matches('/').count() + 1 })
        .max()
        .unwrap_or(0);
    writeln!(out, "flat netlist:")?;
    writeln!(out, "  nets:      {}", nl.num_nets)?;
    writeln!(out, "  gates:     {} ({})", nl.gates.len(), counts_str(&gate_counts(nl)))?;
    writeln!(out, "  groups:    {}", nl.groups.len())?;
    writeln!(out, "  sources:   {} ({})", nl.sources.len(), counts_str(&source_counts(nl)))?;
    writeln!(out, "  resolvers: {}", nl.resolvers.len())?;
    writeln!(out, "  instances: {} (max depth {depth})", nl.instances.len())?;
    let pins = |ps: &[panda_netlist::flat::FlatPin]| {
        if ps.is_empty() {
            "none".to_string()
        } else {
            ps.iter().map(|p| format!("{}[{}]", p.name, p.nets.len())).collect::<Vec<_>>().join(", ")
        }
    };
    writeln!(out, "  top inputs:  {}", pins(&nl.top_inputs))?;
    writeln!(out, "  top outputs: {}", pins(&nl.top_outputs))?;
    Ok(())
}

fn cmd_info(args: &[String], out: &mut dyn Write, _err: &mut dyn Write) -> Result<i32, CliError> {
    let a = Args::parse(args, &[], &[], &[])?;
    let path = a.positionals("info", &["design.json"])?[0].clone();
    let design = load_design(Path::new(&path))?;
    let diags = validate(&design);
    info_report(out, &path, &design, &diags).map_err(out_err)?;
    let errors = diags.iter().any(|d| d.severity == Severity::Error);
    if errors {
        return Ok(1);
    }
    match panda_netlist::flatten(&design) {
        Ok(nl) => {
            write_flat_stats(out, &nl).map_err(out_err)?;
            // Expanded instance count per def.
            let mut per_def: BTreeMap<String, usize> = BTreeMap::new();
            for i in &nl.instances {
                *per_def.entry(i.def.clone()).or_default() += 1;
            }
            writeln!(out, "  expanded instances per def: {}", counts_str(&per_def)).map_err(out_err)?;
            Ok(0)
        }
        Err(diags) => {
            writeln!(out, "flattening failed:").map_err(out_err)?;
            write_diagnostics(out, "  ", &diags).map_err(out_err)?;
            Ok(1)
        }
    }
}

fn info_report(out: &mut dyn Write, path: &str, design: &Design, diags: &[Diagnostic]) -> std::io::Result<()> {
    writeln!(out, "design: {path}")?;
    writeln!(out, "format: {}", design.format)?;
    writeln!(out, "top:    {}", design.top)?;
    writeln!(out, "defs:   {}", design.defs.len())?;
    for d in &design.defs {
        let marker = if d.id == design.top { " (top)" } else { "" };
        let title = if d.name.is_empty() || d.name == d.id { String::new() } else { format!(" \"{}\"", d.name) };
        writeln!(out, "  {}{title}{marker}", d.id)?;
        let pins: Vec<String> =
            d.pins.iter().map(|p| format!("{}:{}[{}]", format!("{:?}", p.dir).to_lowercase(), p.name, p.width)).collect();
        writeln!(out, "    pins:      {}", if pins.is_empty() { "none".into() } else { pins.join(" ") })?;
        let mut atoms = BTreeMap::new();
        let mut subs = BTreeMap::new();
        for i in &d.instances {
            match &i.kind {
                InstKind::Atom(a) => *atoms.entry(atom_kind(a)).or_default() += 1,
                InstKind::Def(id) => *subs.entry(id.clone()).or_default() += 1,
            }
        }
        writeln!(out, "    instances: {} (atoms: {}; defs: {})", d.instances.len(), counts_str(&atoms), counts_str(&subs))?;
        writeln!(out, "    nets:      {}", d.nets.len())?;
    }
    if diags.is_empty() {
        writeln!(out, "diagnostics: none")?;
    } else {
        writeln!(out, "diagnostics: {}", diags.len())?;
        write_diagnostics(out, "  ", diags)?;
    }
    Ok(())
}

fn flat_json(nl: &FlatNetlist) -> serde_json::Value {
    use serde_json::json;
    let pins = |ps: &[panda_netlist::flat::FlatPin]| -> Vec<serde_json::Value> {
        ps.iter().map(|p| json!({"name": p.name, "nets": p.nets})).collect()
    };
    let gates: Vec<_> = nl
        .gates
        .iter()
        .map(|g| {
            json!({
                "op": op_name(g.op),
                "inputs": g.inputs,
                "output": g.output,
                "path": nl.groups.get(g.group as usize).map(|gr| gr.path.as_str()),
                "bit": g.bit,
            })
        })
        .collect();
    let sources: Vec<_> = nl
        .sources
        .iter()
        .map(|s| {
            let kind = match &s.kind {
                SourceKind::Constant(l) => json!({"constant": l.to_char().to_string()}),
                SourceKind::Clock { period, high, phase } => {
                    json!({"clock": {"period": period, "high": high, "phase": phase}})
                }
                SourceKind::PowerOnReset { ticks } => json!({"power_on_reset": {"ticks": ticks}}),
                SourceKind::TopInput { pin, bit } => json!({"top_input": {"pin": pin, "bit": bit}}),
            };
            json!({"kind": kind, "output": s.output})
        })
        .collect();
    let resolvers: Vec<_> = nl
        .resolvers
        .iter()
        .map(|r| json!({"drivers": r.drivers, "termination": format!("{:?}", r.termination).to_lowercase(), "output": r.output}))
        .collect();
    let instances: Vec<_> = nl.instances.iter().map(|i| json!({"path": i.path, "def": i.def})).collect();
    json!({
        "num_nets": nl.num_nets,
        "net_names": nl.net_names,
        "top_inputs": pins(&nl.top_inputs),
        "top_outputs": pins(&nl.top_outputs),
        "gates": gates,
        "sources": sources,
        "resolvers": resolvers,
        "instances": instances,
    })
}

fn cmd_flat(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> Result<i32, CliError> {
    let a = Args::parse(args, &[], &["json"], &[])?;
    let path = a.positionals("flat", &["design.json"])?[0].clone();
    let (_, nl) = load_netlist(&path, err)?;
    if a.flag("json") {
        let s = serde_json::to_string_pretty(&flat_json(&nl)).map_err(|e| CliError::Other(e.to_string()))?;
        writeln!(out, "{s}").map_err(out_err)?;
    } else {
        write_flat_stats(out, &nl).map_err(out_err)?;
    }
    Ok(0)
}

fn cmd_example(args: &[String], out: &mut dyn Write) -> Result<i32, CliError> {
    let a = Args::parse(args, &[], &[], &[])?;
    match a.positional.as_slice() {
        [] => {
            for n in crate::examples::NAMES {
                writeln!(out, "{n}").map_err(out_err)?;
            }
            Ok(0)
        }
        [name] => {
            let d = crate::examples::by_name(name).ok_or_else(|| {
                CliError::Usage(format!("unknown example '{name}' (available: {})", crate::examples::NAMES.join(", ")))
            })?;
            writeln!(out, "{}", d.to_json()).map_err(out_err)?;
            Ok(0)
        }
        _ => Err(CliError::Usage("'panda example' takes at most one name".into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn arg_parsing() {
        let a = Args::parse(
            &s(&["d.json", "--ticks", "5", "--set=A=1", "--set", "B=2", "-v", "--", "--x"]),
            &["ticks", "set"],
            &["verbose"],
            &[("v", "verbose")],
        )
        .unwrap();
        assert_eq!(a.positional, s(&["d.json", "--x"]));
        assert_eq!(a.u64("ticks").unwrap(), Some(5));
        assert_eq!(a.all("set").collect::<Vec<_>>(), ["A=1", "B=2"]);
        assert!(a.flag("verbose"));
        assert!(Args::parse(&s(&["--nope"]), &[], &[], &[]).is_err());
        assert!(Args::parse(&s(&["--ticks"]), &["ticks"], &[], &[]).is_err());
        assert!(Args::parse(&s(&["--json=1"]), &[], &["json"], &[]).is_err());
        let a = Args::parse(&s(&["--ticks", "abc"]), &["ticks"], &[], &[]).unwrap();
        assert!(matches!(a.u64("ticks"), Err(CliError::Usage(_))));
    }
}
