//! Command-line entry points: dispatch, the usage text, and the `info`, `flat` and `example`
//! commands (the others live in their own modules). See [`USAGE`].
//!
//! Every command writes its normal output to `out` and warnings / errors to `err`, so they can be
//! driven from tests without spawning a process.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use panda_netlist::flat::{FlatNetlist, FlatOp, SourceKind};
use panda_netlist::validate::validate;
use panda_netlist::{Atom, Design, Diagnostic, InstKind, Severity};
use panda_timing::TimingOptions;

use crate::args::Args;
use crate::signals::{self, Signal};
use crate::vcd::VcdWriter;
use crate::{CliError, flatten_design, load_design, write_diagnostics};

/// Help text.
pub const USAGE: &str = "\
panda — gate-level digital logic simulator (Timing mode, Fast mode, STA, generators)

USAGE:
    panda run <design.json> [--mode timing|fast] [--backend jit|compiled|reference]
                            [--ticks N | --cycles N] [--clock NET] [--max-ticks N] [--set PIN=VALUE]...
                            [--watch NAMES]... [--vcd FILE] [--trace NAMES]... [--stats]
                            [--p2-limit N] [--osc-limit N]
    panda test <design.json> <vectors.txt> [--mode timing|fast] [--backend jit|compiled|reference]
                            [--clock NET] [--vcd FILE] [--trace NAMES]... [-v|--verbose]
                            [--p2-limit N] [--osc-limit N]
    panda sta <design.json> [--strict]
    panda classify <design.json> [DEF] [-v|--verbose]
    panda kernels <design.json> [--list] [--limit N] [--program]
    panda gen [NAME [OPTIONS] [-o FILE]]
    panda bench <design.json> [--cycles N | --seconds S] [--engines LIST] [--clock NET] [--set PIN=VALUE]...
    panda info <design.json>
    panda flat <design.json> [--json]
    panda example [NAME]
    panda help | --help | -h
    panda --version

COMMANDS:
    run       Simulate. Timing mode (default) is the unit-delay reference engine; --mode fast runs
              the kernel-delta engine through the panda-kernel facade, compiling first. --backend:
              jit (the default) = the FastSim interpreter with the Cranelift JIT installed once its
              native code is compiled; compiled = the FastSim interpreter; reference = FastRef.
              --ticks N advances N ticks (Fast mode: to the end of the phase containing the
              target); --cycles N advances N cycles of the reference clock (--clock, default the
              first running Clock) to the tick before a rising edge; otherwise run until quiescent
              (at most --max-ticks, default 100000). --set applies top-level input values at
              power-on. Prints the watched signals (default: top-level outputs), diagnostics, and
              with --cycles, --stats or Fast mode the throughput (ticks/s, cycles/s; Fast mode
              also the compile time).
    test      Run a test-vector file (format: see the panda_cli::vectors docs). Directives include
              'cycle N' (clocked designs) and 'mode timing|fast' (switch modes mid-file, state is
              carried over). --mode fast starts in Fast mode. Exit code 1 if any check fails.
    sta       Static timing analysis: per-clock minimum and suggested periods, settle time per
              event signature, the critical path and STA diagnostics. --strict: exit code 1 unless
              clean.
    classify  Classify every def (or DEF): combinational / sequential / oscillator, state bits,
              kernels (identical ones grouped; -v lists each), clock-like pins, edge-triggering.
    kernels   Kernel extraction and characterization statistics: kinds, size / input histograms,
              table sharing, states and entries. --list (or --limit N) lists kernels; --program
              also compiles the FastSim program and prints its statistics.
    gen       Generate a design as JSON (stdout, or -o FILE). Without NAME: list generators.
              Blocks: counter, register, regfile, rom FILE, ring-osc, sr-latch, d-latch, dff,
              dff-rst, dff-arst, dff-en, toggle-ff, half-adder, full-adder, adder, incrementer, mux,
              decoder, equality, bus-port, transceiver. Clocked blocks get a testbench (Clock on
              CLK, PowerOnReset on RST; --period P (64), --high, --phase, --reset-ticks) unless
              --bare. Panda-8: gen cpu8 [--program fib|countdown|multiply|bench | --asm FILE |
              --image HEXFILE] [--variant hierarchical|flat|gated] [--rom sop|mux|decoder]
              [--period P] [--reset-cycles N]. --flat inlines the hierarchy.
    bench     Cycle throughput of the engines (timing, fastref, fastsim, jit = FastSim with the
              Cranelift JIT, kernel = the facade with its default backend; default the first four)
              for --seconds S each (default 1), or exactly --cycles N from power-on with the
              top-level outputs compared (exit code 1 on a mismatch).
    info      Defs, instance counts, flat gate/net counts, sources, validation diagnostics.
    flat      Flat-netlist statistics; --json dumps nets, pins, gates, sources and resolvers.
    example   List the built-in example designs, or print one as JSON.

NAMES are comma-separated signal names: top-level pins, flat net names (\"cnt/ff0/m\", \"c[2]\"),
bus base names (\"carry\") or bit ranges (\"carry[3:0]\"). VCD files use 1 tick = 1ns and dump
--trace signals (default: top-level pins and named top-level nets); in Fast mode they are sampled
at phase ends.

EXIT CODES: 0 ok, 1 failure, 2 usage error.
";

/// Run the CLI with `args` (excluding the program name). Returns the process exit code.
///
/// A closed output pipe (`EPIPE`, e.g. `panda info x.json | head`) ends the command quietly with
/// exit code 0: the reader has seen all it wanted.
pub fn main_with_args(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    match dispatch(args, out, err) {
        Ok(code) => code,
        Err(e) if e.is_broken_pipe() => 0,
        Err(e) => {
            let _ = writeln!(err, "error: {e}");
            if matches!(e, CliError::Usage(_)) {
                let _ = writeln!(err, "run 'panda help' for usage");
            }
            e.exit_code()
        }
    }
}

pub(crate) fn io_err(path: &str) -> impl Fn(std::io::Error) -> CliError + '_ {
    move |error| CliError::Io { path: path.to_string(), error }
}

pub(crate) fn out_err(error: std::io::Error) -> CliError {
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
        "run" => crate::run::cmd_run(rest, out, err),
        "test" => crate::run::cmd_test(rest, out, err),
        "sta" => crate::analysis::cmd_sta(rest, out, err),
        "classify" => crate::analysis::cmd_classify(rest, out, err),
        "kernels" => crate::analysis::cmd_kernels(rest, out, err),
        "gen" => crate::generators::cmd_gen(rest, out, err),
        "bench" => crate::bench::cmd_bench(rest, out, err),
        "info" => cmd_info(rest, out, err),
        "flat" => cmd_flat(rest, out, err),
        "example" => cmd_example(rest, out),
        other => Err(CliError::Usage(format!("unknown command '{other}'"))),
    }
}

/// Load, validate and flatten; print warnings to `err`.
pub(crate) fn load_netlist(path: &str, err: &mut dyn Write) -> Result<(Design, Arc<FlatNetlist>), CliError> {
    let design = load_design(Path::new(path))?;
    let (nl, warnings) = flatten_design(&design)?;
    write_diagnostics(err, &format!("{path}: "), &warnings).map_err(out_err)?;
    Ok((design, Arc::new(nl)))
}

pub(crate) fn timing_options(a: &Args) -> Result<TimingOptions, CliError> {
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
pub(crate) fn signal_lists(
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

pub(crate) fn open_vcd(path: &str, design: &Design, nl: &FlatNetlist, sigs: Vec<Signal>) -> Result<VcdWriter<BufWriter<File>>, CliError> {
    let f = File::create(path).map_err(io_err(path))?;
    VcdWriter::new(BufWriter::new(f), nl, &design.top, sigs).map_err(io_err(path))
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
