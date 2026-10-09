//! `panda sta`, `panda classify` and `panda kernels`: the analyses of the kernel compiler.
//!
//! * `sta` — static timing analysis (SPEC §9) through [`panda_kernel::Kernel::timing_report`]:
//!   per-clock minimum periods and suggested periods, settle time per analyzed event signature,
//!   the critical path and the STA diagnostics.
//! * `classify` — package classification (SPEC §4.4) of every def (or one def) through
//!   [`panda_kernel::Kernel::classify`].
//! * `kernels` — kernel extraction and characterization statistics (SPEC §8.3–§8.4) from
//!   [`panda_compile::compile_kernels`]; optionally the compiled FastSim program's statistics.

use std::collections::{BTreeMap, HashSet};
use std::io::{self, Write};
use std::sync::Arc;
use std::time::Instant;

use panda_compile::{KernelKind, KernelOptions as CompileOptions, KernelSet, KernelTable};
use panda_fast::{FastSim, FastSimOptions};
use panda_kernel::{ClassKind, Classification, Edge, Kernel, KernelSummary, TimingSummary};
use panda_netlist::flat::FlatNetlist;
use panda_netlist::{Logic, Severity};
use panda_sta::PhaseSettle;

use crate::CliError;
use crate::args::Args;
use crate::commands::{load_netlist, out_err, timing_options};
use crate::fastmode::{self, KernelSettings, dur};

fn kind_name(k: KernelKind) -> &'static str {
    match k {
        KernelKind::Table => "table",
        KernelKind::Sim => "sim",
        KernelKind::Keeper => "keeper",
    }
}

fn class_name(k: ClassKind) -> &'static str {
    match k {
        ClassKind::Combinational => "combinational",
        ClassKind::Sequential => "sequential",
        ClassKind::Oscillator => "oscillator",
    }
}

fn kernel_for(design: &panda_netlist::Design, a: &Args) -> Result<Kernel, CliError> {
    let settings = KernelSettings {
        timing: timing_options(a)?,
        backend: Default::default(),
        cycle_clock: None,
        max_settle_ticks: crate::vectors::DEFAULT_SETTLE_MAX,
    };
    fastmode::new_kernel(design, &settings, &[])
}

// ---------------------------------------------------------------------------------------------
// sta
// ---------------------------------------------------------------------------------------------

/// `panda sta`.
pub(crate) fn cmd_sta(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> Result<i32, CliError> {
    let a = Args::parse(args, &["p2-limit", "osc-limit"], &["strict"], &[])?;
    let path = a.positionals("sta", &["design.json"])?[0].clone();
    let (design, _) = load_netlist(&path, err)?;
    let mut k = kernel_for(&design, &a)?;
    let t0 = Instant::now();
    let summary = k.timing_report();
    let elapsed = t0.elapsed();
    write_sta(out, &path, &summary).map_err(out_err)?;
    writeln!(out, "analysis time: {}", dur(elapsed)).map_err(out_err)?;
    Ok(if a.flag("strict") && !summary.is_clean() { 1 } else { 0 })
}

fn clocks_str(clocks: &[(String, Logic)]) -> String {
    if clocks.is_empty() {
        return "-".into();
    }
    clocks.iter().map(|(n, v)| format!("{n}={}", v.to_char())).collect::<Vec<_>>().join(",")
}

fn opt_ticks(v: Option<u32>) -> String {
    match v {
        Some(u32::MAX) | None => "never".into(),
        Some(t) => t.to_string(),
    }
}

/// Human-readable STA report.
pub fn write_sta(out: &mut dyn Write, path: &str, s: &TimingSummary) -> io::Result<()> {
    let r = &s.report;
    writeln!(out, "static timing analysis: {path}")?;
    if s.clocks.is_empty() {
        writeln!(out, "clocks: none")?;
    } else {
        writeln!(out, "clocks:")?;
        for c in &s.clocks {
            writeln!(
                out,
                "  {}: period {}, min period {}, suggested {}: {}",
                c.net,
                c.period,
                opt_ticks(c.min_period),
                c.suggested_period.map_or("-".into(), |p| p.to_string()),
                if c.meets_timing { "ok" } else { "TOO FAST" }
            )?;
        }
    }
    if !r.min_period.is_empty() {
        writeln!(out, "min periods:")?;
        for (net, p) in &r.min_period {
            writeln!(out, "  {net}: {}", opt_ticks(Some(*p)))?;
        }
    }
    writeln!(out, "phase settle ({} event signature(s)):", r.phase_settle.len())?;
    write_phase_table(out, &r.phase_settle)?;
    if r.critical_path.is_empty() {
        writeln!(out, "critical path: none")?;
    } else {
        writeln!(out, "critical path ({} nets):", r.critical_path.len())?;
        for (i, n) in r.critical_path.iter().enumerate() {
            writeln!(out, "  {i:>3}  {n}")?;
        }
    }
    let diags: Vec<_> = r.diagnostics.iter().chain(&s.warnings).collect();
    if diags.is_empty() {
        writeln!(out, "diagnostics: none")?;
    } else {
        writeln!(out, "diagnostics: {}", diags.len())?;
        for d in &diags {
            writeln!(out, "  {d}")?;
        }
    }
    let errors = diags.iter().filter(|d| d.severity == Severity::Error).count();
    if s.is_clean() {
        writeln!(out, "result: clean (Fast mode equals Timing mode at phase boundaries)")?;
    } else {
        writeln!(
            out,
            "result: NOT clean ({errors} error(s){})",
            if s.clocks.iter().any(|c| !c.meets_timing) { ", clock(s) below the minimum period" } else { "" }
        )?;
    }
    Ok(())
}

fn write_phase_table(out: &mut dyn Write, rows: &[PhaseSettle]) -> io::Result<()> {
    let ev_w = rows.iter().map(|p| p.event.chars().count()).max().unwrap_or(5).max(5);
    let ck_w = rows.iter().map(|p| clocks_str(&p.clocks).len()).max().unwrap_or(6).max(6);
    writeln!(out, "  {:<ev_w$}  {:<ck_w$}  {:>6}  {:>6}  {:>6}  last change", "event", "clocks", "settle", "gap", "rounds")?;
    for p in rows {
        let pad = ev_w - p.event.chars().count();
        writeln!(
            out,
            "  {}{}  {:<ck_w$}  {:>6}  {:>6}  {:>6}  {}",
            p.event,
            " ".repeat(pad),
            clocks_str(&p.clocks),
            opt_ticks(p.settle),
            p.gap.map_or("-".into(), |g| g.to_string()),
            p.rounds,
            p.critical_path.last().map_or("-", String::as_str)
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// classify
// ---------------------------------------------------------------------------------------------

/// `panda classify`.
pub(crate) fn cmd_classify(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> Result<i32, CliError> {
    let a = Args::parse(args, &["p2-limit", "osc-limit"], &["verbose"], &[("v", "verbose")])?;
    let pos = a.positionals_between("classify", 1, 2, "<design.json> [def]")?;
    let path = pos[0].clone();
    let only = pos.get(1).cloned();
    let (design, _) = load_netlist(&path, err)?;
    let defs: Vec<String> = match &only {
        Some(d) => {
            if design.def(d).is_none() {
                let ids: Vec<&str> = design.defs.iter().map(|d| d.id.as_str()).collect();
                return Err(CliError::Usage(format!("unknown def '{d}' (defs: {})", ids.join(", "))));
            }
            vec![d.clone()]
        }
        None => design.defs.iter().map(|d| d.id.clone()).collect(),
    };
    let mut k = kernel_for(&design, &a)?;
    let mut failed = false;
    for def in &defs {
        let t0 = Instant::now();
        match k.classify(def) {
            Ok(c) => write_classification(out, &c, def == &design.top, a.flag("verbose"), t0.elapsed())
                .map_err(out_err)?,
            Err(diags) => {
                failed = true;
                writeln!(out, "{def}: cannot classify:").map_err(out_err)?;
                crate::write_diagnostics(out, "  ", &diags).map_err(out_err)?;
            }
        }
    }
    Ok(if failed { 1 } else { 0 })
}

/// Group key of a kernel summary for the compact listing.
type KernelGroup = (&'static str, usize, usize, usize, Option<usize>, u32, bool, bool);

fn group_key(s: &KernelSummary) -> KernelGroup {
    (
        kind_name(s.kind),
        s.gates,
        s.inputs,
        s.state_nets,
        s.stable_states,
        s.state_bits,
        s.oscillator,
        s.ternary_combinational,
    )
}

/// Human-readable classification of one def.
pub fn write_classification(
    out: &mut dyn Write,
    c: &Classification,
    top: bool,
    verbose: bool,
    elapsed: std::time::Duration,
) -> io::Result<()> {
    writeln!(out, "{}{}: {}", c.def, if top { " (top)" } else { "" }, class_name(c.kind))?;
    writeln!(
        out,
        "  cyclic: {}; state bits: {}; kernels: {} (analysis {})",
        if c.cyclic { "yes" } else { "no" },
        c.state_bits,
        c.kernels.len(),
        dur(elapsed)
    )?;
    if !c.kernels.is_empty() {
        writeln!(
            out,
            "  {:>5}  {:<6}  {:>5}  {:>6}  {:>5}  {:>6}  {:>4}  {:<10}  location",
            "count", "kind", "gates", "inputs", "nets", "states", "bits", "notes"
        )?;
        let rows: Vec<(usize, &KernelSummary)> = if verbose {
            c.kernels.iter().map(|s| (1, s)).collect()
        } else {
            // Identical summaries collapse into one row (first location as the example).
            let mut order: Vec<KernelGroup> = Vec::new();
            let mut groups: BTreeMap<KernelGroup, (usize, &KernelSummary)> = BTreeMap::new();
            for s in &c.kernels {
                let key = group_key(s);
                groups
                    .entry(key)
                    .and_modify(|e| e.0 += 1)
                    .or_insert_with(|| {
                        order.push(key);
                        (1, s)
                    });
            }
            order.iter().map(|k| groups[k]).collect()
        };
        for (n, s) in rows {
            let mut notes = Vec::new();
            if s.oscillator {
                notes.push("osc");
            }
            if s.ternary_combinational {
                notes.push("comb");
            }
            writeln!(
                out,
                "  {:>5}  {:<6}  {:>5}  {:>6}  {:>5}  {:>6}  {:>4}  {:<10}  {}{}",
                n,
                kind_name(s.kind),
                s.gates,
                s.inputs,
                s.state_nets,
                s.stable_states.map_or("-".into(), |v| v.to_string()),
                s.state_bits,
                if notes.is_empty() { "-".to_string() } else { notes.join(",") },
                s.location,
                if n > 1 { " (e.g.)" } else { "" }
            )?;
        }
    }
    if c.clock_analysis {
        writeln!(
            out,
            "  clock-like pins: {}",
            if c.clock_like_pins.is_empty() { "none".into() } else { c.clock_like_pins.join(", ") }
        )?;
        match &c.edge_triggered {
            Some((pin, e)) => {
                writeln!(out, "  edge-triggered: {pin} {}", if *e == Edge::Rising { "rising" } else { "falling" })?
            }
            None => writeln!(out, "  edge-triggered: no")?,
        }
    } else if c.kind == ClassKind::Sequential {
        writeln!(out, "  clock-like pins: not analyzed (time-varying sources, too many inputs or states)")?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------------------------

/// `panda kernels`.
pub(crate) fn cmd_kernels(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> Result<i32, CliError> {
    let a = Args::parse(args, &["limit"], &["list", "program"], &[])?;
    let path = a.positionals("kernels", &["design.json"])?[0].clone();
    let limit = a.u64("limit")?.map(|l| l as usize);
    let (_, nl) = load_netlist(&path, err)?;
    let t0 = Instant::now();
    let (ks, tables) = panda_compile::compile_kernels(&nl, &CompileOptions::default());
    let elapsed = t0.elapsed();
    write_kernel_stats(out, &path, &nl, &ks, &tables, elapsed).map_err(out_err)?;
    if a.flag("list") || limit.is_some() {
        write_kernel_list(out, &nl, &ks, &tables, limit.unwrap_or(usize::MAX)).map_err(out_err)?;
    }
    if a.flag("program") {
        let t0 = Instant::now();
        let sim = FastSim::from_parts(nl.clone(), FastSimOptions::default(), ks, tables).map_err(CliError::Design)?;
        let elapsed = t0.elapsed();
        let p = sim.program().stats();
        writeln!(out, "FastSim program (compiled in {}):", dur(elapsed)).map_err(out_err)?;
        let lines = [
            format!("  ops:         {} (gate2 {}, gateN {}, resolve {}; notifying {})", p.ops, p.gate2, p.gate_n, p.resolve, p.notifying),
            format!("  partitions:  {} (largest {} ops)", p.partitions, p.max_partition),
            format!("  LUTs:        {} (absorbing {} gates, {} entries)", p.luts, p.lut_gates, p.lut_entries),
            format!("  kernels:     {} ({} with a table op, {} distinct tables)", p.kernels, p.table_kernels, p.ktables),
            format!(
                "  phase specialization: {} kernels; (kernel, clock tuple) pairs: opaque {}, transparent {}, general {}",
                p.spec_kernels, p.phase_classes[0], p.phase_classes[1], p.phase_classes[2]
            ),
        ];
        for l in lines {
            writeln!(out, "{l}").map_err(out_err)?;
        }
    }
    Ok(0)
}

fn hist_str(m: &BTreeMap<usize, usize>) -> String {
    if m.is_empty() {
        return "none".into();
    }
    m.iter().map(|(k, v)| format!("{k}:{v}")).collect::<Vec<_>>().join(" ")
}

/// Kernel extraction / characterization statistics.
pub fn write_kernel_stats(
    out: &mut dyn Write,
    path: &str,
    nl: &FlatNetlist,
    ks: &KernelSet,
    tables: &[Option<Arc<KernelTable>>],
    elapsed: std::time::Duration,
) -> io::Result<()> {
    let h = ks.histogram();
    let kernel_gates: usize = ks.kernels.iter().map(|k| k.gates.len()).sum();
    let kernel_res: usize = ks.kernels.iter().map(|k| k.resolvers.len()).sum();
    writeln!(out, "kernels: {path} (extracted and characterized in {})", dur(elapsed))?;
    writeln!(out, "  netlist:    {} gates, {} resolvers, {} nets", nl.gates.len(), nl.resolvers.len(), nl.num_nets)?;
    writeln!(out, "  kernels:    {} (table {}, sim {}, keeper {})", ks.kernels.len(), h.table, h.sim, h.keeper)?;
    writeln!(
        out,
        "  in kernels: {kernel_gates} gates, {kernel_res} resolvers; combinational: {} nodes",
        ks.comb_order.len()
    )?;
    let state_nets: usize = ks.kernels.iter().map(|k| k.nets.len()).sum();
    writeln!(out, "  state nets: {state_nets}")?;
    let sizes: BTreeMap<usize, usize> = ks.size_histogram().into_iter().collect();
    writeln!(out, "  size histogram (nodes:count): {}", hist_str(&sizes))?;
    let mut inputs: BTreeMap<usize, usize> = BTreeMap::new();
    let mut bits: BTreeMap<usize, usize> = BTreeMap::new();
    for k in &ks.kernels {
        *inputs.entry(k.inputs.len()).or_default() += 1;
        if k.kind != KernelKind::Sim {
            *bits.entry(k.bits()).or_default() += 1;
        }
    }
    writeln!(out, "  input histogram (inputs:count): {}", hist_str(&inputs))?;
    writeln!(out, "  inputs+cut histogram, table/keeper kernels (bits:count): {}", hist_str(&bits))?;

    // Tables: identical kernels share one table (same Arc).
    let mut seen: HashSet<*const KernelTable> = HashSet::new();
    let (mut distinct, mut states, mut entries, mut nonbinary, mut order_sensitive, mut osc) = (0, 0, 0, 0, 0, 0);
    let mut max_latency = 0;
    let mut table_states: BTreeMap<usize, usize> = BTreeMap::new();
    let mut with_table = 0;
    for t in tables.iter().flatten() {
        with_table += 1;
        if !seen.insert(Arc::as_ptr(t)) {
            continue;
        }
        distinct += 1;
        states += t.num_states();
        *table_states.entry(t.num_states()).or_default() += 1;
        entries += t.transitions.len();
        nonbinary += t.transitions.iter().filter(|tr| tr.next.is_none()).count();
        order_sensitive += t.transitions.iter().filter(|tr| tr.order_sensitive).count();
        max_latency = max_latency.max(t.transitions.iter().map(|tr| tr.latency).max().unwrap_or(0));
        if t.is_oscillator() {
            osc += 1;
        }
    }
    let uncharacterized = ks.kernels.iter().zip(tables).filter(|(k, t)| k.kind == KernelKind::Table && t.is_none()).count();
    writeln!(
        out,
        "  tables:     {with_table} kernels with a table, {distinct} distinct ({} shared){}",
        with_table - distinct,
        if uncharacterized > 0 { format!(", {uncharacterized} table kernel(s) too large (simulated)") } else { String::new() }
    )?;
    writeln!(out, "  table stable states histogram (states:tables): {}", hist_str(&table_states))?;
    writeln!(
        out,
        "  table entries: {entries} ({states} states total; {nonbinary} non-binary, {order_sensitive} order-sensitive; max latency {max_latency})"
    )?;
    writeln!(out, "  oscillator tables: {osc}")?;
    Ok(())
}

fn write_kernel_list(
    out: &mut dyn Write,
    nl: &FlatNetlist,
    ks: &KernelSet,
    tables: &[Option<Arc<KernelTable>>],
    limit: usize,
) -> io::Result<()> {
    writeln!(
        out,
        "  {:>5}  {:<6}  {:>5}  {:>4}  {:>6}  {:>7}  {:>4}  {:>4}  {:>6}  location",
        "id", "kind", "gates", "res", "inputs", "outputs", "nets", "cut", "states"
    )?;
    for (k, t) in ks.kernels.iter().zip(tables).take(limit) {
        let loc = k.outputs.first().or(k.nets.first()).map_or("-", |&n| nl.net_names[n as usize].as_str());
        writeln!(
            out,
            "  {:>5}  {:<6}  {:>5}  {:>4}  {:>6}  {:>7}  {:>4}  {:>4}  {:>6}  {loc}",
            k.id,
            kind_name(k.kind),
            k.gates.len(),
            k.resolvers.len(),
            k.inputs.len(),
            k.outputs.len(),
            k.nets.len(),
            k.cut.len(),
            t.as_ref().map_or("-".into(), |t| t.num_states().to_string()),
        )?;
    }
    if ks.kernels.len() > limit {
        writeln!(out, "  ... {} more (use --limit N)", ks.kernels.len() - limit)?;
    }
    Ok(())
}
