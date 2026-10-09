//! Per-design pipeline and classification (SPEC §13 M3 acceptance).
//!
//! For a seed: generate → flatten → kernels + tables → STA at a first time unit → rebuild with
//! clock periods ≥ every STA `min_period` and phases long enough for input changes to settle →
//! STA again (the final report; scale is increased while setup violations remain) → FastRef vs
//! Timing ([`crate::driver`]) → [`Class`].
//!
//! # What explains a mismatch
//!
//! A design is *flagged* when any of these holds:
//! * STA reports an error-severity diagnostic (setup/hold violation, guard hazard, oscillator);
//! * FastRef (or Timing mode) raises one of [`RUNTIME_FLAGS`] after the reset release and not
//!   after the mismatch (power-on `X` transients are outside STA's scope);
//! * the first mismatch involves only `X`/`Z` values and a `Contention` diagnostic was raised in
//!   that window (bus contention creates `X`; STA reasons about binary values);
//! * [`recheck`] attributes a clean mismatch to phases that were too short for a documented
//!   reason (`IdealLatency`, `XSettle`).
//!
//! Everything else that mismatches is [`Class::CleanMismatch`] — a bug.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::Arc;

use panda_compile::{FastOptions, FastRef, KernelOptions, TableCache, compile_kernels_with_cache};
use panda_netlist::flat::FlatNetlist;
use panda_netlist::{DiagKind, Severity, flatten};
use panda_sta::{EventKind, StaOptions, TimingReport, analyze_with};

use crate::driver::{self, DiffResult, DriverOptions, Plan};
use crate::generator::{GenConfig, Generated, generate};

thread_local! {
    /// Kernel tables shared by the designs one worker thread analyzes, with the characterization
    /// options they were built with.
    static TABLES: RefCell<(String, TableCache)> = RefCell::new((String::new(), TableCache::new()));
}

/// Outcome class of one design.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Class {
    /// STA clean, no runtime diagnostic, FastRef = Timing everywhere: good.
    CleanEqual,
    /// STA clean, no runtime diagnostic, but FastRef ≠ Timing: a bug (FastRef, characterization,
    /// kernels, STA conservativeness or the timing engine).
    CleanMismatch,
    /// Flagged (STA error or runtime diagnostic) and FastRef ≠ Timing: explained.
    FlaggedMismatch,
    /// Flagged but FastRef = Timing: STA pessimism (or a harmless runtime diagnostic).
    FlaggedEqual,
    /// Not compared (generator produced an invalid design, or the kernel set was rejected).
    Skipped,
}

/// Runtime diagnostics that explain a mismatch (SPEC §6.3).
pub const RUNTIME_FLAGS: [DiagKind; 4] = [
    DiagKind::DelaySensitive,
    DiagKind::IterationLimit,
    DiagKind::Oscillating,
    DiagKind::Metastable,
];

/// Harness configuration.
#[derive(Clone, Debug)]
pub struct HarnessConfig {
    /// Generator configuration.
    pub generator: GenConfig,
    /// Driver options.
    pub driver: DriverOptions,
    /// STA options.
    pub sta: StaOptions,
    /// FastRef options.
    pub fast: FastOptions,
    /// First time unit tried.
    pub initial_scale: u32,
    /// Largest time unit (beyond: give up shortening setup violations; the design stays flagged).
    pub max_scale: u32,
    /// Count Timing-mode `Metastable`/`Oscillating` diagnostics as flags too.
    pub timing_flags: bool,
    /// Percent of designs whose Timing-mode run uses ideal blocks (SPEC §5.2); the choice is a
    /// function of the seed.
    pub ideal_pct: u32,
    /// Multiplier of the STA input settle bound used to place input changes (see [`recheck`]).
    pub input_settle_factor: u32,
    /// Percent of designs compiled with non-default kernel extraction options (shorter/longer
    /// seed loops, smaller/larger Table kernels; the choice is a function of the seed).
    /// Correctness must not depend on which kernels are found (SPEC §8.3).
    pub extract_variant_pct: u32,
    /// Time-unit multiplier for ideal-block runs. STA analyzes the glass-box gates, but an ideal
    /// block's output always changes after its full (max-depth) latency, so ideal-block designs
    /// can need longer phases than STA's `min_period`.
    pub ideal_scale_factor: u32,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        HarnessConfig {
            generator: GenConfig::default(),
            driver: DriverOptions::default(),
            sta: StaOptions::default(),
            fast: FastOptions::default(),
            initial_scale: 16,
            max_scale: 1024,
            timing_flags: true,
            ideal_pct: 30,
            ideal_scale_factor: 2,
            extract_variant_pct: 20,
            input_settle_factor: 1,
        }
    }
}

/// Everything learned about one design.
#[derive(Clone, Debug)]
pub struct Outcome {
    /// Seed.
    pub seed: u64,
    /// Class.
    pub class: Class,
    /// Time unit used for the comparison.
    pub scale: u32,
    /// Gate count of the flat netlist.
    pub gates: usize,
    /// Kernel count.
    pub kernels: usize,
    /// Kinds of the STA error diagnostics (sorted, deduplicated).
    pub sta_errors: Vec<DiagKind>,
    /// First few STA error messages.
    pub sta_messages: Vec<String>,
    /// Kinds of the runtime flag diagnostics (FastRef, and Timing if enabled).
    pub runtime_flags: Vec<DiagKind>,
    /// The comparison result (`None` if skipped).
    pub diff: Option<DiffResult>,
    /// Generator features.
    pub features: Vec<&'static str>,
    /// Why the design was skipped.
    pub skip_reason: Option<String>,
    /// Other explanations of a mismatch found by the harness (e.g. `"IdealLatency"`: the mismatch
    /// disappears when the ideal-block run gets much longer phases — STA analyzes glass-box gates,
    /// while an ideal block always takes its full latency).
    pub explained: Vec<&'static str>,
    /// Wall time spent on this design (milliseconds).
    pub millis: u64,
    /// Free-form notes of the harness (e.g. the result of a [`recheck`]).
    pub notes: Vec<String>,
}

impl Outcome {
    /// True if flagged (STA error, runtime flag or harness explanation).
    pub fn flagged(&self) -> bool {
        !self.sta_errors.is_empty() || !self.runtime_flags.is_empty() || !self.explained.is_empty()
    }
}

/// STA plus the derived timing parameters.
struct Analysis {
    report: TimingReport,
    /// Max settle of clock/reset events (`None` = unbounded).
    clock_settle: Option<u32>,
    /// Max settle of input events.
    input_settle: Option<u32>,
}

fn analyze(
    flat: &FlatNetlist,
    cfg: &HarnessConfig,
) -> (
    Analysis,
    panda_compile::KernelSet,
    Vec<Option<Arc<panda_compile::KernelTable>>>,
) {
    let kopts = KernelOptions {
        extract: cfg.fast.extract.clone(),
        charz: cfg.fast.charz.clone(),
    };
    let (ks, tables) = TABLES.with(|c| {
        let mut c = c.borrow_mut();
        // One cache per thread and characterization options (identical kernels — D latches,
        // flip-flops — recur across designs and across the time-unit iterations of one design).
        let key = format!("{:?}", kopts.charz);
        if c.0 != key || c.1.len() > 4096 {
            *c = (key, TableCache::new());
        }
        compile_kernels_with_cache(flat, &kopts, &c.1)
    });
    let report = analyze_with(flat, &ks, &tables, &cfg.sta);
    let max_settle = |pred: &dyn Fn(EventKind) -> bool| -> Option<u32> {
        let mut m = 0;
        for p in report.phase_settle.iter().filter(|p| pred(p.kind)) {
            m = m.max(p.settle?);
        }
        Some(m)
    };
    let clock_settle = max_settle(&|k| k != EventKind::Inputs);
    let input_settle = max_settle(&|k| k == EventKind::Inputs);
    (
        Analysis {
            report,
            clock_settle,
            input_settle,
        },
        ks,
        tables,
    )
}

/// Time unit needed so every clock meets its `min_period` and phases leave room for inputs.
fn needed_scale(g: &Generated, a: &Analysis) -> Option<u32> {
    let mut s = 2u32;
    for c in &g.clocks {
        let min = a
            .report
            .min_period_of(&c.name)
            .or_else(|| a.report.max_min_period())
            .unwrap_or(0);
        if min == u32::MAX {
            return None;
        }
        s = s.max(min.div_ceil(c.shape.period_u));
    }
    // Room for an input change: margin (clock settle + 1) + input settle + 1 within the
    // shortest phases (≥ s/2 because of the reset release, ≥ s otherwise).
    let room = a.clock_settle? + a.input_settle.unwrap_or(0) + 3;
    s = s.max(room);
    Some(s.next_multiple_of(2))
}

fn sorted_kinds(it: impl Iterator<Item = DiagKind>) -> Vec<DiagKind> {
    let mut v: Vec<DiagKind> = it.collect();
    v.sort_by_key(|k| format!("{k:?}"));
    v.dedup();
    v
}

/// A design ready for comparison: STA done, FastRef built, plan computed.
pub struct Prepared {
    /// The design (at the final time unit).
    pub generated: Generated,
    /// Its flat netlist.
    pub flat: Arc<FlatNetlist>,
    /// The final STA report.
    pub report: TimingReport,
    /// FastRef, powered on.
    pub fast: FastRef,
    /// Comparison plan.
    pub plan: Plan,
    /// Outcome with everything but the comparison filled in.
    pub outcome: Outcome,
    /// Run Timing mode with ideal blocks.
    pub ideal_blocks: bool,
}

/// Generate the design for `seed` and prepare it (see [`prepare_with`]).
pub fn prepare(seed: u64, cfg: &HarnessConfig) -> Result<Prepared, Box<Outcome>> {
    prepare_with(seed, cfg, false, &|s| generate(seed, &cfg.generator, s))
}

/// Prepare the design `make(scale)`: flatten, kernels + tables, STA; unless `fixed_scale`,
/// rebuild with a larger time unit until every clock meets its STA `min_period`, phases leave room
/// for input changes, and no setup violation remains (bounded by [`HarnessConfig::max_scale`]).
/// Returns the skipped outcome on failure.
pub fn prepare_with(
    seed: u64,
    cfg: &HarnessConfig,
    fixed_scale: bool,
    make: &dyn Fn(u32) -> Generated,
) -> Result<Prepared, Box<Outcome>> {
    let mut out = Outcome {
        seed,
        class: Class::Skipped,
        scale: cfg.initial_scale,
        gates: 0,
        kernels: 0,
        sta_errors: Vec::new(),
        sta_messages: Vec::new(),
        runtime_flags: Vec::new(),
        diff: None,
        features: Vec::new(),
        skip_reason: None,
        explained: Vec::new(),
        millis: 0,
        notes: Vec::new(),
    };
    let ideal_blocks = crate::rng::Rng::new(seed ^ 0x1DEA_1B10_C000).chance(cfg.ideal_pct);
    let mut vr = crate::rng::Rng::new(seed ^ 0xE57_AC7);
    let variant = vr.chance(cfg.extract_variant_pct);
    let owned;
    let cfg = if variant {
        let mut c = cfg.clone();
        c.fast.extract.max_loop = *vr.pick(&[2, 3, 4, 8]);
        c.fast.extract.max_bits = *vr.pick(&[4, 6, 8, 10]);
        owned = c;
        &owned
    } else {
        cfg
    };
    let mut scale = cfg.initial_scale;
    let mut tried = 0;
    let mut stretched = false;
    let (g, flat, a, ks, tables) = loop {
        let g = make(scale);
        scale = g.scale;
        out.features = g.features.clone();
        let flat = match flatten(&g.design) {
            Ok(f) => f,
            Err(e) => {
                out.skip_reason = Some(format!("flatten: {:?}", e.first()));
                return Err(Box::new(out));
            }
        };
        let (a, ks, tables) = analyze(&flat, cfg);
        tried += 1;
        let setup = a.report.of_kind(DiagKind::SetupViolation).next().is_some();
        let next = match needed_scale(&g, &a) {
            _ if fixed_scale => None,
            Some(w) if w > scale => Some(w),
            Some(_) if setup => Some((scale * 3 / 2).next_multiple_of(2)),
            _ => None,
        };
        match next {
            Some(n) if n <= cfg.max_scale && tried < 6 => scale = n,
            _ if ideal_blocks && !fixed_scale && !stretched => {
                stretched = true;
                scale = (scale * cfg.ideal_scale_factor).next_multiple_of(2);
            }
            _ => break (g, flat, a, ks, tables),
        }
    };
    out.scale = scale;
    out.gates = flat.gates.len();
    out.kernels = ks.kernels.len();
    let errors: Vec<_> = a.report.errors().collect();
    out.sta_errors = sorted_kinds(errors.iter().map(|d| d.kind));
    out.sta_messages = errors.iter().take(3).map(|d| d.to_string()).collect();

    let flat = Arc::new(flat);
    let fast = match FastRef::from_parts(flat.clone(), cfg.fast.clone(), ks, tables) {
        Ok(f) => f,
        Err(e) => {
            out.skip_reason = Some(format!("FastRef: {:?}", e.first()));
            return Err(Box::new(out));
        }
    };
    let mut plan = plan_from_report(&a.report, &flat, compare_from(&g), seed, scale / 4);
    plan.input_settle *= u64::from(cfg.input_settle_factor);
    if ideal_blocks {
        plan.input_settle *= u64::from(cfg.ideal_scale_factor);
    }
    if ideal_blocks {
        out.features.push("ideal_blocks");
    }
    if variant {
        out.features.push("extract_variant");
    }
    Ok(Prepared {
        generated: g,
        flat,
        report: a.report,
        fast,
        plan,
        outcome: out,
        ideal_blocks,
    })
}

/// A comparison plan from an STA report: inputs may change one tick after the slowest clock/reset
/// event settled and need the slowest input event's settle time (`fallback` ticks when STA gives
/// no bound). Every top-level input pin may change.
pub fn plan_from_report(
    report: &TimingReport,
    flat: &FlatNetlist,
    compare_from: u64,
    seed: u64,
    fallback: u32,
) -> Plan {
    let max_settle = |inputs: bool| -> u32 {
        let mut m = 0;
        for p in report
            .phase_settle
            .iter()
            .filter(|p| (p.kind == EventKind::Inputs) == inputs)
        {
            m = m.max(p.settle.unwrap_or(fallback));
        }
        m
    };
    Plan {
        input_margin: u64::from(max_settle(false)) + 1,
        input_settle: u64::from(max_settle(true)),
        compare_from,
        inputs: flat
            .top_inputs
            .iter()
            .map(|p| (p.name.clone(), p.nets.len() as u32))
            .collect(),
        seed,
    }
}

/// Compare a prepared design and classify it.
pub fn finish(
    p: Prepared,
    cfg: &HarnessConfig,
    observe: &mut dyn FnMut(&FastRef, &panda_timing::TimingSim, driver::Point, u64),
) -> Outcome {
    let Prepared {
        flat,
        fast,
        plan,
        mut outcome,
        ideal_blocks,
        ..
    } = p;
    let mut dopts = cfg.driver.clone();
    dopts.timing.ideal_blocks |= ideal_blocks;
    let diff = driver::run_observed(&flat, fast, &plan, &dopts, observe);
    // A runtime diagnostic explains the outcome if it was raised after the reset release (power-on
    // X transients are out of scope) and, for a mismatch, not after it.
    let last = diff.first().map_or(u64::MAX, |m| m.sample_tick);
    let in_window = |d: &&panda_netlist::Diagnostic| {
        diag_tick(d).is_none_or(|t| t >= plan.compare_from && t <= last)
    };
    let mut flags: Vec<DiagKind> = diff
        .fast_diags
        .iter()
        .filter(in_window)
        .map(|d| d.kind)
        .filter(|k| RUNTIME_FLAGS.contains(k))
        .collect();
    if cfg.timing_flags {
        flags.extend(
            diff.timing_diags
                .iter()
                .filter(in_window)
                .filter(|d| d.severity != Severity::Info)
                .map(|d| d.kind)
                .filter(|k| matches!(k, DiagKind::Metastable | DiagKind::Oscillating)),
        );
    }
    // X introduced by bus contention (reported at runtime by both engines; STA reasons about
    // binary values only) explains disagreements that involve only X/Z values.
    // Only contention after the reset release and before the mismatch counts (power-on X drivers
    // also report contention in Timing mode).
    if let Some(m) = diff.first()
        && m.only_unknown
        && diff.fast_diags.iter().chain(&diff.timing_diags).any(|d| {
            d.kind == DiagKind::Contention
                && diag_tick(d).is_some_and(|t| t >= plan.compare_from && t <= m.sample_tick)
        })
    {
        flags.push(DiagKind::Contention);
    }
    outcome.runtime_flags = sorted_kinds(flags.into_iter());
    outcome.class = match (outcome.flagged(), diff.equal()) {
        (false, true) => Class::CleanEqual,
        (false, false) => Class::CleanMismatch,
        (true, false) => Class::FlaggedMismatch,
        (true, true) => Class::FlaggedEqual,
    };
    outcome.diff = Some(diff);
    outcome
}

/// The tick named in a diagnostic message (`"... at tick N..."` or `"(tick N)"`), if any.
pub fn diag_tick(d: &panda_netlist::Diagnostic) -> Option<u64> {
    let m = &d.message;
    let rest = match m.find("at tick ") {
        Some(i) => &m[i + 8..],
        None => &m[m.find("(tick ")? + 6..],
    };
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Run the whole pipeline for one seed (with the [`recheck`] of clean-but-mismatch designs).
pub fn run_seed(seed: u64, cfg: &HarnessConfig) -> Outcome {
    let t0 = std::time::Instant::now();
    let mut o = match prepare(seed, cfg) {
        Ok(p) => {
            let g = p.generated.clone();
            let mut o = finish(p, cfg, &mut |_, _, _, _| {});
            recheck(&g, &mut o, cfg);
            o
        }
        Err(o) => *o,
    };
    o.millis = t0.elapsed().as_millis() as u64;
    o
}

/// Multiply every clock period/high/phase and `PowerOnReset` length of a design by `k`.
pub fn stretch_design(design: &panda_netlist::Design, k: u32) -> panda_netlist::Design {
    use panda_netlist::{Atom, InstKind};
    let mut d = design.clone();
    for def in &mut d.defs {
        for inst in &mut def.instances {
            match &mut inst.kind {
                InstKind::Atom(Atom::Clock {
                    period,
                    high,
                    phase,
                }) => {
                    *period *= k;
                    *high *= k;
                    *phase *= k;
                }
                InstKind::Atom(Atom::PowerOnReset { ticks }) => *ticks *= k,
                _ => {}
            }
        }
    }
    d
}

/// Re-examine a clean-but-mismatch outcome: rerun the same design with every period, the reset
/// and the input settle allowance 4× longer. If both engines then agree, the mismatch came from
/// phases that were too short, which STA cannot see in two documented cases:
/// * `"IdealLatency"`: Timing mode ran with ideal blocks, whose outputs always take the full
///   (max-depth) latency, while STA analyzes the glass-box gates;
/// * `"XSettle"`: every disagreeing net involves `X` — STA reasons about binary values (e.g.
///   `a AND NOT a = 0` blocks a path that an `X` on `a` sensitizes), and a persistent `X` (state
///   never reset) makes the real settle time longer.
///
/// Otherwise (binary values, glass-box) the outcome stays a bug; a note records whether longer
/// phases help (an STA setup bound that is too small) or not.
pub fn recheck(g: &Generated, o: &mut Outcome, cfg: &HarnessConfig) {
    if o.class != Class::CleanMismatch {
        return;
    }
    const K: u32 = 4;
    let mut g2 = g.clone();
    g2.design = stretch_design(&g.design, K);
    g2.scale = g.scale * K;
    g2.reset_ticks = g.reset_ticks * K;
    for c in &mut g2.clocks {
        c.period *= K;
        c.high *= K;
        c.phase *= K;
    }
    let mut c2 = cfg.clone();
    c2.input_settle_factor *= K;
    let again = match prepare_with(o.seed, &c2, true, &|_| g2.clone()) {
        Ok(p) => finish(p, &c2, &mut |_, _, _, _| {}),
        Err(o) => *o,
    };
    let agrees = again.diff.as_ref().is_some_and(DiffResult::equal);
    let only_unknown = o
        .diff
        .as_ref()
        .and_then(DiffResult::first)
        .is_some_and(|m| m.only_unknown);
    if agrees && o.features.contains(&"ideal_blocks") {
        o.explained.push("IdealLatency");
    } else if agrees && only_unknown {
        o.explained.push("XSettle");
    } else {
        o.notes
            .push(format!("with 4x longer phases: {:?}", again.class));
        return;
    }
    o.class = Class::FlaggedMismatch;
}

/// First tick that counts: STA does not cover power-on `X` transients, so comparisons start at
/// the reset release (every reset register has been reset by then).
pub fn compare_from(g: &Generated) -> u64 {
    u64::from(g.reset_ticks)
}

/// Aggregated statistics over many designs.
#[derive(Clone, Debug, Default)]
pub struct Summary {
    /// Designs per class.
    pub classes: BTreeMap<Class, usize>,
    /// Seeds of clean-but-mismatch designs.
    pub bugs: Vec<u64>,
    /// STA error kinds over flagged designs (each design counted once per kind).
    pub flag_kinds: BTreeMap<String, usize>,
    /// Flagged-and-equal designs per flag combination.
    pub pessimism_kinds: BTreeMap<String, usize>,
    /// Phases compared in total.
    pub phases: usize,
    /// Input events in total.
    pub input_events: usize,
    /// Designs with power-on (pre-reset) mismatches, not counted.
    pub early_mismatch_designs: usize,
    /// Designs per feature.
    pub features: BTreeMap<&'static str, usize>,
}

impl Summary {
    /// Add one outcome.
    pub fn add(&mut self, o: &Outcome) {
        *self.classes.entry(o.class).or_default() += 1;
        if o.class == Class::CleanMismatch {
            self.bugs.push(o.seed);
        }
        for f in &o.features {
            *self.features.entry(f).or_default() += 1;
        }
        let key = || {
            let mut k: Vec<String> = o.sta_errors.iter().map(|k| format!("{k:?}")).collect();
            k.extend(o.runtime_flags.iter().map(|k| format!("rt:{k:?}")));
            k.extend(o.explained.iter().map(|k| format!("h:{k}")));
            k.join("+")
        };
        if o.flagged() {
            *self.flag_kinds.entry(key()).or_default() += 1;
        }
        if o.class == Class::FlaggedEqual {
            *self.pessimism_kinds.entry(key()).or_default() += 1;
        }
        if let Some(d) = &o.diff {
            self.phases += d.compared;
            self.input_events += d.input_events;
            if d.early_mismatches > 0 {
                self.early_mismatch_designs += 1;
            }
        }
    }

    /// Count of a class.
    pub fn count(&self, c: Class) -> usize {
        self.classes.get(&c).copied().unwrap_or(0)
    }

    /// Flagged-and-equal / flagged (STA pessimism rate).
    pub fn pessimism_rate(&self) -> f64 {
        let fe = self.count(Class::FlaggedEqual);
        let f = fe + self.count(Class::FlaggedMismatch);
        if f == 0 { 0.0 } else { fe as f64 / f as f64 }
    }
}

impl std::fmt::Display for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total: usize = self.classes.values().sum();
        writeln!(
            f,
            "{total} designs, {} phases compared, {} input events",
            self.phases, self.input_events
        )?;
        for (c, n) in &self.classes {
            writeln!(f, "  {c:?}: {n}")?;
        }
        writeln!(
            f,
            "  STA pessimism (flagged-and-equal / flagged): {:.1}%",
            100.0 * self.pessimism_rate()
        )?;
        writeln!(
            f,
            "  designs with ignored power-on mismatches: {}",
            self.early_mismatch_designs
        )?;
        writeln!(f, "  flag kinds: {:?}", self.flag_kinds)?;
        writeln!(f, "  pessimism kinds: {:?}", self.pessimism_kinds)?;
        writeln!(f, "  features: {:?}", self.features)?;
        if !self.bugs.is_empty() {
            writeln!(f, "  CLEAN-BUT-MISMATCH seeds: {:?}", self.bugs)?;
        }
        Ok(())
    }
}

/// Run a range of seeds (in parallel over `threads` OS threads) and summarize. Outcomes are
/// returned in seed order.
pub fn run_seeds(
    seeds: std::ops::Range<u64>,
    cfg: &HarnessConfig,
    threads: usize,
) -> (Summary, Vec<Outcome>) {
    let seeds: Vec<u64> = seeds.collect();
    let threads = threads.clamp(1, seeds.len().max(1));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut outcomes: Vec<Option<Outcome>> = vec![None; seeds.len()];
    let results = std::sync::Mutex::new(&mut outcomes);
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= seeds.len() {
                        break;
                    }
                    let o = run_seed(seeds[i], cfg);
                    results.lock().expect("lock")[i] = Some(o);
                }
            });
        }
    });
    let outcomes: Vec<Outcome> = outcomes.into_iter().map(|o| o.expect("outcome")).collect();
    let mut s = Summary::default();
    for o in &outcomes {
        s.add(o);
    }
    (s, outcomes)
}

/// Number of worker threads to use by default.
pub fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}
