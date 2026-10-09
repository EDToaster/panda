//! panda-sta — static timing analysis (SPEC §9): when is Fast mode guaranteed to equal Timing
//! mode?
//!
//! Fast mode (SPEC §6.2) processes each event in *rounds*: round 0 applies the event's source
//! changes through zero-delay combinational logic; every kernel whose inputs changed then takes one
//! table transition, and the changes of its outputs form the next round. Timing mode (§5) runs the
//! same gates with unit delays. [`analyze`] checks, in unit-delay ticks, the conditions of SPEC §6.3
//! under which both agree at every phase boundary. It is **conservative**: when it reports no
//! error-severity diagnostic and every clock period is at least its [`TimingReport::min_period`],
//! the two modes agree (for binary values; see "Scope" below).
//!
//! # Method
//!
//! For every *event signature* (distinct clock-value change, `PowerOnReset` release, UI input
//! change; see the `schedule` module) the analysis computes, for every net, the ticks at which it
//! may change, separately for every Fast round label, with a path condition (literals over final
//! net values that must hold for the change to happen) and a glitch flag:
//!
//! * combinational logic: min/max path depth, controlling values of statically known and of
//!   already-settled inputs, literal conditions from stable side inputs (so `mux` select lines and
//!   gated clocks are understood: a change that needs `gclk = 0` cannot pass a latch that needs
//!   `gclk = 1`), glitch detection when compatible changes of one round reconverge at different
//!   ticks;
//! * kernels: enumeration of the feasible characterized table entries under everything known about
//!   the kernel's inputs in that round (statically known values before/after, correlations through
//!   inverters, path-condition restrictions). Which state nets can change, the structural minimum
//!   and the tabulated maximum latency, entry-derived literals. Nets that may change only
//!   *transiently* (a pulse back to the initial value) are found by unit-delay simulation of the
//!   entries (exact bursts), from the timed check, or from the ternary A states (other bursts)
//!   and also get a (glitchy) window. Kernels without a table are treated conservatively (any input change may change
//!   every output, settle bound `4 × nets + 8 + 2 × p2_limit`).
//!
//! The analysis iterates to a fixpoint; an event whose rounds exceed `delta_limit`, or whose
//! windows keep growing, may never settle.
//!
//! # Checks
//!
//! * **Setup** (§9.1): the last change of an event (any net, any kernel-internal net) must happen
//!   before the next event: `settle < gap` (a `PowerOnReset` release inside a phase also ends
//!   that phase). `min_period` per clock is the smallest even period (keeping the clock's
//!   high/period ratio) for which every event of that clock settles in its part of the cycle.
//! * **Hold** (§9.2): at every kernel, all input changes of round `r' > r` must arrive no earlier
//!   than the round-`r` transition has settled (the tick of the last kernel-net change). At that
//!   tick the kernel holds the stable state Fast mode computed, so the later burst is an ordinary
//!   table transition in both modes; the condition is therefore exact at equality (`≥`, not `>`).
//!   Overlapping rounds are still accepted when the *merged* burst passes the timed check or is
//!   burst-safe (below), e.g. a flip-flop's own feedback (or another flip-flop's output)
//!   reaching it after its master closed but while its slave is still settling, or data reaching
//!   a latch that is still opening.
//! * **Bursts and guard hazards** (§9.3, §6.3 cond. 3–4): a round's burst at a kernel is *exact*
//!   when one clean input changes, or all changing inputs change cleanly at the same tick; its
//!   output windows come from unit-delay simulation of every feasible entry (each net's own first
//!   and last change, not the whole transition's latency). Any other burst (skewed arrival,
//!   glitching inputs, overlapping rounds) is accepted if it passes the *timed check*: every
//!   feasible entry is simulated with Timing mode's unit delays in ternary logic, each changing
//!   input holding its initial value before its arrival window, `X` inside it and its final value
//!   after it (exact outside the windows, so a clock change that arrives cleanly first is
//!   simulated exactly), and must end in the binary state Fast mode computes (the table
//!   transition, or for overlapping rounds the chain of round transitions, with every possible
//!   intermediate value of an input that changes in several rounds); period-2 collapse is
//!   accounted for. Its output windows are the simulated change ticks. Otherwise the burst must
//!   be *burst-safe*: Eichelberger ternary simulation (changing inputs at `X`, then their final
//!   values) ends in the state the table predicts, for every feasible entry, so arrival order,
//!   skew and glitches cannot matter. A failing burst with glitching inputs is a `GuardHazard`
//!   (glitchy enables, pulse-generated clocks); with clean but skewed inputs it is a
//!   `HoldViolation`. Timed checks have a work budget per event (beyond it, only burst safety is
//!   tried), and kernel output windows only grow during the fixpoint iteration.
//! * **DelaySensitive** (advisory, info severity): a reachable table entry with several inputs
//!   changing in one round is flagged order-sensitive by characterization.
//! * Kernels with an input vector without stable state (oscillators, §6.3 cond. 5) are an
//!   `Oscillating` error, and so is a reachable transition that never settles: a table entry
//!   recorded as non-binary is re-simulated, and if it oscillates (rather than collapsing to `X`)
//!   Timing mode keeps oscillating where Fast mode yields `X`. Kernels without a table (Sim
//!   kernels) must be *proven* to settle: every binary stable state under the round's feasible
//!   inputs is enumerated through a greedy feedback cut (free input bits + cut bits ≤ 16) and every
//!   feasible transition is simulated; if that is impossible or some transition oscillates, the
//!   round is an `Oscillating` error. Kernels with more than 31 inputs are analyzed fully
//!   conservatively (and never proven to settle). A reachable transition ending in `X` (period-2
//!   collapse) is a `Metastable` warning.
//!
//! # Ideal blocks
//!
//! With [`StaOptions::ideal_blocks`] the analysis targets Timing mode with ideal blocks (SPEC
//! §5.2): a block output changes exactly `L_j` ticks after a change of an input in its cone and
//! never glitches internally (only inputs changing at different ticks make it change several
//! times). Block elements absorbed into a kernel as private fan-in make the rounds in which an
//! input reaching them changes delay-independent (ternary burst safety even for one clean change)
//! with the blocks' latency added (the timed check, which assumes unit-delay gates, is not used
//! for them); a block element on a feedback loop of a kernel is an error (Fast mode's transitions
//! simulate its gates one by one).
//!
//! # Scope
//!
//! The analysis reasons about binary values: power-on `X` transients are not covered, and kernels
//! whose inputs can carry `Z` are analyzed conservatively. UI input events are analyzed (hazards,
//! hold, settle time reported in [`TimingReport::phase_settle`]) but have no setup check: when they
//! happen is up to the user. With several clocks, `min_period` of a clock is computed as if its own
//! edges were the only events; the configured schedule is checked exactly over the hyperperiod.
//! If the hyperperiod exceeds [`StaOptions::max_hyperperiod`], every set of edges that can
//! coincide is analyzed with the other clocks unknown, and the gap after it is bounded by the
//! closest possible approach of any later edge (exact over all time, but more pessimistic).

mod cond;
mod event;
mod model;
mod schedule;

use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};

use panda_compile::{
    CharzOptions, KernelKind, KernelOptions, KernelSet, KernelTable, compile_kernels,
    kernel_subnet, subnet::structural_key,
};
use panda_netlist::flat::FlatNetlist;
use panda_netlist::{DiagKind, Diagnostic, Logic, Severity};

use crate::event::Ctx;
use crate::model::{Model, ModelLimits};

/// Kind of an analyzed event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EventKind {
    /// One or more clock edges.
    Clock,
    /// A `PowerOnReset` release (possibly together with clock edges).
    Reset,
    /// UI input changes during a phase.
    Inputs,
}

/// Settle time of one analyzed event signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhaseSettle {
    /// Kind of event.
    pub kind: EventKind,
    /// Description, e.g. `"clk↑"`, `"reset release"`, `"inputs [clk=0]"`.
    pub event: String,
    /// Clock values after the event (running clocks, by clock net name).
    pub clocks: Vec<(String, Logic)>,
    /// Ticks after the event until the last change anywhere; `None` = may never settle.
    pub settle: Option<u32>,
    /// Smallest number of ticks until the next event in the configured schedule (`None` for UI
    /// input events, or when nothing follows).
    pub gap: Option<u32>,
    /// Number of Fast rounds the event can take.
    pub rounds: u32,
    /// Net names along the path to the last change.
    pub critical_path: Vec<String>,
}

/// Result of [`analyze`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TimingReport {
    /// Per running clock (clock net name): the smallest even period (same high/period ratio) for
    /// which the clock's events settle in time. `u32::MAX` if they never settle.
    pub min_period: Vec<(String, u32)>,
    /// Per analyzed event signature: worst settle time after the event.
    pub phase_settle: Vec<PhaseSettle>,
    /// `SetupViolation` / `HoldViolation` / `GuardHazard` / `Oscillating` (errors), `Metastable`
    /// (warning), `DelaySensitive` (info, advisory).
    pub diagnostics: Vec<Diagnostic>,
    /// Net names along the worst setup path: the path of the clock event that requires the longest
    /// clock period (the one that determines `min_period`), or of the slowest event if there is no
    /// clock.
    pub critical_path: Vec<String>,
}

impl TimingReport {
    /// True if no error-severity diagnostic was raised: Fast mode equals Timing mode at phase
    /// boundaries (for the configured clock periods).
    pub fn is_clean(&self) -> bool {
        self.errors().next().is_none()
    }

    /// Error-severity diagnostics.
    pub fn errors(&self) -> impl Iterator<Item = &Diagnostic> {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Error)
    }

    /// Diagnostics of one kind.
    pub fn of_kind(&self, kind: DiagKind) -> impl Iterator<Item = &Diagnostic> {
        self.diagnostics.iter().filter(move |d| d.kind == kind)
    }

    /// `min_period` of the clock driving net `clock`.
    pub fn min_period_of(&self, clock: &str) -> Option<u32> {
        self.min_period
            .iter()
            .find(|(n, _)| n == clock)
            .map(|&(_, p)| p)
    }

    /// The largest `min_period` over all clocks.
    pub fn max_min_period(&self) -> Option<u32> {
        self.min_period.iter().map(|&(_, p)| p).max()
    }
}

/// Tunables of [`analyze_with`].
#[derive(Clone, Debug)]
pub struct StaOptions {
    /// Maximum number of Fast rounds per event (FastRef's `DELTA_LIMIT`). Default 64.
    pub delta_limit: u16,
    /// Table entries one round query may enumerate before falling back to the conservative
    /// answer. Default 4,194,304.
    pub max_enum_work: u64,
    /// Unit-delay simulations per round query for output-glitch detection (beyond: assume
    /// glitches). Default 8192.
    pub max_glitch_sims: usize,
    /// Ternary simulations per round query for burst safety (beyond: unsafe). Default 262,144.
    pub max_ternary: usize,
    /// Analyze UI input events. Default true.
    pub input_events: bool,
    /// Emit advisory `DelaySensitive` diagnostics. Default true.
    pub advisory: bool,
    /// Largest hyperperiod enumerated for the clock schedule. Default 2^20 ticks. Beyond it the
    /// schedule is over-approximated (see the crate docs, "Scope").
    pub max_hyperperiod: u64,
    /// Characterization options (settle bounds and simulator options for kernels).
    pub charz: CharzOptions,
    /// Analyze for Timing mode with ideal blocks (SPEC §5.2, `TimingOptions::ideal_blocks`):
    /// every ideal block output changes exactly its latency after an input in its cone, without
    /// internal glitches. Default false (glass-box gates).
    pub ideal_blocks: bool,
    /// Instance paths simulated at gate level when `ideal_blocks` is on (same meaning as
    /// `TimingOptions::glass_box`).
    pub glass_box: HashSet<String>,
}

impl Default for StaOptions {
    fn default() -> Self {
        StaOptions {
            delta_limit: 64,
            max_enum_work: 1 << 22,
            max_glitch_sims: 8192,
            max_ternary: 1 << 18,
            input_events: true,
            advisory: true,
            max_hyperperiod: 1 << 20,
            charz: CharzOptions::default(),
            ideal_blocks: false,
            glass_box: HashSet::new(),
        }
    }
}

/// Analyze `flat` with its kernels and their tables (`tables[i]` belongs to
/// `kernels.kernels[i]`, `None` for Sim/Keeper kernels), with default options.
///
/// `T` is typically `Arc<KernelTable>` (as returned by [`compile_kernels`]) or `KernelTable`.
pub fn analyze<T: Borrow<KernelTable>>(
    flat: &FlatNetlist,
    kernels: &KernelSet,
    tables: &[Option<T>],
) -> TimingReport {
    analyze_with(flat, kernels, tables, &StaOptions::default())
}

/// Extract and characterize the kernels of `flat` with default options, then [`analyze`].
pub fn analyze_netlist(flat: &FlatNetlist) -> TimingReport {
    let (ks, tables) = compile_kernels(flat, &KernelOptions::default());
    analyze(flat, &ks, &tables)
}

/// [`analyze`] with explicit options.
pub fn analyze_with<T: Borrow<KernelTable>>(
    flat: &FlatNetlist,
    kernels: &KernelSet,
    tables: &[Option<T>],
    opts: &StaOptions,
) -> TimingReport {
    let mut diags = Vec::new();
    // Models: one per (structure, table).
    let mut models: Vec<Model> = Vec::new();
    let mut index: HashMap<(Vec<u32>, usize), usize> = HashMap::new();
    let mut kernel_model = Vec::with_capacity(kernels.kernels.len());
    for (kid, k) in kernels.kernels.iter().enumerate() {
        let table: Option<&KernelTable> =
            tables.get(kid).and_then(|t| t.as_ref()).map(|t| t.borrow());
        let table = table.filter(|_| k.kind == KernelKind::Table);
        if table.is_some_and(KernelTable::is_oscillator) {
            diags.push(
                Diagnostic::error(
                    DiagKind::Oscillating,
                    "kernel has an input vector without a stable state (oscillator): Fast mode \
                     yields X where Timing mode oscillates (SPEC §6.3 condition 5)",
                )
                .at(flat
                    .net_names
                    .get(k.nets[0] as usize)
                    .cloned()
                    .unwrap_or_default()),
            );
        }
        let (sub, map) = kernel_subnet(flat, k);
        let key = (
            structural_key(&sub),
            table.map_or(0, |t| std::ptr::from_ref(t) as usize),
        );
        let mi = *index.entry(key).or_insert_with(|| {
            models.push(Model::new(sub, map, table, &opts.charz));
            models.len() - 1
        });
        kernel_model.push(mi);
    }
    let limits = ModelLimits {
        max_enum_work: opts.max_enum_work,
        max_glitch_sims: opts.max_glitch_sims,
        max_ternary: opts.max_ternary,
    };
    let blocks = if opts.ideal_blocks {
        panda_timing::select_ideal_blocks(flat, &opts.glass_box)
    } else {
        Vec::new()
    };
    let mut ctx = Ctx::new(
        flat,
        kernels,
        models,
        kernel_model,
        limits,
        opts.delta_limit,
        opts.advisory,
        &blocks,
    );
    for (kid, path) in &ctx.ideal_conflicts {
        diags.push(
            Diagnostic::error(
                DiagKind::Other,
                format!(
                    "ideal block '{path}' lies on a feedback loop of this kernel: Fast mode's \
                     kernel transitions simulate its gates one by one, Timing mode evaluates it as \
                     an ideal block (glass-box the block to analyze this design)"
                ),
            )
            .at(ctx.kernel_name(*kid)),
        );
    }
    let sched = schedule::build(flat, opts.max_hyperperiod, opts.input_events);
    if sched.truncated {
        diags.push(Diagnostic {
            kind: DiagKind::Other,
            severity: Severity::Info,
            message: format!(
                "clock hyperperiod exceeds {} ticks: each set of possibly simultaneous edges is \
                 analyzed with the other clocks unknown, and gaps are bounded analytically",
                opts.max_hyperperiod
            ),
            location: None,
        });
    }
    if sched.incomplete {
        diags.push(Diagnostic::error(
            DiagKind::SetupViolation,
            "too many clocks to enumerate their possibly simultaneous edges: the clock schedule \
             was not fully analyzed",
        ));
    }
    let mut report = TimingReport::default();
    // (period required by the event alone, path) of the clock event that limits the period most.
    let mut worst: Option<(f64, Vec<String>)> = None;
    let mut slowest: Option<(u32, Vec<String>)> = None;
    let mut rise_settle: Vec<Option<Option<u32>>> = vec![None; sched.clocks.len()];
    let mut fall_settle: Vec<Option<Option<u32>>> = vec![None; sched.clocks.len()];
    let merge = |slot: &mut Option<Option<u32>>, s: Option<u32>| {
        *slot = Some(match (*slot, s) {
            (None, s) => s,
            (Some(None), _) | (_, None) => None,
            (Some(Some(a)), Some(b)) => Some(a.max(b)),
        });
    };
    for spec in &sched.events {
        let out = ctx.run_event(spec);
        diags.extend(out.diags.iter().cloned());
        if spec.kind == EventKind::Clock {
            for &c in &spec.rising {
                merge(&mut rise_settle[c], out.settle);
            }
            for &c in &spec.falling {
                merge(&mut fall_settle[c], out.settle);
            }
            // Period this event alone requires of each clock changing at it.
            let need = |share: u64, period: u64| {
                out.settle.map_or(f64::INFINITY, |s| {
                    (f64::from(s) + 1.0) * period as f64 / share.max(1) as f64
                })
            };
            let required = spec
                .rising
                .iter()
                .map(|&c| need(sched.clocks[c].high, sched.clocks[c].period))
                .chain(spec.falling.iter().map(|&c| {
                    let k = &sched.clocks[c];
                    need(k.period - k.high, k.period)
                }))
                .fold(0.0, f64::max);
            if worst.as_ref().is_none_or(|(w, _)| required > *w) {
                worst = Some((required, out.path.clone()));
            }
        }
        let gap = spec.gap.map(|g| g.min(u64::from(u32::MAX)) as u32);
        if let Some(g) = gap {
            let settle_ok = out.settle.is_some_and(|s| s < g);
            if !settle_ok && out.settle.is_some() {
                diags.push(
                    Diagnostic::error(
                        DiagKind::SetupViolation,
                        format!(
                            "{}: the event settles {} ticks after it, but the next event follows after \
                             {g} ticks",
                            spec.name,
                            out.settle.unwrap_or(0)
                        ),
                    )
                    .at(out.path.last().cloned().unwrap_or_default()),
                );
            }
        }
        let s = out.settle.unwrap_or(u32::MAX);
        if slowest.as_ref().is_none_or(|(w, _)| s > *w) {
            slowest = Some((s, out.path.clone()));
        }
        report.phase_settle.push(PhaseSettle {
            kind: spec.kind,
            event: spec.name.clone(),
            clocks: spec.clocks_after.clone(),
            settle: out.settle,
            gap,
            rounds: out.rounds,
            critical_path: out.path,
        });
    }
    for (ci, c) in sched.clocks.iter().enumerate() {
        let r = rise_settle[ci].unwrap_or(Some(0));
        let f = fall_settle[ci].unwrap_or(Some(0));
        let p = schedule::min_period(c, r, f).unwrap_or(u32::MAX);
        report.min_period.push((c.name.clone(), p));
    }
    report.critical_path = worst
        .map(|w| w.1)
        .or(slowest.map(|s| s.1))
        .unwrap_or_default();
    // Deduplicate identical diagnostics (e.g. the same hazard found for several events).
    let mut seen = std::collections::HashSet::new();
    diags.retain(|d| seen.insert(d.clone()));
    report.diagnostics = diags;
    report
}
