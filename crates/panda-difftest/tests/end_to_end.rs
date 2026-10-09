//! End-to-end differential test of the whole engine stack (SPEC §6.3, D10, D12, §8.9, §10).
//!
//! For every design (random designs from [`panda_difftest::generator`] and Panda-8 in every
//! variant), one stimulus drives all engines in lockstep:
//!
//! | engine | built through | checked against |
//! |---|---|---|
//! | Timing mode, glass-box (`panda_timing::TimingSim`) | the `panda_kernel` facade, `Mode::Timing` | FastRef at phase ends, on STA-clean designs (SPEC §6.3) |
//! | FastRef (`panda_compile::FastRef`), the oracle | directly | — |
//! | FastSim interpreter (`panda_fast::FastSim`) | directly | FastRef, bit-exact after **every event** |
//! | FastSim + Cranelift JIT (`panda_jit::Jit::install`) | directly | FastRef, bit-exact after every event |
//! | facade Fast mode, `FastBackend::Reference` | `panda_kernel::Kernel` | FastRef, bit-exact at every phase end |
//! | facade Fast mode, `FastBackend::Compiled` | `panda_kernel::Kernel` | FastRef, bit-exact at every phase end |
//! | facade Fast mode, `FastBackend::Jit` (native tier active) | `panda_kernel::Kernel` | FastRef, bit-exact at every phase end |
//!
//! "Bit-exact" covers every net value, `tick()` / `next_event_tick()` and, for the directly driven
//! engines, the diagnostic kinds of every event and the event/round/transition/table-hit/fallback
//! counters. In addition, the independent per-seed pipeline of [`panda_difftest::harness`] (raw
//! `TimingSim` vs FastRef with random initial inputs) must never classify a design as
//! clean-but-mismatch.
//!
//! The stimulus follows the driver protocol of [`panda_difftest::driver`]: input changes are made
//! inside a phase, after the clock event settled (STA bound + 1) and early enough for the input
//! event to settle before the next event; Timing mode applies them at that tick, the Fast engines
//! process them as their own event right after the clock event.

use std::sync::Arc;

use panda_compile::{FastOptions, FastRef, KernelOptions as CompileOptions, compile_kernels};
use panda_cpu8::emu::Emulator;
use panda_cpu8::{Options, RomStyle, Variant, panda8_with, programs};
use panda_difftest::harness::{self, Class, HarnessConfig, RUNTIME_FLAGS};
use panda_difftest::{GenConfig, Rng};
use panda_fast::{FastSim, FastSimOptions};
use panda_jit::Jit;
use panda_kernel::{EngineKind, FastBackend, Kernel, KernelOptions, Mode, StepUnit};
use panda_netlist::{DiagKind, Diagnostic, FlatNetlist, Logic, Severity, Value, flatten};
use panda_timing::TimingOptions;

/// The facade's Fast backends, all checked.
const BACKENDS: [FastBackend; 3] = [
    FastBackend::Reference,
    FastBackend::Compiled,
    FastBackend::Jit,
];

/// First differing net of two value vectors, for messages.
fn first_diff(flat: &FlatNetlist, want: &[Logic], got: &[Logic]) -> Option<String> {
    if want.len() != got.len() {
        return Some(format!("{} nets vs {} nets", want.len(), got.len()));
    }
    let diffs: Vec<usize> = (0..want.len()).filter(|&n| want[n] != got[n]).collect();
    let &n = diffs.first()?;
    Some(format!(
        "{} nets differ, first {} = {} (want {})",
        diffs.len(),
        flat.net_names[n],
        got[n].to_char(),
        want[n].to_char()
    ))
}

/// Sorted diagnostic kinds (the per-event comparison of D12).
fn kinds(diags: &[Diagnostic]) -> Vec<String> {
    let mut k: Vec<String> = diags.iter().map(|d| format!("{:?}", d.kind)).collect();
    k.sort();
    k
}

/// A diagnostic with the tick it belongs to.
struct Timed {
    tick: u64,
    diag: Diagnostic,
}

/// Every engine of the stack on one design, driven in lockstep.
struct Chain {
    label: String,
    flat: Arc<FlatNetlist>,
    fr: FastRef,
    /// FastSim interpreter and FastSim with the JIT installed.
    sims: Vec<(&'static str, FastSim)>,
    /// Facade kernels in Fast mode, one per backend.
    facade: Vec<(FastBackend, Kernel)>,
    /// Facade kernel in Timing mode (glass-box).
    timing: Kernel,
    /// FastRef's diagnostics so far.
    fast_diags: Vec<Timed>,
    /// The Timing kernel's diagnostics so far.
    timing_diags: Vec<Timed>,
    /// Events processed by FastRef after power-on.
    events: u64,
}

impl Chain {
    /// Build every engine for `design` (`flat` must be its flattening) with Fast options `opts`.
    fn new(
        label: String,
        design: &panda_netlist::Design,
        flat: Arc<FlatNetlist>,
        opts: &FastOptions,
    ) -> Chain {
        let fr = FastRef::new(flat.clone(), opts.clone()).expect("FastRef");
        let sopts = FastSimOptions::from(opts.clone());
        let interp = FastSim::new(flat.clone(), sopts.clone()).expect("FastSim");
        let mut jit = FastSim::new(flat.clone(), sopts).expect("FastSim (JIT)");
        let stats = Jit::default()
            .install(&mut jit)
            .unwrap_or_else(|e| panic!("{label}: JIT compile failed: {e}"));
        assert!(jit.native().is_some(), "{label}: native code installed");
        assert_eq!(stats.partitions, jit.program().partitions.len());
        let timing_opts = TimingOptions::default()
            .with_ideal_blocks(false)
            .with_stuck_x_periods(0);
        let kopts = |backend| KernelOptions {
            timing: timing_opts.clone(),
            fast: opts.clone(),
            backend,
            background: false,
            ..KernelOptions::default()
        };
        let facade = BACKENDS
            .iter()
            .map(|&b| {
                let mut k = Kernel::new(kopts(b));
                // Fast mode before loading: the engine powers on at tick 0 like FastRef.
                k.set_mode(Mode::Fast);
                k.load(design.clone())
                    .unwrap_or_else(|e| panic!("{label}: load: {e:?}"));
                k.wait_ready()
                    .unwrap_or_else(|e| panic!("{label}: {b:?} compile: {e:?}"));
                assert_eq!(k.engine(), EngineKind::Fast(b), "{label}: engine");
                if b == FastBackend::Jit {
                    assert!(k.jit_active(), "{label}: facade JIT tier active");
                }
                assert_eq!(
                    k.netlist().expect("netlist").net_names,
                    flat.net_names,
                    "{label}: the facade flattens to the same net numbering"
                );
                (b, k)
            })
            .collect();
        let mut timing = Kernel::new(kopts(FastBackend::Reference));
        timing
            .load(design.clone())
            .unwrap_or_else(|e| panic!("{label}: load: {e:?}"));
        let mut c = Chain {
            label,
            flat,
            fr,
            sims: vec![("interpreter", interp), ("jit", jit)],
            facade,
            timing,
            fast_diags: Vec::new(),
            timing_diags: Vec::new(),
            events: 0,
        };
        c.check_fast("power-on");
        c.check_facade("power-on");
        c
    }

    /// FastSim (both tiers) = FastRef after an event: values, time, diagnostic kinds, counters,
    /// internal invariants.
    fn check_fast(&mut self, what: &str) {
        let label = &self.label;
        let fr_diags = self.fr.take_diagnostics();
        let want_kinds = kinds(&fr_diags);
        let tick = self.fr.tick();
        self.fast_diags.extend(fr_diags.into_iter().map(|d| Timed {
            tick: harness::diag_tick(&d).unwrap_or(tick),
            diag: d,
        }));
        let fs = self.fr.stats();
        for (name, sim) in &mut self.sims {
            let ctx = format!(
                "{label}: {name} vs FastRef after {what} (event {})",
                self.events
            );
            if let Some(d) = first_diff(&self.flat, self.fr.values(), sim.values()) {
                panic!("{ctx}: {d}");
            }
            assert_eq!(sim.tick(), self.fr.tick(), "{ctx}: tick");
            assert_eq!(
                sim.next_event_tick(),
                self.fr.next_event_tick(),
                "{ctx}: next event"
            );
            assert_eq!(
                kinds(&sim.take_diagnostics()),
                want_kinds,
                "{ctx}: diagnostic kinds"
            );
            let ss = sim.stats();
            assert_eq!(
                (
                    ss.events,
                    ss.rounds,
                    ss.max_rounds,
                    ss.transitions,
                    ss.table_hits,
                    ss.sim_fallbacks
                ),
                (
                    fs.events,
                    fs.rounds,
                    fs.max_rounds,
                    fs.transitions,
                    fs.table_hits,
                    fs.sim_fallbacks
                ),
                "{ctx}: counters (events, rounds, max rounds, transitions, table hits, fallbacks)"
            );
            if let Err(e) = sim.check_invariants() {
                panic!("{ctx}: invariants: {e}");
            }
        }
    }

    /// Facade tick FastRef corresponds to (SPEC §6.3: the tick before the next event).
    fn fast_tick(&self) -> u64 {
        match self.fr.next_event_tick() {
            Some(n) => (n - 1).max(self.fr.tick()),
            None => self.fr.tick(),
        }
    }

    /// Every facade Fast backend = FastRef (values and the facade clock).
    fn check_facade(&mut self, what: &str) {
        let tick = self.fast_tick();
        for (b, k) in &mut self.facade {
            let ctx = format!(
                "{}: facade {b:?} vs FastRef after {what} (event {})",
                self.label, self.events
            );
            assert_eq!(
                k.tick(),
                tick,
                "{ctx}: tick (FastRef event @{}, next event {:?})",
                self.fr.tick(),
                self.fr.next_event_tick()
            );
            if let Some(d) = first_diff(&self.flat, self.fr.values(), k.snapshot().values()) {
                panic!("{ctx}: {d}");
            }
            if *b == FastBackend::Jit {
                assert!(k.jit_active(), "{ctx}: the JIT tier stays active");
            }
        }
    }

    /// Give an input change to every engine (Fast engines queue it; Timing applies it at the next
    /// tick).
    fn set_input(&mut self, pin: &str, v: &Value) {
        self.fr.set_input(pin, v).expect("FastRef input");
        for (_, s) in &mut self.sims {
            s.set_input(pin, v).expect("FastSim input");
        }
        for (_, k) in &mut self.facade {
            k.set_input(pin, v).expect("facade input");
        }
        self.timing.set_input(pin, v).expect("Timing input");
    }

    /// The Fast engines process the queued input changes as one event (facade: without
    /// advancing its clock).
    fn input_event(&mut self) {
        self.fr.step_event();
        for (_, s) in &mut self.sims {
            s.step_event();
        }
        self.events += 1;
        self.check_fast("input event");
        for (_, k) in &mut self.facade {
            k.step(0, StepUnit::Tick);
        }
        self.check_facade("input event");
    }

    /// The Fast engines process the next scheduled event (facade: one phase).
    fn clock_event(&mut self) {
        self.fr.step_event();
        for (_, s) in &mut self.sims {
            s.step_event();
        }
        self.events += 1;
        self.check_fast("clock event");
        let last = self.fr.next_event_tick().is_none();
        for (_, k) in &mut self.facade {
            k.step(1, StepUnit::Phase);
            if last {
                // Known facade bug (see `facade_fast_phase_step_processes_the_last_event`):
                // with no source change after the next event, `step(1, Phase)` in Fast mode does
                // not process that event. `run_to` does.
                k.run_to(self.fr.tick());
            }
        }
        self.check_facade("clock event");
    }

    /// Move the Timing kernel's diagnostics into `timing_diags`.
    fn drain_timing(&mut self) {
        let tick = self.timing.tick();
        let d: Vec<Diagnostic> = self.timing.diagnostics().to_vec();
        self.timing.clear_diagnostics();
        self.timing_diags.extend(d.into_iter().map(|d| Timed {
            tick: harness::diag_tick(&d).unwrap_or(tick),
            diag: d,
        }));
    }

    /// Compare the Timing kernel with FastRef: `(differing nets, all of them involve X/Z)`.
    fn timing_diff(&self) -> (usize, bool) {
        let t = self.timing.snapshot();
        let (mut n, mut unknown) = (0, true);
        for (&f, &v) in self.fr.values().iter().zip(t.values()) {
            if f != v {
                n += 1;
                unknown &= !f.is_known() || !v.is_known();
            }
        }
        (n, unknown)
    }
}

/// Stimulus and timing parameters of a run.
struct RunPlan {
    phases: usize,
    /// Ticks after a clock event before inputs may change.
    input_margin: u64,
    /// Settle bound of an input event.
    input_settle: u64,
    /// Comparisons with Timing mode count from this tick on (after the reset release).
    compare_from: u64,
    /// Percent of long-enough phases with an input change.
    input_pct: u32,
    seed: u64,
}

/// A Timing-vs-Fast disagreement.
#[derive(Debug)]
struct TimingMismatch {
    tick: u64,
    nets: usize,
    only_unknown: bool,
    first: Option<String>,
}

/// Result of [`drive`].
#[derive(Debug, Default)]
struct RunResult {
    /// Phase ends (and pre-input points) compared with Timing mode at or after `compare_from`.
    timing_compared: usize,
    timing_mismatches: Vec<TimingMismatch>,
    input_events: usize,
    events: u64,
}

/// Drive every engine of `c` through `plan` (see the module docs).
fn drive(c: &mut Chain, plan: &RunPlan) -> RunResult {
    let mut rng = Rng::new(plan.seed ^ 0xE2E_0000_C4A1);
    let inputs: Vec<(String, u32)> = c
        .flat
        .top_inputs
        .iter()
        .map(|p| (p.name.clone(), p.nets.len() as u32))
        .collect();
    let mut res = RunResult::default();
    let random_changes = |rng: &mut Rng| -> Vec<(String, Value)> {
        let mut out = Vec::new();
        for (n, w) in &inputs {
            if rng.chance(50) {
                let mask = if *w >= 64 { u64::MAX } else { (1u64 << w) - 1 };
                out.push((n.clone(), Value::from_u64(*w, rng.next_u64() & mask)));
            }
        }
        out
    };
    let compare = |c: &mut Chain, res: &mut RunResult| {
        c.drain_timing();
        let tick = c.timing.tick();
        if tick < plan.compare_from {
            return;
        }
        res.timing_compared += 1;
        let (nets, only_unknown) = c.timing_diff();
        if nets > 0 {
            res.timing_mismatches.push(TimingMismatch {
                tick,
                nets,
                only_unknown,
                first: first_diff(&c.flat, c.timing.snapshot().values(), c.fr.values()),
            });
        }
    };
    for _ in 0..plan.phases {
        let e = c.fr.tick();
        let Some(next) = c.fr.next_event_tick() else {
            // No scheduled event left (clockless after the reset release): settle Timing mode,
            // compare, then an input event.
            if inputs.is_empty() {
                break;
            }
            c.timing.step(1, StepUnit::Phase);
            compare(c, &mut res);
            let changes = loop {
                let ch = random_changes(&mut rng);
                if !ch.is_empty() {
                    break ch;
                }
            };
            for (p, v) in &changes {
                c.set_input(p, v);
            }
            if c.fr.has_pending_inputs() {
                res.input_events += 1;
                c.input_event();
            }
            continue;
        };
        let room = next - e;
        let window = plan.input_margin + plan.input_settle + 1;
        if !inputs.is_empty() && room > window && rng.chance(plan.input_pct) {
            let slack = room - window;
            let margin = plan.input_margin + rng.next_u64() % (slack + 1);
            c.timing.run_to(e + margin - 1);
            compare(c, &mut res);
            for (p, v) in &random_changes(&mut rng) {
                c.set_input(p, v);
            }
            if c.fr.has_pending_inputs() {
                res.input_events += 1;
                c.input_event();
            }
        }
        c.timing.run_to(next - 1);
        compare(c, &mut res);
        c.clock_event();
    }
    res.events = c.events;
    res
}

/// Why a Timing-vs-Fast mismatch is explained (SPEC §6.3 conditions as diagnosed at runtime), or
/// `None` if it is not.
fn explanation(c: &Chain, plan: &RunPlan, m: &TimingMismatch) -> Option<String> {
    let window = |t: &&Timed| t.tick >= plan.compare_from && t.tick <= m.tick;
    if let Some(t) = c
        .fast_diags
        .iter()
        .filter(window)
        .find(|t| RUNTIME_FLAGS.contains(&t.diag.kind))
    {
        return Some(format!("FastRef {:?}", t.diag.kind));
    }
    if let Some(t) = c.timing_diags.iter().filter(window).find(|t| {
        t.diag.severity != Severity::Info
            && matches!(t.diag.kind, DiagKind::Metastable | DiagKind::Oscillating)
    }) {
        return Some(format!("Timing {:?}", t.diag.kind));
    }
    let contention = c
        .fast_diags
        .iter()
        .chain(&c.timing_diags)
        .filter(window)
        .any(|t| t.diag.kind == DiagKind::Contention);
    (m.only_unknown && contention).then(|| "Contention (X only)".to_string())
}

/// Random designs: harness classification (raw TimingSim vs FastRef) plus the lockstep chain.
fn random_designs(generator: GenConfig, seeds: std::ops::Range<u64>, phases: usize) {
    let cfg = HarnessConfig {
        generator,
        // Glass-box Timing mode only (ideal blocks are covered by the harness's own tests).
        ideal_pct: 0,
        ..HarnessConfig::default()
    };
    let (mut chained, mut clean, mut flagged, mut skipped) = (0, 0, 0, 0);
    let (mut events, mut timing_compared, mut input_events) = (0u64, 0usize, 0usize);
    let mut explained: Vec<String> = Vec::new();
    let mut pessimism_xsettle = 0;
    for seed in seeds {
        let p = match harness::prepare(seed, &cfg) {
            Ok(p) => p,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        let label = format!("seed {seed}");
        let flat = p.flat.clone();
        let design = p.generated.design.clone();
        let opts = p.fast.options().clone();
        let sta_clean = p.report.errors().next().is_none();
        let plan = RunPlan {
            phases,
            input_margin: p.plan.input_margin,
            input_settle: p.plan.input_settle,
            compare_from: p.plan.compare_from,
            input_pct: 50,
            seed,
        };
        let generated = p.generated.clone();
        // 1. The harness pipeline (raw TimingSim vs FastRef, random initial inputs).
        let mut o = harness::finish(p, &cfg, &mut |_, _, _, _| {});
        harness::recheck(&generated, &mut o, &cfg);
        assert_ne!(
            o.class,
            Class::CleanMismatch,
            "{label}: harness clean-but-mismatch: {:?} {:?}",
            o.diff
                .as_ref()
                .and_then(|d| d.first())
                .map(ToString::to_string),
            o.notes
        );
        // 2. The lockstep chain.
        let mut c = Chain::new(label.clone(), &design, flat, &opts);
        let r = drive(&mut c, &plan);
        chained += 1;
        events += r.events;
        timing_compared += r.timing_compared;
        input_events += r.input_events;
        if !sta_clean {
            flagged += 1;
            continue;
        }
        clean += 1;
        if let Some(m) = r.timing_mismatches.first() {
            match explanation(&c, &plan, m) {
                Some(why) => explained.push(format!("{label}: {why}")),
                // STA reasons about binary values: a persistent X (state never reset) can take
                // longer to settle than STA's bound. The harness's `recheck` documents this case
                // ("XSettle"); it is accepted only when every disagreeing net involves X/Z.
                None if m.only_unknown => pessimism_xsettle += 1,
                None => panic!(
                    "{label}: STA-clean, no runtime diagnostic, but Timing != Fast at tick {} \
                     ({} nets; Timing vs FastRef: {:?}); {} more mismatching points",
                    m.tick,
                    m.nets,
                    m.first,
                    r.timing_mismatches.len() - 1
                ),
            }
        }
    }
    println!(
        "{chained} designs chained ({clean} STA-clean, {flagged} flagged, {skipped} skipped); \
         {events} Fast events bit-exact across FastRef/FastSim/JIT and the 3 facade backends; \
         {timing_compared} Timing comparisons, {input_events} input events; \
         explained Timing mismatches: {explained:?}; X-only unexplained: {pessimism_xsettle}"
    );
    assert!(chained > 0 && clean > 0, "nothing was compared");
    assert!(
        pessimism_xsettle * 10 <= clean.max(10),
        "too many X-only Timing mismatches on clean designs: {pessimism_xsettle} of {clean}"
    );
}

/// Synchronous designs (no asynchronous structures): most of them are STA-clean, so the Timing
/// leg of the chain is exercised. 10% are clockless (reset release and input events only).
fn synchronous() -> GenConfig {
    GenConfig {
        async_pct: 0,
        clockless_pct: 10,
        ..GenConfig::default()
    }
}

#[test]
fn random_designs_full_chain() {
    random_designs(GenConfig::default(), 0..60, 60);
}

#[test]
fn random_synchronous_designs_full_chain() {
    random_designs(synchronous(), 0..80, 60);
}

#[test]
#[ignore = "long: 1000 seeds × 120 phases"]
fn random_designs_full_chain_long() {
    random_designs(GenConfig::default(), 1000..1500, 120);
    random_designs(synchronous(), 1000..1500, 120);
}

/// Panda-8 `variant`/`rom` at its STA minimum period: the whole chain for `cycles` cycles, and
/// FastRef's `OUT` sequence equals the behavioral emulator's.
fn panda8_chain(variant: Variant, rom: RomStyle, program: &programs::Program, cycles: u64) {
    let image = program.image();
    let base = Options::new(variant).with_rom(rom);
    let flat0 = flatten(&panda8_with(&image, &base)).expect("flatten");
    let (ks, tables) = compile_kernels(&flat0, &CompileOptions::default());
    let min = panda_sta::analyze(&flat0, &ks, &tables)
        .max_min_period()
        .expect("min period");
    let period = min.max(8).next_multiple_of(2);
    let opts = base.with_period(period);
    let design = panda8_with(&image, &opts);
    let flat = Arc::new(flatten(&design).expect("flatten"));
    let (ks, tables) = compile_kernels(&flat, &CompileOptions::default());
    let report = panda_sta::analyze(&flat, &ks, &tables);
    let label = format!("Panda-8 {variant:?} {rom:?} {} @ {period}", program.name);
    assert!(report.errors().next().is_none(), "{label}: STA errors");
    let mut c = Chain::new(label.clone(), &design, flat, &FastOptions::default());
    let reset = u64::from(opts.reset_ticks());
    let plan = RunPlan {
        phases: 2 * (reset / u64::from(period) + 2) as usize + 2 * cycles as usize,
        input_margin: 1,
        input_settle: 0,
        compare_from: reset,
        input_pct: 0,
        seed: 0,
    };
    let r = drive(&mut c, &plan);
    if let Some(m) = r.timing_mismatches.first() {
        panic!("{label}: Timing != Fast: {m:?}");
    }
    assert!(
        r.timing_compared >= 2 * cycles as usize,
        "{label}: compared {}",
        r.timing_compared
    );
    println!(
        "{label}: {} events bit-exact, {} Timing comparisons",
        r.events, r.timing_compared
    );
}

/// OUT values strobed per cycle in the facade (Jit backend), compared with the emulator.
fn panda8_outputs_via_facade(
    variant: Variant,
    rom: RomStyle,
    program: &programs::Program,
    cycles: u64,
) {
    let image = program.image();
    let opts = Options::new(variant).with_rom(rom);
    let design = panda8_with(&image, &opts);
    let mut k = Kernel::new(KernelOptions {
        background: false,
        ..KernelOptions::default()
    });
    k.set_mode(Mode::Fast);
    k.load(design).expect("load");
    k.wait_ready().expect("compile");
    assert!(k.jit_active(), "JIT active");
    // Sample once per cycle (the tick before a rising edge), from the reset release on.
    let release = u64::from(opts.reset_ticks());
    let mut emu = Emulator::new(&image);
    let (mut outs, mut golden) = (Vec::new(), Vec::new());
    let mut counted = 0;
    while counted < cycles {
        k.step(1, StepUnit::Cycle);
        if k.tick() < release {
            continue;
        }
        counted += 1;
        if k.read_u64("OUT_STROBE") == Some(1) {
            outs.push(k.read_u64("OUT").expect("OUT binary") as u8);
        }
        if let Some(o) = emu.step().out {
            golden.push(o);
        }
    }
    assert!(!golden.is_empty());
    assert_eq!(
        outs, golden,
        "{variant:?} {rom:?} {}: OUT sequence via the facade",
        program.name
    );
}

#[test]
fn panda8_full_chain_every_variant() {
    for variant in Variant::ALL {
        for rom in [RomStyle::SumOfProducts, RomStyle::MuxTree] {
            panda8_chain(variant, rom, &programs::FIBONACCI, 150);
        }
    }
}

#[test]
fn panda8_outputs_through_the_facade_jit() {
    for variant in Variant::ALL {
        panda8_outputs_via_facade(variant, RomStyle::SumOfProducts, &programs::FIBONACCI, 400);
    }
    panda8_outputs_via_facade(
        Variant::Hierarchical,
        RomStyle::MuxTree,
        &programs::MULTIPLY,
        600,
    );
}

#[test]
#[ignore = "long: every program, 1000 cycles, every variant and ROM style"]
fn panda8_full_chain_long() {
    for p in programs::ALL {
        for variant in Variant::ALL {
            for rom in [
                RomStyle::SumOfProducts,
                RomStyle::MuxTree,
                RomStyle::Decoder,
            ] {
                panda8_chain(variant, rom, &p, 1000);
            }
        }
    }
}

/// Facade Fast mode, `step(1, Phase)` when the next event is the last one (a clockless design's
/// `PowerOnReset` release): the facade must process it, like Timing mode does (SPEC §10 step
/// semantics; panda-kernel docs "Time": "Without any future source change, `Phase` and `Cycle`
/// let pending input changes settle").
#[test]
#[ignore = "known bug in panda-kernel (advance_fast ignores Target::Settle); reported by the audit"]
fn facade_fast_phase_step_processes_the_last_event() {
    use panda_netlist::{DefBuilder, DesignBuilder};
    let mut db = DesignBuilder::new();
    let mut b = DefBuilder::new("por_only");
    let rst = b.power_on_reset(10);
    let n = b.not(rst);
    b.output("Q", n);
    let top = db.add(b.finish());
    let design = db.build(&top);
    for mode in [Mode::Timing, Mode::Fast] {
        for backend in BACKENDS {
            let mut k = Kernel::new(KernelOptions {
                background: false,
                backend,
                ..KernelOptions::default()
            });
            k.set_mode(mode);
            k.load(design.clone()).expect("load");
            k.wait_ready().expect("ready");
            k.step(1, StepUnit::Phase); // to the end of the reset phase (tick 9)
            k.step(1, StepUnit::Phase); // through the release: Q = 1
            assert_eq!(
                k.read_u64("Q"),
                Some(1),
                "{mode:?} {backend:?} at tick {}",
                k.tick()
            );
        }
    }
}
