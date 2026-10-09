//! Shared helpers of the FastSim-vs-FastRef differential tests.
#![allow(dead_code)]

use std::sync::Arc;

use panda_compile::{FastOptions, FastRef, KernelOptions, compile_kernels};
use panda_difftest::Rng;
use panda_fast::{CompileOptions, FastSim, FastSimOptions, OrderStrategy};
use panda_netlist::flat::FlatNetlist;
use panda_netlist::{Diagnostic, Logic, Value};

/// FastRef and FastSim over the same kernel set and tables, both powered on.
pub fn pair(
    flat: &Arc<FlatNetlist>,
    fast: &FastOptions,
    compile: &CompileOptions,
) -> (FastRef, FastSim) {
    let kopts = KernelOptions {
        extract: fast.extract.clone(),
        charz: fast.charz.clone(),
    };
    let (ks, tables) = compile_kernels(flat, &kopts);
    let fr = FastRef::from_parts(flat.clone(), fast.clone(), ks.clone(), tables.clone())
        .expect("FastRef");
    let opts = FastSimOptions {
        fast: fast.clone(),
        compile: compile.clone(),
    };
    let fs = FastSim::from_parts(flat.clone(), opts, ks, tables).expect("FastSim");
    (fr, fs)
}

fn kinds(d: &[Diagnostic]) -> Vec<String> {
    let mut v: Vec<String> = d.iter().map(|d| format!("{:?}", d.kind)).collect();
    v.sort();
    v
}

/// Diagnostic kinds seen (for coverage accounting).
#[derive(Default, Debug)]
pub struct Coverage {
    pub events: u64,
    pub diag_kinds: std::collections::BTreeMap<String, u64>,
    pub sim_fallbacks: u64,
    pub table_hits: u64,
    pub max_rounds: u32,
    /// FastSim transitions by the KOpaque paths (max over the run's engines).
    pub fast_opaque: u64,
    /// FastSim transitions by the KTransparent paths.
    pub fast_transparent: u64,
    /// FastSim fast-path transitions needing per-kernel work.
    pub fast_evals: u64,
    /// LUTs in the programs (sum over designs).
    pub luts: u64,
}

impl Coverage {
    /// Fold another run's coverage in.
    pub fn add(&mut self, c: &Coverage) {
        self.events += c.events;
        for (k, n) in &c.diag_kinds {
            *self.diag_kinds.entry(k.clone()).or_default() += n;
        }
        self.sim_fallbacks += c.sim_fallbacks;
        self.table_hits += c.table_hits;
        self.max_rounds = self.max_rounds.max(c.max_rounds);
        self.fast_opaque += c.fast_opaque;
        self.fast_transparent += c.fast_transparent;
        self.fast_evals += c.fast_evals;
        self.luts += c.luts;
    }
}

/// Compare everything observable after an event; drains both engines' diagnostics.
pub fn check(
    fr: &mut FastRef,
    fs: &mut FastSim,
    flat: &FlatNetlist,
    ctx: &str,
    cov: &mut Coverage,
) {
    if let Err(m) = fs.check_invariants() {
        panic!("{ctx}: FastSim invariant (tick {}): {m}", fs.tick());
    }
    if fr.values() != fs.values() {
        let diffs: Vec<String> = (0..flat.num_nets as usize)
            .filter(|&n| fr.values()[n] != fs.values()[n])
            .take(12)
            .map(|n| {
                format!(
                    "{}: ref {} fast {}",
                    flat.net_names[n],
                    fr.values()[n],
                    fs.values()[n]
                )
            })
            .collect();
        panic!("{ctx}: values differ (tick {}): {diffs:?}", fr.tick());
    }
    assert_eq!(fr.tick(), fs.tick(), "{ctx}: tick");
    assert_eq!(
        fr.next_event_tick(),
        fs.next_event_tick(),
        "{ctx}: next_event_tick"
    );
    assert_eq!(
        fr.has_pending_inputs(),
        fs.has_pending_inputs(),
        "{ctx}: pending"
    );
    let (a, b) = (fr.stats(), fs.stats());
    assert_eq!(
        (
            a.events,
            a.rounds,
            a.max_rounds,
            a.transitions,
            a.table_hits,
            a.sim_fallbacks
        ),
        (
            b.events,
            b.rounds,
            b.max_rounds,
            b.transitions,
            b.table_hits,
            b.sim_fallbacks
        ),
        "{ctx}: stats (events, rounds, max_rounds, transitions, table hits, sim fallbacks)"
    );
    let (da, db) = (fr.take_diagnostics(), fs.take_diagnostics());
    let (ka, kb) = (kinds(&da), kinds(&db));
    assert_eq!(
        ka, kb,
        "{ctx}: diagnostic kinds\nref: {da:#?}\nfast: {db:#?}"
    );
    cov.events += 1;
    for k in ka {
        *cov.diag_kinds.entry(k).or_default() += 1;
    }
    cov.sim_fallbacks = cov.sim_fallbacks.max(b.sim_fallbacks);
    cov.table_hits = cov.table_hits.max(b.table_hits);
    cov.max_rounds = cov.max_rounds.max(b.max_rounds);
    cov.fast_opaque = cov.fast_opaque.max(b.opaque_skips);
    cov.fast_transparent = cov.fast_transparent.max(b.transparent);
    cov.fast_evals = cov.fast_evals.max(b.fast_evals);
}

/// A random value for a `width`-bit pin: usually binary, sometimes with X/Z bits.
pub fn random_value(rng: &mut Rng, width: u32, xz_pct: u32) -> Value {
    if rng.chance(xz_pct) {
        let bits: Vec<Logic> = (0..width)
            .map(|_| *rng.pick(&[Logic::Zero, Logic::One, Logic::X, Logic::Z]))
            .collect();
        Value::from_bits(&bits)
    } else {
        let v = rng.next_u64();
        let v = if width >= 64 {
            v
        } else {
            v & ((1u64 << width) - 1)
        };
        Value::from_u64(width, v)
    }
}

/// Drive both engines with the same random sequence of `actions`: scheduled events, input
/// changes (processed as their own event), occasional resets; compare after every event.
#[allow(clippy::too_many_arguments)]
pub fn random_lockstep(
    fr: &mut FastRef,
    fs: &mut FastSim,
    flat: &Arc<FlatNetlist>,
    rng: &mut Rng,
    actions: usize,
    xz_pct: u32,
    ctx: &str,
    cov: &mut Coverage,
) {
    cov.luts += fs.program().luts.len() as u64;
    check(fr, fs, flat, &format!("{ctx} power-on"), cov);
    let pins: Vec<(String, u32)> = flat
        .top_inputs
        .iter()
        .map(|p| (p.name.clone(), p.nets.len() as u32))
        .collect();
    for i in 0..actions {
        let roll = rng.below(100);
        if roll < 30 && !pins.is_empty() {
            // One or more pin changes, then the input event.
            for _ in 0..1 + rng.below(2) {
                let (name, w) = rng.pick(&pins).clone();
                let v = random_value(rng, w, xz_pct);
                fr.set_input(&name, &v).unwrap();
                fs.set_input(&name, &v).unwrap();
            }
            assert_eq!(
                fr.step_event(),
                fs.step_event(),
                "{ctx} action {i}: step result"
            );
        } else if roll < 31 {
            fr.reset();
            fs.reset();
        } else {
            let a = fr.step_event();
            let b = fs.step_event();
            assert_eq!(a, b, "{ctx} action {i}: step result");
        }
        check(fr, fs, flat, &format!("{ctx} action {i}"), cov);
    }
}

/// Random compiler options: partition sizes, both orders, LUT extraction from tiny to default
/// thresholds (so small random designs get LUTs), kernel specialization on/off.
pub fn random_compile(rng: &mut Rng, partition_choices: &[usize]) -> CompileOptions {
    let mut c = CompileOptions {
        partition_ops: *rng.pick(partition_choices),
        order: if rng.chance(50) {
            OrderStrategy::ConeDfs
        } else {
            OrderStrategy::KernelSet
        },
        ..CompileOptions::default()
    };
    match rng.below(4) {
        0 => {}
        1 => c.lut_max_bits = 0,
        _ => {
            c.lut_max_bits = *rng.pick(&[1, 2, 3, 4, 6, 8, 12]);
            c.lut_min_gates = *rng.pick(&[1, 2, 3, 5, 10]);
        }
    }
    if rng.chance(15) {
        c.lut_max_entries = *rng.pick(&[1, 16, 256]);
    }
    c.lut_weight = *rng.pick(&[1, 4, 16]);
    c.specialize = !rng.chance(20);
    c.fuse_inputs = *rng.pick(&[0, 2, 3, 4, 5, 6, 6, 6]);
    c.data_order_lemma = !rng.chance(20);
    c
}

/// Characterization options that produce partial / budget-stopped order checks, sampled orders,
/// tight settle bounds, missing tables and other period-2 thresholds.
pub fn randomize_charz(rng: &mut Rng, fast: &mut FastOptions) {
    let c = &mut fast.charz;
    if rng.chance(30) {
        c.max_order_work = *rng.pick(&[0, 1, 10, 100, 1000]);
    }
    if rng.chance(20) {
        c.exhaustive_order_max = *rng.pick(&[0, 1, 2, 3]);
        c.order_samples = *rng.pick(&[0, 1, 2, 5]);
    }
    if rng.chance(15) {
        c.settle_bound = Some(*rng.pick(&[1, 2, 4, 8, 20]));
    }
    if rng.chance(10) {
        c.max_table_entries = *rng.pick(&[0, 4, 64, 1000]);
    }
    if rng.chance(15) {
        c.timing.p2_limit = *rng.pick(&[0, 1, 2, 8]);
    }
}

/// Compact netlist + kernel/program dump (debugging).
pub fn dump(flat: &FlatNetlist, fs: &FastSim) {
    let nm = |n: u32| flat.net_names[n as usize].clone();
    for g in &flat.gates {
        let ins: Vec<String> = g.inputs.iter().map(|&n| nm(n)).collect();
        eprintln!("gate {:?}({}) -> {}", g.op, ins.join(","), nm(g.output));
    }
    for r in &flat.resolvers {
        let ins: Vec<String> = r.drivers.iter().map(|&n| nm(n)).collect();
        eprintln!(
            "resolver {:?}({}) -> {}",
            r.termination,
            ins.join(","),
            nm(r.output)
        );
    }
    for s in &flat.sources {
        eprintln!("source {:?} -> {}", s.kind, nm(s.output));
    }
    let p = fs.program();
    for k in &fs.kernels().kernels {
        let ins: Vec<String> = k.inputs.iter().map(|&n| nm(n)).collect();
        let nets: Vec<String> = k.nets.iter().map(|&n| nm(n)).collect();
        let classes: Vec<String> = (0..p.num_tuples)
            .map(|t| format!("{:?}", p.phase_class(k.id, t)))
            .collect();
        eprintln!(
            "kernel {} {:?} in [{}] nets [{}] class {} phases {:?}",
            k.id,
            k.kind,
            ins.join(","),
            nets.join(","),
            p.kernel_class[k.id as usize],
            classes
        );
    }
    eprintln!(
        "timed {:?}",
        p.timed_nets.iter().map(|&n| nm(n)).collect::<Vec<_>>()
    );
}
