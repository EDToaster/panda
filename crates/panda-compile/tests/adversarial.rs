//! Adversarial tests of kernel extraction, characterization and FastRef (review findings and
//! randomized invariants).
//!
//! Invariants checked on random raw netlists (arbitrary feedback, tri-states, resolvers):
//! * the kernel set is structurally valid ([`KernelSet::verify`]);
//! * FastRef with tables and FastRef with every table removed (pure on-the-fly transitions, same
//!   kernel set) publish bit-identical values after every event — the table path is an exact cache
//!   of `T`, including bit order, state ids, X fallback and return to table mode;
//! * the published values are a fixed point of every gate and resolver (a settled state), unless
//!   a kernel was forced to `X` (`Oscillating` / `IterationLimit`).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use panda_compile::charz::{
    CharzOptions, KernelSim, KernelTable, TableCache, characterize, input_vector,
};
use panda_compile::kernels::{ExtractOptions, KernelKind, extract, extract_with_stats};
use panda_compile::{FastOptions, FastRef, KernelOptions, compile_kernels};
use panda_netlist::eval::{eval_gate, resolve};
use panda_netlist::flat::{
    FlatGate, FlatInstance, FlatNetlist, FlatOp, FlatPin, GateGroup, NetIdx, Resolver, Source,
    SourceKind,
};
use panda_netlist::{DiagKind, Dir, Logic, Termination, Value};
use panda_timing::{TimingOptions, TimingSim};

// ---------------------------------------------------------------------------------------------
// Raw netlist builder
// ---------------------------------------------------------------------------------------------

#[derive(Default)]
struct Raw {
    flat: FlatNetlist,
}

impl Raw {
    fn new() -> Raw {
        let mut r = Raw::default();
        r.flat.instances.push(FlatInstance {
            path: String::new(),
            def: "top".into(),
            parent: None,
            children: Vec::new(),
            pins: Vec::new(),
        });
        r
    }

    fn net(&mut self, name: &str) -> NetIdx {
        let n = self.flat.num_nets;
        self.flat.num_nets += 1;
        self.flat.net_names.push(name.to_string());
        self.flat.name_index.insert(name.to_string(), n);
        n
    }

    fn input(&mut self, name: &str) -> NetIdx {
        let n = self.net(name);
        let pin = self.flat.top_inputs.len() as u32;
        self.flat.sources.push(Source {
            kind: SourceKind::TopInput { pin, bit: 0 },
            output: n,
        });
        self.flat.top_inputs.push(FlatPin {
            name: name.to_string(),
            dir: Dir::In,
            nets: vec![n],
        });
        n
    }

    fn source(&mut self, name: &str, kind: SourceKind) -> NetIdx {
        let n = self.net(name);
        self.flat.sources.push(Source { kind, output: n });
        n
    }

    fn output(&mut self, name: &str, net: NetIdx) {
        self.flat.top_outputs.push(FlatPin {
            name: name.to_string(),
            dir: Dir::Out,
            nets: vec![net],
        });
    }

    fn gate_to(&mut self, op: FlatOp, inputs: &[NetIdx], out: NetIdx) {
        let gi = self.flat.gates.len() as u32;
        let group = self.flat.groups.len() as u32;
        self.flat.gates.push(FlatGate {
            op,
            inputs: inputs.to_vec(),
            output: out,
            owner: 0,
            group,
            bit: 0,
        });
        self.flat.groups.push(GateGroup {
            owner: 0,
            path: format!("g{gi}"),
            gates: vec![gi],
        });
    }

    fn gate(&mut self, name: &str, op: FlatOp, inputs: &[NetIdx]) -> NetIdx {
        let n = self.net(name);
        self.gate_to(op, inputs, n);
        n
    }

    fn resolver(&mut self, name: &str, drivers: &[NetIdx], term: Termination) -> NetIdx {
        let n = self.net(name);
        self.flat.resolvers.push(Resolver {
            drivers: drivers.to_vec(),
            termination: term,
            output: n,
        });
        n
    }

    fn done(self) -> Arc<FlatNetlist> {
        Arc::new(self.flat)
    }
}

/// Tiny deterministic PRNG.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut x = self.0;
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^= x >> 33;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn chance(&mut self, num: u64, den: u64) -> bool {
        self.below(den) < num
    }
}

const OPS: [FlatOp; 8] = [
    FlatOp::And,
    FlatOp::Or,
    FlatOp::Nand,
    FlatOp::Nor,
    FlatOp::Xor,
    FlatOp::Xnor,
    FlatOp::Not,
    FlatOp::Buf,
];

/// A random netlist with arbitrary feedback: `ni` inputs, `ng` gates whose inputs are any nets
/// (so loops of every length appear), optionally tri-state drivers merged by resolvers with a
/// random termination, and optionally a clock.
fn random_netlist(seed: u64, ni: usize, ng: usize, tri: bool, clock: bool) -> Arc<FlatNetlist> {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678);
    let mut r = Raw::new();
    let mut ins = Vec::new();
    for i in 0..ni {
        ins.push(r.input(&format!("i{i}")));
    }
    if clock {
        let period = 2 * (2 + rng.below(6)) as u32;
        let high = 1 + rng.below(u64::from(period) - 1) as u32;
        let phase = rng.below(u64::from(period)) as u32;
        ins.push(r.source(
            "clk",
            SourceKind::Clock {
                period,
                high,
                phase,
            },
        ));
    }
    // Pre-allocate gate output nets so inputs can refer forward (feedback).
    let outs: Vec<NetIdx> = (0..ng).map(|g| r.net(&format!("n{g}"))).collect();
    let mut pool: Vec<NetIdx> = ins.clone();
    pool.extend(&outs);
    // Optional resolver nets (driven by tri-states created below).
    let mut res_nets: Vec<(NetIdx, Termination)> = Vec::new();
    if tri {
        for b in 0..1 + rng.below(2) {
            let term = match rng.below(4) {
                0 => Termination::None,
                1 => Termination::PullUp,
                2 => Termination::PullDown,
                _ => Termination::Keeper,
            };
            let n = r.net(&format!("bus{b}"));
            res_nets.push((n, term));
            pool.push(n);
        }
    }
    let pick = |rng: &mut Rng, pool: &[NetIdx]| pool[rng.below(pool.len() as u64) as usize];
    for &out in &outs {
        let op = OPS[rng.below(OPS.len() as u64) as usize];
        let arity = match op {
            FlatOp::Not | FlatOp::Buf => 1,
            _ => 2 + usize::from(rng.chance(1, 4)),
        };
        // Bias towards local feedback: half the time pick among nearby gates.
        let inputs: Vec<NetIdx> = (0..arity)
            .map(|_| {
                if rng.chance(1, 2) {
                    let lo = (out as usize).saturating_sub(4);
                    let hi = (out as usize + 4).min(r.flat.num_nets as usize - 1);
                    let cand = lo as u64 + rng.below((hi - lo + 1) as u64);
                    let cand = cand as NetIdx;
                    if pool.contains(&cand) {
                        cand
                    } else {
                        pick(&mut rng, &pool)
                    }
                } else {
                    pick(&mut rng, &pool)
                }
            })
            .collect();
        r.gate_to(op, &inputs, out);
    }
    for (bi, &(bus, term)) in res_nets.iter().enumerate() {
        let nd = 1 + rng.below(3);
        let mut drv = Vec::new();
        for d in 0..nd {
            let dn = r.net(&format!("bus{bi}_d{d}"));
            let data = pick(&mut rng, &pool);
            let en = pick(&mut rng, &pool);
            r.gate_to(FlatOp::Tri, &[data, en], dn);
            drv.push(dn);
        }
        r.flat.resolvers.push(Resolver {
            drivers: drv,
            termination: term,
            output: bus,
        });
    }
    // A few outputs.
    for o in 0..3.min(ng) {
        let n = outs[rng.below(ng as u64) as usize];
        r.output(&format!("o{o}"), n);
    }
    r.done()
}

/// A copy of `t` keeping only the stable states with `keep(id)` (renumbered); transitions into a
/// dropped state become non-binary (`next = None`).
fn prune_table(t: &KernelTable, keep: impl Fn(u32) -> bool) -> KernelTable {
    let k = t.num_inputs;
    let mut remap = vec![None; t.num_states()];
    let mut out = KernelTable {
        num_inputs: k,
        states: Vec::new(),
        state_index: HashMap::new(),
        transitions: Vec::new(),
        states_by_input: Vec::new(),
        home_input: Vec::new(),
        order_check_partial: t.order_check_partial,
        sensitive_from: Default::default(),
        order_check_stop: None,
    };
    for s in 0..t.num_states() as u32 {
        if keep(s) {
            let id = out.states.len() as u32;
            remap[s as usize] = Some(id);
            out.states.push(t.states[s as usize].clone());
            out.state_index.insert(t.states[s as usize].clone(), id);
            out.home_input.push(t.home_input[s as usize]);
        }
    }
    for s in 0..t.num_states() as u32 {
        if remap[s as usize].is_none() {
            continue;
        }
        for v in 0..1u32 << k {
            let mut tr = *t.transition(s, v);
            tr.next = tr.next.and_then(|n| remap[n as usize]);
            out.transitions.push(tr);
        }
    }
    out.sensitive_from = t
        .sensitive_from
        .iter()
        .filter_map(|&(s, u, v)| remap[s as usize].map(|n| (n, u, v)))
        .collect();
    // Ids are renumbered monotonically: the stop moves to the first kept state at or after it.
    out.order_check_stop = t
        .order_check_stop
        .and_then(|(s, u, v)| match remap[s as usize] {
            Some(n) => Some((n, u, v)),
            None => (s as usize + 1..remap.len())
                .find_map(|o| remap[o])
                .map(|n| (n, 0, 0)),
        });
    out.states_by_input = t
        .states_by_input
        .iter()
        .map(|l| l.iter().filter_map(|&s| remap[s as usize]).collect())
        .collect();
    out
}

// ---------------------------------------------------------------------------------------------
// Invariant checkers
// ---------------------------------------------------------------------------------------------

/// Every gate output equals its function of the current values; every resolver too (a keeper
/// holds its own value when undriven). Returns the violating nets.
fn fixed_point_violations(flat: &FlatNetlist, vals: &[Logic]) -> Vec<String> {
    let mut bad = Vec::new();
    for g in &flat.gates {
        let ins: Vec<Logic> = g.inputs.iter().map(|&n| vals[n as usize]).collect();
        let want = eval_gate(g.op, &ins);
        if want != vals[g.output as usize] {
            bad.push(format!(
                "gate {:?} -> {}: have {} want {want}",
                g.op, flat.net_names[g.output as usize], vals[g.output as usize]
            ));
        }
    }
    for r in &flat.resolvers {
        let ins: Vec<Logic> = r.drivers.iter().map(|&n| vals[n as usize]).collect();
        let want = resolve(&ins, r.termination, vals[r.output as usize]).value;
        if want != vals[r.output as usize] {
            bad.push(format!(
                "resolver -> {}: have {} want {want}",
                flat.net_names[r.output as usize], vals[r.output as usize]
            ));
        }
    }
    bad
}

fn forced_x(diags: &[panda_netlist::Diagnostic]) -> bool {
    diags
        .iter()
        .any(|d| matches!(d.kind, DiagKind::Oscillating | DiagKind::IterationLimit))
}

/// Drive `flat` through random input events (and clock events) with two FastRef engines on the
/// same kernel set: one with tables, one without. Checks the invariants of the module docs.
/// Returns the number of `DelaySensitive` diagnostics raised.
fn check_design(flat: &Arc<FlatNetlist>, opts: &FastOptions, seed: u64, events: usize) -> usize {
    let kopts = KernelOptions {
        extract: opts.extract.clone(),
        charz: opts.charz.clone(),
    };
    let (ks, tables) = compile_kernels(flat, &kopts);
    ks.verify(flat)
        .unwrap_or_else(|e| panic!("seed {seed}: invalid kernel set: {e}"));
    let no_tables = vec![None; tables.len()];
    // Tables missing every other stable state: transitions into a missing state and kernels
    // sitting in one must fall back to on-the-fly simulation, never be wrong.
    let pruned: Vec<Option<Arc<KernelTable>>> = tables
        .iter()
        .map(|t| t.as_ref().map(|t| Arc::new(prune_table(t, |s| s % 2 == 1))))
        .collect();
    // Tables whose order check ran out of budget almost immediately: the engine computes the
    // order sensitivities it needs on demand and must report exactly the same diagnostics.
    let tight = {
        let mut o = opts.clone();
        o.charz.max_order_work = 20;
        o
    };
    let mut cache = TableCache::new();
    let tight_tables: Vec<_> = ks
        .kernels
        .iter()
        .map(|k| cache.get_or_characterize(flat, k, &tight.charz))
        .collect();
    let mut d = FastRef::from_parts(flat.clone(), tight, ks.clone(), tight_tables).unwrap();
    let mut a = FastRef::from_parts(flat.clone(), opts.clone(), ks.clone(), tables).unwrap();
    let mut c = FastRef::from_parts(flat.clone(), opts.clone(), ks.clone(), pruned).unwrap();
    let mut b = FastRef::from_parts(flat.clone(), opts.clone(), ks, no_tables).unwrap();
    let mut rng = Rng(seed ^ 0xABCD);
    let mut diags_a = Vec::new();
    let pins: Vec<String> = flat.top_inputs.iter().map(|p| p.name.clone()).collect();
    for ev in 0..=events {
        if ev > 0 {
            let clocked = a.next_event_tick().is_some();
            if !pins.is_empty() && (!clocked || rng.chance(1, 2)) {
                // 1–2 input changes (a 2-bit change exercises order-sensitive entries).
                for _ in 0..1 + rng.below(2) {
                    let p = &pins[rng.below(pins.len() as u64) as usize];
                    let v = match rng.below(64) {
                        0 if seed.is_multiple_of(2) => Value::from_logic(Logic::X),
                        0..=31 => Value::from_logic(Logic::One),
                        _ => Value::from_logic(Logic::Zero),
                    };
                    a.set_input(p, &v).unwrap();
                    b.set_input(p, &v).unwrap();
                    c.set_input(p, &v).unwrap();
                    d.set_input(p, &v).unwrap();
                }
            }
            let ta = a.step_event();
            let tb = b.step_event();
            c.step_event();
            d.step_event();
            assert_eq!(ta, tb, "seed {seed}: event ticks differ");
        }
        let da = a.take_diagnostics();
        let dd = d.take_diagnostics();
        assert_eq!(
            da, dd,
            "seed {seed} event {ev}: tight-budget tables change diagnostics"
        );
        assert!(a.values() == d.values());
        diags_a.extend(da);
        b.take_diagnostics();
        c.take_diagnostics();
        assert!(
            c.values() == b.values(),
            "seed {seed} event {ev}: pruned-table engine differs from on-the-fly engine"
        );
        let diffs: Vec<String> = (0..flat.num_nets as usize)
            .filter(|&n| a.values()[n] != b.values()[n])
            .map(|n| {
                format!(
                    "{}: table {} sim {}",
                    flat.net_names[n],
                    a.values()[n],
                    b.values()[n]
                )
            })
            .collect();
        assert!(
            diffs.is_empty(),
            "seed {seed} event {ev}: table path differs from on-the-fly path: {diffs:?}"
        );
        if !forced_x(&diags_a) {
            let bad = fixed_point_violations(flat, a.values());
            assert!(
                bad.is_empty(),
                "seed {seed} event {ev}: published values are not settled: {bad:?}"
            );
        }
    }
    diags_a
        .iter()
        .filter(|d| d.kind == DiagKind::DelaySensitive)
        .count()
}

// ---------------------------------------------------------------------------------------------
// Randomized invariants
// ---------------------------------------------------------------------------------------------

#[test]
fn random_feedback_netlists_invariants() {
    let mut delay = 0;
    for seed in 0..400u64 {
        let ng = 4 + (seed % 23) as usize;
        let flat = random_netlist(
            seed,
            2 + (seed % 3) as usize,
            ng,
            seed % 3 == 0,
            seed % 4 == 1,
        );
        let opts = FastOptions::default();
        delay += check_design(&flat, &opts, seed, 30);
    }
    assert!(delay > 0, "no DelaySensitive diagnostic exercised");
}

#[test]
fn random_feedback_netlists_small_kernels() {
    // Short seed loops and tiny tables: many residual Sim kernels and absorbed fan-in.
    for seed in 0..300u64 {
        let ng = 6 + (seed % 17) as usize;
        let flat = random_netlist(seed + 10_000, 3, ng, seed % 2 == 0, seed % 3 == 1);
        let mut opts = FastOptions::default();
        opts.extract.max_loop = 1 + (seed % 4) as usize;
        opts.extract.max_bits = 2 + (seed % 5) as usize;
        check_design(&flat, &opts, seed, 25);
    }
}

#[test]
fn random_feedback_netlists_big_loops() {
    for seed in 0..200u64 {
        let flat = random_netlist(seed + 20_000, 3, 30, seed % 2 == 0, seed % 2 == 1);
        let mut opts = FastOptions::default();
        opts.extract.max_loop = 10;
        opts.extract.max_bits = 14;
        check_design(&flat, &opts, seed, 20);
    }
}

/// A gated ring oscillator whose kernel absorbed a pull-up resolver driven from outside. After the
/// oscillation (kernel forced to all-X) stops, the resolver must again equal its resolved value:
/// it is a zero-delay function of its drivers, not state.
#[test]
fn resolver_inside_kernel_recovers_after_all_x() {
    let mut r = Raw::new();
    let en = r.input("en");
    let x = r.input("x");
    let e = r.input("e");
    let t = r.gate("t", FlatOp::Tri, &[x, e]);
    let obs = r.gate("obs", FlatOp::Buf, &[t]);
    r.output("OBS", obs);
    let res = r.resolver("res", &[t], Termination::PullUp);
    let a = r.net("a");
    let b = r.gate("b", FlatOp::Not, &[a]);
    let c = r.gate("c", FlatOp::Not, &[b]);
    r.gate_to(FlatOp::Nand, &[en, c, res], a);
    r.output("A", a);
    let flat = r.done();
    let mut fr = FastRef::new(flat.clone(), FastOptions::default()).unwrap();
    let ks = fr.kernels();
    assert_eq!(ks.kernels.len(), 1);
    assert_eq!(
        ks.kernels[0].resolvers,
        vec![0],
        "the resolver is private fan-in"
    );
    // Oscillate.
    for (p, v) in [("en", true), ("x", true), ("e", true)] {
        fr.set_input(p, &Value::from_logic(Logic::from_bool(v)))
            .unwrap();
    }
    fr.step_event();
    assert!(
        fr.take_diagnostics()
            .iter()
            .any(|d| d.kind == DiagKind::Oscillating)
    );
    assert_eq!(fr.get(a), Logic::X);
    // Stop it: the loop settles at a = 1 and the resolver reads its driver again.
    fr.set_input("en", &Value::from_logic(Logic::Zero)).unwrap();
    fr.step_event();
    assert_eq!(fr.get(a), Logic::One);
    assert_eq!(fr.get(res), Logic::One, "resolver stuck at the forced X");
    assert!(fixed_point_violations(&flat, fr.values()).is_empty());
}

/// `layers` layers of `w` gates; every gate reads every gate of the previous layer (layer 0 reads
/// the last layer and input `i0`): `w^layers` simple cycles of length `layers`.
fn layered_ring(layers: usize, w: usize) -> Arc<FlatNetlist> {
    let mut r = Raw::new();
    let i0 = r.input("i0");
    let nets: Vec<Vec<NetIdx>> = (0..layers)
        .map(|l| (0..w).map(|j| r.net(&format!("l{l}_{j}"))).collect())
        .collect();
    for l in 0..layers {
        let prev = &nets[(l + layers - 1) % layers];
        for (j, &out) in nets[l].iter().enumerate() {
            let mut ins = prev.clone();
            if l == 0 {
                ins.push(i0);
            }
            let op = if (l + j) % 2 == 0 {
                FlatOp::Nand
            } else {
                FlatOp::Nor
            };
            r.gate_to(op, &ins, out);
        }
    }
    r.output("o", nets[layers - 1][0]);
    r.done()
}

// ---------------------------------------------------------------------------------------------
// Structured adversarial circuits vs Timing mode
// ---------------------------------------------------------------------------------------------

fn extract_variants() -> Vec<ExtractOptions> {
    let d = ExtractOptions::default();
    vec![
        d.clone(),
        ExtractOptions {
            max_loop: 2,
            ..d.clone()
        },
        ExtractOptions {
            max_loop: 12,
            max_bits: 14,
            ..d.clone()
        },
        ExtractOptions {
            max_bits: 1,
            ..d.clone()
        },
        ExtractOptions {
            max_loop: 0,
            ..d.clone()
        },
        ExtractOptions {
            max_cycles_per_region: 0,
            ..d.clone()
        },
    ]
}

/// Apply single-pin changes one at a time (each settles before the next) to Timing mode and to
/// FastRef under several extraction variants; every net must agree after every step. With one
/// input change per event and no reconvergent paths outside kernels there are no races, so the
/// equivalence of SPEC §6.3 holds whatever the kernel boundaries are.
fn async_equivalence(flat: &Arc<FlatNetlist>, steps: &[(&str, Logic)]) {
    for (vi, ext) in extract_variants().into_iter().enumerate() {
        let opts = FastOptions {
            extract: ext,
            ..FastOptions::default()
        };
        let mut fr = FastRef::new(flat.clone(), opts).unwrap();
        fr.kernels().verify(flat).unwrap();
        let mut ts = TimingSim::new(flat.clone(), TimingOptions::default());
        assert!(ts.settle(10_000));
        let compare = |fr: &FastRef, ts: &TimingSim, what: &str| {
            let diffs: Vec<String> = (0..flat.num_nets as usize)
                .filter(|&n| ts.values()[n] != fr.values()[n])
                .map(|n| {
                    format!(
                        "{}: timing {} fast {}",
                        flat.net_names[n],
                        ts.values()[n],
                        fr.values()[n]
                    )
                })
                .collect();
            assert!(diffs.is_empty(), "variant {vi} {what}: {diffs:?}");
        };
        compare(&fr, &ts, "power-on");
        for (si, &(pin, v)) in steps.iter().enumerate() {
            ts.set_input(pin, &Value::from_logic(v)).unwrap();
            fr.set_input(pin, &Value::from_logic(v)).unwrap();
            assert!(ts.settle(10_000), "timing does not settle at step {si}");
            fr.step_event();
            compare(&fr, &ts, &format!("step {si} ({pin}={v})"));
            assert!(fixed_point_violations(flat, fr.values()).is_empty());
        }
        let td = ts.take_diagnostics();
        let fd = fr.take_diagnostics();
        for kind in [DiagKind::Metastable, DiagKind::Contention] {
            assert_eq!(
                td.iter().any(|d| d.kind == kind),
                fd.iter().any(|d| d.kind == kind),
                "variant {vi}: {kind:?} reported by one engine only: {td:?} / {fd:?}"
            );
        }
    }
}

fn bufs(r: &mut Raw, name: &str, from: NetIdx, n: usize) -> NetIdx {
    let mut cur = from;
    for i in 0..n {
        cur = r.gate(&format!("{name}{i}"), FlatOp::Buf, &[cur]);
    }
    cur
}

use Logic::{One as H, Zero as L};

#[test]
fn latch_with_5_and_7_gate_loops() {
    let mut r = Raw::new();
    let s_n = r.input("S_n");
    let r_n = r.input("R_n");
    let a = r.net("a");
    let y = bufs(&mut r, "y", a, 1);
    let b = r.gate("b", FlatOp::Nand, &[r_n, y]);
    let x = bufs(&mut r, "x", b, 2);
    let z = bufs(&mut r, "z", b, 4);
    r.gate_to(FlatOp::Nand, &[s_n, x, z], a);
    r.output("Q", a);
    r.output("QN", b);
    let flat = r.done();
    let ks = extract(&flat, &ExtractOptions::default());
    assert_eq!(ks.kernels.len(), 1, "one kernel covers both loops");
    assert!(ks.comb_order.is_empty());
    async_equivalence(
        &flat,
        &[
            ("S_n", H),
            ("R_n", H),
            ("S_n", L),
            ("S_n", H),
            ("R_n", L),
            ("R_n", H),
            ("S_n", L),
            ("R_n", L),
            ("S_n", H),
            ("R_n", H),
        ],
    );
}

#[test]
fn kernels_sharing_input_nets_and_top_pins() {
    // Four SR latches on the same S_n / R_n nets; the private inverters of two of them also drive
    // top-level outputs (absorbed nets that are pins).
    let mut r = Raw::new();
    let s = r.input("S");
    let rr = r.input("R");
    for i in 0..4 {
        let sn = r.gate(&format!("sn{i}"), FlatOp::Not, &[s]);
        let rn = r.gate(&format!("rn{i}"), FlatOp::Not, &[rr]);
        if i % 2 == 0 {
            r.output(&format!("SN{i}"), sn);
            r.output(&format!("RN{i}"), rn);
        }
        let q = r.net(&format!("q{i}"));
        let qn = r.gate(&format!("qn{i}"), FlatOp::Nand, &[rn, q]);
        r.gate_to(FlatOp::Nand, &[sn, qn], q);
        r.output(&format!("Q{i}"), q);
    }
    let flat = r.done();
    let ks = extract(&flat, &ExtractOptions::default());
    assert_eq!(ks.kernels.len(), 4);
    for k in &ks.kernels {
        assert_eq!(k.gates.len(), 4, "inverters absorbed: {k:?}");
        assert_eq!(k.inputs.len(), 2);
    }
    async_equivalence(
        &flat,
        &[("S", H), ("S", L), ("R", H), ("R", L), ("S", H), ("S", L)],
    );
}

#[test]
fn keeper_bus_in_a_loop() {
    // bus (keeper) = tri(d, e) | tri(q, h);  q = buf(bus): a bus that holds itself through a gate.
    let mut r = Raw::new();
    let d = r.input("d");
    let e = r.input("e");
    let h = r.input("h");
    let t0 = r.gate("t0", FlatOp::Tri, &[d, e]);
    let q = r.net("q");
    let t1 = r.gate("t1", FlatOp::Tri, &[q, h]);
    let bus = r.resolver("bus", &[t0, t1], Termination::Keeper);
    r.gate_to(FlatOp::Buf, &[bus], q);
    r.output("BUS", bus);
    r.output("Q", q);
    let flat = r.done();
    let ks = extract(&flat, &ExtractOptions::default());
    ks.verify(&flat).unwrap();
    assert_eq!(ks.kernels.len(), 1);
    assert_eq!(ks.kernels[0].resolvers, vec![0]);
    async_equivalence(
        &flat,
        &[
            ("e", H),
            ("d", H),
            ("e", L),
            ("h", H),
            ("d", L),
            ("e", H),
            ("e", L),
            ("h", L),
            ("e", H),
            ("e", L),
            ("d", H),
        ],
    );
}

#[test]
fn malik_cyclic_combinational() {
    // Two muxes in a cycle (a = s ? x : b, b = s ? a : y), plus a second cyclic pair in series:
    // combinational for every binary input (Malik 1994), extracted as kernels.
    let mut r = Raw::new();
    let x = r.input("x");
    let y = r.input("y");
    let s = r.input("s");
    let ns = r.gate("ns", FlatOp::Not, &[s]);
    let mut prev = (x, y);
    for stage in 0..2 {
        let a = r.net(&format!("a{stage}"));
        let b = r.net(&format!("b{stage}"));
        let a1 = r.gate(&format!("a1_{stage}"), FlatOp::And, &[prev.0, s]);
        let a2 = r.gate(&format!("a2_{stage}"), FlatOp::And, &[b, ns]);
        r.gate_to(FlatOp::Or, &[a1, a2], a);
        let b1 = r.gate(&format!("b1_{stage}"), FlatOp::And, &[a, s]);
        let b2 = r.gate(&format!("b2_{stage}"), FlatOp::And, &[prev.1, ns]);
        r.gate_to(FlatOp::Or, &[b1, b2], b);
        r.output(&format!("A{stage}"), a);
        r.output(&format!("B{stage}"), b);
        prev = (a, b);
    }
    let flat = r.done();
    async_equivalence(
        &flat,
        &[
            ("s", H),
            ("x", H),
            ("s", L),
            ("y", H),
            ("x", L),
            ("s", H),
            ("y", L),
            ("s", L),
        ],
    );
}

// ---------------------------------------------------------------------------------------------
// Event schedule
// ---------------------------------------------------------------------------------------------

/// FastRef's event ticks are exactly the ticks at which some time-varying source differs from
/// the previous tick (brute force over Timing mode's source values), for awkward clock and POR
/// parameters; values at each event match Timing mode's sources.
#[test]
fn event_schedule_matches_source_changes() {
    let cases: &[&[SourceKind]] = &[
        &[
            SourceKind::Clock {
                period: 6,
                high: 3,
                phase: 0,
            },
            SourceKind::Clock {
                period: 4,
                high: 1,
                phase: 3,
            },
        ],
        &[
            SourceKind::Clock {
                period: 4,
                high: 2,
                phase: 0,
            },
            SourceKind::Clock {
                period: 4,
                high: 2,
                phase: 2,
            }, // edges coincide
            SourceKind::PowerOnReset { ticks: 4 }, // release on an edge
        ],
        &[
            SourceKind::Clock {
                period: 5,
                high: 5,
                phase: 1,
            }, // high == period: constant 1
            SourceKind::Clock {
                period: 5,
                high: 9,
                phase: 0,
            }, // high > period
            SourceKind::Clock {
                period: 3,
                high: 0,
                phase: 2,
            }, // constant 0
            SourceKind::PowerOnReset { ticks: 0 }, // never high
            SourceKind::PowerOnReset { ticks: 7 },
        ],
        &[
            SourceKind::Clock {
                period: 0,
                high: 1,
                phase: 0,
            },
            SourceKind::Clock {
                period: 1,
                high: 1,
                phase: 5,
            },
            SourceKind::Clock {
                period: 7,
                high: 6,
                phase: 40,
            }, // phase > period
            SourceKind::PowerOnReset { ticks: 1 },
        ],
    ];
    for (ci, kinds) in cases.iter().enumerate() {
        let mut r = Raw::new();
        let mut nets = Vec::new();
        for (i, k) in kinds.iter().enumerate() {
            let n = r.source(&format!("s{i}"), k.clone());
            nets.push(n);
            let o = r.gate(&format!("o{i}"), FlatOp::Buf, &[n]);
            r.output(&format!("O{i}"), o);
        }
        let flat = r.done();
        let mut ts = TimingSim::new(flat.clone(), TimingOptions::default());
        let mut want = Vec::new();
        let mut prev: Vec<Logic> = nets.iter().map(|&n| ts.get(n)).collect();
        for t in 1..=60u64 {
            ts.step();
            let cur: Vec<Logic> = nets.iter().map(|&n| ts.get(n)).collect();
            if cur != prev {
                want.push((t, cur.clone()));
            }
            prev = cur;
        }
        let mut fr = FastRef::new(flat.clone(), FastOptions::default()).unwrap();
        let mut got = Vec::new();
        while let Some(t) = fr.step_event() {
            if t > 60 {
                break;
            }
            got.push((t, nets.iter().map(|&n| fr.get(n)).collect::<Vec<_>>()));
        }
        assert_eq!(got, want, "case {ci}");
    }
}

/// An SR latch whose set input is the NAND/AND tree of `w` inputs, all private fan-in.
fn wide_fanin_latch(w: usize) -> Arc<FlatNetlist> {
    let mut r = Raw::new();
    let mut layer: Vec<NetIdx> = (0..w).map(|i| r.input(&format!("i{i}"))).collect();
    let rn = r.input("R_n");
    let mut lvl = 0;
    while layer.len() > 1 {
        let mut next = Vec::new();
        for (j, pair) in layer.chunks(2).enumerate() {
            next.push(if pair.len() == 2 {
                r.gate(&format!("t{lvl}_{j}"), FlatOp::And, pair)
            } else {
                pair[0]
            });
        }
        layer = next;
        lvl += 1;
    }
    let sn = r.gate("sn", FlatOp::Not, &[layer[0]]);
    let q = r.net("q");
    let qn = r.gate("qn", FlatOp::Nand, &[rn, q]);
    r.gate_to(FlatOp::Nand, &[sn, qn], q);
    r.output("Q", q);
    r.done()
}

/// Brute-force check of [`KernelTable::is_order_sensitive`]: for every stable state `s`, every
/// input vector `u` it is stable under and every `v` with ≥ 2 changed bits, apply the changes one
/// at a time in every order with a fresh kernel simulator and compare with the simultaneous
/// result. The answer must depend on `u` exactly as the table says (the per-entry flag alone is
/// the union over all `u`, which made FastRef report `DelaySensitive` for safe transitions).
#[test]
fn order_sensitivity_is_exact_per_previous_input() {
    let mut distinguished = 0;
    let mut unknown = 0;
    for budget in [CharzOptions::default().max_order_work, 40] {
        let charz = CharzOptions {
            max_order_work: budget,
            ..CharzOptions::default()
        };
        let kopts = KernelOptions {
            charz: charz.clone(),
            ..KernelOptions::default()
        };
        for seed in 0..300u64 {
            let flat = random_netlist(seed, 3, 4 + (seed % 12) as usize, false, false);
            let (ks, tables) = compile_kernels(&flat, &kopts);
            for (k, t) in ks.kernels.iter().zip(&tables) {
                let Some(t) = t else { continue };
                if k.inputs.len() > 4 {
                    continue;
                }
                let kk = k.inputs.len();
                let mut sim = KernelSim::new(&flat, k, &charz);
                let outcome = |sim: &mut KernelSim, st: &[Logic], from: u32, to: u32| {
                    let (n, settled) =
                        sim.transition(st, &input_vector(from, kk), &input_vector(to, kk));
                    // A settled state with X is kept (the next change may still lead back to a
                    // stable state); a transition that does not settle is lost.
                    settled.quiescent.then_some(n)
                };
                for (u, list) in t.states_by_input.iter().enumerate() {
                    let u = u as u32;
                    for &s in list {
                        let st = t.states[s as usize].clone();
                        for v in 0..1u32 << kk {
                            let diff = u ^ v;
                            if diff.count_ones() < 2 {
                                continue;
                            }
                            let simul = outcome(&mut sim, &st, u, v);
                            let bits: Vec<u32> =
                                (0..kk as u32).filter(|b| diff >> b & 1 == 1).collect();
                            let mut perm: Vec<usize> = (0..bits.len()).collect();
                            let mut sensitive = false;
                            loop {
                                let (mut cur, mut w) = (Some(st.clone()), u);
                                for &p in &perm {
                                    let nw = w ^ (1 << bits[p]);
                                    cur = cur.and_then(|c| outcome(&mut sim, &c, w, nw));
                                    w = nw;
                                }
                                // Non-binary results compare equal (as in characterization).
                                let id =
                                    |x: &Option<Vec<Logic>>| x.as_ref().and_then(|x| t.lookup(x));
                                if id(&cur) != id(&simul) {
                                    sensitive = true;
                                    break;
                                }
                                if !next_perm(&mut perm) {
                                    break;
                                }
                            }
                            let known = t.order_sensitivity_known(s, u, v);
                            if known.is_none() {
                                unknown += 1;
                                // Unchecked combinations are flagged conservatively, except
                                // the non-canonical input vectors of a partial check.
                                assert!(
                                    t.transition(s, v).order_sensitive
                                        || (t.order_check_partial && t.home_input[s as usize] != u)
                                );
                            }
                            let exact = known.unwrap_or_else(|| {
                                t.check_order_sensitive(&mut sim, &charz, s, u, v)
                            });
                            assert_eq!(
                                exact, sensitive,
                                "budget {budget} seed {seed} kernel {} s {s} u {u:b} v {v:b}",
                                k.id
                            );
                            if t.transition(s, v).order_sensitive && !sensitive {
                                distinguished += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        distinguished > 0,
        "no entry is sensitive from one input vector only"
    );
    assert!(unknown > 0, "the tight budget left nothing unchecked");
}

fn next_perm(p: &mut [usize]) -> bool {
    let n = p.len();
    if n < 2 {
        return false;
    }
    let mut i = n - 1;
    while i > 0 && p[i - 1] >= p[i] {
        i -= 1;
    }
    if i == 0 {
        return false;
    }
    let mut j = n - 1;
    while p[j] <= p[i - 1] {
        j -= 1;
    }
    p.swap(i - 1, j);
    p[i..].reverse();
    true
}

/// Contention inside a kernel is reported at the right net even when the kernel simulator is
/// shared with a structurally identical kernel (the second copy of the circuit).
#[test]
fn kernel_contention_is_located_in_the_right_copy() {
    let mut r = Raw::new();
    let mut buses = Vec::new();
    for c in 0..2 {
        let d = r.input(&format!("d{c}"));
        let e = r.input(&format!("e{c}"));
        let h = r.input(&format!("h{c}"));
        let t0 = r.gate(&format!("t0_{c}"), FlatOp::Tri, &[d, e]);
        let q = r.net(&format!("q{c}"));
        let t1 = r.gate(&format!("t1_{c}"), FlatOp::Tri, &[q, h]);
        let bus = r.resolver(&format!("bus{c}"), &[t0, t1], Termination::Keeper);
        r.gate_to(FlatOp::Buf, &[bus], q);
        r.output(&format!("Q{c}"), q);
        buses.push(bus);
    }
    let flat = r.done();
    let mut fr = FastRef::new(flat.clone(), FastOptions::default()).unwrap();
    assert_eq!(fr.kernels().kernels.len(), 2);
    // Copy 1: hold 1 through the loop, then drive 0 against it.
    for (p, v) in [("d1", H), ("e1", H)] {
        fr.set_input(p, &Value::from_logic(v)).unwrap();
        fr.step_event();
    }
    for (p, v) in [("h1", H), ("e1", L)] {
        fr.set_input(p, &Value::from_logic(v)).unwrap();
        fr.step_event();
    }
    assert_eq!(fr.get(buses[1]), H);
    fr.take_diagnostics();
    for (p, v) in [("d1", L), ("e1", H)] {
        fr.set_input(p, &Value::from_logic(v)).unwrap();
    }
    fr.step_event();
    let diags = fr.take_diagnostics();
    let cont: Vec<_> = diags
        .iter()
        .filter(|d| d.kind == DiagKind::Contention)
        .collect();
    assert_eq!(cont.len(), 1, "{diags:?}");
    assert_eq!(cont[0].location.as_deref(), Some("bus1"));
    assert!(
        diags.iter().all(|d| d.kind != DiagKind::Metastable),
        "{diags:?}"
    );
}

/// Performance pathologies: dense strongly connected regions with astronomically many short
/// cycles. Seeding is bounded by the per-root / per-region work budgets, and fan-in absorption
/// into a kernel that is already over the table budget no longer recomputes a feedback vertex
/// set per candidate (that made `layered_ring(2, 300)` take 2.6 s in release, now ~0.15 s).
#[test]
fn dense_regions_extract_quickly_and_correctly() {
    for &(layers, w) in &[(6usize, 10usize), (3, 60), (2, 150)] {
        let flat = layered_ring(layers, w);
        let t0 = Instant::now();
        let (ks, st) = extract_with_stats(&flat, &ExtractOptions::default());
        let dt = t0.elapsed();
        ks.verify(&flat).unwrap();
        assert!(st.cycles > 0);
        assert!(dt.as_secs() < 30, "{layers}x{w}: extraction took {dt:?}");
        let mut fr = FastRef::new(flat.clone(), FastOptions::default()).unwrap();
        for i in 0..4 {
            fr.set_input("i0", &Value::from_logic(Logic::from_bool(i % 2 == 0)))
                .unwrap();
            fr.step_event();
        }
    }
    // A tight per-region budget truncates seeding; the loops are still covered (Sim kernels).
    let flat = layered_ring(6, 10);
    let opts = ExtractOptions {
        max_seed_work_per_region: 1000,
        ..ExtractOptions::default()
    };
    let (ks, st) = extract_with_stats(&flat, &opts);
    ks.verify(&flat).unwrap();
    assert_eq!(st.regions_truncated, 1);
    assert!(!ks.kernels.is_empty());
}

/// A huge `max_bits` used to make characterization assert (`> 24 inputs`) or try to build a
/// `2^30`-entry table. Oversized kernels now stay untabulated and run on the fly.
#[test]
fn oversized_table_kernels_fall_back_to_simulation() {
    let flat = wide_fanin_latch(30);
    let opts = FastOptions {
        extract: ExtractOptions {
            max_bits: 40,
            ..ExtractOptions::default()
        },
        ..FastOptions::default()
    };
    let mut fr = FastRef::new(flat.clone(), opts).unwrap();
    let ks = fr.kernels();
    assert_eq!(ks.kernels.len(), 1);
    assert_eq!(ks.kernels[0].kind, KernelKind::Table);
    assert!(ks.kernels[0].inputs.len() > 24);
    assert!(fr.tables()[0].is_none());
    // Set (all inputs 1, R_n = 1) then reset (R_n = 0).
    for i in 0..30 {
        fr.set_input(&format!("i{i}"), &Value::from_logic(H))
            .unwrap();
    }
    fr.set_input("R_n", &Value::from_logic(H)).unwrap();
    fr.step_event();
    assert_eq!(fr.read_pin("Q"), Some(Value::from_logic(H)));
    fr.set_input("i3", &Value::from_logic(L)).unwrap();
    fr.step_event();
    assert_eq!(fr.read_pin("Q"), Some(Value::from_logic(H)), "holds");
    fr.set_input("R_n", &Value::from_logic(L)).unwrap();
    fr.step_event();
    assert_eq!(fr.read_pin("Q"), Some(Value::from_logic(L)));
    // A medium kernel (12 inputs + cut) is tabulated unless the entry guard forbids it.
    let flat = wide_fanin_latch(11);
    let ks = extract(&flat, &ExtractOptions::default());
    let k = &ks.kernels[0];
    assert_eq!(k.kind, KernelKind::Table);
    assert!(characterize(&flat, k, &CharzOptions::default()).is_some());
    let tiny = CharzOptions {
        max_table_entries: 1000,
        ..CharzOptions::default()
    };
    assert!(characterize(&flat, k, &tiny).is_none());
}
