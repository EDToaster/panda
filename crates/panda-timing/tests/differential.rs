//! Differential test: the event-driven engine against a naive reference model.
//!
//! The reference evaluates *every* gate on *every* tick straight from `panda_netlist::eval`,
//! re-applies every source on every tick, recomputes every resolver to a fixed point, keeps the
//! per-net alternation history with the textbook recurrence and finds the collapsing set with a
//! plain fixed-point iteration. It shares no code with the engine.
//!
//! The only event-driven aspects it models are the documented ones: a forced gate output persists
//! until one of the gate's inputs is touched, and a forced / collapsed resolver output persists
//! until one of its drivers is touched.
//!
//! Netlists come from two generators: a raw flat-netlist generator (multi-driver nets with all
//! terminations, Z/X constants, odd and degenerate clocks, power-on resets, multi-bit input pins,
//! undriven nets, wide gates, heavy feedback) and the design-level generator in `common` (run
//! through `flatten`). Random actions: steps, input changes (incl. X/Z), `force_net` on any net,
//! `reset`, and `run(k)` (which may fast-forward).

mod common;

use std::sync::Arc;

use common::*;
use panda_netlist::eval;
use panda_netlist::flat::{Driver, FlatGate, FlatNetlist, FlatOp, FlatPin, Resolver, Source, SourceKind};
use panda_netlist::{DesignBuilder, Dir, Logic, Termination, Value, flatten};
use panda_timing::{TimingOptions, TimingSim};

// ---------------------------------------------------------------------------------------------
// Reference model
// ---------------------------------------------------------------------------------------------

fn next_run(last: u64, run: u32, prev: Logic, new: Logic, t: u64) -> u32 {
    if last + 1 == t && run >= 1 && new == prev { run + 1 } else { 1 }
}

#[derive(Clone)]
struct Hist {
    last: Vec<u64>,
    run: Vec<u32>,
    prev: Vec<Logic>,
}

struct Naive {
    nl: Arc<FlatNetlist>,
    drv: Vec<Driver>,
    /// Per net: gates reading it.
    gate_readers: Vec<Vec<usize>>,
    v: Vec<Logic>,
    /// Final values of the previous tick.
    vm1: Vec<Logic>,
    inputs: Vec<Value>,
    tick: u64,
    hist: Hist,
    /// History before the current tick (to rewrite the tick on `force`).
    hist0: Hist,
    last_src: u64,
    limit: u32,
    /// Gate output overridden by `force`: not re-evaluated until an input is touched.
    gate_pinned: Vec<bool>,
    /// Gates that must be evaluated at the next step even if pinned.
    gate_touch: Vec<bool>,
    /// Resolver output overridden (force / collapse): not recomputed until a driver is touched.
    res_pinned: Vec<bool>,
    /// Nets that changed in the last tick (expected `last_changed`).
    changed: Vec<u32>,
    /// Number of nets forced to X by collapses so far.
    collapsed: usize,
}

impl Naive {
    fn new(nl: Arc<FlatNetlist>, p2_limit: u32) -> Self {
        let n = nl.num_nets as usize;
        let mut gate_readers = vec![Vec::new(); n];
        for (i, g) in nl.gates.iter().enumerate() {
            for &inp in &g.inputs {
                gate_readers[inp as usize].push(i);
            }
        }
        let hist = Hist { last: vec![0; n], run: vec![0; n], prev: vec![Logic::X; n] };
        let mut s = Naive {
            drv: nl.drivers(),
            gate_readers,
            v: vec![Logic::Z; n],
            vm1: vec![Logic::Z; n],
            inputs: nl.top_inputs.iter().map(|p| Value::zeros(p.nets.len() as u32)).collect(),
            tick: 0,
            hist0: hist.clone(),
            hist,
            last_src: 0,
            limit: 2 * p2_limit,
            gate_pinned: vec![false; nl.gates.len()],
            gate_touch: vec![false; nl.gates.len()],
            res_pinned: vec![false; nl.resolvers.len()],
            changed: Vec::new(),
            collapsed: 0,
            nl,
        };
        s.reset();
        s
    }

    fn reset(&mut self) {
        let nl = self.nl.clone();
        let n = nl.num_nets as usize;
        for i in 0..n {
            self.v[i] = match self.drv[i] {
                Driver::None => {
                    eval::terminate(nl.undriven_termination.get(&(i as u32)).copied().unwrap_or_default(), Logic::X)
                }
                _ => Logic::X,
            };
        }
        self.apply_sources(0);
        self.gate_pinned.fill(false);
        // Power-on: every gate is evaluated at tick 1 (even if its output is forced at tick 0).
        self.gate_touch.fill(true);
        self.res_pinned.fill(false);
        self.resolve_all();
        self.vm1 = self.v.clone();
        self.hist = Hist { last: vec![0; n], run: vec![0; n], prev: vec![Logic::X; n] };
        self.hist0 = self.hist.clone();
        self.tick = 0;
        self.last_src = 0;
        self.changed.clear();
    }

    fn source_value(&self, k: &SourceKind, t: u64) -> Logic {
        match *k {
            SourceKind::Constant(v) => v,
            SourceKind::Clock { period, high, phase } => {
                let p = u64::from(period.max(1));
                Logic::from_bool((t + p - u64::from(phase) % p) % p < u64::from(high))
            }
            SourceKind::PowerOnReset { ticks } => Logic::from_bool(t < u64::from(ticks)),
            SourceKind::TopInput { pin, bit } => self.inputs[pin as usize].get(bit),
        }
    }

    /// Every source takes its tick-`t` value. Returns true if any source net changed.
    fn apply_sources(&mut self, t: u64) -> bool {
        let nl = self.nl.clone();
        let mut changed = false;
        for s in &nl.sources {
            let v = self.source_value(&s.kind, t);
            changed |= self.v[s.output as usize] != v;
            self.v[s.output as usize] = v;
        }
        changed
    }

    /// Recompute every unpinned resolver until nothing changes.
    fn resolve_all(&mut self) {
        let nl = self.nl.clone();
        loop {
            let mut any = false;
            for (r, res) in nl.resolvers.iter().enumerate() {
                if self.res_pinned[r] {
                    continue;
                }
                let drv: Vec<Logic> = res.drivers.iter().map(|&d| self.v[d as usize]).collect();
                let out = eval::resolve(&drv, res.termination, self.v[res.output as usize]).value;
                if out != self.v[res.output as usize] {
                    self.v[res.output as usize] = out;
                    any = true;
                }
            }
            if !any {
                break;
            }
        }
    }

    /// Unpin the resolvers with a driver in `touched`.
    fn unpin_resolvers(&mut self, touched: &[bool]) {
        for (r, res) in self.nl.resolvers.iter().enumerate() {
            if res.drivers.iter().any(|&d| touched[d as usize]) {
                self.res_pinned[r] = false;
            }
        }
    }

    /// Mark the readers of every touched net for evaluation at the next step.
    fn touch_readers(&mut self, touched: &[bool]) {
        for (n, &t) in touched.iter().enumerate() {
            if t {
                for &g in &self.gate_readers[n] {
                    self.gate_touch[g] = true;
                }
            }
        }
    }

    /// History of tick `t` from `hist0`, `vm1` and the current values.
    fn history(&self, t: u64) -> Hist {
        let mut h = self.hist0.clone();
        if t == 0 {
            return h;
        }
        for n in 0..self.v.len() {
            if self.v[n] != self.vm1[n] {
                h.run[n] = next_run(self.hist0.last[n], self.hist0.run[n], self.hist0.prev[n], self.v[n], t);
                h.prev[n] = self.vm1[n];
                h.last[n] = t;
            }
        }
        h
    }

    fn finish_tick(&mut self, t: u64) {
        self.hist = self.history(t);
        self.changed = if t == 0 {
            Vec::new()
        } else {
            (0..self.v.len() as u32).filter(|&n| self.v[n as usize] != self.vm1[n as usize]).collect()
        };
    }

    /// Driver inputs of a net (`None` if it has no gate/resolver driver).
    fn driver_inputs(&self, n: usize) -> Option<Vec<u32>> {
        match self.drv[n] {
            Driver::Gate(g) => Some(self.nl.gates[g as usize].inputs.clone()),
            Driver::Resolver(r) => Some(self.nl.resolvers[r as usize].drivers.clone()),
            _ => None,
        }
    }

    fn step(&mut self) {
        let nl = self.nl.clone();
        let t = self.tick + 1;
        let n = self.v.len();
        let old = self.v.clone(); // V(t-1)
        // Inputs touched at t-1 (relative to t-2), plus explicit touches.
        let touched_prev: Vec<bool> = (0..n).map(|i| old[i] != self.vm1[i]).collect();
        self.touch_readers(&touched_prev);
        // Gates.
        for (gi, g) in nl.gates.iter().enumerate() {
            if self.gate_pinned[gi] && !self.gate_touch[gi] {
                continue;
            }
            self.gate_pinned[gi] = false;
            let ins: Vec<Logic> = g.inputs.iter().map(|&i| old[i as usize]).collect();
            self.v[g.output as usize] = eval::eval_gate(g.op, &ins);
        }
        self.gate_touch.fill(false);
        // Sources.
        if self.apply_sources(t) {
            self.last_src = t;
        }
        // Resolvers.
        let touched: Vec<bool> = (0..n).map(|i| self.v[i] != old[i]).collect();
        self.unpin_resolvers(&touched);
        self.resolve_all();

        // Period-2 collapse.
        self.vm1 = old;
        self.hist0 = self.hist.clone();
        let pre = self.v.clone();
        let tentative = self.history(t);
        let l = self.limit;
        let mut collapsed = vec![false; n];
        if l > 0 && t - self.last_src >= u64::from(l) {
            let mut set: Vec<bool> = (0..n).map(|i| self.v[i] != self.vm1[i] && tentative.run[i] >= l).collect();
            loop {
                let mut removed = false;
                for i in 0..n {
                    if !set[i] {
                        continue;
                    }
                    let ok = match self.driver_inputs(i) {
                        None => false,
                        Some(ins) => {
                            ins.iter().all(|&j| set[j as usize] || tentative.last[j as usize] + u64::from(l) <= t)
                        }
                    };
                    if !ok {
                        set[i] = false;
                        removed = true;
                    }
                }
                if !removed {
                    break;
                }
            }
            for i in 0..n {
                if set[i] && self.v[i] != Logic::X {
                    self.v[i] = Logic::X;
                    collapsed[i] = true;
                    self.collapsed += 1;
                    match self.drv[i] {
                        Driver::Gate(g) => self.gate_touch[g as usize] = true,
                        Driver::Resolver(r) => self.res_pinned[r as usize] = true,
                        _ => unreachable!(),
                    }
                }
            }
            self.unpin_resolvers_except_held(&collapsed);
            self.resolve_all();
        }
        // Everything set during this tick is "touched" for the next step: changes before the
        // collapse, collapsed nets, and final changes.
        let touched: Vec<bool> =
            (0..n).map(|i| pre[i] != self.vm1[i] || collapsed[i] || self.v[i] != self.vm1[i]).collect();
        // (Readers of nets touched at t are evaluated at t+1 via `touched_prev` unless the net
        // ended where it started; cover that case explicitly.)
        self.touch_readers(&touched);
        self.tick = t;
        self.finish_tick(t);
    }

    /// After a collapse: resolvers reading collapsed nets recompute, except those whose own
    /// output collapsed in this pass (they stay pinned at X).
    fn unpin_resolvers_except_held(&mut self, collapsed: &[bool]) {
        for (r, res) in self.nl.resolvers.iter().enumerate() {
            if collapsed[res.output as usize] {
                continue;
            }
            if res.drivers.iter().any(|&d| collapsed[d as usize]) {
                self.res_pinned[r] = false;
            }
        }
    }

    fn force(&mut self, net: u32, value: Logic) {
        let n = net as usize;
        if self.v[n] == value {
            return;
        }
        self.v[n] = value;
        match self.drv[n] {
            Driver::Gate(g) => self.gate_pinned[g as usize] = true,
            Driver::Resolver(r) => self.res_pinned[r as usize] = true,
            _ => {}
        }
        let mut touched = vec![false; self.v.len()];
        touched[n] = true;
        self.touch_readers(&touched);
        self.unpin_resolvers(&touched);
        let before = self.v.clone();
        self.resolve_all();
        // Resolver outputs changed by the force are touched too.
        let touched: Vec<bool> = (0..self.v.len()).map(|i| self.v[i] != before[i]).collect();
        self.touch_readers(&touched);
        self.last_src = self.tick;
        self.finish_tick(self.tick);
    }
}

// ---------------------------------------------------------------------------------------------
// Raw flat-netlist generator
// ---------------------------------------------------------------------------------------------

struct FlatSpec {
    pins: usize,
    clocks: usize,
    pors: usize,
    consts: usize,
    undriven: usize,
    gates: usize,
    resolvers: usize,
    /// Probability that a gate input may come from anywhere (feedback) instead of earlier nets.
    feedback: f64,
}

fn rand_logic(rng: &mut Rng) -> Logic {
    Logic::from_code(rng.below(4) as u8)
}

fn rand_term(rng: &mut Rng) -> Termination {
    match rng.below(4) {
        0 => Termination::None,
        1 => Termination::PullUp,
        2 => Termination::PullDown,
        _ => Termination::Keeper,
    }
}

fn rand_op(rng: &mut Rng) -> (FlatOp, usize) {
    const OPS: [FlatOp; 9] = [
        FlatOp::And,
        FlatOp::Or,
        FlatOp::Nand,
        FlatOp::Nor,
        FlatOp::Xor,
        FlatOp::Xnor,
        FlatOp::Not,
        FlatOp::Buf,
        FlatOp::Tri,
    ];
    let op = OPS[rng.below(OPS.len())];
    let arity = match op {
        FlatOp::Not | FlatOp::Buf => 1,
        FlatOp::Tri => 2,
        _ => 1 + rng.below(5),
    };
    (op, arity)
}

fn random_flat(rng: &mut Rng, spec: &FlatSpec) -> FlatNetlist {
    let mut nl = FlatNetlist::default();
    let mut next = 0u32;
    let mut fresh = || {
        next += 1;
        next - 1
    };
    let mut private = Vec::new(); // resolver driver nets: read only by their resolver
    for p in 0..spec.pins {
        let w = 1 + rng.below(3);
        let nets: Vec<u32> = (0..w).map(|_| fresh()).collect();
        for (bit, &net) in nets.iter().enumerate() {
            nl.sources.push(Source { kind: SourceKind::TopInput { pin: p as u32, bit: bit as u32 }, output: net });
        }
        nl.top_inputs.push(FlatPin { name: format!("P{p}"), dir: Dir::In, nets });
    }
    for _ in 0..spec.clocks {
        let period = 1 + rng.below(12) as u32;
        let high = rng.below(period as usize + 2) as u32;
        let phase = rng.below(30) as u32;
        nl.sources.push(Source { kind: SourceKind::Clock { period, high, phase }, output: fresh() });
    }
    for _ in 0..spec.pors {
        let ticks = rng.below(30) as u32;
        nl.sources.push(Source { kind: SourceKind::PowerOnReset { ticks }, output: fresh() });
    }
    for _ in 0..spec.consts {
        let v = rand_logic(rng);
        nl.sources.push(Source { kind: SourceKind::Constant(v), output: fresh() });
    }
    for _ in 0..spec.undriven {
        let net = fresh();
        if rng.chance(0.7) {
            nl.undriven_termination.insert(net, rand_term(rng));
        }
    }
    // Structure first (inputs filled in once every net exists, so feedback can reach anywhere).
    let total = spec.gates + spec.resolvers;
    for _ in 0..total {
        if rng.below(total) < spec.resolvers {
            let k = 1 + rng.below(4);
            let mut drivers = Vec::new();
            for _ in 0..k {
                let d = fresh();
                private.push(d);
                if rng.chance(0.1) {
                    let v = [Logic::Z, Logic::Z, Logic::Zero, Logic::One][rng.below(4)];
                    nl.sources.push(Source { kind: SourceKind::Constant(v), output: d });
                } else {
                    let (op, arity) = if rng.chance(0.7) { (FlatOp::Tri, 2) } else { rand_op(rng) };
                    nl.gates.push(FlatGate { op, inputs: vec![0; arity], output: d, owner: 0, group: 0, bit: 0 });
                }
                drivers.push(d);
            }
            nl.resolvers.push(Resolver { drivers, termination: rand_term(rng), output: fresh() });
        } else {
            let (op, arity) = rand_op(rng);
            nl.gates.push(FlatGate { op, inputs: vec![0; arity], output: fresh(), owner: 0, group: 0, bit: 0 });
        }
    }
    let n = next;
    nl.num_nets = n;
    let mut readable_mask = vec![true; n as usize];
    for &p in &private {
        readable_mask[p as usize] = false;
    }
    let readable: Vec<u32> = (0..n).filter(|&i| readable_mask[i as usize]).collect();
    for g in &mut nl.gates {
        let below: Vec<u32> = readable.iter().copied().filter(|&i| i < g.output).collect();
        for inp in &mut g.inputs {
            *inp = if below.is_empty() || rng.chance(spec.feedback) {
                readable[rng.below(readable.len())]
            } else if rng.chance(0.7) && below.len() > 6 {
                // Locality: mostly recent nets, so signals propagate through chains.
                below[below.len() - 1 - rng.below(6)]
            } else {
                below[rng.below(below.len())]
            };
        }
    }
    nl.net_names = (0..n).map(|i| format!("n{i}")).collect();
    nl
}

// ---------------------------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------------------------

struct Stats {
    collapsed: usize,
}

fn check(sim: &TimingSim, naive: &Naive, ctx: &dyn Fn() -> String, what: &str) {
    if sim.values() != &naive.v[..] {
        let nl = sim.netlist();
        let diffs: Vec<String> = (0..naive.v.len())
            .filter(|&n| sim.values()[n] != naive.v[n])
            .take(10)
            .map(|n| format!("{}: sim {} naive {}", nl.net_names[n], sim.values()[n], naive.v[n]))
            .collect();
        panic!("{}: values differ after {what} at tick {}: {diffs:?}", ctx(), sim.tick());
    }
    let mut got = sim.last_changed().to_vec();
    got.sort_unstable();
    assert_eq!(got, naive.changed, "{}: last_changed after {what} at tick {}", ctx(), sim.tick());
    assert_eq!(sim.tick(), naive.tick, "{}: tick after {what}", ctx());
}

fn fuzz(nl: Arc<FlatNetlist>, rng: &mut Rng, p2_limit: u32, actions: usize, label: &str, stats: &mut Stats) {
    let opts = TimingOptions::default().with_p2_limit(p2_limit).with_osc_limit(30);
    let mut sim = TimingSim::new(nl.clone(), opts);
    let mut naive = Naive::new(nl.clone(), p2_limit);
    let mut log: Vec<String> = Vec::new();
    let ctx = |log: &Vec<String>| {
        let tail: Vec<&String> = log.iter().rev().take(6).collect();
        format!("{label} (p2_limit {p2_limit}), recent actions {tail:?}")
    };
    check(&sim, &naive, &|| ctx(&log), "power-on");
    for _ in 0..actions {
        let roll = rng.below(100);
        if roll < 12 && !nl.top_inputs.is_empty() {
            let pin = rng.below(nl.top_inputs.len());
            let w = nl.top_inputs[pin].nets.len();
            let bits: Vec<Logic> = (0..w)
                .map(|_| match rng.below(10) {
                    0 => Logic::X,
                    1 => Logic::Z,
                    k => Logic::from_bool(k % 2 == 0),
                })
                .collect();
            let val = Value::from_bits(&bits);
            let name = nl.top_inputs[pin].name.clone();
            sim.set_input(&name, &val).unwrap();
            assert!(sim.set_input(&name, &Value::zeros(w as u32 + 1)).is_err());
            naive.inputs[pin] = val.clone();
            log.push(format!("set {name}={val}"));
        } else if roll < 15 {
            let net = rng.below(nl.num_nets as usize) as u32;
            let v = rand_logic(rng);
            sim.force_net(net, v);
            naive.force(net, v);
            log.push(format!("force n{net}={v}"));
            check(&sim, &naive, &|| ctx(&log), "force_net");
        } else if roll < 17 {
            // Scramble: force every gate output to a random binary value (like power-on
            // randomization), so feedback loops start from binary states.
            for g in &nl.gates {
                let v = Logic::from_bool(rng.chance(0.5));
                sim.force_net(g.output, v);
                naive.force(g.output, v);
            }
            log.push("scramble".into());
            check(&sim, &naive, &|| ctx(&log), "scramble");
        } else if roll < 18 {
            sim.reset();
            naive.reset();
            log.push("reset".into());
            check(&sim, &naive, &|| ctx(&log), "reset");
        } else if roll < 22 {
            let k = 1 + rng.below(30) as u64;
            sim.run(k);
            for _ in 0..k {
                naive.step();
            }
            log.push(format!("run {k}"));
            check(&sim, &naive, &|| ctx(&log), "run");
        } else {
            sim.step();
            naive.step();
            log.push("step".into());
            check(&sim, &naive, &|| ctx(&log), "step");
        }
    }
    stats.collapsed += naive.collapsed;
}

fn flat_case(seed: u64, spec: &FlatSpec, p2_limit: u32, actions: usize, stats: &mut Stats) {
    let mut rng = Rng::new(seed);
    let nl = Arc::new(random_flat(&mut rng, spec));
    fuzz(nl, &mut rng, p2_limit, actions, &format!("flat seed {seed}"), stats);
}

fn design_case(seed: u64, spec: &RandomSpec, p2_limit: u32, actions: usize, stats: &mut Stats) {
    let mut rng = Rng::new(seed);
    let def = random_design(&mut rng, spec);
    let mut db = DesignBuilder::new();
    let top = db.add(def);
    let nl = Arc::new(flatten(&db.build(&top)).expect("flatten"));
    fuzz(nl, &mut rng, p2_limit, actions, &format!("design seed {seed}"), stats);
}

#[test]
fn flat_acyclic_netlists_match_reference() {
    let mut st = Stats { collapsed: 0 };
    for seed in 0..60 {
        let spec =
            FlatSpec { pins: 3, clocks: 1, pors: 1, consts: 3, undriven: 2, gates: 50, resolvers: 6, feedback: 0.0 };
        flat_case(seed, &spec, 4, 300, &mut st);
    }
}

#[test]
fn flat_cyclic_netlists_match_reference() {
    let mut st = Stats { collapsed: 0 };
    for seed in 1000..1200 {
        let spec = FlatSpec {
            pins: 3,
            clocks: (seed % 3) as usize,
            pors: (seed % 2) as usize,
            consts: 3,
            undriven: 2,
            gates: 40,
            resolvers: 5,
            feedback: 0.25,
        };
        flat_case(seed, &spec, [4, 1, 2, 0][(seed % 4) as usize], 400, &mut st);
    }
    assert!(st.collapsed > 0, "the fuzzer must exercise period-2 collapse");
}

#[test]
fn flat_tiny_dense_loops_match_reference() {
    // Very small, very cyclic netlists: lots of self-loops, latches and metastability.
    let mut st = Stats { collapsed: 0 };
    for seed in 5000..5600 {
        let spec = FlatSpec {
            pins: 2,
            clocks: usize::from(seed % 5 == 0),
            pors: 0,
            consts: 1,
            undriven: 1,
            gates: 8,
            resolvers: 2,
            feedback: 0.6,
        };
        flat_case(seed, &spec, 1 + (seed % 3) as u32, 300, &mut st);
    }
    assert!(st.collapsed > 0, "the fuzzer must exercise period-2 collapse");
}

#[test]
fn design_netlists_match_reference() {
    let mut st = Stats { collapsed: 0 };
    for seed in 0..40 {
        let spec =
            RandomSpec { inputs: 5, gates: 60, feedback: 0.15, buses: 2, clocks: (seed % 2) as usize, locality: 8 };
        design_case(seed, &spec, [4, 1, 0][(seed % 3) as usize], 300, &mut st);
    }
}

/// Long soak (ignored by default): `cargo test --release -p panda-timing --test differential -- --ignored`.
/// `PANDA_SOAK_CASES` sets the number of cases per generator (default 3000).
#[test]
#[ignore = "long soak; run in release with --ignored"]
fn soak() {
    let cases: u64 = std::env::var("PANDA_SOAK_CASES").ok().and_then(|v| v.parse().ok()).unwrap_or(3000);
    let mut st = Stats { collapsed: 0 };
    for seed in 0..cases {
        let mut rng = Rng::new(seed ^ 0xA5A5);
        let spec = FlatSpec {
            pins: 1 + rng.below(3),
            clocks: rng.below(3),
            pors: rng.below(2),
            consts: rng.below(4),
            undriven: rng.below(3),
            gates: 4 + rng.below(60),
            resolvers: rng.below(6),
            feedback: [0.0, 0.1, 0.3, 0.6][rng.below(4)],
        };
        flat_case(100_000 + seed, &spec, rng.below(5) as u32, 500, &mut st);
        if seed % 4 == 0 {
            let spec = RandomSpec {
                inputs: 2 + rng.below(5),
                gates: 10 + rng.below(80),
                feedback: [0.0, 0.1, 0.3][rng.below(3)],
                buses: rng.below(3),
                clocks: rng.below(2),
                locality: rng.below(10),
            };
            design_case(200_000 + seed, &spec, rng.below(5) as u32, 400, &mut st);
        }
    }
    println!("soak: {cases} cases, {} nets collapsed", st.collapsed);
    assert!(st.collapsed > 0);
}
