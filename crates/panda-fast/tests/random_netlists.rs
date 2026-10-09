//! FastSim = FastRef on raw random netlists with arbitrary feedback (loops of every length,
//! tri-state drivers merged by resolvers with every termination incl. keepers, optional clock):
//! the adversarial corner cases of kernel extraction — sim kernels, oscillators, delta-limit
//! freezing, contention inside and outside kernels, X/Z inputs.

mod common;

/// Native-code hook of `common::pair` (none here: the interpreter; the panda-jit crate runs
/// this suite with the JIT installed through it).
#[allow(dead_code)]
fn native_hook(_: &mut panda_fast::FastSim) {}

use std::sync::Arc;

use common::{Coverage, pair, random_compile, random_lockstep};
use panda_compile::FastOptions;
use panda_compile::kernels::ExtractOptions;
use panda_difftest::Rng;
use panda_netlist::flat::{
    FlatGate, FlatInstance, FlatNetlist, FlatOp, FlatPin, GateGroup, NetIdx, Resolver, Source,
    SourceKind,
};
use panda_netlist::{Dir, Termination};

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

struct Raw(FlatNetlist);

impl Raw {
    fn new() -> Raw {
        let mut f = FlatNetlist::default();
        f.instances.push(FlatInstance {
            path: String::new(),
            def: "top".into(),
            parent: None,
            children: Vec::new(),
            pins: Vec::new(),
        });
        Raw(f)
    }
    fn net(&mut self, name: &str) -> NetIdx {
        let n = self.0.num_nets;
        self.0.num_nets += 1;
        self.0.net_names.push(name.to_string());
        self.0.name_index.insert(name.to_string(), n);
        n
    }
    fn input(&mut self, name: &str) -> NetIdx {
        let n = self.net(name);
        let pin = self.0.top_inputs.len() as u32;
        self.0.sources.push(Source {
            kind: SourceKind::TopInput { pin, bit: 0 },
            output: n,
        });
        self.0.top_inputs.push(FlatPin {
            name: name.to_string(),
            dir: Dir::In,
            nets: vec![n],
        });
        n
    }
    fn gate_to(&mut self, op: FlatOp, inputs: &[NetIdx], out: NetIdx) {
        let gi = self.0.gates.len() as u32;
        let group = self.0.groups.len() as u32;
        self.0.gates.push(FlatGate {
            op,
            inputs: inputs.to_vec(),
            output: out,
            owner: 0,
            group,
            bit: 0,
        });
        self.0.groups.push(GateGroup {
            owner: 0,
            path: format!("g{gi}"),
            gates: vec![gi],
        });
    }
}

/// `ni` inputs, `ng` gates reading arbitrary nets (feedback of every length), optional
/// tri-state buses and clock.
fn random_netlist(seed: u64, ni: usize, ng: usize, tri: bool, clock: bool) -> Arc<FlatNetlist> {
    let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678);
    let mut r = Raw::new();
    let mut ins: Vec<NetIdx> = (0..ni).map(|i| r.input(&format!("i{i}"))).collect();
    if clock {
        let period = 2 * (2 + rng.below(6)) as u32;
        let high = 1 + rng.below(period as usize - 1) as u32;
        let phase = rng.below(period as usize) as u32;
        let n = r.net("clk");
        r.0.sources.push(Source {
            kind: SourceKind::Clock {
                period,
                high,
                phase,
            },
            output: n,
        });
        ins.push(n);
    }
    if rng.chance(30) {
        let n = r.net("rst");
        r.0.sources.push(Source {
            kind: SourceKind::PowerOnReset {
                ticks: 3 + rng.below(20) as u32,
            },
            output: n,
        });
        ins.push(n);
    }
    let outs: Vec<NetIdx> = (0..ng).map(|g| r.net(&format!("n{g}"))).collect();
    let mut pool = ins.clone();
    pool.extend(&outs);
    let mut res_nets = Vec::new();
    if tri {
        for b in 0..1 + rng.below(2) {
            let term = *rng.pick(&[
                Termination::None,
                Termination::PullUp,
                Termination::PullDown,
                Termination::Keeper,
            ]);
            let n = r.net(&format!("bus{b}"));
            res_nets.push((n, term));
            pool.push(n);
        }
    }
    for &out in &outs {
        let op = *rng.pick(&OPS);
        let arity = match op {
            FlatOp::Not | FlatOp::Buf => 1,
            _ => 2 + usize::from(rng.chance(25)) + usize::from(rng.chance(10)),
        };
        let inputs: Vec<NetIdx> = (0..arity)
            .map(|_| {
                if rng.chance(50) {
                    let lo = (out as usize).saturating_sub(4);
                    let hi = (out as usize + 4).min(r.0.num_nets as usize - 1);
                    let cand = rng.range(lo, hi) as NetIdx;
                    if pool.contains(&cand) {
                        cand
                    } else {
                        *rng.pick(&pool)
                    }
                } else {
                    *rng.pick(&pool)
                }
            })
            .collect();
        r.gate_to(op, &inputs, out);
    }
    for (bi, &(bus, term)) in res_nets.iter().enumerate() {
        let mut drv = Vec::new();
        for d in 0..1 + rng.below(3) {
            let dn = r.net(&format!("bus{bi}_d{d}"));
            let data = *rng.pick(&pool);
            let en = *rng.pick(&pool);
            r.gate_to(FlatOp::Tri, &[data, en], dn);
            drv.push(dn);
        }
        r.0.resolvers.push(Resolver {
            drivers: drv,
            termination: term,
            output: bus,
        });
    }
    for o in 0..3.min(ng) {
        let n = outs[rng.below(ng)];
        r.0.top_outputs.push(FlatPin {
            name: format!("o{o}"),
            dir: Dir::Out,
            nets: vec![n],
        });
    }
    Arc::new(r.0)
}

fn variants() -> Vec<ExtractOptions> {
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
    ]
}

fn run(seeds: std::ops::Range<u64>, actions: usize) -> Coverage {
    let mut cov = Coverage::default();
    let vars = variants();
    for seed in seeds {
        let mut rng = Rng::new(seed ^ 0x5EED);
        let ni = 1 + rng.below(4);
        let ng = 4 + rng.below(40);
        let flat = random_netlist(seed, ni, ng, rng.chance(40), rng.chance(50));
        let mut fast = FastOptions {
            extract: vars[seed as usize % vars.len()].clone(),
            delta_limit: *rng.pick(&[64, 64, 4, 1]),
            ..FastOptions::default()
        };
        // A separate stream, so the other choices of a seed do not depend on it.
        let mut crng = Rng::new(seed ^ 0xC4A2_0000);
        if crng.chance(40) {
            common::randomize_charz(&mut crng, &mut fast);
        }
        let mut compile = random_compile(&mut rng, &[1, 2, 5, 64]);
        if std::env::var("PANDA_NOLUT").is_ok() {
            compile.lut_max_bits = 0;
        }
        if std::env::var("PANDA_NOSPEC").is_ok() {
            compile.specialize = false;
        }
        let (mut fr, mut fs) = pair(&flat, &fast, &compile);
        if std::env::var("PANDA_DUMP").is_ok() {
            common::dump(&flat, &fs);
        }
        let ctx = format!("raw seed {seed} ({ni} in, {ng} gates, {fast:?}, {compile:?})");
        random_lockstep(
            &mut fr, &mut fs, &flat, &mut rng, actions, 20, &ctx, &mut cov,
        );
    }
    cov
}

#[test]
fn random_feedback_netlists() {
    let cov = run(0..600, 60);
    eprintln!("raw netlists: {cov:?}");
    for kind in ["Oscillating", "IterationLimit", "Contention"] {
        assert!(cov.diag_kinds.contains_key(kind), "no {kind}: {cov:?}");
    }
}

#[test]
#[ignore = "soak"]
fn random_feedback_netlists_soak() {
    let cov = run(600..6600, 120);
    eprintln!("raw netlists soak: {cov:?}");
}

/// Debug helper: `PANDA_SEED=<n> cargo test ... one_raw_seed -- --ignored --nocapture`
/// (`PANDA_NOLUT`, `PANDA_NOSPEC`, `PANDA_DUMP` override / print).
#[test]
#[ignore = "debug helper"]
fn one_raw_seed() {
    let seed: u64 = std::env::var("PANDA_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let cov = run(seed..seed + 1, 120);
    eprintln!("raw seed {seed}: {cov:?}");
}
