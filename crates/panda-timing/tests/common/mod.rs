//! Circuit builders shared by the integration tests.
#![allow(dead_code)]

use std::sync::Arc;

use panda_netlist::{ComponentDef, DefBuilder, DefRef, DesignBuilder, Logic, Sig, Value, flatten};
use panda_timing::{TimingOptions, TimingSim};

/// Flatten `top` (with sub-defs `subs`) and create a simulator.
pub fn sim_with(top: ComponentDef, subs: Vec<ComponentDef>, opts: TimingOptions) -> TimingSim {
    let mut db = DesignBuilder::new();
    for s in subs {
        db.add_once(s);
    }
    let t = db.add(top);
    let design = db.build(&t);
    let flat = flatten(&design).unwrap_or_else(|e| panic!("flatten failed: {e:#?}"));
    TimingSim::new(Arc::new(flat), opts)
}

/// [`sim_with`] using default options.
pub fn sim(top: ComponentDef, subs: Vec<ComponentDef>) -> TimingSim {
    sim_with(top, subs, TimingOptions::default())
}

/// Read a 1-bit signal by name.
pub fn bit(sim: &TimingSim, name: &str) -> Logic {
    let v = sim.read(name).unwrap_or_else(|| panic!("no signal named {name:?}"));
    assert_eq!(v.width(), 1, "{name} is not 1 bit wide");
    v.get(0)
}

/// Set a 1-bit input.
pub fn set(sim: &mut TimingSim, pin: &str, v: bool) {
    sim.set_input(pin, &Value::from_bool(v)).unwrap();
}

/// NAND SR latch def: inputs `S_n`, `R_n` (active low); outputs `Q`, `Q_n`.
pub fn nand_sr(id: &str) -> ComponentDef {
    let mut b = DefBuilder::new(id);
    let s_n = b.input("S_n", 1);
    let r_n = b.input("R_n", 1);
    let qn = b.wire(1);
    let q = b.nand(&[s_n, qn]);
    b.nand_to(&[r_n, q], qn);
    b.name(q, "q");
    b.name(qn, "qn");
    b.output("Q", q);
    b.output("Q_n", qn);
    b.finish()
}

/// NOR SR latch def: inputs `S`, `R` (active high); outputs `Q`, `Q_n`.
pub fn nor_sr(id: &str) -> ComponentDef {
    let mut b = DefBuilder::new(id);
    let s = b.input("S", 1);
    let r = b.input("R", 1);
    let qn = b.wire(1);
    let q = b.nor(&[r, qn]);
    b.nor_to(&[s, q], qn);
    b.name(q, "q");
    b.name(qn, "qn");
    b.output("Q", q);
    b.output("Q_n", qn);
    b.finish()
}

/// Gated D latch from 4 NANDs: inputs `D`, `E`; outputs `Q`, `Q_n`. Transparent while `E = 1`.
pub fn d_latch() -> ComponentDef {
    let mut b = DefBuilder::new("d_latch");
    let d = b.input("D", 1);
    let e = b.input("E", 1);
    let s = b.nand(&[d, e]);
    let r = b.nand(&[s, e]);
    let qn = b.wire(1);
    let q = b.nand(&[s, qn]);
    b.nand_to(&[r, q], qn);
    b.name(q, "q");
    b.name(qn, "qn");
    b.output("Q", q);
    b.output("Q_n", qn);
    b.finish()
}

/// Rising-edge master-slave D flip-flop from two NAND D latches: inputs `D`, `C`; outputs `Q`, `Q_n`.
pub fn ms_dff() -> ComponentDef {
    let latch = DefRef::of(&d_latch());
    let mut b = DefBuilder::new("ms_dff");
    let d = b.input("D", 1);
    let c = b.input("C", 1);
    let nc = b.not(c);
    let m = b.inst_named("master", &latch, &[d, nc]);
    let s = b.inst_named("slave", &latch, &[m[0], c]);
    b.output("Q", s[0]);
    b.output("Q_n", s[1]);
    b.finish()
}

/// Sub-defs needed by anything using [`ms_dff`].
pub fn dff_defs() -> Vec<ComponentDef> {
    vec![d_latch(), ms_dff()]
}

/// 4-bit synchronous up-counter with synchronous reset: inputs `C`, `RST`; output `Q` (4 bits).
/// The internal next-state bus is named `next`.
pub fn counter4() -> ComponentDef {
    let dff = DefRef::of(&ms_dff());
    let mut b = DefBuilder::new("counter4");
    let c = b.input("C", 1);
    let rst = b.input("RST", 1);
    let nrst = b.not(rst);
    let qs: Vec<Sig> = (0..4).map(|_| b.wire(1)).collect();
    let mut carry = b.lit(1, 1);
    let mut ds = Vec::new();
    for (i, &q) in qs.iter().enumerate() {
        let sum = b.xor(&[q, carry]);
        carry = b.and(&[q, carry]);
        let d = b.and(&[sum, nrst]);
        ds.push(d);
        let outs = b.inst_named(&format!("ff{i}"), &dff, &[d, c]);
        b.connect(outs[0], q);
    }
    let next = b.concat(&ds);
    b.name(next, "next");
    let q = b.concat(&qs);
    b.output("Q", q);
    b.finish()
}

/// Small deterministic RNG (xorshift64*), so tests need no dependencies.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        (((self.next_u64() >> 32) * n as u64) >> 32) as usize
    }
    /// True with probability `p`.
    pub fn chance(&mut self, p: f64) -> bool {
        ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64) < p
    }
}

/// Parameters for [`random_design`].
pub struct RandomSpec {
    pub inputs: usize,
    pub gates: usize,
    /// Probability that a gate input is a feedback wire (driven by a later gate).
    pub feedback: f64,
    /// Number of tri-state buses (each with 2..=3 TRI drivers).
    pub buses: usize,
    /// Number of clocks (periods chosen at random).
    pub clocks: usize,
    /// Pick gate inputs mostly from the last `locality` signals (0 = uniform).
    pub locality: usize,
}

/// Random flat-ish design: inputs `I0..`, random 1-bit gates of every kind with 1..=4 inputs,
/// feedback wires, tri-state buses with random terminations and optional clocks.
pub fn random_design(rng: &mut Rng, spec: &RandomSpec) -> ComponentDef {
    use panda_netlist::{GateOp, Termination};
    let mut b = DefBuilder::new("rand");
    let mut pool: Vec<Sig> = (0..spec.inputs).map(|i| b.input(&format!("I{i}"), 1)).collect();
    for _ in 0..spec.clocks {
        let period = 2 * (1 + rng.below(12)) as u32;
        let high = 1 + rng.below(period as usize - 1) as u32;
        let phase = rng.below(period as usize) as u32;
        pool.push(b.clock(period, high, phase));
    }
    pool.push(b.lit(1, 0));
    pool.push(b.lit(1, 1));
    // Feedback wires, each driven by some later gate.
    let n_fb = ((spec.gates as f64) * spec.feedback).ceil() as usize + 1;
    let fb: Vec<Sig> = (0..n_fb).map(|_| b.wire(1)).collect();
    let mut fb_driven = vec![false; n_fb];
    let buses: Vec<Sig> = (0..spec.buses)
        .map(|_| {
            let w = b.wire(1);
            let t = match rng.below(4) {
                0 => Termination::None,
                1 => Termination::PullUp,
                2 => Termination::PullDown,
                _ => Termination::Keeper,
            };
            b.terminate(w, t);
            w
        })
        .collect();
    let ops = [GateOp::And, GateOp::Or, GateOp::Nand, GateOp::Nor, GateOp::Xor, GateOp::Xnor];
    for gi in 0..spec.gates {
        let pick = |rng: &mut Rng, pool: &Vec<Sig>| -> Sig {
            if rng.chance(spec.feedback) {
                fb[rng.below(n_fb)]
            } else if spec.locality > 0 && pool.len() > spec.locality && rng.chance(0.9) {
                pool[pool.len() - 1 - rng.below(spec.locality)]
            } else {
                pool[rng.below(pool.len())]
            }
        };
        let kind = rng.below(10);
        // Drive a feedback wire with this gate (if any are left), else a fresh net.
        let target = if gi % 3 == 0 {
            fb_driven.iter().position(|d| !d).map(|i| {
                fb_driven[i] = true;
                fb[i]
            })
        } else {
            None
        };
        let out = target.unwrap_or_else(|| b.wire(1));
        match kind {
            0 => {
                let a = pick(rng, &pool);
                b.not_to(a, out)
            }
            1 => {
                let a = pick(rng, &pool);
                b.buf_to(a, out)
            }
            _ => {
                let n = 1 + rng.below(4);
                let ins: Vec<Sig> = (0..n).map(|_| pick(rng, &pool)).collect();
                b.gate_to(ops[rng.below(ops.len())], &ins, out)
            }
        }
        pool.push(out);
    }
    for &bus in &buses {
        for _ in 0..2 + rng.below(2) {
            let d = pool[rng.below(pool.len())];
            let en = pool[rng.below(pool.len())];
            b.tri_to(d, en, bus);
        }
        let n = b.not(bus);
        pool.push(n);
    }
    // Expose some signals so nothing is pruned and names exist.
    for (i, s) in pool.iter().rev().take(8).enumerate() {
        b.output(&format!("O{i}"), *s);
    }
    b.finish()
}
