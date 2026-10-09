//! Random hierarchical design generator shared by the differential flattening tests.
//!
//! Designs contain random packages with random nesting, gates (incl. 1-input and reduce gates),
//! splitters (slices, bit-splits, concats, arbitrary overlapping parts), wide buses (> 64 bits),
//! constants with X/Z bits, tri-state buses with terminations, inout pins, multi-driver nets
//! (gates/constants onto shared wires), feedback loops, SR latches, clocks/resets in the top,
//! instance pins left unconnected and def pins left off every net.
#![allow(dead_code)]

use panda_circuits::{nand_sr_latch, ripple_adder};
use panda_netlist::{
    ComponentDef, DefBuilder, DefRef, Design, DesignBuilder, Dir, Endpoint, GateOp, Logic, Sig, Termination, Value,
};

use super::Rng;

pub const OPS: [GateOp; 6] = [GateOp::And, GateOp::Or, GateOp::Nand, GateOp::Nor, GateOp::Xor, GateOp::Xnor];
pub const TERMS: [Termination; 4] = [Termination::None, Termination::PullUp, Termination::PullDown, Termination::Keeper];

pub fn pick<T: Copy>(rng: &mut Rng, v: &[T]) -> T {
    v[rng.below(v.len() as u64) as usize]
}

pub fn chance(rng: &mut Rng, percent: u64) -> bool {
    rng.below(100) < percent
}

pub fn rand_width(rng: &mut Rng) -> u32 {
    match rng.below(20) {
        0 => 65 + rng.below(10) as u32, // wider than one 64-bit word
        1..=8 => 1,
        9..=12 => 2,
        13..=15 => 3,
        16..=17 => 4,
        _ => 8,
    }
}

pub fn rand_logic(rng: &mut Rng, xz_percent: u64) -> Logic {
    if chance(rng, xz_percent) {
        if chance(rng, 50) { Logic::X } else { Logic::Z }
    } else if chance(rng, 50) {
        Logic::One
    } else {
        Logic::Zero
    }
}

pub fn rand_value(rng: &mut Rng, width: u32, xz_percent: u64) -> Value {
    let bits: Vec<Logic> = (0..width).map(|_| rand_logic(rng, xz_percent)).collect();
    Value::from_bits(&bits)
}

/// Builder state for one random def.
struct Body<'r> {
    rng: &'r mut Rng,
    b: DefBuilder,
    /// Readable signals.
    pool: Vec<Sig>,
    /// Shared multi-driver wires (driven by tri/gates, read through the pool).
    buses: Vec<Sig>,
    /// Forward-referenced feedback wires that still need a driver.
    feedback: Vec<Sig>,
}

impl Body<'_> {
    /// A readable signal of exactly `w` bits (slicing / concatenating pool signals as needed).
    fn sig(&mut self, w: u32) -> Sig {
        let exact: Vec<Sig> = self.pool.iter().copied().filter(|s| s.width == w).collect();
        if !exact.is_empty() && chance(self.rng, 80) {
            return pick(self.rng, &exact);
        }
        if self.pool.is_empty() || chance(self.rng, 5) {
            let v = rand_value(self.rng, w, 15);
            let c = self.b.constant(v);
            self.pool.push(c);
            return c;
        }
        let s = pick(self.rng, &self.pool);
        if s.width == w {
            s
        } else if s.width > w {
            let lo = self.rng.below(u64::from(s.width - w + 1)) as u32;
            self.b.slice(s, lo, w)
        } else {
            let rest = self.sig(w - s.width);
            if chance(self.rng, 50) { self.b.concat(&[s, rest]) } else { self.b.concat(&[rest, s]) }
        }
    }

    fn new_bus(&mut self, w: u32) -> Sig {
        let bus = self.b.wire(w);
        let t = pick(self.rng, &TERMS);
        if t != Termination::None && chance(self.rng, 50) {
            self.b.terminate(bus, t);
        }
        self.buses.push(bus);
        self.pool.push(bus);
        bus
    }

    fn bus(&mut self, w: u32) -> Sig {
        let have: Vec<Sig> = self.buses.iter().copied().filter(|s| s.width == w).collect();
        if !have.is_empty() && chance(self.rng, 60) { pick(self.rng, &have) } else { self.new_bus(w) }
    }

    fn step(&mut self, children: &[DefRef]) {
        match self.rng.below(16) {
            0..=3 => {
                let w = pick(self.rng, &self.pool.iter().map(|s| s.width).collect::<Vec<_>>());
                let k = 1 + self.rng.below(3) as usize;
                let ins: Vec<Sig> = (0..k).map(|_| self.sig(w)).collect();
                let op = pick(self.rng, &OPS);
                let o = self.b.gate(op, &ins);
                self.pool.push(o);
            }
            4 => {
                let s = pick(self.rng, &self.pool);
                let o = if chance(self.rng, 50) { self.b.not(s) } else { self.b.buf(s) };
                self.pool.push(o);
            }
            5 => {
                let s = pick(self.rng, &self.pool);
                let op = pick(self.rng, &OPS);
                let o = self.b.reduce(op, s);
                self.pool.push(o);
            }
            6 => {
                let s = pick(self.rng, &self.pool);
                if s.width > 1 {
                    let o = if chance(self.rng, 30) {
                        let bits = self.b.bits(s);
                        pick(self.rng, &bits)
                    } else {
                        let w = 1 + self.rng.below(u64::from(s.width)) as u32;
                        let lo = self.rng.below(u64::from(s.width - w + 1)) as u32;
                        self.b.slice(s, lo, w)
                    };
                    self.pool.push(o);
                }
            }
            13 => {
                // Arbitrary splitter: random (possibly overlapping, unordered) parts.
                let s = pick(self.rng, &self.pool);
                let k = 1 + self.rng.below(4) as usize;
                let parts: Vec<(u32, u32)> = (0..k)
                    .map(|_| {
                        let lo = self.rng.below(u64::from(s.width)) as u32;
                        let hi = lo + self.rng.below(u64::from(s.width - lo)) as u32;
                        (lo, hi)
                    })
                    .collect();
                let outs = self.b.splitter(s, &parts);
                // Sometimes drive a part instead of reading it (the driver then reaches `s`).
                if chance(self.rng, 15) && self.buses.contains(&s) {
                    let p = outs[0];
                    let d = self.sig(p.width);
                    let en = self.sig(1);
                    self.b.tri_to(d, en, p);
                }
                self.pool.extend(outs);
            }
            7 => {
                let k = 2 + self.rng.below(2) as usize;
                let parts: Vec<Sig> = (0..k).map(|_| pick(self.rng, &self.pool)).collect();
                if parts.iter().map(|p| p.width).sum::<u32>() <= 80 {
                    let o = self.b.concat(&parts);
                    self.pool.push(o);
                }
            }
            8 => {
                let w = rand_width(self.rng);
                let v = rand_value(self.rng, w, 20);
                let o = self.b.constant(v);
                self.pool.push(o);
            }
            9 => {
                // Tri-state driver onto a (possibly shared) bus.
                let w = pick(self.rng, &self.pool.iter().map(|s| s.width).collect::<Vec<_>>());
                let d = self.sig(w);
                let en = self.sig(1);
                let bus = self.bus(w);
                self.b.tri_to(d, en, bus);
            }
            10 => {
                // A plain gate or constant onto a bus (non-Z-capable multi-driver net).
                let w = pick(self.rng, &self.pool.iter().map(|s| s.width).collect::<Vec<_>>());
                let bus = self.bus(w);
                if chance(self.rng, 50) {
                    let ins = [self.sig(w), self.sig(w)];
                    self.b.gate_to(pick(self.rng, &OPS), &ins, bus);
                } else {
                    let v = rand_value(self.rng, w, 40);
                    let c = self.b.constant(v);
                    self.b.connect(c, bus);
                }
            }
            11 => {
                // Feedback: a wire read now and driven at the end of the body.
                let w = if chance(self.rng, 70) { 1 } else { rand_width(self.rng) };
                let f = self.b.wire(w);
                self.feedback.push(f);
                self.pool.push(f);
            }
            _ => {
                if children.is_empty() {
                    return;
                }
                let c = children[self.rng.below(children.len() as u64) as usize].clone();
                let mut ins = Vec::new();
                for p in c.pins.iter().filter(|p| p.dir != Dir::Out) {
                    let s = if p.dir == Dir::InOut { self.bus(p.width) } else { self.sig(p.width) };
                    ins.push(s);
                }
                let outs = self.b.inst(&c, &ins);
                self.pool.extend(outs);
            }
        }
    }
}

/// Generate a random def that may instantiate `children`.
pub fn random_def(rng: &mut Rng, id: &str, children: &[DefRef], top: bool) -> ComponentDef {
    let mut body = Body { rng, b: DefBuilder::new(id), pool: Vec::new(), buses: Vec::new(), feedback: Vec::new() };
    let n_in = 1 + body.rng.below(4);
    for i in 0..n_in {
        let w = rand_width(body.rng);
        let s = body.b.input(&format!("I{i}"), w);
        body.pool.push(s);
    }
    if top && chance(body.rng, 40) {
        let period = 2 + body.rng.below(12) as u32;
        let high = 1 + body.rng.below(u64::from(period - 1)) as u32;
        let phase = body.rng.below(u64::from(period) + 3) as u32;
        let c = body.b.clock(period, high, phase);
        body.pool.push(c);
    }
    if top && chance(body.rng, 30) {
        let r = body.b.power_on_reset(body.rng.below(20) as u32);
        body.pool.push(r);
    }
    let n_io = if chance(body.rng, 35) { 1 + body.rng.below(2) } else { 0 };
    for i in 0..n_io {
        let w = rand_width(body.rng);
        let s = body.b.inout(&format!("IO{i}"), w);
        if chance(body.rng, 30) {
            body.b.terminate(s, pick(body.rng, &TERMS));
        }
        body.buses.push(s);
        body.pool.push(s);
    }
    let steps = 4 + body.rng.below(14);
    for _ in 0..steps {
        body.step(children);
    }
    // Drive the feedback wires (mostly through an inverting or AND/OR gate).
    for f in std::mem::take(&mut body.feedback) {
        let ins = [body.sig(f.width), body.sig(f.width)];
        body.b.gate_to(pick(body.rng, &OPS), &ins, f);
    }
    let n_out = 1 + body.rng.below(3);
    for i in 0..n_out {
        let w = pick(body.rng, &body.pool.iter().map(|s| s.width).collect::<Vec<_>>());
        let s = body.sig(w);
        body.b.output(&format!("O{i}"), s);
    }
    let rng = body.rng;
    let mut def = body.b.finish();
    // Take some def pins off every net (their bits become private to the def).
    if chance(rng, 10) {
        let victim = def.pins[rng.below(def.pins.len() as u64) as usize].name.clone();
        for n in &mut def.nets {
            n.ends.retain(|e| *e != Endpoint::Pin(victim.clone()));
            // Keep hierarchical names unambiguous (a net named like a pin it is not on).
            if n.id == victim {
                n.id = format!("{victim}_net");
            }
        }
    }
    // Leave some def-instance pins unconnected (drop their endpoints).
    for n in &mut def.nets {
        n.ends.retain(|e| match e {
            Endpoint::Port { inst, .. } if inst.starts_with('u') => !chance(rng, 8),
            _ => true,
        });
    }
    def
}

/// A random hierarchical design: `n` packages, each may use any earlier package (plus a few
/// library blocks), and a top that uses them.
pub fn random_design(seed: u64) -> Design {
    let mut rng = Rng::new(seed);
    let mut db = DesignBuilder::new();
    let mut children: Vec<DefRef> = Vec::new();
    if chance(&mut rng, 40) {
        children.push(nand_sr_latch(&mut db));
    }
    if chance(&mut rng, 30) {
        children.push(ripple_adder(&mut db, 1 + rng.below(4) as u32));
    }
    let n = 1 + rng.below(6);
    for i in 0..n {
        // Each package sees a random subset of the earlier ones (random nesting depth).
        let avail: Vec<DefRef> = children.iter().filter(|_| chance(&mut rng, 70)).cloned().collect();
        let def = random_def(&mut rng, &format!("pkg{i}"), &avail, false);
        children.push(db.add(def));
    }
    let top = random_def(&mut rng, "top", &children, true);
    let top = db.add(top);
    let mut d = db.build(&top);
    // Drop defs the top does not reach (validation would still check them; flattening and
    // inlining ignore them).
    let mut keep = std::collections::HashSet::from([d.top.clone()]);
    let mut stack = vec![d.top.clone()];
    while let Some(id) = stack.pop() {
        for i in &d.def(&id).expect("def").instances {
            if let panda_netlist::InstKind::Def(c) = &i.kind
                && keep.insert(c.clone())
            {
                stack.push(c.clone());
            }
        }
    }
    d.defs.retain(|x| keep.contains(&x.id));
    d
}

