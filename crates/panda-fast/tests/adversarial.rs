//! Adversarial FastSim-vs-FastRef differential tests: a block-structured random netlist
//! generator aimed at the corners the other generators rarely reach, plus hand-written
//! regression designs.
//!
//! The generator composes, at flat level: NAND / mux / SR latches, master-slave and 6-NAND
//! edge-triggered flip-flops (with power-on-reset clears), ripple counters (derived clocks),
//! gated clocks (non-robust kernels), several clocks with coincident edges, power-on resets
//! released on clock edges, gated ring oscillators, long-loop latches (sim kernels), keeper /
//! pull / floating tri-state buses read by LUT cones, LUT cones over `X`/`Z` constants and raw
//! tri-state outputs, mux-tree ROMs, kernels with constant inputs, wide fan-out nets and kernel
//! banks crossing the 64-kernel bitset word boundary. Every event is checked net by net through
//! `get()` (always-up-to-date nets first, so the lazy-net bookkeeping is exercised without a
//! preceding `values()`), `read()` / `read_pin()`, then `values()`, tick, schedule, statistics
//! and diagnostic kinds.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use common::{Coverage, check, pair, random_compile, random_value};
use panda_compile::FastOptions;
use panda_compile::FastRef;
use panda_compile::kernels::ExtractOptions;
use panda_difftest::Rng;
use panda_fast::{CompileOptions, FastSim};
use panda_netlist::flat::{
    FlatGate, FlatInstance, FlatNetlist, FlatOp, FlatPin, GateGroup, NetIdx, Resolver, Source,
    SourceKind,
};
use panda_netlist::{Dir, Logic, Termination, Value};

// ---------------------------------------------------------------------------------------------
// Flat netlist builder
// ---------------------------------------------------------------------------------------------

/// A minimal flat-netlist builder.
struct B {
    f: FlatNetlist,
}

impl B {
    fn new() -> B {
        let mut f = FlatNetlist::default();
        f.instances.push(FlatInstance {
            path: String::new(),
            def: "top".into(),
            parent: None,
            children: Vec::new(),
            pins: Vec::new(),
        });
        B { f }
    }

    fn net(&mut self, name: &str) -> NetIdx {
        let n = self.f.num_nets;
        let name = if self.f.name_index.contains_key(name) {
            format!("{name}_{n}")
        } else {
            name.to_string()
        };
        self.f.num_nets += 1;
        self.f.net_names.push(name.clone());
        self.f.name_index.insert(name, n);
        n
    }

    /// A top-level input pin of `width` bits (nets `name[i]` if wider than 1).
    fn input_bus(&mut self, name: &str, width: u32) -> Vec<NetIdx> {
        let pin = self.f.top_inputs.len() as u32;
        let nets: Vec<NetIdx> = (0..width)
            .map(|b| {
                let n = if width == 1 {
                    self.net(name)
                } else {
                    self.net(&format!("{name}[{b}]"))
                };
                self.f.sources.push(Source {
                    kind: SourceKind::TopInput { pin, bit: b },
                    output: n,
                });
                n
            })
            .collect();
        self.f.top_inputs.push(FlatPin {
            name: name.to_string(),
            dir: Dir::In,
            nets: nets.clone(),
        });
        nets
    }

    fn input(&mut self, name: &str) -> NetIdx {
        self.input_bus(name, 1)[0]
    }

    fn source(&mut self, name: &str, kind: SourceKind) -> NetIdx {
        let n = self.net(name);
        self.f.sources.push(Source { kind, output: n });
        n
    }

    fn clock(&mut self, name: &str, period: u32, high: u32, phase: u32) -> NetIdx {
        self.source(
            name,
            SourceKind::Clock {
                period,
                high,
                phase,
            },
        )
    }

    fn por(&mut self, name: &str, ticks: u32) -> NetIdx {
        self.source(name, SourceKind::PowerOnReset { ticks })
    }

    fn constant(&mut self, name: &str, v: Logic) -> NetIdx {
        self.source(name, SourceKind::Constant(v))
    }

    fn gate_to(&mut self, op: FlatOp, inputs: &[NetIdx], out: NetIdx) {
        let gi = self.f.gates.len() as u32;
        let group = self.f.groups.len() as u32;
        self.f.gates.push(FlatGate {
            op,
            inputs: inputs.to_vec(),
            output: out,
            owner: 0,
            group,
            bit: 0,
        });
        self.f.groups.push(GateGroup {
            owner: 0,
            path: format!("g{gi}"),
            gates: vec![gi],
        });
    }

    fn gate(&mut self, name: &str, op: FlatOp, inputs: &[NetIdx]) -> NetIdx {
        let out = self.net(name);
        self.gate_to(op, inputs, out);
        out
    }

    fn not(&mut self, name: &str, a: NetIdx) -> NetIdx {
        self.gate(name, FlatOp::Not, &[a])
    }

    fn nand(&mut self, name: &str, ins: &[NetIdx]) -> NetIdx {
        self.gate(name, FlatOp::Nand, ins)
    }

    fn resolver(&mut self, name: &str, drivers: &[NetIdx], term: Termination) -> NetIdx {
        let out = self.net(name);
        self.f.resolvers.push(Resolver {
            drivers: drivers.to_vec(),
            termination: term,
            output: out,
        });
        out
    }

    fn output(&mut self, name: &str, n: NetIdx) {
        self.f.top_outputs.push(FlatPin {
            name: name.to_string(),
            dir: Dir::Out,
            nets: vec![n],
        });
    }

    /// NAND D latch (transparent while `en` = 1); returns `q`.
    fn d_latch(&mut self, p: &str, d: NetIdx, en: NetIdx) -> NetIdx {
        let dn = self.not(&format!("{p}.dn"), d);
        let s = self.nand(&format!("{p}.s"), &[d, en]);
        let r = self.nand(&format!("{p}.r"), &[dn, en]);
        let q = self.net(&format!("{p}.q"));
        let qn = self.net(&format!("{p}.qn"));
        self.gate_to(FlatOp::Nand, &[s, qn], q);
        self.gate_to(FlatOp::Nand, &[r, q], qn);
        q
    }

    /// Mux latch `q = d·en + q·!en` (hazard on `en` changes: order-sensitive entries).
    fn mux_latch(&mut self, p: &str, d: NetIdx, en: NetIdx) -> NetIdx {
        let enn = self.not(&format!("{p}.enn"), en);
        let q = self.net(&format!("{p}.q"));
        let a = self.gate(&format!("{p}.a"), FlatOp::And, &[d, en]);
        let b = self.gate(&format!("{p}.b"), FlatOp::And, &[q, enn]);
        self.gate_to(FlatOp::Or, &[a, b], q);
        q
    }

    /// NAND SR latch (active-low set / reset); returns `q`.
    fn sr_nand(&mut self, p: &str, sn: NetIdx, rn: NetIdx) -> NetIdx {
        let q = self.net(&format!("{p}.q"));
        let qn = self.net(&format!("{p}.qn"));
        self.gate_to(FlatOp::Nand, &[sn, qn], q);
        self.gate_to(FlatOp::Nand, &[rn, q], qn);
        q
    }

    /// NOR SR latch (active-high set / reset); returns `q`.
    fn sr_nor(&mut self, p: &str, s: NetIdx, r: NetIdx) -> NetIdx {
        let q = self.net(&format!("{p}.q"));
        let qn = self.net(&format!("{p}.qn"));
        self.gate_to(FlatOp::Nor, &[r, qn], q);
        self.gate_to(FlatOp::Nor, &[s, q], qn);
        q
    }

    /// Master-slave flip-flop on the rising edge of `clk`.
    fn ms_dff(&mut self, p: &str, d: NetIdx, clk: NetIdx) -> NetIdx {
        let clkn = self.not(&format!("{p}.clkn"), clk);
        let m = self.d_latch(&format!("{p}.m"), d, clkn);
        self.d_latch(&format!("{p}.s"), m, clk)
    }

    /// Classic 6-NAND edge-triggered D flip-flop with active-low asynchronous clear.
    fn dff6(&mut self, p: &str, d: NetIdx, clk: NetIdx, clrn: NetIdx) -> NetIdx {
        let n1 = self.net(&format!("{p}.n1"));
        let n2 = self.net(&format!("{p}.n2"));
        let n3 = self.net(&format!("{p}.n3"));
        let n4 = self.net(&format!("{p}.n4"));
        self.gate_to(FlatOp::Nand, &[n4, n2], n1);
        self.gate_to(FlatOp::Nand, &[n1, clk, clrn], n2);
        self.gate_to(FlatOp::Nand, &[n2, clk, n4], n3);
        self.gate_to(FlatOp::Nand, &[n3, d, clrn], n4);
        let q = self.net(&format!("{p}.q"));
        let qn = self.net(&format!("{p}.qn"));
        self.gate_to(FlatOp::Nand, &[n2, qn], q);
        self.gate_to(FlatOp::Nand, &[q, n3, clrn], qn);
        q
    }

    /// Gated ring oscillator (3 stages, runs while `en` = 1).
    fn ring(&mut self, p: &str, en: NetIdx) -> NetIdx {
        let n0 = self.net(&format!("{p}.r0"));
        let n1 = self.not(&format!("{p}.r1"), n0);
        let n2 = self.not(&format!("{p}.r2"), n1);
        self.gate_to(FlatOp::Nand, &[en, n2], n0);
        n2
    }

    /// Set/hold latch through a loop of `len` buffers (longer than the kernel seed limit: a
    /// residual SCC, i.e. a sim kernel unless the extraction options allow long loops).
    fn long_latch(&mut self, p: &str, set: NetIdx, hold: NetIdx, len: usize) -> NetIdx {
        self.long_latch_nets(p, set, hold, len)[0]
    }

    /// [`B::long_latch`] returning every loop net (head first).
    fn long_latch_nets(&mut self, p: &str, set: NetIdx, hold: NetIdx, len: usize) -> Vec<NetIdx> {
        let head = self.net(&format!("{p}.h"));
        let mut nets = vec![head];
        let mut cur = head;
        for i in 0..len {
            cur = self.gate(&format!("{p}.b{i}"), FlatOp::Buf, &[cur]);
            nets.push(cur);
        }
        let keep = self.gate(&format!("{p}.k"), FlatOp::And, &[cur, hold]);
        self.gate_to(FlatOp::Or, &[set, keep], head);
        nets
    }
}

// ---------------------------------------------------------------------------------------------
// Block-structured random generator
// ---------------------------------------------------------------------------------------------

const OPS2: [FlatOp; 6] = [
    FlatOp::And,
    FlatOp::Or,
    FlatOp::Nand,
    FlatOp::Nor,
    FlatOp::Xor,
    FlatOp::Xnor,
];

/// Generator state: the pool of signals blocks may read.
struct Gen {
    b: B,
    rng: Rng,
    /// Every readable signal.
    pool: Vec<NetIdx>,
    /// Clock-like signals (clocks, functions of clocks, gated clocks, derived clocks).
    clocks: Vec<NetIdx>,
    /// Raw clock source nets.
    raw_clocks: Vec<NetIdx>,
    /// Power-on resets (active high) if any.
    pors: Vec<NetIdx>,
    /// Constants (`0`, `1`, `X`, `Z`).
    consts: Vec<NetIdx>,
    blk: usize,
}

impl Gen {
    fn pick(&mut self) -> NetIdx {
        *self.rng.pick(&self.pool)
    }

    fn pick_clock(&mut self) -> NetIdx {
        if self.clocks.is_empty() || self.rng.chance(15) {
            self.pick()
        } else {
            *self.rng.pick(&self.clocks)
        }
    }

    fn name(&mut self, kind: &str) -> String {
        self.blk += 1;
        format!("{kind}{}", self.blk)
    }

    /// A random combinational cone over `support` (internal nets private to the cone), `size`
    /// gates; optionally with tri-state gates inside. Returns the cone outputs.
    fn cone(&mut self, support: &[NetIdx], size: usize, tri: bool, outs: usize) -> Vec<NetIdx> {
        let p = self.name("c");
        let mut avail = support.to_vec();
        let mut made = Vec::new();
        for i in 0..size {
            let op = if tri && self.rng.chance(12) {
                FlatOp::Tri
            } else if self.rng.chance(12) {
                FlatOp::Not
            } else {
                *self.rng.pick(&OPS2)
            };
            let arity = match op {
                FlatOp::Not => 1,
                FlatOp::Tri => 2,
                _ => 2 + usize::from(self.rng.chance(20)),
            };
            // Prefer recent nets (deep cones), sometimes the support.
            let ins: Vec<NetIdx> = (0..arity)
                .map(|_| {
                    if !made.is_empty() && self.rng.chance(60) {
                        let lo = made.len().saturating_sub(4);
                        made[self.rng.range(lo, made.len() - 1)]
                    } else {
                        *self.rng.pick(&avail)
                    }
                })
                .collect();
            let o = self.b.gate(&format!("{p}.n{i}"), op, &ins);
            made.push(o);
            avail.push(o);
        }
        let mut res = vec![*made.last().unwrap()];
        for _ in 1..outs {
            let o = *self.rng.pick(&made);
            if !res.contains(&o) {
                res.push(o);
            }
        }
        res
    }

    /// Mux-tree ROM over `addr` (data: constants, occasionally `X`).
    fn rom(&mut self, addr: &[NetIdx]) -> NetIdx {
        let p = self.name("rom");
        let mut level: Vec<NetIdx> = (0..1usize << addr.len())
            .map(|i| {
                let v = if self.rng.chance(5) {
                    Logic::X
                } else {
                    Logic::from_bool(self.rng.chance(50))
                };
                self.b.constant(&format!("{p}.k{i}"), v)
            })
            .collect();
        for (bi, &a) in addr.iter().enumerate() {
            let an = self.b.not(&format!("{p}.an{bi}"), a);
            level = level
                .chunks(2)
                .enumerate()
                .map(|(j, c)| {
                    let x = self
                        .b
                        .gate(&format!("{p}.l{bi}x{j}"), FlatOp::And, &[c[0], an]);
                    let y = self
                        .b
                        .gate(&format!("{p}.l{bi}y{j}"), FlatOp::And, &[c[1], a]);
                    self.b.gate(&format!("{p}.l{bi}o{j}"), FlatOp::Or, &[x, y])
                })
                .collect();
        }
        level[0]
    }

    fn block(&mut self) {
        let kind = self.rng.weighted(&[
            10, // 0 d latch
            5,  // 1 mux latch
            5,  // 2 sr nand
            4,  // 3 sr nor
            9,  // 4 ms dff
            6,  // 5 dff6 with clear
            4,  // 6 ripple counter (derived clocks)
            6,  // 7 gated clock
            3,  // 8 ring oscillator
            3,  // 9 long latch
            6,  // 10 tri-state bus
            12, // 11 comb cone (LUT candidate)
            4,  // 12 rom
            3,  // 13 latch bank (wide fan-out, > 64 kernels)
            3,  // 14 constant-input kernel
            4,  // 15 pure clock function
            2,  // 16 wide sim kernel (> 31 inputs, many outputs)
        ]);
        match kind {
            0 => {
                let p = self.name("dl");
                let (d, en) = (self.pick(), self.pick_clock());
                let q = self.b.d_latch(&p, d, en);
                self.pool.push(q);
                if self.rng.chance(40) {
                    // Both latch outputs read outside (two-output kernel).
                    let qn = self.b.f.name_index[&format!("{p}.qn")];
                    self.pool.push(qn);
                }
            }
            1 => {
                let p = self.name("ml");
                let (d, en) = (self.pick(), self.pick_clock());
                let q = self.b.mux_latch(&p, d, en);
                self.pool.push(q);
            }
            2 => {
                let p = self.name("srn");
                let (s, r) = (self.pick(), self.pick());
                let q = self.b.sr_nand(&p, s, r);
                self.pool.push(q);
            }
            3 => {
                let p = self.name("srr");
                let (s, r) = (self.pick(), self.pick());
                let q = self.b.sr_nor(&p, s, r);
                self.pool.push(q);
            }
            4 => {
                let p = self.name("ff");
                let (d, clk) = (self.pick(), self.pick_clock());
                let q = self.b.ms_dff(&p, d, clk);
                self.pool.push(q);
            }
            5 => {
                let p = self.name("f6");
                let (d, clk) = (self.pick(), self.pick_clock());
                let clrn = if !self.pors.is_empty() && self.rng.chance(70) {
                    let r = *self.rng.pick(&self.pors);
                    self.b.not(&format!("{p}.clrn"), r)
                } else {
                    self.pick()
                };
                let q = self.b.dff6(&p, d, clk, clrn);
                self.pool.push(q);
            }
            6 => {
                let p = self.name("rc");
                let mut clk = self.pick_clock();
                for i in 0..1 + self.rng.below(3) {
                    let q = self.b.net(&format!("{p}.t{i}"));
                    let qn = self.b.not(&format!("{p}.tn{i}"), q);
                    let clkn = self.b.not(&format!("{p}.ck{i}"), clk);
                    let m = self.b.d_latch(&format!("{p}.m{i}"), qn, clkn);
                    let s = self.b.d_latch(&format!("{p}.s{i}"), m, clk);
                    self.b.gate_to(FlatOp::Buf, &[s], q);
                    self.pool.push(q);
                    self.clocks.push(q);
                    clk = q;
                }
            }
            7 => {
                let p = self.name("gc");
                let clk = if self.raw_clocks.is_empty() {
                    self.pick()
                } else {
                    *self.rng.pick(&self.raw_clocks)
                };
                let en = self.pick();
                let op = *self
                    .rng
                    .pick(&[FlatOp::And, FlatOp::Or, FlatOp::Nand, FlatOp::Xor]);
                let g = self.b.gate(&format!("{p}.g"), op, &[clk, en]);
                self.clocks.push(g);
                self.pool.push(g);
            }
            8 => {
                let p = self.name("ro");
                let en = self.pick();
                let o = self.b.ring(&p, en);
                self.pool.push(o);
            }
            9 => {
                let p = self.name("ll");
                let (s, h) = (self.pick(), self.pick_clock());
                let len = 3 + self.rng.below(8);
                let nets = self.b.long_latch_nets(&p, s, h, len);
                // Many loop nets read outside: a multi-output kernel (wide output tables).
                let take = if self.rng.chance(50) { nets.len() } else { 1 };
                self.pool.extend(&nets[..take]);
            }
            10 => {
                let p = self.name("bus");
                let term = *self.rng.pick(&[
                    Termination::None,
                    Termination::PullUp,
                    Termination::PullDown,
                    Termination::Keeper,
                ]);
                let drv: Vec<NetIdx> = (0..1 + self.rng.below(3))
                    .map(|i| {
                        let (d, en) = (self.pick(), self.pick());
                        self.b.gate(&format!("{p}.d{i}"), FlatOp::Tri, &[d, en])
                    })
                    .collect();
                let o = self.b.resolver(&p, &drv, term);
                self.pool.push(o);
                if self.rng.chance(30) {
                    // A raw tri-state output read by gates (Z read as X).
                    self.pool.push(drv[0]);
                }
            }
            11 => {
                let ns = 2 + self.rng.below(6);
                let mut support: Vec<NetIdx> = (0..ns).map(|_| self.pick()).collect();
                if self.rng.chance(30) {
                    support.push(*self.rng.pick(&self.consts.clone()));
                }
                support.sort_unstable();
                support.dedup();
                let size = 3 + self.rng.below(40);
                let tri = self.rng.chance(30);
                let outs = 1 + self.rng.below(3);
                let o = self.cone(&support, size, tri, outs);
                self.pool.extend(o);
            }
            12 => {
                let na = 2 + self.rng.below(4);
                let addr: Vec<NetIdx> = (0..na).map(|_| self.pick()).collect();
                let o = self.rom(&addr);
                self.pool.push(o);
            }
            13 => {
                let p = self.name("bank");
                let n = *self.rng.pick(&[40usize, 65, 70, 130]);
                let en = self.pick_clock();
                let d = self.pick();
                let mut qs = Vec::new();
                for i in 0..n {
                    // Alternate data polarities so the bank is not one structural class only.
                    let di = if i % 3 == 0 {
                        d
                    } else {
                        let x = self.pick();
                        self.b.gate(&format!("{p}.x{i}"), FlatOp::Xor, &[d, x])
                    };
                    qs.push(self.b.d_latch(&format!("{p}.l{i}"), di, en));
                }
                // A wide reduction over the bank (one LUT-sized cone is impossible; plain ops).
                let r = self.b.gate(&format!("{p}.xr"), FlatOp::Xor, &qs);
                self.pool.push(r);
                for _ in 0..3 {
                    let q = *self.rng.pick(&qs);
                    self.pool.push(q);
                }
            }
            14 => {
                let p = self.name("kc");
                let c = *self.rng.pick(&self.consts.clone());
                let x = self.pick();
                let q = match self.rng.below(3) {
                    0 => self.b.d_latch(&p, x, c),
                    1 => self.b.d_latch(&p, c, x),
                    _ => self.b.sr_nand(&p, c, x),
                };
                self.pool.push(q);
            }
            16 => {
                // A loop of `len` gates, each also reading `fan` external nets: a sim kernel
                // with more than 31 inputs; every loop net is read outside (many outputs).
                let p = self.name("wk");
                let len = 7 + self.rng.below(30);
                let fan = 1 + self.rng.below(5);
                let head = self.b.net(&format!("{p}.h"));
                let mut cur = head;
                let mut loop_nets = vec![head];
                for i in 0..len {
                    let mut ins = vec![cur];
                    for _ in 0..fan {
                        ins.push(self.pick());
                    }
                    let op = *self.rng.pick(&[FlatOp::Or, FlatOp::And, FlatOp::Xor]);
                    let out = if i + 1 == len {
                        head
                    } else {
                        self.b.net(&format!("{p}.n{i}"))
                    };
                    self.b.gate_to(op, &ins, out);
                    if out != head {
                        loop_nets.push(out);
                    }
                    cur = out;
                }
                let r = self.b.gate(&format!("{p}.r"), FlatOp::Xor, &loop_nets);
                self.pool.push(r);
                self.pool.push(loop_nets[loop_nets.len() / 2]);
            }
            _ => {
                if self.raw_clocks.is_empty() {
                    return;
                }
                let p = self.name("pc");
                let a = *self.rng.pick(&self.raw_clocks);
                let bsig = if self.rng.chance(50) {
                    *self.rng.pick(&self.raw_clocks)
                } else if !self.pors.is_empty() && self.rng.chance(50) {
                    *self.rng.pick(&self.pors)
                } else {
                    *self.rng.pick(&self.consts.clone())
                };
                let op = *self.rng.pick(&OPS2);
                let g = self.b.gate(&format!("{p}.g"), op, &[a, bsig]);
                let g = if self.rng.chance(50) {
                    self.b.not(&format!("{p}.n"), g)
                } else {
                    g
                };
                self.clocks.push(g);
                self.pool.push(g);
            }
        }
    }
}

/// A block-structured random netlist (see the module docs).
fn adversarial_netlist(seed: u64) -> Arc<FlatNetlist> {
    let mut rng = Rng::new(seed.wrapping_mul(0xA24B_AED4_963E_E407) ^ 0xADD5);
    let mut b = B::new();
    let mut pool = Vec::new();
    for i in 0..1 + rng.below(4) {
        pool.push(b.input(&format!("i{i}")));
    }
    if rng.chance(30) {
        pool.extend(b.input_bus("bus_in", 2 + rng.below(4) as u32));
    }
    let mut raw_clocks = Vec::new();
    let nclk = rng.weighted(&[2, 5, 3, 1]);
    let base = *rng.pick(&[2u32, 4, 6, 8]);
    let mut shapes: Vec<(u32, u32, u32)> = Vec::new();
    for c in 0..nclk {
        let shape = if c > 0 && rng.chance(30) {
            // An identical clock (every edge coincides) or one with the same edges shifted.
            let s = shapes[0];
            if rng.chance(50) {
                s
            } else {
                (s.0, s.1, (s.2 + s.0 / 2) % s.0)
            }
        } else {
            let period = base * *rng.pick(&[1u32, 1, 2, 3]);
            let high = 1 + rng.below(period as usize - 1) as u32;
            // Phases that make edges coincide across clocks more often than not.
            let phase = if rng.chance(60) {
                0
            } else {
                rng.below(period as usize) as u32
            };
            (period, high, phase)
        };
        shapes.push(shape);
        raw_clocks.push(b.clock(&format!("clk{c}"), shape.0, shape.1, shape.2));
    }
    match rng.below(12) {
        // Degenerate clocks (never change; odd shapes).
        0 => {
            let (period, high, phase) = *rng.pick(&[
                (0u32, 1u32, 0u32),
                (0, 0, 3),
                (6, 0, 1),
                (4, 4, 0),
                (4, 9, 2),
                (5, 2, 17),
                (1, 1, 0),
                (3, 1, 1),
            ]);
            raw_clocks.push(b.clock("dclk", period, high, phase));
        }
        // More timed sources than phase specialization handles (no tuple classes).
        1 => {
            for c in 0..7 {
                raw_clocks.push(b.clock(&format!("xclk{c}"), 2 * (c + 1), c + 1, c));
            }
        }
        // A hyperperiod too long to enumerate (every tuple pair possible).
        2 => {
            raw_clocks.push(b.clock("hclk0", 2 * 4093, 4093, 0));
            raw_clocks.push(b.clock("hclk1", 2 * 4091, 3, 1));
            raw_clocks.push(b.clock("hclk2", 4, 2, 0));
        }
        _ => {}
    }
    let mut pors = Vec::new();
    if rng.chance(50) {
        // Often released exactly on a clock edge (multiples of the base period).
        let ticks = if rng.chance(60) {
            base * (1 + rng.below(4) as u32)
        } else {
            rng.below(15) as u32
        };
        pors.push(b.por("por", ticks));
    }
    let consts = vec![
        b.constant("k0", Logic::Zero),
        b.constant("k1", Logic::One),
        b.constant("kx", Logic::X),
        b.constant("kz", Logic::Z),
    ];
    if rng.chance(30) {
        // Undriven nets: floating (Z) or with a termination (pull-up/down: binary constants;
        // keeper: X forever).
        for (i, term) in [
            Termination::None,
            Termination::PullUp,
            Termination::PullDown,
            Termination::Keeper,
        ]
        .into_iter()
        .enumerate()
        {
            if rng.chance(60) {
                let n = b.net(&format!("u{i}"));
                if term != Termination::None || rng.chance(50) {
                    b.f.undriven_termination.insert(n, term);
                }
                pool.push(n);
            }
        }
    }
    pool.extend(&raw_clocks);
    pool.extend(&pors);
    if rng.chance(50) {
        pool.extend(&consts);
    }
    let mut g = Gen {
        b,
        rng,
        pool,
        clocks: raw_clocks.clone(),
        raw_clocks,
        pors,
        consts,
        blk: 0,
    };
    let blocks = 3 + g.rng.below(12);
    for _ in 0..blocks {
        g.block();
    }
    // Feedback through combinational logic between blocks: a few gates reading the newest
    // signals drive a late latch (closing loops through kernels).
    if g.rng.chance(40) {
        let p = g.name("fb");
        let (a, c) = (g.pick(), g.pick());
        let x = g.b.gate(&format!("{p}.x"), FlatOp::Xor, &[a, c]);
        let en = g.pick_clock();
        let q = g.b.d_latch(&p, x, en);
        g.pool.push(q);
    }
    let n = g.pool.len();
    for o in 0..3.min(n) {
        let net = g.pool[n - 1 - o];
        g.b.output(&format!("o{o}"), net);
    }
    Arc::new(g.b.f)
}

// ---------------------------------------------------------------------------------------------
// Lockstep driver with per-net get() / read() checks
// ---------------------------------------------------------------------------------------------

/// Compare every net through `get()` (FastSim's always-up-to-date nets first, then the lazy
/// ones), every net name through `read()`, every pin through `read_pin()`; before `values()`.
fn check_get_read(fr: &FastRef, fs: &FastSim, flat: &FlatNetlist, ctx: &str, rng: &mut Rng) {
    let lazy = &fs.program().lazy_owner;
    let n = flat.num_nets as usize;
    let mut order: Vec<usize> = (0..n).filter(|&i| lazy[i] == u32::MAX).collect();
    let mut lazies: Vec<usize> = (0..n).filter(|&i| lazy[i] != u32::MAX).collect();
    // Random order within each group.
    for v in [&mut order, &mut lazies] {
        for i in (1..v.len()).rev() {
            let j = rng.below(i + 1);
            v.swap(i, j);
        }
    }
    order.extend(lazies);
    for net in order {
        let (a, b) = (fr.get(net as u32), fs.get(net as u32));
        assert_eq!(
            a, b,
            "{ctx}: get({}) ref {a} fast {b} (lazy owner {})",
            flat.net_names[net], lazy[net]
        );
    }
    for name in flat.net_names.iter().take(64) {
        assert_eq!(fr.read(name), fs.read(name), "{ctx}: read({name})");
    }
    for p in flat.top_inputs.iter().chain(&flat.top_outputs) {
        assert_eq!(
            fr.read_pin(&p.name),
            fs.read_pin(&p.name),
            "{ctx}: read_pin({})",
            p.name
        );
        assert_eq!(
            fr.read(&p.name),
            fs.read(&p.name),
            "{ctx}: read({})",
            p.name
        );
    }
}

/// The pending diagnostics agree in kind, location and message (as multisets), not only in
/// kind; leaves them pending for [`check`].
fn check_diag_details(fr: &FastRef, fs: &FastSim, ctx: &str) {
    let full = |d: &[panda_netlist::Diagnostic]| {
        let mut v: Vec<String> = d
            .iter()
            .map(|d| {
                format!(
                    "{:?} {:?} {:?} {}",
                    d.kind, d.severity, d.location, d.message
                )
            })
            .collect();
        v.sort();
        v
    };
    assert_eq!(
        full(fr.diagnostics()),
        full(fs.diagnostics()),
        "{ctx}: diagnostic details"
    );
}

/// Drive both engines with an adversarial event sequence: bursts of input changes (the same pin
/// set many times, sometimes back to its old value), X/Z input values, scheduled events, resets.
#[allow(clippy::too_many_arguments)]
fn lockstep(
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
    check_get_read(fr, fs, flat, &format!("{ctx} power-on"), rng);
    check(fr, fs, flat, &format!("{ctx} power-on"), cov);
    let pins: Vec<(String, u32)> = flat
        .top_inputs
        .iter()
        .map(|p| (p.name.clone(), p.nets.len() as u32))
        .collect();
    for i in 0..actions {
        let roll = rng.below(100);
        let actx = format!("{ctx} action {i}");
        if roll < 35 && !pins.is_empty() {
            let burst = if rng.chance(20) { 2 + rng.below(8) } else { 1 };
            for _ in 0..burst {
                let (name, w) = rng.pick(&pins).clone();
                let v = if rng.chance(15) {
                    // Back to the value it has (a no-op after another change of the pin).
                    fr.input_value(&name).unwrap().clone()
                } else {
                    random_value(rng, w, xz_pct)
                };
                fr.set_input(&name, &v).unwrap();
                fs.set_input(&name, &v).unwrap();
            }
            assert_eq!(fr.has_pending_inputs(), fs.has_pending_inputs(), "{actx}");
            if rng.chance(85) {
                assert_eq!(fr.step_event(), fs.step_event(), "{actx}: step result");
            }
        } else if roll < 37 {
            fr.reset();
            fs.reset();
        } else if roll < 42 {
            let n = 1 + rng.below(6) as u64;
            fr.run_events(n);
            fs.run_events(n);
        } else if roll < 46 {
            let t = fr.tick() + rng.below(30) as u64;
            fr.run_until(t);
            fs.run_until(t);
        } else {
            assert_eq!(fr.step_event(), fs.step_event(), "{actx}: step result");
        }
        if rng.chance(50) {
            check_get_read(fr, fs, flat, &actx, rng);
        }
        fs.check_invariants()
            .unwrap_or_else(|m| panic!("{actx}: invariant: {m}"));
        check_diag_details(fr, fs, &actx);
        check(fr, fs, flat, &actx, cov);
    }
}

fn extract_variant(rng: &mut Rng) -> ExtractOptions {
    let d = ExtractOptions::default();
    match rng.below(6) {
        0 => ExtractOptions { max_loop: 2, ..d },
        1 => ExtractOptions {
            max_loop: 12,
            max_bits: 14,
            ..d
        },
        2 => ExtractOptions {
            max_bits: *rng.pick(&[1, 3, 5]),
            ..d
        },
        3 => ExtractOptions { max_loop: 0, ..d },
        _ => d,
    }
}

/// Run one seed of the adversarial generator.
fn run_seed(seed: u64, actions: usize) -> Coverage {
    let mut rng = Rng::new(seed ^ 0xAD7E_5A21);
    let flat = adversarial_netlist(seed);
    let mut fast = FastOptions {
        extract: extract_variant(&mut rng),
        delta_limit: *rng.pick(&[64, 64, 64, 8, 3, 2, 1]),
        ..FastOptions::default()
    };
    common::randomize_charz(&mut rng, &mut fast);
    let mut compile = random_compile(&mut rng, &[1, 2, 3, 8, 64, 1 << 20]);
    if std::env::var("PANDA_NOLUT").is_ok() {
        compile.lut_max_bits = 0;
    }
    if std::env::var("PANDA_NOSPEC").is_ok() {
        compile.specialize = false;
    }
    if std::env::var("PANDA_NOFUSE").is_ok() {
        compile.fuse_inputs = 0;
    }
    let (mut fr, mut fs) = pair(&flat, &fast, &compile);
    if std::env::var("PANDA_DUMP").is_ok() {
        common::dump(&flat, &fs);
    }
    let ctx = format!(
        "adversarial seed {seed} ({} gates, {} kernels, {fast:?}, {compile:?})",
        flat.gates.len(),
        fs.kernels().kernels.len()
    );
    let mut cov = Coverage::default();
    program_coverage(&fs);
    lockstep(
        &mut fr, &mut fs, &flat, &mut rng, actions, 20, &ctx, &mut cov,
    );
    cov
}

/// Program-feature counters over all adversarial seeds (printed by the tests).
static FEATURES: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];
const FEATURE_NAMES: [&str; 11] = [
    "touts2",
    "touts8",
    "touts32",
    "programs",
    "tuple_skip",
    "tuple_fast",
    "tuple_check",
    "tuple_copaque",
    "vfast_used",
    "no_tuples_with_classes",
    "kernel_words>1",
];

fn program_coverage(fs: &FastSim) {
    let p = fs.program();
    let any = |v: &[u64]| v.iter().any(|&w| w != 0);
    let has_t = p.touts.iter().any(|&t| t != u32::MAX);
    let flags = [
        has_t && !p.touts2.is_empty(),
        has_t && p.touts2.is_empty() && !p.touts8.is_empty(),
        has_t && p.touts2.is_empty() && p.touts8.is_empty(),
        true,
        any(&p.tuple_skip),
        any(&p.tuple_fast),
        any(&p.tuple_check),
        any(&p.tuple_copaque),
        p.vfast_used,
        p.num_tuples == 0 && p.kernel_class.iter().any(|&c| c != u32::MAX),
        p.kernel_words > 1,
    ];
    for (f, c) in flags.iter().zip(&FEATURES) {
        if *f {
            c.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn print_features() {
    let v: Vec<String> = FEATURE_NAMES
        .iter()
        .zip(&FEATURES)
        .map(|(n, c)| format!("{n}={}", c.load(Ordering::Relaxed)))
        .collect();
    eprintln!("program features: {}", v.join(" "));
}

fn run_range(seeds: std::ops::Range<u64>, actions: usize) -> Coverage {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let next = AtomicU64::new(seeds.start);
    let total = std::sync::Mutex::new(Coverage::default());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let seed = next.fetch_add(1, Ordering::Relaxed);
                    if seed >= seeds.end {
                        break;
                    }
                    let c = run_seed(seed, actions);
                    total.lock().unwrap().add(&c);
                }
            });
        }
    });
    total.into_inner().unwrap()
}

#[test]
fn adversarial_designs() {
    let cov = run_range(0..400, 120);
    eprintln!("adversarial designs: {cov:?}");
    print_features();
    assert!(cov.luts > 0 && cov.fast_opaque > 0 && cov.fast_transparent > 0);
    for kind in [
        "Oscillating",
        "IterationLimit",
        "Contention",
        "DelaySensitive",
    ] {
        assert!(cov.diag_kinds.contains_key(kind), "no {kind}: {cov:?}");
    }
}

#[test]
#[ignore = "soak"]
fn adversarial_designs_soak() {
    let n: u64 = std::env::var("PANDA_SOAK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5000);
    let cov = run_range(400..400 + n, 200);
    eprintln!("adversarial soak: {cov:?}");
    print_features();
}

/// Debug helper: `PANDA_SEED=<n> cargo test ... one_adversarial_seed -- --ignored --nocapture`.
#[test]
#[ignore = "debug helper"]
fn one_adversarial_seed() {
    let seed: u64 = std::env::var("PANDA_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let cov = run_seed(seed, 200);
    eprintln!("adversarial seed {seed}: {cov:?}");
}

// ---------------------------------------------------------------------------------------------
// Hand-written designs
// ---------------------------------------------------------------------------------------------

/// Lockstep a hand-written design under several compile configurations with a scripted plus
/// random event sequence.
fn run_design(flat: &Arc<FlatNetlist>, name: &str, fast: &FastOptions) {
    let configs = [
        CompileOptions::default(),
        CompileOptions {
            lut_max_bits: 12,
            lut_min_gates: 1,
            partition_ops: 1,
            ..CompileOptions::default()
        },
        CompileOptions {
            lut_max_bits: 4,
            lut_min_gates: 2,
            fuse_inputs: 0,
            ..CompileOptions::default()
        },
        CompileOptions {
            specialize: false,
            lut_max_bits: 0,
            ..CompileOptions::default()
        },
    ];
    for (ci, c) in configs.iter().enumerate() {
        for seed in 0..12u64 {
            let (mut fr, mut fs) = pair(flat, fast, c);
            let mut rng = Rng::new(seed * 7919 + ci as u64);
            let mut cov = Coverage::default();
            let ctx = format!("{name} config {ci} seed {seed}");
            lockstep(&mut fr, &mut fs, flat, &mut rng, 80, 25, &ctx, &mut cov);
        }
    }
}

/// A LUT cone whose support includes an `X` constant, a `Z` constant, a floating tri-state bus
/// and a raw tri-state output; observed through internal nets.
#[test]
fn lut_cone_with_x_z_and_tristate_support() {
    let mut b = B::new();
    let a = b.input_bus("a", 4);
    let en = b.input("en");
    let kx = b.constant("kx", Logic::X);
    let kz = b.constant("kz", Logic::Z);
    let t0 = b.gate("t0", FlatOp::Tri, &[a[0], en]);
    let t1 = b.gate("t1", FlatOp::Tri, &[a[1], a[2]]);
    let bus = b.resolver("bus", &[t0, t1], Termination::None);
    // A ROM-like cone over a[0..4], en, bus, t0 plus the constants.
    let mut layer = vec![a[0], a[1], a[2], a[3], en, bus, t0];
    let mut k = 0;
    for _ in 0..4 {
        let mut next = Vec::new();
        for w in layer.windows(2) {
            k += 1;
            let op = OPS2[k % OPS2.len()];
            next.push(b.gate(&format!("m{k}"), op, &[w[0], w[1]]));
        }
        k += 1;
        next.push(b.gate(&format!("m{k}"), FlatOp::And, &[layer[0], kx]));
        k += 1;
        next.push(b.gate(&format!("m{k}"), FlatOp::Or, &[layer[1], kz]));
        layer = next;
    }
    let out = b.gate("out", FlatOp::Xor, &layer);
    b.output("out", out);
    let flat = Arc::new(b.f);
    run_design(&flat, "lut-xz", &FastOptions::default());
}

/// Two identical clocks plus a shifted one and a POR released on a common edge; registers on
/// each, cross-coupled through a gated clock and a ripple stage.
#[test]
fn coincident_clocks_and_por_on_edge() {
    let mut b = B::new();
    let d = b.input("d");
    let ge = b.input("ge");
    let c0 = b.clock("c0", 4, 2, 0);
    let c1 = b.clock("c1", 4, 2, 0);
    let c2 = b.clock("c2", 8, 4, 0);
    let c3 = b.clock("c3", 4, 1, 2);
    let por = b.por("por", 8);
    let porn = b.not("porn", por);
    let q0 = b.dff6("r0", d, c0, porn);
    let x = b.gate("x", FlatOp::Xor, &[q0, d]);
    let q1 = b.ms_dff("r1", x, c1);
    let g = b.gate("gclk", FlatOp::And, &[c2, ge]);
    let q2 = b.ms_dff("r2", q1, g);
    let both = b.gate("both", FlatOp::And, &[c0, c1]);
    let q3 = b.d_latch("l3", q2, both);
    let q4 = b.mux_latch("l4", q3, c3);
    let gq = b.gate("gq", FlatOp::And, &[c0, q4]);
    let q5 = b.ms_dff("r5", q0, gq);
    b.output("q5", q5);
    b.output("q3", q3);
    let flat = Arc::new(b.f);
    run_design(&flat, "coincident", &FastOptions::default());
    run_design(
        &flat,
        "coincident-d1",
        &FastOptions {
            delta_limit: 1,
            ..FastOptions::default()
        },
    );
}

/// No kernels at all (pure combinational logic with LUTs and buses); and no timed sources.
#[test]
fn design_without_kernels() {
    let mut b = B::new();
    let a = b.input_bus("a", 6);
    let mut cur = a.clone();
    for l in 0..5 {
        cur = cur
            .windows(2)
            .enumerate()
            .map(|(j, w)| b.gate(&format!("l{l}_{j}"), OPS2[(l + j) % 6], &[w[0], w[1]]))
            .collect();
        cur.push(a[l]);
    }
    let t = b.gate("t", FlatOp::Tri, &[cur[0], a[5]]);
    let bus = b.resolver("bus", &[t], Termination::PullUp);
    let o = b.gate("o", FlatOp::Xor, &[bus, cur[1]]);
    b.output("o", o);
    let flat = Arc::new(b.f);
    run_design(&flat, "no-kernels", &FastOptions::default());
}

/// Kernels with constant inputs (a latch with a constant enable / data, an SR latch with a
/// constant set) and an X-constant enable.
#[test]
fn kernels_with_constant_inputs() {
    let mut b = B::new();
    let d = b.input("d");
    let s = b.input("s");
    let k1 = b.constant("k1", Logic::One);
    let k0 = b.constant("k0", Logic::Zero);
    let kx = b.constant("kx", Logic::X);
    let clk = b.clock("clk", 6, 3, 1);
    let q0 = b.d_latch("l0", d, k1);
    let q1 = b.d_latch("l1", k0, s);
    let q2 = b.sr_nand("sr", k1, s);
    let q3 = b.d_latch("l3", d, kx);
    let q4 = b.ms_dff("f4", q0, clk);
    let x = b.gate("x", FlatOp::Xor, &[q1, q2, q3, q4]);
    b.output("x", x);
    let flat = Arc::new(b.f);
    run_design(&flat, "const-kernels", &FastOptions::default());
}

/// A net with very wide fan-out: one enable drives 150 latches (three kernel bitset words) and
/// one data net drives 200 gates.
#[test]
fn very_wide_fanout() {
    let mut b = B::new();
    let d = b.input("d");
    let e = b.input("e");
    let clk = b.clock("clk", 4, 2, 0);
    let en = b.gate("en", FlatOp::And, &[clk, e]);
    let mut qs = Vec::new();
    for i in 0..150 {
        let x = b.gate(
            &format!("x{i}"),
            if i % 2 == 0 { FlatOp::Buf } else { FlatOp::Not },
            &[d],
        );
        let en_i = if i % 5 == 0 { clk } else { en };
        qs.push(b.d_latch(&format!("l{i}"), x, en_i));
    }
    for i in 0..50 {
        let y = b.gate(&format!("y{i}"), FlatOp::Xor, &[d, qs[i], qs[149 - i]]);
        qs.push(y);
    }
    let r = b.gate("r", FlatOp::Xor, &qs);
    b.output("r", r);
    let flat = Arc::new(b.f);
    run_design(&flat, "wide-fanout", &FastOptions::default());
}

/// Gated ring oscillators and a long-loop (sim) latch with delta-limit exhaustion at several
/// limits; the oscillator's output feeds a register and a LUT cone.
#[test]
fn oscillators_sim_kernels_and_delta_limit() {
    let mut b = B::new();
    let en = b.input("en");
    let s = b.input("s");
    let h = b.input("h");
    let clk = b.clock("clk", 4, 2, 0);
    let r = b.ring("ro", en);
    let l = b.long_latch("ll", s, h, 8);
    // Two latches in a loop through an XOR: a delta-iterating pair when both are transparent.
    let q1n = b.net("q1n");
    let q0 = b.d_latch("p0", q1n, en);
    let q1 = b.d_latch("p1", q0, en);
    b.gate_to(FlatOp::Not, &[q1], q1n);
    let x = b.gate("x", FlatOp::Xor, &[r, l, q1]);
    let f = b.ms_dff("f", x, clk);
    b.output("f", f);
    let flat = Arc::new(b.f);
    for dl in [1, 2, 3, 5, 64] {
        run_design(
            &flat,
            &format!("osc-dl{dl}"),
            &FastOptions {
                delta_limit: dl,
                ..FastOptions::default()
            },
        );
    }
}

/// Inputs changed many times between events (including back to the old value): the pending
/// flag and the event's effect must match FastRef's.
#[test]
fn many_input_changes_between_events() {
    let mut b = B::new();
    let d = b.input_bus("d", 3);
    let en = b.input("en");
    let q: Vec<NetIdx> = (0..3)
        .map(|i| b.d_latch(&format!("l{i}"), d[i], en))
        .collect();
    let x = b.gate("x", FlatOp::Xor, &q);
    b.output("x", x);
    let flat = Arc::new(b.f);
    let (mut fr, mut fs) = pair(&flat, &FastOptions::default(), &CompileOptions::default());
    let mut cov = Coverage::default();
    for round in 0..20u64 {
        for k in 0..10u64 {
            let v = Value::from_u64(3, (round * 5 + k) % 8);
            fr.set_input("d", &v).unwrap();
            fs.set_input("d", &v).unwrap();
            let e = Value::from_u64(1, (round + k) % 2);
            fr.set_input("en", &e).unwrap();
            fs.set_input("en", &e).unwrap();
        }
        // Back to the values of before the burst (still pending in both engines).
        assert_eq!(fr.has_pending_inputs(), fs.has_pending_inputs());
        assert_eq!(fr.step_event(), fs.step_event());
        check(&mut fr, &mut fs, &flat, &format!("burst {round}"), &mut cov);
        assert_eq!(fr.step_event(), fs.step_event());
        check(&mut fr, &mut fs, &flat, &format!("idle {round}"), &mut cov);
    }
}

/// Degenerate gate shapes: single-input AND/OR/XOR families, duplicated inputs, very wide gates,
/// gates reading `Z` constants and raw tri-state outputs, inside and outside kernels and LUTs.
#[test]
fn degenerate_gate_shapes() {
    let mut b = B::new();
    let a = b.input_bus("a", 5);
    let clk = b.clock("clk", 4, 2, 1);
    let kz = b.constant("kz", Logic::Z);
    let mut sigs = a.clone();
    for (i, op) in OPS2.iter().enumerate() {
        sigs.push(b.gate(&format!("one{i}"), *op, &[a[i % 5]]));
        sigs.push(b.gate(
            &format!("dup{i}"),
            *op,
            &[a[i % 5], a[i % 5], a[(i + 1) % 5]],
        ));
        sigs.push(b.gate(&format!("z{i}"), *op, &[a[i % 5], kz]));
    }
    let t = b.gate("t", FlatOp::Tri, &[a[0], a[1]]);
    sigs.push(b.gate("tn", FlatOp::Not, &[t]));
    sigs.push(b.gate("tb", FlatOp::Buf, &[t]));
    let wide_ins: Vec<NetIdx> = (0..100).map(|i| sigs[i % sigs.len()]).collect();
    let w = b.gate("wide", FlatOp::Nand, &wide_ins);
    // Single-input gates inside a kernel loop and a latch whose enable is a 1-input AND of clk.
    let q = b.net("q");
    let s = b.gate("s1", FlatOp::Or, &[q]);
    let e = b.gate("e1", FlatOp::And, &[clk]);
    let d = b.gate("d", FlatOp::Xor, &[w, a[2]]);
    let l = b.d_latch("l", d, e);
    b.gate_to(FlatOp::Nor, &[s, a[3], a[3]], q);
    let o = b.gate("o", FlatOp::Xnor, &[l, q, w]);
    b.output("o", o);
    let flat = Arc::new(b.f);
    run_design(&flat, "degenerate-gates", &FastOptions::default());
}

/// Out-of-range compiler tunables (documented as performance-only) must still give a working,
/// exact engine: zero partition size and LUT weight, LUT support above the 16-bit cap, fusion
/// widths outside 2–6, zero merge slack.
#[test]
fn extreme_compile_options() {
    for seed in [3u64, 17, 29, 41] {
        let flat = adversarial_netlist(seed);
        for c in [
            CompileOptions {
                partition_ops: 0,
                lut_weight: 0,
                ..CompileOptions::default()
            },
            CompileOptions {
                lut_max_bits: 40,
                lut_min_gates: 0,
                lut_max_entries: usize::MAX,
                ..CompileOptions::default()
            },
            CompileOptions {
                fuse_inputs: 1,
                lut_merge_slack: 0,
                ..CompileOptions::default()
            },
            CompileOptions {
                fuse_inputs: 100,
                partition_ops: usize::MAX,
                lut_weight: usize::MAX,
                lut_merge_slack: usize::MAX,
                ..CompileOptions::default()
            },
        ] {
            let (mut fr, mut fs) = pair(&flat, &FastOptions::default(), &c);
            let mut rng = Rng::new(seed);
            let mut cov = Coverage::default();
            let ctx = format!("extreme options seed {seed} {c:?}");
            lockstep(&mut fr, &mut fs, &flat, &mut rng, 40, 20, &ctx, &mut cov);
        }
    }
}

/// Regression: a net read by kernels or partitions in more than 255 bitset words (> 16320 ids:
/// a gate-built register file of 8500 flip-flops whose data input is shared, plus 17000 gates
/// reading it in one-op partitions) made compilation fail ("too many readers"). It must
/// compile (escaped record header) and match FastRef.
#[test]
fn huge_fanout_net_compiles() {
    let mut b = B::new();
    let d = b.input("d");
    let we = b.input("we");
    let clk = b.clock("clk", 4, 2, 0);
    let mut qs = Vec::new();
    let mut acc = Vec::new();
    for i in 0..8500 {
        // Each flip-flop's data: d XOR we (absorbed into the master latch: `d` and `we` are
        // read by 8500 kernels spread over 17000 kernel ids).
        let x = b.gate(&format!("x{i}"), FlatOp::Xor, &[d, we]);
        let q = b.ms_dff(&format!("f{i}"), x, clk);
        qs.push(q);
        // Two combinational readers of `d` per bit: with one-op partitions, `d` marks partitions
        // in more than 255 bitset words.
        acc.push(b.gate(&format!("y{i}"), FlatOp::Xor, &[d, q]));
        acc.push(b.gate(&format!("z{i}"), FlatOp::And, &[d, q]));
    }
    while acc.len() > 1 {
        acc = acc
            .chunks(8)
            .enumerate()
            .map(|(j, c)| {
                if c.len() == 1 {
                    c[0]
                } else {
                    let n = format!("r{}_{j}", acc.len());
                    b.gate(&n, FlatOp::Xor, c)
                }
            })
            .collect();
    }
    b.output("r", acc[0]);
    let flat = Arc::new(b.f);
    let c = CompileOptions {
        partition_ops: 1,
        ..CompileOptions::default()
    };
    let (mut fr, mut fs) = pair(&flat, &FastOptions::default(), &c);
    assert!(fs.program().partitions.len() > 255 * 64);
    let mut rng = Rng::new(5);
    let mut cov = Coverage::default();
    lockstep(
        &mut fr,
        &mut fs,
        &flat,
        &mut rng,
        30,
        10,
        "huge fan-out",
        &mut cov,
    );
}

/// `load_state` with perturbed (unsettled, partly `X`/`Z`) states: an optimized engine and an
/// unoptimized one (no specialization, LUTs, fusion; one-op partitions) built from the same
/// kernels stay identical afterwards (values of all nets, tick, schedule, fallback count,
/// diagnostic kinds), including loads in the middle of a phase and at arbitrary ticks.
#[test]
fn load_perturbed_states() {
    let plain = CompileOptions {
        specialize: false,
        lut_max_bits: 0,
        fuse_inputs: 0,
        partition_ops: 1,
        ..CompileOptions::default()
    };
    let n: u64 = std::env::var("PANDA_LOAD_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(150);
    for seed in 0..n {
        let flat = adversarial_netlist(seed);
        let mut rng = Rng::new(seed ^ 0x10AD);
        let fast = FastOptions {
            delta_limit: *rng.pick(&[64, 64, 3]),
            ..FastOptions::default()
        };
        let compile = random_compile(&mut rng, &[1, 3, 8, 64]);
        let (_, mut a) = pair(&flat, &fast, &compile);
        let mut b = FastSim::from_parts(
            flat.clone(),
            panda_fast::FastSimOptions {
                fast: fast.clone(),
                compile: plain.clone(),
            },
            a.kernels().clone(),
            a.tables().to_vec(),
        )
        .unwrap();
        let pins: Vec<(String, u32)> = flat
            .top_inputs
            .iter()
            .map(|p| (p.name.clone(), p.nets.len() as u32))
            .collect();
        for round in 0..6 {
            let ctx = format!("load seed {seed} round {round} ({compile:?})");
            // Some events first.
            for _ in 0..rng.below(20) {
                if !pins.is_empty() && rng.chance(30) {
                    let (name, w) = rng.pick(&pins).clone();
                    let v = random_value(&mut rng, w, 20);
                    a.set_input(&name, &v).unwrap();
                    b.set_input(&name, &v).unwrap();
                }
                assert_eq!(a.step_event(), b.step_event(), "{ctx}");
            }
            // A perturbed state.
            let mut vals = a.values().to_vec();
            for v in &mut vals {
                if rng.chance(15) {
                    *v = *rng.pick(&[Logic::Zero, Logic::One, Logic::X, Logic::Z]);
                }
            }
            let inputs: Vec<Value> = pins
                .iter()
                .map(|(_, w)| random_value(&mut rng, *w, 20))
                .collect();
            let tick = if rng.chance(50) {
                a.tick()
            } else {
                a.tick() + rng.below(50) as u64
            };
            if std::env::var("PANDA_TRACE").is_ok() {
                eprintln!("{ctx}: load at {tick}");
            }
            let fa = a.load_state(tick, &vals, &inputs);
            let fb = b.load_state(tick, &vals, &inputs);
            for e in [&a, &b] {
                e.check_invariants()
                    .unwrap_or_else(|m| panic!("{ctx}: after load: {m}"));
            }
            assert_eq!(fa, fb, "{ctx}: fallen kernels");
            for e in 0..40 {
                let ectx = format!("{ctx} event {e}");
                assert_eq!(a.values(), b.values(), "{ectx}: values");
                assert_eq!(a.tick(), b.tick(), "{ectx}");
                assert_eq!(a.next_event_tick(), b.next_event_tick(), "{ectx}");
                let mut ka: Vec<String> = a
                    .take_diagnostics()
                    .iter()
                    .map(|d| format!("{:?}", d.kind))
                    .collect();
                let mut kb: Vec<String> = b
                    .take_diagnostics()
                    .iter()
                    .map(|d| format!("{:?}", d.kind))
                    .collect();
                ka.sort();
                kb.sort();
                assert_eq!(ka, kb, "{ectx}: diagnostics");
                if !pins.is_empty() && rng.chance(25) {
                    let (name, w) = rng.pick(&pins).clone();
                    let v = random_value(&mut rng, w, 20);
                    a.set_input(&name, &v).unwrap();
                    b.set_input(&name, &v).unwrap();
                }
                assert_eq!(a.step_event(), b.step_event(), "{ectx}: step");
                for e in [&a, &b] {
                    e.check_invariants()
                        .unwrap_or_else(|m| panic!("{ectx}: after step: {m}"));
                }
            }
        }
    }
}

/// Regression: `load_state` used to keep a loaded value on a net without a driver. A pull-up net
/// is a constant `1` to the compiled program (folded into LUTs and kernel phase classes), so a
/// state loading `0` into it desynchronized the specialized paths (a lazy kernel without a
/// reconstructible state: index out of range in `values()`) and LUT outputs from the gates.
/// Undriven nets are constants and now take their value at load, like constant sources.
#[test]
fn load_state_restores_undriven_nets() {
    let mut b = B::new();
    let d = b.input("d");
    let clk = b.clock("clk", 4, 2, 0);
    let pu = b.net("pu");
    b.f.undriven_termination.insert(pu, Termination::PullUp);
    let pd = b.net("pd");
    b.f.undriven_termination.insert(pd, Termination::PullDown);
    let fl = b.net("float");
    // A latch whose enable is clk AND pu (pure: a function of the clock tuple) ...
    let en = b.gate("en", FlatOp::And, &[clk, pu]);
    let q = b.d_latch("l", d, en);
    // ... a latch reading the pull-up net directly as a kernel input (phase class with a
    // constant pure input) ...
    let q2 = b.mux_latch("m", q, clk);
    let q3 = b.d_latch("l3", q2, pu);
    // ... and an edge-triggered flip-flop cleared by the (active-low) pull-up net.
    let q4 = b.dff6("f", q3, clk, pu);
    b.output("q4", q4);
    // ... and a LUT-sized cone over d, q, the pulled nets and the floating net.
    let mut layer = vec![d, q, q3, pu, pd, fl];
    for l in 0..4 {
        let mut next: Vec<NetIdx> = layer
            .windows(2)
            .enumerate()
            .map(|(j, w)| b.gate(&format!("c{l}_{j}"), OPS2[(l + j) % 6], &[w[0], w[1]]))
            .collect();
        next.push(b.gate(&format!("cx{l}"), FlatOp::Xor, &[pu, pd, layer[0]]));
        layer = next;
    }
    let o = b.gate("o", FlatOp::Xor, &layer);
    b.output("o", o);
    let flat = Arc::new(b.f);
    let opt = CompileOptions {
        lut_max_bits: 12,
        lut_min_gates: 2,
        ..CompileOptions::default()
    };
    let plain = CompileOptions {
        specialize: false,
        lut_max_bits: 0,
        fuse_inputs: 0,
        ..CompileOptions::default()
    };
    let (_, mut a) = pair(&flat, &FastOptions::default(), &opt);
    let (_, mut p) = pair(&flat, &FastOptions::default(), &plain);
    assert!(!a.program().luts.is_empty(), "the cone is a LUT");
    for v in [1u64, 0, 1, 1, 0] {
        for e in [&mut a, &mut p] {
            e.set_input("d", &Value::from_u64(1, v)).unwrap();
            e.run_events(3);
        }
    }
    let mut vals = a.values().to_vec();
    vals[pu as usize] = Logic::Zero;
    vals[pd as usize] = Logic::One;
    vals[fl as usize] = Logic::One;
    let inputs = vec![Value::from_u64(1, 1)];
    let tick = a.tick() + 1;
    assert_eq!(
        a.load_state(tick, &vals, &inputs),
        p.load_state(tick, &vals, &inputs)
    );
    for e in 0..40 {
        assert_eq!(a.get(pu), Logic::One, "event {e}");
        assert_eq!(a.get(pd), Logic::Zero, "event {e}");
        assert_eq!(a.get(fl), Logic::Z, "event {e}");
        assert_eq!(a.values(), p.values(), "event {e}");
        if e % 7 == 3 {
            let v = Value::from_u64(1, e % 2);
            a.set_input("d", &v).unwrap();
            p.set_input("d", &v).unwrap();
        }
        assert_eq!(a.step_event(), p.step_event());
    }
}
