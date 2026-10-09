//! Random synchronous design generator.
//!
//! A generated design has
//!
//! * one or two `Clock` sources (`clk0`, `clk1`) whose edges all fall on multiples of a *time
//!   unit* `s` (the `scale`), with several shapes: related periods (1:2, 1:3, 2:3), equal periods
//!   with a phase offset, non-overlapping two-phase clocks, various duty cycles;
//! * a `PowerOnReset` (`rst`) released half a unit into a phase after at least two cycles of every
//!   clock, used as a synchronous reset everywhere;
//! * random top-level inputs `in0..` (mostly 1 bit, some buses);
//! * state elements from `panda-circuits`: `ms_dff` (reset through the data path), `dff_rst`,
//!   `dff_en`, `register`, `register_gated`, `toggle_ff`, `sync_counter`, `dff_arst`, single
//!   transparent latches and two-phase `d_latch` pipelines; clocked by a clock, its inverse, a
//!   latch-gated (ICG) clock or a raw `AND` gated clock;
//! * random combinational logic between them over a pool of signals (register outputs = feedback,
//!   inputs, reset, constants): random gates, `mux2`, ripple adders, incrementers, decoders,
//!   comparators and tri-state buses with pull-up/pull-down (or keeper / no) termination;
//! * at lower weight asynchronous structures: SR latches driven from inputs or logic, ripple
//!   counters, derived clocks (a register clocked by a logic signal), clocks used as data.
//!
//! The structure depends only on the seed; [`generate`] takes the time unit separately so the
//! comparison driver can rebuild the same structure with longer clock periods
//! (see [`crate::driver`]).

use panda_circuits as pc;
use panda_netlist::{DefBuilder, DefRef, Design, DesignBuilder, GateOp, Sig, Termination};

use crate::rng::Rng;

/// Weights and sizes of the generator.
#[derive(Clone, Debug)]
pub struct GenConfig {
    /// State elements per design (inclusive range).
    pub regs: (usize, usize),
    /// Combinational operations per design (inclusive range).
    pub comb: (usize, usize),
    /// Top-level input pins (inclusive range).
    pub inputs: (usize, usize),
    /// Percent of designs with two clocks.
    pub two_clocks_pct: u32,
    /// Percent of designs that contain asynchronous structures (SR latches, ripple counters,
    /// derived clocks, raw gated clocks, clocks as data, ring oscillators). 0 disables them.
    pub async_pct: u32,
    /// Percent of designs fully inlined (no hierarchy hints; SPEC §12 variant b).
    pub inline_pct: u32,
    /// Percent of designs without any clock: only the reset release and input events; state
    /// elements are enabled/clocked by logic of the inputs.
    pub clockless_pct: u32,
}

impl Default for GenConfig {
    fn default() -> Self {
        GenConfig {
            regs: (2, 8),
            comb: (4, 30),
            inputs: (0, 4),
            two_clocks_pct: 35,
            async_pct: 25,
            inline_pct: 25,
            clockless_pct: 5,
        }
    }
}

/// Shape of one clock in time units: period `period_u · s`, high for `high_u · s`, rising at
/// `phase_u · s`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClockShape {
    /// Period in units.
    pub period_u: u32,
    /// High time in units (`0 < high_u < period_u`).
    pub high_u: u32,
    /// Rising-edge offset in units (`< period_u`).
    pub phase_u: u32,
}

/// One clock of a generated design.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClockInfo {
    /// Net name (`clk0`, `clk1`).
    pub name: String,
    /// Shape in units.
    pub shape: ClockShape,
    /// Period in ticks.
    pub period: u32,
    /// High time in ticks.
    pub high: u32,
    /// Phase in ticks.
    pub phase: u32,
}

/// A top-level input pin.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputPin {
    /// Pin name.
    pub name: String,
    /// Width in bits.
    pub width: u32,
}

/// A generated design.
#[derive(Clone, Debug)]
pub struct Generated {
    /// Seed it was generated from.
    pub seed: u64,
    /// Time unit (ticks) it was built with.
    pub scale: u32,
    /// The design.
    pub design: Design,
    /// Its clocks.
    pub clocks: Vec<ClockInfo>,
    /// `PowerOnReset` release tick.
    pub reset_ticks: u32,
    /// Its top-level inputs.
    pub inputs: Vec<InputPin>,
    /// Structures used (for statistics), e.g. `"register_gated"`, `"tri_bus"`.
    pub features: Vec<&'static str>,
}

/// Number of time units of the reset (release at `reset_units(..) · s + s/2`).
fn reset_units(shapes: &[ClockShape], extra: u32) -> u32 {
    let max_p = shapes.iter().map(|c| c.period_u).max().unwrap_or(2);
    2 * max_p + extra
}

/// Draw the clock shapes (first draws of the structure stream).
fn clock_shapes(rng: &mut Rng, cfg: &GenConfig) -> Vec<ClockShape> {
    let half = |p: u32| ClockShape {
        period_u: p,
        high_u: p / 2,
        phase_u: 0,
    };
    if !rng.chance(cfg.two_clocks_pct) {
        let p = *rng.pick(&[2u32, 2, 4, 4, 6]);
        let mut c = half(p);
        if p > 2 && rng.chance(25) {
            c.high_u = rng.range(1, p as usize - 1) as u32;
        }
        c.phase_u = rng.below(p as usize) as u32;
        return vec![c];
    }
    match rng.below(4) {
        0 => {
            // Related periods: k·P.
            let p = *rng.pick(&[2u32, 4]);
            let k = rng.range(2, 3) as u32;
            let a = ClockShape {
                phase_u: rng.below(p as usize) as u32,
                ..half(p)
            };
            let b = ClockShape {
                phase_u: rng.below((p * k) as usize) as u32,
                ..half(p * k)
            };
            vec![a, b]
        }
        1 => {
            // 2:3 periods.
            let a = ClockShape {
                phase_u: rng.below(4) as u32,
                ..half(4)
            };
            let b = ClockShape {
                phase_u: rng.below(6) as u32,
                ..half(6)
            };
            vec![a, b]
        }
        2 => {
            // Equal periods, phase offset.
            let p = *rng.pick(&[4u32, 6]);
            let a = ClockShape {
                phase_u: 0,
                ..half(p)
            };
            let b = ClockShape {
                phase_u: rng.range(1, p as usize - 1) as u32,
                ..half(p)
            };
            vec![a, b]
        }
        _ => {
            // Non-overlapping two-phase clocks.
            let p = *rng.pick(&[4u32, 6]);
            let h = if p == 4 { 1 } else { rng.range(1, 2) as u32 };
            let a = ClockShape {
                period_u: p,
                high_u: h,
                phase_u: 0,
            };
            let b = ClockShape {
                period_u: p,
                high_u: h,
                phase_u: p / 2,
            };
            vec![a, b]
        }
    }
}

/// Library defs used by the generator.
struct Lib {
    ms_dff: DefRef,
    dff_rst: DefRef,
    dff_en: DefRef,
    dff_arst: DefRef,
    toggle_ff: DefRef,
    d_latch: DefRef,
    nand_sr: DefRef,
    nor_sr: DefRef,
}

struct Gen {
    rng: Rng,
    db: DesignBuilder,
    b: DefBuilder,
    lib: Lib,
    /// 1-bit data signals.
    pool: Vec<Sig>,
    /// Clock nets (`clk0`, `clk1`).
    clocks: Vec<Sig>,
    /// Second-phase partner of each clock for latch pipelines.
    nclocks: Vec<Sig>,
    rst: Sig,
    nrst: Sig,
    features: Vec<&'static str>,
    n_out: usize,
    /// This design may contain asynchronous structures.
    asynch: bool,
}

const OPS: [GateOp; 6] = [
    GateOp::And,
    GateOp::Or,
    GateOp::Nand,
    GateOp::Nor,
    GateOp::Xor,
    GateOp::Xnor,
];

impl Gen {
    fn feature(&mut self, f: &'static str) {
        if !self.features.contains(&f) {
            self.features.push(f);
        }
    }

    /// Weight of an asynchronous structure: `w` in async designs, 0 otherwise.
    fn aw(&self, w: u32) -> u32 {
        if self.asynch { w } else { 0 }
    }

    /// A random pool signal, biased towards recent ones (deeper logic).
    fn sig(&mut self) -> Sig {
        let n = self.pool.len();
        if n > 4 && self.rng.chance(50) {
            let lo = n - n.min(6);
            self.pool[self.rng.range(lo, n - 1)]
        } else {
            self.pool[self.rng.below(n)]
        }
    }

    /// A random pool signal or (rarely) a clock used as data.
    fn data_sig(&mut self) -> Sig {
        if !self.clocks.is_empty() && self.rng.chance(self.aw(3)) {
            self.feature("clock_as_data");
            let c = self.rng.below(self.clocks.len());
            return self.clocks[c];
        }
        self.sig()
    }

    fn bus(&mut self, w: u32) -> Sig {
        let bits: Vec<Sig> = (0..w).map(|_| self.data_sig()).collect();
        cat(&mut self.b, &bits)
    }

    fn push_bits(&mut self, s: Sig) {
        if s.width == 1 {
            self.pool.push(s);
        } else {
            let bits = self.b.bits(s);
            self.pool.extend(bits);
        }
    }

    fn maybe_output(&mut self, s: Sig) {
        if self.rng.chance(15) {
            let name = format!("O{}", self.n_out);
            self.n_out += 1;
            self.b.output(&name, s);
        }
    }

    /// One random combinational operation.
    fn comb_op(&mut self) {
        let weights = [30, 10, 8, 4, 3, 4, 3, 5, self.aw(3)];
        match self.rng.weighted(&weights) {
            0 => {
                let op = *self.rng.pick(&OPS);
                let n = self.rng.range(2, 3);
                let ins: Vec<Sig> = (0..n).map(|_| self.data_sig()).collect();
                let g = self.b.gate(op, &ins);
                self.maybe_output(g);
                self.pool.push(g);
            }
            1 => {
                let a = self.data_sig();
                let g = if self.rng.chance(70) {
                    self.b.not(a)
                } else {
                    self.b.buf(a)
                };
                self.pool.push(g);
            }
            2 => {
                self.feature("mux2");
                let w = self.rng.range(1, 2) as u32;
                let m = pc::mux2(&mut self.db, w);
                let (x, y, sel) = (self.bus(w), self.bus(w), self.data_sig());
                let out = self.b.inst(&m, &[x, y, sel])[0];
                self.maybe_output(out);
                self.push_bits(out);
            }
            3 => {
                self.feature("adder");
                let w = self.rng.range(1, 4) as u32;
                let add = pc::ripple_adder(&mut self.db, w);
                let (x, y, cin) = (self.bus(w), self.bus(w), self.data_sig());
                let outs = self.b.inst(&add, &[x, y, cin]);
                self.maybe_output(outs[0]);
                self.push_bits(outs[0]);
                self.pool.push(outs[1]);
            }
            4 => {
                self.feature("incrementer");
                let w = self.rng.range(2, 4) as u32;
                let inc = pc::incrementer(&mut self.db, w);
                let (x, cin) = (self.bus(w), self.data_sig());
                let outs = self.b.inst(&inc, &[x, cin]);
                self.push_bits(outs[0]);
                self.pool.push(outs[1]);
            }
            5 => {
                self.feature("decoder");
                let n = self.rng.range(1, 2) as u32;
                let dec = pc::decoder(&mut self.db, n);
                let (a, en) = (self.bus(n), self.data_sig());
                let y = self.b.inst(&dec, &[a, en])[0];
                self.push_bits(y);
            }
            6 => {
                self.feature("equality");
                let n = self.rng.range(1, 3) as u32;
                let eq = pc::equality(&mut self.db, n);
                let (x, y) = (self.bus(n), self.bus(n));
                let o = self.b.inst(&eq, &[x, y])[0];
                self.pool.push(o);
            }
            7 => self.tri_bus(),
            _ => {
                // SR latch driven by logic (often hazardous).
                self.feature("sr_latch_logic");
                let (s, r) = (self.data_sig(), self.data_sig());
                let d = if self.rng.chance(50) {
                    self.lib.nand_sr.clone()
                } else {
                    self.lib.nor_sr.clone()
                };
                let outs = self.b.inst(&d, &[s, r]);
                self.pool.push(outs[0]);
                if self.rng.chance(30) {
                    self.pool.push(outs[1]);
                }
            }
        }
    }

    /// A tri-state bus: 2–3 `TRI` drivers on one terminated net.
    fn tri_bus(&mut self) {
        self.feature("tri_bus");
        let w = self.rng.range(1, 2) as u32;
        let k = self.rng.range(2, 3);
        let bus = self.b.wire(w);
        let term = match self.rng.weighted(&[50, 25, self.aw(8), self.aw(8)]) {
            0 => Termination::PullUp,
            1 => Termination::PullDown,
            2 => {
                self.feature("keeper");
                Termination::Keeper
            }
            _ => Termination::None,
        };
        if term != Termination::None {
            self.b.terminate(bus, term);
        }
        let enables: Vec<Sig> = if self.rng.chance(60) {
            // One-hot enables from a decoder (no contention when the enable is binary).
            let dec = pc::decoder(&mut self.db, 2);
            let (a, en) = (self.bus(2), self.data_sig());
            let y = self.b.inst(&dec, &[a, en])[0];
            let ys = self.b.bits(y);
            ys[..k].to_vec()
        } else {
            self.feature("tri_contention");
            (0..k).map(|_| self.data_sig()).collect()
        };
        for en in enables {
            let d = self.bus(w);
            self.b.tri_to(d, en, bus);
        }
        self.maybe_output(bus);
        self.push_bits(bus);
    }

    /// A clock for one state element: (clock signal, base clock index).
    fn reg_clock(&mut self) -> Sig {
        if self.clocks.is_empty() {
            // Clockless design: enabled/clocked by logic.
            return self.sig();
        }
        let ci = self.rng.below(self.clocks.len());
        let c = self.clocks[ci];
        match self.rng.weighted(&[70, 12, 8, self.aw(5), self.aw(4)]) {
            0 => c,
            1 => {
                self.feature("inverted_clock");
                self.b.not(c)
            }
            2 => {
                // Latch-based clock gating cell: enable latched while the clock is low.
                self.feature("icg");
                let en0 = self.sig();
                let en = self.b.or(&[en0, self.rst]);
                let nc = self.b.not(c);
                let l = self.b.inst(&self.lib.d_latch.clone(), &[en, nc])[0];
                self.b.and(&[c, l])
            }
            3 => {
                self.feature("raw_gated_clock");
                let en0 = self.sig();
                let en = self.b.or(&[en0, self.rst]);
                self.b.and(&[c, en])
            }
            _ => {
                self.feature("derived_clock");
                self.sig()
            }
        }
    }

    /// Data with synchronous reset: `d AND NOT rst`.
    fn rdata(&mut self) -> Sig {
        let d = self.sig();
        self.b.and(&[d, self.nrst])
    }

    /// Instantiate one state element whose outputs drive `q`.
    fn state_element(&mut self, kind: usize, q: Sig) {
        let w = q.width;
        let out = match kind {
            // ms_dff with reset in the data path.
            0 => {
                self.feature("ms_dff");
                let bits: Vec<Sig> = (0..w)
                    .map(|_| {
                        let d = self.rdata();
                        let ck = self.reg_clock();
                        self.b.inst(&self.lib.ms_dff.clone(), &[d, ck])[0]
                    })
                    .collect();
                cat(&mut self.b, &bits)
            }
            1 => {
                self.feature("dff_rst");
                let bits: Vec<Sig> = (0..w)
                    .map(|_| {
                        let d = self.sig();
                        let ck = self.reg_clock();
                        self.b.inst(&self.lib.dff_rst.clone(), &[d, self.rst, ck])[0]
                    })
                    .collect();
                cat(&mut self.b, &bits)
            }
            2 => {
                self.feature("dff_en");
                let bits: Vec<Sig> = (0..w)
                    .map(|_| {
                        let d = self.rdata();
                        let en0 = self.sig();
                        let en = self.b.or(&[en0, self.rst]);
                        let ck = self.reg_clock();
                        self.b.inst(&self.lib.dff_en.clone(), &[d, en, ck])[0]
                    })
                    .collect();
                cat(&mut self.b, &bits)
            }
            3 | 4 => {
                let gated = kind == 4;
                self.feature(if gated { "register_gated" } else { "register" });
                let def = if gated {
                    pc::register_gated(&mut self.db, w)
                } else {
                    pc::register(&mut self.db, w)
                };
                let d = self.bus(w);
                let load = self.sig();
                let ck = self.reg_clock();
                self.b.inst(&def, &[d, load, self.rst, ck])[0]
            }
            5 => {
                // Two-phase latch pipeline: stages alternate between the clock's two phases.
                self.feature("latch_pipeline");
                let (p1, p2) = if self.clocks.is_empty() {
                    let en = self.sig();
                    let nen = self.b.not(en);
                    (en, nen)
                } else {
                    let ci = self.rng.below(self.clocks.len());
                    (self.clocks[ci], self.nclocks[ci])
                };
                let stages = self.rng.range(2, 4);
                let bits: Vec<Sig> = (0..w)
                    .map(|_| {
                        let mut v = self.rdata();
                        for st in 0..stages {
                            if st > 0 && self.rng.chance(40) {
                                let op = *self.rng.pick(&OPS);
                                let o = self.sig();
                                let g = self.b.gate(op, &[v, o]);
                                v = self.b.and(&[g, self.nrst]);
                            }
                            let en = if (stages - st) % 2 == 1 { p1 } else { p2 };
                            v = self.b.inst(&self.lib.d_latch.clone(), &[v, en])[0];
                            if st + 1 < stages {
                                self.pool.push(v);
                            }
                        }
                        v
                    })
                    .collect();
                cat(&mut self.b, &bits)
            }
            6 => {
                self.feature("latch");
                let bits: Vec<Sig> = (0..w)
                    .map(|_| {
                        let d = self.rdata();
                        let ck = self.reg_clock();
                        self.b.inst(&self.lib.d_latch.clone(), &[d, ck])[0]
                    })
                    .collect();
                cat(&mut self.b, &bits)
            }
            7 => {
                self.feature("sync_counter");
                let def = pc::sync_counter(&mut self.db, w.max(2));
                let ck = self.reg_clock();
                let en = self.sig();
                let o = self.b.inst(&def, &[ck, self.rst, en])[0];
                if w == 1 { self.b.bit(o, 0) } else { o }
            }
            8 => {
                self.feature("toggle_ff");
                let bits: Vec<Sig> = (0..w)
                    .map(|_| {
                        let t = self.sig();
                        let ck = self.reg_clock();
                        self.b.inst(&self.lib.toggle_ff.clone(), &[t, self.rst, ck])[0]
                    })
                    .collect();
                cat(&mut self.b, &bits)
            }
            9 => {
                self.feature("dff_arst");
                let bits: Vec<Sig> = (0..w)
                    .map(|_| {
                        let d = self.sig();
                        let ck = self.reg_clock();
                        self.b.inst(&self.lib.dff_arst.clone(), &[d, self.rst, ck])[0]
                    })
                    .collect();
                cat(&mut self.b, &bits)
            }
            10 => {
                self.feature("ripple_counter");
                let def = pc::ripple_counter(&mut self.db, w.max(2));
                let ck = if self.clocks.is_empty() {
                    self.sig()
                } else {
                    let ci = self.rng.below(self.clocks.len());
                    self.clocks[ci]
                };
                let o = self.b.inst(&def, &[ck, self.rst])[0];
                if w == 1 { self.b.bit(o, 0) } else { o }
            }
            12 | 13 => {
                // Custom gate-level latches: a single transparent latch (12) or a master-slave
                // pair (13) of a random style.
                let style = self.rng.below(4);
                self.feature(match style {
                    0 => "mux_latch",
                    1 => "mux_latch_consensus",
                    2 => "long_loop_latch",
                    _ => "keeper_latch",
                });
                let bits: Vec<Sig> = (0..w)
                    .map(|_| {
                        let d = self.rdata();
                        if kind == 12 {
                            let en = self.reg_clock();
                            self.custom_latch(style, d, en)
                        } else {
                            let ck = self.reg_clock();
                            let nck = self.b.not(ck);
                            let m = self.custom_latch(style, d, nck);
                            self.custom_latch(style, m, ck)
                        }
                    })
                    .collect();
                cat(&mut self.b, &bits)
            }
            _ => {
                // SR latch driven directly by top-level inputs (or logic if there are none).
                self.feature("sr_latch_inputs");
                let bits: Vec<Sig> = (0..w)
                    .map(|_| {
                        let (s, r) = (self.sig(), self.sig());
                        let d = if self.rng.chance(50) {
                            self.lib.nand_sr.clone()
                        } else {
                            self.lib.nor_sr.clone()
                        };
                        self.b.inst(&d, &[s, r])[0]
                    })
                    .collect();
                cat(&mut self.b, &bits)
            }
        };
        self.b.connect(out, q);
    }
}

impl Gen {
    /// A transparent-high latch built directly from gates (no library def):
    /// * 0: mux latch `q = d·en + q·¬en` (has a static hazard when `en` falls);
    /// * 1: mux latch with the consensus term `d·q` (hazard-free);
    /// * 2: NAND D latch whose feedback runs through 4 buffers (a loop longer than
    ///   `KERNEL_MAX_LOOP`: becomes a Sim kernel);
    /// * 3: `TRI(d, en)` onto a keeper-terminated net (a Keeper kernel).
    fn custom_latch(&mut self, style: usize, d: Sig, en: Sig) -> Sig {
        let b = &mut self.b;
        match style {
            0 | 1 => {
                let q = b.wire(1);
                let nen = b.not(en);
                let a = b.and(&[d, en]);
                let h = b.and(&[q, nen]);
                if style == 1 {
                    let c = b.and(&[d, q]);
                    b.gate_to(GateOp::Or, &[a, h, c], q);
                } else {
                    b.gate_to(GateOp::Or, &[a, h], q);
                }
                q
            }
            2 => {
                let nd = b.not(d);
                let s = b.nand(&[d, en]);
                let r = b.nand(&[nd, en]);
                let q = b.wire(1);
                let qn = b.wire(1);
                let mut fq = qn;
                for _ in 0..4 {
                    fq = b.buf(fq);
                }
                b.nand_to(&[s, fq], q);
                b.nand_to(&[r, q], qn);
                q
            }
            _ => {
                let q = b.wire(1);
                b.terminate(q, Termination::Keeper);
                b.tri_to(d, en, q);
                // Buffered read (the keeper net itself is the state).
                b.buf(q)
            }
        }
    }
}

/// Concatenate (LSB first); a single part is returned as is.
fn cat(b: &mut DefBuilder, parts: &[Sig]) -> Sig {
    if parts.len() == 1 {
        parts[0]
    } else {
        b.concat(parts)
    }
}

/// Generate the design for `seed` with time unit `scale` (even, ≥ 2): clock periods are
/// `period_u · scale` ticks. The structure depends only on `seed` and `cfg`.
pub fn generate(seed: u64, cfg: &GenConfig, scale: u32) -> Generated {
    assert!(
        scale >= 2 && scale.is_multiple_of(2),
        "scale must be even and ≥ 2"
    );
    let mut rng = Rng::new(seed);
    let clockless = rng.chance(cfg.clockless_pct);
    let shapes = if clockless {
        Vec::new()
    } else {
        clock_shapes(&mut rng, cfg)
    };
    let reset_extra = rng.below(2) as u32;
    // Usually released half a unit into a phase; sometimes exactly on the unit grid, where it can
    // coincide with clock edges (one simultaneous event).
    let reset_on_grid = rng.chance(20);
    let mut db = DesignBuilder::new();
    let lib = Lib {
        ms_dff: pc::ms_dff(&mut db),
        dff_rst: pc::dff_rst(&mut db),
        dff_en: pc::dff_en(&mut db),
        dff_arst: pc::dff_arst(&mut db),
        toggle_ff: pc::toggle_ff(&mut db),
        d_latch: pc::d_latch(&mut db),
        nand_sr: pc::nand_sr_latch(&mut db),
        nor_sr: pc::nor_sr_latch(&mut db),
    };
    let mut b = DefBuilder::new(&format!("rnd_{seed}"));
    let mut clocks = Vec::new();
    let mut infos = Vec::new();
    for (i, sh) in shapes.iter().enumerate() {
        let (period, high, phase) = (sh.period_u * scale, sh.high_u * scale, sh.phase_u * scale);
        let c = b.clock(period, high, phase);
        let name = format!("clk{i}");
        b.name(c, &name);
        clocks.push(c);
        infos.push(ClockInfo {
            name,
            shape: *sh,
            period,
            high,
            phase,
        });
    }
    // Second phase per clock: the other clock of a two-phase pair, else the inverse.
    let nclocks: Vec<Sig> = if shapes.len() == 2
        && shapes[0].period_u == shapes[1].period_u
        && shapes[0].high_u * 2 < shapes[0].period_u
    {
        vec![clocks[1], clocks[0]]
    } else {
        clocks.iter().map(|&c| b.not(c)).collect()
    };
    let reset_ticks =
        reset_units(&shapes, reset_extra) * scale + if reset_on_grid { 0 } else { scale / 2 };
    let rst = b.power_on_reset(reset_ticks);
    b.name(rst, "rst");
    let nrst = b.not(rst);

    let mut g = Gen {
        rng,
        db,
        b,
        lib,
        pool: vec![rst, nrst],
        clocks,
        nclocks,
        rst,
        nrst,
        features: Vec::new(),
        n_out: 0,
        asynch: false,
    };
    g.asynch = g.rng.chance(cfg.async_pct);
    if g.asynch {
        g.feature("async");
    }
    if shapes.len() == 2 {
        g.feature("two_clocks");
    }
    if reset_on_grid {
        g.feature("reset_on_grid");
    }
    if clockless {
        g.feature("clockless");
    }

    // Inputs.
    let mut n_in = g.rng.range(cfg.inputs.0, cfg.inputs.1);
    if clockless {
        n_in = n_in.max(2);
    }
    let mut inputs = Vec::new();
    for i in 0..n_in {
        let w = if g.rng.chance(20) {
            g.rng.range(2, 4) as u32
        } else {
            1
        };
        let name = format!("in{i}");
        let s = g.b.input(&name, w);
        g.push_bits(s);
        inputs.push(InputPin { name, width: w });
    }
    if g.rng.chance(25) {
        let c = g.b.lit(1, g.rng.below(2) as u64);
        g.pool.push(c);
    }

    // State elements: outputs are forward-declared so logic can feed back from them.
    let n_regs = g.rng.range(cfg.regs.0, cfg.regs.1);
    let a = g.aw(1);
    let kind_weights = [
        20,
        14,
        10,
        8,
        3,
        8,
        4,
        5,
        4,
        3,
        3 * a,
        if n_in >= 2 { 3 * a } else { 0 },
        3,
        5,
    ];
    let n_comb = g.rng.range(cfg.comb.0, cfg.comb.1);
    // (kind, output, comb position at which it is built or None = forward-declared).
    let mut regs = Vec::new();
    for i in 0..n_regs {
        let kind = g.rng.weighted(&kind_weights);
        let w = if matches!(kind, 3 | 4 | 7 | 10) {
            g.rng.range(1, 3) as u32
        } else if g.rng.chance(20) {
            2
        } else {
            1
        };
        let q = g.b.wire(w);
        g.b.name(q, &format!("q{i}"));
        // Most transparent latches are built in the middle of the logic from the signals that
        // exist at that point (no loop through a transparent latch); the others are
        // forward-declared like the flip-flops, so logic can feed back from them.
        let at = if matches!(kind, 5 | 6 | 12) && g.rng.chance(75) {
            Some(g.rng.below(n_comb + 1))
        } else {
            g.push_bits(q);
            None
        };
        regs.push((i, kind, q, at));
    }
    let build = |g: &mut Gen, i: usize, kind: usize, q: Sig, n_in_bits: usize| {
        if kind == 11 {
            // SR latches from inputs: draw S/R from the inputs only.
            let saved = std::mem::take(&mut g.pool);
            g.pool = saved[2..2 + n_in_bits].to_vec();
            g.state_element(kind, q);
            g.pool = saved;
        } else {
            g.state_element(kind, q);
        }
        g.b.output(&format!("Q{i}"), q);
    };
    let n_in_bits: usize = inputs.iter().map(|p| p.width as usize).sum();
    for pos in 0..=n_comb {
        for &(i, kind, q, at) in &regs {
            if at == Some(pos) {
                build(&mut g, i, kind, q, n_in_bits);
                g.push_bits(q);
            }
        }
        if pos < n_comb {
            g.comb_op();
        }
    }
    for &(i, kind, q, at) in &regs {
        if at.is_none() {
            build(&mut g, i, kind, q, n_in_bits);
        }
    }
    if g.rng.chance(10) {
        // A gated ring oscillator at very low weight: flagged by STA, X in Fast mode.
        if g.rng.chance(g.aw(20)) {
            g.feature("ring_oscillator");
            let ro = pc::ring_oscillator(&mut g.db, 3, true);
            let en = g.sig();
            let o = g.b.inst(&ro, &[en])[0];
            g.b.output("RO", o);
        }
    }
    let Gen {
        mut db,
        b,
        mut rng,
        features,
        ..
    } = g;
    let top = db.add(b.finish());
    let mut design = db.build(&top);
    let mut features = features;
    if rng.chance(cfg.inline_pct) {
        design = pc::inline_all(&design);
        features.push("inlined");
    }
    Generated {
        seed,
        scale,
        design,
        clocks: infos,
        reset_ticks,
        inputs,
        features,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_structure_for_every_scale() {
        for seed in 0..40 {
            let a = generate(seed, &GenConfig::default(), 8);
            let b = generate(seed, &GenConfig::default(), 20);
            let fa = panda_netlist::flatten(&a.design).expect("flatten");
            let fb = panda_netlist::flatten(&b.design).expect("flatten");
            assert_eq!(fa.gates.len(), fb.gates.len());
            assert_eq!(fa.net_names, fb.net_names);
            assert_eq!(a.features, b.features);
            for (ca, cb) in a.clocks.iter().zip(&b.clocks) {
                assert_eq!(ca.shape, cb.shape);
                assert_eq!(cb.period, ca.shape.period_u * 20);
            }
        }
    }

    #[test]
    fn deterministic() {
        let a = generate(7, &GenConfig::default(), 10);
        let b = generate(7, &GenConfig::default(), 10);
        assert_eq!(a.design, b.design);
    }
}
