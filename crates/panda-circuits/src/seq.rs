//! Latches, flip-flops and registers — all state from cross-coupled gates (SPEC D2).
//!
//! The flip-flops are rising-edge master-slave designs: a master [`d_latch`] transparent while
//! `CLK = 0` feeding a slave [`d_latch`] transparent while `CLK = 1`. Power-on state is `X`
//! (SPEC §5.4); blocks with a reset reach a known state after one clock edge with `RST = 1`
//! (synchronous) or immediately ([`dff_arst`], asynchronous).

use panda_netlist::{DefBuilder, DefRef, DesignBuilder, Sig};

use crate::arith::mux2;
use crate::util::broadcast;

/// NAND SR latch (active-low set/reset). Id `"nand_sr_latch"`.
///
/// Pins: `S_n` (in, 1), `R_n` (in, 1) → `Q` (out, 1), `Q_n` (out, 1).
/// `S_n = 0` sets, `R_n = 0` resets, both high holds; both low gives `Q = Q_n = 1`.
/// Internal nets `q`, `qn`.
pub fn nand_sr_latch(db: &mut DesignBuilder) -> DefRef {
    let mut b = DefBuilder::new("nand_sr_latch");
    let s_n = b.input("S_n", 1);
    let r_n = b.input("R_n", 1);
    let qn = b.wire(1);
    let q = b.nand(&[s_n, qn]);
    b.nand_to(&[r_n, q], qn);
    b.name(q, "q");
    b.name(qn, "qn");
    b.output("Q", q);
    b.output("Q_n", qn);
    db.add_once(b.finish())
}

/// NOR SR latch (active-high set/reset). Id `"nor_sr_latch"`.
///
/// Pins: `S` (in, 1), `R` (in, 1) → `Q` (out, 1), `Q_n` (out, 1).
/// `S = 1` sets, `R = 1` resets, both low holds; both high gives `Q = Q_n = 0`.
/// Internal nets `q`, `qn`.
pub fn nor_sr_latch(db: &mut DesignBuilder) -> DefRef {
    let mut b = DefBuilder::new("nor_sr_latch");
    let s = b.input("S", 1);
    let r = b.input("R", 1);
    let qn = b.wire(1);
    let q = b.nor(&[r, qn]);
    b.nor_to(&[s, q], qn);
    b.name(q, "q");
    b.name(qn, "qn");
    b.output("Q", q);
    b.output("Q_n", qn);
    db.add_once(b.finish())
}

/// Gated D latch, the classic 4-NAND + NOT form. Id `"d_latch"`.
///
/// Pins: `D` (in, 1), `EN` (in, 1) → `Q` (out, 1), `Q_n` (out, 1). Transparent while `EN = 1`,
/// holds while `EN = 0`.
///
/// `s = NAND(D, EN)`, `r = NAND(NOT D, EN)` steer a NAND SR core (`q`, `qn`).
pub fn d_latch(db: &mut DesignBuilder) -> DefRef {
    let mut b = DefBuilder::new("d_latch");
    let d = b.input("D", 1);
    let en = b.input("EN", 1);
    let nd = b.not(d);
    let s = b.nand(&[d, en]);
    let r = b.nand(&[nd, en]);
    b.name(s, "s");
    b.name(r, "r");
    let qn = b.wire(1);
    let q = b.nand(&[s, qn]);
    b.nand_to(&[r, q], qn);
    b.name(q, "q");
    b.name(qn, "qn");
    b.output("Q", q);
    b.output("Q_n", qn);
    db.add_once(b.finish())
}

/// Gated D latch with asynchronous clear. Id `"d_latch_clr"`.
///
/// Pins: `D` (in, 1), `EN` (in, 1), `CLR` (in, 1) → `Q` (out, 1), `Q_n` (out, 1).
/// `CLR = 1` forces `Q = 0, Q_n = 1` regardless of `EN`; otherwise behaves like [`d_latch`].
///
/// `s = NAND(D, EN, CLR_n)`, `r = NAND(NOT D, EN)`, `q = NAND(s, qn)`, `qn = NAND(r, q, CLR_n)`.
pub fn d_latch_clr(db: &mut DesignBuilder) -> DefRef {
    let mut b = DefBuilder::new("d_latch_clr");
    let d = b.input("D", 1);
    let en = b.input("EN", 1);
    let clr = b.input("CLR", 1);
    let clr_n = b.not(clr);
    let nd = b.not(d);
    let s = b.nand(&[d, en, clr_n]);
    let r = b.nand(&[nd, en]);
    b.name(s, "s");
    b.name(r, "r");
    let qn = b.wire(1);
    let q = b.nand(&[s, qn]);
    b.nand_to(&[r, q, clr_n], qn);
    b.name(q, "q");
    b.name(qn, "qn");
    b.output("Q", q);
    b.output("Q_n", qn);
    db.add_once(b.finish())
}

/// Rising-edge master-slave D flip-flop from two [`d_latch`]es (`master`, `slave`) and a clock
/// inverter. Id `"ms_dff"`.
///
/// Pins: `D` (in, 1), `CLK` (in, 1) → `Q` (out, 1), `Q_n` (out, 1).
pub fn ms_dff(db: &mut DesignBuilder) -> DefRef {
    let latch = d_latch(db);
    let mut b = DefBuilder::new("ms_dff");
    let d = b.input("D", 1);
    let clk = b.input("CLK", 1);
    let nclk = b.not(clk);
    b.name(nclk, "nclk");
    let m = b.inst_named("master", &latch, &[d, nclk]);
    b.name(m[0], "m");
    let s = b.inst_named("slave", &latch, &[m[0], clk]);
    b.output("Q", s[0]);
    b.output("Q_n", s[1]);
    db.add_once(b.finish())
}

/// Rising-edge D flip-flop with asynchronous clear, from two [`d_latch_clr`]es. Id `"dff_arst"`.
///
/// Pins: `D` (in, 1), `CLR` (in, 1), `CLK` (in, 1) → `Q` (out, 1), `Q_n` (out, 1).
/// `CLR = 1` forces `Q = 0` immediately (both latches are cleared).
pub fn dff_arst(db: &mut DesignBuilder) -> DefRef {
    let latch = d_latch_clr(db);
    let mut b = DefBuilder::new("dff_arst");
    let d = b.input("D", 1);
    let clr = b.input("CLR", 1);
    let clk = b.input("CLK", 1);
    let nclk = b.not(clk);
    b.name(nclk, "nclk");
    let m = b.inst_named("master", &latch, &[d, nclk, clr]);
    b.name(m[0], "m");
    let s = b.inst_named("slave", &latch, &[m[0], clk, clr]);
    b.output("Q", s[0]);
    b.output("Q_n", s[1]);
    db.add_once(b.finish())
}

/// Rising-edge D flip-flop with synchronous reset. Id `"dff_rst"`.
///
/// Pins: `D` (in, 1), `RST` (in, 1), `CLK` (in, 1) → `Q` (out, 1), `Q_n` (out, 1).
/// At a rising edge: `Q ← RST ? 0 : D`.
pub fn dff_rst(db: &mut DesignBuilder) -> DefRef {
    let ff = ms_dff(db);
    let mut b = DefBuilder::new("dff_rst");
    let d = b.input("D", 1);
    let rst = b.input("RST", 1);
    let clk = b.input("CLK", 1);
    let nrst = b.not(rst);
    let dd = b.and(&[d, nrst]);
    b.name(dd, "d");
    let o = b.inst_named("ff", &ff, &[dd, clk]);
    b.output("Q", o[0]);
    b.output("Q_n", o[1]);
    db.add_once(b.finish())
}

/// Rising-edge D flip-flop with load enable (hold mux in front of an [`ms_dff`]). Id `"dff_en"`.
///
/// Pins: `D` (in, 1), `EN` (in, 1), `CLK` (in, 1) → `Q` (out, 1), `Q_n` (out, 1).
/// At a rising edge: `Q ← EN ? D : Q`.
pub fn dff_en(db: &mut DesignBuilder) -> DefRef {
    let ff = ms_dff(db);
    let m2 = mux2(db, 1);
    let mut b = DefBuilder::new("dff_en");
    let d = b.input("D", 1);
    let en = b.input("EN", 1);
    let clk = b.input("CLK", 1);
    let q = b.wire(1);
    let dd = b.inst_named("hold", &m2, &[q, d, en])[0];
    b.name(dd, "d");
    let o = b.inst_named("ff", &ff, &[dd, clk]);
    b.connect(o[0], q);
    b.name(q, "q");
    b.output("Q", q);
    b.output("Q_n", o[1]);
    db.add_once(b.finish())
}

/// Toggle flip-flop with synchronous reset. Id `"toggle_ff"`.
///
/// Pins: `T` (in, 1), `RST` (in, 1), `CLK` (in, 1) → `Q` (out, 1), `Q_n` (out, 1).
/// At a rising edge: `Q ← RST ? 0 : Q ^ T`. Without a reset it stays `X` forever (SPEC §5.4).
pub fn toggle_ff(db: &mut DesignBuilder) -> DefRef {
    let ff = ms_dff(db);
    let mut b = DefBuilder::new("toggle_ff");
    let t = b.input("T", 1);
    let rst = b.input("RST", 1);
    let clk = b.input("CLK", 1);
    let q = b.wire(1);
    let x = b.xor(&[q, t]);
    let nrst = b.not(rst);
    let d = b.and(&[x, nrst]);
    b.name(d, "d");
    let o = b.inst_named("ff", &ff, &[d, clk]);
    b.connect(o[0], q);
    b.name(q, "q");
    b.output("Q", q);
    b.output("Q_n", o[1]);
    db.add_once(b.finish())
}

/// `n`-bit register with load enable and synchronous reset (enable-mux style). Id `"register_{n}"`.
///
/// Pins: `D` (in, n), `LOAD` (in, 1), `RST` (in, 1), `CLK` (in, 1) → `Q` (out, n).
/// At a rising edge: `Q ← RST ? 0 : (LOAD ? D : Q)`.
///
/// Next-state logic uses `n`-wide gates (one [`mux2`] + one AND), feeding [`ms_dff`] instances
/// `ff0..`. Internal buses: `q` (state), `d` (next state).
pub fn register(db: &mut DesignBuilder, n: u32) -> DefRef {
    assert!(n >= 1, "register: n must be ≥ 1");
    let ff = ms_dff(db);
    let m2 = mux2(db, n);
    let id = format!("register_{n}");
    let mut b = DefBuilder::new(&id);
    let d = b.input("D", n);
    let load = b.input("LOAD", 1);
    let rst = b.input("RST", 1);
    let clk = b.input("CLK", 1);
    let q = b.wire(n);
    let m = b.inst_named("hold", &m2, &[q, d, load])[0];
    let nrst = b.not(rst);
    let nrst_w = broadcast(&mut b, nrst, n);
    let next = b.and(&[m, nrst_w]);
    b.name(next, "d");
    let q_bits = flops(&mut b, &ff, next, clk);
    b.connect(q_bits, q);
    b.name(q, "q");
    b.output("Q", q);
    db.add_once(b.finish())
}

/// `n`-bit register whose clock is gated by `AND(CLK, LOAD | RST)` — the hobbyist idiom
/// (SPEC §12 variant c). Id `"register_gated_{n}"`.
///
/// Pins: `D` (in, n), `LOAD` (in, 1), `RST` (in, 1), `CLK` (in, 1) → `Q` (out, n) — the same
/// interface and the same behavior as [`register`] provided the gating constraint holds:
/// at a rising edge of `CLK`, `Q ← RST ? 0 : (LOAD ? D : Q)`.
///
/// **Constraint:** `LOAD | RST` must not rise while `CLK` is high, otherwise the gated clock
/// `gclk` produces a spurious rising edge. (Falling while `CLK` is high is harmless.) In practice,
/// change `LOAD`/`RST` only during the low phase of `CLK`.
///
/// Internal nets: `gclk` (gated clock), `d` (next state), `q` (state). Flip-flops `ff0..`.
pub fn register_gated(db: &mut DesignBuilder, n: u32) -> DefRef {
    assert!(n >= 1, "register_gated: n must be ≥ 1");
    let ff = ms_dff(db);
    let id = format!("register_gated_{n}");
    let mut b = DefBuilder::new(&id);
    let d = b.input("D", n);
    let load = b.input("LOAD", 1);
    let rst = b.input("RST", 1);
    let clk = b.input("CLK", 1);
    let en = b.or(&[load, rst]);
    let gclk = b.and(&[clk, en]);
    b.name(gclk, "gclk");
    let nrst = b.not(rst);
    let nrst_w = broadcast(&mut b, nrst, n);
    let next = b.and(&[d, nrst_w]);
    b.name(next, "d");
    let q = flops(&mut b, &ff, next, gclk);
    b.name(q, "q");
    b.output("Q", q);
    db.add_once(b.finish())
}

/// One [`ms_dff`] per bit of `d` (instances `ff0..`), all clocked by `clk`; returns the `Q` bus.
fn flops(b: &mut DefBuilder, ff: &DefRef, d: Sig, clk: Sig) -> Sig {
    let ds = b.bits(d);
    let qs: Vec<Sig> =
        ds.iter().enumerate().map(|(i, &di)| b.inst_named(&format!("ff{i}"), ff, &[di, clk])[0]).collect();
    b.concat(&qs)
}
