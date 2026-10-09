//! Counters.

use panda_netlist::{DefBuilder, DefRef, DesignBuilder, Sig};

use crate::arith::incrementer;
use crate::seq::{dff_arst, register};

/// `n`-bit synchronous up-counter with count enable and synchronous reset.
/// Id `"sync_counter_{n}"`.
///
/// Pins: `CLK` (in, 1), `RST` (in, 1), `EN` (in, 1) → `Q` (out, n).
/// At a rising edge: `Q ← RST ? 0 : (EN ? Q + 1 : Q)` (mod 2^n).
///
/// Built from an [`incrementer`] (`inc`) feeding a [`register`] (`reg`, `LOAD = EN`).
/// Internal bus `next` = `Q + 1`.
pub fn sync_counter(db: &mut DesignBuilder, n: u32) -> DefRef {
    assert!(n >= 1, "sync_counter: n must be ≥ 1");
    let inc = incrementer(db, n);
    let reg = register(db, n);
    let id = format!("sync_counter_{n}");
    let mut b = DefBuilder::new(&id);
    let clk = b.input("CLK", 1);
    let rst = b.input("RST", 1);
    let en = b.input("EN", 1);
    let q = b.wire(n);
    let one = b.lit(1, 1);
    let next = b.inst_named("inc", &inc, &[q, one])[0];
    b.name(next, "next");
    let out = b.inst_named("reg", &reg, &[next, en, rst, clk])[0];
    b.connect(out, q);
    b.output("Q", q);
    db.add_once(b.finish())
}

/// `n`-bit asynchronous (ripple) up-counter with asynchronous reset. Id `"ripple_counter_{n}"`.
///
/// Pins: `CLK` (in, 1), `RST` (in, 1) → `Q` (out, n).
/// Counts rising edges of `CLK`; `RST = 1` clears all stages immediately.
///
/// Stage `i` is a [`dff_arst`] (`t{i}`) wired as a toggle (`D = Q_n`). Stage 0 is clocked by `CLK`,
/// stage `i > 0` by stage `i-1`'s `Q_n` — a derived clock (SPEC §8.5.4), so bit `i` settles about
/// `i` flip-flop delays after the edge.
pub fn ripple_counter(db: &mut DesignBuilder, n: u32) -> DefRef {
    assert!(n >= 1, "ripple_counter: n must be ≥ 1");
    let ff = dff_arst(db);
    let id = format!("ripple_counter_{n}");
    let mut b = DefBuilder::new(&id);
    let clk = b.input("CLK", 1);
    let rst = b.input("RST", 1);
    let mut stage_clk = clk;
    let mut qs: Vec<Sig> = Vec::with_capacity(n as usize);
    for i in 0..n {
        let qn = b.wire(1);
        let o = b.inst_named(&format!("t{i}"), &ff, &[qn, rst, stage_clk]);
        b.connect(o[1], qn);
        qs.push(o[0]);
        stage_clk = qn;
    }
    let q = b.concat(&qs);
    b.output("Q", q);
    db.add_once(b.finish())
}
