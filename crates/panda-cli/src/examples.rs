//! Small example designs built with the [`panda_netlist`] builder.
//!
//! These are the sources of the JSON files shipped under `crates/panda-cli/examples-data/`
//! (a test checks that the files are up to date; run it with `PANDA_BLESS=1` to regenerate them).
//! They are also available from the command line: `panda example <name>` prints one as JSON.
//!
//! | name               | top pins                                   | notes                                     |
//! |--------------------|--------------------------------------------|-------------------------------------------|
//! | `sr_latch`         | in `S_n`, `R_n`; out `Q`, `Q_n`            | two cross-coupled NANDs                   |
//! | `adder4`           | in `A[4]`, `B[4]`, `Cin`; out `S[4]`, `Cout` | ripple-carry adder of 4 full-adder defs  |
//! | `counter4`         | in `clk`, `rst`; out `Q[4]`                | synchronous counter, master-slave DFFs, synchronous reset |
//! | `counter4_clocked` | out `Q[4]`, `clk`                          | `counter4` driven by a `Clock` and a `PowerOnReset` |

use panda_netlist::{ComponentDef, DefBuilder, DefRef, Design, DesignBuilder};

/// Names of all examples, in a stable order.
pub const NAMES: &[&str] = &["sr_latch", "adder4", "counter4", "counter4_clocked"];

/// Build the example design called `name` (see [`NAMES`]).
pub fn by_name(name: &str) -> Option<Design> {
    Some(match name {
        "sr_latch" => sr_latch(),
        "adder4" => adder4(),
        "counter4" => counter4(),
        "counter4_clocked" => counter4_clocked(),
        _ => return None,
    })
}

/// Active-low NAND SR latch: `S_n`, `R_n` → `Q`, `Q_n`.
pub fn sr_latch() -> Design {
    let mut b = DefBuilder::new("sr_latch");
    let s_n = b.input("S_n", 1);
    let r_n = b.input("R_n", 1);
    let qn = b.wire(1);
    let q = b.nand(&[s_n, qn]);
    b.nand_to(&[r_n, q], qn);
    b.output("Q", q);
    b.output("Q_n", qn);
    let mut db = DesignBuilder::new();
    let top = db.add(b.finish());
    db.build(&top)
}

fn full_adder_def() -> ComponentDef {
    let mut b = DefBuilder::new("full_adder");
    let a = b.input("A", 1);
    let bb = b.input("B", 1);
    let c = b.input("Cin", 1);
    let x = b.xor(&[a, bb]);
    b.name(x, "axb");
    let s = b.xor(&[x, c]);
    let t1 = b.and(&[a, bb]);
    let t2 = b.and(&[x, c]);
    let co = b.or(&[t1, t2]);
    b.output("S", s);
    b.output("Cout", co);
    b.finish()
}

/// 4-bit ripple-carry adder: `A[4]`, `B[4]`, `Cin` → `S[4]`, `Cout`. Internal carries are named
/// `carry[0..3]` (bit `i` = carry out of stage `i`).
pub fn adder4() -> Design {
    let mut db = DesignBuilder::new();
    let fa = db.add_once(full_adder_def());
    let mut b = DefBuilder::new("adder4");
    let a = b.input("A", 4);
    let bb = b.input("B", 4);
    let mut c = b.input("Cin", 1);
    let abits = b.bits(a);
    let bbits = b.bits(bb);
    let mut sums = Vec::new();
    let mut carries = Vec::new();
    for i in 0..4 {
        let o = b.inst_named(&format!("fa{i}"), &fa, &[abits[i], bbits[i], c]);
        sums.push(o[0]);
        c = o[1];
        carries.push(c);
    }
    let carry = b.concat(&carries);
    b.name(carry, "carry");
    let s = b.concat(&sums);
    b.output("S", s);
    b.output("Cout", c);
    let top = db.add(b.finish());
    db.build(&top)
}

/// Gated NAND D latch: transparent while `E = 1`.
fn d_latch_def() -> ComponentDef {
    let mut b = DefBuilder::new("d_latch");
    let d = b.input("D", 1);
    let e = b.input("E", 1);
    let dn = b.not(d);
    let s_n = b.nand(&[d, e]);
    let r_n = b.nand(&[dn, e]);
    let qn = b.wire(1);
    let q = b.nand(&[s_n, qn]);
    b.nand_to(&[r_n, q], qn);
    b.output("Q", q);
    b.output("Q_n", qn);
    b.finish()
}

/// Positive-edge master-slave D flip-flop built from two D latches.
fn dff_def(latch: &DefRef) -> ComponentDef {
    let mut b = DefBuilder::new("dff");
    let d = b.input("D", 1);
    let clk = b.input("clk", 1);
    let clk_n = b.not(clk);
    let m = b.inst_named("master", latch, &[d, clk_n]);
    b.name(m[0], "m");
    let s = b.inst_named("slave", latch, &[m[0], clk]);
    b.output("Q", s[0]);
    b.finish()
}

fn counter4_def(db: &mut DesignBuilder) -> DefRef {
    let latch = db.add_once(d_latch_def());
    let dff = db.add_once(dff_def(&latch));
    let mut b = DefBuilder::new("counter4");
    let clk = b.input("clk", 1);
    let rst = b.input("rst", 1);
    let rst_n = b.not(rst);
    // Forward references for the register outputs.
    let q: Vec<_> = (0..4).map(|_| b.wire(1)).collect();
    let mut carry: Option<panda_netlist::Sig> = None;
    let mut next = Vec::new();
    for (i, &qi) in q.iter().enumerate() {
        // Toggle bit i when all lower bits are 1.
        let n = match carry {
            None => b.not(qi),
            Some(c) => b.xor(&[qi, c]),
        };
        carry = Some(match carry {
            None => qi,
            Some(c) => b.and(&[c, qi]),
        });
        let d = b.and(&[n, rst_n]);
        let out = b.inst_named(&format!("ff{i}"), &dff, &[d, clk]);
        b.connect(out[0], qi);
        next.push(d);
    }
    let next = b.concat(&next);
    b.name(next, "next");
    let qbus = b.concat(&q);
    b.output("Q", qbus);
    db.add_once(b.finish())
}

/// 4-bit synchronous up-counter with synchronous active-high reset: `clk`, `rst` → `Q[4]`.
/// The state changes a few ticks after a rising edge of `clk`.
pub fn counter4() -> Design {
    let mut db = DesignBuilder::new();
    let top = counter4_def(&mut db);
    db.build(&top)
}

/// [`counter4`] wrapped in a top def that drives it from a `Clock` (period 20, high 10, rising at
/// tick 10) and a `PowerOnReset` held for 45 ticks. Outputs `Q[4]` and the clock as `clk`.
pub fn counter4_clocked() -> Design {
    let mut db = DesignBuilder::new();
    let counter = counter4_def(&mut db);
    let mut b = DefBuilder::new("counter4_clocked");
    let clk = b.clock(20, 10, 10);
    let rst = b.power_on_reset(45);
    b.name(rst, "rst");
    let q = b.inst_named("cnt", &counter, &[clk, rst]);
    b.output("Q", q[0]);
    b.output("clk", clk);
    let top = db.add(b.finish());
    db.build(&top)
}
