//! Combinational arithmetic and selection blocks.

use panda_netlist::{DefBuilder, DefRef, DesignBuilder, GateOp, Sig};

use crate::util::broadcast;

/// Half adder. Id `"half_adder"`.
///
/// Pins: `A` (in, 1), `B` (in, 1) → `S` (out, 1) = A xor B, `C` (out, 1) = A and B.
pub fn half_adder(db: &mut DesignBuilder) -> DefRef {
    let mut b = DefBuilder::new("half_adder");
    let a = b.input("A", 1);
    let bb = b.input("B", 1);
    let s = b.xor(&[a, bb]);
    let c = b.and(&[a, bb]);
    b.output("S", s);
    b.output("C", c);
    db.add_once(b.finish())
}

/// Full adder. Id `"full_adder"`.
///
/// Pins: `A`, `B`, `CIN` (in, 1) → `S` (out, 1), `COUT` (out, 1).
/// `S = A ^ B ^ CIN`, `COUT = A·B + CIN·(A ^ B)`; carry path depth 2 per bit.
pub fn full_adder(db: &mut DesignBuilder) -> DefRef {
    let mut b = DefBuilder::new("full_adder");
    let a = b.input("A", 1);
    let bb = b.input("B", 1);
    let cin = b.input("CIN", 1);
    let x = b.xor(&[a, bb]);
    b.name(x, "p");
    let s = b.xor(&[x, cin]);
    let g = b.and(&[a, bb]);
    b.name(g, "g");
    let t = b.and(&[x, cin]);
    let cout = b.or(&[g, t]);
    b.output("S", s);
    b.output("COUT", cout);
    db.add_once(b.finish())
}

/// `n`-bit ripple-carry adder built from [`full_adder`] instances `fa0..`. Id `"ripple_adder_{n}"`.
///
/// Pins: `A` (in, n), `B` (in, n), `CIN` (in, 1) → `S` (out, n), `COUT` (out, 1).
/// Internal carries are named `c[0..=n]` (`c[0] = CIN`, `c[n] = COUT`).
pub fn ripple_adder(db: &mut DesignBuilder, n: u32) -> DefRef {
    assert!(n >= 1, "ripple_adder: n must be ≥ 1");
    let fa = full_adder(db);
    let id = format!("ripple_adder_{n}");
    let mut b = DefBuilder::new(&id);
    let a = b.input("A", n);
    let bb = b.input("B", n);
    let cin = b.input("CIN", 1);
    let ab = b.bits(a);
    let bbits = b.bits(bb);
    let mut carry = cin;
    let mut carries = vec![cin];
    let mut sums = Vec::with_capacity(n as usize);
    for i in 0..n as usize {
        let o = b.inst_named(&format!("fa{i}"), &fa, &[ab[i], bbits[i], carry]);
        sums.push(o[0]);
        carry = o[1];
        carries.push(carry);
    }
    let c = b.concat(&carries);
    b.name(c, "c");
    let s = b.concat(&sums);
    b.output("S", s);
    b.output("COUT", carry);
    db.add_once(b.finish())
}

/// `n`-bit incrementer: `Y = A + CIN` from a chain of [`half_adder`]s `ha0..`.
/// Id `"incrementer_{n}"`.
///
/// Pins: `A` (in, n), `CIN` (in, 1) → `Y` (out, n), `COUT` (out, 1).
/// Tie `CIN` to 1 for a plain `+1`; driving it from an enable gives a conditional increment.
pub fn incrementer(db: &mut DesignBuilder, n: u32) -> DefRef {
    assert!(n >= 1, "incrementer: n must be ≥ 1");
    let ha = half_adder(db);
    let id = format!("incrementer_{n}");
    let mut b = DefBuilder::new(&id);
    let a = b.input("A", n);
    let cin = b.input("CIN", 1);
    let ab = b.bits(a);
    let mut carry = cin;
    let mut ys = Vec::with_capacity(n as usize);
    for (i, &ai) in ab.iter().enumerate() {
        let o = b.inst_named(&format!("ha{i}"), &ha, &[ai, carry]);
        ys.push(o[0]);
        carry = o[1];
    }
    let y = b.concat(&ys);
    b.output("Y", y);
    b.output("COUT", carry);
    db.add_once(b.finish())
}

/// `w`-bit 2:1 multiplexer (AND-OR form, `w`-wide gates). Id `"mux2_{w}"`.
///
/// Pins: `A` (in, w), `B` (in, w), `SEL` (in, 1) → `Y` (out, w). `SEL = 0 → A`, `SEL = 1 → B`.
/// Depth 3 from `SEL`, 2 from the data inputs.
pub fn mux2(db: &mut DesignBuilder, w: u32) -> DefRef {
    assert!(w >= 1, "mux2: width must be ≥ 1");
    let id = format!("mux2_{w}");
    let mut b = DefBuilder::new(&id);
    let a = b.input("A", w);
    let bb = b.input("B", w);
    let sel = b.input("SEL", 1);
    let nsel = b.not(sel);
    let selw = broadcast(&mut b, sel, w);
    let nselw = broadcast(&mut b, nsel, w);
    let ta = b.and(&[a, nselw]);
    let tb = b.and(&[bb, selw]);
    let y = b.or(&[ta, tb]);
    b.output("Y", y);
    db.add_once(b.finish())
}

/// `2^n_sel`-input, `w`-bit multiplexer: a binary tree of [`mux2`] instances. Id `"mux_{n_sel}x{w}"`.
///
/// Pins: `D0`, `D1`, …, `D{2^n_sel - 1}` (in, w each), `SEL` (in, n_sel) → `Y` (out, w).
/// `Y = D[SEL]`. Tree level `k` (leaves = level 0) is steered by `SEL[k]`.
pub fn mux(db: &mut DesignBuilder, n_sel: u32, w: u32) -> DefRef {
    assert!((1..=16).contains(&n_sel), "mux: n_sel must be in 1..=16");
    let m2 = mux2(db, w);
    let id = format!("mux_{n_sel}x{w}");
    let mut b = DefBuilder::new(&id);
    let data: Vec<Sig> = (0..1u32 << n_sel).map(|i| b.input(&format!("D{i}"), w)).collect();
    let sel = b.input("SEL", n_sel);
    let sels = b.bits(sel);
    let y = mux_tree(&mut b, &m2, data, &sels, "m");
    b.output("Y", y);
    db.add_once(b.finish())
}

/// Reduce `level` (length `2^sels.len()`) to one signal with a tree of `m2` instances named
/// `{prefix}{level}_{index}`.
pub(crate) fn mux_tree(b: &mut DefBuilder, m2: &DefRef, mut level: Vec<Sig>, sels: &[Sig], prefix: &str) -> Sig {
    assert_eq!(level.len(), 1 << sels.len(), "mux_tree: input count must be 2^select bits");
    for (k, &s) in sels.iter().enumerate() {
        level = level
            .chunks(2)
            .enumerate()
            .map(|(j, pair)| b.inst_named(&format!("{prefix}{k}_{j}"), m2, &[pair[0], pair[1], s])[0])
            .collect();
    }
    level[0]
}

/// `n`-to-`2^n` one-hot decoder with enable. Id `"decoder_{n}"`.
///
/// Pins: `A` (in, n), `EN` (in, 1) → `Y` (out, 2^n). `Y[i] = EN and (A == i)`.
/// Each output is one (n+1)-input AND of true/complemented address bits (depth 2).
pub fn decoder(db: &mut DesignBuilder, n: u32) -> DefRef {
    assert!((1..=12).contains(&n), "decoder: n must be in 1..=12");
    let id = format!("decoder_{n}");
    let mut b = DefBuilder::new(&id);
    let a = b.input("A", n);
    let en = b.input("EN", 1);
    let na = b.not(a);
    let t = b.bits(a);
    let f = b.bits(na);
    let ys: Vec<Sig> = (0..1usize << n)
        .map(|i| {
            let mut ins: Vec<Sig> = (0..n as usize).map(|k| if (i >> k) & 1 == 1 { t[k] } else { f[k] }).collect();
            ins.push(en);
            b.and(&ins)
        })
        .collect();
    let y = b.concat(&ys);
    b.output("Y", y);
    db.add_once(b.finish())
}

/// `n`-bit equality comparator: XNOR then AND-reduce. Id `"equality_{n}"`.
///
/// Pins: `A` (in, n), `B` (in, n) → `EQ` (out, 1).
pub fn equality(db: &mut DesignBuilder, n: u32) -> DefRef {
    assert!(n >= 1, "equality: n must be ≥ 1");
    let id = format!("equality_{n}");
    let mut b = DefBuilder::new(&id);
    let a = b.input("A", n);
    let bb = b.input("B", n);
    let x = b.xnor(&[a, bb]);
    let eq = if n == 1 { x } else { b.reduce(GateOp::And, x) };
    b.output("EQ", eq);
    db.add_once(b.finish())
}
