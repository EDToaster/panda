//! Scale test: flatten a ~1M-gate hierarchical design.
//! Run with `cargo test --release -p panda-netlist --test scale -- --ignored --nocapture`.

use std::time::Instant;

use panda_netlist::*;

fn full_adder() -> ComponentDef {
    let mut b = DefBuilder::new("fa");
    let a = b.input("A", 1);
    let bb = b.input("B", 1);
    let c = b.input("Cin", 1);
    let x = b.xor(&[a, bb]);
    let s = b.xor(&[x, c]);
    let t1 = b.and(&[a, bb]);
    let t2 = b.and(&[x, c]);
    let co = b.or(&[t1, t2]);
    b.output("S", s);
    b.output("Cout", co);
    b.finish()
}

/// `rows` ripple-carry adders of `bits` full adders each, chained through the carry.
fn big_design(rows: usize, bits: u32) -> Design {
    let mut db = DesignBuilder::new();
    let fa = db.add(full_adder());
    let mut rb = DefBuilder::new("row");
    let a = rb.input("A", bits);
    let bb = rb.input("B", bits);
    let mut c = rb.input("Cin", 1);
    let ab = rb.bits(a);
    let bbits = rb.bits(bb);
    let mut sums = Vec::with_capacity(bits as usize);
    for i in 0..bits as usize {
        let o = rb.inst(&fa, &[ab[i], bbits[i], c]);
        sums.push(o[0]);
        c = o[1];
    }
    let s = rb.concat(&sums);
    rb.output("S", s);
    rb.output("Cout", c);
    let row = db.add(rb.finish());
    let mut b = DefBuilder::new("top");
    let a = b.input("A", bits);
    let bb = b.input("B", bits);
    let mut c = b.input("C", 1);
    let mut acc = a;
    for _ in 0..rows {
        let o = b.inst(&row, &[acc, bb, c]);
        acc = o[0];
        c = o[1];
    }
    b.output("S", acc);
    b.output("Cout", c);
    let top = db.add(b.finish());
    db.build(&top)
}

#[test]
fn scale_small_sanity() {
    let d = big_design(3, 8);
    let f = flatten(&d).unwrap();
    assert_eq!(f.gates.len(), 3 * 8 * 5);
    assert_eq!(f.instances.len(), 1 + 3 + 3 * 8);
    let s0 = f.net_by_name("S[0]").unwrap();
    assert_eq!(f.net_by_name("u2/S[0]"), Some(s0));
    assert_eq!(f.net_by_name("u2/u0/S"), Some(s0));
}

#[test]
#[ignore = "release-mode scale test (~1M gates)"]
fn scale_one_million_gates() {
    let t0 = Instant::now();
    let d = big_design(1000, 200);
    let t1 = Instant::now();
    let f = flatten(&d).unwrap();
    let t2 = Instant::now();
    assert_eq!(f.gates.len(), 1_000_000);
    eprintln!(
        "build {:?}, flatten {:?}: {} gates, {} nets, {} names, {} instances",
        t1 - t0,
        t2 - t1,
        f.gates.len(),
        f.num_nets,
        f.name_index.len(),
        f.instances.len()
    );
    let budget = if cfg!(debug_assertions) { 60.0 } else { 5.0 };
    assert!((t2 - t1).as_secs_f64() < budget, "flatten took {:?}", t2 - t1);
}
