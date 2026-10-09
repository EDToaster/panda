//! Exhaustive / randomized behavioral tests of the combinational blocks (Timing mode).

mod common;

use common::*;
use panda_circuits::*;
use panda_netlist::{DesignBuilder, Logic, Value};

#[test]
fn half_adder_exhaustive() {
    let mut db = DesignBuilder::new();
    let top = half_adder(&mut db);
    let mut s = sim_top(db, &top);
    for a in 0..2 {
        for b in 0..2 {
            set(&mut s, "A", a);
            set(&mut s, "B", b);
            settle(&mut s);
            assert_eq!(get(&s, "S"), a ^ b);
            assert_eq!(get(&s, "C"), a & b);
        }
    }
    assert_clean(&mut s);
}

#[test]
fn full_adder_exhaustive() {
    let mut db = DesignBuilder::new();
    let top = full_adder(&mut db);
    let mut s = sim_top(db, &top);
    for v in 0..8u64 {
        let (a, b, c) = (v & 1, (v >> 1) & 1, (v >> 2) & 1);
        set(&mut s, "A", a);
        set(&mut s, "B", b);
        set(&mut s, "CIN", c);
        settle(&mut s);
        let sum = a + b + c;
        assert_eq!(get(&s, "S"), sum & 1, "inputs {a}{b}{c}");
        assert_eq!(get(&s, "COUT"), sum >> 1, "inputs {a}{b}{c}");
    }
    assert_clean(&mut s);
}

#[test]
fn ripple_adder_4_exhaustive() {
    let mut db = DesignBuilder::new();
    let top = ripple_adder(&mut db, 4);
    let mut s = sim_top(db, &top);
    for a in 0..16 {
        for b in 0..16 {
            for c in 0..2 {
                set(&mut s, "A", a);
                set(&mut s, "B", b);
                set(&mut s, "CIN", c);
                settle(&mut s);
                let sum = a + b + c;
                assert_eq!(get(&s, "S"), sum & 15, "{a}+{b}+{c}");
                assert_eq!(get(&s, "COUT"), sum >> 4, "{a}+{b}+{c}");
                // Internal carry chain is probe-able.
                assert_eq!(get(&s, "c") >> 4, sum >> 4);
            }
        }
    }
    assert_clean(&mut s);
}

#[test]
fn ripple_adder_16_random() {
    let mut db = DesignBuilder::new();
    let top = ripple_adder(&mut db, 16);
    let mut s = sim_top(db, &top);
    let mut rng = Rng::new(7);
    let mut cases: Vec<(u64, u64, u64)> = vec![(0xffff, 0, 1), (0xffff, 0xffff, 1), (0x8000, 0x8000, 0)];
    cases.extend((0..300).map(|_| (rng.below(1 << 16), rng.below(1 << 16), rng.below(2))));
    for (a, b, c) in cases {
        set(&mut s, "A", a);
        set(&mut s, "B", b);
        set(&mut s, "CIN", c);
        settle(&mut s);
        let sum = a + b + c;
        assert_eq!(get(&s, "S"), sum & 0xffff, "{a}+{b}+{c}");
        assert_eq!(get(&s, "COUT"), sum >> 16, "{a}+{b}+{c}");
    }
}

#[test]
fn ripple_adder_carry_depth_is_two_per_bit() {
    // From a settled state, a carry rippling through the whole chain takes 2 gate levels per bit
    // (AND + OR), measured on COUT; the input change itself lands one tick after `set_input`.
    let mut db = DesignBuilder::new();
    let top = ripple_adder(&mut db, 8);
    let mut s = sim_top(db, &top);
    set(&mut s, "A", 0xff);
    set(&mut s, "B", 0);
    set(&mut s, "CIN", 0);
    settle(&mut s);
    assert_eq!(get(&s, "COUT"), 0);
    let t0 = s.tick();
    set(&mut s, "CIN", 1);
    while get(&s, "COUT") == 0 {
        s.step();
        assert!(s.tick() - t0 < 100);
    }
    // Input applied at t0+1, then 2 gate levels per bit.
    assert_eq!(s.tick() - t0, 1 + 2 * 8);
}

#[test]
fn incrementer_exhaustive() {
    let mut db = DesignBuilder::new();
    let top = incrementer(&mut db, 5);
    let mut s = sim_top(db, &top);
    for a in 0..32 {
        for c in 0..2 {
            set(&mut s, "A", a);
            set(&mut s, "CIN", c);
            settle(&mut s);
            assert_eq!(get(&s, "Y"), (a + c) & 31);
            assert_eq!(get(&s, "COUT"), (a + c) >> 5);
        }
    }
    assert_clean(&mut s);
}

#[test]
fn mux2_exhaustive_and_wide() {
    let mut db = DesignBuilder::new();
    let top = mux2(&mut db, 1);
    let mut s = sim_top(db, &top);
    for v in 0..8u64 {
        let (a, b, sel) = (v & 1, (v >> 1) & 1, (v >> 2) & 1);
        set(&mut s, "A", a);
        set(&mut s, "B", b);
        set(&mut s, "SEL", sel);
        settle(&mut s);
        assert_eq!(get(&s, "Y"), if sel == 1 { b } else { a });
    }

    let mut db = DesignBuilder::new();
    let top = mux2(&mut db, 12);
    let mut s = sim_top(db, &top);
    let mut rng = Rng::new(3);
    for _ in 0..100 {
        let (a, b, sel) = (rng.below(1 << 12), rng.below(1 << 12), rng.below(2));
        set(&mut s, "A", a);
        set(&mut s, "B", b);
        set(&mut s, "SEL", sel);
        settle(&mut s);
        assert_eq!(get(&s, "Y"), if sel == 1 { b } else { a });
    }
    assert_clean(&mut s);
}

#[test]
fn mux2_x_select_is_pessimistic() {
    let mut db = DesignBuilder::new();
    let top = mux2(&mut db, 2);
    let mut s = sim_top(db, &top);
    set(&mut s, "A", 0);
    set(&mut s, "B", 0);
    s.set_input("SEL", &Value::xs(1)).unwrap();
    settle(&mut s);
    assert_eq!(get(&s, "Y"), 0, "both inputs 0 → 0 even with SEL = X");
    set(&mut s, "A", 1);
    settle(&mut s);
    assert_eq!(s.read("Y").unwrap().get(0), Logic::X);
}

#[test]
fn mux_tree_exhaustive() {
    let mut db = DesignBuilder::new();
    let top = mux(&mut db, 3, 4);
    assert_eq!(top.id, "mux_3x4");
    assert_eq!(top.pins.len(), 8 + 1 + 1);
    let mut s = sim_top(db, &top);
    let data: Vec<u64> = (0..8).map(|i| (i * 5 + 3) % 16).collect();
    for (i, &d) in data.iter().enumerate() {
        set(&mut s, &format!("D{i}"), d);
    }
    for sel in 0..8 {
        set(&mut s, "SEL", sel);
        settle(&mut s);
        assert_eq!(get(&s, "Y"), data[sel as usize]);
    }
    // Changing an unselected input does not affect the output.
    set(&mut s, "SEL", 2);
    set(&mut s, "D5", 15 - data[5]);
    settle(&mut s);
    assert_eq!(get(&s, "Y"), data[2]);
    assert_clean(&mut s);
}

#[test]
fn decoder_exhaustive() {
    let mut db = DesignBuilder::new();
    let top = decoder(&mut db, 3);
    let mut s = sim_top(db, &top);
    for en in 0..2 {
        for a in 0..8 {
            set(&mut s, "A", a);
            set(&mut s, "EN", en);
            settle(&mut s);
            assert_eq!(get(&s, "Y"), if en == 1 { 1 << a } else { 0 }, "A={a} EN={en}");
        }
    }
    assert_clean(&mut s);
}

#[test]
fn equality_exhaustive() {
    for n in [1, 3] {
        let mut db = DesignBuilder::new();
        let top = equality(&mut db, n);
        let mut s = sim_top(db, &top);
        for a in 0..1 << n {
            for b in 0..1 << n {
                set(&mut s, "A", a);
                set(&mut s, "B", b);
                settle(&mut s);
                assert_eq!(get(&s, "EQ"), u64::from(a == b), "n={n} {a}=={b}");
            }
        }
        assert_clean(&mut s);
    }
}

#[test]
fn defs_are_shared_and_ids_encode_parameters() {
    let mut db = DesignBuilder::new();
    let a = ripple_adder(&mut db, 8);
    let b = ripple_adder(&mut db, 8);
    let c = ripple_adder(&mut db, 4);
    assert_eq!(a, b);
    assert_eq!(a.id, "ripple_adder_8");
    assert_eq!(c.id, "ripple_adder_4");
    let names: Vec<_> = a.pins.iter().map(|p| (p.name.as_str(), p.width)).collect();
    assert_eq!(names, [("A", 8), ("B", 8), ("CIN", 1), ("S", 8), ("COUT", 1)]);
    let design = db.build(&a);
    // full_adder, ripple_adder_8, ripple_adder_4 — full_adder appears once.
    assert_eq!(design.defs.iter().filter(|d| d.id == "full_adder").count(), 1);
    assert_eq!(design.defs.len(), 3);
}
