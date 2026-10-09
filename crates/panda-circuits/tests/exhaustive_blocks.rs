//! Exhaustive checks of the combinational blocks against Rust reference functions, over many
//! parameterizations (Timing mode, settled values), plus 4-state checks of the documented gate
//! structure against expressions built from `panda_netlist::eval`.

mod common;

use common::*;
use panda_circuits::*;
use panda_netlist::eval::{and2, not1, or2, xor2};
use panda_netlist::{DefRef, DesignBuilder, Logic, Value};
use panda_timing::TimingSim;

fn sim_of(f: impl FnOnce(&mut DesignBuilder) -> DefRef) -> TimingSim {
    let mut db = DesignBuilder::new();
    let top = f(&mut db);
    sim_top(db, &top)
}

fn apply(s: &mut TimingSim, inputs: &[(&str, u64)]) {
    for &(p, v) in inputs {
        set(s, p, v);
    }
    settle(s);
}

fn mask(n: u32) -> u64 {
    if n >= 64 { u64::MAX } else { (1u64 << n) - 1 }
}

#[test]
fn adders_exhaustive_all_small_widths() {
    let mut s = sim_of(half_adder);
    for v in 0..4u64 {
        apply(&mut s, &[("A", v & 1), ("B", v >> 1)]);
        assert_eq!((get(&s, "S"), get(&s, "C")), ((v & 1) ^ (v >> 1), (v & 1) & (v >> 1)));
    }
    for n in 1..=6u32 {
        let mut s = sim_of(|db| ripple_adder(db, n));
        for a in 0..1u64 << n {
            for b in 0..1u64 << n {
                for c in 0..2 {
                    apply(&mut s, &[("A", a), ("B", b), ("CIN", c)]);
                    let sum = a + b + c;
                    assert_eq!(get(&s, "S"), sum & mask(n), "n={n} {a}+{b}+{c}");
                    assert_eq!(get(&s, "COUT"), sum >> n, "n={n} {a}+{b}+{c}");
                    assert_eq!(get(&s, "c"), (0..=n).fold(0, |acc, i| acc | (((a & mask(i)) + (b & mask(i)) + c) >> i) << i));
                }
            }
        }
        assert_clean(&mut s);
    }
}

#[test]
fn incrementer_exhaustive_all_small_widths() {
    for n in 1..=9u32 {
        let mut s = sim_of(|db| incrementer(db, n));
        for a in 0..1u64 << n {
            for c in 0..2 {
                apply(&mut s, &[("A", a), ("CIN", c)]);
                assert_eq!(get(&s, "Y"), (a + c) & mask(n), "n={n}");
                assert_eq!(get(&s, "COUT"), (a + c) >> n, "n={n}");
            }
        }
        assert_clean(&mut s);
    }
}

#[test]
fn mux_exhaustive_selects_and_every_data_bit() {
    for n_sel in 1..=4u32 {
        for w in [1u32, 2, 3, 65] {
            let mut s = sim_of(|db| mux(db, n_sel, w));
            let mut rng = Rng::new(u64::from(n_sel * 100 + w));
            let k = 1usize << n_sel;
            let data: Vec<Value> = (0..k)
                .map(|_| {
                    let bits: Vec<Logic> =
                        (0..w).map(|_| if rng.below(2) == 1 { Logic::One } else { Logic::Zero }).collect();
                    Value::from_bits(&bits)
                })
                .collect();
            for (i, d) in data.iter().enumerate() {
                s.set_input(&format!("D{i}"), d).unwrap();
            }
            for sel in 0..k {
                apply(&mut s, &[("SEL", sel as u64)]);
                assert_eq!(s.read("Y").unwrap(), data[sel], "n_sel={n_sel} w={w} sel={sel}");
                // Flip one bit of every data input: only the selected one may matter.
                for (i, d) in data.iter().enumerate() {
                    let mut f = d.clone();
                    let b = (i as u32 * 7) % w;
                    f.set(b, not1(f.get(b)));
                    s.set_input(&format!("D{i}"), &f).unwrap();
                    settle(&mut s);
                    let expect = if i == sel { &f } else { &data[sel] };
                    assert_eq!(&s.read("Y").unwrap(), expect, "n_sel={n_sel} w={w} sel={sel} flip D{i}");
                    s.set_input(&format!("D{i}"), d).unwrap();
                }
            }
            settle(&mut s);
            assert_clean(&mut s);
        }
    }
}

#[test]
fn decoder_and_equality_exhaustive() {
    for n in 1..=6u32 {
        let mut s = sim_of(|db| decoder(db, n));
        for en in 0..2 {
            for a in 0..1u64 << n {
                apply(&mut s, &[("A", a), ("EN", en)]);
                let y = s.read("Y").unwrap();
                assert_eq!(y.width(), 1 << n);
                for i in 0..1u64 << n {
                    let expect = if en == 1 && i == a { Logic::One } else { Logic::Zero };
                    assert_eq!(y.get(i as u32), expect, "decoder_{n} A={a} EN={en} bit {i}");
                }
            }
        }
        assert_clean(&mut s);
        let mut s = sim_of(|db| equality(db, n));
        for a in 0..1u64 << n {
            for b in 0..1u64 << n {
                apply(&mut s, &[("A", a), ("B", b)]);
                assert_eq!(get(&s, "EQ"), u64::from(a == b));
            }
        }
        assert_clean(&mut s);
    }
}

#[test]
fn rom_reads_every_address_for_many_shapes() {
    let mut rng = Rng::new(99);
    for addr_bits in 1..=6u32 {
        for data_bits in [1u32, 7, 63, 64] {
            let words = 1usize << addr_bits;
            let len = rng.below(words as u64 + 1) as usize;
            let contents: Vec<u64> = (0..len).map(|_| rng.next_u64() & mask(data_bits)).collect();
            let mut s = sim_of(|db| rom(db, "t", addr_bits, data_bits, &contents));
            for a in 0..words {
                apply(&mut s, &[("A", a as u64)]);
                let expect = contents.get(a).copied().unwrap_or(0);
                assert_eq!(get(&s, "D"), expect, "rom {addr_bits}x{data_bits} A={a}");
            }
            assert_clean(&mut s);
        }
    }
}

fn rand4(rng: &mut Rng) -> Logic {
    match rng.below(5) {
        0 => Logic::X,
        1 => Logic::Z,
        2 | 3 => Logic::One,
        _ => Logic::Zero,
    }
}

fn rand_val(rng: &mut Rng, w: u32) -> Value {
    Value::from_bits(&(0..w).map(|_| rand4(rng)).collect::<Vec<_>>())
}

/// The documented gate structure of `mux2` / `equality` / `decoder` / `full_adder`, evaluated with
/// the reference 4-state gate functions, matches the block's settled outputs for random 4-state
/// inputs (exact X pessimism, Z read as X).
#[test]
fn four_state_structure_matches_documented_formulas() {
    let mut rng = Rng::new(4);
    // mux2: Y = (A & !SEL) | (B & SEL), bitwise.
    let mut s = sim_of(|db| mux2(db, 3));
    for _ in 0..300 {
        let (a, b, sel) = (rand_val(&mut rng, 3), rand_val(&mut rng, 3), rand4(&mut rng));
        s.set_input("A", &a).unwrap();
        s.set_input("B", &b).unwrap();
        s.set_input("SEL", &Value::from_logic(sel)).unwrap();
        settle(&mut s);
        let y = s.read("Y").unwrap();
        for j in 0..3 {
            let e = or2(and2(a.get(j), not1(sel)), and2(b.get(j), sel));
            assert_eq!(y.get(j), e, "mux2 A={a} B={b} SEL={sel:?}");
        }
    }
    // equality: EQ = AND_j XNOR(A_j, B_j).
    let mut s = sim_of(|db| equality(db, 4));
    for _ in 0..300 {
        let (a, b) = (rand_val(&mut rng, 4), rand_val(&mut rng, 4));
        s.set_input("A", &a).unwrap();
        s.set_input("B", &b).unwrap();
        settle(&mut s);
        let e = (0..4).fold(Logic::One, |acc, j| and2(acc, not1(xor2(a.get(j), b.get(j)))));
        assert_eq!(s.read("EQ").unwrap().get(0), e, "eq A={a} B={b}");
    }
    // decoder: Y_i = EN & AND_k (A_k or !A_k).
    let mut s = sim_of(|db| decoder(db, 3));
    for _ in 0..300 {
        let (a, en) = (rand_val(&mut rng, 3), rand4(&mut rng));
        s.set_input("A", &a).unwrap();
        s.set_input("EN", &Value::from_logic(en)).unwrap();
        settle(&mut s);
        let y = s.read("Y").unwrap();
        for i in 0..8u32 {
            let e = (0..3).fold(en, |acc, k| {
                let bit = a.get(k);
                and2(acc, if (i >> k) & 1 == 1 { bit.normalize() } else { not1(bit) })
            });
            assert_eq!(y.get(i), e, "dec A={a} EN={en:?} i={i}");
        }
    }
    // full adder: S = A^B^C, COUT = A·B + C·(A^B).
    let mut s = sim_of(full_adder);
    for _ in 0..300 {
        let (a, b, c) = (rand4(&mut rng), rand4(&mut rng), rand4(&mut rng));
        for (p, v) in [("A", a), ("B", b), ("CIN", c)] {
            s.set_input(p, &Value::from_logic(v)).unwrap();
        }
        settle(&mut s);
        let p = xor2(a, b);
        assert_eq!(s.read("S").unwrap().get(0), xor2(p, c));
        assert_eq!(s.read("COUT").unwrap().get(0), or2(and2(a, b), and2(p, c)));
    }
}
