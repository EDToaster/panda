//! Randomized soundness check of the STA contract: for random small sequential circuits (master-
//! slave flip-flops and latches on the clock, its inverse or gated clocks, random combinational
//! logic mixing state, reset and clock), whenever STA reports no error, FastRef must equal Timing
//! mode on every net at every phase boundary.

mod common;

use common::*;
use panda_circuits::{d_latch, ms_dff};
use panda_netlist::{DefBuilder, Design, DesignBuilder, GateOp, Sig};

/// xorshift64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn chance(&mut self, pct: u64) -> bool {
        self.next() % 100 < pct
    }
}

/// A random design and its clock period.
fn random_design(seed: u64) -> (Design, u32) {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut db = DesignBuilder::new();
    let ff = ms_dff(&mut db);
    let latch = d_latch(&mut db);
    let mut b = DefBuilder::new(&format!("rnd{seed}"));
    let period = 2 * (8 + rng.below(28)) as u32;
    let clk = b.clock(period, period / 2, period / 2);
    let rst = b.power_on_reset(2 * period + period / 4);
    let nrst = b.not(rst);
    let nstate = 2 + rng.below(5);
    let qs: Vec<Sig> = (0..nstate).map(|_| b.wire(1)).collect();
    // pool[0] = clk (rarely used as data), then rst, nrst, state outputs, gate outputs.
    let mut pool: Vec<Sig> = vec![clk, rst, nrst];
    pool.extend(qs.iter().copied());
    if rng.chance(30) {
        let c = b.lit(1, rng.below(2) as u64);
        pool.push(c);
    }
    let pick = |rng: &mut Rng, pool: &[Sig]| {
        if rng.chance(85) {
            pool[1 + rng.below(pool.len() - 1)]
        } else {
            pool[0]
        }
    };
    let ops = [
        GateOp::And,
        GateOp::Or,
        GateOp::Nand,
        GateOp::Nor,
        GateOp::Xor,
        GateOp::Xnor,
    ];
    for _ in 0..4 + rng.below(30) {
        let which = rng.below(ops.len() + 2);
        let g = if which >= ops.len() {
            let a = pick(&mut rng, &pool);
            if which == ops.len() {
                b.not(a)
            } else {
                b.buf(a)
            }
        } else {
            let ins: Vec<Sig> = (0..2 + rng.below(2))
                .map(|_| pick(&mut rng, &pool))
                .collect();
            b.gate(ops[which], &ins)
        };
        pool.push(g);
    }
    for (i, &q) in qs.iter().enumerate() {
        let d0 = pool[3 + rng.below(pool.len() - 3)];
        let d = b.and(&[d0, nrst]);
        let kind = rng.below(10);
        let ck = if kind < 6 {
            clk
        } else if kind < 8 {
            b.not(clk)
        } else {
            let en0 = pool[3 + rng.below(pool.len() - 3)];
            let en = b.or(&[en0, rst]);
            b.and(&[clk, en])
        };
        let out = if rng.chance(25) {
            b.inst_named(&format!("u_l{i}"), &latch, &[d, ck])[0]
        } else {
            b.inst_named(&format!("u_f{i}"), &ff, &[d, ck])[0]
        };
        b.connect(out, q);
        b.output(&format!("Q{i}"), q);
    }
    let top = db.add(b.finish());
    (db.build(&top), period)
}

/// A random design built from synchronous idioms: registers with muxed next-state logic, clock
/// gating through latch-based gating cells (and sometimes without), two-phase latch pipelines and
/// a second clock. Exercises path conditions (mux selects, gated clocks).
fn structured_design(seed: u64) -> (Design, u32) {
    let mut rng = Rng(seed.wrapping_mul(0xD1B5_4A32_D192_ED03) | 1);
    let mut db = DesignBuilder::new();
    let ff = ms_dff(&mut db);
    let latch = d_latch(&mut db);
    let mut b = DefBuilder::new(&format!("st{seed}"));
    let period = 2 * (10 + rng.below(30)) as u32;
    let clk = b.clock(period, period / 2, period / 2);
    let clk2 = if rng.chance(25) {
        let p2 = period * (2 + rng.below(2) as u32);
        Some(b.clock(p2, p2 / 2, (period / 4) * (1 + rng.below(2) as u32)))
    } else {
        None
    };
    let rst = b.power_on_reset(4 * period + period / 4 + 2);
    let nrst = b.not(rst);
    let nclk = b.not(clk);
    let nstate = 3 + rng.below(6);
    let qs: Vec<Sig> = (0..nstate).map(|_| b.wire(1)).collect();
    let mut pool: Vec<Sig> = qs.clone();
    for _ in 0..3 + rng.below(12) {
        let g = match rng.below(4) {
            0 => {
                // mux: sel ? a : c
                let (sel, a, c) = (
                    pool[rng.below(pool.len())],
                    pool[rng.below(pool.len())],
                    pool[rng.below(pool.len())],
                );
                let ns = b.not(sel);
                let x = b.and(&[a, sel]);
                let y = b.and(&[c, ns]);
                b.or(&[x, y])
            }
            1 => {
                let (a, c) = (pool[rng.below(pool.len())], pool[rng.below(pool.len())]);
                b.xor(&[a, c])
            }
            2 => {
                let (a, c) = (pool[rng.below(pool.len())], pool[rng.below(pool.len())]);
                b.nand(&[a, c])
            }
            _ => {
                let a = pool[rng.below(pool.len())];
                b.not(a)
            }
        };
        pool.push(g);
    }
    for (i, &q) in qs.iter().enumerate() {
        let d0 = pool[rng.below(pool.len())];
        let d = b.and(&[d0, nrst]);
        let out = match rng.below(10) {
            0..=4 => b.inst_named(&format!("u_f{i}"), &ff, &[d, clk])[0],
            5 => {
                // gated clock through a latch-based gating cell
                let en0 = pool[rng.below(pool.len())];
                let en = b.or(&[en0, rst]);
                let en_l = b.inst_named(&format!("u_icg{i}"), &latch, &[en, nclk])[0];
                let g = b.and(&[clk, en_l]);
                b.inst_named(&format!("u_g{i}"), &ff, &[d, g])[0]
            }
            6 => {
                // gated clock without a gating cell (often a hazard)
                let en0 = pool[rng.below(pool.len())];
                let en = b.or(&[en0, rst]);
                let g = b.and(&[clk, en]);
                b.inst_named(&format!("u_h{i}"), &ff, &[d, g])[0]
            }
            7 => {
                // two-phase latch pair
                let m = b.inst_named(&format!("u_lm{i}"), &latch, &[d, nclk])[0];
                b.inst_named(&format!("u_ls{i}"), &latch, &[m, clk])[0]
            }
            8 => b.inst_named(&format!("u_n{i}"), &ff, &[d, nclk])[0],
            _ => {
                let c = clk2.unwrap_or(clk);
                b.inst_named(&format!("u_c{i}"), &ff, &[d, c])[0]
            }
        };
        b.connect(out, q);
        b.output(&format!("Q{i}"), q);
    }
    let top = db.add(b.finish());
    (db.build(&top), period)
}

/// Returns (designs, clean, flagged with a real mismatch, unsound seeds).
fn run(seeds: std::ops::Range<u64>) -> (usize, usize, usize, Vec<(u64, Mismatch)>) {
    let (mut n, mut clean, mut confirmed) = (0, 0, 0);
    let mut unsound = Vec::new();
    for seed in seeds {
        let (d, period) = if seed % 2 == 0 {
            random_design(seed)
        } else {
            structured_design(seed)
        };
        let Ok(flat) = panda_netlist::flatten(&d) else {
            continue;
        };
        n += 1;
        let r = sta(&flat);
        let diff = differential(&flat, 160, 5 * u64::from(period) + 1);
        if r.is_clean() {
            clean += 1;
            if let Err(m) = diff {
                unsound.push((seed, m));
            }
        } else if diff.is_err() {
            confirmed += 1;
        }
    }
    (n, clean, confirmed, unsound)
}

#[test]
fn clean_random_designs_have_fastref_equal_timing() {
    let (n, clean, confirmed, unsound) = run(0..500);
    println!("{n} designs, {clean} clean, {confirmed} flagged with a confirmed mismatch");
    assert!(
        unsound.is_empty(),
        "STA clean but FastRef != Timing: {unsound:?}"
    );
    assert!(
        clean >= 20,
        "too few clean designs ({clean}) for the test to mean anything"
    );
    assert!(
        confirmed >= 20,
        "too few confirmed violations ({confirmed})"
    );
}

#[test]
#[ignore = "long: run with --ignored"]
fn clean_random_designs_have_fastref_equal_timing_long() {
    let (n, clean, confirmed, unsound) = run(500..8000);
    println!("{n} designs, {clean} clean, {confirmed} flagged with a confirmed mismatch");
    assert!(
        unsound.is_empty(),
        "STA clean but FastRef != Timing: {unsound:?}"
    );
}

#[test]
#[ignore = "diagnostic aid"]
fn pessimism_report() {
    let mut kinds = std::collections::BTreeMap::new();
    for seed in 0..300u64 {
        let (d, period) = random_design(seed);
        let Ok(flat) = panda_netlist::flatten(&d) else {
            continue;
        };
        let r = sta(&flat);
        if r.is_clean() || differential(&flat, 120, 3 * u64::from(period) + 1).is_err() {
            continue;
        }
        let mut ks: Vec<String> = r.errors().map(|e| format!("{:?}", e.kind)).collect();
        ks.sort();
        ks.dedup();
        let first = r.errors().next().unwrap();
        println!("seed {seed} period {period}: {ks:?} first: {first}");
        *kinds.entry(ks.join("+")).or_insert(0) += 1;
    }
    println!("{kinds:?}");
}
