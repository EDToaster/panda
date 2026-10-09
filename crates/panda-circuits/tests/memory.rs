//! ROM contents and register-file write/read (Timing mode).

mod common;

use common::*;
use panda_circuits::*;
use panda_netlist::{DesignBuilder, Value};

#[test]
fn rom_reads_every_word() {
    let mut rng = Rng::new(5);
    let contents: Vec<u64> = (0..16).map(|_| rng.below(256)).collect();
    let mut db = DesignBuilder::new();
    let r = rom(&mut db, "test", 4, 8, &contents);
    let mut s = sim_top(db, &r);
    for (a, &want) in contents.iter().enumerate() {
        set(&mut s, "A", a as u64);
        settle(&mut s);
        assert_eq!(get(&s, "D"), want, "address {a}");
    }
    assert_clean(&mut s);
}

#[test]
fn rom_short_contents_read_zero_and_x_address() {
    let mut db = DesignBuilder::new();
    let r = rom(&mut db, "short", 3, 4, &[5, 5, 9]);
    let mut s = sim_top(db, &r);
    for a in 0..8u64 {
        set(&mut s, "A", a);
        settle(&mut s);
        assert_eq!(get(&s, "D"), [5, 5, 9, 0, 0, 0, 0, 0][a as usize], "address {a}");
    }
    // AND-OR mux pessimism: with an X select, a bit is known only where both candidates are 0.
    // Words 0 and 1 are both 0b0101.
    s.set_input("A", &Value::parse(3, "0b00x").unwrap()).unwrap();
    settle(&mut s);
    assert_eq!(s.read("D").unwrap(), Value::parse(4, "0b0x0x").unwrap());
    // Words 2 and 3 are 0b1001 and 0.
    s.set_input("A", &Value::parse(3, "0b01x").unwrap()).unwrap();
    settle(&mut s);
    assert_eq!(s.read("D").unwrap(), Value::parse(4, "0bx00x").unwrap());
    // Words 4..7 are all 0: known even with two X address bits.
    s.set_input("A", &Value::parse(3, "0b1xx").unwrap()).unwrap();
    settle(&mut s);
    assert_eq!(get(&s, "D"), 0);
}

#[test]
fn rom_from_hex_256_bytes() {
    let text: String =
        (0..256u32).map(|i| format!("{:02x}{}", (i * 37 + 11) % 256, if i % 16 == 15 { "\n" } else { " " })).collect();
    let text = format!("// program\n{text}");
    let contents = rom_from_hex(&text).unwrap();
    assert_eq!(contents.len(), 256);
    let mut db = DesignBuilder::new();
    let r = rom(&mut db, "prog", 8, 8, &contents);
    let mut s = sim_top(db, &r);
    for a in 0..256u64 {
        set(&mut s, "A", a);
        settle(&mut s);
        assert_eq!(get(&s, "D"), (a * 37 + 11) % 256, "address {a}");
    }
}

#[test]
fn rom_wide_words() {
    let contents = [u64::MAX, 0x0123_4567_89ab_cdef];
    let mut db = DesignBuilder::new();
    let r = rom(&mut db, "wide", 1, 64, &contents);
    let mut s = sim_top(db, &r);
    for (a, &w) in contents.iter().enumerate() {
        set(&mut s, "A", a as u64);
        settle(&mut s);
        assert_eq!(get(&s, "D"), w);
    }
}

fn check_regfile(words: usize, width: u32, style: RegFileStyle, seed: u64) {
    let a = addr_bits(words);
    let mut db = DesignBuilder::new();
    let rf = register_file(&mut db, words, width, style);
    let names: Vec<_> = rf.pins.iter().map(|p| (p.name.as_str(), p.width)).collect();
    assert_eq!(names, [("WD", width), ("WA", a), ("WE", 1), ("RA", a), ("RST", 1), ("CLK", 1), ("RD", width)]);
    let opts = TestbenchOptions::new(64);
    let tb = testbench_with(&mut db, &rf, &opts);
    let mut s = sim_top(db, &tb);
    let k0 = opts.first_edge_after_reset();
    let mut model = vec![0u64; words];
    let mut rng = Rng::new(seed);
    for i in 0..120u64 {
        let wa = rng.below(1 << a);
        let ra = rng.below(1 << a);
        let we = u64::from(rng.below(3) != 0);
        let wd = rng.below(1 << width);
        cycle(&mut s, &opts, k0 + i, &[("WA", wa), ("WE", we), ("WD", wd), ("RA", ra)]);
        if we == 1 && (wa as usize) < words {
            model[wa as usize] = wd;
        }
        let want = model.get(ra as usize).copied().unwrap_or(0);
        assert_eq!(get(&s, "RD"), want, "cycle {i}: RA={ra} (WA={wa} WE={we} WD={wd:#x})");
        for (w, &m) in model.iter().enumerate() {
            assert_eq!(get(&s, &format!("dut/r{w}")), m, "word {w} at cycle {i}");
        }
    }
    assert_clean(&mut s);
}

#[test]
fn regfile_enable_mux_4x8() {
    check_regfile(4, 8, RegFileStyle::EnableMux, 1);
}

#[test]
fn regfile_gated_clock_4x8() {
    check_regfile(4, 8, RegFileStyle::GatedClock, 2);
}

#[test]
fn regfile_non_power_of_two() {
    check_regfile(3, 4, RegFileStyle::EnableMux, 3);
    check_regfile(5, 3, RegFileStyle::GatedClock, 4);
}

#[test]
fn regfile_reset_clears_all_words() {
    let mut db = DesignBuilder::new();
    let rf = register_file(&mut db, 4, 4, RegFileStyle::EnableMux);
    let opts = TestbenchOptions::new(64);
    let tb = testbench_with(&mut db, &rf, &opts);
    let mut s = sim_top(db, &tb);
    // During reset, writes are ignored and every word reads 0.
    set(&mut s, "WE", 1);
    set(&mut s, "WD", 0xf);
    let k0 = opts.first_edge_after_reset();
    run_to(&mut s, opts.rising_edge(k0) - 17);
    for ra in 0..4 {
        assert_eq!(get(&s, &format!("dut/r{ra}")), 0);
    }
}
