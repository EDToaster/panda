//! Counters count (Timing mode).

mod common;

use common::*;
use panda_circuits::*;
use panda_netlist::DesignBuilder;

#[test]
fn sync_counter_counts_wraps_and_pauses() {
    let mut db = DesignBuilder::new();
    let c = sync_counter(&mut db, 4);
    assert_eq!(c.id, "sync_counter_4");
    let opts = TestbenchOptions::new(40);
    let tb = testbench_with(&mut db, &c, &opts);
    let mut s = sim_top(db, &tb);
    set(&mut s, "EN", 1);
    let k0 = opts.first_edge_after_reset();
    run_to(&mut s, opts.rising_edge(k0) - 11);
    assert_eq!(get(&s, "Q"), 0, "reset");
    let mut model = 0u64;
    for i in 0..40u64 {
        let en = u64::from(!(10..14).contains(&i) && i % 7 != 3);
        cycle(&mut s, &opts, k0 + i, &[("EN", en)]);
        model = (model + en) % 16;
        assert_eq!(get(&s, "Q"), model, "cycle {i} EN={en}");
    }
    assert_clean(&mut s);
}

#[test]
fn sync_counter_8_full_wrap() {
    let mut db = DesignBuilder::new();
    let c = sync_counter(&mut db, 8);
    let opts = TestbenchOptions::new(48);
    let tb = testbench_with(&mut db, &c, &opts);
    let mut s = sim_top(db, &tb);
    set(&mut s, "EN", 1);
    let k0 = opts.first_edge_after_reset();
    for i in 0..300u64 {
        cycle(&mut s, &opts, k0 + i, &[]);
        assert_eq!(get(&s, "Q"), (i + 1) % 256, "cycle {i}");
    }
    assert_clean(&mut s);
}

#[test]
fn ripple_counter_counts_and_resets() {
    let n = 5;
    let mut db = DesignBuilder::new();
    let c = ripple_counter(&mut db, n);
    // RST is a user input so it can be pulsed mid-run.
    let opts = TestbenchOptions { reset_pin: "none".into(), ..TestbenchOptions::new(40) };
    let tb = testbench_with(&mut db, &c, &opts);
    let mut s = sim_top(db, &tb);
    cycle(&mut s, &opts, 0, &[("RST", 1)]);
    assert_eq!(get(&s, "Q"), 0);
    let mut model = 0u64;
    for k in 1..80u64 {
        let rst = u64::from(k == 50);
        cycle(&mut s, &opts, k, &[("RST", rst)]);
        model = if rst == 1 { 0 } else { (model + 1) % (1 << n) };
        assert_eq!(get(&s, "Q"), model, "cycle {k}");
    }
    assert_clean(&mut s);
}

#[test]
fn ripple_counter_with_power_on_reset() {
    let mut db = DesignBuilder::new();
    let c = ripple_counter(&mut db, 3);
    let opts = TestbenchOptions::new(32);
    let tb = testbench_with(&mut db, &c, &opts);
    let mut s = sim_top(db, &tb);
    let k0 = opts.first_edge_after_reset();
    for i in 0..20u64 {
        cycle(&mut s, &opts, k0 + i, &[]);
        assert_eq!(get(&s, "Q"), (i + 1) % 8, "cycle {i}");
    }
    assert_clean(&mut s);
}
