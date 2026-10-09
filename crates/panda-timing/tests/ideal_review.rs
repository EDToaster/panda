//! Ideal blocks (SPEC §5.2) at the edges: transport delay longer than the period-2 collapse
//! window, collapse of block outputs with changes still in flight, loops through several
//! packages, lazy internal reads, and `reset()` with pending output changes. Balanced blocks (every
//! input→output path equally long) must behave tick-exactly like their gates.

use std::sync::Arc;

use panda_netlist::flat::FlatNetlist;
use panda_netlist::{DefBuilder, DefRef, DesignBuilder, DiagKind, Logic, Value, flatten};
use panda_timing::{TimingOptions, TimingSim};

fn flat(db: DesignBuilder, top: &DefRef) -> Arc<FlatNetlist> {
    Arc::new(flatten(&db.build(top)).unwrap_or_else(|e| panic!("flatten failed: {e:#?}")))
}

/// A package of `n` buffers in a chain (`A` → `Y`, latency `n`, balanced).
fn delay_pkg(db: &mut DesignBuilder, n: usize) -> DefRef {
    let mut b = DefBuilder::new(&format!("delay{n}"));
    let mut s = b.input("A", 1);
    for _ in 0..n {
        s = b.buf(s);
    }
    b.output("Y", s);
    db.add_once(b.finish())
}

/// Step both simulators `ticks` times, asserting every net outside ideal blocks is equal.
fn lockstep(id: &mut TimingSim, gb: &mut TimingSim, ticks: u64, what: &str) {
    for _ in 0..ticks {
        id.step();
        gb.step();
        let nl = id.netlist().clone();
        for n in 0..nl.num_nets {
            if id.is_ideal_internal(n) {
                continue;
            }
            assert_eq!(
                id.get(n),
                gb.get(n),
                "{what}: net {} at tick {}",
                nl.net_names[n as usize],
                id.tick()
            );
        }
    }
}

fn metastable_count(s: &mut TimingSim) -> usize {
    s.take_diagnostics()
        .iter()
        .filter(|d| d.kind == DiagKind::Metastable)
        .count()
}

/// A period-2 glitch train (the parity of ten staggered taps of `A`) passes through a balanced
/// 12-tick package. The train is not self-sustained, so the gates never collapse it; the ideal
/// block's output must not collapse either, although its inputs stopped changing more than
/// `2 · p2_limit` ticks before its output stops alternating.
#[test]
fn glitch_train_longer_than_the_collapse_window_is_not_collapsed() {
    let mut db = DesignBuilder::new();
    let d12 = delay_pkg(&mut db, 12);
    let mut b = DefBuilder::new("train_top");
    let a = b.input("A", 1);
    let mut taps = vec![a];
    let mut s = a;
    for _ in 0..9 {
        s = b.buf(s);
        taps.push(s);
    }
    let train = b.xor(&taps);
    b.name(train, "train");
    let y = b.inst_named("dly", &d12, &[train])[0];
    b.output("Y", y);
    let top = db.add(b.finish());
    let nl = flat(db, &top);
    let mut id = TimingSim::new(nl.clone(), TimingOptions::default().with_ideal_blocks(true));
    let mut gb = TimingSim::new(nl.clone(), TimingOptions::default());
    assert_eq!(id.ideal_blocks().len(), 1);
    id.reset();
    gb.reset();
    lockstep(&mut id, &mut gb, 40, "settle");
    for v in [true, false, true] {
        id.set_input("A", &Value::from_bool(v)).unwrap();
        gb.set_input("A", &Value::from_bool(v)).unwrap();
        lockstep(&mut id, &mut gb, 40, "train");
    }
    assert_eq!(metastable_count(&mut gb), 0, "gates: not self-sustained");
    assert_eq!(metastable_count(&mut id), 0, "ideal block: not self-sustained either");
}

/// A metastable NAND latch (released 00 → 11) drives a balanced 3-tick package whose output sets
/// a second NAND latch. When the first latch collapses to X, the gates of the package collapse
/// with it (every buffer alternates), so the second latch sees X; the ideal block must not leak
/// the binary values still in flight.
#[test]
fn collapsed_block_output_drops_changes_in_flight() {
    let mut db = DesignBuilder::new();
    let d3 = delay_pkg(&mut db, 3);
    let mut b = DefBuilder::new("meta_top");
    let rel = b.input("REL", 1);
    let rst_n = b.input("RST_N", 1);
    // Metastable latch: S_n = R_n = REL.
    let qn = b.wire(1);
    let q = b.nand(&[rel, qn]);
    b.nand_to(&[rel, q], qn);
    b.name(q, "q");
    let y = b.inst_named("dly", &d3, &[q])[0];
    b.name(y, "y");
    // Second latch: set by y = 0 (active low), reset by RST_N.
    let qn2 = b.wire(1);
    let q2 = b.nand(&[y, qn2]);
    b.nand_to(&[rst_n, q2], qn2);
    b.output("Q2", q2);
    b.output("Q", q);
    let top = db.add(b.finish());
    let nl = flat(db, &top);
    let mut id = TimingSim::new(nl.clone(), TimingOptions::default().with_ideal_blocks(true));
    let mut gb = TimingSim::new(nl.clone(), TimingOptions::default());
    let set = |s: &mut TimingSim, pin: &str, v: bool| s.set_input(pin, &Value::from_bool(v)).unwrap();
    for s in [&mut id, &mut gb] {
        set(s, "REL", false);
        set(s, "RST_N", false);
        s.reset();
    }
    lockstep(&mut id, &mut gb, 20, "reset");
    for s in [&mut id, &mut gb] {
        set(s, "RST_N", true);
    }
    lockstep(&mut id, &mut gb, 20, "hold");
    // Release: period-2 oscillation of q, qn, and (3 ticks later) y; collapse after 2·p2_limit.
    for s in [&mut id, &mut gb] {
        set(s, "REL", true);
    }
    lockstep(&mut id, &mut gb, 60, "release");
    assert!(metastable_count(&mut gb) > 0);
    assert!(metastable_count(&mut id) > 0);
    assert_eq!(gb.read_pin("Q").unwrap().get(0), Logic::X);
}

/// A NAND latch whose two NANDs are separate packages, each followed by `n` buffers inside the
/// package (latency `n + 1`): loops through two ideal blocks behave like the gates, including the
/// metastable release (period `2 (n + 1)`, never collapsed for `n ≥ 1`).
#[test]
fn latch_through_two_packages_with_latency() {
    for n in [0usize, 1, 3] {
        let mut db = DesignBuilder::new();
        let mut pk = DefBuilder::new(&format!("nandbuf{n}"));
        let a = pk.input("A", 1);
        let bb = pk.input("B", 1);
        let mut o = pk.nand(&[a, bb]);
        for _ in 0..n {
            o = pk.buf(o);
        }
        pk.output("Y", o);
        let nb = db.add(pk.finish());
        let mut b = DefBuilder::new("pkg_latch");
        let s_n = b.input("S_n", 1);
        let r_n = b.input("R_n", 1);
        let qn = b.wire(1);
        let q = b.inst_named("n1", &nb, &[s_n, qn])[0];
        let qn2 = b.inst_named("n2", &nb, &[r_n, q])[0];
        b.connect(qn2, qn);
        b.output("Q", q);
        b.output("QN", qn2);
        let top = db.add(b.finish());
        let nl = flat(db, &top);
        let mut id = TimingSim::new(nl.clone(), TimingOptions::default().with_ideal_blocks(true));
        let mut gb = TimingSim::new(nl.clone(), TimingOptions::default());
        assert_eq!(id.ideal_blocks().len(), 2, "n={n}");
        let seq: [(bool, bool, u64); 7] = [
            (true, true, 30),
            (false, true, 30),
            (true, true, 30),
            (true, false, 30),
            (false, false, 30),
            (true, true, 80), // simultaneous release
            (false, true, 30),
        ];
        for s in [&mut id, &mut gb] {
            s.reset();
        }
        for (k, &(sv, rv, ticks)) in seq.iter().enumerate() {
            for s in [&mut id, &mut gb] {
                s.set_input("S_n", &Value::from_bool(sv)).unwrap();
                s.set_input("R_n", &Value::from_bool(rv)).unwrap();
            }
            lockstep(&mut id, &mut gb, ticks, &format!("n={n} step {k}"));
        }
        assert_eq!(metastable_count(&mut id), metastable_count(&mut gb), "n={n}");
    }
}

/// `reset()` while block output changes are still in flight: nothing scheduled before the reset
/// may land after it.
#[test]
fn reset_discards_changes_in_flight() {
    let mut db = DesignBuilder::new();
    let d8 = delay_pkg(&mut db, 8);
    let mut b = DefBuilder::new("rst_top");
    let a = b.input("A", 1);
    let y = b.inst_named("dly", &d8, &[a])[0];
    b.output("Y", y);
    let top = db.add(b.finish());
    let nl = flat(db, &top);
    let mut id = TimingSim::new(nl.clone(), TimingOptions::default().with_ideal_blocks(true));
    let mut gb = TimingSim::new(nl.clone(), TimingOptions::default());
    for s in [&mut id, &mut gb] {
        s.reset();
    }
    lockstep(&mut id, &mut gb, 20, "settle");
    for v in [true, false, true] {
        for s in [&mut id, &mut gb] {
            s.set_input("A", &Value::from_bool(v)).unwrap();
        }
        lockstep(&mut id, &mut gb, 2, "toggle");
    }
    // Reset mid-flight (A = 1 is kept), then compare tick by tick again.
    for s in [&mut id, &mut gb] {
        s.reset();
    }
    lockstep(&mut id, &mut gb, 30, "after reset");
    assert_eq!(id.read_pin("Y").unwrap().get(0), Logic::One);
}

/// Internal nets of an ideal block read lazily (after LUT hits) equal the gate-level values once
/// the inputs are stable, and reading them never changes the simulation.
#[test]
fn lazy_internal_reads_do_not_perturb() {
    let mut db = DesignBuilder::new();
    let adder = panda_circuits::ripple_adder(&mut db, 4);
    let top = panda_circuits::testbench(&mut db, &adder, 4);
    let nl = flat(db, &top);
    let mut a = TimingSim::new(nl.clone(), TimingOptions::default().with_ideal_blocks(true));
    let mut b = TimingSim::new(nl.clone(), TimingOptions::default().with_ideal_blocks(true));
    let mut gb = TimingSim::new(nl.clone(), TimingOptions::default());
    let internal: Vec<u32> = a.ideal_blocks()[0].internal.clone();
    assert!(!internal.is_empty());
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    for s in [&mut a, &mut b, &mut gb] {
        s.reset();
    }
    for round in 0..200 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        for s in [&mut a, &mut b, &mut gb] {
            s.set_input("A", &Value::from_u64(4, x & 15)).unwrap();
            s.set_input("B", &Value::from_u64(4, (x >> 4) & 15)).unwrap();
            s.set_input("CIN", &Value::from_bool(x >> 8 & 1 == 1)).unwrap();
        }
        let dt = 1 + (x >> 9) % 12;
        for _ in 0..dt {
            a.step();
            b.step();
            gb.step();
            // `a` is probed every tick (lazy evaluation), `b` never.
            for &n in &internal {
                let _ = a.get(n);
            }
            assert_eq!(a.values(), b.values(), "round {round}");
        }
        if dt >= 12 {
            for &n in &internal {
                assert_eq!(a.get(n), gb.get(n), "internal net {} round {round}", nl.net_names[n as usize]);
            }
        }
    }
}

/// A self-sustained period-2 loop (`q = NAND(EN, q)`) drives a balanced 3-tick package; an
/// unrelated input toggling every other tick keeps the collapse rule off, so the loop, the
/// package's buffers and its output all alternate long enough to collapse *at the same tick* once
/// the toggling stops. The gates then hold X everywhere; the ideal block must not let the binary
/// values still in flight land after the collapse.
#[test]
fn simultaneous_collapse_of_block_output_and_inputs() {
    let mut db = DesignBuilder::new();
    let d3 = delay_pkg(&mut db, 3);
    let mut b = DefBuilder::new("loop_top");
    let en = b.input("EN", 1);
    let noise = b.input("NOISE", 1);
    let q = b.wire(1);
    b.nand_to(&[en, q], q);
    b.name(q, "q");
    let y = b.inst_named("dly", &d3, &[q])[0];
    b.output("Y", y);
    let nn = b.not(noise);
    b.output("NN", nn);
    let top = db.add(b.finish());
    let nl = flat(db, &top);
    let mut id = TimingSim::new(nl.clone(), TimingOptions::default().with_ideal_blocks(true));
    let mut gb = TimingSim::new(nl.clone(), TimingOptions::default());
    for s in [&mut id, &mut gb] {
        s.reset();
    }
    lockstep(&mut id, &mut gb, 5, "settle");
    for s in [&mut id, &mut gb] {
        s.set_input("EN", &Value::from_bool(true)).unwrap();
    }
    // Keep a source changing every other tick while the loop oscillates.
    for k in 0..20 {
        for s in [&mut id, &mut gb] {
            s.set_input("NOISE", &Value::from_bool(k % 2 == 0)).unwrap();
        }
        lockstep(&mut id, &mut gb, 2, "noise");
    }
    lockstep(&mut id, &mut gb, 40, "collapse");
    assert_eq!(gb.read_pin("Y").unwrap().get(0), Logic::X);
    assert!(metastable_count(&mut gb) > 0);
}
