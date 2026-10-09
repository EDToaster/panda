//! Package classification (SPEC §4.4) of panda-circuits blocks.

use panda_circuits::{
    d_latch, dff_rst, ms_dff, nand_sr_latch, register, ring_oscillator, ripple_adder, sync_counter,
    testbench, toggle_ff,
};
use panda_compile::KernelKind;
use panda_kernel::{ClassKind, Edge, Kernel};
use panda_netlist::{DefBuilder, DesignBuilder, Sig};

/// Cyclic but ternary-combinational: y1 = s ? a : y2, y2 = s ? y1 : b (the loop is broken for
/// either value of s).
fn cyclic_mux(db: &mut DesignBuilder) {
    let mut b = DefBuilder::new("cyclic_mux");
    let s = b.input("S", 1);
    let a = b.input("A", 1);
    let bb = b.input("B", 1);
    let ns = b.not(s);
    let y2: Sig = b.wire(1);
    let t1 = b.and(&[s, a]);
    let t2 = b.and(&[ns, y2]);
    let y1 = b.or(&[t1, t2]);
    let t3 = b.and(&[s, y1]);
    let t4 = b.and(&[ns, bb]);
    b.gate_to(panda_netlist::GateOp::Or, &[t3, t4], y2);
    b.output("Y1", y1);
    b.output("Y2", y2);
    db.add_once(b.finish());
}

fn kernel() -> Kernel {
    let mut db = DesignBuilder::new();
    let ff = ms_dff(&mut db);
    ripple_adder(&mut db, 8);
    ring_oscillator(&mut db, 3, false);
    ring_oscillator(&mut db, 5, true);
    d_latch(&mut db);
    nand_sr_latch(&mut db);
    sync_counter(&mut db, 3);
    register(&mut db, 2);
    register(&mut db, 4);
    dff_rst(&mut db);
    toggle_ff(&mut db);
    cyclic_mux(&mut db);
    let tb = testbench(&mut db, &ff, 16);
    let mut k = Kernel::default();
    k.load(db.build(&tb)).unwrap();
    k
}

#[test]
fn adder_is_combinational() {
    let mut k = kernel();
    let c = k.classify("ripple_adder_8").unwrap();
    assert_eq!(c.kind, ClassKind::Combinational);
    assert!(!c.cyclic);
    assert_eq!(c.state_bits, 0);
    assert!(c.kernels.is_empty());
    assert!(c.clock_like_pins.is_empty());
}

#[test]
fn cyclic_combinational() {
    let mut k = kernel();
    let c = k.classify("cyclic_mux").unwrap();
    assert!(c.cyclic, "{c:?}");
    assert_eq!(c.kind, ClassKind::Combinational, "{c:?}");
    assert!(c.kernels.iter().all(|k| k.ternary_combinational));
}

#[test]
fn ms_dff_is_sequential_with_clock_like_clk() {
    let mut k = kernel();
    let c = k.classify("ms_dff").unwrap();
    assert_eq!(c.kind, ClassKind::Sequential, "{c:?}");
    assert!(c.cyclic);
    assert_eq!(c.state_bits, 2, "master + slave latch: {c:?}");
    assert_eq!(c.kernels.len(), 2);
    assert!(
        c.kernels
            .iter()
            .all(|k| k.kind == KernelKind::Table && !k.oscillator)
    );
    assert!(c.clock_analysis);
    assert_eq!(c.clock_like_pins, vec!["CLK".to_string()]);
    assert_eq!(c.edge_triggered, Some(("CLK".to_string(), Edge::Rising)));
}

#[test]
fn latches_and_registers() {
    let mut k = kernel();
    // A transparent latch has no clock-like pin (EN = 1 passes D through).
    let c = k.classify("d_latch").unwrap();
    assert_eq!(c.kind, ClassKind::Sequential);
    assert_eq!(c.state_bits, 1);
    assert!(c.clock_analysis);
    assert!(c.clock_like_pins.is_empty(), "{c:?}");
    assert_eq!(c.edge_triggered, None);
    // NAND SR latch: sequential, not an oscillator (SPEC §4.4).
    let c = k.classify("nand_sr_latch").unwrap();
    assert_eq!(c.kind, ClassKind::Sequential, "{c:?}");
    assert_eq!(c.state_bits, 1);
    // Synchronous reset / enable: CLK is clock-like, the others are not.
    let c = k.classify("dff_rst").unwrap();
    assert_eq!(c.kind, ClassKind::Sequential);
    assert_eq!(c.clock_like_pins, vec!["CLK".to_string()], "{c:?}");
    assert_eq!(c.edge_triggered, Some(("CLK".to_string(), Edge::Rising)));
    let c = k.classify("sync_counter_3").unwrap();
    assert_eq!(c.kind, ClassKind::Sequential);
    assert_eq!(c.state_bits, 6, "3 flip-flops: {c:?}");
    assert_eq!(c.clock_like_pins, vec!["CLK".to_string()], "{c:?}");
    let c = k.classify("register_2").unwrap();
    assert_eq!(c.kind, ClassKind::Sequential);
    assert_eq!(c.clock_like_pins, vec!["CLK".to_string()], "{c:?}");
    // 7 input bits × 16 register values exceed the exploration bound: no clock-like analysis.
    let c = k.classify("register_4").unwrap();
    assert_eq!(c.kind, ClassKind::Sequential);
    assert_eq!(c.state_bits, 8);
    assert!(!c.clock_analysis && c.clock_like_pins.is_empty());
    let c = k.classify("toggle_ff").unwrap();
    assert_eq!(c.kind, ClassKind::Sequential, "{c:?}");
}

#[test]
fn ring_oscillators() {
    let mut k = kernel();
    for def in ["ring_osc_3", "ring_osc_en_5"] {
        let c = k.classify(def).unwrap();
        assert_eq!(c.kind, ClassKind::Oscillator, "{def}: {c:?}");
        assert!(c.kernels.iter().any(|k| k.oscillator));
    }
}

#[test]
fn errors() {
    let mut k = kernel();
    assert!(k.classify("nope").is_err());
    assert!(Kernel::default().classify("ms_dff").is_err());
    // The top (with Clock / PowerOnReset sources) classifies too; no clock-like exploration.
    let top = k.design().unwrap().top.clone();
    let c = k.classify(&top).unwrap();
    assert_eq!(c.kind, ClassKind::Sequential);
    assert!(!c.clock_analysis);
}
