//! Stuck-X diagnostic (SPEC §5.4) and X-source tracing.

use std::sync::Arc;

use panda_circuits::{half_adder, testbench, toggle_ff};
use panda_netlist::flat::FlatNetlist;
use panda_netlist::{DefBuilder, DefRef, DesignBuilder, DiagKind, Diagnostic, Logic, Termination, Value, flatten};
use panda_timing::{TimingOptions, TimingSim, XSourceKind};

fn flat(db: DesignBuilder, top: &DefRef) -> Arc<FlatNetlist> {
    Arc::new(flatten(&db.build(top)).unwrap_or_else(|e| panic!("flatten failed: {e:#?}")))
}

const PERIOD: u32 = 16;

/// A toggle FF with `T = 1`, `RST = 0` (never reset), clocked with period 16; outputs `Q`, `Q_n`
/// and `S = Q xor 1` through a half-adder package (an ideal-block candidate).
fn unreset_tff() -> Arc<FlatNetlist> {
    let mut db = DesignBuilder::new();
    let tff = toggle_ff(&mut db);
    let ha = half_adder(&mut db);
    let mut b = DefBuilder::new("top");
    let clk = b.clock(PERIOD, PERIOD / 2, PERIOD / 2);
    let one = b.lit(1, 1);
    let zero = b.lit(1, 0);
    let o = b.inst_named("tff", &tff, &[one, zero, clk]);
    let s = b.inst_named("ha", &ha, &[o[0], one]);
    b.output("Q", o[0]);
    b.output("Q_n", o[1]);
    b.output("S", s[0]);
    let top = db.add(b.finish());
    flat(db, &top)
}

fn stuck(d: &[Diagnostic]) -> Vec<&Diagnostic> {
    d.iter().filter(|d| d.kind == DiagKind::StuckX).collect()
}

#[test]
fn unreset_toggle_ff_reports_stuck_x_once_with_loop_origin() {
    for ideal in [false, true] {
        let nl = unreset_tff();
        let mut s = TimingSim::new(nl.clone(), TimingOptions::default().with_ideal_blocks(ideal));
        assert_eq!(s.ideal_blocks().len(), usize::from(ideal), "only the half adder is combinational");
        let check = 16 * u64::from(PERIOD);
        s.run(check - 1);
        assert!(stuck(&s.take_diagnostics()).is_empty(), "too early");
        s.step();
        let d = s.take_diagnostics();
        let st = stuck(&d);
        let locs: Vec<&str> = st.iter().map(|d| d.location.as_deref().unwrap()).collect();
        assert_eq!(locs, ["Q", "Q_n", "S"], "{d:?}");
        for d in &st {
            assert!(d.message.contains("tick 256"), "{}", d.message);
            assert!(d.message.contains("loop [tff/ff/"), "origin is the FF's latch loop: {}", d.message);
        }
        // Once per reset.
        s.run(50 * u64::from(PERIOD));
        assert!(stuck(&s.take_diagnostics()).is_empty());
        // The trace API agrees: every origin is a loop inside the flip-flop.
        let q = nl.net_by_name("S").unwrap();
        let src = s.x_sources(&[q]);
        assert!(!src.is_empty());
        for x in &src {
            assert_eq!(x.kind, XSourceKind::Loop);
            assert!(x.nets.iter().all(|&n| nl.net_names[n as usize].starts_with("tff/")), "{x:?}");
        }
        // Reset re-arms the check.
        s.reset();
        s.run(check);
        assert_eq!(stuck(&s.take_diagnostics()).len(), 3);
    }
}

#[test]
fn stuck_x_period_option() {
    let nl = unreset_tff();
    let mut s = TimingSim::new(nl.clone(), TimingOptions::default().with_stuck_x_periods(0));
    s.run(100 * u64::from(PERIOD));
    assert!(stuck(&s.take_diagnostics()).is_empty(), "disabled");
    let mut s = TimingSim::new(nl, TimingOptions::default().with_stuck_x_periods(3));
    s.run(3 * u64::from(PERIOD) - 1);
    assert!(stuck(&s.take_diagnostics()).is_empty());
    s.step();
    assert_eq!(stuck(&s.take_diagnostics()).len(), 3);
}

#[test]
fn reset_toggle_ff_is_not_stuck() {
    let mut db = DesignBuilder::new();
    let tff = toggle_ff(&mut db);
    let top = testbench(&mut db, &tff, PERIOD);
    let nl = flat(db, &top);
    let mut s = TimingSim::new(nl, TimingOptions::default());
    s.set_input("T", &Value::from_bool(true)).unwrap();
    s.run(40 * u64::from(PERIOD));
    assert!(s.read_pin("Q").unwrap().is_fully_known());
    assert!(stuck(&s.take_diagnostics()).is_empty());
}

#[test]
fn check_waits_for_power_on_reset_release() {
    // A reset held for 100 ticks: the check is at 100 + 16 periods.
    let mut db = DesignBuilder::new();
    let tff = toggle_ff(&mut db);
    let mut b = DefBuilder::new("top");
    let clk = b.clock(PERIOD, PERIOD / 2, PERIOD / 2);
    let por = b.power_on_reset(100);
    let one = b.lit(1, 1);
    // RST is ignored on purpose (tied to the POR only through an AND with 0): stays X.
    let zero = b.lit(1, 0);
    let rst = b.and(&[por, zero]);
    let o = b.inst_named("tff", &tff, &[one, rst, clk]);
    b.output("Q", o[0]);
    let top = db.add(b.finish());
    let nl = flat(db, &top);
    let mut s = TimingSim::new(nl, TimingOptions::default());
    s.run(100 + 16 * u64::from(PERIOD) - 1);
    assert!(stuck(&s.take_diagnostics()).is_empty());
    s.step();
    assert_eq!(stuck(&s.take_diagnostics()).len(), 1);
}

#[test]
fn no_clock_no_check() {
    let mut b = DefBuilder::new("top");
    let a = b.wire(1); // undriven: reads Z -> X at the gate
    let y = b.not(a);
    b.output("Y", y);
    let mut db = DesignBuilder::new();
    let top = db.add(b.finish());
    let nl = flat(db, &top);
    let mut s = TimingSim::new(nl.clone(), TimingOptions::default());
    s.run(100_000);
    assert!(s.take_diagnostics().is_empty());
    assert_eq!(s.read_pin("Y").unwrap().get(0), Logic::X);
    let src = s.x_sources(&[nl.net_by_name("Y").unwrap()]);
    assert_eq!(src.len(), 1);
    assert_eq!(src[0].kind, XSourceKind::Undriven);
}

#[test]
fn x_source_kinds() {
    // Y1 = NOT(constant X); Y2 = resolver of two TRIs driving 0 and 1 (contention); Y3 = AND(0, X)
    // is known (no trace).
    let mut b = DefBuilder::new("top");
    let cx = b.constant(Value::xs(1));
    let y1 = b.not(cx);
    let zero = b.lit(1, 0);
    let one = b.lit(1, 1);
    let bus = b.wire(1);
    b.tri_to(zero, one, bus);
    b.tri_to(one, one, bus);
    b.terminate(bus, Termination::None);
    let y2 = b.buf(bus);
    let y3 = b.and(&[zero, cx]);
    b.output("Y1", y1);
    b.output("Y2", y2);
    b.output("Y3", y3);
    let mut db = DesignBuilder::new();
    let top = db.add(b.finish());
    let nl = flat(db, &top);
    let mut s = TimingSim::new(nl.clone(), TimingOptions::default());
    assert!(s.settle(100));
    let net = |n: &str| nl.net_by_name(n).unwrap();
    let k = |n: &str| s.x_sources(&[net(n)]).iter().map(|x| x.kind).collect::<Vec<_>>();
    assert_eq!(k("Y1"), [XSourceKind::Source]);
    assert_eq!(k("Y2"), [XSourceKind::Contention]);
    assert!(k("Y3").is_empty(), "known nets have no X origin");
    let both = s.x_sources(&[net("Y1"), net("Y2")]);
    assert_eq!(both.len(), 2);
}
