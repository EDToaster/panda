//! Snapshots: round trips in both modes, across modes, and poking an inconsistent state.

mod common;

use common::*;
use panda_cpu8::programs;
use panda_cpu8::{RomStyle, Variant};
use panda_kernel::{Kernel, Mode, StepUnit};
use panda_netlist::Logic;

fn cpu_kernel(mode: Mode) -> (Kernel, Vec<panda_cpu8::emu::ArchState>, u64, u64) {
    let prog = programs::MULTIPLY.image();
    let (design, opts) = cpu_design(&prog, Variant::Hierarchical, RomStyle::SumOfProducts);
    let mut k = Kernel::new(quiet_opts());
    k.set_mode(mode);
    k.load(design).unwrap();
    k.wait_ready().unwrap();
    let c0 = cycle0_tick(&opts);
    k.run_to(c0);
    (k, trace(&prog, 200), c0, u64::from(opts.period))
}

fn round_trip(mode: Mode) {
    let (mut k, want, c0, p) = cpu_kernel(mode);
    k.step(50, StepUnit::Cycle);
    assert_eq!(arch(&k), want[50]);
    let snap = k.snapshot();
    assert_eq!(snap.tick(), c0 + 50 * p);
    assert_eq!(snap.mode(), mode);
    let mut first = Vec::new();
    for _ in 0..30 {
        k.step(1, StepUnit::Cycle);
        first.push(arch(&k));
    }
    k.restore(&snap).unwrap();
    assert_eq!(k.tick(), snap.tick());
    assert_eq!(arch(&k), want[50]);
    // Every net is restored.
    let names = k.netlist().unwrap().net_names.clone();
    for n in names.iter().step_by(7) {
        assert_eq!(k.read(n).map(|v| v.get(0)), snap.get(n), "net {n}");
    }
    let mut second = Vec::new();
    for _ in 0..30 {
        k.step(1, StepUnit::Cycle);
        second.push(arch(&k));
    }
    assert_eq!(first, second);
    assert_eq!(&second[..], &want[51..81]);
}

#[test]
fn round_trip_timing() {
    round_trip(Mode::Timing);
}

#[test]
fn round_trip_fast() {
    round_trip(Mode::Fast);
}

#[test]
fn restore_across_modes() {
    // Timing snapshot restored in Fast mode, and vice versa.
    let (mut t, want, _, _) = cpu_kernel(Mode::Timing);
    let (mut f, _, _, _) = cpu_kernel(Mode::Fast);
    t.step(60, StepUnit::Cycle);
    f.step(20, StepUnit::Cycle);
    let st = t.snapshot();
    let sf = f.snapshot();
    f.restore(&st).unwrap();
    t.restore(&sf).unwrap();
    assert_eq!(f.tick(), st.tick());
    assert_eq!(t.tick(), sf.tick());
    for c in 0..40 {
        f.step(1, StepUnit::Cycle);
        t.step(1, StepUnit::Cycle);
        assert_eq!(
            arch(&f),
            want[61 + c],
            "fast after restoring a timing snapshot"
        );
        assert_eq!(
            arch(&t),
            want[21 + c],
            "timing after restoring a fast snapshot"
        );
    }
}

#[test]
fn snapshot_mid_phase_timing() {
    // In Timing mode the values at any tick determine the future (unit-delay semantics).
    let (mut k, _, _, _) = cpu_kernel(Mode::Timing);
    k.step(30, StepUnit::Cycle);
    k.step(3, StepUnit::Tick); // in the middle of the rising-edge activity
    let s = k.snapshot();
    k.step(1, StepUnit::Phase);
    let names = k.netlist().unwrap().net_names.clone();
    let a: Vec<_> = names.iter().map(|n| k.read(n)).collect();
    k.restore(&s).unwrap();
    k.step(1, StepUnit::Phase);
    let b: Vec<_> = names.iter().map(|n| k.read(n)).collect();
    assert_eq!(a, b);
}

#[test]
fn inconsistent_state_finishes_its_transition_in_fast_mode() {
    let (mut k, _, _, _) = cpu_kernel(Mode::Fast);
    k.step(30, StepUnit::Cycle);
    let mut s = k.snapshot();
    // A slave latch core (cross-coupled NANDs q, qn). At the sampling point CLK = 0, so the slave
    // is opaque (s = r = 1); q = qn = 1 is not a fixed point.
    let flat = k.netlist().unwrap().clone();
    let mut names: Vec<_> = flat
        .name_index
        .keys()
        .filter(|n| n.ends_with("/slave/q"))
        .cloned()
        .collect();
    names.sort();
    let q = names[0].clone();
    let qn = format!("{q}n");
    assert!(s.set(&q, Logic::One) && s.set(&qn, Logic::One), "{q}");
    k.restore(&s).unwrap();
    // The kernel finishes the transition it is caught in with Timing semantics: the opaque NAND
    // pair oscillates with period 2 and collapses to X (it would restart from X only if it did
    // not settle at all).
    assert!(
        k.diagnostics()
            .iter()
            .any(|d| d.message.contains("inconsistent with their inputs")),
        "{:?}",
        k.diagnostics()
    );
    assert_eq!(k.read(&q).map(|x| x.get(0)), Some(Logic::X));
    assert_eq!(k.read(&qn).map(|x| x.get(0)), Some(Logic::X));
    // The same poke in Timing mode is simulated faithfully: the NAND pair resolves (period-2
    // collapse or a race), no fallback needed.
    k.set_mode(Mode::Timing);
    k.restore(&s).unwrap();
    k.step(1, StepUnit::Phase);
    let both_one = k.read(&q).map(|x| x.get(0)) == Some(Logic::One)
        && k.read(&qn).map(|x| x.get(0)) == Some(Logic::One);
    assert!(!both_one, "the poked pair evolved");
}

#[test]
fn pending_inputs_survive_snapshots() {
    use panda_circuits::{sync_counter, testbench};
    use panda_netlist::{DesignBuilder, Value};
    let mut db = DesignBuilder::new();
    let c = sync_counter(&mut db, 4);
    let tb = testbench(&mut db, &c, 16);
    let design = db.build(&tb);
    for mode in [Mode::Timing, Mode::Fast] {
        let mut k = Kernel::default();
        k.set_mode(mode);
        k.load(design.clone()).unwrap();
        k.wait_ready().unwrap();
        k.set_input("EN", &Value::from_bool(true)).unwrap();
        k.step(5, StepUnit::Cycle);
        k.set_input("EN", &Value::from_bool(false)).unwrap();
        k.step(2, StepUnit::Cycle);
        let q0 = k.read_u64("Q").unwrap();
        k.set_input("EN", &Value::from_bool(true)).unwrap(); // pending
        let s = k.snapshot();
        assert_eq!(s.pending_inputs().len(), 1, "{mode:?}");
        k.step(5, StepUnit::Cycle);
        let a = k.read_u64("Q").unwrap();
        // (Timing mode applies the input at the next tick, which is the rising edge itself: whether
        // that edge counts is up to the unit-delay race; Fast mode applies it after the edge.)
        if mode == Mode::Fast {
            assert_eq!(a, (q0 + 5) % 16);
        }
        k.restore(&s).unwrap();
        assert_eq!(k.read_u64("Q"), Some(q0));
        k.step(5, StepUnit::Cycle);
        assert_eq!(k.read_u64("Q"), Some(a), "{mode:?}");
        assert!(
            !k.diagnostics()
                .iter()
                .any(|d| d.message.contains("restarted from X")),
            "{:?}",
            k.diagnostics()
        );
        // Cross-mode restore of the same snapshot.
        let other = if mode == Mode::Timing {
            Mode::Fast
        } else {
            Mode::Timing
        };
        k.set_mode(other);
        k.wait_ready().unwrap();
        k.restore(&s).unwrap();
        k.step(5, StepUnit::Cycle);
        let q = k.read_u64("Q").unwrap();
        if other == Mode::Fast {
            assert_eq!(q, (q0 + 5) % 16, "{mode:?} snapshot restored in {other:?}");
        } else {
            assert!(
                q == (q0 + 5) % 16 || q == (q0 + 4) % 16,
                "{mode:?} snapshot restored in {other:?}: {q}"
            );
        }
        assert!(
            !k.diagnostics()
                .iter()
                .any(|d| d.message.contains("restarted from X")),
            "{:?}",
            k.diagnostics()
        );
    }
}
