//! `FastRef::load_state` (the facade's state loading: mode switches, edits, snapshots).
//!
//! * A settled state loaded into a fresh engine is consistent (no kernel finishes or falls back)
//!   and the loaded engine continues exactly like the original (random designs, Panda-8).
//! * Kernels whose loaded state is inconsistent with their inputs finish their transition with
//!   Timing-mode semantics instead of restarting from `X`.
//!
//! (FastSim's `load_state` is checked bit-exact against this one in panda-fast's suites.)

use std::sync::Arc;

use panda_compile::{FastOptions, FastRef, KernelOptions, LoadReport, compile_kernels};
use panda_difftest::{GenConfig, Rng, generate};
use panda_netlist::flat::FlatNetlist;
use panda_netlist::{DiagKind, Diagnostic, Logic, Value};

fn build(flat: &Arc<FlatNetlist>) -> FastRef {
    let opts = FastOptions::default();
    let kopts = KernelOptions {
        extract: opts.extract.clone(),
        charz: opts.charz.clone(),
    };
    let (ks, tables) = compile_kernels(flat, &kopts);
    FastRef::from_parts(flat.clone(), opts, ks, tables).unwrap()
}

fn kinds(d: Vec<Diagnostic>) -> Vec<DiagKind> {
    d.into_iter().map(|d| d.kind).collect()
}

/// Random input change on `rng` (returns the pin and value).
fn random_input(flat: &FlatNetlist, rng: &mut Rng) -> Option<(String, Value)> {
    if flat.top_inputs.is_empty() {
        return None;
    }
    let p = &flat.top_inputs[rng.below(flat.top_inputs.len())];
    let w = p.nets.len() as u32;
    let v = rng.next_u64() & ((1u64 << w.min(63)) - 1);
    Some((p.name.clone(), Value::from_u64(w, v)))
}

/// Run `a`; at a third of the run load its state into a fresh engine `b`, which must reproduce
/// it and then continue exactly like `a` (while no kernel went to `X` through the delta limit or
/// an oscillation, whose loaded state may legitimately differ).
fn check(flat: Arc<FlatNetlist>, seed: u64, events: usize) {
    let mut a = build(&flat);
    let mut rng = Rng::new(seed);
    let mut loaded: Option<FastRef> = None;
    let mut clean = true;
    for ev in 0..events {
        if rng.chance(20)
            && let Some((pin, v)) = random_input(&flat, &mut rng)
        {
            a.set_input(&pin, &v).unwrap();
            if let Some(b) = &mut loaded {
                b.set_input(&pin, &v).unwrap();
            }
        }
        let ta = a.step_event();
        let da = kinds(a.take_diagnostics());
        clean &= !da
            .iter()
            .any(|k| matches!(k, DiagKind::IterationLimit | DiagKind::Oscillating));
        if let Some(b) = &mut loaded {
            assert_eq!(b.step_event(), ta);
            assert_eq!(b.next_event_tick(), a.next_event_tick());
            if clean {
                assert_eq!(
                    b.values(),
                    a.values(),
                    "loaded copy diverged: seed {seed} event {ev}"
                );
            }
            b.take_diagnostics();
        }
        if ta.is_none() {
            break;
        }
        if ev == events / 3 {
            let mut b = build(&flat);
            let inputs: Vec<Value> = flat
                .top_inputs
                .iter()
                .map(|p| a.input_value(&p.name).unwrap().clone())
                .collect();
            let report = b.load_state(a.tick(), a.values(), &inputs);
            if clean {
                assert_eq!(
                    report,
                    LoadReport::default(),
                    "seed {seed}: settled states are consistent"
                );
                assert_eq!(
                    b.values(),
                    a.values(),
                    "seed {seed}: load_state reproduces the state"
                );
            }
            assert_eq!(b.tick(), a.tick());
            assert_eq!(b.next_event_tick(), a.next_event_tick());
            assert!(!b.has_pending_inputs());
            b.take_diagnostics();
            loaded = Some(b);
        }
    }
}

#[test]
fn reload_random_designs() {
    let cfg = GenConfig::default();
    for seed in 0..40u64 {
        let g = generate(seed, &cfg, 24);
        let flat = Arc::new(panda_netlist::flatten(&g.design).unwrap());
        check(flat, seed, 120);
    }
}

#[test]
fn reload_panda8() {
    use panda_cpu8::{Options, RomStyle, Variant, panda8_with, programs};
    let opts = Options::new(Variant::GatedClockRegfile).with_rom(RomStyle::MuxTree);
    let d = panda8_with(&programs::MULTIPLY.image(), &opts);
    let flat = Arc::new(panda_netlist::flatten(&d).unwrap());
    check(flat, 7, 500);
}

#[test]
fn inconsistent_kernels_finish_their_transition() {
    let mut db = panda_netlist::DesignBuilder::new();
    let l = panda_circuits::d_latch(&mut db);
    let mut b = panda_netlist::DefBuilder::new("top");
    let d = b.input("D", 1);
    let en = b.input("EN", 1);
    let o = b.inst_named("l", &l, &[d, en]);
    b.output("Q", o[0]);
    b.output("QN", o[1]);
    let top = db.add(b.finish());
    let flat = Arc::new(panda_netlist::flatten(&db.build(&top)).unwrap());
    let mut e = build(&flat);
    e.set_input("D", &Value::from_bool(true)).unwrap();
    e.set_input("EN", &Value::from_bool(true)).unwrap();
    e.step_event();
    e.set_input("EN", &Value::from_bool(false)).unwrap();
    e.step_event();
    assert_eq!(e.read("Q").unwrap().to_u64(), Some(1));
    let inputs = vec![Value::from_bool(true), Value::from_bool(false)];
    // Consistent: kept.
    let vals = e.values().to_vec();
    assert_eq!(e.load_state(5, &vals, &inputs), LoadReport::default());
    assert_eq!(e.values(), &vals[..]);
    assert_eq!(e.tick(), 5);
    // Q = QN = 1 while opaque: inconsistent; finishing the transition (the NAND pair
    // oscillates with period 2) collapses to X, as Timing mode would (and stays X: opaque).
    let mut bad = vals.clone();
    let q = flat.net_by_name("l/q").unwrap();
    let qn = flat.net_by_name("l/qn").unwrap();
    bad[q as usize] = Logic::One;
    bad[qn as usize] = Logic::One;
    assert_eq!(
        e.load_state(5, &bad, &inputs),
        LoadReport {
            inconsistent: 1,
            fallen: 0
        }
    );
    assert_eq!(e.get(q), Logic::X);
    // Same while transparent: the transition finishes at the input's value.
    let inputs = vec![Value::from_bool(false), Value::from_bool(true)];
    assert_eq!(e.load_state(6, &bad, &inputs).inconsistent, 1);
    assert_eq!(e.read("Q").unwrap().to_u64(), Some(0));
    assert_eq!(e.read("QN").unwrap().to_u64(), Some(1));
}
