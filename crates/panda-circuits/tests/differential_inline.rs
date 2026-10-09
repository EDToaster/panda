//! Differential check of flattening: for random hierarchical designs (random packages, random
//! nesting, splitters, wide buses, tri-state buses, inout pins, unconnected instance pins,
//! multi-driver nets, constants with X/Z bits, terminations, feedback loops), flattening the
//! design directly and flattening `inline_all(design)` must simulate identically in Timing mode:
//! every top-level output (and every internal net that exists under both naming schemes) must
//! match on every tick under random input stimulus.

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;

use common::Rng;
use common::random::{chance, rand_value, random_design};
use panda_circuits::transform::{PATH_SEP, try_inline_all};
use panda_netlist::flat::FlatNetlist;
use panda_netlist::validate::{has_errors, validate};
use panda_netlist::{InstKind, flatten};
use panda_timing::{TimingOptions, TimingSim};

fn structural_signature(f: &FlatNetlist) -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    for g in &f.gates {
        *m.entry(format!("{:?}/{}", g.op, g.inputs.len())).or_default() += 1;
    }
    for s in &f.sources {
        *m.entry(format!("{:?}", s.kind)).or_default() += 1;
    }
    for r in &f.resolvers {
        *m.entry(format!("res/{}/{:?}", r.drivers.len(), r.termination)).or_default() += 1;
    }
    // Terminated undriven nets that something reads (unread ones are unobservable; `inline_all`
    // drops nets without endpoints).
    let fo = f.fanout();
    let read = f.undriven_termination.keys().filter(|&&n| !fo.gates[n as usize].is_empty() || !fo.resolvers[n as usize].is_empty());
    *m.entry("undriven_term_read".into()).or_default() += read.count();
    m
}

/// Returns `false` if the design was rejected (by both paths consistently).
fn check_seed(seed: u64, ticks: u64) -> bool {
    let design = random_design(seed);
    let ctx = || format!("seed {seed}\n{}", design.to_json());
    let hier = flatten(&design);
    let inl = try_inline_all(&design);
    let hier = match hier {
        Ok(f) => f,
        Err(e) => {
            // The only expected rejection is a termination conflict across a pin boundary; the
            // inlined design must be rejected too.
            assert!(
                e.iter().all(|d| d.message.contains("termination")),
                "unexpected flatten error {e:#?}\n{}",
                ctx()
            );
            let rejected = match &inl {
                Err(_) => true,
                Ok(d) => flatten(d).is_err(),
            };
            assert!(rejected, "hierarchical flatten failed but the inlined design flattened\n{}", ctx());
            return false;
        }
    };
    let inl = inl.unwrap_or_else(|e| panic!("inline failed: {e}\n{}", ctx()));
    assert_eq!(inl.defs.len(), 1);
    assert!(inl.defs[0].instances.iter().all(|i| matches!(i.kind, InstKind::Atom(_))));
    let vd = validate(&inl);
    assert!(!has_errors(&vd), "inlined design invalid: {vd:#?}\n{}", ctx());
    let flat = flatten(&inl).unwrap_or_else(|e| panic!("inlined flatten failed: {e:#?}\n{}", ctx()));
    assert_eq!(structural_signature(&hier), structural_signature(&flat), "structure differs\n{}", ctx());

    let pairs: Vec<(u32, u32)> = flat
        .net_names
        .iter()
        .enumerate()
        .filter(|(_, n)| !n.contains('~'))
        .filter_map(|(i, n)| Some((hier.net_by_name(&n.replace(PATH_SEP, "/"))?, i as u32)))
        .collect();
    let ins: Vec<(String, u32)> = hier.top_inputs.iter().map(|p| (p.name.clone(), p.nets.len() as u32)).collect();
    let outs: Vec<String> = hier.top_outputs.iter().map(|p| p.name.clone()).collect();
    let mut a = TimingSim::new(Arc::new(hier), TimingOptions::default());
    let mut b = TimingSim::new(Arc::new(flat), TimingOptions::default());
    let mut rng = Rng::new(seed ^ 0x5eed);
    let mut diag_a: BTreeMap<String, usize> = BTreeMap::new();
    let mut diag_b: BTreeMap<String, usize> = BTreeMap::new();
    for t in 0..ticks {
        if chance(&mut rng, 25) {
            for (name, w) in &ins {
                if chance(&mut rng, 50) {
                    let v = rand_value(&mut rng, *w, 5);
                    a.set_input(name, &v).unwrap();
                    b.set_input(name, &v).unwrap();
                }
            }
        }
        for o in &outs {
            assert_eq!(a.read_pin(o), b.read_pin(o), "output {o} differs at tick {t}\n{}", ctx());
        }
        for &(x, y) in &pairs {
            assert_eq!(
                a.get(x),
                b.get(y),
                "net {} differs at tick {t}\n{}",
                a.netlist().net_names[x as usize],
                ctx()
            );
        }
        a.step();
        b.step();
        for d in a.take_diagnostics() {
            *diag_a.entry(format!("{:?}", d.kind)).or_default() += 1;
        }
        for d in b.take_diagnostics() {
            *diag_b.entry(format!("{:?}", d.kind)).or_default() += 1;
        }
    }
    // Contention/metastability are per-net, so their counts must agree too.
    assert_eq!(diag_a, diag_b, "diagnostics differ\n{}", ctx());
    true
}

#[test]
fn inline_vs_direct_flatten_random_designs() {
    let mut accepted = 0;
    for seed in 0..300 {
        if check_seed(seed, 120) {
            accepted += 1;
        }
    }
    assert!(accepted > 250, "too many rejected designs ({accepted} accepted)");
}

#[test]
#[ignore = "long-running fuzz; run with --ignored"]
fn inline_vs_direct_flatten_random_designs_long() {
    for seed in 1000..6000 {
        check_seed(seed, 300);
    }
}

