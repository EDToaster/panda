//! Auto-layout invariants, readability metrics and speed on generated designs.
//!
//! * Every def of Panda-8, the flat Panda-8 and a 15k-instance def pass [`layout::check`] (no
//!   overlapping symbols, wires outside symbols, every connection routed, no wire shorts).
//! * Timing budgets (asserted in optimized builds): Panda-8's 22 defs < 50 ms, flat Panda-8
//!   < 300 ms, the 15k-instance def < 2 s.
//!
//! `cargo test --release -p panda-ui --test layout_quality -- --nocapture` prints the metrics.

use std::time::{Duration, Instant};

use panda_cpu8::{Options, RomStyle, Variant, panda8_with, programs};
use panda_netlist::{Design, Endpoint};
use panda_ui::doc::{Document, reachable_defs};
use panda_ui::layout::{self, Metrics};

/// Budgets only hold for optimized code.
fn optimized() -> bool {
    !cfg!(debug_assertions)
}

/// Lay out every def of `design`; returns the document and the layout time (best of `runs`).
fn laid_out(design: &Design, runs: usize) -> (Document, Duration) {
    let mut best = Duration::MAX;
    let mut out = None;
    for _ in 0..runs.max(1) {
        let mut doc = Document::new(design.clone());
        let t0 = Instant::now();
        doc.layout_missing();
        best = best.min(t0.elapsed());
        out = Some(doc);
    }
    (out.expect("one run"), best)
}

/// Check every reachable def and return the summed metrics.
fn check_all(doc: &Document) -> Metrics {
    let mut errs = Vec::new();
    let mut sum = Metrics::default();
    for id in reachable_defs(&doc.design) {
        errs.extend(layout::check(doc, &id));
        let m = layout::metrics(doc, &id);
        sum.width = sum.width.max(m.width);
        sum.height = sum.height.max(m.height);
        sum.wire_len += m.wire_len;
        sum.segments += m.segments;
        sum.bends += m.bends;
        sum.crossings += m.crossings;
        sum.overlaps += m.overlaps;
        sum.touches += m.touches;
    }
    assert!(
        errs.is_empty(),
        "{} problems, first: {:#?}",
        errs.len(),
        &errs[..errs.len().min(20)]
    );
    sum
}

fn report(name: &str, doc: &Document, dt: Duration) {
    eprintln!("== {name}: laid out in {dt:?}");
    for id in reachable_defs(&doc.design) {
        let def = doc.def(&id).expect("def");
        let m = layout::metrics(doc, &id);
        eprintln!(
            "  {id:<40} inst {:>5}  {:>4}x{:<5} wire {:>7}  bends {:>5}  cross {:>6}  shorts {}/{}",
            def.instances.len(),
            m.width,
            m.height,
            m.wire_len,
            m.bends,
            m.crossings,
            m.overlaps,
            m.touches
        );
    }
}

#[test]
fn panda8_hierarchical() {
    let design = panda_ui::examples::panda8();
    let (doc, dt) = laid_out(&design, 5);
    report("Panda-8", &doc, dt);
    let sum = check_all(&doc);
    eprintln!("  total: {sum:?}");
    assert_eq!(reachable_defs(&doc.design).len(), 22);
    if optimized() {
        assert!(dt < Duration::from_millis(50), "Panda-8 layout took {dt:?}");
    }
}

#[test]
fn panda8_flat() {
    let design = panda_cpu8::panda8(
        &programs::BENCH_LOOP.image(),
        panda_cpu8::Variant::Flat,
    );
    let (doc, dt) = laid_out(&design, 3);
    report("flat Panda-8", &doc, dt);
    check_all(&doc);
    if optimized() {
        assert!(dt < Duration::from_millis(300), "flat Panda-8 layout took {dt:?}");
    }
}

/// One def holding `copies` renamed copies of the flat Panda-8 (MuxTree ROM) body (drawing
/// only: it repeats the top-level sources).
fn big_design(copies: usize) -> Design {
    let flat = panda8_with(
        &programs::BENCH_LOOP.image(),
        &Options::new(Variant::Flat).with_rom(RomStyle::MuxTree),
    );
    let src = flat.top_def().expect("top").clone();
    let mut def = src.clone();
    def.instances.clear();
    def.nets.clear();
    def.id = "big".into();
    for c in 0..copies {
        let p = |s: &str| format!("c{c}_{s}");
        for i in &src.instances {
            let mut i = i.clone();
            i.id = p(&i.id);
            def.instances.push(i);
        }
        for n in &src.nets {
            let mut n = n.clone();
            n.id = p(&n.id);
            n.ends = n
                .ends
                .iter()
                .filter_map(|e| match e {
                    Endpoint::Pin(name) if c == 0 => Some(Endpoint::Pin(name.clone())),
                    Endpoint::Pin(_) => None,
                    Endpoint::Port { inst, port } => Some(Endpoint::Port {
                        inst: p(inst),
                        port: port.clone(),
                    }),
                })
                .collect();
            def.nets.push(n);
        }
    }
    Design {
        format: flat.format.clone(),
        defs: vec![def],
        top: "big".into(),
    }
}

#[test]
fn fifteen_k_instances() {
    let design = big_design(6);
    let n = design.top_def().expect("top").instances.len();
    assert!(n >= 15_000, "{n} instances");
    let (doc, dt) = laid_out(&design, 1);
    report("15k instances", &doc, dt);
    check_all(&doc);
    if optimized() {
        assert!(dt < Duration::from_secs(2), "15k-instance layout took {dt:?}");
    }
}
