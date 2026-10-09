//! Package-from-selection correctness: the packaged design simulates exactly like the original
//! (every original flat net, under its new name, tick by tick), the kernel accepts the edit
//! ops, undo restores the original, and the geometry is complete.

use std::collections::HashSet;

use panda_kernel::{Kernel, KernelOptions, Mode, StepUnit};
use panda_netlist::{Design, Dir, InstKind, Value, flatten};

use super::change::DocChange;
use super::package::{PackagePlan, geometry_complete, plan_package};
use crate::doc::Document;
use crate::examples;

/// Original flat name → name after packaging.
fn map_name(name: &str, plan: &PackagePlan, d_paths: &[String]) -> String {
    let sel: HashSet<&str> = plan
        .package
        .instances
        .iter()
        .map(|i| i.id.as_str())
        .collect();
    for p in d_paths {
        let prefix = if p.is_empty() {
            String::new()
        } else {
            format!("{p}/")
        };
        let Some(rest) = name.strip_prefix(&prefix) else {
            continue;
        };
        if let Some((first, _)) = rest.split_once('/') {
            if sel.contains(first) {
                return format!("{prefix}{}/{rest}", plan.instance);
            }
        } else {
            let base = rest.split('[').next().unwrap_or(rest);
            if plan.moved_nets.iter().any(|m| m == base) {
                return format!("{prefix}{}/{rest}", plan.instance);
            }
        }
    }
    name.to_string()
}

fn kernel(d: Design, mode: Mode) -> Kernel {
    let mut k = Kernel::new(KernelOptions {
        background: false,
        ..KernelOptions::default()
    });
    k.load(d).unwrap();
    k.set_mode(mode);
    k
}

/// Package `sel` of def `def` in `design`, then simulate both for `ticks` with pseudo-random
/// inputs, comparing every original net every `every` ticks. Returns the plan.
fn check(design: Design, def: &str, sel: &[String], ticks: u64, every: u64) -> PackagePlan {
    let mut doc = Document::new(design.clone());
    doc.layout_missing();
    let plan = plan_package(&doc, def, sel, "pkg_t").unwrap_or_else(|e| panic!("{def}: {e}"));
    assert_eq!(plan.package.instances.len(), sel.len());
    // Document path (what the UI does) …
    let change: DocChange = (&plan).into();
    let mut doc2 = doc.clone();
    change.apply_doc(&mut doc2).unwrap();
    assert!(geometry_complete(
        &plan.package,
        doc2.editor("pkg_t").unwrap()
    ));
    assert!(doc2.editor(def).unwrap().covers(doc2.def(def).unwrap()));
    assert!(doc2.defs_missing_layout().is_empty());
    // … and the kernel path (what the sim thread does) agree.
    let orig_flat = flatten(&design).unwrap();
    let mut a = kernel(design.clone(), Mode::Timing);
    let mut b = kernel(design.clone(), Mode::Timing);
    b.edit(&plan.ops())
        .unwrap_or_else(|e| panic!("kernel rejected: {e:?}"));
    assert_eq!(b.design().unwrap().defs.len(), doc2.design.defs.len());
    let d_paths: Vec<String> = orig_flat
        .instances
        .iter()
        .filter(|i| i.def == def)
        .map(|i| i.path.clone())
        .collect();
    assert!(!d_paths.is_empty());
    let mut names: Vec<(String, String)> = orig_flat
        .name_index
        .keys()
        .map(|n| (n.clone(), map_name(n, &plan, &d_paths)))
        .collect();
    names.sort();
    let new_flat = b.netlist().unwrap().clone();
    for (o, n) in &names {
        assert!(new_flat.net_by_name(n).is_some(), "{o} → {n} missing");
        assert!(a.read(o).is_some() && b.read(n).is_some(), "{o} unreadable");
    }
    let renamed = names.iter().filter(|(o, n)| o != n).count();
    let inputs: Vec<(String, u32)> = design
        .top_def()
        .unwrap()
        .pins
        .iter()
        .filter(|p| p.dir == Dir::In)
        .map(|p| (p.name.clone(), p.width))
        .collect();
    let mut rng = 0x2545_f491_4f6c_dd1du64;
    let mut t = 0;
    let mut compared = 0usize;
    while t < ticks {
        if t % 23 == 0 {
            for (pin, w) in &inputs {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let v = Value::from_u64(*w, rng & ((1u64 << (*w).min(63)) - 1));
                a.set_input(pin, &v).unwrap();
                b.set_input(pin, &v).unwrap();
            }
        }
        a.step(every, StepUnit::Tick);
        b.step(every, StepUnit::Tick);
        t += every;
        assert_eq!(a.tick(), b.tick());
        for (o, n) in &names {
            assert_eq!(a.read(o), b.read(n), "tick {t}: {o} vs {n}");
            compared += 1;
        }
    }
    assert!(compared > 0);
    eprintln!(
        "{def}: {} selected, {} pins, {} moved nets, {} flat names ({renamed} renamed), \
         {compared} comparisons over {ticks} ticks",
        sel.len(),
        plan.pins.len(),
        plan.moved_nets.len(),
        names.len()
    );
    // Undo restores the original exactly.
    change.revert_doc(&mut doc2).unwrap();
    assert_eq!(doc2.design, doc.design);
    assert_eq!(doc2.editors, doc.editors);
    b.edit(&plan.undo_ops()).unwrap();
    assert_eq!(b.design().unwrap(), &design);
    plan
}

fn ids(design: &Design, def: &str, pick: impl Fn(usize, &str) -> bool) -> Vec<String> {
    design
        .def(def)
        .unwrap()
        .instances
        .iter()
        .enumerate()
        .filter(|(k, i)| pick(*k, &i.id))
        .map(|(_, i)| i.id.clone())
        .collect()
}

#[test]
fn counter_incrementer_package() {
    let d = examples::counter();
    let sel = ids(&d, "sync_counter_4", |_, id| id != "reg");
    let plan = check(d, "sync_counter_4", &sel, 1200, 1);
    let pins: Vec<(&str, Dir, u32)> = plan
        .pins
        .iter()
        .map(|p| (p.decl.name.as_str(), p.decl.dir, p.decl.width))
        .collect();
    // `Q` (read by the incrementer, driven by the register) and `next` (driven inside).
    assert!(pins.contains(&("Q", Dir::In, 4)), "{pins:?}");
    assert!(pins.contains(&("next", Dir::Out, 4)), "{pins:?}");
    assert_eq!(pins.len(), 2);
}

#[test]
fn sr_latch_half_at_top_level() {
    // One gate of a cross-coupled pair: both feedback nets cross.
    let d = examples::sr_latch();
    let top = d.top.clone();
    let first = d.top_def().unwrap().instances[0].id.clone();
    let plan = check(d, &top, &[first], 400, 1);
    assert_eq!(plan.pins.len(), 3);
    assert_eq!(
        plan.pins.iter().filter(|p| p.decl.dir == Dir::Out).count(),
        1
    );
}

#[test]
fn register_file_slices() {
    let d = examples::register_file_4x8();
    let def = d
        .defs
        .iter()
        .find(|x| x.id.starts_with("regfile_"))
        .unwrap()
        .id
        .clone();
    // Decoder + two words, then every other instance (splitters included).
    for sel in [
        ids(&d, &def, |_, id| id == "wdec" || id == "w0" || id == "w1"),
        ids(&d, &def, |k, _| k % 2 == 0),
    ] {
        check(d.clone(), &def, &sel, 900, 3);
    }
}

#[test]
fn panda8_core_slices() {
    let d = examples::panda8();
    let core = d
        .defs
        .iter()
        .find(|x| x.id.contains("core"))
        .unwrap()
        .id
        .clone();
    let n = d.def(&core).unwrap().instances.len();
    assert!(n > 10);
    // Every third instance of the core: many crossing nets, splitters on both sides.
    let sel = ids(&d, &core, |k, _| k % 3 == 1);
    let plan = check(d.clone(), &core, &sel, 128 * 12, 16);
    assert!(plan.pins.len() >= 4);
    // A package of only def instances (sub-packages move along with their contents).
    let sel: Vec<String> = d
        .def(&core)
        .unwrap()
        .instances
        .iter()
        .filter(|i| matches!(i.kind, InstKind::Def(_)))
        .take(3)
        .map(|i| i.id.clone())
        .collect();
    check(d, &core, &sel, 128 * 8, 32);
}

#[test]
fn panda8_fast_mode_after_packaging() {
    // The JIT/compiled path sees the packaged design too: compare top outputs per cycle.
    let d = examples::panda8();
    let mut doc = Document::new(d.clone());
    doc.layout_missing();
    let core = d
        .defs
        .iter()
        .find(|x| x.id.contains("core"))
        .unwrap()
        .id
        .clone();
    let sel = ids(&d, &core, |k, _| k % 2 == 0);
    let plan = plan_package(&doc, &core, &sel, "half").unwrap();
    let mut a = kernel(d.clone(), Mode::Fast);
    let mut b = kernel(d.clone(), Mode::Fast);
    b.edit(&plan.ops()).unwrap();
    a.wait_ready().unwrap();
    b.wait_ready().unwrap();
    let outs: Vec<String> = d
        .top_def()
        .unwrap()
        .pins
        .iter()
        .filter(|p| p.dir == Dir::Out)
        .map(|p| p.name.clone())
        .collect();
    assert!(!outs.is_empty());
    for c in 0..300 {
        a.step(1, StepUnit::Cycle);
        b.step(1, StepUnit::Cycle);
        for o in &outs {
            assert_eq!(a.read(o), b.read(o), "cycle {c}: {o}");
        }
    }
}

#[test]
fn rejects_bad_selections() {
    let d = examples::counter();
    let mut doc = Document::new(d.clone());
    doc.layout_missing();
    let top = d.top.clone();
    assert!(plan_package(&doc, &top, &[], "p").is_err());
    assert!(plan_package(&doc, &top, &["nope".into()], "p").is_err());
    // The testbench's clock source must stay in the top def.
    let clk: Vec<String> = ids(&d, &top, |_, id| id != "dut");
    assert!(plan_package(&doc, &top, &clk, "p").is_err());
    assert!(plan_package(&doc, "sync_counter_4", &["inc".into()], "sync_counter_4").is_err());
    assert!(plan_package(&doc, "sync_counter_4", &["inc".into()], "a/b").is_err());
}
