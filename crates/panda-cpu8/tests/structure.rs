//! Structural properties of the generated designs: purity, pins, hierarchy, gate counts.

use panda_cpu8::generate::Variant;
use panda_cpu8::programs;
use panda_cpu8::{Options, RomStyle, panda8, panda8_with};
use panda_netlist::validate::validate;
use panda_netlist::{Atom, Design, Dir, InstKind, flatten};

const ROMS: [RomStyle; 3] = [RomStyle::SumOfProducts, RomStyle::MuxTree, RomStyle::Decoder];

fn designs() -> Vec<(String, Design)> {
    let mut v = Vec::new();
    for p in programs::ALL {
        for variant in Variant::ALL {
            for rom in ROMS {
                let d = panda8_with(&p.image(), &Options::new(variant).with_rom(rom));
                v.push((format!("{} {variant:?} {rom:?}", p.name), d));
            }
        }
    }
    v
}

#[test]
fn designs_validate_without_any_diagnostic() {
    for (name, d) in designs() {
        let diags = validate(&d);
        assert!(diags.is_empty(), "{name}: {diags:#?}");
        flatten(&d).unwrap_or_else(|e| panic!("{name}: {e:#?}"));
    }
}

#[test]
fn only_gates_splitters_constants_and_top_level_sources() {
    for (name, d) in designs() {
        let mut clocks = 0;
        let mut resets = 0;
        for def in &d.defs {
            let is_top = def.id == d.top;
            for inst in &def.instances {
                match &inst.kind {
                    InstKind::Def(_) => {}
                    InstKind::Atom(a) => match a {
                        Atom::Gate { .. } | Atom::Not { .. } | Atom::Splitter { .. } | Atom::Constant { .. } => {}
                        Atom::Clock { .. } if is_top => clocks += 1,
                        Atom::PowerOnReset { .. } if is_top => resets += 1,
                        other => panic!("{name}: atom {other:?} in def {}", def.id),
                    },
                }
            }
            for net in &def.nets {
                assert_eq!(net.termination, panda_netlist::Termination::None, "{name}: terminated net");
            }
        }
        assert_eq!((clocks, resets), (1, 1), "{name}: one Clock and one PowerOnReset");
        // No user inputs: the design is closed.
        let top = d.top_def().unwrap();
        assert!(top.pins.iter().all(|p| p.dir == Dir::Out), "{name}: top has non-output pins");
    }
}

#[test]
fn top_level_pins() {
    let d = panda8(&programs::COUNTDOWN.image(), Variant::Hierarchical);
    let pins: Vec<(String, u32)> = d.top_def().unwrap().pins.iter().map(|p| (p.name.clone(), p.width)).collect();
    let want = [
        ("OUT", 8),
        ("OUT_STROBE", 1),
        ("HALTED", 1),
        ("PC", 8),
        ("IR", 8),
        ("PHASE", 1),
        ("FLAGS", 2),
        ("R0", 8),
        ("R1", 8),
        ("R2", 8),
        ("R3", 8),
    ];
    let want: Vec<(String, u32)> = want.iter().map(|(n, w)| (n.to_string(), *w)).collect();
    assert_eq!(pins, want);
    // The flat variant keeps the same top interface.
    let f = panda8(&programs::COUNTDOWN.image(), Variant::Flat);
    assert_eq!(f.top_def().unwrap().pins, d.top_def().unwrap().pins);
    assert_eq!(f.top, d.top);
}

#[test]
fn hierarchy_and_flat_structure() {
    let p = programs::FIBONACCI.image();
    let h = panda8(&p, Variant::Hierarchical);
    let ids: Vec<&str> = h.defs.iter().map(|d| d.id.as_str()).collect();
    for want in ["panda8", "panda8_core", "panda8_alu", "panda8_control", "panda8_regfile_4x8", "ripple_adder_8"] {
        assert!(ids.contains(&want), "missing def {want}: {ids:?}");
    }
    assert!(!ids.contains(&"register_gated_8"), "enable-mux design has no gated registers");
    let g = panda8(&p, Variant::GatedClockRegfile);
    let ids: Vec<&str> = g.defs.iter().map(|d| d.id.as_str()).collect();
    for want in ["panda8_core_gated", "panda8_regfile_gated_4x8", "register_gated_8"] {
        assert!(ids.contains(&want), "missing def {want}: {ids:?}");
    }
    let f = panda8(&p, Variant::Flat);
    assert_eq!(f.defs.len(), 1);
    assert!(f.defs[0].instances.iter().all(|i| matches!(i.kind, InstKind::Atom(_))));
}

#[test]
fn core_is_program_independent() {
    let a = panda8(&programs::FIBONACCI.image(), Variant::Hierarchical);
    let b = panda8(&programs::MULTIPLY.image(), Variant::Hierarchical);
    for id in ["panda8_core", "panda8_alu", "panda8_control", "panda8_regfile_4x8"] {
        assert_eq!(a.def(id), b.def(id), "{id}");
    }
    // Only the ROM (content-addressed id) differs.
    let rom_ids = |d: &Design| -> Vec<String> {
        d.defs.iter().filter(|x| x.id.starts_with("panda8_sop_rom_")).map(|x| x.id.clone()).collect()
    };
    assert_eq!(rom_ids(&a).len(), 1);
    assert_ne!(rom_ids(&a), rom_ids(&b));
}

#[test]
fn json_roundtrip() {
    for v in Variant::ALL {
        let d = panda8(&programs::MULTIPLY.image(), v);
        assert_eq!(Design::from_json(&d.to_json()).unwrap(), d);
    }
}

fn gates(d: &Design) -> usize {
    flatten(d).unwrap().gates.len()
}

/// Flat gate counts (documented on [`RomStyle`]); printed for the record.
#[test]
fn gate_counts() {
    let p = programs::MULTIPLY.image();
    let count = |v, rom| gates(&panda8_with(&p, &Options::new(v).with_rom(rom)));
    let empty = |v| gates(&panda8_with(&[], &Options::new(v).with_rom(RomStyle::Decoder)));
    for v in Variant::ALL {
        for rom in ROMS {
            println!("{v:?} {rom:?}: {} gates", count(v, rom));
        }
    }
    // Core alone (decoder ROM of an all-zero program has only its 8 address inverters).
    let core = empty(Variant::Hierarchical) - 8;
    let core_gated = empty(Variant::GatedClockRegfile) - 8;
    println!("core: {core} gates, gated core: {core_gated} gates");
    assert_eq!(core, 1351);
    assert_eq!(core_gated, 1285);
    assert_eq!(count(Variant::Hierarchical, RomStyle::SumOfProducts), 3671);
    assert_eq!(count(Variant::Flat, RomStyle::SumOfProducts), 3671);
    assert_eq!(count(Variant::GatedClockRegfile, RomStyle::SumOfProducts), 3605);
    assert_eq!(count(Variant::Hierarchical, RomStyle::MuxTree), 7726);
    assert_eq!(count(Variant::GatedClockRegfile, RomStyle::MuxTree), 7660);
    assert_eq!(count(Variant::Hierarchical, RomStyle::Decoder), 1428);
    // The default build is inside the SPEC §12 size target.
    for v in Variant::ALL {
        let n = gates(&panda8(&p, v));
        assert!((3_000..=5_000).contains(&n), "{v:?}: {n} gates");
    }
    // Program-independent ROM styles.
    for rom in [RomStyle::SumOfProducts, RomStyle::MuxTree] {
        let a = gates(&panda8_with(&programs::COUNTDOWN.image(), &Options::new(Variant::Flat).with_rom(rom)));
        assert_eq!(a, count(Variant::Flat, rom), "{rom:?}");
    }
}
