//! Panda-8 (SPEC §12): every variant must come out clean, with a reported minimum period at which
//! Timing mode still runs the programs correctly and FastRef equals Timing mode.

mod common;

use common::*;
use panda_cpu8::emu::Emulator;
use panda_cpu8::sim::GateCpu;
use panda_cpu8::{Options, RomStyle, Variant, panda8_with, programs};
use panda_netlist::DiagKind;
use panda_sta::EventKind;

const ROMS: [RomStyle; 2] = [RomStyle::SumOfProducts, RomStyle::MuxTree];

/// Panda-8 needs an even period ≥ 8 (and the reset release sits a quarter period into a low phase).
fn usable_period(p: u32) -> u32 {
    assert!(p < u32::MAX / 2, "unbounded min period");
    p.max(8).next_multiple_of(2)
}

#[test]
fn panda8_all_variants_clean_and_run_at_min_period() {
    let program = programs::FIBONACCI;
    let image = program.image();
    let golden = Emulator::new(&image).run(2_000);
    assert!(!golden.is_empty());
    for variant in Variant::ALL {
        for rom in ROMS {
            let opts = Options::new(variant).with_rom(rom);
            let flat = flat_of(&panda8_with(&image, &opts));
            let r = sta(&flat);
            dump(&format!("{variant:?} {rom:?} @ {}", opts.period), &r);
            assert!(
                r.is_clean(),
                "{variant:?} {rom:?}: {:?}",
                r.errors().collect::<Vec<_>>()
            );
            assert_eq!(r.min_period.len(), 1);
            let min = r.min_period_of("clk").expect("clk min period");
            // The measured settle times (panda-cpu8 DEFAULT_PERIOD docs: 35 / 47 ticks after the
            // rising edge) bound the period from below; the default period works.
            assert!(
                min > 2 * 35,
                "{variant:?} {rom:?}: min period {min} below the measured settle"
            );
            assert!(min <= opts.period, "{variant:?} {rom:?}: min period {min}");
            assert!(!r.critical_path.is_empty());
            let rise = r
                .phase_settle
                .iter()
                .find(|p| p.event == "clk↑")
                .expect("rising-edge event");
            assert!(rise.settle.is_some_and(|s| s < min / 2 + 1));

            // At the minimum period: still clean (including the reset release), and the gate-level
            // CPU in Timing mode produces the emulator's OUT sequence.
            let p = usable_period(min);
            let opts_p = opts.clone().with_period(p);
            let design = panda8_with(&image, &opts_p);
            let rp = sta(&flat_of(&design));
            assert!(
                rp.is_clean(),
                "{variant:?} {rom:?} @ {p}: {:?}",
                rp.errors().collect::<Vec<_>>()
            );
            let reset = rp
                .phase_settle
                .iter()
                .find(|x| x.kind == EventKind::Reset)
                .expect("reset event");
            assert!(reset.settle.unwrap() < reset.gap.unwrap());
            let mut cpu = GateCpu::new(&design).expect("gate cpu");
            let outs = cpu.run_outputs(2_000).expect("run");
            assert_eq!(outs, golden, "{variant:?} {rom:?} @ period {p}");

            // Two ticks below the minimum: a setup violation is reported.
            let below = panda8_with(&image, &opts.clone().with_period(p - 2));
            let rb = sta(&flat_of(&below));
            assert!(
                count(&rb, DiagKind::SetupViolation) > 0,
                "{variant:?} {rom:?} @ {}",
                p - 2
            );
        }
    }
}

#[test]
fn panda8_fastref_equals_timing_at_min_period() {
    // The STA contract: clean + period ≥ min_period ⇒ FastRef equals Timing mode on every net at
    // every phase boundary.
    let image = programs::MULTIPLY.image();
    for variant in Variant::ALL {
        for rom in ROMS {
            let opts = Options::new(variant).with_rom(rom);
            let r = sta(&flat_of(&panda8_with(&image, &opts)));
            let p = usable_period(r.max_min_period().unwrap());
            let opts_p = opts.with_period(p);
            let flat = flat_of(&panda8_with(&image, &opts_p));
            assert!(sta(&flat).is_clean());
            let from = u64::from(opts_p.reset_ticks());
            match differential(&flat, 600, from) {
                Ok(n) => assert!(n > 400, "{variant:?} {rom:?}: only {n} phases compared"),
                Err(m) => panic!("{variant:?} {rom:?} @ {p}: FastRef differs from Timing: {m:?}"),
            }
        }
    }
}

#[test]
fn panda8_ideal_blocks_fastref_equals_timing_at_ideal_min_period() {
    // Timing mode with ideal blocks (SPEC §5.2): analyzed with `StaOptions::ideal_blocks`.
    use panda_sta::{StaOptions, analyze_with};
    let image = programs::MULTIPLY.image();
    let opts_sta = StaOptions {
        ideal_blocks: true,
        ..StaOptions::default()
    };
    for variant in Variant::ALL {
        let rom = RomStyle::SumOfProducts;
        let opts = Options::new(variant).with_rom(rom);
        let flat = flat_of(&panda8_with(&image, &opts));
        let (ks, tables) = panda_compile::compile_kernels(&flat, &Default::default());
        let r = analyze_with(&flat, &ks, &tables, &opts_sta);
        let glass = sta(&flat).max_min_period().unwrap();
        let p = usable_period(r.max_min_period().unwrap());
        println!("{variant:?}: ideal-block min period {p} (glass-box {glass})");
        assert!(
            p >= glass,
            "ideal blocks never settle earlier than their glass-box gates"
        );
        let opts_p = opts.with_period(p);
        let flat = flat_of(&panda8_with(&image, &opts_p));
        let (ks, tables) = panda_compile::compile_kernels(&flat, &Default::default());
        let rp = analyze_with(&flat, &ks, &tables, &opts_sta);
        assert!(
            rp.is_clean(),
            "{variant:?} @ {p}: {:?}",
            rp.errors().collect::<Vec<_>>()
        );
        let from = u64::from(opts_p.reset_ticks());
        let topts = panda_timing::TimingOptions::default().with_ideal_blocks(true);
        match differential_opts(&flat, 600, from, &[], topts) {
            Ok(n) => assert!(n > 400, "{variant:?}: only {n} phases compared"),
            Err(m) => panic!("{variant:?} @ {p}: FastRef differs from ideal Timing: {m:?}"),
        }
    }
}

/// Smallest even period (searching down from `start`) at which Timing mode still produces the
/// emulator's OUT sequence and FastRef equals Timing mode on every net.
fn empirical_min_period(
    image: &[u8],
    opts: &Options,
    start: u32,
    golden: &[u8],
    cycles: u64,
) -> u32 {
    let mut best = start;
    let mut p = start;
    while p >= 8 {
        let o = opts.clone().with_period(p);
        let design = panda8_with(image, &o);
        let ok_out = GateCpu::new(&design)
            .and_then(|mut c| c.run_outputs(cycles))
            .is_ok_and(|outs| outs == golden);
        let ok_eq =
            ok_out && differential(&flat_of(&design), 400, u64::from(o.reset_ticks())).is_ok();
        if !ok_eq {
            break;
        }
        best = p;
        p -= 2;
    }
    best
}

#[test]
#[ignore = "pessimism report (slow): run with --ignored --nocapture"]
fn panda8_min_period_pessimism_report() {
    let program = programs::FIBONACCI;
    let image = program.image();
    let golden = Emulator::new(&image).run(300);
    assert!(!golden.is_empty());
    for variant in Variant::ALL {
        for rom in ROMS {
            let opts = Options::new(variant).with_rom(rom);
            let r = sta(&flat_of(&panda8_with(&image, &opts)));
            let sta_min = usable_period(r.max_min_period().unwrap());
            let emp = empirical_min_period(&image, &opts, sta_min, &golden, 300);
            println!(
                "Panda-8 {variant:?} {rom:?}: STA min period {sta_min}, empirical {emp} \
                 (FastRef = Timing and correct OUT down to it), pessimism {:.0}%",
                100.0 * (f64::from(sta_min) / f64::from(emp) - 1.0)
            );
        }
    }
}
