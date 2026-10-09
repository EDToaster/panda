//! `FastSim::load_state`: a fresh engine loaded with another engine's settled state (values,
//! tick, inputs) behaves identically afterwards (values of all nets, tick, schedule), on Panda-8
//! (every Variant × RomStyle) and on random designs. At the end of an event every kernel's last
//! transition used its current inputs and its state is stable under them, which is exactly what
//! `load_state` reconstructs.

mod common;

/// Native-code hook of `common::pair` (none here: the interpreter; the panda-jit crate runs
/// this suite with the JIT installed through it).
fn native_hook(_: &mut panda_fast::FastSim) {}

use std::sync::Arc;

use common::{Coverage, check, pair, random_compile, random_value};
use panda_compile::{FastOptions, LoadReport};
use panda_cpu8::programs;
use panda_cpu8::{Options, RomStyle, Variant, panda8_with};
use panda_difftest::{GenConfig, Rng, generate};
use panda_fast::{FastSim, FastSimOptions};
use panda_netlist::flat::FlatNetlist;
use panda_netlist::{Logic, Value, flatten};

fn inputs_of(fs: &FastSim, flat: &FlatNetlist) -> Vec<Value> {
    flat.top_inputs
        .iter()
        .map(|p| fs.input_value(&p.name).unwrap().clone())
        .collect()
}

/// Load `a`'s state into a fresh engine (same kernels) and run both in lockstep for `events`
/// events.
fn reload_and_compare(a: &mut FastSim, flat: &Arc<FlatNetlist>, events: u64, ctx: &str) {
    let mut b = FastSim::from_parts(
        flat.clone(),
        a.options().clone(),
        a.kernels().clone(),
        a.tables().to_vec(),
    )
    .unwrap();
    native_hook(&mut b);
    let report = b.load_state(a.tick(), a.values(), &inputs_of(a, flat));
    assert_eq!(
        report,
        LoadReport::default(),
        "{ctx}: every kernel of a settled state is consistent"
    );
    assert_eq!(a.values(), b.values(), "{ctx}: after load");
    assert_eq!(a.next_event_tick(), b.next_event_tick(), "{ctx}: schedule");
    for e in 0..events {
        assert_eq!(a.step_event(), b.step_event(), "{ctx} event {e}");
        assert_eq!(a.tick(), b.tick());
        assert_eq!(a.next_event_tick(), b.next_event_tick());
        if a.values() != b.values() {
            let d: Vec<String> = (0..flat.num_nets as usize)
                .filter(|&n| a.values()[n] != b.values()[n])
                .take(8)
                .map(|n| flat.net_names[n].clone())
                .collect();
            panic!("{ctx} event {e}: values differ: {d:?}");
        }
    }
}

#[test]
fn panda8_reload_mid_run() {
    std::thread::scope(|s| {
        for rom in [
            RomStyle::SumOfProducts,
            RomStyle::MuxTree,
            RomStyle::Decoder,
        ] {
            for v in Variant::ALL {
                s.spawn(move || {
                    let o = Options::new(v).with_rom(rom);
                    let flat =
                        Arc::new(flatten(&panda8_with(&programs::BENCH_LOOP.image(), &o)).unwrap());
                    let opts = FastSimOptions::default();
                    let mut a = FastSim::new(flat.clone(), opts.clone()).unwrap();
                    native_hook(&mut a);
                    for (i, n) in [3u64, 17, 101, 400].into_iter().enumerate() {
                        a.run_events(n);
                        reload_and_compare(&mut a, &flat, 300, &format!("{rom:?} {v:?} #{i}"));
                    }
                });
            }
        }
    });
}

/// One random design: run, reload at several points, compare. Returns the reloads checked.
fn reload_seed(seed: u64) -> usize {
    let mut checked = 0;
    let mut rng = Rng::new(seed ^ 0x10AD);
    let g = generate(seed, &GenConfig::default(), 2 + 2 * rng.below(2) as u32);
    let flat = Arc::new(flatten(&g.design).unwrap());
    let opts = FastSimOptions::from(FastOptions::default());
    let mut a = FastSim::new(flat.clone(), opts).unwrap();
    native_hook(&mut a);
    let pins: Vec<(String, u32)> = flat
        .top_inputs
        .iter()
        .map(|p| (p.name.clone(), p.nets.len() as u32))
        .collect();
    for round in 0..4 {
        for _ in 0..1 + rng.below(30) {
            if !pins.is_empty() && rng.chance(30) {
                let (name, w) = rng.pick(&pins).clone();
                a.set_input(&name, &random_value(&mut rng, w, 0)).unwrap();
            }
            a.step_event();
        }
        a.take_diagnostics();
        // Only settled binary-clean states are guaranteed consistent (X/oscillating kernels or
        // kernels frozen at the delta limit may legitimately fall back to X).
        let mut b = FastSim::from_parts(
            flat.clone(),
            a.options().clone(),
            a.kernels().clone(),
            a.tables().to_vec(),
        )
        .unwrap();
        native_hook(&mut b);
        let report = b.load_state(a.tick(), a.values(), &inputs_of(&a, &flat));
        if report != LoadReport::default() || a.values().iter().any(|v| !v.is_known()) {
            continue;
        }
        reload_and_compare(&mut a, &flat, 40, &format!("seed {seed} round {round}"));
        checked += 1;
    }
    checked
}

#[test]
fn random_designs_reload() {
    let next = std::sync::atomic::AtomicU64::new(0);
    let checked = std::sync::atomic::AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let seed = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if seed >= 160 {
                        break;
                    }
                    checked.fetch_add(reload_seed(seed), std::sync::atomic::Ordering::Relaxed);
                }
            });
        }
    });
    let checked = checked.into_inner();
    assert!(checked > 60, "only {checked} reloads checked");
}

/// Perturbed (unsettled, poked, partly `X`/`Z`) states loaded into FastRef and FastSim over the
/// same kernels: both report the same inconsistent / fallen kernels and stay bit-exact afterwards
/// (values, time, schedule, statistics, diagnostic kinds, FastSim's invariants). Kernels caught
/// mid-transition finish it (`FastRef::load_state` semantics) — including kernels whose finished
/// state changes the inputs of kernels checked before or after them.
fn perturbed_load_seed(
    flat: &Arc<FlatNetlist>,
    seed: u64,
    rounds: usize,
    cov: &mut Coverage,
) -> (usize, usize) {
    let mut rng = Rng::new(seed ^ 0x5EED_10AD);
    let fast = FastOptions {
        delta_limit: *rng.pick(&[64, 64, 3]),
        ..FastOptions::default()
    };
    let compile = random_compile(&mut rng, &[1, 4, 32]);
    let (mut r, mut f) = pair(flat, &fast, &compile);
    let pins: Vec<(String, u32)> = flat
        .top_inputs
        .iter()
        .map(|p| (p.name.clone(), p.nets.len() as u32))
        .collect();
    let (mut finished, mut fallen) = (0, 0);
    check(&mut r, &mut f, flat, &format!("seed {seed} power-on"), cov);
    for round in 0..rounds {
        let ctx = format!("seed {seed} round {round} ({compile:?})");
        for e in 0..rng.below(25) {
            if !pins.is_empty() && rng.chance(30) {
                let (name, w) = rng.pick(&pins).clone();
                let v = random_value(&mut rng, w, 10);
                r.set_input(&name, &v).unwrap();
                f.set_input(&name, &v).unwrap();
            }
            assert_eq!(r.step_event(), f.step_event(), "{ctx} event {e}");
            check(&mut r, &mut f, flat, &format!("{ctx} event {e}"), cov);
        }
        // A perturbed state: mostly binary pokes (kernels that finish their transition), some
        // X/Z.
        let xz = rng.chance(30);
        let pct = *rng.pick(&[2, 5, 15]);
        let mut vals = r.values().to_vec();
        for v in &mut vals {
            if rng.chance(pct) {
                *v = if xz {
                    *rng.pick(&[Logic::Zero, Logic::One, Logic::X, Logic::Z])
                } else {
                    *rng.pick(&[Logic::Zero, Logic::One])
                };
            }
        }
        let inputs: Vec<Value> = pins
            .iter()
            .map(|(name, w)| {
                if rng.chance(30) {
                    random_value(&mut rng, *w, if xz { 20 } else { 0 })
                } else {
                    r.input_value(name).unwrap().clone()
                }
            })
            .collect();
        let tick = if rng.chance(50) {
            r.tick()
        } else {
            r.tick() + rng.below(50) as u64
        };
        let rr = r.load_state(tick, &vals, &inputs);
        let rf = f.load_state(tick, &vals, &inputs);
        assert_eq!(rr, rf, "{ctx}: load reports");
        finished += rr.inconsistent - rr.fallen;
        fallen += rr.fallen;
        check(&mut r, &mut f, flat, &format!("{ctx} after load"), cov);
    }
    (finished, fallen)
}

#[test]
fn perturbed_loads_fastref_equals_fastsim_random_designs() {
    let mut cov = Coverage::default();
    let (mut finished, mut fallen) = (0, 0);
    for seed in 0..60u64 {
        let mut rng = Rng::new(seed ^ 0xF00D);
        let g = generate(seed, &GenConfig::default(), 2 + 2 * rng.below(2) as u32);
        let flat = Arc::new(flatten(&g.design).unwrap());
        let (a, b) = perturbed_load_seed(&flat, seed, 6, &mut cov);
        finished += a;
        fallen += b;
    }
    assert!(
        finished > 100,
        "only {finished} kernels finished a transition"
    );
    assert!(fallen > 0, "no kernel fell back to X");
}

#[test]
fn perturbed_loads_fastref_equals_fastsim_panda8() {
    let mut cov = Coverage::default();
    let mut finished = 0;
    for (i, (rom, v)) in [
        (RomStyle::SumOfProducts, Variant::Hierarchical),
        (RomStyle::MuxTree, Variant::GatedClockRegfile),
    ]
    .into_iter()
    .enumerate()
    {
        let o = Options::new(v).with_rom(rom);
        let flat = Arc::new(flatten(&panda8_with(&programs::BENCH_LOOP.image(), &o)).unwrap());
        finished += perturbed_load_seed(&flat, i as u64, 4, &mut cov).0;
    }
    assert!(finished > 0, "no kernel finished a transition");
}
