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

use common::random_value;
use panda_compile::FastOptions;
use panda_cpu8::programs;
use panda_cpu8::{Options, RomStyle, Variant, panda8_with};
use panda_difftest::{GenConfig, Rng, generate};
use panda_fast::{FastSim, FastSimOptions};
use panda_netlist::flat::FlatNetlist;
use panda_netlist::{Value, flatten};

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
    let fallen = b.load_state(a.tick(), a.values(), &inputs_of(a, flat));
    assert_eq!(
        fallen, 0,
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
        let fallen = b.load_state(a.tick(), a.values(), &inputs_of(&a, &flat));
        if fallen != 0 || a.values().iter().any(|v| !v.is_known()) {
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
