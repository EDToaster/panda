//! Random designs (panda-difftest generator) through the facade.

use panda_difftest::{GenConfig, Rng, generate};
use panda_kernel::{Kernel, KernelOptions, Mode, StepUnit};
use panda_netlist::{DiagKind, Value};

fn inputs(k: &Kernel, rng: &mut Rng) -> Option<(String, Value)> {
    let flat = k.netlist()?;
    if flat.top_inputs.is_empty() {
        return None;
    }
    let p = &flat.top_inputs[rng.below(flat.top_inputs.len())];
    let w = p.nets.len() as u32;
    Some((
        p.name.clone(),
        Value::from_u64(w, rng.next_u64() & ((1u64 << w.min(63)) - 1)),
    ))
}

fn all_values(k: &Kernel) -> Vec<Option<Value>> {
    k.netlist()
        .unwrap()
        .net_names
        .clone()
        .iter()
        .map(|n| k.read(n))
        .collect()
}

/// On STA-clean designs, Fast mode equals Timing mode at every phase end (SPEC §6.3), through the
/// facade's common clock; and switching modes at a phase end loses nothing.
#[test]
fn modes_agree_at_phase_ends_on_clean_designs() {
    let cfg = GenConfig {
        async_pct: 0,
        ..GenConfig::default()
    };
    let (mut clean, mut checked_phases) = (0, 0);
    const SEEDS: u64 = 100;
    for seed in 0..SEEDS {
        let g = generate(seed, &cfg, 80);
        let opts = KernelOptions {
            background: false,
            ..Default::default()
        };
        let mut t = Kernel::new(opts.clone());
        t.load(g.design.clone()).unwrap();
        if !t.timing_report().is_clean() {
            continue;
        }
        clean += 1;
        let mut f = Kernel::new(opts);
        f.set_mode(Mode::Fast);
        f.load(g.design.clone()).unwrap();
        t.step(1, StepUnit::Phase);
        assert_eq!(t.tick(), f.tick(), "seed {seed}");
        let mut rng = Rng::new(seed);
        let release = u64::from(g.reset_ticks);
        for ph in 0..40 {
            // Input changes during a phase, after the edge activity (outside hold windows):
            // Timing mode applies them 5 ticks after the edge; Fast mode processes them as their
            // own event after the edge's event (`step(0, Tick)` applies pending inputs).
            if rng.chance(25)
                && let Some((pin, v)) = inputs(&t, &mut rng)
            {
                t.step(6, StepUnit::Tick);
                t.set_input(&pin, &v).unwrap();
                t.step(1, StepUnit::Phase);
                f.step(1, StepUnit::Phase);
                f.set_input(&pin, &v).unwrap();
                f.step(0, StepUnit::Tick);
            } else {
                t.step(1, StepUnit::Phase);
                f.step(1, StepUnit::Phase);
            }
            assert_eq!(t.tick(), f.tick(), "seed {seed} phase {ph}");
            if t.tick()
                > release + 2 * u64::from(g.clocks.iter().map(|c| c.period).max().unwrap_or(0))
            {
                let (a, b) = (all_values(&t), all_values(&f));
                // Power-on X transients are out of the equivalence's scope: compare binary nets.
                for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                    if x.as_ref().is_some_and(|v| v.is_fully_known())
                        && y.as_ref().is_some_and(|v| v.is_fully_known())
                    {
                        assert_eq!(
                            x,
                            y,
                            "seed {seed} phase {ph} net {}",
                            t.netlist().unwrap().net_names[i]
                        );
                    }
                }
                checked_phases += 1;
            }
        }
        // Lossless round trip of the state at a phase end: Timing → Fast → Timing.
        let before = all_values(&t);
        let tick = t.tick();
        t.set_mode(Mode::Fast);
        assert_eq!(t.tick(), tick);
        assert_eq!(
            all_values(&t),
            before,
            "seed {seed}: Timing → Fast at a phase end"
        );
        t.set_mode(Mode::Timing);
        assert_eq!(t.tick(), tick);
        assert_eq!(all_values(&t), before, "seed {seed}: Fast → Timing");
    }
    eprintln!("{clean} STA-clean designs of {SEEDS}, {checked_phases} phase ends compared");
    assert!(clean >= 8, "too few clean designs ({clean})");
}

/// In Timing mode the net values at any tick determine the future: restoring a snapshot taken at
/// an arbitrary tick replays exactly (absent period-2 collapse, whose history is not part of the
/// state).
#[test]
fn timing_snapshots_replay_exactly() {
    let cfg = GenConfig::default();
    let mut compared = 0;
    for seed in 0..30u64 {
        let g = generate(seed, &cfg, 8);
        let mut k = Kernel::default();
        k.load(g.design.clone()).unwrap();
        let mut rng = Rng::new(seed ^ 0xabc);
        k.step(rng.range(1, 400) as u64, StepUnit::Tick);
        let snap = k.snapshot();
        k.clear_diagnostics();
        let mut first = Vec::new();
        for _ in 0..50 {
            k.step(3, StepUnit::Tick);
            first.push(all_values(&k));
        }
        let meta = k
            .diagnostics()
            .iter()
            .any(|d| d.kind == DiagKind::Metastable);
        k.restore(&snap).unwrap();
        let mut second = Vec::new();
        for _ in 0..50 {
            k.step(3, StepUnit::Tick);
            second.push(all_values(&k));
        }
        let meta = meta
            || k.diagnostics()
                .iter()
                .any(|d| d.kind == DiagKind::Metastable);
        if !meta {
            assert_eq!(first, second, "seed {seed}");
            compared += 1;
        }
    }
    eprintln!("{compared} of 30 designs compared");
    assert!(compared >= 20);
}
