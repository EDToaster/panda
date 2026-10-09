//! The order-sensitivity lemma behind phase certainty (`panda_fast::kspec` module docs): for a
//! kernel classified Opaque or Transparent under a clock tuple, every *data* transition with two
//! or more changed inputs is order-insensitive, so FastSim may skip FastRef's runtime order check
//! for it. Checked here with FastRef's own order checker (`KernelTable::check_order_sensitive`)
//! on Panda-8 and random designs (sampled per kernel class and tuple).

use std::collections::HashSet;
use std::sync::Arc;

use panda_compile::subnet::kernel_subnet;
use panda_compile::{CharzOptions, KernelSim};
use panda_cpu8::programs;
use panda_cpu8::{Options, Variant, panda8_with};
use panda_difftest::{GenConfig, Rng, generate};
use panda_fast::{FastSim, FastSimOptions, PhaseClass};
use panda_netlist::flat::FlatNetlist;
use panda_netlist::flatten;

/// Check up to `budget` data transitions per (table, pure mask, pure bits); returns the number
/// checked.
fn check_design(flat: &Arc<FlatNetlist>, rng: &mut Rng, budget: usize) -> usize {
    let fs = FastSim::new(flat.clone(), FastSimOptions::default()).expect("compile");
    let prog = fs.program();
    let nt = prog.num_tuples as usize;
    let nk = prog.kernel_ops.len();
    let opts = CharzOptions::default();
    let mut done: HashSet<(usize, u32, u32)> = HashSet::new();
    let mut checked = 0;
    for k in 0..nk {
        let Some(table) = fs.tables()[k].as_ref() else {
            continue;
        };
        if prog.kernel_class[k] == u32::MAX {
            continue;
        }
        let mask = prog.kpure_mask[k];
        let kk = table.num_inputs as u32;
        let data = ((1u32 << kk) - 1) & !mask;
        let mut sim = None;
        for t in 0..nt {
            if !matches!(
                prog.phase_class(k as u32, t as u32),
                PhaseClass::Opaque | PhaseClass::Transparent
            ) {
                continue;
            }
            let p = prog.kernel_fast[k * nt + t].pure;
            if !done.insert((Arc::as_ptr(table) as usize, mask, p)) {
                continue;
            }
            let sim = sim.get_or_insert_with(|| {
                let (sub, map) = kernel_subnet(flat, &fs.kernels().kernels[k]);
                KernelSim::from_subnet(Arc::new(sub), map, &opts)
            });
            // Random data vectors u != v (pure bits p) with >= 2 differing bits.
            let ndata = data.count_ones();
            if ndata < 2 {
                continue;
            }
            let spread = |r: u64| -> u32 {
                // Deposit the low bits of `r` into the data positions.
                let (mut out, mut bits, mut i) = (0u32, data, 0);
                while bits != 0 {
                    let b = bits.trailing_zeros();
                    bits &= bits - 1;
                    out |= (((r >> i) & 1) as u32) << b;
                    i += 1;
                }
                out
            };
            let exhaustive = ndata <= 5;
            let pairs: Vec<(u32, u32)> = if exhaustive {
                let n = 1u64 << ndata;
                (0..n)
                    .flat_map(|a| (0..n).map(move |b| (a, b)))
                    .map(|(a, b)| (spread(a) | p, spread(b) | p))
                    .collect()
            } else {
                (0..budget)
                    .map(|_| (spread(rng.next_u64()) | p, spread(rng.next_u64()) | p))
                    .collect()
            };
            for (u, v) in pairs {
                if (u ^ v).count_ones() < 2 {
                    continue;
                }
                for &s in &table.states_by_input[u as usize] {
                    assert!(
                        !table.check_order_sensitive(sim, &opts, s, u, v),
                        "kernel {k} tuple {t}: data transition (s {s}, u {u:b}, v {v:b}) of a \
                         {:?} phase is order-sensitive",
                        prog.phase_class(k as u32, t as u32)
                    );
                    checked += 1;
                }
            }
        }
    }
    checked
}

#[test]
fn opaque_transparent_data_transitions_are_order_insensitive_panda8() {
    let mut rng = Rng::new(0x0D0E_51A5);
    let mut total = 0;
    for v in Variant::ALL {
        let flat = Arc::new(
            flatten(&panda8_with(
                &programs::BENCH_LOOP.image(),
                &Options::new(v),
            ))
            .expect("flatten"),
        );
        total += check_design(&flat, &mut rng, 400);
    }
    assert!(total > 1000, "only {total} transitions checked");
}

#[test]
fn opaque_transparent_data_transitions_are_order_insensitive_random() {
    let mut rng = Rng::new(0x5EED_0DE5);
    let mut total = 0;
    for seed in 0..200 {
        let g = generate(seed, &GenConfig::default(), 2 + (seed % 3) as u32 * 2);
        let flat = Arc::new(flatten(&g.design).expect("flatten"));
        total += check_design(&flat, &mut rng, 200);
    }
    assert!(total > 1000, "only {total} transitions checked");
}
