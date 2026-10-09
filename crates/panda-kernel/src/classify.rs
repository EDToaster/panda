//! Package classification (SPEC §4.4, §8.3 "clock-like pins").

use std::collections::HashMap;
use std::sync::Arc;

use panda_compile::{CharzOptions, FastOptions, KernelKind, KernelOptions, KernelSim, TableCache};
use panda_netlist::flat::FlatNetlist;
use panda_netlist::{DefId, Logic, Value};

use crate::refengine::RefEngine;

/// The class of a package (SPEC §4.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClassKind {
    /// Acyclic, or cyclic but ternary-combinational (every cyclic kernel reaches an all-binary
    /// state from all-`X` for every binary input vector).
    Combinational,
    /// Holds state.
    Sequential,
    /// Some binary input vector leaves a kernel without a binary stable state (e.g. a ring
    /// oscillator). Takes precedence over the other two classes.
    Oscillator,
}

/// Which clock edge an edge-triggered package reacts to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Edge {
    /// 0 → 1.
    Rising,
    /// 1 → 0.
    Falling,
}

/// Summary of one kernel found in a package.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelSummary {
    /// Evaluation strategy chosen by the compiler.
    pub kind: KernelKind,
    /// Name of a representative net (first output, else first state net).
    pub location: String,
    /// Gates inside the kernel.
    pub gates: usize,
    /// Kernel input bits.
    pub inputs: usize,
    /// Nets of the kernel state.
    pub state_nets: usize,
    /// Number of binary stable states (characterized kernels only).
    pub stable_states: Option<usize>,
    /// State bits: `⌈log2(max stable states under one input vector)⌉` for characterized kernels,
    /// 1 for keepers, the number of state nets (an upper bound) otherwise.
    pub state_bits: u32,
    /// The kernel has an input vector without a binary stable state.
    pub oscillator: bool,
    /// Ternary simulation from all-`X` reaches an all-binary state for every binary input
    /// vector (the kernel is cyclic-combinational).
    pub ternary_combinational: bool,
}

/// Result of [`Kernel::classify`](crate::Kernel::classify).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Classification {
    /// The classified def.
    pub def: DefId,
    /// The class.
    pub kind: ClassKind,
    /// The flattened gate graph has a cycle (or a keeper net).
    pub cyclic: bool,
    /// Total state bits (sum over the kernels that are not ternary-combinational).
    pub state_bits: u32,
    /// Kernels found.
    pub kernels: Vec<KernelSummary>,
    /// Clock-like input pins (SPEC §8.3): for each constant value of the pin, the outputs never
    /// change when other inputs change. Empty unless `clock_analysis` is true.
    pub clock_like_pins: Vec<String>,
    /// Edge-triggered: the single clock-like pin and the only edge of it that changes outputs.
    pub edge_triggered: Option<(String, Edge)>,
    /// The clock-like analysis ran (sequential packages without time-varying sources and with
    /// at most [`MAX_EXPLORE_INPUT_BITS`] input bits whose reachable state space was explored
    /// completely within the work bound).
    pub clock_analysis: bool,
}

/// Largest number of input bits for the clock-like-pin exploration.
pub const MAX_EXPLORE_INPUT_BITS: usize = 8;
/// Largest number of explored states for the clock-like-pin exploration.
const MAX_EXPLORE_STATES: usize = 1024;
/// Largest number of kernel input bits for the ternary-combinational check.
const MAX_TERNARY_BITS: usize = 16;

pub(crate) fn classify(
    def: &str,
    flat: Arc<FlatNetlist>,
    fopts: &FastOptions,
    cache: &mut TableCache,
) -> Classification {
    let kopts = KernelOptions {
        extract: fopts.extract.clone(),
        charz: fopts.charz.clone(),
    };
    let (ks, tables) = panda_compile::compile_kernels_with_cache(&flat, &kopts, cache);
    let mut kernels = Vec::new();
    for (k, t) in ks.kernels.iter().zip(&tables) {
        let location = k
            .outputs
            .first()
            .or(k.nets.first())
            .map(|&n| flat.net_names[n as usize].clone())
            .unwrap_or_default();
        let (stable_states, oscillator, bits) = match (k.kind, t) {
            (KernelKind::Table, Some(t)) => {
                let max = t.states_by_input.iter().map(Vec::len).max().unwrap_or(0);
                (Some(t.num_states()), t.is_oscillator(), ceil_log2(max))
            }
            (KernelKind::Keeper, _) => (None, false, 1),
            _ => (
                None,
                uncharacterized_oscillator(&flat, k, &fopts.charz),
                k.nets.len() as u32,
            ),
        };
        let ternary = k.kind != KernelKind::Keeper
            && !oscillator
            && ternary_combinational(&flat, k, &fopts.charz);
        kernels.push(KernelSummary {
            kind: k.kind,
            location,
            gates: k.gates.len(),
            inputs: k.inputs.len(),
            state_nets: k.nets.len(),
            stable_states,
            state_bits: bits,
            oscillator,
            ternary_combinational: ternary,
        });
    }
    let cyclic = !kernels.is_empty();
    let oscillator = kernels.iter().any(|k| k.oscillator);
    let sequential = kernels.iter().any(|k| !k.ternary_combinational);
    let kind = if oscillator {
        ClassKind::Oscillator
    } else if sequential {
        ClassKind::Sequential
    } else {
        ClassKind::Combinational
    };
    let state_bits = kernels
        .iter()
        .filter(|k| !k.ternary_combinational)
        .map(|k| k.state_bits)
        .sum();
    let mut c = Classification {
        def: def.to_string(),
        kind,
        cyclic,
        state_bits,
        kernels,
        clock_like_pins: Vec::new(),
        edge_triggered: None,
        clock_analysis: false,
    };
    if kind == ClassKind::Sequential
        && let Ok(engine) = RefEngine::from_parts(flat.clone(), fopts.clone(), ks, tables)
        && let Some((pins, edge)) = clock_like(&flat, engine)
    {
        c.clock_like_pins = pins;
        c.edge_triggered = edge;
        c.clock_analysis = true;
    }
    c
}

fn ceil_log2(n: usize) -> u32 {
    if n <= 1 {
        0
    } else {
        usize::BITS - (n - 1).leading_zeros()
    }
}

/// Largest number of state nets for [`uncharacterized_oscillator`] (all binary states are
/// enumerated per input vector).
const MAX_OSC_STATE_BITS: usize = 16;
/// Work bound of [`uncharacterized_oscillator`] (fixed-point checks).
const MAX_OSC_CHECKS: usize = 1 << 22;
/// Input vectors are enumerated up to this many kernel inputs, sampled above.
const MAX_OSC_ENUM_INPUTS: usize = 12;

/// The oscillator flag of a kernel without a table (too many inputs or states to characterize):
/// true if one of the examined binary input vectors leaves the kernel without any binary stable
/// state (the table's definition, [`panda_compile::KernelTable::is_oscillator`]), checked by
/// enumerating every binary state. Input vectors are enumerated for small kernels and sampled
/// otherwise (all-0, all-1, every single-bit deviation from those, and pseudo-random vectors), so
/// `true` is always exact evidence while `false` may miss an oscillating vector. Kernels with more
/// than [`MAX_OSC_STATE_BITS`] state nets or with resolvers are not examined.
fn uncharacterized_oscillator(
    flat: &FlatNetlist,
    k: &panda_compile::Kernel,
    charz: &CharzOptions,
) -> bool {
    let (ni, ns) = (k.inputs.len(), k.nets.len());
    // Resolver nets can hold Z (not binary) in a stable state: such kernels are not examined.
    if ns > MAX_OSC_STATE_BITS || k.kind == KernelKind::Keeper || !k.resolvers.is_empty() {
        return false;
    }
    let bit = |v: &[u64], i: usize| v[i / 64] >> (i % 64) & 1 != 0;
    let words = ni.div_ceil(64).max(1);
    let mut vectors: Vec<Vec<u64>> = Vec::new();
    if ni <= MAX_OSC_ENUM_INPUTS {
        vectors.extend((0..1u64 << ni).map(|v| vec![v]));
    } else {
        let ones: Vec<u64> = (0..words)
            .map(|w| {
                let n = (ni - w * 64).min(64);
                if n == 64 { u64::MAX } else { (1u64 << n) - 1 }
            })
            .collect();
        let zeros = vec![0u64; words];
        for base in [&zeros, &ones] {
            vectors.push(base.clone());
            for i in 0..ni {
                let mut v = base.clone();
                v[i / 64] ^= 1 << (i % 64);
                vectors.push(v);
            }
        }
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..256 {
            let v: Vec<u64> = (0..words)
                .map(|w| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    x & ones[w]
                })
                .collect();
            vectors.push(v);
        }
    }
    let per_vector = 1usize << ns;
    vectors.truncate((MAX_OSC_CHECKS / per_vector).max(1));
    let mut sim = KernelSim::new(flat, k, charz);
    let mut inputs = vec![Logic::Zero; ni];
    let mut state = vec![Logic::Zero; ns];
    for v in &vectors {
        for (i, l) in inputs.iter_mut().enumerate() {
            *l = Logic::from_bool(bit(v, i));
        }
        let stable = (0..per_vector as u64).any(|st| {
            for (j, l) in state.iter_mut().enumerate() {
                *l = Logic::from_bool(st >> j & 1 != 0);
            }
            sim.is_fixed_point(&state, &inputs)
        });
        if !stable {
            return true;
        }
    }
    false
}

/// Ternary simulation of the kernel from all-`X` for every binary input vector.
fn ternary_combinational(
    flat: &FlatNetlist,
    k: &panda_compile::Kernel,
    charz: &CharzOptions,
) -> bool {
    if k.inputs.len() > MAX_TERNARY_BITS {
        return false;
    }
    let mut sim = KernelSim::new(flat, k, charz);
    let xs = vec![Logic::X; k.nets.len()];
    let bound = sim.default_bound();
    for v in 0..1u32 << k.inputs.len() {
        let inputs = panda_compile::charz::input_vector(v, k.inputs.len());
        sim.load_state(&xs, &inputs);
        if !sim.settle(bound).quiescent || sim.read_state().contains(&Logic::X) {
            return false;
        }
    }
    true
}

/// Explore the reachable states (Fast-mode transitions, one input change at a time from the
/// power-on state) and find the clock-like pins. `None` if the exploration is not possible
/// within the bounds.
#[allow(clippy::type_complexity)]
fn clock_like(
    flat: &FlatNetlist,
    mut eng: RefEngine,
) -> Option<(Vec<String>, Option<(String, Edge)>)> {
    use panda_netlist::flat::SourceKind;
    if flat.sources.iter().any(|s| {
        matches!(
            s.kind,
            SourceKind::Clock { .. } | SourceKind::PowerOnReset { .. }
        )
    }) {
        return None;
    }
    let pins = &flat.top_inputs;
    let widths: Vec<usize> = pins.iter().map(|p| p.nets.len()).collect();
    let nbits: usize = widths.iter().sum();
    if pins.is_empty() || nbits > MAX_EXPLORE_INPUT_BITS {
        return None;
    }
    let offs: Vec<usize> = widths
        .iter()
        .scan(0, |o, &w| Some(std::mem::replace(o, *o + w)))
        .collect();
    let to_inputs = |v: u32| -> Vec<Value> {
        widths
            .iter()
            .zip(&offs)
            .map(|(&w, &o)| Value::from_u64(w as u32, u64::from(v >> o) & ((1u64 << w) - 1)))
            .collect()
    };
    let out_nets: Vec<u32> = flat
        .top_outputs
        .iter()
        .flat_map(|p| p.nets.iter().copied())
        .collect();
    let outputs = |vals: &[Logic]| -> Option<Vec<Logic>> {
        let o: Vec<Logic> = out_nets.iter().map(|&n| vals[n as usize]).collect();
        o.iter().all(|l| l.is_known()).then_some(o)
    };

    // Nodes: (settled values, input vector they settled under).
    let mut nodes: Vec<(Vec<Logic>, u32)> = vec![(eng.values().to_vec(), 0)];
    let mut index: HashMap<(Vec<Logic>, u32), usize> = HashMap::new();
    index.insert(nodes[0].clone(), 0);
    let mut edges: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < nodes.len() {
        let (vals, u) = nodes[i].clone();
        for v in 0..1u32 << nbits {
            if v == u {
                continue;
            }
            eng.load_state(0, &vals, &to_inputs(u));
            for (p, val) in pins.iter().zip(to_inputs(v)) {
                eng.set_input(&p.name, &val).ok()?;
            }
            eng.step_event();
            let key = (eng.values().to_vec(), v);
            let j = match index.get(&key) {
                Some(&j) => j,
                None => {
                    if nodes.len() >= MAX_EXPLORE_STATES {
                        return None;
                    }
                    index.insert(key.clone(), nodes.len());
                    nodes.push(key);
                    nodes.len() - 1
                }
            };
            edges.push((i, j));
        }
        i += 1;
    }
    eng.take_diagnostics();

    let outs: Vec<Option<Vec<Logic>>> = nodes.iter().map(|(v, _)| outputs(v)).collect();
    let mut clock_like = Vec::new();
    let mut edge = None;
    for (pi, pin) in pins.iter().enumerate() {
        let mask = ((1u32 << widths[pi]) - 1) << offs[pi];
        let others = !mask & ((1u32 << nbits) - 1);
        let (mut dependent, mut observed, mut matters) = (false, others == 0, false);
        let (mut rise, mut fall) = (false, false);
        for &(a, b) in &edges {
            let (ua, ub) = (nodes[a].1, nodes[b].1);
            let (Some(oa), Some(ob)) = (&outs[a], &outs[b]) else {
                continue;
            };
            let changed = oa != ob;
            if ua & mask == ub & mask {
                observed = true;
                dependent |= changed;
            } else if ua & others == ub & others && changed {
                matters = true;
                if widths[pi] == 1 {
                    if ub & mask != 0 {
                        rise = true;
                    } else {
                        fall = true;
                    }
                }
            }
        }
        if !dependent && observed && matters {
            clock_like.push(pin.name.clone());
            if rise != fall {
                edge = Some((
                    pin.name.clone(),
                    if rise { Edge::Rising } else { Edge::Falling },
                ));
            }
        }
    }
    if clock_like.len() != 1 {
        edge = None;
    }
    Some((clock_like, edge))
}

#[cfg(test)]
mod tests {
    use super::ceil_log2;

    #[test]
    fn log2() {
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
    }
}
