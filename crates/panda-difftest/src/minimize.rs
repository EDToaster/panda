//! Delta debugging of failing designs, and a readable netlist dump.

use std::fmt::Write;

use panda_netlist::flat::{FlatNetlist, SourceKind};
use panda_netlist::{Atom, Design, Dir, Endpoint, InstKind, Value};

use crate::generator::Generated;
use crate::harness::{HarnessConfig, Outcome, finish, prepare_with, recheck};

/// Human-readable listing of a flat netlist: sources, gates (with kernel-independent names) and
/// resolvers.
pub fn describe(flat: &FlatNetlist) -> String {
    let n = |i: u32| flat.net_names[i as usize].as_str();
    let mut s = String::new();
    for src in &flat.sources {
        let k = match &src.kind {
            SourceKind::Constant(v) => format!("const {}", v.to_char()),
            SourceKind::Clock {
                period,
                high,
                phase,
            } => format!("clock p={period} h={high} ph={phase}"),
            SourceKind::PowerOnReset { ticks } => format!("por {ticks}"),
            SourceKind::TopInput { pin, bit } => {
                format!("input {}[{bit}]", flat.top_inputs[*pin as usize].name)
            }
        };
        let _ = writeln!(s, "{} = {k}", n(src.output));
    }
    for (i, g) in flat.gates.iter().enumerate() {
        let ins: Vec<&str> = g.inputs.iter().map(|&x| n(x)).collect();
        let _ = writeln!(s, "{} = {:?}({}) #g{i}", n(g.output), g.op, ins.join(", "));
    }
    for r in &flat.resolvers {
        let ins: Vec<&str> = r.drivers.iter().map(|&x| n(x)).collect();
        let _ = writeln!(
            s,
            "{} = resolve[{:?}]({})",
            n(r.output),
            r.termination,
            ins.join(", ")
        );
    }
    s
}

/// Replace the gate instance `id` of the (single-def) top by a constant driving its output net.
fn constantize(design: &mut Design, id: &str, one: bool) -> bool {
    let top_id = design.top.clone();
    let Some(def) = design.defs.iter_mut().find(|d| d.id == top_id) else {
        return false;
    };
    let Some(inst) = def.instances.iter_mut().find(|i| i.id == id) else {
        return false;
    };
    let InstKind::Atom(atom) = &inst.kind else {
        return false;
    };
    if !matches!(
        atom,
        Atom::Gate { .. } | Atom::Not { .. } | Atom::Buf { .. } | Atom::Tri { .. }
    ) {
        return false;
    }
    let Some(out) = atom.ports().into_iter().find(|p| p.dir == Dir::Out) else {
        return false;
    };
    let value = if one {
        Value::ones(out.width)
    } else {
        Value::zeros(out.width)
    };
    inst.kind = InstKind::Atom(Atom::Constant { value });
    for net in &mut def.nets {
        net.ends
            .retain(|e| !matches!(e, Endpoint::Port { inst, port } if inst == id && port != "out"));
    }
    def.nets.retain(|n| !n.ends.is_empty());
    true
}

/// Gate instance ids of the top def.
fn gate_ids(design: &Design) -> Vec<String> {
    let Some(def) = design.top_def() else {
        return Vec::new();
    };
    def.instances
        .iter()
        .filter(|i| {
            matches!(
                &i.kind,
                InstKind::Atom(
                    Atom::Gate { .. } | Atom::Not { .. } | Atom::Buf { .. } | Atom::Tri { .. }
                )
            )
        })
        .map(|i| i.id.clone())
        .collect()
}

/// Evaluate a candidate design at a fixed time unit (with the [`recheck`] of clean mismatches).
pub fn evaluate(base: &Generated, design: Design, cfg: &HarnessConfig) -> Outcome {
    let mut g = base.clone();
    g.design = design;
    match prepare_with(base.seed, cfg, true, &|_| g.clone()) {
        Ok(p) => {
            let mut o = finish(p, cfg, &mut |_, _, _, _| {});
            recheck(&g, &mut o, cfg);
            o
        }
        Err(o) => *o,
    }
}

/// Delta debugging: shrink `g` (fully inlined first) by replacing gates with constants while
/// `keep(outcome)` stays true. Returns the reduced design and the number of evaluations.
/// `budget` bounds the evaluations.
pub fn minimize(
    g: &Generated,
    cfg: &HarnessConfig,
    keep: &dyn Fn(&Outcome) -> bool,
    budget: usize,
) -> (Generated, usize) {
    let mut cur = g.clone();
    cur.design = panda_circuits::inline_all(&g.design);
    let mut evals = 0;
    let try_design = |d: Design, evals: &mut usize| -> bool {
        *evals += 1;
        keep(&evaluate(g, d, cfg))
    };
    if !try_design(cur.design.clone(), &mut evals) {
        return (cur, evals);
    }
    // ddmin over the gate list, constant 0 first, then constant 1 for the survivors.
    for one in [false, true] {
        let mut chunk = gate_ids(&cur.design).len().div_ceil(2).max(1);
        loop {
            let ids = gate_ids(&cur.design);
            let mut progress = false;
            let mut i = 0;
            while i < ids.len() && evals < budget {
                let mut d = cur.design.clone();
                let mut changed = false;
                for id in &ids[i..(i + chunk).min(ids.len())] {
                    changed |= constantize(&mut d, id, one);
                }
                if changed && try_design(d.clone(), &mut evals) {
                    cur.design = d;
                    progress = true;
                }
                i += chunk;
            }
            if evals >= budget {
                break;
            }
            if !progress {
                if chunk == 1 {
                    break;
                }
                chunk = chunk.div_ceil(2);
            }
        }
    }
    (cur, evals)
}

/// Search for clean-but-mismatch variants of a design: `tries` times, replace a random fraction
/// of the gates of the inlined design by constants and evaluate at the design's time unit (the
/// mutation may remove the hazard STA flagged while keeping a mismatch that had another cause).
/// Returns the first variant classified [`Class::CleanMismatch`](crate::harness::Class).
pub fn hunt(
    g: &Generated,
    cfg: &HarnessConfig,
    tries: usize,
    seed: u64,
) -> Option<(Generated, Outcome)> {
    let base = panda_circuits::inline_all(&g.design);
    let ids = gate_ids(&base);
    if ids.is_empty() {
        return None;
    }
    let mut rng = crate::rng::Rng::new(seed ^ 0x4855_4E54);
    for _ in 0..tries {
        let mut d = base.clone();
        let pct = *rng.pick(&[2u32, 5, 10, 20]);
        let mut changed = false;
        for id in &ids {
            if rng.chance(pct) {
                changed |= constantize(&mut d, id, rng.chance(50));
            }
        }
        if !changed {
            continue;
        }
        let o = evaluate(g, d.clone(), cfg);
        if o.class == crate::harness::Class::CleanMismatch {
            let mut v = g.clone();
            v.design = d;
            return Some((v, o));
        }
    }
    None
}
