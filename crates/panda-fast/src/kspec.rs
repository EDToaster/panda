//! Kernel phase specialization (SPEC §8.5, docs/M4-CONTRACT.md "Kernel ops").
//!
//! # State reconstruction
//!
//! For a kernel with a table whose states have binary values on every output net (a state may
//! hold `Z`, e.g. a tri-state driver; such kernels get no class), [`KClass::recon`] maps `(output bits, input vector v)` to the
//! unique stable state under `v` with those outputs (if unique). A kernel whose last transition
//! was to input vector `v` is in a stable state under `v`; if that state is the one `recon`
//! names, the engine need not keep the kernel's internal nets up to date: they are a function of
//! its (always up-to-date) output nets and its inputs, and are materialized on demand
//! ("lazy" kernels).
//!
//! # Clock tuples and pure inputs
//!
//! A kernel input is *pure* if its combinational cone reaches only timed sources (clocks,
//! power-on reset) and binary constants: its value is a function of the clock tuple. An input
//! that depends on a timed source *and* on anything else (a gated clock) is *timed-impure*;
//! kernels with such inputs are never specialized. For the others, in round 0 of a clock event
//! only pure inputs change, and in every later round (and in input events) only the other inputs
//! change. So every transition of a kernel `K` in phase `c` (clock tuple `c`, pure bits `P`)
//! from input vector `u` (its previous transition's inputs; the state `s` is stable under `u`) to
//! `v` (pure bits `P`) is one of
//!
//! * *entry*: `u` has the pure bits of a predecessor tuple of `c` (from the clock schedule) and
//!   the same other bits as `v`;
//! * *data*: `u` has pure bits `P` and differs from `v`.
//!
//! # Classes
//!
//! For each `(K, c)` the pass checks every such `(s, u, v)` against the table:
//!
//! * the entry `T(s, v)` is binary;
//! * [`PhaseClass::Opaque`]: `T(s, v)` has the outputs of `s`, and `recon` names it;
//! * [`PhaseClass::Transparent`]: `T(s, v)` is the same state for every `(s, u)` (a function of
//!   `v` alone), and `recon` names it; its output bits form the `tout` table.
//!
//! Order sensitivity (the `DelaySensitive` diagnostic) does not change the result, and *data*
//! transitions of an O/T phase are never order-sensitive: applying the changed inputs one at a
//! time (FastRef's order check settles after each change) walks through single-input data
//! transitions `(s', u', u'')` from states stable under `u'`, all of which the classification
//! checked, so every step keeps the outputs and lands on `recon(outputs, u'')` (Opaque) or on
//! the state `g(u'')` that depends only on the inputs (Transparent); every order therefore ends
//! in the simultaneous result `T(s, v)`. Only *entry* transitions (two or more pure inputs
//! changing at one clock event) can be sensitive. If every such `(s, u, v)` with two or more
//! changed inputs is known to be insensitive (exact per `(s, u, v)` from the characterization),
//! an opaque kernel is skipped with word-level bitset ops ([`crate::ir::Program::tuple_skip`]);
//! otherwise it takes the per-kernel fast path, which runs FastRef's exact runtime check when
//! two or more inputs change ([`crate::ir::KFast::check_order`]).
//!
//! These are exactly the conditions under which the specialized ops reproduce the table's
//! result for every single transition (the engine additionally requires the kernel to be in a
//! binary table state reached under binary inputs; otherwise it takes the general path).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use panda_compile::{KernelTable, Node};
use panda_netlist::flat::{FlatNetlist, FlatOp, SourceKind};
use panda_netlist::{Logic, Termination};

use crate::compile::{CompileCtx, Pass};
use crate::ir::{
    KClass, KFast, KIN_DATA, KIN_KERNEL_MASK, KIN_POS_MASK, KIN_POS_SHIFT, KIN_TRACK, KernelOpKind,
    NONE, PhaseClass, VACT_OPAQUE,
};
use crate::schedule::Timed;

/// Maximum timed sources for phase specialization (`2^6` tuples).
pub const MAX_TIMED: usize = 6;
/// Maximum `k + outputs` bits of a reconstruction table.
pub const MAX_RECON_BITS: u32 = 18;
/// Work budget (table entries checked) per classification.
const CLASSIFY_BUDGET: u64 = 40_000_000;

/// Dependence of a net on the timed sources.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dep {
    /// A function of the clock tuple: bit `c` of the mask = value under tuple `c`.
    Pure(u64),
    /// Independent of the timed sources.
    Data,
    /// Depends on timed sources and on something else.
    TimedImpure,
}

/// Builds [`KClass`]es and the per-tuple kernel classes.
pub struct KernelSpecPass;

impl Pass for KernelSpecPass {
    fn name(&self) -> &'static str {
        "kspec"
    }

    fn run(&self, cx: &mut CompileCtx<'_>) -> Result<(), String> {
        if cx.program.kernel_ops.len() != cx.ks.kernels.len() {
            return Err("requires the kernels pass".into());
        }
        if !cx.opts.specialize {
            return Ok(());
        }
        build_classes(cx);
        let timed = timed_sources(cx.flat);
        let deps = (timed.len() <= MAX_TIMED).then(|| net_deps(cx.flat, &cx.ks.comb_order, &timed));
        let mut track = match &deps {
            Some(deps) => classify_tuples(cx, &timed, deps),
            None => Vec::new(),
        };
        // Per-vector paths need the flipped bits: every kernel with per-vector fast entries that
        // some tuple leaves general.
        let prog = &mut cx.program;
        let words = prog.kernel_words as usize;
        let nt = prog.num_tuples as usize;
        for k in 0..prog.kernel_ops.len() {
            if prog.vfast[k / 64] & (1 << (k % 64)) == 0 {
                continue;
            }
            let always_tuple = nt > 0
                && (0..nt).all(|t| {
                    let w = t * words + k / 64;
                    (prog.tuple_skip[w] | prog.tuple_fast[w]) & (1 << (k % 64)) != 0
                });
            if !always_tuple {
                track.push(k as u32);
            }
        }
        finalize_kin(prog, deps.as_deref(), track);
        let nk = prog.kernel_ops.len();
        prog.ktout = vec![NONE; nk * nt];
        prog.kpure = vec![0; nk * nt];
        for k in 0..nk {
            for t in 0..nt {
                let kf = prog.kernel_fast[k * nt + t];
                prog.ktout[t * nk + k] = kf.tout;
                prog.kpure[t * nk + k] = kf.pure;
            }
        }
        prog.tmasks = (0..nt * words)
            .map(|i| {
                [
                    prog.tuple_skip[i],
                    prog.tuple_fast[i],
                    prog.tuple_check[i],
                    prog.tuple_copaque[i],
                ]
            })
            .collect();
        prog.touts8 = if prog.touts.iter().all(|&t| t == NONE || t < 0xFF) {
            prog.touts
                .iter()
                .map(|&t| if t == NONE { 0xFF } else { t as u8 })
                .collect()
        } else {
            Vec::new()
        };
        prog.touts2 = if prog.touts.iter().all(|&t| t == NONE || t <= 1) {
            let mut w = vec![0u64; prog.touts.len().div_ceil(32)];
            for (i, &t) in prog.touts.iter().enumerate() {
                let code = if t == NONE { 2 } else { u64::from(t) };
                w[i / 32] |= code << (2 * (i % 32));
            }
            w
        } else {
            Vec::new()
        };
        prog.vfast_used = (0..nk).any(|k| {
            prog.vfast[k / 64] & (1 << (k % 64)) != 0
                && (nt == 0
                    || (0..nt).any(|t| {
                        let w = t * words + k / 64;
                        (prog.tuple_skip[w] | prog.tuple_fast[w]) & (1 << (k % 64)) == 0
                    }))
        });
        Ok(())
    }
}

fn build_classes(cx: &mut CompileCtx<'_>) {
    let prog = &mut cx.program;
    prog.vfast = vec![0; prog.kernel_words as usize];
    let mut memo: HashMap<(u32, Vec<u32>), u32> = HashMap::new();
    for k in 0..prog.kernel_ops.len() {
        let kop = prog.kernel_ops[k];
        let KernelOpKind::KTable { table } = kop.kind else {
            continue;
        };
        let outs = prog.kernel_out_indices(&kop).to_vec();
        if kop.num_inputs + outs.len() as u32 > MAX_RECON_BITS || outs.len() > 31 {
            continue;
        }
        let key = (table, outs.clone());
        let id = match memo.get(&key) {
            Some(&id) => id,
            None => {
                let t = prog.ktables[table as usize].table.clone();
                // Output bits encode 0/1 only: a state with a Z output (tri-state driver)
                // disqualifies the class.
                if t.states
                    .iter()
                    .any(|st| outs.iter().any(|&i| !st[i as usize].is_known()))
                {
                    continue;
                }
                let class = make_class(table, &t, &outs);
                prog.kclasses.push(class);
                let id = prog.kclasses.len() as u32 - 1;
                memo.insert(key, id);
                id
            }
        };
        prog.kernel_class[k] = id;
        if prog.kclasses[id as usize].vact.iter().any(|&a| a != NONE) {
            prog.vfast[k / 64] |= 1 << (k % 64);
        }
        let nets = prog.kernel_state_nets(&kop).to_vec();
        for (j, &net) in nets.iter().enumerate() {
            if !outs.contains(&(j as u32)) {
                prog.lazy_owner[net as usize] = 2 * k as u32 + 1;
            }
        }
    }
}

fn make_class(table: u32, t: &KernelTable, outs: &[u32]) -> KClass {
    let k = t.num_inputs as u32;
    let out_bits: Vec<u32> = t
        .states
        .iter()
        .map(|st| {
            outs.iter().enumerate().fold(0u32, |a, (j, &i)| {
                a | u32::from(st[i as usize] == Logic::One) << j
            })
        })
        .collect();
    const AMBIG: u32 = NONE - 1;
    let mut recon = vec![NONE; 1usize << (k as usize + outs.len())];
    for (v, states) in t.states_by_input.iter().enumerate() {
        for &s in states {
            let key = ((out_bits[s as usize] as usize) << k) | v;
            recon[key] = match recon[key] {
                NONE => s,
                x if x == s => s,
                _ => AMBIG,
            };
        }
    }
    for r in &mut recon {
        if *r == AMBIG {
            *r = NONE;
        }
    }
    let mut class = KClass {
        table,
        num_inputs: k,
        num_outs: outs.len() as u32,
        out_bits,
        recon,
        vact: Vec::new(),
    };
    class.vact = vector_actions(&class, t);
    class
}

/// Per-vector actions ([`KClass::vact`]): for every input vector `v`, check every single-bit
/// predecessor `u = v ^ (1 << i)` and every state `s` stable under `u` (a kernel whose last
/// transition was to `u` is in such a state): if `T(s, v)` is binary, named by the
/// reconstruction table, and either always keeps the outputs of `s` ([`VACT_OPAQUE`]) or is
/// always the same state (its output bits), the transition `u → v` needs no table step. A
/// single changed input can never be order-sensitive.
fn vector_actions(cl: &KClass, t: &KernelTable) -> Vec<u32> {
    let k = cl.num_inputs;
    (0..1u32 << k)
        .map(|v| {
            let (mut opaque, mut transp, mut g) = (true, true, NONE);
            for i in 0..k {
                let u = v ^ (1 << i);
                for &s in &t.states_by_input[u as usize] {
                    let Some(e) = t.transition(s, v).next else {
                        return NONE;
                    };
                    let ob = cl.out_bits[e as usize];
                    if cl.recon(ob, v) != e {
                        return NONE;
                    }
                    opaque &= ob == cl.out_bits[s as usize];
                    if g == NONE {
                        g = e;
                    }
                    transp &= g == e;
                }
            }
            if g == NONE {
                NONE // no reachable configuration
            } else if opaque {
                VACT_OPAQUE
            } else if transp {
                cl.out_bits[g as usize]
            } else {
                NONE
            }
        })
        .collect()
}

/// Timed sources in tuple-bit order.
fn timed_sources(flat: &FlatNetlist) -> Vec<(u32, Timed)> {
    flat.sources
        .iter()
        .filter_map(|s| match s.kind {
            SourceKind::Clock {
                period,
                high,
                phase,
            } => Some((
                s.output,
                Timed::Clock {
                    period: u64::from(period),
                    high: u64::from(high),
                    phase: u64::from(phase),
                },
            )),
            SourceKind::PowerOnReset { ticks } => Some((
                s.output,
                Timed::PowerOnReset {
                    ticks: u64::from(ticks),
                },
            )),
            _ => None,
        })
        .collect()
}

/// Per net: dependence on the timed sources (see [`Dep`]).
fn net_deps(flat: &FlatNetlist, comb: &[Node], timed: &[(u32, Timed)]) -> Vec<Dep> {
    let n = flat.num_nets as usize;
    let nt = 1u32 << timed.len();
    let all = if nt == 64 { !0u64 } else { (1u64 << nt) - 1 };
    // Default: driven by kernels / top inputs / X constants: data.
    let mut dep = vec![Dep::Data; n];
    let mut driven = vec![false; n];
    for g in &flat.gates {
        driven[g.output as usize] = true;
    }
    for r in &flat.resolvers {
        driven[r.output as usize] = true;
    }
    for s in &flat.sources {
        driven[s.output as usize] = true;
        if let SourceKind::Constant(v) = s.kind {
            dep[s.output as usize] = match v {
                Logic::Zero => Dep::Pure(0),
                Logic::One => Dep::Pure(all),
                _ => Dep::Data,
            };
        }
    }
    for net in 0..n {
        if !driven[net] {
            let term = flat
                .undriven_termination
                .get(&(net as u32))
                .copied()
                .unwrap_or(Termination::None);
            dep[net] = match term {
                Termination::PullUp => Dep::Pure(all),
                Termination::PullDown => Dep::Pure(0),
                _ => Dep::Data,
            };
        }
    }
    for (i, &(net, _)) in timed.iter().enumerate() {
        let mut m = 0u64;
        for c in 0..nt {
            if (c >> i) & 1 == 1 {
                m |= 1 << c;
            }
        }
        dep[net as usize] = Dep::Pure(m);
    }
    for &node in comb {
        let (ins, out, op): (&[u32], u32, Option<FlatOp>) = match node {
            Node::Gate(g) => {
                let gate = &flat.gates[g as usize];
                (&gate.inputs, gate.output, Some(gate.op))
            }
            Node::Resolver(r) => {
                let res = &flat.resolvers[r as usize];
                (&res.drivers, res.output, None)
            }
        };
        let timed_dep = ins.iter().any(|&x| match dep[x as usize] {
            Dep::Pure(_) => true, // conservatively: constants count as timed-free below
            Dep::TimedImpure => true,
            Dep::Data => false,
        });
        let all_pure = ins.iter().all(|&x| matches!(dep[x as usize], Dep::Pure(_)));
        let d = match op {
            Some(op) if op != FlatOp::Tri && all_pure && !ins.is_empty() => {
                let mut it = ins.iter().map(|&x| match dep[x as usize] {
                    Dep::Pure(m) => m,
                    _ => unreachable!(),
                });
                let v = match op {
                    FlatOp::And => it.fold(all, |a, b| a & b),
                    FlatOp::Nand => !it.fold(all, |a, b| a & b),
                    FlatOp::Or => it.fold(0, |a, b| a | b),
                    FlatOp::Nor => !it.fold(0, |a, b| a | b),
                    FlatOp::Xor => it.fold(0, |a, b| a ^ b),
                    FlatOp::Xnor => !it.fold(0, |a, b| a ^ b),
                    FlatOp::Not => !it.next().unwrap(),
                    FlatOp::Buf => it.next().unwrap(),
                    FlatOp::Tri => unreachable!(),
                };
                Dep::Pure(v & all)
            }
            _ => {
                // Mixed (or tri-state/resolver): timed-impure if it depends on a non-constant
                // timed source, else data.
                let depends_on_timed = timed_dep
                    && ins.iter().any(|&x| match dep[x as usize] {
                        Dep::Pure(m) => m != 0 && m != all,
                        Dep::TimedImpure => true,
                        Dep::Data => false,
                    });
                if depends_on_timed {
                    Dep::TimedImpure
                } else {
                    Dep::Data
                }
            }
        };
        dep[out as usize] = d;
    }
    dep
}

/// Observed clock-tuple transitions `(previous, next)` of the schedule, or `None` if the
/// schedule could not be covered (then every pair is possible).
fn tuple_transitions(timed: &[(u32, Timed)]) -> Option<HashSet<(u32, u32)>> {
    let tuple_at = |t: u64| -> u32 {
        timed.iter().enumerate().fold(0u32, |a, (i, &(_, src))| {
            a | u32::from(src.value_at(t) == Logic::One) << i
        })
    };
    let mut hyper: u64 = 1;
    let mut por_end: u64 = 0;
    for &(_, src) in timed {
        match src {
            Timed::Clock { period, high, .. } if period > 0 && high > 0 && high < period => {
                let g = gcd(hyper, period);
                hyper = (hyper / g).checked_mul(period)?;
                if hyper > 1 << 24 {
                    return None;
                }
            }
            Timed::PowerOnReset { ticks } => por_end = por_end.max(ticks),
            _ => {}
        }
    }
    let limit = por_end.checked_add(hyper.checked_mul(2)?)?;
    let mut pairs = HashSet::new();
    let mut t = 0u64;
    let mut cur = tuple_at(0);
    let mut events = 0u32;
    while let Some(next) = timed
        .iter()
        .filter_map(|&(_, s)| s.next_change_after(t))
        .min()
    {
        if next > limit {
            break;
        }
        let nt = tuple_at(next);
        pairs.insert((cur, nt));
        cur = nt;
        t = next;
        events += 1;
        if events > 1_000_000 {
            return None;
        }
    }
    Some(pairs)
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Tuple classes of every kernel with a class; returns the kernels whose order checks need the
/// flipped input bits.
fn classify_tuples(cx: &mut CompileCtx<'_>, timed: &[(u32, Timed)], deps: &[Dep]) -> Vec<u32> {
    let lemma = cx.opts.data_order_lemma;
    let nt = 1u32 << timed.len();
    let pairs = tuple_transitions(timed);
    let preds_of = |c: u32| -> Vec<u32> {
        match &pairs {
            Some(p) => {
                let mut v: Vec<u32> = p
                    .iter()
                    .filter(|&&(_, b)| b == c)
                    .map(|&(a, _)| a)
                    .collect();
                v.sort_unstable();
                v
            }
            None => (0..nt).filter(|&a| a != c).collect(),
        }
    };
    let prog = &mut cx.program;
    let nk = prog.kernel_ops.len();
    let words = prog.kernel_words as usize;
    prog.timed_nets = timed.iter().map(|&(n, _)| n).collect();
    prog.num_tuples = nt;
    prog.tuple_skip = vec![0; nt as usize * words];
    prog.tuple_fast = vec![0; nt as usize * words];
    prog.tuple_check = vec![0; nt as usize * words];
    prog.tuple_copaque = vec![0; nt as usize * words];
    prog.kernel_fast = vec![
        KFast {
            tout: NONE,
            check_order: true,
            pure: 0,
        };
        nk * nt as usize
    ];
    prog.kernel_phase = vec![PhaseClass::General; nk * nt as usize];
    prog.kpure_mask = vec![0; nk];
    let preds: Vec<Vec<u32>> = (0..nt).map(preds_of).collect();
    type Key = (u32, u32, u32, Vec<u32>);
    let mut memo: HashMap<Key, (PhaseClass, bool, u32)> = HashMap::new();
    let mut track: Vec<u32> = Vec::new();
    for k in 0..nk {
        let class = prog.kernel_class[k];
        if class == NONE {
            continue;
        }
        let kop = prog.kernel_ops[k];
        let ins = prog.kernel_inputs(&kop).to_vec();
        if ins.iter().any(|&x| deps[x as usize] == Dep::TimedImpure) {
            continue;
        }
        let mut mask = 0u32;
        for (i, &x) in ins.iter().enumerate() {
            if let Dep::Pure(_) = deps[x as usize] {
                mask |= 1 << i;
            }
        }
        prog.kpure_mask[k] = mask;
        let pure_val = |c: u32| -> u32 {
            ins.iter()
                .enumerate()
                .fold(0u32, |a, (i, &x)| match deps[x as usize] {
                    Dep::Pure(m) => a | (((m >> c) & 1) as u32) << i,
                    _ => a,
                })
        };
        for c in 0..nt {
            let p = pure_val(c);
            let mut pp: Vec<u32> = preds[c as usize]
                .iter()
                .map(|&a| pure_val(a))
                .filter(|&x| x != p)
                .collect();
            pp.sort_unstable();
            pp.dedup();
            let key = (class, mask, p, pp);
            let (pc, certain, tout) = match memo.get(&key) {
                Some(&r) => r,
                None => {
                    let cl = &prog.kclasses[class as usize];
                    let t = prog.ktables[cl.table as usize].table.clone();
                    let (pc, certain, table) = classify(cl, &t, mask, key.2, &key.3, lemma);
                    let off = match table {
                        Some(tab) => {
                            let off = prog.touts.len() as u32;
                            prog.touts.extend_from_slice(&tab);
                            off
                        }
                        None => NONE,
                    };
                    memo.insert(key, (pc, certain, off));
                    (pc, certain, off)
                }
            };
            let w = c as usize * words + k / 64;
            let kt = k * nt as usize + c as usize;
            prog.kernel_phase[kt] = pc;
            if matches!(pc, PhaseClass::Opaque | PhaseClass::Transparent) && !certain {
                track.push(k as u32);
            }
            match pc {
                PhaseClass::Opaque if certain => prog.tuple_skip[w] |= 1 << (k % 64),
                PhaseClass::Opaque | PhaseClass::Transparent => {
                    prog.tuple_fast[w] |= 1 << (k % 64);
                    if !certain {
                        prog.tuple_check[w] |= 1 << (k % 64);
                        if pc == PhaseClass::Opaque {
                            prog.tuple_copaque[w] |= 1 << (k % 64);
                        }
                    }
                    prog.kernel_fast[kt] = KFast {
                        tout,
                        check_order: !certain,
                        pure: p,
                    };
                }
                PhaseClass::General => {}
            }
        }
    }
    track
}

/// Rebuild the kernel reader lists: flip tracking ([`KIN_TRACK`]) for `track`; pure nets are not
/// data inputs, and for untracked kernels their bits come from the tuple ([`KFast::pure`]), so
/// those readers are only marked pending, with word masks ([`crate::ir::Program::net_kpend`]).
fn finalize_kin(prog: &mut crate::ir::Program, deps: Option<&[Dep]>, mut track: Vec<u32>) {
    track.sort_unstable();
    track.dedup();
    let n = prog.num_nets as usize;
    let mut kin_start = vec![0u32; n + 1];
    let mut kin_items = Vec::with_capacity(prog.net_kin.items.len());
    let mut kp_start = vec![0u32; n + 1];
    let mut kp_items: Vec<(u32, u64)> = Vec::new();
    for net in 0..n {
        let pure = deps.is_some_and(|d| matches!(d[net], Dep::Pure(_)));
        let mut masks: Vec<(u32, u64)> = Vec::new();
        for &e in prog.net_kin.row(net as u32) {
            let k = e & KIN_KERNEL_MASK;
            let tracked = track.binary_search(&k).is_ok();
            let pos = (e >> KIN_POS_SHIFT) & KIN_POS_MASK;
            if pure && !tracked {
                if pos < 32 {
                    prog.kernel_kin_mask[k as usize] &= !(1 << pos);
                }
                match masks.iter_mut().find(|(w, _)| *w == k / 64) {
                    Some((_, m)) => *m |= 1 << (k % 64),
                    None => masks.push((k / 64, 1 << (k % 64))),
                }
                continue;
            }
            let mut e = e;
            if tracked {
                e |= KIN_TRACK;
            }
            if pure {
                e &= !KIN_DATA;
            }
            kin_items.push(e);
        }
        kp_items.extend(masks);
        kin_start[net + 1] = kin_items.len() as u32;
        kp_start[net + 1] = kp_items.len() as u32;
    }
    prog.net_kin = crate::ir::NetCsr {
        start: kin_start,
        items: kin_items,
    };
    prog.net_kpend = crate::ir::MaskCsr {
        start: kp_start,
        items: kp_items,
    };
}

/// Classify one kernel class under pure-input mask `mask` with pure bits `p` and predecessor
/// pure bits `preds` (see the module docs; `lemma`: data transitions never count against
/// certainty). Returns the class, whether every transition of the
/// phase is known to be order-insensitive, and, for transparent, the output table (`2^k`
/// entries).
fn classify(
    cl: &KClass,
    t: &Arc<KernelTable>,
    mask: u32,
    p: u32,
    preds: &[u32],
    lemma: bool,
) -> (PhaseClass, bool, Option<Vec<u32>>) {
    let k = cl.num_inputs;
    let full = (1u32 << k) - 1;
    let data = full & !mask;
    let mut opaque = true;
    let mut transp = true;
    let mut tout = vec![NONE; 1usize << k];
    let mut work = 0u64;
    let mut certain = true;
    // Enumerate v with pure bits p.
    let mut dv = 0u32;
    loop {
        let v = dv | p;
        let mut g = NONE;
        let mut check = |u: u32, entry: bool, opaque: &mut bool, transp: &mut bool| -> bool {
            for &s in &t.states_by_input[u as usize] {
                work += 1;
                let tr = t.transition(s, v);
                let Some(e) = tr.next else {
                    return false;
                };
                // Data transitions of an O/T phase are order-insensitive by construction (see
                // the module docs); only entry transitions with several flipped (pure) inputs
                // can make the phase uncertain.
                if (entry || !lemma)
                    && (u ^ v).count_ones() >= 2
                    && t.order_sensitivity_known(s, u, v) != Some(false)
                {
                    certain = false;
                }
                let ob_e = cl.out_bits[e as usize];
                let named = cl.recon(ob_e, v) == e;
                if *opaque && (ob_e != cl.out_bits[s as usize] || !named) {
                    *opaque = false;
                }
                if *transp {
                    if g == NONE {
                        g = e;
                    }
                    if g != e || !named {
                        *transp = false;
                    }
                }
                if !*opaque && !*transp {
                    return false;
                }
            }
            true
        };
        let mut ok = true;
        for &q in preds {
            let u = (v & data) | q;
            if !check(u, true, &mut opaque, &mut transp) {
                ok = false;
                break;
            }
        }
        if ok {
            let mut du = 0u32;
            loop {
                let u = du | p;
                if u != v && !check(u, false, &mut opaque, &mut transp) {
                    ok = false;
                    break;
                }
                du = du.wrapping_sub(data) & data;
                if du == 0 {
                    break;
                }
            }
        }
        if !ok || work > CLASSIFY_BUDGET {
            return (PhaseClass::General, false, None);
        }
        if g != NONE {
            tout[v as usize] = cl.out_bits[g as usize];
        }
        dv = dv.wrapping_sub(data) & data;
        if dv == 0 {
            break;
        }
    }
    if opaque {
        (PhaseClass::Opaque, certain, None)
    } else if transp {
        (PhaseClass::Transparent, certain, Some(tout))
    } else {
        (PhaseClass::General, false, None)
    }
}
