//! The bytecode interpreter: partition sweeps, LUT ops and change propagation.
//!
//! A partition runs its interpreter ops ([`crate::ir::HOp`]: 1/2-input gates with embedded
//! tables, fused 4-state truth-table ops, and the rest via [`Op`]); every output change is
//! recorded as a change entry (`record offset << 2 | old value code`) and the partition's
//! entries are propagated in one batch at its end ([`Marks::flush`]): marks only go to later
//! partitions and kernel marks are only read at round boundaries, so batching is exact. The hot
//! loops index unchecked: `VerifyPass` proved every index in range.

use panda_netlist::Logic;
use panda_netlist::eval::{lookup2, resolve};

use crate::compile::REC_ESCAPE;
use crate::ir::{HOp, KIN_KERNEL_MASK, KIN_POS_MASK, KIN_POS_SHIFT, LutDesc, NONE, Op, Program};

/// Change-propagation state written by every op whose output changes.
pub(crate) struct Marks {
    /// Dirty partitions (bitset).
    pub dirty: Vec<u64>,
    /// Kernels with an input change since the last round boundary (bitset).
    pub kpending: Vec<u64>,
    /// Kernels allowed to take the specialized paths (bitset; see `FastSim`): cleared whenever an
    /// input of the kernel becomes `X`/`Z`.
    pub fastok: Vec<u64>,
    /// Boundary log (slot [`Program::net_log`]): for a net read by kernels that changed since the
    /// last round boundary, its value at that boundary (valid iff `bepoch[slot] == epoch`).
    pub bval: Vec<Logic>,
    /// See [`Marks::bval`].
    pub bepoch: Vec<u32>,
    /// Current inter-boundary interval.
    pub epoch: u32,
    /// Per kernel: bit `i` = input `i` is `1` (inputs ≥ 32 not tracked). Equal to the binary
    /// input index whenever the kernel's inputs are binary.
    pub kin: Vec<u32>,
    /// Per tracked kernel ([`Program::tracked`]): its `kin` at the last round boundary (each
    /// input net changes at most once per interval, so `kin ^ kin_prev` are the flipped bits).
    pub kin_prev: Vec<u32>,
    /// Kernels with a data-input change (or an output write outside a transparent evaluation)
    /// since their last transparent evaluation (bitset).
    pub ddirty: Vec<u64>,
}

impl Marks {
    /// Record that `net` changed from `old` to its current value in `values`: mark reader
    /// partitions dirty and reader kernels pending, log the boundary value, and drop reader
    /// kernels from the fast paths if the new value is not binary.
    #[inline]
    pub fn notify(&mut self, prog: &Program, values: &[Logic], net: u32, old: Logic) {
        let off = prog.notify_off[net as usize];
        if off != NONE {
            self.flush(prog, values, &[change(off, old)]);
        }
    }

    /// [`Marks::notify`] for a batch of changes ([`change`] entries naming the changed nets'
    /// records; each net at most once; `values` holds the new values).
    #[inline(never)]
    pub fn flush(&mut self, prog: &Program, values: &[Logic], chg: &[u32]) {
        crate::probe::count_n(crate::probe::NOTIFIES, chg.len() as u64);
        let tf = crate::probe::start();
        self.flush_inner(prog, values, chg);
        crate::probe::stop(crate::probe::FLUSH, tf);
    }

    #[inline(always)]
    fn flush_inner(&mut self, prog: &Program, values: &[Logic], chg: &[u32]) {
        // Raw walk over the verified records (`verify_notify`, run by `VerifyPass` on every
        // program a `FastSim` executes): a change entry names a record offset (from the
        // verified record arrays or `notify_off`), the record lies inside `notify_rec`, its net
        // is < num_nets = `values.len()`, partition words < `dirty.len()`, kernel words < the
        // kernel bitset lengths, kernel ids < `kin.len()`, input positions < the kernel's input
        // count (or 32 = no bit), log slots < `bval.len()` = `bepoch.len()`.
        debug_assert_eq!(self.dirty.len(), prog.partitions.len().div_ceil(64));
        debug_assert_eq!(self.kin.len(), prog.kernel_ops.len());
        debug_assert_eq!(self.bval.len(), prog.num_log as usize);
        let rec = prog.notify_rec.as_ptr();
        let dirty = self.dirty.as_mut_ptr();
        let kpending = self.kpending.as_mut_ptr();
        let ddirty = self.ddirty.as_mut_ptr();
        let fastok = self.fastok.as_mut_ptr();
        let kin = self.kin.as_mut_ptr();
        let epoch = self.epoch;
        let m64 = |lo: u32, hi: u32| u64::from(lo) | u64::from(hi) << 32;
        for &c in chg {
            debug_assert!(((c >> 2) as usize) + 3 <= prog.notify_rec.len());
            // SAFETY: see above.
            unsafe {
                let mut r = rec.add((c >> 2) as usize);
                let h = *r;
                let net = *r.add(1) as usize;
                let log = *r.add(2);
                debug_assert!(net < values.len());
                let (np, nk, nkin) = if h != REC_ESCAPE {
                    r = r.add(3);
                    (h & 0xFF, (h >> 8) & 0xFF, h >> 16)
                } else {
                    // Escaped header: full-word counts (huge fan-out).
                    let c = (*r.add(3), *r.add(4), *r.add(5));
                    r = r.add(6);
                    c
                };
                for _ in 0..np {
                    debug_assert!((*r as usize) < self.dirty.len());
                    *dirty.add(*r as usize) |= m64(*r.add(1), *r.add(2));
                    r = r.add(3);
                }
                // Input-index entries come with kernel groups (verified).
                if nk == 0 {
                    continue;
                }
                // At most one change per net per interval (each op runs at most once per
                // sweep, kernel nets are written once per boundary), so `old` is the boundary
                // value.
                let old = Logic::from_code((c & 3) as u8);
                let new = *values.get_unchecked(net);
                if log != NONE {
                    debug_assert!((log as usize) < self.bval.len());
                    *self.bval.get_unchecked_mut(log as usize) = old;
                    *self.bepoch.get_unchecked_mut(log as usize) = epoch;
                }
                let nb = new.code() >= 2;
                for _ in 0..nk {
                    let w = *r as usize;
                    debug_assert!(w < self.kpending.len());
                    let pend = m64(*r.add(1), *r.add(2));
                    *kpending.add(w) |= pend;
                    *ddirty.add(w) |= m64(*r.add(3), *r.add(4));
                    if nb {
                        *fastok.add(w) &= !pend;
                    }
                    r = r.add(5);
                }
                let flip = u64::from((old == Logic::One) != (new == Logic::One));
                for i in 0..nkin as usize {
                    let e = *r.add(i);
                    let k = (e & KIN_KERNEL_MASK) as usize;
                    debug_assert!(k < self.kin.len());
                    // Position 32 (untracked) shifts the bit out of the low word.
                    *kin.add(k) ^= (flip << ((e >> KIN_POS_SHIFT) & KIN_POS_MASK)) as u32;
                }
            }
        }
    }

    /// A round boundary: start a new interval (the log entries of the previous one are dead).
    #[inline]
    pub fn next_epoch(&mut self, prog: &Program) {
        for &k in &prog.tracked_list {
            self.kin_prev[k as usize] = self.kin[k as usize];
        }
        self.epoch += 1;
        if self.epoch == u32::MAX {
            // `u32::MAX` is the "never logged" sentinel.
            self.bepoch.fill(u32::MAX);
            self.epoch = 0;
        }
    }

    /// Recompute every kernel's input bits from `values` (after writing values directly).
    pub fn init_kin(&mut self, prog: &Program, values: &[Logic]) {
        for (k, kop) in prog.kernel_ops.iter().enumerate() {
            self.kin[k] = prog
                .kernel_inputs(kop)
                .iter()
                .take(32)
                .enumerate()
                .fold(0u32, |a, (i, &n)| {
                    a | u32::from(values[n as usize] == Logic::One) << i
                })
                & prog.kernel_kin_mask[k];
        }
        self.kin_prev.copy_from_slice(&self.kin);
    }

    /// Bits of kernel `k`'s input index that flipped since the last round boundary (tracked
    /// kernels only, see [`crate::ir::KIN_TRACK`]).
    #[inline]
    pub fn flipped(&self, k: u32) -> u32 {
        self.kin[k as usize] ^ self.kin_prev[k as usize]
    }

    /// Value of `net` at the last round boundary.
    #[inline]
    pub fn boundary_value(&self, prog: &Program, values: &[Logic], net: u32) -> Logic {
        let log = prog.net_log[net as usize];
        if log != crate::ir::NONE && self.bepoch[log as usize] == self.epoch {
            self.bval[log as usize]
        } else {
            values[net as usize]
        }
    }
}

/// Mutable runtime state touched by the interpreter (besides the net values).
pub(crate) struct Rt {
    /// Change propagation.
    pub marks: Marks,
    /// LUTs whose internal nets are stale (last evaluated from the table).
    pub lut_stale: Vec<u64>,
    /// Per LUT: the entry its outputs currently hold (last table evaluation), or `NONE` after a
    /// cold evaluation / reset.
    pub lut_last: Vec<u32>,
    /// Per comb resolver: its last evaluation saw contention.
    pub res_contending: Vec<bool>,
    /// Per comb resolver: evaluated during the current event.
    pub res_touched_mark: Vec<bool>,
    /// Comb resolvers evaluated during the current event.
    pub res_touched: Vec<u32>,
    /// Partitions evaluated (counter).
    pub parts_run: u64,
    /// Ops executed (counter; a LUT counts as one op).
    pub ops_run: u64,
    buf: Vec<Logic>,
    /// Changes not yet propagated (`net << 2 | old code`): the current partition's, or a batch of
    /// source / kernel-output writes.
    pub chg: Vec<u32>,
    /// Most output changes one partition can record (its gate ops plus every output and cold-code
    /// net of its LUTs).
    pub max_chg: usize,
}

impl Rt {
    pub fn new(prog: &Program) -> Rt {
        let words = |n: usize| vec![0u64; n.div_ceil(64)];
        let nr = prog.resolvers.len();
        let max_chg = (0..prog.partitions.len())
            .map(|p| {
                prog.partition_ops(p)
                    .iter()
                    .map(|op| match *op {
                        Op::Lut { lut } => {
                            let l = &prog.luts[lut as usize];
                            l.num_outputs as usize + l.cold.len()
                        }
                        _ => 1,
                    })
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(0);
        Rt {
            marks: Marks {
                dirty: words(prog.partitions.len()),
                kpending: words(prog.kernel_ops.len()),
                fastok: words(prog.kernel_ops.len()),
                bval: vec![Logic::X; prog.num_log as usize],
                bepoch: vec![u32::MAX; prog.num_log as usize],
                epoch: 0,
                kin: vec![0; prog.kernel_ops.len()],
                kin_prev: vec![0; prog.kernel_ops.len()],
                ddirty: words(prog.kernel_ops.len()),
            },
            lut_stale: words(prog.luts.len()),
            lut_last: vec![crate::ir::NONE; prog.luts.len()],
            res_contending: vec![false; nr],
            res_touched_mark: vec![false; nr],
            res_touched: Vec::new(),
            parts_run: 0,
            ops_run: 0,
            buf: Vec::new(),
            chg: Vec::with_capacity(max_chg),
            max_chg,
        }
    }
}

#[inline(always)]
pub(crate) fn set_bit(bits: &mut [u64], i: u32) {
    bits[(i >> 6) as usize] |= 1u64 << (i & 63);
}

#[inline(always)]
pub(crate) fn clear_bit(bits: &mut [u64], i: u32) {
    bits[(i >> 6) as usize] &= !(1u64 << (i & 63));
}

#[inline(always)]
pub(crate) fn test_bit(bits: &[u64], i: u32) -> bool {
    bits[(i >> 6) as usize] & (1u64 << (i & 63)) != 0
}

/// Iterate the set bits of a bitset word `w` (index `wi`), calling `f(bit index)`.
#[inline(always)]
pub(crate) fn for_bits(mut w: u64, wi: usize, mut f: impl FnMut(u32)) {
    while w != 0 {
        let b = w.trailing_zeros();
        w &= w - 1;
        f((wi * 64) as u32 + b);
    }
}

/// Write a net from outside the partitions (sources, kernel states); on change mark its readers.
#[inline]
pub(crate) fn set_net(prog: &Program, values: &mut [Logic], marks: &mut Marks, net: u32, v: Logic) {
    let old = values[net as usize];
    if old != v {
        values[net as usize] = v;
        marks.notify(prog, values, net, old);
    }
}

/// A change entry: record offset `off` (< 2^30, checked by `verify_notify`) and the old value.
#[inline(always)]
pub(crate) fn change(off: u32, old: Logic) -> u32 {
    off << 2 | u32::from(old.code())
}

/// `values[net] = v`, recording a change in `chg` (propagated by a later [`Marks::flush`]) if
/// the net has a change-propagation record `off` (else [`NONE`]).
#[inline(always)]
pub(crate) fn set_net_deferred(
    values: &mut [Logic],
    chg: &mut Vec<u32>,
    net: u32,
    off: u32,
    v: Logic,
) {
    let old = std::mem::replace(&mut values[net as usize], v);
    if old != v && off != NONE {
        chg.push(change(off, old));
    }
}

/// Propagate the batched changes in `rt.chg`.
#[inline]
pub(crate) fn flush_deferred(prog: &Program, values: &[Logic], rt: &mut Rt) {
    if !rt.chg.is_empty() {
        rt.marks.flush(prog, values, &rt.chg);
        rt.chg.clear();
    }
}

/// Evaluate dirty partitions in index (= topological) order until none is dirty. Marks only go to
/// later partitions, so one forward scan suffices.
pub(crate) fn sweep(prog: &Program, values: &mut [Logic], rt: &mut Rt) {
    let mut chg = std::mem::take(&mut rt.chg);
    debug_assert!(chg.is_empty());
    chg.reserve(rt.max_chg);
    for w in 0..rt.marks.dirty.len() {
        loop {
            let bits = rt.marks.dirty[w];
            if bits == 0 {
                break;
            }
            rt.marks.dirty[w] = bits & (bits - 1);
            let p = w * 64 + bits.trailing_zeros() as usize;
            run_partition(prog, values, rt, &mut chg, p);
        }
    }
    rt.chg = chg;
}

/// `values[out] = v`; if the output has a change-propagation record `rec` (else [`NONE`]) and
/// the value changed, record the change in `chg`.
#[inline(always)]
fn store(values: &mut [Logic], chg: &mut Vec<u32>, out: u32, v: Logic, rec: u32) {
    debug_assert!((out as usize) < values.len());
    // SAFETY: `VerifyPass` (always run last by `compile_with`; every program a `FastSim` runs
    // comes from it) checked every op output < num_nets = values.len().
    let slot = unsafe { values.get_unchecked_mut(out as usize) };
    let old = *slot;
    *slot = v;
    if rec != NONE && old != v {
        chg.push(change(rec, old));
    }
}

/// Value of a 1/2-input table op.
#[inline(always)]
fn gate2(
    tables: &[u32; crate::ir::table::COUNT],
    values: &[Logic],
    t: u8,
    a: u32,
    b: u32,
) -> Logic {
    debug_assert!((a as usize) < values.len() && (b as usize) < values.len());
    debug_assert!((t as usize) < tables.len());
    // SAFETY: `VerifyPass` checked every operand net < num_nets = values.len() and every table
    // selector < table::COUNT = tables.len().
    unsafe {
        lookup2(
            *tables.get_unchecked(t as usize),
            *values.get_unchecked(a as usize),
            *values.get_unchecked(b as usize),
        )
    }
}

/// Value of an N-input fold op.
#[inline(always)]
fn gate_n(prog: &Program, values: &[Logic], fold: u8, invert: bool, args: u32, n: u32) -> Logic {
    // Branchless N-input fold: `seen` has bit `c` set if some input has code `c`, `parity` is the
    // XOR of all codes. Equal to folding and2/or2/xor2 (Z reads as X; a controlling 0 (AND) /
    // 1 (OR) wins over X; any X/Z makes XOR X).
    let mut seen = 0u32;
    let mut parity = 0u8;
    for &x in &prog.args[args as usize..(args + n) as usize] {
        debug_assert!((x as usize) < values.len());
        // SAFETY: `VerifyPass` checked every operand net < num_nets = values.len().
        let c = unsafe { values.get_unchecked(x as usize) }.code();
        seen |= 1 << c;
        parity ^= c;
    }
    let v = fold_result(fold, seen, parity);
    // not1 when inverting: 0 ↔ 1, X stays X.
    let v = v ^ (u8::from(invert) & !(v >> 1));
    Logic::from_code(v)
}

/// Run a LUT's cold gate code, recording changes of the nets that have readers in `chg`.
fn run_cold(prog: &Program, values: &mut [Logic], chg: &mut Vec<u32>, l: &LutDesc) {
    let range = l.cold.start as usize..l.cold.end as usize;
    for (op, &rec) in prog.cold_ops[range.clone()]
        .iter()
        .zip(&prog.cold_rec[range])
    {
        match *op {
            Op::Gate2 { t, out, a, b, .. } => {
                let v = gate2(&prog.tables, values, t, a, b);
                store(values, chg, out, v, rec);
            }
            Op::GateN {
                fold,
                invert,
                out,
                args,
                n,
                ..
            } => {
                let v = gate_n(prog, values, fold, invert, args, n);
                store(values, chg, out, v, rec);
            }
            Op::Resolve { .. } | Op::Lut { .. } => unreachable!("verified: cold code is gates"),
        }
    }
}

/// Run a LUT's cold gate code without change propagation (materialization: the support is
/// settled, so the outputs keep their values and only the internal nets are refreshed).
pub(crate) fn run_cold_silent(prog: &Program, values: &mut [Logic], l: &LutDesc) {
    for op in &prog.cold_ops[l.cold.start as usize..l.cold.end as usize] {
        match *op {
            Op::Gate2 { t, out, a, b, .. } => {
                values[out as usize] = gate2(&prog.tables, values, t, a, b);
            }
            Op::GateN {
                fold,
                invert,
                out,
                args,
                n,
                ..
            } => {
                values[out as usize] = gate_n(prog, values, fold, invert, args, n);
            }
            Op::Resolve { .. } | Op::Lut { .. } => unreachable!("verified: cold code is gates"),
        }
    }
}

/// Evaluate LUT `lut`: a table read if the support is binary, else the cold gate code. Changes
/// of the outputs are recorded in `chg`.
#[inline(never)]
fn run_lut(prog: &Program, values: &mut [Logic], rt: &mut Rt, chg: &mut Vec<u32>, lut: u32) {
    let l = &prog.luts[lut as usize];
    let mut idx = 0usize;
    let mut nb = 0u8;
    for (i, &x) in prog.lut_support(l).iter().enumerate() {
        let c = values[x as usize].code();
        nb |= c;
        idx |= usize::from(c & 1) << i;
    }
    let last = &mut rt.lut_last[lut as usize];
    if nb & 2 == 0 {
        let e = prog.lut_entries[l.table as usize + idx];
        let outs = prog.lut_outputs(l);
        let recs = &prog.lut_rec[l.outputs as usize..(l.outputs + l.num_outputs) as usize];
        if *last == NONE {
            for (j, (&o, &rec)) in outs.iter().zip(recs).enumerate() {
                let v = Logic::from_code(((e >> j) & 1) as u8);
                store(values, chg, o, v, rec);
            }
        } else {
            // The outputs hold `last` (only this LUT writes them): write the differing bits.
            let mut diff = e ^ *last;
            while diff != 0 {
                let j = diff.trailing_zeros() as usize;
                diff &= diff - 1;
                let v = Logic::from_code(((e >> j) & 1) as u8);
                debug_assert_ne!(values[outs[j] as usize], v);
                store(values, chg, outs[j], v, recs[j]);
            }
        }
        *last = e;
        set_bit(&mut rt.lut_stale, lut);
    } else {
        *last = NONE;
        run_cold(prog, values, chg, l);
        clear_bit(&mut rt.lut_stale, lut);
    }
}

/// Evaluate a combinational resolver op.
#[inline(never)]
fn run_resolve(
    prog: &Program,
    values: &mut [Logic],
    rt: &mut Rt,
    chg: &mut Vec<u32>,
    op: Op,
    rec: u32,
) {
    let Op::Resolve { out, res, .. } = op else {
        unreachable!()
    };
    let r = prog.resolvers[res as usize];
    rt.buf.clear();
    rt.buf.extend(
        prog.args[r.args as usize..(r.args + r.n) as usize]
            .iter()
            .map(|&x| values[x as usize]),
    );
    let rv = resolve(&rt.buf, r.termination, values[out as usize]);
    rt.res_contending[res as usize] = rv.contention;
    if !rt.res_touched_mark[res as usize] {
        rt.res_touched_mark[res as usize] = true;
        rt.res_touched.push(res);
    }
    store(values, chg, out, rv.value, rec);
}

/// Run a non-gate op (N-input gate, resolver, LUT) of a partition.
#[inline(never)]
fn run_special(prog: &Program, values: &mut [Logic], rt: &mut Rt, chg: &mut Vec<u32>, i: usize) {
    debug_assert!(!matches!(prog.ops[i], Op::Gate2 { .. }));
    let (op, rec) = (prog.ops[i], prog.op_rec[i]);
    match op {
        Op::Gate2 { t, out, a, b, .. } => {
            let v = gate2(&prog.tables, values, t, a, b);
            store(values, chg, out, v, rec);
        }
        Op::GateN {
            fold,
            invert,
            out,
            args,
            n,
            ..
        } => {
            let v = gate_n(prog, values, fold, invert, args, n);
            store(values, chg, out, v, rec);
        }
        Op::Resolve { .. } => run_resolve(prog, values, rt, chg, op, rec),
        Op::Lut { lut } => {
            let t = crate::probe::start();
            run_lut(prog, values, rt, chg, lut);
            crate::probe::stop(crate::probe::LUT, t);
        }
    }
}

/// Evaluate every op of partition `p` (from the interpreter encoding [`Program::hops`]), then
/// propagate its output changes. `chg` is empty and has room for every output change of a
/// partition ([`Rt::max_chg`]).
#[inline]
fn run_partition(prog: &Program, values: &mut [Logic], rt: &mut Rt, chg: &mut Vec<u32>, p: usize) {
    let part = prog.partitions[p];
    rt.parts_run += 1;
    rt.ops_run += part.len() as u64;
    let hp = prog.hpart[p];
    let (mut i, end) = (hp.start as usize, hp.end as usize);
    debug_assert!(chg.is_empty() && chg.capacity() >= rt.max_chg);
    loop {
        i += run_gates(
            values,
            &prog.hops[i..end],
            &prog.hrec[i..end],
            &prog.ftables,
            chg,
        );
        if i == end {
            break;
        }
        run_special(prog, values, rt, chg, prog.hops[i].a as usize);
        i += 1;
    }
    if !chg.is_empty() {
        rt.marks.flush(prog, values, chg);
        chg.clear();
    }
}

/// Evaluate the leading gates and fused ops of `hops` (up to the first other special op),
/// recording output changes in `chg` (which has room for them); returns the number of slots
/// consumed.
#[inline(always)]
fn run_gates(
    values: &mut [Logic],
    hops: &[HOp],
    recs: &[u32],
    ftables: &[u64],
    chg: &mut Vec<u32>,
) -> usize {
    debug_assert_eq!(hops.len(), recs.len());
    let vp = values.as_mut_ptr();
    let mut n = chg.len();
    let cp = chg.as_mut_ptr();
    let mut i = 0;
    // SAFETY: `VerifyPass` checked that the encoding is exactly `fuse::lower` of the verified
    // ops: every net is < num_nets = values.len(), a fused op's second slot follows it in the
    // same partition, its table (4^inputs entries) lies inside `ftables`, and `recs` holds the
    // ops' (valid) record offsets; a partition records at most `Rt::max_chg` changes and the
    // caller reserved that much room in `chg` (`n` stays below its capacity).
    unsafe {
        while i < hops.len() {
            let h = hops.get_unchecked(i);
            let o = h.out;
            let v;
            let rec_at = i;
            if o & HOp::SPECIAL == 0 {
                debug_assert!((h.a as usize) < values.len() && (h.b as usize) < values.len());
                let va = (*vp.add(h.a as usize)).code() as u32;
                let vb = (*vp.add(h.b as usize)).code() as u32;
                v = Logic::from_code(((h.table >> (2 * (va | vb << 2))) & 3) as u8);
                i += 1;
            } else if o & HOp::FUSED != 0 {
                debug_assert!(i + 1 < hops.len());
                let h2 = hops.get_unchecked(i + 1);
                let mut idx = (*vp.add(h.a as usize)).code() as usize
                    | ((*vp.add(h.b as usize)).code() as usize) << 2
                    | ((*vp.add(h2.a as usize)).code() as usize) << 4
                    | ((*vp.add(h2.b as usize)).code() as usize) << 6;
                if o & HOp::WIDE != 0 {
                    idx |= ((*vp.add(h2.table as usize)).code() as usize) << 8
                        | ((*vp.add(h2.out as usize)).code() as usize) << 10;
                }
                debug_assert!((h.table as usize + (idx >> 5)) < ftables.len());
                let word = *ftables.get_unchecked(h.table as usize + (idx >> 5));
                v = Logic::from_code(((word >> ((idx & 31) * 2)) & 3) as u8);
                i += 2;
            } else {
                break;
            }
            debug_assert!(((o & HOp::NET) as usize) < values.len());
            let slot = vp.add((o & HOp::NET) as usize);
            let old = *slot;
            *slot = v;
            if o & HOp::NOTIFY != 0 && old != v {
                debug_assert!(n < chg.capacity());
                *cp.add(n) = change(*recs.get_unchecked(rec_at), old);
                n += 1;
            }
        }
        chg.set_len(n);
    }
    i
}

/// Recompute the outputs of the gates absorbed into fused ops ([`Program::fcold`]) from their
/// (up-to-date) inputs.
pub(crate) fn run_fcold_silent(prog: &Program, values: &mut [Logic]) {
    for op in &prog.fcold {
        match *op {
            Op::Gate2 { t, out, a, b, .. } => {
                values[out as usize] = gate2(&prog.tables, values, t, a, b);
            }
            Op::GateN {
                fold,
                invert,
                out,
                args,
                n,
                ..
            } => {
                values[out as usize] = gate_n(prog, values, fold, invert, args, n);
            }
            Op::Resolve { .. } | Op::Lut { .. } => unreachable!("only gates are fused"),
        }
    }
}

/// Result code (0, 1 or 2 = X) of folding `fold` ([`crate::ir::table`] AND/OR/XOR) over inputs
/// whose codes are summarized by `seen` (bit `c` = some input has code `c`) and `parity` (XOR of
/// the codes).
#[inline(always)]
fn fold_result(fold: u8, seen: u32, parity: u8) -> u8 {
    use crate::ir::table;
    const ZERO: u32 = 1;
    const ONE: u32 = 2;
    match fold {
        // A 0 controls; otherwise all-1 is 1, anything else (X/Z present) is X.
        table::AND => {
            if seen & ZERO != 0 {
                0
            } else if seen == ONE {
                1
            } else {
                2
            }
        }
        table::OR => {
            if seen & ONE != 0 {
                1
            } else if seen == ZERO {
                0
            } else {
                2
            }
        }
        // Any X/Z gives X; otherwise the parity of the 1s.
        _ => {
            if seen & !(ZERO | ONE) != 0 {
                2
            } else {
                parity & 1
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::table;
    use panda_netlist::eval::{and2, not1, or2, xor2};

    const ALL: [Logic; 4] = [Logic::Zero, Logic::One, Logic::X, Logic::Z];

    #[test]
    fn fold_result_matches_reference_folds() {
        for n in 1..=5u32 {
            for code in 0..4u32.pow(n) {
                let vals: Vec<Logic> = (0..n)
                    .map(|i| ALL[(code >> (2 * i) & 3) as usize])
                    .collect();
                let (mut seen, mut parity) = (0u32, 0u8);
                for v in &vals {
                    seen |= 1 << v.code();
                    parity ^= v.code();
                }
                let refs = [
                    (table::AND, vals.iter().copied().fold(Logic::One, and2)),
                    (table::OR, vals.iter().copied().fold(Logic::Zero, or2)),
                    (table::XOR, vals.iter().copied().fold(Logic::Zero, xor2)),
                ];
                for (fold, want) in refs {
                    let v = fold_result(fold, seen, parity);
                    assert_eq!(Logic::from_code(v), want, "{fold} {vals:?}");
                    let inv = v ^ (1 & !(v >> 1));
                    assert_eq!(Logic::from_code(inv), not1(want));
                }
            }
        }
    }

    #[test]
    fn bit_helpers() {
        let mut b = vec![0u64; 2];
        set_bit(&mut b, 3);
        set_bit(&mut b, 70);
        assert!(test_bit(&b, 3) && test_bit(&b, 70) && !test_bit(&b, 4));
        clear_bit(&mut b, 3);
        assert!(!test_bit(&b, 3));
        let mut seen = Vec::new();
        for_bits(b[1], 1, |i| seen.push(i));
        assert_eq!(seen, vec![70]);
    }
}
