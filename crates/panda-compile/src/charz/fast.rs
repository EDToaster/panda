//! Fast characterization path: an exact, allocation-free re-implementation of the
//! [`TimingSim`](panda_timing::TimingSim) semantics that characterization needs, specialized to
//! kernel sub-netlists ([`KSim`]), a 64-lane bit-parallel transition engine
//! ([`super::lanes`]), and the enumeration / transition / order passes built on them, with
//! optional worker threads. Tables are bit-identical to the reference path for any thread
//! count.
//!
//! # Speed-ups (all exact)
//!
//! * No `TimingSim` reset / force / name lookups per transition: [`KSim`] loads a state by
//!   copying and keeps its schedule as a gate bitset; states are packed 2 bits per net
//!   ([`Packed`]) and looked up in integer-keyed hash maps.
//! * Transitions from a stable state skip the provably empty power-on evaluation and run 64
//!   input vectors at once ([`super::lanes`]); lanes that might collapse go to [`KSim`].
//! * The transition phase splits states across threads (dynamic scheduling); raw results are
//!   interned in table order afterwards.
//! * The order pass walks the table through a compact successor array ([`Codes`]), all checked
//!   orders of an entry interleaved, with the seeded shuffles computed without divisions
//!   ([`FastMod`]). Entry outcomes are pure functions of the table, so windows of entries are
//!   computed in parallel; the sequential pass replays the budget check and the memo of
//!   on-the-fly steps exactly as the reference accounts them ([`Fast::order_pass`]).
//!
//! # Exactness
//!
//! The reference path ([`super::Characterizer`]) runs every transition through
//! [`super::KernelSim`], i.e. `TimingSim::reset` + `force_net` per state net + `settle`. This
//! module reproduces the *observable* results of that sequence bit for bit (final values,
//! quiescence, latency), not the diagnostics (characterization never reads them):
//!
//! * **Load** (`KernelSim::load_state(state, u)`): inputs = `u`; gate-driven nets = `state`;
//!   resolver outputs = `resolve(drivers, termination, state[j])` (non-keeper terminations ignore
//!   the held value; resolver drivers are never resolver outputs, see the preconditions). The
//!   change history is empty (changes at tick 0 are never tracked), the last source change is
//!   tick 0, and every gate is scheduled for tick 1 (power-on evaluation).
//! * **Steps** follow `TimingSim::step`: gate outputs from the previous tick's values, then the
//!   pending input change (tick 1 only), then the dirty resolvers, then period-2 collapse with the
//!   identical alternation-run tracker and greatest-fixed-point closure. The event queue is a
//!   gate bitset with `TimingSim`'s rule (readers of every net written with a new value, plus
//!   drivers re-scheduled by a collapse); evaluation order within a tick is irrelevant (all
//!   gates read the previous tick's values). A gate the queue skips would not change anyway: its
//!   output equals its function of the current inputs (induction from the power-on evaluation;
//!   collapse re-schedules the driver) — which is also why the bit-parallel engine may evaluate
//!   every gate.
//! * **Settle** follows `KernelSim::settle` (same loop, bound and latency definition).
//!
//! # Preconditions (else the reference path is used)
//!
//! * The sub-netlist has the [`crate::subnet`] layout, at most [`MAX_FAST_NETS`] state nets, and
//!   every state net is driven by exactly one gate or resolver.
//! * No resolver reads a resolver output (so resolver evaluation order is irrelevant).
//! * No ideal blocks can exist (the sub-netlist has only the top instance, or ideal blocks are
//!   off).
//!
//! The equality with the reference path is checked by tests over random kernels, Panda-8 and
//! generated designs (`tests/charz_equivalence.rs`).

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use panda_netlist::eval::{eval_gate, lookup2, resolve, table2};
use panda_netlist::flat::{FlatNetlist, FlatOp};
use panda_netlist::{Logic, Termination};

use super::lanes::{LANES, LaneOut, Lanes};
use super::{
    CharzOptions, KernelTable, MAX_TABLE_INPUTS, SIM_STEP_COST, Transition, XorShift, binomial,
    input_vector, next_permutation,
};
use crate::subnet::SubnetMap;

/// Largest number of state nets the fast path handles (states are packed 2 bits per net into
/// [`Packed`] words).
pub(crate) const MAX_FAST_NETS: usize = 256;

const NONE: u32 = u32::MAX;

/// Hit counters of the rarely taken exact-fallback paths (for coverage tests; see
/// [`super::fast_path_coverage`]).
pub(crate) mod cov {
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Period-2 collapse applied (window open, candidates present).
    pub static COLLAPSE: AtomicU64 = AtomicU64::new(0);
    /// Resolver evaluation during a simulated tick.
    pub static RESOLVER_TICK: AtomicU64 = AtomicU64::new(0);
    /// Order-pass entry needing the sequential path (on-the-fly steps).
    pub static SLOW_ENTRY: AtomicU64 = AtomicU64::new(0);
    /// On-the-fly step simulated (memo miss).
    pub static SIM_STEP: AtomicU64 = AtomicU64::new(0);
    /// Exhaustive orders streamed (beyond the permutation table).
    pub static STREAMED: AtomicU64 = AtomicU64::new(0);
    /// Bit-parallel lanes handed to the scalar simulator (possible collapse).
    pub static LANE_FALLBACK: AtomicU64 = AtomicU64::new(0);

    #[inline]
    pub fn hit(c: &AtomicU64) {
        c.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot() -> [u64; 6] {
        [
            &COLLAPSE,
            &RESOLVER_TICK,
            &SLOW_ENTRY,
            &SIM_STEP,
            &STREAMED,
            &LANE_FALLBACK,
        ]
        .map(|c| c.load(Ordering::Relaxed))
    }
}

// ---------------------------------------------------------------------------------------------
// Small fast hasher for integer keys (FxHash-style multiply/rotate).
// ---------------------------------------------------------------------------------------------

/// FxHash-style hasher for small integer keys (not DoS resistant; keys are internal).
#[derive(Default, Clone, Copy)]
pub(crate) struct FxHasher(u64);

const FX_K: u64 = 0xf135_7aea_2e62_a9c5;

impl FxHasher {
    #[inline]
    fn add(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(26) ^ x).wrapping_mul(FX_K);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut b = [0u8; 8];
            b[..chunk.len()].copy_from_slice(chunk);
            self.add(u64::from_le_bytes(b));
        }
    }
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    #[inline]
    fn write_u128(&mut self, i: u128) {
        self.add(i as u64);
        self.add((i >> 64) as u64);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
}

pub(crate) type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

/// A state vector packed 2 bits per net (the [`Logic`] codes), for at most `64 × W` nets.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Packed<const W: usize>(pub(crate) [u128; W]);

impl<const W: usize> Default for Packed<W> {
    fn default() -> Self {
        Packed([0; W])
    }
}

impl<const W: usize> Hash for Packed<W> {
    #[inline]
    fn hash<H: Hasher>(&self, h: &mut H) {
        for &x in &self.0 {
            h.write_u128(x);
        }
    }
}

impl<const W: usize> Packed<W> {
    /// Largest number of nets.
    pub(crate) const NETS: usize = 64 * W;

    /// Pack `vals` (`len ≤ NETS`).
    #[inline]
    pub(crate) fn pack(vals: &[Logic]) -> Self {
        debug_assert!(vals.len() <= Self::NETS);
        let mut p = [0u128; W];
        for (i, &v) in vals.iter().enumerate() {
            p[i / 64] |= u128::from(v.code()) << (2 * (i % 64));
        }
        Packed(p)
    }

    /// Unpack the first `out.len()` values.
    #[inline]
    pub(crate) fn unpack_into(&self, out: &mut [Logic]) {
        for (i, o) in out.iter_mut().enumerate() {
            *o = Logic::from_code((self.0[i / 64] >> (2 * (i % 64))) as u8 & 3);
        }
    }

    /// True if some value is `X` (code `0b10`). Unused lanes are zero (code `Zero`).
    #[inline]
    pub(crate) fn has_x(&self) -> bool {
        const LO: u128 = u128::MAX / 3; // 0b0101…: bit 0 of every 2-bit lane
        self.0.iter().any(|&p| (p >> 1) & !p & LO != 0)
    }
}

// ---------------------------------------------------------------------------------------------
// The exact kernel simulator.
// ---------------------------------------------------------------------------------------------

/// One gate: packed 2-input table over nets `a`, `b`, or (`tab == 0`) a wide gate (`a` indexes
/// [`KSim::wide`]).
#[derive(Clone, Copy)]
struct GRec {
    tab: u32,
    a: u32,
    b: u32,
    out: u32,
}

#[derive(Clone, Copy)]
struct Wide {
    op: FlatOp,
    start: u32,
    end: u32,
}

#[derive(Clone, Copy)]
struct RRec {
    start: u32,
    end: u32,
    term: Termination,
    out: u32,
}

/// History of a net before its first change in the current tick (`TimingSim`'s `Saved`).
#[derive(Clone, Copy)]
struct Saved {
    before: Logic,
    prev: Logic,
    run: u32,
    last: u64,
}

/// `TimingSim`'s alternation-run update.
#[inline(always)]
fn next_run(last: u64, run: u32, prev: Logic, new: Logic, t: u64) -> u32 {
    if last + 1 == t && run >= 1 && new == prev {
        run.saturating_add(1)
    } else {
        1
    }
}

/// Result of one transition / settle of the fast simulator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Outcome<const W: usize> {
    /// Packed final state.
    pub state: Packed<W>,
    pub quiescent: bool,
    pub latency: u32,
}

/// Exact kernel simulator (see the module docs).
#[derive(Clone)]
pub(crate) struct KSim {
    k: usize,
    m: usize,
    gates: Vec<GRec>,
    wide: Vec<Wide>,
    wide_in: Vec<u32>,
    /// Words per gate bitset.
    words: usize,
    /// Per net (`words` words each): bitset of the gates reading it.
    rmask: Vec<u64>,
    /// CSR: resolvers reading each net.
    rr_start: Vec<u32>,
    rr: Vec<u32>,
    res: Vec<RRec>,
    res_drv: Vec<u32>,
    /// Per net: driving gate / resolver (`NONE` if none).
    drv_gate: Vec<u32>,
    drv_res: Vec<u32>,
    /// `2 × p2_limit` (0 = no collapse).
    limit: u32,
    bound: u64,
    // ---- run state ----
    values: Vec<Logic>,
    last: Vec<u64>,
    run: Vec<u32>,
    prev: Vec<Logic>,
    changed: Vec<u32>,
    saved: Vec<Saved>,
    cands: Vec<u32>,
    needs_fix: bool,
    res_dirty: Vec<bool>,
    any_res_dirty: bool,
    res_hold: Vec<bool>,
    p2_mark: Vec<bool>,
    tick: u64,
    last_src: u64,
    /// Gates scheduled for the next tick (`TimingSim`'s event queue as a bitset).
    sched: Vec<u64>,
    /// Scratch: the gates evaluated in the current tick.
    cur: Vec<u64>,
    /// An input change is pending (applied at the next tick).
    pending: bool,
    /// The next step's gate evaluation is known to change nothing (skip it; the step still
    /// counts).
    skip_eval: bool,
    new_inputs: Vec<Logic>,
    newv: Vec<(u32, Logic)>,
    buf: Vec<Logic>,
    held: Vec<u32>,
}

impl KSim {
    /// Compile `sub` (a [`crate::subnet`] sub-netlist with `map`), or `None` if a precondition of
    /// the fast path fails.
    pub(crate) fn new(sub: &FlatNetlist, map: &SubnetMap, opts: &CharzOptions) -> Option<KSim> {
        let k = map.num_inputs();
        let m = map.num_state();
        let n = k + m;
        if m > MAX_FAST_NETS || sub.num_nets as usize != n {
            return None;
        }
        if opts.timing.ideal_blocks && sub.instances.len() > 1 {
            return None;
        }
        // Inputs: exactly the TopInput sources of bits 0..k on nets 0..k.
        if sub.sources.len() != k {
            return None;
        }
        for s in &sub.sources {
            match s.kind {
                panda_netlist::flat::SourceKind::TopInput { pin: 0, bit }
                    if bit as usize == s.output as usize && (s.output as usize) < k => {}
                _ => return None,
            }
        }
        let mut drv_gate = vec![NONE; n];
        let mut drv_res = vec![NONE; n];
        let mut gates = Vec::with_capacity(sub.gates.len());
        let mut wide = Vec::new();
        let mut wide_in = Vec::new();
        let words = sub.gates.len().div_ceil(64).max(1);
        let mut rmask = vec![0u64; n * words];
        for (gi, g) in sub.gates.iter().enumerate() {
            let out = g.output as usize;
            if out < k || out >= n || drv_gate[out] != NONE {
                return None;
            }
            if g.inputs.iter().any(|&i| i as usize >= n) {
                return None;
            }
            let arity_ok = match g.op {
                FlatOp::Not | FlatOp::Buf => g.inputs.len() == 1,
                FlatOp::Tri => g.inputs.len() == 2,
                _ => !g.inputs.is_empty(),
            };
            if !arity_ok {
                return None;
            }
            drv_gate[out] = gi as u32;
            for &i in &g.inputs {
                rmask[i as usize * words + gi / 64] |= 1 << (gi % 64);
            }
            let op = g.op;
            let rec = match g.inputs.len() {
                1 => GRec {
                    tab: table2(|a, _| eval_gate(op, &[a])),
                    a: g.inputs[0],
                    b: g.inputs[0],
                    out: g.output,
                },
                2 => GRec {
                    tab: table2(|a, b| eval_gate(op, &[a, b])),
                    a: g.inputs[0],
                    b: g.inputs[1],
                    out: g.output,
                },
                _ => {
                    let start = wide_in.len() as u32;
                    wide_in.extend_from_slice(&g.inputs);
                    wide.push(Wide {
                        op,
                        start,
                        end: wide_in.len() as u32,
                    });
                    GRec {
                        tab: 0,
                        a: wide.len() as u32 - 1,
                        b: 0,
                        out: g.output,
                    }
                }
            };
            // A real 1/2-input table is never all-zero codes for every input (that would be a
            // constant-0 gate), so `tab == 0` unambiguously marks wide gates.
            if g.inputs.len() <= 2 && rec.tab == 0 {
                return None;
            }
            gates.push(rec);
        }
        let mut res = Vec::with_capacity(sub.resolvers.len());
        let mut res_drv = Vec::new();
        for (ri, r) in sub.resolvers.iter().enumerate() {
            let out = r.output as usize;
            if out < k || out >= n || drv_gate[out] != NONE || drv_res[out] != NONE {
                return None;
            }
            drv_res[out] = ri as u32;
            let start = res_drv.len() as u32;
            for &d in &r.drivers {
                if d as usize >= n {
                    return None;
                }
                res_drv.push(d);
            }
            res.push(RRec {
                start,
                end: res_drv.len() as u32,
                term: r.termination,
                out: r.output,
            });
        }
        // Every state net driven; no resolver reads a resolver output.
        if (k..n).any(|j| drv_gate[j] == NONE && drv_res[j] == NONE) {
            return None;
        }
        if res_drv.iter().any(|&d| drv_res[d as usize] != NONE) {
            return None;
        }
        let mut rr_lists: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (ri, r) in res.iter().enumerate() {
            for &d in &res_drv[r.start as usize..r.end as usize] {
                if !rr_lists[d as usize].contains(&(ri as u32)) {
                    rr_lists[d as usize].push(ri as u32);
                }
            }
        }
        let mut rr_start = Vec::with_capacity(n + 1);
        let mut rr = Vec::new();
        rr_start.push(0);
        for l in &rr_lists {
            rr.extend_from_slice(l);
            rr_start.push(rr.len() as u32);
        }
        let nr = res.len();
        let ng = gates.len();
        Some(KSim {
            k,
            m,
            gates,
            wide,
            wide_in,
            words,
            rmask,
            rr_start,
            rr,
            res,
            res_drv,
            drv_gate,
            drv_res,
            limit: opts.timing.p2_limit.saturating_mul(2),
            bound: opts.bound_for(m),
            values: vec![Logic::X; n],
            last: vec![0; n],
            run: vec![0; n],
            prev: vec![Logic::X; n],
            changed: Vec::with_capacity(n),
            saved: Vec::with_capacity(n),
            cands: Vec::new(),
            needs_fix: false,
            res_dirty: vec![false; nr],
            any_res_dirty: false,
            res_hold: vec![false; nr],
            p2_mark: vec![false; n],
            tick: 0,
            last_src: 0,
            sched: vec![0; words],
            cur: vec![0; words],
            pending: false,
            skip_eval: false,
            new_inputs: vec![Logic::Zero; k],
            newv: Vec::with_capacity(ng),
            buf: Vec::new(),
            held: Vec::new(),
        })
    }

    /// Number of kernel inputs.
    pub(crate) fn num_inputs(&self) -> usize {
        self.k
    }

    /// Number of state nets.
    pub(crate) fn num_state(&self) -> usize {
        self.m
    }

    #[inline(always)]
    fn eval_gate(&self, g: GRec) -> Logic {
        if g.tab != 0 {
            lookup2(g.tab, self.values[g.a as usize], self.values[g.b as usize])
        } else {
            self.eval_wide(self.wide[g.a as usize])
        }
    }

    #[inline(never)]
    fn eval_wide(&self, w: Wide) -> Logic {
        let mut buf = [Logic::X; 16];
        let ins = &self.wide_in[w.start as usize..w.end as usize];
        if ins.len() <= buf.len() {
            for (b, &i) in buf.iter_mut().zip(ins) {
                *b = self.values[i as usize];
            }
            eval_gate(w.op, &buf[..ins.len()])
        } else {
            let v: Vec<Logic> = ins.iter().map(|&i| self.values[i as usize]).collect();
            eval_gate(w.op, &v)
        }
    }

    #[inline]
    fn resolve_res(&mut self, r: RRec, held: Logic) -> Logic {
        self.buf.clear();
        for &d in &self.res_drv[r.start as usize..r.end as usize] {
            self.buf.push(self.values[d as usize]);
        }
        resolve(&self.buf, r.term, held).value
    }

    /// Load `state` (`m` values) under `inputs` (`k` values): `KernelSim::load_state`.
    fn load(&mut self, state: &[Logic], inputs: &[Logic]) {
        let k = self.k;
        self.values[..k].copy_from_slice(inputs);
        self.values[k..].copy_from_slice(state);
        for ri in 0..self.res.len() {
            let r = self.res[ri];
            let j = r.out as usize;
            self.values[j] = self.resolve_res(r, state[j - k]);
        }
        self.last.fill(0);
        self.run.fill(0);
        self.prev.fill(Logic::X);
        self.changed.clear();
        self.saved.clear();
        self.cands.clear();
        self.needs_fix = false;
        self.res_dirty.fill(false);
        self.any_res_dirty = false;
        self.tick = 0;
        self.last_src = 0;
        // Power-on evaluation: every gate is scheduled for tick 1.
        let ng = self.gates.len();
        for (w, word) in self.sched.iter_mut().enumerate() {
            let lo = w * 64;
            *word = if ng >= lo + 64 {
                u64::MAX
            } else if ng > lo {
                (1u64 << (ng - lo)) - 1
            } else {
                0
            };
        }
        self.pending = false;
        self.skip_eval = false;
    }

    /// Queue an input change (takes effect at the next tick): `KernelSim::apply_inputs`.
    fn apply(&mut self, inputs: &[Logic]) {
        self.new_inputs.copy_from_slice(inputs);
        self.pending = self.values[..self.k] != *inputs;
    }

    /// `Tracker::record`.
    #[inline(always)]
    fn record(&mut self, net: u32, old: Logic, new: Logic, t: u64) {
        let n = net as usize;
        let last = self.last[n];
        if last == t {
            self.needs_fix = true;
            return;
        }
        let (run, prev) = (self.run[n], self.prev[n]);
        self.saved.push(Saved {
            before: old,
            prev,
            run,
            last,
        });
        self.changed.push(net);
        let r = next_run(last, run, prev, new, t);
        self.last[n] = t;
        self.run[n] = r;
        self.prev[n] = old;
        if self.limit != 0 && r >= self.limit {
            self.cands.push(net);
        }
    }

    /// `Tracker::fix`.
    #[cold]
    fn fix(&mut self, t: u64, rebuild: bool) {
        if rebuild {
            self.cands.clear();
        }
        let mut kk = 0;
        for i in 0..self.changed.len() {
            let net = self.changed[i];
            let n = net as usize;
            let s = self.saved[i];
            let cur = self.values[n];
            if cur == s.before {
                self.last[n] = s.last;
                self.run[n] = s.run;
                self.prev[n] = s.prev;
                continue;
            }
            let r = next_run(s.last, s.run, s.prev, cur, t);
            self.last[n] = t;
            self.run[n] = r;
            self.prev[n] = s.before;
            if rebuild && self.limit != 0 && r >= self.limit {
                self.cands.push(net);
            }
            self.changed[kk] = net;
            self.saved[kk] = s;
            kk += 1;
        }
        self.changed.truncate(kk);
        self.saved.truncate(kk);
        self.needs_fix = false;
    }

    /// Effects of writing a new value to `net` on the schedule (`TimingSim::schedule_fanout`).
    #[inline(always)]
    fn fanout(&mut self, net: u32) {
        let n = net as usize;
        let w = self.words;
        for (s, r) in self.sched.iter_mut().zip(&self.rmask[n * w..n * w + w]) {
            *s |= *r;
        }
        let (a, b) = (self.rr_start[n] as usize, self.rr_start[n + 1] as usize);
        if a != b {
            for i in a..b {
                self.res_dirty[self.rr[i] as usize] = true;
            }
            self.any_res_dirty = true;
        }
    }

    /// Write a value from a zero-delay element (`TimingSim::set_net`).
    #[inline]
    fn set_net(&mut self, net: u32, v: Logic, t: u64) {
        let old = self.values[net as usize];
        if old != v {
            self.values[net as usize] = v;
            self.record(net, old, v, t);
            self.fanout(net);
        }
    }

    /// `TimingSim::process_resolvers` (order-independent under the preconditions).
    fn process_resolvers(&mut self, t: u64) {
        if !self.any_res_dirty {
            return;
        }
        if t > 0 {
            cov::hit(&cov::RESOLVER_TICK);
        }
        self.any_res_dirty = false;
        for ri in 0..self.res.len() {
            if !std::mem::replace(&mut self.res_dirty[ri], false) || self.res_hold[ri] {
                continue;
            }
            let r = self.res[ri];
            let v = self.resolve_res(r, self.values[r.out as usize]);
            self.set_net(r.out, v, t);
        }
        // Resolvers never read resolver outputs, so no resolver became dirty again.
        debug_assert!(!self.res_dirty.iter().any(|&d| d));
        self.any_res_dirty = false;
    }

    #[inline]
    fn p2_ok(&self, i: u32, t: u64, limit: u64) -> bool {
        self.p2_mark[i as usize] || self.last[i as usize] + limit <= t
    }

    /// `TimingSim::p2_sustained`.
    fn p2_sustained(&self, net: u32, t: u64, limit: u64) -> bool {
        let g = self.drv_gate[net as usize];
        if g != NONE {
            let rec = self.gates[g as usize];
            return if rec.tab != 0 {
                self.p2_ok(rec.a, t, limit) && self.p2_ok(rec.b, t, limit)
            } else {
                let w = self.wide[rec.a as usize];
                self.wide_in[w.start as usize..w.end as usize]
                    .iter()
                    .all(|&i| self.p2_ok(i, t, limit))
            };
        }
        let r = self.drv_res[net as usize];
        if r != NONE {
            let r = self.res[r as usize];
            return self.res_drv[r.start as usize..r.end as usize]
                .iter()
                .all(|&d| self.p2_ok(d, t, limit));
        }
        false
    }

    /// `TimingSim::collapse`.
    #[cold]
    fn collapse(&mut self, t: u64) {
        let limit = u64::from(self.limit);
        if limit == 0 || t - self.last_src < limit {
            self.cands.clear();
            return;
        }
        cov::hit(&cov::COLLAPSE);
        let cands = std::mem::take(&mut self.cands);
        for &c in &cands {
            self.p2_mark[c as usize] = true;
        }
        // Greatest fixed point (unique; `p2_sustained` is monotone in the marked set).
        loop {
            let mut dropped = false;
            for &c in &cands {
                if self.p2_mark[c as usize] && !self.p2_sustained(c, t, limit) {
                    self.p2_mark[c as usize] = false;
                    dropped = true;
                }
            }
            if !dropped {
                break;
            }
        }
        self.held.clear();
        for &c in &cands {
            let n = c as usize;
            if !std::mem::replace(&mut self.p2_mark[n], false) || self.values[n] == Logic::X {
                continue;
            }
            self.values[n] = Logic::X;
            self.needs_fix = true;
            self.fanout(c);
            let g = self.drv_gate[n];
            if g != NONE {
                self.sched[g as usize / 64] |= 1 << (g % 64);
            }
            let r = self.drv_res[n];
            if r != NONE {
                self.res_hold[r as usize] = true;
                self.held.push(r);
            }
        }
        self.process_resolvers(t);
        for i in 0..self.held.len() {
            let r = self.held[i];
            self.res_hold[r as usize] = false;
        }
        let mut cands = cands;
        cands.clear();
        self.cands = cands;
    }

    /// `TimingSim::step` (values and history only).
    fn step(&mut self) {
        let t = self.tick + 1;
        self.changed.clear();
        self.saved.clear();
        self.cands.clear();
        self.needs_fix = false;
        // Gates: V(t)[out] = f(V(t-1)) for the scheduled gates.
        std::mem::swap(&mut self.sched, &mut self.cur);
        self.sched.fill(0);
        if std::mem::replace(&mut self.skip_eval, false) {
            self.cur.fill(0);
        }
        let mut newv = std::mem::take(&mut self.newv);
        newv.clear();
        for w in 0..self.words {
            let mut bits = self.cur[w];
            while bits != 0 {
                let gi = w * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let g = self.gates[gi];
                let v = self.eval_gate(g);
                if v != self.values[g.out as usize] {
                    newv.push((g.out, v));
                }
            }
        }
        for &(out, v) in &newv {
            let old = self.values[out as usize];
            self.values[out as usize] = v;
            self.record(out, old, v, t);
            self.fanout(out);
        }
        self.newv = newv;
        self.tick = t;
        // Sources.
        if self.pending {
            self.pending = false;
            let mut src_changed = false;
            for i in 0..self.k {
                let v = self.new_inputs[i];
                if self.values[i] != v {
                    src_changed = true;
                    self.set_net(i as u32, v, t);
                }
            }
            if src_changed {
                self.last_src = t;
            }
        }
        // Resolvers.
        self.process_resolvers(t);
        if self.needs_fix {
            self.fix(t, true);
        }
        // Period-2 collapse.
        if !self.cands.is_empty() {
            self.collapse(t);
            if self.needs_fix {
                self.fix(t, false);
            }
        }
    }

    #[inline]
    fn quiescent(&self) -> bool {
        !self.pending && self.sched.iter().all(|&w| w == 0)
    }

    /// `KernelSim::settle(bound)` with the change tick 1.
    fn settle(&mut self, bound: u64) -> (bool, u32) {
        let k = self.k as u32;
        let start = 1u64;
        let mut last = start;
        let mut quiescent = false;
        for _ in 0..bound {
            if self.quiescent() {
                quiescent = true;
                break;
            }
            self.step();
            if self.changed.iter().any(|&n| n >= k) {
                last = self.tick;
            }
        }
        quiescent = quiescent || self.quiescent();
        (
            quiescent,
            last.saturating_sub(start).min(u64::from(u32::MAX)) as u32,
        )
    }

    fn outcome<const W: usize>(&self, quiescent: bool, latency: u32) -> Outcome<W> {
        Outcome {
            state: Packed::pack(&self.values[self.k..]),
            quiescent,
            latency,
        }
    }

    /// `KernelSim::load_state(state, inputs)` + `settle(default bound)`.
    pub(crate) fn load_settle<const W: usize>(
        &mut self,
        state: &[Logic],
        inputs: &[Logic],
    ) -> Outcome<W> {
        self.load(state, inputs);
        let (q, l) = self.settle(self.bound);
        self.outcome(q, l)
    }

    /// `KernelSim::transition(state, old, new)`.
    pub(crate) fn transition<const W: usize>(
        &mut self,
        state: &[Logic],
        old: &[Logic],
        new: &[Logic],
    ) -> Outcome<W> {
        self.load(state, old);
        self.apply(new);
        let (q, l) = self.settle(self.bound);
        self.outcome(q, l)
    }

    /// [`KSim::transition`] from a state that is a fixed point of every gate under `old` (a
    /// stable state under its home input): the power-on evaluation at tick 1 provably changes
    /// nothing, so it is skipped (the rest of the tick runs as usual).
    pub(crate) fn transition_from_stable<const W: usize>(
        &mut self,
        state: &[Logic],
        old: &[Logic],
        new: &[Logic],
    ) -> Outcome<W> {
        self.load(state, old);
        debug_assert!(
            self.gates
                .iter()
                .all(|&g| self.eval_gate(g) == self.values[g.out as usize]),
            "transition_from_stable: not a fixed point"
        );
        self.skip_eval = true;
        self.apply(new);
        let (q, l) = self.settle(self.bound);
        self.outcome(q, l)
    }

    /// `KernelSim::is_fixed_point(state, inputs)`.
    pub(crate) fn is_fixed_point(&mut self, state: &[Logic], inputs: &[Logic]) -> bool {
        let k = self.k;
        self.values[..k].copy_from_slice(inputs);
        self.values[k..].copy_from_slice(state);
        for gi in 0..self.gates.len() {
            let g = self.gates[gi];
            if self.eval_gate(g) != self.values[g.out as usize] {
                return false;
            }
        }
        for ri in 0..self.res.len() {
            let r = self.res[ri];
            let cur = self.values[r.out as usize];
            if self.resolve_res(r, cur) != cur {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------------------------
// Fast characterizer.
// ---------------------------------------------------------------------------------------------

/// Marker in the raw-result array: the entry's result did not settle.
const RAW_LOST: u32 = u32::MAX;

/// Table entries (stable states × input vectors) from which the transition phase uses worker
/// threads.
const PAR_MIN_ENTRIES: usize = 8192;
/// Order-pass window: entries per worker thread.
const ORDER_WINDOW_PER_THREAD: usize = 4096;
/// Order-pass window size without worker threads, and the smallest parallel window.
const ORDER_WINDOW_MIN: usize = 1024;
/// Order-pass entries per dynamically scheduled job.
const ORDER_CHUNK: usize = 128;

/// A decoded table successor code ([`Codes`]): stable state id, interned raw state id, or
/// did not settle.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Cur {
    Id(u32),
    Raw(u32),
    Lost,
}

/// Characterize with the fast path using up to `threads` threads. `None` if a precondition
/// fails (use the reference path); `Some(None)` if the table exceeds the size guards (exactly as
/// the reference path). The result does not depend on `threads`.
pub(crate) fn characterize(
    sub: &FlatNetlist,
    map: &SubnetMap,
    opts: &CharzOptions,
    cut: &[usize],
    order: &[usize],
    threads: usize,
) -> Option<Option<KernelTable>> {
    let sim = KSim::new(sub, map, opts)?;
    let lanes = Lanes::new(sub, map, opts);
    let threads = threads.max(1);
    Some(match sim.num_state() {
        0..=64 => Fast::<1>::new(sim, lanes, opts, threads).run(cut, order),
        65..=128 => Fast::<2>::new(sim, lanes, opts, threads).run(cut, order),
        _ => Fast::<4>::new(sim, lanes, opts, threads).run(cut, order),
    })
}

struct Fast<'a, const W: usize> {
    sim: KSim,
    /// Bit-parallel transition engine (kernels without resolvers).
    lanes: Option<Lanes>,
    opts: &'a CharzOptions,
    threads: usize,
    k: usize,
    m: usize,
    /// Packed stable state → id.
    index: FxMap<Packed<W>, u32>,
    /// Stable states (packed).
    packed: Vec<Packed<W>>,
    /// Interned raw (non-stable) states.
    raw_index: FxMap<Packed<W>, u32>,
    raws: Vec<Packed<W>>,
}

fn set_inputs(buf: &mut [Logic], v: u32) {
    for (i, b) in buf.iter_mut().enumerate() {
        *b = Logic::from_bool(v >> i & 1 == 1);
    }
}

/// Transitions of the stable states `states` (all input vectors each) into `out` (in table
/// order, `order_sensitive` unset). Quiescent results that are not stable states are appended
/// to `raws` as `(entry index within out, packed state)`.
#[allow(clippy::too_many_arguments)]
fn transitions_of<const W: usize>(
    sim: &mut KSim,
    mut lanes: Option<&mut Lanes>,
    index: &FxMap<Packed<W>, u32>,
    packed: &[Packed<W>],
    home: &[u32],
    states: std::ops::Range<usize>,
    out: &mut [Transition],
    raws: &mut Vec<(usize, Packed<W>)>,
) {
    let k = sim.num_inputs();
    let m = sim.num_state();
    let nv = 1u32 << k;
    let mut st = vec![Logic::X; m];
    let mut in_a = vec![Logic::Zero; k];
    let mut in_b = vec![Logic::Zero; k];
    let mut outs = [LaneOut::<W>::default(); LANES];
    let mut pos = 0;
    for s in states {
        packed[s].unpack_into(&mut st);
        set_inputs(&mut in_a, home[s]);
        for v in 0..nv {
            let li = v as usize % LANES;
            if let Some(l) = lanes.as_deref_mut()
                && li == 0
            {
                let cnt = (nv - v).min(LANES as u32) as usize;
                l.run(&st, home[s], v, cnt, &mut outs);
            }
            let o = match lanes.is_some().then_some(outs[li]) {
                Some(lo) if lo.done => Outcome {
                    state: lo.state,
                    quiescent: lo.quiescent,
                    latency: lo.latency,
                },
                _ => {
                    set_inputs(&mut in_b, v);
                    sim.transition_from_stable(&st, &in_a, &in_b)
                }
            };
            let next = if o.quiescent {
                let next = index.get(&o.state).copied();
                if next.is_none() {
                    raws.push((pos, o.state));
                }
                next
            } else {
                None
            };
            out[pos] = Transition {
                next,
                latency: o.latency,
                order_sensitive: false,
            };
            pos += 1;
        }
    }
}

impl<'a, const W: usize> Fast<'a, W> {
    fn new(sim: KSim, lanes: Option<Lanes>, opts: &'a CharzOptions, threads: usize) -> Self {
        let k = sim.num_inputs();
        let m = sim.num_state();
        Fast {
            sim,
            lanes,
            opts,
            threads,
            k,
            m,
            index: FxMap::default(),
            packed: Vec::new(),
            raw_index: FxMap::default(),
            raws: Vec::new(),
        }
    }

    fn intern_raw(&mut self, p: Packed<W>) -> u32 {
        let next = self.raws.len() as u32;
        let id = *self.raw_index.entry(p).or_insert(next);
        if id == next {
            self.raws.push(p);
        }
        id
    }

    fn run(mut self, cut: &[usize], order: &[usize]) -> Option<KernelTable> {
        let k = self.k;
        let m = self.m;
        debug_assert!(k <= MAX_TABLE_INPUTS);
        if k + cut.len() > 2 * MAX_TABLE_INPUTS
            || (1u64 << (k + cut.len())) > self.opts.max_table_entries.max(1 << 12)
        {
            return None;
        }
        let nv = 1u32 << k;
        // ---- stable-state enumeration ----
        let mut home_input: Vec<u32> = Vec::new();
        let mut states_by_input: Vec<Vec<u32>> = vec![Vec::new(); nv as usize];
        let ncut = 1u32 << cut.len();
        let mut cand = vec![Logic::X; m];
        let mut st = vec![Logic::X; m];
        let mut uv = vec![Logic::Zero; k];
        for u in 0..nv {
            set_inputs(&mut uv, u);
            for c in 0..ncut {
                self.candidate(&uv, c, cut, order, &mut cand);
                let o = self.sim.load_settle(&cand, &uv);
                if !o.quiescent || o.state.has_x() {
                    continue;
                }
                o.state.unpack_into(&mut st);
                if !self.sim.is_fixed_point(&st, &uv) {
                    continue;
                }
                let next = self.packed.len() as u32;
                let id = *self.index.entry(o.state).or_insert(next);
                if id == next {
                    self.packed.push(o.state);
                    home_input.push(u);
                }
                let list = &mut states_by_input[u as usize];
                if !list.contains(&id) {
                    list.push(id);
                }
            }
            states_by_input[u as usize].sort_unstable();
        }
        // ---- simultaneous transitions ----
        let ns = self.packed.len();
        if (ns as u64) << k > self.opts.max_table_entries {
            return None;
        }
        let entries = ns << k;
        let mut transitions = vec![
            Transition {
                next: None,
                latency: 0,
                order_sensitive: false,
            };
            entries
        ];
        let workers = if entries >= PAR_MIN_ENTRIES {
            self.threads.min(ns)
        } else {
            1
        };
        // Raw (quiescent, non-stable) results per worker, in table order.
        let mut raw_lists: Vec<Vec<(usize, Packed<W>)>> = Vec::new();
        if workers <= 1 {
            let mut raws = Vec::new();
            transitions_of(
                &mut self.sim,
                self.lanes.as_mut(),
                &self.index,
                &self.packed,
                &home_input,
                0..ns,
                &mut transitions,
                &mut raws,
            );
            raw_lists.push(raws);
        } else {
            // Dynamic scheduling, one stable state per job (transition costs vary a lot
            // between states: collapsing or oscillating transitions run to the settle bound).
            let (index, packed, home) = (&self.index, &self.packed, &home_input);
            let (sim, lanes) = (&self.sim, &self.lanes);
            let jobs: Vec<Mutex<Option<&mut [Transition]>>> = transitions
                .chunks_mut(1 << k)
                .map(|c| Mutex::new(Some(c)))
                .collect();
            let next = AtomicUsize::new(0);
            raw_lists = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..workers)
                    .map(|_| {
                        let (jobs, next) = (&jobs, &next);
                        let mut sim = sim.clone();
                        let mut lanes = lanes.clone();
                        scope.spawn(move || {
                            let mut raws = Vec::new();
                            let mut local = Vec::new();
                            loop {
                                let s = next.fetch_add(1, Ordering::Relaxed);
                                if s >= jobs.len() {
                                    break raws;
                                }
                                let chunk = jobs[s]
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                                    .take()
                                    .expect("each state is taken once");
                                local.clear();
                                transitions_of(
                                    &mut sim,
                                    lanes.as_mut(),
                                    index,
                                    packed,
                                    home,
                                    s..s + 1,
                                    chunk,
                                    &mut local,
                                );
                                raws.extend(local.iter().map(|&(i, p)| ((s << k) + i, p)));
                            }
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
                    .collect()
            });
            // Table order for interning.
            let mut all: Vec<(usize, Packed<W>)> = raw_lists.into_iter().flatten().collect();
            all.sort_unstable_by_key(|e| e.0);
            raw_lists = vec![all];
        }
        // Raw states are interned in table order (as the reference meets them).
        let mut raw_of = vec![RAW_LOST; entries];
        for (i, p) in raw_lists.into_iter().flatten() {
            raw_of[i] = self.intern_raw(p);
        }
        let states: Vec<Vec<Logic>> = self
            .packed
            .iter()
            .map(|&p| {
                let mut v = vec![Logic::X; m];
                p.unpack_into(&mut v);
                v
            })
            .collect();
        let state_index: HashMap<Vec<Logic>, u32> = states
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        let mut table = KernelTable {
            num_inputs: k,
            states,
            state_index,
            transitions,
            states_by_input,
            home_input,
            order_check_partial: false,
            sensitive_from: HashSet::new(),
            order_check_stop: None,
        };
        let codes = Codes::build(&table.transitions, &raw_of, ns);
        drop(raw_of);
        self.order_pass(&mut table, &codes);
        Some(table)
    }

    /// Candidate state for inputs `inputs` and cut assignment `c` (`Characterizer::candidate`).
    fn candidate(
        &self,
        inputs: &[Logic],
        c: u32,
        cut: &[usize],
        order: &[usize],
        st: &mut [Logic],
    ) {
        let k = self.k;
        st.fill(Logic::X);
        for (b, &j) in cut.iter().enumerate() {
            st[j] = Logic::from_bool(c >> b & 1 == 1);
        }
        let val = |st: &[Logic], n: u32| -> Logic {
            if (n as usize) < k {
                inputs[n as usize]
            } else {
                st[n as usize - k]
            }
        };
        let mut buf = [Logic::X; 16];
        let mut big = Vec::new();
        for &j in order {
            let net = k + j;
            let g = self.sim.drv_gate[net];
            st[j] = if g != NONE {
                let rec = self.sim.gates[g as usize];
                if rec.tab != 0 {
                    lookup2(rec.tab, val(st, rec.a), val(st, rec.b))
                } else {
                    let w = self.sim.wide[rec.a as usize];
                    let ins = &self.sim.wide_in[w.start as usize..w.end as usize];
                    if ins.len() <= buf.len() {
                        for (b, &i) in buf.iter_mut().zip(ins) {
                            *b = val(st, i);
                        }
                        eval_gate(w.op, &buf[..ins.len()])
                    } else {
                        big.clear();
                        big.extend(ins.iter().map(|&i| val(st, i)));
                        eval_gate(w.op, &big)
                    }
                }
            } else {
                let r = self.sim.res[self.sim.drv_res[net] as usize];
                big.clear();
                big.extend(
                    self.sim.res_drv[r.start as usize..r.end as usize]
                        .iter()
                        .map(|&d| val(st, d)),
                );
                resolve(&big, r.term, Logic::X).value
            };
        }
    }

    // ---- order-sensitivity pass (identical algorithm, results and work accounting) ----

    /// `Characterizer::order_pass`. Entries are visited in the reference order. Each entry's
    /// outcome — whether some checked order disagrees, how many orders were evaluated, and the
    /// sequence of on-the-fly steps the reference would perform — is a pure function of the
    /// table, so workers compute windows of entries in parallel. The sequential pass then
    /// applies the budget check before every entry and replays only the memo (an on-the-fly
    /// step costs [`SIM_STEP_COST`] when its key is new), exactly as the reference accounts it.
    fn order_pass(&mut self, table: &mut KernelTable, codes: &Codes) {
        let k = self.k;
        let nv = 1u32 << k;
        if k < 2 || table.states.is_empty() {
            return;
        }
        let opts = self.opts;
        let per_home: u64 = (0..=k)
            .filter(|&j| j >= 2)
            .map(|j| binomial(k, j) * orders_for(opts, j) * j as u64)
            .sum();
        let homes_total: u64 = table.states_by_input.iter().map(|l| l.len() as u64).sum();
        let full = homes_total.saturating_mul(per_home) <= opts.max_order_work;
        table.order_check_partial = !full;
        let mut homes: Vec<Vec<u32>> = vec![Vec::new(); table.states.len()];
        if full {
            for (u, list) in table.states_by_input.iter().enumerate() {
                for &s in list {
                    homes[s as usize].push(u as u32);
                }
            }
        } else {
            for (s, &u) in table.home_input.iter().enumerate() {
                homes[s].push(u);
            }
        }
        let ctx = OrderCtx::new(codes, k, opts, &self.index, &self.raws, &self.sim);
        let mut work = 0u64;
        let mut memo: HashSet<StepKey<W>, BuildHasherDefault<FxHasher>> = HashSet::default();
        // One persistent worker per thread (their simulators and scratch survive windows).
        let mut workers: Vec<Worker<'_, '_, W>> =
            (0..self.threads).map(|_| Worker::new(&ctx)).collect();
        let mut cursor = (0usize, 0usize, 0u32); // (state, home index, v)
        let mut window: Vec<(u32, u32, u32)> = Vec::new();
        let mut results: Vec<EntryOut<W>> = Vec::new();
        let (mut done_entries, mut done_work) = (0u64, 0u64);
        loop {
            // Window size: bounded by the estimated number of entries the budget still allows
            // (speculation past the budget stop is wasted).
            let max_cap = if self.threads > 1 {
                self.threads * ORDER_WINDOW_PER_THREAD
            } else {
                ORDER_WINDOW_MIN
            };
            let cap = match done_work.checked_div(done_entries) {
                _ if work > opts.max_order_work => max_cap,
                // No cost estimate yet: a modest first window.
                None => max_cap.min(self.threads * ORDER_WINDOW_MIN),
                Some(avg) => {
                    let est = (opts.max_order_work - work) / avg.max(1);
                    (est.saturating_mul(5) / 4 + 64).clamp(ORDER_WINDOW_MIN as u64, max_cap as u64)
                        as usize
                }
            };
            window.clear();
            while window.len() < cap {
                let (s, hi, v) = cursor;
                if s >= homes.len() {
                    break;
                }
                if hi >= homes[s].len() {
                    cursor = (s + 1, 0, 0);
                    continue;
                }
                if v >= nv {
                    cursor = (s, hi + 1, 0);
                    continue;
                }
                cursor.2 += 1;
                let u = homes[s][hi];
                if (u ^ v).count_ones() >= 2 {
                    window.push((s as u32, u, v));
                }
            }
            if window.is_empty() {
                break;
            }
            results.clear();
            if work <= opts.max_order_work {
                // At least ORDER_WINDOW_MIN entries per worker (spawning is not free).
                let nw = self.threads.min(window.len() / ORDER_WINDOW_MIN).max(1);
                if nw <= 1 {
                    let w = &mut workers[0];
                    results.extend(window.iter().map(|&(s, u, v)| w.entry(s, u, v)));
                } else {
                    // Dynamic scheduling over chunks of the window (costs cluster by state).
                    let window = &window;
                    let chunks = window.len().div_ceil(ORDER_CHUNK);
                    let next = AtomicUsize::new(0);
                    let mut parts: Vec<(usize, Vec<EntryOut<W>>)> = std::thread::scope(|scope| {
                        let handles: Vec<_> = workers[..nw]
                            .iter_mut()
                            .map(|w| {
                                let next = &next;
                                scope.spawn(move || {
                                    let mut out = Vec::new();
                                    loop {
                                        let c = next.fetch_add(1, Ordering::Relaxed);
                                        if c >= chunks {
                                            break out;
                                        }
                                        let r = c * ORDER_CHUNK
                                            ..((c + 1) * ORDER_CHUNK).min(window.len());
                                        let res: Vec<EntryOut<W>> = window[r]
                                            .iter()
                                            .map(|&(s, u, v)| w.entry(s, u, v))
                                            .collect();
                                        out.push((c, res));
                                    }
                                })
                            })
                            .collect();
                        handles
                            .into_iter()
                            .flat_map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
                            .collect()
                    });
                    parts.sort_unstable_by_key(|p| p.0);
                    results.extend(parts.into_iter().flat_map(|p| p.1));
                }
            }
            for (i, &(s, u, v)) in window.iter().enumerate() {
                let idx = ((s as usize) << k) | v as usize;
                if work > opts.max_order_work {
                    // Budget spent: unknown, so conservatively flagged; the exact answer is
                    // computed on demand ([`KernelTable::check_order_sensitive`]).
                    table.order_check_stop.get_or_insert((s, u, v));
                    table.transitions[idx].order_sensitive = true;
                    continue;
                }
                let r = &results[i];
                let before = work;
                work += r.table_work;
                for key in &r.steps {
                    if memo.insert(*key) {
                        work += SIM_STEP_COST;
                    }
                }
                done_entries += 1;
                done_work += work - before;
                if r.sensitive {
                    table.transitions[idx].order_sensitive = true;
                    table.sensitive_from.insert((s, u, v));
                }
            }
        }
    }
}

/// Key of an on-the-fly step: (settled non-binary state, old inputs, new inputs).
type StepKey<const W: usize> = (Packed<W>, u32, u32);

/// Outcome of one order-pass entry (see [`Fast::order_pass`]).
struct EntryOut<const W: usize> {
    /// Some checked order ends differently from the simultaneous change.
    sensitive: bool,
    /// Table steps of the evaluated orders (`j` per order).
    table_work: u64,
    /// The on-the-fly steps the reference performs, in its order (memo hits included).
    steps: Vec<StepKey<W>>,
}

/// A state during a sequential application, with raw states by value.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CurP<const W: usize> {
    Id(u32),
    Raw(Packed<W>),
    Lost,
}

impl<const W: usize> CurP<W> {
    fn id(self) -> Option<u32> {
        match self {
            CurP::Id(x) => Some(x),
            _ => None,
        }
    }
}

/// One order-pass worker: computes entry outcomes (pure functions of the table).
struct Worker<'c, 't, const W: usize> {
    ctx: &'c OrderCtx<'t, W>,
    masks: Vec<u32>,
    /// Simulator for on-the-fly steps (cloned on first use).
    sim: Option<KSim>,
    /// This worker's view of the step results (in front of the shared [`OrderCtx::steps`]).
    local: FxMap<StepKey<W>, CurP<W>>,
    /// Scratch for [`Worker::finish`].
    seen: HashSet<StepKey<W>, BuildHasherDefault<FxHasher>>,
    st: Vec<Logic>,
    in_a: Vec<Logic>,
    in_b: Vec<Logic>,
}

impl<'c, 't, const W: usize> Worker<'c, 't, W> {
    fn new(ctx: &'c OrderCtx<'t, W>) -> Self {
        let (k, m) = (ctx.sim.num_inputs(), ctx.sim.num_state());
        Worker {
            ctx,
            masks: Vec::new(),
            sim: None,
            local: FxMap::default(),
            seen: HashSet::default(),
            st: vec![Logic::X; m],
            in_a: vec![Logic::Zero; k],
            in_b: vec![Logic::Zero; k],
        }
    }

    /// The outcome of entry `(s, u, v)` (`OrderChecker::entry_sensitive`).
    fn entry(&mut self, s: u32, u: u32, v: u32) -> EntryOut<W> {
        let ctx = self.ctx;
        let j = (u ^ v).count_ones() as usize;
        let simul = ctx.simul(s, v);
        let mut masks = std::mem::take(&mut self.masks);
        let out = if ctx.gen_masks(&mut masks, s, u, v) {
            // Fast attempt: all orders interleaved, through the table only.
            let quick = match ctx.codes {
                Codes::U16(c, ns) => {
                    walk_orders_in(c, u32::from(u16::MAX), *ns, ctx.k, &masks, j, s, u, simul)
                }
                Codes::U32(c, ns) => {
                    walk_orders_in(c, u32::MAX, *ns, ctx.k, &masks, j, s, u, simul)
                }
            };
            match quick {
                Some(first) => EntryOut {
                    sensitive: first.is_some(),
                    table_work: (j * first.map_or(masks.len() / j, |i| i + 1)) as u64,
                    steps: Vec::new(),
                },
                None => {
                    cov::hit(&cov::SLOW_ENTRY);
                    let mut out = self.begin();
                    for order in masks.chunks_exact(j) {
                        if self.walk_one(s, u, simul, order, &mut out) {
                            break;
                        }
                    }
                    self.finish(out)
                }
            }
        } else {
            // Exhaustive orders beyond the permutation table: streamed (lexicographic).
            cov::hit(&cov::STREAMED);
            let diff = u ^ v;
            let bits: Vec<u32> = (0..ctx.k as u32)
                .filter(|&b| diff >> b & 1 == 1)
                .map(|b| 1 << b)
                .collect();
            let mut perm: Vec<usize> = (0..j).collect();
            let mut order = vec![0u32; j];
            let mut out = self.begin();
            loop {
                for (o, &p) in order.iter_mut().zip(&perm) {
                    *o = bits[p];
                }
                if self.walk_one(s, u, simul, &order, &mut out) || !next_permutation(&mut perm) {
                    break;
                }
            }
            self.finish(out)
        };
        self.masks = masks;
        out
    }

    fn begin(&mut self) -> EntryOut<W> {
        EntryOut {
            sensitive: false,
            table_work: 0,
            steps: Vec::new(),
        }
    }

    /// Evaluate one order, recording its on-the-fly steps; true if it disagrees with `simul`
    /// (the entry is order-sensitive and evaluation stops).
    fn walk_one(
        &mut self,
        s: u32,
        u: u32,
        simul: Option<u32>,
        order: &[u32],
        out: &mut EntryOut<W>,
    ) -> bool {
        let mut cur = CurP::Id(s);
        let mut w = u;
        for &mask in order {
            let prev = w;
            w ^= mask;
            cur = self.advance(cur, prev, w, &mut out.steps);
        }
        out.table_work += order.len() as u64;
        out.sensitive = cur.id() != simul;
        out.sensitive
    }

    /// Keep each step key once (later occurrences within the entry are memo hits in the
    /// reference; only the entry's total work matters for the budget check).
    fn finish(&mut self, mut out: EntryOut<W>) -> EntryOut<W> {
        if out.steps.len() > 1 {
            self.seen.clear();
            out.steps.retain(|k| self.seen.insert(*k));
        }
        out
    }

    /// One sequential step (`OrderChecker::sequential`'s loop body).
    fn advance(&mut self, cur: CurP<W>, prev: u32, w: u32, steps: &mut Vec<StepKey<W>>) -> CurP<W> {
        let ctx = self.ctx;
        match cur {
            CurP::Id(x) => match ctx.codes.get(((x as usize) << ctx.k) | w as usize) {
                Cur::Id(y) => CurP::Id(y),
                Cur::Raw(r) => CurP::Raw(ctx.raws[r as usize]),
                Cur::Lost => CurP::Lost,
            },
            CurP::Raw(p) => {
                let key = (p, prev, w);
                steps.push(key);
                if let Some(&c) = self.local.get(&key) {
                    return c;
                }
                if let Some(&c) = ctx.step_shard(&key).get(&key) {
                    self.local.insert(key, c);
                    return c;
                }
                cov::hit(&cov::SIM_STEP);
                let sim = self.sim.get_or_insert_with(|| ctx.sim.clone());
                p.unpack_into(&mut self.st);
                set_inputs(&mut self.in_a, prev);
                set_inputs(&mut self.in_b, w);
                let o: Outcome<W> = sim.transition(&self.st, &self.in_a, &self.in_b);
                let c = if !o.quiescent {
                    CurP::Lost
                } else if let Some(&id) = ctx.index.get(&o.state) {
                    CurP::Id(id)
                } else {
                    CurP::Raw(o.state)
                };
                ctx.step_shard(&key).insert(key, c);
                self.local.insert(key, c);
                c
            }
            CurP::Lost => CurP::Lost,
        }
    }
}

/// Number of orders checked for `j` simultaneous changes (`OrderChecker::orders_for`).
fn orders_for(opts: &CharzOptions, j: usize) -> u64 {
    if j <= opts.exhaustive_order_max {
        (1..=j as u64).product()
    } else {
        opts.order_samples as u64
    }
}

/// Successor of every table entry, compactly (the order pass walks it randomly, so it should
/// stay cache-resident): code `< ns` = stable state id; `ns + r` = interned raw state `r`;
/// all ones = did not settle.
enum Codes {
    U16(Vec<u16>, u32),
    U32(Vec<u32>, u32),
}

impl Codes {
    fn build(transitions: &[Transition], raw_of: &[u32], ns: usize) -> Codes {
        let code = |i: usize| -> u32 {
            match transitions[i].next {
                Some(y) => y,
                None if raw_of[i] == RAW_LOST => u32::MAX,
                None => ns as u32 + raw_of[i],
            }
        };
        let max_raw = raw_of
            .iter()
            .filter(|&&r| r != RAW_LOST)
            .max()
            .map_or(0, |&r| r + 1);
        if (ns as u64) + u64::from(max_raw) < u64::from(u16::MAX) {
            Codes::U16(
                (0..transitions.len())
                    .map(|i| match code(i) {
                        u32::MAX => u16::MAX,
                        c => c as u16,
                    })
                    .collect(),
                ns as u32,
            )
        } else {
            Codes::U32((0..transitions.len()).map(code).collect(), ns as u32)
        }
    }

    #[inline(always)]
    fn get(&self, i: usize) -> Cur {
        let (c, ns) = match self {
            Codes::U16(v, ns) => match v[i] {
                u16::MAX => return Cur::Lost,
                c => (u32::from(c), *ns),
            },
            Codes::U32(v, ns) => match v[i] {
                u32::MAX => return Cur::Lost,
                c => (c, *ns),
            },
        };
        if c < ns { Cur::Id(c) } else { Cur::Raw(c - ns) }
    }
}

/// Exact `x % d` for `d ≥ 2` without a division (Lemire, Kaser & Kurz, "Faster Remainder by
/// Direct Computation", 2019: `c = ⌈2^128 / d⌉`, `x mod d = ⌊((c·x) mod 2^128) · d / 2^128⌋`,
/// exact for 64-bit `x` and `d < 2^64`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct FastMod {
    c: u128,
    d: u64,
}

impl FastMod {
    pub(crate) fn new(d: u64) -> FastMod {
        assert!(d >= 2, "FastMod divisor must be at least 2");
        FastMod {
            c: u128::MAX / u128::from(d) + 1,
            d,
        }
    }

    #[inline(always)]
    pub(crate) fn rem(self, x: u64) -> u64 {
        let low = self.c.wrapping_mul(u128::from(x));
        let d = u128::from(self.d);
        let lo = u128::from(low as u64) * d;
        let hi = (low >> 64) * d;
        ((hi + (lo >> 64)) >> 64) as u64
    }
}

/// Largest `j` whose permutations are tabulated (6! = 720 orders).
const PERM_TABLE_MAX: usize = 6;

/// Read-only data of the order pass, shared by worker threads.
struct OrderCtx<'t, const W: usize> {
    codes: &'t Codes,
    k: usize,
    samples: usize,
    exhaustive_max: usize,
    /// `perms[j]`: every permutation of `0..j` in lexicographic order, concatenated (for
    /// `j ≤ min(exhaustive_order_max, PERM_TABLE_MAX)`).
    perms: Vec<Vec<u8>>,
    /// `mods[i]` reduces modulo `i + 1` (Fisher–Yates bounds; index 0 unused).
    mods: Vec<FastMod>,
    /// Stable state → id.
    index: &'t FxMap<Packed<W>, u32>,
    /// Raw states interned by the transition phase (referenced by [`Codes`]).
    raws: &'t [Packed<W>],
    /// The kernel simulator (cloned by workers for on-the-fly steps).
    sim: &'t KSim,
    /// Results of on-the-fly steps computed by any worker (results are pure, so a race only
    /// duplicates work).
    steps: [Mutex<FxMap<StepKey<W>, CurP<W>>>; STEP_SHARDS],
}

/// Shards of [`OrderCtx::steps`].
const STEP_SHARDS: usize = 32;

impl<'t, const W: usize> OrderCtx<'t, W> {
    fn new(
        codes: &'t Codes,
        k: usize,
        opts: &CharzOptions,
        index: &'t FxMap<Packed<W>, u32>,
        raws: &'t [Packed<W>],
        sim: &'t KSim,
    ) -> Self {
        OrderCtx {
            codes,
            k,
            samples: opts.order_samples,
            exhaustive_max: opts.exhaustive_order_max,
            perms: (0..=opts.exhaustive_order_max.min(PERM_TABLE_MAX))
                .map(|j| {
                    let mut out = Vec::new();
                    if j >= 2 {
                        let mut p: Vec<usize> = (0..j).collect();
                        loop {
                            out.extend(p.iter().map(|&x| x as u8));
                            if !next_permutation(&mut p) {
                                break;
                            }
                        }
                    }
                    out
                })
                .collect(),
            mods: (0..=MAX_TABLE_INPUTS as u64)
                .map(|i| FastMod::new((i + 1).max(2)))
                .collect(),
            index,
            raws,
            sim,
            steps: std::array::from_fn(|_| Mutex::new(FxMap::default())),
        }
    }

    fn step_shard(
        &self,
        key: &StepKey<W>,
    ) -> std::sync::MutexGuard<'_, FxMap<StepKey<W>, CurP<W>>> {
        let mut h = FxHasher::default();
        key.hash(&mut h);
        self.steps[(h.finish() >> 40) as usize % STEP_SHARDS]
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The simultaneous result `T(s, v).next`.
    #[inline]
    fn simul(&self, s: u32, v: u32) -> Option<u32> {
        match self.codes.get(((s as usize) << self.k) | v as usize) {
            Cur::Id(x) => Some(x),
            _ => None,
        }
    }

    /// Every order the reference checks for entry `(s, u, v)`, as input-bit masks, into
    /// `masks` (exhaustive orders in lexicographic permutation order; sampled orders as the
    /// reference's cumulative seeded shuffles). False (masks not built) for exhaustive orders of
    /// more than [`PERM_TABLE_MAX`] changes.
    fn gen_masks(&self, masks: &mut Vec<u32>, s: u32, u: u32, v: u32) -> bool {
        let diff = u ^ v;
        let j = diff.count_ones() as usize;
        let mut bits = [0u32; MAX_TABLE_INPUTS];
        let mut n = 0;
        for b in 0..self.k {
            if diff >> b & 1 == 1 {
                bits[n] = 1 << b;
                n += 1;
            }
        }
        masks.clear();
        if j <= self.exhaustive_max {
            if j >= self.perms.len() {
                return false;
            }
            masks.extend(self.perms[j].iter().map(|&p| bits[p as usize]));
        } else {
            let mut rng = XorShift::new(
                0x9E37_79B9_7F4A_7C15 ^ (u64::from(s) << 40) ^ (u64::from(u) << 20) ^ u64::from(v),
            );
            masks.resize(self.samples * j, 0);
            let order = &mut bits[..j];
            for c in 0..self.samples {
                for i in (1..j).rev() {
                    // Same value as `rng.next() % (i + 1)` (the reference's shuffle).
                    let r = self.mods[i].rem(rng.next()) as usize;
                    order.swap(i, r);
                }
                masks[c * j..(c + 1) * j].copy_from_slice(order);
            }
        }
        true
    }
}

/// Walk every order of `orders` (`j` input-bit masks each) from state `s` under `u` through the
/// table only (`lost` = the "did not settle" code, `ns` = number of stable states). `None` if
/// some walk reaches a settled non-binary state (needs on-the-fly steps); else `Some(index of
/// the first order ending differently from simul)`. Orders are walked in chunks of interleaved
/// independent chains.
#[allow(clippy::too_many_arguments)]
fn walk_orders_in<T: Copy + Into<u32>>(
    codes: &[T],
    lost: u32,
    ns: u32,
    k: usize,
    orders: &[u32],
    j: usize,
    s: u32,
    u: u32,
    simul: Option<u32>,
) -> Option<Option<usize>> {
    const CHUNK: usize = 16;
    const LOST: u32 = u32::MAX;
    let n = orders.len() / j.max(1);
    let want = simul.unwrap_or(LOST);
    let mut base = 0;
    while base < n {
        let cnt = CHUNK.min(n - base);
        let mut cur = [s; CHUNK];
        let mut w = [u; CHUNK];
        for i in 0..j {
            for c in 0..cnt {
                w[c] ^= orders[(base + c) * j + i];
                let x = cur[c];
                if x == LOST {
                    continue;
                }
                let code: u32 = codes[((x as usize) << k) | w[c] as usize].into();
                cur[c] = if code == lost {
                    LOST
                } else if code >= ns {
                    return None;
                } else {
                    code
                };
            }
        }
        if let Some(c) = (0..cnt).find(|&c| cur[c] != want) {
            return Some(Some(base + c));
        }
        base += cnt;
    }
    Some(None)
}

/// Exposed for equivalence tests: the fast simulator's `T(state, old, new)`.
pub(crate) fn sim_transition(
    sim: &mut KSim,
    state: &[Logic],
    old: u32,
    new: u32,
) -> (Vec<Logic>, bool, u32) {
    let k = sim.num_inputs();
    let o: Outcome<{ MAX_FAST_NETS / 64 }> =
        sim.transition(state, &input_vector(old, k), &input_vector(new, k));
    let mut out = vec![Logic::X; sim.num_state()];
    o.state.unpack_into(&mut out);
    (out, o.quiescent, o.latency)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrip_and_x_detection() {
        let vals = [Logic::Zero, Logic::One, Logic::Z, Logic::One];
        let p = Packed::<1>::pack(&vals);
        let mut out = [Logic::X; 4];
        p.unpack_into(&mut out);
        assert_eq!(out, vals);
        assert!(!p.has_x());
        assert!(Packed::<1>::pack(&[Logic::Zero, Logic::X, Logic::Z]).has_x());
        for n in [64usize, 65, 128, 200, 256] {
            let mut big = vec![Logic::Z; n];
            big[n / 3] = Logic::One;
            let p = Packed::<4>::pack(&big);
            assert!(!p.has_x());
            let mut out = vec![Logic::X; n];
            p.unpack_into(&mut out);
            assert_eq!(out, big);
            for i in [0, 63, 64, n - 1].into_iter().filter(|&i| i < n) {
                let mut v = big.clone();
                v[i] = Logic::X;
                assert!(Packed::<4>::pack(&v).has_x(), "{n} {i}");
            }
        }
    }

    #[test]
    fn fastmod_is_exact() {
        let mut rng = XorShift::new(12345);
        for d in 2..=200u64 {
            let fm = FastMod::new(d);
            for x in [
                0,
                1,
                d - 1,
                d,
                d + 1,
                u64::MAX,
                u64::MAX - 1,
                u64::MAX / d * d,
            ] {
                assert_eq!(fm.rem(x), x % d, "{x} % {d}");
            }
            for _ in 0..20_000 {
                let x = rng.next();
                assert_eq!(fm.rem(x), x % d, "{x} % {d}");
            }
        }
        for d in [1u64 << 32, (1 << 63) + 12345, u64::MAX] {
            let fm = FastMod::new(d);
            for _ in 0..10_000 {
                let x = rng.next();
                assert_eq!(fm.rem(x), x % d, "{x} % {d}");
            }
        }
    }

    #[test]
    fn fx_hasher_distinguishes() {
        let mut a = FxHasher::default();
        a.write_u32(1);
        let mut b = FxHasher::default();
        b.write_u32(2);
        assert_ne!(a.finish(), b.finish());
    }
}
