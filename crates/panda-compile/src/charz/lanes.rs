//! Bit-parallel transitions: 64 transitions from one stable state (one per new input vector)
//! simulated at once in the two-plane word encoding (SPEC Appendix A).
//!
//! # Exactness
//!
//! Each lane follows exactly the [`KSim`](super::fast::KSim) trajectory of
//! `transition_from_stable(state, home, v)`:
//!
//! * The start state is a fixed point of every gate under the home input, so tick 1 changes
//!   only the inputs (to the lane's `v`).
//! * Every gate is evaluated in every lane at every tick. As in [`super::fast`], this equals the
//!   event-driven evaluation (a gate the event queue would skip cannot change), and a lane's
//!   event queue is non-empty after tick `t` iff a net with a gate reader changed in that lane at
//!   tick `t` — that decides quiescence exactly. A quiescent lane keeps its values.
//! * **Resolvers** (zero delay) are re-evaluated in every lane after the gates (and after the
//!   input change at tick 1), word-parallel with exactly `eval::resolve`'s rule (non-`Z` drivers
//!   all `0` / all `1` → that value, otherwise `X`; no active driver → the termination, a
//!   keeper holding its current value with `Z` read as `X`). `TimingSim` re-evaluates only
//!   resolvers whose drivers changed, which gives the same value except for a resolver held at
//!   `X` by a collapse — and lanes that may collapse are handed back (below). Resolver
//!   preconditions are those of the scalar simulator (no resolver reads a resolver output).
//! * **Period-2 collapse** at tick `t` needs a candidate: a net whose alternation run is
//!   `≥ L = 2·p2_limit`, which implies the net changed on each of the ticks `t-L+1 ..= t`. Every
//!   net keeps a bit-sliced count of consecutive ticks with a change; once some net of a lane
//!   reaches `L`, the lane is flagged and reported as unfinished (the caller recomputes it with
//!   the scalar simulator). An unflagged lane never had a collapse candidate, so its trajectory
//!   is exactly the collapse-free one. With collapse off (`L = 0`) nothing is flagged.
//!
//! Gate word ops are the panda-netlist two-plane ops folded exactly like
//! [`panda_netlist::eval::eval_gate`] (identity-seeded left fold, then inversion); a unit test
//! compares them with `eval_gate` for every op, arity 1–4 and every input combination.

use panda_netlist::eval::{w_and, w_norm, w_not, w_or, w_tri, w_xor};
use panda_netlist::flat::{FlatNetlist, FlatOp};
use panda_netlist::{Logic, Termination};

use super::CharzOptions;
use super::fast::Packed;
use crate::subnet::SubnetMap;

/// Lanes per batch.
pub(crate) const LANES: usize = 64;

/// From this tick on, a batch with at most [`HANDOFF_LANES`] lanes still running stops; those
/// lanes are recomputed by the scalar simulator (exact either way; a pure cost trade-off).
const HANDOFF_TICK: u64 = 24;
/// See [`HANDOFF_TICK`].
const HANDOFF_LANES: u32 = 4;

/// `LANE_BITS[i]` has bit `l` set iff bit `i` of `l` is set (`l < 64`).
const LANE_BITS: [u64; 6] = [
    0xAAAA_AAAA_AAAA_AAAA,
    0xCCCC_CCCC_CCCC_CCCC,
    0xF0F0_F0F0_F0F0_F0F0,
    0xFF00_FF00_FF00_FF00,
    0xFFFF_0000_FFFF_0000,
    0xFFFF_FFFF_0000_0000,
];

/// Transpose a 64×64 bit matrix in place: afterwards bit `c` of `x[r]` is the former bit `r` of
/// `x[c]` (recursive block swaps, Hacker's Delight §7-3 adapted to LSB-first indexing).
fn transpose64(x: &mut [u64; 64]) {
    let mut j = 32usize;
    let mut mask: u64 = 0x0000_0000_FFFF_FFFF;
    while j != 0 {
        let mut kk = 0usize;
        while kk < 64 {
            // Swap the high-j bits of row kk with the low-j bits of row kk + j.
            let t = ((x[kk] >> j) ^ x[kk + j]) & mask;
            x[kk] ^= t << j;
            x[kk + j] ^= t;
            kk = (kk + j + 1) & !j;
        }
        j >>= 1;
        mask ^= mask << j;
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Not,
    Buf,
    Tri,
    And { inv: bool },
    Or { inv: bool },
    Xor { inv: bool },
}

#[derive(Clone, Copy)]
struct LGate {
    kind: Kind,
    /// Inputs `ins[start..end]`.
    start: u32,
    end: u32,
    out: u32,
}

#[derive(Clone, Copy)]
struct LRes {
    /// Drivers `ins[start..end]`.
    start: u32,
    end: u32,
    term: Termination,
    out: u32,
}

/// Word-parallel `eval::resolve(drivers, term, held).value` over lanes (planes `(a, b)`:
/// `0=(0,0)`, `1=(1,0)`, `Z=(0,1)`, `X=(1,1)`).
#[inline]
fn resolve_word(
    drivers: impl Iterator<Item = (u64, u64)>,
    term: Termination,
    held: (u64, u64),
) -> (u64, u64) {
    let (mut any, mut has0, mut has1, mut hasx) = (0u64, 0u64, 0u64, 0u64);
    for (a, b) in drivers {
        any |= a | !b; // not Z
        has0 |= !a & !b;
        has1 |= a & !b;
        hasx |= a & b;
    }
    let only0 = has0 & !has1 & !hasx;
    let only1 = has1 & !has0 & !hasx;
    let x = any & !only0 & !only1;
    let (da, db) = (only1 | x, x);
    let (ta, tb) = match term {
        Termination::None => (0, u64::MAX),
        Termination::PullUp => (u64::MAX, 0),
        Termination::PullDown => (0, 0),
        Termination::Keeper => w_norm(held.0, held.1),
    };
    ((da & any) | (ta & !any), (db & any) | (tb & !any))
}

/// Result of one lane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LaneOut<const W: usize> {
    /// The lane's result is final (else: recompute with the scalar simulator).
    pub done: bool,
    pub quiescent: bool,
    pub latency: u32,
    /// Packed final state.
    pub state: Packed<W>,
}

/// The bit-parallel engine of one kernel.
#[derive(Clone)]
pub(crate) struct Lanes {
    k: usize,
    m: usize,
    gates: Vec<LGate>,
    res: Vec<LRes>,
    ins: Vec<u32>,
    /// Nets read by some gate.
    read: Vec<bool>,
    /// Settle bound (ticks).
    bound: u64,
    /// `L = 2·p2_limit` (0 = no collapse).
    limit: u64,
    /// Bits of the consecutive-change counters (`2^cbits > L`).
    cbits: usize,
    /// Per net, `cbits` planes: consecutive ticks with a change, per lane.
    cnt: Vec<u64>,
    // ---- scratch ----
    a: Vec<u64>,
    b: Vec<u64>,
    na: Vec<u64>,
    nb: Vec<u64>,
    last: [u32; LANES],
    /// Transposition blocks (64 planes each).
    tr: Vec<[u64; 64]>,
}

impl Lanes {
    /// The engine for a fast-path kernel (the caller has checked the layout and resolver
    /// preconditions with [`super::fast::KSim::new`]), or `None` for a zero settle bound.
    pub(crate) fn new(sub: &FlatNetlist, map: &SubnetMap, opts: &CharzOptions) -> Option<Lanes> {
        let k = map.num_inputs();
        let m = map.num_state();
        let n = k + m;
        let mut gates = Vec::with_capacity(sub.gates.len());
        let mut ins = Vec::new();
        let mut read = vec![false; n];
        for g in &sub.gates {
            let kind = match g.op {
                FlatOp::Not => Kind::Not,
                FlatOp::Buf => Kind::Buf,
                FlatOp::Tri => Kind::Tri,
                FlatOp::And => Kind::And { inv: false },
                FlatOp::Nand => Kind::And { inv: true },
                FlatOp::Or => Kind::Or { inv: false },
                FlatOp::Nor => Kind::Or { inv: true },
                FlatOp::Xor => Kind::Xor { inv: false },
                FlatOp::Xnor => Kind::Xor { inv: true },
            };
            let start = ins.len() as u32;
            for &i in &g.inputs {
                read[i as usize] = true;
                ins.push(i);
            }
            gates.push(LGate {
                kind,
                start,
                end: ins.len() as u32,
                out: g.output,
            });
        }
        let mut res = Vec::with_capacity(sub.resolvers.len());
        for r in &sub.resolvers {
            let start = ins.len() as u32;
            ins.extend_from_slice(&r.drivers);
            res.push(LRes {
                start,
                end: ins.len() as u32,
                term: r.termination,
                out: r.output,
            });
        }
        let limit = u64::from(opts.timing.p2_limit.saturating_mul(2));
        let bound = opts.bound_for(m);
        if bound == 0 {
            return None;
        }
        let cbits = (64 - limit.leading_zeros()) as usize;
        Some(Lanes {
            k,
            m,
            gates,
            res,
            ins,
            read,
            bound,
            limit,
            cbits,
            cnt: vec![0; n * cbits],
            a: vec![0; n],
            b: vec![0; n],
            na: vec![0; n],
            nb: vec![0; n],
            last: [0; LANES],
            tr: vec![[0; 64]; (2 * m).div_ceil(64).max(1)],
        })
    }

    /// Re-evaluate every resolver (after the gates of tick `t`, or after the inputs at tick 1).
    /// Returns `(lanes with a state change, lanes whose event queue became non-empty, lanes that
    /// may collapse)`.
    #[inline]
    fn resolvers(&mut self) -> (u64, u64, u64) {
        let (mut chg_all, mut active, mut bad) = (0u64, 0u64, 0u64);
        for ri in 0..self.res.len() {
            let r = self.res[ri];
            let out = r.out as usize;
            let drivers = self.ins[r.start as usize..r.end as usize]
                .iter()
                .map(|&d| (self.a[d as usize], self.b[d as usize]));
            let (a, b) = resolve_word(drivers, r.term, (self.a[out], self.b[out]));
            let chg = (a ^ self.a[out]) | (b ^ self.b[out]);
            if chg != 0 {
                self.a[out] = a;
                self.b[out] = b;
                chg_all |= chg;
                if self.read[out] {
                    active |= chg;
                }
                bad |= self.count(out, chg);
            } else if self.limit != 0 {
                self.reset_count(out);
            }
        }
        (chg_all, active, bad)
    }

    /// Count a change of `net` in lanes `chg` (other lanes restart at 0). Returns the lanes
    /// whose count reached `L` (possible collapse).
    #[inline]
    fn count(&mut self, net: usize, chg: u64) -> u64 {
        if self.limit == 0 {
            return 0;
        }
        let c = &mut self.cnt[net * self.cbits..(net + 1) * self.cbits];
        let mut carry = chg;
        for p in c.iter_mut() {
            let old = *p;
            *p = (old ^ carry) & chg;
            carry &= old;
        }
        // Lanes with count ≥ L (bit-sliced comparison with the constant, MSB first). An
        // overflowing lane has already been flagged (it passed L first).
        let (mut gt, mut eq) = (carry, u64::MAX);
        for i in (0..self.cbits).rev() {
            if self.limit >> i & 1 == 1 {
                eq &= c[i];
            } else {
                gt |= eq & c[i];
                eq &= !c[i];
            }
        }
        (gt | eq) & chg
    }

    /// No change of `net` this tick: its consecutive-change count restarts.
    #[inline]
    fn reset_count(&mut self, net: usize) {
        self.cnt[net * self.cbits..(net + 1) * self.cbits].fill(0);
    }

    #[inline(always)]
    fn eval(&self, g: LGate) -> (u64, u64) {
        let ins = &self.ins[g.start as usize..g.end as usize];
        let at = |i: u32| (self.a[i as usize], self.b[i as usize]);
        match g.kind {
            Kind::Not => {
                let (a, b) = at(ins[0]);
                w_not(a, b)
            }
            Kind::Buf => {
                let (a, b) = at(ins[0]);
                w_norm(a, b)
            }
            Kind::Tri => {
                let (da, db) = at(ins[0]);
                let (ea, eb) = at(ins[1]);
                w_tri(da, db, ea, eb)
            }
            Kind::And { inv } => {
                let (mut a, mut b) = (u64::MAX, 0);
                for &i in ins {
                    let (x, y) = at(i);
                    (a, b) = w_and(a, b, x, y);
                }
                if inv { w_not(a, b) } else { (a, b) }
            }
            Kind::Or { inv } => {
                let (mut a, mut b) = (0, 0);
                for &i in ins {
                    let (x, y) = at(i);
                    (a, b) = w_or(a, b, x, y);
                }
                if inv { w_not(a, b) } else { (a, b) }
            }
            Kind::Xor { inv } => {
                let (mut a, mut b) = (0, 0);
                for &i in ins {
                    let (x, y) = at(i);
                    (a, b) = w_xor(a, b, x, y);
                }
                if inv { w_not(a, b) } else { (a, b) }
            }
        }
    }

    /// Transitions from `state` (a fixed point of every gate under input vector `home`) to the
    /// input vectors `base + l` for lanes `l < lanes` (`lanes ≤ 64`); results in `out[..lanes]`.
    pub(crate) fn run<const W: usize>(
        &mut self,
        state: &[Logic],
        home: u32,
        base: u32,
        lanes: usize,
        out: &mut [LaneOut<W>],
    ) {
        debug_assert!(lanes <= LANES && lanes > 0);
        let k = self.k;
        let lane_mask = if lanes == LANES {
            u64::MAX
        } else {
            (1u64 << lanes) - 1
        };
        // Tick 0: every lane holds the start state under `home`.
        for i in 0..k {
            self.a[i] = if home >> i & 1 == 1 { u64::MAX } else { 0 };
            self.b[i] = 0;
        }
        for (j, &v) in state.iter().enumerate() {
            let (pa, pb) = v.planes();
            self.a[k + j] = if pa { u64::MAX } else { 0 };
            self.b[k + j] = if pb { u64::MAX } else { 0 };
        }
        self.cnt.fill(0);
        let mut bad = 0u64; // lanes that may collapse: recomputed by the caller
        // Tick 1: gates change nothing (fixed point); inputs take the lane's vector.
        let mut active = 0u64; // lanes whose event queue is non-empty
        debug_assert!(base.is_multiple_of(LANES as u32));
        for i in 0..k {
            // Bit i of `base + l` over the lanes (`base` is a multiple of 64).
            let plane = match LANE_BITS.get(i) {
                Some(&p) => p,
                None if base >> i & 1 == 1 => u64::MAX,
                None => 0,
            } & lane_mask;
            let changed = plane ^ self.a[i];
            self.a[i] = plane;
            if self.read[i] {
                active |= changed;
            }
            if changed != 0 {
                bad |= self.count(i, changed);
            }
        }
        if !self.res.is_empty() {
            // Zero-delay resolvers see the new inputs at tick 1 (latency is 0 either way).
            let (_, act, b) = self.resolvers();
            active |= act;
            bad |= b;
        }
        active &= lane_mask;
        // Latency bookkeeping: last tick with a state change (start = 1 → latency 0).
        self.last[..lanes].fill(1);
        let mut done = !active & lane_mask;
        let mut quiescent = done;
        let mut t = 1u64;
        while active & !bad != 0 && t < self.bound {
            if t >= HANDOFF_TICK && (active & !bad).count_ones() <= HANDOFF_LANES {
                // A few long-running lanes: cheaper on the scalar simulator.
                break;
            }
            t += 1;
            for gi in 0..self.gates.len() {
                let g = self.gates[gi];
                let (a, b) = self.eval(g);
                self.na[gi] = a;
                self.nb[gi] = b;
            }
            let mut next_active = 0u64;
            let mut state_chg = 0u64;
            for gi in 0..self.gates.len() {
                let out = self.gates[gi].out as usize;
                let chg = (self.na[gi] ^ self.a[out]) | (self.nb[gi] ^ self.b[out]);
                if chg != 0 {
                    self.a[out] = self.na[gi];
                    self.b[out] = self.nb[gi];
                    state_chg |= chg;
                    if self.read[out] {
                        next_active |= chg;
                    }
                    bad |= self.count(out, chg);
                } else if self.limit != 0 {
                    self.reset_count(out);
                }
            }
            if !self.res.is_empty() {
                let (chg, act, b) = self.resolvers();
                state_chg |= chg;
                next_active |= act;
                bad |= b;
            }
            // Inputs do not change after tick 1.
            if t == 2 && self.limit != 0 {
                for i in 0..k {
                    self.reset_count(i);
                }
            }
            // Only active lanes can change; record the latency tick.
            let mut bits = state_chg & lane_mask;
            while bits != 0 {
                let l = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                self.last[l] = t as u32;
            }
            active = next_active & lane_mask;
            let newly = !active & !done & lane_mask;
            done |= newly;
            quiescent |= newly;
        }
        // Lanes still active at the settle bound are final and not quiescent; flagged and
        // handed-off lanes are unfinished.
        if t >= self.bound {
            done |= active;
        }
        done &= !bad;
        if bad & lane_mask != 0 {
            super::fast::cov::hit(&super::fast::cov::LANE_FALLBACK);
        }
        // Packed states: code bit 0 = a ^ b, bit 1 = b (0=(0,0)→0, 1=(1,0)→1, X=(1,1)→2,
        // Z=(0,1)→3); plane 2j / 2j+1 = code bits of state net j, transposed to lanes in
        // blocks of 64 planes.
        let m = self.m;
        debug_assert!(m <= Packed::<W>::NETS);
        let blocks = (2 * m).div_ceil(64);
        for blk in &mut self.tr[..blocks] {
            *blk = [0; 64];
        }
        for j in 0..m {
            let (a, b) = (self.a[k + j], self.b[k + j]);
            let p0 = 2 * j;
            self.tr[p0 / 64][p0 % 64] = a ^ b;
            self.tr[(p0 + 1) / 64][(p0 + 1) % 64] = b;
        }
        for blk in &mut self.tr[..blocks] {
            transpose64(blk);
        }
        for (l, o) in out.iter_mut().enumerate().take(lanes) {
            let bit = 1u64 << l;
            if done & bit == 0 {
                *o = LaneOut::default();
                continue;
            }
            let mut state = Packed::<W>::default();
            for (bi, blk) in self.tr[..blocks].iter().enumerate() {
                state.0[bi / 2] |= u128::from(blk[l]) << (64 * (bi % 2));
            }
            *o = LaneOut {
                done: true,
                quiescent: quiescent & bit != 0,
                latency: self.last[l] - 1,
                state,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use panda_netlist::eval::eval_gate;

    const ALL: [Logic; 4] = [Logic::Zero, Logic::One, Logic::X, Logic::Z];

    /// Word-parallel resolution equals `eval::resolve` for up to 3 drivers, every termination
    /// and every held value.
    #[test]
    fn resolve_word_matches_resolve() {
        use panda_netlist::eval::resolve;
        let terms = [
            Termination::None,
            Termination::PullUp,
            Termination::PullDown,
            Termination::Keeper,
        ];
        for term in terms {
            for nd in 1..=3usize {
                // combos: drivers (4^nd) × held (4); lanes = combos (≤ 256, in 64-lane chunks)
                let combos = 4usize.pow(nd as u32 + 1);
                for start in (0..combos).step_by(LANES) {
                    let cnt = LANES.min(combos - start);
                    let plane = |slot: usize| -> (u64, u64) {
                        let (mut a, mut b) = (0u64, 0u64);
                        for lane in 0..cnt {
                            let (pa, pb) = ALL[((start + lane) >> (2 * slot)) & 3].planes();
                            a |= u64::from(pa) << lane;
                            b |= u64::from(pb) << lane;
                        }
                        (a, b)
                    };
                    let drivers: Vec<(u64, u64)> = (0..nd).map(plane).collect();
                    let (a, b) = resolve_word(drivers.iter().copied(), term, plane(nd));
                    for lane in 0..cnt {
                        let c = start + lane;
                        let ds: Vec<Logic> = (0..nd).map(|i| ALL[(c >> (2 * i)) & 3]).collect();
                        let held = ALL[(c >> (2 * nd)) & 3];
                        let got = Logic::from_planes(a >> lane & 1 == 1, b >> lane & 1 == 1);
                        assert_eq!(
                            got,
                            resolve(&ds, term, held).value,
                            "{term:?} {ds:?} held {held:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn transpose_matches_naive() {
        let mut seed = 0x1234_5678_9abc_def1u64;
        let mut x = [0u64; 64];
        for r in x.iter_mut() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            *r = seed;
        }
        let orig = x;
        transpose64(&mut x);
        for (r, row) in x.iter().enumerate() {
            for (c, col) in orig.iter().enumerate() {
                assert_eq!(row >> c & 1, col >> r & 1, "({r}, {c})");
            }
        }
    }

    #[test]
    fn lane_bits() {
        for (i, &p) in LANE_BITS.iter().enumerate() {
            for l in 0..64 {
                assert_eq!(p >> l & 1, (l as u64) >> i & 1);
            }
        }
    }

    #[test]
    fn plane_codes() {
        for v in ALL {
            let (a, b) = v.planes();
            let code = u8::from(a ^ b) | u8::from(b) << 1;
            assert_eq!(code, v.code());
        }
    }

    /// Word ops equal `eval_gate` for every op, arity and input combination (lanes = combos).
    #[test]
    fn word_ops_match_eval_gate() {
        let ops = [
            FlatOp::Not,
            FlatOp::Buf,
            FlatOp::Tri,
            FlatOp::And,
            FlatOp::Nand,
            FlatOp::Or,
            FlatOp::Nor,
            FlatOp::Xor,
            FlatOp::Xnor,
        ];
        for op in ops {
            let arities: &[usize] = match op {
                FlatOp::Not | FlatOp::Buf => &[1],
                FlatOp::Tri => &[2],
                _ => &[1, 2, 3, 4],
            };
            for &ar in arities {
                let combos = 4usize.pow(ar as u32);
                let mut l = Lanes {
                    k: ar,
                    m: 1,
                    gates: Vec::new(),
                    res: Vec::new(),
                    tr: Vec::new(),
                    ins: (0..ar as u32).collect(),
                    read: vec![true; ar + 1],
                    bound: 1,
                    limit: 0,
                    cbits: 0,
                    cnt: Vec::new(),
                    a: vec![0; ar + 1],
                    b: vec![0; ar + 1],
                    na: vec![0; 1],
                    nb: vec![0; 1],
                    last: [0; LANES],
                };
                let sub = FlatNetlist {
                    num_nets: ar as u32 + 1,
                    gates: vec![panda_netlist::flat::FlatGate {
                        op,
                        inputs: (0..ar as u32).collect(),
                        output: ar as u32,
                        owner: 0,
                        group: 0,
                        bit: 0,
                    }],
                    ..Default::default()
                };
                let map = SubnetMap {
                    inputs: (0..ar as u32).collect(),
                    nets: vec![ar as u32],
                };
                let built = Lanes::new(&sub, &map, &CharzOptions::default()).expect("lanes");
                l.gates = built.gates;
                for start in (0..combos).step_by(LANES) {
                    for i in 0..ar {
                        let (mut a, mut b) = (0u64, 0u64);
                        for lane in 0..LANES.min(combos - start) {
                            let v = ALL[((start + lane) >> (2 * i)) & 3];
                            let (pa, pb) = v.planes();
                            a |= u64::from(pa) << lane;
                            b |= u64::from(pb) << lane;
                        }
                        l.a[i] = a;
                        l.b[i] = b;
                    }
                    let (a, b) = l.eval(l.gates[0]);
                    for lane in 0..LANES.min(combos - start) {
                        let ins: Vec<Logic> = (0..ar)
                            .map(|i| ALL[((start + lane) >> (2 * i)) & 3])
                            .collect();
                        let got = Logic::from_planes(a >> lane & 1 == 1, b >> lane & 1 == 1);
                        assert_eq!(got, eval_gate(op, &ins), "{op:?} {ins:?}");
                    }
                }
            }
        }
    }
}
