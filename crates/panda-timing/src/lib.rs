//! Timing mode — the reference semantics (SPEC §5).
//!
//! * Discrete ticks. Every [`FlatGate`](panda_netlist::flat::FlatGate) has delay 1; sources and
//!   resolvers have delay 0.
//! * Synchronous update: `V(t+1)[gate.out] = eval_gate(gate, V(t)[gate.inputs])`. Then sources take
//!   their tick-(t+1) values, then resolvers recompute from their (tick t+1) driver nets.
//! * Event-driven: only gates whose inputs changed at tick t are evaluated for t+1.
//! * Gate semantics come exclusively from [`panda_netlist::eval`].
//! * Reset (tick 0): every gate output = X; sources = their tick-0 values (top inputs default to 0
//!   unless set); resolvers computed; keeper state X.
//! * Period-2 collapse (§5.3): if, with no source change during the window, a *self-sustained* set
//!   of nets alternated with period 2 on each of the last `2 * p2_limit` ticks, those nets are
//!   forced to X (and a `Metastable` diagnostic is raised once per net per episode). X then
//!   propagates normally. See "Collapse details" below for the exact rule.
//! * Oscillation (§5.3): if gate events continue for `osc_limit` consecutive ticks with no source
//!   change, an `Oscillating` diagnostic is raised (once per episode) naming some oscillating nets.
//!   The simulation keeps running faithfully.
//! * Contention on a resolver raises a `Contention` diagnostic (see "Episodes").
//! * Input changes via [`TimingSim::set_input`] are applied as source changes at the next tick.
//! * **Ideal blocks** (§5.2, opt-in via [`TimingOptions::ideal_blocks`]): outermost acyclic package
//!   instances behave as `out_j(t + L_j) = f_j(in(t))` (transport delay) instead of being
//!   simulated gate by gate; see the [`ideal`] module for the selection rule, latencies and the
//!   value of internal nets. [`TimingOptions::glass_box`] exempts instances (with their subtrees).
//! * **Stuck X** (§5.4): see "Stuck-X diagnostic" below.
//!
//! # Stuck-X diagnostic
//!
//! Let `P` be the largest period of the running clocks and `R` the latest `PowerOnReset` release
//! tick (0 if there is none). At tick `R + K·P` (`K` = [`TimingOptions::stuck_x_periods`], 0
//! disables the check; never without a running clock) every top-level output pin with an `X` bit
//! raises one `StuckX` warning, located at the pin, naming the X bits and the X origins found by
//! [`x_sources`] (feedback loops of X nets, undriven nets, X sources, contention). The check runs
//! once per reset; it requires stepping through that tick ([`TimingSim::run`] never fast-forwards
//! while a clock is running, so it always does).
//!
//! # Implementation notes
//!
//! These refine the rules above; they do not change them.
//!
//! * **Undriven nets** (no gate, source or resolver) read their termination value
//!   (`Z` for `none`, `1`/`0` for pull-up/down, `X` for a keeper that was never driven).
//! * **Collapse details.** Let `L = 2 * p2_limit` (`p2_limit == 0` disables collapse).
//!   - Every net has an *alternation run*: `run(t) = 0` if `V(t) = V(t-1)`; otherwise
//!     `run(t-1) + 1` if `run(t-1) ≥ 1` and `V(t) = V(t-2)`; otherwise `1`. So `run(t) = k` means
//!     `V(t-k) .. V(t)` alternate between two distinct values. (`run(0) = 0`.)
//!   - At tick `t`, the *candidates* are the nets with `run(t) ≥ L`, provided no source changed on
//!     ticks `t-L+1 ..= t` (tick 0 (reset) counts as a source change, and so does
//!     [`TimingSim::force_net`]).
//!   - The collapsing set `S` is the largest set of candidates such that, for every net in `S`,
//!     every input of its driver (the gate's inputs, or the resolver's driver nets) is in `S` or
//!     did not change on ticks `t-L+1 ..= t`. This is what makes the alternation *self-sustained*:
//!     a metastable latch loop (NAND latch released 00→11, NOT self-loop, master-slave FF with
//!     simultaneous D/clock change) qualifies, while a net that merely carries a long glitch train
//!     (staggered arrival in deep logic, or the parity of a ring oscillator's taps) depends on
//!     inputs that change without alternating, so it is never collapsed. Nets without a driver
//!     are never in `S`.
//!   - Every net in `S` whose value is not already X is set to X at tick `t`. Its driver gate is
//!     re-scheduled, so the synchronous rule `V(t+1) = f(V(t))` holds again from tick `t` on: in a
//!     genuine metastable loop the driver sees X and the loop stays at X. A resolver whose output
//!     collapsed keeps X until one of its drivers changes again. Resolvers reading collapsed nets
//!     see the X at tick `t` (zero delay). `run(t)` is then recomputed from the final values.
//! * **Forcing.** [`TimingSim::force_net`] overrides a net's value at the current tick. It counts as
//!   a source change. A forced gate output keeps its value until the gate is next evaluated (one
//!   tick after one of its inputs changes); a forced resolver output until one of its drivers changes; a
//!   forced source net returns to the source's value at the next tick; an undriven net keeps the
//!   forced value.
//! * **Episodes.** A `Metastable` diagnostic is raised at most once per net per episode, and
//!   `Oscillating` at most once per episode. An episode ends at a source change or when gate activity
//!   stops. A resolver reports `Contention` again only after it has resolved once without contention.
//!   The power-on resolution at tick 0 (gate outputs all X) never reports contention.
//! * **Quiescence** ([`TimingSim::is_quiescent`], [`TimingSim::settle`]): no gate events or input
//!   changes pending, and no time-varying source will change in the future (a running `Clock` never
//!   stops changing; a `PowerOnReset` that has not been released yet will still change).
//! * At most [`MAX_PENDING_DIAGNOSTICS`] untaken diagnostics are stored; the rest are counted
//!   ([`TimingSim::suppressed_diagnostics`]). [`TimingSim::reset`] does not drop pending
//!   diagnostics.
//! * [`TimingSim::run`] fast-forwards over quiescent stretches without evaluating anything; the
//!   result is identical to stepping.
//! * The tick counter is a `u64`; advancing it past `u64::MAX` panics.
//! * **Ideal-block details.** Within a tick, ideal-block outputs due at `t` are applied right
//!   after the gate outputs (they are delay-carrying); blocks whose inputs changed are evaluated at
//!   the end of the tick (after sources, resolvers and collapse), so they see the final tick-`t`
//!   input values. Ticks with pending block-output changes count as activity for the
//!   `Oscillating` rule. For period-2 collapse a block output's "driver inputs" are the block
//!   inputs in its cone, and such an input counts as quiet only if it did not change during the
//!   last `L + L_j - 1` ticks (the output replays older input values for `L_j` ticks; for
//!   `L_j = 1` this is the gate rule).
//!   Collapsing a block output drops its changes still in flight and re-evaluates its driver
//!   (as for gates): it holds X until the value computed from the tick-`t` inputs lands at
//!   `t + L_j`.
//!   [`TimingSim::force_net`] on a block output overrides it until the block next schedules a
//!   change for it; on an internal net it is ignored (internal nets always show the value computed
//!   from the block inputs). Internal nets never appear in [`TimingSim::last_changed`].
//!
//! # Performance
//!
//! The netlist is compiled into struct-of-arrays form: per gate a packed 2-input truth table
//! (built from [`panda_netlist::eval::eval_gate`] via [`eval::table2`]) and two input net indices;
//! gates with more than two inputs fold over a separate CSR input list. Fanout is CSR. Pending gates
//! live in a dense list with a per-gate `scheduled` mark. The hot loop does no allocation and no
//! hashing.

pub mod ideal;
pub mod xtrace;

use std::collections::HashSet;
use std::sync::Arc;

pub use ideal::{IdealBlock, select_ideal_blocks};
pub use xtrace::{XSource, XSourceKind, x_sources};

use panda_netlist::eval;
use panda_netlist::flat::{FlatNetlist, FlatOp, GateIdx, NetIdx, SourceKind};
use panda_netlist::{DiagKind, Diagnostic, Logic, Termination, Value};

/// Sentinel for "no index".
const NONE: u32 = u32::MAX;
/// Maximum number of untaken diagnostics kept; further ones are counted, not stored.
pub const MAX_PENDING_DIAGNOSTICS: usize = 65_536;
/// Maximum number of net names listed in one diagnostic message.
const MAX_NAMES: usize = 8;

/// Tunables of the timing engine.
#[derive(Clone, Debug)]
pub struct TimingOptions {
    /// Period-2 collapse threshold (cycles). Default 4. `0` disables collapse.
    pub p2_limit: u32,
    /// Ticks of continuous activity without source change before `Oscillating`. Default 10_000.
    pub osc_limit: u64,
    /// Evaluate packaged combinational instances as ideal blocks (SPEC §5.2, [`ideal`]).
    /// Default `false` (every gate is simulated).
    pub ideal_blocks: bool,
    /// Instance paths simulated at gate level even when `ideal_blocks` is on (the glass-box
    /// toggle). An entry covers the instance's whole subtree. Default empty.
    pub glass_box: HashSet<String>,
    /// Ideal blocks with at most this many variable inputs are evaluated through lazily filled
    /// LUTs ([`ideal`] module docs). Default 12; `0` disables LUTs (gate evaluation only).
    pub ideal_lut_bits: u32,
    /// Clock periods after reset release before the `StuckX` check (module docs, "Stuck-X
    /// diagnostic"). Default 16; `0` disables it.
    pub stuck_x_periods: u32,
}

impl Default for TimingOptions {
    fn default() -> Self {
        TimingOptions {
            p2_limit: 4,
            osc_limit: 10_000,
            ideal_blocks: false,
            glass_box: HashSet::new(),
            ideal_lut_bits: 12,
            stuck_x_periods: 16,
        }
    }
}

impl TimingOptions {
    /// Builder-style setter for [`TimingOptions::p2_limit`].
    pub fn with_p2_limit(mut self, p2_limit: u32) -> Self {
        self.p2_limit = p2_limit;
        self
    }
    /// Builder-style setter for [`TimingOptions::osc_limit`].
    pub fn with_osc_limit(mut self, osc_limit: u64) -> Self {
        self.osc_limit = osc_limit;
        self
    }
    /// Builder-style setter for [`TimingOptions::ideal_blocks`].
    pub fn with_ideal_blocks(mut self, on: bool) -> Self {
        self.ideal_blocks = on;
        self
    }
    /// Builder-style: add an instance path to [`TimingOptions::glass_box`].
    pub fn with_glass_box(mut self, path: impl Into<String>) -> Self {
        self.glass_box.insert(path.into());
        self
    }
    /// Builder-style setter for [`TimingOptions::ideal_lut_bits`].
    pub fn with_ideal_lut_bits(mut self, bits: u32) -> Self {
        self.ideal_lut_bits = bits;
        self
    }
    /// Builder-style setter for [`TimingOptions::stuck_x_periods`].
    pub fn with_stuck_x_periods(mut self, periods: u32) -> Self {
        self.stuck_x_periods = periods;
        self
    }
}

/// Counters since the last reset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TimingStats {
    /// Ticks simulated (including fast-forwarded quiescent ticks).
    pub ticks: u64,
    /// Gate evaluations performed (gates simulated individually).
    pub gate_evals: u64,
    /// Element (gate / resolver) evaluations performed inside ideal blocks.
    pub block_evals: u64,
    /// Ideal-block updates answered by a LUT row.
    pub lut_hits: u64,
}

/// A time-varying source.
#[derive(Clone, Copy, Debug)]
enum Timed {
    Clock { period: u64, high: u64, phase: u64 },
    PowerOnReset { ticks: u64 },
}

impl Timed {
    /// Value at tick `t` (SPEC §4.1, [`panda_netlist::Atom::Clock`]).
    #[inline]
    fn value_at(self, t: u64) -> Logic {
        match self {
            // `t % period` first: `t + period` would overflow for ticks near `u64::MAX`.
            Timed::Clock { period, high, phase } => {
                Logic::from_bool((t % period + period - phase % period) % period < high)
            }
            Timed::PowerOnReset { ticks } => Logic::from_bool(t < ticks),
        }
    }
    /// True if the value can still change at some tick after `t`.
    fn changes_after(self, t: u64) -> bool {
        match self {
            // `high == 0` is constant 0 and `high >= period` constant 1.
            Timed::Clock { period, high, .. } => high > 0 && high < period,
            Timed::PowerOnReset { ticks } => t < ticks,
        }
    }
}

/// One compiled gate: 16 bytes, so evaluating it touches a single cache line.
///
/// For gates with ≤ 2 inputs, `tab` is the packed truth table indexed by the values of nets `a`
/// and `b` (`b == a` for 1-input gates). For wider gates `tab == 0` (no real gate has the
/// constant-0 table) and `a` indexes [`Compiled::wide`].
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct GateRec {
    tab: u32,
    a: NetIdx,
    b: NetIdx,
    out: NetIdx,
}

/// Gate with more than two inputs: fold `tab` over `inputs[start..end]`, then optionally invert.
#[derive(Clone, Copy, Debug)]
struct WideGate {
    tab: u32,
    start: u32,
    end: u32,
    invert: bool,
}

/// Tag bit marking a resolver (rather than a gate) in the fanout list.
const RES_BIT: u32 = 1 << 31;
/// Tag bit marking an ideal-block input entry in the fanout list (never set together with
/// [`RES_BIT`]). Plain gate entries are `< BLK_BIT`.
const BLK_BIT: u32 = 1 << 30;

/// Packed truth tables of every op, straight from the reference semantics.
pub(crate) struct Tables {
    tab1: [u32; 9],
    tab2: [u32; 9],
    /// Packed NOT table (used to invert wide NAND/NOR/XNOR folds).
    pub(crate) not_tab: u32,
}

impl Tables {
    fn new() -> Self {
        let mut tab1 = [0u32; 9];
        let mut tab2 = [0u32; 9];
        for op in ALL_OPS {
            let i = op_index(op);
            if op != FlatOp::Tri {
                tab1[i] = eval::table2(|a, _| eval::eval_gate(op, &[a]));
            }
            if !matches!(op, FlatOp::Not | FlatOp::Buf) {
                tab2[i] = eval::table2(|a, b| eval::eval_gate(op, &[a, b]));
            }
        }
        Tables { tab1, tab2, not_tab: eval::table2(|a, _| eval::not1(a)) }
    }

    /// Compile flat gate `gi` (`op` over value indices `ins`, writing `out`); wide gates append to
    /// `wide` / `wide_inputs`.
    ///
    /// # Panics
    ///
    /// On a wrong number of inputs for the op.
    pub(crate) fn compile(
        &self,
        op: FlatOp,
        ins: &[u32],
        out: u32,
        wide: &mut Vec<WideGate>,
        wide_inputs: &mut Vec<u32>,
        gi: GateIdx,
    ) -> GateRec {
        // Arity check: a mis-sized NOT/BUF/TRI would otherwise compile to a zero table and be
        // misread as a wide gate.
        let arity_ok = match op {
            FlatOp::Not | FlatOp::Buf => ins.len() == 1,
            FlatOp::Tri => ins.len() == 2,
            _ => !ins.is_empty(),
        };
        assert!(arity_ok, "flat gate {gi}: {op:?} with {} inputs", ins.len());
        let rec = match ins.len() {
            0 => panic!("flat gate {gi} has no inputs"),
            1 => GateRec { tab: self.tab1[op_index(op)], a: ins[0], b: ins[0], out },
            2 => GateRec { tab: self.tab2[op_index(op)], a: ins[0], b: ins[1], out },
            _ => {
                let (base, invert) = match op {
                    FlatOp::And => (FlatOp::And, false),
                    FlatOp::Nand => (FlatOp::And, true),
                    FlatOp::Or => (FlatOp::Or, false),
                    FlatOp::Nor => (FlatOp::Or, true),
                    FlatOp::Xor => (FlatOp::Xor, false),
                    FlatOp::Xnor => (FlatOp::Xor, true),
                    op => panic!("flat gate {gi}: {op:?} with {} inputs", ins.len()),
                };
                let start = wide_inputs.len() as u32;
                wide_inputs.extend_from_slice(ins);
                let w = WideGate { tab: self.tab2[op_index(base)], start, end: wide_inputs.len() as u32, invert };
                wide.push(w);
                GateRec { tab: 0, a: wide.len() as u32 - 1, b: 0, out }
            }
        };
        debug_assert!(ins.len() > 2 || rec.tab != 0);
        rec
    }
}

/// Evaluate compiled gate `g` over `values` (`wide` / `wide_inputs` hold the wide gates).
#[inline(always)]
fn eval_rec(g: GateRec, values: &[Logic], wide: &[WideGate], wide_inputs: &[u32], not_tab: u32) -> Logic {
    if g.tab != 0 {
        eval::lookup2(g.tab, values[g.a as usize], values[g.b as usize])
    } else {
        eval_wide(wide[g.a as usize], values, wide_inputs, not_tab)
    }
}

/// Evaluate a gate with more than two inputs.
#[inline(never)]
fn eval_wide(w: WideGate, values: &[Logic], wide_inputs: &[u32], not_tab: u32) -> Logic {
    let ins = &wide_inputs[w.start as usize..w.end as usize];
    let mut acc = eval::lookup2(w.tab, values[ins[0] as usize], values[ins[1] as usize]);
    for &i in &ins[2..] {
        acc = eval::lookup2(w.tab, acc, values[i as usize]);
    }
    if w.invert { eval::lookup2(not_tab, acc, acc) } else { acc }
}

/// The flat netlist compiled into flat arrays. Immutable after construction.
struct Compiled {
    // ---- gates ----
    gates: Vec<GateRec>,
    wide: Vec<WideGate>,
    wide_inputs: Vec<NetIdx>,
    /// Packed NOT table (used to invert wide NAND/NOR/XNOR folds).
    not_tab: u32,
    /// Gates simulated individually (all gates not inside an ideal block).
    sim_gates: Vec<u32>,
    /// Resolvers simulated individually (all resolvers not inside an ideal block).
    sim_res: Vec<u32>,
    // ---- per net ----
    /// CSR fanout: the readers of net `n` are `fanout[fo_start[n]..fo_start[n+1]]`; an entry is a
    /// gate index, a resolver index tagged with [`RES_BIT`], or an ideal-block input entry tagged
    /// with [`BLK_BIT`]. Gates are deduplicated. Gates and resolvers inside ideal blocks are not
    /// listed (their block's input entries are).
    fo_start: Vec<u32>,
    fanout: Vec<u32>,
    /// Gate driving each net, or `NONE`.
    driver_gate: Vec<u32>,
    /// Resolver driving each net, or `NONE`.
    driver_res: Vec<u32>,
    // ---- resolvers ----
    r_start: Vec<u32>,
    r_drivers: Vec<NetIdx>,
    r_term: Vec<Termination>,
    r_out: Vec<NetIdx>,
    // ---- sources ----
    constants: Vec<(NetIdx, Logic)>,
    timed: Vec<(NetIdx, Timed)>,
    /// Per top input pin, per bit: the source net (`NONE` if that bit has no source).
    input_nets: Vec<Vec<NetIdx>>,
    /// Nets with no driver at all, with their constant value.
    undriven: Vec<(NetIdx, Logic)>,
}

/// Index of an op in the table cache.
fn op_index(op: FlatOp) -> usize {
    match op {
        FlatOp::And => 0,
        FlatOp::Or => 1,
        FlatOp::Nand => 2,
        FlatOp::Nor => 3,
        FlatOp::Xor => 4,
        FlatOp::Xnor => 5,
        FlatOp::Not => 6,
        FlatOp::Buf => 7,
        FlatOp::Tri => 8,
    }
}

const ALL_OPS: [FlatOp; 9] = [
    FlatOp::And,
    FlatOp::Or,
    FlatOp::Nand,
    FlatOp::Nor,
    FlatOp::Xor,
    FlatOp::Xnor,
    FlatOp::Not,
    FlatOp::Buf,
    FlatOp::Tri,
];

impl Compiled {
    fn new(nl: &FlatNetlist, tabs: &Tables, blocks: &ideal::Blocks) -> Self {
        let n = nl.num_nets as usize;
        assert!(nl.gates.len() < BLK_BIT as usize && nl.resolvers.len() < BLK_BIT as usize, "netlist too large");
        let in_block_gate = |g: usize| blocks.gate_block.get(g).is_some_and(|&b| b != NONE);
        let in_block_res = |r: usize| blocks.res_block.get(r).is_some_and(|&b| b != NONE);

        let mut c = Compiled {
            gates: Vec::with_capacity(nl.gates.len()),
            wide: Vec::new(),
            wide_inputs: Vec::new(),
            not_tab: tabs.not_tab,
            sim_gates: (0..nl.gates.len()).filter(|&g| !in_block_gate(g)).map(|g| g as u32).collect(),
            sim_res: (0..nl.resolvers.len()).filter(|&r| !in_block_res(r)).map(|r| r as u32).collect(),
            fo_start: Vec::with_capacity(n + 1),
            fanout: Vec::new(),
            driver_gate: vec![NONE; n],
            driver_res: vec![NONE; n],
            r_start: vec![0],
            r_drivers: Vec::new(),
            r_term: Vec::new(),
            r_out: Vec::new(),
            constants: Vec::new(),
            timed: Vec::new(),
            input_nets: nl.top_inputs.iter().map(|p| vec![NONE; p.nets.len()]).collect(),
            undriven: Vec::new(),
        };
        let mut driven = vec![false; n];
        for (gi, g) in nl.gates.iter().enumerate() {
            let rec = tabs.compile(g.op, &g.inputs, g.output, &mut c.wide, &mut c.wide_inputs, gi as GateIdx);
            c.gates.push(rec);
            c.driver_gate[g.output as usize] = gi as u32;
            driven[g.output as usize] = true;
        }
        let fo = nl.fanout();
        c.fo_start.push(0);
        let mut entries = blocks.net_entries.iter().peekable();
        for (net, (gs, rs)) in fo.gates.iter().zip(&fo.resolvers).enumerate() {
            c.fanout.extend(gs.iter().copied().filter(|&g| !in_block_gate(g as usize)));
            c.fanout.extend(rs.iter().filter(|&&r| !in_block_res(r as usize)).map(|&r| r | RES_BIT));
            while let Some(&&(en, e)) = entries.peek() {
                if en as usize != net {
                    break;
                }
                c.fanout.push(e | BLK_BIT);
                entries.next();
            }
            c.fo_start.push(c.fanout.len() as u32);
        }

        for (ri, r) in nl.resolvers.iter().enumerate() {
            c.r_drivers.extend_from_slice(&r.drivers);
            c.r_start.push(c.r_drivers.len() as u32);
            c.r_term.push(r.termination);
            c.r_out.push(r.output);
            c.driver_res[r.output as usize] = ri as u32;
            driven[r.output as usize] = true;
        }
        for s in &nl.sources {
            driven[s.output as usize] = true;
            match s.kind {
                SourceKind::Constant(v) => c.constants.push((s.output, v)),
                SourceKind::Clock { period, high, phase } => {
                    let period = u64::from(period.max(1));
                    c.timed.push((s.output, Timed::Clock { period, high: u64::from(high), phase: u64::from(phase) }))
                }
                SourceKind::PowerOnReset { ticks } => {
                    c.timed.push((s.output, Timed::PowerOnReset { ticks: u64::from(ticks) }))
                }
                SourceKind::TopInput { pin, bit } => c.input_nets[pin as usize][bit as usize] = s.output,
            }
        }
        for (net, d) in driven.iter().enumerate() {
            if !d {
                let term = nl.undriven_termination.get(&(net as NetIdx)).copied().unwrap_or_default();
                c.undriven.push((net as NetIdx, eval::terminate(term, Logic::X)));
            }
        }
        c
    }

    /// True if `net` is driven by a source (cold: linear scan).
    fn is_source(&self, net: NetIdx) -> bool {
        self.driver_gate[net as usize] == NONE
            && self.driver_res[net as usize] == NONE
            && !self.undriven.iter().any(|&(n, _)| n == net)
    }

    /// Value of the source driving `net` at tick `t` (cold: linear scan).
    fn source_value(&self, net: NetIdx, t: u64, inputs: &[Value]) -> Logic {
        if let Some(&(_, v)) = self.constants.iter().find(|&&(n, _)| n == net) {
            return v;
        }
        if let Some(&(_, s)) = self.timed.iter().find(|&&(n, _)| n == net) {
            return s.value_at(t);
        }
        for (pin, nets) in self.input_nets.iter().enumerate() {
            if let Some(bit) = nets.iter().position(|&n| n == net) {
                return inputs[pin].get(bit as u32);
            }
        }
        unreachable!("net {net} is not driven by a source")
    }

    /// Readers of `net` (gate indices, and resolver indices tagged with [`RES_BIT`]).
    #[inline(always)]
    fn readers(&self, net: NetIdx) -> &[u32] {
        let n = net as usize;
        &self.fanout[self.fo_start[n] as usize..self.fo_start[n + 1] as usize]
    }

    /// Evaluate gate `g` over `values`.
    #[inline(always)]
    fn eval(&self, g: GateRec, values: &[Logic]) -> Logic {
        eval_rec(g, values, &self.wide, &self.wide_inputs, self.not_tab)
    }
}

/// History of a net before its first change in the current tick (to recompute or undo it).
#[derive(Clone, Copy, Debug)]
struct Saved {
    /// Value at the previous tick, `V(t-1)`.
    before: Logic,
    /// Previous `Tracker::prev` (= `V(last-1)`).
    prev: Logic,
    /// Previous `Tracker::run`.
    run: u32,
    /// Previous `Tracker::last`.
    last: u64,
}

/// The alternation run after a change at `t` to `new`, given the history before the change
/// (module docs, "Collapse details").
#[inline(always)]
fn next_run(last: u64, run: u32, prev: Logic, new: Logic, t: u64) -> u32 {
    if last + 1 == t && run >= 1 && new == prev { run.saturating_add(1) } else { 1 }
}

/// Per-net change history: drives period-2 collapse and [`TimingSim::last_changed`].
struct Tracker {
    /// Per net: tick of the last change (0 = never; changes at tick 0 are not tracked).
    last: Vec<u64>,
    /// Per net: alternation run as of tick `last`.
    run: Vec<u32>,
    /// Per net: value before the last change, `V(last - 1)`.
    prev: Vec<Logic>,
    /// Nets changed during the current/most recent tick, unique.
    changed: Vec<NetIdx>,
    /// Parallel to `changed`: the history each net had before this tick.
    saved: Vec<Saved>,
    /// Nets whose run reached the collapse threshold this tick (may contain X-valued nets).
    candidates: Vec<NetIdx>,
    /// `2 * p2_limit` (0 = collapse disabled).
    limit: u32,
    /// Some net changed twice in this tick (or was collapsed): `changed`/`run` must be
    /// recomputed from the final values ([`Tracker::fix`]).
    needs_fix: bool,
}

impl Tracker {
    fn new(n: usize, p2_limit: u32) -> Self {
        Tracker {
            last: vec![0; n],
            run: vec![0; n],
            prev: vec![Logic::X; n],
            changed: Vec::new(),
            saved: Vec::new(),
            candidates: Vec::new(),
            limit: p2_limit.saturating_mul(2),
            needs_fix: false,
        }
    }

    fn reset(&mut self) {
        self.last.fill(0);
        self.run.fill(0);
        self.prev.fill(Logic::X);
        self.begin_tick();
    }

    /// Forget the per-tick lists (start of a tick).
    fn begin_tick(&mut self) {
        self.changed.clear();
        self.saved.clear();
        self.candidates.clear();
        self.needs_fix = false;
    }

    /// Record that `net` changed from `old` to `new` at tick `t`.
    #[inline(always)]
    fn record(&mut self, net: NetIdx, old: Logic, new: Logic, t: u64) {
        let n = net as usize;
        let last = self.last[n];
        if last == t {
            // Changed again within the same tick (or at tick 0, which has no history).
            self.needs_fix = true;
            return;
        }
        let (run, prev) = (self.run[n], self.prev[n]);
        self.saved.push(Saved { before: old, prev, run, last });
        self.changed.push(net);
        let r = next_run(last, run, prev, new, t);
        self.last[n] = t;
        self.run[n] = r;
        self.prev[n] = old;
        if self.limit != 0 && r >= self.limit {
            self.candidates.push(net);
        }
    }

    /// Recompute this tick's history from the final `values`: drop nets that ended where they
    /// started (restoring their history) and recompute runs. If `rebuild` is set, `candidates` is
    /// rebuilt too.
    #[cold]
    fn fix(&mut self, values: &[Logic], t: u64, rebuild: bool) {
        if rebuild {
            self.candidates.clear();
        }
        let mut k = 0;
        for i in 0..self.changed.len() {
            let net = self.changed[i];
            let n = net as usize;
            let s = self.saved[i];
            let cur = values[n];
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
                self.candidates.push(net);
            }
            self.changed[k] = net;
            self.saved[k] = s;
            k += 1;
        }
        self.changed.truncate(k);
        self.saved.truncate(k);
        self.needs_fix = false;
    }
}

/// Unit-delay, event-driven reference simulator (SPEC §5).
pub struct TimingSim {
    netlist: Arc<FlatNetlist>,
    opts: TimingOptions,
    c: Compiled,
    /// Current value of every net.
    values: Vec<Logic>,
    /// `sched[..sched_len]` are the gates to evaluate at the next step. Length = gates + 1, so
    /// branch-free appends never go out of bounds.
    sched: Vec<u32>,
    sched_len: usize,
    /// Spare buffer swapped with `sched` (keeps capacity).
    spare: Vec<u32>,
    scheduled: Vec<bool>,
    /// New output values computed in phase 1 of a step (parallel to the current list).
    new_vals: Vec<(NetIdx, Logic)>,
    // ---- resolvers ----
    res_dirty: Vec<u32>,
    res_marked: Vec<bool>,
    /// Resolver outputs pinned to X by a collapse during the current pass.
    res_hold: Vec<bool>,
    contending: Vec<bool>,
    res_buf: Vec<Logic>,
    // ---- sources ----
    input_values: Vec<Value>,
    inputs_pending: bool,
    /// Source nets overridden by [`TimingSim::force_net`]; they get their source value back at
    /// the next tick.
    forced_sources: Vec<NetIdx>,
    // ---- history / diagnostics ----
    trk: Tracker,
    /// Scratch mark for the collapse closure (all false between steps).
    p2_mark: Vec<bool>,
    tick: u64,
    last_src_change: u64,
    /// Consecutive ticks with gate-output changes.
    active_run: u64,
    osc_reported: bool,
    epoch: u64,
    reported_epoch: Vec<u64>,
    diags: Vec<Diagnostic>,
    /// Diagnostics dropped because `diags` was full.
    suppressed: u64,
    stats: TimingStats,
    // ---- ideal blocks / stuck X ----
    blk: ideal::Blocks,
    /// Scratch for contention reports from block-internal resolvers.
    blk_contention: Vec<(NetIdx, Vec<Logic>)>,
    /// Tick of the stuck-X check (`u64::MAX` = none).
    stuck_check: u64,
}

impl TimingSim {
    /// Compile `netlist` and power on (see [`TimingSim::reset`]).
    ///
    /// # Panics
    ///
    /// If the netlist is malformed (a gate with the wrong number of inputs, a net index out of
    /// range).
    pub fn new(netlist: Arc<FlatNetlist>, opts: TimingOptions) -> Self {
        let tabs = Tables::new();
        let blk = ideal::Blocks::new(&netlist, opts.ideal_blocks, &opts.glass_box, opts.ideal_lut_bits, &tabs);
        let c = Compiled::new(&netlist, &tabs, &blk);
        let n = netlist.num_nets as usize;
        let ng = netlist.gates.len();
        let nr = netlist.resolvers.len();
        let input_values = netlist.top_inputs.iter().map(|p| Value::zeros(p.nets.len() as u32)).collect();
        let mut sim = TimingSim {
            trk: Tracker::new(n, opts.p2_limit),
            p2_mark: vec![false; n],
            opts,
            c,
            values: vec![Logic::X; n],
            sched: vec![0; ng + 1],
            sched_len: 0,
            spare: vec![0; ng + 1],
            scheduled: vec![false; ng],
            new_vals: Vec::with_capacity(ng),
            res_dirty: Vec::with_capacity(nr),
            res_marked: vec![false; nr],
            res_hold: vec![false; nr],
            contending: vec![false; nr],
            res_buf: Vec::new(),
            input_values,
            inputs_pending: false,
            forced_sources: Vec::new(),
            tick: 0,
            last_src_change: 0,
            active_run: 0,
            osc_reported: false,
            epoch: 0,
            reported_epoch: vec![u64::MAX; n],
            diags: Vec::new(),
            suppressed: 0,
            stats: TimingStats::default(),
            blk,
            blk_contention: Vec::new(),
            stuck_check: u64::MAX,
            netlist,
        };
        sim.reset();
        sim
    }

    /// The simulated netlist.
    pub fn netlist(&self) -> &Arc<FlatNetlist> {
        &self.netlist
    }

    /// The options this simulator was created with.
    pub fn options(&self) -> &TimingOptions {
        &self.opts
    }

    /// Back to tick 0 (power-on: all gate outputs X). Input pin values set so far are kept;
    /// pending diagnostics are kept too (take them first if they are not wanted).
    pub fn reset(&mut self) {
        self.values.fill(Logic::Z);
        for g in &self.c.gates {
            self.values[g.out as usize] = Logic::X;
        }
        for &out in &self.c.r_out {
            self.values[out as usize] = Logic::X;
        }
        for &(net, v) in &self.c.undriven {
            self.values[net as usize] = v;
        }
        for &(net, v) in &self.c.constants {
            self.values[net as usize] = v;
        }
        for &(net, t) in &self.c.timed {
            self.values[net as usize] = t.value_at(0);
        }
        for (pin, nets) in self.c.input_nets.iter().enumerate() {
            for (bit, &net) in nets.iter().enumerate() {
                if net != NONE {
                    self.values[net as usize] = self.input_values[pin].get(bit as u32);
                }
            }
        }
        self.inputs_pending = false;
        self.forced_sources.clear();
        self.tick = 0;
        self.last_src_change = 0;
        self.active_run = 0;
        self.osc_reported = false;
        self.next_epoch();
        self.stats = TimingStats::default();

        // Power-on evaluation: every (individually simulated) gate is evaluated for tick 1.
        // Gates inside ideal blocks stay marked as scheduled, so they are never queued.
        let ng = self.c.sim_gates.len();
        self.sched[..ng].copy_from_slice(&self.c.sim_gates);
        self.sched_len = ng;
        self.scheduled.fill(true);

        self.contending.fill(false);
        self.res_hold.fill(false);
        self.res_dirty.clear();
        self.res_dirty.extend_from_slice(&self.c.sim_res);
        self.res_marked.fill(true);
        self.trk.reset();
        self.process_resolvers(0);
        // Ideal blocks: full evaluation from the tick-0 inputs.
        self.blk.reset();
        self.process_blocks(0);
        self.trk.begin_tick();
        self.stuck_check = self.stuck_check_tick();
    }

    /// Set a top-level input pin (by name). Width must match. Takes effect at the next tick.
    pub fn set_input(&mut self, pin: &str, value: &Value) -> Result<(), String> {
        let idx = self
            .netlist
            .top_inputs
            .iter()
            .position(|p| p.name == pin)
            .ok_or_else(|| format!("no top-level input pin '{pin}'"))?;
        let width = self.netlist.top_inputs[idx].nets.len() as u32;
        if value.width() != width {
            return Err(format!("input pin '{pin}' has width {width}, got a {}-bit value", value.width()));
        }
        if self.input_values[idx] != *value {
            self.input_values[idx] = value.clone();
            self.inputs_pending = true;
        }
        Ok(())
    }

    /// The value most recently given to a top-level input pin (default 0), if the pin exists.
    pub fn input_value(&self, pin: &str) -> Option<&Value> {
        let idx = self.netlist.top_inputs.iter().position(|p| p.name == pin)?;
        Some(&self.input_values[idx])
    }

    /// Force a net's value at the current tick (testing / power-on randomization).
    ///
    /// The gates reading the net are scheduled and the resolvers reading it are recomputed
    /// immediately (they have zero delay). The forcing counts as a source change. The net's own
    /// driver is *not* re-evaluated because of it: a forced gate output persists until the gate is
    /// next evaluated (one tick after one of its inputs changes; at tick 1 after a reset, when
    /// every gate is evaluated), a forced resolver output until one of its drivers changes, and an
    /// undriven net keeps the forced value; a forced source net gets its source value back at the
    /// next tick.
    /// [`TimingSim::last_changed`] is updated to include the forced change.
    ///
    /// With ideal blocks: forcing an internal net of a block is ignored; a forced block output keeps
    /// its value until the block next schedules a change for it.
    pub fn force_net(&mut self, net: NetIdx, value: Logic) {
        let n = net as usize;
        let old = self.values[n];
        if old == value || self.blk.is_internal(net) {
            return;
        }
        if !self.blk.is_empty() {
            self.blk.forcing(net);
        }
        let t = self.tick;
        self.values[n] = value;
        self.trk.record(net, old, value, t);
        self.schedule_fanout(net);
        if self.c.is_source(net) {
            self.forced_sources.push(net);
        }
        self.process_resolvers(t);
        if self.blk.has_work() {
            self.process_blocks(t);
        }
        // A forced change rewrites tick `t`: recompute its history from the final values.
        self.trk.fix(&self.values, t, false);
        self.trk.candidates.clear();
        self.source_changed(t);
    }

    /// Advance one tick.
    ///
    /// # Panics
    ///
    /// If the tick counter would overflow `u64`.
    pub fn step(&mut self) {
        let t = self.tick.checked_add(1).expect("tick counter overflow");
        self.trk.begin_tick();

        // ---- gates: V(t)[out] = f(V(t-1)[inputs]) ----
        std::mem::swap(&mut self.sched, &mut self.spare);
        let cur_len = std::mem::replace(&mut self.sched_len, 0);
        let gate_changes;
        {
            let c = &self.c;
            let values = &mut self.values;
            let scheduled = &mut self.scheduled;
            let new_vals = &mut self.new_vals;
            // Phase 1: evaluate against V(t-1); keep only outputs that change. (Each net has one
            // driver, so comparing with the not-yet-updated output value is exact.) Branch-free
            // compaction: always write, advance only on change.
            new_vals.clear();
            new_vals.resize(cur_len, (0, Logic::X));
            let mut k = 0;
            let cur = &self.spare[..cur_len];
            for &g in cur {
                scheduled[g as usize] = false;
                let rec = c.gates[g as usize];
                let v = c.eval(rec, values);
                new_vals[k] = (rec.out, v);
                k += usize::from(v != values[rec.out as usize]);
            }
            new_vals.truncate(k);
            // Phase 2: commit and schedule readers.
            let sched = &mut self.sched;
            let trk = &mut self.trk;
            let res_marked = &mut self.res_marked;
            let res_dirty = &mut self.res_dirty;
            let blk_marked = &mut self.blk.entry_marked;
            let blk_changed = &mut self.blk.changed;
            gate_changes = new_vals.len();
            let mut n = 0usize;
            for &(out, v) in new_vals.iter() {
                let old = values[out as usize];
                values[out as usize] = v;
                trk.record(out, old, v, t);
                for &f in c.readers(out) {
                    if f < BLK_BIT {
                        // Branch-free append: always write, advance only if newly scheduled.
                        sched[n] = f;
                        n += usize::from(!scheduled[f as usize]);
                        scheduled[f as usize] = true;
                    } else if f & RES_BIT != 0 {
                        let r = f & !RES_BIT;
                        if !res_marked[r as usize] {
                            res_marked[r as usize] = true;
                            res_dirty.push(r);
                        }
                    } else {
                        let e = f & !BLK_BIT;
                        if !blk_marked[e as usize] {
                            blk_marked[e as usize] = true;
                            blk_changed.push(e);
                        }
                    }
                }
            }
            self.sched_len = n;
        }
        self.stats.gate_evals += cur_len as u64;
        self.tick = t;
        self.stats.ticks += 1;

        // ---- ideal-block outputs due at t (delay-carrying, like gate outputs) ----
        let gate_changes = if self.blk.pending > 0 { gate_changes + self.land_blocks(t) } else { gate_changes };

        // ---- sources take their tick-t values ----
        let src_changed = self.apply_sources(t);
        if src_changed {
            self.last_src_change = t;
        }

        // ---- resolvers (zero delay) ----
        self.process_resolvers(t);
        if self.trk.needs_fix {
            self.trk.fix(&self.values, t, true);
        }

        // ---- period-2 collapse ----
        if !self.trk.candidates.is_empty() {
            self.collapse(t);
            if self.trk.needs_fix {
                self.trk.fix(&self.values, t, false);
            }
        }

        // ---- ideal blocks whose inputs changed ----
        if self.blk.has_work() {
            self.process_blocks(t);
        }

        // ---- episodes and oscillation ----
        if src_changed {
            self.next_epoch();
            self.osc_reported = false;
        }
        if gate_changes > 0 || self.blk.pending > 0 {
            self.active_run += 1;
            if !self.osc_reported
                && self.active_run >= self.opts.osc_limit
                && t - self.last_src_change >= self.opts.osc_limit
            {
                self.osc_reported = true;
                self.report_oscillation(t);
            }
        } else {
            self.end_activity();
        }

        if t == self.stuck_check {
            self.check_stuck_x(t);
        }
    }

    /// Advance `ticks` ticks. Quiescent stretches are skipped in O(1).
    ///
    /// # Panics
    ///
    /// If the tick counter would overflow `u64`.
    pub fn run(&mut self, ticks: u64) {
        for done in 0..ticks {
            if self.is_quiescent() {
                // Nothing can change any more: fast-forward. A quiescent step changes nothing
                // except ending the current activity episode.
                let rest = ticks - done;
                self.tick = self.tick.checked_add(rest).expect("tick counter overflow");
                self.stats.ticks = self.stats.ticks.saturating_add(rest);
                self.trk.begin_tick();
                self.end_activity();
                return;
            }
            self.step();
        }
    }

    /// Run until no gate events are pending and no source changes at the next tick
    /// (clocks make this never true), or `max_ticks` elapse. Returns true if quiescent.
    ///
    /// A `PowerOnReset` that has not been released yet counts as a pending source change (see
    /// [`TimingSim::is_quiescent`]).
    pub fn settle(&mut self, max_ticks: u64) -> bool {
        for _ in 0..max_ticks {
            if self.is_quiescent() {
                return true;
            }
            self.step();
        }
        self.is_quiescent()
    }

    /// True if nothing will ever change again without outside intervention: no gate events or
    /// input changes are pending and no running clock / unreleased `PowerOnReset` exists.
    pub fn is_quiescent(&self) -> bool {
        self.sched_len == 0
            && self.blk.pending == 0
            && !self.inputs_pending
            && self.forced_sources.is_empty()
            && !self.c.timed.iter().any(|&(_, s)| s.changes_after(self.tick))
    }

    /// Current tick.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Current value of a net.
    ///
    /// For an internal net of an ideal block this is the value computed from the block's current
    /// input values (SPEC §5.2 probing), computed on demand if necessary ([`ideal`] module docs).
    pub fn get(&self, net: NetIdx) -> Logic {
        self.blk.lazy_value(net, &self.values).unwrap_or(self.values[net as usize])
    }

    /// All net values (index = NetIdx).
    ///
    /// Internal nets of ideal blocks that were last updated through a LUT may hold older values
    /// here; call [`TimingSim::refresh_ideal_internal`] first to bring them up to date
    /// ([`TimingSim::get`] / [`TimingSim::read`] are always exact).
    pub fn values(&self) -> &[Logic] {
        &self.values
    }

    /// Recompute the internal nets of ideal blocks last updated through a LUT, so that
    /// [`TimingSim::values`] shows them evaluated from the blocks' current inputs. Does not change
    /// the simulation (outputs, schedule, diagnostics other than internal contention).
    pub fn refresh_ideal_internal(&mut self) {
        if self.blk.any_stale() {
            let mut cont = std::mem::take(&mut self.blk_contention);
            self.blk.refresh(&mut self.values, self.tick, &mut cont);
            self.blk_contention = cont;
            self.emit_block_contention();
        }
    }

    /// The ideal blocks in use (empty unless [`TimingOptions::ideal_blocks`] is set).
    pub fn ideal_blocks(&self) -> &[IdealBlock] {
        &self.blk.info
    }

    /// True if `net` is an internal (not simulated, computed) net of an ideal block.
    pub fn is_ideal_internal(&self, net: NetIdx) -> bool {
        self.blk.is_internal(net)
    }

    /// X origins of the given nets under the current values (see [`x_sources`]); internal nets
    /// of ideal blocks are taken at their exact (on-demand) values.
    pub fn x_sources(&self, nets: &[NetIdx]) -> Vec<XSource> {
        if self.blk.any_stale() {
            let mut v = self.values.clone();
            self.blk.fill_stale(&mut v);
            x_sources(&self.netlist, &v, nets)
        } else {
            x_sources(&self.netlist, &self.values, nets)
        }
    }

    /// Nets whose value changed during the most recent `step` (for VCD dumping), i.e. the nets
    /// whose value differs from the previous tick. Empty after a fast-forwarding [`TimingSim::run`]
    /// and after [`TimingSim::reset`].
    pub fn last_changed(&self) -> &[NetIdx] {
        &self.trk.changed
    }

    /// Number of gates scheduled for evaluation at the next step.
    pub fn pending_events(&self) -> usize {
        self.sched_len
    }

    /// Read a top-level pin (input or output) by name.
    pub fn read_pin(&self, pin: &str) -> Option<Value> {
        let nl = &self.netlist;
        let p = nl.top_inputs.iter().chain(&nl.top_outputs).find(|p| p.name == pin)?;
        Some(self.gather(&p.nets))
    }

    /// Read by hierarchical name: an exact flat net name (`"cpu/alu/c[3]"`, `"q"`), or a multi-bit
    /// net base name (`"cpu/alu/c"` gathers `c[0]..`), or a top-level pin name.
    pub fn read(&self, name: &str) -> Option<Value> {
        let nl = &self.netlist;
        if let Some(net) = nl.net_by_name(name) {
            return Some(Value::from_logic(self.get(net)));
        }
        let mut nets = Vec::new();
        let mut buf = String::with_capacity(name.len() + 6);
        loop {
            buf.clear();
            buf.push_str(name);
            buf.push('[');
            buf.push_str(&nets.len().to_string());
            buf.push(']');
            match nl.net_by_name(&buf) {
                Some(net) => nets.push(net),
                None => break,
            }
        }
        if !nets.is_empty() {
            return Some(self.gather(&nets));
        }
        self.read_pin(name)
    }

    /// Drain the diagnostics raised so far.
    pub fn take_diagnostics(&mut self) -> Vec<Diagnostic> {
        std::mem::take(&mut self.diags)
    }

    /// Diagnostics raised and not yet taken.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diags
    }

    /// Number of diagnostics dropped because more than [`MAX_PENDING_DIAGNOSTICS`] were pending
    /// (never reset; take diagnostics regularly to avoid drops).
    pub fn suppressed_diagnostics(&self) -> u64 {
        self.suppressed
    }

    /// Counters since the last reset.
    pub fn stats(&self) -> TimingStats {
        TimingStats { block_evals: self.blk.evals, lut_hits: self.blk.lut_hits, ..self.stats }
    }

    // -----------------------------------------------------------------------------------------
    // internals
    // -----------------------------------------------------------------------------------------

    fn gather(&self, nets: &[NetIdx]) -> Value {
        let bits: Vec<Logic> = nets.iter().map(|&n| self.get(n)).collect();
        Value::from_bits(&bits)
    }

    fn emit(&mut self, d: Diagnostic) {
        if self.diags.len() < MAX_PENDING_DIAGNOSTICS {
            self.diags.push(d);
        } else {
            self.suppressed += 1;
        }
    }

    fn next_epoch(&mut self) {
        // u64: cannot wrap in practice (one epoch per tick at most, plus resets).
        self.epoch += 1;
    }

    /// A source change at tick `t` from outside the normal source schedule (forcing).
    fn source_changed(&mut self, t: u64) {
        self.last_src_change = t;
        self.next_epoch();
        self.osc_reported = false;
    }

    /// A tick without gate-output changes ends the current activity episode.
    fn end_activity(&mut self) {
        if self.active_run > 0 {
            self.active_run = 0;
            self.osc_reported = false;
            self.next_epoch();
        }
    }

    /// Schedule gate `g` for the next step (no-op if already scheduled).
    fn push_sched(&mut self, g: u32) {
        if !self.scheduled[g as usize] {
            self.scheduled[g as usize] = true;
            self.sched[self.sched_len] = g;
            self.sched_len += 1;
        }
    }

    /// Schedule the gates reading `net` for the next tick, mark the resolvers reading it dirty and
    /// the ideal-block inputs it feeds changed.
    fn schedule_fanout(&mut self, net: NetIdx) {
        let n = net as usize;
        for i in self.c.fo_start[n]..self.c.fo_start[n + 1] {
            let f = self.c.fanout[i as usize];
            if f < BLK_BIT {
                self.push_sched(f);
            } else if f & RES_BIT != 0 {
                let r = (f & !RES_BIT) as usize;
                if !self.res_marked[r] {
                    self.res_marked[r] = true;
                    self.res_dirty.push(r as u32);
                }
            } else {
                let e = (f & !BLK_BIT) as usize;
                if !self.blk.entry_marked[e] {
                    self.blk.entry_marked[e] = true;
                    self.blk.changed.push(e as u32);
                }
            }
        }
    }

    /// Apply the ideal-block output changes due at tick `t`. Returns the number of nets changed.
    fn land_blocks(&mut self, t: u64) -> usize {
        let due = self.blk.take_due(t);
        let mut changed = 0;
        for &(net, v) in &due {
            changed += usize::from(self.set_net(net, v, t));
        }
        self.blk.restore_due(t, due);
        changed
    }

    /// Evaluate the ideal blocks whose inputs changed at tick `t`.
    fn process_blocks(&mut self, t: u64) {
        if self.blk.is_empty() {
            return;
        }
        let mut cont = std::mem::take(&mut self.blk_contention);
        self.blk.process(&mut self.values, t, &mut cont);
        self.blk_contention = cont;
        self.emit_block_contention();
    }

    /// Report contention found on block-internal resolvers.
    fn emit_block_contention(&mut self) {
        if self.blk_contention.is_empty() {
            return;
        }
        let t = self.tick;
        let mut cont = std::mem::take(&mut self.blk_contention);
        for (net, drv) in cont.drain(..) {
            let drv: Vec<String> = drv.iter().map(|v| v.to_string()).collect();
            self.emit(
                Diagnostic::warning(
                    DiagKind::Contention,
                    format!("contention at tick {t} (inside ideal block): drivers [{}]", drv.join(", ")),
                )
                .at(self.netlist.net_names[net as usize].clone()),
            );
        }
        self.blk_contention = cont;
    }

    /// Tick of the stuck-X check for the current options and sources (module docs).
    fn stuck_check_tick(&self) -> u64 {
        let k = u64::from(self.opts.stuck_x_periods);
        let mut period = 0u64;
        let mut release = 0u64;
        for &(_, s) in &self.c.timed {
            match s {
                Timed::Clock { period: p, .. } if s.changes_after(0) => period = period.max(p),
                Timed::PowerOnReset { ticks } => release = release.max(ticks),
                Timed::Clock { .. } => {}
            }
        }
        if k == 0 || period == 0 {
            return u64::MAX;
        }
        release.saturating_add(k.saturating_mul(period))
    }

    /// Raise `StuckX` for every top-level output pin that still has X bits.
    #[cold]
    fn check_stuck_x(&mut self, t: u64) {
        let k = self.opts.stuck_x_periods;
        let mut found = Vec::new();
        for p in &self.netlist.top_outputs {
            let bits: Vec<usize> = p
                .nets
                .iter()
                .enumerate()
                .filter(|&(_, &n)| self.values[n as usize] == Logic::X)
                .map(|(i, _)| i)
                .collect();
            if bits.is_empty() {
                continue;
            }
            let nets: Vec<NetIdx> = bits.iter().map(|&b| p.nets[b]).collect();
            let sources = self.x_sources(&nets);
            let mut origins: Vec<String> = sources
                .iter()
                .take(MAX_NAMES)
                .map(|s| {
                    let kind = match s.kind {
                        XSourceKind::Loop => "loop",
                        XSourceKind::Undriven => "undriven",
                        XSourceKind::Source => "source",
                        XSourceKind::Contention => "contention",
                        XSourceKind::Gate => "gate",
                    };
                    format!("{kind} [{}]", self.name_list(&s.nets))
                })
                .collect();
            if sources.len() > MAX_NAMES {
                origins.push("...".into());
            }
            let bit_list: Vec<String> = bits.iter().map(|b| b.to_string()).collect();
            found.push(
                Diagnostic::warning(
                    DiagKind::StuckX,
                    format!(
                        "output '{}' bits [{}] still X after {k} clock periods (tick {t}); X origins: {}",
                        p.name,
                        bit_list.join(", "),
                        origins.join("; ")
                    ),
                )
                .at(p.name.clone()),
            );
        }
        for d in found {
            self.emit(d);
        }
    }

    /// Set a net at tick `t` from a delay-0 element (source / resolver). Returns true if changed.
    fn set_net(&mut self, net: NetIdx, v: Logic, t: u64) -> bool {
        let old = self.values[net as usize];
        if old == v {
            return false;
        }
        self.values[net as usize] = v;
        self.trk.record(net, old, v, t);
        self.schedule_fanout(net);
        true
    }

    /// Give every source its tick-`t` value. Returns true if any source net changed.
    fn apply_sources(&mut self, t: u64) -> bool {
        let mut changed = false;
        for i in 0..self.c.timed.len() {
            let (net, s) = self.c.timed[i];
            changed |= self.set_net(net, s.value_at(t), t);
        }
        if !self.forced_sources.is_empty() {
            let forced = std::mem::take(&mut self.forced_sources);
            for &net in &forced {
                let v = self.c.source_value(net, t, &self.input_values);
                changed |= self.set_net(net, v, t);
            }
            self.forced_sources = forced;
            self.forced_sources.clear();
        }
        if self.inputs_pending {
            self.inputs_pending = false;
            for pin in 0..self.c.input_nets.len() {
                for bit in 0..self.c.input_nets[pin].len() {
                    let net = self.c.input_nets[pin][bit];
                    if net != NONE {
                        let v = self.input_values[pin].get(bit as u32);
                        changed |= self.set_net(net, v, t);
                    }
                }
            }
        }
        changed
    }

    /// Recompute dirty resolvers until none is dirty.
    fn process_resolvers(&mut self, t: u64) {
        let mut budget = 4 * self.c.r_out.len() + 16;
        while let Some(r) = self.res_dirty.pop() {
            let ri = r as usize;
            self.res_marked[ri] = false;
            if self.res_hold[ri] {
                continue;
            }
            if budget == 0 {
                // Structurally impossible for flattened netlists (resolver drivers are private
                // nets); kept as a guard against malformed input.
                for &r in &self.res_dirty {
                    self.res_marked[r as usize] = false;
                }
                self.res_dirty.clear();
                self.emit(
                    Diagnostic::warning(DiagKind::IterationLimit, "resolver evaluation did not converge")
                        .at(self.netlist.net_names[self.c.r_out[ri] as usize].clone()),
                );
                return;
            }
            budget -= 1;
            let out = self.c.r_out[ri];
            let drivers = &self.c.r_drivers[self.c.r_start[ri] as usize..self.c.r_start[ri + 1] as usize];
            self.res_buf.clear();
            self.res_buf.extend(drivers.iter().map(|&d| self.values[d as usize]));
            let res = eval::resolve(&self.res_buf, self.c.r_term[ri], self.values[out as usize]);
            // Tick 0 is the power-on state: X there means "not yet evaluated", not a driver
            // fight, so it never reports contention.
            let contention = res.contention && t > 0;
            if contention && !self.contending[ri] {
                let drv: Vec<String> = self.res_buf.iter().map(|v| v.to_string()).collect();
                self.emit(
                    Diagnostic::warning(
                        DiagKind::Contention,
                        format!("contention at tick {t}: drivers [{}]", drv.join(", ")),
                    )
                    .at(self.netlist.net_names[out as usize].clone()),
                );
            }
            self.contending[ri] = contention;
            self.set_net(out, res.value, t);
        }
    }

    /// True if net `i` is in the collapse set or did not change during the window `(t-limit, t]`.
    #[inline]
    fn p2_input_ok(&self, i: NetIdx, t: u64, limit: u64) -> bool {
        self.p2_mark[i as usize] || self.trk.last[i as usize] + limit <= t
    }

    /// True if every input of the driver of `net` is in the collapse set or quiet (see
    /// [`TimingSim::p2_input_ok`]). Nets without a gate/resolver driver are never sustained.
    fn p2_sustained(&self, net: NetIdx, t: u64, limit: u64) -> bool {
        if let Some((cone, lat)) = self.blk.output_cone_lat(net) {
            // The output replays its inputs `lat` ticks later: a cone input is quiet only if it
            // did not change during the window shifted back by `lat - 1` ticks (for a chain of
            // `lat` gates this is exactly "the last gate's input is quiet"; `lat = 1` is the
            // plain gate rule).
            let quiet = limit + u64::from(lat.saturating_sub(1));
            return cone.iter().all(|&i| self.p2_mark[i as usize] || self.trk.last[i as usize] + quiet <= t);
        }
        let g = self.c.driver_gate[net as usize];
        if g != NONE {
            let rec = self.c.gates[g as usize];
            return if rec.tab != 0 {
                self.p2_input_ok(rec.a, t, limit) && self.p2_input_ok(rec.b, t, limit)
            } else {
                let w = self.c.wide[rec.a as usize];
                self.c.wide_inputs[w.start as usize..w.end as usize].iter().all(|&i| self.p2_input_ok(i, t, limit))
            };
        }
        let r = self.c.driver_res[net as usize];
        if r != NONE {
            let ri = r as usize;
            let drivers = &self.c.r_drivers[self.c.r_start[ri] as usize..self.c.r_start[ri + 1] as usize];
            return drivers.iter().all(|&d| self.p2_input_ok(d, t, limit));
        }
        false
    }

    /// Apply the period-2 collapse rule to this tick's candidates (module docs, "Collapse
    /// details").
    #[cold]
    fn collapse(&mut self, t: u64) {
        let limit = u64::from(self.trk.limit);
        if limit == 0 || t - self.last_src_change < limit {
            self.trk.candidates.clear();
            return;
        }
        let cands = std::mem::take(&mut self.trk.candidates);
        // Greatest fixed point: start from all candidates, drop any whose driver has an input
        // that is neither in the set nor quiet, and re-check the readers of every dropped net.
        for &net in &cands {
            self.p2_mark[net as usize] = true;
        }
        let mut work = cands.clone();
        while let Some(net) = work.pop() {
            if !self.p2_mark[net as usize] || self.p2_sustained(net, t, limit) {
                continue;
            }
            self.p2_mark[net as usize] = false;
            let mut feeds_block = false;
            for &f in self.c.readers(net) {
                let out = if f < BLK_BIT {
                    self.c.gates[f as usize].out
                } else if f & RES_BIT != 0 {
                    self.c.r_out[(f & !RES_BIT) as usize]
                } else {
                    feeds_block = true;
                    continue;
                };
                if self.p2_mark[out as usize] {
                    work.push(out);
                }
            }
            if feeds_block {
                // Some block output's cone contains `net`: re-check every remaining candidate.
                work.extend(cands.iter().copied().filter(|&c| self.p2_mark[c as usize]));
            }
        }
        let mut report: Vec<NetIdx> = Vec::new();
        let mut held: Vec<u32> = Vec::new();
        for &net in &cands {
            let n = net as usize;
            if !std::mem::replace(&mut self.p2_mark[n], false) || self.values[n] == Logic::X {
                // Not sustained, or already X (alternating with X, or an X that just arrived):
                // nothing to force.
                continue;
            }
            self.values[n] = Logic::X;
            self.trk.needs_fix = true;
            self.schedule_fanout(net);
            // With the closure above, a sustained net always has an alternating driver input
            // (which changed this tick, so the driver is already scheduled), and its resolver's
            // alternating drivers are collapsed too. Re-scheduling the driver and holding the
            // resolver therefore never change the outcome today; they keep the documented rule
            // true by construction.
            if self.blk.output_cone(net).is_some() {
                self.blk.reschedule_output(net);
            } else {
                let g = self.c.driver_gate[n];
                if g != NONE {
                    self.push_sched(g);
                }
            }
            let r = self.c.driver_res[n];
            if r != NONE {
                self.res_hold[r as usize] = true;
                held.push(r);
            }
            if self.reported_epoch[n] != self.epoch {
                self.reported_epoch[n] = self.epoch;
                report.push(net);
            }
        }
        // Zero-delay resolvers downstream of collapsed nets see the X now.
        self.process_resolvers(t);
        for r in held {
            self.res_hold[r as usize] = false;
        }
        let mut cands = cands;
        cands.clear();
        self.trk.candidates = cands;
        if !report.is_empty() {
            let names = self.name_list(&report);
            self.emit(
                Diagnostic::warning(
                    DiagKind::Metastable,
                    format!(
                        "period-2 oscillation for {limit} ticks without source change; forced to X at tick {t}: {names}"
                    ),
                )
                .at(self.netlist.net_names[report[0] as usize].clone()),
            );
        }
    }

    #[cold]
    fn report_oscillation(&mut self, t: u64) {
        let names = self.name_list(&self.trk.changed.clone());
        let loc = self.trk.changed.first().map(|&n| self.netlist.net_names[n as usize].clone());
        let mut d = Diagnostic::warning(
            DiagKind::Oscillating,
            format!(
                "gate activity for {} consecutive ticks without source change (tick {t}); oscillating nets include: {names}",
                self.active_run
            ),
        );
        if let Some(l) = loc {
            d = d.at(l);
        }
        self.emit(d);
    }

    fn name_list(&self, nets: &[NetIdx]) -> String {
        let mut s: Vec<&str> =
            nets.iter().take(MAX_NAMES).map(|&n| self.netlist.net_names[n as usize].as_str()).collect();
        if nets.len() > MAX_NAMES {
            s.push("...");
        }
        s.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use panda_netlist::flat::{FlatGate, FlatNetlist};

    const ALL: [Logic; 4] = [Logic::Zero, Logic::One, Logic::X, Logic::Z];

    /// A netlist with one gate of `op` over `k` input nets (nets 0..k, undriven, then forced).
    fn one_gate(op: FlatOp, k: usize) -> FlatNetlist {
        FlatNetlist {
            num_nets: k as u32 + 1,
            gates: vec![FlatGate { op, inputs: (0..k as u32).collect(), output: k as u32, owner: 0, group: 0, bit: 0 }],
            net_names: (0..=k).map(|i| format!("n{i}")).collect(),
            ..Default::default()
        }
    }

    /// The compiled gate evaluation equals `eval_gate` for every op, arity 1..=4 and every input.
    #[test]
    fn compiled_gates_match_reference() {
        for op in ALL_OPS {
            let arities: &[usize] = match op {
                FlatOp::Not | FlatOp::Buf => &[1],
                FlatOp::Tri => &[2],
                _ => &[1, 2, 3, 4],
            };
            for &k in arities {
                let nl = one_gate(op, k);
                let c = Compiled::new(&nl, &Tables::new(), &ideal::Blocks::default());
                let total = 4usize.pow(k as u32);
                for combo in 0..total {
                    let ins: Vec<Logic> = (0..k).map(|i| ALL[(combo >> (2 * i)) & 3]).collect();
                    let mut values = ins.clone();
                    values.push(Logic::X);
                    let got = c.eval(c.gates[0], &values);
                    assert_eq!(got, eval::eval_gate(op, &ins), "{op:?} {ins:?}");
                }
            }
        }
    }

    #[test]
    fn timed_sources() {
        let clk = Timed::Clock { period: 6, high: 2, phase: 1 };
        let wave: String = (0..13).map(|t| clk.value_at(t).to_char()).collect();
        assert_eq!(wave, "0110000110000");
        let por = Timed::PowerOnReset { ticks: 3 };
        let wave: String = (0..5).map(|t| por.value_at(t).to_char()).collect();
        assert_eq!(wave, "11100");
        assert!(por.changes_after(2));
        assert!(!por.changes_after(3));
    }

    #[test]
    fn timed_sources_near_u64_max() {
        // Regression: `t + period` overflowed for ticks near u64::MAX.
        let clk = Timed::Clock { period: 6, high: 2, phase: 1 };
        for t in [u64::MAX - 20, u64::MAX - 1, u64::MAX] {
            let reference = Logic::from_bool(((t % 6) + 6 - 1) % 6 < 2);
            assert_eq!(clk.value_at(t), reference, "tick {t}");
        }
        let wave: String = (u64::MAX - 11..=u64::MAX - 6).map(|t| clk.value_at(t).to_char()).collect();
        // u64::MAX % 6 == 3, so u64::MAX - 11 ≡ 4 (mod 6): positions 4,5,0,1,2,3; high at 1 and 2 (phase 1).
        assert_eq!(wave, "000110");
    }

    #[test]
    fn constant_clocks_never_change() {
        assert!(!Timed::Clock { period: 4, high: 0, phase: 0 }.changes_after(0));
        assert!(!Timed::Clock { period: 4, high: 4, phase: 0 }.changes_after(0));
        assert!(!Timed::Clock { period: 1, high: 1, phase: 0 }.changes_after(0));
        assert!(Timed::Clock { period: 2, high: 1, phase: 0 }.changes_after(u64::MAX - 1));
    }

    #[test]
    fn undriven_nets_read_z() {
        let nl = one_gate(FlatOp::Not, 1);
        let mut sim = TimingSim::new(Arc::new(nl), TimingOptions::default());
        assert_eq!(sim.get(0), Logic::Z);
        assert_eq!(sim.get(1), Logic::X);
        sim.step();
        assert_eq!(sim.get(1), Logic::X);
        sim.force_net(0, Logic::One);
        sim.step();
        assert_eq!(sim.get(1), Logic::Zero);
        assert_eq!(sim.last_changed(), &[1]);
        assert!(sim.settle(10));
    }
}
