//! FastIR — the compiled program executed by [`crate::FastSim`].
//!
//! Design constraints (docs/M4-CONTRACT.md "Bytecode"):
//!
//! * **Flat state.** All runtime values live in one byte-per-net array (`Logic` codes, index =
//!   [`NetIdx`]), the same layout as `FastRef::values()`.
//! * **Fixed-size instructions.** [`Op`] is 16 bytes, holds only plain integers (net indices, pool
//!   offsets, table selectors) — no pointers, no closures — so a JIT (M5) can translate each
//!   partition to native code one op at a time.
//! * **Partitions are contiguous op ranges** ([`Partition`]) in topological order: evaluating the
//!   dirty partitions in index order evaluates every gate after all its combinational inputs.
//! * **Change propagation is static.** For every net, [`Program::net_parts`] lists the partitions
//!   (other than the driver's own) that read it and [`Program::net_kernels`] the kernels that read
//!   it. An op whose output has any such reader carries `notify = true`.
//!
//! Kernel transitions are separate [`KernelOp`]s executed only at round boundaries (never inside a
//! partition sweep), which keeps the round structure of SPEC §6.2 exact.
//!
//! The enums leave room for the next compiler stages: LUT ops (SPEC §7.1), word ops for slice
//! packing (SPEC §8.7) and phase-specialized kernel ops (`KOpaque`, `KTransparent`, SPEC §8.5).

use std::sync::Arc;

use panda_compile::KernelTable;
use panda_netlist::flat::NetIdx;
use panda_netlist::{Logic, Termination};

/// Sentinel for "none" in `u32` fields.
pub const NONE: u32 = u32::MAX;

/// [`Program::lazy_owner`] of the output of a gate absorbed into a fused op.
pub const LAZY_FUSED: u32 = NONE - 2;

/// [`KClass::vact`] entry: the outputs do not change.
pub const VACT_OPAQUE: u32 = NONE - 1;

/// Shift of the input position in [`Program::net_kin`] entries.
pub const KIN_POS_SHIFT: u32 = 24;
/// Mask of the kernel id in [`Program::net_kin`] entries.
pub const KIN_KERNEL_MASK: u32 = (1 << KIN_POS_SHIFT) - 1;
/// Mask of the (shifted) input position in [`Program::net_kin`] entries.
pub const KIN_POS_MASK: u32 = 0x3F;
/// [`Program::net_kin`] entry flag: track the kernel's flipped input bits per interval (kernels
/// with a [`KFast::check_order`] phase).
pub const KIN_TRACK: u32 = 1 << 31;
/// [`Program::net_kin`] entry flag: the net is not a function of the clock tuple (a data input
/// of the kernel); its changes invalidate the transparent-unchanged skip.
pub const KIN_DATA: u32 = 1 << 30;

/// Selector of a 2-input 2-bit-code table in [`Program::tables`]
/// (see `panda_netlist::eval::lookup2`).
pub mod table {
    /// `and2`.
    pub const AND: u8 = 0;
    /// `or2`.
    pub const OR: u8 = 1;
    /// `xor2`.
    pub const XOR: u8 = 2;
    /// `not1(and2)`.
    pub const NAND: u8 = 3;
    /// `not1(or2)`.
    pub const NOR: u8 = 4;
    /// `not1(xor2)`.
    pub const XNOR: u8 = 5;
    /// `not1(a)` (second operand ignored).
    pub const NOT: u8 = 6;
    /// `a.normalize()` (second operand ignored).
    pub const BUF: u8 = 7;
    /// `tri(d = a, en = b)`.
    pub const TRI: u8 = 8;
    /// Number of tables.
    pub const COUNT: usize = 9;
}

/// The packed 2-bit-code tables indexed by the [`table`] selectors, computed from the reference
/// semantics in `panda_netlist::eval`.
pub fn gate_tables() -> [u32; table::COUNT] {
    use panda_netlist::eval::{not1, table2, tables, tri};
    let t = tables(); // And, Or, Xor, Nand, Nor, Xnor
    [
        t[0],
        t[1],
        t[2],
        t[3],
        t[4],
        t[5],
        table2(|a, _| not1(a)),
        table2(|a, _| a.normalize()),
        table2(tri),
    ]
}

/// One combinational instruction (16 bytes). `notify`: the output net has readers outside the
/// op's partition (later partitions or kernels), so a change must be propagated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// `out = lookup2(tables[t], v[a], v[b])` — every 1- and 2-input gate (Not/Buf use `b = a`).
    Gate2 {
        /// Table selector ([`table`]).
        t: u8,
        /// Propagate changes of `out`.
        notify: bool,
        /// Output net.
        out: u32,
        /// First operand net.
        a: u32,
        /// Second operand net.
        b: u32,
    },
    /// N-input AND/OR/XOR family: `acc = identity; for i in args: acc = lookup2(tables[fold],
    /// acc, v[i])`, then `out = invert ? not1(acc) : acc`.
    GateN {
        /// Fold table ([`table::AND`], [`table::OR`] or [`table::XOR`]).
        fold: u8,
        /// Invert the folded result (NAND/NOR/XNOR).
        invert: bool,
        /// Propagate changes of `out`.
        notify: bool,
        /// Output net.
        out: u32,
        /// Offset of the operand nets in [`Program::args`].
        args: u32,
        /// Number of operands.
        n: u32,
    },
    /// Resolver `res` ([`Program::resolvers`]): `out = resolve(drivers, termination, out)`;
    /// records contention for the diagnostics.
    Resolve {
        /// Propagate changes of `out`.
        notify: bool,
        /// Output net.
        out: u32,
        /// Index into [`Program::resolvers`].
        res: u32,
    },
    /// LUT `lut` ([`Program::luts`], SPEC §7.1): if every support net is binary, the outputs are
    /// read from the binary-indexed table and the cone's internal nets become stale (materialized
    /// on demand); otherwise the cone's gate code ([`LutDesc::cold`]) runs, which is exact 4-state
    /// semantics.
    Lut {
        /// Index into [`Program::luts`].
        lut: u32,
    },
}

const _: () = assert!(std::mem::size_of::<Op>() == 16);

impl Op {
    /// The net written by this op ([`NONE`] for [`Op::Lut`], which writes
    /// [`Program::lut_outputs`]).
    pub fn output(&self) -> NetIdx {
        match *self {
            Op::Gate2 { out, .. } | Op::GateN { out, .. } | Op::Resolve { out, .. } => out,
            Op::Lut { .. } => NONE,
        }
    }

    /// True if a change of the output must be propagated (LUT outputs are always compared).
    pub fn notifies(&self) -> bool {
        match *self {
            Op::Gate2 { notify, .. } | Op::GateN { notify, .. } | Op::Resolve { notify, .. } => {
                notify
            }
            Op::Lut { .. } => true,
        }
    }

    pub(crate) fn set_notify(&mut self, v: bool) {
        match self {
            Op::Gate2 { notify, .. } | Op::GateN { notify, .. } | Op::Resolve { notify, .. } => {
                *notify = v
            }
            Op::Lut { .. } => {}
        }
    }
}

/// The interpreter's encoding of the ops of a partition ([`crate::fuse`]): 16-byte slots.
///
/// * A 1/2-input gate is `out = (table >> 2 * (v[a] | v[b] << 2)) & 3` with the gate's 32-bit
///   2-bit-code table embedded, so the common case needs no dispatch and no table selector.
/// * A fused op ([`HOp::SPECIAL`] | [`HOp::FUSED`]) takes two slots: inputs `a, b` and the second
///   slot's `a, b` (and, if [`HOp::WIDE`], the second slot's `table, out`); `table` is the offset
///   of its 4-state truth table in [`Program::ftables`].
/// * Any other op ([`HOp::SPECIAL`] alone) is executed from [`Program::ops`]`[a]`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HOp {
    /// The gate's packed table (see [`gate_tables`]); unused for special ops.
    pub table: u32,
    /// Output net | [`HOp::NOTIFY`] | [`HOp::SPECIAL`].
    pub out: u32,
    /// First operand net.
    pub a: u32,
    /// Second operand net.
    pub b: u32,
}

impl HOp {
    /// Flag in [`HOp::out`]: propagate changes of the output.
    pub const NOTIFY: u32 = 1 << 31;
    /// Flag in [`HOp::out`]: not a 1/2-input gate (a fused op, or [`Program::ops`]`[a]`).
    pub const SPECIAL: u32 = 1 << 30;
    /// Flag in [`HOp::out`] (with [`HOp::SPECIAL`]): a fused op (two slots).
    pub const FUSED: u32 = 1 << 29;
    /// Flag in [`HOp::out`] (with [`HOp::FUSED`]): a wide fused op (5–6 inputs: the second
    /// slot's `table` and `out` are inputs 4 and 5; 4096-entry table).
    pub const WIDE: u32 = 1 << 28;
    /// Mask of the net index in [`HOp::out`].
    pub const NET: u32 = Self::WIDE - 1;
}

const _: () = assert!(std::mem::size_of::<HOp>() == 16);

/// A LUT (SPEC §7.1): a multi-output combinational cone of 2-state gates replaced by one table
/// indexed by its binary support.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LutDesc {
    /// Offset of the support nets in [`Program::lut_nets`] (bit `i` of the table index = support
    /// net `i`).
    pub support: u32,
    /// Number of support nets (≤ 16).
    pub num_support: u32,
    /// Offset of the output nets in [`Program::lut_nets`] (bit `j` of an entry = output `j`).
    pub outputs: u32,
    /// Number of outputs (≤ 32).
    pub num_outputs: u32,
    /// Offset of the `2^num_support` entries in [`Program::lut_entries`].
    pub table: u32,
    /// The cone's gate code in [`Program::cold_ops`] (topological order): run when some support
    /// net is `X`/`Z`, and to materialize the internal nets.
    pub cold: Partition,
}

/// A combinational resolver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolverDesc {
    /// Offset of the driver nets in [`Program::args`].
    pub args: u32,
    /// Number of drivers.
    pub n: u32,
    /// Termination.
    pub termination: Termination,
    /// Index into `FlatNetlist::resolvers`.
    pub flat_index: u32,
}

/// A partition: the contiguous op range `ops[start..end]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Partition {
    /// First op.
    pub start: u32,
    /// One past the last op.
    pub end: u32,
}

impl Partition {
    /// Number of ops.
    pub fn len(&self) -> usize {
        (self.end - self.start) as usize
    }

    /// True if the partition has no ops.
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// How a kernel transitions (chosen at compile time).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelOpKind {
    /// General table step: `(state id, binary input vector) → next state id` from
    /// [`Program::ktables`]`[table]`, falling back to the on-the-fly simulator whenever the state
    /// or inputs are not binary or the entry is non-binary (exactly as FastRef).
    KTable {
        /// Index into [`Program::ktables`].
        table: u32,
    },
    /// No table: always the on-the-fly simulator (Sim/Keeper kernels, oversized tables).
    KSim,
}

/// Phase class of a kernel under one clock tuple (SPEC §8.5), proven from its table for every
/// transition the kernel can make in that phase (see [`crate::kspec`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PhaseClass {
    /// `KOpaque`: no transition in this phase changes an output; the new state is a function of
    /// the outputs and the inputs, so the transition is a no-op on the observable nets and the
    /// state is recomputed lazily.
    Opaque,
    /// `KTransparent`: the new state depends only on the inputs: the outputs are a table lookup
    /// over the input vector ([`Program::touts`]).
    Transparent,
    /// Neither: general table step (still with lazily reconstructed states where possible).
    General,
}

/// Per-kernel fast path of one phase (a kernel in [`Program::tuple_fast`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KFast {
    /// Offset of the output table in [`Program::touts`] (`KTransparent`), or [`NONE`]
    /// (`KOpaque`: the outputs do not change).
    pub tout: u32,
    /// Some transition of this phase may be order-sensitive (or the characterization could not
    /// tell): when two or more inputs change, run FastRef's exact order check (for the
    /// `DelaySensitive` diagnostic only; the result is the table's either way).
    pub check_order: bool,
    /// The kernel's pure input bits in this tuple (ORed into the tracked input index).
    pub pure: u32,
}

/// Shared per-(kernel table, output positions) data for the specialized kernel ops.
#[derive(Clone, Debug)]
pub struct KClass {
    /// Index into [`Program::ktables`].
    pub table: u32,
    /// Number of inputs `k`.
    pub num_inputs: u32,
    /// Number of outputs (state nets with readers) `o`.
    pub num_outs: u32,
    /// Per state id: the output bits (bit `j` = output `j` is `1`; states are binary).
    pub out_bits: Vec<u32>,
    /// State reconstruction: `recon[(out_bits << k) | v]` = the unique stable state under input
    /// vector `v` with those outputs, or [`NONE`] (none or ambiguous).
    pub recon: Vec<u32>,
    /// Per input vector `v`: the specialized action of a transition to `v` from a vector that
    /// differs in exactly one input ([`crate::kspec`]): [`VACT_OPAQUE`] (outputs unchanged),
    /// output bits (a function of `v`), or [`NONE`] (general path).
    pub vact: Vec<u32>,
}

impl KClass {
    /// The unique stable state under `v` with outputs `o`, or [`NONE`].
    #[inline]
    pub fn recon(&self, o: u32, v: u32) -> u32 {
        self.recon[((o as usize) << self.num_inputs) | v as usize]
    }
}

/// One kernel's transition op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelOp {
    /// Kernel id (index into `KernelSet::kernels`).
    pub kernel: u32,
    /// Transition strategy.
    pub kind: KernelOpKind,
    /// Offset of the input nets in [`Program::kernel_nets`] (bit `i` of the input vector =
    /// input `i`).
    pub inputs: u32,
    /// Number of inputs.
    pub num_inputs: u32,
    /// Offset of the state nets in [`Program::kernel_nets`].
    pub nets: u32,
    /// Number of state nets.
    pub num_nets: u32,
    /// Offset in [`Program::kernel_outs`] of the indices (into the state nets) of the state nets
    /// that have readers outside the kernel (only their changes need propagating).
    pub outs: u32,
    /// Number of such state nets.
    pub num_outs: u32,
    /// Offset of this kernel's last-transition inputs in the runtime input record.
    pub last_in: u32,
}

/// A kernel table in executor-friendly form (shared by structurally identical kernels).
#[derive(Clone, Debug)]
pub struct KTableData {
    /// Number of inputs `k`.
    pub num_inputs: u32,
    /// Number of state nets `m`.
    pub num_nets: u32,
    /// `entries[(state << k) | input]`: bits 0..31 = next state id, or [`KTableData::NON_BINARY`];
    /// bit 31 ([`KTableData::ORDER_FLAG`]) = the entry's `Transition::order_sensitive` flag.
    pub entries: Vec<u32>,
    /// State values, `m` per state, flattened.
    pub states: Vec<Logic>,
    /// The characterization only checked one previous input vector per state (see
    /// `KernelTable::order_check_partial`).
    pub order_check_partial: bool,
    /// The characterized table (state lookup, exact order sensitivity).
    pub table: Arc<KernelTable>,
}

impl KTableData {
    /// Entry bit: `Transition::order_sensitive`.
    pub const ORDER_FLAG: u32 = 1 << 31;
    /// Entry value (low 31 bits): non-binary result (recompute on the fly).
    pub const NON_BINARY: u32 = (1 << 31) - 1;

    /// Build from a characterized table.
    pub fn new(table: Arc<KernelTable>, num_nets: usize) -> KTableData {
        assert!(
            table.states.len() < Self::NON_BINARY as usize,
            "too many kernel states"
        );
        let entries = table
            .transitions
            .iter()
            .map(|t| {
                t.next.unwrap_or(Self::NON_BINARY)
                    | if t.order_sensitive {
                        Self::ORDER_FLAG
                    } else {
                        0
                    }
            })
            .collect();
        let mut states = Vec::with_capacity(table.states.len() * num_nets);
        for s in &table.states {
            debug_assert_eq!(s.len(), num_nets);
            states.extend_from_slice(s);
        }
        KTableData {
            num_inputs: table.num_inputs as u32,
            num_nets: num_nets as u32,
            entries,
            states,
            order_check_partial: table.order_check_partial,
            table,
        }
    }

    /// The packed entry for `(state, input vector)`.
    #[inline]
    pub fn entry(&self, state: u32, input: u32) -> u32 {
        self.entries[((state as usize) << self.num_inputs) | input as usize]
    }

    /// Values of state `id`.
    #[inline]
    pub fn state(&self, id: u32) -> &[Logic] {
        let m = self.num_nets as usize;
        &self.states[id as usize * m..(id as usize + 1) * m]
    }
}

/// Per-net lists of `(bitset word, mask)` pairs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MaskCsr {
    /// Row starts (`num_nets + 1` entries, or empty when there are no rows).
    pub start: Vec<u32>,
    /// Concatenated rows.
    pub items: Vec<(u32, u64)>,
}

impl MaskCsr {
    /// Row of `net` (empty if the table has no rows).
    #[inline]
    pub fn row(&self, net: u32) -> &[(u32, u64)] {
        match self.start.get(net as usize + 1) {
            Some(&e) => &self.items[self.start[net as usize] as usize..e as usize],
            None => &[],
        }
    }
}

/// Compressed per-net adjacency: row `net` is `items[start[net]..start[net + 1]]`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NetCsr {
    /// Row starts (`num_nets + 1` entries).
    pub start: Vec<u32>,
    /// Concatenated rows.
    pub items: Vec<u32>,
}

impl NetCsr {
    /// Build from `(net, item)` pairs (rows sorted ascending and deduplicated).
    pub fn from_pairs(n: usize, pairs: &mut [(u32, u32)]) -> NetCsr {
        pairs.sort_unstable();
        let mut start = vec![0u32; n + 1];
        let mut items = Vec::with_capacity(pairs.len());
        let mut prev = None;
        for &(net, v) in pairs.iter() {
            if prev == Some((net, v)) {
                continue;
            }
            prev = Some((net, v));
            items.push(v);
            start[net as usize + 1] += 1;
        }
        for i in 0..n {
            start[i + 1] += start[i];
        }
        NetCsr { start, items }
    }

    /// Row of `net`.
    #[inline]
    pub fn row(&self, net: u32) -> &[u32] {
        &self.items[self.start[net as usize] as usize..self.start[net as usize + 1] as usize]
    }
}

/// The compiled program.
#[derive(Clone, Debug)]
pub struct Program {
    /// Number of nets (length of the state array).
    pub num_nets: u32,
    /// 2-input 2-bit-code tables ([`table`] selectors).
    pub tables: [u32; table::COUNT],
    /// Combinational ops, partition after partition.
    pub ops: Vec<Op>,
    /// The interpreter's encoding of [`Program::ops`] ([`crate::fuse::lower`]).
    pub hops: Vec<HOp>,
    /// Per [`Program::hops`] slot: the change-propagation record of its output, or [`NONE`].
    pub hrec: Vec<u32>,
    /// Per partition: its range of [`Program::hops`].
    pub hpart: Vec<Partition>,
    /// Truth tables of the fused ops ([`crate::fuse::FTABLE_WORDS`] words each).
    pub ftables: Vec<u64>,
    /// The gates absorbed into fused ops (topological order): their output nets are stale in the
    /// hot path and recomputed on demand.
    pub fcold: Vec<Op>,
    /// Operand pool of [`Op::GateN`] and of resolvers.
    pub args: Vec<u32>,
    /// Combinational resolvers ([`Op::Resolve`]).
    pub resolvers: Vec<ResolverDesc>,
    /// Partitions in topological order.
    pub partitions: Vec<Partition>,
    /// Per net: partitions to mark dirty when it changes (its driver's partition excluded).
    pub net_parts: NetCsr,
    /// Per net: kernels to mark pending when it changes.
    pub net_kernels: NetCsr,
    /// Per net: `(word, mask)` pairs of kernels to mark pending when it changes, for readers whose
    /// input bit is not tracked in the interpreter's input index (pure inputs: their bits come
    /// from the clock tuple, [`KFast::pure`]). Every other reader is in [`Program::net_kin`].
    pub net_kpend: MaskCsr,
    /// Per kernel: the input positions tracked in the interpreter's input index.
    pub kernel_kin_mask: Vec<u32>,
    /// Per net: offset of its change-propagation record in [`Program::notify_rec`], or [`NONE`]
    /// for nets without readers outside their driver's partition.
    pub notify_off: Vec<u32>,
    /// Change-propagation records (32-bit words), derived from [`Program::net_parts`],
    /// [`Program::net_kin`], [`Program::net_kpend`] and the kernel classes (one record per net
    /// with readers; 64-bit masks as `lo, hi` word pairs):
    ///
    /// ```text
    /// [np | nk << 8 | nkin << 16, net, log slot (NONE: no log),
    ///  np x (partition word, mask lo, mask hi),
    ///  nk x (kernel word, pending mask lo, hi, data-input reader mask lo, hi),
    ///  nkin x kernel input-index entry (see net_kin)]
    /// ```
    ///
    /// When `np ≥ 0xFF`, `nk > 0xFF` or `nkin > 0xFFFF`, the first word is the escape `0xFF`
    /// and the counts follow the log slot as three full words (`np, nk, nkin`).
    ///
    /// A change ORs the partition masks into the dirty set, the pending masks into the pending
    /// set and the data masks into the data-dirty set, clears the pending masks from the fast-ok
    /// set if the new value is not binary, logs the boundary value, and updates the kernels'
    /// input indices. The interpreter's change entries name records directly
    /// (`offset << 2 | old value code`), from [`Program::op_rec`], [`Program::cold_rec`],
    /// [`Program::lut_rec`] and [`Program::kernel_out_rec`].
    pub notify_rec: Vec<u32>,
    /// Per op of [`Program::ops`]: the record offset of its output ([`NONE`] if it has no
    /// readers to notify).
    pub op_rec: Vec<u32>,
    /// Per op of [`Program::cold_ops`]: the record offset of its output, or [`NONE`].
    pub cold_rec: Vec<u32>,
    /// Per entry of [`Program::lut_nets`]: the record offset of the net, or [`NONE`].
    pub lut_rec: Vec<u32>,
    /// Per entry of [`Program::kernel_out_nets`]: the record offset of the net, or [`NONE`].
    pub kernel_out_rec: Vec<u32>,
    /// Per kernel with exactly one output net: `(net, record offset)`; `(NONE, NONE)` otherwise.
    pub kout1: Vec<(u32, u32)>,
    /// Kernels with flip tracking (some [`KIN_TRACK`] reader entry; bitset): the interpreter
    /// snapshots their input index at every round boundary, so the bits flipped since are
    /// `kin ^ kin_prev`.
    pub tracked: Vec<u64>,
    /// The kernels of [`Program::tracked`], ascending.
    pub tracked_list: Vec<u32>,
    /// Per net: its slot in the interpreter's boundary log (nets read by kernels), or [`NONE`].
    pub net_log: Vec<u32>,
    /// Number of boundary-log slots.
    pub num_log: u32,
    /// Per net: the kernels reading it with the input position:
    /// `kernel | min(position, 32) << KIN_POS_SHIFT` (the interpreter keeps every kernel's input
    /// index up to date from it), plus the flags [`KIN_TRACK`] and [`KIN_DATA`].
    pub net_kin: NetCsr,
    /// One transition op per kernel (index = kernel id).
    pub kernel_ops: Vec<KernelOp>,
    /// Input and state net pool of [`KernelOp`].
    pub kernel_nets: Vec<u32>,
    /// State-net index pool of [`KernelOp::outs`].
    pub kernel_outs: Vec<u32>,
    /// The nets of [`Program::kernel_outs`] (same offsets).
    pub kernel_out_nets: Vec<u32>,
    /// Kernel tables (deduplicated by identity of the characterized table).
    pub ktables: Vec<KTableData>,
    /// Total kernel inputs (size of the runtime last-transition input record).
    pub kernel_inputs_total: u32,
    /// LUTs ([`Op::Lut`]).
    pub luts: Vec<LutDesc>,
    /// Support and output net pool of [`LutDesc`].
    pub lut_nets: Vec<u32>,
    /// LUT entries (bit `j` = output `j`).
    pub lut_entries: Vec<u32>,
    /// Cold gate code of the LUTs ([`LutDesc::cold`]); never part of a partition.
    pub cold_ops: Vec<Op>,
    /// Gates absorbed by LUTs (each is in exactly one LUT's cold code).
    pub lut_gates: u32,
    /// Per net: owner of its value when it may be stale (materialized on demand): `2 * lut` for a
    /// LUT-internal net, `2 * kernel + 1` for a kernel state net without readers whose kernel
    /// has a [`KClass`], [`LAZY_FUSED`] for the output of a gate absorbed into a fused op;
    /// [`NONE`] for nets that are always up to date.
    pub lazy_owner: Vec<u32>,
    /// Kernel specialization classes.
    pub kclasses: Vec<KClass>,
    /// Per kernel: index into [`Program::kclasses`] or [`NONE`].
    pub kernel_class: Vec<u32>,
    /// Timed source nets (clocks, power-on reset) whose values form the clock tuple: tuple bit
    /// `i` = value of `timed_nets[i]`. Empty when phase specialization is disabled.
    pub timed_nets: Vec<u32>,
    /// Number of clock tuples (`2^timed_nets.len()`, or 0 when disabled).
    pub num_tuples: u32,
    /// Bitset words per kernel bitset.
    pub kernel_words: u32,
    /// Per tuple: kernels whose transitions in this phase are no-ops on the observable nets with
    /// no possible diagnostic (opaque, order-insensitive): skipped with word-level bitset ops
    /// (`kernel_words` words per tuple).
    pub tuple_skip: Vec<u64>,
    /// Per tuple: kernels taking the per-kernel fast path ([`KFast`]).
    pub tuple_fast: Vec<u64>,
    /// Per tuple: kernels whose [`KFast::check_order`] is set.
    pub tuple_check: Vec<u64>,
    /// Per tuple: [`PhaseClass::Opaque`] kernels with [`KFast::check_order`] (skipped like
    /// [`Program::tuple_skip`] when fewer than two of their inputs changed).
    pub tuple_copaque: Vec<u64>,
    /// Kernels with per-vector actions ([`KClass::vact`]), used when the kernel's phase has no
    /// tuple class (or phase specialization is off).
    pub vfast: Vec<u64>,
    /// Per `(kernel, tuple)` (`kernel * num_tuples + tuple`): the fast-path descriptor.
    pub kernel_fast: Vec<KFast>,
    /// [`Program::touts`] as bytes (`0xFF` = not proven), when every proven entry is below
    /// `0xFF` (else empty): a quarter of the cache footprint for the interpreter's lookups.
    pub touts8: Vec<u8>,
    /// [`Program::touts`] as 2-bit entries (`0`/`1` = the single output bit, `2` = not proven;
    /// entry `i` at bits `2 * (i % 32)` of word `i / 32`), when every proven entry is 0 or 1
    /// (else empty): the smallest cache footprint.
    pub touts2: Vec<u64>,
    /// Per tuple and kernel bitset word: `[skip, fast, check, copaque]` (the words of
    /// [`Program::tuple_skip`], [`Program::tuple_fast`], [`Program::tuple_check`],
    /// [`Program::tuple_copaque`]), interleaved for the interpreter.
    pub tmasks: Vec<[u64; 4]>,
    /// Per kernel with tuple classes: its pure inputs (bit `i` = input `i` is a function of the
    /// clock tuple; the tuple fixes these bits to [`KFast::pure`]), else 0.
    pub kpure_mask: Vec<u32>,
    /// Tuple-major copy of [`KFast::tout`] (`tuple * kernels + kernel`) for the interpreter.
    pub ktout: Vec<u32>,
    /// Tuple-major copy of [`KFast::pure`] (`tuple * kernels + kernel`) for the interpreter.
    pub kpure: Vec<u32>,
    /// Some kernel with per-vector actions ([`Program::vfast`]) lacks a tuple class in some tuple
    /// (otherwise the interpreter skips the per-vector pass).
    pub vfast_used: bool,
    /// Per `(kernel, tuple)`: the proven phase class.
    pub kernel_phase: Vec<PhaseClass>,
    /// Transparent output tables (entry = output bits, [`NONE`] = not proven: general path).
    pub touts: Vec<u32>,
    /// Names of the compiler passes that produced this program, in order.
    pub passes: Vec<&'static str>,
}

/// Summary counts of a [`Program`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProgramStats {
    /// Combinational ops.
    pub ops: usize,
    /// [`Op::Gate2`] ops.
    pub gate2: usize,
    /// [`Op::GateN`] ops.
    pub gate_n: usize,
    /// [`Op::Resolve`] ops.
    pub resolve: usize,
    /// Ops whose output changes are propagated.
    pub notifying: usize,
    /// Partitions.
    pub partitions: usize,
    /// Largest partition (ops).
    pub max_partition: usize,
    /// Kernels.
    pub kernels: usize,
    /// Kernels with a table op.
    pub table_kernels: usize,
    /// Distinct kernel tables.
    pub ktables: usize,
    /// [`Op::Lut`] ops.
    pub luts: usize,
    /// Gates absorbed by LUTs.
    pub lut_gates: usize,
    /// Total LUT entries.
    pub lut_entries: usize,
    /// Kernels with a specialization class.
    pub spec_kernels: usize,
    /// `(kernel, tuple)` pairs classified opaque / transparent / general (kernels with a class).
    pub phase_classes: [usize; 3],
}

impl Program {
    /// Op range of partition `p`.
    #[inline]
    pub fn partition_ops(&self, p: usize) -> &[Op] {
        let part = self.partitions[p];
        &self.ops[part.start as usize..part.end as usize]
    }

    /// Input nets of kernel op `k`.
    #[inline]
    pub fn kernel_inputs(&self, k: &KernelOp) -> &[u32] {
        &self.kernel_nets[k.inputs as usize..(k.inputs + k.num_inputs) as usize]
    }

    /// Output nets (state nets with readers) of kernel op `k`, in output-bit order.
    #[inline]
    pub fn kernel_out_nets(&self, k: &KernelOp) -> &[u32] {
        &self.kernel_out_nets[k.outs as usize..(k.outs + k.num_outs) as usize]
    }

    /// Indices (into [`Program::kernel_state_nets`]) of kernel op `k`'s state nets with readers.
    #[inline]
    pub fn kernel_out_indices(&self, k: &KernelOp) -> &[u32] {
        &self.kernel_outs[k.outs as usize..(k.outs + k.num_outs) as usize]
    }

    /// State nets of kernel op `k`.
    #[inline]
    pub fn kernel_state_nets(&self, k: &KernelOp) -> &[u32] {
        &self.kernel_nets[k.nets as usize..(k.nets + k.num_nets) as usize]
    }

    /// Support nets of LUT `l`.
    #[inline]
    pub fn lut_support(&self, l: &LutDesc) -> &[u32] {
        &self.lut_nets[l.support as usize..(l.support + l.num_support) as usize]
    }

    /// Output nets of LUT `l`.
    #[inline]
    pub fn lut_outputs(&self, l: &LutDesc) -> &[u32] {
        &self.lut_nets[l.outputs as usize..(l.outputs + l.num_outputs) as usize]
    }

    /// Phase class of kernel `k` under tuple `t` (kernels without a class are General).
    pub fn phase_class(&self, k: u32, t: u32) -> PhaseClass {
        if t >= self.num_tuples {
            return PhaseClass::General;
        }
        self.kernel_phase[(k * self.num_tuples + t) as usize]
    }

    /// Summary counts.
    pub fn stats(&self) -> ProgramStats {
        let mut s = ProgramStats {
            ops: self.ops.len(),
            partitions: self.partitions.len(),
            max_partition: self
                .partitions
                .iter()
                .map(Partition::len)
                .max()
                .unwrap_or(0),
            kernels: self.kernel_ops.len(),
            ktables: self.ktables.len(),
            ..ProgramStats::default()
        };
        s.luts = self.luts.len();
        s.lut_gates = self.lut_gates as usize;
        s.lut_entries = self.lut_entries.len();
        s.spec_kernels = self.kernel_class.iter().filter(|&&c| c != NONE).count();
        let nt = self.num_tuples as usize;
        for (k, &c) in self.kernel_class.iter().enumerate() {
            if c == NONE {
                continue;
            }
            for t in 0..nt {
                s.phase_classes[self.phase_class(k as u32, t as u32) as usize] += 1;
            }
        }
        for op in &self.ops {
            match op {
                Op::Gate2 { .. } => s.gate2 += 1,
                Op::GateN { .. } => s.gate_n += 1,
                Op::Resolve { .. } => s.resolve += 1,
                Op::Lut { .. } => {}
            }
            if op.notifies() {
                s.notifying += 1;
            }
        }
        s.table_kernels = self
            .kernel_ops
            .iter()
            .filter(|k| matches!(k.kind, KernelOpKind::KTable { .. }))
            .count();
        s
    }
}
