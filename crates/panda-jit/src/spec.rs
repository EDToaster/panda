//! Self-contained descriptions of what gets compiled.
//!
//! The code generator never reads the [`Program`] directly: it compiles a [`PartSpec`] (one
//! partition) or a [`RoundSpec`] (the kernel fast paths of one clock tuple), extracted here.
//! A spec holds *everything* the generated code depends on — net indices, gate tables, decoded
//! change-propagation records, LUT tables, and the program contents behind every host callback
//! (resolver drivers, LUT cold code) — so two equal specs always compile to interchangeable code.
//! That makes specs safe cache keys: a recompiled program reuses the code of every partition
//! whose spec did not change ([`crate::JitCache`]).

use std::hash::{Hash, Hasher};

use panda_fast::Program;
use panda_fast::ir::{LAZY_FUSED, NONE, Op, table};
use panda_fast::native::NotifyRecord;

/// A gate's output store.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Out {
    /// Output net.
    pub net: u32,
    /// Change propagation of the output, if it has readers to notify.
    pub notify: Option<NotifyRecord>,
    /// Write the value to the state array (false only for outputs of fused-away gates that
    /// nothing reads from memory: they are recomputed on demand by `FastSim::values`).
    pub store: bool,
}

/// One op of a partition.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum JOp {
    /// `out = (table >> 2 * (v[a] | v[b] << 2)) & 3`.
    Gate2 {
        table: u32,
        a: u32,
        b: u32,
        out: Out,
    },
    /// N-input AND/OR/XOR fold (`fold` = `panda_fast::ir::table` selector), optionally inverted.
    GateN {
        fold: u8,
        invert: bool,
        args: Vec<u32>,
        out: Out,
    },
    /// Resolver: host callback with the op index. `desc` pins the program contents the callback
    /// uses (drivers, termination, flat index, output, record).
    Resolve {
        op: u32,
        desc: Vec<u64>,
        rec: Option<NotifyRecord>,
    },
    /// LUT.
    Lut(Box<LutSpec>),
}

/// A LUT op.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct LutSpec {
    /// LUT index (runtime `lut_last` / `lut_stale` slot, cold callback argument).
    pub lut: u32,
    /// Support nets (bit `i` of the index = net `i`).
    pub support: Vec<u32>,
    /// Output nets with their change propagation.
    pub outputs: Vec<(u32, Option<NotifyRecord>)>,
    /// Offset of the entries in `Program::lut_entries`.
    pub table: u32,
    /// The entries (pins the table contents).
    pub entries: Vec<u32>,
    /// The cold gate code and its records (pins what the cold callback runs).
    pub cold: Vec<(OpKey, u32)>,
}

/// A hashable copy of an [`Op`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct OpKey(pub [u32; 4], pub Vec<u32>);

impl OpKey {
    fn of(prog: &Program, op: &Op) -> OpKey {
        match *op {
            Op::Gate2 {
                t,
                notify,
                out,
                a,
                b,
            } => OpKey([0, u32::from(t) | u32::from(notify) << 8, out, a], vec![b]),
            Op::GateN {
                fold,
                invert,
                notify,
                out,
                args,
                n,
            } => OpKey(
                [
                    1,
                    u32::from(fold) | u32::from(invert) << 8 | u32::from(notify) << 9,
                    out,
                    n,
                ],
                prog.args[args as usize..(args + n) as usize].to_vec(),
            ),
            Op::Resolve { notify, out, res } => OpKey([2, u32::from(notify), out, res], Vec::new()),
            Op::Lut { lut } => OpKey([3, lut, 0, 0], Vec::new()),
        }
    }
}

/// The lengths of the runtime arrays a program's native code indexes (the `NativeCtx` /
/// `EventCtx` arrays, see [`panda_fast::native`]). Every index the generated code embeds is
/// checked against them before code generation ([`PartSpec::check`],
/// `EventSpec::check`); paranoid code also compares them with the context's lengths at entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Shape {
    /// Nets (`values`).
    pub nets: u32,
    /// Partitions (`EventCtx::parts`).
    pub partitions: u32,
    /// Partition bitset words (`dirty`).
    pub part_words: u32,
    /// Kernels (`kin`, `kin_prev`, `kob`, `pending` pairs).
    pub kernels: u32,
    /// Kernel bitset words (`kpending`, `ddirty`, `fastok`, `lazy`, `unsynced`).
    pub kernel_words: u32,
    /// Boundary-log slots (`bval`, `bepoch`).
    pub log: u32,
    /// LUTs (`lut_last`; `lut_stale` has `luts.div_ceil(64)` words).
    pub luts: u32,
    /// LUT entries (`lut_entries`).
    pub lut_entries: u32,
    /// Clock tuples (`tlast` has `tuples * kernel_words` words).
    pub tuples: u32,
}

/// Largest byte offset the generated code adds to a base pointer as an immediate.
const MAX_OFFSET: u64 = i32::MAX as u64;

impl Shape {
    /// The shape of `prog`'s runtime state; an error if some length does not fit the
    /// generator's 32-bit immediates (`num_nets` is checked against `lazy_owner` etc. by the
    /// program's own verification).
    pub fn of(prog: &Program) -> Result<Shape, String> {
        let n = |x: usize, what: &str| {
            u32::try_from(x)
                .ok()
                .filter(|&v| v < 1 << 28)
                .ok_or_else(|| format!("program too large for the JIT ({what} = {x})"))
        };
        let nk = prog.kernel_ops.len();
        let shape = Shape {
            nets: n(prog.num_nets as usize, "nets")?,
            partitions: n(prog.partitions.len(), "partitions")?,
            part_words: n(prog.partitions.len().div_ceil(64), "partition words")?,
            kernels: n(nk, "kernels")?,
            kernel_words: n(nk.div_ceil(64), "kernel words")?,
            log: n(prog.num_log as usize, "log slots")?,
            luts: n(prog.luts.len(), "LUTs")?,
            lut_entries: n(prog.lut_entries.len(), "LUT entries")?,
            tuples: n(prog.num_tuples as usize, "clock tuples")?,
        };
        if prog.kernel_words != shape.kernel_words
            || prog.lazy_owner.len() != prog.num_nets as usize
            || prog.notify_off.len() != prog.num_nets as usize
            || prog.op_rec.len() != prog.ops.len()
        {
            return Err("inconsistent program arrays".into());
        }
        Ok(shape)
    }

    /// `index < len` for an array of `elem`-byte elements addressed with an immediate offset.
    pub fn idx(index: u64, len: u32, elem: u64, what: &str) -> Result<(), String> {
        if index < u64::from(len) && (index + 1) * elem <= MAX_OFFSET {
            Ok(())
        } else {
            Err(format!("{what} index {index} out of range (length {len})"))
        }
    }

    /// Every index of a decoded change-propagation record in range.
    pub fn check_record(&self, rec: &NotifyRecord) -> Result<(), String> {
        Shape::idx(rec.net.into(), self.nets, 1, "record net")?;
        for &(w, _) in &rec.parts {
            Shape::idx(w.into(), self.part_words, 8, "record partition word")?;
        }
        for &(w, _, _) in &rec.kernels {
            Shape::idx(w.into(), self.kernel_words, 8, "record kernel word")?;
        }
        if !rec.kernels.is_empty() && rec.log != NONE {
            Shape::idx(rec.log.into(), self.log, 4, "record log slot")?;
        }
        for &(k, pos) in &rec.kin {
            Shape::idx(k.into(), self.kernels, 4, "record kernel")?;
            if pos >= 32 {
                return Err(format!("record input position {pos}"));
            }
        }
        Ok(())
    }

    /// A store to `net` with optional change propagation.
    pub fn check_out(&self, net: u32, rec: Option<&NotifyRecord>) -> Result<(), String> {
        Shape::idx(net.into(), self.nets, 1, "output net")?;
        if let Some(r) = rec {
            if r.net != net {
                return Err(format!("record of net {} on output {net}", r.net));
            }
            self.check_record(r)?;
        }
        Ok(())
    }
}

/// One partition.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PartSpec {
    pub ops: Vec<JOp>,
    /// Paranoid code (runtime checks) for this shape; `None`: no runtime checks.
    pub paranoid: Option<Shape>,
}

impl PartSpec {
    /// Every index the generated code embeds is in range for `shape` (the bounds argument of
    /// the generated code's memory safety: all its addresses are base pointers of the context
    /// plus these immediates, except the LUT table read, whose index has `support.len()` bits).
    pub fn check(&self, shape: &Shape) -> Result<(), String> {
        for op in &self.ops {
            match op {
                JOp::Gate2 { a, b, out, .. } => {
                    Shape::idx((*a).into(), shape.nets, 1, "operand")?;
                    Shape::idx((*b).into(), shape.nets, 1, "operand")?;
                    shape.check_out(out.net, out.notify.as_ref())?;
                }
                JOp::GateN {
                    fold, args, out, ..
                } => {
                    if ![table::AND, table::OR, table::XOR].contains(fold) {
                        return Err(format!("fold {fold}"));
                    }
                    for &x in args {
                        Shape::idx(x.into(), shape.nets, 1, "operand")?;
                    }
                    shape.check_out(out.net, out.notify.as_ref())?;
                }
                JOp::Resolve { desc, .. } => {
                    // Evaluated by the host (no memory access in the generated code); the
                    // scheduler and the register cache use its output and drivers.
                    if desc.len() < 4 {
                        return Err("resolver descriptor".into());
                    }
                    Shape::idx(desc[0], shape.nets, 1, "resolver output")?;
                    for &x in &desc[4..] {
                        Shape::idx(x, shape.nets, 1, "resolver driver")?;
                    }
                }
                JOp::Lut(l) => {
                    Shape::idx(l.lut.into(), shape.luts, 4, "LUT")?;
                    Shape::idx((l.lut / 64).into(), shape.luts.div_ceil(64), 8, "LUT word")?;
                    if l.support.len() > 16 || l.outputs.is_empty() || l.outputs.len() > 32 {
                        return Err(format!("LUT {}: shape", l.lut));
                    }
                    for &x in &l.support {
                        Shape::idx(x.into(), shape.nets, 1, "LUT support")?;
                    }
                    let size = 1u64 << l.support.len();
                    if l.entries.len() as u64 != size {
                        return Err(format!("LUT {}: {} entries", l.lut, l.entries.len()));
                    }
                    Shape::idx(
                        u64::from(l.table) + size - 1,
                        shape.lut_entries,
                        4,
                        "LUT table",
                    )?;
                    for (net, rec) in &l.outputs {
                        shape.check_out(*net, rec.as_ref())?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Extract partition `p` of `prog`.
    pub fn extract(prog: &Program, p: usize, paranoid: Option<Shape>) -> PartSpec {
        let part = prog.partitions[p];
        let range = part.start as usize..part.end as usize;
        // Nets some op of this partition reads from memory (host callbacks: resolvers, LUT cold
        // code): their stores cannot be elided.
        let mut mem_read = std::collections::HashSet::new();
        for op in &prog.ops[range.clone()] {
            match *op {
                Op::Resolve { res, .. } => {
                    let r = prog.resolvers[res as usize];
                    mem_read.extend(&prog.args[r.args as usize..(r.args + r.n) as usize]);
                }
                Op::Lut { lut } => {
                    let l = &prog.luts[lut as usize];
                    mem_read.extend(prog.lut_support(l));
                    for c in &prog.cold_ops[l.cold.start as usize..l.cold.end as usize] {
                        match *c {
                            Op::Gate2 { a, b, .. } => mem_read.extend([a, b]),
                            Op::GateN { args, n, .. } => {
                                mem_read.extend(&prog.args[args as usize..(args + n) as usize])
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        let out = |net: u32, rec: u32| Out {
            net,
            notify: (rec != NONE).then(|| prog.record(rec)),
            store: !(rec == NONE
                && prog.lazy_owner[net as usize] == LAZY_FUSED
                && prog.notify_off[net as usize] == NONE
                && !mem_read.contains(&net)),
        };
        let mut ops = Vec::with_capacity(range.len());
        for i in range {
            let rec = prog.op_rec[i];
            ops.push(match prog.ops[i] {
                Op::Gate2 {
                    t, out: o, a, b, ..
                } => JOp::Gate2 {
                    table: prog.tables[t as usize],
                    a,
                    b,
                    out: out(o, rec),
                },
                Op::GateN {
                    fold,
                    invert,
                    out: o,
                    args,
                    n,
                    ..
                } => JOp::GateN {
                    fold,
                    invert,
                    args: prog.args[args as usize..(args + n) as usize].to_vec(),
                    out: out(o, rec),
                },
                Op::Resolve { out: o, res, .. } => {
                    let r = prog.resolvers[res as usize];
                    let mut desc = vec![
                        u64::from(o),
                        u64::from(res),
                        u64::from(r.flat_index),
                        r.termination as u64,
                    ];
                    desc.extend(
                        prog.args[r.args as usize..(r.args + r.n) as usize]
                            .iter()
                            .map(|&x| u64::from(x)),
                    );
                    JOp::Resolve {
                        op: i as u32,
                        desc,
                        rec: (rec != NONE).then(|| prog.record(rec)),
                    }
                }
                Op::Lut { lut } => {
                    let l = &prog.luts[lut as usize];
                    let recs =
                        &prog.lut_rec[l.outputs as usize..(l.outputs + l.num_outputs) as usize];
                    let cold = (l.cold.start as usize..l.cold.end as usize)
                        .map(|c| (OpKey::of(prog, &prog.cold_ops[c]), prog.cold_rec[c]))
                        .collect();
                    JOp::Lut(Box::new(LutSpec {
                        lut,
                        support: prog.lut_support(l).to_vec(),
                        outputs: prog
                            .lut_outputs(l)
                            .iter()
                            .zip(recs)
                            .map(|(&o, &r)| (o, (r != NONE).then(|| prog.record(r))))
                            .collect(),
                        table: l.table,
                        entries: prog.lut_entries
                            [l.table as usize..l.table as usize + (1usize << l.num_support)]
                            .to_vec(),
                        cold,
                    }))
                }
            });
        }
        PartSpec { ops, paranoid }
    }

    /// 64-bit content hash (cache key; equality is checked on a hit).
    pub fn key(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }
}
