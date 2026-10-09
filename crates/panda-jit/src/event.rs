//! Event functions: `FastSim::process_event`'s sweeps and round loop for one clock tuple, with
//! the kernel fast paths specialized to the tuple (contract: [`panda_fast::native::EventCtx`]).
//!
//! For tuple `t` every mask of the interpreter's `fast_round` (`Program::tmasks`, the tracked
//! set, the per-vector set) is a constant, every evaluated kernel's output table is an
//! immediate (≤ 32 entries of 0/1) or a table in the function's data, and every output write
//! carries its decoded change-propagation record. The function:
//!
//! 1. sweeps the dirty partitions (one guarded call per partition, in order);
//! 2. **decides** a round boundary per kernel bitset word, without side effects: the skip /
//!    unchanged-transparent sets by word ops, then one block per evaluated kernel (dispatched on
//!    the bit index) that looks up its output bits and queues a write if they differ from
//!    `kob`; it returns [`EVENT_BAIL`] as soon as a kernel would need anything else;
//! 3. **commits** the bitset updates and counters, starts the next interval and **writes** the
//!    queued outputs with inline change propagation;
//! 4. loops to 1 until nothing is pending or dirty.
//!
//! [`EVENT_BAIL`]: panda_fast::native::EVENT_BAIL

use std::hash::{Hash, Hasher};

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types::{I32, I64};
use cranelift_codegen::ir::{
    AbiParam, Block, Function, InstBuilder, JumpTableData, SigRef, UserFuncName, Value,
};
use cranelift_frontend::{FunctionBuilder, Variable};
use cranelift_module::{DataDescription, Linkage, Module};

use panda_fast::Program;
use panda_fast::ir::{NONE, VACT_OPAQUE};
use panda_fast::native::{EVENT_BAIL, EVENT_DONE, NotifyRecord, ctx_offsets as off};

use crate::codegen::{Compiler, Gen, fault, mf};
use crate::spec::{Out, Shape};

/// Largest tracked-kernel list an event function snapshots inline.
const MAX_TRACKED: usize = 4096;
/// Most partitions swept with one guarded call each (larger programs get no event function).
const MAX_PARTITIONS: usize = 2048;

/// How a kernel's table is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Tab {
    /// 2-bit codes in a 64-bit immediate (≤ 32 entries): 0/1 = output bit, 2 = not proven /
    /// general path, 3 = opaque (per-vector actions).
    Imm(u64),
    /// 2-bit codes (as [`Tab::Imm`]) packed 32 per `u64` at this (8-aligned) offset of the
    /// data blob.
    Packed2(u32),
    /// Bytes at this offset of the data blob: output bits, `0xFF` = general, `0xFE` = opaque.
    Bytes(u32),
    /// `u32`s at this (4-aligned) offset of the data blob, raw ([`NONE`], [`VACT_OPAQUE`]).
    Words(u32),
    /// The output bit is input bit `.0` of the vector (inverted with `.1`) for every vector the
    /// kernel can see (a transparent latch: `Q = D`); always proven.
    Bit(u32, bool),
}

/// Per kernel bitset word: the constant masks of the tuple.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct WordSpec {
    skip: u64,
    fast: u64,
    check: u64,
    copaque: u64,
    tracked: u64,
    /// Fast kernels with a transparent output table in this tuple.
    trans: u64,
    /// Per-vector kernels without a tuple class here.
    vg: u64,
}

/// A kernel evaluated through a table.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct TabK {
    k: u32,
    /// Pure input bits of the tuple (transparent kernels; 0 for per-vector kernels).
    pure: u32,
    /// Mask of the pure input bits (`Program::kpure_mask`; 0 for per-vector kernels).
    pm: u32,
    /// Number of inputs (the table has `2^n_in` entries).
    n_in: u32,
    tab: Tab,
}

/// A kernel whose outputs a round may write.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct WriteK {
    k: u32,
    outs: Vec<(u32, Option<NotifyRecord>)>,
}

/// One clock tuple's event function.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct EventSpec {
    tuple: u32,
    num_tuples: u32,
    num_kernels: u32,
    num_partitions: u32,
    words: Vec<WordSpec>,
    trans: Vec<TabK>,
    vk: Vec<TabK>,
    writes: Vec<WriteK>,
    tracked_list: Vec<u32>,
    /// Per clock-tuple bit: the timed net and its change propagation.
    timed: Vec<(u32, Option<NotifyRecord>)>,
    blob: Vec<u8>,
    /// Decide and write in one pass (see [`EventSpec::extract`]).
    fused: bool,
    /// Paranoid code (runtime checks) for this shape; `None`: no runtime checks.
    paranoid: Option<Shape>,
}

/// Table encoder: deduplicates identical tables in the data blob.
#[derive(Default)]
struct Encoder {
    blob: Vec<u8>,
    seen: std::collections::HashMap<Vec<u32>, Tab>,
}

impl Encoder {
    /// Encode a table (`entries[v]`, raw program values).
    fn encode(&mut self, entries: &[u32]) -> Tab {
        if let Some(&t) = self.seen.get(entries) {
            return t;
        }
        let t = self.encode_new(entries);
        self.seen.insert(entries.to_vec(), t);
        t
    }

    fn align(&mut self, a: usize) -> u32 {
        while !self.blob.len().is_multiple_of(a) {
            self.blob.push(0);
        }
        self.blob.len() as u32
    }

    fn encode_new(&mut self, entries: &[u32]) -> Tab {
        let small = |e: u32| e <= 1 || e == NONE || e == VACT_OPAQUE;
        let code = |e: u32| match e {
            NONE => 2u64,
            VACT_OPAQUE => 3,
            e => u64::from(e),
        };
        if entries.iter().all(|&e| small(e)) {
            if entries.len() <= 32 {
                return Tab::Imm(
                    entries
                        .iter()
                        .enumerate()
                        .fold(0, |t, (i, &e)| t | code(e) << (2 * i)),
                );
            }
            let off = self.align(8);
            for chunk in entries.chunks(32) {
                let w = chunk
                    .iter()
                    .enumerate()
                    .fold(0u64, |t, (i, &e)| t | code(e) << (2 * i));
                self.blob.extend_from_slice(&w.to_ne_bytes());
            }
            return Tab::Packed2(off);
        }
        if entries
            .iter()
            .all(|&e| e < 0xFE || e == NONE || e == VACT_OPAQUE)
        {
            let off = self.blob.len() as u32;
            self.blob.extend(entries.iter().map(|&e| match e {
                NONE => 0xFF,
                VACT_OPAQUE => 0xFE,
                e => e as u8,
            }));
            return Tab::Bytes(off);
        }
        let off = self.align(4);
        for &e in entries {
            self.blob.extend_from_slice(&e.to_ne_bytes());
        }
        Tab::Words(off)
    }
}

/// [`Tab::Bit`] if the transparent output table `entries` (one output) is a single input bit
/// (or its inverse) of every vector `v` with `v & pure_mask == pure` (the vectors the kernel
/// can see in the tuple); the bit must be tracked in the input index (`kin_mask`).
fn copy_bit(entries: &[u32], pure_mask: u32, pure: u32, kin_mask: u32) -> Option<Tab> {
    let n = entries.len().trailing_zeros();
    let seen = || (0..entries.len() as u32).filter(|v| v & pure_mask == pure);
    (0..n)
        .filter(|&i| pure_mask >> i & 1 == 0 && kin_mask >> i & 1 == 1)
        .find_map(|i| {
            [false, true].into_iter().find_map(|inv| {
                seen()
                    .all(|v| entries[v as usize] == ((v >> i) & 1) ^ u32::from(inv))
                    .then_some(Tab::Bit(i, inv))
            })
        })
}

impl EventSpec {
    /// Extract tuple `t`'s event function, or `None` if the program has no tuple classes there
    /// (or something exceeds the generator's limits).
    pub fn extract(prog: &Program, t: u32, paranoid: Option<Shape>) -> Option<EventSpec> {
        if t >= prog.num_tuples
            || prog.tracked_list.len() > MAX_TRACKED
            || prog.partitions.len() > MAX_PARTITIONS
        {
            return None;
        }
        let kw = prog.kernel_words as usize;
        let nk = prog.kernel_ops.len();
        let tu = t as usize;
        let mut enc = Encoder::default();
        let mut spec = EventSpec {
            tuple: t,
            num_tuples: prog.num_tuples,
            num_kernels: nk as u32,
            num_partitions: prog.partitions.len() as u32,
            words: Vec::with_capacity(kw),
            trans: Vec::new(),
            vk: Vec::new(),
            writes: Vec::new(),
            tracked_list: prog.tracked_list.clone(),
            timed: prog
                .timed_nets
                .iter()
                .map(|&n| {
                    let off = prog.notify_off[n as usize];
                    (n, (off != NONE).then(|| prog.record(off)))
                })
                .collect(),
            blob: Vec::new(),
            fused: false,
            paranoid,
        };
        let bit = |m: u64, k: usize| m >> (k % 64) & 1 == 1;
        for w in 0..kw {
            let [skip, fast, check, copaque] = prog.tmasks[tu * kw + w];
            let mut trans = 0u64;
            let vg = if prog.vfast_used {
                prog.vfast[w] & !(prog.tuple_skip[tu * kw + w] | prog.tuple_fast[tu * kw + w])
            } else {
                0
            };
            for b in 0..64 {
                let k = w * 64 + b;
                if k >= nk {
                    break;
                }
                let kop = &prog.kernel_ops[k];
                let n_in = kop.num_inputs as usize;
                let mut writes = false;
                if bit(fast, k) {
                    let tout = prog.ktout[tu * nk + k];
                    if tout != NONE {
                        if n_in > 20 {
                            return None;
                        }
                        trans |= 1 << b;
                        let entries = &prog.touts[tout as usize..tout as usize + (1 << n_in)];
                        let tab = copy_bit(
                            entries,
                            prog.kpure_mask[k],
                            prog.kpure[tu * nk + k],
                            prog.kernel_kin_mask[k],
                        )
                        .unwrap_or_else(|| enc.encode(entries));
                        spec.trans.push(TabK {
                            k: k as u32,
                            pure: prog.kpure[tu * nk + k],
                            pm: prog.kpure_mask[k],
                            n_in: n_in as u32,
                            tab,
                        });
                        writes = true;
                    }
                }
                if bit(vg, k) {
                    let c = prog.kernel_class[k];
                    if c == NONE || n_in > 20 {
                        return None;
                    }
                    // The table is indexed by the input vector (the interpreter's indexing is
                    // bounds-checked; this length is not part of the program's verification).
                    let vact = &prog.kclasses[c as usize].vact;
                    if vact.len() != 1 << n_in {
                        return None;
                    }
                    let tab = enc.encode(vact);
                    spec.vk.push(TabK {
                        k: k as u32,
                        pure: 0,
                        pm: 0,
                        n_in: n_in as u32,
                        tab,
                    });
                    writes = true;
                }
                if writes {
                    if kop.num_outs > 31 {
                        return None;
                    }
                    let outs = prog.kernel_out_nets(kop);
                    let recs =
                        &prog.kernel_out_rec[kop.outs as usize..(kop.outs + kop.num_outs) as usize];
                    spec.writes.push(WriteK {
                        k: k as u32,
                        outs: outs
                            .iter()
                            .zip(recs)
                            .map(|(&n, &r)| (n, (r != NONE).then(|| prog.record(r))))
                            .collect(),
                    });
                }
            }
            spec.words.push(WordSpec {
                skip,
                fast,
                check,
                copaque,
                tracked: prog.tracked[w],
                trans,
                vg,
            });
        }
        spec.blob = enc.blob;
        // Fused decide + write: no per-vector kernels, every transparent table proven for every
        // input vector under the tuple's pure bits, and no transparent kernel reading another's
        // output (so the lookups see the boundary values even with interleaved writes).
        let proven = spec.trans.iter().all(|tk| {
            let kop = &prog.kernel_ops[tk.k as usize];
            let n = kop.num_inputs as usize;
            let tout = prog.ktout[tu * nk + tk.k as usize] as usize;
            let pm = prog.kpure_mask[tk.k as usize];
            (0..1u32 << n)
                .filter(|v| v & pm == tk.pure)
                .all(|v| prog.touts[tout + v as usize] != NONE)
        });
        let outs: std::collections::HashSet<u32> = spec
            .writes
            .iter()
            .flat_map(|w| w.outs.iter().map(|o| o.0))
            .collect();
        let feeds = spec.trans.iter().any(|tk| {
            prog.kernel_inputs(&prog.kernel_ops[tk.k as usize])
                .iter()
                .any(|n| outs.contains(n))
        });
        spec.fused = spec.vk.is_empty() && proven && !feeds;
        Some(spec)
    }

    /// Every index the generated code embeds, and every table read's extent, is in range for
    /// `shape` (with input vectors below `2^n_in`, which the program's verification guarantees
    /// for `kin | pure`, and which paranoid code checks at runtime); the kernel lists are
    /// consistent with the masks.
    pub fn check(&self, shape: &Shape) -> Result<(), String> {
        let nt = u64::from(shape.tuples);
        let kw = u64::from(shape.kernel_words);
        if self.num_tuples != shape.tuples
            || self.tuple >= shape.tuples
            || self.num_kernels != shape.kernels
            || self.num_partitions != shape.partitions
            || self.words.len() as u64 != kw
        {
            return Err("event spec does not match the program".into());
        }
        if nt * kw > 0 {
            let words = u32::try_from(nt * kw).map_err(|_| "too many tlast words")?;
            Shape::idx(nt * kw - 1, words, 8, "tlast word")?;
        }
        if shape.partitions > 0 {
            Shape::idx(
                u64::from(shape.partitions) - 1,
                shape.partitions,
                8,
                "partition",
            )?;
        }
        if shape.kernels > 0 {
            // `pending` holds one 8-byte pair per kernel.
            Shape::idx(
                u64::from(shape.kernels) - 1,
                shape.kernels,
                8,
                "pending pair",
            )?;
        }
        let blob = self.blob.len() as u64;
        let check_tab = |tk: &TabK| -> Result<(), String> {
            Shape::idx(tk.k.into(), shape.kernels, 4, "kernel")?;
            if tk.n_in > 20 || u64::from(tk.pure | tk.pm) >> tk.n_in != 0 {
                return Err(format!("kernel {}: inputs", tk.k));
            }
            let size = 1u64 << tk.n_in;
            let end = match tk.tab {
                Tab::Imm(_) => (size <= 32).then_some(0),
                Tab::Packed2(o) => Some(u64::from(o) + size.div_ceil(32) * 8),
                Tab::Bytes(o) => Some(u64::from(o) + size),
                Tab::Words(o) => Some(u64::from(o) + size * 4),
                Tab::Bit(i, _) => (i < tk.n_in).then_some(0),
            };
            match end {
                Some(e) if e <= blob && e <= i32::MAX as u64 => Ok(()),
                _ => Err(format!("kernel {}: table out of range", tk.k)),
            }
        };
        let (mut ti, mut vi) = (0usize, 0usize);
        for (w, ws) in self.words.iter().enumerate() {
            for b in 0..64u32 {
                let k = (w * 64) as u32 + b;
                if ws.trans >> b & 1 == 1 {
                    let tk = self.trans.get(ti).ok_or("transparent kernel list")?;
                    if tk.k != k {
                        return Err("transparent kernel list order".into());
                    }
                    check_tab(tk)?;
                    ti += 1;
                }
                if ws.vg >> b & 1 == 1 {
                    let tk = self.vk.get(vi).ok_or("per-vector kernel list")?;
                    if tk.k != k {
                        return Err("per-vector kernel list order".into());
                    }
                    check_tab(tk)?;
                    vi += 1;
                }
                let any = ws.trans | ws.vg | ws.skip | ws.fast | ws.check | ws.copaque | ws.tracked;
                if any >> b & 1 == 1 && k >= shape.kernels {
                    return Err(format!("mask bit of kernel {k} beyond the kernels"));
                }
            }
        }
        if ti != self.trans.len() || vi != self.vk.len() {
            return Err("kernel lists longer than the masks".into());
        }
        for (i, wk) in self.writes.iter().enumerate() {
            Shape::idx(wk.k.into(), shape.kernels, 4, "written kernel")?;
            if wk.outs.len() > 31 {
                return Err(format!("kernel {}: {} outputs", wk.k, wk.outs.len()));
            }
            if self.fused && self.trans.get(i).map(|t| t.k) != Some(wk.k) {
                return Err("fused write list order".into());
            }
            for (net, rec) in &wk.outs {
                shape.check_out(*net, rec.as_ref())?;
            }
        }
        if self.fused && (self.writes.len() != self.trans.len() || !self.vk.is_empty()) {
            return Err("fused event spec".into());
        }
        for &k in &self.tracked_list {
            Shape::idx(k.into(), shape.kernels, 4, "tracked kernel")?;
        }
        if self.timed.len() > 31 {
            return Err("too many timed sources".into());
        }
        for (net, rec) in &self.timed {
            shape.check_out(*net, rec.as_ref())?;
        }
        Ok(())
    }

    /// True if decide and write run in one pass.
    pub fn is_fused(&self) -> bool {
        self.fused
    }

    /// 64-bit content hash (cache key; equality is checked on a hit).
    pub fn key(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }
}

/// Compile one event function into `cc`; returns its index among the job's functions.
pub(crate) fn add_event(cc: &mut Compiler, spec: &EventSpec) -> Result<usize, String> {
    let mut sig = cc.ptr_sig(&[I64]);
    sig.returns.push(AbiParam::new(I32));
    let idx = cc.num_funcs();
    let id = cc
        .module
        .declare_function(&format!("event{idx}"), Linkage::Local, &sig)
        .map_err(|e| e.to_string())?;
    let blob_id = if spec.blob.is_empty() {
        None
    } else {
        let d = cc
            .module
            .declare_anonymous_data(false, false)
            .map_err(|e| e.to_string())?;
        let mut desc = DataDescription::new();
        desc.define(spec.blob.clone().into_boxed_slice());
        desc.set_align(8);
        cc.module
            .define_data(d, &desc)
            .map_err(|e| format!("cranelift: {e:?}"))?;
        Some(d)
    };
    cc.ctx.clear();
    cc.ctx.func = Function::with_name_signature(UserFuncName::user(1, idx as u32), sig);
    let cb_sig = cc.ptr_sig(&[I32]);
    let wrap_sig = cc.ptr_sig(&[]);
    let blob_gv = blob_id.map(|d| cc.module.declare_data_in_func(d, &mut cc.ctx.func));
    {
        let mut b = FunctionBuilder::new(&mut cc.ctx.func, &mut cc.fctx);
        let cb = b.import_signature(cb_sig);
        let wrap = b.import_signature(wrap_sig);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        let ctx = b.block_params(entry)[0];
        let ectx = b.block_params(entry)[1];
        let mut g = Gen::new(b, ctx, cb);
        if let Some(shape) = &spec.paranoid {
            g.check_shape(shape);
            let mut acc = g.iconst(0);
            for (o, v) in [
                (off::E_N_TUPLES, shape.tuples),
                (off::E_N_PARTS, shape.partitions),
            ] {
                let x = g.b.ins().load(I32, mf(), ectx, o);
                let d = g.b.ins().bxor_imm_s(x, i64::from(v as i32));
                acc = g.b.ins().bor(acc, d);
            }
            g.fault_if(acc, fault::EVENT_SHAPE);
        }
        let mut e = EventGen::new(g, ectx, spec, blob_gv, wrap);
        e.emit();
        e.g.b.seal_all_blocks();
        e.g.b.finalize(cc.module.target_config());
    }
    cc.define(id)?;
    Ok(idx)
}

/// Pass-1 results of one word (SSA values dominating the commit).
struct WordRes {
    todo: Value,
    skip: Value,
    same: Value,
    /// Evaluated fast kernels (`fast & !same`).
    eval: Value,
    /// Of those, opaque (no transparent table).
    opq: Value,
    /// Of those, transparent (looked up).
    trans_f: Value,
    /// Per-vector transparent kernels (data-dirty bits to set).
    vdd: Value,
    vopq: Value,
    vtrans: Value,
    vevals: Value,
}

struct EventGen<'a, 's> {
    g: Gen<'a>,
    ectx: Value,
    spec: &'s EventSpec,
    blob: Option<Value>,
    wrap: SigRef,
    /// Return [`EVENT_BAIL`].
    bail: Block,
    /// Number of queued writes.
    npend: Variable,
}

impl<'a, 's> EventGen<'a, 's> {
    fn new(
        mut g: Gen<'a>,
        ectx: Value,
        spec: &'s EventSpec,
        blob_gv: Option<cranelift_codegen::ir::GlobalValue>,
        wrap: SigRef,
    ) -> Self {
        let blob = blob_gv.map(|gv| g.b.ins().symbol_value(I64, gv));
        let bail = g.b.create_block();
        let var = |g: &mut Gen<'a>| {
            let v = g.b.declare_var(I64);
            let z = g.b.ins().iconst(I64, 0);
            g.b.def_var(v, z);
            v
        };
        let npend = var(&mut g);
        EventGen {
            g,
            ectx,
            spec,
            blob,
            wrap,
            bail,
            npend,
        }
    }

    fn i64c(&mut self, v: u64) -> Value {
        self.g.b.ins().iconst(I64, v as i64)
    }

    fn eptr(&mut self, offset: i32) -> Value {
        self.g.b.ins().load(I64, mf(), self.ectx, offset)
    }

    /// `if c != 0 { bail }`.
    fn bail_if(&mut self, c: Value) {
        let cont = self.g.b.create_block();
        self.g.b.ins().brif(c, self.bail, &[], cont, &[]);
        self.g.b.switch_to_block(cont);
    }

    /// Kernel `k`'s 32-bit entry of the per-kernel array whose base pointer is at `ptr_off` of
    /// the context (`from_ectx`) or of the native context.
    fn k32(&mut self, ptr_off: i32, from_ectx: bool, k: u32) -> Value {
        let base = if from_ectx {
            self.eptr(ptr_off)
        } else {
            self.g.ctx_ptr(ptr_off)
        };
        self.g.b.ins().load(I32, mf(), base, (k * 4) as i32)
    }

    fn add_var(&mut self, v: Variable, x: Value) {
        let c = self.g.b.use_var(v);
        let n = self.g.b.ins().iadd(c, x);
        self.g.b.def_var(v, n);
    }

    /// Input vector `kin | pure` of a kernel table lookup (paranoid code checks that it is
    /// inside the table and agrees with the tuple's pure bits).
    fn vector(&mut self, tk: &TabK, kin: Value) -> Value {
        let v = if tk.pure != 0 {
            self.g.b.ins().bor_imm_s(kin, i64::from(tk.pure as i32))
        } else {
            kin
        };
        if self.spec.paranoid.is_some() {
            let hi = self.g.b.ins().ushr_imm_u(v, i64::from(tk.n_in));
            self.g.fault_if(hi, fault::TABLE_INDEX);
            if tk.pm != 0 {
                let p = self.g.b.ins().band_imm_s(v, i64::from(tk.pm as i32));
                let bad = self
                    .g
                    .b
                    .ins()
                    .icmp_imm_u(IntCC::NotEqual, p, i64::from(tk.pure));
                self.g.fault_if(bad, fault::PURE_BITS);
            }
        }
        v
    }

    /// Table lookup: `(code, general, opaque)` — the output bits (I32) and conditions (I8) for
    /// "not proven: general path" and "opaque" (per-vector actions).
    fn lookup(&mut self, tab: Tab, v: Value) -> (Value, Value, Value) {
        match tab {
            Tab::Imm(t) => {
                let v64 = self.g.b.ins().uextend(I64, v);
                let sh = self.g.b.ins().iadd(v64, v64);
                let tv = self.i64c(t);
                let s = self.g.b.ins().ushr(tv, sh);
                let c = self.g.b.ins().band_imm_s(s, 3);
                let c = self.g.b.ins().ireduce(I32, c);
                let general = self.g.b.ins().icmp_imm_u(IntCC::Equal, c, 2);
                let opq = self.g.b.ins().icmp_imm_u(IntCC::Equal, c, 3);
                (c, general, opq)
            }
            Tab::Packed2(o) => {
                let blob = self.blob.expect("blob");
                let v64 = self.g.b.ins().uextend(I64, v);
                let word = self.g.b.ins().ushr_imm_u(v64, 5);
                let byte = self.g.b.ins().ishl_imm_u(word, 3);
                let a = self.g.b.ins().iadd(blob, byte);
                let w = self.g.b.ins().load(I64, mf(), a, o as i32);
                let sh = self.g.b.ins().iadd(v64, v64);
                let s = self.g.b.ins().ushr(w, sh);
                let c = self.g.b.ins().band_imm_s(s, 3);
                let c = self.g.b.ins().ireduce(I32, c);
                let general = self.g.b.ins().icmp_imm_u(IntCC::Equal, c, 2);
                let opq = self.g.b.ins().icmp_imm_u(IntCC::Equal, c, 3);
                (c, general, opq)
            }
            Tab::Bytes(o) => {
                let blob = self.blob.expect("blob");
                let v64 = self.g.b.ins().uextend(I64, v);
                let a = self.g.b.ins().iadd(blob, v64);
                let c = self.g.b.ins().uload8(I32, mf(), a, o as i32);
                let general = self.g.b.ins().icmp_imm_u(IntCC::Equal, c, 0xFF);
                let opq = self.g.b.ins().icmp_imm_u(IntCC::Equal, c, 0xFE);
                (c, general, opq)
            }
            Tab::Bit(i, inv) => {
                let s = self.g.b.ins().ushr_imm_u(v, i64::from(i));
                let c = self.g.b.ins().band_imm_s(s, 1);
                let c = if inv {
                    self.g.b.ins().bxor_imm_s(c, 1)
                } else {
                    c
                };
                let f = self.g.b.ins().iconst(cranelift_codegen::ir::types::I8, 0);
                (c, f, f)
            }
            Tab::Words(o) => {
                let blob = self.blob.expect("blob");
                let v64 = self.g.b.ins().uextend(I64, v);
                let byte = self.g.b.ins().ishl_imm_u(v64, 2);
                let a = self.g.b.ins().iadd(blob, byte);
                let c = self.g.b.ins().load(I32, mf(), a, o as i32);
                let general = self.g.b.ins().icmp_imm_u(IntCC::Equal, c, i64::from(NONE));
                let opq = self
                    .g
                    .b
                    .ins()
                    .icmp_imm_u(IntCC::Equal, c, i64::from(VACT_OPAQUE));
                (c, general, opq)
            }
        }
    }

    /// Queue `(k, ob)` if `ob` differs from `kob[k]`.
    fn queue_if_changed(&mut self, k: u32, ob: Value) {
        let kob = self.k32(off::E_KOB, true, k);
        let ne = self.g.b.ins().icmp(IntCC::NotEqual, ob, kob);
        let add = self.g.b.create_block();
        let cont = self.g.b.create_block();
        self.g.b.ins().brif(ne, add, &[], cont, &[]);
        self.g.b.switch_to_block(add);
        let n = self.g.b.use_var(self.npend);
        if self.spec.paranoid.is_some() {
            let full = self.g.b.ins().icmp_imm_u(
                IntCC::UnsignedGreaterThanOrEqual,
                n,
                i64::from(self.spec.num_kernels),
            );
            self.g.fault_if(full, fault::QUEUE);
        }
        let pend = self.eptr(off::E_PENDING);
        let byte = self.g.b.ins().ishl_imm_u(n, 3);
        let a = self.g.b.ins().iadd(pend, byte);
        let kc = self.g.iconst(i64::from(k));
        self.g.b.ins().store(mf(), kc, a, 0);
        self.g.b.ins().store(mf(), ob, a, 4);
        let n1 = self.g.b.ins().iadd_imm_s(n, 1);
        self.g.b.def_var(self.npend, n1);
        self.g.b.ins().jump(cont, &[]);
        self.g.b.switch_to_block(cont);
    }

    /// Loop over the set bits of `mask`, running `body(self, list index, bit)` for each listed
    /// `(bit, list index)` (dispatch on the bit index; unlisted bits bail out).
    fn bit_loop(
        &mut self,
        mask: Value,
        kernels: &[(u32, usize)],
        mut body: impl FnMut(&mut Self, usize, u32),
    ) {
        let m = self.g.b.declare_var(I64);
        self.g.b.def_var(m, mask);
        let header = self.g.b.create_block();
        let dispatch = self.g.b.create_block();
        let exit = self.g.b.create_block();
        self.g.b.ins().jump(header, &[]);
        self.g.b.switch_to_block(header);
        let mv = self.g.b.use_var(m);
        self.g.b.ins().brif(mv, dispatch, &[], exit, &[]);
        self.g.b.switch_to_block(dispatch);
        let mv = self.g.b.use_var(m);
        let m1 = self.g.b.ins().iadd_imm_s(mv, -1);
        let rest = self.g.b.ins().band(mv, m1);
        self.g.b.def_var(m, rest);
        if let [(bit, i)] = *kernels {
            // A single candidate: no dispatch.
            body(self, i, bit);
            self.g.b.ins().jump(header, &[]);
        } else {
            let bitpos = self.g.b.ins().ctz(mv);
            let bitpos = self.g.b.ins().ireduce(I32, bitpos);
            let mut targets = vec![self.bail; 64];
            let mut blocks = Vec::with_capacity(kernels.len());
            for &(bit, i) in kernels {
                let blk = self.g.b.create_block();
                targets[bit as usize] = blk;
                blocks.push((blk, i, bit));
            }
            let calls: Vec<_> = targets
                .iter()
                .map(|&t| self.g.b.func.dfg.block_call(t, &[]))
                .collect();
            let def = self.g.b.func.dfg.block_call(self.bail, &[]);
            let jt = self.g.b.create_jump_table(JumpTableData::new(def, &calls));
            self.g.b.ins().br_table(bitpos, jt);
            for (blk, i, bit) in blocks {
                self.g.b.switch_to_block(blk);
                body(self, i, bit);
                self.g.b.ins().jump(header, &[]);
            }
        }
        self.g.b.switch_to_block(exit);
    }

    /// Sweep the dirty partitions in order (one guarded call each).
    fn sweep(&mut self) {
        let dirty = self.g.ctx_ptr(off::DIRTY);
        let parts = self.eptr(off::E_PARTS);
        let mut call_sig = cranelift_codegen::ir::Signature::new(self.g.b.func.signature.call_conv);
        call_sig.params.push(AbiParam::new(I64));
        call_sig.params.push(AbiParam::new(I32));
        let sig = self.g.b.import_signature(call_sig);
        for p in 0..self.spec.num_partitions {
            let wo = ((p / 64) * 8) as i32;
            let bit = 1u64 << (p % 64);
            let d = self.g.b.ins().load(I64, mf(), dirty, wo);
            let t = self.g.b.ins().band_imm_s(d, bit as i64);
            let call = self.g.b.create_block();
            let next = self.g.b.create_block();
            self.g.b.ins().brif(t, call, &[], next, &[]);
            self.g.b.switch_to_block(call);
            let d2 = self.g.b.ins().band_imm_s(d, !bit as i64);
            self.g.b.ins().store(mf(), d2, dirty, wo);
            let f = self.g.b.ins().load(I64, mf(), parts, (p * 8) as i32);
            let pc = self.g.iconst(i64::from(p));
            let ctx = self.g.ctx;
            self.g.b.ins().call_indirect(sig, f, &[ctx, pc]);
            self.g.b.ins().jump(next, &[]);
            self.g.b.switch_to_block(next);
        }
    }

    /// `Marks::next_epoch`: snapshot the tracked kernels' input bits, advance the epoch.
    fn next_epoch(&mut self) {
        if !self.spec.tracked_list.is_empty() {
            let kin = self.g.ctx_ptr(off::KIN);
            let prev = self.eptr(off::E_KIN_PREV);
            for &k in &self.spec.tracked_list {
                let x = self.g.b.ins().load(I32, mf(), kin, (k * 4) as i32);
                self.g.b.ins().store(mf(), x, prev, (k * 4) as i32);
            }
        }
        let ep_p = self.eptr(off::E_EPOCH);
        let ep = self.g.b.ins().load(I32, mf(), ep_p, 0);
        let ep1 = self.g.b.ins().iadd_imm_s(ep, 1);
        self.g.b.ins().store(mf(), ep1, ep_p, 0);
        let ctx = self.g.ctx;
        self.g.b.ins().store(mf(), ep1, ctx, off::EPOCH);
        let wrapped = self
            .g
            .b
            .ins()
            .icmp_imm_u(IntCC::Equal, ep1, i64::from(u32::MAX));
        let wblk = self.g.b.create_block();
        let after = self.g.b.create_block();
        self.g.b.ins().brif(wrapped, wblk, &[], after, &[]);
        self.g.b.switch_to_block(wblk);
        self.g.b.set_cold_block(wblk);
        let f = self.eptr(off::E_EPOCH_WRAP);
        let wrap = self.wrap;
        self.g.b.ins().call_indirect(wrap, f, &[ctx]);
        let ep_now = self.g.b.ins().load(I32, mf(), ep_p, 0);
        self.g.b.ins().store(mf(), ep_now, ctx, off::EPOCH);
        self.g.b.ins().jump(after, &[]);
        self.g.b.switch_to_block(after);
    }

    /// `ectx[o] += x` (a 64-bit counter).
    fn add_counter(&mut self, o: i32, x: Value) {
        let c = self.g.b.ins().load(I64, mf(), self.ectx, o);
        let n = self.g.b.ins().iadd(c, x);
        self.g.b.ins().store(mf(), n, self.ectx, o);
    }

    /// Return `code` (the counters are kept in the context).
    fn ret(&mut self, code: u32) {
        let c = self.g.iconst(i64::from(code));
        self.g.b.ins().return_(&[c]);
    }

    fn emit(&mut self) {
        let spec = self.spec;
        let kw = spec.words.len();
        let done = self.g.b.create_block();

        // The timed-source changes of a scheduled event, then the initial sweep and the round
        // loop.
        let tm = self.g.b.ins().load(I32, mf(), self.ectx, off::E_TIMED);
        for (i, (net, rec)) in spec.timed.iter().enumerate() {
            let bit = self.g.b.ins().band_imm_s(tm, 1i64 << i);
            let wr = self.g.b.create_block();
            let next = self.g.b.create_block();
            self.g.b.ins().brif(bit, wr, &[], next, &[]);
            self.g.b.switch_to_block(wr);
            let v = self.g.iconst(i64::from((spec.tuple >> i) & 1));
            let out = Out {
                net: *net,
                notify: rec.clone(),
                store: true,
            };
            self.g.out(&out, v);
            self.g.b.ins().jump(next, &[]);
            self.g.b.switch_to_block(next);
        }
        self.sweep();
        let decide = self.g.b.create_block();
        self.g.b.ins().jump(decide, &[]);
        self.g.b.switch_to_block(decide);

        // ---- decide: pending kernels, none explicit ----
        let kp_p = self.g.ctx_ptr(off::KPENDING);
        let uns_p = self.eptr(off::E_UNSYNCED);
        let mut any = self.i64c(0);
        let mut todos = Vec::with_capacity(kw);
        for w in 0..kw {
            let wo = (w * 8) as i32;
            let kp = self.g.b.ins().load(I64, mf(), kp_p, wo);
            let un = self.g.b.ins().load(I64, mf(), uns_p, wo);
            let special = self.g.b.ins().band(kp, un);
            self.bail_if(special);
            any = self.g.b.ins().bor(any, kp);
            todos.push(kp);
        }
        let round = self.g.b.create_block();
        let idle = self.g.b.create_block();
        self.g.b.ins().brif(any, round, &[], idle, &[]);
        self.g.b.switch_to_block(idle);
        self.next_epoch();
        self.g.b.ins().jump(done, &[]);

        self.g.b.switch_to_block(round);
        let rounds = self.g.b.ins().load(I64, mf(), self.ectx, off::E_ROUNDS);
        let lim = self
            .g
            .b
            .ins()
            .load(I32, mf(), self.ectx, off::E_DELTA_LIMIT);
        let lim = self.g.b.ins().uextend(I64, lim);
        let over = self
            .g
            .b
            .ins()
            .icmp(IntCC::UnsignedGreaterThanOrEqual, rounds, lim);
        self.bail_if(over);
        let z = self.i64c(0);
        self.g.b.def_var(self.npend, z);
        let res = self.decide(&todos);

        // ---- commit ----
        self.commit(&res);
        let r1 = self.g.b.ins().iadd_imm_s(rounds, 1);
        self.g.b.ins().store(mf(), r1, self.ectx, off::E_ROUNDS);

        // ---- round boundary, writes ----
        self.next_epoch();
        if spec.fused {
            self.fused_writes(&res);
        } else {
            self.writes();
        }

        // ---- anything left? ----
        let kp_p = self.g.ctx_ptr(off::KPENDING);
        let dirty_p = self.g.ctx_ptr(off::DIRTY);
        let mut left = self.i64c(0);
        for w in 0..kw {
            let x = self.g.b.ins().load(I64, mf(), kp_p, (w * 8) as i32);
            left = self.g.b.ins().bor(left, x);
        }
        for w in 0..spec.num_partitions.div_ceil(64) as usize {
            let x = self.g.b.ins().load(I64, mf(), dirty_p, (w * 8) as i32);
            left = self.g.b.ins().bor(left, x);
        }
        let more = self.g.b.create_block();
        let quiet = self.g.b.create_block();
        self.g.b.ins().brif(left, more, &[], quiet, &[]);
        self.g.b.switch_to_block(quiet);
        self.next_epoch();
        self.g.b.ins().jump(done, &[]);
        self.g.b.switch_to_block(more);
        self.sweep();
        self.g.b.ins().jump(decide, &[]);

        self.g.b.switch_to_block(done);
        self.ret(EVENT_DONE);
        self.g.b.switch_to_block(self.bail);
        self.ret(EVENT_BAIL);
    }

    /// Pass 1 of a round boundary (no side effects except the scratch queue).
    fn decide(&mut self, todos: &[Value]) -> Vec<WordRes> {
        let spec = self.spec;
        let t = spec.tuple as usize;
        let kw = spec.words.len();
        let mut res = Vec::with_capacity(kw);
        let mut ti = 0usize;
        let mut vi = 0usize;
        for (w, ws) in spec.words.iter().enumerate() {
            let wo = (w * 8) as i32;
            let todo = todos[w];
            let fo_p = self.g.ctx_ptr(off::FASTOK);
            let fo = self.g.b.ins().load(I64, mf(), fo_p, wo);
            let ok = self.g.b.ins().band(todo, fo);
            // Kernels with two or more flipped inputs among the tracked checked / copaque ones.
            let mt = (ws.copaque | ws.check) & ws.tracked;
            let multi = if mt != 0 {
                let mv = self.g.b.declare_var(I64);
                let z = self.i64c(0);
                self.g.b.def_var(mv, z);
                let sel = self.g.b.ins().band_imm_s(ok, mt as i64);
                let list: Vec<(u32, usize)> = (0..64)
                    .filter(|b| mt >> b & 1 == 1)
                    .map(|b| (b, b as usize))
                    .collect();
                self.bit_loop(sel, &list, |s, _, bit| {
                    let k = (w * 64) as u32 + bit;
                    let a = s.k32(off::KIN, false, k);
                    let p = s.k32(off::E_KIN_PREV, true, k);
                    let fl = s.g.b.ins().bxor(a, p);
                    let f1 = s.g.b.ins().iadd_imm_s(fl, -1);
                    let x = s.g.b.ins().band(fl, f1);
                    let nz = s.g.b.ins().icmp_imm_u(IntCC::NotEqual, x, 0);
                    let nz = s.g.b.ins().uextend(I64, nz);
                    let sh = s.g.b.ins().ishl_imm_u(nz, i64::from(bit));
                    let cur = s.g.b.use_var(mv);
                    let n = s.g.b.ins().bor(cur, sh);
                    s.g.b.def_var(mv, n);
                });
                self.g.b.use_var(mv)
            } else {
                self.i64c(0)
            };
            // skip = ok & (SKIP | (COPAQUE & !multi))
            let skip = if ws.copaque != 0 {
                let copaque = self.i64c(ws.copaque);
                let cop = self.g.b.ins().band_not(copaque, multi);
                let sk = self.g.b.ins().bor_imm_s(cop, ws.skip as i64);
                self.g.b.ins().band(ok, sk)
            } else {
                self.g.b.ins().band_imm_s(ok, ws.skip as i64)
            };
            // fast = ok & FAST & !skip
            let f = self.g.b.ins().band_imm_s(ok, ws.fast as i64);
            let fast = self.g.b.ins().band_not(f, skip);
            // same = fast & tlast[t] & !ddirty & !checked, checked = fast & CHECK & multi
            let tl_p = self.eptr(off::E_TLAST);
            let tl = self
                .g
                .b
                .ins()
                .load(I64, mf(), tl_p, ((t * kw + w) * 8) as i32);
            let dd_p = self.g.ctx_ptr(off::DDIRTY);
            let dd = self.g.b.ins().load(I64, mf(), dd_p, wo);
            let s1 = self.g.b.ins().band(fast, tl);
            let mut same = self.g.b.ins().band_not(s1, dd);
            let mut checked = None;
            if ws.check != 0 {
                let c = self.g.b.ins().band_imm_s(fast, ws.check as i64);
                let c = self.g.b.ins().band(c, multi);
                same = self.g.b.ins().band_not(same, c);
                checked = Some(c);
            }
            let eval = self.g.b.ins().band_not(fast, same);
            if let Some(c) = checked {
                // An evaluated checked kernel needs FastRef's order check: the host's path.
                let ce = self.g.b.ins().band(c, eval);
                self.bail_if(ce);
            }
            let opq = self.g.b.ins().band_imm_s(eval, !ws.trans as i64);
            let trans_f = self.g.b.ins().band_imm_s(eval, ws.trans as i64);
            if spec.fused {
                // Looked up and written after the commit.
                ti += ws.trans.count_ones() as usize;
            } else if ws.trans != 0 {
                let list: Vec<(u32, usize)> = (0..64u32)
                    .filter(|b| ws.trans >> b & 1 == 1)
                    .map(|b| {
                        let i = ti;
                        ti += 1;
                        (b, i)
                    })
                    .collect();
                self.bit_loop(trans_f, &list, |s, i, _| {
                    let tk = &s.spec.trans[i];
                    let kin = s.k32(off::KIN, false, tk.k);
                    let v = s.vector(tk, kin);
                    let (e, general, _) = s.lookup(tk.tab, v);
                    s.bail_if(general);
                    s.queue_if_changed(tk.k, e);
                });
            }
            // Per-vector kernels.
            let z = self.i64c(0);
            let (g, vdd, vopq, vtrans, vevals) = if ws.vg != 0 {
                let ng = self.g.b.ins().band_imm_s(ok, ws.vg as i64);
                let g = self.g.b.ins().band_not(ng, skip);
                let vars: Vec<Variable> = (0..4).map(|_| self.g.b.declare_var(I64)).collect();
                for &v in &vars {
                    self.g.b.def_var(v, z);
                }
                let (vdd, vopq, vtrans, vevals) = (vars[0], vars[1], vars[2], vars[3]);
                let list: Vec<(u32, usize)> = (0..64u32)
                    .filter(|b| ws.vg >> b & 1 == 1)
                    .map(|b| {
                        let i = vi;
                        vi += 1;
                        (b, i)
                    })
                    .collect();
                self.bit_loop(g, &list, |s, i, bit| {
                    let vk = &s.spec.vk[i];
                    let k = vk.k;
                    let a = s.k32(off::KIN, false, k);
                    let p = s.k32(off::E_KIN_PREV, true, k);
                    let fl = s.g.b.ins().bxor(a, p);
                    let pc = s.g.b.ins().popcnt(fl);
                    let not1 = s.g.b.ins().icmp_imm_u(IntCC::NotEqual, pc, 1);
                    s.bail_if(not1);
                    let a = s.vector(vk, a);
                    let (act, general, is_opq) = s.lookup(vk.tab, a);
                    s.bail_if(general);
                    let one = s.i64c(1);
                    s.add_var(vevals, one);
                    let ob = s.g.b.create_block();
                    let tb = s.g.b.create_block();
                    let fin = s.g.b.create_block();
                    s.g.b.ins().brif(is_opq, ob, &[], tb, &[]);
                    s.g.b.switch_to_block(ob);
                    s.add_var(vopq, one);
                    s.g.b.ins().jump(fin, &[]);
                    s.g.b.switch_to_block(tb);
                    s.add_var(vtrans, one);
                    let x = s.g.b.use_var(vdd);
                    let x = s.g.b.ins().bor_imm_s(x, (1u64 << bit) as i64);
                    s.g.b.def_var(vdd, x);
                    s.queue_if_changed(k, act);
                    s.g.b.ins().jump(fin, &[]);
                    s.g.b.switch_to_block(fin);
                });
                (
                    g,
                    self.g.b.use_var(vdd),
                    self.g.b.use_var(vopq),
                    self.g.b.use_var(vtrans),
                    self.g.b.use_var(vevals),
                )
            } else {
                (z, z, z, z, z)
            };
            // Every pending kernel must be handled.
            let h1 = self.g.b.ins().bor(skip, fast);
            let h = self.g.b.ins().bor(h1, g);
            let rest = self.g.b.ins().band_not(todo, h);
            self.bail_if(rest);
            res.push(WordRes {
                todo,
                skip,
                same,
                eval,
                opq,
                trans_f,
                vdd,
                vopq,
                vtrans,
                vevals,
            });
        }
        res
    }

    /// Pass 2: `fast_round`'s effects and counts; the pending kernels are taken.
    fn commit(&mut self, res: &[WordRes]) {
        let spec = self.spec;
        let t = spec.tuple as usize;
        let kw = spec.words.len();
        let mut c_opq = self.i64c(0);
        let mut c_trans = self.i64c(0);
        let mut c_evals = self.i64c(0);
        for (w, (ws, r)) in spec.words.iter().zip(res).enumerate() {
            let wo = (w * 8) as i32;
            let transp = self.g.b.ins().band_not(r.eval, r.opq);
            if ws.trans != 0 || ws.vg != 0 {
                if ws.trans != 0 {
                    let tl_p = self.eptr(off::E_TLAST);
                    for tt in 0..spec.num_tuples as usize {
                        let o = ((tt * kw + w) * 8) as i32;
                        let x = self.g.b.ins().load(I64, mf(), tl_p, o);
                        let x = self.g.b.ins().band_not(x, transp);
                        let x = if tt == t {
                            self.g.b.ins().bor(x, transp)
                        } else {
                            x
                        };
                        self.g.b.ins().store(mf(), x, tl_p, o);
                    }
                }
                let dd_p = self.g.ctx_ptr(off::DDIRTY);
                let dd = self.g.b.ins().load(I64, mf(), dd_p, wo);
                let dd = self.g.b.ins().band_not(dd, transp);
                let dd = self.g.b.ins().bor(dd, r.vdd);
                self.g.b.ins().store(mf(), dd, dd_p, wo);
            }
            let so = self.g.b.ins().bor(r.skip, r.opq);
            let n = self.g.b.ins().popcnt(so);
            c_opq = self.g.b.ins().iadd(c_opq, n);
            c_opq = self.g.b.ins().iadd(c_opq, r.vopq);
            let st = self.g.b.ins().bor(r.same, transp);
            let n = self.g.b.ins().popcnt(st);
            c_trans = self.g.b.ins().iadd(c_trans, n);
            c_trans = self.g.b.ins().iadd(c_trans, r.vtrans);
            let n = self.g.b.ins().popcnt(r.eval);
            c_evals = self.g.b.ins().iadd(c_evals, n);
            c_evals = self.g.b.ins().iadd(c_evals, r.vevals);
            // Every pending kernel was handled: all become lazy; the pending set is taken.
            let lz_p = self.eptr(off::E_LAZY);
            let lz = self.g.b.ins().load(I64, mf(), lz_p, wo);
            let lz = self.g.b.ins().bor(lz, r.todo);
            self.g.b.ins().store(mf(), lz, lz_p, wo);
            let kp_p = self.g.ctx_ptr(off::KPENDING);
            let z = self.i64c(0);
            self.g.b.ins().store(mf(), z, kp_p, wo);
        }
        self.add_counter(off::E_OPAQUE, c_opq);
        self.add_counter(off::E_TRANSPARENT, c_trans);
        self.add_counter(off::E_EVALS, c_evals);
    }

    /// Fused mode: look up every evaluated transparent kernel and write its outputs if they
    /// changed (exact: no evaluated kernel reads another's output, so every lookup sees the
    /// round's boundary values).
    fn fused_writes(&mut self, res: &[WordRes]) {
        let spec = self.spec;
        let mut ti = 0usize;
        for (ws, r) in spec.words.iter().zip(res) {
            if ws.trans == 0 {
                continue;
            }
            let list: Vec<(u32, usize)> = (0..64u32)
                .filter(|b| ws.trans >> b & 1 == 1)
                .map(|b| {
                    let i = ti;
                    ti += 1;
                    (b, i)
                })
                .collect();
            self.bit_loop(r.trans_f, &list, |s, i, _| {
                let tk = &s.spec.trans[i];
                let kin = s.k32(off::KIN, false, tk.k);
                let v = s.vector(tk, kin);
                let (e, general, _) = s.lookup(tk.tab, v);
                if s.spec.paranoid.is_some() {
                    // Fused only when every entry the kernel can see is proven.
                    s.g.fault_if(general, fault::UNPROVEN);
                }
                let kob = s.k32(off::E_KOB, true, tk.k);
                let ne = s.g.b.ins().icmp(IntCC::NotEqual, e, kob);
                let wb = s.g.b.create_block();
                let cont = s.g.b.create_block();
                s.g.b.ins().brif(ne, wb, &[], cont, &[]);
                s.g.b.switch_to_block(wb);
                debug_assert_eq!(s.spec.writes[i].k, tk.k);
                s.write_kernel(i, e);
                s.g.b.ins().jump(cont, &[]);
                s.g.b.switch_to_block(cont);
            });
        }
    }

    /// Write kernel `writes[wi]`'s outputs from output bits `ob` (and `kob`).
    fn write_kernel(&mut self, wi: usize, ob: Value) {
        let wk = &self.spec.writes[wi];
        let kob = self.eptr(off::E_KOB);
        self.g.b.ins().store(mf(), ob, kob, (wk.k * 4) as i32);
        for (j, (net, rec)) in wk.outs.iter().enumerate() {
            let v = if j == 0 {
                ob
            } else {
                self.g.b.ins().ushr_imm_u(ob, j as i64)
            };
            let v = if wk.outs.len() == 1 {
                v
            } else {
                self.g.b.ins().band_imm_s(v, 1)
            };
            let out = Out {
                net: *net,
                notify: rec.clone(),
                store: true,
            };
            self.g.out(&out, v);
        }
    }

    /// Write the queued outputs (`write_fast`).
    fn writes(&mut self) {
        let spec = self.spec;
        if spec.writes.is_empty() {
            return;
        }
        let i = self.g.b.declare_var(I64);
        let z = self.i64c(0);
        self.g.b.def_var(i, z);
        let header = self.g.b.create_block();
        let body = self.g.b.create_block();
        let exit = self.g.b.create_block();
        let next = self.g.b.create_block();
        let bad = self.g.b.create_block();
        self.g.b.ins().jump(header, &[]);
        self.g.b.switch_to_block(header);
        let iv = self.g.b.use_var(i);
        let n = self.g.b.use_var(self.npend);
        let more = self.g.b.ins().icmp(IntCC::UnsignedLessThan, iv, n);
        self.g.b.ins().brif(more, body, &[], exit, &[]);
        self.g.b.switch_to_block(body);
        let pend = self.eptr(off::E_PENDING);
        let byte = self.g.b.ins().ishl_imm_u(iv, 3);
        let a = self.g.b.ins().iadd(pend, byte);
        let k = self.g.b.ins().load(I32, mf(), a, 0);
        let ob = self.g.b.ins().load(I32, mf(), a, 4);
        let mut targets = vec![bad; spec.num_kernels as usize];
        let mut blocks = Vec::new();
        for (wi, wk) in spec.writes.iter().enumerate() {
            let blk = self.g.b.create_block();
            targets[wk.k as usize] = blk;
            blocks.push((blk, wi));
        }
        // Unlisted kernels are never queued: `bad` returns an error code.
        let def = self.g.b.func.dfg.block_call(bad, &[]);
        let calls: Vec<_> = targets
            .iter()
            .map(|&t| self.g.b.func.dfg.block_call(t, &[]))
            .collect();
        let jt = self.g.b.create_jump_table(JumpTableData::new(def, &calls));
        self.g.b.ins().br_table(k, jt);
        for (blk, wi) in blocks {
            self.g.b.switch_to_block(blk);
            self.write_kernel(wi, ob);
            self.g.b.ins().jump(next, &[]);
        }
        self.g.b.switch_to_block(bad);
        self.g.b.set_cold_block(bad);
        if self.spec.paranoid.is_some() {
            let one = self.g.iconst(1);
            self.g.fault_if(one, fault::WRITE_KERNEL);
        }
        let two = self.g.iconst(2);
        self.g.b.ins().return_(&[two]);
        self.g.b.switch_to_block(next);
        let iv = self.g.b.use_var(i);
        let i1 = self.g.b.ins().iadd_imm_s(iv, 1);
        self.g.b.def_var(i, i1);
        self.g.b.ins().jump(header, &[]);
        self.g.b.switch_to_block(exit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode entry `v` of an encoded table back to the raw program value.
    fn decode(enc: &Encoder, tab: Tab, v: usize) -> u32 {
        let code = |c: u64| match c {
            2 => NONE,
            3 => VACT_OPAQUE,
            c => c as u32,
        };
        match tab {
            Tab::Imm(t) => code((t >> (2 * v)) & 3),
            Tab::Packed2(o) => {
                let a = o as usize + v / 32 * 8;
                let w = u64::from_ne_bytes(enc.blob[a..a + 8].try_into().unwrap());
                code((w >> (2 * (v % 32))) & 3)
            }
            Tab::Bytes(o) => match enc.blob[o as usize + v] {
                0xFF => NONE,
                0xFE => VACT_OPAQUE,
                b => u32::from(b),
            },
            Tab::Words(o) => {
                let a = o as usize + 4 * v;
                u32::from_ne_bytes(enc.blob[a..a + 4].try_into().unwrap())
            }
            Tab::Bit(i, inv) => ((v as u32 >> i) & 1) ^ u32::from(inv),
        }
    }

    #[test]
    fn copy_bits_are_detected_under_pure_bits() {
        // 3 inputs; bit 2 pure (= 1 in this tuple): Q = input 0 whenever bit 2 is set.
        let entries: Vec<u32> = (0..8u32)
            .map(|v| if v & 4 != 0 { v & 1 } else { NONE })
            .collect();
        assert_eq!(copy_bit(&entries, 4, 4, 0b011), Some(Tab::Bit(0, false)));
        // Not tracked in the input index: no.
        assert_eq!(copy_bit(&entries, 4, 4, 0b010), None);
        // Inverted.
        let inv: Vec<u32> = entries
            .iter()
            .map(|&e| if e == NONE { e } else { e ^ 1 })
            .collect();
        assert_eq!(copy_bit(&inv, 4, 4, 0b011), Some(Tab::Bit(0, true)));
        // A function of two inputs: no.
        let and: Vec<u32> = (0..8u32).map(|v| v & (v >> 1) & 1).collect();
        assert_eq!(copy_bit(&and, 0, 0, 0b111), None);
    }

    #[test]
    fn tables_roundtrip_and_dedup() {
        let mut enc = Encoder::default();
        let tables: Vec<Vec<u32>> = vec![
            vec![0, 1, NONE, 1],
            vec![1, VACT_OPAQUE, 0, NONE, 1, 1, 0, 0],
            (0..64).map(|i| [0, 1, NONE][i % 3]).collect(),
            (0..16).map(|i| i * 7 % 200).collect(),
            vec![5, NONE, VACT_OPAQUE, 9],
            vec![0x1_0000, 3, NONE, 1 << 30],
        ];
        let tabs: Vec<Tab> = tables.iter().map(|t| enc.encode(t)).collect();
        assert!(matches!(tabs[0], Tab::Imm(_)));
        assert!(matches!(tabs[1], Tab::Imm(_)));
        assert!(matches!(tabs[2], Tab::Packed2(_)));
        assert!(matches!(tabs[3], Tab::Bytes(_)));
        assert!(matches!(tabs[5], Tab::Words(_)));
        for (t, tab) in tables.iter().zip(&tabs) {
            for (v, &e) in t.iter().enumerate() {
                assert_eq!(decode(&enc, *tab, v), e, "{t:?}[{v}]");
            }
        }
        let len = enc.blob.len();
        assert_eq!(enc.encode(&tables[2]), tabs[2]);
        assert_eq!(enc.encode(&tables[5]), tabs[5]);
        assert_eq!(enc.blob.len(), len, "identical tables are stored once");
    }
}
