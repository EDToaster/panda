//! Cranelift IR generation for [`PartSpec`]s (and the module / code-memory plumbing).
//!
//! A partition becomes one function `extern "C" fn(*mut NativeCtx, u32)`
//! ([`panda_fast::native::PartitionFn`]):
//!
//! * every op is straight-line code over the byte-per-net state, in a register-friendly order
//!   ([`schedule`]): 1/2-input gates are a shift of their packed 2-bit-code table by the operand
//!   codes (`(T >> 2 * (a | b << 2)) & 3`, computed on doubled codes as
//!   `((T << 1) >> (2a' + 8b')) & 6` so the index is one `lea`), N-input gates the interpreter's
//!   branchless seen/parity fold; values produced or loaded shortly before stay in registers,
//!   and outputs of fused-away gates nobody reads from memory are not stored at all;
//! * an output with readers is compared with its old value, and a change branches to an
//!   out-of-line block that applies its decoded change-propagation record directly (OR the
//!   reader-partition masks into the dirty set, the kernel masks into the pending / data-dirty
//!   sets, clear fast-ok bits for a non-binary value, log the boundary value, flip kernel
//!   input-index bits) — the effects of the interpreter's batched `Marks::flush`;
//! * a LUT computes its index and binary check inline, reads its table, writes its outputs only
//!   when the entry differs from the one they hold, and calls the host for the cold path;
//! * resolvers call the host.

use std::collections::HashMap;

use cranelift_codegen::Context;
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types::{I32, I64};
use cranelift_codegen::ir::{
    AbiParam, Function, InstBuilder, MemFlagsData, SigRef, Signature, TrapCode, UserFuncName, Value,
};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module, default_libcall_names};

use panda_fast::ir::{NONE, table};
use panda_fast::native::{NotifyRecord, ctx_offsets as off};

use crate::spec::{JOp, LutSpec, Out, PartSpec, Shape};

/// Owner of JIT code memory: frees it when the last function referencing it is dropped.
pub(crate) struct CodeMem(Option<JITModule>);

// SAFETY: after `finalize_definitions` the module is never used again except to free its
// memory in `drop` (which needs exclusive ownership); the code it holds is immutable, so
// sharing or moving the owner across threads is sound.
unsafe impl Send for CodeMem {}
// SAFETY: see above (no `&self` method touches the module).
unsafe impl Sync for CodeMem {}

impl Drop for CodeMem {
    fn drop(&mut self) {
        if let Some(m) = self.0.take() {
            // SAFETY: `CodeMem` is only dropped when no `NativeCode` / cache entry holding a
            // function pointer into it (each holds an `Arc<CodeMem>`) is alive.
            unsafe { m.free_memory() };
        }
    }
}

/// Runtime-check codes of paranoid native code (the [`panda_fast::native::FaultFn`] argument).
pub(crate) mod fault {
    /// The native context's array lengths differ from the compiled program's.
    pub const SHAPE: u32 = 1;
    /// The event context's lengths differ from the compiled program's.
    pub const EVENT_SHAPE: u32 = 2;
    /// A kernel table index beyond the table.
    pub const TABLE_INDEX: u32 = 3;
    /// A kernel input vector that disagrees with the clock tuple's pure input bits.
    pub const PURE_BITS: u32 = 4;
    /// A fused event function met an unproven table entry.
    pub const UNPROVEN: u32 = 5;
    /// More queued kernel writes than kernels.
    pub const QUEUE: u32 = 6;
    /// A queued write of a kernel without outputs to write.
    pub const WRITE_KERNEL: u32 = 7;
}

/// One compilation job: a fresh JIT module holding the functions compiled in it.
pub(crate) struct Compiler {
    pub(crate) module: JITModule,
    pub(crate) ctx: Context,
    pub(crate) fctx: FunctionBuilderContext,
    funcs: Vec<FuncId>,
    pub(crate) call_conv: CallConv,
    /// Machine-code bytes emitted.
    pub code_bytes: usize,
}

impl Compiler {
    /// A compiler for the host (optimizing for speed).
    pub fn new() -> Result<Compiler, String> {
        let mut fb = settings::builder();
        let set = |fb: &mut settings::Builder, k: &str, v: &str| {
            fb.set(k, v)
                .map_err(|e| format!("cranelift flag {k}={v}: {e}"))
        };
        set(&mut fb, "opt_level", "speed")?;
        set(&mut fb, "use_colocated_libcalls", "false")?;
        set(&mut fb, "is_pic", "false")?;
        set(
            &mut fb,
            "enable_verifier",
            if cfg!(debug_assertions) {
                "true"
            } else {
                "false"
            },
        )?;
        let isa = cranelift_native::builder()
            .map_err(|e| format!("host not supported by cranelift: {e}"))?
            .finish(settings::Flags::new(fb))
            .map_err(|e| format!("cranelift isa: {e}"))?;
        let call_conv = isa.default_call_conv();
        let module = JITModule::new(JITBuilder::with_isa(isa, default_libcall_names()));
        let ctx = module.make_context();
        Ok(Compiler {
            module,
            ctx,
            fctx: FunctionBuilderContext::new(),
            funcs: Vec::new(),
            call_conv,
            code_bytes: 0,
        })
    }

    pub(crate) fn ptr_sig(&self, extra: &[cranelift_codegen::ir::Type]) -> Signature {
        let mut sig = Signature::new(self.call_conv);
        sig.params.push(AbiParam::new(I64));
        for &t in extra {
            sig.params.push(AbiParam::new(t));
        }
        sig
    }

    /// Number of functions defined so far.
    pub(crate) fn num_funcs(&self) -> usize {
        self.funcs.len()
    }

    /// Compile one partition; returns its index among this job's functions.
    pub fn add_partition(&mut self, spec: &PartSpec) -> Result<usize, String> {
        let sig = self.ptr_sig(&[I32]);
        let idx = self.funcs.len();
        let id = self
            .module
            .declare_function(&format!("part{idx}"), Linkage::Local, &sig)
            .map_err(|e| e.to_string())?;
        self.ctx.clear();
        self.ctx.func = Function::with_name_signature(UserFuncName::user(0, idx as u32), sig);
        let cb_sig = self.ptr_sig(&[I32]);
        {
            let mut b = FunctionBuilder::new(&mut self.ctx.func, &mut self.fctx);
            let cb = b.import_signature(cb_sig);
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            b.switch_to_block(entry);
            let ctx = b.block_params(entry)[0];
            let mut g = Gen::new(b, ctx, cb);
            if let Some(shape) = &spec.paranoid {
                g.check_shape(shape);
            }
            g.count(spec.ops.len() as u64);
            for i in schedule(spec) {
                g.op(&spec.ops[i]);
                g.pos += 1;
            }
            g.b.ins().return_(&[]);
            g.b.seal_all_blocks();
            g.b.finalize(self.module.target_config());
        }
        self.define(id)?;
        Ok(idx)
    }

    /// Define the function in `self.ctx` as `id`.
    pub(crate) fn define(&mut self, id: FuncId) -> Result<(), String> {
        let dump = std::env::var_os("PANDA_JIT_DUMP").map(std::path::PathBuf::from);
        if dump.is_some() {
            self.ctx.set_disasm(true);
        }
        let clif = dump.as_ref().map(|_| self.ctx.func.display().to_string());
        self.module
            .define_function(id, &mut self.ctx)
            .map_err(|e| format!("cranelift: {e:?}"))?;
        if let Some(cc) = self.ctx.compiled_code() {
            self.code_bytes += cc.code_info().total_size as usize;
            if let Some(dir) = &dump {
                let _ = std::fs::create_dir_all(dir);
                let name = format!("{}", self.ctx.func.name);
                let text = format!(
                    "{}\n;; ---- disassembly ----\n{}",
                    clif.unwrap_or_default(),
                    cc.vcode.clone().unwrap_or_default()
                );
                let _ = std::fs::write(dir.join(format!("{name}.txt")), text);
            }
        }
        self.funcs.push(id);
        Ok(())
    }

    /// Give up on this job (after a failed definition): free the memory of what was compiled.
    pub fn abandon(self) {
        // SAFETY: nothing was finalized, so no function pointer into the module exists.
        unsafe { self.module.free_memory() };
    }

    /// Link everything; returns the code owner and each function's address (in order).
    pub fn finish(mut self) -> Result<(CodeMem, Vec<*const u8>), String> {
        self.module
            .finalize_definitions()
            .map_err(|e| format!("cranelift: {e}"))?;
        let ptrs = self
            .funcs
            .iter()
            .map(|&f| self.module.get_finalized_function(f))
            .collect();
        Ok((CodeMem(Some(self.module)), ptrs))
    }
}

/// Memory flags of every access: the context and state arrays are valid, aligned and private
/// to the engine thread.
pub(crate) fn mf() -> MemFlagsData {
    MemFlagsData::trusted()
}

/// IR generator for one function over a [`panda_fast::native::NativeCtx`].
pub(crate) struct Gen<'a> {
    pub b: FunctionBuilder<'a>,
    pub ctx: Value,
    pub values: Value,
    /// Callback signature `(ctx, u32)`.
    pub cb: SigRef,
    /// Net values available in SSA registers (defined in the straight-line part), with the
    /// op position that produced or loaded them.
    cache: HashMap<u32, (Value, usize)>,
    /// Position of the op being generated.
    pub pos: usize,
    /// Nets whose value only exists in a register (store elided).
    pinned: std::collections::HashSet<u32>,
}

/// Ops further apart than this reload a stored value instead of keeping it in a register
/// (a reload is one L1 load; a long live range costs a spill and a reload).
const REUSE_WINDOW: usize = 12;

/// Nets an op reads.
fn op_inputs(op: &JOp) -> Vec<u32> {
    match op {
        JOp::Gate2 { a, b, .. } => vec![*a, *b],
        JOp::GateN { args, .. } => args.clone(),
        JOp::Resolve { desc, .. } => desc[4..].iter().map(|&x| x as u32).collect(),
        JOp::Lut(l) => l.support.clone(),
    }
}

/// Nets an op writes.
fn op_outputs(op: &JOp) -> Vec<u32> {
    match op {
        JOp::Gate2 { out, .. } | JOp::GateN { out, .. } => vec![out.net],
        JOp::Resolve { desc, .. } => vec![desc[0] as u32],
        JOp::Lut(l) => l.outputs.iter().map(|o| o.0).collect(),
    }
}

/// Emission order of a partition's ops: a post-order DFS from the ops whose outputs nothing in
/// the partition reads (in program order), so each value is produced right before its
/// consumers (short live ranges, few spills). Any topological order gives the same values and
/// marks: within a sweep, the effects of a partition's ops commute (marks only go to later
/// partitions and to kernels, read at round boundaries). Resolvers keep their program order
/// (the order of the contention diagnostics).
fn schedule(spec: &PartSpec) -> Vec<usize> {
    let n = spec.ops.len();
    let mut producer: HashMap<u32, usize> = HashMap::new();
    for (i, op) in spec.ops.iter().enumerate() {
        for o in op_outputs(op) {
            producer.insert(o, i);
        }
    }
    // Resolvers also depend on the previous resolver: they record contention in evaluation
    // order, which the diagnostics follow.
    let mut prev_res = None;
    let deps: Vec<Vec<usize>> = spec
        .ops
        .iter()
        .enumerate()
        .map(|(i, op)| {
            let mut d: Vec<usize> = op_inputs(op)
                .iter()
                .filter_map(|x| producer.get(x).copied())
                .filter(|&p| p < i)
                .collect();
            if matches!(op, JOp::Resolve { .. }) {
                d.extend(prev_res.replace(i));
            }
            d.sort_unstable();
            d.dedup();
            d
        })
        .collect();
    let mut consumed = vec![false; n];
    for d in &deps {
        for &p in d {
            consumed[p] = true;
        }
    }
    let mut order = Vec::with_capacity(n);
    let mut state = vec![0u8; n]; // 0 new, 1 on stack, 2 emitted
    for root in (0..n).filter(|&i| !consumed[i]).chain(0..n) {
        if state[root] != 0 {
            continue;
        }
        // Iterative post-order DFS.
        let mut stack = vec![(root, 0usize)];
        state[root] = 1;
        while let Some(&mut (i, ref mut next)) = stack.last_mut() {
            if *next < deps[i].len() {
                let d = deps[i][*next];
                *next += 1;
                if state[d] == 0 {
                    state[d] = 1;
                    stack.push((d, 0));
                }
            } else {
                state[i] = 2;
                order.push(i);
                stack.pop();
            }
        }
    }
    order
}

impl<'a> Gen<'a> {
    pub fn new(mut b: FunctionBuilder<'a>, ctx: Value, cb: SigRef) -> Gen<'a> {
        let values = b.ins().load(I64, mf(), ctx, off::VALUES);
        Gen {
            b,
            ctx,
            values,
            cb,
            cache: HashMap::new(),
            pos: 0,
            pinned: std::collections::HashSet::new(),
        }
    }

    /// Paranoid code: call the fault callback (which never returns) if `cond` is nonzero.
    pub fn fault_if(&mut self, cond: Value, code: u32) {
        let fb = self.b.create_block();
        let cont = self.b.create_block();
        self.b.ins().brif(cond, fb, &[], cont, &[]);
        self.b.switch_to_block(fb);
        self.b.set_cold_block(fb);
        let f = self.ctx_ptr(off::FAULT);
        let c = self.iconst(i64::from(code));
        let (cb, ctx) = (self.cb, self.ctx);
        self.b.ins().call_indirect(cb, f, &[ctx, c]);
        self.b.ins().trap(TrapCode::unwrap_user(1));
        self.b.switch_to_block(cont);
    }

    /// Paranoid code: fault unless the context's array lengths are `shape`'s.
    pub fn check_shape(&mut self, shape: &Shape) {
        let mut acc = self.iconst(0);
        for (o, v) in [
            (off::N_VALUES, shape.nets),
            (off::N_PART_WORDS, shape.part_words),
            (off::N_KERNEL_WORDS, shape.kernel_words),
            (off::N_KERNELS, shape.kernels),
            (off::N_LOG, shape.log),
            (off::N_LUTS, shape.luts),
            (off::N_LUT_ENTRIES, shape.lut_entries),
        ] {
            let x = self.b.ins().load(I32, mf(), self.ctx, o);
            let d = self.b.ins().bxor_imm_s(x, i64::from(v as i32));
            acc = self.b.ins().bor(acc, d);
        }
        self.fault_if(acc, fault::SHAPE);
    }

    /// `*parts_run += 1; *ops_run += ops`.
    pub fn count(&mut self, ops: u64) {
        for (o, n) in [(off::PARTS_RUN, 1), (off::OPS_RUN, ops)] {
            let p = self.ctx_ptr(o);
            let x = self.b.ins().load(I64, mf(), p, 0);
            let x = self.b.ins().iadd_imm_s(x, n as i64);
            self.b.ins().store(mf(), x, p, 0);
        }
    }

    /// An `I32` constant.
    pub fn iconst(&mut self, v: i64) -> Value {
        self.b.ins().iconst(I32, v)
    }

    /// Net `n` as a *doubled* code `2 * code` (I64): the form the gate code computes in (see
    /// [`Gen::lookup2`]).
    fn val(&mut self, n: u32) -> Value {
        if let Some(&(v, at)) = self.cache.get(&n)
            && (self.pos - at <= REUSE_WINDOW || self.pinned.contains(&n))
        {
            return v;
        }
        let r = self.b.ins().uload8(I64, mf(), self.values, n as i32);
        let v = self.b.ins().iadd(r, r);
        self.cache.insert(n, (v, self.pos));
        v
    }

    /// Load the pointer at `offset` of the native context.
    pub fn ctx_ptr(&mut self, offset: i32) -> Value {
        self.b.ins().load(I64, mf(), self.ctx, offset)
    }

    fn op(&mut self, op: &JOp) {
        match op {
            JOp::Gate2 { table, a, b, out } => {
                let d = if a == b {
                    // 1-input form: index a | a << 2 = 5a.
                    let t4 = (0..4).fold(0u32, |t, i| t | ((table >> (2 * 5 * i)) & 3) << (2 * i));
                    let da = self.val(*a);
                    self.lookup2(t4, da)
                } else {
                    let da = self.val(*a);
                    let db = self.val(*b);
                    // Shift count 2 * (a | b << 2) = 2a + 8b = da + 4 db (one `lea`).
                    let b4 = self.b.ins().ishl_imm_u(db, 2);
                    let sh = self.b.ins().iadd(da, b4);
                    self.lookup2(*table, sh)
                };
                self.out_doubled(out, d);
            }
            JOp::GateN {
                fold,
                invert,
                args,
                out,
            } => {
                let d = self.fold(*fold, *invert, args);
                self.out_doubled(out, d);
            }
            JOp::Resolve { op, desc, .. } => {
                let f = self.ctx_ptr(off::RESOLVE);
                let a = self.iconst(i64::from(*op));
                let (cb, ctx) = (self.cb, self.ctx);
                self.b.ins().call_indirect(cb, f, &[ctx, a]);
                // The host wrote the output: later reads load it.
                self.cache.remove(&(desc[0] as u32));
            }
            JOp::Lut(l) => self.lut(l),
        }
    }

    fn iconst64(&mut self, v: i64) -> Value {
        self.b.ins().iconst(I64, v)
    }

    /// Doubled table lookup: `2 * ((t >> sh) & 3)` as `((t << 1) >> sh) & 6`, where `sh` is
    /// twice the table index. With doubled operand codes the index arithmetic folds into one
    /// `lea`, so a 2-input gate is `lea`, `shrx`, `and` on its critical path.
    fn lookup2(&mut self, t: u32, sh: Value) -> Value {
        let tv = self.iconst64(i64::from(t) << 1);
        let s = self.b.ins().ushr(tv, sh);
        self.b.ins().band_imm_s(s, 6)
    }

    /// The interpreter's branchless N-input fold (`exec::fold_result`) over doubled codes:
    /// `seen` has bit `2c` set if some input has code `c`; the result is doubled.
    fn fold(&mut self, fold: u8, invert: bool, args: &[u32]) -> Value {
        let one = self.iconst64(1);
        let mut seen = self.iconst64(0);
        let mut parity = self.iconst64(0);
        for &x in args {
            let d = self.val(x);
            let bit = self.b.ins().ishl(one, d);
            seen = self.b.ins().bor(seen, bit);
            parity = self.b.ins().bxor(parity, d);
        }
        let c0 = self.iconst64(0);
        let c2 = self.iconst64(2);
        let c4 = self.iconst64(4);
        let d = match fold {
            table::AND => {
                // A 0 controls; all-1 is 1; else X.
                let z = self.b.ins().band_imm_s(seen, 1);
                let ones = self.b.ins().icmp_imm_u(IntCC::Equal, seen, 4);
                let t = self.b.ins().select(ones, c2, c4);
                self.b.ins().select(z, c0, t)
            }
            table::OR => {
                let o = self.b.ins().band_imm_s(seen, 4);
                let zeros = self.b.ins().icmp_imm_u(IntCC::Equal, seen, 1);
                let t = self.b.ins().select(zeros, c0, c4);
                self.b.ins().select(o, c2, t)
            }
            _ => {
                // Any X/Z gives X; else the parity of the 1s.
                let x = self.b.ins().band_imm_s(seen, 0x50);
                let p = self.b.ins().band_imm_s(parity, 2);
                self.b.ins().select(x, c4, p)
            }
        };
        if invert {
            // not1 on doubled codes: 0 <-> 2, 4 stays 4.
            let h = self.b.ins().ushr_imm_u(d, 2);
            let h = self.b.ins().bxor_imm_s(h, 1);
            let h = self.b.ins().band_imm_s(h, 1);
            let h = self.b.ins().ishl_imm_u(h, 1);
            self.b.ins().bxor(d, h)
        } else {
            d
        }
    }

    /// Store a gate result given as a doubled code `d`, and keep `d` for the readers.
    fn out_doubled(&mut self, out: &Out, d: Value) {
        if out.store || out.notify.is_some() {
            let r = self.b.ins().ushr_imm_u(d, 1);
            self.out(out, r);
        }
        self.cache.insert(out.net, (d, self.pos));
        if !out.store && out.notify.is_none() {
            // Not in memory: every reader must use the register value.
            self.pinned.insert(out.net);
        }
    }

    /// Store code `r` (any integer type) to `out` (change detection + propagation when it has
    /// readers).
    pub fn out(&mut self, out: &Out, r: Value) {
        let ty = self.b.func.dfg.value_type(r);
        match &out.notify {
            Some(rec) => {
                let old = self.b.ins().uload8(ty, mf(), self.values, out.net as i32);
                self.b.ins().istore8(mf(), r, self.values, out.net as i32);
                self.changed(old, r, rec);
            }
            None if out.store => {
                self.b.ins().istore8(mf(), r, self.values, out.net as i32);
            }
            None => {}
        }
    }

    /// `if old != new { propagate rec }` with the propagation out of line.
    fn changed(&mut self, old: Value, new: Value, rec: &NotifyRecord) {
        let ne = self.b.ins().icmp(IntCC::NotEqual, old, new);
        let nb = self.b.create_block();
        let cont = self.b.create_block();
        self.b.ins().brif(ne, nb, &[], cont, &[]);
        self.b.switch_to_block(nb);
        self.b.set_cold_block(nb);
        self.notify(rec, old, new);
        self.b.ins().jump(cont, &[]);
        self.b.switch_to_block(cont);
    }

    /// `word[w] |= m` for 64-bit words at `base`.
    pub fn or64(&mut self, base: Value, w: u32, m: u64) {
        let o = (w * 8) as i32;
        let x = self.b.ins().load(I64, mf(), base, o);
        let x = self.b.ins().bor_imm_s(x, m as i64);
        self.b.ins().store(mf(), x, base, o);
    }

    /// The effects of `Marks::flush` for one changed net.
    fn notify(&mut self, rec: &NotifyRecord, old: Value, new: Value) {
        if !rec.parts.is_empty() {
            let d = self.ctx_ptr(off::DIRTY);
            for &(w, m) in &rec.parts {
                self.or64(d, w, m);
            }
        }
        if rec.kernels.is_empty() {
            return;
        }
        if rec.log != NONE {
            let bval = self.ctx_ptr(off::BVAL);
            self.b.ins().istore8(mf(), old, bval, rec.log as i32);
            let bepoch = self.ctx_ptr(off::BEPOCH);
            let ep = self.b.ins().load(I32, mf(), self.ctx, off::EPOCH);
            self.b.ins().store(mf(), ep, bepoch, (rec.log * 4) as i32);
        }
        let kp = self.ctx_ptr(off::KPENDING);
        let dd = self.ctx_ptr(off::DDIRTY);
        let fo = self.ctx_ptr(off::FASTOK);
        let nb = self
            .b
            .ins()
            .icmp_imm_u(IntCC::UnsignedGreaterThanOrEqual, new, 2);
        for &(w, pend, data) in &rec.kernels {
            if pend != 0 {
                self.or64(kp, w, pend);
                let o = (w * 8) as i32;
                let f = self.b.ins().load(I64, mf(), fo, o);
                let cleared = self.b.ins().band_imm_s(f, !pend as i64);
                let f2 = self.b.ins().select(nb, cleared, f);
                self.b.ins().store(mf(), f2, fo, o);
            }
            if data != 0 {
                self.or64(dd, w, data);
            }
        }
        if !rec.kin.is_empty() {
            let o1 = self.b.ins().icmp_imm_u(IntCC::Equal, old, 1);
            let n1 = self.b.ins().icmp_imm_u(IntCC::Equal, new, 1);
            let fl = self.b.ins().bxor(o1, n1);
            let flip = self.b.ins().uextend(I32, fl);
            let kin = self.ctx_ptr(off::KIN);
            for &(k, pos) in &rec.kin {
                let o = (k * 4) as i32;
                let x = self.b.ins().load(I32, mf(), kin, o);
                let bit = self.b.ins().ishl_imm_u(flip, i64::from(pos));
                let x = self.b.ins().bxor(x, bit);
                self.b.ins().store(mf(), x, kin, o);
            }
        }
    }

    fn lut(&mut self, l: &LutSpec) {
        let mut idx = self.iconst64(0);
        let mut nb = self.iconst64(0);
        for (i, &s) in l.support.iter().enumerate() {
            // Doubled code: bit 1 = the binary value, bit 2 = X/Z.
            let d = self.val(s);
            nb = self.b.ins().bor(nb, d);
            let bit = self.b.ins().band_imm_s(d, 2);
            let bit = if i == 0 {
                self.b.ins().ushr_imm_u(bit, 1)
            } else {
                self.b.ins().ishl_imm_u(bit, i as i64 - 1)
            };
            idx = self.b.ins().bor(idx, bit);
        }
        let unknown = self.b.ins().band_imm_s(nb, 4);
        let cold = self.b.create_block();
        let bin = self.b.create_block();
        let join = self.b.create_block();
        self.b.ins().brif(unknown, cold, &[], bin, &[]);

        self.b.switch_to_block(cold);
        self.b.set_cold_block(cold);
        let f = self.ctx_ptr(off::LUT_COLD);
        let a = self.iconst(i64::from(l.lut));
        let (cb, ctx) = (self.cb, self.ctx);
        self.b.ins().call_indirect(cb, f, &[ctx, a]);
        self.b.ins().jump(join, &[]);

        self.b.switch_to_block(bin);
        let entries = self.ctx_ptr(off::LUT_ENTRIES);
        let byte = self.b.ins().ishl_imm_u(idx, 2);
        let addr = self.b.ins().iadd(entries, byte);
        let e = self.b.ins().load(I32, mf(), addr, (l.table * 4) as i32);
        let last_p = self.ctx_ptr(off::LUT_LAST);
        let lo = (l.lut * 4) as i32;
        let last = self.b.ins().load(I32, mf(), last_p, lo);
        let write = self.b.create_block();
        let stale = self.b.create_block();
        if l.outputs.len() < 32 {
            // `last` is NONE (never equal to an entry of < 32 outputs) or the entry the
            // outputs hold.
            let same = self.b.ins().icmp(IntCC::Equal, e, last);
            self.b.ins().brif(same, stale, &[], write, &[]);
        } else {
            self.b.ins().jump(write, &[]);
        }
        self.b.switch_to_block(write);
        for (j, (net, rec)) in l.outputs.iter().enumerate() {
            let v = self.b.ins().ushr_imm_u(e, j as i64);
            let v = self.b.ins().band_imm_s(v, 1);
            let out = Out {
                net: *net,
                notify: rec.clone(),
                store: true,
            };
            self.out(&out, v);
        }
        let last_p = self.ctx_ptr(off::LUT_LAST);
        self.b.ins().store(mf(), e, last_p, lo);
        self.b.ins().jump(stale, &[]);

        self.b.switch_to_block(stale);
        let st = self.ctx_ptr(off::LUT_STALE);
        self.or64(st, l.lut / 64, 1u64 << (l.lut % 64));
        self.b.ins().jump(join, &[]);

        self.b.switch_to_block(join);
        // The outputs are not cached (two paths write them); later reads load them.
        for (net, _) in &l.outputs {
            self.cache.remove(net);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate(out: u32, a: u32, b: u32) -> JOp {
        JOp::Gate2 {
            table: 0,
            a,
            b,
            out: Out {
                net: out,
                notify: None,
                store: true,
            },
        }
    }

    fn resolve(op: u32, out: u32, drivers: &[u32]) -> JOp {
        let mut desc = vec![u64::from(out), 0, 0, 0];
        desc.extend(drivers.iter().map(|&d| u64::from(d)));
        JOp::Resolve {
            op,
            desc,
            rec: None,
        }
    }

    /// Every op after its in-partition producers; resolvers in program order; a permutation.
    fn check(spec: &PartSpec) {
        let order = schedule(spec);
        let mut pos = vec![usize::MAX; spec.ops.len()];
        for (k, &i) in order.iter().enumerate() {
            assert_eq!(pos[i], usize::MAX, "op {i} scheduled twice");
            pos[i] = k;
        }
        assert!(pos.iter().all(|&p| p != usize::MAX));
        for (i, op) in spec.ops.iter().enumerate() {
            for x in op_inputs(op) {
                if let Some(p) = spec
                    .ops
                    .iter()
                    .position(|o| op_outputs(o).contains(&x))
                    .filter(|&p| p < i)
                {
                    assert!(pos[p] < pos[i], "op {i} before its producer {p}");
                }
            }
        }
        let res: Vec<usize> = (0..spec.ops.len())
            .filter(|&i| matches!(spec.ops[i], JOp::Resolve { .. }))
            .collect();
        assert!(res.windows(2).all(|w| pos[w[0]] < pos[w[1]]));
    }

    #[test]
    fn schedule_is_topological_and_keeps_resolver_order() {
        // Two independent cones interleaved by level, a shared value, resolvers.
        let spec = PartSpec {
            ops: vec![
                gate(10, 0, 1),
                gate(20, 2, 3),
                gate(11, 10, 1),
                gate(21, 20, 3),
                resolve(4, 30, &[11, 21]),
                gate(12, 11, 10),
                resolve(6, 31, &[0, 1]),
                gate(22, 21, 12),
            ],
            paranoid: None,
        };
        check(&spec);
        let order = schedule(&spec);
        // The first cone is emitted before the second one starts (short live ranges).
        assert_eq!(&order[..2], &[0, 2]);
    }

    #[test]
    fn schedule_random_dags() {
        let mut seed = 0x1234_5678_u64;
        let mut rnd = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for _ in 0..200 {
            let n = 1 + rnd(40) as usize;
            let mut ops = Vec::new();
            for i in 0..n {
                let pick = |r: u64| {
                    if r < 3 || i == 0 {
                        r as u32
                    } else {
                        100 + (r % i as u64) as u32
                    }
                };
                let (a, b) = (pick(rnd(10)), pick(rnd(10)));
                ops.push(if rnd(8) == 0 {
                    resolve(i as u32, 100 + i as u32, &[a, b])
                } else {
                    gate(100 + i as u32, a, b)
                });
            }
            check(&PartSpec {
                ops,
                paranoid: None,
            });
        }
    }

    /// A context over private buffers for a partition function of `nets` nets.
    struct FakeState {
        values: Vec<u8>,
        words: Vec<u64>,
        kin: Vec<u32>,
        bval: Vec<u8>,
        bepoch: Vec<u32>,
        lut_last: Vec<u32>,
        counters: [u64; 2],
    }

    unsafe extern "C" fn no_cb(_: *mut panda_fast::native::NativeCtx, _: u32) {
        unreachable!("no callbacks in this test");
    }

    unsafe extern "C" fn test_fault(_: *mut panda_fast::native::NativeCtx, code: u32) {
        eprintln!("FAULT {code}");
        std::process::abort();
    }

    impl FakeState {
        fn new(nets: usize) -> FakeState {
            FakeState {
                values: vec![0; nets],
                words: vec![0; 8],
                kin: vec![0; 4],
                bval: vec![0; 4],
                bepoch: vec![0; 4],
                lut_last: vec![0; 4],
                counters: [0; 2],
            }
        }

        fn ctx(&mut self, n_values: u32) -> panda_fast::native::NativeCtx {
            let w = self.words.as_mut_ptr();
            panda_fast::native::NativeCtx {
                values: self.values.as_mut_ptr(),
                dirty: w,
                kpending: w,
                ddirty: w,
                fastok: w,
                kin: self.kin.as_mut_ptr(),
                bval: self.bval.as_mut_ptr(),
                bepoch: self.bepoch.as_mut_ptr(),
                lut_last: self.lut_last.as_mut_ptr(),
                lut_stale: w,
                lut_entries: self.lut_last.as_ptr(),
                parts_run: &raw mut self.counters[0],
                ops_run: &raw mut self.counters[1],
                env: std::ptr::null_mut(),
                lut_cold: no_cb,
                resolve: no_cb,
                fault: test_fault,
                epoch: 0,
                n_values,
                n_part_words: 1,
                n_kernel_words: 1,
                n_kernels: 1,
                n_log: 1,
                n_luts: 0,
                n_lut_entries: 0,
            }
        }
    }

    const SHAPE: Shape = Shape {
        nets: 3,
        partitions: 1,
        part_words: 1,
        kernels: 1,
        kernel_words: 1,
        log: 1,
        luts: 0,
        lut_entries: 0,
        tuples: 0,
    };

    /// `net 2 = net 0 AND net 1`, compiled (paranoid or not).
    fn and_partition(paranoid: bool) -> (CodeMem, panda_fast::native::PartitionFn) {
        let tables = panda_fast::ir::gate_tables();
        let spec = PartSpec {
            ops: vec![JOp::Gate2 {
                table: tables[usize::from(table::AND)],
                a: 0,
                b: 1,
                out: Out {
                    net: 2,
                    notify: None,
                    store: true,
                },
            }],
            paranoid: paranoid.then_some(SHAPE),
        };
        spec.check(&SHAPE).unwrap();
        let mut cc = Compiler::new().unwrap();
        cc.add_partition(&spec).unwrap();
        let (mem, ptrs) = cc.finish().unwrap();
        // SAFETY: generated with the `PartitionFn` signature; `mem` owns it.
        let f =
            unsafe { std::mem::transmute::<*const u8, panda_fast::native::PartitionFn>(ptrs[0]) };
        (mem, f)
    }

    #[test]
    fn generated_gate_code_matches_the_table() {
        for paranoid in [false, true] {
            let (_mem, f) = and_partition(paranoid);
            let mut st = FakeState::new(3);
            for a in 0..4u8 {
                for b in 0..4u8 {
                    st.values[..2].copy_from_slice(&[a, b]);
                    let mut ctx = st.ctx(3);
                    // SAFETY: the context's arrays have the compiled shape.
                    unsafe { f(&mut ctx, 0) };
                    let want = (panda_fast::ir::gate_tables()[0] >> (2 * (a | b << 2))) & 3;
                    assert_eq!(u32::from(st.values[2]), want, "{a} AND {b}");
                }
            }
            assert_eq!(st.counters, [16, 16]);
        }
        // Static bounds: an operand beyond the nets is rejected before code generation.
        let bad = PartSpec {
            ops: vec![JOp::Gate2 {
                table: 0,
                a: 3,
                b: 0,
                out: Out {
                    net: 2,
                    notify: None,
                    store: true,
                },
            }],
            paranoid: None,
        };
        assert!(bad.check(&SHAPE).is_err());
    }

    /// Child half of `paranoid_code_faults_on_a_mismatched_context`.
    #[test]
    #[ignore = "run by paranoid_code_faults_on_a_mismatched_context in a child process"]
    fn paranoid_fault_child() {
        if std::env::var_os("PANDA_JIT_FAULT_CHILD").is_none() {
            return;
        }
        let (_mem, f) = and_partition(true);
        let mut st = FakeState::new(3);
        let mut ctx = st.ctx(3);
        // SAFETY: matching shape.
        unsafe { f(&mut ctx, 0) };
        eprintln!("MATCHING CONTEXT OK");
        // A context claiming fewer nets than the code indexes: the entry check must fault
        // before any access.
        let mut ctx = st.ctx(2);
        // SAFETY: the arrays are still large enough; only the declared length differs.
        unsafe { f(&mut ctx, 0) };
        eprintln!("NO FAULT");
    }

    #[test]
    fn paranoid_code_faults_on_a_mismatched_context() {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "codegen::tests::paranoid_fault_child",
                "--ignored",
            ])
            .args(["--nocapture", "--test-threads=1"])
            .env("PANDA_JIT_FAULT_CHILD", "1")
            .output()
            .unwrap();
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "child should abort: {err}");
        assert!(err.contains("MATCHING CONTEXT OK"), "{err}");
        assert!(err.contains(&format!("FAULT {}", fault::SHAPE)), "{err}");
        assert!(!err.contains("NO FAULT"), "{err}");
    }
}
