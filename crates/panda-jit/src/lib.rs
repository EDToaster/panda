//! panda-jit — the tier-2 backend of Fast mode (SPEC §8.9): `panda_fast` programs compiled to
//! native code with Cranelift.
//!
//! The JIT plugs into [`panda_fast::FastSim`] through its native-code hook
//! ([`panda_fast::native`]) and runs over the interpreter's flat state arrays with exactly the
//! interpreter's effects, so tiers mix freely and code can be swapped in at any event boundary:
//!
//! * **Partitions** (`codegen`): each partition of the compiled [`Program`] becomes one function:
//!   straight-line 4-state gate code in a register-friendly order (2-bit-code tables in
//!   immediates, doubled codes so a 2-input gate is `lea`/`shrx`/`and`), outputs of fused-away
//!   gates kept in registers only, change detection with each net's change-propagation record
//!   decoded and inlined out of line, LUTs with an inline binary check, and host callbacks for
//!   the cold cases (LUT with `X`/`Z` support, resolvers).
//! * **Events** (`event`): per clock tuple, one function runs an event's timed-source writes,
//!   sweeps (one guarded call per partition) and round loop, with `fast_round`'s masks as
//!   constants, the transparent kernels' output tables as immediates or deduplicated packed
//!   tables, and the output writes with inlined change propagation. At the first round boundary
//!   that needs anything else (a general-path kernel, an order check, explicit last-transition
//!   inputs, the delta limit) it returns at that decision point and the interpreter continues.
//!
//! [`Jit::install`] compiles synchronously; [`Jit::attach`] compiles on a background thread and
//! the engine hot-swaps the code in at its first event boundary after it is ready (until then it
//! interprets). Compiled functions are cached in a [`JitCache`] keyed by a content hash of a
//! self-contained description of everything the function depends on (ops, net indices, decoded
//! records, tables, the program contents behind callbacks; equality is checked on a hit), so a
//! recompiled program only recompiles the partitions and event functions whose content changed.
//! Because generated code embeds net indices, edits that renumber nets invalidate the affected
//! functions (all of them, if the numbering shifts globally).
//!
//! Memory safety: generated code addresses memory only as a context base pointer plus an
//! immediate offset, or plus an index bounded by construction (a LUT index of `support` bits, a
//! kernel table index `kin | pure`, the write queue position). Before code generation every
//! immediate is checked against the program's array lengths (`PartSpec::check`,
//! `EventSpec::check`; a function that fails is not compiled and its partition / tuple is
//! interpreted), and the engine runs code only on the state of the program it was compiled for
//! (`FastSim::install_native` checks identity). The dynamic indices rely on invariants of the
//! verified program and runtime (`kin` only has bits below the kernel's input count, each
//! kernel is queued at most once per round); [`JitOptions::paranoid`] code checks those, and
//! the context's array lengths, at runtime and aborts on a violation.
//!
//! Failures: a function whose code generation fails (Cranelift error or panic, or a failed
//! bounds check) is left out and runs in the interpreter ([`JitStats::failed`]); the rest of
//! the program is compiled again in a fresh module.
//!
//! Exactness: the JIT mirrors the interpreter op by op; `tests/lockstep.rs` compares a JIT and
//! an interpreting engine after every event (all values, every statistic, diagnostics,
//! internal invariants), including mixed tiers, hot swap, `load_state` and recompilation, and
//! `tests/fast_*.rs` run all of panda-fast's FastRef differential suites with the JIT installed.
//!
//! Performance (Panda-8 `bench_loop`, `tests/bench.rs`): about 1.7–1.8× the interpreter's
//! cycles/s; the generated code executes ~2.2× fewer instructions than the interpreter but at a
//! lower IPC (the gate code is latency-bound on the ripple-carry chains, and the kernel round
//! work is per-kernel table lookups and writes). Compiling Panda-8 takes ~30–40 ms (11
//! partitions ≈ 9 ms, four event functions the rest).

mod codegen;
mod event;
mod spec;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use panda_fast::native::{EventFn, NativeCode, NativeSlot, PartitionFn};
use panda_fast::{FastSim, Program};

use crate::codegen::{CodeMem, Compiler};
use crate::event::EventSpec;
use crate::spec::{PartSpec, Shape};

/// JIT tunables.
#[derive(Clone, Debug)]
pub struct JitOptions {
    /// Compile the partitions (default true).
    pub partitions: bool,
    /// Compile the per-clock-tuple event functions (sweeps + kernel round loop; default true).
    pub events: bool,
    /// Cache size (functions) above which entries not used by the latest compilation are
    /// dropped (default 4096).
    pub max_cache_entries: usize,
    /// Paranoid mode: the generated code also checks at runtime that the context's arrays
    /// have the compiled program's lengths, that every kernel table index is inside its table
    /// and agrees with the clock tuple's pure input bits, that fused event functions only meet
    /// proven table entries and that the write queue stays within bounds; a failed check
    /// aborts the process with a message (`panda_fast::native::FaultFn`). The static bounds
    /// checks of every embedded index run in every mode. Default: on in debug builds or when
    /// the environment variable `PANDA_JIT_PARANOID` is set.
    pub paranoid: bool,
    /// Test hook: make the compilation of some functions fail (see [`Inject`]).
    #[doc(hidden)]
    pub inject: Option<fn(Func) -> Inject>,
}

impl Default for JitOptions {
    fn default() -> Self {
        JitOptions {
            partitions: true,
            events: true,
            max_cache_entries: 4096,
            paranoid: cfg!(debug_assertions) || std::env::var_os("PANDA_JIT_PARANOID").is_some(),
            inject: None,
        }
    }
}

/// A function of a compilation (for [`JitOptions::inject`]).
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Func {
    /// Partition `p`.
    Partition(usize),
    /// The event function of clock tuple `t`.
    Event(u32),
}

/// An injected compilation failure (for [`JitOptions::inject`]).
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Inject {
    /// Compile normally.
    None,
    /// Code generation returns an error.
    Error,
    /// Code generation panics.
    Panic,
}

/// What one compilation did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct JitStats {
    /// Partitions in the program.
    pub partitions: usize,
    /// Partitions compiled now.
    pub compiled: usize,
    /// Partitions whose code came from the cache.
    pub reused: usize,
    /// Clock tuples with an event function.
    pub events: usize,
    /// Event functions compiled now.
    pub events_compiled: usize,
    /// Event functions from the cache.
    pub events_reused: usize,
    /// Event functions that decide and write kernel outputs in one pass.
    pub events_fused: usize,
    /// Functions that could not be compiled (failed bounds check, Cranelift error or panic):
    /// their partitions / clock tuples run in the interpreter.
    pub failed: usize,
    /// The reasons of the failures (at most 16).
    pub errors: Vec<String>,
    /// Machine-code bytes emitted now.
    pub code_bytes: usize,
    /// Wall time of the compilation (spec extraction, cache lookups, codegen, linking).
    pub compile_time: Duration,
}

impl JitStats {
    fn fail(&mut self, what: Func, e: String) {
        self.failed += 1;
        if self.errors.len() < 16 {
            self.errors.push(format!("{what:?}: {e}"));
        }
    }
}

/// A function address (plain integer so cache entries are `Send`).
#[derive(Clone, Copy)]
struct Addr(usize);

struct Entry<S> {
    spec: Arc<S>,
    addr: Addr,
    mem: Arc<CodeMem>,
    used: u64,
}

#[derive(Default)]
struct CacheInner {
    parts: HashMap<u64, Vec<Entry<PartSpec>>>,
    events: HashMap<u64, Vec<Entry<EventSpec>>>,
    generation: u64,
}

impl CacheInner {
    fn len(&self) -> usize {
        self.parts.values().map(Vec::len).sum::<usize>()
            + self.events.values().map(Vec::len).sum::<usize>()
    }
}

/// Compiled functions by content (see the crate docs). Shared by every compilation of one
/// [`Jit`] (and its clones).
#[derive(Default)]
pub struct JitCache {
    inner: Mutex<CacheInner>,
}

impl std::fmt::Debug for JitCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitCache")
            .field("entries", &self.len())
            .finish()
    }
}

impl JitCache {
    /// Number of cached functions.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// True if nothing is cached.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, CacheInner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Add a compiled function to the cache, unless an equal spec is there already (a concurrent
/// compilation of the same content).
fn insert<S: PartialEq>(
    map: &mut HashMap<u64, Vec<Entry<S>>>,
    key: u64,
    spec: &Option<Arc<S>>,
    addr: Addr,
    mem: &Arc<CodeMem>,
    generation: u64,
) {
    let spec = spec.as_ref().expect("compiled from a spec");
    let v = map.entry(key).or_default();
    if let Some(e) = v.iter_mut().find(|e| *e.spec == **spec) {
        e.used = e.used.max(generation);
        return;
    }
    v.push(Entry {
        spec: spec.clone(),
        addr,
        mem: mem.clone(),
        used: generation,
    });
}

/// The message of a caught panic.
fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "(no message)".into())
}

fn lookup<S: PartialEq>(
    map: &mut HashMap<u64, Vec<Entry<S>>>,
    key: u64,
    spec: &S,
    generation: u64,
) -> Option<(Addr, Arc<CodeMem>)> {
    let e = map.get_mut(&key)?.iter_mut().find(|e| *e.spec == *spec)?;
    e.used = generation;
    Some((e.addr, e.mem.clone()))
}

/// The JIT compiler (cheap to clone; clones share the cache).
#[derive(Clone, Debug, Default)]
pub struct Jit {
    opts: JitOptions,
    cache: Arc<JitCache>,
}

impl Jit {
    /// A JIT with its own empty cache.
    pub fn new(opts: JitOptions) -> Jit {
        Jit {
            opts,
            cache: Arc::default(),
        }
    }

    /// A JIT with other options sharing this one's cache (the cache keys include everything
    /// the options change in the generated code).
    pub fn with_options(&self, opts: JitOptions) -> Jit {
        Jit {
            opts,
            cache: self.cache.clone(),
        }
    }

    /// The options.
    pub fn options(&self) -> &JitOptions {
        &self.opts
    }

    /// The function cache.
    pub fn cache(&self) -> &Arc<JitCache> {
        &self.cache
    }

    /// Compile `prog` (reusing cached functions).
    ///
    /// A function that cannot be compiled (an index outside the program's arrays, a Cranelift
    /// error or panic) is left out, so its partition / clock tuple runs in the interpreter
    /// ([`JitStats::failed`]); the result is an error only if the JIT cannot run at all here
    /// (unsupported host, program beyond the generator's limits) or every function failed.
    pub fn compile(&self, prog: &Arc<Program>) -> Result<(Arc<NativeCode>, JitStats), String> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.compile_inner(prog)))
            .unwrap_or_else(|p| Err(format!("JIT panicked: {}", panic_message(&*p))))
    }

    fn compile_inner(&self, prog: &Arc<Program>) -> Result<(Arc<NativeCode>, JitStats), String> {
        let t0 = Instant::now();
        let shape = Shape::of(prog)?;
        let paranoid = self.opts.paranoid.then_some(shape);
        let mut stats = JitStats {
            partitions: prog.partitions.len(),
            ..JitStats::default()
        };
        let np = prog.partitions.len();
        let mut part_specs: Vec<Option<Arc<PartSpec>>> = vec![None; np];
        if self.opts.partitions {
            for (p, slot) in part_specs.iter_mut().enumerate() {
                let spec = PartSpec::extract(prog, p, paranoid);
                match spec.check(&shape) {
                    Ok(()) => *slot = Some(Arc::new(spec)),
                    Err(e) => stats.fail(Func::Partition(p), e),
                }
            }
        }
        let mut event_specs: Vec<Option<Arc<EventSpec>>> = Vec::new();
        if self.opts.events {
            for t in 0..prog.num_tuples {
                let spec = EventSpec::extract(prog, t, paranoid);
                event_specs.push(match spec.map(|s| s.check(&shape).map(|()| s)) {
                    Some(Ok(s)) => Some(Arc::new(s)),
                    Some(Err(e)) => {
                        stats.fail(Func::Event(t), e);
                        None
                    }
                    None => None,
                });
            }
        }
        let part_keys: Vec<u64> = part_specs
            .iter()
            .map(|s| s.as_ref().map_or(0, |s| s.key()))
            .collect();
        let event_keys: Vec<u64> = event_specs
            .iter()
            .map(|s| s.as_ref().map_or(0, |s| s.key()))
            .collect();

        // Cache lookups.
        let mut parts: Vec<Option<(Addr, Arc<CodeMem>)>> = vec![None; np];
        let mut events: Vec<Option<(Addr, Arc<CodeMem>)>> = vec![None; event_specs.len()];
        let generation;
        {
            let mut c = self.cache.lock();
            c.generation += 1;
            generation = c.generation;
            for (p, s) in part_specs.iter().enumerate() {
                if let Some(s) = s {
                    parts[p] = lookup(&mut c.parts, part_keys[p], s, generation);
                }
            }
            for (t, s) in event_specs.iter().enumerate() {
                if let Some(s) = s {
                    events[t] = lookup(&mut c.events, event_keys[t], s, generation);
                }
            }
        }
        stats.reused = parts.iter().filter(|p| p.is_some()).count();
        stats.events_reused = events.iter().filter(|r| r.is_some()).count();

        // Compile the misses in one module; a function that fails is dropped and the rest are
        // compiled again in a fresh module (a failed definition may leave a module unusable).
        let mut todo: Vec<Func> = (0..np)
            .filter(|&p| part_specs[p].is_some() && parts[p].is_none())
            .map(Func::Partition)
            .chain(
                (0..event_specs.len())
                    .filter(|&t| event_specs[t].is_some() && events[t].is_none())
                    .map(|t| Func::Event(t as u32)),
            )
            .collect();
        let mut retries = 0;
        while !todo.is_empty() {
            let mut cc = Compiler::new()?;
            let mut failure = None;
            for (i, &f) in todo.iter().enumerate() {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    match self.opts.inject.map_or(Inject::None, |h| h(f)) {
                        Inject::None => {}
                        Inject::Error => return Err("injected failure".to_string()),
                        Inject::Panic => panic!("injected panic"),
                    }
                    match f {
                        Func::Partition(p) => {
                            cc.add_partition(part_specs[p].as_ref().expect("spec"))
                        }
                        Func::Event(t) => event::add_event(
                            &mut cc,
                            event_specs[t as usize].as_ref().expect("spec"),
                        ),
                    }
                }));
                match r {
                    Ok(Ok(_)) => {}
                    Ok(Err(e)) => {
                        failure = Some((i, e, false));
                        break;
                    }
                    Err(p) => {
                        failure = Some((i, format!("panic: {}", panic_message(&*p)), true));
                        break;
                    }
                }
            }
            if let Some((i, e, panicked)) = failure {
                stats.fail(todo.remove(i), e);
                if panicked {
                    // The module's state is unknown after a panic: leak it rather than free it.
                    std::mem::forget(cc);
                } else {
                    cc.abandon();
                }
                retries += 1;
                if retries > 16 {
                    for f in todo.drain(..) {
                        stats.fail(f, "too many failures in this compilation".into());
                    }
                }
                continue;
            }
            let code_bytes = cc.code_bytes;
            let (mem, addrs) = match cc.finish() {
                Ok(r) => r,
                Err(e) => {
                    for f in todo.drain(..) {
                        stats.fail(f, e.clone());
                    }
                    break;
                }
            };
            stats.code_bytes += code_bytes;
            let mem = Arc::new(mem);
            let mut c = self.cache.lock();
            for (&f, a) in todo.iter().zip(addrs) {
                let a = Addr(a as usize);
                match f {
                    Func::Partition(p) => {
                        parts[p] = Some((a, mem.clone()));
                        stats.compiled += 1;
                        insert(
                            &mut c.parts,
                            part_keys[p],
                            &part_specs[p],
                            a,
                            &mem,
                            generation,
                        );
                    }
                    Func::Event(t) => {
                        let t = t as usize;
                        events[t] = Some((a, mem.clone()));
                        stats.events_compiled += 1;
                        insert(
                            &mut c.events,
                            event_keys[t],
                            &event_specs[t],
                            a,
                            &mem,
                            generation,
                        );
                    }
                }
            }
            todo.clear();
            if c.len() > self.opts.max_cache_entries {
                for v in c.parts.values_mut() {
                    v.retain(|e| e.used == generation);
                }
                for v in c.events.values_mut() {
                    v.retain(|e| e.used == generation);
                }
                c.parts.retain(|_, v| !v.is_empty());
                c.events.retain(|_, v| !v.is_empty());
            }
        }
        stats.events = events.iter().filter(|r| r.is_some()).count();
        stats.events_fused = events
            .iter()
            .zip(&event_specs)
            .filter(|(r, s)| r.is_some() && s.as_ref().is_some_and(|s| s.is_fused()))
            .count();
        if stats.failed > 0 && parts.iter().chain(&events).all(Option::is_none) {
            return Err(format!(
                "no function could be compiled: {}",
                stats.errors.join("; ")
            ));
        }

        // Keep every code module referenced.
        let mut keep: Vec<Arc<CodeMem>> = Vec::new();
        let mut add_keep = |m: &Arc<CodeMem>| {
            if !keep.iter().any(|k| Arc::ptr_eq(k, m)) {
                keep.push(m.clone());
            }
        };
        let part_fns: Vec<Option<PartitionFn>> = parts
            .iter()
            .map(|p| {
                p.as_ref().map(|(a, m)| {
                    add_keep(m);
                    // SAFETY: `a` is the address of a finalized function generated for this
                    // signature (`extern "C" fn(*mut NativeCtx, u32)`) by
                    // `Compiler::add_partition`; `m` owns its memory.
                    unsafe { std::mem::transmute::<usize, PartitionFn>(a.0) }
                })
            })
            .collect();
        let event_fns: Vec<Option<EventFn>> = events
            .iter()
            .map(|r| {
                r.as_ref().map(|(a, m)| {
                    add_keep(m);
                    // SAFETY: as above, generated by `event::add_event` with the `EventFn`
                    // signature.
                    unsafe { std::mem::transmute::<usize, EventFn>(a.0) }
                })
            })
            .collect();
        let keep: Vec<Arc<dyn std::any::Any + Send + Sync>> = keep
            .into_iter()
            .map(|m| m as Arc<dyn std::any::Any + Send + Sync>)
            .collect();
        // SAFETY: every function was generated from a spec extracted from `prog` (or an equal
        // spec, which compiles to interchangeable code), whose embedded indices were checked
        // against `prog`'s array lengths (`PartSpec::check` / `EventSpec::check`; the engine
        // runs the code only on `prog`'s state, see `FastSim::install_native`), and implements
        // the hook's semantics; `keep` owns the code memory.
        let code = unsafe { NativeCode::new(prog.clone(), part_fns, event_fns, keep)? };
        stats.compile_time = t0.elapsed();
        Ok((Arc::new(code), stats))
    }

    /// Compile `sim`'s program and install the code now.
    pub fn install(&self, sim: &mut FastSim) -> Result<JitStats, String> {
        let (code, stats) = self.compile(sim.program_arc())?;
        sim.install_native(code)?;
        Ok(stats)
    }

    /// Compile `prog` on a background thread and publish the code to `slot`.
    pub fn compile_in_background(
        &self,
        prog: Arc<Program>,
        slot: Arc<NativeSlot>,
    ) -> JoinHandle<Result<JitStats, String>> {
        let jit = self.clone();
        std::thread::Builder::new()
            .name("panda-jit".into())
            .spawn(move || {
                // The engine holds the other reference: if it is gone (e.g. replaced after an
                // edit), nobody will pick the code up.
                if Arc::strong_count(&slot) == 1 {
                    return Err("abandoned: the engine was dropped".into());
                }
                let (code, stats) = jit.compile(&prog)?;
                if Arc::strong_count(&slot) > 1 {
                    slot.publish(code);
                }
                Ok(stats)
            })
            .expect("spawn the JIT thread")
    }

    /// Compile `sim`'s program in the background; `sim` keeps interpreting and hot-swaps the
    /// native code in at its first event boundary after the code is ready.
    pub fn attach(&self, sim: &mut FastSim) -> JoinHandle<Result<JitStats, String>> {
        let slot = NativeSlot::new();
        sim.set_native_slot(slot.clone());
        self.compile_in_background(sim.program_arc().clone(), slot)
    }
}
