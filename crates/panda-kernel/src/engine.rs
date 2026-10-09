//! Fast-mode engines behind one trait, so the facade can swap them.
//!
//! Adding a backend takes one [`FastBackend`] variant, one `impl_fast_engine!(Type)` line and one
//! arm in [`build_engine`].
//!
//! [`FastBackend::Jit`] is the compiled engine plus the native tier: the `FastSim` starts
//! interpreting at once while `panda_jit` compiles its program (in the background with
//! [`JitTier::background`]), and installs the native code at its next event boundary.

use std::sync::Arc;

use panda_compile::{FastOptions, FastRef, KernelSet, KernelTable};
use panda_fast::{CompileOptions as FastCompileOptions, FastSim, FastSimOptions};
use panda_netlist::flat::FlatNetlist;
use panda_netlist::{Diagnostic, Logic, Value};

/// Outcome of [`FastEngine::load_state`].
pub(crate) use panda_compile::LoadReport;

/// Which Fast-mode engine the facade runs once compilation is done.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FastBackend {
    /// The reference interpreter of SPEC §6.2 (FastRef semantics; the oracle).
    Reference,
    /// The compiled engine `panda_fast::FastSim` (bit-identical to FastRef, much faster). While
    /// its program compiles, a Reference engine (built from the same kernel tables) serves.
    Compiled,
    /// Tiered: Reference while the kernels compile, then the `FastSim` interpreter as soon as its
    /// program is compiled, then the Cranelift JIT (`panda_jit`) hot-swapped in at an event
    /// boundary once its native code is ready (SPEC §8.9). Bit-identical to the others at every
    /// tier. Without the `jit` cargo feature this is [`FastBackend::Compiled`].
    #[default]
    Jit,
}

/// The operations the facade needs from a Fast-mode engine. Method names and semantics are those
/// of [`FastRef`] (including [`FastRef::load_state`]).
pub(crate) trait FastEngine: Send {
    /// Load a state at `tick` (semantics of [`FastRef::load_state`]: inconsistent kernels
    /// finish their transition).
    fn load_state(&mut self, tick: u64, values: &[Logic], inputs: &[Value]) -> LoadReport;
    /// Queue a top-level input change (processed by the next `step_event`).
    fn set_input(&mut self, pin: &str, value: &Value) -> Result<(), String>;
    /// Process one event; `None` if there is nothing to do.
    fn step_event(&mut self) -> Option<u64>;
    /// Tick of the most recently processed event (or of the last `load_state`).
    fn tick(&self) -> u64;
    /// Tick of the next scheduled source event.
    fn next_event_tick(&self) -> Option<u64>;
    /// All net values at the settled point.
    fn values(&self) -> &[Logic];
    /// Read by hierarchical name / pin name.
    fn read(&self, name: &str) -> Option<Value>;
    /// Drain diagnostics.
    fn take_diagnostics(&mut self) -> Vec<Diagnostic>;
    /// Install native code that became ready (normally done at the next event); true if the
    /// engine runs native code.
    fn native_active(&mut self) -> bool {
        false
    }
    /// True while the engine's native code is still being compiled in the background (false
    /// once it is installed, or if the compilation failed: the engine keeps interpreting).
    fn native_pending(&mut self) -> bool {
        false
    }
}

/// Implement [`FastEngine`] for a type with FastRef's method names.
macro_rules! impl_fast_engine {
    ($t:ty) => {
        impl FastEngine for $t {
            fn load_state(&mut self, tick: u64, values: &[Logic], inputs: &[Value]) -> LoadReport {
                <$t as LoadState>::load_state(self, tick, values, inputs)
            }
            fn set_input(&mut self, pin: &str, value: &Value) -> Result<(), String> {
                <$t>::set_input(self, pin, value)
            }
            fn step_event(&mut self) -> Option<u64> {
                <$t>::step_event(self)
            }
            fn tick(&self) -> u64 {
                <$t>::tick(self)
            }
            fn next_event_tick(&self) -> Option<u64> {
                <$t>::next_event_tick(self)
            }
            fn values(&self) -> &[Logic] {
                <$t>::values(self)
            }
            fn read(&self, name: &str) -> Option<Value> {
                <$t>::read(self, name)
            }
            fn take_diagnostics(&mut self) -> Vec<Diagnostic> {
                <$t>::take_diagnostics(self)
            }
            fn native_active(&mut self) -> bool {
                <$t as LoadState>::native_active(self)
            }
            fn native_pending(&mut self) -> bool {
                <$t as LoadState>::native_pending(self)
            }
        }
    };
}

impl_fast_engine!(FastRef);
impl_fast_engine!(CompiledEngine);

impl LoadState for FastRef {
    fn load_state(&mut self, tick: u64, values: &[Logic], inputs: &[Value]) -> LoadReport {
        FastRef::load_state(self, tick, values, inputs)
    }
}

/// The per-type half of [`FastEngine`] (`load_state`, the native tier; the rest by the macro).
pub(crate) trait LoadState {
    fn load_state(&mut self, tick: u64, values: &[Logic], inputs: &[Value]) -> LoadReport;
    fn native_active(&mut self) -> bool {
        false
    }
    fn native_pending(&mut self) -> bool {
        false
    }
}

/// A background JIT compilation (its result: statistics or the reason it failed).
#[cfg(feature = "jit")]
pub(crate) type JitJob = std::thread::JoinHandle<Result<panda_jit::JitStats, String>>;
/// A background JIT compilation (never started without the `jit` feature).
#[cfg(not(feature = "jit"))]
pub(crate) type JitJob = std::thread::JoinHandle<Result<(), String>>;

/// [`FastSim`] (with its native tier for [`FastBackend::Jit`]). `FastSim::load_state` has
/// exactly [`FastRef::load_state`]'s semantics, so states load directly into it.
pub(crate) struct CompiledEngine {
    sim: FastSim,
    /// The background compilation of the native tier, until it finished.
    jit_job: Option<JitJob>,
}

impl CompiledEngine {
    fn set_input(&mut self, pin: &str, value: &Value) -> Result<(), String> {
        self.sim.set_input(pin, value)
    }
    fn step_event(&mut self) -> Option<u64> {
        self.sim.step_event()
    }
    fn tick(&self) -> u64 {
        self.sim.tick()
    }
    fn next_event_tick(&self) -> Option<u64> {
        self.sim.next_event_tick()
    }
    fn values(&self) -> &[Logic] {
        self.sim.values()
    }
    fn read(&self, name: &str) -> Option<Value> {
        self.sim.read(name)
    }
    fn take_diagnostics(&mut self) -> Vec<Diagnostic> {
        self.sim.take_diagnostics()
    }
}

impl LoadState for CompiledEngine {
    fn load_state(&mut self, tick: u64, values: &[Logic], inputs: &[Value]) -> LoadReport {
        self.sim.load_state(tick, values, inputs)
    }

    fn native_active(&mut self) -> bool {
        self.sim.poll_native_slot()
    }

    fn native_pending(&mut self) -> bool {
        match &self.jit_job {
            None => false,
            Some(h) if !h.is_finished() => true,
            Some(_) => {
                // Finished: on success the code waits in the engine's slot (install it now);
                // on failure the engine keeps interpreting.
                let job = self.jit_job.take().expect("checked");
                if matches!(job.join(), Ok(Ok(_))) {
                    self.sim.poll_native_slot();
                }
                false
            }
        }
    }
}

/// Kernel extraction + characterization results shared by the engines, STA and classification.
#[derive(Clone)]
pub(crate) struct Artifacts {
    pub(crate) kernels: Arc<KernelSet>,
    pub(crate) tables: Arc<Vec<Option<Arc<KernelTable>>>>,
}

/// The JIT tier of [`FastBackend::Jit`].
#[derive(Clone, Debug)]
pub(crate) struct JitTier {
    /// The compiler (shared by every compilation of one `Kernel`, so edits only recompile the
    /// partitions they change).
    #[cfg(feature = "jit")]
    pub(crate) jit: panda_jit::Jit,
    /// Compile on a separate thread and hot-swap (else compile before the engine is returned).
    pub(crate) background: bool,
}

impl JitTier {
    pub(crate) fn new(background: bool) -> JitTier {
        JitTier {
            #[cfg(feature = "jit")]
            jit: panda_jit::Jit::default(),
            background,
        }
    }

    /// Give `sim` its native tier; returns the background compilation, if one was started.
    fn attach(&self, sim: &mut FastSim) -> Option<JitJob> {
        #[cfg(feature = "jit")]
        {
            if self.background {
                // The engine keeps interpreting; the code is installed at an event boundary
                // after it is ready. A JIT failure (e.g. a program beyond its limits) leaves
                // the interpreter running.
                Some(self.jit.attach(sim))
            } else {
                // Functions the JIT cannot compile stay interpreted; a total failure leaves the
                // interpreter running.
                let _ = self.jit.install(sim);
                None
            }
        }
        #[cfg(not(feature = "jit"))]
        {
            let _ = (sim, self.background);
            None
        }
    }
}

/// Build (and power on) the engine of `backend` for `flat`.
pub(crate) fn build_engine(
    backend: FastBackend,
    flat: Arc<FlatNetlist>,
    opts: &FastOptions,
    compile: &FastCompileOptions,
    art: &Artifacts,
    jit: &JitTier,
) -> Result<Box<dyn FastEngine>, Vec<Diagnostic>> {
    match backend {
        FastBackend::Reference => {
            let e = FastRef::from_parts(
                flat,
                opts.clone(),
                (*art.kernels).clone(),
                (*art.tables).clone(),
            )?;
            Ok(Box::new(e))
        }
        FastBackend::Compiled | FastBackend::Jit => {
            let sopts = FastSimOptions {
                fast: opts.clone(),
                compile: compile.clone(),
            };
            let mut sim =
                FastSim::from_parts(flat, sopts, (*art.kernels).clone(), (*art.tables).clone())?;
            let jit_job = if backend == FastBackend::Jit {
                jit.attach(&mut sim)
            } else {
                None
            };
            Ok(Box::new(CompiledEngine { sim, jit_job }))
        }
    }
}
