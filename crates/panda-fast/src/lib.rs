//! panda-fast — the compiled Fast-mode engine (SPEC §7, §8.5–§8.9; docs/M4-CONTRACT.md).
//!
//! [`FastSim`] is observably identical to the reference engine `panda_compile::FastRef` after
//! every event (all net values, `tick()`, `next_event_tick()`, the same diagnostic kinds, and
//! the same event/round/transition/table-hit/fallback counts) but executes a compiled
//! [`Program`]:
//!
//! * [`ir`] — FastIR: 16-byte fixed-size ops over a flat byte-per-net state array, partitions as
//!   contiguous op ranges, LUT ops with cold gate code, per-kernel transition ops and the kernel
//!   specialization tables, all in flat arrays (no pointers: JIT-friendly). Next to the
//!   canonical ops it holds the interpreter's own encoding ([`ir::HOp`], built by [`fuse`]) and
//!   the per-net change-propagation records.
//! * [`compile`] — the compiler, a pipeline of passes (order → LUTs → partition → levelize →
//!   lower → notify → kernels → kernel specialization → layout → verify).
//! * [`lut`] — LUT extraction (SPEC §7.1): the Panda-8 ROM in every style becomes one table read;
//!   `X`/`Z` support runs the cone's gate code (exact 4-state semantics).
//! * [`fuse`] — gate fusion for the interpreter: fanout-1 gate chains with ≤ 6 inputs become one
//!   exact 4-state truth-table op (2:1 muxes, adder carry logic).
//! * [`kspec`] — kernel phase specialization (SPEC §8.5): KOpaque / KTransparent per clock tuple
//!   and per-input-vector actions, each proven from the kernel's table for every transition it
//!   can make; data transitions of O/T phases are provably order-insensitive, so FastRef's
//!   runtime order check is only needed for multi-input clock-tuple changes.
//! * [`FastSim`] — the round-faithful executor:
//!
//! ```text
//! event: write changed sources; mark their reader partitions dirty / reader kernels pending
//! loop {
//!     sweep dirty partitions in index (= topological) order (bitset scan); each partition's
//!       output changes are batched and propagated at its end from per-net records (reader
//!       partitions dirty, reader kernels pending / data-dirty, kernel input indices updated)
//!     if no pending kernel has a changed input vector: break
//!     round boundary: every such kernel transitions once with its current inputs:
//!       KOpaque → nothing observable changes (word-level bitset ops),
//!       KTransparent → output bits from a table over the input vector (word-level skip when
//!         its inputs did not change since its last evaluation in this phase),
//!       otherwise the table entry, or the on-the-fly KernelSim exactly as FastRef;
//!       changed outputs are written and propagated in one batch
//!     rounds > delta_limit → X + IterationLimit (as FastRef)
//! }
//! ```
//!
//! Kernels never run mid-sweep, so kernels see exactly the input sequences they see in FastRef.
//! Any topological order of the zero-delay combinational logic settles to the same values, so the
//! values after every sweep equal FastRef's after every `eval_comb`. LUT-internal nets, the
//! outputs of fused-away gates and the internal nets of specialized kernels are not maintained by
//! the hot path; `values()` / `get()` materialize them on demand, so they are always exactly
//! FastRef's.
//!
//! [`native`] is the hook of the native tier (SPEC §8.9): a backend (the Cranelift JIT in
//! `panda-jit`) compiles partitions and per-clock-tuple event functions to machine code over the
//! same runtime state; [`FastSim::install_native`] / [`FastSim::set_native_slot`] swap it in at
//! an event boundary, and the interpreter remains the reference and the fallback for anything
//! the native code does not handle.
//!
//! [`probe`] (feature `prof`) provides cycle / instruction-counter probes of the event loop.

pub mod compile;
mod exec;
pub mod fuse;
mod fxhash;
pub mod ir;
pub mod kspec;
pub mod lut;
pub mod native;
pub mod probe;
pub mod schedule;
mod sim;

pub use compile::{CompileOptions, OrderStrategy};
pub use ir::{Op, PhaseClass, Program, ProgramStats};
pub use native::{NativeCode, NativeSlot};
pub use sim::{FastSim, FastSimOptions, FastSimStats};
