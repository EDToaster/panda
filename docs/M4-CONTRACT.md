# M4 contract — panda-fast (compiled Fast mode)

Read SPEC.md §6 (kernel-delta semantics), §7, §8.5–§8.9, §14.3, and docs/M3-CONTRACT.md.

## Goal
A compiled engine whose observable behaviour is **bit-identical to FastRef** after every event
(all net values, `tick()`, `next_event_tick()`; same diagnostic kinds per event), and which runs
Panda-8 (SumOfProducts ROM) at **≥ 1 MHz** in Timing-equivalent clock cycles on one core of a
Ryzen 7 3700X (release build). FastRef stays the oracle; `panda-difftest`'s generators drive the
differential tests.

## Public API (mirror FastRef so the facade can swap engines)
```rust
pub struct FastSim { .. }
impl FastSim {
    pub fn new(flat: Arc<FlatNetlist>, opts: FastSimOptions) -> Result<FastSim, Vec<Diagnostic>>;
    // same methods as panda_compile::fastref::FastRef:
    // reset, set_input, step_event, run_events, tick, next_event_tick, get, values, read_pin,
    // read, take_diagnostics, stats
    pub fn run_cycles(&mut self, clock_pin_or_net: &str, cycles: u64); // convenience for benches
    pub fn program(&self) -> &Program;   // introspection (op counts, partitions, LUTs)
}
```

## Execution model — round-faithful (exact by construction)
§6.2 is executed literally, only cheaper:
```
event: write changed sources; mark their reader partitions dirty
loop {
    sweep dirty partitions in topological index order (bitset scan); each partition evaluates its
      ops; for every partition output whose value changed: mark reader partitions dirty and reader
      kernels pending
    if no kernel pending: break
    round boundary: every pending kernel transitions once with its current inputs
      (specialized op, see below); outputs that changed mark reader partitions dirty
    rounds > delta_limit → same X/IterationLimit behaviour as FastRef
}
```
Combinational partitions only contain gates/resolvers/LUT ops; kernels never run mid-sweep. This
reproduces FastRef's round structure exactly (kernels see exactly the same input sequences).

### Allowed optimization: single-pass phases
For a clock tuple `c`, kernel K is **robust** if for all stable states s consistent with c and
all dynamic input vectors i1, i2: `T(T(s,i1),i2) = T(s,i2)` (checked from the table). If every
kernel that can transition in phase c is robust and the phase graph (opaque kernels cut) is
acyclic, kernels may be evaluated inline in topological order and the round loop collapses to
one sweep — the fixpoint is unique and equal to FastRef's. Gated clocks (conditional guards) are
generally not robust → round-faithful for those phases. Implement only if needed for the target,
and prove equality by differential tests.

### Kernel ops (per kernel × clock tuple, chosen at compile time from the table)
- `KTable` — general: index = (state id, input vector) → next state id + output values. Falls back
  to `KernelSim` (panda_compile::charz) whenever state or inputs contain X/Z or no table exists —
  exactly as FastRef.
- `KOpaque` — table proves outputs never change in this phase for any dynamic input: the op only
  updates the state id (needed so later phases index correctly). If robust, the state update may
  be deferred (store inputs, compute on next phase change).
- `KTransparent` — table proves state' depends only on inputs (a pure function, e.g. Q = D):
  compile the output function as gate ops / small LUT; state id recomputed lazily.
- Groups of identically-specialized kernels with word-aligned inputs (e.g. the 8 slave latches of
  a register on the same clock) MAY be packed into word ops (slice packing, SPEC §8.7).
Every specialization must give exactly the table's result for every single transition.

## Combinational compilation (SPEC §7)
- Values: one byte per net holding the `Logic` code (same layout as FastRef `values()`), so the
  differential check is a memcmp. Word-packed slots are an internal optimization; values() must
  still be exact (materialize on read).
- Scalar ops: branchless 2-bit-code tables (`eval::tables()` / `lookup2`), N-input folds, Tri,
  resolvers (`eval::resolve`, keeper is a kernel already).
- LUTs (SPEC §7.1): multi-output cones whose support (root nets: sources, kernel outputs, LUT
  outputs) has ≤ `lut_max_bits` (default 12) bits and absorb ≥ `lut_min_gates` gates (default ~24),
  internal nets not read outside the cone (MFFC-style; group outputs with identical support). The
  Panda-8 ROM (8 address bits → 8 data bits) MUST become a single LUT. If any LUT input is X/Z, run
  the cone's gate code instead (kept as a cold partition) — exact X semantics. Internal cone nets
  are materialized lazily for values()/get() (compute from the cone's gate code on demand).
- Partitions: acyclic, topologically ordered, ~50–200 ops each, with a dirty bit; built so that
  each partition is evaluated at most once per sweep.
- Bytecode: compact fixed-size instructions over a flat state array, designed to be JIT-compiled
  by Cranelift in M5 (no pointers, no closures; partitions = contiguous op ranges).

## Benchmarks (required)
`crates/panda-fast/benches` or an #[ignore] test / example binary printing, for Panda-8
(SumOfProducts, MuxTree; Hierarchical, Flat, GatedClockRegfile) running bench_loop:
cycles/s, ns/cycle, partitions evaluated per cycle, ops executed per cycle, kernel transitions
per cycle — for FastRef and FastSim. No profiler is installed (no perf/valgrind); measure with
counters and microbenchmarks.

## Facts from M3 (existing APIs to build on)
- `panda_compile::{compile_kernels, compile_kernels_with_cache, KernelOptions, FastOptions, FastRef}`;
  `kernels::{KernelSet, Kernel{kind,gates,resolvers,inputs,outputs,nets,cut,owner}, Node}` with
  `comb_order` (topological, acyclic) and `kernel_of_gate/resolver`.
- `charz::KernelTable { transition(state,input)->&Transition{next:Option<u32>,latency,order_sensitive}, lookup(&[Logic])->Option<u32> (state id), states_by_input, home_input, is_order_sensitive(state,u,v), is_oscillator() }`,
  `charz::{input_index, input_vector}` (input bit i = kernel.inputs[i]), `KernelSim` (on-the-fly
  transitions incl. X; keeps diagnostics), `TableCache`.
- FastRef semantics details to mirror exactly: read crates/panda-compile/src/fastref.rs (round
  structure, which kernels transition, DelaySensitive dedupe via `is_order_sensitive(state,u,v)`,
  Oscillating/IterationLimit handling, power-on first transition, diagnostics forwarding).
- Differential harness: crates/panda-difftest (`generator::generate(seed,&GenConfig,scale)`,
  `harness`, `driver`, `cpu8::run_cpu8`). For FastSim vs FastRef you can compare after every
  event directly (same event sequence, same inputs) — simpler than the Timing comparison.

## panda-kernel facade (built in parallel, against FastRef first)
SPEC §10 API. Engines behind a trait (Timing = TimingSim, Fast = FastRef now; FastSim plugs in
when ready via `FastBackend::{Reference, Compiled}`). Incremental edits: `EditOp` over the
hierarchical `Design` (add/remove instance, connect/disconnect endpoints, add/remove net,
change atom params, add/remove/replace def) → revalidate → reflatten → recompile only defs whose
content hash changed (TableCache keyed by structural hash already makes kernel tables reusable);
state carried across by net name. Snapshots (`State`) restore by net name. Mode switch Timing →
Fast at quiescence (load kernel states from net values), Fast → Timing (load all net values via
force). Background compilation: compile on a worker thread; until ready, Fast mode requests run
on the previous program or FastRef. Diagnostics unified.
