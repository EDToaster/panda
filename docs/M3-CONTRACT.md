# M3 contract — panda-compile (kernels, FastRef) and panda-sta

Read SPEC.md §6 (kernel-delta semantics), §8.1–§8.4, §9 first. This file pins down module
boundaries and APIs so several agents can work in parallel.

## Crate dependency order
panda-netlist → panda-timing → panda-compile → panda-sta → (panda-fast, panda-kernel)

panda-compile depends on panda-timing: **kernel transitions are computed by running `TimingSim`
on a sub-netlist containing just the kernel** (its input nets become `TopInput` sources). This
guarantees `T` uses exactly Timing-mode semantics (incl. period-2 collapse) with zero duplication.

## panda-compile modules

### `graph`
Node = gate or resolver of a `FlatNetlist`. Utilities: successors/predecessors over nets, Tarjan
SCC (iterative, no recursion), topological order of a subset, bounded simple-cycle enumeration.

### `kernels`
```rust
pub enum KernelKind { Table, Sim, Keeper }
pub struct Kernel {
    pub id: u32,
    pub kind: KernelKind,
    pub gates: Vec<GateIdx>,          // gates inside (incl. absorbed private fan-in)
    pub resolvers: Vec<u32>,          // resolvers inside
    pub inputs: Vec<NetIdx>,          // nets read by the kernel, driven outside (sorted, deduped)
    pub outputs: Vec<NetIdx>,         // kernel nets read outside the kernel or top-level outputs
    pub nets: Vec<NetIdx>,            // all nets driven inside the kernel = the state
    pub owner: InstIdx,               // deepest instance containing every gate (for caching)
}
pub struct KernelSet {
    pub kernels: Vec<Kernel>,
    pub kernel_of_gate: Vec<Option<u32>>,     // per GateIdx
    pub kernel_of_resolver: Vec<Option<u32>>,
    /// Combinational gates/resolvers (not in a kernel) in topological order (kernel outputs and
    /// sources are the roots). Guaranteed acyclic.
    pub comb_order: Vec<Node>,
}
pub struct ExtractOptions { pub max_loop: usize /*6*/, pub max_bits: usize /*12*/, pub max_cycles_per_region: usize }
pub fn extract(flat: &FlatNetlist, opts: &ExtractOptions) -> KernelSet;
```
Algorithm SPEC §8.3: seed short cycles → merge → absorb private fan-in (bounded so the kernel
stays ≤ max_bits inputs+cut where possible) → residual SCCs become Sim kernels → keeper resolvers
become Keeper kernels. Constant-driven gates may be folded first (optional).

### `subnet`
`fn kernel_subnet(flat, kernel) -> (FlatNetlist, mapping)` — standalone netlist for one kernel.

### `charz` (characterization, SPEC §8.4)
```rust
pub struct KernelTable {
    pub num_inputs: usize,
    pub states: Vec<Vec<Logic>>,            // stable states (values of kernel.nets), all binary
    pub state_index: HashMap<Vec<Logic>, u32>,
    /// transitions[state * 2^k + input_vector] (input_vector bit i = kernel.inputs[i])
    pub transitions: Vec<Transition>,
}
pub struct Transition { pub next: Option<u32> /* None = non-binary result */, pub latency: u32, pub order_sensitive: bool }
pub fn characterize(flat, kernel, opts) -> Option<KernelTable>   // None if too big → Sim
```
Kernel tables are cached by a structural hash of the kernel sub-netlist (identical kernels —
e.g. 200 D-latches — characterize once).

### `fastref` — the FastRef engine (SPEC §6.2), the M3 oracle
```rust
pub struct FastOptions { pub delta_limit: u32 /*64*/, pub extract: ExtractOptions }
pub struct FastRef { .. }
impl FastRef {
    pub fn new(flat: Arc<FlatNetlist>, opts: FastOptions) -> Result<FastRef, Vec<Diagnostic>>;
    pub fn reset(&mut self);                         // all X; then processes the tick-0 event
    pub fn set_input(&mut self, pin: &str, v: &Value) -> Result<(), String>; // queued
    /// Process one event: pending input changes if any (as one event at the current tick), else the
    /// next scheduled source event (clock edges / POR release). Returns the event tick, or None if
    /// there is nothing to do (no clocks, no pending inputs).
    pub fn step_event(&mut self) -> Option<u64>;
    pub fn run_events(&mut self, n: u64);
    /// Tick of the most recently processed event. Values correspond to Timing mode at
    /// (next_event_tick - 1) under the §6.3 conditions.
    pub fn tick(&self) -> u64;
    pub fn next_event_tick(&self) -> Option<u64>;
    pub fn get(&self, net: NetIdx) -> Logic;
    pub fn values(&self) -> &[Logic];
    pub fn read_pin(&self, pin: &str) -> Option<Value>;
    pub fn read(&self, name: &str) -> Option<Value>;
    pub fn take_diagnostics(&mut self) -> Vec<Diagnostic>;
    pub fn kernels(&self) -> &KernelSet;
    pub fn stats(&self) -> FastRefStats;             // events, rounds, table hits, sim fallbacks
}
```
Schedule: event ticks = 0, every tick where any Clock output differs from the previous tick, the
PowerOnReset release tick. Simultaneous changes = one event.

## panda-sta (SPEC §9)
```rust
pub struct TimingReport {
    pub min_period: Vec<(String /*clock net name*/, u32)>,
    pub phase_settle: Vec<PhaseSettle>,  // per distinct clock tuple: worst settle ticks after the event
    pub diagnostics: Vec<Diagnostic>,    // SetupViolation / HoldViolation / GuardHazard / DelaySensitive(advisory)
    pub critical_path: Vec<String>,      // net names along the worst setup path
}
pub fn analyze(flat: &FlatNetlist, kernels: &KernelSet, tables: &[Option<KernelTable>]) -> TimingReport;
```
Must be conservative: if STA reports no violations and the clock period ≥ min_period, FastRef and
Timing must agree at phase boundaries (this is what the differential fuzzer checks).
