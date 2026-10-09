# Panda — Digital Logic Sandbox: Design Spec

> Status: **v0.4**: M1–M4 implemented, M5 (JIT) in progress. v0.3 defined Fast mode by the kernel-delta reference algorithm (§6); v0.4 records implementation outcomes (§0, §11, §14.5).
> Implementation language: **Rust**.
> Conventions: **MUST / SHOULD / MAY** are normative. `[OPEN]` marks unresolved questions.
> Evidence behind each decision is summarized in §14.

---

## 0. Decisions Log

| # | Topic | Decision |
|---|-------|----------|
| D1 | Signal values | 4-state: `0`, `1`, `X`, `Z` |
| D2 | State primitives | **Purist**: no latch/FF/RAM atomics. All state arises from gate feedback. |
| D3 | Simulation modes | **Timing mode** (unit-delay, event-driven, what you see when stepping) and **Fast mode** (compiled, synchronous, clock-cycle granularity). |
| D4 | Clocks | Users place `Clock` sources and wire them like inputs. |
| D5 | Memory | No RAM/ROM atomics. Gate-built ROM compiles to real tables (§7.4); gate-built RAM is made cheap by guarded skipping (§8.6). |
| D6 | Product | Open sandbox tool. |
| D7 | Perf target | Reference gate-only CPU (§12) at **≥ 1 MHz** in Fast mode. |
| D8 | Language | Rust. Kernel is a UI-free library; renderer built on top later. |
| D9 | Build order | Kernel first (reference sim → compiler → fast engine → JIT), editor UI last. |
| D10 | Fast ↔ Timing link | Fast mode is **defined** by synchronous semantics and **equals** Timing mode whenever static timing checks pass (§9). |
| D11 | Packaged combinational components | **Ideal blocks**: glitch-free, per-output fixed latency, compiled. |
| D12 | Fast-mode reference | FastRef, a direct interpreter of §6.2, is the oracle. Compiled engines must match it bit-for-bit, including diagnostics kinds per event. |
| D13 | Compiled execution | Round-faithful (kernels transition only at round boundaries). Kernel ops are specialized per clock phase into opaque/transparent fast paths with exactness proven from the tables. Single-pass phases are not used. |
| D14 | Reference CPU ROM | Panda-8 default ROM is a sum-of-products decoder (3,671 gates total). The literal mux-tree ROM (7,726 gates) remains a variant. Both compile to LUTs. |

---

## 1. Goals / Non-Goals

**Goals**
- Place inputs, outputs, gates on a canvas; wire; click inputs to test.
- Package any circuit as a reusable component; auto-classify combinational vs sequential; compile both.
- Arbitrary-width buses.
- A purist, gate-only CPU runs at MHz speed — something no existing hobby tool does (§14.1).

**Non-Goals (v1)**: analog/electrical effects, real ns timing, HDL import/export, memory atomics, multi-threaded simulation (use extra cores for UI and background compilation instead; §14.3).

---

## 2. Architecture

```
┌──────────────┐  ┌───────────────┐  ┌────────────────────┐  ┌──────────────────────┐
│ Editor / UI  │→ │   Netlist     │→ │ Compiler           │→ │ Engines              │
│ (later)      │  │ hierarchical, │  │ flatten, classify, │  │ Timing: event-driven │
│ geometry     │  │ serializable  │  │ kernels, phases,   │  │ Fast:   bytecode     │
└──────────────┘  └───────────────┘  │ partitions → IR    │  │         interpreter  │
                                     └────────────────────┘  │         + Cranelift  │
                                                             └──────────────────────┘
```

### 2.1 Crates

| Crate | Role |
|---|---|
| `panda-netlist` | Data model, serialization, validation, width inference, edit operations |
| `panda-timing` | Timing-mode engine (reference semantics, §5) |
| `panda-compile` | Analysis & lowering: classification, kernel extraction, characterization, phase compilation, STA, partitioning → `FastIR` |
| `panda-fast` | `FastIR` bytecode interpreter |
| `panda-jit` | `FastIR` → Cranelift native code (tier 2) |
| `panda-kernel` | Facade: the API in §10; owns engines, background compilation, hot-swap |
| `panda-cli` | Run netlists, test vectors, VCD dump, benchmarks, differential fuzzing |
| `panda-ui` | Editor/renderer (M6; tech choice deferred — egui/wgpu, Bevy, or web over WASM) |

---

## 3. Signal Model

### 3.1 Values and encodings
Each bit ∈ {`0`,`1`,`X`,`Z`}; bit 0 = LSB. Two in-memory encodings, chosen per operation:

- **Scalar (1-bit nets)** — 2-bit code `0=0, 1=1, 2=X, 3=Z`. A 2-input gate is a 16-entry table of 2-bit results packed in a `u32`:
  `out = (TT >> ((a | b << 2) * 2)) & 3`. Branchless; measured ≈ 10% slower than 2-state (§14.3). N-input gates lower to chains of 2-input ops.
- **Word (buses, sliced logic)** — two bit-planes `(a, b)`: `0=(0,0) 1=(1,0) Z=(0,1) X=(1,1)`, packed in `u64`. Formulas in Appendix A.

There is **no separate 2-state code path**: the measured gain did not justify it.

### 3.2 Gate semantics
- Inputs reading `Z` are treated as `X`.
- Gates are X-pessimistic per gate (Verilog-style): `X AND 0 = 0`, `X AND 1 = X`.
- Optimizations MUST reproduce per-gate semantics exactly, including for X. In particular LUTs index only fully-known inputs and fall back to gate code otherwise (§7.2).

### 3.3 Nets, drivers, resolution
- A net MAY have several drivers. Per bit: drop `Z` drivers → none left: apply the net's **termination** (below) → otherwise all agree and known: that value → else `X` + **contention** diagnostic.
- **Net termination** (net attribute, default `none`):
  - `none` — undriven reads `Z`
  - `pull-up` / `pull-down` — undriven reads `1` / `0`
  - `keeper` — undriven holds its last driven value (power-on `X`)
  This models weak drivers without adding a 5th value or drive strengths.
- An output pin is **Z-capable** if a `TRI` reaches it without passing a gate. The editor warns when a net has ≥ 2 drivers and any is not Z-capable.
- Unconnected inputs read `Z` (→ `X` at the gate).

---

## 4. Components

### 4.1 Atomics

| Category | Atomic | Delay (ticks) | Notes |
|---|---|---|---|
| Logic | `AND OR NAND NOR XOR XNOR` | 1 | N inputs, width-polymorphic (bitwise); optional reduce mode (W-bit → 1-bit) |
| | `NOT BUF` | 1 | width-polymorphic |
| Tri-state | `TRI` | 1 | `en=1 → d`; `en=0 → Z`; `en ∈ {X,Z} → X` |
| Wiring | `Splitter`, `Joiner` | 0 | one W-bit end ↔ bit-range ends |
| | `Constant` | 0 | W-bit value; may contain `X`/`Z` |
| Sources | `Input` | 0 | user-driven; toggle / momentary / typed value |
| | `Clock` | — | `period` (ticks, even, default = auto from STA §9.3), `duty`, `phase`, `enabled` |
| | `PowerOnReset` | — | drives `1` for `N` ticks / `N` cycles after reset, then `0` |
| Sinks | `Output` | — | LED / binary / hex / decimal |
| | `Probe` | — | display only |

None hold state. `Clock` and `PowerOnReset` are sources, so they don't break the purist rule.

### 4.2 Width rules
Pin widths are generic and inferred from connected nets. A bitwise gate's data inputs and output share a width. Mismatches are edit-time errors.

### 4.3 Packages
- A **ComponentDef** = name + ordered pins (`name`, `dir ∈ {in, out, inout}`, `width`) + implementation (atomic or netlist of instances).
- `inout` pins are required for packaging bus transceivers. Internally an `inout` pin is a net shared across the boundary, and resolution happens on the flattened net.
- `Input`/`Output` atomics inside a def become its pins.
- **No recursion**: the def-dependency graph MUST be acyclic.
- **Live link**: edits propagate to all instances. Interface changes mark instances broken until fixed. Defs are content-hashed; every compiler artifact is cached by hash.
- **Clocks and `PowerOnReset` are top-level only.** Packages take them as ordinary input pins. This keeps packages reusable and makes clock nets identifiable after flattening (§8.2).

### 4.4 Package classification (shown in the UI)

| Class | Rule |
|---|---|
| **Combinational** | Flattened gate graph is acyclic, **or** cyclic but *ternary-combinational*: for every binary input, ternary simulation from all-`X` reaches all-binary values (Malik 1994). |
| **Sequential** | Otherwise. Reports state bits, kernels found, and *clock-like pins* (§8.3). If edge fusion applies: "edge-triggered on ↑CLK". |
| **Oscillator** (flag) | Some binary input vector gives a kernel **no binary stable state** (e.g. a ring oscillator). Detected statically; it does not depend on the power-on state. |

A NAND SR latch is **not** flagged as an oscillator. It has stable states for every input; only the simultaneous release is metastable (§5.3).

---

## 5. Timing Mode — reference semantics

### 5.1 Model
- Discrete ticks. Every logic/TRI atomic has delay 1. Wires and wiring atomics have delay 0.
- **Synchronous update** (transport delay): at tick t+1 each gate output equals its function of its inputs at tick t.
- Event-driven: only gates with changed inputs are evaluated.
- This is what the user sees when stepping and probing. It shows glitches, races and oscillation.

### 5.2 Packaged combinational components are ideal blocks (D11)
A combinational package instance behaves as **out_j(t + L_j) = f_j(in(t))**, where L_j is the max gate depth of output j's cone. It is evaluated through its compiled form (§7), and glitches inside it are not produced.
- Rationale: packaging is abstraction. Hiding internal hazards makes packaged behavior robust and makes it fast.
- **Glass-box toggle** (per instance or global): simulate the package's gates instead. Use it to study hazards.
- **Probing inside an ideal block** shows the internal nets evaluated at the block's current settled inputs. Probing never changes behavior: there is no deoptimization side effect.
- Sequential packages are always simulated at gate level in Timing mode. Their speedup comes from Fast mode.

### 5.3 Period-2 collapse (metastability rule)
Under exact simultaneity, unit-delay logic produces period-2 oscillation where real hardware would settle unpredictably. Examples: a NAND SR latch released 00→11; a master-slave FF whose D and clock change on the same tick (§14.4).
- **Rule:** if, with no source change, a set of nets alternates with period 2 for `P2_LIMIT` consecutive cycles (default 4), those nets are set to `X` and a **metastable** diagnostic is raised. The `X` then propagates normally. Cross-coupled gates with an `X` input settle at `X`, which gives a stable, honest result.
- Oscillation with period > 2 (e.g. a 3-inverter ring) is real and is simulated as-is, with an **oscillating** diagnostic after `OSC_LIMIT` ticks.
- The rule is deterministic and seed-free, unlike the random reordering used by Digital, Logisim and Digital Logic Sim (§14.1).

### 5.4 Power-on state
- Default: every net `X`. An un-reset toggle FF stays `X`. That is correct; the fix is a reset, and `PowerOnReset` exists for it. A **stuck-X** diagnostic fires when state is still `X` after N cycles of clock activity, with an **X-source trace** that walks back to the originating kernel.
- Option `random(seed)`: pick random values for **kernel state nets only** (§8.4), then settle. Never randomize all nets: symmetric starts oscillate (§14.4).

---

## 6. Fast Mode — kernel-delta semantics

Fast mode is defined by a simple reference algorithm. Every optimization (§7, §8.5–§8.9) MUST produce identical results to it, and differential tests enforce this. Its relationship to Timing mode is the equivalence contract below.

### 6.1 Ingredients
- **Kernels** (§8.3): small state-holding gate clusters found by the compiler. Each kernel has input nets, output nets, and a **state**: the values of all nets inside it. A kernel's **transition** `T(state, inputs) → state'` is "apply the new inputs all at once and simulate the kernel's gates with Timing-mode semantics (§5.1, including period-2 collapse) until quiescent". Characterized kernels cache `T` in a table (§8.4); others compute it on the fly.
- **Combinational logic**: every gate/resolver not inside a kernel. After kernel extraction, every cycle of the netlist passes through a kernel, so this logic is acyclic given kernel outputs. It is evaluated with **zero delay**: settled values equal the gate functions applied in topological order, which is exactly what unit-delay acyclic logic settles to.
- **Events**: clock edges, `PowerOnReset` release, and UI input changes. The schedule is the merged timeline of all sources (§8.2). Simultaneous events form one event.

### 6.2 The algorithm (per event)
```
given: settled state S (all nets, all kernel states), the event's new source values
round 0:  evaluate combinational logic with new sources, kernel outputs taken from current states
repeat:   for every kernel whose input vector changed since its last transition:
              state ← T(state, inputs)
          re-evaluate combinational logic downstream of kernels whose outputs changed
until no kernel input changed            (bound: DELTA_LIMIT rounds → affected nets X + IterationLimit)
publish S
```
- One **phase** = the interval between consecutive events. `step(Phase)` processes one event.
- UI input changes are applied as their own event right after the next clock event, never simultaneously with it. This mirrors "inputs change after the edge" (hold).
- Initial state: all nets `X`. Kernels with `X` in their state or inputs use on-the-fly transitions (exact; tables index binary values only).

Why this matches hardware intuition: at an edge, the clock-derived kernel inputs change first (round 0 → first transitions), and data that depends on kernel outputs changes afterwards (later rounds). That is exactly the order a hold-time-clean circuit guarantees. A master-slave register costs one round at each edge. Combinational logic runs once per cycle, after the edge that opens the slave latches, because the master latches are opaque when its outputs arrive.

### 6.3 Equivalence with Timing mode (D10)
When Timing mode runs with the clock period ≥ the STA minimum and UI inputs change outside hold windows, Fast mode's values at the end of each phase equal Timing mode's values at the tick just before the next event, **for every net**, provided:
1. **Setup**: every phase settles before the next event (§9.1).
2. **Hold**: at every kernel, data-input changes caused by this event arrive after its clock-derived input changes have settled (§9.2). Rounds model exactly this ordering.
3. **No hazards** on kernel inputs that can change kernel state (guards, SR inputs): their cones produce no glitch for the event (§9.3). Glitches on data inputs of opaque kernels are harmless, and transparent kernels pass a glitch through but then settle.
4. **Order-insensitive bursts**: when several inputs of one kernel change in the same round, the table entry gives the same result for every arrival order (§8.4). Otherwise the result depends on unit-delay skew. Fast mode raises `DelaySensitive` at runtime when such an entry is used.
5. No free-running oscillator kernels. Those yield `X` plus `Oscillating` in Fast mode.

Each violated condition raises a diagnostic with a location. Switching to Timing mode shows the real behavior.

---

## 7. Combinational Compilation

### 7.1 Strategy (per partition)
1. **Constant folding, dead-logic removal, structural hashing** (dedup identical subgraphs).
2. **Slice packing (§8.7)**: map bit-parallel structure onto word ops.
3. **LUT** only for cones with few inputs and many gates: `inputs ≤ LUT_MAX_BITS` (default 12–16) **and** gates absorbed ≥ `LUT_MIN_GATES`. Typical: ROM mux trees, decoders, 7-segment encoders. LUTs are not a general strategy: a measured 4–6-input LUT costs ~4 gate-ops (§14.3).
4. Everything else: **levelized straight-line ops** (scalar table ops / word ops).

### 7.2 4-state LUTs
LUTs are indexed by binary inputs. If any input is `X`/`Z`, the partition runs its gate-level code instead. This keeps exact per-gate X semantics.

### 7.3 Latency
Per output: max gate depth of its cone. Used by ideal blocks (§5.2) and STA (§9).

### 7.4 Consequence: gate-built ROM becomes real memory
A ROM built from `Constant`s and a mux tree is combinational with only address inputs. It compiles to a single table load, even though it is thousands of gates. A **ROM generator** (`panda-cli` tool and later a UI action) builds such a package from a hex file. Its output is ordinary gates, so it stays purist.

---

## 8. Sequential Compilation (Fast mode)

All state comes from gate loops, so the compiler has to *rediscover* state elements. It must not pattern-match known latch topologies: anything a user invents has to work. Correctness never depends on heuristics. Heuristics only affect speed.

### 8.1 Pipeline
```
flatten (keep hierarchy map) → const-fold
  → kernel extraction (8.3) → kernel characterization (8.4)
  → FastRef: direct interpreter of §6.2                      [M3: the oracle]
  → phase specialization (8.5) → STA (9)
  → partitioning + guards (8.6) → slice packing (8.7) → LUTs (7) → FastIR   [M4]
```
Order matters. Running SCC analysis on the raw flattened graph yields **one giant SCC** in any CPU, because register → ALU → register loops close through the latches (§14.4). Kernels have to be found locally first.

### 8.2 Clock nets
Clock nets are those driven by `Clock` sources through buffers/inverters only. Nets that are functions of clocks plus other signals (e.g. `clk AND load`) are **gated clocks**. They are handled as kernel guards in §8.5 and need no special casing.

### 8.3 Kernel extraction
1. **Seed loops**: find all simple cycles of length ≤ `KERNEL_MAX_LOOP` (default 6). Cross-coupled pairs have length 2.
2. **Merge** overlapping cycles into a kernel core.
3. **Absorb private fan-in**: add every gate whose fan-out lies entirely inside the kernel (local clock inverters, the `s`/`r` steering NANDs of a D latch). Without this, a D latch's core sees correlated inputs `s`, `r` that can't be classified, and a toggle FF sees `clk`/`nclk` as independent (§14.4).
4. **Residual cycles**: compute SCCs of the graph with all kernels contracted. Each remaining non-trivial SCC becomes a **sim kernel** (it also absorbs private fan-in). Afterwards every cycle passes through a kernel, so combinational logic is acyclic. Long-loop latches, cyclic-combinational logic and anything unusual land here: correct, just slower.
5. **Keeper-terminated resolvers** hold state, so they become 1-bit kernels.
6. Kernels whose `binary inputs + cut bits ≤ KERNEL_MAX_BITS` (default ≈ 12) are **characterized** (§8.4). The others are sim kernels: `T` is computed on the fly by unit-delay simulation of the kernel's gates (bound = 4 × net count + 8, then `X` + `IterationLimit`).

Correctness never depends on which kernels are found: `T` is exact either way. Heuristics only decide what is fast.

**Clock-like pins** of a package: pins p such that, for each constant value of p, the package outputs do not depend on its data inputs within a phase.

### 8.4 Kernel characterization (Latch Tables)
- **State** = values of *all* nets in the kernel. Absorbed fan-in nets are functions of the inputs, so a state implicitly records the inputs it settled under. This is why results never depend on "previous inputs" (§14.4), and why power-on and saved state are independent of netlist order. The cut set is only used to enumerate states.
- **Stable-state enumeration**: for each binary input vector, assign each binary value to the cut nets, settle with Timing semantics, and keep the binary fixed points. Stable states get dense ids.
- **Transitions**: for every (stable state, new binary input vector), run `T` and record the next state id (or *non-binary*: it collapsed or oscillated to `X`), the output values, and the settle latency per output (used by STA).
- **Order sensitivity**: for entries where j ≥ 2 inputs change, also apply the changes one at a time in every order (each settling before the next; j! ≤ 24 for j ≤ 4, else sampled). If any order disagrees with the simultaneous result, mark the entry **order-sensitive** (§6.3 condition 4). Kernel-internal skew (e.g. a local clock inverter) is modeled exactly because it lies inside the kernel. This is why ternary simulation is *not* used for this check: it treats internal delays as unbounded too, and so flags every toggle-FF edge (§14.4).
- **Ternary simulation** (Eichelberger A+B) is still run per entry as an *advisory* "would race under arbitrary gate delays" lint (`DelaySensitive`, info severity, off by default in the UI).
- At runtime, a kernel with `X` anywhere in its state or inputs uses the on-the-fly `T`. When it returns to a binary stable state it re-enters table mode (look up its id by state hash).
- Tables are cached per (def hash, kernel position) when the kernel lies entirely inside one def. That is the common case, since most users package latches/FFs.

### 8.5 Phase specialization

**8.5.1** For each distinct tuple of clock values, restrict every kernel table to those clock inputs and classify it:

| Class | Meaning | Compiles to |
|---|---|---|
| **Opaque** | next state = state even with data inputs `X` | register read; **breaks cycles** |
| **Transparent** | outputs binary and state-independent for all binary data | combinational function |
| **Conditional** | depends on a non-clock guard (gated clock, enable) | `guard ? f(data) : state` |
| **General** | none of the above (SR/JK behavior, async set/reset) | table step |

**8.5.2** Build the per-phase graph with opaque kernels as cut points and check it is **acyclic at the kernel level**. Gate-level constant propagation alone doesn't break latch loops (§14.4). Residual cycles become iterative kernels with a diagnostic.

**8.5.3 Edge fusion** (semantic, not topological; *optional optimization*, M4+). Under §6.2 with dirty tracking, combinational logic already runs once per cycle. Fusion only removes the master-latch round at the inactive edge. For each group of kernels on the same clock, compose their per-phase tables over one clock cycle. If the composition observed once per cycle equals "sample on edge, hold otherwise", replace the group with an **edge-triggered register**. If an entire clock domain fuses, its phases collapse into a single **cycle program**: evaluate combinational logic once, then commit registers.
- Fused state stays probe-able. A master latch's value is reconstructable from the slave and its D input, since it is transparent in one phase and holds the slave's value in the other.

**8.5.0 Purpose.** Phase specialization turns kernel tables into *bitwise code*. Under fixed clock values, a transparent latch is a copy and a conditional latch is a mux (`q = g ? d : q`). These pack into word ops across a register's bits (§8.7). An 8-bit register's slave latches at the rising edge become one 8-bit word copy instead of 8 table lookups. Kernels classified *general* keep table steps.

**8.5.4 Derived clocks** (ripple counters, clock dividers): a guard that depends on kernel state can't be fixed per phase. These kernels are evaluated with **delta iteration**: after a phase program runs, any changed guard re-triggers the dependent kernels, in topological order of guard dependence, up to a bound. Beyond the bound → `X` + diagnostic. It is correct, but slower in proportion to how much logic is derived-clocked.

### 8.6 Partitions and guarded skipping
- Partition the phase/cycle program into acyclic **partitions of ~50–200 ops** (ESSENT-style). Each has a change flag set when any of its inputs changes, and unchanged partitions are skipped. Per-gate activity tracking is explicitly rejected: it costs more than it saves (§14.3).
- **Dynamic don't-cares** at partition granularity:
  - a register or conditional kernel whose guard is closed doesn't propagate data-input changes;
  - a mux's unselected input doesn't mark it dirty.
- Effect: a write to one word of a gate-built N-word register file activates the decoder plus one word, and a read walks one mux path. This is the replacement for RAM atomics.

### 8.7 Slice packing
Users wire mostly 1-bit gates (e.g. eight full-adder instances), so word parallelism has to be recovered:
- Detect **isomorphic slices**: sets of structurally identical subgraphs whose corresponding inputs and outputs are the bits of the same buses (register bits, bus muxes, bitwise ALU ops). Hierarchy helps: the same def instanced across bus bits is the strong signal.
- Evaluate each slice set as one word op per gate, in two-plane encoding.
- **Not** applied to unrelated gates: gathering inputs costs more than it saves (§14.3).
- Serial chains such as ripple carry stay scalar.

### 8.8 Runtime loop
```
for each phase (or cycle, if fused):
    apply clock tuple; latch pending UI input changes
    for partition in schedule: if dirty → run (scalar table ops / word ops / LUT reads)
    delta-iterate derived-clock kernels (8.5.4)
    commit registers; publish observed nets to UI (throttled to frame rate)
```

### 8.9 Backends (tiered)
- **Tier 1 — interpreter**: compact bytecode, struct-of-arrays state, branchless table ops. Available immediately after any edit.
- **Tier 2 — Cranelift JIT**: compiles each partition to a native function in the background and **hot-swaps it at a cycle boundary**. Both tiers share the same flat state arrays, so swapping is free. Only edited partitions recompile.
- Code size note: 5k gates of straight-line code ≈ 60–100 KB, which exceeds L1I. Partition skipping reduces fetch pressure; keep partitions contiguous in schedule order.
- A WASM-emitting backend (`wasm-encoder`) is the equivalent of tier 2 for a future browser build.

---

## 9. Static Timing Analysis

STA is what makes Fast mode trustworthy (D10). It runs on the per-phase graphs in gate-tick units.

1. **Setup**: the longest path from {inputs, opaque/register outputs} to {kernel data inputs, outputs} ≤ the phase duration. Report the critical path. Time borrowing through transparent latches uses the SMO model.
2. **Hold**: the shortest clock-to-data path at each kernel must exceed the clock skew at that kernel (difference in clock arrival between launching and capturing kernels). In the prototype, D changing on the same tick as `clk↑` gave L0 ≠ Fast in 104 of 400 phases. Those are exactly hold violations (§14.4).
3. **Guard hazards**: a ternary hazard check (single-input change → transient `X`) on every cone that drives a kernel guard. Glitchy enables and pulse-generated clocks (`clk AND NOT delay(clk)`) are reported as "not Fast-mode compatible".
4. **Clock period**:
   - The default Clock period is the STA minimum, rounded up.
   - A user-set period below it raises a warning: *Fast mode assumes timing closure; Timing mode will differ.*
   - The editor shows: `min period: 46 ticks (critical path: ALU → R3.D)`.

---

## 10. Kernel API (sketch)

```rust
pub struct Kernel { /* engines, caches, background compiler */ }

impl Kernel {
    pub fn load(&mut self, netlist: Netlist) -> Result<(), Vec<Diagnostic>>;
    pub fn edit(&mut self, ops: &[EditOp]) -> Result<(), Vec<Diagnostic>>; // incremental
    pub fn set_mode(&mut self, mode: Mode);                 // Timing | Fast
    pub fn set_input(&mut self, pin: PinRef, value: &Value);
    pub fn step(&mut self, n: u64, unit: StepUnit);         // Tick | Phase | Cycle
    pub fn run(&mut self, budget: Duration) -> RunStats;    // run until budget expires
    pub fn read(&self, net: NetPath) -> Value;              // works inside packages
    pub fn diagnostics(&self) -> &[Diagnostic];
    pub fn classify(&self, def: DefId) -> Classification;
    pub fn timing_report(&self) -> TimingReport;
    pub fn snapshot(&self) -> State;  pub fn restore(&mut self, s: &State);
}
```
- **Incremental edits**: invalidate compiler artifacts by def hash and dependency.
  - Timing mode resumes immediately.
  - Fast mode drops to an uncompiled fallback (Timing engine at phase granularity) until tier 1 is ready (target < 50 ms for a CPU-sized design), then tier 2 swaps in later.
  - State carries across recompiles through net paths.
- **Mode switch** maps state via kernel state nets. Timing → Fast is legal only at quiescence.
- Diagnostic kinds: `Contention`, `Metastable`, `Oscillating`, `StuckX`, `DelaySensitive`, `SetupViolation`, `HoldViolation`, `GuardHazard`, `WidthMismatch`, `IterationLimit`.

---

## 11. Serialization

```jsonc
{
  "format": "panda/1",
  "defs": [{
    "id": "def:full_adder", "name": "Full Adder", "version": 3,
    "pins": [{"name":"A","dir":"in","width":1}, /* ... */ {"name":"Cout","dir":"out","width":1}],
    "instances": [{"id":"g1","def":"atom:XOR","params":{"inputs":2}} /* ... */],
    "nets": [{"id":"n1","width":1,"termination":"none","ends":[["pin","A"],["g1","in0"]]}],
    "editor": { /* geometry; ignored by the kernel */ }
  }],
  "top": "def:cpu"
}
```
Compiled artifacts are caches keyed by def content hash and are never part of the format. A plain-text netlist DSL for tests MAY be added to `panda-cli`.

---

## 12. Reference CPU benchmark ("Panda-8")

A gate-only 8-bit CPU, generated programmatically into a netlist, is the acceptance benchmark for D7 and the main differential-test subject.
- 8-bit datapath, 4×8 register file, PC, IR, flags, ripple-carry ALU, 256-byte program ROM (constants + mux tree), output port.
- Master-slave DFFs from NAND kernels, a single clock, synchronous reset via `PowerOnReset`.
- Target size 3–5k gates.
- Variants: (a) packaged hierarchically, (b) fully flat (exercises kernel extraction without hierarchy hints), (c) gated-clock register file (exercises conditional kernels).

---

## 13. Roadmap (kernel first)

| Milestone | Deliverable | Exit criterion |
|---|---|---|
| **M1** | `panda-netlist`, `panda-timing`, `panda-cli` (load, run, test vectors, VCD) | Golden circuits behave per §5: SR/D latches, MS-DFF, counters, ring oscillator, period-2 collapse, Panda-8 runs a program |
| **M2** | Combinational compilation; ideal blocks; classification | Randomized equivalence: ideal block outputs = glass-box settled outputs |
| **M3** | `panda-compile`: kernel extraction + characterization, **FastRef** (§6.2 interpreter), STA; ideal blocks for Timing mode (§5.2) | **Differential fuzzing**: FastRef = Timing at phase boundaries on STA-clean designs; every mismatch is explained by a reported diagnostic. Panda-8 runs correctly in FastRef |
| **M4** | `panda-fast`: phase specialization → FastIR (word ops, partitions + dirty flags + guarded skipping, slice packing, ROM→LUT), bytecode interpreter; optional edge fusion | FastIR engine = FastRef on all fuzz designs; Panda-8 ≥ 1 MHz interpreted |
| **M5** | `panda-jit` (Cranelift, background compile + hot-swap) and `panda-kernel` facade (§10: modes, incremental edits, snapshots) | JIT = interpreter bit-for-bit; ≥ 2× over interpreter on Panda-8; facade API tests |
| **M6** | Editor/renderer over `panda-kernel` | Place, wire, package, click-to-test, waveform view, diagnostics panel |

Engineering rules from the start: struct-of-arrays state (Turing Complete hit a wall at ~1M nodes of over 200 bytes each), no per-instance recursion at runtime (Digital Logic Sim's main cost), and the Timing engine is always the oracle for tests.

---

## 14. Evidence Summary

### 14.1 Prior art
- **No hobby tool compiles gate-built latches into registers.**
  - Turing Complete bans combinational loops and uses built-in state.
  - Digital Logic Sim walks the hierarchy recursively; its LUT and skip optimizations exist only as TODOs.
  - Digital and Logisim are event-driven and break symmetric latch oscillation with (unseeded) randomness.
  - CircuitVerse rejects circuits that never settle.
- **Speed anchors:**
  - Virtual Circuit Board: ~1M ticks/s ≈ 30 kHz for a gate CPU.
  - Digital: 120 kHz example CPU, likely built from built-in components.
  - Verilator: ~900 kHz on a 10k-node *RTL* core.
- **Techniques borrowed:**
  - Arcilator: cut at registers, dedup, small-logic LUTs.
  - ESSENT/GSIM: coarse partitions with activity flags, dynamic don't-cares.
  - Redpiler: constant fold, dedup, prune.

### 14.2 Theory
- **Ternary simulation** (Eichelberger 1965; Brzozowski–Seger) is polynomial and sound for races and oscillation under arbitrary gate/wire delays. In Panda it serves only as a diagnostic and classification tool (§4.4, §8.4, §9.3): treating skew as unbounded makes it too pessimistic for clocked designs to use as semantics.
- Cyclic-but-combinational detection follows Malik 1994.
- LTP tables are valid under **fundamental mode** (multi-input change, then hold until settled), which STA enforces.
- No general published algorithm extracts state elements from arbitrary gate SCCs. Semantic table characterization appears to be new.

### 14.3 Performance (measured on a Ryzen 7 3700X, random 5k-gate netlist)

| Engine | ns/gate-op | Full-evaluation rate (5k gates) |
|---|---|---|
| Switch interpreter | 1.4 | 140 kHz |
| Branchless table interpreter (2-state) | 0.74 | 270 kHz |
| Same, 4-state | 0.81 | 248 kHz |
| clang -O2 straight-line | 0.35 | 576 kHz |
| Event-driven, per gate | 17–20 per evaluation | wins only below ~4% activity |
| SIMD-packing unrelated gates | 2.4 | loses |
| Generic 4/6-LUT mapping | 2.5–3.0 per LUT | ~1.4 gates absorbed per LUT; loses |

- **Path to ≥ 1 MHz** (estimate):
  - 1 evaluation per cycle via fusion.
  - ROM → 1 load.
  - About 60% of logic slice-packed 8-wide: ~1,900 ops ≈ 1.4 µs.
  - Partition skipping at 30–40% activity: ~0.6 µs, i.e. **1.5–2 MHz interpreted, 3–4 MHz JIT**.
  - Missing any one of fusion, slicing, ROM→LUT or skipping lands at 300–700 kHz.
- **Cranelift**: compiles ~10× faster than LLVM; its code is ~14% slower.
- **GPU / multicore**: ruled out for this size. Synchronization latency (≥ 100 ns per barrier) is comparable to the whole cycle budget.

### 14.4 Prototype findings (Python, `scratchpad/proto/`)
- **Confirmed:**
  - Cut + inputs determine quiescent state.
  - Per-phase kernel classification is opaque/transparent and acyclic for MS-DFFs and sync counters.
  - The 2-phase compiled evaluator matched L0 over 400 random phases, including X power-on.
  - Ternary analysis was never wrong against random-delay simulation.
- **Corrections it forced:**
  - State must include all kernel nets, since results otherwise depend on previous inputs.
  - Kernels must absorb private fan-in.
  - Gate-level const-prop doesn't break latch loops; classification must be functional.
  - Unit delay shows period-2 oscillation where hardware settles → §5.3.
  - Ternary is too pessimistic for clocked logic (all edges of a toggle FF → X) → diagnostic only.
  - Same-tick D/clk changes are hold violations → §9.2.
  - Randomizing all nets at power-on causes oscillation → randomize kernel state only.
  - Ring oscillators start quiescent at all-X → detect statically (no binary stable state).

---

## 15. Open Questions (non-blocking)

- **[OPEN-1]** Tunables: `KERNEL_MAX_LOOP` (6), `KERNEL_MAX_BITS` (12), `DELTA_LIMIT`, `LUT_MAX_BITS` (12–16), `LUT_MIN_GATES`, partition size (50–200), `P2_LIMIT` (4). Settle by benchmark in M3/M4.
- **[OPEN-2]** Per-gate-type delays in Timing mode (currently uniform 1). This affects STA and tables but no architecture.
- **[OPEN-3]** Parametric packages (N-bit register) — v2. The netlist must not preclude them.
- **[OPEN-4]** Should Fast mode support multiple asynchronous clock domains beyond hyperperiod scheduling (e.g. incommensurate periods)? Hyperperiod is fine for integer tick periods. Revisit if users hit it.

---

## Appendix A — 4-state word ops (two-plane)

Normalize first (Z→X): `a ← a | b`. All ops are bitwise over `u64`.

| Op | `a_out` | `b_out` |
|---|---|---|
| NOT x | `~a \| b` | `b` |
| x AND y | `a1 & a2` | `a1 & a2 & (b1 \| b2)` |
| x OR y | `a1 \| a2` | `(a1 \| a2) & ~((a1 & ~b1) \| (a2 & ~b2))` |
| x XOR y | `(a1 ^ a2) \| b1 \| b2` | `b1 \| b2` |
| NAND / NOR / XNOR | NOT of the above | |
| TRI(d, en) | `en=1 → d`, `en=0 → (0,1)`, `en=X → (1,1)` | |

## Appendix B — Scalar 2-bit-code gate tables

Code `0,1,2(X),3(Z)`; index = `a | b << 2`; entry = 2-bit result; 16 entries packed in a `u32`. Inputs coded `Z` behave as `X`. Example AND: result `0` if either input is `0`, `1` if both are `1`, else `X`.
