# Panda architecture

This document describes the code as it is. It covers the data flow from a hierarchical design to
the engines, the semantic layers and the equivalences that connect them, the testing strategy,
and where to start reading. [SPEC.md](../SPEC.md) is the normative design. Where the code
deliberately differs from it, this document says so ([Deviations from SPEC](#deviations-from-spec)).

## 1. Data flow

```
 Design (panda-netlist::model, "panda/1" JSON)          hierarchical: defs, pins, instances, nets
   │  validate (validate.rs)  →  flatten (flat.rs, flatten_impl.rs)
   ▼
 FlatNetlist                                            bit-level: 1-bit nets, 1-bit gates (delay 1),
   │                                                    sources (Constant/Clock/POR/inputs, delay 0),
   │                                                    resolvers (multi-driver nets, delay 0),
   │                                                    instance tree + net names for every alias
   ├──────────────► TimingSim (panda-timing)            Timing mode: the reference semantics
   │
   │  kernels::extract (panda-compile)                  SCCs → seed loops ≤ 6 → merge → absorb private
   ▼                                                    fan-in → residual SCCs (Sim) → keepers
 KernelSet { kernels, comb_order }                      every cycle passes through a kernel
   │  TableCache::characterize_all (charz)              per kernel sub-netlist (subnet.rs), keyed by
   ▼                                                    a structural hash, threads, bit-parallel
 Vec<Option<Arc<KernelTable>>>                          stable states, T(state, input), latency,
   │                                                    order sensitivity
   ├──────────────► panda-sta::analyze                  TimingReport: min_period, settle, diagnostics
   ├──────────────► FastRef (panda-compile::fastref)    §6.2 interpreter: the oracle
   │
   │  panda_fast::compile (pass pipeline)               order → LUTs → partitions → levelize → lower
   ▼                                                    → notify → kernel ops → kspec → layout → verify
 Program (FastIR)                                       16-byte ops, partitions = op ranges, LUTs with
   │                                                    cold gate code, kernel ops + specialization
   ├──────────────► FastSim (panda-fast)                interpreter (tier 1)
   │  panda_jit::Jit::compile (Cranelift)
   ▼
 NativeCode (partition fns + per-clock-tuple event fns) installed into FastSim (tier 2)
```

`panda-kernel::Kernel` owns this pipeline for interactive use: it validates and flattens on `load`
and `edit`, runs Timing mode immediately, and compiles Fast mode on a worker thread. While Fast
mode compiles, it is served by tiers in order: the Timing engine at phase granularity, then a
Reference engine (FastRef semantics), then the FastSim interpreter, then the JIT. Each swap
transfers state at a phase or event boundary.

### 1.1 Netlist (`panda-netlist`)

* `value.rs` defines `Logic` (2-bit code `0 1 X Z`) and `Value` (W-bit, two bit-planes).
* `eval.rs` holds the only definition of gate semantics. It has scalar tables (`tables()`,
  `lookup2`), `eval_gate`, `resolve` (termination, contention) and the two-plane word ops of SPEC
  Appendix A. Every engine derives its gate code from it.
* `model.rs` defines `Design` → `ComponentDef` (pins, instances, nets, editor data) → `Instance`
  (`InstKind::Atom(Atom)` or `InstKind::Def(id)`). The atoms are `Gate{op,inputs,width,reduce}`,
  `Not`, `Buf`, `Tri`, `Splitter` (which also serves as joiner), `Constant`,
  `Clock{period,high,phase}` and `PowerOnReset{ticks}`. Top-level `In`/`Out` pins are the inputs
  and outputs.
* `flatten` splits buses into bits and aliases splitters and pins (union-find). It gives every net
  at most one driver by inserting a `Resolver` over private per-driver nets, and records the
  instance tree and every hierarchical alias (`net_names`, `name_index`). Flattening is
  deterministic: the same design always yields the same net numbering. The end-to-end test relies
  on this.

### 1.2 Timing mode (`panda-timing`)

`TimingSim` is event-driven and runs on struct-of-arrays data. At each tick it evaluates gates
whose inputs changed, then updates sources, resolvers, period-2 collapse (`p2_limit`, default 4)
and the oscillation detector (`osc_limit`). It also raises the stuck-X diagnostic, with an X-source
trace from `xtrace.rs`. Ideal blocks (`ideal.rs`) are opt-in (`TimingOptions::ideal_blocks`). They
replace outermost acyclic package instances by transport-delay functions, evaluated through lazily
filled LUTs.

### 1.3 Kernels and tables (`panda-compile`)

* `graph.rs`: the gate/resolver graph, iterative Tarjan SCC, topological orders, bounded simple-cycle
  search.
* `kernels.rs`: extraction (SPEC §8.3, steps listed in the module docs). Kinds: `Table`
  (inputs + cut ≤ `max_bits`, default 12), `Sim` (transition computed on the fly) and `Keeper`.
* `subnet.rs`: a standalone sub-netlist per kernel (its inputs become top inputs) and the
  structural key used for caching. Kernel transitions **are** Timing-mode runs of this sub-netlist,
  so the tables inherit Timing semantics, period-2 collapse included.
* `charz.rs` (+ `charz/fast.rs`, `charz/lanes.rs`, `charz/cache.rs`): characterization. It
  enumerates stable states via the cut, computes `T(s, v)` with latency, and checks order
  sensitivity exhaustively for ≤ 4 simultaneous changes, sampled above that. The fast path is
  bit-parallel and multithreaded and must equal the reference path (`characterize_subnet_reference`)
  exactly. `KernelSim` computes transitions on the fly for `X`/`Z` states and inputs and for Sim
  and Keeper kernels.
* `fastref.rs`: **FastRef**, the per-event algorithm of SPEC §6.2, written for clarity.
  Round 0 evaluates combinational logic dirty-driven in `comb_order`. In each later round, every
  kernel whose input vector changed transitions once, all of them at once. That repeats until no
  kernel input changed. After `delta_limit` rounds (64), still-changing kernels go to `X`, are
  frozen, and raise `IterationLimit`. Table hits are used only for binary state and inputs.
  `DelaySensitive` is raised when an order-sensitive entry is used for the kernel's actual previous
  input vector.

### 1.4 Compiled Fast mode (`panda-fast`)

* `compile.rs` is the pass pipeline listed in the module docs. Partitions default to 32 ops. LUTs
  are cones with ≤ `lut_max_bits` (10) support bits that absorb ≥ `lut_min_gates` (16) gates. The
  Panda-8 ROM in every style becomes LUT reads. A LUT whose support is not binary runs its cold gate
  code, which keeps exact 4-state semantics.
* `fuse.rs`: fanout-1 gate chains with ≤ 4 inputs (`fuse_inputs`) become one exact 4-state
  truth-table op in the interpreter encoding (`ir::HOp`).
* `kspec.rs`: phase specialization per (kernel, clock tuple). It produces `Opaque` (outputs never
  change; only the state id moves) and `Transparent` (outputs are a table over the input vector),
  both proven from the kernel table for every transition the phase allows. Kernels with
  timed-impure inputs (gated clocks) keep the general table path. The order lemma in the module
  docs explains why data transitions of O/T phases skip FastRef's runtime order check.
* `sim.rs` + `exec.rs`: **FastSim**, a round-faithful executor. It sweeps dirty partitions in index
  order (bitset scan), batches change propagation through per-net records, and runs kernel round
  boundaries through the specialized paths or the table/`KernelSim` path exactly as FastRef does.
  Values are one byte per net in the same `Logic` code as FastRef. LUT-internal nets, fused-away
  gate outputs and lazy kernel internals are materialized on read. `values()` is therefore always
  exactly FastRef's, and a differential check is a slice comparison.
* `native.rs`: the tier-2 hook. Native partition functions and per-clock-tuple event functions run
  over the same runtime arrays through a `#[repr(C)]` `NativeCtx`. `NativeSlot` hands
  background-compiled code to a running engine, which installs it at the next event boundary.

### 1.5 JIT (`panda-jit`)

`Jit::compile` extracts a self-contained spec per partition (`spec.rs`) and per clock tuple
(`event.rs`), looks each one up in a content-keyed `JitCache`, and compiles the misses into one
Cranelift module (`codegen.rs`). Event functions handle a whole event (timed-source writes, guarded
partition calls, the round loop with transparent-kernel tables as immediates). At the first round
boundary they cannot handle (a general-path kernel, an order check, the delta limit) they return,
and the interpreter continues from that point. `Jit::install` is synchronous; `Jit::attach`
compiles in the background and hot-swaps.

### 1.6 Facade (`panda-kernel`)

`Kernel` provides `load`, `edit` (`EditOp` batches, atomic, state carried by net name),
`set_mode`, `set_input`, `step(n, Tick|Phase|Cycle)`, `run(budget)`, `run_to`, `read`,
`snapshot`/`restore`, `classify`, `timing_report` and `diagnostics` (deduplicated, with producer
and tick range). Engines sit behind an internal `FastEngine` trait (`engine.rs`). The backends are
`FastBackend::Reference`, `FastBackend::Compiled` and `FastBackend::Jit`; `Jit` is the default and
falls back to `Compiled` without the `jit` feature.

* `refengine.rs` is a fork of `FastRef` plus `load_state`. It lets inconsistent kernels *finish*
  their transition, which mode switches, edits and snapshots need. The `Compiled` backend loads
  states through it, because `FastSim::load_state` restarts inconsistent kernels from `X`. Apart
  from `load_state`, the fork's code is identical to `panda_compile::fastref` today.
* `schedule.rs` holds source timing for `Phase`/`Cycle` targets, and `timing.rs` wraps `TimingSim`
  with state loading.
* Time in Fast mode is reported as `next_event − 1`, the tick at which Timing mode should show the
  same values.

## 2. Semantic layers and the equivalence contract

| Layer | Definition | Engines |
|---|---|---|
| **Timing** | SPEC §5. Unit delay, synchronous update, event-driven. Period-2 collapse to `X` (`Metastable`), `Oscillating`, `Contention`, `StuckX`. | `TimingSim` (glass-box). With ideal blocks, packaged combinational instances use transport-delay semantics instead (opt-in). |
| **Fast** | SPEC §6.2. Per event: zero-delay combinational logic plus kernel rounds, where `T` = the Timing-mode transition of the kernel's sub-netlist. | FastRef (oracle), the facade's `RefEngine`, FastSim (interpreter), FastSim with JIT |

The links between the layers:

1. **Timing ⇔ Fast (SPEC §6.3, D10).** This holds when STA reports no error-severity diagnostic,
   every clock period is ≥ its `min_period`, and inputs change outside hold windows (after the
   clock event settled, early enough to settle before the next event). Under those conditions,
   Fast mode's values after an event equal Timing mode's values at `next_event − 1` **on every
   net**. The scope is binary values after the reset release; power-on `X` transients are outside
   it. Violations are reported as `SetupViolation`, `HoldViolation`, `GuardHazard`, `Oscillating`
   (STA) and `DelaySensitive`, `IterationLimit`, `Oscillating`, `Metastable` (runtime).
2. **FastRef ⇔ FastSim ⇔ JIT (D12).** These are bit-exact after **every event**: all net values,
   `tick()`, `next_event_tick()`, the diagnostic kinds of the event, and the counters for events,
   rounds, max rounds, transitions, table hits and fallbacks. This holds on every design, with no
   conditions.
3. **Facade backends.** `Reference`, `Compiled` and `Jit` equal FastRef at every phase end through
   the `Kernel` API, including across tier hot swaps. Mode switches and snapshots load states with
   the "finish the transition" semantics of `RefEngine::load_state`.

Correctness never depends on heuristics. Kernel extraction limits, LUT and partition sizes,
fusion and specialization only change speed. `T` is exact either way, and specializations are
proven per transition from the tables.

## 3. Testing strategy

Every crate has unit tests and integration tests (`crates/*/tests`). The test profile uses
`opt-level = 2`. Long suites and benchmarks are `#[ignore]`d and named in each file's header.

* **Timing mode** is checked against a naive reference model (`panda-timing/tests/differential.rs`),
  golden circuits (M1), ideal blocks against glass-box, and Panda-8 against the emulator cycle by
  cycle (`panda-cpu8/tests/gate_level.rs`).
* **Flattening** is checked against an independent naive recursive flattener
  (`panda-circuits/tests/reference_flatten.rs`) and a scale test of about 1M gates.
* **Characterization**: the fast path must equal the reference path, bit for bit, over random
  kernels (`panda-compile/tests/charz_equivalence.rs`).
* **Timing ⇔ FastRef**: `panda-difftest` generates random designs (registers, latches, custom
  latches, gated clocks, tri-state buses, async structures, one or two clocks, clockless designs,
  inlined variants, extraction-option variants). It picks clock periods from STA and compares every
  net at every phase end and before every input change. Each design is classified as clean-equal,
  clean-mismatch (a bug), flagged-mismatch or flagged-equal (STA pessimism)
  (`tests/random_designs.rs`, the `difftest` binary). Panda-8 runs in every variant and ROM style
  (`tests/cpu8.rs`). `panda-sta/tests/fuzz.rs` cross-checks STA verdicts separately.
* **FastRef ⇔ FastSim**: panda-fast's suites compare after every event on random designs, raw
  random netlists with arbitrary feedback, adversarial block netlists, Panda-8 and `load_state`.
* **FastSim ⇔ JIT**: `panda-jit/tests/lockstep.rs` compares an interpreting engine and a native
  engine after every event, including mixed tiers, hot swap, `load_state` and recompilation. The
  `fast_*.rs` files rerun every panda-fast suite with the JIT installed.
* **Facade**: mode switches, edits, snapshots, background-compile races, the JIT tier, and random
  designs in both modes (`panda-kernel/tests`).
* **The whole chain** (`panda-difftest/tests/end_to_end.rs`): one stimulus drives Timing mode (facade,
  glass-box), FastRef, the FastSim interpreter, FastSim with JIT, and the facade's three Fast
  backends in lockstep. It runs on random designs and on Panda-8 (every variant, SumOfProducts and
  MuxTree; the ignored long test adds every program and the Decoder ROM). The checks are:
  * FastSim and JIT equal FastRef after every event (values, time, diagnostic kinds, counters,
    `check_invariants`);
  * every facade backend equals FastRef at every phase end;
  * on STA-clean designs, Timing equals FastRef at every phase end and before every input change,
    with any mismatch explained by a runtime diagnostic;
  * the per-seed harness never classifies a design as clean-but-mismatch.

  A run of the ignored long version covered 1,000 random designs (229 STA-clean) and about 136k
  events. Every event was bit-exact, and there was no unexplained Timing mismatch.
* **CLI**: the binary is driven end to end (`panda-cli/tests/binary.rs`, `kernel_cli.rs`).

## 4. Where to start reading

1. `SPEC.md` §5, §6 and §8.3–§8.4: the two semantics and kernels.
2. `crates/panda-netlist/src/value.rs`, then `eval.rs`, then `flat.rs`: values, gate semantics, the
   flat netlist every engine consumes.
3. `crates/panda-timing/src/lib.rs`: the module docs are the precise Timing-mode rules.
4. `crates/panda-compile/src/kernels.rs` → `subnet.rs` → `charz.rs` (module docs) →
   `fastref.rs`. FastRef is the shortest complete statement of Fast mode.
5. `crates/panda-sta/src/lib.rs` (module docs): what "STA-clean" guarantees and how.
6. `crates/panda-fast/src/lib.rs` (the executor loop), then `compile.rs` (pass list), `ir.rs`,
   `kspec.rs` (proofs of the specializations), and `sim.rs`/`exec.rs`.
7. `crates/panda-jit/src/lib.rs`, then `event.rs`/`codegen.rs`.
8. `crates/panda-kernel/src/lib.rs`: the facade's time model, tiers, mode switches, snapshots,
   edits.
9. `crates/panda-difftest/src/driver.rs` and `harness.rs`: how the equivalence contract is tested.
   `tests/end_to_end.rs` shows the whole stack in about 600 lines.

To see it run: `panda gen cpu8 --program bench -o cpu8.json && panda run cpu8.json --mode fast
--cycles 1000000`.

## Deviations from SPEC

These are recorded here so readers are not surprised. Most are deliberate, measured choices
documented in the code.

* **Slice packing (§8.7) is not implemented.** In `panda_fast::compile`, gathering bits into words
  measured about 3× slower than scalar ops. Gate fusion and LUTs are used instead.
* **Edge fusion and single-pass phases (§8.5.3) are not implemented.** Execution is round-faithful,
  as D13 records.
* **Phase classes.** There is no separate *Conditional* class; `Opaque`/`Transparent` are proven
  per (kernel, clock tuple), and gated-clock kernels use the general path.
* **No constant folding, dead-logic removal or structural hashing (§7.1 step 1, §8.1).** Kernel
  tables are deduplicated by structural hash. The JIT deduplicates functions by content.
* **Tunables differ from SPEC/contract values:** partitions of 32 ops (SPEC: 50–200),
  `lut_max_bits` 10 (SPEC: 12–16), `lut_min_gates` 16 (contract: ~24), Sim-kernel settle bound
  `4·nets + 8 + 2·p2_limit` (SPEC: `4·nets + 8`).
* **Ideal blocks are off by default** in `TimingOptions` and in the facade (SPEC §5.2 describes
  them as the default behaviour of packages).
* **The serialized format** stores instances as `{"id", "kind": {"atom": {"atom": "gate", "op":
  "xor", "inputs": 2, "width": 1, ...}}}` or `{"kind": {"def": "<id>"}}`, not
  `{"def": "atom:XOR", "params": {...}}` as SPEC §11 shows. Clocks have
  `period`/`high`/`phase` with no `duty` or `enabled` field and no automatic period.
  `PowerOnReset` has `ticks` only. There are no `Input`/`Output`/`Probe` atoms (pins are declared
  on the def).
* **Not implemented:** the random power-on option (§5.4), the ternary advisory `DelaySensitive` lint
  in characterization (§8.4; STA's `DelaySensitive` is derived from table order sensitivity), the
  SMO time-borrowing model (§9.1), and dynamic don't-cares for unselected mux inputs (§8.6).
