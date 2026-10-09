# Panda

Panda is a gate-level digital logic simulator kernel written in Rust. It is *purist*: there are no
latch, flip-flop or RAM primitives. Every bit of state comes from gate feedback (cross-coupled
NANDs, master-slave flip-flops built from them, and so on), and the simulator finds that state
again on its own.

It has two simulation modes:

* **Timing mode** is the reference semantics: 4-state (`0 1 X Z`), every gate has a delay of one
  tick, and simulation is event-driven. Glitches, races and oscillation all show up. Exactly
  simultaneous release of a latch collapses to `X` with a `Metastable` diagnostic instead of
  oscillating forever (SPEC §5).
* **Fast mode** is synchronous kernel-delta semantics at clock-phase granularity (SPEC §6). The
  compiler extracts small state-holding gate clusters ("kernels") and characterizes each one into
  a transition table. A compiled engine then runs the design millions of cycles per second. When
  static timing analysis (STA) reports no errors, Fast mode gives the same value on every net at
  every phase end as Timing mode (SPEC §6.3).

The reference workload is **Panda-8**, a gate-only 8-bit CPU (about 3.7k gates; SPEC §12). It runs
at about 1 MHz in the Fast-mode interpreter and at about 1.8 MHz with the Cranelift JIT on one core
of a Ryzen 7 3700X (see [Performance](#performance)).

The design spec is [SPEC.md](SPEC.md). The architecture overview, including where to start reading
the code, is [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Milestone contracts are in
[docs/M3-CONTRACT.md](docs/M3-CONTRACT.md) and [docs/M4-CONTRACT.md](docs/M4-CONTRACT.md).

## Crate map

Dependencies point downward. Each crate depends only on crates above it, plus the dev-only test
dependencies.

| Crate | Role |
|---|---|
| `panda-netlist` | Values (`Logic`, `Value`), the hierarchical serializable design model (`panda/1` JSON), a builder API, validation, and flattening to a bit-level `FlatNetlist` with resolvers. Also scalar and two-plane word gate evaluation (`eval`). |
| `panda-timing` | Timing mode: `TimingSim` (unit delay, period-2 collapse, oscillation, stuck-X with X-source trace) and ideal blocks for packaged combinational logic (SPEC §5.2, opt-in). |
| `panda-circuits` | A library of gate-level building blocks: latches, flip-flops, registers, counters, register files, adders, ROM generator, testbench helper, hierarchy inlining. |
| `panda-cpu8` | Panda-8: ISA, assembler, behavioral emulator (the golden model), the gate-level generator (variants Hierarchical / Flat / GatedClockRegfile; ROM styles SumOfProducts / MuxTree / Decoder), and built-in programs. |
| `panda-compile` | Fast-mode analysis: graph algorithms, kernel extraction (§8.3), kernel sub-netlists, characterization into Latch Tables with a structural `TableCache` (§8.4), and **FastRef**, the direct interpreter of §6.2. FastRef is the oracle. |
| `panda-sta` | Static timing analysis (§9): per-event settle times, `min_period` per clock, setup, hold, guard-hazard and oscillator checks. If it reports no errors, Fast mode equals Timing mode. |
| `panda-fast` | **FastSim**, the compiled Fast-mode engine. It compiles a FastIR `Program` (partitions, LUTs, fused gate ops, phase-specialized kernel ops) and interprets it. It is bit-exact with FastRef after every event. It also provides the native-code hook used by the JIT. |
| `panda-jit` | Tier 2: compiles a FastSim `Program` (partitions plus per-clock-tuple event functions) to machine code with Cranelift. The code is hot-swapped at an event boundary and stays bit-exact with the interpreter. |
| `panda-kernel` | The facade (SPEC §10): `Kernel` with modes, stepping by tick, phase or cycle, background compilation through the tiers (Timing fallback → Reference → FastSim → JIT), incremental edits, snapshots, classification, timing report and unified diagnostics. |
| `panda-difftest` | The differential test harness: a random design generator, the Timing-vs-FastRef driver and classifier, minimization, Panda-8 runs, and the end-to-end chain test across every engine. |
| `panda-cli` | The `panda` command-line tool: `run`, `test` (vector files), `sta`, `classify`, `kernels`, `gen`, `bench`, `info`, `flat`, `example`. |

There is no UI yet (milestone M6).

## Quick start

You need a stable Rust toolchain with edition 2024 support (Rust 1.88 or newer: the code uses let chains).

```sh
# Build everything (release; the JIT and the benchmarks need optimizations to be meaningful).
cargo build --release

# Run the test suites. The test profile uses opt-level 2, so the long differential suites stay fast.
cargo test --workspace

# Long / exhaustive suites and benchmarks are #[ignore]d; run them explicitly, e.g.:
cargo test --release -p panda-difftest --test end_to_end -- --ignored --nocapture
cargo test --release -p panda-jit --test bench -- --ignored --nocapture
```

### The `panda` CLI

```sh
alias panda=target/release/panda
panda help

# Generate Panda-8 running the benchmark loop (default ROM: sum-of-products; default variant:
# hierarchical) and simulate it.
panda gen cpu8 --program bench -o cpu8.json
panda info cpu8.json
panda sta cpu8.json                              # min period, critical path, STA diagnostics
panda run cpu8.json --cycles 2000                # Timing mode (unit delay, glass-box)
panda run cpu8.json --mode fast --cycles 1000000 # Fast mode via the panda-kernel facade
panda run cpu8.json --mode fast --backend reference --cycles 20000   # FastRef semantics
panda bench cpu8.json --seconds 1                # throughput of the engines

# Other programs and variants:
panda gen cpu8 --program fib --variant gated --rom mux -o fib.json
panda run fib.json --mode fast --cycles 500 --watch OUT,OUT_STROBE,PC

# Smaller designs, kernels, classification, test vectors, VCD:
panda gen counter -o counter.json
panda classify counter.json -v
panda kernels counter.json --list
panda run counter.json --cycles 20 --vcd counter.vcd
panda example                                    # list the built-in example designs
```

`panda run --mode fast` compiles first and prints the compile time and throughput. Without
`--backend`, Fast mode uses the facade's default backend. VCD output in Fast mode is sampled at phase
ends. See `panda help` for every option.

### Differential fuzzing

```sh
cargo run --release -p panda-difftest -- 0 1000        # Timing vs FastRef on 1000 random designs
cargo run --release -p panda-difftest -- show 42       # one design with its STA report
cargo run --release -p panda-difftest -- cpu8 3000     # Panda-8 in FastRef vs Timing
```

### Using the library

```rust
use panda_circuits::{sync_counter, testbench};
use panda_kernel::{Kernel, Mode, StepUnit};
use panda_netlist::{DesignBuilder, Value};

let mut db = DesignBuilder::new();
let counter = sync_counter(&mut db, 4);
let top = testbench(&mut db, &counter, 16);
let mut k = Kernel::default();
k.load(db.build(&top)).unwrap();
k.set_input("EN", &Value::from_bool(true)).unwrap();
k.step(5, StepUnit::Cycle);          // Timing mode
k.set_mode(Mode::Fast);              // compiles in the background
k.wait_ready().unwrap();
k.step(3, StepUnit::Cycle);          // Fast mode
println!("Q = {:?}", k.read_u64("Q"));
```

## Performance

Measured with `cargo run --release -p panda-difftest --example engines -- 200000 3`. The run was
Panda-8 `bench_loop` at the STA minimum clock period, best of 3, on one core of an AMD Ryzen 7
3700X. The machine was shared with other jobs (load average about 9 on 16 threads), so treat the
numbers as lower bounds. Throughput is in clock cycles per second.

| ROM / variant | gates | period | Timing | FastRef | FastSim (interp.) | FastSim + JIT | facade (`Kernel`, JIT) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SumOfProducts / Hierarchical | 3,671 | 74 | 46.6 k | 25.3 k | 1.02 M | 1.81 M | 1.69 M |
| SumOfProducts / Flat | 3,671 | 74 | 46.5 k | 25.3 k | 1.03 M | 1.81 M | 1.69 M |
| SumOfProducts / GatedClockRegfile | 3,605 | 76 | 45.4 k | 31.5 k | 584 k | 968 k | 936 k |
| MuxTree / Hierarchical | 7,726 | 96 | 30.6 k | 8.2 k | 1.03 M | 1.84 M | 1.69 M |
| MuxTree / Flat | 7,726 | 96 | 30.8 k | 8.4 k | 1.04 M | 1.87 M | 1.69 M |
| MuxTree / GatedClockRegfile | 7,660 | 98 | 29.5 k | 9.0 k | 591 k | 1.02 M | 970 k |

Compile times for the default Panda-8 (SumOfProducts / Hierarchical), cold caches:

| Stage | Time |
|---|---:|
| kernel extraction + characterization (`compile_kernels`) | 16.7 ms |
| FastSim program compilation | 21.2 ms |
| JIT compilation (partitions + per-clock-tuple event functions) | 32.0 ms |
| facade: `load` → Fast mode ready with the JIT active (synchronous) | 85.6 ms |

A few things to know when reading these numbers:

* FastRef is the reference interpreter and is not optimized. On Panda-8 it is slower than Timing
  mode.
* The JIT speedup over the interpreter is 1.66–1.81×. That is below the 2× exit criterion of
  milestone M5.
* The gated-clock variant runs below 1 MHz in the interpreter. Its gated register-file clocks are
  timed-impure kernel inputs, so those kernels are not phase-specialized (`panda_fast::kspec`).
* `panda run --mode fast` on a program that halts (e.g. `--program fib`) reports a much higher
  rate, because a halted CPU barely changes. Use `--program bench` for throughput.

## Correctness strategy, in one paragraph

Timing mode is the semantics. FastRef is a literal interpreter of the Fast-mode algorithm. Every
optimized engine must equal FastRef bit for bit after every event: values, time, diagnostic kinds
and counters. FastRef must equal Timing mode at phase ends on STA-clean designs. The test suites
check every link of that chain on random designs and on Panda-8. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#testing-strategy).

## License

MIT.
