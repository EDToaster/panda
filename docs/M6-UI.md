# M6 — Editor UI (`panda-ui`)

Decisions: **egui + wgpu via eframe 0.36, desktop first** (SPEC D15). Pure Rust; the UI talks to
`panda-kernel` in-process, so Fast mode keeps the Cranelift JIT.

## Prime requirement: the UI must not slow the simulation down

**Acceptance gate:** with the editor open, Panda-8 (SumOfProducts, Hierarchical) running in Fast
mode (JIT), the canvas showing live wire values at 60 fps, and 8 nets in the waveform viewer, the
simulation throughput must be **≥ 95% of headless `Kernel::run` throughput** on the same machine.
The UI must also stay responsive (frame time < 16 ms) while the simulation runs flat out, and
editing gestures must never block on compilation.

Architecture that guarantees it:

```
 UI thread (eframe)                          Sim thread (owns Kernel)
 ─────────────────                           ───────────────────────
 input → Command ──── bounded channel ─────▶ drains commands between run chunks
                                             loop { kernel.run(chunk ≈ 1–2 ms) ;
                                                    every ≥ 1/120 s: publish snapshot }
 frame: read latest snapshot ◀── lock-free ── Snapshot { generation, tick, mode, stats,
        (never waits on sim)     swap         values: watched/visible nets only }
```
- The sim thread never waits on the UI: publication is a lock-free swap of a preallocated buffer
  (triple buffer / `arc-swap`-style), never a mutex the UI could hold.
- Publication cost is bounded and proportional to what is **visible**: the UI tells the sim thread
  which flat nets it currently needs (nets of the def being viewed + waveform nets); the sim
  copies only those (`Kernel` bulk read by flat index). Full-netlist copies are never made per
  frame.
- Publication rate is capped (default 120 Hz) independent of simulation speed; between
  publications the sim runs uninterrupted chunks. Commands are drained only at chunk boundaries
  (an atomic flag says "commands pending" so the hot path checks one atomic per chunk).
- Waveform capture: the sim samples the ≤ N watched nets **per event / per tick** through a cheap
  observer hook into a ring buffer (only while the waveform viewer has signals). Measure its cost;
  it is part of the 95% budget.
- Compilation (kernel extraction, tables, FastSim, JIT) already runs in the background inside the
  facade; edits are applied as `EditOp`s on the sim thread and return immediately.
- The UI thread does no simulation work. Expensive UI-side work (auto-layout, STA report
  rendering) runs off the UI thread or is cached.

Facade additions needed (in `panda-kernel`, minimal): bulk read of net values by flat index
(e.g. `values(&self) -> &[Logic]` or `copy_values(&self, idx: &[NetIdx], out: &mut [Logic])`),
stable flat-index lookup for (instance path, net id, bit) via `netlist().name_index`, and a
per-event observer for waveform sampling (e.g. `run_observed(budget, &mut dyn FnMut(u64 /*tick*/))`
or a sampling list the engine copies after each event). Must not slow `run` when unused.

## Scope (M6)

1. **Document model**: `panda_netlist::Design` + editor geometry stored in `ComponentDef.editor`
   (instance position/rotation/flip, wire polylines per net with junctions, labels, package symbol
   pin layout). Round-trips through the JSON format. Designs without geometry (generated:
   panda-circuits, Panda-8) get an automatic layered layout on open.
2. **Canvas**: pan/zoom (smooth, cursor-anchored), grid, IEEE distinctive gate shapes
   (AND/OR/XOR/NAND/NOR/XNOR/NOT/BUF/TRI), splitters, constants, clocks, POR, package boxes with pin
   labels, wires coloured by live value (0, 1, X, Z, contention), buses thicker with a hex value
   label, culling of off-screen items so 10k+ gate designs stay at 60 fps.
3. **Editing**: select (click, box, shift), move, delete, rotate, copy/paste, place from palette,
   draw wires with orthogonal (Manhattan) routing and auto-junctions, connect to pins/wires,
   inspector for atom parameters / widths / clock period-high-phase / constants / net termination,
   undo/redo of every edit. Every structural edit becomes `EditOp`s applied to the kernel.
4. **Hierarchy & packaging**: open/enter/exit package instances (breadcrumbs), create a package
   from a selection (crossing nets become pins), package symbol auto-layout, library palette
   (atoms, user packages, panda-circuits blocks), classification badge per def
   (combinational/sequential/oscillator, clock-like pins).
5. **Simulation UX**: mode toggle Timing/Fast, run/pause/step (tick/phase/cycle), speed
   (unlimited or target rate), reset, click top-level inputs on the canvas to toggle/edit, live
   values, waveform viewer (add net from canvas, zoom/scroll, bus radix), diagnostics panel with
   jump-to-location and highlighting, STA panel (min period, critical path highlight, violations),
   perf HUD (cycles/s, engine tier: Reference/Compiled/JIT, frame time).
6. **Files**: new/open/save (`.panda.json`), recent files, import ROM from hex, built-in examples
   (SR latch, counter, register file, Panda-8 with programs).

Out of scope for M6: themes beyond light/dark, collaboration, web build.

## Testing

- Pure logic (document model, geometry, routing, layout, command → EditOp mapping, sim host)
  is unit-tested without a GPU.
- UI interaction and rendering: `egui_kittest` (0.36) with the wgpu renderer for **offscreen**
  snapshot images (this machine has an AMD RX 7700 XT with RADV). Agents should inspect
  rendered PNGs to verify visuals. **Do not open windows on the user's desktop**; offscreen only.
- The perf gate is an automated test/bench: headless `Kernel::run` throughput vs the full UI stack
  (sim thread + publisher + an offscreen UI rendering frames at 60 Hz with Panda-8 visible and 8
  waveform signals) — report both and the ratio.
