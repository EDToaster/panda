//! Panda editor UI (milestone M6, see `docs/M6-UI.md`): egui + wgpu via eframe, talking to
//! [`panda_kernel`] in-process on a dedicated sim thread.
//!
//! **Prime requirement:** the UI must not slow the simulation down (≥ 95 % of headless
//! `Kernel::run` throughput). The sim thread owns the kernel and never waits on the UI
//! ([`simhost`]); the UI only reads lock-free snapshots of the nets it shows.
//!
//! # Modules and ownership
//!
//! | module | owner | role |
//! |---|---|---|
//! | [`simhost`] | foundation | sim thread, commands, lock-free snapshots, waveform capture (shadow replay) |
//! | [`doc`] | foundation (editing extends) | `Design` + editor geometry, live-value mapping |
//! | [`geom`] | foundation | grid geometry, symbols and port positions |
//! | [`layout`] | foundation | automatic layered layout + orthogonal routing |
//! | [`canvas`] | foundation | read-only schematic view: pan/zoom, glyphs, live wire colours |
//! | [`app`] | foundation | eframe shell, [`app::AppState`], panel/tool registry |
//! | [`examples`] | foundation (files extends) | built-in designs |
//! | [`edit`] | **editing** | selection, move, wires, palette, inspector, undo/redo |
//! | [`hier`] | **hierarchy** | enter/exit, breadcrumbs, packaging, library palette, classification badges |
//! | [`files`] | **files** | *File* menu: new/open/save(-as), unsaved-changes prompt, recent files, ROM import, examples |
//! | [`simux`] | **sim UX** | run controls, inputs, waveforms, diagnostics, STA, perf HUD |
//!
//! # Extension points
//!
//! * Panels: implement [`app::Panel`] (choose an [`app::Slot`]: menu, top bar, left, right,
//!   bottom tab, status bar) and add it in your module's `register(reg)`.
//! * Canvas input/overlays: implement [`app::CanvasTool`] (gets the canvas response, transform
//!   and [`canvas::Scene`] for hit-testing every frame).
//! * Structural edits: [`app::AppState::submit_edit`] (document and kernel in one step).
//! * Sim-thread work: [`simhost::Command::Custom`] → [`simhost::SimEvent::Custom`], delivered to
//!   [`app::Panel::on_event`].
//! * Highlights: [`app::AppState::highlights`] (items, nets, contention) drawn by the canvas.
//!
//! [`app::panel::register_all`] has one line per feature module; that is the only shared line.

pub mod app;
pub mod canvas;
pub mod doc;
pub mod edit;
pub mod examples;
pub mod files;
pub mod geom;
pub mod hier;
pub mod layout;
pub mod simhost;
pub mod simux;
