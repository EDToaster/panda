//! Simulation UX (owner: the **sim UX** work stream).
//!
//! Mode toggle, run/pause/step, speed, reset (all [`crate::simhost::Command`]s through
//! `AppState::sim`), clickable top-level inputs, the waveform viewer
//! ([`crate::app::AppState::wave`], [`crate::app::AppState::set_wave_signals`]), diagnostics
//! and STA panels, the perf HUD (`AppState::live.snapshot.stats`, `AppState::frame`).
//! Sim-thread work beyond the built-in commands goes through `Command::Custom`.
//!
//! Extend the shell only via [`register`].

use crate::app::Registry;

/// Register this module's panels and tools.
pub fn register(_reg: &mut Registry) {}
