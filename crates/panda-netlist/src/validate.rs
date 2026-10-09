//! Structural validation of a [`Design`] (SPEC §4). Returns every problem found, not just the first.
//!
//! Errors: unknown top/def/instance/port, duplicate ids (defs, instances within a def, nets within a
//! def, pins within a def), recursive def dependency, net endpoint width ≠ net width, atom parameter
//! errors (gate inputs < 1, reduce with inputs ≠ 1, width 0, splitter parts out of range, clock
//! period < 2 or high ∉ (0, period)), Clock/PowerOnReset outside the top def, a pin or port used by
//! more than one net, an `In` pin of the def driven from inside (a gate output on the same net).
//! Warnings: unused pins, nets with ≥2 drivers where some driver is not Z-capable (`MultipleDrivers`).

use crate::model::Design;
use crate::Diagnostic;

pub fn validate(design: &Design) -> Vec<Diagnostic> {
    todo!()
}

/// True if any diagnostic is an error.
pub fn has_errors(diags: &[Diagnostic]) -> bool {
    diags.iter().any(|d| d.severity == crate::Severity::Error)
}
