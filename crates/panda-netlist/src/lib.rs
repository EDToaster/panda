//! Panda netlist: values, the hierarchical design model, a builder API, validation and
//! flattening to a bit-level netlist. See SPEC.md §3, §4, §11.

pub mod builder;
pub mod eval;
pub mod flat;
mod flatten_impl;
mod structure;
pub mod model;
pub mod validate;
pub mod value;

pub use builder::{DefBuilder, DefRef, DesignBuilder, Sig};
pub use flat::{flatten, FlatNetlist};
pub use model::*;
pub use value::{Logic, Value};

use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

/// Diagnostic kinds shared by every crate (SPEC §10).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum DiagKind {
    // structural (validation / flattening)
    WidthMismatch,
    UnknownDef,
    UnknownInstance,
    UnknownPort,
    DuplicateId,
    RecursiveDef,
    SourceNotTop,
    BadAtom,
    BadPinUse,
    MultipleDrivers,
    // simulation
    Contention,
    Metastable,
    Oscillating,
    StuckX,
    IterationLimit,
    // compilation / timing
    DelaySensitive,
    SetupViolation,
    HoldViolation,
    GuardHazard,
    Unsupported,
    Other,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct Diagnostic {
    pub kind: DiagKind,
    pub severity: Severity,
    pub message: String,
    /// Hierarchical location (def id, instance path or net name) when known.
    pub location: Option<String>,
}

impl Diagnostic {
    pub fn error(kind: DiagKind, message: impl Into<String>) -> Self {
        Diagnostic { kind, severity: Severity::Error, message: message.into(), location: None }
    }
    pub fn warning(kind: DiagKind, message: impl Into<String>) -> Self {
        Diagnostic { kind, severity: Severity::Warning, message: message.into(), location: None }
    }
    pub fn at(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} {:?}: {}", self.severity, self.kind, self.message)?;
        if let Some(l) = &self.location {
            write!(f, " (at {l})")?;
        }
        Ok(())
    }
}
