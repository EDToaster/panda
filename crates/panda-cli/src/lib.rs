//! `panda-cli` — the Panda command-line tool as a library (SPEC §2.1, M1 in §13).
//!
//! The `panda` binary is a thin wrapper around [`main_with_args`]. Modules:
//!
//! * [`signals`] — resolving user-given signal names to flat nets; default signal selection;
//! * [`vcd`] — a VCD writer for [`panda_timing::TimingSim`], usable by other crates;
//! * [`vectors`] — the line-based test-vector format and its runner;
//! * [`commands`] — `run`, `test`, `info`, `flat`, `example` (argument parsing and output);
//! * [`examples`] — example designs (the sources of `examples-data/*.json`).
//!
//! Exit codes: `0` success, `1` failure (test failures, design errors, I/O errors),
//! `2` command-line usage errors.

pub mod commands;
pub mod examples;
pub mod signals;
pub mod vcd;
pub mod vectors;

use std::fmt;
use std::io::Write;
use std::path::Path;

use panda_netlist::validate::{has_errors, validate};
use panda_netlist::{Design, Diagnostic, FlatNetlist, Severity, Value, flatten};

pub use commands::main_with_args;

/// Errors surfaced by the CLI. Each maps to an exit code via [`CliError::exit_code`].
#[derive(Debug)]
pub enum CliError {
    /// Bad command line (exit code 2).
    Usage(String),
    /// A file could not be read or written.
    Io {
        /// The path involved.
        path: String,
        /// The underlying error.
        error: std::io::Error,
    },
    /// The design is invalid (validation / flattening errors).
    Design(Vec<Diagnostic>),
    /// Anything else (bad JSON, bad vector file, ...).
    Other(String),
}

impl CliError {
    /// Process exit code for this error.
    pub fn exit_code(&self) -> i32 {
        match self {
            CliError::Usage(_) => 2,
            _ => 1,
        }
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CliError::Usage(m) => write!(f, "{m}"),
            CliError::Io { path, error } => write!(f, "{path}: {error}"),
            CliError::Design(diags) => {
                write!(f, "design has {} error(s):", diags.iter().filter(|d| d.severity == Severity::Error).count())?;
                for d in diags {
                    write!(f, "\n  {d}")?;
                }
                Ok(())
            }
            CliError::Other(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for CliError {}

/// Read a file to a string, mapping errors to [`CliError::Io`].
pub fn read_file(path: &Path) -> Result<String, CliError> {
    std::fs::read_to_string(path).map_err(|error| CliError::Io { path: path.display().to_string(), error })
}

/// Load a design from a JSON file (format `panda/1`, SPEC §11).
pub fn load_design(path: &Path) -> Result<Design, CliError> {
    let src = read_file(path)?;
    Design::from_json(&src).map_err(|e| CliError::Other(format!("{}: invalid design JSON: {e}", path.display())))
}

/// Validate and flatten a design. On success returns the netlist and the non-error diagnostics
/// (warnings / infos) from validation; on failure all diagnostics.
pub fn flatten_design(design: &Design) -> Result<(FlatNetlist, Vec<Diagnostic>), CliError> {
    let diags = validate(design);
    if has_errors(&diags) {
        return Err(CliError::Design(diags));
    }
    let nl = flatten(design).map_err(CliError::Design)?;
    Ok((nl, diags))
}

/// Render a value for humans: single bits as `0`/`1`/`x`/`z`; wider values as binary
/// (`0b0101`), plus hex and decimal when fully known (`0b0101 (0x5 = 5)`).
pub fn format_value(v: &Value) -> String {
    if v.width() == 1 {
        return v.get(0).to_char().to_string();
    }
    match v.to_u64() {
        Some(n) => format!("{v} (0x{n:x} = {n})"),
        None => v.to_string(),
    }
}

/// Write diagnostics, one per line, prefixed by `prefix`.
pub fn write_diagnostics(w: &mut dyn Write, prefix: &str, diags: &[Diagnostic]) -> std::io::Result<()> {
    for d in diags {
        writeln!(w, "{prefix}{d}")?;
    }
    Ok(())
}
