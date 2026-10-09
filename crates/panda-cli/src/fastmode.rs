//! Driving designs through the `panda_kernel` facade (Fast mode and mode switches).
//!
//! The CLI uses the facade for everything that is not plain Timing mode: `panda run --mode fast`,
//! vector files with `mode` directives, `panda sta` and `panda classify`. This module holds the
//! shared pieces: building a [`Kernel`] from command-line settings, entering Fast mode (waiting for
//! the background compiler), reading every net value, and printing the unified diagnostics.

use std::io::{self, Write};
use std::time::{Duration, Instant};

use panda_kernel::{DiagEntry, DiagSource, EngineKind, FastBackend, Kernel, KernelOptions, Mode};
use panda_netlist::{Design, Logic, Value};
use panda_timing::TimingOptions;

use crate::CliError;

/// Parse a mode name (`timing` / `fast`).
pub fn parse_mode(s: &str) -> Option<Mode> {
    match s.to_ascii_lowercase().as_str() {
        "timing" => Some(Mode::Timing),
        "fast" => Some(Mode::Fast),
        _ => None,
    }
}

/// Parse a Fast backend name (`reference` / `compiled` / `jit`, also `ref` / `fastref` / `fastsim`).
pub fn parse_backend(s: &str) -> Option<FastBackend> {
    match s.to_ascii_lowercase().as_str() {
        "reference" | "ref" | "fastref" => Some(FastBackend::Reference),
        "compiled" | "fastsim" => Some(FastBackend::Compiled),
        "jit" | "native" => Some(FastBackend::Jit),
        _ => None,
    }
}

/// Human name of a backend.
pub fn backend_name(b: FastBackend) -> &'static str {
    match b {
        FastBackend::Reference => "reference",
        FastBackend::Compiled => "compiled",
        FastBackend::Jit => "jit",
    }
}

/// Human name of the engine serving a kernel.
pub fn engine_name(e: EngineKind) -> String {
    match e {
        EngineKind::None => "none".into(),
        EngineKind::Timing => "timing".into(),
        EngineKind::TimingFallback => "timing (fallback while compiling)".into(),
        EngineKind::Fast(b) => format!("fast/{}", backend_name(b)),
    }
}

/// Facade settings taken from the command line.
#[derive(Clone, Debug)]
pub struct KernelSettings {
    /// Timing-engine options (`--p2-limit`, `--osc-limit`).
    pub timing: TimingOptions,
    /// Fast-mode engine (`--backend`).
    pub backend: FastBackend,
    /// Reference clock net for cycle steps (`--clock`).
    pub cycle_clock: Option<String>,
    /// Bound for letting a design without source changes settle.
    pub max_settle_ticks: u64,
}

impl KernelSettings {
    /// The facade options. Compilation is synchronous, so that entering Fast mode is a single
    /// measurable step and the program never runs on a stand-in engine.
    pub fn options(&self) -> KernelOptions {
        KernelOptions {
            timing: self.timing.clone(),
            backend: self.backend,
            background: false,
            cycle_clock: self.cycle_clock.clone(),
            max_settle_ticks: self.max_settle_ticks.max(1),
            ..KernelOptions::default()
        }
    }
}

/// Load `design` into a new Timing-mode kernel. `inputs` become part of the power-on state (as
/// with `panda run --set` in Timing mode).
pub fn new_kernel(design: &Design, settings: &KernelSettings, inputs: &[(String, Value)]) -> Result<Kernel, CliError> {
    let mut k = Kernel::new(settings.options());
    k.load(design.clone()).map_err(CliError::Design)?;
    if !inputs.is_empty() {
        for (pin, v) in inputs {
            k.set_input(pin, v).map_err(CliError::Other)?;
        }
        // Reloading powers on again with the inputs set above.
        k.load(design.clone()).map_err(CliError::Design)?;
    }
    Ok(k)
}

/// Switch to `mode` and wait until a real engine of that mode serves (Fast mode: compiled).
/// Returns the time it took (the compile time when entering Fast mode the first time).
pub fn enter_mode(k: &mut Kernel, mode: Mode) -> Result<Duration, String> {
    let t0 = Instant::now();
    k.set_mode(mode);
    k.wait_ready().map_err(|diags| {
        let msgs: Vec<String> = diags.iter().map(ToString::to_string).collect();
        format!("Fast-mode compilation failed: {}", msgs.join("; "))
    })?;
    if mode == Mode::Fast && !matches!(k.engine(), EngineKind::Fast(_)) {
        return Err(format!("Fast mode is served by {} after compiling", engine_name(k.engine())));
    }
    Ok(t0.elapsed())
}

/// Every net value of the engine serving `k` (index = net of [`Kernel::netlist`]).
pub fn values(k: &Kernel) -> Vec<Logic> {
    k.snapshot().values().to_vec()
}

fn source_name(s: DiagSource) -> &'static str {
    match s {
        DiagSource::Design => "design",
        DiagSource::Timing => "timing",
        DiagSource::Fast => "fast",
        DiagSource::Compile => "compile",
        DiagSource::Sta => "sta",
        DiagSource::Kernel => "kernel",
    }
}

/// Write the facade's deduplicated diagnostics, one per line: `{prefix}[source] diagnostic`,
/// with the occurrence count and tick range when it occurred more than once.
pub fn write_entries(w: &mut dyn Write, prefix: &str, entries: &[DiagEntry]) -> io::Result<()> {
    for e in entries {
        let src = source_name(e.source);
        if e.count > 1 {
            writeln!(
                w,
                "{prefix}[{src}] {} (×{}, ticks {}..{})",
                e.diagnostic, e.count, e.first_tick, e.last_tick
            )?;
        } else {
            writeln!(w, "{prefix}[{src}] {}", e.diagnostic)?;
        }
    }
    Ok(())
}

/// Format a rate with an SI suffix (`1.23M`, `45.6k`, `789`).
pub fn si(x: f64) -> String {
    let (v, s) = if x >= 1e9 {
        (x / 1e9, "G")
    } else if x >= 1e6 {
        (x / 1e6, "M")
    } else if x >= 1e3 {
        (x / 1e3, "k")
    } else {
        (x, "")
    };
    if v >= 100.0 {
        format!("{v:.0}{s}")
    } else if v >= 10.0 {
        format!("{v:.1}{s}")
    } else {
        format!("{v:.2}{s}")
    }
}

/// Format a duration in adaptive units (`812 ns`, `3.41 µs`, `12.3 ms`, `1.52 s`).
pub fn dur(d: Duration) -> String {
    let ns = d.as_secs_f64() * 1e9;
    if ns < 1e3 {
        format!("{ns:.0} ns")
    } else if ns < 1e6 {
        format!("{:.2} µs", ns / 1e3)
    } else if ns < 1e9 {
        format!("{:.1} ms", ns / 1e6)
    } else {
        format!("{:.2} s", ns / 1e9)
    }
}

/// `n` units per `elapsed`, guarding against a zero duration.
pub fn rate(n: u64, elapsed: Duration) -> f64 {
    n as f64 / elapsed.as_secs_f64().max(1e-9)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_formatting() {
        assert_eq!(parse_mode("FAST"), Some(Mode::Fast));
        assert_eq!(parse_mode("timing"), Some(Mode::Timing));
        assert_eq!(parse_mode("slow"), None);
        assert_eq!(parse_backend("fastref"), Some(FastBackend::Reference));
        assert_eq!(parse_backend("Compiled"), Some(FastBackend::Compiled));
        assert_eq!(parse_backend("jit"), Some(FastBackend::Jit));
        assert_eq!(parse_backend("turbo"), None);
        assert_eq!(si(1_234_567.0), "1.23M");
        assert_eq!(si(45_600.0), "45.6k");
        assert_eq!(si(789.0), "789");
        assert_eq!(si(2.5e9), "2.50G");
        assert_eq!(dur(Duration::from_nanos(812)), "812 ns");
        assert_eq!(dur(Duration::from_micros(3410)), "3.4 ms");
        assert_eq!(dur(Duration::from_millis(1520)), "1.52 s");
        assert!(rate(10, Duration::ZERO) > 1e9);
    }
}
