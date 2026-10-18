//! Perf HUD (status bar): simulation rate (and target while speed-limited), engine tier
//! (Timing / Reference / Compiled / JIT), snapshot publication rate and cost, UI frame time and
//! frame rate, sim-thread busy fraction and waveform gaps. It only reads the latest snapshot.

use std::time::{Duration, Instant};

use egui::{Color32, RichText, Ui};

use crate::app::{AppState, Panel, Slot};
use crate::simhost::{Snapshot, Speed};

use super::control::si;

/// Measures the rate of a monotonically increasing counter over windows of `window`.
#[derive(Clone, Debug)]
pub struct RateMeter {
    window: Duration,
    start: Option<(Instant, u64)>,
    /// Last measured rate (per second).
    pub rate: f64,
}

impl RateMeter {
    /// A meter with the given window.
    pub fn new(window: Duration) -> RateMeter {
        RateMeter {
            window,
            start: None,
            rate: 0.0,
        }
    }

    /// Feed the counter's value at `now`. Returns the current rate.
    pub fn update(&mut self, now: Instant, count: u64) -> f64 {
        match self.start {
            Some((_, c0)) if count < c0 => self.start = Some((now, count)), // counter reset
            Some((t0, c0)) => {
                let dt = now.duration_since(t0);
                if dt >= self.window {
                    self.rate = (count - c0) as f64 / dt.as_secs_f64();
                    self.start = Some((now, count));
                }
            }
            None => self.start = Some((now, count)),
        }
        self.rate
    }
}

/// The serving engine tier as the HUD names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// No design loaded.
    None,
    /// Timing mode (unit-delay reference engine).
    Timing,
    /// Fast mode requested, still served by the timing engine while it compiles.
    Compiling,
    /// Fast mode on the reference (interpreted) kernel engine.
    Reference,
    /// Fast mode on the compiled FastSim engine.
    Compiled,
    /// Fast mode on the Cranelift JIT.
    Jit,
}

impl Tier {
    /// The tier of a snapshot.
    pub fn of(s: &Snapshot) -> Tier {
        use panda_kernel::{EngineKind, FastBackend};
        match s.engine {
            None | Some(EngineKind::None) => Tier::None,
            Some(EngineKind::Timing) => Tier::Timing,
            Some(EngineKind::TimingFallback) => Tier::Compiling,
            Some(EngineKind::Fast(FastBackend::Reference)) => Tier::Reference,
            Some(EngineKind::Fast(_)) if s.jit => Tier::Jit,
            Some(EngineKind::Fast(_)) => Tier::Compiled,
        }
    }

    /// Short label.
    pub fn label(self) -> &'static str {
        match self {
            Tier::None => "—",
            Tier::Timing => "Timing",
            Tier::Compiling => "Compiling…",
            Tier::Reference => "Reference",
            Tier::Compiled => "Compiled",
            Tier::Jit => "JIT",
        }
    }

    /// Tooltip.
    pub fn description(self) -> &'static str {
        match self {
            Tier::None => "No design loaded",
            Tier::Timing => "Timing mode: unit-delay reference engine (per tick)",
            Tier::Compiling => {
                "Fast mode: compiling in the background; the timing engine serves meanwhile"
            }
            Tier::Reference => "Fast mode: reference kernel engine (interpreted)",
            Tier::Compiled => "Fast mode: compiled kernel engine (FastSim)",
            Tier::Jit => "Fast mode: Cranelift JIT",
        }
    }

    fn color(self, text: Color32) -> Color32 {
        match self {
            Tier::Jit => crate::theme::TEAL,
            Tier::Compiled => crate::theme::STEEL,
            Tier::Reference | Tier::Compiling => crate::theme::OCHRE,
            Tier::None | Tier::Timing => text,
        }
    }
}

/// The simulation-rate text: `"1.70 M cyc/s"`, with the target while speed-limited
/// (`"998 / 1000 cyc/s"`) and `"paused"` / `"idle"` when not advancing.
pub fn rate_text(s: &Snapshot) -> String {
    if !s.running {
        return "paused".into();
    }
    if s.idle {
        return "idle (stable)".into();
    }
    match s.speed {
        Speed::Unlimited => format!("{} cyc/s", si(s.stats.cycles_per_sec)),
        Speed::CyclesPerSec(r) => {
            format!("{} / {} cyc/s", si(s.stats.cycles_per_sec), si(r))
        }
    }
}

/// The status-bar HUD.
#[derive(Debug)]
pub struct Hud {
    publish: RateMeter,
    fps: RateMeter,
}

impl Default for Hud {
    fn default() -> Self {
        Hud {
            publish: RateMeter::new(Duration::from_millis(500)),
            fps: RateMeter::new(Duration::from_millis(500)),
        }
    }
}

impl Panel for Hud {
    fn id(&self) -> &'static str {
        "sim_hud"
    }

    fn title(&self) -> String {
        "Perf".into()
    }

    fn slot(&self) -> Slot {
        Slot::Status
    }

    fn ui(&mut self, ui: &mut Ui, app: &mut AppState) {
        let now = Instant::now();
        let s = &app.live.snapshot;
        let hz = self.publish.update(now, s.stats.publications);
        let fps = self.fps.update(now, app.frame.frames);
        let tier = Tier::of(s);
        ui.label(RichText::new(rate_text(s)).strong().monospace())
            .on_hover_text(format!(
                "{:.0} cycles/s, {} ticks/s; {} cycles since reset, tick {}",
                s.stats.cycles_per_sec,
                si(s.stats.ticks_per_sec),
                s.stats.cycles,
                s.tick
            ));
        ui.label(
            RichText::new(tier.label())
                .color(tier.color(ui.visuals().text_color()))
                .strong(),
        )
        .on_hover_text(tier.description());
        ui.label(format!("pub {hz:.0} Hz · {:.0} µs", s.stats.publish_us))
            .on_hover_text("Snapshot publications per second and mean cost on the sim thread");
        ui.label(format!("busy {:.0}%", s.stats.busy * 100.0))
            .on_hover_text("Fraction of sim-thread wall time spent simulating");
        ui.label(format!("UI {:.2} ms · {fps:.0} fps", app.frame.avg_ms))
            .on_hover_text(format!(
                "UI code time per frame (moving average; last {:.2} ms) and frames per second",
                app.frame.last_ms
            ));
        if app.wave.gaps > 0 {
            ui.label(
                RichText::new(format!("wave gaps {}", app.wave.gaps)).color(crate::theme::OCHRE),
            )
            .on_hover_text(
                "Waveform intervals skipped while the simulation ran faster than capture",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_meter_windows() {
        let t0 = Instant::now();
        let mut m = RateMeter::new(Duration::from_millis(500));
        assert_eq!(m.update(t0, 10), 0.0);
        assert_eq!(m.update(t0 + Duration::from_millis(100), 22), 0.0);
        let r = m.update(t0 + Duration::from_millis(500), 70);
        assert!((r - 120.0).abs() < 1e-9, "{r}");
        // Counter reset (new design): restarts the window, keeps the last rate.
        assert_eq!(m.update(t0 + Duration::from_millis(600), 3), 120.0);
        let r = m.update(t0 + Duration::from_millis(1100), 63);
        assert!((r - 120.0).abs() < 1e-9, "{r}");
    }

    #[test]
    fn tiers_and_rate_text() {
        use panda_kernel::{EngineKind, FastBackend};
        let mut s = Snapshot::default();
        assert_eq!(Tier::of(&s), Tier::None);
        s.engine = Some(EngineKind::TimingFallback);
        assert_eq!(Tier::of(&s).label(), "Compiling…");
        s.engine = Some(EngineKind::Fast(FastBackend::Reference));
        assert_eq!(Tier::of(&s), Tier::Reference);
        s.engine = Some(EngineKind::Fast(FastBackend::Compiled));
        assert_eq!(Tier::of(&s).label(), "Compiled");
        s.jit = true;
        assert_eq!(Tier::of(&s).label(), "JIT");
        assert_eq!(rate_text(&s), "paused");
        s.running = true;
        s.stats.cycles_per_sec = 1_703_000.0;
        assert_eq!(rate_text(&s), "1.70 M cyc/s");
        s.speed = Speed::CyclesPerSec(1000.0);
        s.stats.cycles_per_sec = 998.0;
        assert_eq!(rate_text(&s), "998 / 1000 cyc/s");
        s.idle = true;
        assert_eq!(rate_text(&s), "idle (stable)");
    }
}
