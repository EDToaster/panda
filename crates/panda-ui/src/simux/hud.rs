//! Perf HUD (status bar): simulation rate, engine tier, snapshot publication rate and cost, UI
//! frame time, sim-thread busy fraction and waveform gaps.

use std::time::{Duration, Instant};

use egui::{Color32, RichText, Ui};

use crate::app::{AppState, Panel, Slot};

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

/// The status-bar HUD.
#[derive(Debug)]
pub struct Hud {
    publish: RateMeter,
}

impl Default for Hud {
    fn default() -> Self {
        Hud {
            publish: RateMeter::new(Duration::from_millis(500)),
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
        let s = &app.live.snapshot;
        let hz = self.publish.update(Instant::now(), s.stats.publications);
        let tier = s.tier_label();
        let tier_color = match tier {
            "JIT" => Color32::from_rgb(90, 200, 120),
            "FastSim" => Color32::from_rgb(110, 170, 255),
            "Reference" | "Timing (compiling)" => Color32::from_rgb(230, 180, 60),
            _ => ui.visuals().text_color(),
        };
        ui.label(RichText::new(format!("{} cyc/s", si(s.stats.cycles_per_sec))).strong())
            .on_hover_text(format!(
                "{:.0} cycles/s, {} ticks/s, {} cycles since reset",
                s.stats.cycles_per_sec,
                si(s.stats.ticks_per_sec),
                s.stats.cycles
            ));
        ui.label(RichText::new(tier).color(tier_color).strong())
            .on_hover_text("Serving engine tier (Timing / Reference / FastSim / JIT)");
        ui.label(format!("pub {hz:.0} Hz · {:.0} µs", s.stats.publish_us))
            .on_hover_text("Snapshot publications per second and mean cost on the sim thread");
        ui.label(format!("busy {:.0}%", s.stats.busy * 100.0))
            .on_hover_text("Fraction of sim-thread wall time spent simulating");
        ui.label(format!("UI {:.2} ms", app.frame.avg_ms))
            .on_hover_text("UI code time per frame (moving average)");
        if app.wave.gaps > 0 {
            ui.label(
                RichText::new(format!("wave gaps {}", app.wave.gaps))
                    .color(Color32::from_rgb(230, 180, 60)),
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
}
