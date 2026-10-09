//! The waveform viewer (bottom tab).
//!
//! Signals come from [`AppState::wave`] (the sim host's [`crate::simhost::WaveStore`], fed by
//! shadow capture of exactly the watched signals: per event in Fast mode, per changing tick in
//! Timing mode). Add signals from the canvas context menu ([`add_signal`]); per signal a radix
//! (right-click the name). Ctrl+wheel / pinch zooms at the pointer, drag or shift+wheel
//! scrolls, a click places the cursor, a double click fits the recorded range; "Follow" keeps
//! the newest data at the right edge.
//!
//! Drawing cost is bounded by the plot width ([`super::wave_view::runs`] decimates per pixel).

use std::collections::{HashMap, VecDeque};

use egui::{Align2, Color32, FontId, Rect, Sense, Shape, Stroke, Ui, pos2, vec2};
use panda_netlist::Logic;

use crate::app::{AppState, Panel, Slot};
use crate::canvas::Palette;
use crate::simhost::{Packed, WaveStore};

use super::wave_view::{Radix, Run, RunKind, WaveView, format_value, is_known, nice_step, runs};

const NAME_W: f32 = 140.0;
const VALUE_W: f32 = 96.0;
const ROW_H: f32 = 24.0;
const AXIS_H: f32 = 20.0;

/// Names of the signals currently in the viewer.
pub fn signal_names(app: &AppState) -> Vec<String> {
    app.wave
        .config
        .signals
        .iter()
        .map(|s| s.name.clone())
        .collect()
}

fn set_names(app: &mut AppState, names: &[String]) -> usize {
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    app.set_wave_signals(&refs)
}

/// Add the hierarchical net `name` (e.g. `"PC"`, `"cpu/alu/c"`) to the viewer. False if it
/// does not resolve or is already shown. (Changing the signal set restarts capture.)
pub fn add_signal(app: &mut AppState, name: &str) -> bool {
    let mut names = signal_names(app);
    if names.iter().any(|n| n == name) {
        return false;
    }
    names.push(name.to_string());
    let want = names.len();
    set_names(app, &names) == want
}

/// Remove signal `i` from the viewer.
pub fn remove_signal(app: &mut AppState, i: usize) {
    let mut names = signal_names(app);
    if i < names.len() {
        names.remove(i);
        set_names(app, &names);
    }
}

/// The newest tick with waveform data: while running, the latest recorded change (capture runs
/// behind the simulation); when paused, the kernel tick (capture is complete).
pub fn horizon(store: &WaveStore, tick: u64, running: bool) -> u64 {
    if !running {
        return tick;
    }
    store
        .config
        .signals
        .iter()
        .zip(&store.history)
        .filter(|(s, _)| s.clock.is_none())
        .filter_map(|(_, h)| h.back().map(|&(t, _)| t))
        .max()
        .map_or(tick, |t| t.min(tick))
}

/// First recorded tick over all signals.
fn first_tick(store: &WaveStore) -> Option<u64> {
    store
        .history
        .iter()
        .filter_map(|h| h.front().map(|&(t, _)| t))
        .min()
}

enum RowAction {
    Remove(usize),
    MoveUp(usize),
}

/// The waveform tab.
#[derive(Debug)]
pub struct WaveformPanel {
    /// Axis state.
    pub view: WaveView,
    /// Radix per signal name.
    pub radix: HashMap<String, Radix>,
    last_width: f64,
}

impl WaveformPanel {
    /// An empty viewer (following, 1 tick per pixel).
    pub fn new() -> WaveformPanel {
        WaveformPanel {
            view: WaveView::default(),
            radix: HashMap::new(),
            last_width: 600.0,
        }
    }
}

impl Default for WaveformPanel {
    fn default() -> Self {
        WaveformPanel::new()
    }
}

impl Panel for WaveformPanel {
    fn id(&self) -> &'static str {
        "sim_waveform"
    }

    fn title(&self) -> String {
        "Waveforms".into()
    }

    fn slot(&self) -> Slot {
        Slot::Bottom
    }

    fn ui(&mut self, ui: &mut Ui, app: &mut AppState) {
        let (tick, running, mode) = {
            let s = &app.live.snapshot;
            (s.tick, s.running, s.mode)
        };
        let hz = horizon(&app.wave, tick, running);
        let n = app.wave.config.signals.len();
        // Header.
        ui.horizontal(|ui| {
            ui.toggle_value(&mut self.view.follow, "Follow")
                .on_hover_text("Keep the newest data at the right edge");
            if ui.button("Fit").on_hover_text("Show the whole recorded range").clicked() {
                let t0 = first_tick(&app.wave).unwrap_or(0);
                self.view.follow = false;
                self.view.fit(t0, hz.max(t0 + 1), self.last_width);
            }
            if ui.button("−").on_hover_text("Zoom out").clicked() {
                let w = self.last_width;
                self.view.zoom_at(w / 2.0, w, 0.5);
            }
            if ui.button("+").on_hover_text("Zoom in").clicked() {
                let w = self.last_width;
                self.view.zoom_at(w / 2.0, w, 2.0);
            }
            ui.label(format!("{:.3} ticks/px", self.view.tpp));
            ui.separator();
            match self.view.cursor {
                Some(c) => {
                    ui.label(format!("cursor @ {c}"));
                    if ui.small_button("×").on_hover_text("Remove cursor").clicked() {
                        self.view.cursor = None;
                    }
                }
                None => {
                    ui.weak("click the plot to place the cursor");
                }
            }
            ui.separator();
            ui.weak(match mode {
                panda_kernel::Mode::Timing => "Timing: sampled per tick",
                panda_kernel::Mode::Fast => "Fast: sampled per event",
            });
            if app.wave.gaps > 0 {
                ui.colored_label(
                    Color32::from_rgb(230, 180, 60),
                    format!("{} gap(s)", app.wave.gaps),
                )
                .on_hover_text("Capture skipped intervals while the simulation ran flat out");
            }
            if n > 0
                && ui
                    .button("Clear")
                    .on_hover_text("Remove every signal")
                    .clicked()
            {
                set_names(app, &[]);
            }
        });
        if n == 0 {
            ui.add_space(8.0);
            ui.weak("No signals. Right-click a wire or pin on the canvas → \"Add to waveform\".");
            return;
        }
        let mut action = None;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let width = ui.available_width();
                let height = AXIS_H + ROW_H * n as f32;
                let (rect, _) = ui.allocate_exact_size(vec2(width, height), Sense::hover());
                action = self.draw(ui, app, rect, hz);
            });
        match action {
            Some(RowAction::Remove(i)) => remove_signal(app, i),
            Some(RowAction::MoveUp(i)) if i > 0 => {
                let mut names = signal_names(app);
                names.swap(i - 1, i);
                set_names(app, &names);
            }
            _ => {}
        }
    }
}

impl WaveformPanel {
    fn draw(&mut self, ui: &mut Ui, app: &AppState, rect: Rect, hz: u64) -> Option<RowAction> {
        let pal = Palette::for_dark_mode(ui.visuals().dark_mode);
        let painter = ui.painter_at(rect);
        let plot = Rect::from_min_max(pos2(rect.left() + NAME_W + VALUE_W, rect.top()), rect.max);
        let width = plot.width().max(1.0) as f64;
        self.last_width = width;
        // Interaction on the plot.
        let resp = ui.interact(plot, ui.id().with("wave_plot"), Sense::click_and_drag());
        if resp.hovered() {
            let (zoom, scroll, hover) =
                ui.input(|i| (i.zoom_delta(), i.smooth_scroll_delta, i.pointer.hover_pos()));
            if let Some(p) = hover
                && (zoom - 1.0).abs() > 1e-4
            {
                self.view
                    .zoom_at((p.x - plot.left()) as f64, width, zoom as f64);
            }
            if scroll.x != 0.0 {
                self.view.scroll(scroll.x as f64);
            }
        }
        if resp.dragged() {
            self.view.scroll(resp.drag_delta().x as f64);
        }
        if resp.double_clicked() {
            let t0 = first_tick(&app.wave).unwrap_or(0);
            self.view.follow = false;
            self.view.fit(t0, hz.max(t0 + 1), width);
        } else if resp.clicked()
            && let Some(p) = resp.interact_pointer_pos()
        {
            self.view.set_cursor_at((p.x - plot.left()) as f64);
        }
        self.view.follow_to(hz, width);
        let view = self.view;
        let (t0, t1) = view.range(width);
        // Background and axis.
        painter.rect_filled(rect, 0.0, pal.bg);
        let axis = Rect::from_min_max(rect.min, pos2(rect.right(), rect.top() + AXIS_H));
        painter.rect_filled(axis, 0.0, pal.fill);
        let step = nice_step(view.tpp, 90.0);
        let first = (t0.max(0.0) / step as f64).ceil() as u64 * step;
        let mut t = first;
        while (t as f64) <= t1 {
            let x = plot.left() + view.x_of(t as f64) as f32;
            painter.line_segment(
                [pos2(x, axis.bottom()), pos2(x, rect.bottom())],
                Stroke::new(1.0, pal.grid_major),
            );
            painter.text(
                pos2(x + 3.0, axis.center().y),
                Align2::LEFT_CENTER,
                t.to_string(),
                FontId::monospace(10.0),
                pal.text_dim,
            );
            t = match t.checked_add(step) {
                Some(n) => n,
                None => break,
            };
        }
        painter.text(
            pos2(rect.left() + 6.0, axis.center().y),
            Align2::LEFT_CENTER,
            "signal",
            FontId::proportional(11.0),
            pal.text_dim,
        );
        painter.text(
            pos2(rect.left() + NAME_W + 6.0, axis.center().y),
            Align2::LEFT_CENTER,
            match view.cursor {
                Some(_) => "@ cursor",
                None => "@ latest",
            },
            FontId::proportional(11.0),
            pal.text_dim,
        );
        // Rows.
        let at = view.cursor.unwrap_or(hz);
        let mut action = None;
        for (s, sig) in app.wave.config.signals.iter().enumerate() {
            let top = axis.bottom() + s as f32 * ROW_H;
            let row = Rect::from_min_max(pos2(rect.left(), top), pos2(rect.right(), top + ROW_H));
            if s % 2 == 1 {
                painter.rect_filled(row, 0.0, pal.grid.gamma_multiply(0.5));
            }
            painter.line_segment(
                [pos2(row.left(), row.bottom()), pos2(row.right(), row.bottom())],
                Stroke::new(1.0, pal.grid),
            );
            let w = sig.width();
            let radix = self.radix.get(&sig.name).copied().unwrap_or_default();
            // Name (right-click: radix, move, remove).
            let name_rect = Rect::from_min_size(row.min, vec2(NAME_W, ROW_H));
            let label = if w > 1 {
                format!("{} [{}]", sig.name, w)
            } else {
                sig.name.clone()
            };
            painter.text(
                pos2(name_rect.left() + 6.0, name_rect.center().y),
                Align2::LEFT_CENTER,
                label,
                FontId::proportional(13.0),
                pal.text,
            );
            let nresp = ui
                .interact(name_rect, ui.id().with(("wave_name", s)), Sense::click())
                .on_hover_text(format!("{}: right-click for radix / remove", sig.name));
            nresp.context_menu(|ui| {
                if w > 1 {
                    ui.label("Radix");
                    for r in Radix::ALL {
                        if ui.radio(radix == r, r.label()).clicked() {
                            self.radix.insert(sig.name.clone(), r);
                            ui.close();
                        }
                    }
                    ui.separator();
                }
                if s > 0 && ui.button("Move up").clicked() {
                    action = Some(RowAction::MoveUp(s));
                    ui.close();
                }
                if ui.button("Remove").clicked() {
                    action = Some(RowAction::Remove(s));
                    ui.close();
                }
            });
            // Value at the cursor.
            let value = app
                .wave
                .value_at(s, at)
                .map_or_else(|| "—".to_string(), |p| format_value(p, w, radix));
            let vcolor = match app.wave.value_at(s, at) {
                Some(p) if !is_known(p, w) => pal.x,
                _ => pal.text,
            };
            painter.text(
                pos2(rect.left() + NAME_W + 6.0, row.center().y),
                Align2::LEFT_CENTER,
                value,
                FontId::monospace(12.0),
                vcolor,
            );
            // Plot.
            let lane = Rect::from_min_max(pos2(plot.left(), top), pos2(plot.right(), top + ROW_H));
            let rs = signal_runs(&app.wave, s, &view, width, hz);
            draw_runs(&painter, &pal, lane, &rs, w, radix);
        }
        // Separators of the name / value columns.
        for x in [rect.left() + NAME_W, plot.left()] {
            painter.line_segment(
                [pos2(x, rect.top()), pos2(x, rect.bottom())],
                Stroke::new(1.0, pal.grid_major),
            );
        }
        // Hover line and cursor.
        if let Some(p) = resp.hover_pos() {
            painter.line_segment(
                [pos2(p.x, axis.bottom()), pos2(p.x, rect.bottom())],
                Stroke::new(1.0, pal.text_dim.gamma_multiply(0.4)),
            );
        }
        if let Some(c) = view.cursor {
            let x = plot.left() + view.x_of(c as f64) as f32;
            if x >= plot.left() && x <= plot.right() {
                painter.line_segment(
                    [pos2(x, rect.top()), pos2(x, rect.bottom())],
                    Stroke::new(1.5, pal.accent),
                );
                let g = painter.layout_no_wrap(c.to_string(), FontId::monospace(10.0), pal.bg);
                let r = Rect::from_min_size(pos2(x + 2.0, axis.top() + 2.0), g.size()).expand(2.0);
                painter.rect_filled(r, 2.0, pal.accent);
                painter.galley(r.min + vec2(2.0, 2.0), g, pal.bg);
            }
        }
        action
    }
}

/// Drawable runs of signal `s` (clock signals are synthesized, decimated when too dense).
fn signal_runs(store: &WaveStore, s: usize, view: &WaveView, width: f64, hz: u64) -> Vec<Run> {
    let sig = &store.config.signals[s];
    if let Some(c) = sig.clock {
        let (t0, t1) = view.range(width);
        let t1 = t1.min(hz as f64);
        if t1 <= t0.max(0.0) {
            return Vec::new();
        }
        let x0 = view.x_of(t0.max(0.0)).max(0.0);
        let x1 = view.x_of(t1).min(width);
        // Fewer than ~3 px per half period: a dense block.
        if f64::from(c.high.min(c.period - c.high)) / view.tpp < 3.0 {
            return vec![Run {
                x0,
                x1,
                kind: RunKind::Dense,
            }];
        }
        let h: VecDeque<(u64, Packed)> = store
            .changes_in(s, t0.max(0.0) as u64, t1.ceil() as u64)
            .into_iter()
            .collect();
        return runs(&h, view, width, hz);
    }
    runs(&store.history[s], view, width, hz)
}

fn draw_runs(painter: &egui::Painter, pal: &Palette, lane: Rect, rs: &[Run], w: u32, radix: Radix) {
    let hi = lane.top() + 5.0;
    let lo = lane.bottom() - 5.0;
    let mid = lane.center().y;
    let x = |v: f64| lane.left() + v as f32;
    let mut prev: Option<(f32, f32)> = None; // (x1, y) of the previous 1-bit run
    for r in rs {
        let (x0, x1) = (x(r.x0), x(r.x1));
        match r.kind {
            RunKind::Dense => {
                let c = if w == 1 { pal.one } else { pal.bus };
                painter.rect_filled(
                    Rect::from_min_max(pos2(x0, hi), pos2(x1.max(x0 + 1.0), lo)),
                    0.0,
                    c.gamma_multiply(0.45),
                );
                prev = None;
            }
            RunKind::Value(p) if w == 1 => {
                let l = Logic::from_planes(p.a & 1 == 1, p.b & 1 == 1);
                let (y, color) = match l {
                    Logic::One => (hi, pal.one),
                    Logic::Zero => (lo, pal.zero.gamma_multiply(1.6)),
                    Logic::Z => (mid, pal.z),
                    Logic::X => (mid, pal.x),
                };
                if l == Logic::X {
                    painter.rect_filled(
                        Rect::from_min_max(pos2(x0, hi), pos2(x1, lo)),
                        0.0,
                        pal.x.gamma_multiply(0.3),
                    );
                }
                if let Some((px, py)) = prev
                    && (px - x0).abs() < 0.5
                    && (py - y).abs() > 0.5
                {
                    painter.line_segment([pos2(x0, py), pos2(x0, y)], Stroke::new(1.5, color));
                }
                painter.line_segment([pos2(x0, y), pos2(x1, y)], Stroke::new(1.5, color));
                prev = Some((x1, y));
            }
            RunKind::Value(p) => {
                let known = is_known(p, w);
                let color = if known { pal.bus } else { pal.x };
                let slant = 3.0f32.min((x1 - x0) / 2.0);
                let pts = vec![
                    pos2(x0, mid),
                    pos2(x0 + slant, hi),
                    pos2(x1 - slant, hi),
                    pos2(x1, mid),
                    pos2(x1 - slant, lo),
                    pos2(x0 + slant, lo),
                ];
                if !known {
                    painter.add(Shape::convex_polygon(
                        pts.clone(),
                        pal.x.gamma_multiply(0.25),
                        Stroke::NONE,
                    ));
                }
                painter.add(Shape::closed_line(pts, Stroke::new(1.2, color)));
                let text = format_value(p, w, radix);
                let g = painter.layout_no_wrap(text, FontId::monospace(11.0), pal.text);
                if g.size().x + 2.0 * slant + 4.0 <= x1 - x0 {
                    let c = pos2((x0 + x1) / 2.0, mid);
                    painter.galley(c - g.size() / 2.0, g, pal.text);
                }
                prev = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simhost::{WaveChange, WaveChunk, WaveConfig, WaveSignal};

    #[test]
    fn horizon_lags_while_running() {
        let cfg = WaveConfig {
            id: 3,
            design_gen: 1,
            signals: vec![WaveSignal {
                name: "a".into(),
                nets: vec![0],
                clock: None,
            }],
        };
        let mut st = WaveStore::new(cfg);
        assert_eq!(horizon(&st, 500, true), 500);
        st.append(&WaveChunk {
            config: 3,
            changes: vec![WaveChange {
                tick: 120,
                signal: 0,
                value: Packed::from_bool(true),
            }],
            gap: false,
        });
        assert_eq!(horizon(&st, 500, true), 120);
        assert_eq!(horizon(&st, 500, false), 500);
        assert_eq!(first_tick(&st), Some(120));
    }
}
