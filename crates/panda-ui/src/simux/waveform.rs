//! The waveform viewer (bottom tab).
//!
//! Signals come from [`AppState::wave`] (the sim host's [`crate::simhost::WaveStore`], fed by
//! shadow capture of exactly the watched signals: per event in Fast mode, per changing tick in
//! Timing mode). Add signals from the canvas context menu ([`add_signal`], also the hook for
//! other panels such as the inspector) or the "Add signal" picker; per signal a radix
//! (right-click the name).
//!
//! Navigation: the mouse wheel over the plot scrolls in time, Ctrl+wheel / pinch zooms at the
//! pointer, dragging scrolls, a click places the cursor (values are read out at the cursor), a
//! double click fits the recorded range; "Follow" keeps the newest data at the right edge. The
//! time axis is in ticks (default in Timing mode) or in cycles of a reference clock (default in
//! Fast mode). Intervals the capture lost (the shadow replay fell behind) are hatched.
//!
//! Drawing cost is bounded by the plot width ([`super::wave_view::runs`] decimates per pixel
//! column; rows scrolled out of view are skipped).

use std::collections::{HashMap, VecDeque};

use egui::{Align2, Color32, FontId, Rect, Sense, Shape, Stroke, Ui, Vec2, pos2, vec2};
use panda_kernel::Mode;
use panda_netlist::Logic;
use panda_netlist::flat::{FlatNetlist, SourceKind};

use crate::app::{AppState, Panel, Slot};
use crate::canvas::Palette;
use crate::simhost::{ClockSpec, Packed, SimEvent, WaveStore};

use super::wave_view::{
    GapTracker, Radix, Run, RunKind, TimeUnit, WaveView, axis_marks, filter_candidates,
    format_time, format_value, is_known, runs, signal_candidates,
};

const NAME_W: f32 = 150.0;
const VALUE_W: f32 = 96.0;
const ROW_H: f32 = 24.0;
const AXIS_H: f32 = 20.0;
/// Approximate advance of the 11 pt monospace font (to skip laying out labels that cannot fit).
const LABEL_CHAR_W: f32 = 6.7;
/// Max entries listed by the signal picker.
const PICKER_LIMIT: usize = 200;
/// Colour of capture-gap hatching.
const GAP_COLOR: Color32 = crate::theme::OCHRE;

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

/// The first `Clock` source of a netlist (the reference clock for the cycle axis when no clock
/// signal is in the viewer).
pub fn first_clock(flat: &FlatNetlist) -> Option<ClockSpec> {
    flat.sources.iter().find_map(|s| match s.kind {
        SourceKind::Clock {
            period,
            high,
            phase,
        } => Some(ClockSpec {
            period,
            high,
            phase,
        }),
        _ => None,
    })
}

enum RowAction {
    Remove(usize),
    MoveUp(usize),
}

/// Signal-picker state: candidate names cached per design generation.
#[derive(Debug, Default)]
struct Picker {
    query: String,
    design_gen: u64,
    cands: Vec<String>,
    clock: Option<ClockSpec>,
}

impl Picker {
    fn refresh(&mut self, app: &AppState) {
        let Some(flat) = &app.live.flat else {
            return;
        };
        if self.design_gen == app.live.design_gen && !self.cands.is_empty() {
            return;
        }
        self.design_gen = app.live.design_gen;
        self.cands = signal_candidates(flat.net_names.iter().map(String::as_str));
        self.clock = first_clock(flat);
    }
}

/// The waveform tab.
#[derive(Debug)]
pub struct WaveformPanel {
    /// Axis state.
    pub view: WaveView,
    /// Radix per signal name.
    pub radix: HashMap<String, Radix>,
    /// Time unit chosen by the user (`None`: cycles in Fast mode, ticks in Timing mode).
    pub unit: Option<TimeUnit>,
    /// Capture gaps of the current config.
    pub gaps: GapTracker,
    picker: Picker,
    last_width: f64,
}

impl WaveformPanel {
    /// An empty viewer (following, 1 tick per pixel).
    pub fn new() -> WaveformPanel {
        WaveformPanel {
            view: WaveView::default(),
            radix: HashMap::new(),
            unit: None,
            gaps: GapTracker::default(),
            picker: Picker::default(),
            last_width: 600.0,
        }
    }

    /// The reference clock for the cycle axis: the first clock signal shown, else the design's
    /// first clock.
    fn ref_clock(&self, app: &AppState) -> Option<ClockSpec> {
        app.wave
            .config
            .signals
            .iter()
            .find_map(|s| s.clock)
            .or(self.picker.clock)
    }

    /// The unit in effect (cycles need a clock).
    fn unit(&self, mode: Mode, clk: Option<&ClockSpec>) -> TimeUnit {
        let u = self.unit.unwrap_or(match mode {
            Mode::Fast => TimeUnit::Cycles,
            Mode::Timing => TimeUnit::Ticks,
        });
        if clk.is_none() { TimeUnit::Ticks } else { u }
    }

    fn header(
        &mut self,
        ui: &mut Ui,
        app: &mut AppState,
        hz: u64,
        unit: TimeUnit,
        clk: Option<ClockSpec>,
    ) {
        let mode = app.live.snapshot.mode;
        let n = app.wave.config.signals.len();
        ui.horizontal(|ui| {
            self.picker_button(ui, app);
            ui.separator();
            ui.toggle_value(&mut self.view.follow, "Follow")
                .on_hover_text("Keep the newest data at the right edge");
            if ui
                .button("Fit")
                .on_hover_text("Show the whole recorded range")
                .clicked()
            {
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
            match (unit, clk) {
                (TimeUnit::Cycles, Some(c)) => ui.label(format!(
                    "{:.3} cycles/px",
                    self.view.tpp / f64::from(c.period.max(1))
                )),
                _ => ui.label(format!("{:.3} ticks/px", self.view.tpp)),
            };
            egui::ComboBox::from_id_salt("wave_unit")
                .selected_text(unit.label())
                .width(70.0)
                .show_ui(ui, |ui| {
                    for u in [TimeUnit::Ticks, TimeUnit::Cycles] {
                        let enabled = u == TimeUnit::Ticks || clk.is_some();
                        if ui
                            .add_enabled(enabled, egui::Button::selectable(unit == u, u.label()))
                            .clicked()
                        {
                            self.unit = Some(u);
                        }
                    }
                })
                .response
                .on_hover_text(match clk {
                    Some(c) => format!("Time axis unit (cycle = {} ticks)", c.period),
                    None => "Time axis unit (no clock in the design: ticks only)".into(),
                });
            ui.separator();
            match self.view.cursor {
                Some(c) => {
                    let t = format_time(c, unit, clk.as_ref());
                    let txt = if unit == TimeUnit::Cycles {
                        format!("cursor @ {t} (tick {c})")
                    } else {
                        format!("cursor @ {t}")
                    };
                    ui.label(txt);
                    if ui
                        .small_button("×")
                        .on_hover_text("Remove cursor")
                        .clicked()
                    {
                        self.view.cursor = None;
                    }
                }
                None => {
                    ui.weak("click the plot to place the cursor");
                }
            }
            ui.separator();
            ui.weak(match mode {
                Mode::Timing => "Timing: sampled per tick",
                Mode::Fast => "Fast: sampled per event",
            });
            let gaps = self.gap_count(app);
            if gaps > 0 {
                ui.colored_label(GAP_COLOR, format!("⚠ {gaps} capture gap(s)"))
                    .on_hover_text(
                        "Capture skipped intervals while the simulation ran flat out \
                         (hatched in the plot). Pause or limit the speed for complete waveforms.",
                    );
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
    }

    fn gap_count(&self, app: &AppState) -> usize {
        if self.gaps.config == app.wave.config.id {
            self.gaps.gaps.len().max(app.wave.gaps.min(1) as usize)
        } else {
            app.wave.gaps as usize
        }
    }

    fn picker_button(&mut self, ui: &mut Ui, app: &mut AppState) {
        self.picker.refresh(app);
        ui.menu_button("+ Add signal", |ui| {
            ui.set_min_width(240.0);
            let r = ui.add(
                egui::TextEdit::singleline(&mut self.picker.query)
                    .hint_text("filter nets (e.g. alu c)")
                    .desired_width(230.0),
            );
            if !r.has_focus() && self.picker.query.is_empty() {
                r.request_focus();
            }
            let shown = signal_names(app);
            let hits = filter_candidates(&self.picker.cands, &self.picker.query, PICKER_LIMIT + 1);
            let more = hits.len() > PICKER_LIMIT;
            let mut add = None;
            egui::ScrollArea::vertical()
                .max_height(260.0)
                .show(ui, |ui| {
                    for name in hits.iter().take(PICKER_LIMIT) {
                        if shown.iter().any(|s| s == name) {
                            ui.weak(format!("✔ {name}"));
                        } else if ui.button(*name).clicked() {
                            add = Some(name.to_string());
                        }
                    }
                });
            if hits.is_empty() {
                ui.weak("no matching net");
            } else if more {
                ui.weak(format!("first {PICKER_LIMIT} shown; refine the filter"));
            }
            if let Some(name) = add
                && !add_signal(app, &name)
            {
                app.log.push(format!("cannot watch '{name}'"));
            }
        });
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

    fn on_event(&mut self, ev: &SimEvent, _app: &mut AppState) {
        if let SimEvent::Wave(c) = ev {
            self.gaps.on_chunk(c);
        }
    }

    fn ui(&mut self, ui: &mut Ui, app: &mut AppState) {
        crate::theme::apply(ui.ctx());
        let (tick, running, mode) = {
            let s = &app.live.snapshot;
            (s.tick, s.running, s.mode)
        };
        self.picker.refresh(app);
        let hz = horizon(&app.wave, tick, running);
        let clk = self.ref_clock(app);
        let unit = self.unit(mode, clk.as_ref());
        self.header(ui, app, hz, unit, clk);
        let n = app.wave.config.signals.len();
        if n == 0 {
            ui.add_space(8.0);
            ui.weak(
                "No signals. Use \"+ Add signal\", or right-click a wire or pin on the canvas → \
                 \"Add to waveform\".",
            );
            return;
        }
        let mut action = None;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let width = ui.available_width();
                let height = AXIS_H + ROW_H * n as f32;
                let (rect, _) = ui.allocate_exact_size(vec2(width, height), Sense::hover());
                action = self.draw(ui, app, rect, hz, unit, clk);
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
    /// Wheel / zoom / drag / click handling on the plot.
    fn interact(&mut self, ui: &mut Ui, plot: Rect, app: &AppState, hz: u64) -> egui::Response {
        let width = plot.width().max(1.0) as f64;
        let resp = ui.interact(plot, ui.id().with("wave_plot"), Sense::click_and_drag());
        if resp.hovered() {
            let (zoom, scroll, hover) =
                ui.input(|i| (i.zoom_delta(), i.smooth_scroll_delta, i.pointer.hover_pos()));
            if let Some(p) = hover
                && (zoom - 1.0).abs() > 1e-4
            {
                self.view
                    .zoom_at((p.x - plot.left()) as f64, width, zoom as f64);
            } else if scroll != Vec2::ZERO {
                // Both wheel axes scroll in time over the plot (rows scroll over the names).
                self.view.scroll((scroll.x + scroll.y) as f64);
                ui.input_mut(|i| i.smooth_scroll_delta = Vec2::ZERO);
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
        resp
    }

    fn draw(
        &mut self,
        ui: &mut Ui,
        app: &AppState,
        rect: Rect,
        hz: u64,
        unit: TimeUnit,
        clk: Option<ClockSpec>,
    ) -> Option<RowAction> {
        let pal = Palette::for_dark_mode(ui.visuals().dark_mode);
        let painter = ui.painter_at(rect);
        let plot = Rect::from_min_max(pos2(rect.left() + NAME_W + VALUE_W, rect.top()), rect.max);
        let width = plot.width().max(1.0) as f64;
        self.last_width = width;
        let resp = self.interact(ui, plot, app, hz);
        self.view.follow_to(hz, width);
        let view = self.view;
        let (t0, t1) = view.range(width);
        let px = |t: f64| plot.left() + view.x_of(t) as f32;
        // Background and axis.
        painter.rect_filled(rect, 0.0, pal.bg);
        let axis = Rect::from_min_max(rect.min, pos2(rect.right(), rect.top() + AXIS_H));
        painter.rect_filled(axis, 0.0, pal.fill);
        for m in axis_marks(&view, width, 90.0, unit, clk.as_ref()) {
            let x = px(m.tick as f64);
            painter.line_segment(
                [pos2(x, axis.bottom()), pos2(x, rect.bottom())],
                Stroke::new(1.0, pal.grid_major),
            );
            painter.text(
                pos2(x + 3.0, axis.center().y),
                Align2::LEFT_CENTER,
                m.label,
                FontId::monospace(10.0),
                pal.text_dim,
            );
        }
        // Past the horizon (not simulated / not captured yet).
        let xh = px(hz as f64);
        if xh < plot.right() {
            painter.rect_filled(
                Rect::from_min_max(pos2(xh.max(plot.left()), axis.bottom()), plot.max),
                0.0,
                pal.grid.gamma_multiply(0.35),
            );
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
        // Rows (only those inside the scroll area's viewport).
        let clip = ui.clip_rect();
        let at = view.cursor.unwrap_or(hz);
        let mut action = None;
        for (s, sig) in app.wave.config.signals.iter().enumerate() {
            let top = axis.bottom() + s as f32 * ROW_H;
            let row = Rect::from_min_max(pos2(rect.left(), top), pos2(rect.right(), top + ROW_H));
            if !clip.intersects(row) {
                continue;
            }
            if s % 2 == 1 {
                painter.rect_filled(row, 0.0, pal.grid.gamma_multiply(0.5));
            }
            painter.line_segment(
                [
                    pos2(row.left(), row.bottom()),
                    pos2(row.right(), row.bottom()),
                ],
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
            painter.with_clip_rect(name_rect.shrink(2.0)).text(
                pos2(name_rect.left() + 6.0, name_rect.center().y),
                Align2::LEFT_CENTER,
                label,
                FontId::proportional(13.0),
                pal.text,
            );
            if w > 1 && radix != Radix::Hex {
                painter.text(
                    pos2(name_rect.right() - 4.0, name_rect.center().y),
                    Align2::RIGHT_CENTER,
                    radix_tag(radix),
                    FontId::monospace(9.0),
                    pal.text_dim,
                );
            }
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
            let v = app.wave.value_at(s, at);
            let value = v.map_or_else(|| "—".to_string(), |p| format_value(p, w, radix));
            let vcolor = v.map_or(pal.text_dim, |p| value_color(&pal, p, w, pal.text));
            painter
                .with_clip_rect(Rect::from_min_size(
                    pos2(rect.left() + NAME_W, top),
                    vec2(VALUE_W, ROW_H),
                ))
                .text(
                    pos2(rect.left() + NAME_W + 6.0, row.center().y),
                    Align2::LEFT_CENTER,
                    value,
                    FontId::monospace(12.0),
                    vcolor,
                );
            // Plot.
            let lane = Rect::from_min_max(pos2(plot.left(), top), pos2(plot.right(), top + ROW_H));
            let rs = signal_runs(&app.wave, s, &view, width, hz);
            draw_runs(&painter.with_clip_rect(lane), &pal, lane, &rs, w, radix);
        }
        // Capture gaps (only gaps of the config shown).
        let mut hovered_gap = None;
        let hover = resp.hover_pos();
        if self.gaps.config == app.wave.config.id {
            let gp =
                painter.with_clip_rect(Rect::from_min_max(pos2(plot.left(), rect.top()), rect.max));
            for (a, b) in self.gaps.visible(t0, t1) {
                let x0 = px(a as f64);
                let x1 = px(b as f64).max(x0 + 3.0);
                let band = Rect::from_min_max(pos2(x0, axis.bottom()), pos2(x1, rect.bottom()));
                draw_hatch(&gp, band, GAP_COLOR);
                gp.rect_filled(
                    Rect::from_min_max(pos2(x0, axis.top()), pos2(x1, axis.bottom())),
                    0.0,
                    GAP_COLOR.gamma_multiply(0.8),
                );
                if hover.is_some_and(|p| p.x >= x0 - 2.0 && p.x <= x1 + 2.0) {
                    hovered_gap = Some((a, b));
                }
            }
        }
        // Separators of the name / value columns.
        for x in [rect.left() + NAME_W, plot.left()] {
            painter.line_segment(
                [pos2(x, rect.top()), pos2(x, rect.bottom())],
                Stroke::new(1.0, pal.grid_major),
            );
        }
        // Hover line (+ time readout) and cursor.
        if let Some(p) = hover {
            painter.line_segment(
                [pos2(p.x, axis.bottom()), pos2(p.x, rect.bottom())],
                Stroke::new(1.0, pal.text_dim.gamma_multiply(0.4)),
            );
            let ht = view.tick_at((p.x - plot.left()) as f64).round().max(0.0) as u64;
            let mut txt = format_time(ht, unit, clk.as_ref());
            if let Some(c) = view.cursor {
                let d = ht as i128 - c as i128;
                txt.push_str(&format!("  Δ{d:+}"));
            }
            let g = painter.layout_no_wrap(txt, FontId::monospace(10.0), pal.text);
            let r =
                Rect::from_min_size(pos2(p.x + 4.0, rect.bottom() - g.size().y - 4.0), g.size())
                    .expand(2.0);
            painter.rect_filled(r, 2.0, pal.fill);
            painter.galley(r.min + vec2(2.0, 2.0), g, pal.text);
        }
        if let Some(c) = view.cursor {
            let x = px(c as f64);
            if x >= plot.left() && x <= plot.right() {
                painter.line_segment(
                    [pos2(x, rect.top()), pos2(x, rect.bottom())],
                    Stroke::new(1.5, pal.accent),
                );
                let g = painter.layout_no_wrap(
                    format_time(c, unit, clk.as_ref()),
                    FontId::monospace(10.0),
                    pal.bg,
                );
                let r = Rect::from_min_size(pos2(x + 2.0, axis.top() + 2.0), g.size()).expand(2.0);
                painter.rect_filled(r, 2.0, pal.accent);
                painter.galley(r.min + vec2(2.0, 2.0), g, pal.bg);
            }
        }
        if let Some((a, b)) = hovered_gap {
            resp.on_hover_text_at_pointer(format!(
                "capture gap: no samples between tick {a} and {b} (the shadow capture fell behind)"
            ));
        }
        action
    }
}

/// Short radix tag shown next to a bus name.
fn radix_tag(r: Radix) -> &'static str {
    match r {
        Radix::Hex => "hex",
        Radix::Bin => "bin",
        Radix::Dec => "dec",
        Radix::Signed => "±dec",
    }
}

/// Colour of a value: `known` for 0/1 values, Z colour if every bit is Z, X colour otherwise.
fn value_color(pal: &Palette, p: Packed, w: u32, known: Color32) -> Color32 {
    if is_known(p, w) {
        known
    } else if all_z(p, w) {
        pal.z
    } else {
        pal.x
    }
}

fn all_z(p: Packed, w: u32) -> bool {
    let m = if w >= 64 { u64::MAX } else { (1u64 << w) - 1 };
    p.b & m == m && p.a & m == 0
}

/// Diagonal hatching over `band`.
fn draw_hatch(painter: &egui::Painter, band: Rect, color: Color32) {
    painter.rect_filled(band, 0.0, color.gamma_multiply(0.12));
    let h = band.height();
    let step = 8.0;
    let mut x = band.left() - h;
    let p = painter.with_clip_rect(band.intersect(painter.clip_rect()));
    while x < band.right() {
        p.line_segment(
            [pos2(x, band.bottom()), pos2(x + h, band.top())],
            Stroke::new(1.0, color.gamma_multiply(0.5)),
        );
        x += step;
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
                let color = value_color(pal, p, w, pal.bus);
                let slant = 3.0f32.min((x1 - x0) / 2.0);
                if x1 - x0 < 4.0 {
                    // Too narrow for the hexagon outline: a block like a dense run (so a busy
                    // bus reads as one band, not as noise).
                    painter.rect_filled(
                        Rect::from_min_max(pos2(x0, hi), pos2(x1.max(x0 + 1.0), lo)),
                        0.0,
                        color.gamma_multiply(0.45),
                    );
                    prev = None;
                    continue;
                }
                let pts = vec![
                    pos2(x0, mid),
                    pos2(x0 + slant, hi),
                    pos2(x1 - slant, hi),
                    pos2(x1, mid),
                    pos2(x1 - slant, lo),
                    pos2(x0 + slant, lo),
                ];
                if !is_known(p, w) {
                    painter.add(Shape::convex_polygon(
                        pts.clone(),
                        color.gamma_multiply(0.25),
                        Stroke::NONE,
                    ));
                }
                painter.add(Shape::closed_line(pts, Stroke::new(1.2, color)));
                // Value label, centred in the visible part of the run.
                let room = x1.min(lane.right()) - x0.max(lane.left()) - 2.0 * slant - 4.0;
                if room >= LABEL_CHAR_W {
                    let text = format_value(p, w, radix);
                    let chars = text.chars().count() as f32;
                    let fit = (room / LABEL_CHAR_W).floor() as usize;
                    let text = if chars * LABEL_CHAR_W <= room {
                        text
                    } else if fit >= 2 {
                        // Keep the low digits (they change most), elide the rest.
                        let tail: String =
                            text.chars().skip(text.chars().count() + 1 - fit).collect();
                        format!("…{tail}")
                    } else {
                        String::new()
                    };
                    if !text.is_empty() {
                        let g = painter.layout_no_wrap(text, FontId::monospace(11.0), pal.text);
                        let cx = (x0.max(lane.left()) + x1.min(lane.right())) / 2.0;
                        painter.galley(pos2(cx, mid) - g.size() / 2.0, g, pal.text);
                    }
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

    #[test]
    fn z_colouring() {
        assert!(all_z(Packed { a: 0, b: 0xff }, 8));
        assert!(!all_z(Packed { a: 1, b: 0xff }, 8));
        assert!(!all_z(Packed { a: 0, b: 0x7f }, 8));
    }
}
