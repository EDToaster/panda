//! Diagnostics panel (bottom tab) and the jump overlay (canvas tool).
//!
//! The panel shows the kernel's deduplicated diagnostics ([`AppState::diagnostics`]) grouped by
//! kind, with severity filters, a text filter, per-severity counts, occurrence counts and the
//! first / last tick. Clicking a location jumps there: the view switches to the instance (the
//! shell's hierarchy navigation, [`super::jump_to`]), the net (contention colour for
//! `Contention`) or instance is highlighted through [`AppState::highlights`], and the canvas
//! centres on it.
//!
//! The pure part is [`DiagModel`] (built only when the diagnostics list or the filters change),
//! unit-tested without a GPU. [`JumpOverlay`] is an additive canvas overlay: it paints a pulsing
//! ring around the jump target (visible at every zoom level, even in the overview raster where
//! wire highlights are too thin to see), fading to a steady outline.

use std::sync::Arc;
use std::time::{Duration, Instant};

use egui::{Color32, RichText, Stroke, Ui};
use panda_kernel::{DiagEntry, DiagSource};
use panda_netlist::{DiagKind, Severity};

use crate::app::{AppState, CanvasTool, Panel, Slot};
use crate::canvas::{CanvasResponse, Palette, Scene};
use crate::doc::ItemRef;

use super::locate::{JumpTarget, Mark, mark_bounds, resolve_location};
use super::{SharedRef, jump_to};

/// Sort rank of a severity (errors first).
pub fn severity_rank(s: Severity) -> u8 {
    match s {
        Severity::Error => 0,
        Severity::Warning => 1,
        Severity::Info => 2,
    }
}

/// Colour of a severity.
pub fn severity_color(s: Severity) -> Color32 {
    match s {
        Severity::Error => crate::theme::RED,
        Severity::Warning => crate::theme::OCHRE,
        Severity::Info => crate::theme::TEAL,
    }
}

/// Lower-case name of a severity.
pub fn severity_name(s: Severity) -> &'static str {
    match s {
        Severity::Error => "error",
        Severity::Warning => "warning",
        Severity::Info => "info",
    }
}

/// A small filled circle in `color` (the default fonts have no "●" glyph).
pub fn dot(ui: &mut Ui, color: Color32) -> egui::Response {
    let h = ui.text_style_height(&egui::TextStyle::Body);
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(h * 0.6, h), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), h * 0.24, color);
    resp
}

/// Which severities are shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeverityFilter {
    /// Show errors.
    pub errors: bool,
    /// Show warnings.
    pub warnings: bool,
    /// Show infos.
    pub infos: bool,
}

impl Default for SeverityFilter {
    fn default() -> Self {
        SeverityFilter {
            errors: true,
            warnings: true,
            infos: true,
        }
    }
}

impl SeverityFilter {
    /// True if `s` is shown.
    pub fn shows(&self, s: Severity) -> bool {
        match s {
            Severity::Error => self.errors,
            Severity::Warning => self.warnings,
            Severity::Info => self.infos,
        }
    }
}

/// Diagnostics of one kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiagGroup {
    /// Kind.
    pub kind: DiagKind,
    /// Worst severity in the group.
    pub severity: Severity,
    /// Indices into the entry list (worst first, then by first tick).
    pub entries: Vec<usize>,
    /// Total occurrences.
    pub occurrences: u64,
    /// Earliest first tick of the group.
    pub first_tick: u64,
    /// Latest last tick of the group.
    pub last_tick: u64,
}

/// True if `e` matches the (case-insensitive, already lower-cased) text filter.
fn matches_text(e: &DiagEntry, needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    let d = &e.diagnostic;
    d.message.to_lowercase().contains(needle)
        || d.location
            .as_deref()
            .is_some_and(|l| l.to_lowercase().contains(needle))
        || format!("{:?}", d.kind).to_lowercase().contains(needle)
}

/// Group `entries` by kind (filtered by severity), worst groups first.
pub fn group(entries: &[DiagEntry], filter: SeverityFilter) -> Vec<DiagGroup> {
    group_matching(entries, filter, "")
}

/// [`group`] with an additional text filter (matched against message, location and kind).
pub fn group_matching(entries: &[DiagEntry], filter: SeverityFilter, text: &str) -> Vec<DiagGroup> {
    let needle = text.trim().to_lowercase();
    let mut groups: Vec<DiagGroup> = Vec::new();
    for (i, e) in entries.iter().enumerate() {
        let d = &e.diagnostic;
        if !filter.shows(d.severity) || !matches_text(e, &needle) {
            continue;
        }
        match groups.iter_mut().find(|g| g.kind == d.kind) {
            Some(g) => {
                g.entries.push(i);
                g.occurrences += e.count;
                g.first_tick = g.first_tick.min(e.first_tick);
                g.last_tick = g.last_tick.max(e.last_tick);
                if severity_rank(d.severity) < severity_rank(g.severity) {
                    g.severity = d.severity;
                }
            }
            None => groups.push(DiagGroup {
                kind: d.kind,
                severity: d.severity,
                entries: vec![i],
                occurrences: e.count,
                first_tick: e.first_tick,
                last_tick: e.last_tick,
            }),
        }
    }
    for g in &mut groups {
        g.entries.sort_by_key(|&i| {
            let e = &entries[i];
            (severity_rank(e.diagnostic.severity), e.first_tick, i)
        });
    }
    groups.sort_by_key(|g| (severity_rank(g.severity), format!("{:?}", g.kind)));
    groups
}

/// Short label of a diagnostic producer.
pub fn source_label(s: DiagSource) -> &'static str {
    match s {
        DiagSource::Design => "design",
        DiagSource::Timing => "timing sim",
        DiagSource::Fast => "fast sim",
        DiagSource::Compile => "compiler",
        DiagSource::Sta => "STA",
        DiagSource::Kernel => "kernel",
    }
}

/// `"tick 5"` or `"ticks 5–90"`.
pub fn tick_range(first: u64, last: u64) -> String {
    if first == last {
        format!("tick {first}")
    } else {
        format!("ticks {first}–{last}")
    }
}

/// True if a location names a place in the design (not e.g. `"edit op #3"`).
pub fn is_jumpable(loc: &str) -> bool {
    !loc.trim().is_empty() && !loc.starts_with("edit op #")
}

/// One row of the panel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiagRow {
    /// Index into the entry list.
    pub index: usize,
    /// Severity.
    pub severity: Severity,
    /// Location, if any.
    pub location: Option<String>,
    /// True if the location can be jumped to ([`is_jumpable`]).
    pub jumpable: bool,
    /// Message of the first occurrence.
    pub message: String,
    /// Producer, occurrences and tick range, e.g. `"timing sim · ×12 · ticks 4–90"`.
    pub meta: String,
}

/// One group of the panel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupView {
    /// The group.
    pub group: DiagGroup,
    /// Header text, e.g. `"Contention — 2 warnings · 14 occurrences · ticks 4–90"`.
    pub header: String,
    /// Rows (worst first, then by first tick).
    pub rows: Vec<DiagRow>,
}

/// Everything the diagnostics panel shows (pure; rebuilt only when inputs change).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DiagModel {
    /// Entries per severity (unfiltered): errors, warnings, infos.
    pub counts: [usize; 3],
    /// Occurrences per severity (unfiltered).
    pub occurrences: [u64; 3],
    /// Total entries (unfiltered).
    pub total: usize,
    /// Shown entries.
    pub shown: usize,
    /// Groups after filtering.
    pub groups: Vec<GroupView>,
}

impl DiagModel {
    /// Build the model of `entries` with the given filters.
    pub fn build(entries: &[DiagEntry], filter: SeverityFilter, text: &str) -> DiagModel {
        let mut m = DiagModel {
            total: entries.len(),
            ..DiagModel::default()
        };
        for e in entries {
            let k = severity_rank(e.diagnostic.severity) as usize;
            m.counts[k] += 1;
            m.occurrences[k] += e.count;
        }
        for g in group_matching(entries, filter, text) {
            let mut per = [0usize; 3];
            let rows: Vec<DiagRow> = g
                .entries
                .iter()
                .map(|&i| {
                    let e = &entries[i];
                    let d = &e.diagnostic;
                    per[severity_rank(d.severity) as usize] += 1;
                    let mut meta = source_label(e.source).to_string();
                    if e.count > 1 {
                        meta.push_str(&format!(" · ×{}", e.count));
                    }
                    meta.push_str(" · ");
                    meta.push_str(&tick_range(e.first_tick, e.last_tick));
                    DiagRow {
                        index: i,
                        severity: d.severity,
                        jumpable: d.location.as_deref().is_some_and(is_jumpable),
                        location: d.location.clone(),
                        message: super::glyph_safe(&d.message),
                        meta,
                    }
                })
                .collect();
            let sev: Vec<String> = [Severity::Error, Severity::Warning, Severity::Info]
                .into_iter()
                .zip(per)
                .filter(|&(_, n)| n > 0)
                .map(|(s, n)| format!("{n} {}{}", severity_name(s), if n == 1 { "" } else { "s" }))
                .collect();
            let header = format!(
                "{:?} — {} · {} occurrence{} · {}",
                g.kind,
                sev.join(", "),
                g.occurrences,
                if g.occurrences == 1 { "" } else { "s" },
                tick_range(g.first_tick, g.last_tick)
            );
            m.shown += rows.len();
            m.groups.push(GroupView {
                group: g,
                header,
                rows,
            });
        }
        m
    }
}

/// The diagnostics tab.
pub struct DiagPanel {
    shared: SharedRef,
    /// Severity filter.
    pub filter: SeverityFilter,
    /// Text filter (message, location, kind).
    pub text: String,
    /// Location of the last jump (shown selected).
    pub selected: Option<String>,
    /// Where the last jump went (described with breadcrumbs).
    pub last_target: Option<JumpTarget>,
    model: Option<(usize, SeverityFilter, String, Arc<DiagModel>)>,
}

impl DiagPanel {
    /// A panel sharing `shared` with the rest of the module.
    pub fn new(shared: SharedRef) -> DiagPanel {
        DiagPanel {
            shared,
            filter: SeverityFilter::default(),
            text: String::new(),
            selected: None,
            last_target: None,
            model: None,
        }
    }

    /// The model for the current diagnostics and filters (cached).
    pub fn model(&mut self, app: &AppState) -> Arc<DiagModel> {
        let key = Arc::as_ptr(&app.diagnostics) as usize;
        if let Some((k, f, t, m)) = &self.model
            && *k == key
            && *f == self.filter
            && *t == self.text
        {
            return m.clone();
        }
        let m = Arc::new(DiagModel::build(&app.diagnostics, self.filter, &self.text));
        self.model = Some((key, self.filter, self.text.clone(), m.clone()));
        m
    }

    /// Jump to `loc` (false if it does not resolve).
    pub fn jump(&mut self, app: &mut AppState, loc: &str, kind: DiagKind) -> bool {
        let Some(t) = resolve_location(&app.doc.design, app.live.flat.as_deref(), loc) else {
            app.log.push(format!("cannot locate '{loc}'"));
            return false;
        };
        jump_to(app, &self.shared, &t, kind == DiagKind::Contention);
        self.selected = Some(loc.to_string());
        self.last_target = Some(t);
        true
    }
}

impl Panel for DiagPanel {
    fn id(&self) -> &'static str {
        "sim_diagnostics"
    }

    fn title(&self) -> String {
        "Diagnostics".into()
    }

    fn slot(&self) -> Slot {
        Slot::Bottom
    }

    fn ui(&mut self, ui: &mut Ui, app: &mut AppState) {
        crate::theme::apply(ui.ctx());
        let model = self.model(app);
        ui.horizontal(|ui| {
            for (k, (label, on)) in [
                ("errors", &mut self.filter.errors),
                ("warnings", &mut self.filter.warnings),
                ("info", &mut self.filter.infos),
            ]
            .into_iter()
            .enumerate()
            {
                let s = [Severity::Error, Severity::Warning, Severity::Info][k];
                let text = RichText::new(format!("{} {label}", model.counts[k])).color(if *on {
                    severity_color(s)
                } else {
                    ui.visuals().weak_text_color()
                });
                ui.toggle_value(on, text).on_hover_text(format!(
                    "{} occurrence(s); click to show / hide",
                    model.occurrences[k]
                ));
            }
            ui.separator();
            ui.add(
                egui::TextEdit::singleline(&mut self.text)
                    .hint_text("filter")
                    .desired_width(140.0),
            );
            if self.selected.is_some()
                && ui
                    .button("Clear highlight")
                    .on_hover_text("Remove the jump highlight from the canvas")
                    .clicked()
            {
                self.shared.borrow_mut().jump_hl.clear(app);
                self.selected = None;
                self.last_target = None;
            }
            if let Some(t) = &self.last_target {
                ui.separator();
                ui.weak(format!("at {}", t.describe(&app.doc.design)));
            }
        });
        ui.separator();
        if model.total == 0 {
            ui.weak("No diagnostics.");
            return;
        }
        if model.shown == 0 {
            ui.weak(format!(
                "{} diagnostic(s) hidden by the filters.",
                model.total
            ));
            return;
        }
        let mut jump: Option<(String, DiagKind)> = None;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for g in &model.groups {
                    let head = RichText::new(&g.header)
                        .color(severity_color(g.group.severity))
                        .strong();
                    egui::CollapsingHeader::new(head)
                        .id_salt(("diag_group", format!("{:?}", g.group.kind)))
                        .default_open(true)
                        .show(ui, |ui| {
                            for r in &g.rows {
                                ui.horizontal_wrapped(|ui| {
                                    dot(ui, severity_color(r.severity));
                                    match &r.location {
                                        Some(loc) if r.jumpable => {
                                            let sel = self.selected.as_deref() == Some(loc);
                                            if ui
                                                .selectable_label(
                                                    sel,
                                                    RichText::new(loc).monospace(),
                                                )
                                                .on_hover_text("Jump to this location")
                                                .clicked()
                                            {
                                                jump = Some((loc.clone(), g.group.kind));
                                            }
                                        }
                                        Some(loc) => {
                                            ui.weak(RichText::new(loc).monospace());
                                        }
                                        None => {}
                                    }
                                    ui.label(&r.message);
                                    ui.weak(&r.meta);
                                });
                            }
                        });
                }
            });
        if let Some((loc, kind)) = jump {
            self.jump(app, &loc, kind);
        }
    }
}

/// Time the jump ring pulses before it settles to a steady outline.
pub const PULSE: Duration = Duration::from_millis(2400);

/// Canvas overlay: a ring around each net / item of the last jump ([`super::Shared::jump_hl`]),
/// pulsing for [`PULSE`] after the jump, then a thin steady outline until the highlight is
/// cleared. Paints only; never handles input.
pub struct JumpOverlay {
    shared: SharedRef,
    key: Vec<String>,
    since: Instant,
}

impl JumpOverlay {
    /// An overlay reading the jump highlights of `shared`.
    pub fn new(shared: SharedRef) -> JumpOverlay {
        JumpOverlay {
            shared,
            key: Vec::new(),
            since: Instant::now(),
        }
    }

    /// Marks of the current jump highlight (sorted) and whether it is a contention.
    fn marks(&self) -> (Vec<Mark>, bool) {
        let sh = self.shared.borrow();
        let hl = &sh.jump_hl;
        let mut v: Vec<Mark> = hl.nets.iter().cloned().map(Mark::Net).collect();
        v.extend(hl.items.iter().cloned().map(Mark::Item));
        v.sort_by_key(mark_key);
        (v, !hl.contention.is_empty())
    }
}

fn mark_key(m: &Mark) -> String {
    match m {
        Mark::Net(n) => format!("n:{n}"),
        Mark::Item(ItemRef::Instance(i)) => format!("i:{i}"),
        Mark::Item(ItemRef::Pin(p)) => format!("p:{p}"),
        Mark::None => String::new(),
    }
}

/// Pulse amplitude in `[0, 1]` at `t` after the jump (0 once [`PULSE`] passed).
pub fn pulse_at(t: Duration) -> f32 {
    if t >= PULSE {
        return 0.0;
    }
    let fade = 1.0 - t.as_secs_f32() / PULSE.as_secs_f32();
    let phase = (t.as_secs_f32() * std::f32::consts::TAU * 1.5).cos() * 0.5 + 0.5;
    fade * phase
}

impl CanvasTool for JumpOverlay {
    fn id(&self) -> &'static str {
        "sim_jump_overlay"
    }

    fn canvas(&mut self, ui: &mut Ui, canvas: &CanvasResponse, scene: &Scene, _app: &mut AppState) {
        let (marks, contention) = self.marks();
        if marks.is_empty() {
            self.key.clear();
            return;
        }
        let key: Vec<String> = marks.iter().map(mark_key).collect();
        if key != self.key {
            self.key = key;
            self.since = Instant::now();
        }
        let pal = Palette::for_dark_mode(ui.visuals().dark_mode);
        let color = if contention {
            pal.contention
        } else {
            pal.accent
        };
        let t = self.since.elapsed();
        let p = pulse_at(t);
        let painter = ui.painter_at(canvas.rect);
        for m in &marks {
            let Some(b) = mark_bounds(scene, m) else {
                continue;
            };
            let r = canvas.xf.rect_to_screen(&b).expand(6.0 + 10.0 * p);
            let r = if r.width() < 14.0 || r.height() < 14.0 {
                egui::Rect::from_center_size(r.center(), r.size().max(egui::vec2(14.0, 14.0)))
            } else {
                r
            };
            if p > 0.0 {
                painter.rect_filled(r, 6.0, color.gamma_multiply(0.10 * p));
            }
            painter.rect_stroke(
                r,
                6.0,
                Stroke::new(1.5 + 2.0 * p, color.gamma_multiply(0.55 + 0.45 * p)),
                egui::StrokeKind::Outside,
            );
        }
        if t < PULSE {
            ui.ctx().request_repaint();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use panda_netlist::Diagnostic;

    fn entry(kind: DiagKind, sev: Severity, count: u64, first: u64) -> DiagEntry {
        DiagEntry {
            diagnostic: Diagnostic {
                kind,
                severity: sev,
                message: format!("{kind:?} message"),
                location: Some(format!("n{first}")),
            },
            source: DiagSource::Timing,
            count,
            first_tick: first,
            last_tick: first + count - 1,
        }
    }

    #[test]
    fn groups_by_kind_worst_first() {
        let es = vec![
            entry(DiagKind::Contention, Severity::Warning, 3, 50),
            entry(DiagKind::SetupViolation, Severity::Error, 1, 0),
            entry(DiagKind::Contention, Severity::Warning, 2, 10),
            entry(DiagKind::DelaySensitive, Severity::Info, 1, 0),
            entry(DiagKind::Contention, Severity::Error, 1, 99),
        ];
        let g = group(&es, SeverityFilter::default());
        let kinds: Vec<_> = g.iter().map(|g| g.kind).collect();
        assert_eq!(
            kinds,
            [
                DiagKind::Contention,
                DiagKind::SetupViolation,
                DiagKind::DelaySensitive
            ]
        );
        assert_eq!(g[0].severity, Severity::Error);
        assert_eq!(g[0].entries, vec![4, 2, 0]);
        assert_eq!(g[0].occurrences, 6);
        assert_eq!((g[0].first_tick, g[0].last_tick), (10, 99));
        let only_warn = SeverityFilter {
            errors: false,
            warnings: true,
            infos: false,
        };
        let g = group(&es, only_warn);
        assert_eq!(g.len(), 1);
        assert_eq!((g[0].severity, g[0].entries.len()), (Severity::Warning, 2));
        // Text filter.
        let g = group_matching(&es, SeverityFilter::default(), "SETUP");
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].kind, DiagKind::SetupViolation);
        assert_eq!(
            group_matching(&es, SeverityFilter::default(), "n50")[0].entries,
            [0]
        );
    }

    #[test]
    fn model_counts_rows_and_meta() {
        let mut es = vec![
            entry(DiagKind::Contention, Severity::Warning, 3, 50),
            entry(DiagKind::Contention, Severity::Warning, 1, 7),
            entry(DiagKind::SetupViolation, Severity::Error, 1, 0),
        ];
        es[2].diagnostic.location = Some("edit op #2".into());
        es[2].source = DiagSource::Design;
        let m = DiagModel::build(&es, SeverityFilter::default(), "");
        assert_eq!((m.total, m.shown), (3, 3));
        assert_eq!(m.counts, [1, 2, 0]);
        assert_eq!(m.occurrences, [1, 4, 0]);
        assert_eq!(m.groups.len(), 2);
        let c = &m.groups[1];
        assert_eq!(c.group.kind, DiagKind::Contention);
        assert_eq!(
            c.header,
            "Contention — 2 warnings · 4 occurrences · ticks 7–52"
        );
        assert_eq!(c.rows[0].index, 1);
        assert_eq!(c.rows[0].meta, "timing sim · tick 7");
        assert_eq!(c.rows[1].meta, "timing sim · ×3 · ticks 50–52");
        assert!(c.rows[0].jumpable);
        let e = &m.groups[0];
        assert_eq!(e.header, "SetupViolation — 1 error · 1 occurrence · tick 0");
        assert!(!e.rows[0].jumpable, "edit-op locations are not places");
        let hidden = DiagModel::build(
            &es,
            SeverityFilter {
                errors: false,
                warnings: false,
                infos: true,
            },
            "",
        );
        assert_eq!((hidden.total, hidden.shown), (3, 0));
        assert_eq!(hidden.counts, [1, 2, 0], "counts ignore the filter");
    }

    #[test]
    fn pulse_fades_out() {
        assert!(pulse_at(Duration::ZERO) > 0.99);
        assert_eq!(pulse_at(PULSE), 0.0);
        assert!(pulse_at(PULSE - Duration::from_millis(1)) < 0.01);
        assert_eq!(tick_range(3, 3), "tick 3");
        assert_eq!(tick_range(3, 9), "ticks 3–9");
    }
}
