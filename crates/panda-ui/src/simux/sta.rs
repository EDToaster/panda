//! STA panel (bottom tab): minimum period per clock with an "apply suggested period" button,
//! the critical path (highlighted on the canvas, in whatever view is shown; the overall path or
//! that of any analyzed event), per-event settle times and the timing violations (click a
//! location to jump there).
//!
//! Everything expensive runs on a worker thread: the analysis uses its **own** [`Kernel`]
//! (loaded from a copy of the document, `panda-sta` through [`Kernel::timing_report`]), and the
//! worker also builds the panel's [`StaModel`] (rows, clock sources for the period edits), so
//! neither the UI nor the sim thread waits for it. The UI thread only maps the chosen path's
//! flat-net names onto the nets of the shown view, once per view / result change.
//!
//! "Apply" changes the `Clock` atom driving the clock net (found anywhere in the hierarchy,
//! [`find_clock`]) through the editing core ([`Editor::set_atom`]: wires follow, kernel
//! rejection rolls back) on the panel's own undo history ("Undo period" button).

use std::collections::HashSet;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

use egui::{Color32, RichText, Ui};
use panda_kernel::{EditOp, Kernel, KernelOptions, TimingSummary};
use panda_netlist::flat::FlatNetlist;
use panda_netlist::{
    Atom, ComponentDef, DefId, Design, DiagKind, Diagnostic, Endpoint, InstKind, Severity,
};

use crate::app::{AppState, Panel, Slot};
use crate::edit::{EditTarget, Editor};
use crate::hier::nav::child_path;
use crate::simhost::SimEvent;

use super::diag::{dot, severity_color, severity_rank};
use super::locate::{nets_in_view, path_entry_net, resolve_location};
use super::{SharedRef, jump_to};

/// Colour for "meets timing".
const GOOD: Color32 = crate::theme::TEAL;

/// A `Clock` atom somewhere in the hierarchy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClockSource {
    /// Instance path of the def holding the atom (`""` = top).
    pub path: String,
    /// That def.
    pub def: DefId,
    /// The atom instance id.
    pub instance: String,
    /// Hierarchical name of its output net (`"path/net"`).
    pub net: String,
    /// Configured period.
    pub period: u32,
    /// Configured high time.
    pub high: u32,
    /// Configured phase.
    pub phase: u32,
}

/// Every `Clock` atom of the design, depth-first in instance order. (Validation only allows
/// clocks in the top def; the walk does not rely on it.)
pub fn clock_sources(design: &Design) -> Vec<ClockSource> {
    fn walk(design: &Design, d: &ComponentDef, path: &str, depth: u32, out: &mut Vec<ClockSource>) {
        if depth > 64 {
            return;
        }
        for inst in &d.instances {
            match &inst.kind {
                InstKind::Atom(Atom::Clock {
                    period,
                    high,
                    phase,
                }) => {
                    let net = d.nets.iter().find(|n| {
                        n.ends.iter().any(|e| {
                            matches!(e, Endpoint::Port { inst: i, port } if i == &inst.id && port == "out")
                        })
                    });
                    if let Some(net) = net {
                        out.push(ClockSource {
                            path: path.to_string(),
                            def: d.id.clone(),
                            instance: inst.id.clone(),
                            net: child_path(path, &net.id),
                            period: *period,
                            high: *high,
                            phase: *phase,
                        });
                    }
                }
                InstKind::Def(sub) => {
                    if let Some(sd) = design.def(sub) {
                        walk(design, sd, &child_path(path, &inst.id), depth + 1, out);
                    }
                }
                InstKind::Atom(_) => {}
            }
        }
    }
    let mut out = Vec::new();
    if let Some(top) = design.top_def() {
        walk(design, top, "", 0, &mut out);
    }
    out
}

/// The clock source driving clock net `clock` (an STA clock name: any alias of the flat net;
/// without `flat` the hierarchical net name must match exactly).
pub fn find_clock(design: &Design, flat: Option<&FlatNetlist>, clock: &str) -> Option<ClockSource> {
    let srcs = clock_sources(design);
    if let Some(f) = flat
        && let Some(want) = f.net_by_name(clock)
    {
        return srcs
            .into_iter()
            .find(|s| f.net_by_name(&s.net) == Some(want));
    }
    srcs.into_iter().find(|s| s.net == clock)
}

/// The `Clock` atom with `period`, scaling high time and phase proportionally (the STA
/// suggestion keeps the high/period ratio). `period` is at least 2; high stays in
/// `1..period`.
pub fn scaled_clock(src: &ClockSource, period: u32) -> Atom {
    let period = period.max(2);
    let old = u64::from(src.period.max(1));
    let high = (u64::from(src.high) * u64::from(period)).div_ceil(old) as u32;
    let phase = (u64::from(src.phase) * u64::from(period) / old) as u32;
    Atom::Clock {
        period,
        high: high.clamp(1, period - 1),
        phase: phase % period,
    }
}

/// The raw edit that sets the period of the `Clock` source driving `clock` to `period`
/// ([`find_clock`], [`scaled_clock`]).
pub fn clock_period_edit(
    design: &Design,
    flat: Option<&FlatNetlist>,
    clock: &str,
    period: u32,
) -> Option<EditOp> {
    let src = find_clock(design, flat, clock)?;
    Some(EditOp::SetAtom {
        def: src.def.clone(),
        instance: src.instance.clone(),
        atom: scaled_clock(&src, period),
    })
}

/// Set the period of `src` through the editing core (an undoable action on `editor`'s
/// history; wires follow, a kernel rejection rolls it back).
pub fn apply_period(
    editor: &mut Editor,
    target: &mut dyn EditTarget,
    src: &ClockSource,
    period: u32,
) -> Result<(), String> {
    editor.set_atom(
        target,
        &src.def,
        &src.instance,
        &scaled_clock(src, period),
        false,
    )
}

/// One clock row of the panel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClockRow {
    /// Clock net (STA name).
    pub net: String,
    /// Configured period.
    pub period: u32,
    /// Minimum period as text (`"never settles"`, `"—"` when unknown).
    pub min_period: String,
    /// Suggested period.
    pub suggested: Option<u32>,
    /// Configured period ≥ minimum.
    pub meets: bool,
    /// Period the "Apply" button sets (suggestion differs from the configuration and a clock
    /// source was found).
    pub apply: Option<u32>,
    /// The clock source (for the edit), if found.
    pub source: Option<ClockSource>,
}

/// One violation / finding row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FindingRow {
    /// Severity.
    pub severity: Severity,
    /// Kind.
    pub kind: DiagKind,
    /// Location (jump target), if any.
    pub location: Option<String>,
    /// Message.
    pub message: String,
}

/// A path the canvas can highlight.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathChoice {
    /// Label, e.g. `"critical path"` or `"clk↑ (settle 23)"`.
    pub label: String,
    /// Flat net names along the path.
    pub nets: Vec<String>,
}

/// One settle-time row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SettleRow {
    /// Event description.
    pub event: String,
    /// Settle time (`"never"`).
    pub settle: String,
    /// Gap to the next event (`"—"`).
    pub gap: String,
    /// Fast rounds.
    pub rounds: u32,
    /// Settles within its gap (or has no gap).
    pub ok: bool,
}

/// Everything the STA panel shows (pure; built on the worker thread).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StaModel {
    /// No error and every clock meets timing.
    pub clean: bool,
    /// Headline.
    pub verdict: String,
    /// Clocks.
    pub clocks: Vec<ClockRow>,
    /// Violations and findings (errors first).
    pub findings: Vec<FindingRow>,
    /// Errors among the findings.
    pub errors: usize,
    /// Highlightable paths: `[0]` is the overall critical path, then one per event with a path.
    pub paths: Vec<PathChoice>,
    /// Settle times per event.
    pub settle: Vec<SettleRow>,
}

impl StaModel {
    /// Build the model of `s` for `design` (`flat` maps clock names to sources).
    pub fn build(s: &TimingSummary, design: &Design, flat: Option<&FlatNetlist>) -> StaModel {
        let clocks: Vec<ClockRow> = s
            .clocks
            .iter()
            .map(|c| {
                let source = find_clock(design, flat, &c.net);
                ClockRow {
                    net: c.net.clone(),
                    period: c.period,
                    min_period: match c.min_period {
                        Some(u32::MAX) => "never settles".into(),
                        Some(m) => m.to_string(),
                        None => "—".into(),
                    },
                    suggested: c.suggested_period,
                    meets: c.meets_timing,
                    apply: c
                        .suggested_period
                        .filter(|&p| p != c.period && source.is_some()),
                    source,
                }
            })
            .collect();
        let mut seen = HashSet::new();
        let mut findings: Vec<FindingRow> = s
            .report
            .diagnostics
            .iter()
            .chain(&s.warnings)
            .filter(|d| seen.insert((d.kind, d.location.clone(), d.message.clone())))
            .map(|d: &Diagnostic| FindingRow {
                severity: d.severity,
                kind: d.kind,
                location: d.location.clone(),
                message: super::glyph_safe(&d.message),
            })
            .collect();
        findings.sort_by_key(|f| severity_rank(f.severity));
        let errors = findings
            .iter()
            .filter(|f| f.severity == Severity::Error)
            .count();
        let mut paths = vec![PathChoice {
            label: "critical path".into(),
            nets: s.report.critical_path.clone(),
        }];
        for p in &s.report.phase_settle {
            if !p.critical_path.is_empty() {
                paths.push(PathChoice {
                    label: format!(
                        "{} (settle {})",
                        p.event,
                        p.settle.map_or("never".into(), |v| v.to_string())
                    ),
                    nets: p.critical_path.clone(),
                });
            }
        }
        let settle = s
            .report
            .phase_settle
            .iter()
            .map(|p| SettleRow {
                event: super::glyph_safe(&p.event),
                settle: p.settle.map_or("never".into(), |v| v.to_string()),
                gap: p.gap.map_or("—".into(), |v| v.to_string()),
                rounds: p.rounds,
                ok: match (p.settle, p.gap) {
                    (None, _) => false,
                    (Some(st), Some(g)) => st < g,
                    (Some(_), None) => true,
                },
            })
            .collect();
        let clean = s.is_clean();
        let slow = clocks.iter().filter(|c| !c.meets).count();
        let verdict = if clean {
            "timing clean: Fast mode matches Timing mode at phase ends".to_string()
        } else {
            let mut why = Vec::new();
            if errors > 0 {
                why.push(format!(
                    "{errors} violation{}",
                    if errors == 1 { "" } else { "s" }
                ));
            }
            if slow > 0 {
                why.push(format!(
                    "{slow} clock{} too fast",
                    if slow == 1 { "" } else { "s" }
                ));
            }
            format!("{}: Fast mode may differ from Timing mode", why.join(", "))
        };
        StaModel {
            clean,
            verdict,
            clocks,
            findings,
            errors,
            paths,
            settle,
        }
    }
}

/// Run STA for `design` on a fresh kernel (blocking; call from a worker thread).
pub fn analyze(design: Design) -> Result<TimingSummary, Vec<Diagnostic>> {
    analyze_with_model(design).map(|(s, _)| s)
}

/// [`analyze`] plus the panel model (blocking; call from a worker thread).
pub fn analyze_with_model(design: Design) -> Result<(TimingSummary, StaModel), Vec<Diagnostic>> {
    let mut k = Kernel::new(KernelOptions {
        background: false,
        ..KernelOptions::default()
    });
    k.load(design.clone())?;
    let s = k.timing_report();
    let m = StaModel::build(&s, &design, k.netlist().map(|f| &**f));
    Ok((s, m))
}

/// A finished analysis.
#[derive(Clone, Debug)]
pub struct StaResult {
    /// The report and its model (or the load errors).
    pub report: Result<(TimingSummary, StaModel), Vec<Diagnostic>>,
    /// Document revision analyzed.
    pub revision: u64,
    /// Wall time of the analysis.
    pub elapsed: Duration,
}

impl StaResult {
    /// The model, if the design loaded.
    pub fn model(&self) -> Option<&StaModel> {
        self.report.as_ref().ok().map(|(_, m)| m)
    }
}

struct Job {
    rx: Receiver<StaResult>,
    started: Instant,
}

/// The STA tab.
pub struct StaPanel {
    shared: SharedRef,
    job: Option<Job>,
    /// Latest result.
    pub result: Option<StaResult>,
    /// Highlight the selected path on the canvas.
    pub highlight: bool,
    /// Selected path (index into [`StaModel::paths`]).
    pub path: usize,
    /// Re-run automatically when the document changes (debounced).
    pub auto: bool,
    /// Editing core for the period edits (its own undo history).
    pub editor: Editor,
    hl_key: Option<(String, u64, u64, bool, usize)>,
    result_seq: u64,
    changed_at: Option<(u64, Instant)>,
}

impl StaPanel {
    /// A panel sharing `shared` with the rest of the module.
    pub fn new(shared: SharedRef) -> StaPanel {
        StaPanel {
            shared,
            job: None,
            result: None,
            highlight: true,
            path: 0,
            auto: false,
            editor: Editor::new(),
            hl_key: None,
            result_seq: 0,
            changed_at: None,
        }
    }

    /// Start an analysis of the current document on a worker thread.
    pub fn start(&mut self, app: &AppState) {
        let design = app.doc.design.clone();
        let revision = app.doc.revision;
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("panda-sta".into())
            .spawn(move || {
                let t0 = Instant::now();
                let report = analyze_with_model(design);
                let _ = tx.send(StaResult {
                    report,
                    revision,
                    elapsed: t0.elapsed(),
                });
            })
            .expect("spawn STA thread");
        self.job = Some(Job {
            rx,
            started: Instant::now(),
        });
    }

    /// True while an analysis runs.
    pub fn running(&self) -> bool {
        self.job.is_some()
    }

    /// Block until the running analysis finished (tests, headless tools).
    pub fn wait(&mut self) {
        if let Some(j) = self.job.take()
            && let Ok(r) = j.rx.recv()
        {
            self.install(r);
        }
    }

    fn install(&mut self, r: StaResult) {
        let n = r.model().map_or(1, |m| m.paths.len());
        if self.path >= n {
            self.path = 0;
        }
        self.result = Some(r);
        self.result_seq += 1;
    }

    fn poll(&mut self) {
        if let Some(j) = &self.job {
            match j.rx.try_recv() {
                Ok(r) => {
                    self.job = None;
                    self.install(r);
                }
                Err(TryRecvError::Disconnected) => self.job = None,
                Err(TryRecvError::Empty) => {}
            }
        }
    }

    /// Flat-net names of the selected path (empty without a result).
    pub fn selected_path(&self) -> &[String] {
        self.result
            .as_ref()
            .and_then(StaResult::model)
            .and_then(|m| m.paths.get(self.path))
            .map_or(&[], |p| &p.nets)
    }

    /// Keep the path highlight in sync with the shown view.
    fn sync_highlight(&mut self, app: &mut AppState) {
        let key = (
            app.view.path.clone(),
            app.live.watch_id,
            self.result_seq,
            self.highlight,
            self.path,
        );
        if self.hl_key.as_ref() == Some(&key) {
            return;
        }
        let names = self.selected_path().to_vec();
        let mut sh = self.shared.borrow_mut();
        sh.path_hl.clear(app);
        if !self.highlight || names.is_empty() {
            self.hl_key = Some(key);
            return;
        }
        let (Some(flat), Some(map)) = (app.live.flat.clone(), app.live.map.as_ref()) else {
            return; // retry next frame
        };
        if map.path != app.view.path {
            return;
        }
        for id in nets_in_view(&flat, map, &names) {
            sh.path_hl.add_net(app, &id, false);
        }
        self.hl_key = Some(key);
    }

    fn jump(&self, app: &mut AppState, loc: &str) {
        match resolve_location(&app.doc.design, app.live.flat.as_deref(), loc) {
            Some(t) => jump_to(app, &self.shared, &t, false),
            None => app.log.push(format!("cannot locate '{loc}'")),
        }
    }

    /// Apply the suggested period of clock row `net` (as analyzed): edits the clock source
    /// through the editing core and re-runs the analysis.
    pub fn apply(&mut self, app: &mut AppState, net: &str, period: u32) -> Result<(), String> {
        let analyzed = self
            .result
            .as_ref()
            .filter(|r| r.revision == app.doc.revision);
        let src = analyzed
            .and_then(StaResult::model)
            .and_then(|m| m.clocks.iter().find(|c| c.net == net))
            .and_then(|c| c.source.clone())
            .or_else(|| find_clock(&app.doc.design, app.live.flat.as_deref(), net))
            .ok_or_else(|| format!("no Clock source drives '{net}'"))?;
        apply_period(&mut self.editor, app, &src, period)?;
        app.log.push(format!(
            "clock {} ({}): period {} → {period}",
            net,
            child_path(&src.path, &src.instance),
            src.period
        ));
        self.start(app);
        Ok(())
    }

    fn header(&mut self, ui: &mut Ui, app: &mut AppState) {
        ui.horizontal(|ui| {
            let busy = self.running();
            if ui
                .add_enabled(!busy, egui::Button::new("Run STA"))
                .on_hover_text("Static timing analysis on a worker thread (own kernel)")
                .clicked()
            {
                self.start(app);
            }
            if let Some(j) = &self.job {
                ui.spinner();
                ui.label(format!(
                    "analyzing… {:.1} s",
                    j.started.elapsed().as_secs_f64()
                ));
            } else if let Some(r) = &self.result {
                ui.label(format!(
                    "analyzed in {:.0} ms",
                    r.elapsed.as_secs_f64() * 1e3
                ));
                if r.revision != app.doc.revision {
                    ui.colored_label(
                        severity_color(Severity::Warning),
                        "stale: the design changed",
                    );
                }
            } else {
                ui.weak("not analyzed yet");
            }
            ui.separator();
            ui.checkbox(&mut self.auto, "Re-run on edit")
                .on_hover_text("Analyze again 0.5 s after the design changes");
            ui.checkbox(&mut self.highlight, "Highlight path");
            if let Some(m) = self.result.as_ref().and_then(StaResult::model)
                && m.paths.len() > 1
            {
                let cur = m.paths.get(self.path).map_or("", |p| p.label.as_str());
                egui::ComboBox::from_id_salt("sta_path_choice")
                    .selected_text(cur)
                    .width(220.0)
                    .show_ui(ui, |ui| {
                        for (i, p) in m.paths.iter().enumerate() {
                            ui.selectable_value(&mut self.path, i, &p.label);
                        }
                    });
            }
            if self.editor.history.undo_len() > 0
                && ui
                    .button("Undo period")
                    .on_hover_text("Undo the last period change made here")
                    .clicked()
            {
                match self.editor.undo(app) {
                    Ok(_) => self.start(app),
                    Err(e) => app.log.push(format!("cannot undo: {e}")),
                }
            }
        });
    }
}

impl Panel for StaPanel {
    fn id(&self) -> &'static str {
        "sim_sta"
    }

    fn title(&self) -> String {
        "Timing (STA)".into()
    }

    fn slot(&self) -> Slot {
        Slot::Bottom
    }

    fn on_frame(&mut self, ctx: &egui::Context, app: &mut AppState) {
        self.poll();
        if self.job.is_some() {
            ctx.request_repaint_after(Duration::from_millis(50));
        }
        if self.auto
            && let Some(r) = &self.result
            && r.revision != app.doc.revision
            && self.job.is_none()
        {
            let rev = app.doc.revision;
            match self.changed_at {
                Some((r, t)) if r == rev => {
                    if t.elapsed() >= Duration::from_millis(500) {
                        self.changed_at = None;
                        self.start(app);
                    } else {
                        ctx.request_repaint_after(Duration::from_millis(100));
                    }
                }
                _ => {
                    self.changed_at = Some((rev, Instant::now()));
                    ctx.request_repaint_after(Duration::from_millis(100));
                }
            }
        }
        self.sync_highlight(app);
    }

    fn on_event(&mut self, ev: &SimEvent, app: &mut AppState) {
        if let SimEvent::Edited { id, result: Err(d) } = ev {
            let why = d.first().map_or(String::new(), ToString::to_string);
            if self.editor.kernel_rejected(app, *id, &why) {
                app.log.push(format!("period change rejected: {why}"));
            }
        }
    }

    fn ui(&mut self, ui: &mut Ui, app: &mut AppState) {
        crate::theme::apply(ui.ctx());
        self.header(ui, app);
        ui.separator();
        // Every branch fills the panel, so the bottom panel keeps its size across tabs.
        let fill = |ui: &mut Ui| {
            let s = ui.available_size();
            ui.allocate_space(s);
        };
        let Some(r) = &self.result else {
            ui.weak("Run STA to see the minimum clock periods, the critical path and violations.");
            fill(ui);
            return;
        };
        let m = match &r.report {
            Ok((_, m)) => m.clone(),
            Err(d) => {
                ui.colored_label(
                    severity_color(Severity::Error),
                    format!("design does not load: {} error(s)", d.len()),
                );
                for e in d {
                    ui.label(e.to_string());
                }
                fill(ui);
                return;
            }
        };
        let mut jump: Option<String> = None;
        let mut apply: Option<(String, u32)> = None;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let verdict = if m.clean {
                    RichText::new(format!("✔ {}", m.verdict)).color(GOOD)
                } else {
                    RichText::new(format!("✖ {}", m.verdict)).color(severity_color(Severity::Error))
                };
                ui.label(verdict.strong());
                ui.add_space(4.0);
                ui.columns(2, |cols| {
                    clocks_ui(&mut cols[0], &m, &mut jump, &mut apply);
                    cols[0].add_space(6.0);
                    findings_ui(&mut cols[0], &m, &mut jump);
                    path_ui(&mut cols[1], &m, self.path, &mut jump);
                    cols[1].add_space(6.0);
                    settle_ui(&mut cols[1], &m);
                });
            });
        if let Some(loc) = jump {
            self.jump(app, &loc);
        }
        if let Some((net, p)) = apply
            && let Err(e) = self.apply(app, &net, p)
        {
            app.log.push(format!("cannot set the period: {e}"));
        }
    }
}

fn clocks_ui(
    ui: &mut Ui,
    m: &StaModel,
    jump: &mut Option<String>,
    apply: &mut Option<(String, u32)>,
) {
    ui.strong("Clocks");
    if m.clocks.is_empty() {
        ui.weak("no running clock");
        return;
    }
    egui::Grid::new("sta_clocks")
        .striped(true)
        .num_columns(6)
        .show(ui, |ui| {
            for h in ["clock", "period", "min", "suggested", "status", ""] {
                ui.weak(h);
            }
            ui.end_row();
            for c in &m.clocks {
                if ui
                    .link(RichText::new(&c.net).monospace())
                    .on_hover_text(match &c.source {
                        Some(s) => format!("Clock atom {}", child_path(&s.path, &s.instance)),
                        None => "jump to the clock net".into(),
                    })
                    .clicked()
                {
                    *jump = Some(c.net.clone());
                }
                ui.label(c.period.to_string());
                ui.label(&c.min_period);
                ui.label(c.suggested.map_or("—".into(), |p| p.to_string()));
                if c.meets {
                    ui.colored_label(GOOD, "meets");
                } else {
                    ui.colored_label(severity_color(Severity::Error), "too fast");
                }
                match c.apply {
                    Some(p) => {
                        if ui
                            .button(format!("Apply {p}"))
                            .on_hover_text(
                                "Set the clock period to the STA suggestion (high time and phase \
                                 scale along; undoable)",
                            )
                            .clicked()
                        {
                            *apply = Some((c.net.clone(), p));
                        }
                    }
                    None => {
                        ui.label("");
                    }
                }
                ui.end_row();
            }
        });
}

fn findings_ui(ui: &mut Ui, m: &StaModel, jump: &mut Option<String>) {
    ui.strong(format!(
        "Violations and findings ({}, {} error{})",
        m.findings.len(),
        m.errors,
        if m.errors == 1 { "" } else { "s" }
    ));
    if m.findings.is_empty() {
        ui.weak("none");
    }
    for f in &m.findings {
        ui.horizontal_wrapped(|ui| {
            dot(ui, severity_color(f.severity));
            ui.colored_label(severity_color(f.severity), format!("{:?}", f.kind));
            if let Some(loc) = &f.location {
                let (net, kernel) = path_entry_net(loc);
                let hover = if kernel {
                    "Jump to this sequential kernel (its first output net)"
                } else {
                    "Jump to this location"
                };
                if ui
                    .link(RichText::new(net).monospace())
                    .on_hover_text(hover)
                    .clicked()
                {
                    *jump = Some(net.to_string());
                }
            }
            ui.label(&f.message);
        });
    }
}

fn path_ui(ui: &mut Ui, m: &StaModel, sel: usize, jump: &mut Option<String>) {
    let Some(p) = m.paths.get(sel) else {
        return;
    };
    ui.strong(format!("{} ({} nets)", capitalize(&p.label), p.nets.len()));
    if p.nets.is_empty() {
        ui.weak("no path (nothing changes after the events)");
        return;
    }
    egui::Grid::new("sta_path")
        .num_columns(2)
        .spacing([6.0, 2.0])
        .show(ui, |ui| {
            for (i, n) in p.nets.iter().enumerate() {
                ui.weak(format!("{:>2}", i + 1));
                let (net, kernel) = path_entry_net(n);
                ui.horizontal(|ui| {
                    if ui
                        .link(RichText::new(net).monospace())
                        .on_hover_text("Jump to this net")
                        .clicked()
                    {
                        *jump = Some(net.to_string());
                    }
                    if kernel {
                        ui.weak("(sequential kernel)");
                    }
                });
                ui.end_row();
            }
        });
}

fn settle_ui(ui: &mut Ui, m: &StaModel) {
    egui::CollapsingHeader::new(format!("Settle time per event ({})", m.settle.len()))
        .id_salt("sta_settle")
        .show(ui, |ui| {
            egui::Grid::new("sta_settle_grid")
                .striped(true)
                .num_columns(4)
                .show(ui, |ui| {
                    for h in ["event", "settle", "gap", "rounds"] {
                        ui.weak(h);
                    }
                    ui.end_row();
                    for r in &m.settle {
                        ui.label(&r.event);
                        if r.ok {
                            ui.label(&r.settle);
                        } else {
                            ui.colored_label(severity_color(Severity::Error), &r.settle);
                        }
                        ui.label(&r.gap);
                        ui.label(r.rounds.to_string());
                        ui.end_row();
                    }
                });
        });
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().chain(c).collect(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use panda_netlist::{DefBuilder, DesignBuilder};

    fn clocked() -> Design {
        let mut db = DesignBuilder::new();
        let mut b = DefBuilder::new("top");
        let clk = b.clock(8, 4, 2);
        let clk = b.name(clk, "clk");
        let q = b.not(clk);
        b.output("Q", q);
        let top = db.add(b.finish());
        db.build(&top)
    }

    /// The top clock drives pin `CLK` of package `core` instantiated as `u0`.
    fn packaged() -> Design {
        let mut db = DesignBuilder::new();
        let mut b = DefBuilder::new("core");
        let c = b.input("CLK", 1);
        let q = b.not(c);
        b.output("Q", q);
        let core = db.add(b.finish());
        let mut b = DefBuilder::new("top");
        let clk = b.clock(10, 5, 0);
        let clk = b.name(clk, "clk");
        let q = b.inst_named("u0", &core, &[clk]);
        b.output("Q", q[0]);
        let top = db.add(b.finish());
        db.build(&top)
    }

    #[test]
    fn apply_period_edits_the_clock_source() {
        let d = clocked();
        let flat = panda_netlist::flat::flatten(&d).unwrap();
        let op = clock_period_edit(&d, Some(&flat), "clk", 12).expect("found");
        let EditOp::SetAtom { def, atom, .. } = &op else {
            panic!("{op:?}")
        };
        assert_eq!(def, "top");
        assert_eq!(
            atom,
            &Atom::Clock {
                period: 12,
                high: 6,
                phase: 3
            }
        );
        // Without a flat netlist the net id must match.
        assert!(clock_period_edit(&d, None, "clk", 4).is_some());
        assert!(clock_period_edit(&d, Some(&flat), "Q", 4).is_none());
        // The edit applies.
        let mut k = Kernel::default();
        k.load(d).unwrap();
        k.edit(&[op]).unwrap();
    }

    #[test]
    fn clock_found_through_an_alias_at_depth() {
        let d = packaged();
        let srcs = clock_sources(&d);
        assert_eq!(srcs.len(), 1);
        assert_eq!((srcs[0].path.as_str(), srcs[0].def.as_str()), ("", "top"));
        assert_eq!(srcs[0].net, "clk");
        let flat = panda_netlist::flat::flatten(&d).unwrap();
        // The package-pin alias finds the source at the top.
        let s = find_clock(&d, Some(&flat), "u0/CLK").expect("found through alias");
        assert_eq!(s.instance, srcs[0].instance);
        assert!(find_clock(&d, None, "u0/CLK").is_none());
        assert_eq!(
            scaled_clock(&s, 3),
            Atom::Clock {
                period: 3,
                high: 2,
                phase: 0
            }
        );
        assert_eq!(
            scaled_clock(&s, 0),
            Atom::Clock {
                period: 2,
                high: 1,
                phase: 0
            }
        );
    }

    #[test]
    fn apply_goes_through_the_editing_core_and_undoes() {
        let d = packaged();
        let mut doc = crate::doc::Document::new(d.clone());
        let s = find_clock(&d, None, "clk").unwrap();
        let mut ed = Editor::new();
        apply_period(&mut ed, &mut doc, &s, 20).unwrap();
        let clock_of = |doc: &crate::doc::Document| {
            let top = doc.design.top_def().unwrap();
            top.instances
                .iter()
                .find(|i| i.id == s.instance)
                .unwrap()
                .kind
                .clone()
        };
        assert_eq!(
            clock_of(&doc),
            InstKind::Atom(Atom::Clock {
                period: 20,
                high: 10,
                phase: 0
            })
        );
        ed.undo(&mut doc).unwrap();
        assert_eq!(
            clock_of(&doc),
            InstKind::Atom(Atom::Clock {
                period: 10,
                high: 5,
                phase: 0
            })
        );
    }

    #[test]
    fn analysis_and_model_off_thread() {
        let (s, m) = std::thread::spawn(|| analyze_with_model(clocked()))
            .join()
            .unwrap()
            .expect("loads");
        assert_eq!(s.clocks.len(), 1);
        assert_eq!(s.clocks[0].period, 8);
        assert_eq!(m.clocks.len(), 1);
        assert_eq!(m.clocks[0].net, s.clocks[0].net);
        assert!(m.clocks[0].source.is_some());
        assert_eq!(m.paths[0].label, "critical path");
        assert_eq!(m.paths[0].nets, s.report.critical_path);
        assert_eq!(m.clean, s.is_clean());
        assert_eq!(m.settle.len(), s.report.phase_settle.len());
    }
}
