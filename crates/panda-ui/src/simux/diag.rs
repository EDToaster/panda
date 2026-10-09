//! Diagnostics panel (bottom tab): the kernel's deduplicated diagnostics grouped by kind, with
//! severity filters, occurrence counts and tick ranges. Clicking a location jumps there: the
//! canvas switches to the instance, highlights the net (contention colour for `Contention`) or
//! instance, and centres on it ([`super::locate`], [`super::jump_to`]).

use egui::{Color32, RichText, Ui};
use panda_kernel::DiagEntry;
use panda_netlist::{DiagKind, Severity};

use crate::app::{AppState, Panel, Slot};

use super::locate::resolve_location;
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
        Severity::Error => Color32::from_rgb(240, 80, 80),
        Severity::Warning => Color32::from_rgb(235, 180, 60),
        Severity::Info => Color32::from_rgb(110, 170, 255),
    }
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
}

/// Group `entries` by kind (filtered), worst groups first.
pub fn group(entries: &[DiagEntry], filter: SeverityFilter) -> Vec<DiagGroup> {
    let mut groups: Vec<DiagGroup> = Vec::new();
    for (i, e) in entries.iter().enumerate() {
        let d = &e.diagnostic;
        if !filter.shows(d.severity) {
            continue;
        }
        match groups.iter_mut().find(|g| g.kind == d.kind) {
            Some(g) => {
                g.entries.push(i);
                g.occurrences += e.count;
                if severity_rank(d.severity) < severity_rank(g.severity) {
                    g.severity = d.severity;
                }
            }
            None => groups.push(DiagGroup {
                kind: d.kind,
                severity: d.severity,
                entries: vec![i],
                occurrences: e.count,
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

/// The diagnostics tab.
pub struct DiagPanel {
    shared: SharedRef,
    /// Severity filter.
    pub filter: SeverityFilter,
    /// Location of the last jump (shown selected).
    pub selected: Option<String>,
}

impl DiagPanel {
    /// A panel sharing `shared` with the rest of the module.
    pub fn new(shared: SharedRef) -> DiagPanel {
        DiagPanel {
            shared,
            filter: SeverityFilter::default(),
            selected: None,
        }
    }

    /// Jump to `loc` (false if it does not resolve).
    pub fn jump(&mut self, app: &mut AppState, loc: &str, kind: DiagKind) -> bool {
        let Some(t) = resolve_location(&app.doc.design, app.live.flat.as_deref(), loc) else {
            app.log.push(format!("cannot locate '{loc}'"));
            return false;
        };
        jump_to(app, &self.shared, &t, kind == DiagKind::Contention);
        self.selected = Some(loc.to_string());
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
        let entries = app.diagnostics.clone();
        let count = |s: Severity| {
            entries
                .iter()
                .filter(|e| e.diagnostic.severity == s)
                .count()
        };
        ui.horizontal(|ui| {
            for (s, label, on) in [
                (Severity::Error, "errors", &mut self.filter.errors),
                (Severity::Warning, "warnings", &mut self.filter.warnings),
                (Severity::Info, "info", &mut self.filter.infos),
            ] {
                let text = RichText::new(format!("● {} {label}", count(s))).color(if *on {
                    severity_color(s)
                } else {
                    ui.visuals().weak_text_color()
                });
                ui.toggle_value(on, text);
            }
            ui.separator();
            if ui
                .button("Clear highlight")
                .on_hover_text("Remove the jump highlight from the canvas")
                .clicked()
            {
                self.shared.borrow_mut().jump_hl.clear(app);
                self.selected = None;
            }
        });
        ui.separator();
        if entries.is_empty() {
            ui.weak("No diagnostics.");
            return;
        }
        let groups = group(&entries, self.filter);
        let mut jump: Option<(String, DiagKind)> = None;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for g in &groups {
                    let head = RichText::new(format!(
                        "{:?}  ({} × {}, {} occurrence{})",
                        g.kind,
                        g.entries.len(),
                        match g.severity {
                            Severity::Error => "error",
                            Severity::Warning => "warning",
                            Severity::Info => "info",
                        },
                        g.occurrences,
                        if g.occurrences == 1 { "" } else { "s" }
                    ))
                    .color(severity_color(g.severity))
                    .strong();
                    egui::CollapsingHeader::new(head)
                        .id_salt(("diag_group", format!("{:?}", g.kind)))
                        .default_open(true)
                        .show(ui, |ui| {
                            for &i in &g.entries {
                                let e = &entries[i];
                                let d = &e.diagnostic;
                                ui.horizontal_wrapped(|ui| {
                                    ui.colored_label(severity_color(d.severity), "●");
                                    if let Some(loc) = &d.location {
                                        let sel = self.selected.as_deref() == Some(loc);
                                        if ui
                                            .selectable_label(sel, RichText::new(loc).monospace())
                                            .on_hover_text("Jump to this location")
                                            .clicked()
                                        {
                                            jump = Some((loc.clone(), d.kind));
                                        }
                                    }
                                    ui.label(&d.message);
                                    let mut meta = format!("{:?}", e.source);
                                    if e.count > 1 {
                                        meta.push_str(&format!(
                                            " · ×{} · ticks {}–{}",
                                            e.count, e.first_tick, e.last_tick
                                        ));
                                    } else {
                                        meta.push_str(&format!(" · tick {}", e.first_tick));
                                    }
                                    ui.weak(meta);
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

#[cfg(test)]
mod tests {
    use super::*;
    use panda_kernel::DiagSource;
    use panda_netlist::Diagnostic;

    fn entry(kind: DiagKind, sev: Severity, count: u64, first: u64) -> DiagEntry {
        DiagEntry {
            diagnostic: Diagnostic {
                kind,
                severity: sev,
                message: String::new(),
                location: None,
            },
            source: DiagSource::Timing,
            count,
            first_tick: first,
            last_tick: first,
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
            [DiagKind::Contention, DiagKind::SetupViolation, DiagKind::DelaySensitive]
        );
        assert_eq!(g[0].severity, Severity::Error);
        assert_eq!(g[0].entries, vec![4, 2, 0]);
        assert_eq!(g[0].occurrences, 6);
        let only_warn = SeverityFilter {
            errors: false,
            warnings: true,
            infos: false,
        };
        let g = group(&es, only_warn);
        assert_eq!(g.len(), 1);
        assert_eq!((g[0].severity, g[0].entries.len()), (Severity::Warning, 2));
    }
}
