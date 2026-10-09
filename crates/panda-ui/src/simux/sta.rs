//! STA panel (bottom tab): minimum period per clock with an "apply suggested period" button,
//! the critical path (highlighted on the canvas, in whatever view is shown), per-event settle
//! times and the timing violations (click a location to jump there).
//!
//! The analysis runs on a worker thread with its **own** [`Kernel`] (loaded from a copy of the
//! document), so neither the UI nor the sim thread waits for it.

use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

use egui::{RichText, Ui};
use panda_kernel::{EditOp, Kernel, KernelOptions, TimingSummary};
use panda_netlist::flat::FlatNetlist;
use panda_netlist::{Atom, Design, Diagnostic, Endpoint, InstKind, Severity};

use crate::app::{AppState, Panel, Slot};

use super::diag::severity_color;
use super::locate::{nets_in_view, resolve_location};
use super::{SharedRef, jump_to};

/// The edit that sets the period of the `Clock` source driving top-level net `clock` to
/// `period`, scaling its high time and phase proportionally (as STA's suggestion assumes).
pub fn clock_period_edit(
    design: &Design,
    flat: Option<&FlatNetlist>,
    clock: &str,
    period: u32,
) -> Option<EditOp> {
    let top = design.top_def()?;
    let want = flat.and_then(|f| f.net_by_name(clock));
    let period = period.max(2);
    for inst in &top.instances {
        let InstKind::Atom(Atom::Clock {
            period: old,
            high,
            phase,
        }) = &inst.kind
        else {
            continue;
        };
        let Some(net) = top.nets.iter().find(|n| {
            n.ends.iter().any(|e| {
                matches!(e, Endpoint::Port { inst: i, port } if i == &inst.id && port == "out")
            })
        }) else {
            continue;
        };
        let matches = match (flat, want) {
            (Some(f), Some(w)) => f.net_by_name(&net.id) == Some(w),
            _ => net.id == clock,
        };
        if !matches {
            continue;
        }
        let scale = |v: u32| (u64::from(v) * u64::from(period)).div_ceil(u64::from(*old).max(1));
        let high = (scale(*high) as u32).clamp(1, period - 1);
        let phase = (u64::from(*phase) * u64::from(period) / u64::from(*old).max(1)) as u32;
        return Some(EditOp::SetAtom {
            def: top.id.clone(),
            instance: inst.id.clone(),
            atom: Atom::Clock {
                period,
                high,
                phase: phase % period,
            },
        });
    }
    None
}

/// Run STA for `design` on a fresh kernel (blocking; call from a worker thread).
pub fn analyze(design: Design) -> Result<TimingSummary, Vec<Diagnostic>> {
    let mut k = Kernel::new(KernelOptions {
        background: false,
        ..KernelOptions::default()
    });
    k.load(design)?;
    Ok(k.timing_report())
}

/// A finished analysis.
#[derive(Clone, Debug)]
pub struct StaResult {
    /// The report (or the load errors).
    pub report: Result<TimingSummary, Vec<Diagnostic>>,
    /// Document revision analyzed.
    pub revision: u64,
    /// Wall time of the analysis.
    pub elapsed: Duration,
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
    /// Highlight the critical path on the canvas.
    pub highlight: bool,
    hl_key: Option<(String, u64, u64, bool)>,
    result_seq: u64,
}

impl StaPanel {
    /// A panel sharing `shared` with the rest of the module.
    pub fn new(shared: SharedRef) -> StaPanel {
        StaPanel {
            shared,
            job: None,
            result: None,
            highlight: true,
            hl_key: None,
            result_seq: 0,
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
                let report = analyze(design);
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

    fn poll(&mut self) {
        if let Some(j) = &self.job {
            match j.rx.try_recv() {
                Ok(r) => {
                    self.result = Some(r);
                    self.result_seq += 1;
                    self.job = None;
                }
                Err(TryRecvError::Disconnected) => self.job = None,
                Err(TryRecvError::Empty) => {}
            }
        }
    }

    /// Keep the critical-path highlight in sync with the shown view.
    fn sync_highlight(&mut self, app: &mut AppState) {
        let key = (
            app.view.path.clone(),
            app.live.watch_id,
            self.result_seq,
            self.highlight,
        );
        if self.hl_key.as_ref() == Some(&key) {
            return;
        }
        let mut sh = self.shared.borrow_mut();
        sh.path_hl.clear(app);
        let path = match &self.result {
            Some(StaResult {
                report: Ok(s), ..
            }) if self.highlight => &s.report.critical_path,
            _ => {
                self.hl_key = Some(key);
                return;
            }
        };
        let (Some(flat), Some(map)) = (app.live.flat.clone(), app.live.map.clone()) else {
            return; // retry next frame
        };
        if map.path != app.view.path {
            return;
        }
        for id in nets_in_view(&flat, &map, path) {
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
        self.sync_highlight(app);
    }

    fn ui(&mut self, ui: &mut Ui, app: &mut AppState) {
        let mut jump: Option<String> = None;
        let mut apply: Option<(String, u32)> = None;
        ui.horizontal(|ui| {
            let busy = self.running();
            if ui
                .add_enabled(!busy, egui::Button::new("Run STA"))
                .on_hover_text("Static timing analysis on a worker thread")
                .clicked()
            {
                self.start(app);
            }
            if let Some(j) = &self.job {
                ui.spinner();
                ui.label(format!("analyzing… {:.1} s", j.started.elapsed().as_secs_f64()));
            } else if let Some(r) = &self.result {
                let stale = r.revision != app.doc.revision;
                ui.label(format!("analyzed in {:.0} ms", r.elapsed.as_secs_f64() * 1e3));
                if stale {
                    ui.colored_label(
                        severity_color(Severity::Warning),
                        "stale: the design changed",
                    );
                }
            } else {
                ui.weak("not analyzed yet");
            }
            ui.separator();
            if ui
                .checkbox(&mut self.highlight, "Highlight critical path")
                .changed()
            {
                self.hl_key = None;
            }
        });
        ui.separator();
        let Some(r) = &self.result else {
            return;
        };
        let s = match &r.report {
            Ok(s) => s,
            Err(d) => {
                ui.colored_label(
                    severity_color(Severity::Error),
                    format!("design does not load: {} error(s)", d.len()),
                );
                for e in d {
                    ui.label(e.to_string());
                }
                return;
            }
        };
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let verdict = if s.is_clean() {
                    RichText::new("✔ timing clean: Fast mode matches Timing mode at phase ends")
                        .color(egui::Color32::from_rgb(90, 200, 120))
                } else {
                    RichText::new("✖ timing violations: Fast mode may differ from Timing mode")
                        .color(severity_color(Severity::Error))
                };
                ui.label(verdict.strong());
                ui.add_space(4.0);
                // Clocks.
                ui.strong("Clocks");
                if s.clocks.is_empty() {
                    ui.weak("no running clock");
                }
                egui::Grid::new("sta_clocks")
                    .striped(true)
                    .num_columns(6)
                    .show(ui, |ui| {
                        for h in ["clock", "period", "min period", "suggested", "status", ""] {
                            ui.weak(h);
                        }
                        ui.end_row();
                        for c in &s.clocks {
                            if ui.link(RichText::new(&c.net).monospace()).clicked() {
                                jump = Some(c.net.clone());
                            }
                            ui.label(c.period.to_string());
                            ui.label(match c.min_period {
                                Some(u32::MAX) => "never settles".to_string(),
                                Some(m) => m.to_string(),
                                None => "—".to_string(),
                            });
                            ui.label(c.suggested_period.map_or("—".into(), |p| p.to_string()));
                            if c.meets_timing {
                                ui.colored_label(egui::Color32::from_rgb(90, 200, 120), "meets");
                            } else {
                                ui.colored_label(severity_color(Severity::Error), "too fast");
                            }
                            match c.suggested_period {
                                Some(p) if p != c.period => {
                                    if ui
                                        .button(format!("Apply {p}"))
                                        .on_hover_text(
                                            "Set the clock period to the STA suggestion \
                                             (high time and phase scale along)",
                                        )
                                        .clicked()
                                    {
                                        apply = Some((c.net.clone(), p));
                                    }
                                }
                                _ => {
                                    ui.label("");
                                }
                            }
                            ui.end_row();
                        }
                    });
                ui.add_space(6.0);
                // Critical path.
                ui.strong(format!("Critical path ({} nets)", s.report.critical_path.len()));
                ui.horizontal_wrapped(|ui| {
                    for (i, n) in s.report.critical_path.iter().enumerate() {
                        if i > 0 {
                            ui.weak("→");
                        }
                        if ui.link(RichText::new(n).monospace()).clicked() {
                            jump = Some(n.clone());
                        }
                    }
                });
                ui.add_space(6.0);
                // Violations.
                let diags: Vec<&Diagnostic> =
                    s.report.diagnostics.iter().chain(&s.warnings).collect();
                ui.strong(format!("Violations and findings ({})", diags.len()));
                if diags.is_empty() {
                    ui.weak("none");
                }
                for d in diags {
                    ui.horizontal_wrapped(|ui| {
                        ui.colored_label(severity_color(d.severity), format!("● {:?}", d.kind));
                        if let Some(loc) = &d.location
                            && ui.link(RichText::new(loc).monospace()).clicked()
                        {
                            jump = Some(loc.clone());
                        }
                        ui.label(&d.message);
                    });
                }
                ui.add_space(6.0);
                // Settle times.
                egui::CollapsingHeader::new(format!(
                    "Settle time per event ({})",
                    s.report.phase_settle.len()
                ))
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
                            for p in &s.report.phase_settle {
                                ui.label(&p.event);
                                ui.label(p.settle.map_or("never".into(), |v| v.to_string()));
                                ui.label(p.gap.map_or("—".into(), |v| v.to_string()));
                                ui.label(p.rounds.to_string());
                                ui.end_row();
                            }
                        });
                });
            });
        if let Some(loc) = jump {
            self.jump(app, &loc);
        }
        if let Some((net, p)) = apply {
            let flat = app.live.flat.clone();
            match clock_period_edit(&app.doc.design, flat.as_deref(), &net, p) {
                Some(op) => match app.submit_edit(vec![op]) {
                    Ok(_) => self.start(app),
                    Err(d) => app.log.push(format!("cannot set the period: {d}")),
                },
                None => app.log.push(format!("no Clock source drives '{net}' at the top")),
            }
        }
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
    fn analysis_runs_off_thread_on_its_own_kernel() {
        let r = std::thread::spawn(|| analyze(clocked())).join().unwrap();
        let s = r.expect("loads");
        assert_eq!(s.clocks.len(), 1);
        assert_eq!(s.clocks[0].period, 8);
    }
}
