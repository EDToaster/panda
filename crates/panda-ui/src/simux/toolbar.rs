//! The run-controls toolbar (top bar): Timing/Fast, run/pause, step tick/phase/cycle (× n),
//! speed limit, reset. Shortcuts: F5 run/pause, F10 step (last unit), when no text field has
//! focus. Every gesture is a non-blocking command to the sim thread ([`super::control`]).

use egui::{Key, RichText, Ui};
use panda_kernel::{Mode, StepUnit};

use crate::app::{AppState, Panel, Slot};

use super::control::{RATE_PRESETS, RATE_RANGE, SimControl, ToolbarAction, si};

/// The toolbar panel.
#[derive(Debug, Default)]
pub struct Toolbar {
    /// Control state.
    pub ctl: SimControl,
    synced: bool,
}

impl Toolbar {
    /// Apply an action: send its commands.
    pub fn act(&mut self, app: &mut AppState, action: ToolbarAction) {
        let s = &app.live.snapshot;
        let (mode, running) = (s.mode, s.running);
        for cmd in self.ctl.commands(action, mode, running) {
            app.sim.send(cmd);
        }
    }
}

impl Panel for Toolbar {
    fn id(&self) -> &'static str {
        "sim_toolbar"
    }

    fn title(&self) -> String {
        "Simulation".into()
    }

    fn slot(&self) -> Slot {
        Slot::TopBar
    }

    fn on_frame(&mut self, ctx: &egui::Context, app: &mut AppState) {
        if !self.synced && app.live.snapshot.seq > 0 {
            self.ctl.sync_speed(app.live.snapshot.speed);
            self.synced = true;
        }
        if ctx.memory(|m| m.focused().is_some()) {
            return;
        }
        let (run, step) = ctx.input(|i| (i.key_pressed(Key::F5), i.key_pressed(Key::F10)));
        if run {
            self.act(app, ToolbarAction::ToggleRun);
        }
        if step {
            self.act(app, ToolbarAction::Step);
        }
    }

    fn ui(&mut self, ui: &mut Ui, app: &mut AppState) {
        let s = &app.live.snapshot;
        let (mode, running) = (s.mode, s.running);
        let mut actions = Vec::new();
        ui.spacing_mut().item_spacing.x = 4.0;
        ui.separator();
        // Mode.
        for (m, label) in [(Mode::Timing, "Timing"), (Mode::Fast, "Fast")] {
            if ui
                .selectable_label(mode == m, label)
                .on_hover_text(match m {
                    Mode::Timing => "Unit-delay reference semantics (per tick)",
                    Mode::Fast => "Compiled kernel-delta semantics (per phase, JIT)",
                })
                .clicked()
            {
                actions.push(ToolbarAction::SetMode(m));
            }
        }
        ui.separator();
        // Run / pause.
        let (icon, tip) = if running {
            ("⏸ Pause", "Pause (F5)")
        } else {
            ("▶ Run", "Run continuously (F5)")
        };
        if ui
            .button(RichText::new(icon).strong())
            .on_hover_text(tip)
            .clicked()
        {
            actions.push(ToolbarAction::ToggleRun);
        }
        // Step.
        for (unit, label) in [
            (StepUnit::Tick, "tick"),
            (StepUnit::Phase, "phase"),
            (StepUnit::Cycle, "cycle"),
        ] {
            let b = egui::Button::new(label).selected(self.ctl.step_unit == unit);
            if ui
                .add(b)
                .on_hover_text(format!(
                    "Step: advance {} {label}(s); pauses first if running (F10 repeats)",
                    self.ctl.step_n
                ))
                .clicked()
            {
                actions.push(ToolbarAction::StepBy(unit));
            }
        }
        ui.add(
            egui::DragValue::new(&mut self.ctl.step_n)
                .range(1..=1_000_000_000u64)
                .speed(0.2)
                .prefix("×"),
        )
        .on_hover_text("Units per step");
        ui.separator();
        // Speed.
        let mut limited = self.ctl.limited;
        if ui
            .checkbox(&mut limited, "Limit")
            .on_hover_text("Pace the simulation (the sim thread paces itself)")
            .changed()
        {
            actions.push(ToolbarAction::SetLimited(limited));
        }
        let mut rate = self.ctl.rate;
        let speed = (rate * 0.02).max(0.05);
        let r = ui.add_enabled(
            self.ctl.limited,
            egui::DragValue::new(&mut rate)
                .range(RATE_RANGE.0..=RATE_RANGE.1)
                .speed(speed)
                .max_decimals(1)
                .suffix(" cyc/s"),
        );
        if r.changed() {
            actions.push(ToolbarAction::SetRate(rate));
        }
        ui.menu_button("⏷", |ui| {
            for p in RATE_PRESETS {
                if ui.button(format!("{} cyc/s", si(p))).clicked() {
                    actions.push(ToolbarAction::LimitTo(p));
                    ui.close();
                }
            }
        })
        .response
        .on_hover_text("Speed presets");
        ui.separator();
        if ui
            .button("⟲ Reset")
            .on_hover_text("Power-on reset (inputs keep their values)")
            .clicked()
        {
            actions.push(ToolbarAction::Reset);
        }
        for a in actions {
            self.act(app, a);
        }
    }
}
