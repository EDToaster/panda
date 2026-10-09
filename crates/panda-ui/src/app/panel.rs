//! Extension points of the app shell: [`Panel`]s placed in [`Slot`]s and [`CanvasTool`]s that
//! handle canvas input, both collected in a [`Registry`].
//!
//! A feature module adds itself with one line in [`register_all`] (calling its own
//! `register(reg)`), and touches nothing else of the shell.

use egui::Ui;

use crate::canvas::{CanvasResponse, Scene};
use crate::simhost::SimEvent;

use super::AppState;

/// Where a panel is shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Slot {
    /// A menu in the top bar's menu row (the panel's `ui` runs inside `ui.menu_button(title)`).
    Menu,
    /// Inline in the top bar, after the menus (e.g. run controls, breadcrumbs).
    TopBar,
    /// Left side panel (palette, hierarchy). Several panels become tabs.
    Left,
    /// Right side panel (inspector). Several panels become tabs.
    Right,
    /// Bottom panel (waveforms, diagnostics, STA). Several panels become tabs.
    Bottom,
    /// Status bar (bottom edge), inline.
    Status,
}

/// A piece of UI living in one [`Slot`].
pub trait Panel {
    /// Stable id (used for tab state).
    fn id(&self) -> &'static str;
    /// Tab / menu title.
    fn title(&self) -> String {
        self.id().to_string()
    }
    /// Where it goes.
    fn slot(&self) -> Slot;
    /// Draw the panel.
    fn ui(&mut self, ui: &mut Ui, app: &mut AppState);
    /// Called once per frame before any UI is laid out (keyboard shortcuts, polling).
    fn on_frame(&mut self, _ctx: &egui::Context, _app: &mut AppState) {}
    /// Called for every event from the sim thread (after [`AppState`] processed it).
    fn on_event(&mut self, _ev: &SimEvent, _app: &mut AppState) {}
}

/// Handles input on the canvas and paints overlays (selection boxes, rubber-band wires, probes).
pub trait CanvasTool {
    /// Stable id.
    fn id(&self) -> &'static str;
    /// True while the tool wants primary-button drags (the canvas then does not pan with them).
    fn claims_primary(&self, _app: &AppState) -> bool {
        false
    }
    /// Called every frame after the canvas was drawn: `canvas` has the response, rect and
    /// transform; paint overlays with `ui.painter_at(canvas.rect)`.
    fn canvas(&mut self, ui: &mut Ui, canvas: &CanvasResponse, scene: &Scene, app: &mut AppState);
}

/// All registered panels and tools.
#[derive(Default)]
pub struct Registry {
    /// Panels in registration order.
    pub panels: Vec<Box<dyn Panel>>,
    /// Canvas tools in registration order (all get every frame).
    pub tools: Vec<Box<dyn CanvasTool>>,
}

impl Registry {
    /// Add a panel.
    pub fn panel(&mut self, p: impl Panel + 'static) -> &mut Self {
        self.panels.push(Box::new(p));
        self
    }

    /// Add a canvas tool.
    pub fn tool(&mut self, t: impl CanvasTool + 'static) -> &mut Self {
        self.tools.push(Box::new(t));
        self
    }

    /// Panels of a slot (indices into `panels`).
    pub fn in_slot(&self, slot: Slot) -> Vec<usize> {
        (0..self.panels.len())
            .filter(|&i| self.panels[i].slot() == slot)
            .collect()
    }
}

/// Register every feature module. **One line per module**; each module owns its `register`.
pub fn register_all(reg: &mut Registry) {
    crate::edit::register(reg);
    crate::hier::register(reg);
    crate::simui::register(reg);
    crate::simux::register(reg);
}
