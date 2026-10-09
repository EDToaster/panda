//! Simulation UX (owner: the **sim UX** work stream, docs/M6-UI.md scope 5).
//!
//! | piece | where | slot |
//! |---|---|---|
//! | run controls: Timing/Fast, run/pause, step tick/phase/cycle, speed, reset | [`toolbar`] | top bar |
//! | click / edit inputs, pending-input badges, hover values, "Set input value…" context entry, jump focus | [`canvas_tool`] | canvas tool |
//! | waveform viewer (radix, zoom/scroll/follow, cursor) | [`waveform`] | bottom tab |
//! | diagnostics (grouped, severity filter, jump-to-location) | [`diag`] | bottom tab |
//! | STA (min period per clock, apply suggested period, critical path, violations) | [`sta`] | bottom tab |
//! | perf HUD (cycles/s, engine tier, publish rate, frame time) | [`hud`] | status bar |
//!
//! Pure logic lives in [`control`] (toolbar → [`crate::simhost::Command`]s, input values),
//! [`wave_view`] (waveform axis math, radix, decimation) and [`locate`] (location → view and
//! mark), all unit-tested without a GPU.
//!
//! **Performance contract** (the prime requirement of M6): nothing here waits on the sim thread.
//! Gestures become non-blocking [`crate::simhost::SimHost::send`]s; the sim thread paces a
//! speed limit itself; waveforms are only captured for the signals in the viewer (the sim host's
//! shadow capture); STA runs on its own worker thread with its own kernel; the waveform plot
//! decimates per pixel so its cost does not grow with the recorded history.

pub mod canvas_tool;
pub mod control;
pub mod diag;
pub mod hud;
pub mod locate;
pub mod sta;
pub mod toolbar;
pub mod wave_view;
pub mod waveform;

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use crate::app::{AppState, Registry};
use crate::doc::ItemRef;

use locate::{JumpTarget, Mark};

/// Highlights this module put into [`AppState::highlights`] (so they can be removed again
/// without touching other modules' entries, e.g. the selection).
#[derive(Clone, Debug, Default)]
pub struct OwnedHighlights {
    /// Accent nets.
    pub nets: HashSet<String>,
    /// Contention-coloured nets.
    pub contention: HashSet<String>,
    /// Outlined items.
    pub items: HashSet<ItemRef>,
}

impl OwnedHighlights {
    /// Remove these entries from the app's highlights and forget them.
    pub fn clear(&mut self, app: &mut AppState) {
        for n in self.nets.drain() {
            app.highlights.nets.remove(&n);
        }
        for n in self.contention.drain() {
            app.highlights.contention.remove(&n);
        }
        for i in self.items.drain() {
            app.highlights.items.remove(&i);
        }
    }

    /// Mark a net (in the contention colour if `contention`).
    pub fn add_net(&mut self, app: &mut AppState, id: &str, contention: bool) {
        if contention {
            app.highlights.contention.insert(id.to_string());
            self.contention.insert(id.to_string());
        }
        app.highlights.nets.insert(id.to_string());
        self.nets.insert(id.to_string());
    }

    /// Outline an item.
    pub fn add_item(&mut self, app: &mut AppState, item: ItemRef) {
        app.highlights.items.insert(item.clone());
        self.items.insert(item);
    }
}

/// Where the canvas should centre on the next frame (set by jumps, consumed by the canvas tool).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Focus {
    /// Def the focus refers to (applied once the canvas shows it).
    pub def: String,
    /// What to centre on.
    pub mark: Mark,
}

/// State shared between this module's panels and its canvas tool.
#[derive(Debug, Default)]
pub struct Shared {
    /// Pending canvas focus.
    pub focus: Option<Focus>,
    /// Highlights of the last jump (diagnostics / STA violation).
    pub jump_hl: OwnedHighlights,
    /// Highlights of the STA critical path.
    pub path_hl: OwnedHighlights,
}

/// Handle to [`Shared`] (panels and tools live on the UI thread).
pub type SharedRef = Rc<RefCell<Shared>>;

/// Switch the view to `t`, highlight its mark (contention colour for contention diagnostics)
/// and ask the canvas to centre on it.
pub fn jump_to(app: &mut AppState, shared: &SharedRef, t: &JumpTarget, contention: bool) {
    let mut sh = shared.borrow_mut();
    sh.jump_hl.clear(app);
    if app.view.path != t.path {
        // `enter` resets the highlights; the STA panel re-derives the critical path's.
        app.enter(&t.path);
        sh.path_hl = OwnedHighlights::default();
    }
    match &t.mark {
        Mark::Net(id) => sh.jump_hl.add_net(app, id, contention),
        Mark::Item(i) => sh.jump_hl.add_item(app, i.clone()),
        Mark::None => {}
    }
    sh.focus = Some(Focus {
        def: t.def.clone(),
        mark: t.mark.clone(),
    });
}

/// Register this module's panels and tools.
pub fn register(reg: &mut Registry) {
    let shared: SharedRef = Rc::new(RefCell::new(Shared::default()));
    reg.panel(toolbar::Toolbar::default());
    reg.panel(waveform::WaveformPanel::new());
    reg.panel(diag::DiagPanel::new(shared.clone()));
    reg.panel(sta::StaPanel::new(shared.clone()));
    reg.panel(hud::Hud::default());
    reg.tool(diag::JumpOverlay::new(shared.clone()));
    reg.tool(canvas_tool::SimTool::new(shared));
}

/// Kernel/STA text for display: egui's built-in fonts lack the arrow glyphs the kernel uses for
/// clock edges (`clk↑`), which would render as boxes. Rewrites them as words.
pub fn glyph_safe(text: &str) -> String {
    if !text.contains(['↑', '↓', '→', '←']) {
        return text.to_string();
    }
    text.replace('↑', " rise")
        .replace('↓', " fall")
        .replace('→', "->")
        .replace('←', "<-")
}

#[cfg(test)]
mod glyph_tests {
    #[test]
    fn arrows_become_words() {
        assert_eq!(super::glyph_safe("clk↑: settle 36"), "clk rise: settle 36");
        assert_eq!(super::glyph_safe("a → b"), "a -> b");
        assert_eq!(super::glyph_safe("plain"), "plain");
    }
}
