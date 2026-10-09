//! Hierarchy, packaging and the library (owner: the **hierarchy/files** work stream; the file
//! menu of [`crate::files`] is registered from here too).
//!
//! * Navigation ([`nav`]): breadcrumbs, enter/exit package instances (double-click a package,
//!   `Alt+↑` to go up, or the *Instances* tab). The canvas then shows the def of that instance,
//!   and live values come from that instance's flat nets ([`crate::app::AppState::enter`] →
//!   watch set of [`crate::doc::values::ViewMap`]).
//! * Packaging ([`package`]): create a package from the selected instances (`Ctrl+G`); crossing
//!   nets become pins with names and widths from the nets and directions from their drivers.
//! * Library ([`library`]): atoms, the document's defs and parameterized panda-circuits blocks;
//!   click to arm placement then click the canvas, or drag onto the canvas.
//! * Classification badges ([`classify`]): computed on a worker thread with its own kernel, never
//!   on the sim thread.
//! * Undo ([`change`]): every structural change here is a [`change::DocChange`] (edit ops both
//!   ways + geometry) on this module's [`change::History`]; see there for adopting them into an
//!   editor-wide undo stack.
//!
//! Placement uses this module's own canvas tool ([`ui::HierTool`]); structural edits go through
//! [`crate::app::AppState::submit_edit`] like every other edit, so the sim thread applies them
//! between run chunks and nothing here waits for compilation.

pub mod actions;
pub mod change;
pub mod classify;
pub mod library;
pub mod nav;
pub mod package;
pub mod ui;

#[cfg(test)]
mod package_tests;

use std::cell::RefCell;
use std::rc::Rc;

use crate::app::Registry;

pub use actions::HierState;

/// Register this module's panels and tools (recent files in the user's config directory).
pub fn register(reg: &mut Registry) {
    register_with(reg, HierState::with_default_store());
}

/// Register with explicit state (tests use an in-memory recent-files list). Returns the shared
/// handle so callers can drive the actions directly.
pub fn register_with(reg: &mut Registry, state: HierState) -> ui::Shared {
    let shared: ui::Shared = Rc::new(RefCell::new(state));
    reg.panel(ui::FileMenu(shared.clone()))
        .panel(ui::HierMenu(shared.clone()))
        .panel(ui::Breadcrumbs(shared.clone()))
        .panel(ui::LibraryTab(shared.clone()))
        .panel(ui::TreeTab(shared.clone()))
        .tool(ui::HierTool(shared.clone()));
    shared
}
