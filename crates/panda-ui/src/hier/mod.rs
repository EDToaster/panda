//! Hierarchy, packaging and the library (owner: the **hierarchy** work stream).
//!
//! * Navigation ([`nav`]): breadcrumbs, enter/exit package instances (double-click a package,
//!   `Alt+↑` to go up, the *Instances* tab or the *Hierarchy* menu). The canvas then shows the
//!   def of that instance, and live values come from that instance's flat nets
//!   ([`crate::app::AppState::enter`] → watch set of [`crate::doc::values::ViewMap`]).
//! * Packaging ([`package`]): create a package from the selected instances (`Ctrl+G`); crossing
//!   nets become pins with names and widths from the nets and directions from their drivers;
//!   the selection is replaced by one instance whose symbol pin layout is generated.
//! * Library ([`library`]): atoms, the document's defs and parameterized panda-circuits blocks;
//!   click an entry then the canvas, or drag it onto the canvas (the edit module's place tool,
//!   [`crate::edit::request_place`]).
//! * Classification badges ([`classify`]): computed on a worker thread with its own kernel, never
//!   on the sim or UI thread.
//! * Undo ([`change`]): every structural change here is a [`change::DocChange`] recorded through
//!   the edit module's undo API ([`crate::edit::History`]); the sim thread applies it between
//!   run chunks, nothing here waits for compilation, and a kernel rejection rolls it back.

pub mod actions;
pub mod change;
pub mod classify;
pub mod library;
pub mod manage;
pub mod nav;
pub mod package;
pub mod ui;

#[cfg(test)]
mod package_tests;

use std::cell::RefCell;
use std::rc::Rc;

use crate::app::Registry;

pub use actions::HierState;

/// Register this module's panels and tools.
pub fn register(reg: &mut Registry) {
    register_with(reg, HierState::new());
}

/// Register with explicit state. Returns the shared handle so callers (tests) can drive the
/// actions directly.
pub fn register_with(reg: &mut Registry, state: HierState) -> ui::Shared {
    let shared: ui::Shared = Rc::new(RefCell::new(state));
    reg.panel(ui::HierMenu(shared.clone()))
        .panel(ui::Breadcrumbs(shared.clone()))
        .panel(ui::LibraryTab(shared.clone()))
        .panel(ui::TreeTab(shared.clone()))
        .tool(ui::HierTool(shared.clone()));
    shared
}
