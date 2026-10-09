//! Editing (owner: the **editing** work stream).
//!
//! Select (click, box, Shift) / move (grid-snapped, wires rubber-band) / delete / rotate / flip
//! / copy-paste, placing atoms, pins and packages, Manhattan wire drawing with auto-junctions
//! and net merging/splitting, the inspector, and undo/redo of every action.
//!
//! * [`actions`]: the pure controller ([`Editor`]): every action computes `EditOp`s plus the
//!   geometry change and records an undoable transaction. No egui; unit-tested without a GPU.
//! * [`history`]: undo/redo ([`History`]) and the [`EditTarget`] abstraction (document +
//!   kernel). Structural edits go through [`crate::app::AppState::submit_edit`]: the document
//!   changes at once and the kernel gets the ops on the sim thread (the UI never waits). A
//!   kernel rejection (validation error) rolls the action back.
//! * [`route`]: orthogonal routing, rubber-banding, junctions, net splitting geometry.
//! * [`inspector`]: the right-hand inspector panel (atom parameters, net termination).
//! * `tool`: the canvas tool (gestures, overlays); `menus`: the "Edit" and "Place" menus.
//!
//! # Palette API (for the library palette of the hierarchy module)
//!
//! Call [`request_place`] (or use [`place_button`], which also supports dragging onto the
//! canvas) with a [`PlaceRequest`]; the canvas then shows a ghost under the pointer and places
//! the item on the next click / drop. [`current_selection`] gives the current selection (e.g.
//! for "create package from selection").

pub mod actions;
pub mod history;
pub mod inspector;
mod menus;
pub mod route;
mod tool;

use std::cell::RefCell;
use std::rc::Rc;

use panda_netlist::{Atom, Dir, GateOp, Value};

use crate::app::Registry;

pub use actions::{Anchor, Clipboard, Editor, PlaceRequest, Selection, Tool, WireSeg};
pub use history::{EditTarget, History, Txn};

/// The editor shared by the canvas tool, the menus and the inspector (UI thread only).
pub type SharedEditor = Rc<RefCell<Editor>>;

/// Register this module's panels and tools.
pub fn register(reg: &mut Registry) {
    let ed: SharedEditor = Rc::new(RefCell::new(Editor::new()));
    reg.panel(menus::EditMenu::new(ed.clone()));
    reg.panel(menus::PlaceMenu::new(ed.clone()));
    reg.panel(inspector::Inspector::new(ed.clone()));
    reg.tool(tool::EditTool::new(ed));
}

fn place_id() -> egui::Id {
    egui::Id::new("panda.edit.place_request")
}

fn selection_id() -> egui::Id {
    egui::Id::new("panda.edit.selection")
}

/// Ask the canvas to place `req`: it switches to the place tool (ghost under the pointer) and
/// places the item where the primary button is next released over the canvas (a click, or the
/// end of a drag that started in the palette).
pub fn request_place(ctx: &egui::Context, req: PlaceRequest) {
    ctx.data_mut(|d| d.insert_temp(place_id(), Some(req)));
}

/// Take a pending [`request_place`] (used by the canvas tool).
pub fn take_place_request(ctx: &egui::Context) -> Option<PlaceRequest> {
    ctx.data_mut(|d| d.remove_temp::<Option<PlaceRequest>>(place_id()))
        .flatten()
}

/// A palette button: click it, or drag it onto the canvas, to place `req`.
pub fn place_button(ui: &mut egui::Ui, label: &str, req: PlaceRequest) -> egui::Response {
    let r = ui.add(egui::Button::new(label).sense(egui::Sense::click_and_drag()));
    if r.clicked() || r.drag_started() {
        request_place(ui.ctx(), req);
    }
    r
}

/// The current selection of the viewed def (as of the last frame).
pub fn current_selection(ctx: &egui::Context) -> Selection {
    ctx.data(|d| d.get_temp::<Selection>(selection_id()))
        .unwrap_or_default()
}

fn publish_selection(ctx: &egui::Context, sel: &Selection) {
    ctx.data_mut(|d| d.insert_temp(selection_id(), sel.clone()));
}

/// The basic atoms and pins (for menus and palettes): 2-input gates, NOT, BUF, TRI, an 8-bit
/// splitter, a constant, a clock, a power-on reset, input and output pins.
pub fn basic_palette() -> Vec<PlaceRequest> {
    let gate = |op| {
        PlaceRequest::Atom(Atom::Gate {
            op,
            inputs: 2,
            width: 1,
            reduce: false,
        })
    };
    vec![
        PlaceRequest::Pin {
            dir: Dir::In,
            width: 1,
        },
        PlaceRequest::Pin {
            dir: Dir::Out,
            width: 1,
        },
        gate(GateOp::And),
        gate(GateOp::Or),
        gate(GateOp::Nand),
        gate(GateOp::Nor),
        gate(GateOp::Xor),
        gate(GateOp::Xnor),
        PlaceRequest::Atom(Atom::Not { width: 1 }),
        PlaceRequest::Atom(Atom::Buf { width: 1 }),
        PlaceRequest::Atom(Atom::Tri { width: 1 }),
        PlaceRequest::Atom(Atom::Splitter {
            width: 8,
            parts: vec![(4, 7), (0, 3)],
        }),
        PlaceRequest::Atom(Atom::Constant {
            value: Value::from_bool(true),
        }),
        PlaceRequest::Atom(Atom::Clock {
            period: 16,
            high: 8,
            phase: 0,
        }),
        PlaceRequest::Atom(Atom::PowerOnReset { ticks: 4 }),
    ]
}

#[cfg(test)]
mod tests;
