//! panda-circuits — a library of purist gate-level building blocks and generators (SPEC D2).
//!
//! Every block is built only from gate atoms, splitters and constants (plus `Tri` where a block is
//! explicitly tri-state). State arises from gate feedback, never from a primitive.
//!
//! # Conventions
//!
//! * Generators take `&mut DesignBuilder` and return the [`DefRef`](panda_netlist::DefRef) of the
//!   block. Sub-defs are added with [`DesignBuilder::add_once`](panda_netlist::DesignBuilder::add_once),
//!   so every distinct block appears once in the design and is shared by all its users.
//! * Def ids encode every parameter (`"ripple_adder_8"`, `"mux_3x8"`, `"regfile_gated_4x8"`), so
//!   two calls with equal parameters return the same def. ROM ids also contain a content hash.
//! * Pin lists are documented on each generator **in declaration order**. That order matters for
//!   [`DefBuilder::inst`](panda_netlist::DefBuilder::inst): it takes the `In`/`InOut` pins in
//!   declaration order and returns the `Out` pins in declaration order.
//! * Multi-bit buses are LSB first. Clock pins are named `CLK`, synchronous resets `RST`.
//! * Sequential blocks are rising-edge master-slave designs (two transparent D latches). Inputs
//!   must be stable around the rising edge of `CLK` (setup/hold); the hold requirement is met by
//!   any data path that goes through at least one gate after a flip-flop output.
//!
//! # Modules
//!
//! * [`seq`] — latches, flip-flops, registers.
//! * [`arith`] — adders, incrementer, muxes, decoder, comparator.
//! * [`counters`] — synchronous and ripple counters.
//! * [`rom`](mod@rom) — constant + mux-tree ROMs (SPEC §7.4) and a hex loader.
//! * [`regfile`] — register files (enable-mux and gated-clock styles).
//! * [`misc`] — ring oscillator, tri-state bus port and bidirectional transceiver.
//! * [`testbench`](mod@testbench) — wrap a DUT in a top def with `Clock` / `PowerOnReset` sources.
//! * [`transform`] — design transforms ([`transform::inline_all`]).

pub mod arith;
pub mod counters;
pub mod misc;
pub mod regfile;
pub mod rom;
pub mod seq;
pub mod testbench;
pub mod transform;
mod util;

pub use arith::{decoder, equality, full_adder, half_adder, incrementer, mux, mux2, ripple_adder};
pub use counters::{ripple_counter, sync_counter};
pub use misc::{bus_port, ring_oscillator, transceiver};
pub use regfile::{RegFileStyle, register_file};
pub use rom::{rom, rom_from_hex};
pub use seq::{
    d_latch, d_latch_clr, dff_arst, dff_en, dff_rst, ms_dff, nand_sr_latch, nor_sr_latch, register, register_gated,
    toggle_ff,
};
pub use testbench::{TestbenchOptions, testbench, testbench_with};
pub use transform::inline_all;
pub use util::addr_bits;
