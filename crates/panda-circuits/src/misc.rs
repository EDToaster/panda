//! Oscillators and tri-state bus blocks.

use panda_netlist::{DefBuilder, DefRef, DesignBuilder};

/// Ring oscillator of `n` inverting stages (`n` odd, `n ≥ 3`).
///
/// * `enable = true`: id `"ring_osc_en_{n}"`, pins `EN` (in, 1) → `OUT` (out, 1). Stage 0 is
///   `NAND(EN, OUT)`, the rest are inverters. `EN = 0` holds `OUT` at a fixed level (stage 0 = 1);
///   `EN = 1` oscillates with period `2n` ticks in Timing mode.
/// * `enable = false`: id `"ring_osc_{n}"`, pins `OUT` (out, 1) only; `n` inverters. Started from
///   the all-`X` power-on state it stays `X` (there is nothing to break the symmetry) — useful for
///   oscillator classification, not as a clock.
///
/// Stage outputs are named `s0..s{n-1}`; `OUT = s{n-1}`.
pub fn ring_oscillator(db: &mut DesignBuilder, n: u32, enable: bool) -> DefRef {
    assert!(n >= 3 && n % 2 == 1, "ring_oscillator: n must be odd and ≥ 3");
    let id = if enable { format!("ring_osc_en_{n}") } else { format!("ring_osc_{n}") };
    let mut b = DefBuilder::new(&id);
    let en = enable.then(|| b.input("EN", 1));
    let fb = b.wire(1);
    let mut s = match en {
        Some(en) => b.nand(&[en, fb]),
        None => b.not(fb),
    };
    b.name(s, "s0");
    for i in 1..n {
        let next = if i == n - 1 { fb } else { b.wire(1) };
        b.not_to(s, next);
        s = b.name(next, &format!("s{i}"));
    }
    b.output("OUT", fb);
    db.add_once(b.finish())
}

/// Tri-state bus port: drives a shared bus and reads it back. Id `"bus_port_{w}"`.
///
/// Pins: `D` (in, w), `OE` (in, 1), `BUS` (inout, w) → `Q` (out, w).
/// `OE = 1` drives `D` onto `BUS` (through a `Tri`); `OE = 0` releases it (`Z`). `Q` is a buffered
/// copy of the resolved bus value (delay 1). Connect several ports to one net (optionally with a
/// pull-up/down termination) to build a bus.
pub fn bus_port(db: &mut DesignBuilder, w: u32) -> DefRef {
    assert!(w >= 1, "bus_port: width must be ≥ 1");
    let id = format!("bus_port_{w}");
    let mut b = DefBuilder::new(&id);
    let d = b.input("D", w);
    let oe = b.input("OE", 1);
    let bus = b.inout("BUS", w);
    b.tri_to(d, oe, bus);
    let q = b.buf(bus);
    b.output("Q", q);
    db.add_once(b.finish())
}

/// Bidirectional bus transceiver (74245-style). Id `"transceiver_{w}"`.
///
/// Pins: `A` (inout, w), `B` (inout, w), `DIR` (in, 1), `OE` (in, 1); no `Out` pins.
/// `OE = 1, DIR = 1`: drives `B` from `A`; `OE = 1, DIR = 0`: drives `A` from `B`;
/// `OE = 0`: both sides released (`Z`). Each direction is one `Tri` (delay 1).
///
/// As with the real part, change `DIR` only while `OE = 0`: changing both at once glitches the
/// direction enables (`a2b = OE·DIR`, `b2a = OE·¬DIR`) and briefly drives both sides
/// (a transient `Contention`). Reversing an active bus likewise needs a turnaround (release the
/// driving side first).
pub fn transceiver(db: &mut DesignBuilder, w: u32) -> DefRef {
    assert!(w >= 1, "transceiver: width must be ≥ 1");
    let id = format!("transceiver_{w}");
    let mut b = DefBuilder::new(&id);
    let a = b.inout("A", w);
    let bb = b.inout("B", w);
    let dir = b.input("DIR", 1);
    let oe = b.input("OE", 1);
    let ndir = b.not(dir);
    let a2b = b.and(&[oe, dir]);
    let b2a = b.and(&[oe, ndir]);
    b.name(a2b, "a2b");
    b.name(b2a, "b2a");
    b.tri_to(a, a2b, bb);
    b.tri_to(bb, b2a, a);
    db.add_once(b.finish())
}
