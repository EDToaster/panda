//! Register files: one write port, one read port.

use panda_netlist::{DefBuilder, DefRef, DesignBuilder, Sig};

use crate::arith::{decoder, mux};
use crate::seq::{register, register_gated};
use crate::util::addr_bits;

/// How the per-word load enable is implemented.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RegFileStyle {
    /// Each word is a [`register`]: a hold mux selects between `WD` and the current value; every
    /// flip-flop sees the free-running clock.
    EnableMux,
    /// Each word is a [`register_gated`]: its clock is `AND(CLK, load)` (SPEC §12 variant c).
    /// `WA`, `WE` and `RST` must not change while `CLK` is high (see [`register_gated`]).
    GatedClock,
}

/// A `words × width` register file with one synchronous write port and one combinational read
/// port. Id `"regfile_{words}x{width}"` ([`RegFileStyle::EnableMux`]) or
/// `"regfile_gated_{words}x{width}"` ([`RegFileStyle::GatedClock`]).
///
/// Pins, with `a = addr_bits(words)`:
/// `WD` (in, width), `WA` (in, a), `WE` (in, 1), `RA` (in, a), `RST` (in, 1), `CLK` (in, 1)
/// → `RD` (out, width).
///
/// * Write: at a rising edge of `CLK`, if `WE = 1` and `WA < words`, word `WA` ← `WD`.
/// * Reset: at a rising edge with `RST = 1`, every word ← 0 (takes priority over the write).
/// * Read: `RD = word[RA]` (combinational); addresses `≥ words` read 0.
///
/// Structure: a [`decoder`] (`wdec`, enable = `WE`) drives the `LOAD` of one register per word
/// (`w0..`, outputs named `r0..`), and a [`mux`] tree (`rmux`) selects the read word.
pub fn register_file(db: &mut DesignBuilder, words: usize, width: u32, style: RegFileStyle) -> DefRef {
    assert!(words >= 2, "register_file: need at least 2 words");
    assert!(width >= 1, "register_file: width must be ≥ 1");
    let a = addr_bits(words);
    let dec = decoder(db, a);
    let rmux = mux(db, a, width);
    let (reg, id) = match style {
        RegFileStyle::EnableMux => (register(db, width), format!("regfile_{words}x{width}")),
        RegFileStyle::GatedClock => (register_gated(db, width), format!("regfile_gated_{words}x{width}")),
    };
    let mut b = DefBuilder::new(&id);
    let wd = b.input("WD", width);
    let wa = b.input("WA", a);
    let we = b.input("WE", 1);
    let ra = b.input("RA", a);
    let rst = b.input("RST", 1);
    let clk = b.input("CLK", 1);
    let sel = b.inst_named("wdec", &dec, &[wa, we])[0];
    b.name(sel, "wsel");
    let loads = b.bits(sel);
    let mut read_ins: Vec<Sig> = (0..words)
        .map(|i| {
            let q = b.inst_named(&format!("w{i}"), &reg, &[wd, loads[i], rst, clk])[0];
            b.name(q, &format!("r{i}"))
        })
        .collect();
    if read_ins.len() < 1 << a {
        let zero = b.lit(width, 0);
        read_ins.resize(1 << a, zero);
    }
    read_ins.push(ra);
    let rd = b.inst_named("rmux", &rmux, &read_ins)[0];
    b.output("RD", rd);
    db.add_once(b.finish())
}
