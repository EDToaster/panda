//! Built-in example designs (used by tests, the perf gate, the default document and the
//! **Examples** menu of [`crate::files`]).
//!
//! Keep these constructors stable, tests depend on them. [`all`] lists every example shown in
//! the menu (SR latch, counter, register file, Panda-8 with each built-in program). The designs
//! carry no geometry; [`crate::app::AppState::open_design`] lays them out with
//! [`crate::layout`] on a worker thread.

use panda_circuits::{RegFileStyle, nand_sr_latch, register_file, sync_counter, testbench};
use panda_cpu8::programs::Program;
use panda_cpu8::{Variant, panda8 as gen_panda8, programs};
use panda_netlist::{Design, DesignBuilder};

/// A 4-bit synchronous counter in a testbench (`Clock` period 16, `PowerOnReset`; input `EN`,
/// output `Q`).
pub fn counter() -> Design {
    let mut db = DesignBuilder::new();
    let c = sync_counter(&mut db, 4);
    let top = testbench(&mut db, &c, 16);
    db.build(&top)
}

/// Panda-8 running `bench_loop` (sum-of-products ROM, hierarchical variant, default period).
pub fn panda8() -> Design {
    panda8_program(&programs::BENCH_LOOP)
}

/// Panda-8 running `program` (sum-of-products ROM, hierarchical variant, default period).
pub fn panda8_program(program: &Program) -> Design {
    gen_panda8(&program.image(), Variant::Hierarchical)
}

/// A NAND SR latch as the top def: inputs `S_n`, `R_n` (active low), outputs `Q`, `Q_n`; the
/// two cross-coupled gates are visible at the top level.
pub fn sr_latch() -> Design {
    let mut db = DesignBuilder::new();
    let l = nand_sr_latch(&mut db);
    db.build(&l)
}

/// A 4 × 8 register file (enable-mux style) in a testbench (`Clock` period 32, `PowerOnReset`;
/// inputs `WD`, `WA`, `WE`, `RA`, output `RD`).
pub fn register_file_4x8() -> Design {
    let mut db = DesignBuilder::new();
    let rf = register_file(&mut db, 4, 8, RegFileStyle::EnableMux);
    let top = testbench(&mut db, &rf, 32);
    db.build(&top)
}

/// One entry of the examples menu.
#[derive(Clone, Copy, Debug)]
pub struct Example {
    /// Menu label.
    pub name: &'static str,
    /// One-line description (menu tooltip).
    pub description: &'static str,
    /// Constructor.
    pub build: fn() -> Design,
}

/// Every built-in example, in menu order.
pub fn all() -> Vec<Example> {
    let mut v = vec![
        Example {
            name: "SR latch",
            description: "NAND SR latch: two cross-coupled gates, active-low set/reset",
            build: sr_latch,
        },
        Example {
            name: "Counter",
            description: "4-bit synchronous counter with clock and power-on reset",
            build: counter,
        },
        Example {
            name: "Register file",
            description: "4 x 8 register file (one write port, one read port) in a testbench",
            build: register_file_4x8,
        },
    ];
    // `fn()` pointers cannot capture the program, so each program gets its own constructor.
    fn fib() -> Design {
        panda8_program(&programs::FIBONACCI)
    }
    fn countdown() -> Design {
        panda8_program(&programs::COUNTDOWN)
    }
    fn multiply() -> Design {
        panda8_program(&programs::MULTIPLY)
    }
    type Ctor = fn() -> Design;
    let p8: [(&'static str, Ctor); 4] = [
        ("Panda-8: fibonacci", fib),
        ("Panda-8: countdown", countdown),
        ("Panda-8: multiply", multiply),
        ("Panda-8: bench_loop", panda8),
    ];
    for ((name, build), prog) in p8.into_iter().zip(programs::ALL) {
        v.push(Example {
            name,
            description: prog.description,
            build,
        });
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn menu_names_match_programs() {
        let all = all();
        assert_eq!(all.len(), 3 + programs::ALL.len());
        for (e, p) in all[3..].iter().zip(programs::ALL) {
            assert!(e.name.ends_with(p.name), "{} vs {}", e.name, p.name);
        }
    }
}
