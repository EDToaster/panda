//! Adversarial designs for the STA contract: clocked structures where FastRef (SPEC §6.2) and
//! Timing mode (§5) are most likely to disagree at phase boundaries — glitchy and pulse-generated
//! clocks, reconvergent clock paths, latch chains and loops across both phases, very short hold
//! paths, clock-derived tri-state enables, derived clocks racing data, and several clocks. Each
//! design is swept over a range of clock periods: whenever STA reports it clean, FastRef must equal
//! Timing mode on every net at every phase boundary.

mod common;

use common::*;
use panda_circuits::{d_latch, ms_dff, nand_sr_latch, toggle_ff};
use panda_netlist::{DefBuilder, Design, DesignBuilder, Sig, Termination};

/// Result of a period sweep.
#[derive(Debug, Default)]
struct Sweep {
    clean: Vec<u32>,
    flagged_real: Vec<u32>,
    flagged_equal: Vec<u32>,
    unsound: Vec<(u32, String)>,
}

/// Sweep `make(period)` over `periods`: STA verdict vs FastRef/Timing comparison from
/// `from(period)` on.
fn sweep(
    name: &str,
    periods: impl IntoIterator<Item = u32>,
    make: impl Fn(u32) -> Design,
    from: impl Fn(u32) -> u64,
    events: usize,
) -> Sweep {
    let mut s = Sweep::default();
    for p in periods {
        let flat = flat_of(&make(p));
        let r = sta(&flat);
        let diff = differential(&flat, events, from(p));
        match (r.is_clean(), diff) {
            (true, Ok(_)) => s.clean.push(p),
            (true, Err(m)) => {
                dump(&format!("{name} @ {p} (UNSOUND)"), &r);
                s.unsound.push((p, format!("{m:?}")));
            }
            (false, Err(_)) => s.flagged_real.push(p),
            (false, Ok(_)) => s.flagged_equal.push(p),
        }
    }
    println!(
        "{name}: clean {:?}, flagged+real {:?}, flagged+equal {:?}",
        s.clean, s.flagged_real, s.flagged_equal
    );
    s
}

fn assert_sound(name: &str, s: &Sweep) {
    assert!(
        s.unsound.is_empty(),
        "{name}: STA clean but FastRef != Timing: {:?}",
        s.unsound
    );
}

/// Reset release tick used by the designs below (a quarter period into a low phase).
fn rst_ticks(p: u32) -> u32 {
    2 * p + p / 4
}

fn from_reset(p: u32) -> u64 {
    u64::from(rst_ticks(p) + p)
}

/// A builder with `clk` (period `p`, first rising edge at `p/2`) and `rst`.
fn top(id: &str, p: u32) -> (DefBuilder, Sig, Sig) {
    let mut b = DefBuilder::new(id);
    let clk = b.clock(p, p / 2, p / 2);
    b.name(clk, "clk");
    let rst = b.power_on_reset(rst_ticks(p));
    b.name(rst, "rst");
    (b, clk, rst)
}

fn bufs(b: &mut DefBuilder, mut s: Sig, n: usize) -> Sig {
    for _ in 0..n {
        s = b.buf(s);
    }
    s
}

/// A toggling flip-flop (`T = 1`, sync reset) on `clk`: a data source that changes at every edge.
fn toggler(
    b: &mut DefBuilder,
    db_tff: &panda_netlist::DefRef,
    name: &str,
    clk: Sig,
    rst: Sig,
) -> Sig {
    let one = b.lit(1, 1);
    b.inst_named(name, db_tff, &[one, rst, clk])[0]
}

#[test]
fn gated_clock_enable_changing_while_clock_high() {
    // en toggles at every rising edge (so it changes while clk = 1): gclk = clk · en gets a late
    // rising edge or an early falling edge in the middle of the high phase. A flip-flop on gclk
    // samples the output of a flip-flop on clk.
    let make = |p: u32| {
        let mut db = DesignBuilder::new();
        let tff = toggle_ff(&mut db);
        let ff = ms_dff(&mut db);
        let (mut b, clk, rst) = top(&format!("gce{p}"), p);
        let en = toggler(&mut b, &tff, "en", clk, rst);
        let gclk = b.and(&[clk, en]);
        b.name(gclk, "gclk");
        let a = toggler(&mut b, &tff, "a", clk, rst);
        let nrst = b.not(rst);
        let d = b.and(&[a, nrst]);
        let q = b.inst_named("g", &ff, &[d, gclk])[0];
        b.output("Q", q);
        b.output("A", a);
        b.output("EN", en);
        let t = db.add(b.finish());
        db.build(&t)
    };
    let s = sweep(
        "gated clock (en changes while high)",
        (8..=64).step_by(4),
        make,
        from_reset,
        200,
    );
    assert_sound("gce", &s);
}

#[test]
fn pulse_generator_private_to_one_latch() {
    // pclk = clk · ¬delay_k(clk), used by a single toggle latch (D = ¬Q): the pulse logic may be
    // absorbed into the latch kernel as private fan-in. Pulse width k decides how many times the
    // latch toggles per edge in Timing mode.
    for k in [1usize, 2, 3, 5, 8] {
        let make = move |p: u32| {
            let mut db = DesignBuilder::new();
            let l = d_latch(&mut db);
            let (mut b, clk, _rst) = top(&format!("pulse{k}_{p}"), p);
            let dl = bufs(&mut b, clk, k);
            let nd = b.not(dl);
            let pclk = b.and(&[clk, nd]);
            let qw = b.wire(1);
            let nq = b.not(qw);
            let q = b.inst_named("l", &l, &[nq, pclk])[0];
            b.connect(q, qw);
            b.output("Q", q);
            let t = db.add(b.finish());
            db.build(&t)
        };
        let s = sweep(
            &format!("private pulse latch k={k}"),
            (8..=48).step_by(8),
            make,
            from_reset,
            120,
        );
        assert_sound("pulse", &s);
    }
}

#[test]
fn pulse_generator_shared_by_flip_flops_and_latches() {
    for k in [1usize, 3, 6] {
        let make = move |p: u32| {
            let mut db = DesignBuilder::new();
            let tff = toggle_ff(&mut db);
            let l = d_latch(&mut db);
            let (mut b, clk, rst) = top(&format!("pulses{k}_{p}"), p);
            let dl = bufs(&mut b, clk, k);
            let nd = b.not(dl);
            let pclk = b.and(&[clk, nd]);
            b.name(pclk, "pclk");
            let a = toggler(&mut b, &tff, "a", clk, rst);
            let q0 = b.inst_named("l0", &l, &[a, pclk])[0];
            let q1 = b.inst_named("l1", &l, &[q0, pclk])[0];
            let q2 = toggler(&mut b, &tff, "t", pclk, rst);
            b.output("Q0", q0);
            b.output("Q1", q1);
            b.output("Q2", q2);
            let t = db.add(b.finish());
            db.build(&t)
        };
        let s = sweep(
            &format!("shared pulse k={k}"),
            (8..=48).step_by(8),
            make,
            from_reset,
            120,
        );
        assert_sound("pulse shared", &s);
    }
}

#[test]
fn reconvergent_clock_paths() {
    // gclk = clk · delay_k(clk) (no glitch, but asymmetric edges), and a skewed copy of clk;
    // flip-flops on all three clocks exchange data with short paths.
    for k in [1usize, 2, 4, 7] {
        let make = move |p: u32| {
            let mut db = DesignBuilder::new();
            let ff = ms_dff(&mut db);
            let (mut b, clk, rst) = top(&format!("reconv{k}_{p}"), p);
            let dl = bufs(&mut b, clk, k);
            let and_clk = b.and(&[clk, dl]);
            let or_clk = b.or(&[clk, dl]);
            let nrst = b.not(rst);
            let q0w = b.wire(1);
            let q1w = b.wire(1);
            let q2w = b.wire(1);
            let x0 = b.xor(&[q2w, nrst]);
            let d0 = b.and(&[x0, nrst]);
            let q0 = b.inst_named("f0", &ff, &[d0, clk])[0];
            let d1 = b.and(&[q0w, nrst]);
            let q1 = b.inst_named("f1", &ff, &[d1, and_clk])[0];
            let d2 = b.and(&[q1w, nrst]);
            let q2 = b.inst_named("f2", &ff, &[d2, or_clk])[0];
            b.connect(q0, q0w);
            b.connect(q1, q1w);
            b.connect(q2, q2w);
            b.output("Q0", q0);
            b.output("Q1", q1);
            b.output("Q2", q2);
            let t = db.add(b.finish());
            db.build(&t)
        };
        let s = sweep(
            &format!("reconvergent clocks k={k}"),
            (8..=48).step_by(4),
            make,
            from_reset,
            150,
        );
        assert_sound("reconv", &s);
    }
}

#[test]
fn reconvergent_clock_xor_glitch() {
    // gclk = clk XOR delay_k(clk): a pulse at *both* edges (a clock doubler).
    for k in [1usize, 2, 4] {
        let make = move |p: u32| {
            let mut db = DesignBuilder::new();
            let tff = toggle_ff(&mut db);
            let l = d_latch(&mut db);
            let (mut b, clk, rst) = top(&format!("xorclk{k}_{p}"), p);
            let dl = bufs(&mut b, clk, k);
            let x = b.xor(&[clk, dl]);
            b.name(x, "xclk");
            let q = toggler(&mut b, &tff, "t", x, rst);
            let a = toggler(&mut b, &tff, "a", clk, rst);
            let lq = b.inst_named("l", &l, &[a, x])[0];
            b.output("Q", q);
            b.output("L", lq);
            let t = db.add(b.finish());
            db.build(&t)
        };
        let s = sweep(
            &format!("xor clock doubler k={k}"),
            (8..=48).step_by(8),
            make,
            from_reset,
            150,
        );
        assert_sound("xorclk", &s);
    }
}

#[test]
fn latch_chain_with_long_logic_between_phases() {
    // L1 (clk) → n buffers → L2 (¬clk) → n buffers → L3 (clk) → ¬ → L1: a two-phase ring whose
    // stages need most of a half period (time borrowing would be needed below the minimum).
    for n in [0usize, 3, 9] {
        let make = move |p: u32| {
            let mut db = DesignBuilder::new();
            let l = d_latch(&mut db);
            let (mut b, clk, rst) = top(&format!("chain{n}_{p}"), p);
            let nclk = b.not(clk);
            let nrst = b.not(rst);
            let q3w = b.wire(1);
            let nq3 = b.not(q3w);
            let d1 = b.and(&[nq3, nrst]);
            let q1 = b.inst_named("l1", &l, &[d1, clk])[0];
            let a = bufs(&mut b, q1, n);
            let q2 = b.inst_named("l2", &l, &[a, nclk])[0];
            let c = bufs(&mut b, q2, n);
            let q3 = b.inst_named("l3", &l, &[c, clk])[0];
            b.connect(q3, q3w);
            b.output("Q1", q1);
            b.output("Q2", q2);
            b.output("Q3", q3);
            let t = db.add(b.finish());
            db.build(&t)
        };
        let s = sweep(
            &format!("latch chain n={n}"),
            (8..=64).step_by(4),
            make,
            from_reset,
            150,
        );
        assert_sound("chain", &s);
    }
}

#[test]
fn two_phase_latch_loop_with_overlapping_enables() {
    // L1 transparent while clk = 1, L2 while en2 = ¬delay_k(clk) = 1: at the rising edge both are
    // transparent for k+1 ticks (overlap), so the loop L1 → L2 → ¬ → L1 can race through.
    for k in [0usize, 1, 2, 4, 8] {
        let make = move |p: u32| {
            let mut db = DesignBuilder::new();
            let l = d_latch(&mut db);
            let (mut b, clk, rst) = top(&format!("overlap{k}_{p}"), p);
            let dl = bufs(&mut b, clk, k);
            let en2 = b.not(dl);
            b.name(en2, "en2");
            let nrst = b.not(rst);
            let q2w = b.wire(1);
            let nq2 = b.not(q2w);
            let d1 = b.and(&[nq2, nrst]);
            let q1 = b.inst_named("l1", &l, &[d1, clk])[0];
            let q2 = b.inst_named("l2", &l, &[q1, en2])[0];
            b.connect(q2, q2w);
            b.output("Q1", q1);
            b.output("Q2", q2);
            let t = db.add(b.finish());
            db.build(&t)
        };
        let s = sweep(
            &format!("overlapping latch loop k={k}"),
            (8..=48).step_by(4),
            make,
            from_reset,
            150,
        );
        assert_sound("overlap", &s);
    }
}

#[test]
fn hold_race_through_shortest_paths() {
    // ff2's clock is delayed by k buffers, D2 = Q1 directly (the hold design of `common`), plus
    // a latch whose D comes straight from a flip-flop on the same edge.
    for k in 0..6usize {
        let s = sweep(
            &format!("hold race k={k}"),
            (8..=48).step_by(8),
            |p| hold_design(p, k),
            |p| u64::from(2 * p + p / 4 + p),
            150,
        );
        assert_sound("hold", &s);
    }
}

#[test]
fn tri_state_bus_with_clock_derived_enables() {
    // bus = TRI(a, clk) | TRI(b, ¬delay_k(clk)) with a termination; the bus feeds a flip-flop, a
    // latch and plain logic. Overlapping enables cause transient contention.
    for (k, term) in [
        (0usize, Termination::PullDown),
        (2, Termination::PullDown),
        (0, Termination::Keeper),
        (3, Termination::Keeper),
        (1, Termination::None),
    ] {
        let make = move |p: u32| {
            let mut db = DesignBuilder::new();
            let tff = toggle_ff(&mut db);
            let ff = ms_dff(&mut db);
            let l = d_latch(&mut db);
            let (mut b, clk, rst) = top(&format!("tri{k}{term:?}{p}"), p);
            let a = toggler(&mut b, &tff, "a", clk, rst);
            let na = b.not(a);
            let dl = bufs(&mut b, clk, k);
            let en_b = b.not(dl);
            let bus = b.wire(1);
            b.terminate(bus, term);
            b.tri_to(a, clk, bus);
            b.tri_to(na, en_b, bus);
            b.name(bus, "bus");
            let nclk = b.not(clk);
            let q = b.inst_named("f", &ff, &[bus, nclk])[0];
            let lq = b.inst_named("l", &l, &[bus, clk])[0];
            let x = b.xor(&[bus, a]);
            b.output("Q", q);
            b.output("L", lq);
            b.output("X", x);
            let t = db.add(b.finish());
            db.build(&t)
        };
        let s = sweep(
            &format!("tri bus k={k} {term:?}"),
            (8..=48).step_by(8),
            make,
            from_reset,
            150,
        );
        assert_sound("tri", &s);
    }
}

#[test]
fn ripple_derived_clock_racing_data() {
    // b is clocked by a.Q_n (a derived clock, one flip-flop delay after clk↑) and samples c.Q,
    // where c toggles on clk through `n` buffers: the data path races the derived clock.
    for n in [0usize, 1, 2, 4, 8, 12] {
        let make = move |p: u32| {
            let mut db = DesignBuilder::new();
            let tff = toggle_ff(&mut db);
            let ff = ms_dff(&mut db);
            let (mut b, clk, rst) = top(&format!("ripple_race{n}_{p}"), p);
            let one = b.lit(1, 1);
            let ao = b.inst_named("a", &tff, &[one, rst, clk]);
            let c = toggler(&mut b, &tff, "c", clk, rst);
            let cd = bufs(&mut b, c, n);
            let q = b.inst_named("b", &ff, &[cd, ao[1]])[0];
            b.output("A", ao[0]);
            b.output("C", c);
            b.output("Q", q);
            let t = db.add(b.finish());
            db.build(&t)
        };
        let s = sweep(
            &format!("ripple race n={n}"),
            (8..=64).step_by(8),
            make,
            from_reset,
            150,
        );
        assert_sound("ripple race", &s);
    }
}

#[test]
fn sr_latch_set_by_static_hazard() {
    // S_n = NAND(q, ¬delay_k(q)) pulses low for k+1 ticks whenever q rises (q toggles on clk):
    // the pulse sets a NAND SR latch that is reset by the reset only.
    for k in [0usize, 1, 3] {
        let make = move |p: u32| {
            let mut db = DesignBuilder::new();
            let tff = toggle_ff(&mut db);
            let sr = nand_sr_latch(&mut db);
            let (mut b, clk, rst) = top(&format!("srhaz{k}_{p}"), p);
            let q = toggler(&mut b, &tff, "t", clk, rst);
            let qd = bufs(&mut b, q, k);
            let nqd = b.not(qd);
            let s_n = b.nand(&[q, nqd]);
            b.name(s_n, "s_n");
            let r_n = b.not(rst);
            let o = b.inst_named("sr", &sr, &[s_n, r_n]);
            b.output("Q", q);
            b.output("SR", o[0]);
            let t = db.add(b.finish());
            db.build(&t)
        };
        let s = sweep(
            &format!("SR latch static hazard k={k}"),
            (8..=48).step_by(8),
            make,
            from_reset,
            150,
        );
        assert_sound("srhaz", &s);
    }
}

#[test]
fn clock_used_as_data() {
    // D = clk XOR q (and D = clk AND q) into flip-flops on clk and ¬clk.
    let make = |p: u32| {
        let mut db = DesignBuilder::new();
        let ff = ms_dff(&mut db);
        let (mut b, clk, rst) = top(&format!("clkdata{p}"), p);
        let nrst = b.not(rst);
        let q0w = b.wire(1);
        let x = b.xor(&[clk, q0w]);
        let d0 = b.and(&[x, nrst]);
        let q0 = b.inst_named("f0", &ff, &[d0, clk])[0];
        b.connect(q0, q0w);
        let nclk = b.not(clk);
        let y = b.and(&[clk, q0]);
        let q1 = b.inst_named("f1", &ff, &[y, nclk])[0];
        b.output("Q0", q0);
        b.output("Q1", q1);
        let t = db.add(b.finish());
        db.build(&t)
    };
    let s = sweep("clock as data", (8..=48).step_by(4), make, from_reset, 150);
    assert_sound("clkdata", &s);
}

#[test]
fn gated_ring_oscillator() {
    // A ring oscillator enabled while clk = 1 (directly, or through a flip-flop): 5 stages make
    // a characterized kernel, 7 and 9 stages a Sim kernel (loop longer than KERNEL_MAX_LOOP).
    for n in [3u32, 5, 7, 9, 15] {
        for via_ff in [false, true] {
            let make = move |p: u32| {
                let mut db = DesignBuilder::new();
                let tff = toggle_ff(&mut db);
                let ring = panda_circuits::ring_oscillator(&mut db, n, true);
                let (mut b, clk, rst) = top(&format!("ring{n}{via_ff}{p}"), p);
                let en = if via_ff {
                    toggler(&mut b, &tff, "en", clk, rst)
                } else {
                    clk
                };
                let o = b.inst_named("ring", &ring, &[en])[0];
                b.output("OUT", o);
                let t = db.add(b.finish());
                db.build(&t)
            };
            let s = sweep(
                &format!("gated ring n={n} ff={via_ff}"),
                (8..=256).step_by(24),
                make,
                from_reset,
                60,
            );
            assert_sound("ring", &s);
        }
    }
}

/// A ring oscillator of `n` NAND stages, each with its own enable input (`clk` for stage 0, top
/// inputs `E1..` for the others): too many inputs for a table, so it becomes a Sim kernel.
fn wide_ring(p: u32, n: usize) -> Design {
    let mut db = DesignBuilder::new();
    let (mut b, clk, _rst) = top(&format!("wide_ring{n}_{p}"), p);
    let fb = b.wire(1);
    let mut s = b.nand(&[clk, fb]);
    for i in 1..n {
        let e = b.input(&format!("E{i}"), 1);
        s = if i == n - 1 {
            b.nand_to(&[e, s], fb);
            fb
        } else {
            b.nand(&[e, s])
        };
    }
    b.output("OUT", fb);
    let t = db.add(b.finish());
    db.build(&t)
}

#[test]
fn sim_kernel_ring_oscillator() {
    for n in [15usize, 17] {
        let inputs: Vec<(String, panda_netlist::Value)> = (1..n)
            .map(|i| (format!("E{i}"), panda_netlist::Value::from_bool(true)))
            .collect();
        let inputs: Vec<(&str, panda_netlist::Value)> = inputs
            .iter()
            .map(|(s, v)| (s.as_str(), v.clone()))
            .collect();
        for p in [16u32, 64, 200, 400] {
            let flat = flat_of(&wide_ring(p, n));
            let (ks, _) = panda_compile::compile_kernels(&flat, &Default::default());
            let kinds: Vec<_> = ks.kernels.iter().map(|k| k.kind).collect();
            let r = sta(&flat);
            let diff = differential_with(&flat, 40, from_reset(p), &inputs);
            println!(
                "wide ring n={n} p={p}: kernels {kinds:?} clean {} diff {:?}",
                r.is_clean(),
                diff.as_ref().err()
            );
            if r.is_clean() {
                assert!(diff.is_ok(), "n={n} p={p}: STA clean but {diff:?}");
            }
        }
    }
}

/// One table kernel with stable states for every input vector, but a transition into a limit
/// cycle: a NAND latch (`q`, `qn`) whose `q = 1` state enables a 3-stage ring (`r0..r2`, also
/// enabled by `EN`); the ring feeds back into `q` only through `OR(r2, qn)`, which is 1 while
/// `q = 0`. With `S_n = R_n = 1, EN = 1`, `q = 0` is stable, but from `q = 1` (set during the
/// reset) raising `EN` starts an endless oscillation.
fn latch_with_ring(p: u32, en_from_clk: bool) -> Design {
    let mut db = DesignBuilder::new();
    let (mut b, clk, rst) = top(&format!("latch_ring{en_from_clk}{p}"), p);
    let s_n = b.not(rst);
    let one = b.lit(1, 1);
    let en = if en_from_clk { clk } else { b.input("EN", 1) };
    let qn = b.wire(1);
    let r2 = b.wire(1);
    let o = b.or(&[r2, qn]);
    let q = b.nand(&[s_n, qn, o]);
    b.nand_to(&[one, q], qn);
    let r0 = b.nand(&[q, en, r2, s_n]);
    let r1 = b.not(r0);
    b.not_to(r1, r2);
    b.name(q, "q");
    b.name(r2, "r2");
    b.output("Q", q);
    b.output("R", r2);
    let t = db.add(b.finish());
    db.build(&t)
}

#[test]
fn table_transition_into_limit_cycle() {
    for p in [16u32, 64, 256] {
        let flat = flat_of(&latch_with_ring(p, true));
        let (ks, tables) = panda_compile::compile_kernels(&flat, &Default::default());
        let kinds: Vec<_> = ks.kernels.iter().map(|k| k.kind).collect();
        let osc: Vec<_> = tables
            .iter()
            .map(|t| t.as_ref().map(|t| t.is_oscillator()))
            .collect();
        let r = sta(&flat);
        dump(&format!("latch+ring @ {p}"), &r);
        let diff = differential(&flat, 40, from_reset(p));
        println!(
            "kernels {kinds:?} oscillator {osc:?} clean {} diff {:?}",
            r.is_clean(),
            diff.as_ref().err()
        );
        if r.is_clean() {
            assert!(diff.is_ok(), "p={p}: STA clean but {diff:?}");
        }
    }
}

#[test]
fn metastable_release_by_clock() {
    // A NAND SR latch whose S_n and R_n are both driven by clk (released 00 → 11 at every rising
    // edge: period-2 collapse in both modes), and variants where downstream logic and a flip-flop
    // read the collapsed (X) latch.
    for variant in 0..3 {
        let make = move |p: u32| {
            let mut db = DesignBuilder::new();
            let sr = nand_sr_latch(&mut db);
            let ff = ms_dff(&mut db);
            let (mut b, clk, rst) = top(&format!("meta{variant}_{p}"), p);
            let o = match variant {
                0 => b.inst_named("sr", &sr, &[clk, clk]),
                1 => {
                    let c2 = b.buf(clk);
                    let c1 = b.buf(clk);
                    b.inst_named("sr", &sr, &[c1, c2])
                }
                _ => {
                    let nrst = b.not(rst);
                    let s = b.and(&[clk, nrst]);
                    b.inst_named("sr", &sr, &[s, clk])
                }
            };
            let x = b.xor(&[o[0], o[1]]);
            let nclk = b.not(clk);
            let q = b.inst_named("f", &ff, &[x, nclk])[0];
            b.output("Q", o[0]);
            b.output("X", x);
            b.output("F", q);
            let t = db.add(b.finish());
            db.build(&t)
        };
        let s = sweep(
            &format!("metastable release v{variant}"),
            (8..=64).step_by(8),
            make,
            from_reset,
            100,
        );
        assert_sound("meta", &s);
    }
}

#[test]
fn kernel_with_more_than_32_inputs() {
    // A Sim kernel reading 40 distinct nets (bit masks are 32 bits wide).
    let flat = flat_of(&wide_ring(64, 41));
    let r = sta(&flat);
    dump("wide ring 41", &r);
    assert!(!r.is_clean());
}

#[test]
fn sim_kernel_latch_is_proven_to_settle() {
    // A D latch whose core NANDs read 12 extra enables (top inputs): 14 kernel inputs make a
    // Sim kernel; STA proves every transition settles by enumeration over a feedback cut.
    let mut db = DesignBuilder::new();
    let tff = toggle_ff(&mut db);
    let p = 128;
    let (mut b, clk, rst) = top("wide_latch", p);
    let d = toggler(&mut b, &tff, "a", clk, rst);
    let nclk = b.not(clk);
    let es: Vec<Sig> = (0..12).map(|i| b.input(&format!("E{i}"), 1)).collect();
    let nd = b.not(d);
    let s = b.nand(&[d, nclk]);
    let r = b.nand(&[nd, nclk]);
    let qn = b.wire(1);
    let mut q_in = vec![s, qn];
    q_in.extend(&es[..6]);
    let q = b.nand(&q_in);
    let mut qn_in = vec![r, q];
    qn_in.extend(&es[6..]);
    b.nand_to(&qn_in, qn);
    b.output("Q", q);
    b.output("D", d);
    let t = db.add(b.finish());
    let flat = flat_of(&db.build(&t));
    let (ks, _) = panda_compile::compile_kernels(&flat, &Default::default());
    assert!(
        ks.kernels
            .iter()
            .any(|k| k.kind == panda_compile::KernelKind::Sim)
    );
    let r = sta(&flat);
    dump("wide latch", &r);
    assert_eq!(count(&r, panda_netlist::DiagKind::Oscillating), 0);
    let inputs: Vec<(String, panda_netlist::Value)> = (0..12)
        .map(|i| (format!("E{i}"), panda_netlist::Value::from_bool(true)))
        .collect();
    let inputs: Vec<(&str, panda_netlist::Value)> = inputs
        .iter()
        .map(|(s, v)| (s.as_str(), v.clone()))
        .collect();
    assert!(r.is_clean(), "{:?}", r.errors().collect::<Vec<_>>());
    let diff = differential_with(&flat, 100, from_reset(p), &inputs);
    assert!(diff.is_ok(), "{diff:?}");
}

/// Two toggle flip-flops on clocks of periods `pa` and `pb` (phases 0 and `phb`).
fn two_clocks(pa: u32, pb: u32, phb: u32, reset: u32) -> Design {
    let mut db = DesignBuilder::new();
    let tff = toggle_ff(&mut db);
    let mut b = DefBuilder::new(&format!("two_clk_{pa}_{pb}_{phb}"));
    let ca = b.clock(pa, pa / 2, 0);
    b.name(ca, "ca");
    let cb = b.clock(pb, pb / 2, phb);
    b.name(cb, "cb");
    let rst = b.power_on_reset(reset);
    let one = b.lit(1, 1);
    let qa = b.inst_named("a", &tff, &[one, rst, ca])[0];
    let qb = b.inst_named("b", &tff, &[one, rst, cb])[0];
    b.output("QA", qa);
    b.output("QB", qb);
    let t = db.add(b.finish());
    db.build(&t)
}

#[test]
fn truncated_hyperperiod_is_conservative() {
    // Periods 200 and 202 (hyperperiod 20,200): edges of the two clocks drift by one tick per
    // half period. In the first 4096 ticks they stay ≥ 10 ticks apart, but around tick 5,000 they
    // come within one tick of each other.
    let flat = flat_of(&two_clocks(200, 202, 50, 330));
    let (ks, tables) = panda_compile::compile_kernels(&flat, &Default::default());
    let opts = panda_sta::StaOptions {
        max_hyperperiod: 4096,
        ..Default::default()
    };
    let r = panda_sta::analyze_with(&flat, &ks, &tables, &opts);
    dump("truncated two clocks", &r);
    let diff = differential(&flat, 400, 400);
    assert!(diff.is_err(), "the design really violates setup");
    assert!(!r.is_clean(), "STA clean over a truncated hyperperiod");
    assert!(count(&r, panda_netlist::DiagKind::SetupViolation) > 0);
    // The full analysis agrees.
    assert!(!sta(&flat).is_clean());
    // Far apart (periods 200 and 400, edges 50 ticks apart): clean even when truncated.
    let flat = flat_of(&two_clocks(200, 400, 50, 330));
    let (ks, tables) = panda_compile::compile_kernels(&flat, &Default::default());
    let opts = panda_sta::StaOptions {
        max_hyperperiod: 128,
        ..Default::default()
    };
    let r = panda_sta::analyze_with(&flat, &ks, &tables, &opts);
    dump("truncated far apart", &r);
    assert!(r.is_clean(), "{:?}", r.errors().collect::<Vec<_>>());
    assert!(differential(&flat, 300, 400).is_ok());
}

#[test]
fn two_clocks_with_crossing_data_sweep() {
    // Flip-flop a (clock ca) toggles; flip-flop b (clock cb) samples a.Q; flip-flop c (clock ca)
    // samples b.Q. Coprime-ish half periods and every phase offset.
    let mut clean = 0;
    for (pa, pb) in [
        (34u32, 46u32),
        (64, 96),
        (64, 128),
        (48, 144),
        (80, 160),
        (72, 120),
    ] {
        for phb in (0..pb).step_by(3) {
            let mut db = DesignBuilder::new();
            let tff = toggle_ff(&mut db);
            let ff = ms_dff(&mut db);
            let mut b = DefBuilder::new(&format!("xdom_{pa}_{pb}_{phb}"));
            let ca = b.clock(pa, pa / 2, 0);
            b.name(ca, "ca");
            let cb = b.clock(pb, pb / 2, phb);
            b.name(cb, "cb");
            let rst = b.power_on_reset(2 * pb + pa / 4 + 1);
            let nrst = b.not(rst);
            let one = b.lit(1, 1);
            let qa = b.inst_named("a", &tff, &[one, rst, ca])[0];
            let da = b.and(&[qa, nrst]);
            let qb = b.inst_named("b", &ff, &[da, cb])[0];
            let db_ = b.and(&[qb, nrst]);
            let qc = b.inst_named("c", &ff, &[db_, ca])[0];
            b.output("QA", qa);
            b.output("QB", qb);
            b.output("QC", qc);
            let t = db.add(b.finish());
            let flat = flat_of(&db.build(&t));
            let r = sta(&flat);
            let from = u64::from(3 * pb + pa / 4 + 1);
            let diff = differential(&flat, 400, from);
            if r.is_clean() {
                clean += 1;
                assert!(
                    diff.is_ok(),
                    "{pa}/{pb} phase {phb}: STA clean but {diff:?}"
                );
            }
        }
    }
    println!("{clean} clean two-clock configurations");
    assert!(clean > 0);
}

/// STA for Timing mode with ideal blocks.
fn sta_ideal(flat: &panda_netlist::flat::FlatNetlist) -> panda_sta::TimingReport {
    let (ks, tables) = panda_compile::compile_kernels(flat, &Default::default());
    let opts = panda_sta::StaOptions {
        ideal_blocks: true,
        ..Default::default()
    };
    panda_sta::analyze_with(flat, &ks, &tables, &opts)
}

fn ideal_timing() -> panda_timing::TimingOptions {
    panda_timing::TimingOptions::default().with_ideal_blocks(true)
}

/// A package with a 1-gate path (from `A`) and a 12-gate path (from `B`, tied to 0): its output
/// latency is 12 even when only `A` changes. A flip-flop toggles `A`; two others sample the output.
fn skewed_package_design(p: u32) -> Design {
    let mut db = DesignBuilder::new();
    let mut pk = DefBuilder::new("skewed");
    let a = pk.input("A", 1);
    let bb = pk.input("B", 1);
    let deep = bufs(&mut pk, bb, 11);
    let o = pk.xor(&[a, deep]);
    pk.output("O", o);
    let skewed = db.add(pk.finish());
    let tff = toggle_ff(&mut db);
    let ff = ms_dff(&mut db);
    let (mut b, clk, rst) = top(&format!("ideal_lat{p}"), p);
    let q = toggler(&mut b, &tff, "t", clk, rst);
    let zero = b.lit(1, 0);
    let o = b.inst_named("pkg", &skewed, &[q, zero])[0];
    let nclk = b.not(clk);
    let s = b.inst_named("s", &ff, &[o, nclk])[0];
    let s2 = b.inst_named("s2", &ff, &[o, nclk])[0];
    b.output("S", s);
    b.output("S2", s2);
    let t = db.add(b.finish());
    db.build(&t)
}

#[test]
fn ideal_block_latency_is_its_deepest_path() {
    // Glass-box STA sees the 1-gate path; ideal-block STA must use the 12-tick latency.
    let flat = flat_of(&skewed_package_design(28));
    let glass = sta(&flat);
    dump("skewed glass", &glass);
    let ideal = sta_ideal(&flat);
    dump("skewed ideal", &ideal);
    assert!(glass.is_clean(), "{:?}", glass.errors().collect::<Vec<_>>());
    assert!(differential(&flat, 100, from_reset(28)).is_ok());
    let id = differential_opts(&flat, 100, from_reset(28), &[], ideal_timing());
    assert!(id.is_err(), "the ideal block really takes 12 ticks");
    assert!(
        !ideal.is_clean(),
        "ideal-block STA must see the 12-tick latency"
    );
    let rise = |r: &panda_sta::TimingReport| {
        r.phase_settle
            .iter()
            .find(|x| x.event == "clk↑")
            .and_then(|x| x.settle)
            .unwrap()
    };
    assert!(rise(&ideal) >= rise(&glass) + 11);
    // At the ideal-block minimum period, ideal-block STA is clean and both engines agree.
    let pmin = ideal.min_period_of("clk").unwrap().next_multiple_of(4);
    let flat = flat_of(&skewed_package_design(pmin));
    let ideal = sta_ideal(&flat);
    assert!(
        ideal.is_clean(),
        "{pmin}: {:?}",
        ideal.errors().collect::<Vec<_>>()
    );
    let id = differential_opts(&flat, 100, from_reset(pmin), &[], ideal_timing());
    assert!(id.is_ok(), "{id:?}");
}

#[test]
fn ideal_block_absorbed_into_a_kernel_is_flagged() {
    // A package that glitches at gate level (XNOR(a, delay(a)) pulses low when `a` changes) feeds
    // only the set input of an SR latch, so kernel extraction absorbs it. Fast mode's transition
    // simulates the glitch (the latch is set); Timing mode with ideal blocks has none.
    let mut db = DesignBuilder::new();
    let mut pk = DefBuilder::new("pulser");
    let a = pk.input("A", 1);
    let d = bufs(&mut pk, a, 2);
    let o = pk.xnor(&[a, d]);
    pk.output("O", o);
    let pulser = db.add(pk.finish());
    let tff = toggle_ff(&mut db);
    let sr = nand_sr_latch(&mut db);
    let p = 32;
    let (mut b, clk, rst) = top("ideal_absorb", p);
    let q = toggler(&mut b, &tff, "t", clk, rst);
    let s_n = b.inst_named("pls", &pulser, &[q])[0];
    let r_n = b.not(rst);
    let o = b.inst_named("sr", &sr, &[s_n, r_n]);
    b.output("Q", o[0]);
    b.output("T", q);
    let t = db.add(b.finish());
    let flat = flat_of(&db.build(&t));
    let ideal = sta_ideal(&flat);
    dump("absorbed pulser (ideal)", &ideal);
    let id = differential_opts(&flat, 100, from_reset(p), &[], ideal_timing());
    println!("ideal-mode comparison: {id:?}");
    assert!(
        id.is_err(),
        "FastRef simulates the absorbed glitch, ideal Timing does not"
    );
    assert!(
        !ideal.is_clean(),
        "the absorbed block's glitch decides the latch state"
    );
    let glass = sta(&flat);
    if glass.is_clean() {
        assert!(differential(&flat, 100, from_reset(p)).is_ok());
    }
}

#[test]
#[ignore = "pessimism report: run with --ignored --nocapture"]
fn counters_min_period_pessimism_report() {
    use panda_circuits::{ripple_counter, sync_counter, testbench};
    let en = [("EN", panda_netlist::Value::from_bool(true))];
    type Make = fn(&mut DesignBuilder, u32) -> panda_netlist::DefRef;
    let cases: [(&str, Make, u32, bool); 5] = [
        ("sync_counter", sync_counter, 4, true),
        ("sync_counter", sync_counter, 8, true),
        ("sync_counter", sync_counter, 16, true),
        ("ripple_counter", ripple_counter, 4, false),
        ("ripple_counter", ripple_counter, 8, false),
    ];
    for (name, make, n, has_en) in cases {
        let build = |p: u32| {
            let mut db = DesignBuilder::new();
            let c = make(&mut db, n);
            let tb = testbench(&mut db, &c, p);
            flat_of(&db.build(&tb))
        };
        let inputs: &[(&str, panda_netlist::Value)] = if has_en { &en } else { &[] };
        let mut sta_min = sta(&build(256))
            .min_period_of("clk")
            .unwrap()
            .max(8)
            .next_multiple_of(4);
        // The testbench releases the reset a quarter period after a falling edge, which can need
        // a longer period than the clock's own events (`min_period` does not cover it).
        let reported = sta_min;
        while !sta(&build(sta_min)).is_clean() {
            sta_min += 4;
        }
        let ok = |p: u32| {
            let f = build(p);
            let from = u64::from(panda_circuits::TestbenchOptions::new(p).reset_ticks);
            differential_with(&f, 600, from, inputs).is_ok()
        };
        let mut emp = sta_min;
        while emp > 8 && ok(emp - 4) {
            emp -= 4;
        }
        println!(
            "{name}({n}): STA min period {reported} (clean with the testbench reset from \
             {sta_min}), FastRef = Timing down to {emp} (step 4), pessimism {:.0}%",
            100.0 * (f64::from(sta_min) / f64::from(emp) - 1.0)
        );
    }
}

/// xorshift64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn chance(&mut self, pct: u64) -> bool {
        self.next() % 100 < pct
    }
}

/// A random design centered on clock structure: clock nets built from `clk`, its delayed copies,
/// a second clock and state (gated, XOR-ed, pulse-shaped, divided), driving flip-flops, latches
/// and SR latches whose data come from short and long paths of other state elements.
fn clocky_design(seed: u64, p: u32) -> Design {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut db = DesignBuilder::new();
    let ff = ms_dff(&mut db);
    let latch = d_latch(&mut db);
    let sr = nand_sr_latch(&mut db);
    let (mut b, clk, rst) = top(&format!("clocky{seed}_{p}"), p);
    let nrst = b.not(rst);
    let clk2 = if rng.chance(30) {
        let p2 = p * (2 + rng.below(3) as u32);
        let c = b.clock(p2, p2 / 2, (p / 4) * rng.below(4) as u32);
        b.name(c, "clk2");
        Some(c)
    } else {
        None
    };
    let nstate = 2 + rng.below(5);
    let qs: Vec<Sig> = (0..nstate).map(|_| b.wire(1)).collect();
    // Clock candidates.
    let mut clocks = vec![clk];
    let mut data: Vec<Sig> = qs.clone();
    for _ in 0..1 + rng.below(4) {
        let base = clocks[rng.below(clocks.len())];
        let c = match rng.below(7) {
            0 => {
                let k = rng.below(4);
                bufs(&mut b, base, k + 1)
            }
            1 => b.not(base),
            2 => {
                let en = data[rng.below(data.len())];
                let en = b.or(&[en, rst]);
                b.and(&[base, en])
            }
            3 => {
                let k = 1 + rng.below(3);
                let d = bufs(&mut b, base, k);
                b.xor(&[base, d])
            }
            4 => {
                let k = 1 + rng.below(4);
                let d = bufs(&mut b, base, k);
                let nd = b.not(d);
                b.and(&[base, nd])
            }
            5 => clk2.unwrap_or(base),
            _ => {
                let k = 1 + rng.below(4);
                let d = bufs(&mut b, base, k);
                b.or(&[base, d])
            }
        };
        clocks.push(c);
    }
    // Random logic over the state.
    for _ in 0..rng.below(8) {
        let a = data[rng.below(data.len())];
        let c = data[rng.below(data.len())];
        let g = match rng.below(4) {
            0 => b.xor(&[a, c]),
            1 => b.nand(&[a, c]),
            2 => {
                let k = rng.below(6);
                bufs(&mut b, a, k + 1)
            }
            _ => b.not(a),
        };
        data.push(g);
    }
    for (i, &q) in qs.iter().enumerate() {
        let d0 = data[rng.below(data.len())];
        let d = b.and(&[d0, nrst]);
        let ck = clocks[rng.below(clocks.len())];
        let out = match rng.below(10) {
            0..=4 => b.inst_named(&format!("f{i}"), &ff, &[d, ck])[0],
            5..=7 => b.inst_named(&format!("l{i}"), &latch, &[d, ck])[0],
            _ => {
                // SR latch: set by a NAND of data and clock, reset by the reset.
                let s_n = b.nand(&[d, ck]);
                b.inst_named(&format!("s{i}"), &sr, &[s_n, nrst])[0]
            }
        };
        b.connect(out, q);
        b.output(&format!("Q{i}"), q);
    }
    let t = db.add(b.finish());
    db.build(&t)
}

fn clocky_run(seeds: std::ops::Range<u64>) -> (usize, usize, Vec<(u64, u32, String)>) {
    let (mut n, mut clean) = (0, 0);
    let mut unsound = Vec::new();
    for seed in seeds {
        for p in [16u32, 32, 64] {
            let Ok(flat) = panda_netlist::flatten(&clocky_design(seed, p)) else {
                continue;
            };
            n += 1;
            let r = sta(&flat);
            if !r.is_clean() {
                continue;
            }
            clean += 1;
            if let Err(m) = differential(&flat, 120, from_reset(p)) {
                unsound.push((seed, p, format!("{m:?}")));
            }
        }
    }
    (n, clean, unsound)
}

#[test]
fn clocky_random_designs_are_sound() {
    let (n, clean, unsound) = clocky_run(0..300);
    println!("{n} designs, {clean} clean");
    assert!(
        unsound.is_empty(),
        "STA clean but FastRef != Timing: {unsound:?}"
    );
    assert!(clean >= 20, "only {clean} clean designs");
}

#[test]
#[ignore = "long: run with --ignored"]
fn clocky_random_designs_are_sound_long() {
    let (n, clean, unsound) = clocky_run(300..6000);
    println!("{n} designs, {clean} clean");
    assert!(
        unsound.is_empty(),
        "STA clean but FastRef != Timing: {unsound:?}"
    );
}

#[test]
fn ui_inputs_into_clock_gates_and_latches() {
    // A UI input gates the clock of a flip-flop (gclk = clk · EN changes in mid-phase), feeds an
    // SR latch through a reconvergent (hazardous) path, and a transparent latch directly.
    for variant in 0..3 {
        for p in [16u32, 32, 64, 128] {
            let mut db = DesignBuilder::new();
            let tff = toggle_ff(&mut db);
            let ff = ms_dff(&mut db);
            let l = d_latch(&mut db);
            let sr = nand_sr_latch(&mut db);
            let (mut b, clk, rst) = top(&format!("ui{variant}_{p}"), p);
            let en = b.input("EN", 1);
            let din = b.input("D", 1);
            let a = toggler(&mut b, &tff, "a", clk, rst);
            match variant {
                0 => {
                    let gclk = b.and(&[clk, en]);
                    let q = b.inst_named("g", &ff, &[a, gclk])[0];
                    b.output("Q", q);
                }
                1 => {
                    let nd = b.not(din);
                    let dd = bufs(&mut b, nd, 2);
                    let s_n = b.nand(&[din, dd, en]);
                    let r_n = b.not(rst);
                    let q = b.inst_named("sr", &sr, &[s_n, r_n])[0];
                    b.output("Q", q);
                }
                _ => {
                    let x = b.xor(&[din, a]);
                    let q = b.inst_named("l", &l, &[x, clk])[0];
                    let q2 = b.inst_named("l2", &l, &[en, q])[0];
                    b.output("Q", q);
                    b.output("Q2", q2);
                }
            }
            let t = db.add(b.finish());
            let flat = flat_of(&db.build(&t));
            let r = sta(&flat);
            for seed in 1..6u64 {
                let diff = differential_inputs(&flat, &r, 200, from_reset(p), &["EN", "D"], seed);
                if r.is_clean() {
                    assert!(
                        diff.is_ok(),
                        "variant {variant} @ {p} seed {seed}: STA clean but {diff:?}"
                    );
                }
            }
            println!("ui variant {variant} @ {p}: clean {}", r.is_clean());
        }
    }
}
