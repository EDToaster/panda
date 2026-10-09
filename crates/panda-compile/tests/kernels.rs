//! Kernel extraction and characterization on the golden circuits (SPEC §8.3, §8.4).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use panda_circuits::{d_latch, ms_dff, nand_sr_latch, register, ring_oscillator};
use panda_compile::charz::{
    CharzOptions, KernelSim, KernelTable, TableCache, characterize, input_vector,
};
use panda_compile::kernels::{
    ExtractOptions, Kernel, KernelKind, KernelSet, extract, extract_with_stats,
};
use panda_compile::{KernelOptions, Node, compile_kernels};
use panda_cpu8::{Options, RomStyle, Variant, panda8_with, programs};
use panda_netlist::flat::{
    FlatGate, FlatNetlist, FlatOp, FlatPin, NetIdx, Resolver, Source, SourceKind,
};
use panda_netlist::{DefBuilder, DefRef, Design, DesignBuilder, Dir, Logic, Termination, flatten};
use panda_timing::{TimingOptions, TimingSim};

use Logic::{One as I, Zero as O};

fn flat_of(db: DesignBuilder, top: &DefRef) -> FlatNetlist {
    flatten(&db.build(top)).expect("flatten")
}

fn flat_design(d: &Design) -> FlatNetlist {
    flatten(d).expect("flatten")
}

fn net(flat: &FlatNetlist, name: &str) -> NetIdx {
    flat.net_by_name(name)
        .unwrap_or_else(|| panic!("no net {name}"))
}

fn gate_driving(flat: &FlatNetlist, n: NetIdx) -> u32 {
    flat.gates.iter().position(|g| g.output == n).expect("gate") as u32
}

fn pos(v: &[NetIdx], n: NetIdx) -> usize {
    v.iter()
        .position(|&x| x == n)
        .unwrap_or_else(|| panic!("net {n} not in {v:?}"))
}

/// Input vector index from (position, value) pairs.
fn vec_idx(pairs: &[(usize, bool)]) -> u32 {
    pairs.iter().map(|&(p, b)| u32::from(b) << p).sum()
}

fn state_id(t: &KernelTable, k: &Kernel, vals: &[(NetIdx, Logic)]) -> Vec<u32> {
    (0..t.num_states() as u32)
        .filter(|&s| {
            vals.iter()
                .all(|&(n, v)| t.states[s as usize][pos(&k.nets, n)] == v)
        })
        .collect()
}

fn check(ks: &KernelSet, flat: &FlatNetlist) {
    ks.verify(flat)
        .unwrap_or_else(|e| panic!("invalid kernel set: {e}"));
}

// ---------------------------------------------------------------------------------------------
// Extraction shapes
// ---------------------------------------------------------------------------------------------

#[test]
fn nand_sr_latch_is_one_kernel_of_two_gates() {
    let mut db = DesignBuilder::new();
    let top = nand_sr_latch(&mut db);
    let flat = flat_of(db, &top);
    let ks = extract(&flat, &ExtractOptions::default());
    check(&ks, &flat);
    assert_eq!(ks.kernels.len(), 1);
    let k = &ks.kernels[0];
    assert_eq!(k.gates.len(), 2);
    assert_eq!(k.kind, KernelKind::Table);
    assert_eq!(k.inputs, {
        let mut v = vec![net(&flat, "S_n"), net(&flat, "R_n")];
        v.sort();
        v
    });
    assert_eq!(k.cut.len(), 1);
    assert!(ks.comb_order.is_empty());
}

#[test]
fn d_latch_is_one_kernel_with_steering_and_inverter() {
    let mut db = DesignBuilder::new();
    let top = d_latch(&mut db);
    let flat = flat_of(db, &top);
    let ks = extract(&flat, &ExtractOptions::default());
    check(&ks, &flat);
    assert_eq!(ks.kernels.len(), 1);
    let k = &ks.kernels[0];
    assert_eq!(k.gates.len(), 5, "q, qn, s, r and NOT D");
    assert_eq!(k.gates.len(), flat.gates.len());
    let mut ins = vec![net(&flat, "D"), net(&flat, "EN")];
    ins.sort();
    assert_eq!(k.inputs, ins);
    assert_eq!(k.bits(), 3);
    assert_eq!(k.kind, KernelKind::Table);
}

#[test]
fn ms_dff_is_master_and_slave_with_clock_inverter_absorbed() {
    let mut db = DesignBuilder::new();
    let top = ms_dff(&mut db);
    let flat = flat_of(db, &top);
    let ks = extract(&flat, &ExtractOptions::default());
    check(&ks, &flat);
    assert_eq!(ks.kernels.len(), 2);
    let nclk_gate = gate_driving(&flat, net(&flat, "nclk"));
    let master_q = gate_driving(&flat, net(&flat, "master/q"));
    let slave_q = gate_driving(&flat, net(&flat, "slave/q"));
    let km = ks.kernel_of(Node::Gate(master_q)).unwrap();
    let kslave = ks.kernel_of(Node::Gate(slave_q)).unwrap();
    assert_ne!(km, kslave);
    assert_eq!(
        ks.kernel_of(Node::Gate(nclk_gate)),
        Some(km),
        "nclk is private to the master"
    );
    assert_eq!(ks.kernels[km as usize].gates.len(), 6);
    assert_eq!(ks.kernels[kslave as usize].gates.len(), 5);
    assert!(ks.comb_order.is_empty(), "every gate is in a kernel");
    // master: D, CLK; slave: m (master's q), CLK
    let mut mi = vec![net(&flat, "D"), net(&flat, "CLK")];
    mi.sort();
    assert_eq!(ks.kernels[km as usize].inputs, mi);
    let mut si = vec![net(&flat, "master/q"), net(&flat, "CLK")];
    si.sort();
    assert_eq!(ks.kernels[kslave as usize].inputs, si);
    assert!(
        ks.kernels[km as usize]
            .outputs
            .contains(&net(&flat, "master/q"))
    );
}

#[test]
fn shared_clock_inverter_stays_combinational() {
    // Two d_latches enabled by the same inverted clock: the inverter is shared → comb.
    let mut db = DesignBuilder::new();
    let latch = d_latch(&mut db);
    let mut b = DefBuilder::new("two_latches");
    let d0 = b.input("D0", 1);
    let d1 = b.input("D1", 1);
    let clk = b.input("CLK", 1);
    let nclk = b.not(clk);
    b.name(nclk, "nclk");
    let q0 = b.inst_named("l0", &latch, &[d0, nclk]);
    let q1 = b.inst_named("l1", &latch, &[d1, nclk]);
    b.output("Q0", q0[0]);
    b.output("Q1", q1[0]);
    let top = db.add(b.finish());
    let flat = flat_of(db, &top);
    let ks = extract(&flat, &ExtractOptions::default());
    check(&ks, &flat);
    assert_eq!(ks.kernels.len(), 2);
    let g = gate_driving(&flat, net(&flat, "nclk"));
    assert_eq!(ks.kernel_of(Node::Gate(g)), None);
    assert_eq!(ks.comb_order, vec![Node::Gate(g)]);
}

#[test]
fn register8_is_sixteen_latch_kernels_with_cache_hits() {
    let mut db = DesignBuilder::new();
    let top = register(&mut db, 8);
    let flat = flat_of(db, &top);
    let ks = extract(&flat, &ExtractOptions::default());
    check(&ks, &flat);
    assert_eq!(ks.kernels.len(), 16);
    assert!(
        ks.kernels
            .iter()
            .all(|k| k.kind == KernelKind::Table && k.bits() <= 12)
    );
    let mut cache = TableCache::new();
    let opts = CharzOptions::default();
    for k in &ks.kernels {
        let t = cache
            .get_or_characterize(&flat, k, &opts)
            .expect("table kernel");
        assert!(t.num_states() >= 2);
    }
    println!(
        "register(8): {} misses, {} hits, sizes {:?}",
        cache.misses(),
        cache.hits(),
        ks.size_histogram()
    );
    assert_eq!(cache.misses() + cache.hits(), 16);
    assert_eq!(cache.misses(), 2, "one master table and one slave table");
    assert_eq!(cache.hits(), 14);
}

#[test]
fn ring_oscillator_is_one_kernel_without_stable_state() {
    let mut db = DesignBuilder::new();
    let top = ring_oscillator(&mut db, 3, false);
    let flat = flat_of(db, &top);
    let ks = extract(&flat, &ExtractOptions::default());
    check(&ks, &flat);
    assert_eq!(ks.kernels.len(), 1);
    let k = &ks.kernels[0];
    assert_eq!(k.gates.len(), 3);
    assert_eq!(k.kind, KernelKind::Table);
    let t = characterize(&flat, k, &CharzOptions::default()).unwrap();
    assert_eq!(t.num_states(), 0);
    assert!(t.is_oscillator());
    // On the fly: from a binary start it never settles.
    let mut sim = KernelSim::new(&flat, k, &CharzOptions::default());
    sim.load_state(&[O, I, O], &[]);
    let s = sim.settle(200);
    assert!(!s.quiescent);
    // Enabled variant: EN = 0 has a stable state, EN = 1 has none.
    let mut db = DesignBuilder::new();
    let top = ring_oscillator(&mut db, 5, true);
    let flat = flat_of(db, &top);
    let ks = extract(&flat, &ExtractOptions::default());
    assert_eq!(ks.kernels.len(), 1);
    let t = characterize(&flat, &ks.kernels[0], &CharzOptions::default()).unwrap();
    assert_eq!(t.states_by_input[0].len(), 1);
    assert!(t.states_by_input[1].is_empty());
    let s0 = t.states_by_input[0][0];
    assert_eq!(t.transition(s0, 1).next, None, "enabling oscillates");
}

#[test]
fn self_loop_not_gate_is_a_kernel() {
    let mut db = DesignBuilder::new();
    let mut b = DefBuilder::new("selfloop");
    let w = b.wire(1);
    b.not_to(w, w);
    b.output("OUT", w);
    let top = db.add(b.finish());
    let flat = flat_of(db, &top);
    let ks = extract(&flat, &ExtractOptions::default());
    check(&ks, &flat);
    assert_eq!(ks.kernels.len(), 1);
    let t = characterize(&flat, &ks.kernels[0], &CharzOptions::default()).unwrap();
    assert!(t.is_oscillator());
}

/// `a = s ? x : b`, `b = s ? a : y` — cyclic but combinational (Malik).
fn cyclic_comb() -> FlatNetlist {
    let mut db = DesignBuilder::new();
    let mut bld = DefBuilder::new("cyc");
    let x = bld.input("X", 1);
    let y = bld.input("Y", 1);
    let s = bld.input("S", 1);
    let ns = bld.not(s);
    let a = bld.wire(1);
    let b = bld.wire(1);
    let a1 = bld.and(&[x, s]);
    let a2 = bld.and(&[b, ns]);
    bld.gate_to(panda_netlist::GateOp::Or, &[a1, a2], a);
    let b1 = bld.and(&[a, s]);
    let b2 = bld.and(&[y, ns]);
    bld.gate_to(panda_netlist::GateOp::Or, &[b1, b2], b);
    bld.name(a, "a");
    bld.name(b, "b");
    bld.output("A", a);
    bld.output("B", b);
    let top = db.add(bld.finish());
    flat_of(db, &top)
}

#[test]
fn cyclic_combinational_toy() {
    let flat = cyclic_comb();
    let ks = extract(&flat, &ExtractOptions::default());
    check(&ks, &flat);
    assert_eq!(ks.kernels.len(), 1);
    let k = &ks.kernels[0];
    assert!(matches!(k.kind, KernelKind::Table | KernelKind::Sim));
    assert_eq!(
        k.gates.len(),
        flat.gates.len(),
        "the whole block, including NOT S"
    );
    let t = characterize(&flat, k, &CharzOptions::default()).unwrap();
    let (px, py, ps) = (
        pos(&k.inputs, net(&flat, "X")),
        pos(&k.inputs, net(&flat, "Y")),
        pos(&k.inputs, net(&flat, "S")),
    );
    let (pa, pb) = (pos(&k.nets, net(&flat, "a")), pos(&k.nets, net(&flat, "b")));
    for u in 0..8u32 {
        let (x, y, s) = (u >> px & 1 == 1, u >> py & 1 == 1, u >> ps & 1 == 1);
        let list = &t.states_by_input[u as usize];
        assert_eq!(
            list.len(),
            1,
            "combinational: unique stable state for {u:03b}"
        );
        let st = &t.states[list[0] as usize];
        let want = if s { x } else { y };
        assert_eq!(st[pa], Logic::from_bool(want));
        assert_eq!(st[pb], Logic::from_bool(want));
        // Every state transitions to the unique state of the new input.
        for st_id in 0..t.num_states() as u32 {
            assert_eq!(t.transition(st_id, u).next, Some(list[0]));
        }
    }
    assert!(!t.is_oscillator());
}

#[test]
fn small_max_bits_makes_sim_kernels() {
    let flat = cyclic_comb();
    let opts = ExtractOptions {
        max_bits: 2,
        ..ExtractOptions::default()
    };
    let ks = extract(&flat, &opts);
    check(&ks, &flat);
    assert!(ks.kernels.iter().all(|k| k.kind == KernelKind::Sim));
    assert!(characterize(&flat, &ks.kernels[0], &CharzOptions::default()).is_none());
    // On-the-fly transitions still work: S=1, X=1 → a = b = 1.
    let k = &ks.kernels[0];
    let mut sim = KernelSim::new(&flat, k, &CharzOptions::default());
    let ins: Vec<Logic> = k
        .inputs
        .iter()
        .map(|&n| {
            if n == net(&flat, "S") || n == net(&flat, "X") {
                I
            } else {
                O
            }
        })
        .collect();
    let xs = vec![Logic::X; k.nets.len()];
    let (st, s) = sim.transition(&xs, &vec![Logic::X; k.inputs.len()], &ins);
    assert!(s.quiescent);
    let a = k
        .nets
        .iter()
        .position(|&n| Some(n) == flat.net_by_name("a"));
    if let Some(a) = a {
        assert_eq!(st[a], I);
    }
}

#[test]
fn seed_cap_falls_back_to_sim_kernels() {
    let mut db = DesignBuilder::new();
    let top = register(&mut db, 2);
    let flat = flat_of(db, &top);
    let opts = ExtractOptions {
        max_cycles_per_region: 0,
        ..ExtractOptions::default()
    };
    let (ks, stats) = extract_with_stats(&flat, &opts);
    check(&ks, &flat);
    assert!(stats.regions_skipped > 0);
    assert!(stats.residual_kernels > 0);
    // Every loop is still covered by some kernel.
    assert!(!ks.kernels.is_empty());
}

#[test]
fn keeper_resolver_becomes_keeper_kernel() {
    let mut db = DesignBuilder::new();
    let mut b = DefBuilder::new("keeper_bus");
    let d0 = b.input("D0", 1);
    let e0 = b.input("E0", 1);
    let d1 = b.input("D1", 1);
    let e1 = b.input("E1", 1);
    let bus = b.wire(1);
    b.tri_to(d0, e0, bus);
    b.tri_to(d1, e1, bus);
    b.terminate(bus, Termination::Keeper);
    b.name(bus, "bus");
    let q = b.buf(bus);
    b.output("Q", q);
    let top = db.add(b.finish());
    let flat = flat_of(db, &top);
    assert_eq!(flat.resolvers.len(), 1);
    let ks = extract(&flat, &ExtractOptions::default());
    check(&ks, &flat);
    assert_eq!(ks.kernels.len(), 1);
    let k = &ks.kernels[0];
    assert_eq!(k.kind, KernelKind::Keeper);
    assert_eq!(k.resolvers, vec![0]);
    assert!(characterize(&flat, k, &CharzOptions::default()).is_none());
    // On the fly: drive 1, then release → holds 1.
    let mut sim = KernelSim::new(&flat, k, &CharzOptions::default());
    sim.load_state(&[Logic::X], &[Logic::Z, Logic::Z]);
    sim.apply_inputs(&[I, Logic::Z]);
    assert!(sim.settle(10).quiescent);
    assert_eq!(sim.read_state(), vec![I]);
    sim.apply_inputs(&[Logic::Z, Logic::Z]);
    assert!(sim.settle(10).quiescent);
    assert_eq!(sim.read_state(), vec![I]);
}

// ---------------------------------------------------------------------------------------------
// Characterization
// ---------------------------------------------------------------------------------------------

#[test]
fn sr_latch_table() {
    let mut db = DesignBuilder::new();
    let top = nand_sr_latch(&mut db);
    let flat = flat_of(db, &top);
    let ks = extract(&flat, &ExtractOptions::default());
    let k = &ks.kernels[0];
    let t = characterize(&flat, k, &CharzOptions::default()).unwrap();
    let (q, qn) = (net(&flat, "q"), net(&flat, "qn"));
    let (ps, pr) = (
        pos(&k.inputs, net(&flat, "S_n")),
        pos(&k.inputs, net(&flat, "R_n")),
    );
    assert_eq!(t.num_states(), 3, "Q=1/Q_n=0, Q=0/Q_n=1, Q=Q_n=1");
    let set = state_id(&t, k, &[(q, I), (qn, O)])[0];
    let reset = state_id(&t, k, &[(q, O), (qn, I)])[0];
    let both = state_id(&t, k, &[(q, I), (qn, I)])[0];
    let hold = vec_idx(&[(ps, true), (pr, true)]);
    let s_only = vec_idx(&[(ps, false), (pr, true)]);
    let r_only = vec_idx(&[(ps, true), (pr, false)]);
    let none = vec_idx(&[]);
    assert_eq!(t.states_by_input[hold as usize], {
        let mut v = vec![set, reset];
        v.sort();
        v
    });
    assert_eq!(t.states_by_input[none as usize], vec![both]);
    assert!(!t.is_oscillator());
    // hold
    assert_eq!(t.transition(set, hold).next, Some(set));
    assert_eq!(t.transition(reset, hold).next, Some(reset));
    assert_eq!(t.transition(set, hold).latency, 0);
    // set / reset
    assert_eq!(t.transition(reset, s_only).next, Some(set));
    assert_eq!(t.transition(set, r_only).next, Some(reset));
    assert_eq!(
        t.transition(reset, s_only).latency,
        2,
        "Q rises at +1, Q_n falls at +2"
    );
    // both active
    assert_eq!(t.transition(set, none).next, Some(both));
    // simultaneous release: metastable → period-2 collapse → X → None
    let rel = t.transition(both, hold);
    assert_eq!(rel.next, None);
    assert!(
        rel.order_sensitive,
        "releasing S first or R first gives different states"
    );
    // releasing one at a time is fine and order-insensitive (single change)
    assert_eq!(t.transition(both, s_only).next, Some(set));
    assert!(!t.transition(both, s_only).order_sensitive);
    // set + reset asserted together from a stored state: both orders end in Q=Q_n=1
    assert_eq!(t.transition(set, none).next, Some(both));
    assert!(!t.transition(set, none).order_sensitive);
    // From `set`, swapping S_n=0,R_n=1 → S_n=1,R_n=0 (state `set` is stable under s_only):
    // simultaneous gives reset; S first: hold then reset → reset; R first: both then release S →
    // reset. Order-insensitive.
    assert_eq!(t.transition(set, r_only).next, Some(reset));
    assert!(!t.transition(set, r_only).order_sensitive);
}

#[test]
fn d_latch_table_transparent_and_opaque() {
    let mut db = DesignBuilder::new();
    let top = d_latch(&mut db);
    let flat = flat_of(db, &top);
    let ks = extract(&flat, &ExtractOptions::default());
    let k = &ks.kernels[0];
    let t = characterize(&flat, k, &CharzOptions::default()).unwrap();
    assert_eq!(
        t.num_states(),
        6,
        "EN=0: 2 per D value; EN=1: 1 per D value"
    );
    let (q, qn) = (
        pos(&k.nets, net(&flat, "q")),
        pos(&k.nets, net(&flat, "qn")),
    );
    let (pd, pe) = (
        pos(&k.inputs, net(&flat, "D")),
        pos(&k.inputs, net(&flat, "EN")),
    );
    let mut sensitive = 0;
    for s in 0..t.num_states() as u32 {
        let st = &t.states[s as usize];
        assert_ne!(st[q], st[qn], "no stable state with Q = Q_n");
        for d in [false, true] {
            for en in [false, true] {
                let v = vec_idx(&[(pd, d), (pe, en)]);
                let tr = t.transition(s, v);
                sensitive += usize::from(tr.order_sensitive);
                let Some(n) = tr.next else {
                    // Only a simultaneous D/EN race may be non-binary.
                    continue;
                };
                let nst = &t.states[n as usize];
                if en {
                    assert_eq!(nst[q], Logic::from_bool(d), "transparent: Q follows D");
                } else if !tr.order_sensitive {
                    assert_eq!(nst[q], st[q], "opaque: Q holds (state {s}, D={d})");
                }
                assert_ne!(nst[q], nst[qn]);
            }
        }
    }
    println!(
        "d_latch: {} states, {sensitive} order-sensitive entries",
        t.num_states()
    );
    // Hand-checked entries.
    let id_of = |dv: bool, env: bool, qv: Logic| -> u32 {
        let u = vec_idx(&[(pd, dv), (pe, env)]);
        *t.states_by_input[u as usize]
            .iter()
            .find(|&&s| t.states[s as usize][q] == qv)
            .expect("state")
    };
    // Holding 1 with EN=0, D=0; open → Q becomes 0 after NAND r (1) + qn (2) + q (3).
    let s = id_of(false, false, I);
    let tr = t.transition(s, vec_idx(&[(pd, false), (pe, true)]));
    assert_eq!(t.states[tr.next.unwrap() as usize][q], O);
    assert_eq!(tr.latency, 3);
    assert!(!tr.order_sensitive);
    // Opaque latch ignores D: holding 0 with D=0, D→1 (EN stays 0): Q stays 0, only NOT D changes.
    let s = id_of(false, false, O);
    let tr = t.transition(s, vec_idx(&[(pd, true), (pe, false)]));
    let n = tr.next.unwrap();
    assert_eq!(t.states[n as usize][q], O);
    assert_eq!(n, id_of(true, false, O));
    assert_eq!(tr.latency, 1);
    // Transparent latch tracks D: D=1,EN=1 → D=0.
    let s = id_of(true, true, I);
    let tr = t.transition(s, vec_idx(&[(pd, false), (pe, true)]));
    assert_eq!(tr.next, Some(id_of(false, true, O)));
    // Closing while D stays: holds.
    let tr = t.transition(s, vec_idx(&[(pd, true), (pe, false)]));
    assert_eq!(tr.next, Some(id_of(true, false, I)));
}

#[test]
fn ms_dff_master_and_slave_tables() {
    let mut db = DesignBuilder::new();
    let top = ms_dff(&mut db);
    let flat = flat_of(db, &top);
    let (ks, tables) = compile_kernels(&flat, &KernelOptions::default());
    let mq = net(&flat, "master/q");
    let sq = net(&flat, "slave/q");
    let clk = net(&flat, "CLK");
    let km = ks.kernel_of(Node::Gate(gate_driving(&flat, mq))).unwrap() as usize;
    let kslave = ks.kernel_of(Node::Gate(gate_driving(&flat, sq))).unwrap() as usize;
    let (m, s) = (&ks.kernels[km], &ks.kernels[kslave]);
    let (tm, ts) = (
        tables[km].as_ref().unwrap(),
        tables[kslave].as_ref().unwrap(),
    );
    // Master: transparent while CLK = 0, opaque while CLK = 1.
    let (pd, pc) = (pos(&m.inputs, net(&flat, "D")), pos(&m.inputs, clk));
    let qm = pos(&m.nets, mq);
    for st in 0..tm.num_states() as u32 {
        for d in [false, true] {
            let tr = tm.transition(st, vec_idx(&[(pd, d), (pc, false)]));
            if let Some(n) = tr.next {
                assert_eq!(tm.states[n as usize][qm], Logic::from_bool(d));
            }
            // Opaque master (state reached with CLK = 1): any D with CLK = 1 holds Q.
            let home = tm.home_input[st as usize];
            if home >> pc & 1 == 1 {
                let tr = tm.transition(st, vec_idx(&[(pd, d), (pc, true)]));
                let n = tr.next.expect("opaque master settles");
                assert_eq!(
                    tm.states[n as usize][qm], tm.states[st as usize][qm],
                    "opaque master holds"
                );
                assert!(!tr.order_sensitive);
            }
        }
    }
    // Slave: transparent while CLK = 1.
    let (pm, pc2) = (pos(&s.inputs, mq), pos(&s.inputs, clk));
    let qs = pos(&s.nets, sq);
    for st in 0..ts.num_states() as u32 {
        for d in [false, true] {
            let tr = ts.transition(st, vec_idx(&[(pm, d), (pc2, true)]));
            if let Some(n) = tr.next {
                assert_eq!(ts.states[n as usize][qs], Logic::from_bool(d));
            }
        }
    }
    // The clock inverter inside the master: CLK alone changing never needs D, and a single-input
    // change is never order sensitive.
    for st in 0..tm.num_states() as u32 {
        let home = tm.home_input[st as usize];
        assert!(!tm.transition(st, home ^ (1 << pc)).order_sensitive);
    }
    assert_eq!(
        tm.num_states(),
        6,
        "same shape as a d_latch (nclk is a function of CLK)"
    );
}

#[test]
fn table_independent_of_home_input() {
    // T(s, v) must not depend on which input vector s was reached under.
    for flat in [sr_flat(), dlatch_flat(), cyclic_comb()] {
        let ks = extract(&flat, &ExtractOptions::default());
        for k in &ks.kernels {
            let Some(t) = characterize(&flat, k, &CharzOptions::default()) else {
                continue;
            };
            assert_table_matches_sim(&flat, k, &t);
        }
    }
}

fn sr_flat() -> FlatNetlist {
    let mut db = DesignBuilder::new();
    let top = nand_sr_latch(&mut db);
    flat_of(db, &top)
}

fn dlatch_flat() -> FlatNetlist {
    let mut db = DesignBuilder::new();
    let top = d_latch(&mut db);
    flat_of(db, &top)
}

/// Every (state, home input, new input) on the fly == the table entry.
fn assert_table_matches_sim(flat: &FlatNetlist, k: &Kernel, t: &KernelTable) {
    let opts = CharzOptions::default();
    let mut sim = KernelSim::new(flat, k, &opts);
    let n = k.inputs.len();
    for (u, list) in t.states_by_input.iter().enumerate() {
        for &s in list {
            for v in 0..1u32 << n {
                let (st, settled) = sim.transition(
                    &t.states[s as usize],
                    &input_vector(u as u32, n),
                    &input_vector(v, n),
                );
                let got = if settled.quiescent {
                    t.lookup(&st)
                } else {
                    None
                };
                let tr = t.transition(s, v);
                assert_eq!(
                    got, tr.next,
                    "kernel {}: state {s} home {u:b} → {v:b}",
                    k.id
                );
                if got.is_some() {
                    assert_eq!(settled.latency, tr.latency);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Panda-8
// ---------------------------------------------------------------------------------------------

#[test]
fn panda8_all_state_in_small_table_kernels() {
    let image = programs::FIBONACCI.image();
    for rom in [RomStyle::SumOfProducts, RomStyle::MuxTree] {
        for variant in Variant::ALL {
            let d = panda8_with(&image, &Options::new(variant).with_rom(rom));
            let flat = flat_design(&d);
            let t0 = Instant::now();
            let (ks, stats) = extract_with_stats(&flat, &ExtractOptions::default());
            let dt = t0.elapsed();
            check(&ks, &flat);
            let h = ks.histogram();
            let max_bits = ks.kernels.iter().map(Kernel::bits).max().unwrap_or(0);
            let t1 = Instant::now();
            let mut cache = TableCache::new();
            let opts = CharzOptions::default();
            let tables: Vec<_> = ks
                .kernels
                .iter()
                .map(|k| cache.get_or_characterize(&flat, k, &opts))
                .collect();
            let dc = t1.elapsed();
            let order_sens: usize = tables
                .iter()
                .flatten()
                .map(|t| t.transitions.iter().filter(|x| x.order_sensitive).count())
                .sum();
            println!(
                "panda8 {variant:?} {rom:?}: {} gates, {} kernels {h:?}, sizes {:?}, max bits {max_bits}, \
                 comb {} nodes, cycles {}, absorbed {}, extract {dt:?}, charz {dc:?} ({} misses, {} hits), \
                 {order_sens} order-sensitive entries, partial order checks in {} of {} tables, table bits {:?}",
                flat.gates.len(),
                ks.kernels.len(),
                ks.size_histogram(),
                ks.comb_order.len(),
                stats.cycles,
                stats.absorbed,
                cache.misses(),
                cache.hits(),
                tables
                    .iter()
                    .flatten()
                    .filter(|t| t.order_check_partial)
                    .count(),
                tables.len(),
                {
                    let mut b: Vec<(usize, usize, usize)> = ks
                        .kernels
                        .iter()
                        .zip(&tables)
                        .map(|(k, t)| {
                            (
                                k.inputs.len(),
                                k.cut.len(),
                                t.as_ref().map_or(0, |t| t.num_states()),
                            )
                        })
                        .collect();
                    b.sort();
                    b.dedup();
                    b
                },
            );
            assert_eq!(h.sim, 0, "no Sim kernels");
            assert!(h.table > 0);
            assert!(max_bits <= 12);
            assert!(
                tables
                    .iter()
                    .all(|t| t.as_ref().is_some_and(|t| !t.is_oscillator()))
            );
            assert!(dt.as_secs_f64() < 2.0, "extraction took {dt:?}");
        }
    }
}

/// Disjoint union of `n` copies of `flat` (indices offset; names not unique).
fn replicate(flat: &FlatNetlist, n: usize) -> FlatNetlist {
    let mut out = FlatNetlist::default();
    for c in 0..n {
        let no = out.num_nets;
        let io = out.instances.len() as u32;
        let go = out.gates.len() as u32;
        let grp = out.groups.len() as u32;
        let pin_o = out.top_inputs.len() as u32;
        for g in &flat.gates {
            let mut g = g.clone();
            g.inputs.iter_mut().for_each(|i| *i += no);
            g.output += no;
            g.owner += io;
            g.group += grp;
            out.gates.push(g);
        }
        for gr in &flat.groups {
            let mut gr = gr.clone();
            gr.owner += io;
            gr.gates.iter_mut().for_each(|x| *x += go);
            out.groups.push(gr);
        }
        for r in &flat.resolvers {
            let mut r = r.clone();
            r.drivers.iter_mut().for_each(|i| *i += no);
            r.output += no;
            out.resolvers.push(r);
        }
        for s in &flat.sources {
            let mut s = s.clone();
            s.output += no;
            if let SourceKind::TopInput { pin, .. } = &mut s.kind {
                *pin += pin_o;
            }
            out.sources.push(s);
        }
        for i in &flat.instances {
            let mut i = i.clone();
            i.parent = i.parent.map(|p| p + io);
            i.children.iter_mut().for_each(|x| *x += io);
            i.path = format!("c{c}/{}", i.path);
            out.instances.push(i);
        }
        for p in &flat.top_inputs {
            let mut p = p.clone();
            p.name = format!("c{c}_{}", p.name);
            p.nets.iter_mut().for_each(|x| *x += no);
            out.top_inputs.push(p);
        }
        for p in &flat.top_outputs {
            let mut p = p.clone();
            p.name = format!("c{c}_{}", p.name);
            p.nets.iter_mut().for_each(|x| *x += no);
            out.top_outputs.push(p);
        }
        for (k, v) in &flat.undriven_termination {
            out.undriven_termination.insert(k + no, *v);
        }
        out.net_names
            .extend(flat.net_names.iter().map(|s| format!("c{c}/{s}")));
        out.num_nets += flat.num_nets;
    }
    out
}

#[test]
fn extraction_scales_to_200k_gates() {
    let d = panda8_with(
        &programs::FIBONACCI.image(),
        &Options::new(Variant::Flat).with_rom(RomStyle::MuxTree),
    );
    let one = flat_design(&d);
    let reps = 200_000 / one.gates.len() + 1;
    let big = replicate(&one, reps);
    assert!(big.gates.len() >= 200_000);
    let t0 = Instant::now();
    let ks = extract(&big, &ExtractOptions::default());
    let dt = t0.elapsed();
    let small = extract(&one, &ExtractOptions::default());
    println!(
        "{} gates: {} kernels in {dt:?}",
        big.gates.len(),
        ks.kernels.len()
    );
    assert_eq!(ks.kernels.len(), small.kernels.len() * reps);
    assert_eq!(ks.histogram().sim, 0);
    // Generous bound: debug assertions (a full verify) are on in tests; release is much faster.
    assert!(
        dt.as_secs_f64() < 5.0,
        "extraction of {} gates took {dt:?}",
        big.gates.len()
    );
}

// ---------------------------------------------------------------------------------------------
// Property test: random small kernels, table == on-the-fly transition
// ---------------------------------------------------------------------------------------------

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
}

/// A random netlist: `k` top inputs, `g` random gates over inputs and gate outputs (so feedback is
/// likely), optionally a tri-state bus with a random termination.
fn random_netlist(rng: &mut Rng) -> FlatNetlist {
    let k = 1 + rng.below(3);
    let g = 2 + rng.below(5);
    let bus = rng.below(4) == 0;
    let mut f = FlatNetlist::default();
    // nets: inputs 0..k, gate outputs k..k+g, then (bus) 2 tri nets + resolver output
    let n_gate_out = k + g;
    let total = n_gate_out + if bus { 3 } else { 0 };
    f.num_nets = total as u32;
    for i in 0..k {
        f.sources.push(Source {
            kind: SourceKind::TopInput {
                pin: 0,
                bit: i as u32,
            },
            output: i as NetIdx,
        });
    }
    f.top_inputs.push(FlatPin {
        name: "in".into(),
        dir: Dir::In,
        nets: (0..k as NetIdx).collect(),
    });
    let readable = if bus { total } else { n_gate_out };
    let ops = [
        FlatOp::Nand,
        FlatOp::Nor,
        FlatOp::And,
        FlatOp::Or,
        FlatOp::Xor,
        FlatOp::Not,
        FlatOp::Nand,
        FlatOp::Nor,
    ];
    for j in 0..g {
        let op = ops[rng.below(ops.len())];
        let arity = if op == FlatOp::Not {
            1
        } else {
            2 + rng.below(2)
        };
        let inputs = (0..arity)
            .map(|_| {
                let x = rng.below(readable);
                // keep the bus-internal tri nets private to the resolver
                if bus && (n_gate_out..n_gate_out + 2).contains(&x) {
                    (total - 1) as NetIdx
                } else {
                    x as NetIdx
                }
            })
            .collect();
        f.gates.push(FlatGate {
            op,
            inputs,
            output: (k + j) as NetIdx,
            owner: 0,
            group: j as u32,
            bit: 0,
        });
    }
    if bus {
        for t in 0..2 {
            let d = rng.below(n_gate_out) as NetIdx;
            let en = rng.below(n_gate_out) as NetIdx;
            let j = f.gates.len() as u32;
            f.gates.push(FlatGate {
                op: FlatOp::Tri,
                inputs: vec![d, en],
                output: (n_gate_out + t) as NetIdx,
                owner: 0,
                group: j,
                bit: 0,
            });
        }
        let term = [Termination::None, Termination::PullUp, Termination::Keeper][rng.below(3)];
        f.resolvers.push(Resolver {
            drivers: vec![n_gate_out as NetIdx, (n_gate_out + 1) as NetIdx],
            termination: term,
            output: (total - 1) as NetIdx,
        });
    }
    f.net_names = (0..total).map(|i| format!("n{i}")).collect();
    f
}

#[test]
fn property_random_kernels_table_equals_on_the_fly() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let mut tables = 0;
    let mut entries = 0;
    let mut full_checked = 0;
    for _case in 0..300 {
        let flat = random_netlist(&mut rng);
        let ks = extract(&flat, &ExtractOptions::default());
        check(&ks, &flat);
        for k in &ks.kernels {
            let Some(t) = characterize(&flat, k, &CharzOptions::default()) else {
                continue;
            };
            tables += 1;
            entries += t.transitions.len();
            assert_table_matches_sim(&flat, k, &t);
            // Stable states really are fixed points; transitions land on stable states of v.
            for s in 0..t.num_states() as u32 {
                for v in 0..1u32 << k.inputs.len() {
                    if let Some(n) = t.transition(s, v).next {
                        assert!(t.states_by_input[v as usize].contains(&n));
                    }
                }
            }
            // Whole-netlist cross-check when the kernel reads only top inputs and the netlist is
            // exactly the kernel: the full TimingSim must agree with the table.
            let all_src = k
                .inputs
                .iter()
                .all(|&n| (n as usize) < flat.top_inputs[0].nets.len());
            if all_src && ks.kernels.len() == 1 && ks.comb_order.is_empty() {
                full_checked += 1;
                full_netlist_check(&flat, k, &t);
            }
        }
    }
    println!("{tables} tables, {entries} entries, {full_checked} whole-netlist checks");
    assert!(tables > 50);
    assert!(full_checked > 10);
}

fn full_netlist_check(flat: &FlatNetlist, k: &Kernel, t: &KernelTable) {
    let arc = Arc::new(flat.clone());
    let mut sim = TimingSim::new(arc, TimingOptions::default());
    let nin = flat.top_inputs[0].nets.len();
    let pin_val = |iv: u32| -> panda_netlist::Value {
        // kernel input i is net kernel.inputs[i]; top input bit b is net b
        let mut bits = vec![O; nin];
        for (i, &net) in k.inputs.iter().enumerate() {
            bits[net as usize] = Logic::from_bool(iv >> i & 1 == 1);
        }
        panda_netlist::Value::from_bits(&bits)
    };
    let mut memo: HashMap<(u32, u32), Option<u32>> = HashMap::new();
    for s in 0..t.num_states() as u32 {
        let home = t.home_input[s as usize];
        for v in 0..1u32 << k.inputs.len() {
            sim.set_input("in", &pin_val(home)).unwrap();
            sim.reset();
            let mut order: Vec<usize> = (0..k.nets.len()).collect();
            order.sort_by_key(|&j| flat.resolvers.iter().any(|r| r.output == k.nets[j]));
            for j in order {
                sim.force_net(k.nets[j], t.states[s as usize][j]);
            }
            assert!(sim.settle(1000));
            sim.set_input("in", &pin_val(v)).unwrap();
            let ok = sim.settle(t_bound(k));
            let st: Vec<Logic> = k.nets.iter().map(|&n| sim.get(n)).collect();
            let got = if ok { t.lookup(&st) } else { None };
            memo.insert((s, v), got);
            assert_eq!(got, t.transition(s, v).next, "full sim: state {s} → {v:b}");
            sim.take_diagnostics();
        }
    }
}

fn t_bound(k: &Kernel) -> u64 {
    CharzOptions::default().bound_for(k.nets.len())
}
