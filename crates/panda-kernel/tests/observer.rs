//! The UI-facing facade additions: bulk reads by flat index (`copy_values`, `value`) and the
//! per-event observer (`run_observed`, `step_observed`).

mod common;

use std::time::Duration;

use common::*;
use panda_circuits::{sync_counter, testbench};
use panda_cpu8::programs;
use panda_cpu8::{RomStyle, Variant};
use panda_kernel::{EventObserver, Kernel, Mode, StepUnit};
use panda_netlist::flat::NetIdx;
use panda_netlist::{DesignBuilder, Logic, Value};

/// Records every observation.
struct Recorder {
    nets: Vec<NetIdx>,
    seen: Vec<(u64, Vec<Logic>)>,
}

impl EventObserver for Recorder {
    fn nets(&self) -> &[NetIdx] {
        &self.nets
    }
    fn observe(&mut self, tick: u64, values: &[Logic]) {
        self.seen.push((tick, values.to_vec()));
    }
}

fn counter_kernel(mode: Mode) -> Kernel {
    let mut db = DesignBuilder::new();
    let c = sync_counter(&mut db, 4);
    let top = testbench(&mut db, &c, 16);
    let mut k = Kernel::new(quiet_opts());
    k.set_mode(mode);
    k.load(db.build(&top)).unwrap();
    k.wait_ready().unwrap();
    k.set_input("EN", &Value::from_bool(true)).unwrap();
    k
}

fn q_nets(k: &Kernel) -> Vec<NetIdx> {
    let flat = k.netlist().unwrap();
    (0..4)
        .map(|b| flat.net_by_name(&format!("Q[{b}]")).unwrap())
        .collect()
}

#[test]
fn copy_values_matches_read() {
    for mode in [Mode::Timing, Mode::Fast] {
        let mut k = counter_kernel(mode);
        k.step(7, StepUnit::Cycle);
        let flat = k.netlist().unwrap().clone();
        let all: Vec<NetIdx> = (0..flat.num_nets).collect();
        let mut out = vec![Logic::X; all.len()];
        k.copy_values(&all, &mut out);
        for (n, name) in flat.net_names.iter().enumerate() {
            let r = k.read(name).unwrap();
            assert_eq!(r.width(), 1, "{name}");
            assert_eq!(out[n], r.get(0), "{mode:?} {name}");
            assert_eq!(k.value(n as NetIdx), Some(out[n]));
        }
        assert_eq!(k.value(flat.num_nets), None);
    }
}

#[test]
fn observer_sees_every_fast_event_and_final_state() {
    let mut k = counter_kernel(Mode::Fast);
    let nets = q_nets(&k);
    let mut rec = Recorder {
        nets: nets.clone(),
        seen: Vec::new(),
    };
    k.step_observed(10, StepUnit::Cycle, &mut rec);
    // A clocked design: two events per cycle (rise, fall) plus the input event; ticks increase.
    assert!(rec.seen.len() >= 20, "{} events", rec.seen.len());
    assert!(rec.seen.windows(2).all(|w| w[0].0 <= w[1].0));
    let mut now = vec![Logic::X; 4];
    k.copy_values(&nets, &mut now);
    assert_eq!(rec.seen.last().unwrap().1, now);
    // The counter value increments: the observed Q takes at least 10 distinct values.
    let mut distinct: Vec<Vec<Logic>> = rec.seen.iter().map(|s| s.1.clone()).collect();
    distinct.dedup();
    assert!(distinct.len() >= 10, "{distinct:?}");
}

#[test]
fn observed_and_plain_runs_agree() {
    for mode in [Mode::Timing, Mode::Fast] {
        let mut a = counter_kernel(mode);
        let mut b = counter_kernel(mode);
        let mut rec = Recorder {
            nets: q_nets(&a),
            seen: Vec::new(),
        };
        a.step(13, StepUnit::Cycle);
        b.step_observed(13, StepUnit::Cycle, &mut rec);
        assert_eq!(a.tick(), b.tick());
        assert_eq!(a.read_u64("Q"), b.read_u64("Q"));
        assert!(!rec.seen.is_empty());
        if mode == Mode::Timing {
            // Timing mode: one observation per tick that changed something, at that tick.
            assert!(rec.seen.iter().all(|s| s.0 <= b.tick()));
        }
    }
}

#[test]
fn run_observed_on_panda8() {
    let (design, _) = cpu_design(
        &programs::BENCH_LOOP.image(),
        Variant::Hierarchical,
        RomStyle::SumOfProducts,
    );
    let mut k = Kernel::new(quiet_opts());
    k.set_mode(Mode::Fast);
    k.load(design).unwrap();
    k.wait_ready().unwrap();
    let flat = k.netlist().unwrap().clone();
    let nets: Vec<NetIdx> = (0..8)
        .map(|b| flat.net_by_name(&format!("PC[{b}]")).unwrap())
        .collect();
    let mut rec = Recorder {
        nets,
        seen: Vec::new(),
    };
    let s = k.run_observed(Duration::from_millis(20), &mut rec);
    assert!(s.cycles > 0);
    assert!(
        rec.seen.len() as u64 >= 2 * s.cycles - 2,
        "{} events for {} cycles",
        rec.seen.len(),
        s.cycles
    );
}
