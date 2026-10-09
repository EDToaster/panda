//! The Timing-mode engine of the facade: a [`TimingSim`] plus a tick offset, so that a fresh
//! simulator can be started "at" any tick with any set of net values.

use std::sync::Arc;

use panda_netlist::eval::{eval_gate, resolve};
use panda_netlist::flat::{FlatNetlist, NetIdx};
use panda_netlist::{Diagnostic, Logic, Value};
use panda_timing::{TimingOptions, TimingSim};

use crate::schedule::Schedule;

/// A [`TimingSim`] whose tick is shifted by `offset` (facade time = `offset + sim.tick()`).
pub(crate) struct TimingEngine {
    pub(crate) sim: TimingSim,
    offset: u64,
    /// Facade tick of [`TimingEngine::load`], if loaded rather than powered on: until the engine
    /// advances, its state is the loaded one (gates scheduled by the load are not Timing-mode
    /// activity).
    loaded_at: Option<u64>,
}

impl TimingEngine {
    /// Power on at tick 0 with the given input pin values (by pin index; widths must match).
    pub(crate) fn power_on(
        flat: Arc<FlatNetlist>,
        opts: &TimingOptions,
        inputs: &[Value],
    ) -> TimingEngine {
        let mut sim = TimingSim::new(flat, opts.clone());
        apply_inputs(&mut sim, inputs);
        sim.reset();
        TimingEngine {
            sim,
            offset: 0,
            loaded_at: None,
        }
    }

    /// Start at facade tick `tick` with net `values` (`None` = unknown → `X` for driven nets).
    ///
    /// The simulator is run from power-on to the aligned start tick `t0 ≡ tick` (so every
    /// `Clock` / `PowerOnReset` has its tick-`tick` value and future), then every non-source net is
    /// forced to its value and every gate or resolver whose output is inconsistent with its inputs
    /// is scheduled, so the next step evaluates it (a consistent, settled state stays put).
    /// Diagnostics raised while warming up are discarded.
    pub(crate) fn load(
        flat: Arc<FlatNetlist>,
        opts: &TimingOptions,
        schedule: &Schedule,
        tick: u64,
        values: &[Option<Logic>],
        inputs: &[Value],
    ) -> TimingEngine {
        debug_assert_eq!(values.len(), flat.num_nets as usize);
        let mut eng = TimingEngine::power_on(flat.clone(), opts, inputs);
        let t0 = schedule.aligned_start(tick);
        eng.sim.run(t0);
        eng.offset = tick - eng.sim.tick();
        let mut is_source = vec![false; flat.num_nets as usize];
        for s in &flat.sources {
            is_source[s.output as usize] = true;
        }
        let target: Vec<Logic> = (0..flat.num_nets as usize)
            .map(|n| {
                if is_source[n] {
                    eng.sim.values()[n]
                } else {
                    values[n].unwrap_or(Logic::X)
                }
            })
            .collect();
        for (n, &v) in target.iter().enumerate() {
            if !is_source[n] && eng.sim.values()[n] != v {
                eng.sim.force_net(n as NetIdx, v);
            }
        }
        // Resolvers recompute from their drivers when those are forced; force their outputs last
        // so a carried keeper value wins, then re-check below.
        for r in &flat.resolvers {
            let n = r.output as usize;
            if eng.sim.values()[n] != target[n] {
                eng.sim.force_net(r.output, target[n]);
            }
        }
        // Gates (and resolvers) whose output disagrees with their inputs: schedule them by
        // toggling one of their input nets back and forth (a forced change schedules every reader).
        let mut buf = Vec::new();
        let mut kick: Vec<NetIdx> = Vec::new();
        {
            let vals = eng.sim.values();
            for g in &flat.gates {
                buf.clear();
                buf.extend(g.inputs.iter().map(|&i| vals[i as usize]));
                if eval_gate(g.op, &buf) != vals[g.output as usize] {
                    kick.push(g.inputs[0]);
                }
            }
            for r in &flat.resolvers {
                buf.clear();
                buf.extend(r.drivers.iter().map(|&i| vals[i as usize]));
                let out = vals[r.output as usize];
                if resolve(&buf, r.termination, out).value != out
                    && let Some(&d) = r.drivers.first()
                {
                    kick.push(d);
                }
            }
        }
        kick.sort_unstable();
        kick.dedup();
        for net in kick {
            let v = eng.sim.values()[net as usize];
            let other = if v == Logic::X { Logic::Zero } else { Logic::X };
            eng.sim.force_net(net, other);
            eng.sim.force_net(net, v);
        }
        eng.sim.take_diagnostics();
        eng.loaded_at = Some(tick);
        eng
    }

    /// True if the Timing engine has gate events pending that come from its own simulation (not
    /// merely from a [`TimingEngine::load`] at the current tick).
    pub(crate) fn busy(&self) -> bool {
        self.sim.pending_events() > 0 && self.loaded_at != Some(self.tick())
    }

    /// Facade tick.
    pub(crate) fn tick(&self) -> u64 {
        self.offset + self.sim.tick()
    }

    /// Advance to facade tick `target` (no-op if already there or past it).
    pub(crate) fn run_to(&mut self, target: u64) {
        let now = self.tick();
        if target > now {
            self.sim.run(target - now);
        }
    }

    pub(crate) fn take_diagnostics(&mut self) -> Vec<Diagnostic> {
        self.sim.take_diagnostics()
    }
}

/// Give every top-level input pin its value (by pin index); mismatched widths are skipped.
pub(crate) fn apply_inputs(sim: &mut TimingSim, inputs: &[Value]) {
    let pins: Vec<(String, u32)> = sim
        .netlist()
        .top_inputs
        .iter()
        .map(|p| (p.name.clone(), p.nets.len() as u32))
        .collect();
    for ((name, w), v) in pins.iter().zip(inputs) {
        if v.width() == *w {
            let _ = sim.set_input(name, v);
        }
    }
}
