//! Value Change Dump (IEEE 1364 VCD) writer for [`TimingSim`].
//!
//! * **Timescale:** one simulation tick is written as `1ns` (`$timescale 1ns $end`); timestamps
//!   are tick numbers.
//! * **Values:** 4-state; scalars as `0`/`1`/`x`/`z`, vectors as `b<bits>` MSB first.
//! * **Signals:** any [`Signal`] selection (see [`crate::signals`]); [`default_selection`] picks
//!   the top-level pins plus all explicitly named top-level nets, with `base[i]` bit nets grouped
//!   back into buses. Hierarchical names (`cnt/ff0/m`) are placed in nested `$scope module`
//!   blocks under a root scope named after the top def; the variable reference is the last path
//!   component (whitespace replaced by `_`), with `[msb:lsb]` appended for multi-bit signals.
//! * **Sampling:** call [`VcdWriter::sample`] after every [`TimingSim::step`] (or use
//!   [`advance`] / [`settle`], which do that for you). When exactly one tick has passed since the
//!   previous sample, only the nets in [`TimingSim::last_changed`] are examined; otherwise
//!   (first sample, a `run` of several ticks, a reset, forced nets) every selected net is
//!   compared against the last dumped value, so the dump stays correct, just coarser.
//! * **Resets:** VCD time must not go backwards. When the simulator's tick goes back (after
//!   [`TimingSim::reset`]), the writer continues from one tick after the last written time;
//!   i.e. dump time = simulator tick + an offset that only grows.
//!
//! The writer emits no `$date` so output is deterministic.

use std::collections::HashMap;
use std::io::{self, Write};

use panda_netlist::Logic;
use panda_netlist::flat::{FlatNetlist, NetIdx};
use panda_timing::TimingSim;

pub use crate::signals::{Signal, default_selection};

/// Streams a VCD file for a fixed selection of signals.
pub struct VcdWriter<W: Write> {
    out: W,
    signals: Vec<Signal>,
    codes: Vec<String>,
    /// Net → indices of the signals containing it.
    net_signals: HashMap<NetIdx, Vec<u32>>,
    /// Last dumped value per signal (empty before the first sample).
    last: Vec<Vec<Logic>>,
    dirty: Vec<bool>,
    dirty_list: Vec<u32>,
    /// Simulator tick at the previous sample.
    prev_tick: Option<u64>,
    /// Dump time = sim tick + offset.
    offset: u64,
    /// Last timestamp written.
    last_time: Option<u64>,
    scratch: String,
}

/// VCD identifier code for signal number `i` (printable ASCII `!`..`~`, base 94).
pub fn id_code(mut i: usize) -> String {
    let mut s = String::new();
    loop {
        s.push((b'!' + (i % 94) as u8) as char);
        i /= 94;
        if i == 0 {
            break;
        }
        i -= 1;
    }
    s
}

fn sanitize(s: &str) -> String {
    let t: String = s.chars().map(|c| if c.is_whitespace() { '_' } else { c }).collect();
    if t.is_empty() { "_".into() } else { t }
}

impl<W: Write> VcdWriter<W> {
    /// Write the header for `signals` and return the writer. `top_name` names the root scope
    /// (usually the top def id). No values are written until the first [`VcdWriter::sample`].
    pub fn new(mut out: W, netlist: &FlatNetlist, top_name: &str, signals: Vec<Signal>) -> io::Result<Self> {
        let codes: Vec<String> = (0..signals.len()).map(id_code).collect();
        let mut net_signals: HashMap<NetIdx, Vec<u32>> = HashMap::new();
        for (i, s) in signals.iter().enumerate() {
            for &n in &s.nets {
                if n >= netlist.num_nets {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("signal '{}' refers to net {n} out of range", s.name),
                    ));
                }
                let v = net_signals.entry(n).or_default();
                if v.last() != Some(&(i as u32)) {
                    v.push(i as u32);
                }
            }
        }

        writeln!(out, "$version panda-cli {} $end", env!("CARGO_PKG_VERSION"))?;
        writeln!(out, "$comment 1 tick = 1ns $end")?;
        writeln!(out, "$timescale 1ns $end")?;
        writeln!(out, "$scope module {} $end", sanitize(top_name))?;
        // Group signals by scope path, emitting nested scopes in order of first appearance.
        let mut order: Vec<Vec<String>> = Vec::new();
        let mut by_scope: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
        for (i, s) in signals.iter().enumerate() {
            let mut parts: Vec<String> = s.name.split('/').map(sanitize).collect();
            parts.pop();
            by_scope
                .entry(parts.clone())
                .or_insert_with(|| {
                    order.push(parts);
                    Vec::new()
                })
                .push(i);
        }
        // Sort scopes so that a scope's subtree is contiguous (stable w.r.t. first appearance
        // within siblings is not needed for correctness; lexicographic keeps it deterministic).
        order.sort();
        let mut open: Vec<String> = Vec::new();
        for scope in &order {
            let common = open.iter().zip(scope).take_while(|(a, b)| a == b).count();
            while open.len() > common {
                writeln!(out, "$upscope $end")?;
                open.pop();
            }
            for part in &scope[common..] {
                writeln!(out, "$scope module {part} $end")?;
                open.push(part.clone());
            }
            for &i in &by_scope[scope] {
                let s = &signals[i];
                let leaf = sanitize(s.name.rsplit('/').next().unwrap_or(&s.name));
                if s.width() == 1 {
                    writeln!(out, "$var wire 1 {} {leaf} $end", codes[i])?;
                } else {
                    writeln!(out, "$var wire {} {} {leaf} [{}:0] $end", s.width(), codes[i], s.width() - 1)?;
                }
            }
        }
        for _ in open {
            writeln!(out, "$upscope $end")?;
        }
        writeln!(out, "$upscope $end")?;
        writeln!(out, "$enddefinitions $end")?;

        let n = signals.len();
        Ok(VcdWriter {
            out,
            signals,
            codes,
            net_signals,
            last: vec![Vec::new(); n],
            dirty: vec![false; n],
            dirty_list: Vec::new(),
            prev_tick: None,
            offset: 0,
            last_time: None,
            scratch: String::new(),
        })
    }

    /// The selected signals.
    pub fn signals(&self) -> &[Signal] {
        &self.signals
    }

    /// Record the simulator's current values at its current tick (see the module docs).
    pub fn sample(&mut self, sim: &TimingSim) -> io::Result<()> {
        let tick = sim.tick();
        let incremental = matches!(self.prev_tick, Some(p) if tick == p + 1);
        if let Some(p) = self.prev_tick
            && tick < p
        {
            // Reset: continue after the last written time.
            let resume = self.last_time.map_or(0, |t| t + 1);
            self.offset = resume.saturating_sub(tick).max(self.offset);
        }
        self.prev_tick = Some(tick);
        let time = tick + self.offset;

        self.dirty_list.clear();
        if incremental {
            for &n in sim.last_changed() {
                if let Some(sigs) = self.net_signals.get(&n) {
                    for &s in sigs {
                        if !self.dirty[s as usize] {
                            self.dirty[s as usize] = true;
                            self.dirty_list.push(s);
                        }
                    }
                }
            }
            self.dirty_list.sort_unstable();
        } else {
            self.dirty_list.extend(0..self.signals.len() as u32);
        }

        let first = self.last_time.is_none();
        let mut header_written = false;
        for k in 0..self.dirty_list.len() {
            let s = self.dirty_list[k] as usize;
            self.dirty[s] = false;
            let nets = &self.signals[s].nets;
            let last = &mut self.last[s];
            let changed = last.len() != nets.len() || nets.iter().zip(last.iter()).any(|(&n, &l)| sim.get(n) != l);
            if !changed {
                continue;
            }
            last.clear();
            last.extend(nets.iter().map(|&n| sim.get(n)));
            if !header_written {
                header_written = true;
                // Sampling the same tick twice must not repeat the timestamp.
                if self.last_time.is_none_or(|t| time > t) {
                    writeln!(self.out, "#{time}")?;
                    self.last_time = Some(time);
                }
                if first {
                    writeln!(self.out, "$dumpvars")?;
                }
            }
            self.scratch.clear();
            if nets.len() == 1 {
                self.scratch.push(last[0].to_char());
            } else {
                self.scratch.push('b');
                self.scratch.extend(last.iter().rev().map(|l| l.to_char()));
                self.scratch.push(' ');
            }
            writeln!(self.out, "{}{}", self.scratch, self.codes[s])?;
        }
        if first && header_written {
            writeln!(self.out, "$end")?;
        }
        Ok(())
    }

    /// Write a final timestamp for the simulator's current tick (so viewers show the whole
    /// simulated span), flush, and return the underlying writer.
    pub fn finish(mut self, sim: &TimingSim) -> io::Result<W> {
        let time = sim.tick() + self.offset;
        if self.last_time.is_none_or(|t| time > t) {
            writeln!(self.out, "#{time}")?;
        }
        self.out.flush()?;
        Ok(self.out)
    }
}

/// Advance `sim` by `ticks` ticks, sampling every tick into `vcd` (if any). Quiescent stretches
/// are fast-forwarded.
pub fn advance<W: Write>(sim: &mut TimingSim, ticks: u64, vcd: Option<&mut VcdWriter<W>>) -> io::Result<()> {
    match vcd {
        None => sim.run(ticks),
        Some(w) => {
            for done in 0..ticks {
                if sim.is_quiescent() {
                    sim.run(ticks - done);
                    break;
                }
                sim.step();
                w.sample(sim)?;
            }
        }
    }
    Ok(())
}

/// Run `sim` until quiescent or `max_ticks` elapse, sampling every tick into `vcd` (if any).
/// Returns true if quiescent.
pub fn settle<W: Write>(sim: &mut TimingSim, max_ticks: u64, vcd: Option<&mut VcdWriter<W>>) -> io::Result<bool> {
    match vcd {
        None => Ok(sim.settle(max_ticks)),
        Some(w) => {
            for _ in 0..max_ticks {
                if sim.is_quiescent() {
                    return Ok(true);
                }
                sim.step();
                w.sample(sim)?;
            }
            Ok(sim.is_quiescent())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_codes_unique() {
        let codes: std::collections::HashSet<String> = (0..20_000).map(id_code).collect();
        assert_eq!(codes.len(), 20_000);
        assert_eq!(id_code(0), "!");
        assert_eq!(id_code(93), "~");
        assert_eq!(id_code(94), "!!");
        assert!(codes.iter().all(|c| c.bytes().all(|b| (33..=126).contains(&b))));
    }
}
