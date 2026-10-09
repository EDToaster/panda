//! Waveform capture: a per-event [`EventObserver`] on the sim thread and the UI-side store.
//!
//! A [`WaveRecorder`] samples the watched signals after every event (Fast mode) / changing tick
//! (Timing mode) and records only **changes** ([`WaveChange`]). It runs on the shadow thread
//! (default, see [`super::WaveCapture`]) or on the sim thread; recorded changes are shipped to
//! the UI as [`WaveChunk`]s over the event channel and appended to a bounded per-signal history
//! ([`WaveStore`]). The main simulation runs plain [`panda_kernel::Kernel::run`] unless inline
//! capture is selected and signals are watched.
//!
//! Signals driven directly by a `Clock` source ([`WaveSignal::clock`]) are not sampled: their
//! waveform is a function of the tick, which [`WaveStore`] synthesizes. Sampling cost is
//! proportional to the number of sampled nets (one gather per event inside the engine loop).

use std::collections::VecDeque;

use panda_kernel::EventObserver;
use panda_netlist::flat::{FlatNetlist, NetIdx, SourceKind};
use panda_netlist::{Logic, Value};

/// A `Clock` source's timing (SPEC §4.1): `out = 1` iff
/// `((t + period − phase % period) % period) < high`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClockSpec {
    /// Period (ticks).
    pub period: u32,
    /// Ticks high per period.
    pub high: u32,
    /// Rising-edge offset.
    pub phase: u32,
}

impl ClockSpec {
    /// Value at `tick`.
    pub fn value(&self, tick: u64) -> bool {
        let p = u64::from(self.period);
        (tick + p - u64::from(self.phase) % p) % p < u64::from(self.high)
    }

    /// The clock of the source driving `net`, if `net` is a clock net.
    pub fn of_net(flat: &FlatNetlist, net: NetIdx) -> Option<ClockSpec> {
        flat.sources
            .iter()
            .find(|s| s.output == net)
            .and_then(|s| match s.kind {
                SourceKind::Clock {
                    period,
                    high,
                    phase,
                } => Some(ClockSpec {
                    period,
                    high,
                    phase,
                }),
                _ => None,
            })
    }
}

/// One watched signal: a name and its flat nets (LSB first, ≤ 64 bits).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WaveSignal {
    /// Display name (e.g. the hierarchical net name).
    pub name: String,
    /// Flat nets, LSB first.
    pub nets: Vec<NetIdx>,
    /// Set for a 1-bit signal driven by a `Clock` source: not sampled, synthesized.
    pub clock: Option<ClockSpec>,
}

impl WaveSignal {
    /// A signal for `nets` of `flat` (detects clock nets).
    pub fn new(flat: &FlatNetlist, name: impl Into<String>, nets: Vec<NetIdx>) -> WaveSignal {
        let clock = match nets.as_slice() {
            [n] => ClockSpec::of_net(flat, *n),
            _ => None,
        };
        WaveSignal {
            name: name.into(),
            nets,
            clock,
        }
    }

    /// Bit width (≤ 64 recorded).
    pub fn width(&self) -> u32 {
        self.nets.len().min(64) as u32
    }
}

/// The set of watched waveform signals.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WaveConfig {
    /// Caller-chosen id (echoed in every [`WaveChunk`]).
    pub id: u64,
    /// Design generation the net indices refer to (see `Snapshot::design_gen`).
    pub design_gen: u64,
    /// Signals.
    pub signals: Vec<WaveSignal>,
}

/// A packed ≤ 64-bit 4-state value: two bit planes as in [`Value`] (`0=(0,0) 1=(1,0) Z=(0,1)
/// X=(1,1)`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Packed {
    /// Plane a.
    pub a: u64,
    /// Plane b.
    pub b: u64,
}

impl Packed {
    /// Pack bits (LSB first; bits past 64 are ignored).
    pub fn from_bits(bits: &[Logic]) -> Packed {
        let mut p = Packed::default();
        for (i, &l) in bits.iter().take(64).enumerate() {
            let (a, b) = l.planes();
            p.a |= u64::from(a) << i;
            p.b |= u64::from(b) << i;
        }
        p
    }

    /// A 1-bit binary value.
    pub fn from_bool(v: bool) -> Packed {
        Packed {
            a: u64::from(v),
            b: 0,
        }
    }

    /// Unpack into a [`Value`] of `width` bits.
    pub fn to_value(self, width: u32) -> Value {
        let bits: Vec<Logic> = (0..width.min(64))
            .map(|i| Logic::from_planes(self.a >> i & 1 == 1, self.b >> i & 1 == 1))
            .collect();
        Value::from_bits(&bits)
    }
}

/// A recorded change of signal `signal` to `value` at `tick`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WaveChange {
    /// Tick of the event.
    pub tick: u64,
    /// Signal index in the [`WaveConfig`].
    pub signal: u32,
    /// New value.
    pub value: Packed,
}

/// Changes recorded between two publications.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WaveChunk {
    /// [`WaveConfig::id`] the changes belong to.
    pub config: u64,
    /// Changes in tick order.
    pub changes: Vec<WaveChange>,
    /// Changes were dropped before this chunk (the UI did not drain events in time).
    pub gap: bool,
}

/// The sim-thread observer: samples the watched nets after every event and records changes.
#[derive(Debug, Default)]
pub struct WaveRecorder {
    config: WaveConfig,
    /// Concatenated nets of every sampled (non-clock) signal.
    nets: Vec<NetIdx>,
    /// Signal index of each entry of `nets` (a signal's nets are contiguous).
    sig_of: Vec<u32>,
    /// Range of each signal in `nets` (empty for clock signals).
    ranges: Vec<(usize, usize)>,
    /// Last sample (aligned with `nets`).
    last: Vec<Logic>,
    /// Changes since the last [`WaveRecorder::take_chunk`].
    pending: Vec<WaveChange>,
    /// Cap on `pending` (changes beyond it are dropped and flagged as a gap).
    cap: usize,
    gap: bool,
    /// Has a first sample (the initial values are recorded as changes).
    primed: bool,
}

impl WaveRecorder {
    /// Default cap on changes per chunk.
    pub const DEFAULT_CAP: usize = 1 << 20;

    /// A recorder for `config`.
    pub fn new(config: WaveConfig) -> WaveRecorder {
        let mut nets = Vec::new();
        let mut sig_of = Vec::new();
        let mut ranges = Vec::new();
        for (i, s) in config.signals.iter().enumerate() {
            let a = nets.len();
            if s.clock.is_none() {
                for &n in s.nets.iter().take(64) {
                    nets.push(n);
                    sig_of.push(i as u32);
                }
            }
            ranges.push((a, nets.len()));
        }
        let n = nets.len();
        WaveRecorder {
            config,
            nets,
            sig_of,
            ranges,
            last: vec![Logic::X; n],
            pending: Vec::new(),
            cap: Self::DEFAULT_CAP,
            gap: false,
            primed: false,
        }
    }

    /// True if nothing needs sampling (the host then runs without an observer).
    pub fn is_empty(&self) -> bool {
        self.nets.is_empty()
    }

    /// The configuration.
    pub fn config(&self) -> &WaveConfig {
        &self.config
    }

    /// Record the current values as a full sample (e.g. after a load or while paused) at `tick`.
    pub fn prime(&mut self, tick: u64, values: &[Logic]) {
        self.primed = false;
        self.observe(tick, values);
    }

    /// Take the changes recorded so far (None if there are none).
    pub fn take_chunk(&mut self) -> Option<WaveChunk> {
        if self.pending.is_empty() && !self.gap {
            return None;
        }
        let gap = std::mem::take(&mut self.gap);
        Some(WaveChunk {
            config: self.config.id,
            changes: std::mem::take(&mut self.pending),
            gap,
        })
    }

    fn push(&mut self, tick: u64, s: u32, values: &[Logic]) {
        if self.pending.len() >= self.cap {
            self.gap = true;
            return;
        }
        let (a, b) = self.ranges[s as usize];
        self.pending.push(WaveChange {
            tick,
            signal: s,
            value: Packed::from_bits(&values[a..b]),
        });
    }

    #[cold]
    fn record_all(&mut self, tick: u64, values: &[Logic]) {
        for s in 0..self.ranges.len() {
            let (a, b) = self.ranges[s];
            if a < b {
                self.push(tick, s as u32, values);
            }
        }
        self.last.copy_from_slice(values);
        self.primed = true;
    }

    fn record(&mut self, tick: u64, values: &[Logic]) {
        // Changed nets → their signals (contiguous, so one push per signal).
        let mut prev = u32::MAX;
        for i in 0..values.len() {
            if values[i] != self.last[i] {
                let s = self.sig_of[i];
                if s != prev {
                    self.push(tick, s, values);
                    prev = s;
                }
            }
        }
        self.last.copy_from_slice(values);
    }
}

impl EventObserver for WaveRecorder {
    fn nets(&self) -> &[NetIdx] {
        &self.nets
    }

    #[inline]
    fn observe(&mut self, tick: u64, values: &[Logic]) {
        if !self.primed {
            self.record_all(tick, values);
        } else if values != self.last.as_slice() {
            self.record(tick, values);
        }
    }

    fn observe_batch(&mut self, ticks: &[u64], values: &[Logic]) {
        let n = self.nets.len();
        if n == 0 {
            return;
        }
        for (&t, v) in ticks.iter().zip(values.chunks_exact(n)) {
            self.observe(t, v);
        }
    }
}

/// UI-side waveform history: per signal, a bounded deque of `(tick, value)` changes (clock
/// signals are synthesized from their [`ClockSpec`]).
#[derive(Clone, Debug, Default)]
pub struct WaveStore {
    /// Config the history belongs to.
    pub config: WaveConfig,
    /// Per signal: changes in tick order (empty for clock signals).
    pub history: Vec<VecDeque<(u64, Packed)>>,
    /// Max changes kept per signal.
    pub cap: usize,
    /// Number of chunks that reported lost changes (the sim outran the UI).
    pub gaps: u64,
}

impl WaveStore {
    /// Default per-signal history length.
    pub const DEFAULT_CAP: usize = 1 << 20;
    /// Max edges synthesized per [`WaveStore::changes_in`] call for a clock signal.
    pub const MAX_SYNTH: usize = 1 << 16;

    /// An empty store for `config`.
    pub fn new(config: WaveConfig) -> WaveStore {
        let n = config.signals.len();
        WaveStore {
            config,
            history: vec![VecDeque::new(); n],
            cap: Self::DEFAULT_CAP,
            gaps: 0,
        }
    }

    /// Append a chunk (ignored if it belongs to another config).
    pub fn append(&mut self, chunk: &WaveChunk) {
        if chunk.config != self.config.id {
            return;
        }
        if chunk.gap {
            self.gaps += 1;
        }
        for c in &chunk.changes {
            let Some(h) = self.history.get_mut(c.signal as usize) else {
                continue;
            };
            if h.len() >= self.cap {
                h.pop_front();
            }
            if h.back().is_some_and(|&(t, _)| t > c.tick) {
                // Out of order (e.g. after a capture-mode switch): keep the history sorted.
                let i = h.partition_point(|&(t, _)| t <= c.tick);
                h.insert(i, (c.tick, c.value));
            } else {
                h.push_back((c.tick, c.value));
            }
        }
    }

    /// Value of signal `s` at `tick` (the last change at or before it).
    pub fn value_at(&self, s: usize, tick: u64) -> Option<Packed> {
        if let Some(c) = self.config.signals.get(s)?.clock {
            return Some(Packed::from_bool(c.value(tick)));
        }
        let h = self.history.get(s)?;
        let i = h.partition_point(|&(t, _)| t <= tick);
        (i > 0).then(|| h[i - 1].1)
    }

    /// The value at `t0` (if known) followed by every change in `(t0, t1]`, for drawing.
    pub fn changes_in(&self, s: usize, t0: u64, t1: u64) -> Vec<(u64, Packed)> {
        let Some(sig) = self.config.signals.get(s) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        if let Some(c) = sig.clock {
            let mut v = c.value(t0);
            out.push((t0, Packed::from_bool(v)));
            // Walk edge to edge: at most two per period.
            let mut t = t0;
            while t < t1 && out.len() < Self::MAX_SYNTH {
                let p = u64::from(c.period);
                let pos = (t + p - u64::from(c.phase) % p) % p; // position in the period
                let next = if pos < u64::from(c.high) {
                    t + (u64::from(c.high) - pos)
                } else {
                    t + (p - pos)
                };
                if next > t1 {
                    break;
                }
                v = !v;
                out.push((next, Packed::from_bool(v)));
                t = next;
            }
            return out;
        }
        let h = &self.history[s];
        let i = h.partition_point(|&(t, _)| t <= t0);
        if i > 0 {
            out.push((t0, h[i - 1].1));
        }
        out.extend(h.iter().skip(i).take_while(|&&(t, _)| t <= t1).copied());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig(name: &str, nets: Vec<NetIdx>) -> WaveSignal {
        WaveSignal {
            name: name.into(),
            nets,
            clock: None,
        }
    }

    #[test]
    fn records_only_changes() {
        let cfg = WaveConfig {
            id: 7,
            design_gen: 1,
            signals: vec![sig("a", vec![0]), sig("b", vec![1, 2])],
        };
        let mut r = WaveRecorder::new(cfg.clone());
        use Logic::*;
        r.observe_batch(
            &[10, 11, 12, 13],
            &[
                Zero, One, Zero, Zero, One, Zero, One, One, Zero, One, X, One,
            ],
        );
        let c = r.take_chunk().unwrap();
        assert_eq!(c.config, 7);
        let got: Vec<(u64, u32)> = c.changes.iter().map(|c| (c.tick, c.signal)).collect();
        assert_eq!(got, vec![(10, 0), (10, 1), (12, 0), (13, 1)]);
        assert_eq!(c.changes[3].value.to_value(2), Value::from_bits(&[X, One]));
        assert!(r.take_chunk().is_none());

        let mut st = WaveStore::new(cfg);
        st.append(&c);
        assert_eq!(
            st.value_at(0, 11).unwrap().to_value(1),
            Value::from_bits(&[Zero])
        );
        assert_eq!(
            st.value_at(0, 12).unwrap().to_value(1),
            Value::from_bits(&[One])
        );
        assert_eq!(st.value_at(1, 9), None);
        assert_eq!(
            st.changes_in(0, 11, 20),
            vec![
                (11, Packed::from_bool(false)),
                (12, Packed::from_bool(true))
            ]
        );
    }

    #[test]
    fn clock_signals_are_synthesized_not_sampled() {
        let clk = ClockSpec {
            period: 8,
            high: 4,
            phase: 2,
        };
        let cfg = WaveConfig {
            id: 1,
            design_gen: 1,
            signals: vec![
                WaveSignal {
                    name: "clk".into(),
                    nets: vec![5],
                    clock: Some(clk),
                },
                sig("q", vec![3]),
            ],
        };
        let r = WaveRecorder::new(cfg.clone());
        assert_eq!(r.nets(), &[3]);
        let st = WaveStore::new(cfg);
        // Rises at 2, 10, ...; falls at 6, 14, ...
        for t in 0..40u64 {
            let expect = (t + 8 - 2) % 8 < 4;
            assert_eq!(clk.value(t), expect, "{t}");
            assert_eq!(st.value_at(0, t), Some(Packed::from_bool(expect)));
        }
        let ch = st.changes_in(0, 0, 20);
        let ticks: Vec<u64> = ch.iter().map(|c| c.0).collect();
        assert_eq!(ticks, vec![0, 2, 6, 10, 14, 18]);
        assert!(ch.windows(2).all(|w| w[0].1 != w[1].1));
    }
}
