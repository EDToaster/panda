//! The source schedule of a flat netlist: when do `Clock` and `PowerOnReset` sources change?
//!
//! Uses exactly the formulas of the engines (`Atom::Clock` documentation): a clock with
//! `(period, high, phase)` is 1 at tick `t` iff `(t + period - phase % period) % period < high`;
//! a `PowerOnReset` with `ticks` is 1 for `t < ticks`.

use panda_netlist::flat::{FlatNetlist, NetIdx, SourceKind};

/// A time-varying source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Timed {
    Clock { period: u64, high: u64, phase: u64 },
    PowerOnReset { ticks: u64 },
}

impl Timed {
    /// Value at tick `t` (1 = true).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn value_at(self, t: u64) -> bool {
        match self {
            Timed::Clock {
                period,
                high,
                phase,
            } => {
                if period == 0 {
                    return high > 0;
                }
                (t % period + period - phase % period) % period < high
            }
            Timed::PowerOnReset { ticks } => t < ticks,
        }
    }

    /// First tick `> t` at which the value differs from the value at the previous tick.
    pub(crate) fn next_change_after(self, t: u64) -> Option<u64> {
        match self {
            Timed::Clock {
                period,
                high,
                phase,
            } => {
                if period == 0 || high == 0 || high >= period {
                    return None;
                }
                let r = (t % period + period - phase % period) % period;
                let dist = |c: u64| (c + period - r - 1) % period + 1;
                t.checked_add(dist(0).min(dist(high)))
            }
            Timed::PowerOnReset { ticks } => (ticks > t).then_some(ticks),
        }
    }

    /// First rising edge (0 → 1) strictly after `t` (clocks only).
    pub(crate) fn next_rise_after(self, t: u64) -> Option<u64> {
        match self {
            Timed::Clock {
                period,
                high,
                phase,
            } => {
                if period == 0 || high == 0 || high >= period {
                    return None;
                }
                let r = (t % period + period - phase % period) % period;
                t.checked_add((period - r - 1) % period + 1)
            }
            Timed::PowerOnReset { .. } => None,
        }
    }
}

/// All time-varying sources of a netlist.
#[derive(Clone, Debug, Default)]
pub(crate) struct Schedule {
    /// `(source net, source)`.
    pub(crate) timed: Vec<(NetIdx, Timed)>,
    /// Index into `timed` of the reference clock used by `StepUnit::Cycle`.
    pub(crate) cycle_clock: Option<usize>,
}

impl Schedule {
    /// The schedule of `flat`. `cycle_clock` names the clock net used for `StepUnit::Cycle`
    /// (default: the first clock source).
    pub(crate) fn new(flat: &FlatNetlist, cycle_clock: Option<&str>) -> Schedule {
        let mut timed = Vec::new();
        for s in &flat.sources {
            match s.kind {
                SourceKind::Clock {
                    period,
                    high,
                    phase,
                } => timed.push((
                    s.output,
                    Timed::Clock {
                        period: u64::from(period),
                        high: u64::from(high),
                        phase: u64::from(phase),
                    },
                )),
                SourceKind::PowerOnReset { ticks } => {
                    timed.push((
                        s.output,
                        Timed::PowerOnReset {
                            ticks: u64::from(ticks),
                        },
                    ));
                }
                _ => {}
            }
        }
        let is_clock = |t: &Timed| matches!(t, Timed::Clock { .. });
        let wanted = cycle_clock.and_then(|name| flat.net_by_name(name));
        let cycle_clock = wanted
            .and_then(|net| timed.iter().position(|(n, t)| *n == net && is_clock(t)))
            .or_else(|| timed.iter().position(|(_, t)| is_clock(t)));
        Schedule { timed, cycle_clock }
    }

    /// First source change strictly after `t`.
    pub(crate) fn next_change_after(&self, t: u64) -> Option<u64> {
        self.timed
            .iter()
            .filter_map(|&(_, s)| s.next_change_after(t))
            .min()
    }

    /// First rising edge of the reference clock strictly after `t`.
    pub(crate) fn next_cycle_edge_after(&self, t: u64) -> Option<u64> {
        let (_, c) = self.timed[self.cycle_clock?];
        c.next_rise_after(t)
    }

    /// Number of rising edges of the reference clock in `(from, to]`.
    pub(crate) fn cycle_edges_between(&self, from: u64, to: u64) -> u64 {
        let Some(ci) = self.cycle_clock else { return 0 };
        let Timed::Clock { period, .. } = self.timed[ci].1 else {
            return 0;
        };
        match self.timed[ci].1.next_rise_after(from) {
            Some(first) if first <= to => 1 + (to - first) / period,
            _ => 0,
        }
    }

    /// Smallest tick `t0 ≤ t` such that every source has the same value and the same future at
    /// `t0` as at `t` (`t0 ≡ t` modulo the clocks' hyperperiod, and at or after every
    /// `PowerOnReset` release that happened by `t`). Used to start a fresh timing engine "at" tick
    /// `t`: it is run from 0 to `t0` and the offset `t - t0` is added to its tick.
    pub(crate) fn aligned_start(&self, t: u64) -> u64 {
        let mut hyper: u64 = 1;
        let mut por_max: u64 = 0;
        for &(_, s) in &self.timed {
            match s {
                Timed::Clock { period, high, .. } if period > 0 && high > 0 && high < period => {
                    let g = gcd(hyper, period);
                    hyper = match (hyper / g).checked_mul(period) {
                        Some(h) => h,
                        None => return t,
                    };
                }
                Timed::Clock { .. } => {}
                Timed::PowerOnReset { ticks } => por_max = por_max.max(ticks),
            }
        }
        if t <= por_max {
            return t;
        }
        por_max + (t - por_max) % hyper
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_edges() {
        let c = Timed::Clock {
            period: 10,
            high: 5,
            phase: 5,
        };
        assert!(!c.value_at(0));
        assert!(c.value_at(5));
        assert_eq!(c.next_change_after(0), Some(5));
        assert_eq!(c.next_change_after(5), Some(10));
        assert_eq!(c.next_rise_after(0), Some(5));
        assert_eq!(c.next_rise_after(5), Some(15));
        assert_eq!(c.next_rise_after(4), Some(5));
        for (period, high, phase) in [(7u64, 3u64, 2u64), (4, 1, 0), (6, 5, 13), (2, 1, 1)] {
            let c = Timed::Clock {
                period,
                high,
                phase,
            };
            for t in 0..50u64 {
                let want = (t + 1..t + 100).find(|&u| c.value_at(u) != c.value_at(u - 1));
                assert_eq!(c.next_change_after(t), want);
                let rise = (t + 1..t + 100).find(|&u| c.value_at(u) && !c.value_at(u - 1));
                assert_eq!(c.next_rise_after(t), rise);
            }
        }
    }

    #[test]
    fn aligned_start_matches_values() {
        let s = Schedule {
            timed: vec![
                (
                    0,
                    Timed::Clock {
                        period: 6,
                        high: 3,
                        phase: 1,
                    },
                ),
                (
                    1,
                    Timed::Clock {
                        period: 4,
                        high: 1,
                        phase: 0,
                    },
                ),
                (2, Timed::PowerOnReset { ticks: 17 }),
            ],
            cycle_clock: Some(0),
        };
        for t in 0..200 {
            let t0 = s.aligned_start(t);
            assert!(t0 <= t);
            for &(_, src) in &s.timed {
                for d in 0..30 {
                    assert_eq!(src.value_at(t0 + d), src.value_at(t + d), "t={t} t0={t0}");
                }
            }
        }
        assert_eq!(s.cycle_edges_between(0, 1), 1);
        assert_eq!(s.cycle_edges_between(1, 7), 1);
        assert_eq!(s.cycle_edges_between(1, 13), 2);
        assert_eq!(s.cycle_edges_between(13, 13), 0);
    }
}
