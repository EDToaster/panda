//! Time-varying sources and the event schedule (identical to FastRef's: event ticks are 0, every
//! tick at which some `Clock` output differs from its value at the previous tick, and every
//! `PowerOnReset` release tick).

use panda_netlist::Logic;

/// A time-varying source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Timed {
    /// `Clock { period, high, phase }` (ticks).
    Clock {
        /// Period.
        period: u64,
        /// High time.
        high: u64,
        /// Rising-edge offset.
        phase: u64,
    },
    /// `PowerOnReset` high for ticks `0..ticks`.
    PowerOnReset {
        /// Release tick.
        ticks: u64,
    },
}

impl Timed {
    /// Value at tick `t` (same formula as the timing engine).
    pub fn value_at(self, t: u64) -> Logic {
        match self {
            Timed::Clock {
                period,
                high,
                phase,
            } => {
                if period == 0 {
                    return Logic::from_bool(high > 0);
                }
                Logic::from_bool((t % period + period - phase % period) % period < high)
            }
            Timed::PowerOnReset { ticks } => Logic::from_bool(t < ticks),
        }
    }

    /// First tick `> t` at which the value differs from the value at the previous tick.
    pub fn next_change_after(self, t: u64) -> Option<u64> {
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

    /// First rising edge (tick whose value is 1 while the previous tick's is 0) strictly after
    /// `t`, for a running clock.
    pub fn next_rising_after(self, t: u64) -> Option<u64> {
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
                t.checked_add(period - r)
            }
            Timed::PowerOnReset { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_schedule_brute_force() {
        for (period, high, phase) in [
            (7u64, 3u64, 2u64),
            (4, 1, 0),
            (6, 5, 13),
            (2, 1, 1),
            (10, 5, 5),
        ] {
            let c = Timed::Clock {
                period,
                high,
                phase,
            };
            for t in 0..60u64 {
                let want = (t + 1..t + 100).find(|&u| c.value_at(u) != c.value_at(u - 1));
                assert_eq!(c.next_change_after(t), want);
                let rise = (t + 1..t + 100)
                    .find(|&u| c.value_at(u) == Logic::One && c.value_at(u - 1) == Logic::Zero);
                assert_eq!(
                    c.next_rising_after(t),
                    rise,
                    "{period} {high} {phase} t={t}"
                );
            }
        }
        let flat = Timed::Clock {
            period: 8,
            high: 0,
            phase: 0,
        };
        assert_eq!(flat.next_change_after(3), None);
        assert_eq!(flat.next_rising_after(3), None);
    }

    #[test]
    fn por_schedule() {
        let p = Timed::PowerOnReset { ticks: 20 };
        assert_eq!(p.next_change_after(0), Some(20));
        assert_eq!(p.next_change_after(20), None);
        assert_eq!(Timed::PowerOnReset { ticks: 0 }.next_change_after(0), None);
    }
}
