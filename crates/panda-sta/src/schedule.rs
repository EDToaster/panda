//! The event schedule (SPEC §6.1, docs/M3-CONTRACT.md "Schedule"): clock edges, `PowerOnReset`
//! releases and UI input changes, grouped into the distinct *event signatures* STA analyzes.
//!
//! * **Clock events**: every tick of the hyperperiod at which some running clock changes. Events
//!   with the same (clock values before, clock values after) are one signature; its `gap` is the
//!   smallest distance to the following event over all its occurrences. `PowerOnReset` outputs are
//!   treated as *unknown but stable* (`X`) in clock events, which over-approximates both the
//!   in-reset and the out-of-reset case.
//! * **Reset events**: the release tick of every `PowerOnReset` (several releasing at the same tick
//!   form one event), together with any clock edge at that tick.
//! * **Input events**: top-level inputs change (all of them, at once) while the clocks hold the
//!   values of some phase. FastRef applies UI changes as their own event after a clock event, so
//!   one input event is analyzed per distinct clock tuple. They have no gap (the user decides when
//!   they happen); their settle time is reported.

use std::collections::{BTreeMap, BTreeSet};

use panda_netlist::Logic;
use panda_netlist::flat::{FlatNetlist, NetIdx, SourceKind};

use crate::EventKind;

/// A running clock source (`0 < high < period`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Clock {
    /// Net driven by the clock.
    pub net: NetIdx,
    /// Name of that net.
    pub name: String,
    pub period: u64,
    pub high: u64,
    pub phase: u64,
}

impl Clock {
    /// Value at tick `t` (same formula as the timing engine).
    pub(crate) fn value_at(&self, t: u64) -> bool {
        let p = self.period;
        (t % p + p - self.phase % p) % p < self.high
    }

    /// Last tick `<= t` at which the clock changes (`None` if none).
    fn last_change_at_or_before(&self, t: u64) -> Option<u64> {
        let p = self.period;
        let rise = self.phase % p;
        let fall = (self.phase + self.high) % p;
        let last = |base: u64| (t >= base).then(|| base + (t - base) / p * p);
        match (last(rise), last(fall)) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        }
    }

    /// First tick `> t` at which the clock changes.
    fn next_change_after(&self, t: u64) -> u64 {
        let p = self.period;
        let rise = self.phase % p;
        let fall = (self.phase + self.high) % p;
        next_at(rise, p, t).min(next_at(fall, p, t))
    }
}

/// Smallest `base + k·p > t`.
fn next_at(base: u64, p: u64, t: u64) -> u64 {
    if base > t {
        base
    } else {
        base + ((t - base) / p + 1) * p
    }
}

/// State of one source during an event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SrcState {
    /// Value before the event (`X` = unknown).
    pub pre: Logic,
    /// Value after the event (`X` = unknown).
    pub post: Logic,
    /// The source changes at this event.
    pub changes: bool,
}

impl SrcState {
    fn stable(v: Logic) -> SrcState {
        SrcState {
            pre: v,
            post: v,
            changes: false,
        }
    }
}

/// One event signature to analyze.
#[derive(Clone, Debug)]
pub(crate) struct EventSpec {
    pub kind: EventKind,
    /// Human-readable name, e.g. `"clk↑"`, `"reset release"`, `"inputs [clk=1]"`.
    pub name: String,
    /// Per flat source (same order as `FlatNetlist::sources`).
    pub src: Vec<SrcState>,
    /// Clock values after the event (running clocks only).
    pub clocks_after: Vec<(String, Logic)>,
    /// Smallest number of ticks to the next event (`None` = no next event / user-controlled).
    pub gap: Option<u64>,
    /// Indices (into the clock list) of clocks rising / falling at this event.
    pub rising: Vec<usize>,
    pub falling: Vec<usize>,
}

/// The analyzed schedule.
#[derive(Clone, Debug)]
pub(crate) struct Schedule {
    pub clocks: Vec<Clock>,
    pub events: Vec<EventSpec>,
    /// The hyperperiod was capped: clock events are generalized (other clocks unknown) and gaps
    /// bounded analytically.
    pub truncated: bool,
    /// Too many clocks to enumerate the generalized events: the analysis is incomplete.
    pub incomplete: bool,
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Build the schedule of `flat`. `max_hyperperiod` caps the enumerated clock window.
pub(crate) fn build(flat: &FlatNetlist, max_hyperperiod: u64, input_events: bool) -> Schedule {
    let name = |n: NetIdx| {
        flat.net_names
            .get(n as usize)
            .cloned()
            .unwrap_or_else(|| format!("n{n}"))
    };
    let mut clocks = Vec::new();
    let mut clock_of_src = vec![None; flat.sources.len()];
    let mut por_ticks = BTreeSet::new();
    let mut has_inputs = false;
    // Static state of every source in a clock event.
    let mut base = Vec::with_capacity(flat.sources.len());
    for (si, s) in flat.sources.iter().enumerate() {
        let st = match s.kind {
            SourceKind::Constant(v) => SrcState::stable(v),
            SourceKind::Clock {
                period,
                high,
                phase,
            } => {
                let (period, high, phase) = (u64::from(period), u64::from(high), u64::from(phase));
                if period > 0 && high > 0 && high < period {
                    clock_of_src[si] = Some(clocks.len());
                    clocks.push(Clock {
                        net: s.output,
                        name: name(s.output),
                        period,
                        high,
                        phase,
                    });
                    SrcState::stable(Logic::X)
                } else {
                    SrcState::stable(Logic::from_bool(high >= period && period > 0))
                }
            }
            SourceKind::PowerOnReset { ticks } => {
                if ticks == 0 {
                    SrcState::stable(Logic::Zero)
                } else {
                    por_ticks.insert(u64::from(ticks));
                    SrcState::stable(Logic::X)
                }
            }
            SourceKind::TopInput { .. } => {
                has_inputs = true;
                SrcState::stable(Logic::X)
            }
        };
        base.push(st);
    }

    let mut events = Vec::new();
    let mut truncated = false;
    // Input events see unknown clock values (hyperperiod truncated).
    let mut input_unknown_clocks = false;
    let mut incomplete = false;
    let mut tuples: BTreeSet<Vec<bool>> = BTreeSet::new();
    let edge_name = |rising: &[usize], falling: &[usize], clocks: &[Clock]| -> String {
        let mut parts: Vec<String> = Vec::new();
        for &c in rising {
            parts.push(format!("{}↑", clocks[c].name));
        }
        for &c in falling {
            parts.push(format!("{}↓", clocks[c].name));
        }
        parts.join(" ")
    };
    let make_src = |pre: &[bool], post: &[bool], base: &[SrcState]| -> Vec<SrcState> {
        let mut src = base.to_vec();
        for (si, c) in clock_of_src.iter().enumerate() {
            if let Some(c) = *c {
                src[si] = SrcState {
                    pre: Logic::from_bool(pre[c]),
                    post: Logic::from_bool(post[c]),
                    changes: pre[c] != post[c],
                };
            }
        }
        src
    };

    if !clocks.is_empty() {
        let mut h: u64 = 1;
        for c in &clocks {
            h = h / gcd(h, c.period) * c.period;
            if h > max_hyperperiod {
                h = max_hyperperiod;
                truncated = true;
                break;
            }
        }
        let mut ticks = BTreeSet::new();
        for c in clocks.iter().filter(|_| !truncated) {
            let p = c.period;
            for start in [c.phase % p, (c.phase + c.high) % p] {
                let mut t = start;
                while t < h {
                    ticks.insert(t);
                    t += p;
                }
            }
        }
        let ticks: Vec<u64> = ticks.into_iter().collect();
        // Signature → (clock values before, after; `None` = unknown) and smallest gap.
        let mut sigs: BTreeMap<SigKey, Option<u64>> = BTreeMap::new();
        if truncated {
            // The enumerated prefix need not contain every combination of clock values, nor the
            // closest approach of two clocks' edges: analyze every set of edges that can
            // coincide, with the other clocks unknown (but stable), and bound gaps analytically.
            let (gen_events, over) = generalized_events(&clocks);
            incomplete |= over;
            for (edges, gap) in gen_events {
                let mut pre = vec![None; clocks.len()];
                let mut post = vec![None; clocks.len()];
                for &(c, rising) in &edges {
                    pre[c] = Some(!rising);
                    post[c] = Some(rising);
                }
                sigs.insert((pre, post), Some(gap));
            }
        } else {
            for (i, &t) in ticks.iter().enumerate() {
                let prev = if t == 0 { h - 1 } else { t - 1 };
                let pre: Vec<bool> = clocks.iter().map(|c| c.value_at(prev)).collect();
                let post: Vec<bool> = clocks.iter().map(|c| c.value_at(t)).collect();
                if pre == post {
                    continue;
                }
                let gap = if i + 1 < ticks.len() {
                    ticks[i + 1] - t
                } else {
                    ticks[0] + h - t
                };
                let e = sigs
                    .entry((
                        pre.into_iter().map(Some).collect(),
                        post.into_iter().map(Some).collect(),
                    ))
                    .or_insert(Some(gap));
                *e = Some(e.map_or(gap, |a| a.min(gap)));
            }
        }
        // A reset release inside a phase ends that phase early: the clock event before it must
        // settle before the release.
        for &r in &por_ticks {
            let Some(t) = clocks
                .iter()
                .filter_map(|c| c.last_change_at_or_before(r - 1))
                .max()
            else {
                continue;
            };
            let prev = t.checked_sub(1).unwrap_or(h - 1);
            let pre: Vec<bool> = clocks.iter().map(|c| c.value_at(prev)).collect();
            let post: Vec<bool> = clocks.iter().map(|c| c.value_at(t)).collect();
            let key = if truncated {
                let edge_only = |v: &[bool], c: usize| (pre[c] != post[c]).then_some(v[c]);
                (
                    (0..clocks.len())
                        .map(|c| edge_only(&pre, c))
                        .collect::<Vec<_>>(),
                    (0..clocks.len())
                        .map(|c| edge_only(&post, c))
                        .collect::<Vec<_>>(),
                )
            } else {
                (
                    pre.into_iter().map(Some).collect(),
                    post.into_iter().map(Some).collect(),
                )
            };
            if let Some(g) = sigs.get_mut(&key) {
                *g = Some(g.map_or(r - t, |g| g.min(r - t)));
            }
        }
        let lit = |v: Option<bool>| v.map_or(Logic::X, Logic::from_bool);
        for ((pre, post), gap) in sigs {
            let changes = |c: usize| matches!((pre[c], post[c]), (Some(a), Some(b)) if a != b);
            let rising: Vec<usize> = (0..clocks.len())
                .filter(|&c| changes(c) && post[c] == Some(true))
                .collect();
            let falling: Vec<usize> = (0..clocks.len())
                .filter(|&c| changes(c) && post[c] == Some(false))
                .collect();
            if post.iter().all(Option::is_some) {
                tuples.insert(post.iter().map(|v| v.unwrap_or(false)).collect());
            }
            let mut src = base.clone();
            for (si, c) in clock_of_src.iter().enumerate() {
                if let Some(c) = *c {
                    src[si] = SrcState {
                        pre: lit(pre[c]),
                        post: lit(post[c]),
                        changes: changes(c),
                    };
                }
            }
            events.push(EventSpec {
                kind: EventKind::Clock,
                name: edge_name(&rising, &falling, &clocks),
                src,
                clocks_after: clocks
                    .iter()
                    .zip(&post)
                    .map(|(c, &v)| (c.name.clone(), lit(v)))
                    .collect(),
                gap,
                rising,
                falling,
            });
        }
        if truncated {
            // Input events: any combination of clock values may hold during a phase.
            tuples.clear();
            input_unknown_clocks = true;
        }
    }

    // Reset releases.
    for &r in &por_ticks {
        let pre: Vec<bool> = clocks.iter().map(|c| c.value_at(r - 1)).collect();
        let post: Vec<bool> = clocks.iter().map(|c| c.value_at(r)).collect();
        let mut src = make_src(&pre, &post, &base);
        for (si, s) in flat.sources.iter().enumerate() {
            if let SourceKind::PowerOnReset { ticks } = s.kind {
                let t = u64::from(ticks);
                if t == r {
                    src[si] = SrcState {
                        pre: Logic::One,
                        post: Logic::Zero,
                        changes: true,
                    };
                } else if t > 0 {
                    src[si] = SrcState::stable(Logic::from_bool(r < t));
                }
            }
        }
        let mut next: Option<u64> = clocks.iter().map(|c| c.next_change_after(r)).min();
        if let Some(&n) = por_ticks.range(r + 1..).next() {
            next = Some(next.map_or(n, |x| x.min(n)));
        }
        let rising: Vec<usize> = (0..clocks.len()).filter(|&c| !pre[c] && post[c]).collect();
        let falling: Vec<usize> = (0..clocks.len()).filter(|&c| pre[c] && !post[c]).collect();
        let mut name = String::from("reset release");
        let edges = edge_name(&rising, &falling, &clocks);
        if !edges.is_empty() {
            name.push_str(" + ");
            name.push_str(&edges);
        }
        events.push(EventSpec {
            kind: EventKind::Reset,
            name,
            src,
            clocks_after: clocks
                .iter()
                .zip(&post)
                .map(|(c, &v)| (c.name.clone(), Logic::from_bool(v)))
                .collect(),
            gap: next.map(|n| n - r),
            rising,
            falling,
        });
    }

    // UI input changes, one per clock tuple.
    if input_events && has_inputs {
        if input_unknown_clocks {
            let mut src = base.clone();
            for (si, s) in flat.sources.iter().enumerate() {
                if let SourceKind::TopInput { .. } = s.kind {
                    src[si] = SrcState {
                        pre: Logic::X,
                        post: Logic::X,
                        changes: true,
                    };
                }
            }
            events.push(EventSpec {
                kind: EventKind::Inputs,
                name: "inputs [clocks unknown]".to_string(),
                src,
                clocks_after: clocks.iter().map(|c| (c.name.clone(), Logic::X)).collect(),
                gap: None,
                rising: Vec::new(),
                falling: Vec::new(),
            });
        } else if tuples.is_empty() {
            tuples.insert(Vec::new());
        }
        for tuple in tuples {
            let mut src = make_src(&tuple, &tuple, &base);
            for (si, s) in flat.sources.iter().enumerate() {
                if let SourceKind::TopInput { .. } = s.kind {
                    src[si] = SrcState {
                        pre: Logic::X,
                        post: Logic::X,
                        changes: true,
                    };
                }
            }
            let desc: Vec<String> = clocks
                .iter()
                .zip(&tuple)
                .map(|(c, &v)| format!("{}={}", c.name, u8::from(v)))
                .collect();
            let name = if desc.is_empty() {
                "inputs".to_string()
            } else {
                format!("inputs [{}]", desc.join(", "))
            };
            events.push(EventSpec {
                kind: EventKind::Inputs,
                name,
                src,
                clocks_after: clocks
                    .iter()
                    .zip(&tuple)
                    .map(|(c, &v)| (c.name.clone(), Logic::from_bool(v)))
                    .collect(),
                gap: None,
                rising: Vec::new(),
                falling: Vec::new(),
            });
        }
    }
    Schedule {
        clocks,
        events,
        truncated,
        incomplete,
    }
}

/// A clock edge: `(clock index, rising)`.
type Edge = (usize, bool);

/// An event signature: clock values before and after (`None` = unknown).
type SigKey = (Vec<Option<bool>>, Vec<Option<bool>>);

/// Edge `(clock, rising)` offsets: the edge happens at ticks `offset + k·period`.
fn edge_offset(c: &Clock, rising: bool) -> u64 {
    let p = c.period;
    if rising {
        c.phase % p
    } else {
        (c.phase + c.high) % p
    }
}

/// Smallest positive distance from an `a`-edge to a later `b`-edge (both of the form
/// `offset + k·period`): the differences form `(ob - oa) + gcd·ℤ`; a difference of 0 means the
/// edges coincide (one event), so the next one is a whole `gcd` later.
fn min_edge_gap(a: (u64, u64), b: (u64, u64)) -> u64 {
    let (oa, pa) = a;
    let (ob, pb) = b;
    let g = gcd(pa, pb);
    let r = (ob % g + g - oa % g) % g;
    if r == 0 { g } else { r }
}

/// Every set of clock edges (at most one per clock) that can happen at the same tick, with a
/// lower bound on the distance to the next event after it. Two edges can coincide iff their
/// offsets agree modulo the gcd of their periods, and a set of edges can coincide iff every pair
/// can (generalized Chinese remainder theorem). Used when the hyperperiod is too long to enumerate.
fn generalized_events(clocks: &[Clock]) -> (Vec<(Vec<Edge>, u64)>, bool) {
    let all: Vec<Edge> = (0..clocks.len())
        .flat_map(|c| [(c, true), (c, false)])
        .collect();
    let key = |e: Edge| (edge_offset(&clocks[e.0], e.1), clocks[e.0].period);
    let compatible = |a: Edge, b: Edge| {
        let ((oa, pa), (ob, pb)) = (key(a), key(b));
        let g = gcd(pa, pb);
        a.0 != b.0 && oa % g == ob % g
    };
    // Gap bound of a single edge: nearest later edge of any clock.
    let gap_of = |e: Edge| {
        all.iter()
            .map(|&f| min_edge_gap(key(e), key(f)))
            .min()
            .unwrap_or(u64::MAX)
    };
    let mut out = Vec::new();
    // Depth-first enumeration of compatible sets in index order.
    let mut stack: Vec<(Vec<Edge>, usize)> = vec![(Vec::new(), 0)];
    while let Some((set, next)) = stack.pop() {
        if !set.is_empty() {
            let gap = set.iter().map(|&e| gap_of(e)).min().unwrap_or(u64::MAX);
            out.push((set.clone(), gap));
        }
        if out.len() > MAX_GENERALIZED_EVENTS {
            return (out, true);
        }
        for (i, &e) in all.iter().enumerate().skip(next) {
            if set.iter().all(|&f| compatible(e, f)) {
                let mut s2 = set.clone();
                s2.push(e);
                stack.push((s2, i + 1));
            }
        }
    }
    (out, false)
}

/// Cap on the generalized event signatures of a truncated schedule.
const MAX_GENERALIZED_EVENTS: usize = 4096;

/// Smallest even period `p` for `clock` such that an event of settle time `s_rise` after each
/// rising edge and `s_fall` after each falling edge settles before the next edge, keeping the
/// clock's high/period ratio. `None` settle = unbounded (no period works → `None`).
pub(crate) fn min_period(clock: &Clock, s_rise: Option<u32>, s_fall: Option<u32>) -> Option<u32> {
    let (s_rise, s_fall) = (u64::from(s_rise?), u64::from(s_fall?));
    let mut p: u64 = 2;
    while p <= u64::from(u32::MAX) {
        let h = ((p * clock.high * 2 + clock.period) / (clock.period * 2)).clamp(1, p - 1);
        if s_rise < h && s_fall < p - h {
            return Some(p as u32);
        }
        // Jump close to the answer instead of stepping one by one from 2.
        let need = (s_rise + 1).max(s_fall + 1);
        p = if p < need { need + need % 2 } else { p + 2 };
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clk(period: u64, high: u64, phase: u64) -> Clock {
        Clock {
            net: 0,
            name: "clk".into(),
            period,
            high,
            phase,
        }
    }

    #[test]
    fn value_and_next_change() {
        let c = clk(6, 2, 1);
        let v: Vec<bool> = (0..8).map(|t| c.value_at(t)).collect();
        assert_eq!(v, [false, true, true, false, false, false, false, true]);
        assert_eq!(c.next_change_after(0), 1);
        assert_eq!(c.next_change_after(1), 3);
        assert_eq!(c.next_change_after(3), 7);
    }

    /// A flat netlist with just the given sources (no gates).
    fn sources_only(kinds: &[SourceKind]) -> FlatNetlist {
        let mut f = FlatNetlist {
            num_nets: kinds.len() as u32,
            ..FlatNetlist::default()
        };
        for (i, k) in kinds.iter().enumerate() {
            f.sources.push(panda_netlist::flat::Source {
                kind: k.clone(),
                output: i as NetIdx,
            });
            f.net_names.push(format!("s{i}"));
        }
        f
    }

    #[test]
    fn single_clock_with_reset_release_inside_a_phase() {
        // Clock: period 16, rises at 8 + 16k, falls at 16k. Reset released at 36 (falling edge at
        // 32 → the falling-edge phase is cut to 4 ticks; rising edge follows at 40).
        let f = sources_only(&[
            SourceKind::Clock {
                period: 16,
                high: 8,
                phase: 8,
            },
            SourceKind::PowerOnReset { ticks: 36 },
            SourceKind::Constant(Logic::One),
        ]);
        let s = build(&f, 1 << 20, true);
        assert_eq!(s.clocks.len(), 1);
        assert!(!s.truncated);
        let ev = |name: &str| {
            s.events
                .iter()
                .find(|e| e.name == name)
                .unwrap_or_else(|| panic!("{name}"))
        };
        let rise = ev("s0↑");
        assert_eq!(rise.gap, Some(8));
        assert_eq!(
            rise.src[0],
            SrcState {
                pre: Logic::Zero,
                post: Logic::One,
                changes: true
            }
        );
        assert_eq!(
            rise.src[1].pre,
            Logic::X,
            "reset is unknown in clock events"
        );
        assert_eq!(rise.src[2], SrcState::stable(Logic::One));
        assert_eq!(
            ev("s0↓").gap,
            Some(4),
            "the reset release cuts the low phase"
        );
        let rel = ev("reset release");
        assert_eq!(rel.kind, EventKind::Reset);
        assert_eq!(rel.gap, Some(4));
        assert_eq!(
            rel.src[1],
            SrcState {
                pre: Logic::One,
                post: Logic::Zero,
                changes: true
            }
        );
        assert!(!rel.src[0].changes);
        // No top-level inputs → no input events.
        assert!(s.events.iter().all(|e| e.kind != EventKind::Inputs));
    }

    #[test]
    fn two_clocks_and_coincident_reset() {
        let f = sources_only(&[
            SourceKind::Clock {
                period: 6,
                high: 3,
                phase: 0,
            },
            SourceKind::Clock {
                period: 4,
                high: 2,
                phase: 0,
            },
            SourceKind::PowerOnReset { ticks: 12 },
            SourceKind::TopInput { pin: 0, bit: 0 },
        ]);
        let s = build(&f, 1 << 20, true);
        // Hyperperiod 12: edges at 0 (both rise), 2, 3, 4, 6, 8, 9, 10.
        let clock_events: Vec<_> = s
            .events
            .iter()
            .filter(|e| e.kind == EventKind::Clock)
            .collect();
        let total: usize = clock_events.len();
        assert!(total >= 5, "{total}");
        let both = clock_events
            .iter()
            .find(|e| e.rising.len() == 2)
            .expect("simultaneous rise");
        assert_eq!(both.name, "s0↑ s1↑");
        assert_eq!(both.gap, Some(2));
        assert!(
            clock_events
                .iter()
                .all(|e| e.gap.is_some_and(|g| (1..=2).contains(&g)))
        );
        // Reset at 12 coincides with both clocks rising.
        let rel = s
            .events
            .iter()
            .find(|e| e.kind == EventKind::Reset)
            .unwrap();
        assert_eq!(rel.name, "reset release + s0↑ s1↑");
        assert_eq!(rel.gap, Some(2));
        // Input events: one per distinct clock tuple after an event.
        let inputs: Vec<_> = s
            .events
            .iter()
            .filter(|e| e.kind == EventKind::Inputs)
            .collect();
        assert!(inputs.len() >= 3);
        assert!(
            inputs
                .iter()
                .all(|e| e.src[3].changes && !e.src[0].changes && e.gap.is_none())
        );
        // Disabled input events.
        assert!(
            build(&f, 1 << 20, false)
                .events
                .iter()
                .all(|e| e.kind != EventKind::Inputs)
        );
        // Capped hyperperiod.
        assert!(build(&f, 5, false).truncated);
    }

    #[test]
    fn min_period_respects_duty() {
        let c = clk(10, 5, 5);
        // 50 %: need high > 9 and low > 3 → p = 20.
        assert_eq!(min_period(&c, Some(9), Some(3)), Some(20));
        assert_eq!(min_period(&c, Some(3), Some(9)), Some(20));
        assert_eq!(min_period(&c, Some(0), Some(0)), Some(2));
        assert_eq!(min_period(&c, None, Some(0)), None);
        // 25 % duty: high = round(p/4) > 9 → p = 38 (high 10).
        let q = clk(8, 2, 0);
        assert_eq!(min_period(&q, Some(9), Some(1)), Some(38));
    }

    #[test]
    fn edge_gaps_are_exact_over_all_time() {
        // Brute force over two hyperperiods vs the analytic bound.
        let clocks = [clk(200, 100, 0), clk(202, 101, 50), clk(12, 4, 3)];
        for (ia, a) in clocks.iter().enumerate() {
            for (ib, b) in clocks.iter().enumerate() {
                for (ra, rb) in [(true, true), (true, false), (false, true), (false, false)] {
                    let (oa, ob) = (edge_offset(a, ra), edge_offset(b, rb));
                    let mut best = u64::MAX;
                    for i in 0..300u64 {
                        let ta = oa + i * a.period;
                        for j in 0..300u64 {
                            let tb = ob + j * b.period;
                            if tb > ta && !(ia == ib && ra == rb && tb == ta) {
                                best = best.min(tb - ta);
                            }
                        }
                    }
                    let g = min_edge_gap((oa, a.period), (ob, b.period));
                    // The analytic bound is a lower bound, attained when the edges can coincide
                    // or approach (always, for a long enough window).
                    assert!(g <= best, "{ia}/{ib} {ra}/{rb}: {g} > {best}");
                    assert_eq!(g, best, "{ia}/{ib} {ra}/{rb}");
                }
            }
        }
    }

    #[test]
    fn truncated_schedule_generalizes_events() {
        let f = sources_only(&[
            SourceKind::Clock {
                period: 200,
                high: 100,
                phase: 0,
            },
            SourceKind::Clock {
                period: 202,
                high: 101,
                phase: 50,
            },
            SourceKind::TopInput { pin: 0, bit: 0 },
        ]);
        let s = build(&f, 4096, true);
        assert!(s.truncated && !s.incomplete);
        let clock_events: Vec<_> = s
            .events
            .iter()
            .filter(|e| e.kind == EventKind::Clock)
            .collect();
        // Edge offsets 0 (s0↑), 100 (s0↓), 50 (s1↑), 151 (s1↓), gcd of the periods 2: s1↑ can
        // coincide with either edge of s0, s1↓ with none. 4 single edges + 2 pairs.
        let mut names: Vec<&str> = clock_events.iter().map(|e| e.name.as_str()).collect();
        names.sort_unstable();
        assert_eq!(names, ["s0↑", "s0↑ s1↑", "s0↓", "s1↑", "s1↑ s0↓", "s1↓"]);
        // Some edge of the other clock comes within 1 or 2 ticks (somewhere in the hyperperiod).
        assert!(clock_events.iter().all(|e| e.gap.is_some_and(|g| g <= 2)));
        let single = clock_events.iter().find(|e| e.name == "s0↑").unwrap();
        assert_eq!(single.src[1].pre, Logic::X);
        assert!(!single.src[1].changes);
        // Input events: one, with unknown clocks.
        let inputs: Vec<_> = s
            .events
            .iter()
            .filter(|e| e.kind == EventKind::Inputs)
            .collect();
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].src[0].pre, Logic::X);
    }
}
