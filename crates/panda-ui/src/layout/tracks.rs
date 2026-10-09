//! Channel routing: vertical tracks between two columns.
//!
//! A channel has *points* on its left side (right-facing ports and lane ends of the column on
//! its left) and its right side (left-facing ports and lane starts of the column on its right).
//! A net whose points in the channel lie at one height crosses it straight; every other net gets
//! a vertical *track* with horizontal stubs to its points.
//!
//! Constraints and goals:
//!
//! * **Hard** (no shorts): if net `a` has a left point and net `b` a right point at the same
//!   height, `a`'s track must lie left of `b`'s (their stubs would overlap otherwise). Cycles of
//!   these constraints are broken by splitting a net into two vertical *parts* joined by a
//!   horizontal jog on a free row (a dogleg).
//! * **Soft** (crossings): for two parts whose spans overlap, the order of their tracks decides
//!   how many stubs cross the other's vertical; the cheaper order is kept when the parts are
//!   packed. Parts are packed in a topological order of the hard constraints that prefers the
//!   bundle-friendly order (going down: the upper net right; going up: the upper net left), so
//!   parallel buses come out as crossing-free staircases.
//! * **Compact**: each part takes the lowest free track satisfying its constraints, so parts
//!   with disjoint spans share tracks (left-edge packing).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// A point of a net on one side of a channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChanPt {
    /// Net index.
    pub net: usize,
    /// Height.
    pub y: i32,
    /// On the channel's left side.
    pub left: bool,
}

/// One routed piece of a channel, in channel coordinates (`t` = track index).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Piece {
    /// Straight across the channel at `y`.
    Across { net: usize, y: i32 },
    /// Vertical on track `t` from `y0` to `y1`.
    Vertical { net: usize, t: i32, y0: i32, y1: i32 },
    /// From the left side to track `t` at `y`.
    LeftStub { net: usize, t: i32, y: i32 },
    /// From track `t` to the right side at `y`.
    RightStub { net: usize, t: i32, y: i32 },
    /// Between tracks `t0` and `t1` at `y` (dogleg).
    Jog { net: usize, t0: i32, t1: i32, y: i32 },
}

/// A routed channel.
#[derive(Clone, Debug, Default)]
pub struct ChannelRoute {
    /// Number of tracks used.
    pub tracks: i32,
    /// Pieces.
    pub pieces: Vec<Piece>,
}

/// One vertical part of a net in the channel.
#[derive(Clone, Debug)]
struct Part {
    net: usize,
    /// Points `(y, left)` served by this part (jog ends excluded).
    pts: Vec<(i32, bool)>,
    y0: i32,
    y1: i32,
    /// Bundle-order key (smaller = further left).
    key: f64,
}

impl Part {
    fn new(net: usize, pts: Vec<(i32, bool)>, extra: &[i32]) -> Part {
        let ys = pts.iter().map(|p| p.0).chain(extra.iter().copied());
        let (y0, y1) = ys.fold((i32::MAX, i32::MIN), |(a, b), y| (a.min(y), b.max(y)));
        let mean = |left: bool| {
            let v: Vec<f64> = pts
                .iter()
                .filter(|p| p.1 == left)
                .map(|p| f64::from(p.0))
                .collect();
            (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
        };
        // Going down (right side lower): upper parts to the right, i.e. key = −top.
        // Going up: upper parts to the left, key = top. Mixed or one-sided parts sit between.
        let key = match (mean(true), mean(false)) {
            (Some(l), Some(r)) if r > l => -f64::from(y0) - 1e7,
            (Some(l), Some(r)) if r < l => f64::from(y0) + 1e7,
            _ => f64::from(y0),
        };
        Part {
            net,
            pts,
            y0,
            y1,
            key,
        }
    }

    fn overlaps(&self, o: &Part) -> bool {
        !(self.y1 < o.y0 || self.y0 > o.y1)
    }
}

/// Stubs of `a` crossing `b`'s vertical when `a`'s track is left of `b`'s.
fn cost_left_of(a: &Part, b: &Part) -> usize {
    // a's right stubs run right past b's track; b's left stubs run left past a's track.
    let inside = |y: i32, p: &Part| y > p.y0 && y < p.y1;
    a.pts.iter().filter(|&&(y, l)| !l && inside(y, b)).count()
        + b.pts.iter().filter(|&&(y, l)| l && inside(y, a)).count()
}

/// Route one channel.
pub fn route_channel(pts: &[ChanPt]) -> ChannelRoute {
    let mut by_net: HashMap<usize, Vec<(i32, bool)>> = HashMap::new();
    for p in pts {
        by_net.entry(p.net).or_default().push((p.y, p.left));
    }
    let mut nets: Vec<(usize, Vec<(i32, bool)>)> = by_net.into_iter().collect();
    nets.sort_unstable_by_key(|n| n.0);
    let mut out = ChannelRoute::default();
    let mut parts: Vec<Part> = Vec::new();
    for (net, mut ps) in nets {
        ps.sort_unstable();
        ps.dedup();
        let (lo, hi) = (ps[0].0, ps[ps.len() - 1].0);
        if lo == hi {
            if ps.iter().any(|p| p.1) && ps.iter().any(|p| !p.1) {
                out.pieces.push(Piece::Across { net, y: lo });
            }
            continue;
        }
        parts.push(Part::new(net, ps, &[]));
    }
    if parts.is_empty() {
        return out;
    }
    // Rows holding any point (jogs must avoid them).
    let mut busy: Vec<i32> = pts.iter().map(|p| p.y).collect();
    busy.sort_unstable();
    busy.dedup();
    // Jogs: (part a, part b, y) joining two parts of one net.
    let mut jogs: Vec<(usize, usize, i32)> = Vec::new();
    let mut succ = constraints(&parts);
    for _ in 0..parts.len() {
        let Some(cycle) = find_cycle(&succ) else {
            break;
        };
        if !split_for_cycle(&mut parts, &mut jogs, &cycle, &mut busy) {
            break;
        }
        succ = constraints(&parts);
    }
    let n = parts.len();
    let mut indeg = vec![0usize; n];
    for ss in &succ {
        for &b in ss {
            indeg[b] += 1;
        }
    }
    // Kahn's order preferring small keys (cycles left over: take the smallest pending key).
    struct K(f64, usize);
    impl PartialEq for K {
        fn eq(&self, o: &Self) -> bool {
            self.cmp(o).is_eq()
        }
    }
    impl Eq for K {}
    impl PartialOrd for K {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for K {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            self.0.total_cmp(&o.0).then(self.1.cmp(&o.1))
        }
    }
    let mut heap: BinaryHeap<Reverse<K>> = (0..n)
        .filter(|&i| indeg[i] == 0)
        .map(|i| Reverse(K(parts[i].key, i)))
        .collect();
    let mut placed = vec![false; n];
    let mut track = vec![0i32; n];
    let mut lb = vec![0i32; n];
    // Occupied spans per track.
    let mut used: Vec<Vec<(i32, i32)>> = Vec::new();
    // Parts placed so far, sorted by y0 (for the soft order scan).
    let mut done: Vec<usize> = Vec::with_capacity(n);
    let mut count = 0;
    while count < n {
        let i = match heap.pop() {
            Some(Reverse(K(_, i))) if !placed[i] => i,
            Some(_) => continue,
            None => (0..n)
                .filter(|&i| !placed[i])
                .min_by(|&a, &b| parts[a].key.total_cmp(&parts[b].key))
                .expect("pending part"),
        };
        placed[i] = true;
        count += 1;
        let p = &parts[i];
        // Right of every overlapping placed part that is cheaper on the left (soft order).
        let mut t = lb[i];
        for &j in &done {
            let q = &parts[j];
            if q.overlaps(p) && track[j] >= t && cost_left_of(q, p) < cost_left_of(p, q) {
                t = track[j] + 1;
            }
        }
        // Lowest free track from there.
        while (t as usize) < used.len()
            && !used[t as usize].iter().all(|&(c, d)| p.y1 < c || p.y0 > d)
        {
            t += 1;
        }
        if used.len() <= t as usize {
            used.resize(t as usize + 1, Vec::new());
        }
        used[t as usize].push((p.y0, p.y1));
        track[i] = t;
        done.push(i);
        for &s in &succ[i] {
            lb[s] = lb[s].max(t + 1);
            indeg[s] -= 1;
            if indeg[s] == 0 && !placed[s] {
                heap.push(Reverse(K(parts[s].key, s)));
            }
        }
    }
    out.tracks = used.len() as i32;
    for (i, p) in parts.iter().enumerate() {
        let t = track[i];
        out.pieces.push(Piece::Vertical {
            net: p.net,
            t,
            y0: p.y0,
            y1: p.y1,
        });
        for &(y, left) in &p.pts {
            out.pieces.push(if left {
                Piece::LeftStub { net: p.net, t, y }
            } else {
                Piece::RightStub { net: p.net, t, y }
            });
        }
    }
    for &(a, b, y) in &jogs {
        out.pieces.push(Piece::Jog {
            net: parts[a].net,
            t0: track[a],
            t1: track[b],
            y,
        });
    }
    out
}

/// Hard constraints between parts: `a → b` if `a` has a left point and `b` (another net) a
/// right point at the same height.
fn constraints(parts: &[Part]) -> Vec<Vec<usize>> {
    let mut left_at: HashMap<i32, Vec<usize>> = HashMap::new();
    let mut right_at: HashMap<i32, Vec<usize>> = HashMap::new();
    for (i, p) in parts.iter().enumerate() {
        for &(y, l) in &p.pts {
            if l {
                left_at.entry(y).or_default().push(i);
            } else {
                right_at.entry(y).or_default().push(i);
            }
        }
    }
    let mut succ = vec![Vec::new(); parts.len()];
    for (y, ls) in &left_at {
        if let Some(rs) = right_at.get(y) {
            for &a in ls {
                for &b in rs {
                    if parts[a].net != parts[b].net {
                        succ[a].push(b);
                    }
                }
            }
        }
    }
    for s in &mut succ {
        s.sort_unstable();
        s.dedup();
    }
    succ
}

/// Some cycle of the constraint graph (as a node list), if any.
fn find_cycle(succ: &[Vec<usize>]) -> Option<Vec<usize>> {
    let n = succ.len();
    let mut state = vec![0u8; n];
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for r in 0..n {
        if state[r] != 0 {
            continue;
        }
        state[r] = 1;
        stack.push((r, 0));
        while let Some(&mut (v, ref mut i)) = stack.last_mut() {
            if *i < succ[v].len() {
                let w = succ[v][*i];
                *i += 1;
                match state[w] {
                    0 => {
                        state[w] = 1;
                        stack.push((w, 0));
                    }
                    1 => {
                        let from = stack.iter().position(|&(u, _)| u == w).expect("on stack");
                        return Some(stack[from..].iter().map(|&(u, _)| u).collect());
                    }
                    _ => {}
                }
            } else {
                state[v] = 2;
                stack.pop();
            }
        }
    }
    None
}

/// Break `cycle` by splitting one of its parts in two: one keeps a left point that orders it
/// before its successor, the other a right point that orders it after its predecessor; a jog
/// on a free row (nearest to the two points, preferably between them) joins them. Returns
/// false if no part of the cycle can be split.
fn split_for_cycle(
    parts: &mut Vec<Part>,
    jogs: &mut Vec<(usize, usize, i32)>,
    cycle: &[usize],
    busy: &mut Vec<i32>,
) -> bool {
    let k = cycle.len();
    for ci in 0..k {
        let x = cycle[ci];
        let next = cycle[(ci + 1) % k];
        let prev = cycle[(ci + k - 1) % k];
        // x → next: x's left point at a height where next has a right point.
        let yl = parts[x]
            .pts
            .iter()
            .find(|&&(y, l)| l && parts[next].pts.contains(&(y, false)))
            .map(|p| p.0);
        // prev → x: x's right point at a height where prev has a left point.
        let yr = parts[x]
            .pts
            .iter()
            .find(|&&(y, l)| !l && parts[prev].pts.contains(&(y, true)))
            .map(|p| p.0);
        let (Some(yl), Some(yr)) = (yl, yr) else {
            continue;
        };
        let (lo, hi) = (yl.min(yr), yl.max(yr));
        let free = |y: &i32| busy.binary_search(y).is_err();
        let ym = (lo + 1..hi).find(free).or_else(|| {
            (1..256).find_map(|d| [lo - d, hi + d].into_iter().find(free))
        });
        let Some(ym) = ym else {
            continue;
        };
        // Points up to `lo` stay with the first part, from `hi` on go to the second.
        let p = parts[x].clone();
        let first: Vec<(i32, bool)> = p.pts.iter().copied().filter(|q| q.0 < hi).collect();
        let second: Vec<(i32, bool)> = p.pts.iter().copied().filter(|q| q.0 >= hi).collect();
        parts[x] = Part::new(p.net, first, &[ym]);
        parts.push(Part::new(p.net, second, &[ym]));
        let new = parts.len() - 1;
        // Earlier jogs of x whose row now belongs to the second part follow it.
        for j in jogs.iter_mut() {
            for e in [&mut j.0, &mut j.1] {
                if *e == x && j.2 >= hi {
                    *e = new;
                }
            }
        }
        jogs.push((x, new, ym));
        let at = busy.partition_point(|&y| y < ym);
        busy.insert(at, ym);
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(net: usize, y: i32, left: bool) -> ChanPt {
        ChanPt { net, y, left }
    }

    fn track_of(r: &ChannelRoute, net: usize) -> Vec<i32> {
        r.pieces
            .iter()
            .filter_map(|p| match *p {
                Piece::Vertical { net: n, t, .. } if n == net => Some(t),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn stub_constraints_hold() {
        // Net 0: left at y=0, right at y=4. Net 1: left at y=4, right at y=8.
        // Net 1's left stub at y=4 must end left of net 0's track (net 0's right stub there).
        let r = route_channel(&[pt(0, 0, true), pt(0, 4, false), pt(1, 4, true), pt(1, 8, false)]);
        assert!(track_of(&r, 1)[0] < track_of(&r, 0)[0], "{r:?}");
    }

    #[test]
    fn bundles_route_as_staircases() {
        // Four nets going down by 10 and four going up by 10: no stub crosses another net's
        // vertical (upper net right when going down, left when going up).
        let mut pts = Vec::new();
        for i in 0..4 {
            pts.push(pt(i, 2 * i as i32, true));
            pts.push(pt(i, 10 + 2 * i as i32, false));
            pts.push(pt(10 + i, 40 + 2 * i as i32, true));
            pts.push(pt(10 + i, 30 + 2 * i as i32, false));
        }
        let r = route_channel(&pts);
        for i in 0..3 {
            assert!(track_of(&r, i)[0] > track_of(&r, i + 1)[0], "{r:?}");
            assert!(track_of(&r, 10 + i)[0] < track_of(&r, 11 + i)[0], "{r:?}");
        }
        // The two bundles do not overlap vertically: they share tracks.
        assert_eq!(r.tracks, 4);
    }

    #[test]
    fn constraint_cycles_get_a_dogleg() {
        // Net 0: left y=0, right y=6. Net 1: right y=0, left y=6: 0 → 1 (y=0) and 1 → 0 (y=6).
        let r = route_channel(&[pt(0, 0, true), pt(0, 6, false), pt(1, 0, false), pt(1, 6, true)]);
        let jogs = r
            .pieces
            .iter()
            .filter(|p| matches!(p, Piece::Jog { .. }))
            .count();
        assert_eq!(jogs, 1, "{r:?}");
        // Stubs at the same height never overlap: a left stub's track is left of the right
        // stub's track.
        for y in [0, 6] {
            let l = r.pieces.iter().find_map(|p| match *p {
                Piece::LeftStub { t, y: py, .. } if py == y => Some(t),
                _ => None,
            });
            let rt = r.pieces.iter().find_map(|p| match *p {
                Piece::RightStub { t, y: py, .. } if py == y => Some(t),
                _ => None,
            });
            assert!(l.unwrap() < rt.unwrap(), "{r:?}");
        }
    }
}
