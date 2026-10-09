//! Crossing reduction and vertical placement of the column items (nodes and lanes).
//!
//! * **Order**: alternating barycenter sweeps (an item's key is the mean height of what it
//!   connects to in the neighbouring column, port offsets included), each followed by a
//!   *transpose* pass that swaps adjacent items when that removes crossings. The best order seen
//!   (by the bilayer crossing count of all channels) is kept.
//! * **Y**: order-preserving least-squares placement toward the connected items' heights
//!   ([`super::pav_place`]), in alternating passes.

use std::collections::HashMap;

use super::{LANE_GAP, NODE_GAP, Node, pav_place};
use crate::geom::Side;

/// An item of a column: a node or a lane of a net.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Item {
    /// Node index.
    Node(usize),
    /// Lane index.
    Lane(usize),
}

/// Most barycenter sweeps.
const MAX_SWEEPS: usize = 16;
/// Stop after this many sweeps without improvement.
const PATIENCE: usize = 4;

/// Layout state for ordering and y placement.
pub(super) struct Ctx {
    pub nodes: Vec<Node>,
    /// `(net, column)` per lane.
    pub lanes: Vec<(usize, usize)>,
    /// `(net, port)` per node.
    pub node_nets: Vec<Vec<(usize, usize)>>,
    pub cols: Vec<Vec<Item>>,
    /// Lane y by lane index.
    pub ys: Vec<i32>,
}

/// Mean height and count of a net's points facing one way in a column.
type Anchors = HashMap<usize, (f64, u32)>;

impl Ctx {
    fn height(&self, it: Item) -> i32 {
        match it {
            Item::Node(n) => self.nodes[n].height(),
            Item::Lane(_) => 0,
        }
    }

    fn gap(&self, a: Item, b: Item) -> i32 {
        match (a, b) {
            // Pin markers stack at the pin pitch, so pins line up with package ports.
            (Item::Node(x), Item::Node(y))
                if self.nodes[x].is_pin_marker() && self.nodes[y].is_pin_marker() =>
            {
                0
            }
            (Item::Node(_), Item::Node(_)) => NODE_GAP,
            _ => LANE_GAP,
        }
    }

    fn top(&self, it: Item) -> i32 {
        match it {
            Item::Node(n) => self.nodes[n].y,
            Item::Lane(l) => self.ys[l],
        }
    }

    fn set_top(&mut self, it: Item, y: i32) {
        match it {
            Item::Node(n) => self.nodes[n].y = y,
            Item::Lane(l) => self.ys[l] = y,
        }
    }

    /// Stack column `c` top-down in its current order.
    fn pack(&mut self, c: usize) {
        let mut y = 0;
        let mut prev: Option<Item> = None;
        for i in 0..self.cols[c].len() {
            let it = self.cols[c][i];
            if let Some(p) = prev {
                y += self.height(p) + self.gap(p, it);
            }
            self.set_top(it, y);
            prev = Some(it);
        }
    }

    /// Mean height of each net's points in column `col` facing right (`facing_right`) or left.
    /// Lanes face both ways.
    fn net_anchors(&self, col: usize, facing_right: bool) -> Anchors {
        let mut acc: Anchors = HashMap::new();
        for &it in &self.cols[col] {
            match it {
                Item::Node(n) => {
                    let node = &self.nodes[n];
                    for &(ni, p) in &self.node_nets[n] {
                        let (pos, side) = node.port(p);
                        if (side == Side::Right) == facing_right {
                            let e = acc.entry(ni).or_default();
                            e.0 += f64::from(node.y + pos.y);
                            e.1 += 1;
                        }
                    }
                }
                Item::Lane(l) => {
                    let e = acc.entry(self.lanes[l].0).or_default();
                    e.0 += f64::from(self.ys[l]);
                    e.1 += 1;
                }
            }
        }
        acc
    }

    /// Anchors of the columns left and right of `c` (facing `c`).
    fn side_anchors(&self, c: usize) -> (Anchors, Anchors) {
        let left = if c > 0 {
            self.net_anchors(c - 1, true)
        } else {
            HashMap::new()
        };
        let right = if c + 1 < self.cols.len() {
            self.net_anchors(c + 1, false)
        } else {
            HashMap::new()
        };
        (left, right)
    }

    /// Desired top of item `it` from the anchors of a neighbouring column (`from_left`: the
    /// anchors are on the item's left).
    fn desired(&self, it: Item, anchors: &Anchors, from_left: bool) -> Option<f64> {
        match it {
            Item::Lane(l) => anchors
                .get(&self.lanes[l].0)
                .map(|&(s, n)| s / f64::from(n)),
            Item::Node(n) => {
                let node = &self.nodes[n];
                let (mut s, mut k) = (0.0, 0u32);
                for &(ni, p) in &self.node_nets[n] {
                    let (pos, side) = node.port(p);
                    if (side != Side::Right) == from_left
                        && let Some(&(sum, cnt)) = anchors.get(&ni)
                    {
                        s += sum / f64::from(cnt) - f64::from(pos.y);
                        k += 1;
                    }
                }
                (k > 0).then(|| s / f64::from(k))
            }
        }
    }

    /// Heights (absolute, from the anchors) an item connects to on its left and right.
    fn item_links(&self, it: Item, left: &Anchors, right: &Anchors) -> (Vec<f64>, Vec<f64>) {
        let mean = |a: &Anchors, ni: usize| a.get(&ni).map(|&(s, n)| s / f64::from(n));
        match it {
            Item::Lane(l) => {
                let ni = self.lanes[l].0;
                (
                    mean(left, ni).into_iter().collect(),
                    mean(right, ni).into_iter().collect(),
                )
            }
            Item::Node(n) => {
                let node = &self.nodes[n];
                let (mut ls, mut rs) = (Vec::new(), Vec::new());
                for &(ni, p) in &self.node_nets[n] {
                    if node.port(p).1 == Side::Right {
                        rs.extend(mean(right, ni));
                    } else {
                        ls.extend(mean(left, ni));
                    }
                }
                (ls, rs)
            }
        }
    }

    /// Barycenter reorder of column `c` against column `other`.
    fn reorder(&mut self, c: usize, other: usize) {
        let from_left = other < c;
        let anchors = self.net_anchors(other, from_left);
        let mut keyed: Vec<(f64, usize, Item)> = self.cols[c]
            .iter()
            .enumerate()
            .map(|(i, &it)| {
                let half = f64::from(self.height(it)) / 2.0;
                let k = self
                    .desired(it, &anchors, from_left)
                    .map_or(f64::from(self.top(it)) + half, |d| d + half);
                (k, i, it)
            })
            .collect();
        keyed.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        self.cols[c] = keyed.into_iter().map(|k| k.2).collect();
        self.pack(c);
    }

    /// Swap adjacent items of column `c` where that removes crossings with both neighbours.
    /// Returns true if anything moved.
    fn transpose(&mut self, c: usize) -> bool {
        let n = self.cols[c].len();
        if n < 2 {
            return false;
        }
        let (left, right) = self.side_anchors(c);
        let links: Vec<(Vec<f64>, Vec<f64>)> = self.cols[c]
            .iter()
            .map(|&it| self.item_links(it, &left, &right))
            .collect();
        // Crossings between u's and v's links when u is above v.
        let cross = |u: &(Vec<f64>, Vec<f64>), v: &(Vec<f64>, Vec<f64>)| -> usize {
            let side = |a: &[f64], b: &[f64]| {
                a.iter()
                    .map(|&x| b.iter().filter(|&&y| x > y).count())
                    .sum::<usize>()
            };
            side(&u.0, &v.0) + side(&u.1, &v.1)
        };
        let mut order: Vec<usize> = (0..n).collect();
        let mut moved = false;
        for i in 0..n - 1 {
            let (a, b) = (order[i], order[i + 1]);
            if links[a].0.len() + links[a].1.len() > 64 || links[b].0.len() + links[b].1.len() > 64
            {
                continue;
            }
            if cross(&links[b], &links[a]) < cross(&links[a], &links[b]) {
                order.swap(i, i + 1);
                moved = true;
            }
        }
        if moved {
            let items = std::mem::take(&mut self.cols[c]);
            self.cols[c] = order.into_iter().map(|i| items[i]).collect();
            self.pack(c);
        }
        moved
    }

    /// Bilayer crossing count summed over all channels (each net drawn as a star between its
    /// points on the two sides).
    pub fn crossings(&self) -> u64 {
        let mut total = 0;
        for c in 1..self.cols.len() {
            let l = self.net_anchors_points(c - 1, true);
            let r = self.net_anchors_points(c, false);
            let mut edges: Vec<(f64, f64)> = Vec::new();
            for (ni, ls) in &l {
                let Some(rs) = r.get(ni) else { continue };
                let ml = ls.iter().sum::<f64>() / ls.len() as f64;
                let mr = rs.iter().sum::<f64>() / rs.len() as f64;
                if ls.len() == 1 {
                    edges.extend(rs.iter().map(|&y| (ls[0], y)));
                } else if rs.len() == 1 {
                    edges.extend(ls.iter().map(|&y| (y, rs[0])));
                } else {
                    edges.extend(rs.iter().map(|&y| (ml, y)));
                    edges.extend(ls.iter().map(|&y| (y, mr)));
                }
            }
            edges.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
            let mut ys: Vec<f64> = edges.into_iter().map(|e| e.1).collect();
            total += inversions(&mut ys);
        }
        total
    }

    /// Point heights per net in column `col` facing right (`facing_right`) or left.
    fn net_anchors_points(&self, col: usize, facing_right: bool) -> HashMap<usize, Vec<f64>> {
        let mut acc: HashMap<usize, Vec<f64>> = HashMap::new();
        for &it in &self.cols[col] {
            match it {
                Item::Node(n) => {
                    let node = &self.nodes[n];
                    for &(ni, p) in &self.node_nets[n] {
                        let (pos, side) = node.port(p);
                        if (side == Side::Right) == facing_right {
                            acc.entry(ni).or_default().push(f64::from(node.y + pos.y));
                        }
                    }
                }
                Item::Lane(l) => acc
                    .entry(self.lanes[l].0)
                    .or_default()
                    .push(f64::from(self.ys[l])),
            }
        }
        acc
    }

    /// Crossing reduction: barycenter sweeps + transposes, keeping the best order.
    pub fn order(&mut self) {
        self.ys = vec![0; self.lanes.len()];
        let n_cols = self.cols.len();
        for c in 0..n_cols {
            self.pack(c);
        }
        let mut best = self.crossings();
        let mut best_cols = self.cols.clone();
        let mut stale = 0;
        for it in 0..MAX_SWEEPS {
            if best == 0 {
                break;
            }
            if it % 2 == 0 {
                for c in 1..n_cols {
                    self.reorder(c, c - 1);
                }
            } else {
                for c in (0..n_cols.saturating_sub(1)).rev() {
                    self.reorder(c, c + 1);
                }
            }
            for _ in 0..2 {
                let mut moved = false;
                for c in 0..n_cols {
                    moved |= self.transpose(c);
                }
                if !moved {
                    break;
                }
            }
            let x = self.crossings();
            if x < best {
                best = x;
                best_cols = self.cols.clone();
                stale = 0;
            } else {
                stale += 1;
                if stale >= PATIENCE {
                    break;
                }
            }
        }
        self.cols = best_cols;
        for c in 0..n_cols {
            self.pack(c);
        }
    }

    /// Place column `c` by order-preserving least squares toward the desired positions
    /// (anchors on the left if `left_pass`, falling back to the other side).
    pub fn place_column(&mut self, c: usize, left_pass: bool) {
        let (left, right) = self.side_anchors(c);
        let items = self.cols[c].clone();
        let desired: Vec<Option<f64>> = items
            .iter()
            .map(|&it| {
                let (a, b) = if left_pass {
                    (&left, &right)
                } else {
                    (&right, &left)
                };
                self.desired(it, a, left_pass)
                    .or_else(|| self.desired(it, b, !left_pass))
            })
            .collect();
        let mut gaps = Vec::with_capacity(items.len());
        for i in 0..items.len() {
            gaps.push(if i == 0 {
                0
            } else {
                self.height(items[i - 1]) + self.gap(items[i - 1], items[i])
            });
        }
        let current: Vec<f64> = items.iter().map(|&it| f64::from(self.top(it))).collect();
        let tops = pav_place(&desired, &gaps, &current);
        for (it, y) in items.into_iter().zip(tops) {
            self.set_top(it, y);
        }
    }

    /// Y placement: alternating least-squares passes over the columns.
    pub fn place(&mut self, passes: usize) {
        let n_cols = self.cols.len();
        for pass in 0..passes {
            if pass % 2 == 0 {
                for c in 0..n_cols {
                    self.place_column(c, true);
                }
            } else {
                for c in (0..n_cols).rev() {
                    self.place_column(c, false);
                }
            }
        }
    }
}

/// Number of pairs `i < j` with `v[i] > v[j]` (merge sort; `v` ends up sorted).
fn inversions(v: &mut [f64]) -> u64 {
    let n = v.len();
    if n < 2 {
        return 0;
    }
    let mut buf = v.to_vec();
    let mut count = 0u64;
    let mut width = 1;
    while width < n {
        let mut i = 0;
        while i < n {
            let mid = (i + width).min(n);
            let hi = (i + 2 * width).min(n);
            let (mut a, mut b, mut k) = (i, mid, i);
            while a < mid && b < hi {
                if v[b] < v[a] {
                    count += (mid - a) as u64;
                    buf[k] = v[b];
                    b += 1;
                } else {
                    buf[k] = v[a];
                    a += 1;
                }
                k += 1;
            }
            buf[k..k + mid - a].copy_from_slice(&v[a..mid]);
            k += mid - a;
            buf[k..k + hi - b].copy_from_slice(&v[b..hi]);
            i = hi;
        }
        v.copy_from_slice(&buf);
        width *= 2;
    }
    count
}

#[cfg(test)]
mod tests {
    use super::inversions;

    #[test]
    fn inversion_count() {
        let mut v = vec![3.0, 1.0, 2.0, 2.0, 0.0];
        // (3,1) (3,2) (3,2) (3,0) (1,0) (2,0) (2,0)
        assert_eq!(inversions(&mut v), 7);
        assert_eq!(v, vec![0.0, 1.0, 2.0, 2.0, 3.0]);
    }
}
