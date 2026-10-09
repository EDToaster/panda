//! Wire geometry for editing: Manhattan routing of new wires, rubber-banding of wires attached
//! to moved ports, connected components (net splitting) and junction dots.
//!
//! Everything here is pure grid geometry on segment lists; [`rebuild`] normalizes a segment list
//! into a [`NetGeom`] (collinear segments merged, a vertex at every connection point, junction
//! dots where three or more wire directions or a port and two directions meet).

use std::collections::{HashMap, HashSet};

use crate::doc::NetGeom;
use crate::geom::{GridPt, Side, gp};
use crate::layout::net_geom;

/// A wire segment (axis-aligned; the two points differ in at most one coordinate).
pub type Seg = (GridPt, GridPt);

/// Orientation of the wire leg at a route end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// Leaves / enters horizontally.
    Horizontal,
    /// Leaves / enters vertically.
    Vertical,
}

impl Axis {
    /// Axis of a port on side `s` (wires leave a port perpendicular to its side).
    pub fn of_side(s: Side) -> Axis {
        match s {
            Side::Left | Side::Right => Axis::Horizontal,
            Side::Top | Side::Bottom => Axis::Vertical,
        }
    }
}

/// Segment with its points ordered (`a <= b`), so equal segments compare equal.
pub fn norm(a: GridPt, b: GridPt) -> Seg {
    if a <= b { (a, b) } else { (b, a) }
}

/// True if `p` lies on segment `(a, b)` (end points included).
pub fn on_segment(p: GridPt, (a, b): Seg) -> bool {
    if a.y == b.y && p.y == a.y {
        p.x >= a.x.min(b.x) && p.x <= a.x.max(b.x)
    } else if a.x == b.x && p.x == a.x {
        p.y >= a.y.min(b.y) && p.y <= a.y.max(b.y)
    } else {
        false
    }
}

/// True if `p` lies strictly inside segment `(a, b)`.
pub fn on_interior(p: GridPt, s: Seg) -> bool {
    on_segment(p, s) && p != s.0 && p != s.1
}

/// Squared-free distance of world point `(x, y)` to a segment (grid units).
pub fn distance(x: f32, y: f32, (a, b): Seg) -> f32 {
    let (x0, x1) = (a.x.min(b.x) as f32, a.x.max(b.x) as f32);
    let (y0, y1) = (a.y.min(b.y) as f32, a.y.max(b.y) as f32);
    let dx = (x0 - x).max(0.0).max(x - x1);
    let dy = (y0 - y).max(0.0).max(y - y1);
    (dx * dx + dy * dy).sqrt()
}

/// The grid point of segment `s` nearest to world point `(x, y)`.
pub fn nearest_on(x: f32, y: f32, (a, b): Seg) -> GridPt {
    let cx = (x.round() as i32).clamp(a.x.min(b.x), a.x.max(b.x));
    let cy = (y.round() as i32).clamp(a.y.min(b.y), a.y.max(b.y));
    gp(cx, cy)
}

/// Remove repeated points and merge collinear runs of a polyline.
pub fn simplify(pts: &[GridPt]) -> Vec<GridPt> {
    let mut out: Vec<GridPt> = Vec::with_capacity(pts.len());
    for &p in pts {
        if out.last() == Some(&p) {
            continue;
        }
        if out.len() >= 2 {
            let (a, b) = (out[out.len() - 2], out[out.len() - 1]);
            if (a.x == b.x && b.x == p.x) || (a.y == b.y && b.y == p.y) {
                out.pop();
            }
        }
        out.push(p);
    }
    out
}

/// Orthogonal route from `a` to `b` as a polyline. `a_axis` / `b_axis` give the direction the
/// wire should leave `a` / enter `b` (ports: perpendicular to their side); `None` lets the
/// router choose (dominant direction of the displacement).
///
/// * both horizontal: Z route with the vertical jog halfway (straight if aligned);
/// * both vertical: Z route with the horizontal jog halfway;
/// * mixed: L route with the corner where both legs keep their axis.
pub fn route(a: GridPt, a_axis: Option<Axis>, b: GridPt, b_axis: Option<Axis>) -> Vec<GridPt> {
    if a == b {
        return vec![a];
    }
    let d = b - a;
    let dominant = if d.x.abs() >= d.y.abs() {
        Axis::Horizontal
    } else {
        Axis::Vertical
    };
    let (sa, sb) = match (a_axis, b_axis) {
        (Some(x), Some(y)) => (x, y),
        (Some(x), None) => (
            x,
            if x == Axis::Horizontal {
                Axis::Vertical
            } else {
                Axis::Horizontal
            },
        ),
        (None, Some(y)) => (
            if y == Axis::Horizontal {
                Axis::Vertical
            } else {
                Axis::Horizontal
            },
            y,
        ),
        (None, None) => (
            dominant,
            if dominant == Axis::Horizontal {
                Axis::Vertical
            } else {
                Axis::Horizontal
            },
        ),
    };
    let pts = match (sa, sb) {
        (Axis::Horizontal, Axis::Horizontal) => {
            let mx = a.x + d.x / 2;
            vec![a, gp(mx, a.y), gp(mx, b.y), b]
        }
        (Axis::Vertical, Axis::Vertical) => {
            let my = a.y + d.y / 2;
            vec![a, gp(a.x, my), gp(b.x, my), b]
        }
        (Axis::Horizontal, Axis::Vertical) => vec![a, gp(b.x, a.y), b],
        (Axis::Vertical, Axis::Horizontal) => vec![a, gp(a.x, b.y), b],
    };
    simplify(&pts)
}

/// Segments of a polyline (zero-length ones dropped).
pub fn segments_of(pts: &[GridPt]) -> Vec<Seg> {
    pts.windows(2)
        .filter(|w| w[0] != w[1])
        .map(|w| (w[0], w[1]))
        .collect()
}

/// Normalize segments into a [`NetGeom`]: merge collinear overlaps, keep a vertex at every
/// connection point, junction dots where `≥ 3` directions meet (a connected port at a point
/// counts as one direction).
pub fn rebuild(segs: Vec<Seg>, ports: &[GridPt]) -> NetGeom {
    let mut g = net_geom(segs);
    g.junctions = junctions_with_ports(&g, ports);
    g
}

/// Junction points of `g`, counting a port touching the wire as one more direction.
pub fn junctions_with_ports(g: &NetGeom, ports: &[GridPt]) -> Vec<GridPt> {
    let segs: Vec<Seg> = g.segments().collect();
    let mut deg: HashMap<GridPt, u32> = HashMap::new();
    for &(a, b) in &segs {
        *deg.entry(a).or_default() += 1;
        *deg.entry(b).or_default() += 1;
    }
    let mut candidates: HashSet<GridPt> = deg.keys().copied().collect();
    candidates.extend(ports.iter().copied());
    let mut out: Vec<GridPt> = candidates
        .into_iter()
        .filter(|&p| {
            let d = deg.get(&p).copied().unwrap_or(0);
            let interior = segs.iter().filter(|&&s| on_interior(p, s)).count() as u32;
            let port = u32::from(ports.contains(&p));
            d + 2 * interior >= 1 && d + 2 * interior + port >= 3
        })
        .collect();
    out.sort_unstable();
    out
}

/// Connected components of a segment set (segments touch when they share a point or an end of
/// one lies on the other). Returns segment indices per component, in first-index order.
pub fn components(segs: &[Seg]) -> Vec<Vec<usize>> {
    let n = segs.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    for i in 0..n {
        for j in i + 1..n {
            let (a, b) = (segs[i], segs[j]);
            let touch = on_segment(a.0, b)
                || on_segment(a.1, b)
                || on_segment(b.0, a)
                || on_segment(b.1, a);
            if touch {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                if ri != rj {
                    parent[rj] = ri;
                }
            }
        }
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut slot: HashMap<usize, usize> = HashMap::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        let k = *slot.entry(r).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[k].push(i);
    }
    groups
}

/// Rubber-band a net's segments after some of its points moved (`moved`: old → new position).
///
/// A segment with both ends moved by the same offset is translated. A segment with one end
/// moved is re-routed orthogonally: with an L whose corner sits at the fixed end when that end
/// is a joint of the wire (`joints`: points where the wire continues), so the corner just slides
/// along the neighbouring wire, else with a Z that keeps the original direction at both ends
/// (port-to-port wires). Ends moved by different offsets get a Z route as well.
pub fn stretch(segs: &[Seg], moved: &HashMap<GridPt, GridPt>) -> Vec<Seg> {
    let mut deg: HashMap<GridPt, u32> = HashMap::new();
    for &(a, b) in segs {
        *deg.entry(a).or_default() += 1;
        *deg.entry(b).or_default() += 1;
    }
    let joint = |p: GridPt| deg.get(&p).copied().unwrap_or(0) >= 2;
    let mut out = Vec::with_capacity(segs.len() + 4);
    for &(a, b) in segs {
        let (na, nb) = (
            moved.get(&a).copied().unwrap_or(a),
            moved.get(&b).copied().unwrap_or(b),
        );
        if na == a && nb == b {
            out.push((a, b));
            continue;
        }
        if na - a == nb - b {
            out.push((na, nb));
            continue;
        }
        let axis = if a.y == b.y {
            Axis::Horizontal
        } else {
            Axis::Vertical
        };
        let path = if na == a && joint(a) {
            // L with the corner at the fixed joint `a`: perpendicular leg first.
            match axis {
                Axis::Horizontal => vec![a, gp(a.x, nb.y), nb],
                Axis::Vertical => vec![a, gp(nb.x, a.y), nb],
            }
        } else if nb == b && joint(b) {
            match axis {
                Axis::Horizontal => vec![na, gp(b.x, na.y), b],
                Axis::Vertical => vec![na, gp(na.x, b.y), b],
            }
        } else {
            route(na, Some(axis), nb, Some(axis))
        };
        out.extend(segments_of(&simplify(&path)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn orthogonal(pts: &[GridPt]) -> bool {
        pts.windows(2).all(|w| w[0].x == w[1].x || w[0].y == w[1].y)
    }

    #[test]
    fn routes_are_orthogonal_and_end_correctly() {
        let axes = [None, Some(Axis::Horizontal), Some(Axis::Vertical)];
        for a_axis in axes {
            for b_axis in axes {
                for (dx, dy) in [(5, 0), (0, 4), (6, 3), (-3, 7), (-8, -2), (1, 1)] {
                    let a = gp(2, 3);
                    let b = a + gp(dx, dy);
                    let r = route(a, a_axis, b, b_axis);
                    assert_eq!(r.first(), Some(&a));
                    assert_eq!(r.last(), Some(&b));
                    assert!(orthogonal(&r), "{r:?}");
                    assert!(r.len() <= 4);
                    // (With |dx| < 2 a Z route has no room for its jog.)
                    if let Some(Axis::Horizontal) = a_axis
                        && dy != 0
                        && dx.abs() >= 2
                    {
                        assert_eq!(r[1].y, a.y, "leaves horizontally: {r:?}");
                    }
                }
            }
        }
        // Port-to-port, both horizontal: Z with the jog halfway.
        assert_eq!(
            route(
                gp(0, 0),
                Some(Axis::Horizontal),
                gp(10, 4),
                Some(Axis::Horizontal)
            ),
            vec![gp(0, 0), gp(5, 0), gp(5, 4), gp(10, 4)]
        );
    }

    #[test]
    fn components_split_and_join() {
        let segs = vec![
            (gp(0, 0), gp(4, 0)),
            (gp(4, 0), gp(4, 4)),
            (gp(10, 0), gp(14, 0)),
            // T onto the first segment's interior.
            (gp(2, 0), gp(2, 6)),
        ];
        let c = components(&segs);
        assert_eq!(c, vec![vec![0, 1, 3], vec![2]]);
    }

    #[test]
    fn rebuild_puts_junction_at_t_and_port_branch() {
        let g = rebuild(
            vec![(gp(0, 0), gp(8, 0)), (gp(4, 0), gp(4, 5))],
            &[gp(0, 0), gp(8, 0), gp(4, 5)],
        );
        assert_eq!(g.junctions, vec![gp(4, 0)]);
        // The T point became a vertex of the horizontal line.
        assert!(
            g.polylines
                .iter()
                .any(|l| l.contains(&gp(4, 0)) && l.len() == 3)
        );
        // Two wires ending at one port: a dot there.
        let g = rebuild(
            vec![(gp(0, 0), gp(0, 4)), (gp(0, 0), gp(5, 0))],
            &[gp(0, 0)],
        );
        assert_eq!(g.junctions, vec![gp(0, 0)]);
    }

    #[test]
    fn stretch_keeps_wires_orthogonal() {
        // Port-to-port straight wire, right end moves down by 3: Z route.
        let segs = vec![(gp(0, 0), gp(10, 0))];
        let moved = HashMap::from([(gp(10, 0), gp(10, 3))]);
        let s = stretch(&segs, &moved);
        assert_eq!(
            s,
            vec![
                (gp(0, 0), gp(5, 0)),
                (gp(5, 0), gp(5, 3)),
                (gp(5, 3), gp(10, 3))
            ]
        );
        // Corner wire: moving the end slides the corner's vertical leg.
        let segs = vec![
            (gp(0, 0), gp(4, 0)),
            (gp(4, 0), gp(4, 6)),
            (gp(4, 6), gp(9, 6)),
        ];
        let moved = HashMap::from([(gp(9, 6), gp(9, 8))]);
        let g = rebuild(stretch(&segs, &moved), &[]);
        let all: Vec<Seg> = g.segments().collect();
        assert!(all.iter().all(|&(a, b)| a.x == b.x || a.y == b.y));
        assert!(all.iter().any(|&s| on_segment(gp(9, 8), s)));
        assert!(all.iter().any(|&s| on_segment(gp(0, 0), s)));
        // Everything moved by the same offset: translated.
        let moved = HashMap::from([(gp(0, 0), gp(1, 1)), (gp(4, 0), gp(5, 1))]);
        assert_eq!(
            stretch(&[(gp(0, 0), gp(4, 0))], &moved),
            vec![(gp(1, 1), gp(5, 1))]
        );
    }
}
