//! Pure waveform-viewer model: the time axis (zoom / scroll / follow / cursor), value
//! formatting per radix, and the per-pixel decimation of a signal's change history into drawable
//! runs. No egui painting here; [`super::waveform`] draws what these functions compute.
//!
//! Decimation keeps a frame's cost proportional to the plot width, not to the number of
//! recorded changes: a signal that changes a million times inside the window costs
//! `O(width · log n)` (binary searches into the history), never a walk over every change.

use std::collections::VecDeque;

use panda_netlist::Logic;

use crate::simhost::{ClockSpec, Packed, WaveChunk};

/// Smallest ticks-per-pixel (most zoomed in: 64 px per tick).
pub const MIN_TPP: f64 = 1.0 / 64.0;
/// Largest ticks-per-pixel.
pub const MAX_TPP: f64 = 1e10;

/// The horizontal axis of the waveform plot, in ticks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaveView {
    /// Tick at the left edge of the plot.
    pub start: f64,
    /// Ticks per pixel.
    pub tpp: f64,
    /// Keep the latest tick at the right edge.
    pub follow: bool,
    /// Cursor tick.
    pub cursor: Option<u64>,
}

impl Default for WaveView {
    fn default() -> Self {
        WaveView {
            start: 0.0,
            tpp: 1.0,
            follow: true,
            cursor: None,
        }
    }
}

impl WaveView {
    /// Pixel offset (from the plot's left edge) of `tick`.
    pub fn x_of(&self, tick: f64) -> f64 {
        (tick - self.start) / self.tpp
    }

    /// Tick at pixel offset `x`.
    pub fn tick_at(&self, x: f64) -> f64 {
        self.start + x * self.tpp
    }

    /// Visible tick range `[start, end]` for a plot `width` pixels wide.
    pub fn range(&self, width: f64) -> (f64, f64) {
        (self.start, self.tick_at(width))
    }

    /// Zoom by `factor` (> 1 zooms in) keeping the tick under pixel `x` in place. While
    /// following, the right edge stays anchored instead.
    pub fn zoom_at(&mut self, x: f64, width: f64, factor: f64) {
        if !(factor.is_finite() && factor > 0.0) {
            return;
        }
        let x = if self.follow { width } else { x };
        let t = self.tick_at(x);
        self.tpp = (self.tpp / factor).clamp(MIN_TPP, MAX_TPP);
        self.start = t - x * self.tpp;
    }

    /// Scroll by `dx` pixels (content moves right for positive `dx`). Stops following.
    pub fn scroll(&mut self, dx: f64) {
        if dx != 0.0 {
            self.start -= dx * self.tpp;
            self.follow = false;
        }
    }

    /// Place `latest` at the right edge (when following).
    pub fn follow_to(&mut self, latest: u64, width: f64) {
        if self.follow {
            self.start = latest as f64 - width * self.tpp;
        }
    }

    /// Show `[t0, t1]` across `width` pixels.
    pub fn fit(&mut self, t0: u64, t1: u64, width: f64) {
        let span = (t1.saturating_sub(t0)).max(1) as f64;
        self.tpp = (span / width.max(1.0)).clamp(MIN_TPP, MAX_TPP);
        self.start = t0 as f64;
    }

    /// Place the cursor at pixel `x` (snapped to the nearest tick, never negative).
    pub fn set_cursor_at(&mut self, x: f64) {
        self.cursor = Some(self.tick_at(x).round().max(0.0) as u64);
    }
}

/// A "nice" grid step (1, 2 or 5 × 10ⁿ ticks) at least `min_px` pixels wide.
pub fn nice_step(tpp: f64, min_px: f64) -> u64 {
    let raw = (tpp * min_px).max(1.0);
    let mut p = 1u64;
    loop {
        for m in [1, 2, 5] {
            let s = p.saturating_mul(m);
            if s as f64 >= raw || s == u64::MAX {
                return s;
            }
        }
        p = p.saturating_mul(10);
    }
}

/// Unit of the time axis and of time readouts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TimeUnit {
    /// Simulation ticks (natural in Timing mode).
    #[default]
    Ticks,
    /// Clock cycles of a reference clock (natural in Fast mode, which advances event by event
    /// at clock-phase granularity). Falls back to ticks when no clock is known.
    Cycles,
}

impl TimeUnit {
    /// Menu label.
    pub fn label(self) -> &'static str {
        match self {
            TimeUnit::Ticks => "Ticks",
            TimeUnit::Cycles => "Cycles",
        }
    }
}

/// Tick of rising edge number `k` of `clk` (edge 0 is the first rising edge at or after tick 0).
pub fn rising_edge(clk: &ClockSpec, k: u64) -> u64 {
    let p = u64::from(clk.period.max(1));
    (u64::from(clk.phase) % p).saturating_add(k.saturating_mul(p))
}

/// Position of `tick` in cycles of `clk` (0.0 at the first rising edge; negative before it).
pub fn cycle_of(clk: &ClockSpec, tick: f64) -> f64 {
    let p = f64::from(clk.period.max(1));
    (tick - (f64::from(clk.phase) % p)) / p
}

/// Human-readable time of `tick`: `"1234"` in ticks, `"c12+3"` (cycle 12, 3 ticks after its
/// rising edge) in cycles.
pub fn format_time(tick: u64, unit: TimeUnit, clk: Option<&ClockSpec>) -> String {
    match (unit, clk) {
        (TimeUnit::Cycles, Some(c)) => {
            let p = u64::from(c.period.max(1));
            let first = u64::from(c.phase) % p;
            if tick < first {
                return format!("c-1+{}", tick + p - first);
            }
            let k = (tick - first) / p;
            let off = (tick - first) % p;
            if off == 0 {
                format!("c{k}")
            } else {
                format!("c{k}+{off}")
            }
        }
        _ => tick.to_string(),
    }
}

/// One labelled grid line of the time axis.
#[derive(Clone, Debug, PartialEq)]
pub struct AxisMark {
    /// Tick of the line.
    pub tick: u64,
    /// Label.
    pub label: String,
}

/// Grid lines of the time axis for the visible range of `view` over `width` pixels, spaced at
/// least `min_px` apart: "nice" tick multiples, or "nice" multiples of whole cycles of `clk`
/// (placed on its rising edges) in [`TimeUnit::Cycles`]. At most `width / min_px + 2` marks.
pub fn axis_marks(
    view: &WaveView,
    width: f64,
    min_px: f64,
    unit: TimeUnit,
    clk: Option<&ClockSpec>,
) -> Vec<AxisMark> {
    let (t0, t1) = view.range(width);
    let t0 = t0.max(0.0);
    let mut out = Vec::new();
    if t1 < t0 {
        return out;
    }
    let max_marks = (width / min_px.max(1.0)) as usize + 2;
    match (unit, clk) {
        (TimeUnit::Cycles, Some(c)) => {
            let p = f64::from(c.period.max(1));
            let step = nice_step(view.tpp / p, min_px);
            let k0 = cycle_of(c, t0).max(0.0);
            let mut k = (k0 / step as f64).ceil() as u64 * step;
            while out.len() < max_marks {
                let t = rising_edge(c, k);
                if t as f64 > t1 {
                    break;
                }
                out.push(AxisMark {
                    tick: t,
                    label: format!("c{k}"),
                });
                k = match k.checked_add(step) {
                    Some(n) => n,
                    None => break,
                };
            }
        }
        _ => {
            let step = nice_step(view.tpp, min_px);
            let mut t = (t0 / step as f64).ceil() as u64 * step;
            while (t as f64) <= t1 && out.len() < max_marks {
                out.push(AxisMark {
                    tick: t,
                    label: t.to_string(),
                });
                t = match t.checked_add(step) {
                    Some(n) => n,
                    None => break,
                };
            }
        }
    }
    out
}

/// Where capture lost data, derived from the [`WaveChunk`] stream of one waveform config.
///
/// The shadow capture reports a lost interval by setting [`WaveChunk::gap`] on the next chunk it
/// delivers (whose first change is where recording resumed). The tracker turns that into tick
/// ranges `(last tick recorded before, first tick recorded after)` for drawing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GapTracker {
    /// Config the gaps belong to.
    pub config: u64,
    /// Newest tick recorded so far.
    pub last: Option<u64>,
    /// Lost intervals `(from, to)` in tick order (`from < to`).
    pub gaps: Vec<(u64, u64)>,
    /// A gap was reported but no change after it has arrived yet (its start).
    pub open: Option<u64>,
}

impl GapTracker {
    /// Max gaps remembered (the oldest are dropped).
    pub const MAX_GAPS: usize = 4096;

    /// Account for `chunk` (a chunk of another config resets the tracker to that config).
    pub fn on_chunk(&mut self, chunk: &WaveChunk) {
        if chunk.config != self.config {
            *self = GapTracker {
                config: chunk.config,
                ..GapTracker::default()
            };
        }
        if chunk.gap && self.open.is_none() {
            self.open = Some(self.last.unwrap_or(0));
        }
        let (Some(first), Some(newest)) = (
            chunk.changes.iter().map(|c| c.tick).min(),
            chunk.changes.iter().map(|c| c.tick).max(),
        ) else {
            return;
        };
        if let Some(from) = self.open.take()
            && first > from
        {
            if self.gaps.len() >= Self::MAX_GAPS {
                self.gaps.remove(0);
            }
            self.gaps.push((from, first));
        }
        self.last = Some(self.last.map_or(newest, |l| l.max(newest)));
    }

    /// Gaps overlapping `[t0, t1]`.
    pub fn visible(&self, t0: f64, t1: f64) -> impl Iterator<Item = (u64, u64)> + '_ {
        let i = self.gaps.partition_point(|&(_, b)| (b as f64) < t0);
        self.gaps[i..]
            .iter()
            .copied()
            .take_while(move |&(a, _)| (a as f64) <= t1)
    }
}

/// Signal names offered by the picker for the flat net names of a design: bus bits `x[i]`
/// collapse to `x`; top-level names come first, then by depth and name.
pub fn signal_candidates<'a>(net_names: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let mut v: Vec<String> = net_names
        .into_iter()
        .map(|n| match n.rfind('[') {
            Some(i) if n.ends_with(']') && n[i + 1..n.len() - 1].parse::<u32>().is_ok() => &n[..i],
            _ => n,
        })
        .filter(|n| !n.is_empty())
        .map(str::to_string)
        .collect();
    v.sort_by(|a, b| {
        let da = a.matches('/').count();
        let db = b.matches('/').count();
        da.cmp(&db).then_with(|| a.cmp(b))
    });
    v.dedup();
    v
}

/// Candidates matching every whitespace-separated term of `query` (case-insensitive), at most
/// `limit` of them.
pub fn filter_candidates<'a>(cands: &'a [String], query: &str, limit: usize) -> Vec<&'a str> {
    let terms: Vec<String> = query.split_whitespace().map(str::to_lowercase).collect();
    cands
        .iter()
        .filter(|c| {
            let l = c.to_lowercase();
            terms.iter().all(|t| l.contains(t.as_str()))
        })
        .take(limit)
        .map(String::as_str)
        .collect()
}

/// How a signal's value is printed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Radix {
    /// Hexadecimal (`x` / `z` per fully unknown nibble, `?` for mixed).
    #[default]
    Hex,
    /// Binary, MSB first.
    Bin,
    /// Unsigned decimal.
    Dec,
    /// Two's-complement decimal.
    Signed,
}

impl Radix {
    /// Every radix, for menus.
    pub const ALL: [Radix; 4] = [Radix::Hex, Radix::Bin, Radix::Dec, Radix::Signed];

    /// Menu label.
    pub fn label(self) -> &'static str {
        match self {
            Radix::Hex => "Hex",
            Radix::Bin => "Binary",
            Radix::Dec => "Unsigned",
            Radix::Signed => "Signed",
        }
    }
}

fn bit(p: Packed, i: u32) -> Logic {
    Logic::from_planes(p.a >> i & 1 == 1, p.b >> i & 1 == 1)
}

fn mask(width: u32) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

/// True if every bit of the `width`-bit value is 0 or 1.
pub fn is_known(p: Packed, width: u32) -> bool {
    p.b & mask(width) == 0
}

/// Format a `width`-bit value in `radix`. Unknown bits make decimal radices print `x` (all X),
/// `z` (all Z) or `?` (mixed).
pub fn format_value(p: Packed, width: u32, radix: Radix) -> String {
    let width = width.clamp(1, 64);
    let m = mask(width);
    let (a, b) = (p.a & m, p.b & m);
    if width == 1 {
        return bit(p, 0).to_string();
    }
    let unknown = |a: u64, b: u64, all: u64| -> char {
        if b != all {
            '?'
        } else if a == all {
            'x'
        } else if a == 0 {
            'z'
        } else {
            '?'
        }
    };
    match radix {
        Radix::Bin => (0..width).rev().map(|i| bit(p, i).to_string()).collect(),
        Radix::Hex => {
            let nibbles = width.div_ceil(4);
            let mut s = String::with_capacity(nibbles as usize);
            for k in (0..nibbles).rev() {
                let lo = k * 4;
                let n = (width - lo).min(4);
                let nm = (1u64 << n) - 1;
                let (na, nb) = (a >> lo & nm, b >> lo & nm);
                if nb == 0 {
                    s.push(char::from_digit(na as u32, 16).expect("nibble"));
                } else {
                    s.push(unknown(na, nb, nm));
                }
            }
            s
        }
        Radix::Dec | Radix::Signed if b != 0 => unknown(a, b, m).to_string(),
        Radix::Dec => a.to_string(),
        Radix::Signed => {
            if width < 64 && a >> (width - 1) & 1 == 1 {
                format!("-{}", (!a & m) + 1)
            } else {
                (a as i64).to_string()
            }
        }
    }
}

/// A drawable piece of a signal row, in pixels from the plot's left edge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Run {
    /// Left edge.
    pub x0: f64,
    /// Right edge.
    pub x1: f64,
    /// What to draw.
    pub kind: RunKind,
}

/// Content of a [`Run`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RunKind {
    /// A steady value.
    Value(Packed),
    /// Several changes per pixel (drawn as a filled block).
    Dense,
}

/// Decimate the change history `h` (sorted by tick) for the visible range of `view` over
/// `width` pixels, up to tick `horizon` (nothing is drawn past it, nor before the first
/// recorded change). A change is placed at its exact sub-pixel position when it is alone in its
/// pixel column; columns with two or more changes become [`RunKind::Dense`] (adjacent dense
/// columns merge).
pub fn runs(h: &VecDeque<(u64, Packed)>, view: &WaveView, width: f64, horizon: u64) -> Vec<Run> {
    let mut out: Vec<Run> = Vec::new();
    let (t0, t1) = view.range(width);
    let t1 = t1.min(horizon as f64);
    if t1 < t0 || h.is_empty() {
        return out;
    }
    let x_end = view.x_of(t1).min(width);
    // First change after t0.
    let mut i = h.partition_point(|&(t, _)| (t as f64) <= t0);
    let mut cur: Option<Packed> = (i > 0).then(|| h[i - 1].1);
    let mut x = 0.0f64;
    let push = |out: &mut Vec<Run>, x0: f64, x1: f64, kind: RunKind| {
        if x1 <= x0 {
            return;
        }
        if let (RunKind::Dense, Some(last)) = (kind, out.last_mut())
            && last.kind == RunKind::Dense
            && last.x1 >= x0 - 1e-9
        {
            last.x1 = x1;
            return;
        }
        out.push(Run { x0, x1, kind });
    };
    while i < h.len() && (h[i].0 as f64) <= t1 {
        let tx = view.x_of(h[i].0 as f64).max(0.0);
        let col = tx.floor();
        // Changes inside this pixel column.
        let col_end = view.tick_at(col + 1.0);
        let j = h.partition_point(|&(t, _)| (t as f64) < col_end);
        let j = j.max(i + 1);
        if j == i + 1 {
            if let Some(v) = cur {
                push(&mut out, x, tx, RunKind::Value(v));
            }
            cur = Some(h[i].1);
            x = tx;
        } else {
            if let Some(v) = cur {
                push(&mut out, x, col, RunKind::Value(v));
            }
            // The last column may be partial (clipped at the horizon).
            let x1 = (col + 1.0).min(x_end).max(col + 1e-6);
            push(&mut out, col, x1, RunKind::Dense);
            cur = Some(h[j - 1].1);
            x = col + 1.0;
        }
        i = j;
    }
    if let Some(v) = cur
        && x < x_end
    {
        push(&mut out, x, x_end, RunKind::Value(v));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(v: u64) -> Packed {
        Packed { a: v, b: 0 }
    }

    #[test]
    fn zoom_keeps_anchor_and_follow_keeps_right_edge() {
        let mut v = WaveView {
            start: 100.0,
            tpp: 2.0,
            follow: false,
            cursor: None,
        };
        let anchor = v.tick_at(50.0);
        v.zoom_at(50.0, 400.0, 2.0);
        assert_eq!(v.tpp, 1.0);
        assert!((v.tick_at(50.0) - anchor).abs() < 1e-9);
        // Clamped.
        v.zoom_at(0.0, 400.0, 1e9);
        assert_eq!(v.tpp, MIN_TPP);
        // Following: right edge anchored.
        let mut f = WaveView::default();
        f.follow_to(1000, 400.0);
        assert_eq!(f.range(400.0), (600.0, 1000.0));
        f.zoom_at(10.0, 400.0, 0.5);
        assert_eq!(f.range(400.0), (200.0, 1000.0));
        f.scroll(100.0);
        assert!(!f.follow);
        assert_eq!(f.start, 0.0);
        // Not following: follow_to does nothing.
        f.follow_to(5000, 400.0);
        assert_eq!(f.start, 0.0);
        f.fit(10, 410, 400.0);
        assert_eq!((f.start, f.tpp), (10.0, 1.0));
        f.set_cursor_at(20.4);
        assert_eq!(f.cursor, Some(30));
    }

    #[test]
    fn nice_steps() {
        assert_eq!(nice_step(1.0, 80.0), 100);
        assert_eq!(nice_step(0.1, 80.0), 10);
        assert_eq!(nice_step(0.01, 80.0), 1);
        assert_eq!(nice_step(3.0, 80.0), 500);
        assert_eq!(nice_step(15.0, 80.0), 2000);
    }

    #[test]
    fn radix_formatting() {
        assert_eq!(format_value(p(0x2a), 8, Radix::Hex), "2a");
        assert_eq!(format_value(p(0x2a), 8, Radix::Bin), "00101010");
        assert_eq!(format_value(p(0xfe), 8, Radix::Dec), "254");
        assert_eq!(format_value(p(0xfe), 8, Radix::Signed), "-2");
        assert_eq!(format_value(p(0x7f), 8, Radix::Signed), "127");
        assert_eq!(format_value(p(0x5), 3, Radix::Hex), "5");
        assert_eq!(format_value(p(1), 1, Radix::Hex), "1");
        // X in the high nibble, Z in a low bit.
        let v = Packed { a: 0xf0, b: 0xf1 };
        assert_eq!(format_value(v, 8, Radix::Hex), "x?");
        assert_eq!(format_value(v, 8, Radix::Bin), "xxxx000z");
        assert_eq!(format_value(v, 8, Radix::Dec), "?");
        assert_eq!(format_value(Packed { a: 0, b: 0xff }, 8, Radix::Dec), "z");
        assert_eq!(format_value(Packed { a: 1, b: 1 }, 1, Radix::Hex), "x");
        assert!(is_known(p(3), 2) && !is_known(Packed { a: 0, b: 4 }, 3));
        assert_eq!(format_value(p(u64::MAX), 64, Radix::Signed), "-1");
    }

    #[test]
    fn runs_place_sparse_changes_exactly() {
        let h: VecDeque<_> = [(0, p(0)), (10, p(1)), (25, p(2))].into_iter().collect();
        let v = WaveView {
            start: 0.0,
            tpp: 1.0,
            follow: false,
            cursor: None,
        };
        let r = runs(&h, &v, 100.0, 40);
        assert_eq!(
            r,
            vec![
                Run {
                    x0: 0.0,
                    x1: 10.0,
                    kind: RunKind::Value(p(0))
                },
                Run {
                    x0: 10.0,
                    x1: 25.0,
                    kind: RunKind::Value(p(1))
                },
                Run {
                    x0: 25.0,
                    x1: 40.0,
                    kind: RunKind::Value(p(2))
                },
            ]
        );
        // Starting mid-run: the value at the left edge is known.
        let v2 = WaveView { start: 12.0, ..v };
        let r = runs(&h, &v2, 100.0, 40);
        assert_eq!(
            r[0],
            Run {
                x0: 0.0,
                x1: 13.0,
                kind: RunKind::Value(p(1))
            }
        );
        // Before the first change nothing is drawn.
        let h2: VecDeque<_> = [(50, p(7))].into_iter().collect();
        let r = runs(&h2, &v, 100.0, 80);
        assert_eq!(
            r,
            vec![Run {
                x0: 50.0,
                x1: 80.0,
                kind: RunKind::Value(p(7))
            }]
        );
    }

    #[test]
    fn runs_decimate_dense_history() {
        // A change every tick for a million ticks, viewed over 100 px.
        let h: VecDeque<_> = (0..1_000_000u64).map(|t| (t, p(t & 0xff))).collect();
        let v = WaveView {
            start: 0.0,
            tpp: 10_000.0,
            follow: false,
            cursor: None,
        };
        let r = runs(&h, &v, 100.0, 1_000_000);
        assert_eq!(r.len(), 1, "{r:?}");
        assert_eq!(r[0].kind, RunKind::Dense);
        assert!(r[0].x0 <= 0.0 && r[0].x1 >= 99.0, "{r:?}");
        // Mixed: a dense burst between quiet stretches.
        let mut h: VecDeque<_> = VecDeque::new();
        h.push_back((0, p(0)));
        for t in 500..520 {
            h.push_back((t, p(t)));
        }
        h.push_back((900, p(1)));
        let v = WaveView { tpp: 10.0, ..v };
        let r = runs(&h, &v, 100.0, 1000);
        let kinds: Vec<_> = r.iter().map(|r| r.kind).collect();
        assert_eq!(
            kinds,
            vec![
                RunKind::Value(p(0)),
                RunKind::Dense,
                RunKind::Value(p(519)),
                RunKind::Value(p(1))
            ],
            "{r:?}"
        );
        assert_eq!((r[1].x0, r[1].x1), (50.0, 52.0));
        assert_eq!((r[3].x0, r[3].x1), (90.0, 100.0));
    }
}
