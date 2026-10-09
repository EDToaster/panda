//! The canvas view transform: world (grid units) ↔ screen (points).

use egui::{Pos2, Rect, Vec2, pos2};

use crate::geom::{GridPt, GridRect};

/// Minimum zoom (screen points per grid unit).
pub const MIN_ZOOM: f32 = 0.002;
/// Maximum zoom.
pub const MAX_ZOOM: f32 = 80.0;

/// World → screen: `screen = origin + world * zoom`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewXf {
    /// Screen position of world (0, 0).
    pub origin: Pos2,
    /// Screen points per grid unit.
    pub zoom: f32,
}

impl Default for ViewXf {
    fn default() -> Self {
        ViewXf {
            origin: pos2(40.0, 40.0),
            zoom: 12.0,
        }
    }
}

impl ViewXf {
    /// World point (fractional grid units) to screen.
    pub fn to_screen(&self, w: Pos2) -> Pos2 {
        self.origin + w.to_vec2() * self.zoom
    }

    /// Grid point to screen.
    pub fn grid_to_screen(&self, p: GridPt) -> Pos2 {
        self.to_screen(pos2(p.x as f32, p.y as f32))
    }

    /// Screen to world (fractional grid units).
    pub fn to_world(&self, s: Pos2) -> Pos2 {
        ((s - self.origin) / self.zoom).to_pos2()
    }

    /// Screen to the nearest grid point.
    pub fn to_grid(&self, s: Pos2) -> GridPt {
        let w = self.to_world(s);
        GridPt {
            x: w.x.round() as i32,
            y: w.y.round() as i32,
        }
    }

    /// Screen rectangle of a grid rectangle.
    pub fn rect_to_screen(&self, r: &GridRect) -> Rect {
        Rect::from_min_max(self.grid_to_screen(r.min), self.grid_to_screen(r.max))
    }

    /// World rectangle visible in screen rectangle `screen`.
    pub fn visible_world(&self, screen: Rect) -> Rect {
        Rect::from_min_max(self.to_world(screen.min), self.to_world(screen.max))
    }

    /// Zoom by `factor`, keeping the world point under screen point `anchor` fixed.
    pub fn zoom_at(&mut self, anchor: Pos2, factor: f32) {
        let w = self.to_world(anchor);
        self.zoom = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        self.origin = anchor - w.to_vec2() * self.zoom;
    }

    /// Pan by a screen delta.
    pub fn pan(&mut self, delta: Vec2) {
        self.origin += delta;
    }

    /// Fit world rectangle `world` into screen rectangle `screen` with a margin (points).
    pub fn fit(&mut self, world: Rect, screen: Rect, margin: f32) {
        let avail = (screen.size() - Vec2::splat(2.0 * margin)).max(Vec2::splat(1.0));
        let size = world.size().max(Vec2::splat(1.0));
        self.zoom = (avail.x / size.x)
            .min(avail.y / size.y)
            .clamp(MIN_ZOOM, MAX_ZOOM);
        self.origin = screen.center() - world.center().to_vec2() * self.zoom;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zoom_is_cursor_anchored() {
        let mut v = ViewXf::default();
        let anchor = pos2(300.0, 200.0);
        let w = v.to_world(anchor);
        v.zoom_at(anchor, 1.7);
        assert!((v.to_world(anchor) - w).length() < 1e-4);
        v.zoom_at(anchor, 0.2);
        assert!((v.to_world(anchor) - w).length() < 1e-3);
    }

    #[test]
    fn fit_contains_world() {
        let mut v = ViewXf::default();
        let world = Rect::from_min_max(pos2(-10.0, 5.0), pos2(300.0, 90.0));
        let screen = Rect::from_min_max(pos2(0.0, 0.0), pos2(800.0, 600.0));
        v.fit(world, screen, 10.0);
        let s = Rect::from_min_max(v.to_screen(world.min), v.to_screen(world.max));
        assert!(screen.expand(0.01).contains_rect(s), "{s:?}");
    }
}
