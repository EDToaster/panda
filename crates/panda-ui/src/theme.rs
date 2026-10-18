//! The Panda look: a small, warm palette in the spirit of Sebastian Lague's Digital Logic Sim.
//!
//! Base colours (given):
//!
//! | name      | hex       | used for                                  |
//! |-----------|-----------|-------------------------------------------|
//! | maroon    | `#780000` | NAND / chip bodies, hot accents           |
//! | red       | `#C1121F` | logic 1 ("on" wires)                      |
//! | eggshell  | `#FDF0D5` | background (canvas and panels)            |
//! | navy      | `#003049` | ink: text, chip bodies, bus casings       |
//! | steel     | `#669BBC` | sources, bus cores, selection             |
//!
//! Derived colours complement those: darker / lighter eggshell tones for the grid and widget
//! fills, a slate blue-grey for logic 0 ("off" wires), amber for X (ochre when used as text), a
//! pale steel for Z and selections, plum for contention and two extra chip tones (deep teal and brick) so packages get distinct colours.

use egui::{Color32, CornerRadius, Stroke, Theme, Visuals};

/// `#780000`.
pub const MAROON: Color32 = Color32::from_rgb(0x78, 0x00, 0x00);
/// `#C1121F`.
pub const RED: Color32 = Color32::from_rgb(0xC1, 0x12, 0x1F);
/// `#FDF0D5`.
pub const EGGSHELL: Color32 = Color32::from_rgb(0xFD, 0xF0, 0xD5);
/// `#003049`.
pub const NAVY: Color32 = Color32::from_rgb(0x00, 0x30, 0x49);
/// `#669BBC`.
pub const STEEL: Color32 = Color32::from_rgb(0x66, 0x9B, 0xBC);

/// Slightly darker eggshell (side panels, widget fills).
pub const SAND: Color32 = Color32::from_rgb(0xF6, 0xE4, 0xC0);
/// Darker sand (hovered widgets, grid dots).
pub const DUNE: Color32 = Color32::from_rgb(0xE8, 0xD2, 0xA8);
/// Lightest cream (text fields, pin pills).
pub const CREAM: Color32 = Color32::from_rgb(0xFF, 0xF9, 0xEC);
/// Slate blue-grey, between navy and steel (logic 0, secondary text).
pub const SLATE: Color32 = Color32::from_rgb(0x4F, 0x6D, 0x7A);
/// Deep teal chip tone (between navy and steel).
pub const TEAL: Color32 = Color32::from_rgb(0x1D, 0x5C, 0x7A);
/// Brick chip tone (between maroon and red).
pub const BRICK: Color32 = Color32::from_rgb(0x9E, 0x2A, 0x2B);
/// Amber (unknown values, warnings).
pub const AMBER: Color32 = Color32::from_rgb(0xE0, 0x9F, 0x3E);
/// Ochre: a darker amber, legible as text on eggshell (warnings).
pub const OCHRE: Color32 = Color32::from_rgb(0xA8, 0x65, 0x0F);
/// Ice: a very pale steel (selected toggles, text selection).
pub const ICE: Color32 = Color32::from_rgb(0xC9, 0xDC, 0xE8);
/// Pale steel (high impedance).
pub const MIST: Color32 = Color32::from_rgb(0xA9, 0xC6, 0xD8);
/// Plum (contention).
pub const PLUM: Color32 = Color32::from_rgb(0x8E, 0x2C, 0x84);

/// egui visuals for the app chrome: eggshell panels, navy ink, steel selection.
pub fn visuals() -> Visuals {
    let mut v = Visuals::light();
    let r = CornerRadius::same(6);
    v.override_text_color = Some(NAVY);
    v.weak_text_color = Some(SLATE);
    v.hyperlink_color = RED;
    v.panel_fill = SAND;
    v.window_fill = EGGSHELL;
    v.window_stroke = Stroke::new(1.0, DUNE);
    v.window_corner_radius = CornerRadius::same(10);
    v.menu_corner_radius = CornerRadius::same(8);
    v.faint_bg_color = EGGSHELL;
    v.extreme_bg_color = CREAM;
    v.code_bg_color = DUNE;
    v.warn_fg_color = OCHRE;
    v.error_fg_color = RED;
    v.selection.bg_fill = ICE;
    v.selection.stroke = Stroke::new(1.0, NAVY);
    let w = &mut v.widgets;
    w.noninteractive.bg_fill = SAND;
    w.noninteractive.weak_bg_fill = SAND;
    w.noninteractive.bg_stroke = Stroke::new(1.0, DUNE);
    w.noninteractive.fg_stroke = Stroke::new(1.0, NAVY);
    w.noninteractive.corner_radius = r;
    w.inactive.bg_fill = DUNE;
    w.inactive.weak_bg_fill = SAND;
    w.inactive.bg_stroke = Stroke::NONE;
    w.inactive.fg_stroke = Stroke::new(1.0, NAVY);
    w.inactive.corner_radius = r;
    w.hovered.bg_fill = DUNE;
    w.hovered.weak_bg_fill = DUNE;
    w.hovered.bg_stroke = Stroke::new(1.0, STEEL);
    w.hovered.fg_stroke = Stroke::new(1.5, NAVY);
    w.hovered.corner_radius = r;
    w.active.bg_fill = STEEL;
    w.active.weak_bg_fill = STEEL;
    w.active.bg_stroke = Stroke::new(1.0, NAVY);
    w.active.fg_stroke = Stroke::new(2.0, NAVY);
    w.active.corner_radius = r;
    w.open.bg_fill = DUNE;
    w.open.weak_bg_fill = DUNE;
    w.open.bg_stroke = Stroke::new(1.0, STEEL);
    w.open.fg_stroke = Stroke::new(1.0, NAVY);
    w.open.corner_radius = r;
    v
}

/// Install the Panda look (light theme, [`visuals`]) into `ctx`. Cheap to call every frame.
pub fn apply(ctx: &egui::Context) {
    let want = visuals();
    if ctx.style_of(Theme::Light).visuals != want {
        ctx.set_visuals_of(Theme::Light, want);
    }
    if ctx.theme() != Theme::Light {
        ctx.set_theme(Theme::Light);
    }
}
