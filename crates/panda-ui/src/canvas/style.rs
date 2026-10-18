//! Canvas colours and the wire look of a live value.

use egui::Color32;
use panda_netlist::{Logic, Value};

use crate::theme;

/// Canvas palette (light or dark). The light palette is the Panda look (see [`crate::theme`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Palette {
    /// Background.
    pub bg: Color32,
    /// Minor grid dots.
    pub grid: Color32,
    /// Major grid dots (every 10 units).
    pub grid_major: Color32,
    /// Ink: outlines, pin sockets, splitter bars.
    pub stroke: Color32,
    /// Light fill (pin pills, sockets).
    pub fill: Color32,
    /// Default package (chip) body.
    pub package_fill: Color32,
    /// Pin-marker fill.
    pub package_header: Color32,
    /// Chip body colours; packages pick one from their name, gates use the first.
    pub chips: [Color32; 4],
    /// Body of sources (constants, clocks, power-on reset).
    pub source: Color32,
    /// Text on chip bodies.
    pub chip_text: Color32,
    /// Secondary text on chip bodies (pin names).
    pub chip_text_dim: Color32,
    /// Text.
    pub text: Color32,
    /// Secondary text (pin labels).
    pub text_dim: Color32,
    /// Wire without a live value.
    pub wire: Color32,
    /// Logic 0.
    pub zero: Color32,
    /// Logic 1.
    pub one: Color32,
    /// Unknown.
    pub x: Color32,
    /// High impedance.
    pub z: Color32,
    /// Contention.
    pub contention: Color32,
    /// Known non-zero bus value (bus core; an all-zero bus uses `zero`).
    pub bus: Color32,
    /// Outer casing of buses and the background of bus value badges.
    pub bus_casing: Color32,
    /// Selection / highlight accent.
    pub accent: Color32,
}

impl Palette {
    /// Light theme: the Panda eggshell palette.
    pub const LIGHT: Palette = Palette {
        bg: theme::EGGSHELL,
        grid: theme::DUNE,
        grid_major: Color32::from_rgb(0xD6, 0xBC, 0x8E),
        stroke: theme::NAVY,
        fill: theme::CREAM,
        package_fill: theme::NAVY,
        package_header: theme::CREAM,
        chips: [theme::NAVY, theme::MAROON, theme::TEAL, theme::BRICK],
        source: theme::STEEL,
        chip_text: theme::EGGSHELL,
        chip_text_dim: theme::ICE,
        text: theme::NAVY,
        text_dim: theme::SLATE,
        wire: Color32::from_rgb(0x9A, 0x8C, 0x80),
        zero: theme::SLATE,
        one: theme::RED,
        x: theme::AMBER,
        z: theme::MIST,
        contention: theme::PLUM,
        bus: theme::RED,
        bus_casing: theme::NAVY,
        accent: theme::STEEL,
    };

    /// Dark theme: the same colours on a navy background.
    pub const DARK: Palette = Palette {
        bg: Color32::from_rgb(0x00, 0x1E, 0x2E),
        grid: Color32::from_rgb(0x0A, 0x2C, 0x3F),
        grid_major: Color32::from_rgb(0x15, 0x3B, 0x52),
        stroke: theme::EGGSHELL,
        fill: Color32::from_rgb(0x0A, 0x2C, 0x3F),
        package_fill: theme::TEAL,
        package_header: Color32::from_rgb(0x0A, 0x2C, 0x3F),
        chips: [theme::TEAL, theme::MAROON, theme::STEEL, theme::BRICK],
        source: theme::STEEL,
        chip_text: theme::EGGSHELL,
        chip_text_dim: theme::ICE,
        text: theme::EGGSHELL,
        text_dim: theme::MIST,
        wire: Color32::from_rgb(0x5A, 0x6E, 0x7A),
        zero: Color32::from_rgb(0x3E, 0x5C, 0x6C),
        one: Color32::from_rgb(0xE6, 0x2B, 0x38),
        x: theme::AMBER,
        z: theme::MIST,
        contention: Color32::from_rgb(0xC0, 0x4F, 0xB6),
        bus: Color32::from_rgb(0xE6, 0x2B, 0x38),
        bus_casing: theme::EGGSHELL,
        accent: theme::AMBER,
    };

    /// The palette matching egui's dark/light mode.
    pub fn for_dark_mode(dark: bool) -> Palette {
        if dark { Palette::DARK } else { Palette::LIGHT }
    }

    /// Chip body colour for a package called `name` (stable per name).
    pub fn chip_for(&self, name: &str) -> Color32 {
        self.chips[chip_index(name)]
    }
}

/// Index into [`Palette::chips`] for a package called `name` (FNV-1a of the name).
pub fn chip_index(name: &str) -> usize {
    let mut h: u32 = 0x811c_9dc5;
    for b in name.bytes() {
        h ^= u32::from(b);
        h = h.wrapping_mul(0x0100_0193);
    }
    (h % 4) as usize
}

/// How a wire is drawn.
#[derive(Clone, Debug, PartialEq)]
pub struct WireLook {
    /// Colour.
    pub color: Color32,
    /// Width multiplier (buses are thicker).
    pub width: f32,
    /// Value label for buses (hex), if any.
    pub label: Option<String>,
}

/// Look of a wire of `width` bits carrying `value` (`None`: no live value).
pub fn wire_look(p: &Palette, width: u32, value: Option<&Value>, contention: bool) -> WireLook {
    let bus = width > 1;
    let w = if bus { 2.0 } else { 1.0 };
    let Some(v) = value else {
        return WireLook {
            color: p.wire,
            width: w,
            label: None,
        };
    };
    if contention {
        return WireLook {
            color: p.contention,
            width: w,
            label: bus.then(|| hex_label(v)),
        };
    }
    if !bus {
        let color = match v.get(0) {
            Logic::Zero => p.zero,
            Logic::One => p.one,
            Logic::X => p.x,
            Logic::Z => p.z,
        };
        return WireLook {
            color,
            width: w,
            label: None,
        };
    }
    let mut any_x = false;
    let mut all_z = true;
    let mut any_z = false;
    for b in v.bits() {
        match b {
            Logic::X => any_x = true,
            Logic::Z => any_z = true,
            _ => {}
        }
        all_z &= b == Logic::Z;
    }
    let color = if all_z {
        p.z
    } else if any_x || any_z {
        p.x
    } else if v.bits().all(|b| b == Logic::Zero) {
        p.zero
    } else {
        p.bus
    };
    WireLook {
        color,
        width: w,
        label: Some(hex_label(v)),
    }
}

/// Hex label of a value: `0x3f`; nibbles containing X / Z show `x` / `z`.
pub fn hex_label(v: &Value) -> String {
    let w = v.width();
    let digits = w.div_ceil(4);
    let mut s = String::with_capacity(digits as usize + 2);
    s.push_str("0x");
    for d in (0..digits).rev() {
        let mut n = 0u32;
        let mut x = false;
        let mut z = 0;
        let mut bits = 0;
        for i in 0..4 {
            let b = d * 4 + i;
            if b >= w {
                break;
            }
            bits += 1;
            match v.get(b) {
                Logic::One => n |= 1 << i,
                Logic::Zero => {}
                Logic::X => x = true,
                Logic::Z => z += 1,
            }
        }
        s.push(if x || (z > 0 && z < bits) {
            'x'
        } else if z > 0 {
            'z'
        } else {
            char::from_digit(n, 16).expect("nibble")
        });
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_labels() {
        assert_eq!(hex_label(&Value::from_u64(8, 0x3f)), "0x3f");
        assert_eq!(hex_label(&Value::from_u64(5, 0x13)), "0x13");
        let mut v = Value::from_u64(8, 0xa5);
        v.set(5, Logic::X);
        assert_eq!(hex_label(&v), "0xx5");
        assert_eq!(hex_label(&Value::zs(8)), "0xzz");
    }

    #[test]
    fn looks() {
        let p = Palette::LIGHT;
        assert_eq!(
            wire_look(&p, 1, Some(&Value::from_bool(true)), false).color,
            p.one
        );
        assert_eq!(wire_look(&p, 1, Some(&Value::xs(1)), false).color, p.x);
        assert_eq!(wire_look(&p, 1, Some(&Value::zs(1)), false).color, p.z);
        assert_eq!(
            wire_look(&p, 1, Some(&Value::zeros(1)), true).color,
            p.contention
        );
        let b = wire_look(&p, 8, Some(&Value::from_u64(8, 7)), false);
        assert_eq!((b.color, b.label.as_deref()), (p.bus, Some("0x07")));
        assert!(b.width > 1.0);
        let b0 = wire_look(&p, 8, Some(&Value::zeros(8)), false);
        assert_eq!((b0.color, b0.label.as_deref()), (p.zero, Some("0x00")));
        assert_eq!(wire_look(&p, 8, None, false).color, p.wire);
    }
}
