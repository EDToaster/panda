//! Canvas colours and the wire look of a live value.

use egui::Color32;
use panda_netlist::{Logic, Value};

/// Canvas palette (light or dark).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Palette {
    /// Background.
    pub bg: Color32,
    /// Minor grid lines.
    pub grid: Color32,
    /// Major grid lines.
    pub grid_major: Color32,
    /// Symbol outline.
    pub stroke: Color32,
    /// Symbol fill.
    pub fill: Color32,
    /// Package box fill.
    pub package_fill: Color32,
    /// Package header fill.
    pub package_header: Color32,
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
    /// Known bus value.
    pub bus: Color32,
    /// Selection / highlight accent.
    pub accent: Color32,
}

impl Palette {
    /// Light theme.
    pub const LIGHT: Palette = Palette {
        bg: Color32::from_rgb(250, 250, 247),
        grid: Color32::from_rgb(236, 236, 232),
        grid_major: Color32::from_rgb(222, 222, 216),
        stroke: Color32::from_rgb(40, 44, 52),
        fill: Color32::from_rgb(255, 255, 255),
        package_fill: Color32::from_rgb(244, 247, 252),
        package_header: Color32::from_rgb(222, 230, 244),
        text: Color32::from_rgb(30, 32, 38),
        text_dim: Color32::from_rgb(96, 100, 110),
        wire: Color32::from_rgb(120, 124, 132),
        zero: Color32::from_rgb(28, 110, 48),
        one: Color32::from_rgb(40, 200, 70),
        x: Color32::from_rgb(220, 40, 40),
        z: Color32::from_rgb(40, 110, 230),
        contention: Color32::from_rgb(230, 0, 200),
        bus: Color32::from_rgb(20, 60, 130),
        accent: Color32::from_rgb(255, 140, 0),
    };

    /// Dark theme.
    pub const DARK: Palette = Palette {
        bg: Color32::from_rgb(24, 26, 30),
        grid: Color32::from_rgb(34, 37, 42),
        grid_major: Color32::from_rgb(44, 48, 55),
        stroke: Color32::from_rgb(200, 205, 214),
        fill: Color32::from_rgb(38, 41, 47),
        package_fill: Color32::from_rgb(32, 37, 46),
        package_header: Color32::from_rgb(46, 56, 74),
        text: Color32::from_rgb(225, 228, 235),
        text_dim: Color32::from_rgb(150, 156, 168),
        wire: Color32::from_rgb(120, 126, 136),
        zero: Color32::from_rgb(30, 120, 55),
        one: Color32::from_rgb(70, 235, 100),
        x: Color32::from_rgb(255, 75, 75),
        z: Color32::from_rgb(80, 150, 255),
        contention: Color32::from_rgb(255, 60, 230),
        bus: Color32::from_rgb(110, 170, 255),
        accent: Color32::from_rgb(255, 165, 40),
    };

    /// The palette matching egui's dark/light mode.
    pub fn for_dark_mode(dark: bool) -> Palette {
        if dark { Palette::DARK } else { Palette::LIGHT }
    }
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
    let w = if bus { 2.2 } else { 1.0 };
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
        assert_eq!(wire_look(&p, 8, None, false).color, p.wire);
    }
}
