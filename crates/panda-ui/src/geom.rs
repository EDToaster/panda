//! Schematic geometry shared by the document model, auto-layout, the canvas and editing tools.
//!
//! Coordinates are integer **grid units** ([`GridPt`]); the canvas scales them to pixels. Every
//! placeable item (atom instance, package instance, the def's own pin markers) has a [`Symbol`]:
//! a size and a list of ports in *local* coordinates (origin = top-left, x right, y down).
//! An instance's [`Orient`] (rotation in 90° steps + horizontal flip) maps local coordinates to
//! the rotated box whose top-left corner is the instance position.
//!
//! Port conventions (all ports lie on the symbol boundary, on grid points):
//! * gates: inputs on the left, spaced 2 apart, output on the right;
//! * `Tri`: `d` left, `out` right, `en` at the bottom;
//! * splitters: `whole` left, parts right (top to bottom = part order);
//! * packages: `In` pins left, `Out`/`InOut` pins right (or as given by a [`SymbolLayout`]);
//! * a def's own `In` pins are markers with their port on the right, `Out`/`InOut` on the left.

use std::collections::BTreeMap;

use panda_netlist::{Atom, ComponentDef, Dir, GateOp, PinDecl};
use serde::{Deserialize, Serialize};

/// A point on the integer grid. Serialized as `[x, y]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GridPt {
    /// Grid x (right).
    pub x: i32,
    /// Grid y (down).
    pub y: i32,
}

/// Shorthand constructor.
pub const fn gp(x: i32, y: i32) -> GridPt {
    GridPt { x, y }
}

impl Serialize for GridPt {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        [self.x, self.y].serialize(s)
    }
}

impl<'de> Deserialize<'de> for GridPt {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<GridPt, D::Error> {
        let [x, y] = <[i32; 2]>::deserialize(d)?;
        Ok(GridPt { x, y })
    }
}

impl std::ops::Add for GridPt {
    type Output = GridPt;
    fn add(self, o: GridPt) -> GridPt {
        gp(self.x + o.x, self.y + o.y)
    }
}

impl std::ops::Sub for GridPt {
    type Output = GridPt;
    fn sub(self, o: GridPt) -> GridPt {
        gp(self.x - o.x, self.y - o.y)
    }
}

/// An axis-aligned rectangle in grid units (`min` inclusive, `max` inclusive corner).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GridRect {
    /// Top-left corner.
    pub min: GridPt,
    /// Bottom-right corner.
    pub max: GridPt,
}

impl GridRect {
    /// Rectangle from position and size.
    pub fn from_pos_size(pos: GridPt, size: GridPt) -> GridRect {
        GridRect {
            min: pos,
            max: pos + size,
        }
    }

    /// True if the interiors of the two rectangles intersect (touching edges do not count).
    pub fn overlaps(&self, o: &GridRect) -> bool {
        self.min.x < o.max.x && o.min.x < self.max.x && self.min.y < o.max.y && o.min.y < self.max.y
    }

    /// True if `p` is inside or on the boundary.
    pub fn contains(&self, p: GridPt) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }

    /// Smallest rectangle containing both.
    pub fn union(&self, o: &GridRect) -> GridRect {
        GridRect {
            min: gp(self.min.x.min(o.min.x), self.min.y.min(o.min.y)),
            max: gp(self.max.x.max(o.max.x), self.max.y.max(o.max.y)),
        }
    }
}

/// Side of a symbol a port sits on (in local, unrotated coordinates).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    /// x = 0.
    Left,
    /// x = width.
    Right,
    /// y = 0.
    Top,
    /// y = height.
    Bottom,
}

impl Side {
    /// Unit outward direction (grid).
    pub fn outward(self) -> GridPt {
        match self {
            Side::Left => gp(-1, 0),
            Side::Right => gp(1, 0),
            Side::Top => gp(0, -1),
            Side::Bottom => gp(0, 1),
        }
    }
}

/// Orientation of a placed symbol: optional horizontal flip (applied first), then `rot` × 90°
/// clockwise.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Orient {
    /// Clockwise quarter turns (0..=3).
    #[serde(default)]
    pub rot: u8,
    /// Mirror horizontally (before rotating).
    #[serde(default)]
    pub flip: bool,
}

impl Orient {
    /// Size of the rotated box of a symbol of local size `size`.
    pub fn size(self, size: GridPt) -> GridPt {
        if self.rot % 2 == 1 {
            gp(size.y, size.x)
        } else {
            size
        }
    }

    /// Map a local point of a symbol of local size `size` into the rotated box (origin top-left).
    pub fn apply(self, p: GridPt, size: GridPt) -> GridPt {
        let (w, h) = (size.x, size.y);
        let p = if self.flip { gp(w - p.x, p.y) } else { p };
        match self.rot % 4 {
            0 => p,
            1 => gp(h - p.y, p.x),
            2 => gp(w - p.x, h - p.y),
            _ => gp(p.y, w - p.x),
        }
    }

    /// [`Orient::apply`] for fractional local coordinates (shape outlines).
    pub fn apply_f(self, p: [f32; 2], size: GridPt) -> [f32; 2] {
        let (w, h) = (size.x as f32, size.y as f32);
        let p = if self.flip { [w - p[0], p[1]] } else { p };
        match self.rot % 4 {
            0 => p,
            1 => [h - p[1], p[0]],
            2 => [w - p[0], h - p[1]],
            _ => [p[1], w - p[0]],
        }
    }

    /// Side after orientation.
    pub fn side(self, s: Side) -> Side {
        let s = if self.flip {
            match s {
                Side::Left => Side::Right,
                Side::Right => Side::Left,
                o => o,
            }
        } else {
            s
        };
        let mut s = s;
        for _ in 0..self.rot % 4 {
            s = match s {
                Side::Left => Side::Top,
                Side::Top => Side::Right,
                Side::Right => Side::Bottom,
                Side::Bottom => Side::Left,
            };
        }
        s
    }
}

/// What a symbol depicts (drives the canvas glyph).
#[derive(Clone, Debug, PartialEq)]
pub enum SymbolKind {
    /// N-input gate (IEEE distinctive shape). `reduce` gates have one bus input.
    Gate {
        /// Operation.
        op: GateOp,
        /// Number of inputs.
        inputs: u32,
        /// Reduce mode.
        reduce: bool,
    },
    /// Inverter.
    Not,
    /// Buffer.
    Buf,
    /// Tri-state buffer.
    Tri,
    /// Splitter / joiner with the bit ranges of its parts.
    Splitter {
        /// `(lo, hi)` per part.
        parts: Vec<(u32, u32)>,
    },
    /// Constant with its display text.
    Constant {
        /// Display text, e.g. `0x3f` or `1`.
        text: String,
    },
    /// Clock source.
    Clock,
    /// Power-on reset source.
    PowerOnReset,
    /// A package (def) instance.
    Package {
        /// Def id.
        def: String,
        /// Title shown in the box.
        title: String,
    },
    /// A marker for one of the viewed def's own pins.
    Pin {
        /// Pin direction.
        dir: Dir,
        /// Pin name.
        name: String,
    },
}

/// One port of a symbol in local coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct PortGeom {
    /// Port (or pin) name as used in net endpoints.
    pub name: String,
    /// Position on the boundary (local grid).
    pub pos: GridPt,
    /// Boundary side.
    pub side: Side,
    /// Direction.
    pub dir: Dir,
    /// Bit width.
    pub width: u32,
}

/// Size and ports of a placeable item, in local coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct Symbol {
    /// Glyph kind.
    pub kind: SymbolKind,
    /// Local size (width, height) in grid units.
    pub size: GridPt,
    /// Ports.
    pub ports: Vec<PortGeom>,
}

impl Symbol {
    /// Port by name.
    pub fn port(&self, name: &str) -> Option<&PortGeom> {
        self.ports.iter().find(|p| p.name == name)
    }
}

/// Package symbol layout stored in a def's editor data (overrides the automatic one).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SymbolLayout {
    /// Box size (grid units).
    pub size: GridPt,
    /// Pin placements by pin name.
    pub pins: BTreeMap<String, SymbolPin>,
}

/// Placement of one pin on a package symbol.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolPin {
    /// Side of the box.
    pub side: Side,
    /// Offset along the side (grid units from the top / left corner).
    pub offset: i32,
}

/// Vertical spacing of gate inputs and package pins.
pub const PIN_PITCH: i32 = 2;

fn port(name: impl Into<String>, pos: GridPt, side: Side, dir: Dir, width: u32) -> PortGeom {
    PortGeom {
        name: name.into(),
        pos,
        side,
        dir,
        width,
    }
}

/// Short display text of a constant value: binary digits for ≤ 4 bits, hex above.
pub fn constant_text(v: &panda_netlist::Value) -> String {
    let w = v.width();
    if w == 1 {
        return v.get(0).to_char().to_string();
    }
    match v.to_u64() {
        Some(n) if w <= 64 => format!("{n:#x}"),
        _ => (0..w).rev().map(|i| v.get(i).to_char()).collect(),
    }
}

/// Symbol of an atom instance.
pub fn atom_symbol(atom: &Atom) -> Symbol {
    match atom {
        Atom::Gate {
            op,
            inputs,
            width,
            reduce,
        } => {
            let n = (*inputs).max(1) as i32;
            let h = (PIN_PITCH * n).max(4);
            let mut ports: Vec<PortGeom> = (0..n)
                .map(|i| {
                    // Two inputs on a 4-high body sit at y = 1, 3; more inputs every PIN_PITCH.
                    let y = if n == 1 {
                        h / 2
                    } else {
                        1 + PIN_PITCH * i + (h - PIN_PITCH * n) / 2
                    };
                    port(format!("in{i}"), gp(0, y), Side::Left, Dir::In, *width)
                })
                .collect();
            let ow = if *reduce { 1 } else { *width };
            ports.push(port("out", gp(4, h / 2), Side::Right, Dir::Out, ow));
            Symbol {
                kind: SymbolKind::Gate {
                    op: *op,
                    inputs: *inputs,
                    reduce: *reduce,
                },
                size: gp(4, h),
                ports,
            }
        }
        Atom::Not { width } | Atom::Buf { width } => Symbol {
            kind: if matches!(atom, Atom::Not { .. }) {
                SymbolKind::Not
            } else {
                SymbolKind::Buf
            },
            size: gp(4, 2),
            ports: vec![
                port("in", gp(0, 1), Side::Left, Dir::In, *width),
                port("out", gp(4, 1), Side::Right, Dir::Out, *width),
            ],
        },
        Atom::Tri { width } => Symbol {
            kind: SymbolKind::Tri,
            size: gp(4, 4),
            ports: vec![
                port("d", gp(0, 2), Side::Left, Dir::In, *width),
                port("en", gp(2, 4), Side::Bottom, Dir::In, 1),
                port("out", gp(4, 2), Side::Right, Dir::Out, *width),
            ],
        },
        Atom::Splitter { width, parts } => {
            let k = parts.len().max(1) as i32;
            let h = PIN_PITCH * k;
            let mut ports = vec![port("whole", gp(0, h / 2), Side::Left, Dir::InOut, *width)];
            for (i, (lo, hi)) in parts.iter().enumerate() {
                ports.push(port(
                    format!("p{i}"),
                    gp(2, 1 + PIN_PITCH * i as i32),
                    Side::Right,
                    Dir::InOut,
                    hi - lo + 1,
                ));
            }
            Symbol {
                kind: SymbolKind::Splitter {
                    parts: parts.clone(),
                },
                size: gp(2, h),
                ports,
            }
        }
        Atom::Constant { value } => {
            let text = constant_text(value);
            let w = ((text.chars().count() as f32 * 0.75).ceil() as i32 + 1).max(2);
            Symbol {
                kind: SymbolKind::Constant { text },
                size: gp(w, 2),
                ports: vec![port("out", gp(w, 1), Side::Right, Dir::Out, value.width())],
            }
        }
        Atom::Clock { .. } => Symbol {
            kind: SymbolKind::Clock,
            size: gp(4, 2),
            ports: vec![port("out", gp(4, 1), Side::Right, Dir::Out, 1)],
        },
        Atom::PowerOnReset { .. } => Symbol {
            kind: SymbolKind::PowerOnReset,
            size: gp(4, 2),
            ports: vec![port("out", gp(4, 1), Side::Right, Dir::Out, 1)],
        },
    }
}

/// Approximate label width in grid units (canvas text at the default size: ~0.6 grid per char).
fn text_units(s: &str) -> i32 {
    (s.chars().count() as f32 * 0.6).ceil() as i32
}

/// Symbol of a package instance of `def` (uses `layout` when given and consistent with the
/// def's pins, else the automatic layout: `In` pins left, others right).
pub fn package_symbol(def: &ComponentDef, layout: Option<&SymbolLayout>) -> Symbol {
    let title = if def.name.is_empty() {
        def.id.clone()
    } else {
        def.name.clone()
    };
    let kind = SymbolKind::Package {
        def: def.id.clone(),
        title: title.clone(),
    };
    if let Some(l) = layout
        && def.pins.iter().all(|p| l.pins.contains_key(&p.name))
    {
        let ports = def
            .pins
            .iter()
            .map(|p| {
                let sp = l.pins[&p.name];
                let pos = match sp.side {
                    Side::Left => gp(0, sp.offset),
                    Side::Right => gp(l.size.x, sp.offset),
                    Side::Top => gp(sp.offset, 0),
                    Side::Bottom => gp(sp.offset, l.size.y),
                };
                port(p.name.clone(), pos, sp.side, p.dir, p.width)
            })
            .collect();
        return Symbol {
            kind,
            size: l.size,
            ports,
        };
    }
    let auto = auto_symbol_layout(def);
    package_symbol(def, Some(&auto))
}

/// The automatic package symbol layout: `In` pins on the left, `Out`/`InOut` on the right,
/// [`PIN_PITCH`] apart below a one-row header; width fits the longest labels and the title.
pub fn auto_symbol_layout(def: &ComponentDef) -> SymbolLayout {
    let title = if def.name.is_empty() {
        def.id.as_str()
    } else {
        def.name.as_str()
    };
    let ins: Vec<&PinDecl> = def.pins.iter().filter(|p| p.dir == Dir::In).collect();
    let outs: Vec<&PinDecl> = def.pins.iter().filter(|p| p.dir != Dir::In).collect();
    let lw = ins.iter().map(|p| text_units(&p.name)).max().unwrap_or(0);
    let rw = outs.iter().map(|p| text_units(&p.name)).max().unwrap_or(0);
    let w = (lw + rw + 3).max(text_units(title) + 2).max(6);
    let rows = ins.len().max(outs.len()).max(1) as i32;
    let h = 2 + PIN_PITCH * rows;
    let mut pins = BTreeMap::new();
    for (i, p) in ins.iter().enumerate() {
        pins.insert(
            p.name.clone(),
            SymbolPin {
                side: Side::Left,
                offset: 3 + PIN_PITCH * i as i32,
            },
        );
    }
    for (i, p) in outs.iter().enumerate() {
        pins.insert(
            p.name.clone(),
            SymbolPin {
                side: Side::Right,
                offset: 3 + PIN_PITCH * i as i32,
            },
        );
    }
    SymbolLayout {
        size: gp(w, h),
        pins,
    }
}

/// Symbol of a marker for the viewed def's own pin.
pub fn pin_symbol(pin: &PinDecl) -> Symbol {
    let w = (text_units(&pin.name) + 2).max(4);
    let (pos, side) = match pin.dir {
        Dir::In => (gp(w, 1), Side::Right),
        Dir::Out | Dir::InOut => (gp(0, 1), Side::Left),
    };
    Symbol {
        kind: SymbolKind::Pin {
            dir: pin.dir,
            name: pin.name.clone(),
        },
        size: gp(w, 2),
        ports: vec![port(pin.name.clone(), pos, side, pin.dir, pin.width)],
    }
}

/// Symbols for every instance of `def`, in instance order (`None` for a def use whose def is
/// missing). `symbol_of_def` provides package layouts by def id.
pub fn instance_symbols<'a>(
    design: &'a panda_netlist::Design,
    def: &'a ComponentDef,
    layouts: impl Fn(&str) -> Option<&'a SymbolLayout>,
) -> Vec<Option<Symbol>> {
    def.instances
        .iter()
        .map(|inst| match &inst.kind {
            panda_netlist::InstKind::Atom(a) => Some(atom_symbol(a)),
            panda_netlist::InstKind::Def(id) => {
                design.def(id).map(|d| package_symbol(d, layouts(id)))
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orient_maps_box_onto_box() {
        let size = gp(4, 6);
        for rot in 0..4 {
            for flip in [false, true] {
                let o = Orient { rot, flip };
                let s = o.size(size);
                for x in 0..=4 {
                    for y in 0..=6 {
                        let p = o.apply(gp(x, y), size);
                        assert!(
                            p.x >= 0 && p.y >= 0 && p.x <= s.x && p.y <= s.y,
                            "{o:?} {p:?}"
                        );
                        let f = o.apply_f([x as f32, y as f32], size);
                        assert_eq!(f, [p.x as f32, p.y as f32]);
                    }
                }
                // The side of a boundary point follows the point.
                let q = o.apply(gp(0, 3), size);
                match o.side(Side::Left) {
                    Side::Left => assert_eq!(q.x, 0),
                    Side::Right => assert_eq!(q.x, s.x),
                    Side::Top => assert_eq!(q.y, 0),
                    Side::Bottom => assert_eq!(q.y, s.y),
                }
            }
        }
    }

    #[test]
    fn gate_ports_on_boundary_and_grid() {
        for n in 1..=9 {
            let s = atom_symbol(&Atom::Gate {
                op: GateOp::Nand,
                inputs: n,
                width: 1,
                reduce: false,
            });
            assert_eq!(s.ports.len() as u32, n + 1);
            for p in &s.ports {
                match p.side {
                    Side::Left => assert_eq!(p.pos.x, 0),
                    Side::Right => assert_eq!(p.pos.x, s.size.x),
                    _ => unreachable!(),
                }
                assert!(p.pos.y > 0 && p.pos.y < s.size.y, "{n}: {p:?}");
            }
            let ys: Vec<i32> = s.ports[..n as usize].iter().map(|p| p.pos.y).collect();
            assert!(ys.windows(2).all(|w| w[1] - w[0] == PIN_PITCH));
        }
    }
}
