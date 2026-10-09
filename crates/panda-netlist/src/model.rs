//! Hierarchical, serializable design model (SPEC §4, §11).
//!
//! A [`Design`] is a set of [`ComponentDef`]s plus the id of the top def. Each def has a pin
//! interface and a body of [`Instance`]s connected by [`Net`]s. Instances are either atomics
//! ([`Atom`]) or uses of other defs. The def-dependency graph must be acyclic.
//!
//! The top def's `In` pins are the user-clickable inputs; its `Out` pins are the displayed outputs.
//! `Clock`, `PowerOnReset` may only appear in the top def (SPEC §4.3).

use serde::{Deserialize, Serialize};

use crate::value::Value;

pub type DefId = String;

pub const FORMAT: &str = "panda/1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Design {
    #[serde(default = "default_format")]
    pub format: String,
    pub defs: Vec<ComponentDef>,
    pub top: DefId,
}

fn default_format() -> String {
    FORMAT.to_string()
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Dir {
    In,
    Out,
    InOut,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PinDecl {
    pub name: String,
    pub dir: Dir,
    pub width: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ComponentDef {
    pub id: DefId,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub version: u32,
    pub pins: Vec<PinDecl>,
    #[serde(default)]
    pub instances: Vec<Instance>,
    #[serde(default)]
    pub nets: Vec<Net>,
    /// Editor-only data (geometry, symbol layout). Ignored by the kernel.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub editor: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Instance {
    pub id: String,
    pub kind: InstKind,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum InstKind {
    Atom(Atom),
    Def(DefId),
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum GateOp {
    And,
    Or,
    Nand,
    Nor,
    Xor,
    Xnor,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Termination {
    #[default]
    None,
    PullUp,
    PullDown,
    Keeper,
}

/// Built-in components. Port names and widths are given by [`Atom::ports`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "atom", rename_all = "snake_case")]
pub enum Atom {
    /// N-input bitwise gate. Ports `in0..in{N-1}` (W) → `out` (W).
    /// With `reduce`, `inputs` must be 1: port `in0` (W) → `out` (1), reducing across the bits.
    Gate { op: GateOp, inputs: u32, width: u32, #[serde(default)] reduce: bool },
    /// `in` (W) → `out` (W)
    Not { width: u32 },
    /// `in` (W) → `out` (W)
    Buf { width: u32 },
    /// Tri-state buffer: `d` (W), `en` (1) → `out` (W). `en=1 → d`, `en=0 → Z`, else `X`.
    Tri { width: u32 },
    /// Pure wiring, bidirectional. Port `whole` (W) and ports `p0..pK` where part `i` is the bit
    /// range `lo..=hi` of `whole`. Flattening aliases the bits; it has no delay and no direction.
    /// Serves as both splitter and joiner.
    Splitter { width: u32, parts: Vec<(u32, u32)> },
    /// `out` = `value`. Delay 0.
    Constant { value: Value },
    /// Top-level clock source, `out` (1). At tick t:
    /// `out = 1 iff ((t + period - phase % period) % period) < high`, i.e. it rises at tick
    /// `phase` and stays high for `high` ticks of every `period`. `period ≥ 2`, `0 < high < period`.
    Clock { period: u32, high: u32, #[serde(default)] phase: u32 },
    /// Top-level reset source, `out` (1): drives 1 for ticks `0..ticks` after reset, then 0.
    PowerOnReset { ticks: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortDecl {
    pub name: String,
    pub dir: Dir,
    pub width: u32,
}

impl Atom {
    /// Port interface of the atom. Splitter ports are `InOut`.
    pub fn ports(&self) -> Vec<PortDecl> {
        let p = |name: String, dir, width| PortDecl { name, dir, width };
        match self {
            Atom::Gate { inputs, width, reduce, .. } => {
                let mut v: Vec<_> = (0..*inputs).map(|i| p(format!("in{i}"), Dir::In, *width)).collect();
                v.push(p("out".into(), Dir::Out, if *reduce { 1 } else { *width }));
                v
            }
            Atom::Not { width } | Atom::Buf { width } => {
                vec![p("in".into(), Dir::In, *width), p("out".into(), Dir::Out, *width)]
            }
            Atom::Tri { width } => vec![
                p("d".into(), Dir::In, *width),
                p("en".into(), Dir::In, 1),
                p("out".into(), Dir::Out, *width),
            ],
            Atom::Splitter { width, parts } => {
                let mut v = vec![p("whole".into(), Dir::InOut, *width)];
                for (i, (lo, hi)) in parts.iter().enumerate() {
                    v.push(p(format!("p{i}"), Dir::InOut, hi - lo + 1));
                }
                v
            }
            Atom::Constant { value } => vec![p("out".into(), Dir::Out, value.width())],
            Atom::Clock { .. } | Atom::PowerOnReset { .. } => vec![p("out".into(), Dir::Out, 1)],
        }
    }
}

/// A net connects endpoints that all have the net's width.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Net {
    pub id: String,
    pub width: u32,
    #[serde(default, skip_serializing_if = "is_default_term")]
    pub termination: Termination,
    pub ends: Vec<Endpoint>,
}

fn is_default_term(t: &Termination) -> bool {
    *t == Termination::None
}

/// Serialized as `["pin", "<pin name>"]` (the enclosing def's own pin) or
/// `["<instance id>", "<port name>"]`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Endpoint {
    Pin(String),
    Port { inst: String, port: String },
}

impl Serialize for Endpoint {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Endpoint::Pin(n) => ("pin", n).serialize(s),
            Endpoint::Port { inst, port } => (inst, port).serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for Endpoint {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Endpoint, D::Error> {
        let (a, b) = <(String, String)>::deserialize(d)?;
        Ok(if a == "pin" { Endpoint::Pin(b) } else { Endpoint::Port { inst: a, port: b } })
    }
}

impl Design {
    pub fn def(&self, id: &str) -> Option<&ComponentDef> {
        self.defs.iter().find(|d| d.id == id)
    }
    pub fn top_def(&self) -> Option<&ComponentDef> {
        self.def(&self.top)
    }
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("design serializes")
    }
    pub fn from_json(s: &str) -> Result<Design, String> {
        serde_json::from_str(s).map_err(|e| e.to_string())
    }
}
