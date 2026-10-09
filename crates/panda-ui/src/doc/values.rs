//! Mapping from the schematic of one view to flat nets, for live values.
//!
//! A *view* is one def instance of the flattened design, identified by its instance path
//! (`""` = top, `"cpu/alu"` = a nested package instance). A net `id` of the viewed def with
//! width `W` corresponds to the flat nets named `"<path>/<id>[<bit>]"` (`"<id>[<bit>]"` at the
//! top; 1-bit nets omit `[0]`), which [`FlatNetlist::name_index`] resolves (SPEC §11 names).
//!
//! [`ViewMap`] resolves every net and pin of the viewed def once, and lists the distinct flat
//! nets the view needs ([`ViewMap::watch`]); the sim host copies exactly those into each
//! snapshot, so publication cost is proportional to what is visible.

use std::collections::HashMap;

use panda_netlist::flat::{FlatNetlist, NetIdx};
use panda_netlist::{ComponentDef, Logic, Value};

/// Flat-net name of bit `bit` of a `width`-bit net or pin `id` of the def instance at `path`.
pub fn flat_name(path: &str, id: &str, width: u32, bit: u32) -> String {
    let mut s = String::with_capacity(path.len() + id.len() + 6);
    if !path.is_empty() {
        s.push_str(path);
        s.push('/');
    }
    s.push_str(id);
    if width != 1 {
        s.push('[');
        s.push_str(&bit.to_string());
        s.push(']');
    }
    s
}

/// Flat net index of bit `bit` of net (or pin) `id` (width `width`) in the def instance at
/// `path`.
pub fn flat_index(
    flat: &FlatNetlist,
    path: &str,
    id: &str,
    width: u32,
    bit: u32,
) -> Option<NetIdx> {
    flat.net_by_name(&flat_name(path, id, width, bit))
}

/// Live-value mapping of one view.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ViewMap {
    /// Viewed instance path.
    pub path: String,
    /// Def of that instance.
    pub def: String,
    /// Distinct flat nets of the view (what the sim host publishes for it).
    pub watch: Vec<NetIdx>,
    /// Per net id of the def: per bit, the position in [`ViewMap::watch`] (`None` if the name
    /// did not resolve).
    pub nets: HashMap<String, Vec<Option<u32>>>,
    /// Per pin name of the def, the same.
    pub pins: HashMap<String, Vec<Option<u32>>>,
}

impl ViewMap {
    /// Resolve every net and pin of `def` viewed at instance `path` of `flat`.
    pub fn build(flat: &FlatNetlist, path: &str, def: &ComponentDef) -> ViewMap {
        let mut watch = Vec::new();
        let mut pos: HashMap<NetIdx, u32> = HashMap::new();
        let mut resolve = |id: &str, width: u32| -> Vec<Option<u32>> {
            (0..width)
                .map(|b| {
                    flat_index(flat, path, id, width, b).map(|n| {
                        *pos.entry(n).or_insert_with(|| {
                            watch.push(n);
                            (watch.len() - 1) as u32
                        })
                    })
                })
                .collect()
        };
        let nets = def
            .nets
            .iter()
            .map(|n| (n.id.clone(), resolve(&n.id, n.width)))
            .collect();
        let pins = def
            .pins
            .iter()
            .map(|p| (p.name.clone(), resolve(&p.name, p.width)))
            .collect();
        ViewMap {
            path: path.to_string(),
            def: def.id.clone(),
            watch,
            nets,
            pins,
        }
    }

    /// Value of a net from `values` (aligned with [`ViewMap::watch`]); unresolved bits read `X`.
    pub fn net_value(&self, id: &str, values: &[Logic]) -> Option<Value> {
        self.nets.get(id).map(|bits| gather(bits, values))
    }

    /// Value of a pin, like [`ViewMap::net_value`].
    pub fn pin_value(&self, name: &str, values: &[Logic]) -> Option<Value> {
        self.pins.get(name).map(|bits| gather(bits, values))
    }
}

fn gather(bits: &[Option<u32>], values: &[Logic]) -> Value {
    let v: Vec<Logic> = bits
        .iter()
        .map(|b| {
            b.and_then(|i| values.get(i as usize).copied())
                .unwrap_or(Logic::X)
        })
        .collect();
    Value::from_bits(&v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples;
    use panda_kernel::{Kernel, StepUnit};
    use panda_netlist::{InstKind, flatten};

    #[test]
    fn every_net_of_every_view_resolves() {
        let design = examples::panda8();
        let flat = flatten(&design).unwrap();
        let mut checked = 0;
        for inst in &flat.instances {
            let def = design.def(&inst.def).unwrap();
            let m = ViewMap::build(&flat, &inst.path, def);
            for n in &def.nets {
                assert!(
                    m.nets[&n.id].iter().all(Option::is_some),
                    "{}: {}",
                    inst.path,
                    n.id
                );
                checked += 1;
            }
            for p in &def.pins {
                assert!(
                    m.pins[&p.name].iter().all(Option::is_some),
                    "{}: {}",
                    inst.path,
                    p.name
                );
            }
            let mut w = m.watch.clone();
            w.sort_unstable();
            w.dedup();
            assert_eq!(w.len(), m.watch.len());
        }
        assert!(checked > 1000);
    }

    #[test]
    fn mapped_values_equal_kernel_reads() {
        let design = examples::counter();
        let mut k = Kernel::default();
        k.load(design.clone()).unwrap();
        k.set_input("EN", &Value::from_bool(true)).unwrap();
        k.step(5, StepUnit::Cycle);
        let flat = k.netlist().unwrap().clone();
        for inst in &flat.instances {
            let def = design.def(&inst.def).unwrap();
            let m = ViewMap::build(&flat, &inst.path, def);
            let mut vals = vec![Logic::X; m.watch.len()];
            k.copy_values(&m.watch, &mut vals);
            for n in &def.nets {
                let name = if inst.path.is_empty() {
                    n.id.clone()
                } else {
                    format!("{}/{}", inst.path, n.id)
                };
                assert_eq!(m.net_value(&n.id, &vals), k.read(&name), "{name}");
            }
            // Pins of a package instance equal the nets connected to them outside.
            if let Some(parent) = inst.parent {
                let pdef = design.def(&flat.instances[parent as usize].def).unwrap();
                let id = inst.path.rsplit('/').next().unwrap();
                assert!(
                    pdef.instances
                        .iter()
                        .any(|i| i.id == id && matches!(i.kind, InstKind::Def(_)))
                );
                for p in &def.pins {
                    let name = format!("{}/{}", inst.path, p.name);
                    assert_eq!(m.pin_value(&p.name, &vals), k.read(&name), "{name}");
                }
            }
        }
    }
}
