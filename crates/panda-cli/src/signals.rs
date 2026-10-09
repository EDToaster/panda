//! Resolving user-supplied signal names to flat nets, and the default signal selection.
//!
//! A *signal* is a named, ordered (LSB-first) list of flat nets. Names are resolved by
//! [`resolve`] in this order:
//!
//! 1. a top-level pin name (`A`, `Q`) — the pin's bits;
//! 2. an exact flat net name or alias from [`FlatNetlist::name_index`] (`q`, `cnt/ff0/m`,
//!    `carry[2]`) — one bit;
//! 3. a bus base name (`carry`, `cnt/next`) — the nets `base[0]`, `base[1]`, … found in the name
//!    index, contiguous from bit 0;
//! 4. a bit range `base[hi:lo]` (or `base[lo:hi]`; the result is always LSB-first, i.e. bit
//!    `min(hi, lo)` first) — each `base[i]` must exist (for a 1-bit net, `base[0:0]` is `base`).

use std::collections::HashSet;

use panda_netlist::flat::{FlatNetlist, NetIdx};

/// A named group of flat nets, LSB first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signal {
    /// Display name (as given by the user, or the pin / bus base name).
    pub name: String,
    /// One flat net per bit, LSB first. Never empty.
    pub nets: Vec<NetIdx>,
}

impl Signal {
    /// Width in bits.
    pub fn width(&self) -> u32 {
        self.nets.len() as u32
    }
}

/// Split `"base[i]"` into `("base", i)`.
pub fn split_bit(name: &str) -> Option<(&str, u32)> {
    let inner = name.strip_suffix(']')?;
    let open = inner.rfind('[')?;
    let idx = &inner[open + 1..];
    if idx.is_empty() || !idx.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some((&inner[..open], idx.parse().ok()?))
}

/// Split `"base[hi:lo]"` into `("base", hi, lo)`.
fn split_range(name: &str) -> Option<(&str, u32, u32)> {
    let inner = name.strip_suffix(']')?;
    let open = inner.rfind('[')?;
    let (hi, lo) = inner[open + 1..].split_once(':')?;
    Some((&inner[..open], hi.trim().parse().ok()?, lo.trim().parse().ok()?))
}

/// True for builder-generated net ids (`n` followed by digits), which are not interesting to
/// show by default.
pub fn is_auto_name(name: &str) -> bool {
    name.len() > 1 && name.starts_with('n') && name[1..].bytes().all(|c| c.is_ascii_digit())
}

/// The nets `base[0]`, `base[1]`, … (contiguous from 0) present in the name index.
fn bus_nets(nl: &FlatNetlist, base: &str) -> Vec<NetIdx> {
    let mut nets = Vec::new();
    while let Some(n) = nl.net_by_name(&format!("{base}[{}]", nets.len())) {
        nets.push(n);
    }
    nets
}

/// One bit of a bus by base name and index (`base[i]`, or `base` itself for `i == 0` of a 1-bit
/// net).
fn bus_bit(nl: &FlatNetlist, base: &str, i: u32) -> Option<NetIdx> {
    nl.net_by_name(&format!("{base}[{i}]")).or_else(|| {
        if i == 0 && nl.net_by_name(&format!("{base}[1]")).is_none() {
            nl.net_by_name(base)
        } else {
            None
        }
    })
}

/// Resolve a signal name (see the module docs for the accepted forms).
pub fn resolve(nl: &FlatNetlist, name: &str) -> Result<Signal, String> {
    let name = name.trim();
    let sig = |nets: Vec<NetIdx>| Signal { name: name.to_string(), nets };
    if let Some(p) = nl.top_inputs.iter().chain(&nl.top_outputs).find(|p| p.name == name)
        && !p.nets.is_empty()
    {
        return Ok(sig(p.nets.clone()));
    }
    if let Some(n) = nl.net_by_name(name) {
        return Ok(sig(vec![n]));
    }
    let nets = bus_nets(nl, name);
    if !nets.is_empty() {
        return Ok(sig(nets));
    }
    if let Some((base, hi, lo)) = split_range(name) {
        let (lo, hi) = (lo.min(hi), lo.max(hi));
        let nets = (lo..=hi)
            .map(|i| bus_bit(nl, base, i).ok_or_else(|| format!("no net '{base}[{i}]' (in '{name}')")))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(sig(nets));
    }
    Err(format!("unknown signal '{name}'"))
}

/// Resolve a comma-separated list of names (empty items are skipped).
pub fn resolve_list(nl: &FlatNetlist, names: &str) -> Result<Vec<Signal>, String> {
    names.split(',').map(str::trim).filter(|s| !s.is_empty()).map(|s| resolve(nl, s)).collect()
}

/// The default selection for dumping / watching: all top-level pins (inputs, then outputs, in
/// declaration order), then every explicitly named top-level net, grouped into buses by base
/// name, in flat-net order. Skipped: auto-generated ids (`n<digits>`), nets inside instances
/// (names containing `/`), unconnected atom ports (`.`), private resolver inputs (`~`), and
/// signals whose name or exact net list was already selected.
pub fn default_selection(nl: &FlatNetlist) -> Vec<Signal> {
    let mut out = Vec::new();
    let mut names = HashSet::new();
    let mut netsets: HashSet<Vec<NetIdx>> = HashSet::new();
    let mut push = |s: Signal, out: &mut Vec<Signal>| {
        if names.contains(&s.name) || netsets.contains(&s.nets) {
            return;
        }
        names.insert(s.name.clone());
        netsets.insert(s.nets.clone());
        out.push(s);
    };
    for p in nl.top_inputs.iter().chain(&nl.top_outputs) {
        if !p.nets.is_empty() {
            push(Signal { name: p.name.clone(), nets: p.nets.clone() }, &mut out);
        }
    }
    let mut bases_seen = HashSet::new();
    for name in &nl.net_names {
        if name.contains(['/', '.', '~']) {
            continue;
        }
        let base = split_bit(name).map_or(name.as_str(), |(b, _)| b);
        if is_auto_name(base) || !bases_seen.insert(base.to_string()) {
            continue;
        }
        if let Ok(s) = resolve(nl, base) {
            push(s, &mut out);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_helpers() {
        assert_eq!(split_bit("a[3]"), Some(("a", 3)));
        assert_eq!(split_bit("x/y[12]"), Some(("x/y", 12)));
        assert_eq!(split_bit("a"), None);
        assert_eq!(split_bit("a[]"), None);
        assert_eq!(split_bit("a[1:0]"), None);
        assert_eq!(split_range("a[3:0]"), Some(("a", 3, 0)));
        assert_eq!(split_range("a[0:3]"), Some(("a", 0, 3)));
        assert_eq!(split_range("a[3]"), None);
        assert!(is_auto_name("n12"));
        assert!(!is_auto_name("n"));
        assert!(!is_auto_name("next"));
    }
}
