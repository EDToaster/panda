//! X-source tracing for the stuck-X diagnostic (SPEC §5.4).
//!
//! Starting from some X-valued nets, walk backwards through the *X-valued fan-in*: from a net to
//! the inputs of its driver (gate inputs, or resolver driver nets) whose value is `X` or `Z` (a
//! gate reads `Z` as `X`). The reached nets form a graph; its strongly connected components that
//! have no X-valued fan-in from outside themselves are the **X origins**:
//! * a component with a cycle ([`XSourceKind::Loop`]) — typically a state-holding latch loop that
//!   was never initialized (the "originating kernel");
//! * a single net without a driver ([`XSourceKind::Undriven`]), driven by a source
//!   ([`XSourceKind::Source`], e.g. an X constant), by a resolver whose drivers are known but
//!   disagree ([`XSourceKind::Contention`]), or by a gate that is X although none of its inputs
//!   is ([`XSourceKind::Gate`]: not settled yet, or forced).

use panda_netlist::Logic;
use panda_netlist::flat::{Driver, FlatNetlist, NetIdx};

/// What kind of origin an [`XSource`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XSourceKind {
    /// A feedback loop of X-valued nets (an uninitialized state-holding kernel).
    Loop,
    /// A net without any driver.
    Undriven,
    /// A net driven by a source (constant or top-level input) that is X / Z.
    Source,
    /// A resolver output that is X although no driver is X (contention).
    Contention,
    /// A gate output that is X although no input is X or Z.
    Gate,
}

/// One X origin found by [`x_sources`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct XSource {
    /// Kind of origin.
    pub kind: XSourceKind,
    /// The nets of the origin (sorted); one net unless `kind` is [`XSourceKind::Loop`].
    pub nets: Vec<NetIdx>,
}

fn is_xz(v: Logic) -> bool {
    matches!(v, Logic::X | Logic::Z)
}

/// Find the X origins of the X/Z-valued nets in `start` (see the module docs). Origins are sorted
/// by their smallest net index.
pub fn x_sources(nl: &FlatNetlist, values: &[Logic], start: &[NetIdx]) -> Vec<XSource> {
    let drivers = nl.drivers();
    // Reached nets get dense local indices; edges point from a net to its X-valued fan-in.
    let mut local: Vec<u32> = vec![u32::MAX; nl.num_nets as usize];
    let mut nets: Vec<NetIdx> = Vec::new();
    let mut edges: Vec<Vec<u32>> = Vec::new();
    let mut work: Vec<NetIdx> = Vec::new();
    let visit =
        |n: NetIdx, local: &mut Vec<u32>, nets: &mut Vec<NetIdx>, edges: &mut Vec<Vec<u32>>, work: &mut Vec<NetIdx>| {
            if local[n as usize] == u32::MAX {
                local[n as usize] = nets.len() as u32;
                nets.push(n);
                edges.push(Vec::new());
                work.push(n);
            }
            local[n as usize]
        };
    for &n in start {
        if is_xz(values[n as usize]) {
            visit(n, &mut local, &mut nets, &mut edges, &mut work);
        }
    }
    while let Some(n) = work.pop() {
        let fanin: &[NetIdx] = match drivers[n as usize] {
            Driver::Gate(g) => &nl.gates[g as usize].inputs,
            Driver::Resolver(r) => &nl.resolvers[r as usize].drivers,
            Driver::Source(_) | Driver::None => &[],
        };
        let me = local[n as usize] as usize;
        for &i in fanin {
            // A resolver drops Z drivers: only X drivers make its output X.
            let x = match drivers[n as usize] {
                Driver::Resolver(_) => values[i as usize] == Logic::X,
                _ => is_xz(values[i as usize]),
            };
            if x {
                let li = visit(i, &mut local, &mut nets, &mut edges, &mut work);
                edges[me].push(li);
            }
        }
    }
    let comp = tarjan(&edges);
    let ncomp = comp.iter().copied().max().map_or(0, |m| m as usize + 1);
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); ncomp];
    let mut has_cycle = vec![false; ncomp];
    let mut feeds_from_outside = vec![false; ncomp];
    for (v, es) in edges.iter().enumerate() {
        let c = comp[v] as usize;
        members[c].push(v as u32);
        for &w in es {
            if comp[w as usize] as usize == c {
                has_cycle[c] = true;
            } else {
                feeds_from_outside[c] = true;
            }
        }
    }
    let mut out: Vec<XSource> = Vec::new();
    for c in 0..ncomp {
        if feeds_from_outside[c] {
            continue;
        }
        let mut ns: Vec<NetIdx> = members[c].iter().map(|&v| nets[v as usize]).collect();
        ns.sort_unstable();
        let kind = if has_cycle[c] {
            XSourceKind::Loop
        } else {
            match drivers[ns[0] as usize] {
                Driver::None => XSourceKind::Undriven,
                Driver::Source(_) => XSourceKind::Source,
                Driver::Resolver(_) => XSourceKind::Contention,
                Driver::Gate(_) => XSourceKind::Gate,
            }
        };
        out.push(XSource { kind, nets: ns });
    }
    out.sort_by_key(|s| s.nets[0]);
    out
}

/// Iterative Tarjan SCC; returns the component index of every vertex.
fn tarjan(edges: &[Vec<u32>]) -> Vec<u32> {
    let n = edges.len();
    const UNSET: u32 = u32::MAX;
    let mut index = vec![UNSET; n];
    let mut low = vec![0u32; n];
    let mut on_stack = vec![false; n];
    let mut comp = vec![UNSET; n];
    let mut stack: Vec<u32> = Vec::new();
    let mut next_index = 0u32;
    let mut next_comp = 0u32;
    // Call stack of (vertex, next edge position).
    let mut call: Vec<(u32, usize)> = Vec::new();
    for root in 0..n as u32 {
        if index[root as usize] != UNSET {
            continue;
        }
        call.push((root, 0));
        index[root as usize] = next_index;
        low[root as usize] = next_index;
        next_index += 1;
        stack.push(root);
        on_stack[root as usize] = true;
        while let Some(&mut (v, ref mut pos)) = call.last_mut() {
            let vu = v as usize;
            if *pos < edges[vu].len() {
                let w = edges[vu][*pos];
                *pos += 1;
                let wu = w as usize;
                if index[wu] == UNSET {
                    index[wu] = next_index;
                    low[wu] = next_index;
                    next_index += 1;
                    stack.push(w);
                    on_stack[wu] = true;
                    call.push((w, 0));
                } else if on_stack[wu] {
                    low[vu] = low[vu].min(index[wu]);
                }
            } else {
                call.pop();
                if let Some(&(p, _)) = call.last() {
                    low[p as usize] = low[p as usize].min(low[vu]);
                }
                if low[vu] == index[vu] {
                    while let Some(w) = stack.pop() {
                        on_stack[w as usize] = false;
                        comp[w as usize] = next_comp;
                        if w == v {
                            break;
                        }
                    }
                    next_comp += 1;
                }
            }
        }
    }
    comp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tarjan_components() {
        // 0 -> 1 -> 2 -> 0, 2 -> 3, 4 self-loop
        let e = vec![vec![1], vec![2], vec![0, 3], vec![], vec![4]];
        let c = tarjan(&e);
        assert_eq!(c[0], c[1]);
        assert_eq!(c[1], c[2]);
        assert_ne!(c[2], c[3]);
        assert_ne!(c[4], c[0]);
        assert_ne!(c[4], c[3]);
    }
}
