//! Path conditions: conjunctions of literals `net = value` over the *final* (end-of-event)
//! values of nets.
//!
//! A change of a net during an event carries a [`Cond`] that is a **necessary** condition for the
//! change to happen at all. Dropping literals keeps a condition sound (it only gets weaker), so
//! conditions are truncated to [`MAX_LITS`] literals and merged by intersection. Literals are
//! always stated on *root* nets (see `Ctx::roots`: the net reached by following inverters and
//! buffers backwards), so `sel = 1` and `NOT sel = 0` are recognized as the same literal.

use panda_netlist::flat::NetIdx;

/// Maximum number of literals kept in one condition.
pub(crate) const MAX_LITS: usize = 16;

/// A conjunction of literals `(root net, value)`, sorted by net, one literal per net.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct Cond(Vec<(NetIdx, bool)>);

impl Cond {
    /// The always-true condition.
    pub(crate) fn top() -> Cond {
        Cond(Vec::new())
    }

    /// Build from arbitrary literals; `None` if two literals contradict each other.
    pub(crate) fn from_lits(mut lits: Vec<(NetIdx, bool)>) -> Option<Cond> {
        lits.sort_unstable();
        lits.dedup();
        if lits.windows(2).any(|w| w[0].0 == w[1].0) {
            return None;
        }
        lits.truncate(MAX_LITS);
        Some(Cond(lits))
    }

    /// The literals.
    pub(crate) fn lits(&self) -> &[(NetIdx, bool)] {
        &self.0
    }

    /// Value required for `net`, if any.
    #[cfg(test)]
    pub(crate) fn get(&self, net: NetIdx) -> Option<bool> {
        self.0
            .binary_search_by_key(&net, |l| l.0)
            .ok()
            .map(|i| self.0[i].1)
    }

    /// Literals present in both (the weaker condition implied by either).
    pub(crate) fn intersect(&self, other: &Cond) -> Cond {
        let (mut i, mut j) = (0, 0);
        let mut out = Vec::new();
        while i < self.0.len() && j < other.0.len() {
            let (a, b) = (self.0[i], other.0[j]);
            match a.0.cmp(&b.0) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    if a.1 == b.1 {
                        out.push(a);
                    }
                    i += 1;
                    j += 1;
                }
            }
        }
        Cond(out)
    }

    /// Conjunction of both; `None` if they contradict.
    pub(crate) fn union(&self, other: &Cond) -> Option<Cond> {
        if other.0.is_empty() {
            return Some(self.clone());
        }
        if self.0.is_empty() {
            return Some(other.clone());
        }
        let (mut i, mut j) = (0, 0);
        let mut out = Vec::with_capacity(self.0.len() + other.0.len());
        while i < self.0.len() || j < other.0.len() {
            if j == other.0.len() || (i < self.0.len() && self.0[i].0 < other.0[j].0) {
                out.push(self.0[i]);
                i += 1;
            } else if i == self.0.len() || other.0[j].0 < self.0[i].0 {
                out.push(other.0[j]);
                j += 1;
            } else {
                if self.0[i].1 != other.0[j].1 {
                    return None;
                }
                out.push(self.0[i]);
                i += 1;
                j += 1;
            }
        }
        out.truncate(MAX_LITS);
        Some(Cond(out))
    }

    /// True if both can hold at the same time.
    pub(crate) fn compatible(&self, other: &Cond) -> bool {
        let (mut i, mut j) = (0, 0);
        while i < self.0.len() && j < other.0.len() {
            match self.0[i].0.cmp(&other.0[j].0) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    if self.0[i].1 != other.0[j].1 {
                        return false;
                    }
                    i += 1;
                    j += 1;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_and_conflicts() {
        assert!(Cond::from_lits(vec![(3, true), (3, false)]).is_none());
        let c = Cond::from_lits(vec![(5, true), (2, false), (5, true)]).unwrap();
        assert_eq!(c.lits(), &[(2, false), (5, true)]);
        assert_eq!(c.get(5), Some(true));
        assert_eq!(c.get(4), None);
    }

    #[test]
    fn algebra() {
        let a = Cond::from_lits(vec![(1, true), (2, false), (7, true)]).unwrap();
        let b = Cond::from_lits(vec![(1, true), (2, true), (9, false)]).unwrap();
        assert_eq!(a.intersect(&b).lits(), &[(1, true)]);
        assert!(a.union(&b).is_none());
        assert!(!a.compatible(&b));
        let c = Cond::from_lits(vec![(1, true), (8, true)]).unwrap();
        assert!(a.compatible(&c));
        assert_eq!(
            a.union(&c).unwrap().lits(),
            &[(1, true), (2, false), (7, true), (8, true)]
        );
        assert_eq!(a.union(&Cond::top()).unwrap(), a);
        assert!(Cond::top().intersect(&a).lits().is_empty());
    }

    #[test]
    fn truncation_keeps_soundness() {
        let lits: Vec<_> = (0..40).map(|n| (n, n % 2 == 0)).collect();
        let c = Cond::from_lits(lits).unwrap();
        assert_eq!(c.lits().len(), MAX_LITS);
    }
}
