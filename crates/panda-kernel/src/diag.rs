//! Unified, deduplicated diagnostics.

use std::collections::HashMap;

use panda_netlist::{DiagKind, Diagnostic, Severity};

/// Where a diagnostic came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DiagSource {
    /// Validation / flattening of the design (load or edit).
    Design,
    /// The Timing-mode engine.
    Timing,
    /// The Fast-mode engine.
    Fast,
    /// Kernel compilation (background compiler).
    Compile,
    /// Static timing analysis ([`Kernel::timing_report`](crate::Kernel::timing_report)).
    Sta,
    /// The facade itself (mode switches, state transfer).
    Kernel,
}

/// A deduplicated diagnostic with occurrence information.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiagEntry {
    /// The first occurrence (its message is kept).
    pub diagnostic: Diagnostic,
    /// Producer.
    pub source: DiagSource,
    /// Number of occurrences merged into this entry.
    pub count: u64,
    /// Facade tick of the first occurrence.
    pub first_tick: u64,
    /// Facade tick of the latest occurrence.
    pub last_tick: u64,
}

/// Deduplication key: same kind, severity, location and producer. Messages differ only in
/// details such as the tick, so they are not part of the key — except for diagnostics without a
/// location, where the message is the only thing telling them apart.
type Key = (
    DiagKind,
    Severity,
    Option<String>,
    Option<String>,
    DiagSource,
);

#[derive(Default)]
pub(crate) struct DiagStore {
    entries: Vec<DiagEntry>,
    flat: Vec<Diagnostic>,
    index: HashMap<Key, usize>,
}

impl DiagStore {
    pub(crate) fn push(&mut self, source: DiagSource, tick: u64, d: Diagnostic) {
        let msg_key = d.location.is_none().then(|| d.message.clone());
        let key = (d.kind, d.severity, d.location.clone(), msg_key, source);
        if let Some(&i) = self.index.get(&key) {
            let e = &mut self.entries[i];
            e.count += 1;
            e.last_tick = tick;
            return;
        }
        self.index.insert(key, self.entries.len());
        self.flat.push(d.clone());
        self.entries.push(DiagEntry {
            diagnostic: d,
            source,
            count: 1,
            first_tick: tick,
            last_tick: tick,
        });
    }

    pub(crate) fn extend(
        &mut self,
        source: DiagSource,
        tick: u64,
        ds: impl IntoIterator<Item = Diagnostic>,
    ) {
        for d in ds {
            self.push(source, tick, d);
        }
    }

    pub(crate) fn diagnostics(&self) -> &[Diagnostic] {
        &self.flat
    }

    pub(crate) fn entries(&self) -> &[DiagEntry] {
        &self.entries
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.flat.clear();
        self.index.clear();
    }

    /// Drop every entry produced by `source` (e.g. stale STA results).
    pub(crate) fn clear_source(&mut self, source: DiagSource) {
        let old = std::mem::take(&mut self.entries);
        self.clear();
        for e in old.into_iter().filter(|e| e.source != source) {
            let msg_key = e
                .diagnostic
                .location
                .is_none()
                .then(|| e.diagnostic.message.clone());
            let key = (
                e.diagnostic.kind,
                e.diagnostic.severity,
                e.diagnostic.location.clone(),
                msg_key,
                e.source,
            );
            self.index.insert(key, self.entries.len());
            self.flat.push(e.diagnostic.clone());
            self.entries.push(e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_by_location() {
        let mut s = DiagStore::default();
        s.push(
            DiagSource::Timing,
            1,
            Diagnostic::warning(DiagKind::Contention, "at tick 1").at("n"),
        );
        s.push(
            DiagSource::Timing,
            5,
            Diagnostic::warning(DiagKind::Contention, "at tick 5").at("n"),
        );
        s.push(
            DiagSource::Fast,
            6,
            Diagnostic::warning(DiagKind::Contention, "at tick 6").at("n"),
        );
        s.push(
            DiagSource::Timing,
            7,
            Diagnostic::warning(DiagKind::Contention, "x").at("m"),
        );
        s.push(
            DiagSource::Design,
            0,
            Diagnostic::error(DiagKind::Other, "a"),
        );
        s.push(
            DiagSource::Design,
            0,
            Diagnostic::error(DiagKind::Other, "b"),
        );
        assert_eq!(s.diagnostics().len(), 5);
        assert_eq!(s.entries()[0].count, 2);
        assert_eq!(s.entries()[0].last_tick, 5);
        assert_eq!(s.entries()[0].diagnostic.message, "at tick 1");
        s.clear_source(DiagSource::Timing);
        assert_eq!(s.diagnostics().len(), 3);
        s.push(
            DiagSource::Fast,
            9,
            Diagnostic::warning(DiagKind::Contention, "again").at("n"),
        );
        assert_eq!(s.entries()[0].count, 2);
    }
}
