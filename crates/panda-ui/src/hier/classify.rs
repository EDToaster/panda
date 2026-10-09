//! Classification badges (SPEC §4.4): combinational / sequential / oscillator, state bits and
//! clock-like pins of a def, computed by [`panda_kernel::Kernel::classify`] on a **worker
//! thread with its own kernel** — never on the sim thread or the UI thread, so it costs the
//! simulation nothing but a core for a few milliseconds per def.
//!
//! [`Classifier`] caches results per def id and invalidates them when the def or anything it
//! instantiates changes ([`Classifier::sync`], a structural comparison done only when the
//! document's design was replaced, i.e. never for geometry-only changes).

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use panda_kernel::{ClassKind, Kernel, KernelOptions};
use panda_netlist::{DefId, Design, InstKind};

use crate::doc::Document;

/// Summary shown as a badge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Badge {
    /// The class.
    pub kind: ClassKind,
    /// State bits.
    pub state_bits: u32,
    /// Clock-like input pins.
    pub clock_pins: Vec<String>,
    /// Edge-triggered: `(pin, rising)`.
    pub edge: Option<(String, bool)>,
}

impl Badge {
    /// Short text: `"comb"`, `"seq·8b"`, `"osc"`.
    pub fn short(&self) -> String {
        match self.kind {
            ClassKind::Combinational => "comb".into(),
            ClassKind::Sequential => format!("seq·{}b", self.state_bits),
            ClassKind::Oscillator => "osc".into(),
        }
    }

    /// Long text for tooltips and the breadcrumb bar.
    pub fn long(&self) -> String {
        let mut s = match self.kind {
            ClassKind::Combinational => "combinational".to_string(),
            ClassKind::Sequential => format!("sequential, {} state bits", self.state_bits),
            ClassKind::Oscillator => "oscillator".to_string(),
        };
        if !self.clock_pins.is_empty() {
            s.push_str(&format!(", clock-like: {}", self.clock_pins.join(", ")));
        }
        if let Some((pin, rising)) = &self.edge {
            s.push_str(&format!(
                ", {} edge of {pin}",
                if *rising { "rising" } else { "falling" }
            ));
        }
        s
    }
}

/// State of one def's classification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BadgeState {
    /// Queued or running.
    Pending,
    /// Done.
    Ready(Badge),
    /// The def could not be classified (e.g. invalid on its own).
    Failed(String),
}

struct Job {
    design: Arc<Design>,
    def: DefId,
    epoch: u64,
}

/// Classification cache plus its worker thread.
pub struct Classifier {
    results: HashMap<DefId, BadgeState>,
    queue: VecDeque<DefId>,
    queued: HashSet<DefId>,
    busy: bool,
    epoch: u64,
    design: Option<Arc<Design>>,
    /// Address of the defs buffer of the design last synced. Structural edits replace the
    /// design ([`Document::apply_ops`] clones it) and opening a document replaces it too, so
    /// an unchanged address means a geometry-only change and costs nothing (a deep comparison
    /// of Panda-8 takes ≈ 0.3 ms, too much for every frame of a drag).
    seen: usize,
    tx: Sender<Job>,
    rx: Receiver<(u64, DefId, BadgeState)>,
}

impl Default for Classifier {
    fn default() -> Self {
        Classifier::new()
    }
}

impl Classifier {
    /// Start the worker thread.
    pub fn new() -> Classifier {
        let (tx, jobs) = channel::<Job>();
        let (done, rx) = channel();
        std::thread::Builder::new()
            .name("panda-classify".into())
            .spawn(move || {
                while let Ok(job) = jobs.recv() {
                    let r = classify_def(&job.design, &job.def);
                    if done.send((job.epoch, job.def, r)).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn classify thread");
        Classifier {
            results: HashMap::new(),
            queue: VecDeque::new(),
            queued: HashSet::new(),
            busy: false,
            epoch: 0,
            design: None,
            seen: 0,
            tx,
            rx,
        }
    }

    /// Follow the document: when its design was replaced, drop the results of every def whose
    /// contents (transitively) changed. Free when only geometry changed. (Mutating
    /// `doc.design` in place instead of through [`Document::apply_ops`] is not noticed.)
    pub fn sync(&mut self, doc: &Document) {
        let seen = doc.design.defs.as_ptr() as usize;
        if self.seen == seen && self.design.is_some() {
            return;
        }
        self.seen = seen;
        let same = self
            .design
            .as_deref()
            .is_some_and(|d| d.defs == doc.design.defs && d.top == doc.design.top);
        if same {
            return;
        }
        let old = self.design.replace(Arc::new(doc.design.clone()));
        let changed: HashSet<&str> = match &old {
            None => doc.design.defs.iter().map(|d| d.id.as_str()).collect(),
            Some(o) => doc
                .design
                .defs
                .iter()
                .filter(|d| o.def(&d.id) != Some(*d))
                .map(|d| d.id.as_str())
                .chain(
                    o.defs
                        .iter()
                        .filter(|d| doc.design.def(&d.id).is_none())
                        .map(|d| d.id.as_str()),
                )
                .collect(),
        };
        if changed.is_empty() {
            return;
        }
        let stale: Vec<DefId> = self
            .results
            .keys()
            .filter(|id| {
                doc.design.def(id).is_none()
                    || closure(&doc.design, id).iter().any(|c| changed.contains(c.as_str()))
            })
            .cloned()
            .collect();
        for id in stale {
            self.results.remove(&id);
        }
        // Results still in flight belong to the old design: forget them (re-requested by `get`).
        self.results.retain(|_, v| *v != BadgeState::Pending);
        self.epoch += 1;
        self.busy = false;
        self.queue.clear();
        self.queued.clear();
    }

    /// Collect finished results and start the next job. Call once per frame.
    pub fn poll(&mut self) {
        while let Ok((epoch, def, r)) = self.rx.try_recv() {
            if epoch == self.epoch {
                self.results.insert(def, r);
                self.busy = false;
            }
        }
        if self.busy {
            return;
        }
        let Some(design) = self.design.clone() else {
            return;
        };
        while let Some(def) = self.queue.pop_front() {
            self.queued.remove(&def);
            if self.results.get(&def) != Some(&BadgeState::Pending) {
                continue;
            }
            let job = Job {
                design: design.clone(),
                def,
                epoch: self.epoch,
            };
            self.busy = self.tx.send(job).is_ok();
            break;
        }
    }

    /// The badge of `def`, requesting classification if unknown.
    pub fn get(&mut self, def: &str) -> &BadgeState {
        if !self.results.contains_key(def) {
            self.results.insert(def.to_string(), BadgeState::Pending);
            if self.queued.insert(def.to_string()) {
                self.queue.push_back(def.to_string());
            }
        }
        &self.results[def]
    }

    /// The badge of `def` if already known (does not request it).
    pub fn peek(&self, def: &str) -> Option<&BadgeState> {
        self.results.get(def)
    }

    /// Block until no job is pending (tests).
    pub fn wait_idle(&mut self, timeout: std::time::Duration) -> bool {
        let t0 = std::time::Instant::now();
        loop {
            self.poll();
            if !self.busy && self.queue.is_empty() {
                return true;
            }
            if t0.elapsed() > timeout {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }
}

/// `def` and every def it instantiates (transitively).
pub fn closure(design: &Design, def: &str) -> Vec<DefId> {
    let mut seen: HashSet<DefId> = HashSet::new();
    let mut stack = vec![def.to_string()];
    let mut out = Vec::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id.clone()) {
            continue;
        }
        if let Some(d) = design.def(&id) {
            for i in &d.instances {
                if let InstKind::Def(c) = &i.kind {
                    stack.push(c.clone());
                }
            }
        }
        out.push(id);
    }
    out
}

/// Classify `def` of `design` on a private kernel (the calling thread does the work).
pub fn classify_def(design: &Design, def: &str) -> BadgeState {
    let ids = closure(design, def);
    let sub = Design {
        format: design.format.clone(),
        defs: design
            .defs
            .iter()
            .filter(|d| ids.contains(&d.id))
            .map(|d| {
                let mut d = d.clone();
                d.editor = serde_json::Value::Null;
                d
            })
            .collect(),
        top: def.to_string(),
    };
    let mut k = Kernel::new(KernelOptions {
        background: false,
        ..KernelOptions::default()
    });
    if let Err(e) = k.load(sub) {
        return BadgeState::Failed(
            e.first()
                .map_or_else(|| "invalid".into(), ToString::to_string),
        );
    }
    match k.classify(def) {
        Ok(c) => BadgeState::Ready(Badge {
            kind: c.kind,
            state_bits: c.state_bits,
            clock_pins: c.clock_like_pins,
            edge: c
                .edge_triggered
                .map(|(p, e)| (p, e == panda_kernel::Edge::Rising)),
        }),
        Err(e) => BadgeState::Failed(
            e.first()
                .map_or_else(|| "invalid".into(), ToString::to_string),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples;
    use std::time::Duration;

    #[test]
    fn classifies_counter_defs() {
        let doc = Document::new(examples::counter());
        let mut c = Classifier::new();
        c.sync(&doc);
        for id in ["half_adder", "ms_dff", "sync_counter_4"] {
            c.get(id);
        }
        assert!(c.wait_idle(Duration::from_secs(30)));
        let BadgeState::Ready(b) = c.get("half_adder").clone() else {
            panic!()
        };
        assert_eq!(b.kind, ClassKind::Combinational);
        let BadgeState::Ready(b) = c.get("ms_dff").clone() else {
            panic!()
        };
        assert_eq!(b.kind, ClassKind::Sequential);
        assert_eq!(b.clock_pins, vec!["CLK".to_string()]);
        let BadgeState::Ready(b) = c.get("sync_counter_4").clone() else {
            panic!()
        };
        assert_eq!(b.kind, ClassKind::Sequential);
        assert!(b.long().contains("CLK"), "{}", b.long());
    }

    #[test]
    fn oscillator_and_invalidation() {
        use panda_circuits::ring_oscillator;
        use panda_netlist::DesignBuilder;
        let mut db = DesignBuilder::new();
        let r = ring_oscillator(&mut db, 3, false);
        let mut doc = Document::new(db.build(&r));
        let mut c = Classifier::new();
        c.sync(&doc);
        c.get(&r.id);
        assert!(c.wait_idle(Duration::from_secs(30)));
        assert!(
            matches!(c.peek(&r.id), Some(BadgeState::Ready(b)) if b.kind == ClassKind::Oscillator)
        );
        // A replaced document with the same revision number is noticed.
        let other = Document::new(examples::counter());
        c.sync(&other);
        assert!(c.peek(&r.id).is_none());
        c.get("ms_dff");
        assert!(c.wait_idle(Duration::from_secs(30)));
        assert!(matches!(c.peek("ms_dff"), Some(BadgeState::Ready(_))));
        c.sync(&doc);
        c.get(&r.id);
        assert!(c.wait_idle(Duration::from_secs(30)));
        // A geometry-only revision keeps the result; a structural change drops it.
        doc.touch();
        c.sync(&doc);
        assert!(matches!(c.peek(&r.id), Some(BadgeState::Ready(_))));
        let mut d2 = doc.design.clone();
        d2.defs[0].name = "renamed".into();
        doc.design = d2;
        doc.touch();
        c.sync(&doc);
        assert!(c.peek(&r.id).is_none());
    }
}
