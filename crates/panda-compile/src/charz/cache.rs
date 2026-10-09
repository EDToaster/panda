//! [`TableCache`]: kernel tables keyed by structural key, internally synchronized, with
//! parallel characterization of distinct kernels and a serde snapshot format for persistence.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use panda_netlist::Logic;
use panda_netlist::flat::FlatNetlist;
use serde::{Deserialize, Serialize};

use super::{CharzOptions, KernelTable, Transition, characterize_subnet_with, default_threads};
use crate::kernels::{Kernel, KernelKind};
use crate::subnet::{SubnetMap, kernel_subnet, structural_key};

/// Number of independently locked shards.
const SHARDS: usize = 16;

/// A cache slot: initialized once with the table (`None` = the kernel exceeded the size guards).
type Slot = Arc<OnceLock<Option<Arc<KernelTable>>>>;

/// Cache of kernel tables keyed by the structural key of the kernel sub-netlist
/// ([`structural_key`]): identical kernels (e.g. 200 D latches) are characterized once. A cache
/// must only be used with one [`CharzOptions`] value.
///
/// The cache is internally synchronized (`&self` everywhere; share it with `Arc<TableCache>`):
///
/// * Lookups lock one of 16 shards briefly and never wait for a characterization
///   ([`TableCache::try_get`], [`TableCache::try_tables`]).
/// * [`TableCache::get_or_characterize`] / [`TableCache::characterize_all`] characterize missing
///   kernels; concurrent requests for the same key wait for the one characterization in
///   progress instead of repeating it.
/// * Results never depend on thread counts or on concurrent use: a table is a pure function of
///   the sub-netlist and the options.
#[derive(Default)]
pub struct TableCache {
    shards: [Mutex<HashMap<Vec<u32>, Slot>>; SHARDS],
    hits: AtomicU64,
    misses: AtomicU64,
}

/// A kernel prepared for a cache lookup: its sub-netlist and structural key.
struct Prepared {
    sub: FlatNetlist,
    map: SubnetMap,
    key: Vec<u32>,
}

fn prepare(flat: &FlatNetlist, kernel: &Kernel) -> Option<Prepared> {
    if kernel.kind != KernelKind::Table {
        return None;
    }
    let (sub, map) = kernel_subnet(flat, kernel);
    let key = structural_key(&sub);
    Some(Prepared { sub, map, key })
}

fn shard_of(key: &[u32]) -> usize {
    let mut h = super::fast::FxHasher::default();
    key.hash(&mut h);
    (h.finish() >> 32) as usize % SHARDS
}

/// Characterization-cost estimate used to schedule the longest jobs first.
fn cost(p: &Prepared) -> u64 {
    let k = p.map.num_inputs() as u32;
    (1u64 << k.min(40)) * (p.map.num_state() as u64 + 1)
}

/// Jobs below this [`cost`] run on the calling thread (a thread spawn costs more than they do).
const BIG_JOB: u64 = 4096;

impl TableCache {
    /// An empty cache.
    pub fn new() -> TableCache {
        TableCache::default()
    }

    fn lock(&self, key: &[u32]) -> MutexGuard<'_, HashMap<Vec<u32>, Slot>> {
        self.shards[shard_of(key)]
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The slot of `key`, created empty if absent.
    fn slot(&self, key: &[u32]) -> Slot {
        let mut map = self.lock(key);
        if let Some(s) = map.get(key) {
            return s.clone();
        }
        let s: Slot = Arc::default();
        map.insert(key.to_vec(), s.clone());
        s
    }

    /// Initialize `slot` by characterizing `p` (or wait for a concurrent initialization).
    /// Returns the table and whether this call characterized it.
    fn fill(
        slot: &Slot,
        p: Prepared,
        opts: &CharzOptions,
        threads: usize,
    ) -> (Option<Arc<KernelTable>>, bool) {
        let mut ran = false;
        let t = slot
            .get_or_init(|| {
                ran = true;
                characterize_subnet_with(Arc::new(p.sub), p.map, opts, threads).map(Arc::new)
            })
            .clone();
        (t, ran)
    }

    /// The table of `kernel` (`None` unless it is a Table kernel within the size guards of
    /// [`super::characterize`]), characterizing it on a miss (with up to
    /// [`default_threads`] threads).
    pub fn get_or_characterize(
        &self,
        flat: &FlatNetlist,
        kernel: &Kernel,
        opts: &CharzOptions,
    ) -> Option<Arc<KernelTable>> {
        let p = prepare(flat, kernel)?;
        let slot = self.slot(&p.key);
        if let Some(t) = slot.get() {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return t.clone();
        }
        let (t, ran) = Self::fill(&slot, p, opts, default_threads());
        let counter = if ran { &self.misses } else { &self.hits };
        counter.fetch_add(1, Ordering::Relaxed);
        t
    }

    /// Non-blocking lookup: `Some(table)` if the table of `kernel` is in the cache (`table` is
    /// `None` for kernels that are not Table kernels or exceeded the size guards), `None` if it
    /// has not been characterized yet (or is being characterized right now). Does not count as
    /// a hit or miss.
    pub fn try_get(&self, flat: &FlatNetlist, kernel: &Kernel) -> Option<Option<Arc<KernelTable>>> {
        let Some(p) = prepare(flat, kernel) else {
            return Some(None);
        };
        self.try_get_key(&p.key)
    }

    /// Non-blocking lookup by structural key ([`structural_key`] of the kernel sub-netlist).
    pub fn try_get_key(&self, key: &[u32]) -> Option<Option<Arc<KernelTable>>> {
        let slot = self.lock(key).get(key).cloned()?;
        slot.get().cloned()
    }

    /// Non-blocking: the tables of every kernel of `kernels` (in order, as
    /// [`TableCache::characterize_all`] would return them) if all are cached; `None` if any is
    /// missing or still being characterized.
    pub fn try_tables(
        &self,
        flat: &FlatNetlist,
        kernels: &[Kernel],
    ) -> Option<Vec<Option<Arc<KernelTable>>>> {
        kernels.iter().map(|k| self.try_get(flat, k)).collect()
    }

    /// The tables of `kernels` (`tables[i]` for `kernels[i]`, `None` for non-Table kernels and
    /// for kernels beyond the size guards). Distinct missing kernels are characterized in
    /// parallel on up to `threads` threads (`0` = 1), longest first; large kernels also use
    /// threads internally. Identical kernels are characterized once. The result does not depend
    /// on `threads`; with a single caller, the hit/miss counts do not either (a miss per
    /// distinct newly characterized key, a hit for every other Table kernel).
    pub fn characterize_all(
        &self,
        flat: &FlatNetlist,
        kernels: &[Kernel],
        opts: &CharzOptions,
        threads: usize,
    ) -> Vec<Option<Arc<KernelTable>>> {
        let threads = threads.max(1);
        // Distinct keys in first-occurrence order; per kernel its distinct index.
        let mut index: HashMap<Vec<u32>, usize> = HashMap::new();
        let mut jobs: Vec<(Slot, Option<Prepared>)> = Vec::new();
        let mut of_kernel: Vec<Option<usize>> = Vec::with_capacity(kernels.len());
        for k in kernels {
            let Some(p) = prepare(flat, k) else {
                of_kernel.push(None);
                continue;
            };
            let d = match index.get(&p.key) {
                Some(&d) => d,
                None => {
                    let d = jobs.len();
                    index.insert(p.key.clone(), d);
                    let slot = self.slot(&p.key);
                    let pending = slot.get().is_none().then_some(p);
                    jobs.push((slot, pending));
                    d
                }
            };
            of_kernel.push(Some(d));
        }
        // Characterize the missing distinct kernels, longest first.
        let mut todo: Vec<(usize, Prepared)> = jobs
            .iter_mut()
            .enumerate()
            .filter_map(|(d, (_, p))| p.take().map(|p| (d, p)))
            .collect();
        todo.sort_by_key(|(d, p)| (std::cmp::Reverse(cost(p)), *d));
        let mut ran = vec![false; jobs.len()];
        let slots: Vec<Slot> = jobs.iter().map(|(s, _)| s.clone()).collect();
        let split_at = todo.partition_point(|(_, p)| cost(p) >= BIG_JOB);
        let small: Vec<(usize, Prepared)> = todo.split_off(split_at);
        let big = todo;
        // Big jobs: one worker per job up to `threads`; their inner parallelism shares the
        // threads. Small jobs run on the calling thread meanwhile.
        let outer = threads.min(big.len());
        let inner = (threads / big.len().max(1)).max(1);
        let work: Vec<Mutex<Option<(usize, Prepared)>>> =
            big.into_iter().map(|j| Mutex::new(Some(j))).collect();
        let next = AtomicUsize::new(0);
        let worker = || {
            let mut done = Vec::new();
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(job) = work.get(i) else {
                    break done;
                };
                let Some((d, p)) = job
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take()
                else {
                    continue;
                };
                let (_, r) = Self::fill(&slots[d], p, opts, inner);
                done.push((d, r));
            }
        };
        let run_small = || -> Vec<(usize, bool)> {
            small
                .into_iter()
                .map(|(d, p)| (d, Self::fill(&slots[d], p, opts, 1).1))
                .collect()
        };
        let results: Vec<(usize, bool)> = if outer <= 1 {
            let mut r = worker();
            r.extend(run_small());
            r
        } else {
            std::thread::scope(|scope| {
                let handles: Vec<_> = (0..outer).map(|_| scope.spawn(worker)).collect();
                let mut r = run_small();
                for h in handles {
                    r.extend(h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)));
                }
                r
            })
        };
        for (d, r) in results {
            ran[d] = r;
        }
        // Assemble per kernel (waiting for keys another caller is characterizing).
        let mut first_use = vec![true; jobs.len()];
        of_kernel
            .into_iter()
            .map(|d| {
                let d = d?;
                let miss = ran[d] && std::mem::replace(&mut first_use[d], false);
                let counter = if miss { &self.misses } else { &self.hits };
                counter.fetch_add(1, Ordering::Relaxed);
                let slot = &jobs[d].0;
                match slot.get() {
                    Some(t) => t.clone(),
                    // Initialized by a concurrent caller: wait for it.
                    None => slot.wait().clone(),
                }
            })
            .collect()
    }

    /// Lookups answered from the cache.
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Lookups that characterized a kernel.
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Number of distinct tables (characterized kernels, including those beyond the size
    /// guards).
    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|s| {
                s.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .values()
                    .filter(|slot| slot.get().is_some())
                    .count()
            })
            .sum()
    }

    /// True if the cache holds no table.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert a table under a structural key (e.g. from a snapshot). An existing entry is kept.
    pub fn insert_key(&self, key: Vec<u32>, table: Option<Arc<KernelTable>>) {
        let slot = self.slot(&key);
        let _ = slot.set(table);
    }

    /// Every cached entry, sorted by key.
    fn entries(&self) -> Vec<(Vec<u32>, Option<Arc<KernelTable>>)> {
        let mut out: Vec<(Vec<u32>, Option<Arc<KernelTable>>)> = self
            .shards
            .iter()
            .flat_map(|s| {
                s.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .iter()
                    .filter_map(|(k, slot)| slot.get().map(|t| (k.clone(), t.clone())))
                    .collect::<Vec<_>>()
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// A serializable snapshot of every cached table, tagged with the options it is valid for
    /// (deterministic: entries sorted by key).
    pub fn snapshot(&self, opts: &CharzOptions) -> CacheSnapshot {
        CacheSnapshot {
            format: SNAPSHOT_FORMAT.to_string(),
            version: SNAPSHOT_VERSION,
            options: OptionsTag::of(opts),
            entries: self
                .entries()
                .into_iter()
                .map(|(key, t)| SnapshotEntry {
                    key,
                    table: t.as_deref().map(TableData::from_table),
                })
                .collect(),
        }
    }

    /// Add the tables of `snapshot` (entries already present are kept). Fails, adding nothing,
    /// if the snapshot was made with other options or another format version, or is malformed.
    /// Returns the number of entries in the snapshot.
    pub fn restore(&self, snapshot: &CacheSnapshot, opts: &CharzOptions) -> Result<usize, String> {
        if snapshot.format != SNAPSHOT_FORMAT || snapshot.version != SNAPSHOT_VERSION {
            return Err(format!(
                "unsupported table cache format {:?} version {} (expected {SNAPSHOT_FORMAT:?} \
                 version {SNAPSHOT_VERSION})",
                snapshot.format, snapshot.version
            ));
        }
        if snapshot.options != OptionsTag::of(opts) {
            return Err("table cache was built with different characterization options".into());
        }
        let tables = snapshot
            .entries
            .iter()
            .map(|e| {
                let t = e
                    .table
                    .as_ref()
                    .map(|d| d.to_table().map(Arc::new))
                    .transpose()?;
                Ok((e.key.clone(), t))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let n = tables.len();
        for (key, t) in tables {
            self.insert_key(key, t);
        }
        Ok(n)
    }

    /// Serialize [`TableCache::snapshot`] as JSON.
    pub fn to_json(&self, opts: &CharzOptions) -> String {
        serde_json::to_string(&self.snapshot(opts)).expect("snapshot serializes")
    }

    /// A cache holding the tables of a JSON snapshot ([`TableCache::to_json`]); see
    /// [`TableCache::restore`] for the checks.
    pub fn from_json(json: &str, opts: &CharzOptions) -> Result<TableCache, String> {
        let snap: CacheSnapshot = serde_json::from_str(json).map_err(|e| e.to_string())?;
        let cache = TableCache::new();
        cache.restore(&snap, opts)?;
        Ok(cache)
    }

    /// Write the JSON snapshot to a file.
    pub fn save(&self, path: &std::path::Path, opts: &CharzOptions) -> std::io::Result<()> {
        std::fs::write(path, self.to_json(opts))
    }

    /// Read a JSON snapshot file into a new cache.
    pub fn load(path: &std::path::Path, opts: &CharzOptions) -> Result<TableCache, String> {
        let json = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        TableCache::from_json(&json, opts)
    }
}

// ---------------------------------------------------------------------------------------------
// Snapshot format
// ---------------------------------------------------------------------------------------------

/// [`CacheSnapshot::format`] value.
pub const SNAPSHOT_FORMAT: &str = "panda-kernel-tables";
/// [`CacheSnapshot::version`]: bump whenever characterization results or this format change.
pub const SNAPSHOT_VERSION: u32 = 1;

/// Serializable contents of a [`TableCache`] ([`TableCache::snapshot`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheSnapshot {
    /// Always [`SNAPSHOT_FORMAT`].
    pub format: String,
    /// [`SNAPSHOT_VERSION`] of the writer.
    pub version: u32,
    /// The characterization options the tables are valid for.
    pub options: OptionsTag,
    /// Entries sorted by key.
    pub entries: Vec<SnapshotEntry>,
}

/// The [`CharzOptions`] fields that influence tables.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptionsTag {
    /// `timing.p2_limit`.
    pub p2_limit: u32,
    /// `settle_bound`.
    pub settle_bound: Option<u64>,
    /// `exhaustive_order_max`.
    pub exhaustive_order_max: usize,
    /// `order_samples`.
    pub order_samples: usize,
    /// `max_order_work`.
    pub max_order_work: u64,
    /// `max_table_entries`.
    pub max_table_entries: u64,
}

impl OptionsTag {
    /// The tag of `opts`.
    pub fn of(opts: &CharzOptions) -> OptionsTag {
        OptionsTag {
            p2_limit: opts.timing.p2_limit,
            settle_bound: opts.settle_bound,
            exhaustive_order_max: opts.exhaustive_order_max,
            order_samples: opts.order_samples,
            max_order_work: opts.max_order_work,
            max_table_entries: opts.max_table_entries,
        }
    }
}

/// One cached kernel: structural key and table (`None` = beyond the size guards).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotEntry {
    /// [`structural_key`] of the kernel sub-netlist.
    pub key: Vec<u32>,
    /// The table.
    pub table: Option<TableData>,
}

/// Compact serializable form of a [`KernelTable`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TableData {
    /// [`KernelTable::num_inputs`].
    pub num_inputs: usize,
    /// [`KernelTable::states`], one string of `0`/`1`/`X`/`Z` per state.
    pub states: Vec<String>,
    /// Per entry: next state id, or `-1` for a non-binary result.
    pub next: Vec<i64>,
    /// Per entry: latency.
    pub latency: Vec<u32>,
    /// Entries flagged [`Transition::order_sensitive`] (ascending indices).
    pub order_sensitive: Vec<u32>,
    /// [`KernelTable::states_by_input`].
    pub states_by_input: Vec<Vec<u32>>,
    /// [`KernelTable::home_input`].
    pub home_input: Vec<u32>,
    /// [`KernelTable::order_check_partial`].
    pub order_check_partial: bool,
    /// [`KernelTable::sensitive_from`], sorted.
    pub sensitive_from: Vec<(u32, u32, u32)>,
    /// [`KernelTable::order_check_stop`].
    pub order_check_stop: Option<(u32, u32, u32)>,
}

impl TableData {
    /// The serializable form of `t`.
    pub fn from_table(t: &KernelTable) -> TableData {
        let mut sensitive_from: Vec<(u32, u32, u32)> = t.sensitive_from.iter().copied().collect();
        sensitive_from.sort_unstable();
        TableData {
            num_inputs: t.num_inputs,
            states: t
                .states
                .iter()
                .map(|s| s.iter().map(|v| v.to_char()).collect())
                .collect(),
            next: t
                .transitions
                .iter()
                .map(|x| x.next.map_or(-1, i64::from))
                .collect(),
            latency: t.transitions.iter().map(|x| x.latency).collect(),
            order_sensitive: t
                .transitions
                .iter()
                .enumerate()
                .filter(|(_, x)| x.order_sensitive)
                .map(|(i, _)| i as u32)
                .collect(),
            states_by_input: t.states_by_input.clone(),
            home_input: t.home_input.clone(),
            order_check_partial: t.order_check_partial,
            sensitive_from,
            order_check_stop: t.order_check_stop,
        }
    }

    /// Rebuild the table, validating the structure (no panics on malformed input).
    pub fn to_table(&self) -> Result<KernelTable, String> {
        let k = self.num_inputs;
        if k > super::MAX_TABLE_INPUTS {
            return Err(format!("table with {k} inputs"));
        }
        let ns = self.states.len();
        let entries = ns
            .checked_shl(k as u32)
            .filter(|&e| e >> k == ns)
            .ok_or("table too large")?;
        if self.next.len() != entries || self.latency.len() != entries {
            return Err("transition arrays do not match states × 2^inputs".into());
        }
        if self.states_by_input.len() != 1 << k || self.home_input.len() != ns {
            return Err("per-input / per-state arrays have the wrong length".into());
        }
        let width = self.states.first().map_or(0, |s| s.chars().count());
        let mut states = Vec::with_capacity(ns);
        for s in &self.states {
            let st: Vec<Logic> = s
                .chars()
                .map(|c| Logic::from_char(c).ok_or_else(|| format!("bad state character {c:?}")))
                .collect::<Result<_, _>>()?;
            if st.len() != width || st.contains(&Logic::X) {
                return Err("malformed state".into());
            }
            states.push(st);
        }
        let mut state_index = HashMap::with_capacity(ns);
        for (i, s) in states.iter().enumerate() {
            if state_index.insert(s.clone(), i as u32).is_some() {
                return Err("duplicate state".into());
            }
        }
        let id = |x: u32| -> Result<u32, String> {
            if (x as usize) < ns {
                Ok(x)
            } else {
                Err(format!("state id {x} out of range"))
            }
        };
        let mut transitions = Vec::with_capacity(entries);
        for (&n, &latency) in self.next.iter().zip(&self.latency) {
            let next = match n {
                -1 => None,
                n if n >= 0 && n <= i64::from(u32::MAX) => Some(id(n as u32)?),
                n => return Err(format!("bad next state {n}")),
            };
            transitions.push(Transition {
                next,
                latency,
                order_sensitive: false,
            });
        }
        for &i in &self.order_sensitive {
            transitions
                .get_mut(i as usize)
                .ok_or("order-sensitive index out of range")?
                .order_sensitive = true;
        }
        for list in &self.states_by_input {
            for &s in list {
                id(s)?;
            }
        }
        let nv = 1u64 << k;
        let check_triple = |(s, u, v): (u32, u32, u32)| -> Result<(), String> {
            id(s)?;
            if u64::from(u) >= nv || u64::from(v) >= nv {
                return Err("input vector out of range".into());
            }
            Ok(())
        };
        for &x in &self.sensitive_from {
            check_triple(x)?;
        }
        if let Some(x) = self.order_check_stop {
            check_triple(x)?;
        }
        if self.home_input.iter().any(|&u| u64::from(u) >= nv) {
            return Err("home input out of range".into());
        }
        Ok(KernelTable {
            num_inputs: k,
            states,
            state_index,
            transitions,
            states_by_input: self.states_by_input.clone(),
            home_input: self.home_input.clone(),
            order_check_partial: self.order_check_partial,
            sensitive_from: self.sensitive_from.iter().copied().collect::<HashSet<_>>(),
            order_check_stop: self.order_check_stop,
        })
    }
}
