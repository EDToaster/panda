//! panda-compile — analysis and lowering for Fast mode (SPEC §8).
//!
//! * [`graph`] — the gate/resolver graph of a flat netlist and generic algorithms (SCC, topological
//!   order, bounded cycle enumeration, feedback vertex sets).
//! * [`kernels`] — kernel extraction (SPEC §8.3).
//! * [`subnet`] — standalone sub-netlist of one kernel.
//! * [`charz`] — kernel characterization into Latch Tables and on-the-fly transitions (SPEC §8.4).
//! * [`fastref`] — the FastRef engine, a direct interpreter of the kernel-delta algorithm (SPEC §6.2).

pub mod charz;
pub mod fastref;
pub mod graph;
pub mod kernels;
pub mod subnet;

use std::sync::Arc;

use panda_netlist::FlatNetlist;

pub use charz::{
    CharzOptions, KernelSim, KernelTable, Settled, TableCache, Transition, characterize,
};
pub use fastref::{FastOptions, FastRef, FastRefStats};
pub use graph::Node;
pub use kernels::{ExtractOptions, Kernel, KernelKind, KernelSet, extract};
pub use subnet::{SubnetMap, kernel_subnet};

/// Options of [`compile_kernels`].
#[derive(Clone, Debug, Default)]
pub struct KernelOptions {
    /// Kernel extraction tunables.
    pub extract: ExtractOptions,
    /// Characterization tunables.
    pub charz: CharzOptions,
}

/// Extract the kernels of `flat` and characterize every Table kernel (identical kernels share one
/// table). `tables[i]` belongs to `kernels.kernels[i]` and is `None` for Sim and Keeper kernels
/// (and for a Table kernel exceeding the characterization size guards, see [`characterize`]).
pub fn compile_kernels(
    flat: &FlatNetlist,
    opts: &KernelOptions,
) -> (KernelSet, Vec<Option<Arc<KernelTable>>>) {
    let mut cache = TableCache::new();
    compile_kernels_with_cache(flat, opts, &mut cache)
}

/// [`compile_kernels`] with a caller-provided cache (reusable across compilations with the same
/// [`CharzOptions`]).
pub fn compile_kernels_with_cache(
    flat: &FlatNetlist,
    opts: &KernelOptions,
    cache: &mut TableCache,
) -> (KernelSet, Vec<Option<Arc<KernelTable>>>) {
    let ks = extract(flat, &opts.extract);
    let tables = ks
        .kernels
        .iter()
        .map(|k| cache.get_or_characterize(flat, k, &opts.charz))
        .collect();
    (ks, tables)
}
