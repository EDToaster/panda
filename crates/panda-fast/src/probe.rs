//! Cycle-counter probes for profiling the event loop without an external profiler.
//!
//! With the `prof` feature (x86_64), [`start`]/[`stop`] accumulate time-stamp-counter cycles into
//! [`COUNTERS`] (slots: [`SWEEP`], [`DECIDE`], [`GENERAL`], [`WRITE`], [`LUT`], [`FLUSH`],
//! [`TIMED`]) and [`count_n`] adds to the count slots ([`NOTIFIES`]); without the feature they
//! compile to nothing.
//!
//! Instead of time, the probes can count any hardware event: map a perf event into the process
//! (so user-space `rdpmc` is allowed for it) and store `1 +` its counter index in [`PMC`] — e.g. a
//! retired-instructions event gives an exact instruction profile (see the `instruction_profile`
//! benchmark in `tests/bench.rs`). The probes themselves cost ~10–15 instructions each.

use std::sync::atomic::AtomicU64;
#[cfg(all(feature = "prof", target_arch = "x86_64"))]
use std::sync::atomic::Ordering;

/// Combinational sweeps (including [`LUT`] and [`FLUSH`] work done inside them).
pub const SWEEP: usize = 0;
/// Round-boundary decisions (pending scan, fast paths, order checks, general transitions).
pub const DECIDE: usize = 1;
/// General-path transitions (part of [`DECIDE`]).
pub const GENERAL: usize = 2;
/// State writes after a boundary (including their [`FLUSH`]).
pub const WRITE: usize = 3;
/// LUT ops (part of [`SWEEP`]).
pub const LUT: usize = 4;
/// Change propagation (the interpreter's batched flush; part of the slot it runs in).
pub const FLUSH: usize = 5;
/// Applying the timed sources of an event (including its [`FLUSH`]).
pub const TIMED: usize = 6;
/// Count (not cycles): net changes propagated.
pub const NOTIFIES: usize = 7;
/// Number of slots.
pub const SLOTS: usize = 8;

/// Accumulated cycles per slot (process-wide).
pub static COUNTERS: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];

/// If non-zero: `1 +` the hardware performance counter index the probes read with `rdpmc`
/// instead of the time-stamp counter (set up by the caller: a perf event mapped into this
/// process).
pub static PMC: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Start a probe.
#[inline(always)]
pub fn start() -> u64 {
    #[cfg(all(feature = "prof", target_arch = "x86_64"))]
    {
        let pmc = PMC.load(Ordering::Relaxed);
        if pmc != 0 {
            let (lo, hi): (u32, u32);
            // SAFETY: only reached after the caller enabled user-space `rdpmc` for counter
            // `pmc - 1` (a perf event mapped into this process).
            unsafe {
                std::arch::asm!("rdpmc", in("ecx") pmc - 1, out("eax") lo, out("edx") hi,
                    options(nomem, nostack));
            }
            return u64::from(hi) << 32 | u64::from(lo);
        }
        // SAFETY: `rdtsc` has no preconditions on x86_64.
        unsafe { std::arch::x86_64::_rdtsc() }
    }
    #[cfg(not(all(feature = "prof", target_arch = "x86_64")))]
    {
        0
    }
}

/// Add the cycles since `t0` to `slot` (single-threaded profiling: a plain load + store).
#[inline(always)]
pub fn stop(slot: usize, t0: u64) {
    #[cfg(all(feature = "prof", target_arch = "x86_64"))]
    {
        let c = &COUNTERS[slot];
        c.store(
            c.load(Ordering::Relaxed) + start().wrapping_sub(t0),
            Ordering::Relaxed,
        );
    }
    #[cfg(not(all(feature = "prof", target_arch = "x86_64")))]
    let _ = (slot, t0);
}

/// Add `n` to counter `slot`.
#[inline(always)]
pub fn count_n(slot: usize, n: u64) {
    #[cfg(all(feature = "prof", target_arch = "x86_64"))]
    {
        let c = &COUNTERS[slot];
        c.store(c.load(Ordering::Relaxed) + n, Ordering::Relaxed);
    }
    #[cfg(not(all(feature = "prof", target_arch = "x86_64")))]
    let _ = (slot, n);
}

/// Reset all slots.
pub fn reset() {
    for c in &COUNTERS {
        c.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}
