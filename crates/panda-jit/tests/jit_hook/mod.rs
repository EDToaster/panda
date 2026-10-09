//! Installs the JIT into every `FastSim` the panda-fast suites build (`common::pair` calls the
//! test crate's `native_hook`). Programs are compiled with one shared [`Jit`] (so its cache is
//! exercised too); every fourth engine gets a mix of native and interpreted partitions and
//! event functions, to test mixed tiers.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use panda_fast::FastSim;
use panda_jit::{Jit, JitOptions};

/// Engines given native code so far.
pub static INSTALLED: AtomicU64 = AtomicU64::new(0);

/// Compile `fs`'s program and install it (possibly a mixed-tier subset).
pub fn install(fs: &mut FastSim) {
    static JIT: OnceLock<Jit> = OnceLock::new();
    let jit = JIT.get_or_init(|| Jit::new(JitOptions::default()));
    let (code, _) = jit.compile(fs.program_arc()).expect("JIT compile");
    let n = INSTALLED.fetch_add(1, Ordering::Relaxed);
    let code = match n % 8 {
        3 => Arc::new(code.subset(|p| p % 2 == 0, |_| true)),
        5 => Arc::new(code.subset(|_| true, |_| false)),
        7 => Arc::new(code.subset(|p| p % 3 != 1, |t| t % 2 == 1)),
        _ => code,
    };
    if std::env::var_os("PANDA_JIT_HOOK_VERBOSE").is_some() {
        eprintln!("jit hook: engine {n}: {:?}", code);
    }
    fs.install_native(code).expect("install");
}
