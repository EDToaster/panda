//! panda-fast's `cpu8` differential suite (FastSim = FastRef after every event) with the JIT
//! installed in every engine the suite builds.

mod jit_hook;

#[path = "../../panda-fast/tests/cpu8.rs"]
mod suite;

fn native_hook(fs: &mut panda_fast::FastSim) {
    jit_hook::install(fs);
}
