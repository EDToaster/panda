//! panda-fast's `random_netlists` differential suite (FastSim = FastRef after every event) with the JIT
//! installed in every engine the suite builds.

mod jit_hook;

#[path = "../../panda-fast/tests/random_netlists.rs"]
mod suite;

fn native_hook(fs: &mut panda_fast::FastSim) {
    jit_hook::install(fs);
}
