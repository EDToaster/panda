//! panda-fast's `adversarial` differential suite (FastSim = FastRef after every event) with the JIT
//! installed in every engine the suite builds.

mod jit_hook;

#[path = "../../panda-fast/tests/adversarial.rs"]
mod suite;

fn native_hook(fs: &mut panda_fast::FastSim) {
    jit_hook::install(fs);
}
