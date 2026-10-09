//! A small, fast, non-cryptographic hasher (the "Fx" multiply-rotate hash) for the engine's
//! runtime memo tables, whose keys are a few integers (std's SipHash costs more than the lookup
//! saves there).

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Fx hasher state.
#[derive(Clone, Copy, Default)]
pub(crate) struct FxHasher {
    hash: u64,
}

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl FxHasher {
    #[inline]
    fn add(&mut self, w: u64) {
        self.hash = (self.hash.rotate_left(5) ^ w).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut w = [0u8; 8];
            w[..chunk.len()].copy_from_slice(chunk);
            self.add(u64::from_le_bytes(w));
        }
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

/// A `HashMap` with [`FxHasher`].
pub(crate) type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_works() {
        let mut m: FxHashMap<(u32, u32, u32, u32), bool> = FxHashMap::default();
        for i in 0..1000u32 {
            m.insert((i, i * 3, i ^ 5, 7), i % 3 == 0);
        }
        for i in 0..1000u32 {
            assert_eq!(m.get(&(i, i * 3, i ^ 5, 7)), Some(&(i % 3 == 0)));
        }
        assert_eq!(m.get(&(1, 2, 3, 4)), None);
    }
}
