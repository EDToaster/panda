//! A tiny deterministic PRNG (SplitMix64), so every generated design is reproducible from its seed
//! on every platform without external dependencies.

/// SplitMix64 generator.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    /// Generator for `seed` (any value, including 0).
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x6A09_E667_F3BC_C909)
    }

    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (`n > 0`).
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "Rng::below(0)");
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform in `lo..=hi`.
    pub fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo + 1)
    }

    /// True with probability `pct` percent.
    pub fn chance(&mut self, pct: u32) -> bool {
        self.next_u64() % 100 < u64::from(pct)
    }

    /// A random element of a non-empty slice.
    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }

    /// Index chosen with probability proportional to `weights[i]` (not all zero).
    pub fn weighted(&mut self, weights: &[u32]) -> usize {
        let total: u64 = weights.iter().map(|&w| u64::from(w)).sum();
        assert!(total > 0, "Rng::weighted: all weights zero");
        let mut r = self.next_u64() % total;
        for (i, &w) in weights.iter().enumerate() {
            if r < u64::from(w) {
                return i;
            }
            r -= u64::from(w);
        }
        unreachable!()
    }

    /// An independent generator derived from this one (for sub-streams that must not disturb the
    /// main sequence).
    pub fn fork(&mut self, salt: u64) -> Rng {
        Rng::new(self.next_u64() ^ salt.wrapping_mul(0xD6E8_FEB8_6659_FD93))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_in_range() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        for _ in 0..1000 {
            assert!(a.below(7) < 7);
            let r = a.range(3, 5);
            assert!((3..=5).contains(&r));
            assert!(a.weighted(&[0, 3, 0, 1]) % 2 == 1);
        }
    }
}
