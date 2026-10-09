//! Small helpers shared by the generators.

use panda_netlist::{DefBuilder, Sig};

/// Replicate a 1-bit signal to `width` bits (a Splitter whose parts all alias the same net).
/// Returns `s` itself when `width == 1`.
pub(crate) fn broadcast(b: &mut DefBuilder, s: Sig, width: u32) -> Sig {
    assert_eq!(s.width, 1, "broadcast: source must be 1 bit wide");
    if width == 1 { s } else { b.concat(&vec![s; width as usize]) }
}

/// Number of address bits needed to select one of `words` entries: `ceil(log2(words))`,
/// with a minimum of 1 (a 1-word memory still has a 1-bit address).
pub fn addr_bits(words: usize) -> u32 {
    assert!(words >= 1, "addr_bits: need at least one word");
    (usize::BITS - (words - 1).leading_zeros()).max(1)
}

/// 64-bit FNV-1a hash (deterministic across runs and platforms; used for content-addressed ids).
pub(crate) fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addr_bits_values() {
        assert_eq!(addr_bits(1), 1);
        assert_eq!(addr_bits(2), 1);
        assert_eq!(addr_bits(3), 2);
        assert_eq!(addr_bits(4), 2);
        assert_eq!(addr_bits(5), 3);
        assert_eq!(addr_bits(256), 8);
        assert_eq!(addr_bits(257), 9);
    }

    #[test]
    fn fnv_is_stable() {
        assert_eq!(fnv1a(*b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(*b"a"), 0xaf63_dc4c_8601_ec8c);
    }
}
