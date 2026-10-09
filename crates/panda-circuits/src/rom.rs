//! Gate-built ROM: constants + mux tree (SPEC §7.4), and a hex-file loader.

use std::collections::HashMap;

use panda_netlist::{DefBuilder, DefRef, DesignBuilder, Sig};

use crate::arith::{mux_tree, mux2};
use crate::util::fnv1a;

/// A `2^addr_bits × data_bits` ROM built from `Constant` atoms and a tree of [`mux2`] instances.
///
/// Id `"rom_{name}_{addr_bits}x{data_bits}_{hash}"`, where `hash` is a 16-hex-digit content hash,
/// so ROMs with different contents never share a def and equal ROMs always do.
///
/// Pins: `A` (in, addr_bits) → `D` (out, data_bits). `D = contents[A]`; words beyond
/// `contents.len()` read 0. One `Constant` atom is emitted per *distinct* word value; the mux tree
/// is complete (`2^addr_bits - 1` mux instances `m{level}_{index}`). With an `X` address bit the
/// gates are pessimistic: an output bit is known only where every candidate word has a 0.
///
/// Panics if `name` is not a valid identifier (empty or containing `/`, `[`, `]`), if
/// `addr_bits ∉ 1..=16`, `data_bits ∉ 1..=64`, if `contents` has more than `2^addr_bits` words, or
/// if a word does not fit in `data_bits` bits.
pub fn rom(db: &mut DesignBuilder, name: &str, addr_bits: u32, data_bits: u32, contents: &[u64]) -> DefRef {
    assert!(!name.is_empty() && !name.contains(['/', '[', ']']), "rom: invalid name '{name}'");
    assert!((1..=16).contains(&addr_bits), "rom: addr_bits must be in 1..=16");
    assert!((1..=64).contains(&data_bits), "rom: data_bits must be in 1..=64");
    let words = 1usize << addr_bits;
    assert!(contents.len() <= words, "rom: {} words do not fit in {words}", contents.len());
    if data_bits < 64
        && let Some((i, v)) = contents.iter().enumerate().find(|(_, v)| **v >> data_bits != 0)
    {
        panic!("rom: word {i} = {v:#x} does not fit in {data_bits} bits");
    }
    // Content hash over the significant prefix (trailing zero words are equivalent to absent ones).
    let used = contents.iter().rposition(|&v| v != 0).map_or(0, |i| i + 1);
    let hash = fnv1a(
        [addr_bits, data_bits]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .chain(contents[..used].iter().flat_map(|v| v.to_le_bytes())),
    );
    let id = format!("rom_{name}_{addr_bits}x{data_bits}_{hash:016x}");
    if let Some(r) = db.get(&id) {
        return r;
    }
    let m2 = mux2(db, data_bits);
    let mut b = DefBuilder::new(&id);
    b.set_name(name);
    let a = b.input("A", addr_bits);
    let sels = b.bits(a);
    let mut consts: HashMap<u64, Sig> = HashMap::new();
    let leaves: Vec<Sig> = (0..words)
        .map(|i| {
            let v = contents.get(i).copied().unwrap_or(0);
            *consts.entry(v).or_insert_with(|| b.lit(data_bits, v))
        })
        .collect();
    let d = mux_tree(&mut b, &m2, leaves, &sels, "m");
    b.output("D", d);
    db.add(b.finish())
}

/// Parse a hex memory image into words (for [`rom`]).
///
/// Format (a superset of Verilog `$readmemh` without ranges):
/// * words are hexadecimal, separated by whitespace or commas; an optional `0x` prefix and `_`
///   separators are allowed;
/// * `@addr` (hex, at most `0x100_0000`) sets the address of the next word; skipped words are 0;
///   an address may also move backwards, and a word written again overwrites the earlier one;
/// * comments run from `#`, `;` or `//` to the end of the line.
///
/// Returns the words in address order, or an error naming the offending token and line.
pub fn rom_from_hex(text: &str) -> Result<Vec<u64>, String> {
    let mut out: Vec<u64> = Vec::new();
    let mut addr = 0usize;
    for (lineno, line) in text.lines().enumerate() {
        let line =
            line.find("//").into_iter().chain(line.find('#')).chain(line.find(';')).min().map_or(line, |i| &line[..i]);
        for tok in line.split(|c: char| c.is_whitespace() || c == ',').filter(|t| !t.is_empty()) {
            let parse = |s: &str| -> Result<u64, String> {
                let s: String = s.chars().filter(|&c| c != '_').collect();
                let digits = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(&s);
                u64::from_str_radix(digits, 16).map_err(|e| format!("line {}: bad hex token '{tok}': {e}", lineno + 1))
            };
            if let Some(a) = tok.strip_prefix('@') {
                addr = usize::try_from(parse(a)?).map_err(|e| format!("line {}: {e}", lineno + 1))?;
                if addr > 1 << 24 {
                    return Err(format!("line {}: address {addr:#x} too large", lineno + 1));
                }
                continue;
            }
            let v = parse(tok)?;
            if out.len() <= addr {
                out.resize(addr + 1, 0);
            }
            out[addr] = v;
            addr += 1;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_basic() {
        assert_eq!(rom_from_hex("01 02 ff\n0x10,0X2_0").unwrap(), vec![1, 2, 255, 16, 32]);
    }

    #[test]
    fn hex_comments_and_addresses() {
        let text = "// header\n@4 aa # four\nbb ; five\n@1 11\n";
        assert_eq!(rom_from_hex(text).unwrap(), vec![0, 0x11, 0, 0, 0xaa, 0xbb]);
    }

    #[test]
    fn hex_errors() {
        let e = rom_from_hex("12\n zz").unwrap_err();
        assert!(e.contains("line 2") && e.contains("zz"), "{e}");
        assert!(rom_from_hex("@zz").is_err());
        assert!(rom_from_hex("@ffffffff 1").is_err());
        assert_eq!(rom_from_hex("").unwrap(), Vec::<u64>::new());
    }

    #[test]
    fn rom_ids_are_content_addressed() {
        let mut db = DesignBuilder::new();
        let a = rom(&mut db, "r", 2, 8, &[1, 2, 3]);
        let b = rom(&mut db, "r", 2, 8, &[1, 2, 3, 0]);
        let c = rom(&mut db, "r", 2, 8, &[1, 2, 4]);
        assert_eq!(a, b);
        assert_ne!(a.id, c.id);
        assert!(a.id.starts_with("rom_r_2x8_"));
    }

    #[test]
    #[should_panic(expected = "does not fit")]
    fn rom_rejects_wide_words() {
        let mut db = DesignBuilder::new();
        rom(&mut db, "r", 2, 4, &[16]);
    }
}
