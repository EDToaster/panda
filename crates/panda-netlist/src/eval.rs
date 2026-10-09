//! The single source of truth for 4-state gate semantics (SPEC §3.2, Appendix A/B).
//! Every engine and compiler pass MUST use (or be tested against) these functions.

use crate::flat::FlatOp;
use crate::model::Termination;
use crate::value::Logic;

// ---------------------------------------------------------------------------------------------
// Scalar reference semantics
// ---------------------------------------------------------------------------------------------

#[inline]
pub fn not1(a: Logic) -> Logic {
    match a.normalize() {
        Logic::Zero => Logic::One,
        Logic::One => Logic::Zero,
        _ => Logic::X,
    }
}

#[inline]
pub fn and2(a: Logic, b: Logic) -> Logic {
    match (a.normalize(), b.normalize()) {
        (Logic::Zero, _) | (_, Logic::Zero) => Logic::Zero,
        (Logic::One, Logic::One) => Logic::One,
        _ => Logic::X,
    }
}

#[inline]
pub fn or2(a: Logic, b: Logic) -> Logic {
    match (a.normalize(), b.normalize()) {
        (Logic::One, _) | (_, Logic::One) => Logic::One,
        (Logic::Zero, Logic::Zero) => Logic::Zero,
        _ => Logic::X,
    }
}

#[inline]
pub fn xor2(a: Logic, b: Logic) -> Logic {
    match (a.normalize(), b.normalize()) {
        (Logic::Zero, Logic::Zero) | (Logic::One, Logic::One) => Logic::Zero,
        (Logic::Zero, Logic::One) | (Logic::One, Logic::Zero) => Logic::One,
        _ => Logic::X,
    }
}

/// Tri-state buffer: `en=1 → d` (normalized, so Z on d reads as X), `en=0 → Z`, else X.
#[inline]
pub fn tri(d: Logic, en: Logic) -> Logic {
    match en.normalize() {
        Logic::One => d.normalize(),
        Logic::Zero => Logic::Z,
        _ => Logic::X,
    }
}

/// Evaluate a flat gate over its inputs (in order). N-input gates fold left-to-right; this is
/// exact because the 4-state AND/OR/XOR above are associative and commutative.
pub fn eval_gate(op: FlatOp, inputs: &[Logic]) -> Logic {
    match op {
        FlatOp::Not => not1(inputs[0]),
        FlatOp::Buf => inputs[0].normalize(),
        FlatOp::Tri => tri(inputs[0], inputs[1]),
        FlatOp::And | FlatOp::Nand => {
            let v = inputs.iter().copied().fold(Logic::One, and2);
            if op == FlatOp::Nand { not1(v) } else { v }
        }
        FlatOp::Or | FlatOp::Nor => {
            let v = inputs.iter().copied().fold(Logic::Zero, or2);
            if op == FlatOp::Nor { not1(v) } else { v }
        }
        FlatOp::Xor | FlatOp::Xnor => {
            let v = inputs.iter().copied().fold(Logic::Zero, xor2);
            if op == FlatOp::Xnor { not1(v) } else { v }
        }
    }
}

/// The controlling input value of a gate, if any: an input at this value fixes the output
/// regardless of the other inputs (used for guarded skipping, SPEC §8.6).
pub fn controlling_value(op: FlatOp) -> Option<Logic> {
    match op {
        FlatOp::And | FlatOp::Nand => Some(Logic::Zero),
        FlatOp::Or | FlatOp::Nor => Some(Logic::One),
        _ => None,
    }
}

/// Result of resolving a multi-driver net.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resolved {
    pub value: Logic,
    /// ≥2 non-Z drivers that disagree or include X.
    pub contention: bool,
}

/// Resolve drivers of a net (SPEC §3.3). `keeper_prev` is the keeper's held value (used only for
/// `Termination::Keeper`; pass the previous resolved output). Drivers are raw (Z allowed).
pub fn resolve(drivers: &[Logic], termination: Termination, keeper_prev: Logic) -> Resolved {
    let mut val: Option<Logic> = None;
    let mut contention = false;
    for &d in drivers {
        if d == Logic::Z {
            continue;
        }
        match val {
            None => val = Some(d),
            Some(v) => {
                if v != d || !d.is_known() {
                    contention = true;
                    val = Some(Logic::X);
                }
            }
        }
    }
    let value = match val {
        Some(v) => v,
        None => terminate(termination, keeper_prev),
    };
    Resolved { value, contention }
}

/// Value of a net with no active (non-Z) driver.
#[inline]
pub fn terminate(termination: Termination, keeper_prev: Logic) -> Logic {
    match termination {
        Termination::None => Logic::Z,
        Termination::PullUp => Logic::One,
        Termination::PullDown => Logic::Zero,
        Termination::Keeper => {
            if keeper_prev == Logic::Z { Logic::X } else { keeper_prev }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Scalar 2-bit-code tables (Appendix B): index = a | b << 2, entry = 2-bit result code.
// ---------------------------------------------------------------------------------------------

/// Packed 2-input table for `f`.
pub fn table2(f: impl Fn(Logic, Logic) -> Logic) -> u32 {
    let mut t = 0u32;
    for a in 0..4u8 {
        for b in 0..4u8 {
            let r = f(Logic::from_code(a), Logic::from_code(b)).code() as u32;
            t |= r << (((a | (b << 2)) as u32) * 2);
        }
    }
    t
}

/// Look up a packed table.
#[inline(always)]
pub fn lookup2(table: u32, a: Logic, b: Logic) -> Logic {
    Logic::from_code(((table >> (((a.code() | (b.code() << 2)) as u32) * 2)) & 3) as u8)
}

/// The packed tables for every 2-input op family, computed from the reference functions.
/// Index order: And, Or, Xor, Nand, Nor, Xnor.
pub fn tables() -> [u32; 6] {
    [
        table2(and2),
        table2(or2),
        table2(xor2),
        table2(|a, b| not1(and2(a, b))),
        table2(|a, b| not1(or2(a, b))),
        table2(|a, b| not1(xor2(a, b))),
    ]
}

// ---------------------------------------------------------------------------------------------
// Two-plane word ops (Appendix A). Planes: 0=(0,0) 1=(1,0) Z=(0,1) X=(1,1).
// ---------------------------------------------------------------------------------------------

/// Normalize Z→X: `a |= b`.
#[inline(always)]
pub fn w_norm(a: u64, b: u64) -> (u64, u64) {
    (a | b, b)
}

#[inline(always)]
pub fn w_not(a: u64, b: u64) -> (u64, u64) {
    let (a, b) = w_norm(a, b);
    (!a | b, b)
}

#[inline(always)]
pub fn w_and(a1: u64, b1: u64, a2: u64, b2: u64) -> (u64, u64) {
    let (a1, b1) = w_norm(a1, b1);
    let (a2, b2) = w_norm(a2, b2);
    let a = a1 & a2;
    (a, a & (b1 | b2))
}

#[inline(always)]
pub fn w_or(a1: u64, b1: u64, a2: u64, b2: u64) -> (u64, u64) {
    let (a1, b1) = w_norm(a1, b1);
    let (a2, b2) = w_norm(a2, b2);
    let a = a1 | a2;
    let one = (a1 & !b1) | (a2 & !b2);
    (a, a & !one)
}

#[inline(always)]
pub fn w_xor(a1: u64, b1: u64, a2: u64, b2: u64) -> (u64, u64) {
    let (a1, b1) = w_norm(a1, b1);
    let (a2, b2) = w_norm(a2, b2);
    let b = b1 | b2;
    ((a1 ^ a2) | b, b)
}

/// Tri-state on words: `en` planes select per bit.
#[inline(always)]
pub fn w_tri(da: u64, db: u64, ea: u64, eb: u64) -> (u64, u64) {
    let (da, db) = w_norm(da, db);
    let (ea, eb) = w_norm(ea, eb);
    let en1 = ea & !eb;
    let en0 = !ea;
    let enx = eb;
    // en1 → d ; en0 → Z (0,1) ; enx → X (1,1)
    ((da & en1) | enx, (db & en1) | en0 | enx)
}

#[cfg(test)]
mod tests {
    use super::*;
    const ALL: [Logic; 4] = [Logic::Zero, Logic::One, Logic::X, Logic::Z];

    fn wp(l: Logic) -> (u64, u64) {
        let (a, b) = l.planes();
        (a as u64, b as u64)
    }
    fn wl((a, b): (u64, u64)) -> Logic {
        Logic::from_planes(a & 1 == 1, b & 1 == 1)
    }

    #[test]
    fn word_ops_match_scalar() {
        for x in ALL {
            assert_eq!(wl(w_not(wp(x).0, wp(x).1)), not1(x));
            for y in ALL {
                let (a1, b1) = wp(x);
                let (a2, b2) = wp(y);
                assert_eq!(wl(w_and(a1, b1, a2, b2)), and2(x, y), "and {x} {y}");
                assert_eq!(wl(w_or(a1, b1, a2, b2)), or2(x, y), "or {x} {y}");
                assert_eq!(wl(w_xor(a1, b1, a2, b2)), xor2(x, y), "xor {x} {y}");
                assert_eq!(wl(w_tri(a1, b1, a2, b2)), tri(x, y), "tri {x} {y}");
            }
        }
    }

    #[test]
    fn tables_match_scalar() {
        let t = tables();
        for x in ALL {
            for y in ALL {
                assert_eq!(lookup2(t[0], x, y), and2(x, y));
                assert_eq!(lookup2(t[3], x, y), not1(and2(x, y)));
                assert_eq!(lookup2(t[5], x, y), not1(xor2(x, y)));
            }
        }
    }

    #[test]
    fn associativity() {
        for x in ALL {
            for y in ALL {
                for z in ALL {
                    assert_eq!(and2(and2(x, y), z), and2(x, and2(y, z)));
                    assert_eq!(or2(or2(x, y), z), or2(x, or2(y, z)));
                    assert_eq!(xor2(xor2(x, y), z), xor2(x, xor2(y, z)));
                }
            }
        }
    }

    #[test]
    fn resolution() {
        use Logic::*;
        assert_eq!(resolve(&[Z, Z], Termination::None, X).value, Z);
        assert_eq!(resolve(&[Z, Z], Termination::PullUp, X).value, One);
        assert_eq!(resolve(&[Z, Z], Termination::Keeper, Zero).value, Zero);
        assert_eq!(resolve(&[One, Z], Termination::None, X), Resolved { value: One, contention: false });
        assert_eq!(resolve(&[One, Zero], Termination::None, X), Resolved { value: X, contention: true });
        assert_eq!(resolve(&[One, One], Termination::None, X), Resolved { value: One, contention: false });
    }
}
