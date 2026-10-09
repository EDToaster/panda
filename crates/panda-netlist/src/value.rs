//! 4-state logic values (SPEC §3.1).
//!
//! Two encodings exist in the system:
//! * [`Logic`] — a single bit, scalar code `0=0, 1=1, 2=X, 3=Z`.
//! * [`Value`] — a W-bit vector stored as two bit-planes `(a, b)`:
//!   `0=(0,0) 1=(1,0) Z=(0,1) X=(1,1)`, packed LSB-first into `u64` words.

use std::fmt;

/// One 4-state bit. The discriminant is the scalar code used by table-driven gates.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
#[repr(u8)]
pub enum Logic {
    Zero = 0,
    One = 1,
    #[default]
    X = 2,
    Z = 3,
}

impl Logic {
    #[inline]
    pub const fn from_code(code: u8) -> Logic {
        match code & 3 {
            0 => Logic::Zero,
            1 => Logic::One,
            2 => Logic::X,
            _ => Logic::Z,
        }
    }
    #[inline]
    pub const fn code(self) -> u8 {
        self as u8
    }
    #[inline]
    pub const fn from_bool(b: bool) -> Logic {
        if b { Logic::One } else { Logic::Zero }
    }
    #[inline]
    pub const fn is_known(self) -> bool {
        matches!(self, Logic::Zero | Logic::One)
    }
    /// How a gate input sees this value: `Z` reads as `X` (SPEC §3.2).
    #[inline]
    pub const fn normalize(self) -> Logic {
        match self {
            Logic::Z => Logic::X,
            v => v,
        }
    }
    /// Plane encoding `(a, b)`.
    #[inline]
    pub const fn planes(self) -> (bool, bool) {
        match self {
            Logic::Zero => (false, false),
            Logic::One => (true, false),
            Logic::Z => (false, true),
            Logic::X => (true, true),
        }
    }
    #[inline]
    pub const fn from_planes(a: bool, b: bool) -> Logic {
        match (a, b) {
            (false, false) => Logic::Zero,
            (true, false) => Logic::One,
            (false, true) => Logic::Z,
            (true, true) => Logic::X,
        }
    }
    pub const fn to_char(self) -> char {
        match self {
            Logic::Zero => '0',
            Logic::One => '1',
            Logic::X => 'x',
            Logic::Z => 'z',
        }
    }
    pub fn from_char(c: char) -> Option<Logic> {
        Some(match c {
            '0' => Logic::Zero,
            '1' => Logic::One,
            'x' | 'X' => Logic::X,
            'z' | 'Z' => Logic::Z,
            _ => return None,
        })
    }
}

impl fmt::Display for Logic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

/// A W-bit 4-state vector (two-plane encoding). Bits above `width` in the last word are zero.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Value {
    width: u32,
    a: Vec<u64>,
    b: Vec<u64>,
}

#[inline]
fn words(width: u32) -> usize {
    (width as usize).div_ceil(64).max(1)
}

impl Value {
    fn filled(width: u32, a: bool, b: bool) -> Value {
        let n = words(width);
        let mut v = Value { width, a: vec![if a { !0 } else { 0 }; n], b: vec![if b { !0 } else { 0 }; n] };
        v.mask_top();
        v
    }
    fn mask_top(&mut self) {
        let rem = self.width % 64;
        if rem != 0 {
            let m = (1u64 << rem) - 1;
            let last = self.a.len() - 1;
            self.a[last] &= m;
            self.b[last] &= m;
        }
        if self.width == 0 {
            self.a[0] = 0;
            self.b[0] = 0;
        }
    }
    pub fn zeros(width: u32) -> Value {
        Value::filled(width, false, false)
    }
    pub fn ones(width: u32) -> Value {
        Value::filled(width, true, false)
    }
    pub fn xs(width: u32) -> Value {
        Value::filled(width, true, true)
    }
    pub fn zs(width: u32) -> Value {
        Value::filled(width, false, true)
    }
    /// Known value from the low `width` bits of `v` (width ≤ 64 uses one word; wider widths zero-extend).
    pub fn from_u64(width: u32, v: u64) -> Value {
        let mut out = Value::zeros(width);
        out.a[0] = v;
        out.mask_top();
        out
    }
    pub fn from_bool(b: bool) -> Value {
        Value::from_u64(1, b as u64)
    }
    pub fn from_logic(l: Logic) -> Value {
        Value::from_bits(&[l])
    }
    /// Bits LSB-first.
    pub fn from_bits(bits: &[Logic]) -> Value {
        let mut v = Value::zeros(bits.len() as u32);
        for (i, &l) in bits.iter().enumerate() {
            v.set(i as u32, l);
        }
        v
    }
    /// Construct from raw planes (LSB-first words). Extra high bits are masked off.
    pub fn from_planes(width: u32, a: &[u64], b: &[u64]) -> Value {
        let n = words(width);
        let mut v = Value { width, a: vec![0; n], b: vec![0; n] };
        for i in 0..n {
            v.a[i] = a.get(i).copied().unwrap_or(0);
            v.b[i] = b.get(i).copied().unwrap_or(0);
        }
        v.mask_top();
        v
    }
    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }
    #[inline]
    pub fn plane_a(&self) -> &[u64] {
        &self.a
    }
    #[inline]
    pub fn plane_b(&self) -> &[u64] {
        &self.b
    }
    #[inline]
    pub fn get(&self, bit: u32) -> Logic {
        assert!(bit < self.width, "bit {bit} out of range for width {}", self.width);
        let (w, s) = ((bit / 64) as usize, bit % 64);
        Logic::from_planes((self.a[w] >> s) & 1 == 1, (self.b[w] >> s) & 1 == 1)
    }
    #[inline]
    pub fn set(&mut self, bit: u32, l: Logic) {
        assert!(bit < self.width, "bit {bit} out of range for width {}", self.width);
        let (w, s) = ((bit / 64) as usize, bit % 64);
        let (a, b) = l.planes();
        self.a[w] = (self.a[w] & !(1 << s)) | ((a as u64) << s);
        self.b[w] = (self.b[w] & !(1 << s)) | ((b as u64) << s);
    }
    pub fn bits(&self) -> impl Iterator<Item = Logic> + '_ {
        (0..self.width).map(|i| self.get(i))
    }
    pub fn is_fully_known(&self) -> bool {
        self.b.iter().all(|&w| w == 0)
    }
    /// The value as an integer if fully known and width ≤ 64.
    pub fn to_u64(&self) -> Option<u64> {
        if self.width <= 64 && self.is_fully_known() { Some(self.a[0]) } else { None }
    }
    /// Bits `lo..lo+width` as a new value.
    pub fn slice(&self, lo: u32, width: u32) -> Value {
        Value::from_bits(&(lo..lo + width).map(|i| self.get(i)).collect::<Vec<_>>())
    }

    /// Parse a value of the given width. Accepted forms:
    /// * decimal: `"42"`
    /// * hex: `"0x2a"` / binary: `"0b1x0z"` (binary may contain `x`/`z`)
    /// * a single `"x"` or `"z"` fills the whole width.
    ///
    /// Underscores are ignored. Values wider than `width` are an error.
    pub fn parse(width: u32, s: &str) -> Result<Value, String> {
        let s: String = s.trim().chars().filter(|&c| c != '_').collect();
        let lower = s.to_ascii_lowercase();
        if lower == "x" {
            return Ok(Value::xs(width));
        }
        if lower == "z" {
            return Ok(Value::zs(width));
        }
        if let Some(bin) = lower.strip_prefix("0b") {
            if bin.len() as u32 > width {
                return Err(format!("binary literal '{s}' wider than {width} bits"));
            }
            let mut v = Value::zeros(width);
            for (i, c) in bin.chars().rev().enumerate() {
                let l = Logic::from_char(c).ok_or_else(|| format!("bad binary digit '{c}' in '{s}'"))?;
                v.set(i as u32, l);
            }
            return Ok(v);
        }
        let n = if let Some(hex) = lower.strip_prefix("0x") {
            u128::from_str_radix(hex, 16).map_err(|e| format!("bad hex '{s}': {e}"))?
        } else {
            lower.parse::<u128>().map_err(|e| format!("bad number '{s}': {e}"))?
        };
        if width < 128 && n >> width != 0 {
            return Err(format!("value '{s}' does not fit in {width} bits"));
        }
        let mut v = Value::zeros(width);
        for i in 0..width.min(128) {
            if (n >> i) & 1 == 1 {
                v.set(i, Logic::One);
            }
        }
        Ok(v)
    }
}

impl fmt::Display for Value {
    /// Binary, MSB first, e.g. `0b10x1`. Fully known values ≤ 64 bits also show hex: `0b1010 (0xa)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0b")?;
        for i in (0..self.width).rev() {
            write!(f, "{}", self.get(i))?;
        }
        Ok(())
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value({}'{})", self.width, self)
    }
}

impl serde::Serialize for Value {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&format!("{}'{}", self.width, self))
    }
}

impl<'de> serde::Deserialize<'de> for Value {
    /// Format: `"<width>'<literal>"`, e.g. `"8'0xff"`, `"4'0b10xz"`, `"1'1"`.
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Value, D::Error> {
        let s = String::deserialize(d)?;
        let (w, lit) = s.split_once('\'').ok_or_else(|| serde::de::Error::custom("expected <width>'<literal>"))?;
        let w: u32 = w.parse().map_err(serde::de::Error::custom)?;
        Value::parse(w, lit).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_parse_display() {
        let v = Value::parse(4, "0b10xz").unwrap();
        assert_eq!(v.get(0), Logic::Z);
        assert_eq!(v.get(1), Logic::X);
        assert_eq!(v.get(3), Logic::One);
        assert_eq!(v.to_string(), "0b10xz");
        assert_eq!(Value::parse(8, "0xff").unwrap().to_u64(), Some(255));
        assert!(Value::parse(4, "16").is_err());
        let wide = Value::parse(100, "x").unwrap();
        assert_eq!(wide.get(99), Logic::X);
        let j = serde_json::to_string(&v).unwrap();
        assert_eq!(serde_json::from_str::<Value>(&j).unwrap(), v);
    }
}
