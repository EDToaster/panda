//! A tiny two-pass assembler for the Panda-8 ISA (see [`crate::isa`]).
//!
//! # Syntax
//!
//! ```text
//! ; comment (also `#` or `//`)
//! .equ  N, 10            ; named constant
//! start:                 ; label (may share a line with an instruction)
//!         LDI  r0, N     ; mnemonics and register names are case-insensitive
//! loop:   OUT  r0
//!         SUBI r0, 1
//!         JNZ  loop      ; jump targets are expressions (usually a label)
//!         HALT
//!         .org 0x40      ; continue at address 0x40 (gap filled with 0)
//!         .byte 1, 2, 'A'
//! ```
//!
//! * Instructions: every mnemonic of [`crate::isa`] (`NOP LDI MOV ADD SUB AND OR XOR ADDI SUBI
//!   CMP CMPI OUT JMP JZ JNZ JC JNC HALT`), plus the pseudo-instructions `INC rd` (= `ADDI rd, 1`),
//!   `DEC rd` (= `SUBI rd, 1`) and `CLR rd` (= `LDI rd, 0`).
//! * Expressions: terms joined by `+` / `-`. A term is a decimal, `0x` hex or `0b` binary number,
//!   a character literal `'c'`, a label, or a `.equ` name. The value must lie in `-128..=255`;
//!   negative values are stored in two's complement.
//! * Directives: `.org addr` (forward only), `.byte expr, ...`, `.equ name, expr`
//!   (constants must be defined before use).
//! * Identifiers: `[A-Za-z_.][A-Za-z0-9_.]*`, case-sensitive, and not a register name.
//!
//! The output image is as long as the highest address written (at most 256 bytes).

use std::collections::HashMap;
use std::fmt;

use crate::isa::{Cond, Instr, Reg};

/// An assembly error with its 1-based source line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsmError {
    /// 1-based line number.
    pub line: usize,
    /// Description of the problem.
    pub message: String,
}

impl fmt::Display for AsmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for AsmError {}

/// Assemble `source` into a program image.
pub fn assemble(source: &str) -> Result<Vec<u8>, AsmError> {
    Ok(assemble_with_symbols(source)?.0)
}

/// Assemble `source`; also return the symbol table (labels and `.equ` constants).
pub fn assemble_with_symbols(source: &str) -> Result<(Vec<u8>, HashMap<String, u8>), AsmError> {
    let lines: Vec<Line> = source
        .lines()
        .enumerate()
        .map(|(i, l)| parse_line(i + 1, l))
        .collect::<Result<_, _>>()?;

    // Pass 1: addresses of labels; `.equ` values (which may not use forward references).
    let mut symbols: HashMap<String, u8> = HashMap::new();
    let mut addr = 0usize;
    for l in &lines {
        for label in &l.labels {
            define(&mut symbols, label, check_addr(addr, l.no)?, l.no)?;
        }
        match &l.stmt {
            None => {}
            Some(Stmt::Equ(name, e)) => {
                let v = eval(e, &symbols, l.no)?;
                define(&mut symbols, name, v, l.no)?;
            }
            Some(Stmt::Org(e)) => addr = org_target(e, &symbols, addr, l.no)?,
            Some(Stmt::Byte(es)) => addr += es.len(),
            Some(Stmt::Instr(..)) => addr += 2,
        }
    }
    check_end(addr, lines.last().map_or(1, |l| l.no))?;

    // Pass 2: emit.
    let mut image: Vec<u8> = Vec::new();
    let mut addr = 0usize;
    for l in &lines {
        match &l.stmt {
            None | Some(Stmt::Equ(..)) => {}
            Some(Stmt::Org(e)) => {
                addr = org_target(e, &symbols, addr, l.no)?;
                if image.len() < addr {
                    image.resize(addr, 0);
                }
            }
            Some(Stmt::Byte(es)) => {
                for e in es {
                    let v = eval(e, &symbols, l.no)?;
                    put(&mut image, &mut addr, v);
                }
            }
            Some(Stmt::Instr(mnemonic, ops)) => {
                let instr = build_instr(mnemonic, ops, &symbols, l.no)?;
                let [a, b] = instr.encode();
                put(&mut image, &mut addr, a);
                put(&mut image, &mut addr, b);
            }
        }
    }
    Ok((image, symbols))
}

// ---------------------------------------------------------------------------------------------
// parsing
// ---------------------------------------------------------------------------------------------

struct Line {
    no: usize,
    labels: Vec<String>,
    stmt: Option<Stmt>,
}

enum Stmt {
    Org(Expr),
    Byte(Vec<Expr>),
    Equ(String, Expr),
    /// Upper-cased mnemonic and raw operand strings.
    Instr(String, Vec<String>),
}

/// `(negated, term)` pairs.
type Expr = Vec<(bool, String)>;

fn err(line: usize, message: impl Into<String>) -> AsmError {
    AsmError { line, message: message.into() }
}

fn strip_comment(line: &str) -> &str {
    // A `;`/`#`/`//` inside a character literal is not a comment.
    let bytes = line.as_bytes();
    let mut in_char = false;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\'' => in_char = !in_char,
            b';' | b'#' if !in_char => return &line[..i],
            b'/' if !in_char && bytes.get(i + 1) == Some(&b'/') => return &line[..i],
            _ => {}
        }
        i += 1;
    }
    line
}

fn is_ident(s: &str) -> bool {
    let mut cs = s.chars();
    matches!(cs.next(), Some(c) if c.is_ascii_alphabetic() || c == '_' || c == '.')
        && cs.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
}

fn parse_reg(s: &str) -> Option<Reg> {
    match s.to_ascii_lowercase().as_str() {
        "r0" => Some(Reg::R0),
        "r1" => Some(Reg::R1),
        "r2" => Some(Reg::R2),
        "r3" => Some(Reg::R3),
        _ => None,
    }
}

fn parse_line(no: usize, raw: &str) -> Result<Line, AsmError> {
    let mut rest = strip_comment(raw).trim();
    let mut labels = Vec::new();
    // Leading `label:` prefixes (a `:` inside a char literal is not a label separator).
    while let Some(i) = rest.find(':') {
        let head = rest[..i].trim();
        if head.contains('\'') || head.contains(char::is_whitespace) {
            break;
        }
        if !is_ident(head) || parse_reg(head).is_some() {
            return Err(err(no, format!("invalid label '{head}'")));
        }
        labels.push(head.to_string());
        rest = rest[i + 1..].trim();
    }
    if rest.is_empty() {
        return Ok(Line { no, labels, stmt: None });
    }
    let (word, args) = match rest.find(char::is_whitespace) {
        Some(i) => (&rest[..i], rest[i..].trim()),
        None => (rest, ""),
    };
    let ops: Vec<String> =
        if args.is_empty() { Vec::new() } else { args.split(',').map(|s| s.trim().to_string()).collect() };
    if ops.iter().any(String::is_empty) {
        return Err(err(no, "empty operand"));
    }
    let stmt = match word.to_ascii_lowercase().as_str() {
        ".org" => {
            expect_count(&ops, 1, ".org", no)?;
            Stmt::Org(parse_expr(&ops[0], no)?)
        }
        ".byte" => {
            if ops.is_empty() {
                return Err(err(no, ".byte needs at least one value"));
            }
            Stmt::Byte(ops.iter().map(|o| parse_expr(o, no)).collect::<Result<_, _>>()?)
        }
        ".equ" => {
            expect_count(&ops, 2, ".equ", no)?;
            if !is_ident(&ops[0]) || parse_reg(&ops[0]).is_some() {
                return Err(err(no, format!("invalid constant name '{}'", ops[0])));
            }
            Stmt::Equ(ops[0].clone(), parse_expr(&ops[1], no)?)
        }
        w if w.starts_with('.') => return Err(err(no, format!("unknown directive '{word}'"))),
        _ => Stmt::Instr(word.to_ascii_uppercase(), ops),
    };
    Ok(Line { no, labels, stmt: Some(stmt) })
}

fn expect_count(ops: &[String], n: usize, what: &str, no: usize) -> Result<(), AsmError> {
    if ops.len() == n {
        Ok(())
    } else {
        Err(err(no, format!("{what} expects {n} operand(s), got {}", ops.len())))
    }
}

/// Split an expression into signed terms (`a+b-1`, `-1`, `'c'`).
fn parse_expr(s: &str, no: usize) -> Result<Expr, AsmError> {
    let mut terms = Vec::new();
    let mut neg = false;
    let mut cur = String::new();
    let mut chars = s.chars().peekable();
    let mut expect_term = true;
    while let Some(c) = chars.next() {
        match c {
            '\'' => {
                // Character literal: exactly one char between quotes.
                let ch = chars.next().ok_or_else(|| err(no, format!("unterminated character literal in '{s}'")))?;
                if chars.next() != Some('\'') {
                    return Err(err(no, format!("bad character literal in '{s}'")));
                }
                cur.push('\'');
                cur.push(ch);
                cur.push('\'');
                expect_term = false;
            }
            '+' | '-' if expect_term && cur.is_empty() => {
                if c == '-' {
                    neg = !neg;
                }
            }
            '+' | '-' => {
                terms.push((neg, std::mem::take(&mut cur)));
                neg = c == '-';
                expect_term = true;
            }
            c if c.is_whitespace() => {}
            c => {
                cur.push(c);
                expect_term = false;
            }
        }
    }
    if cur.is_empty() {
        return Err(err(no, format!("malformed expression '{s}'")));
    }
    terms.push((neg, cur));
    Ok(terms)
}

fn parse_number(t: &str) -> Option<i64> {
    let lower = t.to_ascii_lowercase();
    let lower = lower.replace('_', "");
    if let Some(h) = lower.strip_prefix("0x") {
        i64::from_str_radix(h, 16).ok()
    } else if let Some(b) = lower.strip_prefix("0b") {
        i64::from_str_radix(b, 2).ok()
    } else if lower.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        lower.parse().ok()
    } else {
        None
    }
}

fn eval(e: &Expr, symbols: &HashMap<String, u8>, no: usize) -> Result<u8, AsmError> {
    let mut acc: i64 = 0;
    for (neg, t) in e {
        let v = if let Some(n) = parse_number(t) {
            n
        } else if t.len() == 3 && t.starts_with('\'') && t.ends_with('\'') {
            let c = t.chars().nth(1).expect("length checked");
            if !c.is_ascii() {
                return Err(err(no, format!("non-ASCII character literal {t}")));
            }
            c as i64
        } else if is_ident(t) {
            if parse_reg(t).is_some() {
                return Err(err(no, format!("register '{t}' used where a value is expected")));
            }
            i64::from(*symbols.get(t.as_str()).ok_or_else(|| err(no, format!("undefined symbol '{t}'")))?)
        } else {
            return Err(err(no, format!("bad value '{t}'")));
        };
        acc += if *neg { -v } else { v };
        if acc.abs() > 1 << 20 {
            return Err(err(no, "expression overflow"));
        }
    }
    if !(-128..=255).contains(&acc) {
        return Err(err(no, format!("value {acc} does not fit in a byte")));
    }
    Ok(acc as u8)
}

fn define(symbols: &mut HashMap<String, u8>, name: &str, v: u8, no: usize) -> Result<(), AsmError> {
    if symbols.insert(name.to_string(), v).is_some() {
        return Err(err(no, format!("symbol '{name}' defined twice")));
    }
    Ok(())
}

fn check_addr(addr: usize, no: usize) -> Result<u8, AsmError> {
    u8::try_from(addr).map_err(|_| err(no, format!("address {addr:#x} is beyond the 256-byte ROM")))
}

fn check_end(addr: usize, no: usize) -> Result<(), AsmError> {
    if addr > 256 { Err(err(no, format!("program is {addr} bytes; the ROM holds 256"))) } else { Ok(()) }
}

fn org_target(e: &Expr, symbols: &HashMap<String, u8>, addr: usize, no: usize) -> Result<usize, AsmError> {
    let t = usize::from(eval(e, symbols, no)?);
    if t < addr {
        return Err(err(no, format!(".org {t:#x} moves backwards (current address {addr:#x})")));
    }
    Ok(t)
}

fn put(image: &mut Vec<u8>, addr: &mut usize, v: u8) {
    if image.len() <= *addr {
        image.resize(*addr + 1, 0);
    }
    image[*addr] = v;
    *addr += 1;
}

fn build_instr(m: &str, ops: &[String], symbols: &HashMap<String, u8>, no: usize) -> Result<Instr, AsmError> {
    let reg = |s: &String| parse_reg(s).ok_or_else(|| err(no, format!("{m}: expected a register, got '{s}'")));
    let val = |s: &String| eval(&parse_expr(s, no)?, symbols, no);
    let n = |k: usize| expect_count(ops, k, m, no);
    Ok(match m {
        "NOP" => n(0).map(|_| Instr::Nop)?,
        "HALT" => n(0).map(|_| Instr::Halt)?,
        "LDI" | "ADDI" | "SUBI" | "CMPI" => {
            n(2)?;
            let (d, i) = (reg(&ops[0])?, val(&ops[1])?);
            match m {
                "LDI" => Instr::Ldi(d, i),
                "ADDI" => Instr::Addi(d, i),
                "SUBI" => Instr::Subi(d, i),
                _ => Instr::Cmpi(d, i),
            }
        }
        "MOV" | "ADD" | "SUB" | "AND" | "OR" | "XOR" | "CMP" => {
            n(2)?;
            let (d, s) = (reg(&ops[0])?, reg(&ops[1])?);
            match m {
                "MOV" => Instr::Mov(d, s),
                "ADD" => Instr::Add(d, s),
                "SUB" => Instr::Sub(d, s),
                "AND" => Instr::And(d, s),
                "OR" => Instr::Or(d, s),
                "XOR" => Instr::Xor(d, s),
                _ => Instr::Cmp(d, s),
            }
        }
        "INC" | "DEC" | "CLR" => {
            n(1)?;
            let d = reg(&ops[0])?;
            match m {
                "INC" => Instr::Addi(d, 1),
                "DEC" => Instr::Subi(d, 1),
                _ => Instr::Ldi(d, 0),
            }
        }
        "OUT" => {
            n(1)?;
            Instr::Out(reg(&ops[0])?)
        }
        "JMP" | "JZ" | "JNZ" | "JC" | "JNC" => {
            n(1)?;
            let a = val(&ops[0])?;
            match m {
                "JMP" => Instr::Jmp(a),
                "JZ" => Instr::Jcc(Cond::Z, a),
                "JNZ" => Instr::Jcc(Cond::Nz, a),
                "JC" => Instr::Jcc(Cond::C, a),
                _ => Instr::Jcc(Cond::Nc, a),
            }
        }
        _ => return Err(err(no, format!("unknown mnemonic '{m}'"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expressions() {
        let s = HashMap::from([("L".to_string(), 0x10u8)]);
        let e = |t: &str| eval(&parse_expr(t, 1).unwrap(), &s, 1);
        assert_eq!(e("1+2").unwrap(), 3);
        assert_eq!(e("-1").unwrap(), 0xff);
        assert_eq!(e("L+2").unwrap(), 0x12);
        assert_eq!(e("L - 1").unwrap(), 0x0f);
        assert_eq!(e("'A'").unwrap(), 65);
        assert_eq!(e("0b101").unwrap(), 5);
        assert_eq!(e("0xF_F").unwrap(), 255);
        assert!(e("256").is_err());
        assert!(e("-129").is_err());
        assert!(e("nope").is_err());
        assert!(e("r1").is_err());
    }

    #[test]
    fn comments_respect_char_literals() {
        assert_eq!(strip_comment("LDI r0, ';' ; c"), "LDI r0, ';' ");
        assert_eq!(strip_comment("x // y"), "x ");
    }
}
