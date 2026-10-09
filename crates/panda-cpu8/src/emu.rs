//! Cycle-level behavioral emulator of the Panda-8 microarchitecture (see [`crate::isa`]).
//!
//! One [`Emulator::step`] is one rising clock edge. The emulator models exactly the state the
//! gate-level design holds (including `IR`, the fetch/execute `PHASE` and the `OUT_STROBE`
//! flip-flop), so a gate-level run can be compared against it cycle by cycle.

use crate::isa::{Instr, Reg};

/// Complete architectural (and microarchitectural) state, as visible on the debug pins of the
/// gate-level design.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ArchState {
    /// Program counter.
    pub pc: u8,
    /// Instruction register (opcode byte of the current instruction).
    pub ir: u8,
    /// `false` = fetch cycle, `true` = execute cycle.
    pub execute: bool,
    /// General registers `r0..r3`.
    pub regs: [u8; 4],
    /// Zero flag.
    pub z: bool,
    /// Carry flag.
    pub c: bool,
    /// Output port register.
    pub out: u8,
    /// Output strobe: high for the cycle after an `OUT` executed.
    pub out_strobe: bool,
    /// Set by `HALT`; freezes everything else.
    pub halted: bool,
}

impl ArchState {
    /// The state right after reset (everything 0, about to fetch from address 0).
    pub fn reset() -> Self {
        Self::default()
    }

    /// Register value.
    pub fn reg(&self, r: Reg) -> u8 {
        self.regs[r.index()]
    }
}

/// What happened at one clock edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepInfo {
    /// Instruction executed at this edge (`None` for a fetch edge or while halted).
    pub executed: Option<Instr>,
    /// Value written to the output port at this edge, if any.
    pub out: Option<u8>,
}

/// The behavioral emulator.
#[derive(Clone, Debug)]
pub struct Emulator {
    rom: [u8; 256],
    state: ArchState,
    cycles: u64,
}

impl Emulator {
    /// Emulator for `program` (at most 256 bytes; the rest of the ROM reads 0), in the reset state.
    pub fn new(program: &[u8]) -> Self {
        assert!(program.len() <= 256, "program of {} bytes does not fit the 256-byte ROM", program.len());
        let mut rom = [0u8; 256];
        rom[..program.len()].copy_from_slice(program);
        Emulator { rom, state: ArchState::reset(), cycles: 0 }
    }

    /// Current state.
    pub fn state(&self) -> &ArchState {
        &self.state
    }

    /// Clock edges since reset.
    pub fn cycles(&self) -> u64 {
        self.cycles
    }

    /// ROM contents.
    pub fn rom(&self) -> &[u8; 256] {
        &self.rom
    }

    /// Back to the reset state.
    pub fn reset(&mut self) {
        self.state = ArchState::reset();
        self.cycles = 0;
    }

    /// Advance one clock cycle (one rising edge).
    pub fn step(&mut self) -> StepInfo {
        self.cycles += 1;
        let s = &mut self.state;
        let mut info = StepInfo { executed: None, out: None };
        // The strobe flip-flop is loaded every cycle.
        s.out_strobe = false;
        if s.halted {
            return info;
        }
        let rom_d = self.rom[usize::from(s.pc)];
        let pc_inc = s.pc.wrapping_add(1);
        if !s.execute {
            s.ir = rom_d;
            s.pc = pc_inc;
            s.execute = true;
            return info;
        }
        let instr = Instr::decode(s.ir, rom_d);
        info.executed = Some(instr);
        s.execute = false;
        s.pc = pc_inc;
        let regs = s.regs;
        let r = |reg: Reg| regs[reg.index()];
        match instr {
            Instr::Nop => {}
            Instr::Ldi(d, i) => s.regs[d.index()] = i,
            Instr::Mov(d, x) => s.regs[d.index()] = r(x),
            Instr::Add(d, x) => {
                let v = arith(s, r(d), r(x), false);
                s.regs[d.index()] = v;
            }
            Instr::Addi(d, i) => {
                let v = arith(s, r(d), i, false);
                s.regs[d.index()] = v;
            }
            Instr::Sub(d, x) => {
                let v = arith(s, r(d), r(x), true);
                s.regs[d.index()] = v;
            }
            Instr::Subi(d, i) => {
                let v = arith(s, r(d), i, true);
                s.regs[d.index()] = v;
            }
            Instr::Cmp(d, x) => {
                arith(s, r(d), r(x), true);
            }
            Instr::Cmpi(d, i) => {
                arith(s, r(d), i, true);
            }
            Instr::And(d, x) => {
                let v = r(d) & r(x);
                logic(s, d, v);
            }
            Instr::Or(d, x) => {
                let v = r(d) | r(x);
                logic(s, d, v);
            }
            Instr::Xor(d, x) => {
                let v = r(d) ^ r(x);
                logic(s, d, v);
            }
            Instr::Out(x) => {
                s.out = r(x);
                s.out_strobe = true;
                info.out = Some(s.out);
            }
            Instr::Jmp(a) => s.pc = a,
            Instr::Jcc(c, a) => {
                if c.holds(s.z, s.c) {
                    s.pc = a;
                }
            }
            Instr::Halt => s.halted = true,
        }
        info
    }

    /// Run until `HALT` or until `max_cycles` edges have elapsed; return the values written to
    /// the output port, in order.
    pub fn run(&mut self, max_cycles: u64) -> Vec<u8> {
        let mut outs = Vec::new();
        for _ in 0..max_cycles {
            if self.state.halted {
                break;
            }
            if let Some(v) = self.step().out {
                outs.push(v);
            }
        }
        outs
    }

    /// The states after each of the next `cycles` edges (the state *before* the first step is not
    /// included).
    pub fn trace(&mut self, cycles: u64) -> Vec<ArchState> {
        (0..cycles)
            .map(|_| {
                self.step();
                self.state
            })
            .collect()
    }
}

/// `a + b` (or `a - b` as `a + !b + 1`); sets Z and C (carry out = not borrow).
fn arith(s: &mut ArchState, a: u8, b: u8, sub: bool) -> u8 {
    let b = if sub { !b } else { b };
    let sum = u16::from(a) + u16::from(b) + u16::from(sub);
    let v = sum as u8;
    s.z = v == 0;
    s.c = sum > 0xff;
    v
}

/// Write a logic result; Z from the result, C cleared.
fn logic(s: &mut ArchState, d: Reg, v: u8) {
    s.regs[d.index()] = v;
    s.z = v == 0;
    s.c = false;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::Cond;

    fn prog(instrs: &[Instr]) -> Vec<u8> {
        instrs.iter().flat_map(|i| i.encode()).collect()
    }

    #[test]
    fn two_cycles_per_instruction() {
        let mut e = Emulator::new(&prog(&[Instr::Ldi(Reg::R1, 7), Instr::Halt]));
        e.step();
        assert_eq!(*e.state(), ArchState { pc: 1, ir: 0x14, execute: true, ..ArchState::reset() });
        e.step();
        assert_eq!(e.state().regs, [0, 7, 0, 0]);
        assert_eq!((e.state().pc, e.state().execute), (2, false));
        e.step();
        e.step();
        assert!(e.state().halted);
        assert_eq!(e.state().pc, 4);
        let frozen = *e.state();
        e.step();
        assert_eq!(*e.state(), frozen);
    }

    #[test]
    fn arithmetic_flags() {
        let mut s = ArchState::reset();
        assert_eq!(arith(&mut s, 200, 100, false), 44);
        assert!(s.c && !s.z);
        assert_eq!(arith(&mut s, 5, 5, true), 0);
        assert!(s.c && s.z, "5-5: no borrow, zero");
        assert_eq!(arith(&mut s, 4, 5, true), 0xff);
        assert!(!s.c && !s.z, "4-5 borrows");
        assert_eq!(arith(&mut s, 0, 0, false), 0);
        assert!(!s.c && s.z);
    }

    #[test]
    fn strobe_and_branches() {
        let p = prog(&[
            Instr::Ldi(Reg::R0, 3),
            Instr::Out(Reg::R0), // addr 2
            Instr::Subi(Reg::R0, 1),
            Instr::Jcc(Cond::Nz, 2),
            Instr::Halt,
        ]);
        let mut e = Emulator::new(&p);
        assert_eq!(e.run(1000), vec![3, 2, 1]);
        assert!(e.state().halted);
        // Strobe is high exactly in the cycle after each OUT execute edge.
        let mut e = Emulator::new(&p);
        let trace = e.trace(40);
        let strobes: Vec<usize> = trace.iter().enumerate().filter(|(_, s)| s.out_strobe).map(|(i, _)| i).collect();
        assert_eq!(strobes.len(), 3);
        assert!(strobes.windows(2).all(|w| w[1] - w[0] == 6), "{strobes:?}");
    }

    #[test]
    fn logic_ops_clear_carry() {
        let p = prog(&[
            Instr::Ldi(Reg::R0, 0xff),
            Instr::Addi(Reg::R0, 1), // C=1, Z=1
            Instr::Ldi(Reg::R1, 0x0f),
            Instr::Ldi(Reg::R2, 0xf0),
            Instr::And(Reg::R1, Reg::R2), // 0 -> Z=1, C=0
            Instr::Halt,
        ]);
        let mut e = Emulator::new(&p);
        e.run(100);
        let s = e.state();
        assert_eq!(s.regs, [0, 0, 0xf0, 0]);
        assert!(s.z && !s.c);
    }
}
