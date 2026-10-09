//! Gate-level generator for the Panda-8 CPU (SPEC §12).
//!
//! The generated design uses only gate atoms, splitters and constants, plus one `Clock` and one
//! `PowerOnReset` in the top def. All state lives in master-slave flip-flops built from NAND
//! latches ([`panda_circuits::seq`]).
//!
//! # Hierarchy ([`Variant::Hierarchical`])
//!
//! ```text
//! panda8                        top: Clock, PowerOnReset, cpu, rom
//! ├── rom   panda8_sop_rom_…    256×8 program ROM (see RomStyle; default: constants +
//! │                             decoder_8 + AND-OR multiplexer)
//! └── cpu   panda8_core
//!     ├── ctl     panda8_control      opcode decoder_4 + control equations
//!     ├── rf      panda8_regfile_4x8  4×8 registers, decoder_2 write select, 2 read muxes
//!     ├── bsel    mux2_8              B operand: register or immediate
//!     ├── alu     panda8_alu          ripple_adder_8, AND/OR/XOR, result muxes, Z/C
//!     ├── pcinc   incrementer_8
//!     ├── pcsel   mux2_8              next PC: PC+1 or jump target
//!     └── pc, ir, phase, flags, outp, strobe, halt   register_N (sync reset, load enable)
//! ```
//!
//! # Top-level pins (all `Out`)
//!
//! | pin          | width | meaning                                        |
//! |--------------|-------|------------------------------------------------|
//! | `OUT`        | 8     | output port register                           |
//! | `OUT_STROBE` | 1     | high for the cycle after each `OUT`            |
//! | `HALTED`     | 1     | `HALT` executed                                |
//! | `PC`         | 8     | program counter (debug)                        |
//! | `IR`         | 8     | instruction register (debug)                   |
//! | `PHASE`      | 1     | 0 = fetch cycle, 1 = execute cycle (debug)     |
//! | `FLAGS`      | 2     | bit 0 = Z, bit 1 = C (debug)                   |
//! | `R0`..`R3`   | 8     | register file contents (debug)                 |
//!
//! # Clocking and reset
//!
//! The clock has period `P` ([`Options::period`]), 50 % duty, starts low and rises at
//! `P/2 + k·P`. `PowerOnReset` holds `RST = 1` for [`Options::reset_cycles`] rising edges and is
//! released a quarter period into a low phase (away from any edge); every register is reset
//! synchronously. Rising edge number `reset_cycles` is the first one that executes (a fetch).
//!
//! # Gated-clock register file ([`Variant::GatedClockRegfile`])
//!
//! Each register-file word is a [`panda_circuits::register_gated`] whose clock is
//! `AND(CLK, LOAD | RST)`. Because the write enables are produced by logic that switches right
//! after the rising edge (while `CLK` is high), each enable passes through a latch that is
//! transparent while `CLK` is low (a standard latch-based clock-gating cell, `icg0..icg3` and
//! `icg_rst`). This keeps the gated clocks glitch-free; it is the conditional-kernel workload of
//! SPEC §12 variant (c).
//!
//! Side effect worth knowing: words that are not being written keep their master latches
//! transparent, so every ALU glitch on `WD` passes through them. With the timing engine's default
//! `p2_limit = 4`, long glitch trains there are reported as `Metastable` (they recover; see
//! [`crate::sim::GateCpu::new`]).

use panda_circuits::{
    d_latch, decoder, incrementer, inline_all, mux, mux2, register, register_gated, ripple_adder, rom,
};
use panda_netlist::{DefBuilder, DefRef, Design, DesignBuilder, GateOp, Sig};

/// Default clock period in ticks.
///
/// Measured settle times (longest time from a clock edge to the last net change, over every
/// built-in program, variant and ROM style; re-measured by
/// `tests/gate_level.rs::settle_time_fits_default_period`):
///
/// | ROM style                  | after rising edge | after falling edge |
/// |----------------------------|-------------------|--------------------|
/// | [`RomStyle::SumOfProducts`]| 35                | 6                  |
/// | [`RomStyle::MuxTree`]      | 47                | 6                  |
/// | [`RomStyle::Decoder`]      | 35                | 6                  |
///
/// The rising-edge activity must finish before the next rising edge and the falling-edge activity
/// (master latches opening) before the next rising edge half a period later. A period of 128
/// ticks leaves a margin of at least 2.7× on the former and 10× on the latter.
pub const DEFAULT_PERIOD: u32 = 128;

/// Default number of rising edges during which `PowerOnReset` holds the CPU in reset.
pub const DEFAULT_RESET_CYCLES: u32 = 2;

/// Design variant (SPEC §12).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Variant {
    /// (a) Packaged hierarchically (defs for ALU, control, register file, registers, ROM, …).
    Hierarchical,
    /// (b) Fully inlined into a single def ([`panda_circuits::inline_all`]): no hierarchy hints.
    Flat,
    /// (c) Hierarchical, with a gated-clock register file.
    GatedClockRegfile,
}

impl Variant {
    /// All variants.
    pub const ALL: [Variant; 3] = [Variant::Hierarchical, Variant::Flat, Variant::GatedClockRegfile];
}

/// How the program ROM is built. All styles are combinational functions of the 8 address bits
/// built from gates and `Constant`s only, so they compile to a single table (SPEC §7.4).
///
/// Flat gate counts of the whole CPU (core: 1351 gates; 1285 with the gated register file):
///
/// | ROM style | ROM gates | CPU total (Hierarchical / Flat) | CPU total (GatedClockRegfile) |
/// |-----------|-----------|---------------------------------|-------------------------------|
/// | [`SumOfProducts`](RomStyle::SumOfProducts) (default) | 2320 | 3671 | 3605 |
/// | [`MuxTree`](RomStyle::MuxTree) | 6375 | 7726 | 7660 |
/// | [`Decoder`](RomStyle::Decoder) | program-dependent (77 for `multiply`) | 1428 | 1362 |
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum RomStyle {
    /// Complete 256-leaf mux tree over `Constant`s ([`panda_circuits::rom()`]), the literal SPEC §12
    /// construction. 255 `mux2_8` instances, 6375 gates regardless of the program — on its own
    /// more than the SPEC §12 size target for the whole CPU.
    MuxTree,
    /// Constants + one-hot sum-of-products multiplexer: an 8→256 [`decoder`], one 8-bit AND of
    /// each decoder line with that word's `Constant`, and one 256-input 8-bit OR. 2320 gates
    /// regardless of the program; with it the whole CPU lands inside the SPEC §12 size target
    /// (3–5k gates).
    #[default]
    SumOfProducts,
    /// Address decoder + OR planes: one 8-input AND per non-zero ROM byte and one wide OR per
    /// data bit (plus 8 inverters). Much smaller (≈ program length + 16 gates); the gate count
    /// depends on the program.
    Decoder,
}

/// Generator options.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Options {
    /// Design variant.
    pub variant: Variant,
    /// Program ROM construction.
    pub rom: RomStyle,
    /// Clock period in ticks (even, ≥ 8).
    pub period: u32,
    /// Rising edges held in reset (≥ 1).
    pub reset_cycles: u32,
}

impl Options {
    /// Defaults ([`DEFAULT_PERIOD`], [`DEFAULT_RESET_CYCLES`], [`RomStyle::SumOfProducts`]) for a
    /// variant.
    pub fn new(variant: Variant) -> Self {
        Options { variant, rom: RomStyle::default(), period: DEFAULT_PERIOD, reset_cycles: DEFAULT_RESET_CYCLES }
    }

    /// Same options with another clock period.
    pub fn with_period(mut self, period: u32) -> Self {
        self.period = period;
        self
    }

    /// Same options with another ROM style.
    pub fn with_rom(mut self, rom: RomStyle) -> Self {
        self.rom = rom;
        self
    }

    /// Tick at which `PowerOnReset` releases (a quarter period into a low clock phase).
    pub fn reset_ticks(&self) -> u32 {
        self.reset_cycles * self.period + self.period / 4
    }
}

impl Default for Options {
    fn default() -> Self {
        Options::new(Variant::Hierarchical)
    }
}

/// Generate the Panda-8 design running `program` (≤ 256 bytes), with default options.
pub fn panda8(program: &[u8], variant: Variant) -> Design {
    panda8_with(program, &Options::new(variant))
}

/// Generate the Panda-8 design running `program` (≤ 256 bytes).
///
/// Panics if the program is longer than 256 bytes, the period is odd or below 8, or
/// `reset_cycles` is 0.
pub fn panda8_with(program: &[u8], opts: &Options) -> Design {
    assert!(program.len() <= 256, "panda8: program of {} bytes does not fit the 256-byte ROM", program.len());
    assert!(opts.period >= 8 && opts.period.is_multiple_of(2), "panda8: period must be even and ≥ 8");
    assert!(opts.reset_cycles >= 1, "panda8: reset_cycles must be ≥ 1");
    let mut db = DesignBuilder::new();
    let gated = opts.variant == Variant::GatedClockRegfile;
    let core = core(&mut db, gated);
    let rom = match opts.rom {
        RomStyle::MuxTree => {
            let words: Vec<u64> = program.iter().map(|&b| u64::from(b)).collect();
            rom(&mut db, "panda8_prog", 8, 8, &words)
        }
        RomStyle::SumOfProducts => sop_rom(&mut db, program),
        RomStyle::Decoder => decoder_rom(&mut db, program),
    };
    let top = top(&mut db, &core, &rom, opts);
    let design = db.build(&top);
    match opts.variant {
        Variant::Flat => inline_all(&design),
        Variant::Hierarchical | Variant::GatedClockRegfile => design,
    }
}

/// Replicate a 1-bit signal to `w` bits.
fn bcast(b: &mut DefBuilder, s: Sig, w: u32) -> Sig {
    b.concat(&vec![s; w as usize])
}

/// Top def `panda8`: sources, ROM and CPU core.
fn top(db: &mut DesignBuilder, core: &DefRef, rom: &DefRef, opts: &Options) -> DefRef {
    let p = opts.period;
    let mut b = DefBuilder::new("panda8");
    b.set_name("Panda-8");
    let clk = b.clock(p, p / 2, p / 2);
    b.name(clk, "clk");
    let rst = b.power_on_reset(opts.reset_ticks());
    b.name(rst, "rst");
    let rom_a = b.wire(8);
    b.name(rom_a, "rom_a");
    let rom_d = b.inst_named("rom", rom, &[rom_a])[0];
    b.name(rom_d, "rom_d");
    let o = b.inst_map("cpu", core, &[("CLK", clk), ("RST", rst), ("ROM_D", rom_d), ("ROM_A", rom_a)]);
    for pin in ["OUT", "OUT_STROBE", "HALTED", "PC", "IR", "PHASE", "FLAGS", "R0", "R1", "R2", "R3"] {
        b.output(pin, o[pin]);
    }
    db.add(b.finish())
}

/// CPU core. Id `panda8_core` / `panda8_core_gated`.
///
/// Pins: `CLK`, `RST`, `ROM_D` (in, 8) → `ROM_A` (8), `OUT` (8), `OUT_STROBE`, `HALTED`, `PC` (8),
/// `IR` (8), `PHASE`, `FLAGS` (2), `R0`..`R3` (8).
fn core(db: &mut DesignBuilder, gated: bool) -> DefRef {
    let id = if gated { "panda8_core_gated" } else { "panda8_core" };
    if let Some(r) = db.get(id) {
        return r;
    }
    let reg8 = register(db, 8);
    let reg2 = register(db, 2);
    let reg1 = register(db, 1);
    let inc = incrementer(db, 8);
    let m2 = mux2(db, 8);
    let alu = alu(db);
    let ctl = control(db);
    let rf = regfile(db, gated);

    let mut b = DefBuilder::new(id);
    let clk = b.input("CLK", 1);
    let rst = b.input("RST", 1);
    let rom_d = b.input("ROM_D", 8);
    let pc = b.wire(8);
    let ir = b.wire(8);
    let phase = b.wire(1);
    let halted = b.wire(1);
    let flags = b.wire(2);
    let one = b.lit(1, 1);

    let c = b.inst_map("ctl", &ctl, &[("IR", ir), ("PHASE", phase), ("HALTED", halted), ("FLAGS", flags)]);
    let rd = b.slice(ir, 2, 2);
    let rs = b.slice(ir, 0, 2);
    let alu_y = b.wire(8);
    let r = b.inst_map(
        "rf",
        &rf,
        &[("WD", alu_y), ("WA", rd), ("WE", c["WE"]), ("RA", rd), ("RB", rs), ("RST", rst), ("CLK", clk)],
    );
    let bval = b.inst_named("bsel", &m2, &[r["QB"], rom_d, c["B_IMM"]])[0];
    b.name(bval, "b");
    let a = b.inst_map(
        "alu",
        &alu,
        &[
            ("A", r["QA"]),
            ("B", bval),
            ("SUB", c["SUB"]),
            ("SEL_B", c["SEL_B"]),
            ("SEL_LOGIC", c["SEL_LOGIC"]),
            ("OP_AND", c["OP_AND"]),
            ("OP_OR", c["OP_OR"]),
            ("OP_XOR", c["OP_XOR"]),
            ("Y", alu_y),
        ],
    );
    b.name(alu_y, "y");

    // Flags register: bit 0 = Z, bit 1 = C.
    let fl_d = b.concat(&[a["ZF"], a["CF"]]);
    let q = b.inst_named("flags", &reg2, &[fl_d, c["FLAG_WE"], rst, clk])[0];
    b.connect(q, flags);
    // Program counter.
    let pc_inc = b.inst_named("pcinc", &inc, &[pc, one])[0];
    b.name(pc_inc, "pc_inc");
    let pc_next = b.inst_named("pcsel", &m2, &[pc_inc, rom_d, c["TAKEN"]])[0];
    b.name(pc_next, "pc_next");
    let q = b.inst_named("pc", &reg8, &[pc_next, c["RUN"], rst, clk])[0];
    b.connect(q, pc);
    // Instruction register, phase, output port, strobe, halt.
    let q = b.inst_named("ir", &reg8, &[rom_d, c["IR_LD"], rst, clk])[0];
    b.connect(q, ir);
    let nphase = b.not(phase);
    let q = b.inst_named("phase", &reg1, &[nphase, c["RUN"], rst, clk])[0];
    b.connect(q, phase);
    let out = b.inst_named("outp", &reg8, &[r["QB"], c["OUT_LD"], rst, clk])[0];
    let strobe = b.inst_named("strobe", &reg1, &[c["OUT_LD"], one, rst, clk])[0];
    let q = b.inst_named("halt", &reg1, &[one, c["HALT_LD"], rst, clk])[0];
    b.connect(q, halted);

    b.name(pc, "pc");
    b.name(ir, "ir");
    b.name(phase, "phase_q");
    b.name(halted, "halted");
    b.name(flags, "flags_q");
    b.output("ROM_A", pc);
    b.output("OUT", out);
    b.output("OUT_STROBE", strobe);
    b.output("HALTED", halted);
    b.output("PC", pc);
    b.output("IR", ir);
    b.output("PHASE", phase);
    b.output("FLAGS", flags);
    for i in 0..4 {
        b.output(&format!("R{i}"), r[&format!("R{i}")]);
    }
    db.add(b.finish())
}

/// Control unit. Id `panda8_control`.
///
/// Pins: `IR` (in, 8), `PHASE`, `HALTED`, `FLAGS` (in, 2) → `RUN`, `IR_LD`, `WE`, `FLAG_WE`,
/// `B_IMM`, `SUB`, `SEL_B`, `SEL_LOGIC`, `OP_AND`, `OP_OR`, `OP_XOR`, `OUT_LD`, `TAKEN`, `HALT_LD`.
///
/// The opcode nibble goes through a [`decoder`] (`opdec`, one-hot `d0..d15`); every control signal
/// is an OR of decoder lines, qualified by `exec = PHASE & !HALTED` where it causes a write.
fn control(db: &mut DesignBuilder) -> DefRef {
    use crate::isa::opcode::*;
    let dec = decoder(db, 4);
    let mut b = DefBuilder::new("panda8_control");
    let ir = b.input("IR", 8);
    let phase = b.input("PHASE", 1);
    let halted = b.input("HALTED", 1);
    let flags = b.input("FLAGS", 2);
    let op = b.slice(ir, 4, 4);
    let one = b.lit(1, 1);
    let onehot = b.inst_named("opdec", &dec, &[op, one])[0];
    b.name(onehot, "d");
    let d = b.bits(onehot);
    let line = |o: u8| d[usize::from(o)];
    let any = |b: &mut DefBuilder, ops: &[u8]| {
        let ins: Vec<Sig> = ops.iter().map(|&o| line(o)).collect();
        b.or(&ins)
    };

    let run = b.not(halted);
    b.name(run, "run");
    let nphase = b.not(phase);
    let exec = b.and(&[phase, run]);
    b.name(exec, "exec");
    let fetch = b.and(&[nphase, run]);
    b.name(fetch, "fetch");

    let writes = any(&mut b, &[LDI, MOV, ADD, SUB, AND, OR, XOR, ADDI, SUBI]);
    let we = b.and(&[exec, writes]);
    let sets_flags = any(&mut b, &[ADD, SUB, AND, OR, XOR, ADDI, SUBI, CMP, CMPI]);
    let flag_we = b.and(&[exec, sets_flags]);
    let b_imm = any(&mut b, &[LDI, ADDI, SUBI, CMPI]);
    let sub = any(&mut b, &[SUB, SUBI, CMP, CMPI]);
    let sel_b = any(&mut b, &[LDI, MOV]);
    let sel_logic = any(&mut b, &[AND, OR, XOR]);
    let out_ld = b.and(&[exec, line(OUT)]);
    let halt_ld = b.and(&[exec, line(HALT)]);

    // Branch condition: cc = IR[3:2]; flag = cc[1] ? C : Z; holds = flag ^ cc[0].
    let c0 = b.bit(ir, 2);
    let c1 = b.bit(ir, 3);
    let zf = b.bit(flags, 0);
    let cf = b.bit(flags, 1);
    let nc1 = b.not(c1);
    let tz = b.and(&[nc1, zf]);
    let tc = b.and(&[c1, cf]);
    let flag = b.or(&[tz, tc]);
    let holds = b.xor(&[flag, c0]);
    b.name(holds, "cond");
    let jcc = b.and(&[line(JCC), holds]);
    let branch = b.or(&[line(JMP), jcc]);
    let taken = b.and(&[exec, branch]);

    b.output("RUN", run);
    b.output("IR_LD", fetch);
    b.output("WE", we);
    b.output("FLAG_WE", flag_we);
    b.output("B_IMM", b_imm);
    b.output("SUB", sub);
    b.output("SEL_B", sel_b);
    b.output("SEL_LOGIC", sel_logic);
    b.output("OP_AND", line(AND));
    b.output("OP_OR", line(OR));
    b.output("OP_XOR", line(XOR));
    b.output("OUT_LD", out_ld);
    b.output("TAKEN", taken);
    b.output("HALT_LD", halt_ld);
    db.add_once(b.finish())
}

/// ALU. Id `panda8_alu`.
///
/// Pins: `A`, `B` (in, 8), `SUB`, `SEL_B`, `SEL_LOGIC`, `OP_AND`, `OP_OR`, `OP_XOR` (in, 1)
/// → `Y` (8), `CF`, `ZF`.
///
/// `sum = A + (B ^ SUB…) + SUB` (ripple adder `add`); `logic` = the selected one of AND/OR/XOR;
/// `Y = SEL_B ? B : (SEL_LOGIC ? logic : sum)`; `CF = carry & !SEL_LOGIC`; `ZF = (Y == 0)`.
fn alu(db: &mut DesignBuilder) -> DefRef {
    let add = ripple_adder(db, 8);
    let m2 = mux2(db, 8);
    let mut b = DefBuilder::new("panda8_alu");
    let a = b.input("A", 8);
    let bb = b.input("B", 8);
    let sub = b.input("SUB", 1);
    let sel_b = b.input("SEL_B", 1);
    let sel_logic = b.input("SEL_LOGIC", 1);
    let op_and = b.input("OP_AND", 1);
    let op_or = b.input("OP_OR", 1);
    let op_xor = b.input("OP_XOR", 1);

    let subw = bcast(&mut b, sub, 8);
    let b2 = b.xor(&[bb, subw]);
    b.name(b2, "b_inv");
    let s = b.inst_named("add", &add, &[a, b2, sub]);
    let (sum, cout) = (s[0], s[1]);
    b.name(sum, "sum");
    let and_w = bcast(&mut b, op_and, 8);
    let or_w = bcast(&mut b, op_or, 8);
    let xor_w = bcast(&mut b, op_xor, 8);
    let t_and = b.and(&[a, bb, and_w]);
    let ab_or = b.or(&[a, bb]);
    let t_or = b.and(&[ab_or, or_w]);
    let ab_xor = b.xor(&[a, bb]);
    let t_xor = b.and(&[ab_xor, xor_w]);
    let logic = b.or(&[t_and, t_or, t_xor]);
    b.name(logic, "logic");
    let r1 = b.inst_named("lsel", &m2, &[sum, logic, sel_logic])[0];
    let y = b.inst_named("bsel", &m2, &[r1, bb, sel_b])[0];
    let nsl = b.not(sel_logic);
    let cf = b.and(&[cout, nsl]);
    let zf = b.reduce(GateOp::Nor, y);
    b.output("Y", y);
    b.output("CF", cf);
    b.output("ZF", zf);
    db.add_once(b.finish())
}

/// 4×8 register file with one write port and two read ports.
/// Id `panda8_regfile_4x8` / `panda8_regfile_gated_4x8`.
///
/// Pins: `WD` (in, 8), `WA` (in, 2), `WE`, `RA` (in, 2), `RB` (in, 2), `RST`, `CLK`
/// → `QA` (8), `QB` (8), `R0`..`R3` (8).
///
/// At a rising edge: `RST ? all ← 0 : (WE ? word[WA] ← WD)`. `QA = word[RA]`, `QB = word[RB]`.
/// A [`decoder`] (`wdec`) produces the per-word loads; words are `w0..w3`; read muxes `rmux_a`,
/// `rmux_b`. The gated style adds the clock-gating latches described in the module docs.
fn regfile(db: &mut DesignBuilder, gated: bool) -> DefRef {
    let dec = decoder(db, 2);
    let rmux = mux(db, 2, 8);
    let latch = gated.then(|| d_latch(db));
    let (reg, id) = if gated {
        (register_gated(db, 8), "panda8_regfile_gated_4x8")
    } else {
        (register(db, 8), "panda8_regfile_4x8")
    };
    let mut b = DefBuilder::new(id);
    let wd = b.input("WD", 8);
    let wa = b.input("WA", 2);
    let we = b.input("WE", 1);
    let ra = b.input("RA", 2);
    let rb = b.input("RB", 2);
    let rst = b.input("RST", 1);
    let clk = b.input("CLK", 1);
    let sel = b.inst_named("wdec", &dec, &[wa, we])[0];
    b.name(sel, "wsel");
    let mut loads = b.bits(sel);
    let mut rst_w = rst;
    if let Some(latch) = &latch {
        let nclk = b.not(clk);
        b.name(nclk, "nclk");
        for (i, l) in loads.iter_mut().enumerate() {
            *l = b.inst_named(&format!("icg{i}"), latch, &[*l, nclk])[0];
            b.name(*l, &format!("ld{i}"));
        }
        rst_w = b.inst_named("icg_rst", latch, &[rst, nclk])[0];
        b.name(rst_w, "rst_l");
    }
    let words: Vec<Sig> = (0..4)
        .map(|i| {
            let q = b.inst_named(&format!("w{i}"), &reg, &[wd, loads[i], rst_w, clk])[0];
            b.name(q, &format!("r{i}"))
        })
        .collect();
    let mut ins_a = words.clone();
    ins_a.push(ra);
    let qa = b.inst_named("rmux_a", &rmux, &ins_a)[0];
    let mut ins_b = words.clone();
    ins_b.push(rb);
    let qb = b.inst_named("rmux_b", &rmux, &ins_b)[0];
    b.output("QA", qa);
    b.output("QB", qb);
    for (i, w) in words.iter().enumerate() {
        b.output(&format!("R{i}"), *w);
    }
    db.add_once(b.finish())
}

/// Decoder-style ROM ([`RomStyle::Decoder`]). Id `panda8_drom_{hash}` (content hash).
///
/// Pins: `A` (in, 8) → `D` (out, 8), like [`panda_circuits::rom`]. Line `l{addr}` is the 8-input
/// AND of the matching true/complemented address bits; data bit `k` is the OR of the lines of
/// every byte with bit `k` set (constant 0 when there are none).
fn decoder_rom(db: &mut DesignBuilder, program: &[u8]) -> DefRef {
    let used = program.iter().rposition(|&v| v != 0).map_or(0, |i| i + 1);
    let id = format!("panda8_drom_{:016x}", fnv1a(&program[..used]));
    if let Some(r) = db.get(&id) {
        return r;
    }
    let mut b = DefBuilder::new(&id);
    let a = b.input("A", 8);
    let na = b.not(a);
    let t = b.bits(a);
    let f = b.bits(na);
    let mut lines: Vec<(u8, Sig)> = Vec::new();
    for (addr, &v) in program.iter().enumerate().filter(|(_, v)| **v != 0) {
        let ins: Vec<Sig> = (0..8).map(|k| if (addr >> k) & 1 == 1 { t[k] } else { f[k] }).collect();
        let l = b.and(&ins);
        b.name(l, &format!("l{addr}"));
        lines.push((v, l));
    }
    let bits: Vec<Sig> = (0..8)
        .map(|k| {
            let ins: Vec<Sig> = lines.iter().filter(|(v, _)| (v >> k) & 1 == 1).map(|&(_, l)| l).collect();
            match ins.len() {
                0 => b.lit(1, 0),
                1 => ins[0],
                _ => b.or(&ins),
            }
        })
        .collect();
    let d = b.concat(&bits);
    b.output("D", d);
    db.add(b.finish())
}

/// Sum-of-products ROM ([`RomStyle::SumOfProducts`]). Id `panda8_sop_rom_{hash}`.
///
/// Pins: `A` (in, 8) → `D` (out, 8). `D = OR_i(bcast(line_i) AND word_i)` where `line_i` is output
/// `i` of a [`decoder`] (`adec`) and `word_i` a `Constant` (one per distinct value; absent words
/// are 0).
fn sop_rom(db: &mut DesignBuilder, program: &[u8]) -> DefRef {
    let used = program.iter().rposition(|&v| v != 0).map_or(0, |i| i + 1);
    let id = format!("panda8_sop_rom_{:016x}", fnv1a(&program[..used]));
    if let Some(r) = db.get(&id) {
        return r;
    }
    let dec = decoder(db, 8);
    let mut b = DefBuilder::new(&id);
    let a = b.input("A", 8);
    let one = b.lit(1, 1);
    let onehot = b.inst_named("adec", &dec, &[a, one])[0];
    b.name(onehot, "line");
    let lines = b.bits(onehot);
    let mut consts: std::collections::HashMap<u8, Sig> = std::collections::HashMap::new();
    let terms: Vec<Sig> = lines
        .iter()
        .enumerate()
        .map(|(i, &l)| {
            let v = program.get(i).copied().unwrap_or(0);
            let word = *consts.entry(v).or_insert_with(|| b.lit(8, u64::from(v)));
            let lw = bcast(&mut b, l, 8);
            b.and(&[lw, word])
        })
        .collect();
    let d = b.or(&terms);
    b.output("D", d);
    db.add(b.finish())
}

/// 64-bit FNV-1a (deterministic content hash for def ids).
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3))
}
