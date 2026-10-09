//! `panda gen`: generate designs (JSON on stdout) from `panda_circuits` blocks and the Panda-8 CPU.
//!
//! Sequential blocks with a `CLK` pin are wrapped in a testbench by default
//! ([`panda_circuits::testbench_with`]): `CLK` is driven by a `Clock` (net `clk`), `RST` by a
//! `PowerOnReset` (net `rst`), every other pin becomes a top-level pin of the same name. `--bare`
//! emits the block itself as the top def. `--flat` inlines the hierarchy into one def.

use std::io::Write;
use std::path::Path;

use panda_circuits::{RegFileStyle, TestbenchOptions};
use panda_cpu8::{Options as CpuOptions, RomStyle, Variant};
use panda_netlist::{DefRef, Design, DesignBuilder, Dir};

use crate::args::Args;
use crate::commands::{io_err, out_err};
use crate::{CliError, read_file};

/// Generator names with a one-line description (for `panda gen` / `panda gen list`).
pub const GENERATORS: &[(&str, &str)] = &[
    ("counter", "n-bit counter: --bits N (4), --style sync|ripple (sync); pins CLK RST [EN] -> Q"),
    ("register", "n-bit register with load enable and sync reset: --bits N (8), --gated (gated clock)"),
    ("regfile", "register file: --words N (4), --bits W (8), --style mux|gated (mux)"),
    ("rom", "ROM from a hex image: rom FILE (or --hex FILE), --addr-bits A, --data-bits D (auto)"),
    ("ring-osc", "ring oscillator: --stages N (5, odd), --enable (adds an EN input)"),
    ("sr-latch", "SR latch: NAND (S_n, R_n) or --nor (S, R)"),
    ("d-latch", "gated D latch (D, EN); --clr adds an asynchronous CLR"),
    ("dff", "master-slave D flip-flop (D, CLK)"),
    ("dff-rst", "D flip-flop with synchronous reset (D, RST, CLK)"),
    ("dff-arst", "D flip-flop with asynchronous clear (D, CLR, CLK)"),
    ("dff-en", "D flip-flop with enable (D, EN, CLK)"),
    ("toggle-ff", "toggle flip-flop (T, RST, CLK)"),
    ("half-adder", "half adder (A, B -> S, C)"),
    ("full-adder", "full adder (A, B, CIN -> S, COUT)"),
    ("adder", "ripple-carry adder: --bits N (8)"),
    ("incrementer", "incrementer: --bits N (8)"),
    ("mux", "multiplexer: --sel S (1), --bits W (8)"),
    ("decoder", "decoder with enable: --bits N (3)"),
    ("equality", "equality comparator: --bits N (8)"),
    ("bus-port", "tri-state bus port: --bits W (8)"),
    ("transceiver", "bidirectional bus transceiver: --bits W (8)"),
    (
        "cpu8",
        "Panda-8 CPU: --program fib|countdown|multiply|bench (fib) or --asm FILE or --image FILE (hex), \
         --variant hierarchical|flat|gated, --rom sop|mux|decoder, --period P (128), --reset-cycles N (2)",
    ),
];

/// Default testbench clock period (ticks). With the default sizes, every clocked block meets its
/// STA minimum period at this period (the largest is a 6-bit ripple counter at 52; wider ripple
/// counters need `--period`).
pub const DEFAULT_TB_PERIOD: u32 = 64;

const VALUED: &[&str] = &[
    "bits", "words", "style", "hex", "addr-bits", "data-bits", "stages", "sel", "period", "high", "phase",
    "reset-ticks", "program", "asm", "image", "variant", "rom", "reset-cycles", "output",
];
const FLAGS: &[&str] = &["gated", "enable", "nor", "clr", "bare", "flat"];

/// `panda gen`.
pub(crate) fn cmd_gen(args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> Result<i32, CliError> {
    let a = Args::parse(args, VALUED, FLAGS, &[("o", "output")])?;
    let (name, rest) = match a.positional.split_first() {
        None => return list(out),
        Some((n, _)) if n == "list" => return list(out),
        Some((n, rest)) => (n.as_str(), rest),
    };
    let design = generate(name, rest, &a)?;
    let design = if a.flag("flat") { panda_circuits::inline_all(&design) } else { design };
    let json = design.to_json();
    match a.last("output") {
        Some(p) => std::fs::write(p, json + "\n").map_err(io_err(p))?,
        None => writeln!(out, "{json}").map_err(out_err)?,
    }
    writeln!(err, "panda gen {name}: top '{}', {} def(s)", design.top, design.defs.len()).map_err(out_err)?;
    Ok(0)
}

fn list(out: &mut dyn Write) -> Result<i32, CliError> {
    let w = GENERATORS.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
    for (n, d) in GENERATORS {
        writeln!(out, "{n:<w$}  {d}").map_err(out_err)?;
    }
    Ok(0)
}

fn opt_u32(a: &Args, name: &str, default: u32) -> Result<u32, CliError> {
    Ok(a.u32(name)?.unwrap_or(default))
}

fn check(cond: bool, msg: impl FnOnce() -> String) -> Result<(), CliError> {
    if cond { Ok(()) } else { Err(CliError::Usage(msg())) }
}

fn no_extra(name: &str, rest: &[String]) -> Result<(), CliError> {
    check(rest.is_empty(), || format!("'panda gen {name}' takes no further arguments, got '{}'", rest.join(" ")))
}

/// Build the design described by `panda gen` arguments (`args` = generator name and options,
/// e.g. `["counter", "--bits", "8"]`), including `--flat`.
pub fn generate_design(args: &[String]) -> Result<Design, CliError> {
    let a = Args::parse(args, VALUED, FLAGS, &[("o", "output")])?;
    let (name, rest) = a
        .positional
        .split_first()
        .ok_or_else(|| CliError::Usage("'panda gen' needs a generator name".into()))?;
    let design = generate(name, rest, &a)?;
    Ok(if a.flag("flat") { panda_circuits::inline_all(&design) } else { design })
}

/// Build the design of generator `name`.
pub(crate) fn generate(name: &str, rest: &[String], a: &Args) -> Result<Design, CliError> {
    if name == "cpu8" {
        no_extra(name, rest)?;
        return cpu8(a);
    }
    if name != "rom" {
        no_extra(name, rest)?;
    }
    let mut db = DesignBuilder::new();
    let bits = |default: u32| -> Result<u32, CliError> {
        let b = opt_u32(a, "bits", default)?;
        check(b >= 1, || "--bits must be ≥ 1".into())?;
        Ok(b)
    };
    let dut: DefRef = match name {
        "counter" => {
            let n = bits(4)?;
            match a.choice("style", &["sync", "ripple"])?.unwrap_or("sync") {
                "sync" => panda_circuits::sync_counter(&mut db, n),
                _ => panda_circuits::ripple_counter(&mut db, n),
            }
        }
        "register" => {
            let n = bits(8)?;
            if a.flag("gated") { panda_circuits::register_gated(&mut db, n) } else { panda_circuits::register(&mut db, n) }
        }
        "regfile" => {
            let words = opt_u32(a, "words", 4)? as usize;
            check(words >= 2, || "--words must be ≥ 2".into())?;
            let style = match a.choice("style", &["mux", "gated"])?.unwrap_or("mux") {
                "mux" => RegFileStyle::EnableMux,
                _ => RegFileStyle::GatedClock,
            };
            panda_circuits::register_file(&mut db, words, bits(8)?, style)
        }
        "rom" => rom(&mut db, rest, a)?,
        "ring-osc" => {
            let n = opt_u32(a, "stages", 5)?;
            check(n >= 3 && n % 2 == 1, || "--stages must be odd and ≥ 3".into())?;
            panda_circuits::ring_oscillator(&mut db, n, a.flag("enable"))
        }
        "sr-latch" => {
            if a.flag("nor") { panda_circuits::nor_sr_latch(&mut db) } else { panda_circuits::nand_sr_latch(&mut db) }
        }
        "d-latch" => {
            if a.flag("clr") { panda_circuits::d_latch_clr(&mut db) } else { panda_circuits::d_latch(&mut db) }
        }
        "dff" => panda_circuits::ms_dff(&mut db),
        "dff-rst" => panda_circuits::dff_rst(&mut db),
        "dff-arst" => panda_circuits::dff_arst(&mut db),
        "dff-en" => panda_circuits::dff_en(&mut db),
        "toggle-ff" => panda_circuits::toggle_ff(&mut db),
        "half-adder" => panda_circuits::half_adder(&mut db),
        "full-adder" => panda_circuits::full_adder(&mut db),
        "adder" => panda_circuits::ripple_adder(&mut db, bits(8)?),
        "incrementer" => panda_circuits::incrementer(&mut db, bits(8)?),
        "mux" => {
            let sel = opt_u32(a, "sel", 1)?;
            check((1..=8).contains(&sel), || "--sel must be in 1..=8".into())?;
            panda_circuits::mux(&mut db, sel, bits(8)?)
        }
        "decoder" => {
            let n = bits(3)?;
            check(n <= 10, || "--bits must be ≤ 10 for a decoder".into())?;
            panda_circuits::decoder(&mut db, n)
        }
        "equality" => panda_circuits::equality(&mut db, bits(8)?),
        "bus-port" => panda_circuits::bus_port(&mut db, bits(8)?),
        "transceiver" => panda_circuits::transceiver(&mut db, bits(8)?),
        other => {
            let names: Vec<&str> = GENERATORS.iter().map(|(n, _)| *n).collect();
            return Err(CliError::Usage(format!("unknown generator '{other}' (available: {})", names.join(", "))));
        }
    };
    let clocked = dut.pins.iter().any(|p| p.dir == Dir::In && p.name == "CLK" && p.width == 1);
    let top = if clocked && !a.flag("bare") { testbench(&mut db, &dut, a)? } else { dut };
    Ok(db.build(&top))
}

fn testbench(db: &mut DesignBuilder, dut: &DefRef, a: &Args) -> Result<DefRef, CliError> {
    let period = opt_u32(a, "period", DEFAULT_TB_PERIOD)?;
    check(period >= 4, || "--period must be ≥ 4".into())?;
    let mut o = TestbenchOptions::new(period);
    if let Some(h) = a.u32("high")? {
        o.high = h;
    }
    if let Some(p) = a.u32("phase")? {
        o.phase = p;
    }
    if let Some(r) = a.u32("reset-ticks")? {
        o.reset_ticks = r;
    }
    check(o.high > 0 && o.high < o.period, || "--high must be in 1..period".into())?;
    check(o.phase < o.period, || "--phase must be < period".into())?;
    Ok(panda_circuits::testbench_with(db, dut, &o))
}

fn rom(db: &mut DesignBuilder, rest: &[String], a: &Args) -> Result<DefRef, CliError> {
    let file = match (rest, a.last("hex")) {
        ([f], None) => f.clone(),
        ([], Some(f)) => f.to_string(),
        _ => return Err(CliError::Usage("'panda gen rom' needs exactly one hex file (rom FILE or --hex FILE)".into())),
    };
    let text = read_file(Path::new(&file))?;
    let words = panda_circuits::rom_from_hex(&text).map_err(|e| CliError::Other(format!("{file}: {e}")))?;
    check(!words.is_empty(), || format!("{file}: no words"))?;
    let need_addr = panda_circuits::addr_bits(words.len()).max(1);
    let addr = opt_u32(a, "addr-bits", need_addr)?;
    check((1..=16).contains(&addr), || "--addr-bits must be in 1..=16".into())?;
    check(words.len() <= 1usize << addr, || format!("{} words do not fit in {addr} address bits", words.len()))?;
    let max = words.iter().copied().max().unwrap_or(0);
    let need_data = (u64::BITS - max.leading_zeros()).max(1);
    let data = opt_u32(a, "data-bits", need_data.clamp(8, 64))?;
    check((1..=64).contains(&data), || "--data-bits must be in 1..=64".into())?;
    check(data >= need_data, || format!("word {max:#x} does not fit in {data} data bits"))?;
    Ok(panda_circuits::rom(db, "rom", addr, data, &words))
}

fn cpu8(a: &Args) -> Result<Design, CliError> {
    let sources = ["program", "asm", "image"].iter().filter(|o| a.last(o).is_some()).count();
    check(sources <= 1, || "give at most one of --program, --asm, --image".into())?;
    let image: Vec<u8> = if let Some(f) = a.last("asm") {
        let src = read_file(Path::new(f))?;
        panda_cpu8::asm::assemble(&src).map_err(|e| CliError::Other(format!("{f}:{}: {}", e.line, e.message)))?
    } else if let Some(f) = a.last("image") {
        let text = read_file(Path::new(f))?;
        let words = panda_circuits::rom_from_hex(&text).map_err(|e| CliError::Other(format!("{f}: {e}")))?;
        words
            .iter()
            .map(|&w| u8::try_from(w).map_err(|_| CliError::Other(format!("{f}: word {w:#x} is not a byte"))))
            .collect::<Result<_, _>>()?
    } else {
        let name = a.last("program").unwrap_or("fib");
        let canonical = match name {
            "fib" => "fibonacci",
            "bench" => "bench_loop",
            n => n,
        };
        let names: Vec<&str> = panda_cpu8::programs::ALL.iter().map(|p| p.name).collect();
        panda_cpu8::programs::by_name(canonical)
            .ok_or_else(|| CliError::Usage(format!("--program: unknown program '{name}' (available: {})", names.join(", "))))?
            .image()
    };
    check(image.len() <= 256, || format!("program of {} bytes does not fit the 256-byte ROM", image.len()))?;
    let variant = match a.choice("variant", &["hierarchical", "hier", "flat", "gated"])?.unwrap_or("hierarchical") {
        "flat" => Variant::Flat,
        "gated" => Variant::GatedClockRegfile,
        _ => Variant::Hierarchical,
    };
    let rom = match a.choice("rom", &["sop", "mux", "decoder"])?.unwrap_or("sop") {
        "mux" => RomStyle::MuxTree,
        "decoder" => RomStyle::Decoder,
        _ => RomStyle::SumOfProducts,
    };
    let mut o = CpuOptions::new(variant).with_rom(rom);
    if let Some(p) = a.u32("period")? {
        check(p >= 8 && p % 2 == 0, || "--period must be even and ≥ 8".into())?;
        o.period = p;
    }
    if let Some(r) = a.u32("reset-cycles")? {
        check(r >= 1, || "--reset-cycles must be ≥ 1".into())?;
        o.reset_cycles = r;
    }
    Ok(panda_cpu8::panda8_with(&image, &o))
}
