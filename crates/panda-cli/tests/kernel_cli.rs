//! End-to-end tests of the kernel-facing commands of the `panda` binary: `run --mode fast`,
//! vector-file `mode` / `cycle` directives, `sta`, `classify`, `kernels`, `gen` and `bench`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use panda_netlist::Design;

fn data(name: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("examples-data").join(name).display().to_string()
}

/// A fresh scratch directory under the target dir.
fn scratch(test: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("panda-cli-kernel-tests").join(test);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Run the binary; returns (exit code, stdout, stderr).
fn panda(args: &[&str]) -> (i32, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_panda")).args(args).output().expect("spawn panda");
    (o.status.code().unwrap_or(-1), String::from_utf8(o.stdout).unwrap(), String::from_utf8(o.stderr).unwrap())
}

/// Run and require exit code 0; returns stdout.
fn ok(args: &[&str]) -> String {
    let (c, out, err) = panda(args);
    assert_eq!(c, 0, "panda {args:?} failed:\n{out}\n{err}");
    out
}

/// The `NAME = value` lines printed by `panda run`.
fn watched(out: &str) -> BTreeMap<String, String> {
    out.lines()
        .filter(|l| l.starts_with("  ") && l.contains(" = "))
        .map(|l| {
            let (k, v) = l.trim().split_once(" = ").unwrap();
            (k.trim().to_string(), v.trim().to_string())
        })
        .collect()
}

/// The `tick N:` of `panda run`.
fn tick(out: &str) -> u64 {
    out.lines().next().unwrap().strip_prefix("tick ").unwrap().split(':').next().unwrap().parse().unwrap()
}

/// `panda gen ARGS -o DIR/NAME.json`; returns the path.
fn gen_to(dir: &Path, name: &str, args: &[&str]) -> String {
    let path = dir.join(format!("{name}.json")).display().to_string();
    let mut a = vec!["gen"];
    a.extend_from_slice(args);
    a.extend_from_slice(&["-o", &path]);
    let (c, out, err) = panda(&a);
    assert_eq!(c, 0, "gen {args:?}: {out}{err}");
    assert!(err.contains("panda gen"), "{err}");
    path
}

#[test]
fn help_lists_every_command() {
    let out = ok(&["help"]);
    for s in [
        "panda run <design.json> [--mode timing|fast] [--backend reference|compiled]",
        "[--ticks N | --cycles N]",
        "panda sta <design.json>",
        "panda classify <design.json> [DEF]",
        "panda kernels <design.json>",
        "panda gen [NAME",
        "panda bench <design.json>",
        "'mode timing|fast'",
        "'cycle N'",
    ] {
        assert!(out.contains(s), "help lacks {s:?}:\n{out}");
    }
}

#[test]
fn run_fast_matches_timing_at_cycle_ends() {
    let d = data("counter4_clocked.json");
    let timing = ok(&["run", &d, "--cycles", "12"]);
    assert!(timing.contains("ran 12 cycles of clk"), "{timing}");
    assert!(timing.contains("mode: timing"), "{timing}");
    assert!(timing.contains("cycles/s"), "{timing}");
    for backend in ["compiled", "reference"] {
        let fast = ok(&["run", &d, "--cycles", "12", "--mode", "fast", "--backend", backend]);
        assert_eq!(tick(&fast), tick(&timing), "{fast}");
        assert_eq!(watched(&fast), watched(&timing), "{backend}:\n{fast}\nvs\n{timing}");
        assert!(fast.contains(&format!("mode: fast ({backend} backend; engine fast/{backend}; compiled in")), "{fast}");
        assert!(fast.contains("cycles of clk = "), "{fast}");
    }
    // Q after 12 cycle steps (tick 229): edges 50..210 counted = 9.
    assert_eq!(watched(&timing)["Q"], "0b1001 (0x9 = 9)");
    // --ticks counts from power-on in both modes; Fast mode ends at the end of the phase.
    let t = ok(&["run", &d, "--ticks", "100", "--mode", "fast"]);
    assert_eq!(tick(&t), 109, "{t}");
    assert!(t.contains("ran to the end of the phase containing tick 100"), "{t}");
}

#[test]
fn run_fast_on_combinational_and_latch_designs() {
    let out = ok(&["run", &data("adder4.json"), "--mode", "fast", "--set", "A=9,B=5,Cin=1"]);
    assert!(out.contains(": settled"), "{out}");
    assert_eq!(watched(&out)["S"], "0b1111 (0xf = 15)");
    assert_eq!(watched(&out)["Cout"], "0");
    assert!(!out.contains("throughput"), "no time passes on an unclocked design:\n{out}");
    let out = ok(&["run", &data("sr_latch.json"), "--mode", "fast", "--set", "S_n=0,R_n=1", "--stats"]);
    assert_eq!(watched(&out)["Q"], "1");
    assert!(out.contains("throughput"), "{out}");
    // The --watch list works inside packages in Fast mode too.
    let out = ok(&["run", &data("counter4_clocked.json"), "--mode", "fast", "--cycles", "5", "--watch", "cnt/next,rst"]);
    assert_eq!(watched(&out).keys().collect::<Vec<_>>(), ["cnt/next", "rst"]);
}

#[test]
fn run_usage_errors() {
    let d = data("counter4_clocked.json");
    for (args, msg) in [
        (vec!["run", &d, "--backend", "compiled"], "--backend needs --mode fast"),
        (vec!["run", &d, "--mode", "warp"], "--mode: expected one of timing, fast"),
        (vec!["run", &d, "--mode", "fast", "--backend", "jit"], "--backend: expected reference or compiled"),
        (vec!["run", &d, "--ticks", "5", "--cycles", "5"], "either --ticks or --cycles"),
        (vec!["run", &d, "--cycles", "5", "--clock", "nope"], "--clock: no net named 'nope'"),
        (vec!["run", &d, "--cycles", "5", "--clock", "rst"], "not driven by a running Clock"),
    ] {
        let (c, _, err) = panda(&args);
        assert_eq!(c, 2, "{args:?}: {err}");
        assert!(err.contains(msg), "{args:?}: {err}");
    }
    let (c, _, err) = panda(&["run", &data("adder4.json"), "--cycles", "3"]);
    assert_eq!(c, 2);
    assert!(err.contains("--cycles needs a design with a running Clock"), "{err}");
}

#[test]
fn run_fast_writes_phase_sampled_vcd() {
    let dir = scratch("fast_vcd");
    let vcd = dir.join("f.vcd");
    ok(&["run", &data("counter4_clocked.json"), "--mode", "fast", "--cycles", "5", "--vcd", vcd.to_str().unwrap()]);
    let text = std::fs::read_to_string(&vcd).unwrap();
    assert!(text.contains("$enddefinitions $end"), "{text}");
    assert!(text.contains("$dumpvars"), "{text}");
    // Phase ends of the clock (edges every 10 ticks) up to the last cycle end, tick 89.
    for t in ["#19", "#29", "#59", "#69", "#89"] {
        assert!(text.lines().any(|l| l == t), "missing {t}:\n{text}");
    }
    assert!(text.contains("b0010 "), "Q reaches 2:\n{text}");
}

#[test]
fn vector_files_switch_modes() {
    let d = data("counter4_clocked.json");
    let v = data("counter4_clocked_modes.vec");
    for extra in [&[][..], &["--backend", "reference"], &["--mode", "fast"]] {
        let mut args = vec!["test", d.as_str(), v.as_str(), "-v"];
        args.extend_from_slice(extra);
        let out = ok(&args);
        assert!(out.contains("PASS: 9 of 9 checks passed"), "{extra:?}:\n{out}");
    }
    // Every example passes in Fast mode with both backends (times count from power-on).
    for name in panda_cli::examples::NAMES {
        for backend in ["compiled", "reference"] {
            let out = ok(&[
                "test",
                &data(&format!("{name}.json")),
                &data(&format!("{name}.vec")),
                "--mode",
                "fast",
                "--backend",
                backend,
            ]);
            assert!(out.contains("PASS:"), "{name} {backend}: {out}");
        }
    }
}

#[test]
fn vector_cycle_and_mode_errors() {
    let dir = scratch("vec_errors");
    let vec = dir.join("bad.vec");
    std::fs::write(&vec, "cycle 2\nmode warp\nmode\n").unwrap();
    let (c, _, err) = panda(&["test", &data("adder4.json"), vec.to_str().unwrap()]);
    assert_eq!(c, 1);
    assert!(err.contains("bad.vec:1: error: 'cycle' needs a design with a running Clock"), "{err}");
    assert!(err.contains("bad.vec:2: error: unknown mode 'warp' (expected timing or fast)"), "{err}");
    assert!(err.contains("bad.vec:3: error: 'mode' needs 'timing' or 'fast'"), "{err}");

    // `cycle` in Timing mode on the plain runner, failing check with line number.
    std::fs::write(&vec, "cycle 4\nexpect Q=1\ncycle\nexpect Q=3\n").unwrap();
    let (c, out, _) = panda(&["test", &data("counter4_clocked.json"), vec.to_str().unwrap()]);
    assert_eq!(c, 1, "{out}");
    assert!(out.contains("bad.vec:4: [t=89] Q: expected 0b0011, got 0b0010"), "{out}");

    // Settling a clocked design fails in Timing mode, passes (inputs applied) in Fast mode.
    std::fs::write(&vec, "settle 50\n").unwrap();
    let (c, out, _) = panda(&["test", &data("counter4_clocked.json"), vec.to_str().unwrap()]);
    assert_eq!(c, 1, "{out}");
    let out = ok(&["test", &data("counter4_clocked.json"), vec.to_str().unwrap(), "--mode", "fast"]);
    assert!(out.contains("PASS: 0 of 0"), "{out}");
}

#[test]
fn vector_vcd_through_the_kernel() {
    let dir = scratch("vec_vcd");
    let vcd = dir.join("t.vcd");
    let out = ok(&[
        "test",
        &data("counter4_clocked.json"),
        &data("counter4_clocked_modes.vec"),
        "--vcd",
        vcd.to_str().unwrap(),
    ]);
    assert!(out.contains("PASS"), "{out}");
    let text = std::fs::read_to_string(&vcd).unwrap();
    assert!(text.contains("$enddefinitions $end"));
    // Timing-mode stretch sampled per tick (tick 12 is mid-phase), Fast-mode stretch per phase.
    let times: Vec<u64> = text.lines().filter_map(|l| l.strip_prefix('#')?.parse().ok()).collect();
    assert!(times.windows(2).all(|w| w[0] < w[1]), "VCD time must increase: {times:?}");
    assert!(times.contains(&12) || times.contains(&13), "{times:?}");
}

#[test]
fn sta_reports_periods_paths_and_diagnostics() {
    // The example clock (period 20) is faster than STA's conservative minimum.
    let out = ok(&["sta", &data("counter4_clocked.json")]);
    for s in ["static timing analysis:", "clocks:", "  clk: period 20, min period 24, suggested 24: TOO FAST",
        "min periods:", "phase settle (3 event signature(s))", "critical path (", "result: NOT clean", "analysis time:"]
    {
        assert!(out.contains(s), "lacks {s:?}:\n{out}");
    }
    assert!(out.contains("clk↑"), "{out}");
    let dir = scratch("sta");
    let cnt = gen_to(&dir, "cnt", &["counter", "--bits", "4"]);
    let out = ok(&["sta", &cnt, "--strict"]);
    assert!(out.contains("  clk: period 64, min period 24, suggested 24: ok"), "{out}");
    assert!(out.contains("result: clean"), "{out}");
    let out = ok(&["sta", &data("adder4.json")]);
    assert!(out.contains("clocks: none"), "{out}");
    assert!(out.contains("result: clean"), "{out}");

    // An oscillator is not Fast-mode clean: reported, and --strict turns it into exit code 1.
    let ro = gen_to(&dir, "ro", &["ring-osc", "--enable"]);
    let out = ok(&["sta", &ro]);
    assert!(out.contains("Oscillating"), "{out}");
    assert!(out.contains("result: NOT clean"), "{out}");
    let (c, _, _) = panda(&["sta", &ro, "--strict"]);
    assert_eq!(c, 1);
    // A clock below its minimum period.
    let fast = gen_to(&dir, "fast", &["counter", "--bits", "4", "--period", "8"]);
    let (c, out, _) = panda(&["sta", &fast, "--strict"]);
    assert_eq!(c, 1, "{out}");
    assert!(out.contains("TOO FAST"), "{out}");
}

#[test]
fn classify_every_def_or_one() {
    let out = ok(&["classify", &data("counter4.json")]);
    assert!(out.contains("d_latch: sequential"), "{out}");
    assert!(out.contains("dff: sequential"), "{out}");
    assert!(out.contains("  clock-like pins: clk"), "{out}");
    assert!(out.contains("  edge-triggered: clk rising"), "{out}");
    assert!(out.contains("counter4 (top): sequential"), "{out}");
    assert!(out.contains("cyclic: yes; state bits: 4; kernels: 4"), "{out}");

    let out = ok(&["classify", &data("adder4.json"), "full_adder"]);
    assert!(out.starts_with("full_adder: combinational"), "{out}");
    assert!(out.contains("cyclic: no; state bits: 0; kernels: 0"), "{out}");
    assert!(!out.contains("adder4"), "only the requested def:\n{out}");

    let dir = scratch("classify");
    let ro = gen_to(&dir, "ro", &["ring-osc", "--stages", "3"]);
    let out = ok(&["classify", &ro]);
    assert!(out.contains("ring_osc_3 (top): oscillator"), "{out}");
    assert!(out.contains("osc"), "{out}");
    // Grouped vs verbose kernel rows.
    let out = ok(&["classify", &data("counter4.json"), "counter4"]);
    assert!(out.contains("(e.g.)"), "{out}");
    let out = ok(&["classify", &data("counter4.json"), "counter4", "-v"]);
    assert!(!out.contains("(e.g.)"), "{out}");

    let (c, _, err) = panda(&["classify", &data("adder4.json"), "nope"]);
    assert_eq!(c, 2);
    assert!(err.contains("unknown def 'nope'"), "{err}");
    assert_eq!(panda(&["classify"]).0, 2);
}

#[test]
fn kernels_statistics() {
    let d = data("counter4.json");
    let out = ok(&["kernels", &d]);
    assert!(out.contains("  kernels:    4 (table 4, sim 0, keeper 0)"), "{out}");
    assert!(out.contains("size histogram"), "{out}");
    assert!(out.contains("tables:     4 kernels with a table, 2 distinct (2 shared)"), "{out}");
    assert!(!out.contains("location"), "no list without --list:\n{out}");
    let out = ok(&["kernels", &d, "--list", "--program"]);
    assert!(out.contains("location"), "{out}");
    assert_eq!(out.lines().filter(|l| l.contains("  table  ")).count(), 4, "{out}");
    assert!(out.contains("FastSim program (compiled in"), "{out}");
    assert!(out.contains("phase specialization:"), "{out}");
    let out = ok(&["kernels", &d, "--limit", "3"]);
    assert!(out.contains("... 1 more (use --limit N)"), "{out}");
    let out = ok(&["kernels", &data("adder4.json")]);
    assert!(out.contains("kernels:    0"), "{out}");
}

#[test]
fn gen_blocks_and_testbenches() {
    let out = ok(&["gen"]);
    for g in ["counter", "regfile", "rom", "ring-osc", "sr-latch", "cpu8"] {
        assert!(out.lines().any(|l| l.starts_with(g)), "{g} not listed:\n{out}");
    }
    assert_eq!(ok(&["gen", "list"]), out);

    // JSON on stdout, loadable; clocked blocks get a testbench.
    let json = ok(&["gen", "counter", "--bits", "3", "--period", "32"]);
    let d = Design::from_json(&json).unwrap();
    assert_eq!(d.top, "tb_sync_counter_3_p32_h16_ph16_r72");
    let json = ok(&["gen", "counter", "--bits", "3", "--bare"]);
    assert_eq!(Design::from_json(&json).unwrap().top, "sync_counter_3");
    let json = ok(&["gen", "adder", "--bits", "4", "--flat"]);
    let d = Design::from_json(&json).unwrap();
    assert_eq!(d.defs.len(), 1, "--flat inlines");

    let dir = scratch("gen");
    // Every generator produces a valid design that simulates in both modes.
    let hex = dir.join("rom.hex");
    std::fs::write(&hex, "@0 3e 01 # comment\n@4 ff\n").unwrap();
    let hex = hex.to_str().unwrap();
    for (name, args) in [
        ("counter", &["counter", "--style", "ripple", "--bits", "3"][..]),
        ("register", &["register", "--bits", "4", "--gated"]),
        ("regfile", &["regfile", "--words", "4", "--bits", "4"]),
        ("rom", &["rom", hex]),
        ("ringosc", &["ring-osc", "--enable"]),
        ("srnor", &["sr-latch", "--nor"]),
        ("dlatch", &["d-latch", "--clr"]),
        ("dff", &["dff"]),
        ("dffrst", &["dff-rst"]),
        ("dffarst", &["dff-arst"]),
        ("dffen", &["dff-en"]),
        ("tff", &["toggle-ff"]),
        ("ha", &["half-adder"]),
        ("fa", &["full-adder"]),
        ("inc", &["incrementer", "--bits", "4"]),
        ("mux", &["mux", "--sel", "2", "--bits", "2"]),
        ("dec", &["decoder", "--bits", "2"]),
        ("eq", &["equality", "--bits", "4"]),
        ("bus", &["bus-port", "--bits", "2"]),
        ("xcvr", &["transceiver", "--bits", "2"]),
    ] {
        let p = gen_to(&dir, name, args);
        ok(&["info", &p]);
        ok(&["run", &p, "--ticks", "300"]);
        ok(&["run", &p, "--ticks", "300", "--mode", "fast"]);
    }
    // The ROM reads its image.
    let rom = gen_to(&dir, "rom2", &["rom", "--hex", hex, "--data-bits", "8"]);
    let out = ok(&["run", &rom, "--set", "A=1"]);
    assert_eq!(watched(&out)["D"], "0b00000001 (0x1 = 1)");
    let out = ok(&["run", &rom, "--set", "A=4", "--mode", "fast"]);
    assert_eq!(watched(&out)["D"], "0b11111111 (0xff = 255)");

    // A generated counter testbench counts in both modes.
    let c = gen_to(&dir, "cnt", &["counter", "--bits", "4"]);
    let t = ok(&["run", &c, "--set", "EN=1", "--cycles", "10"]);
    let f = ok(&["run", &c, "--set", "EN=1", "--cycles", "10", "--mode", "fast"]);
    assert_eq!(watched(&t), watched(&f));
    // Reset released at 144; edges at 160, 224, ..., 544 (cycle end 607): 7 counted.
    assert_eq!(watched(&t)["Q"], "0b0111 (0x7 = 7)");
}

#[test]
fn gen_errors() {
    let dir = scratch("gen_errors");
    for (args, msg) in [
        (vec!["gen", "flux"], "unknown generator 'flux'"),
        (vec!["gen", "counter", "--style", "johnson"], "--style: expected one of sync, ripple"),
        (vec!["gen", "ring-osc", "--stages", "4"], "--stages must be odd"),
        (vec!["gen", "counter", "extra"], "takes no further arguments"),
        (vec!["gen", "rom"], "needs exactly one hex file"),
        (vec!["gen", "cpu8", "--program", "doom"], "unknown program 'doom'"),
        (vec!["gen", "cpu8", "--program", "fib", "--asm", "x.s"], "at most one of"),
        (vec!["gen", "cpu8", "--period", "7"], "--period must be even"),
        (vec!["gen", "counter", "--period", "2"], "--period must be ≥ 4"),
    ] {
        let (c, _, err) = panda(&args);
        assert_eq!(c, 2, "{args:?}: {err}");
        assert!(err.contains(msg), "{args:?}: {err}");
    }
    let asm = dir.join("bad.s");
    std::fs::write(&asm, "LDI r0, 1\nFROB r1\n").unwrap();
    let (c, _, err) = panda(&["gen", "cpu8", "--asm", asm.to_str().unwrap()]);
    assert_eq!(c, 1);
    assert!(err.contains("bad.s:2:"), "{err}");
    let (c, _, err) = panda(&["gen", "rom", dir.join("missing.hex").to_str().unwrap()]);
    assert_eq!(c, 1);
    assert!(err.contains("missing.hex"), "{err}");
}

#[test]
fn gen_cpu8_runs_programs_in_both_modes() {
    let dir = scratch("cpu8");
    // A user program through the assembler: outputs 3, 2, 1, then halts.
    let asm = dir.join("count3.s");
    std::fs::write(&asm, "LDI r0, 3\nloop: OUT r0\nDEC r0\nJNZ loop\nHALT\n").unwrap();
    let user = gen_to(&dir, "user", &["cpu8", "--asm", asm.to_str().unwrap(), "--rom", "decoder"]);
    // The same program as a hex image.
    let image = panda_cpu8::asm::assemble("LDI r0, 3\nloop: OUT r0\nDEC r0\nJNZ loop\nHALT\n").unwrap();
    let hex = dir.join("count3.hex");
    std::fs::write(&hex, image.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" ")).unwrap();
    let img = gen_to(&dir, "img", &["cpu8", "--image", hex.to_str().unwrap(), "--rom", "decoder"]);
    for p in [&user, &img] {
        let t = ok(&["run", p, "--cycles", "40", "--watch", "OUT,HALTED"]);
        let f = ok(&["run", p, "--cycles", "40", "--watch", "OUT,HALTED", "--mode", "fast"]);
        assert_eq!(watched(&t), watched(&f), "{t}\n{f}");
        assert_eq!(watched(&t)["OUT"], "0b00000001 (0x1 = 1)", "{t}");
        assert_eq!(watched(&t)["HALTED"], "1", "{t}");
    }

    // Built-in programs, variants and ROM styles.
    let cd = gen_to(&dir, "countdown", &["cpu8", "--program", "countdown", "--variant", "flat"]);
    let t = ok(&["run", &cd, "--cycles", "120", "--watch", "OUT,HALTED,PC"]);
    for backend in ["compiled", "reference"] {
        let f = ok(&["run", &cd, "--cycles", "120", "--watch", "OUT,HALTED,PC", "--mode", "fast", "--backend", backend]);
        assert_eq!(watched(&t), watched(&f), "{backend}: {t}\n{f}");
    }
    assert_eq!(watched(&t)["OUT"], "0b00000000 (0x0 = 0)", "{t}");
    assert_eq!(watched(&t)["HALTED"], "1", "{t}");
    let g = gen_to(&dir, "gated", &["cpu8", "--program", "multiply", "--variant", "gated", "--rom", "mux", "--period", "160"]);
    let d = Design::from_json(&std::fs::read_to_string(&g).unwrap()).unwrap();
    assert!(d.defs.iter().any(|d| d.id.contains("gated")), "gated regfile variant");
    let out = ok(&["run", &g, "--cycles", "30", "--mode", "fast", "--watch", "PC"]);
    assert!(out.contains("ran 30 cycles of clk"), "{out}");
}

#[test]
fn bench_compares_engines() {
    let d = data("counter4_clocked.json");
    let out = ok(&["bench", &d, "--cycles", "300", "--engines", "timing,fastref,fastsim,kernel"]);
    for s in ["bench: ", "kernels: 4 (table 4", "  timing ", "  fastref ", "  fastsim ", "  kernel ", "speedup vs timing:",
        "outputs agree at tick 5989 across 4 engines"]
    {
        assert!(out.contains(s), "lacks {s:?}:\n{out}");
    }
    assert_eq!(out.matches("300 cycles in").count(), 4, "{out}");
    assert!(out.contains("gate evals/cycle"), "{out}");
    assert!(out.contains("ops, "), "{out}");

    let out = ok(&["bench", &d, "--seconds", "0.05", "--engines", "fastsim"]);
    assert!(out.contains("50.0 ms per engine"), "{out}");
    assert!(out.contains("cycles/s"), "{out}");
    assert!(!out.contains("speedup"), "{out}");

    for (args, msg) in [
        (vec!["bench", data("adder4.json").leak() as &str], "needs a design with a running Clock"),
        (vec!["bench", &d, "--engines", "timing,warp"], "unknown engine 'warp'"),
        (vec!["bench", &d, "--cycles", "5", "--seconds", "1"], "either --cycles or --seconds"),
        (vec!["bench", &d, "--seconds=0"], "expected a positive number"),
        (vec!["bench", &d, "--cycles", "0"], "--cycles must be ≥ 1"),
    ] {
        let (c, _, err) = panda(&args);
        assert_eq!(c, 2, "{args:?}: {err}");
        assert!(err.contains(msg), "{args:?}: {err}");
    }
}

#[test]
fn library_entry_point_generates_designs() {
    let args: Vec<String> = ["counter", "--bits", "2", "--bare"].iter().map(|s| s.to_string()).collect();
    let d = panda_cli::generators::generate_design(&args).unwrap();
    assert_eq!(d.top, "sync_counter_2");
    let args: Vec<String> = ["cpu8", "--program", "bench", "--rom", "decoder"].iter().map(|s| s.to_string()).collect();
    let d = panda_cli::generators::generate_design(&args).unwrap();
    assert_eq!(d.top, "panda8");
    assert!(panda_cli::generators::generate_design(&[]).is_err());
}
