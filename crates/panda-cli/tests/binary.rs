//! End-to-end tests of the `panda` binary and of [`panda_cli::main_with_args`].

use std::path::{Path, PathBuf};
use std::process::Command;

fn data(name: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("examples-data").join(name).display().to_string()
}

/// A fresh scratch directory under the target dir.
fn scratch(test: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("panda-cli-tests").join(test);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Run the binary; returns (exit code, stdout, stderr).
fn panda(args: &[&str]) -> (i32, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_panda")).args(args).output().expect("spawn panda");
    (o.status.code().unwrap_or(-1), String::from_utf8(o.stdout).unwrap(), String::from_utf8(o.stderr).unwrap())
}

/// Run in-process via the library entry point.
fn lib(args: &[&str]) -> (i32, String, String) {
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let code = panda_cli::main_with_args(&args, &mut out, &mut err);
    (code, String::from_utf8(out).unwrap(), String::from_utf8(err).unwrap())
}

#[test]
fn help_version_and_usage_errors() {
    let (c, out, _) = panda(&["help"]);
    assert_eq!(c, 0);
    assert!(out.contains("panda run <design.json>"));
    assert_eq!(panda(&["--help"]).0, 0);
    let (c, out, _) = panda(&["--version"]);
    assert_eq!(c, 0);
    assert!(out.starts_with("panda "));
    let (c, _, err) = panda(&[]);
    assert_eq!(c, 2);
    assert!(err.contains("USAGE"));
    let (c, _, err) = panda(&["frobnicate"]);
    assert_eq!(c, 2);
    assert!(err.contains("unknown command 'frobnicate'"));
    assert_eq!(panda(&["run"]).0, 2);
    assert_eq!(panda(&["run", &data("adder4.json"), "--bogus"]).0, 2);
    assert_eq!(panda(&["test", &data("adder4.json")]).0, 2);
}

#[test]
fn test_command_passes_on_examples() {
    for name in panda_cli::examples::NAMES {
        let (c, out, err) = panda(&["test", &data(&format!("{name}.json")), &data(&format!("{name}.vec"))]);
        assert_eq!(c, 0, "{name}: {out}{err}");
        assert!(out.contains("PASS:"), "{out}");
    }
    let (c, out, _) = panda(&["test", &data("sr_latch.json"), &data("sr_latch.vec"), "-v"]);
    assert_eq!(c, 0);
    assert!(out.contains("ok   "));
    assert!(out.contains("Metastable"), "{out}");
}

#[test]
fn test_command_fails_with_line_numbers() {
    let dir = scratch("test_fail");
    let vec = dir.join("bad.vec");
    std::fs::write(&vec, "set A=3 B=4\nsettle\nexpect S=7\nexpect S=8 Cout=0\n").unwrap();
    let (c, out, _) = panda(&["test", &data("adder4.json"), vec.to_str().unwrap()]);
    assert_eq!(c, 1, "{out}");
    assert!(out.contains("bad.vec:4: "), "{out}");
    assert!(out.contains("S: expected 0b1000, got 0b0111"), "{out}");
    assert!(out.contains("FAIL: 2 of 3 checks passed, 1 failed"), "{out}");

    std::fs::write(&vec, "set A=3\nexpect Nope=1\nfly\n").unwrap();
    let (c, _, err) = panda(&["test", &data("adder4.json"), vec.to_str().unwrap()]);
    assert_eq!(c, 1);
    assert!(err.contains("bad.vec:2: error: unknown signal 'Nope'"), "{err}");
    assert!(err.contains("bad.vec:3: error: unknown command 'fly'"), "{err}");

    let (c, _, err) = panda(&["test", &data("adder4.json"), dir.join("missing.vec").to_str().unwrap()]);
    assert_eq!(c, 1);
    assert!(err.contains("missing.vec"), "{err}");
}

#[test]
fn run_command_settles_and_prints_watched() {
    let (c, out, err) = panda(&["run", &data("adder4.json"), "--set", "A=0x7", "--set=B=9,Cin=1", "--watch", "S,Cout", "--watch=carry[1:0]"]);
    assert_eq!(c, 0, "{err}");
    assert!(out.contains(": settled"), "{out}");
    assert!(out.contains("S          = 0b0001 (0x1 = 1)"), "{out}");
    assert!(out.contains("Cout       = 1"), "{out}");
    assert!(out.contains("carry[1:0] = 0b11 (0x3 = 3)"), "{out}");

    // Default watch = top outputs.
    let (c, out, _) = panda(&["run", &data("counter4_clocked.json"), "--ticks", "100"]);
    assert_eq!(c, 0);
    assert!(out.starts_with("tick 100: ran 100 ticks\n"), "{out}");
    assert!(out.contains("Q   = 0b0011"), "{out}");
    assert!(out.contains("clk = "), "{out}");

    // A clocked design never settles.
    let (c, out, _) = panda(&["run", &data("counter4_clocked.json"), "--max-ticks", "50"]);
    assert_eq!(c, 0);
    assert!(out.contains("not quiescent after 50 ticks"), "{out}");

    // The metastable SR latch reports a diagnostic.
    let (c, out, _) = panda(&["run", &data("sr_latch.json"), "--set", "S_n=1,R_n=1"]);
    assert_eq!(c, 0);
    assert!(out.contains("Q   = x"), "{out}");
    let (c, out, _) = panda(&["run", &data("sr_latch.json"), "--ticks", "2", "--p2-limit", "0"]);
    assert_eq!(c, 0, "{out}");

    // Bad --set / --watch are usage errors.
    let (c, _, err) = panda(&["run", &data("adder4.json"), "--set", "S=1"]);
    assert_eq!(c, 2);
    assert!(err.contains("not a top-level input"), "{err}");
    assert_eq!(panda(&["run", &data("adder4.json"), "--set", "A=99"]).0, 2);
    assert_eq!(panda(&["run", &data("adder4.json"), "--watch", "zzz"]).0, 2);
    assert_eq!(panda(&["run", &data("adder4.json"), "--ticks", "-3"]).0, 2);
}

#[test]
fn run_and_test_write_vcd() {
    let dir = scratch("vcd");
    let vcd = dir.join("c.vcd");
    let (c, _, err) = panda(&["run", &data("counter4_clocked.json"), "--ticks", "120", "--vcd", vcd.to_str().unwrap()]);
    assert_eq!(c, 0, "{err}");
    let text = std::fs::read_to_string(&vcd).unwrap();
    assert!(text.contains("$timescale 1ns $end"));
    assert!(text.contains("$var wire 4 ! Q [3:0] $end"), "{text}");
    assert!(text.contains("$var wire 1 # rst $end"), "{text}");
    assert!(text.contains("#45\n0#"), "reset release at tick 45: {text}");
    assert!(text.contains("\n#120\n"), "final timestamp: {text}");

    let vcd2 = dir.join("t.vcd");
    let (c, _, err) = panda(&[
        "test",
        &data("counter4.json"),
        &data("counter4.vec"),
        "--vcd",
        vcd2.to_str().unwrap(),
        "--trace",
        "Q,cnt_missing_ok_no",
    ]);
    assert_eq!(c, 2, "bad trace name is a usage error: {err}");
    let (c, _, err) =
        panda(&["test", &data("counter4.json"), &data("counter4.vec"), "--vcd", vcd2.to_str().unwrap(), "--trace", "Q,ff0/m"]);
    assert_eq!(c, 0, "{err}");
    let text = std::fs::read_to_string(&vcd2).unwrap();
    assert!(text.contains("$scope module ff0 $end"), "{text}");
    assert!(text.contains("$var wire 1 \" m $end"), "{text}");

    let (c, _, err) = panda(&["run", &data("adder4.json"), "--vcd", dir.join("no/such/dir.vcd").to_str().unwrap()]);
    assert_eq!(c, 1);
    assert!(err.contains("dir.vcd"), "{err}");
}

#[test]
fn info_command() {
    let (c, out, _) = panda(&["info", &data("counter4_clocked.json")]);
    assert_eq!(c, 0);
    assert!(out.contains("top:    counter4_clocked"), "{out}");
    assert!(out.contains("defs:   4"), "{out}");
    assert!(out.contains("counter4_clocked (top)"), "{out}");
    assert!(out.contains("instances: 5 (atoms: nand 4, not 1; defs: none)"), "{out}");
    assert!(out.contains("gates:     56 (and 7, nand 32, not 14, xor 3)"), "{out}");
    assert!(out.contains("sources:   2 (clock 1, power-on-reset 1)"), "{out}");
    assert!(out.contains("d_latch 8"), "{out}");
    assert!(out.contains("diagnostics: none"), "{out}");
}

#[test]
fn info_and_run_report_invalid_designs() {
    let dir = scratch("invalid");
    let bad = dir.join("bad.json");
    let mut d = panda_cli::examples::adder4();
    d.top = "missing_top".into();
    std::fs::write(&bad, d.to_json()).unwrap();
    let (c, out, _) = panda(&["info", bad.to_str().unwrap()]);
    assert_eq!(c, 1);
    assert!(out.contains("UnknownDef"), "{out}");
    let (c, _, err) = panda(&["run", bad.to_str().unwrap()]);
    assert_eq!(c, 1);
    assert!(err.contains("design has 1 error(s)"), "{err}");

    std::fs::write(&bad, "{not json").unwrap();
    let (c, _, err) = panda(&["flat", bad.to_str().unwrap()]);
    assert_eq!(c, 1);
    assert!(err.contains("invalid design JSON"), "{err}");
}

#[test]
fn flat_command() {
    let (c, out, _) = panda(&["flat", &data("adder4.json")]);
    assert_eq!(c, 0);
    assert!(out.contains("top inputs:  A[4], B[4], Cin[1]"), "{out}");
    let (c, out, _) = panda(&["flat", &data("adder4.json"), "--json"]);
    assert_eq!(c, 0);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    let n = v["num_nets"].as_u64().unwrap();
    assert_eq!(v["net_names"].as_array().unwrap().len() as u64, n);
    assert_eq!(v["gates"].as_array().unwrap().len(), 20);
    assert_eq!(v["gates"][0]["path"].as_str().unwrap(), "fa0/g0");
    assert_eq!(v["top_inputs"][0]["name"], "A");
    assert_eq!(v["sources"].as_array().unwrap().len(), 9);
}

#[test]
fn example_command_round_trips() {
    let (c, out, _) = lib(&["example"]);
    assert_eq!(c, 0);
    assert_eq!(out.lines().collect::<Vec<_>>(), panda_cli::examples::NAMES);
    let (c, out, _) = lib(&["example", "counter4"]);
    assert_eq!(c, 0);
    assert_eq!(panda_netlist::Design::from_json(&out).unwrap(), panda_cli::examples::counter4());
    assert_eq!(lib(&["example", "nope"]).0, 2);
}

#[test]
fn library_entry_point_matches_binary() {
    for args in [
        vec!["info".to_string(), data("adder4.json")],
        vec!["test".into(), data("adder4.json"), data("adder4.vec"), "-v".into()],
        vec!["run".into(), data("counter4_clocked.json"), "--ticks".into(), "77".into()],
    ] {
        let a: Vec<&str> = args.iter().map(String::as_str).collect();
        assert_eq!(lib(&a), panda(&a), "{args:?}");
    }
}
