//! Library-level tests: signal resolution, the vector runner and the VCD writer.

use std::collections::HashMap;
use std::sync::Arc;

use panda_cli::signals::{self, Signal};
use panda_cli::vcd::{self, VcdWriter};
use panda_cli::vectors::{Command, Program};
use panda_cli::{examples, flatten_design};
use panda_netlist::{DefBuilder, Design, DesignBuilder, FlatNetlist, Logic, Value};
use panda_timing::{TimingOptions, TimingSim};

fn netlist(d: &Design) -> Arc<FlatNetlist> {
    Arc::new(flatten_design(d).unwrap().0)
}

fn sim(nl: &Arc<FlatNetlist>) -> TimingSim {
    TimingSim::new(nl.clone(), TimingOptions::default())
}

fn single(top: DefBuilder) -> Design {
    let mut db = DesignBuilder::new();
    let t = db.add(top.finish());
    db.build(&t)
}

/// 3-inverter ring oscillator with an enable: `en` → `y`.
fn ring() -> Design {
    let mut b = DefBuilder::new("ring");
    let en = b.input("en", 1);
    let back = b.wire(1);
    let a = b.nand(&[en, back]);
    let c = b.not(a);
    b.not_to(c, back);
    b.output("y", back);
    single(b)
}

/// Tri-state buffer: `d`, `en` → `y`.
fn tristate() -> Design {
    let mut b = DefBuilder::new("tristate");
    let d = b.input("d", 2);
    let en = b.input("en", 1);
    let y = b.tri(d, en);
    b.output("y", y);
    single(b)
}

// ---------------------------------------------------------------- signals

#[test]
fn resolve_names() {
    let nl = netlist(&examples::adder4());
    let s = signals::resolve(&nl, "A").unwrap();
    assert_eq!(s.nets, nl.top_inputs[0].nets);
    assert_eq!(signals::resolve(&nl, "carry").unwrap().width(), 4);
    let r = signals::resolve(&nl, "carry[3:1]").unwrap();
    assert_eq!(r.width(), 3);
    assert_eq!(r.nets[0], nl.net_by_name("carry[1]").unwrap());
    assert_eq!(signals::resolve(&nl, "carry[1:3]").unwrap().nets, r.nets, "range order is normalized");
    assert_eq!(signals::resolve(&nl, "carry[2]").unwrap().width(), 1);
    assert_eq!(signals::resolve(&nl, "fa1/axb").unwrap().width(), 1);
    assert_eq!(signals::resolve(&nl, "Cin[0:0]").unwrap().width(), 1);
    assert!(signals::resolve(&nl, "nope").is_err());
    assert!(signals::resolve(&nl, "carry[5:0]").is_err());
    let list = signals::resolve_list(&nl, "A, S,,Cout").unwrap();
    assert_eq!(list.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), ["A", "S", "Cout"]);
    assert!(signals::resolve_list(&nl, "A,bogus").is_err());
}

#[test]
fn default_selection_groups_buses_and_skips_auto_names() {
    let nl = netlist(&examples::adder4());
    let sel = signals::default_selection(&nl);
    let names: Vec<_> = sel.iter().map(|s| (s.name.as_str(), s.width())).collect();
    assert_eq!(names, [("A", 4), ("B", 4), ("Cin", 1), ("S", 4), ("Cout", 1), ("carry", 4)]);

    let nl = netlist(&examples::counter4());
    let names: Vec<_> = signals::default_selection(&nl).into_iter().map(|s| s.name).collect();
    assert_eq!(names, ["clk", "rst", "Q", "next"]);
}

// ---------------------------------------------------------------- vectors

#[test]
fn parse_errors_are_collected_with_line_numbers() {
    let nl = netlist(&examples::adder4());
    let src = "\
# comment only

set A=1 B=99          # B too wide
bogus 3
expect S=0b1-1 Q=1
set S=1
run x
settle 10 20
pulse A
pulse
expect
set A
reset now
";
    let errs = Program::parse(src, &nl).unwrap_err();
    let lines: Vec<usize> = errs.iter().map(|e| e.line).collect();
    assert_eq!(lines, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "{errs:#?}");
    assert!(errs[0].message.contains("B"), "{}", errs[0]);
    assert!(errs[2].message.contains("unknown signal 'Q'"), "{}", errs[2]);
    assert!(errs[3].message.contains("not a top-level input"), "{}", errs[3]);
    assert!(errs[6].message.contains("1-bit"), "{}", errs[6]);
    assert_eq!(errs[0].to_string(), format!("line 3: {}", errs[0].message));
}

#[test]
fn parse_valid_program() {
    let nl = netlist(&examples::counter4());
    let p = Program::parse("set clk=0 rst=1 # x\n\ntick\nrun 1_000\nsettle\npulse clk 3\nexpect Q=- next=0b----\nprint Q\nreset\n", &nl)
        .unwrap();
    assert_eq!(p.steps.len(), 8);
    assert_eq!(p.steps[1].line, 3);
    assert!(matches!(p.steps[1].command, Command::Run(1)));
    assert!(matches!(p.steps[2].command, Command::Run(1000)));
    assert!(matches!(p.steps[3].command, Command::Settle(panda_cli::vectors::DEFAULT_SETTLE_MAX)));
    assert!(matches!(&p.steps[4].command, Command::Pulse { pin, count: 3 } if pin == "clk"));
    assert_eq!(p.expect_count(), 2);
}

#[test]
fn failing_checks_are_reported() {
    let nl = netlist(&examples::adder4());
    let p = Program::parse("set A=1 B=1\nsettle\nexpect S=2 Cout=1\nexpect S=0b--1-\nprint S carry\n", &nl).unwrap();
    let mut s = sim(&nl);
    let r = p.run(&mut s);
    assert!(!r.passed());
    assert_eq!(r.failures(), 1);
    assert_eq!(r.checks.len(), 3);
    let fail = r.checks.iter().find(|c| !c.passed).unwrap();
    assert_eq!((fail.line, fail.what.as_str(), fail.expected.as_str(), fail.actual.as_str()), (3, "Cout", "0b1", "0b0"));
    assert_eq!(r.printed.len(), 2);
    assert_eq!(r.printed[0].value, Value::from_u64(4, 2));

    let mut text = Vec::new();
    r.write(&mut text, "t.vec", false).unwrap();
    let text = String::from_utf8(text).unwrap();
    assert!(text.contains("FAIL t.vec:3: "), "{text}");
    assert!(text.contains("Cout: expected 0b1, got 0b0"), "{text}");
    assert!(text.contains("t.vec:5: [t="), "{text}");
    assert!(text.trim_end().ends_with("FAIL: 2 of 3 checks passed, 1 failed"), "{text}");
    assert!(!text.contains("ok  "), "passing checks hidden without verbose");
    let mut verbose = Vec::new();
    r.write(&mut verbose, "t.vec", true).unwrap();
    assert_eq!(String::from_utf8(verbose).unwrap().matches("ok   t.vec").count(), 2);
}

#[test]
fn settle_failure_and_oscillation_diagnostic() {
    let nl = netlist(&ring());
    let p = Program::parse("set en=1\nsettle 50\nexpect y=-\n", &nl).unwrap();
    let mut s = TimingSim::new(nl.clone(), TimingOptions::default().with_osc_limit(20));
    let r = p.run(&mut s);
    assert!(!r.passed());
    assert_eq!(r.checks[0].line, 2);
    assert_eq!(r.checks[0].actual, "still active");
    assert!(r.checks[1].passed);
    assert!(r.diagnostics.iter().any(|d| d.diagnostic.kind == panda_netlist::DiagKind::Oscillating && d.line == 2));

    // A free-running clock never settles either.
    let nl = netlist(&examples::counter4_clocked());
    let r = Program::parse("settle 100\n", &nl).unwrap().run(&mut sim(&nl));
    assert_eq!(r.failures(), 1);
}

#[test]
fn pulse_failure_is_reported() {
    let nl = netlist(&ring());
    // `en` pulses never settle because the ring oscillates while en=1.
    let r = Program::parse("pulse en 3\n", &nl).unwrap().run(&mut TimingSim::new(
        nl.clone(),
        TimingOptions::default().with_osc_limit(1_000_000_000),
    ));
    assert_eq!(r.checks.len(), 1);
    assert_eq!(r.checks[0].what, "pulse en");
    assert!(!r.passed());
}

#[test]
fn example_vector_files_pass() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples-data");
    for name in examples::NAMES {
        let nl = netlist(&examples::by_name(name).unwrap());
        let src = std::fs::read_to_string(dir.join(format!("{name}.vec"))).unwrap();
        let p = Program::parse(&src, &nl).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let r = p.run(&mut sim(&nl));
        assert!(r.passed(), "{name}: {:#?}", r.checks.iter().filter(|c| !c.passed).collect::<Vec<_>>());
        assert_eq!(r.checks.len(), p.expect_count());
    }
}

#[test]
fn x_and_z_expectations_are_exact() {
    let nl = netlist(&tristate());
    let p = Program::parse("set d=0b10 en=0\nsettle\nexpect y=z\nexpect y=x\nset en=1\nsettle\nexpect y=2\nexpect y=0b1-\n", &nl).unwrap();
    let r = p.run(&mut sim(&nl));
    let res: Vec<bool> = r.checks.iter().map(|c| c.passed).collect();
    assert_eq!(res, [true, false, true, true]);
}

// ---------------------------------------------------------------- VCD

/// Minimal VCD reader: returns (`code → (scope path + name, width)`, `time → [(code, value)]`).
#[allow(clippy::type_complexity)]
fn parse_vcd(text: &str) -> (HashMap<String, (String, u32)>, Vec<(u64, Vec<(String, String)>)>) {
    let mut vars = HashMap::new();
    let mut scopes: Vec<String> = Vec::new();
    let mut changes: Vec<(u64, Vec<(String, String)>)> = Vec::new();
    let mut in_defs = true;
    for line in text.lines() {
        let w: Vec<&str> = line.split_whitespace().collect();
        if w.is_empty() {
            continue;
        }
        if in_defs {
            match w[0] {
                "$scope" => scopes.push(w[2].to_string()),
                "$upscope" => {
                    scopes.pop();
                }
                "$var" => {
                    let full = [scopes[1..].to_vec(), vec![w[4].to_string()]].concat().join("/");
                    vars.insert(w[3].to_string(), (full, w[2].parse().unwrap()));
                }
                "$enddefinitions" => in_defs = false,
                _ => {}
            }
            continue;
        }
        if let Some(t) = w[0].strip_prefix('#') {
            let t: u64 = t.parse().unwrap();
            assert!(changes.last().is_none_or(|(p, _)| *p < t), "time not increasing at #{t}");
            changes.push((t, Vec::new()));
        } else if w[0] == "$dumpvars" || w[0] == "$end" {
        } else if let Some(bits) = w[0].strip_prefix('b') {
            changes.last_mut().unwrap().1.push((w[1].to_string(), bits.to_string()));
        } else {
            let (v, code) = w[0].split_at(1);
            changes.last_mut().unwrap().1.push((code.to_string(), v.to_string()));
        }
    }
    (vars, changes)
}

fn bits_msb_first(sim: &TimingSim, s: &Signal) -> String {
    s.nets.iter().rev().map(|&n| sim.get(n).to_char()).collect()
}

/// Dump with per-tick sampling and check that replaying the VCD reproduces the exact value of
/// every selected signal at every tick.
#[test]
fn vcd_replay_matches_simulation() {
    let nl = netlist(&examples::counter4_clocked());
    let mut sel = signals::default_selection(&nl);
    sel.push(signals::resolve(&nl, "cnt/ff0/m").unwrap());
    sel.push(signals::resolve(&nl, "cnt/next").unwrap());
    let mut s = sim(&nl);
    let mut w = VcdWriter::new(Vec::new(), &nl, "counter4_clocked", sel.clone()).unwrap();
    let mut expected: Vec<Vec<String>> = Vec::new();
    w.sample(&s).unwrap();
    expected.push(sel.iter().map(|g| bits_msb_first(&s, g)).collect());
    for _ in 0..300 {
        s.step();
        w.sample(&s).unwrap();
        expected.push(sel.iter().map(|g| bits_msb_first(&s, g)).collect());
    }
    let text = String::from_utf8(w.finish(&s).unwrap()).unwrap();
    assert!(text.contains("$timescale 1ns $end"));
    assert!(text.contains("$scope module counter4_clocked $end"));
    assert!(text.contains("$scope module cnt $end"));
    assert!(text.contains("$scope module ff0 $end"));
    assert!(text.contains("$var wire 4 ! Q [3:0] $end"), "{text}");
    assert!(text.contains("\n#300\n"));

    let (vars, changes) = parse_vcd(&text);
    let names: HashMap<&str, usize> = sel.iter().enumerate().map(|(i, g)| (g.name.as_str(), i)).collect();
    let mut cur: Vec<String> = vec![String::new(); sel.len()];
    let mut ci = 0;
    for (t, exp) in expected.iter().enumerate() {
        while ci < changes.len() && changes[ci].0 == t as u64 {
            for (code, v) in &changes[ci].1 {
                let (name, width) = &vars[code];
                assert_eq!(v.len() as u32, *width);
                cur[names[name.as_str()]] = v.clone();
            }
            ci += 1;
        }
        assert_eq!(&cur, exp, "mismatch at tick {t}");
    }
}

#[test]
fn vcd_coarse_sampling_and_reset_keep_time_monotonic() {
    let nl = netlist(&examples::counter4());
    let sel = signals::default_selection(&nl);
    let mut s = sim(&nl);
    let mut w = VcdWriter::new(Vec::new(), &nl, "counter4", sel.clone()).unwrap();
    w.sample(&s).unwrap();
    s.set_input("rst", &Value::from_bool(true)).unwrap();
    s.run(10); // several ticks at once: full comparison
    w.sample(&s).unwrap();
    w.sample(&s).unwrap(); // same tick twice: nothing new
    s.reset();
    w.sample(&s).unwrap();
    vcd::advance(&mut s, 5, Some(&mut w)).unwrap();
    let ok = vcd::settle(&mut s, 1000, Some(&mut w)).unwrap();
    assert!(ok);
    let text = String::from_utf8(w.finish(&s).unwrap()).unwrap();
    let (_, changes) = parse_vcd(&text); // asserts strictly increasing times
    let times: Vec<u64> = changes.iter().map(|c| c.0).collect();
    assert_eq!(times[0], 0);
    assert!(times.contains(&10));
    assert!(times.contains(&11), "reset continues right after the last time: {times:?}");
    assert_eq!(text.matches("$dumpvars").count(), 1);
}

#[test]
fn vcd_writes_z_and_x() {
    let nl = netlist(&tristate());
    let mut s = sim(&nl);
    let sel = signals::default_selection(&nl);
    let mut w = VcdWriter::new(Vec::new(), &nl, "tristate", sel).unwrap();
    w.sample(&s).unwrap();
    s.set_input("d", &Value::parse(2, "0b1x").unwrap()).unwrap();
    s.set_input("en", &Value::from_bool(true)).unwrap();
    vcd::settle(&mut s, 100, Some(&mut w)).unwrap();
    let text = String::from_utf8(w.finish(&s).unwrap()).unwrap();
    assert!(text.contains("bzz "), "{text}");
    assert!(text.contains("b1x "), "{text}");
    assert_eq!(s.read("y").unwrap().get(1), Logic::One);
}

#[test]
fn vectors_with_vcd_trace() {
    let nl = netlist(&examples::sr_latch());
    let src = "set S_n=0 R_n=1\nsettle\nexpect Q=1\nreset\nsettle\nexpect Q=1\n";
    let p = Program::parse(src, &nl).unwrap();
    let mut s = sim(&nl);
    let mut w = VcdWriter::new(Vec::new(), &nl, "sr_latch", signals::default_selection(&nl)).unwrap();
    let r = p.run_traced(&mut s, &mut w).unwrap();
    assert!(r.passed(), "{:?}", r.checks);
    let text = String::from_utf8(w.finish(&s).unwrap()).unwrap();
    parse_vcd(&text);
    assert!(text.contains("$var wire 1 ! S_n $end"));
}

#[test]
fn vcd_rejects_out_of_range_nets() {
    let nl = netlist(&examples::sr_latch());
    let bad = vec![Signal { name: "bad".into(), nets: vec![nl.num_nets + 5] }];
    assert!(VcdWriter::new(Vec::new(), &nl, "x", bad).is_err());
}
