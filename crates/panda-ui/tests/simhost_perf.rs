//! Sim-isolation stress scenarios: try to break the ≥ 95 % gate of docs/M6-UI.md with large
//! designs, many watched nets and waveform signals, Timing mode, churn from the UI and slow UI
//! frames. Every scenario compares headless `Kernel::run` with the full sim host (sim thread +
//! publisher + shadow capture + a UI thread polling at its frame rate), **interleaved** and
//! best-of-N, so the ratio stays meaningful on a loaded machine.
//!
//! ```text
//! cargo test --release -p panda-ui --test simhost_perf -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Environment: `PERF_SECS` (seconds per measurement, default 2), `PERF_ROUNDS` (default 3),
//! `PERF_ONLY` (comma-separated scenario names), `PERF_WORKERS` (shadow workers of scenarios
//! that do not set them). No GPU is involved (see `perf_gate.rs` for the
//! rendered UI): the "UI" here is a thread doing what `AppState::update` does each frame.

use std::sync::Arc;
use std::time::{Duration, Instant};

use panda_kernel::{EditOp, Kernel, KernelOptions, Mode};
use panda_netlist::flat::{FlatNetlist, NetIdx};
use panda_netlist::{Design, DesignBuilder, GateOp, Net, Sig, Termination, Value};
use panda_ui::examples;
use panda_ui::simhost::{
    Command, SimEvent, SimHost, SimOptions, Speed, WatchSet, WaveConfig, WaveSignal, WaveStore,
};

fn env_or(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// `n` copies of Panda-8 under one top (each with its own clock and reset).
fn panda8_x(n: usize) -> Design {
    let p8 = examples::panda8();
    let mut db = DesignBuilder::new();
    let top_ref = {
        db.merge(p8.clone());
        db.get(&p8.top).expect("panda8 top")
    };
    let mut b = panda_netlist::DefBuilder::new("panda8_x");
    let mut outs = Vec::new();
    for i in 0..n {
        let o = b.inst_named(&format!("cpu{i}"), &top_ref, &[]);
        outs.push(o);
    }
    // Expose the first copy's PC so the design has an output.
    let pc_pin = top_ref
        .pins
        .iter()
        .filter(|p| p.dir == panda_netlist::Dir::Out)
        .position(|p| p.name == "PC")
        .expect("PC pin");
    b.output("PC", outs[0][pc_pin]);
    let top = db.add(b.finish());
    db.build(&top)
}

/// A small deterministic PRNG (xorshift).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// A synchronous random design of about `gates` gates: `regs` 16-bit registers whose next state
/// is a random layered network of 2-input gates over the register outputs (plus a counter so
/// the state keeps changing).
fn random_design(gates: usize, regs: usize, seed: u64) -> Design {
    let mut db = DesignBuilder::new();
    let reg = panda_circuits::register(&mut db, 16);
    let mut b = panda_netlist::DefBuilder::new("random_top");
    let clk = b.clock(16, 8, 8);
    b.name(clk, "clk");
    let rst = b.power_on_reset(40);
    b.name(rst, "rst");
    let one = b.lit(1, 1);
    let mut rng = Rng(seed | 1);
    let qs: Vec<Sig> = (0..regs).map(|_| b.wire(16)).collect();
    let mut pool: Vec<Sig> = qs.iter().flat_map(|&q| b.bits(q)).collect();
    let ops = [
        GateOp::And,
        GateOp::Or,
        GateOp::Xor,
        GateOp::Nand,
        GateOp::Nor,
    ];
    let start = pool.len();
    for _ in 0..gates {
        // Prefer recent signals: layered, bounded depth.
        let span = pool.len().min(4096);
        let a = pool[pool.len() - 1 - rng.below(span)];
        let c = pool[rng.below(pool.len())];
        let op = ops[rng.below(ops.len())];
        let g = b.gate(op, &[a, c]);
        pool.push(g);
    }
    let fresh = &pool[start..];
    for (i, &q) in qs.iter().enumerate() {
        let bits: Vec<Sig> = (0..16)
            .map(|k| fresh[fresh.len() - 1 - ((i * 16 + k) * 7919) % fresh.len()])
            .collect();
        let mut d = b.concat(&bits);
        if i == 0 {
            // Register 0 counts: XOR with an incrementing pattern keeps everything moving.
            let qb = b.bits(q);
            let mut carry = one;
            let mut sum = Vec::new();
            for &x in &qb {
                sum.push(b.xor(&[x, carry]));
                carry = b.and(&[x, carry]);
            }
            let inc = b.concat(&sum);
            d = b.xor(&[d, inc]);
        }
        let o = b.inst_map(
            &format!("r{i}"),
            &reg,
            &[("D", d), ("LOAD", one), ("RST", rst), ("CLK", clk)],
        );
        b.connect(o["Q"], q);
        b.name(q, &format!("q{i}"));
    }
    b.output("Q0", qs[0]);
    let top = db.add(b.finish());
    db.build(&top)
}

/// What the UI side does in a scenario.
#[derive(Clone, Copy, Debug, Default)]
struct UiLoad {
    /// Frame period of the UI thread.
    frame: Duration,
    /// Watch every net of the design (worst case of "nets of the viewed def").
    watch_all: bool,
    /// Waveform signals (first nets of the design, 1 net each; or 8-bit buses).
    wave_signals: usize,
    /// Waveform signals are 8-bit buses.
    wave_buses: bool,
    /// Send a structural edit every `n` frames.
    edit_every: Option<u64>,
    /// Switch the watch set every frame (deep view switches).
    watch_churn: bool,
    /// Toggle Fast/Timing every `n` frames.
    mode_every: Option<u64>,
    /// Do not poll events at all (a frozen UI).
    frozen: bool,
    /// Shadow replay workers (0: default).
    workers: usize,
}

struct HostResult {
    rate: f64,
    publish_us: f64,
    pubs_per_s: f64,
    gaps: u64,
    wave_changes: usize,
}

fn kernel_for(design: &Design, mode: Mode) -> Kernel {
    let mut k = Kernel::new(KernelOptions::default());
    k.set_mode(mode);
    k.load(design.clone()).unwrap();
    k.wait_ready().unwrap();
    if mode == Mode::Fast {
        assert!(k.wait_jit(Duration::from_secs(120)), "JIT did not activate");
    }
    k.run(Duration::from_millis(200));
    k
}

fn headless(design: &Design, mode: Mode, secs: f64) -> f64 {
    let mut k = kernel_for(design, mode);
    let s = k.run(Duration::from_secs_f64(secs));
    s.cycles as f64 / s.elapsed.as_secs_f64()
}

fn wait_until(
    h: &mut SimHost,
    secs: f64,
    mut pred: impl FnMut(&panda_ui::simhost::Snapshot) -> bool,
) {
    let t0 = Instant::now();
    loop {
        h.poll_events();
        if pred(h.snapshot()) {
            return;
        }
        assert!(
            t0.elapsed().as_secs_f64() < secs,
            "timed out: {:?}",
            h.snapshot()
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Multi-bit nets `name[0..8]` of `flat` (base names of 8-bit buses), or single nets.
fn wave_signals(flat: &FlatNetlist, n: usize, buses: bool) -> Vec<WaveSignal> {
    let mut out = Vec::new();
    if buses {
        let mut seen = std::collections::HashSet::new();
        for name in &flat.net_names {
            let Some(base) = name.strip_suffix("[0]") else {
                continue;
            };
            if !seen.insert(base.to_string()) {
                continue;
            }
            let nets: Vec<NetIdx> = (0..8)
                .map_while(|i| flat.net_by_name(&format!("{base}[{i}]")))
                .collect();
            if nets.len() == 8 {
                out.push(WaveSignal::new(flat, base, nets));
            }
            if out.len() == n {
                break;
            }
        }
    } else {
        for (i, name) in flat.net_names.iter().enumerate().step_by(7) {
            if out.len() == n {
                break;
            }
            out.push(WaveSignal::new(flat, name, vec![i as NetIdx]));
        }
    }
    out
}

fn host(design: &Design, mode: Mode, secs: f64, ui: UiLoad) -> HostResult {
    let mut opts = SimOptions::default();
    let env_workers = env_or("PERF_WORKERS", 0.0) as usize;
    if ui.workers > 0 {
        opts.shadow_workers = ui.workers;
    } else if env_workers > 0 {
        opts.shadow_workers = env_workers;
    }
    let mut h = SimHost::spawn(opts);
    h.send(Command::Load(Box::new(design.clone())));
    let (gen_, flat) = loop {
        if let Some(SimEvent::Netlist { design_gen, flat }) = h.wait_event(Duration::from_secs(60))
        {
            break (design_gen, flat);
        }
    };
    h.send(Command::SetMode(mode));
    wait_until(&mut h, 120.0, |s| {
        if mode == Mode::Fast {
            s.jit
        } else {
            s.engine == Some(panda_kernel::EngineKind::Timing)
        }
    });
    let all: Vec<NetIdx> = (0..flat.num_nets as NetIdx).collect();
    let top_nets: Vec<NetIdx> = all.iter().copied().step_by(3).take(500).collect();
    let deep_nets: Vec<NetIdx> = all.iter().copied().skip(1).step_by(5).take(300).collect();
    let watch = if ui.watch_all {
        all.clone()
    } else {
        top_nets.clone()
    };
    h.send(Command::SetWatch(WatchSet {
        id: 1,
        design_gen: gen_,
        nets: watch,
    }));
    let mut store = WaveStore::default();
    if ui.wave_signals > 0 {
        let cfg = WaveConfig {
            id: 7,
            design_gen: gen_,
            signals: wave_signals(&flat, ui.wave_signals, ui.wave_buses),
        };
        store = WaveStore::new(cfg.clone());
        h.send(Command::SetWave(cfg));
    }
    h.send(Command::Run);
    wait_until(&mut h, 30.0, |s| s.running && s.stats.cycles > 0);
    // Warm up.
    let t_end = Instant::now() + Duration::from_millis(300);
    while Instant::now() < t_end {
        for ev in h.poll_events() {
            if let SimEvent::Wave(c) = ev {
                store.append(&c);
            }
        }
        std::thread::sleep(ui.frame.min(Duration::from_millis(16)));
    }
    let s0 = h.snapshot().clone();
    let g0 = store.gaps;
    let w0: usize = store.history.iter().map(|q| q.len()).sum();
    let t0 = Instant::now();
    let mut frame = 0u64;
    let mut cur_gen = gen_;
    let mut edit_n = 0u64;
    let mut fast = mode == Mode::Fast;
    while t0.elapsed().as_secs_f64() < secs {
        frame += 1;
        if !ui.frozen {
            for ev in h.poll_events() {
                match ev {
                    SimEvent::Wave(c) => store.append(&c),
                    SimEvent::Netlist { design_gen, .. } => cur_gen = design_gen,
                    _ => {}
                }
            }
            let _ = h.snapshot();
        }
        if ui.watch_churn {
            let nets = if frame.is_multiple_of(2) {
                &top_nets
            } else {
                &deep_nets
            };
            h.send(Command::SetWatch(WatchSet {
                id: frame,
                design_gen: cur_gen,
                nets: nets.clone(),
            }));
        }
        if let Some(n) = ui.edit_every
            && frame.is_multiple_of(n)
        {
            edit_n += 1;
            let net = format!("__perf_probe_{}", edit_n / 2);
            let op = if edit_n % 2 == 1 {
                EditOp::AddNet {
                    def: design.top.clone(),
                    net: Net {
                        id: net,
                        width: 1,
                        termination: Termination::None,
                        ends: Vec::new(),
                    },
                }
            } else {
                EditOp::RemoveNet {
                    def: design.top.clone(),
                    id: net,
                }
            };
            h.send(Command::Edit {
                id: edit_n,
                ops: vec![op],
            });
        }
        if let Some(n) = ui.mode_every
            && frame.is_multiple_of(n)
        {
            fast = !fast;
            h.send(Command::SetMode(if fast {
                Mode::Fast
            } else {
                Mode::Timing
            }));
        }
        std::thread::sleep(ui.frame);
    }
    // One more publication, then read the counters.
    std::thread::sleep(Duration::from_millis(30));
    for ev in h.poll_events() {
        if let SimEvent::Wave(c) = ev {
            store.append(&c);
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let s1 = h.snapshot().clone();
    let w1: usize = store.history.iter().map(|q| q.len()).sum();
    HostResult {
        rate: (s1.stats.cycles - s0.stats.cycles) as f64 / elapsed,
        publish_us: s1.stats.publish_us,
        pubs_per_s: (s1.seq - s0.seq) as f64 / elapsed,
        gaps: store.gaps - g0,
        wave_changes: w1.saturating_sub(w0),
    }
}

struct Scenario {
    name: &'static str,
    design: Arc<Design>,
    mode: Mode,
    ui: UiLoad,
    /// Gated (≥ 95 %) or report-only.
    gated: bool,
}

fn ui60() -> UiLoad {
    UiLoad {
        frame: Duration::from_millis(16),
        ..UiLoad::default()
    }
}

#[test]
#[ignore = "perf: run with --release --ignored --nocapture --test-threads=1"]
fn sim_isolation_scenarios() {
    let secs = env_or("PERF_SECS", 2.0);
    let rounds = env_or("PERF_ROUNDS", 3.0) as usize;
    let only: Option<Vec<String>> = std::env::var("PERF_ONLY")
        .ok()
        .map(|s| s.split(',').map(str::to_string).collect());
    let p8 = Arc::new(examples::panda8());
    let p8x26 = Arc::new(panda8_x(26));
    let rnd = Arc::new(random_design(100_000, 64, 0x5eed));
    let mut scenarios = Vec::new();
    for (name, w, buses) in [
        ("p8-8waves-w1", 1, false),
        ("p8-8waves-w2", 2, false),
        ("p8-8waves-w4", 4, false),
        ("p8-8buswaves-w1", 1, true),
        ("p8-8buswaves-w2", 2, true),
        ("p8-8buswaves-w4", 4, true),
    ] {
        scenarios.push(Scenario {
            name,
            design: p8.clone(),
            mode: Mode::Fast,
            ui: UiLoad {
                wave_signals: 8,
                wave_buses: buses,
                workers: w,
                ..ui60()
            },
            gated: true,
        });
    }
    scenarios.extend([
        Scenario {
            name: "p8-fast",
            design: p8.clone(),
            mode: Mode::Fast,
            ui: ui60(),
            gated: true,
        },
        Scenario {
            name: "p8-fast-watch-all",
            design: p8.clone(),
            mode: Mode::Fast,
            ui: UiLoad {
                watch_all: true,
                ..ui60()
            },
            gated: true,
        },
        Scenario {
            name: "p8-fast-64-waves",
            design: p8.clone(),
            mode: Mode::Fast,
            ui: UiLoad {
                wave_signals: 64,
                ..ui60()
            },
            gated: true,
        },
        Scenario {
            name: "p8-fast-64-bus-waves",
            design: p8.clone(),
            mode: Mode::Fast,
            ui: UiLoad {
                wave_signals: 64,
                wave_buses: true,
                ..ui60()
            },
            gated: true,
        },
        Scenario {
            name: "p8-fast-slow-ui",
            design: p8.clone(),
            mode: Mode::Fast,
            ui: UiLoad {
                frame: Duration::from_millis(500),
                wave_signals: 8,
                ..ui60()
            },
            gated: true,
        },
        Scenario {
            name: "p8-fast-frozen-ui",
            design: p8.clone(),
            mode: Mode::Fast,
            ui: UiLoad {
                frozen: true,
                wave_signals: 8,
                ..ui60()
            },
            gated: true,
        },
        Scenario {
            name: "p8-fast-watch-churn",
            design: p8.clone(),
            mode: Mode::Fast,
            ui: UiLoad {
                watch_churn: true,
                wave_signals: 8,
                ..ui60()
            },
            gated: true,
        },
        Scenario {
            name: "p8-timing",
            design: p8.clone(),
            mode: Mode::Timing,
            ui: UiLoad {
                wave_signals: 8,
                ..ui60()
            },
            gated: true,
        },
        Scenario {
            name: "p8x26-fast",
            design: p8x26.clone(),
            mode: Mode::Fast,
            ui: UiLoad {
                wave_signals: 8,
                ..ui60()
            },
            gated: true,
        },
        Scenario {
            name: "p8x26-fast-watch-all",
            design: p8x26.clone(),
            mode: Mode::Fast,
            ui: UiLoad {
                watch_all: true,
                wave_signals: 8,
                ..ui60()
            },
            gated: true,
        },
        Scenario {
            name: "rnd100k-fast",
            design: rnd.clone(),
            mode: Mode::Fast,
            ui: UiLoad {
                wave_signals: 8,
                ..ui60()
            },
            gated: true,
        },
        Scenario {
            name: "rnd100k-fast-watch-all",
            design: rnd.clone(),
            mode: Mode::Fast,
            ui: UiLoad {
                watch_all: true,
                wave_signals: 64,
                ..ui60()
            },
            gated: true,
        },
        Scenario {
            name: "rnd100k-timing",
            design: rnd.clone(),
            mode: Mode::Timing,
            ui: UiLoad {
                wave_signals: 8,
                ..ui60()
            },
            gated: true,
        },
        // Report-only: the kernel recompiles after every edit (Timing fallback meanwhile) and a
        // mode switch changes engines; these measure the cost, not the isolation.
        Scenario {
            name: "p8-fast-edits-2Hz",
            design: p8.clone(),
            mode: Mode::Fast,
            ui: UiLoad {
                edit_every: Some(30),
                wave_signals: 8,
                ..ui60()
            },
            gated: false,
        },
        Scenario {
            name: "p8-mode-switch-1Hz",
            design: p8.clone(),
            mode: Mode::Fast,
            ui: UiLoad {
                mode_every: Some(60),
                ..ui60()
            },
            gated: false,
        },
    ]);
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    eprintln!("load average at start: {}", load.trim());
    let mut table = Vec::new();
    let mut failures = Vec::new();
    for sc in &scenarios {
        if only
            .as_ref()
            .is_some_and(|o| !o.iter().any(|n| n == sc.name))
        {
            continue;
        }
        let mut hs = Vec::new();
        let mut fs = Vec::new();
        let mut last = None;
        for r in 0..rounds {
            let hr = headless(&sc.design, sc.mode, secs);
            let f = host(&sc.design, sc.mode, secs, sc.ui);
            eprintln!(
                "{:<24} round {r}: headless {:>10.0} c/s, host {:>10.0} c/s ({:5.1} %), pubs {:5.1}/s, publish {:6.1} µs, wave changes {}, gaps {}",
                sc.name,
                hr,
                f.rate,
                100.0 * f.rate / hr,
                f.pubs_per_s,
                f.publish_us,
                f.wave_changes,
                f.gaps
            );
            hs.push(hr);
            fs.push(f.rate);
            last = Some(f);
        }
        let best = |v: &[f64]| v.iter().copied().fold(0.0, f64::max);
        let ratio = best(&fs) / best(&hs);
        let f = last.expect("rounds ≥ 1");
        table.push(format!(
            "| {:<24} | {:>11.0} | {:>11.0} | {:>6.1} % | {:>6.1} | {:>7.1} | {:>5} | {} |",
            sc.name,
            best(&hs),
            best(&fs),
            100.0 * ratio,
            f.pubs_per_s,
            f.publish_us,
            f.gaps,
            if sc.gated { "gated" } else { "report" }
        ));
        if sc.gated && ratio < 0.95 {
            failures.push(format!("{}: {:.1} %", sc.name, 100.0 * ratio));
        }
    }
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    eprintln!("load average at end: {}", load.trim());
    eprintln!(
        "| scenario                 | headless c/s | host c/s   | ratio    | pubs/s | pub µs  | gaps  | gate |"
    );
    eprintln!("|---|---|---|---|---|---|---|---|");
    for row in &table {
        eprintln!("{row}");
    }
    assert!(failures.is_empty(), "below 95 %: {failures:?}");
}

/// Target-rate pacing accuracy (achieved / target) at several rates, Panda-8 Fast mode.
#[test]
#[ignore = "perf: run with --release --ignored --nocapture --test-threads=1"]
fn pacing_accuracy() {
    let design = examples::panda8();
    let mut h = SimHost::spawn(SimOptions::default());
    h.send(Command::Load(Box::new(design)));
    wait_until(&mut h, 60.0, |s| s.design_gen > 0);
    h.send(Command::SetMode(Mode::Fast));
    wait_until(&mut h, 120.0, |s| s.jit);
    eprintln!("| target c/s | achieved c/s | ratio |");
    eprintln!("|---|---|---|");
    let mut worst: f64 = 1.0;
    for rate in [10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0] {
        h.send(Command::SetSpeed(Speed::CyclesPerSec(rate)));
        h.send(Command::Run);
        std::thread::sleep(Duration::from_millis(300));
        wait_until(&mut h, 5.0, |s| s.running);
        let s0 = h.snapshot().clone();
        let t0 = Instant::now();
        std::thread::sleep(Duration::from_secs(2));
        h.poll_events();
        let s1 = h.snapshot().clone();
        let got = (s1.stats.cycles - s0.stats.cycles) as f64 / t0.elapsed().as_secs_f64();
        eprintln!("| {rate:>10.0} | {got:>12.1} | {:>5.3} |", got / rate);
        worst = worst.min(got / rate).min(rate / got.max(1e-9));
        h.send(Command::Pause);
    }
    assert!(worst > 0.8, "pacing off by more than 20 %");
}

/// Keeps the generator honest: the random design is about the requested size and runs.
#[test]
fn random_design_has_the_requested_size() {
    let d = random_design(2_000, 4, 1);
    let mut k = Kernel::new(KernelOptions::default());
    k.load(d).unwrap();
    let flat = k.netlist().unwrap();
    assert!(flat.num_nets > 2_000, "{} nets", flat.num_nets);
    k.step(4, panda_kernel::StepUnit::Cycle);
    let _ = Value::from_bool(true);
}
