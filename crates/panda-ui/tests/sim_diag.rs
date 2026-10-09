//! Diagnostics and STA panels (`panda_ui::simux::{diag, sta, locate}`): location mapping at
//! depth (pure), the STA model on Panda-8 (worker thread), and offscreen (egui_kittest + wgpu)
//! jump / highlight / apply-period interaction.
//!
//! Writes `docs/screenshots/sim-diag-*.png` (inspect them after visual changes).

use std::path::PathBuf;
use std::time::{Duration, Instant};

use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use panda_netlist::{Atom, DefBuilder, Design, DesignBuilder, DiagKind, InstKind};
use panda_ui::app::{AppState, PandaApp, def_at_path};
use panda_ui::doc::ItemRef;
use panda_ui::examples;
use panda_ui::simhost::SimOptions;
use panda_ui::simux::locate::{Mark, resolve_location};
use panda_ui::simux::sta::{analyze_with_model, find_clock};

// ------------------------------------------------------------------------------------------
// Pure: location mapping
// ------------------------------------------------------------------------------------------

/// Every canonical flat-net name of Panda-8 (nets at every depth, bits of buses) maps to a
/// view showing that net (or the pin / instance carrying it).
#[test]
fn panda8_every_net_location_resolves() {
    let d = examples::panda8();
    let flat = panda_netlist::flat::flatten(&d).expect("flattens");
    let mut max_depth = 0;
    let mut nets = 0;
    for name in &flat.net_names {
        let t = resolve_location(&d, Some(&flat), name)
            .unwrap_or_else(|| panic!("{name} does not resolve"));
        assert_ne!(t.mark, Mark::None, "{name}");
        assert_eq!(
            def_at_path(&d, &t.path).as_deref(),
            Some(t.def.as_str()),
            "{name}"
        );
        // The marked net (if a net) exists in the def viewed there.
        if let Mark::Net(id) = &t.mark {
            let def = d.def(&t.def).unwrap();
            assert!(def.nets.iter().any(|n| &n.id == id), "{name} → {id}");
            nets += 1;
        }
        max_depth = max_depth.max(t.path.matches('/').count() + usize::from(!t.path.is_empty()));
    }
    eprintln!(
        "{} names resolved, {nets} as nets, max depth {max_depth}",
        flat.net_names.len()
    );
    assert!(max_depth >= 2, "Panda-8 has nets at depth ≥ 2");
}

/// Instances and pins at depth, validation-style locations and unknown names.
#[test]
fn panda8_instance_and_validation_locations() {
    let d = examples::panda8();
    // Find a package instance two levels down and an atom inside it.
    let top = d.top_def().unwrap();
    let (p1, d1) = top
        .instances
        .iter()
        .find_map(|i| match &i.kind {
            InstKind::Def(sub) => Some((i.id.clone(), d.def(sub).unwrap())),
            InstKind::Atom(_) => None,
        })
        .expect("a package at the top");
    let (p2, d2) = d1
        .instances
        .iter()
        .find_map(|i| match &i.kind {
            InstKind::Def(sub) => Some((format!("{p1}/{}", i.id), d.def(sub).unwrap())),
            InstKind::Atom(_) => None,
        })
        .expect("a package at depth 2");
    let atom = d2
        .instances
        .iter()
        .find(|i| matches!(i.kind, InstKind::Atom(_)))
        .expect("an atom at depth 3");
    let t = resolve_location(&d, None, &format!("{p2}/{}", atom.id)).expect("instance");
    assert_eq!(t.path, p2);
    assert_eq!(t.def, d2.id);
    assert_eq!(t.mark, Mark::Item(ItemRef::Instance(atom.id.clone())));
    // A package instance: marked in its parent.
    let t = resolve_location(&d, None, &p2).expect("package instance");
    assert_eq!(
        (t.path.as_str(), t.def.as_str()),
        (p1.as_str(), d1.id.as_str())
    );
    // A pin of the package at depth 2.
    let pin = &d2.pins[0];
    let t = resolve_location(&d, None, &format!("{p2}/{}", pin.name)).expect("pin");
    assert_eq!(t.path, p2);
    assert!(
        matches!(&t.mark, Mark::Net(_) | Mark::Item(ItemRef::Pin(_))),
        "{:?}",
        t.mark
    );
    // Validation form "<def>: net <id>" goes to the first instance of that def.
    let net = &d2.nets[0];
    let t = resolve_location(&d, None, &format!("{}: net {}", d2.id, net.id)).expect("validation");
    assert_eq!(t.def, d2.id);
    assert_eq!(def_at_path(&d, &t.path).as_deref(), Some(d2.id.as_str()));
    assert_eq!(t.mark, Mark::Net(net.id.clone()));
    // Breadcrumbs from the hierarchy navigation.
    let crumbs = t.crumbs(&d);
    assert_eq!(crumbs.last().unwrap().def, d2.id);
    assert!(t.describe(&d).ends_with(&format!(": net {}", net.id)));
    // Not places.
    assert!(resolve_location(&d, None, "edit op #4").is_none());
    assert!(resolve_location(&d, None, "no/such/net").is_none());
    assert!(resolve_location(&d, None, "").is_none());
}

/// STA on Panda-8 on a worker thread: the model's clocks map to the top `Clock` source and
/// every net of the critical path is a jumpable location.
#[test]
fn panda8_sta_model() {
    let d = examples::panda8();
    let t0 = Instant::now();
    let dd = d.clone();
    let (s, m) = std::thread::spawn(move || analyze_with_model(dd))
        .join()
        .unwrap()
        .expect("Panda-8 loads");
    eprintln!(
        "Panda-8 STA (fresh kernel, incl. compile): {:.2} s; clocks {:?}; critical path {} nets; \
         {} findings ({} errors); verdict: {}",
        t0.elapsed().as_secs_f64(),
        m.clocks
            .iter()
            .map(|c| (&c.net, c.period, &c.min_period))
            .collect::<Vec<_>>(),
        m.paths[0].nets.len(),
        m.findings.len(),
        m.errors,
        m.verdict
    );
    assert!(!m.clocks.is_empty());
    for c in &m.clocks {
        let src = c.source.as_ref().expect("clock source found");
        assert_eq!(src.path, "", "clocks live in the top def");
        assert_eq!(src.period, c.period);
    }
    assert!(!m.paths[0].nets.is_empty(), "Panda-8 has a critical path");
    assert_eq!(m.paths[0].nets, s.report.critical_path);
    let flat = panda_netlist::flat::flatten(&d).unwrap();
    for n in &m.paths[0].nets {
        assert!(
            resolve_location(&d, Some(&flat), n).is_some(),
            "path net {n}"
        );
    }
    for c in &m.clocks {
        assert!(find_clock(&d, Some(&flat), &c.net).is_some());
    }
}

// ------------------------------------------------------------------------------------------
// Offscreen UI
// ------------------------------------------------------------------------------------------

fn screenshot_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/screenshots")
        .join(format!("sim-diag-{name}.png"))
}

fn harness(app: PandaApp, w: f32, h: f32) -> Harness<'static, PandaApp> {
    Harness::builder()
        .with_size(egui::vec2(w, h))
        .wgpu()
        .build_ui_state(|ui, app: &mut PandaApp| app.show(ui), app)
}

fn step_until(
    h: &mut Harness<'_, PandaApp>,
    secs: f64,
    what: &str,
    mut pred: impl FnMut(&PandaApp) -> bool,
) {
    let t0 = Instant::now();
    while !pred(h.state()) {
        assert!(
            t0.elapsed().as_secs_f64() < secs,
            "timed out waiting for {what}"
        );
        h.step();
        std::thread::sleep(Duration::from_millis(5));
    }
    h.step();
}

fn frames(h: &mut Harness<'_, PandaApp>, n: usize) {
    for _ in 0..n {
        h.step();
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn save(h: &mut Harness<'_, PandaApp>, name: &str) {
    let img = h.render().expect("render");
    let p = screenshot_path(name);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    img.save(&p).unwrap();
}

fn click(h: &mut Harness<'_, PandaApp>, label: &str) {
    h.step();
    h.query_all_by_label(label)
        .next()
        .unwrap_or_else(|| panic!("no widget labelled {label:?}"))
        .click();
    h.step();
    h.step();
}

fn click_contains(h: &mut Harness<'_, PandaApp>, label: &str) {
    h.step();
    h.query_all_by_label_contains(label)
        .next()
        .unwrap_or_else(|| panic!("no widget with label containing {label:?}"))
        .click();
    h.step();
    h.step();
}

/// Make the bottom panel `dy` points taller (sets the panel's stored size, as a drag of its
/// edge would).
fn grow_bottom(h: &mut Harness<'_, PandaApp>, dy: f32) {
    use egui::containers::panel::PanelState;
    h.step();
    let id = egui::Id::new("panda_bottom");
    let ctx = h.ctx.clone();
    let mut st = PanelState::load(&ctx, id).expect("bottom panel shown");
    st.outer_rect.min.y -= dy;
    ctx.data_mut(|d| d.insert_persisted(id, st));
    h.step();
    h.step();
    let now = PanelState::load(&ctx, id).unwrap().size().y;
    assert!(
        now >= st.outer_rect.height() - 1.0,
        "bottom panel resized ({now})"
    );
}

fn ready(state: AppState, w: f32, hgt: f32) -> Harness<'static, PandaApp> {
    let mut h = harness(PandaApp::new(state), w, hgt);
    step_until(&mut h, 30.0, "live values", |a| {
        a.state.live.values().is_some()
    });
    h
}

fn wait_sta(h: &mut Harness<'_, PandaApp>, secs: u64) -> f64 {
    let t0 = Instant::now();
    while h.query_by_label_contains("analyzed in").is_none() {
        assert!(t0.elapsed() < Duration::from_secs(secs), "STA timed out");
        h.step();
        std::thread::sleep(Duration::from_millis(10));
    }
    t0.elapsed().as_secs_f64()
}

/// A testbench whose package `u0` drives net `bus` from a clock buffer and a constant
/// (contention inside the package).
fn contention_design() -> Design {
    let mut db = DesignBuilder::new();
    let mut b = DefBuilder::new("noisy_core");
    let clk = b.input("CLK", 1);
    let shared = b.buf(clk);
    let shared = b.name(shared, "bus");
    let zero = b.lit(1, 0);
    b.connect(shared, zero);
    let q = b.not(shared);
    b.output("S", q);
    let core = db.add(b.finish());
    let mut b = DefBuilder::new("top");
    let clk = b.clock(8, 4, 4);
    let clk = b.name(clk, "clk");
    let o = b.inst_named("u0", &core, &[clk]);
    b.output("S", o[0]);
    let top = db.add(b.finish());
    db.build(&top)
}

/// Run until a contention is reported, open the diagnostics tab, filter, click the
/// contention's location: the view enters `u0`, `bus` is in the contention colour and the
/// jump overlay rings it.
#[test]
fn diagnostics_contention_jump_and_filters() {
    let mut state = AppState::new(contention_design(), SimOptions::default());
    state.wait_layout();
    let mut h = ready(state, 1300.0, 820.0);
    click(&mut h, "▶ Run");
    step_until(&mut h, 30.0, "contention diagnostic", |a| {
        a.state
            .diagnostics
            .iter()
            .any(|e| e.diagnostic.kind == DiagKind::Contention && e.count > 2)
    });
    click(&mut h, "⏸ Pause");
    click(&mut h, "Diagnostics");
    grow_bottom(&mut h, 140.0);
    // Group header with counts and the tick range.
    assert!(
        h.query_by_label_contains("Contention — ").is_some(),
        "contention group header"
    );
    let loc = h
        .state()
        .state
        .diagnostics
        .iter()
        .find(|e| e.diagnostic.kind == DiagKind::Contention)
        .and_then(|e| e.diagnostic.location.clone())
        .expect("located");
    eprintln!("contention at {loc}");
    // Hide warnings: the contention row disappears; show them again.
    click_contains(&mut h, "warnings");
    assert!(h.query_by_label_contains("Contention — ").is_none());
    click_contains(&mut h, "warnings");
    assert!(h.query_by_label_contains("Contention — ").is_some());
    click(&mut h, &loc);
    frames(&mut h, 6);
    {
        let s = &h.state().state;
        assert_eq!(s.view.path, "u0", "jumped into the package");
        assert!(
            s.highlights.contention.contains("bus"),
            "{:?}",
            s.highlights
        );
        assert!(s.highlights.nets.contains("bus"));
    }
    assert!(
        h.query_by_label_contains("› u0: net bus").is_some(),
        "breadcrumb description of the jump"
    );
    save(&mut h, "contention");
    // The validation warning's location ("noisy_core: net bus") jumps too.
    click(&mut h, "noisy_core: net bus");
    assert_eq!(h.state().state.view.path, "u0");
    // Clearing removes only our highlight.
    click(&mut h, "Clear highlight");
    assert!(!h.state().state.highlights.contention.contains("bus"));
}

/// STA on Panda-8 viewed inside `cpu`: critical path highlighted, the panel lists clocks and
/// the path; clicking a path net jumps into its package.
#[test]
fn sta_panel_panda8() {
    let mut state = AppState::new(examples::panda8(), SimOptions::default());
    state.wait_layout();
    let mut h = ready(state, 1500.0, 950.0);
    assert!(h.state_mut().state.enter("cpu"));
    h.state_mut().state.canvas.request_fit();
    step_until(&mut h, 20.0, "cpu view", |a| {
        a.state.live.values().is_some_and(|l| l.map.path == "cpu")
    });
    // Some live values first (wires are X at tick 0).
    h.state_mut()
        .state
        .sim
        .send(panda_ui::simhost::Command::Step {
            n: 40,
            unit: panda_kernel::StepUnit::Cycle,
        });
    step_until(&mut h, 20.0, "stepped", |a| {
        a.state.live.snapshot.tick > 1000
    });
    click(&mut h, "Timing (STA)");
    grow_bottom(&mut h, 300.0);
    click(&mut h, "Run STA");
    let secs = wait_sta(&mut h, 180);
    eprintln!("STA on Panda-8 in the UI: {secs:.1} s wall (worker thread, incl. compile)");
    frames(&mut h, 4);
    let hl = h.state().state.highlights.nets.len();
    eprintln!("critical-path nets highlighted in the cpu view: {hl}");
    assert!(hl > 0, "critical path highlighted");
    assert!(h.query_by_label_contains("Critical path (").is_some());
    assert!(h.query_by_label("Clocks").is_some());
    // The UI kept rendering while STA ran: frames are cheap.
    let f = h.state().state.frame;
    eprintln!("frame avg {:.2} ms over {} frames", f.avg_ms, f.frames);
    save(&mut h, "sta-panda8");
    // Jump to the deepest net of the path.
    let deepest = {
        let d = &h.state().state.doc.design;
        let flat = h.state().state.live.flat.clone().unwrap();
        let path = panda_ui::simux::sta::analyze_with_model(d.clone())
            .unwrap()
            .1
            .paths[0]
            .nets
            .clone();
        // Among the first rows (visible without scrolling the panel).
        path.into_iter()
            .take(10)
            .filter_map(|n| resolve_location(d, Some(&flat), &n).map(|t| (n, t)))
            .max_by_key(|(_, t)| t.path.matches('/').count() + usize::from(!t.path.is_empty()))
            .expect("a path net")
    };
    eprintln!("jumping to {} → {:?}", deepest.0, deepest.1);
    click(&mut h, &deepest.0);
    frames(&mut h, 6);
    let s = &h.state().state;
    assert_eq!(s.view.path, deepest.1.path);
    if let Mark::Net(id) = &deepest.1.mark {
        assert!(s.highlights.nets.contains(id));
    }
    save(&mut h, "sta-panda8-jump");
}

/// A counter with a too-fast clock: "Apply" sets the suggested period through the editing
/// core (re-analysis meets timing), "Undo period" restores it.
#[test]
fn sta_apply_suggested_period_and_undo() {
    let mut db = DesignBuilder::new();
    let c = panda_circuits::sync_counter(&mut db, 4);
    let top = panda_circuits::testbench(&mut db, &c, 4);
    let design = db.build(&top);
    let mut state = AppState::new(design, SimOptions::default());
    state.wait_layout();
    let mut h = ready(state, 1300.0, 820.0);
    click(&mut h, "Timing (STA)");
    grow_bottom(&mut h, 160.0);
    click(&mut h, "Run STA");
    wait_sta(&mut h, 90);
    frames(&mut h, 2);
    assert!(
        h.query_by_label("too fast").is_some(),
        "period 4 must be too fast"
    );
    save(&mut h, "sta-too-fast");
    let period = |h: &Harness<'_, PandaApp>| {
        h.state()
            .state
            .doc
            .design
            .top_def()
            .unwrap()
            .instances
            .iter()
            .find_map(|i| match &i.kind {
                InstKind::Atom(Atom::Clock { period, .. }) => Some(*period),
                _ => None,
            })
            .unwrap()
    };
    click_contains(&mut h, "Apply ");
    frames(&mut h, 2);
    let applied = period(&h);
    assert!(applied > 4, "clock period applied: {applied}");
    let t0 = Instant::now();
    loop {
        h.step();
        if h.query_by_label("meets").is_some() && h.query_by_label_contains("analyzed in").is_some()
        {
            break;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(90),
            "re-analysis timed out"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    frames(&mut h, 2);
    save(&mut h, "sta-applied");
    click(&mut h, "Undo period");
    frames(&mut h, 2);
    assert_eq!(period(&h), 4, "undo restores the period");
    // The kernel accepted both edits (no rejection logged).
    assert!(
        !h.state().state.log.iter().any(|l| l.contains("rejected")),
        "{:?}",
        h.state().state.log
    );
}
