//! Canvas scaling: a single def with 10k+ gate symbols (six copies of the flattened MuxTree Panda-8),
//! auto-laid out and rendered offscreen at several zoom levels. Reports layout time and frame
//! times; asserts the canvas's own mesh build stays well inside a 60 fps frame.

use std::time::Instant;

use egui_kittest::Harness;
use panda_cpu8::{Options, RomStyle, Variant, panda8_with, programs};
use panda_netlist::{Design, Endpoint, InstKind};
use panda_ui::canvas::{Canvas, Highlights};
use panda_ui::doc::Document;

/// One def holding `copies` renamed copies of the flat Panda-8 (MuxTree ROM) body. Only for
/// drawing: not a valid simulation design (it repeats the top-level sources).
fn big_design(copies: usize) -> Design {
    let flat = panda8_with(
        &programs::BENCH_LOOP.image(),
        &Options::new(Variant::Flat).with_rom(RomStyle::MuxTree),
    );
    let src = flat.top_def().unwrap().clone();
    let mut def = src.clone();
    def.instances.clear();
    def.nets.clear();
    def.id = "big".into();
    for c in 0..copies {
        let p = |s: &str| format!("c{c}_{s}");
        for i in &src.instances {
            let mut i = i.clone();
            i.id = p(&i.id);
            def.instances.push(i);
        }
        for n in &src.nets {
            let mut n = n.clone();
            n.id = p(&n.id);
            n.ends = n
                .ends
                .iter()
                .filter_map(|e| match e {
                    Endpoint::Pin(name) if c == 0 => Some(Endpoint::Pin(name.clone())),
                    Endpoint::Pin(_) => None,
                    Endpoint::Port { inst, port } => Some(Endpoint::Port {
                        inst: p(inst),
                        port: port.clone(),
                    }),
                })
                .collect();
            def.nets.push(n);
        }
    }
    Design {
        format: flat.format.clone(),
        defs: vec![def],
        top: "big".into(),
    }
}

#[test]
#[ignore = "benchmark (~10 s): cargo test --release -p panda-ui --test canvas_scale -- --ignored --nocapture"]
fn canvas_scales_to_10k_gates() {
    let design = big_design(6);
    let def = design.top_def().unwrap();
    let gates = def
        .instances
        .iter()
        .filter(|i| {
            matches!(
                &i.kind,
                InstKind::Atom(panda_netlist::Atom::Gate { .. } | panda_netlist::Atom::Not { .. })
            )
        })
        .count();
    let mut doc = Document::new(design);
    let t0 = Instant::now();
    doc.layout_missing();
    let layout = t0.elapsed();
    let errs = panda_ui::layout::check(&doc, "big");
    assert!(
        errs.is_empty(),
        "{} layout problems: {:?}",
        errs.len(),
        &errs[..errs.len().min(5)]
    );
    let n_inst = doc.def("big").unwrap().instances.len();
    eprintln!("{n_inst} instances ({gates} gates) laid out in {layout:?}");
    assert!(gates >= 10_000);

    struct S {
        doc: Document,
        canvas: Canvas,
        hl: Highlights,
    }
    let mut h = Harness::builder()
        .with_size(egui::vec2(1600.0, 1000.0))
        .wgpu()
        .build_ui_state(
            |ui, s: &mut S| {
                egui::CentralPanel::default().show(ui, |ui| {
                    s.canvas.show(ui, &s.doc, "big", None, &s.hl);
                });
            },
            S {
                doc,
                canvas: Canvas::new(),
                hl: Highlights::default(),
            },
        );
    h.step();
    // Zoom levels: fit-all, then a mid zoom and a close-up around the middle.
    let mut report = Vec::new();
    for (name, zoom) in [
        ("fit", None),
        ("far", Some(1.0f32)),
        ("boxes", Some(2.0f32)),
        ("mid", Some(3.0f32)),
        ("close", Some(12.0f32)),
    ] {
        if let Some(z) = zoom {
            let c = &mut h.state_mut().canvas;
            let b = c.scene().unwrap().bounds_rect();
            let mid = b.center();
            c.xf.zoom = z;
            c.xf.origin = egui::pos2(800.0, 500.0) - mid.to_vec2() * z;
        }
        let mut frame = Vec::new();
        let mut build = Vec::new();
        for _ in 0..30 {
            let f0 = Instant::now();
            h.step();
            h.render().unwrap();
            frame.push(f0.elapsed().as_secs_f64() * 1e3);
            build.push(h.state().canvas.stats.build_ms);
        }
        frame.sort_by(f64::total_cmp);
        build.sort_by(f64::total_cmp);
        let st = h.state().canvas.stats;
        let img = h.render().unwrap();
        if matches!(name, "fit" | "boxes") {
            img.save(format!(
                "{}/../../docs/screenshots/foundation-10k-{name}.png",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap();
        }
        eprintln!(
            "{name:>5}: zoom {:.2}, {} items, {} segments, {} vertices, {} texts; mesh build median {:.2} ms (max {:.2}); \
             frame (step+render+readback) median {:.2} ms",
            h.state().canvas.xf.zoom,
            st.items,
            st.segments,
            st.vertices,
            st.texts,
            build[15],
            build[29],
            frame[15]
        );
        report.push(build[15]);
    }
    assert!(
        report.iter().all(|&b| b < 8.0),
        "canvas mesh build too slow: {report:?}"
    );
}
