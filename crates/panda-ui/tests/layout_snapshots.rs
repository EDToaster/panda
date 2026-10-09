//! Offscreen renders of auto-laid-out designs and of package symbols in every orientation.
//!
//! Writes `docs/screenshots/layout-<name>.png` (set `LAYOUT_SHOT_TAG` to prefix the names, e.g.
//! for before/after comparisons; `LAYOUT_SHOT_DIR` overrides the directory). Inspect them after changing the layout or the glyphs.

use std::path::PathBuf;

use egui_kittest::Harness;
use panda_netlist::{DefBuilder, Design, DesignBuilder};
use panda_ui::canvas::{Canvas, Highlights};
use panda_ui::doc::{Document, InstGeom, ItemRef};
use panda_ui::geom::{Orient, gp};

fn shot_path(name: &str) -> PathBuf {
    let tag = std::env::var("LAYOUT_SHOT_TAG").unwrap_or_default();
    let tag = if tag.is_empty() { tag } else { format!("{tag}-") };
    let dir = std::env::var("LAYOUT_SHOT_DIR").map_or_else(
        |_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/screenshots"),
        PathBuf::from,
    );
    dir.join(format!("layout-{tag}{name}.png"))
}

struct View {
    doc: Document,
    def: String,
    canvas: Canvas,
    hl: Highlights,
}

fn harness(doc: Document, def: &str, w: f32, h: f32) -> Harness<'static, View> {
    let mut h = Harness::builder()
        .with_size(egui::vec2(w, h))
        .wgpu()
        .build_ui_state(
            |ui, v: &mut View| {
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(ui, |ui| {
                        v.canvas.show(ui, &v.doc, &v.def, None, &v.hl);
                    });
            },
            View {
                doc,
                def: def.to_string(),
                canvas: Canvas::new(),
                hl: Highlights::default(),
            },
        );
    h.step();
    h.step();
    h
}

fn save(h: &mut Harness<'_, View>, name: &str) {
    h.step();
    let img = h.render().expect("render");
    let p = shot_path(name);
    std::fs::create_dir_all(p.parent().expect("dir")).expect("mkdir");
    img.save(&p).expect("save");
}

/// Zoom the view to `zoom` points per grid unit centred on world point `at`.
fn look_at(h: &mut Harness<'_, View>, at: egui::Pos2, zoom: f32) {
    let size = h.ctx.content_rect().size();
    let c = &mut h.state_mut().canvas;
    c.xf.zoom = zoom;
    c.xf.origin = egui::pos2(size.x / 2.0, size.y / 2.0) - at.to_vec2() * zoom;
}

fn laid_out(design: Design) -> Document {
    let mut doc = Document::new(design);
    doc.layout_missing();
    doc
}

#[test]
fn panda8_views() {
    let doc = laid_out(panda_ui::examples::panda8());
    let mut h = harness(doc.clone(), &doc.design.top, 1400.0, 860.0);
    save(&mut h, "panda8-top");
    let core = "panda8_core";
    let mut h = harness(doc.clone(), core, 1400.0, 860.0);
    save(&mut h, "panda8-core");
    // A readable zoom on the core's left half.
    let b = h.state().canvas.scene().expect("scene").bounds_rect();
    look_at(&mut h, egui::pos2(b.min.x + b.width() * 0.3, b.center().y), 9.0);
    save(&mut h, "panda8-core-zoom");
}

#[test]
fn panda8_flat_views() {
    let design = panda_cpu8::panda8(
        &panda_cpu8::programs::BENCH_LOOP.image(),
        panda_cpu8::Variant::Flat,
    );
    let doc = laid_out(design);
    let top = doc.design.top.clone();
    let mut h = harness(doc.clone(), &top, 1400.0, 860.0);
    save(&mut h, "flat-fit");
    // Around the ALU's adder and the register file.
    for (name, prefix, zoom) in [
        ("flat-alu", "cpu.alu.", 5.0),
        ("flat-regfile", "cpu.rf.", 2.5),
        ("flat-rom", "rom.", 0.25),
        ("flat-cpu", "cpu.", 0.9),
        ("flat-control", "cpu.ctl.", 7.0),
    ] {
        let pos: Vec<_> = doc.editors[&top]
            .instances
            .iter()
            .filter(|(id, _)| id.starts_with(prefix))
            .map(|(_, g)| g.pos)
            .collect();
        assert!(!pos.is_empty(), "{prefix}");
        let cx = pos.iter().map(|p| p.x as f32).sum::<f32>() / pos.len() as f32;
        let cy = pos.iter().map(|p| p.y as f32).sum::<f32>() / pos.len() as f32;
        look_at(&mut h, egui::pos2(cx, cy), zoom);
        save(&mut h, name);
    }
}

/// A package with pins on all four sides, instantiated in all eight orientations.
#[test]
fn package_orientations() {
    let mut pkg = DefBuilder::new("alu4");
    let ins: Vec<_> = ["A", "B", "CIN", "SEL"]
        .iter()
        .map(|n| pkg.input(n, 1))
        .collect();
    let y = pkg.and(&ins[..2]);
    let c = pkg.or(&ins[1..3]);
    let z = pkg.xor(&ins[2..4]);
    pkg.output("Y", y);
    pkg.output("COUT", c);
    pkg.output("ZERO", z);
    // A second package keeps the automatic symbol (title band, pins left and right).
    let mut reg = DefBuilder::new("register_8");
    let rins: Vec<_> = ["D", "LOAD", "RST", "CLK"]
        .iter()
        .map(|n| reg.input(n, if *n == "D" { 8 } else { 1 }))
        .collect();
    let q = reg.buf(rins[0]);
    reg.output("Q", q);
    let mut db = DesignBuilder::new();
    let pkg = db.add(pkg.finish());
    let reg = db.add(reg.finish());
    let mut top = DefBuilder::new("top");
    let tins: Vec<_> = ["a", "b", "cin", "sel"]
        .iter()
        .map(|n| top.input(n, 1))
        .collect();
    for i in 0..8 {
        top.inst_named(&format!("u{i}"), &pkg, &tins);
    }
    let d8 = top.input("d", 8);
    for i in 0..8 {
        top.inst_named(&format!("r{i}"), &reg, &[d8, tins[0], tins[1], tins[2]]);
    }
    let top = db.add(top.finish());
    let design = db.build(&top);
    let mut doc = laid_out(design);
    // Pins on all sides: SEL on top, ZERO at the bottom.
    let sym = doc.editors.get_mut("alu4").expect("pkg").symbol.get_or_insert_with(|| {
        panda_ui::geom::auto_symbol_layout(doc.design.def("alu4").expect("def"))
    });
    use panda_ui::geom::{Side, SymbolPin};
    sym.size = gp(sym.size.x.max(10), sym.size.y + 2);
    sym.pins.insert("SEL".into(), SymbolPin { side: Side::Top, offset: 4 });
    sym.pins.insert("ZERO".into(), SymbolPin { side: Side::Bottom, offset: 6 });
    let ed = doc.editors.get_mut("top").expect("top");
    ed.wires.clear();
    for (prefix, y0) in [("u", 4), ("r", 48)] {
        for i in 0..8 {
            let o = Orient {
                rot: (i % 4) as u8,
                flip: i >= 4,
            };
            ed.instances.insert(
                format!("{prefix}{i}"),
                InstGeom {
                    pos: gp(4 + (i % 4) * 22, y0 + (i / 4) * 20),
                    orient: o,
                },
            );
        }
    }
    for (k, g) in ed.pins.values_mut().enumerate() {
        g.pos = gp(96, 4 + 2 * k as i32);
    }
    doc.revision += 1;
    let items = doc.placed("top");
    assert_eq!(
        items.iter().filter(|p| matches!(p.item, ItemRef::Instance(_))).count(),
        16
    );
    let mut h = harness(doc, "top", 1100.0, 900.0);
    save(&mut h, "package-orientations");
}
