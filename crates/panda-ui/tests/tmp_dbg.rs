use std::time::Duration;
use egui::{Event, Modifiers, PointerButton, Pos2};
use egui_kittest::Harness;
use panda_ui::app::{AppState, PandaApp};
use panda_ui::doc::ItemRef;
use panda_ui::examples;
use panda_ui::simhost::SimOptions;

#[test]
fn dbg() {
    let mut state = AppState::new(examples::counter(), SimOptions::default());
    state.wait_layout();
    let mut h = Harness::builder().with_size(egui::vec2(1100.0, 700.0)).wgpu()
        .build_ui_state(|ui, app: &mut PandaApp| app.show(ui), PandaApp::new(state));
    for _ in 0..40 { h.step(); std::thread::sleep(Duration::from_millis(10)); }
    h.state_mut().state.canvas.request_fit();
    for _ in 0..3 { h.step(); }
    let s = &h.state().state;
    let scene = s.canvas.scene().unwrap();
    let p = scene.item(&ItemRef::Pin("EN".into())).unwrap();
    let pos = s.canvas.xf.rect_to_screen(&p.rect()).center();
    let w = s.canvas.xf.to_world(pos);
    eprintln!("pos {pos:?} world {w:?} hit {:?} rect {:?}", scene.hit_item(w).map(|p| p.item.clone()), p.rect());
    eprintln!("live EN {:?}", s.live.values().and_then(|l| l.map.pin_value("EN", l.values)));
    h.hover_at(pos); h.step();
    h.event(Event::PointerButton { pos, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    h.step();
    h.event(Event::PointerButton { pos, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    h.step();
    for _ in 0..40 { h.step(); std::thread::sleep(Duration::from_millis(10)); }
    h.render().unwrap().save("/tmp/claude-1000/-home-toasty-src-panda/48e8d594-7e38-4bcb-9d87-900ef5b51218/scratchpad/dbg.png").unwrap();
    let s = &h.state().state;
    eprintln!("live EN after {:?} log {:?}", s.live.values().and_then(|l| l.map.pin_value("EN", l.values)), s.log);
}
