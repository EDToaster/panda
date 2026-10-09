//! The Panda editor (desktop). Usage: `panda-ui [design.panda.json]`.

use panda_ui::app::{AppState, PandaApp};
use panda_ui::doc::Document;
use panda_ui::examples;
use panda_ui::simhost::SimOptions;

fn main() -> eframe::Result {
    let doc = match std::env::args().nth(1) {
        Some(path) => {
            let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
            let mut d = Document::from_json(&text).unwrap_or_else(|e| panic!("{path}: {e}"));
            d.path = Some(path.into());
            d
        }
        None => Document::new(examples::counter()),
    };
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("Panda"),
        ..Default::default()
    };
    eframe::run_native(
        "Panda",
        options,
        Box::new(move |_cc| {
            Ok(Box::new(PandaApp::new(AppState::from_document(
                doc,
                SimOptions::default(),
            ))))
        }),
    )
}
