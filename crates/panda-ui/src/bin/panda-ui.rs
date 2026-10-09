//! The Panda editor (desktop). Usage: `panda-ui [design.panda.json]`.

use panda_ui::app::{AppState, PandaApp};
use panda_ui::doc::Document;
use panda_ui::examples;
use panda_ui::simhost::SimOptions;

fn main() -> eframe::Result {
    let doc = match std::env::args().nth(1) {
        Some(path) => match panda_ui::files::read_document(std::path::Path::new(&path)) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("panda-ui: {e}");
                std::process::exit(1);
            }
        },
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
