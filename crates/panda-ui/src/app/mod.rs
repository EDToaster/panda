//! The eframe application shell and the shared [`AppState`].
//!
//! Layout: top bar (menus from [`Slot::Menu`] panels, then [`Slot::TopBar`] panels), left and
//! right side panels, a bottom panel (tabs), a status bar, and the canvas in the centre. Slots
//! without registered panels are hidden. Feature modules extend the shell only through
//! [`panel::Registry`] (see [`panel::register_all`]).
//!
//! [`AppState`] is what panels and tools work on: the [`Document`], the [`SimHost`] handle, the
//! current view, live values, waveform history, diagnostics and highlights. Its per-frame
//! [`AppState::update`] drains sim events, reads the latest snapshot (never blocking) and keeps
//! the sim thread's watch set equal to the nets of the visible def.

pub mod panel;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::Instant;

use egui::Ui;
use panda_kernel::{DiagEntry, EditOp};
use panda_netlist::flat::FlatNetlist;
use panda_netlist::{DefId, Design, Diagnostic, InstKind, Logic};

use crate::canvas::{Canvas, Highlights, LiveValues};
use crate::doc::values::ViewMap;
use crate::doc::{Document, EditorData};
use crate::simhost::{
    Command, SimEvent, SimHost, SimOptions, Snapshot, WatchSet, WaveConfig, WaveSignal, WaveStore,
};

pub use panel::{CanvasTool, Panel, Registry, Slot};

/// The current view: which def instance the canvas shows.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ViewState {
    /// Instance path (`""` = top).
    pub path: String,
    /// Def of that instance.
    pub def: DefId,
}

/// Frame timing measured by the shell.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    /// CPU time of the last frame's UI code (ms).
    pub last_ms: f64,
    /// Exponential moving average (ms).
    pub avg_ms: f64,
    /// Frames shown.
    pub frames: u64,
}

/// Live-value state derived from the sim thread.
#[derive(Default)]
pub struct Live {
    /// Latest snapshot (copied each frame; only the visible nets).
    pub snapshot: Snapshot,
    /// Netlist of the current design generation.
    pub flat: Option<Arc<FlatNetlist>>,
    /// Its generation.
    pub design_gen: u64,
    /// Mapping of the current view (built for `design_gen`).
    pub map: Option<ViewMap>,
    /// Watch set id sent for `map`.
    pub watch_id: u64,
}

impl Live {
    /// Live values for the canvas, if the snapshot matches the current mapping.
    pub fn values(&self) -> Option<LiveValues<'_>> {
        let map = self.map.as_ref()?;
        (self.snapshot.watch_id == self.watch_id && self.snapshot.design_gen == self.design_gen)
            .then_some(LiveValues {
                map,
                values: &self.snapshot.values,
            })
    }
}

/// Shared state of the editor.
pub struct AppState {
    /// The open document.
    pub doc: Document,
    /// The simulation thread.
    pub sim: SimHost,
    /// Current view.
    pub view: ViewState,
    /// Live values.
    pub live: Live,
    /// The canvas.
    pub canvas: Canvas,
    /// Highlights drawn by the canvas (selection, hovered nets, diagnostics).
    pub highlights: Highlights,
    /// Waveform history of the watched signals.
    pub wave: WaveStore,
    /// Latest diagnostics list from the kernel.
    pub diagnostics: Arc<Vec<DiagEntry>>,
    /// Messages for the user (newest last).
    pub log: Vec<String>,
    /// Frame timing.
    pub frame: FrameStats,
    layout_job: Option<Receiver<HashMap<DefId, EditorData>>>,
    watched_key: Option<(u64, String, u64)>,
    next_id: u64,
}

impl AppState {
    /// Start the sim thread and open `design`.
    pub fn new(design: Design, opts: SimOptions) -> AppState {
        AppState::from_document(Document::new(design), opts)
    }

    /// Start the sim thread and open `doc` (keeps its path and geometry).
    pub fn from_document(doc: Document, opts: SimOptions) -> AppState {
        let mut s = AppState {
            doc: Document::new(Design {
                format: doc.design.format.clone(),
                defs: Vec::new(),
                top: String::new(),
            }),
            sim: SimHost::spawn(opts),
            view: ViewState::default(),
            live: Live::default(),
            canvas: Canvas::new(),
            highlights: Highlights::default(),
            wave: WaveStore::default(),
            diagnostics: Arc::new(Vec::new()),
            log: Vec::new(),
            frame: FrameStats::default(),
            layout_job: None,
            watched_key: None,
            next_id: 1,
        };
        s.open_document(doc);
        s
    }

    /// A fresh id for commands / watch sets.
    pub fn next_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    /// Replace the document by `design`: load it into the kernel, view its top, and lay out
    /// defs without geometry on a worker thread.
    pub fn open_design(&mut self, design: Design) {
        self.open_document(Document::new(design));
    }

    /// Open a [`Document`] (keeps its path and geometry; lays out defs that have none).
    pub fn open_document(&mut self, doc: Document) {
        let design = doc.to_design();
        self.view = ViewState {
            path: String::new(),
            def: design.top.clone(),
        };
        self.doc = doc;
        self.live = Live::default();
        self.watched_key = None;
        self.wave = WaveStore::default();
        self.highlights = Highlights::default();
        self.canvas.request_fit();
        self.sim.send(Command::Load(Box::new(design)));
        self.start_layout();
    }

    fn start_layout(&mut self) {
        let missing = self.doc.defs_missing_layout();
        if missing.is_empty() {
            self.layout_job = None;
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let design = self.doc.design.clone();
        let editors = self.doc.editors.clone();
        std::thread::Builder::new()
            .name("panda-layout".into())
            .spawn(move || {
                let _ = tx.send(crate::layout::layout_defs(&design, &editors, &missing));
            })
            .expect("spawn layout thread");
        self.layout_job = Some(rx);
    }

    /// True while auto-layout runs.
    pub fn layout_pending(&self) -> bool {
        self.layout_job.is_some()
    }

    /// Block until auto-layout finished (tests, headless tools).
    pub fn wait_layout(&mut self) {
        if let Some(rx) = self.layout_job.take()
            && let Ok(l) = rx.recv()
        {
            self.doc.install_layouts(l);
        }
    }

    /// View the def instance at `path` (`""` = top). False if the path does not exist.
    pub fn enter(&mut self, path: &str) -> bool {
        let Some(def) = def_at_path(&self.doc.design, path) else {
            return false;
        };
        self.view = ViewState {
            path: path.to_string(),
            def,
        };
        self.highlights = Highlights::default();
        true
    }

    /// Apply structural edits to the document and the kernel (the kernel answers with
    /// [`SimEvent::Edited`] carrying the returned id). On a local error nothing changes.
    pub fn submit_edit(&mut self, ops: Vec<EditOp>) -> Result<u64, Diagnostic> {
        self.doc.apply_ops(&ops)?;
        let id = self.next_id();
        self.sim.send(Command::Edit { id, ops });
        Ok(id)
    }

    /// Watch waveform signals by hierarchical net name (e.g. `"PC"`, `"cpu/alu/c"`): every bit
    /// of the named net. Unknown names are skipped. Returns the number of signals watched.
    pub fn set_wave_signals(&mut self, names: &[&str]) -> usize {
        let Some(flat) = self.live.flat.clone() else {
            return 0;
        };
        let signals: Vec<WaveSignal> = names
            .iter()
            .filter_map(|&name| {
                let nets = resolve_net(&flat, name)?;
                Some(WaveSignal::new(&flat, name, nets))
            })
            .collect();
        let n = signals.len();
        let cfg = WaveConfig {
            id: self.next_id(),
            design_gen: self.live.design_gen,
            signals,
        };
        self.wave = WaveStore::new(cfg.clone());
        self.sim.send(Command::SetWave(cfg));
        n
    }

    /// Per-frame bookkeeping: install finished layouts, drain sim events, refresh the watch
    /// set, read the latest snapshot. Returns the events for the panels.
    pub fn update(&mut self) -> Vec<SimEvent> {
        if let Some(rx) = &self.layout_job {
            match rx.try_recv() {
                Ok(l) => {
                    self.doc.install_layouts(l);
                    self.layout_job = None;
                }
                Err(TryRecvError::Disconnected) => self.layout_job = None,
                Err(TryRecvError::Empty) => {}
            }
        }
        let events = self.sim.poll_events();
        for ev in &events {
            match ev {
                SimEvent::Netlist { design_gen, flat } => {
                    self.live.flat = Some(flat.clone());
                    self.live.design_gen = *design_gen;
                    self.live.map = None;
                    self.watched_key = None;
                    // Re-resolve waveform signals by name for the new netlist.
                    let names: Vec<String> = self
                        .wave
                        .config
                        .signals
                        .iter()
                        .map(|s| s.name.clone())
                        .collect();
                    if !names.is_empty() {
                        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
                        self.set_wave_signals(&refs);
                    }
                }
                SimEvent::LoadFailed(d) => {
                    self.log.push(format!("load failed: {} error(s)", d.len()))
                }
                SimEvent::Edited { result: Err(d), .. } => self.log.push(format!(
                    "edit rejected by the kernel: {}",
                    d.first().map_or(String::new(), ToString::to_string)
                )),
                SimEvent::Error(e) => self.log.push(e.clone()),
                SimEvent::Diagnostics(d) => self.diagnostics = d.clone(),
                SimEvent::Wave(c) => self.wave.append(c),
                _ => {}
            }
        }
        self.refresh_watch();
        let snap = self.sim.snapshot();
        self.live.snapshot.clone_from(snap);
        events
    }

    fn refresh_watch(&mut self) {
        let Some(flat) = self.live.flat.clone() else {
            return;
        };
        let key = (
            self.live.design_gen,
            self.view.path.clone(),
            self.doc.revision,
        );
        if self.watched_key.as_ref() == Some(&key) {
            return;
        }
        let Some(def) = self.doc.def(&self.view.def) else {
            return;
        };
        let map = ViewMap::build(&flat, &self.view.path, def);
        let id = self.next_id();
        self.sim.send(Command::SetWatch(WatchSet {
            id,
            design_gen: self.live.design_gen,
            nets: map.watch.clone(),
        }));
        self.live.map = Some(map);
        self.live.watch_id = id;
        self.watched_key = Some(key);
    }
}

/// Every bit of net `name` (a 1-bit net, or `name[0]`, `name[1]`, …).
pub fn resolve_net(flat: &FlatNetlist, name: &str) -> Option<Vec<panda_netlist::flat::NetIdx>> {
    if let Some(n) = flat.net_by_name(name) {
        return Some(vec![n]);
    }
    let mut v = Vec::new();
    while let Some(n) = flat.net_by_name(&format!("{name}[{}]", v.len())) {
        v.push(n);
    }
    (!v.is_empty()).then_some(v)
}

/// Def of the instance at `path` (`""` = top) by walking instance ids.
pub fn def_at_path(design: &Design, path: &str) -> Option<DefId> {
    let mut def = design.top_def()?;
    if path.is_empty() {
        return Some(def.id.clone());
    }
    for id in path.split('/') {
        let inst = def.instances.iter().find(|i| i.id == id)?;
        let InstKind::Def(d) = &inst.kind else {
            return None;
        };
        def = design.def(d)?;
    }
    Some(def.id.clone())
}

/// The eframe app: shared state plus registered panels and tools.
pub struct PandaApp {
    /// Shared state.
    pub state: AppState,
    /// Panels and tools.
    pub registry: Registry,
    tabs: HashMap<Slot, usize>,
}

impl PandaApp {
    /// App with every feature module registered.
    pub fn new(state: AppState) -> PandaApp {
        let mut registry = Registry::default();
        panel::register_all(&mut registry);
        PandaApp::with_registry(state, registry)
    }

    /// App with an explicit registry.
    pub fn with_registry(state: AppState, registry: Registry) -> PandaApp {
        PandaApp {
            state,
            registry,
            tabs: HashMap::new(),
        }
    }

    /// Draw one frame into `ui` (the root ui of eframe, or a kittest harness).
    pub fn show(&mut self, ui: &mut Ui) {
        let t0 = Instant::now();
        let events = self.state.update();
        for p in &mut self.registry.panels {
            p.on_frame(ui.ctx(), &mut self.state);
        }
        for ev in &events {
            for p in &mut self.registry.panels {
                p.on_event(ev, &mut self.state);
            }
        }
        self.top_bar(ui);
        self.status_bar(ui);
        self.side(ui, Slot::Left);
        self.side(ui, Slot::Right);
        self.side(ui, Slot::Bottom);
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| self.central(ui));
        if self.state.live.snapshot.running || self.state.layout_pending() {
            ui.ctx().request_repaint();
        } else {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(100));
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let f = &mut self.state.frame;
        f.last_ms = ms;
        f.avg_ms = if f.frames == 0 {
            ms
        } else {
            f.avg_ms * 0.95 + ms * 0.05
        };
        f.frames += 1;
    }

    fn top_bar(&mut self, ui: &mut Ui) {
        let menus = self.registry.in_slot(Slot::Menu);
        let inline = self.registry.in_slot(Slot::TopBar);
        let state = &mut self.state;
        let panels = &mut self.registry.panels;
        egui::Panel::top("panda_top").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.strong("Panda");
                ui.separator();
                for &i in &menus {
                    let title = panels[i].title();
                    ui.menu_button(title, |ui| panels[i].ui(ui, state));
                }
                if !menus.is_empty() {
                    ui.separator();
                }
                for &i in &inline {
                    panels[i].ui(ui, state);
                }
                if inline.is_empty() {
                    let path = if state.view.path.is_empty() {
                        "/".to_string()
                    } else {
                        format!("/{}", state.view.path)
                    };
                    ui.label(format!("{}  ({})", path, state.view.def));
                }
            });
        });
    }

    fn status_bar(&mut self, ui: &mut Ui) {
        let status = self.registry.in_slot(Slot::Status);
        let state = &mut self.state;
        let panels = &mut self.registry.panels;
        egui::Panel::bottom("panda_status").show(ui, |ui| {
            ui.horizontal(|ui| {
                let s = &state.live.snapshot;
                let engine = s.tier_label();
                ui.label(format!(
                    "tick {}  ·  {:?}  ·  {}  ·  {}  ·  {:.0} cycles/s  ·  frame {:.1} ms",
                    s.tick,
                    s.mode,
                    engine,
                    if s.running { "running" } else { "paused" },
                    s.stats.cycles_per_sec,
                    state.frame.avg_ms
                ));
                if state.layout_pending() {
                    ui.separator();
                    ui.label("laying out…");
                }
                for &i in &status {
                    ui.separator();
                    panels[i].ui(ui, state);
                }
            });
        });
    }

    fn side(&mut self, ui: &mut Ui, slot: Slot) {
        let ids = self.registry.in_slot(slot);
        if ids.is_empty() {
            return;
        }
        let panel = match slot {
            Slot::Left => egui::Panel::left("panda_left")
                .default_size(220.0)
                .resizable(true),
            Slot::Right => egui::Panel::right("panda_right")
                .default_size(260.0)
                .resizable(true),
            _ => egui::Panel::bottom("panda_bottom")
                .default_size(200.0)
                .resizable(true),
        };
        let tab = self.tabs.entry(slot).or_insert(0);
        let state = &mut self.state;
        let panels = &mut self.registry.panels;
        panel.show(ui, |ui| {
            if ids.len() > 1 {
                ui.horizontal(|ui| {
                    for (k, &i) in ids.iter().enumerate() {
                        ui.selectable_value(tab, k, panels[i].title());
                    }
                });
                ui.separator();
            }
            let i = ids[(*tab).min(ids.len() - 1)];
            panels[i].ui(ui, state);
        });
    }

    fn central(&mut self, ui: &mut Ui) {
        let state = &mut self.state;
        if state.doc.editor(&state.view.def).is_none() {
            ui.centered_and_justified(|ui| {
                ui.label(if state.layout_pending() {
                    "Laying out…"
                } else {
                    "No geometry for this def"
                });
            });
            return;
        }
        state.canvas.primary_pans = !self.registry.tools.iter().any(|t| t.claims_primary(state));
        let live = state.live.values();
        let resp = state
            .canvas
            .show(ui, &state.doc, &state.view.def, live, &state.highlights);
        let scene = state.canvas.scene().expect("shown").clone();
        for t in &mut self.registry.tools {
            t.canvas(ui, &resp, &scene, state);
        }
    }
}

impl eframe::App for PandaApp {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        self.show(ui);
    }
}

/// Logic values of a watch list (helper for panels).
pub fn logic_bits(values: &[Logic], idx: &[Option<u32>]) -> Vec<Logic> {
    idx.iter()
        .map(|i| {
            i.and_then(|i| values.get(i as usize).copied())
                .unwrap_or(Logic::X)
        })
        .collect()
}
