//! Files (owner: the **files** work stream): new / open / save `.panda.json` documents
//! (geometry preserved in each def's `editor` field), dirty tracking with an unsaved-changes
//! prompt, the recent-files list (persisted in eframe's storage), ROM import from a hex image and
//! the built-in examples ([`crate::examples::all`]).
//!
//! * This file: pure I/O and model code (no egui): [`read_document`], [`write_document`],
//!   [`RecentFiles`], [`build_rom`] (ROM defs from hex images).
//! * [`session`]: the GPU-free controller of the *File* menu ([`session::FileSession`]): which
//!   action runs when, the unsaved-changes prompt, Save-as chaining, ROM placement.
//! * [`dialog`]: where paths come from ([`dialog::PathPicker`]): native dialogs via `rfd`
//!   (cargo feature `dialogs`, default on; run on a worker thread so the UI keeps drawing), an
//!   in-app text prompt, or a scripted picker for tests.
//! * [`menu`]: the egui *File* menu panel, its shortcuts and modals.

pub mod dialog;
pub mod menu;
pub mod session;

use std::path::{Path, PathBuf};

use panda_circuits::addr_bits;
use panda_netlist::{ComponentDef, DefId, Design, DesignBuilder, FORMAT};

use crate::doc::Document;

/// File name suffix of Panda documents.
pub const EXTENSION: &str = ".panda.json";

/// Maximum entries in [`RecentFiles`].
pub const MAX_RECENT: usize = 10;

/// Register the *File* menu ([`menu::FileMenu`]) with native dialogs when the `dialogs` feature
/// is on (else in-app path prompts) and recent files persisted in eframe's storage.
pub fn register(reg: &mut crate::app::Registry) {
    reg.panel(menu::FileMenu::new(session::FileSession::new(
        dialog::default_picker(),
    )));
}

/// An empty design: one top def `main` without pins.
pub fn new_design() -> Design {
    Design {
        format: FORMAT.to_string(),
        defs: vec![ComponentDef {
            id: "main".into(),
            name: "main".into(),
            version: 0,
            pins: Vec::new(),
            instances: Vec::new(),
            nets: Vec::new(),
            editor: serde_json::Value::Null,
        }],
        top: "main".into(),
    }
}

/// Read a `panda/1` document from `path` (its path is remembered for saving).
pub fn read_document(path: &Path) -> Result<Document, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut doc = Document::from_json(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    if doc.design.top_def().is_none() {
        return Err(format!(
            "{}: top def '{}' is missing",
            path.display(),
            doc.design.top
        ));
    }
    doc.path = Some(path.to_path_buf());
    Ok(doc)
}

/// Write `doc` (with geometry) to `path` atomically (temporary file + rename); remembers the
/// path and clears the dirty flag.
pub fn write_document(doc: &mut Document, path: &Path) -> Result<(), String> {
    let json = doc.to_json();
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json.as_bytes()).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))?;
    doc.path = Some(path.to_path_buf());
    doc.dirty = false;
    Ok(())
}

/// `path` with [`EXTENSION`] appended unless it already ends in `.json`.
pub fn with_extension(path: &str) -> PathBuf {
    let p = path.trim();
    if p.ends_with(".json") {
        PathBuf::from(p)
    } else {
        PathBuf::from(format!("{p}{EXTENSION}"))
    }
}

/// Most-recently-used document paths (newest first, at most [`MAX_RECENT`]). The *File* menu
/// persists it in eframe's storage ([`menu::store_recent`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecentFiles {
    /// Newest first.
    pub paths: Vec<PathBuf>,
}

impl RecentFiles {
    /// An empty list.
    pub fn in_memory() -> RecentFiles {
        RecentFiles::default()
    }

    /// A list of `paths` (newest first; deduplicated and capped).
    pub fn with_paths(paths: impl IntoIterator<Item = PathBuf>) -> RecentFiles {
        let mut v: Vec<PathBuf> = Vec::new();
        for p in paths {
            if !v.contains(&p) {
                v.push(p);
            }
        }
        v.truncate(MAX_RECENT);
        RecentFiles { paths: v }
    }

    /// Move `path` (canonicalized when possible) to the front; dedup, cap at [`MAX_RECENT`].
    pub fn push(&mut self, path: &Path) {
        let p = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        self.paths.retain(|q| q != &p);
        self.paths.insert(0, p);
        self.paths.truncate(MAX_RECENT);
    }

    /// Drop `path` (e.g. it no longer exists).
    pub fn remove(&mut self, path: &Path) {
        self.paths.retain(|q| q != path);
    }
}

/// A ROM name derived from a file name: letters, digits and `_` only (others become `_`).
pub fn rom_name(file: &Path) -> String {
    let stem = file
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("rom")
        .split('.')
        .next()
        .unwrap_or("rom");
    let s: String = stem
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() {
        "rom".into()
    } else if s.starts_with(|c: char| c.is_ascii_digit()) {
        format!("rom_{s}")
    } else {
        s
    }
}

/// True if `name` is usable as a ROM / def name: non-empty, ASCII letters, digits and `_`, not
/// starting with a digit.
pub fn valid_name(name: &str) -> bool {
    let mut c = name.chars();
    c.next()
        .is_some_and(|f| f.is_ascii_alphabetic() || f == '_')
        && c.all(|x| x.is_ascii_alphanumeric() || x == '_')
}

/// A ROM def built from a hex image ([`build_rom`]), not yet added to a document.
#[derive(Clone, Debug, PartialEq)]
pub struct BuiltRom {
    /// Id of the ROM def (content-addressed: `rom_<name>_<a>x<d>_<hash>`).
    pub def: DefId,
    /// Every def the ROM needs (the ROM and its mux), dependencies first.
    pub defs: Vec<ComponentDef>,
    /// Address bits.
    pub addr_bits: u32,
    /// Data bits.
    pub data_bits: u32,
    /// Words read from the image.
    pub words: usize,
}

/// Build a ROM def named `name` from a hex image ([`panda_circuits::rom_from_hex`]). Address
/// bits fit the image (at least 1); data bits are `data_bits`, or the widest word (at least 1)
/// if `None`. A name that is not [`valid_name`] becomes `rom`.
pub fn build_rom(hex: &str, name: &str, data_bits: Option<u32>) -> Result<BuiltRom, String> {
    let words = panda_circuits::rom_from_hex(hex)?;
    if words.is_empty() {
        return Err("the hex image contains no words".into());
    }
    if words.len() > 1 << 16 {
        return Err(format!(
            "{} words exceed the 64Ki-word ROM limit",
            words.len()
        ));
    }
    let a = addr_bits(words.len()).max(1);
    let widest = words
        .iter()
        .map(|w| 64 - w.leading_zeros())
        .max()
        .unwrap_or(0)
        .max(1);
    let d = data_bits.unwrap_or(widest);
    if !(1..=64).contains(&d) || d < widest {
        return Err(format!(
            "data width {d} does not fit the widest word ({widest} bits)"
        ));
    }
    let name = if valid_name(name) { name } else { "rom" };
    let mut db = DesignBuilder::new();
    let r = panda_circuits::rom(&mut db, name, a, d, &words);
    let built = db.build(&r);
    Ok(BuiltRom {
        def: r.id,
        defs: built.defs,
        addr_bits: a,
        data_bits: d,
        words: words.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples;
    use panda_kernel::Kernel;
    use panda_netlist::Value;

    fn tmp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("panda-ui-files-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn save_open_round_trip_is_identical() {
        let dir = tmp_dir("rt");
        for (k, e) in examples::all().iter().take(3).enumerate() {
            let mut doc = Document::new((e.build)());
            doc.layout_missing();
            let p = dir.join(format!("d{k}{EXTENSION}"));
            write_document(&mut doc, &p).unwrap();
            assert!(!doc.dirty);
            let back = read_document(&p).unwrap();
            assert_eq!(back.design.defs.len(), doc.design.defs.len());
            assert_eq!(back.editors, doc.editors, "{}", e.name);
            assert_eq!(back.to_design(), doc.to_design(), "{}", e.name);
            assert_eq!(back.to_json(), std::fs::read_to_string(&p).unwrap());
            assert_eq!(back.path.as_deref(), Some(p.as_path()));
        }
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn new_design_loads() {
        let mut k = Kernel::default();
        k.load(new_design()).unwrap();
        assert_eq!(with_extension("a/b"), PathBuf::from("a/b.panda.json"));
        assert_eq!(with_extension("x.json"), PathBuf::from("x.json"));
    }

    #[test]
    fn rom_build_reads_words() {
        let mut doc = Document::new(new_design());
        let hex = "// test image\n12 34 @4 ab\n0x7f";
        let name = rom_name(Path::new("/x/my-prog.hex"));
        assert_eq!(name, "my_prog");
        assert_eq!(rom_name(Path::new("9.hex")), "rom_9");
        let b = build_rom(hex, &name, None).unwrap();
        assert_eq!((b.addr_bits, b.data_bits, b.words), (3, 8, 6));
        let ops: Vec<_> = b
            .defs
            .iter()
            .cloned()
            .map(panda_kernel::EditOp::AddDef)
            .collect();
        doc.apply_ops(&ops).unwrap();
        assert!(doc.def(&b.def).unwrap().name.contains("my_prog"));
        let mut d = doc.design.clone();
        d.top = b.def.clone();
        let mut k = Kernel::default();
        k.load(d).unwrap();
        for (a, want) in [(0u64, 0x12u64), (1, 0x34), (2, 0), (4, 0xab), (5, 0x7f)] {
            k.set_input("A", &Value::from_u64(3, a)).unwrap();
            k.step(200, panda_kernel::StepUnit::Tick);
            assert_eq!(k.read_u64("D"), Some(want), "A={a}");
        }
        assert!(build_rom("zz", "r", None).is_err());
        assert!(build_rom("1ff", "r", Some(8)).is_err());
        assert!(build_rom("", "r", None).is_err());
        assert!(valid_name("a_1") && !valid_name("1a") && !valid_name("a-b") && !valid_name(""));
    }
}
