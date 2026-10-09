//! Files (owner: the **hierarchy/files** work stream): new / open / save `.panda.json`
//! documents (geometry preserved in each def's `editor` field), the recent-files list, ROM
//! import from a hex image and the built-in examples ([`crate::examples::all`]).
//!
//! This module is pure I/O and model code; the *File* menu that drives it lives in
//! [`crate::hier`] (it shares that module's state: undo history, library, prompts).
//!
//! File dialogs: the editor asks for paths with an in-app prompt (no desktop portal needed, so
//! tests and headless runs work). A native dialog (e.g. the `rfd` crate) can be plugged in
//! behind a cargo feature without touching this module: it only needs to produce a path for
//! [`read_document`] / [`write_document`] / [`import_rom`].

use std::path::{Path, PathBuf};

use panda_circuits::addr_bits;
use panda_netlist::{ComponentDef, DefId, Design, DesignBuilder, FORMAT};

use crate::doc::Document;
use crate::hier::change::DocChange;

/// File name suffix of Panda documents.
pub const EXTENSION: &str = ".panda.json";

/// Maximum entries in [`RecentFiles`].
pub const MAX_RECENT: usize = 10;

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

/// Most-recently-used document paths, optionally persisted as JSON.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecentFiles {
    /// Newest first.
    pub paths: Vec<PathBuf>,
    /// Where the list is stored (`None` = in memory only).
    pub store: Option<PathBuf>,
}

impl RecentFiles {
    /// An in-memory list (tests).
    pub fn in_memory() -> RecentFiles {
        RecentFiles::default()
    }

    /// Load from `store` (missing or unreadable → empty list that will be saved there).
    pub fn load(store: PathBuf) -> RecentFiles {
        let paths = std::fs::read_to_string(&store)
            .ok()
            .and_then(|s| serde_json::from_str::<Vec<PathBuf>>(&s).ok())
            .unwrap_or_default();
        RecentFiles {
            paths,
            store: Some(store),
        }
    }

    /// The default store: `$XDG_CONFIG_HOME/panda/recent.json` (or `~/.config/…`).
    pub fn default_store() -> Option<PathBuf> {
        let base = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
        Some(base.join("panda").join("recent.json"))
    }

    /// Move `path` to the front (dedup, cap at [`MAX_RECENT`]) and persist.
    pub fn push(&mut self, path: &Path) {
        let p = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        self.paths.retain(|q| q != &p);
        self.paths.insert(0, p);
        self.paths.truncate(MAX_RECENT);
        self.save();
    }

    /// Drop `path` (e.g. it no longer exists) and persist.
    pub fn remove(&mut self, path: &Path) {
        self.paths.retain(|q| q != path);
        self.save();
    }

    fn save(&self) {
        let Some(store) = &self.store else { return };
        if let Some(dir) = store.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(s) = serde_json::to_string_pretty(&self.paths) {
            let _ = std::fs::write(store, s);
        }
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
    if s.is_empty() { "rom".into() } else { s }
}

/// Result of [`import_rom`].
#[derive(Clone, Debug, PartialEq)]
pub struct RomImport {
    /// The change adding the ROM def (and the mux def it uses, if new).
    pub change: DocChange,
    /// Id of the ROM def.
    pub def: DefId,
    /// Address bits.
    pub addr_bits: u32,
    /// Data bits.
    pub data_bits: u32,
    /// Words read.
    pub words: usize,
}

/// Build a ROM def from a hex image ([`panda_circuits::rom_from_hex`]) as a change adding it to
/// `doc`. Address bits fit the image (at least 1); data bits are `data_bits`, or the widest word
/// (at least 1) if `None`.
pub fn import_rom(
    doc: &Document,
    hex: &str,
    name: &str,
    data_bits: Option<u32>,
) -> Result<RomImport, String> {
    let words = panda_circuits::rom_from_hex(hex)?;
    if words.is_empty() {
        return Err("the hex image contains no words".into());
    }
    if words.len() > 1 << 16 {
        return Err(format!("{} words exceed the 64Ki-word ROM limit", words.len()));
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
    let name = if crate::hier::package::valid_ident(name) {
        name.to_string()
    } else {
        "rom".to_string()
    };
    let mut db = DesignBuilder::new();
    let r = panda_circuits::rom(&mut db, &name, a, d, &words);
    let built = db.build(&r);
    let new_defs: Vec<ComponentDef> = built
        .defs
        .into_iter()
        .filter(|x| doc.design.def(&x.id).is_none())
        .collect();
    let change = crate::hier::library::add_defs_change(doc, &new_defs, &format!("Import ROM {name}"))
        .ok_or_else(|| format!("the document already has ROM '{}'", r.id))?;
    Ok(RomImport {
        change,
        def: r.id,
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
    fn recent_files_dedup_cap_persist() {
        let dir = tmp_dir("recent");
        let store = dir.join("cfg/recent.json");
        let mut r = RecentFiles::load(store.clone());
        assert!(r.paths.is_empty());
        for i in 0..14 {
            r.push(&dir.join(format!("f{i}.panda.json")));
        }
        r.push(&dir.join("f3.panda.json"));
        assert_eq!(r.paths.len(), MAX_RECENT);
        assert!(r.paths[0].ends_with("f3.panda.json"));
        assert_eq!(r.paths.iter().filter(|p| p.ends_with("f3.panda.json")).count(), 1);
        let again = RecentFiles::load(store);
        assert_eq!(again.paths, r.paths);
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
    fn rom_import_reads_words() {
        let mut doc = Document::new(new_design());
        let hex = "// test image\n12 34 @4 ab\n0x7f";
        let imp = import_rom(&doc, hex, &rom_name(Path::new("/x/my-prog.hex")), None).unwrap();
        assert_eq!((imp.addr_bits, imp.data_bits, imp.words), (3, 8, 6));
        imp.change.apply_doc(&mut doc).unwrap();
        assert!(doc.def(&imp.def).is_some());
        assert!(doc.def(&imp.def).unwrap().name.contains("my_prog"));
        assert!(doc.editor(&imp.def).is_some());
        // Read the ROM through a kernel: wrap it as the top.
        let mut d = doc.design.clone();
        d.top = imp.def.clone();
        let mut k = Kernel::default();
        k.load(d).unwrap();
        for (a, want) in [(0u64, 0x12u64), (1, 0x34), (2, 0), (4, 0xab), (5, 0x7f)] {
            k.set_input("A", &Value::from_u64(3, a)).unwrap();
            k.step(200, panda_kernel::StepUnit::Tick);
            assert_eq!(k.read_u64("D"), Some(want), "A={a}");
        }
        imp.change.revert_doc(&mut doc).unwrap();
        assert_eq!(doc.design, new_design());
        assert!(import_rom(&doc, "zz", "r", None).is_err());
        assert!(import_rom(&doc, "1ff", "r", Some(8)).is_err());
    }
}
