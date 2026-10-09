//! The JSON designs under `examples-data/` are generated from `panda_cli::examples`.
//! Run `PANDA_BLESS=1 cargo test -p panda-cli --test examples_data` to regenerate them.

use std::path::PathBuf;

use panda_cli::examples;
use panda_netlist::Design;

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples-data")
}

#[test]
fn example_json_files_are_up_to_date() {
    let bless = std::env::var_os("PANDA_BLESS").is_some();
    for name in examples::NAMES {
        let design = examples::by_name(name).unwrap();
        let json = design.to_json() + "\n";
        let path = data_dir().join(format!("{name}.json"));
        if bless {
            std::fs::write(&path, &json).unwrap();
            continue;
        }
        let on_disk = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{}: {e} (run with PANDA_BLESS=1 to generate)", path.display()));
        assert_eq!(on_disk, json, "{} is stale (run with PANDA_BLESS=1 to regenerate)", path.display());
        assert_eq!(Design::from_json(&on_disk).unwrap(), design);
    }
}

#[test]
fn examples_are_valid_and_flatten() {
    for name in examples::NAMES {
        let design = examples::by_name(name).unwrap();
        let (nl, warnings) = panda_cli::flatten_design(&design).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(warnings.is_empty(), "{name}: unexpected warnings {warnings:?}");
        assert!(!nl.gates.is_empty());
    }
    assert!(examples::by_name("nope").is_none());
}
