use std::env;
use std::fs;
use std::path::{Path, PathBuf};

mod artifact_fingerprint_policy;

use artifact_fingerprint_policy::{
    COMPILER_INPUT_CRATES, VM_ARTIFACT_ABI_REVISION, fingerprint, is_scoped_identity,
};

fn main() {
    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("Cargo must provide CARGO_MANIFEST_DIR"),
    );
    let workspace_members_root = manifest_dir
        .parent()
        .expect("dm-lifecycle must live below the workspace crates directory");
    let mut inputs = Vec::new();
    for crate_name in COMPILER_INPUT_CRATES {
        let engine_member_root = workspace_members_root.join(crate_name);
        collect_source_inputs(crate_name, &engine_member_root, &mut inputs);
    }
    let revision = workspace_members_root.join(VM_ARTIFACT_ABI_REVISION);
    inputs.push((VM_ARTIFACT_ABI_REVISION.to_owned(), revision));
    debug_assert!(
        inputs
            .iter()
            .all(|(identity, _)| is_scoped_identity(identity))
    );
    inputs.sort_by(|left, right| left.0.cmp(&right.0));

    let mut owned = Vec::with_capacity(inputs.len());
    for (identity, path) in &inputs {
        let bytes = fs::read(path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        owned.push((identity.as_str(), bytes));
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let fingerprint = fingerprint(
        owned
            .iter()
            .map(|(identity, bytes)| (*identity, bytes.as_slice())),
    );
    let generated = format!(
        "pub(crate) const GENERATED_ENGINE_SEMANTICS_FINGERPRINT: [u8; 16] = {fingerprint:?};\n"
    );
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must provide OUT_DIR"))
        .join("engine_semantics_fingerprint.rs");
    if !fs::read_to_string(&output).is_ok_and(|existing| existing == generated) {
        fs::write(&output, generated).expect("failed to write engine semantics fingerprint");
    }

    let target = env::var("TARGET").expect("Cargo must provide TARGET");
    println!("cargo:rustc-env=DREAM64_ENGINE_TARGET={target}");
}

fn collect_source_inputs(crate_name: &str, crate_dir: &Path, output: &mut Vec<(String, PathBuf)>) {
    let manifest = crate_dir.join("Cargo.toml");
    println!("cargo:rerun-if-changed={}", crate_dir.join("src").display());
    output.push((format!("{crate_name}/Cargo.toml"), manifest));
    collect_rust_files(crate_name, crate_dir, &crate_dir.join("src"), output);
}

fn collect_rust_files(
    crate_name: &str,
    crate_dir: &Path,
    directory: &Path,
    output: &mut Vec<(String, PathBuf)>,
) {
    let mut entries = fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", directory.display()))
        .map(|entry| entry.expect("failed to inspect engine source input").path())
        .collect::<Vec<_>>();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect_rust_files(crate_name, crate_dir, &path, output);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            let relative = path
                .strip_prefix(crate_dir)
                .expect("engine source must be below its crate directory")
                .to_string_lossy()
                .replace('\\', "/");
            output.push((format!("{crate_name}/{relative}"), path));
        }
    }
}
