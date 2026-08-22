//! Portable compiled-artifact fingerprint policy shared by the build script and tests.

/// Frontend and lowering crates whose source can change compiled DM bytecode.
pub const COMPILER_INPUT_CRATES: [&str; 8] = [
    "dm-core",
    "dm-project",
    "dm-lexer",
    "dm-syntax",
    "dm-object-tree",
    "dm-compiler",
    "dm-lowering",
    "dm-semantics",
];

/// Explicit compatibility promise for the portable executable payload.
pub const VM_ARTIFACT_ABI_REVISION: &str = "dm-vm/artifact-abi-revision.txt";

/// Returns whether a workspace-relative input belongs to artifact semantics.
pub fn is_scoped_identity(identity: &str) -> bool {
    identity == VM_ARTIFACT_ABI_REVISION
        || COMPILER_INPUT_CRATES.iter().any(|crate_name| {
            identity == format!("{crate_name}/Cargo.toml")
                || identity.starts_with(&format!("{crate_name}/src/"))
        })
}

/// Hashes already-selected portable semantics inputs deterministically.
pub fn fingerprint<'a>(inputs: impl IntoIterator<Item = (&'a str, &'a [u8])>) -> [u8; 16] {
    let mut inputs = inputs.into_iter().collect::<Vec<_>>();
    inputs.sort_by(|left, right| left.0.cmp(right.0));
    let mut digest = md5::Context::new();
    digest.consume(b"DREAM64-PORTABLE-ARTIFACT-SEMANTICS\0\x01");
    hash_u64(&mut digest, inputs.len() as u64);
    for (identity, bytes) in inputs {
        hash_bytes(&mut digest, identity.as_bytes());
        hash_bytes(&mut digest, bytes);
    }
    digest.compute().0
}

fn hash_u64(context: &mut md5::Context, value: u64) {
    context.consume(value.to_le_bytes());
}

fn hash_bytes(context: &mut md5::Context, bytes: &[u8]) {
    hash_u64(context, bytes.len() as u64);
    context.consume(bytes);
}
