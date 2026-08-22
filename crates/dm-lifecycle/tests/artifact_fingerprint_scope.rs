#[path = "../artifact_fingerprint_policy.rs"]
mod policy;

#[test]
fn host_runtime_and_lifecycle_sources_are_out_of_scope() {
    assert!(!policy::is_scoped_identity("dm-lifecycle/src/main.rs"));
    assert!(!policy::is_scoped_identity("dm-lifecycle/src/lib.rs"));
    assert!(!policy::is_scoped_identity("dm-runtime/src/lib.rs"));
    assert!(!policy::is_scoped_identity("dm-client/src/main.rs"));
    assert!(!policy::is_scoped_identity(
        "dm-lifecycle/artifact_fingerprint_policy.rs"
    ));
    assert!(policy::is_scoped_identity("dm-compiler/src/lib.rs"));
    assert!(policy::is_scoped_identity("dm-semantics/src/lib.rs"));
}

#[test]
fn compiler_input_and_explicit_vm_revision_change_the_fingerprint() {
    let baseline = policy::fingerprint([
        ("dm-compiler/src/lib.rs", &b"compiler-a"[..]),
        (policy::VM_ARTIFACT_ABI_REVISION, &b"1\n"[..]),
    ]);
    let compiler_change = policy::fingerprint([
        ("dm-compiler/src/lib.rs", &b"compiler-b"[..]),
        (policy::VM_ARTIFACT_ABI_REVISION, &b"1\n"[..]),
    ]);
    let revision_change = policy::fingerprint([
        ("dm-compiler/src/lib.rs", &b"compiler-a"[..]),
        (policy::VM_ARTIFACT_ABI_REVISION, &b"2\n"[..]),
    ]);
    assert_ne!(baseline, compiler_change);
    assert_ne!(baseline, revision_change);
}
