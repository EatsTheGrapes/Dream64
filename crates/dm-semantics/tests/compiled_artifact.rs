mod common;
use common::*;

#[test]
fn compiled_executable_artifact_round_trips_eager_module_and_semantic_mapping() {
    let compilation = TestProject::compile(
        "/datum/base\n\tproc/value()\n\t\treturn 1\n/datum/child\n\tparent_type = /datum/base\n\tvalue()\n\t\treturn ..() + 1\n/proc/read(datum/child/source)\n\treturn source.value()\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let executable = registry
        .compile_vm_all_symbolic_deferred(&compilation)
        .expect("symbolic module should link")
        .into_fully_eager()
        .expect("fixture procedures should lower eagerly");

    let encoded = executable
        .encode_compiled_artifact()
        .expect("eager executable should encode");
    let segments = executable
        .encode_compiled_artifact_segments()
        .expect("segmented executable should encode");
    assert_eq!(segments.len(), 3);
    assert_eq!(segments.concat(), encoded);
    assert_eq!(
        encoded,
        executable
            .encode_compiled_artifact()
            .expect("encoding should be deterministic")
    );
    let decoded =
        ExecutableProcedures::decode_compiled_artifact(&encoded).expect("executable should decode");
    assert_eq!(decoded.module(), executable.module());
    assert_eq!(decoded.stats(), executable.stats());
    for procedure in registry.procedures() {
        for implementation in &procedure.implementations {
            let before = executable
                .implementation(implementation.id)
                .expect("linked implementation should exist");
            let after = decoded
                .implementation(implementation.id)
                .expect("decoded implementation should exist");
            assert_eq!(
                executable.module().procedure_path(before),
                decoded.module().procedure_path(after)
            );
        }
    }
    assert_eq!(decoded.module().deferred_procedure_count(), 0);

    let mut bad_header = encoded.clone();
    bad_header[0] ^= 0xff;
    assert!(ExecutableProcedures::decode_compiled_artifact(&bad_header).is_err());
    assert!(ExecutableProcedures::decode_compiled_artifact(&encoded[..encoded.len() - 1]).is_err());
    let mut trailing = encoded;
    trailing.push(0);
    assert!(ExecutableProcedures::decode_compiled_artifact(&trailing).is_err());
}
