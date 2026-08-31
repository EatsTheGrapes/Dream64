mod common;
use common::*;

#[test]
fn indexes_a_base_procedure_with_source_identity() {
    let compilation = TestProject::compile("/datum/base\n\tproc/run()\n\t\treturn 1\n");
    let registry = ProcedureRegistry::build(&compilation);
    let procedure = procedure_by_path(&registry, "/datum/base/proc/run");

    assert!(procedure.owner_type.is_some());
    assert_eq!(procedure.implementations.len(), 1);
    assert_eq!(
        procedure.implementations[0].kind,
        ProcedureImplementationKind::Declaration
    );
    assert_eq!(procedure.implementations[0].definition_index, 1);
    assert!(!procedure.implementations[0].span.is_empty());
    assert_eq!(procedure.implementations[0].parent_target, None);
    assert_eq!(
        procedure.effective_target,
        Some(procedure.implementations[0].id)
    );
}

#[test]
fn indexes_a_global_procedure_without_an_owner_or_parent() {
    let compilation = TestProject::compile("/proc/global_run()\n\treturn 1\n");
    let registry = ProcedureRegistry::build(&compilation);
    let procedure = procedure_by_path(&registry, "/proc/global_run");

    assert_eq!(procedure.owner_type, None);
    assert_eq!(procedure.inherited_procedure, None);
    assert_eq!(procedure.implementations[0].parent_target, None);
}

#[test]
fn follows_a_resolved_explicit_parent_type() {
    let compilation = TestProject::compile(
        "/datum/original\n\tproc/run()\n\t\treturn 1\n/datum/alternate\n\tproc/run()\n\t\treturn 2\n/custom\n\tparent_type = /datum/alternate\n\trun()\n\t\treturn ..()\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let original = procedure_by_path(&registry, "/datum/original/proc/run");
    let alternate = procedure_by_path(&registry, "/datum/alternate/proc/run");
    let custom = procedure_by_path(&registry, "/custom/proc/run");

    assert_ne!(custom.inherited_procedure, Some(original.id));
    assert_eq!(custom.inherited_procedure, Some(alternate.id));
    assert_eq!(
        custom.implementations[0].parent_target,
        alternate.effective_target
    );
}

#[test]
fn links_a_child_override_to_the_inherited_effective_body() {
    let compilation = TestProject::compile(
        "/datum/base\n\tproc/run()\n\t\treturn 1\n/datum/base/child\n\trun()\n\t\treturn ..()\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let base = procedure_by_path(&registry, "/datum/base/proc/run");
    let child = procedure_by_path(&registry, "/datum/base/child/proc/run");

    assert_eq!(child.inherited_procedure, Some(base.id));
    assert_eq!(child.implementations.len(), 1);
    assert_eq!(
        child.implementations[0].parent_target,
        base.effective_target
    );
}

#[test]
fn chains_multiple_reopenings_in_expanded_source_order() {
    let compilation = TestProject::compile(
        "/datum/base\n\tproc/run()\n\t\treturn 1\n/datum/base\n\trun()\n\t\treturn 2\n/datum/base\n\trun()\n\t\treturn 3\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let procedure = procedure_by_path(&registry, "/datum/base/proc/run");

    assert_eq!(procedure.implementations.len(), 3);
    assert!(
        procedure
            .implementations
            .windows(2)
            .all(|pair| pair[0].ordinal < pair[1].ordinal)
    );
    assert_eq!(procedure.implementations[0].parent_target, None);
    assert_eq!(
        procedure.implementations[1].parent_target,
        Some(procedure.implementations[0].id)
    );
    assert_eq!(
        procedure.implementations[2].parent_target,
        Some(procedure.implementations[1].id)
    );
    assert_eq!(
        procedure.effective_target,
        Some(procedure.implementations[2].id)
    );
}

#[test]
fn independent_body_compilation_links_known_external_calls_to_stubs() {
    let compilation = TestProject::compile(
        "/proc/helper()\n\treturn 1\n/proc/caller()\n\treturn \"[helper()]\"\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let caller = procedure_by_path(&registry, "/proc/caller")
        .effective_target
        .expect("caller implementation should exist");
    let results = registry.compile_vm_bodies_independently(&compilation, [caller]);
    assert_eq!(results.len(), 1);
    results
        .into_iter()
        .next()
        .expect("caller result should exist")
        .1
        .expect("known external call should lower through an inert inventory stub");
}

#[test]
fn avd_empty_variadic_signature_preserves_rhs_input_constraints() {
    let compilation = TestProject::compile(concat!(
        "/datum/admin_verb/set_server_fps/__avd_do_verb(client/user,)\n",
        "\tvar/cfg_fps = 20\n",
        "\tvar/new_fps = round(input(user, \"FPS\", \"FPS\", 20) as num | null)\n",
        "\tif(new_fps <= 0)\n",
        "\t\treturn cfg_fps\n",
        "\treturn new_fps\n",
    ));
    let registry = ProcedureRegistry::build(&compilation);
    let executable = registry
        .compile_vm_all_symbolic_deferred(&compilation)
        .expect("AVD-shaped symbolic module should link")
        .into_fully_eager()
        .expect("RHS input constraints must survive semantic normalization");

    assert_eq!(executable.module().deferred_procedure_count(), 0);
    assert!(
        executable.module().procedure_paths().any(|path| {
            path.starts_with("/datum/admin_verb/set_server_fps/proc/__avd_do_verb@")
        })
    );
}

#[test]
fn lazy_registry_matches_eager_dependencies_and_defers_body_analysis() {
    let compilation = TestProject::compile(concat!(
        "/datum/base/proc/New()\n",
        "/datum/base/proc/ping()\n\treturn 1\n",
        "/datum/base/child/New()\n\t..()\n",
        "/datum/base/child/ping()\n\treturn 2\n",
        "/datum/runner/proc/run(datum/base/value)\n",
        "\tfor(var/path in typesof(/datum/base/proc))\n",
        "\t\tcall(value, path)()\n",
        "\tvar/datum/base/child/item = new\n",
        "\treturn item.ping()\n",
    ));
    let eager = ProcedureRegistry::build(&compilation);
    let lazy = ProcedureRegistry::build_lazy(&compilation);
    assert!(!lazy.dependencies_initialized());
    assert_eq!(lazy.procedures(), eager.procedures());
    let root = procedure_by_path(&eager, "/datum/runner/proc/run")
        .effective_target
        .unwrap();
    assert_eq!(
        lazy.implementation_closure_with_stats(&compilation, [root]),
        eager.implementation_closure_with_stats(&compilation, [root]),
    );
    assert!(lazy.dependencies_initialized());
    assert_eq!(lazy.build_stats(), eager.build_stats());
    assert_eq!(
        lazy.compile_vm_implementations_symbolic_dynamic(&compilation, [root])
            .map(|executable| executable.stats().clone()),
        eager
            .compile_vm_implementations_symbolic_dynamic(&compilation, [root])
            .map(|executable| executable.stats().clone()),
    );
}
