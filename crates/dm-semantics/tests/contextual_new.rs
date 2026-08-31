mod common;
use common::*;

#[test]
fn contextual_new_in_typed_list_assignment_allocates_the_list_type() {
    let compilation = TestProject::compile(
        "/proc/build()\n\tvar/list/L = list()\n\tL[new()] = new()\n\treturn L[1]\n",
    );
    let result = execute_effective(&compilation, "/proc/build", &[]);
    assert!(matches!(result, Ok(Value::List(_))), "result: {result:?}");
}

#[test]
fn inferred_new_uses_every_statically_proven_destination_family() {
    let cases = [
        (
            "typed local wrapper and ternary",
            "/datum/item\n/proc/build(var/flag)\n\tvar/datum/item/value = (flag ? new() : new())\n\treturn value.type\n",
        ),
        (
            "implicit src field",
            "/datum/holder\n\tvar/datum/item/value\n\tproc/build()\n\t\tvalue = new()\n\t\treturn value.type\n/datum/item\n",
        ),
        (
            "explicit and chained member fields",
            "/datum/outer\n\tvar/datum/inner/child\n\tproc/build()\n\t\tchild.value = new()\n\t\treturn child.value.type\n/datum/inner\n\tvar/datum/item/value\n/datum/item\n",
        ),
        (
            "typed global",
            "/var/datum/item/shared\n/proc/build()\n\tshared = new()\n\treturn shared.type\n/datum/item\n",
        ),
    ];
    for (label, source) in cases {
        let compilation = TestProject::compile(source);
        ProcedureRegistry::build(&compilation)
            .compile_vm(&compilation)
            .unwrap_or_else(|error| panic!("{label} should compile: {error:?}"));
    }
}

#[test]
fn title_icon_field_assignment_infers_icon_and_preserves_the_resource() {
    let compilation = TestProject::compile(concat!(
        "/datum/controller/subsystem/title\n",
        "\tvar/icon/icon\n",
        "\tproc/Initialize()\n",
        "\t\ticon = new(fcopy_rsc(\"icons/runtime/default_title.dmi\"))\n",
        "\t\treturn icon.icon\n",
        "/proc/run_title_initialize()\n",
        "\tvar/datum/controller/subsystem/title/title = new\n",
        "\treturn title.Initialize()\n",
    ));

    assert_eq!(
        execute_effective(&compilation, "/proc/run_title_initialize", &[]),
        Ok(Value::file("icons/runtime/default_title.dmi")),
    );
}

#[test]
fn inferred_new_follows_typed_global_controller_member_destination() {
    let compilation = TestProject::compile(
        "/datum/ghost_arena\n\tNew(var/source, var/marker)\n/datum/controller/global_vars\n\tvar/datum/ghost_arena/ghost_arena\n\tvar/first_arena_marker\n/var/global/datum/controller/global_vars/GLOB = new /datum/controller/global_vars\n/obj/effect/ghost_arena_corner/Initialize()\n\tGLOB.ghost_arena = new(src, GLOB.first_arena_marker)\n",
    );
    ProcedureRegistry::build(&compilation)
        .compile_vm(&compilation)
        .expect("typed GLOB member must qualify inferred new");
}

#[test]
fn inferred_new_uses_logical_assignment_and_compact_macro_local_context() {
    for source in [
        "/datum/cassette\n\tvar/datum/cassette_data/cassette_data\n\tproc/LateInitialize()\n\t\tcassette_data ||= new\n/datum/cassette_data\n",
        "/datum/sort_instance\n/var/global/datum/sort_instance/shared_sorter = new /datum/sort_instance\n/proc/sortTim()\n\tvar/datum/sort_instance/sorter = shared_sorter; if(isnull(sorter)){ sorter = new; }\n",
    ] {
        let compilation = TestProject::compile(source);
        ProcedureRegistry::build(&compilation)
            .compile_vm(&compilation)
            .expect("destination context must survive logical/compact assignment");
    }
}

#[test]
fn inferred_new_uses_slash_typed_parameter_without_var_keyword() {
    let compilation = TestProject::compile(
        "/datum/tgui\n/proc/ui_interact(mob/user, datum/tgui/ui)\n\tif(!ui)\n\t\tui = new(user)\n",
    );
    ProcedureRegistry::build(&compilation)
        .compile_vm(&compilation)
        .expect("typed parameter destination must qualify new");
}

#[test]
fn inferred_new_follows_builtin_mob_client_into_a_typed_safe_field() {
    let compilation = TestProject::compile(concat!(
        "/datum/meta_token_holder\n",
        "\tvar/client/owner\n",
        "/client\n",
        "\tvar/datum/meta_token_holder/client_token_holder\n",
        "/mob/Login()\n",
        "\tclient?.client_token_holder = new(client)\n",
    ));
    ProcedureRegistry::build(&compilation)
        .compile_vm(&compilation)
        .expect("the built-in typed mob.client edge must qualify safe-field new");
}

#[test]
fn inferred_new_uses_typed_parameter_default_and_for_receiver_field() {
    let compilation = TestProject::compile(
        "/datum/point\n/proc/copy_to(datum/point/p = new)\n\treturn p\n/datum/gas\n/obj/pipe\n\tvar/datum/gas/air_temporary\n/proc/store(var/list/members)\n\tfor(var/obj/pipe/member in members)\n\t\tmember.air_temporary = new\n",
    );
    ProcedureRegistry::build(&compilation)
        .compile_vm(&compilation)
        .expect("parameter defaults and typed loop receivers qualify new");
}

#[test]
fn inferred_new_rejects_contextless_and_unresolved_destinations() {
    for source in [
        "/proc/build()\n\treturn new()\n",
        "/proc/build()\n\tvar/value\n\tvalue = new()\n",
    ] {
        let compilation = TestProject::compile(source);
        let error = ProcedureRegistry::build(&compilation)
            .compile_vm(&compilation)
            .expect_err("unproven inferred new must be rejected");
        assert!(
            error
                .message
                .contains("no statically resolved destination type")
        );
    }
}

#[test]
fn construction_closure_narrows_typed_and_subtypesof_families() {
    let compilation = TestProject::compile(
        "/proc/from_loop()\n\tfor(var/path in subtypesof(/datum/base))\n\t\tvar/datum/value = new path\n/proc/from_typesof()\n\tfor(var/path in typesof(/datum/base))\n\t\tvar/datum/value = new path\n/proc/from_typecache()\n\tfor(var/datum/base/path as anything in typecacheof(path = /datum/base, ignore_root_path = TRUE))\n\t\tvar/datum/value = new path\n/proc/from_typed(datum/base/path)\n\tvar/datum/value = new path\n/proc/from_unknown(path)\n\tvar/datum/value = new path\n/proc/from_newlist()\n\treturn newlist(/datum/base/child)\n/proc/from_dynamic_newlist(path)\n\treturn newlist(path)\n/datum/base/New()\n/datum/base/child/New()\n/datum/unrelated/New()\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let target = |path| {
        procedure_by_path(&registry, path)
            .effective_target
            .expect("implementation")
    };
    let base_new = target("/datum/base/proc/New");
    let child_new = target("/datum/base/child/proc/New");
    let unrelated_new = target("/datum/unrelated/proc/New");

    for entry in [
        "/proc/from_loop",
        "/proc/from_typesof",
        "/proc/from_typecache",
        "/proc/from_typed",
    ] {
        let closure = registry.implementation_closure(&compilation, [target(entry)]);
        assert!(
            closure.contains(&base_new),
            "{entry} should retain base New"
        );
        assert!(
            closure.contains(&child_new),
            "{entry} should retain descendant New"
        );
        assert!(
            !closure.contains(&unrelated_new),
            "{entry} must omit unrelated New"
        );
    }

    let unknown = registry.implementation_closure(&compilation, [target("/proc/from_unknown")]);
    assert!(unknown.contains(&base_new));
    assert!(unknown.contains(&child_new));
    assert!(
        unknown.contains(&unrelated_new),
        "a genuinely untyped construction must retain all New candidates"
    );

    let newlist = registry.implementation_closure(&compilation, [target("/proc/from_newlist")]);
    assert!(newlist.contains(&child_new));
    assert!(!newlist.contains(&unrelated_new));
    let dynamic_newlist =
        registry.implementation_closure(&compilation, [target("/proc/from_dynamic_newlist")]);
    assert!(dynamic_newlist.contains(&base_new));
    assert!(dynamic_newlist.contains(&child_new));
    assert!(dynamic_newlist.contains(&unrelated_new));
}

#[test]
fn explicit_construction_links_and_runs_glob_new_before_returning() {
    let compilation = TestProject::compile(
        "/var/global/datum/controller/global_vars/GLOB\n/datum/controller/global_vars/New(marker)\n\tGLOB = src\n\tsrc.marker = marker\n/proc/entry()\n\tvar/datum/controller/global_vars/created = new /datum/controller/global_vars(17)\n\treturn GLOB == created && created.marker == 17\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let entry = procedure_by_path(&registry, "/proc/entry")
        .effective_target
        .unwrap();
    let constructor = procedure_by_path(&registry, "/datum/controller/global_vars/proc/New")
        .effective_target
        .unwrap();
    let closure = registry.implementation_closure(&compilation, [entry]);
    assert!(closure.contains(&constructor));

    let executable = registry
        .compile_vm_implementations(&compilation, [entry])
        .expect("constructor dependency should link");
    let mut state = ExecutionState::new();
    assert_eq!(
        execute_module_in_context(
            executable.module(),
            executable.implementation(entry).unwrap(),
            &[],
            &mut state,
            &ExecutionContext::default(),
        ),
        Ok(Value::number(1.0))
    );
}

#[test]
fn subtype_construction_uses_inherited_new_once_with_arguments() {
    let compilation = TestProject::compile(
        "/var/global/constructor_calls = 0\n/datum/base\n\tvar/marker\n/datum/base/New(marker)\n\tconstructor_calls += 1\n\tsrc.marker = marker\n/datum/base/child\n/proc/entry()\n\tvar/datum/base/child/created = new /datum/base/child(23)\n\treturn constructor_calls * 100 + created.marker\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let entry = procedure_by_path(&registry, "/proc/entry")
        .effective_target
        .unwrap();
    let inherited = procedure_by_path(&registry, "/datum/base/proc/New")
        .effective_target
        .unwrap();
    let closure = registry.implementation_closure(&compilation, [entry]);
    assert!(closure.contains(&inherited));

    let executable = registry
        .compile_vm_implementations(&compilation, [entry])
        .expect("inherited constructor dependency should link");
    let mut state = ExecutionState::new();
    state.set_global(
        FieldName::parse("constructor_calls").unwrap(),
        Value::number(0.0),
    );
    assert_eq!(
        execute_module_in_context(
            executable.module(),
            executable.implementation(entry).unwrap(),
            &[],
            &mut state,
            &ExecutionContext::default(),
        ),
        Ok(Value::number(123.0))
    );
}

#[test]
fn dynamic_subsystem_catalog_construction_runs_each_generated_new_and_sets_global() {
    let compilation = TestProject::compile(
        "/var/global/datum/controller/subsystem/processing/dcs/SSdcs\n/datum/controller/subsystem\n/datum/controller/subsystem/processing\n/datum/controller/subsystem/processing/dcs/New()\n\tSSdcs = src\n/proc/entry()\n\tvar/list/subsystem_types = typesof(/datum/controller/subsystem) - /datum/controller/subsystem\n\tfor(var/I in subsystem_types)\n\t\tnew I\n\treturn istype(SSdcs, /datum/controller/subsystem/processing/dcs)\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let entry = procedure_by_path(&registry, "/proc/entry")
        .effective_target
        .unwrap();
    let constructor = procedure_by_path(
        &registry,
        "/datum/controller/subsystem/processing/dcs/proc/New",
    )
    .effective_target
    .unwrap();
    let closure = registry.implementation_closure(&compilation, [entry]);
    assert!(closure.contains(&constructor));
    let executable = registry
        .compile_vm_implementations(&compilation, [entry])
        .expect("dynamic subsystem constructor family should link");
    let mut state = ExecutionState::new();
    let subsystem = TypePath::parse("/datum/controller/subsystem").unwrap();
    let processing = TypePath::parse("/datum/controller/subsystem/processing").unwrap();
    let dcs = TypePath::parse("/datum/controller/subsystem/processing/dcs").unwrap();
    state.set_type_paths([subsystem.clone(), processing.clone(), dcs.clone()]);
    state.set_type_parents(BTreeMap::from([
        (
            subsystem.clone(),
            Some(TypePath::parse("/datum/controller").unwrap()),
        ),
        (processing.clone(), Some(subsystem)),
        (dcs, Some(processing)),
    ]));
    state.set_global(FieldName::parse("SSdcs").unwrap(), Value::Null);
    assert_eq!(
        execute_module_in_context(
            executable.module(),
            executable.implementation(entry).unwrap(),
            &[],
            &mut state,
            &ExecutionContext::default(),
        ),
        Ok(Value::number(1.0))
    );
}
