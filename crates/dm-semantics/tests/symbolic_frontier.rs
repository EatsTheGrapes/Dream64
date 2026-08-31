mod common;
use common::*;

#[test]
fn all_symbolic_bootstrap_module_defers_unreachable_invalid_body() {
    let compilation = TestProject::compile(
        "/proc/reached()\n\treturn 7\n/proc/unreachable_invalid()\n\tvar/const/answer = 42\n\tanswer = 9\n\treturn answer\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let reached = procedure_by_path(&registry, "/proc/reached")
        .effective_target
        .expect("reached implementation");
    let invalid = procedure_by_path(&registry, "/proc/unreachable_invalid")
        .effective_target
        .expect("invalid implementation");
    let executable = registry
        .compile_vm_all_symbolic_deferred(&compilation)
        .expect("unreached body errors must remain deferred");
    assert_eq!(executable.module().deferred_procedure_count(), 2);
    assert_eq!(
        executable.module().materialized_deferred_procedure_count(),
        0
    );

    let mut state = ExecutionState::new();
    assert_eq!(
        execute_module_in_context(
            executable.module(),
            executable.implementation(reached).unwrap(),
            &[],
            &mut state,
            &ExecutionContext::default(),
        ),
        Ok(Value::number(7.0))
    );
    assert_eq!(
        executable.module().materialized_deferred_procedure_count(),
        1,
        "only the reached initializer callee should lower"
    );
    assert!(
        execute_module_in_context(
            executable.module(),
            executable.implementation(invalid).unwrap(),
            &[],
            &mut state,
            &ExecutionContext::default(),
        )
        .is_err(),
        "the deferred validation error must surface if the bad body is reached"
    );
}

#[test]
fn initializer_frontier_omits_unrelated_procedure_specs() {
    let compilation = TestProject::compile(
        "/proc/reached()\n\treturn helper()\n/proc/helper()\n\treturn 7\n/proc/unrelated_invalid()\n\tvar/const/answer = 42\n\tanswer = 9\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let reached = procedure_by_path(&registry, "/proc/reached")
        .effective_target
        .expect("reached implementation");
    let invalid = procedure_by_path(&registry, "/proc/unrelated_invalid")
        .effective_target
        .expect("invalid implementation");
    let executable = registry
        .compile_vm_initializer_frontier_symbolic_deferred(&compilation, ["reached"])
        .expect("frontier should link");
    assert_eq!(
        executable.module().deferred_procedure_count(),
        2,
        "the named root and its static callee should be retained"
    );
    assert!(
        executable.implementation(invalid).is_none(),
        "an unrelated body must not even receive a bootstrap module spec"
    );
    let mut state = ExecutionState::new();
    assert_eq!(
        execute_module_in_context(
            executable.module(),
            executable.implementation(reached).unwrap(),
            &[],
            &mut state,
            &ExecutionContext::default(),
        ),
        Ok(Value::number(7.0))
    );
    assert_eq!(
        executable.module().materialized_deferred_procedure_count(),
        2
    );
}

#[test]
fn complete_symbolic_lifecycle_table_keeps_unproven_runtime_methods_deferred() {
    let compilation = TestProject::compile(concat!(
        "/proc/startup()\n\treturn 1\n",
        "/datum/runtime_only/proc/invoke_from_data()\n\treturn 9\n",
    ));
    let registry = ProcedureRegistry::build(&compilation);
    let startup = procedure_by_path(&registry, "/proc/startup")
        .effective_target
        .unwrap();
    let runtime_only = procedure_by_path(&registry, "/datum/runtime_only/proc/invoke_from_data")
        .effective_target
        .unwrap();
    assert!(
        !registry
            .implementation_closure(&compilation, [startup])
            .contains(&runtime_only),
        "fixture must be outside the statically proven closure"
    );
    let executable = registry
        .compile_vm_all_symbolic_with_eager_roots(&compilation, [startup])
        .expect("complete deferred table should link");
    assert!(executable.implementation(startup).is_some());
    assert!(executable.implementation(runtime_only).is_some());
    assert!(executable.module().deferred_procedure_count() >= 1);
}

#[test]
fn typesof_procedure_family_links_generated_managed_global_initializers() {
    let compilation = TestProject::compile(concat!(
        "/datum/controller/global_vars/var/global/list/species_list\n",
        "/datum/controller/global_vars/var/global/list/crafting_recipes\n",
        "/datum/controller/global_vars/proc/InitGlobalspecies_list()\n",
        "\tspecies_list = list()\n",
        "/datum/controller/global_vars/proc/InitGlobalcrafting_recipes()\n",
        "\tcrafting_recipes = list()\n",
        "/datum/controller/global_vars/Initialize()\n",
        "\tfor(var/glob_proc in typesof(/datum/controller/global_vars/proc))\n",
        "\t\tcall(src, glob_proc)()\n",
        "/datum/unrelated/proc/not_an_initializer()\n",
    ));
    let registry = ProcedureRegistry::build(&compilation);
    let initialize = procedure_by_path(&registry, "/datum/controller/global_vars/proc/Initialize")
        .effective_target
        .expect("Initialize implementation");
    let closure = registry.implementation_closure(&compilation, [initialize]);

    for path in [
        "/datum/controller/global_vars/proc/InitGlobalspecies_list",
        "/datum/controller/global_vars/proc/InitGlobalcrafting_recipes",
    ] {
        let target = procedure_by_path(&registry, path)
            .effective_target
            .expect("generated managed-global initializer");
        assert!(closure.contains(&target), "closure omitted {path}");
    }
    let unrelated = procedure_by_path(&registry, "/datum/unrelated/proc/not_an_initializer")
        .effective_target
        .expect("unrelated implementation");
    assert!(!closure.contains(&unrelated));

    registry
        .compile_vm_implementations_symbolic_dynamic(&compilation, [initialize])
        .expect("bounded procedure family must be linked into the symbolic module");
}

#[test]
fn typesof_procedure_family_index_ignores_large_unrelated_registry() {
    let mut source = String::from(concat!(
        "/datum/target/proc/first()\n",
        "/datum/target/proc/second()\n",
        "/datum/target/proc/Initialize()\n",
        "\tfor(var/path in typesof(/datum/target/proc))\n",
        "\t\tcall(src, path)()\n",
    ));
    for index in 0..2_048 {
        source.push_str(&format!(
            "/datum/unrelated_{index}/proc/run()\n\treturn {index}\n"
        ));
    }
    let compilation = TestProject::compile(&source);
    let registry = ProcedureRegistry::build(&compilation);
    let initialize = procedure_by_path(&registry, "/datum/target/proc/Initialize")
        .effective_target
        .unwrap();
    let closure = registry.implementation_closure(&compilation, [initialize]);
    for path in ["/datum/target/proc/first", "/datum/target/proc/second"] {
        assert!(
            closure.contains(&procedure_by_path(&registry, path).effective_target.unwrap()),
            "indexed family omitted {path}",
        );
    }
    assert!(
        !closure.contains(
            &procedure_by_path(&registry, "/datum/unrelated_2047/proc/run")
                .effective_target
                .unwrap()
        )
    );
}

#[test]
fn reopened_initglobal_is_enumerated_and_invoked_by_procedure_typesof() {
    let compilation = TestProject::compile(concat!(
        "/datum/controller/global_vars\n\tvar/trace = 0\n",
        "/datum/controller/global_vars/proc/InitGlobalhuds_by_category()\n\ttrace += 1\n",
        "/datum/controller/global_vars/InitGlobalhuds_by_category()\n\t..()\n\ttrace += 10\n",
        "/datum/controller/global_vars/proc/InitGlobalhuds()\n\ttrace *= 2\n",
        "/datum/controller/global_vars/Initialize()\n",
        "\tfor(var/glob_proc in typesof(/datum/controller/global_vars/proc))\n",
        "\t\tcall(src, glob_proc)()\n",
        "\treturn trace\n",
    ));
    let registry = ProcedureRegistry::build(&compilation);
    let initialize = procedure_by_path(&registry, "/datum/controller/global_vars/proc/Initialize")
        .effective_target
        .unwrap();
    let closure = registry.implementation_closure(&compilation, [initialize]);
    for path in [
        "/datum/controller/global_vars/proc/InitGlobalhuds_by_category",
        "/datum/controller/global_vars/proc/InitGlobalhuds",
    ] {
        let target = procedure_by_path(&registry, path).effective_target.unwrap();
        assert!(closure.contains(&target), "closure omitted {path}");
    }
    let executable = registry
        .compile_vm_implementations_symbolic_dynamic(&compilation, [initialize])
        .expect("reopened managed global family should compile");
    let catalog = executable
        .module()
        .procedure_type_paths()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    for path in [
        "/datum/controller/global_vars/proc/InitGlobalhuds_by_category",
        "/datum/controller/global_vars/proc/InitGlobalhuds",
    ] {
        assert!(
            catalog.iter().any(|entry| entry == path),
            "catalog omitted {path}: {catalog:?}"
        );
    }
    let entry = executable.implementation(initialize).unwrap();
    let mut state = ExecutionState::new();
    let receiver = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/controller/global_vars").unwrap());
    state
        .heap_mut()
        .set_datum_field(
            receiver,
            FieldName::parse("trace").unwrap(),
            Value::number(0.0),
        )
        .unwrap();
    assert_eq!(
        execute_module_in_context(
            executable.module(),
            entry,
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(receiver), Value::Null),
        ),
        Ok(Value::number(22.0))
    );
}
