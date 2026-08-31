mod common;
use common::*;

#[test]
fn compiler_predicates_bypass_synthetic_static_call_targets() {
    let compilation = TestProject::compile(
        "/proc/check(atom/value)\n\treturn isturf(value) + isnull(value) + istype(value, /atom)\n",
    );
    let executable = ProcedureRegistry::build(&compilation)
        .compile_vm(&compilation)
        .expect("compiler predicates should link");
    let entry = executable
        .module()
        .effective_procedure_id("/proc/check")
        .expect("check procedure");
    let program = executable.module().procedure(entry).expect("check body");
    assert_eq!(
        program
            .instructions
            .iter()
            .filter(|instruction| matches!(instruction, Instruction::TypePredicate { .. }))
            .count(),
        3
    );
    assert!(
        !program
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::Call { .. })),
        "language predicates must not pay a synthetic procedure call: {:?}",
        program.instructions
    );
}

#[test]
fn links_standard_location_predicates_as_variadic_builtins() {
    let compilation = TestProject::compile(concat!(
        "/proc/valid_locations()\n",
        "\treturn isarea(new /area, new /area/station) + isobj(new /obj, new /obj/item) + ismob(new /mob, new /mob/living)\n",
        "/proc/invalid_locations()\n",
        "\treturn isarea(new /turf) + isobj(new /mob) + ismob(new /obj)\n",
    ));

    assert_eq!(
        execute_effective(&compilation, "/proc/valid_locations", &[]),
        Ok(Value::number(3.0))
    );
    assert_eq!(
        execute_effective(&compilation, "/proc/invalid_locations", &[]),
        Ok(Value::number(0.0))
    );
}

#[test]
fn links_direction_text_and_orange_standard_builtins() {
    let compilation = TestProject::compile(concat!(
        "/proc/classify()\n",
        "\treturn istext(\"hello\") + istext(3)\n",
        "/atom/example/proc/neighbors(other)\n",
        "\treturn get_dir(src, other) + length(orange(1, src))\n",
    ));
    let registry = ProcedureRegistry::build(&compilation);
    registry
        .compile_vm(&compilation)
        .expect("standard direction/text/orange builtins should link");
    assert_eq!(
        execute_effective(&compilation, "/proc/classify", &[]),
        Ok(Value::number(1.0))
    );
}

#[test]
fn headless_byond_membership_query_is_callable_and_false() {
    let compilation = TestProject::compile(
        "/client/proc/check_member()\n\treturn IsByondMember() || src.IsByondMember()\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let procedure = procedure_by_path(&registry, "/client/proc/check_member");
    let target = procedure.effective_target.expect("procedure body");
    let executable = registry
        .compile_vm_implementations(&compilation, [target])
        .expect("the engine membership query should link in headless mode");
    let entry = executable
        .implementation(target)
        .expect("check_member linked");
    let mut state = ExecutionState::new();
    state.set_type_parents(
        [
            (TypePath::parse("/datum").unwrap(), None),
            (
                TypePath::parse("/client").unwrap(),
                Some(TypePath::parse("/datum").unwrap()),
            ),
        ]
        .into(),
    );
    let client = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/client").unwrap());
    assert_eq!(
        execute_module_in_context(
            executable.module(),
            entry,
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(client), Value::Null),
        )
        .expect("membership query should execute"),
        Value::number(0.0),
    );
}
