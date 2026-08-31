mod common;
use common::*;

#[test]
fn caller_exposes_the_actual_calling_frame_as_a_callee_datum() {
    let compilation = TestProject::compile(
        "/datum/example/proc/outer()\n\treturn inner()\n/datum/example/proc/inner()\n\treturn caller.src == src && isnull(caller.caller)\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let outer = procedure_by_path(&registry, "/datum/example/proc/outer");
    let target = outer.effective_target.unwrap();
    let executable = registry
        .compile_vm_implementations(&compilation, [target])
        .expect("caller should lower as an implicit proc variable");
    let mut state = ExecutionState::new();
    let datum = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/example").unwrap());
    assert_eq!(
        execute_module_in_context(
            executable.module(),
            executable.implementation(target).unwrap(),
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(datum), Value::Null),
        ),
        Ok(Value::number(1.0))
    );
}

#[test]
fn nested_list_assignment_preserves_get_element_result_value() {
    let compilation = TestProject::compile(
        "/datum/element\n/datum/element/child\n/datum/manager\n\tvar/list/elements_by_type\n/datum/manager/New()\n\telements_by_type = list()\n/datum/manager/proc/GetElement(list/arguments)\n\tvar/datum/element/eletype = arguments[1]\n\tvar/element_id = eletype\n\t. = elements_by_type[element_id]\n\tif(.)\n\t\treturn\n\t. = elements_by_type[element_id] = new eletype\n/proc/entry()\n\tvar/datum/manager/manager = new\n\tmanager.GetElement(list(/datum/element/child))\n\treturn istype(manager.GetElement(list(/datum/element/child)), /datum/element/child)\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let entry = procedure_by_path(&registry, "/proc/entry")
        .effective_target
        .unwrap();
    let executable = registry
        .compile_vm_implementations(&compilation, [entry])
        .expect("GetElement-shaped procedure should compile");
    let mut state = ExecutionState::new();
    state.set_type_parents(BTreeMap::from([
        (
            TypePath::parse("/datum/element/child").unwrap(),
            Some(TypePath::parse("/datum/element").unwrap()),
        ),
        (
            TypePath::parse("/datum/manager").unwrap(),
            Some(TypePath::parse("/datum").unwrap()),
        ),
    ]));
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
fn world_profile_override_parent_call_reaches_engine_native() {
    let compilation =
        TestProject::compile("/world/Profile(command, type, format)\n\treturn ..()\n");
    let registry = ProcedureRegistry::build(&compilation);
    let profile = procedure_by_path(&registry, "/world/proc/Profile");
    let target = profile.effective_target.expect("profile override");
    let executable = registry
        .compile_vm_implementations(&compilation, [target])
        .expect("native parent should link");
    let entry = executable
        .implementation(target)
        .expect("override is linked");
    let mut state = ExecutionState::new();
    let world = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/world").unwrap());

    let result = execute_module_in_context(
        executable.module(),
        entry,
        &[Value::number(2.0)],
        &mut state,
        &ExecutionContext::new(Value::Datum(world), Value::Null),
    )
    .expect("native profile call should execute");
    let Value::List(profile) = result else {
        panic!("non-JSON profile data should be a list");
    };
    assert_eq!(state.heap().list(profile).unwrap().len(), 6);
}

#[test]
fn world_config_and_open_port_overrides_reach_engine_natives() {
    let compilation = TestProject::compile(concat!(
        "/world/SetConfig(config_set, param, value)\n\treturn ..()\n",
        "/world/GetConfig(config_set, param)\n\treturn ..()\n",
        "/world/OpenPort(port)\n\treturn ..()\n",
    ));
    let registry = ProcedureRegistry::build(&compilation);
    let targets = [
        "/world/proc/SetConfig",
        "/world/proc/GetConfig",
        "/world/proc/OpenPort",
    ]
    .map(|path| procedure_by_path(&registry, path).effective_target.unwrap());
    let executable = registry
        .compile_vm_implementations(&compilation, targets)
        .expect("world native parents should link");
    let mut state = ExecutionState::new();
    let world = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/world").unwrap());
    let context = ExecutionContext::new(Value::Datum(world), Value::Null);
    let execute_target = |target, arguments: &[Value], state: &mut ExecutionState| {
        execute_module_in_context(
            executable.module(),
            executable.implementation(target).unwrap(),
            arguments,
            state,
            &context,
        )
        .unwrap()
    };
    assert_eq!(
        execute_target(
            targets[0],
            &[
                Value::text("env"),
                Value::text("DREAM64_TEST"),
                Value::text("set")
            ],
            &mut state,
        ),
        Value::Null
    );
    assert_eq!(
        execute_target(
            targets[1],
            &[Value::text("env"), Value::text("DREAM64_TEST")],
            &mut state,
        ),
        Value::text("set")
    );
    assert_eq!(
        execute_target(targets[2], &[Value::number(4321.0)], &mut state),
        Value::number(1.0)
    );
    assert_eq!(
        state
            .heap()
            .datum_field(world, &dm_value::FieldName::parse("port").unwrap())
            .unwrap(),
        &Value::number(4321.0)
    );
}

#[test]
fn selected_static_call_uses_object_tree_ancestor_not_lexical_path_ancestor() {
    let compilation = TestProject::compile(
        "/datum/proc/RegisterSignals()\n\treturn 42\n/area/centcom/proc/Initialize()\n\treturn RegisterSignals()\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let entry = procedure_by_path(&registry, "/area/centcom/proc/Initialize");
    let executable = registry
        .compile_vm_implementations(
            &compilation,
            entry
                .implementations
                .iter()
                .map(|implementation| implementation.id),
        )
        .expect("a call inherited from /datum should be linked into the selected module");
    let entry = executable
        .implementation(entry.effective_target.expect("entry has a body"))
        .expect("entry should be linked");
    assert_eq!(
        execute_module(executable.module(), entry, &[]),
        Ok(Value::number(42.0))
    );
}

#[test]
fn selected_method_includes_and_resolves_direct_helper_calls() {
    let compilation = TestProject::compile(
        "/datum/receiver\n\tproc/entry()\n\t\treturn helper()\n\tproc/helper()\n\t\treturn 9\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let entry = procedure_by_path(&registry, "/datum/receiver/proc/entry");
    let executable = registry
        .compile_vm_implementations(
            &compilation,
            entry
                .implementations
                .iter()
                .map(|implementation| implementation.id),
        )
        .expect("direct helper method should be included");
    let entry = executable
        .implementation(entry.effective_target.expect("entry has a body"))
        .expect("entry should be linked");
    let mut state = ExecutionState::new();
    let receiver = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/receiver").unwrap());
    assert_eq!(
        execute_module_in_context(
            executable.module(),
            entry,
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(receiver), Value::Null),
        ),
        Ok(Value::number(9.0))
    );
}

#[test]
fn owner_bare_call_resolves_callable_verb_like_byond_proc_dispatch() {
    let compilation = TestProject::compile(
        "/mob/living/proc/say()\n\treturn succumb()\n/mob/living/verb/succumb()\n\treturn 5\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let say = procedure_by_path(&registry, "/mob/living/proc/say")
        .effective_target
        .unwrap();
    let succumb = procedure_by_path(&registry, "/mob/living/verb/succumb")
        .effective_target
        .unwrap();
    let closure = registry.implementation_closure(&compilation, [say]);
    assert!(closure.contains(&succumb));
    registry
        .compile_vm_implementations(&compilation, [say])
        .expect("verbs are callable by bare owner method name");
}

#[test]
fn executes_an_inherited_override_and_reuses_omitted_arguments() {
    let compilation = TestProject::compile(
        "/datum/base\n\tproc/run(value = 2)\n\t\treturn value + 1\n/datum/base/child\n\trun(value = 2)\n\t\treturn ..() + 10\n",
    );

    assert_eq!(
        execute_effective(&compilation, "/datum/base/child/proc/run", &[]),
        Ok(Value::number(13.0))
    );
    assert_eq!(
        execute_effective(&compilation, "/datum/base/child/proc/run", &[Value::Null]),
        Ok(Value::number(13.0)),
        "BYOND applies the declared default to an explicitly null parameter before forwarding it to the parent",
    );
}

#[test]
fn executes_multiple_reopenings_through_the_exact_parent_chain() {
    let compilation = TestProject::compile(
        "/datum/base\n\tproc/run()\n\t\treturn 1\n/datum/base\n\trun()\n\t\treturn ..() + 1\n/datum/base\n\trun()\n\t\treturn ..() + 1\n",
    );

    assert_eq!(
        execute_effective(&compilation, "/datum/base/proc/run", &[]),
        Ok(Value::number(3.0))
    );
}

#[test]
fn executes_parent_target_from_explicit_parent_type_and_explicit_arguments() {
    let compilation = TestProject::compile(
        "/datum/alternate\n\tproc/run(value = 1)\n\t\treturn value * 2\n/custom\n\tparent_type = /datum/alternate\n\trun(value = 3)\n\t\treturn ..(value + 1)\n",
    );

    assert_eq!(
        execute_effective(&compilation, "/custom/proc/run", &[]),
        Ok(Value::number(8.0))
    );
}

#[test]
fn terminal_new_and_del_parent_calls_resolve_to_engine_hooks() {
    let compilation = TestProject::compile(
        "/datum/species\n\tNew()\n\t\treturn ..()\n\tDel()\n\t\treturn ..()\n",
    );
    assert_eq!(
        execute_effective(&compilation, "/datum/species/proc/New", &[]),
        Ok(Value::Null),
        "a subtype constructor may terminate at BYOND's engine /datum/New",
    );
    assert_eq!(
        execute_effective(&compilation, "/datum/species/proc/Del", &[]),
        Ok(Value::Null),
        "a subtype destructor may terminate at BYOND's engine /datum/Del",
    );
}

#[test]
fn movable_bump_parent_chain_terminates_at_the_engine_native() {
    let compilation = TestProject::compile(concat!(
        "/atom/movable/Bump(atom/obstacle)\n",
        "\t. = ..()\n",
        "\treturn isnull(.) * 7\n",
        "/obj/crate/Bump(atom/obstacle)\n",
        "\treturn ..() + 1\n",
    ));

    assert_eq!(
        execute_effective(&compilation, "/atom/movable/proc/Bump", &[Value::Null]),
        Ok(Value::number(7.0)),
        "the project base override must observe BYOND's null terminal result",
    );
    assert_eq!(
        execute_effective(&compilation, "/obj/crate/proc/Bump", &[Value::Null]),
        Ok(Value::number(8.0)),
        "a descendant override must traverse the project base before the native terminal",
    );
}

#[test]
fn descendant_movable_bump_can_reach_the_native_without_a_source_base() {
    let compilation =
        TestProject::compile("/obj/crate/Bump(atom/obstacle)\n\treturn isnull(..())\n");

    assert_eq!(
        execute_effective(&compilation, "/obj/crate/proc/Bump", &[Value::Null]),
        Ok(Value::number(1.0)),
    );
}

#[test]
fn unrelated_bump_name_does_not_bind_to_the_movable_native() {
    let compilation = TestProject::compile("/datum/example/Bump()\n\treturn ..()\n");
    let error = execute_effective(&compilation, "/datum/example/proc/Bump", &[])
        .expect_err("a datum procedure named Bump has no engine movable parent");

    assert_eq!(
        error.message,
        "parent procedure call has no resolved target"
    );
}

#[test]
fn engine_generator_icon_and_walk_surfaces_lower_eagerly_and_execute() {
    let compilation = TestProject::compile(concat!(
        "/proc/Rand()\n\treturn 99\n",
        "/generator/proc/RandList()\n\treturn Rand()\n",
        "/icon/proc/Opaque(background = \"#000000\")\n",
        "\tSwapColor(null, background)\n",
        "\treturn src\n",
        "/proc/generator_result()\n",
        "\tvar/generator/value = generator(\"num\", 4, 4)\n",
        "\treturn value.RandList()\n",
        "/proc/icon_result()\n",
        "\tvar/icon/value = icon()\n",
        "\treturn value.Opaque()\n",
        "/proc/_walk(ref, dir, lag)\n\twalk(ref, dir, lag)\n",
        "/proc/_walk_towards(ref, target, lag)\n\twalk_towards(ref, target, lag)\n",
        "/proc/_walk_to(ref, target, minimum, lag)\n",
        "\twalk_to(ref, target, minimum, lag)\n",
        "/proc/_walk_away(ref, target, maximum, lag)\n",
        "\twalk_away(ref, target, maximum, lag)\n",
        "/proc/_walk_rand(ref, lag)\n\twalk_rand(ref, lag)\n",
    ));
    let registry = ProcedureRegistry::build(&compilation);
    let generator_target = procedure_by_path(&registry, "/proc/generator_result")
        .effective_target
        .unwrap();
    let icon_target = procedure_by_path(&registry, "/proc/icon_result")
        .effective_target
        .unwrap();
    let executable = registry
        .compile_vm_all_symbolic_deferred(&compilation)
        .expect("engine surface fixture should link")
        .into_fully_eager()
        .expect("all Monk-shaped engine calls should lower eagerly");
    assert_eq!(executable.module().deferred_procedure_count(), 0);

    let mut state = ExecutionState::new();
    assert_eq!(
        execute_module_in_state(
            executable.module(),
            executable.implementation(generator_target).unwrap(),
            &[],
            &mut state,
        ),
        Ok(Value::number(4.0)),
        "the engine-owned generator member must win over a same-name global proc",
    );
    let Value::Datum(icon) = execute_module_in_state(
        executable.module(),
        executable.implementation(icon_target).unwrap(),
        &[],
        &mut state,
    )
    .expect("the native icon member should execute") else {
        panic!("Opaque should return its icon receiver")
    };
    let operations_field = FieldName::parse("_dream64_icon_operations").unwrap();
    let Value::List(operations) = state
        .heap()
        .datum_field(icon, &operations_field)
        .expect("SwapColor should record an icon operation")
    else {
        panic!("icon operations should be stored in a list")
    };
    let [(_, Value::List(operation))] = state
        .heap()
        .list(*operations)
        .unwrap()
        .positions()
        .collect::<Vec<_>>()
        .as_slice()
    else {
        panic!("Opaque should perform exactly one icon operation")
    };
    assert_eq!(
        state.heap().list(*operation).unwrap().get(1),
        Ok(&Value::text("SwapColor")),
    );
}

#[test]
fn project_generator_member_overrides_engine_rand_in_full_and_independent_modules() {
    let compilation = TestProject::compile(concat!(
        "/generator/Rand()\n\treturn 99\n",
        "/generator/proc/RandList()\n\treturn Rand()\n",
        "/proc/run()\n",
        "\tvar/generator/value = generator(\"num\", 4, 4)\n",
        "\treturn value.RandList()\n",
    ));
    assert_eq!(
        execute_effective(&compilation, "/proc/run", &[]),
        Ok(Value::number(99.0)),
    );

    let registry = ProcedureRegistry::build(&compilation);
    let rand_list = procedure_by_path(&registry, "/generator/proc/RandList")
        .effective_target
        .unwrap();
    let mut independently = registry.compile_vm_bodies_independently(&compilation, [rand_list]);
    let (compiled_id, result) = independently
        .pop()
        .expect("independent RandList body should be present");
    assert_eq!(compiled_id, rand_list);
    result.expect("a project member override should remain a valid independent call target");
}

#[test]
fn missing_parent_target_is_a_source_mapped_runtime_error() {
    let compilation = TestProject::compile("/proc/orphan()\n\treturn ..()\n");
    let registry = ProcedureRegistry::build(&compilation);
    let procedure = procedure_by_path(&registry, "/proc/orphan");
    let implementation = procedure.implementations[0];
    assert_eq!(implementation.parent_target, None);
    let expected_span = compilation
        .syntax(implementation.file_id)
        .expect("source syntax should exist")
        .definitions[implementation.definition_index]
        .body[0]
        .span;
    let error = execute_effective(&compilation, "/proc/orphan", &[])
        .expect_err("orphan parent call should fail at runtime");

    assert_eq!(
        error.message,
        "parent procedure call has no resolved target"
    );
    assert_eq!(error.source_span, Some(expected_span));
    assert_eq!(error.call_stack.len(), 1);
}

#[test]
fn engine_owned_topic_and_click_methods_supply_terminal_parent_targets() {
    let compilation = TestProject::compile(
        "/datum/Topic(href, list/href_list)\n\treturn ..()\n/client/Click(object, location, control, params)\n\treturn ..()\n/proc/run()\n\tvar/datum/target = new\n\tvar/client/user = new\n\treturn isnull(target.Topic(\"x\", list())) + isnull(user.Click(null, null, null, null))\n",
    );
    assert_eq!(
        execute_effective(&compilation, "/proc/run", &[]),
        Ok(Value::number(2.0)),
    );
}

#[test]
fn engine_owned_client_click_dispatches_the_addressed_atom() {
    let compilation = TestProject::compile(
        "var/global/clicked = 0\n/atom/Click(location, control, params)\n\tclicked = (control == \"map\" && params == \"left=1\")\n\treturn 7\n/client/Click(object, location, control, params)\n\treturn ..()\n/proc/run()\n\tvar/atom/target = new\n\tvar/client/user = new\n\treturn user.Click(target, null, \"map\", \"left=1\") + clicked * 10\n",
    );
    assert_eq!(
        execute_effective(&compilation, "/proc/run", &[]),
        Ok(Value::number(17.0)),
    );
}

#[test]
fn parent_failure_preserves_both_source_mapped_frames() {
    let compilation = TestProject::compile(
        "/datum/base\n\tproc/run()\n\t\treturn \"text\" + 1\n/datum/base/child\n\trun()\n\t\treturn ..()\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let base = procedure_by_path(&registry, "/datum/base/proc/run");
    let base_implementation = base.implementations[0];
    let expected_span = compilation
        .syntax(base_implementation.file_id)
        .expect("source syntax should exist")
        .definitions[base_implementation.definition_index]
        .body[0]
        .span;
    let error = execute_effective(&compilation, "/datum/base/child/proc/run", &[])
        .expect_err("parent numeric failure should propagate");

    assert_eq!(error.source_span, Some(expected_span));
    assert_eq!(error.call_stack.len(), 2);
    assert!(
        error.call_stack[0]
            .procedure
            .contains("/datum/base/child/proc/run")
    );
    assert!(
        error.call_stack[1]
            .procedure
            .contains("/datum/base/proc/run")
    );
    assert_eq!(error.call_stack[1].source_span, Some(expected_span));
}
