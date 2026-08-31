mod common;
use common::*;

#[test]
fn selected_dynamic_literal_call_includes_matching_method_implementation() {
    let compilation = TestProject::compile(
        "/datum/receiver\n\tproc/entry()\n\t\treturn call(src, \"register\")()\n\tproc/register()\n\t\treturn 9\n",
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
        .expect("literal dynamic method should be included");
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
fn typed_global_member_call_links_exact_runtime_receiver_target() {
    let compilation = TestProject::compile(
        "var/global/datum/log_holder/logger\n/proc/entry()\n\treturn logger.Log(4)\n/datum/log_holder/proc/Log(value)\n\treturn value + 3\n/datum/unrelated/proc/Log(value)\n\treturn value + 100\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let entry = procedure_by_path(&registry, "/proc/entry");
    let entry_target = entry.effective_target.expect("entry implementation");
    let (closure, _) = registry.implementation_closure_with_stats(&compilation, [entry_target]);
    let log_target = procedure_by_path(&registry, "/datum/log_holder/proc/Log")
        .effective_target
        .expect("logger implementation");
    let unrelated = procedure_by_path(&registry, "/datum/unrelated/proc/Log")
        .effective_target
        .expect("unrelated implementation");
    assert!(closure.contains(&log_target));
    assert!(!closure.contains(&unrelated));

    let executable = registry
        .compile_vm_implementations(&compilation, [entry_target])
        .expect("typed member target should link");
    let mut state = ExecutionState::new();
    let logger = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/log_holder").unwrap());
    state.set_global(
        dm_value::FieldName::parse("logger").unwrap(),
        Value::Datum(logger),
    );
    assert_eq!(
        execute_module_in_context(
            executable.module(),
            executable.implementation(entry_target).unwrap(),
            &[],
            &mut state,
            &ExecutionContext::default(),
        ),
        Ok(Value::number(7.0))
    );
}

#[test]
fn interpolated_world_member_call_links_runtime_candidates() {
    let compilation = TestProject::compile(
        "/world/proc/get_world_state_for_logging()\n\treturn 7\n/datum/log_entry/proc/render()\n\tvar/list/entries = list()\n\tentries.Add(\"[world.get_world_state_for_logging()]\")\n\treturn entries[1]\n/datum/unrelated/proc/get_world_state_for_logging()\n\treturn 99\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let entry = procedure_by_path(&registry, "/datum/log_entry/proc/render")
        .effective_target
        .unwrap();
    let closure = registry.implementation_closure(&compilation, [entry]);
    let world = procedure_by_path(&registry, "/world/proc/get_world_state_for_logging")
        .effective_target
        .unwrap();
    let unrelated = procedure_by_path(
        &registry,
        "/datum/unrelated/proc/get_world_state_for_logging",
    )
    .effective_target
    .unwrap();
    assert!(closure.contains(&world));
    assert!(closure.contains(&unrelated));

    let executable = registry
        .compile_vm_implementations_symbolic_dynamic(&compilation, [entry])
        .expect("macro-shaped nested world member target should link");
    assert!(executable.implementation(world).is_some());
    let mut state = ExecutionState::new();
    let world_datum = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/world").unwrap());
    state.set_global(
        dm_value::FieldName::parse("world").unwrap(),
        Value::Datum(world_datum),
    );
    let log_entry = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/log_entry").unwrap());
    assert_eq!(
        execute_module_in_context(
            executable.module(),
            executable.implementation(entry).unwrap(),
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(log_entry), Value::Null),
        ),
        Ok(Value::text("7"))
    );
}

#[test]
fn typed_global_field_chain_links_exact_runtime_receiver_target() {
    let compilation = TestProject::compile(
        "var/global/datum/globals/GLOB\n/datum/globals/var/datum/log_holder/logger\n/proc/entry()\n\treturn GLOB.logger.Log(4)\n/datum/log_holder/proc/Log(value)\n\treturn value + 3\n/datum/unrelated/proc/Log(value)\n\treturn value + 100\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let entry = procedure_by_path(&registry, "/proc/entry")
        .effective_target
        .expect("entry implementation");
    let closure = registry.implementation_closure(&compilation, [entry]);
    let log = procedure_by_path(&registry, "/datum/log_holder/proc/Log")
        .effective_target
        .expect("logger implementation");
    let unrelated = procedure_by_path(&registry, "/datum/unrelated/proc/Log")
        .effective_target
        .expect("unrelated implementation");
    assert!(closure.contains(&log));
    assert!(!closure.contains(&unrelated));
}

#[test]
fn implicit_inherited_typed_field_member_call_links_exact_target() {
    let compilation = TestProject::compile(
        "/datum/base/var/datum/log_holder/logger\n/datum/child\n\tparent_type = /datum/base\n\tproc/entry()\n\t\treturn logger.Log()\n/datum/log_holder/proc/Log()\n\treturn 7\n/datum/unrelated/proc/Log()\n\treturn 100\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let entry = procedure_by_path(&registry, "/datum/child/proc/entry")
        .effective_target
        .expect("entry implementation");
    let closure = registry.implementation_closure(&compilation, [entry]);
    let log = procedure_by_path(&registry, "/datum/log_holder/proc/Log")
        .effective_target
        .expect("logger implementation");
    let unrelated = procedure_by_path(&registry, "/datum/unrelated/proc/Log")
        .effective_target
        .expect("unrelated implementation");
    assert!(closure.contains(&log));
    assert!(!closure.contains(&unrelated));
}

#[test]
fn untyped_member_call_links_broad_candidates_and_dispatches_runtime_override() {
    let compilation = TestProject::compile(
        "/proc/entry(receiver)\n\treturn receiver.Log()\n/datum/base/proc/Log()\n\treturn 1\n/datum/child\n\tparent_type = /datum/base\n/datum/child/Log()\n\treturn 2\n/datum/other/proc/Log()\n\treturn 3\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let entry = procedure_by_path(&registry, "/proc/entry");
    let entry_target = entry.effective_target.expect("entry implementation");
    let (closure, _) = registry.implementation_closure_with_stats(&compilation, [entry_target]);
    for path in [
        "/datum/base/proc/Log",
        "/datum/child/proc/Log",
        "/datum/other/proc/Log",
    ] {
        assert!(
            closure.contains(
                &procedure_by_path(&registry, path)
                    .effective_target
                    .expect("member implementation")
            ),
            "untyped receiver must retain candidate {path}"
        );
    }
    let executable = registry
        .compile_vm_implementations(&compilation, [entry_target])
        .expect("broad dynamic member closure should link");
    let mut state = ExecutionState::new();
    let child = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/child").unwrap());
    assert_eq!(
        execute_module_in_context(
            executable.module(),
            executable.implementation(entry_target).unwrap(),
            &[Value::Datum(child)],
            &mut state,
            &ExecutionContext::default(),
        ),
        Ok(Value::number(2.0))
    );
}

#[test]
fn untyped_member_candidates_are_symbolic_until_runtime_dispatch() {
    let compilation = TestProject::compile(
        "/proc/entry(receiver)\n\treturn receiver.Log()\n/datum/child/proc/Log()\n\treturn 2\n/datum/unrelated/proc/Log()\n\treturn 100\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let entry = procedure_by_path(&registry, "/proc/entry")
        .effective_target
        .expect("entry implementation");
    assert_eq!(
        registry.eager_implementation_closure(&compilation, [entry]),
        BTreeSet::from([entry]),
        "genuinely untyped candidates must remain symbolic at the boot gate"
    );
    let executable = registry
        .compile_vm_implementations_symbolic_dynamic(&compilation, [entry])
        .expect("symbolic dynamic module should link");
    assert_eq!(executable.module().deferred_procedure_count(), 2);
    assert_eq!(
        executable.module().materialized_deferred_procedure_count(),
        0
    );

    let mut state = ExecutionState::new();
    let child = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/child").unwrap());
    assert_eq!(
        execute_module_in_context(
            executable.module(),
            executable.implementation(entry).unwrap(),
            &[Value::Datum(child)],
            &mut state,
            &ExecutionContext::default(),
        ),
        Ok(Value::number(2.0))
    );
    assert_eq!(
        executable.module().materialized_deferred_procedure_count(),
        1,
        "only the runtime-selected override should compile"
    );
}

#[test]
fn typed_virtual_overrides_are_symbolic_but_declared_base_is_eager() {
    let compilation = TestProject::compile(
        "/proc/entry(datum/base/receiver)\n\treturn receiver.Log()\n/datum/base/proc/Log()\n\treturn 1\n/datum/base/child/Log()\n\treturn expensive_helper()\n/datum/base/child/proc/expensive_helper()\n\treturn 2\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let entry = procedure_by_path(&registry, "/proc/entry")
        .effective_target
        .unwrap();
    let base = procedure_by_path(&registry, "/datum/base/proc/Log")
        .effective_target
        .unwrap();
    let child = procedure_by_path(&registry, "/datum/base/child/proc/Log")
        .effective_target
        .unwrap();
    let helper = procedure_by_path(&registry, "/datum/base/child/proc/expensive_helper")
        .effective_target
        .unwrap();
    let eager = registry.eager_implementation_closure(&compilation, [entry]);
    assert!(eager.contains(&base));
    assert!(!eager.contains(&child));
    assert!(!eager.contains(&helper));
    let full = registry.implementation_closure(&compilation, [entry]);
    assert!(full.contains(&child));
    assert!(full.contains(&helper));
}

#[test]
fn typed_local_member_call_after_dynamic_new_retains_admin_verb_method_family() {
    let compilation = TestProject::compile(concat!(
        "/datum/admin_verb/proc/__avd_check_should_exist()\n\treturn 1\n",
        "/datum/admin_verb/AdminVOX/__avd_check_should_exist()\n\treturn 0\n",
        "/datum/controller/subsystem/admin_verbs/proc/setup_verb_list()\n",
        "\tvar/datum/admin_verb/verb_type = /datum/admin_verb/AdminVOX\n",
        "\tvar/datum/admin_verb/verb_singleton = new verb_type\n",
        "\treturn verb_singleton.__avd_check_should_exist()\n",
    ));
    let registry = ProcedureRegistry::build(&compilation);
    let setup = procedure_by_path(
        &registry,
        "/datum/controller/subsystem/admin_verbs/proc/setup_verb_list",
    )
    .effective_target
    .unwrap();
    let base = procedure_by_path(&registry, "/datum/admin_verb/proc/__avd_check_should_exist")
        .effective_target
        .unwrap();
    let override_target = procedure_by_path(
        &registry,
        "/datum/admin_verb/AdminVOX/proc/__avd_check_should_exist",
    )
    .effective_target
    .unwrap();
    let closure = registry.implementation_closure(&compilation, [setup]);
    assert!(closure.contains(&base), "base method must be retained");
    assert!(
        closure.contains(&override_target),
        "compatible generated overrides must be retained"
    );
}

#[test]
fn typed_global_member_call_links_exact_method_not_bare_global_proc() {
    let compilation = TestProject::compile(
        "/datum/log_holder/proc/Log()\n\treturn 1\n/proc/Log()\n\treturn 2\n/var/global/datum/log_holder/logger = new /datum/log_holder\n/proc/log_world()\n\treturn logger.Log()\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let entry = procedure_by_path(&registry, "/proc/log_world")
        .effective_target
        .unwrap();
    let closure = registry.implementation_closure(&compilation, [entry]);
    let member = procedure_by_path(&registry, "/datum/log_holder/proc/Log")
        .effective_target
        .unwrap();
    let bare = procedure_by_path(&registry, "/proc/Log")
        .effective_target
        .unwrap();
    assert!(closure.contains(&member));
    assert!(
        !closure.contains(&bare),
        "member syntax must not become a bare static call"
    );
    registry
        .compile_vm_implementations(&compilation, [entry])
        .expect("logger.Log linked");
}

#[test]
fn ternary_false_arm_global_call_is_not_confused_with_colon_member_call() {
    let compilation = TestProject::compile(
        "/proc/format_text(var/x)\n\treturn x\n/proc/get_area_name(var/x, var/format_text)\n\treturn x\n/datum/holder/proc/get_area_name()\n\treturn 0\n/proc/run(var/datum/holder/A)\n\tvar/location = A ? format_text(A.name) : get_area_name(src, format_text=TRUE)\n\tA:get_area_name()\n\treturn location\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let entry = procedure_by_path(&registry, "/proc/run")
        .effective_target
        .unwrap();
    let closure = registry.implementation_closure(&compilation, [entry]);
    assert!(
        closure.contains(
            &procedure_by_path(&registry, "/proc/get_area_name")
                .effective_target
                .unwrap()
        )
    );
}

#[test]
fn dependency_closure_uses_preindexed_dynamic_selector_candidates() {
    let compilation = TestProject::compile(
        "/proc/entry()\n\treturn call(src, \"register\")()\n/datum/one/proc/register()\n\treturn 1\n/datum/two/proc/register()\n\treturn 2\n/datum/irrelevant/proc/alpha()\n\treturn 3\n/datum/irrelevant/proc/beta()\n\treturn 4\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let entry = procedure_by_path(&registry, "/proc/entry")
        .effective_target
        .expect("entry implementation");
    let (closure, stats) = registry.implementation_closure_with_stats(&compilation, [entry]);

    assert_eq!(
        closure.len(),
        3,
        "entry and both matching methods are reachable"
    );
    assert_eq!(stats.dynamic_selectors_resolved, 1);
    assert_eq!(
        stats.dynamic_candidates_considered, 2,
        "unrelated procedures must not be scanned as dynamic candidates"
    );
    assert_eq!(stats.bodies_visited, 3);
}

#[test]
fn inherited_bare_call_retains_and_dispatches_runtime_subtype_override() {
    let compilation = TestProject::compile(concat!(
        "/atom/proc/add_debris_element()\n\treturn 1\n",
        "/obj/Initialize()\n\treturn add_debris_element()\n",
        "/obj/effect/statclick/ticket_list\n",
        "/obj/structure/barricade/wooden/add_debris_element()\n\treturn 9\n",
        "/datum/unrelated/add_debris_element()\n\treturn 100\n",
    ));
    let registry = ProcedureRegistry::build(&compilation);
    let initialize = procedure_by_path(&registry, "/obj/proc/Initialize")
        .effective_target
        .unwrap();
    let base = procedure_by_path(&registry, "/atom/proc/add_debris_element")
        .effective_target
        .unwrap();
    let override_target = procedure_by_path(
        &registry,
        "/obj/structure/barricade/wooden/proc/add_debris_element",
    )
    .effective_target
    .unwrap();
    let unrelated = procedure_by_path(&registry, "/datum/unrelated/proc/add_debris_element")
        .effective_target
        .unwrap();
    let closure = registry.implementation_closure(&compilation, [initialize]);
    assert!(closure.contains(&base));
    assert!(closure.contains(&override_target));
    assert!(!closure.contains(&unrelated));
    assert!(
        !registry
            .eager_implementation_closure(&compilation, [initialize])
            .contains(&override_target),
        "compatible virtual overrides should stay deferred until dispatched",
    );

    let executable = registry
        .compile_vm_implementations_symbolic_dynamic(&compilation, [initialize])
        .unwrap();
    let mut state = ExecutionState::new();
    let wooden = TypePath::parse("/obj/structure/barricade/wooden").unwrap();
    state.set_type_parents(BTreeMap::from([
        (
            wooden.clone(),
            Some(TypePath::parse("/obj/structure/barricade").unwrap()),
        ),
        (
            TypePath::parse("/obj/structure/barricade").unwrap(),
            Some(TypePath::parse("/obj/structure").unwrap()),
        ),
        (
            TypePath::parse("/obj/structure").unwrap(),
            Some(TypePath::parse("/obj").unwrap()),
        ),
        (
            TypePath::parse("/obj").unwrap(),
            Some(TypePath::parse("/atom").unwrap()),
        ),
    ]));
    let receiver = state.heap_mut().allocate_datum(wooden);
    assert_eq!(
        execute_module_in_context(
            executable.module(),
            executable.implementation(initialize).unwrap(),
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(receiver), Value::Null),
        ),
        Ok(Value::number(9.0)),
    );
}

#[test]
fn inherited_bare_call_after_nested_ternary_colon_is_retained() {
    let compilation = TestProject::compile(concat!(
        "/atom/proc/drop_location()\n\treturn 7\n",
        "/obj/proc/forward(value)\n\treturn value\n",
        "/obj/proc/click_alt(user)\n",
        "\treturn forward(user ? user : drop_location())\n",
    ));
    let registry = ProcedureRegistry::build(&compilation);
    let click_alt = procedure_by_path(&registry, "/obj/proc/click_alt")
        .effective_target
        .expect("click_alt body");
    let drop_location = procedure_by_path(&registry, "/atom/proc/drop_location")
        .effective_target
        .expect("drop_location body");

    assert!(
        registry
            .implementation_closure(&compilation, [click_alt])
            .contains(&drop_location),
        "a ternary nested in a call must retain its bare inherited false-arm call",
    );
    registry
        .compile_vm_implementations(&compilation, [click_alt])
        .expect("the retained nested-ternary call should resolve during lowering");
}
