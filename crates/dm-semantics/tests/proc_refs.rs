mod common;
use common::*;

#[test]
fn first_class_proc_path_links_through_local_argument_and_field_call() {
    let compilation = TestProject::compile(
        "/datum/sorter\n\tvar/cmp\n/datum/sorter/proc/run(comparator)\n\tvar/local_cmp = comparator\n\tsrc.cmp = local_cmp\n\treturn call(src.cmp)(2, 7)\n/proc/cmp_subsystem_init(a, b)\n\treturn b - a\n/proc/unrelated()\n\treturn 99\n/proc/entry(comparator = /proc/cmp_subsystem_init)\n\tvar/list/refs = list(/proc/cmp_subsystem_init, /datum/sorter)\n\tvar/datum/sorter/sorter = new\n\treturn sorter.run(refs[1] || comparator)\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let entry = procedure_by_path(&registry, "/proc/entry")
        .effective_target
        .unwrap();
    let comparator = procedure_by_path(&registry, "/proc/cmp_subsystem_init")
        .effective_target
        .unwrap();
    let unrelated = procedure_by_path(&registry, "/proc/unrelated")
        .effective_target
        .unwrap();
    let closure = registry.implementation_closure(&compilation, [entry]);
    assert!(closure.contains(&comparator));
    assert!(!closure.contains(&unrelated));

    let executable = registry
        .compile_vm_implementations(&compilation, [entry])
        .expect("first-class comparator reference should link");
    let mut state = ExecutionState::new();
    assert_eq!(
        execute_module_in_context(
            executable.module(),
            executable.implementation(entry).unwrap(),
            &[],
            &mut state,
            &ExecutionContext::default(),
        ),
        Ok(Value::number(5.0))
    );
}

#[test]
fn relative_proc_ref_retains_signal_callback() {
    let compilation = TestProject::compile(
        "/datum/handler/proc/register()\n\tvar/callback = nameof(.proc/new_item_created)\n\treturn call(src, callback)()\n/datum/handler/proc/new_item_created()\n\treturn 42\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let register = procedure_by_path(&registry, "/datum/handler/proc/register")
        .effective_target
        .unwrap();
    let callback = procedure_by_path(&registry, "/datum/handler/proc/new_item_created")
        .effective_target
        .unwrap();

    assert!(
        registry
            .implementation_closure(&compilation, [register])
            .contains(&callback),
        "PROC_REF-style nameof(.proc/name) callbacks must remain linked",
    );
}

#[test]
fn typed_proc_ref_retains_signal_callback_for_subtype_receiver() {
    let compilation = TestProject::compile(
        "/datum/module/proc/register(datum/module/syndicate/receiver)\n\tvar/callback = nameof(/datum/module.proc/add_overlay)\n\treturn call(receiver, callback)()\n/datum/module/proc/add_overlay()\n\treturn 42\n/datum/module/syndicate\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let register = procedure_by_path(&registry, "/datum/module/proc/register")
        .effective_target
        .unwrap();
    let callback = procedure_by_path(&registry, "/datum/module/proc/add_overlay")
        .effective_target
        .unwrap();

    assert!(
        registry
            .implementation_closure(&compilation, [register])
            .contains(&callback),
        "TYPE_PROC_REF-style nameof(/owner.proc/name) must retain the callback",
    );
    let executable = registry
        .compile_vm_implementations(&compilation, [register])
        .expect("typed callback should link");
    let mut state = ExecutionState::new();
    state.set_type_parents(
        [
            (TypePath::parse("/datum").unwrap(), None),
            (
                TypePath::parse("/datum/module").unwrap(),
                Some(TypePath::parse("/datum").unwrap()),
            ),
            (
                TypePath::parse("/datum/module/syndicate").unwrap(),
                Some(TypePath::parse("/datum/module").unwrap()),
            ),
        ]
        .into(),
    );
    let receiver = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/module/syndicate").unwrap());
    assert_eq!(
        execute_module_in_context(
            executable.module(),
            executable.implementation(register).unwrap(),
            &[Value::Datum(receiver)],
            &mut state,
            &ExecutionContext::default(),
        ),
        Ok(Value::number(42.0)),
    );
}

#[test]
fn literal_text2path_proc_reference_retains_inferable_symbol() {
    let compilation = TestProject::compile(
        "/proc/cmp_value(a, b)\n\treturn a - b\n/proc/entry()\n\tvar/cmp = text2path(\"/proc/cmp_value\")\n\treturn call(cmp)(9, 4)\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    assert_eq!(
        registry.build_stats().static_proc_reference_index_lookups,
        1,
        "one literal reference should require one indexed lookup regardless of registry size"
    );
    let entry = procedure_by_path(&registry, "/proc/entry")
        .effective_target
        .unwrap();
    let comparator = procedure_by_path(&registry, "/proc/cmp_value")
        .effective_target
        .unwrap();
    assert!(
        registry
            .implementation_closure(&compilation, [entry])
            .contains(&comparator)
    );
    let executable = registry
        .compile_vm_implementations(&compilation, [entry])
        .unwrap();
    let mut state = ExecutionState::new();
    state.set_type_paths([TypePath::parse("/proc/cmp_value").unwrap()]);
    assert_eq!(
        execute_module_in_context(
            executable.module(),
            executable.implementation(entry).unwrap(),
            &[],
            &mut state,
            &ExecutionContext::default(),
        ),
        Ok(Value::number(5.0))
    );
}

#[test]
fn project_sort_wrapper_and_comparator_reference_are_retained_transitively() {
    let compilation = TestProject::compile(concat!(
        "/proc/cmp_desc(left, right)\n\treturn right - left\n",
        "/proc/sort_list(values, comparator)\n",
        "\treturn call(comparator)(values[1], values[2])\n",
        "/proc/entry()\n\treturn sort_list(list(1, 3), /proc/cmp_desc)\n",
        "/proc/unrelated()\n\treturn 99\n",
    ));
    let registry = ProcedureRegistry::build(&compilation);
    let entry = procedure_by_path(&registry, "/proc/entry")
        .effective_target
        .unwrap();
    let wrapper = procedure_by_path(&registry, "/proc/sort_list")
        .effective_target
        .unwrap();
    let comparator = procedure_by_path(&registry, "/proc/cmp_desc")
        .effective_target
        .unwrap();
    let unrelated = procedure_by_path(&registry, "/proc/unrelated")
        .effective_target
        .unwrap();
    let closure = registry.implementation_closure(&compilation, [entry]);
    assert!(closure.contains(&wrapper));
    assert!(closure.contains(&comparator));
    assert!(!closure.contains(&unrelated));

    let executable = registry
        .compile_vm_implementations(&compilation, [entry])
        .expect("project sort wrapper and comparator should link transitively");
    let mut state = ExecutionState::new();
    assert_eq!(
        execute_module_in_context(
            executable.module(),
            executable.implementation(entry).unwrap(),
            &[],
            &mut state,
            &ExecutionContext::default(),
        ),
        Ok(Value::number(2.0)),
    );
}
