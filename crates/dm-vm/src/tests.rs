use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dm_core::{DmNumberBits, SourceSpan};
use dm_dmf::{ControlTree, parse as parse_dmf};
use dm_lexer::{SpannedToken, TokenKind, lex};
use dm_syntax::{DefinitionKind, parse};
use dm_value::{DatumId, FieldName, ListId, ModifiedTypePath, TypePath, ValueError};

use super::compile::{
    condition_tokens, dm_builtin_numeric_constant, interpolated_expression_close,
};
use super::{
    BULK_INIT_LOW_YIELD_STREAK, CANONICAL_TYPE2PARENT_SOURCE, CallFrame, CompoundListIndexOperator,
    DEFAULT_HEAP_IDENTITY_CEILING, ExecutionContext, ExecutionLimits, ExecutionState,
    InitializerBinding, InstanceInitializer, Instruction, LocalClientPromptKind,
    LocalClientPromptResponse, LocalClientUiEvent, MAX_EFFECTIVE_INITIAL_VALUE_CACHE_ENTRIES,
    MAX_EFFECTIVE_INITIAL_VALUE_CACHE_FIELDS_PER_TYPE, MAXIMUM_HIGH_YIELD_COLLECTION_GROWTH,
    MAXIMUM_LOW_YIELD_COLLECTION_GROWTH, MAXIMUM_MODERATE_YIELD_COLLECTION_GROWTH,
    MINIMUM_HEAP_COLLECTION_GROWTH, Module, ProcedureId, ProcedureSpec, Program,
    REGISTER_SIGNAL_FAST_CACHE, TypePredicateKind, Value, VerbParameterType,
    adaptive_heap_collection_growth, advance_scheduler, allocate_initialized_datum,
    allocate_matrix, assign_datum_field, bulk_init_aware_collection_growth, compile_initializer,
    compile_initializer_into_module, compile_module, compile_module_specs,
    compile_module_specs_selective, compile_module_specs_selective_with_errors,
    compile_module_with_global_fields, compile_procedure,
    compile_procedure_with_resolver_and_fields, datum_field_or_initial, datum_field_or_shared,
    execute, execute_in_context, execute_in_state, execute_module, execute_module_in_context,
    execute_module_in_state, execute_module_with_limits, execute_module_with_limits_in_state,
    execute_with_limits, execute_with_limits_in_state, initial_value_or_engine_root,
    instance_initializer_plan, is_subtype, make_frame, matrix_components, next_module_identity,
    packed_dispatch_counters, prepare_iteration_consumes_fresh_block, read_list_value,
    try_run_numeric_dispatch_block, try_run_packed_numeric_dispatch_block,
    try_run_register_signal_fast_path, try_run_rich_numeric_dispatch_block,
};
use super::{atom_contents_iteration_snapshot, world_contents_iteration_snapshot};

#[test]
fn builtin_mob_sight_flag_family_has_byond_bit_values() {
    for (name, expected) in [
        ("BLIND", 1.0),
        ("SEE_MOBS", 4.0),
        ("SEEMOBS", 4.0),
        ("SEE_OBJS", 8.0),
        ("SEEOBJS", 8.0),
        ("SEE_TURFS", 16.0),
        ("SEETURFS", 16.0),
        ("SEE_SELF", 32.0),
        ("SEE_INFRA", 64.0),
        ("SEE_PIXELS", 256.0),
        ("SEE_THRU", 512.0),
        ("SEE_BLACKNESS", 1024.0),
    ] {
        assert_eq!(dm_builtin_numeric_constant(name), Some(expected), "{name}");
    }
}

#[test]
fn builtin_generator_distribution_constants_have_byond_values() {
    assert_eq!(dm_builtin_numeric_constant("UNIFORM_RAND"), Some(0.0));
    assert_eq!(dm_builtin_numeric_constant("NORMAL_RAND"), Some(1.0));

    let syntax = parse("/proc/_generator(rand = UNIFORM_RAND)\n\treturn rand\n")
        .expect("generator wrapper fixture should parse");
    let module = compile_module(&syntax.definitions)
        .expect("UNIFORM_RAND should lower in a parameter default");
    assert_eq!(
        execute_module(
            &module,
            module.procedure_id("/proc/_generator").unwrap(),
            &[],
        ),
        Ok(Value::number(0.0)),
    );
}

#[test]
fn interpolation_close_skips_brackets_inside_nested_quotes() {
    let expression = r#"src ? "nested[value]" : fallback]tail"#;
    let close = interpolated_expression_close(expression, 0).expect("outer close should exist");
    assert_eq!(
        &expression[..=close],
        r#"src ? "nested[value]" : fallback]"#
    );
}

#[test]
fn canonical_type2parent_call_preserves_lexical_parent_results() {
    let source =
        format!("{CANONICAL_TYPE2PARENT_SOURCE}/proc/run(child)\n\treturn type2parent(child)\n");
    let syntax = parse(&source).expect("canonical helper fixture should parse");
    let module = compile_module(&syntax.definitions).expect("canonical helper should compile");
    assert!(super::canonical_type2parent_program(
        module
            .procedure(
                module
                    .procedure_id("/proc/type2parent")
                    .expect("helper entry")
            )
            .expect("helper program")
    ));
    let entry = module.procedure_id("/proc/run").expect("run entry");
    for (child, expected) in [
        ("/datum", None),
        ("/obj", Some("/atom/movable")),
        ("/mob", Some("/atom/movable")),
        ("/area", Some("/atom")),
        ("/turf", Some("/atom")),
        ("/atom", Some("/datum")),
        ("/datum/component/foo", Some("/datum/component")),
    ] {
        let result = execute_module(
            &module,
            entry,
            &[Value::TypePath(
                TypePath::parse(child).expect("valid child path"),
            )],
        )
        .expect("canonical helper call should execute");
        assert_eq!(
            result,
            expected.map_or(Value::Null, |path| {
                Value::TypePath(TypePath::parse(path).expect("valid expected path"))
            }),
            "child {child}"
        );
    }
}

#[test]
fn customized_type2parent_remains_an_ordinary_dm_call() {
    let syntax = parse(
            "/proc/type2parent(child)\n\treturn /datum/custom\n/proc/run(child)\n\treturn type2parent(child)\n",
        )
        .expect("custom helper fixture should parse");
    let module = compile_module(&syntax.definitions).expect("custom helper should compile");
    let result = execute_module(
        &module,
        module.procedure_id("/proc/run").expect("run entry"),
        &[Value::TypePath(
            TypePath::parse("/datum/component/foo").expect("valid child path"),
        )],
    )
    .expect("custom helper call should execute");
    assert_eq!(
        result,
        Value::TypePath(TypePath::parse("/datum/custom").expect("valid custom result"))
    );
}

#[test]
#[ignore = "release-only canonical type2parent dispatch-gate benchmark"]
fn canonical_type2parent_dispatch_gate_release_benchmark() {
    const ITERATIONS: usize = 5_000_000;
    let canonical_syntax = parse(CANONICAL_TYPE2PARENT_SOURCE).unwrap();
    let canonical_module = compile_module(&canonical_syntax.definitions).unwrap();
    let canonical_id = canonical_module.procedure_id("/proc/type2parent").unwrap();
    let canonical_program = canonical_module.procedure(canonical_id).unwrap();
    let ordinary_syntax = parse("/proc/build_coordinate(value)\n\treturn value\n").unwrap();
    let ordinary_module = compile_module(&ordinary_syntax.definitions).unwrap();
    let ordinary_id = ordinary_module
        .procedure_id("/proc/build_coordinate")
        .unwrap();
    let ordinary_program = ordinary_module.procedure(ordinary_id).unwrap();

    // Warm both the canonical bytecode and positive target-classification caches.
    std::hint::black_box(super::canonical_type2parent_program(canonical_program));
    std::hint::black_box(super::canonical_type2parent_target(
        &canonical_module,
        canonical_id,
        canonical_program,
    ));

    let old_non_target_started = std::time::Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(super::canonical_type2parent_program(std::hint::black_box(
            ordinary_program,
        )));
    }
    let old_non_target = old_non_target_started.elapsed();
    let gated_non_target_started = std::time::Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(super::canonical_type2parent_target(
            &ordinary_module,
            ordinary_id,
            std::hint::black_box(ordinary_program),
        ));
    }
    let gated_non_target = gated_non_target_started.elapsed();

    let old_target_started = std::time::Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(super::canonical_type2parent_program(std::hint::black_box(
            canonical_program,
        )));
    }
    let old_target = old_target_started.elapsed();
    let cached_target_started = std::time::Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(super::canonical_type2parent_target(
            &canonical_module,
            canonical_id,
            std::hint::black_box(canonical_program),
        ));
    }
    let cached_target = cached_target_started.elapsed();

    eprintln!(
        "type2parent dispatch non_target_old_ms={} gated_ms={} speedup={:.2}x target_old_ms={} cached_ms={} speedup={:.2}x",
        old_non_target.as_millis(),
        gated_non_target.as_millis(),
        old_non_target.as_secs_f64() / gated_non_target.as_secs_f64(),
        old_target.as_millis(),
        cached_target.as_millis(),
        old_target.as_secs_f64() / cached_target.as_secs_f64(),
    );
}

fn execute_source(source: &str, argument: f32) -> Value {
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");
    execute(&program, &[Value::number(argument)]).expect("procedure should execute")
}

#[test]
fn procedure_specs_resolve_implicit_owner_calls_through_the_path_index() {
    let source = parse(
        "/datum/example/proc/value()\n\treturn 17\n/datum/example/proc/read()\n\treturn value()\n",
    )
    .expect("source should parse");
    let specs = [
        ProcedureSpec {
            path: "/datum/example/proc/value@0".to_owned(),
            definition: &source.definitions[0],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
        ProcedureSpec {
            path: "/datum/example/proc/read@0".to_owned(),
            definition: &source.definitions[1],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
    ];
    let module = compile_module_specs(&specs).expect("implicit owner call should resolve");
    let entry = module
        .procedure_id("/datum/example/proc/read@0")
        .expect("read entry should exist");
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(17.0)));
}

#[test]
fn text_template_fills_empty_and_whitespace_holes_and_honors_escaped_brackets() {
    let syntax =
        parse("/proc/run()\n\treturn text(\"before [] [ ] \\[literal\\] after\", \"one\", 2)\n")
            .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("text() should compile");
    let entry = module.procedure_id("/proc/run").expect("entry");
    assert_eq!(
        execute_module(&module, entry, &[]),
        Ok(Value::text("before one 2 [literal] after"))
    );
}

#[test]
fn interpolated_string_unescapes_literal_brackets_around_expression() {
    let syntax = parse("/proc/run()\n\tvar/value = \"-1,\"\n\treturn \"\\[[value]0\\]\"\n")
        .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("interpolation should compile");
    let entry = module.procedure_id("/proc/run").expect("entry");

    assert_eq!(
        execute_module(&module, entry, &[]),
        Ok(Value::text("[-1,0]")),
    );
}

#[test]
fn ordinary_quoted_strings_decode_escapes_without_changing_raw_strings() {
    let syntax = parse(concat!(
        "/proc/run()\n",
        "\tvar/value = 7\n",
        "\treturn list(\"line\\nnext\\ttab\\\\slash\\\"quote\", ",
        "\"\\[value]\", \"\\\\[value]\", @'\\n\\t\\\\')\n",
    ))
    .expect("escaped string source should parse");
    let module = compile_module(&syntax.definitions).expect("escaped strings should compile");
    let mut state = ExecutionState::new();
    let result = execute_module_in_state(
        &module,
        module.procedure_id("/proc/run").expect("entry"),
        &[],
        &mut state,
    )
    .expect("escaped strings should execute");
    let Value::List(result) = result else {
        panic!("escaped string fixture should return a list");
    };
    let values = state
        .heap()
        .list(result)
        .unwrap()
        .positions()
        .map(|(_, value)| value.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        values,
        vec![
            Value::text("line\nnext\ttab\\slash\"quote"),
            Value::text("[value]"),
            Value::text("\\7"),
            Value::text(r"\n\t\\"),
        ]
    );
}

#[test]
fn apply_text_macros_runtime_improper_prefix_terminates_and_removes_marker() {
    let syntax = parse(
            r#"/proc/apply_text_macros(string)
	var/next_backslash = findtext(string, "\\")
	if(!next_backslash)
		return string
	var/leng = length(string)
	var/next_space = findtext(string, " ", next_backslash + length(string[next_backslash]))
	if(!next_space)
		next_space = leng - next_backslash
	if(!next_space)
		return string
	var/base = next_backslash == 1 ? "" : copytext(string, 1, next_backslash)
	var/macro = lowertext(copytext(string, next_backslash + length(string[next_backslash]), next_space))
	var/rest = next_backslash > leng ? "" : copytext(string, next_space + length(string[next_space]))
	switch(macro)
		if("proper")
			rest = text("\proper []", rest)
		if("improper")
			rest = text("\improper []", rest)
		else
			return base
	. = base
	if(rest)
		. += .(rest)
"#,
        )
        .expect("apply_text_macros fixture should parse");
    let module = compile_module(&syntax.definitions).expect("text macro fixture should compile");
    let entry = module
        .procedure_id("/proc/apply_text_macros")
        .expect("apply_text_macros entry exists");
    assert_eq!(
        execute_module(
            &module,
            entry,
            &[Value::text(r"\improper Operative Remembrance Plaque")],
        ),
        Ok(Value::text("Operative Remembrance Plaque")),
    );
}

#[test]
fn text_macro_prefix_suffix_roman_and_pronoun_family_uses_original_value() {
    let syntax = parse(
        r#"/proc/run(item)
	return list(
		text("\the []", item),
		text("\A []", item),
		text("\roman []", 7),
		text("[]\th", 2),
		text("[]\s", 2),
		text("[]\he", item),
		text("[]\she", item),
		text("[]\his", item),
		text("[]\himself", item),
		text("[]\herself", item),
		text("[]\hers", item),
		"\the [item]",
		"[item]\hers",
		"\Roman[7]",
	)
"#,
    )
    .expect("text macro family fixture should parse");
    let module = compile_module(&syntax.definitions).expect("text macros should compile");
    let mut state = ExecutionState::new();
    let item = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/example").unwrap());
    state
        .heap_mut()
        .set_datum_field(
            item,
            FieldName::parse("name").unwrap(),
            Value::text("apple"),
        )
        .unwrap();
    state
        .heap_mut()
        .set_datum_field(
            item,
            FieldName::parse("gender").unwrap(),
            Value::text("female"),
        )
        .unwrap();
    let result = execute_module_in_state(
        &module,
        module.procedure_id("/proc/run").expect("run entry exists"),
        &[Value::Datum(item)],
        &mut state,
    )
    .expect("text macro family should execute");
    let Value::List(result) = result else {
        panic!("text macro fixture should return a list");
    };
    let values = state
        .heap()
        .list(result)
        .unwrap()
        .positions()
        .map(|(_, value)| value.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        values,
        [
            "the apple",
            "An apple",
            "vii",
            "2nd",
            "2s",
            "appleshe",
            "appleshe",
            "appleher",
            "appleherself",
            "appleherself",
            "applehers",
            "the apple",
            "applehers",
            "VII",
        ]
        .map(Value::text)
    );
}

#[test]
fn crash_expression_is_lazy_behind_null_conditional_access() {
    let syntax = parse(
            "/proc/run()\n\tvar/value = null\n\tvalue?.field = CRASH(\"skipped rhs\")\n\tvar/result = value?.method(CRASH(\"skipped argument\"))\n\treturn isnull(result)\n",
        )
        .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("CRASH expression should compile");
    let entry = module.procedure_id("/proc/run").expect("entry");
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(1.0)));
}

#[test]
fn newlist_uses_full_inherited_defaults_and_constructor_semantics() {
    let syntax = parse(
            "/obj/item/New()\n\tsrc.new_ran = 7\n/proc/run()\n\treturn newlist(/obj/item/apc_powercord, /obj/item/apc_powercord)\n",
        )
        .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("newlist should compile");
    let entry = module.procedure_id("/proc/run").expect("entry");
    let apc = TypePath::parse("/obj/item/apc_powercord").unwrap();
    let mut state = ExecutionState::new();
    state.set_type_parents(BTreeMap::from([
        (TypePath::parse("/datum").unwrap(), None),
        (
            TypePath::parse("/atom").unwrap(),
            Some(TypePath::parse("/datum").unwrap()),
        ),
        (
            TypePath::parse("/atom/movable").unwrap(),
            Some(TypePath::parse("/atom").unwrap()),
        ),
        (
            TypePath::parse("/obj").unwrap(),
            Some(TypePath::parse("/atom/movable").unwrap()),
        ),
        (
            TypePath::parse("/obj/item").unwrap(),
            Some(TypePath::parse("/obj").unwrap()),
        ),
        (apc.clone(), Some(TypePath::parse("/obj/item").unwrap())),
    ]));
    state.set_initial_values(BTreeMap::from([(
        apc,
        BTreeMap::from([(field("gc_destroyed"), Value::number(0.0))]),
    )]));

    let Value::List(items) = execute_module_in_state(&module, entry, &[], &mut state)
        .expect("newlist should construct both items")
    else {
        panic!("newlist should return a list");
    };
    let created = state
        .heap()
        .list(items)
        .unwrap()
        .positions()
        .map(|(_, value)| match value {
            Value::Datum(datum) => *datum,
            value => panic!("newlist should contain datums, received {value}"),
        })
        .collect::<Vec<_>>();
    assert_eq!(created.len(), 2);
    assert_ne!(created[0], created[1]);
    for datum in created {
        assert_eq!(
            datum_field_or_initial(&state, datum, &field("gc_destroyed")),
            Ok(Value::number(0.0)),
            "newlist must apply effective inherited defaults before New",
        );
        assert_eq!(
            state.heap().datum_field(datum, &field("new_ran")),
            Ok(&Value::number(7.0)),
            "newlist must dispatch inherited New for each fresh object",
        );
    }
}

#[test]
fn src_assignment_rebinds_subsequent_bare_method_dispatch() {
    let syntax = parse(
            "/datum/A/proc/who()\n\treturn 1\n/datum/B/proc/who()\n\treturn 2\n/datum/A/proc/test()\n\tsrc = new /datum/B\n\treturn who()\n/proc/run()\n\tvar/datum/A/item = new /datum/A\n\treturn item.test()\n",
        )
        .expect("source should parse");
    let definitions = &syntax.definitions;
    let specs = [
        ProcedureSpec {
            path: "/datum/A/proc/who".to_owned(),
            definition: &definitions[0],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
        ProcedureSpec {
            path: "/datum/B/proc/who".to_owned(),
            definition: &definitions[1],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
        ProcedureSpec {
            path: "/datum/A/proc/test".to_owned(),
            definition: &definitions[2],
            parent: None,
            static_calls: BTreeMap::from([("who".to_owned(), 0)]),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
        ProcedureSpec {
            path: "/proc/run".to_owned(),
            definition: &definitions[3],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
    ];
    let module = compile_module_specs(&specs).expect("src rebinding family compiles");
    let entry = module.procedure_id("/proc/run").expect("entry");
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(2.0)));
}

#[test]
fn exact_list_allocation_constructs_heap_list_identity() {
    let syntax = parse("/proc/run()\n\tvar/list/items = new /list\n\treturn islist(items)\n")
        .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("new /list compiles");
    let entry = module.procedure_id("/proc/run").expect("entry");
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(1.0)));
}

#[test]
fn modified_type_construction_applies_overrides_after_declared_initial_values() {
    let syntax = parse(
            "/proc/run()\n\tvar/datum/plain = new /datum/example\n\tvar/datum/changed = new /datum/example {a=6;b=8}\n\treturn plain.a + plain.b + changed.a + changed.b\n",
        )
        .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("modified type should compile");
    let entry = module.procedure_id("/proc/run").expect("entry");
    let path = TypePath::parse("/datum/example").expect("type path");
    let mut state = ExecutionState::new();
    state.set_initial_values(BTreeMap::from([(
        path,
        BTreeMap::from([
            (field("a"), Value::number(5.0)),
            (field("b"), Value::number(7.0)),
        ]),
    )]));
    assert_eq!(
        execute_module_in_state(&module, entry, &[], &mut state),
        Ok(Value::number(26.0))
    );
}

#[test]
fn modified_type_paths_are_list_keys_and_dynamic_new_operands() {
    let syntax = parse(
            "/proc/run()\n\tvar/amount = 15\n\tvar/list/cache = list(/datum/example{a = amount} = 4)\n\tvar/kind = /datum/example{a = amount}\n\tvar/datum/created = new kind\n\treturn cache[kind] + created.a + created.b\n",
        )
        .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("modified path values compile");
    let entry = module.procedure_id("/proc/run").expect("entry");
    let path = TypePath::parse("/datum/example").expect("type path");
    let mut state = ExecutionState::new();
    state.set_initial_values(BTreeMap::from([(
        path,
        BTreeMap::from([
            (field("a"), Value::number(1.0)),
            (field("b"), Value::number(2.0)),
        ]),
    )]));

    assert_eq!(
        execute_module_in_state(&module, entry, &[], &mut state),
        Ok(Value::number(21.0)),
        "the modified path must retain its evaluated key identity and override defaults after allocation"
    );
}

#[test]
fn runtime_new_accepts_current_src_type_field_in_destroy_recovery_shape() {
    let syntax = parse(
            "/datum/wound_pregen_data/Destroy(force)\n\tvar/error_message = \"[src] destroyed\"\n\tif(!force)\n\t\treturn 1\n\tvar/replacement = new src.type\n\treturn replacement.type\n",
        )
        .expect("wound recovery source should parse");
    compile_module(&syntax.definitions)
        .expect("src.type is a runtime new operand, not an ordinary local");
}

#[test]
fn infinity_constants_interpolate_and_complex_raw_strings_use_custom_delimiters() {
    let syntax = parse(
            "/proc/run()\n\tvar/a = 1#INF\n\tvar/b = -1#INF\n\tvar/c = -1#IND\n\tvar/raw = @(END)\nhello worldEND\n\treturn (\"[a]\" == \"inf\") + (\"[b]\" == \"-inf\") + (\"[c]\" == \"nan\") + (raw == \"hello world\")\n",
        )
        .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("constant expressions compile");
    let entry = module.procedure_id("/proc/run").expect("entry");
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(4.0)));
}

#[test]
fn assign_into_is_direct_assignment_and_output_statement_does_not_shift_receiver() {
    let syntax = parse(
            "/proc/run()\n\tvar/value = 5\n\tvalue := 10\n\tvalue << 1\n\tvar/other = 3\n\tother := null\n\treturn value + isnull(other)\n",
        )
        .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("operator statements compile");
    let entry = module.procedure_id("/proc/run").expect("entry");
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(11.0)));
}

#[test]
fn comma_locals_logical_assignment_and_procedure_scope_name_follow_dm_rules() {
    let syntax = parse(
            "/datum/proc/foo()\n\tset name = \"display\"\n\treturn\n/proc/run()\n\tvar/v1,v2\n\tv1 = 0\n\tv2 = 1\n\tv1 ||= 5\n\tv2 &&= 7\n\treturn v1 + v2 + (/datum/proc/foo::name == \"foo\")\n",
        )
        .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("operator parser family compiles");
    let entry = module.procedure_id("/proc/run").expect("entry");
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(13.0)));
}

#[test]
fn alist_constructs_ordered_key_value_storage_and_preserves_its_runtime_type() {
    let syntax = parse(
            "/proc/run()\n\tvar/alist/inner = alist(\"one\" = 1, \"two\" = 2)\n\tvar/alist/items = alist(\"left\" = inner, \"right\" = 3)\n\titems += alist(\"right\" = 9, \"extra\" = 4)\n\tvar/alist/copy = items.Copy()\n\tif(!istype(items[\"left\"], /alist)) return 0\n\tif(items[\"right\"] != 3 || items[\"extra\"] != 4) return 0\n\tif(length(items) != 3 || !istype(copy, /alist)) return 0\n\treturn copy[\"left\"][\"two\"]\n",
        )
        .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("alist family should compile");
    let entry = module.procedure_id("/proc/run").expect("entry");
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(2.0)));
}

#[test]
fn alist_numeric_keys_remain_associative_across_nested_writes_and_mutation() {
    let syntax = parse(concat!(
        "/proc/run()\n",
        "\tvar/alist/groups = alist(1 = list(), 2 = list(), 3 = list())\n",
        "\tgroups[2][\"trait\"] = 7\n",
        "\tgroups[2][\"trait\"]++\n",
        "\tgroups[3] += list(\"x\")\n",
        "\tvar/list/ordinary = list(10, 20)\n",
        "\treturn groups[2][\"trait\"] + length(groups[3]) + ordinary[2]\n",
    ))
    .expect("numeric alist source should parse");
    let module = compile_module(&syntax.definitions).expect("numeric alist should compile");
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/run").unwrap(), &[]),
        Ok(Value::number(29.0)),
    );
}

#[test]
fn list_length_is_writable_and_values_cut_filters_associations_by_numeric_value() {
    let syntax = parse(
            "/proc/run()\n\tvar/list/items = list(\"a\" = 1, \"b\" = 2, \"c\" = 0)\n\tvar/removed = values_cut_over(items, 1, TRUE)\n\tvar/list/plain = list(1, 2, 3, 4)\n\tplain.len--\n\tplain.len -= 1\n\tplain.len = 1\n\treturn removed * 10 + length(items) + length(plain)\n",
        )
        .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("list mutation family compiles");
    let entry = module.procedure_id("/proc/run").expect("entry");
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(22.0)));

    let negative = parse(
            "/proc/run()\n\tvar/list/items = list()\n\titems.len--\n\titems.len = -4\n\tvar/list/overlays = list(1)\n\toverlays.len += -2\n\treturn items.len + overlays.len\n",
        )
        .expect("negative source parses");
    let negative = compile_module(&negative.definitions).expect("negative source compiles");
    let entry = negative.procedure_id("/proc/run").expect("entry");
    assert_eq!(
        execute_module(&negative, entry, &[]),
        Ok(Value::number(0.0))
    );
}

#[test]
fn condition_tokens_accepts_braced_macro_conditions_with_following_tokens() {
    let tokens = lex("if(!(flags_1 & INITIALIZED_1)) { var/previous = 1")
        .expect("condition source should lex");
    let condition = condition_tokens(&tokens[1..], "if").expect("condition should compile");
    assert!(matches!(condition[0].kind, TokenKind::Operator(ref op) if op == "!"));
}

#[test]
fn try_catch_binds_arbitrary_thrown_values_and_skips_catch_normally() {
    let syntax = parse(
            "/proc/run(should_throw)\n\tvar/result = 1\n\ttry\n\t\tif (should_throw)\n\t\t\tthrow 5\n\t\tresult = 2\n\tcatch(var/error)\n\t\tresult = error + 10\n\treturn result\n",
        )
        .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("try/catch should compile");
    let entry = module.procedure_id("/proc/run").expect("entry");
    assert_eq!(
        execute_module(&module, entry, &[Value::number(0.0)]),
        Ok(Value::number(2.0))
    );
    assert_eq!(
        execute_module(&module, entry, &[Value::number(1.0)]),
        Ok(Value::number(15.0))
    );
}

#[test]
fn both_returning_try_catch_omits_dead_program_end_jump() {
    let syntax = parse(
            "/proc/run(should_throw)\n\ttry\n\t\tif(should_throw)\n\t\t\tthrow 5\n\t\treturn 7\n\tcatch\n\t\treturn 9\n",
        )
        .expect("both-returning try/catch should parse");
    let module = compile_module(&syntax.definitions).expect("try/catch should compile");
    let entry = module.procedure_id("/proc/run").expect("entry");
    let program = module.procedure(entry).expect("compiled program");

    assert!(!program.instructions.iter().any(
            |instruction| matches!(instruction, Instruction::Jump(target) if *target >= program.instructions.len())
        ));
    assert_eq!(
        execute_module(&module, entry, &[Value::number(0.0)]),
        Ok(Value::number(7.0)),
    );
    assert_eq!(
        execute_module(&module, entry, &[Value::number(1.0)]),
        Ok(Value::number(9.0)),
    );
}

#[test]
fn thrown_values_unwind_calls_and_nested_handlers_choose_the_nearest_catch() {
    let syntax = parse(
            "/proc/run()\n\tvar/result\n\ttry\n\t\ttry\n\t\t\thelper()\n\t\tcatch(var/inner)\n\t\t\tresult = inner + 1\n\t\t\tthrow 10\n\tcatch(var/outer)\n\t\tresult += outer\n\treturn result\n/proc/helper()\n\tthrow 5\n",
        )
        .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("nested try/catch should compile");
    let entry = module.procedure_id("/proc/run").expect("entry");
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(16.0)));
}

#[test]
fn catch_without_binding_consumes_the_exception_and_uncaught_throw_errors() {
    let caught = parse("/proc/run()\n\ttry\n\t\tthrow \"test\"\n\tcatch\n\t\treturn 7\n")
        .expect("source should parse");
    let caught = compile_module(&caught.definitions).expect("catch should compile");
    let entry = caught.procedure_id("/proc/run").expect("entry");
    assert_eq!(execute_module(&caught, entry, &[]), Ok(Value::number(7.0)));

    let uncaught = parse("/proc/run()\n\tthrow \"test\"\n").expect("source should parse");
    let uncaught = compile_module(&uncaught.definitions).expect("throw should compile");
    let entry = uncaught.procedure_id("/proc/run").expect("entry");
    let error = execute_module(&uncaught, entry, &[]).expect_err("throw should escape");
    assert!(error.message.contains("uncaught exception:"));
    assert!(error.message.contains("test"));
}

#[test]
fn absolute_type_path_expressions_lower_to_type_path_values() {
    let syntax =
        parse("/proc/type_path()\n\treturn /obj/item/tool\n").expect("source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("type path expression should compile");

    assert_eq!(
        execute(&program, &[]),
        Ok(Value::TypePath(TypePath::parse("/obj/item/tool").unwrap()))
    );
}

#[test]
fn resource_literals_and_file_constructor_are_distinct_from_plain_text() {
    let syntax = parse(
            "/proc/resource_path()\n\treturn 'sound/effects/piano_hit.ogg'\n/proc/file_kinds()\n\treturn isfile(\"maps/test.dmm\") * 100 + isfile(file(\"maps/test.dmm\")) * 10 + isfile('maps/test.dmm')\n",
        )
            .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("resource values should compile");

    assert_eq!(
        execute_module(
            &module,
            module.procedure_id("/proc/resource_path").unwrap(),
            &[],
        ),
        Ok(Value::file("sound/effects/piano_hit.ogg"))
    );
    assert_eq!(
        execute_module(
            &module,
            module.procedure_id("/proc/file_kinds").unwrap(),
            &[],
        ),
        Ok(Value::number(11.0))
    );
}

#[test]
fn top_level_semicolons_split_macro_style_statements() {
    let syntax = parse("/proc/semicolon_statements()\n\tvar/value = 1; value += 2; return value\n")
        .expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("semicolon-separated statements should compile");

    assert_eq!(execute(&program, &[]), Ok(Value::number(3.0)));
}

#[test]
fn compact_brace_macro_body_is_lowered_as_an_indented_block() {
    // This is the shape produced by Monkestation's lazy-list helpers at
    // the bottom of an already-indented `for`/`if` body.  The statement
    // following `}` must remain outside the inner conditional.
    let syntax = parse(
            "/proc/compact_macro(flag)\n\tvar/value = 0\n\tif(flag)\n\t\tif(!value) { value = 4; } value += 3;\n\treturn value\n",
        )
        .expect("source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("compact macro body should compile");

    assert_eq!(
        execute(&program, &[Value::number(1.0)]),
        Ok(Value::number(7.0))
    );
    assert_eq!(
        execute(&program, &[Value::number(0.0)]),
        Ok(Value::number(0.0))
    );
}

#[test]
fn compact_do_while_macro_body_preserves_nested_block_indentation() {
    // Trait helper macros use a `do { ... } while (0)` wrapper to make a
    // multi-statement expansion behave as one source statement.  Its
    // nested brace blocks must remain children of the synthetic `do`
    // block, while the trailing `while` returns to the caller's level.
    let syntax = parse(
            "/proc/compact_do_macro(flag)\n\tvar/value = 0\n\tdo { var/local = 1; if(flag) { local += 2; } else { local += 4; } value = local; } while(0)\n\treturn value\n",
        )
        .expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("compact do/while macro body should compile");

    assert_eq!(
        execute(&program, &[Value::number(1.0)]),
        Ok(Value::number(3.0))
    );
    assert_eq!(
        execute(&program, &[Value::number(0.0)]),
        Ok(Value::number(5.0))
    );
}

#[test]
fn conditional_accepts_a_preprocessor_retained_opening_brace() {
    let syntax = parse("/proc/brace_if(flag)\n\tif(flag) {\n\t\treturn 1\n\t}\n\treturn 0\n")
        .expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("brace-terminated condition should compile");

    assert_eq!(
        execute(&program, &[Value::number(1.0)]),
        Ok(Value::number(1.0))
    );
    assert_eq!(
        execute(&program, &[Value::number(0.0)]),
        Ok(Value::number(0.0))
    );
}

#[test]
fn conditional_accepts_same_line_return_body() {
    let syntax = parse("/proc/inline_if(flag)\n\tif(flag) return 7\n\treturn 3\n")
        .expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("same-line conditional body should compile");

    assert_eq!(
        execute(&program, &[Value::number(1.0)]),
        Ok(Value::number(7.0))
    );
    assert_eq!(
        execute(&program, &[Value::number(0.0)]),
        Ok(Value::number(3.0))
    );
}

#[test]
fn conditional_accepts_same_line_else_body() {
    let syntax = parse("/proc/inline_else(flag)\n\tif(flag) return 7\n\telse return 3\n")
        .expect("source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("same-line else body should compile");

    assert_eq!(
        execute(&program, &[Value::number(1.0)]),
        Ok(Value::number(7.0))
    );
    assert_eq!(
        execute(&program, &[Value::number(0.0)]),
        Ok(Value::number(3.0))
    );
}

#[test]
fn conditional_preserves_else_if_condition_with_inline_body() {
    let syntax = parse(
            "/proc/inline_else_if(value)\n\tif(value == 1) return 1\n\telse if(value == 2) return 2\n\telse return 3\n",
        )
        .expect("source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("same-line else-if body should compile");

    assert_eq!(
        execute(&program, &[Value::number(1.0)]),
        Ok(Value::number(1.0))
    );
    assert_eq!(
        execute(&program, &[Value::number(2.0)]),
        Ok(Value::number(2.0))
    );
    assert_eq!(
        execute(&program, &[Value::number(3.0)]),
        Ok(Value::number(3.0))
    );
}

#[test]
fn conditional_accepts_same_line_continue_body() {
    let syntax = parse(
            "/proc/inline_continue()\n\tvar/total = 0\n\tfor(var/i = 0; i < 4; i++)\n\t\tif(i == 2) continue\n\t\ttotal += i\n\treturn total\n",
        )
        .expect("source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("same-line continue body should compile");

    assert_eq!(execute(&program, &[]), Ok(Value::number(4.0)));
}

#[test]
fn new_type_path_allocates_a_datum_and_discards_constructor_arguments() {
    let syntax =
        parse("/proc/build()\n\tvar/item = new /datum/example(41, \"ignored\")\n\treturn item\n")
            .expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("new should compile");
    assert!(matches!(
        program.instructions.as_slice(),
        [
            Instruction::PushTypePath(_),
            Instruction::PushNumber(_),
            Instruction::PushText(_),
            Instruction::AllocateDatum {
                argument_count: 2,
                ..
            },
            Instruction::StoreLocal(_),
            Instruction::LoadLocal(_),
            Instruction::Return,
        ]
    ));
    let mut state = ExecutionState::new();
    let value = execute_in_state(&program, &[], &mut state).expect("new should execute");
    let Value::Datum(datum) = value else {
        panic!("new should return a datum");
    };
    assert_eq!(
        state
            .heap()
            .datum(datum)
            .expect("datum must be live")
            .type_path(),
        &TypePath::parse("/datum/example").unwrap()
    );
}

#[test]
fn named_constructor_arguments_bind_sparse_new_parameters() {
    let syntax = parse(concat!(
            "/datum/media_source/object/New(track, volume, mixer_channel, atom/movable/source, max_distance = 10)\n",
            "\tsrc.received_track = track\n",
            "\tsrc.received_volume = volume\n",
            "\tsrc.received_mixer_channel = mixer_channel\n",
            "\tsrc.received_source = source\n",
            "\tsrc.received_max_distance = max_distance\n",
            "/obj/machinery/jukebox/proc/build_source()\n",
            "\treturn new /datum/media_source/object(volume = 100, mixer_channel = 1019, source = src)\n",
            "/proc/run()\n",
            "\tvar/obj/machinery/jukebox/jukebox = new\n",
            "\tvar/datum/media_source/object/media = jukebox.build_source()\n",
            "\treturn isnull(media.received_track) && media.received_volume == 100 && media.received_mixer_channel == 1019 && media.received_source == jukebox && media.received_max_distance == 10\n",
        ))
        .expect("jukebox constructor fixture should parse");
    let module =
        compile_module(&syntax.definitions).expect("jukebox constructor fixture should compile");
    let build_source = module
        .procedure(
            module
                .procedure_id("/obj/machinery/jukebox/proc/build_source")
                .unwrap(),
        )
        .unwrap();
    assert!(build_source.instructions.iter().any(|instruction| matches!(
        instruction,
        Instruction::AllocateDatum {
            argument_names,
            ..
        } if argument_names == &[
            Some("volume".to_owned()),
            Some("mixer_channel".to_owned()),
            Some("source".to_owned()),
        ]
    )));
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/run").unwrap(), &[],),
        Ok(Value::number(1.0))
    );
}

#[test]
fn named_dynamic_call_arguments_bind_sparse_reagent_parameters() {
    let syntax = parse(concat!(
            "/datum/reagents/proc/add_reagent(datum/reagent/reagent_type, amount, list/data = null, reagtemp = 293.15, added_purity = null, added_ph = null, no_react = 0, override_base_ph = 0, ignore_splitting = 0, datum/callback/creation_callback = null)\n",
            "\treturn reagent_type == /datum/reagent/blood && amount == 200 && islist(data) && reagtemp == 293.15 && isnull(added_purity) && isnull(added_ph) && no_react == 0 && override_base_ph == 0 && ignore_splitting == 0 && istype(creation_callback, /datum/callback)\n",
            "/proc/run()\n",
            "\tvar/datum/reagents/reagents = new\n",
            "\tvar/datum/callback/callback = new\n",
            "\treturn reagents.add_reagent(/datum/reagent/blood, 200, list(\"blood_type\" = \"A+\"), creation_callback = callback)\n",
        ))
        .expect("blood-pack reagent fixture should parse");
    let module =
        compile_module(&syntax.definitions).expect("blood-pack reagent fixture should compile");
    let run = module
        .procedure(module.procedure_id("/proc/run").unwrap())
        .unwrap();
    assert!(run.instructions.iter().any(|instruction| matches!(
        instruction,
        Instruction::CallDynamic { argument_names, .. }
            if argument_names == &[
                None,
                None,
                None,
                Some("creation_callback".to_owned()),
            ]
    )));
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/run").unwrap(), &[]),
        Ok(Value::number(1.0))
    );
}

#[test]
fn callback_varargs_append_runtime_arguments_without_a_phantom_null() {
    let syntax = parse(concat!(
        "/datum/receiver/proc/on_created(value)\n",
        "\treturn value\n",
        "/datum/callback/New(object, delegate, ...)\n",
        "\tsrc.object = object\n",
        "\tsrc.delegate = delegate\n",
        "\tif(length(args) > 2)\n",
        "\t\tsrc.arguments = args.Copy(3)\n",
        "\telse\n",
        "\t\tsrc.arguments = list()\n",
        "/datum/callback/proc/Invoke(...)\n",
        "\tvar/list/calling_arguments = src.arguments\n",
        "\tif(length(args))\n",
        "\t\tif(length(src.arguments))\n",
        "\t\t\tcalling_arguments = calling_arguments + args\n",
        "\t\telse\n",
        "\t\t\tcalling_arguments = args\n",
        "\treturn call(src.object, src.delegate)(arglist(calling_arguments))\n",
        "/proc/run()\n",
        "\tvar/datum/receiver/receiver = new\n",
        "\tvar/datum/callback/callback = new(receiver, \"on_created\")\n",
        "\treturn callback.Invoke(73)\n",
    ))
    .expect("callback fixture should parse");
    let module = compile_module(&syntax.definitions).expect("callback fixture should compile");
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/run").unwrap(), &[]),
        Ok(Value::number(73.0))
    );
}

#[test]
fn sleeping_constructor_suspends_the_caller_until_new_finishes() {
    let syntax = parse(concat!(
        "/datum/parsed_map/New()\n",
        "\tvar/list/grid_sets = list()\n",
        "\tgrid_sets += \"first\"\n",
        "\tsleep(1)\n",
        "\tgrid_sets += \"second\"\n",
        "\tsrc.gridSets = grid_sets\n",
        "\treturn 999\n",
        "/proc/load_map()\n",
        "\tvar/datum/parsed_map/parsed = new\n",
        "\treturn length(parsed.gridSets)\n",
    ))
    .expect("sleeping constructor source should parse");
    let module = compile_module(&syntax.definitions)
        .expect("sleeping constructor and caller should compile");
    let entry = module
        .procedure_id("/proc/load_map")
        .expect("map loader entry should exist");
    let mut state = ExecutionState::new();

    assert_eq!(
        execute_module_in_state(&module, entry, &[], &mut state),
        Ok(Value::Null),
        "the outer loader must yield with its synchronous New() call"
    );
    assert_eq!(state.scheduled_task_count(), 1);
    assert_eq!(
        advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
        Ok(vec![Value::number(2.0)]),
        "New() must finish before the caller observes the constructed datum"
    );
}

#[test]
fn scheduler_yield_collects_temporary_lists_without_breaking_frame_aliases() {
    let syntax = parse(concat!(
        "/proc/work()\n",
        "\tvar/list/kept = list(list(7))\n",
        "\tvar/list/discard = list()\n",
        "\tdiscard = list()\n",
        "\tdiscard = list()\n",
        "\tdiscard = list()\n",
        "\tdiscard = list()\n",
        "\tsleep(1)\n",
        "\treturn kept[1][1]\n",
    ))
    .expect("list collection fixture should parse");
    let module = compile_module(&syntax.definitions).expect("fixture should compile");
    let entry = module.procedure_id("/proc/work").unwrap();
    let mut state = ExecutionState::new();
    state.next_list_collection = 4;

    assert_eq!(
        execute_module_in_state(&module, entry, &[], &mut state),
        Ok(Value::Null)
    );
    assert_eq!(state.scheduled_task_count(), 1);
    assert_eq!(
        state.heap().live_list_count(),
        3,
        "only the nested kept lists and the current local should survive"
    );
    assert_eq!(
        advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
        Ok(vec![Value::number(7.0)])
    );
}

#[test]
fn arglist_caller_and_callee_keep_the_original_list_across_sleep() {
    let syntax = parse(concat!(
        "/datum/atom/proc/Initialize(mapload)\n",
        "\tsleep(1)\n",
        "\treturn 1\n",
        "/datum/atoms/proc/InitAtom(datum/atom/A, list/arguments)\n",
        "\tvar/result = A.Initialize(arglist(arguments))\n",
        "\tif(result == 1)\n",
        "\t\treturn arguments[1]\n",
        "\treturn 0\n",
        "/datum/atoms/proc/CreateAtoms()\n",
        "\tvar/list/mapload_arg = list(TRUE)\n",
        "\tvar/datum/atom/A = new\n",
        "\treturn src.InitAtom(A, mapload_arg)\n",
    ))
    .expect("InitAtom-shaped sleeping arglist source should parse");
    let module = compile_module(&syntax.definitions).expect("fixture should compile");
    let entry = module
        .procedure_id("/datum/atoms/proc/CreateAtoms")
        .expect("CreateAtoms entry should exist");
    let mut state = ExecutionState::new();
    let subsystem = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/atoms").unwrap());
    let context = ExecutionContext::new(Value::Datum(subsystem), Value::Null);
    state.next_list_collection = 1;

    assert_eq!(
        super::execute_module_with_limits_in_context(
            &module,
            entry,
            &[],
            ExecutionLimits::default(),
            &mut state,
            &context,
        ),
        Ok(Value::Null),
    );
    assert_eq!(state.scheduled_task_count(), 1);
    assert_eq!(
        advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
        Ok(vec![Value::number(1.0)]),
        "arglist expansion must not let GC reclaim a caller-visible list",
    );
}

#[test]
fn nested_spawn_minus_one_gc_preserves_outer_frame_lists() {
    let syntax = parse(concat!(
        "/proc/inner()\n",
        "\tspawn(-1)\n",
        "\t\tvar/list/transient = list(1, 2, 3)\n",
        "\t\tsleep(1)\n",
        "\treturn 0\n",
        "/proc/outer()\n",
        "\tvar/list/kept = list(7)\n",
        "\tinner()\n",
        "\treturn kept[1]\n",
    ))
    .expect("nested spawn(-1) GC source should parse");
    let module = compile_module(&syntax.definitions).expect("fixture should compile");
    let mut state = ExecutionState::new();
    state.next_list_collection = 1;

    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/outer").unwrap(),
            &[],
            &mut state,
        ),
        Ok(Value::number(7.0)),
        "a reentrant spawn(-1) collection must retain every outer caller frame",
    );
    assert_eq!(state.scheduled_task_count(), 1);
}

#[test]
fn reentrant_del_gc_preserves_outer_frame_lists() {
    let syntax = parse(concat!(
        "/datum/victim/Del()\n",
        "\tvar/list/transient = list(1, 2, 3)\n",
        "\tsleep(1)\n",
        "/proc/remove(datum/victim/value)\n",
        "\tdel(value)\n",
        "/proc/outer()\n",
        "\tvar/list/kept = list(9)\n",
        "\tvar/datum/victim/value = new\n",
        "\tremove(value)\n",
        "\treturn kept[1]\n",
    ))
    .expect("reentrant Del GC source should parse");
    let module = compile_module(&syntax.definitions).expect("fixture should compile");
    let mut state = ExecutionState::new();
    state.next_list_collection = 1;

    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/outer").unwrap(),
            &[],
            &mut state,
        ),
        Ok(Value::number(9.0)),
        "a reentrant Del collection must retain every outer caller frame",
    );
}

#[test]
fn heap_gc_growth_window_adapts_to_reclaim_yield() {
    assert_eq!(
        adaptive_heap_collection_growth(4_000_000, 100_000),
        MAXIMUM_LOW_YIELD_COLLECTION_GROWTH,
        "a low reclaim yield must still respect the production peak-memory cap",
    );
    assert_eq!(
        adaptive_heap_collection_growth(4_000_000, 400_000),
        MAXIMUM_MODERATE_YIELD_COLLECTION_GROWTH,
        "moderate reclaim yield must respect the same production peak-memory cap",
    );
    assert_eq!(
        adaptive_heap_collection_growth(4_000_000, 2_000_000),
        100_000,
        "high reclaim yield should return to a pressure-sensitive 2.5% window",
    );
}

#[test]
fn heap_gc_growth_window_has_deterministic_memory_bounds() {
    assert_eq!(
        adaptive_heap_collection_growth(1, 0),
        MINIMUM_HEAP_COLLECTION_GROWTH,
    );
    assert_eq!(
        adaptive_heap_collection_growth(usize::MAX, 0),
        MAXIMUM_LOW_YIELD_COLLECTION_GROWTH,
    );
    assert_eq!(
        adaptive_heap_collection_growth(usize::MAX / 2, usize::MAX / 16),
        MAXIMUM_MODERATE_YIELD_COLLECTION_GROWTH,
    );
    assert_eq!(
        adaptive_heap_collection_growth(usize::MAX / 4, usize::MAX / 2),
        MAXIMUM_HIGH_YIELD_COLLECTION_GROWTH,
    );
}

#[test]
fn bulk_init_growth_matches_base_policy_before_a_low_yield_streak() {
    // Below the streak threshold the window is exactly the base policy, so an
    // ordinary churny heap is unaffected by bulk-init awareness.
    for streak in 0..BULK_INIT_LOW_YIELD_STREAK {
        assert_eq!(
            bulk_init_aware_collection_growth(
                10_000_000,
                50_000,
                streak,
                DEFAULT_HEAP_IDENTITY_CEILING
            ),
            adaptive_heap_collection_growth(10_000_000, 50_000),
            "streak {streak} must not widen the window",
        );
    }
}

#[test]
fn bulk_init_growth_widens_toward_the_ceiling_during_a_monotonic_phase() {
    let ceiling = 16_000_000;
    let growth =
        bulk_init_aware_collection_growth(10_000_000, 50_000, BULK_INIT_LOW_YIELD_STREAK, ceiling);
    assert_eq!(
        growth, 6_000_000,
        "a sustained near-zero-yield streak runs the window to the ceiling",
    );
    assert_eq!(
        10_000_000 + growth,
        ceiling,
        "the next collection lands exactly at the identity ceiling",
    );
    assert!(
        growth > adaptive_heap_collection_growth(10_000_000, 50_000) * 10,
        "the widened window must dwarf the low-yield cap it replaces",
    );
}

#[test]
fn bulk_init_growth_step_is_a_bounded_doubling() {
    // Far from the ceiling the window grows by at most the current live size,
    // so committed memory at worst doubles between passes.
    assert_eq!(
        bulk_init_aware_collection_growth(2_000_000, 0, BULK_INIT_LOW_YIELD_STREAK, 64_000_000),
        2_000_000,
    );
}

#[test]
fn bulk_init_growth_resumes_base_policy_at_or_above_the_ceiling() {
    for &live in &[16_000_000usize, 20_000_000] {
        assert_eq!(
            bulk_init_aware_collection_growth(live, 0, BULK_INIT_LOW_YIELD_STREAK + 5, 16_000_000),
            adaptive_heap_collection_growth(live, 0),
            "at/above the ceiling the tight base window forces frequent passes",
        );
    }
}

#[test]
fn heap_collection_tracks_and_resets_the_near_zero_yield_streak() {
    let mut state = ExecutionState::new();

    // A large, fully-reachable list population models a monotonic bulk-init
    // phase: every collection visits many identities and frees ~none of them.
    for _ in 0..8_192 {
        let list = state.heap.allocate_list();
        state.host_value_roots.push(Value::List(list));
    }
    for expected in 1..=4u32 {
        state.next_list_collection = 1;
        state.maybe_collect_unreachable_lists(&[]);
        assert_eq!(
            state.low_yield_collection_streak, expected,
            "each near-zero-yield collection extends the streak",
        );
    }

    // Dropping every root makes the next collection high-yield, which ends the
    // phase and snaps the streak back to zero.
    state.host_value_roots.clear();
    state.next_list_collection = 1;
    state.maybe_collect_unreachable_lists(&[]);
    assert_eq!(
        state.low_yield_collection_streak, 0,
        "a high-yield collection ends the bulk-allocation phase",
    );
}

#[test]
fn list_gc_roots_later_same_tick_scheduler_continuations() {
    let syntax = parse(concat!(
        "/proc/collector()\n",
        "\tsleep(1)\n",
        "\tsleep(1)\n",
        "/proc/victim(target)\n",
        "\tvar/list/held = list(7)\n",
        "\tsleep(1)\n",
        "\ttarget.observed = held[1]\n",
        "\treturn held[1]\n",
        "/proc/start(target)\n",
        "\tspawn(0) collector()\n",
        "\tspawn(0) victim(target)\n",
    ))
    .expect("same-tick scheduler GC fixture should parse");
    let module = compile_module(&syntax.definitions).expect("fixture should compile");
    let mut state = ExecutionState::new();
    let holder = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/holder").unwrap());

    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/start").unwrap(),
            &[Value::Datum(holder)],
            &mut state,
        ),
        Ok(Value::Null)
    );
    assert_eq!(state.scheduled_task_count(), 2);
    assert_eq!(
        advance_scheduler(&module, 0, ExecutionLimits::default(), &mut state),
        Ok(Vec::new()),
        "both tasks should first suspend for the same next tick"
    );
    assert_eq!(state.scheduled_task_count(), 2);

    state.next_list_collection = 1;
    assert_eq!(
        advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
        Ok(vec![Value::Null]),
        "GC in the first due task must retain the later task's hidden frame roots"
    );
    assert_eq!(
        state.heap().datum_field(holder, &field("observed")),
        Ok(&Value::number(7.0))
    );
}

#[test]
fn list_gc_large_scalar_default_catalog_builds_bounded_root_snapshot() {
    let mut state = ExecutionState::new();
    let direct = state.heap_mut().allocate_list();
    let nested = state.heap_mut().allocate_list();
    let garbage = state.heap_mut().allocate_list();

    let mut defaults = BTreeMap::new();
    for index in 0..100_000 {
        defaults.insert(
            FieldName::parse(&format!("scalar_{index}")).unwrap(),
            Value::number(index as f32),
        );
    }
    defaults.insert(field("direct_list"), Value::List(direct));
    defaults.insert(
        field("modified_path"),
        Value::ModifiedTypePath(Arc::new(ModifiedTypePath::new(
            TypePath::parse("/datum/outer").unwrap(),
            vec![(
                field("nested"),
                Value::ModifiedTypePath(Arc::new(ModifiedTypePath::new(
                    TypePath::parse("/datum/inner").unwrap(),
                    vec![(field("held"), Value::List(nested))],
                ))),
            )],
        ))),
    );
    state.set_initial_values(BTreeMap::from([(
        TypePath::parse("/datum/catalog_entry").unwrap(),
        defaults,
    )]));

    assert!(state.initial_value_datum_roots.is_empty());
    assert_eq!(&*state.initial_value_list_roots, [direct, nested]);

    state.next_list_collection = 1;
    state.maybe_collect_unreachable_lists(&[]);
    assert!(state.heap().list(direct).is_ok());
    assert!(state.heap().list(nested).is_ok());
    assert!(state.heap().list(garbage).is_err());
}

#[test]
fn heap_gc_reclaims_unrooted_datums_and_preserves_runtime_roots() {
    let mut state = ExecutionState::new();
    let rooted = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/rooted").unwrap());
    let child = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/child").unwrap());
    let rooted_list = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .set_datum_field(rooted, field("items"), Value::List(rooted_list))
        .unwrap();
    state
        .heap_mut()
        .list_mut(rooted_list)
        .unwrap()
        .add(Value::Datum(child));
    state.set_global(field("rooted"), Value::Datum(rooted));

    let garbage = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/qdeleted").unwrap());
    let garbage_list = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .set_datum_field(garbage, field("cycle"), Value::List(garbage_list))
        .unwrap();
    state
        .heap_mut()
        .list_mut(garbage_list)
        .unwrap()
        .add(Value::Datum(garbage));

    state.next_list_collection = 1;
    state.maybe_collect_unreachable_lists(&[]);
    assert!(state.heap().datum(rooted).is_ok());
    assert!(state.heap().datum(child).is_ok());
    assert!(state.heap().list(rooted_list).is_ok());
    assert!(state.heap().datum(garbage).is_err());
    assert!(state.heap().list(garbage_list).is_err());
}

#[test]
fn heap_gc_roots_engine_post_return_signal_graphs() {
    let syntax = parse("/proc/noop()\n\treturn\n").expect("frame fixture should parse");
    let module = compile_module(&syntax.definitions).expect("frame fixture should compile");
    let procedure = module.procedure_id("/proc/noop").unwrap();
    let program = module.procedure(procedure).unwrap();
    let mut state = ExecutionState::new();

    let listener = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/listener").unwrap());
    let source = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/source").unwrap());
    let signal_procs = state.heap_mut().allocate_list();
    let source_procs = state.heap_mut().allocate_list();
    let callback_state = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(callback_state)
        .unwrap()
        .add(Value::number(7.0));
    state
        .heap_mut()
        .list_mut(source_procs)
        .unwrap()
        .set_key(Value::text("signal"), Value::List(callback_state));
    state
        .heap_mut()
        .list_mut(signal_procs)
        .unwrap()
        .set_key(Value::Datum(source), Value::List(source_procs));
    state
        .heap_mut()
        .set_datum_field(listener, field("_signal_procs"), Value::List(signal_procs))
        .unwrap();

    let garbage = state.heap_mut().allocate_list();
    let parent_context = ExecutionContext::new(Value::Datum(listener), Value::Null);
    let parent = make_frame(procedure, program, &[], &parent_context);
    let child_context = ExecutionContext::new(Value::Null, Value::Null);
    let mut child = make_frame(procedure, program, &[], &child_context);
    child.set_engine_post_return(Some(Box::new(parent)));

    state.next_list_collection = 1;
    state.maybe_collect_unreachable_lists(&[child]);

    assert!(state.heap().datum(listener).is_ok());
    assert!(state.heap().datum(source).is_ok());
    assert!(state.heap().list(signal_procs).is_ok());
    assert!(state.heap().list(source_procs).is_ok());
    assert!(state.heap().list(callback_state).is_ok());
    assert!(state.heap().list(garbage).is_err());
}

#[test]
fn sized_list_construction_supports_gc_and_master_initialization_shapes() {
    let syntax = parse(
            "/proc/build_gc_queues(count)\n\tvar/list/queues = new /list(count)\n\tfor(var/i in 1 to count)\n\t\tqueues[i] = list()\n\treturn queues\n/proc/build_stages(count)\n\tvar/list/stages = new(count)\n\tfor(var/i in 1 to count)\n\t\tstages[i] = list(i)\n\treturn stages\n",
        )
        .expect("sized list construction should parse");
    let module = compile_module(&syntax.definitions).expect("sized lists should compile");

    let mut state = ExecutionState::new();
    for (path, count) in [("/proc/build_gc_queues", 5.0), ("/proc/build_stages", 2.0)] {
        let Value::List(list) = execute_module_in_state(
            &module,
            module.procedure_id(path).unwrap(),
            &[Value::number(count)],
            &mut state,
        )
        .expect("sized list writes should stay in bounds") else {
            panic!("sized construction should return a list");
        };
        assert_eq!(state.heap().list(list).unwrap().len(), count as usize);
        assert!(
            state
                .heap()
                .list(list)
                .unwrap()
                .positions()
                .all(|(_, value)| matches!(value, Value::List(_)))
        );
    }
}

#[test]
fn multidimensional_new_list_builds_independent_null_filled_rows() {
    let syntax = parse("/proc/build()\n\treturn new /list(2, 3)\n")
        .expect("multidimensional list should parse");
    let program = compile_procedure(&syntax.definitions[0]).unwrap();
    let mut state = ExecutionState::new();
    let Value::List(outer) = execute_in_state(&program, &[], &mut state).unwrap() else {
        panic!("multidimensional construction should return a list");
    };
    let rows = state
        .heap()
        .list(outer)
        .unwrap()
        .positions()
        .map(|(_, value)| match value {
            Value::List(row) => *row,
            _ => panic!("outer positions should be row lists"),
        })
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 2);
    assert_ne!(rows[0], rows[1]);
    for row in rows {
        let row = state.heap().list(row).unwrap();
        assert_eq!(row.len(), 3);
        assert!(
            row.positions()
                .all(|(_, value)| matches!(value, Value::Null))
        );
    }
}

#[test]
fn sized_list_rejects_fractional_negative_and_text_dimensions() {
    let syntax = parse("/proc/build(size)\n\treturn new /list(size)\n").unwrap();
    let program = compile_procedure(&syntax.definitions[0]).unwrap();
    for invalid in [Value::number(-1.0), Value::number(1.5), Value::text("3")] {
        let error = execute(&program, &[invalid]).expect_err("invalid dimension must fail");
        assert!(error.message.contains("list dimension"));
    }
}

#[test]
fn trailing_slash_type_path_is_canonicalized_in_list_keys() {
    let syntax = parse("/proc/build()\n\treturn list(/datum/example/ = 7)\n")
        .expect("trailing-slash type path should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("type-path list should compile");
    let mut state = ExecutionState::new();
    let Value::List(list) =
        execute_in_state(&program, &[], &mut state).expect("initializer should execute")
    else {
        panic!("initializer should return a list");
    };
    let key = Value::TypePath(TypePath::parse("/datum/example").unwrap());
    assert_eq!(
        state.heap().list(list).unwrap().get_key(&key),
        Ok(&Value::number(7.0))
    );
}

#[test]
fn runtime_created_atoms_register_with_world_and_receive_contents() {
    let mut state = ExecutionState::new();
    let world = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/world").unwrap());
    let world_contents = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .set_datum_field(world, field("contents"), Value::List(world_contents))
        .unwrap();
    state.set_global(field("world"), Value::Datum(world));

    let atom =
        allocate_initialized_datum(&mut state, TypePath::parse("/obj/item/runtime").unwrap())
            .expect("runtime atom should allocate");
    assert!(state.heap().datum_field(atom, &field("contents")).is_err());
    let datum_contents = Value::List(state.ensure_contents(atom).unwrap());

    assert!(matches!(datum_contents, Value::List(_)));
    assert!(
        state
            .heap()
            .list(world_contents)
            .unwrap()
            .contains(&Value::Datum(atom))
    );
}

#[test]
fn runtime_datums_share_unchanged_scalar_defaults_without_becoming_gc_roots() {
    let mut state = ExecutionState::new();
    let path = TypePath::parse("/datum/sparse_runtime").unwrap();
    let defaults = (0..96)
        .map(|index| {
            (
                field(&format!("default_{index}")),
                Value::number(index as f32),
            )
        })
        .collect::<BTreeMap<_, _>>();
    state.set_initial_values(BTreeMap::from([(path.clone(), defaults)]));

    let mut datums = Vec::new();
    for _ in 0..4_096 {
        datums.push(
            allocate_initialized_datum(&mut state, path.clone())
                .expect("sparse runtime datum should allocate"),
        );
    }
    assert!(datums.iter().all(|datum| {
        state
            .heap()
            .datum(*datum)
            .is_ok_and(|datum| datum.fields_are_empty())
    }));
    assert_eq!(
        datum_field_or_initial(&state, datums[0], &field("default_95")),
        Ok(Value::number(95.0)),
    );

    state
        .heap_mut()
        .set_datum_field(datums[0], field("default_5"), Value::number(700.0))
        .unwrap();
    assert_eq!(
        datum_field_or_initial(&state, datums[0], &field("default_5")),
        Ok(Value::number(700.0)),
    );
    assert_eq!(state.heap().datum(datums[0]).unwrap().field_len(), 1);
    assert!(!state.compact_default_datums.contains(&datums[0]));

    let (reclaimed_datums, _) = state
        .heap_mut()
        .collect_unreachable_values_from_ids(&[datums[0]], &[]);
    assert_eq!(reclaimed_datums, datums.len() - 1);
    assert!(state.heap().datum(datums[0]).is_ok());
}

#[test]
fn sparse_effective_initial_cache_covers_inheritance_engine_roots_and_misses() {
    let mut state = ExecutionState::new();
    let parent = TypePath::parse("/datum/cache_parent").unwrap();
    let child = TypePath::parse("/datum/cache_parent/child").unwrap();
    let synthetic = TypePath::parse("/obj/cache_synthetic").unwrap();
    let inherited = field("inherited");
    let alpha = field("alpha");
    let density = field("density");
    let known_null = field("name");
    let absent = field("definitely_absent");
    state.set_type_parents(BTreeMap::from([
        (parent.clone(), None),
        (child.clone(), Some(parent.clone())),
    ]));
    state.set_initial_values(BTreeMap::from([
        (
            parent,
            BTreeMap::from([(inherited.clone(), Value::number(7.0))]),
        ),
        (
            TypePath::parse("/atom").unwrap(),
            BTreeMap::from([(alpha.clone(), Value::number(201.0))]),
        ),
    ]));

    let inherited_datum = state.heap_mut().allocate_datum(child.clone());
    let synthetic_datum = state.heap_mut().allocate_datum(synthetic.clone());
    assert_eq!(
        state.initial_value(&child, &inherited),
        Some(&Value::number(7.0)),
        "the public borrowed catalog lookup must retain inherited behavior",
    );
    for _ in 0..2 {
        assert_eq!(
            datum_field_or_initial(&state, inherited_datum, &inherited),
            Ok(Value::number(7.0)),
        );
        assert_eq!(
            datum_field_or_initial(&state, synthetic_datum, &alpha),
            Ok(Value::number(201.0)),
        );
        assert_eq!(
            datum_field_or_initial(&state, synthetic_datum, &density),
            Ok(Value::number(0.0)),
        );
        assert_eq!(
            datum_field_or_initial(&state, synthetic_datum, &known_null),
            Ok(Value::Null),
        );
        assert!(matches!(
            datum_field_or_initial(&state, synthetic_datum, &absent),
            Err(ValueError::MissingField(_)),
        ));
    }

    let cache = state.effective_initial_value_cache.borrow();
    assert_eq!(
        cache.get(&child).and_then(|fields| fields.get(&inherited)),
        Some(&Some(Value::number(7.0))),
    );
    let synthetic_fields = cache.get(&synthetic).expect("synthetic reads are cached");
    assert_eq!(
        synthetic_fields.get(&alpha),
        Some(&Some(Value::number(201.0))),
    );
    assert_eq!(
        synthetic_fields.get(&density),
        Some(&Some(Value::number(0.0))),
    );
    assert_eq!(synthetic_fields.get(&known_null), Some(&Some(Value::Null)));
    assert_eq!(synthetic_fields.get(&absent), Some(&None));
}

#[test]
fn sparse_effective_initial_cache_invalidates_with_catalog_metadata() {
    let mut state = ExecutionState::new();
    let parent_a = TypePath::parse("/datum/cache_a").unwrap();
    let parent_b = TypePath::parse("/datum/cache_b").unwrap();
    let child = TypePath::parse("/datum/cache_child").unwrap();
    let cached = field("cached");
    let values = BTreeMap::from([
        (
            parent_a.clone(),
            BTreeMap::from([(cached.clone(), Value::number(1.0))]),
        ),
        (
            parent_b.clone(),
            BTreeMap::from([(cached.clone(), Value::number(2.0))]),
        ),
    ]);
    state.set_initial_values(values.clone());
    state.set_type_parents(BTreeMap::from([
        (parent_a.clone(), None),
        (parent_b.clone(), None),
        (child.clone(), Some(parent_a.clone())),
    ]));

    let prime = |state: &ExecutionState, expected: f32| {
        assert_eq!(
            initial_value_or_engine_root(state, &child, &cached),
            Some(Value::number(expected)),
        );
        assert!(!state.effective_initial_value_cache.borrow().is_empty());
    };
    prime(&state, 1.0);

    state.set_type_paths([child.clone()]);
    assert!(state.effective_initial_value_cache.borrow().is_empty());
    prime(&state, 1.0);

    state.set_shared_type_paths(Arc::new(BTreeSet::from([child.clone()])));
    assert!(state.effective_initial_value_cache.borrow().is_empty());
    prime(&state, 1.0);

    state.set_type_parents(BTreeMap::from([
        (parent_a.clone(), None),
        (parent_b.clone(), None),
        (child.clone(), Some(parent_b.clone())),
    ]));
    assert!(state.effective_initial_value_cache.borrow().is_empty());
    prime(&state, 2.0);

    state.set_shared_type_parents(Arc::new(BTreeMap::from([
        (parent_a.clone(), None),
        (parent_b.clone(), None),
        (child.clone(), Some(parent_a.clone())),
    ])));
    assert!(state.effective_initial_value_cache.borrow().is_empty());
    prime(&state, 1.0);

    let mut replaced_values = values;
    replaced_values
        .get_mut(&parent_a)
        .unwrap()
        .insert(cached.clone(), Value::number(3.0));
    state.set_initial_values(replaced_values.clone());
    assert!(state.effective_initial_value_cache.borrow().is_empty());
    prime(&state, 3.0);

    replaced_values
        .get_mut(&parent_a)
        .unwrap()
        .insert(cached.clone(), Value::number(4.0));
    state.set_shared_initial_values(Arc::new(replaced_values));
    assert!(state.effective_initial_value_cache.borrow().is_empty());
    prime(&state, 4.0);
}

#[test]
fn sparse_effective_initial_cache_is_bounded_without_eviction_churn() {
    let mut state = ExecutionState::new();
    let hot_path = TypePath::parse("/datum/cache_hot").unwrap();
    let cold_path = TypePath::parse("/datum/cache_cold").unwrap();
    let fields = (0..=MAX_EFFECTIVE_INITIAL_VALUE_CACHE_FIELDS_PER_TYPE)
        .map(|index| {
            (
                field(&format!("cached_{index}")),
                Value::number(index as f32),
            )
        })
        .collect::<BTreeMap<_, _>>();
    state.set_initial_values(BTreeMap::from([
        (hot_path.clone(), fields.clone()),
        (cold_path.clone(), fields),
    ]));

    for index in 0..MAX_EFFECTIVE_INITIAL_VALUE_CACHE_FIELDS_PER_TYPE {
        assert_eq!(
            initial_value_or_engine_root(&state, &hot_path, &field(&format!("cached_{index}")),),
            Some(Value::number(index as f32)),
        );
    }
    assert_eq!(
        state.effective_initial_value_cache_entries.get(),
        MAX_EFFECTIVE_INITIAL_VALUE_CACHE_FIELDS_PER_TYPE,
    );
    let overflow = field(&format!(
        "cached_{}",
        MAX_EFFECTIVE_INITIAL_VALUE_CACHE_FIELDS_PER_TYPE
    ));
    assert_eq!(
        initial_value_or_engine_root(&state, &hot_path, &overflow),
        Some(Value::number(
            MAX_EFFECTIVE_INITIAL_VALUE_CACHE_FIELDS_PER_TYPE as f32
        )),
    );
    assert_eq!(
        state
            .effective_initial_value_cache
            .borrow()
            .get(&hot_path)
            .map(HashMap::len),
        Some(MAX_EFFECTIVE_INITIAL_VALUE_CACHE_FIELDS_PER_TYPE),
    );

    // Simulate a globally saturated cache without constructing half a
    // million fixture entries. New cold answers remain correct but are not
    // admitted, while an existing hot entry remains available and stable.
    state
        .effective_initial_value_cache_entries
        .set(MAX_EFFECTIVE_INITIAL_VALUE_CACHE_ENTRIES);
    let cold_field = field("cached_0");
    assert_eq!(
        initial_value_or_engine_root(&state, &cold_path, &cold_field),
        Some(Value::number(0.0)),
    );
    assert!(
        !state
            .effective_initial_value_cache
            .borrow()
            .contains_key(&cold_path)
    );
    assert_eq!(
        initial_value_or_engine_root(&state, &hot_path, &cold_field),
        Some(Value::number(0.0)),
    );
    assert_eq!(
        state.effective_initial_value_cache_entries.get(),
        MAX_EFFECTIVE_INITIAL_VALUE_CACHE_ENTRIES,
    );
}

#[test]
fn movable_constructor_receives_mob_location_and_contents_before_new() {
    let syntax = parse(concat!(
        "/obj/item/constructor_probe/New(where)\n",
        "\tsrc.saw_location = (src.loc == where)\n",
        "\tsrc.saw_contents = (src in where.contents)\n",
        "/proc/build_constructor_probe()\n",
        "\tvar/mob/preview_holder/holder = new /mob/preview_holder\n",
        "\tvar/obj/item/constructor_probe/item = new /obj/item/constructor_probe(holder)\n",
        "\treturn list(holder, item)\n",
    ))
    .expect("item-in-mob constructor fixture should parse");
    let module = compile_module(&syntax.definitions)
        .expect("item-in-mob constructor fixture should compile");
    let entry = module
        .procedure_id("/proc/build_constructor_probe")
        .expect("constructor fixture entry should exist");
    let mut state = ExecutionState::new();

    let Value::List(result) = execute_module_in_state(&module, entry, &[], &mut state)
        .expect("item-in-mob construction should execute")
    else {
        panic!("constructor fixture should return a list");
    };
    let values = state
        .heap()
        .list(result)
        .unwrap()
        .positions()
        .map(|(_, value)| value.clone())
        .collect::<Vec<_>>();
    let (Value::Datum(holder), Value::Datum(item)) = (&values[0], &values[1]) else {
        panic!("constructor fixture should return the holder and item");
    };
    let item = *item;
    let holder = *holder;

    assert_eq!(
        state.heap().datum_field(item, &field("loc")),
        Ok(&Value::Datum(holder)),
        "a mob is a valid movable constructor location"
    );
    assert_eq!(
        state.heap().datum_field(item, &field("saw_location")),
        Ok(&Value::number(1.0)),
        "loc must be installed before New runs"
    );
    assert_eq!(
        state.heap().datum_field(item, &field("saw_contents")),
        Ok(&Value::number(1.0)),
        "the containing mob's contents must be synchronized before New runs"
    );
    let Value::List(contents) = state
        .heap()
        .datum_field(holder, &field("contents"))
        .unwrap()
    else {
        panic!("mob should have a contents list");
    };
    assert!(
        state
            .heap()
            .list(*contents)
            .unwrap()
            .contains(&Value::Datum(item)),
        "the constructed item remains in its mob's contents after New"
    );
}

#[test]
fn direct_loc_assignment_synchronizes_container_contents() {
    let syntax =
        parse("/proc/move(atom, target)\n\tatom.loc = target\n\treturn atom.loc\n").unwrap();
    let program = compile_procedure(&syntax.definitions[0]).unwrap();
    let mut state = ExecutionState::new();
    let old = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/turf/old").unwrap());
    let new = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/turf/new").unwrap());
    let atom = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/item").unwrap());
    let old_contents = state.heap_mut().allocate_list();
    let new_contents = state.heap_mut().allocate_list();
    for (container, list) in [(old, old_contents), (new, new_contents)] {
        state
            .heap_mut()
            .set_datum_field(container, field("contents"), Value::List(list))
            .unwrap();
    }
    state
        .heap_mut()
        .list_mut(old_contents)
        .unwrap()
        .add(Value::Datum(atom));
    state
        .heap_mut()
        .set_datum_field(atom, field("loc"), Value::Datum(old))
        .unwrap();

    assert_eq!(
        execute_in_state(
            &program,
            &[Value::Datum(atom), Value::Datum(new)],
            &mut state
        ),
        Ok(Value::Datum(new))
    );
    assert!(
        !state
            .heap()
            .list(old_contents)
            .unwrap()
            .contains(&Value::Datum(atom))
    );
    assert!(
        state
            .heap()
            .list(new_contents)
            .unwrap()
            .contains(&Value::Datum(atom))
    );
}

#[test]
fn image_loc_is_visual_context_and_does_not_mutate_turf_contents() {
    let syntax = parse(
        "/proc/place(image/visual, turf/target)\n\tvisual.loc = target\n\treturn visual.loc\n",
    )
    .unwrap();
    let program = compile_procedure(&syntax.definitions[0]).unwrap();
    let mut state = ExecutionState::new();
    let turf = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/turf/floor").unwrap());
    let contents = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .set_datum_field(turf, field("contents"), Value::List(contents))
        .unwrap();
    let image = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/image").unwrap());
    state
        .heap_mut()
        .set_datum_field(image, field("loc"), Value::Null)
        .unwrap();

    assert_eq!(
        execute_in_state(
            &program,
            &[Value::Datum(image), Value::Datum(turf)],
            &mut state,
        ),
        Ok(Value::Datum(turf)),
    );
    assert_eq!(
        state.heap().datum_field(image, &field("loc")),
        Ok(&Value::Datum(turf)),
    );
    assert!(
        !state
            .heap()
            .list(contents)
            .unwrap()
            .contains(&Value::Datum(image)),
        "an image loc is not physical turf containment",
    );
}

#[test]
fn runtime_new_type_and_proc_ref_macro_expansion_compile() {
    let syntax = parse(
            "/proc/build(starting_organ)\n\tvar/item = new starting_organ(src)\n\treturn list((nameof(.proc/on_entered)), item)\n",
        )
        .expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("runtime new type and expanded PROC_REF should compile");
    assert!(program.instructions.iter().any(|instruction| matches!(
        instruction,
        Instruction::AllocateDatum {
            argument_count: 1,
            ..
        }
    )));
}

fn manual_program(instructions: Vec<Instruction>, parameter_count: usize) -> Program {
    let instruction_count = instructions.len();
    Program {
        wait_for: true,
        parameter_count,
        parameter_names: vec![String::new(); parameter_count],
        verb_parameter_types: vec![crate::VerbParameterType::Unsupported; parameter_count],
        verb_name: None,
        local_count: parameter_count,
        instructions,
        source_spans: (0..instruction_count)
            .map(|index| SourceSpan::new(index * 10, index * 10 + 1))
            .collect(),
    }
}

fn field(name: &str) -> FieldName {
    FieldName::parse(name).unwrap()
}

#[test]
fn false_tick_check_peephole_preserves_branch_and_step_budget() {
    let program = manual_program(
        vec![
            Instruction::LoadGlobal(field("world")),
            Instruction::LoadField(field("tick_usage")),
            Instruction::LoadGlobal(field("current_ticklimit")),
            Instruction::Greater,
            Instruction::JumpIfFalse(7),
            Instruction::PushNumber(DmNumberBits::from_f32(1.0)),
            Instruction::Return,
            Instruction::PushNumber(DmNumberBits::from_f32(2.0)),
            Instruction::Return,
        ],
        0,
    );
    let make_state = |usage| {
        let mut state = ExecutionState::new();
        let world = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/world").unwrap());
        state
            .heap_mut()
            .set_datum_field(world, field("tick_usage"), Value::number(usage))
            .unwrap();
        state.set_global(field("world"), Value::Datum(world));
        state.set_global(field("current_ticklimit"), Value::number(50.0));
        state
    };

    let limits = ExecutionLimits {
        max_call_depth: 16,
        max_steps: 7,
        wall_clock_budget: None,
    };
    let mut false_state = make_state(40.0);
    assert_eq!(
        execute_with_limits_in_state(&program, &[], limits, &mut false_state),
        Ok(Value::number(2.0)),
    );
    let mut true_state = make_state(60.0);
    assert_eq!(
        execute_with_limits_in_state(&program, &[], limits, &mut true_state),
        Ok(Value::number(1.0)),
    );
    let mut exhausted_state = make_state(40.0);
    assert!(
        execute_with_limits_in_state(
            &program,
            &[],
            ExecutionLimits {
                max_call_depth: 16,
                max_steps: 6,
                wall_clock_budget: None,
            },
            &mut exhausted_state,
        )
        .is_err(),
    );
}

#[test]
fn numeric_loop_branch_peephole_preserves_bounds_and_step_budget() {
    let bounded = manual_program(
        vec![
            Instruction::LoadLocal(0),
            Instruction::LoadLocal(1),
            Instruction::LessEqual,
            Instruction::JumpIfFalse(6),
            Instruction::PushNumber(DmNumberBits::from_f32(1.0)),
            Instruction::Return,
            Instruction::PushNumber(DmNumberBits::from_f32(2.0)),
            Instruction::Return,
        ],
        2,
    );
    let limits = ExecutionLimits {
        max_call_depth: 16,
        max_steps: 6,
        wall_clock_budget: None,
    };
    assert_eq!(
        execute_with_limits(&bounded, &[Value::number(3.0), Value::number(4.0)], limits),
        Ok(Value::number(1.0)),
    );
    assert_eq!(
        execute_with_limits(&bounded, &[Value::number(5.0), Value::number(4.0)], limits),
        Ok(Value::number(2.0)),
    );
    assert!(
        execute_with_limits(
            &bounded,
            &[Value::number(3.0), Value::number(4.0)],
            ExecutionLimits {
                max_call_depth: 16,
                max_steps: 5,
                wall_clock_budget: None,
            },
        )
        .is_err(),
    );

    let list_bounded = manual_program(
        vec![
            Instruction::LoadLocal(0),
            Instruction::ListLengthLocal(1),
            Instruction::LessEqual,
            Instruction::JumpIfFalse(6),
            Instruction::PushNumber(DmNumberBits::from_f32(1.0)),
            Instruction::Return,
            Instruction::PushNumber(DmNumberBits::from_f32(2.0)),
            Instruction::Return,
        ],
        2,
    );
    let mut state = ExecutionState::new();
    let list = state.heap_mut().allocate_list();
    let values = state.heap_mut().list_mut(list).unwrap();
    values.add(Value::number(10.0));
    values.add(Value::number(20.0));
    assert_eq!(
        execute_with_limits_in_state(
            &list_bounded,
            &[Value::number(2.0), Value::List(list)],
            limits,
            &mut state,
        ),
        Ok(Value::number(1.0)),
    );

    let update = manual_program(
        vec![
            Instruction::LoadLocal(0),
            Instruction::PushNumber(DmNumberBits::from_f32(1.0)),
            Instruction::Add,
            Instruction::StoreLocal(0),
            Instruction::LoadLocal(0),
            Instruction::Return,
        ],
        1,
    );
    assert_eq!(
        execute_with_limits(&update, &[Value::number(41.0)], limits),
        Ok(Value::number(42.0)),
    );
    assert_eq!(
        execute_with_limits(&update, &[Value::Null], limits),
        Ok(Value::number(1.0)),
    );
}

#[test]
fn numeric_dispatch_block_preserves_complex_loop_and_step_budget() {
    let program = manual_program(
        vec![
            Instruction::LoadLocal(0),
            Instruction::PushNumber(DmNumberBits::from_f32(1.0)),
            Instruction::Add,
            Instruction::StoreLocal(0),
            Instruction::LoadLocal(0),
            Instruction::LoadLocal(1),
            Instruction::LessEqual,
            Instruction::LoadLocal(0),
            Instruction::PushNumber(DmNumberBits::from_f32(0.0)),
            Instruction::Greater,
            Instruction::And,
            Instruction::JumpIfFalse(13),
            Instruction::Jump(0),
            Instruction::LoadLocal(0),
            Instruction::Return,
        ],
        2,
    );
    let arguments = [Value::number(0.0), Value::number(3.0)];
    assert_eq!(
        execute_with_limits(
            &program,
            &arguments,
            ExecutionLimits {
                max_call_depth: 16,
                max_steps: 53,
                wall_clock_budget: None,
            },
        ),
        Ok(Value::number(4.0)),
    );
    assert!(
        execute_with_limits(
            &program,
            &arguments,
            ExecutionLimits {
                max_call_depth: 16,
                max_steps: 52,
                wall_clock_budget: None,
            },
        )
        .is_err(),
    );

    let text_fallback = manual_program(
        vec![
            Instruction::PushText(Arc::from("a")),
            Instruction::PushText(Arc::from("b")),
            Instruction::Add,
            Instruction::Return,
        ],
        0,
    );
    assert_eq!(
        execute_with_limits(
            &text_fallback,
            &[],
            ExecutionLimits {
                max_call_depth: 16,
                max_steps: 4,
                wall_clock_budget: None,
            },
        ),
        Ok(Value::text("ab")),
    );

    let mut parameter_alias = manual_program(
        vec![
            Instruction::PushNumber(DmNumberBits::from_f32(5.0)),
            Instruction::StoreLocal(0),
            Instruction::MakeArgs,
            Instruction::PushNumber(DmNumberBits::from_f32(1.0)),
            Instruction::IndexList,
            Instruction::Return,
        ],
        1,
    );
    parameter_alias.parameter_names[0] = "value".to_owned();
    assert_eq!(
        execute_with_limits(
            &parameter_alias,
            &[Value::number(2.0)],
            ExecutionLimits {
                max_call_depth: 16,
                max_steps: 6,
                wall_clock_budget: None,
            },
        ),
        Ok(Value::number(5.0)),
    );
}

#[test]
fn packed_numeric_dispatch_matches_rich_state_and_side_exits() {
    let mut numeric = manual_program(
        vec![
            Instruction::LoadLocal(0),
            Instruction::PushNumber(DmNumberBits::from_f32(2.0)),
            Instruction::Multiply,
            Instruction::StoreLocal(1),
            Instruction::LoadLocal(1),
            Instruction::PushNumber(DmNumberBits::from_f32(8.0)),
            Instruction::GreaterEqual,
            Instruction::StoreResult,
            Instruction::LoadResult,
            Instruction::Not,
            Instruction::JumpIfFalse(13),
            Instruction::PushNull,
            Instruction::Pop,
            Instruction::Return,
        ],
        1,
    );
    numeric.local_count = 2;
    let state = ExecutionState::new();
    let arguments = [Value::number(4.0)];
    let context = ExecutionContext::default();
    let mut rich = make_frame(ProcedureId(0), &numeric, &arguments, &context);
    let mut packed = rich.clone();
    let rich_steps = try_run_rich_numeric_dispatch_block(&numeric, &mut rich, 64, &state);
    let packed_steps = try_run_packed_numeric_dispatch_block(&numeric, &mut packed, 64, &state);
    assert_eq!(packed_steps, rich_steps);
    assert_eq!(packed.instruction, rich.instruction);
    assert_eq!(packed.locals, rich.locals);
    assert_eq!(packed.stack, rich.stack);
    assert_eq!(packed.result, rich.result);

    let side_exit = manual_program(
        vec![
            Instruction::PushNumber(DmNumberBits::from_f32(1.0)),
            Instruction::Pop,
            Instruction::PushText(Arc::from("rich")),
            Instruction::Return,
        ],
        0,
    );
    let mut rich = make_frame(ProcedureId(0), &side_exit, &[], &context);
    let mut packed = rich.clone();
    assert_eq!(
        try_run_packed_numeric_dispatch_block(&side_exit, &mut packed, 64, &state),
        Some(2)
    );
    assert_eq!(
        try_run_rich_numeric_dispatch_block(&side_exit, &mut rich, 2, &state),
        Some(2)
    );
    assert_eq!(packed.instruction, rich.instruction);
    assert_eq!(packed.stack, rich.stack);
}

#[test]
fn packed_numeric_state_persists_across_budgets_and_materializes_on_side_exit() {
    let mut program = manual_program(
        vec![
            Instruction::LoadLocal(0),
            Instruction::PushNumber(DmNumberBits::from_f32(1.0)),
            Instruction::Add,
            Instruction::StoreLocal(0),
            Instruction::Jump(0),
            Instruction::Return,
        ],
        0,
    );
    program.local_count = 1;
    let state = ExecutionState::new();
    let mut frame = make_frame(ProcedureId(0), &program, &[], &ExecutionContext::default());
    assert_eq!(
        try_run_packed_numeric_dispatch_block(&program, &mut frame, 5, &state),
        Some(5)
    );
    assert!(
        frame
            .cold()
            .is_some_and(|cold| cold.packed_numeric_state.is_some())
    );
    assert_eq!(frame.locals[0], Value::Null);
    assert_eq!(
        try_run_packed_numeric_dispatch_block(&program, &mut frame, 5, &state),
        Some(5)
    );
    frame.instruction = 5;
    assert_eq!(
        try_run_packed_numeric_dispatch_block(&program, &mut frame, 5, &state),
        None
    );
    assert_eq!(frame.locals[0], Value::number(2.0));
    assert!(
        frame
            .cold()
            .is_none_or(|cold| cold.packed_numeric_state.is_none())
    );
}

#[test]
fn adaptive_packed_entry_declines_short_procedures_and_enters_sustained_loops() {
    let short = manual_program(
        vec![
            Instruction::PushNumber(DmNumberBits::from_f32(1.0)),
            Instruction::StoreResult,
            Instruction::Return,
        ],
        0,
    );
    let state = ExecutionState::new();
    let mut short_frame = make_frame(ProcedureId(0), &short, &[], &ExecutionContext::default());
    let (_, declines_before) = packed_dispatch_counters();
    assert_eq!(
        try_run_numeric_dispatch_block(&short, &mut short_frame, 100, &state),
        Some(2)
    );
    let after_short = packed_dispatch_counters();
    assert!(after_short.1 > declines_before);
    assert!(
        short_frame
            .cold()
            .is_none_or(|cold| cold.packed_numeric_state.is_none())
    );

    let sustained = manual_program(
        vec![
            Instruction::PushNumber(DmNumberBits::from_f32(1.0)),
            Instruction::Pop,
            Instruction::Jump(0),
        ],
        0,
    );
    let mut sustained_frame = make_frame(
        ProcedureId(0),
        &sustained,
        &[],
        &ExecutionContext::default(),
    );
    assert_eq!(
        try_run_numeric_dispatch_block(&sustained, &mut sustained_frame, 100, &state),
        Some(100)
    );
    let after_sustained = packed_dispatch_counters();
    assert!(after_sustained.0 > after_short.0);
    assert!(
        sustained_frame
            .cold()
            .is_some_and(|cold| cold.packed_numeric_state.is_some())
    );
}

#[test]
#[ignore = "release-only packed numeric dispatch microbenchmark"]
fn packed_numeric_dispatch_benchmark() {
    const ROUNDS: usize = 1_000_000;
    let program = manual_program(
        vec![
            Instruction::LoadLocal(0),
            Instruction::PushNumber(DmNumberBits::from_f32(2.0)),
            Instruction::Multiply,
            Instruction::PushNumber(DmNumberBits::from_f32(1.0)),
            Instruction::Add,
            Instruction::StoreResult,
            Instruction::Return,
        ],
        1,
    );
    let state = ExecutionState::new();
    let context = ExecutionContext::default();
    let arguments = [Value::number(20.5)];
    let mut rich = make_frame(ProcedureId(0), &program, &arguments, &context);
    let started = Instant::now();
    for _ in 0..ROUNDS {
        rich.instruction = 0;
        rich.stack.clear();
        std::hint::black_box(try_run_rich_numeric_dispatch_block(
            &program, &mut rich, 32, &state,
        ));
    }
    let rich_elapsed = started.elapsed();
    let mut packed = make_frame(ProcedureId(0), &program, &arguments, &context);
    let started = Instant::now();
    for _ in 0..ROUNDS {
        packed.instruction = 0;
        packed.stack.clear();
        std::hint::black_box(try_run_packed_numeric_dispatch_block(
            &program,
            &mut packed,
            32,
            &state,
        ));
    }
    let packed_elapsed = started.elapsed();
    eprintln!(
        "packed-numeric rounds={ROUNDS} rich_ms={} packed_ms={} speedup={:.3}",
        rich_elapsed.as_millis(),
        packed_elapsed.as_millis(),
        rich_elapsed.as_secs_f64() / packed_elapsed.as_secs_f64(),
    );
    assert_eq!(rich.result, packed.result);

    const BLOCKS: usize = 100_000;
    const STEPS_PER_BLOCK: u64 = 100;
    let mut sustained = manual_program(
        vec![
            Instruction::LoadLocal(0),
            Instruction::PushNumber(DmNumberBits::from_f32(1.0)),
            Instruction::Add,
            Instruction::StoreLocal(0),
            Instruction::Jump(0),
            Instruction::Return,
        ],
        0,
    );
    sustained.local_count = 1;
    let mut rich = make_frame(ProcedureId(0), &sustained, &[], &context);
    let started = Instant::now();
    for _ in 0..BLOCKS {
        std::hint::black_box(try_run_rich_numeric_dispatch_block(
            &sustained,
            &mut rich,
            STEPS_PER_BLOCK,
            &state,
        ));
    }
    let rich_elapsed = started.elapsed();
    let mut packed = make_frame(ProcedureId(0), &sustained, &[], &context);
    let started = Instant::now();
    for _ in 0..BLOCKS {
        std::hint::black_box(try_run_packed_numeric_dispatch_block(
            &sustained,
            &mut packed,
            STEPS_PER_BLOCK,
            &state,
        ));
    }
    let packed_elapsed = started.elapsed();
    eprintln!(
        "packed-persistent blocks={BLOCKS} steps_per_block={STEPS_PER_BLOCK} rich_ms={} packed_ms={} speedup={:.3}",
        rich_elapsed.as_millis(),
        packed_elapsed.as_millis(),
        rich_elapsed.as_secs_f64() / packed_elapsed.as_secs_f64(),
    );
    packed.instruction = 5;
    assert_eq!(
        try_run_packed_numeric_dispatch_block(&sustained, &mut packed, 1, &state),
        None
    );
    assert_eq!(rich.locals, packed.locals);
}

fn expression_tokens(source: &str) -> Vec<SpannedToken> {
    lex(source)
        .expect("expression should lex")
        .into_iter()
        .filter(|token| {
            !matches!(
                token.kind,
                TokenKind::LineStart { .. } | TokenKind::Newline | TokenKind::LineContinuation
            )
        })
        .collect()
}

#[test]
fn initializer_lowering_uses_explicit_bindings_and_source_spans() {
    let tokens = expression_tokens("base + src.increment + global.offset");
    let bindings = BTreeMap::from([("base".to_owned(), InitializerBinding::Global(field("base")))]);
    let initializer =
        compile_initializer(&tokens, &bindings, None).expect("bound initializer should compile");
    let program = initializer
        .module()
        .procedure(initializer.entry())
        .expect("initializer entry should exist");

    assert_eq!(program.instructions.len(), program.source_spans.len());
    assert!(program.source_spans.iter().all(|span| !span.is_empty()));

    let mut state = ExecutionState::new();
    state.set_global(field("base"), Value::number(2.0));
    state.set_global(field("offset"), Value::number(3.0));
    let src = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/example").unwrap());
    state
        .heap_mut()
        .set_datum_field(src, field("increment"), Value::number(4.0))
        .unwrap();
    let context = ExecutionContext::new(Value::Datum(src), Value::Null);

    assert_eq!(
        execute_module_in_context(
            initializer.module(),
            initializer.entry(),
            &[],
            &mut state,
            &context,
        ),
        Ok(Value::number(9.0))
    );
}

#[test]
fn initializer_calls_only_real_linked_global_procedures() {
    let syntax =
        parse("/proc/double(value)\n\treturn value * 2\n").expect("procedure should parse");
    let procedures = compile_module(&syntax.definitions).expect("procedure should compile");
    let tokens = expression_tokens("double(4)");
    let initializer = compile_initializer(&tokens, &BTreeMap::new(), Some(&procedures))
        .expect("linked call should compile");

    assert_eq!(
        execute_module(initializer.module(), initializer.entry(), &[]),
        Ok(Value::number(8.0))
    );
    assert!(
        compile_initializer(
            &expression_tokens("invented_builtin()"),
            &BTreeMap::new(),
            None,
        )
        .is_err(),
        "unregistered names must not become fake built-ins"
    );
}

#[test]
fn appended_initializers_scan_module_call_names_once() {
    let source = (0..64)
        .map(|index| format!("/proc/p{index}()\n\treturn {index}\n"))
        .collect::<String>();
    let syntax = parse(&source).expect("procedures should parse");
    let mut module = compile_module(&syntax.definitions).expect("procedures should compile");

    for _ in 0..32 {
        compile_initializer_into_module(&expression_tokens("p0()"), &BTreeMap::new(), &mut module)
            .expect("initializer should append");
    }

    assert_eq!(module.initializer_call_name_index_builds(), 1);
    assert_eq!(module.initializer_call_name_symbols_scanned(), 64);
}

#[test]
fn compact_wordcode_is_a_cache_and_appending_invalidates_it() {
    let syntax = parse("/proc/base()\n\treturn 7\n").expect("procedure should parse");
    let mut module = compile_module(&syntax.definitions).expect("procedure should compile");
    let semantic_copy = module.clone();
    module
        .install_compact_wordcode()
        .expect("compact wordcode should install");
    let stale_attachment = module.compact_wordcode.0.clone();

    assert_eq!(module, semantic_copy, "execution caches are not semantics");
    let initializer = compile_initializer_into_module(
        &expression_tokens("base()"),
        &BTreeMap::new(),
        &mut module,
    )
    .expect("initializer should append");
    assert!(
        module.compact_wordcode().is_none(),
        "appending must invalidate stale instruction ranges"
    );
    // Model an in-flight dispatcher that retained the immutable attachment
    // before the append. Coverage absence is a cache miss, not corruption.
    module.compact_wordcode.0 = stale_attachment;
    assert_eq!(
        execute_module(&module, initializer, &[]),
        Ok(Value::number(7.0)),
        "an initializer appended after compact attachment executes through rich fallback",
    );
}

#[test]
fn cloned_modules_share_immutable_programs_but_append_independently() {
    let syntax = parse("/proc/base()\n\treturn 7\n").expect("procedure should parse");
    let module = compile_module(&syntax.definitions).expect("procedure should compile");
    let mut clone = module.clone();

    assert!(Arc::ptr_eq(&module.procedures[0], &clone.procedures[0]));
    let entry =
        compile_initializer_into_module(&expression_tokens("base()"), &BTreeMap::new(), &mut clone)
            .expect("initializer should append only to the clone");

    assert_eq!(module.procedure_count(), 1);
    assert_eq!(clone.procedure_count(), 2);
    assert!(Arc::ptr_eq(&module.procedures[0], &clone.procedures[0]));
    assert_eq!(execute_module(&clone, entry, &[]), Ok(Value::number(7.0)));
}

#[test]
fn explicit_src_and_usr_fields_support_compound_assignment() {
    let source = "/proc/update()\n\tsrc.count += usr.increment\n\treturn src.count\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("fields should compile");
    let mut state = ExecutionState::new();
    let src = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/source").unwrap());
    let usr = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/mob/user").unwrap());
    state
        .heap_mut()
        .set_datum_field(src, field("count"), Value::number(3.0))
        .unwrap();
    state
        .heap_mut()
        .set_datum_field(usr, field("increment"), Value::number(2.0))
        .unwrap();
    let context = ExecutionContext::new(Value::Datum(src), Value::Datum(usr));

    assert_eq!(
        execute_in_context(&program, &[], &mut state, &context),
        Ok(Value::number(5.0))
    );
    assert!(
        state
            .heap()
            .datum_field(src, &field("count"))
            .unwrap()
            .semantic_eq(&Value::number(5.0))
    );
}

#[test]
fn standalone_prefix_and_postfix_increments_are_valid_statements() {
    let source = "/proc/update()\n\tcount++\n\tvar/debt = 2\n\t--debt\n\treturn count - debt\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure_with_resolver_and_fields(
        &syntax.definitions[0],
        &HashMap::new(),
        &BTreeMap::from([("count".to_owned(), field("count"))]),
        &BTreeMap::new(),
        &BTreeMap::new(),
    )
    .expect("increments should compile");
    let mut state = ExecutionState::new();
    let src = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/source").unwrap());
    state
        .heap_mut()
        .set_datum_field(src, field("count"), Value::number(3.0))
        .unwrap();
    assert_eq!(
        execute_in_context(
            &program,
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(src), Value::Null),
        ),
        Ok(Value::number(3.0))
    );
}

#[test]
fn standalone_increment_updates_a_bound_global() {
    let syntax = parse("/proc/update()\n\tuid++\n\treturn uid\n").expect("source should parse");
    let program = compile_procedure_with_resolver_and_fields(
        &syntax.definitions[0],
        &HashMap::new(),
        &BTreeMap::new(),
        &BTreeMap::from([("uid".to_owned(), field("qualified_uid"))]),
        &BTreeMap::new(),
    )
    .expect("a bound global increment should compile");
    let mut state = ExecutionState::new();
    state.set_global(field("qualified_uid"), Value::number(4.0));

    assert_eq!(
        execute_in_state(&program, &[], &mut state),
        Ok(Value::number(5.0))
    );
    assert_eq!(
        state.global(&field("qualified_uid")),
        Some(&Value::number(5.0))
    );
}

#[test]
fn link_builtin_preserves_headless_redirect_payload() {
    let syntax =
        parse("/proc/run()\n\treturn link(\"byond://server\")\n").expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("link should compile as a builtin");
    let entry = module.procedure_id("/proc/run").expect("run entry");
    assert_eq!(
        execute_module(&module, entry, &[]),
        Ok(Value::text("byond://server"))
    );
}

#[test]
fn clamp_accepts_reversed_numeric_bounds() {
    let source = "/proc/test()\n\treturn clamp(15, 10, 0)\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("clamp should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::number(10.0)));
}

#[test]
fn clamp_list_returns_new_clamped_numeric_values() {
    let source = "/proc/test()\n\tvar/list/input = list(-10, \"skip\", 5, 40)\n\tvar/list/output = clamp(input, 1, 10)\n\treturn output[1] * 100 + output[2] * 10 + output[3]\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("list clamp should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::number(160.0)));
}

#[test]
fn ckey_and_ckey_ex_accept_null_like_byond() {
    let source = "/proc/test()\n\treturn ckey(null)".to_owned() + "\n";
    let syntax = parse(&source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("ckey should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::text("")));

    let source = "/proc/test()\n\treturn ckey(123)".to_owned() + "\n";
    let syntax = parse(&source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("ckey should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::text("123")));

    let source = "/proc/test()\n\treturn ckeyEx(null)".to_owned() + "\n";
    let syntax = parse(&source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("ckeyEx should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::text("")));
}

#[test]
fn inverse_trig_builtins_use_dm_degrees_and_fallbacks() {
    let source = "/proc/test()\n\treturn round(arctan(3, 4)) + round(arctan(-1, 1)) + round(arcsin(1)) + round(arccos(0)) + arcsin(2)\n";
    let syntax = parse(source).expect("source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("inverse trig builtins should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::number(368.0)));
}

#[test]
fn prefix_increment_is_an_expression_for_list_indexing() {
    let source =
        "/proc/test()\n\tvar/list/values = list(10, 20)\n\tvar/i = 0\n\treturn values[++i]\n";
    let syntax = parse(source).expect("source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("prefix increment should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::number(10.0)));
}

#[test]
fn increment_expressions_follow_byond_coercion_and_return_rules() {
    let source = "/proc/test()\n\tvar/a = 1\n\tvar/old = a++\n\tvar/new_value = ++a\n\tvar/text_value = \"bad\"\n\tvar/text_new = ++text_value\n\tvar/null_value = null\n\tvar/null_new = ++null_value\n\tvar/list/values = list(1)\n\tvar/list_old = values[1]++\n\tvar/list_new = values[1]\n\treturn old * 10000 + new_value * 1000 + text_new * 100 + null_new * 10 + list_old + list_new\n";
    let syntax = parse(source).expect("source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("increment expressions should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::number(13_113.0)));
}

#[test]
fn incrementing_an_absent_associative_key_inserts_one() {
    let source = "/proc/test(target)\n\tvar/list/counter = list()\n\tvar/old = counter[target]++\n\treturn isnull(old) + (counter[target] == 1)\n";
    let syntax = parse(source).expect("beauty-counter-shaped source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("beauty-counter-shaped mutation should compile");
    let mut state = ExecutionState::new();
    let target = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/effect/decal/cleanable").unwrap());
    assert_eq!(
        execute_in_state(&program, &[Value::Datum(target)], &mut state),
        Ok(Value::number(2.0))
    );
}

#[test]
fn decrement_expressions_preserve_postfix_old_value() {
    let source = "/proc/test()\n\tvar/value = 3\n\tvar/old = value--\n\tvar/new_value = --value\n\treturn old * 10 + new_value\n";
    let syntax = parse(source).expect("source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("decrement expressions should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::number(31.0)));
}

#[test]
fn field_increment_expressions_mutate_once_and_return_correct_value() {
    let source =
        "/proc/test()\n\tvar/old = count++\n\tvar/current = ++count\n\treturn old * 10 + current\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure_with_resolver_and_fields(
        &syntax.definitions[0],
        &HashMap::new(),
        &BTreeMap::from([("count".to_owned(), field("count"))]),
        &BTreeMap::new(),
        &BTreeMap::new(),
    )
    .expect("field mutation should compile");
    let mut state = ExecutionState::new();
    let src = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/source").unwrap());
    state
        .heap_mut()
        .set_datum_field(src, field("count"), Value::number(3.0))
        .unwrap();
    assert_eq!(
        execute_in_context(
            &program,
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(src), Value::Null),
        ),
        Ok(Value::number(35.0))
    );
    assert_eq!(
        state.heap().datum_field(src, &field("count")),
        Ok(&Value::number(5.0))
    );
}

#[test]
fn src_and_usr_aliases_observe_the_same_datum_write() {
    let source = "/proc/alias()\n\tsrc.value = 7\n\treturn usr.value\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("fields should compile");
    let mut state = ExecutionState::new();
    let datum = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/shared").unwrap());
    let context = ExecutionContext::new(Value::Datum(datum), Value::Datum(datum));

    assert_eq!(
        execute_in_context(&program, &[], &mut state, &context),
        Ok(Value::number(7.0))
    );
}

#[test]
fn globals_persist_across_executions_and_compound_updates() {
    let set_source =
        parse("/proc/set_global()\n\tglobal.counter = 4\n\treturn global.counter\n").unwrap();
    let increment_source =
        parse("/proc/increment_global()\n\tglobal.counter += 1\n\treturn global.counter\n")
            .unwrap();
    let setter = compile_procedure(&set_source.definitions[0]).unwrap();
    let incrementer = compile_procedure(&increment_source.definitions[0]).unwrap();
    let mut state = ExecutionState::new();

    assert_eq!(
        execute_in_state(&setter, &[], &mut state),
        Ok(Value::number(4.0))
    );
    assert_eq!(
        execute_in_state(&incrementer, &[], &mut state),
        Ok(Value::number(5.0))
    );
    assert!(
        state
            .global(&field("counter"))
            .unwrap()
            .semantic_eq(&Value::number(5.0))
    );
    assert_eq!(state.globals().count(), 1);
}

#[test]
fn dense_globals_preserve_order_replacement_and_slot_reuse() {
    let mut state = ExecutionState::new();
    assert_eq!(state.set_global(field("zeta"), Value::number(1.0)), None);
    assert_eq!(state.set_global(field("alpha"), Value::number(2.0)), None);
    assert_eq!(
        state.set_global(field("zeta"), Value::number(3.0)),
        Some(Value::number(1.0))
    );
    assert_eq!(
        state
            .globals()
            .map(|(name, value)| (name.as_str(), value.clone()))
            .collect::<Vec<_>>(),
        vec![("alpha", Value::number(2.0)), ("zeta", Value::number(3.0)),]
    );
    assert_eq!(
        state.delete_global(&field("alpha")),
        Some(Value::number(2.0))
    );
    assert_eq!(state.set_global(field("middle"), Value::number(4.0)), None);
    assert_eq!(state.global(&field("zeta")), Some(&Value::number(3.0)));
    assert_eq!(
        state
            .globals()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        vec!["middle", "zeta"]
    );
}

#[test]
fn lowercase_global_namespace_remains_distinct_from_declared_glob_datum() {
    let syntax = parse(
            "/datum/revision/proc/load()\n\treturn 7\n/datum/controller/global_vars/proc/InitGlobalrevdata()\n\tsrc.revdata = new /datum/revision\n/datum/controller/global_vars/Initialize()\n\tfor(var/glob_proc in typesof(/datum/controller/global_vars/proc))\n\t\tcall(src, glob_proc)()\n/proc/early_log()\n\tGLOB.config_error_log = \"early.log\"\n\treturn GLOB.config_error_log\n/proc/run()\n\tGLOB.Initialize()\n\tglobal.counter += 1\n\treturn GLOB.revdata.load() + global.counter\n",
        )
        .unwrap();
    let module = compile_module_with_global_fields(
        &syntax.definitions,
        &BTreeMap::from([
            ("GLOB".to_owned(), field("GLOB")),
            ("counter".to_owned(), field("counter")),
            (
                "GLOB.config_error_log".to_owned(),
                FieldName::static_storage("/datum/controller/global_vars/var/config_error_log"),
            ),
        ]),
    )
    .unwrap();
    let mut state = ExecutionState::new();
    state.set_global(field("GLOB"), Value::Null);
    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/early_log").unwrap(),
            &[],
            &mut state,
        ),
        Ok(Value::text("early.log"))
    );
    let glob = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/controller/global_vars").unwrap());
    state
        .heap_mut()
        .set_datum_field(glob, field("revdata"), Value::Null)
        .unwrap();
    state.set_global(field("GLOB"), Value::Datum(glob));
    state.set_global(field("counter"), Value::number(4.0));
    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/run").unwrap(),
            &[],
            &mut state,
        ),
        Ok(Value::number(12.0))
    );
    assert!(matches!(
        state.heap().datum_field(glob, &field("revdata")),
        Ok(Value::Datum(_))
    ));
    assert_eq!(state.global(&field("counter")), Some(&Value::number(5.0)));
}

#[test]
fn assignment_expressions_store_and_yield_the_assigned_value() {
    let source = parse(
            "/proc/locals_and_list(items)\n\tvar/local = 1\n\treturn (local = 5) + (items[1] = local)\n/proc/global_assignment()\n\treturn (global.counter = 9)\n",
        )
        .unwrap();
    let module = compile_module(&source.definitions).unwrap();
    let local_entry = module.procedure_id("/proc/locals_and_list").unwrap();
    let global_entry = module.procedure_id("/proc/global_assignment").unwrap();
    let mut state = ExecutionState::new();
    let list = state.heap.allocate_list();
    state.heap.list_mut(list).unwrap().add(Value::number(0.0));

    assert_eq!(
        execute_module_in_state(&module, local_entry, &[Value::List(list)], &mut state),
        Ok(Value::number(10.0))
    );
    assert_eq!(
        state.heap.list(list).unwrap().positions().next(),
        Some((1, &Value::number(5.0)))
    );
    assert_eq!(
        execute_module_in_state(&module, global_entry, &[], &mut state),
        Ok(Value::number(9.0))
    );
    assert_eq!(state.global(&field("counter")), Some(&Value::number(9.0)));
}

#[test]
fn nameof_procedure_reference_lowers_to_the_procedure_name() {
    let source = parse(
            "/proc/main()\n\treturn capture(nameof(.proc/on_signal))\n/proc/capture(value)\n\treturn value\n",
        )
        .unwrap();
    let module = compile_module(&source.definitions).unwrap();
    let entry = module.procedure_id("/proc/main").unwrap();

    assert_eq!(
        execute_module(&module, entry, &[]),
        Ok(Value::text("on_signal"))
    );
}

#[test]
fn nameof_accepts_type_and_static_member_references() {
    let source = parse(
        "/proc/main()\n\treturn list(nameof(/datum/example.proc/run), nameof(type::field))\n",
    )
    .unwrap();
    let module = compile_module(&source.definitions).unwrap();
    let entry = module.procedure_id("/proc/main").unwrap();
    let mut state = ExecutionState::new();

    let result = execute_module_in_state(&module, entry, &[], &mut state).unwrap();
    let Value::List(list) = result else {
        panic!("expected list result");
    };
    assert_eq!(
        state
            .heap()
            .list(list)
            .unwrap()
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>(),
        vec![Value::text("run"), Value::text("field")]
    );
}

#[test]
fn named_and_parent_calls_preserve_object_context() {
    let source =
        parse("/proc/main()\n\treturn helper()\n/proc/helper()\n\treturn usr.value\n").unwrap();
    let module = compile_module(&source.definitions).unwrap();
    let entry = module.procedure_id("/proc/main").unwrap();
    let mut state = ExecutionState::new();
    let usr = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/mob/user").unwrap());
    state
        .heap_mut()
        .set_datum_field(usr, field("value"), Value::number(6.0))
        .unwrap();
    let context = ExecutionContext::new(Value::Null, Value::Datum(usr));
    assert_eq!(
        execute_module_in_context(&module, entry, &[], &mut state, &context),
        Ok(Value::number(6.0))
    );

    let parent_source =
        parse("/proc/base()\n\treturn src.value\n/proc/child()\n\treturn ..()\n").unwrap();
    let parent_module = compile_module_specs(&[
        ProcedureSpec {
            path: "/proc/base@0".to_owned(),
            definition: &parent_source.definitions[0],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
        ProcedureSpec {
            path: "/proc/child@1".to_owned(),
            definition: &parent_source.definitions[1],
            parent: Some(0),
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
    ])
    .unwrap();
    let child = parent_module.procedure_id_at(1).unwrap();
    let src = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/source").unwrap());
    state
        .heap_mut()
        .set_datum_field(src, field("value"), Value::number(8.0))
        .unwrap();
    let context = ExecutionContext::new(Value::Datum(src), Value::Null);
    assert_eq!(
        execute_module_in_context(&parent_module, child, &[], &mut state, &context),
        Ok(Value::number(8.0))
    );

    let current_source = parse(
        "/proc/recurse(depth)\n\tif(depth <= 0)\n\t\treturn src.value\n\treturn .(depth - 1)\n",
    )
    .unwrap();
    let current_program = compile_procedure(&current_source.definitions[0]).unwrap();
    assert_eq!(
        execute_in_context(
            &current_program,
            &[Value::number(2.0)],
            &mut state,
            &context,
        ),
        Ok(Value::number(8.0))
    );
}

#[test]
fn symbolic_dynamic_target_compiles_once_and_survives_scheduler_yield() {
    let source = parse(
            "/proc/entry(receiver)\n\treturn receiver.run()\n/datum/child/proc/run()\n\tsleep(1)\n\treturn 9\n",
        )
        .unwrap();
    let specs = [
        ProcedureSpec {
            path: "/proc/entry@0".to_owned(),
            definition: &source.definitions[0],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
        ProcedureSpec {
            path: "/datum/child/proc/run@0".to_owned(),
            definition: &source.definitions[1],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
    ];
    let module = compile_module_specs_selective(
        &specs,
        &[BTreeMap::new(), BTreeMap::new()],
        &BTreeSet::from([0]),
    )
    .unwrap();
    assert_eq!(module.deferred_procedure_count(), 1);
    assert_eq!(module.materialized_deferred_procedure_count(), 0);
    let deferred_id = module.procedure_id_at(1).unwrap();
    let cloned_module = module.clone();
    assert!(Arc::ptr_eq(&module.deferred, &cloned_module.deferred));
    let original_deferred = module.deferred.get(&deferred_id).unwrap();
    let cloned_deferred = cloned_module.deferred.get(&deferred_id).unwrap();
    assert!(Arc::ptr_eq(
        &original_deferred.definition,
        &cloned_deferred.definition
    ));
    assert!(Arc::ptr_eq(
        &original_deferred.targets,
        &cloned_deferred.targets
    ));
    assert!(Arc::ptr_eq(
        &original_deferred.compiled,
        &cloned_deferred.compiled
    ));

    let mut state = ExecutionState::new();
    let receiver = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/child").unwrap());
    let entry = module.procedure_id_at(0).unwrap();
    assert_eq!(
        execute_module_in_context(
            &module,
            entry,
            &[Value::Datum(receiver)],
            &mut state,
            &ExecutionContext::default(),
        ),
        Ok(Value::Null)
    );
    assert_eq!(module.materialized_deferred_procedure_count(), 1);
    assert_eq!(state.scheduled_task_count(), 1);
    assert_eq!(
        advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
        Ok(vec![Value::number(9.0)])
    );
    assert_eq!(module.materialized_deferred_procedure_count(), 1);
}

#[test]
fn deferred_semantic_error_blocks_only_when_runtime_selects_symbol() {
    let source = parse(
        "/proc/entry(receiver)\n\treturn receiver.run()\n/datum/child/proc/run()\n\treturn 9\n",
    )
    .unwrap();
    let specs = [
        ProcedureSpec {
            path: "/proc/entry@0".to_owned(),
            definition: &source.definitions[0],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
        ProcedureSpec {
            path: "/datum/child/proc/run@0".to_owned(),
            definition: &source.definitions[1],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
    ];
    let module = compile_module_specs_selective_with_errors(
        &specs,
        &[BTreeMap::new(), BTreeMap::new()],
        &BTreeSet::from([0]),
        &BTreeMap::from([(
            1,
            super::CompileError {
                message: "deferred source semantic failure".to_owned(),
            },
        )]),
    )
    .expect("unselected deferred semantic error must not block module linking");
    assert_eq!(module.materialized_deferred_procedure_count(), 0);

    let mut state = ExecutionState::new();
    let receiver = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/child").unwrap());
    let error = execute_module_in_context(
        &module,
        module.procedure_id_at(0).unwrap(),
        &[Value::Datum(receiver)],
        &mut state,
        &ExecutionContext::default(),
    )
    .expect_err("selecting the invalid deferred symbol must fail");
    assert_eq!(error.message, "deferred source semantic failure");
    assert_eq!(module.materialized_deferred_procedure_count(), 1);
}

#[test]
fn fully_eager_module_preserves_indexes_identity_and_dynamic_dispatch() {
    let source = parse(
            "/proc/entry(receiver)\n\treturn receiver.run()\n/datum/alpha/proc/run()\n\treturn 11\n/datum/beta/proc/run()\n\treturn 22\n",
        )
        .unwrap();
    let specs = [
        ProcedureSpec {
            path: "/proc/entry@0".to_owned(),
            definition: &source.definitions[0],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
        ProcedureSpec {
            path: "/datum/alpha/proc/run@0".to_owned(),
            definition: &source.definitions[1],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
        ProcedureSpec {
            path: "/datum/beta/proc/run@0".to_owned(),
            definition: &source.definitions[2],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
    ];
    let module = compile_module_specs_selective(
        &specs,
        &[BTreeMap::new(), BTreeMap::new(), BTreeMap::new()],
        &BTreeSet::from([0]),
    )
    .unwrap();
    let lazy_observer = module.clone();
    let expected_identity = module.identity.0;
    let expected_paths = module.paths.clone();
    let expected_names = module.names.clone();
    let expected_dynamic_names = module.dynamic_names.clone();
    let expected_procedure_types = module.procedure_types.clone();

    let eager = module.clone().into_fully_eager().unwrap();
    let independently_eager = module.into_fully_eager().unwrap();
    assert_eq!(eager.deferred_procedure_count(), 0);
    assert_eq!(eager.materialized_deferred_procedure_count(), 0);
    assert_eq!(eager.identity.0, expected_identity);
    assert_eq!(eager.paths, expected_paths);
    assert_eq!(eager.names, expected_names);
    assert_eq!(eager.dynamic_names, expected_dynamic_names);
    assert_eq!(eager.procedure_types, expected_procedure_types);
    assert_eq!(eager, independently_eager);
    assert_eq!(eager, eager.clone());

    assert_eq!(lazy_observer.deferred_procedure_count(), 2);
    assert_eq!(lazy_observer.materialized_deferred_procedure_count(), 0);
    let entry = eager.procedure_id_at(0).unwrap();
    let mut state = ExecutionState::new();
    for (path, expected) in [("/datum/alpha", 11.0), ("/datum/beta", 22.0)] {
        let receiver = state
            .heap_mut()
            .allocate_datum(TypePath::parse(path).unwrap());
        assert_eq!(
            execute_module_in_context(
                &eager,
                entry,
                &[Value::Datum(receiver)],
                &mut state,
                &ExecutionContext::default(),
            ),
            Ok(Value::number(expected))
        );
    }
    assert_eq!(lazy_observer.materialized_deferred_procedure_count(), 0);
}

#[test]
fn fully_eager_module_surfaces_first_preflight_error_without_materializing_clones() {
    let source =
        parse("/proc/entry()\n\treturn 1\n/proc/first()\n\treturn 2\n/proc/second()\n\treturn 3\n")
            .unwrap();
    let specs = source
        .definitions
        .iter()
        .enumerate()
        .map(|(index, definition)| ProcedureSpec {
            path: ["/proc/entry@0", "/proc/first@0", "/proc/second@0"][index].to_owned(),
            definition,
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        })
        .collect::<Vec<_>>();
    let module = compile_module_specs_selective_with_errors(
        &specs,
        &[BTreeMap::new(), BTreeMap::new(), BTreeMap::new()],
        &BTreeSet::from([0]),
        &BTreeMap::from([
            (
                2,
                super::CompileError {
                    message: "second preflight failure".to_owned(),
                },
            ),
            (
                1,
                super::CompileError {
                    message: "first preflight failure".to_owned(),
                },
            ),
        ]),
    )
    .unwrap();
    let observer = module.clone();

    let error = module.into_fully_eager().unwrap_err();
    assert_eq!(error.message, "/proc/first@0: first preflight failure");
    assert_eq!(observer.materialized_deferred_procedure_count(), 0);
}

#[test]
fn fully_eager_module_surfaces_lowering_errors_in_procedure_order() {
    let source = parse(
            "/proc/entry()\n\treturn 1\n/proc/first_bad()\n\treturn unknown_first\n/proc/second_bad()\n\treturn unknown_second\n",
        )
        .unwrap();
    let specs = source
        .definitions
        .iter()
        .enumerate()
        .map(|(index, definition)| ProcedureSpec {
            path: ["/proc/entry@0", "/proc/first_bad@0", "/proc/second_bad@0"][index].to_owned(),
            definition,
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        })
        .collect::<Vec<_>>();
    let module = compile_module_specs_selective(
        &specs,
        &[BTreeMap::new(), BTreeMap::new(), BTreeMap::new()],
        &BTreeSet::from([0]),
    )
    .unwrap();

    let error = module.into_fully_eager().unwrap_err();
    assert!(error.message.starts_with("/proc/first_bad@0: "));
    assert!(error.message.contains("unknown local \"unknown_first\""));
}

#[test]
fn bounded_fully_eager_diagnostics_compile_good_bodies_and_report_stable_failures() {
    let source = parse(concat!(
        "/proc/entry()\n\treturn 1\n",
        "/proc/good_before()\n\treturn 11\n",
        "/proc/first_bad()\n\treturn unknown_first\n",
        "/proc/good_after()\n\treturn 22\n",
        "/proc/second_bad()\n\treturn unknown_second\n",
        "/proc/third_bad()\n\treturn unknown_third\n",
    ))
    .unwrap();
    let paths = [
        "/proc/entry@0",
        "/proc/good_before@0",
        "/proc/first_bad@0",
        "/proc/good_after@0",
        "/proc/second_bad@0",
        "/proc/third_bad@0",
    ];
    let specs = source
        .definitions
        .iter()
        .enumerate()
        .map(|(index, definition)| ProcedureSpec {
            path: paths[index].to_owned(),
            definition,
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        })
        .collect::<Vec<_>>();
    let mut module = compile_module_specs_selective(
        &specs,
        &vec![BTreeMap::new(); specs.len()],
        &BTreeSet::from([0]),
    )
    .unwrap();

    let errors = module
        .materialize_fully_eager_bounded(2)
        .expect_err("three independent deferred bodies must fail");
    assert_eq!(errors.total_failures(), 3);
    assert_eq!(errors.successful_procedures(), 2);
    assert_eq!(errors.diagnostics().len(), 2);
    assert!(
        errors.diagnostics()[0]
            .message
            .starts_with("/proc/first_bad@0: ")
    );
    assert!(
        errors.diagnostics()[0]
            .message
            .contains("unknown local \"unknown_first\"")
    );
    assert!(
        errors.diagnostics()[1]
            .message
            .starts_with("/proc/second_bad@0: ")
    );
    assert!(
        errors.diagnostics()[1]
            .message
            .contains("unknown local \"unknown_second\"")
    );
    let rendered = errors.to_string();
    assert!(rendered.starts_with(
            "3 deferred procedures failed eager compilation; 2 compiled successfully; showing first 2 failures\n- /proc/first_bad@0: "
        ));
    assert!(!rendered.contains("third_bad"));

    assert_eq!(module.deferred_procedure_count(), 3);
    let mut state = ExecutionState::new();
    for (index, expected) in [(1, 11.0), (3, 22.0)] {
        assert_eq!(
            execute_module_in_context(
                &module,
                module.procedure_id_at(index).unwrap(),
                &[],
                &mut state,
                &ExecutionContext::default(),
            ),
            Ok(Value::number(expected))
        );
    }
    assert_eq!(module.deferred_procedure_count(), 3);
}

#[test]
fn static_call_statement_executes_and_discards_its_result() {
    let source = parse(
            "/proc/entry()\n\thelper()\n\treturn global.calls\n/proc/helper()\n\tglobal.calls += 1\n\treturn 99\n",
        )
        .expect("source should parse");
    let module = compile_module(&source.definitions).expect("module should compile");
    let entry = module
        .procedure_id("/proc/entry")
        .expect("entry should exist");
    let mut state = ExecutionState::new();
    state.set_global(field("calls"), Value::number(0.0));

    assert_eq!(
        execute_module_in_context(
            &module,
            entry,
            &[],
            &mut state,
            &ExecutionContext::default(),
        ),
        Ok(Value::number(1.0))
    );
}

#[test]
fn parent_call_statement_executes_and_discards_its_result() {
    let source = parse(
            "/proc/base()\n\tglobal.calls += 1\n\treturn 99\n/proc/child()\n\t..()\n\treturn global.calls\n",
        )
        .expect("source should parse");
    let module = compile_module_specs(&[
        ProcedureSpec {
            path: "/proc/base@0".to_owned(),
            definition: &source.definitions[0],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
        ProcedureSpec {
            path: "/proc/child@1".to_owned(),
            definition: &source.definitions[1],
            parent: Some(0),
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
    ])
    .expect("resolved parent specs should compile");
    let child = module.procedure_id_at(1).expect("child should exist");
    let mut state = ExecutionState::new();
    state.set_global(field("calls"), Value::number(0.0));

    assert_eq!(
        execute_module_in_context(
            &module,
            child,
            &[],
            &mut state,
            &ExecutionContext::default(),
        ),
        Ok(Value::number(1.0))
    );
    let program = module.procedure(child).expect("child program should exist");
    assert!(program.instructions.windows(2).any(|instructions| matches!(
        instructions,
        [Instruction::CallParent { .. }, Instruction::Pop]
    )));
}

#[test]
fn keyword_style_call_arguments_compile_in_source_order() {
    let source = parse(
            "/proc/entry()\n\treturn helper(first = 3, second = 4)\n/proc/helper(first, second)\n\treturn first * 10 + second\n",
        )
        .expect("source should parse");
    let module =
        compile_module(&source.definitions).expect("keyword-style call arguments should compile");
    let entry = module
        .procedure_id("/proc/entry")
        .expect("entry procedure should resolve");

    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(34.0)));
}

#[test]
fn keyword_arguments_bind_declared_slots_and_leave_omitted_defaults_unsupplied() {
    // Exact shape produced by Monkestation's addtimer macro: metadata is
    // named past two omitted optional slots and must not slide into flags.
    let source = parse(
            "/proc/entry()\n\treturn helper(1, 0, file = \"globals.dm\", line = 47)\n/proc/helper(callback, wait = 0, flags = 0, timer_subsystem, file, line)\n\treturn flags + line + (file == \"globals.dm\") * 100 + isnull(timer_subsystem) * 1000\n",
        )
        .expect("timer-shaped named call should parse");
    let module =
        compile_module(&source.definitions).expect("timer-shaped named call should compile");
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/entry").unwrap(), &[]),
        Ok(Value::number(1147.0))
    );
}

#[test]
fn keyword_style_arguments_in_discarded_calls_do_not_become_assignments() {
    let source = parse(
            "/proc/entry()\n\thelper(is_directional = TRUE, is_beam = TRUE)\n\treturn 7\n/proc/helper(first, second)\n\treturn first && second\n",
        )
        .expect("source should parse");
    let module =
        compile_module(&source.definitions).expect("discarded keyword-style calls should compile");
    let entry = module
        .procedure_id("/proc/entry")
        .expect("entry procedure should resolve");

    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(7.0)));
}

#[test]
fn keyword_style_arguments_in_datum_calls_do_not_become_assignments() {
    // Lifecycle code commonly invokes inherited datum procedures such as
    // `AddComponent(...)` rather than a global helper.  The argument
    // labels are still call syntax in that postfix form: they must not be
    // parsed as assignments to bare locals on the caller.
    let source = parse(
            "/proc/entry(receiver)\n\treceiver.AddComponent(/datum/component/overlay_lighting, is_directional = TRUE, is_beam = TRUE)\n\treturn 7\n",
        )
        .expect("source should parse");
    compile_procedure(&source.definitions[0])
        .expect("datum calls with keyword-style arguments should compile");
}

#[test]
fn keyword_style_arguments_in_bare_datum_calls_do_not_become_assignments() {
    // An inherited datum call is commonly written without an explicit
    // receiver in an atom lifecycle hook.  This is the form used by
    // `/atom/movable/Initialize` in downstream SS13 codebases.
    let source = parse(
            "/atom/movable/proc/Initialize()\n\tAddComponent(/datum/component/overlay_lighting, is_directional = TRUE, is_beam = TRUE)\n/atom/proc/AddComponent(first, second, third)\n\treturn\n",
        )
        .expect("source should parse");
    compile_module_specs(&[
        ProcedureSpec {
            path: "/atom/movable/proc/Initialize@0".to_owned(),
            definition: &source.definitions[0],
            parent: None,
            static_calls: BTreeMap::from([("AddComponent".to_owned(), 1)]),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
        ProcedureSpec {
            path: "/atom/proc/AddComponent@0".to_owned(),
            definition: &source.definitions[1],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
    ])
    .expect("bare datum calls with keyword-style arguments should compile");
}

#[test]
fn macro_wrapped_named_arguments_become_textual_list_keys() {
    // Monkestation's `AddComponent` macro expands to this exact shape.
    // The labels survive only as associative list keys once expansion has
    // occurred, so they must not be lowered as caller locals.
    let source = parse(
            "/proc/entry()\n\t_AddComponent(list(/datum/component/overlay_lighting, is_directional = TRUE, is_beam = TRUE))\n\treturn 7\n/proc/_AddComponent(raw_args)\n\treturn 0\n",
        )
        .expect("source should parse");
    let module =
        compile_module(&source.definitions).expect("macro-wrapped named arguments should compile");
    let entry = module
        .procedure_id("/proc/entry")
        .expect("entry procedure should resolve");

    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(7.0)));
}

#[test]
fn weighted_pick_style_semicolons_select_a_builtin_candidate() {
    let source = parse("/proc/entry()\n\treturn pick(10; 3, 1; 4)\n").expect("source should parse");
    let module = compile_module(&source.definitions).expect("weighted pick syntax should compile");
    let entry = module
        .procedure_id("/proc/entry")
        .expect("entry should exist");

    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(3.0)));
    assert!(module.procedure(entry).expect("entry program should exist").instructions.iter().any(
            |instruction| matches!(instruction, Instruction::Pick { weighted } if weighted == &vec![true, true]),
        ));
}

#[test]
fn namespace_qualified_call_is_parsed_as_static_call() {
    let source =
        parse("/proc/entry()\n\tTypeA::helper()\n\treturn 11\n/proc/helper()\n\treturn 11\n")
            .expect("source should parse");
    let module = compile_module(&source.definitions)
        .expect("namespace-qualified static calls should compile");
    let entry = module
        .procedure_id("/proc/entry")
        .expect("entry should resolve");
    let helper = module
        .procedure_id("/proc/helper")
        .expect("helper should resolve");

    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(11.0)));
    let entry_program = module.procedure(entry).expect("entry program should exist");
    assert!(
        entry_program.instructions.iter().any(|instruction| {
            matches!(instruction, Instruction::Call { procedure, .. } if *procedure == helper)
        }),
        "namespace-qualified call should resolve to a real static call",
    );
}

#[test]
fn spawn_statement_runs_only_when_its_scheduler_delay_elapses() {
    let source = parse(
        "/proc/entry()\n\tspawn(1)\n\t\thelper()\n\treturn 11\n/proc/helper()\n\treturn 22\n",
    )
    .expect("source should parse");
    let module = compile_module(&source.definitions).expect("spawn statement should compile");
    let entry = module
        .procedure_id("/proc/entry")
        .expect("entry should resolve");

    let mut state = ExecutionState::new();
    assert_eq!(
        execute_module_in_state(&module, entry, &[], &mut state),
        Ok(Value::number(11.0))
    );
    assert_eq!(
        advance_scheduler(&module, 0, ExecutionLimits::default(), &mut state),
        Ok(Vec::new())
    );
    assert_eq!(
        advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
        Ok(vec![Value::Null])
    );
}

#[test]
fn inline_spawned_assignment_is_one_detached_statement() {
    let source = parse(concat!(
        "/proc/entry()\n",
        "\tspawn(1) global.spawn_value = 9\n",
        "\treturn 1\n",
        "/proc/read_spawn_value()\n",
        "\treturn global.spawn_value\n",
    ))
    .expect("source should parse");
    let module =
        compile_module(&source.definitions).expect("inline spawned assignment should compile");
    let entry = module.procedure_id("/proc/entry").expect("entry");
    let mut state = ExecutionState::new();
    assert_eq!(
        execute_module_in_state(&module, entry, &[], &mut state),
        Ok(Value::number(1.0))
    );
    assert_eq!(
        advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
        Ok(vec![Value::Null])
    );
    let read = module
        .procedure_id("/proc/read_spawn_value")
        .expect("reader");
    assert_eq!(
        execute_module_in_state(&module, read, &[], &mut state),
        Ok(Value::number(9.0))
    );
}

#[test]
fn compound_pointer_dereference_updates_the_pointed_local() {
    let source = parse(concat!(
        "/proc/scale(pointer)\n",
        "\t*pointer *= 4\n",
        "/proc/entry()\n",
        "\tvar/value = 3\n",
        "\tscale(&value)\n",
        "\treturn value\n",
    ))
    .expect("source should parse");
    let module =
        compile_module(&source.definitions).expect("compound pointer dereference should compile");
    let entry = module.procedure_id("/proc/entry").expect("entry");
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(12.0)));
}

#[test]
fn pick_expands_arglist_before_applying_single_list_semantics() {
    let source = parse(concat!(
        "/proc/expanded_pick(...)\n",
        "\treturn pick(arglist(args))\n",
        "/proc/entry()\n",
        "\treturn expanded_pick(list(9))\n",
    ))
    .expect("source should parse");
    let module = compile_module(&source.definitions).expect("pick(arglist(...)) should compile");
    let entry = module.procedure_id("/proc/entry").expect("entry");
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(9.0)));
}

#[test]
fn dynamic_field_in_nested_conditional_false_arm_keeps_outer_ternary_delimiter() {
    let source = parse(concat!(
        "/proc/entry()\n",
        "\tvar/datum/item = new\n",
        "\tvar/list/cache = list(/datum = 7)\n",
        "\treturn cache ? cache[(ispath(item) ? item : item:type)] : 0\n",
    ))
    .expect("source should parse");
    let module = compile_module(&source.definitions)
        .expect("nested conditional dynamic field should compile");
    let entry = module.procedure_id("/proc/entry").expect("entry");
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(7.0)));
}

#[test]
fn positive_spawn_delays_floor_by_tick_lag_with_a_one_tick_minimum() {
    let source = parse(concat!(
        "/proc/entry(delay)\n",
        "\tspawn(delay)\n",
        "\t\treturn 7\n",
        "\treturn 11\n",
    ))
    .expect("spawn delay fixture should parse");
    let module = compile_module(&source.definitions).expect("spawn delay fixture should compile");
    let entry = module.procedure_id("/proc/entry").unwrap();

    for (delay, due_tick) in [(0.1, 1), (1.0, 1), (3.0, 1), (4.9, 2)] {
        let mut state = ExecutionState::new();
        let world = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/world").unwrap());
        state
            .heap_mut()
            .set_datum_field(world, field("tick_lag"), Value::number(2.0))
            .unwrap();
        state.set_global(field("world"), Value::Datum(world));

        assert_eq!(
            execute_module_in_state(&module, entry, &[Value::number(delay)], &mut state,),
            Ok(Value::number(11.0)),
        );
        assert_eq!(state.next_scheduled_tick(), Some(due_tick));
        assert_eq!(
            advance_scheduler(
                &module,
                due_tick.saturating_sub(1),
                ExecutionLimits::default(),
                &mut state,
            ),
            Ok(Vec::new()),
        );
        assert_eq!(
            advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
            Ok(vec![Value::number(7.0)]),
        );
    }
}

#[test]
fn positive_sleep_delays_resume_nested_callers_on_byond_tick_deadlines() {
    let source = parse(concat!(
        "/proc/entry(delay)\n",
        "\treturn sleeping_wrapper(delay) + 1\n",
        "/proc/sleeping_wrapper(delay)\n",
        "\tsleep(delay)\n",
        "\treturn 20\n",
    ))
    .expect("sleep delay fixture should parse");
    let module = compile_module(&source.definitions).expect("sleep delay fixture should compile");
    let entry = module.procedure_id("/proc/entry").unwrap();

    for (delay, due_tick) in [(0.1, 1), (1.0, 1), (3.0, 1), (4.9, 2)] {
        let mut state = ExecutionState::new();
        let world = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/world").unwrap());
        state
            .heap_mut()
            .set_datum_field(world, field("tick_lag"), Value::number(2.0))
            .unwrap();
        state.set_global(field("world"), Value::Datum(world));

        assert_eq!(
            execute_module_in_state(&module, entry, &[Value::number(delay)], &mut state,),
            Ok(Value::Null),
        );
        assert_eq!(state.next_scheduled_tick(), Some(due_tick));
        assert_eq!(
            advance_scheduler(
                &module,
                due_tick.saturating_sub(1),
                ExecutionLimits::default(),
                &mut state,
            ),
            Ok(Vec::new()),
        );
        assert_eq!(
            advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
            Ok(vec![Value::number(21.0)]),
        );
    }
}

#[test]
fn bounded_scheduler_progress_reports_suspended_map_frame_without_sampling() {
    let source =
        parse("/datum/parsed_map/proc/_tgm_load(file)\n\tsleep(5)\n\treturn file\n").unwrap();
    let module = compile_module(&source.definitions).unwrap();
    let entry = module
        .procedure_id("/datum/parsed_map/proc/_tgm_load")
        .unwrap();
    let mut state = ExecutionState::new();
    let parsed_map = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/parsed_map").unwrap());
    let models = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(models)
        .unwrap()
        .add(Value::text("aaa"));
    state
        .heap_mut()
        .set_datum_field(parsed_map, field("grid_models"), Value::List(models))
        .unwrap();
    let context = ExecutionContext::new(Value::Datum(parsed_map), Value::Null);
    assert_eq!(
        execute_module_in_context(
            &module,
            entry,
            &[Value::text("maps/test.dmm")],
            &mut state,
            &context,
        ),
        Ok(Value::Null)
    );
    let lines = state.bounded_scheduler_progress(&module);
    assert_eq!(lines.len(), 1);
    assert!(lines[0].contains("procedure=/datum/parsed_map/proc/_tgm_load"));
    assert!(lines[0].contains("parameters=[file=\"maps/test.dmm\"]"));
    assert!(lines[0].contains("map=[grid_models=list(len=1)]"));
}

#[test]
fn changing_tick_lag_does_not_move_an_existing_scheduler_deadline() {
    let source = parse("/proc/entry()\n\tspawn(4.9)\n\t\treturn 7\n\treturn 11\n")
        .expect("fixed-deadline fixture should parse");
    let module =
        compile_module(&source.definitions).expect("fixed-deadline fixture should compile");
    let entry = module.procedure_id("/proc/entry").unwrap();
    let mut state = ExecutionState::new();
    let world = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/world").unwrap());
    state
        .heap_mut()
        .set_datum_field(world, field("tick_lag"), Value::number(2.0))
        .unwrap();
    state.set_global(field("world"), Value::Datum(world));

    assert_eq!(
        execute_module_in_state(&module, entry, &[], &mut state),
        Ok(Value::number(11.0)),
    );
    assert_eq!(state.next_scheduled_tick(), Some(2));

    state
        .heap_mut()
        .set_datum_field(world, field("tick_lag"), Value::number(10.0))
        .unwrap();
    assert_eq!(state.next_scheduled_tick(), Some(2));
    assert_eq!(
        advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
        Ok(Vec::new()),
    );
    assert_eq!(
        advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
        Ok(vec![Value::number(7.0)]),
    );
}

#[test]
fn scheduler_orders_only_due_work_and_retains_future_work() {
    let source = parse(concat!(
        "/proc/entry()\n",
        "\tspawn(3)\n\t\treturn 30\n",
        "\tspawn(1)\n\t\treturn 10\n",
        "\tspawn(1)\n\t\treturn 11\n",
        "\tspawn(5)\n\t\treturn 50\n",
    ))
    .expect("scheduler ordering fixture should parse");
    let module =
        compile_module(&source.definitions).expect("scheduler ordering fixture should compile");
    let entry = module.procedure_id("/proc/entry").unwrap();
    let mut state = ExecutionState::new();

    assert_eq!(
        execute_module_in_state(&module, entry, &[], &mut state),
        Ok(Value::Null),
    );
    assert_eq!(state.scheduled_task_count(), 4);
    let metrics = state.continuation_metrics();
    assert_eq!(metrics.continuations, 4);
    assert_eq!(metrics.frames, 4);
    assert!(metrics.retained_values >= 4);
    assert_eq!(metrics.frame_header_bytes, std::mem::size_of::<CallFrame>());
    assert_eq!(
        advance_scheduler(&module, 3, ExecutionLimits::default(), &mut state),
        Ok(vec![
            Value::number(10.0),
            Value::number(11.0),
            Value::number(30.0),
        ]),
    );
    assert_eq!(state.scheduled_task_count(), 1);
    assert_eq!(state.continuation_metrics().continuations, 1);
    assert_eq!(state.next_scheduled_tick(), Some(5));
    assert_eq!(
        advance_scheduler(&module, 2, ExecutionLimits::default(), &mut state),
        Ok(vec![Value::number(50.0)]),
    );
}

#[test]
fn owned_continuation_preserves_exception_state_across_repeated_yields() {
    let source = parse(concat!(
        "/proc/entry()\n",
        "\tspawn(1)\n\t\treturn guarded()\n",
        "/proc/guarded()\n",
        "\ttry\n",
        "\t\tsleep(1)\n",
        "\t\tthrow \"boom\"\n",
        "\tcatch(var/error)\n",
        "\t\treturn 77\n",
    ))
    .expect("continuation exception fixture should parse");
    let module =
        compile_module(&source.definitions).expect("continuation exception fixture should compile");
    let entry = module.procedure_id("/proc/entry").unwrap();
    let mut state = ExecutionState::new();

    assert_eq!(
        execute_module_in_state(&module, entry, &[], &mut state),
        Ok(Value::Null),
    );
    assert_eq!(
        advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
        Ok(Vec::new()),
    );
    let suspended = state.continuation_metrics();
    assert_eq!(suspended.continuations, 1);
    assert_eq!(suspended.frames, 2);
    assert!(suspended.cold_frames >= 1);
    assert_eq!(
        advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
        Ok(vec![Value::number(77.0)]),
    );
    let empty = state.continuation_metrics();
    assert_eq!(empty.continuations, 0);
    assert_eq!(empty.frames, 0);
    assert_eq!(empty.cold_frames, 0);
    assert_eq!(empty.retained_values, 0);
    assert_eq!(empty.frame_header_bytes, std::mem::size_of::<CallFrame>());
    assert!(empty.rare_inline_bytes_avoided >= 200);
}

#[test]
fn scheduler_error_restores_unrun_due_work_in_fifo_order() {
    let source = parse(concat!(
        "/proc/entry()\n",
        "\tspawn(1)\n\t\tthrow \"boom\"\n",
        "\tspawn(1)\n\t\treturn 20\n",
        "\tspawn(1)\n\t\treturn 30\n",
    ))
    .expect("scheduler error fixture should parse");
    let module =
        compile_module(&source.definitions).expect("scheduler error fixture should compile");
    let entry = module.procedure_id("/proc/entry").unwrap();
    let mut state = ExecutionState::new();

    assert_eq!(
        execute_module_in_state(&module, entry, &[], &mut state),
        Ok(Value::Null),
    );
    assert!(advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state).is_err());
    assert_eq!(state.scheduled_task_count(), 2);
    assert_eq!(
        advance_scheduler(&module, 0, ExecutionLimits::default(), &mut state),
        Ok(vec![Value::number(20.0), Value::number(30.0)]),
    );
}

#[test]
fn execution_state_rejects_off_owner_thread_mutation_but_allows_reads() {
    let state = ExecutionState::new();
    let read = std::thread::spawn(move || state.heap().live_list_count());
    assert_eq!(read.join().expect("immutable heap read should succeed"), 0);

    let mut state = ExecutionState::new();
    let mutation = std::thread::spawn(move || {
        state.set_global(field("forbidden"), Value::number(1.0));
    });
    let panic = mutation
        .join()
        .expect_err("off-owner live mutation must panic");
    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .unwrap_or_default();
    assert!(message.contains("off its owner thread"));
}

#[test]
fn scheduler_rejects_off_owner_thread_dispatch() {
    let source = parse("/proc/entry()\n\treturn null\n").unwrap();
    let module = compile_module(&source.definitions).unwrap();
    let mut state = ExecutionState::new();
    let dispatch = std::thread::spawn(move || {
        let _ = advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state);
    });
    assert!(dispatch.join().is_err());
}

#[test]
fn scheduler_advances_world_clock_and_short_work_observes_fresh_tick_usage() {
    let source = parse(
            "/proc/entry()\n\tspawn(3)\n\t\treturn_usage()\n/proc/return_usage()\n\tworld.observed = world.tick_usage\n",
        )
        .unwrap();
    let module = compile_module(&source.definitions).unwrap();
    let entry = module.procedure_id("/proc/entry").unwrap();
    let mut state = ExecutionState::new();
    let world = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/world").unwrap());
    for (name, value) in [
        ("tick_lag", 2.0),
        ("fps", 5.0),
        ("time", 0.0),
        ("timeofday", 863_999.0),
        ("tick_usage", 0.0),
    ] {
        state
            .heap_mut()
            .set_datum_field(world, field(name), Value::number(value))
            .unwrap();
    }
    state.set_global(field("world"), Value::Datum(world));

    assert_eq!(
        execute_module_in_state(&module, entry, &[], &mut state),
        Ok(Value::Null)
    );
    assert_eq!(state.next_scheduled_tick(), Some(1));
    assert_eq!(
        advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
        Ok(vec![Value::Null])
    );
    assert_eq!(crate::world_numeric_field(&state, "time"), Some(2.0));
    assert_eq!(
        state.heap().datum_field(world, &field("observed")),
        Ok(&Value::number(0.0))
    );
    assert_eq!(crate::world_numeric_field(&state, "timeofday"), Some(1.0));
    assert_eq!(crate::world_numeric_field(&state, "tick_usage"), Some(0.0));
}

#[test]
#[ignore = "release microbenchmark"]
fn cached_world_numeric_field_microbenchmark() {
    const ITERATIONS: usize = 2_000_000;
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(FieldName::parse(std::hint::black_box("tick_usage")).unwrap());
    }
    let parsed = started.elapsed();

    let started = Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(
            super::cached_world_numeric_field(std::hint::black_box("tick_usage"))
                .unwrap()
                .clone(),
        );
    }
    let cached = started.elapsed();
    eprintln!(
        "world-field-cache: parse={parsed:?} cached={cached:?} speedup={:.2}x",
        parsed.as_secs_f64() / cached.as_secs_f64()
    );
    assert!(cached < parsed);
}

#[test]
fn scheduled_overtime_work_yields_past_the_standalone_ten_million_step_guard() {
    let source = parse(concat!(
        "/proc/start()\n",
        "\tspawn(0)\n",
        "\t\tworker()\n",
        "/proc/worker()\n",
        "\twhile(global.progress < 1000000)\n",
        "\t\tglobal.progress += 1\n",
        "\t\tif(world.tick_usage > 196)\n",
        "\t\t\tsleep(world.tick_lag)\n",
        "\treturn global.progress\n",
    ))
    .expect("overtime scheduler source should parse");
    let module = compile_module(&source.definitions).expect("overtime worker should compile");
    let worker = module.procedure_id("/proc/worker").unwrap();
    let start = module.procedure_id("/proc/start").unwrap();

    let make_state = || {
        let mut state = ExecutionState::new();
        let world = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/world").unwrap());
        for (name, value) in [("tick_lag", 1.0), ("tick_usage", 0.0)] {
            state
                .heap_mut()
                .set_datum_field(world, field(name), Value::number(value))
                .unwrap();
        }
        state.set_global(field("world"), Value::Datum(world));
        state.set_global(field("progress"), Value::number(0.0));
        state
    };

    let mut standalone = make_state();
    let error = execute_module_with_limits_in_state(
        &module,
        worker,
        &[],
        ExecutionLimits::default(),
        &mut standalone,
    )
    .expect_err("the same unscheduled worker must retain the standalone guard");
    assert_eq!(error.message, "instruction budget of 10000000 exhausted");

    let mut scheduled = make_state();
    execute_module_in_state(&module, start, &[], &mut scheduled).unwrap();
    let mut rounds = 0;
    let mut completed = Vec::new();
    // tick_usage is derived from real wall-clock elapsed time (see
    // account_scheduler_tick_usage), so how many rounds this worker needs
    // to finish depends on host speed, not a fixed instruction count.
    // Keep this bound generous so slower/loaded machines don't flake.
    while scheduled.scheduled_task_count() != 0 && rounds < 100_000 {
        let advance = u64::from(rounds != 0);
        completed.extend(
            advance_scheduler(&module, advance, ExecutionLimits::default(), &mut scheduled)
                .expect("explicit tick checks should keep each slice below the hard guard"),
        );
        rounds += 1;
    }

    assert!(rounds > 1, "the long worker should cooperatively yield");
    assert_eq!(scheduled.scheduled_task_count(), 0);
    assert_eq!(completed, vec![Value::Null]);
    assert_eq!(
        scheduled.global(&field("progress")),
        Some(&Value::number(1_000_000.0))
    );
}

#[test]
fn scheduled_step_budget_preserves_the_continuation_between_dispatch_slices() {
    let source = parse(concat!(
        "/proc/start()\n",
        "\tspawn(0)\n",
        "\t\tworker()\n",
        "/proc/worker()\n",
        "\tvar/local_progress = 0\n",
        "\twhile(local_progress < 6)\n",
        "\t\tlocal_progress += 1\n",
        "\tglobal.progress = local_progress\n",
        "\treturn local_progress\n",
    ))
    .expect("scheduled continuation source should parse");
    let module =
        compile_module(&source.definitions).expect("scheduled continuation source should compile");
    let worker = module.procedure_id("/proc/worker").unwrap();
    let start = module.procedure_id("/proc/start").unwrap();
    let limits = ExecutionLimits {
        max_steps: 5,
        ..ExecutionLimits::default()
    };

    let mut standalone = ExecutionState::new();
    let error = execute_module_with_limits_in_state(&module, worker, &[], limits, &mut standalone)
        .expect_err("unscheduled work must retain the hard step guard");
    assert_eq!(error.message, "instruction budget of 5 exhausted");

    let mut scheduled = ExecutionState::new();
    scheduled.set_global(field("progress"), Value::number(0.0));
    execute_module_in_state(&module, start, &[], &mut scheduled).unwrap();
    let mut rounds = 0;
    let mut completed = Vec::new();
    while scheduled.scheduled_task_count() != 0 && rounds < 100 {
        completed.extend(
            advance_scheduler(&module, 0, limits, &mut scheduled)
                .expect("a finite scheduled body should slice instead of failing"),
        );
        rounds += 1;
    }

    assert!(rounds > 1, "the fixture must cross at least one step slice");
    assert_eq!(scheduled.scheduled_task_count(), 0);
    assert_eq!(completed, vec![Value::Null]);
    assert_eq!(
        scheduled.global(&field("progress")),
        Some(&Value::number(6.0)),
        "the resumed frame must retain its local values and exact instruction",
    );
}

#[test]
fn scheduled_wall_budget_returns_to_host_and_resumes_exact_continuation() {
    let source = parse(concat!(
        "/proc/start()\n",
        "\tspawn(0)\n",
        "\t\tworker()\n",
        "/proc/worker()\n",
        "\tvar/local_progress = 0\n",
        "\twhile(local_progress < 100000)\n",
        "\t\tlocal_progress += 1\n",
        "\tglobal.progress = local_progress\n",
        "\treturn local_progress\n",
    ))
    .unwrap();
    let module = compile_module(&source.definitions).unwrap();
    let start = module.procedure_id("/proc/start").unwrap();
    let mut state = ExecutionState::new();
    state.set_global(field("progress"), Value::number(0.0));
    execute_module_in_state(&module, start, &[], &mut state).unwrap();

    let started = Instant::now();
    assert_eq!(
        advance_scheduler(
            &module,
            0,
            ExecutionLimits {
                wall_clock_budget: Some(Duration::ZERO),
                ..ExecutionLimits::default()
            },
            &mut state,
        ),
        Ok(Vec::new()),
    );
    assert!(started.elapsed() < Duration::from_secs(1));
    assert_eq!(state.scheduled_task_count(), 1);
    assert_eq!(state.global(&field("progress")), Some(&Value::number(0.0)));

    assert_eq!(
        advance_scheduler(&module, 0, ExecutionLimits::default(), &mut state),
        Ok(vec![Value::Null]),
    );
    assert_eq!(state.scheduled_task_count(), 0);
    assert_eq!(
        state.global(&field("progress")),
        Some(&Value::number(100000.0)),
        "wall slicing must not skip or replay logical instructions",
    );
}

#[test]
fn scheduler_wall_budget_bounds_the_whole_due_queue_and_preserves_fifo() {
    let source = parse(concat!(
        "/proc/start()\n",
        "\tspawn(0)\n\t\treturn 1\n",
        "\tspawn(0)\n\t\treturn 2\n",
        "\tspawn(0)\n\t\treturn 3\n",
    ))
    .unwrap();
    let module = compile_module(&source.definitions).unwrap();
    let start = module.procedure_id("/proc/start").unwrap();
    let mut state = ExecutionState::new();
    execute_module_in_state(&module, start, &[], &mut state).unwrap();

    let started = Instant::now();
    assert_eq!(
        advance_scheduler(
            &module,
            0,
            ExecutionLimits {
                wall_clock_budget: Some(Duration::ZERO),
                ..ExecutionLimits::default()
            },
            &mut state,
        ),
        Ok(Vec::new()),
    );
    assert!(started.elapsed() < Duration::from_secs(1));
    assert_eq!(state.scheduled_task_count(), 3);
    assert_eq!(
        advance_scheduler(&module, 0, ExecutionLimits::default(), &mut state),
        Ok(vec![
            Value::number(1.0),
            Value::number(2.0),
            Value::number(3.0),
        ]),
    );
}

#[test]
fn world_fps_and_tick_lag_assignments_remain_reciprocal() {
    let source = parse(
            "/proc/set_fps()\n\tworld.fps = 20\n\treturn world.tick_lag\n/proc/set_lag()\n\tworld.tick_lag = 2\n\treturn world.fps\n",
        )
        .unwrap();
    let module = compile_module(&source.definitions).unwrap();
    let mut state = ExecutionState::new();
    let world = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/world").unwrap());
    state.set_global(field("world"), Value::Datum(world));

    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/set_fps").unwrap(),
            &[],
            &mut state,
        ),
        Ok(Value::number(0.5))
    );
    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/set_lag").unwrap(),
            &[],
            &mut state,
        ),
        Ok(Value::number(5.0))
    );
}

#[test]
fn spawn_without_parentheses_defaults_to_zero_delay_for_inline_and_block_bodies() {
    for source in [
        "/proc/entry()\n\tspawn helper()\n\treturn 1\n/proc/helper()\n\treturn 2\n",
        "/proc/entry()\n\tspawn {\n\t\thelper()\n\t}\n\treturn 1\n/proc/helper()\n\treturn 2\n",
    ] {
        let syntax = parse(source).expect("source should parse");
        let module =
            compile_module(&syntax.definitions).expect("parenthesis-free spawn should compile");
        let entry = module.procedure_id("/proc/entry").expect("entry");
        let mut state = ExecutionState::new();
        assert_eq!(
            execute_module_in_state(&module, entry, &[], &mut state),
            Ok(Value::number(1.0))
        );
        assert_eq!(
            advance_scheduler(&module, 0, ExecutionLimits::default(), &mut state),
            Ok(vec![Value::Null])
        );
    }
}

#[test]
fn negative_spawn_advances_caller_after_immediate_dynamic_call() {
    // Exact shape emitted by Monkestation's INVOKE_ASYNC macro after an
    // initializer saves its parent result: spawn(-1) must run load_map
    // immediately, then resume after Spawn without disturbing `.`.
    let syntax = parse(concat!(
        "/obj/proc/Initialize()\n",
        "\treturn 41\n",
        "/obj/modular_map_root/proc/Initialize()\n",
        "\t. = ..()\n",
        "\tspawn(-1)\n",
        "\t\tcall(0 || src, \"load_map\")()\n",
        "/obj/modular_map_root/proc/load_map()\n",
        "\tglobal.loads += 1\n",
    ))
    .expect("INVOKE_ASYNC-shaped source should parse");
    let module = compile_module_specs(&[
        ProcedureSpec {
            path: "/obj/proc/Initialize@0".to_owned(),
            definition: &syntax.definitions[0],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
        ProcedureSpec {
            path: "/obj/modular_map_root/proc/Initialize@1".to_owned(),
            definition: &syntax.definitions[1],
            parent: Some(0),
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
        ProcedureSpec {
            path: "/obj/modular_map_root/proc/load_map@2".to_owned(),
            definition: &syntax.definitions[2],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
    ])
    .expect("INVOKE_ASYNC-shaped procedure chain should compile");
    let mut state = ExecutionState::new();
    state.set_global(field("loads"), Value::number(0.0));
    state.set_type_parents(BTreeMap::from([
        (
            TypePath::parse("/obj/modular_map_root").unwrap(),
            Some(TypePath::parse("/obj").unwrap()),
        ),
        (TypePath::parse("/obj").unwrap(), None),
    ]));
    let root = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/modular_map_root").unwrap());
    let initialize = module.procedure_id_at(1).unwrap();

    assert_eq!(
        execute_module_in_context(
            &module,
            initialize,
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(root), Value::Null),
        ),
        Ok(Value::number(41.0))
    );
    assert_eq!(state.global(&field("loads")), Some(&Value::number(1.0)));
    assert_eq!(state.scheduled_task_count(), 0);
}

#[test]
fn sleep_yields_and_resumes_the_full_procedure_frame() {
    let source = parse("/proc/entry()\n\tvar/value = sleep(1)\n\treturn value + 11\n")
        .expect("source should parse");
    let module = compile_module(&source.definitions).expect("sleep should compile");
    let entry = module
        .procedure_id("/proc/entry")
        .expect("entry should resolve");
    let mut state = ExecutionState::new();

    assert_eq!(
        execute_module_in_state(&module, entry, &[], &mut state),
        Ok(Value::Null),
        "a yielding entry returns control to the scheduler"
    );
    assert_eq!(
        advance_scheduler(&module, 0, ExecutionLimits::default(), &mut state),
        Ok(Vec::new())
    );
    assert_eq!(
        advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
        Ok(vec![Value::number(11.0)])
    );
}

#[test]
fn sleep_preserves_callers_waiting_on_a_nested_call() {
    let source =
        parse("/proc/entry()\n\treturn helper() + 1\n/proc/helper()\n\tsleep(1)\n\treturn 2\n")
            .expect("source should parse");
    let module = compile_module(&source.definitions).expect("sleep should compile");
    let entry = module
        .procedure_id("/proc/entry")
        .expect("entry should resolve");
    let mut state = ExecutionState::new();

    assert_eq!(
        execute_module_in_state(&module, entry, &[], &mut state),
        Ok(Value::Null)
    );
    assert_eq!(
        advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
        Ok(vec![Value::number(3.0)])
    );
}

#[test]
fn repeated_nested_blocks_may_redeclare_macro_locals() {
    let source = parse(
            "/proc/repeated_scopes()\n\tvar/total = 0\n\tdo { var/_L = 1; total += _L; } while(0)\n\tdo { var/_L = 2; total += _L; } while(0)\n\treturn total\n",
        )
        .expect("repeated scoped locals should parse");
    let module = compile_module(&source.definitions)
        .expect("nested blocks should permit repeated local names");
    let entry = module.procedure_id("/proc/repeated_scopes").expect("entry");
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(3.0)));
}

#[test]
fn copytext_char_uses_character_positions_and_negative_offsets() {
    let source = parse(
            "/proc/middle()\n\treturn copytext_char(\"AéB\", 2, 3)\n/proc/tail()\n\treturn copytext_char(\"Hi there\", -5)\n",
        )
        .expect("copytext_char source should parse");
    let module = compile_module(&source.definitions).expect("copytext_char should compile");
    let middle = module.procedure_id("/proc/middle").expect("middle");
    let tail = module.procedure_id("/proc/tail").expect("tail");
    assert_eq!(execute_module(&module, middle, &[]), Ok(Value::text("é")));
    assert_eq!(execute_module(&module, tail, &[]), Ok(Value::text("there")));
}

#[test]
fn file_line_text_indexing_is_one_based_character_text() {
    let syntax = parse(concat!(
        "/proc/classify_lines()\n",
        "\tvar/list/lines = splittext(\"- comment\\n	key:value\\nÅngström\", \"\\n\")\n",
        "\tvar/result = \"\"\n",
        "\tfor(var/line in lines)\n",
        "\t\tresult += line[1]\n",
        "\treturn list(result, lines[3][1], lines[1][99])\n",
    ))
    .expect("advertisement/config line indexing shape should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("text indexing should lower for file2list consumers");
    let mut state = ExecutionState::new();
    let Value::List(result) = execute_in_state(&program, &[], &mut state).unwrap() else {
        panic!("line classifier should return a list");
    };
    assert_eq!(
        state
            .heap()
            .list(result)
            .unwrap()
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>(),
        vec![Value::text("-\tÅ"), Value::text("Å"), Value::Null]
    );

    let assignment =
        parse("/proc/write_text()\n\tvar/value = \"abc\"\n\tvalue[1] = \"z\"\n").unwrap();
    let assignment = compile_procedure(&assignment.definitions[0]).unwrap();
    assert!(
        execute(&assignment, &[]).is_err(),
        "BYOND text values are immutable through index assignment"
    );
}

#[test]
fn ascii_text_index_fast_path_preserves_unicode_and_bounds_semantics() {
    assert_eq!(
        crate::indexed_text_character("\ticon_state = value;", 20),
        Value::text(";")
    );
    assert_eq!(
        crate::indexed_text_character("Ångström", 1),
        Value::text("Å")
    );
    assert_eq!(crate::indexed_text_character("abc", 99), Value::Null);
}

#[test]
#[ignore = "startup TGM line-suffix indexing benchmark"]
fn tgm_ascii_line_suffix_index_benchmark() {
    const ROUNDS: usize = 1_000_000;
    let line = "\tcolor = \"#ffffffff\";";
    let index = line.len();
    let started = Instant::now();
    for _ in 0..ROUNDS {
        std::hint::black_box(line.chars().nth(index - 1));
    }
    let unicode_iterator = started.elapsed();
    let started = Instant::now();
    for _ in 0..ROUNDS {
        std::hint::black_box(crate::indexed_text_character(line, index));
    }
    let ascii_index = started.elapsed();
    eprintln!(
        "tgm-line-index rounds={ROUNDS} iterator_ms={} ascii_ms={} speedup={:.2}",
        unicode_iterator.as_millis(),
        ascii_index.as_millis(),
        unicode_iterator.as_secs_f64() / ascii_index.as_secs_f64(),
    );
    assert!(ascii_index < unicode_iterator);
}

#[test]
fn block_enumerates_inclusive_turf_rectangles() {
    let source = parse("/proc/box(start, finish)\n\treturn block(start, finish)\n")
        .expect("block source should parse");
    let module = compile_module(&source.definitions).expect("block should compile");
    let entry = module.procedure_id("/proc/box").expect("box");
    let mut state = ExecutionState::new();
    let turf_path = TypePath::parse("/turf/test").expect("turf path");
    let mut turfs = Vec::new();
    for (x_value, y_value) in [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (2.0, 2.0)] {
        let turf = state.heap_mut().allocate_datum(turf_path.clone());
        state
            .heap_mut()
            .set_datum_field(turf, field("x"), Value::number(x_value))
            .unwrap();
        state
            .heap_mut()
            .set_datum_field(turf, field("y"), Value::number(y_value))
            .unwrap();
        state
            .heap_mut()
            .set_datum_field(turf, field("z"), Value::number(1.0))
            .unwrap();
        turfs.push(turf);
    }
    let result = execute_module_in_state(
        &module,
        entry,
        &[Value::Datum(turfs[3]), Value::Datum(turfs[0])],
        &mut state,
    )
    .expect("block should execute");
    let Value::List(list) = result else {
        panic!("block should return a list");
    };
    assert_eq!(state.heap().list(list).expect("block list").len(), 4);
}

#[test]
fn block_uses_world_coordinate_index_in_zyx_order() {
    let source = parse("/proc/box(start, finish)\n\treturn block(start, finish)\n")
        .expect("block source should parse");
    let module = compile_module(&source.definitions).expect("block should compile");
    let entry = module.procedure_id("/proc/box").expect("box");
    let mut state = ExecutionState::new();
    let turf_path = TypePath::parse("/turf/indexed").expect("turf path");
    let coordinates = [
        (2, 2, 2),
        (1, 2, 1),
        (2, 1, 2),
        (1, 1, 1),
        (2, 2, 1),
        (1, 1, 2),
        (2, 1, 1),
        (1, 2, 2),
    ];
    let mut by_coordinate = BTreeMap::new();
    for (x_value, y_value, z_value) in coordinates {
        let turf = state.heap_mut().allocate_datum(turf_path.clone());
        for (name, value) in [("x", x_value), ("y", y_value), ("z", z_value)] {
            state
                .heap_mut()
                .set_datum_field(turf, field(name), Value::number(value as f32))
                .expect("coordinate field should be writable");
        }
        state.world_turfs.insert((x_value, y_value, z_value), turf);
        by_coordinate.insert((x_value, y_value, z_value), turf);
    }
    // A duplicate coordinate that is only present in the heap proves the
    // indexed path does not regress to scanning every allocated datum.
    let unindexed = state.heap_mut().allocate_datum(turf_path);
    for (name, value) in [("x", 1), ("y", 1), ("z", 1)] {
        state
            .heap_mut()
            .set_datum_field(unindexed, field(name), Value::number(value as f32))
            .expect("unindexed coordinate field should be writable");
    }

    let result = execute_module_in_state(
        &module,
        entry,
        &[
            Value::Datum(by_coordinate[&(2, 2, 2)]),
            Value::Datum(by_coordinate[&(1, 1, 1)]),
        ],
        &mut state,
    )
    .expect("indexed block should execute");
    let Value::List(list) = result else {
        panic!("block should return a list");
    };
    let values = state
        .heap()
        .list(list)
        .expect("block list should be live")
        .positions()
        .map(|(_, value)| value.clone())
        .collect::<Vec<_>>();
    let expected = [
        (1, 1, 1),
        (2, 1, 1),
        (1, 2, 1),
        (2, 2, 1),
        (1, 1, 2),
        (2, 1, 2),
        (1, 2, 2),
        (2, 2, 2),
    ]
    .map(|coordinate| Value::Datum(by_coordinate[&coordinate]));
    assert_eq!(values, expected);
}

#[test]
fn prepare_iteration_only_reuses_an_immediately_fresh_block_list() {
    let source = parse(
            "/proc/direct()\n\tfor(var/turf/T in block(1, 1, 1, 2, 2, 1))\n\t\t. += 1\n/proc/aliased()\n\tvar/list/tiles = block(1, 1, 1, 2, 2, 1)\n\tfor(var/turf/T in tiles)\n\t\t. += 1\n",
        )
        .expect("block iteration source should parse");
    let module = compile_module(&source.definitions).expect("block loops should compile");
    let direct = module.procedures[module.procedure_id("/proc/direct").unwrap().index()].as_ref();
    let aliased = module.procedures[module.procedure_id("/proc/aliased").unwrap().index()].as_ref();
    let direct_prepare = direct
        .instructions
        .iter()
        .position(|i| matches!(i, Instruction::PrepareIteration))
        .unwrap();
    let aliased_prepare = aliased
        .instructions
        .iter()
        .position(|i| matches!(i, Instruction::PrepareIteration))
        .unwrap();
    assert!(prepare_iteration_consumes_fresh_block(
        direct,
        direct_prepare
    ));
    assert!(!prepare_iteration_consumes_fresh_block(
        aliased,
        aliased_prepare
    ));

    let mut bypass = direct.clone();
    bypass.instructions.push(Instruction::Jump(direct_prepare));
    assert!(!prepare_iteration_consumes_fresh_block(
        &bypass,
        direct_prepare
    ));
}

#[test]
fn aliased_for_in_list_keeps_snapshot_order_under_mutation() {
    let source = parse(
            "/proc/run()\n\tvar/list/items = list(1, 2, 3)\n\tvar/total = 0\n\tfor(var/item in items)\n\t\ttotal += item\n\t\titems.Cut(1, 2)\n\treturn total\n",
        )
        .expect("mutation loop should parse");
    let module = compile_module(&source.definitions).expect("mutation loop should compile");
    let entry = module.procedure_id("/proc/run").expect("run");
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(6.0)));
}

#[test]
#[ignore = "release-only microbenchmark; run explicitly"]
fn benchmark_fresh_block_iteration_snapshot_elision_65025_entries() {
    const ENTRIES: usize = 65_025;
    // `DmList` snapshots are copy-on-write, so use enough iterations to
    // measure the identity allocation/Arc churn instead of timer noise.
    const ROUNDS: usize = 100_000;
    let mut state = ExecutionState::new();
    let source = state.heap_mut().allocate_list();
    for value in 0..ENTRIES {
        state
            .heap_mut()
            .list_mut(source)
            .expect("benchmark list")
            .add(Value::number(value as f32));
    }

    let baseline_start = Instant::now();
    for _ in 0..ROUNDS {
        let snapshot = state.heap_mut().copy_list(source).expect("snapshot");
        std::hint::black_box(snapshot);
    }
    let baseline = baseline_start.elapsed();
    let elided_start = Instant::now();
    for _ in 0..ROUNDS {
        std::hint::black_box(source);
    }
    let elided = elided_start.elapsed();
    eprintln!(
        "fresh block PrepareIteration, {ENTRIES} entries x {ROUNDS}: snapshot={baseline:?} elided={elided:?}"
    );
    assert!(elided < baseline, "snapshot elision must remain measurable");
}

#[test]
fn random_builtins_are_deterministic_and_respect_their_bounds() {
    let source = parse(
            "/proc/unit()\n\treturn rand()\n/proc/range()\n\treturn rand(4, 6)\n/proc/reversed()\n\treturn rand(10, 1)\n/proc/chance()\n\treturn prob(100)\n",
        )
                .expect("source should parse");
    let module = compile_module(&source.definitions).expect("random builtins should compile");
    let range = module
        .procedure_id("/proc/range")
        .expect("range should exist");
    let unit = module
        .procedure_id("/proc/unit")
        .expect("unit should exist");
    let reversed = module
        .procedure_id("/proc/reversed")
        .expect("reversed range should exist");
    let chance = module
        .procedure_id("/proc/chance")
        .expect("chance should exist");
    let first = execute_module(&module, range, &[]).expect("rand should execute");
    let second = execute_module(&module, range, &[]).expect("fresh states should reproduce rand");
    assert_eq!(first, second);
    assert!(matches!(first.as_number(), Some(value) if (4.0..=6.0).contains(&value)));
    let unit_value = execute_module(&module, unit, &[]).expect("rand() should execute");
    assert!(
        matches!(unit_value.as_number(), Some(value) if (0.0..1.0).contains(&value)),
        "rand() returned {unit_value}"
    );
    assert_eq!(execute_module(&module, chance, &[]), Ok(Value::number(1.0)));
    let reversed_value =
        execute_module(&module, reversed, &[]).expect("BYOND swaps reversed rand bounds");
    assert!(
        matches!(reversed_value.as_number(), Some(value) if (1.0..=10.0).contains(&value)),
        "reversed rand returned {reversed_value}",
    );
}

#[test]
fn rand_seed_resets_the_stream_consumed_by_random_builtins() {
    let source = parse(
            "/proc/seeded(seed)\n\trand_seed(seed)\n\treturn rand(1, 1000000) * 100 + pick(10, 20, 30) + prob(50)\n",
        )
        .expect("rand_seed source should parse");
    let module = compile_module(&source.definitions).expect("rand_seed should compile");
    let entry = module.procedure_id("/proc/seeded").expect("seeded proc");
    let mut state = ExecutionState::new();
    let first = execute_module_in_state(&module, entry, &[Value::number(29051994.0)], &mut state)
        .expect("first seeded sequence");
    let repeated =
        execute_module_in_state(&module, entry, &[Value::number(29051994.0)], &mut state)
            .expect("repeated seeded sequence");
    assert_eq!(first, repeated, "reseeding must reproduce the whole stream");
}

#[test]
fn roll_supports_numeric_and_encoded_dice_forms() {
    let source = dm_syntax::parse(
        "/proc/numeric()\n\treturn roll(3, 6)\n/proc/encoded()\n\treturn roll(\"2d4+5\")\n",
    )
    .expect("dice source should parse");
    let module = compile_module(&source.definitions).expect("roll should compile");
    let numeric = execute_module(&module, module.procedure_id("/proc/numeric").unwrap(), &[])
        .expect("numeric dice should execute");
    let encoded = execute_module(&module, module.procedure_id("/proc/encoded").unwrap(), &[])
        .expect("encoded dice should execute");
    assert!(
        numeric
            .as_number()
            .is_some_and(|value| (3.0..=18.0).contains(&value))
    );
    assert!(
        encoded
            .as_number()
            .is_some_and(|value| (7.0..=13.0).contains(&value))
    );
}

#[test]
fn round_builtin_preserves_byond_floor_and_nearest_multiple_forms() {
    let source = parse(
            "/proc/floor_form()\n\treturn round(-1.45)\n/proc/nearest()\n\treturn round(1.99, 1)\n/proc/step()\n\treturn round(1.45, 1.5)\n/proc/negative_tie()\n\treturn round(-1.5, 1)\n/proc/zero_multiple()\n\treturn round(-1.45, 0)\n/proc/negative_multiple()\n\treturn round(2.2, -0.5)\n",
        )
        .expect("round builtin source should parse");
    let module = compile_module(&source.definitions).expect("round builtin should compile");
    for (path, expected) in [
        ("/proc/floor_form", -2.0),
        ("/proc/nearest", 2.0),
        ("/proc/step", 1.5),
        ("/proc/negative_tie", -1.0),
        ("/proc/zero_multiple", -2.0),
        ("/proc/negative_multiple", 2.0),
    ] {
        let procedure = module
            .procedure_id(path)
            .expect("round procedure should exist");
        assert_eq!(
            execute_module(&module, procedure, &[]),
            Ok(Value::number(expected))
        );
    }
    assert!(module.procedures.iter().any(|program| {
        program
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::Round { .. }))
    }));
}

#[test]
fn calls_accept_a_trailing_argument_separator() {
    let source = parse(
            "/proc/entry()\n\treturn helper(3, 4,)\n/proc/helper(first, second)\n\treturn first * 10 + second\n",
        )
        .expect("source should parse");
    let module = compile_module(&source.definitions)
        .expect("a call with a trailing separator should compile");
    let entry = module
        .procedure_id("/proc/entry")
        .expect("entry procedure should resolve");

    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(34.0)));
}

#[test]
fn calls_accept_omitted_positional_arguments() {
    let source = parse(
            "/proc/entry()\n\treturn helper(3,, 4)\n/proc/helper(first, omitted, third)\n\treturn first * 10 + third + isnull(omitted)\n",
        )
        .expect("source should parse");
    let module =
        compile_module(&source.definitions).expect("interior omitted arguments should compile");
    let entry = module.procedure_id("/proc/entry").unwrap();
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(35.0)));
}

#[test]
fn expression_produced_procedure_selectors_are_invocable() {
    let source = parse(
            "/proc/entry()\n\treturn selector()(4)\n/proc/selector()\n\treturn \"/proc/helper\"\n/proc/helper(value)\n\treturn value + 3\n",
        )
        .expect("source should parse");
    let module = compile_module(&source.definitions)
        .expect("a procedure selector returned by an expression should compile");
    let entry = module.procedure_id("/proc/entry").unwrap();
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(7.0)));
}

#[test]
fn call_ext_retains_both_selectors_and_call_arguments() {
    let source = parse("/proc/entry()\n\treturn call_ext(\"bridge.dll\", \"run\")(1, \"two\")\n")
        .expect("call_ext source should parse");
    let program = compile_procedure(&source.definitions[0])
        .expect("call_ext selector and invocation should compile");
    assert!(
        program.instructions.iter().any(|instruction| matches!(
            instruction,
            Instruction::ExternalCall { argument_count: 2 }
        ))
    );
    let error = execute(&program, &[]).expect_err("headless execution has no host bridge");
    assert!(error.message.contains("bridge.dll"));
    assert!(error.message.contains("run"));
    assert!(error.message.contains("installed host bridge"));
}

#[test]
fn special_result_supports_indexed_assignment() {
    let source =
        parse("/proc/entry()\n\t. = list()\n\t.[\"answer\"] = 42\n\treturn .[\"answer\"]\n")
            .expect("source should parse");
    let program = compile_procedure(&source.definitions[0])
        .expect("indexed special-result assignment should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::number(42.0)));
}

#[test]
fn dynamic_call_dispatches_receiver_methods_from_text_and_type_path_selectors() {
    let source = parse(
            "/datum/receiver/proc/entry(selector)\n\treturn call(src, selector)(4)\n/datum/receiver/proc/run(value)\n\treturn src.base + value\n",
        )
        .unwrap();
    let module = compile_module_specs(&[
        ProcedureSpec {
            path: "/datum/receiver/proc/entry@0".to_owned(),
            definition: &source.definitions[0],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
        ProcedureSpec {
            path: "/datum/receiver/proc/run@0".to_owned(),
            definition: &source.definitions[1],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
    ])
    .unwrap();
    let entry = module.procedure_id_at(0).unwrap();
    let mut state = ExecutionState::new();
    let receiver = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/receiver").unwrap());
    state
        .heap_mut()
        .set_datum_field(receiver, field("base"), Value::number(3.0))
        .unwrap();
    let context = ExecutionContext::new(Value::Datum(receiver), Value::Null);

    assert_eq!(
        execute_module_in_context(
            &module,
            entry,
            &[Value::Text("run".into())],
            &mut state,
            &context,
        ),
        Ok(Value::number(7.0))
    );
    assert_eq!(
        execute_module_in_context(
            &module,
            entry,
            &[Value::TypePath(
                TypePath::parse("/datum/receiver/proc/run").unwrap(),
            )],
            &mut state,
            &context,
        ),
        Ok(Value::number(7.0))
    );
}

#[test]
fn dynamic_client_members_resolve_verbs_without_basename_ambiguity() {
    let source = parse(
            "/client/proc/entry(selector)\n\treturn call(src, selector)()\n/client/proc/refresh_tgui()\n\treturn 1\n/client/verb/refresh_tgui()\n\treturn 2\n",
        )
        .expect("client proc and verb fixture should parse");
    let paths = [
        "/client/proc/entry@0",
        "/client/proc/refresh_tgui@0",
        "/client/verb/refresh_tgui@0",
    ];
    let specs = source
        .definitions
        .iter()
        .zip(paths)
        .map(|(definition, path)| ProcedureSpec {
            path: path.to_owned(),
            definition,
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        })
        .collect::<Vec<_>>();
    let module = compile_module_specs(&specs).expect("client member fixture should compile");
    let mut state = ExecutionState::new();
    let client = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/client").unwrap());
    let context = ExecutionContext::new(Value::Datum(client), Value::Null);
    let entry = module.procedure_id_at(0).unwrap();

    assert_eq!(
        execute_module_in_context(
            &module,
            entry,
            &[Value::text("refresh_tgui")],
            &mut state,
            &context,
        ),
        Ok(Value::number(1.0)),
        "bare selectors deterministically prefer proc over verb"
    );
    assert_eq!(
        execute_module_in_context(
            &module,
            entry,
            &[Value::text("verb/refresh_tgui")],
            &mut state,
            &context,
        ),
        Ok(Value::number(2.0))
    );
    assert_eq!(
        execute_module_in_context(
            &module,
            entry,
            &[Value::TypePath(
                TypePath::parse("/client/verb/refresh_tgui").unwrap(),
            )],
            &mut state,
            &context,
        ),
        Ok(Value::number(2.0))
    );
}

#[test]
fn bare_dynamic_member_prefers_inherited_proc_before_child_verb() {
    let source = parse(
            "/datum/base/proc/run()\n\treturn 1\n/datum/child/proc/entry(selector)\n\treturn call(src, selector)()\n/datum/child/verb/run()\n\treturn 2\n",
        )
        .expect("inherited proc and child verb fixture should parse");
    let module = compile_module(&source.definitions).expect("member fixture should compile");
    let child = TypePath::parse("/datum/child").unwrap();
    let mut state = ExecutionState::new();
    state.set_type_parents(BTreeMap::from([(
        child.clone(),
        Some(TypePath::parse("/datum/base").unwrap()),
    )]));
    let receiver = state.heap_mut().allocate_datum(child);
    let context = ExecutionContext::new(Value::Datum(receiver), Value::Null);
    let entry = module
        .procedure_id("/datum/child/proc/entry")
        .expect("entry should resolve");

    assert_eq!(
        execute_module_in_context(&module, entry, &[Value::text("run")], &mut state, &context,),
        Ok(Value::number(1.0)),
        "an unqualified call searches the complete proc hierarchy first"
    );
    assert_eq!(
        execute_module_in_context(
            &module,
            entry,
            &[Value::text("verb/run")],
            &mut state,
            &context,
        ),
        Ok(Value::number(2.0)),
        "an explicit verb selector still reaches the child verb"
    );
}

#[test]
fn constant_member_selectors_are_embedded_without_changing_dynamic_call_semantics() {
    let source = parse(concat!(
        "/datum/receiver/proc/static_entry()\n\treturn src.run(4)\n",
        "/datum/receiver/proc/dynamic_entry(selector)\n\treturn call(src, selector)(4)\n",
        "/datum/receiver/proc/run(value)\n\treturn src.base + value\n",
    ))
    .expect("member-call fixture should parse");
    let module = compile_module(&source.definitions).expect("member calls should compile");
    let static_entry = module
        .procedure(
            module
                .procedure_id("/datum/receiver/proc/static_entry")
                .unwrap(),
        )
        .unwrap();
    assert!(static_entry.instructions.iter().any(|instruction| matches!(
        instruction,
        Instruction::CallDynamic {
            static_selector: Some(selector),
            ..
        } if selector == "run"
    )));
    assert!(!static_entry.instructions.iter().any(
            |instruction| matches!(instruction, Instruction::PushText(selector) if selector.as_ref() == "run")
        ));
    let dynamic_entry = module
        .procedure(
            module
                .procedure_id("/datum/receiver/proc/dynamic_entry")
                .unwrap(),
        )
        .unwrap();
    assert!(
        dynamic_entry
            .instructions
            .iter()
            .any(|instruction| matches!(
                instruction,
                Instruction::CallDynamic {
                    static_selector: None,
                    ..
                }
            ))
    );

    let mut state = ExecutionState::new();
    let receiver = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/receiver").unwrap());
    state
        .heap_mut()
        .set_datum_field(receiver, field("base"), Value::number(3.0))
        .unwrap();
    let context = ExecutionContext::new(Value::Datum(receiver), Value::Null);
    assert_eq!(
        execute_module_in_context(
            &module,
            module
                .procedure_id("/datum/receiver/proc/static_entry")
                .unwrap(),
            &[],
            &mut state,
            &context,
        ),
        Ok(Value::number(7.0))
    );
    assert_eq!(
        execute_module_in_context(
            &module,
            module
                .procedure_id("/datum/receiver/proc/dynamic_entry")
                .unwrap(),
            &[Value::text("run")],
            &mut state,
            &context,
        ),
        Ok(Value::number(7.0))
    );
}

#[test]
fn constant_text_pushes_share_the_immutable_bytecode_allocation() {
    let source = parse("/proc/run()\n\treturn \"atom_after_successful_initialize\"\n")
        .expect("constant text fixture should parse");
    let program =
        compile_procedure(&source.definitions[0]).expect("constant text fixture should compile");
    let constant = program
        .instructions
        .iter()
        .find_map(|instruction| match instruction {
            Instruction::PushText(text) => Some(text),
            _ => None,
        })
        .expect("constant text instruction");
    let first = execute(&program, &[]).expect("first execution");
    let second = execute(&program, &[]).expect("second execution");
    let (Value::Text(first), Value::Text(second)) = (first, second) else {
        panic!("constant text execution should return text");
    };
    assert_eq!(first.as_ref(), "atom_after_successful_initialize");
    assert!(Arc::ptr_eq(constant, &first));
    assert!(Arc::ptr_eq(&first, &second));
}

#[test]
fn dynamic_call_follows_runtime_parent_type_outside_lexical_path() {
    let source = parse(concat!(
        "/atom/proc/add_debris_element()\n\treturn 17\n",
        "/obj/effect/statclick/ticket_list/proc/Initialize()\n",
        "\treturn call(src, \"add_debris_element\")()\n",
    ))
    .unwrap();
    let module = compile_module(&source.definitions).unwrap();
    let initialize = module
        .procedure_id("/obj/effect/statclick/ticket_list/proc/Initialize")
        .unwrap();
    let ticket = TypePath::parse("/obj/effect/statclick/ticket_list").unwrap();
    let mut state = ExecutionState::new();
    state.set_type_parents(BTreeMap::from([
        (
            ticket.clone(),
            Some(TypePath::parse("/obj/effect/statclick").unwrap()),
        ),
        (
            TypePath::parse("/obj/effect/statclick").unwrap(),
            Some(TypePath::parse("/obj/effect").unwrap()),
        ),
        (
            TypePath::parse("/obj/effect").unwrap(),
            Some(TypePath::parse("/obj").unwrap()),
        ),
        (
            TypePath::parse("/obj").unwrap(),
            Some(TypePath::parse("/atom").unwrap()),
        ),
        (
            TypePath::parse("/atom").unwrap(),
            Some(TypePath::parse("/datum").unwrap()),
        ),
    ]));
    let receiver = state.heap_mut().allocate_datum(ticket);
    assert_eq!(
        execute_module_in_context(
            &module,
            initialize,
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(receiver), Value::Null),
        ),
        Ok(Value::number(17.0)),
        "ticket-list Initialize must reach /atom through /obj's actual parent_type",
    );
}

#[test]
fn dynamic_lookup_cost_depends_on_parent_depth_not_symbol_count() {
    let mut text = String::new();
    for index in 0..2_000 {
        use std::fmt::Write as _;
        writeln!(text, "/datum/noise{index}/proc/cold()\n\treturn {index}").unwrap();
    }
    text.push_str(concat!(
        "/datum/base/proc/run()\n\treturn 1\n",
        "/datum/base/proc/run()\n\treturn 2\n",
        "/datum/child/proc/entry()\n",
        "\t. = 0\n",
        "\tfor(var/i in 1 to 100)\n",
        "\t\t. += call(src, \"run\")()\n",
        "\treturn .\n",
    ));
    let source = parse(&text).expect("large dynamic-dispatch fixture should parse");
    let mut run_reopening = 0usize;
    let specs = source
        .definitions
        .iter()
        .map(|definition| {
            let base = definition.path.to_string();
            let path = if base == "/datum/base/proc/run" {
                let path = format!("{base}@{run_reopening}");
                run_reopening += 1;
                path
            } else {
                format!("{base}@0")
            };
            ProcedureSpec {
                path,
                definition,
                parent: None,
                static_calls: BTreeMap::new(),
                src_fields: BTreeMap::new(),
                global_fields: BTreeMap::new(),
            }
        })
        .collect::<Vec<_>>();
    let module = compile_module_specs(&specs).expect("large symbolic module should link");
    let entry = module
        .procedure_id("/datum/child/proc/entry@0")
        .expect("entry should retain its stable id");
    let child = TypePath::parse("/datum/child").unwrap();
    let mut state = ExecutionState::new();
    state.set_type_parents(BTreeMap::from([(
        child.clone(),
        Some(TypePath::parse("/datum/base").unwrap()),
    )]));
    let receiver = state.heap_mut().allocate_datum(child);

    crate::DYNAMIC_LOOKUP_PROBES.set(0);
    assert_eq!(
        execute_module_in_context(
            &module,
            entry,
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(receiver), Value::Null),
        ),
        Ok(Value::number(200.0)),
        "the latest reopening must win every dynamic dispatch",
    );
    assert_eq!(
        crate::DYNAMIC_LOOKUP_PROBES.get(),
        2,
        "the first call should probe child and parent; the next 99 calls must use the receiver cache",
    );
}

#[test]
fn dynamic_receiver_cache_retains_misses_and_invalidates_with_parent_metadata() {
    let source = parse("/datum/base/proc/run()\n\treturn 7\n").unwrap();
    let module = compile_module(&source.definitions).unwrap();
    let child = TypePath::parse("/datum/child").unwrap();
    let base = TypePath::parse("/datum/base").unwrap();
    let mut state = ExecutionState::new();
    state.set_type_parents(BTreeMap::from([(child.clone(), None)]));
    let receiver = state.heap_mut().allocate_datum(child.clone());
    let receiver = Value::Datum(receiver);
    let selector = Value::text("run");
    let context = ExecutionContext::default();

    crate::DYNAMIC_LOOKUP_PROBES.set(0);
    for _ in 0..2 {
        assert!(
            crate::dynamic_call_target(&module, &mut state, &receiver, &selector, &context, false,)
                .is_err()
        );
    }
    assert_eq!(
        crate::DYNAMIC_LOOKUP_PROBES.get(),
        4,
        "the first miss probes proc and verb once each across the child and lexical datum owners; the second lookup must use the cached miss",
    );

    state.set_type_parents(BTreeMap::from([(child, Some(base))]));
    let (target, _) =
        crate::dynamic_call_target(&module, &mut state, &receiver, &selector, &context, false)
            .expect("replacing parent metadata must invalidate the cached miss");
    assert_eq!(module.procedure_path(target), Some("/datum/base/proc/run"));
    assert_eq!(
        crate::DYNAMIC_LOOKUP_PROBES.get(),
        6,
        "invalidating parent metadata must add exactly the child/base proc probes before resolving",
    );
}

#[test]
fn dynamic_callsite_cache_retains_receiver_identity_and_invalidates_with_parents() {
    let source = parse("/datum/base/proc/run()\n\treturn 7\n").unwrap();
    let module = compile_module(&source.definitions).unwrap();
    let base = TypePath::parse("/datum/base").unwrap();
    let child = TypePath::parse("/datum/child").unwrap();
    let mut state = ExecutionState::new();
    state.set_type_parents(BTreeMap::from([(child.clone(), Some(base.clone()))]));
    let receiver = Value::Datum(state.heap_mut().allocate_datum(child.clone()));
    let context = ExecutionContext::default();
    let caller = module.procedure_id_at(0).unwrap();

    for _ in 0..2 {
        let (target, _) = crate::dynamic_call_target_named_at_callsite(
            &module,
            &mut state,
            &receiver,
            "run",
            &context,
            false,
            Some((caller, 17)),
        )
        .unwrap();
        assert_eq!(module.procedure_path(target), Some("/datum/base/proc/run"));
    }
    assert_eq!(state.dynamic_callsite_targets.len(), 1);
    let retained_path = &state
        .dynamic_callsite_targets
        .values()
        .next()
        .expect("callsite entry")
        .0;
    assert_eq!(retained_path, &child);

    state.set_type_parents(BTreeMap::from([(child, Some(base))]));
    assert!(state.dynamic_callsite_targets.is_empty());
}

#[test]
#[ignore = "focused dynamic member-call cache-hit microbenchmark"]
fn dynamic_member_named_cache_hit_avoids_selector_allocation_benchmark() {
    const ITERATIONS: usize = 2_000_000;
    let source = parse("/datum/example/proc/run()\n\treturn 1\n").unwrap();
    let module = compile_module(&source.definitions).unwrap();
    let path = TypePath::parse("/datum/example").unwrap();
    let mut state = ExecutionState::new();
    state.set_type_parents(BTreeMap::from([(path.clone(), None)]));
    let receiver = Value::Datum(state.heap_mut().allocate_datum(path));
    let context = ExecutionContext::default();

    crate::dynamic_call_target_named(&module, &mut state, &receiver, "run", &context, false)
        .unwrap();

    let allocated_started = std::time::Instant::now();
    for _ in 0..ITERATIONS {
        let selector = std::hint::black_box(Value::text("run"));
        std::hint::black_box(
            crate::dynamic_call_target(&module, &mut state, &receiver, &selector, &context, false)
                .unwrap(),
        );
    }
    let allocated = allocated_started.elapsed();

    let borrowed_started = std::time::Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(
            crate::dynamic_call_target_named(
                &module,
                &mut state,
                &receiver,
                std::hint::black_box("run"),
                &context,
                false,
            )
            .unwrap(),
        );
    }
    let borrowed = borrowed_started.elapsed();

    let callsite_started = std::time::Instant::now();
    let caller = module.procedure_id_at(0).unwrap();
    for _ in 0..ITERATIONS {
        std::hint::black_box(
            crate::dynamic_call_target_named_at_callsite(
                &module,
                &mut state,
                &receiver,
                std::hint::black_box("run"),
                &context,
                false,
                Some((caller, 3)),
            )
            .unwrap(),
        );
    }
    let callsite = callsite_started.elapsed();

    eprintln!(
        "dynamic member cache hit allocated_selector_ms={} borrowed_selector_ms={} callsite_ms={} borrowed_to_callsite={:.2}x",
        allocated.as_millis(),
        borrowed.as_millis(),
        callsite.as_millis(),
        borrowed.as_secs_f64() / callsite.as_secs_f64(),
    );
}

#[test]
fn atoms_profile_path_activation_is_exact_across_reopening_suffixes() {
    assert!(crate::is_atoms_initialize_path(
        "/datum/controller/subsystem/atoms/proc/Initialize"
    ));
    assert!(crate::is_atoms_initialize_path(
        "/datum/controller/subsystem/atoms/proc/Initialize@3"
    ));
    assert!(!crate::is_atoms_initialize_path(
        "/datum/controller/subsystem/atoms/proc/InitializeAtoms@0"
    ));
    assert!(!crate::is_atoms_initialize_path(
        "/datum/controller/subsystem/assets/proc/Initialize@0"
    ));
    assert!(crate::is_subsystem_initialize_path(
        "/datum/controller/subsystem/lighting/proc/Initialize"
    ));
    assert!(crate::is_subsystem_initialize_path(
        "/datum/controller/subsystem/processing/atoms/proc/Initialize@3"
    ));
    assert!(!crate::is_subsystem_initialize_path(
        "/datum/controller/subsystem/lighting/proc/fire"
    ));
    assert!(!crate::is_subsystem_initialize_path(
        "/datum/controller/master/proc/Initialize"
    ));
}

#[test]
fn atoms_profile_survives_step_slices_counts_root_once_and_finishes_with_it() {
    let source = parse(concat!(
        "/datum/controller/subsystem/atoms/proc/Initialize()\n",
        "\tvar/total = 0\n",
        "\tfor(var/i in 1 to 8)\n",
        "\t\ttotal += i\n",
        "\treturn total\n",
    ))
    .unwrap();
    let module = compile_module(&source.definitions).unwrap();
    let entry = module
        .procedure_id(crate::ATOMS_INITIALIZE_PATH)
        .expect("SSatoms entry");
    let program = module.procedure(entry).unwrap();
    let mut frame = crate::make_frame(entry, program, &[], &ExecutionContext::default());
    frame.atoms_profile_root = true;
    let mut frames = vec![frame];
    let mut state = ExecutionState::new();
    state.atoms_profile = Some(crate::AtomsProfile {
        started: Instant::now(),
        last_snapshot: Instant::now(),
        startup_root: None,
        total_instructions: 0,
        instruction_categories: None,
        samples: HashMap::new(),
        wall_sample_nanos: HashMap::new(),
        frame_entries: HashMap::new(),
        paths: HashMap::new(),
        instruction_samples: HashMap::new(),
        instruction_wall_nanos: HashMap::new(),
        instruction_labels: HashMap::new(),
    });
    let limits = ExecutionLimits {
        max_call_depth: 64,
        max_steps: 3,
        wall_clock_budget: None,
    };
    let mut slices = 0;
    loop {
        match crate::run_frames(
            &module,
            frames,
            limits,
            crate::StepBudgetBehavior::YieldScheduledContinuation,
            &mut state,
        )
        .unwrap()
        {
            crate::FrameRunOutcome::Yielded {
                frames: continuation,
                ..
            } => {
                slices += 1;
                let profile = state
                    .atoms_profile
                    .as_ref()
                    .expect("profile survives yield");
                let key = crate::AtomsProfileProcedure {
                    module_identity: module.identity.0,
                    procedure: entry,
                };
                assert_eq!(profile.frame_entries.get(&key), Some(&1));
                assert_eq!(profile.total_instructions, slices * limits.max_steps);
                frames = continuation;
            }
            crate::FrameRunOutcome::Complete(value) => {
                assert_eq!(value, Value::number(36.0));
                break;
            }
            crate::FrameRunOutcome::Prompted { .. } => {
                panic!("numeric profiler fixture cannot prompt")
            }
        }
    }
    assert!(slices > 1);
    assert!(
        state.atoms_profile.is_none(),
        "returning the marked root must finish and remove its profiler"
    );
}

#[test]
fn tgm_profile_survives_yields_counts_nested_calls_and_finishes_with_root() {
    let source = parse(
            "/proc/child(value)\n\treturn value + 1\n/proc/root()\n\tvar/total = 0\n\tfor(var/i in 1 to 8)\n\t\ttotal += child(i)\n\treturn total\n",
        )
        .unwrap();
    let module = compile_module(&source.definitions).unwrap();
    let root = module.procedure_id("/proc/root").unwrap();
    let child = module.procedure_id("/proc/child").unwrap();
    let mut frame = crate::make_frame(
        root,
        module.procedure(root).unwrap(),
        &[],
        &ExecutionContext::default(),
    );
    frame.tgm_profile_root = true;
    let mut frames = vec![frame];
    let mut state = ExecutionState::new();
    state.tgm_profile = Some(crate::TgmProfile {
        started: Instant::now(),
        total_instructions: 0,
        procedure_samples: HashMap::new(),
        instruction_samples: HashMap::new(),
        paths: HashMap::new(),
        instruction_labels: HashMap::new(),
    });
    let limits = ExecutionLimits {
        max_call_depth: 64,
        max_steps: 3,
        wall_clock_budget: None,
    };
    let mut saw_child = false;
    loop {
        match crate::run_frames(
            &module,
            frames,
            limits,
            crate::StepBudgetBehavior::YieldScheduledContinuation,
            &mut state,
        )
        .unwrap()
        {
            crate::FrameRunOutcome::Yielded {
                frames: continuation,
                ..
            } => {
                let profile = state.tgm_profile.as_ref().expect("profile survives yield");
                saw_child |=
                    profile
                        .procedure_samples
                        .contains_key(&crate::AtomsProfileProcedure {
                            module_identity: module.identity.0,
                            procedure: child,
                        });
                frames = continuation;
            }
            crate::FrameRunOutcome::Complete(value) => {
                assert_eq!(value, Value::number(44.0));
                break;
            }
            crate::FrameRunOutcome::Prompted { .. } => panic!("fixture cannot prompt"),
        }
    }
    assert!(saw_child, "nested child procedure must be sampled");
    assert!(state.tgm_profile.is_none(), "root return finishes profiler");
}

#[test]
fn atoms_profile_report_ranks_samples_then_entries() {
    let source = parse(concat!(
        "/proc/first()\n\treturn 1\n",
        "/proc/second()\n\treturn 2\n",
    ))
    .unwrap();
    let module = compile_module(&source.definitions).unwrap();
    let first = module.procedure_id("/proc/first").unwrap();
    let second = module.procedure_id("/proc/second").unwrap();
    let first_key = crate::AtomsProfileProcedure {
        module_identity: module.identity.0,
        procedure: first,
    };
    let second_key = crate::AtomsProfileProcedure {
        module_identity: module.identity.0,
        procedure: second,
    };
    let profile = crate::AtomsProfile {
        started: Instant::now(),
        last_snapshot: Instant::now(),
        startup_root: None,
        total_instructions: 12_345,
        instruction_categories: None,
        samples: HashMap::from([(first_key, 2), (second_key, 3)]),
        wall_sample_nanos: HashMap::new(),
        frame_entries: HashMap::from([(first_key, 20), (second_key, 1)]),
        paths: HashMap::from([
            (first_key, "/proc/first".to_owned()),
            (second_key, "/proc/second".to_owned()),
        ]),
        instruction_samples: HashMap::new(),
        instruction_wall_nanos: HashMap::new(),
        instruction_labels: HashMap::new(),
    };
    let lines = crate::atoms_profile_lines(&profile);
    assert!(lines[0].contains("total_instructions=12345 samples=5 procedures=2"));
    assert!(lines[1].contains("rank=1 samples=3 entries=1 procedure=/proc/second"));
    assert!(lines[2].contains("rank=2 samples=2 entries=20 procedure=/proc/first"));

    let startup = crate::AtomsProfile {
        startup_root: Some("/datum/controller/subsystem/lighting/proc/Initialize@2".to_owned()),
        ..profile
    };
    let lines = crate::atoms_profile_lines(&startup);
    assert!(lines[0].contains(
        "startup-profile-summary subsystem=/datum/controller/subsystem/lighting/proc/Initialize@2"
    ));
    assert!(lines[1].contains(
            "startup-profile-rank subsystem=/datum/controller/subsystem/lighting/proc/Initialize@2 rank=1"
        ));
}

#[test]
fn startup_instruction_profile_classifies_and_reports_fixed_categories() {
    assert_eq!(
        crate::startup_instruction_category(&Instruction::IndexList),
        0
    );
    assert_eq!(
        crate::startup_instruction_category(&Instruction::SetListIndex),
        1
    );
    assert_eq!(
        crate::startup_instruction_category(&Instruction::LoadField(
            FieldName::parse("name").unwrap()
        )),
        2
    );
    assert_eq!(crate::startup_instruction_category(&Instruction::Return), 4);
    assert_eq!(
        crate::startup_instruction_category(&Instruction::Jump(0)),
        5
    );

    let started = Instant::now();
    let profile = crate::AtomsProfile {
        started,
        last_snapshot: started,
        startup_root: Some("/datum/controller/subsystem/mapping/proc/Initialize@1".to_owned()),
        total_instructions: 28,
        instruction_categories: Some([10, 3, 4, 2, 5, 1, 3]),
        samples: HashMap::new(),
        wall_sample_nanos: HashMap::new(),
        frame_entries: HashMap::new(),
        paths: HashMap::new(),
        instruction_samples: HashMap::new(),
        instruction_wall_nanos: HashMap::new(),
        instruction_labels: HashMap::new(),
    };
    let lines = crate::atoms_profile_lines(&profile);
    assert_eq!(lines.len(), 2);
    assert!(
        lines[1].contains(
            "list_read=10 list_write=3 field_read=4 field_write=2 call=5 branch=1 other=3"
        )
    );
}

#[test]
fn atoms_profile_periodic_snapshot_preserves_cumulative_counts() {
    let started = Instant::now();
    let key = crate::AtomsProfileProcedure {
        module_identity: 7,
        procedure: crate::ProcedureId::from_index(3).expect("small procedure id is valid"),
    };
    let mut profile = crate::AtomsProfile {
        started,
        last_snapshot: started,
        startup_root: None,
        total_instructions: 12_345,
        instruction_categories: None,
        samples: HashMap::from([(key, 3)]),
        wall_sample_nanos: HashMap::new(),
        frame_entries: HashMap::from([(key, 9)]),
        paths: HashMap::from([(key, "/proc/hot".to_owned())]),
        instruction_samples: HashMap::new(),
        instruction_wall_nanos: HashMap::new(),
        instruction_labels: HashMap::new(),
    };
    assert!(
        crate::atoms_profile_snapshot_lines_if_due(
            &mut profile,
            started + std::time::Duration::from_secs(59),
            std::time::Duration::from_secs(60),
        )
        .is_none()
    );
    let lines = crate::atoms_profile_snapshot_lines_if_due(
        &mut profile,
        started + std::time::Duration::from_secs(60),
        std::time::Duration::from_secs(60),
    )
    .expect("the sixty-second boundary emits a snapshot");
    assert!(lines[0].contains("atoms-profile-snapshot"));
    assert!(lines[0].contains("total_instructions=12345 samples=3 procedures=1"));
    assert_eq!(profile.total_instructions, 12_345);
    assert_eq!(profile.samples.get(&key), Some(&3));
    assert_eq!(profile.frame_entries.get(&key), Some(&9));
    assert!(
        crate::atoms_profile_snapshot_lines_if_due(
            &mut profile,
            started + std::time::Duration::from_secs(119),
            std::time::Duration::from_secs(60),
        )
        .is_none(),
        "the reporting interval restarts without resetting profile counts"
    );
}

#[test]
fn owned_call_frame_preserves_arguments_locals_and_supplied_flags() {
    let source = parse("/proc/callee(a, b, c, ...)\n\tvar/x\n\treturn a + b + c\n").unwrap();
    let module = compile_module(&source.definitions).unwrap();
    let entry = module.procedure_id("/proc/callee").unwrap();
    let program = module.procedure(entry).unwrap();
    let context = ExecutionContext::default();
    let arguments = vec![Value::number(1.0), Value::number(2.0)];
    let borrowed = crate::make_frame(entry, program, &arguments, &context);
    let owned = crate::make_frame_owned(entry, program, arguments, &context);
    assert_eq!(owned.arguments, borrowed.arguments);
    assert_eq!(owned.locals, borrowed.locals);
    assert_eq!(owned.supplied_parameters, borrowed.supplied_parameters);
    assert_eq!(
        owned.arguments.len(),
        3,
        "omitted c is padded in implicit args"
    );
    assert_eq!(owned.arguments[2], Value::Null);
    assert_eq!(
        owned.supplied_parameters.as_slice(),
        [true, true, false, false]
    );
    assert!(!owned.locals.spilled(), "common locals must stay inline");
    assert!(
        !owned.stack.spilled(),
        "common operand stack must stay inline"
    );
    assert!(
        !owned.supplied_parameters.spilled(),
        "common supplied-parameter flags must stay inline"
    );
    assert!(
        !owned.arguments.spilled(),
        "common implicit argument vectors must stay inline"
    );

    let extras = vec![
        Value::number(1.0),
        Value::number(2.0),
        Value::number(3.0),
        Value::number(4.0),
        Value::number(5.0),
    ];
    let borrowed = crate::make_frame(entry, program, &extras, &context);
    let owned = crate::make_frame_owned(entry, program, extras, &context);
    assert_eq!(owned.arguments, borrowed.arguments);
    assert_eq!(owned.locals, borrowed.locals);
    assert_eq!(owned.supplied_parameters, borrowed.supplied_parameters);
    assert_eq!(
        owned.arguments.len(),
        5,
        "variadic extras remain in implicit args"
    );
    assert!(
        !owned.arguments.spilled(),
        "small variadic argument vectors must stay inline"
    );
}

#[test]
fn call_frame_argument_storage_spills_without_changing_dm_args() {
    let source = parse("/proc/callee(a, b, c, d, e, f, g, h, i, j)\n\treturn args.len\n").unwrap();
    let module = compile_module(&source.definitions).unwrap();
    let entry = module.procedure_id("/proc/callee").unwrap();
    let program = module.procedure(entry).unwrap();
    let arguments = (0..10)
        .map(|value| Value::number(value as f32))
        .collect::<Vec<_>>();
    let frame = crate::make_frame(entry, program, &arguments, &ExecutionContext::default());

    assert!(
        frame.arguments.spilled(),
        "large calls use the heap fallback"
    );
    assert_eq!(frame.arguments.as_slice(), arguments.as_slice());
    assert_eq!(&frame.locals[..arguments.len()], arguments.as_slice());
    assert!(
        frame.locals[arguments.len()..]
            .iter()
            .all(|value| *value == Value::Null)
    );
    assert!(frame.supplied_parameters.iter().all(|supplied| *supplied));
}

#[test]
#[ignore = "bounded release-only allocation microbenchmark"]
fn benchmark_inline_atom_call_arguments_against_heap_vec() {
    use std::hint::black_box;
    use std::time::Instant;

    const ITERATIONS: usize = 2_380_000;
    let source = parse("/proc/init_atom(atom, list/arguments)\n\treturn atom\n").unwrap();
    let module = compile_module(&source.definitions).unwrap();
    let entry = module.procedure_id("/proc/init_atom").unwrap();
    let program = module.procedure(entry).unwrap();
    let values = [Value::Null, Value::Null];

    let started = Instant::now();
    let mut inline_checksum = 0usize;
    for _ in 0..ITERATIONS {
        let mut arguments = black_box(values.as_slice())
            .iter()
            .cloned()
            .collect::<smallvec::SmallVec<[Value; 8]>>();
        arguments.resize(
            arguments.len().max(crate::declared_argument_count(program)),
            Value::Null,
        );
        inline_checksum = inline_checksum.wrapping_add(black_box(arguments).len());
    }
    let inline = started.elapsed();

    // Exact former storage strategy: each call cloned its supplied values
    // into an independently allocated Vec before padding declared args.
    let started = Instant::now();
    let mut heap_checksum = 0usize;
    for _ in 0..ITERATIONS {
        let mut arguments = black_box(values.as_slice()).to_vec();
        arguments.resize(
            arguments.len().max(crate::declared_argument_count(program)),
            Value::Null,
        );
        heap_checksum = heap_checksum.wrapping_add(black_box(arguments).len());
    }
    let heap = started.elapsed();

    assert_eq!(inline_checksum, heap_checksum);
    eprintln!(
        "atom-call-arguments iterations={ITERATIONS} inline_ms={} heap_vec_ms={} ratio={:.3}",
        inline.as_millis(),
        heap.as_millis(),
        inline.as_secs_f64() / heap.as_secs_f64(),
    );
}

#[test]
fn hascall_reflects_lists_inherited_project_methods_and_native_types() {
    let syntax = parse(concat!(
        "/datum/base/proc/DebugValue()\n",
        "\treturn 1\n",
        "/proc/check(receiver, selector)\n",
        "\treturn hascall(receiver, selector)\n",
    ))
    .unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let check = module.procedure_id("/proc/check").unwrap();
    let mut state = ExecutionState::new();
    state.set_type_parents(BTreeMap::from([
        (
            TypePath::parse("/datum/base").unwrap(),
            Some(TypePath::parse("/datum").unwrap()),
        ),
        (
            TypePath::parse("/datum/child").unwrap(),
            Some(TypePath::parse("/datum/base").unwrap()),
        ),
    ]));
    let child = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/child").unwrap());
    let list = state.heap_mut().allocate_list();
    let run = |receiver, selector, state: &mut ExecutionState| {
        execute_module_in_state(&module, check, &[receiver, selector], state)
    };

    assert_eq!(
        run(Value::List(list), Value::text("Cut"), &mut state),
        Ok(Value::number(1.0)),
        "init_special_list_names detects native list-like Cut"
    );
    assert_eq!(
        run(Value::Datum(child), Value::text("DebugValue"), &mut state),
        Ok(Value::number(1.0)),
        "_debug_variable_value sees inherited project procedures"
    );
    assert_eq!(
        run(
            Value::TypePath(TypePath::parse("/datum/child").unwrap()),
            Value::TypePath(TypePath::parse("/datum/base/proc/DebugValue").unwrap()),
            &mut state,
        ),
        Ok(Value::number(1.0)),
        "procedure-path selectors are matched by their callable name"
    );
    assert_eq!(
        run(
            Value::TypePath(TypePath::parse("/icon").unwrap()),
            Value::text("Width"),
            &mut state,
        ),
        Ok(Value::number(1.0))
    );
    assert_eq!(
        run(Value::Datum(child), Value::text("Missing"), &mut state),
        Ok(Value::number(0.0))
    );
    assert_eq!(
        run(Value::Null, Value::text("Cut"), &mut state),
        Ok(Value::number(0.0))
    );
}

#[test]
fn dynamic_call_canonicalizes_global_proc_selectors_without_double_proc_segment() {
    let source = parse(
        "/proc/entry(selector)\n\treturn call(selector)(4)\n/proc/Log(value)\n\treturn value + 3\n",
    )
    .expect("source should parse");
    let module = compile_module(&source.definitions).expect("module should compile");
    let entry = module.procedure_id("/proc/entry").unwrap();

    for selector in [
        Value::text("Log"),
        Value::text("proc/Log"),
        Value::TypePath(TypePath::parse("/proc/Log").unwrap()),
    ] {
        assert_eq!(
            execute_module(&module, entry, &[selector]),
            Ok(Value::number(7.0))
        );
    }
}

#[test]
fn dynamic_member_call_on_null_is_not_reinterpreted_as_a_global_proc() {
    let source = parse(
            "/proc/entry()\n\tvar/datum/logger = null\n\treturn logger.Log(4)\n/proc/Log(value)\n\treturn value + 3\n",
        )
        .expect("source should parse");
    let module = compile_module(&source.definitions).expect("module should compile");
    let error = execute_module(&module, module.procedure_id("/proc/entry").unwrap(), &[])
        .expect_err("null member calls must remain datum calls");
    assert!(error.message.contains("procedure on null"));
}

#[test]
fn bare_owner_proc_with_arglist_binds_current_src_not_global_namespace() {
    let source = parse(
            "/datum/log_holder/proc/init_logging()\n\tvar/list/arg_list = list(4)\n\treturn Log(arglist(arg_list))\n/datum/log_holder/proc/Log(value)\n\treturn src.base + value\n",
        )
        .expect("Monk-shaped logger source should parse");
    let module = compile_module_specs(&[
        ProcedureSpec {
            path: "/datum/log_holder/proc/init_logging@0".to_owned(),
            definition: &source.definitions[0],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
        ProcedureSpec {
            path: "/datum/log_holder/proc/Log@0".to_owned(),
            definition: &source.definitions[1],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
    ])
    .expect("bare Log call should owner-resolve");
    let mut state = ExecutionState::new();
    let logger = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/log_holder").unwrap());
    state
        .heap_mut()
        .set_datum_field(logger, field("base"), Value::number(3.0))
        .unwrap();

    assert_eq!(
        execute_module_in_context(
            &module,
            module.procedure_id_at(0).unwrap(),
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(logger), Value::Null),
        ),
        Ok(Value::number(7.0))
    );
}

#[test]
#[allow(clippy::vec_init_then_push)]
fn null_conditional_field_index_and_call_short_circuit_without_rhs_evaluation() {
    let source = parse(
            "/datum/example/proc/read(value, list/values)\n\tvar/a = value?.field\n\tvar/b = values?[bump()]\n\tvar/c = value?:take(bump())\n\tvar/d = value?.client.prefs.take(bump())\n\tvalue?.field = bump()\n\tvalues?[bump()] = bump()\n\treturn isnull(a) + isnull(b) + isnull(c) + isnull(d) + global.calls\n/datum/example/proc/take(value)\n\treturn value\n/proc/bump()\n\tglobal.calls += 1\n\treturn 1\n",
        )
        .expect("null-conditional source should parse");
    let mut specs = Vec::new();
    specs.push(ProcedureSpec {
        path: "/datum/example/proc/read@0".to_owned(),
        definition: &source.definitions[0],
        parent: None,
        static_calls: BTreeMap::from([("bump".to_owned(), 2)]),
        src_fields: BTreeMap::new(),
        global_fields: BTreeMap::from([("calls".to_owned(), field("calls"))]),
    });
    specs.push(ProcedureSpec {
        path: "/datum/example/proc/take@0".to_owned(),
        definition: source.definitions.last().expect("procedure definition"),
        parent: None,
        static_calls: BTreeMap::new(),
        src_fields: BTreeMap::new(),
        global_fields: BTreeMap::from([("calls".to_owned(), field("calls"))]),
    });
    specs.push(ProcedureSpec {
        path: "/proc/bump@0".to_owned(),
        definition: &source.definitions[2],
        parent: None,
        static_calls: BTreeMap::new(),
        src_fields: BTreeMap::new(),
        global_fields: BTreeMap::from([("calls".to_owned(), field("calls"))]),
    });
    let module = compile_module_specs(&specs).expect("null-conditional source should compile");
    let mut state = ExecutionState::new();
    state.set_global(field("calls"), Value::number(0.0));
    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id_at(0).expect("read entry"),
            &[Value::Null, Value::Null],
            &mut state,
        ),
        Ok(Value::number(4.0))
    );
    assert_eq!(state.global(&field("calls")), Some(&Value::number(0.0)));
}

#[test]
fn null_conditional_access_executes_normally_for_live_receivers() {
    let source = parse(
            "/datum/example/proc/read(list/values)\n\tvar/a = src?.field\n\tvar/b = values?[1]\n\treturn a + b\n",
        )
        .expect("live null-conditional source should parse");
    let module = compile_module_specs(&[ProcedureSpec {
        path: "/datum/example/proc/read@0".to_owned(),
        definition: &source.definitions[0],
        parent: None,
        static_calls: BTreeMap::new(),
        src_fields: BTreeMap::from([("field".to_owned(), field("field"))]),
        global_fields: BTreeMap::new(),
    }])
    .expect("live null-conditional source should compile");
    let mut state = ExecutionState::new();
    let datum = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/example").unwrap());
    state
        .heap_mut()
        .set_datum_field(datum, field("field"), Value::number(4.0))
        .unwrap();
    let list = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(list)
        .unwrap()
        .add(Value::number(5.0));
    assert_eq!(
        execute_module_in_context(
            &module,
            module.procedure_id_at(0).expect("read entry"),
            &[Value::List(list)],
            &mut state,
            &ExecutionContext::new(Value::Datum(datum), Value::Null),
        ),
        Ok(Value::number(9.0))
    );
}

#[test]
fn dotted_datum_calls_lower_to_dynamic_dispatch() {
    let source = parse(
            "/datum/receiver/proc/entry()\n\treturn src.run(4)\n/datum/receiver/proc/run(value)\n\treturn src.base + value\n",
        )
        .expect("source should parse");
    let module = compile_module_specs(&[
        ProcedureSpec {
            path: "/datum/receiver/proc/entry@0".to_owned(),
            definition: &source.definitions[0],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
        ProcedureSpec {
            path: "/datum/receiver/proc/run@0".to_owned(),
            definition: &source.definitions[1],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
    ])
    .expect("dotted datum call should compile");
    let entry = module.procedure_id_at(0).expect("entry should exist");
    let mut state = ExecutionState::new();
    let receiver = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/receiver").expect("type path"));
    state
        .heap_mut()
        .set_datum_field(receiver, field("base"), Value::number(3.0))
        .expect("datum should be live");

    assert_eq!(
        execute_module_in_context(
            &module,
            entry,
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(receiver), Value::Null),
        ),
        Ok(Value::number(7.0))
    );
}

#[test]
fn bare_src_field_dotted_call_is_a_valid_side_effect_statement() {
    let source = parse(
            "/datum/item/proc/Initialize()\n\tatom_storage.set_holdable(4)\n\treturn 9\n/datum/storage/proc/set_holdable(value)\n\tsrc.value = value\n",
        )
        .expect("source should parse");
    let module = compile_module_specs(&[
        ProcedureSpec {
            path: "/datum/item/proc/Initialize@0".to_owned(),
            definition: &source.definitions[0],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::from([("atom_storage".to_owned(), field("atom_storage"))]),
            global_fields: BTreeMap::new(),
        },
        ProcedureSpec {
            path: "/datum/storage/proc/set_holdable@0".to_owned(),
            definition: &source.definitions[1],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
    ])
    .expect("bare field dotted call should compile");
    let mut state = ExecutionState::new();
    let item = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/item").expect("item type"));
    let storage = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/storage").expect("storage type"));
    state
        .heap_mut()
        .set_datum_field(item, field("atom_storage"), Value::Datum(storage))
        .expect("item should be live");

    assert_eq!(
        execute_module_in_context(
            &module,
            module.procedure_id_at(0).expect("Initialize should exist"),
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(item), Value::Null),
        ),
        Ok(Value::number(9.0))
    );
    assert!(
        state
            .heap()
            .datum_field(storage, &field("value"))
            .expect("storage should be live")
            .semantic_eq(&Value::number(4.0))
    );
}

#[test]
fn field_errors_retain_source_mapping_for_null_missing_and_stale_receivers() {
    let syntax = parse("/proc/read()\n\treturn src.missing\n").unwrap();
    let span = syntax.definitions[0].body[0].span;
    let program = compile_procedure(&syntax.definitions[0]).unwrap();
    let mut state = ExecutionState::new();
    let null_error =
        execute_in_context(&program, &[], &mut state, &ExecutionContext::default()).unwrap_err();
    assert_eq!(null_error.message, "field read received null");
    assert_eq!(null_error.source_span, Some(span));

    let datum = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/source").unwrap());
    let context = ExecutionContext::new(Value::Datum(datum), Value::Null);
    let missing_error = execute_in_context(&program, &[], &mut state, &context).unwrap_err();
    assert_eq!(
        missing_error.message,
        "datum field FieldName(\"missing\") is absent"
    );
    assert_eq!(missing_error.source_span, Some(span));

    state.heap_mut().destroy_datum(datum).unwrap();
    let stale_error = execute_in_context(&program, &[], &mut state, &context).unwrap_err();
    assert_eq!(stale_error.message, "field read received null");
    assert_eq!(stale_error.source_span, Some(span));
}

#[test]
fn logical_assignment_short_circuits_locals_fields_and_list_entries() {
    let source = parse(
            "/datum/example/proc/run()\n\tvar/local\n\tlocal ||= 3\n\tvar/list/values = list()\n\tvalues[\"entry\"] ||= 4\n\tsrc.flag ||= 5\n\treturn local + values[\"entry\"] + src.flag\n",
        )
        .expect("logical assignment source should parse");
    let module = compile_module_specs(&[ProcedureSpec {
        path: "/datum/example/proc/run@0".to_owned(),
        definition: &source.definitions[0],
        parent: None,
        static_calls: BTreeMap::new(),
        src_fields: BTreeMap::from([("flag".to_owned(), field("flag"))]),
        global_fields: BTreeMap::new(),
    }])
    .expect("logical assignments should compile");
    let entry = module.procedure_id_at(0).expect("entry");
    let mut state = ExecutionState::new();
    let datum = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/example").unwrap());
    state
        .heap_mut()
        .set_datum_field(datum, field("flag"), Value::Null)
        .unwrap();
    assert_eq!(
        execute_module_in_context(
            &module,
            entry,
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(datum), Value::Null),
        ),
        Ok(Value::number(12.0))
    );
}

#[test]
fn logical_assignment_to_parameter_member_preserves_receiver_and_result() {
    let source = parse(
            "/datum/listener/proc/register(datum/target)\n\tvar/list/lookup = (target._listen_lookup ||= list())\n\tlookup[\"signal\"] = src\n\treturn lookup[\"signal\"] == src\n",
        )
        .expect("member logical assignment source should parse");
    let module =
        compile_module(&source.definitions).expect("member logical assignment should compile");
    let entry = module.procedure_id_at(0).expect("entry");
    let mut state = ExecutionState::new();
    let listener = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/listener").unwrap());
    let target = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/target").unwrap());
    state
        .heap_mut()
        .set_datum_field(target, field("_listen_lookup"), Value::Null)
        .unwrap();
    assert_eq!(
        execute_module_in_context(
            &module,
            entry,
            &[Value::Datum(target)],
            &mut state,
            &ExecutionContext::new(Value::Datum(listener), Value::Null),
        ),
        Ok(Value::number(1.0))
    );
    assert!(matches!(
        state.heap().datum_field(target, &field("_listen_lookup")),
        Ok(Value::List(_))
    ));
}

#[test]
fn logical_assignment_to_bare_src_field_preserves_receiver_and_result() {
    let source = parse(
            "/datum/listener/proc/register(datum/target)\n\tvar/list/procs = (_signal_procs ||= list())\n\tvar/list/target_procs = (procs[target] ||= list())\n\tvar/list/lookup = (target._listen_lookup ||= list())\n\tlookup[\"signal\"] = src\n\treturn target_procs.len + (lookup[\"signal\"] == src)\n",
        )
        .expect("RegisterSignal-shaped source should parse");
    let module = compile_module_specs(&[ProcedureSpec {
        path: "/datum/listener/proc/register@0".to_owned(),
        definition: &source.definitions[0],
        parent: None,
        static_calls: BTreeMap::new(),
        src_fields: BTreeMap::from([("_signal_procs".to_owned(), field("_signal_procs"))]),
        global_fields: BTreeMap::new(),
    }])
    .expect("RegisterSignal-shaped assignment should compile");
    let entry = module.procedure_id_at(0).expect("entry");
    let mut state = ExecutionState::new();
    let listener = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/listener").unwrap());
    let target = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/target").unwrap());
    state
        .heap_mut()
        .set_datum_field(listener, field("_signal_procs"), Value::Null)
        .unwrap();
    state
        .heap_mut()
        .set_datum_field(target, field("_listen_lookup"), Value::Null)
        .unwrap();
    assert_eq!(
        execute_module_in_context(
            &module,
            entry,
            &[Value::Datum(target)],
            &mut state,
            &ExecutionContext::new(Value::Datum(listener), Value::Null),
        ),
        Ok(Value::number(1.0))
    );
    assert!(matches!(
        state.heap().datum_field(listener, &field("_signal_procs")),
        Ok(Value::List(_))
    ));
}

fn production_register_signal_fixture() -> Module {
    let mut instructions = vec![Instruction::PushNull; 140];
    instructions[10] = Instruction::LoadField(field("gc_destroyed"));
    instructions[22] = Instruction::LoadDeclaredField(field("gc_destroyed"));
    instructions[26] = Instruction::LoadLocal(1);
    instructions[27] = Instruction::TypePredicate {
        kind: TypePredicateKind::IsList,
        argument_count: 1,
    };
    instructions[70] = Instruction::LogicalOrEmptyListField(field("_signal_procs"));
    instructions[74] = Instruction::LogicalOrEmptyListIndex;
    instructions[77] = Instruction::LogicalOrEmptyListField(field("_listen_lookup"));
    instructions[80] = Instruction::IndexLocalList(9);
    instructions[86] = Instruction::SetListIndex;
    instructions[111] = Instruction::IndexLocalList(10);
    instructions[114] = Instruction::TypePredicate {
        kind: TypePredicateKind::IsNull,
        argument_count: 1,
    };
    // Production signals.dm stores at 120 and jumps at 121. Reversing
    // these slots silently disabled the native path for every real call.
    instructions[120] = Instruction::SetListIndex;
    instructions[121] = Instruction::Jump(138);
    instructions[138] = Instruction::LoadResult;
    instructions[139] = Instruction::Return;
    let instruction_count = instructions.len();
    let program = Program {
        wait_for: true,
        parameter_count: 4,
        parameter_names: vec![String::new(); 4],
        verb_parameter_types: vec![VerbParameterType::Unsupported; 4],
        verb_name: None,
        local_count: 14,
        instructions,
        source_spans: vec![SourceSpan::new(0, 1); instruction_count],
    };
    Module {
        identity: next_module_identity(),
        procedures: vec![Arc::new(program)],
        paths: vec!["/datum/proc/RegisterSignal@0".to_owned()],
        names: HashMap::new(),
        dynamic_names: HashMap::new(),
        deferred: Arc::new(HashMap::new()),
        procedure_types: vec![TypePath::parse("/datum").unwrap()],
        initializer_call_names: None,
        compact_wordcode: Default::default(),
        semantic_digests: Default::default(),
    }
}

fn signal_fixture_datum(state: &mut ExecutionState) -> DatumId {
    let datum = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/listener").unwrap());
    for name in ["gc_destroyed", "_signal_procs", "_listen_lookup"] {
        state
            .heap_mut()
            .set_datum_field(datum, field(name), Value::Null)
            .unwrap();
    }
    datum
}

fn run_signal_fixture(
    module: &Module,
    state: &mut ExecutionState,
    src: DatumId,
    target: DatumId,
    signal: &Value,
    callback: Value,
    override_value: Option<Value>,
) -> Option<u64> {
    let program = module.procedure(ProcedureId(0)).unwrap();
    let mut arguments = vec![Value::Datum(target), signal.clone(), callback];
    if let Some(value) = override_value {
        arguments.push(value);
    }
    let mut frame = make_frame(
        ProcedureId(0),
        program,
        &arguments,
        &ExecutionContext::new(Value::Datum(src), Value::Null),
    );
    let result =
        try_run_register_signal_fast_path(module, ProcedureId(0), program, &mut frame, 100, state);
    if result.is_some() {
        assert_eq!(frame.instruction, 138);
    }
    result
}

fn signal_fixture_lookup(state: &ExecutionState, target: DatumId, signal: &Value) -> Value {
    let Value::List(lookup) =
        datum_field_or_shared(state, target, &field("_listen_lookup")).unwrap()
    else {
        panic!("expected listener lookup")
    };
    read_list_value(
        &state.heap,
        lookup,
        signal,
        state.is_associative_list(lookup),
    )
    .unwrap()
}

#[test]
fn register_signal_fast_path_recognizes_production_slots_and_promotes_listeners() {
    REGISTER_SIGNAL_FAST_CACHE.with(|cache| cache.borrow_mut().clear());
    let module = production_register_signal_fixture();
    let program = module.procedure(ProcedureId(0)).unwrap();
    assert!(super::compile_register_signal_trace(&module, ProcedureId(0), program).is_some());

    let mut state = ExecutionState::new();
    let target = signal_fixture_datum(&mut state);
    let first = signal_fixture_datum(&mut state);
    let second = signal_fixture_datum(&mut state);
    let third = signal_fixture_datum(&mut state);
    let signal = Value::text("prepare");
    for (listener, callback) in [
        (first, "first_callback"),
        (second, "second_callback"),
        (third, "third_callback"),
    ] {
        assert_eq!(
            run_signal_fixture(
                &module,
                &mut state,
                listener,
                target,
                &signal,
                Value::text(callback),
                None,
            ),
            Some(56)
        );
    }
    let Value::List(listeners) = signal_fixture_lookup(&state, target, &signal) else {
        panic!("multiple listeners should promote the scalar lookup to a list")
    };
    let Value::List(first_procs) =
        datum_field_or_shared(&state, first, &field("_signal_procs")).unwrap()
    else {
        panic!("listener should own a signal procedure map")
    };
    let Value::List(first_target_procs) = read_list_value(
        state.heap(),
        first_procs,
        &Value::Datum(target),
        state.is_associative_list(first_procs),
    )
    .unwrap() else {
        panic!("listener should index its callback map by target")
    };
    assert!(state.is_associative_list(first_procs));
    assert!(state.is_associative_list(first_target_procs));
    assert_eq!(
        read_list_value(
            state.heap(),
            first_target_procs,
            &signal,
            state.is_associative_list(first_target_procs),
        )
        .unwrap(),
        Value::text("first_callback")
    );
    assert_eq!(
        state
            .heap()
            .list(listeners)
            .unwrap()
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>(),
        vec![
            Value::Datum(first),
            Value::Datum(second),
            Value::Datum(third)
        ]
    );
}

#[test]
fn single_listener_signal_graph_survives_forced_quiescent_gc() {
    REGISTER_SIGNAL_FAST_CACHE.with(|cache| cache.borrow_mut().clear());
    let register = production_register_signal_fixture();
    let mut state = ExecutionState::new();
    let target = signal_fixture_datum(&mut state);
    let listener = signal_fixture_datum(&mut state);
    let signal = Value::text("prepare");
    let callback = Value::text("single_callback");

    assert_eq!(
        run_signal_fixture(
            &register,
            &mut state,
            listener,
            target,
            &signal,
            callback.clone(),
            None,
        ),
        Some(56)
    );
    assert_eq!(
        signal_fixture_lookup(&state, target, &signal),
        Value::Datum(listener),
        "one listener stays scalar in target._listen_lookup[signal]"
    );

    // At a quiescent host boundary the target is the sole external root.
    // Its scalar lookup must retain the listener datum, whose nested
    // _signal_procs[target][signal] lists in turn retain the callback.
    state.set_global(field("rooted_signal_target"), Value::Datum(target));
    let _garbage = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/unreachable_signal_fixture").unwrap());
    let compaction = state.compact_quiescent_heap();
    assert_eq!(
        signal_fixture_lookup(&state, target, &signal),
        Value::Datum(listener),
        "forced GC must traverse scalar associative-list datum values"
    );

    let reader = parse(
        "/proc/read_signal(datum/target, signal)\n\
             \tvar/datum/listener = target._listen_lookup[signal]\n\
             \treturn listener._signal_procs[target][signal]\n",
    )
    .expect("single-listener signal reader should parse");
    let reader =
        compile_module(&reader.definitions).expect("single-listener signal reader should compile");
    let entry = reader.procedure_id("/proc/read_signal").unwrap();
    assert!(matches!(
        reader.procedure(entry).unwrap().instructions.as_slice(),
        [
            Instruction::LoadLocal(0),
            Instruction::LoadDeclaredField(_),
            Instruction::LoadLocal(1),
            Instruction::IndexList,
            Instruction::StoreLocal(3),
            Instruction::LoadLocal(3),
            Instruction::LoadDeclaredField(_),
            Instruction::LoadLocal(0),
            Instruction::IndexList,
            Instruction::LoadLocal(1),
            Instruction::IndexList,
            Instruction::Return,
        ]
    ));
    assert_eq!(
        execute_module_in_state(&reader, entry, &[Value::Datum(target), signal], &mut state,),
        Ok(callback),
        "the post-GC nested read must not become null at the signal lookup"
    );
    assert!(
        compaction.reclaimed_datums > 0 || compaction.reclaimed_lists > 0,
        "the forced collection should reclaim unrelated fixture state"
    );
}

#[test]
fn register_signal_fast_path_uses_runtime_override_truthiness() {
    REGISTER_SIGNAL_FAST_CACHE.with(|cache| cache.borrow_mut().clear());
    let module = production_register_signal_fixture();
    let mut state = ExecutionState::new();
    let target = signal_fixture_datum(&mut state);
    let listener = signal_fixture_datum(&mut state);
    let signal = Value::text("prepare");

    assert_eq!(
        run_signal_fixture(
            &module,
            &mut state,
            listener,
            target,
            &signal,
            Value::text("original"),
            None,
        ),
        Some(56)
    );
    assert_eq!(
        run_signal_fixture(
            &module,
            &mut state,
            listener,
            target,
            &signal,
            Value::text("ignored"),
            Some(Value::number(0.0)),
        ),
        None,
        "a supplied false override must preserve the warning bytecode path"
    );
    assert_eq!(
        run_signal_fixture(
            &module,
            &mut state,
            listener,
            target,
            &signal,
            Value::text("replacement"),
            Some(Value::number(1.0)),
        ),
        Some(54)
    );
    let Value::List(listeners) = signal_fixture_lookup(&state, target, &signal) else {
        panic!("override should preserve the listener relationship")
    };
    assert_eq!(state.heap().list(listeners).unwrap().len(), 2);
}

#[test]
fn rooted_register_block_matches_interpreter_shape_and_materializes_lists() {
    let canonical = parse(
            "/datum/listener/proc/register(datum/target)\n\tvar/list/procs = (_signal_procs ||= list())\n\tvar/list/target_procs = (procs[target] ||= list())\n\tvar/list/lookup = (target._listen_lookup ||= list())\n\treturn target_procs\n",
        )
        .unwrap();
    let fallback = parse(
            "/datum/listener/proc/register(datum/target)\n\tvar/noop = null\n\tvar/list/procs = (_signal_procs ||= list())\n\tvar/list/target_procs = (procs[target] ||= list())\n\tvar/list/lookup = (target._listen_lookup ||= list())\n\treturn target_procs\n",
        )
        .unwrap();
    let build = |definition: &dm_syntax::Definition| {
        compile_module_specs(&[ProcedureSpec {
            path: "/datum/listener/proc/register@0".to_owned(),
            definition,
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::from([("_signal_procs".to_owned(), field("_signal_procs"))]),
            global_fields: BTreeMap::new(),
        }])
        .unwrap()
    };
    let jit_module = build(&canonical.definitions[0]);
    let interpreter_module = build(&fallback.definitions[0]);
    assert!(super::compile_rooted_list_trace(&jit_module.procedures[0]).is_some());
    assert!(super::compile_rooted_list_trace(&interpreter_module.procedures[0]).is_none());

    let run = |module: &Module| {
        let mut state = ExecutionState::new();
        let listener = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/listener").unwrap());
        let target = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/target").unwrap());
        state
            .heap_mut()
            .set_datum_field(listener, field("_signal_procs"), Value::Null)
            .unwrap();
        state
            .heap_mut()
            .set_datum_field(target, field("_listen_lookup"), Value::Null)
            .unwrap();
        let result = execute_module_in_context(
            module,
            module.procedure_id_at(0).unwrap(),
            &[Value::Datum(target)],
            &mut state,
            &ExecutionContext::new(Value::Datum(listener), Value::Null),
        )
        .unwrap();
        let Value::List(result_list) = result else {
            panic!("expected list")
        };
        assert!(state.heap().list(result_list).is_ok());
        assert!(matches!(
            state.heap().datum_field(listener, &field("_signal_procs")),
            Ok(Value::List(_))
        ));
        assert!(matches!(
            state.heap().datum_field(target, &field("_listen_lookup")),
            Ok(Value::List(_))
        ));
        state.heap().list(result_list).unwrap().len()
    };
    assert_eq!(run(&jit_module), run(&interpreter_module));
}

#[test]
#[ignore = "release-only rooted list trace microbenchmark"]
fn rooted_register_block_release_microbenchmark() {
    fn build(extra: &str) -> Module {
        let source = parse(&format!(
                "/datum/listener/proc/register(datum/target)\n{extra}\tvar/list/procs = (_signal_procs ||= list())\n\tvar/list/target_procs = (procs[target] ||= list())\n\tvar/list/lookup = (target._listen_lookup ||= list())\n\treturn target_procs\n"
            )).unwrap();
        compile_module_specs(&[ProcedureSpec {
            path: "/datum/listener/proc/register@0".to_owned(),
            definition: &source.definitions[0],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::from([("_signal_procs".to_owned(), field("_signal_procs"))]),
            global_fields: BTreeMap::new(),
        }])
        .unwrap()
    }
    fn fixture() -> (ExecutionState, dm_value::DatumId, dm_value::DatumId) {
        let mut state = ExecutionState::new();
        let src = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/listener").unwrap());
        let target = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/target").unwrap());
        state
            .heap_mut()
            .set_datum_field(src, field("_signal_procs"), Value::Null)
            .unwrap();
        state
            .heap_mut()
            .set_datum_field(target, field("_listen_lookup"), Value::Null)
            .unwrap();
        (state, src, target)
    }
    let jit = build("");
    let fallback = build("\tvar/noop = null\n");
    let (mut jit_state, jit_src, jit_target) = fixture();
    let (mut fallback_state, fallback_src, fallback_target) = fixture();
    let iterations = 1_000;
    let started = Instant::now();
    for _ in 0..iterations {
        execute_module_in_context(
            &jit,
            jit.procedure_id_at(0).unwrap(),
            &[Value::Datum(jit_target)],
            &mut jit_state,
            &ExecutionContext::new(Value::Datum(jit_src), Value::Null),
        )
        .unwrap();
    }
    let jit_elapsed = started.elapsed();
    let started = Instant::now();
    for _ in 0..iterations {
        execute_module_in_context(
            &fallback,
            fallback.procedure_id_at(0).unwrap(),
            &[Value::Datum(fallback_target)],
            &mut fallback_state,
            &ExecutionContext::new(Value::Datum(fallback_src), Value::Null),
        )
        .unwrap();
    }
    let fallback_elapsed = started.elapsed();
    eprintln!(
        "rooted-register iterations={iterations} jit={jit_elapsed:?} interpreter={fallback_elapsed:?} speedup={:.2}x",
        fallback_elapsed.as_secs_f64() / jit_elapsed.as_secs_f64()
    );
}

#[test]
fn logical_or_assignment_ast_and_empty_list_superinstruction_shapes_are_exact() {
    let parsed = parse("/proc/shape(value)\n\treturn (value ||= list())\n").unwrap();
    let tokens = &parsed.definitions[0].body[0].tokens;
    let expression = super::compile::ExpressionParser::new(&tokens[1..])
        .parse()
        .unwrap();
    assert!(matches!(
        expression,
        super::compile::Expression::LogicalOrAssignment { target, value }
            if matches!(target.as_ref(), super::compile::Expression::Local(name) if name == "value")
                && matches!(value.as_ref(), super::compile::Expression::List(entries) if entries.is_empty())
    ));

    let source =
        parse("/proc/run()\n\tvar/local\n\tlocal ||= list()\n\tcache ||= list()\n\treturn local\n")
            .unwrap();
    let module = compile_module_specs(&[ProcedureSpec {
        path: "/proc/run@0".to_owned(),
        definition: &source.definitions[0],
        parent: None,
        static_calls: BTreeMap::new(),
        src_fields: BTreeMap::new(),
        global_fields: BTreeMap::from([("cache".to_owned(), field("cache"))]),
    }])
    .unwrap();
    let instructions = &module
        .procedure(module.procedure_id_at(0).unwrap())
        .unwrap()
        .instructions;
    assert!(
        instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::LogicalOrEmptyListLocal(_)))
    );
    assert!(instructions.iter().any(|instruction| matches!(
        instruction,
        Instruction::LogicalOrEmptyListGlobal(name) if name.as_str() == "cache"
    )));

    let register = parse(
            "/datum/listener/proc/register(datum/target)\n\tvar/list/procs = (_signal_procs ||= list())\n\tvar/list/target_procs = (procs[target] ||= list())\n\treturn (target._listen_lookup ||= list())\n",
        )
        .unwrap();
    let module = compile_module_specs(&[ProcedureSpec {
        path: "/datum/listener/proc/register@0".to_owned(),
        definition: &register.definitions[0],
        parent: None,
        static_calls: BTreeMap::new(),
        src_fields: BTreeMap::from([("_signal_procs".to_owned(), field("_signal_procs"))]),
        global_fields: BTreeMap::new(),
    }])
    .unwrap();
    let instructions = &module
        .procedure(module.procedure_id_at(0).unwrap())
        .unwrap()
        .instructions;
    assert_eq!(
        instructions
            .iter()
            .filter(|instruction| matches!(instruction, Instruction::LogicalOrEmptyListField(_)))
            .count(),
        2
    );
    assert_eq!(
        instructions
            .iter()
            .filter(|instruction| matches!(instruction, Instruction::LogicalOrEmptyListIndex))
            .count(),
        1
    );
}

#[test]
fn logical_or_empty_list_truthy_skips_allocation_and_falsey_allocates_one_alias() {
    let syntax = parse("/proc/run(value)\n\treturn (value ||= list())\n").unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let entry = module.procedure_id("/proc/run").unwrap();
    assert!(
        module
            .procedure(entry)
            .unwrap()
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::LogicalOrEmptyListLocal(_)))
    );

    let mut state = ExecutionState::new();
    let existing = state.heap_mut().allocate_list();
    let before = state.heap().live_list_count();
    assert_eq!(
        execute_module_in_state(&module, entry, &[Value::List(existing)], &mut state),
        Ok(Value::List(existing))
    );
    assert_eq!(state.heap().live_list_count(), before);

    let before = state.heap().live_list_count();
    let Value::List(created) =
        execute_module_in_state(&module, entry, &[Value::Null], &mut state).unwrap()
    else {
        panic!("falsey logical assignment must return its allocated list")
    };
    assert_eq!(state.heap().live_list_count(), before + 1);
    assert!(state.heap().list(created).is_ok());
}

#[test]
fn logical_or_empty_list_preserves_pointer_aliases_and_list_copy_on_write() {
    let syntax = parse(
            "/proc/pointer()\n\tvar/value\n\tvar/pointer = &value\n\tvar/list/result = (value ||= list())\n\treturn pointer == result\n/proc/cow(list/value)\n\tvar/list/copy = value.Copy()\n\tvar/list/result = (value ||= list())\n\tresult.Add(2)\n\treturn (result == value) * 100 + value.len * 10 + copy.len\n",
        )
        .unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/pointer").unwrap(), &[]),
        Ok(Value::number(1.0))
    );

    let mut state = ExecutionState::new();
    let original = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(original)
        .unwrap()
        .add(Value::number(1.0));
    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/cow").unwrap(),
            &[Value::List(original)],
            &mut state,
        ),
        Ok(Value::number(121.0))
    );
}

#[test]
fn logical_or_empty_list_writes_through_global_and_datum_vars_proxies() {
    let source = parse(
            "/proc/global_proxy()\n\treturn (global.vars[\"cache\"] ||= list())\n/proc/datum_proxy(datum/target)\n\treturn (target.vars[\"slot\"] ||= list())\n",
        )
        .unwrap();
    let module = compile_module_with_global_fields(
        &source.definitions,
        &BTreeMap::from([("cache".to_owned(), field("cache"))]),
    )
    .unwrap();
    let mut state = ExecutionState::new();
    state.set_global(field("cache"), Value::Null);
    let global_result = execute_module_in_state(
        &module,
        module.procedure_id("/proc/global_proxy").unwrap(),
        &[],
        &mut state,
    )
    .unwrap();
    assert_eq!(state.global(&field("cache")), Some(&global_result));

    let datum = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/target").unwrap());
    state
        .heap_mut()
        .set_datum_field(datum, field("slot"), Value::Null)
        .unwrap();
    let datum_result = execute_module_in_state(
        &module,
        module.procedure_id("/proc/datum_proxy").unwrap(),
        &[Value::Datum(datum)],
        &mut state,
    )
    .unwrap();
    assert_eq!(
        state.heap().datum_field(datum, &field("slot")),
        Ok(&datum_result)
    );
}

#[test]
fn logical_or_empty_list_reads_materializing_lvalues_exactly_once() {
    let source = parse(
            "/proc/field_once(savefile/target)\n\treturn (target.dir ||= list())\n/proc/index_once(savefile/target)\n\treturn (target[\"entry\"] ||= list())\n",
        )
        .unwrap();
    let module = compile_module(&source.definitions).unwrap();

    let mut state = ExecutionState::new();
    let savefile = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/savefile").unwrap());
    let before_lists = state.heap().live_list_count();
    assert!(matches!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/field_once").unwrap(),
            &[Value::Datum(savefile)],
            &mut state,
        ),
        Ok(Value::List(_))
    ));
    assert_eq!(state.heap().live_list_count(), before_lists + 1);

    let before_datums = state.heap().live_datum_count();
    let before_lists = state.heap().live_list_count();
    assert!(matches!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/index_once").unwrap(),
            &[Value::Datum(savefile)],
            &mut state,
        ),
        Ok(Value::Datum(_))
    ));
    assert_eq!(state.heap().live_datum_count(), before_datums + 1);
    assert_eq!(state.heap().live_list_count(), before_lists);
}

#[test]
fn logical_or_empty_list_captures_dynamic_references_once_and_preserves_errors() {
    let source = parse(
            "/datum/helper/proc/next()\n\tcounter += 1\n\treturn holder\n/datum/helper/proc/run()\n\treturn (next().slot ||= list())\n/datum/helper/proc/excluded()\n\treturn (next().slot ||= list(1))\n",
        )
        .unwrap();
    let module = compile_module_specs(&[
        ProcedureSpec {
            path: "/datum/helper/proc/next@0".to_owned(),
            definition: &source.definitions[0],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::from([
                ("counter".to_owned(), field("counter")),
                ("holder".to_owned(), field("holder")),
            ]),
            global_fields: BTreeMap::new(),
        },
        ProcedureSpec {
            path: "/datum/helper/proc/run@0".to_owned(),
            definition: &source.definitions[1],
            parent: None,
            static_calls: BTreeMap::from([("next".to_owned(), 0)]),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
        ProcedureSpec {
            path: "/datum/helper/proc/excluded@0".to_owned(),
            definition: &source.definitions[2],
            parent: None,
            static_calls: BTreeMap::from([("next".to_owned(), 0)]),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
    ])
    .unwrap();
    assert!(
        module
            .procedure(module.procedure_id_at(1).unwrap())
            .unwrap()
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::LogicalOrEmptyListField(_)))
    );
    assert!(
        !module
            .procedure(module.procedure_id_at(2).unwrap())
            .unwrap()
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::LogicalOrEmptyListField(_)))
    );

    let mut state = ExecutionState::new();
    let helper = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/helper").unwrap());
    let holder = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/holder").unwrap());
    state
        .heap_mut()
        .set_datum_field(helper, field("counter"), Value::number(0.0))
        .unwrap();
    state
        .heap_mut()
        .set_datum_field(helper, field("holder"), Value::Datum(holder))
        .unwrap();
    state
        .heap_mut()
        .set_datum_field(holder, field("slot"), Value::Null)
        .unwrap();
    execute_module_in_context(
        &module,
        module.procedure_id_at(1).unwrap(),
        &[],
        &mut state,
        &ExecutionContext::new(Value::Datum(helper), Value::Null),
    )
    .unwrap();
    assert_eq!(
        state.heap().datum_field(helper, &field("counter")),
        Ok(&Value::number(1.0))
    );

    let error_source = parse(
            "/proc/field_error(datum/target)\n\treturn (target.slot ||= list())\n/proc/index_error(list/target)\n\treturn (target[\"x\"] ||= list())\n",
        )
        .unwrap();
    let errors = compile_module(&error_source.definitions).unwrap();
    let field_error = execute_module(
        &errors,
        errors.procedure_id("/proc/field_error").unwrap(),
        &[Value::Null],
    )
    .unwrap_err();
    assert_eq!(field_error.message, "field read received null");
    assert!(field_error.source_span.is_some());
    let index_error = execute_module(
        &errors,
        errors.procedure_id("/proc/index_error").unwrap(),
        &[Value::number(7.0)],
    )
    .unwrap_err();
    assert_eq!(index_error.message, "list index operation received 7");
    assert!(index_error.source_span.is_some());
}

#[test]
fn plane_macro_nested_scope_keeps_cached_locals_visible() {
    let source = parse(
            "/proc/plane_macro(flag, other)\n\tvar/output = 0\n\tdo { if(flag) { var/_cached_plane = 7; var/_our_turf = other; if(_our_turf) { var/key = \"[_cached_plane]\"; output = _cached_plane; } else if(other) { output = _cached_plane; } else { output = _cached_plane; } } else { output = 2; } } while(0)\n\treturn output\n",
        )
        .expect("plane macro source should parse");
    let module = compile_module(&source.definitions).expect("plane macro scope should compile");
    let entry = module.procedure_id("/proc/plane_macro").expect("entry");
    assert_eq!(
        execute_module(&module, entry, &[Value::number(1.0), Value::number(1.0)]),
        Ok(Value::number(7.0))
    );
}

#[test]
fn compact_brace_scope_survives_physical_macro_lines() {
    let source = parse(
            "/proc/plane_macro(flag, inner)\n\tvar/output = 0\n\tif(flag) { var/_cached_plane = 7; if(inner) { output = 1; } else if(flag) { output = 2;\n\t} else { output = _cached_plane; } } else { output = 3; }\n\treturn output\n",
        )
        .expect("continued compact macro body should parse");
    let module =
        compile_module(&source.definitions).expect("continued compact macro scope should compile");
    let entry = module.procedure_id("/proc/plane_macro").expect("entry");
    assert_eq!(
        execute_module(&module, entry, &[Value::number(1.0), Value::Null]),
        Ok(Value::number(2.0))
    );
}

#[test]
fn list_binary_operators_return_new_lists_without_mutating_the_left_operand() {
    let source = parse(
            "/proc/run()\n\tvar/list/a = list(1, 2, 2, 3)\n\tvar/list/b = list(2, 4)\n\tvar/list/added = a + b\n\tvar/list/subtracted = a - b\n\tvar/list/unioned = a | b\n\tvar/list/masked = a & b\n\tvar/list/xored = a ^ b\n\treturn a.len + added.len + subtracted.len + unioned.len + masked.len + xored.len + (a[2] == 2) + (unioned[4] == 4)\n",
        )
        .expect("list operator source should parse");
    let module = compile_module(&source.definitions).expect("list operators should compile");
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/run").unwrap(), &[]),
        Ok(Value::number(24.0))
    );
}

#[test]
fn null_bitand_list_is_zero_and_optional_intersection_iterates_nothing() {
    let source = parse(concat!(
        "/proc/run(list/data)\n",
        "\tvar/list/known = list(\"id\", \"name\")\n",
        "\tvar/count = 0\n",
        "\tfor(var/name in (data & known))\n",
        "\t\tcount++\n",
        "\treturn count + (data & known)\n",
    ))
    .expect("optional list intersection should parse");
    let module =
        compile_module(&source.definitions).expect("optional list intersection should compile");
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/run").unwrap(), &[]),
        Ok(Value::number(0.0))
    );
}

#[test]
fn compound_list_operators_mutate_shared_alias_identity() {
    let source = parse(
            "/proc/run()\n\tvar/list/a = list(1, 2)\n\tvar/list/alias = a\n\ta += list(2, 3)\n\tvar/after_add = alias.len\n\ta -= 2\n\tvar/after_remove = alias.len\n\ta |= list(3, 4)\n\tvar/after_union = alias.len\n\ta &= list(1, 4)\n\tvar/after_mask = alias.len\n\ta ^= list(4, 5)\n\treturn after_add + after_remove + after_union + after_mask + alias.len + (alias[1] == 1) + (alias[2] == 5)\n",
        )
        .expect("compound list operator source should parse");
    let module =
        compile_module(&source.definitions).expect("compound list operators should compile");
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/run").unwrap(), &[]),
        Ok(Value::number(17.0))
    );
}

#[test]
fn bulk_list_subtraction_ignores_rhs_associations_and_preserves_surviving_values() {
    let source = parse(concat!(
            "/proc/run()\n",
            "\tvar/list/left = list(\"x\")\n",
            "\tleft[\"key\"] = \"associated\"\n",
            "\tleft.Add(\"key\", /obj/item)\n",
            "\tleft[\"keep\"] = \"value\"\n",
            "\tvar/list/right = list()\n",
            "\tright[\"key\"] = \"ignored\"\n",
            "\tright.Add(/obj/item)\n",
            "\tvar/list/copied = left - right\n",
            "\tleft -= right\n",
            "\treturn left.len + (left[1] == \"x\") + (left[2] == \"key\") + (left[3] == \"keep\") + (left[\"key\"] == \"associated\") + (left[\"keep\"] == \"value\") + copied.len + (copied[1] == \"x\") + (copied[2] == \"key\") + (copied[3] == \"keep\") + (copied[\"key\"] == \"associated\") + (copied[\"keep\"] == \"value\")\n",
        ))
        .expect("bulk subtraction source should parse");
    let module =
        compile_module(&source.definitions).expect("bulk subtraction source should compile");
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/run").unwrap(), &[]),
        Ok(Value::number(16.0)),
    );
}

#[test]
fn null_plus_equals_list_initializes_a_lazy_list() {
    let source = parse(
        "/proc/queue(value)\n\tvar/list/waiting\n\twaiting += list(value)\n\treturn waiting[1]\n",
    )
    .expect("lazy list source should parse");
    let module = compile_module(&source.definitions).expect("lazy list should compile");
    let entry = module.procedure_id("/proc/queue").expect("queue proc");
    assert_eq!(
        execute_module(&module, entry, &[Value::number(7.0)]),
        Ok(Value::number(7.0))
    );
}

#[test]
fn missing_association_plus_equals_list_initializes_nested_collection() {
    let source = parse(
            "/proc/queue()\n\tvar/list/groups = list()\n\tgroups[\"master\"] += list(/datum/one)\n\tgroups[\"master\"] += list(/datum/two)\n\treturn groups[\"master\"].len\n",
        )
        .expect("nested lazy list source should parse");
    let module = compile_module(&source.definitions).expect("nested lazy list should compile");
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/queue").unwrap(), &[]),
        Ok(Value::number(2.0))
    );
}

#[test]
fn compound_list_assignment_expression_returns_assigned_value() {
    let source = "/proc/run()\n\tvar/list/values = list(\"count\" = 3)\n\tvar/result = (values[\"count\"] += 2)\n\treturn result + values[\"count\"]\n";
    assert_eq!(execute_source(source, 0.0), Value::number(10.0));
}

#[test]
fn physical_line_continuation_is_expression_whitespace() {
    let source = "/proc/run()\n\treturn 1 + \\\n\t\t2\n";
    assert_eq!(execute_source(source, 0.0), Value::number(3.0));
}

#[test]
fn multiline_parenthesized_lists_are_one_logical_statement() {
    let source = "/proc/run()\n\tvar/list/values = list(\n\t\tlist(1, 2),\n\t\tlist(3)\n\t\t,list(4, 5, 6),\n\t)\n\treturn values.len\n";
    assert_eq!(execute_source(source, 0.0), Value::number(3.0));
}

#[test]
fn multiline_list_with_safe_index_ternary_joins_its_closing_parenthesis() {
    let source = concat!(
        "/proc/run()\n",
        "\tvar/list/traits\n",
        "\t. = list(\n",
        "\t\t\"sensors\" = 7,\n",
        "\t\t\"link_allowed\" = ((traits?[\"ai\"] ? TRUE : FALSE) || FALSE),\n",
        "\t)\n",
        "\treturn .[\"sensors\"] + .[\"link_allowed\"]\n",
    );
    assert_eq!(execute_source(source, 0.0), Value::number(7.0));
}

#[test]
fn input_as_constraint_stops_at_enclosing_call_boundary() {
    let source = "/proc/wrap(value)\n\treturn value\n/proc/run()\n\treturn wrap(input(src, \"Prompt\") as null|message)\n";
    assert_eq!(execute_source(source, 0.0), Value::number(0.0));
}

#[test]
fn connected_input_prompt_suspends_and_resumes_with_typed_response() {
    let syntax = parse(
            "/proc/run(user)\n\treturn input(user, \"What?\", \"Prompt title\", \"seed\") as null|message\n",
        )
        .unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let entry = module.procedure_id("/proc/run").unwrap();
    let mut state = ExecutionState::new();
    let client = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/client").unwrap());
    let mob = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/mob").unwrap());
    state.client.attach_mob(client, mob);
    state.install_client_session(client, ControlTree::default());
    state.set_local_client_interactive(client, true).unwrap();

    assert_eq!(
        execute_module_in_state(&module, entry, &[Value::Datum(mob)], &mut state),
        Ok(Value::Null)
    );
    assert_eq!(state.pending_local_prompt_count(), 1);
    assert_eq!(
        state.take_local_client_outbound_events(client),
        vec![LocalClientUiEvent::Prompt {
            id: 1,
            kind: LocalClientPromptKind::Message,
            title: "Prompt title".into(),
            message: "What?".into(),
            default: "seed".into(),
            choices: Vec::new(),
            can_cancel: true,
        }]
    );
    state
        .submit_local_prompt_response(
            client,
            1,
            LocalClientPromptResponse::Text("answered".into()),
        )
        .unwrap();
    assert_eq!(state.pending_local_prompt_count(), 0);
    assert_eq!(
        advance_scheduler(&module, 0, ExecutionLimits::default(), &mut state).unwrap(),
        vec![Value::text("answered")]
    );
}

#[test]
fn skin_only_preflight_client_uses_input_default_without_suspending() {
    let syntax = parse(
        "/proc/run(user)\n\treturn input(user, \"What?\", \"Title\", \"preflight\") as text\n",
    )
    .unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let entry = module.procedure_id("/proc/run").unwrap();
    let mut state = ExecutionState::new();
    let client = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/client").unwrap());
    let mob = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/mob").unwrap());
    state.client.attach_mob(client, mob);
    state.install_client_session(client, ControlTree::default());

    assert_eq!(
        execute_module_in_state(&module, entry, &[Value::Datum(mob)], &mut state),
        Ok(Value::text("preflight"))
    );
    assert_eq!(state.pending_local_prompt_count(), 0);
    assert!(state.take_local_client_outbound_events(client).is_empty());
}

#[test]
fn connected_list_prompt_returns_original_choice_value() {
    let syntax = parse(
            "/proc/run(user)\n\treturn input(user, \"Role?\", \"Choose\", \"Doctor\") as null|anything in list(\"Engineer\", \"Doctor\")\n",
        )
        .unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let entry = module.procedure_id("/proc/run").unwrap();
    let mut state = ExecutionState::new();
    let client = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/client").unwrap());
    let mob = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/mob").unwrap());
    state.client.attach_mob(client, mob);
    state.install_client_session(client, ControlTree::default());
    state.set_local_client_interactive(client, true).unwrap();

    execute_module_in_state(&module, entry, &[Value::Datum(mob)], &mut state).unwrap();
    assert_eq!(
        state.take_local_client_outbound_events(client),
        vec![LocalClientUiEvent::Prompt {
            id: 1,
            kind: LocalClientPromptKind::List,
            title: "Choose".into(),
            message: "Role?".into(),
            default: "Doctor".into(),
            choices: vec!["Engineer".into(), "Doctor".into()],
            can_cancel: true,
        }]
    );
    state
        .submit_local_prompt_response(client, 1, LocalClientPromptResponse::Choice(0))
        .unwrap();
    assert_eq!(
        advance_scheduler(&module, 0, ExecutionLimits::default(), &mut state).unwrap(),
        vec![Value::text("Engineer")]
    );
}

#[test]
fn connected_alert_prompt_returns_selected_button() {
    let syntax = parse(
        "/proc/run(user)\n\treturn alert(user, \"Continue?\", \"Question\", \"Yes\", \"No\")\n",
    )
    .unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let entry = module.procedure_id("/proc/run").unwrap();
    let mut state = ExecutionState::new();
    let client = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/client").unwrap());
    let mob = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/mob").unwrap());
    state.client.attach_mob(client, mob);
    state.install_client_session(client, ControlTree::default());
    state.set_local_client_interactive(client, true).unwrap();

    execute_module_in_state(&module, entry, &[Value::Datum(mob)], &mut state).unwrap();
    let events = state.take_local_client_outbound_events(client);
    let [LocalClientUiEvent::Prompt { choices, .. }] = events.as_slice() else {
        panic!("alert must emit one prompt")
    };
    assert_eq!(choices, &["Yes", "No"]);
    state
        .submit_local_prompt_response(client, 1, LocalClientPromptResponse::Choice(1))
        .unwrap();
    assert_eq!(
        advance_scheduler(&module, 0, ExecutionLimits::default(), &mut state).unwrap(),
        vec![Value::text("No")]
    );
}

#[test]
fn documented_list_methods_and_len_execute_natively() {
    let source = parse(
            "/proc/run()\n\tvar/list/values = list(\"a\", \"b\", \"c\")\n\tvalues.Add(list(\"d\", \"e\"))\n\tvar/list/copied = values.Copy(2, 5)\n\tvalues.Cut(2, 3)\n\tvar/found = values.Find(\"d\")\n\tvar/next_index = values.Insert(2, list(\"x\", \"y\"))\n\tvalues.Splice(-1, 0, \"z\")\n\tvalues.Swap(1, 6)\n\tvalues.len = 7\n\tvar/removed = values.Remove(\"d\")\n\tvar/removed_all = values.RemoveAll(\"x\")\n\treturn copied.len + (copied[1] == \"b\") + (copied[3] == \"d\") + found + next_index + removed + removed_all + values.len + (values[1] == \"z\") + (values[2] == \"y\")\n",
        )
        .expect("list method source should parse");
    let module = compile_module(&source.definitions).expect("list methods should compile");
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/run").unwrap(), &[]),
        Ok(Value::number(21.0))
    );
}

#[test]
fn list_join_defaults_match_monkestation_bodypart_render_keys() {
    let source = parse(
        "/datum/bodypart/proc/generate_icon_key()\n\
             \treturn list(\"human\", \"left_arm\", 2)\n\
             /proc/run()\n\
             \tvar/datum/bodypart/limb = new\n\
             \tvar/exact_monk_shape = limb.generate_icon_key().Join()\n\
             \tvar/null_glue = list(\"a\", \"b\", \"c\", \"d\").Join(null, 2, 0)\n\
             \tvar/range = list(\"a\", \"b\", \"c\", \"d\").Join(\"-\", 2, 4)\n\
             \tvar/negative_end = list(\"a\", \"b\", \"c\", \"d\").Join(\"-\", 1, -1)\n\
             \tvar/empty_range = list(\"a\", \"b\").Join(\",\", 0, 0)\n\
             \treturn list(exact_monk_shape, null_glue, range, negative_end, empty_range)\n",
    )
    .expect("Monkestation bodypart Join fixture should parse");
    let module = compile_module(&source.definitions)
        .expect("Monkestation bodypart Join fixture should compile");
    let mut state = ExecutionState::new();
    let Value::List(result) = execute_module_in_state(
        &module,
        module.procedure_id("/proc/run").unwrap(),
        &[],
        &mut state,
    )
    .expect("all documented Join default and range forms should execute") else {
        panic!("Join fixture should return its observations");
    };
    let result = state.heap().list(result).unwrap();
    assert_eq!(result.get(1), Ok(&Value::text("humanleft_arm2")));
    assert_eq!(result.get(2), Ok(&Value::text("bcd")));
    assert_eq!(result.get(3), Ok(&Value::text("b-c")));
    assert_eq!(result.get(4), Ok(&Value::text("a-b-c")));
    assert_eq!(result.get(5), Ok(&Value::text("")));
}

#[test]
fn vis_contents_and_vis_locs_stay_reciprocal_across_mutation_and_destroy_cleanup() {
    let syntax = parse(
            "/proc/link(atom/owner, atom/child)\n\towner.vis_contents = null\n\tchild.vis_locs = null\n\towner.vis_contents.Add(child)\n\tvar/added = (owner.vis_contents.len == 1 && child.vis_locs[1] == owner)\n\towner.vis_contents.Remove(child)\n\tvar/removed = (owner.vis_contents.len == 0 && child.vis_locs.len == 0)\n\towner.vis_contents = list(child, child, null)\n\tvar/assigned = (owner.vis_contents.len == 1 && child.vis_locs.len == 1)\n\towner.vis_contents.Cut()\n\tvar/cut = (owner.vis_contents.len == 0 && child.vis_locs.len == 0)\n\towner.vis_contents += child\n\tchild.vis_locs = null\n\tvar/destroy_cleanup = (owner.vis_contents.len == 0 && child.vis_locs.len == 0)\n\treturn added + removed + assigned + cut + destroy_cleanup\n",
        )
        .expect("visibility relationship fixture should parse");
    let module = compile_module(&syntax.definitions)
        .expect("visibility relationship fixture should compile");
    let mut state = ExecutionState::new();
    let owner = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/atom/movable/owner").unwrap());
    let child = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/atom/movable/child").unwrap());

    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id_at(0).unwrap(),
            &[Value::Datum(owner), Value::Datum(child)],
            &mut state,
        ),
        Ok(Value::number(5.0))
    );
}

#[test]
fn scalar_and_bulk_vis_contents_mutation_match_and_cleanup_reciprocals() {
    let syntax = parse(
            "/proc/run(atom/scalar_owner, atom/bulk_owner, atom/first, atom/second)\n\tscalar_owner.vis_contents = null\n\tbulk_owner.vis_contents = null\n\tfirst.vis_locs = null\n\tsecond.vis_locs = null\n\tscalar_owner.vis_contents += first\n\tscalar_owner.vis_contents.Add(second)\n\tscalar_owner.vis_contents += first\n\tscalar_owner.vis_contents += null\n\tvar/scalar_shape = (scalar_owner.vis_contents.len == 2 && first.vis_locs.len == 1 && first.vis_locs[1] == scalar_owner && second.vis_locs.len == 1 && second.vis_locs[1] == scalar_owner)\n\tbulk_owner.vis_contents += list(first, second, first, null)\n\tvar/bulk_shape = (bulk_owner.vis_contents.len == 2 && first.vis_locs.len == 2 && first.vis_locs[2] == bulk_owner && second.vis_locs.len == 2 && second.vis_locs[2] == bulk_owner)\n\tscalar_owner.vis_contents -= first\n\tscalar_owner.vis_contents.Remove(second)\n\tvar/scalar_cleanup = (scalar_owner.vis_contents.len == 0 && first.vis_locs.len == 1 && first.vis_locs[1] == bulk_owner && second.vis_locs.len == 1 && second.vis_locs[1] == bulk_owner)\n\tbulk_owner.vis_contents -= list(first, second)\n\tvar/bulk_cleanup = (bulk_owner.vis_contents.len == 0 && first.vis_locs.len == 0 && second.vis_locs.len == 0)\n\treturn scalar_shape + bulk_shape + scalar_cleanup + bulk_cleanup\n",
        )
        .expect("scalar and bulk visibility fixture should parse");
    let module = compile_module(&syntax.definitions)
        .expect("scalar and bulk visibility fixture should compile");
    let mut state = ExecutionState::new();
    let datums = ["scalar_owner", "bulk_owner", "first", "second"].map(|name| {
        state
            .heap_mut()
            .allocate_datum(TypePath::parse(&format!("/atom/movable/{name}")).unwrap())
    });

    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/run").unwrap(),
            &datums.map(Value::Datum),
            &mut state,
        ),
        Ok(Value::number(4.0))
    );
}

#[test]
fn atom_native_lists_materialize_lazily_and_keep_stable_identity() {
    let syntax = parse(
            "/proc/read_vis_contents(atom/target)\n\tvar/list/first = target.vis_contents\n\tvar/list/second = target.vis_contents\n\treturn first == second\n/proc/read_vis_locs_through_vars(atom/target)\n\treturn target.vars[\"vis_locs\"] == target.vis_locs\n/proc/read_appearance_and_verb_lists(atom/target)\n\tvar/list/overlays = target.overlays\n\toverlays.Add(\"state\")\n\tvar/list/underlays = target.underlays\n\tvar/list/filters = target.vars[\"filters\"]\n\tvar/list/verbs = target.verbs\n\treturn (overlays == target.overlays) + (target.overlays.len == 1) + (underlays == target.underlays) + (filters == target.filters) + (verbs == target.verbs)\n",
        )
        .expect("lazy visibility field fixture should parse");
    let module =
        compile_module(&syntax.definitions).expect("lazy visibility field fixture should compile");
    let mut state = ExecutionState::new();
    let atom =
        allocate_initialized_datum(&mut state, TypePath::parse("/obj/item").unwrap()).unwrap();

    assert_eq!(
        state.heap().live_list_count(),
        0,
        "untouched native atom lists must remain unallocated"
    );
    assert!(
        state
            .heap()
            .datum_field(atom, &field("vis_contents"))
            .is_err()
    );
    assert!(state.heap().datum_field(atom, &field("vis_locs")).is_err());

    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/read_vis_contents").unwrap(),
            &[Value::Datum(atom)],
            &mut state,
        ),
        Ok(Value::number(1.0))
    );
    assert_eq!(
        state.heap().live_list_count(),
        1,
        "the first vis_contents read materializes exactly one stable list"
    );

    assert_eq!(
        execute_module_in_state(
            &module,
            module
                .procedure_id("/proc/read_vis_locs_through_vars")
                .unwrap(),
            &[Value::Datum(atom)],
            &mut state,
        ),
        Ok(Value::number(1.0))
    );
    assert_eq!(
        state.heap().live_list_count(),
        2,
        "direct datum.vars indexing avoids a reflection proxy while vis_locs stays lazy"
    );

    assert_eq!(
        execute_module_in_state(
            &module,
            module
                .procedure_id("/proc/read_appearance_and_verb_lists")
                .unwrap(),
            &[Value::Datum(atom)],
            &mut state,
        ),
        Ok(Value::number(5.0))
    );
    assert_eq!(
        state.heap().live_list_count(),
        6,
        "four first accesses materialize overlays, underlays, filters, and verbs once without a reflection proxy"
    );
}

#[test]
fn mutable_appearance_overlay_lists_materialize_lazily_for_scalar_addition() {
    let syntax = parse(
            "/proc/run()\n\tvar/mutable_appearance/container = new\n\tvar/mutable_appearance/member = new\n\tcontainer.overlays += member\n\treturn list(container.overlays.len, container.overlays[1] == member, container.overlays == container.overlays)\n/mutable_appearance\n",
        )
        .expect("mutable appearance list fixture should parse");
    let module = compile_module(
        &syntax
            .definitions
            .iter()
            .filter(|definition| matches!(definition.kind, DefinitionKind::Procedure))
            .cloned()
            .collect::<Vec<_>>(),
    )
    .expect("mutable appearance list fixture should compile");
    let entry = module.procedure_id("/proc/run").unwrap();
    let mut state = ExecutionState::new();
    state.set_type_parents(
        [
            (TypePath::parse("/datum").unwrap(), None),
            (
                TypePath::parse("/image").unwrap(),
                Some(TypePath::parse("/datum").unwrap()),
            ),
            (
                TypePath::parse("/mutable_appearance").unwrap(),
                Some(TypePath::parse("/image").unwrap()),
            ),
        ]
        .into(),
    );
    let Value::List(result) = execute_module_with_limits_in_state(
        &module,
        entry,
        &[],
        ExecutionLimits::default(),
        &mut state,
    )
    .expect("mutable appearance overlay addition should execute") else {
        panic!("fixture should return a list")
    };
    assert_eq!(
        state
            .heap()
            .list(result)
            .unwrap()
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>(),
        vec![Value::number(1.0), Value::number(1.0), Value::number(1.0)]
    );
}

#[test]
fn list_copy_and_swap_keep_associative_values_attached_to_keys() {
    let source = parse(
            "/proc/run()\n\tvar/list/values = list(\"red\" = 1, \"blue\" = 2, \"green\" = 3)\n\tvar/list/copied = values.Copy()\n\tvalues.Swap(1, 3)\n\treturn (values[1] == \"green\") + (values[\"green\"] == 3) + (copied[1] == \"red\") + (copied[\"red\"] == 1)\n",
        )
        .expect("associative list method source should parse");
    let module =
        compile_module(&source.definitions).expect("associative list methods should compile");
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/run").unwrap(), &[]),
        Ok(Value::number(4.0))
    );
}

#[test]
fn documented_native_builtins_cover_text_math_and_type_helpers() {
    let source = parse(
            "/proc/native(kind)\n\tvar/path = text2path(\"/datum/child\")\n\tif(!path)\n\t\treturn 0\n\treturn (2 ** 3 ** 2) + floor(1.9) + abs(-2) + findlasttext(\"/datum/child\", \"/\") + initial(kind.flag)\n",
        )
        .expect("native builtin source should parse");
    let module = compile_module(&source.definitions).expect("native builtins should compile");
    let mut state = ExecutionState::new();
    let base = TypePath::parse("/datum/base").unwrap();
    let child = TypePath::parse("/datum/child").unwrap();
    state.set_type_paths([base.clone(), child.clone()]);
    state.set_type_parents(BTreeMap::from([
        (base.clone(), Some(TypePath::parse("/datum").unwrap())),
        (child.clone(), Some(base.clone())),
    ]));
    state.set_initial_values(BTreeMap::from([(
        child.clone(),
        BTreeMap::from([(field("flag"), Value::number(7.0))]),
    )]));
    let result = execute_module_in_state(
        &module,
        module.procedure_id("/proc/native").unwrap(),
        &[Value::TypePath(child)],
        &mut state,
    )
    .expect("native builtin procedure should execute");
    // 2 ** (3 ** 2) = 512; floor=1; abs=2; final slash is byte 7; initial=7.
    assert_eq!(result, Value::number(529.0));
}

#[test]
fn namespaced_runtime_type_value_reads_its_initial_field() {
    let source = parse("/proc/read_mode(component_type)\n\treturn component_type::dupe_mode\n")
        .expect("namespaced value source should parse");
    let module =
        compile_module(&source.definitions).expect("namespaced runtime type value should compile");
    let component = TypePath::parse("/datum/component/example").unwrap();
    let mut state = ExecutionState::new();
    state.set_initial_values(BTreeMap::from([(
        component.clone(),
        BTreeMap::from([(field("dupe_mode"), Value::number(3.0))]),
    )]));
    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/read_mode").unwrap(),
            &[Value::TypePath(component)],
            &mut state,
        ),
        Ok(Value::number(3.0))
    );
}

#[test]
fn typed_typepath_field_read_uses_the_declared_initial_value() {
    let source = parse("/proc/read(datum/sound_effect/sfx)\n\treturn sfx.key\n")
        .expect("sound-key-shaped source should parse");
    let module =
        compile_module(&source.definitions).expect("typed type-path field access should compile");
    let child = TypePath::parse("/datum/sound_effect/child").unwrap();
    let mut state = ExecutionState::new();
    state.set_initial_values(BTreeMap::from([(
        child.clone(),
        BTreeMap::from([(field("key"), Value::text("child-key"))]),
    )]));
    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/read").unwrap(),
            &[Value::TypePath(child)],
            &mut state,
        ),
        Ok(Value::text("child-key"))
    );
}

#[test]
fn initial_supports_runtime_vars_index_field_names() {
    let source = parse("/proc/read(target, variable)\n\treturn initial(target.vars[variable])\n")
        .expect("dynamic initial source should parse");
    let module =
        compile_module(&source.definitions).expect("initial(object.vars[name]) should compile");
    let path = TypePath::parse("/datum/example").unwrap();
    let mut state = ExecutionState::new();
    state.set_initial_values(BTreeMap::from([(
        path.clone(),
        BTreeMap::from([(field("flag"), Value::number(7.0))]),
    )]));
    let datum = state.heap_mut().allocate_datum(path);
    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/read").unwrap(),
            &[Value::Datum(datum), Value::text("flag")],
            &mut state,
        ),
        Ok(Value::number(7.0))
    );
}

#[test]
fn scope_operator_supports_type_src_and_global_values() {
    let type_source = parse("/proc/read()\n\treturn /datum/example::flag\n")
        .expect("type scope source should parse");
    let type_program =
        compile_procedure(&type_source.definitions[0]).expect("type scope should compile");
    let path = TypePath::parse("/datum/example").unwrap();
    let mut state = ExecutionState::new();
    state.set_initial_values(BTreeMap::from([(
        path.clone(),
        BTreeMap::from([(field("flag"), Value::number(7.0))]),
    )]));
    assert_eq!(
        execute_in_context(&type_program, &[], &mut state, &ExecutionContext::default(),),
        Ok(Value::number(7.0))
    );

    let global_source =
        parse("/proc/read()\n\treturn ::answer\n").expect("global scope source should parse");
    let global_program =
        compile_procedure(&global_source.definitions[0]).expect("global scope should compile");
    state.set_global(field("answer"), Value::number(42.0));
    assert_eq!(
        execute_in_context(
            &global_program,
            &[],
            &mut state,
            &ExecutionContext::default(),
        ),
        Ok(Value::number(42.0))
    );

    let src_source =
        parse("/proc/read()\n\treturn src::flag\n").expect("src scope source should parse");
    let src_program =
        compile_procedure(&src_source.definitions[0]).expect("src scope should compile");
    let src = state.heap_mut().allocate_datum(path);
    assert_eq!(
        execute_in_context(
            &src_program,
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(src), Value::Null),
        ),
        Ok(Value::number(7.0))
    );
}

#[test]
fn type_predicates_follow_runtime_parent_catalog_not_path_spelling() {
    let source =
        parse("/proc/check(value)\n\treturn istype(value, /atom/movable) && ismovable(value)\n")
            .expect("predicate source should parse");
    let module = compile_module(&source.definitions).expect("predicate source should compile");
    let mut state = ExecutionState::new();
    let obj = TypePath::parse("/obj/item").unwrap();
    state.set_type_parents(BTreeMap::from([
        (obj.clone(), Some(TypePath::parse("/obj").unwrap())),
        (
            TypePath::parse("/obj").unwrap(),
            Some(TypePath::parse("/atom/movable").unwrap()),
        ),
        (
            TypePath::parse("/atom/movable").unwrap(),
            Some(TypePath::parse("/atom").unwrap()),
        ),
    ]));
    let datum = state.heap_mut().allocate_datum(obj);
    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/check").unwrap(),
            &[Value::Datum(datum)],
            &mut state,
        ),
        Ok(Value::number(1.0))
    );
}

#[test]
fn direction_and_icon_builtins_cover_lifecycle_shapes() {
    let source = parse(
            "/proc/directions()\n\treturn NORTH + SOUTH + EAST + WEST + NORTHEAST + NORTHWEST + SOUTHEAST + SOUTHWEST\n/proc/icon_resource()\n\treturn isicon('icons/test.dmi')\n",
        )
        .expect("builtin source should parse");
    let module = compile_module(&source.definitions).expect("builtins should compile");
    assert_eq!(
        execute_module(
            &module,
            module.procedure_id("/proc/directions").unwrap(),
            &[]
        ),
        Ok(Value::number(45.0))
    );
    assert_eq!(
        execute_module(
            &module,
            module.procedure_id("/proc/icon_resource").unwrap(),
            &[]
        ),
        Ok(Value::number(1.0))
    );
}

#[test]
fn direction_mask_fast_path_preserves_dm_validation_boundaries() {
    for (value, expected) in [
        (-1.0, None),
        (-0.0, Some(0)),
        (0.0, Some(0)),
        (1.0, Some(1)),
        (15.0, Some(15)),
        (63.0, Some(63)),
        (64.0, None),
        (1.5, None),
        (f32::INFINITY, None),
        (f32::NAN, None),
    ] {
        assert_eq!(crate::dm_direction_bits(value), expected, "value={value}");
    }
}

#[test]
fn world_coordinate_fast_path_preserves_integer_parse_boundaries() {
    for (value, expected) in [
        (f32::NEG_INFINITY, None),
        (-2_147_483_648.0, Some(i32::MIN)),
        (-1.0, Some(-1)),
        (-0.0, Some(0)),
        (1.0, Some(1)),
        (1.5, None),
        (16_777_216.0, Some(16_777_216)),
        (2_147_483_648.0, None),
        (f32::INFINITY, None),
        (f32::NAN, None),
    ] {
        assert_eq!(crate::dm_world_coordinate(value), expected, "value={value}");
    }
}

#[test]
#[ignore = "release-only world-coordinate conversion microbenchmark"]
fn world_coordinate_conversion_release_microbenchmark() {
    let rounds = 2_000_000usize;
    let started = Instant::now();
    let mut decimal_sum = 0_i64;
    for index in 0..rounds {
        let value = std::hint::black_box((index & 65_535) as f32);
        decimal_sum += value.to_string().parse::<i32>().unwrap() as i64;
    }
    let decimal_elapsed = started.elapsed();
    let started = Instant::now();
    let mut direct_sum = 0_i64;
    for index in 0..rounds {
        let value = std::hint::black_box((index & 65_535) as f32);
        direct_sum += crate::dm_world_coordinate(value).unwrap() as i64;
    }
    let direct_elapsed = started.elapsed();
    eprintln!(
        "world-coordinate-conversion rounds={rounds} decimal_ms={} direct_ms={}",
        decimal_elapsed.as_millis(),
        direct_elapsed.as_millis(),
    );
    assert_eq!(decimal_sum, direct_sum);
}

#[test]
#[ignore = "release-only direction-mask conversion microbenchmark"]
fn direction_mask_conversion_release_microbenchmark() {
    let rounds = 2_000_000usize;
    let started = Instant::now();
    let mut decimal_sum = 0_i64;
    for index in 0..rounds {
        let value = std::hint::black_box((index & 63) as f32);
        decimal_sum += value.to_string().parse::<i16>().unwrap() as i64;
    }
    let decimal_elapsed = started.elapsed();
    let started = Instant::now();
    let mut direct_sum = 0_i64;
    for index in 0..rounds {
        let value = std::hint::black_box((index & 63) as f32);
        direct_sum += crate::dm_direction_bits(value).unwrap() as i64;
    }
    let direct_elapsed = started.elapsed();
    eprintln!(
        "direction-mask-conversion rounds={rounds} decimal_ms={} direct_ms={}",
        decimal_elapsed.as_millis(),
        direct_elapsed.as_millis(),
    );
    assert_eq!(decimal_sum, direct_sum);
}

#[test]
fn shared_value_migration_preserves_scalar_execution() {
    let program = manual_program(
        vec![
            Instruction::PushNumber(DmNumberBits::from_f32(2.0)),
            Instruction::PushNumber(DmNumberBits::from_f32(3.0)),
            Instruction::Add,
            Instruction::Return,
        ],
        0,
    );

    assert_eq!(execute(&program, &[]), Ok(Value::number(5.0)));
}

#[test]
fn datum_type_field_reflects_the_heap_runtime_type() {
    let program = manual_program(
        vec![
            Instruction::LoadSrc,
            Instruction::LoadField(field("type")),
            Instruction::Return,
        ],
        0,
    );
    let mut state = ExecutionState::new();
    let path = TypePath::parse("/obj/machinery/example").unwrap();
    let datum = state.heap_mut().allocate_datum(path.clone());

    assert_eq!(
        execute_in_context(
            &program,
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(datum), Value::Null),
        ),
        Ok(Value::TypePath(path))
    );
}

#[test]
fn list_construction_allocates_heap_storage_in_source_order() {
    let program = manual_program(
        vec![
            Instruction::PushNumber(DmNumberBits::from_f32(7.0)),
            Instruction::PushText(Arc::from("second")),
            Instruction::MakeList(2),
            Instruction::Return,
        ],
        0,
    );
    let mut state = ExecutionState::new();
    let result = execute_in_state(&program, &[], &mut state).unwrap();
    let Value::List(list) = result else {
        panic!("MakeList must return a list handle");
    };

    let values = state.heap().list(list).unwrap();
    assert!(values.get(1).unwrap().semantic_eq(&Value::number(7.0)));
    assert!(values.get(2).unwrap().semantic_eq(&Value::text("second")));
}

#[test]
fn list_aliases_observe_heap_mutation_across_executions() {
    let program = manual_program(
        vec![
            Instruction::LoadLocal(0),
            Instruction::PushNumber(DmNumberBits::from_f32(1.0)),
            Instruction::IndexList,
            Instruction::Return,
        ],
        1,
    );
    let mut state = ExecutionState::new();
    let list = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(list)
        .unwrap()
        .add(Value::number(4.0));
    let alias = Value::List(list);

    assert_eq!(
        execute_in_state(&program, std::slice::from_ref(&alias), &mut state),
        Ok(Value::number(4.0))
    );
    state
        .heap_mut()
        .list_mut(list)
        .unwrap()
        .set(1, Value::number(9.0))
        .unwrap();
    assert_eq!(
        execute_in_state(&program, &[alias], &mut state),
        Ok(Value::number(9.0))
    );
}

#[test]
fn local_list_index_compiles_to_one_receiver_lookup_and_preserves_semantics() {
    let syntax = parse("/proc/read(list/values, index)\n\treturn values[index]\n").unwrap();
    let program = compile_procedure(&syntax.definitions[0]).unwrap();
    assert!(
        program
            .instructions
            .contains(&Instruction::IndexLocalList(0))
    );
    assert!(!program.instructions.contains(&Instruction::IndexList));

    let mut state = ExecutionState::new();
    let list = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(list)
        .unwrap()
        .add(Value::text("mapped"));
    assert_eq!(
        execute_in_state(
            &program,
            &[Value::List(list), Value::number(1.0)],
            &mut state,
        ),
        Ok(Value::text("mapped"))
    );
}

#[test]
fn stale_list_indexing_maps_to_source_aware_runtime_error() {
    let program = manual_program(
        vec![
            Instruction::LoadLocal(0),
            Instruction::PushNumber(DmNumberBits::from_f32(1.0)),
            Instruction::IndexList,
            Instruction::Return,
        ],
        1,
    );
    let mut state = ExecutionState::new();
    let stale_list = state.heap_mut().allocate_list();
    state.heap_mut().destroy_list(stale_list).unwrap();
    let error = execute_in_state(&program, &[Value::List(stale_list)], &mut state)
        .expect_err("a stale handle must never resolve through the VM");

    assert_eq!(error.message, "list index operation received null");
    assert_eq!(error.instruction, 2);
    assert_eq!(error.source_span, Some(SourceSpan::new(20, 21)));
    assert_eq!(error.call_stack.len(), 1);
}

#[test]
fn direct_loc_assignment_rejects_descendant_containment_cycles() {
    let syntax =
        parse("/proc/move(atom, target)\n\tatom.loc = target\n\treturn atom.loc\n").unwrap();
    let program = compile_procedure(&syntax.definitions[0]).unwrap();
    let mut state = ExecutionState::new();
    let turf = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/turf/floor").unwrap());
    let parent = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/storage").unwrap());
    let child = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/item").unwrap());
    for (name, value) in [("x", 1.0), ("y", 1.0), ("z", 1.0)] {
        state
            .heap_mut()
            .set_datum_field(turf, field(name), Value::number(value))
            .unwrap();
    }
    super::builtins::move_movable_to_atom(&mut state, parent, turf).unwrap();
    super::builtins::move_movable_to_atom(&mut state, child, parent).unwrap();

    assert_eq!(
        execute_in_state(
            &program,
            &[Value::Datum(parent), Value::Datum(child)],
            &mut state,
        ),
        Ok(Value::Datum(turf)),
    );
    assert_eq!(
        state.heap().datum_field(parent, &field("loc")),
        Ok(&Value::Datum(turf)),
    );
}

#[test]
fn atom_contents_read_materializes_one_stable_list() {
    let syntax = parse(
            "/proc/read(atom/target)\n\tvar/list/first = target.contents\n\tvar/list/second = target.vars[\"contents\"]\n\treturn first == second\n",
        )
        .unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let mut state = ExecutionState::new();
    let atom =
        allocate_initialized_datum(&mut state, TypePath::parse("/obj/item").unwrap()).unwrap();
    assert!(state.heap().datum_field(atom, &field("contents")).is_err());
    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/read").unwrap(),
            &[Value::Datum(atom)],
            &mut state,
        ),
        Ok(Value::number(1.0)),
    );
    assert!(matches!(
        state.heap().datum_field(atom, &field("contents")),
        Ok(Value::List(_))
    ));
}

#[test]
fn declared_field_read_is_null_for_runtime_incompatible_typed_value() {
    let syntax = parse(
            "/proc/read_typed(obj/item/clothing/suit/space/suit)\n\treturn suit.cell\n/proc/read_dynamic(suit)\n\treturn suit.cell\n",
        )
        .expect("typed-field compatibility fixture should parse");
    let module = compile_module(&syntax.definitions)
        .expect("typed-field compatibility fixture should compile");
    let mut state = ExecutionState::new();
    let explorer = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/item/clothing/suit/hooded/explorer").unwrap());

    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/read_typed").unwrap(),
            &[Value::Datum(explorer)],
            &mut state,
        ),
        Ok(Value::Null),
        "a field valid on the declared /space type reads null when an unrelated runtime value lacks it",
    );
    let error = execute_module_in_state(
        &module,
        module.procedure_id("/proc/read_dynamic").unwrap(),
        &[Value::Datum(explorer)],
        &mut state,
    )
    .expect_err("a genuinely dynamic missing field must remain an error");
    assert!(
        error
            .message
            .contains("field FieldName(\"cell\") is absent")
    );
}

#[test]
fn declared_field_quickening_hits_and_revalidates_shifted_slots() {
    let source = parse(concat!(
        "/proc/read_quickened(obj/item/clothing/suit/space/target)\n",
        "\treturn target.cell\n",
    ))
    .unwrap();
    let module = compile_module(&source.definitions).unwrap();
    let entry = module.procedure_id("/proc/read_quickened").unwrap();
    let mut state = ExecutionState::new();
    let datum = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/item/clothing/suit/space").unwrap());
    state
        .heap_mut()
        .set_datum_field(datum, field("prefix"), Value::number(1.0))
        .unwrap();
    state
        .heap_mut()
        .set_datum_field(datum, field("cell"), Value::number(42.0))
        .unwrap();

    for _ in 0..2 {
        assert_eq!(
            execute_module_in_state(&module, entry, &[Value::Datum(datum)], &mut state),
            Ok(Value::number(42.0))
        );
    }
    let warm = state.declared_field_quickening_metrics();
    assert_eq!(warm.misses, 1);
    assert_eq!(warm.hits, 1);

    state
        .heap_mut()
        .delete_datum_field(datum, &field("prefix"))
        .unwrap();
    assert_eq!(
        execute_module_in_state(&module, entry, &[Value::Datum(datum)], &mut state),
        Ok(Value::number(42.0))
    );
    let shifted = state.declared_field_quickening_metrics();
    assert_eq!(shifted.invalidations, 1);
    assert_eq!(shifted.misses, 2);
}

#[test]
fn ordinary_static_field_reads_quicken_without_changing_missing_field_errors() {
    let source = parse(concat!(
        "/proc/read_dynamic(target)\n",
        "\treturn target.cell\n",
    ))
    .unwrap();
    let module = compile_module(&source.definitions).unwrap();
    let entry = module.procedure_id("/proc/read_dynamic").unwrap();
    assert!(module
            .resolve_procedure(entry)
            .unwrap()
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::LoadField(name) if name.as_str() == "cell")));
    let mut state = ExecutionState::new();
    let datum = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/profiled").unwrap());
    state
        .heap_mut()
        .set_datum_field(datum, field("cell"), Value::number(42.0))
        .unwrap();

    for _ in 0..2 {
        assert_eq!(
            execute_module_in_state(&module, entry, &[Value::Datum(datum)], &mut state),
            Ok(Value::number(42.0))
        );
    }
    let warm = state.static_field_quickening_metrics();
    assert_eq!(warm.misses, 1);
    assert_eq!(warm.hits, 1);

    state
        .heap_mut()
        .delete_datum_field(datum, &field("cell"))
        .unwrap();
    let error = execute_module_in_state(&module, entry, &[Value::Datum(datum)], &mut state)
        .expect_err("ordinary missing fields must remain runtime errors");
    assert!(
        error
            .message
            .contains("field FieldName(\"cell\") is absent")
    );
    let invalidated = state.static_field_quickening_metrics();
    assert_eq!(invalidated.invalidations, 1);
}

#[test]
#[ignore = "release-only declared-field dense-slot benchmark"]
fn declared_field_dense_slot_benchmark() {
    const READS: usize = 5_000_000;
    let mut state = ExecutionState::new();
    let datum = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/profiled").unwrap());
    for index in 0..64 {
        state
            .heap_mut()
            .set_datum_field(
                datum,
                field(&format!("profiled_{index}")),
                Value::number(index as f32),
            )
            .unwrap();
    }
    let target = field("profiled_47");
    let record = state.heap().datum(datum).unwrap();
    let slot = record.field_slot(&target).unwrap();
    let started = Instant::now();
    for _ in 0..READS {
        std::hint::black_box(record.field(&target).unwrap());
    }
    let named = started.elapsed();
    let started = Instant::now();
    for _ in 0..READS {
        std::hint::black_box(record.field_at_validated_slot(slot, &target).unwrap());
    }
    let quickened = started.elapsed();
    eprintln!(
        "declared-field reads={READS} named_ms={} quickened_ms={} speedup={:.3}",
        named.as_millis(),
        quickened.as_millis(),
        named.as_secs_f64() / quickened.as_secs_f64(),
    );
}

#[test]
fn list_gc_roots_materialized_args_across_scheduler_yield() {
    let syntax = parse(concat!(
        "/proc/work(value)\n",
        "\tvar/count = args.len\n",
        "\tsleep(1)\n",
        "\treturn args[1] + count\n",
    ))
    .expect("args GC fixture should parse");
    let module = compile_module(&syntax.definitions).expect("fixture should compile");
    let entry = module.procedure_id("/proc/work").unwrap();
    let mut state = ExecutionState::new();
    state.next_list_collection = 1;

    assert_eq!(
        execute_module_in_state(&module, entry, &[Value::number(6.0)], &mut state),
        Ok(Value::Null),
    );
    assert_eq!(state.scheduled_task_count(), 1);
    assert_eq!(
        advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
        Ok(vec![Value::number(7.0)]),
    );
}

#[test]
fn list_gc_roots_reused_argument_list_through_nested_calls_and_many_yields() {
    let syntax = parse(concat!(
        "/proc/init_atom(list/arguments)\n",
        "\treturn consume(arglist(arguments))\n",
        "/proc/consume(mapload)\n",
        "\treturn mapload\n",
        "/proc/create_atoms()\n",
        "\tvar/list/mapload_arg = list(1)\n",
        "\tvar/total = 0\n",
        "\tfor(var/I in 1 to 40)\n",
        "\t\tsleep(1)\n",
        "\t\ttotal += init_atom(mapload_arg)\n",
        "\treturn total\n",
    ))
    .expect("nested argument-list GC fixture should parse");
    let module = compile_module(&syntax.definitions).expect("fixture should compile");
    let entry = module.procedure_id("/proc/create_atoms").unwrap();
    let mut state = ExecutionState::new();
    state.next_list_collection = 1;

    assert_eq!(
        execute_module_in_state(&module, entry, &[], &mut state),
        Ok(Value::Null),
    );
    for tick in 1..40 {
        assert!(
            advance_scheduler(&module, tick, ExecutionLimits::default(), &mut state)
                .expect("continuation must retain its reusable argument list")
                .is_empty()
        );
        state.next_list_collection = 1;
    }
    state.next_list_collection = 1;
    assert_eq!(
        advance_scheduler(&module, 40, ExecutionLimits::default(), &mut state),
        Ok(vec![Value::number(40.0)]),
    );
}

#[test]
fn list_gc_roots_outer_frames_during_reentrant_instance_initializer_yield() {
    let syntax = parse(concat!(
        "/proc/outer()\n",
        "\tvar/list/arguments = list(1)\n",
        "\tvar/datum/reentrant/value = new /datum/reentrant\n",
        "\treturn arguments[1]\n",
    ))
    .expect("reentrant outer-frame fixture should parse");
    let module = compile_module(&syntax.definitions).expect("outer fixture should compile");
    let initializer_syntax = parse(concat!(
        "/proc/dynamic_default()\n",
        "\tsleep(1)\n",
        "\treturn 7\n",
    ))
    .expect("reentrant initializer fixture should parse");
    let initializer_module = Arc::new(
        compile_module(&initializer_syntax.definitions)
            .expect("initializer fixture should compile"),
    );
    let initializer_entry = initializer_module
        .procedure_id("/proc/dynamic_default")
        .expect("initializer entry should exist");
    let path = TypePath::parse("/datum/reentrant").unwrap();
    let mut state = ExecutionState::new();
    state.set_instance_initializers(
        Arc::new(BTreeMap::from([(
            path,
            vec![InstanceInitializer::Program {
                field: field("value"),
                entry: initializer_entry,
            }],
        )])),
        Some(initializer_module),
    );
    state.next_list_collection = 1;

    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/outer").unwrap(),
            &[],
            &mut state,
        ),
        Ok(Value::number(1.0)),
        "a nested initializer collection must retain the caller's live lists",
    );
}

#[test]
fn parallel_lowering_reassembles_results_and_errors_in_source_order() {
    let values = crate::compile::parallel_collect_ordered(257, |index| -> Result<usize, usize> {
        // Perturb completion order without making the test timing-sensitive.
        std::thread::yield_now();
        Ok(index * 3)
    })
    .unwrap();
    assert_eq!(values, (0..257).map(|index| index * 3).collect::<Vec<_>>());

    let error = crate::compile::parallel_collect_ordered(257, |index| {
        if matches!(index, 17 | 201) {
            Err(index)
        } else {
            Ok(index)
        }
    })
    .unwrap_err();
    assert_eq!(error, 17, "the earliest source diagnostic must win");
}

#[test]
#[ignore = "manual multicore lowering throughput benchmark"]
fn benchmark_parallel_procedure_lowering() {
    let source = (0..4_096)
            .map(|index| {
                format!(
                    "/proc/worker_{index}(value)\n\tvar/total = 0\n\tfor(var/i in 1 to 20)\n\t\ttotal += value * i + {index}\n\treturn total\n"
                )
            })
            .collect::<String>();
    let syntax = parse(&source).expect("benchmark source should parse");
    let started = Instant::now();
    let module = compile_module(&syntax.definitions).expect("benchmark module should compile");
    eprintln!(
        "parallel-lowering-benchmark procedures={} elapsed_ms={} workers={}",
        module.paths.len(),
        started.elapsed().as_millis(),
        std::thread::available_parallelism().map_or(1, usize::from),
    );
}

#[test]
fn dynamic_new_resolves_a_registered_textual_type_path() {
    let syntax = parse("/proc/build(kind)\n\treturn new kind\n")
        .expect("textual dynamic-new fixture should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("textual dynamic-new fixture should compile");
    let path = TypePath::parse("/obj/item/gun/ballistic/revolver/mateba").unwrap();
    let mut state = ExecutionState::new();
    state.set_type_paths([path.clone()]);

    let result = execute_in_state(&program, &[Value::text(path.as_str())], &mut state)
        .expect("a registered textual type path should construct dynamically");
    let Value::Datum(datum) = result else {
        panic!("dynamic new should return a datum");
    };
    assert_eq!(state.heap().datum(datum).unwrap().type_path(), &path);
}

#[test]
fn dynamic_new_rejects_an_unknown_textual_type_path() {
    let syntax = parse("/proc/build(kind)\n\treturn new kind\n")
        .expect("unknown textual dynamic-new fixture should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("unknown textual dynamic-new fixture should compile");
    let mut state = ExecutionState::new();

    let error = execute_in_state(
        &program,
        &[Value::text("/obj/item/not_registered")],
        &mut state,
    )
    .expect_err("an unknown textual type path must not synthesize a type");
    assert!(error.message.contains("new requires a type path"));
}

#[test]
fn stale_list_assignments_treated_as_null_receivers() {
    let program = manual_program(
        vec![
            Instruction::LoadLocal(0),
            Instruction::PushNumber(DmNumberBits::from_f32(1.0)),
            Instruction::PushNumber(DmNumberBits::from_f32(7.0)),
            Instruction::SetListIndex,
            Instruction::Return,
        ],
        1,
    );
    let mut state = ExecutionState::new();
    let stale_list = state.heap_mut().allocate_list();
    state.heap_mut().destroy_list(stale_list).unwrap();
    let error = execute_in_state(&program, &[Value::List(stale_list)], &mut state)
        .expect_err("stale list assignments should match null receiver behavior");

    assert_eq!(error.message, "list assignment received null");
    assert!(error.instruction >= 1 && error.instruction <= 3);
    assert!(!error.message.contains("stale"));
}

#[test]
fn stale_list_keep_assignments_match_null_receiver_behavior() {
    let syntax = parse("/proc/run(list/targets)\n\tvar/value = (targets[1] = 7)\n\treturn value")
        .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("module should compile");
    let entry = module
        .procedure_id("/proc/run")
        .expect("entry should exist");
    let mut state = ExecutionState::new();
    let stale_list = state.heap_mut().allocate_list();
    state.heap_mut().destroy_list(stale_list).unwrap();

    let error = execute_module_in_state(&module, entry, &[Value::List(stale_list)], &mut state)
        .unwrap_err();

    assert_eq!(error.message, "list assignment received null");
    assert!(!error.message.contains("stale"));
}

#[test]
fn stale_list_mutation_treated_as_null_receiver() {
    let syntax =
        parse("/proc/run(list/targets)\n\ttargets[1]++\n\treturn 1").expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("module should compile");
    let mut state = ExecutionState::new();
    let stale_list = state.heap_mut().allocate_list();
    state.heap_mut().destroy_list(stale_list).unwrap();
    let entry = module
        .procedure_id("/proc/run")
        .expect("entry should exist");

    let error = execute_module_in_state(&module, entry, &[Value::List(stale_list)], &mut state)
        .unwrap_err();

    assert_eq!(
        error.message,
        "list mutation requires a list, received null"
    );
    assert!(!error.message.contains("stale"));
}

#[test]
fn stale_list_length_treated_as_null() {
    let syntax =
        parse("/proc/run(list/targets)\n\treturn length(targets)").expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("module should compile");
    let entry = module
        .procedure_id("/proc/run")
        .expect("entry should exist");
    let mut state = ExecutionState::new();
    let stale_list = state.heap_mut().allocate_list();
    state.heap_mut().destroy_list(stale_list).unwrap();

    let result = execute_module_in_state(&module, entry, &[Value::List(stale_list)], &mut state)
        .expect("stale list should execute as null in length checks");

    assert_eq!(result, Value::number(0.0));
}

#[test]
fn stale_list_prepare_iteration_treated_as_null_receiver() {
    let syntax = parse("/proc/run(list/items)\n\tvar/total = 0\n\tfor(var/item in items)\n\t\ttotal++\n\treturn total")
            .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("module should compile");
    let entry = module
        .procedure_id("/proc/run")
        .expect("entry should exist");
    let mut state = ExecutionState::new();
    let stale_list = state.heap_mut().allocate_list();
    state.heap_mut().destroy_list(stale_list).unwrap();

    let result = execute_module_in_state(&module, entry, &[Value::List(stale_list)], &mut state)
        .expect("stale list should execute as null iterable");

    assert_eq!(result, Value::number(0.0));
}

#[test]
fn stale_datum_mutate_field_treated_as_null_receiver() {
    let syntax = parse(
        "/proc/run()\n\tvar/obj/target = new /obj\n\tdel(target)\n\ttarget.layer += 1\n\treturn 1",
    )
    .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("module should compile");
    let entry = module
        .procedure_id("/proc/run")
        .expect("entry should exist");
    let error = execute_module(&module, entry, &[]).unwrap_err();

    assert_eq!(error.message, "field read received null");
    assert!(!error.message.contains("stale"));
}

#[test]
fn list_instructions_consume_the_existing_shared_budget() {
    let program = manual_program(
        vec![
            Instruction::PushNumber(DmNumberBits::from_f32(1.0)),
            Instruction::MakeList(1),
            Instruction::Return,
        ],
        0,
    );
    let mut state = ExecutionState::new();
    let error = execute_with_limits_in_state(
        &program,
        &[],
        ExecutionLimits {
            max_steps: 2,
            ..ExecutionLimits::default()
        },
        &mut state,
    )
    .expect_err("Return must require its own instruction-budget unit");

    assert_eq!(error.message, "instruction budget of 2 exhausted");
    assert_eq!(error.instruction, 2);
    assert_eq!(error.source_span, Some(SourceSpan::new(20, 21)));
}

#[test]
fn compiles_locals_and_executes_binary32_arithmetic() {
    let source = "/proc/probe(input)\n\tvar/doubled = input * 2\n\treturn doubled + 3\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");
    let result = execute(&program, &[Value::number(4.0)]).expect("procedure should execute");

    assert_eq!(result, Value::number(11.0));
    assert_eq!(program.instructions.len(), program.source_spans.len());
    // This procedure never observes `args`, so execution begins directly
    // with its first source statement.
    assert_eq!(program.source_spans[0], syntax.definitions[0].body[0].span);
}

#[test]
fn observes_operator_precedence_and_parentheses() {
    let result = execute_source("/proc/probe(input)\n\treturn (input + 3) * 2\n", 4.0);

    assert_eq!(result, Value::number(14.0));
}

#[test]
fn parenthesized_expression_statements_discard_their_result() {
    let syntax =
        parse("/proc/probe(input)\n\tvar/value = 0\n\t((value = input + 1))\n\treturn value\n")
            .expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("parenthesized expression statement should compile");

    assert_eq!(
        execute(&program, &[Value::number(41.0)]),
        Ok(Value::number(42.0))
    );
    assert!(
        program
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::Pop))
    );
}

#[test]
fn executes_bitwise_operators_with_dm_integer_coercion_and_precedence() {
    let source = "/proc/probe()\n\treturn (0xFFFFFF & 6) + (7 ^ 3 | 8) + (9.9 & 3)\n";
    let syntax = parse(source).expect("source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("bitwise expressions should compile");

    // -1 & 6 = 6, (7 ^ 3) | 8 = 12, and 9.9 truncates to 9 before
    // bitwise conjunction with 3, giving 1.
    assert_eq!(execute(&program, &[]), Ok(Value::number(19.0)));
    assert!(program.instructions.iter().any(|instruction| {
        matches!(
            instruction,
            Instruction::BitAnd | Instruction::BitOr | Instruction::BitXor
        )
    }));
}

#[test]
fn executes_unary_bitwise_complement_with_dm_integer_coercion() {
    let source = "/proc/probe()\n\treturn ~9.9 + ~0\n";
    let syntax = parse(source).expect("source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("unary bitwise complement should compile");

    // ~9 and ~0 are 24-bit complements. Their binary32 sum rounds to
    // the nearest representable value at this magnitude.
    assert_eq!(execute(&program, &[]), Ok(Value::number(33_554_420.0)));
    assert!(
        program
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::BitNot))
    );
}

#[test]
fn bitwise_compound_assignments_update_locals_and_list_indices() {
    let source = "/proc/probe(items)\n\tvar/value = 14\n\tvalue &= 11\n\tvalue |= 16\n\titems[1] ^= value\n\treturn items[1]\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("bitwise compound assignments should compile");
    let mut state = ExecutionState::new();
    let list = state.heap.allocate_list();
    state.heap.list_mut(list).unwrap().add(Value::number(7.0));

    // ((14 & 11) | 16) = 26; 7 ^ 26 = 29.
    assert_eq!(
        execute_in_state(&program, &[Value::List(list)], &mut state),
        Ok(Value::number(29.0))
    );
}

#[test]
fn shift_operators_and_compound_assignments_use_byond_24_bit_semantics() {
    let source = "/proc/probe(items)\n\tvar/value = 3 << 2\n\tvalue >>= 1\n\titems[1] <<= value\n\treturn (8 >> 2) + items[1] + (1 << 33)\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("shift expressions and assignments should compile");
    let mut state = ExecutionState::new();
    let list = state.heap.allocate_list();
    state.heap.list_mut(list).unwrap().add(Value::number(1.0));

    // value is (3 << 2) >> 1 = 6; item becomes 1 << 6 = 64.
    // 8 >> 2 is 2, and counts >=24 shift every effective bit away.
    assert_eq!(
        execute_in_state(&program, &[Value::List(list)], &mut state),
        Ok(Value::number(66.0))
    );
    assert!(program.instructions.iter().any(|instruction| {
        matches!(
            instruction,
            Instruction::ShiftLeft | Instruction::ShiftRight
        )
    }));
}

#[test]
fn documented_pure_standard_procs_cover_sort_params_and_number_text() {
    let source = parse(
            "/proc/probe()\n\tvar/list/p = params2list(\"a=one+two&b=%26\")\n\tif(p[\"a\"] != \"one two\" || p[\"b\"] != \"&\")\n\t\treturn 0\n\tif(list2params(p) != \"a=one+two&b=%26\")\n\t\treturn 0\n\tif(lentext(\"abc\") != 3)\n\t\treturn 0\n\tif(sorttext(\"A\", \"b\") != 1 || sorttextEx(\"a\", \"B\") != -1 || sorttext(null, /obj) != 1)\n\t\treturn 0\n\tif(num2text(11, 2, 16) != \"0b\")\n\t\treturn 0\n\treturn 1\n",
        )
        .expect("pure standard-proc source should parse");
    let module = compile_module(&source.definitions).expect("pure standard procs should compile");
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/probe").unwrap(), &[]),
        Ok(Value::number(1.0))
    );
}

#[test]
fn host_decoded_world_params_support_membership_and_associated_values() {
    let source = parse(
            "/proc/probe(list/params)\n\treturn (\"no-init\" in params) && params[\"mode\"] == \"fast start\"\n",
        )
        .expect("world parameter probe should parse");
    let program =
        compile_procedure(&source.definitions[0]).expect("world parameter probe should compile");
    let mut state = ExecutionState::new();
    let params = state
        .decode_params_list("no-init&mode=fast+start")
        .expect("valid world parameters should decode");

    assert_eq!(
        execute_in_state(&program, &[params], &mut state),
        Ok(Value::number(1.0))
    );
}

#[test]
fn list2params_omits_equals_for_positional_entries() {
    let source = parse("/proc/probe()\n\treturn list2params(list(\"alpha beta\", 2))\n")
        .expect("list2params positional source should parse");
    let module =
        compile_module(&source.definitions).expect("list2params positional source should compile");
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/probe").unwrap(), &[]),
        Ok(Value::text("alpha+beta&2"))
    );
}

#[test]
fn bespoke_element_ids_keep_positional_and_named_arguments_distinct() {
    let source = parse(
            "/proc/element_id(tool)\n\tvar/list/fullid = list(\"/datum/element/processable\")\n\tvar/list/named = list()\n\tfullid += \"[tool]\"\n\tnamed[\"table_required\"] = 1\n\tfullid += named\n\treturn list2params(fullid)\n/proc/probe()\n\tvar/knife = element_id(1)\n\tvar/saw = element_id(2)\n\treturn knife != saw && knife == element_id(1)\n",
        )
        .expect("bespoke element id source should parse");
    let module =
        compile_module(&source.definitions).expect("bespoke element id source should compile");
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/probe").unwrap(), &[]),
        Ok(Value::number(1.0))
    );
}

#[test]
fn associative_cache_assignment_keeps_distinct_dynamic_new_keys() {
    let source = parse(
            "/proc/probe()\n\tvar/list/cache = list()\n\tvar/kind = /datum/example\n\tvar/first = cache[\"knife\"]\n\tif(first)\n\t\treturn -1\n\tfirst = cache[\"knife\"] = new kind\n\tvar/second = cache[\"saw\"]\n\tif(second)\n\t\treturn -2\n\tsecond = cache[\"saw\"] = new kind\n\treturn first != second\n",
        )
        .expect("associative cache source should parse");
    let module =
        compile_module(&source.definitions).expect("associative cache source should compile");
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/probe").unwrap(), &[]),
        Ok(Value::number(1.0))
    );
}

#[test]
fn stack_trace_helper_reports_without_aborting_its_caller() {
    let source = parse(
            "/proc/_stack_trace(message)\n\tCRASH(message)\n/proc/probe()\n\t_stack_trace(\"diagnostic only\")\n\treturn 42\n/proc/direct()\n\tCRASH(\"fatal\")\n",
        )
        .expect("stack trace helper source should parse");
    let module =
        compile_module(&source.definitions).expect("stack trace helper source should compile");
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/probe").unwrap(), &[]),
        Ok(Value::number(42.0))
    );
    assert!(
        execute_module(&module, module.procedure_id("/proc/direct").unwrap(), &[]).is_err(),
        "direct CRASH must remain fatal to its execution"
    );
}

#[test]
fn stale_send_signal_callback_aborts_only_the_signal_proc() {
    let source = parse(
            "/datum/proc/_SendSignal()\n\tvar/list/missing\n\treturn missing[\"callback\"]\n/proc/probe()\n\tvar/datum/target = new\n\ttarget._SendSignal()\n\treturn 42\n",
        )
        .expect("signal recovery source should parse");
    let module =
        compile_module(&source.definitions).expect("signal recovery source should compile");
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/probe").unwrap(), &[]),
        Ok(Value::number(42.0)),
        "a stale DCS listener must not unwind the caller or Master scheduler task"
    );
}

#[test]
fn single_signal_unregister_removes_both_reciprocal_edges() {
    let source = parse(
        "/datum/listener/proc/probe(datum/target)\n\
\tvar/list/procs = (_signal_procs ||= list())\n\
\tvar/list/target_procs = (procs[target] ||= list())\n\
\ttarget_procs[\"signal\"] = \"callback\"\n\
\tvar/list/lookup = (target._listen_lookup ||= list())\n\
\tlookup[\"signal\"] = src\n\
\tif(!_signal_procs || !_signal_procs[target] || !lookup)\n\
\t\treturn 900\n\
\tswitch(length(lookup[\"signal\"]))\n\
\t\tif(0)\n\
\t\t\tif(lookup[\"signal\"] != src)\n\
\t\t\t\treturn 800\n\
\t\t\tlookup -= \"signal\"\n\
\t_signal_procs[target] -= \"signal\"\n\
\tif(!_signal_procs[target].len)\n\
\t\t_signal_procs -= target\n\
\treturn !!lookup[\"signal\"] * 100 + !!_signal_procs[target] * 10 + lookup.len\n",
    )
    .expect("reciprocal signal graph source should parse");
    let module = compile_module_specs(&[ProcedureSpec {
        path: "/datum/listener/proc/probe@0".to_owned(),
        definition: &source.definitions[0],
        parent: None,
        static_calls: BTreeMap::new(),
        src_fields: BTreeMap::from([("_signal_procs".to_owned(), field("_signal_procs"))]),
        global_fields: BTreeMap::new(),
    }])
    .expect("reciprocal signal graph source should compile");
    let entry = module.procedure_id_at(0).expect("entry");
    let mut state = ExecutionState::new();
    let listener = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/listener").unwrap());
    let target = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/target").unwrap());
    state
        .heap_mut()
        .set_datum_field(listener, field("_signal_procs"), Value::Null)
        .unwrap();
    state
        .heap_mut()
        .set_datum_field(target, field("_listen_lookup"), Value::Null)
        .unwrap();
    assert_eq!(
        execute_module_in_context(
            &module,
            entry,
            &[Value::Datum(target)],
            &mut state,
            &ExecutionContext::new(Value::Datum(listener), Value::Null),
        ),
        Ok(Value::number(0.0)),
        "single-listener unregister must remove both the target lookup and listener callback"
    );
}

#[test]
fn stale_single_signal_edge_is_removed_when_callback_lookup_is_missing() {
    let source = parse(
        "/datum/proc/_SendSignal(sigtype, list/arguments)\n\
\tvar/target = _listen_lookup[sigtype]\n\
\tif(!length(target))\n\
\t\tvar/datum/listening_datum = target\n\
\t\treturn call(listening_datum, listening_datum._signal_procs[src][sigtype])(arglist(arguments))\n\
/proc/probe(datum/sender, list/arguments)\n\
\treturn sender._SendSignal(\"signal\", arguments)\n",
    )
    .expect("stale DCS source should parse");
    let module = compile_module_specs(&[
        ProcedureSpec {
            path: "/datum/proc/_SendSignal@0".to_owned(),
            definition: &source.definitions[0],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::from([("_listen_lookup".to_owned(), field("_listen_lookup"))]),
            global_fields: BTreeMap::new(),
        },
        ProcedureSpec {
            path: "/proc/probe@1".to_owned(),
            definition: &source.definitions[1],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
    ])
    .expect("stale DCS source should compile");
    let entry = module.procedure_id_at(1).expect("entry");
    let mut state = ExecutionState::new();
    let sender = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/sender").unwrap());
    let listener = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/listener").unwrap());
    let lookup = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(lookup)
        .unwrap()
        .set_key(Value::text("signal"), Value::Datum(listener));
    state
        .heap_mut()
        .set_datum_field(sender, field("_listen_lookup"), Value::List(lookup))
        .unwrap();
    let signal_procs = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .set_datum_field(listener, field("_signal_procs"), Value::List(signal_procs))
        .unwrap();
    let arguments = state.heap_mut().allocate_list();
    assert_eq!(
        execute_module_in_context(
            &module,
            entry,
            &[Value::Datum(sender), Value::List(arguments)],
            &mut state,
            &ExecutionContext::default(),
        ),
        Ok(Value::Null),
    );
    assert_eq!(
        state.heap().datum_field(sender, &field("_listen_lookup")),
        Ok(&Value::Null),
        "repair must remove the stale scalar lookup from the sender"
    );
}

#[test]
fn list_literals_preserve_omitted_interior_arguments_as_null() {
    let source = parse(
            "/proc/probe()\n\tvar/list/values = list(1,,3,)\n\treturn values.len == 3 && values[1] == 1 && isnull(values[2]) && values[3] == 3\n",
        )
        .expect("omitted list argument source should parse");
    let module =
        compile_module(&source.definitions).expect("omitted list argument source should compile");
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/probe").unwrap(), &[]),
        Ok(Value::number(1.0))
    );
}

#[test]
fn initial_field_on_null_returns_null() {
    let source = parse(
        "/proc/probe()\n\tvar/datum/example = null\n\treturn isnull(initial(example.value))\n",
    )
    .expect("null initial source should parse");
    let module = compile_module(&source.definitions).expect("null initial source should compile");
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/probe").unwrap(), &[]),
        Ok(Value::number(1.0))
    );
}

#[test]
fn documented_operator_semantics_cover_short_circuit_modulo_compare_and_equivalence() {
    let source = parse(
            "/proc/probe()\n\tvar/list/a = list(\"key\" = 7, 2)\n\tvar/list/b = list(\"key\" = 7, 2)\n\tvar/list/c = list(\"key\" = 8, 2)\n\tvar/legacy = 5.9 % 2.1\n\tvar/fractional = 5.5 %% 2\n\tlegacy %= 2\n\tfractional %%= 1.25\n\tif((a ~= b) != 1 || (a ~! c) != 1)\n\t\treturn -100\n\tif((3 <=> 4) != -1 || (\"b\" <=> \"a\") != 1 || (1 <> 2) != 1)\n\t\treturn -101\n\tif((99 in null) != 0)\n\t\treturn -102\n\tvar/or_value = \"\" || \"fallback\"\n\tvar/and_value = \"left\" && \"right\"\n\tvar/skip_or = 1 || list()[99]\n\tvar/skip_and = 0 && list()[99]\n\tif(or_value != \"fallback\" || and_value != \"right\" || skip_or != 1 || skip_and != 0)\n\t\treturn -103\n\treturn legacy + fractional\n",
        )
        .expect("documented operator source should parse");
    let module = compile_module(&source.definitions).expect("documented operators should compile");
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/probe").unwrap(), &[]),
        Ok(Value::number(1.25))
    );
}

#[test]
fn bitwise_operators_use_byonds_24_effective_bits() {
    let source = parse(
            "/proc/probe()\n\tvar/a = ~0\n\tvar/b = 1 << 24\n\tvar/c = 0xFFFFFF >> 23\n\treturn a + b + c\n",
        )
        .expect("bitwise source should parse");
    let module = compile_module(&source.definitions).expect("bitwise source should compile");
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/probe").unwrap(), &[]),
        Ok(Value::number(16_777_216.0))
    );
}

#[test]
fn conditional_expressions_associate_right() {
    let source = "/proc/probe(input)\n\treturn input == 1 ? 10 : input == 2 ? 20 : 30\n";
    let syntax = parse(source).expect("source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("conditional expressions should compile");

    assert_eq!(
        execute(&program, &[Value::number(1.0)]),
        Ok(Value::number(10.0))
    );
    assert_eq!(
        execute(&program, &[Value::number(2.0)]),
        Ok(Value::number(20.0))
    );
    assert_eq!(
        execute(&program, &[Value::number(3.0)]),
        Ok(Value::number(30.0))
    );

    let short_circuit =
        parse("/proc/short_circuit()\n\treturn TRUE ? 7 : 1 in 2\n").expect("source should parse");
    let program = compile_procedure(&short_circuit.definitions[0])
        .expect("conditional expressions should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::number(7.0)));

    // `:` is both the conditional delimiter and DreamMaker's dynamic
    // member operator.  A member access in the false arm must not be
    // mistaken for a second conditional delimiter.
    let dynamic_false_arm =
        parse("/proc/dynamic_false_arm(input)\n\treturn input ? 7 : input:type\n")
            .expect("dynamic-member conditional source should parse");
    let program = compile_procedure(&dynamic_false_arm.definitions[0])
        .expect("dynamic member in the false arm should compile");
    assert!(
        execute(&program, &[Value::Null])
            .expect_err("reading a dynamic field from null should fail at runtime")
            .message
            .contains("field read received null")
    );

    let nested_false_arm = parse("/proc/nested(a, b)\n\treturn a ? (b ? 10 : 20) : 30\n")
        .expect("nested conditional source should parse");
    let program = compile_procedure(&nested_false_arm.definitions[0])
        .expect("an outer delimiter after a nested false arm should compile");
    assert_eq!(
        execute(&program, &[Value::number(1.0), Value::number(0.0)]),
        Ok(Value::number(20.0))
    );

    let macro_nested = parse(
            "/proc/macro_nested(a, b, c, d, e, f, g)\n\treturn ((a) ? (b?[\"x\"] ? -9 : (-9) - (((c) ? (d ? e[f] : g) : 0) + 1)) : (-9))\n",
        )
        .expect("macro-expanded nested conditional source should parse");
    compile_procedure(&macro_nested.definitions[0])
        .expect("nested conditional delimiters should remain distinct from dynamic access");

    let kirby_name =
        parse("/proc/kirby_name(dead)\n\treturn \"[dead ? \"dead \":null]potted plant\"\n")
            .expect("Kirby plant interpolation source should parse");
    let program = compile_procedure(&kirby_name.definitions[0])
        .expect("an attached :null must remain the ternary separator");
    assert_eq!(
        execute(&program, &[Value::number(1.0)]),
        Ok(Value::text("dead potted plant")),
    );
    assert_eq!(
        execute(&program, &[Value::number(0.0)]),
        Ok(Value::text("potted plant")),
    );
}

#[test]
fn compiles_in_as_a_relational_list_membership_operator() {
    let source = "/proc/probe(input)\n\treturn input + 1 in list(2, 4, \"key\" = 9)\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");

    assert!(
        program
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::Contains))
    );
    assert_eq!(
        execute(&program, &[Value::number(1.0)]),
        Ok(Value::number(1.0))
    );
    assert_eq!(
        execute(&program, &[Value::number(3.0)]),
        Ok(Value::number(1.0))
    );
    assert_eq!(
        execute(&program, &[Value::number(8.0)]),
        Ok(Value::number(0.0))
    );
}

#[test]
fn in_checks_associative_keys_but_not_associative_values() {
    let source = "/proc/probe(input)\n\treturn input in list(\"key\" = 9)\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");

    assert_eq!(
        execute(&program, &[Value::text("key")]),
        Ok(Value::number(1.0))
    );
    assert_eq!(
        execute(&program, &[Value::number(9.0)]),
        Ok(Value::number(0.0))
    );
}

#[test]
fn in_treats_an_atom_container_as_its_contents_list() {
    let syntax = parse("/proc/probe(needle, container)\n\treturn needle in container\n")
        .expect("atom membership fixture should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("atom membership fixture should compile");
    let mut state = ExecutionState::new();
    let turf = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/turf/floor").unwrap());
    let present = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/machinery/atmospherics/pipe").unwrap());
    let absent = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/machinery/atmospherics/pipe/absent").unwrap());
    let contents = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(contents)
        .unwrap()
        .add(Value::Datum(present));
    state
        .heap_mut()
        .set_datum_field(turf, field("contents"), Value::List(contents))
        .unwrap();
    state
        .heap_mut()
        .set_datum_field(present, field("loc"), Value::Datum(turf))
        .unwrap();

    assert_eq!(
        execute_in_state(
            &program,
            &[Value::Datum(present), Value::Datum(turf)],
            &mut state,
        ),
        Ok(Value::number(1.0)),
        "an adjacent atmos node must be found in the turf returned by get_step",
    );
    assert_eq!(
        execute_in_state(
            &program,
            &[Value::Datum(absent), Value::Datum(turf)],
            &mut state,
        ),
        Ok(Value::number(0.0)),
    );
}

#[test]
fn movable_locs_tracks_location_and_multitile_bounds() {
    let syntax = parse(concat!(
        "/proc/probe(atom/movable/thing, turf/second)\n",
        "\treturn thing.locs.len * 100 + (second in thing.locs) * 10 + (thing.loc in thing.locs)\n",
    ))
    .expect("movable locs fixture should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("movable locs fixture should compile");
    let mut state = ExecutionState::new();
    let atom = TypePath::parse("/atom").unwrap();
    let movable = TypePath::parse("/atom/movable").unwrap();
    let tram = TypePath::parse("/obj/structure/transport/linear/tram").unwrap();
    let turf = TypePath::parse("/turf").unwrap();
    let floor = TypePath::parse("/turf/floor").unwrap();
    state.set_type_parents(BTreeMap::from([
        (atom.clone(), None),
        (movable.clone(), Some(atom.clone())),
        (tram.clone(), Some(movable)),
        (turf.clone(), Some(atom)),
        (floor.clone(), Some(turf)),
    ]));

    let first = state.heap_mut().allocate_datum(floor.clone());
    let second = state.heap_mut().allocate_datum(floor);
    for (datum, x) in [(first, 4.0), (second, 5.0)] {
        for (name, value) in [("x", x), ("y", 7.0), ("z", 2.0)] {
            state
                .heap_mut()
                .set_datum_field(datum, field(name), Value::number(value))
                .unwrap();
        }
    }
    state.world_turfs.insert((4, 7, 2), first);
    state.world_turfs.insert((5, 7, 2), second);

    let thing = state.heap_mut().allocate_datum(tram);
    for (name, value) in [
        ("loc", Value::Datum(first)),
        ("bound_x", Value::number(0.0)),
        ("bound_y", Value::number(0.0)),
        ("bound_width", Value::number(64.0)),
        ("bound_height", Value::number(32.0)),
    ] {
        state
            .heap_mut()
            .set_datum_field(thing, field(name), value)
            .unwrap();
    }

    assert_eq!(
        execute_in_state(
            &program,
            &[Value::Datum(thing), Value::Datum(second)],
            &mut state,
        ),
        Ok(Value::number(211.0)),
        "locs must expose both the base turf and every turf overlapped by bounds",
    );
}

#[test]
fn rejects_unknown_locals_during_compilation() {
    let syntax =
        parse("/proc/probe(input)\n\treturn missing + input\n").expect("source should parse");
    let error = compile_procedure(&syntax.definitions[0])
        .expect_err("unknown local should fail compilation");

    assert!(error.message.contains("unknown local"));
}

#[test]
fn executes_assignment_and_nested_if_else_blocks() {
    let source = "/proc/clamp(input)\n\tvar/result = input\n\tif(result < 0)\n\t\tresult = 0\n\telse\n\t\tif(result > 10)\n\t\t\tresult = 10\n\treturn result\n";

    assert_eq!(execute_source(source, -2.0), Value::number(0.0));
    assert_eq!(execute_source(source, 7.0), Value::number(7.0));
    assert_eq!(execute_source(source, 18.0), Value::number(10.0));
}

#[test]
fn recognizes_when_both_conditional_branches_return() {
    let source = "/proc/sign(input)\n\tif(input < 0)\n\t\treturn -1\n\telse\n\t\treturn 1\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");

    assert_eq!(
        execute(&program, &[Value::number(-2.0)]),
        Ok(Value::number(-1.0))
    );
    assert_eq!(
        execute(&program, &[Value::number(2.0)]),
        Ok(Value::number(1.0))
    );
    assert_eq!(program.instructions.len(), program.source_spans.len());
    assert!(!program.instructions.iter().any(
            |instruction| matches!(instruction, Instruction::Jump(target) if *target >= program.instructions.len())
        ));
}

#[test]
fn production_shaped_get_voice_omits_dead_if_else_end_jump() {
    let source = "/atom/movable/virtualspeaker/proc/GetVoice(bool)\n\tif(bool && realvoice)\n\t\treturn realvoice\n\telse\n\t\treturn \"[src]\"\n";
    let syntax = parse(source).expect("production-shaped GetVoice source should parse");
    let program = compile_procedure_with_resolver_and_fields(
        &syntax.definitions[0],
        &HashMap::new(),
        &BTreeMap::from([("realvoice".to_owned(), field("realvoice"))]),
        &BTreeMap::new(),
        &BTreeMap::new(),
    )
    .expect("production-shaped GetVoice should compile");

    assert_eq!(program.instructions.len(), 14);
    assert!(!program.instructions.iter().any(
            |instruction| matches!(instruction, Instruction::Jump(target) if *target >= program.instructions.len())
        ));
}

#[test]
fn calls_forward_declared_procedures_with_positional_arguments() {
    let source = "/proc/main(input)\n\treturn add(input, 3)\n/proc/add(left, right)\n\treturn left + right\n";
    let syntax = parse(source).expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("module should compile");
    let entry = module
        .procedure_id("/proc/main")
        .expect("entry procedure should exist");

    assert_eq!(
        execute_module(&module, entry, &[Value::number(8.0)]),
        Ok(Value::number(11.0))
    );
}

#[test]
fn arglist_expands_inside_static_and_dynamic_call_arguments() {
    let static_source = parse(
            "/proc/entry()\n\treturn combine(1, arglist(list(2, 3)), 4)\n/proc/combine(a, b, c, d)\n\treturn a + b + c + d\n",
        )
        .expect("static arglist source should parse");
    let module = compile_module(&static_source.definitions).expect("static arglist should compile");
    let entry = module
        .procedure_id("/proc/entry")
        .expect("entry should resolve");
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(10.0)));

    let dynamic_source = parse(
            "/datum/receiver/proc/entry()\n\treturn call(src, \"combine\")(1, arglist(list(2, 3)), 4)\n/datum/receiver/proc/combine(a, b, c, d)\n\treturn a + b + c + d\n",
        )
        .expect("dynamic arglist source should parse");
    let module = compile_module_specs(&[
        ProcedureSpec {
            path: "/datum/receiver/proc/entry@0".to_owned(),
            definition: &dynamic_source.definitions[0],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
        ProcedureSpec {
            path: "/datum/receiver/proc/combine@0".to_owned(),
            definition: &dynamic_source.definitions[1],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
    ])
    .expect("dynamic arglist should compile");
    let mut state = ExecutionState::new();
    let receiver = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/receiver").expect("type path"));
    assert_eq!(
        execute_module_in_context(
            &module,
            module.procedure_id_at(0).expect("entry should resolve"),
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(receiver), Value::Null),
        ),
        Ok(Value::number(10.0))
    );
}

#[test]
fn arglist_null_expands_to_zero_callback_arguments() {
    let source = parse(
            "/proc/invoke(arguments)\n\treturn call(/proc/target)(arglist(arguments))\n/proc/target(value = 9)\n\treturn value\n",
        )
        .expect("callback-shaped arglist source should parse");
    let module = compile_module(&source.definitions).expect("arglist(null) should compile");
    let entry = module.procedure_id("/proc/invoke").expect("invoke proc");
    assert_eq!(
        execute_module(&module, entry, &[Value::Null]),
        Ok(Value::number(9.0))
    );
}

#[test]
fn arglist_expands_associative_component_arguments_to_their_values() {
    // Monkestation's AddComponent macro first captures named arguments in
    // a list, then Component.New copies them and forwards that list with
    // arglist(). Ordinary list iteration yields the associative key, but
    // arglist must supply its value to Initialize.
    let source = parse(
            "/datum/component\n\
             \tvar/unobserved_flags = 0\n\
             /datum/component/proc/New(list/raw_args)\n\
             \tvar/list/arguments = raw_args.Copy(2)\n\
             \tsrc.Initialize(arglist(arguments))\n\
             /datum/component/proc/Initialize(list/initial_reagents = list(9), unobserved_flags = 0)\n\
             \tsrc.unobserved_flags = unobserved_flags + initial_reagents.len\n\
             /datum/component/proc/RegisterWithParent()\n\
             \treturn src.unobserved_flags & 5\n\
             /proc/run()\n\
             \tvar/datum/component/value = new /datum/component(list(null, unobserved_flags = 5))\n\
             \treturn value.RegisterWithParent()\n",
        )
        .expect("component-shaped arglist source should parse");
    let procedures = source
        .definitions
        .iter()
        .filter(|definition| matches!(definition.kind, DefinitionKind::Procedure))
        .cloned()
        .collect::<Vec<_>>();
    let module =
        compile_module(&procedures).expect("component-shaped arglist source should compile");
    let entry = module.procedure_id("/proc/run").expect("run should link");

    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(4.0)));
}

#[test]
fn executes_recursive_calls_on_explicit_frames() {
    let source = "/proc/factorial(input)\n\tif(input <= 1)\n\t\treturn 1\n\treturn input * factorial(input - 1)\n";
    let syntax = parse(source).expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("module should compile");
    let entry = module
        .procedure_id("/proc/factorial")
        .expect("entry procedure should exist");

    assert_eq!(
        execute_module(&module, entry, &[Value::number(5.0)]),
        Ok(Value::number(120.0))
    );
}

#[test]
fn binds_missing_arguments_to_null_and_retains_extra_arguments() {
    let source = "/proc/identity(input)\n\treturn input\n";
    let syntax = parse(source).expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("module should compile");
    let entry = module
        .procedure_id("/proc/identity")
        .expect("entry procedure should exist");

    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::Null));
    assert_eq!(
        execute_module(&module, entry, &[Value::number(7.0), Value::number(99.0)]),
        Ok(Value::number(7.0))
    );
}

#[test]
fn bounds_recursion_and_reports_the_source_mapped_call_stack() {
    let source = "/proc/recurse(input)\n\treturn recurse(input)\n";
    let syntax = parse(source).expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("module should compile");
    let entry = module
        .procedure_id("/proc/recurse")
        .expect("entry procedure should exist");
    let error = execute_module_with_limits(
        &module,
        entry,
        &[Value::number(1.0)],
        ExecutionLimits {
            max_call_depth: 3,
            ..ExecutionLimits::default()
        },
    )
    .expect_err("unbounded recursion should reach the explicit limit");

    assert!(error.message.contains("maximum call depth 3"));
    assert_eq!(error.call_stack.len(), 3);
    assert!(error.source_span.is_some());
    assert!(
        error
            .call_stack
            .iter()
            .all(|trace| trace.procedure == "/proc/recurse" && trace.source_span.is_some())
    );
}

#[test]
fn maps_callee_runtime_errors_and_preserves_caller_context() {
    let source = "/proc/main()\n\treturn broken()\n/proc/broken()\n\treturn \"text\" + 1\n";
    let syntax = parse(source).expect("source should parse");
    let expected_span = syntax.definitions[1].body[0].span;
    let module = compile_module(&syntax.definitions).expect("module should compile");
    let entry = module
        .procedure_id("/proc/main")
        .expect("entry procedure should exist");
    let error =
        execute_module(&module, entry, &[]).expect_err("numeric operation on text should fail");

    assert!(
        error
            .message
            .contains("addition requires compatible DM values")
    );
    assert_eq!(error.source_span, Some(expected_span));
    assert_eq!(error.call_stack.len(), 2);
    assert_eq!(error.call_stack[0].procedure, "/proc/main");
    assert_eq!(error.call_stack[1].procedure, "/proc/broken");
    assert_eq!(error.call_stack[1].source_span, Some(expected_span));
    let diagnostic = error.to_string();
    assert!(diagnostic.contains("call stack:\n  /proc/broken at instruction"));
    assert!(diagnostic.contains("\n  /proc/main at instruction"));
    assert!(diagnostic.contains(&format!(
        "(source {}..{})",
        expected_span.start, expected_span.end
    )));
}

#[test]
fn current_call_uses_explicit_positional_arguments() {
    let source =
        "/proc/countdown(value)\n\tif(value <= 0)\n\t\treturn 0\n\treturn 1 + .(value - 1)\n";
    let syntax = parse(source).expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("module should compile");
    let entry = module
        .procedure_id("/proc/countdown")
        .expect("entry procedure should exist");

    assert_eq!(
        execute_module(&module, entry, &[Value::number(4.0)]),
        Ok(Value::number(4.0))
    );
}

#[test]
fn argumentless_current_call_reuses_original_frame_arguments() {
    let source = "/proc/recurse(value, stop)\n\tstop = 1\n\treturn .()\n";
    let syntax = parse(source).expect("source should parse");
    let call_span = syntax.definitions[0].body[1].span;
    let module = compile_module(&syntax.definitions).expect("module should compile");
    let entry = module
        .procedure_id("/proc/recurse")
        .expect("entry procedure should exist");
    let error = execute_module_with_limits(
        &module,
        entry,
        &[Value::number(7.0), Value::Null, Value::number(99.0)],
        ExecutionLimits {
            max_call_depth: 4,
            ..ExecutionLimits::default()
        },
    )
    .expect_err("reused original arguments should keep recursing");

    assert!(error.message.contains("maximum call depth 4"));
    assert_eq!(error.source_span, Some(call_span));
    assert_eq!(error.call_stack.len(), 4);
    assert!(error.call_stack.iter().all(|trace| {
        trace.procedure == "/proc/recurse" && trace.source_span == Some(call_span)
    }));
}

#[test]
fn unresolved_parent_call_reports_source_mapped_runtime_error() {
    let syntax = parse("/proc/child()\n\treturn ..()\n").expect("source should parse");
    let span = syntax.definitions[0].body[0].span;
    let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");
    let error = execute(&program, &[]).expect_err("unresolved parent should fail at runtime");

    assert_eq!(
        error.message,
        "parent procedure call has no resolved target"
    );
    assert_eq!(error.source_span, Some(span));
}

#[test]
fn while_supports_zero_and_multiple_iterations() {
    let source = "/proc/count(limit)\n\tvar/result = 0\n\twhile(result < limit)\n\t\tresult = result + 1\n\treturn result\n";

    assert_eq!(execute_source(source, 0.0), Value::number(0.0));
    assert_eq!(execute_source(source, 5.0), Value::number(5.0));
}

#[test]
fn do_while_executes_before_testing_and_routes_loop_control_to_condition() {
    let source = "/proc/count(limit)\n\tvar/result = 0\n\tdo\n\t\tresult = result + 1\n\t\tif(result == 2)\n\t\t\tcontinue\n\t\tif(result > limit)\n\t\t\tbreak\n\twhile(result <= limit)\n\treturn result\n";

    // A post-test loop enters even when the condition will be false.
    assert_eq!(execute_source(source, 0.0), Value::number(1.0));
    // `continue` tests the condition, and `break` exits without testing.
    assert_eq!(execute_source(source, 3.0), Value::number(4.0));
}

#[test]
fn do_while_accepts_byond_single_statement_body() {
    // The DM reference defines the body as a Statement, which may be a
    // block or one statement. One level of indentation is sufficient; it
    // does not require a nested multi-line block.
    let source = "/proc/count(limit)\n\tvar/result = 0\n\tdo\n\t\tresult += 1\n\twhile(result < limit)\n\treturn result\n";

    assert_eq!(execute_source(source, 0.0), Value::number(1.0));
    assert_eq!(execute_source(source, 4.0), Value::number(4.0));
}

#[test]
fn do_while_accepts_multiline_braced_macro_body() {
    // Continued macros and generated DM commonly spell statement blocks
    // with braces. The lexer retains the whole delimited region as one
    // logical line, then compact-statement normalization must recover the
    // same structure as an indented DM block.
    let source = "/proc/count(limit)\n\tvar/result = 0\n\tdo {\n\t\tresult += 1;\n\t\tif(result == 2) {\n\t\t\tcontinue;\n\t\t}\n\t\tif(result > limit) {\n\t\t\tbreak;\n\t\t}\n\t} while(result <= limit)\n\treturn result\n";

    assert_eq!(execute_source(source, 0.0), Value::number(1.0));
    assert_eq!(execute_source(source, 3.0), Value::number(4.0));
}

#[test]
fn conditional_accepts_inline_braced_do_while_macro_statement() {
    let source = "/proc/run(enabled)\n\tvar/result = 0\n\tif(enabled) do { result += 2; } while(0); result += 1\n\treturn result\n";

    assert_eq!(execute_source(source, 0.0), Value::number(1.0));
    assert_eq!(execute_source(source, 1.0), Value::number(3.0));
}

#[test]
fn switch_matches_values_ranges_and_default_after_evaluating_selector_once() {
    let source = "/proc/classify(value)\n\tvar/calls = 0\n\tswitch(value + 0)\n\t\tif(1, 3)\n\t\t\treturn 10\n\t\tif(4 to 6)\n\t\t\treturn 20\n\t\telse\n\t\t\treturn 30\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("switch should compile");

    assert_eq!(
        execute(&program, &[Value::number(1.0)]),
        Ok(Value::number(10.0))
    );
    assert_eq!(
        execute(&program, &[Value::number(3.0)]),
        Ok(Value::number(10.0))
    );
    assert_eq!(
        execute(&program, &[Value::number(4.0)]),
        Ok(Value::number(20.0))
    );
    assert_eq!(
        execute(&program, &[Value::number(6.0)]),
        Ok(Value::number(20.0))
    );
    assert_eq!(
        execute(&program, &[Value::number(7.0)]),
        Ok(Value::number(30.0))
    );
}

#[test]
fn switch_rejects_case_after_default() {
    let source = "/proc/invalid(value)\n\tswitch(value)\n\t\telse\n\t\t\treturn 1\n\t\tif(2)\n\t\t\treturn 2\n";
    let syntax = parse(source).expect("source should parse");
    let error =
        compile_procedure(&syntax.definitions[0]).expect_err("case after default must not compile");

    assert_eq!(error.message, "switch case cannot follow an else default");
}

#[test]
fn do_requires_indented_body_and_trailing_while() {
    for (source, expected) in [
        (
            "/proc/invalid()\n\tdo\n",
            "do statement requires an indented body",
        ),
        (
            "/proc/invalid()\n\tdo\n\t\treturn 1\n",
            "do statement requires a trailing while condition",
        ),
    ] {
        let syntax = parse(source).expect("source should parse");
        let error = compile_procedure(&syntax.definitions[0])
            .expect_err("invalid do loop should not compile");

        assert_eq!(error.message, expected);
    }
}

#[test]
fn break_and_continue_work_inside_nested_conditionals() {
    let source = "/proc/filter(limit)\n\tvar/index = 0\n\tvar/total = 0\n\twhile(index < limit)\n\t\tindex = index + 1\n\t\tif(index == 2)\n\t\t\tcontinue\n\t\tif(index > 4)\n\t\t\tbreak\n\t\ttotal = total + index\n\treturn total\n";
    let syntax = parse(source).expect("source should parse");
    let while_span = syntax.definitions[0].body[2].span;
    let continue_span = syntax.definitions[0].body[5].span;
    let break_span = syntax.definitions[0].body[7].span;
    let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");

    assert_eq!(
        execute(&program, &[Value::number(10.0)]),
        Ok(Value::number(8.0))
    );
    assert_eq!(program.instructions.len(), program.source_spans.len());
    assert!(program.instructions.iter().zip(&program.source_spans).any(
        |(instruction, span)| matches!(instruction, Instruction::JumpIfFalse(_))
            && *span == while_span
    ));
    assert!(program.instructions.iter().zip(&program.source_spans).any(
        |(instruction, span)| matches!(instruction, Instruction::Jump(_)) && *span == continue_span
    ));
    assert!(program.instructions.iter().zip(&program.source_spans).any(
        |(instruction, span)| matches!(instruction, Instruction::Jump(_)) && *span == break_span
    ));
}

#[test]
fn nested_loops_patch_break_and_continue_to_the_innermost_loop() {
    let source = "/proc/nested(limit)\n\tvar/outer = 0\n\tvar/total = 0\n\twhile(outer < limit)\n\t\touter = outer + 1\n\t\tvar/inner = 0\n\t\twhile(inner < 5)\n\t\t\tinner = inner + 1\n\t\t\tif(inner == 2)\n\t\t\t\tcontinue\n\t\t\tif(inner == 4)\n\t\t\t\tbreak\n\t\t\ttotal = total + 1\n\treturn total\n";

    assert_eq!(execute_source(source, 3.0), Value::number(6.0));
}

#[test]
fn labeled_loop_break_exits_the_selected_loop() {
    let source = "/proc/run()\n\tvar/result = 0\n\touter:\n\t\tfor(var/x in 1 to 3)\n\t\t\tfor(var/y in 1 to 3)\n\t\t\t\tresult += 1\n\t\t\t\tbreak outer\n\treturn result\n";

    assert_eq!(execute_source(source, 0.0), Value::number(1.0));
}

#[test]
fn labeled_loop_continue_advances_the_selected_loop() {
    let source = "/proc/run()\n\tvar/result = 0\n\touter:\n\t\tfor(var/x in 1 to 3)\n\t\t\tfor(var/y in 1 to 3)\n\t\t\t\tresult += 1\n\t\t\t\tcontinue outer\n\treturn result\n";
    assert_eq!(execute_source(source, 0.0), Value::number(3.0));
}

#[test]
fn rejects_break_and_continue_outside_loops() {
    for (statement, expected) in [
        ("break", "break outside a loop"),
        ("continue", "continue outside a loop"),
    ] {
        let source = format!("/proc/invalid()\n\t{statement}\n");
        let syntax = parse(&source).expect("source should parse");
        let error = compile_procedure(&syntax.definitions[0])
            .expect_err("loop control outside a loop should fail");

        assert_eq!(error.message, expected);
    }
}

#[test]
fn instruction_budget_terminates_an_infinite_while_with_source_context() {
    let source = "/proc/spin()\n\twhile(1)\n\t\tcontinue\n";
    let syntax = parse(source).expect("source should parse");
    let while_span = syntax.definitions[0].body[0].span;
    let module = compile_module(&syntax.definitions).expect("module should compile");
    let entry = module
        .procedure_id("/proc/spin")
        .expect("entry procedure should exist");
    let error = execute_module_with_limits(
        &module,
        entry,
        &[],
        ExecutionLimits {
            max_steps: 6,
            ..ExecutionLimits::default()
        },
    )
    .expect_err("infinite loop should exhaust its instruction budget");

    assert_eq!(error.message, "instruction budget of 6 exhausted");
    assert_eq!(error.source_span, Some(while_span));
    assert_eq!(error.call_stack.len(), 1);
    assert_eq!(error.call_stack[0].procedure, "/proc/spin");
    assert_eq!(error.call_stack[0].source_span, Some(while_span));
}

#[test]
fn exact_standalone_instruction_budget_completes_the_final_return() {
    let source = "/proc/increment(value)\n\treturn value + 1\n";
    let syntax = parse(source).expect("source should parse");
    let return_span = syntax.definitions[0].body[0].span;
    let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");
    let exact_steps = u64::try_from(program.instructions.len())
        .expect("test program instruction count should fit u64");

    assert_eq!(
        execute_with_limits(
            &program,
            &[Value::number(4.0)],
            ExecutionLimits {
                max_steps: exact_steps,
                ..ExecutionLimits::default()
            },
        ),
        Ok(Value::number(5.0))
    );
    let error = execute_with_limits(
        &program,
        &[Value::number(4.0)],
        ExecutionLimits {
            max_steps: exact_steps - 1,
            ..ExecutionLimits::default()
        },
    )
    .expect_err("one fewer step should stop before Return");
    assert_eq!(error.source_span, Some(return_span));
    assert_eq!(error.call_stack[0].procedure, "<standalone>");
}

#[test]
fn instruction_budget_is_shared_across_procedure_calls() {
    let source = "/proc/main()\n\treturn helper()\n/proc/helper()\n\treturn 7\n";
    let syntax = parse(source).expect("source should parse");
    let helper_span = syntax.definitions[1].body[0].span;
    let module = compile_module(&syntax.definitions).expect("module should compile");
    let entry = module
        .procedure_id("/proc/main")
        .expect("entry procedure should exist");

    assert_eq!(
        execute_module_with_limits(
            &module,
            entry,
            &[],
            ExecutionLimits {
                max_steps: 4,
                ..ExecutionLimits::default()
            },
        ),
        Ok(Value::number(7.0))
    );
    let error = execute_module_with_limits(
        &module,
        entry,
        &[],
        ExecutionLimits {
            max_steps: 2,
            ..ExecutionLimits::default()
        },
    )
    .expect_err("caller and callee should consume one shared budget");

    assert_eq!(error.source_span, Some(helper_span));
    assert_eq!(error.call_stack.len(), 2);
    assert_eq!(error.call_stack[0].procedure, "/proc/main");
    assert_eq!(error.call_stack[1].procedure, "/proc/helper");
    assert_eq!(error.call_stack[1].source_span, Some(helper_span));
}

#[test]
fn c_style_for_supports_scoped_initializer_and_postfix_increment() {
    let source = "/proc/sum(limit)\n\tvar/total = 0\n\tfor(var/i = 0; i < limit; i++)\n\t\ttotal = total + i\n\treturn total\n";

    assert_eq!(execute_source(source, 0.0), Value::number(0.0));
    assert_eq!(execute_source(source, 5.0), Value::number(10.0));

    let escaped =
        parse("/proc/invalid()\n\tfor(var/i = 0; i < 1; i++)\n\t\tcontinue\n\treturn i\n")
            .expect("source should parse");
    let error = compile_procedure(&escaped.definitions[0])
        .expect_err("for initializer should be scoped to its loop");
    assert_eq!(error.message, "unknown local \"i\"");

    let comma_source = "/proc/sum(limit)\n\tvar/total = 0\n\tfor(var/i = 0, i < limit, i++)\n\t\ttotal += i\n\treturn total\n";
    assert_eq!(execute_source(comma_source, 5.0), Value::number(10.0));
}

#[test]
fn for_to_range_is_inclusive_and_continue_runs_its_increment() {
    let source = "/proc/sum(first, last)\n\tvar/total = 0\n\tfor(var/i in first to last)\n\t\tif(i == first)\n\t\t\tcontinue\n\t\ttotal += i\n\treturn total\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("range loop should compile");
    assert_eq!(
        program
            .instructions
            .iter()
            .filter(|instruction| matches!(instruction, Instruction::LessEqual))
            .count(),
        1,
        "the implicit +1 step should compile to one inclusive comparison"
    );
    assert!(!program.instructions.iter().any(|instruction| matches!(
        instruction,
        Instruction::GreaterEqual | Instruction::Less | Instruction::And | Instruction::Or
    )));
    assert_eq!(
        execute(&program, &[Value::number(2.0), Value::number(5.0)]),
        Ok(Value::number(12.0))
    );
    assert_eq!(
        execute(&program, &[Value::number(5.0), Value::number(2.0)]),
        Ok(Value::number(0.0))
    );
}

#[test]
fn for_to_range_honors_explicit_positive_and_negative_steps() {
    let source = "/proc/ranges()\n\tvar/total = 0\n\tfor(var/i in 5 to 1 step -2)\n\t\ttotal += i\n\tfor(var/j in 1 to 5 step 2)\n\t\ttotal += j\n\treturn total\n";
    let syntax = parse(source).expect("source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("stepped range loops should compile");

    // 5 + 3 + 1, then 1 + 3 + 5.
    assert_eq!(execute(&program, &[]), Ok(Value::number(18.0)));
}

#[test]
fn for_to_range_evaluates_its_step_once() {
    let source = "/proc/step_once()\n\tvar/step = 2\n\tvar/total = 0\n\tfor(var/i in 1 to 5 step step)\n\t\ttotal += i\n\t\tstep = 1\n\treturn total\n";
    let syntax = parse(source).expect("source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("range step expression should compile");

    assert_eq!(execute(&program, &[]), Ok(Value::number(9.0)));
}

#[test]
fn c_style_for_supports_prefix_decrement_and_optional_clauses() {
    let decrement = "/proc/sum(limit)\n\tvar/total = 0\n\tfor(var/i = limit; i > 0; --i)\n\t\ttotal = total + i\n\treturn total\n";
    assert_eq!(execute_source(decrement, 3.0), Value::number(6.0));

    let optional = "/proc/once()\n\tfor(;;)\n\t\tbreak\n\treturn 9\n";
    let syntax = parse(optional).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::number(9.0)));

    let empty =
        "/proc/count()\n\tvar/i = 0\n\tfor()\n\t\tif(i > 3)\n\t\t\tbreak\n\t\ti++\n\treturn i\n";
    let syntax = parse(empty).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("empty for should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::number(4.0)));

    let one_separator = "/proc/count()\n\tvar/i = 1\n\tvar/count = 0\n\tfor(, i++ <= 3)\n\t\tcount++\n\treturn count\n";
    let syntax = parse(one_separator).expect("source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("short comma for should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::number(3.0)));
}

#[test]
fn for_continue_runs_increment_and_break_exits_the_loop() {
    let source = "/proc/filter(limit)\n\tvar/total = 0\n\tfor(var/i = 0; i < limit; i++)\n\t\tif(i == 1)\n\t\t\tcontinue\n\t\tif(i == 4)\n\t\t\tbreak\n\t\ttotal = total + i\n\treturn total\n";
    let syntax = parse(source).expect("source should parse");
    let for_span = syntax.definitions[0].body[1].span;
    let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");

    assert_eq!(
        execute(&program, &[Value::number(10.0)]),
        Ok(Value::number(5.0))
    );
    assert!(program.instructions.iter().zip(&program.source_spans).any(
        |(instruction, span)| matches!(instruction, Instruction::StoreLocal(_))
            && *span == for_span
    ));
}

#[test]
fn nested_for_loops_patch_control_to_the_innermost_loop() {
    let source = "/proc/nested(limit)\n\tvar/total = 0\n\tfor(var/i = 0; i < limit; i++)\n\t\tfor(var/j = 0; j < 4; j++)\n\t\t\tif(j == 1)\n\t\t\t\tcontinue\n\t\t\tif(j == 3)\n\t\t\t\tbreak\n\t\t\ttotal = total + 1\n\treturn total\n";

    assert_eq!(execute_source(source, 3.0), Value::number(6.0));
}

#[test]
fn infinite_for_obeys_step_budget_and_for_in_compiles() {
    let source = "/proc/spin()\n\tfor(;;)\n\t\tcontinue\n";
    let syntax = parse(source).expect("source should parse");
    let for_span = syntax.definitions[0].body[0].span;
    let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");
    let error = execute_with_limits(
        &program,
        &[],
        ExecutionLimits {
            max_steps: 7,
            ..ExecutionLimits::default()
        },
    )
    .expect_err("infinite for should exhaust its step budget");
    assert_eq!(error.message, "instruction budget of 7 exhausted");
    assert_eq!(error.source_span, Some(for_span));

    let list_iteration = parse("/proc/list_loop(items)\n\tfor(var/item in items)\n\t\tcontinue\n")
        .expect("source should parse");
    let program = compile_procedure(&list_iteration.definitions[0])
        .expect("for-in list iteration should compile");
    assert!(
        program
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::NextLocalListIteration { .. }))
    );
    assert_eq!(
        execute(&program, &[Value::Null]),
        Ok(Value::Null),
        "BYOND treats for-in over null as an empty iteration"
    );
    for non_iterable in [
        Value::number(0.0),
        Value::text("not a container"),
        Value::TypePath(TypePath::parse("/datum/example").unwrap()),
    ] {
        assert_eq!(
            execute(&program, &[non_iterable]),
            Ok(Value::Null),
            "BYOND treats every scalar for-in operand as an empty iteration"
        );
    }
}

#[test]
fn for_in_and_for_to_accept_existing_iterator_locals() {
    let list_source = "/proc/sum()\n\tvar/item\n\tvar/total = 0\n\tfor(item in list(1, 2, 3))\n\t\ttotal += item\n\treturn total\n";
    let syntax = parse(list_source).expect("source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("existing for-in iterator should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::number(6.0)));

    let range_source = "/proc/sum()\n\tvar/item\n\tvar/total = 0\n\tfor(item in 1 to 3)\n\t\ttotal += item\n\treturn total\n";
    let syntax = parse(range_source).expect("source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("existing for-to iterator should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::number(6.0)));

    let assignment_range = "/proc/sum()\n\tvar/total = 0\n\tfor(var/item = 1 to 8 step 3)\n\t\ttotal += item\n\treturn total\n";
    let syntax = parse(assignment_range).expect("source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("assignment-style for-to should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::number(12.0)));

    let empty_range =
        "/proc/check()\n\tvar/item = -1\n\tfor(item = 1 to 0)\n\t\tcontinue\n\treturn item\n";
    let syntax = parse(empty_range).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("empty existing-variable range should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::number(-1.0)));
}

#[test]
fn for_in_snapshots_values_when_the_source_removes_the_current_entry() {
    let source = "/proc/alcohol_shape()\n\tvar/list/containers = list(1, 2, 3, 4)\n\tvar/list/seen = list()\n\tfor(var/typepath in containers)\n\t\tcontainers -= typepath\n\t\tseen += typepath\n\treturn seen.len * 10 + containers.len\n";
    let syntax = parse(source).expect("alcohol initializer shape should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("mutating list iteration should compile");
    assert_eq!(
        execute(&program, &[]),
        Ok(Value::number(40.0)),
        "all four snapshotted entries must be visited and removed",
    );
}

#[test]
fn type_for_loop_enumerates_only_live_matching_datums() {
    let syntax = parse(
        "/proc/count()\n\tvar/total = 0\n\tfor(var/datum/a/item)\n\t\ttotal++\n\treturn total\n",
    )
    .expect("type loop should parse");
    let module = compile_module(&syntax.definitions).expect("type loop should compile");
    let mut state = ExecutionState::new();
    let datum = TypePath::parse("/datum").unwrap();
    let a = TypePath::parse("/datum/a").unwrap();
    let child = TypePath::parse("/datum/a/child").unwrap();
    let b = TypePath::parse("/datum/b").unwrap();
    state.set_type_parents(BTreeMap::from([
        (datum.clone(), None),
        (a.clone(), Some(datum.clone())),
        (child.clone(), Some(a.clone())),
        (b.clone(), Some(datum)),
    ]));
    state.heap_mut().allocate_datum(a);
    state.heap_mut().allocate_datum(child);
    state.heap_mut().allocate_datum(b);
    let entry = module.procedure_id("/proc/count").expect("entry");
    assert_eq!(
        execute_module_in_state(&module, entry, &[], &mut state),
        Ok(Value::number(2.0))
    );
}

#[test]
fn associative_for_loop_binds_keys_values_and_writable_targets() {
    let syntax = parse(
            "/proc/run()\n\tvar/list/items = list(\"a\", \"b\" = 5)\n\tvar/total = 0\n\tfor(var/key, value in items)\n\t\ttotal += (key == \"a\") + value\n\tvar/existing_key\n\tvar/existing_value\n\tfor(existing_key, existing_value in items)\n\t\ttotal += 0\n\tvar/list/out = list(null, null)\n\tfor(out[1], out[2] in items)\n\t\ttotal += 0\n\treturn total + (existing_key == \"b\") + (existing_value == 5) + (out[1] == \"b\") + (out[2] == 5)\n",
        )
        .expect("associative loop should parse");
    let module = compile_module(&syntax.definitions).expect("associative loop should compile");
    let entry = module.procedure_id("/proc/run").expect("entry");
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(10.0)));
}

#[test]
fn exotic_c_style_for_headers_follow_byond_declaration_and_range_fakeouts() {
    let syntax = parse(
            "/proc/run()\n\tvar/out1 = 0\n\tfor(var/x = 2 in 1 to 20; x < 6; x++)\n\t\tout1 += x\n\tvar/out2 = 0\n\tfor(var/y in 1 to 5;)\n\t\tout2 += y\n\tvar/out3 = 0\n\tfor(var/z = 5 in 1 to 20; z < 10)\n\t\tout3 += z\n\t\tout3++\n\t\tif(out3 > 10)\n\t\t\tbreak\n\tvar/out4 = 0\n\tfor(var/a && var/b, a < b + 4, a += 2)\n\t\tout4++\n\treturn out1 * 1000 + out2 * 100 + out3 * 10 + out4\n",
        )
        .expect("exotic loops should parse");
    let module = compile_module(&syntax.definitions).expect("exotic loops should compile");
    let entry = module.procedure_id("/proc/run").expect("entry");
    assert_eq!(
        execute_module(&module, entry, &[]),
        Ok(Value::number(15_622.0))
    );
}

#[test]
fn range_for_can_reuse_bare_and_explicit_src_field_iterators() {
    for iterator in ["idx", "src.idx"] {
        let source = format!(
            "/datum/example/proc/run()\n\tfor({iterator} in 1 to 5)\n\t\tc += idx\n\treturn c\n"
        );
        let syntax = parse(&source).expect("field range loop should parse");
        let fields = BTreeMap::from([
            ("idx".to_owned(), field("idx")),
            ("c".to_owned(), field("c")),
        ]);
        let program = compile_procedure_with_resolver_and_fields(
            &syntax.definitions[0],
            &HashMap::new(),
            &fields,
            &BTreeMap::new(),
            &BTreeMap::new(),
        )
        .expect("field range loop should compile");
        let mut state = ExecutionState::new();
        let src = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/example").unwrap());
        state
            .heap_mut()
            .set_datum_field(src, field("idx"), Value::number(0.0))
            .unwrap();
        state
            .heap_mut()
            .set_datum_field(src, field("c"), Value::number(0.0))
            .unwrap();
        let result = execute_in_context(
            &program,
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(src), Value::Null),
        );
        assert_eq!(result, Ok(Value::number(15.0)));
    }
}

#[test]
fn typed_for_in_binding_ignores_as_qualifier() {
    let source = "/proc/typed_loop()\n\tfor(var/turf/area_turf as anything in list(1))\n\t\tarea_turf = null\n\treturn 7\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("typed for-in binding should use area_turf, not the as qualifier");

    assert_eq!(execute(&program, &[]), Ok(Value::number(7.0)));
}

#[test]
fn typed_for_in_as_anything_keeps_typepath_values() {
    let syntax = parse(
            "/proc/count(list/types)\n\tvar/visited = 0\n\tfor(var/datum/language/language_type as anything in types)\n\t\tvisited++\n\treturn visited\n",
        )
        .expect("language prototype loop should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("language prototype loop should compile");
    let mut state = ExecutionState::new();
    let types = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(types)
        .unwrap()
        .add(Value::TypePath(
            TypePath::parse("/datum/language/common").unwrap(),
        ));
    assert_eq!(
        execute_in_state(&program, &[Value::List(types)], &mut state),
        Ok(Value::number(1.0))
    );
}

#[test]
fn typed_for_in_skips_typepaths_and_scalars_in_mixed_seed_gene_lists() {
    let syntax = parse(
            "/proc/cleanup(list/genes)\n\tvar/visited = 0\n\tfor(var/datum/plant_gene/gene in genes)\n\t\tvisited++\n\treturn visited\n",
        )
        .expect("seed cleanup loop should parse");
    let module = compile_module(&syntax.definitions).expect("seed cleanup loop should compile");
    let entry = module.procedure_id("/proc/cleanup").unwrap();
    let mut state = ExecutionState::new();
    let gene = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/plant_gene").unwrap());
    let genes = state.heap_mut().allocate_list();
    for value in [
        Value::TypePath(TypePath::parse("/datum/plant_gene/trait").unwrap()),
        Value::Datum(gene),
        Value::number(7.0),
    ] {
        state.heap_mut().list_mut(genes).unwrap().add(value);
    }
    assert_eq!(
        execute_module_in_state(&module, entry, &[Value::List(genes)], &mut state),
        Ok(Value::number(1.0))
    );
}

#[test]
fn list_literals_support_bracket_reads_and_writes() {
    let source =
        "/proc/list_access()\n\tvar/items = list(1, 2, 3)\n\titems[2] = 9\n\treturn items[2]\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("lists should compile");

    assert_eq!(execute(&program, &[]), Ok(Value::number(9.0)));
}

#[test]
fn list_assignment_preserves_alias_identity() {
    let source = "/proc/update(items)\n\titems[1] = 12\n\treturn items[1]\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("indexing should compile");
    let mut state = ExecutionState::new();
    let list = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(list)
        .unwrap()
        .add(Value::number(1.0));

    assert_eq!(
        execute_in_state(&program, &[Value::List(list)], &mut state),
        Ok(Value::number(12.0))
    );
    assert!(
        state
            .heap()
            .list(list)
            .unwrap()
            .get(1)
            .unwrap()
            .semantic_eq(&Value::number(12.0))
    );
}

#[test]
fn numeric_index_assignment_rebinds_an_associative_slot_in_place() {
    // Monkestation's /datum/species/proc/on_species_loss rebinds an associative
    // `mutation_index` entry by its numeric position: `L.Find(key)` yields a
    // position, then `L[position] = new_key` rebinds that slot. BYOND keeps the
    // slot and its order, replacing the key and dropping the association.
    let source = concat!(
        "/proc/rebind(mutation_index, default_genes, old_key, new_key)\n",
        "\tvar/location = mutation_index.Find(old_key)\n",
        "\tmutation_index[location] = new_key\n",
        "\tdefault_genes[location] = mutation_index[location]\n",
        "\treturn \"[location]|[mutation_index[location]]|[default_genes[location]]\"\n",
    );
    let syntax = parse(source).expect("species-loss shaped source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("numeric rebind of an associative slot should compile");
    assert!(
        program
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::SetListIndex))
    );

    let mut state = ExecutionState::new();
    let mutation_index = state.heap_mut().allocate_list();
    let default_genes = state.heap_mut().allocate_list();
    for (key, value) in [("a", 1.0), ("b", 2.0), ("c", 3.0)] {
        state
            .heap_mut()
            .list_mut(mutation_index)
            .unwrap()
            .set_key(Value::text(key), Value::number(value));
        state
            .heap_mut()
            .list_mut(default_genes)
            .unwrap()
            .set_key(Value::text(key), Value::text("seq"));
    }

    assert_eq!(
        execute_in_state(
            &program,
            &[
                Value::List(mutation_index),
                Value::List(default_genes),
                Value::text("b"),
                Value::text("q"),
            ],
            &mut state,
        ),
        Ok(Value::text("2|q|q"))
    );

    let values = state.heap().list(mutation_index).unwrap();
    assert_eq!(values.len(), 3);
    assert!(values.get(1).unwrap().semantic_eq(&Value::text("a")));
    assert!(values.get(2).unwrap().semantic_eq(&Value::text("q")));
    assert!(values.get(3).unwrap().semantic_eq(&Value::text("c")));
    assert!(matches!(
        values.get_key(&Value::text("b")),
        Err(ValueError::MissingKey)
    ));
    // The untouched keys keep their associations.
    assert!(
        values
            .get_key(&Value::text("a"))
            .unwrap()
            .semantic_eq(&Value::number(1.0))
    );

    let genes = state.heap().list(default_genes).unwrap();
    assert!(genes.get(2).unwrap().semantic_eq(&Value::text("q")));
    assert!(matches!(
        genes.get_key(&Value::text("b")),
        Err(ValueError::MissingKey)
    ));
}

#[test]
fn compound_list_index_assignment_updates_positional_and_associative_entries() {
    let source = "/proc/update(items)\n\titems[1] += 4\n\titems[\"score\"] *= 3\n\treturn items[1] + items[\"score\"]\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("compound list-index assignments should compile");
    let mut state = ExecutionState::new();
    let list = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(list)
        .unwrap()
        .add(Value::number(2.0));
    state
        .heap_mut()
        .list_mut(list)
        .unwrap()
        .set_key(Value::text("score"), Value::number(5.0));

    assert!(program.instructions.iter().any(|instruction| {
        matches!(
            instruction,
            Instruction::CompoundListIndex(CompoundListIndexOperator::Add)
        )
    }));
    assert_eq!(
        execute_in_state(&program, &[Value::List(list)], &mut state),
        Ok(Value::number(21.0))
    );
    let values = state.heap().list(list).unwrap();
    assert!(values.get(1).unwrap().semantic_eq(&Value::number(6.0)));
    assert!(
        values
            .get_key(&Value::text("score"))
            .unwrap()
            .semantic_eq(&Value::number(15.0))
    );
}

#[test]
fn numeric_index_assignment_at_len_plus_one_appends() {
    let source = "/proc/run()\n\tvar/list/output = list()\n\toutput[length(output) + 1] = \"a\"\n\toutput[length(output) + 1] = \"b\"\n\treturn output.Join()\n";
    let syntax = parse(source).expect("orange-output-shaped source should parse");
    let module = compile_module(&syntax.definitions)
        .expect("orange-output-shaped numeric append should compile");
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/run").unwrap(), &[]),
        Ok(Value::text("ab"))
    );
}

#[test]
fn fractional_sequence_indexes_truncate_toward_zero_for_read_write_and_text() {
    let syntax = parse(concat!(
        "/proc/server_maint_shape()\n",
        "\tvar/list/values = list(10, 20, 30)\n",
        "\tvar/position_in_loop = (1 / 5) + 1\n",
        "\tvar/read\n",
        "\tif(!(position_in_loop % 1))\n",
        "\t\tread = values[position_in_loop]\n",
        "\tvalues[2.8] = 25\n",
        "\tvalues[3.9] += 5\n",
        "\treturn list(read, values[2], values[3], \"abc\"[2.9])\n",
        "/proc/sub_one_index_is_invalid()\n",
        "\treturn list(1)[0.9]\n",
    ))
    .expect("fractional index source should parse");
    let module = compile_module(&syntax.definitions).expect("fractional indexes should compile");
    let mut state = ExecutionState::new();
    let Value::List(result) = execute_module_in_state(
        &module,
        module.procedure_id("/proc/server_maint_shape").unwrap(),
        &[],
        &mut state,
    )
    .expect("BYOND truncates positive fractional sequence indexes toward zero") else {
        panic!("server-maint shape should return a list")
    };
    assert_eq!(
        state
            .heap()
            .list(result)
            .unwrap()
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>(),
        vec![
            Value::number(10.0),
            Value::number(25.0),
            Value::number(35.0),
            Value::text("b"),
        ]
    );

    let error = execute_module_in_state(
        &module,
        module
            .procedure_id("/proc/sub_one_index_is_invalid")
            .unwrap(),
        &[],
        &mut state,
    )
    .expect_err("fractional indexes that truncate to zero remain invalid");
    assert_eq!(
        error.message,
        "list index must truncate to a positive number, received 0.9"
    );
}

#[test]
fn compound_associative_add_uses_null_identity_for_datums() {
    let syntax = parse(
            "/proc/add_route(list/paths, route, datum/value)\n\tpaths[route] += value\n\treturn paths[route]\n",
        )
        .expect("heretic route source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("heretic route source should compile");
    let mut state = ExecutionState::new();
    let paths = state.heap_mut().allocate_list();
    let value = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/heretic_route").unwrap());
    assert_eq!(
        execute_in_state(
            &program,
            &[Value::List(paths), Value::text("ash"), Value::Datum(value)],
            &mut state,
        ),
        Ok(Value::Datum(value))
    );
}

#[test]
fn compound_associative_bit_or_uses_zero_for_missing_key() {
    let syntax = parse(
            "/proc/add_smoothing_group(list/cache, key, bit)\n\tcache[key] |= bit\n\treturn cache[key]\n",
        )
        .expect("smoothing cache source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("compound associative bit-or should compile");
    let mut state = ExecutionState::new();
    let cache = state.heap_mut().allocate_list();

    assert_eq!(
        execute_in_state(
            &program,
            &[Value::List(cache), Value::text("0"), Value::number(8.0)],
            &mut state,
        ),
        Ok(Value::number(8.0)),
    );
}

#[test]
fn compound_associative_union_mutates_nested_list() {
    let syntax = parse(
            "/proc/grant(list/languages, language, source)\n\tlanguages[language] = list()\n\tlanguages[language] |= source\n\treturn languages[language].len\n",
        )
        .expect("lazy language source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("lazy language source should compile");
    let mut state = ExecutionState::new();
    let languages = state.heap_mut().allocate_list();
    assert_eq!(
        execute_in_state(
            &program,
            &[
                Value::List(languages),
                Value::text("common"),
                Value::number(1.0)
            ],
            &mut state,
        ),
        Ok(Value::number(1.0))
    );
}

#[test]
fn associative_literals_lookup_update_and_iterate_in_source_order() {
    let lookup = "/proc/lookup()\n\tvar/items = list(1, \"first\" = 10, 2, \"second\" = 20)\n\titems[\"first\"] = 11\n\treturn items[\"first\"]\n";
    let syntax = parse(lookup).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("associations should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::number(11.0)));

    let iteration = "/proc/order()\n\tvar/result = 0\n\tfor(var/item in list(1, \"key\" = 10, 2))\n\t\tif(item == \"key\")\n\t\t\tresult = result * 10 + 9\n\t\telse\n\t\t\tresult = result * 10 + item\n\treturn result\n";
    let syntax = parse(iteration).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("iteration should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::number(192.0)));
}

#[test]
fn for_in_break_continue_and_nesting_target_the_innermost_loop() {
    let source = "/proc/nested_lists()\n\tvar/total = 0\n\tfor(var/outer in list(1, 2))\n\t\tfor(var/inner in list(1, 2, 3, 4))\n\t\t\tif(inner == 2)\n\t\t\t\tcontinue\n\t\t\tif(inner == 4)\n\t\t\t\tbreak\n\t\t\ttotal = total + outer * inner\n\treturn total\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("nested lists should compile");

    assert_eq!(execute(&program, &[]), Ok(Value::number(12.0)));
}

#[test]
fn world_and_atom_iterables_enumerate_live_contents() {
    let syntax = parse(
            "/proc/count_world()\n\tvar/count = 0\n\tfor(var/atom/item as anything in world)\n\t\tcount++\n\treturn count\n",
        )
        .unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let mut state = ExecutionState::new();
    let world = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/world").unwrap());
    let contents = state.heap_mut().allocate_list();
    for path in ["/turf/open", "/obj/item"] {
        let datum = state
            .heap_mut()
            .allocate_datum(TypePath::parse(path).unwrap());
        state
            .heap_mut()
            .list_mut(contents)
            .unwrap()
            .add(Value::Datum(datum));
    }
    state
        .heap_mut()
        .set_datum_field(world, field("contents"), Value::List(contents))
        .unwrap();
    state.set_global(field("world"), Value::Datum(world));
    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/count_world").unwrap(),
            &[],
            &mut state,
        ),
        Ok(Value::number(2.0))
    );
}

#[test]
fn runtime_initial_atom_prototypes_do_not_enter_world_contents() {
    let mut state = ExecutionState::new();
    let world = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/world").unwrap());
    let world_contents = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .set_datum_field(world, field("contents"), Value::List(world_contents))
        .unwrap();
    state.set_global(field("world"), Value::Datum(world));

    let prototype_type = TypePath::parse("/obj/item/prototype_only").unwrap();
    state.set_type_parents(BTreeMap::from([
        (
            prototype_type.clone(),
            Some(TypePath::parse("/obj/item").unwrap()),
        ),
        (
            TypePath::parse("/obj/item").unwrap(),
            Some(TypePath::parse("/obj").unwrap()),
        ),
        (
            TypePath::parse("/obj").unwrap(),
            Some(TypePath::parse("/atom/movable").unwrap()),
        ),
    ]));
    state.set_initial_values(BTreeMap::from([(
        prototype_type.clone(),
        BTreeMap::from([(field("dynamic_default"), Value::Null)]),
    )]));
    state.set_instance_initializers(
        Arc::new(BTreeMap::from([(
            prototype_type.clone(),
            vec![InstanceInitializer::Constant {
                field: field("dynamic_default"),
                value: Value::number(7.0),
            }],
        )])),
        None,
    );

    assert_eq!(
        crate::runtime_initial_field_value(&mut state, &prototype_type, &field("dynamic_default"),),
        Ok(Value::number(7.0))
    );
    assert_eq!(state.initial_prototypes.len(), 1);
    assert_eq!(state.heap().list(world_contents).unwrap().len(), 0);
}

#[test]
fn initializer_plans_cache_parent_order_and_invalidate_with_metadata() {
    use std::hint::black_box;
    use std::time::Instant;

    let mut state = ExecutionState::new();
    let root = TypePath::parse("/datum/plan_root").unwrap();
    let parent = TypePath::parse("/datum/plan_root/parent").unwrap();
    let child = TypePath::parse("/datum/plan_root/parent/child").unwrap();
    state.set_type_parents(BTreeMap::from([
        (root.clone(), None),
        (parent.clone(), Some(root.clone())),
        (child.clone(), Some(parent.clone())),
    ]));
    state.set_instance_initializers(
        Arc::new(BTreeMap::from([
            (
                root.clone(),
                vec![InstanceInitializer::Constant {
                    field: field("root_value"),
                    value: Value::number(1.0),
                }],
            ),
            (
                parent.clone(),
                vec![InstanceInitializer::Constant {
                    field: field("parent_value"),
                    value: Value::number(2.0),
                }],
            ),
            (
                child.clone(),
                vec![InstanceInitializer::Constant {
                    field: field("child_value"),
                    value: Value::number(3.0),
                }],
            ),
        ])),
        None,
    );

    let first = instance_initializer_plan(&mut state, &child);
    let second = instance_initializer_plan(&mut state, &child);
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(first.len(), 3);
    assert_eq!(
        first
            .iter()
            .map(|initializer| match initializer {
                InstanceInitializer::Constant { field, .. }
                | InstanceInitializer::Program { field, .. } => field.as_str(),
            })
            .collect::<Vec<_>>(),
        vec!["root_value", "parent_value", "child_value"]
    );

    let iterations = 100_000;
    let cached_started = Instant::now();
    for _ in 0..iterations {
        black_box(instance_initializer_plan(&mut state, black_box(&child)));
    }
    let cached = cached_started.elapsed();

    let rebuilt_started = Instant::now();
    for _ in 0..iterations {
        let mut hierarchy = Vec::new();
        let mut current = Some(child.clone());
        while let Some(path) = current {
            hierarchy.push(path.clone());
            current = state.type_parent(&path).cloned();
        }
        hierarchy.reverse();
        black_box(
            hierarchy
                .into_iter()
                .flat_map(|path| {
                    state
                        .instance_initializers
                        .get(&path)
                        .into_iter()
                        .flatten()
                        .cloned()
                })
                .collect::<Vec<_>>(),
        );
    }
    let rebuilt_elapsed = rebuilt_started.elapsed();
    eprintln!(
        "initializer-plan-cache iterations={iterations} cached={cached:?} rebuilt={rebuilt_elapsed:?} speedup={:.2}x",
        rebuilt_elapsed.as_secs_f64() / cached.as_secs_f64()
    );

    state.set_type_parents(BTreeMap::from([(child.clone(), None)]));
    let rebuilt = instance_initializer_plan(&mut state, &child);
    assert_eq!(rebuilt.len(), 1);
    assert!(!Arc::ptr_eq(&first, &rebuilt));
}

#[test]
fn runtime_initial_field_cache_preserves_inheritance_null_overrides_and_invalidation() {
    let mut state = ExecutionState::new();
    let parent = TypePath::parse("/datum/initial_cache_parent").unwrap();
    let child = TypePath::parse("/datum/initial_cache_parent/child").unwrap();
    let inherited = field("inherited");
    let overridden_null = field("overridden_null");
    let runtime = field("runtime");
    state.set_type_parents(BTreeMap::from([
        (parent.clone(), None),
        (child.clone(), Some(parent.clone())),
    ]));
    state.set_initial_values(BTreeMap::from([
        (
            parent.clone(),
            BTreeMap::from([
                (inherited.clone(), Value::number(7.0)),
                (overridden_null.clone(), Value::number(8.0)),
                (runtime.clone(), Value::Null),
            ]),
        ),
        (
            child.clone(),
            BTreeMap::from([(overridden_null.clone(), Value::Null)]),
        ),
    ]));
    state.set_instance_initializers(
        Arc::new(BTreeMap::from([(
            parent.clone(),
            vec![InstanceInitializer::Constant {
                field: runtime.clone(),
                value: Value::number(11.0),
            }],
        )])),
        None,
    );

    for _ in 0..2 {
        assert_eq!(
            crate::runtime_initial_field_value(&mut state, &child, &inherited),
            Ok(Value::number(7.0)),
        );
        assert_eq!(
            crate::runtime_initial_field_value(&mut state, &child, &overridden_null),
            Ok(Value::Null),
        );
        assert_eq!(
            crate::runtime_initial_field_value(&mut state, &child, &runtime),
            Ok(Value::number(11.0)),
        );
    }
    assert_eq!(state.initial_field_value_cache_entries, 3);
    assert_eq!(state.initial_prototypes.len(), 1);

    state.set_initial_values(BTreeMap::from([(
        parent,
        BTreeMap::from([(inherited.clone(), Value::number(13.0))]),
    )]));
    assert!(state.initial_field_value_cache.is_empty());
    assert_eq!(
        crate::runtime_initial_field_value(&mut state, &child, &inherited),
        Ok(Value::number(13.0)),
    );
}

#[test]
fn one_turf_geometry_rebuild_installs_world_dimension_fields() {
    let mut state = ExecutionState::new();
    let world_path = TypePath::parse("/world").unwrap();
    for name in ["maxx", "maxy", "maxz"] {
        assert_eq!(
            crate::engine_builtin_initial_value(&world_path, &field(name)),
            Some(Value::number(0.0)),
            "world dimensions exist before map geometry is installed",
        );
    }
    let world = state.heap_mut().allocate_datum(world_path);
    let turf = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/turf/open").unwrap());
    for name in ["x", "y", "z"] {
        state
            .heap_mut()
            .set_datum_field(turf, field(name), Value::number(1.0))
            .unwrap();
    }
    state.rebuild_world_geometry();
    for name in ["maxx", "maxy", "maxz"] {
        assert_eq!(
            state.heap().datum_field(world, &field(name)),
            Ok(&Value::number(1.0)),
        );
    }
    assert_eq!(state.turf_at(1, 1, 1), Some(turf));
    assert_eq!(state.world_turf_lookup, vec![Some(turf)]);
}

#[test]
#[ignore = "release-only parsed-map coordinate lookup benchmark"]
fn parsed_map_coordinate_lookup_benchmark() {
    let mut ordered = BTreeMap::new();
    let mut indexed = vec![0usize; 255 * 255 * 2];
    let mut ordinal = 0usize;
    for z in 1..=2 {
        for y in 1..=255 {
            for x in 1..=255 {
                ordinal += 1;
                ordered.insert((x, y, z), ordinal);
                let index = (((z - 1) * 255 + (y - 1)) * 255 + (x - 1)) as usize;
                indexed[index] = ordinal;
            }
        }
    }
    let coordinates = ordered.keys().copied().collect::<Vec<_>>();
    let rounds = 8;
    let ordered_started = Instant::now();
    let mut ordered_sum = 0usize;
    for _ in 0..rounds {
        for coordinate in &coordinates {
            ordered_sum ^= std::hint::black_box(*ordered.get(coordinate).unwrap());
        }
    }
    let ordered_elapsed = ordered_started.elapsed();
    let indexed_started = Instant::now();
    let mut indexed_sum = 0usize;
    for _ in 0..rounds {
        for &(x, y, z) in &coordinates {
            let index = (((z - 1) * 255 + (y - 1)) * 255 + (x - 1)) as usize;
            indexed_sum ^= std::hint::black_box(indexed[index]);
        }
    }
    let indexed_elapsed = indexed_started.elapsed();
    assert_eq!(ordered_sum, indexed_sum);
    eprintln!(
        "parsed-map-coordinate-lookups={} ordered_ms={} indexed_ms={} speedup={:.2}",
        coordinates.len() * rounds,
        ordered_elapsed.as_millis(),
        indexed_elapsed.as_millis(),
        ordered_elapsed.as_secs_f64() / indexed_elapsed.as_secs_f64(),
    );
}

#[test]
fn list_length_numeric_fast_path_preserves_rounding_and_resize_coercion() {
    for length in [0, 1, 4, 16_777_216, 16_777_217, u32::MAX as usize] {
        let decimal = length.to_string().parse::<f32>().unwrap();
        assert_eq!(crate::dm_list_length_number(length), decimal);
    }
    for length in [0.0, 1.9, 4.0, 16_777_217.0, f32::MAX] {
        let prior = length
            .trunc()
            .to_string()
            .parse::<usize>()
            .unwrap_or(usize::MAX);
        assert_eq!(crate::dm_list_resize_length(length), prior);
    }
}

#[test]
#[ignore = "release-only list-length conversion microbenchmark"]
fn list_length_conversion_release_microbenchmark() {
    let rounds = 2_000_000usize;
    let started = Instant::now();
    let mut decimal_sum = 0.0f32;
    for index in 0..rounds {
        decimal_sum += std::hint::black_box(index)
            .to_string()
            .parse::<f32>()
            .unwrap();
    }
    let decimal_elapsed = started.elapsed();
    let started = Instant::now();
    let mut direct_sum = 0.0f32;
    for index in 0..rounds {
        direct_sum += crate::dm_list_length_number(std::hint::black_box(index));
    }
    let direct_elapsed = started.elapsed();
    eprintln!(
        "list-length-conversion rounds={rounds} decimal_ms={} direct_ms={} sums={decimal_sum}/{direct_sum}",
        decimal_elapsed.as_millis(),
        direct_elapsed.as_millis(),
    );
    assert_eq!(decimal_sum, direct_sum);
}

#[test]
fn world_geometry_registers_atoms_once_and_iteration_uses_byond_category_order() {
    let syntax = parse(
            "/proc/world_order()\n\tvar/list/result = list()\n\tfor(var/atom/item as anything in world)\n\t\tresult += item\n\treturn result\n",
        )
        .expect("world iteration source should parse");
    let module = compile_module(&syntax.definitions).expect("world iteration should compile");
    let mut state = ExecutionState::new();
    let world = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/world").unwrap());
    for (name, value) in [
        ("maxx", Value::number(2.0)),
        ("maxy", Value::number(1.0)),
        ("maxz", Value::number(1.0)),
        (
            "area",
            Value::TypePath(TypePath::parse("/area/default").unwrap()),
        ),
        (
            "turf",
            Value::TypePath(TypePath::parse("/turf/default").unwrap()),
        ),
    ] {
        state
            .heap_mut()
            .set_datum_field(world, field(name), value)
            .unwrap();
    }
    state.set_global(field("world"), Value::Datum(world));
    state.resize_world_geometry(world, (2, 1, 1)).unwrap();
    let object =
        allocate_initialized_datum(&mut state, TypePath::parse("/obj/item").unwrap()).unwrap();
    let mob =
        allocate_initialized_datum(&mut state, TypePath::parse("/mob/living/player").unwrap())
            .unwrap();

    let Value::List(contents) = state.heap().datum_field(world, &field("contents")).unwrap() else {
        panic!("world.contents should be a list")
    };
    let registered = state
        .heap()
        .list(*contents)
        .unwrap()
        .positions()
        .map(|(_, value)| value.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        registered.len(),
        5,
        "two turfs must not be registered twice"
    );
    assert_eq!(
        registered
            .iter()
            .filter_map(|value| match value {
                Value::Datum(datum) => Some(*datum),
                _ => None,
            })
            .collect::<std::collections::HashSet<_>>()
            .len(),
        registered.len(),
        "world.contents must contain one entry per atom",
    );

    let Value::List(order) = execute_module_in_state(
        &module,
        module.procedure_id("/proc/world_order").unwrap(),
        &[],
        &mut state,
    )
    .expect("world iteration should execute") else {
        panic!("world iteration should return a list")
    };
    let paths = state
        .heap()
        .list(order)
        .unwrap()
        .positions()
        .map(|(_, value)| match value {
            Value::Datum(datum) => state
                .heap()
                .datum(*datum)
                .unwrap()
                .type_path()
                .as_str()
                .to_owned(),
            value => panic!("world iteration yielded non-atom {value:?}"),
        })
        .collect::<Vec<_>>();
    assert_eq!(
        paths,
        [
            "/mob/living/player",
            "/obj/item",
            "/area/default",
            "/turf/default",
            "/turf/default",
        ]
    );
    assert!(registered.contains(&Value::Datum(object)));
    assert!(registered.contains(&Value::Datum(mob)));
}

#[test]
fn world_contents_iteration_snapshot_batch_fill_matches_per_element_add() {
    // Replays the pre-`extend_positional` population: bucket by BYOND category,
    // then build the list with per-element `add`.
    fn reference_snapshot(state: &mut ExecutionState, contents: ListId) -> ListId {
        let values: Vec<Value> = state
            .heap()
            .list(contents)
            .unwrap()
            .positions()
            .map(|(_, value)| value.clone())
            .collect();
        let mob = TypePath::parse("/mob").unwrap();
        let movable = TypePath::parse("/atom/movable").unwrap();
        let area = TypePath::parse("/area").unwrap();
        let turf = TypePath::parse("/turf").unwrap();
        let mut buckets: [Vec<Value>; 5] = std::array::from_fn(|_| Vec::new());
        for value in values {
            let path = match &value {
                Value::Datum(datum) => state
                    .heap()
                    .datum(*datum)
                    .ok()
                    .map(|datum| datum.type_path().clone()),
                _ => None,
            };
            let category = match &path {
                Some(path) if is_subtype(state, path, &mob) => 0,
                Some(path) if is_subtype(state, path, &movable) => 1,
                Some(path) if is_subtype(state, path, &area) => 2,
                Some(path) if is_subtype(state, path, &turf) => 3,
                _ => 4,
            };
            buckets[category].push(value);
        }
        let snapshot = state.heap_mut().allocate_list();
        for value in buckets.into_iter().flatten() {
            state.heap_mut().list_mut(snapshot).unwrap().add(value);
        }
        snapshot
    }

    let mut state = ExecutionState::new();
    let datum = |state: &mut ExecutionState, path: &str| {
        state
            .heap_mut()
            .allocate_datum(TypePath::parse(path).unwrap())
    };
    let mob_a = datum(&mut state, "/mob/living/carbon/human");
    let obj_a = datum(&mut state, "/obj/item/gun");
    let area_a = datum(&mut state, "/area/station/engineering");
    let turf_a = datum(&mut state, "/turf/open/floor/plating");
    let obj_b = datum(&mut state, "/obj/machinery/door");
    let mob_b = datum(&mut state, "/mob/living/silicon/robot");
    let turf_b = datum(&mut state, "/turf/closed/wall");
    let plain = datum(&mut state, "/datum/reagent");

    let contents = state.heap_mut().allocate_list();
    for value in [turf_a, obj_a, mob_a, area_a, plain, obj_b, mob_b, turf_b] {
        state
            .heap_mut()
            .list_mut(contents)
            .unwrap()
            .add(Value::Datum(value));
    }
    state
        .heap_mut()
        .list_mut(contents)
        .unwrap()
        .add(Value::number(7.0));

    let reference = reference_snapshot(&mut state, contents);
    let snapshot = world_contents_iteration_snapshot(&mut state, contents).unwrap();

    let positions = |state: &ExecutionState, id| {
        state
            .heap()
            .list(id)
            .unwrap()
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(positions(&state, snapshot), positions(&state, reference));
    assert_eq!(
        positions(&state, snapshot),
        vec![
            Value::Datum(mob_a),
            Value::Datum(mob_b),
            Value::Datum(obj_a),
            Value::Datum(obj_b),
            Value::Datum(area_a),
            Value::Datum(turf_a),
            Value::Datum(turf_b),
            Value::Datum(plain),
            Value::number(7.0),
        ],
    );
    let list = state.heap().list(snapshot).unwrap();
    assert_eq!(list.len(), 9);
    assert_eq!(
        list.associative_len(),
        0,
        "a PrepareIteration snapshot is a plain positional list"
    );
}

#[test]
fn atom_contents_iteration_snapshot_batch_fill_matches_per_element_add() {
    let mut state = ExecutionState::new();
    let area = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/area/space").unwrap());
    let other = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/area/station").unwrap());
    let member = |state: &mut ExecutionState, path: &str, loc: DatumId| {
        let id = state
            .heap_mut()
            .allocate_datum(TypePath::parse(path).unwrap());
        state
            .heap_mut()
            .set_datum_field(id, field("loc"), Value::Datum(loc))
            .unwrap();
        id
    };
    let t1 = member(&mut state, "/turf/open/space", area);
    let stray = member(&mut state, "/turf/open/floor", other);
    let t2 = member(&mut state, "/turf/open/space/nearstar", area);
    let t3 = member(&mut state, "/turf/closed/wall", area);
    let no_loc = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/turf/open/space").unwrap());

    let contents = state.heap_mut().allocate_list();
    for value in [t1, stray, t2, no_loc, t3] {
        state
            .heap_mut()
            .list_mut(contents)
            .unwrap()
            .add(Value::Datum(value));
    }
    state
        .heap_mut()
        .list_mut(contents)
        .unwrap()
        .add(Value::text("junk"));

    // Pre-`extend_positional` reference: per-element `add` of the members whose
    // `loc` resolves to the owner, in their original contents order.
    let members: Vec<Value> = state
        .heap()
        .list(contents)
        .unwrap()
        .positions()
        .map(|(_, value)| value.clone())
        .collect();
    let reference = state.heap_mut().allocate_list();
    for value in members {
        let Value::Datum(datum) = value else { continue };
        let owns = matches!(
            state.heap().datum_field(datum, &field("loc")),
            Ok(Value::Datum(loc)) if *loc == area
        );
        if owns {
            state
                .heap_mut()
                .list_mut(reference)
                .unwrap()
                .add(Value::Datum(datum));
        }
    }

    let snapshot = atom_contents_iteration_snapshot(&mut state, area, contents).unwrap();
    let positions = |state: &ExecutionState, id| {
        state
            .heap()
            .list(id)
            .unwrap()
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(positions(&state, snapshot), positions(&state, reference));
    assert_eq!(
        positions(&state, snapshot),
        vec![Value::Datum(t1), Value::Datum(t2), Value::Datum(t3)],
    );
    assert_eq!(
        state.heap().list(snapshot).unwrap().associative_len(),
        0,
        "an atom contents snapshot is a plain positional list"
    );
}

#[test]
fn parameter_literal_default_applies_only_when_argument_is_omitted() {
    let source = "/proc/defaulted(value = 5)\n\treturn value\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");

    assert_eq!(execute(&program, &[]), Ok(Value::number(5.0)));
    assert_eq!(
        execute(&program, &[Value::Null]),
        Ok(Value::number(5.0)),
        "BYOND and OpenDream apply a parameter default when the supplied value is null",
    );
    assert_eq!(
        execute(&program, &[Value::number(9.0)]),
        Ok(Value::number(9.0))
    );

    let text = parse("/proc/text_default(value = \"fallback\")\n\treturn value\n")
        .expect("source should parse");
    let program = compile_procedure(&text.definitions[0]).expect("procedure should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::text("fallback")));
}

#[test]
fn dm_boolean_constants_work_in_defaults_and_expressions() {
    let source = "/proc/booleans(enabled = TRUE, disabled = FALSE)\n\tif(disabled)\n\t\treturn 99\n\treturn enabled + TRUE\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("DM boolean constants should compile as numeric literals");

    assert_eq!(execute(&program, &[]), Ok(Value::number(2.0)));
    assert_eq!(
        execute(&program, &[Value::Null, Value::number(1.0)]),
        Ok(Value::number(99.0))
    );
}

#[test]
fn dm_profile_command_constants_are_byond_bitflags() {
    let source = "/proc/profile_flags()\n\treturn PROFILE_START + PROFILE_REFRESH + PROFILE_STOP + PROFILE_CLEAR + PROFILE_RESTART + PROFILE_AVERAGE\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("BYOND profiling constants should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::number(9.0)));
}

#[test]
fn dm_blend_constants_work_in_defaults_and_expressions() {
    let source = "/proc/blend(mode = BLEND_MULTIPLY)\n\treturn mode + BLEND_INSET_OVERLAY + BLEND_DEFAULT + BLEND_OVERLAY + BLEND_ADD + BLEND_SUBTRACT\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("BYOND blend constants should compile as numeric literals");

    assert_eq!(execute(&program, &[]), Ok(Value::number(15.0)));
    assert_eq!(
        execute(&program, &[Value::number(2.0)]),
        Ok(Value::number(13.0))
    );
}

#[test]
fn dm_filter_and_mask_constant_families_have_byond_values() {
    let syntax = parse(concat!(
            "/proc/filter_constants()\n",
            "\treturn list(MASK_INVERSE, MASK_SWAP, FILTER_OVERLAY, FILTER_UNDERLAY, OUTLINE_SHARP, OUTLINE_SQUARE, WAVE_BOUNDED, WAVE_SIDEWAYS, FILTER_COLOR_RGB, FILTER_COLOR_HSV, FILTER_COLOR_HSL, FILTER_COLOR_HCY)\n",
        ))
        .expect("filter constants should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("the complete BYOND filter flag family should compile");
    let mut state = ExecutionState::new();
    let Value::List(values) = execute_in_state(&program, &[], &mut state).unwrap() else {
        panic!("constant inventory should return a list");
    };
    assert_eq!(
        state
            .heap()
            .list(values)
            .unwrap()
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>(),
        [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0].map(Value::number)
    );
}

#[test]
fn dm_reset_appearance_constants_are_appearance_flag_bits() {
    let source =
        "/proc/appearance_flags()\n\treturn RESET_TRANSFORM | RESET_COLOR | RESET_ALPHA | 1\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("appearance constants should compile as BYOND numeric constants");

    assert_eq!(execute(&program, &[]), Ok(Value::number(57.0)));
}

#[test]
fn dm_appearance_flags_are_the_documented_byond_bit_positions() {
    let source = "/proc/appearance_flags()\n\treturn KEEP_TOGETHER | KEEP_APART | LONG_GLIDE | RESET_TRANSFORM | RESET_COLOR | RESET_ALPHA | PIXEL_SCALE | TILE_BOUND | INHERIT_ID | NO_CLIENT_COLOR | RESET_CONTENTS | PLANE_MASTER | PASS_MOUSE | TILE_MOVER\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("appearance flags should compile as BYOND numeric constants");

    assert_eq!(execute(&program, &[]), Ok(Value::number(16_383.0)));
}

#[test]
fn replacetext_builtin_family_replaces_text_with_byond_bounds() {
    let source = "/proc/rewrite()\n\tvar/exact = replacetextEx_char(\"Port Bow / port bow\", \"Port Bow\", \"Northwest\")\n\tvar/insensitive = replacetext_char(exact, \"port bow\", \"Southwest\")\n\treturn replacetextEx(insensitive, \"Northwest\", \"East\", 1, 10)\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("replacetext builtin family should compile");

    assert!(program.instructions.iter().any(|instruction| {
        matches!(
            instruction,
            Instruction::ReplaceText {
                exact: true,
                character_indices: true,
                ..
            }
        )
    }));
    assert_eq!(execute(&program, &[]), Ok(Value::text("East / Southwest")));
}

#[test]
fn replacetext_uses_byond_scalar_replacement_and_typed_text_inputs() {
    let source = concat!(
        "/proc/uplink_description()\n",
        "\tvar/desc = \"minimum %MIN seconds\"\n",
        "\tdesc = replacetext(desc, \"%MIN\", 90)\n",
        "\tvar/empty_needle = replacetext(\"abc\", 2, \"x\")\n",
        "\tvar/removed = replacetext(\"abc\", \"b\", null)\n",
        "\tvar/nontext_source = replacetext(123, \"2\", 9)\n",
        "\treturn list(desc, empty_needle, removed, nontext_source)\n",
    );
    let syntax = parse(source).expect("replacetext coercion fixture should parse");
    let module =
        compile_module(&syntax.definitions).expect("replacetext coercion fixture should compile");
    let entry = module
        .procedure_id("/proc/uplink_description")
        .expect("fixture entry should link");
    let mut state = ExecutionState::new();
    let result = execute_module_in_state(&module, entry, &[], &mut state)
        .expect("BYOND replacetext coercion should execute");
    let Value::List(result) = result else {
        panic!("fixture should return a list");
    };
    let values = state
        .heap()
        .list(result)
        .unwrap()
        .positions()
        .map(|(_, value)| value.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        values,
        vec![
            Value::text("minimum 90 seconds"),
            Value::text("axbxc"),
            Value::text("ac"),
            Value::Null,
        ]
    );
}

#[test]
fn typed_and_uninitialized_locals_start_as_null() {
    let source = "/proc/locals()\n\tvar/datum/example/typed\n\tvar/plain\n\tif(typed || plain)\n\t\treturn 0\n\treturn 7\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("typed locals without initializers should compile");

    assert_eq!(execute(&program, &[]), Ok(Value::number(7.0)));
}

#[test]
fn later_bare_new_assignment_uses_the_local_declared_type() {
    let syntax = parse(concat!(
        "/datum/feedback_variable\n",
        "\tvar/key\n",
        "\tvar/key_type\n",
        "/datum/feedback_variable/New(new_key, new_key_type)\n",
        "\tkey = new_key\n",
        "\tkey_type = new_key_type\n",
        "/proc/find_feedback_datum()\n",
        "\tvar/datum/feedback_variable/FV\n",
        "\tFV = new(\"sect_chosen\", \"text\")\n",
        "\treturn istype(FV, /datum/feedback_variable)\n",
    ))
    .expect("feedback-variable constructor shape should parse");
    let procedures = syntax
        .definitions
        .iter()
        .filter(|definition| definition.kind == DefinitionKind::Procedure)
        .cloned()
        .collect::<Vec<_>>();
    let module = compile_module(&procedures)
        .expect("later assignment should infer the typed local constructor path");
    let entry = module
        .procedure_id("/proc/find_feedback_datum")
        .expect("feedback helper should be linked");

    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(1.0)));
}

#[test]
fn suffix_array_locals_use_declared_name_and_dynamic_dimensions() {
    let syntax = parse(
            "/proc/one(roomSize)\n\tvar/storage[roomSize]\n\treturn storage.len\n/proc/multi(x, y)\n\tvar/list/grid[x][y]\n\treturn grid[1].len\n",
        ).expect("suffix array source");
    let module = compile_module(&syntax.definitions).expect("suffix arrays compile");
    assert_eq!(
        execute_module(
            &module,
            module.procedure_id("/proc/one").unwrap(),
            &[Value::number(4.0)]
        ),
        Ok(Value::number(4.0)),
    );
    assert_eq!(
        execute_module(
            &module,
            module.procedure_id("/proc/multi").unwrap(),
            &[Value::number(2.0), Value::number(3.0)]
        ),
        Ok(Value::number(3.0)),
    );
}

#[test]
fn unnamed_varargs_parameter_reserves_its_argument_slot() {
    let source = "/proc/with_varargs(first, ...)\n\tvar/after = first\n\treturn after\n";
    let syntax = parse(source).expect("source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("unnamed varargs should compile");

    assert_eq!(program.parameter_count, 2);
    // Two declared argument positions, one ordinary local, and implicit
    // per-call `args`.
    assert_eq!(program.local_count, 4);
    assert_eq!(
        execute(&program, &[Value::number(9.0)]),
        Ok(Value::number(9.0))
    );
}

#[test]
fn unused_implicit_args_does_not_allocate_a_heap_list() {
    let source = "/proc/hot_path(value)\n\treturn value + 1\n";
    let syntax = parse(source).expect("ordinary procedure should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("ordinary procedure should compile");
    assert!(
        !program
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::MakeArgs)),
        "unused special args must remain lazy"
    );

    let mut state = ExecutionState::new();
    assert_eq!(
        execute_in_state(&program, &[Value::number(4.0)], &mut state),
        Ok(Value::number(5.0))
    );
    assert_eq!(
        state.heap().live_list_count(),
        0,
        "a procedure that cannot observe args must not allocate its list"
    );
}

#[test]
fn implicit_args_is_a_per_call_list_of_all_supplied_values() {
    let source =
        "/proc/collect(first)\n\tif(length(args) != 3)\n\t\treturn 0\n\treturn args[3] + first\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("implicit args should compile as a local list");

    assert!(
        program
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::MakeArgs))
    );
    let mut state = ExecutionState::new();
    assert_eq!(
        execute_in_state(
            &program,
            &[Value::number(2.0), Value::number(5.0), Value::number(11.0)],
            &mut state,
        ),
        Ok(Value::number(13.0))
    );
    assert_eq!(
        state.heap().live_list_count(),
        1,
        "referencing args must materialize one BYOND argument list"
    );
    assert_eq!(
        execute_in_state(&program, &[], &mut state),
        Ok(Value::number(0.0))
    );
}

#[test]
fn implicit_args_reflects_live_parameter_defaults_and_assignments() {
    let syntax = parse(concat!(
        "/proc/receive(value)\n",
        "\treturn value\n",
        "/proc/observe(value = 7)\n",
        "\tvalue += 1\n",
        "\treturn args[1]\n",
        "/proc/forward(first, value = 7)\n",
        "\tvalue += 1\n",
        "\treturn receive(arglist(args.Copy(2)))\n",
    ))
    .expect("live args forwarding fixture should parse");
    let module = compile_module(&syntax.definitions).expect("live args fixture should link");
    let observe = module
        .procedure_id("/proc/observe")
        .expect("observe entry should exist");
    let entry = module
        .procedure_id("/proc/forward")
        .expect("forward entry should exist");

    assert_eq!(
        execute_module(&module, observe, &[Value::Null]),
        Ok(Value::number(8.0)),
        "direct args indexing observes the live parameter slot",
    );
    assert_eq!(
        execute_module(&module, entry, &[Value::number(1.0), Value::Null]),
        Ok(Value::number(8.0)),
        "args.Copy(2) must forward the live post-default, post-assignment parameter slot",
    );
}

#[test]
#[ignore = "microbenchmark; run explicitly with --ignored --nocapture"]
fn benchmark_plain_local_store_move_avoids_value_clone() {
    const ITERATIONS: usize = 5_000_000;
    let template = Value::text("production-shaped startup local payload");

    let mut old_slot = Value::Null;
    let old_started = std::time::Instant::now();
    for _ in 0..ITERATIONS {
        let value = std::hint::black_box(template.clone());
        old_slot = value.clone();
        std::hint::black_box(&value);
    }
    let old_elapsed = old_started.elapsed();
    std::hint::black_box(&old_slot);

    let mut new_slot = Value::Null;
    let new_started = std::time::Instant::now();
    for _ in 0..ITERATIONS {
        let value = std::hint::black_box(template.clone());
        new_slot = value;
        std::hint::black_box(&new_slot);
    }
    let new_elapsed = new_started.elapsed();
    std::hint::black_box(&new_slot);

    eprintln!(
        "plain StoreLocal old_clone_ms={} new_move_ms={} speedup={:.2}x",
        old_elapsed.as_millis(),
        new_elapsed.as_millis(),
        old_elapsed.as_secs_f64() / new_elapsed.as_secs_f64(),
    );
}

#[test]
#[ignore = "release-only per-call trace-filter lookup benchmark"]
fn cached_procedure_argument_trace_filter_release_benchmark() {
    const ITERATIONS: usize = 5_000_000;

    let uncached_started = std::time::Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(std::env::var("DREAM64_TRACE_PROC_ARGS").ok());
    }
    let uncached = uncached_started.elapsed();

    let cached_started = std::time::Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(crate::procedure_argument_trace_filter());
    }
    let cached = cached_started.elapsed();

    eprintln!(
        "per-call trace filter uncached_env_ms={} cached_once_ms={} speedup={:.2}x",
        uncached.as_millis(),
        cached.as_millis(),
        uncached.as_secs_f64() / cached.as_secs_f64(),
    );
}

#[test]
fn shuttle_trace_is_explicitly_opt_in() {
    for value in [Some("1"), Some("true"), Some("yes"), Some("on")] {
        assert!(crate::diagnostic_env_truthy(value), "{value:?}");
    }
    for value in [None, Some(""), Some("0"), Some("false"), Some("TRUE")] {
        assert!(!crate::diagnostic_env_truthy(value), "{value:?}");
    }
}

#[test]
#[ignore = "release-only disabled shuttle-trace call-path benchmark"]
fn disabled_shuttle_trace_call_path_release_benchmark() {
    const ITERATIONS: usize = 5_000_000;
    let source = parse("/datum/example/proc/run(value)\n\treturn value\n").unwrap();
    let module = compile_module(&source.definitions).unwrap();
    let procedure = module.procedure_id("/datum/example/proc/run").unwrap();
    let arguments = [Value::number(4.0)];

    let old_started = std::time::Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(crate::shuttle_trace_slot_from_arguments(&arguments));
        let path = std::hint::black_box(module.procedure_path(procedure).unwrap());
        std::hint::black_box(crate::shuttle_trace_is_late_shuttle_move(path));
        std::hint::black_box(crate::shuttle_trace_is_nullify_node(path));
        std::hint::black_box(crate::shuttle_trace_is_atmos_init(path));
    }
    let old = old_started.elapsed();

    let new_started = std::time::Instant::now();
    for _ in 0..ITERATIONS {
        if std::hint::black_box(crate::shuttle_trace_enabled()) {
            std::hint::black_box(crate::shuttle_trace_slot_from_arguments(&arguments));
        }
    }
    let new = new_started.elapsed();

    eprintln!(
        "shuttle trace default old_diagnostic_ms={} disabled_opt_in_ms={} speedup={:.2}x",
        old.as_millis(),
        new.as_millis(),
        old.as_secs_f64() / new.as_secs_f64(),
    );
}

#[test]
#[ignore = "release-only disabled startup-profiler entry benchmark"]
fn disabled_startup_profiler_entry_release_benchmark() {
    const ITERATIONS: usize = 5_000_000;
    let path = "/datum/parsed_map/proc/build_coordinate@214645";

    let old_started = Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(crate::is_atoms_initialize_path(path));
        std::hint::black_box(crate::is_subsystem_initialize_path(path));
    }
    let old = old_started.elapsed();

    let new_started = Instant::now();
    for _ in 0..ITERATIONS {
        let profiling_enabled = std::hint::black_box(false);
        if profiling_enabled {
            std::hint::black_box(crate::is_atoms_initialize_path(path));
            std::hint::black_box(crate::is_subsystem_initialize_path(path));
        }
    }
    let new = new_started.elapsed();

    eprintln!(
        "startup profile entry old_path_checks_ms={} disabled_gate_ms={} speedup={:.2}x",
        old.as_millis(),
        new.as_millis(),
        old.as_secs_f64() / new.as_secs_f64(),
    );
}

#[test]
#[ignore = "release-only owned value canonicalization benchmark"]
fn owned_value_canonicalization_release_benchmark() {
    const ITERATIONS: usize = 5_000_000;
    let heap = dm_value::ValueHeap::new();
    let template = Value::text("mapping-model-key");

    let old_started = Instant::now();
    for _ in 0..ITERATIONS {
        let value = std::hint::black_box(template.clone());
        std::hint::black_box(crate::canonicalize_value(&heap, &value));
    }
    let old = old_started.elapsed();

    let new_started = Instant::now();
    for _ in 0..ITERATIONS {
        let value = std::hint::black_box(template.clone());
        std::hint::black_box(crate::canonicalize_owned_value(&heap, value));
    }
    let new = new_started.elapsed();

    eprintln!(
        "value canonicalization old_clone_ms={} owned_move_ms={} speedup={:.2}x",
        old.as_millis(),
        new.as_millis(),
        old.as_secs_f64() / new.as_secs_f64(),
    );
}

#[test]
fn implicit_args_pads_omitted_declared_parameters_like_byond_atom_new() {
    let source = concat!(
        "/proc/rewrite_first(loc, ...)\n",
        "\targs[1] = 7\n",
        "\treturn length(args) * 10 + loc\n",
    );
    let syntax = parse(source).expect("atom/New-shaped args fixture should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("atom/New-shaped args fixture should compile");

    assert_eq!(program.parameter_count, 2);
    assert_eq!(
        execute(&program, &[]),
        Ok(Value::number(17.0)),
        "omitted named slots are padded, but an unnamed varargs marker is not an args entry"
    );
    assert_eq!(
        execute(
            &program,
            &[Value::number(1.0), Value::number(2.0), Value::number(3.0),],
        ),
        Ok(Value::number(37.0)),
        "extra supplied arguments remain visible after declared slots"
    );
}

#[test]
fn constructor_args_pad_an_omitted_atom_location_slot() {
    let syntax = parse(concat!(
        "/obj/args_fixture/New(loc, other)\n",
        "\targs[1] = 7\n",
        "\tglobal.args_observed = length(args) * 10 + args[1]\n",
        "/proc/run_args_fixture()\n",
        "\tnew /obj/args_fixture\n",
        "\treturn global.args_observed\n",
    ))
    .expect("atom constructor args fixture should parse");
    let module =
        compile_module(&syntax.definitions).expect("atom constructor args fixture should link");
    let entry = module
        .procedure_id("/proc/run_args_fixture")
        .expect("fixture entry should exist");

    assert_eq!(
        execute_module(&module, entry, &[]),
        Ok(Value::number(27.0)),
        "new /obj without an explicit loc still gives New both declared args slots"
    );
}

#[test]
fn multiple_parameter_defaults_evaluate_in_declaration_order() {
    let source =
        "/proc/combine(first = 1 + 1, second = 3, third = 4)\n\treturn first + second + third\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");

    assert_eq!(execute(&program, &[]), Ok(Value::number(9.0)));
    assert_eq!(
        execute(&program, &[Value::number(10.0)]),
        Ok(Value::number(17.0))
    );
    assert_eq!(
        execute(
            &program,
            &[Value::number(10.0), Value::Null, Value::number(1.0)],
        ),
        Ok(Value::number(14.0)),
        "BYOND treats explicit null like an omitted argument and applies the default",
    );
}

#[test]
fn defaults_interact_with_explicit_and_argument_reusing_current_calls() {
    let countdown =
        "/proc/countdown(value = 3)\n\tif(value <= 0)\n\t\treturn 0\n\treturn 1 + .(value - 1)\n";
    let syntax = parse(countdown).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::number(3.0)));

    let reapply = "/proc/reapply(value = 1)\n\tvalue = 0\n\treturn .()\n";
    let syntax = parse(reapply).expect("source should parse");
    let call_span = syntax.definitions[0].body[1].span;
    let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");
    let error = execute_with_limits(
        &program,
        &[],
        ExecutionLimits {
            max_call_depth: 3,
            ..ExecutionLimits::default()
        },
    )
    .expect_err(".() should reuse omission and reapply the default in each frame");
    assert!(error.message.contains("maximum call depth 3"));
    assert_eq!(error.source_span, Some(call_span));
    assert_eq!(error.call_stack.len(), 3);
}

#[test]
fn parameter_defaults_are_general_runtime_expressions() {
    let source = parse(
            "/proc/add_one(value)\n\treturn value + 1\n\n/proc/defaulted(first = 2, second = add_one(first), third = second * 10)\n\treturn first + second + third\n",
        )
        .expect("source should parse");
    let module = compile_module(&source.definitions)
        .expect("defaults should support parameter references and procedure calls");

    // Defaults execute at invocation time, in parameter order.  Each
    // later default observes values supplied or defaulted for its
    // predecessors, while supplied arguments skip only their own default.
    assert_eq!(
        execute_module(&module, module.names["/proc/defaulted"], &[]),
        Ok(Value::number(35.0))
    );
    assert_eq!(
        execute_module(
            &module,
            module.names["/proc/defaulted"],
            &[Value::number(7.0)],
        ),
        Ok(Value::number(95.0))
    );
}

#[test]
fn special_result_starts_null_and_is_returned_on_fallthrough() {
    let syntax = parse("/proc/empty()\n").expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");

    assert_eq!(execute(&program, &[]), Ok(Value::Null));
    assert!(
        program
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::LoadResult))
    );
}

#[test]
fn special_result_supports_reads_assignments_and_compound_assignments() {
    let source = "/proc/result()\n\t. = 2\n\t. += 3\n\t. *= 4\n\treturn .\n";
    let syntax = parse(source).expect("source should parse");
    let assignment_span = syntax.definitions[0].body[0].span;
    let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");

    assert_eq!(execute(&program, &[]), Ok(Value::number(20.0)));
    assert!(program.instructions.iter().zip(&program.source_spans).any(
        |(instruction, span)| matches!(instruction, Instruction::StoreResult)
            && *span == assignment_span
    ));
}

#[test]
fn indexed_assignment_statement_evaluates_rhs_before_index() {
    let source = "/proc/run()\n\tvar/list/output = list()\n\tvar/index = 1\n\toutput[\"[index - 1]\"] = ++index\n\treturn output[\"1\"]\n";
    let syntax = parse(source).expect("RHS-first indexed assignment should parse");
    let module =
        compile_module(&syntax.definitions).expect("RHS-first indexed assignment should compile");
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/run").unwrap(), &[]),
        Ok(Value::number(2.0)),
    );
}

#[test]
fn atom_iteration_ignores_entries_that_are_not_actually_located_in_the_owner() {
    let syntax = parse(
            "/proc/count(atom/container)\n\tvar/count = 0\n\tfor(var/atom/movable/item as anything in container)\n\t\tcount++\n\treturn count\n",
        )
        .unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let mut state = ExecutionState::new();
    let owner = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/storage").unwrap());
    let child = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/item").unwrap());
    let contents = state.ensure_contents(owner).unwrap();
    state
        .heap_mut()
        .set_datum_field(child, field("loc"), Value::Datum(owner))
        .unwrap();
    state
        .heap_mut()
        .list_mut(contents)
        .unwrap()
        .add(Value::Datum(child));
    // A stale/self entry can otherwise make recursive contents notification
    // call the same movable forever. Atom contents are defined by loc, so
    // this entry is not observable as a member of owner.contents.
    state
        .heap_mut()
        .list_mut(contents)
        .unwrap()
        .add(Value::Datum(owner));

    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/count").unwrap(),
            &[Value::Datum(owner)],
            &mut state,
        ),
        Ok(Value::number(1.0)),
    );
}

#[test]
fn stale_datum_ref_reads_as_null_in_vars_and_list_elements() {
    let syntax = parse(
            "/proc/run(list/parents)\n\tvar/value = parents[1]\n\tvar/copy = value\n\tif(copy == null)\n\t\treturn 1\n\treturn 0\n",
        )
        .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("procedure should compile");
    let entry = module.procedure_id("/proc/run").unwrap();
    let mut state = ExecutionState::new();
    let parent = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/example").unwrap());
    let parents = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(parents)
        .unwrap()
        .add(Value::Datum(parent));
    state.heap_mut().destroy_datum(parent).unwrap();

    let result = execute_module_in_state(&module, entry, &[Value::List(parents)], &mut state)
        .unwrap()
        .as_number();
    assert_eq!(result, Some(1.0));
}

#[test]
fn stale_datum_ref_truthiness_and_null_equality_are_consistent() {
    let syntax = parse(
        "/proc/run(list/parents)\n\tif(parents[1])\n\t\treturn 0\n\treturn parents[1] == null\n",
    )
    .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("procedure should compile");
    let entry = module.procedure_id("/proc/run").unwrap();
    let mut state = ExecutionState::new();
    let parent = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/example").unwrap());
    let parents = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(parents)
        .unwrap()
        .add(Value::Datum(parent));
    state.heap_mut().destroy_datum(parent).unwrap();

    let result = execute_module_in_state(&module, entry, &[Value::List(parents)], &mut state)
        .unwrap()
        .as_number();
    assert_eq!(result, Some(1.0));
}

#[test]
fn stale_datum_proc_access_reports_null_without_stale_handle_errors() {
    let syntax = parse(
            "/datum/example/proc/ping()\n\treturn 7\n/proc/run(list/parents)\n\treturn parents[1].ping()\n",
        )
        .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("procedure should compile");
    let entry = module.procedure_id("/proc/run").unwrap();
    let mut state = ExecutionState::new();
    let parent = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/example").unwrap());
    let parents = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(parents)
        .unwrap()
        .add(Value::Datum(parent));
    state.heap_mut().destroy_datum(parent).unwrap();

    let error =
        execute_module_in_state(&module, entry, &[Value::List(parents)], &mut state).unwrap_err();
    assert!(error.message.contains("cannot call a procedure on null"));
    assert!(!error.message.contains("stale datum"));
}

#[test]
fn stale_parent_slot_remains_readable_after_copy_and_list_lookup() {
    let syntax = parse(
            "/proc/run(list/parents)\n\tvar/list/cached = parents\n\tif(cached[1] == null)\n\t\treturn 1\n\treturn 0\n",
        )
        .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("procedure should compile");
    let entry = module.procedure_id("/proc/run").unwrap();
    let mut state = ExecutionState::new();
    let stale_parent = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/example").unwrap());
    let parents = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(parents)
        .unwrap()
        .add(Value::Datum(stale_parent));
    state.heap_mut().destroy_datum(stale_parent).unwrap();

    let result = execute_module_in_state(&module, entry, &[Value::List(parents)], &mut state)
        .unwrap()
        .as_number();
    assert_eq!(result, Some(1.0));
}

#[test]
fn values_equivalent_treats_nested_stale_refs_as_null() {
    let mut state = ExecutionState::new();
    let left_key = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/example").expect("type path"));
    let left_value = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/example").expect("type path"));
    let left = state.heap_mut().allocate_list();
    let right = state.heap_mut().allocate_list();
    let left_result = state
        .heap_mut()
        .list_mut(left)
        .expect("left list should be mutable")
        .set_key(Value::Datum(left_key), Value::Datum(left_value));
    assert!(left_result.is_none());
    let right_result = state
        .heap_mut()
        .list_mut(right)
        .expect("right list should be mutable")
        .set_key(Value::Null, Value::Null);
    assert!(right_result.is_none());
    state.heap_mut().destroy_datum(left_key).unwrap();
    state.heap_mut().destroy_datum(left_value).unwrap();

    assert!(
        super::values_equivalent(&Value::List(left), &Value::List(right), state.heap())
            .expect("equivalence should be comparable"),
        "stale key/value entries should be canonicalized to null"
    );
}

#[test]
fn compound_add_concatenates_text_onto_null() {
    let syntax = parse(
            "/proc/build_trigger(trigger)\n\tvar/all_triggers\n\tall_triggers += trigger\n\tall_triggers += null\n\treturn all_triggers\n",
        )
        .expect("voice trigger source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("voice trigger source should compile");
    assert_eq!(
        execute(&program, &[Value::text("come\\s*here")]),
        Ok(Value::text("come\\s*here"))
    );
}

#[test]
fn special_result_survives_branches_and_loops() {
    let source = "/proc/accumulate(input)\n\t. = 0\n\twhile(input > 0)\n\t\tif(input == 2)\n\t\t\t. += 10\n\t\telse\n\t\t\t. += input\n\t\tinput = input - 1\n";

    assert_eq!(execute_source(source, 3.0), Value::number(14.0));
}

#[test]
fn explicit_return_takes_precedence_over_special_result() {
    let syntax = parse("/proc/result()\n\t. = 5\n\treturn 9\n").expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");

    assert_eq!(execute(&program, &[]), Ok(Value::number(9.0)));
}

#[test]
fn type_predicate_builtins_classify_null_numbers_paths_and_subtypes() {
    for (source, expected) in [
        ("/proc/test()\n\treturn isnull(null)\n", 1.0),
        ("/proc/test()\n\treturn isnum(3)\n", 1.0),
        ("/proc/test()\n\treturn ispath(/datum/example)\n", 1.0),
        ("/proc/test()\n\treturn islist(list(1))\n", 1.0),
        ("/proc/test()\n\treturn islist(3)\n", 0.0),
        ("/proc/test()\n\treturn ismovable(new /atom/movable)\n", 1.0),
        (
            "/proc/test()\n\treturn ismovable(new /atom/movable/child)\n",
            1.0,
        ),
        ("/proc/test()\n\treturn ismovable(new /obj/item)\n", 1.0),
        ("/proc/test()\n\treturn ismovable(new /mob/living)\n", 1.0),
        ("/proc/test()\n\treturn ismovable(new /atom)\n", 0.0),
        ("/proc/test()\n\treturn ismovable(/obj/item)\n", 0.0),
        ("/proc/test()\n\treturn isturf(new /turf)\n", 1.0),
        ("/proc/test()\n\treturn isturf(new /turf/open/floor)\n", 1.0),
        ("/proc/test()\n\treturn isturf(new /obj/item)\n", 0.0),
        ("/proc/test()\n\treturn isturf(/turf/open/floor)\n", 0.0),
        ("/proc/test()\n\treturn isloc(new /area)\n", 1.0),
        ("/proc/test()\n\treturn isloc(new /turf/open/floor)\n", 1.0),
        ("/proc/test()\n\treturn isloc(new /obj/item)\n", 1.0),
        ("/proc/test()\n\treturn isloc(new /mob/living)\n", 1.0),
        (
            "/proc/test()\n\treturn isloc(new /turf, new /obj, new /mob)\n",
            1.0,
        ),
        ("/proc/test()\n\treturn isloc(new /turf, 3)\n", 0.0),
        ("/proc/test()\n\treturn isloc(/turf)\n", 0.0),
        (
            "/proc/test()\n\treturn istype(/datum/example, /datum)\n",
            0.0,
        ),
        (
            "/proc/test()\n\treturn istype(new /datum/example, /datum)\n",
            1.0,
        ),
        (
            "/proc/test()\n\tvar/list/ranks = list(\"Admin\")\n\treturn istype(ranks)\n",
            1.0,
        ),
        ("/proc/test()\n\treturn istype(3, /datum)\n", 0.0),
    ] {
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0])
            .expect("type predicate builtin should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::number(expected)));
        assert!(
            program
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instruction::TypePredicate { .. }))
        );
    }
}

#[test]
fn subtype_intervals_match_parent_walks_for_project_tree() {
    let datum = TypePath::parse("/datum").unwrap();
    let base = TypePath::parse("/datum/base").unwrap();
    let child = TypePath::parse("/datum/base/child").unwrap();
    let sibling = TypePath::parse("/datum/sibling").unwrap();
    let mut state = ExecutionState::new();
    state.set_type_parents(BTreeMap::from([
        (datum.clone(), None),
        (base.clone(), Some(datum.clone())),
        (child.clone(), Some(base.clone())),
        (sibling.clone(), Some(datum.clone())),
    ]));

    assert!(is_subtype(&state, &child, &child));
    assert!(is_subtype(&state, &child, &base));
    assert!(is_subtype(&state, &child, &datum));
    assert!(!is_subtype(&state, &base, &child));
    assert!(!is_subtype(&state, &child, &sibling));
}

#[test]
fn qualified_istype_keeps_client_colour_type_paths_on_the_constructor_branch() {
    let syntax = parse(concat!(
        "/proc/add_client_colour(colour_type_or_datum)\n",
        "\tvar/datum/client_colour/colour\n",
        "\tif(istype(colour_type_or_datum, /datum/client_colour))\n",
        "\t\tcolour = colour_type_or_datum\n",
        "\telse if(ispath(colour_type_or_datum, /datum/client_colour))\n",
        "\t\tcolour = new colour_type_or_datum\n",
        "\treturn colour\n",
        "/proc/qdel_fixture(datum/to_delete)\n",
        "\tif(!istype(to_delete))\n",
        "\t\treturn 0\n",
        "\tto_delete.gc_destroyed = 1\n",
        "\treturn to_delete.gc_destroyed\n",
        "/proc/run()\n",
        "\tvar/datum/client_colour/colour = add_client_colour(/datum/client_colour/blind)\n",
        "\tif(ispath(colour) || !istype(colour, /datum/client_colour))\n",
        "\t\treturn -2\n",
        "\treturn qdel_fixture(colour)\n",
    ))
    .expect("client-colour qdel fixture should parse");
    let procedures = syntax
        .definitions
        .iter()
        .filter(|definition| definition.kind == DefinitionKind::Procedure)
        .cloned()
        .collect::<Vec<_>>();
    let module = compile_module(&procedures)
        .expect("client-colour type path should compile through dynamic new");
    let entry = module.procedure_id("/proc/run").expect("run entry");
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(1.0)));
}

#[test]
fn unqualified_istype_uses_typed_parameter_and_local_declarations() {
    let syntax = parse(concat!(
            "/proc/remove_from_active(turf/open/T)\n",
            "\tif(istype(T))\n",
            "\t\treturn 9\n",
            "\treturn 7\n",
            "/proc/typed_parameter_is_list(list/datum/example/value)\n",
            "\treturn istype(value)\n",
            "/proc/typed_local_is_list(input)\n",
            "\tvar/list/datum/example/value = input\n",
            "\treturn istype(value)\n",
            "/proc/run()\n",
            "\tvar/turf/closed/wall = new\n",
            "\tvar/turf/open/floor = new\n",
            "\tvar/datum/example/not_a_list = new\n",
            "\tvar/list/items = list()\n",
            "\treturn remove_from_active(wall) * 1000 + remove_from_active(floor) * 100 + typed_parameter_is_list(not_a_list) * 10 + typed_parameter_is_list(items) + typed_local_is_list(not_a_list) * 10000 + typed_local_is_list(items) * 100000\n",
        ))
        .expect("typed istype fixture should parse");
    let module = compile_module(&syntax.definitions).expect("typed istype fixture should compile");
    let entry = module.procedure_id("/proc/run").expect("run entry");

    // The closed turf must fail the declared /turf/open guard without
    // touching the open-only field. Both parameter and local /list
    // declarations must reject an arbitrary datum and accept a list.
    assert_eq!(
        execute_module(&module, entry, &[]),
        Ok(Value::number(107_901.0))
    );
}

#[test]
fn typed_parameter_inference_ignores_repeated_name_in_default_expression() {
    let syntax = parse(concat!(
        "/proc/typed(datum/value = value)\n",
        "\treturn istype(value)\n",
        "/proc/run()\n",
        "\tvar/datum/example/item = new\n",
        "\treturn typed(item)\n",
    ))
    .expect("repeated parameter-name fixture should parse");
    let module = compile_module(&syntax.definitions)
        .expect("repeated parameter-name fixture should compile");
    let entry = module.procedure_id("/proc/run").expect("run entry");

    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(1.0)));
}

#[test]
fn typed_local_inference_ignores_repeated_name_in_initializer() {
    let syntax = parse(concat!(
        "/proc/typed_local(input)\n",
        "\tvar/datum/value = value\n",
        "\tvalue = input\n",
        "\treturn istype(value)\n",
        "/proc/run()\n",
        "\tvar/datum/example/item = new\n",
        "\treturn typed_local(item)\n",
    ))
    .expect("repeated local-name fixture should parse");
    let module =
        compile_module(&syntax.definitions).expect("repeated local-name fixture should compile");
    let entry = module.procedure_id("/proc/run").expect("run entry");

    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(1.0)));
}

#[test]
fn waitfor_directives_set_procedure_call_scheduling() {
    for (value, waits) in [("FALSE", false), ("TRUE", true), ("0", false), ("1", true)] {
        let syntax = parse(&format!(
            "/proc/scheduled()\n\tset waitfor = {value}\n\treturn 17\n"
        ))
        .expect("source should parse");
        let program =
            compile_procedure(&syntax.definitions[0]).expect("waitfor directive should compile");

        assert_eq!(execute(&program, &[]), Ok(Value::number(17.0)));
        assert_eq!(program.wait_for, waits);
    }
}

#[test]
fn waitfor_false_detaches_at_sleep_and_returns_current_dot_to_caller() {
    let syntax = parse(
            "/proc/c()\n\tsleep(1)\n\tsleep(1)\n\treturn 99\n\n/proc/b()\n\tset waitfor = FALSE\n\t. = 7\n\treturn c()\n\n/proc/a()\n\treturn b() + 1\n",
        )
        .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("waitfor chain should compile");
    let entry = module.procedure_id("/proc/a").unwrap();
    let mut state = ExecutionState::new();

    assert_eq!(
        execute_module_in_state(&module, entry, &[], &mut state),
        Ok(Value::number(8.0))
    );
    assert_eq!(state.scheduled_task_count(), 1);
    assert_eq!(
        advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
        Ok(vec![])
    );
    assert_eq!(state.scheduled_task_count(), 1);
    assert_eq!(
        advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
        Ok(vec![Value::number(99.0)])
    );
}

#[test]
fn waitfor_false_without_sleep_returns_normally_and_post_sleep_errors_are_scheduled() {
    let syntax = parse(
            "/proc/plain()\n\tset waitfor = 0\n\treturn 12\n\n/proc/fails_later()\n\tset waitfor = 0\n\t. = 4\n\tsleep(1)\n\tCRASH(\"later\")\n",
        )
        .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("waitfor procedures compile");
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/plain").unwrap(), &[]),
        Ok(Value::number(12.0))
    );

    let mut state = ExecutionState::new();
    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/fails_later").unwrap(),
            &[],
            &mut state,
        ),
        Ok(Value::number(4.0))
    );
    let error = advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state)
        .expect_err("errors after detachment belong to the scheduled continuation");
    assert!(error.message.contains("later"));
}

#[test]
fn waitfor_false_preserves_spawned_deletion_and_detached_src_context() {
    let syntax = parse(
            "/proc/run()\n\tset waitfor = FALSE\n\t. = 3\n\tspawn(1)\n\t\tqdel(src)\n\tsleep(2)\n\treturn 9\n",
        )
        .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("procedure should compile");
    let entry = module.procedure_id("/proc/run").unwrap();
    let mut state = ExecutionState::new();
    let datum = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/test").unwrap());
    let context = ExecutionContext::new(Value::Datum(datum), Value::Null);

    assert_eq!(
        execute_module_in_context(&module, entry, &[], &mut state, &context),
        Ok(Value::number(3.0))
    );
    assert_eq!(state.scheduled_task_count(), 2);
    advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state)
        .expect("spawned deletion should run");
    assert!(state.heap().datum(datum).is_err());
    assert_eq!(state.scheduled_task_count(), 1);
    assert_eq!(
        advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
        Ok(vec![Value::number(9.0)])
    );
}

#[test]
fn verb_set_directives_are_non_executable_metadata() {
    let syntax = parse(
            "/proc/metadata()\n\tset hidden = TRUE\n\tset category = \"Admin\"\n\tset desc = \"Example\"\n",
        )
        .expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("BYOND verb set directives should compile as metadata");
    assert_eq!(execute(&program, &[]), Ok(Value::Null));
}

#[test]
fn crash_statement_compiles_and_a_false_guard_skips_it() {
    let syntax =
        parse("/proc/guarded()\n\tif(FALSE)\n\t\tCRASH(\"should not execute\")\n\treturn 17\n")
            .expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("CRASH should compile even when its branch is not taken");

    assert_eq!(execute(&program, &[]), Ok(Value::number(17.0)));
}

#[test]
fn crash_statement_returns_a_source_mapped_runtime_error() {
    let syntax = parse("/proc/fail()\n\tif(TRUE)\n\t\tCRASH(\"loading id is required\")\n")
        .expect("source should parse");
    let crash_span = syntax.definitions[0].body[1].span;
    let program =
        compile_procedure(&syntax.definitions[0]).expect("CRASH statement should compile");
    let error = execute(&program, &[]).expect_err("taken CRASH must stop execution");

    assert_eq!(error.message, "CRASH: \"loading id is required\"");
    assert_eq!(error.source_span, Some(crash_span));
    assert_eq!(error.call_stack.len(), 1);
    assert_eq!(error.call_stack[0].procedure, "<standalone>");
    assert_eq!(error.call_stack[0].source_span, Some(crash_span));
}

#[test]
fn headless_locate_consumes_arguments_and_returns_null() {
    let syntax = parse("/proc/find()\n\treturn locate(1, 2, 3)\n").expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("headless locate should compile without a user procedure");

    assert!(
        program
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::Locate { argument_count: 3 }))
    );
    assert_eq!(execute(&program, &[]), Ok(Value::Null));
}

#[test]
fn world_dimension_changes_materialize_and_remove_coordinate_turfs() {
    let source = parse(
            "/world/proc/incrementMaxZ()\n\tworld.maxz++\n/proc/grow()\n\tworld.incrementMaxZ()\n\tworld.maxx = 2\n\tworld.maxy = 2\n\tvar/turf/found = locate(2, 2, 2)\n\treturn istype(found, /turf) + (found.x == 2) + (found.y == 2) + (found.z == 2) + istype(found.loc, /area)\n/proc/shrink()\n\tworld.maxz = 1\n\treturn isnull(locate(2, 2, 2))\n",
        )
        .expect("world geometry source should parse");
    let module = compile_module(&source.definitions).expect("world geometry should compile");
    let mut state = ExecutionState::new();
    let world = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/world").unwrap());
    for (name, value) in [
        ("maxx", Value::number(1.0)),
        ("maxy", Value::number(1.0)),
        ("maxz", Value::number(1.0)),
        ("area", Value::TypePath(TypePath::parse("/area").unwrap())),
        ("turf", Value::TypePath(TypePath::parse("/turf").unwrap())),
    ] {
        state
            .heap_mut()
            .set_datum_field(world, field(name), value)
            .unwrap();
    }
    state.set_global(field("world"), Value::Datum(world));

    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/grow").unwrap(),
            &[],
            &mut state,
        ),
        Ok(Value::number(5.0)),
    );
    assert_eq!(state.world_turfs.len(), 8);
    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/shrink").unwrap(),
            &[],
            &mut state,
        ),
        Ok(Value::number(1.0)),
    );
    assert_eq!(state.world_turfs.len(), 4);
}

#[test]
fn world_geometry_honors_sparse_declared_area_and_turf_types_and_refreshes_area_cache() {
    let world_path = TypePath::parse("/world").unwrap();
    let space_area = TypePath::parse("/area/space").unwrap();
    let replacement_area = TypePath::parse("/area/replacement").unwrap();
    let space_turf = TypePath::parse("/turf/open/space/basic").unwrap();
    let mut state = ExecutionState::new();
    state.set_initial_values(BTreeMap::from([(
        world_path.clone(),
        BTreeMap::from([
            (field("area"), Value::TypePath(space_area.clone())),
            (field("turf"), Value::TypePath(space_turf.clone())),
        ]),
    )]));
    let world = state.heap_mut().allocate_datum(world_path);

    state
        .resize_world_geometry(world, (1, 1, 1))
        .expect("sparse world defaults should create initial geometry");
    let first_turf = state.turf_at(1, 1, 1).unwrap();
    assert_eq!(
        state.heap().datum(first_turf).unwrap().type_path(),
        &space_turf
    );
    let Value::Datum(first_area) = datum_field_or_initial(&state, first_turf, &field("loc"))
        .expect("new turf should have an area")
    else {
        panic!("new turf loc should be an area datum");
    };
    assert_eq!(
        state.heap().datum(first_area).unwrap().type_path(),
        &space_area
    );

    state
        .heap_mut()
        .set_datum_field(
            world,
            field("area"),
            Value::TypePath(replacement_area.clone()),
        )
        .unwrap();
    state
        .resize_world_geometry(world, (1, 1, 2))
        .expect("changed world.area should refresh cached default area");
    let second_turf = state.turf_at(1, 1, 2).unwrap();
    let Value::Datum(second_area) = datum_field_or_initial(&state, second_turf, &field("loc"))
        .expect("expanded turf should have an area")
    else {
        panic!("expanded turf loc should be an area datum");
    };
    assert_eq!(
        state.heap().datum(second_area).unwrap().type_path(),
        &replacement_area
    );
}

#[test]
fn incremental_world_growth_keeps_existing_turf_identity_and_fills_new_slabs() {
    let world_path = TypePath::parse("/world").unwrap();
    let mut state = ExecutionState::new();
    state.set_initial_values(BTreeMap::from([(
        world_path.clone(),
        BTreeMap::from([
            (
                field("area"),
                Value::TypePath(TypePath::parse("/area").unwrap()),
            ),
            (
                field("turf"),
                Value::TypePath(TypePath::parse("/turf").unwrap()),
            ),
        ]),
    )]));
    let world = state.heap_mut().allocate_datum(world_path);

    state.resize_world_geometry(world, (3, 3, 2)).unwrap();
    let mut original = std::collections::BTreeMap::new();
    for z in 1..=2 {
        for y in 1..=3 {
            for x in 1..=3 {
                original.insert((x, y, z), state.turf_at(x, y, z).expect("turf exists"));
            }
        }
    }

    // Grow one slab at a time, exactly as a DM engine drives `world.maxz++`.
    state.resize_world_geometry(world, (3, 3, 3)).unwrap();
    state.resize_world_geometry(world, (3, 3, 4)).unwrap();

    // The fast existence probe must not recreate turfs that already exist.
    for (&(x, y, z), &turf) in &original {
        assert_eq!(
            state.turf_at(x, y, z),
            Some(turf),
            "existing turf identity at {x},{y},{z} must be stable across incremental growth"
        );
    }
    // New slabs are fully populated.
    for z in 3..=4 {
        for y in 1..=3 {
            for x in 1..=3 {
                let turf = state.turf_at(x, y, z).expect("new slab turf exists");
                assert_eq!(
                    state.heap().datum_field(turf, &field("z")).unwrap(),
                    &Value::number(z as f32)
                );
            }
        }
    }

    // Shrinking then regrowing (a separate resize each way, as an engine does)
    // drops the removed slab and rebuilds it with fresh turfs.
    let dropped = state.turf_at(1, 1, 4).unwrap();
    state.resize_world_geometry(world, (3, 3, 2)).unwrap();
    assert_eq!(state.turf_at(1, 1, 4), None);
    state.resize_world_geometry(world, (3, 3, 4)).unwrap();
    let regrown = state.turf_at(1, 1, 4).expect("regrown slab turf exists");
    assert_ne!(regrown, dropped);
    // The slab that was never removed keeps its identity throughout.
    assert_eq!(state.turf_at(1, 1, 1), Some(original[&(1, 1, 1)]));
}

#[test]
fn world_geometry_preserves_runtime_turf_initializer_programs() {
    let turf_path = TypePath::parse("/turf/runtime_initialized").unwrap();
    let world_path = TypePath::parse("/world").unwrap();
    let syntax = parse("/proc/initial_density()\n\treturn 7\n").unwrap();
    let module = Arc::new(compile_module(&syntax.definitions).unwrap());
    let entry = module.procedure_id("/proc/initial_density").unwrap();
    let mut state = ExecutionState::new();
    state.set_initial_values(BTreeMap::from([(
        world_path.clone(),
        BTreeMap::from([(field("turf"), Value::TypePath(turf_path.clone()))]),
    )]));
    state.set_instance_initializers(
        Arc::new(BTreeMap::from([(
            turf_path,
            vec![InstanceInitializer::Program {
                field: field("density"),
                entry,
            }],
        )])),
        Some(module),
    );
    let world = state.heap_mut().allocate_datum(world_path);
    state.resize_world_geometry(world, (2, 1, 1)).unwrap();
    for x in 1..=2 {
        let turf = state.turf_at(x, 1, 1).unwrap();
        assert_eq!(
            state.heap().datum_field(turf, &field("density")).unwrap(),
            &Value::number(7.0)
        );
    }
}

#[test]
fn bulk_turf_fill_replays_runtime_initializers_without_aliasing_list_state() {
    // `world.turf` with a runtime initializer that allocates a fresh list per
    // instance: the bulk fill must run the program once for a template and then
    // give every sibling its own distinct, independently mutable list.
    let turf_path = TypePath::parse("/turf/space").unwrap();
    let world_path = TypePath::parse("/world").unwrap();
    let syntax = parse(
        "/proc/init_scalar()\n\treturn 42\n/proc/init_list()\n\tvar/list/L = list()\n\tL += \"seed\"\n\treturn L\n",
    )
    .unwrap();
    let module = Arc::new(compile_module(&syntax.definitions).unwrap());
    let scalar_entry = module.procedure_id("/proc/init_scalar").unwrap();
    let list_entry = module.procedure_id("/proc/init_list").unwrap();
    let mut state = ExecutionState::new();
    state.set_initial_values(BTreeMap::from([(
        world_path.clone(),
        BTreeMap::from([(field("turf"), Value::TypePath(turf_path.clone()))]),
    )]));
    state.set_instance_initializers(
        Arc::new(BTreeMap::from([(
            turf_path,
            vec![
                InstanceInitializer::Program {
                    field: field("blueprint_data"),
                    entry: list_entry,
                },
                InstanceInitializer::Program {
                    field: field("temperature"),
                    entry: scalar_entry,
                },
            ],
        )])),
        Some(module),
    );
    let world = state.heap_mut().allocate_datum(world_path);
    state.set_global(field("world"), Value::Datum(world));
    state.resize_world_geometry(world, (3, 2, 1)).unwrap();

    let mut lists = Vec::new();
    for y in 1..=2 {
        for x in 1..=3 {
            let turf = state.turf_at(x, y, 1).expect("turf exists");
            assert_eq!(
                state
                    .heap()
                    .datum_field(turf, &field("temperature"))
                    .unwrap(),
                &Value::number(42.0),
                "scalar initializer result is replayed onto every cell"
            );
            let Value::List(list) = state
                .heap()
                .datum_field(turf, &field("blueprint_data"))
                .unwrap()
            else {
                panic!("list initializer result must be a list on every cell");
            };
            assert_eq!(state.heap().list(*list).unwrap().len(), 1);
            lists.push(*list);
        }
    }
    // Every cell owns a distinct list handle.
    let mut unique = lists.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(
        unique.len(),
        lists.len(),
        "no two cells share a list handle"
    );

    // Mutating one cell's list leaves the others untouched.
    state
        .heap_mut()
        .list_mut(lists[0])
        .unwrap()
        .add(Value::text("mutated"));
    assert_eq!(state.heap().list(lists[0]).unwrap().len(), 2);
    assert_eq!(state.heap().list(lists[1]).unwrap().len(), 1);
}

#[test]
fn engine_created_turfs_share_effective_defaults_at_world_scale() {
    let syntax = parse(
            "/proc/observe(turf/cell)\n\tvar/list/reflection = cell.vars\n\tvar/before = cell.density + reflection[\"density\"]\n\tvar/declared = initial(cell.density)\n\tvar/native_bounds = bounds_dist(cell, cell)\n\t++cell.density\n\treturn list(before, declared, native_bounds, cell.density, reflection[\"density\"], cell.name)\n/proc/retype(turf/cell)\n\tvar/turf/replaced = new /turf/compact/changed(cell)\n\treturn list(replaced, replaced.density, replaced.name, replaced.x, replaced.vars[\"density\"])\n",
        )
        .expect("compact turf fixture should parse");
    let module = compile_module(&syntax.definitions).expect("compact turf fixture compiles");
    let mut state = ExecutionState::new();
    let world = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/world").unwrap());
    let world_contents = state.heap_mut().allocate_list();
    for (name, value) in [
        (
            "area",
            Value::TypePath(TypePath::parse("/area/default").unwrap()),
        ),
        (
            "turf",
            Value::TypePath(TypePath::parse("/turf/compact").unwrap()),
        ),
        ("contents", Value::List(world_contents)),
    ] {
        state
            .heap_mut()
            .set_datum_field(world, field(name), value)
            .unwrap();
    }
    state.set_global(field("world"), Value::Datum(world));
    state.set_type_parents(BTreeMap::from([
        (
            TypePath::parse("/area/default").unwrap(),
            Some(TypePath::parse("/area").unwrap()),
        ),
        (
            TypePath::parse("/turf/compact").unwrap(),
            Some(TypePath::parse("/turf").unwrap()),
        ),
        (
            TypePath::parse("/turf/compact/changed").unwrap(),
            Some(TypePath::parse("/turf/compact").unwrap()),
        ),
    ]));
    let effective = |path: &str, density: f32, name: &str| {
        let mut fields = BTreeMap::from([
            (field("contents"), Value::Null),
            (field("density"), Value::number(density)),
            (field("bound_height"), Value::number(64.0)),
            (field("bound_width"), Value::number(64.0)),
            (field("loc"), Value::Null),
            (field("name"), Value::text(name)),
            (field("x"), Value::number(0.0)),
            (field("y"), Value::number(0.0)),
            (field("z"), Value::number(0.0)),
        ]);
        for index in 0..96 {
            fields.insert(
                field(&format!("inherited_scalar_{index}")),
                Value::number(index as f32),
            );
        }
        (TypePath::parse(path).unwrap(), fields)
    };
    state.set_initial_values(BTreeMap::from([
        effective("/turf/compact", 1.0, "compact"),
        effective("/turf/compact/changed", 7.0, "changed"),
    ]));
    let changed_path = TypePath::parse("/turf/compact/changed").unwrap();
    state.set_instance_initializers(
        Arc::new(BTreeMap::from([(
            changed_path,
            vec![
                InstanceInitializer::Constant {
                    field: field("x"),
                    value: Value::number(0.0),
                },
                InstanceInitializer::Constant {
                    field: field("y"),
                    value: Value::number(0.0),
                },
                InstanceInitializer::Constant {
                    field: field("z"),
                    value: Value::number(0.0),
                },
                InstanceInitializer::Constant {
                    field: field("loc"),
                    value: Value::Null,
                },
                InstanceInitializer::Constant {
                    field: field("contents"),
                    value: Value::Null,
                },
            ],
        )])),
        None,
    );

    let lists_before_resize = state.heap().live_list_count();
    state
        .resize_world_geometry(world, (255, 255, 2))
        .expect("two full-size z levels should materialize compactly");
    let turf_count = 255 * 255 * 2;
    assert_eq!(state.world_turfs.len(), turf_count);
    let materialized_fields = state
        .world_turfs
        .values()
        .map(|turf| state.heap().datum(*turf).unwrap().field_len())
        .sum::<usize>();
    assert_eq!(materialized_fields, turf_count * 4);
    assert_eq!(
        state.heap().live_list_count() - lists_before_resize,
        1,
        "world expansion allocates the shared area's contents only, saving one empty list per turf",
    );
    let effective_fields_per_turf = 9 + 96;
    let field_slot_bytes = std::mem::size_of::<(FieldName, Value)>();
    let compact_field_bytes = materialized_fields * field_slot_bytes;
    let eager_field_bytes = turf_count * effective_fields_per_turf * field_slot_bytes;
    eprintln!(
        "compact-turf-scale turfs={turf_count} fields={materialized_fields} field_slot_bytes={field_slot_bytes} compact_field_bytes={compact_field_bytes} eager_field_bytes={eager_field_bytes} datum_inline_bytes={} contents_list_inline_bytes={}",
        std::mem::size_of::<dm_value::Datum>(),
        std::mem::size_of::<dm_value::DmList>(),
    );
    assert!(eager_field_bytes >= compact_field_bytes * 20);
    assert!(state.world_turfs.values().all(|turf| {
        state
            .heap()
            .datum(*turf)
            .is_ok_and(|datum| datum.field_len() == 4)
    }));

    let cell = state.turf_at(255, 255, 2).expect("corner turf exists");
    let observed = execute_module_in_state(
        &module,
        module.procedure_id("/proc/observe").unwrap(),
        &[Value::Datum(cell)],
        &mut state,
    )
    .expect("shared defaults should read and mutate through fields and vars");
    let Value::List(observed) = observed else {
        panic!("observe should return a list")
    };
    assert_eq!(
        state
            .heap()
            .list(observed)
            .unwrap()
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>(),
        [
            Value::number(2.0),
            Value::number(1.0),
            Value::number(-64.0),
            Value::number(2.0),
            Value::number(2.0),
            Value::text("compact"),
        ]
    );
    assert_eq!(state.heap().datum(cell).unwrap().field_len(), 5);
    state.ensure_contents(cell).unwrap();

    let structural_fields = ["x", "y", "z", "loc", "contents"];
    let structural_before = structural_fields.map(|name| {
        state
            .heap()
            .datum_field(cell, &field(name))
            .unwrap_or_else(|error| panic!("missing structural field {name}: {error}"))
            .clone()
    });

    let retyped = execute_module_in_state(
        &module,
        module.procedure_id("/proc/retype").unwrap(),
        &[Value::Datum(cell)],
        &mut state,
    )
    .expect("turf replacement should preserve the compact map cell");
    let Value::List(retyped) = retyped else {
        panic!("retype should return a list")
    };
    assert_eq!(
        state
            .heap()
            .list(retyped)
            .unwrap()
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>(),
        [
            Value::Datum(cell),
            Value::number(7.0),
            Value::text("changed"),
            Value::number(255.0),
            Value::number(7.0),
        ]
    );
    for (name, expected) in structural_fields.into_iter().zip(structural_before) {
        assert_eq!(
            state.heap().datum_field(cell, &field(name)),
            Ok(&expected),
            "retype constant replay must preserve map-cell field {name}",
        );
    }
    assert_eq!(state.heap().datum(cell).unwrap().field_len(), 5);
}

#[test]
fn spatial_contents_and_atom_new_preserve_byond_map_cell_identity() {
    let source = parse(
            "/turf/floor/New(where)\n\tsrc.saw_cell = (src == where) + (src.x == 2) + (src.y == 2) + (src.z == 2)\n/obj/item/New(where)\n\tsrc.saw_location = (src.loc == where) + (src.x == 2) + (src.y == 2) + (src.z == 2)\n/proc/load_cell(area/target_area)\n\tvar/turf/original = locate(2, 2, 2)\n\ttarget_area.contents.Add(original)\n\tvar/turf/replaced = new /turf/floor(original)\n\tvar/obj/item = new /obj/item(original)\n\treturn list(original, replaced, item)\n",
        )
        .expect("spatial construction source should parse");
    let module = compile_module(&source.definitions).expect("spatial construction should compile");
    let mut state = ExecutionState::new();
    let world = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/world").unwrap());
    for (name, value) in [
        ("maxx", Value::number(2.0)),
        ("maxy", Value::number(2.0)),
        ("maxz", Value::number(2.0)),
        (
            "area",
            Value::TypePath(TypePath::parse("/area/default").unwrap()),
        ),
        (
            "turf",
            Value::TypePath(TypePath::parse("/turf/default").unwrap()),
        ),
    ] {
        state
            .heap_mut()
            .set_datum_field(world, field(name), value)
            .unwrap();
    }
    state.set_global(field("world"), Value::Datum(world));
    state.resize_world_geometry(world, (2, 2, 2)).unwrap();
    let original = state.turf_at(2, 2, 2).expect("corner turf");
    let old_area = match state.heap().datum_field(original, &field("loc")).unwrap() {
        Value::Datum(area) => *area,
        value => panic!("expected old area, got {value:?}"),
    };
    let new_area =
        allocate_initialized_datum(&mut state, TypePath::parse("/area/replacement").unwrap())
            .unwrap();

    let result = execute_module_in_state(
        &module,
        module.procedure_id("/proc/load_cell").unwrap(),
        &[Value::Datum(new_area)],
        &mut state,
    )
    .expect("map-shaped spatial operations should execute");
    let Value::List(result) = result else {
        panic!("expected result list")
    };
    let values = state
        .heap()
        .list(result)
        .unwrap()
        .positions()
        .map(|(_, value)| value.clone())
        .collect::<Vec<_>>();
    assert_eq!(values[0], Value::Datum(original));
    assert_eq!(
        values[1],
        Value::Datum(original),
        "new turf preserves cell identity"
    );
    let Value::Datum(item) = values[2] else {
        panic!("expected created item")
    };
    let turf = state.heap().datum(original).unwrap();
    assert_eq!(turf.type_path().as_str(), "/turf/floor");
    assert_eq!(turf.field(&field("loc")), Ok(&Value::Datum(new_area)));
    assert_eq!(turf.field(&field("saw_cell")), Ok(&Value::number(4.0)));
    let item_datum = state.heap().datum(item).unwrap();
    assert_eq!(item_datum.field(&field("loc")), Ok(&Value::Datum(original)));
    assert_eq!(
        item_datum.field(&field("saw_location")),
        Ok(&Value::number(4.0)),
        "movable loc and coordinates exist before New"
    );
    let old_contents = match state
        .heap()
        .datum_field(old_area, &field("contents"))
        .unwrap()
    {
        Value::List(list) => *list,
        _ => unreachable!(),
    };
    assert!(
        !state
            .heap()
            .list(old_contents)
            .unwrap()
            .contains(&Value::Datum(original))
    );
    let area_contents = match state
        .heap()
        .datum_field(new_area, &field("contents"))
        .unwrap()
    {
        Value::List(list) => *list,
        _ => unreachable!(),
    };
    let area_values = state.heap().list(area_contents).unwrap();
    assert_eq!(
        area_values
            .positions()
            .filter(|(_, value)| value.semantic_eq(&Value::Datum(original)))
            .count(),
        1
    );
    assert_eq!(
        area_values
            .positions()
            .filter(|(_, value)| value.semantic_eq(&Value::Datum(item)))
            .count(),
        1
    );
}

#[test]
fn datum_vars_loc_write_uses_engine_spatial_assignment() {
    let source = parse(
            "/proc/preload_loc(atom/movable/thing, turf/destination)\n\tthing.vars[\"loc\"] = destination\n\treturn thing.loc\n",
        )
        .expect("preloader-shaped vars write should parse");
    let module = compile_module(&source.definitions).expect("vars write should compile");
    let preload = module
        .procedure(module.procedure_id("/proc/preload_loc").unwrap())
        .unwrap();
    assert!(
        preload
            .instructions
            .contains(&Instruction::StoreDynamicField),
        "preloader vars assignment should bypass full datum.vars materialization"
    );
    assert!(!preload.instructions.contains(&Instruction::LoadDatumVars));
    let mut state = ExecutionState::new();
    let world = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/world").unwrap());
    for (name, value) in [
        ("maxx", Value::number(2.0)),
        ("maxy", Value::number(1.0)),
        ("maxz", Value::number(1.0)),
        ("area", Value::TypePath(TypePath::parse("/area").unwrap())),
        ("turf", Value::TypePath(TypePath::parse("/turf").unwrap())),
    ] {
        state
            .heap_mut()
            .set_datum_field(world, field(name), value)
            .unwrap();
    }
    state.set_global(field("world"), Value::Datum(world));
    state.resize_world_geometry(world, (2, 1, 1)).unwrap();
    let old_turf = state.turf_at(1, 1, 1).unwrap();
    let new_turf = state.turf_at(2, 1, 1).unwrap();
    let movable = allocate_initialized_datum(
        &mut state,
        TypePath::parse("/atom/movable/preloaded").unwrap(),
    )
    .unwrap();
    super::builtins::move_movable_to_turf(&mut state, movable, old_turf).unwrap();

    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/preload_loc").unwrap(),
            &[Value::Datum(movable), Value::Datum(new_turf)],
            &mut state,
        ),
        Ok(Value::Datum(new_turf))
    );
    for (turf, expected) in [(old_turf, 0), (new_turf, 1)] {
        let Value::List(contents) = state.heap().datum_field(turf, &field("contents")).unwrap()
        else {
            panic!("turf contents must be a list")
        };
        assert_eq!(
            state
                .heap()
                .list(*contents)
                .unwrap()
                .positions()
                .filter(|(_, value)| value.semantic_eq(&Value::Datum(movable)))
                .count(),
            expected
        );
    }
    let datum = state.heap().datum(movable).unwrap();
    assert_eq!(datum.field(&field("x")), Ok(&Value::number(2.0)));
    assert_eq!(datum.field(&field("y")), Ok(&Value::number(1.0)));
    assert_eq!(datum.field(&field("z")), Ok(&Value::number(1.0)));
    assert!(!state.datum_vars_by_datum.contains_key(&movable));
}

#[test]
#[ignore = "startup mapping datum.vars materialization benchmark"]
fn preloader_dynamic_field_write_benchmark() {
    const FIELDS: usize = 128;
    const ROUNDS: usize = 250;
    let source = parse(concat!(
        "/proc/direct(datum/target, name, value)\n",
        "\ttarget.vars[name] = value\n",
        "/proc/materialized(datum/target, name, value)\n",
        "\tvar/list/reflection = target.vars\n",
        "\treflection[name] = value\n",
    ))
    .unwrap();
    let module = compile_module(&source.definitions).unwrap();
    let direct = module.procedure_id("/proc/direct").unwrap();
    let materialized = module.procedure_id("/proc/materialized").unwrap();
    let path = TypePath::parse("/datum/mapping_fixture").unwrap();
    let defaults = (0..FIELDS)
        .map(|index| {
            (
                FieldName::parse(&format!("field_{index}")).unwrap(),
                Value::number(index as f32),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let run = |procedure| {
        let mut state = ExecutionState::new();
        state.set_initial_values(BTreeMap::from([(path.clone(), defaults.clone())]));
        let started = Instant::now();
        for index in 0..ROUNDS {
            let datum = state.heap_mut().allocate_datum(path.clone());
            execute_module_in_state(
                &module,
                procedure,
                &[
                    Value::Datum(datum),
                    Value::text("field_127"),
                    Value::number(index as f32),
                ],
                &mut state,
            )
            .unwrap();
        }
        started.elapsed()
    };
    let old = run(materialized);
    let direct = run(direct);
    eprintln!(
        "preloader-vars fields={FIELDS} rounds={ROUNDS} materialized_ms={} direct_ms={} speedup={:.2}",
        old.as_millis(),
        direct.as_millis(),
        old.as_secs_f64() / direct.as_secs_f64(),
    );
    assert!(direct < old);
}

#[test]
fn regex_builtin_constructs_a_regex_datum_with_pattern_and_flags() {
    let syntax = parse("/proc/build()\n\treturn regex(@\"[a-z]+\", \"ig\")\n")
        .expect("regex source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("the built-in regex constructor should compile");
    assert!(
        program
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::MakeRegex { argument_count: 2 }))
    );

    let mut state = ExecutionState::new();
    let result =
        execute_in_state(&program, &[], &mut state).expect("regex constructor should execute");
    let Value::Datum(regex) = result else {
        panic!("regex constructor should return a datum");
    };
    let datum = state.heap().datum(regex).expect("regex datum should exist");
    assert_eq!(datum.type_path().to_string(), "/regex");
    assert_eq!(
        datum_field_or_initial(&state, regex, &field("datum_flags")),
        Ok(Value::number(0.0))
    );
    assert_eq!(
        datum_field_or_initial(&state, regex, &field("tag")),
        Ok(Value::Null)
    );
    assert_eq!(
        datum.field(&field("_dream64_pattern")),
        Ok(&Value::text("[a-z]+"))
    );
    assert_eq!(datum.field(&field("text")), Ok(&Value::Null));
    assert_eq!(datum.field(&field("flags")), Ok(&Value::text("ig")));
    assert_eq!(datum.field(&field("match")), Ok(&Value::Null));
    assert_eq!(datum.field(&field("index")), Ok(&Value::number(0.0)));
    assert_eq!(datum.field(&field("group")), Ok(&Value::Null));
    assert_eq!(datum.field(&field("next")), Ok(&Value::Null));
}

#[test]
fn word_filter_findtext_accepts_regex_needles() {
    let syntax = parse(
        "/proc/run(value)\n\tvar/regex/word = regex(@\"^\\w+$\")\n\treturn findtext(value, word)\n",
    )
    .expect("word-filter regex source should parse");
    let module = compile_module(&syntax.definitions).expect("regex findtext should compile");
    let entry = module.procedure_id("/proc/run").unwrap();
    assert_eq!(
        execute_module(&module, entry, &[Value::text("admin")]),
        Ok(Value::number(1.0))
    );
    assert_eq!(
        execute_module(&module, entry, &[Value::text("admin help")]),
        Ok(Value::number(0.0))
    );
}

#[test]
fn parsed_map_searches_treat_null_file_text_as_an_empty_haystack() {
    let syntax = parse(concat!(
        "/proc/run()\n",
        "\tvar/tfile = null\n",
        "\tvar/regex/matches_tgm = regex(@\"^//MAP CONVERTED BY dmm2tgm.py\")\n",
        "\tvar/regex/dmm_regex = regex(@\"[A-Za-z]+\", \"g\")\n",
        "\treturn findtext(tfile, matches_tgm) + dmm_regex.Find(tfile, 1)\n",
    ))
    .expect("parsed-map null-file search source should parse");
    let module =
        compile_module(&syntax.definitions).expect("parsed-map null-file searches should compile");
    assert_eq!(
        execute_module(&module, module.procedure_id("/proc/run").unwrap(), &[]),
        Ok(Value::number(0.0)),
    );
}

#[test]
fn declared_initial_fields_remain_visible_on_sparse_non_map_datums() {
    let syntax = parse(concat!(
        "/proc/attach(location)\n",
        "\tif(!location.important_recursive_contents)\n",
        "\t\tlocation.important_recursive_contents = list()\n",
        "\treturn length(location.important_recursive_contents)\n",
        "/proc/read_unknown(location)\n",
        "\treturn location.not_a_declared_field\n",
    ))
    .expect("area-sensitive sparse-field source should parse");
    let module = compile_module(&syntax.definitions)
        .expect("area-sensitive sparse-field source should compile");
    let path = TypePath::parse("/obj/item/stack/ore/gold").unwrap();
    let mut state = ExecutionState::new();
    let movable = TypePath::parse("/atom/movable").unwrap();
    state.set_type_parents(BTreeMap::from([
        (path.clone(), Some(movable.clone())),
        (movable.clone(), None),
    ]));
    state.set_initial_values(BTreeMap::from([(
        movable,
        BTreeMap::from([(field("important_recursive_contents"), Value::Null)]),
    )]));
    // Model a datum produced by an allocation path that retained only its
    // identity/type. It is intentionally not in compact_default_datums.
    let ore = state.heap_mut().allocate_datum(path);

    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/attach").unwrap(),
            &[Value::Datum(ore)],
            &mut state,
        ),
        Ok(Value::number(0.0)),
    );
    assert!(matches!(
        state
            .heap()
            .datum_field(ore, &field("important_recursive_contents")),
        Ok(Value::List(_)),
    ));
    let error = execute_module_in_state(
        &module,
        module.procedure_id("/proc/read_unknown").unwrap(),
        &[Value::Datum(ore)],
        &mut state,
    )
    .expect_err("a genuinely unknown field must remain an error");
    assert!(error.to_string().contains("not_a_declared_field"));
}

#[test]
fn immune_system_findtext_treats_a_blood_type_datum_as_no_match() {
    let syntax = parse("/proc/run(value)\n\treturn findtext(value, \"+\")\n")
        .expect("immune-system findtext source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("immune-system findtext source should compile");
    let mut state = ExecutionState::new();
    let blood_type = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/blood_type/human/o_plus").unwrap());
    state
        .heap_mut()
        .set_datum_field(blood_type, field("name"), Value::text("O+"))
        .unwrap();

    assert_eq!(
        execute_in_state(&program, &[Value::Datum(blood_type)], &mut state),
        Ok(Value::number(0.0)),
    );
    assert_eq!(
        execute_in_state(&program, &[Value::text("O+")], &mut state),
        Ok(Value::number(2.0)),
    );
}

#[test]
fn multiline_global_regex_find_advances_and_populates_capture_groups() {
    let syntax = parse(
            "/proc/run(text)\n\tvar/regex/entries = new(@\"^(?!#)(.+?)\\s+=\\s+(.+)\", \"gm\")\n\tvar/result = \"\"\n\twhile(entries.Find(text))\n\t\tresult += \"[entries.group[1]]:[entries.group[2]]|\"\n\treturn result\n",
        )
        .expect("admins regex source should parse");
    let module = compile_module(&syntax.definitions).expect("regex.Find should compile");
    let entry = module.procedure_id("/proc/run").unwrap();
    assert_eq!(
        execute_module(
            &module,
            entry,
            &[Value::text(
                "# ignored = nope\nAlice = Admin\nBob = Moderator\n"
            )],
        ),
        Ok(Value::text("Alice:Admin|Bob:Moderator|"))
    );
}

#[test]
fn global_regex_exposes_byond_next_and_text_for_dmm_style_explicit_sweeps() {
    let syntax = parse(concat!(
        "/proc/run(text)\n",
        "\tvar/regex/entries = new(@\"([A-Za-z]+)\", \"g\")\n",
        "\tvar/stored_index = 1\n",
        "\tvar/count = 0\n",
        "\twhile(entries.Find(text, stored_index))\n",
        "\t\tstored_index = entries.next\n",
        "\t\tcount++\n",
        "\treturn list(count, stored_index, entries.index, entries.match, entries.text)\n",
    ))
    .expect("DMM-style regex source should parse");
    let module = compile_module(&syntax.definitions).expect("DMM-style regex should compile");
    let mut state = ExecutionState::new();
    let result = execute_module_in_state(
        &module,
        module.procedure_id("/proc/run").unwrap(),
        &[Value::text("aa bb")],
        &mut state,
    )
    .expect("DMM-style regex sweep should execute");
    let Value::List(result) = result else {
        panic!("regex state should be returned in a list")
    };
    let values = state
        .heap()
        .list(result)
        .unwrap()
        .positions()
        .map(|(_, value)| value.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        values,
        vec![
            Value::number(2.0),
            Value::number(6.0),
            Value::number(4.0),
            Value::text("bb"),
            Value::text("aa bb"),
        ]
    );
}

#[test]
fn station_dmm_regex_parses_model_and_coordinate_records_in_one_global_sweep() {
    let syntax = parse(
            r#"/proc/run(tfile)
	var/regex/dmm_regex = new(@'"([a-zA-Z]+)" = (?:\(\n|\()((?:.|\n)*?)\)\n(?!\t)|\((\d+),(\d+),(\d+)\) = \{"([a-zA-Z\n]*)"\}', "g")
	var/stored_index = 1
	var/list/grid_models = list()
	var/coordinate_records = 0
	var/grid_line_count = 0
	while(dmm_regex.Find(tfile, stored_index))
		stored_index = dmm_regex.next
		var/list/regex_output = dmm_regex.group
		if(regex_output[1])
			grid_models[regex_output[1]] = regex_output[2]
		else if(regex_output[3])
			coordinate_records++
			var/list/grid_lines = splittext(regex_output[6], "\n")
			grid_line_count = length(grid_lines)
	return length(grid_models) * 100 + coordinate_records * 10 + grid_line_count
"#,
        )
        .expect("station DMM regex source should parse");
    let module = compile_module(&syntax.definitions).expect("station DMM regex should compile");
    assert_eq!(
        execute_module(
            &module,
            module.procedure_id("/proc/run").unwrap(),
            &[Value::text(
                "\"a\" = (/area,/turf)\n(1,1,1) = {\"\naaa\nbbb\n\"}\n"
            )],
        ),
        Ok(Value::number(114.0))
    );
}

#[test]
fn native_discover_offset_scans_fixed_width_grid_keys_in_source_order() {
    let mut state = ExecutionState::new();
    let field = |name| FieldName::parse(name).unwrap();
    let template = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/map_template").unwrap());
    let parsed = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/parsed_map").unwrap());
    let models = state.heap_mut().allocate_list();
    state.mark_associative_list(models);
    state
        .heap_mut()
        .list_mut(models)
        .unwrap()
        .set_key(Value::text("aa"), Value::text("/turf,/obj/other"));
    state.heap_mut().list_mut(models).unwrap().set_key(
        Value::text("bb"),
        Value::text("/turf,/OBJ/MODULAR_MAP_CONNECTOR"),
    );
    let lines = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(lines)
        .unwrap()
        .extend_positional([Value::text("aaaaaaaa"), Value::text("aabbbbaa")]);
    let grid = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/grid_set").unwrap());
    for (name, value) in [
        ("xcrd", Value::number(10.0)),
        ("ycrd", Value::number(20.0)),
        ("gridLines", Value::List(lines)),
    ] {
        state
            .heap_mut()
            .set_datum_field(grid, field(name), value)
            .unwrap();
    }
    let grids = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(grids)
        .unwrap()
        .add(Value::Datum(grid));
    for (name, value) in [
        ("grid_models", Value::List(models)),
        ("gridSets", Value::List(grids)),
        ("key_len", Value::number(2.0)),
    ] {
        state
            .heap_mut()
            .set_datum_field(parsed, field(name), value)
            .unwrap();
    }
    state
        .heap_mut()
        .set_datum_field(template, field("cached_map"), Value::Datum(parsed))
        .unwrap();

    let result = super::discover_offset_native(
        template,
        &Value::TypePath(TypePath::parse("/obj/modular_map_connector").unwrap()),
        &mut state,
    )
    .expect("canonical heap shape should use native scan");
    let Value::List(result) = result else {
        panic!("marker should produce an offset")
    };
    assert_eq!(
        state.heap().list(result).unwrap().get(1),
        Ok(&Value::number(11.0))
    );
    assert_eq!(
        state.heap().list(result).unwrap().get(2),
        Ok(&Value::number(19.0))
    );
}

#[test]
fn canonical_preload_size_uses_artifact_measurement_and_updates_dimensions() {
    let syntax = parse(
            "/datum/map_template/proc/preload_size(path, cache = FALSE)\n\tvar/parsed\n\tvar/bounds\n\treturn bounds\n/proc/run(datum/map_template/template)\n\treturn template.preload_size(\"_maps/test.dmm\", FALSE)\n",
        )
        .unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let mut state = ExecutionState::new();
    state.set_dmm_measurements(Arc::new(BTreeMap::from([(
        "_maps/test.dmm".to_owned(),
        super::DmmMeasurement {
            digest: md5::compute(b"fixture map").0,
            bounds: [3, 5, 2, 8, 11, 2],
        },
    )])));
    let template = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/map_template").unwrap());
    let result = execute_module_in_state(
        &module,
        module.procedure_id("/proc/run").unwrap(),
        &[Value::Datum(template)],
        &mut state,
    )
    .unwrap();
    let Value::List(bounds) = result else {
        panic!("preload_size should return cached bounds");
    };
    let values = state
        .heap()
        .list(bounds)
        .unwrap()
        .positions()
        .map(|(_, value)| value.as_number().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(values, vec![3.0, 5.0, 2.0, 8.0, 11.0, 2.0]);
    assert_eq!(
        state
            .heap()
            .datum_field(template, &FieldName::parse("width").unwrap())
            .unwrap()
            .as_number(),
        Some(8.0)
    );
    assert_eq!(
        state
            .heap()
            .datum_field(template, &FieldName::parse("height").unwrap())
            .unwrap()
            .as_number(),
        Some(11.0)
    );
}

#[test]
fn parsed_dmm_new_materializes_artifact_fields_and_changed_source_falls_back() {
    let syntax = parse(
            concat!(
                "/datum/parsed_map/New(tfile, x_lower, x_upper, y_lower, y_upper, z_lower, z_upper, measureOnly = FALSE)\n\treturn 7\n",
                "/proc/start(datum/parsed_map/parsed)\n\tspawn(0)\n\t\tworker(parsed)\n",
                "/proc/worker(datum/parsed_map/parsed)\n\tparsed.New(file(\"_maps/test.dmm\"))\n\tsleep(1)\n\treturn parsed\n",
            ),
        )
        .unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let procedure = module
        .procedure_id("/datum/parsed_map/New")
        .or_else(|| module.procedure_id("/datum/parsed_map/proc/New"))
        .unwrap();
    let digest = md5::compute(b"cached source").0;
    let parsed = super::ParsedDmm {
        digest,
        tgm: true,
        key_len: 1,
        line_len: 2,
        bounds: [2, 3, 4, 3, 4, 4],
        models: vec![("a".to_owned(), "/turf".to_owned())],
        grids: vec![super::ParsedDmmGrid {
            x: 2,
            y: 3,
            z: 4,
            lines: vec!["aa".to_owned(), "aa".to_owned()],
        }],
    };
    let mut state = ExecutionState::new();
    state.set_parsed_dmm_cache(Arc::new(BTreeMap::from([(
        "_maps/test.dmm".to_owned(),
        parsed,
    )])));
    let datum = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/parsed_map").unwrap());
    let context = ExecutionContext::new(Value::Datum(datum), Value::Null);
    let result = execute_module_in_context(
        &module,
        procedure,
        &[Value::File(Arc::from("_maps/test.dmm"))],
        &mut state,
        &context,
    )
    .unwrap();
    assert_eq!(result, Value::Null);
    let get = |state: &ExecutionState, name: &str| {
        state
            .heap()
            .datum_field(datum, &FieldName::parse(name).unwrap())
            .unwrap()
            .clone()
    };
    assert_eq!(get(&state, "map_format"), Value::text("tgm"));
    assert_eq!(get(&state, "key_len"), Value::number(1.0));
    assert_eq!(get(&state, "line_len"), Value::number(2.0));
    let Value::List(bounds) = get(&state, "bounds") else {
        panic!("bounds should be a list")
    };
    let Value::List(parsed_bounds) = get(&state, "parsed_bounds") else {
        panic!("parsed_bounds should be a list")
    };
    assert_ne!(bounds, parsed_bounds);
    let Value::List(grid_sets) = get(&state, "gridSets") else {
        panic!("gridSets should be a list")
    };
    let Value::Datum(grid) = state.heap().list(grid_sets).unwrap().get(1).unwrap() else {
        panic!("grid set should be a datum")
    };
    assert_eq!(
        state.heap().datum(*grid).unwrap().type_path().as_str(),
        "/datum/grid_set"
    );

    let yielded = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/parsed_map").unwrap());
    execute_module_in_state(
        &module,
        module.procedure_id("/proc/start").unwrap(),
        &[Value::Datum(yielded)],
        &mut state,
    )
    .unwrap();
    assert!(
        advance_scheduler(&module, 0, ExecutionLimits::default(), &mut state)
            .unwrap()
            .is_empty()
    );
    assert_eq!(state.scheduled_task_count(), 1);
    assert_eq!(
        advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state)
            .unwrap()
            .len(),
        1
    );
    assert!(matches!(
        state
            .heap()
            .datum_field(yielded, &FieldName::parse("gridSets").unwrap())
            .unwrap(),
        Value::List(_)
    ));

    let root = std::env::temp_dir().join(format!(
        "dream64-parsed-dmm-fallback-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(root.join("_maps")).unwrap();
    std::fs::write(root.join("_maps/test.dmm"), "changed source").unwrap();
    let mut fallback = ExecutionState::new();
    fallback.set_project_root(root.clone());
    fallback.set_parsed_dmm_cache(state.parsed_dmm_cache());
    let fallback_datum = fallback
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/parsed_map").unwrap());
    let result = execute_module_in_context(
        &module,
        procedure,
        &[Value::File(Arc::from("_maps/test.dmm"))],
        &mut fallback,
        &ExecutionContext::new(Value::Datum(fallback_datum), Value::Null),
    )
    .unwrap();
    assert_eq!(result, Value::number(7.0));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn tgm_tick_sidecar_resumes_through_tick_body_and_intercepts_before_rich_loop_jump() {
    let syntax = parse("/proc/run()\n\treturn\n").unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let procedure = module.procedure_id("/proc/run").unwrap();
    let program = module.procedure(procedure).unwrap();
    let event = crate::tgm_planner::CommitEvent::SafepointOnly(crate::tgm_planner::Ordinal {
        grid: 0,
        line: 0,
    });
    for resumed_pc in 423..=445 {
        let mut plan = crate::tgm_planner::Plan::default();
        plan.events.push(event.clone());
        let mut frame = make_frame(procedure, program, &[], &ExecutionContext::default());
        frame.instruction = resumed_pc;
        frame.cold_mut().tgm_load = Some(crate::TgmLoadContinuation {
            plan: Arc::new(plan),
            cursor: crate::tgm_planner::CommitCursor::default(),
            phase: crate::TgmLoadPhase::Tick,
            model_cache: Value::Null,
            models: BTreeMap::new(),
            bounds: Value::Null,
            coordinate_target: None,
        });
        let mut state = ExecutionState::new();
        assert!(matches!(
            crate::drive_tgm_load(&module, procedure, program, &mut frame, &mut state, 1),
            crate::TgmDrive::None
        ));
        frame.instruction = 446;
        assert!(matches!(
            crate::drive_tgm_load(&module, procedure, program, &mut frame, &mut state, 1),
            crate::TgmDrive::Continue
        ));
        let sidecar = frame.cold().unwrap().tgm_load.as_ref().unwrap();
        assert!(sidecar.cursor.is_complete(&sidecar.plan));
        assert_eq!(frame.instruction, 279);
    }
}

#[test]
fn tgm_rejection_diagnostics_ignore_unrelated_procedure_paths() {
    let syntax =
        parse("/proc/unrelated()\n\treturn\n/datum/parsed_map/proc/_tgm_load()\n\treturn\n")
            .unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let unrelated = module.procedure_id("/proc/unrelated").unwrap();
    let canonical = module
        .procedure_id("/datum/parsed_map/proc/_tgm_load")
        .unwrap();
    assert!(!crate::canonical_tgm_load_path(&module, unrelated));
    assert!(crate::canonical_tgm_load_path(&module, canonical));
}

#[test]
fn tgm_attach_accepts_pre_iterator_and_legacy_seams_only() {
    let syntax = parse("/proc/run()\n\treturn\n").unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let procedure = module.procedure_id("/proc/run").unwrap();
    let program = module.procedure(procedure).unwrap();
    let mut frame = make_frame(procedure, program, &[], &ExecutionContext::default());
    frame.instruction = 274;
    assert_eq!(crate::tgm_attach_location(&frame), Some(true));
    frame.stack.push(Value::number(1.0));
    assert_eq!(crate::tgm_attach_location(&frame), None);
    frame.stack.clear();
    frame.instruction = 279;
    assert_eq!(crate::tgm_attach_location(&frame), Some(false));
    frame.instruction = 275;
    assert_eq!(crate::tgm_attach_location(&frame), None);
}

#[test]
fn tgm_build_cache_simple_member_matches_unedited_effects_and_falls_back_cleanly() {
    let syntax = parse("/proc/run()\n\treturn\n").unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let procedure = module.procedure_id("/proc/run").unwrap();
    let program = module.procedure(procedure).unwrap();
    let mut state = ExecutionState::new();
    let paths =
        ["/area/test", "/turf/test", "/obj/test"].map(|path| TypePath::parse(path).unwrap());
    state.set_type_paths(
        [TypePath::parse("/atom").unwrap()]
            .into_iter()
            .chain(paths.iter().cloned()),
    );
    state.set_type_parents(BTreeMap::from([
        (paths[0].clone(), Some(TypePath::parse("/area").unwrap())),
        (paths[1].clone(), Some(TypePath::parse("/turf").unwrap())),
        (paths[2].clone(), Some(TypePath::parse("/obj").unwrap())),
    ]));
    let default = state.heap_mut().allocate_list();
    let wrapped = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(wrapped)
        .unwrap()
        .add(Value::List(default));
    let members = state.heap_mut().allocate_list();
    let attributes = state.heap_mut().allocate_list();
    let mut frame = make_frame(procedure, program, &[], &ExecutionContext::default());
    frame.locals.resize(24, Value::Null);
    frame.instruction = 98;
    frame.locals[5] = Value::List(default);
    frame.locals[6] = Value::List(wrapped);
    frame.locals[10] = Value::number(0.0);
    frame.locals[15] = Value::List(members);
    frame.locals[16] = Value::List(attributes);
    // A real TGM space model is newline-split before this loop. Each
    // invocation must consume exactly one line and return to PC260 so the
    // rich iterator can advance to the next member.
    for (index, path) in paths.iter().enumerate() {
        frame.instruction = 98;
        frame.locals[17] = Value::text(if index + 1 == paths.len() {
            path.as_str().to_owned()
        } else {
            format!("{},", path.as_str())
        });
        assert_eq!(
            crate::run_tgm_build_cache_simple_member(&mut frame, &mut state),
            Some(1)
        );
        assert_eq!(frame.instruction, 260);
    }
    assert_eq!(frame.locals[8], Value::text("/obj/test"));
    assert_eq!(frame.locals[20], Value::text("t"));
    assert_eq!(frame.locals[23], Value::TypePath(paths[2].clone()));
    assert_eq!(
        state
            .heap()
            .list(members)
            .unwrap()
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>(),
        paths
            .iter()
            .cloned()
            .map(Value::TypePath)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        state
            .heap()
            .list(attributes)
            .unwrap()
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>(),
        vec![Value::List(default); 3]
    );

    for rejected in [
        "name=\"edited\";",
        "/area/test,/missing/path",
        "/turf/test{",
        "prefix/area/test,/obj/test",
    ] {
        let rejected_members = state.heap_mut().allocate_list();
        let rejected_attributes = state.heap_mut().allocate_list();
        frame.instruction = 98;
        frame.locals[10] = Value::number(f32::from(rejected.contains(';')));
        frame.locals[15] = Value::List(rejected_members);
        frame.locals[16] = Value::List(rejected_attributes);
        frame.locals[17] = Value::text(rejected);
        assert_eq!(
            crate::run_tgm_build_cache_simple_member(&mut frame, &mut state),
            None
        );
        assert_eq!(frame.instruction, 98);
        assert_eq!(state.heap().list(rejected_members).unwrap().len(), 0);
        assert_eq!(state.heap().list(rejected_attributes).unwrap().len(), 0);
    }
}

#[test]
#[ignore = "focused debug microbenchmark; run explicitly when changing the TGM cache tier"]
fn tgm_build_cache_simple_member_debug_benchmark() {
    const MEMBERS: usize = 10_000;
    let syntax = parse("/proc/run(n)\n\tvar/list/members = list()\n\tvar/list/attrs = list()\n\tvar/list/default = list()\n\tvar/list/wrapped = list(default)\n\tfor(var/i in 1 to n)\n\t\tattrs += wrapped\n\t\tmembers += /turf/test\n\treturn members.len\n").unwrap();
    let rich_module = compile_module(&syntax.definitions).unwrap();
    let rich_started = std::time::Instant::now();
    let rich_result = execute_module(
        &rich_module,
        rich_module.procedure_id("/proc/run").unwrap(),
        &[Value::number(MEMBERS as f32)],
    )
    .unwrap();
    let rich_elapsed = rich_started.elapsed();
    assert_eq!(rich_result, Value::number(MEMBERS as f32));

    let procedure = rich_module.procedure_id("/proc/run").unwrap();
    let program = rich_module.procedure(procedure).unwrap();
    let mut state = ExecutionState::new();
    let path = TypePath::parse("/turf/test").unwrap();
    state.set_type_paths([TypePath::parse("/atom").unwrap(), path.clone()]);
    state.set_type_parents(BTreeMap::from([(
        path,
        Some(TypePath::parse("/turf").unwrap()),
    )]));
    let default = state.heap_mut().allocate_list();
    let wrapped = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(wrapped)
        .unwrap()
        .add(Value::List(default));
    let members = state.heap_mut().allocate_list();
    let attributes = state.heap_mut().allocate_list();
    let mut frame = make_frame(procedure, program, &[], &ExecutionContext::default());
    frame.locals.resize(24, Value::Null);
    frame.locals[5] = Value::List(default);
    frame.locals[6] = Value::List(wrapped);
    frame.locals[10] = Value::number(0.0);
    frame.locals[15] = Value::List(members);
    frame.locals[16] = Value::List(attributes);
    frame.locals[17] = Value::text("/turf/test,");
    let native_started = std::time::Instant::now();
    for _ in 0..MEMBERS {
        frame.instruction = 98;
        assert_eq!(
            crate::run_tgm_build_cache_simple_member(&mut frame, &mut state),
            Some(1)
        );
    }
    let native_elapsed = native_started.elapsed();
    assert_eq!(state.heap().list(members).unwrap().len(), MEMBERS);
    assert_eq!(state.heap().list(attributes).unwrap().len(), MEMBERS);
    eprintln!(
        "tgm-build-cache plain-member rich_ms={} native_ms={} speedup={:.2}x",
        rich_elapsed.as_millis(),
        native_elapsed.as_millis(),
        rich_elapsed.as_secs_f64() / native_elapsed.as_secs_f64()
    );
}

#[test]
fn tgm_sidecar_reads_canonical_compiler_local_slots() {
    let syntax = parse("/proc/run()\n\treturn\n").unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let procedure = module.procedure_id("/proc/run").unwrap();
    let program = module.procedure(procedure).unwrap();
    let mut frame = make_frame(procedure, program, &[], &ExecutionContext::default());
    frame.locals.resize(41, Value::Null);
    for slot in 0..=2 {
        frame.locals[slot] = Value::number(1.0);
    }
    frame.locals[2] = Value::number(5.0);
    for slot in 3..=4 {
        frame.locals[slot] = Value::number(0.0);
    }
    for slot in 5..=8 {
        frame.locals[slot] = Value::number(if slot % 2 == 0 { 10.0 } else { 1.0 });
    }
    // tg/Monke's INFINITY macro is the finite value 1e31.
    frame.locals[9] = Value::number(-1e31_f32);
    frame.locals[10] = Value::number(1e31_f32);

    let mut state = ExecutionState::new();
    let world = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/world").unwrap());
    for name in ["maxx", "maxy"] {
        state
            .heap_mut()
            .set_datum_field(world, FieldName::parse(name).unwrap(), Value::number(10.0))
            .unwrap();
    }
    state.set_global(field("world"), Value::Datum(world));
    let model_cache = state.heap_mut().allocate_list();
    let _ = state
        .heap_mut()
        .list_mut(model_cache)
        .unwrap()
        .set_key(Value::text("a"), Value::number(1.0));
    let lines = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(lines)
        .unwrap()
        .add(Value::text("a"));
    let grid = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/grid_set").unwrap());
    for (name, value) in [("xcrd", 1.0), ("ycrd", 1.0), ("zcrd", 1.0)] {
        state
            .heap_mut()
            .set_datum_field(grid, FieldName::parse(name).unwrap(), Value::number(value))
            .unwrap();
    }
    state
        .heap_mut()
        .set_datum_field(
            grid,
            FieldName::parse("gridLines").unwrap(),
            Value::List(lines),
        )
        .unwrap();
    let grid_sets = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(grid_sets)
        .unwrap()
        .add(Value::Datum(grid));
    let bounds = state.heap_mut().allocate_list();

    // The compiler reserves slot 13 for `.`, so source locals start at 14.
    frame.locals[14] = Value::List(model_cache);
    // Lavaland has no model matching world.area + world.turf, so
    // build_cache leaves SPACE_KEY null. This must mean "skip nothing".
    frame.locals[15] = Value::Null;
    frame.locals[16] = Value::List(bounds);
    frame.locals[38] = Value::List(grid_sets);
    frame.locals[39] = Value::number(1.0);

    let continuation = crate::build_tgm_load_continuation(&frame, &state)
        .expect("canonical executable slots should install the TGM sidecar");
    assert_eq!(continuation.model_cache, Value::List(model_cache));
    assert_eq!(continuation.bounds, Value::List(bounds));
    assert!(continuation.plan.missing_models.is_empty());
    assert!(matches!(
        continuation.plan.events.as_slice(),
        [crate::tgm_planner::CommitEvent::Cell(cell)] if (cell.x, cell.y, cell.z) == (1, 1, 5)
    ));
}

#[test]
fn tgm_area_then_turf_model_rehomes_the_indexed_cell() {
    let mut state = ExecutionState::new();
    let space = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/area/space").unwrap());
    let lava = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/area/lavaland/surface/outdoors/unexplored").unwrap());
    let turf = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/turf/open/space/basic").unwrap());
    for (name, value) in [("x", 4.0), ("y", 7.0), ("z", 5.0)] {
        state
            .heap_mut()
            .set_datum_field(turf, field(name), Value::number(value))
            .unwrap();
    }
    state
        .heap_mut()
        .set_datum_field(turf, field("loc"), Value::Datum(space))
        .unwrap();
    state.world_turfs.insert((4, 7, 5), turf);
    state.world_areas.insert((4, 7, 5), space);
    let space_contents = state.ensure_contents(space).unwrap();
    state
        .heap_mut()
        .list_mut(space_contents)
        .unwrap()
        .add(Value::Datum(turf));

    // This is the observable engine portion of build_coordinate's model
    // order: area.contents.Add(crds), followed by new turf_type(crds).
    let lava_contents = state.ensure_contents(lava).unwrap();
    super::builtins::execute_list_method("Add", lava_contents, &[Value::Datum(turf)], &mut state)
        .unwrap()
        .unwrap();
    let replaced = super::allocate_or_replace_engine_datum(
        &mut state,
        TypePath::parse("/turf/open/floor/plating/asteroid/basalt/lava_land_surface").unwrap(),
        &[Value::Datum(turf)],
    )
    .unwrap();

    assert_eq!(replaced, turf, "map turf replacement preserves identity");
    assert_eq!(state.turf_at(4, 7, 5), Some(turf));
    assert_eq!(state.world_areas.get(&(4, 7, 5)), Some(&lava));
    assert_eq!(
        state.heap().datum_field(turf, &field("loc")),
        Ok(&Value::Datum(lava))
    );
    assert!(
        state
            .heap()
            .list(lava_contents)
            .unwrap()
            .contains(&Value::Datum(turf))
    );
    assert!(
        !state
            .heap()
            .list(space_contents)
            .unwrap()
            .contains(&Value::Datum(turf))
    );
}

#[test]
fn build_coordinate_prefix_matches_rich_area_add_and_preserves_turf_seam() {
    let syntax = parse("/proc/run(a,b,c,d,e)\n\treturn\n").unwrap();
    let mut module = compile_module(&syntax.definitions).unwrap();
    let procedure = module.procedure_id("/proc/run").unwrap();
    let program = Program {
        wait_for: true,
        parameter_count: 5,
        parameter_names: vec![String::new(); 5],
        verb_parameter_types: vec![VerbParameterType::Unsupported; 5],
        verb_name: None,
        local_count: 31,
        instructions: vec![Instruction::PushNull; 405],
        source_spans: vec![SourceSpan::new(0, 0); 405],
    };
    module.procedures[procedure.index()] = Arc::new(program);
    module.paths[procedure.index()] = "/datum/parsed_map/proc/build_coordinate@fixture".to_owned();
    module.semantic_digests =
        crate::bytecode::ProcedureSemanticDigestAttachment(Some(Arc::from(vec![
            crate::CANONICAL_MONKE_BUILD_COORDINATE_DIGEST,
        ])));
    let program = module.procedure(procedure).unwrap();

    let mut state = ExecutionState::new();
    let old_area = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/area/space").unwrap());
    let new_area = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/area/lavaland/surface/outdoors/unexplored").unwrap());
    let turf = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/turf/open/space/basic").unwrap());
    state
        .heap_mut()
        .set_datum_field(turf, field("loc"), Value::Datum(old_area))
        .unwrap();
    let old_contents = state.ensure_contents(old_area).unwrap();
    state
        .heap_mut()
        .list_mut(old_contents)
        .unwrap()
        .add(Value::Datum(turf));
    let default_list = state.heap_mut().allocate_list();
    let members = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(members)
        .unwrap()
        .extend_positional([
            Value::TypePath(TypePath::parse("/turf/open/ashplanet/basalt").unwrap()),
            Value::TypePath(TypePath::parse("/area/lavaland/surface/outdoors/unexplored").unwrap()),
        ]);
    let attributes = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(attributes)
        .unwrap()
        .extend_positional([Value::List(default_list), Value::List(default_list)]);
    let model = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(model)
        .unwrap()
        .extend_positional([Value::List(members), Value::List(attributes)]);
    let loaded = state.heap_mut().allocate_list();
    let _ = state.heap_mut().list_mut(loaded).unwrap().set_key(
        Value::TypePath(TypePath::parse("/area/lavaland/surface/outdoors/unexplored").unwrap()),
        Value::Datum(new_area),
    );
    let src = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/parsed_map").unwrap());
    state
        .heap_mut()
        .set_datum_field(src, field("loaded_areas"), Value::List(loaded))
        .unwrap();
    state
        .heap_mut()
        .set_datum_field(src, field("turf_blacklist"), Value::Null)
        .unwrap();
    state.set_global(
            FieldName::parse("__dm_static_2f646174756d2f636f6e74726f6c6c65722f676c6f62616c5f766172732f7661722f6d61705f6d6f64656c5f64656661756c74").unwrap(),
            Value::List(default_list),
        );
    state.set_global(
            FieldName::parse("__dm_static_2f646174756d2f636f6e74726f6c6c65722f676c6f62616c5f766172732f7661722f7573655f7072656c6f61646572").unwrap(),
            Value::number(0.0),
        );
    let context = ExecutionContext::new(Value::Datum(src), Value::Null);
    let mut frame = make_frame(
        procedure,
        program,
        &[
            Value::List(model),
            Value::Datum(turf),
            Value::number(1.0),
            Value::number(0.0),
            Value::number(1.0),
        ],
        &context,
    );
    assert!(crate::try_run_build_coordinate_prefix(
        &module, procedure, program, &mut frame, &mut state
    ));
    assert_eq!(frame.instruction, 235, "ordinary turf/New seam is retained");
    assert_eq!(frame.locals[6], Value::number(1.0));
    assert_eq!(
        state.heap().datum_field(turf, &field("loc")),
        Ok(&Value::Datum(new_area))
    );
    let new_contents = state.ensure_contents(new_area).unwrap();
    assert!(
        state
            .heap()
            .list(new_contents)
            .unwrap()
            .contains(&Value::Datum(turf))
    );

    frame.instruction = 0;
    frame.locals[4] = Value::number(0.0);
    assert!(!crate::try_run_build_coordinate_prefix(
        &module, procedure, program, &mut frame, &mut state
    ));
    assert_eq!(frame.instruction, 0, "new_z=false preserves rich fallback");
}

#[test]
fn ruin_affected_turfs_batch_is_atomic_at_budget_boundaries_and_stops_on_first_reject() {
    let program = Program {
        wait_for: true,
        parameter_count: 4,
        parameter_names: vec![String::new(); 4],
        verb_parameter_types: vec![VerbParameterType::Unsupported; 4],
        verb_name: None,
        local_count: 29,
        instructions: vec![Instruction::Return],
        source_spans: vec![SourceSpan::new(0, 0)],
    };
    let make_state = || {
        let mut state = ExecutionState::new();
        let area = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/area/test").unwrap());
        let normal = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/turf").unwrap());
        let rejected = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/turf").unwrap());
        for turf in [normal, rejected] {
            state
                .heap_mut()
                .set_datum_field(turf, field("loc"), Value::Datum(area))
                .unwrap();
        }
        for (turf, x) in [(normal, 1.0), (rejected, 2.0)] {
            for (name, value) in [("x", x), ("y", 1.0), ("z", 1.0)] {
                state
                    .heap_mut()
                    .set_datum_field(turf, field(name), Value::number(value))
                    .unwrap();
            }
        }
        state.world_turfs.insert((1, 1, 1), normal);
        state.world_turfs.insert((2, 1, 1), rejected);
        state.rebuild_world_turf_lookup();
        state
            .heap_mut()
            .set_datum_field(normal, field("turf_flags"), Value::number(0.0))
            .unwrap();
        state
            .heap_mut()
            .set_datum_field(rejected, field("turf_flags"), Value::number(16.0))
            .unwrap();
        let affected = state.heap_mut().allocate_list();
        state
            .heap_mut()
            .list_mut(affected)
            .unwrap()
            .add(Value::Datum(normal));
        state
            .heap_mut()
            .list_mut(affected)
            .unwrap()
            .add(Value::Datum(rejected));
        (state, area, affected)
    };

    for budget in 0..41 {
        let (mut state, _, affected) = make_state();
        let areas = state.heap_mut().allocate_list();
        let mut frame = make_frame(ProcedureId(0), &program, &[], &ExecutionContext::default());
        frame.instruction = 74;
        frame.locals[9] = Value::number(1.0);
        frame.locals[11] = Value::List(areas);
        frame.locals[13] = Value::List(affected);
        frame.locals[14] = Value::number(1.0);
        assert_eq!(
            crate::run_ruin_affected_turfs_batch(&mut frame, budget, &mut state),
            None
        );
        assert_eq!(frame.locals[14], Value::number(1.0));
        assert_eq!(state.heap().list(areas).unwrap().len(), 0);
    }

    let (mut state, area, affected) = make_state();
    let areas = state.heap_mut().allocate_list();
    let mut frame = make_frame(ProcedureId(0), &program, &[], &ExecutionContext::default());
    frame.instruction = 74;
    frame.locals[9] = Value::number(1.0);
    frame.locals[11] = Value::List(areas);
    frame.locals[13] = Value::List(affected);
    frame.locals[14] = Value::number(1.0);
    assert_eq!(
        crate::run_ruin_affected_turfs_batch(&mut frame, 61, &mut state),
        Some(61)
    );
    assert_eq!(frame.instruction, 116);
    assert_eq!(frame.locals[9], Value::number(0.0));
    assert_eq!(
        state
            .heap()
            .list(areas)
            .unwrap()
            .get_key(&Value::Datum(area))
            .unwrap(),
        &Value::number(1.0)
    );
}

#[test]
fn ruin_candidate_scan_accepts_compact_dispatch_call_side_exit() {
    let syntax = parse("/proc/run()\n\treturn\n").unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let procedure = module.procedure_id("/proc/run").unwrap();
    let program = module.procedure(procedure).unwrap();
    let mut frame = make_frame(procedure, program, &[], &ExecutionContext::default());
    frame.locals.resize(29, Value::Null);
    frame.locals[8] = Value::number(42.0);

    frame.instruction = 63;
    assert_eq!(crate::ruin_scan_attach_at_call(&frame), Some(false));

    frame.instruction = 65;
    frame.stack = vec![Value::number(42.0), Value::number(1.0)].into();
    assert_eq!(crate::ruin_scan_attach_at_call(&frame), Some(true));

    frame.stack[1] = Value::number(0.0);
    assert_eq!(crate::ruin_scan_attach_at_call(&frame), None);
    frame.stack = vec![Value::number(42.0), Value::number(1.0), Value::Null].into();
    assert_eq!(crate::ruin_scan_attach_at_call(&frame), None);
}

#[test]
fn ruin_rejection_witness_is_revalidated_and_never_caches_success() {
    let mut state = ExecutionState::new();
    let turf = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/turf/test").unwrap());
    state
        .heap_mut()
        .set_datum_field(turf, field("turf_flags"), Value::number(16.0))
        .unwrap();
    state.world_turfs.insert((2, 3, 1), turf);
    state.rebuild_world_turf_lookup();
    let bounds = (1, 1, 1, 5, 5, 1);
    state
        .ruin_rejection_witnesses
        .entry(1)
        .or_default()
        .insert((3, 2), turf);
    let turf_flags = field("turf_flags");
    assert!(crate::revalidated_ruin_rejection(
        &mut state,
        bounds,
        &turf_flags
    ));
    assert!(crate::revalidated_ruin_rejection(
        &mut state,
        (2, 2, 1, 2, 3, 1),
        &turf_flags
    ));

    state
        .heap_mut()
        .set_datum_field(turf, turf_flags.clone(), Value::number(0.0))
        .unwrap();
    assert!(!crate::revalidated_ruin_rejection(
        &mut state,
        bounds,
        &turf_flags
    ));
    assert!(
        state
            .ruin_rejection_witnesses
            .get(&1)
            .is_none_or(BTreeMap::is_empty)
    );
    assert!(!crate::revalidated_ruin_rejection(
        &mut state,
        bounds,
        &turf_flags
    ));
}

#[test]
fn ruin_candidate_scan_preserves_first_rejection_and_success_materialization_order() {
    let syntax = parse("/proc/run()\n\treturn\n").unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let procedure = module.procedure_id("/proc/run").unwrap();
    let program = module.procedure(procedure).unwrap();
    for rejected in [Some(0), Some(2), Some(4), None] {
        let mut state = ExecutionState::new();
        let area = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/area/test").unwrap());
        let allowed = state.heap_mut().allocate_list();
        crate::write_list_value(
            &mut state.heap,
            allowed,
            Value::TypePath(TypePath::parse("/area/test").unwrap()),
            Value::number(1.0),
            false,
        )
        .unwrap();
        for x in 1..=5 {
            let turf = state
                .heap_mut()
                .allocate_datum(TypePath::parse("/turf").unwrap());
            for (name, value) in [("x", x as f32), ("y", 1.0), ("z", 1.0)] {
                state
                    .heap_mut()
                    .set_datum_field(turf, field(name), Value::number(value))
                    .unwrap();
            }
            state
                .heap_mut()
                .set_datum_field(turf, field("loc"), Value::Datum(area))
                .unwrap();
            state
                .heap_mut()
                .set_datum_field(
                    turf,
                    field("turf_flags"),
                    Value::number(if rejected == Some(x - 1) { 16.0 } else { 0.0 }),
                )
                .unwrap();
            state.world_turfs.insert((x, 1, 1), turf);
        }
        state.rebuild_world_turf_lookup();
        let mut frame = make_frame(procedure, program, &[], &ExecutionContext::default());
        frame.instruction = 63;
        frame.locals.resize(29, Value::Null);
        frame.locals[1] = Value::List(allowed);
        frame.locals[9] = Value::number(1.0);
        frame.cold_mut().ruin_scan = Some(crate::RuinCandidateScan {
            low: (1, 1, 1),
            next: (1, 1, 1),
            high: (5, 1, 1),
            empty: false,
            turfs: Vec::new(),
            areas: Vec::new(),
            validating: false,
            validate_index: 0,
        });
        assert!(matches!(
            crate::drive_ruin_candidate_scan(
                &module, procedure, program, &mut frame, &mut state, 1
            ),
            crate::TgmDrive::Continue
        ));
        if rejected.is_some() {
            assert_eq!(frame.instruction, 14);
            assert_eq!(frame.locals[9], Value::number(0.0));
            assert!(matches!(frame.locals[10], Value::Null));
        } else {
            assert_eq!(frame.instruction, 145);
            let Value::List(turfs) = frame.locals[10] else {
                panic!("successful scan must materialize affected turfs")
            };
            assert_eq!(state.heap().list(turfs).unwrap().len(), 5);
            let Value::List(areas) = frame.locals[11] else {
                panic!("successful scan must materialize affected areas")
            };
            assert_eq!(state.heap().list(areas).unwrap().len(), 1);
        }
    }
}

#[test]
fn splittext_regex_delimiters_preserve_bounds_and_optional_matches() {
    let source = parse(concat!(
        "/proc/run()\n",
        "\tvar/regex/delimiter = regex(@\"[,:]\", \"g\")\n",
        "\tvar/list/plain = splittext(\"a,b:c\", delimiter)\n",
        "\tvar/list/included = splittext(\"a,b:c\", delimiter, 1, 0, TRUE)\n",
        "\treturn list(plain, included)\n",
    ))
    .expect("regex split source should parse");
    let module = compile_module(&source.definitions).expect("regex split should compile");
    let mut state = ExecutionState::new();
    let result = execute_module_in_state(
        &module,
        module.procedure_id("/proc/run").expect("run exists"),
        &[],
        &mut state,
    )
    .expect("regex split should execute");
    let Value::List(result) = result else {
        panic!("regex split should return both lists");
    };
    let lists = state
        .heap()
        .list(result)
        .unwrap()
        .positions()
        .map(|(_, value)| value.clone())
        .collect::<Vec<_>>();
    let values = |value: &Value| {
        let Value::List(list) = value else {
            panic!("split result should be a list");
        };
        state
            .heap()
            .list(*list)
            .unwrap()
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        values(&lists[0]),
        vec![Value::text("a"), Value::text("b"), Value::text("c")]
    );
    assert_eq!(
        values(&lists[1]),
        vec![
            Value::text("a"),
            Value::text(","),
            Value::text("b"),
            Value::text(":"),
            Value::text("c"),
        ]
    );
}

#[test]
fn replacetext_regex_invokes_proc_replacement_with_match_and_capture_groups() {
    let syntax = parse(
            "/proc/capitalize(t)\n\tif(t)\n\t\tvar/first = t[1]\n\t\treturn uppertext(first) + copytext(t, 1 + length(first))\n\treturn t\n/proc/run(input)\n\tvar/regex/word = new(@\"([A-z]+)\", \"g\")\n\treturn replacetext(input, word, /proc/capitalize)\n",
        )
        .expect("regex proc-replacement source should parse");
    let module =
        compile_module(&syntax.definitions).expect("regex proc-replacement source should compile");
    let entry = module.procedure_id("/proc/run").unwrap();
    assert_eq!(
        execute_module(&module, entry, &[Value::text("hello world")]),
        Ok(Value::text("Hello World"))
    );
}

#[test]
fn replacetext_regex_returns_null_for_null_haystack() {
    let syntax = parse(
            "/proc/run(input)\n\tvar/static/regex/css = new(@\"[^a-zA-Z0-9]\", \"g\")\n\treturn replacetext(input, css, \"\")\n",
        )
        .expect("regex replacetext fixture should parse");
    let module =
        compile_module(&syntax.definitions).expect("regex replacetext fixture should compile");
    let entry = module.procedure_id("/proc/run").unwrap();
    assert_eq!(
        execute_module(&module, entry, &[Value::Null]),
        Ok(Value::Null),
        "BYOND/OpenDream return null rather than erroring for a non-text regex haystack",
    );
}

#[test]
fn regex_replace_method_sanitizes_all_global_matches() {
    let syntax = parse(
            "/proc/sanitize(value)\n\tvar/regex/forbidden = new(@\"[^A-Za-z0-9._-]\", \"g\")\n\treturn forbidden.Replace(value, \"\")\n",
        )
        .expect("regex.Replace source should parse");
    let module = compile_module(&syntax.definitions).expect("regex.Replace should compile");
    let entry = module.procedure_id("/proc/sanitize").unwrap();
    assert_eq!(
        execute_module(&module, entry, &[Value::text("pirates: heavy?/gang")]),
        // OpenDream/BYOND advance at least one position after an empty
        // replacement, so directly adjacent matches are considered on
        // the next replacement call rather than the same sweep.
        Ok(Value::text("pirates heavy/gang"))
    );
}

#[test]
fn mutable_appearance_builtin_constructs_its_builtin_datum() {
    let syntax = parse("/proc/build()\n\treturn mutable_appearance('icons/test.dmi', \"state\")\n")
        .expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("the mutable_appearance constructor should compile");
    assert!(program.instructions.iter().any(|instruction| matches!(
        instruction,
        Instruction::MakeMutableAppearance { argument_count: 2 }
    )));

    let mut state = ExecutionState::new();
    let result = execute_in_state(&program, &[], &mut state)
        .expect("mutable_appearance constructor should execute");
    let Value::Datum(appearance) = result else {
        panic!("mutable_appearance constructor should return a datum");
    };
    assert_eq!(
        state
            .heap()
            .datum(appearance)
            .expect("mutable appearance datum should exist")
            .type_path()
            .to_string(),
        "/mutable_appearance"
    );
}

#[test]
fn project_mutable_appearance_helper_receives_human_preview_named_arguments() {
    let syntax = parse(
            "/proc/mutable_appearance(icon, icon_state = \"\", layer = 2, offset_spokesman, plane = -32767, alpha = 255, appearance_flags = 0, offset_const)\n\tvar/mutable_appearance/appearance = new()\n\tappearance.icon = icon\n\tappearance.icon_state = icon_state\n\tappearance.layer = layer\n\tappearance.plane = plane\n\tappearance.alpha = alpha\n\tappearance.appearance_flags = appearance_flags\n\treturn appearance\n/proc/build_preview_overlay()\n\treturn mutable_appearance('icons/mob/species/human/bodyparts.dmi', layer = -7, appearance_flags = 64)\n",
        )
        .expect("human-preview mutable appearance fixture should parse");
    let module = compile_module(&syntax.definitions)
        .expect("project mutable appearance helper should override the engine fallback");
    let entry = module
        .procedure_id("/proc/build_preview_overlay")
        .expect("preview fixture entry should exist");
    let mut state = ExecutionState::new();
    let result = execute_module_in_state(&module, entry, &[], &mut state)
        .expect("human-preview mutable appearance fixture should execute");
    let Value::Datum(appearance) = result else {
        panic!("project mutable appearance helper should return its datum");
    };
    let appearance = state
        .heap()
        .datum(appearance)
        .expect("returned mutable appearance should remain live");

    assert_eq!(
        appearance.field(&field("icon")),
        Ok(&Value::file("icons/mob/species/human/bodyparts.dmi"))
    );
    assert_eq!(
        appearance.field(&field("icon_state")),
        Ok(&Value::text("")),
        "the omitted icon_state must use the project helper's default"
    );
    assert_eq!(appearance.field(&field("layer")), Ok(&Value::number(-7.0)));
    assert_eq!(
        appearance.field(&field("plane")),
        Ok(&Value::number(-32767.0)),
        "later omitted defaults must still execute around named arguments"
    );
    assert_eq!(appearance.field(&field("alpha")), Ok(&Value::number(255.0)));
    assert_eq!(
        appearance.field(&field("appearance_flags")),
        Ok(&Value::number(64.0))
    );
}

#[test]
fn matrix_constructor_methods_and_equivalence_use_affine_components() {
    let syntax = parse(
            "/proc/run()\n\tvar/matrix/value = matrix(1, 2, 3, 4, 5, 6)\n\tvalue.Add(matrix(7, 8, 9, 10, 11, 12))\n\tvalue.Subtract(matrix(7, 8, 9, 10, 11, 12))\n\tvalue.Multiply(matrix(7, 8, 9, 10, 11, 12))\n\treturn value ~= matrix(39, 54, 78, 54, 75, 108)\n",
        )
        .expect("matrix source should parse");
    let module = compile_module(&syntax.definitions).expect("matrix source should compile");
    let entry = module.procedure_id("/proc/run").expect("entry");
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(1.0)));
}

#[test]
fn matrix_binary_scaling_matches_rune_spawn_transform() {
    let syntax = parse(
            "/proc/run()\n\tvar/matrix/original = matrix()\n\tvar/matrix/scaled = original * 2\n\tvar/matrix/restored = scaled / 2\n\treturn original ~= matrix(1, 0, 0, 0, 1, 0) && scaled ~= matrix(2, 0, 0, 0, 2, 0) && restored ~= original\n",
        )
        .expect("rune-spawn matrix source should parse");
    let module =
        compile_module(&syntax.definitions).expect("rune-spawn matrix source should compile");
    let entry = module.procedure_id("/proc/run").expect("entry");
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(1.0)));
}

#[test]
fn project_matrix_members_dispatch_before_native_fallback() {
    let syntax = parse(
            "/matrix/proc/get_x_shift()\n\treturn 23\n/proc/run()\n\tvar/matrix/value = matrix()\n\treturn value.get_x_shift()\n",
        )
        .expect("project matrix member source should parse");
    let module = compile_module(&syntax.definitions).expect("project matrix member should compile");
    let entry = module.procedure_id("/proc/run").expect("entry");
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(23.0)));
}

#[test]
fn matrix_null_constructs_identity_for_absent_atom_transform() {
    let syntax = parse(
            "/proc/run(value)\n\tvar/matrix/copied = matrix(value)\n\treturn copied ~= matrix(1, 0, 0, 0, 1, 0)\n",
        )
        .expect("matrix null source should parse");
    let module = compile_module(&syntax.definitions).expect("matrix null source should compile");
    let entry = module.procedure_id("/proc/run").expect("entry");
    assert_eq!(
        execute_module(&module, entry, &[Value::Null]),
        Ok(Value::number(1.0))
    );
}

#[test]
fn atom_transform_read_materializes_an_identity_matrix_for_native_methods() {
    let syntax = parse(
            "/proc/run(obj/item)\n\titem.transform.Scale(2, 2)\n\tvar/matrix/copy = item.transform\n\treturn copy ~= matrix(1, 0, 0, 0, 1, 0)\n/obj\n",
        )
        .expect("atom transform fixture should parse");
    let definitions = syntax
        .definitions
        .iter()
        .filter(|definition| matches!(definition.kind, DefinitionKind::Procedure))
        .cloned()
        .collect::<Vec<_>>();
    let module = compile_module(&definitions).expect("atom transform fixture should compile");
    let entry = module.procedure_id("/proc/run").unwrap();
    let mut state = ExecutionState::new();
    state.set_type_parents(
        [
            (TypePath::parse("/datum").unwrap(), None),
            (
                TypePath::parse("/atom").unwrap(),
                Some(TypePath::parse("/datum").unwrap()),
            ),
            (
                TypePath::parse("/obj").unwrap(),
                Some(TypePath::parse("/atom").unwrap()),
            ),
        ]
        .into(),
    );
    state.set_initial_values(BTreeMap::from([(
        TypePath::parse("/obj").unwrap(),
        BTreeMap::from([(field("transform"), Value::Null)]),
    )]));
    let item = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj").unwrap());
    state
        .heap_mut()
        .set_datum_field(item, field("transform"), Value::Null)
        .unwrap();
    assert_eq!(
        execute_module_with_limits_in_state(
            &module,
            entry,
            &[Value::Datum(item)],
            ExecutionLimits::default(),
            &mut state,
        ),
        Ok(Value::number(1.0)),
        "each appearance transform read returns an identity matrix copy, matching OpenDream",
    );
}

#[test]
fn matrix_transform_methods_mutate_the_six_public_fields() {
    let syntax = parse(
            "/proc/run()\n\tvar/matrix/value = matrix(1, 2, 3, 4, 5, 6)\n\tvalue.Translate(2)\n\tvalue.Turn(90)\n\treturn value ~= matrix(4, 5, 8, -1, -2, -5)\n",
        )
        .expect("matrix source should parse");
    let module = compile_module(&syntax.definitions).expect("matrix source should compile");
    let entry = module.procedure_id("/proc/run").expect("entry");
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(1.0)));
}

#[test]
fn matrix_scale_applies_independent_horizontal_and_vertical_factors() {
    let syntax = parse(
            "/proc/run()\n\tvar/matrix/value = matrix(1, 2, 3, 4, 5, 6)\n\tvalue.Scale(2, 3)\n\treturn value ~= matrix(2, 4, 6, 12, 15, 18)\n",
        )
        .expect("matrix scale source should parse");
    let module = compile_module(&syntax.definitions).expect("matrix scale source should compile");
    let entry = module.procedure_id("/proc/run").expect("entry");
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(1.0)));
}

#[test]
fn turn_builtin_rotates_a_matrix_copy_without_mutating_the_source() {
    let syntax = parse(
            "/proc/run()\n\tvar/matrix/original = matrix(1, 2, 3, 4, 5, 6)\n\tvar/matrix/rotated = turn(original, 90)\n\treturn original != rotated && original ~= matrix(1, 2, 3, 4, 5, 6) && rotated ~= matrix(4, 5, 6, -1, -2, -3)\n",
        )
        .expect("turn matrix source should parse");
    let module = compile_module(&syntax.definitions).expect("turn matrix source should compile");
    let entry = module.procedure_id("/proc/run").expect("entry");
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(1.0)));
}

#[test]
fn turn_builtin_clones_and_rotates_an_icon() {
    let syntax = parse(
            "/proc/run()\n\tvar/icon/original = icon('icons/test.dmi')\n\tvar/icon/rotated = turn(original, 90)\n\treturn isicon(rotated) && original != rotated && original.icon == rotated.icon\n",
        )
        .expect("turn icon source should parse");
    let module = compile_module(&syntax.definitions).expect("turn icon source should compile");
    let entry = module.procedure_id("/proc/run").expect("entry");
    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(1.0)));
}

#[test]
fn optional_field_and_index_operators_parse_as_access_expressions() {
    let fields = parse("/proc/probe(value)\n\treturn value?.name\n")
        .expect("optional field source should parse");
    compile_procedure(&fields.definitions[0]).expect("optional field access should compile");

    let index = parse("/proc/probe(value)\n\treturn value?[1]\n")
        .expect("optional index source should parse");
    compile_procedure(&index.definitions[0]).expect("optional index access should compile");
}

#[test]
fn headless_locate_in_container_is_not_list_membership_and_supports_nesting() {
    let syntax = parse("/proc/find()\n\treturn locate(locate(1, 2, 3) in null, 4, 5) in null\n")
        .expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("headless locate in a container should compile");

    assert_eq!(
        program
            .instructions
            .iter()
            .filter(|instruction| matches!(instruction, Instruction::LocateIn { .. }))
            .count(),
        2
    );
    assert_eq!(execute(&program, &[]), Ok(Value::Null));
}

#[test]
fn locate_type_in_list_returns_first_matching_runtime_datum() {
    let syntax = parse("/proc/find(target, items)\n\treturn locate(target) in items\n")
        .expect("locate-in source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("locate-in source should compile");
    let mut state = ExecutionState::new();
    let processor_type =
        TypePath::parse("/datum/controller/subsystem/processing").expect("processor type path");
    let processor = state.heap_mut().allocate_datum(processor_type.clone());
    let items = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(items)
        .unwrap()
        .add(Value::Datum(processor));

    assert_eq!(
        execute_in_state(
            &program,
            &[Value::TypePath(processor_type), Value::List(items)],
            &mut state,
        ),
        Ok(Value::Datum(processor))
    );
}

#[test]
fn two_argument_locate_searches_container_contents_like_mechpad() {
    let syntax = parse("/proc/find(target, turf/container)\n\treturn locate(target, container)\n")
        .expect("two-argument locate source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("two-argument locate source should compile");
    let mut state = ExecutionState::new();
    let container = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/turf/floor").unwrap());
    let contents = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .set_datum_field(container, field("contents"), Value::List(contents))
        .unwrap();
    let decoy = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/machinery/other").unwrap());
    let mechpad = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/machinery/mechpad/mining").unwrap());
    let contents_list = state.heap_mut().list_mut(contents).unwrap();
    contents_list.add(Value::Datum(decoy));
    contents_list.add(Value::Datum(mechpad));

    assert_eq!(
        execute_in_state(
            &program,
            &[
                Value::TypePath(TypePath::parse("/obj/machinery/mechpad").unwrap()),
                Value::Datum(container),
            ],
            &mut state,
        ),
        Ok(Value::Datum(mechpad)),
    );
}

#[test]
fn empty_locate_infers_typed_local_destination_for_contractor_preview() {
    let syntax = parse(concat!(
        "/proc/find_declared(list/held_items)\n",
        "\tvar/obj/item/melee/baton/telescopic/contractor_baton/baton = locate() in held_items\n",
        "\treturn baton.flags_1\n",
        "/proc/find_assigned(list/held_items)\n",
        "\tvar/obj/item/melee/baton/telescopic/contractor_baton/baton\n",
        "\tbaton = locate() in held_items\n",
        "\treturn baton.flags_1\n",
    ))
    .expect("contractor preview locate fixture should parse");
    let module = compile_module(&syntax.definitions)
        .expect("contextual empty locate should infer its typed destination");
    let mut state = ExecutionState::new();
    let baton_base = TypePath::parse("/obj/item/melee/baton/telescopic/contractor_baton").unwrap();
    let baton_type =
        TypePath::parse("/obj/item/melee/baton/telescopic/contractor_baton/loaded").unwrap();
    let decoy_type = TypePath::parse("/obj/item/decoy").unwrap();
    state.set_type_parents(BTreeMap::from([
        (baton_base.clone(), None),
        (baton_type.clone(), Some(baton_base)),
        (decoy_type.clone(), None),
    ]));
    let decoy = state.heap_mut().allocate_datum(decoy_type);
    state
        .heap_mut()
        .set_datum_field(decoy, field("flags_1"), Value::number(99.0))
        .unwrap();
    let baton = state.heap_mut().allocate_datum(baton_type);
    state
        .heap_mut()
        .set_datum_field(baton, field("flags_1"), Value::number(1.0))
        .unwrap();
    let held_items = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(held_items)
        .unwrap()
        .add(Value::Datum(decoy));
    state
        .heap_mut()
        .list_mut(held_items)
        .unwrap()
        .add(Value::Datum(baton));

    for path in ["/proc/find_declared", "/proc/find_assigned"] {
        assert_eq!(
            execute_module_in_state(
                &module,
                module.procedure_id(path).unwrap(),
                &[Value::List(held_items)],
                &mut state,
            ),
            Ok(Value::number(1.0)),
            "{path} must lower locate() as locate(the declared type)",
        );
    }
}

#[test]
fn length_builtin_counts_text_bytes_and_list_entries() {
    let source = "/proc/measure()\n\tvar/text_length = length(\"aé\")\n\tvar/list_length = length(list(10, 20, \"key\" = 30))\n\treturn text_length * 10 + list_length\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("length builtin should compile for text and lists");

    assert!(
        program
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::Length))
    );
    // DM's regular text operations use legacy byte positions, so `é`
    // contributes two UTF-8 bytes. Associative list entries contribute
    // one entry, like positional list values.
    assert_eq!(execute(&program, &[]), Ok(Value::number(33.0)));
}

#[test]
fn length_builtin_returns_zero_for_non_sequence_values() {
    let source =
        "/proc/measure()\n\treturn length(/turf/baseturf_bottom) + length(42) + length(null)\n";
    let syntax = parse(source).expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("length probes should compile");

    assert_eq!(execute(&program, &[]), Ok(Value::number(0.0)));
}

#[test]
fn ref_builtin_returns_stable_byond_style_heap_identity_text() {
    let syntax = parse(
            "/proc/list_ref()\n\tvar/item = list()\n\treturn ref(item)\n\n/proc/datum_ref()\n\treturn ref(new /datum/example)\n\n/proc/scalar_ref()\n\treturn ref(42)\n",
        )
        .expect("ref source should parse");
    let list_program =
        compile_procedure(&syntax.definitions[0]).expect("list ref builtin should compile");
    let datum_program =
        compile_procedure(&syntax.definitions[1]).expect("datum ref builtin should compile");
    let scalar_program =
        compile_procedure(&syntax.definitions[2]).expect("scalar ref builtin should compile");

    assert!(
        list_program
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::Ref))
    );
    let mut state = ExecutionState::new();
    assert_eq!(
        execute_in_state(&list_program, &[], &mut state),
        // Unobserved implicit `args` stays lazy, so this is the first list.
        Ok(Value::text("[0xe000001]"))
    );
    assert_eq!(
        execute_in_state(&datum_program, &[], &mut state),
        Ok(Value::text("[0xd000001]"))
    );
    assert_eq!(
        execute_in_state(&scalar_program, &[], &mut state),
        Ok(Value::Null)
    );
}

#[test]
fn locate_one_argument_resolves_weakrefs_lists_tags_and_types() {
    let syntax = parse(
            "/proc/run(D, L)\n\treturn locate(ref(D)) == D && locate(ref(L)) == L && locate(\"stack-canary\") == D && locate(/datum/target) == D\n",
        )
        .expect("locate reference source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("locate reference source should compile");
    let mut state = ExecutionState::new();
    let datum = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/target").unwrap());
    state
        .heap_mut()
        .set_datum_field(datum, field("tag"), Value::text("stack-canary"))
        .unwrap();
    let list = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(list)
        .unwrap()
        .add(Value::Datum(datum));
    assert_eq!(
        execute_in_state(
            &program,
            &[Value::Datum(datum), Value::List(list)],
            &mut state,
        ),
        Ok(Value::number(1.0))
    );
}

#[test]
fn native_walks_schedule_replace_stop_and_terminate_without_dm_tasks() {
    let syntax = parse(concat!(
        "/proc/directional(ref, direction, lag)\n\twalk(ref, direction, lag)\n",
        "/proc/stop(ref)\n\twalk(ref, 0)\n",
        "/proc/towards(ref, target, lag)\n\twalk_towards(ref, target, lag)\n",
        "/proc/to(ref, target, minimum, lag)\n\twalk_to(ref, target, minimum, lag)\n",
        "/proc/away(ref, target, maximum, lag)\n\twalk_away(ref, target, maximum, lag)\n",
        "/proc/random(ref, lag)\n\twalk_rand(ref, lag)\n",
    ))
    .expect("walk surface fixture should parse");
    let module = compile_module(&syntax.definitions).expect("every walk surface should compile");
    let mut state = ExecutionState::new();
    let mut turfs = Vec::new();
    for x in 1..=4 {
        let turf = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/turf/walk_test").unwrap());
        for (name, value) in [("x", x), ("y", 1), ("z", 1)] {
            state
                .heap_mut()
                .set_datum_field(turf, field(name), Value::number(value as f32))
                .unwrap();
        }
        state.world_turfs.insert((x, 1, 1), turf);
        turfs.push(turf);
    }
    let movable = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/walk_test").unwrap());
    let target = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/walk_target").unwrap());
    for (datum, x, loc) in [(movable, 1, turfs[0]), (target, 4, turfs[3])] {
        for (name, value) in [("x", x), ("y", 1), ("z", 1)] {
            state
                .heap_mut()
                .set_datum_field(datum, field(name), Value::number(value as f32))
                .unwrap();
        }
        state
            .heap_mut()
            .set_datum_field(datum, field("loc"), Value::Datum(loc))
            .unwrap();
    }
    let x = |state: &ExecutionState| {
        state
            .heap()
            .datum_field(movable, &field("x"))
            .unwrap()
            .as_number()
            .unwrap()
    };
    let run = |name: &str, arguments: &[Value], state: &mut ExecutionState| {
        execute_module_in_state(
            &module,
            module.procedure_id(&format!("/proc/{name}")).unwrap(),
            arguments,
            state,
        )
    };

    assert_eq!(
        run(
            "directional",
            &[
                Value::Datum(movable),
                Value::number(4.0),
                Value::number(2.0)
            ],
            &mut state,
        ),
        Ok(Value::Null),
    );
    assert_eq!(state.scheduled_task_count(), 0);
    assert_eq!(state.next_scheduled_tick(), Some(2));
    assert_eq!(x(&state), 1.0);
    assert_eq!(
        advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state),
        Ok(Vec::new()),
    );
    assert_eq!(x(&state), 1.0);
    advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state).unwrap();
    assert_eq!(x(&state), 2.0);

    run(
        "directional",
        &[
            Value::Datum(movable),
            Value::number(8.0),
            Value::number(1.0),
        ],
        &mut state,
    )
    .unwrap();
    assert_eq!(state.next_scheduled_tick(), Some(3));
    advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state).unwrap();
    assert_eq!(x(&state), 1.0, "a later walk replaces the previous one");
    run("stop", &[Value::Datum(movable)], &mut state).unwrap();
    assert_eq!(state.next_scheduled_tick(), None);
    advance_scheduler(&module, 5, ExecutionLimits::default(), &mut state).unwrap();
    assert_eq!(x(&state), 1.0, "walk(ref, 0) stops future motion");

    run(
        "to",
        &[
            Value::Datum(movable),
            Value::Datum(target),
            Value::number(1.0),
            Value::number(1.0),
        ],
        &mut state,
    )
    .unwrap();
    for expected in [2.0, 3.0] {
        advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state).unwrap();
        assert_eq!(x(&state), expected);
    }
    advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state).unwrap();
    assert_eq!(x(&state), 3.0);
    assert!(
        state.native_walks.is_empty(),
        "walk_to stops at minimum range"
    );

    run(
        "away",
        &[
            Value::Datum(movable),
            Value::Datum(target),
            Value::number(1.0),
            Value::number(1.0),
        ],
        &mut state,
    )
    .unwrap();
    advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state).unwrap();
    assert_eq!(x(&state), 2.0);
    advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state).unwrap();
    assert_eq!(x(&state), 2.0);
    assert!(
        state.native_walks.is_empty(),
        "walk_away stops beyond maximum range"
    );
}

#[test]
fn mapping_multiz_get_step_resolves_up_down_and_world_bounds() {
    let syntax = parse(
            "/proc/neighbors(turf/center, turf/above, turf/below)\n\treturn list(get_step(center, UP), get_step(center, DOWN), get_step(above, UP), get_step(below, DOWN), get_step(center, UP | NORTH))\n",
        )
        .expect("Mapping-shaped vertical get_step fixture should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("Mapping-shaped vertical get_step fixture should compile");
    let mut state = ExecutionState::new();
    let coordinates = [
        ("/turf/center", 4, 9, 2),
        ("/turf/above", 4, 9, 3),
        ("/turf/below", 4, 9, 1),
        ("/turf/above_north", 4, 10, 3),
    ];
    let mut turfs = Vec::new();
    for (path, x, y, z) in coordinates {
        let turf = state
            .heap_mut()
            .allocate_datum(TypePath::parse(path).unwrap());
        for (name, value) in [("x", x), ("y", y), ("z", z)] {
            state
                .heap_mut()
                .set_datum_field(turf, field(name), Value::number(value as f32))
                .unwrap();
        }
        state.world_turfs.insert((x, y, z), turf);
        turfs.push(turf);
    }
    let [center, above, below, above_north] = turfs.as_slice() else {
        unreachable!("fixture has four turfs")
    };

    let result = execute_in_state(
        &program,
        &[
            Value::Datum(*center),
            Value::Datum(*above),
            Value::Datum(*below),
        ],
        &mut state,
    )
    .expect("vertical get_step should execute");
    let Value::List(result) = result else {
        panic!("fixture should return a list")
    };
    assert_eq!(
        state
            .heap()
            .list(result)
            .unwrap()
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>(),
        [
            Value::Datum(*above),
            Value::Datum(*below),
            Value::Null,
            Value::Null,
            Value::Datum(*above_north),
        ]
    );
}

#[test]
fn get_step_does_not_resolve_zero_coordinate_turf_prototype() {
    let syntax = parse("/proc/turf_of(atom/source)\n\treturn get_step(source, 0)\n")
        .expect("get_turf-shaped source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("get_turf-shaped source should compile");
    let mut state = ExecutionState::new();
    let source = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/mob/dummy").unwrap());
    let prototype = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/turf/prototype").unwrap());
    for datum in [source, prototype] {
        for name in ["x", "y", "z"] {
            state
                .heap_mut()
                .set_datum_field(datum, field(name), Value::number(0.0))
                .unwrap();
        }
    }

    assert_eq!(
        execute_in_state(&program, &[Value::Datum(source)], &mut state),
        Ok(Value::Null)
    );
}

#[test]
fn get_turf_of_arbitrary_datum_returns_null_without_coordinate_probe() {
    let syntax = parse("/proc/turf_of(source)\n\treturn get_step(source, 0)\n")
        .expect("get_turf-shaped source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("get_turf-shaped source should compile");
    let mut state = ExecutionState::new();
    let component = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/component/burning").unwrap());
    let parent = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/item/test").unwrap());
    state
        .heap_mut()
        .set_datum_field(component, field("parent"), Value::Datum(parent))
        .unwrap();

    assert_eq!(
        execute_in_state(&program, &[Value::Datum(component)], &mut state),
        Ok(Value::Null),
    );
}

#[test]
fn get_turf_reads_sparse_runtime_coordinate_and_loc_defaults() {
    let syntax = parse("/proc/turf_of(atom/source)\n\treturn get_step(source, 0)\n")
        .expect("get_turf-shaped source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("get_turf-shaped source should compile");
    let mut state = ExecutionState::new();
    let source_type = TypePath::parse("/obj/item/transient").unwrap();
    state.set_initial_values(BTreeMap::from([(
        source_type.clone(),
        BTreeMap::from([
            (field("loc"), Value::Null),
            (field("x"), Value::number(0.0)),
            (field("y"), Value::number(0.0)),
            (field("z"), Value::number(0.0)),
        ]),
    )]));
    let source = allocate_initialized_datum(&mut state, source_type)
        .expect("sparse runtime atom should allocate");
    for name in ["loc", "x", "y", "z"] {
        assert!(
            state
                .heap()
                .datum(source)
                .unwrap()
                .field(&field(name))
                .is_err(),
            "unchanged {name} default should stay sparse"
        );
    }

    assert_eq!(
        execute_in_state(&program, &[Value::Datum(source)], &mut state),
        Ok(Value::Null)
    );
}

#[test]
fn get_turf_walks_nested_movable_locations_before_reading_coordinates() {
    let syntax = parse("/proc/turf_of(atom/source)\n\treturn get_step(source, 0)\n")
        .expect("get_turf-shaped source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("get_turf-shaped source should compile");
    let mut state = ExecutionState::new();
    let turf = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/turf/open/floor").unwrap());
    for (name, value) in [("x", 8.0), ("y", 9.0), ("z", 1.0)] {
        state
            .heap_mut()
            .set_datum_field(turf, field(name), Value::number(value))
            .unwrap();
    }
    state.world_turfs.insert((8, 9, 1), turf);
    let goldgrub = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/mob/living/basic/mining/goldgrub").unwrap());
    let ore = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/item/stack/ore/gold").unwrap());
    for (datum, parent) in [(goldgrub, turf), (ore, goldgrub)] {
        for name in ["x", "y", "z"] {
            state
                .heap_mut()
                .set_datum_field(datum, field(name), Value::number(0.0))
                .unwrap();
        }
        state
            .heap_mut()
            .set_datum_field(datum, field("loc"), Value::Datum(parent))
            .unwrap();
    }

    assert_eq!(
        execute_in_state(&program, &[Value::Datum(ore)], &mut state),
        Ok(Value::Datum(turf)),
    );
}

#[test]
fn get_turf_cycle_protection_handles_loc_chains_beyond_inline_capacity() {
    let syntax = parse("/proc/turf_of(atom/source)\n\treturn get_step(source, 0)\n")
        .expect("get_turf-shaped source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("get_turf-shaped source should compile");
    let mut state = ExecutionState::new();
    let chain = (0..12)
        .map(|_| {
            state
                .heap_mut()
                .allocate_datum(TypePath::parse("/obj/item/container").unwrap())
        })
        .collect::<Vec<_>>();
    for pair in chain.windows(2) {
        state
            .heap_mut()
            .set_datum_field(pair[0], field("loc"), Value::Datum(pair[1]))
            .unwrap();
    }
    state
        .heap_mut()
        .set_datum_field(chain[11], field("loc"), Value::Datum(chain[3]))
        .unwrap();
    for name in ["x", "y", "z"] {
        state
            .heap_mut()
            .set_datum_field(chain[0], field(name), Value::number(0.0))
            .unwrap();
    }

    assert_eq!(
        execute_in_state(&program, &[Value::Datum(chain[0])], &mut state),
        Ok(Value::Null),
    );
}

#[test]
#[ignore = "bounded get_step visited-set allocation microbenchmark"]
fn get_step_visited_storage_microbenchmark() {
    const ITERATIONS: usize = 1_000_000;
    let chain = [11_u32, 22, 33, 44];
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        let mut visited = std::collections::HashSet::new();
        for value in chain {
            assert!(visited.insert(std::hint::black_box(value)));
        }
        std::hint::black_box(visited);
    }
    let hash_set = started.elapsed();
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        let mut visited = smallvec::SmallVec::<[u32; 8]>::new();
        for value in chain {
            assert!(!visited.contains(&std::hint::black_box(value)));
            visited.push(value);
        }
        std::hint::black_box(visited);
    }
    let small_vec = started.elapsed();
    eprintln!(
        "get-step-visited iterations={ITERATIONS} hash_set_ms={} small_vec_ms={}",
        hash_set.as_millis(),
        small_vec.as_millis(),
    );
}

#[test]
fn get_dist_resolves_nested_contents_to_their_containing_turf() {
    let syntax = parse(
            "/proc/check(atom/left, atom/right)\n\treturn list(get_dist(left, right), get_step(left, 0), get_step(right, 0))\n",
        )
        .expect("shock-paddles-shaped source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("nested spatial builtins should compile");
    let mut state = ExecutionState::new();
    let turf = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/turf/open/floor").unwrap());
    for (name, value) in [("x", 12.0), ("y", 17.0), ("z", 1.0)] {
        state
            .heap_mut()
            .set_datum_field(turf, field(name), Value::number(value))
            .unwrap();
    }
    state.world_turfs.insert((12, 17, 1), turf);

    let crate_datum = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/structure/closet/crate").unwrap());
    let defib = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/item/defibrillator").unwrap());
    let paddles = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/item/shockpaddles").unwrap());
    for (datum, parent, coordinates) in [
        (crate_datum, turf, (12.0, 17.0, 1.0)),
        (defib, crate_datum, (12.0, 17.0, 1.0)),
        (paddles, defib, (0.0, 0.0, 0.0)),
    ] {
        state
            .heap_mut()
            .set_datum_field(datum, field("loc"), Value::Datum(parent))
            .unwrap();
        for (name, value) in [
            ("x", coordinates.0),
            ("y", coordinates.1),
            ("z", coordinates.2),
        ] {
            state
                .heap_mut()
                .set_datum_field(datum, field(name), Value::number(value))
                .unwrap();
        }
    }

    let Value::List(result) = execute_in_state(
        &program,
        &[Value::Datum(paddles), Value::Datum(defib)],
        &mut state,
    )
    .expect("contained spatial query should execute") else {
        panic!("expected result list");
    };
    let values = state
        .heap()
        .list(result)
        .unwrap()
        .positions()
        .map(|(_, value)| value.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        values,
        [Value::number(0.0), Value::Datum(turf), Value::Datum(turf)]
    );
}

#[test]
fn zero_argument_viewers_uses_usr_like_the_emote_important_path() {
    let syntax = parse(
            "/proc/run_emote()\n\tvar/count = 0\n\tfor(var/mob/living/viewer in viewers())\n\t\tcount++\n\treturn count\n",
        )
        .expect("important-emote-shaped source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("zero-argument viewers should compile");
    let mut state = ExecutionState::new();
    let center = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/mob/living/center").unwrap());
    let viewer = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/mob/living/viewer").unwrap());
    for (datum, x) in [(center, 5.0), (viewer, 6.0)] {
        for (name, value) in [("x", x), ("y", 5.0), ("z", 1.0)] {
            state
                .heap_mut()
                .set_datum_field(datum, field(name), Value::number(value))
                .unwrap();
        }
    }

    assert_eq!(
        execute_in_context(
            &program,
            &[],
            &mut state,
            &ExecutionContext::new(Value::Null, Value::Datum(center)),
        ),
        Ok(Value::number(2.0)),
    );
    assert_eq!(
        execute_in_context(&program, &[], &mut state, &ExecutionContext::default(),),
        Ok(Value::number(0.0)),
    );
}

#[test]
fn get_step_finds_cardinal_diagonal_and_same_coordinate_turfs() {
    let syntax =
        parse("/proc/step_from(source, direction)\n\treturn get_step(source, direction)\n")
            .expect("source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("get_step builtin should compile");
    assert!(
        program
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::GetStep))
    );

    let mut state = ExecutionState::new();
    let origin = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/turf/open/origin").expect("type path"));
    let north_east = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/turf/open/north_east").expect("type path"));
    let west = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/turf/open/west").expect("type path"));
    for (datum, x, y) in [
        (origin, 4.0, 9.0),
        (north_east, 5.0, 10.0),
        (west, 3.0, 9.0),
    ] {
        for (name, value) in [("x", x), ("y", y), ("z", 2.0)] {
            state
                .heap_mut()
                .set_datum_field(datum, field(name), Value::number(value))
                .expect("coordinate should be set");
        }
    }

    assert_eq!(
        execute_in_state(
            &program,
            &[Value::Datum(origin), Value::number(5.0)],
            &mut state
        ),
        Ok(Value::Datum(north_east))
    );
    assert_eq!(
        execute_in_state(
            &program,
            &[Value::Datum(origin), Value::number(8.0)],
            &mut state
        ),
        Ok(Value::Datum(west))
    );
    assert_eq!(
        execute_in_state(
            &program,
            &[Value::Datum(origin), Value::number(0.0)],
            &mut state
        ),
        Ok(Value::Datum(origin))
    );
    assert_eq!(
        execute_in_state(
            &program,
            &[Value::Datum(origin), Value::number(1.0)],
            &mut state
        ),
        Ok(Value::Null)
    );
    let towards =
        parse("/proc/towards(source, target)\n\treturn get_step_towards(source, target)\n")
            .unwrap();
    let towards = compile_procedure(&towards.definitions[0]).unwrap();
    let target = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/target").unwrap());
    for (name, value) in [("x", 20.0), ("y", 30.0), ("z", 2.0)] {
        state
            .heap_mut()
            .set_datum_field(target, field(name), Value::number(value))
            .unwrap();
    }
    assert_eq!(
        execute_in_state(
            &towards,
            &[Value::Datum(origin), Value::Datum(target)],
            &mut state,
        ),
        Ok(Value::Datum(north_east))
    );
}

#[test]
fn resource_regex_json_and_headless_ui_natives_follow_byond_contracts() {
    let syntax = parse(
            "/proc/resource(value)\n\treturn fcopy_rsc(value)\n/proc/quote(value)\n\treturn REGEX_QUOTE(value)\n/proc/pretty_flag()\n\treturn JSON_PRETTY_PRINT\n/proc/mask_inverse()\n\treturn MASK_INVERSE\n/proc/floor_value(value, multiple)\n\treturn FLOOR(value, multiple)\n/proc/ui(client)\n\twinset(client, \"main\", \"flash=5\")\n\treturn browse(\"<b>ready</b>\", \"window=status\")\n/proc/window_exists(client, control)\n\treturn winexists(client, control)\n/proc/choose(client)\n\treturn alert(client, \"Continue?\", \"Dream64\", \"Yes\", \"No\")\n/proc/colors()\n\tvar/icon/value = icon()\n\tvalue.MapColors(1,0,0, 0,1,0, 0,0,1, 0,0,0)\n\tvalue.Blend(\"#ffffff\", ICON_SUBTRACT, 2, 3)\n\tvalue.SetIntensity(0.25, 0.5, 0.75)\n\treturn value\n",
        )
        .unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let mut state = ExecutionState::new();
    let run = |path: &str, arguments: &[Value], state: &mut ExecutionState| {
        execute_module_in_state(
            &module,
            module.procedure_id(path).unwrap(),
            arguments,
            state,
        )
        .unwrap()
    };
    assert_eq!(
        run("/proc/resource", &[Value::text("icons/a.dmi")], &mut state),
        Value::file("icons/a.dmi")
    );
    assert_eq!(
        run("/proc/quote", &[Value::text("a+b.c?")], &mut state),
        Value::text("a\\+b\\.c\\?")
    );
    assert_eq!(
        run("/proc/pretty_flag", &[], &mut state),
        Value::number(1.0)
    );
    assert_eq!(
        run("/proc/mask_inverse", &[], &mut state),
        Value::number(1.0)
    );
    assert_eq!(
        run(
            "/proc/floor_value",
            &[Value::number(17.0), Value::number(5.0)],
            &mut state,
        ),
        Value::number(15.0)
    );

    let client = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/client").unwrap());
    assert_eq!(
        run("/proc/choose", &[Value::Datum(client)], &mut state),
        Value::text("Yes")
    );
    let browse = run("/proc/ui", &[Value::Datum(client)], &mut state);
    assert!(matches!(browse, Value::List(_)));
    let settings = state
        .heap()
        .datum_field(client, &field("_dream64_winset"))
        .unwrap();
    let Value::List(settings) = settings else {
        panic!("winset state should be a list");
    };
    assert_eq!(
        state
            .heap()
            .list(*settings)
            .unwrap()
            .get_key(&Value::text("main")),
        Ok(&Value::text("flash=5"))
    );
    assert_eq!(
        run(
            "/proc/window_exists",
            &[Value::Datum(client), Value::text("main")],
            &mut state,
        ),
        Value::number(1.0)
    );
    let Value::Datum(icon) = run("/proc/colors", &[], &mut state) else {
        panic!("icon() should return an icon datum");
    };
    let Value::List(matrix) = state
        .heap()
        .datum_field(icon, &field("_dream64_color_matrix"))
        .unwrap()
    else {
        panic!("MapColors should retain the headless matrix");
    };
    assert_eq!(state.heap().list(*matrix).unwrap().len(), 12);
    assert_eq!(
        state.heap().list(*matrix).unwrap().get(1),
        Ok(&Value::number(0.25))
    );
    assert_eq!(
        state.heap().list(*matrix).unwrap().get(5),
        Ok(&Value::number(0.5))
    );
    assert_eq!(
        state.heap().list(*matrix).unwrap().get(9),
        Ok(&Value::number(0.75))
    );
    let Value::List(blends) = state
        .heap()
        .datum_field(icon, &field("_dream64_blends"))
        .unwrap()
    else {
        panic!("Blend should retain its headless composition operation");
    };
    assert_eq!(state.heap().list(*blends).unwrap().len(), 1);
}

#[test]
fn icon_datums_materialize_as_resources_without_becoming_files_themselves() {
    let syntax = parse(
            "/proc/body_zone_asset()\n\tvar/icon/body_zone = icon('icons/hud/screen_gen.dmi', \"head\")\n\tvar/was_icon_file = isfile(body_zone)\n\tif(!isfile(body_zone))\n\t\tbody_zone = fcopy_rsc(body_zone)\n\treturn list(was_icon_file, isfile(body_zone), isfile(\"icons/hud/screen_gen.dmi\"), body_zone)\n/proc/cloned_body_zone_asset()\n\tvar/icon/original = icon('icons/hud/screen_gen.dmi', \"chest\")\n\treturn fcopy_rsc(icon(original))\n",
        )
        .unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let mut state = ExecutionState::new();
    let result = execute_module_in_state(
        &module,
        module.procedure_id("/proc/body_zone_asset").unwrap(),
        &[],
        &mut state,
    )
    .unwrap();
    let Value::List(result) = result else {
        panic!("asset contract should return its observations");
    };
    let result = state.heap().list(result).unwrap();
    assert_eq!(result.get(1), Ok(&Value::number(0.0)));
    assert_eq!(result.get(2), Ok(&Value::number(1.0)));
    assert_eq!(result.get(3), Ok(&Value::number(0.0)));
    assert_eq!(result.get(4), Ok(&Value::file("icons/hud/screen_gen.dmi")));
    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/cloned_body_zone_asset").unwrap(),
            &[],
            &mut state,
        ),
        Ok(Value::file("icons/hud/screen_gen.dmi"))
    );
}

#[test]
fn byond_control_freak_constants_are_available_to_dm_code() {
    let syntax =
        parse("/proc/control_flags()\n\treturn CONTROL_FREAK_SKIN | CONTROL_FREAK_MACROS\n")
            .unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    assert_eq!(
        execute_module(
            &module,
            module.procedure_id("/proc/control_flags").unwrap(),
            &[]
        ),
        Ok(Value::number(3.0))
    );
}

#[test]
fn contextual_icon_new_matches_icon_builtin_constructor_fields() {
    let syntax = parse(
            "/proc/contextual()\n\tvar/icon/value = new /icon(fcopy_rsc(\"icons/title.dmi\"), \"idle\", 4, 2, 1)\n\treturn value\n/proc/direct()\n\treturn icon(fcopy_rsc(\"icons/title.dmi\"), \"idle\", 4, 2, 1)\n/proc/title_shaped()\n\tvar/icon/title\n\ttitle = new /icon(fcopy_rsc(\"icons/runtime/default_title.dmi\"))\n\treturn title\n",
        )
        .unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let mut state = ExecutionState::new();
    let run = |path: &str, state: &mut ExecutionState| {
        let value =
            execute_module_in_state(&module, module.procedure_id(path).unwrap(), &[], state)
                .unwrap();
        let Value::Datum(datum) = value else {
            panic!("icon constructor should return a datum");
        };
        datum
    };
    let contextual = run("/proc/contextual", &mut state);
    let direct = run("/proc/direct", &mut state);
    for name in ["icon", "icon_state", "dir", "frame", "moving"] {
        assert_eq!(
            state.heap().datum_field(contextual, &field(name)),
            state.heap().datum_field(direct, &field(name)),
            "contextual new should preserve the builtin {name} field",
        );
    }
    let title = run("/proc/title_shaped", &mut state);
    assert_eq!(
        state.heap().datum_field(title, &field("icon")),
        Ok(&Value::file("icons/runtime/default_title.dmi")),
    );
}

#[test]
fn icon_icon_states_method_dispatches_natively_against_backing_dmi() {
    // Monkestation's greyscale_previews subsystem calls `map_icon.IconStates()`
    // on a statically typed `var/icon`. This must resolve as a native method,
    // not fall through to a dynamic `/icon/proc/IconStates` lookup.
    use std::io::Write as _;

    use flate2::Compression;
    use flate2::write::ZlibEncoder;

    let root = std::env::temp_dir().join(format!(
        "dream64-icon-states-dispatch-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("icons")).unwrap();
    let description = concat!(
        "# BEGIN DMI\n",
        "version = 4.0\n",
        "width = 32\n",
        "height = 32\n",
        "state = \"error\"\n",
        "dirs = 1\n",
        "frames = 1\n",
        "state = \"template\"\n",
        "dirs = 1\n",
        "frames = 1\n",
        "# END DMI\n",
    );
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(description.as_bytes()).unwrap();
    let compressed = encoder.finish().unwrap();
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut push_chunk = |kind: &[u8; 4], data: &[u8]| {
        png.extend_from_slice(&(data.len() as u32).to_be_bytes());
        png.extend_from_slice(kind);
        png.extend_from_slice(data);
        png.extend_from_slice(&[0; 4]);
    };
    let mut header = Vec::new();
    header.extend_from_slice(&32u32.to_be_bytes());
    header.extend_from_slice(&32u32.to_be_bytes());
    header.extend_from_slice(&[8, 6, 0, 0, 0]);
    push_chunk(b"IHDR", &header);
    let mut text = b"Description\0\0".to_vec();
    text.extend_from_slice(&compressed);
    push_chunk(b"zTXt", &text);
    push_chunk(b"IEND", &[]);
    std::fs::write(root.join("icons/greyscale.dmi"), png).unwrap();

    let syntax = parse(
        "/proc/probe()\n\
             \tvar/icon/map_icon = icon('icons/greyscale.dmi')\n\
             \treturn list(\"template\" in map_icon.IconStates(), \"missing\" in map_icon.IconStates())\n",
    )
    .unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let mut state = ExecutionState::new();
    state.set_project_root(root.clone());
    let result = execute_module_in_state(
        &module,
        module.procedure_id("/proc/probe").unwrap(),
        &[],
        &mut state,
    )
    .unwrap();
    let Value::List(result) = result else {
        panic!("probe should return a list");
    };
    let result = state.heap().list(result).unwrap();
    assert_eq!(result.get(1), Ok(&Value::number(1.0)));
    assert_eq!(result.get(2), Ok(&Value::number(0.0)));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn icon_copy_constructor_preserves_render_state_and_mutates_independently() {
    // Exact constructor shape at the start of Monkestation's getFlatIcon:
    // `flat_template = icon(file, state); flat = icon(flat_template)`.
    let syntax = parse(
        "/proc/get_flat_icon_constructor_probe()\n\
             \tvar/icon/flat_template = icon('icons/blanks/32x32.dmi', \"nothing\")\n\
             \tflat_template.Scale(48, 64)\n\
             \tflat_template.DrawBox(\"#ffffff\", 1, 1, 8, 8)\n\
             \tvar/icon/flat = icon(flat_template)\n\
             \tflat.DrawBox(\"#000000\", 2, 2, 4, 4)\n\
             \tflat.Scale(20, 30)\n\
             \treturn list(flat_template, flat)\n",
    )
    .expect("getFlatIcon-shaped icon copy fixture should parse");
    let module = compile_module(&syntax.definitions)
        .expect("getFlatIcon-shaped icon copy fixture should compile");
    let mut state = ExecutionState::new();
    let Value::List(result) = execute_module_in_state(
        &module,
        module
            .procedure_id("/proc/get_flat_icon_constructor_probe")
            .unwrap(),
        &[],
        &mut state,
    )
    .expect("icon copy constructor should execute") else {
        panic!("icon copy fixture should return its two icons");
    };
    let values = state
        .heap()
        .list(result)
        .unwrap()
        .positions()
        .map(|(_, value)| value.clone())
        .collect::<Vec<_>>();
    let [Value::Datum(template), Value::Datum(flat)] = values.as_slice() else {
        panic!("icon copy fixture should return two icon datums");
    };

    assert_ne!(template, flat, "the copy must have independent identity");
    assert_eq!(
        state.heap().datum_field(*template, &field("icon")),
        Ok(&Value::file("icons/blanks/32x32.dmi")),
    );
    assert_eq!(
        state.heap().datum_field(*flat, &field("icon")),
        Ok(&Value::file("icons/blanks/32x32.dmi")),
        "the clone must retain the backing resource instead of nesting the source datum",
    );
    assert_eq!(
        state
            .heap()
            .datum_field(*template, &field("_dream64_width")),
        Ok(&Value::number(48.0)),
        "mutating the clone must not change the template width",
    );
    assert_eq!(
        state
            .heap()
            .datum_field(*template, &field("_dream64_height")),
        Ok(&Value::number(64.0)),
        "mutating the clone must not change the template height",
    );
    assert_eq!(
        state.heap().datum_field(*flat, &field("_dream64_width")),
        Ok(&Value::number(20.0)),
    );
    assert_eq!(
        state.heap().datum_field(*flat, &field("_dream64_height")),
        Ok(&Value::number(30.0)),
    );

    let Value::List(template_operations) = state
        .heap()
        .datum_field(*template, &field("_dream64_icon_operations"))
        .unwrap()
    else {
        panic!("template should retain its operation list");
    };
    let Value::List(flat_operations) = state
        .heap()
        .datum_field(*flat, &field("_dream64_icon_operations"))
        .unwrap()
    else {
        panic!("clone should retain its copied operation list");
    };
    assert_ne!(
        template_operations, flat_operations,
        "the operation list itself must be copied",
    );
    assert_eq!(state.heap().list(*template_operations).unwrap().len(), 2);
    assert_eq!(
        state.heap().list(*flat_operations).unwrap().len(),
        4,
        "operations added to the clone must not leak back to the template",
    );
}

#[test]
fn fcopy_of_a_mutated_icon_writes_a_composited_dmi() {
    // A real BYOND-authored 32x32 template (states "box" = opaque #808080
    // bottom-left quadrant, and "stripe").
    const TEMPLATE: &[u8] = include_bytes!("../../../fixtures/oracle/icon_ops/template.dmi");
    let root = std::env::temp_dir().join(format!(
        "dream64-icon-fcopy-composite-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("icons")).unwrap();
    std::fs::write(root.join("icons/template.dmi"), TEMPLATE).unwrap();

    let syntax = parse(
        "/proc/probe()\n\
             \tvar/icon/sprite = icon('icons/template.dmi', \"box\")\n\
             \tsprite.Scale(16, 16)\n\
             \tsprite.Blend(\"#ff0000\", ICON_MULTIPLY)\n\
             \treturn fcopy(sprite, \"gen/out.dmi\")\n",
    )
    .expect("mutated-icon fcopy fixture should parse");
    let module = compile_module(&syntax.definitions).expect("fixture compiles");
    let mut state = ExecutionState::new();
    state.set_project_root(root.clone());
    let result = execute_module_in_state(
        &module,
        module.procedure_id("/proc/probe").unwrap(),
        &[],
        &mut state,
    )
    .unwrap();
    assert_eq!(result, Value::number(1.0), "fcopy must report success");

    let written = std::fs::read(root.join("gen/out.dmi")).expect("output DMI written");
    let dmi = dm_icon::IconBitmap::from_dmi_bytes(&written).expect("output is a valid PNG DMI");
    assert_eq!((dmi.width, dmi.height), (16, 16), "Scale(16,16) must apply");
    assert_eq!(dmi.state_names(), vec!["box".to_owned()]);
    // #ff0000 multiplied over the #808080 fill -> #800000, opaque.
    let opaque_red = dmi.states[0].cells[0]
        .pixels
        .iter()
        .any(|p| *p == [128, 0, 0, 255]);
    assert!(opaque_red, "the colourised fill pixels must survive fcopy");

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn icon_geometry_methods_mutate_headless_dimensions_and_dispatch_both_ways() {
    let syntax = parse(
            "/icon/proc/resize_inside()\n\tScale(48, 64)\n\treturn Width() * 100 + Height()\n/proc/resize_outside()\n\tvar/icon/value = icon()\n\tvalue.Scale(20)\n\tvalue.Crop(2, 3, 11, 18)\n\tvalue.Shift(NORTH, 2)\n\tvalue.DrawBox(\"#ffffff\", 1, 1, 2, 2)\n\tvalue.Insert(icon(), \"state\")\n\treturn value.Width() * 100 + value.Height()\n",
        )
        .unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let mut state = ExecutionState::new();
    let icon = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/icon").unwrap());
    assert_eq!(
        execute_module_in_context(
            &module,
            module.procedure_id("/icon/proc/resize_inside").unwrap(),
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(icon), Value::Null),
        ),
        Ok(Value::number(4864.0))
    );
    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/resize_outside").unwrap(),
            &[],
            &mut state,
        ),
        Ok(Value::number(1016.0))
    );
}

#[test]
fn dynamic_icon_calls_cover_turn_flip_and_swap_color() {
    let syntax = parse(
            "/proc/run()\n\tvar/icon/value = icon()\n\tcall(value, \"Turn\")(90)\n\tcall(value, \"Flip\")(1)\n\tcall(value, \"SwapColor\")(\"#000000\", \"#ffffff\")\n\treturn value\n",
        )
        .expect("dynamic icon call source should parse");
    let module = compile_module(&syntax.definitions).expect("dynamic icon calls should compile");
    let mut state = ExecutionState::new();
    let Value::Datum(icon) = execute_module_in_state(
        &module,
        module.procedure_id("/proc/run").unwrap(),
        &[],
        &mut state,
    )
    .expect("dynamic icon calls should use the native method bridge") else {
        panic!("dynamic icon fixture should return its icon")
    };
    let Value::List(operations) = state
        .heap()
        .datum_field(icon, &field("_dream64_icon_operations"))
        .unwrap()
    else {
        panic!("icon operations should be recorded")
    };
    assert_eq!(state.heap().list(*operations).unwrap().len(), 3);
}

#[test]
fn procedure_static_list_persists_for_protected_holder_shape() {
    let syntax = parse(
            "/datum/manager/proc/get_protected(list/list_ref)\n\tvar/static/list/protected_lists\n\tif(list_ref)\n\t\tprotected_lists = list_ref\n\treturn protected_lists\n/datum/manager/proc/update(key, list/value)\n\tvar/list/protected = src.get_protected()\n\tprotected[key] = value\n\treturn protected[key]\n/proc/run()\n\tvar/datum/manager/manager = new /datum/manager\n\tmanager.get_protected(list())\n\treturn manager.update(\"ADMIN\", list(1, 2))\n",
        )
        .unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let mut state = ExecutionState::new();
    let result = execute_module_in_state(
        &module,
        module.procedure_id("/proc/run").unwrap(),
        &[],
        &mut state,
    )
    .unwrap();
    let Value::List(value) = result else {
        panic!("procedure-static protected storage should remain a list");
    };
    assert_eq!(state.heap().list(value).unwrap().len(), 2);
}

#[test]
#[ignore = "bounded procedure-static lookup allocation microbenchmark"]
fn procedure_static_borrowed_path_lookup_microbenchmark() {
    const ITERATIONS: usize = 1_000_000;
    let path = "/datum/parsed_map/proc/build_coordinate@214645";
    let flat = BTreeMap::from([((path.to_owned(), 7_u16), Value::number(1.0))]);
    let grouped = BTreeMap::from([(
        path.to_owned(),
        BTreeMap::from([(7_u16, Value::number(1.0))]),
    )]);

    let started = Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(flat.get(&(path.to_owned(), 7_u16)));
    }
    let allocating_flat = started.elapsed();
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(grouped.get(path).and_then(|slots| slots.get(&7_u16)));
    }
    let borrowed_grouped = started.elapsed();
    eprintln!(
        "procedure-static-lookup iterations={ITERATIONS} allocating_flat_ms={} borrowed_grouped_ms={}",
        allocating_flat.as_millis(),
        borrowed_grouped.as_millis(),
    );
}

#[test]
fn implicit_src_engine_methods_resolve_for_icon_and_matrix_procs() {
    let syntax = parse(
            "/icon/proc/colors()\n\tMapColors(1,0,0, 0,1,0, 0,0,1, 0,0,0)\n\tBlend(\"#808080\", ICON_MULTIPLY)\n\tSetIntensity(0.5)\n\treturn src\n/matrix/proc/rotate()\n\tTurn(90)\n\treturn src\n",
        )
        .unwrap();
    let icon_program = compile_procedure(&syntax.definitions[0]).unwrap();
    let matrix_program = compile_procedure(&syntax.definitions[1]).unwrap();
    let mut state = ExecutionState::new();
    let icon = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/icon").unwrap());
    assert_eq!(
        execute_in_context(
            &icon_program,
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(icon), Value::Null),
        ),
        Ok(Value::Datum(icon))
    );
    assert!(matches!(
        state
            .heap()
            .datum_field(icon, &field("_dream64_color_matrix")),
        Ok(Value::List(_))
    ));
    assert!(matches!(
        state.heap().datum_field(icon, &field("_dream64_blends")),
        Ok(Value::List(_))
    ));
    let Value::List(intensity) = state
        .heap()
        .datum_field(icon, &field("_dream64_color_matrix"))
        .unwrap()
    else {
        panic!("SetIntensity should lower to an icon color matrix");
    };
    for index in [1, 5, 9] {
        assert_eq!(
            state.heap().list(*intensity).unwrap().get(index),
            Ok(&Value::number(0.5))
        );
    }

    let matrix = allocate_matrix([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], state.heap_mut()).unwrap();
    assert_eq!(
        execute_in_context(
            &matrix_program,
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(matrix), Value::Null),
        ),
        Ok(Value::Datum(matrix))
    );
    assert_eq!(
        matrix_components(matrix, state.heap()).unwrap(),
        [0.0, 1.0, 0.0, -1.0, 0.0, 0.0]
    );
}

fn compile_range_programs() -> (Program, Program, Program) {
    let syntax = parse(
            "/proc/normal(distance, center)\n\treturn range(distance, center)\n/proc/reversed(center, distance)\n\treturn range(center, distance)\n/proc/implicit(distance)\n\treturn range(distance)\n",
        )
        .expect("range source should parse");
    let normal = compile_procedure(&syntax.definitions[0]).expect("range should compile");
    let reversed =
        compile_procedure(&syntax.definitions[1]).expect("reversed range should compile");
    let implicit =
        compile_procedure(&syntax.definitions[2]).expect("implicit range should compile");
    (normal, reversed, implicit)
}

#[test]
fn range_returns_all_same_z_atoms_in_a_square_and_accepts_reversed_arguments() {
    let (normal, reversed, implicit) = compile_range_programs();
    assert!(
        normal
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::Range { argument_count: 2 }))
    );
    assert!(
        implicit
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::Range { argument_count: 1 }))
    );

    let mut state = ExecutionState::new();
    let center = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/turf/open/center").expect("type path"));
    let adjacent = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/item/adjacent").expect("type path"));
    let diagonal = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/mob/living/diagonal").expect("type path"));
    let far = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/turf/open/far").expect("type path"));
    let other_z = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/item/other_z").expect("type path"));
    let area = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/area/test").expect("type path"));
    for (datum, x, y, z) in [
        (center, 10.0, 10.0, 1.0),
        (adjacent, 11.0, 10.0, 1.0),
        (diagonal, 9.0, 9.0, 1.0),
        (far, 12.0, 10.0, 1.0),
        (other_z, 10.0, 10.0, 2.0),
        (area, 10.0, 10.0, 1.0),
    ] {
        for (name, value) in [("x", x), ("y", y), ("z", z)] {
            state
                .heap_mut()
                .set_datum_field(datum, field(name), Value::number(value))
                .expect("coordinate should be set");
        }
    }
    let values = |value: Value, state: &ExecutionState| {
        let Value::List(list) = value else {
            panic!("range should return a list");
        };
        state
            .heap()
            .list(list)
            .expect("range list should be live")
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>()
    };
    let normal_values = values(
        execute_in_state(
            &normal,
            &[Value::number(1.0), Value::Datum(center)],
            &mut state,
        )
        .expect("normal range should execute"),
        &state,
    );
    assert_eq!(
        normal_values,
        vec![
            Value::Datum(center),
            Value::Datum(adjacent),
            Value::Datum(diagonal)
        ]
    );
    let reversed_values = values(
        execute_in_state(
            &reversed,
            &[Value::Datum(center), Value::number(1.0)],
            &mut state,
        )
        .expect("reversed range should execute"),
        &state,
    );
    assert_eq!(reversed_values, normal_values);
    let context = ExecutionContext::new(Value::Datum(center), Value::Null);
    let implicit_values = values(
        execute_in_context(&implicit, &[Value::number(0.0)], &mut state, &context)
            .expect("implicit range should execute"),
        &state,
    );
    assert_eq!(implicit_values, vec![Value::Datum(center)]);
}

#[test]
fn range_uses_indexed_center_area_and_turf_contents_in_byond_order() {
    let syntax = parse("/proc/nearby(distance, center)\n\treturn range(distance, center)\n")
        .expect("range source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("range should compile");
    let mut state = ExecutionState::new();
    let area_a = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/area/a").expect("area path"));
    let area_b = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/area/b").expect("area path"));
    let turf_path = TypePath::parse("/turf/indexed").expect("turf path");
    let mut turfs = BTreeMap::new();
    for coordinate in [(2, 2, 1), (1, 2, 1), (2, 1, 1), (1, 1, 1)] {
        let turf = state.heap_mut().allocate_datum(turf_path.clone());
        for (name, value) in [
            ("x", coordinate.0),
            ("y", coordinate.1),
            ("z", coordinate.2),
        ] {
            state
                .heap_mut()
                .set_datum_field(turf, field(name), Value::number(value as f32))
                .expect("coordinate should be writable");
        }
        state.world_turfs.insert(coordinate, turf);
        state.world_areas.insert(
            coordinate,
            if coordinate.0 == 1 && coordinate.1 == 2 {
                area_b
            } else {
                area_a
            },
        );
        state
            .ensure_contents(turf)
            .expect("indexed turf should expose contents");
        turfs.insert(coordinate, turf);
    }

    let object_path = TypePath::parse("/obj/indexed").expect("object path");
    let mut objects = BTreeMap::new();
    for coordinate in [(1, 2, 1), (2, 2, 1), (1, 1, 1), (2, 1, 1)] {
        let object = state.heap_mut().allocate_datum(object_path.clone());
        let contents = state
            .heap()
            .datum_field(turfs[&coordinate], &field("contents"))
            .ok()
            .and_then(|value| match value {
                Value::List(list) => Some(*list),
                _ => None,
            })
            .expect("turf contents should be a list");
        state
            .heap_mut()
            .list_mut(contents)
            .expect("contents list should be live")
            .add(Value::Datum(object));
        objects.insert(coordinate, object);
    }
    let unindexed = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/unindexed").expect("object path"));
    for (name, value) in [("x", 2), ("y", 1), ("z", 1)] {
        state
            .heap_mut()
            .set_datum_field(unindexed, field(name), Value::number(value as f32))
            .expect("unindexed coordinate should be writable");
    }

    let result = execute_in_state(
        &program,
        &[Value::number(1.0), Value::Datum(turfs[&(2, 1, 1)])],
        &mut state,
    )
    .expect("indexed range should execute");
    let Value::List(result) = result else {
        panic!("range should return a list");
    };
    let values = state
        .heap()
        .list(result)
        .expect("range list should be live")
        .positions()
        .map(|(_, value)| value.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        values,
        vec![
            Value::Datum(turfs[&(2, 1, 1)]),
            Value::Datum(area_a),
            Value::Datum(objects[&(2, 1, 1)]),
            Value::Datum(turfs[&(1, 1, 1)]),
            Value::Datum(objects[&(1, 1, 1)]),
            Value::Datum(turfs[&(1, 2, 1)]),
            Value::Datum(area_b),
            Value::Datum(objects[&(1, 2, 1)]),
            Value::Datum(turfs[&(2, 2, 1)]),
            Value::Datum(objects[&(2, 2, 1)]),
        ]
    );
}

#[test]
fn area_coordinate_fields_come_from_contained_world_cells() {
    let syntax = parse(
        "/proc/read_z(area/place)\n\treturn place.z\n/proc/read_x(area/place)\n\treturn place.x\n",
    )
    .expect("area coordinate fixture should parse");
    let module = compile_module(&syntax.definitions).expect("fixture should compile");
    let mut state = ExecutionState::new();
    let area = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/area/weather_target").unwrap());
    state.world_areas.insert((7, 11, 5), area);
    state.world_areas.insert((9, 12, 5), area);

    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/read_z").unwrap(),
            &[Value::Datum(area)],
            &mut state,
        ),
        Ok(Value::number(5.0)),
        "area.z must expose the z-level of a contained turf for areas_in_z registration",
    );
    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/read_x").unwrap(),
            &[Value::Datum(area)],
            &mut state,
        ),
        Ok(Value::number(7.0)),
    );
    assert_eq!(
        crate::builtins::datum_coordinates(&state, &Value::Datum(area)),
        Some((7.0, 11.0, 5.0)),
    );
}

#[test]
fn qdel_builtin_removes_a_datum_from_heap() {
    let syntax = parse("/proc/test(v)\n\tqdel(v)\n").expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("qdel call should compile");
    assert!(
            program
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instruction::StandardBuiltin { name, .. } if name == "qdel"))
        );

    let mut state = ExecutionState::new();
    let datum = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/example").expect("type path"));
    let result = execute_in_state(&program, &[Value::Datum(datum)], &mut state)
        .expect("qdel should execute");
    assert_eq!(result, Value::Null);
    assert!(state.heap().datum(datum).is_err());
}

#[test]
fn del_builtin_destroys_the_target_list_itself() {
    let syntax = parse("/proc/test(v)\n\tdel(v)\n").expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("del should compile");
    let mut state = ExecutionState::new();
    let list = state.heap_mut().allocate_list();
    execute_in_state(&program, &[Value::List(list)], &mut state)
        .expect("del should destroy a list");
    assert!(state.heap().list(list).is_err());
}

#[test]
fn qdel_builtin_is_idempotent_on_stale_reference() {
    let syntax =
        parse("/proc/test(v)\n\tqdel(v)\n\tqdel(v)\n\treturn 1").expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("qdel call should compile");
    let mut state = ExecutionState::new();
    let target = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/example").expect("type path"));

    assert_eq!(
        execute_in_state(&program, &[Value::Datum(target)], &mut state),
        Ok(Value::number(1.0))
    );
    assert!(state.heap().datum(target).is_err());
}

#[test]
fn del_builtin_is_idempotent_on_stale_reference() {
    let syntax =
        parse("/proc/test(v)\n\tdel(v)\n\tdel(v)\n\treturn 1").expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("del call should compile");
    let mut state = ExecutionState::new();
    let target = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/example").expect("type path"));

    assert_eq!(
        execute_in_state(&program, &[Value::Datum(target)], &mut state),
        Ok(Value::number(1.0))
    );
    assert!(state.heap().datum(target).is_err());
}

#[test]
fn deleted_datum_references_are_false_in_conditions() {
    let syntax =
        parse("/proc/run(target)\n\tdel(target)\n\treturn !target\n").expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("deletion module should compile");
    let entry = module
        .procedure_id("/proc/run")
        .expect("entry should exist");
    let mut state = ExecutionState::new();
    let target = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/example").expect("type path"));

    assert_eq!(
        execute_module_in_state(&module, entry, &[Value::Datum(target)], &mut state),
        Ok(Value::number(1.0))
    );
    assert!(state.heap().datum(target).is_err());
}

#[test]
fn del_dispatches_effective_hook_before_invalidating_the_datum() {
    let syntax = parse(
            "/proc/run(v)\n\tdel(v)\n\treturn global.calls\n/datum/example/Del()\n\tglobal.calls += 1\n",
        )
        .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("deletion module should compile");
    let entry = module
        .procedure_id("/proc/run")
        .expect("entry should exist");
    let mut state = ExecutionState::new();
    state.set_global(field("calls"), Value::number(0.0));
    let datum = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/example").unwrap());

    assert_eq!(
        execute_module_in_state(&module, entry, &[Value::Datum(datum)], &mut state),
        Ok(Value::number(1.0))
    );
    assert!(state.heap().datum(datum).is_err());
}

#[test]
fn del_finalizes_after_hook_failure_and_tolerates_reentrant_del() {
    for body in ["\tCRASH(\"boom\")\n", "\tdel(src)\n"] {
        let syntax = parse(&format!(
            "/proc/run(v)\n\tdel(v)\n/datum/example/Del()\n\tglobal.calls += 1\n{body}"
        ))
        .expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("deletion module should compile");
        let entry = module
            .procedure_id("/proc/run")
            .expect("entry should exist");
        let mut state = ExecutionState::new();
        state.set_global(field("calls"), Value::number(0.0));
        let datum = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/example").unwrap());

        let result = execute_module_in_state(&module, entry, &[Value::Datum(datum)], &mut state);
        if body.contains("CRASH") {
            assert!(result.is_err());
        } else {
            assert_eq!(result, Ok(Value::Null));
        }
        assert!(state.heap().datum(datum).is_err());
        assert_eq!(state.global(&field("calls")), Some(&Value::number(1.0)));
    }
}

#[test]
fn project_qdel_procedure_shadows_the_native_fallback() {
    let syntax = parse(
        "/proc/qdel(v)\n\tglobal.calls += 1\n/proc/run(v)\n\tqdel(v)\n\treturn global.calls\n",
    )
    .expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("qdel override should compile");
    let entry = module
        .procedure_id("/proc/run")
        .expect("entry should exist");
    let mut state = ExecutionState::new();
    state.set_global(field("calls"), Value::number(0.0));
    let datum = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/example").unwrap());

    assert_eq!(
        execute_module_in_state(&module, entry, &[Value::Datum(datum)], &mut state),
        Ok(Value::number(1.0))
    );
    assert!(state.heap().datum(datum).is_ok());
}

#[test]
fn project_file2list_wrapper_shadows_the_native_fallback() {
    let syntax = parse(concat!(
        "/proc/file2list(path, separator = \"|\", trim = TRUE)\n",
        "\treturn list(path, separator, trim)\n",
        "/proc/run()\n",
        "\treturn file2list(\"virtual.txt\", \"::\", FALSE)\n",
    ))
    .unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let mut state = ExecutionState::new();
    let Value::List(result) = execute_module_in_state(
        &module,
        module.procedure_id("/proc/run").unwrap(),
        &[],
        &mut state,
    )
    .unwrap() else {
        panic!("project wrapper should run instead of native filesystem access");
    };
    assert_eq!(
        state
            .heap()
            .list(result)
            .unwrap()
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>(),
        vec![
            Value::text("virtual.txt"),
            Value::text("::"),
            Value::number(0.0)
        ]
    );
}

#[test]
fn project_file2list_wrapper_returns_empty_for_missing_config_include() {
    let syntax = parse(concat!(
        "/proc/trim(value)\n",
        "\treturn trimtext(value)\n",
        "/proc/file2list(filename, separator = \"\\n\", trim_file = TRUE)\n",
        "\tif(trim_file)\n",
        "\t\treturn splittext(trim(file2text(filename)), separator)\n",
        "\treturn splittext(file2text(filename), separator)\n",
        "/proc/run()\n",
        "\treturn file2list(\"config/auxtools.txt\")\n",
    ))
    .unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let mut state = ExecutionState::new();
    let root = std::env::temp_dir().join(format!("dream64-missing-config-{}", std::process::id()));
    std::fs::create_dir_all(root.join("config")).unwrap();
    state.set_project_root(root.clone());
    let result = execute_module_in_state(
        &module,
        module.procedure_id("/proc/run").unwrap(),
        &[],
        &mut state,
    )
    .unwrap();
    let Value::List(result) = result else {
        panic!("missing include must produce an empty list");
    };
    assert!(state.heap().list(result).unwrap().is_empty());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn project_sort_list_wrapper_wins_and_dispatches_comparator_proc_reference() {
    let syntax = parse(concat!(
        "/proc/cmp_desc(left, right)\n\treturn right - left\n",
        "/proc/sort_list(values, comparator)\n",
        "\tvar/list/result = values.Copy()\n",
        "\tvar/temporary\n",
        "\tif(call(comparator)(result[1], result[2]) > 0)\n",
        "\t\ttemporary = result[1]\n",
        "\t\tresult[1] = result[2]\n",
        "\t\tresult[2] = temporary\n",
        "\treturn result\n",
        "/proc/run()\n\treturn sort_list(list(1, 3), /proc/cmp_desc)\n",
    ))
    .expect("project sort wrapper should parse");
    let module = compile_module(&syntax.definitions)
        .expect("project sort wrapper and comparator should link");
    let run = module.procedure_id("/proc/run").expect("run procedure");
    assert!(
            module
                .procedure(run)
                .expect("run program")
                .instructions
                .iter()
                .all(|instruction| !matches!(instruction, Instruction::StandardBuiltin { name, .. } if name == "sort_list")),
            "a declared project sort_list must not be replaced by a native builtin",
        );
    let mut state = ExecutionState::new();
    let Value::List(sorted) = execute_module_in_state(&module, run, &[], &mut state)
        .expect("project comparator should execute")
    else {
        panic!("project sort wrapper should return a list")
    };
    assert_eq!(
        state
            .heap()
            .list(sorted)
            .expect("sorted list")
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>(),
        vec![Value::number(3.0), Value::number(1.0)],
    );
}

#[test]
fn typecacheof_builtin_returns_descendant_type_map() {
    let syntax =
        parse("/proc/test()\n\treturn typecacheof(/datum)\n").expect("source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("typecacheof call should compile");
    assert!(
            program
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instruction::StandardBuiltin { name, .. } if name == "typecacheof"))
        );

    let mut state = ExecutionState::new();
    let base = TypePath::parse("/datum").expect("type path");
    let child = TypePath::parse("/datum/child").expect("type path");
    let grandchild = TypePath::parse("/datum/child/grandchild").expect("type path");
    state.set_type_paths([base.clone(), child.clone(), grandchild.clone()]);

    let result = execute_in_state(&program, &[], &mut state).expect("typecacheof should execute");
    let Value::List(cache) = result else {
        panic!("typecacheof should return a list");
    };
    let cache = state
        .heap()
        .list(cache)
        .expect("type cache list should exist");
    assert_eq!(
        cache.get_key(&Value::TypePath(base)),
        Ok(&Value::number(1.0))
    );
    assert_eq!(
        cache.get_key(&Value::TypePath(child)),
        Ok(&Value::number(1.0))
    );
    assert_eq!(
        cache.get_key(&Value::TypePath(grandchild)),
        Ok(&Value::number(1.0))
    );
}

#[test]
fn typecacheof_unions_a_list_of_base_paths() {
    let syntax = parse("/proc/test()\n\treturn typecacheof(list(null, /datum/one, /obj/item))\n")
        .expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("typecache list compiles");
    let mut state = ExecutionState::new();
    let one = TypePath::parse("/datum/one").unwrap();
    let one_child = TypePath::parse("/datum/one/child").unwrap();
    let item = TypePath::parse("/obj/item").unwrap();
    let unrelated = TypePath::parse("/mob").unwrap();
    state.set_type_paths([
        one.clone(),
        one_child.clone(),
        item.clone(),
        unrelated.clone(),
    ]);
    let Value::List(cache) = execute_in_state(&program, &[], &mut state).unwrap() else {
        panic!("typecacheof should return a list");
    };
    let cache = state.heap().list(cache).unwrap();
    for included in [one, one_child, item] {
        assert_eq!(
            cache.get_key(&Value::TypePath(included)),
            Ok(&Value::number(1.0))
        );
    }
    assert!(cache.get_key(&Value::TypePath(unrelated)).is_err());
}

#[test]
fn min_and_max_accept_variadic_values_and_single_lists() {
    let syntax = parse("/proc/test()\n\treturn min(8, 3, 5) + max(list(2, 9, 4))\n")
        .expect("source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("extrema compile");
    assert_eq!(execute(&program, &[]), Ok(Value::number(12.0)));
}

#[test]
fn image_builtin_constructs_image_datum_with_icon_fields() {
    let syntax = parse("/proc/build()\n\treturn image(null, null, \"state\", 4, 2)\n")
        .expect("source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("image constructor should compile");
    assert!(
            program
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instruction::StandardBuiltin { name, .. } if name == "image"))
        );
    let mut state = ExecutionState::new();
    state.set_initial_values(BTreeMap::from([(
        TypePath::parse("/image").unwrap(),
        BTreeMap::from([(field("filter_data"), Value::Null)]),
    )]));
    let result =
        execute_in_state(&program, &[], &mut state).expect("image constructor should execute");
    let Value::Datum(image) = result else {
        panic!("image should return a datum");
    };
    let datum = state.heap().datum(image).expect("image datum should exist");
    assert_eq!(datum.type_path(), &TypePath::parse("/image").unwrap());
    assert_eq!(datum.field(&field("icon")), Ok(&Value::Null));
    assert_eq!(datum.field(&field("loc")), Ok(&Value::Null));
    assert_eq!(datum.field(&field("icon_state")), Ok(&Value::text("state")));
    assert_eq!(datum.field(&field("layer")), Ok(&Value::number(4.0)));
    assert_eq!(datum.field(&field("dir")), Ok(&Value::number(2.0)));
    assert_eq!(datum.field(&field("alpha")), Ok(&Value::number(255.0)));
    assert_eq!(datum.field(&field("filter_data")), Ok(&Value::Null));
    assert_eq!(
        datum.field(&field("appearance_flags")),
        Ok(&Value::number(0.0))
    );
    let Value::List(overlays) = datum.field(&field("overlays")).unwrap() else {
        panic!("image overlays should be a live engine list");
    };
    assert_eq!(state.heap().list(*overlays).unwrap().len(), 0);
}

#[test]
fn sound_builtin_materializes_complete_byond_defaults() {
    let syntax = parse("/proc/build()\n\treturn sound()\n").unwrap();
    let program = compile_procedure(&syntax.definitions[0]).unwrap();
    let mut state = ExecutionState::new();
    let Value::Datum(sound) = execute_in_state(&program, &[], &mut state).unwrap() else {
        panic!("sound() should return a datum");
    };
    let datum = state.heap().datum(sound).unwrap();
    for name in ["file", "repeat", "wait", "channel", "offset"] {
        assert_eq!(datum.field(&field(name)), Ok(&Value::Null), "{name}");
    }
    assert_eq!(datum.field(&field("volume")), Ok(&Value::number(100.0)));
    assert_eq!(datum.field(&field("frequency")), Ok(&Value::number(0.0)));
    assert_eq!(datum.field(&field("pan")), Ok(&Value::number(0.0)));
}

#[test]
fn image_from_layer_image_copies_appearance_before_named_overrides() {
    let syntax = parse(
            "/proc/build_flatten_layer()\n\tvar/image/layer_image = image('icons/source.dmi', \"source\", 4, 8, 3, -2)\n\tlayer_image.alpha = 123\n\tlayer_image.blend_mode = 2\n\tlayer_image.plane = -7\n\tlayer_image.overlays += image('icons/overlay.dmi', \"overlay\")\n\tvar/image/copy = image(layer_image, icon_state = \"override\", layer = 9)\n\treturn list(layer_image, copy)\n",
        )
        .expect("getFlatIcon-shaped image copy source should parse");
    let module = compile_module(&syntax.definitions)
        .expect("getFlatIcon-shaped image copy source should compile");
    let procedure = module
        .procedure_id("/proc/build_flatten_layer")
        .expect("image copy fixture entry should exist");
    let mut state = ExecutionState::new();
    let Value::List(result) = execute_module_in_state(&module, procedure, &[], &mut state)
        .expect("image(layer_image) should execute")
    else {
        panic!("image copy fixture should return both images");
    };
    let values = state.heap().list(result).unwrap();
    let Value::Datum(source) = values.get(1).unwrap() else {
        panic!("first result should be the source image");
    };
    let Value::Datum(copy) = values.get(2).unwrap() else {
        panic!("second result should be the copied image");
    };
    for (name, expected) in [
        ("icon", Value::file("icons/source.dmi")),
        ("icon_state", Value::text("override")),
        ("layer", Value::number(9.0)),
        ("dir", Value::number(8.0)),
        ("pixel_x", Value::number(3.0)),
        ("pixel_y", Value::number(-2.0)),
        ("alpha", Value::number(123.0)),
        ("blend_mode", Value::number(2.0)),
        ("plane", Value::number(-7.0)),
    ] {
        assert_eq!(
            state.heap().datum_field(*copy, &field(name)),
            Ok(&expected),
            "copied image field {name}"
        );
    }
    assert_eq!(
        state.heap().datum_field(*source, &field("icon_state")),
        Ok(&Value::text("source")),
        "explicit copy overrides must not mutate the source"
    );
    let Value::List(source_overlays) = state
        .heap()
        .datum_field(*source, &field("overlays"))
        .unwrap()
    else {
        panic!("source overlays should be a list");
    };
    let Value::List(copy_overlays) = state.heap().datum_field(*copy, &field("overlays")).unwrap()
    else {
        panic!("copied overlays should be a list");
    };
    assert_ne!(
        source_overlays, copy_overlays,
        "appearance collections must have independent list identities"
    );
    assert_eq!(state.heap().list(*source_overlays).unwrap().len(), 1);
    assert_eq!(state.heap().list(*copy_overlays).unwrap().len(), 1);
}

#[test]
fn mutable_appearance_new_copies_decal_image_before_dm_constructor() {
    let syntax = parse(
        "/mutable_appearance/New(mutable_appearance/to_copy)\n\
             \tif(!to_copy)\n\
             \t\tsrc.plane = -32767\n\
             /proc/build_smoothed_decal()\n\
             \tvar/temp_image = image('icons/turf/floors/neon.dmi', null, \"neon-3\", 4, 8)\n\
             \tvar/mutable_appearance/pic = new(temp_image)\n\
             \treturn list(pic.icon, pic.icon_state, pic.layer, pic.dir, pic.plane)\n",
    )
    .expect("decal appearance-copy fixture should parse");
    let module =
        compile_module(&syntax.definitions).expect("decal appearance-copy fixture should compile");
    let procedure = module
        .procedure_id("/proc/build_smoothed_decal")
        .expect("decal appearance-copy fixture should have an entry proc");
    let mut state = ExecutionState::new();
    let Value::List(result) = execute_module_in_state(&module, procedure, &[], &mut state)
        .expect("typed mutable-appearance construction should copy the image")
    else {
        panic!("decal appearance-copy fixture should return a list")
    };
    let values = state
        .heap()
        .list(result)
        .unwrap()
        .positions()
        .map(|(_, value)| value.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        values,
        vec![
            Value::file("icons/turf/floors/neon.dmi"),
            Value::text("neon-3"),
            Value::number(4.0),
            Value::number(8.0),
            Value::number(0.0),
        ],
        "OpenDream/BYOND copy the complete source image before /mutable_appearance/New runs",
    );
}

#[test]
fn image_accepts_mutable_appearance_atom_and_icon_sources() {
    let syntax = parse(
            "/proc/build_sources()\n\tvar/mutable_appearance/mutable = new()\n\tmutable.icon = 'icons/mutable.dmi'\n\tmutable.icon_state = \"mutable\"\n\tmutable.pixel_x = 7\n\tvar/obj/source = new\n\tsource.icon = 'icons/atom.dmi'\n\tsource.icon_state = \"atom\"\n\tvar/icon/icon_source = icon('icons/icon.dmi')\n\treturn list(image(mutable), image(source), image(icon_source))\n",
        )
        .expect("appearance source fixture should parse");
    let module =
        compile_module(&syntax.definitions).expect("appearance source fixture should compile");
    let procedure = module
        .procedure_id("/proc/build_sources")
        .expect("appearance source fixture entry should exist");
    let mut state = ExecutionState::new();
    let Value::List(result) =
        execute_module_in_state(&module, procedure, &[], &mut state).expect("source copies")
    else {
        panic!("source copy fixture should return a list");
    };
    let values = state.heap().list(result).unwrap();
    let images = (1..=3)
        .map(|index| match values.get(index).unwrap() {
            Value::Datum(image) => *image,
            value => panic!("source copy result {index} should be an image, got {value}"),
        })
        .collect::<Vec<_>>();
    for (image, icon, icon_state) in [
        (images[0], "icons/mutable.dmi", "mutable"),
        (images[1], "icons/atom.dmi", "atom"),
        (images[2], "icons/icon.dmi", ""),
    ] {
        assert_eq!(
            state.heap().datum_field(image, &field("icon")),
            Ok(&Value::file(icon))
        );
        if !icon_state.is_empty() {
            assert_eq!(
                state.heap().datum_field(image, &field("icon_state")),
                Ok(&Value::text(icon_state))
            );
        }
    }
    assert_eq!(
        state.heap().datum_field(images[0], &field("pixel_x")),
        Ok(&Value::number(7.0))
    );
}

#[test]
fn typesof_null_is_empty_for_typecache_style_root_lists() {
    let syntax = parse(
            "/proc/build_cache()\n\tvar/list/roots = list(null, /mob)\n\tvar/list/result = list()\n\tfor(var/root in roots)\n\t\tfor(var/path in typesof(root))\n\t\t\tresult += path\n\treturn result\n",
        )
        .expect("typecache-shaped source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("typecache-shaped source compiles");
    let mut state = ExecutionState::new();
    let mob = TypePath::parse("/mob").unwrap();
    let child = TypePath::parse("/mob/living").unwrap();
    state.set_type_paths([mob.clone(), child.clone(), TypePath::parse("/obj").unwrap()]);

    let Value::List(result) = execute_in_state(&program, &[], &mut state)
        .expect("BYOND filters the null typesof selector")
    else {
        panic!("expected expanded type list")
    };
    let values = state
        .heap()
        .list(result)
        .unwrap()
        .positions()
        .map(|(_, value)| value.clone())
        .collect::<Vec<_>>();
    assert_eq!(values, vec![Value::TypePath(mob), Value::TypePath(child)]);
}

#[test]
fn savefile_index_output_and_export_text_support_icon_base64_pipeline() {
    let source = parse(
            "/proc/round_trip()\n\tvar/savefile/cache = new /savefile(\"memory.sav\")\n\tvar/icon/value = icon()\n\tcache[\"dummy\"] << value\n\tvar/exported = cache.ExportText(\"dummy\")\n\tvar/list/partial = splittext(exported, \"{\")\n\treturn list(exported, replacetext(copytext_char(partial[2], 3, -5), \"\\n\", \"\"))\n",
        )
        .expect("savefile round-trip source should parse");
    let module = compile_module(&source.definitions).expect("savefile round-trip compiles");
    let mut state = ExecutionState::new();
    let result = execute_module_in_state(
        &module,
        module.procedure_id("/proc/round_trip").unwrap(),
        &[],
        &mut state,
    )
    .expect("indexed savefile output and ExportText should execute");
    let Value::List(result) = result else {
        panic!("ExportText pipeline should return a list")
    };
    let values = state
        .heap()
        .list(result)
        .unwrap()
        .positions()
        .map(|(_, value)| value.clone())
        .collect::<Vec<_>>();
    let Value::Text(exported) = &values[0] else {
        panic!("ExportText should return text")
    };
    assert!(exported.starts_with("dummy = {\""));
    assert!(exported.contains("ZHJlYW02NA=="));
    assert_eq!(values[1], Value::text("ZHJlYW02NA=="));
}

#[test]
fn savefile_cd_dir_eof_and_input_follow_byond_navigation() {
    let source = parse(
            "/proc/savefile_walk()\n\tvar/savefile/cache = new /savefile(\"memory.sav\")\n\tcache.cd = \"/prefs\"\n\tcache[\"volume\"] << 9\n\tcache.cd = \"volume\"\n\tvar/sequential\n\tcache >> sequential\n\tvar/at_entry = cache.eof\n\tcache.cd = \"/prefs\"\n\tvar/keyed\n\tcache[\"volume\"] >> keyed\n\tvar/list/names = cache.dir\n\tcache.cd = \"/missing\"\n\treturn list(sequential, keyed, at_entry, names.len, cache.eof)\n",
        )
        .expect("savefile navigation source should parse");
    let module = compile_module(&source.definitions).expect("savefile navigation compiles");
    let mut state = ExecutionState::new();
    let Value::List(result) = execute_module_in_state(
        &module,
        module.procedure_id("/proc/savefile_walk").unwrap(),
        &[],
        &mut state,
    )
    .expect("savefile navigation and input should execute") else {
        panic!("expected result list")
    };
    let values = state
        .heap()
        .list(result)
        .unwrap()
        .positions()
        .map(|(_, value)| value.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        values,
        vec![
            Value::number(9.0),
            Value::number(9.0),
            Value::number(0.0),
            Value::number(1.0),
            Value::number(1.0),
        ]
    );
}

#[test]
fn gate44_byond_parser_forms_compile_as_one_language_family() {
    let source = parse(
            "/client/proc/verb_metadata()\n\tset name = \"Example\"\n\tset category = \"Admin\"\n\tset desc = \"Description\"\n\treturn 1\n/proc/inline_bodies(list/items)\n\tvar/total = 0\n\tfor(var/item in items) total += item\n\twhile(total > 10) total--\n\treturn total\n/proc/read_old_save(savefile/file)\n\tvar/value\n\tfile >> value\n\treturn value\n/proc/write_pointer(pointer)\n\t*pointer = 0\n/proc/use_pointer()\n\tvar/x = 4\n\twrite_pointer(&x)\n\treturn x\n/datum/proc/safe_initial(mob/who)\n\treturn initial(who.client?.mouse_override_icon)\n/proc/expanded_min(list/values)\n\treturn min(arglist(values))\n/proc/named_image()\n\treturn image(\"icon\" = 'icon.dmi')\n",
        )
        .expect("Gate44 compatibility source should parse");
    let module =
        compile_module(&source.definitions).expect("Gate44 compatibility forms should compile");
    let mut state = ExecutionState::new();
    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/use_pointer").unwrap(),
            &[],
            &mut state,
        ),
        Ok(Value::number(0.0)),
        "address-of and dereference preserve an output-parameter alias"
    );
}

#[test]
fn else_for_and_post_conditional_while_retain_indented_bodies() {
    let source = parse(
            "/proc/admin_shape(list/flags, exact)\n\tif(exact)\n\t\treturn 1\n\telse for(var/flag in flags)\n\t\tif(!flag)\n\t\t\tcontinue\n\t\t. += flag\n\treturn .\n/proc/map_shape(text, matcher)\n\tif(isfile(text))\n\t\ttext = file2text(text)\n\telse if(isnull(text))\n\t\treturn\n\tvar/list/bounds = list(1, 1, 1)\n\tif(findtext(text, \"tgm\"))\n\t\t. = 1\n\telse\n\t\t. = 2\n\tvar/stored_index = 1\n\tvar/list/regex_output\n\twhile(matcher.Find(text, stored_index))\n\t\tstored_index = matcher.next\n\t\tregex_output = matcher.group\n\treturn stored_index\n",
        )
        .expect("combined control-flow shapes should parse");
    compile_module(&source.definitions)
        .expect("combined else-for and following while bodies should compile");
}

#[test]
fn empty_while_uses_condition_side_effects_without_consuming_next_sibling() {
    let source = parse(
            "/proc/skip_blanks(list/lines)\n\tvar/leading_blanks = 0\n\twhile(leading_blanks < length(lines) && lines[++leading_blanks] == \"\")\n\tif(leading_blanks > 1)\n\t\treturn leading_blanks\n\treturn 0\n",
        )
        .expect("empty while source should parse");
    let module =
        compile_module(&source.definitions).expect("BYOND permits a condition-only empty while");
    let mut state = ExecutionState::new();
    let lines = state.heap_mut().allocate_list();
    for value in ["", "", "occupied"] {
        state
            .heap_mut()
            .list_mut(lines)
            .unwrap()
            .add(Value::text(value));
    }
    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/skip_blanks").unwrap(),
            &[Value::List(lines)],
            &mut state,
        ),
        Ok(Value::number(3.0)),
        "the following if remains a sibling and observes condition mutation"
    );
}

#[test]
fn typesof_builtin_includes_the_selector_and_registered_descendants() {
    let syntax =
        parse("/proc/types()\n\treturn typesof(/datum)\n").expect("typesof source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("typesof builtin should compile");
    assert!(
        program
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::TypesOf { .. }))
    );

    let mut state = ExecutionState::new();
    state.set_type_paths([
        TypePath::parse("/obj").expect("type path"),
        TypePath::parse("/datum/child").expect("type path"),
        TypePath::parse("/datum").expect("type path"),
        TypePath::parse("/datum/child/grandchild").expect("type path"),
    ]);
    let result = execute_in_state(&program, &[], &mut state)
        .expect("typesof should execute against the catalog");
    let Value::List(list) = result else {
        panic!("typesof should return a list");
    };
    let values = state
        .heap()
        .list(list)
        .expect("typesof result list should be live")
        .positions()
        .map(|(_, value)| value.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        values,
        vec![
            Value::TypePath(TypePath::parse("/datum").expect("type path")),
            Value::TypePath(TypePath::parse("/datum/child").expect("type path")),
            Value::TypePath(TypePath::parse("/datum/child/grandchild").expect("type path")),
        ]
    );
}

#[test]
fn typesof_invalid_text_is_empty_like_byond_516() {
    let syntax = parse(
        "/proc/types()\n\treturn typesof(\"Devoted Followers\").len + typesof(\"/datum\").len\n",
    )
    .expect("textual typesof source should parse");
    let program =
        compile_procedure(&syntax.definitions[0]).expect("textual typesof source should compile");
    let mut state = ExecutionState::new();
    state.set_type_paths([
        TypePath::parse("/datum").unwrap(),
        TypePath::parse("/datum/child").unwrap(),
    ]);
    assert_eq!(
        execute_in_state(&program, &[], &mut state),
        Ok(Value::number(2.0))
    );
}

#[test]
fn procedure_typesof_preserves_source_order_for_generated_global_initializers() {
    let syntax = parse(
            "/datum/controller/global_vars/proc/InitGlobalhuds_by_category()\n\treturn\n/datum/controller/global_vars/proc/InitGlobalhuds()\n\treturn\n/proc/catalog()\n\treturn typesof(/datum/controller/global_vars/proc)\n",
        )
        .expect("managed global initializer catalog should parse");
    let module = compile_module(&syntax.definitions)
        .expect("managed global initializer catalog should compile");
    let mut state = ExecutionState::new();
    let result = execute_module_in_state(
        &module,
        module.procedure_id("/proc/catalog").unwrap(),
        &[],
        &mut state,
    )
    .expect("procedure typesof should execute");
    let Value::List(list) = result else {
        panic!("procedure typesof should return a list")
    };
    let values = state
        .heap()
        .list(list)
        .unwrap()
        .positions()
        .map(|(_, value)| value.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        values,
        vec![
            Value::TypePath(
                TypePath::parse("/datum/controller/global_vars/proc/InitGlobalhuds_by_category")
                    .unwrap()
            ),
            Value::TypePath(
                TypePath::parse("/datum/controller/global_vars/proc/InitGlobalhuds").unwrap()
            ),
        ]
    );
}

#[test]
fn reopened_generated_initglobal_remains_a_procedure_type_but_initialize_does_not() {
    let syntax = parse(
            "/datum/controller/global_vars/InitGlobalhuds_by_category()\n\treturn\n/datum/controller/global_vars/Initialize()\n\treturn\n",
        )
        .expect("shorthand controller overrides should parse");
    let initglobal = ProcedureSpec {
        path: "/datum/controller/global_vars/proc/InitGlobalhuds_by_category@2".to_owned(),
        definition: &syntax.definitions[0],
        parent: None,
        static_calls: BTreeMap::new(),
        src_fields: BTreeMap::new(),
        global_fields: BTreeMap::new(),
    };
    let initialize = ProcedureSpec {
        path: "/datum/controller/global_vars/proc/Initialize@3".to_owned(),
        definition: &syntax.definitions[1],
        parent: None,
        static_calls: BTreeMap::new(),
        src_fields: BTreeMap::new(),
        global_fields: BTreeMap::new(),
    };
    assert!(super::procedure_spec_is_type_path(&initglobal));
    assert!(!super::procedure_spec_is_type_path(&initialize));
}

#[test]
fn typesof_accepts_multiple_families_in_argument_order_and_deduplicates() {
    let syntax = parse(concat!(
            "/proc/clock_turfs()\n",
            "\treturn typesof(/turf/open/floor/bronze, /turf/open/indestructible/reebe_flooring, /turf/closed/wall/clockwork, /turf/open/floor/engine/clockwork)\n",
            "/proc/overlap()\n",
            "\treturn typesof(/datum, /datum/child, null)\n",
        ))
        .expect("multi-selector typesof source should parse");
    let module = compile_module(&syntax.definitions)
        .expect("generated managed-global typesof initializer should compile");
    let mut state = ExecutionState::new();
    state.set_type_paths([
        TypePath::parse("/datum").unwrap(),
        TypePath::parse("/datum/child").unwrap(),
        TypePath::parse("/datum/child/grandchild").unwrap(),
        TypePath::parse("/turf/open/floor/bronze").unwrap(),
        TypePath::parse("/turf/open/floor/bronze/gear").unwrap(),
    ]);

    let clock = execute_module_in_state(
        &module,
        module.procedure_id("/proc/clock_turfs").unwrap(),
        &[],
        &mut state,
    )
    .expect("clock turf families should execute");
    let Value::List(clock) = clock else {
        panic!("typesof should return a list");
    };
    let clock = state
        .heap()
        .list(clock)
        .unwrap()
        .positions()
        .map(|(_, value)| value.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        clock,
        [
            "/turf/open/floor/bronze",
            "/turf/open/floor/bronze/gear",
            "/turf/open/indestructible/reebe_flooring",
            "/turf/closed/wall/clockwork",
            "/turf/open/floor/engine/clockwork",
        ]
        .map(|path| Value::TypePath(TypePath::parse(path).unwrap()))
    );

    let overlap = execute_module_in_state(
        &module,
        module.procedure_id("/proc/overlap").unwrap(),
        &[],
        &mut state,
    )
    .expect("overlapping families and null should execute");
    let Value::List(overlap) = overlap else {
        panic!("typesof should return a list");
    };
    assert_eq!(state.heap().list(overlap).unwrap().len(), 3);
}

#[test]
fn typesof_can_use_a_shared_immutable_catalog() {
    let catalog = Arc::new(
        [
            TypePath::parse("/datum").expect("type path"),
            TypePath::parse("/datum/child").expect("type path"),
        ]
        .into_iter()
        .collect(),
    );
    let mut state = ExecutionState::new();
    state.set_shared_type_paths(Arc::clone(&catalog));

    assert_eq!(Arc::strong_count(&catalog), 2);
    assert_eq!(
        state.type_paths().cloned().collect::<Vec<_>>(),
        vec![
            TypePath::parse("/datum").expect("type path"),
            TypePath::parse("/datum/child").expect("type path"),
        ]
    );
}

#[test]
fn special_result_can_receive_resolved_parent_call() {
    let source = "/proc/base(value = 4)\n\t. = value\n/proc/child(value = 4)\n\t. = ..()\n";
    let syntax = parse(source).expect("source should parse");
    let parent_assignment_span = syntax.definitions[1].body[0].span;
    let module = compile_module_specs(&[
        ProcedureSpec {
            path: "/proc/base@0".to_owned(),
            definition: &syntax.definitions[0],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
        ProcedureSpec {
            path: "/proc/child@1".to_owned(),
            definition: &syntax.definitions[1],
            parent: Some(0),
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
    ])
    .expect("resolved parent specs should compile");
    let entry = module
        .procedure_id_at(1)
        .expect("child spec should have a VM identity");

    assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(4.0)));
    let child = module.procedure(entry).expect("child program should exist");
    assert!(child.instructions.iter().zip(&child.source_spans).any(
        |(instruction, span)| matches!(instruction, Instruction::StoreResult)
            && *span == parent_assignment_span
    ));
}

#[test]
fn vector_constructor_operators_and_methods_use_three_numeric_components() {
    let source = concat!(
        "/proc/run()\n",
        "\tvar/vector/a = vector(3, 3)\n",
        "\tvar/vector/b = vector(4, 4, 4)\n",
        "\tvar/vector/c = a + b\n",
        "\ta *= b\n",
        "\tvar/vector/i = vector(1, 1).Interpolate(vector(12, 124, 91), 0.5)\n",
        "\tvar/vector/n = vector(3, 4)\n",
        "\tn.Normalize()\n",
        "\treturn c.x + c.z + a.x + i.x + i.y + i.z + n.size\n",
    );
    let syntax = parse(source).expect("source should parse");
    let module = compile_module(&syntax.definitions).expect("vector source should compile");
    let entry = module.procedure_id("/proc/run").expect("run should exist");

    assert_eq!(
        execute_module(&module, entry, &[]),
        Ok(Value::number(138.5))
    );
}

#[test]
fn animate_applies_named_values_and_continues_the_last_sequence_headlessly() {
    let syntax = parse(
            "/proc/run()\n\tanimate(src, alpha = 128, time = 5)\n\tanimate(pixel_x = 12, time = 2)\n\treturn src.alpha + src.pixel_x\n",
        )
        .expect("animate source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("animate should compile");
    let mut state = ExecutionState::new();
    let object = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj").expect("obj path"));
    let context = ExecutionContext::new(Value::Datum(object), Value::Null);

    assert_eq!(
        execute_in_context(&program, &[], &mut state, &context),
        Ok(Value::number(140.0))
    );
}

#[test]
fn flick_does_not_mutate_the_persistent_icon_state() {
    let syntax = parse("/proc/run()\n\tflick(\"opening\", src)\n\treturn src.icon_state\n")
        .expect("flick source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("flick should compile");
    let mut state = ExecutionState::new();
    let object = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj").expect("obj path"));
    state
        .heap_mut()
        .set_datum_field(object, field("icon_state"), Value::text("closed"))
        .expect("icon state should materialize");
    let context = ExecutionContext::new(Value::Datum(object), Value::Null);

    assert_eq!(
        execute_in_context(&program, &[], &mut state, &context),
        Ok(Value::text("closed"))
    );
}

#[test]
fn filter_preserves_named_properties_on_a_filter_datum() {
    let syntax =
        parse("/proc/run()\n\tvar/f = filter(type = \"blur\", size = 4)\n\treturn f.size\n")
            .expect("filter source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("filter should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::number(4.0)));
}

#[test]
fn filter_datum_inherits_engine_datum_storage() {
    let syntax = parse("/proc/run()\n\tvar/f = filter(type = \"blur\")\n\treturn f.datum_flags\n")
        .expect("filter source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("filter should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::number(0.0)));
}

#[test]
fn filter_arglist_spreads_associative_entries_as_named_properties() {
    let syntax = parse(
            "/proc/run()\n\tvar/list/arguments = list(\"type\" = \"blur\", \"size\" = 7)\n\tvar/f = filter(arglist(arguments))\n\treturn f.size\n",
        )
        .expect("filter arglist source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("filter arglist should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::number(7.0)));
}

#[test]
fn deeply_nested_macro_ternaries_remain_one_call_argument() {
    let source = "/proc/run(a, b, t, m, p)\n\treturn helper(((a) ? (b?[\"[-9]\"] ? -9 : (-9) - ((110 - -100) * (((a && (isloc(t)))) ? (t.z ? (m[t.z]) : ((p) ? p[\"[t.plane]\"] : t.plane)) : 0))) : (-9)), 1)\n/proc/helper(value, other)\n\treturn value\n";
    let syntax = parse(source).expect("nested conditional source should parse");
    compile_module(&syntax.definitions)
        .expect("nested conditional must not consume the call delimiter");
}

#[test]
fn safe_index_does_not_expose_named_call_assignment_as_statement_assignment() {
    let source = "/proc/run(mapping, flags)\n\thelper(mapping?[\"key\"], add_appearance_flags = flags)\n\treturn 1\n/proc/helper(value, add_appearance_flags)\n\treturn value\n";
    let syntax = parse(source).expect("safe-index call source should parse");
    compile_module(&syntax.definitions)
        .expect("named argument inside a safe-index call must remain nested");
}

#[test]
fn output_preserves_message_and_control_for_later_client_routing() {
    let syntax = parse(
        "/proc/run()\n\tvar/o = output(\"score: 5\", \"scorepane.output\")\n\treturn o.control\n",
    )
    .expect("output source should parse");
    let program = compile_procedure(&syntax.definitions[0]).expect("output should compile");
    assert_eq!(execute(&program, &[]), Ok(Value::text("scorepane.output")));
}

#[test]
fn local_client_movement_commits_at_boundary_and_updates_snapshot() {
    let mut state = ExecutionState::new();
    let west = state
        .heap
        .allocate_datum(TypePath::parse("/turf/open/floor/west").unwrap());
    let east = state
        .heap
        .allocate_datum(TypePath::parse("/turf/open/floor/east").unwrap());
    for (turf, x) in [(west, 1), (east, 2)] {
        for (name, value) in [("x", x), ("y", 1), ("z", 1)] {
            state
                .heap
                .set_datum_field(turf, field(name), Value::number(value as f32))
                .unwrap();
        }
        state.ensure_contents(turf).unwrap();
        state.world_turfs.insert((x, 1, 1), turf);
    }
    let attached = state.create_attached_local_client().unwrap();
    let (client, mob) = (attached.client, attached.mob);
    assert_eq!((attached.x, attached.y, attached.z), (1, 1, 1));
    for (name, value) in [
        ("icon", Value::file("icons/mob/player.dmi")),
        ("icon_state", Value::text("idle")),
        ("dir", Value::number(4.0)),
        ("layer", Value::number(4.0)),
        ("plane", Value::number(1.0)),
        ("pixel_x", Value::number(3.0)),
        ("alpha", Value::number(200.0)),
    ] {
        state.heap.set_datum_field(mob, field(name), value).unwrap();
    }
    let overlay = state
        .heap
        .allocate_datum(TypePath::parse("/mutable_appearance").unwrap());
    for (name, value) in [
        ("icon", Value::file("icons/effects/aura.dmi")),
        ("icon_state", Value::text("glow")),
        ("layer", Value::number(5.0)),
    ] {
        state
            .heap
            .set_datum_field(overlay, field(name), value)
            .unwrap();
    }
    let overlays = state.heap.allocate_list();
    state
        .heap
        .list_mut(overlays)
        .unwrap()
        .add(Value::Datum(overlay));
    state
        .heap
        .set_datum_field(mob, field("overlays"), Value::List(overlays))
        .unwrap();
    let before = state.local_client_map_snapshot(1);
    assert!(
        before
            .tiles
            .iter()
            .find(|tile| tile.x == 1)
            .unwrap()
            .occupants
            .contains(&mob)
    );
    let mob_appearance = before
        .tiles
        .iter()
        .find(|tile| tile.x == 1)
        .unwrap()
        .appearances
        .iter()
        .find(|appearance| appearance.datum == mob)
        .unwrap();
    assert_eq!(mob_appearance.icon.as_deref(), Some("icons/mob/player.dmi"));
    assert_eq!(mob_appearance.icon_state.as_deref(), Some("idle"));
    assert_eq!(
        (
            mob_appearance.dir,
            mob_appearance.layer,
            mob_appearance.plane
        ),
        (4, 4.0, 1.0)
    );
    assert_eq!((mob_appearance.pixel_x, mob_appearance.alpha), (3.0, 200.0));
    assert_eq!(
        mob_appearance.overlays[0].icon_state.as_deref(),
        Some("glow")
    );

    state
        .queue_local_movement(client, crate::LocalMovementDirection::East)
        .unwrap();
    assert_eq!(
        state.local_client_state(client).unwrap().x,
        1,
        "queued input must not mutate before boundary"
    );
    assert_eq!(state.apply_local_client_commands().unwrap()[0].x, 2);

    let after = state.local_client_map_snapshot(1);
    assert!(
        !after
            .tiles
            .iter()
            .find(|tile| tile.x == 1)
            .unwrap()
            .occupants
            .contains(&mob)
    );
    assert!(
        after
            .tiles
            .iter()
            .find(|tile| tile.x == 2)
            .unwrap()
            .occupants
            .contains(&mob)
    );
    assert_eq!((after.width, after.height, after.z), (2, 1, 1));
}

#[test]
fn named_top_screen_axis_is_not_misread_as_a_map_control() {
    let mut state = ExecutionState::new();
    let turf = state.heap.allocate_datum(TypePath::parse("/turf").unwrap());
    for name in ["x", "y", "z"] {
        state
            .heap
            .set_datum_field(turf, field(name), Value::number(1.0))
            .unwrap();
    }
    state.ensure_contents(turf).unwrap();
    state.world_turfs.insert((1, 1, 1), turf);
    let attached = state.create_attached_local_client().unwrap();
    let button = state
        .heap
        .allocate_datum(TypePath::parse("/atom/movable/screen/lobby/button").unwrap());
    state
        .heap
        .set_datum_field(
            button,
            field("screen_loc"),
            Value::text("TOP:-87,CENTER:+100"),
        )
        .unwrap();
    let Value::List(screen) = state
        .heap
        .datum_field(attached.client, &field("screen"))
        .unwrap()
        .clone()
    else {
        panic!("attached client screen must be a list");
    };
    state
        .heap
        .list_mut(screen)
        .unwrap()
        .add(Value::Datum(button));

    let snapshot = state.local_client_map_snapshot_for(Some(attached.client), 1);
    assert_eq!(snapshot.screen.len(), 1);
    assert_eq!(snapshot.screen[0].map_control, None);
    assert_eq!(snapshot.screen[0].screen_loc, "TOP:-87,CENTER:+100");
}

#[test]
fn local_client_camera_prefers_eye_over_mob_location() {
    let mut state = ExecutionState::new();
    for (x, y) in [(1, 1), (17, 23)] {
        let turf = state.heap.allocate_datum(TypePath::parse("/turf").unwrap());
        state.ensure_contents(turf).unwrap();
        state.world_turfs.insert((x, y, 1), turf);
    }
    let attached = state.create_attached_local_client().unwrap();
    assert_eq!(
        state
            .local_client_view_coordinates(attached.client)
            .unwrap(),
        (1, 1, 1)
    );
    let eye = state.world_turfs[&(17, 23, 1)];
    state
        .heap
        .set_datum_field(attached.client, field("eye"), Value::Datum(eye))
        .unwrap();
    assert_eq!(
        state
            .local_client_view_coordinates(attached.client)
            .unwrap(),
        (17, 23, 1)
    );
}

#[test]
fn local_screen_pointer_validates_membership_generation_and_runs_with_usr() {
    let syntax = parse(
            "var/global/clicked\nvar/global/click_usr\nvar/global/click_location\nvar/global/click_params\n/obj/screen_button/proc/Click(location, params)\n\tclicked = src\n\tclick_usr = usr\n\tclick_location = location\n\tclick_params = params\n",
        )
        .expect("screen pointer fixture should parse");
    let procedures = syntax
        .definitions
        .iter()
        .filter(|definition| matches!(definition.kind, DefinitionKind::Procedure))
        .cloned()
        .collect::<Vec<_>>();
    let globals = ["clicked", "click_usr", "click_location", "click_params"]
        .into_iter()
        .map(|name| (name.to_owned(), field(name)))
        .collect();
    let module = compile_module_with_global_fields(&procedures, &globals)
        .expect("screen pointer fixture should compile");
    let mut state = ExecutionState::new();
    for name in ["clicked", "click_usr", "click_location", "click_params"] {
        state.set_global(field(name), Value::Null);
    }
    let turf = state
        .heap
        .allocate_datum(TypePath::parse("/turf/open").unwrap());
    for name in ["x", "y", "z"] {
        state
            .heap
            .set_datum_field(turf, field(name), Value::number(1.0))
            .unwrap();
    }
    state.ensure_contents(turf).unwrap();
    state.world_turfs.insert((1, 1, 1), turf);
    let attached = state.create_attached_local_client().unwrap();
    let owned = state
        .heap
        .allocate_datum(TypePath::parse("/obj/screen_button").unwrap());
    let foreign = state
        .heap
        .allocate_datum(TypePath::parse("/obj/screen_button").unwrap());
    let Value::List(screen) = state
        .heap
        .datum_field(attached.client, &field("screen"))
        .unwrap()
        .clone()
    else {
        panic!("attached client screen must be a list");
    };
    state
        .heap
        .list_mut(screen)
        .unwrap()
        .add(Value::Datum(owned));

    assert!(
        state
            .queue_local_screen_pointer(
                &module,
                attached.client,
                foreign.index(),
                foreign.generation(),
                crate::LocalScreenPointerEvent::Click,
                "main.map",
                "screen-loc=1,1",
            )
            .unwrap_err()
            .contains("not owned")
    );
    assert!(
        state
            .queue_local_screen_pointer(
                &module,
                attached.client,
                owned.index(),
                owned.generation().wrapping_add(1),
                crate::LocalScreenPointerEvent::Click,
                "main.map",
                "screen-loc=1,1",
            )
            .unwrap_err()
            .contains("stale")
    );
    state
        .queue_local_screen_pointer(
            &module,
            attached.client,
            owned.index(),
            owned.generation(),
            crate::LocalScreenPointerEvent::Click,
            "main.map",
            "screen-loc=1,1",
        )
        .unwrap();
    advance_scheduler(&module, 0, ExecutionLimits::default(), &mut state).unwrap();
    assert_eq!(state.global(&field("clicked")), Some(&Value::Datum(owned)));
    assert_eq!(
        state.global(&field("click_usr")),
        Some(&Value::Datum(attached.mob))
    );
    assert_eq!(
        state.global(&field("click_location")),
        Some(&Value::text("main.map"))
    );
    assert_eq!(
        state.global(&field("click_params")),
        Some(&Value::text("screen-loc=1,1"))
    );
}

#[test]
fn local_map_pointer_validates_cell_and_routes_through_client_click() {
    let syntax = parse(
            "var/global/clicked\nvar/global/click_usr\nvar/global/click_control\nvar/global/click_params\n/client/proc/Click(object, location, control, params)\n\tclicked = object\n\tclick_usr = usr\n\tclick_control = control\n\tclick_params = params\n",
        )
        .expect("map pointer fixture should parse");
    let procedures = syntax
        .definitions
        .iter()
        .filter(|definition| matches!(definition.kind, DefinitionKind::Procedure))
        .cloned()
        .collect::<Vec<_>>();
    let globals = ["clicked", "click_usr", "click_control", "click_params"]
        .into_iter()
        .map(|name| (name.to_owned(), field(name)))
        .collect();
    let module = compile_module_with_global_fields(&procedures, &globals)
        .expect("map pointer fixture should compile");
    let mut state = ExecutionState::new();
    for name in ["clicked", "click_usr", "click_control", "click_params"] {
        state.set_global(field(name), Value::Null);
    }
    let turf = state
        .heap
        .allocate_datum(TypePath::parse("/turf/open").unwrap());
    for (name, value) in [("x", 4), ("y", 6), ("z", 1)] {
        state
            .heap
            .set_datum_field(turf, field(name), Value::number(value as f32))
            .unwrap();
    }
    state.ensure_contents(turf).unwrap();
    state.world_turfs.insert((4, 6, 1), turf);
    let target = state
        .heap
        .allocate_datum(TypePath::parse("/obj/item").unwrap());
    state
        .heap
        .set_datum_field(target, field("loc"), Value::Datum(turf))
        .unwrap();
    let attached = state.create_attached_local_client().unwrap();
    assert!(
        state
            .queue_local_map_pointer(
                &module,
                attached.client,
                target.index(),
                target.generation(),
                5,
                6,
                1,
                "main.map",
                "left=1",
            )
            .unwrap_err()
            .contains("outside")
    );
    state
        .queue_local_map_pointer(
            &module,
            attached.client,
            target.index(),
            target.generation(),
            4,
            6,
            1,
            "main.map",
            "left=1",
        )
        .unwrap();
    advance_scheduler(&module, 0, ExecutionLimits::default(), &mut state).unwrap();
    assert_eq!(state.global(&field("clicked")), Some(&Value::Datum(target)));
    assert_eq!(
        state.global(&field("click_usr")),
        Some(&Value::Datum(attached.mob))
    );
    assert_eq!(
        state.global(&field("click_control")),
        Some(&Value::text("main.map"))
    );
    assert_eq!(
        state.global(&field("click_params")),
        Some(&Value::text("left=1"))
    );
}

#[test]
fn local_guest_new_observes_bound_identity_and_resumes_after_sleep() {
    let syntax = parse(
            "var/global/seen_key\nvar/global/seen_ckey\nvar/global/seen_connection\nvar/global/seen_version\nvar/global/seen_build\nvar/global/seen_windows\nvar/global/seen_screen\nvar/global/seen_fps\nvar/global/seen_view\nvar/global/seen_mob\nvar/global/seen_hud\nvar/global/mob_before_parent\nvar/global/persistent_seen_login\nvar/global/login_count = 0\nvar/global/new_stage = 0\n/client/proc/New()\n\tmob_before_parent = src.mob\n\tsrc.persistent_client = new /datum/persistent_client_fixture\n\t..()\n\tseen_key = src.key\n\tseen_ckey = src.ckey\n\tseen_connection = src.connection\n\tseen_version = src.byond_version\n\tseen_build = src.byond_build\n\tseen_windows = islist(src.tgui_windows)\n\tseen_screen = islist(src.screen)\n\tseen_fps = src.fps\n\tseen_view = src.view\n\tseen_mob = src.mob\n\tseen_hud = src.mob.hud_used\n\tnew_stage = 1\n\tsleep(1)\n\tnew_stage = 2\n/mob/dead/new_player/proc/Login()\n\tpersistent_seen_login = !!src.client.persistent_client\n\tsrc.hud_used = \"new-player-hud\"\n\tlogin_count += 1\n",
        )
        .expect("client New fixture should parse");
    let procedures = syntax
        .definitions
        .iter()
        .filter(|definition| matches!(definition.kind, DefinitionKind::Procedure))
        .cloned()
        .collect::<Vec<_>>();
    let globals = [
        "seen_key",
        "seen_ckey",
        "seen_connection",
        "seen_version",
        "seen_build",
        "seen_windows",
        "seen_screen",
        "seen_fps",
        "seen_view",
        "seen_mob",
        "seen_hud",
        "mob_before_parent",
        "persistent_seen_login",
        "login_count",
        "new_stage",
    ]
    .into_iter()
    .map(|name| (name.to_owned(), field(name)))
    .collect();
    let module = compile_module_with_global_fields(&procedures, &globals)
        .expect("client New fixture should compile");
    let mut state = ExecutionState::new();
    let initializer_syntax = parse("/proc/client_windows()\n\treturn list()\n")
        .expect("client list initializer should parse");
    let initializer_module = Arc::new(
        compile_module(&initializer_syntax.definitions)
            .expect("client list initializer should compile"),
    );
    let initializer_entry = initializer_module
        .procedure_id("/proc/client_windows")
        .expect("client initializer entry exists");
    let client_path = TypePath::parse("/client").unwrap();
    state.set_initial_values(BTreeMap::from([(
        client_path.clone(),
        BTreeMap::from([(field("tgui_windows"), Value::Null)]),
    )]));
    state.set_instance_initializers(
        Arc::new(BTreeMap::from([(
            client_path,
            vec![InstanceInitializer::Program {
                field: field("tgui_windows"),
                entry: initializer_entry,
            }],
        )])),
        Some(initializer_module),
    );
    for name in [
        "seen_key",
        "seen_ckey",
        "seen_connection",
        "seen_version",
        "seen_build",
        "seen_windows",
        "seen_screen",
        "seen_fps",
        "seen_view",
        "seen_mob",
        "seen_hud",
        "mob_before_parent",
        "persistent_seen_login",
        "login_count",
        "new_stage",
    ] {
        state.set_global(field(name), Value::Null);
    }
    let world = state
        .heap
        .allocate_datum(TypePath::parse("/world").unwrap());
    let connection_mob = TypePath::parse("/mob/dead/new_player").unwrap();
    state
        .heap
        .set_datum_field(world, field("mob"), Value::TypePath(connection_mob.clone()))
        .unwrap();
    state.set_type_parents(BTreeMap::from([
        (TypePath::parse("/datum").unwrap(), None),
        (
            TypePath::parse("/mob").unwrap(),
            Some(TypePath::parse("/datum").unwrap()),
        ),
        (
            TypePath::parse("/mob/dead").unwrap(),
            Some(TypePath::parse("/mob").unwrap()),
        ),
        (
            connection_mob.clone(),
            Some(TypePath::parse("/mob/dead").unwrap()),
        ),
        (
            TypePath::parse("/client").unwrap(),
            Some(TypePath::parse("/datum").unwrap()),
        ),
    ]));
    let turf = state
        .heap
        .allocate_datum(TypePath::parse("/turf/open/floor").unwrap());
    for (name, value) in [("x", 1), ("y", 1), ("z", 1)] {
        state
            .heap
            .set_datum_field(turf, field(name), Value::number(value as f32))
            .unwrap();
    }
    state.ensure_contents(turf).unwrap();
    state.world_turfs.insert((1, 1, 1), turf);

    let attached = state.connect_local_guest(&module).unwrap();
    assert_eq!(state.scheduled_task_count(), 1);
    assert_eq!(state.global(&field("new_stage")), Some(&Value::Null));
    advance_scheduler(&module, 0, ExecutionLimits::default(), &mut state).unwrap();
    assert_eq!(
        state.global(&field("seen_key")),
        Some(&Value::text("Guest-1"))
    );
    assert_eq!(
        state.global(&field("mob_before_parent")),
        Some(&Value::Null),
        "initial client/New must not observe the reserved connection mob"
    );
    assert_eq!(
        state.global(&field("persistent_seen_login")),
        Some(&Value::number(1.0)),
        "the client body must initialize persistent state before mob/Login"
    );
    assert_eq!(
        state.global(&field("seen_ckey")),
        Some(&Value::text("guest-1"))
    );
    assert_eq!(
        state.global(&field("seen_connection")),
        Some(&Value::text("seeker"))
    );
    assert_eq!(
        state.global(&field("seen_version")),
        Some(&Value::number(516.0))
    );
    assert_eq!(
        state.global(&field("seen_build")),
        Some(&Value::number(1680.0))
    );
    assert_eq!(
        state.global(&field("seen_windows")),
        Some(&Value::number(1.0))
    );
    assert_eq!(
        state.global(&field("seen_screen")),
        Some(&Value::number(1.0))
    );
    assert_eq!(state.global(&field("seen_fps")), Some(&Value::number(10.0)));
    assert_eq!(state.global(&field("seen_view")), Some(&Value::number(5.0)));
    assert_eq!(
        state.global(&field("login_count")),
        Some(&Value::number(1.0))
    );
    assert_eq!(
        state.global(&field("seen_mob")),
        Some(&Value::Datum(attached.mob))
    );
    assert_eq!(
        state.heap().datum(attached.mob).unwrap().type_path(),
        &connection_mob
    );
    assert_eq!(
        state.global(&field("seen_hud")),
        Some(&Value::text("new-player-hud"))
    );
    assert_eq!(state.global(&field("new_stage")), Some(&Value::number(1.0)));
    assert_eq!(state.scheduled_task_count(), 1);
    advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state).unwrap();
    assert_eq!(state.global(&field("new_stage")), Some(&Value::number(2.0)));
}

#[test]
fn connection_mob_constructs_with_client_before_login() {
    let syntax = parse(
            "var/global/initialize_client\nvar/global/splash\nvar/global/login_saw_splash\n/client/proc/New()\n\t..()\n/atom/proc/New()\n\tsrc.Initialize(FALSE)\n/atom/proc/Initialize(mapload)\n\treturn\n/mob/dead/new_player/proc/Initialize(mapload)\n\tinitialize_client = src.client\n\tvar/obj/splash/button = new\n\tsrc.client.screen += button\n\tsplash = button\n/mob/dead/new_player/proc/Login()\n\tlogin_saw_splash = length(src.client.screen)\n",
        )
        .expect("connection mob construction fixture should parse");
    let procedures = syntax
        .definitions
        .iter()
        .filter(|definition| matches!(definition.kind, DefinitionKind::Procedure))
        .cloned()
        .collect::<Vec<_>>();
    let globals = ["initialize_client", "splash", "login_saw_splash"]
        .into_iter()
        .map(|name| (name.to_owned(), field(name)))
        .collect();
    let module = compile_module_with_global_fields(&procedures, &globals)
        .expect("connection mob construction fixture should compile");
    let mut state = ExecutionState::new();
    for name in ["initialize_client", "splash", "login_saw_splash"] {
        state.set_global(field(name), Value::Null);
    }
    let world = state
        .heap
        .allocate_datum(TypePath::parse("/world").unwrap());
    let mob_type = TypePath::parse("/mob/dead/new_player").unwrap();
    state
        .heap
        .set_datum_field(world, field("mob"), Value::TypePath(mob_type.clone()))
        .unwrap();
    state.set_type_parents(BTreeMap::from([
        (TypePath::parse("/datum").unwrap(), None),
        (
            TypePath::parse("/atom").unwrap(),
            Some(TypePath::parse("/datum").unwrap()),
        ),
        (
            TypePath::parse("/mob").unwrap(),
            Some(TypePath::parse("/atom").unwrap()),
        ),
        (
            TypePath::parse("/mob/dead").unwrap(),
            Some(TypePath::parse("/mob").unwrap()),
        ),
        (
            mob_type.clone(),
            Some(TypePath::parse("/mob/dead").unwrap()),
        ),
        (
            TypePath::parse("/client").unwrap(),
            Some(TypePath::parse("/datum").unwrap()),
        ),
        (
            TypePath::parse("/obj").unwrap(),
            Some(TypePath::parse("/atom").unwrap()),
        ),
        (
            TypePath::parse("/obj/splash").unwrap(),
            Some(TypePath::parse("/obj").unwrap()),
        ),
    ]));
    let turf = state.heap.allocate_datum(TypePath::parse("/turf").unwrap());
    for (name, value) in [("x", 1), ("y", 1), ("z", 1)] {
        state
            .heap
            .set_datum_field(turf, field(name), Value::number(value as f32))
            .unwrap();
    }
    state.ensure_contents(turf).unwrap();
    state.world_turfs.insert((1, 1, 1), turf);

    let attached = state.connect_local_guest(&module).unwrap();
    advance_scheduler(&module, 0, ExecutionLimits::default(), &mut state).unwrap();
    assert_eq!(
        state.global(&field("initialize_client")),
        Some(&Value::Datum(attached.client)),
        "Initialize must observe the reciprocal client binding"
    );
    assert_eq!(
        state.global(&field("login_saw_splash")),
        Some(&Value::number(1.0)),
        "Login must run only after New/Initialize populated client.screen"
    );
    assert_eq!(
        state.scheduled_task_count(),
        0,
        "construction runs exactly once"
    );
}

#[test]
fn local_browser_topic_dispatches_href_list_and_resolved_hsrc() {
    let syntax = parse(
            "var/global/seen_href\nvar/global/seen_action\nvar/global/seen_hsrc\nvar/global/seen_usr\nvar/global/seen_src\n/client/proc/Topic(href, href_list, hsrc, hsrc_command)\n\tseen_href = href\n\tseen_action = href_list[\"action\"]\n\tseen_hsrc = hsrc\n\tseen_usr = usr\n\tseen_src = src\n",
        )
        .expect("browser Topic fixture should parse");
    let globals = [
        "seen_href",
        "seen_action",
        "seen_hsrc",
        "seen_usr",
        "seen_src",
    ]
    .into_iter()
    .map(|name| (name.to_owned(), field(name)))
    .collect();
    let procedures = syntax
        .definitions
        .iter()
        .filter(|definition| matches!(definition.kind, DefinitionKind::Procedure))
        .cloned()
        .collect::<Vec<_>>();
    let module = compile_module_with_global_fields(&procedures, &globals)
        .expect("browser Topic fixture should compile");
    let mut state = ExecutionState::new();
    for name in [
        "seen_href",
        "seen_action",
        "seen_hsrc",
        "seen_usr",
        "seen_src",
    ] {
        state.set_global(field(name), Value::Null);
    }
    let client = state
        .heap
        .allocate_datum(TypePath::parse("/client").unwrap());
    let mob = state.heap.allocate_datum(TypePath::parse("/mob").unwrap());
    let source = state
        .heap
        .allocate_datum(TypePath::parse("/datum/source").unwrap());
    state.client.attach_mob(client, mob);
    let topic = format!("byond://?src=[0xd{:06x}]&action=ready", source.index() + 1);

    state
        .queue_local_browser_topic(&module, client, &topic)
        .unwrap();
    advance_scheduler(&module, 0, ExecutionLimits::default(), &mut state).unwrap();

    assert_eq!(state.global(&field("seen_href")), Some(&Value::text(topic)));
    assert_eq!(
        state.global(&field("seen_action")),
        Some(&Value::text("ready"))
    );
    assert_eq!(
        state.global(&field("seen_hsrc")),
        Some(&Value::Datum(source))
    );
    assert_eq!(state.global(&field("seen_usr")), Some(&Value::Datum(mob)));
    assert_eq!(
        state.global(&field("seen_src")),
        Some(&Value::Datum(client))
    );
}

#[test]
fn local_client_command_resolves_normalized_verb_and_quoted_argument() {
    let syntax = parse(
            "var/global/seen_command\nvar/global/seen_number\nvar/global/seen_target\n/client/verb/fix_tgui_panel(message as text)\n\tset name = \"Fix chat\"\n\tseen_command = message\n/client/verb/update_ping(time as num)\n\tseen_number = time\n/client/verb/inspect_target(target as obj)\n\tset name = \"Inspect\"\n\tseen_target = target\n",
        )
        .expect("client command fixture should parse");
    let globals = [
        ("seen_command".to_owned(), field("seen_command")),
        ("seen_number".to_owned(), field("seen_number")),
        ("seen_target".to_owned(), field("seen_target")),
    ]
    .into_iter()
    .collect();
    let procedures = syntax
        .definitions
        .iter()
        .filter(|definition| {
            matches!(
                definition.kind,
                DefinitionKind::Procedure | DefinitionKind::Verb
            )
        })
        .cloned()
        .collect::<Vec<_>>();
    let module = compile_module_with_global_fields(&procedures, &globals)
        .expect("client command fixture should compile");
    let mut state = ExecutionState::new();
    state.set_global(field("seen_command"), Value::Null);
    state.set_global(field("seen_number"), Value::Null);
    state.set_global(field("seen_target"), Value::Null);
    let client = state
        .heap
        .allocate_datum(TypePath::parse("/client").unwrap());
    let mob = state.heap.allocate_datum(TypePath::parse("/mob").unwrap());
    let target = state
        .heap
        .allocate_datum(TypePath::parse("/obj/item").unwrap());
    state.client.attach_mob(client, mob);
    state.install_client_session(client, ControlTree::default());
    state.set_local_client_interactive(client, true).unwrap();
    state
        .populate_local_verb_inventory(&module, client)
        .unwrap();
    state.populate_local_verb_inventory(&module, mob).unwrap();

    state
        .queue_local_client_command(&module, client, "fix-chat \"lobby now\"")
        .unwrap();
    state
        .queue_local_client_command(&module, client, "update-ping 12.5")
        .unwrap();
    assert!(
        state
            .queue_local_client_command(&module, client, "update-ping nope")
            .unwrap_err()
            .contains("invalid number")
    );
    state
        .queue_local_client_command(&module, client, "inspect")
        .unwrap();
    let events = state.take_local_client_outbound_events(client);
    let [LocalClientUiEvent::Prompt { kind, choices, .. }] = events.as_slice() else {
        panic!("unsupported verb target must open one picker")
    };
    assert_eq!(*kind, LocalClientPromptKind::List);
    assert_eq!(choices.len(), 1);
    state
        .submit_local_prompt_response(client, 1, LocalClientPromptResponse::Choice(0))
        .unwrap();
    advance_scheduler(&module, 0, ExecutionLimits::default(), &mut state).unwrap();

    assert_eq!(
        state.global(&field("seen_command")),
        Some(&Value::text("lobby now"))
    );
    assert_eq!(
        state.global(&field("seen_number")),
        Some(&Value::number(12.5))
    );
    assert_eq!(
        state.global(&field("seen_target")),
        Some(&Value::Datum(target))
    );

    let Value::List(verbs) = state
        .heap
        .datum_field(client, &field("verbs"))
        .unwrap()
        .clone()
    else {
        panic!("client verbs must be materialized");
    };
    state
        .heap
        .list_mut(verbs)
        .unwrap()
        .remove_first(&Value::TypePath(
            TypePath::parse("/client/verb/fix_tgui_panel").unwrap(),
        ));
    assert!(
        state
            .queue_local_client_command(&module, client, "fix-chat \"blocked\"")
            .unwrap_err()
            .contains("unknown client command")
    );
}

#[test]
fn live_client_attachment_follows_client_mob_reassignment() {
    let mut state = ExecutionState::new();
    let client = state
        .heap
        .allocate_datum(TypePath::parse("/client").unwrap());
    let lobby_mob = state
        .heap
        .allocate_datum(TypePath::parse("/mob/lobby").unwrap());
    let player_mob = state
        .heap
        .allocate_datum(TypePath::parse("/mob/player").unwrap());
    state.client.attach_mob(client, lobby_mob);
    state.install_client_session(client, ControlTree::default());

    assign_datum_field(
        &mut state,
        client,
        FieldName::parse("mob").unwrap(),
        Value::Datum(player_mob),
    )
    .unwrap();
    assert_eq!(state.client.attached_mob(client), Some(player_mob));

    assign_datum_field(
        &mut state,
        client,
        FieldName::parse("mob").unwrap(),
        Value::Null,
    )
    .unwrap();
    assert!(state.client.attached_mob(client).is_none());
}

#[test]
fn local_guest_installs_skin_before_new_and_emits_lobby_ui_in_order() {
    let syntax = parse(
            "/client/proc/New()\n\t..()\n\twinset(null, \"main\", \"title=Lobby\")\n\tsrc.mob << output(\"Welcome\", \"output\")\n\tsrc.mob << browse_rsc('oracle.txt', \"oracle.txt\")\n\tsrc.mob << browse(\"<h1>Lobby</h1>\", \"window=browser\")\n\tsrc.mob << sound('lobby.ogg', 1, 0, 7, 80, 22050, -25)\n/mob/proc/Login()\n\treturn\n",
        )
        .expect("lobby client New fixture should parse");
    let module = compile_module(&syntax.definitions).expect("lobby client New should compile");
    let mut state = ExecutionState::new();
    let skin = parse_dmf(
        "window \"main\"\n\telem \"main\"\n\t\ttype = MAIN\n\telem \"browser\"\n\t\ttype = BROWSER\n\telem \"output\"\n\t\ttype = OUTPUT\n",
    );
    state.set_local_client_skin(ControlTree::from_document(&skin));
    let root = std::env::temp_dir().join(format!("dream64-lobby-ui-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("oracle.txt"), b"oracle bytes").unwrap();
    state.set_project_root(root.clone());
    let turf = state.heap.allocate_datum(TypePath::parse("/turf").unwrap());
    for (name, value) in [("x", 1), ("y", 1), ("z", 1)] {
        state
            .heap
            .set_datum_field(turf, field(name), Value::number(value as f32))
            .unwrap();
    }
    state.ensure_contents(turf).unwrap();
    state.world_turfs.insert((1, 1, 1), turf);

    let attached = state.connect_local_guest(&module).unwrap();
    let session = state.client_session(attached.client).unwrap();
    assert!(session.ui().tree().control("main", "browser").is_some());
    assert!(
        state
            .take_local_client_outbound_events(attached.client)
            .is_empty()
    );
    advance_scheduler(&module, 0, ExecutionLimits::default(), &mut state).unwrap();
    assert_eq!(
        state.take_local_client_outbound_events(attached.client),
        vec![
            LocalClientUiEvent::Winset {
                control: "main".to_owned(),
                parameters: "title=Lobby".to_owned(),
            },
            LocalClientUiEvent::Output {
                control: "output".to_owned(),
                message: "Welcome".to_owned(),
            },
            LocalClientUiEvent::BrowseResource {
                name: "oracle.txt".to_owned(),
                bytes: b"oracle bytes".to_vec(),
            },
            LocalClientUiEvent::Browse {
                window: "browser".to_owned(),
                html: "<h1>Lobby</h1>".to_owned(),
            },
            LocalClientUiEvent::Sound {
                file: Some("lobby.ogg".to_owned()),
                channel: 7,
                repeat: true,
                volume: 80.0,
                frequency: 22050.0,
                pan: -25.0,
            },
        ]
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn eager_compact_control_and_expression_families_lower_together() {
    let source = concat!(
        "/proc/run(flag)\n",
        "\tvar/i = 0\n",
        "\t++i\n",
        "\twhile(i < 3) i++\n",
        "\tvar/list/values = list(10, 20)\n",
        "\tvar/picked = values[flag ? 1 : 2]\n",
        "\tif(i in 1 to 4)\n",
        "\t\tswitch(picked)\n",
        "\t\t\tif(10) return i + picked\n",
        "\t\t\telse return 99\n",
        "\treturn 0\n",
    );
    let syntax = parse(source).expect("compact eager-family source should parse");
    let module = compile_module(&syntax.definitions)
        .expect("compact control, prefix mutation, range, and ternary index should compile");
    let entry = module.procedure_id("/proc/run").expect("run should exist");
    assert_eq!(
        execute_module(&module, entry, &[Value::number(1.0)]),
        Ok(Value::number(13.0))
    );
    assert_eq!(
        execute_module(&module, entry, &[Value::number(0.0)]),
        Ok(Value::number(99.0))
    );
}

#[test]
fn gate3_prefix_compact_switch_colon_and_optional_call_shapes_compile() {
    let source = concat!(
        "/datum/worker/proc/queue(value)\n\treturn value\n",
        "/proc/run(worker, current_vote, choice)\n",
        "\tvar/count = 0\n",
        "\t++count\n",
        "\tcurrent_vote?.reset()\n",
        "\tvar/result = worker:queue(count)\n",
        "\tswitch(choice)\n",
        "\t\tif(1) return result\n",
        "\t\telse return 0\n",
    );
    let syntax = parse(source).expect("gate3 syntax shapes should parse");
    compile_module(&syntax.definitions)
        .expect("prefix, compact switch, colon call, and optional call should lower");
}

#[test]
fn gate4_macro_expanded_statement_and_expression_shapes_compile() {
    let cases = [
        (
            "verb metadata",
            "/verb/succumb()\n\tset hidden = TRUE\n\treturn 1\n",
        ),
        (
            "increments",
            "/proc/run(target)\n\t++.\n\t++target.AdminProcCallCount\n",
        ),
        (
            "colon ternary",
            "/proc/run(target)\n\treturn target ? target:client : (target:current?:client)\n",
        ),
        (
            "inline for",
            "/proc/run(generated_actions)\n\tif(generated_actions) { for(var/I in generated_actions) qdel(I); generated_actions.Cut(); }\n",
        ),
        (
            "empty switch",
            "/proc/run(x)\n\tswitch(x)\n\t\tif(1)\n\t\tif(2) return 2\n",
        ),
        (
            "input suffix",
            "/proc/run(items)\n\treturn input(null, \"pick\", \"title\", null) as null|anything in items\n",
        ),
    ];
    for (name, source) in cases {
        let syntax = parse(source).unwrap_or_else(|error| panic!("{name}: {error}"));
        compile_module(&syntax.definitions)
            .unwrap_or_else(|error| panic!("{name}: {}", error.message));
    }
}

#[test]
fn deferred_gate_parser_families_compile_byond_shapes() {
    let source = concat!(
        "/proc/UnregisterSignal(target, signals)\n\treturn\n",
        "/datum/proc/Topic(href, href_list[])\n\treturn 0\n",
        "/client/Topic(href, href_list, hsrc, hsrc_command)\n\treturn ..()\n",
        "/client/mentor/Topic(href, href_list, hsrc)\n\treturn ..()\n",
        "/datum/example/Destroy(force)\n",
        "\tvar/list/limbs = list()\n",
        "\tUnregisterSignal(src, list(1, 2, 3,))\n",
        "\tif(limbs) { for(var/_K, V in limbs) qdel(V); limbs.Cut(); }\n",
        "\treturn ..()\n",
        "/proc/world_loop()\n",
        "\tvar/count = 0\n",
        "\tfor(var/Obj in world)\n",
        "\t\tcount++\n",
        "\treturn count\n",
        "/proc/empty_case(value)\n",
        "\tswitch(value)\n",
        "\t\tif(1)\n",
        "\t\tif(2) return 2\n",
        "/proc/trailing_case(value)\n",
        "\tswitch(value)\n",
        "\t\tif(1, 2, 3,) return 1\n",
    );
    let syntax = parse(source).expect("deferred gate syntax families should parse");
    compile_module(&syntax.definitions)
        .expect("Topic, nested trailing comma, untyped world loop, and empty case lower");
}

#[test]
fn global_vars_is_a_live_iterable_namespace_over_global_storage() {
    let source = concat!(
        "/proc/run()\n",
        "\tvar/list/reflection = global.vars\n",
        "\tvar/total = 0\n",
        "\tfor(var/name in reflection)\n",
        "\t\ttotal += reflection[name]\n",
        "\tglobal.counter = 5\n",
        "\tvar/live_read = reflection[\"counter\"]\n",
        "\treflection[\"counter\"] = 7\n",
        "\treturn total * 100 + live_read * 10 + global.counter\n",
    );
    let syntax = parse(source).expect("global.vars source should parse");
    let program = compile_procedure(&syntax.definitions[0])
        .expect("global.vars iteration and indexed writes should compile");
    let mut state = ExecutionState::new();
    state.set_global(field("counter"), Value::number(3.0));
    let qualified = FieldName::static_storage("/datum/example/var/static/shared");
    state.set_global(qualified, Value::number(8.0));

    assert_eq!(
        execute_in_state(&program, &[], &mut state),
        Ok(Value::number(1157.0))
    );
    assert_eq!(state.global(&field("counter")), Some(&Value::number(7.0)));
}

#[test]
fn dynamic_field_access_resolves_inherited_shared_storage_after_instance_fields() {
    let source = concat!(
        "/proc/read_member(datum/value)\n\treturn value.all_layers\n",
        "/proc/write_member(datum/value, replacement)\n",
        "\tvalue.all_layers = replacement\n",
        "\treturn value.all_layers\n",
    );
    let syntax = parse(source).expect("dynamic shared-member source should parse");
    let module =
        compile_module(&syntax.definitions).expect("dynamic shared-member source should compile");
    let mut state = ExecutionState::new();
    let runtime_type = TypePath::parse("/datum/bodypart_overlay/mutant").unwrap();
    let storage = FieldName::static_storage("/datum/bodypart_overlay/var/all_layers");
    state.set_shared_fields(Arc::new(BTreeMap::from([(
        runtime_type.clone(),
        BTreeMap::from([(field("all_layers"), storage.clone())]),
    )])));
    state.set_global(storage.clone(), Value::number(7.0));

    let inherited = state.heap_mut().allocate_datum(runtime_type.clone());
    let read = module.procedure_id("/proc/read_member").unwrap();
    let write = module.procedure_id("/proc/write_member").unwrap();
    assert_eq!(
        execute_module_in_state(&module, read, &[Value::Datum(inherited)], &mut state),
        Ok(Value::number(7.0))
    );
    assert_eq!(
        execute_module_in_state(
            &module,
            write,
            &[Value::Datum(inherited), Value::number(9.0)],
            &mut state,
        ),
        Ok(Value::number(9.0))
    );
    assert_eq!(state.global(&storage), Some(&Value::number(9.0)));

    let shadowed = state.heap_mut().allocate_datum(runtime_type);
    state
        .heap_mut()
        .set_datum_field(shadowed, field("all_layers"), Value::number(3.0))
        .unwrap();
    assert_eq!(
        execute_module_in_state(
            &module,
            write,
            &[Value::Datum(shadowed), Value::number(4.0)],
            &mut state,
        ),
        Ok(Value::number(4.0))
    );
    assert_eq!(state.global(&storage), Some(&Value::number(9.0)));
}

#[test]
fn scope_operator_static_assignment_and_read_target_the_shared_slot() {
    // DM `Type::name` scope operator: the semantic layer binds it to a
    // `"<type path>::<name>"` key in `global_fields`; VM lowering must redirect
    // the write, the compound update, and the read to that qualified static
    // slot instead of failing on a non-writable target.
    let source = concat!(
        "/proc/meta_gas_list()\n",
        "\t/datum/gas_mixture::gas_meta = 5\n",
        "\t/datum/gas_mixture::gas_meta += 2\n",
        "\treturn /datum/gas_mixture::gas_meta\n",
    );
    let syntax = parse(source).expect("scope-operator source should parse");
    let procedures = syntax
        .definitions
        .iter()
        .filter(|definition| matches!(definition.kind, DefinitionKind::Procedure))
        .cloned()
        .collect::<Vec<_>>();
    let storage = FieldName::static_storage("/datum/gas_mixture/var/gas_meta");
    let globals = BTreeMap::from([("/datum/gas_mixture::gas_meta".to_owned(), storage.clone())]);
    let module = compile_module_with_global_fields(&procedures, &globals)
        .expect("scope-operator statics should compile");
    let entry = module.procedure_id("/proc/meta_gas_list").unwrap();
    let program = module.procedure(entry).unwrap();
    assert!(
        program
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::StoreGlobal(field) if field == &storage)),
        "{:?}",
        program.instructions
    );
    assert!(
        program.instructions.iter().any(
            |instruction| matches!(instruction, Instruction::LoadGlobal(field) if field == &storage)
        ),
        "{:?}",
        program.instructions
    );
    assert!(!program.instructions.iter().any(|instruction| matches!(
        instruction,
        Instruction::StoreField(_)
            | Instruction::LoadInitialGlobal(_)
            | Instruction::InitialField(_)
    )));

    let mut state = ExecutionState::new();
    assert_eq!(
        execute_module_in_state(&module, entry, &[], &mut state),
        Ok(Value::number(7.0))
    );
    assert_eq!(state.global(&storage), Some(&Value::number(7.0)));
}

#[test]
fn atom_appearance_read_returns_a_truthy_snapshot_for_overlay_normalization() {
    let source = concat!(
        "/proc/normalize(atom/target)\n",
        "\tvar/list/new_overlays = list(target)\n",
        "\tnew_overlays[1] = target.appearance\n",
        "\tfor(var/overlay in new_overlays)\n",
        "\t\tif(!overlay)\n",
        "\t\t\tnew_overlays -= overlay\n",
        "\treturn new_overlays[1]\n",
    );
    let syntax = parse(source).expect("appearance normalization source should parse");
    let module = compile_module(&syntax.definitions)
        .expect("appearance normalization source should compile");
    let mut state = ExecutionState::new();
    let target = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/item/uniform").unwrap());
    for (name, value) in [
        ("appearance", Value::Null),
        ("icon", Value::file("icons/uniform.dmi")),
        ("icon_state", Value::text("uniform")),
        ("dir", Value::number(2.0)),
    ] {
        state
            .heap_mut()
            .set_datum_field(target, field(name), value)
            .unwrap();
    }

    let normalized = execute_module_in_state(
        &module,
        module.procedure_id("/proc/normalize").unwrap(),
        &[Value::Datum(target)],
        &mut state,
    )
    .expect("a copied atom appearance must remain in the overlay list");
    let Value::Datum(appearance) = normalized else {
        panic!("atom.appearance should produce an appearance datum")
    };
    let appearance = state.heap().datum(appearance).unwrap();
    assert_eq!(appearance.type_path().as_str(), "/mutable_appearance");
    assert_eq!(
        appearance.field(&field("icon_state")),
        Ok(&Value::text("uniform"))
    );
    assert_eq!(appearance.field(&field("dir")), Ok(&Value::number(2.0)));
}

#[test]
fn project_insert_method_wins_over_native_icon_insert() {
    let source = concat!(
        "/datum/sheet/proc/Insert(key, value)\n\treturn key == \"antag\" && value == 7\n",
        "/datum/sheet/proc/create_spritesheets()\n\treturn Insert(\"antag\", 7)\n",
        "/proc/run()\n",
        "\tvar/datum/sheet/sheet = new\n",
        "\treturn sheet.create_spritesheets()\n",
    );
    let syntax = parse(source).expect("project Insert source should parse");
    let module = compile_module(&syntax.definitions).expect("project Insert should compile");
    let create = module
        .procedure_id("/datum/sheet/proc/create_spritesheets")
        .unwrap();
    assert!(module
            .resolve_procedure(create)
            .unwrap()
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::NativeSrcMethod { name, .. } if name == "Insert")));
    let mut state = ExecutionState::new();
    assert_eq!(
        execute_module_in_state(
            &module,
            module.procedure_id("/proc/run").unwrap(),
            &[],
            &mut state,
        ),
        Ok(Value::number(1.0))
    );
}

#[test]
fn nested_waitfor_false_processing_loop_does_not_block_staged_initializer() {
    let source = concat!(
        "/proc/main()\n\tinitialize()\n\treturn 1\n",
        "/proc/initialize()\n",
        "\tset waitfor = 0\n",
        "\tsleep(1)\n",
        "\tglobal.stage = 1\n",
        "\tstart_processing()\n",
        "\tglobal.stage = 2\n",
        "\tsleep(1)\n",
        "\tglobal.stage = 3\n",
        "/proc/start_processing()\n",
        "\tset waitfor = 0\n",
        "\tglobal.loops += 1\n",
        "\tsleep(1)\n",
        "\tglobal.loops += 10\n",
    );
    let syntax = parse(source).expect("staged waitfor source should parse");
    let module = compile_module(&syntax.definitions).expect("staged waitfor source should compile");
    let mut state = ExecutionState::new();
    state.set_global(field("stage"), Value::number(0.0));
    state.set_global(field("loops"), Value::number(0.0));
    let main = module.procedure_id("/proc/main").expect("main exists");

    assert_eq!(
        execute_module_in_state(&module, main, &[], &mut state),
        Ok(Value::number(1.0))
    );
    assert_eq!(state.scheduled_task_count(), 1);
    advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state)
        .expect("first initialization slice should run");
    assert_eq!(state.global(&field("stage")), Some(&Value::number(2.0)));
    assert_eq!(state.global(&field("loops")), Some(&Value::number(1.0)));
    assert_eq!(state.scheduled_task_count(), 2);
    advance_scheduler(&module, 1, ExecutionLimits::default(), &mut state)
        .expect("processing and initialization continuations should both run");
    assert_eq!(state.global(&field("stage")), Some(&Value::number(3.0)));
    assert_eq!(state.global(&field("loops")), Some(&Value::number(11.0)));
}

#[test]
fn uppertext_preserves_non_text_apc_direction_fallback() {
    let source = concat!(
        "/proc/dir2text(direction)\n",
        "\tswitch(direction)\n",
        "\t\tif(1) return \"north\"\n",
        "\treturn 0\n",
        "/proc/run(direction)\n",
        "\treturn list(uppertext(dir2text(direction)), lowertext(null), uppertext(\"east\"))\n",
    );
    let syntax = parse(source).expect("APC direction coercion source should parse");
    let module =
        compile_module(&syntax.definitions).expect("APC direction coercion should compile");
    let mut state = ExecutionState::new();
    let result = execute_module_in_state(
        &module,
        module.procedure_id("/proc/run").unwrap(),
        &[Value::number(0.0)],
        &mut state,
    )
    .expect("non-text case conversion must not fail");
    let Value::List(result) = result else {
        panic!("run should return a list")
    };
    let result = state.heap().list(result).unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result.get(1), Ok(&Value::number(0.0)));
    assert_eq!(result.get(2), Ok(&Value::Null));
    assert_eq!(result.get(3), Ok(&Value::text("EAST")));
}

#[test]
fn synthesized_movable_in_turf_contents_inherits_engine_atom_contract() {
    let source = concat!(
        "/turf/proc/probe()\n",
        "\tfor(var/atom/movable/item as anything in src.contents)\n",
        "\t\treturn item.density + item.alpha + item.dir + initial(item.density) + initial(item.alpha) + initial(item.dir) + item.vars[\"density\"] + item.vars[\"alpha\"] + item.vars[\"dir\"]\n",
        "\treturn -1\n",
    );
    let syntax = parse(source).expect("atom-contract fixture should parse");
    let module = compile_module(&syntax.definitions).expect("atom-contract fixture should compile");
    let mut state = ExecutionState::new();
    // This is deliberately a raw synthesized path with no type/default
    // catalogs at all, matching native/runtime construction failures seen
    // in Monk's Atoms pass. Engine-owned atom state must stand alone.

    let turf = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/turf/open/floor").unwrap());
    let synthesized = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/obj/effect/spawner/runtime_loot").unwrap());
    let contents = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(contents)
        .unwrap()
        .add(Value::Datum(synthesized));
    state
        .heap_mut()
        .set_datum_field(turf, field("contents"), Value::List(contents))
        .unwrap();
    state
        .heap_mut()
        .set_datum_field(synthesized, field("loc"), Value::Datum(turf))
        .unwrap();

    assert_eq!(
        execute_module_in_context(
            &module,
            module.procedure_id("/turf/proc/probe").unwrap(),
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(turf), Value::Null),
        ),
        Ok(Value::number(771.0)),
    );
}

#[test]
fn synthesized_engine_types_merge_every_builtin_owner_layer() {
    let mut state = ExecutionState::new();
    let catalogs = [
        ("/atom", "atom_field", 1.0),
        ("/atom/movable", "movable_field", 2.0),
        ("/obj", "obj_field", 3.0),
        ("/mob", "mob_field", 4.0),
        ("/turf", "turf_field", 5.0),
        ("/area", "area_field", 6.0),
    ]
    .into_iter()
    .map(|(path, name, value)| {
        (
            TypePath::parse(path).unwrap(),
            BTreeMap::from([(field(name), Value::number(value))]),
        )
    })
    .collect();
    state.set_initial_values(catalogs);

    for (runtime_type, expected) in [
        (
            "/obj/effect/runtime",
            &["atom_field", "movable_field", "obj_field"][..],
        ),
        (
            "/mob/living/runtime",
            &["atom_field", "movable_field", "mob_field"][..],
        ),
        (
            "/atom/movable/runtime",
            &["atom_field", "movable_field"][..],
        ),
        ("/turf/runtime", &["atom_field", "turf_field"][..]),
        ("/area/runtime", &["atom_field", "area_field"][..]),
    ] {
        let datum = state
            .heap_mut()
            .allocate_datum(TypePath::parse(runtime_type).unwrap());
        for name in expected {
            assert!(
                datum_field_or_initial(&state, datum, &field(name)).is_ok(),
                "{runtime_type} lost inherited engine field {name}",
            );
        }
    }
}

#[test]
fn every_standard_engine_field_exists_without_project_metadata() {
    let mut state = ExecutionState::new();
    for runtime_type in [
        "/obj/runtime",
        "/mob/runtime",
        "/atom/movable/runtime",
        "/turf/runtime",
        "/area/runtime",
        "/image/runtime",
        "/client/runtime",
        "/particles/runtime",
    ] {
        let path = TypePath::parse(runtime_type).unwrap();
        let datum = state.heap_mut().allocate_datum(path.clone());
        let expected = super::engine_root_paths(&path)
            .iter()
            .flat_map(|owner| super::engine_owner_field_names(owner))
            .collect::<BTreeSet<_>>();
        assert!(
            !expected.is_empty(),
            "{runtime_type} has no engine contract"
        );
        for name in expected {
            assert!(
                datum_field_or_initial(&state, datum, &field(name)).is_ok(),
                "{runtime_type} is missing engine-owned field {name}",
            );
        }
    }
}

#[test]
fn simple_for_field_assignment_bulk_path_preserves_values_under_tight_budget() {
    let syntax = parse(
            "/proc/run(list/items, value)\n\tfor(var/turf/item as anything in items)\n\t\titem.luminosity = value\n\treturn 7\n",
        )
        .unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let entry = module.procedure_id("/proc/run").unwrap();
    let mut state = ExecutionState::new();
    let items = state.heap_mut().allocate_list();
    let mut datums = Vec::new();
    for _ in 0..128 {
        let datum = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/turf/test").unwrap());
        state
            .heap_mut()
            .list_mut(items)
            .unwrap()
            .add(Value::Datum(datum));
        datums.push(datum);
    }
    let result = execute_module_with_limits_in_state(
        &module,
        entry,
        &[Value::List(items), Value::number(73.0)],
        ExecutionLimits {
            max_call_depth: 8,
            max_steps: 4,
            wall_clock_budget: None,
        },
        &mut state,
    )
    .unwrap();
    assert_eq!(result.as_number(), Some(7.0));
    for datum in datums {
        assert_eq!(
            datum_field_or_initial(&state, datum, &field("luminosity"))
                .unwrap()
                .as_number(),
            Some(73.0),
        );
    }
}

#[test]
fn as_anything_keeps_declared_receiver_fields_without_runtime_filtering() {
    let syntax = parse(
            "/proc/run(list/items)\n\tfor(var/atom/movable/item as anything in items)\n\t\tif(isnull(item.important_recursive_contents))\n\t\t\titem.important_recursive_contents = list()\n\t\treturn islist(item.important_recursive_contents)\n\treturn 0\n",
        )
        .expect("typed as-anything loop should parse");
    let module = compile_module(&syntax.definitions).expect("fixture should compile");
    let entry = module.procedure_id("/proc/run").unwrap();
    let mut state = ExecutionState::new();
    let turf = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/turf/test").unwrap());
    let items = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(items)
        .unwrap()
        .add(Value::Datum(turf));

    assert_eq!(
        execute_module_in_state(&module, entry, &[Value::List(items)], &mut state),
        Ok(Value::number(1.0)),
    );
    assert!(matches!(
        state
            .heap()
            .datum_field(turf, &field("important_recursive_contents")),
        Ok(Value::List(_))
    ));
}

#[test]
fn simple_for_field_assignment_falls_back_before_mixed_receiver_error() {
    let syntax =
        parse("/proc/run(list/items)\n\tfor(var/item in items)\n\t\titem.luminosity = 9\n")
            .unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let entry = module.procedure_id("/proc/run").unwrap();
    let mut state = ExecutionState::new();
    let datum = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/turf/test").unwrap());
    let items = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(items)
        .unwrap()
        .add(Value::Datum(datum));
    state
        .heap_mut()
        .list_mut(items)
        .unwrap()
        .add(Value::number(1.0));

    let error =
        execute_module_in_state(&module, entry, &[Value::List(items)], &mut state).unwrap_err();
    assert!(error.message.contains("field write requires a datum"));
    assert_eq!(
        datum_field_or_initial(&state, datum, &field("luminosity"))
            .unwrap()
            .as_number(),
        Some(9.0),
    );
}

#[test]
fn simple_for_field_assignment_snapshots_contents_and_uses_spatial_writes() {
    let syntax = parse(
            "/proc/run(list/items)\n\tfor(var/atom/movable/item as anything in items)\n\t\titem.loc = null\n\treturn 1\n",
        )
        .unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let entry = module.procedure_id("/proc/run").unwrap();
    let mut state = ExecutionState::new();
    let container = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/container").unwrap());
    let contents = state.ensure_contents(container).unwrap();
    let mut members = Vec::new();
    for _ in 0..64 {
        let member = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/obj/test").unwrap());
        super::assign_datum_or_shared_field(
            &mut state,
            member,
            field("loc"),
            Value::Datum(container),
        )
        .unwrap();
        members.push(member);
    }
    assert_eq!(state.heap().list(contents).unwrap().len(), members.len());

    let result = execute_module_with_limits_in_state(
        &module,
        entry,
        &[Value::List(contents)],
        ExecutionLimits {
            max_call_depth: 8,
            max_steps: 4,
            wall_clock_budget: None,
        },
        &mut state,
    )
    .unwrap();
    assert_eq!(result.as_number(), Some(1.0));
    assert!(state.heap().list(contents).unwrap().is_empty());
    for member in members {
        assert!(matches!(
            datum_field_or_initial(&state, member, &field("loc")),
            Ok(Value::Null)
        ));
    }
}

#[test]
fn numeric_jit_matches_interpreter_and_guards_dynamic_values() {
    let syntax =
        parse("/proc/calculate(a, b)\n\treturn -(a * 2 + b)").expect("numeric JIT fixture parses");
    let module = compile_module(&syntax.definitions).expect("numeric JIT fixture compiles");
    let entry = module.procedure_id("/proc/calculate").unwrap();

    // Numeric arguments select native execution when JIT is enabled.
    assert_eq!(
        execute_module(&module, entry, &[Value::number(3.0), Value::number(4.0)]),
        Ok(Value::number(-10.0)),
    );
    // The same bytecode shape with a dynamic non-number must fall back to
    // the interpreter instead of entering the numeric ABI.
    assert!(execute_module(&module, entry, &[Value::text("3"), Value::number(4.0)]).is_err());
}

#[test]
#[ignore = "bounded procedure-entry JIT configuration microbenchmark"]
fn cached_jit_configuration_microbenchmark() {
    const ITERATIONS: usize = 1_000_000;
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(std::env::var_os("DREAM64_DISABLE_JIT").is_some());
    }
    let environment = started.elapsed();
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(super::jit_disabled());
    }
    let cached = started.elapsed();
    eprintln!(
        "jit-config iterations={ITERATIONS} environment_ms={} cached_ms={}",
        environment.as_millis(),
        cached.as_millis(),
    );
}

#[test]
#[ignore = "release-only negative numeric-JIT entry microbenchmark"]
fn numeric_jit_negative_prefix_gate_release_microbenchmark() {
    const ITERATIONS: usize = 10_000_000;
    // Representative of mapping procedures: a cheap argument guard uses
    // DM truthiness (`Not`), which the generic numeric tier cannot lower.
    let program = manual_program(
        vec![
            Instruction::LoadLocal(1),
            Instruction::Not,
            Instruction::JumpIfFalse(4),
            Instruction::Return,
            Instruction::LoadField(field("members")),
            Instruction::Return,
        ],
        2,
    );
    let key = (7_u64, crate::ProcedureId::from_index(42).unwrap());
    let negative = HashMap::from([(key, None::<u8>)]);

    let started = Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(negative.get(&std::hint::black_box(key)));
    }
    let cached_negative = started.elapsed();
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(super::numeric_jit_prefix_candidate(&program));
    }
    let prefix_gate = started.elapsed();
    eprintln!(
        "numeric-jit-negative iterations={ITERATIONS} cache_ms={} prefix_ms={}",
        cached_negative.as_millis(),
        prefix_gate.as_millis(),
    );
}

#[test]
fn numeric_jit_lowers_isolated_locals_and_cfg_conservatively() {
    let syntax = parse(
            "/proc/calculate(a, b)\n\tvar/result = a + b\n\tif(result > 10)\n\t\tresult = result * 2\n\treturn result",
        )
        .expect("numeric CFG fixture parses");
    let module = compile_module(&syntax.definitions).expect("numeric CFG fixture compiles");
    let entry = module.procedure_id("/proc/calculate").unwrap();
    let program = &module.procedures[entry.index()];
    let lowered = crate::numeric_trace_instructions(program).expect("safe numeric CFG lowers");
    assert!(
        lowered
            .iter()
            .any(|instruction| matches!(instruction, dm_jit::NumericInstruction::StoreLocal(_)))
    );
    assert!(
        lowered
            .iter()
            .any(|instruction| matches!(instruction, dm_jit::NumericInstruction::GreaterThan))
    );
    assert!(
        lowered
            .iter()
            .any(|instruction| matches!(instruction, dm_jit::NumericInstruction::JumpIfFalse(_)))
    );
    assert!(matches!(
        lowered.last(),
        Some(dm_jit::NumericInstruction::Return)
    ));

    // Updating a declared argument is observable through DM's live args
    // semantics and must not enter this isolated-locals tier.
    let syntax = parse("/proc/update(a)\n\ta = a + 1\n\treturn a").unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let entry = module.procedure_id("/proc/update").unwrap();
    assert!(crate::numeric_trace_instructions(&module.procedures[entry.index()]).is_none());
}

#[test]
fn numeric_jit_loop_resumes_at_budget_safepoints() {
    let source = "/proc/count(limit)\n\tvar/i = 0\n\twhile(i < limit)\n\t\ti = i + 1\n\treturn i";
    let syntax = parse(source).unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let entry = module.procedure_id("/proc/count").unwrap();
    assert!(crate::numeric_trace_instructions(&module.procedures[entry.index()]).is_some());

    // Force an equivalent copy through the interpreter by appending an
    // unreachable unsupported opcode after Return. This avoids changing
    // the process-wide JIT environment variable in a parallel test suite.
    let syntax = parse(source).unwrap();
    let mut reference = compile_module(&syntax.definitions).unwrap();
    let reference_entry = reference.procedure_id("/proc/count").unwrap();
    Arc::make_mut(&mut reference.procedures[reference_entry.index()])
        .instructions
        .push(Instruction::PushNull);
    assert!(
        crate::numeric_trace_instructions(&reference.procedures[reference_entry.index()]).is_none()
    );
    assert_eq!(
        execute_module(&reference, reference_entry, &[Value::number(25.0)]),
        Ok(Value::number(25.0))
    );

    let mut frames = vec![crate::make_frame(
        entry,
        &module.procedures[entry.index()],
        &[Value::number(25.0)],
        &ExecutionContext::default(),
    )];
    let mut state = ExecutionState::new();
    let limits = ExecutionLimits {
        max_call_depth: 8,
        max_steps: 5,
        wall_clock_budget: None,
    };
    let mut native_continuation_seen = false;
    loop {
        match crate::run_frames(
            &module,
            frames,
            limits,
            crate::StepBudgetBehavior::YieldScheduledContinuation,
            &mut state,
        )
        .unwrap()
        {
            crate::FrameRunOutcome::Yielded { frames: next, .. } => {
                native_continuation_seen |= next[0].numeric_jit_state().is_some();
                frames = next;
            }
            crate::FrameRunOutcome::Complete(value) => {
                assert_eq!(value, Value::number(25.0));
                break;
            }
            crate::FrameRunOutcome::Prompted { .. } => {
                panic!("numeric JIT fixture cannot prompt")
            }
        }
    }
    assert!(
        native_continuation_seen,
        "loop should yield from native execution"
    );
}

#[test]
fn lumcount_field_jit_matches_interpreter_and_queue_action() {
    let syntax = parse(concat!(
        "/datum/lighting_corner/proc/update_lumcount(delta_r, delta_g, delta_b)\n",
        "\tif(!(delta_r || delta_g || delta_b))\n\t\treturn\n",
        "\tlum_r += delta_r\n\tlum_g += delta_g\n\tlum_b += delta_b\n",
        "\tif(!needs_update)\n\t\tneeds_update = 1\n",
        "\t\tSSlighting.corners_queue += src\n",
    ))
    .unwrap();
    let module = compile_module_specs(&[ProcedureSpec {
        path: "/datum/lighting_corner/proc/update_lumcount".to_owned(),
        definition: &syntax.definitions[0],
        parent: None,
        static_calls: BTreeMap::new(),
        src_fields: BTreeMap::from([
            ("lum_r".to_owned(), field("lum_r")),
            ("lum_g".to_owned(), field("lum_g")),
            ("lum_b".to_owned(), field("lum_b")),
            ("needs_update".to_owned(), field("needs_update")),
        ]),
        global_fields: BTreeMap::from([("SSlighting".to_owned(), field("SSlighting"))]),
    }])
    .unwrap();
    let entry = module
        .procedure_id("/datum/lighting_corner/proc/update_lumcount")
        .unwrap();
    assert!(
        crate::compile_lumcount_trace(&module.procedures[entry.index()]).is_some(),
        "locals={} len={} instructions={:#?}",
        module.procedures[entry.index()].local_count,
        module.procedures[entry.index()].instructions.len(),
        module.procedures[entry.index()].instructions
    );
    let mut reference = module.clone();
    reference.identity = crate::next_module_identity();
    Arc::make_mut(&mut reference.procedures[entry.index()])
        .instructions
        .push(Instruction::PushNull);

    let run = |module: &Module| {
        let mut state = ExecutionState::new();
        let corner = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/lighting_corner").unwrap());
        for (name, value) in [
            ("lum_r", 1.0),
            ("lum_g", 2.0),
            ("lum_b", 3.0),
            ("needs_update", 0.0),
        ] {
            state
                .heap_mut()
                .set_datum_field(corner, field(name), Value::number(value))
                .unwrap();
        }
        let lighting = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/controller/subsystem/lighting").unwrap());
        let queue = state.heap_mut().allocate_list();
        state
            .heap_mut()
            .set_datum_field(lighting, field("corners_queue"), Value::List(queue))
            .unwrap();
        state.set_global(field("SSlighting"), Value::Datum(lighting));
        let result = execute_module_in_context(
            module,
            entry,
            &[Value::number(0.5), Value::number(1.0), Value::number(-0.5)],
            &mut state,
            &ExecutionContext::new(Value::Datum(corner), Value::Null),
        )
        .unwrap();
        let fields = ["lum_r", "lum_g", "lum_b", "needs_update"]
            .map(|name| datum_field_or_initial(&state, corner, &field(name)).unwrap());
        let queued = state
            .heap()
            .list(queue)
            .unwrap()
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>();
        (result, fields, queued)
    };
    let native = run(&module);
    let interpreted = run(&reference);
    assert_eq!(native, interpreted);
    assert_eq!(native.0, Value::Null);
    assert_eq!(
        native.1,
        [
            Value::number(1.5),
            Value::number(3.0),
            Value::number(2.5),
            Value::number(1.0)
        ]
    );
    assert_eq!(native.2.len(), 1);
}

#[test]
#[ignore = "local release microbenchmark"]
fn lumcount_jit_release_microbenchmark() {
    const CALLS: usize = 100_000;
    let syntax = parse(concat!(
        "/datum/lighting_corner/proc/update_lumcount(delta_r, delta_g, delta_b)\n",
        "\tif(!(delta_r || delta_g || delta_b))\n\t\treturn\n",
        "\tlum_r += delta_r\n\tlum_g += delta_g\n\tlum_b += delta_b\n",
        "\tif(!needs_update)\n\t\tneeds_update = 1\n",
        "\t\tSSlighting.corners_queue += src\n",
    ))
    .unwrap();
    let spec = || ProcedureSpec {
        path: "/datum/lighting_corner/proc/update_lumcount".to_owned(),
        definition: &syntax.definitions[0],
        parent: None,
        static_calls: BTreeMap::new(),
        src_fields: BTreeMap::from([
            ("lum_r".to_owned(), field("lum_r")),
            ("lum_g".to_owned(), field("lum_g")),
            ("lum_b".to_owned(), field("lum_b")),
            ("needs_update".to_owned(), field("needs_update")),
        ]),
        global_fields: BTreeMap::from([("SSlighting".to_owned(), field("SSlighting"))]),
    };
    let module = compile_module_specs(&[spec()]).unwrap();
    let entry = module
        .procedure_id("/datum/lighting_corner/proc/update_lumcount")
        .unwrap();
    let mut reference = compile_module_specs(&[spec()]).unwrap();
    Arc::make_mut(&mut reference.procedures[entry.index()])
        .instructions
        .push(Instruction::PushNull);

    let setup = || {
        let mut state = ExecutionState::new();
        let corner = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/lighting_corner").unwrap());
        for name in ["lum_r", "lum_g", "lum_b"] {
            state
                .heap_mut()
                .set_datum_field(corner, field(name), Value::number(0.0))
                .unwrap();
        }
        state
            .heap_mut()
            .set_datum_field(corner, field("needs_update"), Value::number(1.0))
            .unwrap();
        let lighting = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/controller/subsystem/lighting").unwrap());
        let queue = state.heap_mut().allocate_list();
        state
            .heap_mut()
            .set_datum_field(lighting, field("corners_queue"), Value::List(queue))
            .unwrap();
        state.set_global(field("SSlighting"), Value::Datum(lighting));
        (state, corner)
    };
    let arguments = [
        Value::number(0.01),
        Value::number(0.02),
        Value::number(0.03),
    ];
    let (mut native_state, native_corner) = setup();
    let native_context = ExecutionContext::new(Value::Datum(native_corner), Value::Null);
    let started = Instant::now();
    for _ in 0..CALLS {
        execute_module_in_context(
            &module,
            entry,
            &arguments,
            &mut native_state,
            &native_context,
        )
        .unwrap();
    }
    let native_elapsed = started.elapsed();

    let (mut reference_state, reference_corner) = setup();
    let reference_context = ExecutionContext::new(Value::Datum(reference_corner), Value::Null);
    let started = Instant::now();
    for _ in 0..CALLS {
        execute_module_in_context(
            &reference,
            entry,
            &arguments,
            &mut reference_state,
            &reference_context,
        )
        .unwrap();
    }
    let reference_elapsed = started.elapsed();
    eprintln!(
        "lumcount end-to-end calls={CALLS} jit_ms={} interpreter_ms={} speedup={:.2}",
        native_elapsed.as_millis(),
        reference_elapsed.as_millis(),
        reference_elapsed.as_secs_f64() / native_elapsed.as_secs_f64(),
    );
    let zero_arguments = [Value::number(0.0), Value::number(0.0), Value::number(0.0)];
    let started = Instant::now();
    for _ in 0..CALLS {
        execute_module_in_context(
            &module,
            entry,
            &zero_arguments,
            &mut native_state,
            &native_context,
        )
        .unwrap();
    }
    let native_zero = started.elapsed();
    let started = Instant::now();
    for _ in 0..CALLS {
        execute_module_in_context(
            &reference,
            entry,
            &zero_arguments,
            &mut reference_state,
            &reference_context,
        )
        .unwrap();
    }
    let reference_zero = started.elapsed();
    eprintln!(
        "lumcount zero-delta calls={CALLS} jit_ms={} interpreter_ms={} speedup={:.2}",
        native_zero.as_millis(),
        reference_zero.as_millis(),
        reference_zero.as_secs_f64() / native_zero.as_secs_f64(),
    );
}

#[test]
#[ignore = "local release microbenchmark"]
fn simple_for_field_assignment_release_microbenchmark() {
    const ITEMS: usize = 135_170;
    let syntax = parse(
            "/proc/fast(list/items, value)\n\tfor(var/turf/item as anything in items)\n\t\titem.luminosity = value\n/proc/bytecode(list/items, value)\n\tfor(var/turf/item as anything in items)\n\t\tif(item)\n\t\t\titem.luminosity = value\n",
        )
        .unwrap();
    let module = compile_module(&syntax.definitions).unwrap();
    let mut state = ExecutionState::new();
    let items = state.heap_mut().allocate_list();
    for _ in 0..ITEMS {
        let datum = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/turf/test").unwrap());
        state
            .heap_mut()
            .list_mut(items)
            .unwrap()
            .add(Value::Datum(datum));
    }
    let bytecode = module.procedure_id("/proc/bytecode").unwrap();
    let started = Instant::now();
    execute_module_in_state(
        &module,
        bytecode,
        &[Value::List(items), Value::number(1.0)],
        &mut state,
    )
    .unwrap();
    let bytecode_elapsed = started.elapsed();

    let fast = module.procedure_id("/proc/fast").unwrap();
    let started = Instant::now();
    execute_module_in_state(
        &module,
        fast,
        &[Value::List(items), Value::number(2.0)],
        &mut state,
    )
    .unwrap();
    let fast_elapsed = started.elapsed();
    eprintln!(
        "simple iteration field assignment items={ITEMS} bytecode_ms={} fast_ms={} speedup={:.2}",
        bytecode_elapsed.as_millis(),
        fast_elapsed.as_millis(),
        bytecode_elapsed.as_secs_f64() / fast_elapsed.as_secs_f64(),
    );
}

fn synthetic_builtin_module(name: &str, body: &str, caller: &str) -> Module {
    let source = format!("{caller}\n{body}");
    let syntax = parse(&source).unwrap();
    let specs = [
        ProcedureSpec {
            path: "/proc/run".to_owned(),
            definition: &syntax.definitions[0],
            parent: None,
            static_calls: BTreeMap::from([(name.to_owned(), 1)]),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
        ProcedureSpec {
            path: format!("/proc/{name}@dream64_builtin"),
            definition: &syntax.definitions[1],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        },
    ];
    compile_module_specs(&specs).unwrap()
}

#[test]
fn canonical_synthetic_max_executes_without_a_callee_frame() {
    let module = synthetic_builtin_module(
        "max",
        "/proc/max(...)\n\tvar/list/values = args\n\tif(length(args) == 1 && islist(args[1]))\n\t\tvalues = args[1]\n\tif(!length(values))\n\t\treturn null\n\tvar/result = values[1]\n\tfor(var/value in values)\n\t\tif(value > result)\n\t\t\tresult = value\n\treturn result\n",
        "/proc/run()\n\treturn max(3, 9, 4)",
    );
    let entry = module.procedure_id("/proc/run").unwrap();
    let caller_steps = module.procedure(entry).unwrap().instructions.len() as u64;
    assert_eq!(
        execute_module_with_limits(
            &module,
            entry,
            &[],
            ExecutionLimits {
                max_call_depth: 8,
                max_steps: caller_steps,
                wall_clock_budget: None,
            },
        )
        .unwrap()
        .as_number(),
        Some(9.0),
    );
}

#[test]
fn canonical_synthetic_istext_matches_all_value_families() {
    let body = "/proc/istext(value)\n\treturn !isnull(value) && !isnum(value) && !ispath(value) && !islist(value) && !istype(value)\n";
    let module =
        synthetic_builtin_module("istext", body, "/proc/run(value)\n\treturn istext(value)");
    let entry = module.procedure_id("/proc/run").unwrap();
    let mut state = ExecutionState::new();
    let list = state.heap_mut().allocate_list();
    let datum = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/test").unwrap());
    for (value, expected) in [
        (Value::Null, 0.0),
        (Value::number(1.0), 0.0),
        (Value::TypePath(TypePath::parse("/datum").unwrap()), 0.0),
        (Value::List(list), 0.0),
        (Value::Datum(datum), 0.0),
        (Value::text("yes"), 1.0),
        (Value::file("icon.dmi"), 1.0),
    ] {
        assert_eq!(
            execute_module_in_state(&module, entry, &[value], &mut state)
                .unwrap()
                .as_number(),
            Some(expected),
        );
    }
}

#[test]
fn customized_synthetic_named_body_is_not_bypassed() {
    let module = synthetic_builtin_module(
        "max",
        "/proc/max(...)\n\treturn 99\n",
        "/proc/run()\n\treturn max(3, 9, 4)",
    );
    let entry = module.procedure_id("/proc/run").unwrap();
    assert_eq!(
        execute_module(&module, entry, &[]).unwrap().as_number(),
        Some(99.0)
    );
}

#[test]
fn canonical_synthetic_builtin_error_falls_back_to_exact_callee_trace() {
    let module = synthetic_builtin_module(
        "max",
        "/proc/max(...)\n\tvar/list/values = args\n\tif(length(args) == 1 && islist(args[1]))\n\t\tvalues = args[1]\n\tif(!length(values))\n\t\treturn null\n\tvar/result = values[1]\n\tfor(var/value in values)\n\t\tif(value > result)\n\t\t\tresult = value\n\treturn result\n",
        "/proc/run(value)\n\treturn max(value)",
    );
    let entry = module.procedure_id("/proc/run").unwrap();
    let mut state = ExecutionState::new();
    let stale = state.heap_mut().allocate_list();
    state.heap_mut().destroy_list(stale).unwrap();
    let result =
        execute_module_in_state(&module, entry, &[Value::List(stale)], &mut state).unwrap();
    assert_eq!(result, Value::Null);
}

#[test]
fn standard_builtin_arguments_stay_inline_and_preserve_stack_order() {
    let mut stack = smallvec::SmallVec::<[Value; 8]>::from_vec(vec![
        Value::text("sentinel"),
        Value::number(1.0),
        Value::number(2.0),
        Value::number(3.0),
    ]);
    let arguments = super::pop_builtin_arguments(&mut stack, 3);
    assert!(
        !arguments.spilled(),
        "common builtin arities must not allocate"
    );
    assert_eq!(
        arguments.as_slice(),
        [Value::number(1.0), Value::number(2.0), Value::number(3.0)]
    );
    assert_eq!(stack.as_slice(), [Value::text("sentinel")]);
}

#[test]
#[ignore = "focused allocation microbenchmark; run explicitly with --release --ignored"]
fn benchmark_inline_builtin_arguments_against_split_off() {
    use std::hint::black_box;
    use std::time::Instant;

    const ITERATIONS: usize = 2_000_000;
    let values = [Value::number(1.0), Value::number(2.0), Value::number(3.0)];

    let mut old_stack = Vec::with_capacity(3);
    let old_started = Instant::now();
    for _ in 0..ITERATIONS {
        old_stack.extend(values.iter().cloned());
        let start = old_stack.len() - 3;
        black_box(old_stack.split_off(start));
    }
    let old_elapsed = old_started.elapsed();

    let mut inline_stack = smallvec::SmallVec::<[Value; 8]>::new();
    let inline_started = Instant::now();
    for _ in 0..ITERATIONS {
        inline_stack.extend(values.iter().cloned());
        black_box(super::pop_builtin_arguments(&mut inline_stack, 3));
    }
    let inline_elapsed = inline_started.elapsed();

    eprintln!(
        "builtin-argument-benchmark iterations={ITERATIONS} split_off_ms={} inline_ms={} speedup={:.2}",
        old_elapsed.as_millis(),
        inline_elapsed.as_millis(),
        old_elapsed.as_secs_f64() / inline_elapsed.as_secs_f64(),
    );
}

#[test]
#[ignore = "local release microbenchmark"]
fn canonical_synthetic_builtin_release_microbenchmark() {
    const CALLS: usize = 50_000;
    let body = "/proc/max(...)\n\tvar/list/values = args\n\tif(length(args) == 1 && islist(args[1]))\n\t\tvalues = args[1]\n\tif(!length(values))\n\t\treturn null\n\tvar/result = values[1]\n\tfor(var/value in values)\n\t\tif(value > result)\n\t\t\tresult = value\n\treturn result\n";
    let caller = format!(
        "/proc/run()\n\tvar/result\n\tfor(var/i in 1 to {CALLS})\n\t\tresult = max(i, i + 1, i - 1)\n\treturn result"
    );
    let fast = synthetic_builtin_module("max", body, &caller);
    let mut bytecode = fast.clone();
    let target = bytecode.procedure_id("/proc/max@dream64_builtin").unwrap();
    bytecode.paths[target.index()] = "/proc/max@benchmark_bytecode".to_owned();

    let entry = fast.procedure_id("/proc/run").unwrap();
    let started = Instant::now();
    execute_module(&bytecode, entry, &[]).unwrap();
    let bytecode_elapsed = started.elapsed();
    let started = Instant::now();
    execute_module(&fast, entry, &[]).unwrap();
    let fast_elapsed = started.elapsed();
    eprintln!(
        "canonical synthetic max calls={CALLS} bytecode_ms={} fast_ms={} speedup={:.2}",
        bytecode_elapsed.as_millis(),
        fast_elapsed.as_millis(),
        bytecode_elapsed.as_secs_f64() / fast_elapsed.as_secs_f64(),
    );
}

/// Guards the primitives the tgstation timsort port leans on: an unset/`FALSE`
/// parameter is falsy in the `fetchElement` ternary, `list.Copy()` preserves
/// length, in-bounds `L[i] = x` does not resize, and a plain `L[i]` never
/// degrades into the associative `L[L[i]]` form.
#[test]
fn timsort_ternary_copy_and_index_primitives_are_sound() {
    let source = concat!(
        "/proc/ternary_unset(associative)\n",
        "\treturn (associative) ? 100 : 200\n",
        "/proc/param_default(x = 5)\n",
        "\treturn x\n",
        "/proc/copy_len()\n",
        "\tvar/list/a = list(1, 2, 3, 4, 5)\n",
        "\tvar/list/b = a.Copy()\n",
        "\treturn b.len\n",
        "/proc/copy_range_len()\n",
        "\tvar/list/a = list(1, 2, 3, 4, 5)\n",
        "\tvar/list/b = a.Copy(2, 4)\n",
        "\treturn b.len\n",
        "/proc/idx_assign_len()\n",
        "\tvar/list/a = list(10, 20, 30, 40)\n",
        "\ta[2] = a[3]\n",
        "\ta[3] = a[1]\n",
        "\treturn a.len\n",
        "/proc/nested_index()\n",
        "\tvar/list/L = list(7, 2)\n",
        "\tvar/associative = null\n",
        "\treturn (associative) ? L[L[1]] : L[1]\n",
        "/proc/multi_default(a, b = 1, c = 0, d = 1, e = 0)\n",
        "\treturn (c) ? 100 : (b + d + e + 200)\n",
        "/datum/box2/proc/setup(v)\n",
        "\tsrc.assoc = v\n",
        "/datum/box2/proc/probe2(list/L)\n",
        "\treturn (src.assoc) ? L[L[1]] : L[1]\n",
        "/proc/datum_field_false_ternary()\n",
        "\tvar/datum/box2/b = new\n",
        "\tb.setup(0)\n",
        "\tvar/list/L = list(7, 2)\n",
        "\treturn b.probe2(L)\n",
    );
    let syntax = parse(source).expect("parse repro");
    let module = compile_module(&syntax.definitions).expect("compile repro");
    for (name, expected) in [
        ("/proc/ternary_unset", 200.0),
        ("/proc/param_default", 5.0),
        ("/proc/copy_len", 5.0),
        ("/proc/copy_range_len", 2.0),
        ("/proc/idx_assign_len", 4.0),
        ("/proc/nested_index", 7.0),
        ("/proc/multi_default", 202.0),
        ("/proc/datum_field_false_ternary", 7.0),
    ] {
        let id = module.procedure_id(name).unwrap();
        let got = execute_module(&module, id, &[]);
        eprintln!("{name} => {got:?} (expected {expected})");
        assert_eq!(got, Ok(Value::number(expected)), "{name}");
    }
}

/// Runs a near-verbatim port of tgstation's `/datum/sort_instance` timsort
/// (natural-run detection, `mergeCollapse`/`mergeForceCollapse`, `gallopLeft`/
/// `gallopRight`, `mergeLo`/`mergeHi`, and the `move_element`/`move_range`/
/// `reverse_range` list helpers) over numeric and datum element lists of every
/// size class and initial ordering. Regression cover for the reported
/// "DM list position N exceeds length M" fault inside `gallopRight`.
#[test]
fn timsort_full_port_merge_path_holds_bounds() {
    let source = include_str!("../../../fixtures/runtime/timsort_repro/vm_port.dm");
    let syntax = parse(source).expect("parse timsort fixture");
    let module = compile_module(&syntax.definitions).expect("compile timsort fixture");
    for entry_name in ["/proc/run_repro", "/proc/run_repro_datum"] {
        let entry = module.procedure_id(entry_name).unwrap();
        for mode in [0.0, 1.0, 2.0] {
            for count in [8.0, 33.0, 40.0, 64.0, 100.0] {
                let got =
                    execute_module(&module, entry, &[Value::number(count), Value::number(mode)]);
                eprintln!("{entry_name} mode={mode} count={count} => {got:?}");
                assert_eq!(
                    got,
                    Ok(Value::text(format!("OK len={count}"))),
                    "{entry_name} mode={mode} count={count}"
                );
            }
        }
    }
}
