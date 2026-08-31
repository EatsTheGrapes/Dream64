mod common;
use common::*;

#[test]
fn function_macro_brace_blocks_keep_locals_visible_to_nested_children() {
    let compilation = TestProject::compile(
        "#define WRAP(value) \\\n\tdo {\\
\t\tif(value) {\\
\t\t\tvar/_cached_plane = value;\\
\t\t\tif(_cached_plane) {\\
\t\t\t\tvalue = _cached_plane;\\
\t\t\t}\\
\t\t}\\
\t} while(FALSE)\n\n/proc/run(value)\n\tWRAP(value)\n\treturn value\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    registry
        .compile_vm(&compilation)
        .expect("macro-expanded brace locals should compile");
}

#[test]
fn multiline_macro_brace_blocks_keep_locals_visible_to_nested_children() {
    let compilation = TestProject::compile(
        r#"#define GET_NEW_PLANE(new_value, multiplier) (blacklist?["[new_value]"] ? new_value : (new_value) - multiplier)
#define WRAP(value) \
	do {\
		if(value) {\
			var/_cached_plane = value;\
			var/turf/_our_turf = value;\
			if(_our_turf) {\
				value = GET_NEW_PLANE(_cached_plane, 1);\
			} else if(value) {\
				value = _cached_plane;\
			}\
		}\
	} while(FALSE)

/proc/run(value, blacklist)
	WRAP(value)
	return value
"#,
    );
    let registry = ProcedureRegistry::build(&compilation);
    registry
        .compile_vm(&compilation)
        .expect("macro-expanded brace locals should compile");
}

#[test]
fn typed_global_macro_declaration_is_visible_as_a_bare_global() {
    let compilation = TestProject::compile(
        r#"#define GLOBAL_REAL(X, Typepath) var/global##Typepath/##X;

GLOBAL_REAL(Master, /datum/controller/master)

/proc/run()
	return Master
"#,
    );
    let registry = ProcedureRegistry::build(&compilation);
    registry
        .compile_vm(&compilation)
        .expect("typed global declarations should resolve by bare name");
}

#[test]
fn typed_global_proc_parameters_keep_if_lines_in_the_procedure_body() {
    let compilation = TestProject::compile(
        "/proc/overwrite_field_if_available(datum/record/base, datum/record/other, field_name)\n\tif(!istype(base) || !istype(other))\n\t\treturn\n\tif(other.vars[field_name])\n\t\tbase.vars[field_name] = other.vars[field_name]\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    assert!(
        registry
            .procedures()
            .iter()
            .all(|procedure| procedure.path.to_string() != "/proc/if"),
        "if statements must not become phantom global procedures"
    );
    assert!(
        registry.procedures().iter().any(|procedure| {
            procedure.path.to_string() == "/proc/overwrite_field_if_available"
        }),
        "the typed global procedure should be indexed"
    );
}

#[test]
fn proc_pseudo_macro_is_the_current_canonical_procedure_reference() {
    let compilation = TestProject::compile(
        "/datum/example/proc/reenter(again)\n\tif(again)\n\t\treturn call(src, __PROC__)(0)\n\treturn 7\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let procedure = procedure_by_path(&registry, "/datum/example/proc/reenter");
    let target = procedure.effective_target.unwrap();
    let executable = registry
        .compile_vm_implementations(&compilation, [target])
        .expect("__PROC__ should lower as a procedure reference");
    let mut state = ExecutionState::new();
    let datum = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/example").unwrap());
    assert_eq!(
        execute_module_in_context(
            executable.module(),
            executable.implementation(target).unwrap(),
            &[Value::number(1.0)],
            &mut state,
            &ExecutionContext::new(Value::Datum(datum), Value::Null),
        ),
        Ok(Value::number(7.0))
    );
}

#[test]
fn managed_global_macro_retains_underscore_named_initializer() {
    let compilation = TestProject::compile(concat!(
        "#define GLOBAL_MANAGED(X, InitValue) /datum/controller/global_vars/proc/InitGlobal##X(){ X = InitValue; }\n",
        "#define GLOBAL_RAW(X) /datum/controller/global_vars/var/global##X\n",
        "#define GLOBAL_LIST_INIT(X, InitValue) GLOBAL_RAW(/list/##X); GLOBAL_MANAGED(X, InitValue)\n",
        "#define GLOBAL_LIST_EMPTY(X) GLOBAL_LIST_INIT(X, list())\n",
        "GLOBAL_LIST_EMPTY(all_huds)\n",
        "GLOBAL_LIST_EMPTY(huds_by_category)\n",
        "GLOBAL_LIST_INIT(huds, list(1))\n",
        "/datum/controller/global_vars/Initialize()\n",
        "\tfor(var/glob_proc in typesof(/datum/controller/global_vars/proc))\n",
        "\t\tcall(src, glob_proc)()\n",
    ));
    let registry = ProcedureRegistry::build(&compilation);
    for path in [
        "/datum/controller/global_vars/proc/InitGlobalall_huds",
        "/datum/controller/global_vars/proc/InitGlobalhuds_by_category",
        "/datum/controller/global_vars/proc/InitGlobalhuds",
    ] {
        assert!(
            registry
                .procedures()
                .iter()
                .any(|procedure| procedure.path.to_string() == path),
            "managed global macro omitted {path}"
        );
    }
}
