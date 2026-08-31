mod common;
use common::*;

#[test]
fn typed_parameters_and_locals_allow_runtime_cross_branch_values() {
    let incompatible = TestProject::compile(
        "/proc/replace(turf/bar as turf)\n\tbar = new /obj(null)\n/proc/RunTest()\n\treturn\n",
    );
    ProcedureRegistry::build(&incompatible)
        .compile_vm(&incompatible)
        .expect("BYOND path annotations do not reject unrelated runtime assignments");

    let compatible = TestProject::compile(
        "/obj/item\n/proc/replace(obj/bar as obj)\n\tbar = new /obj/item(null)\n/proc/local()\n\tvar/obj/bar = new /obj/item\n\treturn bar\n",
    );
    ProcedureRegistry::build(&compatible)
        .compile_vm(&compatible)
        .expect("subtype construction should satisfy the declared type");
}

#[test]
fn validates_typed_sources_and_proven_datum_return_paths() {
    let dynamic_assignment = TestProject::compile(
        "/datum/base\n/obj/item\n/proc/copy()\n\tvar/datum/base/target\n\tvar/obj/item/source\n\ttarget = source\n",
    );
    ProcedureRegistry::build(&dynamic_assignment)
        .compile_vm(&dynamic_assignment)
        .expect("DreamMaker permits runtime values to flow through path annotations");

    let incompatible_return = TestProject::compile(
        "/datum/base\n/obj/item\n/proc/build() as /datum/base\n\treturn /obj/item\n",
    );
    ProcedureRegistry::build(&incompatible_return)
        .compile_vm(&incompatible_return)
        .expect("BYOND return annotations do not reject body values");

    let compatible = TestProject::compile(
        "/datum/base\n/datum/base/child\n/proc/copy() as /datum/base\n\tvar/datum/base/target\n\tvar/datum/base/child/source\n\ttarget = source\n\treturn /datum/base/child\n",
    );
    ProcedureRegistry::build(&compatible)
        .compile_vm(&compatible)
        .expect("subtype sources and return paths should satisfy base constraints");
}

#[test]
fn validates_proven_scalar_annotations_without_rejecting_null_unions() {
    let bad_local = TestProject::compile(
        "/proc/value() as num\n\tvar/const/result = \"wrong\"\n\treturn result\n",
    );
    ProcedureRegistry::build(&bad_local)
        .compile_vm(&bad_local)
        .expect("BYOND scalar return annotations do not reject body values");

    let bad_parameter =
        TestProject::compile("/proc/value(var/input = \"text\" as text) as num\n\treturn input\n");
    ProcedureRegistry::build(&bad_parameter)
        .compile_vm(&bad_parameter)
        .expect("BYOND permits dynamic scalar return values");

    let compatible = TestProject::compile(
        "/proc/number() as num\n\tvar/value = 5 as num|null\n\tvalue = null\n\tvalue = 7\n\treturn value\n/proc/text_value(var/input = \"ok\" as text) as text\n\treturn input\n",
    );
    ProcedureRegistry::build(&compatible)
        .compile_vm(&compatible)
        .expect("matching scalar annotations and nullable assignments should compile");
}

#[test]
fn inherits_override_return_constraints_and_propagates_static_call_types() {
    let inherited_mismatch = TestProject::compile(
        "/datum/proc/value() as num\n\treturn 5\n/datum/child/value()\n\treturn \"wrong\"\n",
    );
    ProcedureRegistry::build(&inherited_mismatch)
        .compile_vm(&inherited_mismatch)
        .expect("an inherited annotation constrains signatures, not body values");

    let changed_signature = TestProject::compile(
        "/datum/proc/value() as num\n\treturn 5\n/datum/child/value() as text\n\treturn \"wrong\"\n",
    );
    assert!(
        ProcedureRegistry::build(&changed_signature)
            .compile_vm(&changed_signature)
            .expect_err("override cannot replace numeric return with text")
            .message
            .contains("changes its inherited scalar return type")
    );

    let call_mismatch = TestProject::compile(
        "/proc/text_value() as text\n\treturn \"text\"\n/proc/number_value() as num\n\treturn text_value()\n",
    );
    ProcedureRegistry::build(&call_mismatch)
        .compile_vm(&call_mismatch)
        .expect("BYOND permits a differently annotated call result to be returned");

    let compatible = TestProject::compile(
        "/datum/base\n/datum/base/child\n/proc/build_child() as /datum/base/child\n\treturn /datum/base/child\n/proc/build() as /datum/base\n\treturn build_child()\n",
    );
    ProcedureRegistry::build(&compatible)
        .compile_vm(&compatible)
        .expect("statically called subtype return should satisfy base return");
}

#[test]
fn validates_scalar_field_overrides_against_late_inherited_declarations() {
    let incompatible = TestProject::compile(
        "/datum/base/child\n\tvalue = \"wrong\"\n/datum/base\n\tvar/value = 5 as num\n/proc/RunTest()\n\treturn\n",
    );
    assert!(
        ProcedureRegistry::build(&incompatible)
            .compile_vm(&incompatible)
            .expect_err("text override cannot satisfy inherited numeric field")
            .message
            .contains("field override /datum/base/child/var/value")
    );

    let compatible = TestProject::compile(
        "/datum/base/child\n\tvalue = null\n/datum/base\n\tvar/value = 5 as num|null\n/proc/RunTest()\n\treturn\n",
    );
    ProcedureRegistry::build(&compatible)
        .compile_vm(&compatible)
        .expect("nullable inherited field should accept null subtype default");
}

#[test]
fn infers_returns_from_proven_receiver_fields_and_methods() {
    let field_mismatch = TestProject::compile(
        "/datum/value_holder\n\tvar/bar = 5 as num\n\tproc/read() as text\n\t\tvar/datum/value_holder/D = new\n\t\treturn D.bar\n",
    );
    ProcedureRegistry::build(&field_mismatch)
        .compile_vm(&field_mismatch)
        .expect("typed fields remain runtime values at a return site");

    let method_mismatch = TestProject::compile(
        "/datum/producer/proc/value() as text\n\treturn \"text\"\n/proc/read() as num\n\tvar/datum/producer/P = new\n\treturn P.value()\n",
    );
    ProcedureRegistry::build(&method_mismatch)
        .compile_vm(&method_mismatch)
        .expect("typed method results remain runtime values at a return site");

    let compatible = TestProject::compile(
        "/datum/base\n/datum/base/child\n/datum/producer/proc/value() as /datum/base/child\n\treturn /datum/base/child\n/proc/read() as /datum/base\n\tvar/datum/producer/P = new\n\treturn P.value()\n",
    );
    ProcedureRegistry::build(&compatible)
        .compile_vm(&compatible)
        .expect("typed receiver method subtype should satisfy base return");
}

#[test]
fn late_base_signature_constrains_early_override_chain() {
    let compilation = TestProject::compile(
        "/datum/do/re/mi/fa/so/f()\n\treturn 5\n/datum/do/re/f()\n\treturn ..() + \" re\"\n/datum/do/re/mi/fa/f()\n\treturn ..() + \" fa\"\n/datum/do/re/mi/f()\n\treturn ..() + \" mi\"\n/datum/do/proc/f() as text\n\treturn \"do\"\n",
    );
    ProcedureRegistry::build(&compilation)
        .compile_vm(&compilation)
        .expect("late inherited signatures do not constrain override body values");
}

#[test]
fn infers_only_proven_scalar_composite_results() {
    let incompatible = TestProject::compile("/proc/ternary_value() as text\n\treturn 1 ? 2 : 3\n");
    ProcedureRegistry::build(&incompatible)
        .compile_vm(&incompatible)
        .expect("return annotations do not reject numeric ternaries");

    let list_mismatch =
        TestProject::compile("/proc/list_value() as text\n\treturn list(1, 2, 3)[1]\n");
    ProcedureRegistry::build(&list_mismatch)
        .compile_vm(&list_mismatch)
        .expect("return annotations do not reject list-index results");

    let compatible = TestProject::compile(
        "/datum/proc/value() as text\n\treturn \"base\"\n/datum/child/value()\n\treturn ..() + \" child\"\n/proc/number() as num\n\treturn (1 ? 2 : 3) + list(4, 5)[1]\n",
    );
    ProcedureRegistry::build(&compatible)
        .compile_vm(&compatible)
        .expect("matching proven composites should compile");
}

#[test]
fn mutable_unannotated_locals_do_not_acquire_static_scalar_types() {
    let compilation =
        TestProject::compile("/datum/proc/foo() as num\n\tvar/meep = 5\n\treturn meep\n");
    ProcedureRegistry::build(&compilation)
        .compile_vm(&compilation)
        .expect("dynamic locals may be returned from annotated procedures");
}

#[test]
fn comma_grouped_bare_locals_do_not_form_a_fake_type_path() {
    let compilation = TestProject::compile(
        "/proc/is_guest_key(key)\n\tvar/i, ch, len = 3\n\ti = 1\n\tch = 2\n\treturn i + ch + len\n",
    );
    ProcedureRegistry::build(&compilation)
        .compile_vm(&compilation)
        .expect("grouped bare locals must remain independent untyped declarations");
}

#[test]
fn narrows_truthy_ternaries_and_invalidates_facts_on_dynamic_writes() {
    let narrowed = TestProject::compile(
        "/datum/test1\n/datum/test2/proc/meep() as num\n\treturn 5\n/datum/test3/proc/meep() as text\n\treturn \"bad\"\n/proc/read() as num\n\tvar/datum/test1/T1 = new\n\tvar/datum/test2/T2 = new\n\tvar/datum/test3/T3 = new\n\treturn (T1 ? T2 : T3).meep()\n",
    );
    ProcedureRegistry::build(&narrowed)
        .compile_vm(&narrowed)
        .expect("a local initialized with new is proven truthy");

    let invalidated = TestProject::compile(
        "/datum/test1\n/datum/test2/proc/meep() as num\n\treturn 5\n/datum/test3/proc/meep() as text\n\treturn \"bad\"\n/proc/read(value) as num\n\tvar/datum/test1/T1 = new\n\tvar/datum/test2/T2 = new\n\tvar/datum/test3/T3 = new\n\tT1 = value\n\treturn (T1 ? T2 : T3).meep()\n",
    );
    ProcedureRegistry::build(&invalidated)
        .compile_vm(&invalidated)
        .expect("an unknown write invalidates the truth fact and stays unchecked");
}

#[test]
fn proven_untyped_local_alias_narrows_until_dynamic_reassignment() {
    let compilation = TestProject::compile(
        "var/global/datum/globals/GLOB\n/datum/globals/var/datum/log_holder/logger\n/proc/narrowed()\n\tvar/alias = GLOB.logger\n\treturn alias.Log()\n/proc/invalidated(value)\n\tvar/alias = GLOB.logger\n\talias = value\n\treturn alias.Log()\n/datum/log_holder/proc/Log()\n\treturn 7\n/datum/unrelated/proc/Log()\n\treturn 100\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let log = procedure_by_path(&registry, "/datum/log_holder/proc/Log")
        .effective_target
        .unwrap();
    let unrelated = procedure_by_path(&registry, "/datum/unrelated/proc/Log")
        .effective_target
        .unwrap();
    let narrowed = procedure_by_path(&registry, "/proc/narrowed")
        .effective_target
        .unwrap();
    let closure = registry.implementation_closure(&compilation, [narrowed]);
    assert!(closure.contains(&log));
    assert!(!closure.contains(&unrelated));

    let invalidated = procedure_by_path(&registry, "/proc/invalidated")
        .effective_target
        .unwrap();
    let closure = registry.implementation_closure(&compilation, [invalidated]);
    assert!(closure.contains(&log));
    assert!(closure.contains(&unrelated));
}

#[test]
fn typed_procedure_return_chain_narrows_member_dispatch() {
    let compilation = TestProject::compile(
        "/proc/get_logger() as /datum/log_holder\n\treturn null\n/datum/provider/proc/get_logger() as /datum/log_holder\n\treturn null\n/proc/from_global()\n\treturn get_logger()?.Log()\n/datum/provider/proc/from_member()\n\treturn (src.get_logger()).Log()\n/datum/log_holder/proc/Log()\n\treturn 7\n/datum/unrelated/proc/Log()\n\treturn 100\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let unrelated = procedure_by_path(&registry, "/datum/unrelated/proc/Log")
        .effective_target
        .unwrap();
    for path in ["/proc/from_global", "/datum/provider/proc/from_member"] {
        let entry = procedure_by_path(&registry, path).effective_target.unwrap();
        let closure = registry.implementation_closure(&compilation, [entry]);
        assert!(
            !closure.contains(&unrelated),
            "typed return receiver in {path} must not link unrelated Log"
        );
    }
}

#[test]
fn unqualified_istype_uses_typed_src_field_declarations() {
    let compilation = TestProject::compile(
        "/obj/item\n/obj/item/space\n/obj/item/explorer\n/datum/holder\n\tvar/obj/item/space/suit\n\tproc/check()\n\t\treturn istype(suit)\n/proc/run()\n\tvar/datum/holder/holder = new\n\tholder.suit = new /obj/item/explorer\n\tvar/incompatible = holder.check()\n\tholder.suit = new /obj/item/space\n\treturn incompatible * 10 + holder.check()\n",
    );
    assert_eq!(
        execute_effective(&compilation, "/proc/run", &[]),
        Ok(Value::number(1.0)),
    );
}
