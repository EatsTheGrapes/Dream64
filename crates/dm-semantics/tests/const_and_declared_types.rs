mod common;
use common::*;

#[test]
fn rejects_writes_to_global_local_and_inherited_const_variables() {
    let cases = [
        (
            "global",
            "var/const/answer = 42\n/proc/RunTest()\n\tanswer = 7\n",
            "answer",
        ),
        (
            "local",
            "/proc/RunTest()\n\tvar/const/answer = 42\n\tanswer += 1\n",
            "answer",
        ),
        (
            "inherited",
            "/datum/base\n\tvar/const/answer = 42\n/datum/base/child\n\tproc/change()\n\t\tanswer = 7\n",
            "answer",
        ),
        (
            "prefix mutation",
            "/proc/RunTest()\n\tvar/const/answer = 42\n\t++answer\n",
            "answer",
        ),
        (
            "typed local receiver",
            "/obj\n\tvar/const/answer = 42\n/proc/RunTest()\n\tvar/obj/o = new\n\to.answer = 7\n",
            "answer",
        ),
        (
            "typed field receiver",
            "/obj\n\tvar/const/answer = 42\n/datum/holder\n\tvar/obj/item\n\tproc/change()\n\t\titem.answer = 7\n",
            "answer",
        ),
    ];

    for (label, source, name) in cases {
        let compilation = TestProject::compile(source);
        let error = ProcedureRegistry::build(&compilation)
            .compile_vm(&compilation)
            .expect_err("const assignment should fail compilation");
        assert!(
            error.message.contains(&format!("const variable `{name}`")),
            "{label}: {}",
            error.message
        );
    }
}

#[test]
fn mutable_local_shadowing_a_const_field_remains_assignable() {
    let compilation = TestProject::compile(
        "/datum/base\n\tvar/const/answer = 42\n\tproc/read()\n\t\tvar/answer = 1\n\t\tanswer = 2\n\t\treturn answer\n",
    );
    ProcedureRegistry::build(&compilation)
        .compile_vm(&compilation)
        .expect("mutable local should shadow the const field");
}

#[test]
fn typed_receiver_const_check_does_not_guess_from_the_field_name() {
    let compilation = TestProject::compile(
        "/obj\n\tvar/const/answer = 42\n/datum/other\n\tvar/answer = 1\n/proc/RunTest()\n\tvar/datum/other/o = new\n\to.answer = 7\n",
    );
    ProcedureRegistry::build(&compilation)
        .compile_vm(&compilation)
        .expect("the proven receiver field is mutable despite a same-name const elsewhere");
}

#[test]
fn rejects_unknown_declared_types_without_confusing_type_named_fields() {
    let unknown = TestProject::compile(
        "/datum/later\n\tvar/datum/laterrr/aa = new(0)\n/proc/RunTest()\n\treturn\n",
    );
    assert!(
        ProcedureRegistry::build(&unknown)
            .compile_vm(&unknown)
            .expect_err("unknown field declaration type must be rejected")
            .message
            .contains("unknown declared type `/datum/laterrr`")
    );

    let name_clash = TestProject::compile(
        "var/datum/later/later\n/datum/later\n\tvar/datum/later/later = new(0)\n/proc/RunTest()\n\treturn\n",
    );
    ProcedureRegistry::build(&name_clash)
        .compile_vm(&name_clash)
        .expect("a field may have the same name as its declared type");

    let typed_list = TestProject::compile(
        "/datum/item\n/datum/holder\n\tvar/final/list/datum/item/items = list()\n/proc/RunTest()\n\treturn\n",
    );
    ProcedureRegistry::build(&typed_list)
        .compile_vm(&typed_list)
        .expect("a typed-list element path must not be treated as a /list subtype");

    let project_descendant = TestProject::compile(
        "/obj/item/weapon\n/datum/holder\n\tvar/obj/item/weapon/gun/ballistic/owner_gun\n/proc/RunTest()\n\treturn\n",
    );
    ProcedureRegistry::build(&project_descendant)
        .compile_vm(&project_descendant)
        .expect("BYOND accepts an annotated descendant beneath a project-defined type");

    let unresolved_annotation = TestProject::compile(
        "/datum/holder\n\tvar/datum/forward_declared_later/value\n/proc/RunTest()\n\treturn\n",
    );
    ProcedureRegistry::build(&unresolved_annotation)
        .compile_vm(&unresolved_annotation)
        .expect("BYOND accepts unresolved field annotations without an initializer");

    let unknown_local = TestProject::compile("/proc/read()\n\tvar/datum/missing/value\n\treturn\n");
    assert!(
        ProcedureRegistry::build(&unknown_local)
            .compile_vm(&unknown_local)
            .expect_err("unknown local declaration type must be rejected")
            .message
            .contains("unknown declared type `/datum/missing`")
    );

    let unknown_parameter = TestProject::compile("/proc/read(var/datum/missing/value)\n\treturn\n");
    assert!(
        ProcedureRegistry::build(&unknown_parameter)
            .compile_vm(&unknown_parameter)
            .expect_err("unknown parameter declaration type must be rejected")
            .message
            .contains("unknown declared type `/datum/missing`")
    );

    let unknown_return = TestProject::compile("/proc/read() as /datum/missing\n\treturn null\n");
    assert!(
        ProcedureRegistry::build(&unknown_return)
            .compile_vm(&unknown_return)
            .expect_err("unknown procedure return type must be rejected")
            .message
            .contains("unknown declared procedure return type `/datum/missing`")
    );
}

#[test]
fn accepts_remaining_declared_type_inference_shapes() {
    let cases = [
        (
            "inherited typed field override",
            "/datum/test/thing\n\tvar/list/foo = list()\n/datum/test/thing/stuff\n\tfoo = new()\n/proc/RunTest()\n\treturn\n",
        ),
        (
            "nested list assignment",
            "/proc/RunTest()\n\tvar/list/L = list()\n\tL[new()] = new()\n\treturn\n",
        ),
        (
            "late derived field",
            "/datum/later\n\tvar/datum/pointless_base/a\n/datum/pointless_base/derived/var/x = 7\n/proc/RunTest()\n\tvar/datum/later/L = new\n\tL.a = new /datum/pointless_base/derived()\n\treturn\n",
        ),
        (
            "BYOND input-qualified parameters",
            "/area/target\n/datum/thing\n/mob/player\n/client/proc/jump_to_area(area/target in world)\n\treturn target\n/client/proc/debug_variables(datum/thing in world)\n\treturn thing\n/proc/togglebuildmode(mob/M in global.player_list)\n\treturn M\n",
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
fn union_as_annotation_does_not_form_a_fake_type_path() {
    let compilation =
        TestProject::compile("/proc/run(var/atom/location as mob|obj|turf)\n\treturn location\n");
    ProcedureRegistry::build(&compilation)
        .compile_vm(&compilation)
        .expect("DM union annotation must remain valid");
}

#[test]
fn input_constraints_do_not_form_fake_declared_types() {
    let compilation = TestProject::compile(
        "/proc/plain(message as message)\n\treturn message\n/proc/typed(mob/M as mob in world)\n\treturn M\n/proc/untyped(target as turf in world)\n\treturn target\n",
    );
    ProcedureRegistry::build(&compilation)
        .compile_vm(&compilation)
        .expect("input constraints must not be interpreted as datum paths");
}

#[test]
fn suffix_array_local_is_a_list_not_a_fake_declared_type() {
    let compilation = TestProject::compile(
        "/area/misc/hilbertshotel/proc/storeRoom(roomSize)\n\tvar/storage[roomSize]\n\treturn storage\n",
    );
    ProcedureRegistry::build(&compilation)
        .compile_vm(&compilation)
        .expect("suffix array local must compile as a list");
}
