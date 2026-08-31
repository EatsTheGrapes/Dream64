mod common;
use common::*;

#[test]
fn resolves_upward_search_path_expressions_before_vm_lowering() {
    let cases = [
        (
            "deep search",
            "/datum/a/b/c\n/datum/d\n/proc/RunTest()\n\treturn /datum/a/b/c.d\n",
            "/proc/RunTest",
            "/datum/d",
        ),
        (
            "procedure namespace",
            "/atom/proc/fn()\n\treturn\n/proc/RunTest()\n\treturn /atom./proc/fn\n",
            "/proc/RunTest",
            "/atom/proc/fn",
        ),
        (
            "contextual search",
            "/datum/foo\n/datum/bar/proc/find()\n\treturn .foo\n",
            "/datum/bar/proc/find",
            "/datum/foo",
        ),
        (
            "empty suffix",
            "/datum/foo\n/proc/RunTest()\n\treturn /datum/foo.\n",
            "/proc/RunTest",
            "/datum/foo",
        ),
    ];

    for (label, source, procedure, expected) in cases {
        let compilation = TestProject::compile(source);
        assert_eq!(
            execute_effective(&compilation, procedure, &[]),
            Ok(Value::TypePath(TypePath::parse(expected).unwrap())),
            "{label}"
        );
    }
}

#[test]
fn lowers_bare_inherited_fields_as_src_fields_after_local_resolution() {
    let compilation = TestProject::compile(
        "/datum/base\n\tvar/loading_id = 1\n/datum/base/child\n\tproc/run(loading_id)\n\t\tvar/local = loading_id\n\t\tloading_id = local\n\t\treturn src.loading_id\n/datum/base/child\n\tproc/use_field()\n\t\tloading_id += 1\n\t\treturn loading_id\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let procedure = procedure_by_path(&registry, "/datum/base/child/proc/use_field");
    let executable = registry
        .compile_vm_implementations(
            &compilation,
            procedure.implementations.iter().map(|body| body.id),
        )
        .expect("bare inherited field should compile");
    let entry = executable
        .implementation(procedure.effective_target.expect("procedure has a body"))
        .expect("implementation should be present");
    let program = executable
        .module()
        .procedure(entry)
        .expect("program should exist");

    assert!(program.instructions.windows(3).any(|instructions| matches!(
        instructions,
        [Instruction::LoadSrc, Instruction::Duplicate, Instruction::LoadField(field)]
            if field.as_str() == "loading_id"
    )));
    assert!(program.instructions.iter().any(|instruction| matches!(
        instruction,
        Instruction::StoreField(field) if field.as_str() == "loading_id"
    )));
}

#[test]
fn lowers_existing_instance_field_as_undeclared_for_loop_target() {
    let compilation = TestProject::compile(
        "/obj/machine\n\tvar/cointype = /obj/coin\n\tInitialize()\n\t\tfor(cointype in typesof(/obj/coin))\n\t\t\tvar/obj/coin/value = new cointype\n/obj/coin\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let procedure = procedure_by_path(&registry, "/obj/machine/proc/Initialize");
    registry
        .compile_vm_implementations(
            &compilation,
            procedure.implementations.iter().map(|body| body.id),
        )
        .expect("an undeclared for target should bind an existing src field");
}

#[test]
fn lowers_standard_atom_fields_only_for_their_builtin_hierarchy() {
    let compilation = TestProject::compile(
        "/atom/proc/offsets()\n\tif(pixel_x == 0 && pixel_y == 0)\n\t\treturn list(pixel_w, pixel_z)\n/obj/example\n\tproc/read()\n\t\tloc = src\n\t\tpixel_x += 1\n\t\talpha -= 1\n\t\treturn list(dir, color, desc, blend_mode, alpha, appearance_flags, layer, plane, transform, overlays, underlays, vis_contents, vis_locs, x, y, z)\n\tDestroy()\n\t\tvis_locs = null\n\t\tif(length(vis_contents))\n\t\t\tvis_contents.Cut()\n/datum/example\n\tproc/read()\n\t\treturn alpha\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let object = procedure_by_path(&registry, "/obj/example/proc/read");
    registry
        .compile_vm_implementations(
            &compilation,
            object.implementations.iter().map(|body| body.id),
        )
        .expect("standard atom fields should compile as src fields");
    let destroy = procedure_by_path(&registry, "/obj/example/proc/Destroy");
    registry
        .compile_vm_implementations(
            &compilation,
            destroy.implementations.iter().map(|body| body.id),
        )
        .expect("vis_locs and vis_contents should bind in Destroy");
    let atom = procedure_by_path(&registry, "/atom/proc/offsets");
    registry
        .compile_vm_implementations(
            &compilation,
            atom.implementations.iter().map(|body| body.id),
        )
        .expect("pixel offsets are engine fields on /atom itself");

    let datum = procedure_by_path(&registry, "/datum/example/proc/read");
    let error = registry
        .compile_vm_implementations(
            &compilation,
            datum.implementations.iter().map(|body| body.id),
        )
        .expect_err("atom fields must not become datum locals");
    assert!(
        error.message.contains("unknown local \"alpha\""),
        "unexpected diagnostic: {}",
        error.message
    );
}

#[test]
fn switch_arm_local_does_not_hide_documented_atom_and_particle_fields() {
    let compilation = TestProject::compile(concat!(
        "/obj/machinery/chem_recipe_debug/proc/ui_act(action)\n",
        "\tswitch(action)\n",
        "\t\tif(\"setTargetList\")\n",
        "\t\t\tvar/text = \"local\"\n",
        "\t\t\tif(!text)\n",
        "\t\t\t\treturn 1\n",
        "\t\tif(\"setEdit\")\n",
        "\t\t\tif(!text)\n",
        "\t\t\t\treturn 2\n",
        "/particles/proc/return_ui_representation()\n",
        "\treturn color_change\n",
    ));
    let registry = ProcedureRegistry::build(&compilation);
    let ui_act = procedure_by_path(&registry, "/obj/machinery/chem_recipe_debug/proc/ui_act");
    let particles = procedure_by_path(&registry, "/particles/proc/return_ui_representation");
    let executable = registry
        .compile_vm_implementations(
            &compilation,
            [
                ui_act.effective_target.expect("ui_act body"),
                particles
                    .effective_target
                    .expect("particle representation body"),
            ],
        )
        .expect("BYOND engine fields should bind outside the local's lexical switch arm");

    for (target, expected_field) in [
        (ui_act.effective_target.unwrap(), "text"),
        (particles.effective_target.unwrap(), "color_change"),
    ] {
        let entry = executable
            .implementation(target)
            .expect("selected implementation should be linked");
        let program = executable
            .module()
            .procedure(entry)
            .expect("selected implementation should have bytecode");
        assert!(program.instructions.iter().any(|instruction| matches!(
            instruction,
            Instruction::LoadField(field) if field.as_str() == expected_field
        )));
    }
}

#[test]
fn mulebot_initialize_binds_deprecated_atom_suffix_field() {
    let compilation = TestProject::compile(
        "/atom\n/mob\n/mob/living\n/mob/living/simple_animal\n/mob/living/simple_animal/bot\n/mob/living/simple_animal/bot/mulebot\n\tvar/id\n\tproc/set_id(value)\n\t\tid = value\n\tInitialize(mapload)\n\t\tset_id(suffix || id || \"fallback\")\n\t\tsuffix = null\n\t\treturn suffix\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let initialize = procedure_by_path(
        &registry,
        "/mob/living/simple_animal/bot/mulebot/proc/Initialize",
    );
    let executable = registry
        .compile_vm_implementations(
            &compilation,
            initialize.implementations.iter().map(|body| body.id),
        )
        .expect("BYOND's deprecated /atom/suffix field must bind through mob inheritance");
    let entry = executable
        .implementation(initialize.effective_target.expect("Initialize has a body"))
        .expect("Initialize implementation should be compiled");
    let program = executable
        .module()
        .procedure(entry)
        .expect("Initialize program should exist");

    assert!(program.instructions.iter().any(|instruction| matches!(
        instruction,
        Instruction::LoadField(field) if field.as_str() == "suffix"
    )));
    assert!(program.instructions.iter().any(|instruction| matches!(
        instruction,
        Instruction::StoreField(field) if field.as_str() == "suffix"
    )));
}

#[test]
fn lowers_documented_image_and_mob_engine_fields() {
    let compilation = TestProject::compile(
        "/image/proc/update()\n\toverlays += src\n\tappearance_flags |= 1\n\tdir = 4\n\treturn overlays\n/mob/proc/update_vision()\n\tsight |= 1\n\tsee_invisible = 2\n\treturn sight + see_invisible + initial(sight) + initial(see_invisible)\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    for path in ["/image/proc/update", "/mob/proc/update_vision"] {
        let procedure = procedure_by_path(&registry, path);
        registry
            .compile_vm_implementations(
                &compilation,
                [procedure.effective_target.expect("procedure body")],
            )
            .unwrap_or_else(|error| panic!("{path} should bind engine fields: {error:?}"));
    }
}

#[test]
fn lowers_client_matrix_and_atom_appearance_engine_fields() {
    let compilation = TestProject::compile(
        "/client/proc/read_engine_state()\n\treturn list(connection, address, computer_id, view, screen, verbs)\n/matrix/proc/read_components()\n\treturn a + b + c + d + e + f\n/atom/proc/read_appearance()\n\treturn list(appearance, filters)\n/atom/movable/proc/read_step_offsets()\n\treturn step_x + step_y\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    for path in [
        "/client/proc/read_engine_state",
        "/matrix/proc/read_components",
        "/atom/proc/read_appearance",
        "/atom/movable/proc/read_step_offsets",
    ] {
        let procedure = procedure_by_path(&registry, path);
        registry
            .compile_vm_implementations(
                &compilation,
                [procedure.effective_target.expect("procedure body")],
            )
            .unwrap_or_else(|error| panic!("{path} should bind engine fields: {error:?}"));
    }
}

#[test]
fn client_mouse_pointer_icon_is_a_null_initialized_engine_field() {
    let compilation = TestProject::compile(
        "/client/MouseDown(value)\n\tif(initial(mouse_pointer_icon))\n\t\treturn \"unexpected initial pointer\"\n\tmouse_pointer_icon = value\n\treturn mouse_pointer_icon\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let procedure = procedure_by_path(&registry, "/client/proc/MouseDown");
    let target = procedure
        .effective_target
        .expect("MouseDown should have a body");
    let executable = registry
        .compile_vm_implementations(&compilation, [target])
        .expect("the documented client field should bind during lowering");
    let entry = executable
        .implementation(target)
        .expect("MouseDown should be linked");
    let mut state = ExecutionState::new();
    let client = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/client").unwrap());
    let pointer = Value::file("cursor.dmi");
    assert_eq!(
        execute_module_in_context(
            executable.module(),
            entry,
            std::slice::from_ref(&pointer),
            &mut state,
            &ExecutionContext::new(Value::Datum(client), Value::Null),
        ),
        Ok(pointer.clone())
    );
    assert_eq!(
        state
            .heap()
            .datum_field(client, &FieldName::parse("mouse_pointer_icon").unwrap()),
        Ok(&pointer)
    );
}

#[test]
fn lowers_standard_datum_type_field_for_all_datums() {
    let compilation =
        TestProject::compile("/datum/example\n\tproc/read()\n\t\treturn list(type, tag)\n");
    let registry = ProcedureRegistry::build(&compilation);
    let datum = procedure_by_path(&registry, "/datum/example/proc/read");
    registry
        .compile_vm_implementations(
            &compilation,
            datum.implementations.iter().map(|body| body.id),
        )
        .expect("datum type should compile as its built-in src field");
}

#[test]
fn lowers_documented_world_host_fields_as_builtin_src_fields() {
    let compilation = TestProject::compile(
        "/world/proc/read_host()\n\tworld.log = file(\"data/dd.log\")\n\treturn list(name, hub, hub_password, internet_address, address, status, port, params, log, area, mob, turf, byond_version, byond_build, cache_lifespan, executor, game_state, host, loop_checks, map_format, map_cpu, movement_mode, process, reachable, sleep_offline, tick_usage, url, version, view, visibility)\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let procedure = procedure_by_path(&registry, "/world/proc/read_host");
    registry
        .compile_vm_implementations(
            &compilation,
            [procedure.effective_target.expect("procedure body")],
        )
        .expect("documented world host fields should lower as src fields");
}

#[test]
fn lifecycle_bodies_bind_implicit_type_fields_on_every_datum_receiver() {
    let compilation = TestProject::compile(
        "/obj/example/proc/read_type()\n\treturn type\n/obj/example/proc/read_parent()\n\treturn parent_type\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let read_type = procedure_by_path(&registry, "/obj/example/proc/read_type");
    let read_parent = procedure_by_path(&registry, "/obj/example/proc/read_parent");
    let executable = registry
        .compile_vm_implementations(
            &compilation,
            [
                read_type.effective_target.unwrap(),
                read_parent.effective_target.unwrap(),
            ],
        )
        .expect("implicit datum type fields should compile");
    assert_eq!(executable.stats().src_field_bindings, 2);
}

#[test]
fn typed_atom_fields_are_inherited_by_obj_lifecycle_bodies() {
    let compilation = TestProject::compile(
        "/datum/reagents\n/atom\n\tvar/datum/reagents/reagents = null\n/obj/item/example/proc/Initialize()\n\treturn reagents\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let initialize = procedure_by_path(&registry, "/obj/item/example/proc/Initialize");
    let executable = registry
        .compile_vm_implementations(&compilation, [initialize.effective_target.unwrap()])
        .expect("typed /atom fields should be inherited by /obj procedures");
    assert_eq!(executable.stats().src_field_bindings, 1);
}

#[test]
fn interpolated_text_retains_inherited_fields_with_nested_quoted_arguments() {
    let compilation = TestProject::compile(
        r#"/datum/reagents
	var/reagent_list
/atom
	var/datum/reagents/reagents = null
/proc/pretty(value, join_text)
	return value
/obj/item/example/proc/Initialize()
	return "contents: [pretty(reagents.reagent_list, join_text = ", ")]"
"#,
    );
    let registry = ProcedureRegistry::build(&compilation);
    let initialize = procedure_by_path(&registry, "/obj/item/example/proc/Initialize");
    registry
        .compile_vm_implementations(
            &compilation,
            registry.implementation_closure(&compilation, [initialize.effective_target.unwrap()]),
        )
        .expect("interpolated inherited receiver fields should compile");
}

#[test]
fn module_specs_copy_only_bindings_referenced_by_each_body() {
    let compilation = TestProject::compile(
        "var/global/used = 4\nvar/global/unused_one = 8\nvar/global/unused_two = 16\n/datum/example\n\tvar/field_used = 3\n\tvar/field_unused = 9\n\tproc/run()\n\t\treturn used + field_used\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let run = procedure_by_path(&registry, "/datum/example/proc/run");
    let executable = registry
        .compile_vm_implementations(
            &compilation,
            [run.effective_target.expect("run implementation")],
        )
        .expect("referenced bindings compile");

    assert_eq!(executable.stats().global_field_bindings, 1);
    assert_eq!(executable.stats().src_field_bindings, 1);
    assert_eq!(executable.stats().static_registry_builds, 1);
    let entry = executable
        .implementation(run.effective_target.expect("run implementation"))
        .expect("run is linked");
    let mut state = ExecutionState::new();
    state.set_global(
        dm_value::FieldName::parse("used").unwrap(),
        Value::number(4.0),
    );
    let datum = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/example").unwrap());
    state
        .heap_mut()
        .set_datum_field(
            datum,
            dm_value::FieldName::parse("field_used").unwrap(),
            Value::number(3.0),
        )
        .unwrap();
    assert_eq!(
        execute_module_in_context(
            executable.module(),
            entry,
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(datum), Value::Null),
        ),
        Ok(Value::number(7.0))
    );
}

#[test]
fn module_binding_lookup_work_depends_on_references_not_global_inventory() {
    let compile = |unused_globals: usize| {
        let mut source = "var/global/used = 4\n/proc/run()\n\treturn used\n".to_owned();
        for index in 0..unused_globals {
            source.push_str(&format!("var/global/unused_{index} = {index}\n"));
        }
        let compilation = TestProject::compile(&source);
        let registry = ProcedureRegistry::build(&compilation);
        let run = procedure_by_path(&registry, "/proc/run")
            .effective_target
            .expect("run implementation");
        let executable = registry
            .compile_vm_implementations(&compilation, [run])
            .expect("run should link");
        (
            executable.stats().global_binding_index_lookups,
            executable.stats().typed_global_index_lookups,
            executable.stats().global_field_bindings,
        )
    };

    assert_eq!(
        compile(2),
        compile(200),
        "unreferenced project globals must not increase per-body binding work"
    );
}

#[test]
fn inherited_field_binding_work_depends_on_references_not_owner_inventory() {
    let compile = |unused_fields: usize| {
        let mut source = "/datum/base\n\tvar/used = 4\n".to_owned();
        for index in 0..unused_fields {
            source.push_str(&format!("\tvar/unused_{index} = {index}\n"));
        }
        source.push_str("/datum/base/child/proc/read()\n\treturn used\n");
        let compilation = TestProject::compile(&source);
        let registry = ProcedureRegistry::build(&compilation);
        let read = procedure_by_path(&registry, "/datum/base/child/proc/read")
            .effective_target
            .expect("read implementation");
        let executable = registry
            .compile_vm_implementations(&compilation, [read])
            .expect("inherited field should link");
        (
            executable.stats().inherited_field_name_lookups,
            executable.stats().src_field_bindings,
        )
    };

    assert_eq!(compile(2), compile(200));
}
