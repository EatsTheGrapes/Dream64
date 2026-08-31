mod common;
use common::*;

#[test]
fn typed_receiver_static_access_uses_one_inherited_qualified_slot() {
    let compilation = TestProject::compile(
        "/datum/base\n\tvar/static/shared = 3\n/datum/child\n\tparent_type = /datum/base\n/proc/read(var/datum/child/other)\n\tother.shared = 9\n\treturn initial(other.shared)\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let executable = registry
        .compile_vm(&compilation)
        .expect("typed static member access");
    let procedure = procedure_by_path(&registry, "/proc/read")
        .effective_target
        .and_then(|id| executable.implementation(id))
        .expect("read implementation");
    let instructions = &executable
        .module()
        .procedure(procedure)
        .expect("program")
        .instructions;
    let stores = instructions
        .iter()
        .filter_map(|instruction| match instruction {
            dm_vm::Instruction::StoreGlobal(field) => Some(field.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let initials = instructions
        .iter()
        .filter_map(|instruction| match instruction {
            dm_vm::Instruction::LoadInitialGlobal(field) => Some(field.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(stores.len(), 1);
    assert_eq!(initials, stores);
}

#[test]
fn typed_for_in_receiver_reads_inherited_static_list_by_shared_identity() {
    let compilation = TestProject::compile(
        "/datum/bodypart_overlay\n\tvar/static/list/all_layers = list(1, 2, 4)\n/datum/bodypart_overlay/mutant\n/proc/read_layers(list/bodypart_overlays)\n\tvar/list/first\n\tfor(var/datum/bodypart_overlay/overlay as anything in bodypart_overlays)\n\t\tfirst = overlay.all_layers\n\treturn first\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let entry = procedure_by_path(&registry, "/proc/read_layers")
        .effective_target
        .expect("read_layers implementation");
    let executable = registry
        .compile_vm_implementations(&compilation, [entry])
        .expect("typed loop receiver static access should lower");
    let program = executable
        .module()
        .procedure(executable.implementation(entry).unwrap())
        .expect("read_layers program");
    let storage = FieldName::static_storage("/datum/bodypart_overlay/var/all_layers");
    assert!(program.instructions.iter().any(
        |instruction| matches!(instruction, Instruction::LoadGlobal(field) if field == &storage)
    ));
    assert!(!program.instructions.iter().any(
        |instruction| matches!(instruction, Instruction::LoadField(field) if field.as_str() == "all_layers")
    ));

    let mut state = ExecutionState::new();
    let shared = state.heap_mut().allocate_list();
    for layer in [1.0, 2.0, 4.0] {
        state
            .heap_mut()
            .list_mut(shared)
            .unwrap()
            .add(Value::number(layer));
    }
    let overlay = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/datum/bodypart_overlay/mutant").unwrap());
    let overlays = state.heap_mut().allocate_list();
    state
        .heap_mut()
        .list_mut(overlays)
        .unwrap()
        .add(Value::Datum(overlay));
    state.set_global(storage, Value::List(shared));

    assert_eq!(
        execute_module_in_context(
            executable.module(),
            executable.implementation(entry).unwrap(),
            &[Value::List(overlays)],
            &mut state,
            &ExecutionContext::default(),
        ),
        Ok(Value::List(shared)),
        "every instance receiver must observe the one inherited static list",
    );
}

#[test]
fn typed_global_receiver_static_increment_uses_qualified_shared_slot() {
    let compilation = TestProject::compile(
        "var/global/datum/controller/master/Master\n/datum/controller/master\n\tvar/static/restart_count = 0\n/proc/Recreate_MC()\n\treturn ++Master.restart_count\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let executable = registry
        .compile_vm(&compilation)
        .expect("master static mutation");
    let procedure = procedure_by_path(&registry, "/proc/Recreate_MC")
        .effective_target
        .and_then(|id| executable.implementation(id))
        .expect("Recreate_MC implementation");
    let instructions = &executable
        .module()
        .procedure(procedure)
        .unwrap()
        .instructions;
    let qualified =
        dm_value::FieldName::static_storage("/datum/controller/master/var/restart_count");
    assert!(
        instructions.iter().any(|instruction| matches!(
            instruction,
            dm_vm::Instruction::MutateGlobal { name, delta: 1, prefix: true } if name == &qualified
        )),
        "{instructions:?}"
    );
    assert!(!instructions.iter().any(|instruction| matches!(
        instruction,
        dm_vm::Instruction::MutateField { name, .. } if name.as_str() == "restart_count"
    )));
}

#[test]
fn bare_type_static_in_owner_method_uses_qualified_shared_slot() {
    let compilation = TestProject::compile(
        "/datum/controller/master\n\tvar/static/random_seed\n/datum/controller/master/New()\n\tif(!random_seed)\n\t\trandom_seed = 7\n\treturn random_seed\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let executable = registry.compile_vm(&compilation).expect("master static");
    let procedure = procedure_by_path(&registry, "/datum/controller/master/proc/New")
        .effective_target
        .and_then(|id| executable.implementation(id))
        .expect("New implementation");
    let instructions = &executable
        .module()
        .procedure(procedure)
        .unwrap()
        .instructions;
    let qualified = dm_value::FieldName::static_storage("/datum/controller/master/var/random_seed");
    assert!(instructions.iter().any(
        |instruction| matches!(instruction, dm_vm::Instruction::LoadGlobal(field) if field == &qualified)
    ), "{instructions:?}");
    assert!(instructions.iter().any(
        |instruction| matches!(instruction, dm_vm::Instruction::StoreGlobal(field) if field == &qualified)
    ), "{instructions:?}");
}

#[test]
fn scope_operator_static_write_and_read_use_qualified_shared_slot() {
    // DM's `Type::name` scope operator reads and writes the type's shared
    // `static` slot directly. Both sides must resolve to the qualified
    // `__dm_static_` global, never an instance field or an `initial()` snapshot.
    let compilation = TestProject::compile(
        "/datum/gas_mixture\n\tvar/static/list/gas_meta\n/proc/meta_gas_list()\n\t/datum/gas_mixture::gas_meta = list(1, 2)\n\treturn /datum/gas_mixture::gas_meta\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let executable = registry
        .compile_vm(&compilation)
        .expect("scope-operator static access should lower");
    let procedure = procedure_by_path(&registry, "/proc/meta_gas_list")
        .effective_target
        .and_then(|id| executable.implementation(id))
        .expect("meta_gas_list implementation");
    let instructions = &executable
        .module()
        .procedure(procedure)
        .unwrap()
        .instructions;
    let qualified = FieldName::static_storage("/datum/gas_mixture/var/gas_meta");
    assert!(
        instructions.iter().any(
            |instruction| matches!(instruction, Instruction::StoreGlobal(field) if field == &qualified)
        ),
        "{instructions:?}"
    );
    assert!(
        instructions.iter().any(
            |instruction| matches!(instruction, Instruction::LoadGlobal(field) if field == &qualified)
        ),
        "{instructions:?}"
    );
    assert!(!instructions.iter().any(|instruction| matches!(
        instruction,
        Instruction::StoreField(field) | Instruction::LoadField(field) if field.as_str() == "gas_meta"
    )));
    assert!(!instructions.iter().any(|instruction| matches!(
        instruction,
        Instruction::LoadInitialGlobal(_) | Instruction::InitialField(_)
    )));
}

#[test]
fn inherited_scope_operator_static_resolves_to_ancestor_slot() {
    // `/datum/child::shared` must bind to the ancestor's declaration when the
    // static is only declared on `/datum/base`.
    let compilation = TestProject::compile(
        "/datum/base\n\tvar/static/shared = 1\n/datum/child\n\tparent_type = /datum/base\n/proc/poke()\n\t/datum/child::shared = 5\n\treturn /datum/child::shared\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let executable = registry
        .compile_vm(&compilation)
        .expect("inherited scope-operator static should lower");
    let procedure = procedure_by_path(&registry, "/proc/poke")
        .effective_target
        .and_then(|id| executable.implementation(id))
        .expect("poke implementation");
    let instructions = &executable
        .module()
        .procedure(procedure)
        .unwrap()
        .instructions;
    let qualified = FieldName::static_storage("/datum/base/var/shared");
    assert!(
        instructions.iter().any(
            |instruction| matches!(instruction, Instruction::StoreGlobal(field) if field == &qualified)
        ),
        "{instructions:?}"
    );
    assert!(
        instructions.iter().any(
            |instruction| matches!(instruction, Instruction::LoadGlobal(field) if field == &qualified)
        ),
        "{instructions:?}"
    );
}

#[test]
fn scope_operator_compound_assignment_uses_qualified_shared_slot() {
    let compilation = TestProject::compile(
        "/datum/base\n\tvar/static/tally = 0\n/proc/bump()\n\t/datum/base::tally += 3\n\treturn /datum/base::tally\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let executable = registry
        .compile_vm(&compilation)
        .expect("compound scope op");
    let procedure = procedure_by_path(&registry, "/proc/bump")
        .effective_target
        .and_then(|id| executable.implementation(id))
        .expect("bump implementation");
    let instructions = &executable
        .module()
        .procedure(procedure)
        .unwrap()
        .instructions;
    let qualified = FieldName::static_storage("/datum/base/var/tally");
    assert!(
        instructions.iter().any(
            |instruction| matches!(instruction, Instruction::StoreGlobal(field) if field == &qualified)
        ),
        "{instructions:?}"
    );
    assert!(
        instructions
            .iter()
            .filter(
                |instruction| matches!(instruction, Instruction::LoadGlobal(field) if field == &qualified)
            )
            .count()
            >= 2,
        "compound assignment loads the slot to fold, then the return reads it: {instructions:?}"
    );
    assert!(!instructions.iter().any(|instruction| matches!(
        instruction,
        Instruction::MutateField { name, .. } if name.as_str() == "tally"
    )));
}

#[test]
fn scope_operator_on_plain_instance_var_keeps_initial_semantics() {
    // A non-shared instance var accessed as `Type::name` is not writable
    // through a shared slot and reads as an `initial()` snapshot, exactly as
    // before this feature existed.
    let compilation = TestProject::compile(
        "/datum/thing\n\tvar/plain = 7\n/proc/peek()\n\treturn /datum/thing::plain\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let executable = registry
        .compile_vm(&compilation)
        .expect("scope-operator instance read should still lower");
    let procedure = procedure_by_path(&registry, "/proc/peek")
        .effective_target
        .and_then(|id| executable.implementation(id))
        .expect("peek implementation");
    let instructions = &executable
        .module()
        .procedure(procedure)
        .unwrap()
        .instructions;
    assert!(
        instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::InitialField(field) if field.as_str() == "plain")),
        "{instructions:?}"
    );
    assert!(!instructions.iter().any(|instruction| matches!(
        instruction,
        Instruction::LoadGlobal(_) | Instruction::StoreGlobal(_)
    )));
}

#[test]
fn true_instance_field_wins_over_inherited_same_name_static() {
    let compilation = TestProject::compile(
        "/datum/base\n\tvar/static/value\n/datum/base/child\n\tvar/value\n/datum/base/child/proc/Run()\n\tvalue = 4\n\treturn value\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    let executable = registry.compile_vm(&compilation).expect("field collision");
    let procedure = procedure_by_path(&registry, "/datum/base/child/proc/Run")
        .effective_target
        .and_then(|id| executable.implementation(id))
        .expect("Run implementation");
    let instructions = &executable
        .module()
        .procedure(procedure)
        .unwrap()
        .instructions;
    assert!(instructions.iter().any(
        |instruction| matches!(instruction, dm_vm::Instruction::StoreField(field) if field.as_str() == "value")
    ));
    assert!(instructions.iter().any(
        |instruction| matches!(instruction, dm_vm::Instruction::LoadField(field) if field.as_str() == "value")
    ));
}

#[test]
fn suffix_array_instance_fields_are_registered_and_inherited() {
    let compilation = TestProject::compile(
        "/datum/dna\n\tvar/mutation_index[4]\n\tproc/set_entry(index, value)\n\t\tmutation_index[index] = value\n\t\treturn mutation_index[index]\n/mob/living/carbon\n\tvar/list/overlays_standing[8]\n/mob/living/carbon/human/proc/read_overlay(index)\n\treturn overlays_standing[index]\n",
    );
    let registry = ProcedureRegistry::build(&compilation);
    for path in [
        "/datum/dna/proc/set_entry",
        "/mob/living/carbon/human/proc/read_overlay",
    ] {
        let procedure = procedure_by_path(&registry, path);
        registry
            .compile_vm_implementations(
                &compilation,
                [procedure.effective_target.expect("procedure body")],
            )
            .unwrap_or_else(|error| panic!("{path} should inherit suffix-array field: {error:?}"));
    }
}
