//! The interpreter execution loop and its instruction helpers.
//!
//! Module layout: this crate splits `execution` into `state`, `frame`, `scheduler`, `run`,
//! and `support` so the deterministic engine stays navigable by concern.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use crate::builtins;
use crate::builtins::{
    execute_external_call, execute_list_binary_operator, execute_list_compound_operator,
    execute_list_method, execute_output, execute_regex_method, execute_standard_builtin,
    execute_standard_builtin_with_usr, is_regex_datum, is_subtype,
};
use crate::bytecode::{
    CompoundAssignmentOperator, CompoundListIndexOperator, Instruction, ListEntryKind, Module,
    ProcedureId, Program, TypePredicateKind,
};
use crate::compile::{EXPANDED_ARGUMENT_COUNT, to_local_index};
use crate::value_ops::{
    allocate_dm_array, allocate_matrix, allocate_or_replace_engine_datum, allocate_vector,
    apply_icon_blend, apply_icon_map_colors, apply_icon_set_intensity, area_coordinate_field,
    assign_datum_or_shared_field, atom_contents_iteration_snapshot, block_builtin, builtin_length,
    canonicalize_owned_value, canonicalize_value, compare_values, construct_matrix,
    construct_sized_list, construct_vector, constructor_target_if_present, copy_text_builtin,
    datum_field_or_shared, datum_field_requires_special_read, deterministic_unit,
    direction_towards_builtin, dm_list_length_number, dm_list_resize_length, dynamic_call_target,
    dynamic_call_target_named_at_callsite, engine_root_initial_field_maps, engine_root_paths,
    execute_animate, execute_del, execute_icon_method, execute_matrix_binary,
    execute_matrix_compound, execute_matrix_method, execute_scalar_add,
    execute_scalar_compound_assignment, execute_vector_binary, execute_vector_compound,
    execute_vector_method, fractional_remainder, get_step_builtin, hascall_builtin,
    indexed_text_character, initial_value_or_engine_root, integer_remainder, is_area_type_path,
    is_icon_datum, is_matrix_datum, is_vector_datum, locate_in_container, locate_single,
    logical_or_empty_list_field, logical_or_empty_list_index, matrix_components,
    mutate_scalar_value, order_image_arguments, pick_value, pop, pop_builtin_arguments, pop_number,
    random_integer, range_builtin, read_list_value, ref_builtin, replace_text_builtin,
    replace_text_regex, roll_dice, round_builtin, runtime_argument_count,
    runtime_initial_field_value, runtime_truthy, savefile_current_directory,
    savefile_directory_entries, savefile_export_value, savefile_resolve_path, scalar_number_string,
    type_predicate_builtin, typesof_builtin, validate_jump, value_to_list_index, values_equal,
    values_equivalent, vector_zip, world_contents_iteration_snapshot, write_datum_vars,
    write_list_value,
};
use crate::{
    AtomsProfile, AtomsProfileInstruction, AtomsProfileProcedure, CallTrace, CompactWordcodeImage,
    ExceptionHandler, ExecutionLimits, PendingLocalPrompt, PendingPromptContinuation, RuntimeError,
    STARTUP_INSTRUCTION_CATEGORY_COUNT, ShuttleTracePostReturn, SimpleIterationValue, TgmDrive,
    TgmProfile, allocate_initialized_datum, assign_datum_field, atoms_profile_enabled,
    atoms_profile_snapshot_lines_if_due, boot_dashboard_enabled, boot_trace_enabled,
    canonical_istext, canonical_static_native_builtin, canonical_tgm_load_path,
    canonical_type2parent, canonical_type2parent_target, datum_field_or_initial,
    datum_shared_storage, drive_ruin_candidate_scan, drive_tgm_load, dynamic_call_target_named,
    emit_atoms_profile, emit_tgm_profile, engine_builtin_initial_fields,
    engine_builtin_initial_value, execute_compact_fast_instruction, false_tick_check_target,
    is_atom_type_path, is_atoms_initialize_path, is_subsystem_initialize_path,
    lazy_atom_list_field, local_prompt_spec, mark_boot_trace_frame, numeric_dispatch_candidate,
    prepare_iteration_consumes_fresh_block, register_prompt, shuttle_trace_emit_snapshot,
    shuttle_trace_enabled, shuttle_trace_prepare_call, shuttle_trace_slot_from_arguments,
    simple_iteration_field_assignment, startup_instruction_category,
    startup_instruction_profile_enabled, startup_profile_enabled, tgm_profiling_enabled,
    trace_tgm_route, try_run_build_coordinate_prefix, try_run_camera_chunk_fast_path,
    try_run_discover_offset_fast_path, try_run_dmm_preload_measurement_fast_path,
    try_run_guarded_jit, try_run_numeric_dispatch_block, try_run_numeric_local_update,
    try_run_numeric_loop_branch, try_run_parsed_dmm_new_fast_path,
    try_run_register_signal_fast_path, try_run_rooted_list_jit, try_run_ruin_affected_turfs_batch,
    try_run_tgm_build_cache_simple_member,
};
use dm_jit::NumericRunOutcome;
use dm_value::{FieldName, ModifiedTypePath, TypePath, Value, ValueError};
use smallvec::SmallVec;

use crate::execution::frame::CallFrame;
use crate::execution::frame::FrameRunOutcome;
use crate::execution::frame::InlineValueStackExt;
use crate::execution::frame::StepBudgetBehavior;
use crate::execution::frame::declared_argument_count;
use crate::execution::frame::forwarded_frame_arguments;
use crate::execution::frame::frame_context;
use crate::execution::frame::make_frame;
use crate::execution::frame::make_frame_named;
use crate::execution::frame::make_frame_owned;
use crate::execution::frame::preserve_reentrant_frame_roots;
use crate::execution::frame::synchronize_frame_argument_write;
use crate::execution::scheduler::account_scheduler_tick_usage;
use crate::execution::scheduler::materialize_callee_chain;
use crate::execution::scheduler::schedule_frames;
use crate::execution::state::ExecutionState;
use crate::value_ops::{bitwise_binary, bitwise_not, bitwise_shift};

pub(crate) fn run_frames(
    module: &Module,
    mut frames: Vec<CallFrame>,
    limits: ExecutionLimits,
    step_budget_behavior: StepBudgetBehavior,
    state: &mut ExecutionState,
) -> Result<FrameRunOutcome, RuntimeError> {
    // Observability flags are process-global and immutable after their first
    // read. Cache them once per dispatch instead of paying a OnceLock atomic
    // load on every interpreted instruction in long startup loops.
    let trace_enabled = boot_trace_enabled();
    let dashboard_enabled = boot_dashboard_enabled();
    let atoms_profiling_enabled = atoms_profile_enabled();
    let startup_profiling_enabled = startup_profile_enabled();
    let ordinary_field_fast_path_enabled =
        std::env::var_os("DREAM64_DISABLE_ORDINARY_FIELD_FAST_PATH").is_none();
    let compact_wordcode = std::env::var_os("DREAM64_DISABLE_COMPACT_WORDCODE")
        .is_none()
        .then(|| module.compact_wordcode())
        .flatten();
    let mut remaining_steps = limits.max_steps;
    let mut executed_steps = 0u64;
    let mut heartbeat = Instant::now();
    let mut instruction_batch = Instant::now();
    let mut slow_batch_report = Instant::now();
    let mut prior_instruction: Option<(Instant, ProcedureId, usize)> = None;
    let wall_clock_started = Instant::now();
    let wall_clock_budget = (step_budget_behavior
        == StepBudgetBehavior::YieldScheduledContinuation)
        .then_some(limits.wall_clock_budget)
        .flatten();
    let mut next_wall_clock_poll = 0_u64;
    // A frame retains only its stable procedure identity so scheduled continuations
    // remain self-contained. Cache the immutable program for the currently executing
    // identity within one dispatch, resolving again only after a call/return switches
    // procedures or when a continuation starts a new dispatch.
    let mut active_program: Option<(ProcedureId, &Program)> = None;
    loop {
        if let Some(budget) = wall_clock_budget
            && executed_steps >= next_wall_clock_poll
        {
            if wall_clock_started.elapsed() >= budget {
                state.maybe_collect_unreachable_lists(&frames);
                return Ok(FrameRunOutcome::Yielded { frames, delay: 0.0 });
            }
            next_wall_clock_poll = executed_steps.saturating_add(256);
        }
        if trace_enabled
            && let Some((started, prior_procedure, prior_index)) = prior_instruction.take()
            && started.elapsed().as_millis() >= 250
        {
            eprintln!(
                "boot-vm: slow-instruction elapsed_ms={} procedure={} instruction={}",
                started.elapsed().as_millis(),
                module
                    .paths
                    .get(prior_procedure.index())
                    .map_or("<missing>", String::as_str),
                prior_index,
            );
        }
        let frame_index = frames.len() - 1;
        let procedure = frames[frame_index].procedure;
        let instruction_index = frames[frame_index].instruction;
        let program = match active_program {
            Some((active_procedure, program)) if active_procedure == procedure => program,
            _ => {
                let program = module
                    .resolve_procedure(procedure)
                    .map_err(|message| execution_error(module, &frames, message))?;
                active_program = Some((procedure, program));
                program
            }
        };
        if tgm_profiling_enabled()
            && state.tgm_profile.is_none()
            && instruction_index == 0
            && canonical_tgm_load_path(module, procedure)
        {
            state.tgm_profile = Some(TgmProfile {
                started: Instant::now(),
                total_instructions: 0,
                procedure_samples: HashMap::new(),
                instruction_samples: HashMap::new(),
                paths: HashMap::new(),
                instruction_labels: HashMap::new(),
            });
            frames[frame_index].tgm_profile_root = true;
            eprintln!(
                "boot-vm: tgm-profile-begin procedure={}",
                module.procedure_path(procedure).unwrap_or("<missing>")
            );
        }
        if let Some(profile) = &mut state.tgm_profile {
            let procedure_key = AtomsProfileProcedure {
                module_identity: module.identity.0,
                procedure,
            };
            let instruction_key = AtomsProfileInstruction {
                module_identity: module.identity.0,
                procedure,
                instruction: instruction_index,
            };
            profile.total_instructions = profile.total_instructions.saturating_add(1);
            *profile.procedure_samples.entry(procedure_key).or_default() += 1;
            *profile
                .instruction_samples
                .entry(instruction_key)
                .or_default() += 1;
            profile.paths.entry(procedure_key).or_insert_with(|| {
                module
                    .procedure_path(procedure)
                    .unwrap_or("<missing>")
                    .to_owned()
            });
            profile
                .instruction_labels
                .entry(instruction_key)
                .or_insert_with(|| {
                    program
                        .instructions
                        .get(instruction_index)
                        .map_or_else(|| "<missing>".to_owned(), |value| format!("{value:?}"))
                });
        }
        trace_tgm_route(module, procedure, program, &mut frames[frame_index], state);
        if let Some(accounted_steps) = try_run_tgm_build_cache_simple_member(
            module,
            procedure,
            program,
            &mut frames[frame_index],
            remaining_steps,
            state,
        ) {
            let scheduler_batches_before = executed_steps / 4_096;
            remaining_steps = remaining_steps.saturating_sub(accounted_steps);
            executed_steps += accounted_steps;
            for _ in scheduler_batches_before..(executed_steps / 4_096) {
                account_scheduler_tick_usage(state);
            }
            if let Some(profile) = &mut state.atoms_profile {
                profile.total_instructions =
                    profile.total_instructions.saturating_add(accounted_steps);
            }
            if let Some(profile) = &mut state.tgm_profile {
                // PC98 was sampled above; retain exact aggregate logical work.
                profile.total_instructions = profile
                    .total_instructions
                    .saturating_add(accounted_steps.saturating_sub(1));
            }
            continue;
        }
        if remaining_steps >= 32
            && try_run_build_coordinate_prefix(
                module,
                procedure,
                program,
                &mut frames[frame_index],
                state,
            )
        {
            remaining_steps -= 32;
            executed_steps += 32;
            continue;
        }
        if instruction_index == 0
            && !frames[frame_index].atoms_profile_entry_counted
            && (atoms_profiling_enabled
                || startup_profiling_enabled
                || state.atoms_profile.is_some())
        {
            frames[frame_index].atoms_profile_entry_counted = true;
            let procedure_path = module.procedure_path(procedure);
            let is_atoms_root = procedure_path.is_some_and(is_atoms_initialize_path);
            let startup_root = procedure_path
                .filter(|path| is_subsystem_initialize_path(path) && startup_profiling_enabled);
            if state.atoms_profile.is_none()
                && ((is_atoms_root && atoms_profiling_enabled) || startup_root.is_some())
            {
                let started = Instant::now();
                // Keep DREAM64_PROFILE_ATOMS byte-for-byte compatible when both
                // samplers are enabled. Every other subsystem uses its canonical
                // Initialize path to identify independent snapshots.
                let startup_root = (!is_atoms_root || !atoms_profile_enabled())
                    .then(|| startup_root.map(ToOwned::to_owned))
                    .flatten();
                state.atoms_profile = Some(AtomsProfile {
                    started,
                    last_snapshot: started,
                    startup_root: startup_root.clone(),
                    total_instructions: 0,
                    instruction_categories: startup_instruction_profile_enabled()
                        .then_some([0; STARTUP_INSTRUCTION_CATEGORY_COUNT]),
                    samples: HashMap::new(),
                    wall_sample_nanos: HashMap::new(),
                    frame_entries: HashMap::new(),
                    paths: HashMap::new(),
                    instruction_samples: HashMap::new(),
                    instruction_wall_nanos: HashMap::new(),
                    instruction_labels: HashMap::new(),
                });
                frames[frame_index].atoms_profile_root = true;
                if let Some(root) = startup_root {
                    eprintln!("boot-vm: startup-profile-begin subsystem={root}");
                } else {
                    eprintln!(
                        "boot-vm: atoms-profile-begin procedure={}",
                        procedure_path.unwrap_or("<missing>")
                    );
                }
            }
            if let Some(profile) = &mut state.atoms_profile {
                let key = AtomsProfileProcedure {
                    module_identity: module.identity.0,
                    procedure,
                };
                profile.paths.entry(key).or_insert_with(|| {
                    module
                        .procedure_path(procedure)
                        .unwrap_or("<missing>")
                        .to_owned()
                });
                *profile.frame_entries.entry(key).or_default() += 1;
            }
        }
        match drive_ruin_candidate_scan(
            module,
            procedure,
            program,
            &mut frames[frame_index],
            state,
            remaining_steps,
        ) {
            TgmDrive::None => {}
            TgmDrive::Continue => {
                remaining_steps = remaining_steps.saturating_sub(1);
                executed_steps = executed_steps.saturating_add(1);
                continue;
            }
            TgmDrive::Push(child) => {
                frames.push(child);
                continue;
            }
            TgmDrive::Error(message) => {
                return Err(execution_error(module, &frames, message));
            }
        }
        match drive_tgm_load(
            module,
            procedure,
            program,
            &mut frames[frame_index],
            state,
            remaining_steps,
        ) {
            TgmDrive::None => {}
            TgmDrive::Continue => {
                remaining_steps = remaining_steps.saturating_sub(1);
                executed_steps = executed_steps.saturating_add(1);
                continue;
            }
            TgmDrive::Push(child) => {
                remaining_steps = remaining_steps.saturating_sub(1);
                executed_steps = executed_steps.saturating_add(1);
                frames.push(child);
                continue;
            }
            TgmDrive::Error(message) => {
                return Err(execution_error(module, &frames, message));
            }
        }
        // Canonical camera chunk lookup tier. The plane-offset branch contains
        // world-coordinate resolution and stays in bytecode; the ordinary
        // branch is pure coordinate bucketing plus one associative lookup.
        if instruction_index == 0
            && let Some(accounted_steps) = try_run_discover_offset_fast_path(
                module,
                procedure,
                program,
                &mut frames[frame_index],
                remaining_steps,
                state,
            )
        {
            remaining_steps = remaining_steps.saturating_sub(accounted_steps);
            executed_steps += accounted_steps;
            continue;
        }
        if instruction_index == 0
            && let Some(accounted_steps) = try_run_parsed_dmm_new_fast_path(
                module,
                procedure,
                program,
                &mut frames[frame_index],
                remaining_steps,
                state,
            )
        {
            remaining_steps = remaining_steps.saturating_sub(accounted_steps);
            executed_steps += accounted_steps;
            continue;
        }
        if instruction_index == 0
            && let Some(accounted_steps) = try_run_dmm_preload_measurement_fast_path(
                module,
                procedure,
                program,
                &mut frames[frame_index],
                remaining_steps,
                state,
            )
        {
            remaining_steps = remaining_steps.saturating_sub(accounted_steps);
            executed_steps += accounted_steps;
            continue;
        }
        if instruction_index == 0
            && let Some(accounted_steps) = try_run_camera_chunk_fast_path(
                module,
                procedure,
                program,
                &mut frames[frame_index],
                if wall_clock_budget.is_some() {
                    remaining_steps.min(256)
                } else {
                    remaining_steps
                },
                state,
            )
        {
            let scheduler_batches_before = executed_steps / 4_096;
            remaining_steps = remaining_steps.saturating_sub(accounted_steps);
            executed_steps += accounted_steps;
            for _ in scheduler_batches_before..(executed_steps / 4_096) {
                account_scheduler_tick_usage(state);
            }
            if let Some(profile) = &mut state.atoms_profile {
                profile.total_instructions =
                    profile.total_instructions.saturating_add(accounted_steps);
            }
            continue;
        }
        // Canonical DCS registration tier: batch the first-registration case
        // that dominates atom initialization. The helper side-exits before
        // mutation for every override or warning
        // path, leaving those cases to the reference bytecode interpreter.
        if instruction_index == 0
            && let Some(accounted_steps) = try_run_register_signal_fast_path(
                module,
                procedure,
                program,
                &mut frames[frame_index],
                remaining_steps,
                state,
            )
        {
            let scheduler_batches_before = executed_steps / 4_096;
            remaining_steps = remaining_steps.saturating_sub(accounted_steps);
            executed_steps += accounted_steps;
            for _ in scheduler_batches_before..(executed_steps / 4_096) {
                account_scheduler_tick_usage(state);
            }
            if let Some(profile) = &mut state.atoms_profile {
                profile.total_instructions =
                    profile.total_instructions.saturating_add(accounted_steps);
            }
            continue;
        }
        // Rooted-value tier: execute one prevalidated list-heavy basic block
        // atomically, leaving Return to the ordinary interpreter machinery.
        if instruction_index == 0
            && let Some(accounted_steps) = try_run_rooted_list_jit(
                module,
                procedure,
                program,
                &mut frames[frame_index],
                remaining_steps,
                state,
            )
        {
            let scheduler_batches_before = executed_steps / 4_096;
            remaining_steps = remaining_steps.saturating_sub(accounted_steps);
            executed_steps += accounted_steps;
            for _ in scheduler_batches_before..(executed_steps / 4_096) {
                account_scheduler_tick_usage(state);
            }
            if let Some(profile) = &mut state.atoms_profile {
                profile.total_instructions =
                    profile.total_instructions.saturating_add(accounted_steps);
            }
            continue;
        }
        // Tier zero: an entire straight-line, numeric-only procedure can bypass
        // bytecode dispatch. Runtime type guards keep general DM coercion and
        // error behavior in the reference interpreter.
        if instruction_index == 0
            && remaining_steps > 0
            && let Some((outcome, returns_null)) = try_run_guarded_jit(
                module,
                procedure,
                program,
                &mut frames[frame_index],
                remaining_steps,
                state,
            )
        {
            let (accounted_steps, result) = match outcome {
                NumericRunOutcome::Returned { value, steps } => {
                    // Native Return has no VM-visible side effect. Replay that
                    // final instruction through the ordinary arm below so call
                    // unwinding, tracing, and scheduler behavior stay unified.
                    (u64::from(steps.saturating_sub(1)), Some(value))
                }
                NumericRunOutcome::BudgetExhausted { steps, .. } => (u64::from(steps), None),
            };
            let scheduler_batches_before = executed_steps / 4_096;
            remaining_steps = remaining_steps.saturating_sub(accounted_steps);
            executed_steps += accounted_steps;
            for _ in scheduler_batches_before..(executed_steps / 4_096) {
                account_scheduler_tick_usage(state);
            }
            if let Some(profile) = &mut state.atoms_profile {
                profile.total_instructions =
                    profile.total_instructions.saturating_add(accounted_steps);
            }
            if let Some(result) = result {
                frames[frame_index].set_numeric_jit_state(None);
                frames[frame_index].stack.push(if returns_null {
                    Value::Null
                } else {
                    Value::number(result)
                });
                frames[frame_index].instruction = program.instructions.len() - 1;
            }
        }
        let numeric_loop_steps =
            (!trace_enabled && !dashboard_enabled && state.atoms_profile.is_none())
                .then(|| {
                    try_run_numeric_loop_branch(
                        program,
                        &mut frames[frame_index],
                        if wall_clock_budget.is_some() {
                            remaining_steps.min(256)
                        } else {
                            remaining_steps
                        },
                        state,
                    )
                    .or_else(|| {
                        try_run_numeric_local_update(
                            program,
                            &mut frames[frame_index],
                            if wall_clock_budget.is_some() {
                                remaining_steps.min(256)
                            } else {
                                remaining_steps
                            },
                            state,
                        )
                    })
                })
                .flatten();
        if let Some(accounted_steps) = numeric_loop_steps {
            static REPORTED: OnceLock<()> = OnceLock::new();
            REPORTED.get_or_init(|| {
                eprintln!(
                    "boot-vm: native-peephole enabled optimization=numeric-loop-superinstructions"
                );
            });
            let scheduler_batches_before = executed_steps / 4_096;
            remaining_steps -= accounted_steps;
            executed_steps += accounted_steps;
            for _ in scheduler_batches_before..(executed_steps / 4_096) {
                account_scheduler_tick_usage(state);
            }
            continue;
        }
        let ruin_batch_steps =
            (!trace_enabled && !dashboard_enabled && state.atoms_profile.is_none())
                .then(|| {
                    try_run_ruin_affected_turfs_batch(
                        module,
                        procedure,
                        program,
                        &mut frames[frame_index],
                        if wall_clock_budget.is_some() {
                            // This guarded loop is a native superinstruction.
                            // Its own bounded work quantum is independent of the
                            // legacy instruction ceiling; the outer deadline
                            // remains the production latency authority.
                            256
                        } else {
                            remaining_steps
                        },
                        state,
                    )
                })
                .flatten();
        if let Some(accounted_steps) = ruin_batch_steps {
            // Retain rich-equivalent work in the native metrics, but charge one
            // VM step under a wall-clock-bounded production run, matching the
            // way an engine builtin hides its internal host work. Non-wall runs
            // preserve exact rich instruction accounting for parity tests.
            let charged_steps = if wall_clock_budget.is_some() {
                1
            } else {
                accounted_steps
            };
            let scheduler_batches_before = executed_steps / 4_096;
            remaining_steps -= charged_steps;
            executed_steps += charged_steps;
            for _ in scheduler_batches_before..(executed_steps / 4_096) {
                account_scheduler_tick_usage(state);
            }
            continue;
        }
        let steps_to_scheduler_accounting = 4_096 - executed_steps % 4_096;
        let quick_block_budget = remaining_steps.min(steps_to_scheduler_accounting).min(256);
        let quick_block_steps = (!trace_enabled
            && !dashboard_enabled
            && state.atoms_profile.is_none()
            && program
                .instructions
                .get(frames[frame_index].instruction)
                .is_some_and(numeric_dispatch_candidate))
        .then(|| {
            try_run_numeric_dispatch_block(
                program,
                &mut frames[frame_index],
                quick_block_budget,
                state,
            )
        })
        .flatten();
        if let Some(accounted_steps) = quick_block_steps {
            static REPORTED: OnceLock<()> = OnceLock::new();
            REPORTED.get_or_init(|| {
                eprintln!("boot-vm: tier1 enabled optimization=numeric-dispatch-block");
            });
            let scheduler_batches_before = executed_steps / 4_096;
            remaining_steps -= accounted_steps;
            executed_steps += accounted_steps;
            for _ in scheduler_batches_before..(executed_steps / 4_096) {
                account_scheduler_tick_usage(state);
            }
            continue;
        }
        let instruction_index = frames[frame_index].instruction;
        let Some(instruction) = program.instructions.get(instruction_index) else {
            return Err(execution_error(
                module,
                &frames,
                "program ended without Return",
            ));
        };
        if remaining_steps == 0 {
            if step_budget_behavior == StepBudgetBehavior::YieldScheduledContinuation {
                if trace_enabled || dashboard_enabled {
                    eprintln!(
                        "boot-vm: scheduler-step-slice steps={} depth={} procedure={} instruction={}",
                        limits.max_steps,
                        frames.len(),
                        module
                            .paths
                            .get(procedure.index())
                            .map_or("<missing>", String::as_str),
                        instruction_index,
                    );
                }
                state.maybe_collect_unreachable_lists(&frames);
                return Ok(FrameRunOutcome::Yielded { frames, delay: 0.0 });
            }
            return Err(execution_error(
                module,
                &frames,
                format!("instruction budget of {} exhausted", limits.max_steps),
            ));
        }
        remaining_steps -= 1;
        executed_steps += 1;
        if let Some(profile) = &mut state.atoms_profile {
            profile.total_instructions = profile.total_instructions.saturating_add(1);
            if let Some(counts) = &mut profile.instruction_categories {
                let category = startup_instruction_category(instruction);
                counts[category] = counts[category].saturating_add(1);
            }
        }
        if executed_steps.is_multiple_of(4_096) {
            let batch_elapsed = (dashboard_enabled || state.atoms_profile.is_some())
                .then(|| instruction_batch.elapsed());
            account_scheduler_tick_usage(state);
            if let Some(profile) = &mut state.atoms_profile {
                let key = AtomsProfileProcedure {
                    module_identity: module.identity.0,
                    procedure,
                };
                profile.paths.entry(key).or_insert_with(|| {
                    module
                        .procedure_path(procedure)
                        .unwrap_or("<missing>")
                        .to_owned()
                });
                *profile.samples.entry(key).or_default() += 1;
                if let Some(batch_elapsed) = batch_elapsed {
                    *profile.wall_sample_nanos.entry(key).or_default() += batch_elapsed.as_nanos();
                }
                if profile.instruction_categories.is_some() {
                    let instruction_key = AtomsProfileInstruction {
                        module_identity: module.identity.0,
                        procedure,
                        instruction: instruction_index,
                    };
                    *profile
                        .instruction_samples
                        .entry(instruction_key)
                        .or_default() += 1;
                    if let Some(batch_elapsed) = batch_elapsed {
                        *profile
                            .instruction_wall_nanos
                            .entry(instruction_key)
                            .or_default() += batch_elapsed.as_nanos();
                    }
                    profile
                        .instruction_labels
                        .entry(instruction_key)
                        .or_insert_with(|| {
                            let span = program.source_spans.get(instruction_index).copied();
                            format!(
                                "{} instruction={} opcode={instruction:?} source={}..{}",
                                module.procedure_path(procedure).unwrap_or("<missing>"),
                                instruction_index,
                                span.map_or(0, |span| span.start),
                                span.map_or(0, |span| span.end),
                            )
                        });
                }
                if let Some(lines) = atoms_profile_snapshot_lines_if_due(
                    profile,
                    Instant::now(),
                    Duration::from_secs(60),
                ) {
                    for line in lines {
                        eprintln!("{line}");
                    }
                }
            }
            if dashboard_enabled {
                let batch_elapsed = batch_elapsed.expect("dashboard captures batch elapsed time");
                if batch_elapsed.as_millis() >= 250 && slow_batch_report.elapsed().as_secs() >= 30 {
                    let span = program.source_spans.get(instruction_index).copied();
                    eprintln!(
                        "boot-vm: slow-step-batch steps=4096 elapsed_ms={} depth={} procedure={} instruction={} source={}..{}",
                        batch_elapsed.as_millis(),
                        frames.len(),
                        module
                            .paths
                            .get(procedure.index())
                            .map_or("<missing>", String::as_str),
                        instruction_index,
                        span.map_or(0, |span| span.start),
                        span.map_or(0, |span| span.end),
                    );
                    slow_batch_report = Instant::now();
                }
                instruction_batch = Instant::now();
            } else if batch_elapsed.is_some() {
                instruction_batch = Instant::now();
            }
        }
        if trace_enabled {
            prior_instruction = Some((Instant::now(), procedure, instruction_index));
        }
        if (trace_enabled || dashboard_enabled)
            && executed_steps.is_multiple_of(1_000_000)
            && heartbeat.elapsed().as_secs() >= 30
        {
            eprintln!(
                "boot-vm: heartbeat steps={} depth={} procedure={} instruction={}",
                executed_steps,
                frames.len(),
                module
                    .paths
                    .get(procedure.index())
                    .map_or("<missing>", String::as_str),
                instruction_index,
            );
            heartbeat = Instant::now();
        }

        if let Some(compact_wordcode) = compact_wordcode {
            // Compact wordcode is a validated acceleration cache, not semantic
            // state. Runtime-appended initializer procedures can legitimately
            // be absent from an older attached image; missing coverage must
            // side-exit to the rich instruction already resolved above.
            if let Some(operation) = compact_wordcode
                .word(procedure, instruction_index)
                .and_then(CompactWordcodeImage::fast_instruction)
            {
                execute_compact_fast_instruction(operation, &mut frames[frame_index], state)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].instruction += 1;
                continue;
            }
        }

        // Local-list superinstructions keep ordinary IndexList and ListLength
        // as their single semantic implementations. Materialize the receiver
        // here without LoadLocal's redundant dispatch and canonicalization.
        let fused_list_instruction;
        let instruction = if let Instruction::IndexLocalList(slot) = instruction {
            let key = pop(&mut frames[frame_index].stack)
                .map_err(|message| execution_error(module, &frames, message))?;
            let Some(mut receiver) = frames[frame_index].locals.get(usize::from(*slot)).cloned()
            else {
                return Err(execution_error(
                    module,
                    &frames,
                    format!("invalid local slot {slot}"),
                ));
            };
            if let Value::List(reference) = receiver
                && state.reference_lists.contains(&reference)
            {
                receiver = state
                    .heap
                    .list(reference)
                    .and_then(|values| values.get(1))
                    .cloned()
                    .map_err(|error| execution_error(module, &frames, error.to_string()))?;
            }
            frames[frame_index].stack.push(receiver);
            frames[frame_index].stack.push(key);
            fused_list_instruction = Instruction::IndexList;
            &fused_list_instruction
        } else if let Instruction::ListLengthLocal(slot) = instruction {
            let Some(mut receiver) = frames[frame_index].locals.get(usize::from(*slot)).cloned()
            else {
                return Err(execution_error(
                    module,
                    &frames,
                    format!("invalid local slot {slot}"),
                ));
            };
            if let Value::List(reference) = receiver
                && state.reference_lists.contains(&reference)
            {
                receiver = state
                    .heap
                    .list(reference)
                    .and_then(|values| values.get(1))
                    .cloned()
                    .map_err(|error| execution_error(module, &frames, error.to_string()))?;
            }
            frames[frame_index].stack.push(receiver);
            fused_list_instruction = Instruction::ListLength;
            &fused_list_instruction
        } else {
            instruction
        };

        match instruction {
            Instruction::PushNull => frames[frame_index].stack.push(Value::Null),
            Instruction::PushNumber(number) => {
                frames[frame_index].stack.push(Value::Number(*number));
            }
            Instruction::PushText(text) => frames[frame_index]
                .stack
                .push(Value::Text(Arc::clone(text))),
            Instruction::PushFile(path) => {
                frames[frame_index].stack.push(Value::file(path.as_str()))
            }
            Instruction::PushTypePath(path) => {
                frames[frame_index]
                    .stack
                    .push(Value::TypePath(path.clone()));
            }
            Instruction::MakeModifiedTypePath { fields } => {
                let stack = &mut frames[frame_index].stack;
                if stack.len() < fields.len() + 1 {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let values_start = stack.len() - fields.len();
                let base_index = values_start - 1;
                let Value::TypePath(base) = stack[base_index].clone() else {
                    return Err(execution_error(
                        module,
                        &frames,
                        "modified type requires a base type path",
                    ));
                };
                let overrides = fields
                    .iter()
                    .cloned()
                    .zip(stack[values_start..].iter().cloned())
                    .collect();
                stack.truncate(base_index);
                stack.push(Value::ModifiedTypePath(Arc::new(ModifiedTypePath::new(
                    base, overrides,
                ))));
            }
            Instruction::ExpandArgumentLists {
                argument_count,
                argument_names,
                expanded_indices,
            } => {
                let count = usize::from(*argument_count);
                if count > frames[frame_index].stack.len() {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let source = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let mut expanded = Vec::new();
                let mut expanded_names = Vec::new();
                let mut expanded_roots = SmallVec::<[Value; 2]>::new();
                for (index, value) in source.into_iter().enumerate() {
                    let index = u16::try_from(index).expect("source argument count is u16");
                    if expanded_indices.binary_search(&index).is_ok() {
                        // BYOND treats `arglist(null)` as an empty argument
                        // vector. Callback.Invoke relies on this when neither
                        // its constructor nor invocation supplied arguments.
                        if matches!(value, Value::Null) {
                            continue;
                        }
                        let Value::List(list) = value else {
                            return Err(execution_error(
                                module,
                                &frames,
                                "arglist requires a list value",
                            ));
                        };
                        expanded_roots.push(Value::List(list));
                        let list = state
                            .heap
                            .list(list)
                            .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                        // OpenDream's FromArgumentList contract mirrors
                        // BYOND: associative string keys are parameter names,
                        // while ordinary entries retain their positional
                        // index. This distinction is essential for component
                        // macros, whose named arguments can be sparse and in a
                        // different order than Initialize's declaration.
                        for (_, value) in list.positions() {
                            if let Ok(associated) = list.get_key(value) {
                                let Value::Text(name) = value else {
                                    return Err(execution_error(
                                        module,
                                        &frames,
                                        "arglist contains a non-text named argument",
                                    ));
                                };
                                expanded.push(associated.clone());
                                expanded_names.push(Some(name.to_string()));
                            } else {
                                expanded.push(value.clone());
                                expanded_names.push(None);
                            }
                        }
                    } else {
                        expanded.push(value);
                        expanded_names.push(
                            argument_names
                                .get(usize::from(index))
                                .cloned()
                                .unwrap_or(None),
                        );
                    }
                }
                let expanded_count = u16::try_from(expanded.len()).map_err(|_| {
                    execution_error(
                        module,
                        &frames,
                        "expanded call has more than 65535 arguments",
                    )
                })?;
                let stack = &mut frames[frame_index].stack;
                stack.extend(expanded);
                stack.push(Value::number(f32::from(expanded_count)));
                frames[frame_index].set_pending_argument_names(expanded_names);
                frames[frame_index].set_pending_argument_roots(expanded_roots);
            }
            Instruction::AllocateDatum {
                argument_count,
                argument_names,
            } => {
                let expanded_argument_names = (*argument_count == EXPANDED_ARGUMENT_COUNT)
                    .then(|| frames[frame_index].take_pending_argument_names())
                    .flatten();
                let expanded_argument_roots = (*argument_count == EXPANDED_ARGUMENT_COUNT)
                    .then(|| frames[frame_index].take_pending_argument_roots())
                    .unwrap_or_default();
                let count_result =
                    runtime_argument_count(&mut frames[frame_index].stack, *argument_count);
                let count =
                    count_result.map_err(|message| execution_error(module, &frames, message))?;
                let stack = &mut frames[frame_index].stack;
                if stack.len() < count + 1 {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let arguments_start = stack.len() - count;
                let type_path_index = arguments_start - 1;
                let constructor_type = stack[type_path_index].clone();
                let (type_path, overrides) = match &constructor_type {
                    Value::TypePath(path) => (path.clone(), None),
                    Value::ModifiedTypePath(modified) => {
                        (modified.base().clone(), Some(modified.clone()))
                    }
                    // BYOND accepts a textual type spelling as the operand to
                    // dynamic `new`. Map-authored variables can retain this
                    // representation verbatim, as with display-case
                    // `start_showpiece_type` overrides. Resolve it through the
                    // registered runtime catalog just like text2path(); an
                    // unknown string remains an invalid constructor operand.
                    Value::Text(text) => {
                        let Some(path) = state
                            .type_paths
                            .iter()
                            .find(|path| path.as_str() == text.as_ref())
                            .cloned()
                        else {
                            return Err(execution_error(
                                module,
                                &frames,
                                format!("new requires a type path, received {constructor_type}"),
                            ));
                        };
                        (path, None)
                    }
                    _ => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("new requires a type path, received {constructor_type}"),
                        ));
                    }
                };
                let arguments = stack[arguments_start..]
                    .iter()
                    .cloned()
                    .collect::<SmallVec<[Value; 8]>>();
                stack.truncate(type_path_index);
                let is_movable = builtins::is_movable_path(type_path.as_str());
                let allocated = if type_path.as_str() == "/list" {
                    Value::List(
                        construct_sized_list(&arguments, &mut state.heap)
                            .map_err(|message| execution_error(module, &frames, message))?,
                    )
                } else {
                    let datum = if type_path.as_str() == "/matrix" {
                        construct_matrix(&arguments, &mut state.heap)
                            .map_err(|message| execution_error(module, &frames, message))?
                    } else if type_path.as_str() == "/vector" {
                        construct_vector(&arguments, &mut state.heap)
                            .map_err(|message| execution_error(module, &frames, message))?
                    } else if type_path.as_str() == "/regex" {
                        let datum = allocate_initialized_datum(state, type_path.clone())
                            .map_err(|message| execution_error(module, &frames, message))?;
                        for (name, value) in [
                            (
                                "_dream64_pattern",
                                arguments.first().cloned().unwrap_or(Value::Null),
                            ),
                            ("flags", arguments.get(1).cloned().unwrap_or(Value::Null)),
                            ("text", Value::Null),
                            ("match", Value::Null),
                            ("index", Value::number(0.0)),
                            ("group", Value::Null),
                            ("next", Value::Null),
                            ("_dream64_cursor", Value::number(0.0)),
                            ("_dream64_haystack", Value::Null),
                        ] {
                            state
                                .heap_mut()
                                .set_datum_field(
                                    datum,
                                    FieldName::parse(name).expect("regex field is valid"),
                                    value,
                                )
                                .map_err(|error| {
                                    execution_error(module, &frames, error.to_string())
                                })?;
                        }
                        datum
                    } else {
                        // Runtime field initializer programs re-enter the VM.
                        // Their collector sees the nested frames, so explicitly
                        // retain this interpreter's frames until initialization
                        // returns (notably InitAtom's reusable arglist list).
                        let root_len = preserve_reentrant_frame_roots(state, &frames);
                        let allocated =
                            allocate_or_replace_engine_datum(state, type_path.clone(), &arguments);
                        state.host_value_roots.truncate(root_len);
                        allocated.map_err(|message| execution_error(module, &frames, message))?
                    };
                    Value::Datum(datum)
                };
                if let (Value::Datum(datum), Some(modified)) = (&allocated, overrides) {
                    for (field, value) in modified.overrides() {
                        state
                            .heap_mut()
                            .set_datum_field(*datum, field.clone(), value.clone())
                            .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                    }
                }
                if let Value::Datum(datum) = &allocated {
                    if is_movable
                        && let Some(Value::Datum(location)) = arguments.first()
                        && state
                            .heap
                            .datum(*location)
                            .is_ok_and(|datum| is_atom_type_path(datum.type_path()))
                    {
                        builtins::move_movable_to_atom(state, *datum, *location)
                            .map_err(|message| execution_error(module, &frames, message))?;
                    }
                    if let Some((constructor, context)) = constructor_target_if_present(
                        module,
                        state,
                        *datum,
                        &frame_context(&frames[frame_index]),
                    ) {
                        if frames.len() >= limits.max_call_depth {
                            return Err(execution_error(
                                module,
                                &frames,
                                format!("maximum call depth {} exceeded", limits.max_call_depth),
                            ));
                        }
                        let constructor_program = module
                            .resolve_procedure(constructor)
                            .map_err(|message| execution_error(module, &frames, message))?;
                        let constructor_names = expanded_argument_names
                            .as_deref()
                            .unwrap_or(argument_names.as_slice());
                        let mut constructor_frame = if constructor_names.iter().any(Option::is_some)
                        {
                            make_frame_named(
                                constructor,
                                constructor_program,
                                &arguments,
                                constructor_names,
                                &context,
                            )
                        } else {
                            make_frame_owned(constructor, constructor_program, arguments, &context)
                        };
                        constructor_frame.set_retained_call_roots(expanded_argument_roots);
                        constructor_frame.set_caller_result_override(Some(allocated.clone()));
                        mark_boot_trace_frame(
                            &mut constructor_frame,
                            module,
                            state,
                            executed_steps,
                        );
                        frames.push(constructor_frame);
                        continue;
                    }
                }
                frames[frame_index].stack.push(allocated);
            }
            Instruction::AllocateCurrentDatum { argument_count } => {
                let expanded_argument_names = (*argument_count == EXPANDED_ARGUMENT_COUNT)
                    .then(|| frames[frame_index].take_pending_argument_names())
                    .flatten();
                let expanded_argument_roots = (*argument_count == EXPANDED_ARGUMENT_COUNT)
                    .then(|| frames[frame_index].take_pending_argument_roots())
                    .unwrap_or_default();
                let count = runtime_argument_count(&mut frames[frame_index].stack, *argument_count)
                    .map_err(|message| execution_error(module, &frames, message))?;
                if frames[frame_index].stack.len() < count {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let type_path = match frames[frame_index].src.clone() {
                    Value::Datum(datum) => state
                        .heap
                        .datum(datum)
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?
                        .type_path()
                        .clone(),
                    Value::TypePath(path) => path,
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("unqualified new requires datum src, received {value}"),
                        ));
                    }
                };
                let arguments_start = frames[frame_index].stack.len() - count;
                let arguments = frames[frame_index].stack[arguments_start..]
                    .iter()
                    .cloned()
                    .collect::<SmallVec<[Value; 8]>>();
                frames[frame_index].stack.truncate(arguments_start);
                let root_len = preserve_reentrant_frame_roots(state, &frames);
                let allocated =
                    allocate_or_replace_engine_datum(state, type_path.clone(), &arguments);
                state.host_value_roots.truncate(root_len);
                let datum =
                    allocated.map_err(|message| execution_error(module, &frames, message))?;
                if builtins::is_movable_path(type_path.as_str())
                    && let Some(Value::Datum(location)) = arguments.first()
                    && state
                        .heap
                        .datum(*location)
                        .is_ok_and(|datum| is_atom_type_path(datum.type_path()))
                {
                    builtins::move_movable_to_atom(state, datum, *location)
                        .map_err(|message| execution_error(module, &frames, message))?;
                }
                if let Some((constructor, context)) = constructor_target_if_present(
                    module,
                    state,
                    datum,
                    &frame_context(&frames[frame_index]),
                ) {
                    if frames.len() >= limits.max_call_depth {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("maximum call depth {} exceeded", limits.max_call_depth),
                        ));
                    }
                    let constructor_program = module
                        .resolve_procedure(constructor)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    let constructor_names = expanded_argument_names.as_deref().unwrap_or(&[]);
                    let mut constructor_frame = if constructor_names.iter().any(Option::is_some) {
                        make_frame_named(
                            constructor,
                            constructor_program,
                            &arguments,
                            constructor_names,
                            &context,
                        )
                    } else {
                        make_frame_owned(constructor, constructor_program, arguments, &context)
                    };
                    constructor_frame.set_retained_call_roots(expanded_argument_roots);
                    constructor_frame.set_caller_result_override(Some(Value::Datum(datum)));
                    mark_boot_trace_frame(&mut constructor_frame, module, state, executed_steps);
                    frames.push(constructor_frame);
                    continue;
                }
                frames[frame_index].stack.push(Value::Datum(datum));
            }
            Instruction::MakeRegex { argument_count } => {
                let count = usize::from(*argument_count);
                if !(1..=2).contains(&count) || frames[frame_index].stack.len() < count {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid regex constructor stack",
                    ));
                }
                let arguments = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let pattern = arguments[0].clone();
                let flags = arguments.get(1).cloned().unwrap_or(Value::Null);
                let type_path =
                    TypePath::parse("/regex").expect("the built-in regex type path is valid");
                let pattern_name = FieldName::parse("_dream64_pattern")
                    .expect("the built-in regex pattern field name is valid");
                let flags_name = FieldName::parse("flags")
                    .expect("the built-in regex flags field name is valid");
                let datum = allocate_initialized_datum(state, type_path)
                    .map_err(|message| execution_error(module, &frames, message))?;
                for (name, value) in [
                    (pattern_name, pattern),
                    (flags_name, flags),
                    (
                        FieldName::parse("text").expect("regex field is valid"),
                        Value::Null,
                    ),
                    (
                        FieldName::parse("match").expect("regex field is valid"),
                        Value::Null,
                    ),
                    (
                        FieldName::parse("index").expect("regex field is valid"),
                        Value::number(0.0),
                    ),
                    (
                        FieldName::parse("group").expect("regex field is valid"),
                        Value::Null,
                    ),
                    (
                        FieldName::parse("next").expect("regex field is valid"),
                        Value::Null,
                    ),
                    (
                        FieldName::parse("_dream64_cursor").expect("regex field is valid"),
                        Value::number(0.0),
                    ),
                    (
                        FieldName::parse("_dream64_haystack").expect("regex field is valid"),
                        Value::Null,
                    ),
                ] {
                    state
                        .heap
                        .set_datum_field(datum, name, value)
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                }
                frames[frame_index].stack.push(Value::Datum(datum));
            }
            Instruction::MakeMutableAppearance { argument_count } => {
                let count = usize::from(*argument_count);
                if frames[frame_index].stack.len() < count {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid mutable_appearance constructor stack",
                    ));
                }
                let stack = &mut frames[frame_index].stack;
                stack.truncate(stack.len() - count);
                let type_path = TypePath::parse("/mutable_appearance")
                    .expect("the built-in mutable_appearance type path is valid");
                let datum = state.heap.allocate_datum(type_path);
                stack.push(Value::Datum(datum));
            }
            Instruction::MakeMatrix { argument_count } => {
                let count = usize::from(*argument_count);
                if frames[frame_index].stack.len() < count {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid matrix constructor stack",
                    ));
                }
                let arguments = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let datum = construct_matrix(&arguments, &mut state.heap)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(Value::Datum(datum));
            }
            Instruction::MakeVector { argument_count } => {
                let count = usize::from(*argument_count);
                if frames[frame_index].stack.len() < count {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid vector constructor stack",
                    ));
                }
                let arguments = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let datum = construct_vector(&arguments, &mut state.heap)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(Value::Datum(datum));
            }
            Instruction::ReplaceText {
                argument_count,
                exact,
                character_indices,
            } => {
                let count = usize::from(*argument_count);
                if !(3..=5).contains(&count) || frames[frame_index].stack.len() < count {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid replacetext builtin stack",
                    ));
                }
                let arguments = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let value = if let Value::Datum(regex) = arguments[1]
                    && is_regex_datum(regex, state)
                {
                    let caller_context = frame_context(&frames[frame_index]);
                    let root_len = preserve_reentrant_frame_roots(state, &frames);
                    let result = replace_text_regex(
                        module,
                        state,
                        regex,
                        &arguments,
                        *character_indices,
                        &caller_context,
                    );
                    state.host_value_roots.truncate(root_len);
                    result.map_err(|message| execution_error(module, &frames, message))?
                } else {
                    replace_text_builtin(&arguments, *exact, *character_indices, &state.heap)
                        .map_err(|message| execution_error(module, &frames, message))?
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::CopyText {
                argument_count,
                character_indices,
            } => {
                let count = usize::from(*argument_count);
                if !(1..=3).contains(&count) || frames[frame_index].stack.len() < count {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid copytext builtin stack",
                    ));
                }
                let arguments = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let value = copy_text_builtin(&arguments, *character_indices, &state.heap)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(Value::text(value));
            }
            Instruction::StandardBuiltin {
                name,
                argument_count,
                argument_names,
            } => {
                let count = usize::from(*argument_count);
                if count > frames[frame_index].stack.len() {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                // Builtins overwhelmingly take only a handful of values. Keep those arguments
                // inline instead of allocating a fresh heap Vec at every call site; atoms init
                // alone executes millions of these small native dispatches.
                let arguments = pop_builtin_arguments(&mut frames[frame_index].stack, count);
                let ordered_arguments;
                let arguments: &[Value] = if name == "image" {
                    ordered_arguments = order_image_arguments(&arguments, &argument_names)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    &ordered_arguments
                } else {
                    &arguments
                };
                let usr = frames[frame_index].usr.clone();
                if let Some(prompt) = local_prompt_spec(&name, arguments, &usr, state)
                    .map_err(|message| execution_error(module, &frames, message))?
                {
                    frames[frame_index].instruction += 1;
                    state.emit_local_client_ui_event(prompt.client, prompt.event);
                    return Ok(FrameRunOutcome::Prompted {
                        id: prompt.id,
                        prompt: PendingLocalPrompt {
                            client: prompt.client,
                            kind: prompt.kind,
                            choices: prompt.choices,
                            can_cancel: prompt.can_cancel,
                            continuation: PendingPromptContinuation::Frames(frames),
                        },
                    });
                }
                let value = if name == "del" {
                    let caller_context = frame_context(&frames[frame_index]);
                    let root_len = preserve_reentrant_frame_roots(state, &frames);
                    let result = execute_del(module, &arguments, state, &caller_context);
                    state.host_value_roots.truncate(root_len);
                    result?
                } else {
                    let builtin_name = name.split_once('@').map_or(name.as_str(), |(name, _)| name);
                    execute_standard_builtin_with_usr(builtin_name, &arguments, state, &usr)
                        .map_err(|message| execution_error(module, &frames, message))?
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::NativeSrcMethod {
                name,
                argument_count,
            } => {
                let count = usize::from(*argument_count);
                if count > frames[frame_index].stack.len() {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let arguments = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let Value::Datum(src) = frames[frame_index].src else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("native method {name} requires a datum src"),
                    ));
                };
                // Parser-level recognition of engine method names cannot by
                // itself decide whether the current project type declares a
                // same-named proc. Resolve that virtual project method first;
                // only fall back to the native icon/matrix implementation
                // when the runtime type has no such proc.
                if let Ok((target, context)) = dynamic_call_target(
                    module,
                    state,
                    &Value::Datum(src),
                    &Value::text(name.as_str()),
                    &frame_context(&frames[frame_index]),
                    false,
                ) {
                    if frames.len() >= limits.max_call_depth {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("maximum call depth {} exceeded", limits.max_call_depth),
                        ));
                    }
                    let target_program = module
                        .resolve_procedure(target)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    let mut target_frame = make_frame(target, target_program, &arguments, &context);
                    mark_boot_trace_frame(&mut target_frame, module, state, executed_steps);
                    frames.push(target_frame);
                    continue;
                }
                let value = match name.as_str() {
                    "MapColors" if is_icon_datum(src, &state.heap) => {
                        apply_icon_map_colors(src, &arguments, &mut state.heap)
                            .map_err(|message| execution_error(module, &frames, message))?;
                        Value::Null
                    }
                    "Blend" if is_icon_datum(src, &state.heap) => {
                        apply_icon_blend(src, &arguments, &mut state.heap)
                            .map_err(|message| execution_error(module, &frames, message))?;
                        Value::Null
                    }
                    "SetIntensity" if is_icon_datum(src, &state.heap) => {
                        apply_icon_set_intensity(src, &arguments, &mut state.heap)
                            .map_err(|message| execution_error(module, &frames, message))?;
                        Value::Null
                    }
                    method if is_icon_datum(src, &state.heap) => {
                        execute_icon_method(src, method, &arguments, &mut state.heap)
                            .map_err(|message| execution_error(module, &frames, message))?
                    }
                    method
                        if is_matrix_datum(src, &state.heap)
                            && matches!(
                                method,
                                "Add"
                                    | "Subtract"
                                    | "Multiply"
                                    | "Scale"
                                    | "Translate"
                                    | "Turn"
                                    | "Invert"
                            ) =>
                    {
                        execute_matrix_method(src, method, &arguments, &mut state.heap)
                            .map_err(|message| execution_error(module, &frames, message))?
                    }
                    _ => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("unknown native method {name} for src"),
                        ));
                    }
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::Output => {
                let value = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let target = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                if let Value::Datum(entry) = target
                    && let Some((savefile, key)) = state.savefile_entries.get(&entry).cloned()
                {
                    state
                        .savefiles
                        .entry(savefile)
                        .or_default()
                        .entries
                        .insert(key, value);
                } else {
                    execute_output(&target, &value, state)
                        .map_err(|message| execution_error(module, &frames, message))?;
                }
            }
            Instruction::Input => {
                let target = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let value = match target {
                    Value::Datum(entry) if state.savefile_entries.contains_key(&entry) => {
                        let (savefile, key) = state.savefile_entries[&entry].clone();
                        state
                            .savefiles
                            .get(&savefile)
                            .and_then(|savefile| savefile.entries.get(&key))
                            .cloned()
                            .unwrap_or(Value::Null)
                    }
                    Value::Datum(savefile)
                        if state.heap.datum(savefile).is_ok_and(|datum| {
                            let path = datum.type_path().as_str();
                            path == "/savefile" || path.starts_with("/savefile/")
                        }) =>
                    {
                        let savefile = state.savefiles.entry(savefile).or_default();
                        let key = if savefile.cd.is_empty() {
                            "/"
                        } else {
                            &savefile.cd
                        };
                        savefile.entries.get(key).cloned().unwrap_or(Value::Null)
                    }
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("savefile input received {value}"),
                        ));
                    }
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::ExternalCall { argument_count } => {
                let count = usize::from(*argument_count) + 2;
                if count > frames[frame_index].stack.len() {
                    return Err(execution_error(
                        module,
                        &frames,
                        "external call stack underflow",
                    ));
                }
                let values = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let value = execute_external_call(&values[0], &values[1], &values[2..], state)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(value);
            }
            Instruction::Animate {
                argument_names,
                expanded_indices,
            } => {
                let count = argument_names.len();
                if count > frames[frame_index].stack.len() {
                    return Err(execution_error(module, &frames, "animate stack underflow"));
                }
                let arguments = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let mut names = Vec::new();
                let mut values = Vec::new();
                for (index, (name, value)) in argument_names.iter().zip(arguments).enumerate() {
                    if expanded_indices
                        .binary_search(
                            &to_local_index(index).expect("animate argument count is u16"),
                        )
                        .is_ok()
                    {
                        let Value::List(list) = value else {
                            return Err(execution_error(
                                module,
                                &frames,
                                "arglist requires a list value",
                            ));
                        };
                        let list = state
                            .heap
                            .list(list)
                            .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                        for (_, positional) in list.positions() {
                            names.push(None);
                            values.push(positional.clone());
                        }
                        for (key, associated) in list.associations() {
                            if let Value::Text(key) = key {
                                names.push(Some(key.to_string()));
                                values.push(associated.clone());
                            }
                        }
                    } else {
                        names.push(name.clone());
                        values.push(value);
                    }
                }
                let value = execute_animate(&names, &values, state)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(value);
            }
            Instruction::MakeFilter {
                argument_names,
                expanded_indices,
            } => {
                let count = argument_names.len();
                if count > frames[frame_index].stack.len() {
                    return Err(execution_error(module, &frames, "filter stack underflow"));
                }
                let arguments = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let filter = allocate_initialized_datum(
                    state,
                    TypePath::parse("/dm_filter").expect("canonical filter path"),
                )
                .map_err(|message| execution_error(module, &frames, message))?;
                let mut fields = Vec::new();
                for (index, (name, value)) in argument_names.iter().zip(arguments).enumerate() {
                    if expanded_indices
                        .binary_search(
                            &to_local_index(index).expect("filter argument count is u16"),
                        )
                        .is_ok()
                    {
                        let Value::List(list) = value else {
                            return Err(execution_error(
                                module,
                                &frames,
                                "arglist requires a list value",
                            ));
                        };
                        let list = state
                            .heap
                            .list(list)
                            .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                        fields.extend(list.associations().filter_map(|(key, value)| match key {
                            Value::Text(key) => Some((key.to_string(), value.clone())),
                            _ => None,
                        }));
                        continue;
                    }
                    let field = name.clone().unwrap_or_else(|| {
                        if index == 0 {
                            "type".to_owned()
                        } else {
                            format!("arg{}", index + 1)
                        }
                    });
                    fields.push((field, value));
                }
                for (field, value) in fields {
                    state
                        .heap_mut()
                        .set_datum_field(
                            filter,
                            FieldName::parse(&field).map_err(|error| {
                                execution_error(module, &frames, error.to_string())
                            })?,
                            value,
                        )
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                }
                frames[frame_index].stack.push(Value::Datum(filter));
            }
            Instruction::Sleep => {
                let delay = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let delay = delay.as_number().ok_or_else(|| {
                    execution_error(
                        module,
                        &frames,
                        format!("sleep delay must be numeric, received {delay}"),
                    )
                })?;
                frames[frame_index].stack.push(Value::Null);
                frames[frame_index].instruction += 1;
                if let Some(detach_at) = frames.iter().rposition(|frame| {
                    !frame.detached_waitfor
                        && module
                            .procedure(frame.procedure)
                            .is_some_and(|program| !program.wait_for)
                }) {
                    let detached_result = frames[detach_at]
                        .caller_result_override()
                        .cloned()
                        .unwrap_or_else(|| frames[detach_at].result.clone());
                    let mut detached = frames.split_off(detach_at);
                    detached[0].detached_waitfor = true;
                    schedule_frames(state, detached, delay);
                    if let Some(caller) = frames.last_mut() {
                        // The caller continues exactly as if the waitfor=0
                        // procedure returned its current `.` value. The
                        // detached continuation's eventual return is ignored.
                        caller.stack.push(detached_result);
                        caller.instruction += 1;
                        continue;
                    }
                    return Ok(FrameRunOutcome::Complete(detached_result));
                }
                state.maybe_collect_unreachable_lists(&frames);
                return Ok(FrameRunOutcome::Yielded { frames, delay });
            }
            Instruction::Length => {
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let length = match builtin_length(&value, &state.heap) {
                    Ok(length) => length,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                frames[frame_index].stack.push(Value::number(length));
            }
            Instruction::Ref => {
                let value = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(ref_builtin(&value));
            }
            Instruction::GetStep => {
                let direction = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let source = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let value = get_step_builtin(&source, &direction, state)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(value);
            }
            Instruction::GetStepTowards => {
                let target = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let source = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let direction = direction_towards_builtin(&source, &target, state)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let value = get_step_builtin(&source, &direction, state)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(value);
            }
            Instruction::Range { argument_count } => {
                let count = usize::from(*argument_count);
                if !(1..=2).contains(&count) || frames[frame_index].stack.len() < count {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid range builtin stack",
                    ));
                }
                let arguments = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let value = range_builtin(&arguments, &frames[frame_index].src, state)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(value);
            }
            Instruction::Block { argument_count } => {
                let count = usize::from(*argument_count);
                if !(count == 2 || (3..=6).contains(&count))
                    || frames[frame_index].stack.len() < count
                {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid block builtin stack",
                    ));
                }
                let arguments = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let value = block_builtin(&arguments, state)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(value);
            }
            Instruction::TypesOf { argument_count } => {
                let count = usize::from(*argument_count);
                if count == 0 || frames[frame_index].stack.len() < count {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid typesof builtin stack",
                    ));
                }
                let selectors = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let list = state.heap.allocate_list();
                let mut seen = BTreeSet::new();
                for selector in selectors {
                    let paths = if let Value::TypePath(root) = &selector
                        && (root.as_str() == "/proc" || root.as_str().ends_with("/proc"))
                    {
                        let prefix = format!("{}/", root.as_str());
                        module
                            .procedure_types
                            .iter()
                            .filter(|path| {
                                path.as_str() == root.as_str() || path.as_str().starts_with(&prefix)
                            })
                            .cloned()
                            .collect::<Vec<_>>()
                    } else {
                        typesof_builtin(&selector, &state.heap, &state.type_paths)
                            .map_err(|message| execution_error(module, &frames, message))?
                    };
                    for path in paths {
                        if !seen.insert(path.clone()) {
                            continue;
                        }
                        state
                            .heap
                            .list_mut(list)
                            .expect("a newly allocated list handle must be live")
                            .add(Value::TypePath(path));
                    }
                }
                frames[frame_index].stack.push(Value::List(list));
            }
            Instruction::HasCall => {
                let selector = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let receiver = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index]
                    .stack
                    .push(Value::number(f32::from(hascall_builtin(
                        module, state, &receiver, &selector,
                    ))));
            }
            Instruction::TypeInstances(target) => {
                let matches = state
                    .heap
                    .datums()
                    .filter(|(_, datum)| is_subtype(state, datum.type_path(), &target))
                    .map(|(id, _)| id)
                    .collect::<Vec<_>>();
                let list = state.heap.allocate_list();
                for datum in matches {
                    state
                        .heap
                        .list_mut(list)
                        .expect("new type-instance list is live")
                        .add(Value::Datum(datum));
                }
                frames[frame_index].stack.push(Value::List(list));
            }
            Instruction::Rand { argument_count } => {
                let count = usize::from(*argument_count);
                if count > 2 || frames[frame_index].stack.len() < count {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid rand builtin stack",
                    ));
                }
                let arguments = pop_builtin_arguments(&mut frames[frame_index].stack, count);
                let value = random_integer(&arguments, &mut state.random_state)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(Value::number(value));
            }
            Instruction::Roll { argument_count } => {
                let count = usize::from(*argument_count);
                if !(1..=2).contains(&count) || frames[frame_index].stack.len() < count {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid roll builtin stack",
                    ));
                }
                let stack_length = frames[frame_index].stack.len();
                let arguments = frames[frame_index].stack.split_off(stack_length - count);
                let value = roll_dice(&arguments, &mut state.random_state)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(Value::number(value));
            }
            Instruction::Pick { weighted } => {
                let value_count = weighted
                    .iter()
                    .map(|is_weighted| 1 + usize::from(*is_weighted))
                    .sum::<usize>();
                if value_count > frames[frame_index].stack.len() {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid pick builtin stack",
                    ));
                }
                let stack_length = frames[frame_index].stack.len();
                let values = frames[frame_index]
                    .stack
                    .split_off(stack_length - value_count);
                let value = pick_value(&values, &weighted, &state.heap, &mut state.random_state)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(value);
            }
            Instruction::PickExpandedArguments => {
                frames[frame_index].clear_pending_argument_names();
                frames[frame_index].clear_pending_argument_roots();
                let count =
                    runtime_argument_count(&mut frames[frame_index].stack, EXPANDED_ARGUMENT_COUNT)
                        .map_err(|message| execution_error(module, &frames, message))?;
                if count > frames[frame_index].stack.len() {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid expanded pick builtin stack",
                    ));
                }
                let stack_length = frames[frame_index].stack.len();
                let values = frames[frame_index].stack.split_off(stack_length - count);
                let weighted = vec![false; count];
                let value = pick_value(&values, &weighted, &state.heap, &mut state.random_state)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(value);
            }
            Instruction::Prob => {
                let chance = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let chance = chance.as_number().ok_or_else(|| {
                    execution_error(
                        module,
                        &frames,
                        format!("prob requires a number, received {chance}"),
                    )
                })?;
                let result =
                    deterministic_unit(&mut state.random_state) * 100.0 < chance.clamp(0.0, 100.0);
                frames[frame_index]
                    .stack
                    .push(Value::number(f32::from(result)));
            }
            Instruction::Round { argument_count } => {
                let count = usize::from(*argument_count);
                if !(1..=2).contains(&count) || frames[frame_index].stack.len() < count {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid round builtin stack",
                    ));
                }
                let stack = &mut frames[frame_index].stack;
                let arguments = stack.split_off(stack.len() - count);
                let value = round_builtin(&arguments)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(Value::number(value));
            }
            Instruction::TypePredicate {
                kind,
                argument_count,
            } => {
                let kind = *kind;
                let count = usize::from(*argument_count);
                let valid_count = match kind {
                    TypePredicateKind::IsType | TypePredicateKind::IsPath => {
                        (1..=2).contains(&count)
                    }
                    TypePredicateKind::IsLoc
                    | TypePredicateKind::IsMovable
                    | TypePredicateKind::IsTurf => count >= 1,
                    _ => count == 1,
                };
                if !valid_count || frames[frame_index].stack.len() < count {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid type predicate builtin stack",
                    ));
                }
                let arguments = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let result = type_predicate_builtin(kind, &arguments, state)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index]
                    .stack
                    .push(Value::number(f32::from(result)));
            }
            Instruction::MakeList(item_count) => {
                let count = usize::from(*item_count);
                let stack_length = frames[frame_index].stack.len();
                if count > stack_length {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let items = frames[frame_index].stack.split_off(stack_length - count);
                let list = state.heap.allocate_list();
                for item in items {
                    state
                        .heap
                        .list_mut(list)
                        .expect("a newly allocated list handle must be live")
                        .add(item);
                }
                frames[frame_index].stack.push(Value::List(list));
            }
            Instruction::MakeArray(dimension_count) => {
                let count = usize::from(*dimension_count);
                if frames[frame_index].stack.len() < count {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let stack_len = frames[frame_index].stack.len();
                let values = frames[frame_index].stack.split_off(stack_len - count);
                let mut sizes = Vec::with_capacity(count);
                for value in values {
                    let Some(size) = value.as_number() else {
                        return Err(execution_error(
                            module,
                            &frames,
                            "array dimension must be numeric",
                        ));
                    };
                    sizes.push(size.max(0.0).floor() as usize);
                }
                let array = allocate_dm_array(&mut state.heap, &sizes, 0);
                frames[frame_index].stack.push(Value::List(array));
            }
            Instruction::MakeArgs => {
                let list = state.heap.allocate_list();
                // `args` reflects the live formal-parameter slots. Defaults
                // and assignments performed since frame creation are visible,
                // while variadic values beyond the declared parameters remain
                // intact. OpenDream exposes the same state through
                // DMProcState.GetArguments().
                let arguments = forwarded_frame_arguments(&frames[frame_index], &program);
                for value in arguments {
                    state
                        .heap
                        .list_mut(list)
                        .expect("a newly allocated list handle must be live")
                        .add(value);
                }
                frames[frame_index].args_list = Some(list);
                frames[frame_index].stack.push(Value::List(list));
            }
            Instruction::MakeListEntries(kinds) => {
                let value_count = kinds.iter().try_fold(0_usize, |count, kind| {
                    count.checked_add(match kind {
                        ListEntryKind::Positional => 1,
                        ListEntryKind::Associative => 2,
                    })
                });
                let Some(value_count) = value_count else {
                    return Err(execution_error(
                        module,
                        &frames,
                        "list literal is too large",
                    ));
                };
                let stack_length = frames[frame_index].stack.len();
                if value_count > stack_length {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let values = frames[frame_index]
                    .stack
                    .split_off(stack_length - value_count);
                let list = state.heap.allocate_list();
                let entries = state
                    .heap
                    .list_mut(list)
                    .expect("a newly allocated list handle must be live");
                let mut values = values.into_iter();
                for kind in kinds {
                    match kind {
                        ListEntryKind::Positional => {
                            entries.add(values.next().expect("validated literal stack shape"));
                        }
                        ListEntryKind::Associative => {
                            let key = values.next().expect("validated literal stack shape");
                            let value = values.next().expect("validated literal stack shape");
                            entries.set_key(key, value);
                        }
                    }
                }
                frames[frame_index].stack.push(Value::List(list));
            }
            Instruction::MakeAssociativeListEntries(kinds) => {
                let value_count = kinds.iter().try_fold(0_usize, |count, kind| {
                    count.checked_add(match kind {
                        ListEntryKind::Positional => 1,
                        ListEntryKind::Associative => 2,
                    })
                });
                let Some(value_count) = value_count else {
                    return Err(execution_error(
                        module,
                        &frames,
                        "alist literal is too large",
                    ));
                };
                let stack_length = frames[frame_index].stack.len();
                if value_count > stack_length {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let values = frames[frame_index]
                    .stack
                    .split_off(stack_length - value_count);
                let list = state.heap.allocate_list();
                let entries = state.heap.list_mut(list).expect("new alist is live");
                let mut values = values.into_iter();
                for kind in kinds {
                    match kind {
                        ListEntryKind::Positional => {
                            let key = values.next().expect("alist entry count was validated");
                            entries.set_key(key, Value::Null);
                        }
                        ListEntryKind::Associative => {
                            let key = values.next().expect("alist key count was validated");
                            let value = values.next().expect("alist value count was validated");
                            entries.set_key(key, value);
                        }
                    }
                }
                state.mark_associative_list(list);
                frames[frame_index].stack.push(Value::List(list));
            }
            Instruction::LogicalOrEmptyListLocal(slot) => {
                let slot = *slot;
                let Some(mut current) = frames[frame_index].locals.get(usize::from(slot)).cloned()
                else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("invalid local slot {slot}"),
                    ));
                };
                if let Value::List(list) = current
                    && state.reference_lists.contains(&list)
                {
                    current = state
                        .heap
                        .list(list)
                        .and_then(|values| values.get(1))
                        .cloned()
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                }
                let value = if runtime_truthy(&state.heap, &current)
                    .map_err(|message| execution_error(module, &frames, message))?
                {
                    current
                } else {
                    let value = Value::List(state.heap.allocate_list());
                    let Some(local) = frames[frame_index].locals.get_mut(usize::from(slot)) else {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("invalid local slot {slot}"),
                        ));
                    };
                    if let Value::List(list) = local
                        && state.reference_lists.contains(list)
                    {
                        state
                            .heap
                            .list_mut(*list)
                            .and_then(|values| values.set(1, value.clone()))
                            .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                    } else {
                        *local = value.clone();
                    }
                    let parameter = usize::from(slot);
                    if parameter < declared_argument_count(program) {
                        frames[frame_index].arguments[parameter] = value.clone();
                        if let Some(args) = frames[frame_index].args_list {
                            state
                                .heap
                                .list_mut(args)
                                .and_then(|values| values.set(parameter + 1, value.clone()))
                                .map_err(|error| {
                                    execution_error(module, &frames, error.to_string())
                                })?;
                        }
                    }
                    if frames[frame_index].static_locals.contains(&slot) {
                        let path = module
                            .procedure_path(frames[frame_index].procedure)
                            .unwrap_or("<unknown procedure>")
                            .to_owned();
                        state
                            .procedure_static_locals
                            .entry(path)
                            .or_default()
                            .insert(slot, value.clone());
                    }
                    value
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::LogicalOrEmptyListGlobal(name) => {
                let Some(current) = state.global(name).cloned() else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("runtime global {name:?} is absent"),
                    ));
                };
                let value = if runtime_truthy(&state.heap, &current)
                    .map_err(|message| execution_error(module, &frames, message))?
                {
                    current
                } else {
                    let value = Value::List(state.heap.allocate_list());
                    state.set_global(name.clone(), value.clone());
                    value
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::LogicalOrEmptyListField(name) => {
                let receiver = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let receiver = canonicalize_owned_value(&state.heap, receiver);
                let value = logical_or_empty_list_field(state, receiver, name)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(value);
            }
            Instruction::LogicalOrEmptyListIndex => {
                let key = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let receiver = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let receiver = canonicalize_owned_value(&state.heap, receiver);
                let value = logical_or_empty_list_index(state, receiver, key)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(value);
            }
            Instruction::IndexList => {
                let key = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let receiver = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                // Ordinary list reads validate the arena generation inside
                // `read_list_value`; canonicalizing here performed the same
                // heap lookup twice for every live mapping-list access.
                let receiver = match receiver {
                    Value::Datum(datum) if state.heap.datum(datum).is_err() => Value::Null,
                    value => value,
                };
                if let Value::Text(text) = &receiver {
                    let index = value_to_list_index(&key)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    let value = indexed_text_character(text, index);
                    frames[frame_index].stack.push(value);
                    frames[frame_index].instruction += 1;
                    continue;
                }
                if let Value::Datum(savefile) = receiver
                    && state.heap.datum(savefile).is_ok_and(|datum| {
                        let path = datum.type_path().as_str();
                        path == "/savefile" || path.starts_with("/savefile/")
                    })
                {
                    let key = match key {
                        Value::Text(key) => key.to_string(),
                        value => value.to_string(),
                    };
                    let key = savefile_resolve_path(
                        &state.savefiles.entry(savefile).or_default().cd,
                        &key,
                    );
                    let entry = state
                        .heap
                        .allocate_datum(TypePath::parse("/savefile/entry").unwrap());
                    state.savefile_entries.insert(entry, (savefile, key));
                    frames[frame_index].stack.push(Value::Datum(entry));
                    frames[frame_index].instruction += 1;
                    continue;
                }
                let list = match receiver {
                    Value::List(list) => list,
                    Value::Null
                        if frames.len() > 1
                            && module.procedure_path(procedure).is_some_and(|path| {
                                path == "/datum/proc/_SendSignal"
                                    || path.contains("/proc/_SendSignal@")
                            }) =>
                    {
                        // A receiver can unregister itself during a nested DCS
                        // callback. If its callback table no longer contains
                        // this sender but the sender still holds the scalar
                        // lookup edge, remove that provably stale reciprocal
                        // edge before aborting only this signal dispatch.
                        let signal_frame = &frames[frame_index];
                        let sender = match signal_frame.src {
                            Value::Datum(sender) => Some(sender),
                            _ => None,
                        };
                        let listener = signal_frame
                            .locals
                            .get(4)
                            .or_else(|| signal_frame.locals.get(3))
                            .and_then(|value| match value {
                                Value::Datum(listener) => Some(*listener),
                                _ => None,
                            });
                        let listen_lookup_field =
                            FieldName::parse("_listen_lookup").expect("DCS field name is valid");
                        let lookup = sender.and_then(|sender| {
                            state
                                .heap
                                .datum_field(sender, &listen_lookup_field)
                                .ok()
                                .and_then(|value| match value {
                                    Value::List(lookup) => Some((sender, *lookup)),
                                    _ => None,
                                })
                        });
                        let repaired = match (lookup, listener) {
                            (Some((sender, lookup)), Some(listener)) => {
                                let is_stale_scalar = state
                                    .heap
                                    .list(lookup)
                                    .ok()
                                    .and_then(|lookup| lookup.get_key(&key).ok())
                                    .is_some_and(|value| {
                                        value.semantic_eq(&Value::Datum(listener))
                                    });
                                if is_stale_scalar {
                                    let empty = {
                                        let lookup = state
                                            .heap
                                            .list_mut(lookup)
                                            .expect("live DCS lookup was just read");
                                        lookup.remove_key(&key);
                                        lookup.len() == 0
                                    };
                                    if empty {
                                        state
                                            .heap
                                            .set_datum_field(
                                                sender,
                                                listen_lookup_field,
                                                Value::Null,
                                            )
                                            .expect("live DCS sender was just read");
                                    }
                                    true
                                } else {
                                    false
                                }
                            }
                            _ => false,
                        };
                        let error =
                            execution_error(module, &frames, "list index operation received null");
                        if repaired {
                            if std::env::var_os("DREAM64_TRACE_SIGNAL_MISS").is_some() {
                                eprintln!("dream64 repaired stale signal edge: {error}");
                            }
                        } else {
                            eprintln!("dream64 recovered signal runtime: {error}");
                        }
                        if std::env::var_os("DREAM64_TRACE_SIGNAL_MISS").is_some() {
                            let signal_frame = &frames[frame_index];
                            eprintln!(
                                "dream64 signal miss diagnostic: src={:?} instruction={} key={:?} locals={:?} arguments={:?}",
                                signal_frame.src,
                                signal_frame.instruction,
                                key,
                                signal_frame.locals,
                                signal_frame.arguments,
                            );
                            if let Some(caller) = frames.get(frame_index.saturating_sub(1)) {
                                eprintln!(
                                    "dream64 signal miss caller: src={:?} procedure={:?} instruction={} locals={:?}",
                                    caller.src,
                                    module.procedure_path(caller.procedure),
                                    caller.instruction,
                                    caller.locals,
                                );
                            }
                        }
                        frames.pop().expect("nested signal frame exists");
                        let caller = frames.last_mut().expect("signal caller exists");
                        caller.stack.push(Value::Null);
                        caller.instruction += 1;
                        continue;
                    }
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("list index operation received {value}"),
                        ));
                    }
                };
                if (state.global_vars_proxy == Some(list)
                    || state.datum_vars_proxies.contains_key(&list))
                    && state.heap.list(list).is_err()
                {
                    return Err(execution_error(
                        module,
                        &frames,
                        "list index operation received null",
                    ));
                }
                let value = if state.global_vars_proxy == Some(list) {
                    match &key {
                        Value::Text(name) => FieldName::parse(name)
                            .ok()
                            .and_then(|name| state.global(&name).cloned())
                            .unwrap_or(Value::Null),
                        _ => read_list_value(&state.heap, list, &key, false).unwrap_or(Value::Null),
                    }
                } else if let Some(datum) = state.datum_vars_proxies.get(&list).copied() {
                    match &key {
                        Value::Text(name) => {
                            let field = FieldName::parse(name).ok();
                            if let Some(value) = field
                                .as_ref()
                                .map(|field| lazy_atom_list_field(state, datum, field))
                                .transpose()
                                .map_err(|message| execution_error(module, &frames, message))?
                                .flatten()
                            {
                                value
                            } else {
                                let shared = field
                                    .as_ref()
                                    .and_then(|field| datum_shared_storage(state, datum, field));
                                shared
                                    .and_then(|storage| state.global(&storage).cloned())
                                    .or_else(|| {
                                        field.and_then(|field| {
                                            datum_field_or_initial(state, datum, &field).ok()
                                        })
                                    })
                                    .unwrap_or(Value::Null)
                            }
                        }
                        _ => read_list_value(&state.heap, list, &key, false).unwrap_or(Value::Null),
                    }
                } else {
                    match read_list_value(&state.heap, list, &key, state.is_associative_list(list))
                    {
                        Ok(value) => value,
                        // BYOND associative lookup returns null for an absent key.
                        // Lazy-list idioms such as `lists[target] ||= list()` rely
                        // on this before inserting the new association.
                        Err(ValueError::MissingKey) => Value::Null,
                        Err(ValueError::StaleList(_)) => {
                            return Err(execution_error(
                                module,
                                &frames,
                                "list index operation received null",
                            ));
                        }
                        Err(error) => {
                            return Err(execution_error(module, &frames, error.to_string()));
                        }
                    }
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::IndexLocalList(_) => {
                unreachable!("local list indexing is normalized before dispatch")
            }
            Instruction::ListLengthLocal(_) => {
                unreachable!("local list length is normalized before dispatch")
            }
            Instruction::NextLocalListIteration {
                list_slot,
                index_slot,
                item_slot,
                exit,
            } => {
                let list_slot = usize::from(*list_slot);
                let index_slot = usize::from(*index_slot);
                let item_slot = usize::from(*item_slot);
                let Some(Value::List(list)) = frames[frame_index].locals.get(list_slot).cloned()
                else {
                    return Err(execution_error(
                        module,
                        &frames,
                        "local list iteration received a non-list snapshot",
                    ));
                };
                let Some(Value::Number(index)) =
                    frames[frame_index].locals.get(index_slot).cloned()
                else {
                    return Err(execution_error(
                        module,
                        &frames,
                        "local list iteration received a non-numeric index",
                    ));
                };
                // PrepareIteration always places a private positional snapshot
                // in this compiler-owned local. Read its length and current
                // value through one arena lookup instead of re-entering the
                // general associative IndexList path after the bounds check.
                // Keep the binary32 length comparison before index conversion:
                // very large and fractional indices must fail in the same
                // order as the unspecialized seven-instruction header.
                let values = state
                    .heap
                    .list(list)
                    .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                let length = values.len();
                if index.to_f32() > dm_list_length_number(length) {
                    frames[frame_index].instruction = *exit;
                    continue;
                }
                let key = Value::Number(index);
                let positional_index = value_to_list_index(&key)
                    .map_err(|error| execution_error(module, &frames, error))?;
                let value = values
                    .get(positional_index)
                    .cloned()
                    .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                let value = canonicalize_owned_value(&state.heap, value);
                let Some(item) = frames[frame_index].locals.get(item_slot) else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("invalid local slot {item_slot}"),
                    ));
                };
                if let Value::List(reference) = item
                    && state.reference_lists.contains(reference)
                {
                    state
                        .heap
                        .list_mut(*reference)
                        .and_then(|values| values.set(1, value))
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                } else {
                    frames[frame_index].locals[item_slot] = value;
                }
                frames[frame_index].instruction += 7;
                continue;
            }
            Instruction::SetListIndex => {
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let key = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let receiver = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let list = match canonicalize_owned_value(&state.heap, receiver) {
                    Value::List(list) => list,
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("list assignment received {value}"),
                        ));
                    }
                };
                if state.is_visibility_list(list) {
                    return Err(execution_error(
                        module,
                        &frames,
                        "cannot write to an index of a visibility relationship list",
                    ));
                }
                let associative = state.is_associative_list(list);
                let args_write = (frames[frame_index].args_list == Some(list))
                    .then(|| (key.clone(), value.clone()));
                if state.global_vars_proxy == Some(list) {
                    let Value::Text(name) = key else {
                        return Err(execution_error(
                            module,
                            &frames,
                            "global.vars writes require a text key",
                        ));
                    };
                    let name = FieldName::parse(&name)
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                    state.set_global(name, value);
                } else if let Some(datum) = state.datum_vars_proxies.get(&list).copied() {
                    write_datum_vars(state, datum, list, key, value)
                        .map_err(|message| execution_error(module, &frames, message))?;
                } else if let Err(error) =
                    write_list_value(&mut state.heap, list, key, value, associative)
                {
                    return Err(execution_error(module, &frames, error.to_string()));
                }
                if let Some((key, value)) = args_write {
                    synchronize_frame_argument_write(
                        &mut frames[frame_index],
                        &program,
                        &key,
                        value,
                    )
                    .map_err(|message| execution_error(module, &frames, message))?;
                }
            }
            Instruction::SetListIndexKeep => {
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let key = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let receiver = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let list = match canonicalize_owned_value(&state.heap, receiver) {
                    Value::List(list) => list,
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("list assignment received {value}"),
                        ));
                    }
                };
                if state.is_visibility_list(list) {
                    return Err(execution_error(
                        module,
                        &frames,
                        "cannot write to an index of a visibility relationship list",
                    ));
                }
                let associative = state.is_associative_list(list);
                if state.global_vars_proxy == Some(list) {
                    let Value::Text(name) = key else {
                        return Err(execution_error(
                            module,
                            &frames,
                            "global.vars writes require a text key",
                        ));
                    };
                    let name = FieldName::parse(&name)
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                    state.set_global(name, value.clone());
                } else if let Some(datum) = state.datum_vars_proxies.get(&list).copied() {
                    write_datum_vars(state, datum, list, key, value.clone())
                        .map_err(|message| execution_error(module, &frames, message))?;
                } else if let Err(error) =
                    write_list_value(&mut state.heap, list, key, value.clone(), associative)
                {
                    return Err(execution_error(module, &frames, error.to_string()));
                }
                frames[frame_index].stack.push(value);
            }
            Instruction::CompoundListIndex(operator)
            | Instruction::CompoundListIndexKeep(operator) => {
                let operator = *operator;
                let keep = matches!(instruction, Instruction::CompoundListIndexKeep(_));
                let right = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let key = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let list = match pop(&mut frames[frame_index].stack) {
                    Ok(Value::List(list)) => list,
                    Ok(value) => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("list assignment received {value}"),
                        ));
                    }
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                if state.is_visibility_list(list) {
                    return Err(execution_error(
                        module,
                        &frames,
                        "cannot write to an index of a visibility relationship list",
                    ));
                }
                let associative = state.is_associative_list(list);
                let current = if state.global_vars_proxy == Some(list) {
                    let Value::Text(name) = &key else {
                        return Err(execution_error(
                            module,
                            &frames,
                            "global.vars writes require a text key",
                        ));
                    };
                    FieldName::parse(name)
                        .ok()
                        .and_then(|name| state.global(&name).cloned())
                        .unwrap_or(Value::Null)
                } else if let Some(datum) = state.datum_vars_proxies.get(&list).copied() {
                    let Value::Text(name) = &key else {
                        return Err(execution_error(
                            module,
                            &frames,
                            "datum.vars writes require a text key",
                        ));
                    };
                    let field = FieldName::parse(name)
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                    datum_shared_storage(state, datum, &field)
                        .and_then(|storage| state.global(&storage).cloned())
                        .or_else(|| datum_field_or_initial(state, datum, &field).ok())
                        .unwrap_or(Value::Null)
                } else {
                    match read_list_value(&state.heap, list, &key, associative) {
                        Ok(value) => value,
                        Err(ValueError::MissingKey) => Value::Null,
                        Err(error) => {
                            return Err(execution_error(module, &frames, error.to_string()));
                        }
                    }
                };
                let value = match (&current, &right, operator) {
                    (Value::Null, _, CompoundListIndexOperator::Add) => right,
                    (_, Value::Null, CompoundListIndexOperator::Add) => current,
                    (Value::Text(_), Value::Text(_), CompoundListIndexOperator::Add) => {
                        execute_scalar_add(current, right)
                            .map_err(|message| execution_error(module, &frames, message))?
                    }
                    (Value::List(current), _, operator) => execute_list_compound_operator(
                        compound_assignment_from_list_index(operator),
                        *current,
                        &right,
                        state,
                    )
                    .map_err(|message| execution_error(module, &frames, message))?,
                    _ => {
                        let left = scalar_number_string(current.clone())
                            .map_err(|message| execution_error(module, &frames, message))?;
                        let right = scalar_number_string(right.clone())
                            .map_err(|message| execution_error(module, &frames, message))?;
                        Value::number(execute_compound_list_index_operation(operator, left, right))
                    }
                };
                if state.global_vars_proxy == Some(list) {
                    let Value::Text(name) = key else {
                        return Err(execution_error(
                            module,
                            &frames,
                            "global.vars writes require a text key",
                        ));
                    };
                    let name = FieldName::parse(&name)
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                    state.set_global(name, value.clone());
                } else if let Some(datum) = state.datum_vars_proxies.get(&list).copied() {
                    write_datum_vars(state, datum, list, key, value.clone())
                        .map_err(|message| execution_error(module, &frames, message))?;
                } else if let Err(error) =
                    write_list_value(&mut state.heap, list, key, value.clone(), associative)
                {
                    return Err(execution_error(module, &frames, error.to_string()));
                }
                if keep {
                    frames[frame_index].stack.push(value);
                }
            }
            Instruction::ListLength => {
                let target = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let target = canonicalize_owned_value(&state.heap, target);
                let length = match target {
                    Value::Null => 0,
                    Value::List(list) => match state.heap.list(list) {
                        Ok(values) => values.len(),
                        Err(error) => {
                            return Err(execution_error(module, &frames, error.to_string()));
                        }
                    },
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("list length operation received {value}"),
                        ));
                    }
                };
                let length = dm_list_length_number(length);
                frames[frame_index].stack.push(Value::number(length));
            }
            Instruction::PrepareIteration => {
                let consumes_fresh_block =
                    prepare_iteration_consumes_fresh_block(program, instruction_index);
                let iterable = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let iterable = canonicalize_owned_value(&state.heap, iterable);
                let contents_owner = match &iterable {
                    Value::Datum(datum) => Some(*datum),
                    Value::List(list) => state.contents_owners.get(list).copied(),
                    _ => None,
                };
                let world_contents = match &iterable {
                    Value::Datum(datum) => state.heap.datum(*datum).is_ok_and(|datum| {
                        let path = datum.type_path().as_str();
                        path == "/world" || path.starts_with("/world/")
                    }),
                    Value::List(list) => state
                        .contents_owners
                        .get(list)
                        .and_then(|owner| state.heap.datum(*owner).ok())
                        .is_some_and(|datum| {
                            let path = datum.type_path().as_str();
                            path == "/world" || path.starts_with("/world/")
                        }),
                    _ => false,
                };
                let iterable = match iterable {
                    Value::Datum(datum) => state
                        .heap
                        .datum_field(
                            datum,
                            &FieldName::parse("contents")
                                .expect("built-in contents field is valid"),
                        )
                        .ok()
                        .cloned()
                        .unwrap_or(Value::Null),
                    value => value,
                };
                // BYOND snapshots ordinary list values (and associative
                // mappings) when entering a for-in loop. Mutating the source
                // during the body must not skip shifted entries or append new
                // entries to the active enumeration. OpenDream mirrors this
                // with CopyToArray/CopyAssocValues before creating its
                // enumerator. `world.contents` is the engine-owned exception:
                // its observable order is mobs, other movables, areas, then
                // turfs, independent of their allocation order.
                let iterable = match iterable {
                    Value::List(list) if world_contents => Value::List(
                        world_contents_iteration_snapshot(state, list)
                            .map_err(|error| execution_error(module, &frames, error))?,
                    ),
                    Value::List(list) if contents_owner.is_some() => Value::List(
                        atom_contents_iteration_snapshot(
                            state,
                            contents_owner.expect("contents owner exists"),
                            list,
                        )
                        .map_err(|error| execution_error(module, &frames, error))?,
                    ),
                    // `block()` has just allocated this list and its only
                    // handle is the stack value consumed here. With no
                    // alternate entry, copying cannot improve isolation.
                    Value::List(list) if consumes_fresh_block => Value::List(list),
                    Value::List(list) => Value::List(
                        state
                            .heap
                            .copy_list(list)
                            .map_err(|error| execution_error(module, &frames, error.to_string()))?,
                    ),
                    // BYOND ignores floats, strings, type paths, files, null,
                    // and other non-container values in a for-in header.
                    // Model that as one fresh empty snapshot so the normal
                    // loop machinery simply executes zero iterations.
                    _ => Value::List(state.heap.allocate_list()),
                };
                if let (Value::List(snapshot), Some(assignment)) = (
                    &iterable,
                    simple_iteration_field_assignment(program, instruction_index),
                ) {
                    let item_is_pointer = frames[frame_index]
                        .locals
                        .get(usize::from(assignment.item_slot))
                        .is_some_and(|value| {
                            matches!(value, Value::List(list) if state.reference_lists.contains(list))
                        });
                    let value = match &assignment.value {
                        SimpleIterationValue::Null => Some(Value::Null),
                        SimpleIterationValue::Number(value) => Some(Value::Number(*value)),
                        SimpleIterationValue::Text(value) => Some(Value::text(value.as_str())),
                        SimpleIterationValue::File(value) => Some(Value::file(value.as_str())),
                        SimpleIterationValue::TypePath(value) => {
                            Some(Value::TypePath(value.clone()))
                        }
                        SimpleIterationValue::Local(slot) => frames[frame_index]
                            .locals
                            .get(usize::from(*slot))
                            .cloned()
                            .and_then(|value| match value {
                                Value::List(list) if state.reference_lists.contains(&list) => state
                                    .heap
                                    .list(list)
                                    .ok()
                                    .and_then(|values| values.get(1).ok())
                                    .cloned(),
                                value => Some(value),
                            }),
                    };
                    let datums = state.heap.list(*snapshot).ok().and_then(|values| {
                        values
                            .positions()
                            .map(|(_, value)| match value {
                                Value::Datum(datum) if state.heap.datum(*datum).is_ok() => {
                                    Some(*datum)
                                }
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>()
                    });
                    if !item_is_pointer && let (Some(value), Some(datums)) = (value, datums) {
                        frames[frame_index].locals[usize::from(assignment.list_slot)] =
                            iterable.clone();
                        for (index, datum) in datums.iter().copied().enumerate() {
                            frames[frame_index].locals[usize::from(assignment.index_slot)] =
                                Value::number((index + 1) as f32);
                            frames[frame_index].locals[usize::from(assignment.item_slot)] =
                                Value::Datum(datum);
                            frames[frame_index].instruction = assignment.store_instruction;
                            assign_datum_or_shared_field(
                                state,
                                datum,
                                assignment.field.clone(),
                                value.clone(),
                            )
                            .map_err(|message| execution_error(module, &frames, message))?;
                        }
                        frames[frame_index].locals[usize::from(assignment.index_slot)] =
                            Value::number((datums.len() + 1) as f32);
                        frames[frame_index].instruction = assignment.exit_instruction;
                        continue;
                    }
                }
                frames[frame_index].stack.push(iterable);
            }
            Instruction::IterationTypeFilter(target) => {
                let value = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let matches = match value {
                    Value::Datum(datum) => state
                        .heap()
                        .datum(datum)
                        .is_ok_and(|datum| is_subtype(state, datum.type_path(), &target)),
                    Value::List(_) => target.as_str() == "/list" || target.as_str() == "/alist",
                    _ => false,
                };
                frames[frame_index]
                    .stack
                    .push(Value::number(f32::from(matches)));
            }
            Instruction::LoadSrc => {
                let src = canonicalize_value(&state.heap, &frames[frame_index].src);
                frames[frame_index].stack.push(src);
            }
            Instruction::StoreSrc => {
                let src = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].src = src;
            }
            Instruction::LoadUsr => {
                let usr = canonicalize_value(&state.heap, &frames[frame_index].usr);
                frames[frame_index].stack.push(usr);
            }
            Instruction::LoadCaller => {
                let caller = if frame_index == 0 {
                    Value::Null
                } else {
                    materialize_callee_chain(module, state, &frames[..frame_index])
                        .map_err(|message| execution_error(module, &frames, message))?
                };
                frames[frame_index]
                    .stack
                    .push(canonicalize_owned_value(&state.heap, caller));
            }
            instruction @ (Instruction::LoadField(name) | Instruction::LoadDeclaredField(name)) => {
                let statically_declared = matches!(instruction, Instruction::LoadDeclaredField(_));
                let receiver = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let receiver = canonicalize_owned_value(&state.heap, receiver);
                let value = match receiver {
                    Value::TypePath(path) => match name.as_str() {
                        "type" => Value::TypePath(path),
                        "parent_type" => state
                            .type_parent(&path)
                            .cloned()
                            .map_or(Value::Null, Value::TypePath),
                        _ => state
                            .initial_value(&path, &name)
                            .cloned()
                            .unwrap_or(Value::Null),
                    },
                    Value::ModifiedTypePath(path) => match name.as_str() {
                        "type" => Value::TypePath(path.base().clone()),
                        "parent_type" => state
                            .type_parent(path.base())
                            .cloned()
                            .map_or(Value::Null, Value::TypePath),
                        _ => path
                            .overrides()
                            .iter()
                            .rev()
                            .find(|(field, _)| field == name)
                            .map(|(_, value)| value.clone())
                            .or_else(|| state.initial_value(path.base(), &name).cloned())
                            .unwrap_or(Value::Null),
                    },
                    Value::List(list) if name.as_str() == "len" => {
                        let len = state
                            .heap
                            .list(list)
                            .map_err(|error| execution_error(module, &frames, error.to_string()))?
                            .len();
                        Value::number(dm_list_length_number(len))
                    }
                    Value::Datum(datum) => {
                        // Both declared and ordinary static member reads have
                        // an immutable field name at this callsite. Cache the
                        // physical slot for either form; the record validates
                        // the name/layout on every hit, while special engine
                        // fields stay on the rich path below.
                        let quickening_key = ordinary_field_fast_path_enabled
                            .then(|| {
                                u16::try_from(instruction_index).ok().map(|instruction| {
                                    (
                                        module.identity.0,
                                        frames[frame_index].procedure,
                                        instruction,
                                    )
                                })
                            })
                            .flatten();
                        let quickened_value = quickening_key.and_then(|key| {
                            let slot = state.declared_field_slots.get(&key).copied()?;
                            let record = state.heap.datum(datum).ok()?;
                            if datum_field_requires_special_read(record.type_path(), name) {
                                return None;
                            }
                            match record.field_at_validated_slot(usize::from(slot), name) {
                                Some(value) => {
                                    state.declared_field_quickening.hits =
                                        state.declared_field_quickening.hits.saturating_add(1);
                                    Some(value.clone())
                                }
                                None => {
                                    state.declared_field_quickening.invalidations = state
                                        .declared_field_quickening
                                        .invalidations
                                        .saturating_add(1);
                                    state.declared_field_slots.remove(&key);
                                    None
                                }
                            }
                        });
                        let ordinary_value = if quickened_value.is_some() {
                            None
                        } else {
                            match state.heap.datum(datum) {
                                Ok(record)
                                    if ordinary_field_fast_path_enabled
                                        && !datum_field_requires_special_read(
                                            record.type_path(),
                                            name,
                                        ) =>
                                {
                                    Some(datum_field_or_shared(state, datum, name))
                                }
                                Ok(_) => None,
                                Err(error) => {
                                    return Err(execution_error(
                                        module,
                                        &frames,
                                        error.to_string(),
                                    ));
                                }
                            }
                        };
                        if let Some(value) = quickened_value {
                            value
                        } else if let Some(value) = ordinary_value {
                            if let Some(key) = quickening_key {
                                state.declared_field_quickening.misses =
                                    state.declared_field_quickening.misses.saturating_add(1);
                                if let Ok(record) = state.heap.datum(datum)
                                    && let Some(slot) = record.field_slot(name)
                                    && let Ok(slot) = u16::try_from(slot)
                                {
                                    state.declared_field_slots.insert(key, slot);
                                }
                            }
                            match value {
                                Ok(value) => value,
                                Err(ValueError::MissingField(_)) if statically_declared => {
                                    Value::Null
                                }
                                Err(error) => {
                                    if matches!(error, ValueError::MissingField(_)) {
                                        let runtime_type = state
                                            .heap
                                            .datum(datum)
                                            .expect("live datum was validated above")
                                            .type_path();
                                        eprintln!(
                                            "boot-vm: missing-field receiver_type={} field={} engine_roots={:?} canonical_default={:?}",
                                            runtime_type,
                                            name,
                                            engine_root_paths(runtime_type),
                                            engine_builtin_initial_value(runtime_type, name),
                                        );
                                    }
                                    return Err(execution_error(
                                        module,
                                        &frames,
                                        error.to_string(),
                                    ));
                                }
                            }
                        } else {
                            let runtime_type = match state.heap.datum(datum) {
                                Ok(datum) => datum.type_path().clone(),
                                Err(error) => {
                                    return Err(execution_error(
                                        module,
                                        &frames,
                                        error.to_string(),
                                    ));
                                }
                            };
                            if name.as_str() == "type" {
                                Value::TypePath(runtime_type)
                            } else if name.as_str() == "parent_type" {
                                state
                                    .type_parent(&runtime_type)
                                    .cloned()
                                    .map_or(Value::Null, Value::TypePath)
                            } else if name.as_str() == "appearance"
                                && builtins::is_appearance_source(&runtime_type)
                                && matches!(
                                    datum_field_or_initial(state, datum, &name),
                                    Ok(Value::Null) | Err(ValueError::MissingField(_))
                                )
                            {
                                builtins::appearance_snapshot_builtin(datum, state)
                                    .map_err(|message| execution_error(module, &frames, message))?
                            } else if name.as_str() == "transform"
                                && builtins::is_appearance_source(&runtime_type)
                                && matches!(
                                    datum_field_or_initial(state, datum, &name),
                                    Ok(Value::Null) | Err(ValueError::MissingField(_))
                                )
                            {
                                Value::Datum(
                                    allocate_matrix(
                                        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                        &mut state.heap,
                                    )
                                    .map_err(|message| execution_error(module, &frames, message))?,
                                )
                            } else if is_area_type_path(&runtime_type)
                                && matches!(name.as_str(), "x" | "y" | "z")
                                && let Some(coordinate) = area_coordinate_field(state, datum, &name)
                            {
                                coordinate
                            } else if let Some(value) = lazy_atom_list_field(state, datum, &name)
                                .map_err(|message| execution_error(module, &frames, message))?
                            {
                                value
                            } else if runtime_type.as_str() == "/savefile"
                                || runtime_type.as_str().starts_with("/savefile/")
                            {
                                match name.as_str() {
                                    "cd" => Value::text(
                                        savefile_current_directory(
                                            &state.savefiles.entry(datum).or_default().cd,
                                        )
                                        .to_owned(),
                                    ),
                                    "eof" => {
                                        let savefile = state.savefiles.entry(datum).or_default();
                                        let path = savefile_current_directory(&savefile.cd);
                                        Value::number(if savefile.entries.contains_key(path) {
                                            0.0
                                        } else {
                                            1.0
                                        })
                                    }
                                    "dir" => {
                                        let children = savefile_directory_entries(
                                            state.savefiles.entry(datum).or_default(),
                                        );
                                        let list = state.heap.allocate_list();
                                        let values =
                                            state.heap.list_mut(list).map_err(|error| {
                                                execution_error(module, &frames, error.to_string())
                                            })?;
                                        for child in children {
                                            values.add(Value::text(child));
                                        }
                                        Value::List(list)
                                    }
                                    _ => match state.heap.datum_field(datum, &name) {
                                        Ok(value) => value.clone(),
                                        Err(error) => {
                                            return Err(execution_error(
                                                module,
                                                &frames,
                                                error.to_string(),
                                            ));
                                        }
                                    },
                                }
                            } else {
                                match datum_field_or_shared(state, datum, &name) {
                                    Ok(value) => value,
                                    Err(ValueError::MissingField(_)) if statically_declared => {
                                        Value::Null
                                    }
                                    Err(error) => {
                                        if matches!(error, ValueError::MissingField(_)) {
                                            eprintln!(
                                                "boot-vm: missing-field receiver_type={} field={} engine_roots={:?} canonical_default={:?}",
                                                runtime_type,
                                                name,
                                                engine_root_paths(&runtime_type),
                                                engine_builtin_initial_value(&runtime_type, &name),
                                            );
                                        }
                                        return Err(execution_error(
                                            module,
                                            &frames,
                                            error.to_string(),
                                        ));
                                    }
                                }
                            }
                        }
                    }
                    Value::Null => {
                        return Err(execution_error(module, &frames, "field read received null"));
                    }
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("field read requires a datum, received {value}"),
                        ));
                    }
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::InitialField(name) => {
                let receiver = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let (runtime_type, type_scope) = match receiver {
                    Value::Null => {
                        frames[frame_index].stack.push(Value::Null);
                        frames[frame_index].instruction += 1;
                        continue;
                    }
                    Value::TypePath(path) => (path, true),
                    Value::Datum(datum) => match state.heap.datum(datum) {
                        Ok(datum) => (datum.type_path().clone(), false),
                        Err(error) => {
                            return Err(execution_error(module, &frames, error.to_string()));
                        }
                    },
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!(
                                "initial requires a datum or type path receiver, received {value}"
                            ),
                        ));
                    }
                };
                let value = if type_scope {
                    runtime_initial_field_value(state, &runtime_type, &name)
                        .map_err(|message| execution_error(module, &frames, message))?
                } else {
                    initial_value_or_engine_root(state, &runtime_type, &name).unwrap_or(Value::Null)
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::InitialDynamicField => {
                let key = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let receiver = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let field = match key {
                    Value::Text(name) => match FieldName::parse(name.as_ref()) {
                        Ok(field) => field,
                        Err(error) => {
                            return Err(execution_error(module, &frames, error.to_string()));
                        }
                    },
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("initial vars index requires text, received {value}"),
                        ));
                    }
                };
                let (runtime_type, type_scope) = match receiver {
                    Value::Null => {
                        frames[frame_index].stack.push(Value::Null);
                        frames[frame_index].instruction += 1;
                        continue;
                    }
                    Value::TypePath(path) => (path, true),
                    Value::Datum(datum) => match state.heap.datum(datum) {
                        Ok(datum) => (datum.type_path().clone(), false),
                        Err(error) => {
                            return Err(execution_error(module, &frames, error.to_string()));
                        }
                    },
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!(
                                "initial requires a datum or type path receiver, received {value}"
                            ),
                        ));
                    }
                };
                let value = if type_scope {
                    runtime_initial_field_value(state, &runtime_type, &field)
                        .map_err(|message| execution_error(module, &frames, message))?
                } else {
                    initial_value_or_engine_root(state, &runtime_type, &field)
                        .unwrap_or(Value::Null)
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::StoreField(name) | Instruction::StoreFieldKeep(name) => {
                let keep = matches!(instruction, Instruction::StoreFieldKeep(_));
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let receiver = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let receiver = canonicalize_owned_value(&state.heap, receiver);
                match receiver {
                    Value::Datum(datum) => {
                        assign_datum_or_shared_field(state, datum, name.clone(), value.clone())
                            .map_err(|message| execution_error(module, &frames, message))?;
                    }
                    Value::List(list) if name.as_str() == "len" => {
                        let visibility_before = state
                            .is_visibility_list(list)
                            .then(|| state.visibility_members(list))
                            .transpose()
                            .map_err(|message| execution_error(module, &frames, message))?;
                        let new_len = match &value {
                            Value::Number(number) if number.to_f32().is_finite() => {
                                // BYOND clips negative list lengths to zero. This is
                                // observable during normal SS13 stack merging, where an
                                // emptied stack can refresh its overlays before deletion.
                                dm_list_resize_length(number.to_f32().trunc().max(0.0))
                            }
                            _ => 0,
                        };
                        if state.is_associative_list(list) && new_len != 0 {
                            return Err(execution_error(
                                module,
                                &frames,
                                "alist length can only be assigned zero",
                            ));
                        }
                        if let Err(error) = state
                            .heap
                            .list_mut(list)
                            .and_then(|values| values.resize(new_len))
                        {
                            return Err(execution_error(module, &frames, error.to_string()));
                        }
                        if let Some(before) = visibility_before {
                            state
                                .normalize_and_synchronize_visibility_list(list, &before)
                                .map_err(|message| execution_error(module, &frames, message))?;
                        }
                    }
                    Value::Null => {
                        return Err(execution_error(
                            module,
                            &frames,
                            "field write received null",
                        ));
                    }
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("field write requires a datum or list.len, received {value}"),
                        ));
                    }
                }
                if keep {
                    frames[frame_index].stack.push(value);
                }
            }
            Instruction::LoadGlobal(name) => {
                if remaining_steps >= 4
                    && let Some(target) =
                        false_tick_check_target(&program.instructions, instruction_index, state)
                {
                    let scheduler_batches_before = executed_steps / 4_096;
                    remaining_steps -= 4;
                    executed_steps += 4;
                    for _ in scheduler_batches_before..(executed_steps / 4_096) {
                        account_scheduler_tick_usage(state);
                    }
                    if let Some(profile) = &mut state.atoms_profile {
                        profile.total_instructions = profile.total_instructions.saturating_add(4);
                        if let Some(counts) = &mut profile.instruction_categories {
                            for skipped in
                                &program.instructions[instruction_index + 1..instruction_index + 5]
                            {
                                let category = startup_instruction_category(skipped);
                                counts[category] = counts[category].saturating_add(1);
                            }
                        }
                    }
                    static REPORTED: OnceLock<()> = OnceLock::new();
                    REPORTED.get_or_init(|| {
                        eprintln!(
                            "boot-vm: native-peephole enabled optimization=false-tick-check-skip"
                        );
                    });
                    frames[frame_index].instruction = target;
                    continue;
                }
                let Some(value) = state.global(&name).cloned() else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("runtime global {name:?} is absent"),
                    ));
                };
                let value = canonicalize_owned_value(&state.heap, value);
                if trace_enabled && name.as_str() == "SSdcs" {
                    eprintln!(
                        "boot-vm: global-read name=SSdcs value={} procedure={}",
                        value,
                        module
                            .procedure_path(frames[frame_index].procedure)
                            .unwrap_or("<unknown>")
                    );
                }
                frames[frame_index].stack.push(value);
            }
            Instruction::LoadGlobalVars => {
                let list = if let Some(list) = state.global_vars_proxy {
                    list
                } else {
                    let list = state.heap.allocate_list();
                    for name in state.globals.keys() {
                        state
                            .heap
                            .list_mut(list)
                            .expect("new global.vars proxy is live")
                            .add(Value::text(name.as_str()));
                    }
                    state.mark_associative_list(list);
                    state.global_vars_proxy = Some(list);
                    list
                };
                frames[frame_index].stack.push(Value::List(list));
            }
            Instruction::LoadDatumVars => {
                let datum = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => match canonicalize_value(&state.heap, &value) {
                        Value::Datum(datum) => datum,
                        _ => {
                            return Err(execution_error(
                                module,
                                &frames,
                                format!("vars requires a datum, received {value}"),
                            ));
                        }
                    },
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let list = if let Some(list) = state.datum_vars_by_datum.get(&datum).copied() {
                    list
                } else {
                    let runtime_type = state
                        .heap
                        .datum(datum)
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?
                        .type_path()
                        .clone();
                    // Merge base-to-derived engine roots. A synthesized
                    // `/atom/movable/...` path may have a movable catalog
                    // containing only movable-owned fields; selecting that
                    // one map used to hide `/atom.density` and every other
                    // base appearance field from `datum.vars`.
                    let mut initial = engine_builtin_initial_fields(&runtime_type);
                    for values in engine_root_initial_field_maps(state, &runtime_type).rev() {
                        initial.extend(
                            values
                                .iter()
                                .map(|(field, value)| (field.clone(), value.clone())),
                        );
                    }
                    initial.extend(state.inherited_initial_values(&runtime_type));
                    let initial = initial.into_iter().collect::<Vec<_>>();
                    let instance = state
                        .heap
                        .datum_fields(datum)
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?
                        .map(|(field, value)| (field.clone(), value.clone()))
                        .collect::<Vec<_>>();
                    let shared = state
                        .shared_fields
                        .get(&runtime_type)
                        .cloned()
                        .unwrap_or_default();
                    let list = state.heap.allocate_list();
                    for (field, value) in initial {
                        state
                            .heap
                            .list_mut(list)
                            .expect("new datum.vars proxy is live")
                            .set_key(Value::text(field.as_str()), value);
                    }
                    for (field, value) in instance {
                        state
                            .heap
                            .list_mut(list)
                            .expect("new datum.vars proxy is live")
                            .set_key(Value::text(field.as_str()), value);
                    }
                    for (name, storage) in shared {
                        let value = state.global(&storage).cloned().unwrap_or(Value::Null);
                        state
                            .heap
                            .list_mut(list)
                            .expect("new datum.vars proxy is live")
                            .set_key(Value::text(name.as_str()), value);
                    }
                    state.mark_associative_list(list);
                    state.datum_vars_proxies.insert(list, datum);
                    state.datum_vars_by_datum.insert(datum, list);
                    list
                };
                frames[frame_index].stack.push(Value::List(list));
            }
            Instruction::LoadDynamicField => {
                let key = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let receiver = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let Value::Datum(datum) = canonicalize_value(&state.heap, &receiver) else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("vars requires a datum, received {receiver}"),
                    ));
                };
                let value = match key {
                    Value::Text(name) => {
                        let field = FieldName::parse(&name)
                            .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                        if let Some(value) = lazy_atom_list_field(state, datum, &field)
                            .map_err(|message| execution_error(module, &frames, message))?
                        {
                            value
                        } else if let Some(storage) = datum_shared_storage(state, datum, &field) {
                            state.global(&storage).cloned().unwrap_or(Value::Null)
                        } else {
                            datum_field_or_initial(state, datum, &field).unwrap_or(Value::Null)
                        }
                    }
                    _ => Value::Null,
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::StoreDynamicField => {
                let value = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let key = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let receiver = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let Value::Datum(datum) = canonicalize_value(&state.heap, &receiver) else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("vars requires a datum, received {receiver}"),
                    ));
                };
                let Value::Text(name) = &key else {
                    return Err(execution_error(
                        module,
                        &frames,
                        "datum.vars writes require a text key",
                    ));
                };
                let field = FieldName::parse(name)
                    .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                if let Some(storage) = datum_shared_storage(state, datum, &field) {
                    state.set_global(storage, value.clone());
                } else {
                    assign_datum_field(state, datum, field, value.clone())
                        .map_err(|message| execution_error(module, &frames, message))?;
                }
                if let Some(list) = state.datum_vars_by_datum.get(&datum).copied() {
                    write_list_value(&mut state.heap, list, key, value, false)
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                }
            }
            Instruction::LoadInitialGlobal(name) => {
                let value = state
                    .initial_globals
                    .get(&name)
                    .cloned()
                    .unwrap_or(Value::Null);
                frames[frame_index].stack.push(value);
            }
            Instruction::StoreGlobal(name) => {
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                if trace_enabled && name.as_str() == "SSdcs" {
                    eprintln!(
                        "boot-vm: global-write name=SSdcs value={} procedure={}",
                        value,
                        module
                            .procedure_path(frames[frame_index].procedure)
                            .unwrap_or("<unknown>")
                    );
                }
                state.set_global(name.clone(), value);
            }
            Instruction::MutateLocal {
                slot,
                delta,
                prefix,
            } => {
                let (slot, delta, prefix) = (*slot, *delta, *prefix);
                let Some(current) = frames[frame_index].locals.get(usize::from(slot)).cloned()
                else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("invalid local slot {slot}"),
                    ));
                };
                let (result, updated) = mutate_scalar_value(current, delta, prefix)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].locals[usize::from(slot)] = updated;
                frames[frame_index].stack.push(result);
            }
            Instruction::MutateGlobal {
                name,
                delta,
                prefix,
            } => {
                let (delta, prefix) = (*delta, *prefix);
                let current = state.global(&name).cloned().unwrap_or(Value::Null);
                let (result, updated) = mutate_scalar_value(current, delta, prefix)
                    .map_err(|message| execution_error(module, &frames, message))?;
                state.set_global(name.clone(), updated);
                frames[frame_index].stack.push(result);
            }
            Instruction::MutateResult { delta, prefix } => {
                let (delta, prefix) = (*delta, *prefix);
                let current = frames[frame_index].result.clone();
                let (result, updated) = mutate_scalar_value(current, delta, prefix)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].result = updated;
                frames[frame_index].stack.push(result);
            }
            Instruction::MutateField {
                name,
                delta,
                prefix,
            } => {
                let (delta, prefix) = (*delta, *prefix);
                let receiver = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let receiver = canonicalize_owned_value(&state.heap, receiver);
                let current = match &receiver {
                    Value::Datum(datum) => datum_field_or_initial(state, *datum, &name)
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?
                        .clone(),
                    Value::List(list) if name.as_str() == "len" => {
                        let len = state
                            .heap
                            .list(*list)
                            .map_err(|error| execution_error(module, &frames, error.to_string()))?
                            .len();
                        Value::number(len as f32)
                    }
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!(
                                "increment/decrement field requires a datum or list.len, received {value}"
                            ),
                        ));
                    }
                };
                let (result, updated) = mutate_scalar_value(current, delta, prefix)
                    .map_err(|message| execution_error(module, &frames, message))?;
                match receiver {
                    Value::Datum(datum) => {
                        state
                            .heap
                            .set_datum_field(datum, name.clone(), updated)
                            .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                    }
                    Value::List(list) => {
                        let visibility_before = state
                            .is_visibility_list(list)
                            .then(|| state.visibility_members(list))
                            .transpose()
                            .map_err(|message| execution_error(module, &frames, message))?;
                        let length = updated.as_number().unwrap_or(0.0).trunc().max(0.0);
                        if state.is_associative_list(list) && length != 0.0 {
                            return Err(execution_error(
                                module,
                                &frames,
                                "alist length can only be assigned zero",
                            ));
                        }
                        let new_len = length as usize;
                        state
                            .heap
                            .list_mut(list)
                            .and_then(|values| values.resize(new_len))
                            .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                        if let Some(before) = visibility_before {
                            state
                                .normalize_and_synchronize_visibility_list(list, &before)
                                .map_err(|message| execution_error(module, &frames, message))?;
                        }
                    }
                    _ => unreachable!("receiver was validated above"),
                }
                frames[frame_index].stack.push(result);
            }
            Instruction::MutateListIndex { delta, prefix } => {
                let (delta, prefix) = (*delta, *prefix);
                let key = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let receiver = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let list = match canonicalize_owned_value(&state.heap, receiver) {
                    Value::List(list) => list,
                    value => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("list mutation requires a list, received {value}"),
                        ));
                    }
                };
                if state.is_visibility_list(list) {
                    return Err(execution_error(
                        module,
                        &frames,
                        "cannot write to an index of a visibility relationship list",
                    ));
                }
                let associative = state.is_associative_list(list);
                let current = match read_list_value(&state.heap, list, &key, associative) {
                    Ok(value) => value,
                    // BYOND treats an absent associative entry like null for
                    // postfix/prefix mutation. Idioms such as
                    // `counter[target]++` therefore insert 1 on first use.
                    Err(ValueError::MissingKey) => Value::Null,
                    Err(error) => {
                        return Err(execution_error(module, &frames, error.to_string()));
                    }
                };
                let (result, updated) = mutate_scalar_value(current, delta, prefix)
                    .map_err(|message| execution_error(module, &frames, message))?;
                write_list_value(&mut state.heap, list, key, updated, associative)
                    .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                frames[frame_index].stack.push(result);
            }
            Instruction::Duplicate => {
                let Some(value) = frames[frame_index].stack.last().cloned() else {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::PrepareRhsFirstIndexAssignment => {
                let stack = &mut frames[frame_index].stack;
                if stack.len() < 3 {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let len = stack.len();
                stack[len - 3..].rotate_left(1);
            }
            Instruction::AddressLocal(slot) => {
                let slot = *slot;
                let Some(local) = frames[frame_index].locals.get_mut(usize::from(slot)) else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("invalid local slot {slot}"),
                    ));
                };
                let reference = match local {
                    Value::List(list) if state.reference_lists.contains(list) => *list,
                    value => {
                        let list = state.heap.allocate_list();
                        state
                            .heap
                            .list_mut(list)
                            .expect("new pointer cell is live")
                            .add(value.clone());
                        state.reference_lists.insert(list);
                        *value = Value::List(list);
                        list
                    }
                };
                frames[frame_index].stack.push(Value::List(reference));
            }
            Instruction::LoadLocalRaw(slot) => {
                let slot = *slot;
                let Some(value) = frames[frame_index].locals.get(usize::from(slot)) else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("invalid local slot {slot}"),
                    ));
                };
                let value = canonicalize_value(&state.heap, value);
                frames[frame_index].stack.push(value);
            }
            Instruction::LoadLocal(slot) => {
                let slot = *slot;
                let Some(mut value) = frames[frame_index].locals.get(usize::from(slot)).cloned()
                else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("invalid local slot {slot}"),
                    ));
                };
                if let Value::List(list) = value
                    && state.reference_lists.contains(&list)
                {
                    value = state
                        .heap
                        .list(list)
                        .and_then(|values| values.get(1))
                        .cloned()
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                }
                frames[frame_index]
                    .stack
                    .push(canonicalize_owned_value(&state.heap, value));
            }
            Instruction::StoreLocal(slot) => {
                let slot = *slot;
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let local_index = usize::from(slot);
                let Some(local) = frames[frame_index].locals.get(local_index) else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("invalid local slot {slot}"),
                    ));
                };
                let reference_list = match local {
                    Value::List(list) if state.reference_lists.contains(list) => Some(*list),
                    _ => None,
                };
                let parameter = local_index < frames[frame_index].declared_argument_count;
                let static_local = frames[frame_index].static_locals.contains(&slot);

                // Plain locals are overwhelmingly the common startup case. Move the
                // popped value directly into the slot, avoiding an Arc/value clone and
                // all argument/static synchronization work.
                if reference_list.is_none() && !parameter && !static_local {
                    frames[frame_index].locals[local_index] = value;
                    frames[frame_index].instruction += 1;
                    continue;
                }

                if let Some(list) = reference_list {
                    state
                        .heap
                        .list_mut(list)
                        .and_then(|values| values.set(1, value.clone()))
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                } else {
                    frames[frame_index].locals[local_index] = value.clone();
                }
                if parameter {
                    frames[frame_index].arguments[local_index] = value.clone();
                    if let Some(args) = frames[frame_index].args_list {
                        state
                            .heap
                            .list_mut(args)
                            .and_then(|values| values.set(local_index + 1, value.clone()))
                            .map_err(|error| execution_error(module, &frames, error.to_string()))?;
                    }
                }
                if static_local {
                    let path = module
                        .procedure_path(frames[frame_index].procedure)
                        .unwrap_or("<unknown procedure>")
                        .to_owned();
                    state
                        .procedure_static_locals
                        .entry(path)
                        .or_default()
                        .insert(slot, value);
                }
            }
            Instruction::LoadStaticLocalOrJump { slot, target } => {
                let (slot, target) = (*slot, *target);
                let path = module
                    .procedure_path(frames[frame_index].procedure)
                    .unwrap_or("<unknown procedure>");
                if let Some(value) = state
                    .procedure_static_locals
                    .get(path)
                    .and_then(|slots| slots.get(&slot))
                    .cloned()
                {
                    let Some(local) = frames[frame_index].locals.get_mut(usize::from(slot)) else {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("invalid static local slot {slot}"),
                        ));
                    };
                    *local = value;
                    frames[frame_index].static_locals.push(slot);
                    frames[frame_index].instruction = target.saturating_sub(1);
                }
            }
            Instruction::InitializeStaticLocal(slot) => {
                let slot = *slot;
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let path = module
                    .procedure_path(frames[frame_index].procedure)
                    .unwrap_or("<unknown procedure>")
                    .to_owned();
                state
                    .procedure_static_locals
                    .entry(path)
                    .or_default()
                    .insert(slot, value.clone());
                frames[frame_index].static_locals.push(slot);
                frames[frame_index].stack.push(value);
            }
            Instruction::LoadResult => {
                let result = frames[frame_index].result.clone();
                frames[frame_index].stack.push(result);
            }
            Instruction::StoreUsr => {
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                frames[frame_index].usr = value;
            }
            Instruction::StoreResult => {
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                frames[frame_index].result = value;
            }
            Instruction::Pop => {
                if let Err(message) = pop(&mut frames[frame_index].stack) {
                    return Err(execution_error(module, &frames, message));
                }
            }
            Instruction::Crash => {
                let message = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                // Monkestation's `stack_trace()` deliberately calls CRASH in
                // a tiny helper proc to make BYOND print a stack without
                // aborting the caller. A runtime in the nested helper ends
                // that helper and yields null to its caller; it does not tear
                // down the entire execution chain. Keep direct CRASH strict.
                if module
                    .procedure_path(frames[frame_index].procedure)
                    .is_some_and(|path| {
                        path == "/proc/_stack_trace" || path.contains("/proc/_stack_trace@")
                    })
                {
                    eprintln!("dream64 stack trace: {message}");
                    frames.pop().expect("stack-trace helper frame exists");
                    let Some(caller) = frames.last_mut() else {
                        return Ok(FrameRunOutcome::Complete(Value::Null));
                    };
                    caller.stack.push(Value::Null);
                    caller.instruction += 1;
                    continue;
                }
                return Err(execution_error(
                    module,
                    &frames,
                    format!("CRASH: {message}"),
                ));
            }
            Instruction::BeginTry { catch, end, local } => {
                let (catch, end, local) = (*catch, *end, *local);
                if catch >= program.instructions.len() || end >= program.instructions.len() {
                    return Err(execution_error(
                        module,
                        &frames,
                        "exception handler target is outside the procedure",
                    ));
                }
                let stack_depth = frames[frame_index].stack.len();
                frames[frame_index]
                    .exception_handlers_mut()
                    .push(ExceptionHandler {
                        start: instruction_index + 1,
                        end,
                        catch,
                        local,
                        stack_depth,
                    });
            }
            Instruction::EndTry => {
                if frames[frame_index].exception_handlers_mut().pop().is_none() {
                    return Err(execution_error(
                        module,
                        &frames,
                        "EndTry without an active exception handler",
                    ));
                }
            }
            Instruction::Throw => {
                let thrown = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let mut handler = None;
                for candidate_frame in (0..frames.len()).rev() {
                    let current = frames[candidate_frame].instruction;
                    if let Some(position) = frames[candidate_frame]
                        .exception_handlers()
                        .iter()
                        .rposition(|handler| handler.start <= current && current <= handler.end)
                    {
                        handler = Some((candidate_frame, position));
                        break;
                    }
                }
                let Some((handler_frame, handler_position)) = handler else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("uncaught exception: {thrown}"),
                    ));
                };
                frames.truncate(handler_frame + 1);
                let handler = frames[handler_frame]
                    .exception_handlers_mut()
                    .remove(handler_position);
                frames[handler_frame]
                    .exception_handlers_mut()
                    .truncate(handler_position);
                frames[handler_frame].stack.truncate(handler.stack_depth);
                if let Some(slot) = handler.local {
                    let Some(local) = frames[handler_frame].locals.get_mut(usize::from(slot))
                    else {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("invalid catch local {slot}"),
                        ));
                    };
                    *local = thrown;
                }
                frames[handler_frame].instruction = handler.catch;
                continue;
            }
            Instruction::Locate { argument_count } => {
                let count = usize::from(*argument_count);
                let stack_length = frames[frame_index].stack.len();
                if count > stack_length {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let arguments = frames[frame_index].stack.split_off(stack_length - count);
                let located = if let [search] = arguments.as_slice() {
                    locate_single(search, state)
                } else if let [search, container] = arguments.as_slice() {
                    locate_in_container(search, container, state)
                        .map_err(|message| execution_error(module, &frames, message))?
                } else if let [x, y, z] = arguments.as_slice() {
                    let integer = |value: &Value| {
                        value.as_number().and_then(|value| {
                            (value.is_finite()
                                && value.fract() == 0.0
                                && value >= i32::MIN as f32
                                && value <= i32::MAX as f32)
                                .then(|| {
                                    #[allow(clippy::cast_possible_truncation)]
                                    {
                                        value as i32
                                    }
                                })
                        })
                    };
                    match (integer(x), integer(y), integer(z)) {
                        (Some(x), Some(y), Some(z)) => {
                            state.turf_at(x, y, z).map_or(Value::Null, Value::Datum)
                        }
                        _ => Value::Null,
                    }
                } else {
                    Value::Null
                };
                frames[frame_index].stack.push(located);
            }
            Instruction::LocateIn { argument_count } => {
                let count = usize::from(*argument_count)
                    .checked_add(1)
                    .expect("u16 argument count plus container fits usize");
                let stack_length = frames[frame_index].stack.len();
                if count > stack_length {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let values = frames[frame_index].stack.split_off(stack_length - count);
                let located = if let [search, container] = values.as_slice() {
                    locate_in_container(search, container, state)
                        .map_err(|message| execution_error(module, &frames, message))?
                } else {
                    Value::Null
                };
                frames[frame_index].stack.push(located);
            }
            Instruction::Negate => {
                let value = match pop_number(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                frames[frame_index].stack.push(Value::number(-value));
            }
            Instruction::BitNot => {
                let value = match pop_number(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                frames[frame_index]
                    .stack
                    .push(Value::number(bitwise_not(value)));
            }
            Instruction::Not => {
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let is_truthy = runtime_truthy(&state.heap, &value)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index]
                    .stack
                    .push(Value::number(f32::from(!is_truthy)));
            }
            Instruction::CompoundAssignment(operator) => {
                let operator = *operator;
                let right = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let left = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let value = if let Value::Datum(datum) = left {
                    if is_matrix_datum(datum, &state.heap) {
                        execute_matrix_compound(operator, datum, &right, &mut state.heap)
                            .map_err(|message| execution_error(module, &frames, message))?
                    } else if is_vector_datum(datum, &state.heap) {
                        execute_vector_compound(operator, datum, &right, &mut state.heap)
                            .map_err(|message| execution_error(module, &frames, message))?
                    } else {
                        execute_scalar_compound_assignment(operator, Value::Datum(datum), right)
                            .map_err(|message| execution_error(module, &frames, message))?
                    }
                } else if let Value::List(list) = left {
                    execute_list_compound_operator(operator, list, &right, state)
                        .map_err(|message| execution_error(module, &frames, message))?
                } else {
                    execute_scalar_compound_assignment(operator, left, right)
                        .map_err(|message| execution_error(module, &frames, message))?
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::Add => {
                let right = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let left = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let value = if let (Value::Datum(left), Value::Datum(right)) = (&left, &right)
                    && is_matrix_datum(*left, &state.heap)
                    && is_matrix_datum(*right, &state.heap)
                {
                    let left_values = matrix_components(*left, &state.heap)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    let right_values = matrix_components(*right, &state.heap)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    let datum = allocate_matrix(
                        std::array::from_fn(|index| left_values[index] + right_values[index]),
                        &mut state.heap,
                    )
                    .map_err(|message| execution_error(module, &frames, message))?;
                    Value::Datum(datum)
                } else if let (Value::Datum(left), Value::Datum(right)) = (&left, &right)
                    && is_vector_datum(*left, &state.heap)
                    && is_vector_datum(*right, &state.heap)
                {
                    Value::Datum(
                        allocate_vector(
                            vector_zip(*left, *right, &state.heap, |a, b| a + b)
                                .map_err(|message| execution_error(module, &frames, message))?,
                            &mut state.heap,
                        )
                        .map_err(|message| execution_error(module, &frames, message))?,
                    )
                } else if let Value::List(list) = left {
                    execute_list_binary_operator("+", list, &right, state)
                        .map_err(|message| execution_error(module, &frames, message))?
                } else {
                    execute_scalar_add(left, right)
                        .map_err(|message| execution_error(module, &frames, message))?
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::Subtract
            | Instruction::BitAnd
            | Instruction::BitOr
            | Instruction::BitXor => {
                let right = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let left = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let operator = match instruction {
                    Instruction::Subtract => "-",
                    Instruction::BitAnd => "&",
                    Instruction::BitOr => "|",
                    Instruction::BitXor => "^",
                    _ => unreachable!(),
                };
                let value = if matches!(instruction, Instruction::Subtract)
                    && let (Value::Datum(left), Value::Datum(right)) = (&left, &right)
                    && is_matrix_datum(*left, &state.heap)
                    && is_matrix_datum(*right, &state.heap)
                {
                    let left_values = matrix_components(*left, &state.heap)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    let right_values = matrix_components(*right, &state.heap)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    let datum = allocate_matrix(
                        std::array::from_fn(|index| left_values[index] - right_values[index]),
                        &mut state.heap,
                    )
                    .map_err(|message| execution_error(module, &frames, message))?;
                    Value::Datum(datum)
                } else if matches!(instruction, Instruction::Subtract)
                    && let (Value::Datum(left), Value::Datum(right)) = (&left, &right)
                    && is_vector_datum(*left, &state.heap)
                    && is_vector_datum(*right, &state.heap)
                {
                    Value::Datum(
                        allocate_vector(
                            vector_zip(*left, *right, &state.heap, |a, b| a - b)
                                .map_err(|message| execution_error(module, &frames, message))?,
                            &mut state.heap,
                        )
                        .map_err(|message| execution_error(module, &frames, message))?,
                    )
                } else if let Value::List(list) = left {
                    execute_list_binary_operator(operator, list, &right, state)
                        .map_err(|message| execution_error(module, &frames, message))?
                } else if matches!(instruction, Instruction::BitAnd)
                    && (matches!(left, Value::Null) || matches!(right, Value::Null))
                {
                    // BYOND treats a null scalar bitwise intersection as 0,
                    // even when the other operand is a list. This makes
                    // optional-list filters such as `(data & vars)` safely
                    // iterable when `data` is absent.
                    Value::number(0.0)
                } else {
                    let left = scalar_number_string(left)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    let right = scalar_number_string(right)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    Value::number(execute_numeric_binary(&instruction, left, right))
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::Multiply
            | Instruction::Power
            | Instruction::Divide
            | Instruction::Remainder
            | Instruction::FractionalRemainder
            | Instruction::ShiftLeft
            | Instruction::ShiftRight => {
                let right = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let left = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let vector_operator = match instruction {
                    Instruction::Multiply => Some("*"),
                    Instruction::Divide => Some("/"),
                    _ => None,
                };
                let value = if let Value::Datum(datum) = left
                    && is_matrix_datum(datum, &state.heap)
                    && let Some(operator) = vector_operator
                {
                    execute_matrix_binary(operator, datum, &right, &mut state.heap)
                        .map_err(|message| execution_error(module, &frames, message))?
                } else if let Value::Datum(datum) = left
                    && is_vector_datum(datum, &state.heap)
                    && let Some(operator) = vector_operator
                {
                    execute_vector_binary(operator, datum, &right, &mut state.heap)
                        .map_err(|message| execution_error(module, &frames, message))?
                } else {
                    let left = scalar_number_string(left)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    let right = scalar_number_string(right)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    Value::number(execute_numeric_binary(&instruction, left, right))
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::Less
            | Instruction::LessEqual
            | Instruction::Greater
            | Instruction::GreaterEqual => {
                let right = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let left = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let comparison = compare_values(&left, &right)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let result = match instruction {
                    Instruction::Less => comparison.is_some_and(std::cmp::Ordering::is_lt),
                    Instruction::LessEqual => comparison.is_some_and(std::cmp::Ordering::is_le),
                    Instruction::Greater => comparison.is_some_and(std::cmp::Ordering::is_gt),
                    Instruction::GreaterEqual => comparison.is_some_and(std::cmp::Ordering::is_ge),
                    _ => unreachable!(),
                };
                frames[frame_index]
                    .stack
                    .push(Value::number(f32::from(result)));
            }
            Instruction::Compare => {
                let right = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let left = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let comparison = compare_values(&left, &right)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let value = comparison.map_or(0.0, |value| match value {
                    std::cmp::Ordering::Less => -1.0,
                    std::cmp::Ordering::Equal => 0.0,
                    std::cmp::Ordering::Greater => 1.0,
                });
                frames[frame_index].stack.push(Value::number(value));
            }
            Instruction::Equal | Instruction::NotEqual => {
                let right = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let left = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let equal = values_equal(&state.heap, &left, &right);
                let result = if matches!(instruction, Instruction::NotEqual) {
                    !equal
                } else {
                    equal
                };
                frames[frame_index]
                    .stack
                    .push(Value::number(f32::from(result)));
            }
            Instruction::Equivalent | Instruction::NotEquivalent => {
                let right = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let left = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let equivalent = values_equivalent(&left, &right, &state.heap)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let result = if matches!(instruction, Instruction::NotEquivalent) {
                    !equivalent
                } else {
                    equivalent
                };
                frames[frame_index]
                    .stack
                    .push(Value::number(f32::from(result)));
            }
            Instruction::Contains => {
                let container = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let needle = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                // BYOND treats atoms (including turfs and world) as their
                // `contents` list when they appear on the right-hand side of
                // binary `in`. This is the same container coercion used by a
                // for-in header. In particular, `node in get_step(src, dir)`
                // tests whether the adjacent turf contains that node.
                let container = canonicalize_owned_value(&state.heap, container);
                let container = match container {
                    Value::Datum(datum) => datum_field_or_initial(
                        state,
                        datum,
                        &FieldName::parse("contents").expect("built-in contents field is valid"),
                    )
                    .unwrap_or(Value::Null),
                    value => value,
                };
                let contains = if let Value::List(list) = container {
                    state
                        .heap
                        .list(list)
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?
                        .positions()
                        .any(|(_, value)| values_equal(&state.heap, &needle, value))
                } else {
                    false
                };
                frames[frame_index]
                    .stack
                    .push(Value::number(f32::from(contains)));
            }
            Instruction::And | Instruction::Or => {
                let right = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => runtime_truthy(&state.heap, &value)
                        .map_err(|message| execution_error(module, &frames, message))?,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let left = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => runtime_truthy(&state.heap, &value)
                        .map_err(|message| execution_error(module, &frames, message))?,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let result = if matches!(instruction, Instruction::And) {
                    left && right
                } else {
                    left || right
                };
                frames[frame_index]
                    .stack
                    .push(Value::number(f32::from(result)));
            }
            Instruction::JumpIfNull(target) => {
                let target = *target;
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                if matches!(canonicalize_owned_value(&state.heap, value), Value::Null) {
                    if let Err(message) = validate_jump(target, program.instructions.len()) {
                        return Err(execution_error(module, &frames, message));
                    }
                    frames[frame_index].instruction = target;
                    continue;
                }
            }
            Instruction::JumpIfFalse(target) => {
                let target = *target;
                let condition = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                if !runtime_truthy(&state.heap, &condition)
                    .map_err(|message| execution_error(module, &frames, message))?
                {
                    if let Err(message) = validate_jump(target, program.instructions.len()) {
                        return Err(execution_error(module, &frames, message));
                    }
                    frames[frame_index].instruction = target;
                    continue;
                }
            }
            Instruction::Jump(target) => {
                let target = *target;
                if let Err(message) = validate_jump(target, program.instructions.len()) {
                    return Err(execution_error(module, &frames, message));
                }
                frames[frame_index].instruction = target;
                continue;
            }
            Instruction::JumpIfArgumentSupplied { parameter, target } => {
                let parameter = usize::from(*parameter);
                let target = *target;
                if frames[frame_index]
                    .supplied_parameters
                    .get(parameter)
                    .copied()
                    .unwrap_or(false)
                    && !matches!(frames[frame_index].locals.get(parameter), Some(Value::Null))
                {
                    if let Err(message) = validate_jump(target, program.instructions.len()) {
                        return Err(execution_error(module, &frames, message));
                    }
                    frames[frame_index].instruction = target;
                    continue;
                }
            }
            Instruction::Call {
                procedure: target,
                argument_count,
                argument_names,
            } => {
                let mut target = *target;
                let argument_count = *argument_count;
                if frames.len() >= limits.max_call_depth {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("maximum call depth {} exceeded", limits.max_call_depth),
                    ));
                }
                let expanded_argument_names = (argument_count == EXPANDED_ARGUMENT_COUNT)
                    .then(|| frames[frame_index].take_pending_argument_names())
                    .flatten();
                let expanded_argument_roots = (argument_count == EXPANDED_ARGUMENT_COUNT)
                    .then(|| frames[frame_index].take_pending_argument_roots())
                    .unwrap_or_default();
                let count = runtime_argument_count(&mut frames[frame_index].stack, argument_count)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let stack_length = frames[frame_index].stack.len();
                if count > stack_length {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let arguments = pop_builtin_arguments(&mut frames[frame_index].stack, count);
                let mut context = frame_context(&frames[frame_index]);
                if let Some(path) = module.procedure_path(target)
                    && let Some((_, selector)) = path.rsplit_once("/proc/")
                    && !path.starts_with("/proc/")
                    && matches!(frames[frame_index].src, Value::Datum(_))
                {
                    let selector = selector.split('@').next().unwrap_or(selector);
                    let (dynamic_target, dynamic_context) = dynamic_call_target_named(
                        module,
                        state,
                        &frames[frame_index].src,
                        selector,
                        &context,
                        false,
                    )
                    .map_err(|message| execution_error(module, &frames, message))?;
                    target = dynamic_target;
                    context = dynamic_context;
                }
                let target_program = module
                    .resolve_procedure(target)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let names = expanded_argument_names
                    .as_deref()
                    .unwrap_or(&argument_names);
                if names.iter().all(Option::is_none)
                    && let Some(name) =
                        canonical_static_native_builtin(module, target, target_program)
                {
                    let value = if name == "istext" {
                        arguments.first().map(canonical_istext).ok_or_else(|| {
                            "canonical istext call is missing its required argument".to_owned()
                        })
                    } else {
                        execute_standard_builtin(name, &arguments, state)
                    };
                    // min/max are read-only before they can report an error.
                    // Fall through to the canonical DM frame on failure so
                    // its exact callee source and call stack are retained.
                    if let Ok(value) = value {
                        frames[frame_index].stack.push(value);
                        frames[frame_index].instruction += 1;
                        continue;
                    }
                }
                // Monkestation's canonical type2parent helper implements a
                // lexical type-path parent with several text operations. It is
                // called millions of times while components register during
                // map load. Only bypass the DM frame when the resolved body is
                // bytecode-identical to that helper and its argument is the
                // common type-path case; customized helpers and all coercion
                // cases continue through ordinary DM execution.
                if arguments.len() == 1
                    && canonical_type2parent_target(module, target, target_program)
                    && let Value::TypePath(path) = &arguments[0]
                {
                    frames[frame_index]
                        .stack
                        .push(canonical_type2parent(&path).map_or(Value::Null, Value::TypePath));
                    frames[frame_index].instruction += 1;
                    continue;
                }
                let mut target_frame = if names.iter().all(Option::is_none) {
                    make_frame_owned(target, target_program, arguments, &context)
                } else {
                    make_frame_named(target, target_program, &arguments, names, &context)
                };
                target_frame.set_retained_call_roots(expanded_argument_roots);
                if shuttle_trace_enabled() {
                    let slot = shuttle_trace_slot_from_arguments(&target_frame.arguments);
                    shuttle_trace_prepare_call(
                        module,
                        state,
                        &frames[frame_index],
                        target,
                        slot,
                        &mut target_frame,
                    );
                }
                mark_boot_trace_frame(&mut target_frame, module, state, executed_steps);
                frames.push(target_frame);
                continue;
            }
            Instruction::CallCurrent { argument_count } => {
                let argument_count = *argument_count;
                if frames.len() >= limits.max_call_depth {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("maximum call depth {} exceeded", limits.max_call_depth),
                    ));
                }
                let expanded_argument_names = argument_count
                    .filter(|count| *count == EXPANDED_ARGUMENT_COUNT)
                    .and_then(|_| frames[frame_index].take_pending_argument_names());
                let expanded_argument_roots = argument_count
                    .filter(|count| *count == EXPANDED_ARGUMENT_COUNT)
                    .map(|_| frames[frame_index].take_pending_argument_roots())
                    .unwrap_or_default();
                let arguments = if let Some(argument_count) = argument_count {
                    let count =
                        runtime_argument_count(&mut frames[frame_index].stack, argument_count)
                            .map_err(|message| execution_error(module, &frames, message))?;
                    let stack_length = frames[frame_index].stack.len();
                    if count > stack_length {
                        return Err(execution_error(module, &frames, "bytecode stack underflow"));
                    }
                    pop_builtin_arguments(&mut frames[frame_index].stack, count)
                } else {
                    forwarded_frame_arguments(&frames[frame_index], program)
                };
                let context = frame_context(&frames[frame_index]);
                let mut target_frame = if expanded_argument_names
                    .as_deref()
                    .is_none_or(|names| names.iter().all(Option::is_none))
                {
                    make_frame_owned(procedure, program, arguments, &context)
                } else {
                    make_frame_named(
                        procedure,
                        program,
                        &arguments,
                        expanded_argument_names
                            .as_deref()
                            .expect("named arguments exist"),
                        &context,
                    )
                };
                target_frame.set_retained_call_roots(expanded_argument_roots);
                if shuttle_trace_enabled() {
                    let slot = shuttle_trace_slot_from_arguments(&target_frame.arguments);
                    shuttle_trace_prepare_call(
                        module,
                        state,
                        &frames[frame_index],
                        procedure,
                        slot,
                        &mut target_frame,
                    );
                }
                frames.push(target_frame);
                continue;
            }
            Instruction::CallParent {
                procedure: target,
                argument_count,
            } => {
                let mut target = *target;
                let argument_count = *argument_count;
                if frames.len() >= limits.max_call_depth {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("maximum call depth {} exceeded", limits.max_call_depth),
                    ));
                }
                let mut engine_parent_context = None;
                let mut engine_post_parent_frame = None;
                let client_new_engine_boundary = module
                    .procedure_path(procedure)
                    .is_some_and(|path| path.contains("/client/proc/New"))
                    && target.is_none_or(|parent| {
                        module
                            .procedure_path(parent)
                            .is_some_and(|path| path.starts_with("/datum/proc/New@dream64_native"))
                    });
                if client_new_engine_boundary
                    && let Value::Datum(client) = &frames[frame_index].src
                    && let Some(mob) = state.local_client_mobs.get(client).copied()
                {
                    let client = *client;
                    let mob_is_pending = !matches!(
                        state
                            .heap
                            .datum_field(client, &FieldName::parse("mob").unwrap()),
                        Ok(Value::Datum(_))
                    );
                    if mob_is_pending {
                        state.attach_local_client(client, mob).map_err(|message| {
                            execution_error(
                                module,
                                &frames,
                                format!("client connection: {message}"),
                            )
                        })?;
                        let key = state
                            .heap
                            .datum_field(client, &FieldName::parse("key").unwrap())
                            .ok()
                            .cloned();
                        if let Some(key) = key
                            && datum_field_or_initial(state, mob, &FieldName::parse("key").unwrap())
                                .is_ok()
                        {
                            assign_datum_field(state, mob, FieldName::parse("key").unwrap(), key)
                                .map_err(|message| execution_error(module, &frames, message))?;
                        }
                        let caller_context = frame_context(&frames[frame_index]);
                        let (login, login_context) = dynamic_call_target_named(
                            module,
                            state,
                            &Value::Datum(mob),
                            "Login",
                            &caller_context,
                            false,
                        )
                        .map_err(|message| execution_error(module, &frames, message))?;
                        // A connection mob is an ordinary constructed atom. Its
                        // New/Initialize chain must observe the reciprocal client
                        // binding and complete before Login (new-player splash
                        // creation depends on precisely this ordering).
                        if let Some((constructor, constructor_context)) =
                            constructor_target_if_present(module, state, mob, &caller_context)
                        {
                            let login_program = module
                                .resolve_procedure(login)
                                .map_err(|message| execution_error(module, &frames, message))?;
                            engine_post_parent_frame =
                                Some(make_frame(login, login_program, &[], &login_context));
                            target = Some(constructor);
                            engine_parent_context = Some(constructor_context);
                        } else {
                            target = Some(login);
                            engine_parent_context = Some(login_context);
                        }
                    }
                }
                let Some(target) = target else {
                    if client_new_engine_boundary {
                        frames[frame_index].stack.push(Value::Null);
                        frames[frame_index].instruction += 1;
                        continue;
                    }
                    return Err(execution_error(
                        module,
                        &frames,
                        "parent procedure call has no resolved target",
                    ));
                };
                let expanded_argument_names = argument_count
                    .filter(|count| *count == EXPANDED_ARGUMENT_COUNT)
                    .and_then(|_| frames[frame_index].take_pending_argument_names());
                let expanded_argument_roots = argument_count
                    .filter(|count| *count == EXPANDED_ARGUMENT_COUNT)
                    .map(|_| frames[frame_index].take_pending_argument_roots())
                    .unwrap_or_default();
                let arguments = if let Some(argument_count) = argument_count {
                    let count =
                        runtime_argument_count(&mut frames[frame_index].stack, argument_count)
                            .map_err(|message| execution_error(module, &frames, message))?;
                    let stack_length = frames[frame_index].stack.len();
                    if count > stack_length {
                        return Err(execution_error(module, &frames, "bytecode stack underflow"));
                    }
                    pop_builtin_arguments(&mut frames[frame_index].stack, count)
                } else {
                    forwarded_frame_arguments(&frames[frame_index], program)
                };
                let target_program = module
                    .resolve_procedure(target)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let context =
                    engine_parent_context.unwrap_or_else(|| frame_context(&frames[frame_index]));
                let mut target_frame = if expanded_argument_names
                    .as_deref()
                    .is_none_or(|names| names.iter().all(Option::is_none))
                {
                    make_frame_owned(target, target_program, arguments, &context)
                } else {
                    make_frame_named(
                        target,
                        target_program,
                        &arguments,
                        expanded_argument_names
                            .as_deref()
                            .expect("named arguments exist"),
                        &context,
                    )
                };
                target_frame.set_retained_call_roots(expanded_argument_roots);
                if shuttle_trace_enabled() {
                    let slot = shuttle_trace_slot_from_arguments(&target_frame.arguments);
                    shuttle_trace_prepare_call(
                        module,
                        state,
                        &frames[frame_index],
                        target,
                        slot,
                        &mut target_frame,
                    );
                }
                target_frame.set_engine_post_return(engine_post_parent_frame.map(Box::new));
                frames.push(target_frame);
                continue;
            }
            Instruction::CallDynamic {
                static_selector,
                argument_count,
                argument_names,
                null_receiver_is_global,
            } => {
                let argument_count = *argument_count;
                let null_receiver_is_global = *null_receiver_is_global;
                let expanded_argument_names = (argument_count == EXPANDED_ARGUMENT_COUNT)
                    .then(|| frames[frame_index].take_pending_argument_names())
                    .flatten();
                let expanded_argument_roots = (argument_count == EXPANDED_ARGUMENT_COUNT)
                    .then(|| frames[frame_index].take_pending_argument_roots())
                    .unwrap_or_default();
                let count = runtime_argument_count(&mut frames[frame_index].stack, argument_count)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let stack_length = frames[frame_index].stack.len();
                // A constant selector is embedded in the instruction. Fully
                // dynamic call() selectors retain the original stack shape.
                let prefix_count = 1 + usize::from(static_selector.is_none());
                if stack_length < count + prefix_count {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let arguments = pop_builtin_arguments(&mut frames[frame_index].stack, count);
                let selector = static_selector.is_none().then(|| {
                    frames[frame_index]
                        .stack
                        .pop()
                        .expect("stack length was checked")
                });
                let receiver = frames[frame_index]
                    .stack
                    .pop()
                    .expect("stack length was checked");
                let selector_text = static_selector.as_deref().or_else(|| match &selector {
                    Some(Value::Text(selector)) => Some(selector.as_ref()),
                    _ => None,
                });
                if let Value::List(list) = receiver {
                    let Some(method) = selector_text else {
                        let selector = selector.as_ref().expect("dynamic selector exists");
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("list procedure selector must be text, received {selector}"),
                        ));
                    };
                    let Some(result) = execute_list_method(method, list, &arguments, state) else {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("unknown /list procedure {method:?}"),
                        ));
                    };
                    let result =
                        result.map_err(|message| execution_error(module, &frames, message))?;
                    frames[frame_index].stack.push(result);
                } else if matches!(receiver, Value::Datum(_))
                    && let Some(method) = selector_text
                    && let Ok((target, context)) = dynamic_call_target_named_at_callsite(
                        module,
                        state,
                        &receiver,
                        method,
                        &frame_context(&frames[frame_index]),
                        false,
                        static_selector
                            .is_some()
                            .then_some((procedure, instruction_index)),
                    )
                {
                    if frames.len() >= limits.max_call_depth {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("maximum call depth {} exceeded", limits.max_call_depth),
                        ));
                    }
                    let target_program = module
                        .resolve_procedure(target)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    let names = expanded_argument_names
                        .as_deref()
                        .or((!argument_names.is_empty()).then_some(argument_names.as_slice()));
                    let mut target_frame =
                        if names.is_none_or(|names| names.iter().all(Option::is_none)) {
                            make_frame_owned(target, target_program, arguments, &context)
                        } else {
                            make_frame_named(
                                target,
                                target_program,
                                &arguments,
                                names.expect("named arguments exist"),
                                &context,
                            )
                        };
                    target_frame.set_retained_call_roots(expanded_argument_roots);
                    if shuttle_trace_enabled() {
                        let slot = shuttle_trace_slot_from_arguments(&target_frame.arguments);
                        shuttle_trace_prepare_call(
                            module,
                            state,
                            &frames[frame_index],
                            target,
                            slot,
                            &mut target_frame,
                        );
                    }
                    mark_boot_trace_frame(&mut target_frame, module, state, executed_steps);
                    frames.push(target_frame);
                    continue;
                } else if let (Value::Datum(savefile), Some(method)) = (&receiver, selector_text)
                    && state.heap.datum(*savefile).is_ok_and(|datum| {
                        let path = datum.type_path().as_str();
                        path == "/savefile" || path.starts_with("/savefile/")
                    })
                    && method == "ExportText"
                {
                    let key = arguments
                        .first()
                        .and_then(|value| match value {
                            Value::Text(value) => Some(value.as_ref()),
                            _ => None,
                        })
                        .unwrap_or("");
                    let encoded = state
                        .savefiles
                        .get(savefile)
                        .and_then(|savefile| {
                            let path = savefile_resolve_path(&savefile.cd, key);
                            savefile.entries.get(&path)
                        })
                        .map_or_else(String::new, savefile_export_value);
                    frames[frame_index]
                        .stack
                        .push(Value::text(format!("{key} = {{\"\n{encoded}\n\"}}\n\n")));
                } else if let (Value::Datum(datum), Some(method)) = (&receiver, selector_text)
                    && is_matrix_datum(*datum, &state.heap)
                {
                    let result = execute_matrix_method(*datum, method, &arguments, &mut state.heap)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    frames[frame_index].stack.push(result);
                } else if let (Value::Datum(datum), Some(method)) = (&receiver, selector_text)
                    && is_vector_datum(*datum, &state.heap)
                {
                    let result = execute_vector_method(*datum, method, &arguments, &mut state.heap)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    frames[frame_index].stack.push(result);
                } else if let (Value::Datum(datum), Some(method)) = (&receiver, selector_text)
                    && is_regex_datum(*datum, state)
                {
                    let result = if method == "Replace" {
                        if !(2..=4).contains(&arguments.len()) {
                            return Err(execution_error(
                                module,
                                &frames,
                                "unknown or invalid /regex procedure \"Replace\"",
                            ));
                        }
                        let mut replacement_arguments = Vec::with_capacity(arguments.len() + 1);
                        replacement_arguments.push(arguments[0].clone());
                        replacement_arguments.push(Value::Datum(*datum));
                        replacement_arguments
                            .push(arguments.get(1).cloned().unwrap_or(Value::Null));
                        replacement_arguments.extend(arguments.iter().skip(2).cloned());
                        let caller_context = frame_context(&frames[frame_index]);
                        let root_len = preserve_reentrant_frame_roots(state, &frames);
                        let result = replace_text_regex(
                            module,
                            state,
                            *datum,
                            &replacement_arguments,
                            false,
                            &caller_context,
                        );
                        state.host_value_roots.truncate(root_len);
                        result.map_err(|message| execution_error(module, &frames, message))?
                    } else {
                        execute_regex_method(*datum, method, &arguments, state)
                            .map_err(|message| execution_error(module, &frames, message))?
                    };
                    frames[frame_index].stack.push(result);
                } else if let (Value::Datum(datum), Some(method)) = (&receiver, selector_text)
                    && is_icon_datum(*datum, &state.heap)
                    && matches!(
                        method,
                        "MapColors"
                            | "Blend"
                            | "SetIntensity"
                            | "Scale"
                            | "Crop"
                            | "Shift"
                            | "Width"
                            | "Height"
                            | "DrawBox"
                            | "Insert"
                            | "GetPixel"
                            | "Turn"
                            | "Flip"
                            | "SwapColor"
                    )
                {
                    let result = match method {
                        "MapColors" => apply_icon_map_colors(*datum, &arguments, &mut state.heap)
                            .map(|()| Value::Null),
                        "Blend" => apply_icon_blend(*datum, &arguments, &mut state.heap)
                            .map(|()| Value::Null),
                        "SetIntensity" => {
                            apply_icon_set_intensity(*datum, &arguments, &mut state.heap)
                                .map(|()| Value::Null)
                        }
                        method => execute_icon_method(*datum, method, &arguments, &mut state.heap),
                    }
                    .map_err(|message| execution_error(module, &frames, message))?;
                    frames[frame_index].stack.push(result);
                } else {
                    if frames.len() >= limits.max_call_depth {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("maximum call depth {} exceeded", limits.max_call_depth),
                        ));
                    }
                    let caller_context = frame_context(&frames[frame_index]);
                    let target = if let Some(selector) = static_selector.as_deref() {
                        dynamic_call_target_named(
                            module,
                            state,
                            &receiver,
                            selector,
                            &caller_context,
                            null_receiver_is_global,
                        )
                    } else {
                        dynamic_call_target(
                            module,
                            state,
                            &receiver,
                            selector.as_ref().expect("dynamic selector exists"),
                            &caller_context,
                            null_receiver_is_global,
                        )
                    };
                    let (target, context) =
                        target.map_err(|message| execution_error(module, &frames, message))?;
                    let target_program = module
                        .resolve_procedure(target)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    let mut target_frame =
                        make_frame_owned(target, target_program, arguments, &context);
                    if shuttle_trace_enabled() {
                        let slot = shuttle_trace_slot_from_arguments(&target_frame.arguments);
                        shuttle_trace_prepare_call(
                            module,
                            state,
                            &frames[frame_index],
                            target,
                            slot,
                            &mut target_frame,
                        );
                    }
                    mark_boot_trace_frame(&mut target_frame, module, state, executed_steps);
                    frames.push(target_frame);
                    continue;
                }
            }
            Instruction::Spawn { entry } => {
                let entry = *entry;
                let delay = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let delay = delay.as_number().ok_or_else(|| {
                    execution_error(
                        module,
                        &frames,
                        format!("spawn delay must be numeric, received {delay}"),
                    )
                })?;
                let mut spawned = frames[frame_index].clone();
                spawned.instruction = entry;
                spawned.stack.clear();
                if delay.is_sign_negative() {
                    // `spawn(-1)` runs the detached body synchronously until
                    // its first block, but the caller has still consumed the
                    // Spawn instruction.  The recursive frame runner returns
                    // directly to this match arm, so advance the retained
                    // caller here; otherwise it executes Spawn a second time
                    // with the delay already popped and underflows its stack.
                    frames[frame_index].instruction += 1;
                    let root_len = preserve_reentrant_frame_roots(state, &frames);
                    let outcome =
                        run_frames(module, vec![spawned], limits, step_budget_behavior, state);
                    state.host_value_roots.truncate(root_len);
                    match outcome? {
                        FrameRunOutcome::Complete(_) => {}
                        FrameRunOutcome::Yielded { frames, delay } => {
                            schedule_frames(state, frames, delay);
                        }
                        FrameRunOutcome::Prompted { id, prompt } => {
                            register_prompt(state, id, prompt);
                        }
                    }
                    continue;
                }
                schedule_frames(state, vec![spawned], delay);
            }
            Instruction::Return => {
                let result = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let mut finished = frames.pop().expect("returning frame exists");
                let finish_atoms_profile = finished.atoms_profile_root;
                let finish_tgm_profile = finished.tgm_profile_root;
                let result = finished.caller_result_override().cloned().unwrap_or(result);
                if trace_enabled
                    && module
                        .procedure_path(finished.procedure)
                        .is_some_and(|path| {
                            path.contains("/subsystem/processing/dcs/proc/GetElement@")
                        })
                {
                    eprintln!("boot-vm: dcs-get-element-result value={result}");
                }
                if let Some(cold) = finished.cold()
                    && let Some(started) = cold.boot_trace_started
                {
                    let (datum_delta, list_delta, deferred_delta) =
                        cold.boot_trace_heap
                            .map_or((0, 0, 0), |(datums, lists, deferred)| {
                                (
                                    state.heap.live_datum_count() as i128 - datums as i128,
                                    state.heap.live_list_count() as i128 - lists as i128,
                                    module.materialized_deferred_procedure_count() as i128
                                        - deferred as i128,
                                )
                            });
                    eprintln!(
                        "boot-vm: initializer-end path={} elapsed_ms={} steps={} datum_delta={} list_delta={} deferred_delta={}",
                        module
                            .paths
                            .get(finished.procedure.index())
                            .map_or("<missing>", String::as_str),
                        started.elapsed().as_millis(),
                        executed_steps.saturating_sub(cold.boot_trace_step),
                        datum_delta,
                        list_delta,
                        deferred_delta,
                    );
                }
                if finish_atoms_profile && let Some(profile) = state.atoms_profile.take() {
                    emit_atoms_profile(&profile);
                }
                if finish_tgm_profile && let Some(profile) = state.tgm_profile.take() {
                    emit_tgm_profile(&profile);
                }
                if let Some(post_return) = finished.take_shuttle_trace_post_return() {
                    if let Value::Datum(component) = finished.src {
                        match post_return {
                            ShuttleTracePostReturn::NullifyNode { slot } => {
                                shuttle_trace_emit_snapshot(
                                    state,
                                    component,
                                    "nullify-node-after",
                                    slot,
                                );
                            }
                            ShuttleTracePostReturn::AtmosInit => {
                                shuttle_trace_emit_snapshot(
                                    state,
                                    component,
                                    "atmos-init-after",
                                    None,
                                );
                            }
                        }
                    }
                }
                if let Some(post_return) = finished.take_engine_post_return() {
                    let mut post_return = *post_return;
                    post_return.set_caller_result_override(Some(result));
                    frames.push(post_return);
                    continue;
                }
                let Some(caller) = frames.last_mut() else {
                    return Ok(FrameRunOutcome::Complete(result));
                };
                caller.stack.push(result);
                caller.instruction += 1;
                continue;
            }
        }
        frames[frame_index].instruction += 1;
    }
}
fn execution_error(
    module: &Module,
    frames: &[CallFrame],
    message: impl Into<String>,
) -> RuntimeError {
    let instruction = frames.last().map_or(0, |frame| frame.instruction);
    let source_span = frames.last().and_then(|frame| {
        module
            .procedure(frame.procedure)
            .and_then(|program| program.source_spans.get(frame.instruction))
            .copied()
    });
    RuntimeError {
        message: message.into(),
        instruction,
        source_span,
        call_stack: frames
            .iter()
            .map(|frame| trace(module, frame.procedure, frame.instruction))
            .collect(),
    }
}

pub(crate) fn trace(module: &Module, procedure: ProcedureId, instruction: usize) -> CallTrace {
    CallTrace {
        procedure: module
            .procedure_path(procedure)
            .unwrap_or("<invalid procedure>")
            .to_owned(),
        instruction,
        source_span: module
            .procedure(procedure)
            .and_then(|program| program.source_spans.get(instruction))
            .copied(),
    }
}

fn execute_numeric_binary(instruction: &Instruction, left: f32, right: f32) -> f32 {
    match instruction {
        Instruction::Add => left + right,
        Instruction::Subtract => left - right,
        Instruction::Multiply => left * right,
        Instruction::Power => left.powf(right),
        Instruction::Divide => left / right,
        Instruction::Remainder => integer_remainder(left, right),
        Instruction::FractionalRemainder => fractional_remainder(left, right),
        Instruction::BitAnd => bitwise_binary(left, right, |left, right| left & right),
        Instruction::BitOr => bitwise_binary(left, right, |left, right| left | right),
        Instruction::BitXor => bitwise_binary(left, right, |left, right| left ^ right),
        Instruction::ShiftLeft => bitwise_shift(left, right, |left, right| left << right),
        Instruction::ShiftRight => bitwise_shift(left, right, |left, right| left >> right),
        Instruction::Less => f32::from(left < right),
        Instruction::LessEqual => f32::from(left <= right),
        Instruction::Greater => f32::from(left > right),
        Instruction::GreaterEqual => f32::from(left >= right),
        _ => unreachable!("instruction came from the numeric operation group"),
    }
}

fn execute_compound_list_index_operation(
    operator: CompoundListIndexOperator,
    left: f32,
    right: f32,
) -> f32 {
    match operator {
        CompoundListIndexOperator::Add => left + right,
        CompoundListIndexOperator::Subtract => left - right,
        CompoundListIndexOperator::Multiply => left * right,
        CompoundListIndexOperator::Divide => left / right,
        CompoundListIndexOperator::Remainder => integer_remainder(left, right),
        CompoundListIndexOperator::FractionalRemainder => fractional_remainder(left, right),
        CompoundListIndexOperator::BitAnd => {
            bitwise_binary(left, right, |left, right| left & right)
        }
        CompoundListIndexOperator::BitOr => bitwise_binary(left, right, |left, right| left | right),
        CompoundListIndexOperator::BitXor => {
            bitwise_binary(left, right, |left, right| left ^ right)
        }
        CompoundListIndexOperator::ShiftLeft => {
            bitwise_shift(left, right, |left, right| left << right)
        }
        CompoundListIndexOperator::ShiftRight => {
            bitwise_shift(left, right, |left, right| left >> right)
        }
    }
}

fn compound_assignment_from_list_index(
    operator: CompoundListIndexOperator,
) -> CompoundAssignmentOperator {
    match operator {
        CompoundListIndexOperator::Add => CompoundAssignmentOperator::Add,
        CompoundListIndexOperator::Subtract => CompoundAssignmentOperator::Subtract,
        CompoundListIndexOperator::Multiply => CompoundAssignmentOperator::Multiply,
        CompoundListIndexOperator::Divide => CompoundAssignmentOperator::Divide,
        CompoundListIndexOperator::Remainder => CompoundAssignmentOperator::Remainder,
        CompoundListIndexOperator::FractionalRemainder => {
            CompoundAssignmentOperator::FractionalRemainder
        }
        CompoundListIndexOperator::BitAnd => CompoundAssignmentOperator::BitAnd,
        CompoundListIndexOperator::BitOr => CompoundAssignmentOperator::BitOr,
        CompoundListIndexOperator::BitXor => CompoundAssignmentOperator::BitXor,
        CompoundListIndexOperator::ShiftLeft => CompoundAssignmentOperator::ShiftLeft,
        CompoundListIndexOperator::ShiftRight => CompoundAssignmentOperator::ShiftRight,
    }
}
