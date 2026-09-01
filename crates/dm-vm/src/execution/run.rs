//! The interpreter execution loop and its instruction helpers.
//!
//! Module layout: this crate splits `execution` into `state`, `frame`, `scheduler`, `run`,
//! `run_support`, and `support` so the deterministic engine stays navigable by concern.

use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use crate::bytecode::{Instruction, Module, ProcedureId, Program};
use crate::{
    AtomsProfile, AtomsProfileInstruction, AtomsProfileProcedure, CompactWordcodeImage,
    ExecutionLimits, RuntimeError, STARTUP_INSTRUCTION_CATEGORY_COUNT, TgmDrive, TgmProfile,
    atoms_profile_enabled, atoms_profile_snapshot_lines_if_due, boot_dashboard_enabled,
    boot_trace_enabled, canonical_tgm_load_path, drive_ruin_candidate_scan, drive_tgm_load,
    execute_compact_fast_instruction, is_atoms_initialize_path, is_subsystem_initialize_path,
    numeric_dispatch_candidate, slow_instruction_trace_threshold, startup_instruction_category,
    startup_instruction_profile_enabled, startup_profile_enabled, tgm_profiling_enabled,
    trace_tgm_route, try_run_build_coordinate_prefix, try_run_camera_chunk_fast_path,
    try_run_discover_offset_fast_path, try_run_dmm_preload_measurement_fast_path,
    try_run_guarded_jit, try_run_numeric_dispatch_block, try_run_numeric_local_update,
    try_run_numeric_loop_branch, try_run_parsed_dmm_new_fast_path,
    try_run_register_signal_fast_path, try_run_rooted_list_jit, try_run_ruin_affected_turfs_batch,
    try_run_tgm_build_cache_simple_member,
};
use dm_jit::NumericRunOutcome;
use dm_value::Value;

use crate::execution::frame::CallFrame;
use crate::execution::frame::FrameRunOutcome;
use crate::execution::frame::StepBudgetBehavior;
use crate::execution::interpreter::{DispatchFlow, dispatch_instruction};
use crate::execution::run_support::execution_error;
use crate::execution::scheduler::account_scheduler_tick_usage;
use crate::execution::state::ExecutionState;
use crate::value_ops::pop;

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
    // Diagnostic: env-gated per-instruction wall-time watchdog for locating a
    // single builtin/native op that overshoots the scheduler wall deadline.
    let slow_instruction_threshold = slow_instruction_trace_threshold();
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

        let slow_instruction_started = slow_instruction_threshold.map(|_| Instant::now());
        let dispatch_flow = dispatch_instruction(
            module,
            state,
            &mut frames,
            frame_index,
            procedure,
            instruction_index,
            program,
            instruction,
            limits,
            step_budget_behavior,
            &mut executed_steps,
            &mut remaining_steps,
            trace_enabled,
            ordinary_field_fast_path_enabled,
        )?;
        if let Some(started) = slow_instruction_started {
            let elapsed = started.elapsed();
            if slow_instruction_threshold.is_some_and(|threshold| elapsed >= threshold) {
                let span = program.source_spans.get(instruction_index).copied();
                let mut opcode = format!("{instruction:?}");
                if opcode.len() > 160 {
                    let cut = (0..=160)
                        .rev()
                        .find(|index| opcode.is_char_boundary(*index))
                        .unwrap_or(0);
                    opcode.truncate(cut);
                    opcode.push('…');
                }
                eprintln!(
                    "boot-vm: slow-instruction elapsed_us={} depth={} procedure={} instruction={} source={}..{} opcode={opcode}",
                    elapsed.as_micros(),
                    frames.len(),
                    module
                        .paths
                        .get(procedure.index())
                        .map_or("<missing>", String::as_str),
                    instruction_index,
                    span.map_or(0, |span| span.start),
                    span.map_or(0, |span| span.end),
                );
            }
        }
        match dispatch_flow {
            DispatchFlow::Exit(outcome) => return *outcome,
            DispatchFlow::Continue => (),
        }
    }
}
