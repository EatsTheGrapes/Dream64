//! The deterministic scheduler: frame queues, yield accounting, and step budgets.
//!
//! Module layout: this crate splits `execution` into `state`, `frame`, `scheduler`, `run`,
//! and `support` so the deterministic engine stays navigable by concern.

use std::time::Instant;

use crate::builtins::advance_native_walks;
use crate::bytecode::Module;
use crate::{
    ExecutionLimits, RuntimeError, ScheduledSpawn, advance_headless_world_clock, register_prompt,
    set_world_numeric_field, world_numeric_field,
};
use dm_value::{FieldName, TypePath, Value};

use crate::execution::frame::CallFrame;
use crate::execution::frame::FrameRunOutcome;
use crate::execution::frame::OwnedContinuation;
use crate::execution::frame::StepBudgetBehavior;
use crate::execution::frame::VmContinuationId;
use crate::execution::run::run_frames;
use crate::execution::state::ExecutionState;

pub(crate) fn materialize_callee_chain(
    module: &Module,
    state: &mut ExecutionState,
    callers: &[CallFrame],
) -> Result<Value, String> {
    let callee_path = TypePath::parse("/callee").expect("built-in /callee path");
    let mut previous = Value::Null;
    for frame in callers {
        let args = state.heap.allocate_list();
        for argument in &frame.arguments {
            state
                .heap
                .list_mut(args)
                .map_err(|error| error.to_string())?
                .add(argument.clone());
        }
        let datum = state.heap.allocate_datum(callee_path.clone());
        let procedure = module
            .procedure_path(frame.procedure)
            .unwrap_or("/proc")
            .split('@')
            .next()
            .unwrap_or("/proc");
        let procedure_value = TypePath::parse(procedure)
            .map(Value::TypePath)
            .unwrap_or_else(|_| Value::text(procedure));
        for (name, value) in [
            ("caller", previous.clone()),
            ("src", frame.src.clone()),
            ("usr", frame.usr.clone()),
            ("args", Value::List(args)),
            ("type", procedure_value),
            ("file", Value::Null),
            ("line", Value::number(0.0)),
        ] {
            state
                .heap
                .set_datum_field(
                    datum,
                    FieldName::parse(name).expect("built-in callee field"),
                    value,
                )
                .map_err(|error| error.to_string())?;
        }
        previous = Value::Datum(datum);
    }
    Ok(previous)
}

/// Advances the deterministic scheduler and runs every spawned body whose
/// delay has elapsed. Tasks with the same deadline retain source order.
///
/// # Errors
///
/// Returns a runtime error when a due spawned body fails.
pub fn advance_scheduler(
    module: &Module,
    ticks: u64,
    limits: ExecutionLimits,
    state: &mut ExecutionState,
) -> Result<Vec<Value>, RuntimeError> {
    state.assert_owner_thread();
    state.scheduler_tick = state.scheduler_tick.saturating_add(ticks);
    advance_headless_world_clock(state, ticks);
    advance_native_walks(state);
    let mut due = Vec::new();
    let mut future = Vec::with_capacity(state.scheduled_spawns.len());
    for spawn in state.scheduled_spawns.drain(..) {
        if spawn.due_tick <= state.scheduler_tick {
            due.push(spawn);
        } else {
            future.push(spawn);
        }
    }
    state.scheduled_spawns = future;
    due.sort_unstable_by_key(|spawn| (spawn.due_tick, spawn.sequence));
    // Pop in source order while retaining every not-yet-run due continuation
    // inside ExecutionState, where the collector can traverse its frame roots.
    due.reverse();
    debug_assert!(state.scheduler_inflight.is_empty());
    state.scheduler_inflight = due;
    // BYOND/OpenDream expose elapsed host-tick percentage, which can exceed
    // 100 when a proc overruns its tick. Monk deliberately raises its stage-2
    // startup limit to about 196%, so pinning due work at exactly 100% prevents
    // MAPLOADING_CHECK_TICK from ever reaching stoplag(). Track the actual
    // dispatch slice instead, sampling outside the per-instruction hot path.
    state.scheduler_tick_started = (!state.scheduler_inflight.is_empty()).then(Instant::now);
    let dispatch_started = Instant::now();
    set_world_numeric_field(state, "tick_usage", 0.0);
    let mut completed = Vec::new();
    while let Some(spawn) = state.scheduler_inflight.pop() {
        let mut slice_limits = limits;
        if let Some(budget) = limits.wall_clock_budget {
            let Some(remaining) = budget.checked_sub(dispatch_started.elapsed()) else {
                state.scheduler_inflight.push(spawn);
                let remaining = std::mem::take(&mut state.scheduler_inflight);
                state.scheduled_spawns.extend(remaining.into_iter().rev());
                break;
            };
            slice_limits.wall_clock_budget = Some(remaining);
        }
        let outcome = match run_frames(
            module,
            spawn.frames.into_frames(),
            slice_limits,
            StepBudgetBehavior::YieldScheduledContinuation,
            state,
        ) {
            Ok(outcome) => outcome,
            Err(error) => {
                let remaining = std::mem::take(&mut state.scheduler_inflight);
                state.scheduled_spawns.extend(remaining.into_iter().rev());
                state.scheduler_tick_started = None;
                set_world_numeric_field(state, "tick_usage", 0.0);
                return Err(error);
            }
        };
        match outcome {
            FrameRunOutcome::Complete(value) => {
                state.host_value_roots.push(value.clone());
                completed.push(value);
            }
            FrameRunOutcome::Yielded { frames, delay } => schedule_frames(state, frames, delay),
            FrameRunOutcome::Prompted { id, prompt } => register_prompt(state, id, prompt),
        }
    }
    state.scheduler_tick_started = None;
    set_world_numeric_field(state, "tick_usage", 0.0);
    Ok(completed)
}

pub(crate) fn account_scheduler_tick_usage(state: &mut ExecutionState) {
    let Some(started) = state.scheduler_tick_started else {
        return;
    };
    let tick_lag = world_numeric_field(state, "tick_lag")
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(1.0);
    let tick_duration = f64::from(tick_lag) / 10.0;
    let usage = (started.elapsed().as_secs_f64() / tick_duration * 100.0) as f32;
    set_world_numeric_field(state, "tick_usage", usage);
}

pub(crate) fn schedule_frames(state: &mut ExecutionState, frames: Vec<CallFrame>, delay: f32) {
    let sequence = state.scheduler_sequence;
    state.scheduler_sequence = state.scheduler_sequence.saturating_add(1);
    let tick_lag = world_numeric_field(state, "tick_lag")
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(1.0);
    let delay_ticks = if delay.is_finite() && delay > 0.0 {
        ((f64::from(delay) / f64::from(tick_lag)).floor() as u64).max(1)
    } else {
        0
    };
    state.scheduled_spawns.push(ScheduledSpawn {
        due_tick: state.scheduler_tick.saturating_add(delay_ticks),
        sequence,
        frames: OwnedContinuation::new(VmContinuationId(sequence), frames),
    });
}
