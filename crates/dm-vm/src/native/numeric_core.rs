//! Numeric-core acceleration: world-datum access, the clock, and the numeric
//! dispatch/loop fast paths.


use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use crate::builtins::{execute_standard_builtin, is_subtype};
use crate::bytecode::{CompoundAssignmentOperator, Instruction, Module, ProcedureId, Program, TypePredicateKind};
use crate::value_ops::{canonicalize_owned_value, canonicalize_value, dm_list_length_number, dynamic_call_target_named, get_step_builtin, is_area_type_path, is_turf_type_path, logical_or_empty_list_field, logical_or_empty_list_index, pop, read_list_value, runtime_truthy, stringify_dm_value, values_equal};
use crate::{CallFrame, ExecutionState, PackedNumericState, declared_argument_count, frame_context, make_frame};
use dm_jit::{CompiledNumericTrace, NumericExecutionState, NumericInstruction, NumericRunOutcome, compile_numeric_field_trace, compile_numeric_trace, compile_safe_rooted_block};
use dm_value::{DatumId, FieldName, PackedValue, TypePath, Value, ValueError};

fn world_datum(state: &ExecutionState) -> Option<DatumId> {
    static WORLD: std::sync::OnceLock<FieldName> = std::sync::OnceLock::new();
    state
        .global(WORLD.get_or_init(|| FieldName::parse("world").expect("built-in world global")))
        .and_then(|value| match value {
            Value::Datum(world) => Some(*world),
            _ => None,
        })
}

pub(crate) fn cached_world_numeric_field(name: &str) -> Option<&'static FieldName> {
    static TICK_LAG: std::sync::OnceLock<FieldName> = std::sync::OnceLock::new();
    static TICK_USAGE: std::sync::OnceLock<FieldName> = std::sync::OnceLock::new();
    static TIME: std::sync::OnceLock<FieldName> = std::sync::OnceLock::new();
    static TIMEOFDAY: std::sync::OnceLock<FieldName> = std::sync::OnceLock::new();
    let (slot, canonical) = match name {
        "tick_lag" => (&TICK_LAG, "tick_lag"),
        "tick_usage" => (&TICK_USAGE, "tick_usage"),
        "time" => (&TIME, "time"),
        "timeofday" => (&TIMEOFDAY, "timeofday"),
        _ => return None,
    };
    Some(slot.get_or_init(|| FieldName::parse(canonical).expect("built-in world numeric field")))
}

pub(crate) fn world_numeric_field(state: &ExecutionState, name: &str) -> Option<f32> {
    let parsed;
    let field = if let Some(field) = cached_world_numeric_field(name) {
        field
    } else {
        parsed = FieldName::parse(name).ok()?;
        &parsed
    };
    state
        .heap
        .datum_field(world_datum(state)?, field)
        .ok()?
        .as_number()
}

// MAPLOADING_CHECK_TICK expands this comparison into five bytecodes at every
// hot map/cache/atom loop site. When its condition is false, jumping directly
// to the compiler-provided false target is exactly equivalent and avoids four
// additional dispatches. Non-numeric or structurally different cases stay in
// the reference interpreter, including the complete stoplag/yielding branch.
pub(crate) fn false_tick_check_target(
    instructions: &[Instruction],
    instruction_index: usize,
    state: &ExecutionState,
) -> Option<usize> {
    let [
        Instruction::LoadGlobal(world_name),
        Instruction::LoadField(tick_usage_name),
        Instruction::LoadGlobal(limit_name),
        Instruction::Greater,
        Instruction::JumpIfFalse(target),
    ] = instructions.get(instruction_index..instruction_index.checked_add(5)?)?
    else {
        return None;
    };
    if world_name.as_str() != "world" || tick_usage_name.as_str() != "tick_usage" {
        return None;
    }
    let Value::Datum(world) = state.global(world_name)? else {
        return None;
    };
    if world_datum(state) != Some(*world) {
        return None;
    }
    let usage = datum_field_or_shared(state, *world, tick_usage_name)
        .ok()?
        .as_number()?;
    if *target > instructions.len() {
        return None;
    }
    let limit = state.global(limit_name)?.as_number()?;
    (!(usage > limit)).then_some(*target)
}

pub(crate) fn try_run_numeric_loop_branch(
    program: &Program,
    frame: &mut CallFrame,
    remaining_steps: u64,
    state: &ExecutionState,
) -> Option<u64> {
    const ACCOUNTED_STEPS: u64 = 4;
    if remaining_steps < ACCOUNTED_STEPS {
        return None;
    }
    let instruction = frame.instruction;
    let instructions = program.instructions.get(instruction..instruction + 4)?;
    let Instruction::LoadLocal(left_slot) = instructions[0] else {
        return None;
    };
    let left = frame.locals.get(usize::from(left_slot))?.clone();
    let right = match &instructions[1] {
        Instruction::LoadLocal(slot) => frame.locals.get(usize::from(*slot))?.clone(),
        Instruction::PushNumber(number) => Value::Number(*number),
        Instruction::ListLengthLocal(slot) => {
            let mut receiver = frame.locals.get(usize::from(*slot))?.clone();
            if let Value::List(list) = receiver
                && state.reference_lists.contains(&list)
            {
                receiver = state.heap.list(list).ok()?.get(1).ok()?.clone();
            }
            let receiver = canonicalize_owned_value(&state.heap, receiver);
            let length = match receiver {
                Value::Null => 0,
                Value::List(list) => state.heap.list(list).ok()?.len(),
                _ => return None,
            };
            Value::number(dm_list_length_number(length))
        }
        _ => return None,
    };
    let comparison = compare_values(&left, &right).ok()??;
    let condition = match instructions[2] {
        Instruction::Less => comparison.is_lt(),
        Instruction::LessEqual => comparison.is_le(),
        Instruction::Greater => comparison.is_gt(),
        Instruction::GreaterEqual => comparison.is_ge(),
        _ => return None,
    };
    let Instruction::JumpIfFalse(target) = instructions[3] else {
        return None;
    };
    frame.instruction = if condition { instruction + 4 } else { target };
    Some(ACCOUNTED_STEPS)
}

pub(crate) fn try_run_numeric_local_update(
    program: &Program,
    frame: &mut CallFrame,
    remaining_steps: u64,
    state: &ExecutionState,
) -> Option<u64> {
    const ACCOUNTED_STEPS: u64 = 4;
    if remaining_steps < ACCOUNTED_STEPS {
        return None;
    }
    let instruction = frame.instruction;
    let instructions = program.instructions.get(instruction..instruction + 4)?;
    let Instruction::LoadLocal(load_slot) = instructions[0] else {
        return None;
    };
    let Instruction::PushNumber(delta) = instructions[1] else {
        return None;
    };
    let Instruction::StoreLocal(store_slot) = instructions[3] else {
        return None;
    };
    let store_index = usize::from(store_slot);
    let store = frame.locals.get(store_index)?;
    if store_index < frame.declared_argument_count
        || frame.static_locals.contains(&store_slot)
        || matches!(store, Value::List(list) if state.reference_lists.contains(list))
    {
        return None;
    }
    let mut current = frame.locals.get(usize::from(load_slot))?.clone();
    if let Value::List(list) = current
        && state.reference_lists.contains(&list)
    {
        current = state.heap.list(list).ok()?.get(1).ok()?.clone();
    }
    let current = canonicalize_owned_value(&state.heap, current);
    let current = match current {
        Value::Null => 0.0,
        Value::Number(number) => number.to_f32(),
        _ => return None,
    };
    let delta = delta.to_f32();
    let updated = match instructions[2] {
        Instruction::Add => current + delta,
        Instruction::Subtract => current - delta,
        _ => return None,
    };
    frame.locals[store_index] = Value::number(updated);
    frame.instruction = instruction + 4;
    Some(ACCOUNTED_STEPS)
}

fn quick_numeric_value(value: &Value) -> Option<f32> {
    match value {
        Value::Null => Some(0.0),
        Value::Number(number) => Some(number.to_f32()),
        _ => None,
    }
}

pub(crate) fn numeric_dispatch_candidate(instruction: &Instruction) -> bool {
    matches!(
        instruction,
        Instruction::PushNull
            | Instruction::PushNumber(_)
            | Instruction::PushText(_)
            | Instruction::LoadLocal(_)
            | Instruction::StoreLocal(_)
            | Instruction::LoadResult
            | Instruction::StoreResult
            | Instruction::Duplicate
            | Instruction::Pop
            | Instruction::ListLengthLocal(_)
            | Instruction::Add
            | Instruction::Subtract
            | Instruction::Multiply
            | Instruction::Divide
            | Instruction::Less
            | Instruction::LessEqual
            | Instruction::Greater
            | Instruction::GreaterEqual
            | Instruction::Negate
            | Instruction::Not
            | Instruction::And
            | Instruction::Or
            | Instruction::JumpIfFalse(_)
            | Instruction::Jump(_)
    )
}

pub(crate) fn try_run_numeric_dispatch_block(
    program: &Program,
    frame: &mut CallFrame,
    max_steps: u64,
    state: &ExecutionState,
) -> Option<u64> {
    static PACKED_FORCED: OnceLock<bool> = OnceLock::new();
    static PACKED_DISABLED: OnceLock<bool> = OnceLock::new();
    let disabled = *PACKED_DISABLED.get_or_init(|| {
        std::env::var_os("DREAM64_DISABLE_PACKED_VALUE_STACK").is_some_and(|value| {
            !matches!(value.to_string_lossy().trim(), "" | "0" | "false" | "no")
        })
    });
    let forced = *PACKED_FORCED.get_or_init(|| {
        std::env::var_os("DREAM64_ENABLE_PACKED_VALUE_STACK").is_some_and(|value| {
            !matches!(value.to_string_lossy().trim(), "" | "0" | "false" | "no")
        })
    });
    if !disabled {
        let retained = frame
            .cold()
            .is_some_and(|cold| cold.packed_numeric_state.is_some());
        if retained || forced || predicts_profitable_packed_run(program, frame.instruction) {
            PACKED_ADAPTIVE_ENTRIES.fetch_add(1, Ordering::Relaxed);
            if let Some(steps) =
                try_run_packed_numeric_dispatch_block(program, frame, max_steps, state)
            {
                return Some(steps);
            }
        } else {
            PACKED_ADAPTIVE_DECLINES.fetch_add(1, Ordering::Relaxed);
        }
    }
    try_run_rich_numeric_dispatch_block(program, frame, max_steps, state)
}

const PACKED_ADAPTIVE_MIN_STEPS: usize = 24;
static PACKED_ADAPTIVE_ENTRIES: AtomicU64 = AtomicU64::new(0);
static PACKED_ADAPTIVE_DECLINES: AtomicU64 = AtomicU64::new(0);

/// Counts adaptive packed-dispatch entry attempts and short-run declines.
#[must_use]
pub fn packed_dispatch_counters() -> (u64, u64) {
    (
        PACKED_ADAPTIVE_ENTRIES.load(Ordering::Relaxed),
        PACKED_ADAPTIVE_DECLINES.load(Ordering::Relaxed),
    )
}

fn predicts_profitable_packed_run(program: &Program, start: usize) -> bool {
    let mut instruction = start;
    for _ in 0..PACKED_ADAPTIVE_MIN_STEPS {
        let Some(opcode) = program.instructions.get(instruction) else {
            return false;
        };
        if !numeric_dispatch_candidate(opcode)
            || matches!(
                opcode,
                Instruction::PushText(_) | Instruction::ListLengthLocal(_)
            )
        {
            return false;
        }
        match opcode {
            Instruction::Jump(target) => instruction = *target,
            // Conditional control needs runtime stack information; require an
            // already-retained packed state instead of guessing profitability.
            Instruction::JumpIfFalse(_) => return false,
            _ => instruction += 1,
        }
    }
    true
}

pub(crate) fn try_run_packed_numeric_dispatch_block(
    program: &Program,
    frame: &mut CallFrame,
    max_steps: u64,
    _state: &ExecutionState,
) -> Option<u64> {
    if max_steps == 0 {
        return None;
    }
    let mut packed = frame
        .take_packed_numeric_state()
        .or_else(|| PackedNumericState::from_rich(frame))?;
    let mut steps = 0_u64;
    while steps < max_steps {
        let Some(instruction) = program.instructions.get(frame.instruction) else {
            break;
        };
        let mut advance = true;
        match instruction {
            Instruction::PushNull => packed.stack.push(PackedValue::null()),
            Instruction::PushNumber(number) => {
                packed.stack.push(PackedValue::number_bits(*number));
            }
            Instruction::LoadLocal(slot) => {
                let Some(value) = packed.locals.get(usize::from(*slot)).copied() else {
                    break;
                };
                packed.stack.push(value);
            }
            Instruction::StoreLocal(slot) => {
                let local_index = usize::from(*slot);
                let Some(local) = packed.locals.get_mut(local_index) else {
                    break;
                };
                if local_index < frame.declared_argument_count || frame.static_locals.contains(slot)
                {
                    break;
                }
                let Some(value) = packed.stack.pop() else {
                    break;
                };
                *local = value;
            }
            Instruction::LoadResult => {
                packed.stack.push(packed.result);
            }
            Instruction::StoreResult => {
                let Some(value) = packed.stack.pop() else {
                    break;
                };
                packed.result = value;
            }
            Instruction::Duplicate => {
                let Some(value) = packed.stack.last().copied() else {
                    break;
                };
                packed.stack.push(value);
            }
            Instruction::Pop => {
                if packed.stack.pop().is_none() {
                    break;
                }
            }
            Instruction::ListLengthLocal(_) => break,
            Instruction::Add
            | Instruction::Subtract
            | Instruction::Multiply
            | Instruction::Divide
            | Instruction::And
            | Instruction::Or
            | Instruction::Less
            | Instruction::LessEqual
            | Instruction::Greater
            | Instruction::GreaterEqual => {
                let len = packed.stack.len();
                if len < 2 {
                    break;
                }
                let (Some(left), Some(right)) = (
                    packed.stack[len - 2].as_number_or_null(),
                    packed.stack[len - 1].as_number_or_null(),
                ) else {
                    break;
                };
                if matches!(
                    instruction,
                    Instruction::Less
                        | Instruction::LessEqual
                        | Instruction::Greater
                        | Instruction::GreaterEqual
                ) && left.partial_cmp(&right).is_none()
                {
                    break;
                }
                let value = match instruction {
                    Instruction::Add => left + right,
                    Instruction::Subtract => left - right,
                    Instruction::Multiply => left * right,
                    Instruction::Divide => left / right,
                    Instruction::And => f32::from(left != 0.0 && right != 0.0),
                    Instruction::Or => f32::from(left != 0.0 || right != 0.0),
                    Instruction::Less => f32::from(left < right),
                    Instruction::LessEqual => f32::from(left <= right),
                    Instruction::Greater => f32::from(left > right),
                    Instruction::GreaterEqual => f32::from(left >= right),
                    _ => unreachable!(),
                };
                packed.stack.truncate(len - 2);
                packed.stack.push(PackedValue::number(value));
            }
            Instruction::Negate | Instruction::Not => {
                let Some(last) = packed.stack.last_mut() else {
                    break;
                };
                let Some(value) = last.as_number_or_null() else {
                    break;
                };
                let value = if matches!(instruction, Instruction::Negate) {
                    -value
                } else {
                    f32::from(value == 0.0)
                };
                *last = PackedValue::number(value);
            }
            Instruction::JumpIfFalse(target) => {
                let Some(condition) = packed
                    .stack
                    .last()
                    .and_then(|value| value.as_number_or_null())
                else {
                    break;
                };
                if *target >= program.instructions.len() {
                    break;
                }
                packed.stack.pop();
                if condition == 0.0 {
                    frame.instruction = *target;
                    advance = false;
                }
            }
            Instruction::Jump(target) => {
                if *target >= program.instructions.len() {
                    break;
                }
                frame.instruction = *target;
                advance = false;
            }
            _ => break,
        }
        steps += 1;
        if advance {
            frame.instruction += 1;
        }
    }
    if steps == 0 {
        packed.materialize(frame);
        frame.set_packed_numeric_state(None);
        return None;
    }
    if steps == max_steps {
        frame.set_packed_numeric_state(Some(packed));
    } else {
        packed.materialize(frame);
        frame.set_packed_numeric_state(None);
    }
    Some(steps)
}

pub(crate) fn try_run_rich_numeric_dispatch_block(
    program: &Program,
    frame: &mut CallFrame,
    max_steps: u64,
    state: &ExecutionState,
) -> Option<u64> {
    if max_steps == 0 {
        return None;
    }
    let mut steps = 0_u64;
    while steps < max_steps {
        let instruction = program.instructions.get(frame.instruction)?;
        let mut advance = true;
        match instruction {
            Instruction::PushNull => frame.stack.push(Value::Null),
            Instruction::PushNumber(number) => frame.stack.push(Value::Number(*number)),
            Instruction::PushText(text) => frame.stack.push(Value::Text(Arc::clone(text))),
            Instruction::LoadLocal(slot) => {
                let mut value = frame.locals.get(usize::from(*slot))?.clone();
                if let Value::List(list) = value
                    && state.reference_lists.contains(&list)
                {
                    let Ok(reference) = state.heap.list(list) else {
                        break;
                    };
                    let Ok(referenced) = reference.get(1) else {
                        break;
                    };
                    value = referenced.clone();
                }
                frame
                    .stack
                    .push(canonicalize_owned_value(&state.heap, value));
            }
            Instruction::StoreLocal(slot) => {
                let local_index = usize::from(*slot);
                let Some(local) = frame.locals.get(local_index) else {
                    break;
                };
                if local_index < frame.declared_argument_count
                    || frame.static_locals.contains(slot)
                    || matches!(local, Value::List(list) if state.reference_lists.contains(list))
                {
                    break;
                }
                let Some(value) = frame.stack.pop() else {
                    break;
                };
                frame.locals[local_index] = value;
            }
            Instruction::LoadResult => frame.stack.push(frame.result.clone()),
            Instruction::StoreResult => frame.result = frame.stack.pop()?,
            Instruction::Duplicate => frame.stack.push(frame.stack.last()?.clone()),
            Instruction::Pop => {
                frame.stack.pop()?;
            }
            Instruction::ListLengthLocal(slot) => {
                let mut receiver = frame.locals.get(usize::from(*slot))?.clone();
                if let Value::List(list) = receiver
                    && state.reference_lists.contains(&list)
                {
                    let Ok(reference) = state.heap.list(list) else {
                        break;
                    };
                    let Ok(referenced) = reference.get(1) else {
                        break;
                    };
                    receiver = referenced.clone();
                }
                let receiver = canonicalize_owned_value(&state.heap, receiver);
                let length = match receiver {
                    Value::Null => 0,
                    Value::List(list) => state.heap.list(list).ok()?.len(),
                    _ => break,
                };
                frame
                    .stack
                    .push(Value::number(dm_list_length_number(length)));
            }
            Instruction::Add
            | Instruction::Subtract
            | Instruction::Multiply
            | Instruction::Divide => {
                let len = frame.stack.len();
                if len < 2 {
                    break;
                }
                let left = quick_numeric_value(&frame.stack[len - 2]);
                let right = quick_numeric_value(&frame.stack[len - 1]);
                let (Some(left), Some(right)) = (left, right) else {
                    break;
                };
                let value = match instruction {
                    Instruction::Add => left + right,
                    Instruction::Subtract => left - right,
                    Instruction::Multiply => left * right,
                    Instruction::Divide => left / right,
                    _ => unreachable!(),
                };
                frame.stack.truncate(len - 2);
                frame.stack.push(Value::number(value));
            }
            Instruction::Less
            | Instruction::LessEqual
            | Instruction::Greater
            | Instruction::GreaterEqual => {
                let len = frame.stack.len();
                if len < 2 {
                    break;
                }
                let left = &frame.stack[len - 2];
                let right = &frame.stack[len - 1];
                let Ok(Some(comparison)) = compare_values(left, right) else {
                    break;
                };
                let value = match instruction {
                    Instruction::Less => comparison.is_lt(),
                    Instruction::LessEqual => comparison.is_le(),
                    Instruction::Greater => comparison.is_gt(),
                    Instruction::GreaterEqual => comparison.is_ge(),
                    _ => unreachable!(),
                };
                frame.stack.truncate(len - 2);
                frame.stack.push(Value::number(f32::from(value)));
            }
            Instruction::Negate => {
                let value = quick_numeric_value(frame.stack.last()?)?;
                *frame.stack.last_mut()? = Value::number(-value);
            }
            Instruction::Not => {
                let value = quick_numeric_value(frame.stack.last()?)?;
                *frame.stack.last_mut()? = Value::number(f32::from(value == 0.0));
            }
            Instruction::And | Instruction::Or => {
                let len = frame.stack.len();
                if len < 2 {
                    break;
                }
                let left = quick_numeric_value(&frame.stack[len - 2]);
                let right = quick_numeric_value(&frame.stack[len - 1]);
                let (Some(left), Some(right)) = (left, right) else {
                    break;
                };
                let value = if matches!(instruction, Instruction::And) {
                    left != 0.0 && right != 0.0
                } else {
                    left != 0.0 || right != 0.0
                };
                frame.stack.truncate(len - 2);
                frame.stack.push(Value::number(f32::from(value)));
            }
            Instruction::JumpIfFalse(target) => {
                let Some(condition) = frame.stack.last().and_then(quick_numeric_value) else {
                    break;
                };
                frame.stack.pop();
                if condition == 0.0 {
                    if *target >= program.instructions.len() {
                        break;
                    }
                    frame.instruction = *target;
                    advance = false;
                }
            }
            Instruction::Jump(target) => {
                if *target >= program.instructions.len() {
                    break;
                }
                frame.instruction = *target;
                advance = false;
            }
            _ => break,
        }
        steps += 1;
        if advance {
            frame.instruction += 1;
        }
    }
    (steps > 0).then_some(steps)
}

pub(crate) fn set_world_numeric_field(state: &mut ExecutionState, name: &str, value: f32) {
    let Some(world) = world_datum(state) else {
        return;
    };
    let field = cached_world_numeric_field(name)
        .cloned()
        .unwrap_or_else(|| FieldName::parse(name).expect("world numeric field"));
    let _ = state
        .heap
        .set_datum_field(world, field, Value::number(value));
}

pub(crate) fn advance_headless_world_clock(state: &mut ExecutionState, ticks: u64) {
    if ticks == 0 {
        return;
    }
    let tick_lag = world_numeric_field(state, "tick_lag")
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(1.0);
    let elapsed = (ticks as f64 * f64::from(tick_lag)) as f32;
    let time = world_numeric_field(state, "time").unwrap_or(0.0) + elapsed;
    let timeofday =
        (world_numeric_field(state, "timeofday").unwrap_or(0.0) + elapsed).rem_euclid(864_000.0);
    set_world_numeric_field(state, "time", time);
    set_world_numeric_field(state, "timeofday", timeofday);
}
