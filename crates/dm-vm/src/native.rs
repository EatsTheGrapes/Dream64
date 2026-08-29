//! Embedded native accelerations for the reference interpreter.
//!
//! The interpreter is contract-bound to a deterministic bytecode algorithm,
//! but for a narrow set of *canonical* procedure shapes it shortcuts straight
//! into native Rust when the compiled program provably matches the shape.
//! World-clock numerics, `type2parent`/`istext` compaction, the camera chunk,
//! `RegisterSignal`, rooted-list, and lumcount loops, and — for Monke's `.dmm`
//! pipeline — the TGM/ruin/dmm-load drives all live here so that trust
//! criteria, instrumentation counters, and canonical matchers stay co-located
//! with the code that uses them.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use crate::builtins;
use crate::builtins::{execute_standard_builtin, is_subtype};
use crate::bytecode::{
    CompoundAssignmentOperator, Instruction, Module, ProcedureId, Program, TypePredicateKind,
};
use crate::compact_wordcode;
use crate::compile::compile_procedure;
use crate::tgm_planner;
use crate::value_ops::{
    ExecutionContext, assign_datum_or_shared_field, canonicalize_owned_value, canonicalize_value,
    compare_values, datum_field_or_initial, datum_field_or_shared, dm_list_length_number,
    dynamic_call_target_named, get_step_builtin, is_area_type_path, is_turf_type_path,
    logical_or_empty_list_field, logical_or_empty_list_index, pop, read_list_value, runtime_truthy,
    stringify_dm_value, values_equal, write_list_value,
};
use crate::{
    CallFrame, ExecutionState, PackedNumericState, RuinCandidateScan, TgmLoadContinuation,
    TgmLoadPhase, declared_argument_count, frame_context, make_frame,
};

use dm_jit::{
    CompiledNumericTrace, CompiledRootedBlock, NumericInstruction, NumericRunOutcome,
    RootedBlockOutcome, compile_numeric_field_trace, compile_numeric_trace,
    compile_safe_rooted_block,
};
use dm_value::{DatumId, FieldName, ListId, PackedValue, TypePath, Value, ValueError};
use smallvec::SmallVec;

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

pub(crate) const CANONICAL_TYPE2PARENT_SOURCE: &str = "/proc/type2parent(child)\n\
\tvar/string_type = \"[child]\"\n\
\tvar/last_slash = findlasttext(string_type, \"/\")\n\
\tif(last_slash == 1)\n\
\t\tswitch(child)\n\
\t\t\tif(/datum)\n\
\t\t\t\treturn null\n\
\t\t\tif(/obj, /mob)\n\
\t\t\t\treturn /atom/movable\n\
\t\t\tif(/area, /turf)\n\
\t\t\t\treturn /atom\n\
\t\t\telse\n\
\t\t\t\treturn /datum\n\
\treturn text2path(copytext(string_type, 1, last_slash))\n";

pub(crate) fn canonical_type2parent_program(program: &Program) -> bool {
    static CANONICAL: OnceLock<Program> = OnceLock::new();
    let canonical = CANONICAL.get_or_init(|| {
        let syntax = dm_syntax::parse(CANONICAL_TYPE2PARENT_SOURCE)
            .expect("canonical type2parent source is valid");
        compile_procedure(
            syntax
                .definitions
                .first()
                .expect("canonical type2parent definition exists"),
        )
        .expect("canonical type2parent procedure compiles")
    });
    program.wait_for == canonical.wait_for
        && program.parameter_count == canonical.parameter_count
        && program.parameter_names == canonical.parameter_names
        && program.local_count == canonical.local_count
        && program.instructions == canonical.instructions
}

const CANONICAL_MONKE_TGM_LOAD_DIGEST: [u8; 32] = [
    0x14, 0xf7, 0x2e, 0x36, 0x1e, 0x09, 0xa4, 0x7b, 0x78, 0x60, 0x4d, 0x87, 0x1a, 0x22, 0xdb, 0x79,
    0xc1, 0x1f, 0xd6, 0x05, 0x03, 0xa7, 0x31, 0x8a, 0x22, 0xac, 0x4b, 0xab, 0xac, 0xfa, 0xfb, 0x72,
];
pub(crate) const CANONICAL_MONKE_BUILD_COORDINATE_DIGEST: [u8; 32] = [
    0x9d, 0xae, 0x83, 0x67, 0x40, 0x60, 0x6e, 0xaf, 0xa9, 0x0d, 0x8b, 0xc3, 0xa6, 0x4b, 0xfc, 0x23,
    0x3a, 0x6a, 0x3f, 0x35, 0x55, 0x7f, 0x4b, 0x52, 0xb8, 0xc2, 0xbc, 0xf5, 0xb2, 0xe2, 0x66, 0x79,
];
static NATIVE_TGM_LOAD_ACTIVATIONS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static NATIVE_TGM_PLANNED_CELLS: AtomicU64 = AtomicU64::new(0);
static NATIVE_TGM_PLANNED_SAFEPOINTS: AtomicU64 = AtomicU64::new(0);
static NATIVE_TGM_COMMITTED_CELLS: AtomicU64 = AtomicU64::new(0);
static NATIVE_TGM_BUILD_CACHE_MEMBERS: AtomicU64 = AtomicU64::new(0);
static NATIVE_TGM_BUILD_CACHE_LOGICAL_STEPS: AtomicU64 = AtomicU64::new(0);
static NATIVE_TGM_TARGET_RESOLUTIONS: AtomicU64 = AtomicU64::new(0);
static NATIVE_TGM_TARGET_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static NATIVE_TGM_COMMIT_SAMPLES: OnceLock<std::sync::Mutex<Vec<String>>> = OnceLock::new();
static NATIVE_BUILD_COORDINATE_PREFIX_ACTIVATIONS: AtomicU64 = AtomicU64::new(0);
static NATIVE_BUILD_COORDINATE_PREFIX_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static NATIVE_RUIN_BATCH_ACTIVATIONS: AtomicU64 = AtomicU64::new(0);
static NATIVE_RUIN_BATCH_LOGICAL_STEPS: AtomicU64 = AtomicU64::new(0);
static NATIVE_DISCOVER_OFFSET_ACTIVATIONS: AtomicU64 = AtomicU64::new(0);

/// Returns the process-wide number of canonical `_tgm_load` frames that
/// actually installed the native commit sidecar.
#[must_use]
pub fn native_tgm_load_activations() -> u64 {
    NATIVE_TGM_LOAD_ACTIVATIONS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Returns `(planned cells, elided-space safepoints, completed cell commits)`.
#[must_use]
pub fn native_tgm_load_metrics() -> (u64, u64, u64) {
    (
        NATIVE_TGM_PLANNED_CELLS.load(Ordering::Relaxed),
        NATIVE_TGM_PLANNED_SAFEPOINTS.load(Ordering::Relaxed),
        NATIVE_TGM_COMMITTED_CELLS.load(Ordering::Relaxed),
    )
}

/// Returns `(simple canonical members, replaced logical instructions)`.
#[must_use]
pub fn native_tgm_build_cache_metrics() -> (u64, u64) {
    (
        NATIVE_TGM_BUILD_CACHE_MEMBERS.load(Ordering::Relaxed),
        NATIVE_TGM_BUILD_CACHE_LOGICAL_STEPS.load(Ordering::Relaxed),
    )
}

/// Returns `(dynamic build_coordinate resolutions, validated cache hits)`.
#[must_use]
pub fn native_tgm_target_cache_metrics() -> (u64, u64) {
    (
        NATIVE_TGM_TARGET_RESOLUTIONS.load(Ordering::Relaxed),
        NATIVE_TGM_TARGET_CACHE_HITS.load(Ordering::Relaxed),
    )
}

/// Returns bounded post-commit samples from native TGM loading.
#[must_use]
pub fn native_tgm_commit_samples() -> Vec<String> {
    NATIVE_TGM_COMMIT_SAMPLES
        .get()
        .and_then(|samples| samples.lock().ok().map(|samples| samples.clone()))
        .unwrap_or_default()
}

/// Returns `(activations, guarded fallbacks)` for the canonical map-cell prefix.
#[must_use]
pub fn native_build_coordinate_prefix_metrics() -> (u64, u64) {
    (
        NATIVE_BUILD_COORDINATE_PREFIX_ACTIVATIONS.load(Ordering::Relaxed),
        NATIVE_BUILD_COORDINATE_PREFIX_FALLBACKS.load(Ordering::Relaxed),
    )
}
static NATIVE_RUIN_SCAN_ACTIVATIONS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static NATIVE_RUIN_SCAN_CELLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static NATIVE_RUIN_SCAN_REJECTIONS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static NATIVE_RUIN_SCAN_SUCCESSES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static NATIVE_RUIN_REJECTION_CACHE_HITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static NATIVE_RUIN_FLAG_REJECTIONS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static NATIVE_RUIN_AREA_REJECTIONS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static NATIVE_RUIN_AREA_REJECTION_SAMPLES: std::sync::OnceLock<std::sync::Mutex<Vec<String>>> =
    std::sync::OnceLock::new();

/// Process-wide guarded ruin-candidate scan counters.
#[must_use]
pub fn native_ruin_scan_metrics() -> (u64, u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        NATIVE_RUIN_SCAN_ACTIVATIONS.load(Relaxed),
        NATIVE_RUIN_SCAN_CELLS.load(Relaxed),
        NATIVE_RUIN_SCAN_REJECTIONS.load(Relaxed),
        NATIVE_RUIN_SCAN_SUCCESSES.load(Relaxed),
    )
}

/// Returns the number of ruin candidates rejected by a revalidated cached witness.
#[must_use]
pub fn native_ruin_rejection_cache_hits() -> u64 {
    NATIVE_RUIN_REJECTION_CACHE_HITS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Returns `(NO_RUINS flag rejects, area-whitelist rejects)`.
#[must_use]
pub fn native_ruin_rejection_causes() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        NATIVE_RUIN_FLAG_REJECTIONS.load(Relaxed),
        NATIVE_RUIN_AREA_REJECTIONS.load(Relaxed),
    )
}

/// Returns bounded diagnostics captured for area-whitelist rejection.
#[must_use]
pub fn native_ruin_area_rejection_samples() -> Vec<String> {
    NATIVE_RUIN_AREA_REJECTION_SAMPLES
        .get()
        .and_then(|samples| samples.lock().ok().map(|samples| samples.clone()))
        .unwrap_or_default()
}

/// Returns process-wide guarded ruin-scan batch and logical-step counters.
#[must_use]
pub fn native_ruin_batch_metrics() -> (u64, u64) {
    (
        NATIVE_RUIN_BATCH_ACTIVATIONS.load(Ordering::Relaxed),
        NATIVE_RUIN_BATCH_LOGICAL_STEPS.load(Ordering::Relaxed),
    )
}

/// Returns the process-wide number of guarded map-template offset scans.
#[must_use]
pub fn native_discover_offset_activations() -> u64 {
    NATIVE_DISCOVER_OFFSET_ACTIVATIONS.load(Ordering::Relaxed)
}

const CANONICAL_MONKE_RUIN_TRY_TO_PLACE_DIGEST: [u8; 32] = [
    0x03, 0xab, 0x38, 0x41, 0x98, 0x62, 0xc9, 0xdb, 0xd2, 0x19, 0x01, 0x39, 0xdb, 0x4c, 0x9d, 0xa5,
    0xc9, 0xe3, 0x02, 0x3b, 0x65, 0xce, 0xe8, 0x9c, 0x8c, 0xd4, 0x65, 0xf7, 0xf6, 0x9f, 0x5b, 0x5e,
];
const CANONICAL_MONKE_GET_AFFECTED_TURFS_DIGEST: [u8; 32] = [
    0x45, 0x27, 0xea, 0x56, 0x5d, 0xcc, 0xc3, 0xef, 0xd6, 0x5d, 0xbb, 0xfe, 0xf5, 0x85, 0x92, 0xd1,
    0x08, 0xe4, 0xa9, 0x04, 0x02, 0x1e, 0x7c, 0x8a, 0xf3, 0xcd, 0x46, 0x20, 0x98, 0x9a, 0x70, 0x80,
];

fn trusted_get_affected_turfs_target(module: &Module) -> bool {
    let procedure = ProcedureId(19_821);
    let Some(program) = module.procedure(procedure) else {
        return false;
    };
    module.procedure_path(procedure).is_some_and(|path| {
        path.split('@').next() == Some("/datum/map_template/proc/get_affected_turfs")
    }) && program.parameter_count == 2
        && program.local_count == 5
        && program.instructions.len() == 51
        && matches!(
            program.instructions.get(25),
            Some(Instruction::Locate { argument_count: 3 })
        )
        && matches!(
            program.instructions.get(49),
            Some(Instruction::Block { argument_count: 2 })
        )
        && matches!(program.instructions.get(50), Some(Instruction::Return))
        && module.procedure_semantic_digest(procedure)
            == Some(CANONICAL_MONKE_GET_AFFECTED_TURFS_DIGEST)
}

fn trusted_ruin_try_to_place_target(
    module: &Module,
    procedure: ProcedureId,
    program: &Program,
) -> bool {
    module.procedure_path(procedure).is_some_and(|path| {
        path.split('@').next() == Some("/datum/map_template/ruin/proc/try_to_place")
    }) && program.parameter_count == 4
        && program.local_count == 29
        && program.instructions.len() == 244
        && matches!(
            program.instructions.get(74),
            Some(Instruction::NextLocalListIteration {
                list_slot: 13,
                index_slot: 14,
                item_slot: 12,
                exit: 116
            })
        )
        && matches!(program.instructions.get(85), Some(Instruction::LoadDeclaredField(field)) if field.as_str() == "turf_flags")
        && matches!(program.instructions.get(95), Some(Instruction::Call { procedure, argument_count: 1, .. }) if procedure.index() == 68_206)
        && matches!(
            program.instructions.get(110),
            Some(Instruction::SetListIndex)
        )
        && matches!(program.instructions.get(115), Some(Instruction::Jump(74)))
        && module.procedure_semantic_digest(procedure)
            == Some(CANONICAL_MONKE_RUIN_TRY_TO_PLACE_DIGEST)
}

pub(crate) fn try_run_ruin_affected_turfs_batch(
    module: &Module,
    procedure: ProcedureId,
    program: &Program,
    frame: &mut CallFrame,
    budget: u64,
    state: &mut ExecutionState,
) -> Option<u64> {
    if !trusted_ruin_try_to_place_target(module, procedure, program) {
        return None;
    }
    let steps = run_ruin_affected_turfs_batch(frame, budget, state)?;
    NATIVE_RUIN_BATCH_ACTIVATIONS.fetch_add(1, Ordering::Relaxed);
    NATIVE_RUIN_BATCH_LOGICAL_STEPS.fetch_add(steps, Ordering::Relaxed);
    Some(steps)
}

pub(crate) fn run_ruin_affected_turfs_batch(
    frame: &mut CallFrame,
    budget: u64,
    state: &mut ExecutionState,
) -> Option<u64> {
    if frame.instruction != 74 || budget < 15 || !frame.stack.is_empty() {
        return None;
    }
    let Value::List(snapshot) = frame.locals.get(13)?.clone() else {
        return None;
    };
    let Value::List(affected_areas) = frame.locals.get(11)?.clone() else {
        return None;
    };
    let turf = TypePath::parse("/turf").ok()?;
    let turf_flags = FieldName::parse("turf_flags").ok()?;
    let loc = FieldName::parse("loc").ok()?;
    let mut steps = 0_u64;
    loop {
        let index = tgm_number(frame.locals.get(14)?)?;
        let values = state.heap.list(snapshot).ok()?;
        if index < 1 || index as usize > values.len() {
            if budget - steps < 1 {
                break;
            }
            frame.instruction = 116;
            steps += 1;
            break;
        }
        let check = read_list_value(
            &state.heap,
            snapshot,
            &Value::number(index as f32),
            state.is_associative_list(snapshot),
        )
        .ok()?;
        let is_turf = matches!(check, Value::Datum(datum) if state.heap.datum(datum).is_ok_and(|record| is_subtype(state, record.type_path(), &turf)));
        let flags = match check {
            Value::Datum(datum) if is_turf => datum_field_or_initial(state, datum, &turf_flags)
                .ok()?
                .as_number()
                .unwrap_or(0.0) as i32,
            _ => 0,
        };
        let cost = if !is_turf {
            15
        } else if flags & (1 << 4) != 0 {
            20
        } else {
            // PC95's canonical isarea body executes its builtin and Return in
            // addition to the parent Call instruction represented in the dump.
            41
        };
        if budget - steps < cost {
            break;
        }
        frame.locals[12] = check.clone();
        if !is_turf {
            frame.locals[14] = Value::number((index + 1) as f32);
            steps += cost;
            continue;
        }
        if flags & (1 << 4) != 0 {
            frame.locals[9] = Value::number(0.0);
            frame.instruction = 116;
            steps += cost;
            break;
        }
        let stepped = get_step_builtin(&check, &Value::number(0.0), state).ok()?;
        let area = match stepped {
            Value::Datum(datum) => datum_field_or_initial(state, datum, &loc)
                .ok()
                .unwrap_or(Value::Null),
            _ => Value::Null,
        };
        frame.locals[15] = area.clone();
        let associative = state.is_associative_list(affected_areas);
        write_list_value(
            &mut state.heap,
            affected_areas,
            area,
            Value::number(1.0),
            associative,
        )
        .ok()?;
        frame.locals[14] = Value::number((index + 1) as f32);
        steps += cost;
        if budget - steps < 15 {
            break;
        }
    }
    (steps != 0).then_some(steps)
}

pub(crate) fn canonical_tgm_load_path(module: &Module, procedure: ProcedureId) -> bool {
    module
        .procedure_path(procedure)
        .is_some_and(|path| path.split('@').next() == Some("/datum/parsed_map/proc/_tgm_load"))
}

const CANONICAL_MONKE_TGM_BUILD_CACHE_DIGEST: [u8; 32] = [
    0x9f, 0x69, 0xa0, 0x56, 0xaf, 0xb4, 0xbf, 0xb2, 0x88, 0x92, 0x3a, 0x17, 0x9b, 0x59, 0x8d, 0xc5,
    0xe6, 0x2f, 0x3a, 0x3b, 0xac, 0xac, 0xaa, 0x5d, 0x6f, 0x96, 0xf8, 0x97, 0x21, 0xf5, 0xdb, 0xbc,
];

fn trusted_tgm_load_target(module: &Module, procedure: ProcedureId, program: &Program) -> bool {
    canonical_tgm_load_path(module, procedure)
        && program.parameter_count == 13
        && program.local_count >= 13
        && module.procedure_semantic_digest(procedure) == Some(CANONICAL_MONKE_TGM_LOAD_DIGEST)
}

fn trusted_tgm_build_cache_target(
    module: &Module,
    procedure: ProcedureId,
    program: &Program,
) -> bool {
    let shape = module.procedure_path(procedure).is_some_and(|path| {
        path.split('@').next() == Some("/datum/parsed_map/proc/tgm_build_cache")
    }) && program.parameter_count == 2
        && program.local_count == 24
        && program.instructions.len() == 338;
    if !shape {
        return false;
    }
    thread_local! {
        static TRUSTED: std::cell::RefCell<HashMap<(u64, ProcedureId), bool>> =
            std::cell::RefCell::new(HashMap::new());
    }
    TRUSTED.with(|trusted| {
        *trusted
            .borrow_mut()
            .entry((module.identity.0, procedure))
            .or_insert_with(|| {
                module.procedure_semantic_digest(procedure)
                    == Some(CANONICAL_MONKE_TGM_BUILD_CACHE_DIGEST)
            })
    })
}

pub(crate) fn run_tgm_build_cache_simple_member(
    frame: &mut CallFrame,
    state: &mut ExecutionState,
) -> Option<usize> {
    if frame.instruction != 98 || !frame.stack.is_empty() {
        return None;
    }
    if runtime_truthy(&state.heap, frame.locals.get(10).unwrap_or(&Value::Null)).unwrap_or(true) {
        return None;
    }
    let Some(Value::Text(line)) = frame.locals.get(17) else {
        return None;
    };
    let line = Arc::clone(line);
    if line.is_empty() || !line.is_ascii() {
        return None;
    }
    let last = line.as_bytes()[line.len() - 1];
    if matches!(last, b';' | b'{' | b'}') {
        return None;
    }
    // `lines` is produced by splittext(model, "\n"), so this is exactly one
    // member. A comma may only be its terminal TGM delimiter; an interior
    // comma is unsupported syntax and must remain on the rich path.
    let path_text = line.strip_suffix(',').unwrap_or(line.as_ref());
    if path_text.is_empty()
        || !path_text.starts_with('/')
        || path_text.trim() != path_text
        || path_text.contains(',')
    {
        return None;
    }
    let Some(path) = state.type_paths.get(path_text).cloned() else {
        return None;
    };
    static ATOM_PATH: OnceLock<TypePath> = OnceLock::new();
    let atom = ATOM_PATH.get_or_init(|| TypePath::parse("/atom").expect("built-in atom path"));
    if !builtins::is_subtype(state, &path, atom) {
        return None;
    }
    let (
        Some(Value::List(default_list)),
        Some(Value::List(wrapped_default)),
        Some(Value::List(members)),
        Some(Value::List(attributes)),
    ) = (
        frame.locals.get(5),
        frame.locals.get(6),
        frame.locals.get(15),
        frame.locals.get(16),
    )
    else {
        return None;
    };
    let (default_list, wrapped_default, members, attributes) =
        (*default_list, *wrapped_default, *members, *attributes);
    if members == attributes
        || members == default_list
        || members == wrapped_default
        || attributes == default_list
        || attributes == wrapped_default
        || state.heap.list(default_list).is_err()
        || state.heap.list(members).is_err()
        || state.heap.list(attributes).is_err()
    {
        return None;
    }
    let Ok(wrapper) = state.heap.list(wrapped_default) else {
        return None;
    };
    if wrapper.len() != 1
        || wrapper.associations().next().is_some()
        || wrapper
            .positions()
            .next()
            .is_none_or(|(_, value)| value != &Value::List(default_list))
    {
        return None;
    }
    state
        .heap
        .list_mut(attributes)
        .expect("validated attributes list")
        .add(Value::List(default_list));
    state
        .heap
        .list_mut(members)
        .expect("validated members list")
        .add(Value::TypePath(path.clone()));
    frame.locals[20] = Value::text((last as char).to_string());
    frame.locals[8] = Value::text(path_text);
    frame.locals[23] = Value::TypePath(path);
    // Continue the rich inner iterator. PC265 is only valid after every
    // newline-delimited member has naturally exhausted.
    frame.instruction = 260;
    Some(1)
}

#[inline]
pub(crate) fn try_run_tgm_build_cache_simple_member(
    module: &Module,
    procedure: ProcedureId,
    program: &Program,
    frame: &mut CallFrame,
    remaining_steps: u64,
    state: &mut ExecutionState,
) -> Option<u64> {
    // Charge one guarded native engine operation. The rich MAPLOADING tick
    // already ran at PCs75-97 and retains its exact scheduler/yield behavior.
    const LOGICAL_STEPS: u64 = 32;
    if frame.instruction != 98
        || remaining_steps < LOGICAL_STEPS
        || !trusted_tgm_build_cache_target(module, procedure, program)
    {
        return None;
    }
    let members = run_tgm_build_cache_simple_member(frame, state)?;
    NATIVE_TGM_BUILD_CACHE_MEMBERS.fetch_add(members as u64, Ordering::Relaxed);
    NATIVE_TGM_BUILD_CACHE_LOGICAL_STEPS.fetch_add(LOGICAL_STEPS, Ordering::Relaxed);
    Some(LOGICAL_STEPS)
}

fn tgm_number(value: &Value) -> Option<i32> {
    value.as_number().map(|value| value as i32)
}

pub(crate) fn try_run_build_coordinate_prefix(
    module: &Module,
    procedure: ProcedureId,
    program: &Program,
    frame: &mut CallFrame,
    state: &mut ExecutionState,
) -> bool {
    if frame.instruction != 0
        || !frame.stack.is_empty()
        || program.parameter_count != 5
        || program.local_count != 31
        || program.instructions.len() != 405
        || !module.procedure_path(procedure).is_some_and(|path| {
            path.split('@').next() == Some("/datum/parsed_map/proc/build_coordinate")
        })
        || module.procedure_semantic_digest(procedure)
            != Some(CANONICAL_MONKE_BUILD_COORDINATE_DIGEST)
    {
        return false;
    }
    let fallback = || {
        NATIVE_BUILD_COORDINATE_PREFIX_FALLBACKS.fetch_add(1, Ordering::Relaxed);
        false
    };
    let Value::Datum(src) = frame.src else {
        return fallback();
    };
    if !state.heap.datum(src).is_ok_and(|datum| {
        let path = datum.type_path().as_str();
        path == "/datum/parsed_map" || path.starts_with("/datum/parsed_map/")
    }) {
        return fallback();
    }
    let Some(Value::Datum(turf)) = frame.locals.get(1).cloned() else {
        return fallback();
    };
    if !state
        .heap
        .datum(turf)
        .is_ok_and(|datum| is_turf_type_path(datum.type_path()))
        || !runtime_truthy(&state.heap, frame.locals.get(4).unwrap_or(&Value::Null))
            .is_ok_and(|value| value)
    {
        return fallback();
    }
    let Value::List(model) = frame.locals.first().cloned().unwrap_or(Value::Null) else {
        return fallback();
    };
    let Ok(model) = state.heap.list(model) else {
        return fallback();
    };
    let (Ok(Value::List(members)), Ok(Value::List(attributes))) = (model.get(1), model.get(2))
    else {
        return fallback();
    };
    let (members, attributes) = (*members, *attributes);
    let Ok(members_list) = state.heap.list(members) else {
        return fallback();
    };
    let len = members_list.len();
    if len < 2 || state.heap.list(attributes).is_err() {
        return fallback();
    }
    let Ok(Value::TypePath(area_path)) = members_list.get(len).cloned() else {
        return fallback();
    };
    if area_path.as_str() == "/area/template_noop" || !is_area_type_path(&area_path) {
        return fallback();
    }
    let default_name = FieldName::parse("__dm_static_2f646174756d2f636f6e74726f6c6c65722f676c6f62616c5f766172732f7661722f6d61705f6d6f64656c5f64656661756c74").expect("canonical map default global");
    let Some(Value::List(default_list)) = state.global(&default_name).cloned() else {
        return fallback();
    };
    if state.heap.list(default_list).is_err()
        || state
            .heap
            .list(attributes)
            .ok()
            .and_then(|list| list.get(len).ok())
            != Some(&Value::List(default_list))
    {
        return fallback();
    }
    let preloader_name = FieldName::parse("__dm_static_2f646174756d2f636f6e74726f6c6c65722f676c6f62616c5f766172732f7661722f7573655f7072656c6f61646572").expect("canonical preloader global");
    if state
        .global(&preloader_name)
        .is_none_or(|value| runtime_truthy(&state.heap, value).unwrap_or(true))
    {
        return fallback();
    }
    let blacklist = FieldName::parse("turf_blacklist").expect("canonical blacklist field");
    match datum_field_or_initial(state, src, &blacklist).ok() {
        None | Some(Value::Null) => {}
        Some(Value::List(list)) => {
            let Ok(value) = read_list_value(
                &state.heap,
                list,
                &Value::Datum(turf),
                state.is_associative_list(list),
            ) else {
                return fallback();
            };
            if runtime_truthy(&state.heap, &value).unwrap_or(true) {
                return fallback();
            }
        }
        _ => return fallback(),
    }
    let loaded = FieldName::parse("loaded_areas").expect("canonical loaded areas field");
    let Ok(Value::List(loaded)) = datum_field_or_initial(state, src, &loaded) else {
        return fallback();
    };
    let Ok(Value::Datum(area)) = read_list_value(
        &state.heap,
        loaded,
        &Value::TypePath(area_path),
        state.is_associative_list(loaded),
    ) else {
        return fallback();
    };
    if !state
        .heap
        .datum(area)
        .is_ok_and(|datum| is_area_type_path(datum.type_path()))
    {
        return fallback();
    }

    // Every fallible shape check precedes the first mutation. This is the
    // engine behavior behind canonical area.contents.Add(crds).
    if builtins::move_turf_to_area(state, turf, area).is_err() {
        return fallback();
    }
    frame.locals[6] = Value::number((len - 1) as f32);
    frame.locals[7] = Value::List(members);
    frame.locals[8] = Value::List(attributes);
    frame.locals[9] = Value::List(default_list);
    frame.locals[10] = Value::Null;
    frame.locals[11] = Value::Datum(area);
    frame.locals[12] = Value::Null;
    frame.locals[27] = Value::Null;
    frame.instruction = 235;
    NATIVE_BUILD_COORDINATE_PREFIX_ACTIVATIONS.fetch_add(1, Ordering::Relaxed);
    true
}

static NATIVE_TGM_CONTINUATION_REJECTIONS: std::sync::OnceLock<std::sync::Mutex<Vec<String>>> =
    std::sync::OnceLock::new();
static NATIVE_TGM_CONTINUATION_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static NATIVE_TGM_ROUTE_SAMPLES: std::sync::OnceLock<std::sync::Mutex<Vec<String>>> =
    std::sync::OnceLock::new();

/// Returns bounded diagnostics for canonical `_tgm_load` frames that safely
/// fell back at the native continuation attachment seam.
#[must_use]
pub fn native_tgm_continuation_rejections() -> Vec<String> {
    NATIVE_TGM_CONTINUATION_REJECTIONS
        .get()
        .and_then(|samples| samples.lock().ok().map(|samples| samples.clone()))
        .unwrap_or_default()
}

/// Returns bounded canonical `_dmm_load`/`_tgm_load` route diagnostics.
#[must_use]
pub fn native_tgm_route_samples() -> Vec<String> {
    NATIVE_TGM_ROUTE_SAMPLES
        .get()
        .and_then(|samples| samples.lock().ok().map(|samples| samples.clone()))
        .unwrap_or_default()
}

fn canonical_tgm_route_kind(module: &Module, procedure: ProcedureId) -> Option<bool> {
    match module.procedure_path(procedure)?.split('@').next()? {
        "/datum/parsed_map/proc/_tgm_load" => Some(true),
        "/datum/parsed_map/proc/_dmm_load" => Some(false),
        _ => None,
    }
}

pub(crate) fn trace_tgm_route(
    module: &Module,
    procedure: ProcedureId,
    program: &Program,
    frame: &mut CallFrame,
    state: &ExecutionState,
) {
    let Some(is_tgm) = canonical_tgm_route_kind(module, procedure) else {
        return;
    };
    let path = if is_tgm {
        "/datum/parsed_map/proc/_tgm_load"
    } else {
        "/datum/parsed_map/proc/_dmm_load"
    };
    let milestone = if frame.instruction == 0 {
        1
    } else if is_tgm && matches!(frame.instruction, 274 | 279) {
        2
    } else if program
        .instructions
        .get(frame.instruction)
        .is_some_and(|instruction| matches!(instruction, Instruction::Return))
    {
        4
    } else {
        return;
    };
    if frame
        .cold()
        .is_some_and(|cold| cold.tgm_route_trace_mask & milestone != 0)
    {
        return;
    }
    frame.cold_mut().tgm_route_trace_mask |= milestone;
    let list_len = |value: Option<&Value>| match value {
        Some(Value::List(list)) => state.heap.list(*list).ok().map(dm_value::DmList::len),
        _ => None,
    };
    let src_field = |name: &str| match frame.src {
        Value::Datum(src) => state
            .heap
            .datum_field(src, &FieldName::parse(name).ok()?)
            .ok()
            .cloned(),
        _ => None,
    };
    let sample = format!(
        "path={path} procedure={} pc={} milestone={} src={:?} args={:?} map_format={:?} src_gridSets_len={:?} local38_len={:?} local14_len={:?} local15={:?} result={:?}",
        procedure.index(),
        frame.instruction,
        match milestone {
            1 => "entry",
            2 => "pre-loop",
            _ => "return",
        },
        frame.src,
        frame
            .locals
            .iter()
            .take(program.parameter_count)
            .collect::<Vec<_>>(),
        src_field("map_format"),
        list_len(src_field("gridSets").as_ref()),
        list_len(frame.locals.get(38)),
        list_len(frame.locals.get(14)),
        frame.locals.get(15),
        frame.result,
    );
    let samples = NATIVE_TGM_ROUTE_SAMPLES.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    if let Ok(mut samples) = samples.lock()
        && samples.len() < 64
    {
        samples.push(sample);
    }
}

fn record_tgm_continuation_rejection(frame: &CallFrame, state: &ExecutionState) {
    let attempt = NATIVE_TGM_CONTINUATION_ATTEMPTS.fetch_add(1, Ordering::Relaxed) + 1;
    let kind = |value: Option<&Value>| match value {
        Some(Value::Null) => "null",
        Some(Value::Number(_)) => "number",
        Some(Value::Text(_)) => "text",
        Some(Value::File(_)) => "file",
        Some(Value::TypePath(_)) => "typepath",
        Some(Value::ModifiedTypePath(_)) => "modified-typepath",
        Some(Value::Datum(_)) => "datum",
        Some(Value::List(_)) => "list",
        None => "missing",
    };
    let reason = match (
        frame.locals.get(38),
        frame.locals.get(14),
        frame.locals.get(15),
    ) {
        (Some(Value::List(_)), Some(Value::List(_)), Some(Value::Text(_) | Value::Null)) => {
            let grid_count = frame
                .locals
                .get(38)
                .and_then(|value| match value {
                    Value::List(list) => state.heap.list(*list).ok(),
                    _ => None,
                })
                .map_or(0, dm_value::DmList::len);
            let (model_positions, model_associations) = frame
                .locals
                .get(14)
                .and_then(|value| match value {
                    Value::List(list) => state.heap.list(*list).ok(),
                    _ => None,
                })
                .map_or((0, 0), |list| {
                    (list.positions().count(), list.associations().count())
                });
            let grid_list = match frame.locals.get(38) {
                Some(Value::List(list)) => state.heap.list(*list).ok(),
                _ => None,
            };
            let mut detail = None;
            if let Some(grids) = grid_list {
                let fields = ["xcrd", "ycrd", "zcrd", "gridLines"]
                    .map(|name| FieldName::parse(name).unwrap());
                for (index, value) in grids.positions() {
                    let Value::Datum(grid) = value else {
                        detail = Some(format!("grid[{index}]-kind={}", kind(Some(value))));
                        break;
                    };
                    for field in &fields {
                        let value = state.heap.datum_field(*grid, field).ok();
                        let valid = if field.as_str() == "gridLines" {
                            matches!(value, Some(Value::List(_)))
                        } else {
                            value.and_then(Value::as_number).is_some()
                        };
                        if !valid {
                            detail = Some(format!(
                                "grid[{index}].{}-kind={}",
                                field.as_str(),
                                kind(value)
                            ));
                            break;
                        }
                    }
                    if detail.is_some() {
                        break;
                    }
                    if let Ok(Value::List(lines)) = state.heap.datum_field(*grid, &fields[3]) {
                        if let Ok(lines) = state.heap.list(*lines) {
                            if let Some((line, value)) = lines
                                .positions()
                                .find(|(_, value)| !matches!(value, Value::Text(_)))
                            {
                                detail = Some(format!(
                                    "grid[{index}].gridLines[{line}]-kind={}",
                                    kind(Some(value))
                                ));
                                break;
                            }
                        } else {
                            detail = Some(format!("grid[{index}].gridLines-stale"));
                            break;
                        }
                    }
                }
            } else {
                detail = Some("grid-list-stale".to_owned());
            }
            if detail.is_none() {
                for slot in [0_usize, 1, 2, 5, 6, 7, 8, 39] {
                    if frame.locals.get(slot).and_then(Value::as_number).is_none() {
                        detail = Some(format!(
                            "numeric-local[{slot}]-kind={}",
                            kind(frame.locals.get(slot))
                        ));
                        break;
                    }
                }
            }
            format!(
                "{} grids={grid_count} model_positions={model_positions} model_associations={model_associations} space_key_kind={} detail={}",
                detail.as_deref().unwrap_or("late-shape-or-missing-model"),
                kind(frame.locals.get(15)),
                detail.as_deref().unwrap_or("none")
            )
        }
        (Some(Value::List(_)), Some(Value::List(_)), _) => {
            format!("space-key-shape value={:?}", frame.locals.get(15))
        }
        (Some(Value::List(_)), _, _) => {
            format!("model-cache-shape value={:?}", frame.locals.get(14))
        }
        _ => format!("grid-sets-shape value={:?}", frame.locals.get(38)),
    };
    let samples =
        NATIVE_TGM_CONTINUATION_REJECTIONS.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    if let Ok(mut samples) = samples.lock()
        && samples.len() < 32
    {
        samples.push(format!("attempt={attempt} {reason}"));
    }
}

pub(crate) fn build_tgm_load_continuation(
    frame: &CallFrame,
    state: &ExecutionState,
) -> Option<TgmLoadContinuation> {
    // Slot 13 is the compiler-owned return value (`.`). The canonical DM
    // locals declared by `_tgm_load` therefore begin at 14; keep these slot
    // numbers aligned with the executable dump rather than the source-local
    // ordinal.
    let Value::List(grid_sets) = frame.locals.get(38)? else {
        return None;
    };
    let Value::List(model_cache) = frame.locals.get(14)? else {
        return None;
    };
    let space_key = match frame.locals.get(15)? {
        Value::Text(value) => Some(Arc::clone(value)),
        Value::Null => None,
        _ => return None,
    };
    let mut models = BTreeMap::new();
    let mut model_keys = BTreeSet::new();
    for (key, value) in state.heap.list(*model_cache).ok()?.associations() {
        let Value::Text(key) = key else { continue };
        model_keys.insert(Arc::clone(key));
        models.insert(Arc::clone(key), value.clone());
    }
    let xcrd = FieldName::parse("xcrd").ok()?;
    let ycrd = FieldName::parse("ycrd").ok()?;
    let zcrd = FieldName::parse("zcrd").ok()?;
    let grid_lines = FieldName::parse("gridLines").ok()?;
    let mut grids = Vec::new();
    for (_, value) in state.heap.list(*grid_sets).ok()?.positions() {
        let Value::Datum(grid) = value else {
            return None;
        };
        let Value::List(lines) = state.heap.datum_field(*grid, &grid_lines).ok()? else {
            return None;
        };
        let lines = state
            .heap
            .list(*lines)
            .ok()?
            .positions()
            .map(|(_, value)| match value {
                Value::Text(value) => Some(Arc::clone(value)),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        grids.push(tgm_planner::GridSet {
            x: tgm_number(state.heap.datum_field(*grid, &xcrd).ok()?)?,
            y: tgm_number(state.heap.datum_field(*grid, &ycrd).ok()?)?,
            z: tgm_number(state.heap.datum_field(*grid, &zcrd).ok()?)?,
            lines: lines.into(),
        });
    }
    let finite_bound = |value: &Value| {
        value
            .as_number()
            // tg/Monke defines INFINITY as the finite sentinel 1e31 rather
            // than IEEE infinity. A direct Rust float-to-int cast saturates
            // those defaults to i32::{MIN,MAX}, incorrectly turning an
            // unbounded load into an enormous Z translation.
            .filter(|value| {
                value.is_finite() && *value > i32::MIN as f32 && *value < i32::MAX as f32
            })
            .map(|value| value as i32)
    };
    let config = tgm_planner::Config {
        x_offset: tgm_number(frame.locals.first()?)?,
        y_offset: tgm_number(frame.locals.get(1)?)?,
        z_offset: tgm_number(frame.locals.get(2)?)?,
        crop_map: runtime_truthy(&state.heap, frame.locals.get(3)?).ok()?,
        no_changeturf: runtime_truthy(&state.heap, frame.locals.get(4)?).ok()?,
        x_lower: tgm_number(frame.locals.get(5)?)?,
        x_upper: tgm_number(frame.locals.get(6)?)?,
        y_lower: tgm_number(frame.locals.get(7)?)?,
        y_upper: tgm_number(frame.locals.get(8)?)?,
        z_lower: finite_bound(frame.locals.get(9)?),
        z_upper: finite_bound(frame.locals.get(10)?),
        world_max_x: world_numeric_field(state, "maxx")? as i32,
        world_max_y: world_numeric_field(state, "maxy")? as i32,
        // Local 39 is the pre-expansion z threshold captured by the canonical
        // setup; using world.maxz here would incorrectly enable AfterChange on
        // levels that `_tgm_load` created immediately before this loop.
        world_max_z: tgm_number(frame.locals.get(39)?)?,
        space_key,
        model_keys: Arc::new(model_keys),
    };
    let plan = tgm_planner::prepare(&grids, &config);
    let original_path = match frame.src {
        Value::Datum(src) => state
            .heap
            .datum_field(src, &FieldName::parse("original_path").ok()?)
            .ok()
            .cloned(),
        _ => None,
    };
    let grid_summary = |grid: Option<&tgm_planner::GridSet>| {
        grid.map(|grid| (grid.x, grid.y, grid.z, grid.lines.len()))
    };
    let samples = NATIVE_TGM_ROUTE_SAMPLES.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    if let Ok(mut samples) = samples.lock()
        && samples.len() < 64
    {
        samples.push(format!(
            "planned-sidecar original_path={original_path:?} grids={} first={:?} last={:?} offsets=({},{},{}) crop={} no_changeturf={} x_bounds=({}, {}) y_bounds=({}, {}) z_bounds={:?}..{:?} world=({},{},{}) space_key={:?} model_keys={} events={} cells={} safepoints={} missing={} bounds={:?}",
            grids.len(),
            grid_summary(grids.first()),
            grid_summary(grids.last()),
            config.x_offset,
            config.y_offset,
            config.z_offset,
            config.crop_map,
            config.no_changeturf,
            config.x_lower,
            config.x_upper,
            config.y_lower,
            config.y_upper,
            config.z_lower,
            config.z_upper,
            config.world_max_x,
            config.world_max_y,
            config.world_max_z,
            config.space_key,
            config.model_keys.len(),
            plan.events.len(),
            plan.cells.len(),
            plan.events.len().saturating_sub(plan.cells.len() + plan.missing_models.len()),
            plan.missing_models.len(),
            plan.bounds,
        ));
    }
    // The rich failure branch first calls map_loader_stop and then CRASHes.
    // Until that callback is represented as an ordered native event, retain
    // the complete bytecode path whenever validation found a missing model.
    if !plan.missing_models.is_empty() {
        let missing = &plan.missing_models[0];
        let samples =
            NATIVE_TGM_CONTINUATION_REJECTIONS.get_or_init(|| std::sync::Mutex::new(Vec::new()));
        if let Ok(mut samples) = samples.lock()
            && samples.len() < 32
        {
            samples.push(format!(
                "missing-model key={:?} coordinate=({},{},{}) missing_count={} model_count={}",
                missing.model_key,
                missing.x,
                missing.y,
                missing.z,
                plan.missing_models.len(),
                models.len()
            ));
        }
        return None;
    }
    Some(TgmLoadContinuation {
        plan: Arc::new(plan),
        cursor: tgm_planner::CommitCursor::default(),
        phase: TgmLoadPhase::Commit,
        model_cache: Value::List(*model_cache),
        models,
        bounds: frame.locals.get(16)?.clone(),
        coordinate_target: None,
    })
}

pub(crate) enum TgmDrive {
    None,
    Continue,
    Push(CallFrame),
    Error(String),
}

fn advance_ruin_scan_coordinate(scan: &mut RuinCandidateScan) {
    if scan.next.0 < scan.high.0 {
        scan.next.0 += 1;
    } else if scan.next.1 < scan.high.1 {
        scan.next.0 = scan.low.0;
        scan.next.1 += 1;
    } else if scan.next.2 < scan.high.2 {
        scan.next.0 = scan.low.0;
        scan.next.1 = scan.low.1;
        scan.next.2 += 1;
    } else {
        scan.empty = true;
    }
}

pub(crate) fn ruin_scan_attach_at_call(frame: &CallFrame) -> Option<bool> {
    if frame.instruction == 63 && frame.stack.is_empty() {
        return Some(false);
    }
    (frame.instruction == 65
        && frame.stack.len() == 2
        && frame.stack.first() == frame.locals.get(8)
        && frame
            .stack
            .get(1)
            .is_some_and(|value| value.as_number() == Some(1.0)))
    .then_some(true)
}

pub(crate) fn revalidated_ruin_rejection(
    state: &mut ExecutionState,
    bounds: (i32, i32, i32, i32, i32, i32),
    turf_flags: &FieldName,
) -> bool {
    let (low_x, low_y, z, high_x, high_y, _) = bounds;
    let Some(by_coordinate) = state.ruin_rejection_witnesses.get(&z) else {
        return false;
    };
    let candidates = by_coordinate
        .range((low_y, i32::MIN)..=(high_y, i32::MAX))
        .filter(|((_, x), _)| (low_x..=high_x).contains(x))
        .map(|(&(y, x), &turf)| ((x, y, z), turf))
        .collect::<Vec<_>>();
    for (coordinate, witness) in candidates {
        let still_rejects = state.turf_at(coordinate.0, coordinate.1, coordinate.2)
            == Some(witness)
            && datum_field_or_initial(state, witness, turf_flags)
                .ok()
                .and_then(|value| value.as_number())
                .is_some_and(|flags| flags as i32 & (1 << 4) != 0);
        if still_rejects {
            return true;
        }
        if let Some(by_coordinate) = state.ruin_rejection_witnesses.get_mut(&z) {
            by_coordinate.remove(&(coordinate.1, coordinate.0));
        }
    }
    false
}

pub(crate) fn drive_ruin_candidate_scan(
    module: &Module,
    procedure: ProcedureId,
    program: &Program,
    frame: &mut CallFrame,
    state: &mut ExecutionState,
    remaining_steps: u64,
) -> TgmDrive {
    if remaining_steps == 0 {
        return TgmDrive::None;
    }
    if frame
        .cold()
        .and_then(|cold| cold.ruin_scan.as_ref())
        .is_none()
    {
        // Compact numeric dispatch can execute the two argument-producing
        // instructions at 63-64 before side-exiting on the call at 65.
        // Accept that equivalent, fully verified entry state as well.
        let Some(attach_at_call) = ruin_scan_attach_at_call(frame) else {
            return TgmDrive::None;
        };
        if !trusted_ruin_try_to_place_target(module, procedure, program)
            || !trusted_get_affected_turfs_target(module)
        {
            return TgmDrive::None;
        }
        if attach_at_call {
            frame.stack.clear();
        }
        let Value::Datum(center) = frame.locals.get(8).cloned().unwrap_or(Value::Null) else {
            return TgmDrive::None;
        };
        let coordinate = |name: &str| {
            datum_field_or_initial(state, center, &FieldName::parse(name).ok()?)
                .ok()?
                .as_number()
                .map(|value| value as i32)
        };
        let dimension = |name: &str| {
            let Value::Datum(src) = frame.src else {
                return None;
            };
            datum_field_or_initial(state, src, &FieldName::parse(name).ok()?)
                .ok()?
                .as_number()
                .map(|value| value.round() as i32)
        };
        let center_coordinate = (coordinate("x"), coordinate("y"), coordinate("z"));
        let (Some(center_x), Some(center_y), Some(center_z)) = center_coordinate else {
            return TgmDrive::None;
        };
        let (Some(width), Some(height)) = (dimension("width"), dimension("height")) else {
            return TgmDrive::None;
        };
        let requested_low = (
            center_x - (width as f32 / 2.0).round() as i32,
            center_y - (height as f32 / 2.0).round() as i32,
            center_z,
        );
        let low = state
            .turf_at(requested_low.0, requested_low.1, requested_low.2)
            .map_or((center_x, center_y, center_z), |_| requested_low);
        let high = (low.0 + width - 1, low.1 + height - 1, low.2);
        let empty = state.turf_at(high.0, high.1, high.2).is_none();
        let bounds = (low.0, low.1, low.2, high.0, high.1, high.2);
        if revalidated_ruin_rejection(state, bounds, &FieldName::parse("turf_flags").unwrap()) {
            frame.locals[9] = Value::number(0.0);
            frame.instruction = 14;
            NATIVE_RUIN_SCAN_ACTIVATIONS.fetch_add(1, Ordering::Relaxed);
            NATIVE_RUIN_SCAN_REJECTIONS.fetch_add(1, Ordering::Relaxed);
            NATIVE_RUIN_FLAG_REJECTIONS.fetch_add(1, Ordering::Relaxed);
            NATIVE_RUIN_REJECTION_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
            return TgmDrive::Continue;
        }
        frame.cold_mut().ruin_scan = Some(RuinCandidateScan {
            low,
            next: low,
            high,
            empty,
            turfs: Vec::new(),
            areas: Vec::new(),
            validating: false,
            validate_index: 0,
        });
        NATIVE_RUIN_SCAN_ACTIVATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let turf_flags = FieldName::parse("turf_flags").unwrap();
    let loc = FieldName::parse("loc").unwrap();
    for _ in 0..256 {
        let (validating, empty, next) = {
            let scan = frame.cold().unwrap().ruin_scan.as_ref().unwrap();
            (scan.validating, scan.empty, scan.next)
        };
        if !validating && !empty {
            if let Some(turf) = state.turf_at(next.0, next.1, next.2) {
                NATIVE_RUIN_SCAN_CELLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let flags = datum_field_or_initial(state, turf, &turf_flags)
                    .ok()
                    .and_then(|value| value.as_number())
                    .unwrap_or(0.0) as i32;
                if flags & (1 << 4) != 0 {
                    let witness_count: usize = state
                        .ruin_rejection_witnesses
                        .values()
                        .map(BTreeMap::len)
                        .sum();
                    if witness_count >= 131_072 {
                        state.ruin_rejection_witnesses.clear();
                    }
                    state
                        .ruin_rejection_witnesses
                        .entry(next.2)
                        .or_default()
                        .insert((next.1, next.0), turf);
                    frame.cold_mut().ruin_scan = None;
                    frame.locals[9] = Value::number(0.0);
                    frame.instruction = 14;
                    NATIVE_RUIN_SCAN_REJECTIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    NATIVE_RUIN_FLAG_REJECTIONS.fetch_add(1, Ordering::Relaxed);
                    return TgmDrive::Continue;
                }
                let area = datum_field_or_initial(state, turf, &loc)
                    .ok()
                    .unwrap_or(Value::Null);
                let scan = frame.cold_mut().ruin_scan.as_mut().unwrap();
                scan.turfs.push(turf);
                if !scan
                    .areas
                    .iter()
                    .any(|existing| values_equal(&state.heap, existing, &area))
                {
                    scan.areas.push(area);
                }
            }
            let scan = frame.cold_mut().ruin_scan.as_mut().unwrap();
            advance_ruin_scan_coordinate(scan);
            continue;
        }
        if !validating {
            frame.cold_mut().ruin_scan.as_mut().unwrap().validating = true;
            continue;
        }
        let (index, area) = {
            let scan = frame.cold().unwrap().ruin_scan.as_ref().unwrap();
            (
                scan.validate_index,
                scan.areas.get(scan.validate_index).cloned(),
            )
        };
        let Some(area) = area else {
            let scan = frame.cold_mut().ruin_scan.take().unwrap();
            let affected_turfs = state.heap.allocate_list();
            let affected_areas = state.heap.allocate_list();
            state
                .heap
                .list_mut(affected_turfs)
                .ok()
                .map(|list| list.extend_positional(scan.turfs.into_iter().map(Value::Datum)));
            for area in scan.areas {
                let associative = state.is_associative_list(affected_areas);
                if write_list_value(
                    &mut state.heap,
                    affected_areas,
                    area,
                    Value::number(1.0),
                    associative,
                )
                .is_err()
                {
                    return TgmDrive::Error("ruin affected-area materialization failed".to_owned());
                }
            }
            frame.locals[9] = Value::number(1.0);
            frame.locals[10] = Value::List(affected_turfs);
            frame.locals[11] = Value::List(affected_areas);
            frame.instruction = 145;
            NATIVE_RUIN_SCAN_SUCCESSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return TgmDrive::Continue;
        };
        let allowed = if let Some(Value::List(list)) = frame.locals.get(1).cloned() {
            list
        } else {
            frame.cold_mut().ruin_scan = None;
            frame.instruction = 63;
            return TgmDrive::None;
        };
        let area_type = match area {
            Value::Datum(area) => state.heap.datum(area).ok().map_or(Value::Null, |datum| {
                Value::TypePath(datum.type_path().clone())
            }),
            _ => Value::Null,
        };
        let allowed_value = read_list_value(
            &state.heap,
            allowed,
            &area_type,
            state.is_associative_list(allowed),
        )
        .unwrap_or(Value::Null);
        if !runtime_truthy(&state.heap, &allowed_value).unwrap_or(false) {
            NATIVE_RUIN_AREA_REJECTIONS.fetch_add(1, Ordering::Relaxed);
            let samples = NATIVE_RUIN_AREA_REJECTION_SAMPLES
                .get_or_init(|| std::sync::Mutex::new(Vec::new()));
            if let Ok(mut samples) = samples.lock()
                && samples.len() < 16
            {
                let z = frame
                    .cold()
                    .and_then(|cold| cold.ruin_scan.as_ref())
                    .map_or(0, |scan| scan.low.2);
                let allowed_entries = state.heap.list(allowed).ok().map_or_else(
                    || "<stale>".to_owned(),
                    |list| {
                        list.associations()
                            .take(8)
                            .map(|(key, value)| format!("{key:?}={value:?}"))
                            .collect::<Vec<_>>()
                            .join(",")
                    },
                );
                samples.push(format!(
                    "z={z} actual={area_type:?} lookup={allowed_value:?} allowed=[{allowed_entries}]"
                ));
            }
            frame.cold_mut().ruin_scan = None;
            frame.locals[9] = Value::number(0.0);
            frame.instruction = 14;
            NATIVE_RUIN_SCAN_REJECTIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return TgmDrive::Continue;
        }
        frame.cold_mut().ruin_scan.as_mut().unwrap().validate_index = index + 1;
    }
    TgmDrive::Continue
}

pub(crate) fn drive_tgm_load(
    module: &Module,
    procedure: ProcedureId,
    program: &Program,
    frame: &mut CallFrame,
    state: &mut ExecutionState,
    remaining_steps: u64,
) -> TgmDrive {
    if remaining_steps == 0 {
        return TgmDrive::None;
    }
    if frame
        .cold()
        .and_then(|cold| cold.tgm_load.as_ref())
        .is_none()
    {
        let Some(attach_before_iterator) = tgm_attach_location(frame) else {
            return TgmDrive::None;
        };
        if !trusted_tgm_load_target(module, procedure, program) {
            if !canonical_tgm_load_path(module, procedure) {
                return TgmDrive::None;
            }
            let samples = NATIVE_TGM_CONTINUATION_REJECTIONS
                .get_or_init(|| std::sync::Mutex::new(Vec::new()));
            if let Ok(mut samples) = samples.lock()
                && samples.len() < 32
            {
                samples.push(format!(
                    "guard-mismatch procedure={} path={:?} params={} locals={} instructions={} digest={:?}",
                    procedure.index(),
                    module.procedure_path(procedure),
                    program.parameter_count,
                    program.local_count,
                    program.instructions.len(),
                    module.procedure_semantic_digest(procedure)
                ));
            }
            return TgmDrive::None;
        }
        let Some(sidecar) = build_tgm_load_continuation(frame, state) else {
            record_tgm_continuation_rejection(frame, state);
            return TgmDrive::None;
        };
        let (cells, safepoints) = sidecar.plan.events.iter().fold(
            (0_u64, 0_u64),
            |(cells, safepoints), event| match event {
                tgm_planner::CommitEvent::Cell(_) => (cells + 1, safepoints),
                tgm_planner::CommitEvent::SafepointOnly(_) => (cells, safepoints + 1),
                tgm_planner::CommitEvent::MissingModel(_) => (cells, safepoints),
            },
        );
        NATIVE_TGM_PLANNED_CELLS.fetch_add(cells, Ordering::Relaxed);
        NATIVE_TGM_PLANNED_SAFEPOINTS.fetch_add(safepoints, Ordering::Relaxed);
        frame.cold_mut().tgm_load = Some(sidecar);
        if attach_before_iterator {
            // PCs 274-278 only initialize rich iterator locals. The native
            // plan owns the same grid snapshot and never consumes those slots.
            frame.instruction = 279;
        }
        NATIVE_TGM_LOAD_ACTIVATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let phase = frame
        .cold()
        .and_then(|cold| cold.tgm_load.as_ref())
        .map(|sidecar| sidecar.phase.clone())
        .expect("TGM sidecar exists");
    match phase {
        TgmLoadPhase::AwaitCoordinate if frame.instruction == 280 => {
            let _ = frame.stack.pop();
            let committed_cell = frame
                .cold()
                .and_then(|cold| cold.tgm_load.as_ref())
                .and_then(|sidecar| sidecar.cursor.peek(&sidecar.plan))
                .and_then(|event| match event {
                    tgm_planner::CommitEvent::Cell(cell) => Some(cell.clone()),
                    _ => None,
                });
            if let Some(cell) = committed_cell {
                NATIVE_TGM_COMMITTED_CELLS.fetch_add(1, Ordering::Relaxed);
                let samples =
                    NATIVE_TGM_COMMIT_SAMPLES.get_or_init(|| std::sync::Mutex::new(Vec::new()));
                if let Ok(mut samples) = samples.lock()
                    && samples.len() < 16
                {
                    let turf = state.turf_at(cell.x, cell.y, cell.z);
                    let (turf_type, area_type) = turf.map_or_else(
                        || ("<missing>".to_owned(), "<missing>".to_owned()),
                        |turf| {
                            let turf_type = state
                                .heap
                                .datum(turf)
                                .map_or("<stale>".to_owned(), |datum| {
                                    datum.type_path().to_string()
                                });
                            let area_type = state
                                .world_areas
                                .get(&(cell.x, cell.y, cell.z))
                                .and_then(|area| state.heap.datum(*area).ok())
                                .map_or("<missing>".to_owned(), |area| {
                                    area.type_path().to_string()
                                });
                            (turf_type, area_type)
                        },
                    );
                    samples.push(format!(
                        "coord=({},{},{}) model={} turf={} area={}",
                        cell.x, cell.y, cell.z, cell.model_key, turf_type, area_type
                    ));
                }
            }
            let sidecar = frame.cold_mut().tgm_load.as_mut().unwrap();
            sidecar.phase = TgmLoadPhase::Tick;
            frame.instruction = 423;
            TgmDrive::Continue
        }
        // PC446 is the first instruction after MAPLOADING_CHECK_TICK. Do not
        // execute the rich loop's PC446-450 index increment/jump or it would
        // re-enter PC334 and commit the same grid cells a second time.
        TgmLoadPhase::Tick if frame.instruction == 446 => {
            let sidecar = frame.cold_mut().tgm_load.as_mut().unwrap();
            sidecar.cursor.acknowledge(&sidecar.plan);
            sidecar.phase = TgmLoadPhase::Commit;
            frame.instruction = 279;
            TgmDrive::Continue
        }
        TgmLoadPhase::Commit if frame.instruction == 279 => {
            let event = {
                let sidecar = frame.cold().unwrap().tgm_load.as_ref().unwrap();
                sidecar.cursor.peek(&sidecar.plan).cloned()
            };
            match event {
                Some(tgm_planner::CommitEvent::Cell(cell)) => {
                    let model = frame
                        .cold()
                        .unwrap()
                        .tgm_load
                        .as_ref()
                        .unwrap()
                        .models
                        .get(&cell.model_key)
                        .cloned()
                        .expect("planner validated model key");
                    let coordinate = state
                        .turf_at(cell.x, cell.y, cell.z)
                        .map_or(Value::Null, Value::Datum);
                    let context = frame_context(frame);
                    let receiver_type = match frame.src {
                        Value::Datum(src) => state
                            .heap
                            .datum(src)
                            .ok()
                            .map(|datum| datum.type_path().clone()),
                        _ => None,
                    };
                    let cached_target = frame
                        .cold()
                        .and_then(|cold| cold.tgm_load.as_ref())
                        .and_then(|sidecar| sidecar.coordinate_target.as_ref())
                        .filter(|(cached_type, _)| Some(cached_type) == receiver_type.as_ref())
                        .map(|(_, target)| *target);
                    let (target, context) = if let Some(target) = cached_target {
                        NATIVE_TGM_TARGET_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
                        (
                            target,
                            ExecutionContext::new(frame.src.clone(), context.usr.clone()),
                        )
                    } else {
                        NATIVE_TGM_TARGET_RESOLUTIONS.fetch_add(1, Ordering::Relaxed);
                        let Ok((target, context)) = dynamic_call_target_named(
                            module,
                            state,
                            &frame.src,
                            "build_coordinate",
                            &context,
                            false,
                        ) else {
                            return TgmDrive::Error(
                                "TGM build_coordinate target disappeared".to_owned(),
                            );
                        };
                        if let Some(receiver_type) = receiver_type {
                            frame
                                .cold_mut()
                                .tgm_load
                                .as_mut()
                                .unwrap()
                                .coordinate_target = Some((receiver_type, target));
                        }
                        (target, context)
                    };
                    let Ok(target_program) = module.resolve_procedure(target) else {
                        return TgmDrive::Error("TGM build_coordinate body disappeared".to_owned());
                    };
                    let child = make_frame(
                        target,
                        target_program,
                        &[
                            model,
                            coordinate,
                            Value::number(f32::from(cell.no_afterchange)),
                            frame.locals[11].clone(),
                            frame.locals[12].clone(),
                        ],
                        &context,
                    );
                    frame.cold_mut().tgm_load.as_mut().unwrap().phase =
                        TgmLoadPhase::AwaitCoordinate;
                    TgmDrive::Push(child)
                }
                Some(tgm_planner::CommitEvent::SafepointOnly(_)) => {
                    frame.cold_mut().tgm_load.as_mut().unwrap().phase = TgmLoadPhase::Tick;
                    frame.instruction = 423;
                    TgmDrive::Continue
                }
                Some(tgm_planner::CommitEvent::MissingModel(missing)) => {
                    TgmDrive::Error(format!("Undefined model key in DMM: {}", missing.model_key))
                }
                None => {
                    let sidecar = frame.cold_mut().tgm_load.take().unwrap();
                    if let (Value::List(bounds), Some(measured)) =
                        (sidecar.bounds, sidecar.plan.bounds)
                    {
                        if let Ok(values) = state.heap.list_mut(bounds) {
                            for (index, value) in [
                                measured.min_x,
                                measured.min_y,
                                measured.min_z,
                                measured.max_x,
                                measured.max_y,
                                measured.max_z,
                            ]
                            .into_iter()
                            .enumerate()
                            {
                                let _ = values.set(index + 1, Value::number(value as f32));
                            }
                        }
                    }
                    frame.stack.push(Value::number(1.0));
                    frame.instruction = 506;
                    TgmDrive::Continue
                }
            }
        }
        _ => TgmDrive::None,
    }
}

pub(crate) fn tgm_attach_location(frame: &CallFrame) -> Option<bool> {
    if frame.instruction == 279 {
        return Some(false);
    }
    (frame.instruction == 274 && frame.stack.is_empty()).then_some(true)
}

pub(crate) fn canonical_type2parent_target(
    module: &Module,
    procedure: ProcedureId,
    program: &Program,
) -> bool {
    let Some(path) = module.procedure_path(procedure) else {
        return false;
    };
    if path != "/proc/type2parent"
        && !path
            .strip_prefix("/proc/type2parent@")
            .is_some_and(|suffix| {
                !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
            })
    {
        return false;
    }

    thread_local! {
        static CACHE: std::cell::RefCell<HashMap<(u64, ProcedureId), bool>> =
            std::cell::RefCell::new(HashMap::new());
    }
    let key = (module.identity.0, procedure);
    CACHE.with(|cache| {
        *cache
            .borrow_mut()
            .entry(key)
            .or_insert_with(|| canonical_type2parent_program(program))
    })
}

pub(crate) fn canonical_type2parent(path: &TypePath) -> Option<TypePath> {
    let path = path.as_str();
    match path {
        "/datum" => None,
        "/obj" | "/mob" => TypePath::parse("/atom/movable").ok(),
        "/area" | "/turf" => TypePath::parse("/atom").ok(),
        _ => path.rfind('/').and_then(|slash| {
            if slash == 0 {
                TypePath::parse("/datum").ok()
            } else {
                TypePath::parse(&path[..slash]).ok()
            }
        }),
    }
}

pub(crate) fn canonical_static_native_builtin(
    module: &Module,
    procedure: ProcedureId,
    program: &Program,
) -> Option<&'static str> {
    fn matches_canonical(
        program: &Program,
        source: &str,
        canonical: &'static OnceLock<Program>,
    ) -> bool {
        let canonical = canonical.get_or_init(|| {
            let syntax = dm_syntax::parse(source).expect("canonical native builtin should parse");
            compile_procedure(
                syntax
                    .definitions
                    .first()
                    .expect("canonical native builtin definition exists"),
            )
            .expect("canonical native builtin should compile")
        });
        program.wait_for == canonical.wait_for
            && program.parameter_count == canonical.parameter_count
            && program.parameter_names == canonical.parameter_names
            && program.local_count == canonical.local_count
            && program.instructions == canonical.instructions
    }

    static IS_TEXT: OnceLock<Program> = OnceLock::new();
    static MIN: OnceLock<Program> = OnceLock::new();
    static MAX: OnceLock<Program> = OnceLock::new();
    let path = module.procedure_path(procedure)?;
    let (name, source, canonical) = match path {
        "/proc/istext@dream64_builtin" => (
            "istext",
            "/proc/istext(value)\n\treturn !isnull(value) && !isnum(value) && !ispath(value) && !islist(value) && !istype(value)\n",
            &IS_TEXT,
        ),
        "/proc/min@dream64_builtin" => (
            "min",
            "/proc/min(...)\n\tvar/list/values = args\n\tif(length(args) == 1 && islist(args[1]))\n\t\tvalues = args[1]\n\tif(!length(values))\n\t\treturn null\n\tvar/result = values[1]\n\tfor(var/value in values)\n\t\tif(value < result)\n\t\t\tresult = value\n\treturn result\n",
            &MIN,
        ),
        "/proc/max@dream64_builtin" => (
            "max",
            "/proc/max(...)\n\tvar/list/values = args\n\tif(length(args) == 1 && islist(args[1]))\n\t\tvalues = args[1]\n\tif(!length(values))\n\t\treturn null\n\tvar/result = values[1]\n\tfor(var/value in values)\n\t\tif(value > result)\n\t\t\tresult = value\n\treturn result\n",
            &MAX,
        ),
        _ => return None,
    };
    matches_canonical(program, source, canonical).then_some(name)
}

pub(crate) fn canonical_istext(value: &Value) -> Value {
    Value::number(f32::from(!matches!(
        value,
        Value::Null | Value::Number(_) | Value::TypePath(_) | Value::Datum(_) | Value::List(_)
    )))
}

#[inline(always)]
pub(crate) fn execute_compact_fast_instruction(
    operation: compact_wordcode::CompactFastInstruction,
    frame: &mut CallFrame,
    state: &ExecutionState,
) -> Result<(), String> {
    use crate::compact_wordcode::CompactFastInstruction;

    match operation {
        CompactFastInstruction::PushNull => frame.stack.push(Value::Null),
        CompactFastInstruction::LoadSrc => {
            frame
                .stack
                .push(canonicalize_value(&state.heap, &frame.src));
        }
        CompactFastInstruction::StoreSrc => frame.src = pop(&mut frame.stack)?,
        CompactFastInstruction::LoadUsr => {
            frame
                .stack
                .push(canonicalize_value(&state.heap, &frame.usr));
        }
        CompactFastInstruction::StoreUsr => frame.usr = pop(&mut frame.stack)?,
        CompactFastInstruction::LoadResult => frame.stack.push(frame.result.clone()),
        CompactFastInstruction::StoreResult => frame.result = pop(&mut frame.stack)?,
        CompactFastInstruction::Pop => {
            pop(&mut frame.stack)?;
        }
        CompactFastInstruction::Duplicate => {
            let value = frame
                .stack
                .last()
                .cloned()
                .ok_or_else(|| "bytecode stack underflow".to_owned())?;
            frame.stack.push(value);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]

fn normalized_dmm_cache_path(path: &str) -> Option<String> {
    let path = path.replace('\\', "/");
    if path.is_empty() || path.starts_with('/') || path.contains(':') {
        return None;
    }
    let mut normalized = Vec::new();
    for component in path.split('/') {
        match component {
            "" | "." => {}
            ".." => return None,
            component => normalized.push(component.to_ascii_lowercase()),
        }
    }
    (!normalized.is_empty()).then(|| normalized.join("/"))
}

fn artifact_dmm_source_matches(state: &ExecutionState, path: &str, digest: [u8; 16]) -> bool {
    let Some(root) = state.project_root() else {
        return true;
    };
    let candidate = root.join(path.replace('/', std::path::MAIN_SEPARATOR_STR));
    if !candidate.exists() {
        return true;
    }
    if !candidate.is_file() {
        return false;
    }
    let Some(canonical_root) = std::fs::canonicalize(root).ok() else {
        return false;
    };
    let Some(canonical_candidate) = std::fs::canonicalize(candidate).ok() else {
        return false;
    };
    canonical_candidate.starts_with(canonical_root)
        && std::fs::read(canonical_candidate)
            .ok()
            .is_some_and(|bytes| md5::compute(bytes).0 == digest)
}

const CANONICAL_MONKE_DISCOVER_OFFSET_DIGEST: [u8; 32] = [
    0xe4, 0xa0, 0xe8, 0x26, 0x6a, 0xf4, 0xdd, 0x8e, 0xec, 0x7d, 0x8a, 0x26, 0xc8, 0x68, 0x91, 0x1a,
    0x61, 0xc8, 0xde, 0x87, 0xdd, 0xce, 0x68, 0xf0, 0xaf, 0x30, 0x2d, 0x16, 0xfc, 0xcd, 0x57, 0x6f,
];

fn trusted_discover_offset_target(
    module: &Module,
    procedure: ProcedureId,
    program: &Program,
) -> bool {
    module.procedure_path(procedure).is_some_and(|path| {
        path.split('@').next() == Some("/datum/map_template/proc/discover_offset")
    }) && program.parameter_count == 1
        && program.local_count == 18
        && program.instructions.len() == 131
        && matches!(program.instructions.get(23), Some(Instruction::StandardBuiltin { name, argument_count: 2, .. }) if name == "findtext")
        && matches!(
            program.instructions.get(54),
            Some(Instruction::NextLocalListIteration { .. })
        )
        && matches!(
            program.instructions.get(99),
            Some(Instruction::CopyText {
                argument_count: 3,
                character_indices: false
            })
        )
        && matches!(program.instructions.get(104), Some(Instruction::MakeListEntries(entries)) if entries.len() == 2)
        && module.procedure_semantic_digest(procedure)
            == Some(CANONICAL_MONKE_DISCOVER_OFFSET_DIGEST)
}

fn list_iteration_snapshot(state: &ExecutionState, list: ListId) -> Option<Vec<Value>> {
    let list = state.heap.list(list).ok()?;
    (1..=list.len())
        .map(|index| list.get(index).ok().cloned())
        .collect()
}

pub(crate) fn discover_offset_native(
    src: DatumId,
    marker: &Value,
    state: &mut ExecutionState,
) -> Option<Value> {
    const MAX_MODEL_ENTRIES: usize = 1 << 20;
    const MAX_GRID_LINES: usize = 1 << 20;
    // Keep the synchronous native tier bounded and column arithmetic exactly
    // representable in DM's f32 number domain. Larger/custom inputs side-exit.
    const MAX_SCANNED_BYTES: usize = 8 * 1024 * 1024;
    let field = |name| FieldName::parse(name).ok();
    let Value::Datum(cached_map) =
        datum_field_or_initial(state, src, &field("cached_map")?).ok()?
    else {
        return None;
    };
    let Value::List(models) =
        datum_field_or_initial(state, cached_map, &field("grid_models")?).ok()?
    else {
        return None;
    };
    let model_keys = list_iteration_snapshot(state, models)?;
    if model_keys.len() > MAX_MODEL_ENTRIES {
        return None;
    }
    let marker = stringify_dm_value(marker, &state.heap).ok()?;
    let mut selected_key = Value::Null;
    for key in model_keys {
        selected_key = key.clone();
        let model =
            read_list_value(&state.heap, models, &key, state.is_associative_list(models)).ok()?;
        let found =
            execute_standard_builtin("findtext", &[model, Value::text(marker.as_str())], state)
                .ok()?;
        if runtime_truthy(&state.heap, &found).ok()? {
            break;
        }
    }

    let Value::List(grid_sets) =
        datum_field_or_initial(state, cached_map, &field("gridSets")?).ok()?
    else {
        return None;
    };
    let key_len = datum_field_or_initial(state, cached_map, &field("key_len")?)
        .ok()?
        .as_number()?;
    if !key_len.is_finite() || key_len.fract() != 0.0 || !(1.0..=64.0).contains(&key_len) {
        return None;
    }
    let key_len = key_len as usize;
    let Value::Text(selected_key) = selected_key else {
        return Some(Value::Null);
    };
    if !selected_key.is_ascii() || selected_key.len() != key_len {
        return None;
    }
    let grids = list_iteration_snapshot(state, grid_sets)?;
    let mut scanned_lines = 0_usize;
    let mut scanned_bytes = 0_usize;
    for grid in grids {
        let Value::Datum(grid) = grid else {
            return None;
        };
        let x = datum_field_or_initial(state, grid, &field("xcrd")?)
            .ok()?
            .as_number()?;
        let mut y = datum_field_or_initial(state, grid, &field("ycrd")?)
            .ok()?
            .as_number()?;
        let Value::List(lines) = datum_field_or_initial(state, grid, &field("gridLines")?).ok()?
        else {
            return None;
        };
        for line in list_iteration_snapshot(state, lines)? {
            scanned_lines = scanned_lines.checked_add(1)?;
            if scanned_lines > MAX_GRID_LINES {
                return None;
            }
            let Value::Text(line) = line else {
                return None;
            };
            if !line.is_ascii() {
                return None;
            }
            scanned_bytes = scanned_bytes.checked_add(line.len())?;
            if scanned_bytes > MAX_SCANNED_BYTES {
                return None;
            }
            for (column, chunk) in line.as_bytes().chunks_exact(key_len).enumerate() {
                if chunk == selected_key.as_bytes() {
                    let result = state.heap.allocate_list();
                    state
                        .heap
                        .list_mut(result)
                        .ok()?
                        .extend_positional([Value::number(x + column as f32), Value::number(y)]);
                    return Some(Value::List(result));
                }
            }
            y -= 1.0;
        }
    }
    Some(Value::Null)
}

pub(crate) fn try_run_discover_offset_fast_path(
    module: &Module,
    procedure: ProcedureId,
    program: &Program,
    frame: &mut CallFrame,
    remaining_steps: u64,
    state: &mut ExecutionState,
) -> Option<u64> {
    if remaining_steps < 16
        || !frame.stack.is_empty()
        || !trusted_discover_offset_target(module, procedure, program)
    {
        return None;
    }
    let Value::Datum(src) = frame.src else {
        return None;
    };
    let result = discover_offset_native(src, frame.locals.first()?, state)?;
    let return_index = program
        .instructions
        .iter()
        .rposition(|instruction| matches!(instruction, Instruction::Return))?;
    frame.stack.push(result);
    frame.instruction = return_index;
    NATIVE_DISCOVER_OFFSET_ACTIVATIONS.fetch_add(1, Ordering::Relaxed);
    Some(16)
}

pub(crate) fn try_run_parsed_dmm_new_fast_path(
    module: &Module,
    procedure: ProcedureId,
    program: &Program,
    frame: &mut CallFrame,
    remaining_steps: u64,
    state: &mut ExecutionState,
) -> Option<u64> {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("DREAM64_DISABLE_PARSED_DMM_CACHE").is_some()) {
        return None;
    }
    let canonical_path = module.procedure_path(procedure)?.split('@').next()?;
    if remaining_steps < 32
        || !matches!(
            canonical_path,
            "/datum/parsed_map/New" | "/datum/parsed_map/proc/New"
        )
        || program.parameter_count != 8
        || frame.locals.len() < 8
        || !frame.stack.is_empty()
        || frame
            .locals
            .get(1..8)?
            .iter()
            .any(|value| !matches!(value, Value::Null))
    {
        return None;
    }
    let Value::Datum(src) = frame.src else {
        return None;
    };
    if state.heap.datum(src).ok()?.type_path().as_str() != "/datum/parsed_map" {
        return None;
    }
    let Value::File(file) = frame.locals.first()? else {
        return None;
    };
    let normalized = normalized_dmm_cache_path(file)?;
    let parsed = state.parsed_dmm_cache.get(&normalized)?.clone();
    if !artifact_dmm_source_matches(state, file, parsed.digest) {
        return None;
    }

    let allocate_bounds = |state: &mut ExecutionState| -> Option<ListId> {
        let list = state.heap.allocate_list();
        for coordinate in parsed.bounds {
            state
                .heap
                .list_mut(list)
                .ok()?
                .add(Value::number(coordinate as f32));
        }
        Some(list)
    };
    let bounds = allocate_bounds(state)?;
    let parsed_bounds = allocate_bounds(state)?;
    let models = state.heap.allocate_list();
    state.mark_associative_list(models);
    for (key, model) in &parsed.models {
        write_list_value(
            &mut state.heap,
            models,
            Value::text(key.as_str()),
            Value::text(model.as_str()),
            true,
        )
        .ok()?;
    }
    let grid_sets = state.heap.allocate_list();
    let grid_type = TypePath::parse("/datum/grid_set").ok()?;
    let field = |name| FieldName::parse(name).ok();
    for grid in &parsed.grids {
        let datum = state.heap.allocate_datum(grid_type.clone());
        let lines = state.heap.allocate_list();
        for line in &grid.lines {
            state
                .heap
                .list_mut(lines)
                .ok()?
                .add(Value::text(line.as_str()));
        }
        for (name, value) in [
            ("xcrd", Value::number(grid.x as f32)),
            ("ycrd", Value::number(grid.y as f32)),
            ("zcrd", Value::number(grid.z as f32)),
            ("gridLines", Value::List(lines)),
        ] {
            state
                .heap
                .set_datum_field(datum, field(name)?, value)
                .ok()?;
        }
        state
            .heap
            .list_mut(grid_sets)
            .ok()?
            .add(Value::Datum(datum));
    }
    for (name, value) in [
        ("original_path", Value::Text(Arc::clone(file))),
        (
            "map_format",
            Value::text(if parsed.tgm { "tgm" } else { "dmm" }),
        ),
        ("key_len", Value::number(parsed.key_len as f32)),
        ("line_len", Value::number(parsed.line_len as f32)),
        ("grid_models", Value::List(models)),
        ("gridSets", Value::List(grid_sets)),
        ("bounds", Value::List(bounds)),
        ("parsed_bounds", Value::List(parsed_bounds)),
    ] {
        state.heap.set_datum_field(src, field(name)?, value).ok()?;
    }
    let return_index = program
        .instructions
        .iter()
        .rposition(|instruction| matches!(instruction, Instruction::Return))?;
    frame.stack.push(Value::Null);
    frame.instruction = return_index;
    Some(32)
}

pub(crate) fn try_run_dmm_preload_measurement_fast_path(
    module: &Module,
    procedure: ProcedureId,
    program: &Program,
    frame: &mut CallFrame,
    remaining_steps: u64,
    state: &mut ExecutionState,
) -> Option<u64> {
    if remaining_steps < 8
        || module.procedure_path(procedure)?.split('@').next()?
            != "/datum/map_template/proc/preload_size"
        || program.parameter_count != 2
        || program.local_count < 2
        || !frame.stack.is_empty()
    {
        return None;
    }
    let Value::Datum(src) = frame.src else {
        return None;
    };
    let path = match frame.locals.first()? {
        Value::File(path) | Value::Text(path) => path.as_ref(),
        _ => return None,
    };
    // `cache=TRUE` must construct and retain the parsed-map datum.
    if runtime_truthy(&state.heap, frame.locals.get(1)?).ok()? {
        return None;
    }
    let measurement = *state
        .dmm_measurements
        .get(&normalized_dmm_cache_path(path)?)?;
    if !artifact_dmm_source_matches(state, path, measurement.digest) {
        return None;
    }
    let bounds = state.heap.allocate_list();
    for coordinate in measurement.bounds {
        state
            .heap
            .list_mut(bounds)
            .ok()?
            .add(Value::number(coordinate as f32));
    }
    let width = FieldName::parse("width").ok()?;
    let height = FieldName::parse("height").ok()?;
    state
        .heap
        .set_datum_field(src, width, Value::number(measurement.bounds[3] as f32))
        .ok()?;
    state
        .heap
        .set_datum_field(src, height, Value::number(measurement.bounds[4] as f32))
        .ok()?;
    let return_index = program
        .instructions
        .iter()
        .rposition(|instruction| matches!(instruction, Instruction::Return))?;
    frame.stack.push(Value::List(bounds));
    frame.instruction = return_index;
    Some(8)
}

thread_local! {
    static NUMERIC_JIT_CACHE: RefCell<HashMap<(u64, ProcedureId), Option<CompiledNumericTrace>>> =
        RefCell::new(HashMap::new());
    static LUMCOUNT_JIT_CACHE: RefCell<HashMap<(u64, ProcedureId), Option<LumcountTrace>>> =
        RefCell::new(HashMap::new());
    static ROOTED_LIST_JIT_CACHE: RefCell<HashMap<(u64, ProcedureId), Option<RootedListTrace>>> =
        RefCell::new(HashMap::new());
    pub(crate) static REGISTER_SIGNAL_FAST_CACHE: RefCell<HashMap<(u64, ProcedureId), Option<RegisterSignalTrace>>> =
        RefCell::new(HashMap::new());
    static CAMERA_CHUNK_FAST_CACHE: RefCell<HashMap<(u64, ProcedureId), Option<CameraChunkTrace>>> =
        RefCell::new(HashMap::new());
}

struct CameraChunkTrace {
    mapping_global: FieldName,
    plane_offset: FieldName,
    chunks: FieldName,
}

fn compile_camera_chunk_trace(
    module: &Module,
    procedure: ProcedureId,
    program: &Program,
) -> Option<CameraChunkTrace> {
    let canonical_path = module.procedure_path(procedure)?.split('@').next()?;
    if canonical_path != "/datum/cameranet/proc/get_camera_chunk"
        || program.parameter_count != 3
        || program.local_count != 5
        || program.instructions.len() != 55
    {
        return None;
    }
    let instructions = program.instructions.as_slice();
    let Instruction::LoadGlobal(mapping_global) = &instructions[18] else {
        return None;
    };
    let Instruction::LoadDeclaredField(plane_offset) = &instructions[19] else {
        return None;
    };
    let Instruction::LoadField(chunks) = &instructions[47] else {
        return None;
    };
    let number_at = |index| match &instructions[index] {
        Instruction::PushNumber(number) => Some(number.to_f32()),
        _ => None,
    };
    let call_at = |index| match &instructions[index] {
        Instruction::Call {
            procedure,
            argument_count: 2,
            ..
        } => Some(*procedure),
        _ => None,
    };
    let max_target = call_at(7)?;
    let max_program = module.resolve_procedure(max_target).ok()?;
    let canonical = number_at(1) == Some(8.0)
        && number_at(4) == Some(8.0)
        && number_at(6) == Some(1.0)
        && number_at(10) == Some(8.0)
        && number_at(13) == Some(8.0)
        && number_at(15) == Some(1.0)
        && number_at(26) == Some(0.0)
        && number_at(27) == Some(0.0)
        && call_at(16) == Some(max_target)
        && canonical_static_native_builtin(module, max_target, max_program) == Some("max")
        && matches!(instructions[0], Instruction::LoadLocal(0))
        && matches!(instructions[2], Instruction::Divide)
        && matches!(instructions[3], Instruction::Round { argument_count: 1 })
        && matches!(instructions[5], Instruction::Multiply)
        && matches!(
            instructions[7],
            Instruction::Call {
                argument_count: 2,
                ..
            }
        )
        && matches!(instructions[8], Instruction::StoreLocal(0))
        && matches!(instructions[9], Instruction::LoadLocal(1))
        && matches!(instructions[11], Instruction::Divide)
        && matches!(instructions[12], Instruction::Round { argument_count: 1 })
        && matches!(instructions[14], Instruction::Multiply)
        && matches!(
            instructions[16],
            Instruction::Call {
                argument_count: 2,
                ..
            }
        )
        && matches!(instructions[17], Instruction::StoreLocal(1))
        && matches!(instructions[20], Instruction::JumpIfFalse(26))
        && matches!(instructions[28], Instruction::NotEqual)
        && matches!(instructions[29], Instruction::JumpIfFalse(46))
        && matches!(&instructions[48], Instruction::PushText(template) if template.as_ref() == "[],[],[]")
        && matches!(instructions[49], Instruction::LoadLocal(0))
        && matches!(instructions[50], Instruction::LoadLocal(1))
        && matches!(instructions[51], Instruction::LoadLocal(2))
        && matches!(&instructions[52], Instruction::StandardBuiltin { name, argument_count: 4, .. } if name == "text")
        && matches!(instructions[53], Instruction::IndexList)
        && matches!(instructions[54], Instruction::Return);
    canonical.then(|| CameraChunkTrace {
        mapping_global: mapping_global.clone(),
        plane_offset: plane_offset.clone(),
        chunks: chunks.clone(),
    })
}

pub(crate) fn try_run_camera_chunk_fast_path(
    module: &Module,
    procedure: ProcedureId,
    program: &Program,
    frame: &mut CallFrame,
    remaining_steps: u64,
    state: &mut ExecutionState,
) -> Option<u64> {
    if remaining_steps < 33
        || program.instructions.len() != 55
        || program.parameter_count != 3
        || program.local_count != 5
        || !frame.stack.is_empty()
    {
        return None;
    }
    let Value::Datum(src) = frame.src else {
        return None;
    };
    let x = frame.locals.first()?.as_number()?;
    let y = frame.locals.get(1)?.as_number()?;
    let z = frame.locals.get(2)?.as_number()?;
    if !x.is_finite() || !y.is_finite() || !z.is_finite() {
        return None;
    }
    let key = (module.identity.0, procedure);
    CAMERA_CHUNK_FAST_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let trace = cache
            .entry(key)
            .or_insert_with(|| compile_camera_chunk_trace(module, procedure, program))
            .as_ref()?;
        let Value::Datum(mapping) = state.global(&trace.mapping_global)?.clone() else {
            return None;
        };
        let plane_offset = datum_field_or_initial(state, mapping, &trace.plane_offset).ok()?;
        if runtime_truthy(&state.heap, &plane_offset).ok()? {
            return None;
        }
        let Value::List(chunks) = datum_field_or_shared(state, src, &trace.chunks).ok()? else {
            return None;
        };
        if state.heap.list(chunks).is_err()
            || state.global_vars_proxy == Some(chunks)
            || state.datum_vars_proxies.contains_key(&chunks)
        {
            return None;
        }
        let x = ((x / 8.0).floor() * 8.0).max(1.0);
        let y = ((y / 8.0).floor() * 8.0).max(1.0);
        let key = Value::text(format!(
            "{},{},{}",
            Value::number(x),
            Value::number(y),
            Value::number(z)
        ));
        let result =
            match read_list_value(&state.heap, chunks, &key, state.is_associative_list(chunks)) {
                Ok(value) => value,
                Err(ValueError::MissingKey) => Value::Null,
                Err(_) => return None,
            };
        frame.locals[0] = Value::number(x);
        frame.locals[1] = Value::number(y);
        frame.stack.push(result);
        frame.instruction = 54;
        Some(33)
    })
}

pub(crate) struct RegisterSignalTrace {
    gc_destroyed: FieldName,
    signal_procs: FieldName,
    listen_lookup: FieldName,
}

pub(crate) fn compile_register_signal_trace(
    module: &Module,
    procedure: ProcedureId,
    program: &Program,
) -> Option<RegisterSignalTrace> {
    let canonical_path = module.procedure_path(procedure)?.split('@').next()?;
    if canonical_path != "/datum/proc/RegisterSignal"
        || program.parameter_count != 4
        || program.local_count != 14
        || program.instructions.len() != 140
    {
        return None;
    }
    let instructions = program.instructions.as_slice();
    let Instruction::LoadField(gc_destroyed) = &instructions[10] else {
        return None;
    };
    let Instruction::LoadDeclaredField(target_gc_destroyed) = &instructions[22] else {
        return None;
    };
    let Instruction::LogicalOrEmptyListField(signal_procs) = &instructions[70] else {
        return None;
    };
    let Instruction::LogicalOrEmptyListField(listen_lookup) = &instructions[77] else {
        return None;
    };
    if gc_destroyed != target_gc_destroyed
        || gc_destroyed.as_str() != "gc_destroyed"
        || signal_procs.as_str() != "_signal_procs"
        || listen_lookup.as_str() != "_listen_lookup"
        || !matches!(instructions[26], Instruction::LoadLocal(1))
        || !matches!(
            instructions[27],
            Instruction::TypePredicate {
                kind: TypePredicateKind::IsList,
                argument_count: 1
            }
        )
        || !matches!(instructions[74], Instruction::LogicalOrEmptyListIndex)
        || !matches!(instructions[80], Instruction::IndexLocalList(9))
        || !matches!(instructions[86], Instruction::SetListIndex)
        || !matches!(instructions[111], Instruction::IndexLocalList(10))
        || !matches!(
            instructions[114],
            Instruction::TypePredicate {
                kind: TypePredicateKind::IsNull,
                argument_count: 1
            }
        )
        || !matches!(instructions[120], Instruction::SetListIndex)
        || !matches!(instructions[121], Instruction::Jump(138))
        || !matches!(instructions[138], Instruction::LoadResult)
        || !matches!(instructions[139], Instruction::Return)
    {
        return None;
    }
    Some(RegisterSignalTrace {
        gc_destroyed: gc_destroyed.clone(),
        signal_procs: signal_procs.clone(),
        listen_lookup: listen_lookup.clone(),
    })
}

pub(crate) fn try_run_register_signal_fast_path(
    module: &Module,
    procedure: ProcedureId,
    program: &Program,
    frame: &mut CallFrame,
    remaining_steps: u64,
    state: &mut ExecutionState,
) -> Option<u64> {
    // This is the overwhelmingly common first-registration path. Overrides,
    // list promotion, warning behavior, and unusual receivers stay in the
    // bytecode interpreter before any mutation occurs.
    if remaining_steps < 54
        || program.instructions.len() != 140
        || program.parameter_count != 4
        || program.local_count != 14
        || !frame.stack.is_empty()
    {
        return None;
    }
    let override_supplied = frame.supplied_parameters.get(3).copied().unwrap_or(false);
    let accounted_steps = if override_supplied { 54 } else { 56 };
    if remaining_steps < accounted_steps {
        return None;
    }
    let Value::Datum(src) = frame.src else {
        return None;
    };
    let Value::Datum(target) = frame.locals.first()?.clone() else {
        return None;
    };
    let signal_type = frame.locals.get(1)?.clone();
    let proctype = frame.locals.get(2)?.clone();
    let override_enabled =
        runtime_truthy(&state.heap, frame.locals.get(3).unwrap_or(&Value::Null)).ok()?;
    // Signals are canonically text. Restricting the native path here retains
    // the interpreter's exact coercion/error behavior for every odd key type.
    if !matches!(signal_type, Value::Text(_)) {
        return None;
    }
    let key = (module.identity.0, procedure);
    REGISTER_SIGNAL_FAST_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let trace = cache
            .entry(key)
            .or_insert_with(|| compile_register_signal_trace(module, procedure, program))
            .as_ref()?;
        let src_destroyed = datum_field_or_initial(state, src, &trace.gc_destroyed).ok()?;
        let target_destroyed = datum_field_or_initial(state, target, &trace.gc_destroyed).ok()?;
        if runtime_truthy(&state.heap, &src_destroyed).ok()?
            || runtime_truthy(&state.heap, &target_destroyed).ok()?
        {
            return None;
        }
        let ordinary_list = |state: &ExecutionState, list: ListId| {
            !state.reference_lists.contains(&list)
                && !state.is_visibility_list(list)
                && state.global_vars_proxy != Some(list)
                && !state.datum_vars_proxies.contains_key(&list)
                && state.heap.list(list).is_ok()
        };
        let procs_value = datum_field_or_shared(state, src, &trace.signal_procs).ok()?;
        let procs = if runtime_truthy(&state.heap, &procs_value).ok()? {
            let Value::List(procs) = procs_value else {
                return None;
            };
            ordinary_list(state, procs).then_some(procs)
        } else {
            None
        };
        let lookup_value = datum_field_or_shared(state, target, &trace.listen_lookup).ok()?;
        let lookup = if runtime_truthy(&state.heap, &lookup_value).ok()? {
            let Value::List(lookup) = lookup_value else {
                return None;
            };
            ordinary_list(state, lookup).then_some(lookup)
        } else {
            None
        };
        if procs.is_none() && runtime_truthy(&state.heap, &procs_value).ok()?
            || lookup.is_none() && runtime_truthy(&state.heap, &lookup_value).ok()?
        {
            return None;
        }
        let target_procs = if let Some(procs) = procs {
            let current = match read_list_value(
                &state.heap,
                procs,
                &Value::Datum(target),
                state.is_associative_list(procs),
            ) {
                Ok(value) => value,
                Err(ValueError::MissingKey) => Value::Null,
                Err(_) => return None,
            };
            if runtime_truthy(&state.heap, &current).ok()? {
                let Value::List(target_procs) = current else {
                    return None;
                };
                if !ordinary_list(state, target_procs) {
                    return None;
                }
                Some(target_procs)
            } else {
                None
            }
        } else {
            None
        };
        let existing = if let Some(target_procs) = target_procs {
            match read_list_value(
                &state.heap,
                target_procs,
                &signal_type,
                state.is_associative_list(target_procs),
            ) {
                Ok(value) => value,
                Err(ValueError::MissingKey) => Value::Null,
                Err(_) => return None,
            }
        } else {
            Value::Null
        };
        // Formatting the warning and collecting its DM stack trace are
        // observable. Side-exit before mutation so bytecode performs it once.
        if runtime_truthy(&state.heap, &existing).ok()? && !override_enabled {
            return None;
        }
        let looked_up = if let Some(lookup) = lookup {
            match read_list_value(
                &state.heap,
                lookup,
                &signal_type,
                state.is_associative_list(lookup),
            ) {
                Ok(value) => value,
                Err(ValueError::MissingKey) => Value::Null,
                Err(_) => return None,
            }
        } else {
            Value::Null
        };
        if let Value::List(listeners) = &looked_up
            && !ordinary_list(state, *listeners)
        {
            return None;
        }

        // Every fallible read and shape guard is complete. Materialize the
        // exact `||= list()` chain, then perform the two canonical associations.
        let procs = if let Some(procs) = procs {
            procs
        } else {
            let procs = state.heap.allocate_list();
            assign_datum_or_shared_field(
                state,
                src,
                trace.signal_procs.clone(),
                Value::List(procs),
            )
            .ok()?;
            procs
        };
        let target_procs = if let Some(target_procs) = target_procs {
            target_procs
        } else {
            let target_procs = state.heap.allocate_list();
            state
                .heap
                .list_mut(procs)
                .ok()?
                .set_key(Value::Datum(target), Value::List(target_procs));
            state.mark_associative_list(procs);
            target_procs
        };
        let lookup = if let Some(lookup) = lookup {
            lookup
        } else {
            let lookup = state.heap.allocate_list();
            assign_datum_or_shared_field(
                state,
                target,
                trace.listen_lookup.clone(),
                Value::List(lookup),
            )
            .ok()?;
            lookup
        };
        state
            .heap
            .list_mut(target_procs)
            .ok()?
            .set_key(signal_type.clone(), proctype);
        state.mark_associative_list(target_procs);
        match looked_up {
            Value::Null => {
                state
                    .heap
                    .list_mut(lookup)
                    .ok()?
                    .set_key(signal_type, Value::Datum(src));
                state.mark_associative_list(lookup);
            }
            Value::List(listeners) => {
                state.heap.list_mut(listeners).ok()?.add(Value::Datum(src));
            }
            listener => {
                let listeners = state.heap.allocate_list();
                let values = state.heap.list_mut(listeners).ok()?;
                values.add(listener);
                values.add(Value::Datum(src));
                state
                    .heap
                    .list_mut(lookup)
                    .ok()?
                    .set_key(signal_type, Value::List(listeners));
                state.mark_associative_list(lookup);
            }
        }
        frame.instruction = 138;
        Some(accounted_steps)
    })
}

pub(crate) struct RootedListTrace {
    compiled: CompiledRootedBlock,
    source_field: FieldName,
    target_field: FieldName,
}

pub(crate) fn compile_rooted_list_trace(program: &Program) -> Option<RootedListTrace> {
    let [
        Instruction::LoadSrc,
        Instruction::LogicalOrEmptyListField(source_field),
        Instruction::StoreLocal(2),
        Instruction::LoadLocal(2),
        Instruction::LoadLocal(0),
        Instruction::LogicalOrEmptyListIndex,
        Instruction::StoreLocal(3),
        Instruction::LoadLocal(0),
        Instruction::LogicalOrEmptyListField(target_field),
        Instruction::StoreLocal(4),
        Instruction::LoadLocal(3),
        Instruction::Return,
    ] = program.instructions.as_slice()
    else {
        return None;
    };
    if program.parameter_count < 1 || program.local_count < 5 {
        return None;
    }
    Some(RootedListTrace {
        compiled: compile_safe_rooted_block().ok()?,
        source_field: source_field.clone(),
        target_field: target_field.clone(),
    })
}

pub(crate) fn try_run_rooted_list_jit(
    module: &Module,
    procedure: ProcedureId,
    program: &Program,
    frame: &mut CallFrame,
    remaining_steps: u64,
    state: &mut ExecutionState,
) -> Option<u64> {
    // The current VM-owned helper batch is correctness-complete but its
    // end-to-end release benchmark is slower than bytecode dispatch. Keep it
    // opt-in in production until helpers execute in native code directly.
    // Reject the unique rooted trace shape before consulting configuration or
    // a thread-local cache. Almost every procedure enters here and cannot
    // possibly match this exact eleven-instruction tier.
    if remaining_steps < 11
        || program.instructions.len() != 11
        || program.parameter_count < 1
        || program.local_count < 5
        || !rooted_jit_enabled()
        || jit_disabled()
    {
        return None;
    }
    let Value::Datum(src) = frame.src else {
        return None;
    };
    let Value::Datum(target) = frame.locals.first()?.clone() else {
        return None;
    };
    let key = (module.identity.0, procedure);
    ROOTED_LIST_JIT_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let trace = cache
            .entry(key)
            .or_insert_with(|| compile_rooted_list_trace(program))
            .as_ref()?;

        // Make the batch infallible before its first mutation. Any shape or
        // heap state with observable error/side-exit behavior stays entirely
        // in the reference interpreter.
        let source_value = datum_field_or_initial(state, src, &trace.source_field).ok()?;
        let source_truthy = runtime_truthy(&state.heap, &source_value).ok()?;
        if source_truthy {
            let Value::List(list) = source_value else {
                return None;
            };
            state.heap.list(list).ok()?;
        }
        let target_value = datum_field_or_initial(state, target, &trace.target_field).ok()?;
        runtime_truthy(&state.heap, &target_value).ok()?;

        let mut values =
            SmallVec::<[Value; 8]>::from_vec(vec![Value::Datum(src), Value::Datum(target)]);
        let mut roots = [0_u32, 1, 0, 0, 0];
        let mut stack = Vec::with_capacity(2);
        let source_field = trace.source_field.clone();
        let target_field = trace.target_field.clone();
        let mut dispatch =
            |roots: &mut [u32], stack: &mut [u32], stack_len: &mut usize, start_pc, budget| {
                if start_pc != 0 || budget < 11 || roots.len() < 5 || stack.is_empty() {
                    return RootedBlockOutcome::BudgetExhausted {
                        instruction: start_pc,
                        steps: 0,
                    };
                }
                let procs = logical_or_empty_list_field(
                    state,
                    values[roots[0] as usize].clone(),
                    &source_field,
                )
                .expect("rooted list trace prevalidated source field");
                values.push(procs.clone());
                roots[2] = (values.len() - 1) as u32;
                let target_procs =
                    logical_or_empty_list_index(state, procs, values[roots[1] as usize].clone())
                        .expect("rooted list trace prevalidated list receiver");
                values.push(target_procs);
                roots[3] = (values.len() - 1) as u32;
                let lookup = logical_or_empty_list_field(
                    state,
                    values[roots[1] as usize].clone(),
                    &target_field,
                )
                .expect("rooted list trace prevalidated target field");
                values.push(lookup);
                roots[4] = (values.len() - 1) as u32;
                stack[0] = roots[3];
                *stack_len = 1;
                RootedBlockOutcome::Completed {
                    instruction: 11,
                    steps: 11,
                }
            };
        let RootedBlockOutcome::Completed {
            instruction: 11,
            steps: 11,
        } = trace
            .compiled
            .run_with(&mut roots, &mut stack, 0, 11, &mut dispatch)
        else {
            return None;
        };
        frame.locals[2] = values[roots[2] as usize].clone();
        frame.locals[3] = values[roots[3] as usize].clone();
        frame.locals[4] = values[roots[4] as usize].clone();
        frame.stack.clear();
        frame
            .stack
            .extend(stack.into_iter().map(|slot| values[slot as usize].clone()));
        frame.instruction = 11;
        Some(11)
    })
}

pub(crate) struct LumcountTrace {
    compiled: CompiledNumericTrace,
    fields: [FieldName; 4],
    lighting_global: FieldName,
    queue_field: FieldName,
}

pub(crate) fn try_run_guarded_jit(
    module: &Module,
    procedure: ProcedureId,
    program: &Program,
    frame: &mut CallFrame,
    remaining_steps: u64,
    state: &mut ExecutionState,
) -> Option<(NumericRunOutcome, bool)> {
    // Keep the runtime kill switch authoritative for every native tier. This
    // is also essential for trustworthy whole-server A/B diagnosis: the
    // specialized field trace must not remain active in the "JIT disabled"
    // process while the generic numeric tier is bypassed.
    if jit_disabled() {
        return None;
    }
    // Lumcount is an exact 48-instruction/four-local trace. Do not make every
    // unrelated procedure pay a second thread-local negative-cache lookup.
    if program.instructions.len() == 48
        && program.local_count == 4
        && let Some(outcome) =
            try_run_lumcount_jit(module, procedure, program, frame, remaining_steps, state)
    {
        return Some((outcome, true));
    }
    // Every generic numeric trace must lower every instruction. Most DM
    // procedures expose a disqualifying heap/dynamic opcode immediately; a
    // four-op necessary-condition gate avoids hashing into the thread-local
    // negative cache on each of their millions of invocations. Returning true
    // is deliberately conservative and leaves full validation to the compiler.
    if !numeric_jit_prefix_candidate(program) {
        return None;
    }
    try_run_numeric_jit(module, procedure, program, frame, remaining_steps)
        .map(|outcome| (outcome, false))
}

pub(crate) fn numeric_jit_prefix_candidate(program: &Program) -> bool {
    !program.instructions.is_empty()
        && program.instructions.iter().take(4).all(|instruction| {
            matches!(
                instruction,
                Instruction::PushNumber(_)
                    | Instruction::LoadLocal(_)
                    | Instruction::StoreLocal(_)
                    | Instruction::Add
                    | Instruction::Subtract
                    | Instruction::Multiply
                    | Instruction::Divide
                    | Instruction::Negate
                    | Instruction::Equal
                    | Instruction::NotEqual
                    | Instruction::Less
                    | Instruction::LessEqual
                    | Instruction::Greater
                    | Instruction::GreaterEqual
                    | Instruction::Jump(_)
                    | Instruction::JumpIfFalse(_)
                    | Instruction::Return
            )
        })
}

pub(crate) fn jit_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var_os("DREAM64_DISABLE_JIT").is_some())
}

fn rooted_jit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| cfg!(test) || std::env::var_os("DREAM64_ENABLE_ROOTED_JIT").is_some())
}

fn try_run_lumcount_jit(
    module: &Module,
    procedure: ProcedureId,
    program: &Program,
    frame: &mut CallFrame,
    remaining_steps: u64,
    state: &mut ExecutionState,
) -> Option<NumericRunOutcome> {
    // This batched trace intentionally runs atomically. Near a scheduler
    // boundary the interpreter retains exact per-opcode yield points.
    if remaining_steps < 48 {
        return None;
    }
    let Value::Datum(src) = frame.src else {
        return None;
    };
    let key = (module.identity.0, procedure);
    LUMCOUNT_JIT_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let trace = cache
            .entry(key)
            .or_insert_with(|| compile_lumcount_trace(program));
        let trace = trace.as_ref()?;
        let mut numeric_locals = SmallVec::<[f32; 8]>::new();
        numeric_locals.resize(program.local_count, 0.0);
        for (index, local) in frame.locals.iter().take(3).enumerate() {
            numeric_locals[index] = local.as_number()?;
        }
        // The canonical procedure returns before observing src, fields, the
        // lighting global, or its queue when every delta is zero. Preserve
        // that ordering and avoid all heap guards/native entry on this very
        // common no-op path.
        if numeric_locals[..3].iter().all(|value| *value == 0.0) {
            return Some(NumericRunOutcome::Returned {
                value: 0.0,
                steps: 13,
            });
        }
        let field_values = trace
            .fields
            .iter()
            .map(|field| datum_field_or_initial(state, src, field).ok()?.as_number())
            .collect::<Option<SmallVec<[f32; 8]>>>()?;
        let Value::Datum(lighting) = state.global(&trace.lighting_global)?.clone() else {
            return None;
        };
        let Value::List(queue) =
            datum_field_or_initial(state, lighting, &trace.queue_field).ok()?
        else {
            return None;
        };
        if state.heap.list(queue).is_err() {
            return None;
        }
        if let Some(native) = frame.numeric_jit_state_mut() {
            native.fields.copy_from_slice(&field_values);
        } else {
            frame.set_numeric_jit_state(
                trace
                    .compiled
                    .initial_state_with_fields(&numeric_locals, &field_values),
            );
        }
        let budget = u32::try_from(remaining_steps).unwrap_or(u32::MAX);
        let outcome = trace
            .compiled
            .run_budgeted(frame.numeric_jit_state_mut()?, budget)?;
        let native = frame.numeric_jit_state_mut()?;
        for (index, field) in trace.fields.iter().enumerate() {
            if native.dirty_fields & (1_u64 << index) != 0 {
                state
                    .heap
                    .set_datum_field(src, field.clone(), Value::number(native.fields[index]))
                    .ok()?;
            }
        }
        native.dirty_fields = 0;
        if native.action_bits & 1 != 0 {
            state.heap.list_mut(queue).ok()?.add(Value::Datum(src));
        }
        native.action_bits = 0;
        let NumericRunOutcome::Returned { value, .. } = outcome else {
            return None;
        };
        let first_truthy = numeric_locals[0] != 0.0;
        let second_truthy = numeric_locals[1] != 0.0;
        let third_truthy = numeric_locals[2] != 0.0;
        let exact_steps = if first_truthy {
            31 + u32::from(field_values[3] == 0.0) * 9
        } else if second_truthy {
            34 + u32::from(field_values[3] == 0.0) * 9
        } else if third_truthy {
            35 + u32::from(field_values[3] == 0.0) * 9
        } else {
            13
        };
        Some(NumericRunOutcome::Returned {
            value,
            steps: exact_steps,
        })
    })
}

pub(crate) fn compile_lumcount_trace(program: &Program) -> Option<LumcountTrace> {
    let instructions = program.instructions.as_slice();
    if program.local_count != 4 || instructions.len() != 48 {
        return None;
    }
    let field_at = |index| match instructions.get(index)? {
        Instruction::LoadField(field) | Instruction::StoreField(field) => Some(field.clone()),
        _ => None,
    };
    let global_at = |index| match instructions.get(index)? {
        Instruction::LoadGlobal(field) => Some(field.clone()),
        _ => None,
    };
    let lum_r = field_at(17)?;
    let lum_g = field_at(23)?;
    let lum_b = field_at(29)?;
    let needs_update = field_at(34)?;
    let queue_field = field_at(42)?;
    let lighting_global = global_at(40)?;
    let canonical = matches!(instructions,
        [Instruction::LoadLocal(0), Instruction::Duplicate, Instruction::JumpIfFalse(4), Instruction::Jump(6), Instruction::Pop,
         Instruction::LoadLocal(1), Instruction::Duplicate, Instruction::JumpIfFalse(9), Instruction::Jump(11), Instruction::Pop,
         Instruction::LoadLocal(2), Instruction::Not, Instruction::JumpIfFalse(15), Instruction::LoadResult, Instruction::Return,
         Instruction::LoadSrc, Instruction::Duplicate, Instruction::LoadField(_), Instruction::LoadLocal(0), Instruction::CompoundAssignment(CompoundAssignmentOperator::Add), Instruction::StoreField(_),
         Instruction::LoadSrc, Instruction::Duplicate, Instruction::LoadField(_), Instruction::LoadLocal(1), Instruction::CompoundAssignment(CompoundAssignmentOperator::Add), Instruction::StoreField(_),
         Instruction::LoadSrc, Instruction::Duplicate, Instruction::LoadField(_), Instruction::LoadLocal(2), Instruction::CompoundAssignment(CompoundAssignmentOperator::Add), Instruction::StoreField(_),
         Instruction::LoadSrc, Instruction::LoadField(_), Instruction::Not, Instruction::JumpIfFalse(46), Instruction::LoadSrc, Instruction::PushNumber(one), Instruction::StoreField(_),
         Instruction::LoadGlobal(_), Instruction::Duplicate, Instruction::LoadField(_), Instruction::LoadSrc, Instruction::CompoundAssignment(CompoundAssignmentOperator::Add), Instruction::StoreField(_), Instruction::LoadResult, Instruction::Return]
         if one.to_f32() == 1.0)
        && field_at(20)? == lum_r
        && field_at(26)? == lum_g
        && field_at(32)? == lum_b
        && field_at(39)? == needs_update
        && field_at(45)? == queue_field;
    if !canonical {
        return None;
    }
    let native = vec![
        NumericInstruction::LoadLocal(0),
        NumericInstruction::Constant(0.0),
        NumericInstruction::NotEqual,
        NumericInstruction::JumpIfFalse(5),
        NumericInstruction::Jump(14),
        NumericInstruction::LoadLocal(1),
        NumericInstruction::Constant(0.0),
        NumericInstruction::NotEqual,
        NumericInstruction::JumpIfFalse(10),
        NumericInstruction::Jump(14),
        NumericInstruction::LoadLocal(2),
        NumericInstruction::Constant(0.0),
        NumericInstruction::NotEqual,
        NumericInstruction::JumpIfFalse(37),
        NumericInstruction::LoadField(0),
        NumericInstruction::LoadLocal(0),
        NumericInstruction::Add,
        NumericInstruction::StoreField(0),
        NumericInstruction::LoadField(1),
        NumericInstruction::LoadLocal(1),
        NumericInstruction::Add,
        NumericInstruction::StoreField(1),
        NumericInstruction::LoadField(2),
        NumericInstruction::LoadLocal(2),
        NumericInstruction::Add,
        NumericInstruction::StoreField(2),
        NumericInstruction::LoadField(3),
        NumericInstruction::Constant(0.0),
        NumericInstruction::Equal,
        NumericInstruction::JumpIfFalse(35),
        NumericInstruction::Constant(1.0),
        NumericInstruction::StoreField(3),
        NumericInstruction::RaiseAction(0),
        NumericInstruction::Constant(0.0),
        NumericInstruction::Return,
        NumericInstruction::Constant(0.0),
        NumericInstruction::Return,
        NumericInstruction::Constant(0.0),
        NumericInstruction::Return,
    ];
    let compiled = compile_numeric_field_trace(&native, program.local_count, 4)
        .inspect_err(|error| eprintln!("lumcount JIT compile rejected: {error}"))
        .ok()?;
    Some(LumcountTrace {
        compiled,
        fields: [lum_r, lum_g, lum_b, needs_update],
        lighting_global,
        queue_field,
    })
}

fn try_run_numeric_jit(
    module: &Module,
    procedure: ProcedureId,
    program: &Program,
    frame: &mut CallFrame,
    remaining_steps: u64,
) -> Option<NumericRunOutcome> {
    let key = (module.identity.0, procedure);
    NUMERIC_JIT_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let trace = cache.entry(key).or_insert_with(|| {
            numeric_trace_instructions(program).and_then(|instructions| {
                compile_numeric_trace(&instructions, program.local_count).ok()
            })
        });
        let trace = trace.as_ref()?;
        if frame.numeric_jit_state().is_none() {
            let mut numeric_locals = vec![0.0; program.local_count];
            for (index, local) in frame.locals.iter().enumerate() {
                if let Some(value) = local.as_number() {
                    numeric_locals[index] = value;
                } else if !matches!(local, Value::Null)
                    || index < declared_argument_count(program)
                    || !local_is_definitely_initialized_before_load(program, index)
                {
                    return None;
                }
            }
            frame.set_numeric_jit_state(trace.initial_state(&numeric_locals));
        }
        let budget = u32::try_from(remaining_steps).unwrap_or(u32::MAX);
        trace.run_budgeted(frame.numeric_jit_state_mut()?, budget)
    })
}

fn local_is_definitely_initialized_before_load(program: &Program, local: usize) -> bool {
    let Some(first_load) = program.instructions.iter().position(
        |instruction| matches!(instruction, Instruction::LoadLocal(slot) if usize::from(*slot) == local),
    ) else {
        return true;
    };
    let Some(first_store) = program.instructions[..first_load].iter().position(
        |instruction| matches!(instruction, Instruction::StoreLocal(slot) if usize::from(*slot) == local),
    ) else {
        return false;
    };
    // No edge originating before the initializer may skip over it.
    !program.instructions[..=first_store]
        .iter()
        .any(|instruction| {
            matches!(instruction,
            Instruction::Jump(target) | Instruction::JumpIfFalse(target) if *target > first_store)
        })
}

pub(crate) fn numeric_trace_instructions(program: &Program) -> Option<Vec<NumericInstruction>> {
    if program.instructions.is_empty()
        || program.instructions.iter().any(|instruction| {
            matches!(
                instruction,
                Instruction::MakeArgs | Instruction::AddressLocal(_)
            )
        })
    {
        return None;
    }
    let declared_arguments = declared_argument_count(program);
    program
        .instructions
        .iter()
        .map(|instruction| match instruction {
            Instruction::PushNumber(number) => Some(NumericInstruction::Constant(number.to_f32())),
            Instruction::LoadLocal(slot) => Some(NumericInstruction::LoadLocal(*slot)),
            // Writing a declared argument is observable through the live args
            // vector even when MakeArgs does not occur in this procedure. Keep
            // those procedures in the reference interpreter.
            Instruction::StoreLocal(slot) if usize::from(*slot) >= declared_arguments => {
                Some(NumericInstruction::StoreLocal(*slot))
            }
            Instruction::Add => Some(NumericInstruction::Add),
            Instruction::Subtract => Some(NumericInstruction::Subtract),
            Instruction::Multiply => Some(NumericInstruction::Multiply),
            Instruction::Divide => Some(NumericInstruction::Divide),
            Instruction::Negate => Some(NumericInstruction::Negate),
            Instruction::Equal => Some(NumericInstruction::Equal),
            Instruction::NotEqual => Some(NumericInstruction::NotEqual),
            Instruction::Less => Some(NumericInstruction::LessThan),
            Instruction::LessEqual => Some(NumericInstruction::LessThanOrEqual),
            Instruction::Greater => Some(NumericInstruction::GreaterThan),
            Instruction::GreaterEqual => Some(NumericInstruction::GreaterThanOrEqual),
            Instruction::Jump(target) => u32::try_from(*target).ok().map(NumericInstruction::Jump),
            Instruction::JumpIfFalse(target) => u32::try_from(*target)
                .ok()
                .map(NumericInstruction::JumpIfFalse),
            Instruction::Return => Some(NumericInstruction::Return),
            _ => None,
        })
        .collect()
}
