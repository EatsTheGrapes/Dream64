//! Trace-compiled fast-path JITs: camera chunks, RegisterSignal, rooted-list
//! loops, lumcount loops, and dmm-discovery/digest verification.

use crate::builtins::execute_standard_builtin;
use crate::bytecode::{
    CompoundAssignmentOperator, Instruction, Module, ProcedureId, Program, TypePredicateKind,
};
use crate::compact_wordcode;
use crate::value_ops::{
    assign_datum_or_shared_field, canonicalize_owned_value, canonicalize_value,
    datum_field_or_initial, datum_field_or_shared, dm_list_length_number,
    dynamic_call_target_named_at_callsite, logical_or_empty_list_field,
    logical_or_empty_list_index, pop, read_list_value, runtime_truthy, stringify_dm_value,
    value_to_list_index, write_list_value,
};
use crate::{CallFrame, ExecutionState, declared_argument_count, frame_context};
use dm_jit::{
    CompiledNumericTrace, CompiledRootedBlock, NumericInstruction, NumericRunOutcome,
    RegionCallbacks, RootedBlockOutcome, compile_numeric_field_trace,
    compile_numeric_field_trace_at, compile_safe_rooted_block,
};
use dm_value::{DatumId, FieldName, ListId, TypePath, Value, ValueError};
use smallvec::SmallVec;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

// The trace fast paths canonicalize type2parent-style shapes and read the dmm
// digest metrics from the tgm/ruin cluster.
use super::tgm_ruin::{NATIVE_DISCOVER_OFFSET_ACTIVATIONS, canonical_static_native_builtin};

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
    0x8f, 0x79, 0x53, 0x8a, 0x78, 0x5f, 0xed, 0xea, 0x5f, 0xcb, 0x51, 0x37, 0xd1, 0xf8, 0x8c, 0xb9,
    0xa4, 0x1e, 0xbf, 0x42, 0x68, 0x70, 0xa8, 0x8d, 0xa5, 0xbe, 0x5f, 0x3f, 0x58, 0x99, 0x43, 0x70,
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

/// Diagnostic counters for the Cranelift whole-procedure numeric JIT
/// (`try_run_guarded_jit`). `COMPILED`/`REJECTED` count *distinct procedures*
/// the first time each is considered; `RUNS`/`STEPS` count trace invocations
/// and the reference instruction budget they retired. On a Monkestation boot
/// these answer whether the native JIT contributes anything at all.
pub(crate) static GUARDED_JIT_NUMERIC_COMPILED: AtomicU64 = AtomicU64::new(0);
pub(crate) static GUARDED_JIT_NUMERIC_REJECTED: AtomicU64 = AtomicU64::new(0);
pub(crate) static GUARDED_JIT_LUMCOUNT_COMPILED: AtomicU64 = AtomicU64::new(0);
pub(crate) static GUARDED_JIT_LUMCOUNT_REJECTED: AtomicU64 = AtomicU64::new(0);
pub(crate) static GUARDED_JIT_RUNS: AtomicU64 = AtomicU64::new(0);
pub(crate) static GUARDED_JIT_STEPS: AtomicU64 = AtomicU64::new(0);

/// `(numeric_compiled, numeric_rejected, lumcount_compiled, lumcount_rejected,
/// runs, steps)` for the Cranelift whole-procedure JIT over the whole run.
#[must_use]
pub fn guarded_jit_telemetry() -> (u64, u64, u64, u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        GUARDED_JIT_NUMERIC_COMPILED.load(Relaxed),
        GUARDED_JIT_NUMERIC_REJECTED.load(Relaxed),
        GUARDED_JIT_LUMCOUNT_COMPILED.load(Relaxed),
        GUARDED_JIT_LUMCOUNT_REJECTED.load(Relaxed),
        GUARDED_JIT_RUNS.load(Relaxed),
        GUARDED_JIT_STEPS.load(Relaxed),
    )
}

thread_local! {
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
    // Keep the runtime kill switch authoritative for every native tier.
    if jit_disabled() {
        return None;
    }
    // Lumcount is an exact 48-instruction/four-local trace. The generic
    // numeric core (constants/locals/arithmetic/comparisons/branches) runs
    // through the region tier instead — see `region_at` in the sidecar
    // and `try_run_region_numeric_jit` below.
    if program.instructions.len() == 48
        && program.local_count == 4
        && let Some(outcome) =
            try_run_lumcount_jit(module, procedure, program, frame, remaining_steps, state)
    {
        GUARDED_JIT_RUNS.fetch_add(1, Ordering::Relaxed);
        let steps = match outcome {
            NumericRunOutcome::Returned { steps, .. }
            | NumericRunOutcome::BudgetExhausted { steps, .. }
            | NumericRunOutcome::SideExit { steps, .. } => u64::from(steps),
        };
        GUARDED_JIT_STEPS.fetch_add(steps, Ordering::Relaxed);
        return Some((outcome, true));
    }
    None
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
        let trace = cache.entry(key).or_insert_with(|| {
            let compiled = compile_lumcount_trace(program);
            if compiled.is_some() {
                GUARDED_JIT_LUMCOUNT_COMPILED.fetch_add(1, Ordering::Relaxed);
            } else {
                GUARDED_JIT_LUMCOUNT_REJECTED.fetch_add(1, Ordering::Relaxed);
            }
            compiled
        });
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
        let outcome = trace.compiled.run_budgeted(
            frame.numeric_jit_state_mut()?,
            budget,
            &mut NoDynamicFieldAccess,
        )?;
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
    let compiled = compile_numeric_field_trace(&native, program.local_count, 4, 0, 0)
        .inspect_err(|error| eprintln!("lumcount JIT compile rejected: {error}"))
        .ok()?;
    Some(LumcountTrace {
        compiled,
        fields: [lum_r, lum_g, lum_b, needs_update],
        lighting_global,
        queue_field,
    })
}

thread_local! {
    /// Cache for [`try_run_corner_apply_loop_jit`]'s canonical shape match,
    /// keyed one level deeper than the whole-procedure caches above: this
    /// trace matches a *loop embedded inside* a much larger procedure
    /// (`/datum/light_source/proc/update_corners`), not a whole small one,
    /// so the loop's own entry `pc` is part of the cache key alongside
    /// `(module identity, procedure)`.
    static CORNER_APPLY_JIT_CACHE: RefCell<HashMap<(u64, ProcedureId, usize), Option<CornerApplyBody>>> =
        RefCell::new(HashMap::new());
}

/// One hand-matched `for (var/datum/lighting_corner/corner as anything in
/// new_corners)` loop body — `update_corners`' own `APPLY_CORNER`+`LAZYADD`
/// inner loop, the single largest measured contributor within that
/// procedure's own ~10% of a real Monkestation boot (per
/// `DREAM64_PROFILE_PROCEDURE_PCS` diagnostics). A bespoke trace alongside
/// [`LumcountTrace`]/camera-chunk/RegisterSignal, but unlike those, this
/// isn't a whole procedure: it starts at a `NextLocalListIteration` found
/// partway through a much larger compiled artifact, so every jump target is
/// validated relative to that PC (`entry_pc`) rather than as an absolute
/// literal — the shape must keep matching regardless of where a future
/// recompile happens to place it.
pub(crate) struct CornerApplyBody {
    list_slot: u16,
    index_slot: u16,
    item_slot: u16,
    exit_pc: usize,
    corner_x: FieldName,
    corner_y: FieldName,
    light_outer_range: FieldName,
    light_falloff_curve: FieldName,
    effect_str: FieldName,
    affecting: FieldName,
    turf_x_local: u16,
    turf_y_local: u16,
    range_divisor_local: u16,
    light_power_local: u16,
    applied_lum_r_local: u16,
    applied_lum_g_local: u16,
    applied_lum_b_local: u16,
    lum_r_local: u16,
    lum_g_local: u16,
    lum_b_local: u16,
}

/// Validates the exact bytecode shape of `update_corners`' corner-apply
/// loop starting at `entry_pc` (a `NextLocalListIteration`), or declines.
/// `NextLocalListIteration`'s own fused runtime semantics
/// (`execution/interpreter.rs`) already skip straight from `entry_pc` to
/// `entry_pc + 7` on every real dispatch — the six instructions in between
/// are dead code — but they're still matched here (as inert padding) so a
/// coincidental, unrelated `NextLocalListIteration` elsewhere, followed by
/// different real code, can never be mistaken for this loop.
// `float_cmp`: every comparison here checks for an exact bytecode-level
// constant the compiler either emitted or didn't (e.g. "is this literally
// `PushNumber(2.0)`"), never an approximate runtime computation — the same
// exact-bit-pattern check `compile_lumcount_trace`'s own canonical match
// above already relies on. `similar_names`: the parallel `_r`/`_g`/`_b`
// locals mirror the DM source's own `_lum_r`/`_lum_g`/`_lum_b` naming
// exactly; renaming them to be less similar would only make this harder to
// audit against that source, not safer.
#[allow(clippy::too_many_lines, clippy::float_cmp, clippy::similar_names)]
pub(crate) fn compile_corner_apply_body(
    program: &Program,
    entry_pc: usize,
) -> Option<CornerApplyBody> {
    let instructions = &program.instructions;
    let at = |offset: usize| instructions.get(entry_pc.checked_add(offset)?);
    let local_at = |offset: usize| match at(offset)? {
        Instruction::LoadLocal(slot) => Some(*slot),
        _ => None,
    };
    let Instruction::NextLocalListIteration {
        list_slot,
        index_slot,
        item_slot,
        exit,
    } = at(0)?
    else {
        return None;
    };
    let (list_slot, index_slot, item_slot, exit_pc) = (*list_slot, *index_slot, *item_slot, *exit);
    if !matches!(at(1)?, Instruction::ListLengthLocal(slot) if *slot == list_slot)
        || !matches!(at(2)?, Instruction::LessEqual)
        || !matches!(at(3)?, Instruction::JumpIfFalse(target) if *target == exit_pc)
        || local_at(4)? != index_slot
        || !matches!(at(5)?, Instruction::IndexLocalList(slot) if *slot == list_slot)
        || !matches!(at(6)?, Instruction::StoreLocal(slot) if *slot == item_slot)
        || local_at(7)? != item_slot
    {
        return None;
    }
    let Instruction::LoadDeclaredField(corner_x) = at(8)? else {
        return None;
    };
    let turf_x_local = local_at(9)?;
    if !matches!(at(10)?, Instruction::Subtract)
        || !matches!(at(11)?, Instruction::PushNumber(n) if n.to_f32() == 2.0)
        || !matches!(at(12)?, Instruction::Power)
        || local_at(13)? != item_slot
    {
        return None;
    }
    let Instruction::LoadDeclaredField(corner_y) = at(14)? else {
        return None;
    };
    let turf_y_local = local_at(15)?;
    if !matches!(at(16)?, Instruction::Subtract)
        || !matches!(at(17)?, Instruction::PushNumber(n) if n.to_f32() == 2.0)
        || !matches!(at(18)?, Instruction::Power)
        || !matches!(at(19)?, Instruction::Add)
        || !matches!(at(20)?, Instruction::PushNumber(n) if n.to_f32() == 0.5)
        || !matches!(at(21)?, Instruction::Power)
        || !matches!(at(22)?, Instruction::LoadSrc)
    {
        return None;
    }
    let Instruction::LoadField(light_outer_range) = at(23)? else {
        return None;
    };
    if !matches!(at(24)?, Instruction::Subtract) {
        return None;
    }
    let range_divisor_local = local_at(25)?;
    if !matches!(at(26)?, Instruction::Divide)
        || !matches!(at(27)?, Instruction::Negate)
        || !matches!(at(28)?, Instruction::PushNumber(n) if n.to_f32() == 0.0)
        || !matches!(at(29)?, Instruction::PushNumber(n) if n.to_f32() == 1.0)
        || !matches!(
            at(30)?,
            Instruction::StandardBuiltin { name, argument_count: 3, .. } if name == "clamp"
        )
        || !matches!(at(31)?, Instruction::LoadSrc)
    {
        return None;
    }
    let Instruction::LoadField(light_falloff_curve) = at(32)? else {
        return None;
    };
    if !matches!(at(33)?, Instruction::Power)
        || !matches!(at(34)?, Instruction::StoreResult)
        || !matches!(at(35)?, Instruction::LoadResult)
    {
        return None;
    }
    let light_power_local = local_at(36)?;
    if !matches!(at(37)?, Instruction::PushNumber(n) if n.to_f32() == 2.0)
        || !matches!(at(38)?, Instruction::Power)
        || !matches!(
            at(39)?,
            Instruction::CompoundAssignment(CompoundAssignmentOperator::Multiply)
        )
        || !matches!(at(40)?, Instruction::StoreResult)
        || !matches!(at(41)?, Instruction::LoadResult)
        || local_at(42)? != light_power_local
        || !matches!(at(43)?, Instruction::PushNumber(n) if n.to_f32() == 0.0)
        || !matches!(at(44)?, Instruction::Less)
        || !matches!(at(45)?, Instruction::JumpIfFalse(target) if *target == entry_pc + 49)
        || !matches!(at(46)?, Instruction::PushNumber(n) if n.to_f32() == 1.0)
        || !matches!(at(47)?, Instruction::Negate)
        || !matches!(at(48)?, Instruction::Jump(target) if *target == entry_pc + 50)
        || !matches!(at(49)?, Instruction::PushNumber(n) if n.to_f32() == 1.0)
        || !matches!(
            at(50)?,
            Instruction::CompoundAssignment(CompoundAssignmentOperator::Multiply)
        )
        || !matches!(at(51)?, Instruction::StoreResult)
        || !matches!(at(52)?, Instruction::LoadSrc)
    {
        return None;
    }
    let Instruction::LoadField(effect_str) = at(53)? else {
        return None;
    };
    if local_at(54)? != item_slot || !matches!(at(55)?, Instruction::IndexList) {
        return None;
    }
    let old_local = match at(56)? {
        Instruction::StoreLocal(slot) => *slot,
        _ => return None,
    };
    if local_at(57)? != item_slot || !matches!(at(58)?, Instruction::LoadResult) {
        return None;
    }
    let lum_r_local = local_at(59)?;
    if !matches!(at(60)?, Instruction::Multiply) || local_at(61)? != old_local {
        return None;
    }
    let applied_lum_r_local = local_at(62)?;
    if !matches!(at(63)?, Instruction::Multiply)
        || !matches!(at(64)?, Instruction::Subtract)
        || !matches!(at(65)?, Instruction::LoadResult)
    {
        return None;
    }
    let lum_g_local = local_at(66)?;
    if !matches!(at(67)?, Instruction::Multiply) || local_at(68)? != old_local {
        return None;
    }
    let applied_lum_g_local = local_at(69)?;
    if !matches!(at(70)?, Instruction::Multiply)
        || !matches!(at(71)?, Instruction::Subtract)
        || !matches!(at(72)?, Instruction::LoadResult)
    {
        return None;
    }
    let lum_b_local = local_at(73)?;
    if !matches!(at(74)?, Instruction::Multiply) || local_at(75)? != old_local {
        return None;
    }
    let applied_lum_b_local = local_at(76)?;
    if !matches!(at(77)?, Instruction::Multiply) || !matches!(at(78)?, Instruction::Subtract) {
        return None;
    }
    if !matches!(
        at(79)?,
        Instruction::CallDynamic {
            static_selector: Some(selector),
            argument_count: 3,
            null_receiver_is_global: false,
            ..
        } if selector == "update_lumcount"
    ) {
        return None;
    }
    if !matches!(at(80)?, Instruction::Pop)
        || !matches!(at(81)?, Instruction::LoadResult)
        || !matches!(at(82)?, Instruction::PushNumber(n) if n.to_f32() == 0.0)
        || !matches!(at(83)?, Instruction::NotEqual)
        || !matches!(at(84)?, Instruction::JumpIfFalse(target) if *target == entry_pc + 104)
        || local_at(85)? != item_slot
    {
        return None;
    }
    let Instruction::LoadDeclaredField(affecting) = at(86)? else {
        return None;
    };
    if !matches!(at(87)?, Instruction::Not)
        || !matches!(at(88)?, Instruction::JumpIfFalse(target) if *target == entry_pc + 92)
        || local_at(89)? != item_slot
        || !matches!(at(90)?, Instruction::MakeListEntries(entries) if entries.is_empty())
        || !matches!(at(91)?, Instruction::StoreField(name) if name == affecting)
        || local_at(92)? != item_slot
        || !matches!(at(93)?, Instruction::Duplicate)
        || !matches!(at(94)?, Instruction::LoadField(name) if name == affecting)
        || !matches!(at(95)?, Instruction::LoadSrc)
        || !matches!(
            at(96)?,
            Instruction::CompoundAssignment(CompoundAssignmentOperator::Add)
        )
        || !matches!(at(97)?, Instruction::StoreField(name) if name == affecting)
        || !matches!(at(98)?, Instruction::LoadResult)
        || !matches!(at(99)?, Instruction::LoadSrc)
    {
        return None;
    }
    let Instruction::LoadField(effect_str_write) = at(100)? else {
        return None;
    };
    if effect_str_write != effect_str || local_at(101)? != item_slot {
        return None;
    }
    if !matches!(at(102)?, Instruction::PrepareRhsFirstIndexAssignment)
        || !matches!(at(103)?, Instruction::SetListIndex)
        || local_at(104)? != index_slot
        || !matches!(at(105)?, Instruction::PushNumber(n) if n.to_f32() == 1.0)
        || !matches!(at(106)?, Instruction::Add)
        || !matches!(at(107)?, Instruction::StoreLocal(slot) if *slot == index_slot)
        || !matches!(at(108)?, Instruction::Jump(target) if *target == entry_pc)
    {
        return None;
    }

    Some(CornerApplyBody {
        list_slot,
        index_slot,
        item_slot,
        exit_pc,
        corner_x: corner_x.clone(),
        corner_y: corner_y.clone(),
        light_outer_range: light_outer_range.clone(),
        light_falloff_curve: light_falloff_curve.clone(),
        effect_str: effect_str.clone(),
        affecting: affecting.clone(),
        turf_x_local,
        turf_y_local,
        range_divisor_local,
        light_power_local,
        applied_lum_r_local,
        applied_lum_g_local,
        applied_lum_b_local,
        lum_r_local,
        lum_g_local,
        lum_b_local,
    })
}

/// Drives a validated [`CornerApplyBody`] for as many corners as the
/// remaining step budget allows, replacing `update_corners`' own
/// `APPLY_CORNER`+`LAZYADD` loop iteration-for-iteration with native
/// computation — reusing the *already-compiled* [`LumcountTrace`] for
/// `update_lumcount` (`LUMCOUNT_JIT_CACHE`, shared with the ordinary
/// interpreted call path — a corner whose runtime type overrides
/// `update_lumcount` with something non-canonical simply never populates
/// that cache with a `Some`, so this declines for that one corner exactly
/// as it would for any other guard failure) rather than constructing a real
/// nested call. Declines (returning `None`) the instant anything doesn't
/// match what was validated, leaving `frame.instruction` untouched so the
/// ordinary interpreter safely redoes the *entire* current call from
/// scratch — every write this function makes happens only after its last
/// possible decline point (the `update_lumcount` resolution), exactly
/// mirroring the region tier's own side-exit discipline.
#[allow(clippy::too_many_lines)]
pub(crate) fn try_run_corner_apply_loop_jit(
    module: &Module,
    procedure: ProcedureId,
    program: &Program,
    frame: &mut CallFrame,
    remaining_steps: u64,
    state: &mut ExecutionState,
) -> Option<u64> {
    if jit_disabled() {
        return None;
    }
    let entry_pc = frame.instruction;
    if !matches!(
        program.instructions.get(entry_pc),
        Some(Instruction::NextLocalListIteration { .. })
    ) {
        return None;
    }
    let Value::Datum(src) = frame.src else {
        return None;
    };
    let caller_context = frame_context(frame);
    let key = (module.identity.0, procedure, entry_pc);
    CORNER_APPLY_JIT_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let body = cache
            .entry(key)
            .or_insert_with(|| compile_corner_apply_body(program, entry_pc));
        let body = body.as_ref()?;
        let mut steps = 0_u64;
        loop {
            // A whole iteration's worst case is well under 128 steps; leave
            // that much headroom rather than risk overrunning the budget.
            // Decline outright (rather than report zero progress) when even
            // the first iteration doesn't fit: `remaining_steps` is a
            // whole-call budget that legitimately runs this low near a
            // slice boundary, and `run.rs`'s caller re-enters unconditionally
            // on `Some(_)` — returning `Some(0)` here would spin forever
            // re-checking the same unmet budget instead of falling through
            // to the interpreter, which correctly drains the last steps.
            if steps + 128 > remaining_steps {
                if steps == 0 {
                    return None;
                }
                frame.instruction = entry_pc;
                return Some(steps);
            }
            // Replicates `Instruction::NextLocalListIteration`'s own fused
            // has-next-and-fetch semantics (`execution/interpreter.rs`)
            // exactly, since the dead bytecode padding this trace matched
            // is never actually dispatched at runtime.
            let Some(Value::List(list)) = frame.locals.get(usize::from(body.list_slot)).cloned()
            else {
                return None;
            };
            let Some(Value::Number(index)) =
                frame.locals.get(usize::from(body.index_slot)).cloned()
            else {
                return None;
            };
            let Ok(values) = state.heap.list(list) else {
                return None;
            };
            let length = values.len();
            if index.to_f32() > dm_list_length_number(length) {
                steps += 1;
                frame.instruction = body.exit_pc;
                return Some(steps);
            }
            let Ok(positional_index) = value_to_list_index(&Value::Number(index)) else {
                return None;
            };
            let Ok(item) = values.get(positional_index).cloned() else {
                return None;
            };
            let item = canonicalize_owned_value(&state.heap, item);
            let Value::Datum(corner) = item else {
                return None;
            };
            if let Some(Value::List(reference)) = frame.locals.get(usize::from(body.item_slot))
                && state.reference_lists.contains(reference)
            {
                // The rare reference-list aliasing path `NextLocalListIteration`
                // itself handles — decline rather than duplicate that here.
                return None;
            }
            frame.locals[usize::from(body.item_slot)] = Value::Datum(corner);
            steps += 1;

            let corner_x = datum_field_or_shared(state, corner, &body.corner_x)
                .ok()?
                .as_number()?;
            let corner_y = datum_field_or_shared(state, corner, &body.corner_y)
                .ok()?
                .as_number()?;
            let turf_x = frame
                .locals
                .get(usize::from(body.turf_x_local))?
                .as_number()?;
            let turf_y = frame
                .locals
                .get(usize::from(body.turf_y_local))?
                .as_number()?;
            let light_outer_range = datum_field_or_shared(state, src, &body.light_outer_range)
                .ok()?
                .as_number()?;
            let range_divisor = frame
                .locals
                .get(usize::from(body.range_divisor_local))?
                .as_number()?;
            let light_falloff_curve = datum_field_or_shared(state, src, &body.light_falloff_curve)
                .ok()?
                .as_number()?;
            let light_power = frame
                .locals
                .get(usize::from(body.light_power_local))?
                .as_number()?;
            steps += 38;

            // LUM_FALLOFF(corner) * (_light_power ** 2) * sign(_light_power)
            let distance = ((corner_x - turf_x).powi(2) + (corner_y - turf_y).powi(2)).powf(0.5);
            let falloff = (-((distance - light_outer_range) / range_divisor))
                .clamp(0.0, 1.0)
                .powf(light_falloff_curve);
            let mut delta = falloff * light_power.powi(2);
            if light_power < 0.0 {
                delta = -delta;
                steps += 4;
            } else {
                steps += 2;
            }

            // var/OLD = effect_str[corner]
            let Value::List(effect_str_list) =
                datum_field_or_shared(state, src, &body.effect_str).ok()?
            else {
                return None;
            };
            let old =
                match read_list_value(&state.heap, effect_str_list, &Value::Datum(corner), false) {
                    Ok(Value::Number(n)) => n.to_f32(),
                    Ok(Value::Null) | Err(ValueError::MissingKey) => 0.0,
                    _ => return None,
                };
            let lum_r = frame
                .locals
                .get(usize::from(body.lum_r_local))?
                .as_number()?;
            let applied_lum_r = frame
                .locals
                .get(usize::from(body.applied_lum_r_local))?
                .as_number()?;
            let lum_g = frame
                .locals
                .get(usize::from(body.lum_g_local))?
                .as_number()?;
            let applied_lum_g = frame
                .locals
                .get(usize::from(body.applied_lum_g_local))?
                .as_number()?;
            let lum_b = frame
                .locals
                .get(usize::from(body.lum_b_local))?
                .as_number()?;
            let applied_lum_b = frame
                .locals
                .get(usize::from(body.applied_lum_b_local))?
                .as_number()?;
            let arg_r = delta.mul_add(lum_r, -(old * applied_lum_r));
            let arg_g = delta.mul_add(lum_g, -(old * applied_lum_g));
            let arg_b = delta.mul_add(lum_b, -(old * applied_lum_b));
            steps += 35;

            // corner.update_lumcount(arg_r, arg_g, arg_b) — resolved through
            // the exact same callsite cache the interpreter itself would
            // populate/hit for this bytecode position, then driven through
            // the shared `LumcountTrace` cache rather than a real call.
            let Ok((target, _context)) = dynamic_call_target_named_at_callsite(
                module,
                state,
                &Value::Datum(corner),
                "update_lumcount",
                &caller_context,
                false,
                Some((procedure, entry_pc + 79)),
            ) else {
                return None;
            };
            let lum_result = LUMCOUNT_JIT_CACHE.with(|lumcount_cache| {
                let mut lumcount_cache = lumcount_cache.borrow_mut();
                let trace = lumcount_cache
                    .entry((module.identity.0, target))
                    .or_insert_with(|| {
                        let target_program = module.resolve_procedure(target).ok()?;
                        compile_lumcount_trace(target_program)
                    });
                let trace = trace.as_ref()?;
                if arg_r == 0.0 && arg_g == 0.0 && arg_b == 0.0 {
                    return Some(0.0);
                }
                let field_values: SmallVec<[f32; 8]> = trace
                    .fields
                    .iter()
                    .map(|field| {
                        datum_field_or_initial(state, corner, field)
                            .ok()?
                            .as_number()
                    })
                    .collect::<Option<_>>()?;
                let mut numeric_state = trace
                    .compiled
                    .initial_state_with_fields(&[arg_r, arg_g, arg_b, 0.0], &field_values)?;
                let outcome = trace.compiled.run_budgeted(
                    &mut numeric_state,
                    64,
                    &mut NoDynamicFieldAccess,
                )?;
                for (index, field) in trace.fields.iter().enumerate() {
                    if numeric_state.dirty_fields & (1_u64 << index) != 0 {
                        state
                            .heap
                            .set_datum_field(
                                corner,
                                field.clone(),
                                Value::number(numeric_state.fields[index]),
                            )
                            .ok()?;
                    }
                }
                if numeric_state.action_bits & 1 != 0 {
                    let Value::Datum(lighting) = state.global(&trace.lighting_global)?.clone()
                    else {
                        return None;
                    };
                    let Value::List(queue) =
                        datum_field_or_initial(state, lighting, &trace.queue_field).ok()?
                    else {
                        return None;
                    };
                    state.heap.list_mut(queue).ok()?.add(Value::Datum(corner));
                }
                let NumericRunOutcome::Returned { value, .. } = outcome else {
                    return None;
                };
                Some(value)
            });
            let lum_result = lum_result?;
            steps += 4;

            if lum_result != 0.0 {
                steps += 4;
                let affecting_value = datum_field_or_shared(state, corner, &body.affecting).ok()?;
                let affecting_list = if runtime_truthy(&state.heap, &affecting_value).ok()? {
                    let Value::List(list) = affecting_value else {
                        return None;
                    };
                    list
                } else {
                    let list = state.heap.allocate_list();
                    state
                        .heap
                        .set_datum_field(corner, body.affecting.clone(), Value::List(list))
                        .ok()?;
                    steps += 3;
                    list
                };
                state
                    .heap
                    .list_mut(affecting_list)
                    .ok()?
                    .add(Value::Datum(src));
                let effect_str_is_associative = state.is_associative_list(effect_str_list);
                write_list_value(
                    &mut state.heap,
                    effect_str_list,
                    Value::Datum(corner),
                    Value::number(lum_result),
                    effect_str_is_associative,
                )
                .ok()?;
                steps += 12;
            }

            let next_index = index.to_f32() + 1.0;
            frame.locals[usize::from(body.index_slot)] = Value::number(next_index);
            steps += 5;
        }
    })
}

/// Translates and compiles `program` for the region tier's numeric core, or
/// declines. Called at most once per `(procedure, entry pc)` — from
/// `ProcedureSidecar::poll_region_at`, once that slot's own warm-up counter
/// crosses the threshold — so, unlike the pre-region design, this pays a
/// translation/compile attempt per *distinct* hot site (a procedure's own
/// entry, plus — since milestone 7 — any call-resume candidate site that
/// independently gets hot), not on every call.
/// A compiled region plus the field-name and global-name tables its
/// `*Dynamic` instructions index into. `dm-jit` only ever sees dense `u16`
/// indices — resolving one back to a `FieldName` is entirely a `dm-vm`
/// concern, so this wrapper (not `CompiledNumericTrace` alone) is what
/// `PcCache::Region` stores from Milestone 3 on.
pub(crate) struct CompiledRegion {
    pub(crate) trace: CompiledNumericTrace,
    pub(crate) field_names: Vec<FieldName>,
    pub(crate) global_names: Vec<FieldName>,
}

/// Compiles a region entering at `entry_pc` — 0 for a procedure's own entry
/// (the only case before milestone 7), or an arbitrary other reachable
/// instruction for one of milestone 7's call-resume candidates. See
/// `numeric_trace_instructions_at`.
pub(crate) fn compile_region_trace_at(
    module: &Module,
    program: &Program,
    entry_pc: usize,
) -> Option<CompiledRegion> {
    let compiled = numeric_trace_instructions_at(module, program, entry_pc).and_then(
        |(instructions, field_names, global_names, local_count)| {
            compile_numeric_field_trace_at(
                &instructions,
                local_count,
                0,
                field_names.len(),
                global_names.len(),
                entry_pc,
            )
            .ok()
            .map(|trace| CompiledRegion {
                trace,
                field_names,
                global_names,
            })
        },
    );
    if compiled.is_some() {
        GUARDED_JIT_NUMERIC_COMPILED.fetch_add(1, Ordering::Relaxed);
    } else {
        GUARDED_JIT_NUMERIC_REJECTED.fetch_add(1, Ordering::Relaxed);
    }
    compiled
}

/// `dm-jit`'s `RegionCallbacks` implementor for one region call: a receiver
/// (`src`, resolved once up front) plus the field/global name tables to
/// translate a `*Dynamic` instruction's dense index back into a `FieldName`,
/// and the live `&mut ExecutionState` to actually answer with. Trait methods
/// taking `&mut self` mean each callback gets its own fresh, non-overlapping
/// borrow of `state` in turn — native code never calls two of them at once,
/// so no `RefCell` is needed the way two independent closures would have.
struct RegionDispatch<'a> {
    state: &'a mut ExecutionState,
    src: Option<DatumId>,
    field_names: &'a [FieldName],
    global_names: &'a [FieldName],
}

impl RegionCallbacks for RegionDispatch<'_> {
    fn load_field(&mut self, field_index: u32) -> Option<f32> {
        let name = self.field_names.get(usize::try_from(field_index).ok()?)?;
        datum_field_or_shared(self.state, self.src?, name)
            .ok()?
            .as_number()
    }

    fn store_field(&mut self, field_index: u32, value: f32) -> bool {
        let Some(name) = self
            .field_names
            .get(usize::try_from(field_index).ok().unwrap_or(usize::MAX))
            .cloned()
        else {
            return false;
        };
        let Some(src) = self.src else { return false };
        assign_datum_or_shared_field(self.state, src, name, Value::number(value)).is_ok()
    }

    fn load_global(&mut self, global_index: u32) -> Option<f32> {
        let name = self.global_names.get(usize::try_from(global_index).ok()?)?;
        self.state.global(name)?.as_number()
    }

    fn store_global(&mut self, global_index: u32, value: f32) -> bool {
        let Some(name) = self
            .global_names
            .get(usize::try_from(global_index).ok().unwrap_or(usize::MAX))
            .cloned()
        else {
            return false;
        };
        self.state.set_global(name, Value::number(value));
        true
    }
}

/// A `RegionCallbacks` that declines everything, for compiled traces with no
/// `*Dynamic` instruction to answer — lumcount's bespoke trace guards its
/// fixed field set through the older flat pre-fetched array instead (see
/// `try_run_lumcount_jit`), so it never calls any of these.
struct NoDynamicFieldAccess;

impl RegionCallbacks for NoDynamicFieldAccess {
    fn load_field(&mut self, _field_index: u32) -> Option<f32> {
        None
    }
    fn store_field(&mut self, _field_index: u32, _value: f32) -> bool {
        false
    }
    fn load_global(&mut self, _global_index: u32) -> Option<f32> {
        None
    }
    fn store_global(&mut self, _global_index: u32, _value: f32) -> bool {
        false
    }
}

/// Drives a region installed by `compile_region_trace_at` for one call, exactly
/// as the pre-region whole-procedure numeric JIT drove its own cached trace:
/// guard every live local as a definite number (or a not-yet-read null),
/// then resume or start `frame.numeric_jit_state` and run the budgeted trace.
/// Declining here (a non-numeric-shaped local) is a per-*call* decision, not a
/// procedure-wide one — the installed region stays available for the next
/// call whose locals do qualify.
pub(crate) fn try_run_region_numeric_jit(
    region: &CompiledRegion,
    program: &Program,
    frame: &mut CallFrame,
    remaining_steps: u64,
    state: &mut ExecutionState,
    entry_pc: usize,
) -> Option<NumericRunOutcome> {
    if frame.numeric_jit_state().is_none() {
        // May exceed `program.local_count`: an inlined leaf call's own
        // (renumbered) locals live past the procedure's own declared ones,
        // and always get written by that inline's argument-binding
        // `StoreLocal`s before ever being read — 0.0 is a safe placeholder
        // for those slots until then, the same way an uninitialized-but-safe
        // real local already defaults to 0.0 below.
        let mut numeric_locals = vec![0.0; region.trace.local_count()];
        for (index, local) in frame.locals.iter().enumerate() {
            if let Some(value) = local.as_number() {
                numeric_locals[index] = value;
            } else if !matches!(local, Value::Null)
                || index < declared_argument_count(program)
                || !local_is_definitely_initialized_before_load(program, entry_pc, index)
            {
                return None;
            }
        }
        frame.set_numeric_jit_state(
            region
                .trace
                .initial_state_at(&numeric_locals, entry_pc as u32),
        );
    }
    let budget = u32::try_from(remaining_steps).unwrap_or(u32::MAX);
    // `src` is captured by value (a `DatumId` is `Copy`) before the
    // `numeric_jit_state_mut()` borrow below, exactly like `try_run_lumcount_jit`
    // reads it once up front rather than holding a live borrow of `frame`.
    let src = match frame.src {
        Value::Datum(id) => Some(id),
        _ => None,
    };
    let mut dispatch = RegionDispatch {
        state,
        src,
        field_names: &region.field_names,
        global_names: &region.global_names,
    };
    let outcome = region
        .trace
        .run_budgeted(frame.numeric_jit_state_mut()?, budget, &mut dispatch);
    if let Some(outcome) = &outcome {
        GUARDED_JIT_RUNS.fetch_add(1, Ordering::Relaxed);
        let steps = match outcome {
            NumericRunOutcome::Returned { steps, .. }
            | NumericRunOutcome::BudgetExhausted { steps, .. }
            | NumericRunOutcome::SideExit { steps, .. } => u64::from(*steps),
        };
        GUARDED_JIT_STEPS.fetch_add(steps, Ordering::Relaxed);
    }
    outcome
}

/// Whether `local` is provably written before it's ever read, from
/// `start_pc` onward — i.e., whether a still-`Null` `local` is safe to seed
/// as this milestone's 0.0 placeholder, since nothing between `start_pc`
/// and the first read can observe it before a real write overwrites it.
/// `start_pc` is 0 for a whole-procedure region's own entry; milestone 7's
/// mid-procedure resume points use their own real `start_pc`, bounding the
/// search to only the code that region actually covers — a store *before*
/// `start_pc` doesn't help this region, since it cold-starts fresh from
/// `frame.locals` at `start_pc`, not from the procedure's full history.
fn local_is_definitely_initialized_before_load(
    program: &Program,
    start_pc: usize,
    local: usize,
) -> bool {
    let Some(first_load) = program.instructions[start_pc..].iter().position(
        |instruction| matches!(instruction, Instruction::LoadLocal(slot) if usize::from(*slot) == local),
    ).map(|offset| offset + start_pc) else {
        return true;
    };
    let Some(first_store) = program.instructions[start_pc..first_load].iter().position(
        |instruction| matches!(instruction, Instruction::StoreLocal(slot) if usize::from(*slot) == local),
    ).map(|offset| offset + start_pc) else {
        return false;
    };
    // No edge originating between `start_pc` and the initializer may skip
    // over it. A target before `start_pc` can't happen here: the region
    // that installed this analysis already rejected any such jump outright
    // (see `numeric_trace_instructions_at`), so this trace was never
    // compiled in the first place if one existed.
    !program.instructions[start_pc..=first_store]
        .iter()
        .any(|instruction| {
            matches!(instruction,
            Instruction::Jump(target) | Instruction::JumpIfFalse(target) if *target > first_store)
        })
}

/// Milestone 7: whether `program`'s instruction at `call_pc` (a
/// `Call`/`CallCurrent`/`CallParent`/`AllocateCurrentDatum` that just
/// side-exited) has a safe resume point past its own result — a real
/// bytecode position provably reached *only* by falling through from a
/// single, already-understood instruction, with an empty operand stack, so
/// a fresh region can cold-start there exactly like PC 0 already does. This
/// is deliberately a narrow, local check rather than a general
/// whole-procedure operand-stack analysis (which would need an accurate
/// pop/push count for every `Instruction` variant to get right — real risk
/// for comparatively little gain over this): a call's own result lands on
/// `frame.stack` when it returns, so the position right after it has depth
/// 1 (just that result) — if the *next* instruction is one of the few
/// whose stack effect is simple and certain (`Pop` discards it,
/// `StoreResult` or `StoreLocal` both pop exactly one value and push
/// nothing), the position after *that* has depth 0, unconditionally.
/// Requiring the candidate to never be a jump target *anywhere* in the
/// procedure (checked across every jump-shaped instruction
/// `reachable_from_entry` also recognizes) is what makes "reached only by
/// falling through" airtight without inspecting anything else in the
/// procedure at all.
pub(crate) fn safe_call_resume_pc(program: &Program, call_pc: usize) -> Option<usize> {
    let consumer_pc = call_pc.checked_add(1)?;
    let resume_pc = call_pc.checked_add(2)?;
    if resume_pc >= program.instructions.len() {
        return None;
    }
    if !matches!(
        program.instructions.get(consumer_pc)?,
        Instruction::Pop | Instruction::StoreResult | Instruction::StoreLocal(_)
    ) {
        return None;
    }
    let is_jump_target_of = |instruction: &Instruction| -> bool {
        matches!(instruction,
            Instruction::Jump(target)
            | Instruction::JumpIfFalse(target)
            | Instruction::JumpIfNull(target)
            | Instruction::LoadStaticLocalOrJump { target, .. }
            | Instruction::JumpIfArgumentSupplied { target, .. }
                if *target == resume_pc)
    };
    if program.instructions.iter().any(is_jump_target_of) {
        return None;
    }
    Some(resume_pc)
}

/// Every instruction reachable from `entry_pc` (0 for a whole procedure, or
/// milestone 7's own non-zero entry points), by DM bytecode's own control-flow
/// edges (`Jump`/`JumpIfFalse`/`JumpIfNull`/`LoadStaticLocalOrJump`/
/// `JumpIfArgumentSupplied`/`Return`; every other instruction falls through).
/// A DM procedure whose every real path already returns still gets a
/// compiler-appended trailing `LoadResult; Return` for the implicit
/// fall-off-the-end case — dead code, but common enough (any exhaustive
/// `if`/`else` or `switch` produces it) that `numeric_trace_instructions`
/// tolerates an unsupported opcode there instead of rejecting the whole
/// procedure over code that can never execute.
fn reachable_from_entry(instructions: &[Instruction], entry_pc: usize) -> Vec<bool> {
    let mut reachable = vec![false; instructions.len()];
    let mut stack = vec![entry_pc];
    while let Some(pc) = stack.pop() {
        if reachable.get(pc).copied().unwrap_or(true) {
            continue;
        }
        reachable[pc] = true;
        match &instructions[pc] {
            Instruction::Jump(target) => stack.push(*target),
            Instruction::Return => {}
            Instruction::JumpIfFalse(target)
            | Instruction::JumpIfNull(target)
            | Instruction::LoadStaticLocalOrJump { target, .. }
            | Instruction::JumpIfArgumentSupplied { target, .. } => {
                stack.push(*target);
                stack.push(pc + 1);
            }
            _ => stack.push(pc + 1),
        }
    }
    reachable
}

/// Finds `name`'s index in a dense name table (the region's field-name table
/// for `LoadFieldDynamic`/`StoreFieldDynamic`, or its global-name table for
/// `LoadGlobalDynamic`/`StoreGlobalDynamic`), adding it if this is the first
/// reference. `*Dynamic` instructions carry this index, not a `FieldName` —
/// `dm-jit` never sees one.
fn resolve_name_index(names: &mut Vec<FieldName>, name: &FieldName) -> Option<u16> {
    let index = names
        .iter()
        .position(|existing| existing == name)
        .unwrap_or_else(|| {
            names.push(name.clone());
            names.len() - 1
        });
    u16::try_from(index).ok()
}

/// Translates one procedure's bytecode into the region tier's closed numeric
/// IR, alongside the distinct field and global names it references
/// dynamically (the `*Dynamic` tables `compile_region_trace_at` hands to
/// `dm-jit`).
///
/// `LoadSrc` always translates to `NumericInstruction::LoadSrc`, and every
/// `LoadField`/`StoreField`/`LoadGlobal`/`StoreGlobal` always translates to
/// its `*Dynamic` counterpart — this translator no longer proves the
/// receiver is `src` itself for fields (earlier versions used a
/// 1-instruction lookback, sound only for reads: a store's receiver sits
/// under an arbitrary-length value expression, not immediately below the
/// store), and globals have no receiver to prove anything about at all.
/// `dm-jit`'s `validate` carries that proof, via `StackKind` tracking — see
/// the "Milestone 3" module doc in `dm-jit/src/lib.rs`. A procedure using
/// `src` any other way (storing it in a local, testing it as a branch
/// condition, returning it directly) fails validation there and this whole
/// function's caller falls back to the interpreter, exactly as it always has
/// for any other unsupported shape.
/// A lowered trace: its instructions, the field/global name tables its
/// `*Dynamic` instructions index into, and the total local-slot count it
/// needs (which can exceed `program.local_count` — see milestone 6's
/// `try_inline_leaf_call`, whose spliced-in callee locals live past the
/// source procedure's own).
type NumericTraceLowering = (
    Vec<NumericInstruction>,
    Vec<FieldName>,
    Vec<FieldName>,
    usize,
);

// The per-instruction-kind dispatch is one large match by design — the same
// reasoning `run_frames_inner`'s own `#[allow(clippy::too_many_lines)]`
// gives: splitting it would only move each arm behind another call boundary
// without making any single arm simpler.
pub(crate) fn numeric_trace_instructions(
    module: &Module,
    program: &Program,
) -> Option<NumericTraceLowering> {
    numeric_trace_instructions_at(module, program, 0)
}

/// Milestone 7: as `numeric_trace_instructions`, but for a region entering
/// at an arbitrary reachable instruction instead of a procedure's own entry
/// (PC 0) — resuming native execution after a call, at a real-bytecode
/// position `safe_call_resume_pc` has proven has an empty operand stack and
/// is never a jump target from anywhere else in the procedure. Positions
/// before `entry_pc` are never examined at all — filled with an inert
/// placeholder instead, since `entry_pc`-seeded reachability (both here and
/// in `dm-jit`'s own `validate`, seeded identically via
/// `compile_numeric_field_trace_at`) never visits them, so the placeholder
/// is never actually reached by anything; it exists only to keep this
/// trace's positions numbered identically to the real bytecode's; every
/// side-exit's resume-PC packing, and every jump target, depends on that.
#[allow(clippy::too_many_lines)]
pub(crate) fn numeric_trace_instructions_at(
    module: &Module,
    program: &Program,
    entry_pc: usize,
) -> Option<NumericTraceLowering> {
    if program.instructions.is_empty()
        || entry_pc >= program.instructions.len()
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
    let reachable = reachable_from_entry(&program.instructions, entry_pc);
    let mut field_names: Vec<FieldName> = Vec::new();
    let mut global_names: Vec<FieldName> = Vec::new();
    let mut instructions = Vec::with_capacity(program.instructions.len());
    let mut local_count = program.local_count;
    // Milestone 5: a call/allocation ends the compiled prefix instead of
    // rejecting the whole procedure, but only when it's reached by a pure
    // straight line from entry — no branch anywhere before it. Bytecode
    // *array* order isn't execution order once a jump exists (an earlier
    // branch could skip the call entirely, or reach later code some other
    // way), so accepting one after a branch would risk silently discarding
    // a reachable path rather than just being conservative. `seen_branch`
    // keeps this truncation to the one shape it's actually proven for.
    let mut seen_branch = false;
    // Milestone 6: inlining a leaf call (see `try_inline_leaf_call`) needs
    // fresh local slots for the callee's own locals, renumbered past
    // whatever the caller and any earlier inline already used — which only
    // stays sound if nothing in the *caller* can jump into the middle of a
    // splice. `seen_branch` above only guards what's *before* a call site
    // (sufficient for M5's truncation, which never looks past that point
    // anyway); inlining instead continues translating everything after the
    // splice, so a branch *anywhere* in the caller — including after the
    // call — is disqualifying: dm-jit instruction positions are only valid
    // resume PCs for later side-exits as long as they stay 1:1 with real
    // bytecode positions, and a splice breaks that identity for everything
    // after it. Requiring the whole caller branch-free sidesteps rewriting
    // jump targets entirely, rather than risking getting that math wrong.
    // Only `[entry_pc..]` is *this* trace, so only that range's own
    // branch-freedom matters here — whatever precedes `entry_pc` (a prior
    // call site, its own branches, ...) belongs to a different trace, if
    // any, and has no bearing on this one's own soundness.
    let caller_is_branch_free = !program.instructions[entry_pc..].iter().any(|instruction| {
        matches!(
            instruction,
            Instruction::Jump(_) | Instruction::JumpIfFalse(_)
        )
    });
    for (pc, instruction) in program.instructions.iter().enumerate() {
        if pc < entry_pc {
            instructions.push(NumericInstruction::Return);
            continue;
        }
        let translated = match instruction {
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
            Instruction::Not => Some(NumericInstruction::Not),
            Instruction::And => Some(NumericInstruction::And),
            Instruction::Or => Some(NumericInstruction::Or),
            Instruction::Equal => Some(NumericInstruction::Equal),
            Instruction::NotEqual => Some(NumericInstruction::NotEqual),
            Instruction::Less => Some(NumericInstruction::LessThan),
            Instruction::LessEqual => Some(NumericInstruction::LessThanOrEqual),
            Instruction::Greater => Some(NumericInstruction::GreaterThan),
            Instruction::GreaterEqual => Some(NumericInstruction::GreaterThanOrEqual),
            Instruction::Jump(target) => {
                // A target before `entry_pc` would jump into a position
                // this trace never actually translated (the inert
                // placeholder above) — reject outright rather than silently
                // treating dead placeholder content as real code. Only
                // relevant for a non-zero `entry_pc`: a whole-procedure
                // trace (`entry_pc == 0`) can never see this, since nothing
                // exists before position 0.
                if *target < entry_pc {
                    return None;
                }
                seen_branch = true;
                u32::try_from(*target).ok().map(NumericInstruction::Jump)
            }
            Instruction::JumpIfFalse(target) => {
                if *target < entry_pc {
                    return None;
                }
                seen_branch = true;
                u32::try_from(*target)
                    .ok()
                    .map(NumericInstruction::JumpIfFalse)
            }
            Instruction::Return => Some(NumericInstruction::Return),
            Instruction::LoadSrc => Some(NumericInstruction::LoadSrc),
            Instruction::LoadField(name) => {
                resolve_name_index(&mut field_names, name).map(NumericInstruction::LoadFieldDynamic)
            }
            Instruction::StoreField(name) => resolve_name_index(&mut field_names, name)
                .map(NumericInstruction::StoreFieldDynamic),
            Instruction::LoadGlobal(name) => resolve_name_index(&mut global_names, name)
                .map(NumericInstruction::LoadGlobalDynamic),
            Instruction::StoreGlobal(name) => resolve_name_index(&mut global_names, name)
                .map(NumericInstruction::StoreGlobalDynamic),
            Instruction::Call {
                procedure,
                argument_count,
                ..
            } if !seen_branch && reachable[pc] => {
                if caller_is_branch_free
                    && let Some((inlined, locals_used)) =
                        try_inline_leaf_call(module, *procedure, *argument_count, local_count)
                {
                    local_count += locals_used;
                    instructions.extend(inlined);
                    continue;
                }
                instructions.push(NumericInstruction::CallSideExit {
                    argument_count: *argument_count,
                });
                break;
            }
            Instruction::CallCurrent { argument_count }
            | Instruction::CallParent { argument_count, .. }
                if !seen_branch && reachable[pc] =>
            {
                instructions.push(NumericInstruction::CallSideExit {
                    argument_count: argument_count.unwrap_or(0),
                });
                break;
            }
            Instruction::AllocateCurrentDatum { argument_count }
                if !seen_branch && reachable[pc] =>
            {
                instructions.push(NumericInstruction::CallSideExit {
                    argument_count: *argument_count,
                });
                break;
            }
            _ if !reachable[pc] => Some(NumericInstruction::Return),
            _ => None,
        };
        instructions.push(translated?);
    }
    Some((instructions, field_names, global_names, local_count))
}

/// Milestone 6: attempts to splice `procedure`'s own body directly into the
/// caller's trace in place of a `Call`, at compile time, instead of M5's
/// unconditional `CallSideExit`. Scoped narrowly, matching every other
/// milestone's own narrowing: the callee must translate to a *pure*,
/// branch-free arithmetic sequence (no field/global access, no calls or
/// allocations of its own — bounding this to exactly one level of inlining
/// by construction, since the recursive `numeric_trace_instructions` call
/// below can then never itself encounter a `Call` to attempt inlining
/// again) ending in exactly one `Return` as its last instruction, with its
/// declared parameter count exactly equal to its total local count (no
/// extra temp locals to reason about defaulting). The caller itself must
/// ALSO be entirely branch-free (checked by the caller of this function,
/// `caller_is_branch_free`) for the reason explained there.
///
/// A qualifying callee's `Return` is simply dropped: the value it would
/// have popped already sits on the native operand stack — since native
/// execution never actually "returns" anywhere, it only ever pushes and
/// pops the one shared stack — so leaving it there is exactly equivalent to
/// the call having returned it, for the caller's own following
/// instructions to keep consuming normally.
fn try_inline_leaf_call(
    module: &Module,
    procedure: ProcedureId,
    argument_count: u16,
    local_offset: usize,
) -> Option<(Vec<NumericInstruction>, usize)> {
    let callee_program = module.resolve_procedure(procedure).ok()?;
    // `local_count` almost always exceeds the declared parameter count — DM
    // reserves extra compiler-internal slots (the implicit `.` variable
    // among them) regardless of whether a given procedure body ever
    // touches them. The call site's own argument count only ever supplies
    // the *declared* parameters, so that's what has to match here, not the
    // total.
    if usize::from(argument_count) != declared_argument_count(callee_program) {
        return None;
    }
    // Every local beyond the declared parameters starts at this splice's
    // default (0.0, from `try_run_region_numeric_jit`'s entry seeding) —
    // safe only if the callee provably never reads one before writing it
    // first (the same check, and the same reasoning, `try_run_region_numeric_jit`
    // already applies to a region's own top-level entry locals). A local
    // read before any write — most plausibly DM's implicit `.` — would
    // otherwise silently read this splice's 0.0 instead of `.`'s real
    // default of `null`.
    if (declared_argument_count(callee_program)..callee_program.local_count)
        .any(|local| !local_is_definitely_initialized_before_load(callee_program, 0, local))
    {
        return None;
    }
    // A branch's jump target is an absolute index into the *callee's own*
    // instruction array, which this splice never rewrites (only
    // `LoadLocal`/`StoreLocal` indices get renumbered below) — so any
    // internal jump would land on the wrong instruction once spliced into a
    // new position in the caller's sequence. A call/allocation of its own
    // would need this same inlining logic recursively, with its own
    // local-offset bookkeeping layered on top of this splice's — not worth
    // the complexity for a first cut, so it's rejected outright rather than
    // attempted.
    if callee_program.instructions.iter().any(|instruction| {
        matches!(
            instruction,
            Instruction::Jump(_)
                | Instruction::JumpIfFalse(_)
                | Instruction::Call { .. }
                | Instruction::CallCurrent { .. }
                | Instruction::CallParent { .. }
                | Instruction::CallDynamic { .. }
                | Instruction::AllocateDatum { .. }
                | Instruction::AllocateCurrentDatum { .. }
        )
    }) {
        return None;
    }
    let (callee_instructions, callee_fields, callee_globals, _) =
        numeric_trace_instructions(module, callee_program)?;
    if !callee_fields.is_empty() || !callee_globals.is_empty() {
        return None;
    }
    // Dead code after an early `return` (valid, if unusual, DM source) can
    // still translate to more than one `Return`, or to a non-`Return`
    // instruction trailing the real one — either way, only a *single*
    // `Return` as the *last* instruction is safe to drop and splice.
    if callee_instructions
        .iter()
        .filter(|instruction| matches!(instruction, NumericInstruction::Return))
        .count()
        != 1
        || !matches!(callee_instructions.last(), Some(NumericInstruction::Return))
    {
        return None;
    }
    let local_offset = u16::try_from(local_offset).ok()?;
    let mut spliced = Vec::with_capacity(callee_instructions.len() + usize::from(argument_count));
    // Arguments already sit on the caller's native operand stack in push
    // order (bottom-to-top = first-to-last, per `CallSideExit`'s own
    // rematerialization in `run.rs`); bind them into the callee's own
    // (renumbered) parameter slots by popping in reverse, exactly like an
    // ordinary interpreted call would.
    for callee_local in (0..argument_count).rev() {
        spliced.push(NumericInstruction::StoreLocal(local_offset + callee_local));
    }
    for instruction in &callee_instructions[..callee_instructions.len() - 1] {
        spliced.push(renumber_local(*instruction, local_offset));
    }
    // The full local range this inline claims, including the unused-but-
    // reserved extras beyond the declared parameters — not just
    // `argument_count` — so a later inline in the same caller starts its
    // own renumbering past all of them, not just the bound ones.
    Some((spliced, callee_program.local_count))
}

fn renumber_local(instruction: NumericInstruction, offset: u16) -> NumericInstruction {
    match instruction {
        NumericInstruction::LoadLocal(local) => NumericInstruction::LoadLocal(local + offset),
        NumericInstruction::StoreLocal(local) => NumericInstruction::StoreLocal(local + offset),
        other => other,
    }
}
