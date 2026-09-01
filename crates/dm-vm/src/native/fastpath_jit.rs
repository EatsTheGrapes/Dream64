//! Trace-compiled fast-path JITs: camera chunks, RegisterSignal, rooted-list
//! loops, lumcount loops, and dmm-discovery/digest verification.

use crate::builtins::execute_standard_builtin;
use crate::bytecode::{
    CompoundAssignmentOperator, Instruction, Module, ProcedureId, Program, TypePredicateKind,
};
use crate::compact_wordcode;
use crate::value_ops::{
    assign_datum_or_shared_field, canonicalize_value, datum_field_or_initial,
    datum_field_or_shared, logical_or_empty_list_field, logical_or_empty_list_index, pop,
    read_list_value, runtime_truthy, stringify_dm_value, write_list_value,
};
use crate::{CallFrame, ExecutionState, declared_argument_count};
use dm_jit::{
    CompiledNumericTrace, CompiledRootedBlock, NumericInstruction, NumericRunOutcome,
    RootedBlockOutcome, compile_numeric_field_trace, compile_numeric_trace,
    compile_safe_rooted_block,
};
use dm_value::{DatumId, FieldName, ListId, TypePath, Value, ValueError};
use smallvec::SmallVec;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
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
    0xfe, 0xec, 0xe2, 0x68, 0x2b, 0x40, 0x28, 0xb3, 0xc5, 0x57, 0x93, 0x2f, 0x52, 0x48, 0xde, 0xfb,
    0x6c, 0x08, 0x13, 0x39, 0x32, 0xc9, 0x47, 0x05, 0x0e, 0xa9, 0xaf, 0xf3, 0x78, 0x6b, 0x8c, 0xce,
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
