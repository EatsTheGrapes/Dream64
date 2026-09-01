//! Monke .dmm TGM/ruin loading drives and canonical type2parent/istext
//! compaction, plus their instrumentation counters.

use crate::builtins;
use crate::builtins::is_subtype;
use crate::bytecode::{Instruction, Module, ProcedureId, Program};
use crate::compile::compile_procedure;
use crate::tgm_planner;
use crate::value_ops::{
    ExecutionContext, datum_field_or_initial, dynamic_call_target_named, get_step_builtin,
    is_area_type_path, is_turf_type_path, read_list_value, runtime_truthy, values_equal,
    write_list_value,
};
use crate::{
    CallFrame, ExecutionState, RuinCandidateScan, TgmLoadContinuation, TgmLoadPhase, frame_context,
    make_frame,
};
use dm_value::{FieldName, TypePath, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

// The tgm/ruin drives read the world clock through the numeric-core sibling.
use super::numeric_core::world_numeric_field;

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
    0x85, 0x4b, 0x62, 0x18, 0x63, 0x82, 0x20, 0x9f, 0x21, 0xc2, 0x6a, 0x32, 0xb0, 0x0b, 0x04, 0x49,
    0x36, 0x17, 0x87, 0xa8, 0xe8, 0x30, 0xd0, 0xc4, 0x59, 0x91, 0xf2, 0x1e, 0xd1, 0xdd, 0xac, 0x8a,
];
pub(crate) const CANONICAL_MONKE_BUILD_COORDINATE_DIGEST: [u8; 32] = [
    0xb6, 0x0a, 0xef, 0x32, 0xef, 0x56, 0x9c, 0xf2, 0x67, 0x4a, 0x81, 0x30, 0xb5, 0x7a, 0x7a, 0xd4,
    0xb4, 0x90, 0x87, 0x80, 0x0a, 0xf5, 0x82, 0xc5, 0x2d, 0x0a, 0xcd, 0x9f, 0x1d, 0x69, 0xf9, 0xc0,
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
pub(crate) static NATIVE_DISCOVER_OFFSET_ACTIVATIONS: AtomicU64 = AtomicU64::new(0);

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
    0xfa, 0x0c, 0x5c, 0x43, 0xc2, 0x93, 0xb7, 0x67, 0x3c, 0xe6, 0xce, 0xad, 0xf7, 0xed, 0xce, 0xdb,
    0xbd, 0x99, 0x33, 0x06, 0x1c, 0x5a, 0x2b, 0x59, 0x90, 0xa6, 0x0d, 0xba, 0xbb, 0x40, 0x74, 0x6a,
];
const CANONICAL_MONKE_GET_AFFECTED_TURFS_DIGEST: [u8; 32] = [
    0x4a, 0x1e, 0xdc, 0x92, 0xb4, 0xa2, 0x60, 0x70, 0xa9, 0x9d, 0x61, 0xea, 0xc7, 0x62, 0x2e, 0x8b,
    0x98, 0xf3, 0x3f, 0x84, 0xf8, 0x12, 0x9e, 0x77, 0x2e, 0x3d, 0xd5, 0xc7, 0xd6, 0x4d, 0xbc, 0x0d,
];

fn trusted_get_affected_turfs_target(module: &Module) -> bool {
    // Resolve by canonical path rather than a pinned numeric id: procedure
    // numbering shifts whenever the production project gains or loses any
    // procedure, which would otherwise silently disable the guarded ruin scan.
    let Some(procedure) = module
        .effective_procedure_id("/datum/map_template/proc/get_affected_turfs")
        .or_else(|| module.procedure_id("/datum/map_template/proc/get_affected_turfs"))
    else {
        return false;
    };
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
        // The exact call target (the `isarea` arm of the `get_area` macro) is
        // already fixed by the semantic digest below; only its structural shape
        // is asserted here so a renumbered target id does not disable the guard.
        && matches!(program.instructions.get(95), Some(Instruction::Call { argument_count: 1, .. }))
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

/// Emits a single boot-log line when the canonical `_tgm_load` is present but
/// its pinned semantic identity no longer matches. The digest depends on the
/// numbering of every procedure the loader calls, so it drifts whenever the
/// production project or the lowering pipeline changes even though the loader's
/// own DM body did not. Without this notice the only symptom is every map cell
/// silently falling back to the reference interpreter, which is far slower per
/// tick and can keep a heavy boot from ever reaching lobby pregame.
fn warn_tgm_guard_mismatch_once(module: &Module, procedure: ProcedureId) {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        eprintln!(
            "boot-vm: native TGM map loader disabled — /datum/parsed_map/proc/_tgm_load semantic digest {:02x?} does not match the pinned CANONICAL_MONKE_TGM_LOAD_DIGEST; re-pin crates/dm-vm/src/native/tgm_ruin.rs against the current project",
            module.procedure_semantic_digest(procedure),
        );
    });
}

const CANONICAL_MONKE_TGM_BUILD_CACHE_DIGEST: [u8; 32] = [
    0x8c, 0x40, 0xc0, 0x46, 0xba, 0x52, 0xa6, 0x53, 0x38, 0xf5, 0xcf, 0x6d, 0xa7, 0xb3, 0x79, 0x4e,
    0x54, 0x9a, 0xe9, 0xf3, 0xd2, 0x2e, 0xea, 0x76, 0xeb, 0x42, 0xd7, 0x78, 0xe9, 0xc3, 0x71, 0xad,
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
            warn_tgm_guard_mismatch_once(module, procedure);
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
