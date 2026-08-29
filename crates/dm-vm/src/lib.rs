//! Portable stack bytecode and the deterministic reference interpreter.

#![cfg_attr(not(test), deny(missing_docs))]

mod builtins;
mod bytecode;
mod compact_wordcode;
mod compile;
mod compile_expr;
mod compile_stmt;
mod execution;
mod local_client;
mod module_codec;
mod native;
mod profiling;
mod ready_snapshot;
pub mod tgm_planner;
mod value_ops;
pub mod worker_lane;

pub use bytecode::{
    CompoundAssignmentOperator, CompoundListIndexOperator, InitializerBinding,
    InitializerCompileContext, InitializerProgram, InstanceInitializer, Instruction, ListEntryKind,
    Module, ProcedureId, Program, TypePredicateKind, VerbParameterType,
};
pub use compact_wordcode::{CompactProcedureRecord, CompactWordcodeError, CompactWordcodeImage};
pub use compile::{
    append_initializer_program, compile_initializer, compile_initializer_into_module,
    compile_initializer_program, compile_module, compile_module_specs,
    compile_module_specs_selective, compile_module_specs_selective_with_errors,
    compile_module_specs_with_global_types, compile_module_with_global_fields, compile_procedure,
    initializer_compile_context,
};
pub use module_codec::ModuleCodecError;
pub use ready_snapshot::ReadyWorldCoreSnapshot;
pub use value_ops::ExecutionContext;

pub use local_client::{
    LocalClientAppearance, LocalClientError, LocalClientMapSnapshot, LocalClientMapTile,
    LocalClientPromptKind, LocalClientPromptResponse, LocalClientScreenAppearance,
    LocalClientState, LocalClientUiEvent, LocalMovementDirection, LocalScreenPointerEvent,
};

pub(crate) use local_client::{
    ExceptionHandler, PendingLocalPrompt, PendingPromptContinuation, PendingVerbInvocation,
    SavefileState, ScheduledSpawn, local_prompt_spec, queue_next_verb_prompt, register_prompt,
};

pub use execution::{
    ContinuationMetrics, DeclaredFieldQuickeningMetrics, ExecutionState, VmContinuationId,
    advance_scheduler,
};

// Interpreter-internal execution items still consulted by the crate root and
// its remaining run-loop support. Widened to crate reach so sibling modules can
// keep touching the moved implementation while it settles behind `execution`.
#[cfg(test)]
pub(crate) use execution::adaptive_heap_collection_growth;
#[cfg(test)]
pub(crate) use execution::make_frame_owned;
pub(crate) use execution::{
    CallFrame, CallFrameCold, OwnedContinuation, PackedNumericState, RuinCandidateScan,
    TgmLoadContinuation, TgmLoadPhase,
};
pub(crate) use execution::{
    FrameRunOutcome, StepBudgetBehavior, declared_argument_count, frame_context, make_frame,
    run_frames, schedule_frames, trace,
};
#[cfg(test)]
pub(crate) use execution::{
    MAXIMUM_HIGH_YIELD_COLLECTION_GROWTH, MAXIMUM_LOW_YIELD_COLLECTION_GROWTH,
    MAXIMUM_MODERATE_YIELD_COLLECTION_GROWTH, MINIMUM_HEAP_COLLECTION_GROWTH,
};

#[cfg(test)]
pub(crate) use value_ops::DYNAMIC_LOOKUP_PROBES;

// Compiler helpers the bytecode IR module calls back into for deferred
// procedure materialization. Kept pub(crate) and re-exported from the crate
// root so `bytecode` can reach them without becoming compile-coupled.
pub(crate) use compile::{compile_error, compile_procedure_with_resolver_and_fields};

pub(crate) use value_ops::{
    HeapReference, allocate_initialized_datum, allocate_matrix, allocate_vector,
    assign_datum_field, clone_icon_datum, compare_values, datum_field_or_initial,
    datum_shared_storage, deterministic_unit, dynamic_call_target_named,
    engine_builtin_initial_fields, engine_builtin_initial_value, engine_root_initial_value,
    execute_icon_method, get_step_builtin, initialize_existing_datum, instance_initializer_plan,
    is_area_type_path, is_atom_type_path, is_icon_datum, is_matrix_datum, is_turf_type_path,
    lazy_atom_list_field, matrix_components, matrix_product, parse_heap_reference, runtime_truthy,
    vector_components,
};

#[cfg(test)]
pub(crate) use value_ops::{
    allocate_or_replace_engine_datum, dm_direction_bits, dm_list_resize_length,
    dm_world_coordinate, dynamic_call_target, dynamic_call_target_named_at_callsite,
    engine_owner_field_names, engine_root_paths, indexed_text_character,
    initial_value_or_engine_root, pop_builtin_arguments, runtime_initial_field_value,
    values_equivalent,
};

pub(crate) use profiling::{
    AtomsProfile, AtomsProfileInstruction, AtomsProfileProcedure,
    STARTUP_INSTRUCTION_CATEGORY_COUNT, ShuttleTracePostReturn, TgmProfile, atoms_profile_enabled,
    atoms_profile_snapshot_lines_if_due, boot_dashboard_enabled, boot_trace_enabled,
    emit_atoms_profile, emit_tgm_profile, is_atoms_initialize_path, is_subsystem_initialize_path,
    mark_boot_trace_frame, shuttle_trace_emit_snapshot, shuttle_trace_enabled,
    shuttle_trace_prepare_call, shuttle_trace_slot_from_arguments, startup_instruction_category,
    startup_instruction_profile_enabled, startup_profile_enabled, tgm_profiling_enabled,
};

#[cfg(test)]
pub(crate) use profiling::{
    ATOMS_INITIALIZE_PATH, atoms_profile_lines, diagnostic_env_truthy,
    procedure_argument_trace_filter, shuttle_trace_is_atmos_init,
    shuttle_trace_is_late_shuttle_move, shuttle_trace_is_nullify_node,
};

pub use native::{
    native_build_coordinate_prefix_metrics, native_discover_offset_activations,
    native_ruin_area_rejection_samples, native_ruin_batch_metrics,
    native_ruin_rejection_cache_hits, native_ruin_rejection_causes, native_ruin_scan_metrics,
    native_tgm_build_cache_metrics, native_tgm_commit_samples, native_tgm_continuation_rejections,
    native_tgm_load_activations, native_tgm_load_metrics, native_tgm_route_samples,
    native_tgm_target_cache_metrics, packed_dispatch_counters,
};

pub(crate) use native::{
    TgmDrive, advance_headless_world_clock, canonical_istext, canonical_static_native_builtin,
    canonical_tgm_load_path, canonical_type2parent, canonical_type2parent_target,
    drive_ruin_candidate_scan, drive_tgm_load, execute_compact_fast_instruction,
    false_tick_check_target, numeric_dispatch_candidate, set_world_numeric_field, trace_tgm_route,
    try_run_build_coordinate_prefix, try_run_camera_chunk_fast_path,
    try_run_discover_offset_fast_path, try_run_dmm_preload_measurement_fast_path,
    try_run_guarded_jit, try_run_numeric_dispatch_block, try_run_numeric_local_update,
    try_run_numeric_loop_branch, try_run_parsed_dmm_new_fast_path,
    try_run_register_signal_fast_path, try_run_rooted_list_jit, try_run_ruin_affected_turfs_batch,
    try_run_tgm_build_cache_simple_member, world_numeric_field,
};

#[cfg(test)]
pub(crate) use native::{
    CANONICAL_MONKE_BUILD_COORDINATE_DIGEST, CANONICAL_TYPE2PARENT_SOURCE,
    REGISTER_SIGNAL_FAST_CACHE, build_tgm_load_continuation, cached_world_numeric_field,
    canonical_type2parent_program, compile_lumcount_trace, compile_register_signal_trace,
    compile_rooted_list_trace, discover_offset_native, jit_disabled, numeric_jit_prefix_candidate,
    numeric_trace_instructions, revalidated_ruin_rejection, ruin_scan_attach_at_call,
    run_ruin_affected_turfs_batch, run_tgm_build_cache_simple_member, tgm_attach_location,
    try_run_packed_numeric_dispatch_block, try_run_rich_numeric_dispatch_block,
};

#[cfg(test)]
pub(crate) use builtins::is_subtype;

#[cfg(test)]
pub(crate) use value_ops::{
    assign_datum_or_shared_field, canonicalize_owned_value, canonicalize_value,
    datum_field_or_shared, dm_list_length_number, read_list_value, write_list_value,
};

use bytecode::next_module_identity;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dm_core::{DmNumberBits, SourceSpan};
use dm_jit::NumericExecutionState;
use dm_syntax::{Definition, DefinitionKind};
pub use dm_value::Value;
use dm_value::{DatumId, FieldName, ListId, PackedValue, TypePath, ValueError, ValueHeap};

// Dream Maker compiles text macros to non-printing format characters. Keep
// the same representation boundary in Dream64 so the formatter still has the
// original interpolation value available for articles, ordinals, and gendered
// pronouns.
const TEXT_MACRO_THE: char = '\u{f100}';
const TEXT_MACRO_THE_UPPER: char = '\u{f101}';
const TEXT_MACRO_A: char = '\u{f102}';
const TEXT_MACRO_A_UPPER: char = '\u{f103}';
const TEXT_MACRO_PROPER: char = '\u{f104}';
const TEXT_MACRO_IMPROPER: char = '\u{f105}';
const TEXT_MACRO_ROMAN: char = '\u{f106}';
const TEXT_MACRO_ROMAN_UPPER: char = '\u{f107}';
const TEXT_MACRO_ORDINAL: char = '\u{f108}';
const TEXT_MACRO_PLURAL: char = '\u{f109}';
const TEXT_MACRO_SUBJECT: char = '\u{f10a}';
const TEXT_MACRO_SUBJECT_UPPER: char = '\u{f10b}';
const TEXT_MACRO_POSSESSIVE_ADJECTIVE: char = '\u{f10c}';
const TEXT_MACRO_POSSESSIVE_ADJECTIVE_UPPER: char = '\u{f10d}';
const TEXT_MACRO_OBJECT: char = '\u{f10e}';
const TEXT_MACRO_REFLEXIVE: char = '\u{f10f}';
const TEXT_MACRO_POSSESSIVE: char = '\u{f110}';
const TEXT_MACRO_POSSESSIVE_UPPER: char = '\u{f111}';

/// One independently identified procedure body supplied by a semantic layer.
#[derive(Clone, Debug)]
pub struct ProcedureSpec<'definition> {
    /// Unique diagnostic path for stack traces and lookup.
    pub path: String,
    /// Parsed procedure definition to compile.
    pub definition: &'definition Definition,
    /// Index of the exact parent implementation in the same spec slice.
    pub parent: Option<usize>,
    /// Semantically resolved bare-call targets, keyed by selector.
    ///
    /// This preserves object-tree inheritance when it differs from lexical
    /// path ancestry, such as `/area` inheriting `/datum`.
    pub static_calls: BTreeMap<String, usize>,
    /// Bare identifiers that resolve to fields on the executing procedure's
    /// `src` datum when they do not name a parameter or local.
    pub src_fields: BTreeMap<String, FieldName>,
    /// Bare identifiers that resolve to persistent runtime globals when they
    /// do not name a parameter, local, or `src` field.
    pub global_fields: BTreeMap<String, FieldName>,
}

fn procedure_spec_is_type_path(spec: &ProcedureSpec<'_>) -> bool {
    if spec.definition.kind == DefinitionKind::Procedure {
        return true;
    }
    // Generated managed-global procedures can be reopened by a shorthand
    // owner override. The effective spec then carries ProcedureOverride even
    // though the canonical declaration is a real `/proc/InitGlobal*` member
    // and BYOND includes it in typesof(owner/proc). Do not admit ordinary
    // shorthand overrides such as Initialize/New into the procedure catalog.
    spec.path
        .split_once('@')
        .map_or(spec.path.as_str(), |(path, _)| path)
        .rsplit_once("/proc/")
        .is_some_and(|(_, selector)| selector.starts_with("InitGlobal"))
}

pub(crate) fn procedure_type_catalog_from_specs(specs: &[ProcedureSpec<'_>]) -> Vec<TypePath> {
    let mut catalog = specs
        .iter()
        .filter(|spec| procedure_spec_is_type_path(spec))
        .filter_map(|spec| {
            let (path, ordinal) = spec
                .path
                .rsplit_once('@')
                .map_or((spec.path.as_str(), usize::MAX), |(path, ordinal)| {
                    (path, ordinal.parse::<usize>().unwrap_or(usize::MAX))
                });
            TypePath::parse(path).ok().map(|path| (ordinal, path))
        })
        .collect::<Vec<_>>();
    catalog.sort_by_key(|(ordinal, _)| *ordinal);
    catalog.into_iter().map(|(_, path)| path).collect()
}

/// Failure while compiling the initial executable subset.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompileError {
    /// Human-readable diagnostic.
    pub message: String,
}

impl fmt::Display for CompileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for CompileError {}

/// Bounded diagnostics from one complete deferred-procedure materialization.
///
/// The total failure count is retained even when the diagnostic sample is
/// capped. Successfully lowered procedures are installed before this error is
/// returned, allowing callers that keep the module to inspect or execute the
/// valid portion without compiling it a second time.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FullyEagerCompileErrors {
    diagnostics: Vec<CompileError>,
    total_failures: usize,
    successful_procedures: usize,
}

impl FullyEagerCompileErrors {
    /// First failures in stable procedure identity order, bounded by the
    /// caller-provided diagnostic limit.
    #[must_use]
    pub fn diagnostics(&self) -> &[CompileError] {
        &self.diagnostics
    }

    /// Total number of deferred procedures that failed, including diagnostics
    /// omitted by the caller-provided limit.
    #[must_use]
    pub const fn total_failures(&self) -> usize {
        self.total_failures
    }

    /// Deferred procedures that lowered and were installed successfully during
    /// the same complete pass.
    #[must_use]
    pub const fn successful_procedures(&self) -> usize {
        self.successful_procedures
    }
}

impl fmt::Display for FullyEagerCompileErrors {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shown = self.diagnostics.len();
        write!(
            formatter,
            "{} deferred procedures failed eager compilation; {} compiled successfully; showing first {} failures",
            self.total_failures, self.successful_procedures, shown
        )?;
        for diagnostic in &self.diagnostics {
            write!(formatter, "\n- {}", diagnostic.message)?;
        }
        Ok(())
    }
}

impl std::error::Error for FullyEagerCompileErrors {}

/// Failure while executing portable bytecode.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeError {
    /// Human-readable runtime diagnostic.
    pub message: String,
    /// Instruction index at which execution failed.
    pub instruction: usize,
    /// Source span associated with the failing instruction, when available.
    pub source_span: Option<SourceSpan>,
    /// Active procedures from the entry point through the failing frame.
    pub call_stack: Vec<CallTrace>,
}

/// One source-mapped procedure in a runtime error's call stack.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallTrace {
    /// Canonical procedure path.
    pub procedure: String,
    /// Instruction active in this frame.
    pub instruction: usize,
    /// Source span associated with the active instruction, when available.
    pub source_span: Option<SourceSpan>,
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} at instruction {}",
            self.message, self.instruction
        )?;
        if let Some(span) = self.source_span {
            write!(formatter, " (source {}..{})", span.start, span.end)?;
        }
        if !self.call_stack.is_empty() {
            formatter.write_str("\ncall stack:")?;
            for trace in self.call_stack.iter().rev() {
                write!(
                    formatter,
                    "\n  {} at instruction {}",
                    trace.procedure, trace.instruction
                )?;
                if let Some(span) = trace.source_span {
                    write!(formatter, " (source {}..{})", span.start, span.end)?;
                }
            }
        }
        Ok(())
    }
}

impl std::error::Error for RuntimeError {}

/// Limits applied by the deterministic reference interpreter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionLimits {
    /// Maximum number of simultaneously active procedure frames.
    pub max_call_depth: usize,
    /// Maximum bytecode instructions executed across all call frames in one
    /// interpreter dispatch. Standalone exhaustion is an error; scheduled
    /// exhaustion retains the continuation for a same-tick dispatch slice.
    pub max_steps: u64,
    /// Optional host wall-clock budget for one scheduled continuation slice.
    /// Standalone execution ignores this limit to remain deterministic.
    pub wall_clock_budget: Option<Duration>,
}

impl Default for ExecutionLimits {
    fn default() -> Self {
        Self {
            max_call_depth: 1_024,
            max_steps: 10_000_000,
            wall_clock_budget: None,
        }
    }
}

const MAX_EFFECTIVE_INITIAL_VALUE_CACHE_ENTRIES: usize = 524_288;
const MAX_EFFECTIVE_INITIAL_VALUE_CACHE_FIELDS_PER_TYPE: usize = 8;
const MAX_INSTANCE_INITIALIZER_PLAN_CACHE_ENTRIES: usize = 16_384;

/// Dense per-world storage for DM globals.
///
/// Compiled instructions still carry symbolic names while the runtime-image
/// migration is in progress, but each name resolves once to a stable numeric
/// slot. Reads and writes then touch a contiguous value table instead of a
/// tree node. The ordered name set preserves deterministic reflection and
/// snapshot output independently of the hot lookup representation.
#[derive(Default)]
struct GlobalStore {
    names: std::collections::BTreeSet<FieldName>,
    slots_by_name: HashMap<FieldName, u32>,
    slots: Vec<Option<Value>>,
    free_slots: Vec<u32>,
}

impl GlobalStore {
    fn get(&self, name: &FieldName) -> Option<&Value> {
        let slot = usize::try_from(*self.slots_by_name.get(name)?).ok()?;
        self.slots.get(slot)?.as_ref()
    }

    fn insert(&mut self, name: FieldName, value: Value) -> Option<Value> {
        if let Some(slot) = self.slots_by_name.get(&name).copied() {
            return self.slots[slot as usize].replace(value);
        }
        let slot = self.free_slots.pop().unwrap_or_else(|| {
            let slot = u32::try_from(self.slots.len())
                .expect("DM global slot count exceeds the 32-bit runtime identity space");
            self.slots.push(None);
            slot
        });
        self.slots[slot as usize] = Some(value);
        self.names.insert(name.clone());
        self.slots_by_name.insert(name, slot);
        None
    }

    fn remove(&mut self, name: &FieldName) -> Option<Value> {
        let slot = self.slots_by_name.remove(name)?;
        self.names.remove(name);
        let value = self.slots[slot as usize].take();
        self.free_slots.push(slot);
        value
    }

    fn keys(&self) -> impl Iterator<Item = &FieldName> {
        self.names.iter()
    }

    fn values(&self) -> impl Iterator<Item = &Value> {
        self.slots.iter().filter_map(Option::as_ref)
    }

    fn iter(&self) -> impl Iterator<Item = (&FieldName, &Value)> {
        self.names
            .iter()
            .filter_map(|name| self.get(name).map(|value| (name, value)))
    }
}

impl FromIterator<(FieldName, Value)> for GlobalStore {
    fn from_iter<T: IntoIterator<Item = (FieldName, Value)>>(iter: T) -> Self {
        let mut globals = Self::default();
        for (name, value) in iter {
            globals.insert(name, value);
        }
        globals
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Result of a host-requested heap collection at a quiescent VM boundary.
pub struct QuiescentHeapCompaction {
    /// Unreachable datum slots released by the collection.
    pub reclaimed_datums: usize,
    /// Unreachable list slots released by the collection.
    pub reclaimed_lists: usize,
    /// Wall-clock time spent tracing, reclaiming, and compacting storage.
    pub elapsed: Duration,
}

/// Exact bounds measured from one immutable project DMM resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DmmMeasurement {
    /// MD5 identity of the exact map bytes measured at artifact construction.
    pub digest: [u8; 16],
    /// Minimum and maximum X, Y, and Z coordinates in BYOND bounds order.
    pub bounds: [i32; 6],
}

/// One immutable parsed-map coordinate record supplied by a runtime artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParsedDmmGrid {
    /// One-based starting X coordinate.
    pub x: i32,
    /// One-based starting Y coordinate.
    pub y: i32,
    /// One-based Z coordinate.
    pub z: i32,
    /// Top-to-bottom encoded map-key lines.
    pub lines: Vec<String>,
}

/// Complete immutable parser product for one project DMM resource.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParsedDmm {
    /// MD5 identity of the exact source bytes.
    pub digest: [u8; 16],
    /// Whether the source uses TGM model formatting.
    pub tgm: bool,
    /// Uniform map-key byte width.
    pub key_len: u32,
    /// Encoded grid-line byte width.
    pub line_len: u32,
    /// Bounds in MAP_MINX..MAP_MAXZ order.
    pub bounds: [i32; 6],
    /// Source-ordered model key/body pairs.
    pub models: Vec<(String, String)>,
    /// Source-ordered coordinate records.
    pub grids: Vec<ParsedDmmGrid>,
}

#[derive(Clone, Debug)]
struct NativeWalk {
    due_tick: u64,
    sequence: u64,
    lag: u64,
    kind: NativeWalkKind,
}

impl NativeWalk {
    fn target(&self) -> Option<DatumId> {
        match self.kind {
            NativeWalkKind::Direction(_) | NativeWalkKind::Random => None,
            NativeWalkKind::Towards(target)
            | NativeWalkKind::To { target, .. }
            | NativeWalkKind::Away { target, .. } => Some(target),
        }
    }
}

#[derive(Clone, Debug)]
enum NativeWalkKind {
    Direction(i16),
    Random,
    Towards(DatumId),
    To { target: DatumId, minimum: f32 },
    Away { target: DatumId, maximum: f32 },
}

#[derive(Clone, Debug)]
enum SimpleIterationValue {
    Null,
    Number(DmNumberBits),
    Text(String),
    File(String),
    TypePath(TypePath),
    Local(u16),
}

#[derive(Clone, Debug)]
struct SimpleIterationFieldAssignment {
    list_slot: u16,
    index_slot: u16,
    item_slot: u16,
    value: SimpleIterationValue,
    field: FieldName,
    store_instruction: usize,
    exit_instruction: usize,
}

/// Proves that `PrepareIteration` consumes the sole handle to a list just
/// allocated by `block()`. The adjacent instructions rule out stores and
/// duplicates; rejecting every non-fallthrough entry to `prepare` prevents a
/// branch, exception handler, or spawned frame from bypassing the allocation.
fn prepare_iteration_consumes_fresh_block(program: &Program, prepare: usize) -> bool {
    if prepare == 0
        || !matches!(
            program.instructions.get(prepare - 1),
            Some(Instruction::Block { .. })
        )
    {
        return false;
    }

    !program
        .instructions
        .iter()
        .any(|instruction| match instruction {
            Instruction::JumpIfNull(target)
            | Instruction::JumpIfFalse(target)
            | Instruction::Jump(target) => *target == prepare,
            Instruction::LoadStaticLocalOrJump { target, .. }
            | Instruction::JumpIfArgumentSupplied { target, .. } => *target == prepare,
            Instruction::BeginTry { catch, end, .. } => *catch == prepare || *end == prepare,
            Instruction::Spawn { entry } => *entry == prepare,
            _ => false,
        })
}

fn simple_iteration_field_assignment(
    program: &Program,
    prepare: usize,
) -> Option<SimpleIterationFieldAssignment> {
    let tail = program.instructions.get(prepare + 1..prepare + 19)?;
    let [
        Instruction::StoreLocal(list_slot),
        Instruction::PushNumber(one),
        Instruction::StoreLocal(index_slot),
        condition_header,
        Instruction::ListLengthLocal(condition_list),
        Instruction::LessEqual,
        Instruction::JumpIfFalse(exit_instruction),
        Instruction::LoadLocal(iteration_index),
        Instruction::IndexLocalList(iteration_list),
        Instruction::StoreLocal(item_slot),
        Instruction::LoadLocal(receiver_slot),
        value,
        Instruction::StoreField(field),
        Instruction::LoadLocal(increment_index),
        Instruction::PushNumber(increment),
        Instruction::Add,
        Instruction::StoreLocal(stored_index),
        Instruction::Jump(condition_target),
    ] = tail
    else {
        return None;
    };
    let condition = prepare + 4;
    let exit = prepare + 19;
    let condition_index_matches = match condition_header {
        Instruction::LoadLocal(condition_index) => condition_index == index_slot,
        Instruction::NextLocalListIteration {
            list_slot: fused_list,
            index_slot: fused_index,
            item_slot: fused_item,
            exit: fused_exit,
        } => {
            fused_list == list_slot
                && fused_index == index_slot
                && fused_item == item_slot
                && fused_exit == exit_instruction
        }
        _ => false,
    };
    if one.to_f32() != 1.0
        || increment.to_f32() != 1.0
        || !condition_index_matches
        || condition_list != list_slot
        || iteration_list != list_slot
        || iteration_index != index_slot
        || receiver_slot != item_slot
        || increment_index != index_slot
        || stored_index != index_slot
        || *condition_target != condition
        || *exit_instruction != exit
    {
        return None;
    }
    let value = match value {
        Instruction::PushNull => SimpleIterationValue::Null,
        Instruction::PushNumber(value) => SimpleIterationValue::Number(*value),
        Instruction::PushText(value) => SimpleIterationValue::Text(value.to_string()),
        Instruction::PushFile(value) => SimpleIterationValue::File(value.clone()),
        Instruction::PushTypePath(value) => SimpleIterationValue::TypePath(value.clone()),
        Instruction::LoadLocal(slot)
            if slot != item_slot && slot != list_slot && slot != index_slot =>
        {
            SimpleIterationValue::Local(*slot)
        }
        _ => return None,
    };
    Some(SimpleIterationFieldAssignment {
        list_slot: *list_slot,
        index_slot: *index_slot,
        item_slot: *item_slot,
        value,
        field: field.clone(),
        store_instruction: prepare + 13,
        exit_instruction: exit,
    })
}

/// Executes one standalone program to completion on the reference interpreter.
///
/// Calls cannot occur in a standalone program; use [`execute_module`] for
/// programs produced by [`compile_module`].
///
/// # Errors
///
/// Returns [`RuntimeError`] for invalid bytecode stack/local access or
/// operations on values of unsupported types.
pub fn execute(program: &Program, arguments: &[Value]) -> Result<Value, RuntimeError> {
    execute_with_limits(program, arguments, ExecutionLimits::default())
}

/// Executes one standalone program against persistent runtime state.
///
/// # Errors
///
/// Returns [`RuntimeError`] for invalid bytecode or value operations.
pub fn execute_in_state(
    program: &Program,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, RuntimeError> {
    execute_with_limits_in_state(program, arguments, ExecutionLimits::default(), state)
}

/// Executes one standalone program with persistent state and object context.
///
/// # Errors
///
/// Returns [`RuntimeError`] for invalid bytecode or value operations.
pub fn execute_in_context(
    program: &Program,
    arguments: &[Value],
    state: &mut ExecutionState,
    context: &ExecutionContext,
) -> Result<Value, RuntimeError> {
    let entry = ProcedureId(0);
    let module = Module {
        identity: next_module_identity(),
        procedures: vec![Arc::new(program.clone())],
        paths: vec!["<standalone>".to_owned()],
        names: HashMap::new(),
        dynamic_names: HashMap::new(),
        deferred: Arc::new(HashMap::new()),
        procedure_types: Vec::new(),
        initializer_call_names: None,
        compact_wordcode: Default::default(),
        semantic_digests: Default::default(),
    };
    execute_module_with_limits_in_context(
        &module,
        entry,
        arguments,
        ExecutionLimits::default(),
        state,
        context,
    )
}

/// Executes one standalone program with explicit deterministic safety limits.
///
/// # Errors
///
/// Returns [`RuntimeError`] for the same failures as [`execute`], including
/// call-depth or total-instruction budget exhaustion.
pub fn execute_with_limits(
    program: &Program,
    arguments: &[Value],
    limits: ExecutionLimits,
) -> Result<Value, RuntimeError> {
    let mut state = ExecutionState::new();
    execute_with_limits_in_state(program, arguments, limits, &mut state)
}

/// Executes one standalone program with persistent state and explicit limits.
///
/// # Errors
///
/// Returns [`RuntimeError`] for invalid bytecode, unsupported value operations,
/// stale heap references, or safety-limit exhaustion.
pub fn execute_with_limits_in_state(
    program: &Program,
    arguments: &[Value],
    limits: ExecutionLimits,
    state: &mut ExecutionState,
) -> Result<Value, RuntimeError> {
    let entry = ProcedureId(0);
    let module = Module {
        identity: next_module_identity(),
        procedures: vec![Arc::new(program.clone())],
        paths: vec!["<standalone>".to_owned()],
        names: HashMap::new(),
        dynamic_names: HashMap::new(),
        deferred: Arc::new(HashMap::new()),
        procedure_types: Vec::new(),
        initializer_call_names: None,
        compact_wordcode: Default::default(),
        semantic_digests: Default::default(),
    };
    execute_module_with_limits_in_state(&module, entry, arguments, limits, state)
}

/// Executes a procedure from a compiled module with default safety limits.
///
/// Declared parameters are bound positionally. Missing parameters are `null`,
/// and extra supplied values are retained in the frame for future `args`
/// support, matching DM's permissive call arity.
///
/// # Errors
///
/// Returns [`RuntimeError`] for invalid procedure identities or bytecode,
/// unsupported value operations, and call-depth exhaustion.
pub fn execute_module(
    module: &Module,
    entry: ProcedureId,
    arguments: &[Value],
) -> Result<Value, RuntimeError> {
    execute_module_with_limits(module, entry, arguments, ExecutionLimits::default())
}

/// Executes a module procedure against persistent runtime state.
///
/// # Errors
///
/// Returns [`RuntimeError`] for the same failures as [`execute_module`].
pub fn execute_module_in_state(
    module: &Module,
    entry: ProcedureId,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, RuntimeError> {
    execute_module_with_limits_in_state(module, entry, arguments, ExecutionLimits::default(), state)
}

/// Executes a module procedure with persistent state and object context.
///
/// # Errors
///
/// Returns [`RuntimeError`] for invalid bytecode or value operations.
pub fn execute_module_in_context(
    module: &Module,
    entry: ProcedureId,
    arguments: &[Value],
    state: &mut ExecutionState,
    context: &ExecutionContext,
) -> Result<Value, RuntimeError> {
    execute_module_with_limits_in_context(
        module,
        entry,
        arguments,
        ExecutionLimits::default(),
        state,
        context,
    )
}

/// Executes a module procedure with explicit deterministic safety limits.
///
/// # Errors
///
/// Returns [`RuntimeError`] for the same failures as [`execute_module`].
pub fn execute_module_with_limits(
    module: &Module,
    entry: ProcedureId,
    arguments: &[Value],
    limits: ExecutionLimits,
) -> Result<Value, RuntimeError> {
    let mut state = ExecutionState::new();
    execute_module_with_limits_in_state(module, entry, arguments, limits, &mut state)
}

/// Executes a module procedure against persistent state with explicit limits.
///
/// # Errors
///
/// Returns [`RuntimeError`] for invalid bytecode, unsupported value operations,
/// stale heap references, or safety-limit exhaustion.
pub fn execute_module_with_limits_in_state(
    module: &Module,
    entry: ProcedureId,
    arguments: &[Value],
    limits: ExecutionLimits,
    state: &mut ExecutionState,
) -> Result<Value, RuntimeError> {
    execute_module_with_limits_in_context(
        module,
        entry,
        arguments,
        limits,
        state,
        &ExecutionContext::default(),
    )
}

/// Executes a module procedure with persistent state, context, and limits.
///
/// Current, parent, and resolved procedure calls inherit both `src` and `usr`
/// unchanged from their caller frame.
///
/// # Errors
///
/// Returns [`RuntimeError`] for invalid bytecode, value operations, stale
/// handles, missing fields/globals, or safety-limit exhaustion.
pub fn execute_module_with_limits_in_context(
    module: &Module,
    entry: ProcedureId,
    arguments: &[Value],
    limits: ExecutionLimits,
    state: &mut ExecutionState,
    context: &ExecutionContext,
) -> Result<Value, RuntimeError> {
    state.assert_owner_thread();
    let program = module
        .resolve_procedure(entry)
        .map_err(|message| RuntimeError {
            message,
            instruction: 0,
            source_span: None,
            call_stack: Vec::new(),
        })?;
    if limits.max_call_depth == 0 {
        return Err(RuntimeError {
            message: "maximum call depth must be at least one".to_owned(),
            instruction: 0,
            source_span: program.source_spans.first().copied(),
            call_stack: vec![trace(module, entry, 0)],
        });
    }

    let frames = vec![make_frame(entry, program, arguments, context)];
    finish_frame_run(
        module,
        run_frames(module, frames, limits, StepBudgetBehavior::Error, state)?,
        state,
    )
}

fn finish_frame_run(
    module: &Module,
    outcome: FrameRunOutcome,
    state: &mut ExecutionState,
) -> Result<Value, RuntimeError> {
    match outcome {
        FrameRunOutcome::Complete(value) => {
            state.host_value_roots.push(value.clone());
            Ok(value)
        }
        FrameRunOutcome::Yielded { frames, delay } => {
            schedule_frames(state, frames, delay);
            let _ = module;
            Ok(Value::Null)
        }
        FrameRunOutcome::Prompted { id, prompt } => {
            register_prompt(state, id, prompt);
            let _ = module;
            Ok(Value::Null)
        }
    }
}

#[cfg(test)]
mod tests;
