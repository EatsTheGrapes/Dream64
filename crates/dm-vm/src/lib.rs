//! Portable stack bytecode and the deterministic reference interpreter.

#![cfg_attr(not(test), deny(missing_docs))]

mod builtins;
mod bytecode;
mod compact_wordcode;
mod compile;
mod execution;
mod module_codec;
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

pub use execution::{
    ContinuationMetrics, DeclaredFieldQuickeningMetrics, ExecutionContext, ExecutionState,
    VmContinuationId, advance_scheduler,
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
    assign_datum_field, assign_datum_or_shared_field, canonicalize_owned_value, canonicalize_value,
    clone_icon_datum, compare_values, datum_field_or_initial, datum_field_or_shared,
    datum_shared_storage, deterministic_unit, dm_list_length_number, dynamic_call_target_named,
    engine_builtin_initial_fields, engine_builtin_initial_value, engine_root_initial_value,
    execute_icon_method, get_step_builtin, initialize_existing_datum, instance_initializer_plan,
    is_area_type_path, is_atom_type_path, is_icon_datum, is_matrix_datum, is_turf_type_path,
    lazy_atom_list_field, logical_or_empty_list_field, logical_or_empty_list_index,
    matrix_components, matrix_product, parse_heap_reference, pop, read_list_value, runtime_truthy,
    stringify_dm_value, values_equal, vector_components, write_list_value,
};

#[cfg(test)]
pub(crate) use value_ops::{
    allocate_or_replace_engine_datum, dm_direction_bits, dm_list_resize_length,
    dm_world_coordinate, dynamic_call_target, dynamic_call_target_named_at_callsite,
    engine_owner_field_names, engine_root_paths, indexed_text_character,
    initial_value_or_engine_root, pop_builtin_arguments, runtime_initial_field_value,
    values_equivalent,
};

use bytecode::next_module_identity;

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use builtins::{execute_standard_builtin, is_subtype};

use dm_core::{DmNumberBits, SourceSpan};
use dm_dmf::Diagnostic;
use dm_jit::{
    CompiledNumericTrace, CompiledRootedBlock, NumericExecutionState, NumericInstruction,
    NumericRunOutcome, RootedBlockOutcome, compile_numeric_field_trace, compile_numeric_trace,
    compile_safe_rooted_block,
};
use dm_syntax::{Definition, DefinitionKind};
pub use dm_value::Value;
use dm_value::{DatumId, FieldName, ListId, PackedValue, TypePath, ValueError, ValueHeap};
use smallvec::SmallVec;

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

/// A local client could not be created from the supplied DMF skin.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalClientError {
    /// DMF diagnostics that prevented session creation.
    pub diagnostics: Vec<Diagnostic>,
}

/// One cardinal movement requested by a locally attached client.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalMovementDirection {
    /// Increase Y by one.
    North,
    /// Decrease Y by one.
    South,
    /// Increase X by one.
    East,
    /// Decrease X by one.
    West,
}

/// Authoritative location of one locally attached client and mob.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalClientState {
    /// Client datum identity.
    pub client: DatumId,
    /// Controlled mob identity.
    pub mob: DatumId,
    /// Current turf X coordinate.
    pub x: i32,
    /// Current turf Y coordinate.
    pub y: i32,
    /// Current turf Z coordinate.
    pub z: i32,
}

/// One authoritative UI operation emitted by DM for a connected local client.
#[derive(Clone, Debug, PartialEq)]
pub enum LocalClientUiEvent {
    /// Open a URL requested by BYOND's `link()` builtin.
    Link {
        /// Absolute URL supplied by game code.
        url: String,
    },
    /// Mutate named DMF control properties.
    Winset {
        /// Target DMF control address.
        control: String,
        /// BYOND winset parameter string.
        parameters: String,
    },
    /// Append a message to an output control.
    Output {
        /// Target output control address.
        control: String,
        /// Text appended to the control.
        message: String,
    },
    /// Register a browser-visible resource under a logical name.
    BrowseResource {
        /// Browser-visible logical resource name.
        name: String,
        /// Complete resource payload.
        bytes: Vec<u8>,
    },
    /// Display HTML in a browser control.
    Browse {
        /// BYOND browser window/control selector.
        window: String,
        /// HTML document body.
        html: String,
    },
    /// Display a modal prompt and suspend the calling DM continuation.
    Prompt {
        /// Stable response token scoped to the connected client.
        id: u64,
        /// Native prompt presentation and response conversion.
        kind: LocalClientPromptKind,
        /// Window caption.
        title: String,
        /// Prompt body.
        message: String,
        /// Initial editable value or selected button.
        default: String,
        /// Alert buttons or list-picker display values.
        choices: Vec<String>,
        /// Whether closing the prompt may yield null.
        can_cancel: bool,
    },
    /// Play, replace, or stop one BYOND sound channel.
    Sound {
        /// Project-relative audio resource; `None` stops the channel.
        file: Option<String>,
        /// BYOND channel number, where zero is fire-and-forget.
        channel: i32,
        /// Whether playback loops.
        repeat: bool,
        /// Volume percentage.
        volume: f32,
        /// Requested playback frequency.
        frequency: f32,
        /// Stereo pan from -100 through 100.
        pan: f32,
    },
}

/// Native presentation class for one local-client prompt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalClientPromptKind {
    /// Single-line text or command text.
    Text,
    /// Multi-line message text.
    Message,
    /// Floating-point number.
    Number,
    /// Color text such as `#rrggbb`.
    Color,
    /// Project-relative file/icon/sound path.
    File,
    /// One of a fixed set of values.
    List,
    /// One of one through three alert buttons.
    Alert,
}

/// Typed answer supplied by the native client for a pending prompt.
#[derive(Clone, Debug, PartialEq)]
pub enum LocalClientPromptResponse {
    /// User cancelled a nullable prompt.
    Null,
    /// Text, message, or color response.
    Text(String),
    /// Numeric response.
    Number(f32),
    /// Zero-based alert/list choice index.
    Choice(usize),
}

/// One stable map cell copied out of the runtime heap.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalClientMapTile {
    /// Turf X coordinate.
    pub x: i32,
    /// Turf Y coordinate.
    pub y: i32,
    /// Canonical turf type path.
    pub type_path: String,
    /// DM-visible color converted to stable text when present.
    pub color: Option<String>,
    /// Materialized atom identities currently contained by this turf.
    pub occupants: Vec<DatumId>,
    /// Turf and contained atoms in stable plane/layer/insertion draw order.
    pub appearances: Vec<LocalClientAppearance>,
}

/// Owned DM appearance data required by the local renderer.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalClientAppearance {
    /// Source atom or appearance datum when one exists.
    pub datum: DatumId,
    /// Canonical runtime type path.
    pub type_path: String,
    /// Backing DMI/resource path after unwrapping `/icon` objects.
    pub icon: Option<String>,
    /// Selected icon state.
    pub icon_state: Option<String>,
    /// BYOND direction bitfield.
    pub dir: i32,
    /// Draw layer.
    pub layer: f32,
    /// Draw plane.
    pub plane: f32,
    /// BYOND appearance behavior bitfield, kept at the VM's narrow integer width.
    pub appearance_flags: i32,
    /// BYOND mouse hit-test policy value.
    pub mouse_opacity: i32,
    /// X pixel offset.
    pub pixel_x: f32,
    /// Y pixel offset.
    pub pixel_y: f32,
    /// W pixel offset.
    pub pixel_w: f32,
    /// Z pixel offset.
    pub pixel_z: f32,
    /// DM color represented as stable transport text.
    pub color: Option<String>,
    /// Alpha in BYOND's 0 through 255 range.
    pub alpha: f32,
    /// HTML-like BYOND maptext attached to this appearance.
    pub maptext: Option<String>,
    /// Maptext box width in pixels.
    pub maptext_width: f32,
    /// Maptext box height in pixels.
    pub maptext_height: f32,
    /// Maptext X offset in pixels.
    pub maptext_x: f32,
    /// Maptext Y offset in pixels.
    pub maptext_y: f32,
    /// Nested underlays in stable plane/layer/insertion order.
    pub underlays: Vec<LocalClientAppearance>,
    /// Nested overlays in stable plane/layer/insertion order.
    pub overlays: Vec<LocalClientAppearance>,
}

/// Owned map snapshot suitable for transport to a local client.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalClientMapSnapshot {
    /// Maximum X coordinate represented by the world at this Z level.
    pub width: i32,
    /// Maximum Y coordinate represented by the world at this Z level.
    pub height: i32,
    /// Selected Z level.
    pub z: i32,
    /// Tiles in deterministic Y-then-X world index order.
    pub tiles: Vec<LocalClientMapTile>,
    /// HUD/screen atoms from the attached client's `screen` list.
    pub screen: Vec<LocalClientScreenAppearance>,
}

/// One client-screen appearance and its BYOND screen-space selector.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalClientScreenAppearance {
    /// Optional DMF map-control selector prefix.
    pub map_control: Option<String>,
    /// BYOND `screen_loc` expression used for viewport placement.
    pub screen_loc: String,
    /// Stable insertion position in `client.screen`.
    pub insertion: usize,
    /// Fully expanded appearance tree.
    pub appearance: LocalClientAppearance,
}

/// Mouse transition delivered to a client-owned screen atom.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalScreenPointerEvent {
    /// Pointer entered the atom's visible pixels.
    Entered,
    /// Pointer left the atom's visible pixels.
    Exited,
    /// Primary pointer button activated the atom.
    Click,
}

impl fmt::Display for LocalClientError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "DMF contains {} error diagnostic(s)",
            self.diagnostics.len()
        )
    }
}

impl std::error::Error for LocalClientError {}

#[derive(Clone, Debug, Default)]
struct SavefileState {
    entries: HashMap<String, Value>,
    cd: String,
}

#[derive(Clone, Debug)]
enum ShuttleTracePostReturn {
    AtmosInit,
    NullifyNode { slot: Option<usize> },
}

#[derive(Debug)]
struct AtomsProfile {
    started: Instant,
    last_snapshot: Instant,
    startup_root: Option<String>,
    total_instructions: u64,
    instruction_categories: Option<[u64; STARTUP_INSTRUCTION_CATEGORY_COUNT]>,
    samples: HashMap<AtomsProfileProcedure, u64>,
    // Approximate wall time sampled at the existing 4,096-step checkpoints.
    // Native helpers retain logical instruction accounting, so this separates
    // expensive interpreter work from cheaply replayed reference budgets.
    wall_sample_nanos: HashMap<AtomsProfileProcedure, u128>,
    frame_entries: HashMap<AtomsProfileProcedure, u64>,
    paths: HashMap<AtomsProfileProcedure, String>,
    instruction_samples: HashMap<AtomsProfileInstruction, u64>,
    instruction_wall_nanos: HashMap<AtomsProfileInstruction, u128>,
    instruction_labels: HashMap<AtomsProfileInstruction, String>,
}

#[derive(Debug)]
struct TgmProfile {
    started: Instant,
    total_instructions: u64,
    procedure_samples: HashMap<AtomsProfileProcedure, u64>,
    instruction_samples: HashMap<AtomsProfileInstruction, u64>,
    paths: HashMap<AtomsProfileProcedure, String>,
    instruction_labels: HashMap<AtomsProfileInstruction, String>,
}

fn tgm_profiling_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("DREAM64_PROFILE_TGM").is_some())
}

fn emit_tgm_profile(profile: &TgmProfile) {
    let mut procedures = profile.procedure_samples.iter().collect::<Vec<_>>();
    procedures.sort_by_key(|(_, samples)| std::cmp::Reverse(**samples));
    eprintln!(
        "boot-vm: tgm-profile-summary elapsed_ms={} instructions={} procedures={}",
        profile.started.elapsed().as_millis(),
        profile.total_instructions,
        procedures.len()
    );
    for (key, samples) in procedures.into_iter().take(32) {
        eprintln!(
            "boot-vm: tgm-profile-procedure samples={} path={}",
            samples,
            profile.paths.get(key).map_or("<missing>", String::as_str)
        );
    }
    let mut instructions = profile.instruction_samples.iter().collect::<Vec<_>>();
    instructions.sort_by_key(|(_, samples)| std::cmp::Reverse(**samples));
    for (key, samples) in instructions.into_iter().take(64) {
        eprintln!(
            "boot-vm: tgm-profile-opcode samples={} path={} pc={} instruction={}",
            samples,
            profile
                .paths
                .get(&AtomsProfileProcedure {
                    module_identity: key.module_identity,
                    procedure: key.procedure,
                })
                .map_or("<missing>", String::as_str),
            key.instruction,
            profile
                .instruction_labels
                .get(key)
                .map_or("<missing>", String::as_str),
        );
    }
}

const STARTUP_INSTRUCTION_CATEGORY_COUNT: usize = 7;

fn startup_instruction_category(instruction: &Instruction) -> usize {
    match instruction {
        Instruction::IndexList
        | Instruction::IndexLocalList(_)
        | Instruction::NextLocalListIteration { .. }
        | Instruction::ListLength
        | Instruction::ListLengthLocal(_)
        | Instruction::Contains
        | Instruction::PrepareIteration => 0,
        Instruction::SetListIndex
        | Instruction::SetListIndexKeep
        | Instruction::CompoundListIndex(_)
        | Instruction::CompoundListIndexKeep(_)
        | Instruction::MutateListIndex { .. }
        | Instruction::LogicalOrEmptyListIndex => 1,
        Instruction::LoadField(_)
        | Instruction::LoadDeclaredField(_)
        | Instruction::LoadGlobal(_)
        | Instruction::InitialField(_)
        | Instruction::InitialDynamicField
        | Instruction::LoadDynamicField => 2,
        Instruction::StoreField(_)
        | Instruction::StoreFieldKeep(_)
        | Instruction::StoreGlobal(_)
        | Instruction::MutateField { .. }
        | Instruction::StoreDynamicField => 3,
        Instruction::Call { .. }
        | Instruction::CallCurrent { .. }
        | Instruction::CallParent { .. }
        | Instruction::CallDynamic { .. }
        | Instruction::StandardBuiltin { .. }
        | Instruction::NativeSrcMethod { .. }
        | Instruction::Return => 4,
        Instruction::JumpIfNull(_)
        | Instruction::JumpIfFalse(_)
        | Instruction::Jump(_)
        | Instruction::JumpIfArgumentSupplied { .. }
        | Instruction::IterationTypeFilter(_) => 5,
        _ => 6,
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct AtomsProfileProcedure {
    module_identity: u64,
    procedure: ProcedureId,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct AtomsProfileInstruction {
    module_identity: u64,
    procedure: ProcedureId,
    instruction: usize,
}

#[derive(Clone, Debug)]
struct ExceptionHandler {
    start: usize,
    end: usize,
    catch: usize,
    local: Option<u16>,
    stack_depth: usize,
}

#[derive(Clone, Debug)]
struct ScheduledSpawn {
    due_tick: u64,
    sequence: u64,
    frames: OwnedContinuation,
}

#[derive(Clone, Debug)]
struct PendingLocalPrompt {
    client: DatumId,
    kind: LocalClientPromptKind,
    choices: Vec<Value>,
    can_cancel: bool,
    continuation: PendingPromptContinuation,
}

#[derive(Clone, Debug)]
enum PendingPromptContinuation {
    Frames(Vec<CallFrame>),
    Verb(PendingVerbInvocation),
}

#[derive(Clone, Debug)]
struct PendingVerbInvocation {
    frame: CallFrame,
    parameter_types: Vec<VerbParameterType>,
    parameter_names: Vec<String>,
    verb_name: String,
    parameter: usize,
}

struct LocalPromptSpec {
    id: u64,
    client: DatumId,
    kind: LocalClientPromptKind,
    choices: Vec<Value>,
    can_cancel: bool,
    event: LocalClientUiEvent,
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

fn local_prompt_client(
    state: &ExecutionState,
    arguments: &[Value],
    usr: &Value,
) -> Option<DatumId> {
    arguments
        .first()
        .into_iter()
        .chain(std::iter::once(usr))
        .filter_map(|value| match value {
            Value::Datum(datum) => Some(*datum),
            _ => None,
        })
        .find_map(|datum| {
            if state.interactive_local_clients.contains(&datum) {
                Some(datum)
            } else {
                state.local_client_mobs.iter().find_map(|(client, mob)| {
                    (*mob == datum && state.interactive_local_clients.contains(client))
                        .then_some(*client)
                })
            }
        })
}

fn prompt_value_text(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::Text(value)) => value.to_string(),
        Some(Value::Number(value)) => value.to_f32().to_string(),
        Some(value) => value.to_string(),
    }
}

fn local_prompt_spec(
    name: &str,
    arguments: &[Value],
    usr: &Value,
    state: &mut ExecutionState,
) -> Result<Option<LocalPromptSpec>, String> {
    let base_name = name.split_once('@').map_or(name, |(name, _)| name);
    if !matches!(base_name, "input" | "alert") {
        return Ok(None);
    }
    let Some(client) = local_prompt_client(state, arguments, usr) else {
        return Ok(None);
    };
    let explicit_usr = arguments
        .first()
        .is_some_and(|value| matches!(value, Value::Datum(_) | Value::Null));
    let base = usize::from(explicit_usr);
    let (kind, title, message, default, choices, can_cancel) = if base_name == "alert" {
        let choices = arguments
            .iter()
            .skip(base + 2)
            .filter(|value| !matches!(value, Value::Null))
            .cloned()
            .collect::<Vec<_>>();
        let choices = if choices.is_empty() {
            vec![Value::text("Ok")]
        } else {
            choices
        };
        (
            LocalClientPromptKind::Alert,
            prompt_value_text(arguments.get(base + 1)),
            prompt_value_text(arguments.get(base)),
            prompt_value_text(choices.first()),
            choices,
            false,
        )
    } else {
        let type_marker = name.split_once('@').map_or("", |(_, marker)| marker);
        let list = type_marker.split('+').any(|part| part == "list");
        let choices = if list {
            arguments
                .last()
                .and_then(|value| match value {
                    Value::List(list) => state.heap.list(*list).ok(),
                    _ => None,
                })
                .map(|values| values.positions().map(|(_, value)| value.clone()).collect())
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let default = arguments.get(base + 2);
        let kind = if list {
            LocalClientPromptKind::List
        } else if type_marker.split('+').any(|part| part == "num")
            || (type_marker.is_empty() && matches!(default, Some(Value::Number(_))))
        {
            LocalClientPromptKind::Number
        } else if type_marker.split('+').any(|part| part == "message") {
            LocalClientPromptKind::Message
        } else if type_marker.split('+').any(|part| part == "color") {
            LocalClientPromptKind::Color
        } else if type_marker
            .split('+')
            .any(|part| matches!(part, "file" | "icon" | "sound"))
        {
            LocalClientPromptKind::File
        } else {
            LocalClientPromptKind::Text
        };
        (
            kind,
            prompt_value_text(arguments.get(base + 1)),
            prompt_value_text(arguments.get(base)),
            prompt_value_text(default),
            choices,
            type_marker.split('+').any(|part| part == "null"),
        )
    };
    state.local_prompt_sequence = state.local_prompt_sequence.saturating_add(1);
    let id = state.local_prompt_sequence;
    let display_choices = choices
        .iter()
        .map(|value| prompt_value_text(Some(value)))
        .collect();
    Ok(Some(LocalPromptSpec {
        id,
        client,
        kind,
        choices,
        can_cancel,
        event: LocalClientUiEvent::Prompt {
            id,
            kind,
            title,
            message,
            default,
            choices: display_choices,
            can_cancel,
        },
    }))
}

fn register_prompt(state: &mut ExecutionState, id: u64, prompt: PendingLocalPrompt) {
    state.pending_local_prompts.insert(id, prompt);
}

fn collect_prompt_appearance_datums(
    appearance: &LocalClientAppearance,
    seen: &mut HashSet<DatumId>,
    values: &mut Vec<Value>,
) {
    if seen.insert(appearance.datum) {
        values.push(Value::Datum(appearance.datum));
    }
    for child in appearance
        .underlays
        .iter()
        .chain(appearance.overlays.iter())
    {
        collect_prompt_appearance_datums(child, seen, values);
    }
}

fn local_verb_prompt_candidates(state: &ExecutionState, client: DatumId) -> Vec<Value> {
    let mut seen = HashSet::new();
    let mut values = Vec::new();
    if let Ok(attached) = state.local_client_state(client) {
        let snapshot = state.local_client_map_snapshot_for(Some(client), attached.z);
        for tile in snapshot.tiles {
            for appearance in &tile.appearances {
                collect_prompt_appearance_datums(appearance, &mut seen, &mut values);
            }
            for occupant in tile.occupants {
                if seen.insert(occupant) {
                    values.push(Value::Datum(occupant));
                }
            }
        }
        for screen in snapshot.screen {
            collect_prompt_appearance_datums(&screen.appearance, &mut seen, &mut values);
        }
    }
    for datum in [Some(client), state.local_client_mobs.get(&client).copied()]
        .into_iter()
        .flatten()
    {
        if seen.insert(datum) {
            values.push(Value::Datum(datum));
        }
    }
    values
}

fn verb_atom_type_allows(state: &ExecutionState, value: &Value, mask: u8) -> bool {
    let Value::Datum(datum) = value else {
        return false;
    };
    let Ok(datum) = state.heap.datum(*datum) else {
        return false;
    };
    let path = datum.type_path().as_str();
    (mask & 1 != 0 && (path == "/obj" || path.starts_with("/obj/")))
        || (mask & 2 != 0 && (path == "/mob" || path.starts_with("/mob/")))
        || (mask & 4 != 0 && (path == "/turf" || path.starts_with("/turf/")))
        || (mask & 8 != 0 && (path == "/area" || path.starts_with("/area/")))
}

fn local_verb_choice_label(state: &ExecutionState, value: &Value) -> String {
    let Value::Datum(datum) = value else {
        return prompt_value_text(Some(value));
    };
    state
        .heap
        .datum(*datum)
        .map(|value| format!("{} [0x{:x}]", value.type_path().as_str(), datum.index() + 1))
        .unwrap_or_else(|_| value.to_string())
}

fn queue_next_verb_prompt(
    state: &mut ExecutionState,
    client: DatumId,
    mut invocation: PendingVerbInvocation,
) -> Result<(), String> {
    let Some(parameter) = invocation
        .frame
        .supplied_parameters
        .iter()
        .position(|supplied| !supplied)
    else {
        schedule_frames(state, vec![invocation.frame], 0.0);
        return Ok(());
    };
    invocation.parameter = parameter;
    let kind = match invocation.parameter_types[parameter] {
        VerbParameterType::Text => LocalClientPromptKind::Text,
        VerbParameterType::Message => LocalClientPromptKind::Message,
        VerbParameterType::Number => LocalClientPromptKind::Number,
        VerbParameterType::Color => LocalClientPromptKind::Color,
        VerbParameterType::File => LocalClientPromptKind::File,
        VerbParameterType::Atom(_)
        | VerbParameterType::Anything
        | VerbParameterType::Unsupported => LocalClientPromptKind::List,
    };
    let choices = if kind == LocalClientPromptKind::List {
        let candidates = local_verb_prompt_candidates(state, client);
        match invocation.parameter_types[parameter] {
            VerbParameterType::Atom(mask) => {
                let mut filtered = candidates
                    .into_iter()
                    .filter(|value| verb_atom_type_allows(state, value, mask))
                    .collect::<Vec<_>>();
                if filtered.is_empty() {
                    filtered.extend(
                        state
                            .heap
                            .datums()
                            .map(|(datum, _)| Value::Datum(datum))
                            .filter(|value| verb_atom_type_allows(state, value, mask))
                            .take(256),
                    );
                }
                filtered
            }
            _ => candidates,
        }
    } else {
        Vec::new()
    };
    let display_choices = choices
        .iter()
        .map(|value| local_verb_choice_label(state, value))
        .collect::<Vec<_>>();
    state.local_prompt_sequence = state.local_prompt_sequence.saturating_add(1);
    let id = state.local_prompt_sequence;
    let parameter_name = invocation
        .parameter_names
        .get(parameter)
        .filter(|name| !name.is_empty())
        .cloned()
        .unwrap_or_else(|| format!("Argument {}", parameter + 1));
    state.emit_local_client_ui_event(
        client,
        LocalClientUiEvent::Prompt {
            id,
            kind,
            title: invocation.verb_name.clone(),
            message: parameter_name,
            default: String::new(),
            choices: display_choices,
            can_cancel: true,
        },
    );
    state.pending_local_prompts.insert(
        id,
        PendingLocalPrompt {
            client,
            kind,
            choices,
            can_cancel: true,
            continuation: PendingPromptContinuation::Verb(invocation),
        },
    );
    Ok(())
}

fn world_datum(state: &ExecutionState) -> Option<DatumId> {
    static WORLD: std::sync::OnceLock<FieldName> = std::sync::OnceLock::new();
    state
        .global(WORLD.get_or_init(|| FieldName::parse("world").expect("built-in world global")))
        .and_then(|value| match value {
            Value::Datum(world) => Some(*world),
            _ => None,
        })
}

fn cached_world_numeric_field(name: &str) -> Option<&'static FieldName> {
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

fn world_numeric_field(state: &ExecutionState, name: &str) -> Option<f32> {
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
fn false_tick_check_target(
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

fn try_run_numeric_loop_branch(
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

fn try_run_numeric_local_update(
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

fn numeric_dispatch_candidate(instruction: &Instruction) -> bool {
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

fn try_run_numeric_dispatch_block(
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

fn try_run_packed_numeric_dispatch_block(
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

fn try_run_rich_numeric_dispatch_block(
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

fn set_world_numeric_field(state: &mut ExecutionState, name: &str, value: f32) {
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

fn advance_headless_world_clock(state: &mut ExecutionState, ticks: u64) {
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

const CANONICAL_TYPE2PARENT_SOURCE: &str = "/proc/type2parent(child)\n\
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

fn canonical_type2parent_program(program: &Program) -> bool {
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
const CANONICAL_MONKE_BUILD_COORDINATE_DIGEST: [u8; 32] = [
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

fn try_run_ruin_affected_turfs_batch(
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

fn run_ruin_affected_turfs_batch(
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

fn canonical_tgm_load_path(module: &Module, procedure: ProcedureId) -> bool {
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

fn run_tgm_build_cache_simple_member(
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
fn try_run_tgm_build_cache_simple_member(
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

fn try_run_build_coordinate_prefix(
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

fn trace_tgm_route(
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
        Some(Value::List(list)) => state.heap.list(*list).ok().map(|list| list.len()),
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
                .map_or(0, |list| list.len());
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

fn build_tgm_load_continuation(
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

enum TgmDrive {
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

fn ruin_scan_attach_at_call(frame: &CallFrame) -> Option<bool> {
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

fn revalidated_ruin_rejection(
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

fn drive_ruin_candidate_scan(
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
        let allowed = match frame.locals.get(1).cloned() {
            Some(Value::List(list)) => list,
            _ => {
                frame.cold_mut().ruin_scan = None;
                frame.instruction = 63;
                return TgmDrive::None;
            }
        };
        let area_type = match area {
            Value::Datum(area) => state
                .heap
                .datum(area)
                .ok()
                .map(|datum| Value::TypePath(datum.type_path().clone()))
                .unwrap_or(Value::Null),
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
                let allowed_entries = state
                    .heap
                    .list(allowed)
                    .ok()
                    .map(|list| {
                        list.associations()
                            .take(8)
                            .map(|(key, value)| format!("{key:?}={value:?}"))
                            .collect::<Vec<_>>()
                            .join(",")
                    })
                    .unwrap_or_else(|| "<stale>".to_owned());
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

fn drive_tgm_load(
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

fn tgm_attach_location(frame: &CallFrame) -> Option<bool> {
    if frame.instruction == 279 {
        return Some(false);
    }
    (frame.instruction == 274 && frame.stack.is_empty()).then_some(true)
}

fn canonical_type2parent_target(
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

fn canonical_type2parent(path: &TypePath) -> Option<TypePath> {
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

fn canonical_static_native_builtin(
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

fn canonical_istext(value: &Value) -> Value {
    Value::number(f32::from(!matches!(
        value,
        Value::Null | Value::Number(_) | Value::TypePath(_) | Value::Datum(_) | Value::List(_)
    )))
}

#[inline(always)]
fn execute_compact_fast_instruction(
    operation: compact_wordcode::CompactFastInstruction,
    frame: &mut CallFrame,
    state: &ExecutionState,
) -> Result<(), String> {
    use compact_wordcode::CompactFastInstruction;

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

fn discover_offset_native(
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

fn try_run_discover_offset_fast_path(
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

fn try_run_parsed_dmm_new_fast_path(
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

fn try_run_dmm_preload_measurement_fast_path(
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
    static REGISTER_SIGNAL_FAST_CACHE: RefCell<HashMap<(u64, ProcedureId), Option<RegisterSignalTrace>>> =
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

fn try_run_camera_chunk_fast_path(
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

struct RegisterSignalTrace {
    gc_destroyed: FieldName,
    signal_procs: FieldName,
    listen_lookup: FieldName,
}

fn compile_register_signal_trace(
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

fn try_run_register_signal_fast_path(
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
        if matches!(procs, None) && runtime_truthy(&state.heap, &procs_value).ok()?
            || matches!(lookup, None) && runtime_truthy(&state.heap, &lookup_value).ok()?
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

struct RootedListTrace {
    compiled: CompiledRootedBlock,
    source_field: FieldName,
    target_field: FieldName,
}

fn compile_rooted_list_trace(program: &Program) -> Option<RootedListTrace> {
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

fn try_run_rooted_list_jit(
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

struct LumcountTrace {
    compiled: CompiledNumericTrace,
    fields: [FieldName; 4],
    lighting_global: FieldName,
    queue_field: FieldName,
}

fn try_run_guarded_jit(
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

fn numeric_jit_prefix_candidate(program: &Program) -> bool {
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

fn jit_disabled() -> bool {
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

fn compile_lumcount_trace(program: &Program) -> Option<LumcountTrace> {
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

fn numeric_trace_instructions(program: &Program) -> Option<Vec<NumericInstruction>> {
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

fn shuttle_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        diagnostic_env_truthy(std::env::var("DREAM64_SHUTTLE_TRACE").ok().as_deref())
    })
}

fn diagnostic_env_truthy(value: Option<&str>) -> bool {
    value.is_some_and(|value| matches!(value, "1" | "true" | "yes" | "on"))
}

fn shuttle_trace_target_filter() -> &'static Option<String> {
    static FILTER: OnceLock<Option<String>> = OnceLock::new();
    FILTER.get_or_init(|| {
        Some(
            std::env::var("DREAM64_SHUTTLE_TRACE_TARGET")
                .unwrap_or_else(|_| "/mixer/layer4".to_owned()),
        )
    })
}

fn shuttle_trace_is_late_shuttle_move(path: &str) -> bool {
    path.contains("lateShuttleMove")
}

fn shuttle_trace_is_nullify_node(path: &str) -> bool {
    path.contains("nullify_node@") || path.ends_with("/nullify_node")
}

fn shuttle_trace_is_atmos_init(path: &str) -> bool {
    path.contains("atmos_init@") || path.ends_with("/atmos_init")
}

fn shuttle_trace_matches_target(state: &ExecutionState, datum: DatumId) -> bool {
    shuttle_trace_target_filter()
        .as_ref()
        .is_some_and(|target| {
            state
                .heap
                .datum(datum)
                .is_ok_and(|datum| datum.type_path().as_str().contains(target))
        })
}

fn shuttle_trace_slot_from_value(value: &Value) -> Option<usize> {
    let value = value.as_number()?;
    if !value.is_finite() || value.fract() != 0.0 {
        return None;
    }
    let value = value.trunc() as i64;
    usize::try_from(value).ok().filter(|slot| *slot > 0)
}

fn shuttle_trace_slot_from_arguments(arguments: &[Value]) -> Option<usize> {
    arguments.first().and_then(shuttle_trace_slot_from_value)
}

fn shuttle_trace_turn_dir(direction: i32, angle: i32) -> i32 {
    const DIRECTIONS: [i32; 8] = [1, 9, 8, 10, 2, 6, 4, 5];
    let Some(index) = DIRECTIONS
        .iter()
        .position(|candidate| *candidate == direction)
    else {
        return direction;
    };
    let steps = angle / 45;
    DIRECTIONS[(index as i32 + steps).rem_euclid(8) as usize]
}

fn shuttle_trace_expected_slot_direction(
    dir_value: f32,
    flipped: bool,
    slot: usize,
) -> Option<i32> {
    if !dir_value.is_finite() {
        return None;
    }
    if dir_value.fract() != 0.0 || dir_value < i32::MIN as f32 || dir_value > i32::MAX as f32 {
        return None;
    }
    let direction = dir_value.trunc() as i32;
    let mut node1 = shuttle_trace_turn_dir(direction, -180);
    let node2 = shuttle_trace_turn_dir(direction, -90);
    let mut node3 = shuttle_trace_turn_dir(direction, 0);
    if flipped {
        node1 = shuttle_trace_turn_dir(node1, 180);
        node3 = shuttle_trace_turn_dir(node3, 180);
    }
    match slot {
        1 => Some(node1),
        2 => Some(node2),
        3 => Some(node3),
        _ => None,
    }
}

fn shuttle_trace_list_len(state: &ExecutionState, list: Option<ListId>) -> usize {
    list.and_then(|list| state.heap.list(list).ok())
        .map_or(0, |list| list.len())
}

fn shuttle_trace_list_slot(
    state: &ExecutionState,
    list: Option<ListId>,
    slot: usize,
) -> Option<Value> {
    let list = list?;
    state.heap.list(list).ok()?.get(slot).ok().cloned()
}

fn shuttle_trace_field_text(state: &ExecutionState, datum: DatumId, name: &str) -> String {
    let field = FieldName::parse(name).expect("built-in atom variable");
    match state.heap.datum_field(datum, &field) {
        Ok(value) => value.to_string(),
        Err(_) => "<missing>".to_owned(),
    }
}

fn shuttle_trace_field_u8(state: &ExecutionState, datum: DatumId, name: &str) -> Option<usize> {
    let field = FieldName::parse(name).expect("built-in atom variable");
    state
        .heap
        .datum_field(datum, &field)
        .ok()
        .and_then(Value::as_number)
        .filter(|value| value.is_finite() && value.fract() == 0.0)
        .and_then(|value| {
            let value = value as i64;
            usize::try_from(value).ok().filter(|slot| *slot > 0)
        })
}

fn shuttle_trace_field_bool(state: &ExecutionState, datum: DatumId, name: &str) -> bool {
    let field = FieldName::parse(name).expect("built-in atom variable");
    state
        .heap
        .datum_field(datum, &field)
        .ok()
        .and_then(Value::as_number)
        .is_some_and(|value| value != 0.0)
}

fn shuttle_trace_field_number(state: &ExecutionState, datum: DatumId, name: &str) -> Option<f32> {
    let field = FieldName::parse(name).expect("built-in atom variable");
    state
        .heap
        .datum_field(datum, &field)
        .ok()
        .and_then(Value::as_number)
}

fn shuttle_trace_list_field(state: &ExecutionState, datum: DatumId, name: &str) -> Option<ListId> {
    let field = FieldName::parse(name).expect("built-in atom variable");
    match state.heap.datum_field(datum, &field) {
        Ok(Value::List(list)) => Some(*list),
        _ => None,
    }
}

fn shuttle_trace_datum_type(state: &ExecutionState, target: DatumId) -> String {
    state
        .heap
        .datum(target)
        .map(|datum| datum.type_path().to_owned().to_string())
        .unwrap_or_else(|_| "<missing>".to_owned())
}

fn shuttle_trace_value_ref(value: Option<&Value>) -> String {
    match value {
        Some(Value::Datum(datum)) => format!("datum({datum:?})"),
        Some(Value::Null) => "null".to_owned(),
        Some(value) => format!("other({value})"),
        None => "none".to_owned(),
    }
}

fn shuttle_trace_prepare_call(
    module: &Module,
    state: &ExecutionState,
    caller: &CallFrame,
    procedure: ProcedureId,
    nullify_slot: Option<usize>,
    target_frame: &mut CallFrame,
) {
    target_frame.set_shuttle_trace_target(caller.shuttle_trace_target());
    target_frame.set_shuttle_trace_post_return(None);

    if !shuttle_trace_enabled() {
        return;
    }
    let Some(path) = module.procedure_path(procedure) else {
        return;
    };

    if shuttle_trace_is_late_shuttle_move(path)
        && let Value::Datum(target) = target_frame.src
        && shuttle_trace_matches_target(state, target)
    {
        target_frame.set_shuttle_trace_target(Some(target));
        shuttle_trace_emit_snapshot(state, target, "lateShuttleMove-entry", None);
        return;
    }

    let Some(shuttle_target) = target_frame.shuttle_trace_target() else {
        return;
    };
    if shuttle_trace_is_nullify_node(path) {
        let slot = nullify_slot;
        shuttle_trace_emit_snapshot(state, shuttle_target, "nullify-node-before", slot);
        target_frame
            .set_shuttle_trace_post_return(Some(ShuttleTracePostReturn::NullifyNode { slot }));
    }
    if shuttle_trace_is_atmos_init(path) {
        shuttle_trace_emit_snapshot(state, shuttle_target, "atmos-init-before", None);
        target_frame.set_shuttle_trace_post_return(Some(ShuttleTracePostReturn::AtmosInit));
    }
}

fn shuttle_trace_emit_snapshot(
    state: &ExecutionState,
    component: DatumId,
    event: &str,
    note: Option<usize>,
) {
    if !shuttle_trace_enabled() {
        return;
    }
    let component_type = shuttle_trace_datum_type(state, component);
    let target_dir = shuttle_trace_field_number(state, component, "dir").unwrap_or_default();
    let target_flipped = shuttle_trace_field_bool(state, component, "flipped");
    let device_type = shuttle_trace_field_u8(state, component, "device_type").unwrap_or(3);
    let nodes = shuttle_trace_list_field(state, component, "nodes");
    let parents = shuttle_trace_list_field(state, component, "parents");
    let node_len = shuttle_trace_list_len(state, nodes);
    let parent_len = shuttle_trace_list_len(state, parents);
    let slot_count = *[node_len, parent_len, device_type]
        .iter()
        .max()
        .unwrap_or(&3)
        .max(&1);
    let note = note.map_or_else(|| "n/a".to_owned(), |slot| slot.to_string());
    let target_location = shuttle_trace_field_text(state, component, "loc");
    let target_x = shuttle_trace_field_text(state, component, "x");
    let target_y = shuttle_trace_field_text(state, component, "y");
    let target_z = shuttle_trace_field_text(state, component, "z");
    eprintln!(
        "shuttle-trace event={event} component={component:?} type={component_type} note={note} \
device_type={device_type} target_dir={target_dir} target_flipped={target_flipped} \
target_loc={target_location} target_xyz={target_x},{target_y},{target_z}"
    );
    for slot in 1..=slot_count {
        let expected_direction =
            shuttle_trace_expected_slot_direction(target_dir, target_flipped, slot).unwrap_or(-1);
        let node_value = shuttle_trace_list_slot(state, nodes, slot);
        let parent_value = shuttle_trace_list_slot(state, parents, slot);
        let node = match node_value {
            Some(Value::Datum(node)) => Some(node),
            _ => None,
        };
        let parent = match parent_value {
            Some(Value::Datum(parent)) => Some(parent),
            _ => None,
        };
        let node_type = node
            .and_then(|datum| state.heap.datum(datum).ok())
            .map(|datum| datum.type_path().to_string())
            .unwrap_or_else(|| "<null>".to_owned());
        let parent_type = parent
            .and_then(|datum| state.heap.datum(datum).ok())
            .map(|datum| datum.type_path().to_string())
            .unwrap_or_else(|| "<null>".to_owned());
        let node_x = node
            .map(|datum| shuttle_trace_field_text(state, datum, "x"))
            .unwrap_or_else(|| "null".to_owned());
        let node_y = node
            .map(|datum| shuttle_trace_field_text(state, datum, "y"))
            .unwrap_or_else(|| "null".to_owned());
        let node_z = node
            .map(|datum| shuttle_trace_field_text(state, datum, "z"))
            .unwrap_or_else(|| "null".to_owned());
        let node_dir = node
            .map(|datum| shuttle_trace_field_text(state, datum, "dir"))
            .unwrap_or_else(|| "null".to_owned());
        let node_pipe = node
            .map(|datum| shuttle_trace_field_text(state, datum, "piping_layer"))
            .unwrap_or_else(|| "null".to_owned());
        let node_loc = node
            .and_then(|datum| {
                let field = FieldName::parse("loc").expect("built-in atom variable");
                state.heap.datum_field(datum, &field).ok().cloned()
            })
            .map(|value| shuttle_trace_value_ref(Some(&value)))
            .unwrap_or_else(|| "null".to_owned());
        eprintln!(
            "shuttle-trace event={event} component={component:?} slot={slot} \
expected_dir={expected_direction} node={node_ref} node_type={node_type} node_x={node_x} node_y={node_y} node_z={node_z} \
node_dir={node_dir} node_pipe={node_pipe} node_loc={node_loc} parent={parent_ref} parent_type={parent_type}",
            component = component,
            event = event,
            slot = slot,
            expected_direction = expected_direction,
            node_ref = shuttle_trace_value_ref(node_value.as_ref()),
            node_type = node_type,
            node_x = node_x,
            node_y = node_y,
            node_z = node_z,
            node_dir = node_dir,
            node_pipe = node_pipe,
            node_loc = node_loc,
            parent_ref = shuttle_trace_value_ref(parent_value.as_ref()),
            parent_type = parent_type
        );
    }
}

pub(crate) fn boot_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("DREAM64_BOOT_TRACE").is_some())
}

fn boot_dashboard_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("DREAM64_BOOT_DASHBOARD").is_some())
}

fn atoms_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("DREAM64_PROFILE_ATOMS").is_some())
}

fn startup_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("DREAM64_PROFILE_STARTUP").is_some())
}

fn startup_instruction_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("DREAM64_PROFILE_STARTUP_OPCODES").is_some())
}

const ATOMS_INITIALIZE_PATH: &str = "/datum/controller/subsystem/atoms/proc/Initialize";

fn is_atoms_initialize_path(path: &str) -> bool {
    path.split_once('@').map_or(path, |(base, _)| base) == ATOMS_INITIALIZE_PATH
}

fn is_subsystem_initialize_path(path: &str) -> bool {
    let path = path.split_once('@').map_or(path, |(base, _)| base);
    path.strip_prefix("/datum/controller/subsystem/")
        .and_then(|suffix| suffix.rsplit_once("/proc/"))
        .is_some_and(|(owner, selector)| !owner.is_empty() && selector == "Initialize")
}

fn atoms_profile_lines_with_event(profile: &AtomsProfile, event: &str) -> Vec<String> {
    let mut procedures = profile
        .samples
        .keys()
        .chain(profile.frame_entries.keys())
        .copied()
        .collect::<Vec<_>>();
    procedures.sort_unstable_by_key(|procedure| (procedure.module_identity, procedure.procedure));
    procedures.dedup();
    procedures.sort_by(|left, right| {
        let left_samples = profile.samples.get(left).copied().unwrap_or(0);
        let right_samples = profile.samples.get(right).copied().unwrap_or(0);
        let left_entries = profile.frame_entries.get(left).copied().unwrap_or(0);
        let right_entries = profile.frame_entries.get(right).copied().unwrap_or(0);
        right_samples
            .cmp(&left_samples)
            .then_with(|| right_entries.cmp(&left_entries))
            .then_with(|| {
                profile
                    .paths
                    .get(left)
                    .map_or("<missing>", String::as_str)
                    .cmp(profile.paths.get(right).map_or("<missing>", String::as_str))
            })
    });
    let sample_count = profile.samples.values().copied().sum::<u64>();
    let mut lines = vec![if let Some(root) = &profile.startup_root {
        format!(
            "boot-vm: startup-profile-{event} subsystem={root} elapsed_ms={} total_instructions={} samples={} procedures={}",
            profile.started.elapsed().as_millis(),
            profile.total_instructions,
            sample_count,
            procedures.len(),
        )
    } else {
        format!(
            "boot-vm: atoms-profile-{event} elapsed_ms={} total_instructions={} samples={} procedures={}",
            profile.started.elapsed().as_millis(),
            profile.total_instructions,
            sample_count,
            procedures.len(),
        )
    }];
    if let Some(counts) = profile.instruction_categories {
        let prefix = if let Some(root) = &profile.startup_root {
            format!("boot-vm: startup-profile-opcodes subsystem={root}")
        } else {
            "boot-vm: atoms-profile-opcodes".to_owned()
        };
        lines.push(format!(
            "{prefix} list_read={} list_write={} field_read={} field_write={} call={} branch={} other={}",
            counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], counts[6],
        ));
        let mut instructions = profile
            .instruction_samples
            .iter()
            .map(|(instruction, samples)| (*instruction, *samples))
            .collect::<Vec<_>>();
        instructions.sort_by(|(left, left_samples), (right, right_samples)| {
            right_samples.cmp(left_samples).then_with(|| {
                profile
                    .instruction_labels
                    .get(left)
                    .map_or("<missing>", String::as_str)
                    .cmp(
                        profile
                            .instruction_labels
                            .get(right)
                            .map_or("<missing>", String::as_str),
                    )
            })
        });
        for (rank, (instruction, samples)) in instructions.into_iter().take(30).enumerate() {
            let label = profile
                .instruction_labels
                .get(&instruction)
                .map_or("<missing>", String::as_str);
            lines.push(format!(
                "{prefix} hot_pc_rank={} samples={samples} {label}",
                rank + 1,
            ));
        }
        let mut wall_instructions = profile
            .instruction_wall_nanos
            .iter()
            .map(|(instruction, nanos)| (*instruction, *nanos))
            .collect::<Vec<_>>();
        wall_instructions.sort_by(|(left, left_nanos), (right, right_nanos)| {
            right_nanos.cmp(left_nanos).then_with(|| {
                profile
                    .instruction_labels
                    .get(left)
                    .map_or("<missing>", String::as_str)
                    .cmp(
                        profile
                            .instruction_labels
                            .get(right)
                            .map_or("<missing>", String::as_str),
                    )
            })
        });
        for (rank, (instruction, nanos)) in wall_instructions.into_iter().take(30).enumerate() {
            let label = profile
                .instruction_labels
                .get(&instruction)
                .map_or("<missing>", String::as_str);
            lines.push(format!(
                "{prefix} wall_pc_rank={} sampled_ms={} {label}",
                rank + 1,
                nanos / 1_000_000,
            ));
        }
    }
    let mut wall_samples = profile
        .wall_sample_nanos
        .iter()
        .map(|(procedure, nanos)| (*procedure, *nanos))
        .collect::<Vec<_>>();
    wall_samples.sort_by(|(left, left_nanos), (right, right_nanos)| {
        right_nanos.cmp(left_nanos).then_with(|| {
            profile
                .paths
                .get(left)
                .map_or("<missing>", String::as_str)
                .cmp(profile.paths.get(right).map_or("<missing>", String::as_str))
        })
    });
    for (rank, (procedure, nanos)) in wall_samples.into_iter().take(30).enumerate() {
        let path = profile
            .paths
            .get(&procedure)
            .map_or("<missing>", String::as_str);
        let prefix = if let Some(root) = &profile.startup_root {
            format!("boot-vm: startup-profile-wall subsystem={root}")
        } else {
            "boot-vm: atoms-profile-wall".to_owned()
        };
        lines.push(format!(
            "{prefix} rank={} sampled_ms={} procedure={path}",
            rank + 1,
            nanos / 1_000_000,
        ));
    }
    for (rank, procedure) in procedures.into_iter().take(30).enumerate() {
        let samples = profile.samples.get(&procedure).copied().unwrap_or(0);
        let entries = profile.frame_entries.get(&procedure).copied().unwrap_or(0);
        let path = profile
            .paths
            .get(&procedure)
            .map_or("<missing>", String::as_str);
        lines.push(if let Some(root) = &profile.startup_root {
            format!(
                "boot-vm: startup-profile-rank subsystem={root} rank={} samples={samples} entries={entries} procedure={path}",
                rank + 1,
            )
        } else {
            format!(
                "boot-vm: atoms-profile-rank rank={} samples={samples} entries={entries} procedure={path}",
                rank + 1,
            )
        });
    }
    lines
}

fn atoms_profile_lines(profile: &AtomsProfile) -> Vec<String> {
    atoms_profile_lines_with_event(profile, "summary")
}

fn atoms_profile_snapshot_lines_if_due(
    profile: &mut AtomsProfile,
    now: Instant,
    interval: Duration,
) -> Option<Vec<String>> {
    if now.duration_since(profile.last_snapshot) < interval {
        return None;
    }
    profile.last_snapshot = now;
    Some(atoms_profile_lines_with_event(profile, "snapshot"))
}

fn emit_atoms_profile(profile: &AtomsProfile) {
    for line in atoms_profile_lines(profile) {
        eprintln!("{line}");
    }
}

fn procedure_argument_trace_filter() -> &'static Option<String> {
    static FILTER: OnceLock<Option<String>> = OnceLock::new();
    FILTER.get_or_init(|| std::env::var("DREAM64_TRACE_PROC_ARGS").ok())
}

fn mark_boot_trace_frame(
    frame: &mut CallFrame,
    module: &Module,
    state: &ExecutionState,
    executed_steps: u64,
) {
    let trace_enabled = boot_trace_enabled();
    let argument_trace = procedure_argument_trace_filter();
    if !trace_enabled && !boot_dashboard_enabled() && argument_trace.is_none() {
        return;
    }
    let path = module
        .paths
        .get(frame.procedure.index())
        .map_or("<missing>", String::as_str);
    if argument_trace
        .as_deref()
        .is_some_and(|needle| path.contains(needle))
    {
        eprintln!(
            "boot-vm: proc-arguments path={path} src={} arguments=[{}]",
            boot_trace_describe_value(&frame.src, state),
            frame
                .arguments
                .iter()
                .map(|value| boot_trace_describe_value(value, state))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    // Monkestation's title subsystem owns the authoritative startup display:
    // Master reports subsystem begin/end here, and Mapping/Assets use the same
    // API for their indented child rows. Mirror those semantic events into the
    // headless trace so the visible console can render the real checklist
    // instead of guessing progress from procedure materialization.
    if path.contains("/datum/controller/subsystem/title/proc/add_init_text@") {
        eprintln!(
            "boot-vm: init-display|event=add|category={}|name={}|stage={}|seconds={}",
            boot_trace_display_value(frame.arguments.first()),
            boot_trace_display_value(frame.arguments.get(1)),
            boot_trace_display_value(frame.arguments.get(2)),
            boot_trace_display_value(frame.arguments.get(3)),
        );
    } else if path.contains("/datum/controller/subsystem/title/proc/remove_init_text@") {
        eprintln!(
            "boot-vm: init-display|event=remove|category={}",
            boot_trace_display_value(frame.arguments.first()),
        );
    }
    // The polished boot dashboard needs only the authoritative title
    // subsystem events above. Keep the much heavier initializer/heartbeat
    // instrumentation behind DREAM64_BOOT_TRACE so visible production boots
    // can show real progress without paying full-trace overhead.
    if !trace_enabled {
        return;
    }
    let traced = path.contains("/datum/controller/global_vars/proc/InitGlobal")
        || [
            "/proc/make_datum_reference_lists",
            "/proc/init_sprite_accessories",
            "/proc/init_species_list",
            "/proc/init_hair_gradients",
            "/proc/init_keybindings",
            "/proc/init_emote_list",
            "/proc/init_crafting_recipes",
            "/proc/init_crafting_recipes_atoms",
            "/proc/init_religion_sects",
        ]
        .iter()
        .any(|suffix| path.ends_with(suffix));
    if !traced {
        return;
    }
    eprintln!("boot-vm: initializer-begin path={path}");
    let cold = frame.cold_mut();
    cold.boot_trace_started = Some(Instant::now());
    cold.boot_trace_heap = Some((
        state.heap.live_datum_count(),
        state.heap.live_list_count(),
        module.materialized_deferred_procedure_count(),
    ));
    cold.boot_trace_step = executed_steps;
}

fn boot_trace_describe_value(value: &Value, state: &ExecutionState) -> String {
    let Value::Datum(datum) = value else {
        return value.to_string();
    };
    let Ok(record) = state.heap.datum(*datum) else {
        return value.to_string();
    };
    let loc = record
        .field(&FieldName::parse("loc").expect("built-in loc field is valid"))
        .map_or_else(|_| "<unset>".to_owned(), ToString::to_string);
    let contents = record
        .field(&FieldName::parse("contents").expect("built-in contents field is valid"))
        .ok()
        .and_then(|value| match value {
            Value::List(list) => state.heap.list(*list).ok().map(|list| list.len()),
            _ => None,
        })
        .map_or_else(|| "<unset>".to_owned(), |length| length.to_string());
    format!(
        "{}{{type={},loc={},contents_len={}}}",
        value,
        record.type_path(),
        loc,
        contents
    )
}

fn boot_trace_display_value(value: Option<&Value>) -> String {
    let raw = match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::Text(text) | Value::File(text)) => text.to_string(),
        Some(Value::TypePath(path)) => path.to_string(),
        Some(Value::Number(number)) => number.to_f32().to_string(),
        Some(value) => value.to_string(),
    };
    raw.replace(['\r', '\n', '|'], " ")
}

#[cfg(test)]
mod tests;
